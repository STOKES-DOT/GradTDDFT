from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import freeze, unfreeze
from pyscf import ao2mo, dft, fci, gto, scf

from td_graddft import neural_xc
from td_graddft.reference_legacy import restricted_reference_from_pyscf
from td_graddft.spectra import HARTREE_TO_EV
from td_graddft.training import (
    ExcitedStateFineTuneConfig,
    ExcitedStateFineTuner,
    GroundStateDatum,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    make_ground_state_train_step,
    predict_excitation_energies,
    predict_ground_state_total_energy,
    predict_oscillator_strengths,
    save_params_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end H2 demo for two-stage TD-GradDFT training: "
            "ground-state neural XC fit followed by long-range excited-state fine-tuning."
        )
    )
    parser.add_argument("--bond-length", type=float, default=0.74)
    parser.add_argument("--basis", type=str, default="6-31g")
    parser.add_argument("--xc-ref", type=str, default="b3lyp")
    parser.add_argument("--semilocal-xc", type=str, default="b3lyp")
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--use-tda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--base-steps", type=int, default=120)
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--base-hidden-dims", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--lr-steps", type=int, default=160)
    parser.add_argument("--lr-learning-rate", type=float, default=5e-2)
    parser.add_argument("--lr-hidden-dims", type=int, nargs="+", default=[64, 64, 32])
    parser.add_argument("--lr-alpha-scale", type=float, default=0.2)
    parser.add_argument("--weight-energy", type=float, default=1.0)
    parser.add_argument("--weight-oscillator-strength", type=float, default=0.25)
    parser.add_argument("--weight-ground-state-energy", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/h2_long_range_excited_finetune",
    )
    return parser.parse_args()


def _build_h2_mol(r_angstrom: float, basis: str) -> gto.Mole:
    mol = gto.Mole()
    mol.atom = f"""
    H 0.000000 0.000000 {-0.5 * r_angstrom:.10f}
    H 0.000000 0.000000 {+0.5 * r_angstrom:.10f}
    """
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.spin = 0
    mol.charge = 0
    mol.verbose = 0
    mol.build()
    return mol


def _build_rks_reference(
    mol: gto.Mole,
    *,
    xc: str,
):
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"RKS did not converge for H2 {xc}/{mol.basis}.")
    return mf, restricted_reference_from_pyscf(
        mf,
        compute_local_hfx_features=True,
        hfx_omega_values=(0.0, 0.4),
        hfx_chunk_size=256,
    )


def _compute_fci_targets(
    mol: gto.Mole,
    *,
    nstates: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge for FCI reference generation.")

    h1_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri_mo = ao2mo.kernel(mol, mf.mo_coeff)
    norb = h1_mo.shape[0]
    nelec = mol.nelectron

    solver = fci.direct_spin0.FCI(mol)
    nroots = max(2, int(nstates) + 1)
    e_roots, ci_roots = solver.kernel(h1_mo, eri_mo, norb, nelec, nroots=nroots)
    e_roots = np.asarray(e_roots, dtype=float).reshape(-1)
    if e_roots.size < 2:
        raise RuntimeError("FCI did not return any excited states for H2.")

    ground_total = float(e_roots[0] + mol.energy_nuc())
    dipole_ao = -mol.intor_symmetric("int1e_r", comp=3)
    dipole_mo = np.einsum("xuv,up,vq->xpq", dipole_ao, mf.mo_coeff, mf.mo_coeff)

    n_compare = min(int(nstates), int(e_roots.size - 1))
    excitation_energies = np.zeros((n_compare,), dtype=float)
    oscillator = np.zeros((n_compare,), dtype=float)
    for idx in range(n_compare):
        root = idx + 1
        excitation_energies[idx] = float(e_roots[root] - e_roots[0])
        tdm1 = fci.direct_spin0.trans_rdm1(ci_roots[0], ci_roots[root], norb, nelec)
        mu = np.einsum("xpq,qp->x", dipole_mo, np.asarray(tdm1, dtype=float))
        oscillator[idx] = float((2.0 / 3.0) * excitation_energies[idx] * np.dot(mu, mu))
    return ground_total, excitation_energies, oscillator


def _mae(reference: np.ndarray, prediction: np.ndarray, *, scale: float = 1.0) -> float:
    n = min(int(reference.size), int(prediction.size))
    if n <= 0:
        return float("nan")
    return float(np.mean(np.abs(prediction[:n] - reference[:n])) * scale)


def _write_state_predictions(
    path: Path,
    *,
    fci_energies: np.ndarray,
    fci_oscillator_strengths: np.ndarray,
    base_energies: np.ndarray,
    base_oscillator_strengths: np.ndarray,
    corrected_energies: np.ndarray,
    corrected_oscillator_strengths: np.ndarray,
) -> None:
    n = min(
        int(fci_energies.size),
        int(fci_oscillator_strengths.size),
        int(base_energies.size),
        int(base_oscillator_strengths.size),
        int(corrected_energies.size),
        int(corrected_oscillator_strengths.size),
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "state",
                "fci_energy_eV",
                "base_energy_eV",
                "corrected_energy_eV",
                "fci_oscillator_strength",
                "base_oscillator_strength",
                "corrected_oscillator_strength",
            ]
        )
        for idx in range(n):
            writer.writerow(
                [
                    idx + 1,
                    float(fci_energies[idx] * HARTREE_TO_EV),
                    float(base_energies[idx] * HARTREE_TO_EV),
                    float(corrected_energies[idx] * HARTREE_TO_EV),
                    float(fci_oscillator_strengths[idx]),
                    float(base_oscillator_strengths[idx]),
                    float(corrected_oscillator_strengths[idx]),
                ]
            )


def _initialize_near_zero_lr_params(
    functional,
    molecule,
    *,
    seed: int,
    alpha_value: float = 5e-3,
    gamma_value: float = 1.0,
    random_scale: float = 1e-2,
):
    params = unfreeze(functional.init(jax.random.PRNGKey(int(seed)), molecule))
    params = jax.tree_util.tree_map(
        lambda value: jnp.asarray(value) * jnp.asarray(random_scale, dtype=jnp.asarray(value).dtype),
        params,
    )

    alpha_scale = float(getattr(functional.model, "alpha_scale", 1.0))
    alpha_target = max(float(alpha_value) / max(alpha_scale, 1e-8), 1e-6)
    gamma_floor = float(getattr(functional.model, "gamma_floor", 1e-3))
    gamma_target = max(float(gamma_value) - gamma_floor, 1e-6)
    params["params"]["AlphaHead"]["bias"] = jnp.full_like(
        params["params"]["AlphaHead"]["bias"],
        jnp.log(jnp.expm1(jnp.asarray(alpha_target))),
    )
    params["params"]["GammaHead"]["bias"] = jnp.full_like(
        params["params"]["GammaHead"]["bias"],
        jnp.log(jnp.expm1(jnp.asarray(gamma_target))),
    )
    return freeze(params)


def _train_base_ground_state(
    functional,
    datum: GroundStateDatum,
    *,
    steps: int,
    learning_rate: float,
    seed: int,
    log_interval: int,
):
    training_config = GroundStateTrainingConfig(mode="fixed_density")
    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(int(seed)),
        datum.molecule,
        optax.adam(float(learning_rate)),
    )
    train_step = make_ground_state_train_step(functional, training_config=training_config)

    loss_history: list[float] = []
    best_loss = float("inf")
    best_params = state.params
    best_step = 0

    for step in range(1, int(steps) + 1):
        state, metrics = train_step(state, datum)
        loss = float(metrics["loss"])
        loss_history.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_params = state.params
            best_step = step
        if step == 1 or step == int(steps) or step % max(1, int(log_interval)) == 0:
            print(
                f"[stage1] step={step:4d}/{steps} loss={loss:.8e}",
                flush=True,
            )

    return best_params, {
        "best_loss": best_loss,
        "best_step": best_step,
        "loss_history": loss_history,
    }


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    mol = _build_h2_mol(float(args.bond_length), basis=str(args.basis))
    _, reference = _build_rks_reference(mol, xc=str(args.xc_ref))
    fci_ground_total, fci_energies, fci_oscillator_strengths = _compute_fci_targets(
        mol,
        nstates=int(args.states),
    )

    base_functional = neural_xc.Functional(
        semilocal_xc=str(args.semilocal_xc),
        hidden_dims=tuple(int(v) for v in args.base_hidden_dims),
        architecture="residual",
        input_feature_mode="dm21_original",
        hf_input_mode="spin_resolved",
        response_hf_mode="nonlocal_exchange_only",
        strict_dm21_feature_alignment=True,
        name="h2_stage1_neural_xc",
    )
    base_datum = GroundStateDatum(
        molecule=reference,
        target_total_energy=jnp.asarray(fci_ground_total),
    )
    base_params, base_train = _train_base_ground_state(
        base_functional,
        base_datum,
        steps=int(args.base_steps),
        learning_rate=float(args.base_lr),
        seed=int(args.seed),
        log_interval=int(args.log_interval),
    )

    base_ground_energy = float(
        predict_ground_state_total_energy(
            base_params,
            base_functional,
            reference,
            training_config=GroundStateTrainingConfig(mode="fixed_density"),
        )
    )
    base_energies = np.asarray(
        predict_excitation_energies(
            base_params,
            base_functional,
            reference,
            nstates=int(args.states),
            use_tda=bool(args.use_tda),
        ),
        dtype=float,
    )
    base_oscillator_strengths = np.asarray(
        predict_oscillator_strengths(
            base_params,
            base_functional,
            reference,
            nstates=int(args.states),
            use_tda=bool(args.use_tda),
        ),
        dtype=float,
    )

    lr_functional = neural_xc.LongRangeCorrection(
        base_functional=base_functional,
        hidden_dims=tuple(int(v) for v in args.lr_hidden_dims),
        alpha_scale=float(args.lr_alpha_scale),
        name="h2_stage2_long_range_xc",
    )
    lr_params = _initialize_near_zero_lr_params(
        lr_functional,
        reference,
        seed=int(args.seed) + 1,
        gamma_value=1.0,
    )
    combined_initial_params = lr_functional.combine_params(base_params, lr_params)

    fine_tune_datum = GroundStateDatum(
        molecule=reference,
        target_total_energy=jnp.asarray(fci_ground_total),
        target_excitation_energies=jnp.asarray(fci_energies),
        target_oscillator_strengths=jnp.asarray(fci_oscillator_strengths),
    )
    fine_tune_config = ExcitedStateFineTuneConfig(
        steps=int(args.lr_steps),
        learning_rate=float(args.lr_learning_rate),
        excited_states=tuple(range(1, int(args.states) + 1)),
        use_tda=bool(args.use_tda),
        weight_energy=float(args.weight_energy),
        weight_oscillator_strength=float(args.weight_oscillator_strength),
        weight_ground_state_energy=float(args.weight_ground_state_energy),
        freeze_ground_state_params=True,
        trainable_path_prefixes=("lr_correction",),
        log_interval=int(args.log_interval),
    )
    fine_tuner = ExcitedStateFineTuner(
        fine_tune_config,
        lr_functional,
        combined_initial_params,
    )
    result = fine_tuner.fine_tune(fine_tune_datum)

    corrected_ground_energy = float(
        predict_ground_state_total_energy(
            result.params,
            lr_functional,
            reference,
            training_config=GroundStateTrainingConfig(mode="fixed_density"),
        )
    )
    corrected_energies = np.asarray(
        predict_excitation_energies(
            result.params,
            lr_functional,
            reference,
            nstates=int(args.states),
            use_tda=bool(args.use_tda),
        ),
        dtype=float,
    )
    corrected_oscillator_strengths = np.asarray(
        predict_oscillator_strengths(
            result.params,
            lr_functional,
            reference,
            nstates=int(args.states),
            use_tda=bool(args.use_tda),
        ),
        dtype=float,
    )

    base_checkpoint, _ = save_params_checkpoint(
        outdir / "stage1_base_params.msgpack",
        base_params,
        metadata={
            "basis": str(args.basis),
            "bond_length_angstrom": float(args.bond_length),
            "xc_ref": str(args.xc_ref),
            "semilocal_xc": str(args.semilocal_xc),
            "base_hidden_dims": [int(v) for v in args.base_hidden_dims],
        },
    )
    corrected_checkpoint, _ = save_params_checkpoint(
        outdir / "stage2_long_range_params.msgpack",
        result.params,
        metadata={
            "basis": str(args.basis),
            "bond_length_angstrom": float(args.bond_length),
            "xc_ref": str(args.xc_ref),
            "states": int(args.states),
            "use_tda": bool(args.use_tda),
            "lr_hidden_dims": [int(v) for v in args.lr_hidden_dims],
            "lr_alpha_scale": float(args.lr_alpha_scale),
        },
    )

    _write_state_predictions(
        outdir / "state_predictions.csv",
        fci_energies=fci_energies,
        fci_oscillator_strengths=fci_oscillator_strengths,
        base_energies=base_energies,
        base_oscillator_strengths=base_oscillator_strengths,
        corrected_energies=corrected_energies,
        corrected_oscillator_strengths=corrected_oscillator_strengths,
    )

    summary = {
        "system": "H2",
        "bond_length_angstrom": float(args.bond_length),
        "basis": str(args.basis),
        "xc_ref": str(args.xc_ref),
        "semilocal_xc": str(args.semilocal_xc),
        "states": int(args.states),
        "solver": "tda" if bool(args.use_tda) else "casida",
        "stage1": {
            "steps": int(args.base_steps),
            "learning_rate": float(args.base_lr),
            "hidden_dims": [int(v) for v in args.base_hidden_dims],
            "best_loss": float(base_train["best_loss"]),
            "best_step": int(base_train["best_step"]),
            "ground_total_energy_hartree": float(base_ground_energy),
            "ground_abs_error_ev": float(abs(base_ground_energy - fci_ground_total) * HARTREE_TO_EV),
            "excitation_mae_ev": _mae(fci_energies, base_energies, scale=HARTREE_TO_EV),
            "oscillator_strength_mae": _mae(fci_oscillator_strengths, base_oscillator_strengths),
            "checkpoint": str(base_checkpoint),
        },
        "stage2": {
            "steps": int(args.lr_steps),
            "learning_rate": float(args.lr_learning_rate),
            "hidden_dims": [int(v) for v in args.lr_hidden_dims],
            "alpha_scale": float(args.lr_alpha_scale),
            "weight_energy": float(args.weight_energy),
            "weight_oscillator_strength": float(args.weight_oscillator_strength),
            "weight_ground_state_energy": float(args.weight_ground_state_energy),
            "initial_loss": float(result.initial_loss),
            "final_loss": float(result.final_loss),
            "best_loss": float(result.best_loss),
            "best_step": int(result.best_step),
            "ground_total_energy_hartree": float(corrected_ground_energy),
            "ground_abs_error_ev": float(
                abs(corrected_ground_energy - fci_ground_total) * HARTREE_TO_EV
            ),
            "excitation_mae_ev": _mae(fci_energies, corrected_energies, scale=HARTREE_TO_EV),
            "oscillator_strength_mae": _mae(
                fci_oscillator_strengths,
                corrected_oscillator_strengths,
            ),
            "checkpoint": str(corrected_checkpoint),
        },
        "reference": {
            "ground_total_energy_hartree": float(fci_ground_total),
            "excitation_energies_hartree": [float(v) for v in fci_energies.tolist()],
            "oscillator_strengths": [float(v) for v in fci_oscillator_strengths.tolist()],
        },
        "wall_time_s": float(time.perf_counter() - t0),
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
