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
from pyscf import dft, gto

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
    save_params_checkpoint,
)


BENZENE_GEOM = """
C   0.000000   1.396792   0.000000
C   1.209657   0.698396   0.000000
C   1.209657  -0.698396   0.000000
C   0.000000  -1.396792   0.000000
C  -1.209657  -0.698396   0.000000
C  -1.209657   0.698396   0.000000
H   0.000000   2.484212   0.000000
H   2.151390   1.242106   0.000000
H   2.151390  -1.242106   0.000000
H   0.000000  -2.484212   0.000000
H  -2.151390  -1.242106   0.000000
H  -2.151390   1.242106   0.000000
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage single-point benzene overfit: stage-1 trains the base functional on "
            "ground-state total energy, stage-2 trains only a pair-kernel nonlocal f_xc correction for "
            "S1/S2/S3."
        )
    )
    parser.add_argument("--basis", type=str, default="sto-3g")
    parser.add_argument("--xc-ref", type=str, default="b3lyp")
    parser.add_argument("--semilocal-xc", type=str, default="b3lyp")
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--use-tda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--base-steps", type=int, default=50)
    parser.add_argument("--base-lr", type=float, default=2e-3)
    parser.add_argument("--base-hidden-dims", type=int, nargs="+", default=[96, 96, 96])
    parser.add_argument("--lr-steps", type=int, default=50)
    parser.add_argument("--lr-learning-rate", type=float, default=5e-4)
    parser.add_argument("--lr-hidden-dims", type=int, nargs="+", default=[64, 64, 32])
    parser.add_argument("--lr-alpha-scale", type=float, default=0.1)
    parser.add_argument("--lr-gamma-init", type=float, default=1.0)
    parser.add_argument("--lr-distance-scale", type=float, default=1.0)
    parser.add_argument("--lr-max-pair-points", type=int, default=128)
    parser.add_argument("--energy-loss", choices=("mse", "mae"), default="mse")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/benzene_equilibrium_nonlocal_fxcoverfit",
    )
    return parser.parse_args()


def _build_benzene_mol(basis: str) -> gto.Mole:
    mol = gto.Mole()
    mol.atom = BENZENE_GEOM
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
        raise RuntimeError(f"RKS did not converge for benzene {xc}/{mol.basis}.")
    reference = restricted_reference_from_pyscf(
        mf,
        compute_local_hfx_features=True,
        hfx_omega_values=(0.0, 0.4),
        hfx_chunk_size=256,
    )
    return mf, reference


def _compute_reference_excitations(
    mf,
    *,
    nstates: int,
    use_tda: bool,
) -> np.ndarray:
    td = mf.TDA() if use_tda else mf.TDDFT()
    td.nstates = int(nstates)
    td.conv_tol = 1e-8
    td.max_cycle = 200
    td.kernel()
    energies = np.asarray(td.e, dtype=float).reshape(-1)
    if energies.size < int(nstates):
        raise RuntimeError(
            f"Reference {'TDA' if use_tda else 'TDDFT'} returned only {energies.size} states."
        )
    return energies[: int(nstates)]


def _mae(reference: np.ndarray, prediction: np.ndarray, *, scale: float = 1.0) -> float:
    n = min(int(reference.size), int(prediction.size))
    if n <= 0:
        return float("nan")
    return float(np.mean(np.abs(prediction[:n] - reference[:n])) * scale)


def _max_abs(reference: np.ndarray, prediction: np.ndarray, *, scale: float = 1.0) -> float:
    n = min(int(reference.size), int(prediction.size))
    if n <= 0:
        return float("nan")
    return float(np.max(np.abs(prediction[:n] - reference[:n])) * scale)


def _write_state_predictions(
    path: Path,
    *,
    reference_energies: np.ndarray,
    stage1_energies: np.ndarray,
    stage2_energies: np.ndarray,
) -> None:
    n = min(
        int(reference_energies.size),
        int(stage1_energies.size),
        int(stage2_energies.size),
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "state",
                "reference_energy_ev",
                "stage1_energy_ev",
                "stage2_energy_ev",
                "stage1_abs_error_ev",
                "stage2_abs_error_ev",
            ]
        )
        for idx in range(n):
            ref_ev = float(reference_energies[idx] * HARTREE_TO_EV)
            stage1_ev = float(stage1_energies[idx] * HARTREE_TO_EV)
            stage2_ev = float(stage2_energies[idx] * HARTREE_TO_EV)
            writer.writerow(
                [
                    idx + 1,
                    ref_ev,
                    stage1_ev,
                    stage2_ev,
                    abs(stage1_ev - ref_ev),
                    abs(stage2_ev - ref_ev),
                ]
            )


def _write_loss_history(path: Path, values: list[float] | tuple[float, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "loss"])
        for idx, value in enumerate(values):
            writer.writerow([idx, float(value)])


def _initialize_small_lr_params(
    functional,
    molecule,
    *,
    seed: int,
    alpha_value: float = 5e-2,
    gamma_value: float = 1.0,
    random_scale: float = 1e-2,
):
    params = functional.init_from_molecule(jax.random.PRNGKey(int(seed)), molecule)
    params = jax.tree_util.tree_map(
        lambda value: jnp.asarray(value) * jnp.asarray(random_scale, dtype=jnp.asarray(value).dtype),
        params,
    )
    params = dict(params)
    params_collection = dict(params["params"])

    alpha_scale = float(getattr(functional.model, "alpha_scale", 1.0))
    alpha_target = max(float(alpha_value) / max(alpha_scale, 1e-8), 1e-6)
    gamma_floor = float(getattr(functional.model, "gamma_floor", 1e-3))
    gamma_target = max(float(gamma_value) - gamma_floor, 1e-6)

    params_collection["AlphaHead"] = dict(params_collection["AlphaHead"])
    params_collection["GammaHead"] = dict(params_collection["GammaHead"])
    params_collection["AlphaHead"]["bias"] = jnp.full_like(
        params_collection["AlphaHead"]["bias"],
        jnp.log(jnp.expm1(jnp.asarray(alpha_target))),
    )
    params_collection["GammaHead"]["bias"] = jnp.full_like(
        params_collection["GammaHead"]["bias"],
        jnp.log(jnp.expm1(jnp.asarray(gamma_target))),
    )
    params["params"] = params_collection
    return jax.tree_util.tree_map(jnp.asarray, params)


def _train_base_ground_state(
    functional,
    datum: GroundStateDatum,
    *,
    steps: int,
    learning_rate: float,
    seed: int,
    log_interval: int,
):
    training_config = GroundStateTrainingConfig(
        mode="fixed_density",
        energy_mse_weight=1.0,
        energy_mae_weight=0.0,
        energy_normalization="none",
    )
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
    target_total_energy = float(jnp.asarray(datum.target_total_energy))

    for step in range(1, int(steps) + 1):
        state, _ = train_step(state, datum)
        predicted_energy = float(
            predict_ground_state_total_energy(
                state.params,
                functional,
                datum.molecule,
                training_config=training_config,
            )
        )
        loss = float((predicted_energy - target_total_energy) ** 2)
        loss_history.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_params = state.params
            best_step = step
        if step == 1 or step == int(steps) or step % max(1, int(log_interval)) == 0:
            print(f"[stage1] step={step:4d}/{steps} loss={loss:.8e}", flush=True)

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
    mol = _build_benzene_mol(str(args.basis))
    mf, reference = _build_rks_reference(mol, xc=str(args.xc_ref))
    reference_energies = _compute_reference_excitations(
        mf,
        nstates=int(args.states),
        use_tda=bool(args.use_tda),
    )
    reference_ground_total = float(mf.e_tot)

    base_functional = neural_xc.Functional(
        semilocal_xc=str(args.semilocal_xc),
        hidden_dims=tuple(int(v) for v in args.base_hidden_dims),
        architecture="residual",
        input_feature_mode="dm21_original",
        hf_input_mode="spin_resolved",
        response_hf_mode="nonlocal_exchange_only",
        strict_dm21_feature_alignment=True,
        name="benzene_stage1_neural_xc",
    )
    base_datum = GroundStateDatum(
        molecule=reference,
        target_total_energy=jnp.asarray(reference_ground_total),
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

    lr_functional = neural_xc.LongRangeCorrection(
        base_functional=base_functional,
        hidden_dims=tuple(int(v) for v in args.lr_hidden_dims),
        alpha_scale=float(args.lr_alpha_scale),
        distance_scale=float(args.lr_distance_scale),
        max_pair_points=int(args.lr_max_pair_points),
        name="benzene_stage2_pair_kernel_xc",
    )
    lr_params = _initialize_small_lr_params(
        lr_functional,
        reference,
        seed=int(args.seed) + 1,
        alpha_value=float(args.lr_alpha_scale),
        gamma_value=float(args.lr_gamma_init),
    )
    combined_initial_params = lr_functional.combine_params(base_params, lr_params)

    fine_tune_datum = GroundStateDatum(
        molecule=reference,
        target_total_energy=jnp.asarray(reference_ground_total),
        target_excitation_energies=jnp.asarray(reference_energies),
    )
    fine_tune_config = ExcitedStateFineTuneConfig(
        steps=int(args.lr_steps),
        learning_rate=float(args.lr_learning_rate),
        excited_states=tuple(range(1, int(args.states) + 1)),
        use_tda=bool(args.use_tda),
        weight_energy=1.0,
        energy_loss=str(args.energy_loss),
        weight_oscillator_strength=0.0,
        weight_spectrum=0.0,
        weight_ground_state_energy=0.0,
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

    stage2_params = result.best_params if fine_tune_config.select_params == "best_loss" else result.params
    corrected_ground_energy = float(
        predict_ground_state_total_energy(
            stage2_params,
            lr_functional,
            reference,
            training_config=GroundStateTrainingConfig(mode="fixed_density"),
        )
    )
    corrected_energies = np.asarray(
        predict_excitation_energies(
            stage2_params,
            lr_functional,
            reference,
            nstates=int(args.states),
            use_tda=bool(args.use_tda),
        ),
        dtype=float,
    )

    stage1_checkpoint, _ = save_params_checkpoint(
        outdir / "stage1_base_params.msgpack",
        base_params,
        metadata={
            "system": "benzene",
            "basis": str(args.basis),
            "xc_ref": str(args.xc_ref),
            "semilocal_xc": str(args.semilocal_xc),
            "base_hidden_dims": [int(v) for v in args.base_hidden_dims],
        },
    )
    stage2_checkpoint, _ = save_params_checkpoint(
        outdir / "stage2_nonlocal_fxc_params.msgpack",
        stage2_params,
        metadata={
            "system": "benzene",
            "basis": str(args.basis),
            "xc_ref": str(args.xc_ref),
            "states": int(args.states),
            "use_tda": bool(args.use_tda),
            "lr_hidden_dims": [int(v) for v in args.lr_hidden_dims],
            "lr_alpha_scale": float(args.lr_alpha_scale),
            "lr_gamma_init": float(args.lr_gamma_init),
            "lr_distance_scale": float(args.lr_distance_scale),
            "lr_max_pair_points": int(args.lr_max_pair_points),
            "energy_loss": str(args.energy_loss),
        },
    )

    state_csv = outdir / "state_predictions.csv"
    _write_state_predictions(
        state_csv,
        reference_energies=reference_energies,
        stage1_energies=base_energies,
        stage2_energies=corrected_energies,
    )
    stage1_loss_csv = outdir / "stage1_loss_history.csv"
    stage2_loss_csv = outdir / "stage2_loss_history.csv"
    _write_loss_history(stage1_loss_csv, base_train["loss_history"])
    _write_loss_history(stage2_loss_csv, result.loss_history)

    summary = {
        "system": "benzene",
        "geometry": "equilibrium_planar",
        "basis": str(args.basis),
        "xc_ref": str(args.xc_ref),
        "semilocal_xc": str(args.semilocal_xc),
        "states": int(args.states),
        "solver": "tda" if bool(args.use_tda) else "casida",
        "reference_grid_points": int(np.asarray(reference.grid.coords).shape[0]),
        "stage2_response_grid_points": int(
            min(int(args.lr_max_pair_points), int(np.asarray(reference.grid.coords).shape[0]))
        ),
        "reference": {
            "ground_total_energy_hartree": reference_ground_total,
            "excitation_energies_hartree": [float(v) for v in reference_energies.tolist()],
            "excitation_energies_ev": [float(v * HARTREE_TO_EV) for v in reference_energies.tolist()],
        },
        "stage1": {
            "steps": int(args.base_steps),
            "learning_rate": float(args.base_lr),
            "hidden_dims": [int(v) for v in args.base_hidden_dims],
            "best_loss": float(base_train["best_loss"]),
            "best_step": int(base_train["best_step"]),
            "ground_total_energy_hartree": base_ground_energy,
            "ground_abs_error_ev": float(abs(base_ground_energy - reference_ground_total) * HARTREE_TO_EV),
            "excitation_mae_ev": _mae(reference_energies, base_energies, scale=HARTREE_TO_EV),
            "excitation_max_abs_ev": _max_abs(
                reference_energies,
                base_energies,
                scale=HARTREE_TO_EV,
            ),
            "excitation_energies_hartree": [float(v) for v in base_energies.tolist()],
            "checkpoint": str(stage1_checkpoint),
            "loss_history_csv": str(stage1_loss_csv),
        },
        "stage2": {
            "steps": int(args.lr_steps),
            "learning_rate": float(args.lr_learning_rate),
            "hidden_dims": [int(v) for v in args.lr_hidden_dims],
            "alpha_scale": float(args.lr_alpha_scale),
            "gamma_init": float(args.lr_gamma_init),
            "distance_scale": float(args.lr_distance_scale),
            "max_pair_points": int(args.lr_max_pair_points),
            "energy_loss": str(args.energy_loss),
            "initial_loss": float(result.initial_loss),
            "final_loss": float(result.final_loss),
            "best_loss": float(result.best_loss),
            "best_step": int(result.best_step),
            "ground_total_energy_hartree": corrected_ground_energy,
            "ground_abs_error_ev": float(
                abs(corrected_ground_energy - reference_ground_total) * HARTREE_TO_EV
            ),
            "excitation_mae_ev": _mae(
                reference_energies,
                corrected_energies,
                scale=HARTREE_TO_EV,
            ),
            "excitation_max_abs_ev": _max_abs(
                reference_energies,
                corrected_energies,
                scale=HARTREE_TO_EV,
            ),
            "excitation_energies_hartree": [float(v) for v in corrected_energies.tolist()],
            "checkpoint": str(stage2_checkpoint),
            "loss_history_csv": str(stage2_loss_csv),
        },
        "artifacts": {
            "state_predictions_csv": str(state_csv),
        },
        "wall_time_s": float(time.perf_counter() - t0),
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
