from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".mplconfig"))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import freeze, unfreeze
from pyscf import ao2mo, dft, fci, gto, scf

from td_graddft.jax_libxc import b3lyp_component_basis
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
from td_graddft_tools import build_ground_state_target_bundle


@dataclass(frozen=True)
class H2CurvePoint:
    r_angstrom: float
    molecule: Any
    fci_ground_total_h: float
    fci_excitation_energies_h: np.ndarray
    fci_oscillator_strengths: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a two-stage H2 dissociation-curve experiment with 10 ground-state "
            "training points and 5 excited-state fine-tune points."
        )
    )
    parser.add_argument("--basis", default="6-31g")
    parser.add_argument("--xc-ref", default="b3lyp")
    parser.add_argument("--r-min", type=float, default=0.05)
    parser.add_argument("--r-max", type=float, default=5.0)
    parser.add_argument("--ground-train-points", type=int, default=10)
    parser.add_argument("--fine-tune-points", type=int, default=5)
    parser.add_argument("--eval-points", type=int, default=100)
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--use-tda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--base-steps", type=int, default=500)
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--base-hidden-dims", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--lr-steps", type=int, default=300)
    parser.add_argument("--lr-learning-rate", type=float, default=5e-2)
    parser.add_argument("--lr-hidden-dims", type=int, nargs="+", default=[64, 64, 32])
    parser.add_argument("--lr-alpha-scale", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument(
        "--outdir",
        default="outputs/h2_dissociation_stage2_10x5_s123_mae",
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


def _run_rhf_with_fallback(mol: gto.Mole) -> scf.hf.RHF:
    attempts = (
        dict(dm0=None, init_guess="minao", damping=0.0, level_shift=0.0, max_cycle=100),
        dict(dm0=None, init_guess="atom", damping=0.3, level_shift=0.5, max_cycle=200),
    )
    for kwargs in attempts:
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.max_cycle = int(kwargs["max_cycle"])
        mf.damping = float(kwargs["damping"])
        mf.level_shift = float(kwargs["level_shift"])
        mf.diis_start_cycle = 1
        mf.init_guess = str(kwargs["init_guess"])
        mf.kernel(dm0=kwargs["dm0"])
        if mf.converged:
            return mf
    mf = scf.RHF(mol).newton()
    mf.conv_tol = 1e-12
    mf.max_cycle = 80
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"RHF did not converge for H2 at R={mol.atom}.")
    return mf


def _build_rks_reference(mol: gto.Mole, *, xc: str):
    attempts = (
        dict(init_guess="minao", damping=0.0, level_shift=0.0, max_cycle=120, newton=False),
        dict(init_guess="atom", damping=0.3, level_shift=0.5, max_cycle=200, newton=False),
        dict(init_guess="atom", damping=0.0, level_shift=0.0, max_cycle=80, newton=True),
    )
    for kwargs in attempts:
        mf = dft.RKS(mol)
        mf.xc = str(xc)
        mf.grids.level = 0
        mf.conv_tol = 1e-10
        mf.max_cycle = int(kwargs["max_cycle"])
        mf.damping = float(kwargs["damping"])
        mf.level_shift = float(kwargs["level_shift"])
        mf.diis_start_cycle = 1
        mf.init_guess = str(kwargs["init_guess"])
        if kwargs["newton"]:
            mf = mf.newton()
            mf.xc = str(xc)
            mf.conv_tol = 1e-10
            mf.max_cycle = int(kwargs["max_cycle"])
        mf.kernel()
        if mf.converged:
            return restricted_reference_from_pyscf(
                mf,
                compute_local_hfx_features=True,
                hfx_omega_values=(0.0, 0.4),
                hfx_chunk_size=256,
            )
    raise RuntimeError(f"RKS did not converge for H2 {xc}/{mol.basis}.")


def _compute_fci_targets(
    mol: gto.Mole,
    *,
    nstates: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    mf = _run_rhf_with_fallback(mol)
    h1_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri_mo = ao2mo.kernel(mol, mf.mo_coeff)
    norb = h1_mo.shape[0]
    nelec = mol.nelectron

    solver = fci.direct_spin0.FCI(mol)
    nroots = max(2, int(nstates) + 1)
    e_roots, ci_roots = solver.kernel(h1_mo, eri_mo, norb, nelec, nroots=nroots)
    e_roots = np.asarray(e_roots, dtype=float).reshape(-1)
    ci_roots = list(ci_roots) if isinstance(ci_roots, (list, tuple)) else [ci_roots]
    if e_roots.size < 2:
        raise RuntimeError("FCI did not return excited states for H2.")
    if int(e_roots.size - 1) < int(nstates):
        raise RuntimeError(
            f"Requested {int(nstates)} excited states for H2/{mol.basis}, "
            f"but only {int(e_roots.size - 1)} roots were available."
        )

    ground_total = float(e_roots[0] + mol.energy_nuc())
    dipole_ao = -mol.intor_symmetric("int1e_r", comp=3)
    dipole_mo = np.einsum("xuv,up,vq->xpq", dipole_ao, mf.mo_coeff, mf.mo_coeff, optimize=True)

    n_compare = int(nstates)
    excitation_energies = np.zeros((n_compare,), dtype=float)
    oscillator_strengths = np.zeros((n_compare,), dtype=float)
    for idx in range(n_compare):
        root = idx + 1
        excitation_energies[idx] = float(e_roots[root] - e_roots[0])
        tdm1 = np.asarray(
            fci.direct_spin0.trans_rdm1(ci_roots[0], ci_roots[root], norb, nelec),
            dtype=float,
        )
        mu = np.einsum("xpq,qp->x", dipole_mo, tdm1, optimize=True)
        oscillator_strengths[idx] = float(
            (2.0 / 3.0) * excitation_energies[idx] * np.dot(mu, mu)
        )
    return ground_total, excitation_energies, oscillator_strengths


def _build_curve_points(
    r_values: np.ndarray,
    *,
    basis: str,
    xc_ref: str,
    nstates: int,
    label: str,
) -> list[H2CurvePoint]:
    points: list[H2CurvePoint] = []
    for idx, r_angstrom in enumerate(r_values, start=1):
        mol = _build_h2_mol(float(r_angstrom), basis)
        reference = _build_rks_reference(mol, xc=xc_ref)
        ground_total, excitation_energies, oscillator_strengths = _compute_fci_targets(
            mol,
            nstates=nstates,
        )
        points.append(
            H2CurvePoint(
                r_angstrom=float(r_angstrom),
                molecule=reference,
                fci_ground_total_h=ground_total,
                fci_excitation_energies_h=excitation_energies,
                fci_oscillator_strengths=oscillator_strengths,
            )
        )
        s1_ev = (
            float(excitation_energies[0] * HARTREE_TO_EV)
            if excitation_energies.size > 0
            else float("nan")
        )
        print(
            f"[{label}] {idx:3d}/{len(r_values):3d} "
            f"R={float(r_angstrom):.4f} A "
            f"E0={ground_total:.10f} Eh "
            f"S1={s1_ev:.6f} eV",
            flush=True,
        )
    return points


def _train_base_ground_state(
    functional: Any,
    dataset: tuple[GroundStateDatum, ...],
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
        dataset[0].molecule,
        optax.adam(float(learning_rate)),
    )
    train_step = make_ground_state_train_step(functional, training_config=training_config)

    best_loss = float("inf")
    best_params = state.params
    best_step = 0
    loss_history: list[float] = []

    for step in range(1, int(steps) + 1):
        state, metrics = train_step(state, dataset)
        loss_value = float(metrics["loss"])
        loss_history.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            best_params = state.params
            best_step = step
        if step == 1 or step == int(steps) or step % max(1, int(log_interval)) == 0:
            print(
                f"[stage1] step={step:4d}/{int(steps)} loss={loss_value:.8e}",
                flush=True,
            )

    return best_params, {
        "best_loss": best_loss,
        "best_step": best_step,
        "loss_history": loss_history,
    }


def _initialize_near_zero_lr_params(
    functional,
    molecule: Any,
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


def _select_fine_tune_points(
    train_points: list[H2CurvePoint],
    *,
    fine_tune_points: int,
) -> list[H2CurvePoint]:
    if int(fine_tune_points) <= 0:
        raise ValueError("fine_tune_points must be positive.")
    if int(fine_tune_points) > len(train_points):
        raise ValueError("fine_tune_points cannot exceed the number of ground-state train points.")
    indices = np.linspace(0, len(train_points) - 1, int(fine_tune_points), dtype=int)
    return [train_points[int(index)] for index in indices]


def _save_fine_tune_bundles(
    points: list[H2CurvePoint],
    *,
    basis: str,
    outdir: Path,
) -> list[str]:
    bundle_dir = outdir / "fine_tune_bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_paths: list[str] = []
    for point in points:
        bundle = build_ground_state_target_bundle(
            point.molecule,
            system_label=f"h2_r_{point.r_angstrom:.4f}A",
            basis_name=basis,
            target_total_energy=point.fci_ground_total_h,
            target_s1_energy=(
                point.fci_excitation_energies_h[0]
                if point.fci_excitation_energies_h.size > 0
                else None
            ),
            target_excitation_energies=point.fci_excitation_energies_h,
            target_oscillator_strengths=point.fci_oscillator_strengths,
        )
        bundle_paths.append(
            str(bundle.save(bundle_dir / f"h2_r_{point.r_angstrom:.4f}A_targets"))
        )
    return bundle_paths


def _fine_tune_dataset(points: list[H2CurvePoint]) -> tuple[GroundStateDatum, ...]:
    return tuple(
        GroundStateDatum(
            molecule=point.molecule,
            target_total_energy=jnp.asarray(point.fci_ground_total_h),
            target_s1_energy=(
                jnp.asarray(point.fci_excitation_energies_h[0])
                if point.fci_excitation_energies_h.size > 0
                else None
            ),
            target_excitation_energies=jnp.asarray(point.fci_excitation_energies_h),
            target_oscillator_strengths=jnp.asarray(point.fci_oscillator_strengths),
        )
        for point in points
    )


def _evaluate_curve(
    points: list[H2CurvePoint],
    *,
    params: Any,
    functional: Any,
    nstates: int,
    use_tda: bool,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    rows: list[dict[str, float]] = []
    ground_errors_ev: list[float] = []
    state_errors_ev: list[list[float]] = [[] for _ in range(int(nstates))]

    for point in points:
        predicted_ground_h = float(
            predict_ground_state_total_energy(
                params,
                functional,
                point.molecule,
                training_config=GroundStateTrainingConfig(mode="fixed_density"),
            )
        )
        predicted_excitations_h = np.asarray(
            predict_excitation_energies(
                params,
                functional,
                point.molecule,
                nstates=int(nstates),
                use_tda=bool(use_tda),
            ),
            dtype=float,
        )

        row: dict[str, float] = {
            "r_angstrom": float(point.r_angstrom),
            "fci_ground_total_h": float(point.fci_ground_total_h),
            "predicted_ground_total_h": predicted_ground_h,
        }
        ground_errors_ev.append(abs(predicted_ground_h - point.fci_ground_total_h) * HARTREE_TO_EV)

        for state_idx in range(int(nstates)):
            fci_energy_h = (
                float(point.fci_excitation_energies_h[state_idx])
                if state_idx < int(point.fci_excitation_energies_h.size)
                else float("nan")
            )
            predicted_energy_h = (
                float(predicted_excitations_h[state_idx])
                if state_idx < int(predicted_excitations_h.size)
                else float("nan")
            )
            row[f"fci_s{state_idx + 1}_ev"] = fci_energy_h * HARTREE_TO_EV
            row[f"predicted_s{state_idx + 1}_ev"] = predicted_energy_h * HARTREE_TO_EV
            if np.isfinite(fci_energy_h) and np.isfinite(predicted_energy_h):
                state_errors_ev[state_idx].append(
                    abs(predicted_energy_h - fci_energy_h) * HARTREE_TO_EV
                )
        rows.append(row)

    metrics = {
        "ground_mae_ev": float(np.mean(ground_errors_ev)) if ground_errors_ev else float("nan"),
    }
    all_excitation_errors = [error for errors in state_errors_ev for error in errors]
    metrics["excitation_mae_ev"] = (
        float(np.mean(all_excitation_errors)) if all_excitation_errors else float("nan")
    )
    for state_idx, errors in enumerate(state_errors_ev, start=1):
        metrics[f"s{state_idx}_mae_ev"] = float(np.mean(errors)) if errors else float("nan")
    return rows, metrics


def _write_curve_csv(
    path: Path,
    *,
    base_rows: list[dict[str, float]],
    corrected_rows: list[dict[str, float]],
    nstates: int,
) -> None:
    fieldnames = [
        "r_angstrom",
        "fci_ground_total_h",
        "base_ground_total_h",
        "corrected_ground_total_h",
    ]
    for state_idx in range(1, int(nstates) + 1):
        fieldnames.extend(
            [
                f"fci_s{state_idx}_ev",
                f"base_s{state_idx}_ev",
                f"corrected_s{state_idx}_ev",
            ]
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for base_row, corrected_row in zip(base_rows, corrected_rows, strict=True):
            merged = {
                "r_angstrom": base_row["r_angstrom"],
                "fci_ground_total_h": base_row["fci_ground_total_h"],
                "base_ground_total_h": base_row["predicted_ground_total_h"],
                "corrected_ground_total_h": corrected_row["predicted_ground_total_h"],
            }
            for state_idx in range(1, int(nstates) + 1):
                merged[f"fci_s{state_idx}_ev"] = base_row[f"fci_s{state_idx}_ev"]
                merged[f"base_s{state_idx}_ev"] = base_row[f"predicted_s{state_idx}_ev"]
                merged[f"corrected_s{state_idx}_ev"] = corrected_row[f"predicted_s{state_idx}_ev"]
            writer.writerow(merged)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ground_r_values = np.linspace(
        float(args.r_min),
        float(args.r_max),
        int(args.ground_train_points),
    )
    eval_r_values = np.linspace(
        float(args.r_min),
        float(args.r_max),
        int(args.eval_points),
    )

    start_time = time.perf_counter()
    print("[setup] building ground-state training references", flush=True)
    ground_train_points = _build_curve_points(
        ground_r_values,
        basis=str(args.basis),
        xc_ref=str(args.xc_ref),
        nstates=int(args.states),
        label="ground-train",
    )
    fine_tune_points = _select_fine_tune_points(
        ground_train_points,
        fine_tune_points=int(args.fine_tune_points),
    )
    bundle_paths = _save_fine_tune_bundles(
        fine_tune_points,
        basis=str(args.basis),
        outdir=outdir,
    )

    train_dataset = tuple(
        GroundStateDatum(
            molecule=point.molecule,
            target_total_energy=jnp.asarray(point.fci_ground_total_h),
        )
        for point in ground_train_points
    )
    base_functional = neural_xc.Functional(
        semilocal_xc=tuple(str(name) for name in b3lyp_component_basis()),
        hidden_dims=tuple(int(value) for value in args.base_hidden_dims),
        architecture="residual",
        input_feature_mode="dm21_original",
        hf_input_mode="spin_resolved",
        response_hf_mode="nonlocal_exchange_only",
        strict_dm21_feature_alignment=True,
        name="h2_curve_stage1_neural_xc",
    )
    base_params, base_train = _train_base_ground_state(
        base_functional,
        train_dataset,
        steps=int(args.base_steps),
        learning_rate=float(args.base_lr),
        seed=int(args.seed),
        log_interval=int(args.log_interval),
    )

    stage1_checkpoint, stage1_meta = save_params_checkpoint(
        outdir / "stage1_base_params.msgpack",
        base_params,
        metadata={
            "basis": str(args.basis),
            "xc_ref": str(args.xc_ref),
            "ground_train_points": [float(value) for value in ground_r_values],
            "states": int(args.states),
            "energy_loss": "mae",
            "base_hidden_dims": [int(value) for value in args.base_hidden_dims],
        },
    )

    lr_functional = neural_xc.LongRangeCorrection(
        base_functional=base_functional,
        hidden_dims=tuple(int(value) for value in args.lr_hidden_dims),
        alpha_scale=float(args.lr_alpha_scale),
        name="h2_curve_stage2_long_range_xc",
    )
    lr_params = _initialize_near_zero_lr_params(
        lr_functional,
        fine_tune_points[0].molecule,
        seed=int(args.seed) + 1,
    )
    combined_initial_params = lr_functional.combine_params(base_params, lr_params)
    fine_tune_data = _fine_tune_dataset(fine_tune_points)
    fine_tune_config = ExcitedStateFineTuneConfig(
        steps=int(args.lr_steps),
        learning_rate=float(args.lr_learning_rate),
        excited_states=tuple(range(1, int(args.states) + 1)),
        use_tda=bool(args.use_tda),
        weight_energy=1.0,
        energy_loss="mae",
        weight_oscillator_strength=0.0,
        weight_ground_state_energy=0.0,
        freeze_ground_state_params=True,
        trainable_path_prefixes=("lr_correction",),
        log_interval=int(args.log_interval),
    )
    fine_tune_start = time.perf_counter()
    fine_tune_result = ExcitedStateFineTuner(
        fine_tune_config,
        lr_functional,
        combined_initial_params,
    ).fine_tune(fine_tune_data)
    fine_tune_wall_time_s = float(time.perf_counter() - fine_tune_start)

    stage2_checkpoint, stage2_meta = save_params_checkpoint(
        outdir / "stage2_long_range_params.msgpack",
        fine_tune_result.params,
        metadata={
            "basis": str(args.basis),
            "xc_ref": str(args.xc_ref),
            "ground_train_points": [float(value) for value in ground_r_values],
            "fine_tune_points": [float(point.r_angstrom) for point in fine_tune_points],
            "states": int(args.states),
            "energy_loss": "mae",
            "lr_hidden_dims": [int(value) for value in args.lr_hidden_dims],
        },
    )

    print("[eval] building dense dissociation-curve references", flush=True)
    eval_points = _build_curve_points(
        eval_r_values,
        basis=str(args.basis),
        xc_ref=str(args.xc_ref),
        nstates=int(args.states),
        label="eval",
    )
    base_fine_rows, base_fine_metrics = _evaluate_curve(
        fine_tune_points,
        params=base_params,
        functional=base_functional,
        nstates=int(args.states),
        use_tda=bool(args.use_tda),
    )
    corrected_fine_rows, corrected_fine_metrics = _evaluate_curve(
        fine_tune_points,
        params=fine_tune_result.params,
        functional=lr_functional,
        nstates=int(args.states),
        use_tda=bool(args.use_tda),
    )
    base_eval_rows, base_eval_metrics = _evaluate_curve(
        eval_points,
        params=base_params,
        functional=base_functional,
        nstates=int(args.states),
        use_tda=bool(args.use_tda),
    )
    corrected_eval_rows, corrected_eval_metrics = _evaluate_curve(
        eval_points,
        params=fine_tune_result.params,
        functional=lr_functional,
        nstates=int(args.states),
        use_tda=bool(args.use_tda),
    )
    _write_curve_csv(
        outdir / "dense_curve_predictions.csv",
        base_rows=base_eval_rows,
        corrected_rows=corrected_eval_rows,
        nstates=int(args.states),
    )

    summary = {
        "basis": str(args.basis),
        "xc_ref": str(args.xc_ref),
        "r_min": float(args.r_min),
        "r_max": float(args.r_max),
        "ground_train_points": [float(value) for value in ground_r_values],
        "fine_tune_points": [float(point.r_angstrom) for point in fine_tune_points],
        "eval_points": int(args.eval_points),
        "states": int(args.states),
        "use_tda": bool(args.use_tda),
        "energy_loss": "mae",
        "base_steps": int(args.base_steps),
        "base_learning_rate": float(args.base_lr),
        "lr_steps": int(args.lr_steps),
        "lr_learning_rate": float(args.lr_learning_rate),
        "base_train": base_train,
        "fine_tune": {
            "initial_loss": float(fine_tune_result.initial_loss),
            "final_loss": float(fine_tune_result.final_loss),
            "best_loss": float(fine_tune_result.best_loss),
            "best_step": int(fine_tune_result.best_step),
            "wall_time_s": fine_tune_wall_time_s,
        },
        "metrics": {
            "fine_tune_points_base": base_fine_metrics,
            "fine_tune_points_corrected": corrected_fine_metrics,
            "dense_curve_base": base_eval_metrics,
            "dense_curve_corrected": corrected_eval_metrics,
        },
        "artifacts": {
            "stage1_checkpoint": str(stage1_checkpoint),
            "stage1_checkpoint_meta": str(stage1_meta) if stage1_meta is not None else None,
            "stage2_checkpoint": str(stage2_checkpoint),
            "stage2_checkpoint_meta": str(stage2_meta) if stage2_meta is not None else None,
            "fine_tune_bundles": bundle_paths,
            "dense_curve_predictions_csv": str(outdir / "dense_curve_predictions.csv"),
        },
        "wall_time_s": float(time.perf_counter() - start_time),
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(
        "[done] "
        f"dense S1/S2/S3 MAE base="
        f"{base_eval_metrics['s1_mae_ev']:.4f}/"
        f"{base_eval_metrics['s2_mae_ev']:.4f}/"
        f"{base_eval_metrics['s3_mae_ev']:.4f} eV, "
        f"corrected="
        f"{corrected_eval_metrics['s1_mae_ev']:.4f}/"
        f"{corrected_eval_metrics['s2_mae_ev']:.4f}/"
        f"{corrected_eval_metrics['s3_mae_ev']:.4f} eV",
        flush=True,
    )


if __name__ == "__main__":
    main()
