from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from pyscf import dft, gto

from td_graddft import neural_xc
from td_graddft.jax_libxc import b3lyp_component_basis
from td_graddft.neural_xc import (
    GRADDFT_DEFAULT_DM21_HIDDEN_DIMS,
    GRADDFT_DEFAULT_INPUT_FEATURE_MODE,
    GRADDFT_DEFAULT_NETWORK_ARCHITECTURE,
)
from td_graddft.reference_legacy import restricted_reference_from_pyscf
from td_graddft.spectra import HARTREE_TO_EV
from td_graddft.training import (
    ExcitedStateDatum,
    GroundStateCoreDatum,
    GroundStateDatum,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    ground_state_mse_loss,
    make_ground_state_train_step,
    predict_excitation_energies,
    predict_ground_state_molecule,
    predict_ground_state_total_energy,
    save_params_checkpoint,
)


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"[{_timestamp()}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


@dataclass(frozen=True)
class ReferenceRow:
    system: str
    split: str
    atom: str
    unit: str
    charge: int
    spin: int
    basis: str
    ccsd_total_energy_h: float
    s1_excitation_h: float


@dataclass(frozen=True)
class PreparedReference:
    row: ReferenceRow
    molecule: Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train the neural functional on a reusable closed-shell EOM-EE-CCSD S1 CSV, "
            "using self-consistent or fixed-density mode, then report train/validation "
            "and benzene generalization errors."
        )
    )
    p.add_argument("--reference-csv", required=True)
    p.add_argument("--basis", default="6-31g")
    p.add_argument("--xc", default="b3lyp")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--lr-decay-every", type=int, default=500)
    p.add_argument("--lr-decay-factor", type=float, default=0.5)
    p.add_argument("--training-mode", choices=("fixed_density", "self_consistent"), default="self_consistent")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=list(GRADDFT_DEFAULT_DM21_HIDDEN_DIMS))
    p.add_argument(
        "--network-architecture",
        choices=("simple_mlp", "graddft_residual"),
        default=GRADDFT_DEFAULT_NETWORK_ARCHITECTURE,
    )
    p.add_argument(
        "--input-feature-mode",
        choices=("enhanced", "dm21_original"),
        default=GRADDFT_DEFAULT_INPUT_FEATURE_MODE,
    )
    p.add_argument("--include-pt2-channel", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--pt2-channel-mode",
        choices=("scaled_projected", "local_exact"),
        default="scaled_projected",
    )
    p.add_argument("--semilocal-xc", nargs="+", default=list(b3lyp_component_basis()))
    p.add_argument("--s1-weight", type=float, default=1.0)
    p.add_argument("--s1-use-tda", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-use-tda", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--energy-mse-weight", type=float, default=0.0)
    p.add_argument("--energy-mae-weight", type=float, default=0.0)
    p.add_argument("--density-constraint-weight", type=float, default=0.0)
    p.add_argument("--grids-level", type=int, default=0)
    p.add_argument("--reference-scf-max-cycle", type=int, default=100)
    p.add_argument("--reference-scf-conv-tol", type=float, default=1e-10)
    p.add_argument("--train-scf-max-cycle", type=int, default=16)
    p.add_argument("--train-scf-damping", type=float, default=0.25)
    p.add_argument("--train-scf-conv-tol-density", type=float, default=1e-8)
    p.add_argument("--train-scf-vxc-clip", type=float, default=20.0)
    p.add_argument(
        "--scf-iterate-selection",
        choices=("final", "best_rms", "first_converged"),
        default="best_rms",
    )
    p.add_argument(
        "--scf-gradient-mode",
        choices=("unrolled", "implicit_commutator"),
        default="implicit_commutator",
    )
    p.add_argument(
        "--scf-implicit-diff-solver",
        choices=("normal_cg", "gmres", "bicgstab"),
        default="normal_cg",
    )
    p.add_argument("--scf-implicit-diff-tolerance", type=float, default=1e-6)
    p.add_argument("--scf-implicit-diff-regularization", type=float, default=1e-3)
    p.add_argument("--scf-implicit-diff-restart", type=int, default=12)
    p.add_argument("--scf-require-convergence", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--grad-clip-norm", type=float, default=None)
    p.add_argument("--scf-stop-gradient-on-unconverged", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--scf-stop-gradient-rms-threshold", type=float, default=None)
    p.add_argument("--scf-warm-start", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--scf-warm-start-update-interval", type=int, default=1)
    p.add_argument(
        "--recover-nonfinite-steps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After each implicit self-consistent update, run an extra full loss "
            "evaluation and revert the step if it becomes non-finite."
        ),
    )
    p.add_argument("--jit-eval", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--jit-train", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--outdir", default="outputs/closed_shell_s1_self_consistent_train")
    return p.parse_args()


def _metric_scalar(metrics: dict[str, Any], key: str, default: float = float("nan")) -> float:
    if key not in metrics:
        return default
    arr = jnp.asarray(metrics[key])
    if int(arr.size) <= 0:
        return default
    return float(arr.reshape(-1)[0])


def _metric_mean(metrics: dict[str, Any], key: str, default: float = float("nan")) -> float:
    if key not in metrics:
        return default
    arr = jnp.asarray(metrics[key])
    if int(arr.size) <= 0:
        return default
    return float(jnp.mean(arr))


def _with_scf_initial_density(molecule: Any, density: Any) -> Any:
    density_arr = jnp.asarray(density)
    if is_dataclass(molecule):
        if "scf_initial_density" in getattr(molecule, "__dataclass_fields__", {}):
            return replace(molecule, scf_initial_density=density_arr)
        return molecule
    setattr(molecule, "scf_initial_density", density_arr)
    return molecule


def _spin_summed_density_matrix(molecule: Any) -> jnp.ndarray:
    density = jnp.asarray(molecule.rdm1)
    if density.ndim == 3:
        return density.sum(axis=0)
    return density


def _refresh_dataset_scf_warm_start_cache(
    dataset: tuple[GroundStateDatum, ...],
    *,
    params: Any,
    functional: Any,
    training_config: GroundStateTrainingConfig,
) -> tuple[GroundStateDatum, ...]:
    if training_config.mode != "self_consistent":
        return dataset
    refreshed: list[GroundStateDatum] = []
    for datum in dataset:
        predicted_molecule = predict_ground_state_molecule(
            params,
            functional,
            datum.molecule,
            training_config=training_config,
        )
        refreshed.append(
            replace(
                datum,
                molecule=_with_scf_initial_density(
                    datum.molecule,
                    jax.lax.stop_gradient(_spin_summed_density_matrix(predicted_molecule)),
                ),
            )
        )
    return tuple(refreshed)


def _tree_all_finite(tree: Any) -> bool:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return True
    return all(bool(jnp.all(jnp.isfinite(jnp.asarray(leaf)))) for leaf in leaves)


def _loss_and_metrics_all_finite(loss: Any, metrics: dict[str, Any]) -> bool:
    return bool(jnp.all(jnp.isfinite(jnp.asarray(loss)))) and _tree_all_finite(metrics)


def _load_reference_rows(path: Path, *, basis: str) -> list[ReferenceRow]:
    rows: list[ReferenceRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row["basis"]).lower() != str(basis).lower():
                continue
            rows.append(
                ReferenceRow(
                    system=str(row["system"]),
                    split=str(row["split"]),
                    atom=str(row["atom"]),
                    unit=str(row["unit"]),
                    charge=int(row["charge"]),
                    spin=int(row["spin"]),
                    basis=str(row["basis"]),
                    ccsd_total_energy_h=float(row["ccsd_total_energy_h"]),
                    s1_excitation_h=float(row["s1_excitation_h"]),
                )
            )
    if not rows:
        raise ValueError(f"No rows with basis={basis!r} found in {path}.")
    return rows


def _run_rks_reference(
    row: ReferenceRow,
    *,
    xc: str,
    grids_level: int,
    scf_conv_tol: float,
    scf_max_cycle: int,
) -> Any:
    mol = gto.M(
        atom=row.atom.replace(";", "\n"),
        unit=row.unit,
        basis=row.basis,
        charge=row.charge,
        spin=row.spin,
        cart=True,
        verbose=0,
    )
    attempts = (
        dict(init_guess="minao", damping=0.15, level_shift=0.0, max_cycle=scf_max_cycle, use_newton=False),
        dict(init_guess="atom", damping=0.3, level_shift=0.5, max_cycle=max(scf_max_cycle, 180), use_newton=False),
        dict(init_guess="atom", damping=0.0, level_shift=0.0, max_cycle=max(scf_max_cycle, 80), use_newton=True),
    )
    last_mf = None
    for cfg in attempts:
        mf = dft.RKS(mol)
        mf.xc = str(xc)
        mf.grids.level = int(grids_level)
        mf.conv_tol = float(scf_conv_tol)
        mf.max_cycle = int(cfg["max_cycle"])
        mf.damping = float(cfg["damping"])
        mf.level_shift = float(cfg["level_shift"])
        mf.diis_start_cycle = 1
        mf.init_guess = str(cfg["init_guess"])
        if cfg["use_newton"]:
            mf = mf.newton()
            mf.xc = str(xc)
            mf.grids.level = int(grids_level)
            mf.conv_tol = float(scf_conv_tol)
            mf.max_cycle = int(cfg["max_cycle"])
        mf.kernel()
        last_mf = mf
        if bool(mf.converged):
            return mf
    raise RuntimeError(f"PySCF RKS did not converge for {row.system}.")


def _prepare_references(
    rows: list[ReferenceRow],
    *,
    args: argparse.Namespace,
    logger: RunLogger,
) -> list[PreparedReference]:
    prepared: list[PreparedReference] = []
    for idx, row in enumerate(rows, start=1):
        logger.log(f"[ref] {idx}/{len(rows)} build {row.system} ({row.split})")
        mf = _run_rks_reference(
            row,
            xc=str(args.xc),
            grids_level=int(args.grids_level),
            scf_conv_tol=float(args.reference_scf_conv_tol),
            scf_max_cycle=int(args.reference_scf_max_cycle),
        )
        reference = restricted_reference_from_pyscf(
            mf,
            compute_local_hfx_features=(str(args.input_feature_mode) == "dm21_original"),
            compute_local_hfx_aux=(str(args.input_feature_mode) == "dm21_original"),
            compute_local_pt2_features=bool(args.include_pt2_channel),
        )
        prepared.append(PreparedReference(row=row, molecule=reference))
    return prepared


def _build_dataset(
    prepared: list[PreparedReference],
    *,
    s1_weight: float,
    density_constraint_weight: float,
) -> tuple[GroundStateDatum, ...]:
    return tuple(
        GroundStateDatum.from_parts(
            ref.molecule,
            core=GroundStateCoreDatum(
                target_total_energy=jnp.asarray(ref.row.ccsd_total_energy_h, dtype=jnp.float64),
                density_constraint_weight=float(density_constraint_weight),
            ),
            excited_state=ExcitedStateDatum(
                target_s1_energy=jnp.asarray(ref.row.s1_excitation_h, dtype=jnp.float64),
                s1_constraint_weight=float(s1_weight),
            ),
        )
        for ref in prepared
    )


def _evaluate_dataset(
    prepared: list[PreparedReference],
    *,
    params: Any,
    functional: Any,
    training_config: GroundStateTrainingConfig,
    use_tda: bool,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    s1_err_ev: list[float] = []
    energy_err_ev: list[float] = []
    for ref in prepared:
        predicted_molecule = predict_ground_state_molecule(
            params,
            functional,
            ref.molecule,
            training_config=training_config,
        )
        predicted_total_h = float(
            predict_ground_state_total_energy(
                params,
                functional,
                ref.molecule,
                training_config=training_config,
            )
        )
        predicted_s1_h = float(
            np.asarray(
                predict_excitation_energies(
                    params,
                    functional,
                    predicted_molecule,
                    nstates=1,
                    use_tda=bool(use_tda),
                )
            ).reshape(-1)[0]
        )
        s1_abs_err_ev = abs(predicted_s1_h - ref.row.s1_excitation_h) * HARTREE_TO_EV
        total_abs_err_ev = abs(predicted_total_h - ref.row.ccsd_total_energy_h) * HARTREE_TO_EV
        s1_err_ev.append(float(s1_abs_err_ev))
        energy_err_ev.append(float(total_abs_err_ev))
        rows.append(
            {
                "system": ref.row.system,
                "split": ref.row.split,
                "target_total_energy_h": float(ref.row.ccsd_total_energy_h),
                "predicted_total_energy_h": float(predicted_total_h),
                "target_s1_h": float(ref.row.s1_excitation_h),
                "predicted_s1_h": float(predicted_s1_h),
                "target_s1_ev": float(ref.row.s1_excitation_h * HARTREE_TO_EV),
                "predicted_s1_ev": float(predicted_s1_h * HARTREE_TO_EV),
                "s1_abs_err_ev": float(s1_abs_err_ev),
                "total_abs_err_ev": float(total_abs_err_ev),
            }
        )
    metrics = {
        "s1_mae_ev": float(np.mean(s1_err_ev)) if s1_err_ev else float("nan"),
        "s1_max_ev": float(np.max(s1_err_ev)) if s1_err_ev else float("nan"),
        "total_mae_ev": float(np.mean(energy_err_ev)) if energy_err_ev else float("nan"),
    }
    return rows, metrics


def _write_prediction_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_training_history(path: Path, training: dict[str, Any], *, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    steps = np.asarray(training["history_steps"], dtype=np.int64)
    loss = np.asarray(training["loss_history"], dtype=np.float64)
    axes[0].plot(steps, np.maximum(loss, 1e-16), lw=1.6, label="pre-update train loss")
    eval_steps = np.asarray(training["eval_steps"], dtype=np.int64)
    train_eval_loss = np.asarray(training["eval_train_loss_history"], dtype=np.float64)
    val_eval_loss = np.asarray(training["eval_val_loss_history"], dtype=np.float64)
    train_eval_s1 = np.asarray(training["eval_train_s1_mae_history"], dtype=np.float64)
    val_eval_s1 = np.asarray(training["eval_val_s1_mae_history"], dtype=np.float64)
    axes[0].plot(eval_steps, np.maximum(train_eval_loss, 1e-16), "o-", ms=2.6, lw=1.0, label="train loss")
    if val_eval_loss.size:
        axes[0].plot(eval_steps, np.maximum(val_eval_loss, 1e-16), "o-", ms=2.6, lw=1.0, label="val loss")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(eval_steps, np.maximum(train_eval_s1, 1e-16), "o-", ms=2.6, lw=1.0, label="train S1 MAE")
    if val_eval_s1.size:
        axes[1].plot(eval_steps, np.maximum(val_eval_s1, 1e-16), "o-", ms=2.6, lw=1.0, label="val S1 MAE")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("S1 MAE (Eh)")
    axes[1].set_title("Generalization")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_training_history_csv(path: Path, training: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "loss_pre_update",
                "s1_mae_pre_update",
                "grad_norm",
                "grad_abs_max",
                "param_update_norm",
                "nonfinite_grad_fraction",
                "train_loss_reevaluated",
                "train_s1_mae_reevaluated",
                "val_loss_reevaluated",
                "val_s1_mae_reevaluated",
            ]
        )
        eval_map = {
            int(step): (
                float(train_loss),
                float(train_s1),
                float(val_loss),
                float(val_s1),
            )
            for step, train_loss, train_s1, val_loss, val_s1 in zip(
                training["eval_steps"],
                training["eval_train_loss_history"],
                training["eval_train_s1_mae_history"],
                training["eval_val_loss_history"],
                training["eval_val_s1_mae_history"],
                strict=False,
            )
        }
        for idx, step in enumerate(training["history_steps"]):
            train_loss, train_s1, val_loss, val_s1 = eval_map.get(
                int(step),
                (float("nan"), float("nan"), float("nan"), float("nan")),
            )
            writer.writerow(
                [
                    int(step),
                    float(training["loss_history"][idx]),
                    float(training["s1_mae_history"][idx]),
                    float(training["grad_norm_history"][idx]),
                    float(training["grad_abs_max_history"][idx]),
                    float(training["param_update_norm_history"][idx]),
                    float(training["nonfinite_grad_fraction_history"][idx]),
                    train_loss,
                    train_s1,
                    val_loss,
                    val_s1,
                ]
            )


def _train(
    train_dataset: tuple[GroundStateDatum, ...],
    val_dataset: tuple[GroundStateDatum, ...],
    *,
    init_molecule: Any,
    args: argparse.Namespace,
    logger: RunLogger,
) -> dict[str, Any]:
    functional = neural_xc.Functional(
        architecture=str(args.network_architecture),
        semilocal_xc=tuple(str(name) for name in args.semilocal_xc),
        hidden_dims=tuple(int(value) for value in args.hidden_dims),
        input_feature_mode=str(args.input_feature_mode),
        include_pt2_channel=bool(args.include_pt2_channel),
        pt2_channel_mode=str(args.pt2_channel_mode),
        name=f"neural_xc_closed_shell_{str(args.training_mode)}",
    )
    training_config = GroundStateTrainingConfig(
        mode=str(args.training_mode),
        energy_mse_weight=float(args.energy_mse_weight),
        energy_mae_weight=float(args.energy_mae_weight),
        s1_constraint_use_tda=bool(args.s1_use_tda),
        scf_max_cycle=int(args.train_scf_max_cycle),
        scf_damping=float(args.train_scf_damping),
        scf_conv_tol_density=float(args.train_scf_conv_tol_density),
        scf_vxc_clip=float(args.train_scf_vxc_clip),
        scf_iterate_selection=str(args.scf_iterate_selection),
        scf_require_convergence=bool(args.scf_require_convergence),
        scf_stop_gradient_on_unconverged=bool(args.scf_stop_gradient_on_unconverged),
        scf_stop_gradient_rms_threshold=args.scf_stop_gradient_rms_threshold,
        scf_gradient_mode=str(args.scf_gradient_mode),
        scf_implicit_diff_solver=str(args.scf_implicit_diff_solver),
        scf_implicit_diff_tolerance=float(args.scf_implicit_diff_tolerance),
        scf_implicit_diff_regularization=float(args.scf_implicit_diff_regularization),
        scf_implicit_diff_restart=int(args.scf_implicit_diff_restart),
    )
    if int(args.lr_decay_every) > 0:
        lr_schedule = optax.exponential_decay(
            init_value=float(args.learning_rate),
            transition_steps=int(args.lr_decay_every),
            decay_rate=float(args.lr_decay_factor),
            staircase=True,
        )
        base_optimizer = optax.adam(lr_schedule)
    else:
        lr_schedule = None
        base_optimizer = optax.adam(float(args.learning_rate))
    optimizer = (
        optax.chain(optax.clip_by_global_norm(float(args.grad_clip_norm)), base_optimizer)
        if args.grad_clip_norm is not None and float(args.grad_clip_norm) > 0.0
        else base_optimizer
    )
    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(int(args.seed)),
        init_molecule,
        optimizer,
    )
    train_step = make_ground_state_train_step(functional, training_config=training_config)
    use_warm_start = bool(args.scf_warm_start) and training_config.mode == "self_consistent"
    train_dataset_current = train_dataset
    val_dataset_current = val_dataset
    if use_warm_start:
        train_eval_fn = lambda params, dataset: ground_state_mse_loss(  # noqa: E731
            params,
            functional,
            dataset,
            training_config=training_config,
        )
        val_eval_fn = (
            (lambda params, dataset: ground_state_mse_loss(  # noqa: E731
                params,
                functional,
                dataset,
                training_config=training_config,
            ))
            if val_dataset
            else None
        )
        eager_train_step = lambda current_state, dataset: train_step(current_state, dataset)  # noqa: E731
        compiled_train_eval = train_eval_fn
        compiled_val_eval = val_eval_fn
        compiled_train_step = eager_train_step
        train_step_mode = "eager"
        if bool(args.jit_train) or bool(args.jit_eval):
            logger.log("[train] disabled JIT because scf_warm_start requires dynamic dataset updates")
        initial_train_loss, initial_train_metrics = compiled_train_eval(state.params, train_dataset_current)
    else:
        train_eval_fn = lambda params: ground_state_mse_loss(  # noqa: E731
            params,
            functional,
            train_dataset,
            training_config=training_config,
        )
        val_eval_fn = (
            (lambda params: ground_state_mse_loss(  # noqa: E731
                params,
                functional,
                val_dataset,
                training_config=training_config,
            ))
            if val_dataset
            else None
        )
        eager_train_step = lambda current_state: train_step(current_state, train_dataset)  # noqa: E731
        compiled_train_eval = jax.jit(train_eval_fn) if bool(args.jit_eval) else train_eval_fn
        compiled_val_eval = (
            jax.jit(val_eval_fn) if (val_eval_fn is not None and bool(args.jit_eval)) else val_eval_fn
        )
        compiled_train_step = eager_train_step
        train_step_mode = "eager"
        if bool(args.jit_train):
            candidate = jax.jit(eager_train_step)
            try:
                _ = candidate.lower(state).compile()
                compiled_train_step = candidate
                train_step_mode = "jit"
            except Exception as exc:
                logger.log(f"[train] jit compilation failed for multi-molecule train step: {exc!r}")
        initial_train_loss, initial_train_metrics = compiled_train_eval(state.params)
    if bool(args.scf_warm_start):
        train_dataset_current = _refresh_dataset_scf_warm_start_cache(
            train_dataset_current,
            params=state.params,
            functional=functional,
            training_config=training_config,
        )
    if compiled_val_eval is not None:
        initial_val_loss, initial_val_metrics = (
            compiled_val_eval(state.params, val_dataset_current)
            if use_warm_start
            else compiled_val_eval(state.params)
        )
        if bool(args.scf_warm_start):
            val_dataset_current = _refresh_dataset_scf_warm_start_cache(
                val_dataset_current,
                params=state.params,
                functional=functional,
                training_config=training_config,
            )
        best_score = _metric_mean(initial_val_metrics, "s1_mae", float(initial_val_loss))
    else:
        initial_val_loss, initial_val_metrics = None, None
        best_score = _metric_mean(initial_train_metrics, "s1_mae", float(initial_train_loss))
    best_params = state.params
    best_step = 0

    history_steps = [0]
    loss_history = [float(initial_train_loss)]
    s1_mae_history = [_metric_mean(initial_train_metrics, "s1_mae", 0.0)]
    grad_norm_history = [float("nan")]
    grad_abs_max_history = [float("nan")]
    param_update_norm_history = [float("nan")]
    nonfinite_grad_fraction_history = [0.0]
    eval_steps = [0]
    eval_train_loss_history = [float(initial_train_loss)]
    eval_train_s1_mae_history = [_metric_mean(initial_train_metrics, "s1_mae", float("nan"))]
    eval_val_loss_history = [float(initial_val_loss) if initial_val_loss is not None else float("nan")]
    eval_val_s1_mae_history = [_metric_mean(initial_val_metrics, "s1_mae", float("nan")) if initial_val_metrics is not None else float("nan")]

    logger.log(
        "[train] "
        f"steps={int(args.steps)} mode={str(args.training_mode)} objective=s1_only_{'tda' if bool(args.s1_use_tda) else 'casida'} "
        f"train_step_mode={train_step_mode} train_size={len(train_dataset)} val_size={len(val_dataset)}"
    )

    t0 = time.perf_counter()
    post_update_recoveries = 0
    guard_post_update = (
        bool(args.recover_nonfinite_steps)
        and training_config.mode == "self_consistent"
        and str(args.scf_gradient_mode) == "implicit_commutator"
    )
    for step in range(1, int(args.steps) + 1):
        prev_state = state
        prev_train_dataset_current = train_dataset_current
        state, train_metrics = (
            compiled_train_step(state, train_dataset_current)
            if use_warm_start
            else compiled_train_step(state)
        )
        reverted_step = False
        if not _tree_all_finite(state.params):
            state = prev_state
            train_dataset_current = prev_train_dataset_current
            reverted_step = True
            logger.log(f"[train] non-finite params at step {step}; reverted update")
        elif (
            bool(args.scf_warm_start)
            and training_config.mode == "self_consistent"
            and step % max(1, int(args.scf_warm_start_update_interval)) == 0
        ):
            train_dataset_current = _refresh_dataset_scf_warm_start_cache(
                train_dataset_current,
                params=state.params,
                functional=functional,
                training_config=training_config,
            )
        if guard_post_update and not reverted_step:
            guarded_train_loss, guarded_train_metrics = (
                compiled_train_eval(state.params, train_dataset_current)
                if use_warm_start
                else compiled_train_eval(state.params)
            )
            if not _loss_and_metrics_all_finite(guarded_train_loss, guarded_train_metrics):
                state = prev_state
                train_dataset_current = prev_train_dataset_current
                reverted_step = True
                post_update_recoveries += 1
                logger.log(
                    f"[train] non-finite post-update train eval at step {step}; reverted update"
                )

        train_loss_val = _metric_scalar(train_metrics, "loss")
        train_s1_mae_val = _metric_mean(train_metrics, "s1_mae", float("nan"))
        grad_norm_val = _metric_scalar(train_metrics, "grad_norm")
        grad_abs_max_val = _metric_scalar(train_metrics, "grad_abs_max")
        update_norm_val = (
            0.0 if reverted_step else _metric_scalar(train_metrics, "param_update_norm")
        )
        nonfinite_grad_fraction_val = _metric_scalar(train_metrics, "nonfinite_grad_fraction", 0.0)
        history_steps.append(step)
        loss_history.append(train_loss_val)
        s1_mae_history.append(train_s1_mae_val)
        grad_norm_history.append(grad_norm_val)
        grad_abs_max_history.append(grad_abs_max_val)
        param_update_norm_history.append(update_norm_val)
        nonfinite_grad_fraction_history.append(nonfinite_grad_fraction_val)

        should_eval = (
            step == 1
            or step == int(args.steps)
            or step % max(1, int(args.eval_interval)) == 0
        )
        if should_eval:
            eval_train_loss, eval_train_metrics = (
                compiled_train_eval(state.params, train_dataset_current)
                if use_warm_start
                else compiled_train_eval(state.params)
            )
            eval_steps.append(step)
            eval_train_loss_history.append(float(eval_train_loss))
            eval_train_s1_mae_history.append(_metric_mean(eval_train_metrics, "s1_mae", float("nan")))
            if compiled_val_eval is not None:
                eval_val_loss, eval_val_metrics = (
                    compiled_val_eval(state.params, val_dataset_current)
                    if use_warm_start
                    else compiled_val_eval(state.params)
                )
                if bool(args.scf_warm_start):
                    val_dataset_current = _refresh_dataset_scf_warm_start_cache(
                        val_dataset_current,
                        params=state.params,
                        functional=functional,
                        training_config=training_config,
                    )
                val_loss_val = float(eval_val_loss)
                val_s1_mae_val = _metric_mean(eval_val_metrics, "s1_mae", float("nan"))
                eval_val_loss_history.append(val_loss_val)
                eval_val_s1_mae_history.append(val_s1_mae_val)
                score = val_s1_mae_val
            else:
                eval_val_loss_history.append(float("nan"))
                eval_val_s1_mae_history.append(float("nan"))
                score = _metric_mean(eval_train_metrics, "s1_mae", float(eval_train_loss))
            if score < best_score:
                best_score = score
                best_params = state.params
                best_step = step
            current_lr = float(lr_schedule(step - 1)) if lr_schedule is not None else float(args.learning_rate)
            logger.log(
                "[train] "
                f"step={step:4d}/{int(args.steps):4d} "
                f"train_loss={float(eval_train_loss):.8e} "
                f"train_s1_mae={_metric_mean(eval_train_metrics, 's1_mae', float('nan')):.8e} "
                f"val_loss={eval_val_loss_history[-1]:.8e} "
                f"val_s1_mae={eval_val_s1_mae_history[-1]:.8e} "
                f"train_scf_stop_grad_frac={_metric_mean(eval_train_metrics, 'scf_stop_gradient_fraction', float('nan')):.8e} "
                f"grad_norm={grad_norm_val:.8e} "
                f"grad_abs_max={grad_abs_max_val:.8e} "
                f"update_norm={update_norm_val:.8e} "
                f"lr={current_lr:.8e} "
                f"recoveries={post_update_recoveries:d}"
            )

    elapsed_s = time.perf_counter() - t0
    final_train_loss, final_train_metrics = (
        compiled_train_eval(state.params, train_dataset_current)
        if use_warm_start
        else compiled_train_eval(state.params)
    )
    final_val_loss = None
    final_val_metrics = None
    if compiled_val_eval is not None:
        final_val_loss, final_val_metrics = (
            compiled_val_eval(state.params, val_dataset_current)
            if use_warm_start
            else compiled_val_eval(state.params)
        )
    logger.log(
        "[train] done "
        f"final_train_loss={float(final_train_loss):.8e} "
        f"best_val_s1_mae={best_score:.8e}@{best_step} "
        f"elapsed_s={elapsed_s:.2f}"
    )
    return {
        "functional": functional,
        "training_config": training_config,
        "best_params": best_params,
        "final_params": state.params,
        "best_step": int(best_step),
        "best_score": float(best_score),
        "final_train_loss": float(final_train_loss),
        "final_train_s1_mae": _metric_mean(final_train_metrics, "s1_mae", float("nan")),
        "final_val_loss": float(final_val_loss) if final_val_loss is not None else float("nan"),
        "final_val_s1_mae": _metric_mean(final_val_metrics, "s1_mae", float("nan")) if final_val_metrics is not None else float("nan"),
        "elapsed_s": float(elapsed_s),
        "history_steps": history_steps,
        "loss_history": loss_history,
        "s1_mae_history": s1_mae_history,
        "grad_norm_history": grad_norm_history,
        "grad_abs_max_history": grad_abs_max_history,
        "param_update_norm_history": param_update_norm_history,
        "nonfinite_grad_fraction_history": nonfinite_grad_fraction_history,
        "eval_steps": eval_steps,
        "eval_train_loss_history": eval_train_loss_history,
        "eval_train_s1_mae_history": eval_train_s1_mae_history,
        "eval_val_loss_history": eval_val_loss_history,
        "eval_val_s1_mae_history": eval_val_s1_mae_history,
        "post_update_recoveries": int(post_update_recoveries),
    }


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(outdir / "run.log")

    rows = _load_reference_rows(Path(args.reference_csv), basis=str(args.basis))
    train_rows = [row for row in rows if row.split == "train"]
    val_rows = [row for row in rows if row.split == "validation"]
    test_rows = [row for row in rows if row.split == "test"]
    if not train_rows:
        raise ValueError("Training split is empty.")
    logger.log(
        "Config: "
        f"reference_csv={args.reference_csv}, basis={args.basis}, xc={args.xc}, "
        f"steps={args.steps}, mode={args.training_mode}, include_pt2_channel={bool(args.include_pt2_channel)}, "
        f"pt2_channel_mode={args.pt2_channel_mode if bool(args.include_pt2_channel) else 'none'}, "
        f"train={len(train_rows)}, validation={len(val_rows)}, test={len(test_rows)}"
    )

    prepared_train = _prepare_references(train_rows, args=args, logger=logger)
    prepared_val = _prepare_references(val_rows, args=args, logger=logger)
    prepared_test = _prepare_references(test_rows, args=args, logger=logger)

    train_dataset = _build_dataset(
        prepared_train,
        s1_weight=float(args.s1_weight),
        density_constraint_weight=float(args.density_constraint_weight),
    )
    val_dataset = _build_dataset(
        prepared_val,
        s1_weight=float(args.s1_weight),
        density_constraint_weight=float(args.density_constraint_weight),
    )

    training = _train(
        train_dataset,
        val_dataset,
        init_molecule=prepared_train[0].molecule,
        args=args,
        logger=logger,
    )

    params = training["best_params"]
    functional = training["functional"]
    training_config = training["training_config"]
    use_tda = bool(args.eval_use_tda)
    train_pred_rows, train_metrics = _evaluate_dataset(
        prepared_train,
        params=params,
        functional=functional,
        training_config=training_config,
        use_tda=use_tda,
    )
    val_pred_rows, val_metrics = _evaluate_dataset(
        prepared_val,
        params=params,
        functional=functional,
        training_config=training_config,
        use_tda=use_tda,
    )
    test_pred_rows, test_metrics = _evaluate_dataset(
        prepared_test,
        params=params,
        functional=functional,
        training_config=training_config,
        use_tda=use_tda,
    )

    predictions_csv = outdir / "predictions.csv"
    _write_prediction_csv(predictions_csv, train_pred_rows + val_pred_rows + test_pred_rows)
    training_png = outdir / "training_loss.png"
    _plot_training_history(
        training_png,
        training,
        title=f"Closed-shell S1 {'TDA' if bool(args.s1_use_tda) else 'Casida'} self-consistent training",
    )
    training_history_csv = outdir / "training_history.csv"
    _write_training_history_csv(training_history_csv, training)

    checkpoint_path = outdir / "neural_xc_params.msgpack"
    checkpoint_path, checkpoint_meta_path = save_params_checkpoint(
        checkpoint_path,
        params,
        metadata={
            "reference_csv": str(args.reference_csv),
            "basis": str(args.basis),
            "xc": str(args.xc),
            "training_mode": str(args.training_mode),
            "scf_stop_gradient_on_unconverged": bool(args.scf_stop_gradient_on_unconverged),
            "scf_stop_gradient_rms_threshold": args.scf_stop_gradient_rms_threshold,
            "scf_warm_start": bool(args.scf_warm_start),
            "scf_warm_start_update_interval": int(args.scf_warm_start_update_interval),
            "recover_nonfinite_steps": bool(args.recover_nonfinite_steps),
            "scf_gradient_mode": str(args.scf_gradient_mode),
            "scf_implicit_diff_solver": str(args.scf_implicit_diff_solver),
            "scf_implicit_diff_tolerance": float(args.scf_implicit_diff_tolerance),
            "scf_implicit_diff_regularization": float(args.scf_implicit_diff_regularization),
            "scf_implicit_diff_restart": int(args.scf_implicit_diff_restart),
            "include_pt2_channel": bool(args.include_pt2_channel),
            "pt2_channel_mode": str(args.pt2_channel_mode) if bool(args.include_pt2_channel) else None,
            "s1_use_tda": bool(args.s1_use_tda),
            "eval_use_tda": bool(args.eval_use_tda),
            "steps": int(args.steps),
            "best_step": int(training["best_step"]),
            "train_systems": [row.system for row in train_rows],
            "validation_systems": [row.system for row in val_rows],
            "test_systems": [row.system for row in test_rows],
        },
    )

    benzene_row = next((row for row in test_pred_rows if row["system"] == "benzene"), None)
    summary = {
        "reference_csv": str(args.reference_csv),
        "basis": str(args.basis),
        "xc": str(args.xc),
        "training_mode": str(args.training_mode),
        "objective": f"s1_only_{'tda' if bool(args.s1_use_tda) else 'casida'}",
        "evaluation_solver": "tda" if bool(args.eval_use_tda) else "casida",
        "scf_stop_gradient_on_unconverged": bool(args.scf_stop_gradient_on_unconverged),
        "scf_stop_gradient_rms_threshold": args.scf_stop_gradient_rms_threshold,
        "scf_warm_start": bool(args.scf_warm_start),
        "scf_warm_start_update_interval": int(args.scf_warm_start_update_interval),
        "recover_nonfinite_steps": bool(args.recover_nonfinite_steps),
        "scf_gradient_mode": str(args.scf_gradient_mode),
        "scf_implicit_diff_solver": str(args.scf_implicit_diff_solver),
        "scf_implicit_diff_tolerance": float(args.scf_implicit_diff_tolerance),
        "scf_implicit_diff_regularization": float(args.scf_implicit_diff_regularization),
        "scf_implicit_diff_restart": int(args.scf_implicit_diff_restart),
        "include_pt2_channel": bool(args.include_pt2_channel),
        "pt2_channel_mode": str(args.pt2_channel_mode) if bool(args.include_pt2_channel) else None,
        "steps": int(args.steps),
        "best_step": int(training["best_step"]),
        "best_validation_s1_mae_h": float(training["best_score"]),
        "final_train_loss": float(training["final_train_loss"]),
        "final_train_s1_mae_h": float(training["final_train_s1_mae"]),
        "final_val_loss": float(training["final_val_loss"]),
        "final_val_s1_mae_h": float(training["final_val_s1_mae"]),
        "train_s1_mae_ev": float(train_metrics["s1_mae_ev"]),
        "validation_s1_mae_ev": float(val_metrics["s1_mae_ev"]),
        "test_s1_mae_ev": float(test_metrics["s1_mae_ev"]),
        "benzene_s1_predicted_ev": float(benzene_row["predicted_s1_ev"]) if benzene_row is not None else None,
        "benzene_s1_target_ev": float(benzene_row["target_s1_ev"]) if benzene_row is not None else None,
        "benzene_s1_abs_err_ev": float(benzene_row["s1_abs_err_ev"]) if benzene_row is not None else None,
        "predictions_csv": str(predictions_csv),
        "training_curve_png": str(training_png),
        "training_history_csv": str(training_history_csv),
        "checkpoint": str(checkpoint_path),
        "checkpoint_meta": str(checkpoint_meta_path) if checkpoint_meta_path is not None else None,
        "train_systems": [row.system for row in train_rows],
        "validation_systems": [row.system for row in val_rows],
        "test_systems": [row.system for row in test_rows],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    logger.log(f"Wrote predictions: {predictions_csv}")
    logger.log(f"Wrote summary   : {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
