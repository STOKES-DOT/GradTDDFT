from __future__ import annotations

import argparse
import csv
from dataclasses import is_dataclass, replace
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

import h5py
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax
from pyscf import ao2mo, fci, gto, scf

from td_graddft import neural_xc
from td_graddft.data.hdf5_cache import read_restricted_molecule, write_restricted_molecule
from td_graddft.neural_xc import (
    DEFAULT_INPUT_FEATURE_MODE,
    DEFAULT_NEURAL_XC_RESPONSE_HF_MODE,
    DEFAULT_NETWORK_ARCHITECTURE,
    DEFAULT_NETWORK_HIDDEN_DIMS,
)
from td_graddft.spectra import HARTREE_TO_EV
from td_graddft.training import (
    ExcitedStateDatum,
    GroundStateCoreDatum,
    GroundStateDatum,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    ground_state_mse_loss,
    load_params_checkpoint,
    make_ground_state_predictor,
    make_ground_state_loss_and_grad,
    make_ground_state_train_step,
    predict_excitation_energies,
    predict_ground_state_molecule,
    predict_oscillator_strengths,
    save_params_checkpoint,
)


_HELPER_PATH = Path(__file__).with_name("h2_self_consistent_ground_train5_dense100_vs_fci.py")
_HELPER_SPEC = importlib.util.spec_from_file_location("_h2_ground_vs_fci_helpers", _HELPER_PATH)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise RuntimeError(f"Failed to load helper module from {_HELPER_PATH}")
_HELPERS = importlib.util.module_from_spec(_HELPER_SPEC)
sys.modules[_HELPER_SPEC.name] = _HELPERS
_HELPER_SPEC.loader.exec_module(_HELPERS)


RunLogger = _HELPERS.RunLogger
build_reference_curve = _HELPERS.build_reference_curve
write_dense_csv = _HELPERS.write_dense_csv

_DEFAULT_SEMILOCAL_XC = ("lda_x", "gga_x_b88", "lda_c_vwn_rpa", "gga_c_lyp")
_OBJECTIVE_CHOICES = ("auto", "e0_only", "s1_only", "joint")


def _get_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _normalize_input_feature_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode in {"dm21_original", "canonical"}:
        return "canonical"
    if mode == "enhanced":
        return "enhanced"
    raise ValueError(
        f"Unsupported input feature mode {value!r}. Expected enhanced, canonical, or dm21_original."
    )


def _normalize_scf_gradient_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode in {"implicit_commutator", "impl"}:
        return "impl"
    if mode in {"unrolled", "expl"}:
        return "expl"
    raise ValueError(
        f"Unsupported SCF gradient mode {value!r}. Expected impl, expl, implicit_commutator, or unrolled."
    )


def _normalize_objective(value: str) -> str:
    objective = str(value).strip().lower()
    if objective in _OBJECTIVE_CHOICES:
        return objective
    raise ValueError(
        f"Unsupported objective {value!r}. Expected one of {', '.join(_OBJECTIVE_CHOICES)}."
    )


def _has_ground_supervision(args: argparse.Namespace) -> bool:
    return any(
        float(weight) > 0.0
        for weight in (
            args.energy_mse_weight,
            args.energy_mae_weight,
            args.density_constraint_weight,
        )
    )


def _has_s1_supervision(args: argparse.Namespace) -> bool:
    return float(args.s1_weight) > 0.0


def _resolved_objective_kind(args: argparse.Namespace) -> str:
    objective = str(args.objective)
    if objective != "auto":
        return objective
    has_ground = _has_ground_supervision(args)
    has_s1 = _has_s1_supervision(args)
    if has_ground and has_s1:
        return "joint"
    if has_ground:
        return "e0_only"
    if has_s1:
        return "s1_only"
    raise ValueError(
        "No active supervision terms remain. Enable S1 supervision, ground-state supervision, or set --objective explicitly."
    )


def _objective_solver_name(args: argparse.Namespace) -> str:
    return "tda" if bool(args.s1_use_tda) else "casida"


def _objective_name(args: argparse.Namespace) -> str:
    kind = _resolved_objective_kind(args)
    if kind == "e0_only":
        return "e0_only"
    return f"{kind}_{_objective_solver_name(args)}"


def _objective_display_label(args: argparse.Namespace) -> str:
    kind = _resolved_objective_kind(args)
    if kind == "e0_only":
        return "E0-only"
    solver_label = "TDA" if bool(args.s1_use_tda) else "Casida"
    if kind == "s1_only":
        return f"S1-total-only {solver_label}"
    return f"Joint {solver_label}"


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.input_feature_mode = _normalize_input_feature_mode(args.input_feature_mode)
    args.scf_gradient_mode = _normalize_scf_gradient_mode(args.scf_gradient_mode)
    args.objective = _normalize_objective(args.objective)
    if str(args.objective) == "e0_only":
        args.s1_weight = 0.0
        if not _has_ground_supervision(args):
            args.energy_mae_weight = 1.0
    elif str(args.objective) == "s1_only":
        args.s1_weight = 1.0 if float(args.s1_weight) <= 0.0 else float(args.s1_weight)
        args.energy_mse_weight = 0.0
        args.energy_mae_weight = 0.0
        args.density_constraint_weight = 0.0
    elif str(args.objective) == "joint":
        args.s1_weight = 1.0 if float(args.s1_weight) <= 0.0 else float(args.s1_weight)
        if not _has_ground_supervision(args):
            args.energy_mae_weight = 1.0
    _resolved_objective_kind(args)
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train Neural_xc on the H2 first excited-state total energy curve "
            "E1(R)=E0(R)+Omega1(R), using TDA for the excitation contribution, "
            "then compare dense TDA results against FCI and plot the equilibrium "
            "stick spectrum."
        )
    )
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--xc", default="b3lyp")
    p.add_argument("--system-label", default="H2")
    p.add_argument("--atom1", default="H")
    p.add_argument("--atom2", default="H")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--spin", type=int, default=0)
    p.add_argument(
        "--external-s1-total-csv",
        default=None,
        help=(
            "Optional CSV containing an external first-excited-state total-energy "
            "curve. When provided, FCI reference construction is skipped and this "
            "curve supplies target_first_excited_total_energy."
        ),
    )
    p.add_argument("--external-r-column", default="r_angstrom")
    p.add_argument("--external-s1-total-column", default="mr_ccca_hartree")
    p.add_argument(
        "--external-reference-label",
        default=None,
        help="Label used in plots/summary for the external target curve.",
    )
    p.add_argument("--r-min", type=float, default=0.05)
    p.add_argument("--r-max", type=float, default=5.0)
    p.add_argument(
        "--train-r-values",
        type=float,
        nargs="+",
        default=None,
        help="Optional explicit training bond lengths in Angstrom; overrides linspace train points.",
    )
    p.add_argument("--train-points", type=int, default=5)
    p.add_argument("--dense-points", type=int, default=100)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument(
        "--reference-cache",
        default=None,
        help="Optional HDF5 file used to cache reference molecules, grids, and integrals.",
    )
    p.add_argument(
        "--rebuild-reference-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rebuild cached reference groups even when they already exist.",
    )
    p.add_argument(
        "--stream-train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Train from per-geometry dynamic inputs instead of capturing the full "
            "training set inside one JIT graph."
        ),
    )
    p.add_argument(
        "--skip-initial-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="In streaming mode, enter the first train step without a pre-train loss evaluation.",
    )
    p.add_argument(
        "--defer-dense-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Build dense references only after training, reducing peak pre-train memory.",
    )
    p.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional Flax msgpack checkpoint used to initialize the Neural_xc params.",
    )
    p.add_argument(
        "--fixed-density-reference-checkpoint",
        default=None,
        help=(
            "Optional Flax msgpack checkpoint used to build frozen fixed-density "
            "molecules via self-consistent prediction before S1 training/evaluation."
        ),
    )
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--lr-decay-every", type=int, default=500)
    p.add_argument("--lr-decay-factor", type=float, default=0.5)
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help=(
            "Write neural_xc_params_latest.msgpack every N optimizer steps. "
            "Best checkpoints are saved immediately when the training loss improves; "
            "set to 0 to disable periodic latest checkpoints."
        ),
    )
    p.add_argument(
        "--training-mode",
        choices=("fixed_density", "self_consistent"),
        default="self_consistent",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=list(DEFAULT_NETWORK_HIDDEN_DIMS),
    )
    p.add_argument(
        "--network-architecture",
        choices=("simple_mlp", "graddft_residual"),
        default=DEFAULT_NETWORK_ARCHITECTURE,
    )
    p.add_argument(
        "--input-feature-mode",
        choices=("enhanced", "canonical", "dm21_original"),
        default=DEFAULT_INPUT_FEATURE_MODE,
    )
    p.add_argument(
        "--include-pt2-channel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add the projected restricted MP2 local channel to the Neural_xc basis.",
    )
    p.add_argument(
        "--include-hfx-channel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add the projected local-HF exchange channel to the Neural_xc basis.",
    )
    p.add_argument(
        "--response-hf-mode",
        choices=("approx", "strict"),
        default=DEFAULT_NEURAL_XC_RESPONSE_HF_MODE,
        help=(
            "Excited-state response mode for the local-HF channel. Strict HF "
            "response is currently gated by the response implementation."
        ),
    )
    p.add_argument(
        "--pt2-channel-mode",
        choices=("scaled_projected", "local_exact"),
        default="scaled_projected",
        help="Choose the PT2 local channel representation when PT2 is enabled.",
    )
    p.add_argument(
        "--response-pt2-mode",
        choices=("approx", "strict"),
        default="approx",
        help=(
            "PT2 response mode. 'approx' keeps PT2 as a frozen energy-density "
            "channel in the response kernel; 'strict' solves the no-PT2 response "
            "and adds a post-hoc second-order correction."
        ),
    )
    p.add_argument(
        "--semilocal-xc",
        nargs="+",
        default=list(_DEFAULT_SEMILOCAL_XC),
        help="Neural_xc semilocal basis channels.",
    )
    p.add_argument(
        "--s1-weight",
        type=float,
        default=1.0,
        help="Weight for the first excited-state total-energy supervision term.",
    )
    p.add_argument(
        "--s1-use-tda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use TDA, not full Casida, for S1 supervision during training.",
    )
    p.add_argument(
        "--eval-use-tda",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use TDA for dense-curve evaluation and equilibrium spectrum. Defaults to the S1 supervision mode.",
    )
    p.add_argument(
        "--objective",
        choices=_OBJECTIVE_CHOICES,
        default="auto",
        help=(
            "Training objective selection. 'auto' infers the mode from the active "
            "loss weights; explicit modes disable or backfill supervision weights "
            "to match E0-only, S1-total-only, or joint training."
        ),
    )
    p.add_argument(
        "--energy-mse-weight",
        type=float,
        default=0.0,
        help="Ground-state energy MSE weight. Keep at 0 for S1-total-only training.",
    )
    p.add_argument(
        "--energy-mae-weight",
        type=float,
        default=0.0,
        help="Ground-state energy MAE weight. Keep at 0 for S1-total-only training.",
    )
    p.add_argument(
        "--density-constraint-weight",
        type=float,
        default=0.0,
        help="Density-matching weight. Keep at 0 for pure S1 training.",
    )
    p.add_argument("--grids-level", type=int, default=2)
    p.add_argument("--max-l", type=int, default=3)
    p.add_argument(
        "--grid-ao-backend",
        choices=("jax", "pyscf"),
        default="jax",
    )
    p.add_argument(
        "--integral-backend",
        choices=("jax", "cpu", "gpu", "libcint"),
        default="cpu",
    )
    p.add_argument(
        "--jk-backend",
        choices=("full", "df"),
        default="full",
    )
    p.add_argument("--df-tol", type=float, default=1e-10)
    p.add_argument("--df-max-rank", type=int, default=None)
    p.add_argument("--reference-scf-max-cycle", type=int, default=80)
    p.add_argument("--reference-scf-conv-tol", type=float, default=1e-10)
    p.add_argument("--reference-scf-conv-tol-density", type=float, default=1e-8)
    p.add_argument("--reference-scf-damping", type=float, default=0.15)
    p.add_argument("--reference-scf-potential-clip", type=float, default=20.0)
    p.add_argument(
        "--reference-scf-backend",
        choices=("pyscf", "jax_rks"),
        default="pyscf",
        help="SCF backend used to build the training/evaluation reference molecules.",
    )
    p.add_argument(
        "--train-scf-max-cycle",
        type=int,
        default=16,
        help=(
            "SCF scan safety cap during training. Use 0 only for the helper "
            "default high cap when debugging convergence."
        ),
    )
    p.add_argument("--train-scf-damping", type=float, default=0.25)
    p.add_argument("--train-scf-conv-tol-energy", type=float, default=None)
    p.add_argument(
        "--train-scf-convergence-metric",
        choices=("energy_and_residual", "energy"),
        default="energy_and_residual",
    )
    p.add_argument("--train-scf-conv-tol-density", type=float, default=1e-8)
    p.add_argument("--train-scf-vxc-clip", type=float, default=20.0)
    p.add_argument(
        "--scf-iterate-selection",
        choices=("final", "best_rms", "first_converged"),
        default="best_rms",
    )
    p.add_argument(
        "--scf-gradient-mode",
        choices=("expl", "impl", "unrolled", "implicit_commutator"),
        default="impl",
    )
    p.add_argument(
        "--scf-implicit-diff-solver",
        choices=("normal_cg", "gmres", "bicgstab"),
        default="normal_cg",
        help="Deprecated compatibility option; the current implicit SCF path does not use a named linear solver.",
    )
    p.add_argument("--scf-implicit-diff-max-iter", type=int, default=6)
    p.add_argument("--scf-implicit-diff-clip", type=float, default=1e4)
    p.add_argument("--scf-implicit-diff-tolerance", type=float, default=1e-6)
    p.add_argument("--scf-implicit-diff-regularization", type=float, default=1e-3)
    p.add_argument(
        "--scf-implicit-diff-restart",
        type=int,
        default=12,
        help="Deprecated compatibility option; retained so older launch scripts keep parsing.",
    )
    p.add_argument(
        "--scf-require-convergence",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--grad-clip-norm", type=float, default=None)
    p.add_argument(
        "--scf-warm-start",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
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
    p.add_argument(
        "--jit-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="JIT-compile the loss evaluation closure over the 5-point training set.",
    )
    p.add_argument(
        "--jit-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attempt to JIT the S1-total-only train step.",
    )
    p.add_argument(
        "--equilibrium-spectrum-nstates",
        type=int,
        default=3,
        help="Number of TDA states used for the equilibrium stick spectrum.",
    )
    p.add_argument(
        "--skip-equilibrium-spectrum",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip the equilibrium stick spectrum. Useful for external C2 targets where FCI is not rebuilt.",
    )
    p.add_argument(
        "--excited-nstates",
        type=int,
        default=3,
        help="Number of reference excited states to build for each geometry.",
    )
    p.add_argument(
        "--outdir",
        default="outputs/h2_s1_tda_train5_dense100_vs_fci",
    )
    return _normalize_args(p.parse_args(argv))


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


def _s1_total_supervision_penalty(metrics: dict[str, Any]) -> float:
    if "first_excited_total_penalty" in metrics:
        return _metric_mean(metrics, "first_excited_total_penalty", 0.0)
    return _metric_mean(metrics, "s1_penalty", 0.0)


def _s1_total_supervision_mse(metrics: dict[str, Any]) -> float:
    if "first_excited_total_mse" in metrics:
        return _metric_mean(metrics, "first_excited_total_mse", 0.0)
    return _metric_mean(metrics, "s1_mse", 0.0)


def _s1_total_supervision_mae(metrics: dict[str, Any]) -> float:
    if "first_excited_total_mae" in metrics:
        return _metric_mean(metrics, "first_excited_total_mae", 0.0)
    mse = _s1_total_supervision_mse(metrics)
    if np.isfinite(mse):
        return float(np.sqrt(max(mse, 0.0)))
    return _metric_mean(metrics, "s1_mae", 0.0)


def _s1_total_predicted_metric(metrics: dict[str, Any]) -> float:
    if "first_excited_total_predicted" in metrics:
        return _metric_mean(metrics, "first_excited_total_predicted", float("nan"))
    return _metric_mean(metrics, "s1_predicted", float("nan"))


def _s1_total_target_metric(metrics: dict[str, Any]) -> float:
    if "first_excited_total_target" in metrics:
        return _metric_mean(metrics, "first_excited_total_target", float("nan"))
    return _metric_mean(metrics, "s1_target", float("nan"))


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


def _tree_l2_norm(tree: Any) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    total = jnp.asarray(0.0, dtype=jnp.float32)
    for leaf in leaves:
        arr = jnp.asarray(leaf)
        arr = jnp.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        total = total + jnp.sum(jnp.square(arr.astype(jnp.float32)))
    return jnp.sqrt(total)


def _loss_and_metrics_all_finite(loss: Any, metrics: dict[str, Any]) -> bool:
    return bool(jnp.all(jnp.isfinite(jnp.asarray(loss)))) and _tree_all_finite(metrics)


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _resolved_train_r_values(args: argparse.Namespace) -> np.ndarray:
    if getattr(args, "train_r_values", None) is None:
        return np.linspace(float(args.r_min), float(args.r_max), int(args.train_points))
    return np.asarray(args.train_r_values, dtype=np.float64)


def _training_checkpoint_metadata(
    args: argparse.Namespace,
    *,
    checkpoint_kind: str,
    parameter_state: str,
    step: int,
    train_r_values: np.ndarray | None = None,
    eval_use_tda: bool | None = None,
    loss: float | None = None,
    s1_total_mae: float | None = None,
    s1_total_mse: float | None = None,
    s1_total_penalty: float | None = None,
    learning_rate: float | None = None,
    min_loss: float | None = None,
    min_loss_step: int | None = None,
) -> dict[str, Any]:
    if train_r_values is None:
        train_r_values = _resolved_train_r_values(args)
    resolved_eval_use_tda = (
        bool(args.s1_use_tda) if eval_use_tda is None else bool(eval_use_tda)
    )
    return {
        "checkpoint_kind": str(checkpoint_kind),
        "parameter_state": str(parameter_state),
        "step": int(step),
        "loss": _finite_float_or_none(loss),
        "s1_total_mae": _finite_float_or_none(s1_total_mae),
        "s1_total_mse": _finite_float_or_none(s1_total_mse),
        "s1_total_penalty": _finite_float_or_none(s1_total_penalty),
        "learning_rate": _finite_float_or_none(learning_rate),
        "min_loss": _finite_float_or_none(min_loss),
        "min_loss_step": None if min_loss_step is None else int(min_loss_step),
        "basis": str(args.basis),
        "xc": str(args.xc),
        "system_label": _system_label(args),
        "atom1": str(getattr(args, "atom1", "H")),
        "atom2": str(getattr(args, "atom2", "H")),
        "charge": int(getattr(args, "charge", 0)),
        "spin": int(getattr(args, "spin", 0)),
        "external_s1_total_csv": (
            None
            if getattr(args, "external_s1_total_csv", None) is None
            else str(args.external_s1_total_csv)
        ),
        "external_s1_total_column": str(
            getattr(args, "external_s1_total_column", "")
        ),
        "external_reference_label": _reference_label(args),
        "training_mode": str(args.training_mode),
        "reference_scf_backend": str(args.reference_scf_backend),
        "objective": _objective_name(args),
        "objective_kind": _resolved_objective_kind(args),
        "include_hfx_channel": bool(args.include_hfx_channel),
        "response_hf_mode": str(args.response_hf_mode),
        "include_pt2_channel": bool(args.include_pt2_channel),
        "pt2_channel_mode": (
            str(args.pt2_channel_mode) if bool(args.include_pt2_channel) else None
        ),
        "response_pt2_mode": (
            str(args.response_pt2_mode) if bool(args.include_pt2_channel) else None
        ),
        "s1_weight": float(args.s1_weight),
        "energy_mse_weight": float(args.energy_mse_weight),
        "energy_mae_weight": float(args.energy_mae_weight),
        "density_constraint_weight": float(args.density_constraint_weight),
        "s1_use_tda": bool(args.s1_use_tda),
        "eval_use_tda": resolved_eval_use_tda,
        "scf_warm_start": bool(args.scf_warm_start),
        "scf_warm_start_update_interval": int(args.scf_warm_start_update_interval),
        "recover_nonfinite_steps": bool(args.recover_nonfinite_steps),
        "scf_gradient_mode": str(args.scf_gradient_mode),
        "scf_implicit_diff_max_iter": int(args.scf_implicit_diff_max_iter),
        "scf_implicit_diff_clip": float(args.scf_implicit_diff_clip),
        "scf_implicit_diff_tolerance": float(args.scf_implicit_diff_tolerance),
        "scf_implicit_diff_regularization": float(args.scf_implicit_diff_regularization),
        "steps": int(args.steps),
        "learning_rate": float(args.learning_rate),
        "learning_rate_initial": float(args.learning_rate),
        "lr_decay_every": int(args.lr_decay_every),
        "lr_decay_factor": float(args.lr_decay_factor),
        "checkpoint_every": int(args.checkpoint_every),
        "hidden_dims": [int(value) for value in args.hidden_dims],
        "train_r_values_angstrom": [float(value) for value in train_r_values],
        "dense_points": int(args.dense_points),
    }


def _save_params_checkpoint_atomic(
    path: Path,
    params: Any,
    *,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_checkpoint, tmp_meta = save_params_checkpoint(
        tmp_path,
        params,
        metadata=metadata,
    )
    tmp_checkpoint.replace(path)
    final_meta_path = path.with_suffix(path.suffix + ".meta.json")
    if tmp_meta is None:
        final_meta_path.write_text("{}", encoding="utf-8")
    else:
        tmp_meta.replace(final_meta_path)
    return path, final_meta_path


def _save_training_checkpoint(
    args: argparse.Namespace,
    *,
    kind: str,
    params: Any,
    step: int,
    parameter_state: str,
    loss: float | None = None,
    s1_total_mae: float | None = None,
    s1_total_mse: float | None = None,
    s1_total_penalty: float | None = None,
    learning_rate: float | None = None,
    min_loss: float | None = None,
    min_loss_step: int | None = None,
) -> tuple[Path, Path]:
    path = Path(args.outdir) / f"neural_xc_params_{kind}.msgpack"
    metadata = _training_checkpoint_metadata(
        args,
        checkpoint_kind=kind,
        parameter_state=parameter_state,
        step=step,
        loss=loss,
        s1_total_mae=s1_total_mae,
        s1_total_mse=s1_total_mse,
        s1_total_penalty=s1_total_penalty,
        learning_rate=learning_rate,
        min_loss=min_loss,
        min_loss_step=min_loss_step,
    )
    return _save_params_checkpoint_atomic(path, params, metadata=metadata)


def _make_s1_functional(args: argparse.Namespace) -> Any:
    return neural_xc.Functional(
        architecture=str(args.network_architecture),
        semilocal_xc=tuple(str(name) for name in args.semilocal_xc),
        hidden_dims=tuple(int(value) for value in args.hidden_dims),
        input_feature_mode=str(args.input_feature_mode),
        include_hfx_channel=bool(args.include_hfx_channel),
        include_pt2_channel=bool(args.include_pt2_channel),
        pt2_channel_mode=str(args.pt2_channel_mode),
        response_hf_mode=str(args.response_hf_mode),
        response_pt2_mode=str(args.response_pt2_mode),
        name=f"neural_xc_h2_s1_tda_{str(args.training_mode)}",
    )


def _reference_s1_total_energy_h(point: Any) -> float:
    total_energies = np.asarray(getattr(point, "fci_total_energies_h", ()), dtype=np.float64)
    if int(total_energies.size) > 1:
        return float(total_energies[1])
    if int(np.asarray(point.fci_excitation_energies_h).size) < 1:
        raise ValueError("Reference point must provide at least one FCI excitation energy.")
    return float(point.fci_energy_h) + float(point.fci_excitation_energies_h[0])


def _load_external_s1_total_curve(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray] | None:
    csv_path = getattr(args, "external_s1_total_csv", None)
    if not csv_path:
        return None
    path = Path(str(csv_path))
    r_column = str(getattr(args, "external_r_column", "r_angstrom"))
    energy_column = str(getattr(args, "external_s1_total_column", "mr_ccca_hartree"))
    rows: list[tuple[float, float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"External target CSV {path} has no header.")
        missing = [name for name in (r_column, energy_column) if name not in reader.fieldnames]
        if missing:
            raise KeyError(f"External target CSV {path} is missing columns: {missing}")
        for row in reader:
            raw_r = str(row.get(r_column, "")).strip()
            raw_e = str(row.get(energy_column, "")).strip()
            if not raw_r or not raw_e:
                continue
            rows.append((float(raw_r), float(raw_e)))
    if len(rows) < 2:
        raise ValueError(
            f"External target CSV {path} needs at least two finite rows in {r_column}/{energy_column}."
        )
    rows = sorted(rows)
    return (
        np.asarray([row[0] for row in rows], dtype=np.float64),
        np.asarray([row[1] for row in rows], dtype=np.float64),
    )


def _external_s1_total_at_r(
    curve: tuple[np.ndarray, np.ndarray] | None,
    r_angstrom: float,
) -> float | None:
    if curve is None:
        return None
    r_values, e_values = curve
    r = float(r_angstrom)
    r_min = float(np.min(r_values))
    r_max = float(np.max(r_values))
    if r < r_min - 1e-10 or r > r_max + 1e-10:
        raise ValueError(
            f"Requested R={r:.8g} A lies outside external target range "
            f"[{r_min:.8g}, {r_max:.8g}] A."
        )
    return float(np.interp(r, r_values, e_values))


def _system_label(args: argparse.Namespace) -> str:
    return str(getattr(args, "system_label", "H2") or "H2")


def _reference_label(args: argparse.Namespace) -> str:
    explicit = getattr(args, "external_reference_label", None)
    if explicit:
        return str(explicit)
    if getattr(args, "external_s1_total_csv", None):
        return str(getattr(args, "external_s1_total_column", "external E1"))
    return "FCI"


def build_s1_training_data(
    points: list[Any],
    *,
    s1_weight: float,
    density_constraint_weight: float,
) -> tuple[GroundStateDatum, ...]:
    data: list[GroundStateDatum] = []
    for point in points:
        if int(np.asarray(point.fci_excitation_energies_h).size) < 1:
            raise ValueError("Every training point must provide at least one FCI excitation energy.")
        target_first_excited_total_energy = _reference_s1_total_energy_h(point)
        data.append(
            GroundStateDatum.from_parts(
                point.molecule,
                core=GroundStateCoreDatum(
                    target_total_energy=jnp.asarray(point.fci_energy_h, dtype=jnp.float64),
                    target_density_matrix=jnp.asarray(point.fci_density_matrix, dtype=jnp.float64),
                    density_constraint_weight=float(density_constraint_weight),
                ),
                excited_state=ExcitedStateDatum(
                    target_first_excited_total_energy=jnp.asarray(
                        target_first_excited_total_energy,
                        dtype=jnp.float64,
                    ),
                    first_excited_total_energy_constraint_weight=float(s1_weight),
                ),
            )
        )
    return tuple(data)


def _write_reference_point(group: Any, point: Any) -> None:
    group.attrs["r_angstrom"] = float(point.r_angstrom)
    group.attrs["atom"] = str(point.atom)
    group.attrs["fci_energy_h"] = float(point.fci_energy_h)
    group.attrs["fci_electron_count"] = float(point.fci_electron_count)
    for name in (
        "fci_total_energies_h",
        "fci_excitation_energies_h",
        "fci_density_grid",
        "fci_density_matrix",
    ):
        if name in group:
            del group[name]
        group.create_dataset(name, data=np.asarray(getattr(point, name)), compression="gzip")
    write_restricted_molecule(group.require_group("molecule"), point.molecule)


def _read_reference_point(group: Any) -> Any:
    return _HELPERS.ReferencePoint(
        r_angstrom=float(group.attrs["r_angstrom"]),
        atom=str(group.attrs["atom"]),
        molecule=read_restricted_molecule(group["molecule"]),
        fci_energy_h=float(group.attrs["fci_energy_h"]),
        fci_total_energies_h=np.asarray(group["fci_total_energies_h"][()], dtype=np.float64),
        fci_excitation_energies_h=np.asarray(
            group["fci_excitation_energies_h"][()],
            dtype=np.float64,
        ),
        fci_density_grid=np.asarray(group["fci_density_grid"][()], dtype=np.float64),
        fci_density_matrix=np.asarray(group["fci_density_matrix"][()], dtype=np.float64),
        fci_electron_count=float(group.attrs["fci_electron_count"]),
    )


def _cache_group_name(label: str, args: argparse.Namespace) -> str:
    pt2 = "pt2" if bool(args.include_pt2_channel) else "nopt2"
    npoints = int(args.train_points if label == "train" else args.dense_points)
    target_csv = getattr(args, "external_s1_total_csv", None)
    target_tag = (
        "internal_fci"
        if not target_csv
        else f"external={Path(str(target_csv)).stem}/col={str(getattr(args, 'external_s1_total_column', ''))}"
    )
    return (
        f"{label}/basis={str(args.basis).replace('/', '_')}/"
        f"system={_system_label(args)}/"
        f"atoms={str(getattr(args, 'atom1', 'H'))}-{str(getattr(args, 'atom2', 'H'))}/"
        f"charge={int(getattr(args, 'charge', 0))}/"
        f"spin={int(getattr(args, 'spin', 0))}/"
        f"grid={int(args.grids_level)}/"
        f"r={float(args.r_min):.8g}-{float(args.r_max):.8g}/"
        f"n={npoints}/{pt2}/"
        f"pt2mode={str(args.pt2_channel_mode) if bool(args.include_pt2_channel) else 'none'}/"
        f"target={target_tag}/"
        "density=dm-v2"
    )


def _save_reference_points_hdf5(path: Path, group_name: str, points: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as handle:
        if group_name in handle:
            del handle[group_name]
        group = handle.create_group(group_name)
        group.attrs["count"] = int(len(points))
        for idx, point in enumerate(points):
            _write_reference_point(group.create_group(f"point_{idx:04d}"), point)


def _load_reference_points_hdf5(path: Path, group_name: str) -> list[Any]:
    with h5py.File(path, "r") as handle:
        group = handle[group_name]
        count = int(group.attrs["count"])
        return [_read_reference_point(group[f"point_{idx:04d}"]) for idx in range(count)]


def _has_hdf5_group(path: Path, group_name: str) -> bool:
    if not path.exists():
        return False
    with h5py.File(path, "r") as handle:
        return group_name in handle


def _build_reference_curve_for_s1(
    r_values: np.ndarray,
    *,
    args: argparse.Namespace,
    logger: RunLogger,
    label: str,
) -> list[Any]:
    external_curve = _load_external_s1_total_curve(args)
    if external_curve is None:
        return build_reference_curve(r_values, args=args, logger=logger, label=label)
    points: list[Any] = []
    rhf_dm0 = None
    t0 = time.perf_counter()
    for idx, r_val in enumerate(r_values, start=1):
        target_e1 = _external_s1_total_at_r(external_curve, float(r_val))
        point, rhf_dm0 = _HELPERS.build_reference_point(
            float(r_val),
            atom1=str(getattr(args, "atom1", "H")),
            atom2=str(getattr(args, "atom2", "H")),
            charge=int(getattr(args, "charge", 0)),
            spin=int(getattr(args, "spin", 0)),
            basis=str(args.basis),
            xc=str(args.xc),
            grids_level=int(args.grids_level),
            max_l=int(args.max_l),
            grid_ao_backend=str(args.grid_ao_backend),
            integral_backend=str(args.integral_backend),
            jk_backend=str(args.jk_backend),
            df_tol=float(args.df_tol),
            df_max_rank=args.df_max_rank,
            reference_scf_max_cycle=int(args.reference_scf_max_cycle),
            reference_scf_conv_tol=float(args.reference_scf_conv_tol),
            reference_scf_conv_tol_density=float(args.reference_scf_conv_tol_density),
            reference_scf_damping=float(args.reference_scf_damping),
            reference_scf_potential_clip=float(args.reference_scf_potential_clip),
            excited_nstates=int(args.excited_nstates),
            fci_dm0=rhf_dm0,
            compute_local_hfx_features=(str(args.input_feature_mode) == "canonical"),
            compute_local_pt2_features=bool(getattr(args, "include_pt2_channel", False)),
            reference_scf_backend=str(getattr(args, "reference_scf_backend", "pyscf")),
            external_s1_total_energy_h=target_e1,
        )
        s1_total = _reference_s1_total_energy_h(point)
        s1_gap = (
            float(point.fci_excitation_energies_h[0])
            if point.fci_excitation_energies_h.size > 0
            else float("nan")
        )
        points.append(point)
        logger.log(
            f"[{label}] {idx:3d}/{len(r_values):3d} "
            f"R={point.r_angstrom:.4f} A "
            f"E0_ref={point.fci_energy_h:.10f} Eh "
            f"E1_target={s1_total:.10f} Eh "
            f"pseudo_gap={s1_gap:.10f} Eh "
            f"grid_n={int(np.asarray(point.molecule.grid.weights).size)}"
        )
    logger.log(f"[{label}] done in {time.perf_counter() - t0:.2f} s")
    return points


def _get_or_build_reference_curve(
    r_values: np.ndarray,
    *,
    args: argparse.Namespace,
    logger: RunLogger,
    label: str,
) -> list[Any]:
    cache_arg = getattr(args, "reference_cache", None)
    cache_label = "train" if label.startswith("train") else "dense"
    if cache_arg is None:
        return _build_reference_curve_for_s1(r_values, args=args, logger=logger, label=label)
    cache_path = Path(str(cache_arg))
    group_name = _cache_group_name(cache_label, args)
    if _has_hdf5_group(cache_path, group_name) and not bool(args.rebuild_reference_cache):
        logger.log(f"[{label}] loading cached references from {cache_path}:{group_name}")
        points = _load_reference_points_hdf5(cache_path, group_name)
        logger.log(f"[{label}] loaded {len(points)} cached references")
        return points
    points = _build_reference_curve_for_s1(r_values, args=args, logger=logger, label=label)
    logger.log(f"[{label}] writing cached references to {cache_path}:{group_name}")
    _save_reference_points_hdf5(cache_path, group_name, points)
    return points


def _tree_add(left: Any | None, right: Any) -> Any:
    if left is None:
        return right
    return jax.tree_util.tree_map(lambda a, b: a + b, left, right)


def _tree_scale(tree: Any, scale: float) -> Any:
    return jax.tree_util.tree_map(lambda value: value * scale, tree)


def _streaming_average_eval(
    params: Any,
    data: tuple[GroundStateDatum, ...],
    eval_kernel: Any,
) -> tuple[float, dict[str, float]]:
    losses: list[float] = []
    s1_penalties: list[float] = []
    s1_maes: list[float] = []
    s1_mses: list[float] = []
    for datum in data:
        loss, metrics = eval_kernel(params, datum)
        losses.append(float(loss))
        s1_penalties.append(_s1_total_supervision_penalty(metrics))
        s1_maes.append(_s1_total_supervision_mae(metrics))
        s1_mses.append(_s1_total_supervision_mse(metrics))
    return float(np.mean(losses)), {
        "s1_penalty": float(np.mean(s1_penalties)),
        "s1_mae": float(np.mean(s1_maes)),
        "s1_mse": float(np.mean(s1_mses)),
    }


def _self_consistent_prediction_config(args: argparse.Namespace) -> GroundStateTrainingConfig:
    return GroundStateTrainingConfig(
        mode="self_consistent",
        energy_mse_weight=0.0,
        energy_mae_weight=0.0,
        scf_max_cycle=_HELPERS._resolve_train_scf_max_cycle(args.train_scf_max_cycle),
        scf_damping=float(args.train_scf_damping),
        scf_conv_tol_energy=args.train_scf_conv_tol_energy,
        scf_convergence_metric=str(args.train_scf_convergence_metric),
        scf_conv_tol_density=float(args.train_scf_conv_tol_density),
        scf_vxc_clip=float(args.train_scf_vxc_clip),
        scf_iterate_selection=str(args.scf_iterate_selection),
        scf_require_convergence=bool(args.scf_require_convergence),
        scf_gradient_mode=str(args.scf_gradient_mode),
        scf_implicit_diff_max_iter=int(args.scf_implicit_diff_max_iter),
        scf_implicit_diff_clip=float(args.scf_implicit_diff_clip),
        scf_implicit_diff_tolerance=float(args.scf_implicit_diff_tolerance),
        scf_implicit_diff_regularization=float(args.scf_implicit_diff_regularization),
    )


def _rebuild_points_with_fixed_density_checkpoint(
    points: list[Any],
    *,
    args: argparse.Namespace,
    logger: Any,
    label: str,
    checkpoint_path: str | None,
) -> list[Any]:
    if not points or checkpoint_path is None:
        return points
    functional = _make_s1_functional(args)
    template = functional.init_from_molecule(
        jax.random.PRNGKey(int(args.seed)),
        points[0].molecule,
    )
    params = load_params_checkpoint(Path(str(checkpoint_path)), template=template)
    prediction_config = _self_consistent_prediction_config(args)
    rebuilt: list[Any] = []
    t0 = time.perf_counter()
    logger.log(
        f"[{label}] rebuilding {len(points)} fixed-density molecules from checkpoint {checkpoint_path}"
    )
    for idx, point in enumerate(points, start=1):
        predicted_molecule = predict_ground_state_molecule(
            params,
            functional,
            point.molecule,
            training_config=prediction_config,
        )
        rebuilt.append(replace(point, molecule=predicted_molecule))
        logger.log(
            f"[{label}] {idx:3d}/{len(points):3d} "
            f"R={float(point.r_angstrom):.4f} A "
            "fixed density rebuilt"
        )
    logger.log(f"[{label}] done in {time.perf_counter() - t0:.2f} s")
    return rebuilt


def _train_functional_streaming(
    train_points: list[Any],
    training_data: tuple[GroundStateDatum, ...],
    *,
    args: argparse.Namespace,
    logger: RunLogger,
    functional: Any,
    gs_training: GroundStateTrainingConfig,
    optimizer: optax.GradientTransformation,
) -> dict[str, Any]:
    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(int(args.seed)),
        train_points[0].molecule,
        optimizer,
    )
    if args.init_checkpoint:
        init_checkpoint = Path(str(args.init_checkpoint))
        state = state.replace(
            params=load_params_checkpoint(init_checkpoint, template=state.params)
        )
        logger.log(f"[train_init] loaded params from checkpoint: {init_checkpoint}")

    loss_and_grad_kernel = make_ground_state_loss_and_grad(
        functional,
        training_config=gs_training,
    )
    eval_kernel = lambda params, datum: ground_state_mse_loss(  # noqa: E731
        params,
        functional,
        datum,
        training_config=gs_training,
    )
    if bool(args.jit_train):
        loss_and_grad_kernel = jax.jit(loss_and_grad_kernel)
    if bool(args.jit_eval):
        eval_kernel = jax.jit(eval_kernel)

    lr_schedule = (
        optax.exponential_decay(
            init_value=float(args.learning_rate),
            transition_steps=int(args.lr_decay_every),
            decay_rate=float(args.lr_decay_factor),
            staircase=True,
        )
        if int(args.lr_decay_every) > 0
        else None
    )
    logger.log("[train_init] streaming mode: one geometry per JIT call; averaging grads")
    if bool(args.skip_initial_eval):
        initial_loss_val = float("nan")
        initial_metrics = {"s1_penalty": 0.0, "s1_mae": 0.0, "s1_mse": 0.0}
        min_loss = float("inf")
        logger.log("[train_init] skipped initial eval")
    else:
        initial_loss_val, initial_metrics = _streaming_average_eval(
            state.params,
            training_data,
            eval_kernel,
        )
        min_loss = initial_loss_val
    min_loss_step = 0
    best_params = state.params
    history_steps = [0]
    loss_history = [initial_loss_val]
    s1_penalty_history = [initial_metrics["s1_penalty"]]
    s1_mae_history = [initial_metrics["s1_mae"]]
    s1_mse_history = [initial_metrics["s1_mse"]]
    grad_norm_history = [float("nan")]
    grad_abs_max_history = [float("nan")]
    param_update_norm_history = [float("nan")]
    nonfinite_grad_fraction_history = [0.0]
    eval_steps = [0]
    eval_loss_history = [initial_loss_val]
    eval_s1_mae_history = [initial_metrics["s1_mae"]]
    logger.log(
        "[train] "
        f"steps={int(args.steps)} "
        f"lr={float(args.learning_rate):.6g} "
        f"mode={str(args.training_mode)} "
        f"objective={_objective_name(args)} "
        f"s1_weight={float(args.s1_weight):.6g} "
        "train_step_mode=stream_single_geometry"
    )
    initial_lr = (
        float(lr_schedule(0))
        if lr_schedule is not None
        else float(args.learning_rate)
    )
    latest_checkpoint_path, latest_checkpoint_meta_path = _save_training_checkpoint(
        args,
        kind="latest",
        params=state.params,
        step=0,
        parameter_state="initial",
        loss=initial_loss_val,
        s1_total_mae=initial_metrics["s1_mae"],
        s1_total_mse=initial_metrics["s1_mse"],
        s1_total_penalty=initial_metrics["s1_penalty"],
        learning_rate=initial_lr,
        min_loss=min_loss,
        min_loss_step=min_loss_step,
    )
    best_checkpoint_path: Path | None = None
    best_checkpoint_meta_path: Path | None = None
    if np.isfinite(initial_loss_val):
        best_checkpoint_path, best_checkpoint_meta_path = _save_training_checkpoint(
            args,
            kind="best",
            params=best_params,
            step=0,
            parameter_state="initial",
            loss=min_loss,
            s1_total_mae=initial_metrics["s1_mae"],
            s1_total_mse=initial_metrics["s1_mse"],
            s1_total_penalty=initial_metrics["s1_penalty"],
            learning_rate=initial_lr,
            min_loss=min_loss,
            min_loss_step=min_loss_step,
        )

    t0 = time.perf_counter()
    for step in range(1, int(args.steps) + 1):
        params_for_loss = state.params
        grad_sum = None
        losses: list[float] = []
        s1_penalties: list[float] = []
        s1_maes: list[float] = []
        s1_mses: list[float] = []
        grad_norms: list[float] = []
        grad_abs_maxes: list[float] = []
        nonfinite_fracs: list[float] = []
        for datum in training_data:
            loss, metrics, grads = loss_and_grad_kernel(params_for_loss, datum)
            losses.append(float(loss))
            s1_penalties.append(_s1_total_supervision_penalty(metrics))
            s1_maes.append(_s1_total_supervision_mae(metrics))
            s1_mses.append(_s1_total_supervision_mse(metrics))
            grad_norms.append(_metric_scalar(metrics, "grad_norm"))
            grad_abs_maxes.append(_metric_scalar(metrics, "grad_abs_max"))
            nonfinite_fracs.append(_metric_scalar(metrics, "nonfinite_grad_fraction", 0.0))
            grad_sum = _tree_add(grad_sum, grads)
        grads_avg = _tree_scale(grad_sum, 1.0 / max(1, len(training_data)))
        train_loss_val = float(np.mean(losses))
        train_s1_penalty_val = float(np.mean(s1_penalties))
        train_s1_mae_val = float(np.mean(s1_maes))
        train_s1_mse_val = float(np.mean(s1_mses))
        grad_norm_val = float(np.mean(grad_norms))
        grad_abs_max_val = float(np.max(grad_abs_maxes))
        nonfinite_grad_fraction_val = float(np.mean(nonfinite_fracs))
        best_updated = False
        if train_loss_val < min_loss:
            min_loss = train_loss_val
            min_loss_step = step
            best_params = params_for_loss
            best_updated = True

        prev_state = state
        state = state.apply_gradients(grads=grads_avg)
        param_delta = jax.tree_util.tree_map(
            lambda new, old: new - old,
            state.params,
            prev_state.params,
        )
        reverted_step = False
        if not _tree_all_finite(state.params):
            state = prev_state
            reverted_step = True
            logger.log(f"[train] non-finite params detected at step {step}; reverted update")

        param_update_norm_val = 0.0 if reverted_step else float(_tree_l2_norm(param_delta))

        history_steps.append(step)
        loss_history.append(train_loss_val)
        s1_penalty_history.append(train_s1_penalty_val)
        s1_mae_history.append(train_s1_mae_val)
        s1_mse_history.append(train_s1_mse_val)
        grad_norm_history.append(grad_norm_val)
        grad_abs_max_history.append(grad_abs_max_val)
        param_update_norm_history.append(param_update_norm_val)
        nonfinite_grad_fraction_history.append(nonfinite_grad_fraction_val)
        current_lr = (
            float(lr_schedule(step - 1))
            if lr_schedule is not None
            else float(args.learning_rate)
        )
        if best_updated:
            best_checkpoint_path, best_checkpoint_meta_path = _save_training_checkpoint(
                args,
                kind="best",
                params=best_params,
                step=min_loss_step,
                parameter_state="pre_update",
                loss=min_loss,
                s1_total_mae=train_s1_mae_val,
                s1_total_mse=train_s1_mse_val,
                s1_total_penalty=train_s1_penalty_val,
                learning_rate=current_lr,
                min_loss=min_loss,
                min_loss_step=min_loss_step,
            )
        if (
            int(args.checkpoint_every) > 0
            and (step == 1 or step % int(args.checkpoint_every) == 0 or step == int(args.steps))
        ):
            latest_checkpoint_path, latest_checkpoint_meta_path = _save_training_checkpoint(
                args,
                kind="latest",
                params=state.params,
                step=step,
                parameter_state="post_update",
                loss=train_loss_val,
                s1_total_mae=train_s1_mae_val,
                s1_total_mse=train_s1_mse_val,
                s1_total_penalty=train_s1_penalty_val,
                learning_rate=current_lr,
                min_loss=min_loss,
                min_loss_step=min_loss_step,
            )
        if step == 1 or step % 10 == 0 or step == int(args.steps):
            eval_steps.append(step)
            eval_loss_history.append(train_loss_val)
            eval_s1_mae_history.append(train_s1_mae_val)
            logger.log(
                "[train] "
                f"step={step:4d}/{int(args.steps):4d} "
                f"loss={train_loss_val:.8e} "
                f"s1_total_mae={train_s1_mae_val:.8e} "
                f"grad_norm={grad_norm_val:.8e} "
                f"grad_abs_max={grad_abs_max_val:.8e} "
                f"update_norm={param_update_norm_val:.8e} "
                f"lr={current_lr:.8e}"
            )

    elapsed_s = time.perf_counter() - t0
    final_loss_val, final_metrics = _streaming_average_eval(
        state.params,
        training_data,
        eval_kernel,
    )
    if final_loss_val < min_loss:
        min_loss = final_loss_val
        min_loss_step = int(args.steps)
        best_params = state.params
        best_checkpoint_path, best_checkpoint_meta_path = _save_training_checkpoint(
            args,
            kind="best",
            params=best_params,
            step=min_loss_step,
            parameter_state="final",
            loss=min_loss,
            s1_total_mae=final_metrics["s1_mae"],
            s1_total_mse=final_metrics["s1_mse"],
            s1_total_penalty=final_metrics["s1_penalty"],
            learning_rate=(
                float(lr_schedule(max(0, int(args.steps) - 1)))
                if lr_schedule is not None
                else float(args.learning_rate)
            ),
            min_loss=min_loss,
            min_loss_step=min_loss_step,
        )
    loss_history[-1] = final_loss_val
    s1_penalty_history[-1] = final_metrics["s1_penalty"]
    s1_mae_history[-1] = final_metrics["s1_mae"]
    s1_mse_history[-1] = final_metrics["s1_mse"]
    final_lr = (
        float(lr_schedule(max(0, int(args.steps) - 1)))
        if lr_schedule is not None
        else float(args.learning_rate)
    )
    final_checkpoint_path, final_checkpoint_meta_path = _save_training_checkpoint(
        args,
        kind="final",
        params=state.params,
        step=int(args.steps),
        parameter_state="final",
        loss=final_loss_val,
        s1_total_mae=final_metrics["s1_mae"],
        s1_total_mse=final_metrics["s1_mse"],
        s1_total_penalty=final_metrics["s1_penalty"],
        learning_rate=final_lr,
        min_loss=min_loss,
        min_loss_step=min_loss_step,
    )
    latest_checkpoint_path, latest_checkpoint_meta_path = _save_training_checkpoint(
        args,
        kind="latest",
        params=state.params,
        step=int(args.steps),
        parameter_state="final",
        loss=final_loss_val,
        s1_total_mae=final_metrics["s1_mae"],
        s1_total_mse=final_metrics["s1_mse"],
        s1_total_penalty=final_metrics["s1_penalty"],
        learning_rate=final_lr,
        min_loss=min_loss,
        min_loss_step=min_loss_step,
    )
    logger.log(
        "[train] done "
        f"final_loss={final_loss_val:.8e} "
        f"min_loss={min_loss:.8e}@{min_loss_step} "
        f"elapsed_s={elapsed_s:.2f}"
    )
    return {
        "functional": functional,
        "training_config": gs_training,
        "best_params": best_params,
        "final_loss": final_loss_val,
        "min_loss": min_loss,
        "min_loss_step": min_loss_step,
        "elapsed_s": elapsed_s,
        "history_steps": history_steps,
        "loss_history": loss_history,
        "s1_penalty_history": s1_penalty_history,
        "s1_mae_history": s1_mae_history,
        "s1_mse_history": s1_mse_history,
        "grad_norm_history": grad_norm_history,
        "grad_abs_max_history": grad_abs_max_history,
        "param_update_norm_history": param_update_norm_history,
        "nonfinite_grad_fraction_history": nonfinite_grad_fraction_history,
        "eval_steps": eval_steps,
        "eval_loss_history": eval_loss_history,
        "eval_s1_mae_history": eval_s1_mae_history,
        "latest_checkpoint": str(latest_checkpoint_path),
        "latest_checkpoint_meta": str(latest_checkpoint_meta_path),
        "best_checkpoint": None if best_checkpoint_path is None else str(best_checkpoint_path),
        "best_checkpoint_meta": (
            None if best_checkpoint_meta_path is None else str(best_checkpoint_meta_path)
        ),
        "final_checkpoint": str(final_checkpoint_path),
        "final_checkpoint_meta": str(final_checkpoint_meta_path),
    }


def train_functional(
    train_points: list[Any],
    *,
    args: argparse.Namespace,
    logger: RunLogger,
) -> dict[str, Any]:
    if not train_points:
        raise ValueError("train_points must not be empty.")

    training_data = build_s1_training_data(
        train_points,
        s1_weight=float(args.s1_weight),
        density_constraint_weight=float(args.density_constraint_weight),
    )
    functional = _make_s1_functional(args)
    coefficient_prior = neural_xc.resolve_coefficient_prior_values(
        tuple(str(name) for name in args.semilocal_xc)
    )
    if coefficient_prior is not None and bool(args.include_pt2_channel):
        n_semilocal = len(tuple(str(name) for name in args.semilocal_xc))
        if len(coefficient_prior) == n_semilocal + 1:
            coefficient_prior = (
                tuple(coefficient_prior[:n_semilocal])
                + (0.0,)
                + tuple(coefficient_prior[n_semilocal:])
            )
    logger.log(
        "[init] coefficient_prior="
        f"{None if coefficient_prior is None else tuple(float(x) for x in coefficient_prior)} "
        f"include_pt2_channel={bool(args.include_pt2_channel)}"
    )
    gs_training = GroundStateTrainingConfig(
        mode=str(args.training_mode),
        energy_mse_weight=float(args.energy_mse_weight),
        energy_mae_weight=float(args.energy_mae_weight),
        s1_constraint_use_tda=bool(args.s1_use_tda),
        scf_max_cycle=_HELPERS._resolve_train_scf_max_cycle(args.train_scf_max_cycle),
        scf_damping=float(args.train_scf_damping),
        scf_conv_tol_energy=args.train_scf_conv_tol_energy,
        scf_convergence_metric=str(args.train_scf_convergence_metric),
        scf_conv_tol_density=float(args.train_scf_conv_tol_density),
        scf_vxc_clip=float(args.train_scf_vxc_clip),
        scf_iterate_selection=str(args.scf_iterate_selection),
        scf_require_convergence=bool(args.scf_require_convergence),
        scf_gradient_mode=str(args.scf_gradient_mode),
        scf_implicit_diff_max_iter=int(args.scf_implicit_diff_max_iter),
        scf_implicit_diff_clip=float(args.scf_implicit_diff_clip),
        scf_implicit_diff_tolerance=float(args.scf_implicit_diff_tolerance),
        scf_implicit_diff_regularization=float(args.scf_implicit_diff_regularization),
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

    if args.grad_clip_norm is not None and float(args.grad_clip_norm) > 0.0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(float(args.grad_clip_norm)),
            base_optimizer,
        )
    else:
        optimizer = base_optimizer

    if bool(args.stream_train):
        return _train_functional_streaming(
            train_points,
            training_data,
            args=args,
            logger=logger,
            functional=functional,
            gs_training=gs_training,
            optimizer=optimizer,
        )

    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(int(args.seed)),
        train_points[0].molecule,
        optimizer,
    )
    if args.init_checkpoint:
        init_checkpoint = Path(str(args.init_checkpoint))
        state = state.replace(
            params=load_params_checkpoint(init_checkpoint, template=state.params)
        )
        logger.log(f"[train_init] loaded params from checkpoint: {init_checkpoint}")
    train_step = make_ground_state_train_step(functional, training_config=gs_training)
    loss_and_grad_kernel = make_ground_state_loss_and_grad(
        functional,
        training_config=gs_training,
    )
    use_warm_start = bool(args.scf_warm_start) and gs_training.mode == "self_consistent"
    use_split_jit_train = (
        bool(args.jit_train)
        and gs_training.mode == "self_consistent"
        and str(args.scf_gradient_mode) == "implicit_commutator"
    )
    train_dataset_current = training_data
    def make_eval_fn(dataset):
        return lambda params: ground_state_mse_loss(  # noqa: E731
            params,
            functional,
            dataset,
            training_config=gs_training,
        )

    def make_train_step_fn(dataset):
        return lambda current_state: train_step(current_state, dataset)  # noqa: E731

    def make_train_kernel_fn(dataset):
        return lambda params: loss_and_grad_kernel(params, dataset)  # noqa: E731

    logger.log("[train_init] building compiled_eval")
    eval_build_t0 = time.perf_counter()
    compiled_eval = (
        jax.jit(make_eval_fn(train_dataset_current))
        if bool(args.jit_eval)
        else make_eval_fn(train_dataset_current)
    )
    logger.log(
        f"[train_init] built compiled_eval in {time.perf_counter() - eval_build_t0:.2f} s"
    )
    if use_split_jit_train:
        logger.log("[train_init] building compiled_train_kernel")
        train_build_t0 = time.perf_counter()
        eager_train_kernel = make_train_kernel_fn(train_dataset_current)
        compiled_train_kernel = (
            jax.jit(eager_train_kernel) if use_split_jit_train else eager_train_kernel
        )
        compiled_train_step = None
        logger.log(
            f"[train_init] built compiled_train_kernel wrapper in {time.perf_counter() - train_build_t0:.2f} s"
        )
    else:
        logger.log("[train_init] building compiled_train_step")
        train_build_t0 = time.perf_counter()
        eager_train_step = make_train_step_fn(train_dataset_current)
        compiled_train_step = (
            jax.jit(eager_train_step) if bool(args.jit_train) else eager_train_step
        )
        compiled_train_kernel = None
        logger.log(
            f"[train_init] built compiled_train_step wrapper in {time.perf_counter() - train_build_t0:.2f} s"
        )
    train_step_mode = "eager"
    if use_split_jit_train:
        candidate_train_kernel = compiled_train_kernel
        try:
            logger.log("[train_init] starting jit compilation for train_kernel")
            jit_compile_t0 = time.perf_counter()
            _ = candidate_train_kernel.lower(state.params).compile()
            compiled_train_kernel = candidate_train_kernel
            train_step_mode = "jit_loss_only"
            logger.log(
                f"[train_init] finished jit compilation for train_kernel in {time.perf_counter() - jit_compile_t0:.2f} s"
            )
        except Exception as exc:  # pragma: no cover - best effort runtime path
            logger.log(f"[train] jit compilation failed for objective train kernel: {exc!r}")
            compiled_train_kernel = make_train_kernel_fn(train_dataset_current)
    elif bool(args.jit_train):
        candidate_train_step = compiled_train_step
        try:
            logger.log("[train_init] starting jit compilation for train_step")
            jit_compile_t0 = time.perf_counter()
            _ = candidate_train_step.lower(state).compile()
            compiled_train_step = candidate_train_step
            train_step_mode = "jit"
            logger.log(
                f"[train_init] finished jit compilation for train_step in {time.perf_counter() - jit_compile_t0:.2f} s"
            )
        except Exception as exc:  # pragma: no cover - best effort runtime path
            logger.log(f"[train] jit compilation failed for objective train step: {exc!r}")
            compiled_train_step = make_train_step_fn(train_dataset_current)

    logger.log("[train_init] evaluating initial loss")
    initial_eval_t0 = time.perf_counter()
    initial_loss, initial_metrics = compiled_eval(state.params)
    logger.log(
        f"[train_init] evaluated initial loss in {time.perf_counter() - initial_eval_t0:.2f} s"
    )
    if use_warm_start:
        train_dataset_current = _refresh_dataset_scf_warm_start_cache(
            train_dataset_current,
            params=state.params,
            functional=functional,
            training_config=gs_training,
        )
        compiled_eval = (
            jax.jit(make_eval_fn(train_dataset_current))
            if bool(args.jit_eval)
            else make_eval_fn(train_dataset_current)
        )
        if use_split_jit_train:
            compiled_train_kernel = (
                jax.jit(make_train_kernel_fn(train_dataset_current))
                if train_step_mode == "jit_loss_only"
                else make_train_kernel_fn(train_dataset_current)
            )
        else:
            compiled_train_step = (
                jax.jit(make_train_step_fn(train_dataset_current))
                if train_step_mode == "jit"
                else make_train_step_fn(train_dataset_current)
            )
    initial_loss_val = float(initial_loss)
    min_loss = initial_loss_val
    min_loss_step = 0
    best_params = state.params

    history_steps = [0]
    loss_history = [initial_loss_val]
    s1_penalty_history = [_s1_total_supervision_penalty(initial_metrics)]
    s1_mae_history = [_s1_total_supervision_mae(initial_metrics)]
    s1_mse_history = [_s1_total_supervision_mse(initial_metrics)]
    grad_norm_history = [float("nan")]
    grad_abs_max_history = [float("nan")]
    param_update_norm_history = [float("nan")]
    nonfinite_grad_fraction_history = [0.0]
    eval_steps = [0]
    eval_loss_history = [initial_loss_val]
    eval_s1_mae_history = [_s1_total_supervision_mae(initial_metrics)]

    initial_lr = (
        float(lr_schedule(0))
        if lr_schedule is not None
        else float(args.learning_rate)
    )
    latest_checkpoint_path, latest_checkpoint_meta_path = _save_training_checkpoint(
        args,
        kind="latest",
        params=state.params,
        step=0,
        parameter_state="initial",
        loss=initial_loss_val,
        s1_total_mae=_s1_total_supervision_mae(initial_metrics),
        s1_total_mse=_s1_total_supervision_mse(initial_metrics),
        s1_total_penalty=_s1_total_supervision_penalty(initial_metrics),
        learning_rate=initial_lr,
        min_loss=min_loss,
        min_loss_step=min_loss_step,
    )
    best_checkpoint_path, best_checkpoint_meta_path = _save_training_checkpoint(
        args,
        kind="best",
        params=best_params,
        step=0,
        parameter_state="initial",
        loss=min_loss,
        s1_total_mae=_s1_total_supervision_mae(initial_metrics),
        s1_total_mse=_s1_total_supervision_mse(initial_metrics),
        s1_total_penalty=_s1_total_supervision_penalty(initial_metrics),
        learning_rate=initial_lr,
        min_loss=min_loss,
        min_loss_step=min_loss_step,
    )

    logger.log(
        "[train] "
        f"steps={int(args.steps)} "
        f"lr={float(args.learning_rate):.6g} "
        f"mode={str(args.training_mode)} "
        f"objective={_objective_name(args)} "
        f"s1_weight={float(args.s1_weight):.6g} "
        f"train_step_mode={train_step_mode}"
    )

    t0 = time.perf_counter()
    post_update_recoveries = 0
    guard_post_update = (
        bool(args.recover_nonfinite_steps)
        and gs_training.mode == "self_consistent"
        and str(args.scf_gradient_mode) == "implicit_commutator"
    )
    for step in range(1, int(args.steps) + 1):
        prev_state = state
        prev_train_dataset_current = train_dataset_current
        if use_split_jit_train:
            _, train_metrics, grads = compiled_train_kernel(state.params)
            new_state = state.apply_gradients(grads=grads)
            param_delta = jax.tree_util.tree_map(
                lambda new, old: new - old,
                new_state.params,
                state.params,
            )
            train_metrics = dict(train_metrics)
            train_metrics["param_update_norm"] = jnp.asarray(
                [_tree_l2_norm(param_delta)],
                dtype=jnp.asarray(train_metrics["loss"]).dtype,
            )
            train_metrics["param_norm"] = jnp.asarray(
                [_tree_l2_norm(state.params)],
                dtype=jnp.asarray(train_metrics["loss"]).dtype,
            )
            state = new_state
        else:
            state, train_metrics = compiled_train_step(state)
        reverted_step = False
        if not _tree_all_finite(state.params):
            state = prev_state
            train_dataset_current = prev_train_dataset_current
            reverted_step = True
            logger.log(f"[train] non-finite params detected at step {step}; reverted update")
        elif (
            use_warm_start
            and step % max(1, int(args.scf_warm_start_update_interval)) == 0
        ):
            train_dataset_current = _refresh_dataset_scf_warm_start_cache(
                train_dataset_current,
                params=state.params,
                functional=functional,
                training_config=gs_training,
            )
            compiled_eval = (
                jax.jit(make_eval_fn(train_dataset_current))
                if bool(args.jit_eval)
                else make_eval_fn(train_dataset_current)
            )
            if use_split_jit_train:
                compiled_train_kernel = (
                    jax.jit(make_train_kernel_fn(train_dataset_current))
                    if train_step_mode == "jit_loss_only"
                    else make_train_kernel_fn(train_dataset_current)
                )
            else:
                compiled_train_step = (
                    jax.jit(make_train_step_fn(train_dataset_current))
                    if train_step_mode == "jit"
                    else make_train_step_fn(train_dataset_current)
                )
        if guard_post_update and not reverted_step:
            guarded_loss, guarded_metrics = compiled_eval(state.params)
            if not _loss_and_metrics_all_finite(guarded_loss, guarded_metrics):
                state = prev_state
                train_dataset_current = prev_train_dataset_current
                reverted_step = True
                post_update_recoveries += 1
                logger.log(
                    f"[train] non-finite post-update eval at step {step}; reverted update"
                )

        grad_norm_val = _metric_scalar(train_metrics, "grad_norm")
        grad_abs_max_val = _metric_scalar(train_metrics, "grad_abs_max")
        param_update_norm_val = (
            0.0 if reverted_step else _metric_scalar(train_metrics, "param_update_norm")
        )
        nonfinite_grad_fraction_val = _metric_scalar(train_metrics, "nonfinite_grad_fraction", 0.0)
        train_loss_val = _metric_scalar(train_metrics, "loss")
        train_s1_penalty_val = _s1_total_supervision_penalty(train_metrics)
        train_s1_mae_val = _s1_total_supervision_mae(train_metrics)
        train_s1_mse_val = _s1_total_supervision_mse(train_metrics)

        grad_norm_history.append(grad_norm_val)
        grad_abs_max_history.append(grad_abs_max_val)
        param_update_norm_history.append(param_update_norm_val)
        nonfinite_grad_fraction_history.append(nonfinite_grad_fraction_val)

        # ``train_metrics`` is evaluated on ``prev_state.params`` before the
        # optimizer update, so the loss observed during loop iteration ``step``
        # corresponds to the parameter state at ``step - 1``.
        best_updated = False
        if step >= 2:
            tracked_step = step - 1
            history_steps.append(tracked_step)
            loss_history.append(train_loss_val)
            s1_penalty_history.append(train_s1_penalty_val)
            s1_mae_history.append(train_s1_mae_val)
            s1_mse_history.append(train_s1_mse_val)
            if train_loss_val < min_loss:
                min_loss = train_loss_val
                min_loss_step = tracked_step
                best_params = prev_state.params
                best_updated = True

        current_lr = float(lr_schedule(step - 1)) if lr_schedule is not None else float(args.learning_rate)
        if best_updated:
            best_checkpoint_path, best_checkpoint_meta_path = _save_training_checkpoint(
                args,
                kind="best",
                params=best_params,
                step=min_loss_step,
                parameter_state="pre_update",
                loss=min_loss,
                s1_total_mae=train_s1_mae_val,
                s1_total_mse=train_s1_mse_val,
                s1_total_penalty=train_s1_penalty_val,
                learning_rate=current_lr,
                min_loss=min_loss,
                min_loss_step=min_loss_step,
            )
        if (
            int(args.checkpoint_every) > 0
            and (step == 1 or step % int(args.checkpoint_every) == 0 or step == int(args.steps))
        ):
            latest_checkpoint_path, latest_checkpoint_meta_path = _save_training_checkpoint(
                args,
                kind="latest",
                params=state.params,
                step=step,
                parameter_state="post_update",
                loss=train_loss_val,
                s1_total_mae=train_s1_mae_val,
                s1_total_mse=train_s1_mse_val,
                s1_total_penalty=train_s1_penalty_val,
                learning_rate=current_lr,
                min_loss=min_loss,
                min_loss_step=min_loss_step,
            )

        if step == 1 or step % 10 == 0 or step == int(args.steps):
            eval_steps.append(step)
            eval_loss_history.append(train_loss_val)
            eval_s1_mae_history.append(train_s1_mae_val)
            logger.log(
                "[train] "
                f"step={step:4d}/{int(args.steps):4d} "
                f"loss={train_loss_val:.8e} "
                f"s1_total_mae={train_s1_mae_val:.8e} "
                f"s1_total_pred_h={_s1_total_predicted_metric(train_metrics):.8e} "
                f"s1_total_target_h={_s1_total_target_metric(train_metrics):.8e} "
                f"grad_norm={grad_norm_val:.8e} "
                f"grad_abs_max={grad_abs_max_val:.8e} "
                f"update_norm={param_update_norm_val:.8e} "
                f"lr={current_lr:.8e} "
                f"recoveries={post_update_recoveries:d}"
            )

    elapsed_s = time.perf_counter() - t0
    final_loss, final_metrics = compiled_eval(state.params)
    final_loss_val = float(final_loss)
    final_step = int(args.steps)
    final_s1_penalty = _s1_total_supervision_penalty(final_metrics)
    final_s1_mae = _s1_total_supervision_mae(final_metrics)
    final_s1_mse = _s1_total_supervision_mse(final_metrics)
    if history_steps and int(history_steps[-1]) == final_step:
        loss_history[-1] = final_loss_val
        s1_penalty_history[-1] = final_s1_penalty
        s1_mae_history[-1] = final_s1_mae
        s1_mse_history[-1] = final_s1_mse
    else:
        history_steps.append(final_step)
        loss_history.append(final_loss_val)
        s1_penalty_history.append(final_s1_penalty)
        s1_mae_history.append(final_s1_mae)
        s1_mse_history.append(final_s1_mse)
    if final_loss_val < min_loss:
        min_loss = final_loss_val
        min_loss_step = final_step
        best_params = state.params
        best_checkpoint_path, best_checkpoint_meta_path = _save_training_checkpoint(
            args,
            kind="best",
            params=best_params,
            step=min_loss_step,
            parameter_state="final",
            loss=min_loss,
            s1_total_mae=final_s1_mae,
            s1_total_mse=final_s1_mse,
            s1_total_penalty=final_s1_penalty,
            learning_rate=(
                float(lr_schedule(max(0, final_step - 1)))
                if lr_schedule is not None
                else float(args.learning_rate)
            ),
            min_loss=min_loss,
            min_loss_step=min_loss_step,
        )

    final_lr = (
        float(lr_schedule(max(0, final_step - 1)))
        if lr_schedule is not None
        else float(args.learning_rate)
    )
    final_checkpoint_path, final_checkpoint_meta_path = _save_training_checkpoint(
        args,
        kind="final",
        params=state.params,
        step=final_step,
        parameter_state="final",
        loss=final_loss_val,
        s1_total_mae=final_s1_mae,
        s1_total_mse=final_s1_mse,
        s1_total_penalty=final_s1_penalty,
        learning_rate=final_lr,
        min_loss=min_loss,
        min_loss_step=min_loss_step,
    )
    latest_checkpoint_path, latest_checkpoint_meta_path = _save_training_checkpoint(
        args,
        kind="latest",
        params=state.params,
        step=final_step,
        parameter_state="final",
        loss=final_loss_val,
        s1_total_mae=final_s1_mae,
        s1_total_mse=final_s1_mse,
        s1_total_penalty=final_s1_penalty,
        learning_rate=final_lr,
        min_loss=min_loss,
        min_loss_step=min_loss_step,
    )

    logger.log(
        "[train] done "
        f"final_loss={final_loss_val:.8e} "
        f"min_loss={min_loss:.8e}@{min_loss_step} "
        f"elapsed_s={elapsed_s:.2f}"
    )

    return {
        "functional": functional,
        "training_config": gs_training,
        "best_params": best_params,
        "final_loss": final_loss_val,
        "min_loss": min_loss,
        "min_loss_step": min_loss_step,
        "elapsed_s": elapsed_s,
        "history_steps": history_steps,
        "loss_history": loss_history,
        "s1_penalty_history": s1_penalty_history,
        "s1_mae_history": s1_mae_history,
        "s1_mse_history": s1_mse_history,
        "grad_norm_history": grad_norm_history,
        "grad_abs_max_history": grad_abs_max_history,
        "param_update_norm_history": param_update_norm_history,
        "nonfinite_grad_fraction_history": nonfinite_grad_fraction_history,
        "post_update_recoveries": int(post_update_recoveries),
        "eval_steps": eval_steps,
        "eval_loss_history": eval_loss_history,
        "eval_s1_mae_history": eval_s1_mae_history,
        "latest_checkpoint": str(latest_checkpoint_path),
        "latest_checkpoint_meta": str(latest_checkpoint_meta_path),
        "best_checkpoint": str(best_checkpoint_path),
        "best_checkpoint_meta": str(best_checkpoint_meta_path),
        "final_checkpoint": str(final_checkpoint_path),
        "final_checkpoint_meta": str(final_checkpoint_meta_path),
    }


def evaluate_dense_curve_tda(
    dense_points: list[Any],
    *,
    params: Any,
    functional: Any,
    training_config: GroundStateTrainingConfig,
    logger: RunLogger,
    use_tda: bool,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    excited_rows: list[dict[str, float]] = []
    predictor = make_ground_state_predictor(
        functional,
        training_config=training_config,
    )
    t0 = time.perf_counter()
    for idx, point in enumerate(dense_points, start=1):
        predicted_energy_h_arr, predicted_molecule = predictor(params, point.molecule)
        predicted_energy_h = float(predicted_energy_h_arr)
        predicted_density = np.asarray(predicted_molecule.density(), dtype=np.float64).sum(axis=-1)
        weights = np.asarray(point.molecule.grid.weights, dtype=np.float64)
        predicted_dm_total = np.asarray(predicted_molecule.rdm1, dtype=np.float64).sum(axis=0)
        predicted_electron_count = float(np.dot(weights, predicted_density))
        density_l1, density_l2, density_linf = _HELPERS._density_error_metrics(
            weights,
            predicted_density,
            point.fci_density_grid,
        )
        predicted_s1 = np.asarray(
            predict_excitation_energies(
                params,
                functional,
                predicted_molecule,
                nstates=1,
                use_tda=bool(use_tda),
            ),
            dtype=np.float64,
        )
        predicted_strengths = np.asarray(
            predict_oscillator_strengths(
                params,
                functional,
                predicted_molecule,
                nstates=1,
                use_tda=bool(use_tda),
            ),
            dtype=np.float64,
        )
        pred_gap = float(predicted_s1[0]) if predicted_s1.size > 0 else float("nan")
        pred_strength = (
            float(predicted_strengths[0]) if predicted_strengths.size > 0 else float("nan")
        )
        fci_gap = (
            float(point.fci_excitation_energies_h[0])
            if int(np.asarray(point.fci_excitation_energies_h).size) > 0
            else float("nan")
        )
        fci_total = _reference_s1_total_energy_h(point) if np.isfinite(fci_gap) else float("nan")
        pred_total = float(predicted_energy_h + pred_gap) if np.isfinite(pred_gap) else float("nan")
        rows.append(
            {
                "r_angstrom": float(point.r_angstrom),
                "fci_energy_h": float(point.fci_energy_h),
                "exact_energy_h": float(point.fci_energy_h),
                "predicted_energy_h": predicted_energy_h,
                "energy_abs_err_ev": abs(predicted_energy_h - point.fci_energy_h) * HARTREE_TO_EV,
                "fci_s1_total_energy_h": fci_total,
                "exact_s1_total_energy_h": fci_total,
                "predicted_s1_total_energy_h": pred_total,
                "s1_total_abs_err_ev": abs(pred_total - fci_total) * HARTREE_TO_EV,
                "fci_s1_h": fci_gap,
                "exact_s1_h": fci_gap,
                "predicted_s1_h": pred_gap,
                "s1_gap_abs_err_ev": abs(pred_gap - fci_gap) * HARTREE_TO_EV,
                "predicted_s1_oscillator_strength": pred_strength,
                "fci_electron_count": float(point.fci_electron_count),
                "exact_electron_count": float(point.fci_electron_count),
                "predicted_electron_count": predicted_electron_count,
                "electron_count_abs_err": abs(predicted_electron_count - point.fci_electron_count),
                "density_l1": density_l1,
                "density_l2": density_l2,
                "density_linf": density_linf,
                "predicted_dm_trace": float(np.trace(predicted_dm_total)),
            }
        )
        solver_name = "tda" if bool(use_tda) else "casida"
        solver_label = "TDA" if bool(use_tda) else "Casida"
        if np.isfinite(fci_gap) and np.isfinite(pred_gap):
            excited_rows.append(
                {
                    "r_angstrom": float(point.r_angstrom),
                    "solver": solver_name,
                    "state_index": 1,
                    "fci_total_energy_h": fci_total,
                    "exact_total_energy_h": fci_total,
                    "predicted_total_energy_h": pred_total,
                    "total_abs_err_ev": abs(pred_total - fci_total) * HARTREE_TO_EV,
                    "fci_excitation_h": fci_gap,
                    "exact_excitation_h": fci_gap,
                    "predicted_excitation_h": pred_gap,
                    "gap_abs_err_ev": abs(pred_gap - fci_gap) * HARTREE_TO_EV,
                    "predicted_oscillator_strength": pred_strength,
                }
            )
        if idx == 1 or idx % max(1, len(dense_points) // 10) == 0 or idx == len(dense_points):
            logger.log(
                f"[eval] {idx:3d}/{len(dense_points):3d} "
                f"R={point.r_angstrom:.4f} A "
                f"E0_pred={predicted_energy_h:.10f} Eh "
                f"S1_total_err={abs(pred_total - fci_total) * HARTREE_TO_EV:.6e} eV "
                f"S1_{solver_label}_gap_err={abs(pred_gap - fci_gap) * HARTREE_TO_EV:.6e} eV "
                f"f_{solver_label}={pred_strength:.6e}"
            )

    logger.log(f"[eval] done in {time.perf_counter() - t0:.2f} s")
    return rows, excited_rows


def plot_dense_summary(
    path: Path,
    rows: list[dict[str, float]],
    *,
    train_r_values: np.ndarray,
    basis: str,
    xc: str,
    training_mode: str,
    objective_label: str,
    use_tda: bool,
    system_label: str,
    reference_label: str,
) -> None:
    plt = _get_plt()
    r = np.asarray([row["r_angstrom"] for row in rows], dtype=np.float64)
    fci_energy = np.asarray([row["fci_energy_h"] for row in rows], dtype=np.float64)
    pred_energy = np.asarray([row["predicted_energy_h"] for row in rows], dtype=np.float64)
    energy_err_ev = np.asarray([row["energy_abs_err_ev"] for row in rows], dtype=np.float64)
    fci_s1 = np.asarray([row["fci_s1_h"] for row in rows], dtype=np.float64)
    pred_s1 = np.asarray([row["predicted_s1_h"] for row in rows], dtype=np.float64)
    fci_s1_total = np.asarray([row["fci_s1_total_energy_h"] for row in rows], dtype=np.float64)
    pred_s1_total = np.asarray([row["predicted_s1_total_energy_h"] for row in rows], dtype=np.float64)
    s1_total_err_ev = np.asarray([row["s1_total_abs_err_ev"] for row in rows], dtype=np.float64)
    s1_gap_err_ev = np.asarray([row["s1_gap_abs_err_ev"] for row in rows], dtype=np.float64)
    density_l2 = np.asarray([row["density_l2"] for row in rows], dtype=np.float64)

    solver_label = "TDA" if bool(use_tda) else "Casida"
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))

    ax = axes[0, 0]
    ax.plot(r, fci_energy, lw=2.0, label=f"{reference_label} ground")
    ax.plot(r, pred_energy, lw=2.0, label=f"Neural_xc {training_mode}")
    ax.scatter(
        train_r_values,
        np.interp(train_r_values, r, fci_energy),
        s=36,
        c="black",
        marker="o",
        label="5 training points",
        zorder=5,
    )
    ax.set_xlabel(f"{system_label} bond distance (Angstrom)")
    ax.set_ylabel("Total energy (Hartree)")
    ax.set_title("Ground-State Curve")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    ax.plot(r, fci_s1_total, lw=2.0, label=f"{reference_label} E1 total")
    ax.plot(r, pred_s1_total, lw=2.0, label=f"Neural {solver_label} S1 total")
    ax.scatter(
        train_r_values,
        np.interp(train_r_values, r, fci_s1_total),
        s=36,
        c="black",
        marker="o",
        zorder=5,
    )
    ax.set_xlabel(f"{system_label} bond distance (Angstrom)")
    ax.set_ylabel("Total energy (Hartree)")
    ax.set_title(f"S1 Total-Energy Curve ({solver_label})")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 0]
    ax.plot(r, np.maximum(energy_err_ev, 1e-16), lw=1.8, label="Ground abs. err. (eV)")
    ax.plot(r, np.maximum(density_l2, 1e-16), lw=1.8, label="Density L2")
    ax.set_xlabel(f"{system_label} bond distance (Angstrom)")
    ax.set_ylabel("Error")
    ax.set_yscale("log")
    ax.set_title("Ground-State Diagnostics")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    ax.plot(r, np.maximum(s1_total_err_ev, 1e-16), lw=1.9, label="S1 total abs. err. (eV)")
    ax.plot(r, np.maximum(s1_gap_err_ev, 1e-16), lw=1.4, alpha=0.75, label="S1 gap abs. err. (eV)")
    ax.set_xlabel(f"{system_label} bond distance (Angstrom)")
    ax.set_ylabel("Error (eV)")
    ax.set_yscale("log")
    ax.set_title(f"S1 Total-Energy Error ({solver_label})")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle(
        f"{system_label} {objective_label} training vs {reference_label} | {xc}/{basis}",
        y=0.985,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _fci_spectrum_lines(
    atom: str,
    *,
    basis: str,
    nroots: int,
) -> list[dict[str, float]]:
    mol = gto.M(
        atom=atom,
        unit="Angstrom",
        basis=basis,
        spin=0,
        charge=0,
        cart=True,
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"RHF did not converge for atom spec: {atom}")

    mo_coeff = np.asarray(mf.mo_coeff, dtype=np.float64)
    h1_mo = mo_coeff.T @ np.asarray(mf.get_hcore(), dtype=np.float64) @ mo_coeff
    eri_mo = ao2mo.kernel(mol, mo_coeff)
    cisolver = fci.direct_spin0.FCI(mol)
    e_ci, ci_vec = cisolver.kernel(
        h1_mo,
        eri_mo,
        h1_mo.shape[0],
        mol.nelectron,
        nroots=max(2, int(nroots)),
    )
    e_roots = np.asarray(e_ci, dtype=np.float64).reshape(-1)
    ci_roots = ci_vec if isinstance(ci_vec, (list, tuple)) else [ci_vec]
    exc_h = e_roots[1:] - e_roots[0]
    dip_ao = np.asarray(mol.intor_symmetric("int1e_r", comp=3), dtype=np.float64)
    lines: list[dict[str, float]] = []
    for idx in range(1, len(ci_roots)):
        trdm1_mo = np.asarray(
            cisolver.trans_rdm1(ci_roots[0], ci_roots[idx], h1_mo.shape[0], mol.nelectron),
            dtype=np.float64,
        )
        trdm1_ao = mo_coeff @ trdm1_mo @ mo_coeff.T
        mu = np.einsum("xpq,pq->x", dip_ao, trdm1_ao, optimize=True)
        strength = (2.0 / 3.0) * exc_h[idx - 1] * float(np.dot(mu, mu))
        lines.append(
            {
                "state_index": int(idx),
                "excitation_h": float(exc_h[idx - 1]),
                "excitation_ev": float(exc_h[idx - 1] * HARTREE_TO_EV),
                "oscillator_strength": float(strength),
            }
        )
    return lines


def _lorentzian_spectrum(
    grid_ev: np.ndarray,
    lines: list[dict[str, float]],
    *,
    eta_ev: float,
) -> np.ndarray:
    curve = np.zeros_like(grid_ev, dtype=np.float64)
    for line in lines:
        e = float(line["excitation_ev"])
        f = float(line["oscillator_strength"])
        curve += f * (eta_ev / np.pi) / ((grid_ev - e) ** 2 + eta_ev**2)
    return curve


def write_equilibrium_spectrum_csv(
    sticks_path: Path,
    broadened_path: Path,
    *,
    r_angstrom: float,
    solver_label: str,
    eta_ev: float,
    series: list[tuple[str, list[dict[str, float]], str]],
    grid_ev: np.ndarray,
    broadened: dict[str, np.ndarray],
) -> None:
    with sticks_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "series",
                "state_index",
                "r_angstrom",
                "evaluation_solver",
                "excitation_h",
                "excitation_ev",
                "oscillator_strength",
            ]
        )
        for label, lines, _ in series:
            for line in lines:
                writer.writerow(
                    [
                        label,
                        int(line["state_index"]),
                        float(r_angstrom),
                        solver_label.lower(),
                        float(line["excitation_h"]),
                        float(line["excitation_ev"]),
                        float(line["oscillator_strength"]),
                    ]
                )

    with broadened_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "series",
                "r_angstrom",
                "evaluation_solver",
                "eta_ev",
                "energy_ev",
                "intensity",
            ]
        )
        for label, _, _ in series:
            curve = np.asarray(broadened[label], dtype=np.float64)
            for energy_ev, intensity in zip(grid_ev, curve, strict=True):
                writer.writerow(
                    [
                        label,
                        float(r_angstrom),
                        solver_label.lower(),
                        float(eta_ev),
                        float(energy_ev),
                        float(intensity),
                    ]
                )


def plot_equilibrium_spectrum(
    fig_path: Path,
    json_path: Path,
    *,
    point: Any,
    basis: str,
    functional: Any,
    params: Any,
    training_config: GroundStateTrainingConfig,
    nstates: int,
    use_tda: bool,
) -> tuple[Path, Path]:
    plt = _get_plt()
    predicted_molecule = predict_ground_state_molecule(
        params,
        functional,
        point.molecule,
        training_config=training_config,
    )
    neural_lines = [
        {
            "state_index": int(i + 1),
            "excitation_h": float(energy_h),
            "excitation_ev": float(energy_h * HARTREE_TO_EV),
            "oscillator_strength": float(strength),
        }
        for i, (energy_h, strength) in enumerate(
            zip(
                np.asarray(
                    predict_excitation_energies(
                        params,
                        functional,
                        predicted_molecule,
                        nstates=int(nstates),
                        use_tda=bool(use_tda),
                    ),
                    dtype=np.float64,
                ),
                np.asarray(
                    predict_oscillator_strengths(
                        params,
                        functional,
                        predicted_molecule,
                        nstates=int(nstates),
                        use_tda=bool(use_tda),
                    ),
                    dtype=np.float64,
                ),
                strict=True,
            )
        )
    ]
    fci_lines = _fci_spectrum_lines(
        point.atom,
        basis=str(basis),
        nroots=int(nstates) + 1,
    )
    solver_label = "TDA" if bool(use_tda) else "Casida"
    payload = {
        "r_angstrom": float(point.r_angstrom),
        "fci_lines": fci_lines,
        "neural_tda_lines": neural_lines,
        "evaluation_solver": solver_label.lower(),
    }

    series = [
        ("FCI", fci_lines, "#111111"),
        (f"Neural {solver_label}", neural_lines, "#0f766e"),
    ]
    emin = min(line["excitation_ev"] for _, lines, _ in series for line in lines) - 2.0
    emax = max(line["excitation_ev"] for _, lines, _ in series for line in lines) + 2.0
    grid_ev = np.linspace(emin, emax, 2000)
    eta_ev = 0.35
    broadened = {
        label: _lorentzian_spectrum(grid_ev, lines, eta_ev=eta_ev)
        for label, lines, _ in series
    }
    sticks_csv = json_path.with_name(f"{json_path.stem}_sticks.csv")
    broadened_csv = json_path.with_name(f"{json_path.stem}_broadened.csv")
    write_equilibrium_spectrum_csv(
        sticks_csv,
        broadened_csv,
        r_angstrom=float(point.r_angstrom),
        solver_label=solver_label,
        eta_ev=eta_ev,
        series=series,
        grid_ev=grid_ev,
        broadened=broadened,
    )
    payload.update(
        {
            "eta_ev": float(eta_ev),
            "spectrum_sticks_csv": str(sticks_csv),
            "spectrum_broadened_csv": str(broadened_csv),
        }
    )
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(7.8, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.2]},
    )
    for idx, (label, lines, color) in enumerate(series):
        offset = 1 - idx
        ax0.hlines(offset, emin, emax, color="#d4d4d8", linewidth=0.8)
        for line in lines:
            ax0.vlines(
                line["excitation_ev"],
                offset,
                offset + line["oscillator_strength"],
                color=color,
                linewidth=2.4,
            )
            ax0.plot(
                line["excitation_ev"],
                offset + line["oscillator_strength"],
                "o",
                color=color,
                markersize=4,
            )
        ax0.text(
            emin + 0.2,
            offset + 0.72,
            label,
            color=color,
            fontsize=10,
            weight="bold",
            va="center",
        )
        ax1.plot(
            grid_ev,
            broadened[label],
            color=color,
            linewidth=2.2,
            label=label,
        )

    ax0.set_ylim(-0.2, 2.2)
    ax0.set_yticks([])
    ax0.set_ylabel("Stick lines")
    ax0.set_title(f"H2 equilibrium {solver_label} stick spectrum at R = {point.r_angstrom:.2f} A")
    ax1.set_xlabel("Excitation Energy (eV)")
    ax1.set_ylabel("Broadened Intensity (arb. units)")
    ax1.legend(frameon=False, loc="upper right")
    for ax in (ax0, ax1):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(emin, emax)
        ax.grid(alpha=0.15, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return sticks_csv, broadened_csv


def plot_training_loss(
    path: Path,
    training: dict[str, Any],
    *,
    title: str,
    objective_kind: str,
) -> None:
    plt = _get_plt()
    steps = np.asarray(
        training.get("history_steps", np.arange(len(training["loss_history"]))),
        dtype=np.int64,
    )
    loss = np.asarray(training["loss_history"], dtype=np.float64)
    s1_mae = np.asarray(training["s1_mae_history"], dtype=np.float64)
    eval_steps = np.asarray(training.get("eval_steps", []), dtype=np.int64)
    eval_loss = np.asarray(training.get("eval_loss_history", []), dtype=np.float64)
    eval_s1_mae = np.asarray(training.get("eval_s1_mae_history", []), dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].plot(steps, np.maximum(loss, 1e-16), lw=1.9, label="pre-update loss")
    if eval_steps.size and eval_loss.size:
        axes[0].plot(
            eval_steps,
            np.maximum(eval_loss, 1e-16),
            "o-",
            ms=2.8,
            lw=1.0,
            alpha=0.8,
            label="re-evaluated loss",
        )
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].set_title("Training Loss")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(steps, np.maximum(s1_mae, 1e-16), lw=1.9, label="pre-update E1 MAE")
    if eval_steps.size and eval_s1_mae.size:
        axes[1].plot(
            eval_steps,
            np.maximum(eval_s1_mae, 1e-16),
            "o-",
            ms=2.8,
            lw=1.0,
            alpha=0.8,
            label="re-evaluated E1 MAE",
        )
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("E1 total-energy MAE (Eh)")
    axes[1].set_yscale("log")
    axes[1].set_title("S1 Total-Energy Supervision" if str(objective_kind) != "e0_only" else "S1 Diagnostic")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_training_history_csv(path: Path, training: dict[str, Any]) -> None:
    steps = np.asarray(
        training.get("history_steps", np.arange(len(training["loss_history"]))),
        dtype=np.int64,
    )
    loss = np.asarray(training["loss_history"], dtype=np.float64)
    s1_penalty = np.asarray(training["s1_penalty_history"], dtype=np.float64)
    s1_mae = np.asarray(training["s1_mae_history"], dtype=np.float64)
    s1_mse = np.asarray(training["s1_mse_history"], dtype=np.float64)
    grad_norm = np.asarray(training["grad_norm_history"], dtype=np.float64)
    grad_abs_max = np.asarray(training["grad_abs_max_history"], dtype=np.float64)
    update_norm = np.asarray(training["param_update_norm_history"], dtype=np.float64)
    nonfinite_frac = np.asarray(training["nonfinite_grad_fraction_history"], dtype=np.float64)
    eval_map = {
        int(step): (float(eval_loss), float(eval_s1))
        for step, eval_loss, eval_s1 in zip(
            training.get("eval_steps", []),
            training.get("eval_loss_history", []),
            training.get("eval_s1_mae_history", []),
            strict=False,
        )
    }
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "loss_pre_update",
                "e1_total_penalty_pre_update",
                "e1_total_mae_pre_update",
                "e1_total_mse_pre_update",
                "grad_norm",
                "grad_abs_max",
                "param_update_norm",
                "nonfinite_grad_fraction",
                "loss_reevaluated",
                "e1_total_mae_reevaluated",
            ]
        )
        for idx, step in enumerate(steps):
            eval_loss_value, eval_s1_value = eval_map.get(int(step), (float("nan"), float("nan")))
            writer.writerow(
                [
                    int(step),
                    float(loss[idx]),
                    float(s1_penalty[idx]),
                    float(s1_mae[idx]),
                    float(s1_mse[idx]),
                    float(grad_norm[idx]),
                    float(grad_abs_max[idx]),
                    float(update_norm[idx]),
                    float(nonfinite_frac[idx]),
                    eval_loss_value,
                    eval_s1_value,
                ]
            )


def save_training_artifacts(
    *,
    outdir: Path,
    args: argparse.Namespace,
    training: dict[str, Any],
    params: Any,
    train_r_values: np.ndarray,
    eval_use_tda: bool,
) -> tuple[Path, Path, Path, Path]:
    training_png = outdir / "training_loss.png"
    plot_training_loss(
        training_png,
        training,
        title=f"{_system_label(args)} {_objective_display_label(args)} training loss",
        objective_kind=_resolved_objective_kind(args),
    )

    training_history_csv = outdir / "training_history.csv"
    write_training_history_csv(training_history_csv, training)

    checkpoint_path = outdir / "neural_xc_params.msgpack"
    checkpoint_path, checkpoint_meta_path = save_params_checkpoint(
        checkpoint_path,
        params,
        metadata=_training_checkpoint_metadata(
            args,
            checkpoint_kind="best_compat",
            parameter_state="best",
            step=int(training.get("min_loss_step", 0)),
            train_r_values=train_r_values,
            eval_use_tda=eval_use_tda,
            loss=float(training.get("min_loss", float("nan"))),
            min_loss=float(training.get("min_loss", float("nan"))),
            min_loss_step=int(training.get("min_loss_step", 0)),
        ),
    )
    return training_png, training_history_csv, checkpoint_path, checkpoint_meta_path


def write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    train_r_values: np.ndarray,
    dense_rows: list[dict[str, float]],
    training: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_meta_path: Path | None,
    equilibrium_point: Any,
) -> dict[str, Any]:
    ground_err_ev = np.asarray([row["energy_abs_err_ev"] for row in dense_rows], dtype=np.float64)
    s1_gap_err_ev = np.asarray([row["s1_gap_abs_err_ev"] for row in dense_rows], dtype=np.float64)
    density_l2 = np.asarray([row["density_l2"] for row in dense_rows], dtype=np.float64)
    predicted_s1 = np.asarray([row["predicted_s1_h"] for row in dense_rows], dtype=np.float64)
    fci_s1 = np.asarray([row["fci_s1_h"] for row in dense_rows], dtype=np.float64)
    predicted_e1_total = np.asarray(
        [row["predicted_s1_total_energy_h"] for row in dense_rows],
        dtype=np.float64,
    )
    fci_e1_total = np.asarray(
        [row["fci_s1_total_energy_h"] for row in dense_rows],
        dtype=np.float64,
    )
    s1_total_err_ev = np.asarray(
        [row["s1_total_abs_err_ev"] for row in dense_rows],
        dtype=np.float64,
    )
    eq_idx = int(np.argmin(np.abs(np.asarray([row["r_angstrom"] for row in dense_rows]) - equilibrium_point.r_angstrom)))

    eval_use_tda = bool(args.s1_use_tda) if args.eval_use_tda is None else bool(args.eval_use_tda)
    evaluation_solver = "tda" if eval_use_tda else "casida"
    objective_name = _objective_name(args)
    summary = {
        "system_label": _system_label(args),
        "atom1": str(getattr(args, "atom1", "H")),
        "atom2": str(getattr(args, "atom2", "H")),
        "charge": int(getattr(args, "charge", 0)),
        "spin": int(getattr(args, "spin", 0)),
        "basis": str(args.basis),
        "xc": str(args.xc),
        "reference_label": _reference_label(args),
        "external_s1_total_csv": (
            None
            if getattr(args, "external_s1_total_csv", None) is None
            else str(args.external_s1_total_csv)
        ),
        "external_s1_total_column": str(getattr(args, "external_s1_total_column", "")),
        "training_mode": str(args.training_mode),
        "reference_scf_backend": str(args.reference_scf_backend),
        "objective": objective_name,
        "objective_kind": _resolved_objective_kind(args),
        "scf_warm_start": bool(args.scf_warm_start),
        "scf_warm_start_update_interval": int(args.scf_warm_start_update_interval),
        "recover_nonfinite_steps": bool(args.recover_nonfinite_steps),
        "include_hfx_channel": bool(args.include_hfx_channel),
        "response_hf_mode": str(args.response_hf_mode),
        "include_pt2_channel": bool(args.include_pt2_channel),
        "pt2_channel_mode": str(args.pt2_channel_mode) if bool(args.include_pt2_channel) else None,
        "response_pt2_mode": (
            str(args.response_pt2_mode) if bool(args.include_pt2_channel) else None
        ),
        "s1_weight": float(args.s1_weight),
        "energy_mse_weight": float(args.energy_mse_weight),
        "energy_mae_weight": float(args.energy_mae_weight),
        "density_constraint_weight": float(args.density_constraint_weight),
        "s1_use_tda": bool(args.s1_use_tda),
        "eval_use_tda": eval_use_tda,
        "evaluation_solver": evaluation_solver,
        "dense_points": int(args.dense_points),
        "steps": int(args.steps),
        "final_loss": float(training["final_loss"]),
        "min_loss": float(training["min_loss"]),
        "min_loss_step": int(training["min_loss_step"]),
        "ground_mae_ev": float(ground_err_ev.mean()),
        "ground_max_ev": float(ground_err_ev.max()),
        "s1_total_mae_ev": float(s1_total_err_ev.mean()),
        "s1_total_max_ev": float(s1_total_err_ev.max()),
        "s1_gap_mae_ev": float(s1_gap_err_ev.mean()),
        "s1_gap_max_ev": float(s1_gap_err_ev.max()),
        "density_l2_mean": float(density_l2.mean()),
        "density_l2_max": float(density_l2.max()),
        "equilibrium_r_angstrom": float(equilibrium_point.r_angstrom),
        "equilibrium_fci_s1_ev": float(fci_s1[eq_idx] * HARTREE_TO_EV),
        "equilibrium_predicted_s1_ev": float(predicted_s1[eq_idx] * HARTREE_TO_EV),
        "equilibrium_s1_total_abs_err_ev": float(s1_total_err_ev[eq_idx]),
        "equilibrium_s1_abs_err_ev": float(s1_gap_err_ev[eq_idx]),
        "train_r_values_angstrom": [float(value) for value in train_r_values],
        "checkpoint": str(checkpoint_path),
        "checkpoint_meta": str(checkpoint_meta_path) if checkpoint_meta_path is not None else None,
        "best_checkpoint": training.get("best_checkpoint"),
        "best_checkpoint_meta": training.get("best_checkpoint_meta"),
        "latest_checkpoint": training.get("latest_checkpoint"),
        "latest_checkpoint_meta": training.get("latest_checkpoint_meta"),
        "final_checkpoint": training.get("final_checkpoint"),
        "final_checkpoint_meta": training.get("final_checkpoint_meta"),
    }

    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            f"{_system_label(args)} {_objective_display_label(args)} "
            f"Neural_xc vs {_reference_label(args)} summary\n"
        )
        for key, value in summary.items():
            handle.write(f"{key} = {value}\n")
    return summary


def main() -> None:
    args = parse_args()
    if args.train_r_values is not None:
        args.train_points = int(np.asarray(args.train_r_values, dtype=np.float64).size)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(outdir / "run.log")

    logger.log(
        "Config: "
        f"system={_system_label(args)}, atoms={str(args.atom1)}-{str(args.atom2)}, "
        f"charge={int(args.charge)}, spin={int(args.spin)}, "
        f"basis={args.basis}, xc={args.xc}, "
        f"R=[{args.r_min},{args.r_max}], train_points={args.train_points}, "
        f"dense_points={args.dense_points}, steps={args.steps}, "
        f"lr={args.learning_rate}, checkpoint_every={int(args.checkpoint_every)}, "
        f"training_mode={args.training_mode}, "
        f"objective={_objective_name(args)}, include_pt2_channel={bool(args.include_pt2_channel)}, "
        f"include_hfx_channel={bool(args.include_hfx_channel)}, "
        f"response_hf_mode={str(args.response_hf_mode)}, "
        f"pt2_channel_mode={str(args.pt2_channel_mode) if bool(args.include_pt2_channel) else 'none'}, "
        f"response_pt2_mode={str(args.response_pt2_mode) if bool(args.include_pt2_channel) else 'none'}, "
        f"reference_scf_backend={str(args.reference_scf_backend)}, "
        f"s1_weight={float(args.s1_weight)}, energy_mse_weight={float(args.energy_mse_weight)}, "
        f"energy_mae_weight={float(args.energy_mae_weight)}, density_constraint_weight={float(args.density_constraint_weight)}, "
        f"train_solver={'tda' if bool(args.s1_use_tda) else 'casida'}, "
        f"eval_solver={'tda' if (bool(args.s1_use_tda) if args.eval_use_tda is None else bool(args.eval_use_tda)) else 'casida'}, "
        f"reference_label={_reference_label(args)}"
    )
    logger.log("Loading runtime dependencies...")
    _HELPERS._load_runtime_dependencies(logger)
    logger.log("Runtime dependencies loaded.")

    if args.train_r_values is None:
        train_r_values = np.linspace(float(args.r_min), float(args.r_max), int(args.train_points))
    else:
        train_r_values = np.asarray(args.train_r_values, dtype=np.float64)
        if train_r_values.ndim != 1 or int(train_r_values.size) <= 0:
            raise ValueError("--train-r-values must provide at least one bond length.")
        args.train_points = int(train_r_values.size)
    dense_r_values = np.linspace(float(args.r_min), float(args.r_max), int(args.dense_points))

    logger.log(
        f"Building {int(args.train_points)}-point training references "
        f"({_reference_label(args)} target + strict-JAX reference)..."
    )
    train_points = _get_or_build_reference_curve(
        train_r_values,
        args=args,
        logger=logger,
        label="train_ref",
    )
    dense_points = None
    if not bool(args.defer_dense_eval):
        logger.log(
            f"Building {int(args.dense_points)}-point dense references "
            f"({_reference_label(args)} target + strict-JAX reference)..."
        )
        dense_points = _get_or_build_reference_curve(
            dense_r_values,
            args=args,
            logger=logger,
            label="dense_ref",
        )

    fixed_density_reference_checkpoint = None
    if str(args.training_mode) == "fixed_density":
        fixed_density_reference_checkpoint = (
            args.fixed_density_reference_checkpoint
            if args.fixed_density_reference_checkpoint is not None
            else args.init_checkpoint
        )
        if fixed_density_reference_checkpoint is not None:
            train_points = _rebuild_points_with_fixed_density_checkpoint(
                train_points,
                args=args,
                logger=logger,
                label="train_fixed_density_ref",
                checkpoint_path=fixed_density_reference_checkpoint,
            )
            if dense_points is not None:
                dense_points = _rebuild_points_with_fixed_density_checkpoint(
                    dense_points,
                    args=args,
                    logger=logger,
                    label="dense_fixed_density_ref",
                    checkpoint_path=fixed_density_reference_checkpoint,
                )

    training = train_functional(
        train_points,
        args=args,
        logger=logger,
    )
    params = training["best_params"]
    functional = training["functional"]
    gs_training = training["training_config"]

    eval_use_tda = bool(args.s1_use_tda) if args.eval_use_tda is None else bool(args.eval_use_tda)
    training_png, training_history_csv, checkpoint_path, checkpoint_meta_path = save_training_artifacts(
        outdir=outdir,
        args=args,
        training=training,
        params=params,
        train_r_values=train_r_values,
        eval_use_tda=eval_use_tda,
    )
    logger.log(f"Wrote training curve : {training_png}")
    logger.log(f"Wrote training history: {training_history_csv}")
    logger.log(f"Wrote pre-eval params : {checkpoint_path}")

    if dense_points is None:
        logger.log(
            f"Building {int(args.dense_points)}-point dense references after training "
            f"({_reference_label(args)} target + strict-JAX reference)..."
        )
        dense_points = _get_or_build_reference_curve(
            dense_r_values,
            args=args,
            logger=logger,
            label="dense_ref",
        )
        if fixed_density_reference_checkpoint is not None:
            dense_points = _rebuild_points_with_fixed_density_checkpoint(
                dense_points,
                args=args,
                logger=logger,
                label="dense_fixed_density_ref",
                checkpoint_path=fixed_density_reference_checkpoint,
            )

    eval_solver_label = "TDA" if eval_use_tda else "Casida"
    logger.log(
        f"Evaluating dense {int(args.dense_points)}-point {eval_solver_label} S1 curve..."
    )
    dense_rows, excited_rows = evaluate_dense_curve_tda(
        dense_points,
        params=params,
        functional=functional,
        training_config=gs_training,
        logger=logger,
        use_tda=eval_use_tda,
    )

    dense_csv = outdir / "h2_s1_tda_dense_curve.csv"
    excited_csv = outdir / "h2_s1_tda_excited_curve.csv"
    reference_points_csv = outdir / "h2_s1_reference_points.csv"
    write_dense_csv(dense_csv, dense_rows)
    write_dense_csv(excited_csv, excited_rows)
    write_dense_csv(
        reference_points_csv,
        [
            {
                "r_angstrom": float(point.r_angstrom),
                "fci_energy_h": float(point.fci_energy_h),
                "exact_energy_h": float(point.fci_energy_h),
                "fci_s1_h": (
                    float(point.fci_excitation_energies_h[0])
                    if int(np.asarray(point.fci_excitation_energies_h).size) > 0
                    else float("nan")
                ),
                "exact_s1_h": (
                    float(point.fci_excitation_energies_h[0])
                    if int(np.asarray(point.fci_excitation_energies_h).size) > 0
                    else float("nan")
                ),
                "fci_electron_count": float(point.fci_electron_count),
                "exact_electron_count": float(point.fci_electron_count),
                "reference_backend": str(getattr(args, "reference_scf_backend", "pyscf")),
                "reference_converged": 1,
                "reference_excited_method": (
                    "external_s1_total"
                    if getattr(args, "external_s1_total_csv", None)
                    else "fci"
                ),
                "reference_label": _reference_label(args),
            }
            for point in train_points
        ],
    )

    curve_png = outdir / "h2_s1_tda_dense_curve.png"
    plot_dense_summary(
        curve_png,
        dense_rows,
        train_r_values=train_r_values,
        basis=str(args.basis),
        xc=str(args.xc),
        training_mode=str(args.training_mode),
        objective_label=_objective_display_label(args),
        use_tda=eval_use_tda,
        system_label=_system_label(args),
        reference_label=_reference_label(args),
    )

    equilibrium_point = min(dense_points, key=lambda point: float(point.fci_energy_h))
    spectrum_png = None
    spectrum_json = None
    spectrum_sticks_csv = None
    spectrum_broadened_csv = None
    if not bool(args.skip_equilibrium_spectrum):
        spectrum_png = outdir / "h2_equilibrium_tda_spectrum_vs_fci.png"
        spectrum_json = outdir / "h2_equilibrium_tda_spectrum_vs_fci.json"
        spectrum_sticks_csv, spectrum_broadened_csv = plot_equilibrium_spectrum(
            spectrum_png,
            spectrum_json,
            point=equilibrium_point,
            basis=str(args.basis),
            functional=functional,
            params=params,
            training_config=gs_training,
            nstates=int(args.equilibrium_spectrum_nstates),
            use_tda=eval_use_tda,
        )

    summary_txt = outdir / "summary.txt"
    summary_json = outdir / "summary.json"
    summary = write_summary(
        summary_txt,
        args=args,
        train_r_values=train_r_values,
        dense_rows=dense_rows,
        training=training,
        checkpoint_path=checkpoint_path,
        checkpoint_meta_path=checkpoint_meta_path,
        equilibrium_point=equilibrium_point,
    )
    summary.update(
        {
            "dense_csv": str(dense_csv),
            "dense_curve_csv": str(dense_csv),
            "excited_csv": str(excited_csv),
            "excited_curve_csv": str(excited_csv),
            "reference_points_csv": str(reference_points_csv),
            "curve_png": str(curve_png),
            "figure_png": str(curve_png),
            "spectrum_png": None if spectrum_png is None else str(spectrum_png),
            "spectrum_json": None if spectrum_json is None else str(spectrum_json),
            "spectrum_sticks_csv": (
                None if spectrum_sticks_csv is None else str(spectrum_sticks_csv)
            ),
            "spectrum_broadened_csv": (
                None if spectrum_broadened_csv is None else str(spectrum_broadened_csv)
            ),
            "training_curve_png": str(training_png),
            "summary_txt": str(summary_txt),
            "visualization_manifest": str(outdir / "visualization_manifest.json"),
        }
    )
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    manifest_figures = [
        {
            "figure": str(curve_png),
            "data_files": [
                str(dense_csv),
                str(excited_csv),
                str(reference_points_csv),
                str(training_history_csv),
            ],
            "x": "r_angstrom",
            "y": [
                "fci_energy_h",
                "exact_energy_h",
                "predicted_energy_h",
                "fci_s1_total_energy_h",
                "exact_s1_total_energy_h",
                "predicted_s1_total_energy_h",
                "fci_s1_h",
                "exact_s1_h",
                "predicted_s1_h",
                "energy_abs_err_ev",
                "s1_total_abs_err_ev",
                "s1_gap_abs_err_ev",
                "density_l2",
            ],
        },
        {
            "figure": str(training_png),
            "data_files": [str(training_history_csv)],
            "x": "step",
            "y": [
                "loss_pre_update",
                "e1_total_penalty_pre_update",
                "e1_total_mae_pre_update",
                "e1_total_mse_pre_update",
                "grad_norm",
                "grad_abs_max",
                "param_update_norm",
            ],
        },
    ]
    if spectrum_png is not None and spectrum_sticks_csv is not None and spectrum_broadened_csv is not None:
        manifest_figures.append(
            {
                "figure": str(spectrum_png),
                "data_files": [str(spectrum_sticks_csv), str(spectrum_broadened_csv)],
                "x": "excitation_ev / energy_ev",
                "y": ["oscillator_strength", "intensity"],
            }
        )
    visualization_manifest = {
        "paper_experiment": "Bond-Scan S0/S1 Benchmarks",
        "description": (
            f"Data files needed to reproduce {_system_label(args)} bond-scan "
            "training visualizations."
        ),
        "figures": manifest_figures,
    }
    (outdir / "visualization_manifest.json").write_text(
        json.dumps(visualization_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    logger.log(f"Wrote dense csv : {dense_csv}")
    logger.log(f"Wrote excited csv: {excited_csv}")
    logger.log(f"Wrote curve png : {curve_png}")
    if spectrum_png is not None:
        logger.log(f"Wrote spectrum  : {spectrum_png}")
        logger.log(f"Wrote spectrum csv: {spectrum_sticks_csv}, {spectrum_broadened_csv}")
    logger.log(f"Wrote summary   : {summary_txt}")
    logger.log(f"Wrote params    : {checkpoint_path}")


if __name__ == "__main__":
    main()
