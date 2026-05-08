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

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax
from pyscf import ao2mo, fci, gto, scf

from td_graddft import neural_xc
from td_graddft.neural_xc import (
    GRADDFT_DEFAULT_DM21_HIDDEN_DIMS,
    GRADDFT_DEFAULT_INPUT_FEATURE_MODE,
    GRADDFT_DEFAULT_NETWORK_ARCHITECTURE,
)
from td_graddft.spectra import HARTREE_TO_EV
from td_graddft.tddft.test_module import LocalHFKhhResponseFunctionalWrapper
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


def _get_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train Neural_xc on the H2 S1 excitation gap only, using TDA for both "
            "training supervision and evaluation, then compare dense TDA results "
            "against FCI and plot the equilibrium stick spectrum."
        )
    )
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--xc", default="b3lyp")
    p.add_argument("--r-min", type=float, default=0.05)
    p.add_argument("--r-max", type=float, default=5.0)
    p.add_argument("--train-points", type=int, default=5)
    p.add_argument("--dense-points", type=int, default=100)
    p.add_argument("--steps", type=int, default=2000)
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
        "--training-mode",
        choices=("fixed_density", "self_consistent"),
        default="fixed_density",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=list(GRADDFT_DEFAULT_DM21_HIDDEN_DIMS),
    )
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
    p.add_argument(
        "--include-pt2-channel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add the projected restricted MP2 local channel to the Neural_xc basis.",
    )
    p.add_argument(
        "--pt2-channel-mode",
        choices=("scaled_projected", "local_exact"),
        default="scaled_projected",
        help="Choose the PT2 local channel representation when PT2 is enabled.",
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
        help="Weight for the S1 excitation-energy supervision term.",
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
        "--energy-mse-weight",
        type=float,
        default=0.0,
        help="Ground-state energy MSE weight. Keep at 0 for S1-only training.",
    )
    p.add_argument(
        "--energy-mae-weight",
        type=float,
        default=0.0,
        help="Ground-state energy MAE weight. Keep at 0 for S1-only training.",
    )
    p.add_argument(
        "--density-constraint-weight",
        type=float,
        default=0.0,
        help="Density-matching weight. Keep at 0 for pure S1 training.",
    )
    p.add_argument("--grids-level", type=int, default=0)
    p.add_argument("--max-l", type=int, default=3)
    p.add_argument(
        "--grid-ao-backend",
        choices=("jax", "pyscf"),
        default="jax",
    )
    p.add_argument(
        "--integral-backend",
        choices=("jax", "libcint"),
        default="libcint",
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
    p.add_argument(
        "--scf-require-convergence",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--grad-clip-norm", type=float, default=None)
    p.add_argument(
        "--scf-stop-gradient-on-unconverged",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--scf-stop-gradient-rms-threshold", type=float, default=None)
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
        help="Attempt to JIT the S1-only train step.",
    )
    p.add_argument(
        "--equilibrium-spectrum-nstates",
        type=int,
        default=3,
        help="Number of TDA states used for the equilibrium stick spectrum.",
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
    p.add_argument(
        "--test-module-local-hf-khh-response",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Experimental: replace the TD-response-only functional binding with a "
            "local HF K_hh wrapper. This forces response_hf_mode=local_projected."
        ),
    )
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


def _make_s1_functional(args: argparse.Namespace) -> Any:
    response_hf_mode = (
        "local_projected"
        if bool(args.test_module_local_hf_khh_response)
        else "nonlocal_exchange_only"
    )
    base_functional = neural_xc.Functional(
        architecture=str(args.network_architecture),
        semilocal_xc=tuple(str(name) for name in args.semilocal_xc),
        hidden_dims=tuple(int(value) for value in args.hidden_dims),
        input_feature_mode=str(args.input_feature_mode),
        include_pt2_channel=bool(args.include_pt2_channel),
        pt2_channel_mode=str(args.pt2_channel_mode),
        response_hf_mode=response_hf_mode,
        name=f"neural_xc_h2_s1_tda_{str(args.training_mode)}",
    )
    if not bool(args.test_module_local_hf_khh_response):
        return base_functional
    return LocalHFKhhResponseFunctionalWrapper(base_functional=base_functional)


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
        data.append(
            GroundStateDatum.from_parts(
                point.molecule,
                core=GroundStateCoreDatum(
                    target_total_energy=jnp.asarray(point.fci_energy_h, dtype=jnp.float64),
                    target_density_matrix=jnp.asarray(point.fci_dm_ao, dtype=jnp.float64),
                    density_constraint_weight=float(density_constraint_weight),
                ),
                excited_state=ExcitedStateDatum(
                    target_s1_energy=jnp.asarray(
                        float(point.fci_excitation_energies_h[0]),
                        dtype=jnp.float64,
                    ),
                    s1_constraint_weight=float(s1_weight),
                ),
            )
        )
    return tuple(data)


def _self_consistent_prediction_config(args: argparse.Namespace) -> GroundStateTrainingConfig:
    return GroundStateTrainingConfig(
        mode="self_consistent",
        energy_mse_weight=0.0,
        energy_mae_weight=0.0,
        scf_max_cycle=int(args.train_scf_max_cycle),
        scf_damping=float(args.train_scf_damping),
        scf_conv_tol_density=float(args.train_scf_conv_tol_density),
        scf_vxc_clip=float(args.train_scf_vxc_clip),
        scf_iterate_selection=str(args.scf_iterate_selection),
        scf_require_convergence=bool(args.scf_require_convergence),
        scf_gradient_mode=str(args.scf_gradient_mode),
        scf_implicit_diff_solver=str(args.scf_implicit_diff_solver),
        scf_implicit_diff_tolerance=float(args.scf_implicit_diff_tolerance),
        scf_implicit_diff_regularization=float(args.scf_implicit_diff_regularization),
        scf_implicit_diff_restart=int(args.scf_implicit_diff_restart),
        scf_stop_gradient_on_unconverged=bool(args.scf_stop_gradient_on_unconverged),
        scf_stop_gradient_rms_threshold=args.scf_stop_gradient_rms_threshold,
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
    gs_training = GroundStateTrainingConfig(
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

    if args.grad_clip_norm is not None and float(args.grad_clip_norm) > 0.0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(float(args.grad_clip_norm)),
            base_optimizer,
        )
    else:
        optimizer = base_optimizer

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
            logger.log(f"[train] jit compilation failed for S1-only train kernel: {exc!r}")
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
            logger.log(f"[train] jit compilation failed for S1-only train step: {exc!r}")
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
    s1_penalty_history = [_metric_mean(initial_metrics, "s1_penalty", 0.0)]
    s1_mae_history = [_metric_mean(initial_metrics, "s1_mae", 0.0)]
    s1_mse_history = [_metric_mean(initial_metrics, "s1_mse", 0.0)]
    grad_norm_history = [float("nan")]
    grad_abs_max_history = [float("nan")]
    param_update_norm_history = [float("nan")]
    nonfinite_grad_fraction_history = [0.0]
    eval_steps = [0]
    eval_loss_history = [initial_loss_val]
    eval_s1_mae_history = [_metric_mean(initial_metrics, "s1_mae", 0.0)]

    logger.log(
        "[train] "
        f"steps={int(args.steps)} "
        f"lr={float(args.learning_rate):.6g} "
        f"mode={str(args.training_mode)} "
        f"objective=s1_only_{'tda' if bool(args.s1_use_tda) else 'casida'} "
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
        train_s1_penalty_val = _metric_mean(train_metrics, "s1_penalty", 0.0)
        train_s1_mae_val = _metric_mean(train_metrics, "s1_mae", 0.0)
        train_s1_mse_val = _metric_mean(train_metrics, "s1_mse", 0.0)

        grad_norm_history.append(grad_norm_val)
        grad_abs_max_history.append(grad_abs_max_val)
        param_update_norm_history.append(param_update_norm_val)
        nonfinite_grad_fraction_history.append(nonfinite_grad_fraction_val)

        # ``train_metrics`` is evaluated on ``prev_state.params`` before the
        # optimizer update, so the loss observed during loop iteration ``step``
        # corresponds to the parameter state at ``step - 1``.
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

        if step == 1 or step % 10 == 0 or step == int(args.steps):
            eval_steps.append(step)
            eval_loss_history.append(train_loss_val)
            eval_s1_mae_history.append(train_s1_mae_val)
            current_lr = float(lr_schedule(step - 1)) if lr_schedule is not None else float(args.learning_rate)
            logger.log(
                "[train] "
                f"step={step:4d}/{int(args.steps):4d} "
                f"loss={train_loss_val:.8e} "
                f"s1_mae={train_s1_mae_val:.8e} "
                f"s1_pred_h={_metric_mean(train_metrics, 's1_predicted', float('nan')):.8e} "
                f"s1_target_h={_metric_mean(train_metrics, 's1_target', float('nan')):.8e} "
                f"scf_stop_grad_frac={_metric_mean(train_metrics, 'scf_stop_gradient_fraction', float('nan')):.8e} "
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
    final_s1_penalty = _metric_mean(final_metrics, "s1_penalty", 0.0)
    final_s1_mae = _metric_mean(final_metrics, "s1_mae", 0.0)
    final_s1_mse = _metric_mean(final_metrics, "s1_mse", 0.0)
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
        rows.append(
            {
                "r_angstrom": float(point.r_angstrom),
                "fci_energy_h": float(point.fci_energy_h),
                "predicted_energy_h": predicted_energy_h,
                "energy_abs_err_ev": abs(predicted_energy_h - point.fci_energy_h) * HARTREE_TO_EV,
                "fci_s1_h": fci_gap,
                "predicted_s1_h": pred_gap,
                "s1_gap_abs_err_ev": abs(pred_gap - fci_gap) * HARTREE_TO_EV,
                "predicted_s1_oscillator_strength": pred_strength,
                "fci_electron_count": float(point.fci_electron_count),
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
            fci_total = float(point.fci_total_energies_h[1])
            pred_total = float(predicted_energy_h + pred_gap)
            excited_rows.append(
                {
                    "r_angstrom": float(point.r_angstrom),
                    "solver": solver_name,
                    "state_index": 1,
                    "fci_total_energy_h": fci_total,
                    "predicted_total_energy_h": pred_total,
                    "total_abs_err_ev": abs(pred_total - fci_total) * HARTREE_TO_EV,
                    "fci_excitation_h": fci_gap,
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
    use_tda: bool,
) -> None:
    plt = _get_plt()
    r = np.asarray([row["r_angstrom"] for row in rows], dtype=np.float64)
    fci_energy = np.asarray([row["fci_energy_h"] for row in rows], dtype=np.float64)
    pred_energy = np.asarray([row["predicted_energy_h"] for row in rows], dtype=np.float64)
    energy_err_ev = np.asarray([row["energy_abs_err_ev"] for row in rows], dtype=np.float64)
    fci_s1 = np.asarray([row["fci_s1_h"] for row in rows], dtype=np.float64)
    pred_s1 = np.asarray([row["predicted_s1_h"] for row in rows], dtype=np.float64)
    s1_gap_err_ev = np.asarray([row["s1_gap_abs_err_ev"] for row in rows], dtype=np.float64)
    density_l2 = np.asarray([row["density_l2"] for row in rows], dtype=np.float64)

    solver_label = "TDA" if bool(use_tda) else "Casida"
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))

    ax = axes[0, 0]
    ax.plot(r, fci_energy, lw=2.0, label="FCI ground")
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
    ax.set_xlabel("H-H distance (Angstrom)")
    ax.set_ylabel("Total energy (Hartree)")
    ax.set_title("Ground-State Curve")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    ax.plot(r, fci_s1 * HARTREE_TO_EV, lw=2.0, label="FCI S1 gap")
    ax.plot(r, pred_s1 * HARTREE_TO_EV, lw=2.0, label=f"Neural {solver_label} S1 gap")
    ax.scatter(
        train_r_values,
        np.interp(train_r_values, r, fci_s1 * HARTREE_TO_EV),
        s=36,
        c="black",
        marker="o",
        zorder=5,
    )
    ax.set_xlabel("H-H distance (Angstrom)")
    ax.set_ylabel("Excitation energy (eV)")
    ax.set_title(f"S1 Gap Curve ({solver_label})")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 0]
    ax.plot(r, np.maximum(energy_err_ev, 1e-16), lw=1.8, label="Ground abs. err. (eV)")
    ax.plot(r, np.maximum(density_l2, 1e-16), lw=1.8, label="Density L2")
    ax.set_xlabel("H-H distance (Angstrom)")
    ax.set_ylabel("Error")
    ax.set_yscale("log")
    ax.set_title("Ground-State Diagnostics")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    ax.plot(r, np.maximum(s1_gap_err_ev, 1e-16), lw=1.9, label="S1 gap abs. err. (eV)")
    ax.set_xlabel("H-H distance (Angstrom)")
    ax.set_ylabel("Error (eV)")
    ax.set_yscale("log")
    ax.set_title(f"S1 Gap Error ({solver_label})")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle(f"H2 S1-only {solver_label} training vs FCI | {xc}/{basis}", y=0.985)
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
) -> None:
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
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    series = [
        ("FCI", fci_lines, "#111111"),
        (f"Neural {solver_label}", neural_lines, "#0f766e"),
    ]
    emin = min(line["excitation_ev"] for _, lines, _ in series for line in lines) - 2.0
    emax = max(line["excitation_ev"] for _, lines, _ in series for line in lines) + 2.0
    grid_ev = np.linspace(emin, emax, 2000)
    eta_ev = 0.35

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
            _lorentzian_spectrum(grid_ev, lines, eta_ev=eta_ev),
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


def plot_training_loss(path: Path, training: dict[str, Any], *, title: str) -> None:
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

    axes[1].plot(steps, np.maximum(s1_mae, 1e-16), lw=1.9, label="pre-update S1 MAE")
    if eval_steps.size and eval_s1_mae.size:
        axes[1].plot(
            eval_steps,
            np.maximum(eval_s1_mae, 1e-16),
            "o-",
            ms=2.8,
            lw=1.0,
            alpha=0.8,
            label="re-evaluated S1 MAE",
        )
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("S1 MAE (Eh)")
    axes[1].set_yscale("log")
    axes[1].set_title("S1 Supervision")
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
                "s1_penalty_pre_update",
                "s1_mae_pre_update",
                "s1_mse_pre_update",
                "grad_norm",
                "grad_abs_max",
                "param_update_norm",
                "nonfinite_grad_fraction",
                "loss_reevaluated",
                "s1_mae_reevaluated",
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
    eq_idx = int(np.argmin(np.abs(np.asarray([row["r_angstrom"] for row in dense_rows]) - equilibrium_point.r_angstrom)))

    eval_use_tda = bool(args.s1_use_tda) if args.eval_use_tda is None else bool(args.eval_use_tda)
    evaluation_solver = "tda" if eval_use_tda else "casida"
    objective_name = f"s1_only_{'tda' if bool(args.s1_use_tda) else 'casida'}"
    summary = {
        "basis": str(args.basis),
        "xc": str(args.xc),
        "training_mode": str(args.training_mode),
        "objective": objective_name,
        "scf_stop_gradient_on_unconverged": bool(args.scf_stop_gradient_on_unconverged),
        "scf_stop_gradient_rms_threshold": args.scf_stop_gradient_rms_threshold,
        "scf_warm_start": bool(args.scf_warm_start),
        "scf_warm_start_update_interval": int(args.scf_warm_start_update_interval),
        "recover_nonfinite_steps": bool(args.recover_nonfinite_steps),
        "include_pt2_channel": bool(args.include_pt2_channel),
        "pt2_channel_mode": str(args.pt2_channel_mode) if bool(args.include_pt2_channel) else None,
        "s1_weight": float(args.s1_weight),
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
        "s1_gap_mae_ev": float(s1_gap_err_ev.mean()),
        "s1_gap_max_ev": float(s1_gap_err_ev.max()),
        "density_l2_mean": float(density_l2.mean()),
        "density_l2_max": float(density_l2.max()),
        "equilibrium_r_angstrom": float(equilibrium_point.r_angstrom),
        "equilibrium_fci_s1_ev": float(fci_s1[eq_idx] * HARTREE_TO_EV),
        "equilibrium_predicted_s1_ev": float(predicted_s1[eq_idx] * HARTREE_TO_EV),
        "equilibrium_s1_abs_err_ev": float(s1_gap_err_ev[eq_idx]),
        "train_r_values_angstrom": [float(value) for value in train_r_values],
        "checkpoint": str(checkpoint_path),
        "checkpoint_meta": str(checkpoint_meta_path) if checkpoint_meta_path is not None else None,
    }

    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            f"H2 S1-only {evaluation_solver.upper()} Neural_xc vs FCI summary\n"
        )
        for key, value in summary.items():
            handle.write(f"{key} = {value}\n")
    return summary


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(outdir / "run.log")

    logger.log(
        "Config: "
        f"basis={args.basis}, xc={args.xc}, "
        f"R=[{args.r_min},{args.r_max}], train_points={args.train_points}, "
        f"dense_points={args.dense_points}, steps={args.steps}, "
        f"lr={args.learning_rate}, training_mode={args.training_mode}, "
        f"objective=s1_only_{'tda' if bool(args.s1_use_tda) else 'casida'}, include_pt2_channel={bool(args.include_pt2_channel)}, "
        f"pt2_channel_mode={str(args.pt2_channel_mode) if bool(args.include_pt2_channel) else 'none'}, "
        f"test_module_local_hf_khh_response={bool(args.test_module_local_hf_khh_response)}, "
        f"s1_weight={float(args.s1_weight)}, density_constraint_weight={float(args.density_constraint_weight)}, "
        f"train_solver={'tda' if bool(args.s1_use_tda) else 'casida'}, "
        f"eval_solver={'tda' if (bool(args.s1_use_tda) if args.eval_use_tda is None else bool(args.eval_use_tda)) else 'casida'}"
    )
    logger.log("Loading runtime dependencies...")
    _HELPERS._load_runtime_dependencies(logger)
    logger.log("Runtime dependencies loaded.")

    train_r_values = np.linspace(float(args.r_min), float(args.r_max), int(args.train_points))
    dense_r_values = np.linspace(float(args.r_min), float(args.r_max), int(args.dense_points))

    logger.log(f"Building {int(args.train_points)}-point training references (FCI + strict-JAX reference)...")
    train_points = build_reference_curve(
        train_r_values,
        args=args,
        logger=logger,
        label="train_ref",
    )
    logger.log(f"Building {int(args.dense_points)}-point dense references (FCI + strict-JAX reference)...")
    dense_points = build_reference_curve(
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
    eval_solver_label = "TDA" if eval_use_tda else "Casida"
    logger.log(
        f"Evaluating dense {int(args.dense_points)}-point {eval_solver_label} S1 curve and equilibrium spectrum..."
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
    write_dense_csv(dense_csv, dense_rows)
    write_dense_csv(excited_csv, excited_rows)

    curve_png = outdir / "h2_s1_tda_dense_curve.png"
    plot_dense_summary(
        curve_png,
        dense_rows,
        train_r_values=train_r_values,
        basis=str(args.basis),
        xc=str(args.xc),
        training_mode=str(args.training_mode),
        use_tda=eval_use_tda,
    )

    equilibrium_point = min(dense_points, key=lambda point: float(point.fci_energy_h))
    spectrum_png = outdir / "h2_equilibrium_tda_spectrum_vs_fci.png"
    spectrum_json = outdir / "h2_equilibrium_tda_spectrum_vs_fci.json"
    plot_equilibrium_spectrum(
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

    training_png = outdir / "training_loss.png"
    plot_training_loss(
        training_png,
        training,
        title=f"H2 S1-only {'TDA' if bool(args.s1_use_tda) else 'Casida'} training loss",
    )
    training_history_csv = outdir / "training_history.csv"
    write_training_history_csv(training_history_csv, training)

    checkpoint_path = outdir / "neural_xc_params.msgpack"
    checkpoint_path, checkpoint_meta_path = save_params_checkpoint(
        checkpoint_path,
        params,
        metadata={
            "basis": str(args.basis),
            "xc": str(args.xc),
            "training_mode": str(args.training_mode),
            "objective": f"s1_only_{'tda' if bool(args.s1_use_tda) else 'casida'}",
            "include_pt2_channel": bool(args.include_pt2_channel),
            "pt2_channel_mode": (
                str(args.pt2_channel_mode) if bool(args.include_pt2_channel) else None
            ),
            "s1_weight": float(args.s1_weight),
            "s1_use_tda": bool(args.s1_use_tda),
            "eval_use_tda": eval_use_tda,
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
            "steps": int(args.steps),
            "learning_rate": float(args.learning_rate),
            "hidden_dims": [int(value) for value in args.hidden_dims],
            "train_r_values_angstrom": [float(value) for value in train_r_values],
            "dense_points": int(args.dense_points),
            "test_module_local_hf_khh_response": bool(args.test_module_local_hf_khh_response),
        },
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
            "excited_csv": str(excited_csv),
            "curve_png": str(curve_png),
            "spectrum_png": str(spectrum_png),
            "spectrum_json": str(spectrum_json),
            "training_curve_png": str(training_png),
            "summary_txt": str(summary_txt),
        }
    )
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    logger.log(f"Wrote dense csv : {dense_csv}")
    logger.log(f"Wrote excited csv: {excited_csv}")
    logger.log(f"Wrote curve png : {curve_png}")
    logger.log(f"Wrote spectrum  : {spectrum_png}")
    logger.log(f"Wrote summary   : {summary_txt}")
    logger.log(f"Wrote params    : {checkpoint_path}")


if __name__ == "__main__":
    main()
