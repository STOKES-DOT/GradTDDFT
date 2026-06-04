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

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax
from pyscf import gto, scf, tdscf

from td_graddft.data.hdf5_cache import read_unrestricted_molecule, write_unrestricted_molecule
from td_graddft import neural_xc
from td_graddft.neural_xc import (
    DEFAULT_INPUT_FEATURE_MODE,
    DEFAULT_NETWORK_ARCHITECTURE,
    DEFAULT_NETWORK_HIDDEN_DIMS,
)
from td_graddft.scf import UKSConfig, unrestricted_molecule_from_spec_with_jax_uks
from td_graddft.training import (
    ExcitedStateDatum,
    ExcitedStateTrainingConfig,
    GroundStateCoreDatum,
    GroundStateCoreTrainingConfig,
    GroundStateDatum,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    ground_state_mse_loss_pointwise_dataset,
    make_ground_state_train_step,
    make_ground_state_predictor,
    predict_excitation_energies,
    save_params_checkpoint,
)

HARTREE_TO_EV = 27.211386245988
_DEFAULT_SEMILOCAL_XC = ("lda_x", "gga_x_b88", "lda_c_vwn_rpa", "gga_c_lyp")
_TRAIN_SCF_SAFETY_MAX_CYCLE = 32
_JAX_UKS_CACHE_VERSION = "spinpolarized-diis-v1"
_OBJECTIVE_CHOICES = ("auto", "e0_only", "s1_only", "joint")


@dataclass(frozen=True)
class ReferencePoint:
    r_angstrom: float
    atom: str
    molecule: Any
    exact_energy_h: float
    exact_total_energies_h: np.ndarray
    exact_excitation_energies_h: np.ndarray
    exact_dm_ao: np.ndarray
    exact_density_grid: np.ndarray
    exact_electron_count: float
    reference_backend: str
    reference_converged: bool
    reference_excited_method: str


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def build_h2plus_atom(r_angstrom: float) -> str:
    return f"H 0 0 {-0.5 * r_angstrom:.12f}; H 0 0 {0.5 * r_angstrom:.12f}"


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
        return f"S1-only {solver_label}"
    return f"Joint {solver_label}"


def _response_pt2_mode_label(args: argparse.Namespace) -> str:
    return str(args.response_pt2_mode) if bool(args.include_pt2_channel) else "none"


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.input_feature_mode == "dm21_original":
        args.input_feature_mode = "canonical"
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


def _move_scf_to_gpu(mf: Any) -> Any:
    try:
        import gpu4pyscf  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("--reference-scf-device gpu requires gpu4pyscf.") from exc
    to_gpu = getattr(mf, "to_gpu", None)
    if not callable(to_gpu):
        raise RuntimeError("gpu4pyscf did not expose to_gpu() on this SCF object.")
    return to_gpu()


def _move_scf_to_cpu(mf: Any) -> Any:
    to_cpu = getattr(mf, "to_cpu", None)
    return to_cpu() if callable(to_cpu) else mf


def _build_reference_scf(
    mol: Any,
    *,
    excited_xc: str | None,
) -> Any:
    if excited_xc is None:
        return scf.UHF(mol)
    mf = scf.UKS(mol)
    mf.xc = str(excited_xc)
    return mf


def _sorted_orbital_excitation_energies(mf: Any, *, occupation_tolerance: float = 1e-8) -> np.ndarray:
    mo_occ = np.asarray(mf.mo_occ, dtype=np.float64)
    mo_energy = np.asarray(mf.mo_energy, dtype=np.float64)
    gaps: list[float] = []
    for spin in range(int(mo_occ.shape[0])):
        occ_idx = np.where(mo_occ[spin] > occupation_tolerance)[0]
        vir_idx = np.where(mo_occ[spin] <= occupation_tolerance)[0]
        for i in occ_idx.tolist():
            for a in vir_idx.tolist():
                gaps.append(float(mo_energy[spin, a] - mo_energy[spin, i]))
    gaps_arr = np.asarray(sorted(value for value in gaps if value > 0.0), dtype=np.float64)
    return gaps_arr


def solve_h2plus_with_pyscf(
    atom: str,
    *,
    basis: str,
    nroots: int,
    excited_method: str,
    excited_xc: str | None,
    reference_scf_device: str,
    max_cycle: int,
    conv_tol: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, str, bool]:
    mol = gto.M(
        atom=atom,
        unit="Angstrom",
        basis=basis,
        charge=1,
        spin=1,
        cart=True,
        verbose=0,
    )
    mf = _build_reference_scf(mol, excited_xc=excited_xc)
    mf.max_cycle = int(max_cycle)
    mf.conv_tol = float(conv_tol)
    backend = "cpu"
    if str(reference_scf_device) == "gpu":
        try:
            mf = _move_scf_to_gpu(mf)
            backend = "gpu"
        except Exception as exc:
            backend = f"cpu_fallback:{type(exc).__name__}"
    energy = float(mf.kernel())
    mf_cpu = _move_scf_to_cpu(mf)
    dm_spin = np.asarray(mf_cpu.make_rdm1(), dtype=np.float64)
    dm_ao = dm_spin.sum(axis=0) if dm_spin.ndim == 3 else dm_spin
    method = str(excited_method).lower()
    if method == "orbital":
        excitation_energies = _sorted_orbital_excitation_energies(mf_cpu)
    elif method in {"tda", "tddft"}:
        td = tdscf.TDA(mf_cpu) if method == "tda" else tdscf.TDDFT(mf_cpu)
        td.nstates = int(nroots)
        td.kernel()
        excitation_energies = np.asarray(td.e, dtype=np.float64).reshape(-1)
    else:
        raise ValueError(f"Unsupported H2+ reference excited-state method {excited_method!r}.")
    excitation_energies = excitation_energies[: int(nroots)]
    total_energies = np.concatenate(
        [
            np.asarray([energy], dtype=np.float64),
            energy + excitation_energies,
        ]
    )
    return (
        energy,
        total_energies,
        excitation_energies,
        dm_ao,
        backend,
        bool(getattr(mf_cpu, "converged", True)),
    )


def build_reference_point(
    r_angstrom: float,
    *,
    args: argparse.Namespace,
) -> ReferencePoint:
    atom = build_h2plus_atom(r_angstrom)
    (
        exact_energy_h,
        exact_total_energies_h,
        exact_excitation_energies_h,
        exact_dm_ao,
        reference_backend,
        reference_converged,
    ) = solve_h2plus_with_pyscf(
        atom,
        basis=str(args.basis),
        nroots=max(1, int(args.nroots)),
        excited_method=str(args.reference_excited_method),
        excited_xc=(
            None
            if str(getattr(args, "reference_excited_xc", "") or "").strip() == ""
            else str(args.reference_excited_xc)
        ),
        reference_scf_device=str(args.reference_scf_device),
        max_cycle=int(args.reference_scf_max_cycle),
        conv_tol=float(args.reference_scf_conv_tol),
    )
    reference = unrestricted_molecule_from_spec_with_jax_uks(
        atom=atom,
        basis=str(args.basis),
        xc_spec=str(args.xc),
        unit="Angstrom",
        charge=1,
        spin=1,
        cart=True,
        grids_level=int(args.grids_level),
        max_l=int(args.max_l),
        uks_config=UKSConfig(
            xc_spec=str(args.xc),
            max_cycle=int(args.reference_scf_max_cycle),
            conv_tol=float(args.reference_scf_conv_tol),
            conv_tol_density=float(args.reference_scf_conv_tol_density),
            damping=float(args.reference_scf_damping),
            potential_clip=float(args.reference_scf_potential_clip),
        ),
        grid_ao_backend="jax",
        integral_backend=str(args.integral_backend),
        compute_local_hfx_features=(str(args.input_feature_mode) == "canonical"),
        compute_local_hfx_aux=(str(args.input_feature_mode) == "canonical"),
        compute_local_pt2_features=bool(args.include_pt2_channel),
        verbose=0,
    )
    ao = np.asarray(reference.ao, dtype=np.float64)
    weights = np.asarray(reference.grid.weights, dtype=np.float64)
    exact_density_grid = np.einsum("pq,rp,rq->r", exact_dm_ao, ao, ao, optimize=True)
    return ReferencePoint(
        r_angstrom=float(r_angstrom),
        atom=atom,
        molecule=reference,
        exact_energy_h=float(exact_energy_h),
        exact_total_energies_h=exact_total_energies_h,
        exact_excitation_energies_h=exact_excitation_energies_h,
        exact_dm_ao=exact_dm_ao,
        exact_density_grid=exact_density_grid,
        exact_electron_count=float(np.dot(weights, exact_density_grid)),
        reference_backend=reference_backend,
        reference_converged=reference_converged,
        reference_excited_method=str(args.reference_excited_method),
    )


def _reference_cache_path(args: argparse.Namespace) -> Path | None:
    value = str(getattr(args, "reference_cache", "") or "").strip()
    if not value:
        return None
    return Path(value)


def _reference_cache_key(r_angstrom: float, args: argparse.Namespace) -> str:
    basis = str(args.basis).replace("/", "_")
    xc = str(args.xc).replace("/", "_")
    feature_mode = "canonical" if str(args.input_feature_mode) == "dm21_original" else str(args.input_feature_mode)
    reference_excited_method = str(args.reference_excited_method)
    reference_excited_xc = str(getattr(args, "reference_excited_xc", "") or "hf").replace("/", "_")
    pt2_flag = "on" if bool(args.include_pt2_channel) else "off"
    return (
        f"h2plus/basis={basis}/xc={xc}/grid={int(args.grids_level)}/"
        f"max_l={int(args.max_l)}/integral={str(args.integral_backend)}/"
        f"features={feature_mode}/pt2={pt2_flag}/uks={_JAX_UKS_CACHE_VERSION}/"
        f"refexc={reference_excited_method}:{reference_excited_xc}/r={float(r_angstrom):.10f}"
    )


def _write_reference_point(group: Any, point: ReferencePoint) -> None:
    group.attrs["r_angstrom"] = float(point.r_angstrom)
    group.attrs["atom"] = str(point.atom)
    group.attrs["exact_energy_h"] = float(point.exact_energy_h)
    group.attrs["exact_electron_count"] = float(point.exact_electron_count)
    group.attrs["reference_backend"] = str(point.reference_backend)
    group.attrs["reference_converged"] = bool(point.reference_converged)
    group.attrs["reference_excited_method"] = str(point.reference_excited_method)
    for name in (
        "exact_total_energies_h",
        "exact_excitation_energies_h",
        "exact_dm_ao",
        "exact_density_grid",
    ):
        if name in group:
            del group[name]
        group.create_dataset(name, data=np.asarray(getattr(point, name)), compression="gzip")
    molecule_group = group.require_group("molecule")
    write_unrestricted_molecule(molecule_group, point.molecule)


def _read_reference_point(group: Any) -> ReferencePoint:
    return ReferencePoint(
        r_angstrom=float(group.attrs["r_angstrom"]),
        atom=str(group.attrs["atom"]),
        molecule=read_unrestricted_molecule(group["molecule"]),
        exact_energy_h=float(group.attrs["exact_energy_h"]),
        exact_total_energies_h=np.asarray(group["exact_total_energies_h"][()], dtype=np.float64),
        exact_excitation_energies_h=np.asarray(
            group["exact_excitation_energies_h"][()],
            dtype=np.float64,
        ),
        exact_dm_ao=np.asarray(group["exact_dm_ao"][()], dtype=np.float64),
        exact_density_grid=np.asarray(group["exact_density_grid"][()], dtype=np.float64),
        exact_electron_count=float(group.attrs["exact_electron_count"]),
        reference_backend=str(group.attrs["reference_backend"]),
        reference_converged=bool(group.attrs["reference_converged"]),
        reference_excited_method=str(group.attrs["reference_excited_method"]),
    )


def get_or_build_reference_point(
    r_angstrom: float,
    *,
    args: argparse.Namespace,
    logger: RunLogger,
) -> ReferencePoint:
    cache_path = _reference_cache_path(args)
    key = _reference_cache_key(float(r_angstrom), args)
    if cache_path is not None and cache_path.exists() and not bool(args.rebuild_reference_cache):
        try:
            import h5py

            with h5py.File(cache_path, "r") as handle:
                if key in handle:
                    logger.log(f"[ref_cache] hit R={float(r_angstrom):.4f}: {cache_path}::{key}")
                    return _read_reference_point(handle[key])
        except Exception as exc:
            logger.log(f"[ref_cache] miss/error R={float(r_angstrom):.4f}: {exc!r}; rebuilding")
    point = build_reference_point(float(r_angstrom), args=args)
    if cache_path is not None:
        try:
            import h5py

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(cache_path, "a") as handle:
                if key in handle:
                    del handle[key]
                _write_reference_point(handle.create_group(key), point)
            logger.log(f"[ref_cache] wrote R={float(r_angstrom):.4f}: {cache_path}::{key}")
        except Exception as exc:
            logger.log(f"[ref_cache] write failed R={float(r_angstrom):.4f}: {exc!r}")
    return point


def _metric_mean(metrics: dict[str, Any], key: str, default: float = float("nan")) -> float:
    if key not in metrics:
        return default
    arr = jnp.asarray(metrics[key])
    if int(arr.size) <= 0:
        return default
    return float(jnp.mean(arr))


def _tree_all_finite(tree: Any) -> bool:
    leaves = jax.tree_util.tree_leaves(tree)
    return all(bool(jnp.all(jnp.isfinite(jnp.asarray(leaf)))) for leaf in leaves)


def build_training_data(
    points: list[ReferencePoint],
    *,
    s1_weight: float,
    density_constraint_weight: float,
    density_matrix_constraint_weight: float,
) -> tuple[GroundStateDatum, ...]:
    data: list[GroundStateDatum] = []
    for point in points:
        if int(np.asarray(point.exact_excitation_energies_h).size) < 1:
            raise ValueError("Every H2+ training point must provide at least one excitation energy.")
        data.append(
            GroundStateDatum.from_parts(
                point.molecule,
                core=GroundStateCoreDatum(
                    target_total_energy=jnp.asarray(point.exact_energy_h, dtype=jnp.float64),
                    target_density_matrix=jnp.asarray(point.exact_dm_ao, dtype=jnp.float64),
                    density_constraint_weight=float(density_constraint_weight),
                    density_matrix_constraint_weight=float(density_matrix_constraint_weight),
                ),
                excited_state=ExcitedStateDatum(
                    target_s1_energy=jnp.asarray(
                        float(point.exact_excitation_energies_h[0]),
                        dtype=jnp.float64,
                    ),
                    s1_constraint_weight=float(s1_weight),
                ),
            ),
        )
    return tuple(data)


def train_functional(points: list[ReferencePoint], *, args: argparse.Namespace, logger: RunLogger):
    train_data = build_training_data(
        points,
        s1_weight=float(args.s1_weight),
        density_constraint_weight=float(args.density_constraint_weight),
        density_matrix_constraint_weight=float(args.density_matrix_constraint_weight),
    )
    functional = neural_xc.Functional(
        semilocal_xc=tuple(str(name) for name in args.semilocal_xc),
        hidden_dims=tuple(int(value) for value in args.hidden_dims),
        architecture=str(args.network_architecture),
        input_feature_mode=str(args.input_feature_mode),
        include_pt2_channel=bool(args.include_pt2_channel),
        pt2_channel_mode=str(args.pt2_channel_mode),
        response_hf_mode="approx",
        response_pt2_mode=str(args.response_pt2_mode),
        name="neural_xc_h2plus_s1_tda",
    )
    coefficient_prior = neural_xc.resolve_coefficient_prior_values(
        tuple(str(name) for name in args.semilocal_xc)
    )
    training_config = GroundStateTrainingConfig.from_parts(
        core=GroundStateCoreTrainingConfig(
            mode="self_consistent",
            energy_mse_weight=float(args.energy_mse_weight),
            energy_mae_weight=float(args.energy_mae_weight),
            energy_normalization=str(args.energy_normalization),
            coefficient_prior_weight=float(args.coefficient_prior_weight),
            coefficient_prior_values=coefficient_prior,
            scf_max_cycle=(
                _TRAIN_SCF_SAFETY_MAX_CYCLE
                if int(args.train_scf_max_cycle) <= 0
                else int(args.train_scf_max_cycle)
            ),
            scf_damping=float(args.train_scf_damping),
            scf_level_shift=float(args.train_scf_level_shift),
            scf_conv_tol_energy=args.train_scf_conv_tol_energy,
            scf_convergence_metric=str(args.train_scf_convergence_metric),
            scf_conv_tol_density=float(args.train_scf_conv_tol_density),
            scf_vxc_clip=float(args.train_scf_vxc_clip),
            scf_iterate_selection=str(args.scf_iterate_selection),
            scf_gradient_mode=str(args.scf_gradient_mode),
            scf_implicit_diff_solver=str(args.scf_implicit_diff_solver),
            scf_implicit_diff_tolerance=float(args.scf_implicit_diff_tolerance),
            scf_implicit_diff_regularization=float(args.scf_implicit_diff_regularization),
            scf_implicit_diff_restart=int(args.scf_implicit_diff_restart),
        ),
        excited_state=ExcitedStateTrainingConfig(
            s1_constraint_use_tda=bool(args.s1_use_tda),
        ),
    )
    lr_schedule = optax.exponential_decay(
        init_value=float(args.learning_rate),
        transition_steps=max(1, int(args.lr_decay_every)),
        decay_rate=float(args.lr_decay_factor),
        staircase=True,
    )
    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(int(args.seed)),
        points[0].molecule,
        optax.adam(lr_schedule),
    )
    eval_fn = lambda params: ground_state_mse_loss_pointwise_dataset(  # noqa: E731
        params,
        functional,
        train_data,
        training_config=training_config,
    )
    train_step = make_ground_state_train_step(
        functional,
        training_config=training_config,
        loss_fn=ground_state_mse_loss_pointwise_dataset,
    )
    step_fn = lambda current_state: train_step(current_state, train_data)  # noqa: E731
    compiled_eval = jax.jit(eval_fn) if bool(args.jit_eval) else eval_fn
    compiled_step = jax.jit(step_fn) if bool(args.jit_train) else step_fn

    if bool(args.skip_initial_eval):
        logger.log("[train] skipped initial loss evaluation")
        initial_loss = jnp.asarray(float("nan"), dtype=jnp.float64)
        initial_metrics: dict[str, Any] = {}
        min_loss = float("inf")
    else:
        logger.log("[train] running initial loss evaluation")
        initial_eval_t0 = time.perf_counter()
        initial_loss, initial_metrics = compiled_eval(state.params)
        logger.log(
            "[train] initial loss evaluation finished "
            f"in {time.perf_counter() - initial_eval_t0:.2f} s"
        )
        min_loss = float(initial_loss)
    best_params = state.params
    min_loss_step = 0
    rows = [
        {
            "step": 0,
            "loss": float(initial_loss),
            "energy_mae_h": _metric_mean(initial_metrics, "energy_mae"),
            "density_mse": _metric_mean(initial_metrics, "density_mse"),
            "density_penalty": _metric_mean(initial_metrics, "density_penalty"),
            "density_matrix_mse": _metric_mean(initial_metrics, "density_matrix_mse"),
            "density_matrix_penalty": _metric_mean(initial_metrics, "density_matrix_penalty"),
            "s1_penalty": _metric_mean(initial_metrics, "s1_penalty", 0.0),
            "s1_mae_h": _metric_mean(initial_metrics, "s1_mae", 0.0),
            "scf_converged_fraction": _metric_mean(initial_metrics, "scf_converged_fraction"),
            "scf_cycles_mean": _metric_mean(initial_metrics, "scf_cycles_mean"),
            "scf_cycles_max": _metric_mean(initial_metrics, "scf_cycles_max"),
            "grad_norm": float("nan"),
            "param_update_norm": float("nan"),
            "lr": float(args.learning_rate),
        }
    ]
    logger.log(
        "[train] "
        f"steps={int(args.steps)} lr={float(args.learning_rate):.6g} "
        f"lr_decay_every={int(args.lr_decay_every)} lr_decay_factor={float(args.lr_decay_factor):.6g} "
        f"objective={_objective_name(args)}"
    )
    t0 = time.perf_counter()
    for step in range(1, int(args.steps) + 1):
        if step == 1:
            logger.log("[train] running first optimization step")
            first_step_t0 = time.perf_counter()
        prev_state = state
        state, metrics = compiled_step(state)
        if step == 1:
            logger.log(
                "[train] first optimization step finished "
                f"in {time.perf_counter() - first_step_t0:.2f} s"
            )
        if not _tree_all_finite(state.params):
            state = prev_state
            logger.log(f"[train] non-finite params at step {step}; reverted update")
        loss = _metric_mean(metrics, "loss")
        if step >= 2 and loss < min_loss:
            min_loss = loss
            min_loss_step = step - 1
            best_params = prev_state.params
        row = {
            "step": step,
            "loss": loss,
            "energy_mae_h": _metric_mean(metrics, "energy_mae"),
            "density_mse": _metric_mean(metrics, "density_mse"),
            "density_penalty": _metric_mean(metrics, "density_penalty"),
            "density_matrix_mse": _metric_mean(metrics, "density_matrix_mse"),
            "density_matrix_penalty": _metric_mean(metrics, "density_matrix_penalty"),
            "s1_penalty": _metric_mean(metrics, "s1_penalty", 0.0),
            "s1_mae_h": _metric_mean(metrics, "s1_mae", 0.0),
            "scf_converged_fraction": _metric_mean(metrics, "scf_converged_fraction"),
            "scf_cycles_mean": _metric_mean(metrics, "scf_cycles_mean"),
            "scf_cycles_max": _metric_mean(metrics, "scf_cycles_max"),
            "grad_norm": _metric_mean(metrics, "grad_norm"),
            "param_update_norm": _metric_mean(metrics, "param_update_norm"),
            "lr": float(lr_schedule(step - 1)),
        }
        rows.append(row)
        if step == 1 or step % int(args.log_every) == 0 or step == int(args.steps):
            logger.log(
                "[train] "
                f"step={step:4d}/{int(args.steps):4d} loss={row['loss']:.8e} "
                f"energy_mae={row['energy_mae_h']:.8e} "
                f"density_mse={row['density_mse']:.8e} "
                f"dm_mse={row['density_matrix_mse']:.8e} "
                f"s1_mae={row['s1_mae_h']:.8e} "
                f"scf_conv_frac={row['scf_converged_fraction']:.6f} "
                f"scf_cycles_max={row['scf_cycles_max']:.6f} "
                f"grad_norm={row['grad_norm']:.8e} lr={row['lr']:.8e}"
            )
    final_loss, final_metrics = compiled_eval(state.params)
    if float(final_loss) < min_loss:
        min_loss = float(final_loss)
        min_loss_step = int(args.steps)
        best_params = state.params
    return {
        "functional": functional,
        "training_config": training_config,
        "params": state.params,
        "best_params": best_params,
        "history": rows,
        "elapsed_s": time.perf_counter() - t0,
        "final_loss": float(final_loss),
        "final_energy_mae_h": _metric_mean(final_metrics, "energy_mae"),
        "final_s1_mae_h": _metric_mean(final_metrics, "s1_mae", 0.0),
        "min_loss": min_loss,
        "min_loss_step": min_loss_step,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evaluate_curve(
    points: list[ReferencePoint],
    *,
    params: Any,
    functional: Any,
    training_config: GroundStateTrainingConfig,
    use_tda: bool,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    predictor = make_ground_state_predictor(functional, training_config=training_config)
    rows: list[dict[str, float]] = []
    excited_rows: list[dict[str, float]] = []
    for point in points:
        predicted_energy_h_arr, predicted_molecule = predictor(params, point.molecule)
        predicted_density = np.asarray(predicted_molecule.density(), dtype=np.float64)
        if predicted_density.ndim == 2:
            predicted_density = predicted_density.sum(axis=-1)
        weights = np.asarray(point.molecule.grid.weights, dtype=np.float64)
        diff = predicted_density - point.exact_density_grid
        predicted_gap_h = float(
            predict_excitation_energies(
                params,
                functional,
                predicted_molecule,
                nstates=1,
                use_tda=bool(use_tda),
            )[0]
        )
        exact_gap_h = (
            float(point.exact_excitation_energies_h[0])
            if int(np.asarray(point.exact_excitation_energies_h).size) > 0
            else float("nan")
        )
        rows.append(
            {
                "r_angstrom": float(point.r_angstrom),
                "exact_energy_h": float(point.exact_energy_h),
                "predicted_energy_h": float(predicted_energy_h_arr),
                "energy_abs_err_ev": abs(float(predicted_energy_h_arr) - point.exact_energy_h)
                * HARTREE_TO_EV,
                "exact_electron_count": float(point.exact_electron_count),
                "predicted_electron_count": float(np.dot(weights, predicted_density)),
                "density_l1": float(np.dot(weights, np.abs(diff))),
                "density_l2": float(np.sqrt(np.dot(weights, diff * diff))),
                "density_linf": float(np.max(np.abs(diff))),
                "exact_s1_h": exact_gap_h,
                "predicted_s1_h": predicted_gap_h,
                "s1_gap_abs_err_ev": abs(predicted_gap_h - exact_gap_h) * HARTREE_TO_EV,
            }
        )
        if np.isfinite(exact_gap_h):
            exact_total_h = (
                float(point.exact_total_energies_h[1])
                if int(np.asarray(point.exact_total_energies_h).size) > 1
                else float(point.exact_energy_h + exact_gap_h)
            )
            pred_total_h = float(predicted_energy_h_arr + predicted_gap_h)
            excited_rows.append(
                {
                    "r_angstrom": float(point.r_angstrom),
                    "exact_total_energy_h": exact_total_h,
                    "predicted_total_energy_h": pred_total_h,
                    "total_abs_err_ev": abs(pred_total_h - exact_total_h) * HARTREE_TO_EV,
                    "exact_excitation_h": exact_gap_h,
                    "predicted_excitation_h": predicted_gap_h,
                    "gap_abs_err_ev": abs(predicted_gap_h - exact_gap_h) * HARTREE_TO_EV,
                }
            )
    return rows, excited_rows


def plot_outputs(
    outdir: Path,
    history: list[dict[str, float]],
    curve_rows: list[dict[str, float]],
    *,
    objective_label: str,
    use_tda: bool,
    train_rows: list[dict[str, float]] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.asarray([row["step"] for row in history], dtype=float)
    loss = np.asarray([row["loss"] for row in history], dtype=float)
    dm = np.asarray([row["density_matrix_mse"] for row in history], dtype=float)
    s1_mae = np.asarray([row.get("s1_mae_h", 0.0) for row in history], dtype=float)
    r = np.asarray([row["r_angstrom"] for row in curve_rows], dtype=float)
    exact = np.asarray([row["exact_energy_h"] for row in curve_rows], dtype=float)
    pred = np.asarray([row["predicted_energy_h"] for row in curve_rows], dtype=float)
    err = np.asarray([row["energy_abs_err_ev"] for row in curve_rows], dtype=float)
    exact_s1 = np.asarray([row["exact_s1_h"] for row in curve_rows], dtype=float)
    pred_s1 = np.asarray([row["predicted_s1_h"] for row in curve_rows], dtype=float)
    s1_err = np.asarray([row["s1_gap_abs_err_ev"] for row in curve_rows], dtype=float)
    train_r = np.asarray([row["r_angstrom"] for row in train_rows or ()], dtype=float)
    train_exact = np.asarray([row["exact_energy_h"] for row in train_rows or ()], dtype=float)

    solver_label = "TDA" if bool(use_tda) else "Casida"
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))
    ax = axes[0, 0]
    ax.plot(steps, np.maximum(loss, 1e-18), label="loss")
    ax.plot(steps, np.maximum(dm, 1e-18), label="AO DM MSE")
    ax.plot(steps, np.maximum(s1_mae, 1e-18), label="S1 MAE (Ha)")
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    ax.set_title("Training Diagnostics")

    ax = axes[0, 1]
    ax.plot(r, exact, lw=1.8, label="exact ground")
    ax.plot(r, pred, lw=1.8, label="neural ground")
    if train_r.size:
        ax.scatter(
            train_r,
            train_exact,
            s=34,
            color="black",
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
            label="train points",
        )
    ax.set_xlabel("R (Angstrom)")
    ax.set_ylabel("Energy (Ha)")
    ax.legend(frameon=False)
    ax.set_title("Ground-State Curve")

    ax = axes[1, 0]
    ax.plot(r, exact_s1 * HARTREE_TO_EV, lw=1.8, label="exact S1 gap")
    ax.plot(r, pred_s1 * HARTREE_TO_EV, lw=1.8, label=f"neural {solver_label} S1 gap")
    ax.set_xlabel("R (Angstrom)")
    ax.set_ylabel("Excitation energy (eV)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    ax.set_title(f"S1 Gap Curve ({solver_label})")

    ax = axes[1, 1]
    ax.plot(r, np.maximum(err, 1e-16), lw=1.8, label="ground abs err (eV)")
    ax.plot(r, np.maximum(s1_err, 1e-16), lw=1.8, label="S1 gap abs err (eV)")
    ax.set_yscale("log")
    ax.set_xlabel("R (Angstrom)")
    ax.set_ylabel("Error")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    ax.set_title("Dense-Curve Errors")

    fig.suptitle(f"H2+ {objective_label} training vs reference | {solver_label}", y=0.985)
    fig.tight_layout()
    fig.savefig(outdir / "h2plus_s1_training_and_curve.png", dpi=180)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Self-consistent Neural XC training for the H2+ dissociation curve with "
            "optional S1 supervision from orbital/TDA/TDDFT reference data."
        )
    )
    p.add_argument("--basis", default="def2-svp")
    p.add_argument("--xc", default="b3lyp")
    p.add_argument("--r-min", type=float, default=0.4)
    p.add_argument("--r-max", type=float, default=6.0)
    p.add_argument("--train-points", type=int, default=5)
    p.add_argument("--dense-points", type=int, default=100)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument(
        "--reference-cache",
        default="outputs/reference_cache/h2plus_s1_references.h5",
        help="HDF5 cache for H2+ reference molecules, grids, integrals, and excited-state targets.",
    )
    p.add_argument("--rebuild-reference-cache", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--lr-decay-every", type=int, default=200)
    p.add_argument("--lr-decay-factor", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=list(DEFAULT_NETWORK_HIDDEN_DIMS))
    p.add_argument("--network-architecture", choices=("simple_mlp", "graddft_residual"), default=DEFAULT_NETWORK_ARCHITECTURE)
    p.add_argument("--input-feature-mode", choices=("enhanced", "canonical", "dm21_original"), default=DEFAULT_INPUT_FEATURE_MODE)
    p.add_argument("--semilocal-xc", nargs="+", default=list(_DEFAULT_SEMILOCAL_XC))
    p.add_argument("--reference-excited-method", choices=("orbital", "tda", "tddft"), default="tda")
    p.add_argument(
        "--reference-excited-xc",
        default="",
        help="Optional XC string for the PySCF TD reference. Empty uses UHF/TDHF.",
    )
    p.add_argument("--s1-weight", type=float, default=1.0)
    p.add_argument("--s1-use-tda", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-use-tda", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--objective", choices=_OBJECTIVE_CHOICES, default="auto")
    p.add_argument("--energy-mse-weight", type=float, default=0.0)
    p.add_argument("--energy-mae-weight", type=float, default=0.0)
    p.add_argument("--energy-normalization", choices=("none", "per_electron", "per_atom"), default="none")
    p.add_argument("--density-constraint-weight", type=float, default=0.0)
    p.add_argument("--density-matrix-constraint-weight", type=float, default=0.0)
    p.add_argument("--coefficient-prior-weight", type=float, default=0.0)
    p.add_argument("--include-pt2-channel", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--pt2-channel-mode",
        choices=("scaled_projected", "local_exact"),
        default="scaled_projected",
    )
    p.add_argument(
        "--response-pt2-mode",
        choices=("approx", "strict"),
        default="approx",
    )
    p.add_argument("--grids-level", type=int, default=2)
    p.add_argument("--max-l", type=int, default=3)
    p.add_argument("--integral-backend", choices=("jax", "cpu", "gpu", "libcint"), default="gpu")
    p.add_argument("--reference-scf-device", choices=("cpu", "gpu"), default="gpu")
    p.add_argument("--reference-scf-max-cycle", type=int, default=160)
    p.add_argument("--reference-scf-conv-tol", type=float, default=1e-10)
    p.add_argument("--reference-scf-conv-tol-density", type=float, default=1e-8)
    p.add_argument("--reference-scf-damping", type=float, default=0.15)
    p.add_argument("--reference-scf-potential-clip", type=float, default=20.0)
    p.add_argument("--train-scf-max-cycle", type=int, default=0)
    p.add_argument("--train-scf-damping", type=float, default=0.25)
    p.add_argument("--train-scf-level-shift", type=float, default=0.0)
    p.add_argument("--train-scf-conv-tol-energy", type=float, default=1e-6)
    p.add_argument("--train-scf-convergence-metric", choices=("energy_and_residual", "energy"), default="energy")
    p.add_argument("--train-scf-conv-tol-density", type=float, default=1e-8)
    p.add_argument("--train-scf-vxc-clip", type=float, default=20.0)
    p.add_argument("--scf-iterate-selection", choices=("final", "best_rms", "first_converged"), default="best_rms")
    p.add_argument("--scf-gradient-mode", choices=("expl", "impl"), default="impl")
    p.add_argument("--scf-implicit-diff-solver", choices=("normal_cg", "gmres", "bicgstab"), default="normal_cg")
    p.add_argument("--scf-implicit-diff-tolerance", type=float, default=1e-6)
    p.add_argument("--scf-implicit-diff-regularization", type=float, default=1e-3)
    p.add_argument("--scf-implicit-diff-restart", type=int, default=12)
    p.add_argument("--nroots", type=int, default=4)
    p.add_argument("--skip-initial-eval", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--skip-final-evaluation", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--jit-train", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--jit-eval", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--outdir", default="outputs/h2plus_s1_tda_train5_dense100")
    return _normalize_args(p.parse_args(argv))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(outdir / "run.log")
    eval_use_tda = bool(args.s1_use_tda) if args.eval_use_tda is None else bool(args.eval_use_tda)
    eval_solver_label = "TDA" if eval_use_tda else "Casida"
    logger.log(
        "Config: H2+ "
        f"basis={args.basis} grid={args.grids_level} R=[{args.r_min},{args.r_max}] "
        f"train_points={args.train_points} dense_points={args.dense_points} steps={args.steps} "
        f"reference_scf_device={args.reference_scf_device} integral_backend={args.integral_backend} "
        f"jax_backend={jax.default_backend()} "
        f"objective={_objective_name(args)} reference_excited_method={args.reference_excited_method} "
        f"train_solver={'tda' if bool(args.s1_use_tda) else 'casida'} eval_solver={eval_solver_label.lower()} "
        f"include_pt2_channel={bool(args.include_pt2_channel)} "
        f"pt2_channel_mode={str(args.pt2_channel_mode) if bool(args.include_pt2_channel) else 'none'} "
        f"response_pt2_mode={_response_pt2_mode_label(args)}"
    )
    train_r = np.linspace(float(args.r_min), float(args.r_max), int(args.train_points))
    dense_r = np.linspace(float(args.r_min), float(args.r_max), int(args.dense_points))
    train_points = []
    for idx, r_value in enumerate(train_r, start=1):
        point = get_or_build_reference_point(float(r_value), args=args, logger=logger)
        train_points.append(point)
        logger.log(
            f"[train_ref] {idx:3d}/{len(train_r):3d} R={point.r_angstrom:.4f} "
            f"E_ref={point.exact_energy_h:.10f} "
            f"S1_ref={float(point.exact_excitation_energies_h[0]) if int(np.asarray(point.exact_excitation_energies_h).size) > 0 else float('nan'):.10f} "
            f"backend={point.reference_backend} converged={int(point.reference_converged)} "
            f"grid_n={int(np.asarray(point.molecule.grid.weights).size)}"
        )
    dense_points = []
    for idx, r_value in enumerate(dense_r, start=1):
        point = get_or_build_reference_point(float(r_value), args=args, logger=logger)
        dense_points.append(point)
        if idx == 1 or idx == len(dense_r) or idx % 20 == 0:
            logger.log(f"[dense_ref] {idx:3d}/{len(dense_r):3d} R={point.r_angstrom:.4f}")

    training = train_functional(train_points, args=args, logger=logger)
    write_rows(outdir / "training_history.csv", training["history"])
    train_rows = [
        {
            "r_angstrom": point.r_angstrom,
            "exact_energy_h": point.exact_energy_h,
            "exact_s1_h": (
                float(point.exact_excitation_energies_h[0])
                if int(np.asarray(point.exact_excitation_energies_h).size) > 0
                else float("nan")
            ),
            "exact_electron_count": point.exact_electron_count,
            "reference_backend": point.reference_backend,
            "reference_converged": int(point.reference_converged),
            "reference_excited_method": point.reference_excited_method,
        }
        for point in train_points
    ]
    write_rows(outdir / "h2plus_s1_reference_points.csv", train_rows)
    curve_rows: list[dict[str, float]] = []
    excited_rows: list[dict[str, float]] = []
    if bool(args.skip_final_evaluation):
        logger.log("[eval] skipped final dense-curve evaluation")
    else:
        logger.log("[eval] running dense-curve evaluation")
        eval_t0 = time.perf_counter()
        curve_rows, excited_rows = evaluate_curve(
            dense_points,
            params=training["best_params"],
            functional=training["functional"],
            training_config=training["training_config"],
            use_tda=eval_use_tda,
        )
        logger.log(f"[eval] dense-curve evaluation finished in {time.perf_counter() - eval_t0:.2f} s")
        write_rows(outdir / "h2plus_s1_tda_dense_curve.csv", curve_rows)
        write_rows(outdir / "h2plus_s1_tda_excited_curve.csv", excited_rows)
    save_params_checkpoint(
        outdir / "neural_xc_params.msgpack",
        training["best_params"],
        metadata={
            "system": "H2+",
            "basis": str(args.basis),
            "grid_level": int(args.grids_level),
            "steps": int(args.steps),
            "objective": _objective_name(args),
            "objective_kind": _resolved_objective_kind(args),
            "reference_excited_method": str(args.reference_excited_method),
            "reference_excited_xc": str(args.reference_excited_xc),
            "include_pt2_channel": bool(args.include_pt2_channel),
            "pt2_channel_mode": (
                str(args.pt2_channel_mode) if bool(args.include_pt2_channel) else None
            ),
            "response_pt2_mode": _response_pt2_mode_label(args),
            "s1_weight": float(args.s1_weight),
            "s1_use_tda": bool(args.s1_use_tda),
            "eval_use_tda": bool(eval_use_tda),
            "energy_mse_weight": float(args.energy_mse_weight),
            "energy_mae_weight": float(args.energy_mae_weight),
            "density_constraint_weight": float(args.density_constraint_weight),
            "density_matrix_constraint_weight": float(args.density_matrix_constraint_weight),
            "reference_scf_device": str(args.reference_scf_device),
            "integral_backend": str(args.integral_backend),
        },
    )
    if bool(args.skip_final_evaluation):
        logger.log("[plot] skipped because final evaluation was disabled")
    else:
        try:
            plot_outputs(
                outdir,
                training["history"],
                curve_rows,
                objective_label=_objective_display_label(args),
                use_tda=eval_use_tda,
                train_rows=train_rows,
            )
        except Exception as exc:
            logger.log(f"[plot] skipped after error: {exc!r}")
    summary = {
        "system": "H2+",
        "basis": str(args.basis),
        "grid_level": int(args.grids_level),
        "steps": int(args.steps),
        "objective": _objective_name(args),
        "objective_kind": _resolved_objective_kind(args),
        "reference_excited_method": str(args.reference_excited_method),
        "reference_excited_xc": str(args.reference_excited_xc),
        "include_pt2_channel": bool(args.include_pt2_channel),
        "pt2_channel_mode": (
            str(args.pt2_channel_mode) if bool(args.include_pt2_channel) else None
        ),
        "response_pt2_mode": _response_pt2_mode_label(args),
        "s1_use_tda": bool(args.s1_use_tda),
        "eval_use_tda": bool(eval_use_tda),
        "learning_rate": float(args.learning_rate),
        "lr_decay_every": int(args.lr_decay_every),
        "lr_decay_factor": float(args.lr_decay_factor),
        "s1_weight": float(args.s1_weight),
        "energy_mse_weight": float(args.energy_mse_weight),
        "energy_mae_weight": float(args.energy_mae_weight),
        "density_constraint_weight": float(args.density_constraint_weight),
        "density_matrix_constraint_weight": float(args.density_matrix_constraint_weight),
        "reference_scf_device": str(args.reference_scf_device),
        "integral_backend": str(args.integral_backend),
        "skip_initial_eval": bool(args.skip_initial_eval),
        "skip_final_evaluation": bool(args.skip_final_evaluation),
        "elapsed_s": float(training["elapsed_s"]),
        "final_loss": float(training["final_loss"]),
        "final_energy_mae_ev": float(training["final_energy_mae_h"]) * HARTREE_TO_EV,
        "final_s1_mae_ev": float(training["final_s1_mae_h"]) * HARTREE_TO_EV,
        "min_loss": float(training["min_loss"]),
        "min_loss_step": int(training["min_loss_step"]),
        "dense_energy_mae_ev": (
            float(np.mean([row["energy_abs_err_ev"] for row in curve_rows]))
            if curve_rows
            else None
        ),
        "dense_s1_gap_mae_ev": (
            float(np.mean([row["s1_gap_abs_err_ev"] for row in curve_rows]))
            if curve_rows
            else None
        ),
        "training_history_csv": str(outdir / "training_history.csv"),
        "dense_curve_csv": (
            str(outdir / "h2plus_s1_tda_dense_curve.csv") if curve_rows else None
        ),
        "excited_curve_csv": (
            str(outdir / "h2plus_s1_tda_excited_curve.csv") if excited_rows else None
        ),
        "reference_points_csv": str(outdir / "h2plus_s1_reference_points.csv"),
        "figure_png": (
            str(outdir / "h2plus_s1_training_and_curve.png") if curve_rows else None
        ),
        "visualization_manifest": str(outdir / "visualization_manifest.json"),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "paper_experiment": "Bond-Scan S0/S1 Benchmarks",
        "description": "Data files needed to reproduce H2+ S0/S1 bond-scan visualizations.",
        "figures": (
            [
                {
                    "figure": str(outdir / "h2plus_s1_training_and_curve.png"),
                    "data_files": [
                        str(outdir / "training_history.csv"),
                        str(outdir / "h2plus_s1_tda_dense_curve.csv"),
                        str(outdir / "h2plus_s1_tda_excited_curve.csv"),
                        str(outdir / "h2plus_s1_reference_points.csv"),
                    ],
                    "x": ["step", "r_angstrom"],
                    "y": [
                        "loss",
                        "density_matrix_mse",
                        "exact_energy_h",
                        "predicted_energy_h",
                        "exact_s1_h",
                        "predicted_s1_h",
                    ],
                }
            ]
            if curve_rows
            else []
        ),
        "metadata_files": [str(outdir / "summary.json"), str(outdir / "neural_xc_params.msgpack.meta.json")],
    }
    (outdir / "visualization_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.log(
        "[summary] "
        f"final_energy_mae={summary['final_energy_mae_ev']:.8e} eV "
        f"final_s1_mae={summary['final_s1_mae_ev']:.8e} eV "
        f"dense_energy_mae={summary['dense_energy_mae_ev'] if summary['dense_energy_mae_ev'] is not None else float('nan'):.8e} eV "
        f"dense_s1_mae={summary['dense_s1_gap_mae_ev'] if summary['dense_s1_gap_mae_ev'] is not None else float('nan'):.8e} eV outdir={outdir}"
    )
    return summary


if __name__ == "__main__":
    main()
