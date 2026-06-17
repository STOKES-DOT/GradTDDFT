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
from pyscf import gto, scf

from td_graddft.data.hdf5_cache import read_unrestricted_molecule, write_unrestricted_molecule
from td_graddft import neural_xc
from td_graddft.neural_xc import (
    DEFAULT_INPUT_FEATURE_MODE,
    DEFAULT_NETWORK_ARCHITECTURE,
    DEFAULT_NETWORK_HIDDEN_DIMS,
)
from td_graddft.scf import UKSConfig, unrestricted_molecule_from_spec_with_jax_uks
from td_graddft.training import (
    GroundStateCoreDatum,
    GroundStateCoreTrainingConfig,
    GroundStateDatum,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    ground_state_mse_loss_pointwise_dataset,
    make_ground_state_train_step,
    make_ground_state_predictor,
    save_params_checkpoint,
)

HARTREE_TO_EV = 27.211386245988
_DEFAULT_SEMILOCAL_XC = ("lda_x", "gga_x_b88", "lda_c_vwn_rpa", "gga_c_lyp")
_TRAIN_SCF_SAFETY_MAX_CYCLE = 512
_JAX_UKS_CACHE_VERSION = "spinpolarized-fci-density-v2"


@dataclass(frozen=True)
class ReferencePoint:
    r_angstrom: float
    atom: str
    molecule: Any
    exact_energy_h: float
    exact_total_energies_h: np.ndarray
    exact_density_grid: np.ndarray
    exact_electron_count: float
    reference_backend: str
    reference_converged: bool


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


def solve_h2plus_fci_one_electron(
    atom: str,
    *,
    basis: str,
    nroots: int,
) -> tuple[float, np.ndarray, np.ndarray, str, bool]:
    mol = gto.M(
        atom=atom,
        unit="Angstrom",
        basis=basis,
        charge=1,
        spin=1,
        cart=True,
        verbose=0,
    )
    hcore = mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc")
    overlap = mol.intor_symmetric("int1e_ovlp")
    orbital_energies, mo_coeff = scf.hf.eig(hcore, overlap)
    total_energies = np.asarray(orbital_energies[: max(1, int(nroots))] + mol.energy_nuc())
    if total_energies.size < int(nroots):
        total_energies = np.pad(total_energies, (0, int(nroots) - total_energies.size), mode="edge")
    occupied_orbital = np.asarray(mo_coeff[:, 0], dtype=np.float64)
    rdm1_ao = np.outer(occupied_orbital, occupied_orbital)
    return float(total_energies[0]), total_energies, rdm1_ao, "fci_one_electron", True


def build_reference_point(
    r_angstrom: float,
    *,
    args: argparse.Namespace,
) -> ReferencePoint:
    atom = build_h2plus_atom(r_angstrom)
    (
        exact_energy_h,
        exact_total_energies_h,
        exact_rdm1_ao,
        reference_backend,
        reference_converged,
    ) = solve_h2plus_fci_one_electron(
        atom,
        basis=str(args.basis),
        nroots=max(1, int(args.nroots)),
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
        verbose=0,
    )
    ao = np.asarray(reference.ao, dtype=np.float64)
    weights = np.asarray(reference.grid.weights, dtype=np.float64)
    exact_density_grid = np.einsum("pq,rp,rq->r", exact_rdm1_ao, ao, ao, optimize=True)
    return ReferencePoint(
        r_angstrom=float(r_angstrom),
        atom=atom,
        molecule=reference,
        exact_energy_h=float(exact_energy_h),
        exact_total_energies_h=exact_total_energies_h,
        exact_density_grid=exact_density_grid,
        exact_electron_count=float(np.dot(weights, exact_density_grid)),
        reference_backend=reference_backend,
        reference_converged=reference_converged,
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
    return (
        f"h2plus/basis={basis}/xc={xc}/grid={int(args.grids_level)}/"
        f"max_l={int(args.max_l)}/integral={str(args.integral_backend)}/"
        f"features={feature_mode}/uks={_JAX_UKS_CACHE_VERSION}/r={float(r_angstrom):.10f}"
    )


def _write_reference_point(group: Any, point: ReferencePoint) -> None:
    group.attrs["r_angstrom"] = float(point.r_angstrom)
    group.attrs["atom"] = str(point.atom)
    group.attrs["exact_energy_h"] = float(point.exact_energy_h)
    group.attrs["exact_electron_count"] = float(point.exact_electron_count)
    group.attrs["reference_backend"] = str(point.reference_backend)
    group.attrs["reference_converged"] = bool(point.reference_converged)
    for name in ("exact_total_energies_h", "exact_density_grid"):
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
        exact_density_grid=np.asarray(group["exact_density_grid"][()], dtype=np.float64),
        exact_electron_count=float(group.attrs["exact_electron_count"]),
        reference_backend=str(group.attrs["reference_backend"]),
        reference_converged=bool(group.attrs["reference_converged"]),
    )


def _validate_cached_reference_point(point: ReferencePoint, args: argparse.Namespace) -> tuple[bool, str]:
    """Reject cache entries whose molecule integrals do not match their geometry."""

    mol = gto.M(
        atom=str(point.atom),
        unit="Angstrom",
        basis=str(args.basis),
        charge=1,
        spin=1,
        cart=True,
        verbose=0,
    )
    expected_hcore = np.asarray(mol.intor_symmetric("int1e_kin"), dtype=np.float64) + np.asarray(
        mol.intor_symmetric("int1e_nuc"),
        dtype=np.float64,
    )
    expected_overlap = np.asarray(mol.intor_symmetric("int1e_ovlp"), dtype=np.float64)
    cached = point.molecule
    cached_hcore = np.asarray(cached.h1e, dtype=np.float64)
    cached_overlap = np.asarray(cached.overlap_matrix, dtype=np.float64)
    if not np.allclose(cached_hcore, expected_hcore, rtol=1e-9, atol=1e-8):
        max_diff = float(np.max(np.abs(cached_hcore - expected_hcore)))
        return False, f"hcore mismatch max_abs={max_diff:.3e}"
    if not np.allclose(cached_overlap, expected_overlap, rtol=1e-9, atol=1e-8):
        max_diff = float(np.max(np.abs(cached_overlap - expected_overlap)))
        return False, f"overlap mismatch max_abs={max_diff:.3e}"
    cached_nuclear = float(cached.nuclear_repulsion)
    expected_nuclear = float(mol.energy_nuc())
    if abs(cached_nuclear - expected_nuclear) > 1e-8:
        return False, f"nuclear_repulsion mismatch cached={cached_nuclear:.12e} expected={expected_nuclear:.12e}"
    cached_coords = getattr(cached, "atom_coords", None)
    if cached_coords is not None:
        expected_coords = np.asarray(mol.atom_coords(), dtype=np.float64)
        cached_coords_arr = np.asarray(cached_coords, dtype=np.float64)
        if not np.allclose(cached_coords_arr, expected_coords, rtol=1e-9, atol=1e-8):
            max_diff = float(np.max(np.abs(cached_coords_arr - expected_coords)))
            return False, f"atom_coords mismatch max_abs={max_diff:.3e}"
    return True, "ok"


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
                    point = _read_reference_point(handle[key])
                    valid, reason = _validate_cached_reference_point(point, args)
                    if valid:
                        logger.log(f"[ref_cache] hit R={float(r_angstrom):.4f}: {cache_path}::{key}")
                        return point
                    logger.log(
                        f"[ref_cache] invalid R={float(r_angstrom):.4f}: {reason}; rebuilding {cache_path}::{key}"
                    )
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
    density_constraint_weight: float,
) -> tuple[GroundStateDatum, ...]:
    return tuple(
        GroundStateDatum.from_parts(
            point.molecule,
            core=GroundStateCoreDatum(
                target_total_energy=jnp.asarray(point.exact_energy_h, dtype=jnp.float64),
                target_density=jnp.asarray(point.exact_density_grid, dtype=jnp.float64),
                density_constraint_weight=float(density_constraint_weight),
            ),
        )
        for point in points
    )


def train_functional(points: list[ReferencePoint], *, args: argparse.Namespace, logger: RunLogger):
    train_data = build_training_data(
        points,
        density_constraint_weight=float(args.density_constraint_weight),
    )
    functional = neural_xc.Functional(
        semilocal_xc=tuple(str(name) for name in args.semilocal_xc),
        hidden_dims=tuple(int(value) for value in args.hidden_dims),
        architecture=str(args.network_architecture),
        input_feature_mode=str(args.input_feature_mode),
        include_pt2_channel=False,
        name="neural_xc_h2plus_fci_ground",
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
            scf_conv_tol_energy=args.train_scf_conv_tol_energy,
            scf_convergence_metric=str(args.train_scf_convergence_metric),
            scf_conv_tol_density=float(args.train_scf_conv_tol_density),
            scf_vxc_clip=float(args.train_scf_vxc_clip),
            scf_iterate_selection=str(args.scf_iterate_selection),
            scf_gradient_mode=str(args.scf_gradient_mode),
            scf_implicit_diff_tolerance=float(args.scf_implicit_diff_tolerance),
            scf_implicit_diff_regularization=float(args.scf_implicit_diff_regularization),
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

    initial_loss, initial_metrics = compiled_eval(state.params)
    best_params = state.params
    min_loss = float(initial_loss)
    min_loss_step = 0
    rows = [
        {
            "step": 0,
            "loss": float(initial_loss),
            "energy_mae_h": _metric_mean(initial_metrics, "energy_mae"),
            "density_mse": _metric_mean(initial_metrics, "density_mse"),
            "density_penalty": _metric_mean(initial_metrics, "density_penalty"),
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
        f"lr_decay_every={int(args.lr_decay_every)} lr_decay_factor={float(args.lr_decay_factor):.6g}"
    )
    t0 = time.perf_counter()
    for step in range(1, int(args.steps) + 1):
        prev_state = state
        state, metrics = compiled_step(state)
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
) -> list[dict[str, float]]:
    predictor = make_ground_state_predictor(functional, training_config=training_config)
    rows = []
    for point in points:
        predicted_energy_h_arr, predicted_molecule = predictor(params, point.molecule)
        predicted_density = np.asarray(predicted_molecule.density(), dtype=np.float64)
        if predicted_density.ndim == 2:
            predicted_density = predicted_density.sum(axis=-1)
        weights = np.asarray(point.molecule.grid.weights, dtype=np.float64)
        diff = predicted_density - point.exact_density_grid
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
            }
        )
    return rows


def plot_outputs(
    outdir: Path,
    history: list[dict[str, float]],
    curve_rows: list[dict[str, float]],
    train_rows: list[dict[str, float]] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(rows: list[dict[str, float]], key: str, default: float = float("nan")) -> np.ndarray:
        return np.asarray([float(row.get(key, default)) for row in rows], dtype=float)

    def positive(values: np.ndarray, floor: float = 1e-18) -> np.ndarray:
        return np.maximum(np.asarray(values, dtype=float), floor)

    steps = series(history, "step")
    loss = series(history, "loss")
    density_penalty = series(history, "density_penalty", 0.0)

    r = series(curve_rows, "r_angstrom")
    exact = series(curve_rows, "exact_energy_h")
    pred = series(curve_rows, "predicted_energy_h")
    err = series(curve_rows, "energy_abs_err_ev")
    train_r = series(train_rows or [], "r_angstrom")
    train_exact = series(train_rows or [], "exact_energy_h")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    axes = axes.ravel()

    ax = axes[0]
    ax.plot(steps, positive(loss), lw=2.6, label="total loss")
    ax.set_yscale("log")
    ax.set_title("Training Total Loss")
    ax.set_xlabel("Step")
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.plot(steps, positive(density_penalty), lw=2.6, color="tab:green", label="density penalty")
    ax.set_yscale("log")
    ax.set_title("Density Constraint Penalty")
    ax.set_xlabel("Step")
    ax.grid(alpha=0.2)

    ax = axes[2]
    ax.plot(r, exact, lw=3.0, label="exact")
    ax.plot(r, pred, lw=3.0, label="neural")
    if train_r.size:
        ax.scatter(
            train_r,
            train_exact,
            s=58,
            color="black",
            edgecolors="white",
            linewidths=1.1,
            zorder=5,
            label="train points",
        )
    ax.set_title("Ground-State Curve")
    ax.set_xlabel("R (Angstrom)")
    ax.set_ylabel("Energy (Ha)")
    ax.legend(frameon=False, fontsize=11, prop={"weight": "bold", "size": 11})
    ax.grid(alpha=0.2)

    ax = axes[3]
    ax.plot(r, positive(err, 1e-12), lw=3.0, color="tab:red")
    ax.set_yscale("log")
    ax.set_title("Absolute Energy Error")
    ax.set_xlabel("R (Angstrom)")
    ax.set_ylabel("eV")
    ax.grid(alpha=0.2)

    for ax in axes:
        ax.tick_params(axis="both", which="major", labelsize=11, width=1.6, length=5.5)
        ax.tick_params(axis="both", which="minor", width=1.2, length=3.5)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)
        title = ax.get_title()
        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")
        ax.xaxis.label.set_size(12)
        ax.xaxis.label.set_weight("bold")
        ax.yaxis.label.set_size(12)
        ax.yaxis.label.set_weight("bold")
    fig.tight_layout()
    fig.savefig(outdir / "h2plus_ground_training_and_curve.png", dpi=180)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-consistent Neural XC training for H2+ ground-state PES.")
    p.add_argument("--basis", default="def2-svp")
    p.add_argument("--xc", default="b3lyp")
    p.add_argument("--r-min", type=float, default=0.4)
    p.add_argument("--r-max", type=float, default=6.0)
    p.add_argument("--train-points", type=int, default=5)
    p.add_argument("--dense-points", type=int, default=100)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument(
        "--reference-cache",
        default="outputs/reference_cache/h2plus_ground_references.h5",
        help="HDF5 cache for H2+ reference molecules, grids, integrals, and HFX features.",
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
    p.add_argument("--energy-mse-weight", type=float, default=1.0)
    p.add_argument("--energy-mae-weight", type=float, default=1.0)
    p.add_argument("--energy-normalization", choices=("none", "per_electron", "per_atom"), default="none")
    p.add_argument("--density-constraint-weight", type=float, default=1.0)
    p.add_argument("--coefficient-prior-weight", type=float, default=0.0)
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
    p.add_argument("--train-scf-conv-tol-energy", type=float, default=1e-6)
    p.add_argument("--train-scf-convergence-metric", choices=("energy_and_residual", "energy"), default="energy")
    p.add_argument("--train-scf-conv-tol-density", type=float, default=1e-8)
    p.add_argument("--train-scf-vxc-clip", type=float, default=20.0)
    p.add_argument("--scf-iterate-selection", choices=("final", "best_rms", "first_converged"), default="best_rms")
    p.add_argument("--scf-gradient-mode", choices=("expl", "impl"), default="impl")
    p.add_argument("--scf-implicit-diff-tolerance", type=float, default=1e-6)
    p.add_argument("--scf-implicit-diff-regularization", type=float, default=1e-3)
    p.add_argument("--nroots", type=int, default=4)
    p.add_argument("--jit-train", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--jit-eval", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--outdir", default="outputs/h2plus_fci_ground_train5_dense100")
    args = p.parse_args(argv)
    if args.input_feature_mode == "dm21_original":
        args.input_feature_mode = "canonical"
    return args


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(outdir / "run.log")
    logger.log(
        "Config: H2+ "
        f"basis={args.basis} grid={args.grids_level} R=[{args.r_min},{args.r_max}] "
        f"train_points={args.train_points} dense_points={args.dense_points} steps={args.steps} "
        f"reference_method=fci_one_electron integral_backend={args.integral_backend}"
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
    curve_rows = evaluate_curve(
        dense_points,
        params=training["best_params"],
        functional=training["functional"],
        training_config=training["training_config"],
    )
    write_rows(outdir / "training_history.csv", training["history"])
    write_rows(outdir / "h2plus_ground_dense_curve.csv", curve_rows)
    train_rows = [
        {
            "r_angstrom": point.r_angstrom,
            "exact_energy_h": point.exact_energy_h,
            "exact_electron_count": point.exact_electron_count,
            "reference_backend": point.reference_backend,
            "reference_converged": int(point.reference_converged),
        }
        for point in train_points
    ]
    write_rows(outdir / "h2plus_reference_points.csv", train_rows)
    save_params_checkpoint(
        outdir / "neural_xc_params.msgpack",
        training["best_params"],
        metadata={
            "system": "H2+",
            "basis": str(args.basis),
            "grid_level": int(args.grids_level),
            "steps": int(args.steps),
            "density_constraint_weight": float(args.density_constraint_weight),
            "integral_backend": str(args.integral_backend),
        },
    )
    try:
        plot_outputs(outdir, training["history"], curve_rows, train_rows)
    except Exception as exc:
        logger.log(f"[plot] skipped after error: {exc!r}")
    summary = {
        "system": "H2+",
        "basis": str(args.basis),
        "grid_level": int(args.grids_level),
        "steps": int(args.steps),
        "learning_rate": float(args.learning_rate),
        "lr_decay_every": int(args.lr_decay_every),
        "lr_decay_factor": float(args.lr_decay_factor),
        "density_constraint_weight": float(args.density_constraint_weight),
        "integral_backend": str(args.integral_backend),
        "elapsed_s": float(training["elapsed_s"]),
        "final_loss": float(training["final_loss"]),
        "final_energy_mae_ev": float(training["final_energy_mae_h"]) * HARTREE_TO_EV,
        "min_loss": float(training["min_loss"]),
        "min_loss_step": int(training["min_loss_step"]),
        "dense_energy_mae_ev": float(np.mean([row["energy_abs_err_ev"] for row in curve_rows])),
        "training_history_csv": str(outdir / "training_history.csv"),
        "dense_curve_csv": str(outdir / "h2plus_ground_dense_curve.csv"),
        "reference_points_csv": str(outdir / "h2plus_reference_points.csv"),
        "figure_png": str(outdir / "h2plus_ground_training_and_curve.png"),
        "visualization_manifest": str(outdir / "visualization_manifest.json"),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "paper_experiment": "Ground-State Potential-Energy Surfaces",
        "description": "Data files needed to reproduce H2+ ground-state PES visualizations.",
        "figures": [
            {
                "figure": str(outdir / "h2plus_ground_training_and_curve.png"),
                "data_files": [
                    str(outdir / "training_history.csv"),
                    str(outdir / "h2plus_ground_dense_curve.csv"),
                    str(outdir / "h2plus_reference_points.csv"),
                ],
                "x": ["step", "r_angstrom"],
                "y": [
                    "loss",
                    "density_mse",
                    "exact_energy_h",
                    "predicted_energy_h",
                ],
            }
        ],
        "metadata_files": [str(outdir / "summary.json"), str(outdir / "neural_xc_params.msgpack.meta.json")],
    }
    (outdir / "visualization_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.log(
        "[summary] "
        f"final_energy_mae={summary['final_energy_mae_ev']:.8e} eV "
        f"dense_energy_mae={summary['dense_energy_mae_ev']:.8e} eV outdir={outdir}"
    )
    return summary


if __name__ == "__main__":
    main()
