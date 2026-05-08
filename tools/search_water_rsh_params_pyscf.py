from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/td_graddft_matplotlib")

import jax
import jax.numpy as jnp
import numpy as np

from td_graddft.jax_libxc import RestrictedFeatureBundle, eval_xc_energy_density
from td_graddft.nn_rsh import ResolvedRSHParameters, make_pyscf_rsh_spec


WATER_GEOMETRY = """
O  0.000000  0.000000  0.117790
H  0.000000  0.755453 -0.471161
H  0.000000 -0.755453 -0.471161
"""


@dataclass(frozen=True)
class RSHSearchParams:
    sr_hf_fraction: float
    lr_hf_fraction: float
    paper_beta: float
    omega: float
    lr_coordinate: float


@dataclass(frozen=True)
class StateResult:
    charge: int
    spin: int
    method: str
    energy: float
    homo: float
    lumo: float
    converged: bool
    nelec: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derivative-free PySCF search for water RSH parameters under "
            "Koopmans IP/EA constraints."
        ),
    )
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument(
        "--local-xc",
        default="td_graddft_lc_wpbe",
        help=(
            "td_graddft_lc_wpbe uses the project's strict JAX "
            "GGA_X_WPBEH(omega)+GGA_C_PBE evaluator."
        ),
    )
    parser.add_argument("--density-floor", type=float, default=1e-12)
    parser.add_argument("--no-jit", action="store_true")
    parser.add_argument("--sr-bounds", default="0.0,0.6")
    parser.add_argument(
        "--lr-coordinate-bounds",
        default="0.0,1.0",
        help=(
            "Bounds for t in lr = sr + t * (1 - sr), enforcing lr >= sr."
        ),
    )
    parser.add_argument("--omega-bounds", default="0.05,0.80")
    parser.add_argument("--maxiter", type=int, default=6)
    parser.add_argument("--popsize", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--polish-maxiter", type=int, default=25)
    parser.add_argument("--scf-max-cycle", type=int, default=80)
    parser.add_argument("--conv-tol", type=float, default=1e-9)
    parser.add_argument("--occupation-tol", type=float, default=1e-8)
    parser.add_argument("--nonconverged-penalty", type=float, default=100.0)
    parser.add_argument("--koopmans-ip-weight", type=float, default=1.0)
    parser.add_argument("--koopmans-ea-weight", type=float, default=1.0)
    parser.add_argument("--koopmans-lumo-ea-weight", type=float, default=0.0)
    parser.add_argument(
        "--koopmans-loss-kind",
        choices=("absolute", "squared"),
        default="squared",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/water_rsh_params_pyscf_de_search",
    )
    return parser.parse_args()


def _parse_float_pair(raw: str, *, name: str) -> tuple[float, float]:
    values = tuple(float(part.strip()) for part in str(raw).split(",") if part.strip())
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly lower,upper.")
    lower, upper = float(values[0]), float(values[1])
    if lower > upper:
        raise ValueError(f"{name} lower bound must be <= upper bound.")
    return lower, upper


def _params_from_coordinate(vector: Any) -> RSHSearchParams:
    arr = np.asarray(vector, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"Expected 3 search coordinates, got shape {arr.shape}.")
    sr = float(arr[0])
    lr_coord = float(arr[1])
    omega = float(arr[2])
    lr = sr + lr_coord * (1.0 - sr)
    return RSHSearchParams(
        sr_hf_fraction=sr,
        lr_hf_fraction=lr,
        paper_beta=lr - sr,
        omega=omega,
        lr_coordinate=lr_coord,
    )


@lru_cache(maxsize=8)
def _restricted_value_grad_kernel(density_floor: float, use_jit: bool):
    floor = float(density_floor)

    def point_energy(variables: jax.Array, omega: jax.Array) -> jax.Array:
        rho = jnp.maximum(variables[0], floor)
        sigma = jnp.maximum(variables[1], 0.0)
        features = RestrictedFeatureBundle(
            rho_a=0.5 * rho,
            rho_b=0.5 * rho,
            sigma_aa=0.25 * sigma,
            sigma_ab=0.25 * sigma,
            sigma_bb=0.25 * sigma,
            tau_a=jnp.asarray(0.0, dtype=variables.dtype),
            tau_b=jnp.asarray(0.0, dtype=variables.dtype),
        )
        return eval_xc_energy_density("lc_wpbe_local", features, omega=omega)

    mapped = jax.vmap(jax.value_and_grad(point_energy, argnums=0), in_axes=(0, None))
    return jax.jit(mapped) if bool(use_jit) else mapped


@lru_cache(maxsize=8)
def _unrestricted_value_grad_kernel(density_floor: float, use_jit: bool):
    floor = float(density_floor)

    def point_energy(variables: jax.Array, omega: jax.Array) -> jax.Array:
        rho_a = jnp.maximum(variables[0], floor)
        rho_b = jnp.maximum(variables[1], floor)
        features = RestrictedFeatureBundle(
            rho_a=rho_a,
            rho_b=rho_b,
            sigma_aa=jnp.maximum(variables[2], 0.0),
            sigma_ab=variables[3],
            sigma_bb=jnp.maximum(variables[4], 0.0),
            tau_a=jnp.asarray(0.0, dtype=variables.dtype),
            tau_b=jnp.asarray(0.0, dtype=variables.dtype),
        )
        return eval_xc_energy_density("lc_wpbe_local", features, omega=omega)

    mapped = jax.vmap(jax.value_and_grad(point_energy, argnums=0), in_axes=(0, None))
    return jax.jit(mapped) if bool(use_jit) else mapped


def _make_td_graddft_lc_wpbe_eval_xc(
    *,
    omega: float,
    density_floor: float,
    use_jit: bool,
):
    restricted_kernel = _restricted_value_grad_kernel(float(density_floor), bool(use_jit))
    unrestricted_kernel = _unrestricted_value_grad_kernel(float(density_floor), bool(use_jit))
    omega_value = float(omega)
    floor = float(density_floor)

    def eval_xc(
        xc_code: Any,
        rho: Any,
        spin: int = 0,
        relativity: int = 0,
        deriv: int = 1,
        omega: float | None = None,
        verbose: Any = None,
    ):
        del xc_code, relativity, verbose
        if int(deriv) > 1:
            raise NotImplementedError("The PySCF RSH parameter search needs only deriv<=1.")
        active_omega = omega_value if omega is None else float(omega)
        if int(spin) == 0:
            rho_np = np.asarray(rho, dtype=np.float64)
            if rho_np.ndim != 2 or rho_np.shape[0] < 4:
                raise ValueError(f"Expected restricted GGA rho shape (4, ngrids), got {rho_np.shape}.")
            rho_total = np.maximum(rho_np[0], 0.0)
            sigma = np.einsum("xg,xg->g", rho_np[1:4], rho_np[1:4])
            variables = jnp.stack(
                [jnp.asarray(rho_total), jnp.asarray(np.maximum(sigma, 0.0))],
                axis=-1,
            )
            energy_density, grad = restricted_kernel(
                variables,
                jnp.asarray(active_omega, dtype=variables.dtype),
            )
            energy_density_np = np.asarray(jax.device_get(energy_density), dtype=np.float64)
            grad_np = np.asarray(jax.device_get(grad), dtype=np.float64)
            exc = np.zeros_like(rho_total)
            mask = rho_total > floor
            exc[mask] = energy_density_np[mask] / rho_total[mask]
            if int(deriv) == 0:
                return exc, None, None, None
            vrho = np.nan_to_num(grad_np[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
            vsigma = np.nan_to_num(grad_np[:, 1], nan=0.0, posinf=0.0, neginf=0.0)
            return exc, (vrho, vsigma, None, None), None, None

        rho_a_np, rho_b_np = [np.asarray(block, dtype=np.float64) for block in rho]
        if rho_a_np.ndim != 2 or rho_b_np.ndim != 2 or rho_a_np.shape[0] < 4 or rho_b_np.shape[0] < 4:
            raise ValueError(
                "Expected unrestricted GGA rho blocks with shape (4, ngrids), "
                f"got {rho_a_np.shape} and {rho_b_np.shape}."
            )
        rho_a = np.maximum(rho_a_np[0], 0.0)
        rho_b = np.maximum(rho_b_np[0], 0.0)
        grad_a = rho_a_np[1:4]
        grad_b = rho_b_np[1:4]
        sigma_aa = np.einsum("xg,xg->g", grad_a, grad_a)
        sigma_ab = np.einsum("xg,xg->g", grad_a, grad_b)
        sigma_bb = np.einsum("xg,xg->g", grad_b, grad_b)
        variables = jnp.stack(
            [
                jnp.asarray(rho_a),
                jnp.asarray(rho_b),
                jnp.asarray(np.maximum(sigma_aa, 0.0)),
                jnp.asarray(sigma_ab),
                jnp.asarray(np.maximum(sigma_bb, 0.0)),
            ],
            axis=-1,
        )
        energy_density, grad = unrestricted_kernel(
            variables,
            jnp.asarray(active_omega, dtype=variables.dtype),
        )
        energy_density_np = np.asarray(jax.device_get(energy_density), dtype=np.float64)
        grad_np = np.asarray(jax.device_get(grad), dtype=np.float64)
        rho_total = rho_a + rho_b
        exc = np.zeros_like(rho_total)
        mask = rho_total > floor
        exc[mask] = energy_density_np[mask] / rho_total[mask]
        if int(deriv) == 0:
            return exc, None, None, None
        vrho = np.nan_to_num(grad_np[:, 0:2], nan=0.0, posinf=0.0, neginf=0.0)
        vsigma = np.nan_to_num(grad_np[:, 2:5], nan=0.0, posinf=0.0, neginf=0.0)
        return exc, (vrho, vsigma, None, None), None, None

    return eval_xc


def _as_spin_blocks(values: Any) -> list[np.ndarray]:
    arr = np.asarray(values)
    if arr.ndim == 1:
        return [arr]
    if arr.ndim == 2:
        return [np.asarray(block) for block in arr]
    raise ValueError(f"Expected orbital array with rank 1 or 2, got shape {arr.shape}.")


def _frontier_energy(
    mo_energy: Any,
    mo_occ: Any,
    *,
    occupied: bool,
    occupation_tol: float,
) -> float:
    energies = _as_spin_blocks(mo_energy)
    occs = _as_spin_blocks(mo_occ)
    selected: list[np.ndarray] = []
    for energy_block, occ_block in zip(energies, occs, strict=True):
        mask = occ_block > occupation_tol if occupied else occ_block <= occupation_tol
        selected.append(np.asarray(energy_block)[mask])
    values = np.concatenate(selected) if selected else np.asarray([], dtype=float)
    if values.size == 0:
        kind = "occupied" if occupied else "virtual"
        raise RuntimeError(f"No {kind} orbital found.")
    return float(np.max(values) if occupied else np.min(values))


def _build_mol(*, basis: str, charge: int, spin: int):
    from pyscf import gto

    return gto.M(
        atom=WATER_GEOMETRY,
        unit="Angstrom",
        basis=str(basis),
        charge=int(charge),
        spin=int(spin),
        verbose=0,
    )


def _run_state(
    *,
    params: RSHSearchParams,
    basis: str,
    local_xc: str,
    density_floor: float,
    use_jit: bool,
    charge: int,
    spin: int,
    grid_level: int,
    scf_max_cycle: int,
    conv_tol: float,
    occupation_tol: float,
) -> StateResult:
    from pyscf import dft

    mol = _build_mol(basis=basis, charge=charge, spin=spin)
    if spin == 0 and charge == 0:
        mf = dft.RKS(mol)
        method = "RKS"
    else:
        mf = dft.UKS(mol)
        method = "UKS"
    mf.xc = "LC_WPBE" if str(local_xc).lower() == "td_graddft_lc_wpbe" else "LC_WPBE"
    mf.grids.level = int(grid_level)
    mf.max_cycle = int(scf_max_cycle)
    mf.conv_tol = float(conv_tol)
    resolved = ResolvedRSHParameters(
        sr_hf_fraction=jnp.asarray(float(params.sr_hf_fraction)),
        lr_hf_fraction=jnp.asarray(float(params.lr_hf_fraction)),
        omega=jnp.asarray(float(params.omega)),
    )
    xc_description = (
        _make_td_graddft_lc_wpbe_eval_xc(
            omega=float(params.omega),
            density_floor=float(density_floor),
            use_jit=bool(use_jit),
        )
        if str(local_xc).lower() == "td_graddft_lc_wpbe"
        else str(local_xc)
    )
    spec = make_pyscf_rsh_spec(
        xc_description=xc_description,
        xctype="GGA",
        resolved_params=resolved,
    )
    spec.install_into_mf(mf)
    energy = float(mf.kernel())
    homo = _frontier_energy(
        mf.mo_energy,
        mf.mo_occ,
        occupied=True,
        occupation_tol=float(occupation_tol),
    )
    lumo = _frontier_energy(
        mf.mo_energy,
        mf.mo_occ,
        occupied=False,
        occupation_tol=float(occupation_tol),
    )
    return StateResult(
        charge=int(charge),
        spin=int(spin),
        method=method,
        energy=energy,
        homo=homo,
        lumo=lumo,
        converged=bool(mf.converged),
        nelec=int(mol.nelectron),
    )


def _penalty(value: float, *, kind: str) -> float:
    return abs(value) if kind == "absolute" else value * value


def _evaluate(
    *,
    vector: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    params = _params_from_coordinate(vector)
    neutral = _run_state(
        params=params,
        basis=str(args.basis),
        local_xc=str(args.local_xc),
        density_floor=float(args.density_floor),
        use_jit=not bool(args.no_jit),
        charge=0,
        spin=0,
        grid_level=int(args.grid_level),
        scf_max_cycle=int(args.scf_max_cycle),
        conv_tol=float(args.conv_tol),
        occupation_tol=float(args.occupation_tol),
    )
    cation = _run_state(
        params=params,
        basis=str(args.basis),
        local_xc=str(args.local_xc),
        density_floor=float(args.density_floor),
        use_jit=not bool(args.no_jit),
        charge=1,
        spin=1,
        grid_level=int(args.grid_level),
        scf_max_cycle=int(args.scf_max_cycle),
        conv_tol=float(args.conv_tol),
        occupation_tol=float(args.occupation_tol),
    )
    anion = _run_state(
        params=params,
        basis=str(args.basis),
        local_xc=str(args.local_xc),
        density_floor=float(args.density_floor),
        use_jit=not bool(args.no_jit),
        charge=-1,
        spin=1,
        grid_level=int(args.grid_level),
        scf_max_cycle=int(args.scf_max_cycle),
        conv_tol=float(args.conv_tol),
        occupation_tol=float(args.occupation_tol),
    )

    ip_residual = neutral.homo + cation.energy - neutral.energy
    ea_residual = anion.homo + neutral.energy - anion.energy
    lumo_ea_residual = neutral.lumo + neutral.energy - anion.energy
    gap_residual = (neutral.lumo - neutral.homo) - (
        cation.energy + anion.energy - 2.0 * neutral.energy
    )
    loss = (
        float(args.koopmans_ip_weight)
        * _penalty(ip_residual, kind=str(args.koopmans_loss_kind))
        + float(args.koopmans_ea_weight)
        * _penalty(ea_residual, kind=str(args.koopmans_loss_kind))
        + float(args.koopmans_lumo_ea_weight)
        * _penalty(lumo_ea_residual, kind=str(args.koopmans_loss_kind))
    )
    all_converged = bool(neutral.converged and cation.converged and anion.converged)
    if not all_converged:
        loss += float(args.nonconverged_penalty)
    return {
        "sr_hf_fraction": float(params.sr_hf_fraction),
        "lr_hf_fraction": float(params.lr_hf_fraction),
        "paper_alpha": float(params.sr_hf_fraction),
        "paper_beta": float(params.paper_beta),
        "omega": float(params.omega),
        "lr_coordinate": float(params.lr_coordinate),
        "loss": float(loss),
        "koopmans_ip_residual": float(ip_residual),
        "koopmans_ea_residual": float(ea_residual),
        "koopmans_lumo_ea_residual": float(lumo_ea_residual),
        "koopmans_gap_residual": float(gap_residual),
        "koopmans_ip_mae": float(abs(ip_residual)),
        "koopmans_ea_mae": float(abs(ea_residual)),
        "koopmans_lumo_ea_mae": float(abs(lumo_ea_residual)),
        "koopmans_gap_mae": float(abs(gap_residual)),
        "neutral": asdict(neutral),
        "cation": asdict(cation),
        "anion": asdict(anion),
        "all_converged": all_converged,
    }


def _write_history_csv(history: list[dict[str, Any]], outdir: Path) -> Path:
    csv_path = outdir / "rsh_param_search_history.csv"
    fieldnames = [
        "eval",
        "sr_hf_fraction",
        "lr_hf_fraction",
        "paper_beta",
        "omega",
        "lr_coordinate",
        "loss",
        "koopmans_ip_residual",
        "koopmans_ea_residual",
        "koopmans_lumo_ea_residual",
        "koopmans_gap_residual",
        "koopmans_ip_mae",
        "koopmans_ea_mae",
        "koopmans_lumo_ea_mae",
        "koopmans_gap_mae",
        "all_converged",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return csv_path


def _write_plot(history: list[dict[str, Any]], outdir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    evals = [int(row["eval"]) for row in history]
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.4), sharex=True)
    axes[0].plot(evals, [float(row["loss"]) for row in history], "-o", ms=3, label="loss")
    axes[0].plot(
        evals,
        [float(row["koopmans_ip_mae"]) for row in history],
        "--",
        label="|HOMO(N)+IP|",
    )
    axes[0].plot(
        evals,
        [float(row["koopmans_ea_mae"]) for row in history],
        "--",
        label="|HOMO(N+1)+EA|",
    )
    axes[0].plot(
        evals,
        [float(row["koopmans_lumo_ea_mae"]) for row in history],
        ":",
        label="|LUMO(N)+EA|",
    )
    axes[0].set_ylabel("Hartree")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(evals, [float(row["sr_hf_fraction"]) for row in history], label="sr")
    axes[1].plot(evals, [float(row["lr_hf_fraction"]) for row in history], label="lr")
    axes[1].plot(evals, [float(row["omega"]) for row in history], label="omega")
    axes[1].set_xlabel("objective evaluation")
    axes[1].set_ylabel("RSH parameter")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    path = outdir / "rsh_param_search.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bounds = [
        _parse_float_pair(args.sr_bounds, name="--sr-bounds"),
        _parse_float_pair(args.lr_coordinate_bounds, name="--lr-coordinate-bounds"),
        _parse_float_pair(args.omega_bounds, name="--omega-bounds"),
    ]
    cache: dict[tuple[float, float, float], dict[str, Any]] = {}
    history: list[dict[str, Any]] = []

    def objective(vector: Any) -> float:
        key = tuple(round(float(value), 10) for value in np.asarray(vector, dtype=float))
        if key not in cache:
            try:
                row = _evaluate(vector=vector, args=args)
            except Exception as exc:
                params = _params_from_coordinate(vector)
                row = {
                    "sr_hf_fraction": params.sr_hf_fraction,
                    "lr_hf_fraction": params.lr_hf_fraction,
                    "paper_alpha": params.sr_hf_fraction,
                    "paper_beta": params.paper_beta,
                    "omega": params.omega,
                    "lr_coordinate": params.lr_coordinate,
                    "loss": float(args.nonconverged_penalty),
                    "error": repr(exc),
                    "all_converged": False,
                    "koopmans_ip_residual": float("nan"),
                    "koopmans_ea_residual": float("nan"),
                    "koopmans_lumo_ea_residual": float("nan"),
                    "koopmans_gap_residual": float("nan"),
                    "koopmans_ip_mae": float("nan"),
                    "koopmans_ea_mae": float("nan"),
                    "koopmans_lumo_ea_mae": float("nan"),
                    "koopmans_gap_mae": float("nan"),
                }
            row["eval"] = len(history)
            cache[key] = row
            history.append(row)
            print(
                f"eval={row['eval']:03d} "
                f"sr={row['sr_hf_fraction']:.4f} "
                f"lr={row['lr_hf_fraction']:.4f} "
                f"omega={row['omega']:.4f} "
                f"loss={row['loss']:.6e} "
                f"kip={row['koopmans_ip_mae']:.3e} "
                f"kea={row['koopmans_ea_mae']:.3e} "
                f"conv={row['all_converged']}",
                flush=True,
            )
        return float(cache[key]["loss"])

    from scipy.optimize import differential_evolution, minimize

    de_result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=int(args.maxiter),
        popsize=int(args.popsize),
        seed=int(args.seed),
        polish=False,
        updating="immediate",
        workers=1,
    )
    polish_result = minimize(
        objective,
        np.asarray(de_result.x, dtype=float),
        method="Powell",
        bounds=bounds,
        options={
            "maxiter": int(args.polish_maxiter),
            "xtol": 1e-3,
            "ftol": 1e-8,
            "disp": False,
        },
    )

    best_row = min(history, key=lambda row: float(row["loss"]))
    csv_path = _write_history_csv(history, outdir)
    plot_path = _write_plot(history, outdir)
    summary = {
        "system": "water",
        "backend": "pyscf_define_xc",
        "local_xc": str(args.local_xc),
        "td_graddft_jit": not bool(args.no_jit),
        "basis": str(args.basis),
        "grid_level": int(args.grid_level),
        "search_coordinates": "sr, lr_coordinate, omega with lr=sr+lr_coordinate*(1-sr)",
        "bounds": {
            "sr": list(bounds[0]),
            "lr_coordinate": list(bounds[1]),
            "omega": list(bounds[2]),
        },
        "differential_evolution": {
            "success": bool(de_result.success),
            "message": str(de_result.message),
            "nfev": int(de_result.nfev),
            "x": [float(value) for value in np.asarray(de_result.x).reshape(-1)],
            "fun": float(de_result.fun),
        },
        "powell_polish": {
            "success": bool(polish_result.success),
            "message": str(polish_result.message),
            "nfev": int(polish_result.nfev),
            "x": [float(value) for value in np.asarray(polish_result.x).reshape(-1)],
            "fun": float(polish_result.fun),
        },
        "best_loss": float(best_row["loss"]),
        "best_row": best_row,
        "n_evaluations": int(len(history)),
        "scf_max_cycle": int(args.scf_max_cycle),
        "conv_tol": float(args.conv_tol),
        "koopmans_ip_weight": float(args.koopmans_ip_weight),
        "koopmans_ea_weight": float(args.koopmans_ea_weight),
        "koopmans_lumo_ea_weight": float(args.koopmans_lumo_ea_weight),
        "koopmans_loss_kind": str(args.koopmans_loss_kind),
        "history_csv": str(csv_path),
        "plot": str(plot_path) if plot_path is not None else None,
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
