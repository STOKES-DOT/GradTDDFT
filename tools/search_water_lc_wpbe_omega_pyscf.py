from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


WATER_GEOMETRY = """
O  0.000000  0.000000  0.117790
H  0.000000  0.755453 -0.471161
H  0.000000 -0.755453 -0.471161
"""


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
            "Search LC-wPBE omega with PySCF's native RSH path and a "
            "Koopmans HOMO/IP-EA objective on water."
        ),
    )
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--xc", default="LC_WPBE")
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument("--omega-bounds", default="0.13,0.30")
    parser.add_argument("--baseline-omegas", default="0.4")
    parser.add_argument("--xatol", type=float, default=1e-3)
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--scf-max-cycle", type=int, default=50)
    parser.add_argument("--conv-tol", type=float, default=1e-9)
    parser.add_argument("--occupation-tol", type=float, default=1e-8)
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
        default="outputs/water_lc_wpbe_omega_pyscf_search",
    )
    return parser.parse_args()


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


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


def _run_pyscf_state(
    *,
    basis: str,
    xc: str,
    omega: float,
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
    mf.xc = str(xc)
    mf.grids.level = int(grid_level)
    mf.max_cycle = int(scf_max_cycle)
    mf.conv_tol = float(conv_tol)
    # PySCF uses NumInt.omega to override the RSH exponent consistently in
    # libxc evaluation and in the attenuated exact-exchange build.
    mf._numint.omega = float(omega)
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


def _evaluate_omega(
    *,
    omega: float,
    basis: str,
    xc: str,
    grid_level: int,
    scf_max_cycle: int,
    conv_tol: float,
    occupation_tol: float,
    koopmans_ip_weight: float,
    koopmans_ea_weight: float,
    koopmans_lumo_ea_weight: float,
    koopmans_loss_kind: str,
) -> dict[str, Any]:
    neutral = _run_pyscf_state(
        basis=basis,
        xc=xc,
        omega=omega,
        charge=0,
        spin=0,
        grid_level=grid_level,
        scf_max_cycle=scf_max_cycle,
        conv_tol=conv_tol,
        occupation_tol=occupation_tol,
    )
    cation = _run_pyscf_state(
        basis=basis,
        xc=xc,
        omega=omega,
        charge=1,
        spin=1,
        grid_level=grid_level,
        scf_max_cycle=scf_max_cycle,
        conv_tol=conv_tol,
        occupation_tol=occupation_tol,
    )
    anion = _run_pyscf_state(
        basis=basis,
        xc=xc,
        omega=omega,
        charge=-1,
        spin=1,
        grid_level=grid_level,
        scf_max_cycle=scf_max_cycle,
        conv_tol=conv_tol,
        occupation_tol=occupation_tol,
    )

    ip_residual = neutral.homo + cation.energy - neutral.energy
    ea_residual = anion.homo + neutral.energy - anion.energy
    lumo_ea_residual = neutral.lumo + neutral.energy - anion.energy
    gap_residual = (neutral.lumo - neutral.homo) - (
        cation.energy + anion.energy - 2.0 * neutral.energy
    )
    loss = (
        float(koopmans_ip_weight) * _penalty(ip_residual, kind=koopmans_loss_kind)
        + float(koopmans_ea_weight) * _penalty(ea_residual, kind=koopmans_loss_kind)
        + float(koopmans_lumo_ea_weight)
        * _penalty(lumo_ea_residual, kind=koopmans_loss_kind)
    )
    return {
        "omega": float(omega),
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
        "all_converged": bool(neutral.converged and cation.converged and anion.converged),
    }


def _bounded_omega_search(
    objective: Callable[[float], float],
    *,
    bounds: tuple[float, float],
    xatol: float,
    maxiter: int,
) -> tuple[Any, list[dict[str, float]]]:
    from scipy.optimize import minimize_scalar

    history: list[dict[str, float]] = []

    def wrapped(omega: float) -> float:
        omega_value = float(omega)
        loss = float(objective(omega_value))
        history.append(
            {
                "eval": float(len(history)),
                "omega": omega_value,
                "loss": loss,
            }
        )
        return loss

    result = minimize_scalar(
        wrapped,
        bounds=(float(bounds[0]), float(bounds[1])),
        method="bounded",
        options={"xatol": float(xatol), "maxiter": int(maxiter)},
    )
    return result, history


def _write_plot(history: list[dict[str, Any]], outdir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    evals = [int(row["eval"]) for row in history]
    losses = [float(row["loss"]) for row in history]
    omegas = [float(row["omega"]) for row in history]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
    axes[0].plot(evals, losses, "-o", label="loss")
    axes[0].set_ylabel("PySCF Koopmans loss")
    axes[0].grid(alpha=0.25, linewidth=0.6)
    axes[0].legend(frameon=False)
    axes[1].plot(evals, omegas, "-o", color="#b23a48", label="omega")
    axes[1].set_xlabel("scipy objective evaluation")
    axes[1].set_ylabel("omega")
    axes[1].grid(alpha=0.25, linewidth=0.6)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    path = outdir / "omega_pyscf_search.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    bounds_raw = _parse_float_tuple(str(args.omega_bounds))
    if len(bounds_raw) != 2:
        raise ValueError("--omega-bounds must contain exactly lower,upper.")
    bounds = (float(bounds_raw[0]), float(bounds_raw[1]))
    cache: dict[float, dict[str, Any]] = {}

    def evaluate(omega: float) -> dict[str, Any]:
        key = round(float(omega), 10)
        if key not in cache:
            row = _evaluate_omega(
                omega=float(omega),
                basis=str(args.basis),
                xc=str(args.xc),
                grid_level=int(args.grid_level),
                scf_max_cycle=int(args.scf_max_cycle),
                conv_tol=float(args.conv_tol),
                occupation_tol=float(args.occupation_tol),
                koopmans_ip_weight=float(args.koopmans_ip_weight),
                koopmans_ea_weight=float(args.koopmans_ea_weight),
                koopmans_lumo_ea_weight=float(args.koopmans_lumo_ea_weight),
                koopmans_loss_kind=str(args.koopmans_loss_kind),
            )
            cache[key] = row
            print(
                f"eval={len(cache):03d} omega={row['omega']:.6f} "
                f"loss={row['loss']:.6e} "
                f"kip={row['koopmans_ip_mae']:.3e} "
                f"kea={row['koopmans_ea_mae']:.3e} "
                f"conv={row['all_converged']}",
                flush=True,
            )
        return cache[key]

    def objective(omega: float) -> float:
        return float(evaluate(float(omega))["loss"])

    baselines: list[dict[str, Any]] = []
    for omega in _parse_float_tuple(str(args.baseline_omegas)):
        row = dict(evaluate(float(omega)))
        row["label"] = f"baseline_omega_{omega:g}"
        baselines.append(row)

    result, scipy_history = _bounded_omega_search(
        objective,
        bounds=bounds,
        xatol=float(args.xatol),
        maxiter=int(args.maxiter),
    )
    history: list[dict[str, Any]] = []
    for scipy_row in scipy_history:
        key = round(float(scipy_row["omega"]), 10)
        row = dict(cache[key])
        row["eval"] = int(scipy_row["eval"])
        history.append(row)

    best_key = min(cache, key=lambda key: float(cache[key]["loss"]))
    best_row = cache[best_key]
    plot_path = _write_plot(history, outdir)
    summary = {
        "backend": "pyscf",
        "rsh_entry": "mf.xc=LC_WPBE with mf._numint.omega override",
        "xc": str(args.xc),
        "basis": str(args.basis),
        "grid_level": int(args.grid_level),
        "bounds": [float(bounds[0]), float(bounds[1])],
        "xatol": float(args.xatol),
        "maxiter": int(args.maxiter),
        "scf_max_cycle": int(args.scf_max_cycle),
        "conv_tol": float(args.conv_tol),
        "scipy_success": bool(result.success),
        "scipy_message": str(result.message),
        "scipy_nfev": int(result.nfev),
        "scipy_x": float(result.x),
        "scipy_fun": float(result.fun),
        "best_loss": float(best_row["loss"]),
        "best_omega": float(best_row["omega"]),
        "best_row": best_row,
        "baselines": baselines,
        "history": history,
        "koopmans_ip_weight": float(args.koopmans_ip_weight),
        "koopmans_ea_weight": float(args.koopmans_ea_weight),
        "koopmans_lumo_ea_weight": float(args.koopmans_lumo_ea_weight),
        "koopmans_loss_kind": str(args.koopmans_loss_kind),
        "plot": str(plot_path) if plot_path is not None else None,
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"wrote {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
