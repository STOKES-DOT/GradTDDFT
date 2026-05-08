from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from td_graddft_tools.geomopt_freq import (
    GeometryOptimizationConfig,
    make_excited_state_surface,
    make_ground_state_surface,
    run_geometry_optimization,
)


def _build_h2_mf(r_angstrom: float, *, basis: str, xc: str):
    from pyscf import dft, gto

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

    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"PySCF SCF did not converge at R={r_angstrom:.3f} Angstrom.")
    return mf


def _scan_h2_curve(
    r_grid: np.ndarray,
    *,
    basis: str,
    xc: str,
) -> tuple[np.ndarray, np.ndarray]:
    e0 = np.zeros_like(r_grid)
    e1 = np.zeros_like(r_grid)
    for i, r in enumerate(r_grid):
        mf = _build_h2_mf(float(r), basis=basis, xc=xc)
        e0_i = float(mf.e_tot)
        td = mf.TDA()
        td.nstates = 3
        td.kernel()
        if len(td.e) < 1:
            raise RuntimeError(f"TDA returned no excited roots at R={r:.3f} Angstrom.")
        e1_i = e0_i + float(td.e[0])
        e0[i] = e0_i
        e1[i] = e1_i
    return e0, e1


def _interp_surface(
    r_grid: np.ndarray,
    values: np.ndarray,
    *,
    penalty_scale: float = 50.0,
):
    r = jnp.asarray(r_grid)
    v = jnp.asarray(values)
    r_min = float(np.min(r_grid))
    r_max = float(np.max(r_grid))

    def surface_from_coords(coords: jnp.ndarray) -> jnp.ndarray:
        bond = jnp.linalg.norm(coords[1] - coords[0])
        inside = jnp.interp(bond, r, v)
        # Softly keep optimization within the scan window.
        boundary = penalty_scale * (
            jax.nn.relu(r_min - bond) ** 2 + jax.nn.relu(bond - r_max) ** 2
        )
        return inside + boundary

    return surface_from_coords


def _bond_length(coords: jnp.ndarray) -> float:
    return float(jnp.linalg.norm(coords[1] - coords[0]))


def _write_curve_csv(
    path: Path,
    r_grid: np.ndarray,
    e0: np.ndarray,
    e1: np.ndarray,
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "R_Angstrom",
                "E0_Hartree",
                "E1_Hartree",
                "Excitation1_Hartree",
            ]
        )
        for r, g, ex in zip(r_grid, e0, e1, strict=True):
            writer.writerow([float(r), float(g), float(ex), float(ex - g)])


def _write_summary(
    path: Path,
    *,
    basis: str,
    xc: str,
    r0_opt: float,
    r1_opt: float,
    e0_opt: float,
    e1_opt: float,
    coords0: jnp.ndarray,
    coords1: jnp.ndarray,
) -> None:
    with path.open("w") as f:
        f.write("H2 dissociation + geometry optimization summary\n")
        f.write(f"method = {xc}/{basis}\n")
        f.write(f"ground optimum R (A) = {r0_opt:.8f}\n")
        f.write(f"excited-1 optimum R (A) = {r1_opt:.8f}\n")
        f.write(f"ground optimum E (Ha) = {e0_opt:.10f}\n")
        f.write(f"excited-1 optimum E (Ha) = {e1_opt:.10f}\n")
        f.write("\nGround optimized coordinates (Angstrom):\n")
        f.write(np.array2string(np.asarray(coords0), precision=8))
        f.write("\n\nExcited-1 optimized coordinates (Angstrom):\n")
        f.write(np.array2string(np.asarray(coords1), precision=8))
        f.write("\n")


def _plot_curves(
    path: Path,
    r_grid: np.ndarray,
    e0: np.ndarray,
    e1: np.ndarray,
    *,
    r0_opt: float,
    r1_opt: float,
    e0_opt: float,
    e1_opt: float,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(r_grid, e0, lw=2.0, label="Ground State (E0)")
    ax.plot(r_grid, e1, lw=2.0, label="First Excited State (E1)")

    ax.scatter([r0_opt], [e0_opt], s=60, zorder=4)
    ax.scatter([r1_opt], [e1_opt], s=60, zorder=4)
    ax.annotate(
        f"GS optimum\nR={r0_opt:.3f} A",
        xy=(r0_opt, e0_opt),
        xytext=(12, 10),
        textcoords="offset points",
        fontsize=9,
    )
    ax.annotate(
        f"ES1 optimum\nR={r1_opt:.3f} A",
        xy=(r1_opt, e1_opt),
        xytext=(12, -28),
        textcoords="offset points",
        fontsize=9,
    )

    ax.set_xlabel("H-H Distance (Angstrom)")
    ax.set_ylabel("Total Energy (Hartree)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H2 ground/excited dissociation curves with geometry optimization markers.",
    )
    parser.add_argument("--basis", type=str, default="sto-3g")
    parser.add_argument("--xc", type=str, default="b3lyp")
    parser.add_argument("--r-min", type=float, default=0.4)
    parser.add_argument("--r-max", type=float, default=3.0)
    parser.add_argument("--points", type=int, default=31)
    parser.add_argument("--outdir", type=str, default="outputs/h2_b3lyp_sto3g_dissociation")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / ".mplconfig").mkdir(parents=True, exist_ok=True)

    r_grid = np.linspace(args.r_min, args.r_max, args.points)
    e0, e1 = _scan_h2_curve(r_grid, basis=args.basis, xc=args.xc)
    d1 = e1 - e0

    ground_surface = make_ground_state_surface(_interp_surface(r_grid, e0), label="h2_ground")
    excited_surface = make_excited_state_surface(
        ground_energy_fn=_interp_surface(r_grid, e0),
        excitation_energy_fn=lambda coords: jnp.asarray(
            [jnp.interp(jnp.linalg.norm(coords[1] - coords[0]), jnp.asarray(r_grid), jnp.asarray(d1))]
        ),
        state_index=0,
        label="h2_excited_1",
    )

    initial_coords = jnp.asarray(
        [
            [0.0, 0.0, -0.8],
            [0.0, 0.0, +0.8],
        ]
    )
    opt_cfg = GeometryOptimizationConfig(
        max_steps=400,
        learning_rate=0.04,
        convergence_grad_norm=1e-8,
        convergence_step_norm=1e-8,
    )
    opt_ground = run_geometry_optimization(ground_surface, initial_coords, opt_cfg)
    opt_excited = run_geometry_optimization(excited_surface, initial_coords, opt_cfg)

    r0_opt = _bond_length(opt_ground.optimized_coordinates)
    r1_opt = _bond_length(opt_excited.optimized_coordinates)
    e0_opt = float(np.interp(r0_opt, r_grid, e0))
    e1_opt = float(np.interp(r1_opt, r_grid, e1))

    csv_path = outdir / "h2_dissociation_curve.csv"
    png_path = outdir / "h2_dissociation_curve.png"
    summary_path = outdir / "summary.txt"

    _write_curve_csv(csv_path, r_grid, e0, e1)
    _plot_curves(
        png_path,
        r_grid,
        e0,
        e1,
        r0_opt=r0_opt,
        r1_opt=r1_opt,
        e0_opt=e0_opt,
        e1_opt=e1_opt,
        title=f"H2 Dissociation Curves ({args.xc}/{args.basis})",
    )
    _write_summary(
        summary_path,
        basis=args.basis,
        xc=args.xc,
        r0_opt=r0_opt,
        r1_opt=r1_opt,
        e0_opt=e0_opt,
        e1_opt=e1_opt,
        coords0=opt_ground.optimized_coordinates,
        coords1=opt_excited.optimized_coordinates,
    )

    print(f"Wrote curve table to: {csv_path}")
    print(f"Wrote curve plot  to: {png_path}")
    print(f"Wrote summary     to: {summary_path}")


if __name__ == "__main__":
    main()
