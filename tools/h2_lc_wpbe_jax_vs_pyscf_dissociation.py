from __future__ import annotations

import argparse
import csv
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/td_graddft_matplotlib")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from td_graddft.xc_backend.jax_libxc import (
    RestrictedFeatureBundle,
    eval_xc_energy_density,
)
from td_graddft.nn_rsh import get_rsh_functional_preset, make_pyscf_rsh_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute an H2 LC-wPBE dissociation curve with TD-GradDFT's JAX "
            "local functional and compare against PySCF's built-in LC_WPBE."
        )
    )
    parser.add_argument("--points", type=int, default=100)
    parser.add_argument("--r-min", type=float, default=0.40)
    parser.add_argument("--r-max", type=float, default=5.00)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--grid-level", type=int, default=3)
    parser.add_argument("--conv-tol", type=float, default=1e-10)
    parser.add_argument("--max-cycle", type=int, default=120)
    parser.add_argument("--density-floor", type=float, default=1e-12)
    parser.add_argument(
        "--no-jit",
        action="store_true",
        help="Disable JIT around the TD-GradDFT/JAX local value-gradient kernel.",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/h2_lc_wpbe_jax_vs_pyscf_dissociation",
    )
    return parser.parse_args()


@lru_cache(maxsize=8)
def _restricted_rho_sigma_value_grad_kernel(density_floor: float, use_jit: bool):
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


def make_td_graddft_lc_wpbe_eval_xc(*, omega: float, density_floor: float, use_jit: bool):
    kernel = _restricted_rho_sigma_value_grad_kernel(float(density_floor), bool(use_jit))
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
        if int(spin) != 0:
            raise NotImplementedError("This H2 comparison uses restricted spin=0 RKS only.")
        if int(deriv) > 1:
            raise NotImplementedError("Only deriv<=1 is needed for the RKS dissociation curve.")

        rho_np = np.asarray(rho, dtype=np.float64)
        if rho_np.ndim != 2 or rho_np.shape[0] < 4:
            raise ValueError(f"Expected restricted GGA rho shape (4, ngrids), got {rho_np.shape}.")
        rho_total = np.maximum(rho_np[0], 0.0)
        sigma = np.einsum("xg,xg->g", rho_np[1:4], rho_np[1:4])
        variables = jnp.stack(
            [jnp.asarray(rho_total), jnp.asarray(np.maximum(sigma, 0.0))],
            axis=-1,
        )
        active_omega = omega_value if omega is None else float(omega)
        energy_density, grad = kernel(
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

        vrho = grad_np[:, 0].copy()
        vsigma = grad_np[:, 1].copy()
        vrho[~np.isfinite(vrho)] = 0.0
        vsigma[~np.isfinite(vsigma)] = 0.0
        return exc, (vrho, vsigma, None, None), None, None

    return eval_xc


def build_mol(r_angstrom: float, *, basis: str):
    from pyscf import gto

    return gto.M(
        atom=f"H 0 0 0; H 0 0 {float(r_angstrom):.12f}",
        unit="Angstrom",
        basis=str(basis),
        charge=0,
        spin=0,
        verbose=0,
    )


def run_pyscf_lc_wpbe(
    r_angstrom: float,
    *,
    basis: str,
    grid_level: int,
    conv_tol: float,
    max_cycle: int,
    dm0: np.ndarray | None,
):
    from pyscf import dft

    mol = build_mol(r_angstrom, basis=basis)
    mf = dft.RKS(mol)
    mf.xc = "LC_WPBE"
    mf.grids.level = int(grid_level)
    mf.conv_tol = float(conv_tol)
    mf.max_cycle = int(max_cycle)
    energy = float(mf.kernel(dm0=dm0))
    return mf, energy


def run_td_graddft_lc_wpbe(
    r_angstrom: float,
    *,
    basis: str,
    grid_level: int,
    conv_tol: float,
    max_cycle: int,
    density_floor: float,
    use_jit: bool,
    dm0: np.ndarray | None,
):
    from pyscf import dft

    preset = get_rsh_functional_preset("lc-wpbe")
    params = preset.default_params
    omega = float(params.omega)
    mol = build_mol(r_angstrom, basis=basis)
    mf = dft.RKS(mol)
    mf.grids.level = int(grid_level)
    mf.conv_tol = float(conv_tol)
    mf.max_cycle = int(max_cycle)
    # Keep a PySCF-parseable label for ancillary metadata checks; define_xc_
    # below overrides the actual local evaluator and RSH coefficients.
    mf.xc = "LC_WPBE"
    spec = make_pyscf_rsh_spec(
        xc_description=make_td_graddft_lc_wpbe_eval_xc(
            omega=omega,
            density_floor=float(density_floor),
            use_jit=bool(use_jit),
        ),
        xctype="GGA",
        resolved_params=params,
    )
    spec.install_into_mf(mf)
    energy = float(mf.kernel(dm0=dm0))
    return mf, energy


def _frontier_gap(mf: Any) -> tuple[float, float]:
    occ = np.asarray(mf.mo_occ)
    eps = np.asarray(mf.mo_energy)
    occupied = np.where(occ > 1e-8)[0]
    virtual = np.where(occ <= 1e-8)[0]
    homo = float(eps[occupied[-1]]) if occupied.size else float("nan")
    lumo = float(eps[virtual[0]]) if virtual.size else float("nan")
    return homo, lumo


def write_outputs(rows: list[dict[str, Any]], args: argparse.Namespace, outdir: Path) -> None:
    csv_path = outdir / "h2_lc_wpbe_jax_vs_pyscf_dissociation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    r = np.asarray([row["r_angstrom"] for row in rows], dtype=float)
    pyscf_e = np.asarray([row["pyscf_lc_wpbe_energy_ha"] for row in rows], dtype=float)
    ours_e = np.asarray([row["td_graddft_lc_wpbe_energy_ha"] for row in rows], dtype=float)
    diff = ours_e - pyscf_e

    fig, (ax_curve, ax_diff) = plt.subplots(
        2,
        1,
        figsize=(8.2, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.0]},
    )
    ax_curve.plot(r, pyscf_e, lw=2.0, label="PySCF LC_WPBE")
    ax_curve.plot(r, ours_e, lw=1.8, ls="--", label="TD-GradDFT/JAX LC-wPBE")
    ax_curve.set_ylabel("Total energy (Ha)")
    ax_curve.grid(alpha=0.25)
    ax_curve.legend(frameon=False)
    ax_curve.set_title(f"H2 LC-wPBE Dissociation ({args.basis}, {len(rows)} points)")

    ax_diff.axhline(0.0, color="0.25", lw=1.0)
    ax_diff.plot(r, diff * 1.0e6, color="#b42318", lw=1.8)
    ax_diff.set_xlabel("H-H distance (Angstrom)")
    ax_diff.set_ylabel("Ours - PySCF (uHa)")
    ax_diff.grid(alpha=0.25)

    fig.tight_layout()
    png_path = outdir / "h2_lc_wpbe_jax_vs_pyscf_dissociation.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    summary = {
        "system": "H2",
        "functional": "LC-wPBE",
        "comparison": "TD-GradDFT/JAX lc_wpbe_local + PySCF RSH coefficients vs PySCF LC_WPBE",
        "basis": str(args.basis),
        "grid_level": int(args.grid_level),
        "td_graddft_jit": not bool(args.no_jit),
        "points": int(len(rows)),
        "r_min_angstrom": float(np.min(r)),
        "r_max_angstrom": float(np.max(r)),
        "max_abs_diff_ha": float(np.max(np.abs(diff))),
        "mae_diff_ha": float(np.mean(np.abs(diff))),
        "max_abs_diff_microha": float(np.max(np.abs(diff)) * 1.0e6),
        "mae_diff_microha": float(np.mean(np.abs(diff)) * 1.0e6),
        "pyscf_converged_points": int(sum(bool(row["pyscf_converged"]) for row in rows)),
        "td_graddft_converged_points": int(
            sum(bool(row["td_graddft_converged"]) for row in rows)
        ),
        "csv": str(csv_path),
        "plot": str(png_path),
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    if int(args.points) < 2:
        raise ValueError("--points must be at least 2.")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    distances = np.linspace(float(args.r_min), float(args.r_max), int(args.points))
    rows: list[dict[str, Any]] = []
    pyscf_dm0: np.ndarray | None = None
    ours_dm0: np.ndarray | None = None

    for index, r_angstrom in enumerate(distances, start=1):
        pyscf_mf, pyscf_energy = run_pyscf_lc_wpbe(
            float(r_angstrom),
            basis=str(args.basis),
            grid_level=int(args.grid_level),
            conv_tol=float(args.conv_tol),
            max_cycle=int(args.max_cycle),
            dm0=pyscf_dm0,
        )
        ours_mf, ours_energy = run_td_graddft_lc_wpbe(
            float(r_angstrom),
            basis=str(args.basis),
            grid_level=int(args.grid_level),
            conv_tol=float(args.conv_tol),
            max_cycle=int(args.max_cycle),
            density_floor=float(args.density_floor),
            use_jit=not bool(args.no_jit),
            dm0=ours_dm0,
        )
        pyscf_dm0 = np.asarray(pyscf_mf.make_rdm1())
        ours_dm0 = np.asarray(ours_mf.make_rdm1())
        pyscf_homo, pyscf_lumo = _frontier_gap(pyscf_mf)
        ours_homo, ours_lumo = _frontier_gap(ours_mf)
        diff = ours_energy - pyscf_energy
        rows.append(
            {
                "point": index,
                "r_angstrom": float(r_angstrom),
                "pyscf_lc_wpbe_energy_ha": pyscf_energy,
                "td_graddft_lc_wpbe_energy_ha": ours_energy,
                "diff_ha": float(diff),
                "diff_microha": float(diff * 1.0e6),
                "pyscf_converged": bool(pyscf_mf.converged),
                "td_graddft_converged": bool(ours_mf.converged),
                "pyscf_homo_ha": pyscf_homo,
                "pyscf_lumo_ha": pyscf_lumo,
                "td_graddft_homo_ha": ours_homo,
                "td_graddft_lumo_ha": ours_lumo,
            }
        )
        print(
            f"{index:3d}/{len(distances)} R={r_angstrom:.4f} A "
            f"PySCF={pyscf_energy:.12f} TD-GradDFT={ours_energy:.12f} "
            f"diff={diff * 1.0e6:.3f} uHa",
            flush=True,
        )

    write_outputs(rows, args, outdir)


if __name__ == "__main__":
    main()
