from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto

from td_graddft import neural_xc

BOHR_TO_ANG = 0.52917721092
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG


def _load_h2_helpers():
    helper_path = Path(__file__).with_name("h2_self_consistent_ground_train5_dense100_vs_fci.py")
    spec = importlib.util.spec_from_file_location("_h2_ground_vs_fci_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load helper module from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare an axial strict MP2 local integrand gauge against the current "
            "projected/scaled PT2 channel used by the Neural_xc implementation."
        )
    )
    p.add_argument("--r-angstrom", type=float, default=0.75)
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--xc", default="b3lyp")
    p.add_argument("--grids-level", type=int, default=0)
    p.add_argument("--max-l", type=int, default=3)
    p.add_argument("--grid-ao-backend", choices=("jax", "pyscf"), default="jax")
    p.add_argument("--integral-backend", choices=("jax", "libcint"), default="libcint")
    p.add_argument("--jk-backend", choices=("full", "df"), default="full")
    p.add_argument("--df-tol", type=float, default=1e-10)
    p.add_argument("--df-max-rank", type=int, default=None)
    p.add_argument("--reference-scf-max-cycle", type=int, default=80)
    p.add_argument("--reference-scf-conv-tol", type=float, default=1e-10)
    p.add_argument("--reference-scf-conv-tol-density", type=float, default=1e-8)
    p.add_argument("--reference-scf-damping", type=float, default=0.15)
    p.add_argument("--reference-scf-potential-clip", type=float, default=20.0)
    p.add_argument("--z-min", type=float, default=-2.0, help="Lower axis bound in Angstrom.")
    p.add_argument("--z-max", type=float, default=2.0, help="Upper axis bound in Angstrom.")
    p.add_argument("--z-points", type=int, default=401)
    p.add_argument(
        "--outdir",
        default="outputs/h2_pt2_axis_profile_compare",
    )
    return p.parse_args()


def _strict_mp2_axis_profiles(
    atom: str,
    basis: str,
    molecule: object,
    *,
    z_angstrom: np.ndarray,
) -> dict[str, np.ndarray | float]:
    rep_tensor = np.asarray(molecule.rep_tensor, dtype=np.float64)
    mo_coeff = np.asarray(molecule.mo_coeff, dtype=np.float64)
    mo_occ = np.asarray(molecule.mo_occ, dtype=np.float64)
    mo_energy = np.asarray(molecule.mo_energy, dtype=np.float64)
    if mo_coeff.ndim == 3:
        mo_coeff = mo_coeff[0]
    if mo_occ.ndim == 2:
        mo_occ = mo_occ[0]
    if mo_energy.ndim == 2:
        mo_energy = mo_energy[0]

    nocc = int(molecule.nocc)
    orbo = mo_coeff[:, :nocc]
    orbv = mo_coeff[:, nocc:]
    eps_occ = mo_energy[:nocc]
    eps_vir = mo_energy[nocc:]

    eri_ovov = np.einsum(
        "pqrs,pi,qa,rj,sb->iajb",
        rep_tensor,
        orbo,
        orbv,
        orbo,
        orbv,
        optimize=True,
    )
    denom = (
        eps_occ[:, None, None, None]
        + eps_occ[None, None, :, None]
        - eps_vir[None, :, None, None]
        - eps_vir[None, None, None, :]
    )
    pair_weights = (2.0 * eri_ovov - np.transpose(eri_ovov, (0, 3, 2, 1))) / denom
    total_energy = float(np.sum(eri_ovov * pair_weights))

    grid_ao = np.asarray(molecule.ao, dtype=np.float64)
    rho_o_grid = np.einsum("rp,pi->ri", grid_ao, orbo, optimize=True)
    rho_v_grid = np.einsum("rp,pa->ra", grid_ao, orbv, optimize=True)
    rho_ov_grid = np.einsum("ri,ra->ria", rho_o_grid, rho_v_grid, optimize=True)
    pair_potential_grid = np.einsum(
        "gp,gq,pqrs,rj,sb->gjb",
        grid_ao,
        grid_ao,
        rep_tensor,
        orbo,
        orbv,
        optimize=True,
    )
    raw_grid = np.einsum(
        "ria,rjb,iajb->r",
        rho_ov_grid,
        pair_potential_grid,
        pair_weights,
        optimize=True,
    )
    grid_weights = np.asarray(molecule.grid.weights, dtype=np.float64)
    projected_energy = float(np.dot(grid_weights, raw_grid))
    scale = total_energy / projected_energy

    mol = gto.M(
        atom=atom,
        basis=basis,
        unit="Angstrom",
        spin=0,
        charge=0,
        cart=True,
        verbose=0,
    )
    coords_ang = np.zeros((int(z_angstrom.size), 3), dtype=np.float64)
    coords_ang[:, 2] = z_angstrom
    ao_axis = mol.eval_gto("GTOval_cart", coords_ang * ANG_TO_BOHR)

    rho_o_axis = np.einsum("rp,pi->ri", ao_axis, orbo, optimize=True)
    rho_v_axis = np.einsum("rp,pa->ra", ao_axis, orbv, optimize=True)
    rho_ov_axis = np.einsum("ri,ra->ria", rho_o_axis, rho_v_axis, optimize=True)
    pair_potential_axis = np.einsum(
        "gp,gq,pqrs,rj,sb->gjb",
        ao_axis,
        ao_axis,
        rep_tensor,
        orbo,
        orbv,
        optimize=True,
    )
    raw_axis = np.einsum(
        "ria,rjb,iajb->r",
        rho_ov_axis,
        pair_potential_axis,
        pair_weights,
        optimize=True,
    )
    projected_axis = scale * raw_axis
    clipped_axis = np.clip(projected_axis, -5.0, 5.0)
    return {
        "z_angstrom": z_angstrom,
        "raw_axis": raw_axis,
        "projected_axis": projected_axis,
        "clipped_axis": clipped_axis,
        "scale": float(scale),
        "total_energy_h": total_energy,
        "grid_projected_energy_h": projected_energy,
        "raw_grid": raw_grid,
    }


def _write_csv(path: Path, profiles: dict[str, np.ndarray | float]) -> None:
    z_ang = np.asarray(profiles["z_angstrom"], dtype=np.float64)
    raw_axis = np.asarray(profiles["raw_axis"], dtype=np.float64)
    projected_axis = np.asarray(profiles["projected_axis"], dtype=np.float64)
    clipped_axis = np.asarray(profiles["clipped_axis"], dtype=np.float64)

    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "z_angstrom,strict_raw_mp2,projected_scaled_mp2,projected_clipped_mp2,"
            "shape_normalized_strict,shape_normalized_projected\n"
        )
        raw_min = float(np.min(raw_axis))
        proj_min = float(np.min(projected_axis))
        for z, raw, proj, clip in zip(z_ang, raw_axis, projected_axis, clipped_axis, strict=False):
            strict_norm = raw / raw_min if raw_min != 0.0 else 0.0
            proj_norm = proj / proj_min if proj_min != 0.0 else 0.0
            handle.write(
                f"{z:.10f},{raw:.16e},{proj:.16e},{clip:.16e},"
                f"{strict_norm:.16e},{proj_norm:.16e}\n"
            )


def _plot_profiles(
    path: Path,
    *,
    r_angstrom: float,
    profiles: dict[str, np.ndarray | float],
) -> None:
    z_ang = np.asarray(profiles["z_angstrom"], dtype=np.float64)
    raw_axis = np.asarray(profiles["raw_axis"], dtype=np.float64)
    projected_axis = np.asarray(profiles["projected_axis"], dtype=np.float64)
    clipped_axis = np.asarray(profiles["clipped_axis"], dtype=np.float64)
    scale = float(profiles["scale"])

    raw_min = float(np.min(raw_axis))
    proj_min = float(np.min(projected_axis))
    strict_norm = raw_axis / raw_min if raw_min != 0.0 else raw_axis
    proj_norm = projected_axis / proj_min if proj_min != 0.0 else projected_axis

    ratio = np.zeros_like(projected_axis)
    mask = np.abs(raw_axis) > 1e-14
    ratio[mask] = projected_axis[mask] / raw_axis[mask]

    fig, axes = plt.subplots(3, 1, figsize=(9.2, 9.5), dpi=180, sharex=True)

    axes[0].plot(z_ang, raw_axis, lw=2.2, color="#111111", label="Strict raw axial MP2 gauge")
    axes[0].plot(z_ang, projected_axis, lw=2.0, color="#d55e00", label="Projected/scaled channel")
    axes[0].plot(z_ang, clipped_axis, lw=1.6, color="#0072b2", ls="--", label="Projected channel after clip")
    axes[0].set_ylabel("Local MP2 Contribution (Eh)")
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].grid(alpha=0.25, linestyle="--", linewidth=0.6)

    axes[1].plot(z_ang, strict_norm, lw=2.2, color="#111111", label="Strict raw (normalized)")
    axes[1].plot(z_ang, proj_norm, lw=2.0, color="#d55e00", label="Projected/scaled (normalized)")
    axes[1].set_ylabel("Shape-Normalized")
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].grid(alpha=0.25, linestyle="--", linewidth=0.6)

    axes[2].plot(z_ang, ratio, lw=2.0, color="#009e73")
    axes[2].axhline(scale, lw=1.0, color="#cc79a7", ls="--", alpha=0.8)
    axes[2].set_ylabel("Projected / Strict")
    axes[2].set_xlabel("z Along H-H Axis (Angstrom)")
    axes[2].grid(alpha=0.25, linestyle="--", linewidth=0.6)

    half_bond = 0.5 * float(r_angstrom)
    for ax in axes:
        ax.axvline(-half_bond, color="#666666", lw=1.0, ls=":", alpha=0.9)
        ax.axvline(half_bond, color="#666666", lw=1.0, ls=":", alpha=0.9)

    fig.suptitle(
        "H2 Axial MP2 Local Correction: Strict Gauge vs Projected PT2 Channel\n"
        f"R = {r_angstrom:.2f} A, nuclei at z = +/-{half_bond:.3f} A, scale = {scale:.6f}"
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    helpers = _load_h2_helpers()
    point, _ = helpers.build_reference_point(
        float(args.r_angstrom),
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
        excited_nstates=3,
        fci_dm0=None,
        compute_local_hfx_features=True,
    )

    z_angstrom = np.linspace(float(args.z_min), float(args.z_max), int(args.z_points), dtype=np.float64)
    profiles = _strict_mp2_axis_profiles(
        point.atom,
        str(args.basis),
        point.molecule,
        z_angstrom=z_angstrom,
    )

    functional = neural_xc.Functional(include_pt2_channel=True)
    projected_grid = np.asarray(functional.projected_pt2_grid_contribution(point.molecule), dtype=np.float64)
    grid_weights = np.asarray(point.molecule.grid.weights, dtype=np.float64)

    csv_path = outdir / "h2_pt2_axis_profile_compare.csv"
    png_path = outdir / "h2_pt2_axis_profile_compare.png"
    json_path = outdir / "h2_pt2_axis_profile_compare.json"

    _write_csv(csv_path, profiles)
    _plot_profiles(png_path, r_angstrom=float(args.r_angstrom), profiles=profiles)

    raw_axis = np.asarray(profiles["raw_axis"], dtype=np.float64)
    projected_axis = np.asarray(profiles["projected_axis"], dtype=np.float64)
    clipped_axis = np.asarray(profiles["clipped_axis"], dtype=np.float64)
    raw_grid = np.asarray(profiles["raw_grid"], dtype=np.float64)
    projected_grid_formula = np.clip(float(profiles["scale"]) * raw_grid, -5.0, 5.0)
    ratio_mask = np.abs(raw_axis) > 1e-14
    ratio = projected_axis[ratio_mask] / raw_axis[ratio_mask]
    summary = {
        "basis": str(args.basis),
        "xc": str(args.xc),
        "r_angstrom": float(args.r_angstrom),
        "z_min_angstrom": float(args.z_min),
        "z_max_angstrom": float(args.z_max),
        "z_points": int(args.z_points),
        "strict_mp2_total_energy_h": float(profiles["total_energy_h"]),
        "grid_integral_of_raw_gauge_h": float(profiles["grid_projected_energy_h"]),
        "projection_scale_factor": float(profiles["scale"]),
        "strict_axis_min_h": float(np.min(raw_axis)),
        "projected_axis_min_h": float(np.min(projected_axis)),
        "clipped_axis_min_h": float(np.min(clipped_axis)),
        "max_abs_difference_projected_vs_clipped_h": float(np.max(np.abs(projected_axis - clipped_axis))),
        "max_abs_difference_grid_impl_vs_formula_h": float(
            np.max(np.abs(projected_grid_formula - projected_grid))
        ),
        "ratio_mean_over_nonzero_points": float(np.mean(ratio)) if ratio.size else None,
        "ratio_std_over_nonzero_points": float(np.std(ratio)) if ratio.size else None,
        "grid_projected_energy_h_from_impl": float(np.dot(grid_weights, projected_grid)),
        "grid_pt2_min_h": float(np.min(projected_grid)),
        "grid_pt2_max_h": float(np.max(projected_grid)),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(csv_path)
    print(png_path)
    print(json_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
