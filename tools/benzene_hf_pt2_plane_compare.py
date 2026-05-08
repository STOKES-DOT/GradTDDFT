from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pyscf import dft, gto

from td_graddft import neural_xc
from td_graddft.reference_legacy import restricted_reference_from_pyscf

ANG_TO_BOHR = 1.0 / 0.52917721092


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare strict raw HF/MP2 local gauges against the projected/scaled "
            "HF and PT2 channels on the benzene molecular plane."
        )
    )
    p.add_argument("--basis", default="6-31g")
    p.add_argument("--plane-min", type=float, default=-3.2)
    p.add_argument("--plane-max", type=float, default=3.2)
    p.add_argument("--plane-points", type=int, default=121)
    p.add_argument("--grids-level", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=256)
    p.add_argument(
        "--no-clip",
        action="store_true",
        help="Disable clipping of the projected HF/PT2 channels.",
    )
    p.add_argument("--outdir", default="outputs/benzene_hf_pt2_plane_compare")
    return p.parse_args()


def benzene_atom_string(*, cc_bond: float = 1.397, ch_bond: float = 1.089) -> str:
    atoms: list[str] = []
    hydrogen_radius = cc_bond + ch_bond
    for idx in range(6):
        angle = np.deg2rad(60.0 * idx)
        cx = cc_bond * np.cos(angle)
        cy = cc_bond * np.sin(angle)
        hx = hydrogen_radius * np.cos(angle)
        hy = hydrogen_radius * np.sin(angle)
        atoms.append(f"C {cx:.12f} {cy:.12f} 0.000000000000")
        atoms.append(f"H {hx:.12f} {hy:.12f} 0.000000000000")
    return "; ".join(atoms)


def build_benzene_reference(*, basis: str, grids_level: int):
    atom = benzene_atom_string()
    mol = gto.M(
        atom=atom,
        basis=basis,
        unit="Angstrom",
        spin=0,
        charge=0,
        cart=True,
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = "hf"
    mf.grids.level = int(grids_level)
    mf.conv_tol = 1e-10
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF HF reference for benzene did not converge.")
    ref = restricted_reference_from_pyscf(
        mf,
        compute_local_hfx_features=True,
        compute_local_hfx_aux=True,
    )
    return atom, mol, mf, ref


def _hf_projection_data(ref, *, clip: float | None = 5.0) -> dict[str, np.ndarray | float]:
    rep = np.asarray(ref.rep_tensor, dtype=np.float64)
    rdm1 = np.asarray(ref.rdm1, dtype=np.float64)
    ao_grid = np.asarray(ref.ao, dtype=np.float64)
    weights = np.asarray(ref.grid.weights, dtype=np.float64)

    if rdm1.ndim == 2:
        rdm1 = np.stack([0.5 * rdm1, 0.5 * rdm1], axis=0)

    exchange_mats = np.einsum("prqs,xrs->xpq", rep, rdm1, optimize=True)
    raw_grid_spin = -0.5 * np.einsum("rp,xpq,rq->xr", ao_grid, exchange_mats, ao_grid, optimize=True)
    raw_grid = np.sum(raw_grid_spin, axis=0)
    exact_energy = float(-0.5 * np.einsum("xpr,xpr->", rdm1, exchange_mats, optimize=True))
    projected_energy = float(np.dot(weights, raw_grid))
    scale = exact_energy / projected_energy
    projected_grid = scale * raw_grid
    if clip is not None:
        projected_grid = np.clip(projected_grid, -clip, clip)

    return {
        "exchange_mats": exchange_mats,
        "raw_grid": raw_grid,
        "exact_energy_h": exact_energy,
        "projected_energy_h": projected_energy,
        "scale": float(scale),
        "projected_grid": projected_grid,
    }


def _pt2_projection_data(ref, *, clip: float | None = 5.0) -> dict[str, np.ndarray | float]:
    rep = np.asarray(ref.rep_tensor, dtype=np.float64)
    ao_grid = np.asarray(ref.ao, dtype=np.float64)
    weights = np.asarray(ref.grid.weights, dtype=np.float64)
    mo_coeff = np.asarray(ref.mo_coeff, dtype=np.float64)
    mo_occ = np.asarray(ref.mo_occ, dtype=np.float64)
    mo_energy = np.asarray(ref.mo_energy, dtype=np.float64)
    if mo_coeff.ndim == 3:
        mo_coeff = mo_coeff[0]
    if mo_occ.ndim == 2:
        mo_occ = mo_occ[0]
    if mo_energy.ndim == 2:
        mo_energy = mo_energy[0]

    nocc = int(ref.nocc)
    orbo = mo_coeff[:, :nocc]
    orbv = mo_coeff[:, nocc:]
    eps_occ = mo_energy[:nocc]
    eps_vir = mo_energy[nocc:]

    eri_ovov = np.einsum(
        "pqrs,pi,qa,rj,sb->iajb",
        rep,
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

    rho_o_grid = np.einsum("rp,pi->ri", ao_grid, orbo, optimize=True)
    rho_v_grid = np.einsum("rp,pa->ra", ao_grid, orbv, optimize=True)
    rho_ov_grid = np.einsum("ri,ra->ria", rho_o_grid, rho_v_grid, optimize=True)

    pqjb = np.einsum("pqrs,rj,sb->pqjb", rep, orbo, orbv, optimize=True)
    pair_potential_grid = np.einsum(
        "gp,gq,pqjb->gjb",
        ao_grid,
        ao_grid,
        pqjb,
        optimize=True,
    )
    raw_grid = np.einsum(
        "ria,rjb,iajb->r",
        rho_ov_grid,
        pair_potential_grid,
        pair_weights,
        optimize=True,
    )
    projected_energy = float(np.dot(weights, raw_grid))
    scale = total_energy / projected_energy
    projected_grid = scale * raw_grid
    if clip is not None:
        projected_grid = np.clip(projected_grid, -clip, clip)

    return {
        "orbo": orbo,
        "orbv": orbv,
        "pair_weights": pair_weights,
        "pqjb": pqjb,
        "raw_grid": raw_grid,
        "total_energy_h": total_energy,
        "projected_energy_h": projected_energy,
        "scale": float(scale),
        "projected_grid": projected_grid,
    }


def _plane_coordinates(plane_min: float, plane_max: float, plane_points: int):
    x = np.linspace(plane_min, plane_max, plane_points, dtype=np.float64)
    y = np.linspace(plane_min, plane_max, plane_points, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    coords_ang = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size, dtype=np.float64)], axis=1)
    return x, y, xx, yy, coords_ang


def _hf_plane_profiles(
    mol,
    ref,
    plane_coords_ang: np.ndarray,
    hf_data: dict[str, np.ndarray | float],
    *,
    chunk_size: int,
    clip: float | None = 5.0,
):
    ao_chunks: list[np.ndarray] = []
    raw_chunks: list[np.ndarray] = []
    projected_chunks: list[np.ndarray] = []
    exchange_mats = np.asarray(hf_data["exchange_mats"], dtype=np.float64)
    scale = float(hf_data["scale"])

    for start in range(0, plane_coords_ang.shape[0], chunk_size):
        stop = min(start + chunk_size, plane_coords_ang.shape[0])
        ao_chunk = mol.eval_gto("GTOval_cart", plane_coords_ang[start:stop] * ANG_TO_BOHR)
        raw_spin = -0.5 * np.einsum("gp,xpq,gq->xg", ao_chunk, exchange_mats, ao_chunk, optimize=True)
        raw_chunk = np.sum(raw_spin, axis=0)
        ao_chunks.append(ao_chunk)
        raw_chunks.append(raw_chunk)
        projected_chunk = scale * raw_chunk
        if clip is not None:
            projected_chunk = np.clip(projected_chunk, -clip, clip)
        projected_chunks.append(projected_chunk)

    return np.concatenate(raw_chunks), np.concatenate(projected_chunks), np.concatenate(ao_chunks, axis=0)


def _pt2_plane_profiles(
    plane_ao: np.ndarray,
    pt2_data: dict[str, np.ndarray | float],
    *,
    chunk_size: int,
    clip: float | None = 5.0,
):
    orbo = np.asarray(pt2_data["orbo"], dtype=np.float64)
    orbv = np.asarray(pt2_data["orbv"], dtype=np.float64)
    pair_weights = np.asarray(pt2_data["pair_weights"], dtype=np.float64)
    pqjb = np.asarray(pt2_data["pqjb"], dtype=np.float64)
    scale = float(pt2_data["scale"])

    raw_chunks: list[np.ndarray] = []
    projected_chunks: list[np.ndarray] = []
    ngrid = int(plane_ao.shape[0])
    for start in range(0, ngrid, chunk_size):
        stop = min(start + chunk_size, ngrid)
        ao_chunk = plane_ao[start:stop]
        rho_o = np.einsum("gp,pi->gi", ao_chunk, orbo, optimize=True)
        rho_v = np.einsum("gp,pa->ga", ao_chunk, orbv, optimize=True)
        rho_ov = np.einsum("gi,ga->gia", rho_o, rho_v, optimize=True)
        pair_potential = np.einsum("gp,gq,pqjb->gjb", ao_chunk, ao_chunk, pqjb, optimize=True)
        raw_chunk = np.einsum("gia,gjb,iajb->g", rho_ov, pair_potential, pair_weights, optimize=True)
        raw_chunks.append(raw_chunk)
        projected_chunk = scale * raw_chunk
        if clip is not None:
            projected_chunk = np.clip(projected_chunk, -clip, clip)
        projected_chunks.append(projected_chunk)
    return np.concatenate(raw_chunks), np.concatenate(projected_chunks)


def _shape_normalized_difference(raw: np.ndarray, projected: np.ndarray) -> np.ndarray:
    raw_min = float(np.min(raw))
    proj_min = float(np.min(projected))
    if raw_min == 0.0 or proj_min == 0.0:
        return np.zeros_like(raw)
    return projected / proj_min - raw / raw_min


def _plot_row(fig, axes, *, xx, yy, nuclei_xy, raw, projected, diff, label: str, scale: float):
    cmap_main = "magma"
    cmap_diff = "coolwarm"
    vmin_main = min(float(np.min(raw)), float(np.min(projected)))
    vmax_main = max(float(np.max(raw)), float(np.max(projected)))
    if abs(vmax_main - vmin_main) < 1e-14:
        vmax_main = vmin_main + 1e-12
    diff_abs = max(float(np.max(np.abs(diff))), 1e-12)

    images = [
        axes[0].imshow(raw, origin="lower", extent=[xx.min(), xx.max(), yy.min(), yy.max()], cmap=cmap_main, vmin=vmin_main, vmax=vmax_main),
        axes[1].imshow(projected, origin="lower", extent=[xx.min(), xx.max(), yy.min(), yy.max()], cmap=cmap_main, vmin=vmin_main, vmax=vmax_main),
        axes[2].imshow(diff, origin="lower", extent=[xx.min(), xx.max(), yy.min(), yy.max()], cmap=cmap_diff, vmin=-diff_abs, vmax=diff_abs),
    ]
    titles = [
        f"{label}: strict raw gauge",
        f"{label}: projected/scaled",
        f"{label}: normalized shape diff",
    ]
    for ax, im, title in zip(axes, images, titles, strict=False):
        ax.set_title(title, fontsize=10)
        ax.scatter(nuclei_xy[:, 0], nuclei_xy[:, 1], s=12, c="cyan", edgecolors="black", linewidths=0.4)
        ax.set_aspect("equal")
        ax.set_xlabel("x (Angstrom)")
        ax.set_ylabel("y (Angstrom)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[1].text(
        0.02,
        0.98,
        f"scale = {scale:.6f}",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55),
    )


def _plot_error_row(fig, axes, *, xx, yy, nuclei_xy, raw, projected, error, label: str, scale: float):
    cmap_main = "magma"
    cmap_err = "coolwarm"
    vmin_main = min(float(np.min(raw)), float(np.min(projected)))
    vmax_main = max(float(np.max(raw)), float(np.max(projected)))
    if abs(vmax_main - vmin_main) < 1e-14:
        vmax_main = vmin_main + 1e-12
    err_abs = max(float(np.max(np.abs(error))), 1e-12)

    images = [
        axes[0].imshow(raw, origin="lower", extent=[xx.min(), xx.max(), yy.min(), yy.max()], cmap=cmap_main, vmin=vmin_main, vmax=vmax_main),
        axes[1].imshow(projected, origin="lower", extent=[xx.min(), xx.max(), yy.min(), yy.max()], cmap=cmap_main, vmin=vmin_main, vmax=vmax_main),
        axes[2].imshow(error, origin="lower", extent=[xx.min(), xx.max(), yy.min(), yy.max()], cmap=cmap_err, vmin=-err_abs, vmax=err_abs),
    ]
    titles = [
        f"{label}: strict raw gauge",
        f"{label}: projected/scaled",
        f"{label}: projection error",
    ]
    for ax, im, title in zip(axes, images, titles, strict=False):
        ax.set_title(title, fontsize=10)
        ax.scatter(nuclei_xy[:, 0], nuclei_xy[:, 1], s=12, c="cyan", edgecolors="black", linewidths=0.4)
        ax.set_aspect("equal")
        ax.set_xlabel("x (Angstrom)")
        ax.set_ylabel("y (Angstrom)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[1].text(
        0.02,
        0.98,
        f"scale = {scale:.6f}",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55),
    )


def _plot_axis_profiles(
    *,
    x: np.ndarray,
    hf_raw: np.ndarray,
    hf_projected: np.ndarray,
    pt2_raw: np.ndarray,
    pt2_projected: np.ndarray,
    carbon_x: np.ndarray,
    scale_hf: float,
    scale_pt2: float,
    clip: float | None,
    outpath: Path,
):
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.6), dpi=200, sharex=True)
    series = [
        (axes[0], hf_raw, hf_projected, "HF exchange", scale_hf, "#1f77b4", "#d62728"),
        (axes[1], pt2_raw, pt2_projected, "MP2 correlation", scale_pt2, "#2ca02c", "#ff7f0e"),
    ]

    for ax, raw, projected, title, scale, raw_color, proj_color in series:
        ax.plot(x, raw, color=raw_color, lw=2.0, label="strict raw gauge")
        ax.plot(x, projected, color=proj_color, lw=2.0, ls="--", label="projected/scaled")
        for idx, xpos in enumerate(carbon_x):
            ax.axvline(xpos, color="black", lw=1.0, ls=":", alpha=0.7)
            ax.text(
                xpos,
                0.98,
                "C",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9,
                color="black",
            )
        ax.set_ylabel("Local contribution (Eh)")
        ax.set_title(f"{title} along the C-C symmetry axis", fontsize=11)
        ax.grid(alpha=0.2, linewidth=0.7)
        ax.legend(frameon=False, loc="best")
        ax.text(
            0.02,
            0.96,
            f"scale = {scale:.6f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
        )

    clip_text = "no clip" if clip is None else f"clip = {clip:.1f}"
    axes[1].set_xlabel("x along symmetry axis (Angstrom)")
    fig.suptitle(
        "Benzene HF/PT2 Profiles Along the Symmetry Axis Through Opposite Carbon Atoms\n"
        f"Reference: restricted HF / 6-31G, y = 0 A, {clip_text}"
    )
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def _plot_axis_errors(
    *,
    x: np.ndarray,
    hf_error: np.ndarray,
    pt2_error: np.ndarray,
    carbon_x: np.ndarray,
    clip: float | None,
    outpath: Path,
):
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.8), dpi=200, sharex=True)
    series = [
        (axes[0], hf_error, "HF exchange projection error", "#d62728"),
        (axes[1], pt2_error, "MP2 correlation projection error", "#ff7f0e"),
    ]
    for ax, err, title, color in series:
        ax.plot(x, err, color=color, lw=2.0)
        ax.axhline(0.0, color="black", lw=1.0, ls="--", alpha=0.7)
        for xpos in carbon_x:
            ax.axvline(xpos, color="black", lw=1.0, ls=":", alpha=0.7)
            ax.text(
                xpos,
                0.98,
                "C",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9,
                color="black",
            )
        ax.set_ylabel("Projected - raw (Eh)")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.2, linewidth=0.7)
    clip_text = "no clip" if clip is None else f"clip = {clip:.1f}"
    axes[1].set_xlabel("x along symmetry axis (Angstrom)")
    fig.suptitle(
        "Benzene Projection Error Along the Symmetry Axis Through Opposite Carbon Atoms\n"
        f"Reference: restricted HF / 6-31G, y = 0 A, {clip_text}"
    )
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    atom, mol, _, ref = build_benzene_reference(
        basis=str(args.basis),
        grids_level=int(args.grids_level),
    )

    clip_value = None if bool(args.no_clip) else 5.0
    functional = neural_xc.Functional(
        include_pt2_channel=True,
        response_kernel_clip=clip_value,
    )
    hf_proj_impl = np.asarray(functional.projected_hf_grid_contribution_components(ref)[0], dtype=np.float64)
    pt2_proj_impl = np.asarray(functional.projected_pt2_grid_contribution(ref), dtype=np.float64)

    hf_data = _hf_projection_data(ref, clip=clip_value)
    pt2_data = _pt2_projection_data(ref, clip=clip_value)

    x, y, xx, yy, plane_coords_ang = _plane_coordinates(
        float(args.plane_min),
        float(args.plane_max),
        int(args.plane_points),
    )
    hf_raw_plane, hf_proj_plane, plane_ao = _hf_plane_profiles(
        mol,
        ref,
        plane_coords_ang,
        hf_data,
        chunk_size=int(args.chunk_size),
        clip=clip_value,
    )
    pt2_raw_plane, pt2_proj_plane = _pt2_plane_profiles(
        plane_ao,
        pt2_data,
        chunk_size=int(args.chunk_size),
        clip=clip_value,
    )
    if clip_value is None:
        hf_clip_hits = 0
        pt2_clip_hits = 0
    else:
        hf_clip_hits = int(np.count_nonzero(np.abs(float(hf_data["scale"]) * hf_raw_plane) > clip_value))
        pt2_clip_hits = int(np.count_nonzero(np.abs(float(pt2_data["scale"]) * pt2_raw_plane) > clip_value))

    hf_diff = _shape_normalized_difference(hf_raw_plane, hf_proj_plane).reshape(xx.shape)
    pt2_diff = _shape_normalized_difference(pt2_raw_plane, pt2_proj_plane).reshape(xx.shape)
    hf_raw_grid = hf_raw_plane.reshape(xx.shape)
    hf_proj_grid = hf_proj_plane.reshape(xx.shape)
    pt2_raw_grid = pt2_raw_plane.reshape(xx.shape)
    pt2_proj_grid = pt2_proj_plane.reshape(xx.shape)
    hf_err_grid = hf_proj_grid - hf_raw_grid
    pt2_err_grid = pt2_proj_grid - pt2_raw_grid

    nuclei = []
    for line in atom.split(";"):
        parts = line.strip().split()
        nuclei.append((float(parts[1]), float(parts[2])))
    nuclei_xy = np.asarray(nuclei, dtype=np.float64)
    carbon_x = np.asarray(
        [xy[0] for idx, xy in enumerate(nuclei_xy) if idx % 2 == 0 and abs(xy[1]) < 1e-10],
        dtype=np.float64,
    )

    fig, axes = plt.subplots(2, 3, figsize=(12.8, 8.2), dpi=180)
    _plot_row(
        fig,
        axes[0],
        xx=xx,
        yy=yy,
        nuclei_xy=nuclei_xy,
        raw=hf_raw_grid,
        projected=hf_proj_grid,
        diff=hf_diff,
        label="HF exchange",
        scale=float(hf_data["scale"]),
    )
    _plot_row(
        fig,
        axes[1],
        xx=xx,
        yy=yy,
        nuclei_xy=nuclei_xy,
        raw=pt2_raw_grid,
        projected=pt2_proj_grid,
        diff=pt2_diff,
        label="MP2 correlation",
        scale=float(pt2_data["scale"]),
    )
    fig.suptitle(
        "Benzene In-Plane Comparison of Strict Raw Gauges and Projected Channels\n"
        f"Reference: restricted HF / {args.basis.upper()}, plane z = 0 A, "
        f"{'no clip' if clip_value is None else f'clip = {clip_value:.1f}'}"
    )
    fig.tight_layout()

    suffix = "_noclip" if clip_value is None else ""
    png_path = outdir / f"benzene_hf_pt2_plane_compare{suffix}.png"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    fig_err, axes_err = plt.subplots(2, 3, figsize=(12.8, 8.2), dpi=180)
    _plot_error_row(
        fig_err,
        axes_err[0],
        xx=xx,
        yy=yy,
        nuclei_xy=nuclei_xy,
        raw=hf_raw_grid,
        projected=hf_proj_grid,
        error=hf_err_grid,
        label="HF exchange",
        scale=float(hf_data["scale"]),
    )
    _plot_error_row(
        fig_err,
        axes_err[1],
        xx=xx,
        yy=yy,
        nuclei_xy=nuclei_xy,
        raw=pt2_raw_grid,
        projected=pt2_proj_grid,
        error=pt2_err_grid,
        label="MP2 correlation",
        scale=float(pt2_data["scale"]),
    )
    fig_err.suptitle(
        "Benzene In-Plane Projection Error Relative to the Strict Raw Gauges\n"
        f"Reference: restricted HF / {args.basis.upper()}, plane z = 0 A, "
        f"{'no clip' if clip_value is None else f'clip = {clip_value:.1f}'}"
    )
    fig_err.tight_layout()
    err_png_path = outdir / f"benzene_hf_pt2_projection_error{suffix}.png"
    fig_err.savefig(err_png_path, bbox_inches="tight")
    plt.close(fig_err)

    axis_idx = int(np.argmin(np.abs(y)))
    axis_png_path = outdir / f"benzene_hf_pt2_axis_profile{suffix}.png"
    _plot_axis_profiles(
        x=x,
        hf_raw=hf_raw_grid[axis_idx],
        hf_projected=hf_proj_grid[axis_idx],
        pt2_raw=pt2_raw_grid[axis_idx],
        pt2_projected=pt2_proj_grid[axis_idx],
        carbon_x=carbon_x,
        scale_hf=float(hf_data["scale"]),
        scale_pt2=float(pt2_data["scale"]),
        clip=clip_value,
        outpath=axis_png_path,
    )
    axis_err_png_path = outdir / f"benzene_hf_pt2_axis_error{suffix}.png"
    _plot_axis_errors(
        x=x,
        hf_error=hf_err_grid[axis_idx],
        pt2_error=pt2_err_grid[axis_idx],
        carbon_x=carbon_x,
        clip=clip_value,
        outpath=axis_err_png_path,
    )

    summary = {
        "reference_method": f"restricted HF/{args.basis}",
        "plane": "z = 0 Angstrom",
        "clip": None if clip_value is None else float(clip_value),
        "plane_min_angstrom": float(args.plane_min),
        "plane_max_angstrom": float(args.plane_max),
        "plane_points": int(args.plane_points),
        "hf_scale_factor": float(hf_data["scale"]),
        "hf_exact_energy_h": float(hf_data["exact_energy_h"]),
        "hf_raw_grid_integral_h": float(hf_data["projected_energy_h"]),
        "hf_impl_formula_max_abs_diff_h": float(np.max(np.abs(hf_proj_impl - hf_data["projected_grid"]))),
        "hf_shape_diff_max_abs": float(np.max(np.abs(hf_diff))),
        "hf_clip_hit_count_on_plane": hf_clip_hits,
        "hf_plane_raw_min_h": float(np.min(hf_raw_plane)),
        "hf_plane_raw_max_h": float(np.max(hf_raw_plane)),
        "hf_plane_projected_min_h": float(np.min(hf_proj_plane)),
        "hf_plane_projected_max_h": float(np.max(hf_proj_plane)),
        "pt2_scale_factor": float(pt2_data["scale"]),
        "pt2_total_energy_h": float(pt2_data["total_energy_h"]),
        "pt2_raw_grid_integral_h": float(pt2_data["projected_energy_h"]),
        "pt2_impl_formula_max_abs_diff_h": float(np.max(np.abs(pt2_proj_impl - pt2_data["projected_grid"]))),
        "pt2_shape_diff_max_abs": float(np.max(np.abs(pt2_diff))),
        "pt2_clip_hit_count_on_plane": pt2_clip_hits,
        "pt2_plane_raw_min_h": float(np.min(pt2_raw_plane)),
        "pt2_plane_raw_max_h": float(np.max(pt2_raw_plane)),
        "pt2_plane_projected_min_h": float(np.min(pt2_proj_plane)),
        "pt2_plane_projected_max_h": float(np.max(pt2_proj_plane)),
        "axis_y_angstrom": float(y[axis_idx]),
        "axis_profile_png": str(axis_png_path),
        "projection_error_png": str(err_png_path),
        "axis_error_png": str(axis_err_png_path),
        "axis_carbon_x_angstrom": carbon_x.tolist(),
        "hf_plane_abs_error_max_h": float(np.max(np.abs(hf_err_grid))),
        "hf_plane_abs_error_mae_h": float(np.mean(np.abs(hf_err_grid))),
        "hf_plane_abs_error_rmse_h": float(np.sqrt(np.mean(hf_err_grid**2))),
        "pt2_plane_abs_error_max_h": float(np.max(np.abs(pt2_err_grid))),
        "pt2_plane_abs_error_mae_h": float(np.mean(np.abs(pt2_err_grid))),
        "pt2_plane_abs_error_rmse_h": float(np.sqrt(np.mean(pt2_err_grid**2))),
        "hf_axis_abs_error_max_h": float(np.max(np.abs(hf_err_grid[axis_idx]))),
        "hf_axis_abs_error_mae_h": float(np.mean(np.abs(hf_err_grid[axis_idx]))),
        "pt2_axis_abs_error_max_h": float(np.max(np.abs(pt2_err_grid[axis_idx]))),
        "pt2_axis_abs_error_mae_h": float(np.mean(np.abs(pt2_err_grid[axis_idx]))),
    }
    json_path = outdir / f"benzene_hf_pt2_plane_compare{suffix}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(png_path)
    print(json_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
