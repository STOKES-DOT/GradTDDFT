from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


def _restricted_channel(mo_coeff: Any, mo_occ: Any) -> tuple[np.ndarray, np.ndarray]:
    coeff = np.asarray(mo_coeff, dtype=float)
    occ = np.asarray(mo_occ, dtype=float)
    if coeff.ndim == 3:
        coeff = coeff[0]
    if occ.ndim == 2:
        occ = occ[0]
    return coeff, occ


def _safe_label(label: str) -> str:
    return label.replace("+", "p").replace("-", "m").replace(" ", "_").lower()


def orbital_indices(mo_occ: Any) -> dict[str, int]:
    _, occ = _restricted_channel(np.eye(1), mo_occ)
    occupied = np.where(occ > 1e-8)[0]
    virtual = np.where(occ <= 1e-8)[0]
    if occupied.size < 2:
        raise RuntimeError("Need at least 2 occupied orbitals for HOMO-1/HOMO rendering.")
    if virtual.size < 2:
        raise RuntimeError("Need at least 2 virtual orbitals for LUMO/LUMO+1 rendering.")
    return {
        "HOMO-1": int(occupied[-2]),
        "HOMO": int(occupied[-1]),
        "LUMO": int(virtual[0]),
        "LUMO+1": int(virtual[1]),
    }


def _match_frontier_pair(
    *,
    ref_coeff: np.ndarray,
    neural_coeff: np.ndarray,
    overlap: np.ndarray,
    ref_pair: tuple[int, int],
    candidate_pool: np.ndarray,
) -> tuple[int, int]:
    """Match a 2-orbital frontier pair by maximizing |<psi_ref|S|psi_neural>|."""
    if candidate_pool.size < 2:
        raise RuntimeError("Need at least two candidate orbitals for frontier matching.")

    ref_pair_vec = np.asarray(ref_coeff[:, [ref_pair[0], ref_pair[1]]], dtype=float)
    cand_pair_vec = np.asarray(neural_coeff[:, candidate_pool], dtype=float)
    score = np.abs(ref_pair_vec.T @ overlap @ cand_pair_vec)

    best_score = -1.0
    best_0 = 0
    best_1 = 1
    for j0 in range(candidate_pool.size):
        for j1 in range(candidate_pool.size):
            if j0 == j1:
                continue
            current = float(score[0, j0] + score[1, j1])
            if current > best_score:
                best_score = current
                best_0 = j0
                best_1 = j1

    return int(candidate_pool[best_0]), int(candidate_pool[best_1])


def _prepare_cairo_runtime() -> None:
    homebrew_lib = "/opt/homebrew/lib"
    current = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    parts = [p for p in current.split(":") if p]
    if homebrew_lib not in parts:
        parts.insert(0, homebrew_lib)
    if "/usr/lib" not in parts:
        parts.append("/usr/lib")
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(parts)


def render_restricted_orbital_surfaces(
    *,
    reference_mol: Any,
    reference_mo_coeff: Any,
    reference_mo_occ: Any,
    neural_molecule: Any,
    overlap_matrix: Any,
    xyzrender_src: str,
    outdir: Path,
    iso: float = 0.05,
    diff_iso: float = 0.03,
    mo_blur: float = 1.2,
    mo_upsample: int = 4,
    cube_grid: int = 48,
    canvas_size: int = 620,
    match_frontier_by_overlap: bool = True,
    frontier_match_window: int = 6,
    labels: Iterable[str] = ("HOMO-1", "HOMO", "LUMO", "LUMO+1"),
) -> tuple[
    dict[str, dict[str, Path]],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    xyzrender_path = Path(xyzrender_src)
    if not xyzrender_path.exists():
        raise FileNotFoundError(f"xyzrender source not found: {xyzrender_src}")

    _prepare_cairo_runtime()
    if xyzrender_src not in sys.path:
        sys.path.insert(0, xyzrender_src)

    from xyzrender import load as xr_load, render as xr_render
    from pyscf.tools import cubegen

    orbital_dir = outdir / "orbital_surfaces"
    cube_ref_dir = orbital_dir / "cubes" / "reference"
    cube_neural_dir = orbital_dir / "cubes" / "neural"
    cube_diff_dir = orbital_dir / "cubes" / "difference"
    png_ref_dir = orbital_dir / "png" / "reference"
    png_neural_dir = orbital_dir / "png" / "neural"
    png_diff_dir = orbital_dir / "png" / "difference"
    for path in (
        cube_ref_dir,
        cube_neural_dir,
        cube_diff_dir,
        png_ref_dir,
        png_neural_dir,
        png_diff_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    ref_coeff, ref_occ = _restricted_channel(reference_mo_coeff, reference_mo_occ)
    neural_coeff, neural_occ = _restricted_channel(neural_molecule.mo_coeff, neural_molecule.mo_occ)
    ref_idx = orbital_indices(ref_occ)
    neural_idx = orbital_indices(neural_occ)
    overlap = np.asarray(overlap_matrix, dtype=float)

    if match_frontier_by_overlap:
        occupied_ref = np.where(ref_occ > 1e-8)[0]
        virtual_ref = np.where(ref_occ <= 1e-8)[0]
        occupied_neural = np.where(neural_occ > 1e-8)[0]
        virtual_neural = np.where(neural_occ <= 1e-8)[0]

        occ_window = int(np.clip(frontier_match_window, 2, occupied_neural.size))
        vir_window = int(np.clip(frontier_match_window, 2, virtual_neural.size))

        occ_pool = occupied_neural[-occ_window:]
        vir_pool = virtual_neural[:vir_window]

        matched_homo_m1, matched_homo = _match_frontier_pair(
            ref_coeff=ref_coeff,
            neural_coeff=neural_coeff,
            overlap=overlap,
            ref_pair=(int(occupied_ref[-2]), int(occupied_ref[-1])),
            candidate_pool=occ_pool,
        )
        matched_lumo, matched_lumo_p1 = _match_frontier_pair(
            ref_coeff=ref_coeff,
            neural_coeff=neural_coeff,
            overlap=overlap,
            ref_pair=(int(virtual_ref[0]), int(virtual_ref[1])),
            candidate_pool=vir_pool,
        )

        neural_idx["HOMO-1"] = matched_homo_m1
        neural_idx["HOMO"] = matched_homo
        neural_idx["LUMO"] = matched_lumo
        neural_idx["LUMO+1"] = matched_lumo_p1

    outputs: dict[str, dict[str, Path]] = {}
    diff_norms: dict[str, float] = {}
    aligned_overlaps: dict[str, float] = {}
    diff_iso_used: dict[str, float] = {}
    diff_scale_used: dict[str, float] = {}

    for label in labels:
        ref_vec = np.asarray(ref_coeff[:, ref_idx[label]], dtype=float)
        neural_vec = np.asarray(neural_coeff[:, neural_idx[label]], dtype=float)
        phase = float(ref_vec.T @ overlap @ neural_vec)
        if phase < 0.0:
            neural_vec = -neural_vec
            phase = -phase
        diff_vec = neural_vec - ref_vec

        diff_norm = float(np.sqrt(np.maximum(diff_vec.T @ overlap @ diff_vec, 0.0)))
        diff_norms[label] = diff_norm
        aligned_overlaps[label] = phase

        stem = _safe_label(label)
        cube_ref = cube_ref_dir / f"{stem}.cube"
        cube_neural = cube_neural_dir / f"{stem}.cube"
        cube_diff = cube_diff_dir / f"{stem}.cube"
        png_ref = png_ref_dir / f"{stem}.png"
        png_neural = png_neural_dir / f"{stem}.png"
        png_diff = png_diff_dir / f"{stem}.png"

        cubegen.orbital(
            reference_mol,
            str(cube_ref),
            ref_vec,
            nx=cube_grid,
            ny=cube_grid,
            nz=cube_grid,
        )
        cubegen.orbital(
            reference_mol,
            str(cube_neural),
            neural_vec,
            nx=cube_grid,
            ny=cube_grid,
            nz=cube_grid,
        )
        cubegen.orbital(
            reference_mol,
            str(cube_diff),
            diff_vec,
            nx=cube_grid,
            ny=cube_grid,
            nz=cube_grid,
        )

        diff_scale = 1.0
        diff_mol = xr_load(str(cube_diff))
        grid_data = getattr(getattr(diff_mol, "cube_data", None), "grid_data", None)
        max_abs = float(np.max(np.abs(np.asarray(grid_data)))) if grid_data is not None else 0.0
        target_peak = max(diff_iso * 2.5, 2e-2)
        if 0.0 < max_abs < target_peak:
            diff_scale = target_peak / max_abs
            cubegen.orbital(
                reference_mol,
                str(cube_diff),
                diff_vec * diff_scale,
                nx=cube_grid,
                ny=cube_grid,
                nz=cube_grid,
            )
            diff_mol = xr_load(str(cube_diff))
            grid_data = getattr(getattr(diff_mol, "cube_data", None), "grid_data", None)
            max_abs = float(np.max(np.abs(np.asarray(grid_data)))) if grid_data is not None else 0.0

        if max_abs > 1e-8:
            current_diff_iso = min(diff_iso, 0.5 * max_abs)
            current_diff_iso = max(current_diff_iso, 0.1 * max_abs)
        else:
            current_diff_iso = diff_iso
        diff_iso_used[label] = float(current_diff_iso)
        diff_scale_used[label] = float(diff_scale)

        for cube_path, png_path, current_iso in (
            (cube_ref, png_ref, iso),
            (cube_neural, png_neural, iso),
            (cube_diff, png_diff, current_diff_iso),
        ):
            cube_mol = xr_load(str(cube_path))
            xr_render(
                cube_mol,
                output=str(png_path),
                config="flat",
                hy=True,
                mo=True,
                iso=current_iso,
                mo_blur=mo_blur,
                mo_upsample=mo_upsample,
                transparent=True,
                canvas_size=canvas_size,
                mo_pos_color="#2F80ED",
                mo_neg_color="#C0392B",
            )

        outputs[label] = {
            "reference": png_ref,
            "neural": png_neural,
            "difference": png_diff,
        }

    return outputs, diff_norms, aligned_overlaps, diff_iso_used, diff_scale_used


def plot_orbital_compare_panel(
    *,
    orbital_label: str,
    ref_png: Path,
    neural_png: Path,
    diff_png: Path,
    iso: float,
    diff_iso: float,
    overlap_val: float,
    diff_norm: float,
    diff_scale: float,
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.3))
    for ax, title, path in (
        (axes[0], f"Reference {orbital_label}\niso=±{iso:.3f}", ref_png),
        (axes[1], f"Neural_xc {orbital_label}\niso=±{iso:.3f}", neural_png),
        (axes[2], f"Difference Δ{orbital_label}\niso=±{diff_iso:.4f}", diff_png),
    ):
        img = plt.imread(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=10.5, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{orbital_label} Real vs Neural_xc", fontsize=12.5, y=0.96)
    fig.text(
        0.5,
        0.04,
        f"overlap={overlap_val:.4f}   ||Δψ||_S={diff_norm:.4f}   Δscale={diff_scale:.2f}",
        ha="center",
        va="center",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.86, "edgecolor": "0.8"},
    )
    fig.subplots_adjust(left=0.02, right=0.985, top=0.84, bottom=0.13, wspace=0.035)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
