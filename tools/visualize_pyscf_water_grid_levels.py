#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BOHR_TO_ANGSTROM = 0.529177210903

WATER_GEOMETRY = """
O  0.000000  0.000000  0.117790
H  0.000000  0.755453 -0.471161
H  0.000000 -0.755453 -0.471161
"""

ATOM_COLORS = {
    "O": "#D1493F",
    "H": "#2C7FB8",
}


@dataclass(frozen=True)
class GridLevelData:
    level: int
    coords_angstrom: np.ndarray
    weights_bohr3: np.ndarray
    sample_indices: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize PySCF DFT grid levels for a water molecule.",
    )
    parser.add_argument("--basis", default="def2-svp", help="PySCF basis used to build the molecule.")
    parser.add_argument(
        "--levels",
        default="0,1,2,3,4,5",
        help="Comma-separated PySCF Grids.level values.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/pyscf_water_grid_levels"),
        help="Output directory for figures and tables.",
    )
    parser.add_argument(
        "--max-points-per-level",
        type=int,
        default=60000,
        help="Maximum points plotted in 3D per level. Use 0 to plot all points.",
    )
    parser.add_argument(
        "--slice-half-width",
        type=float,
        default=0.08,
        help="Half-width in Angstrom for the molecular-plane x slice.",
    )
    parser.add_argument("--seed", type=int, default=20260527, help="Random seed for deterministic plotting samples.")
    parser.add_argument("--dpi", type=int, default=220, help="Figure DPI.")
    return parser.parse_args()


def _configure_matplotlib(outdir: Path) -> None:
    mpl_config = outdir / ".mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config.resolve()))
    xdg_cache = mpl_config / "xdg-cache"
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache.resolve()))


def _parse_levels(raw: str) -> list[int]:
    levels = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not levels:
        raise ValueError("At least one grid level is required.")
    return levels


def _build_water_mol(basis: str):
    from pyscf import gto

    return gto.M(
        atom=WATER_GEOMETRY,
        unit="Angstrom",
        basis=str(basis),
        charge=0,
        spin=0,
        verbose=0,
    )


def _build_grid_level(mol, level: int, max_points: int, rng: np.random.Generator) -> GridLevelData:
    from pyscf import dft

    grids = dft.gen_grid.Grids(mol)
    grids.level = int(level)
    grids.build()
    coords_angstrom = np.asarray(grids.coords, dtype=np.float64) * BOHR_TO_ANGSTROM
    weights_bohr3 = np.asarray(grids.weights, dtype=np.float64)

    total = coords_angstrom.shape[0]
    if max_points > 0 and total > max_points:
        sample_indices = np.sort(rng.choice(total, size=max_points, replace=False))
    else:
        sample_indices = np.arange(total)

    return GridLevelData(
        level=int(level),
        coords_angstrom=coords_angstrom,
        weights_bohr3=weights_bohr3,
        sample_indices=sample_indices,
    )


def _atom_symbols_and_coords(mol) -> tuple[list[str], np.ndarray]:
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    coords = np.asarray(mol.atom_coords(unit="Angstrom"), dtype=np.float64)
    return symbols, coords


def _axis_limits(datasets: list[GridLevelData], atom_coords: np.ndarray) -> tuple[tuple[float, float], ...]:
    all_coords = np.concatenate([data.coords_angstrom for data in datasets] + [atom_coords], axis=0)
    mins = all_coords.min(axis=0)
    maxs = all_coords.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    radius *= 1.04
    return tuple((float(c - radius), float(c + radius)) for c in center)


def _log10_abs_weights(weights: np.ndarray) -> np.ndarray:
    abs_weights = np.abs(np.asarray(weights, dtype=np.float64))
    positive = abs_weights[np.isfinite(abs_weights) & (abs_weights > 0.0)]
    floor = float(positive.min()) if positive.size else np.finfo(np.float64).tiny
    return np.log10(np.maximum(abs_weights, floor))


def _grid_shape(n_items: int) -> tuple[int, int]:
    ncols = 3 if n_items > 2 else n_items
    nrows = int(np.ceil(n_items / ncols))
    return nrows, ncols


def _setup_common_2d_axis(ax, title: str) -> None:
    ax.set_title(title, fontsize=10, pad=7)
    ax.set_xlabel("y / Angstrom")
    ax.set_ylabel("z / Angstrom")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#E2E7EB", linewidth=0.6)


def _plot_atoms_2d(ax, symbols: list[str], atom_coords: np.ndarray) -> None:
    for symbol, coord in zip(symbols, atom_coords):
        color = ATOM_COLORS.get(symbol, "#333333")
        size = 72 if symbol == "O" else 44
        ax.scatter(coord[1], coord[2], s=size, c=color, edgecolors="white", linewidths=0.9, zorder=10)
        ax.text(coord[1], coord[2] + 0.11, symbol, ha="center", va="bottom", fontsize=8, color="#172026")


def _plot_atoms_3d(ax, symbols: list[str], atom_coords: np.ndarray) -> None:
    for symbol, coord in zip(symbols, atom_coords):
        color = ATOM_COLORS.get(symbol, "#333333")
        size = 54 if symbol == "O" else 34
        ax.scatter(coord[0], coord[1], coord[2], s=size, c=color, edgecolors="white", linewidths=0.8)
        ax.text(coord[0], coord[1], coord[2] + 0.18, symbol, ha="center", va="bottom", fontsize=7)


def plot_3d_levels(
    datasets: list[GridLevelData],
    symbols: list[str],
    atom_coords: np.ndarray,
    axis_limits: tuple[tuple[float, float], ...],
    outpath: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows, ncols = _grid_shape(len(datasets))
    fig = plt.figure(figsize=(4.4 * ncols, 3.8 * nrows))
    fig.suptitle("PySCF water molecular grids: 3D point distribution", fontsize=14, fontweight="bold")

    all_log_weights = np.concatenate([_log10_abs_weights(data.weights_bohr3[data.sample_indices]) for data in datasets])
    vmin = float(np.nanpercentile(all_log_weights, 1.0))
    vmax = float(np.nanpercentile(all_log_weights, 99.0))

    axes = []
    scatter = None
    for idx, data in enumerate(datasets, start=1):
        ax = fig.add_subplot(nrows, ncols, idx, projection="3d")
        axes.append(ax)
        sample = data.sample_indices
        coords = data.coords_angstrom[sample]
        log_weights = _log10_abs_weights(data.weights_bohr3[sample])
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=log_weights,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=0.35,
            alpha=0.38,
            linewidths=0.0,
            rasterized=True,
        )
        _plot_atoms_3d(ax, symbols, atom_coords)
        total = data.coords_angstrom.shape[0]
        shown = sample.shape[0]
        ax.set_title(f"level {data.level}: {shown:,}/{total:,} points", fontsize=9)
        ax.set_xlabel("x / A", labelpad=-2)
        ax.set_ylabel("y / A", labelpad=-2)
        ax.set_zlabel("z / A", labelpad=-2)
        ax.set_xlim(*axis_limits[0])
        ax.set_ylim(*axis_limits[1])
        ax.set_zlim(*axis_limits[2])
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=18, azim=-62)
        ax.tick_params(labelsize=7, pad=0)

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.18, top=0.9, wspace=0.02, hspace=0.18)
    if scatter is not None:
        cbar_ax = fig.add_axes([0.27, 0.07, 0.46, 0.018])
        cbar = fig.colorbar(
            scatter,
            cax=cbar_ax,
            orientation="horizontal",
        )
        cbar.set_label("log10(abs(grid weight) / Bohr^3)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_yz_slice_levels(
    datasets: list[GridLevelData],
    symbols: list[str],
    atom_coords: np.ndarray,
    outpath: Path,
    *,
    slice_half_width: float,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows, ncols = _grid_shape(len(datasets))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False)
    fig.suptitle(
        f"PySCF water molecular grids: |x| <= {slice_half_width:.2f} Angstrom slice",
        fontsize=14,
        fontweight="bold",
    )

    yz_all = np.concatenate([data.coords_angstrom[:, 1:3] for data in datasets] + [atom_coords[:, 1:3]], axis=0)
    y_min, z_min = yz_all.min(axis=0)
    y_max, z_max = yz_all.max(axis=0)
    margin = 0.25

    for ax, data in zip(axes.ravel(), datasets):
        coords = data.coords_angstrom
        mask = np.abs(coords[:, 0]) <= slice_half_width
        weights = data.weights_bohr3[mask]
        log_weights = _log10_abs_weights(weights)
        ax.scatter(
            coords[mask, 1],
            coords[mask, 2],
            c=log_weights,
            cmap="magma",
            s=1.4,
            alpha=0.5,
            linewidths=0.0,
            rasterized=True,
        )
        _plot_atoms_2d(ax, symbols, atom_coords)
        _setup_common_2d_axis(ax, f"level {data.level}: {int(mask.sum()):,}/{coords.shape[0]:,} points")
        ax.set_xlim(float(y_min - margin), float(y_max + margin))
        ax.set_ylim(float(z_min - margin), float(z_max + margin))

    for ax in axes.ravel()[len(datasets):]:
        ax.axis("off")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_radial_histogram(
    datasets: list[GridLevelData],
    atom_coords: np.ndarray,
    outpath: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(datasets)))
    bins = np.linspace(0.0, 8.0, 121)
    for color, data in zip(colors, datasets):
        diff = data.coords_angstrom[:, None, :] - atom_coords[None, :, :]
        nearest = np.linalg.norm(diff, axis=2).min(axis=1)
        ax.hist(
            nearest,
            bins=bins,
            histtype="step",
            linewidth=1.5,
            color=color,
            label=f"level {data.level} ({data.coords_angstrom.shape[0]:,})",
        )
    ax.set_xlabel("distance to nearest nucleus / Angstrom")
    ax.set_ylabel("grid point count")
    ax.set_title("Radial shell distribution by PySCF grid level", fontsize=12, fontweight="bold")
    ax.grid(color="#E2E7EB", linewidth=0.7)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_stats(
    datasets: list[GridLevelData],
    atom_coords: np.ndarray,
    out_csv: Path,
    out_json: Path,
    *,
    basis: str,
    slice_half_width: float,
) -> None:
    rows = []
    for data in datasets:
        coords = data.coords_angstrom
        weights = data.weights_bohr3
        nearest = np.linalg.norm(coords[:, None, :] - atom_coords[None, :, :], axis=2).min(axis=1)
        row = {
            "level": data.level,
            "basis": basis,
            "n_points": int(coords.shape[0]),
            "n_points_in_yz_slice": int((np.abs(coords[:, 0]) <= slice_half_width).sum()),
            "sampled_points_3d": int(data.sample_indices.shape[0]),
            "n_negative_weights": int((weights < 0.0).sum()),
            "n_zero_weights": int((weights == 0.0).sum()),
            "weight_sum_bohr3": float(weights.sum()),
            "weight_min_bohr3": float(weights.min()),
            "weight_median_bohr3": float(np.median(weights)),
            "weight_max_bohr3": float(weights.max()),
            "nearest_atom_radius_min_angstrom": float(nearest.min()),
            "nearest_atom_radius_median_angstrom": float(np.median(nearest)),
            "nearest_atom_radius_p95_angstrom": float(np.percentile(nearest, 95.0)),
            "nearest_atom_radius_max_angstrom": float(nearest.max()),
            "bbox_x_min_angstrom": float(coords[:, 0].min()),
            "bbox_x_max_angstrom": float(coords[:, 0].max()),
            "bbox_y_min_angstrom": float(coords[:, 1].min()),
            "bbox_y_max_angstrom": float(coords[:, 1].max()),
            "bbox_z_min_angstrom": float(coords[:, 2].min()),
            "bbox_z_max_angstrom": float(coords[:, 2].max()),
        }
        rows.append(row)

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    out_json.write_text(json.dumps({"basis": basis, "levels": rows}, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib(args.outdir)

    levels = _parse_levels(args.levels)
    rng = np.random.default_rng(args.seed)
    mol = _build_water_mol(args.basis)
    symbols, atom_coords = _atom_symbols_and_coords(mol)
    datasets = [
        _build_grid_level(mol, level, int(args.max_points_per_level), rng)
        for level in levels
    ]

    axis_limits = _axis_limits(datasets, atom_coords)
    fig_3d = args.outdir / "water_grid_levels_3d.png"
    fig_slice = args.outdir / "water_grid_levels_yz_slice.png"
    fig_hist = args.outdir / "water_grid_levels_radial_hist.png"
    stats_csv = args.outdir / "water_grid_levels_stats.csv"
    stats_json = args.outdir / "water_grid_levels_stats.json"

    plot_3d_levels(datasets, symbols, atom_coords, axis_limits, fig_3d, int(args.dpi))
    plot_yz_slice_levels(
        datasets,
        symbols,
        atom_coords,
        fig_slice,
        slice_half_width=float(args.slice_half_width),
        dpi=int(args.dpi),
    )
    plot_radial_histogram(datasets, atom_coords, fig_hist, int(args.dpi))
    write_stats(
        datasets,
        atom_coords,
        stats_csv,
        stats_json,
        basis=str(args.basis),
        slice_half_width=float(args.slice_half_width),
    )

    for path in (fig_3d, fig_slice, fig_hist, stats_csv, stats_json):
        print(f"wrote={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
