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

from td_graddft import neural_xc

_HELPER_PATH = Path(__file__).with_name("benzene_hf_pt2_plane_compare.py")
_HELPER_SPEC = importlib.util.spec_from_file_location("_benzene_hf_pt2_plane_compare", _HELPER_PATH)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise RuntimeError(f"Failed to load helper module from {_HELPER_PATH}")
_HELPERS = importlib.util.module_from_spec(_HELPER_SPEC)
sys.modules[_HELPER_SPEC.name] = _HELPERS
_HELPER_SPEC.loader.exec_module(_HELPERS)

build_benzene_reference = _HELPERS.build_benzene_reference
_pt2_projection_data = _HELPERS._pt2_projection_data
_pt2_plane_profiles = _HELPERS._pt2_plane_profiles
_plane_coordinates = _HELPERS._plane_coordinates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare the legacy scaled/projected PT2 channel against the exact local "
            "pair-gauge PT2 channel on the benzene molecular plane."
        )
    )
    p.add_argument("--basis", default="6-31g")
    p.add_argument("--plane-min", type=float, default=-3.2)
    p.add_argument("--plane-max", type=float, default=3.2)
    p.add_argument("--plane-points", type=int, default=121)
    p.add_argument("--grids-level", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=256)
    p.add_argument("--axis-min", type=float, default=-3.2)
    p.add_argument("--axis-max", type=float, default=3.2)
    p.add_argument("--axis-points", type=int, default=401)
    p.add_argument("--outdir", default="outputs/benzene_pt2_channel_mode_compare")
    return p.parse_args()


def _plot_plane_compare(
    *,
    x: np.ndarray,
    y: np.ndarray,
    legacy_projected: np.ndarray,
    local_exact: np.ndarray,
    outpath: Path,
) -> None:
    diff = legacy_projected - local_exact
    vmax = max(
        float(np.max(np.abs(legacy_projected))),
        float(np.max(np.abs(local_exact))),
        1e-12,
    )
    dvmax = max(float(np.max(np.abs(diff))), 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), dpi=220, constrained_layout=True)
    panels = [
        (legacy_projected, "Legacy scaled/projected PT2", "RdBu_r", vmax),
        (local_exact, "Exact local pair gauge PT2", "RdBu_r", vmax),
        (diff, "Legacy - exact local", "coolwarm", dvmax),
    ]
    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
    for ax, (field, title, cmap, lim) in zip(axes, panels):
        im = ax.imshow(
            field,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=-lim,
            vmax=lim,
            aspect="equal",
        )
        ax.set_title(title)
        ax.set_xlabel("x (Angstrom)")
        ax.set_ylabel("y (Angstrom)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("Benzene PT2 local channel on the molecular plane\nReference: restricted HF / 6-31G")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def _plot_axis_compare(
    *,
    x: np.ndarray,
    legacy_projected: np.ndarray,
    local_exact: np.ndarray,
    carbon_x: np.ndarray,
    scale: float,
    outpath: Path,
) -> None:
    diff = legacy_projected - local_exact
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.4), dpi=220, sharex=True)

    axes[0].plot(x, legacy_projected, lw=2.0, color="#1f77b4", label="Legacy scaled/projected")
    axes[0].plot(x, local_exact, lw=2.0, color="#d62728", label="Exact local pair gauge")
    for xpos in carbon_x:
        axes[0].axvline(xpos, color="black", lw=1.0, ls=":", alpha=0.75)
        axes[0].text(
            xpos,
            0.98,
            "C",
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
        )
    axes[0].text(
        0.02,
        0.96,
        f"legacy scale = {scale:.6f}",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"),
    )
    axes[0].set_ylabel("Local contribution (Eh)")
    axes[0].set_title("Along the symmetry axis through opposite carbon atoms")
    axes[0].grid(alpha=0.2, linewidth=0.7)
    axes[0].legend(frameon=False, loc="best")

    axes[1].plot(x, diff, lw=2.0, color="#6f42c1")
    axes[1].axhline(0.0, color="black", lw=1.0, ls="--", alpha=0.7)
    for xpos in carbon_x:
        axes[1].axvline(xpos, color="black", lw=1.0, ls=":", alpha=0.75)
        axes[1].text(
            xpos,
            0.98,
            "C",
            transform=axes[1].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
        )
    axes[1].set_xlabel("x (Angstrom)")
    axes[1].set_ylabel("Legacy - exact local (Eh)")
    axes[1].grid(alpha=0.2, linewidth=0.7)

    fig.suptitle("Benzene PT2 channel comparison on the symmetry axis\nReference: restricted HF / 6-31G, y = 0 A")
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

    legacy_functional = neural_xc.Functional(
        semilocal_xc="pbe",
        hidden_dims=(8, 8),
        include_pt2_channel=True,
        pt2_channel_mode="scaled_projected",
        response_kernel_clip=None,
    )
    exact_functional = neural_xc.Functional(
        semilocal_xc="pbe",
        hidden_dims=(8, 8),
        include_pt2_channel=True,
        pt2_channel_mode="local_exact",
        response_kernel_clip=None,
    )
    legacy_grid = np.asarray(
        legacy_functional.projected_pt2_grid_contribution(ref),
        dtype=np.float64,
    )
    exact_grid = np.asarray(
        exact_functional.projected_pt2_grid_contribution(ref),
        dtype=np.float64,
    )

    pt2_data = _pt2_projection_data(ref, clip=None)

    x, y, _, _, plane_coords_ang = _plane_coordinates(
        float(args.plane_min),
        float(args.plane_max),
        int(args.plane_points),
    )
    plane_ao = mol.eval_gto("GTOval_cart", plane_coords_ang / 0.52917721092)
    local_exact_plane, legacy_plane = _pt2_plane_profiles(
        plane_ao,
        pt2_data,
        chunk_size=int(args.chunk_size),
        clip=None,
    )
    legacy_plane_2d = legacy_plane.reshape(len(y), len(x))
    local_exact_plane_2d = local_exact_plane.reshape(len(y), len(x))

    axis_x = np.linspace(float(args.axis_min), float(args.axis_max), int(args.axis_points), dtype=np.float64)
    axis_coords_ang = np.stack([axis_x, np.zeros_like(axis_x), np.zeros_like(axis_x)], axis=1)
    axis_ao = mol.eval_gto("GTOval_cart", axis_coords_ang / 0.52917721092)
    local_exact_axis, legacy_axis = _pt2_plane_profiles(
        axis_ao,
        pt2_data,
        chunk_size=int(args.chunk_size),
        clip=None,
    )

    nuclei = []
    for line in atom.split(";"):
        parts = line.strip().split()
        nuclei.append((float(parts[1]), float(parts[2])))
    nuclei_xy = np.asarray(nuclei, dtype=np.float64)
    carbon_x = np.asarray(
        [xy[0] for idx, xy in enumerate(nuclei_xy) if idx % 2 == 0 and abs(xy[1]) < 1e-10],
        dtype=np.float64,
    )

    plane_png = outdir / "benzene_pt2_channel_mode_plane_compare.png"
    axis_png = outdir / "benzene_pt2_channel_mode_axis_compare.png"
    _plot_plane_compare(
        x=x,
        y=y,
        legacy_projected=legacy_plane_2d,
        local_exact=local_exact_plane_2d,
        outpath=plane_png,
    )
    _plot_axis_compare(
        x=axis_x,
        legacy_projected=legacy_axis,
        local_exact=local_exact_axis,
        carbon_x=carbon_x,
        scale=float(pt2_data["scale"]),
        outpath=axis_png,
    )

    grid_weights = np.asarray(ref.grid.weights, dtype=np.float64)
    summary = {
        "reference_method": f"restricted HF/{args.basis}",
        "legacy_scale_factor": float(pt2_data["scale"]),
        "grid_legacy_integral_h": float(np.dot(grid_weights, legacy_grid)),
        "grid_local_exact_integral_h": float(np.dot(grid_weights, exact_grid)),
        "exact_mp2_h": float(pt2_data["total_energy_h"]),
        "grid_diff_mae_h": float(np.mean(np.abs(legacy_grid - exact_grid))),
        "grid_diff_max_abs_h": float(np.max(np.abs(legacy_grid - exact_grid))),
        "plane_diff_mae_h": float(np.mean(np.abs(legacy_plane_2d - local_exact_plane_2d))),
        "plane_diff_max_abs_h": float(np.max(np.abs(legacy_plane_2d - local_exact_plane_2d))),
        "axis_diff_mae_h": float(np.mean(np.abs(legacy_axis - local_exact_axis))),
        "axis_diff_max_abs_h": float(np.max(np.abs(legacy_axis - local_exact_axis))),
        "plane_png": str(plane_png),
        "axis_png": str(axis_png),
    }
    json_path = outdir / "benzene_pt2_channel_mode_compare.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(plane_png)
    print(axis_png)
    print(json_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
