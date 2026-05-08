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


ANG_TO_BOHR = 1.0 / 0.52917721092

_HELPER_PATH = Path(__file__).with_name("benzene_hf_pt2_plane_compare.py")
_HELPER_SPEC = importlib.util.spec_from_file_location("_benzene_hf_pt2_plane_compare", _HELPER_PATH)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise RuntimeError(f"Failed to load helper module from {_HELPER_PATH}")
_HELPERS = importlib.util.module_from_spec(_HELPER_SPEC)
sys.modules[_HELPER_SPEC.name] = _HELPERS
_HELPER_SPEC.loader.exec_module(_HELPERS)

build_benzene_reference = _HELPERS.build_benzene_reference
_hf_projection_data = _HELPERS._hf_projection_data
_hf_plane_profiles = _HELPERS._hf_plane_profiles


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare the current AO-based HF raw/projected channel against a strict "
            "MO-pair HF raw gauge along the benzene symmetry axis through opposite carbon atoms."
        )
    )
    p.add_argument("--basis", default="6-31g")
    p.add_argument("--axis-min", type=float, default=-3.2)
    p.add_argument("--axis-max", type=float, default=3.2)
    p.add_argument("--axis-points", type=int, default=401)
    p.add_argument("--grids-level", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=256)
    p.add_argument("--outdir", default="outputs/benzene_hf_mopair_axis_compare")
    return p.parse_args()


def _strict_mopair_hf_raw_axis(mol, ref, axis_coords_ang: np.ndarray) -> np.ndarray:
    mo_coeff = np.asarray(ref.mo_coeff, dtype=np.float64)
    mo_occ = np.asarray(ref.mo_occ, dtype=np.float64)
    if mo_coeff.ndim == 3:
        mo_coeff = mo_coeff[0]
    if mo_occ.ndim == 2:
        mo_occ = mo_occ[0]
    nocc = int(ref.nocc)
    orbo = mo_coeff[:, :nocc]

    ao_axis = mol.eval_gto("GTOval_cart", axis_coords_ang * ANG_TO_BOHR)
    mo_axis = ao_axis @ orbo

    values = np.zeros(axis_coords_ang.shape[0], dtype=np.float64)
    for idx, coord_ang in enumerate(axis_coords_ang):
        coord_bohr = tuple(float(x) for x in coord_ang * ANG_TO_BOHR)
        with mol.with_rinv_origin(coord_bohr):
            v_ao = mol.intor("int1e_rinv_cart")
        v_mo = orbo.T @ v_ao @ orbo
        values[idx] = -np.einsum("i,j,ij->", mo_axis[idx], mo_axis[idx], v_mo, optimize=True)
    return values


def _plot_axis_compare(
    *,
    x: np.ndarray,
    current_raw: np.ndarray,
    current_projected: np.ndarray,
    strict_mopair_raw: np.ndarray,
    carbon_x: np.ndarray,
    scale: float,
    outpath: Path,
) -> None:
    err_current_raw = current_raw - strict_mopair_raw
    err_projected = current_projected - strict_mopair_raw

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.5), dpi=220, sharex=True)

    axes[0].plot(x, strict_mopair_raw, color="#111111", lw=2.2, label="strict MO-pair raw")
    axes[0].plot(x, current_raw, color="#1f77b4", lw=2.0, label="current AO raw")
    axes[0].plot(x, current_projected, color="#d62728", lw=2.0, ls="--", label="current projected")
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
        f"current AO scale = {scale:.6f}",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"),
    )
    axes[0].set_ylabel("Local contribution (Eh)")
    axes[0].set_title("HF local gauge along the C-C symmetry axis")
    axes[0].grid(alpha=0.2, linewidth=0.7)
    axes[0].legend(frameon=False, loc="best")

    axes[1].plot(x, err_current_raw, color="#1f77b4", lw=2.0, label="current AO raw - strict MO-pair")
    axes[1].plot(x, err_projected, color="#d62728", lw=2.0, ls="--", label="current projected - strict MO-pair")
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
    axes[1].set_xlabel("x along symmetry axis (Angstrom)")
    axes[1].set_ylabel("Error (Eh)")
    axes[1].set_title("Error relative to the strict MO-pair raw gauge")
    axes[1].grid(alpha=0.2, linewidth=0.7)
    axes[1].legend(frameon=False, loc="best")

    fig.suptitle(
        "Benzene HF Gauge Comparison Along the Symmetry Axis Through Opposite Carbon Atoms\n"
        "Reference: restricted HF / 6-31G, y = 0 A"
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

    x = np.linspace(float(args.axis_min), float(args.axis_max), int(args.axis_points), dtype=np.float64)
    axis_coords_ang = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)

    hf_data = _hf_projection_data(ref, clip=None)
    current_raw, current_projected, _ = _hf_plane_profiles(
        mol,
        ref,
        axis_coords_ang,
        hf_data,
        chunk_size=int(args.chunk_size),
        clip=None,
    )
    strict_mopair_raw = _strict_mopair_hf_raw_axis(mol, ref, axis_coords_ang)

    nuclei = []
    for line in atom.split(";"):
        parts = line.strip().split()
        nuclei.append((float(parts[1]), float(parts[2])))
    nuclei_xy = np.asarray(nuclei, dtype=np.float64)
    carbon_x = np.asarray(
        [xy[0] for idx, xy in enumerate(nuclei_xy) if idx % 2 == 0 and abs(xy[1]) < 1e-10],
        dtype=np.float64,
    )

    png_path = outdir / "benzene_hf_mopair_axis_compare.png"
    _plot_axis_compare(
        x=x,
        current_raw=current_raw,
        current_projected=current_projected,
        strict_mopair_raw=strict_mopair_raw,
        carbon_x=carbon_x,
        scale=float(hf_data["scale"]),
        outpath=png_path,
    )

    summary = {
        "reference_method": f"restricted HF/{args.basis}",
        "axis_y_angstrom": 0.0,
        "axis_min_angstrom": float(args.axis_min),
        "axis_max_angstrom": float(args.axis_max),
        "axis_points": int(args.axis_points),
        "carbon_x_angstrom": carbon_x.tolist(),
        "current_ao_scale_factor": float(hf_data["scale"]),
        "current_raw_min_h": float(np.min(current_raw)),
        "current_raw_max_h": float(np.max(current_raw)),
        "current_projected_min_h": float(np.min(current_projected)),
        "current_projected_max_h": float(np.max(current_projected)),
        "strict_mopair_raw_min_h": float(np.min(strict_mopair_raw)),
        "strict_mopair_raw_max_h": float(np.max(strict_mopair_raw)),
        "current_raw_vs_strict_max_abs_h": float(np.max(np.abs(current_raw - strict_mopair_raw))),
        "current_raw_vs_strict_mae_h": float(np.mean(np.abs(current_raw - strict_mopair_raw))),
        "current_projected_vs_strict_max_abs_h": float(np.max(np.abs(current_projected - strict_mopair_raw))),
        "current_projected_vs_strict_mae_h": float(np.mean(np.abs(current_projected - strict_mopair_raw))),
        "png": str(png_path),
    }
    json_path = outdir / "benzene_hf_mopair_axis_compare.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(png_path)
    print(json_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
