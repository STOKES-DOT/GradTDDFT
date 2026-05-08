from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np


REFERENCE_COLOR = "#111111"
PREDICTED_COLOR = "#c44e17"


def _load_curve_csv(
    path: Path,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"Empty curve csv: {path}")

    header = rows[0].keys()

    def _col(name: str) -> np.ndarray:
        return np.asarray([float(row[name]) for row in rows], dtype=float)

    predicted_e0_key = next(
        (
            key
            for key in header
            if key.startswith("E0_") and key != "E0_ref_Hartree"
        ),
        None,
    )
    if predicted_e0_key is None:
        raise KeyError("Failed to locate predicted E0 column in csv header.")

    objective_name = predicted_e0_key[len("E0_") : -len("_Hartree")]
    predicted_e1_key = f"E1_{objective_name}_Hartree"
    predicted_gap_key = f"Gap1_{objective_name}_eV"

    return (
        objective_name,
        _col("R_Angstrom"),
        _col("E0_ref_Hartree"),
        _col("E1_ref_Hartree"),
        _col("Gap1_ref_eV"),
        _col(predicted_e0_key),
        _col(predicted_e1_key),
        _col(predicted_gap_key),
    )


def _window_values(
    x: np.ndarray,
    y_ref: np.ndarray,
    y_pred: np.ndarray,
    x_lo: float,
    x_hi: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = (x >= x_lo) & (x <= x_hi)
    if not np.any(mask):
        raise ValueError(f"No data points in window [{x_lo}, {x_hi}]")
    return x[mask], y_ref[mask], y_pred[mask]


def _set_local_ylim(ax, ref_values: np.ndarray, pred_values: np.ndarray) -> None:
    y_all = np.concatenate([ref_values, pred_values])
    y_min = float(np.min(y_all))
    y_max = float(np.max(y_all))
    span = y_max - y_min
    pad = max(0.015 * max(1.0, abs(y_min), abs(y_max)), 0.10 * span, 1e-6)
    ax.set_ylim(y_min - pad, y_max + pad)


def _plot_row(
    axes: np.ndarray,
    *,
    r: np.ndarray,
    ref_values: np.ndarray,
    pred_values: np.ndarray,
    y_label: str,
    windows: list[tuple[str, float, float]],
) -> None:
    for ax, (title, x_lo, x_hi) in zip(axes, windows, strict=True):
        x_local, ref_local, pred_local = _window_values(r, ref_values, pred_values, x_lo, x_hi)
        ax.plot(x_local, ref_local, color=REFERENCE_COLOR, lw=2.2, label="FCI Reference")
        ax.plot(x_local, pred_local, color=PREDICTED_COLOR, lw=1.95, label="Neural_xc")
        ax.set_xlim(x_lo, x_hi)
        _set_local_ylim(ax, ref_local, pred_local)
        ax.grid(alpha=0.24)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("H-H Distance (Angstrom)")
        ax.set_ylabel(y_label)


def _objective_label(name: str) -> str:
    return name.replace("_", " ").title()


def _build_windows(
    r: np.ndarray,
    e0_ref: np.ndarray,
    *,
    short_hi: float,
    eq_half_width: float,
    mid_lo: float,
    mid_hi: float,
    asym_lo: float,
) -> list[tuple[str, float, float]]:
    r_min = float(np.min(r))
    r_max = float(np.max(r))
    eq_r = float(r[int(np.argmin(e0_ref))])
    eq_lo = max(r_min, eq_r - eq_half_width)
    eq_hi = min(r_max, eq_r + eq_half_width)

    windows = [
        ("Full Range", r_min, r_max),
        (f"Short Bond [{r_min:.2f}, {min(short_hi, r_max):.2f}] A", r_min, min(short_hi, r_max)),
        (f"Near Eq [{eq_lo:.2f}, {eq_hi:.2f}] A", eq_lo, eq_hi),
        (f"Mid Dissociation [{max(mid_lo, r_min):.2f}, {min(mid_hi, r_max):.2f}] A", max(mid_lo, r_min), min(mid_hi, r_max)),
        (f"Asymptotic [{max(asym_lo, r_min):.2f}, {r_max:.2f}] A", max(asym_lo, r_min), r_max),
    ]
    # Drop degenerate windows if the requested range collapses.
    filtered = []
    for title, x_lo, x_hi in windows:
        if x_hi - x_lo > 1e-8:
            filtered.append((title, x_lo, x_hi))
    return filtered


def _plot_key_regions(
    out_png: Path,
    *,
    method_label: str,
    objective_name: str,
    r: np.ndarray,
    e0_ref: np.ndarray,
    e1_ref: np.ndarray,
    gap_ref: np.ndarray,
    e0_pred: np.ndarray,
    e1_pred: np.ndarray,
    gap_pred: np.ndarray,
    windows: list[tuple[str, float, float]],
) -> None:
    ncols = len(windows)
    fig, axes = plt.subplots(3, ncols, figsize=(4.2 * ncols, 9.2))

    _plot_row(
        axes[0],
        r=r,
        ref_values=e0_ref,
        pred_values=e0_pred,
        y_label="E0 (Hartree)",
        windows=windows,
    )
    _plot_row(
        axes[1],
        r=r,
        ref_values=e1_ref,
        pred_values=e1_pred,
        y_label="E1 (Hartree)",
        windows=windows,
    )
    _plot_row(
        axes[2],
        r=r,
        ref_values=gap_ref,
        pred_values=gap_pred,
        y_label="S1 Gap (eV)",
        windows=windows,
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
        fontsize=11,
    )
    fig.suptitle(
        f"H2 Dense Dissociation | {method_label} | {_objective_label(objective_name)}",
        fontsize=14,
        y=1.02,
    )
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.07, top=0.88, wspace=0.26, hspace=0.32)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replot dense H2 dissociation csv with several key-region zoom panels."
    )
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out-png", type=str, required=True)
    parser.add_argument("--method-label", type=str, default="TDA")
    parser.add_argument("--short-hi", type=float, default=0.55)
    parser.add_argument("--eq-half-width", type=float, default=0.25)
    parser.add_argument("--mid-lo", type=float, default=1.0)
    parser.add_argument("--mid-hi", type=float, default=2.5)
    parser.add_argument("--asym-lo", type=float, default=3.0)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    (
        objective_name,
        r,
        e0_ref,
        e1_ref,
        gap_ref,
        e0_pred,
        e1_pred,
        gap_pred,
    ) = _load_curve_csv(csv_path)
    windows = _build_windows(
        r,
        e0_ref,
        short_hi=float(args.short_hi),
        eq_half_width=float(args.eq_half_width),
        mid_lo=float(args.mid_lo),
        mid_hi=float(args.mid_hi),
        asym_lo=float(args.asym_lo),
    )
    _plot_key_regions(
        out_png,
        method_label=str(args.method_label),
        objective_name=objective_name,
        r=r,
        e0_ref=e0_ref,
        e1_ref=e1_ref,
        gap_ref=gap_ref,
        e0_pred=e0_pred,
        e1_pred=e1_pred,
        gap_pred=gap_pred,
        windows=windows,
    )
    print(f"wrote={out_png}")
    print(f"windows={windows}")


if __name__ == "__main__":
    main()
