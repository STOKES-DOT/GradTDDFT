#!/usr/bin/env python3
"""Render the H2 S1-only TDA HF-channel curve from dense benchmark CSV."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-tdgraddft")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D


HARTREE_TO_EV = 27.211386245988

ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "h2_s1_tda_dense_curve.csv"
VIS_CSV = ROOT / "h2_s1_hf_curve_visualization_data.csv"
OUT_PNG = ROOT / "h2_s1_hf_curve.png"
OUT_SVG = ROOT / "h2_s1_hf_curve.svg"

TRAIN_R = np.array([0.4, 1.8, 3.2, 4.6, 6.0], dtype=float)

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
BLUE = {"base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"}
ORANGE = {"base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"}
GOLD = {"base": "#FFE15B", "mid": "#B8A037", "dark": "#736422"}
NEUTRAL = {"base": "#C5CAD3", "mid": "#7A828F", "dark": "#464C55"}


def load_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: float(value) for key, value in row.items()} for row in reader]


def write_visual_csv(rows: list[dict[str, float]]) -> None:
    fieldnames = [
        "r_angstrom",
        "is_train_point",
        "fci_s0_energy_h",
        "neural_s0_energy_h",
        "fci_s1_total_energy_h",
        "neural_s1_total_energy_h",
        "fci_s1_excitation_ev",
        "neural_s1_excitation_ev",
        "s0_abs_error_ev",
        "s1_gap_abs_error_ev",
        "neural_s1_oscillator_strength",
    ]
    with VIS_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            r = row["r_angstrom"]
            out = {
                "r_angstrom": r,
                "is_train_point": int(np.any(np.isclose(TRAIN_R, r, atol=2.0e-3))),
                "fci_s0_energy_h": row["fci_energy_h"],
                "neural_s0_energy_h": row["predicted_energy_h"],
                "fci_s1_total_energy_h": row["fci_energy_h"] + row["fci_s1_h"],
                "neural_s1_total_energy_h": row["predicted_energy_h"] + row["predicted_s1_h"],
                "fci_s1_excitation_ev": row["fci_s1_h"] * HARTREE_TO_EV,
                "neural_s1_excitation_ev": row["predicted_s1_h"] * HARTREE_TO_EV,
                "s0_abs_error_ev": row["energy_abs_err_ev"],
                "s1_gap_abs_error_ev": row["s1_gap_abs_err_ev"],
                "neural_s1_oscillator_strength": row[
                    "predicted_s1_oscillator_strength"
                ],
            }
            writer.writerow(out)


def add_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(
        0.075,
        0.985,
        title,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.075,
        0.952,
        subtitle,
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
    )


def style_axis(ax: plt.Axes, *, xlabel: str | None = None, ylabel: str) -> None:
    ax.set_facecolor(TOKENS["panel"])
    ax.grid(True, axis="both", color=TOKENS["grid"], linestyle=":", linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.tick_params(axis="both", colors=TOKENS["muted"], labelsize=8.5, length=0)
    ax.set_ylabel(ylabel, fontsize=9, color=TOKENS["ink"])
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=9, color=TOKENS["ink"])
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)


def mark_training_points(ax: plt.Axes) -> None:
    ymin, ymax = ax.get_ylim()
    for r in TRAIN_R:
        ax.axvline(r, color=NEUTRAL["base"], linestyle=":", linewidth=0.7, zorder=0)
    ax.set_ylim(ymin, ymax)


def plot_curve(ax: plt.Axes, r: np.ndarray, ref: np.ndarray, pred: np.ndarray) -> None:
    ax.plot(r, ref, color=BLUE["dark"], linewidth=1.35, label="FCI reference")
    ax.plot(
        r,
        pred,
        color=ORANGE["mid"],
        linewidth=1.35,
        linestyle="--",
        label="Neural XC (HF channel)",
    )
    ax.scatter(
        TRAIN_R,
        np.interp(TRAIN_R, r, pred),
        s=20,
        color=GOLD["base"],
        edgecolor=GOLD["dark"],
        linewidth=0.7,
        zorder=4,
        label="Training geometries",
    )


def main() -> None:
    rows = load_rows(INPUT_CSV)
    write_visual_csv(rows)

    r = np.array([row["r_angstrom"] for row in rows], dtype=float)
    fci_s0 = np.array([row["fci_energy_h"] for row in rows], dtype=float)
    pred_s0 = np.array([row["predicted_energy_h"] for row in rows], dtype=float)
    fci_gap_ev = np.array([row["fci_s1_h"] * HARTREE_TO_EV for row in rows], dtype=float)
    pred_gap_ev = np.array(
        [row["predicted_s1_h"] * HARTREE_TO_EV for row in rows], dtype=float
    )
    fci_s1_total = np.array(
        [row["fci_energy_h"] + row["fci_s1_h"] for row in rows], dtype=float
    )
    pred_s1_total = np.array(
        [row["predicted_energy_h"] + row["predicted_s1_h"] for row in rows],
        dtype=float,
    )
    s0_err_ev = np.array([row["energy_abs_err_ev"] for row in rows], dtype=float)
    gap_err_ev = np.array([row["s1_gap_abs_err_ev"] for row in rows], dtype=float)

    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.09, top=0.86, wspace=0.24, hspace=0.27)
    add_header(
        fig,
        "H2 S1-only TDA training with the HF channel",
        "B3LYP/def2-SVP; five training geometries marked in gold; dense evaluation on 100 bond lengths.",
    )

    ax = axes[0, 0]
    plot_curve(ax, r, fci_s0, pred_s0)
    style_axis(ax, ylabel="S0 total energy (Eh)")
    ax.text(0.01, 0.96, "A", transform=ax.transAxes, fontweight="bold", color=TOKENS["ink"])
    mark_training_points(ax)

    ax = axes[0, 1]
    plot_curve(ax, r, fci_gap_ev, pred_gap_ev)
    style_axis(ax, ylabel="S1 excitation energy (eV)")
    ax.text(0.01, 0.96, "B", transform=ax.transAxes, fontweight="bold", color=TOKENS["ink"])
    mark_training_points(ax)

    ax = axes[1, 0]
    plot_curve(ax, r, fci_s1_total, pred_s1_total)
    style_axis(ax, xlabel="H-H distance (Angstrom)", ylabel="S1 total energy (Eh)")
    ax.text(0.01, 0.96, "C", transform=ax.transAxes, fontweight="bold", color=TOKENS["ink"])
    mark_training_points(ax)

    ax = axes[1, 1]
    ax.plot(r, s0_err_ev, color=BLUE["dark"], linewidth=1.25, label="S0 total error")
    ax.plot(
        r,
        gap_err_ev,
        color=ORANGE["mid"],
        linewidth=1.25,
        linestyle="--",
        label="S1 gap error",
    )
    ax.scatter(
        TRAIN_R,
        np.interp(TRAIN_R, r, gap_err_ev),
        s=20,
        color=GOLD["base"],
        edgecolor=GOLD["dark"],
        linewidth=0.7,
        zorder=4,
    )
    style_axis(ax, xlabel="H-H distance (Angstrom)", ylabel="Absolute error (eV)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.text(0.01, 0.96, "D", transform=ax.transAxes, fontweight="bold", color=TOKENS["ink"])
    ax.legend(loc="upper left", frameon=False, fontsize=8.2, handlelength=2.4)
    mark_training_points(ax)

    handles = [
        Line2D([0], [0], color=BLUE["dark"], linewidth=1.35, label="FCI reference"),
        Line2D(
            [0],
            [0],
            color=ORANGE["mid"],
            linewidth=1.35,
            linestyle="--",
            label="Neural XC (HF channel)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=GOLD["base"],
            markeredgecolor=GOLD["dark"],
            markersize=5,
            label="Training geometries",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.075, 0.912),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        handlelength=2.8,
    )

    for ax in axes.flat:
        ax.set_xlim(r.min(), r.max())

    fig.savefig(OUT_PNG, dpi=320)
    fig.savefig(OUT_SVG)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_SVG}")
    print(f"Wrote {VIS_CSV}")


if __name__ == "__main__":
    main()
