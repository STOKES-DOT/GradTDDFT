from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OBJECTIVE_LABELS = {
    "ground_only": "Ground Only",
    "ground_plus_s1": "Ground + S1 Gap",
    "s1_only": "S1 Gap Only",
    "e1_total_only": "E1 Total Only",
}

OBJECTIVE_COLORS = {
    "ground_only": "#1d4ed8",
    "ground_plus_s1": "#059669",
    "s1_only": "#c2410c",
    "e1_total_only": "#7c3aed",
}

REFERENCE_COLOR = "#111827"
ZOOM_HIGHLIGHT = "#cbd5e1"


def _load_dense_curve_csv(path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not rows:
            raise ValueError(f"No rows found in {path}")
        fieldnames = list(reader.fieldnames or [])

    data: dict[str, np.ndarray] = {}
    for name in fieldnames:
        data[name] = np.asarray([float(row[name]) for row in rows], dtype=float)

    objectives: list[str] = []
    for name in fieldnames:
        if name.startswith("E0_") and name.endswith("_Hartree") and name != "E0_ref_Hartree":
            objectives.append(name[len("E0_") : -len("_Hartree")])
    return data, objectives


def _expanded_limits(a: np.ndarray, b: np.ndarray, *, pad_ratio: float = 0.08) -> tuple[float, float]:
    values = np.concatenate([np.asarray(a, dtype=float), np.asarray(b, dtype=float)])
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    span = max(vmax - vmin, 1e-8)
    pad = span * pad_ratio
    return vmin - pad, vmax + pad


def _compute_metrics(data: dict[str, np.ndarray], objective: str) -> tuple[float, float, float]:
    e0_ref = data["E0_ref_Hartree"]
    e1_ref = data["E1_ref_Hartree"]
    gap_ref = data["Gap1_ref_eV"]
    e0_pred = data[f"E0_{objective}_Hartree"]
    e1_pred = data[f"E1_{objective}_Hartree"]
    gap_pred = data[f"Gap1_{objective}_eV"]
    hartree_to_ev = 27.211386245988
    mae_ground = float(np.mean(np.abs(e0_pred - e0_ref)) * hartree_to_ev)
    mae_excited = float(np.mean(np.abs(e1_pred - e1_ref)) * hartree_to_ev)
    mae_gap = float(np.mean(np.abs(gap_pred - gap_ref)))
    return mae_ground, mae_excited, mae_gap


def _plot_objective(
    *,
    data: dict[str, np.ndarray],
    objective: str,
    zoom_r_min: float,
    zoom_r_max: float,
    outpath: Path,
) -> None:
    r = data["R_Angstrom"]
    zoom_mask = (r >= float(zoom_r_min)) & (r <= float(zoom_r_max))
    if not np.any(zoom_mask):
        raise ValueError("Zoom window does not overlap the curve grid.")

    color = OBJECTIVE_COLORS.get(objective, "#2563eb")
    label = OBJECTIVE_LABELS.get(objective, objective)
    e0_ref = data["E0_ref_Hartree"]
    e1_ref = data["E1_ref_Hartree"]
    gap_ref = data["Gap1_ref_eV"]
    e0_pred = data[f"E0_{objective}_Hartree"]
    e1_pred = data[f"E1_{objective}_Hartree"]
    gap_pred = data[f"Gap1_{objective}_eV"]
    mae_ground, mae_excited, mae_gap = _compute_metrics(data, objective)

    fig, axes = plt.subplots(3, 2, figsize=(13.0, 11.2), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.08, hspace=0.03, wspace=0.02)

    row_specs = [
        ("Ground-State Energy", "Energy (Hartree)", e0_ref, e0_pred),
        ("First-Excited Energy", "Energy (Hartree)", e1_ref, e1_pred),
        ("S1 Gap", "Excitation Energy (eV)", gap_ref, gap_pred),
    ]

    for row_idx, (title, ylabel, ref_values, pred_values) in enumerate(row_specs):
        ax_full = axes[row_idx, 0]
        ax_zoom = axes[row_idx, 1]

        ax_full.plot(r, ref_values, color=REFERENCE_COLOR, lw=2.35, label="FCI Reference")
        ax_full.plot(r, pred_values, color=color, lw=2.0, label=label)
        ax_full.axvspan(
            float(zoom_r_min),
            float(zoom_r_max),
            color=ZOOM_HIGHLIGHT,
            alpha=0.2,
            zorder=0,
        )
        ax_full.set_title(f"{title} | Full Range", fontsize=11.5)
        ax_full.set_ylabel(ylabel)
        ax_full.grid(alpha=0.22, linewidth=0.7)

        ax_zoom.plot(r, ref_values, color=REFERENCE_COLOR, lw=2.35)
        ax_zoom.plot(r, pred_values, color=color, lw=2.0)
        ax_zoom.set_title(f"{title} | Zoomed Region", fontsize=11.5)
        ax_zoom.set_xlim(float(zoom_r_min), float(zoom_r_max))
        ax_zoom.set_ylim(*_expanded_limits(ref_values[zoom_mask], pred_values[zoom_mask]))
        ax_zoom.grid(alpha=0.22, linewidth=0.7)

        if row_idx == 0:
            ax_full.legend(
                loc="best",
                frameon=True,
                framealpha=0.92,
                edgecolor="#d1d5db",
                fontsize=9.5,
            )

    axes[2, 0].set_xlabel("H-H Distance (Angstrom)")
    axes[2, 1].set_xlabel("H-H Distance (Angstrom)")

    fig.suptitle(
        (
            f"H2 Dissociation Curves | {label}\n"
            f"MAE(E0) = {mae_ground:.3f} eV | "
            f"MAE(E1) = {mae_excited:.3f} eV | "
            f"MAE(S1 Gap) = {mae_gap:.3f} eV"
        ),
        fontsize=15,
    )
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replot one-figure-per-objective H2 dense dissociation curves from an existing CSV."
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to dense_dissociation_curves.csv")
    parser.add_argument("--zoom-r-min", type=float, default=0.5)
    parser.add_argument("--zoom-r-max", type=float, default=5.0)
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=None,
        help="Subset of objectives to plot. Defaults to all objectives found in the CSV.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Directory to store the per-objective PNG figures.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data, discovered_objectives = _load_dense_curve_csv(csv_path)
    objectives = discovered_objectives if args.objectives is None else list(args.objectives)
    for objective in objectives:
        if objective not in discovered_objectives:
            raise ValueError(f"Objective {objective!r} not found in {csv_path}")
        _plot_objective(
            data=data,
            objective=objective,
            zoom_r_min=float(args.zoom_r_min),
            zoom_r_max=float(args.zoom_r_max),
            outpath=outdir / f"{objective}_curve_report.png",
        )


if __name__ == "__main__":
    main()
