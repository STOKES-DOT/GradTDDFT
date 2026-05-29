from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))
matplotlib.use("Agg")

import matplotlib.pyplot as plt


SERIES = [
    ("FCI", "fci", "#111111"),
    ("NN + PT2", "neural_tda_pt2_mae", "#d95f02"),
    ("NN no PT2", "neural_tda_nopt2_mae", "#1f77b4"),
    ("B3LYP TDA", "pyscf_b3lyp_tda", "#1b9e77"),
    ("B3LYP TDDFT", "pyscf_b3lyp_tddft", "#cc79a7"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redraw the H2 equilibrium spectrum using only the S1 lines from local JSON data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/h2_s1_tda_pt2localresp_mae_ablation/h2_equilibrium_tda_pt2localresp_mae_ablation_b3lyp_vs_fci.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/h2_s1_tda_pt2localresp_mae_ablation/h2_equilibrium_s1_only_zoom_b3lyp_vs_fci.png"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="CSV table containing the S1 stick-spectrum values used by the plot.",
    )
    parser.add_argument(
        "--margin-ev",
        type=float,
        default=0.25,
        help="Horizontal margin added on both sides of the S1 window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(args.input.read_text())

    series: list[tuple[str, float, float, str]] = []
    for label, key, color in SERIES:
        if key == "fci":
            point = data["fci"][0]
        else:
            point = data[key]
        series.append((label, float(point["excitation_ev"]), float(point["oscillator_strength"]), color))

    energies = [energy for _, energy, _, _ in series]
    max_f = max(fosc for _, _, fosc, _ in series)
    xmin = min(energies) - args.margin_ev
    xmax = max(energies) + args.margin_ev

    fig, ax = plt.subplots(figsize=(8.2, 5.8))

    placements = [
        (-0.10, 0.10, "right"),
        (-0.06, 0.03, "right"),
        (0.00, 0.07, "center"),
        (-0.05, 0.09, "right"),
        (0.09, 0.05, "left"),
    ]
    placement_by_label = {
        label: placements[idx]
        for idx, (label, _, _, _) in enumerate(sorted(series, key=lambda item: item[1]))
    }

    for label, energy, fosc, color in series:
        ax.vlines(energy, 0.0, fosc, color=color, linewidth=3.0)
        ax.plot(energy, fosc, "o", color=color, markersize=10)
        text_dx, text_dy, ha = placement_by_label[label]
        ax.annotate(
            f"{label}\n{energy:.2f} eV",
            xy=(energy, fosc),
            xytext=(energy + text_dx, fosc + text_dy),
            color=color,
            fontsize=10,
            ha=ha,
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.18",
                "fc": "white",
                "ec": "none",
                "alpha": 0.78,
            },
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.0, max_f + 0.18)
    ax.set_xlabel("Excitation Energy (eV)")
    ax.set_ylabel("Oscillator Strength")
    ax.set_title(f"H2 Equilibrium S1 Stick Spectrum at R = {data['r_angstrom']:.2f} A", pad=14)
    ax.grid(alpha=0.2, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    output_csv = args.output_csv if args.output_csv is not None else args.output.with_suffix(".csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "series_key",
                "label",
                "r_angstrom",
                "excitation_ev",
                "oscillator_strength",
            ]
        )
        for label, key, _ in SERIES:
            if key == "fci":
                point = data["fci"][0]
            else:
                point = data[key]
            writer.writerow(
                [
                    key,
                    label,
                    float(data["r_angstrom"]),
                    float(point["excitation_ev"]),
                    float(point["oscillator_strength"]),
                ]
            )


if __name__ == "__main__":
    main()
