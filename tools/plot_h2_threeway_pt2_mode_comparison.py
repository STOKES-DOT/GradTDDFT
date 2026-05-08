from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


SERIES = [
    ("FCI", "fci", "#111111"),
    ("NN no PT2", "nopt2", "#1f77b4"),
    ("NN + PT2 scaled", "pt2_scaled_projected", "#d95f02"),
    ("NN + PT2 local", "pt2_local_exact", "#2ca02c"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the H2 equilibrium S1 spectrum comparing no-PT2, PT2 scaled_projected, and PT2 local_exact."
    )
    parser.add_argument(
        "--nopt2-summary",
        type=Path,
        default=Path(
            "outputs/h2_s1_tda_train5_dense100_fixed_density_ep2000_nopt2_mae_dm21hf/summary.json"
        ),
    )
    parser.add_argument(
        "--scaled-summary",
        type=Path,
        default=Path(
            "outputs/h2_s1_tda_train5_dense100_fixed_density_ep2000_pt2_scaled_projected_mae_dm21hf/summary.json"
        ),
    )
    parser.add_argument(
        "--local-summary",
        type=Path,
        default=Path(
            "outputs/h2_s1_tda_train5_dense100_fixed_density_ep2000_pt2_local_exact_mae_dm21hf/summary.json"
        ),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path(
            "outputs/h2_s1_tda_threeway_pt2_mode_compare/h2_equilibrium_s1_threeway_pt2_mode_compare.png"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/h2_s1_tda_threeway_pt2_mode_compare/h2_equilibrium_s1_threeway_pt2_mode_compare.json"
        ),
    )
    parser.add_argument(
        "--margin-ev",
        type=float,
        default=0.28,
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_spectrum(summary_path: Path) -> tuple[dict, dict]:
    summary = _load_json(summary_path)
    spectrum = _load_json(Path(summary["spectrum_json"]))
    return summary, spectrum


def main() -> None:
    args = parse_args()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)

    nopt2_summary, nopt2_spec = _load_spectrum(args.nopt2_summary)
    scaled_summary, scaled_spec = _load_spectrum(args.scaled_summary)
    local_summary, local_spec = _load_spectrum(args.local_summary)

    fci_line = nopt2_spec["fci_lines"][0]
    series = {
        "fci": {
            "label": "FCI",
            "excitation_ev": float(fci_line["excitation_ev"]),
            "oscillator_strength": float(fci_line["oscillator_strength"]),
            "s1_gap_mae_ev": 0.0,
        },
        "nopt2": {
            "label": "NN no PT2",
            "excitation_ev": float(nopt2_spec["neural_tda_lines"][0]["excitation_ev"]),
            "oscillator_strength": float(nopt2_spec["neural_tda_lines"][0]["oscillator_strength"]),
            "s1_gap_mae_ev": float(nopt2_summary["s1_gap_mae_ev"]),
        },
        "pt2_scaled_projected": {
            "label": "NN + PT2 scaled",
            "excitation_ev": float(scaled_spec["neural_tda_lines"][0]["excitation_ev"]),
            "oscillator_strength": float(scaled_spec["neural_tda_lines"][0]["oscillator_strength"]),
            "s1_gap_mae_ev": float(scaled_summary["s1_gap_mae_ev"]),
        },
        "pt2_local_exact": {
            "label": "NN + PT2 local",
            "excitation_ev": float(local_spec["neural_tda_lines"][0]["excitation_ev"]),
            "oscillator_strength": float(local_spec["neural_tda_lines"][0]["oscillator_strength"]),
            "s1_gap_mae_ev": float(local_summary["s1_gap_mae_ev"]),
        },
    }

    ordered = [
        (
            series[key]["label"],
            float(series[key]["excitation_ev"]),
            float(series[key]["oscillator_strength"]),
            color,
        )
        for label, key, color in SERIES
    ]

    energies = [energy for _, energy, _, _ in ordered]
    max_f = max(fosc for _, _, fosc, _ in ordered)
    xmin = min(energies) - float(args.margin_ev)
    xmax = max(energies) + float(args.margin_ev)

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    placements = [
        (-0.08, 0.05, "right"),
        (-0.05, 0.10, "right"),
        (0.00, 0.06, "center"),
        (0.08, 0.10, "left"),
    ]
    placement_by_label = {
        label: placements[idx]
        for idx, (label, _, _, _) in enumerate(sorted(ordered, key=lambda item: item[1]))
    }

    for label, energy, fosc, color in ordered:
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
    ax.set_title(f"H2 Equilibrium S1 at R = {nopt2_spec['r_angstrom']:.2f} A", pad=14)
    ax.grid(alpha=0.2, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(args.output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "r_angstrom": float(nopt2_spec["r_angstrom"]),
        "basis": str(nopt2_summary["basis"]),
        "training_mode": str(nopt2_summary["training_mode"]),
        "objective": str(nopt2_summary["objective"]),
        "series": series,
        "output_png": str(args.output_png),
    }
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
