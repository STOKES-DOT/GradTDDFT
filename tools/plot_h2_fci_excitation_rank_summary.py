from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a summary figure for H2 FCI excitation-rank analysis."
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=(
            "outputs/h2_fci_excitation_rank_analysis_631gstar_equilibrium/"
            "h2_fci_excitation_rank_report.json"
        ),
        help="Input report JSON produced by h2_fci_excitation_rank_analysis.py",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=(
            "outputs/h2_fci_excitation_rank_analysis_631gstar_equilibrium/"
            "h2_fci_excitation_rank_summary.png"
        ),
        help="Output summary figure path",
    )
    return parser.parse_args()


def _state_sort_key(label: str) -> int:
    if label.startswith("S") and label[1:].isdigit():
        return int(label[1:])
    return 9999


def main() -> None:
    args = parse_args()
    report_path = Path(args.report_json)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    states = sorted(report["states"], key=lambda s: _state_sort_key(s["state_label"]))
    state_labels = [s["state_label"] for s in states]

    ranks = sorted(
        {
            int(row["rank"])
            for state in states
            for row in state.get("rank_summary", [])
        }
    )
    rank_to_idx = {rank: idx for idx, rank in enumerate(ranks)}

    n_states = len(states)
    n_ranks = len(ranks)

    exc_ev = np.asarray([float(s["excitation_energy_ev"]) for s in states], dtype=np.float64)
    weights = np.zeros((n_states, n_ranks), dtype=np.float64)
    e_rank_total = np.zeros((n_states, n_ranks), dtype=np.float64)
    for is_state, state in enumerate(states):
        for row in state["rank_summary"]:
            ir = rank_to_idx[int(row["rank"])]
            weights[is_state, ir] = float(row["weight_percent"])
            e_rank_total[is_state, ir] = float(row["energy_total_h"])

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))

    # Panel 1: excitation energies.
    ax0 = axes[0]
    x = np.arange(n_states, dtype=np.float64)
    bar = ax0.bar(x, exc_ev, color="#4C78A8", width=0.62)
    ax0.set_xticks(x, state_labels)
    ax0.set_ylabel("Excitation Energy (eV)")
    ax0.set_title("FCI Singlet Excitation Energies")
    ax0.grid(axis="y", alpha=0.25)
    for rect, val in zip(bar, exc_ev, strict=True):
        ax0.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + 0.15,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Panel 2: CI weight by excitation rank.
    ax1 = axes[1]
    bottoms = np.zeros((n_states,), dtype=np.float64)
    cmap = plt.get_cmap("Set2")
    for ir, rank in enumerate(ranks):
        vals = weights[:, ir]
        ax1.bar(
            x,
            vals,
            bottom=bottoms,
            width=0.62,
            label=f"Rank {rank}",
            color=cmap(ir % cmap.N),
        )
        bottoms += vals
    ax1.set_xticks(x, state_labels)
    ax1.set_ylabel("CI Weight (%)")
    ax1.set_ylim(0.0, 100.0)
    ax1.set_title("Wavefunction Composition by Excitation Rank")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(frameon=False, loc="upper right")

    # Panel 3: energy contribution by rank (c_r^T H c).
    ax2 = axes[2]
    width = 0.7 / max(1, n_ranks)
    for ir, rank in enumerate(ranks):
        offset = (ir - (n_ranks - 1) / 2.0) * width
        ax2.bar(
            x + offset,
            e_rank_total[:, ir],
            width=width,
            label=f"Rank {rank}",
            color=cmap(ir % cmap.N),
        )
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_xticks(x, state_labels)
    ax2.set_ylabel("Energy Contribution (Hartree)")
    ax2.set_title("Rank-Resolved Energy Contribution")
    ax2.grid(axis="y", alpha=0.25)

    system = report["system"]
    fig.suptitle(
        "H2 FCI Excited-State Rank Analysis  |  "
        f"R={system['bond_length_angstrom']:.4f} Å, {system['basis']}",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"[done] wrote figure: {out_path}")


if __name__ == "__main__":
    main()
