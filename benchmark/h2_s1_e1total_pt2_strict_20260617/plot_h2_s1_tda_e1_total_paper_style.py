from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory


ROOT = Path("/Users/jiaoyuan/Documents/GitHub/TD-GradDFT")
RUN_DIR = ROOT / "benchmark/h2_s1_e1total_pt2_strict_20260617"
PAPER_FIG_DIRS = [
    ROOT / "paper/tdgraddft-paper/figures",
    ROOT / "paper_review/tdgraddft-paper/figures",
]
CSV_PATH = RUN_DIR / "h2_s1_tda_dense_curve.csv"
SUMMARY_PATH = RUN_DIR / "summary.json"
OUT_STEM = "h2_s1_tda_e1_total_dissociation_paper_style"
EV_PER_H = 27.211386245988
KCAL_PER_MOL_TO_MEV = 43.3641153087705


def read_rows() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with CSV_PATH.open(newline="") as handle:
        for row in csv.DictReader(handle):
            ref = float(row["fci_s1_total_energy_h"])
            pred = float(row["predicted_s1_total_energy_h"])
            rows.append(
                {
                    "r_angstrom": float(row["r_angstrom"]),
                    "reference_energy_h": ref,
                    "predicted_energy_h": pred,
                    "signed_error_mev": (pred - ref) * EV_PER_H * 1000.0,
                    "s1_total_abs_err_ev": float(row["s1_total_abs_err_ev"]),
                }
            )
    rows.sort(key=lambda item: item["r_angstrom"])
    return rows


def interp(x_values: list[float], y_values: list[float], x: float) -> float:
    if x <= x_values[0]:
        return y_values[0]
    if x >= x_values[-1]:
        return y_values[-1]
    for left_idx in range(len(x_values) - 1):
        x0 = x_values[left_idx]
        x1 = x_values[left_idx + 1]
        if x0 <= x <= x1:
            y0 = y_values[left_idx]
            y1 = y_values[left_idx + 1]
            weight = (x - x0) / (x1 - x0)
            return y0 + weight * (y1 - y0)
    return y_values[-1]


def setup_axes(ax: plt.Axes) -> None:
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=10.5,
        width=1.65,
        length=4.8,
        direction="in",
        top=True,
        right=True,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        width=1.2,
        length=2.6,
        direction="in",
        top=True,
        right=True,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.65)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def main() -> None:
    rows = read_rows()
    summary = json.loads(SUMMARY_PATH.read_text())
    x_values = [row["r_angstrom"] for row in rows]
    reference_values = [row["reference_energy_h"] for row in rows]
    predicted_values = [row["predicted_energy_h"] for row in rows]
    errors = [row["signed_error_mev"] for row in rows]
    abs_errors = [abs(value) for value in errors]
    max_index = max(range(len(rows)), key=lambda index: abs_errors[index])
    mae_ev = sum(abs_errors) / len(abs_errors) / 1000.0
    max_abs_ev = abs_errors[max_index] / 1000.0
    train_values = [float(value) for value in summary["train_r_values_angstrom"]]
    train_reference = [interp(x_values, reference_values, value) for value in train_values]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(3.05, 3.55),
        sharex=True,
        gridspec_kw={"height_ratios": [2.45, 1.0], "hspace": 0.08},
    )
    fig.patch.set_facecolor("white")
    ax_top.set_facecolor("white")
    ax_bottom.set_facecolor("white")

    ax_top.plot(
        x_values,
        reference_values,
        color="#111111",
        lw=2.35,
        ls=(0, (5.2, 3.2)),
        label="FCI reference",
        zorder=4,
    )
    ax_top.plot(
        x_values,
        predicted_values,
        color="#d95f02",
        lw=2.45,
        label=f"Neural TDA ({mae_ev * 1000.0:.1f} meV)",
        zorder=5,
    )
    ax_top.scatter(
        train_values,
        train_reference,
        marker="D",
        s=37,
        facecolor="#b2182b",
        edgecolor="white",
        linewidth=0.9,
        label="Training points",
        zorder=7,
    )
    ax_top.text(
        0.10,
        0.76,
        r"H$_2$ $B\,{}^1\Sigma_u^+$",
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=12.8,
        fontweight="bold",
    )
    ax_top.set_ylabel(r"$E_1(R)$ (Ha)", fontsize=12.5, fontweight="bold", labelpad=2.0)
    ax_top.legend(
        loc="upper right",
        fontsize=6.8,
        frameon=False,
        handlelength=2.4,
        borderaxespad=0.2,
        labelspacing=0.30,
    )
    ax_top.grid(False)

    ax_bottom.axhspan(
        -KCAL_PER_MOL_TO_MEV,
        KCAL_PER_MOL_TO_MEV,
        color="0.86",
        alpha=0.78,
        zorder=0,
    )
    ax_bottom.axhline(0.0, color="0.25", lw=1.15, zorder=2)
    ax_bottom.plot(x_values, errors, color="#d95f02", lw=2.05, zorder=4)
    ax_bottom.scatter(
        train_values,
        [0.0 for _ in train_values],
        marker="D",
        s=34,
        facecolor="#b2182b",
        edgecolor="white",
        linewidth=0.9,
        zorder=7,
    )
    band_label_transform = blended_transform_factory(ax_bottom.transAxes, ax_bottom.transData)
    ax_bottom.text(
        0.95,
        0.72 * KCAL_PER_MOL_TO_MEV,
        r"$\pm 1$ kcal mol$^{-1}$",
        transform=band_label_transform,
        ha="right",
        va="center",
        fontsize=8.6,
        fontweight="bold",
        color="0.35",
        zorder=6,
    )
    ax_bottom.set_xlabel(r"$R$ ($\AA$)", fontsize=13.0, fontweight="bold", labelpad=5.0)
    ax_bottom.set_ylabel(r"$\Delta E_1$ (meV)", fontsize=12.0, fontweight="bold", labelpad=2.0)
    ax_bottom.set_xticks([1, 2, 3, 4, 5, 6])
    ax_bottom.grid(False)

    x_min, x_max = min(x_values), max(x_values)
    ax_top.set_xlim(x_min - 0.03 * (x_max - x_min), x_max + 0.03 * (x_max - x_min))
    y_all = reference_values + predicted_values
    y_min, y_max = min(y_all), max(y_all)
    y_span = y_max - y_min
    ax_top.set_ylim(y_min - 0.04 * y_span, y_max + 0.08 * y_span)
    err_limit = max(max(abs_errors) * 1.22, KCAL_PER_MOL_TO_MEV * 1.55)
    ax_bottom.set_ylim(-err_limit, err_limit)
    for axis in (ax_top, ax_bottom):
        setup_axes(axis)

    fig.align_ylabels([ax_top, ax_bottom])
    fig.subplots_adjust(left=0.30, right=0.985, top=0.985, bottom=0.155)

    output_dirs = [RUN_DIR, *[directory for directory in PAPER_FIG_DIRS if directory.exists()]]
    outputs: dict[str, str] = {}
    for directory in output_dirs:
        base = directory / OUT_STEM
        fig.savefig(base.with_suffix(".pdf"))
        fig.savefig(base.with_suffix(".png"), dpi=420)
        fig.savefig(base.with_suffix(".svg"))
        outputs[str(directory)] = str(base.with_suffix(".pdf"))
    plt.close(fig)

    metrics = {
        "system": "H2",
        "state": "S1",
        "spectroscopic_state": "B 1Sigma_u+",
        "quantity": "S1 total energy E1(R)",
        "solver": "tda",
        "basis": summary["basis"],
        "objective": summary["objective"],
        "response_hf_mode": summary["response_hf_mode"],
        "response_pt2_mode": summary["response_pt2_mode"],
        "pt2_channel_mode": summary["pt2_channel_mode"],
        "dense_mae_ev_from_csv": mae_ev,
        "summary_s1_total_mae_ev": float(summary["s1_total_mae_ev"]),
        "dense_max_abs_error_ev_from_csv": max_abs_ev,
        "dense_max_abs_error_r_angstrom": rows[max_index]["r_angstrom"],
        "summary_s1_total_max_ev": float(summary["s1_total_max_ev"]),
        "delta_e_unit": "meV",
        "chemical_accuracy_band_mev": KCAL_PER_MOL_TO_MEV,
        "source_csv": str(CSV_PATH),
        "outputs": outputs,
    }
    (RUN_DIR / f"{OUT_STEM}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
