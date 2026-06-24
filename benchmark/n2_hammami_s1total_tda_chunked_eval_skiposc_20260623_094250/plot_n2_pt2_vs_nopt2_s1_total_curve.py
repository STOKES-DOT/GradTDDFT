from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/Users/jiaoyuan/Documents/GitHub/TD-GradDFT")
RUN_DIR = ROOT / "benchmark/n2_hammami_s1total_tda_chunked_eval_skiposc_20260623_094250"
CSV_PATHS = {
    "no_pt2": RUN_DIR / "nopt2_dense_curve_35pt_merged.csv",
    "pt2_strict": RUN_DIR / "pt2strict_dense_curve_35pt_merged.csv",
}
OUT_STEM = "n2_pt2_vs_nopt2_s1_total_dissociation_curve"
EV_PER_HARTREE = 27.211386245988
TRAIN_R_VALUES = [0.8, 1.1, 1.6, 2.0, 2.2, 2.5, 3.0]


def _read_curve(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            reference_h = float(row["fci_s1_total_energy_h"])
            predicted_h = float(row["predicted_s1_total_energy_h"])
            rows.append(
                {
                    "dense_index": float(row["dense_index"]),
                    "r_angstrom": float(row["r_angstrom"]),
                    "reference_s1_total_h": reference_h,
                    "predicted_s1_total_h": predicted_h,
                    "signed_error_ev": (predicted_h - reference_h) * EV_PER_HARTREE,
                    "abs_error_ev": float(row["s1_total_abs_err_ev"]),
                }
            )
    rows.sort(key=lambda item: item["r_angstrom"])
    return rows


def _interp(x_values: list[float], y_values: list[float], x: float) -> float:
    if x <= x_values[0]:
        return y_values[0]
    if x >= x_values[-1]:
        return y_values[-1]
    for index in range(len(x_values) - 1):
        x0 = x_values[index]
        x1 = x_values[index + 1]
        if x0 <= x <= x1:
            y0 = y_values[index]
            y1 = y_values[index + 1]
            weight = (x - x0) / (x1 - x0)
            return y0 + weight * (y1 - y0)
    return y_values[-1]


def _setup_axis(ax: plt.Axes) -> None:
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=10.0,
        width=1.55,
        length=4.6,
        direction="in",
        top=True,
        right=True,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        width=1.05,
        length=2.5,
        direction="in",
        top=True,
        right=True,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.55)


def _summary(rows: list[dict[str, float]]) -> dict[str, float]:
    abs_errors = [row["abs_error_ev"] for row in rows]
    max_index = max(range(len(rows)), key=lambda index: abs_errors[index])
    return {
        "mae_ev": sum(abs_errors) / len(abs_errors),
        "max_abs_error_ev": abs_errors[max_index],
        "max_abs_error_r_angstrom": rows[max_index]["r_angstrom"],
    }


def _write_visualization_csv(
    no_pt2: list[dict[str, float]], pt2_strict: list[dict[str, float]], path: Path
) -> None:
    fieldnames = [
        "dense_index",
        "r_angstrom",
        "hammami_s1_total_hartree",
        "neural_nopt2_s1_total_hartree",
        "neural_pt2strict_s1_total_hartree",
        "nopt2_signed_error_ev",
        "pt2strict_signed_error_ev",
        "nopt2_abs_error_ev",
        "pt2strict_abs_error_ev",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_no, row_pt2 in zip(no_pt2, pt2_strict, strict=True):
            writer.writerow(
                {
                    "dense_index": int(row_no["dense_index"]),
                    "r_angstrom": f"{row_no['r_angstrom']:.12g}",
                    "hammami_s1_total_hartree": f"{row_no['reference_s1_total_h']:.12g}",
                    "neural_nopt2_s1_total_hartree": f"{row_no['predicted_s1_total_h']:.12g}",
                    "neural_pt2strict_s1_total_hartree": f"{row_pt2['predicted_s1_total_h']:.12g}",
                    "nopt2_signed_error_ev": f"{row_no['signed_error_ev']:.12g}",
                    "pt2strict_signed_error_ev": f"{row_pt2['signed_error_ev']:.12g}",
                    "nopt2_abs_error_ev": f"{row_no['abs_error_ev']:.12g}",
                    "pt2strict_abs_error_ev": f"{row_pt2['abs_error_ev']:.12g}",
                }
            )


def main() -> None:
    no_pt2 = _read_curve(CSV_PATHS["no_pt2"])
    pt2_strict = _read_curve(CSV_PATHS["pt2_strict"])
    x_values = [row["r_angstrom"] for row in no_pt2]
    reference_values = [row["reference_s1_total_h"] for row in no_pt2]
    no_pt2_values = [row["predicted_s1_total_h"] for row in no_pt2]
    pt2_values = [row["predicted_s1_total_h"] for row in pt2_strict]
    no_pt2_errors = [row["signed_error_ev"] for row in no_pt2]
    pt2_errors = [row["signed_error_ev"] for row in pt2_strict]
    no_pt2_summary = _summary(no_pt2)
    pt2_summary = _summary(pt2_strict)
    train_reference = [
        _interp(x_values, reference_values, train_r) for train_r in TRAIN_R_VALUES
    ]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(4.35, 3.95),
        sharex=True,
        gridspec_kw={"height_ratios": [2.35, 1.0], "hspace": 0.08},
    )
    fig.patch.set_facecolor("white")

    ax_top.plot(
        x_values,
        reference_values,
        color="#111111",
        lw=2.25,
        ls=(0, (5.0, 3.0)),
        label="Large-CAS reference",
        zorder=4,
    )
    ax_top.plot(
        x_values,
        no_pt2_values,
        color="#d95f02",
        lw=2.25,
        label=f"no-PT2 MAE={no_pt2_summary['mae_ev']:.2f} eV",
        zorder=5,
    )
    ax_top.plot(
        x_values,
        pt2_values,
        color="#1b9e77",
        lw=2.25,
        label=f"strict PT2 MAE={pt2_summary['mae_ev']:.2f} eV",
        zorder=6,
    )
    ax_top.scatter(
        TRAIN_R_VALUES,
        train_reference,
        marker="D",
        s=31,
        facecolor="#b2182b",
        edgecolor="white",
        linewidth=0.8,
        label="Training R",
        zorder=7,
    )
    ax_top.text(
        0.075,
        0.76,
        r"N$_2$ S$_1$ total",
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=12.2,
        fontweight="bold",
    )
    ax_top.set_ylabel(r"$E_1(R)$ (Ha)", fontsize=11.2, fontweight="bold")
    ax_top.legend(
        loc="upper right",
        fontsize=6.3,
        frameon=False,
        handlelength=2.15,
        borderaxespad=0.2,
        labelspacing=0.24,
    )

    ax_bottom.axhline(0.0, color="0.25", lw=1.05, zorder=2)
    ax_bottom.plot(x_values, no_pt2_errors, color="#d95f02", lw=2.0, zorder=4)
    ax_bottom.plot(x_values, pt2_errors, color="#1b9e77", lw=2.0, zorder=5)
    ax_bottom.scatter(
        TRAIN_R_VALUES,
        [0.0 for _ in TRAIN_R_VALUES],
        marker="D",
        s=27,
        facecolor="#b2182b",
        edgecolor="white",
        linewidth=0.8,
        zorder=7,
    )
    ax_bottom.set_xlabel(r"$R$ ($\AA$)", fontsize=11.8, fontweight="bold")
    ax_bottom.set_ylabel(r"$\Delta E_1$ (eV)", fontsize=10.8, fontweight="bold")
    ax_bottom.set_xticks([0.8, 1.2, 1.6, 2.0, 2.4, 2.8])

    x_span = max(x_values) - min(x_values)
    ax_top.set_xlim(min(x_values) - 0.03 * x_span, max(x_values) + 0.03 * x_span)
    y_values = reference_values + no_pt2_values + pt2_values
    y_span = max(y_values) - min(y_values)
    ax_top.set_ylim(min(y_values) - 0.05 * y_span, max(y_values) + 0.10 * y_span)
    all_errors = no_pt2_errors + pt2_errors
    err_padding = 0.12 * (max(all_errors) - min(all_errors))
    ax_bottom.set_ylim(min(all_errors) - err_padding, max(all_errors) + err_padding)

    for axis in (ax_top, ax_bottom):
        _setup_axis(axis)
        axis.grid(False)

    fig.align_ylabels([ax_top, ax_bottom])
    fig.subplots_adjust(left=0.195, right=0.985, top=0.985, bottom=0.14)

    visualization_csv = RUN_DIR / "n2_pt2_vs_nopt2_s1_total_curve_visualization_data.csv"
    _write_visualization_csv(no_pt2, pt2_strict, visualization_csv)

    outputs = {}
    for suffix in [".png", ".pdf", ".svg"]:
        path = RUN_DIR / f"{OUT_STEM}{suffix}"
        if suffix == ".png":
            fig.savefig(path, dpi=420)
        else:
            fig.savefig(path)
        outputs[suffix.lstrip(".")] = str(path)
    plt.close(fig)

    metrics = {
        "system": "N2",
        "state": "S1",
        "quantity": "S1 total energy E1(R)",
        "basis": "def2-tzvp",
        "xc": "b3lyp",
        "solver": "TDA",
        "reference": "Hammami2026 large-CAS A1Pi_g",
        "n_points": len(no_pt2),
        "train_r_values_angstrom": TRAIN_R_VALUES,
        "no_pt2": no_pt2_summary,
        "pt2_strict": pt2_summary,
        "source_csvs": {key: str(value) for key, value in CSV_PATHS.items()},
        "visualization_csv": str(visualization_csv),
        "outputs": outputs,
    }
    metrics_path = RUN_DIR / f"{OUT_STEM}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
