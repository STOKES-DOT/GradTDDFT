from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/Users/jiaoyuan/Documents/GitHub/TD-GradDFT")
RUN_DIR = ROOT / "benchmark/n2_hammami_s1total_tda_chunked_eval_skiposc_20260623_094250"
CSV_PATH = RUN_DIR / "nopt2_dense_curve_35pt_merged.csv"
OUT_STEM = "n2_nopt2_s1_total_dissociation_curve"
EV_PER_HARTREE = 27.211386245988
TRAIN_R_VALUES = [0.8, 1.1, 1.6, 2.0, 2.2, 2.5, 3.0]


def _read_rows() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with CSV_PATH.open(newline="") as handle:
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


def _write_visualization_csv(rows: list[dict[str, float]], path: Path) -> None:
    fieldnames = [
        "dense_index",
        "r_angstrom",
        "hammami_s1_total_hartree",
        "neural_nopt2_s1_total_hartree",
        "signed_error_ev",
        "abs_error_ev",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dense_index": int(row["dense_index"]),
                    "r_angstrom": f"{row['r_angstrom']:.12g}",
                    "hammami_s1_total_hartree": f"{row['reference_s1_total_h']:.12g}",
                    "neural_nopt2_s1_total_hartree": f"{row['predicted_s1_total_h']:.12g}",
                    "signed_error_ev": f"{row['signed_error_ev']:.12g}",
                    "abs_error_ev": f"{row['abs_error_ev']:.12g}",
                }
            )


def main() -> None:
    rows = _read_rows()
    x_values = [row["r_angstrom"] for row in rows]
    reference_values = [row["reference_s1_total_h"] for row in rows]
    predicted_values = [row["predicted_s1_total_h"] for row in rows]
    signed_errors = [row["signed_error_ev"] for row in rows]
    abs_errors = [row["abs_error_ev"] for row in rows]
    max_index = max(range(len(rows)), key=lambda index: abs_errors[index])
    mae_ev = sum(abs_errors) / len(abs_errors)
    max_abs_ev = abs_errors[max_index]
    train_reference = [
        _interp(x_values, reference_values, train_r) for train_r in TRAIN_R_VALUES
    ]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(4.05, 3.85),
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
        predicted_values,
        color="#d95f02",
        lw=2.35,
        label=f"Neural no-PT2, MAE={mae_ev:.2f} eV",
        zorder=5,
    )
    ax_top.scatter(
        TRAIN_R_VALUES,
        train_reference,
        marker="D",
        s=34,
        facecolor="#b2182b",
        edgecolor="white",
        linewidth=0.8,
        label="Training R",
        zorder=7,
    )
    ax_top.text(
        0.08,
        0.76,
        r"N$_2$ S$_1$ total",
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=12.4,
        fontweight="bold",
    )
    ax_top.set_ylabel(r"$E_1(R)$ (Ha)", fontsize=11.2, fontweight="bold")
    ax_top.legend(
        loc="upper right",
        fontsize=6.4,
        frameon=False,
        handlelength=2.3,
        borderaxespad=0.2,
        labelspacing=0.28,
    )

    ax_bottom.axhline(0.0, color="0.25", lw=1.05, zorder=2)
    ax_bottom.plot(x_values, signed_errors, color="#d95f02", lw=2.0, zorder=4)
    ax_bottom.scatter(
        TRAIN_R_VALUES,
        [0.0 for _ in TRAIN_R_VALUES],
        marker="D",
        s=30,
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
    y_values = reference_values + predicted_values
    y_span = max(y_values) - min(y_values)
    ax_top.set_ylim(min(y_values) - 0.05 * y_span, max(y_values) + 0.10 * y_span)
    err_padding = 0.12 * (max(signed_errors) - min(signed_errors))
    ax_bottom.set_ylim(min(signed_errors) - err_padding, max(signed_errors) + err_padding)

    for axis in (ax_top, ax_bottom):
        _setup_axis(axis)
        axis.grid(False)

    fig.align_ylabels([ax_top, ax_bottom])
    fig.subplots_adjust(left=0.205, right=0.985, top=0.985, bottom=0.145)

    visualization_csv = RUN_DIR / "n2_nopt2_s1_total_curve_visualization_data.csv"
    _write_visualization_csv(rows, visualization_csv)

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
        "model": "hfx no-PT2",
        "basis": "def2-tzvp",
        "xc": "b3lyp",
        "solver": "TDA",
        "reference": "Hammami2026 large-CAS A1Pi_g",
        "n_points": len(rows),
        "train_r_values_angstrom": TRAIN_R_VALUES,
        "s1_total_mae_ev": mae_ev,
        "s1_total_max_abs_err_ev": max_abs_ev,
        "s1_total_max_abs_err_r_angstrom": rows[max_index]["r_angstrom"],
        "source_csv": str(CSV_PATH),
        "visualization_csv": str(visualization_csv),
        "outputs": outputs,
    }
    metrics_path = RUN_DIR / f"{OUT_STEM}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
