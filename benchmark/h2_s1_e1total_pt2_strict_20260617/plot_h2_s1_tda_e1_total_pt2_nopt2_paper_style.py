from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory


ROOT = Path("/Users/jiaoyuan/Documents/GitHub/TD-GradDFT")
RUN_DIR = ROOT / "benchmark/h2_s1_e1total_pt2_strict_20260617"
NO_PT2_RUN_DIR = ROOT / "benchmark/h2_nopt2_dense100_visualization_20260624"
PAPER_FIG_DIRS = [
    ROOT / "paper/tdgraddft-paper/figures",
    ROOT / "paper_review/tdgraddft-paper/figures",
]

EV_PER_H = 27.211386245988
KCAL_PER_MOL_TO_MEV = 43.3641153087705
DEFAULT_TRAIN_R_VALUES = [0.4, 1.8, 3.2, 4.6, 6.0]
COMBINED_OUT_STEM = "h2_s1_tda_e1_total_pt2_nopt2_dissociation_paper_style"

CASES = [
    {
        "key": "pt2",
        "artifact_dir": RUN_DIR,
        "csv": RUN_DIR / "h2_s1_tda_dense_curve.csv",
        "summary": RUN_DIR / "summary.json",
        "out_stem": "h2_pt2_s1_tda_e1_total_dissociation_paper_style",
        "model_label": "Neural TDA+PT2",
        "panel_label": r"(a) H$_2$ $B\,{}^1\Sigma_u^+$, PT2",
        "color": "#d95f02",
        "train_r_values": DEFAULT_TRAIN_R_VALUES,
    },
    {
        "key": "no_pt2",
        "artifact_dir": NO_PT2_RUN_DIR,
        "csv": NO_PT2_RUN_DIR / "h2_s1_tda_dense_curve.csv",
        "summary": NO_PT2_RUN_DIR / "summary.json",
        "out_stem": "h2_nopt2_s1_tda_e1_total_dissociation_paper_style",
        "model_label": "Neural TDA/no-PT2",
        "panel_label": r"(b) H$_2$ $B\,{}^1\Sigma_u^+$, no-PT2",
        "color": "#2b6cb0",
    },
]


def read_curve(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            ref = float(row["fci_s1_total_energy_h"])
            pred = float(row["predicted_s1_total_energy_h"])
            rows.append(
                {
                    "dense_index": float(row.get("dense_index", index)),
                    "r_angstrom": float(row["r_angstrom"]),
                    "reference_energy_h": ref,
                    "predicted_energy_h": pred,
                    "signed_error_mev": (pred - ref) * EV_PER_H * 1000.0,
                    "abs_error_ev": float(row["s1_total_abs_err_ev"]),
                    "reference_gap_h": float(row["fci_s1_h"]),
                    "predicted_gap_h": float(row["predicted_s1_h"]),
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


def load_summary(case: dict[str, object]) -> dict[str, object]:
    summary_path = case.get("summary")
    if not summary_path:
        return {}
    path = Path(summary_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def case_train_r_values(
    case: dict[str, object],
    summary: dict[str, object],
) -> list[float]:
    values = case.get("train_r_values", summary.get("train_r_values_angstrom"))
    if values is None:
        values = DEFAULT_TRAIN_R_VALUES
    return [float(value) for value in values]


def compute_metrics(rows: list[dict[str, float]], case: dict[str, object]) -> dict[str, object]:
    summary = load_summary(case)
    train_r_values = case_train_r_values(case, summary)
    errors = [row["signed_error_mev"] for row in rows]
    abs_errors = [abs(value) for value in errors]
    max_index = max(range(len(rows)), key=lambda index: abs_errors[index])
    mae_ev = sum(abs_errors) / len(abs_errors) / 1000.0
    rmse_ev = math.sqrt(sum(value * value for value in errors) / len(errors)) / 1000.0
    max_abs_ev = abs_errors[max_index] / 1000.0
    metrics: dict[str, object] = {
        "system": "H2",
        "state": "S1",
        "spectroscopic_state": "B 1Sigma_u+",
        "quantity": "S1 total energy E1(R)",
        "basis": str(summary.get("basis", "def2-svp")),
        "xc": str(summary.get("xc", "b3lyp")),
        "solver": "TDA",
        "reference": "FCI",
        "model_label": case["model_label"],
        "n_points": len(rows),
        "train_r_values_angstrom": train_r_values,
        "dense_mae_ev_from_csv": mae_ev,
        "dense_rmse_ev_from_csv": rmse_ev,
        "dense_max_abs_error_ev_from_csv": max_abs_ev,
        "dense_max_abs_error_r_angstrom": rows[max_index]["r_angstrom"],
        "delta_e_unit": "meV",
        "chemical_accuracy_band_mev": KCAL_PER_MOL_TO_MEV,
        "source_csv": str(case["csv"]),
    }
    metrics.update(
        {
            "objective": summary.get("objective", "s1_only_tda"),
            "response_hf_mode": summary.get("response_hf_mode", "approx"),
            "response_pt2_mode": summary.get("response_pt2_mode") or "not_used",
            "pt2_channel_mode": summary.get("pt2_channel_mode") or "disabled",
        }
    )
    if "s1_total_mae_ev" in summary:
        metrics["summary_s1_total_mae_ev"] = float(summary["s1_total_mae_ev"])
    if "s1_total_max_ev" in summary:
        metrics["summary_s1_total_max_ev"] = float(summary["s1_total_max_ev"])
    return metrics


def write_visualization_csv(
    rows: list[dict[str, float]],
    case: dict[str, object],
    metrics: dict[str, object],
) -> None:
    out = Path(case["artifact_dir"]) / f"{case['out_stem']}_visualization_data.csv"
    train_set = {round(value, 8) for value in metrics["train_r_values_angstrom"]}
    with out.open("w", newline="") as handle:
        fieldnames = [
            "dense_index",
            "r_angstrom",
            "reference_energy_h",
            "predicted_energy_h",
            "signed_error_mev",
            "abs_error_ev",
            "reference_gap_ev",
            "predicted_gap_ev",
            "is_training_point",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dense_index": int(row["dense_index"]),
                    "r_angstrom": f"{row['r_angstrom']:.12g}",
                    "reference_energy_h": f"{row['reference_energy_h']:.12g}",
                    "predicted_energy_h": f"{row['predicted_energy_h']:.12g}",
                    "signed_error_mev": f"{row['signed_error_mev']:.12g}",
                    "abs_error_ev": f"{row['abs_error_ev']:.12g}",
                    "reference_gap_ev": f"{row['reference_gap_h'] * EV_PER_H:.12g}",
                    "predicted_gap_ev": f"{row['predicted_gap_h'] * EV_PER_H:.12g}",
                    "is_training_point": round(row["r_angstrom"], 8) in train_set,
                }
            )
    metrics["visualization_csv"] = str(out)


def draw_case(
    ax_top: plt.Axes,
    ax_bottom: plt.Axes,
    rows: list[dict[str, float]],
    case: dict[str, object],
    metrics: dict[str, object],
    *,
    panel_label: str | None = None,
    show_ylabel: bool = True,
) -> None:
    x_values = [row["r_angstrom"] for row in rows]
    reference_values = [row["reference_energy_h"] for row in rows]
    predicted_values = [row["predicted_energy_h"] for row in rows]
    errors = [row["signed_error_mev"] for row in rows]
    abs_errors = [abs(value) for value in errors]
    train_r_values = [float(value) for value in metrics["train_r_values_angstrom"]]
    train_reference = [interp(x_values, reference_values, value) for value in train_r_values]
    color = str(case["color"])
    mae_mev = float(metrics["dense_mae_ev_from_csv"]) * 1000.0

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
        color=color,
        lw=2.45,
        label=f"{case['model_label']} ({mae_mev:.1f} meV)",
        zorder=5,
    )
    ax_top.scatter(
        train_r_values,
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
        0.07,
        0.80,
        panel_label if panel_label is not None else r"H$_2$ $B\,{}^1\Sigma_u^+$",
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=11.6 if panel_label is not None else 12.8,
        fontweight="bold",
    )
    if show_ylabel:
        ax_top.set_ylabel(r"$E_1(R)$ (Ha)", fontsize=12.5, fontweight="bold", labelpad=2.0)
    ax_top.legend(
        loc="upper right",
        fontsize=6.7,
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
    ax_bottom.plot(x_values, errors, color=color, lw=2.05, zorder=4)
    ax_bottom.scatter(
        train_r_values,
        [0.0 for _ in train_r_values],
        marker="D",
        s=34,
        facecolor="#b2182b",
        edgecolor="white",
        linewidth=0.9,
        zorder=7,
    )
    wide_error_scale = max(abs_errors) > 5.0 * KCAL_PER_MOL_TO_MEV
    if wide_error_scale:
        ax_bottom.text(
            0.055,
            0.88,
            r"gray band: $\pm 1$ kcal mol$^{-1}$",
            transform=ax_bottom.transAxes,
            ha="left",
            va="center",
            fontsize=7.8,
            fontweight="bold",
            color="0.35",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
            zorder=8,
        )
    else:
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
    if show_ylabel:
        ax_bottom.set_ylabel(r"$\Delta E_1$ (meV)", fontsize=12.0, fontweight="bold", labelpad=2.0)
    ax_bottom.set_xticks([1, 2, 3, 4, 5, 6])
    ax_bottom.grid(False)

    x_min, x_max = min(x_values), max(x_values)
    x_span = x_max - x_min
    ax_top.set_xlim(x_min - 0.03 * x_span, x_max + 0.03 * x_span)
    y_all = reference_values + predicted_values
    y_min, y_max = min(y_all), max(y_all)
    y_span = y_max - y_min
    ax_top.set_ylim(y_min - 0.04 * y_span, y_max + 0.09 * y_span)
    err_limit = max(max(abs_errors) * 1.18, KCAL_PER_MOL_TO_MEV * 1.55)
    ax_bottom.set_ylim(-err_limit, err_limit)
    for axis in (ax_top, ax_bottom):
        setup_axes(axis)


def save_figure(fig: plt.Figure, out_stem: str, artifact_dir: Path) -> dict[str, str]:
    output_dirs = [artifact_dir, *[directory for directory in PAPER_FIG_DIRS if directory.exists()]]
    outputs: dict[str, str] = {}
    for directory in output_dirs:
        base = directory / out_stem
        fig.savefig(base.with_suffix(".pdf"))
        fig.savefig(base.with_suffix(".png"), dpi=420)
        fig.savefig(base.with_suffix(".svg"))
        label = "run" if directory == artifact_dir else directory.parent.parent.name
        outputs[f"{label}_pdf"] = str(base.with_suffix(".pdf"))
        outputs[f"{label}_png"] = str(base.with_suffix(".png"))
        outputs[f"{label}_svg"] = str(base.with_suffix(".svg"))
    return outputs


def make_single(case: dict[str, object], rows: list[dict[str, float]], metrics: dict[str, object]) -> None:
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
    draw_case(ax_top, ax_bottom, rows, case, metrics)
    fig.align_ylabels([ax_top, ax_bottom])
    fig.subplots_adjust(left=0.30, right=0.985, top=0.985, bottom=0.155)
    metrics["outputs"] = save_figure(fig, str(case["out_stem"]), Path(case["artifact_dir"]))
    plt.close(fig)
    write_visualization_csv(rows, case, metrics)
    (Path(case["artifact_dir"]) / f"{case['out_stem']}_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )


def make_combined(case_data: list[tuple[dict[str, object], list[dict[str, float]], dict[str, object]]]) -> dict[str, object]:
    fig = plt.figure(figsize=(6.70, 3.80))
    grid = fig.add_gridspec(2, 2, height_ratios=[2.45, 1.0], hspace=0.08, wspace=0.26)
    axes = [
        (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])),
        (fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])),
    ]
    fig.patch.set_facecolor("white")
    for index, ((case, rows, metrics), (ax_top, ax_bottom)) in enumerate(
        zip(case_data, axes, strict=True)
    ):
        ax_top.set_facecolor("white")
        ax_bottom.set_facecolor("white")
        draw_case(
            ax_top,
            ax_bottom,
            rows,
            case,
            metrics,
            panel_label=str(case["panel_label"]),
            show_ylabel=index == 0,
        )
        ax_top.tick_params(axis="x", which="both", labelbottom=False)
        for label in ax_top.get_xticklabels():
            label.set_visible(False)
    fig.align_ylabels([axis for pair in axes for axis in pair])
    fig.subplots_adjust(left=0.145, right=0.992, top=0.982, bottom=0.155)
    outputs = save_figure(fig, COMBINED_OUT_STEM, RUN_DIR)
    plt.close(fig)
    combined_metrics = {
        "system": "H2",
        "state": "S1",
        "spectroscopic_state": "B 1Sigma_u+",
        "quantity": "S1 total energy E1(R)",
        "basis": "case-specific",
        "xc": "b3lyp",
        "solver": "TDA",
        "reference": "FCI",
        "n_points": len(case_data[0][1]),
        "train_r_values_angstrom": {
            str(case["key"]): metrics["train_r_values_angstrom"]
            for case, _rows, metrics in case_data
        },
        "cases": {
            str(case["key"]): {
                "model_label": metrics["model_label"],
                "basis": metrics["basis"],
                "xc": metrics["xc"],
                "train_r_values_angstrom": metrics["train_r_values_angstrom"],
                "dense_mae_ev_from_csv": metrics["dense_mae_ev_from_csv"],
                "dense_rmse_ev_from_csv": metrics["dense_rmse_ev_from_csv"],
                "dense_max_abs_error_ev_from_csv": metrics["dense_max_abs_error_ev_from_csv"],
                "dense_max_abs_error_r_angstrom": metrics["dense_max_abs_error_r_angstrom"],
                "source_csv": metrics["source_csv"],
                "visualization_csv": metrics["visualization_csv"],
            }
            for case, _rows, metrics in case_data
        },
        "outputs": outputs,
    }
    (RUN_DIR / f"{COMBINED_OUT_STEM}_metrics.json").write_text(
        json.dumps(combined_metrics, indent=2)
    )
    return combined_metrics


def main() -> None:
    case_data = []
    for case in CASES:
        rows = read_curve(Path(case["csv"]))
        metrics = compute_metrics(rows, case)
        make_single(case, rows, metrics)
        case_data.append((case, rows, metrics))
    combined_metrics = make_combined(case_data)
    print(json.dumps(combined_metrics, indent=2))


if __name__ == "__main__":
    main()
