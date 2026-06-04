from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

HARTREE_TO_EV = 27.211386245988


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_loss(path: Path, history_csv: Path, *, title: str, loss_column: str = "loss") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_csv(history_csv)
    steps = np.asarray([_as_float(row, "step") for row in rows], dtype=float)
    loss = np.asarray([_as_float(row, loss_column) for row in rows], dtype=float)
    finite = np.isfinite(steps) & np.isfinite(loss)
    if not np.any(finite):
        raise ValueError(f"No finite loss data in {history_csv}")

    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    ax.plot(steps[finite], np.maximum(loss[finite], 1e-18), lw=1.9, color="#1f5f8b")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total loss")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _curve_arrays(curve_csv: Path, *, reference_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _read_csv(curve_csv)
    r = np.asarray([_as_float(row, "r_angstrom") for row in rows], dtype=float)
    ref = np.asarray([_as_float(row, reference_key) for row in rows], dtype=float)
    pred = np.asarray([_as_float(row, "predicted_energy_h") for row in rows], dtype=float)
    order = np.argsort(r)
    return r[order], ref[order], pred[order]


def _train_points_from_summary(summary: dict[str, Any]) -> np.ndarray:
    values = summary.get("train_r_values_angstrom", [])
    return np.asarray([float(value) for value in values], dtype=float)


def _train_points_from_csv(path: Path, *, reference_key: str) -> tuple[np.ndarray, np.ndarray]:
    rows = _read_csv(path)
    r = np.asarray([_as_float(row, "r_angstrom") for row in rows], dtype=float)
    ref = np.asarray([_as_float(row, reference_key) for row in rows], dtype=float)
    finite = np.isfinite(r) & np.isfinite(ref)
    return r[finite], ref[finite]


def _plot_curve(
    path: Path,
    curve_csv: Path,
    *,
    reference_key: str,
    train_r: np.ndarray | None = None,
    train_reference: np.ndarray | None = None,
    title: str,
    x_label: str,
    reference_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r, ref, pred = _curve_arrays(curve_csv, reference_key=reference_key)
    finite_curve = np.isfinite(r) & np.isfinite(ref) & np.isfinite(pred)
    if not np.any(finite_curve):
        raise ValueError(f"No finite curve data in {curve_csv}")
    r = r[finite_curve]
    ref = ref[finite_curve]
    pred = pred[finite_curve]

    if train_r is None or train_r.size == 0:
        train_r = r
    if train_reference is None or train_reference.size == 0:
        train_reference = np.interp(train_r, r, ref)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    ax = axes[0]
    ax.plot(r, ref, lw=2.0, color="#20242a", label=reference_label)
    ax.plot(r, pred, lw=2.0, color="#1f77b4", label="Neural XC")
    ax.scatter(
        train_r,
        train_reference,
        s=38,
        color="#111111",
        edgecolors="white",
        linewidths=0.7,
        zorder=5,
        label="Training points",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Energy (Hartree)")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8.5)

    err = np.abs(pred - ref) * HARTREE_TO_EV
    axes[1].plot(r, np.maximum(err, 1e-16), lw=1.9, color="#b33b2e")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("Abs. error (eV)")
    axes[1].set_title("Energy error", loc="left", fontsize=11, fontweight="bold")
    axes[1].grid(alpha=0.25)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_h2_neutral(outdir: Path) -> dict[str, str]:
    summary = _load_json(outdir / "summary.json")
    train_r = _train_points_from_summary(summary)
    loss_png = outdir / "loss_total_only.png"
    curve_png = outdir / "dissociation_curve_uniform.png"
    _plot_loss(
        loss_png,
        outdir / "training_curve.csv",
        title="H2 Neutral Ground-State Training",
    )
    _plot_curve(
        curve_png,
        outdir / "h2_fci_ground_vs_neural_dense_curve.csv",
        reference_key="fci_energy_h",
        train_r=train_r,
        title="H2 Neutral Ground-State Dissociation",
        x_label="H-H distance (Angstrom)",
        reference_label="FCI",
    )
    return {
        "loss": str(loss_png),
        "curve": str(curve_png),
        "history_csv": str(outdir / "training_curve.csv"),
        "curve_csv": str(outdir / "h2_fci_ground_vs_neural_dense_curve.csv"),
    }


def _plot_h2plus(outdir: Path) -> dict[str, str]:
    train_r, train_ref = _train_points_from_csv(
        outdir / "h2plus_reference_points.csv",
        reference_key="exact_energy_h",
    )
    loss_png = outdir / "loss_total_only.png"
    curve_png = outdir / "dissociation_curve_uniform.png"
    _plot_loss(
        loss_png,
        outdir / "training_history.csv",
        title="H2+ Ion Ground-State Training",
    )
    _plot_curve(
        curve_png,
        outdir / "h2plus_ground_dense_curve.csv",
        reference_key="exact_energy_h",
        train_r=train_r,
        train_reference=train_ref,
        title="H2+ Ion Ground-State Dissociation",
        x_label="H-H distance (Angstrom)",
        reference_label="Exact",
    )
    return {
        "loss": str(loss_png),
        "curve": str(curve_png),
        "history_csv": str(outdir / "training_history.csv"),
        "curve_csv": str(outdir / "h2plus_ground_dense_curve.csv"),
        "train_points_csv": str(outdir / "h2plus_reference_points.csv"),
    }


def _plot_n2(outdir: Path) -> dict[str, str]:
    train_r, train_ref = _train_points_from_csv(
        outdir / "n2_ccsdt_reference_points.csv",
        reference_key="reference_energy_h",
    )
    loss_png = outdir / "loss_total_only.png"
    curve_png = outdir / "dissociation_curve_uniform.png"
    _plot_loss(
        loss_png,
        outdir / "training_history.csv",
        title="N2 Neutral Ground-State Training",
    )
    _plot_curve(
        curve_png,
        outdir / "n2_ccsdt_ground_predictions.csv",
        reference_key="reference_energy_h",
        train_r=train_r,
        train_reference=train_ref,
        title="N2 Neutral Ground-State Dissociation",
        x_label="N-N distance (Angstrom)",
        reference_label="CCSD(T)",
    )
    return {
        "loss": str(loss_png),
        "curve": str(curve_png),
        "history_csv": str(outdir / "training_history.csv"),
        "curve_csv": str(outdir / "n2_ccsdt_ground_predictions.csv"),
        "train_points_csv": str(outdir / "n2_ccsdt_reference_points.csv"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Uniform plots for the diatomic training suite.")
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--h2-dir", default=None)
    parser.add_argument("--h2plus-dir", default=None)
    parser.add_argument("--n2-dir", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    suite_root = Path(args.suite_root)
    outputs: dict[str, Any] = {}
    task_specs = {
        "h2_neutral_ground": (Path(args.h2_dir) if args.h2_dir else suite_root / "h2_neutral_ground", _plot_h2_neutral),
        "h2plus_ion_ground": (Path(args.h2plus_dir) if args.h2plus_dir else suite_root / "h2plus_ion_ground", _plot_h2plus),
        "n2_neutral_ground": (Path(args.n2_dir) if args.n2_dir else suite_root / "n2_neutral_ground", _plot_n2),
    }
    for name, (outdir, plotter) in task_specs.items():
        outputs[name] = plotter(outdir)
    manifest_path = suite_root / "uniform_visualization_manifest.json"
    manifest_path.write_text(json.dumps(outputs, indent=2, sort_keys=True), encoding="utf-8")
    print(f"uniform_visualization_manifest={manifest_path}")
    return outputs


if __name__ == "__main__":
    main()
