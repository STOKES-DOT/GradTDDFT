from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def _load_curve_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"Empty curve csv: {path}")

    def _col(name: str) -> np.ndarray:
        return np.asarray([float(row[name]) for row in rows], dtype=float)

    r = _col("R_Angstrom")

    ref_e0_key = next((k for k in rows[0].keys() if k.startswith("E0_") and "NeuralXC" not in k), None)
    ref_e1_key = next((k for k in rows[0].keys() if k.startswith("E1_") and "NeuralXC" not in k), None)
    if ref_e0_key is None or ref_e1_key is None:
        raise KeyError("Failed to locate reference E0/E1 columns in csv header.")

    ref_e0 = _col(ref_e0_key)
    neu_e0 = _col("E0_NeuralXC_Hartree")
    ref_e1 = _col(ref_e1_key)
    neu_e1 = _col("E1_NeuralXC_Hartree")
    return r, ref_e0, neu_e0, ref_e1, neu_e1


def _plot(
    out_png: Path,
    *,
    r: np.ndarray,
    ref_e0: np.ndarray,
    neu_e0: np.ndarray,
    ref_e1: np.ndarray,
    neu_e1: np.ndarray,
    zoom_half_width: float,
) -> tuple[float, float]:
    eq_idx = int(np.argmin(ref_e0))
    r_eq = float(r[eq_idx])
    e_eq = float(ref_e0[eq_idx])

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.plot(r, ref_e0, color="#1f77b4", lw=2.3, label="FCI Ground")
    ax.plot(r, neu_e0, color="#1f77b4", lw=2.0, ls="--", label="Neural_xc Ground")
    ax.plot(r, ref_e1, color="#d62728", lw=2.3, label="FCI First Excited")
    ax.plot(r, neu_e1, color="#d62728", lw=2.0, ls="--", label="Neural_xc First Excited")
    ax.axvline(r_eq, color="0.35", lw=1.1, ls=":", alpha=0.9)
    ax.scatter([r_eq], [e_eq], color="black", s=26, zorder=6)
    ax.text(
        r_eq + 0.03,
        e_eq + 0.05,
        f"R_eq = {r_eq:.3f} Å",
        fontsize=10,
        color="0.2",
    )
    ax.set_xlabel("H-H Distance (Angstrom)")
    ax.set_ylabel("Total Energy (Hartree)")
    ax.set_title("H2 Dissociation Curves: Ground + First Excited")
    ax.grid(alpha=0.25)
    ax.legend(
        frameon=False,
        ncol=2,
        fontsize=9.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
    )

    x0 = max(float(r.min()), r_eq - zoom_half_width)
    x1 = min(float(r.max()), r_eq + zoom_half_width)
    mask = (r >= x0) & (r <= x1)
    y_local = np.concatenate([ref_e0[mask], neu_e0[mask], ref_e1[mask], neu_e1[mask]])
    y0 = float(np.min(y_local))
    y1 = float(np.max(y_local))
    y_pad = max(0.03, 0.08 * (y1 - y0))

    axins = inset_axes(ax, width="45%", height="42%", loc="upper right", borderpad=1.2)
    axins.plot(r, ref_e0, color="#1f77b4", lw=2.0)
    axins.plot(r, neu_e0, color="#1f77b4", lw=1.8, ls="--")
    axins.plot(r, ref_e1, color="#d62728", lw=2.0)
    axins.plot(r, neu_e1, color="#d62728", lw=1.8, ls="--")
    axins.axvline(r_eq, color="0.35", lw=1.0, ls=":", alpha=0.9)
    axins.set_xlim(x0, x1)
    axins.set_ylim(y0 - y_pad, y1 + y_pad)
    axins.set_title("Zoom Near Equilibrium", fontsize=9.5)
    axins.grid(alpha=0.25)
    axins.tick_params(labelsize=8.5)
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.45", lw=0.9)

    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.24, top=0.90)
    fig.savefig(out_png, dpi=240)
    plt.close(fig)
    return r_eq, e_eq


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replot H2 ground/excited dissociation curves in one figure with equilibrium zoom-in.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="outputs/h2_fci_excited_infer_from_ground_ckpt_r005_5A_p201_retry/h2_fci_vs_neural_curve.csv",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default="outputs/h2_fci_excited_infer_from_ground_ckpt_r005_5A_p201_retry/h2_fci_vs_neural_curve_combined_zoom.png",
    )
    parser.add_argument(
        "--zoom-half-width",
        type=float,
        default=0.25,
        help="Half width (Angstrom) of the equilibrium neighborhood shown in the inset.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    r, ref_e0, neu_e0, ref_e1, neu_e1 = _load_curve_csv(csv_path)
    r_eq, e_eq = _plot(
        out_png,
        r=r,
        ref_e0=ref_e0,
        neu_e0=neu_e0,
        ref_e1=ref_e1,
        neu_e1=neu_e1,
        zoom_half_width=float(args.zoom_half_width),
    )
    print(f"wrote={out_png}")
    print(f"equilibrium_R_A={r_eq:.6f}")
    print(f"equilibrium_E0_H={e_eq:.12f}")


if __name__ == "__main__":
    main()
