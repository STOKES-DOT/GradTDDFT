from __future__ import annotations

import csv
import math
import os
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", str(Path("benchmark") / ".mplconfig"))

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MOLECULE_ORDER = ["water", "co", "n2", "ethylene", "formaldehyde", "benzene"]
MOLECULE_LABELS = {
    "water": r"H$_2$O",
    "co": "CO",
    "n2": r"N$_2$",
    "ethylene": r"C$_2$H$_4$",
    "formaldehyde": r"CH$_2$O",
    "benzene": r"C$_6$H$_6$",
}
COLORS = {
    "water": "#2C7FB8",
    "co": "#41AB5D",
    "n2": "#F16913",
    "ethylene": "#756BB1",
    "formaldehyde": "#D95F0E",
    "benzene": "#636363",
}
MARKERS = {"tda": "o", "tddft": "s"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def as_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        x = float(value)
    except ValueError:
        return None
    if not math.isfinite(x):
        return None
    return x


def panel_parity(ax, viz: list[dict[str, str]]) -> None:
    rows = [r for r in viz if r["support_status"] == "graddft_supported"]
    for mol in MOLECULE_ORDER:
        for solver in ("tda", "tddft"):
            xs = []
            ys = []
            for row in rows:
                if row["molecule"] != mol or row["response_solver"] != solver:
                    continue
                x = as_float(row["pyscf_excitation_ev"])
                y = as_float(row["graddft_excitation_ev"])
                if x is not None and y is not None:
                    xs.append(x)
                    ys.append(y)
            if xs:
                ax.scatter(
                    xs,
                    ys,
                    s=38,
                    marker=MARKERS[solver],
                    color=COLORS[mol],
                    edgecolor="white",
                    linewidth=0.5,
                    alpha=0.9,
                )

    values = [
        as_float(row["pyscf_excitation_ev"])
        for row in rows
        if as_float(row["pyscf_excitation_ev"]) is not None
    ]
    lo = min(values) - 0.2
    hi = max(values) + 0.2
    ax.plot([lo, hi], [lo, hi], color="#20242A", lw=1.0, linestyle="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("PySCF excitation energy / eV")
    ax.set_ylabel("GradTDDFT excitation energy / eV")
    ax.grid(color="#E6EAF0", lw=0.7)
    ax.set_title("Excitation-energy parity", loc="left", fontweight="bold", fontsize=12)

    molecule_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[m], label=MOLECULE_LABELS[m], markersize=6)
        for m in MOLECULE_ORDER
    ]
    solver_handles = [
        Line2D([0], [0], marker="o", color="#20242A", linestyle="None", label="TDA", markersize=5),
        Line2D([0], [0], marker="s", color="#20242A", linestyle="None", label="TDDFT", markersize=5),
    ]
    leg1 = ax.legend(handles=molecule_handles, frameon=False, fontsize=7, ncol=2, loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=solver_handles, frameon=False, fontsize=7, loc="lower right")


def main() -> None:
    viz = read_csv(ROOT / "visualization_data.csv")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    panel_parity(ax, viz)
    fig.suptitle(
        "AD-generated response kernel reproduces PySCF TDDFT excitations",
        fontsize=12,
        fontweight="bold",
    )

    png = FIG_DIR / "excitation_energy_parity_draft.png"
    pdf = FIG_DIR / "excitation_energy_parity_draft.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
