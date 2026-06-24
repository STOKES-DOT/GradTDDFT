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


def gaussian_spectrum(
    energies: list[float],
    strengths: list[float],
    grid: list[float],
    sigma: float,
) -> list[float]:
    norm = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    spectrum: list[float] = []
    for x in grid:
        y = 0.0
        for energy, strength in zip(energies, strengths):
            y += strength * norm * math.exp(-0.5 * ((x - energy) / sigma) ** 2)
        spectrum.append(y)
    return spectrum


def benzene_rows(viz: list[dict[str, str]], solver: str) -> list[dict[str, str]]:
    return [
        row
        for row in viz
        if row["molecule"] == "benzene"
        and row["spin_channel"] == "singlet"
        and row["response_solver"] == solver
        and row["support_status"] == "graddft_supported"
    ]


def extract_transition_data(rows: list[dict[str, str]], prefix: str) -> tuple[list[float], list[float]]:
    energies: list[float] = []
    strengths: list[float] = []
    for row in rows:
        energy = as_float(row[f"{prefix}_excitation_ev"])
        strength = as_float(row[f"{prefix}_oscillator_strength"])
        if energy is None or strength is None:
            continue
        energies.append(energy)
        strengths.append(max(strength, 0.0))
    return energies, strengths


def draw_panel(ax, rows: list[dict[str, str]], solver_label: str, sigma: float) -> None:
    pyscf_e, pyscf_f = extract_transition_data(rows, "pyscf")
    gdft_e, gdft_f = extract_transition_data(rows, "graddft")
    e_min = min(pyscf_e + gdft_e) - 0.55
    e_max = max(pyscf_e + gdft_e) + 0.55
    n = 700
    grid = [e_min + (e_max - e_min) * i / (n - 1) for i in range(n)]
    pyscf_y = gaussian_spectrum(pyscf_e, pyscf_f, grid, sigma)
    gdft_y = gaussian_spectrum(gdft_e, gdft_f, grid, sigma)
    y_max = max(max(pyscf_y), max(gdft_y), max(pyscf_f + gdft_f) / (sigma * math.sqrt(2.0 * math.pi)))
    stick_scale = 0.72 * y_max / max(pyscf_f + gdft_f) if max(pyscf_f + gdft_f) > 0 else 1.0

    ax.plot(grid, pyscf_y, color="#20242A", lw=1.8, label="PySCF")
    ax.plot(grid, gdft_y, color="#D95F0E", lw=1.6, linestyle="--", label="GradTDDFT")
    for energy, strength in zip(pyscf_e, pyscf_f):
        ax.vlines(energy, 0.0, strength * stick_scale, color="#20242A", lw=1.0, alpha=0.58)
    for energy, strength in zip(gdft_e, gdft_f):
        ax.vlines(energy, 0.0, strength * stick_scale, color="#D95F0E", lw=0.8, alpha=0.50, linestyle="--")

    ax.set_title(solver_label, loc="left", fontweight="bold", fontsize=11)
    ax.set_xlabel("Excitation energy / eV")
    ax.set_ylabel("Broadened oscillator strength")
    ax.set_xlim(e_min, e_max)
    ax.set_ylim(0.0, y_max * 1.10 if y_max > 0 else 1.0)
    ax.grid(color="#E6EAF0", lw=0.7)
    ax.text(
        0.98,
        0.92,
        rf"$\sigma={sigma:.2f}$ eV",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#616A76",
    )


def main() -> None:
    viz = read_csv(ROOT / "visualization_data.csv")
    sigma = 0.12
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

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
    draw_panel(axes[0], benzene_rows(viz, "tda"), "Benzene TDA", sigma)
    draw_panel(axes[1], benzene_rows(viz, "tddft"), "Benzene full TDDFT", sigma)
    fig.suptitle("Benzene absorption spectrum from oscillator strengths", fontsize=12, fontweight="bold")
    handles = [
        Line2D([0], [0], color="#20242A", lw=1.8, label="PySCF"),
        Line2D([0], [0], color="#D95F0E", lw=1.6, linestyle="--", label="GradTDDFT"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=9)

    png = FIG_DIR / "benzene_absorption_spectrum_draft.png"
    pdf = FIG_DIR / "benzene_absorption_spectrum_draft.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
