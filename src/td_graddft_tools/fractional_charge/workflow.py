from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from td_graddft.spectra import HARTREE_TO_EV

from .analysis import (
    FractionalChargeAnalysisResult,
    FractionalChargeAnalysisConfig,
    FractionalChargeEnergyEvaluator,
    analyze_fractional_charge_linearity,
)


@dataclass(frozen=True)
class FractionalChargeOutputConfig:
    outdir: Path = Path("outputs")
    prefix: str = "fractional_charge"
    title: str = "Fractional Charge Linearity"
    energy_unit: Literal["ha", "ev"] = "ha"


@dataclass(frozen=True)
class FractionalChargeWorkflowResult:
    analysis: FractionalChargeAnalysisResult
    csv_path: Path
    png_path: Path
    summary_path: Path


def write_fractional_charge_csv(
    result: FractionalChargeAnalysisResult,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "charge_delta",
                "electron_count",
                "energy_ha",
                "piecewise_linear_energy_ha",
                "deviation_ha",
                "homo_occupation",
                "lumo_occupation",
            ]
        )
        for row in zip(
            np.asarray(result.charge_deltas, dtype=float),
            np.asarray(result.electron_counts, dtype=float),
            np.asarray(result.energies_ha, dtype=float),
            np.asarray(result.piecewise_linear_energies_ha, dtype=float),
            np.asarray(result.deviation_ha, dtype=float),
            np.asarray(result.homo_occupations, dtype=float),
            np.asarray(result.lumo_occupations, dtype=float),
            strict=True,
        ):
            writer.writerow(row)


def plot_fractional_charge_analysis(
    result: FractionalChargeAnalysisResult,
    path: Path,
    *,
    title: str,
    energy_unit: Literal["ha", "ev"] = "ha",
) -> None:
    scale = HARTREE_TO_EV if energy_unit == "ev" else 1.0
    unit_label = "eV" if energy_unit == "ev" else "Ha"
    x = np.asarray(result.charge_deltas, dtype=float)
    energies = np.asarray(result.energies_ha, dtype=float) * scale
    piecewise = np.asarray(result.piecewise_linear_energies_ha, dtype=float) * scale
    deviation = np.asarray(result.deviation_ha, dtype=float) * scale
    homo_occ = np.asarray(result.homo_occupations, dtype=float)
    lumo_occ = np.asarray(result.lumo_occupations, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    axes[0].plot(x, energies, lw=2.1, color="#1f77b4", label="Energy")
    axes[0].plot(x, piecewise, "--", lw=1.8, color="#555555", label="Piecewise linear")
    axes[0].set_xlabel("Fractional charge ΔN")
    axes[0].set_ylabel(f"Energy ({unit_label})")
    axes[0].set_title("Energy Curve")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(x, deviation, lw=2.0, color="#c0392b", label="Deviation")
    axes[1].plot(x, homo_occ, lw=1.6, color="#2a9d8f", label="HOMO occ.")
    axes[1].plot(x, lumo_occ, lw=1.6, color="#e9c46a", label="LUMO occ.")
    axes[1].axhline(0.0, color="#555555", lw=1.0, ls="--")
    axes[1].set_xlabel("Fractional charge ΔN")
    axes[1].set_ylabel(f"Deviation ({unit_label}) / occupation")
    axes[1].set_title("Linearity Deviation")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_fractional_charge_summary(
    result: FractionalChargeAnalysisResult,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"max_abs_deviation_ha={result.max_abs_deviation_ha:.10e}\n")
        handle.write(f"mean_abs_deviation_ha={result.mean_abs_deviation_ha:.10e}\n")
        handle.write(f"rms_deviation_ha={result.rms_deviation_ha:.10e}\n")
        handle.write(f"left_endpoint_slope_ha={result.left_endpoint_slope_ha:.10e}\n")
        handle.write(f"right_endpoint_slope_ha={result.right_endpoint_slope_ha:.10e}\n")


def run_fractional_charge_workflow(
    molecule: object,
    energy_evaluator: FractionalChargeEnergyEvaluator,
    *,
    analysis_config: FractionalChargeAnalysisConfig | None = None,
    output_config: FractionalChargeOutputConfig | None = None,
) -> FractionalChargeWorkflowResult:
    analysis = analyze_fractional_charge_linearity(
        molecule,
        energy_evaluator,
        config=analysis_config,
    )
    output = FractionalChargeOutputConfig() if output_config is None else output_config
    output.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = output.outdir / f"{output.prefix}.csv"
    png_path = output.outdir / f"{output.prefix}.png"
    summary_path = output.outdir / f"{output.prefix}_summary.txt"
    write_fractional_charge_csv(analysis, csv_path)
    plot_fractional_charge_analysis(
        analysis,
        png_path,
        title=output.title,
        energy_unit=output.energy_unit,
    )
    write_fractional_charge_summary(analysis, summary_path)
    return FractionalChargeWorkflowResult(
        analysis=analysis,
        csv_path=csv_path,
        png_path=png_path,
        summary_path=summary_path,
    )
