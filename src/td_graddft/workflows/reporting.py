from __future__ import annotations

import csv
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from td_graddft.spectra import HARTREE_TO_EV

from .types import (
    MoleculeRun,
    NeuralExcitedStateRun,
    OutputConfig,
    OutputPaths,
    PipelineRun,
    SpectrumRun,
    TrainingRun,
)


def ensure_output_dirs(output: OutputConfig) -> None:
    output.outdir.mkdir(parents=True, exist_ok=True)
    (output.outdir / ".mplconfig").mkdir(parents=True, exist_ok=True)


def build_output_paths(output: OutputConfig) -> OutputPaths:
    spectrum_csv = output.outdir / f"{output.prefix}.csv"
    spectrum_png = output.outdir / f"{output.prefix}.png"
    training_stem = output.training_prefix or f"{output.prefix}_training_curve"
    training_csv = output.outdir / f"{training_stem}.csv"
    training_png = output.outdir / f"{training_stem}.png"
    return OutputPaths(
        spectrum_csv=spectrum_csv,
        spectrum_png=spectrum_png,
        training_csv=training_csv if output.write_training_curves else None,
        training_png=training_png if output.write_training_curves else None,
    )


def write_state_comparison_csv(
    path: Path,
    reference: MoleculeRun,
    neural: NeuralExcitedStateRun,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "state",
                "b3lyp_energy_eV",
                "b3lyp_oscillator_strength",
                "neural_energy_eV",
                "neural_oscillator_strength",
            ]
        )
        nrows = max(reference.energies_au.size, neural.energies_au.size)
        for i in range(nrows):
            writer.writerow(
                [
                    i + 1,
                    float(reference.energies_au[i] * HARTREE_TO_EV)
                    if i < reference.energies_au.size
                    else "",
                    float(reference.oscillator_strengths[i])
                    if i < reference.oscillator_strengths.size
                    else "",
                    float(neural.energies_au[i] * HARTREE_TO_EV)
                    if i < neural.energies_au.size
                    else "",
                    float(neural.oscillator_strengths[i])
                    if i < neural.oscillator_strengths.size
                    else "",
                ]
            )


def plot_absorption_spectrum(
    path: Path,
    spectrum: SpectrumRun,
    *,
    title: str,
    reference_label: str,
    neural_label: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(spectrum.grid_ev, spectrum.reference_curve, label=reference_label, lw=2.0)
    axes[0].plot(spectrum.grid_ev, spectrum.neural_curve, label=neural_label, lw=2.0)
    axes[0].set_xlabel("Energy (eV)")
    axes[0].set_ylabel("Absorption (arb. units)")
    axes[0].set_title("Full Range")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    axes[1].plot(
        spectrum.grid_ev[spectrum.low_energy_mask],
        spectrum.reference_curve[spectrum.low_energy_mask],
        label=reference_label,
        lw=2.0,
    )
    axes[1].plot(
        spectrum.grid_ev[spectrum.low_energy_mask],
        spectrum.neural_curve[spectrum.low_energy_mask],
        label=neural_label,
        lw=2.0,
    )
    axes[1].set_xlabel("Energy (eV)")
    axes[1].set_ylabel("Absorption (arb. units)")
    axes[1].set_title("Low-Energy Zoom")
    axes[1].grid(alpha=0.2)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_training_curve_csv(path: Path, training: TrainingRun) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "loss",
                "density_penalty",
                "stationarity_penalty",
                "coefficient_prior_penalty",
                "grad_norm",
                "grad_abs_max",
                "param_update_norm",
                "nonfinite_grad_fraction",
            ]
        )
        for step, (
            loss,
            density_penalty,
            stationarity_penalty,
            coefficient_prior_penalty,
            grad_norm,
            grad_abs_max,
            param_update_norm,
            nonfinite_grad_fraction,
        ) in enumerate(
            zip(
                training.loss_history,
                training.density_penalty_history,
                training.stationarity_penalty_history,
                training.coefficient_prior_penalty_history,
                training.grad_norm_history,
                training.grad_abs_max_history,
                training.param_update_norm_history,
                training.nonfinite_grad_fraction_history,
                strict=True,
            )
        ):
            writer.writerow(
                [
                    step,
                    loss,
                    density_penalty,
                    stationarity_penalty,
                    coefficient_prior_penalty,
                    grad_norm,
                    grad_abs_max,
                    param_update_norm,
                    nonfinite_grad_fraction,
                ]
            )


def plot_training_curves(path: Path, training: TrainingRun, *, title: str) -> None:
    steps = np.arange(len(training.loss_history))
    loss_values = np.asarray(training.loss_history, dtype=float)
    density_penalty_values = np.asarray(training.density_penalty_history, dtype=float)
    stationarity_penalty_values = np.asarray(training.stationarity_penalty_history, dtype=float)
    coefficient_prior_penalty_values = np.asarray(
        training.coefficient_prior_penalty_history,
        dtype=float,
    )

    # Keep the default report compact when only the main loss is active.
    if not (
        np.any(density_penalty_values > 0.0)
        or np.any(stationarity_penalty_values > 0.0)
        or np.any(coefficient_prior_penalty_values > 0.0)
    ):
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.4))
        ax.plot(steps, np.maximum(loss_values, 1e-16), lw=1.9)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.0))

    axes[0].plot(steps, np.maximum(loss_values, 1e-16), lw=1.8)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.2)

    if np.any(density_penalty_values > 0.0):
        axes[1].plot(steps, np.maximum(density_penalty_values, 1e-16), lw=1.8)
        axes[1].set_yscale("log")
    else:
        axes[1].plot(steps, density_penalty_values, lw=1.8)
        axes[1].set_yscale("linear")
        axes[1].text(
            0.03,
            0.92,
            "All density penalties are zero",
            transform=axes[1].transAxes,
            fontsize=9,
            ha="left",
            va="top",
        )
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Density Penalty")
    axes[1].set_title("Density Matching")
    axes[1].grid(alpha=0.2)

    if np.any(stationarity_penalty_values > 0.0):
        axes[2].plot(steps, np.maximum(stationarity_penalty_values, 1e-16), lw=1.8)
        axes[2].set_yscale("log")
    else:
        axes[2].plot(steps, stationarity_penalty_values, lw=1.8)
        axes[2].set_yscale("linear")
        axes[2].text(
            0.03,
            0.92,
            "All stationarity penalties are zero",
            transform=axes[2].transAxes,
            fontsize=9,
            ha="left",
            va="top",
        )
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Stationarity Penalty")
    axes[2].set_title("Fock OV Constraint")
    axes[2].grid(alpha=0.2)

    if np.any(coefficient_prior_penalty_values > 0.0):
        axes[3].plot(steps, np.maximum(coefficient_prior_penalty_values, 1e-16), lw=1.8)
        axes[3].set_yscale("log")
    else:
        axes[3].plot(steps, coefficient_prior_penalty_values, lw=1.8)
        axes[3].set_yscale("linear")
        axes[3].text(
            0.03,
            0.92,
            "All coefficient priors are zero",
            transform=axes[3].transAxes,
            fontsize=9,
            ha="left",
            va="top",
        )
    axes[3].set_xlabel("Step")
    axes[3].set_ylabel("Coefficient Prior")
    axes[3].set_title("DM21-Style Prior")
    axes[3].grid(alpha=0.2)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_outputs(
    *,
    reference: MoleculeRun,
    training: TrainingRun,
    neural: NeuralExcitedStateRun,
    spectrum: SpectrumRun,
    output: OutputConfig,
) -> OutputPaths:
    ensure_output_dirs(output)
    paths = build_output_paths(output)

    write_state_comparison_csv(paths.spectrum_csv, reference, neural)
    neural_label = output.neural_label_template.format(solver=neural.solver_label)
    plot_absorption_spectrum(
        paths.spectrum_png,
        spectrum,
        title=output.title,
        reference_label=output.reference_label,
        neural_label=neural_label,
    )

    if output.write_training_curves and paths.training_csv and paths.training_png:
        write_training_curve_csv(paths.training_csv, training)
        plot_training_curves(
            paths.training_png,
            training,
            title=f"{output.title} (Training Curves)",
        )

    return paths


def print_run_summary(run: PipelineRun, *, print_all_states: bool = True) -> None:
    reference = run.reference
    training = run.training
    neural = run.neural
    spectrum = run.spectrum
    outputs = run.outputs

    print(f"System: {run.system_label}")
    print(f"SCF reference energy (Hartree): {reference.molecule.mf_energy:.10f}")
    print(f"Neural fitted energy (Hartree): {training.trained_energy:.10f}")
    print(f"Neural hybrid fraction: {training.trained_hybrid_fraction:.6f}")
    print(f"Excited-state solver: {neural.solver_label}")
    print(
        f"nocc={reference.nocc}, nvir={reference.nvir}, "
        f"nstates={reference.nstates} (full={reference.nstates_full})"
    )
    print(f"Training loss: {training.initial_loss:.6e} -> {training.final_loss:.6e}")
    print(f"Training min loss: {training.min_loss:.6e} (step {training.min_loss_step})")
    print(f"Selected training params: step {training.min_loss_step} (minimum train loss)")
    print(
        "Density penalty: "
        f"{training.initial_density_penalty:.6e} -> {training.final_density_penalty:.6e}"
    )
    print(
        "Stationarity penalty: "
        f"{training.initial_stationarity_penalty:.6e} -> "
        f"{training.final_stationarity_penalty:.6e}"
    )
    print(
        "Coefficient prior penalty: "
        f"{training.initial_coefficient_prior_penalty:.6e} -> "
        f"{training.final_coefficient_prior_penalty:.6e}"
    )
    print(
        f"First-{spectrum.compared_states} states MAE: "
        f"{spectrum.low_energy_mae_ev:.3f} eV"
    )
    print(
        "Wall time (s): "
        f"SCF={reference.scf_elapsed_s:.2f}, PySCF-TDDFT={reference.tddft_elapsed_s:.2f}, "
        f"Train={training.elapsed_s:.2f}, Neural-TDDFT={neural.elapsed_s:.2f}"
    )
    print("")

    if print_all_states:
        def _format_cell(value: float | None, width: int, precision: int) -> str:
            if value is None or not np.isfinite(value):
                return f"{'N/A':>{width}}"
            return f"{value:>{width}.{precision}f}"

        print("State  B3LYP_eV  B3LYP_f   Neural_eV  Neural_f")
        nrows = max(reference.energies_au.size, neural.energies_au.size)
        for i in range(nrows):
            ref_e = (
                float(reference.energies_au[i] * HARTREE_TO_EV)
                if i < reference.energies_au.size
                else None
            )
            ref_f = (
                float(reference.oscillator_strengths[i])
                if i < reference.oscillator_strengths.size
                else None
            )
            neu_e = (
                float(neural.energies_au[i] * HARTREE_TO_EV)
                if i < neural.energies_au.size
                else None
            )
            neu_f = (
                float(neural.oscillator_strengths[i])
                if i < neural.oscillator_strengths.size
                else None
            )
            print(
                f"{i + 1:>3}  "
                f"{_format_cell(ref_e, 8, 3)}  "
                f"{_format_cell(ref_f, 8, 4)}  "
                f"{_format_cell(neu_e, 9, 3)}  "
                f"{_format_cell(neu_f, 8, 4)}"
            )
        print("")

    print(f"Wrote spectrum table to {outputs.spectrum_csv}")
    print(f"Wrote spectrum plot  to {outputs.spectrum_png}")
    if outputs.training_csv is not None and outputs.training_png is not None:
        print(f"Wrote training curve table to {outputs.training_csv}")
        print(f"Wrote training curve plot  to {outputs.training_png}")
