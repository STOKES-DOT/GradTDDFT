from __future__ import annotations

import argparse
import csv
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np

from td_graddft.workflows import (
    NeuralXCTrainingConfig,
    OutputConfig,
    SimulationConfig,
    SpectrumGridConfig,
    run_and_report,
)


def make_water_mf():
    from pyscf import dft, gto

    mol = gto.Mole()
    mol.atom = """
    O  0.000000  0.000000  0.117790
    H  0.000000  0.755453 -0.471161
    H  0.000000 -0.755453 -0.471161
    """
    mol.unit = "Angstrom"
    mol.basis = "sto-3g"
    mol.spin = 0
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = "b3lyp"
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF SCF did not converge for water.")
    return mf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare single- vs multi-channel semilocal Neural_xc on water."
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", default="outputs/water_semilocal_compare")
    return parser.parse_args()


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_training_overlay(path: Path, runs: list[tuple[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for label, run in runs:
        values = np.maximum(np.asarray(run.training.loss_history, dtype=float), 1e-16)
        ax.plot(np.arange(values.size), values, lw=2.0, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Ground-state loss")
    ax.set_title("Water Neural_xc Training Comparison")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_spectrum_overlay(path: Path, runs: list[tuple[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    reference_drawn = False
    for label, run in runs:
        grid = np.asarray(run.spectrum.grid_ev)
        reference_curve = np.asarray(run.spectrum.reference_curve)
        neural_curve = np.asarray(run.spectrum.neural_curve)
        if not reference_drawn:
            ax.plot(grid, reference_curve, lw=2.2, color="black", label="B3LYP reference")
            reference_drawn = True
        ax.plot(grid, neural_curve, lw=2.0, label=f"Neural_xc {label}")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Absorption (arb. units)")
    ax.set_title("Water Absorption Spectrum Comparison")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base_training = NeuralXCTrainingConfig(
        steps=args.steps,
        learning_rate=args.learning_rate,
        density_constraint_weight=0.0,
        energy_mse_weight=1.0,
        energy_mae_weight=1.0,
        energy_normalization="none",
        seed=args.seed,
        hidden_dims=(64, 64, 64),
        functional_name="water_neural_xc_fit",
    )
    simulation = SimulationConfig(nstates=-1)
    spectrum = SpectrumGridConfig(
        eta_ev=0.15,
        grid_min_ev=0.0,
        grid_points=2200,
        max_padding_ev=2.0,
        zoom_min_ev=5.0,
        zoom_max_ev=20.0,
        compare_states=8,
    )

    configs = [
        (
            "single_channel",
            replace(
                base_training,
                semilocal_xc="b3lyp_sl_approx",
                functional_name="water_neural_xc_single_channel",
            ),
        ),
        (
            "multi_channel",
            replace(
                base_training,
                semilocal_xc=("lda_x", "gga_x_b88", "lda_c_pw", "gga_c_pbe"),
                functional_name="water_neural_xc_multi_channel",
            ),
        ),
    ]

    runs: list[tuple[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for label, training in configs:
        run = run_and_report(
            system_label=f"H2O B3LYP/STO-3G [{label}]",
            mf_builder=make_water_mf,
            training_config=training,
            simulation_config=simulation,
            spectrum_config=spectrum,
            output_config=OutputConfig(
                outdir=outdir,
                prefix=f"water_{label}",
                title=f"H2O Absorption Spectrum: {label}",
                reference_label="PySCF TDDFT B3LYP/STO-3G",
                neural_label_template=f"Neural_xc {label} ({{solver}})",
                write_training_curves=True,
                training_prefix=f"water_{label}_training",
            ),
            print_all_states=False,
        )
        runs.append((label, run))
        summary_rows.append(
            {
                "label": label,
                "semilocal_xc": training.semilocal_xc,
                "hf_input_mode": training.hf_input_mode,
                "coefficient_positivity": training.coefficient_positivity,
                "hidden_dims": training.hidden_dims,
                "steps": training.steps,
                "learning_rate": training.learning_rate,
                "initial_loss": run.training.initial_loss,
                "final_loss": run.training.final_loss,
                "min_loss": run.training.min_loss,
                "min_loss_step": run.training.min_loss_step,
                "trained_energy": run.training.trained_energy,
                "trained_hybrid_fraction": run.training.trained_hybrid_fraction,
                "low_energy_mae_ev": run.spectrum.low_energy_mae_ev,
                "solver_label": run.neural.solver_label,
                "spectrum_csv": str(run.outputs.spectrum_csv),
                "spectrum_png": str(run.outputs.spectrum_png),
                "training_csv": str(run.outputs.training_csv),
                "training_png": str(run.outputs.training_png),
            }
        )

    _write_summary_csv(outdir / "water_semilocal_compare_summary.csv", summary_rows)
    _plot_training_overlay(outdir / "water_semilocal_compare_training.png", runs)
    _plot_spectrum_overlay(outdir / "water_semilocal_compare_spectrum.png", runs)


if __name__ == "__main__":
    main()
