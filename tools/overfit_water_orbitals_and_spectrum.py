from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyscf import dft, gto

from td_graddft.jax_libxc import b3lyp_component_basis
from td_graddft.scf import DifferentiableSCF, DifferentiableSCFConfig
from td_graddft.spectra import HARTREE_TO_EV
from td_graddft.workflows.core import run_pipeline_core
from td_graddft.workflows.reporting import plot_training_curves, write_training_curve_csv
from td_graddft.workflows.types import (
    NeuralXCTrainingConfig,
    SimulationConfig,
    SpectrumGridConfig,
)


WATER_GEOM = """
O  0.000000  0.000000  0.117790
H  0.000000  0.755453 -0.471161
H  0.000000 -0.755453 -0.471161
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-molecule H2O Neural_xc overfit with MO/spectrum comparison."
    )
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--xc", default="b3lyp")
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--learning-rate", type=float, default=5e-3)
    p.add_argument(
        "--loss-mode",
        choices=("mixed", "excited_only"),
        default="mixed",
        help="Use the default mixed objective or supervise only excited states.",
    )
    p.add_argument(
        "--energy-mse-weight",
        type=float,
        default=1.0,
        help="Ground-state energy MSE weight when --loss-mode=mixed.",
    )
    p.add_argument(
        "--energy-mae-weight",
        type=float,
        default=0.0,
        help="Ground-state energy MAE weight when --loss-mode=mixed.",
    )
    p.add_argument(
        "--grad-clip-norm",
        type=float,
        default=None,
        help="Global gradient clipping norm for training stability (disabled when omitted).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 64, 64])
    p.add_argument(
        "--semilocal-xc",
        nargs="+",
        default=list(b3lyp_component_basis()),
        help="Semilocal basis channels used by Neural_xc.",
    )
    p.add_argument("--nstates", type=int, default=10)
    p.add_argument("--eta-ev", type=float, default=0.10)
    p.add_argument("--grid-min-ev", type=float, default=0.0)
    p.add_argument("--grid-points", type=int, default=1800)
    p.add_argument("--zoom-min-ev", type=float, default=5.0)
    p.add_argument("--zoom-max-ev", type=float, default=20.0)
    p.add_argument("--density-weight", type=float, default=1e-3)
    p.add_argument(
        "--s1-weight",
        type=float,
        default=0.0,
        help="Weight for first-excitation-energy (S1) supervision term.",
    )
    p.add_argument(
        "--s1-use-tda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use TDA (faster/more stable) when evaluating S1 supervision during training.",
    )
    p.add_argument(
        "--excited-weight",
        type=float,
        default=0.0,
        help="Weight for multi-state excitation-energy supervision term.",
    )
    p.add_argument(
        "--excited-nstates",
        type=int,
        default=3,
        help="Number of lowest excitation states supervised when --excited-weight > 0.",
    )
    p.add_argument(
        "--excited-use-tda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use TDA when evaluating multi-state excitation supervision during training.",
    )
    p.add_argument(
        "--spectrum-weight",
        type=float,
        default=0.0,
        help="Weight for broadened absorption-spectrum supervision during training.",
    )
    p.add_argument(
        "--spectrum-nstates",
        type=int,
        default=0,
        help="Number of lowest states used to build the supervised spectrum; 0 means use --nstates.",
    )
    p.add_argument(
        "--spectrum-use-tda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use TDA when evaluating spectrum supervision during training.",
    )
    p.add_argument(
        "--spectrum-eta-ev",
        type=float,
        default=None,
        help="Lorentzian broadening used by spectrum supervision; defaults to --eta-ev.",
    )
    p.add_argument("--training-mode", choices=("fixed_density", "self_consistent"), default="fixed_density")
    p.add_argument("--scf-max-cycle", type=int, default=16)
    p.add_argument("--scf-damping", type=float, default=0.30)
    p.add_argument("--scf-conv-tol-density", type=float, default=1e-8)
    p.add_argument("--scf-vxc-clip", type=float, default=20.0)
    p.add_argument(
        "--scf-iterate-selection",
        choices=("final", "best_rms", "first_converged"),
        default="best_rms",
    )
    p.add_argument(
        "--scf-gradient-mode",
        choices=("auto", "unrolled", "implicit_commutator"),
        default="auto",
    )
    p.add_argument(
        "--recover-nonfinite-steps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recover a step when parameters become non-finite after an optimizer update.",
    )
    p.add_argument("--outdir", default="outputs/water_single_overfit_orbital_spectrum")
    return p.parse_args()


def _make_water_mf(*, basis: str, xc: str):
    mol = gto.Mole()
    mol.atom = WATER_GEOM
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.spin = 0
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF reference SCF did not converge.")
    return mf


def _restricted_vector(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr[0]
    raise ValueError(f"Expected shape (nmo,) or (spin, nmo), got {arr.shape}.")


def _restricted_occ(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr.sum(axis=0)
    raise ValueError(f"Expected shape (nmo,) or (spin, nmo), got {arr.shape}.")


def _write_orbital_csv(
    path: Path,
    mo_occ: np.ndarray,
    ref_mo_ha: np.ndarray,
    neural_mo_ha: np.ndarray,
) -> dict[str, float]:
    diff = neural_mo_ha - ref_mo_ha
    abs_diff = np.abs(diff)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "mo_index",
                "occupation",
                "ref_mo_ha",
                "neural_mo_ha",
                "diff_ha",
                "abs_diff_ha",
                "ref_mo_ev",
                "neural_mo_ev",
                "abs_diff_mev",
            ]
        )
        for i in range(ref_mo_ha.size):
            w.writerow(
                [
                    i,
                    float(mo_occ[i]),
                    float(ref_mo_ha[i]),
                    float(neural_mo_ha[i]),
                    float(diff[i]),
                    float(abs_diff[i]),
                    float(ref_mo_ha[i] * HARTREE_TO_EV),
                    float(neural_mo_ha[i] * HARTREE_TO_EV),
                    float(abs_diff[i] * HARTREE_TO_EV * 1000.0),
                ]
            )
    return {
        "orbital_mae_ha": float(abs_diff.mean()),
        "orbital_max_abs_ha": float(abs_diff.max()),
        "orbital_rmse_ha": float(np.sqrt(np.mean(diff**2))),
        "orbital_mae_mev": float(abs_diff.mean() * HARTREE_TO_EV * 1000.0),
        "orbital_max_abs_mev": float(abs_diff.max() * HARTREE_TO_EV * 1000.0),
    }


def _write_state_csv(
    path: Path,
    ref_e_au: np.ndarray,
    ref_f: np.ndarray,
    neural_e_au: np.ndarray,
    neural_f: np.ndarray,
) -> dict[str, float]:
    n = int(min(ref_e_au.size, ref_f.size, neural_e_au.size, neural_f.size))
    if n <= 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "state",
                    "ref_energy_ev",
                    "neural_energy_ev",
                    "abs_diff_ev",
                    "ref_osc",
                    "neural_osc",
                    "abs_diff_osc",
                ]
            )
        return {
            "nstate_compared": 0.0,
            "state_mae_ev": float("nan"),
            "state_max_abs_ev": float("nan"),
            "osc_mae": float("nan"),
            "osc_max_abs": float("nan"),
        }

    ref_e_ev = ref_e_au[:n] * HARTREE_TO_EV
    neural_e_ev = neural_e_au[:n] * HARTREE_TO_EV
    de = np.abs(neural_e_ev - ref_e_ev)
    df = np.abs(neural_f[:n] - ref_f[:n])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "state",
                "ref_energy_ev",
                "neural_energy_ev",
                "abs_diff_ev",
                "ref_osc",
                "neural_osc",
                "abs_diff_osc",
            ]
        )
        for i in range(n):
            w.writerow(
                [
                    i + 1,
                    float(ref_e_ev[i]),
                    float(neural_e_ev[i]),
                    float(de[i]),
                    float(ref_f[i]),
                    float(neural_f[i]),
                    float(df[i]),
                ]
            )
    return {
        "nstate_compared": float(n),
        "state_mae_ev": float(de.mean()),
        "state_max_abs_ev": float(de.max()),
        "osc_mae": float(df.mean()),
        "osc_max_abs": float(df.max()),
    }


def _write_spectrum(
    *,
    curve_csv: Path,
    curve_png: Path,
    grid_ev: np.ndarray,
    ref_curve: np.ndarray,
    neural_curve: np.ndarray,
    title: str,
) -> None:
    with curve_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["energy_ev", "ref_absorption", "neural_absorption"])
        for i in range(grid_ev.size):
            w.writerow([float(grid_ev[i]), float(ref_curve[i]), float(neural_curve[i])])

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(grid_ev, ref_curve, lw=2.0, label="PySCF TDDFT")
    ax.plot(grid_ev, neural_curve, lw=2.0, label="Neural_xc TDDFT")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Absorption (arb. units)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(curve_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    loss_mode = str(args.loss_mode)
    energy_mse_weight = float(args.energy_mse_weight)
    energy_mae_weight = float(args.energy_mae_weight)
    density_weight = float(args.density_weight)
    spectrum_weight = float(args.spectrum_weight)
    if loss_mode == "excited_only":
        energy_mse_weight = 0.0
        energy_mae_weight = 0.0
        density_weight = 0.0
        if (
            float(args.s1_weight) == 0.0
            and float(args.excited_weight) == 0.0
            and spectrum_weight == 0.0
        ):
            raise ValueError(
                "excited_only loss requires --s1-weight, --excited-weight, or --spectrum-weight to be non-zero."
            )
    spectrum_nstates = int(args.spectrum_nstates)
    if spectrum_nstates <= 0:
        spectrum_nstates = int(args.nstates)
    spectrum_eta_ev = float(args.eta_ev) if args.spectrum_eta_ev is None else float(args.spectrum_eta_ev)

    training = NeuralXCTrainingConfig(
        steps=int(args.steps),
        learning_rate=float(args.learning_rate),
        gradient_clip_norm=(None if args.grad_clip_norm is None else float(args.grad_clip_norm)),
        density_constraint_weight=density_weight,
        s1_constraint_weight=float(args.s1_weight),
        s1_constraint_use_tda=bool(args.s1_use_tda),
        excitation_constraint_weight=float(args.excited_weight),
        excitation_constraint_nstates=max(1, int(args.excited_nstates)),
        excitation_constraint_use_tda=bool(args.excited_use_tda),
        spectrum_constraint_weight=spectrum_weight,
        spectrum_constraint_nstates=max(1, spectrum_nstates),
        spectrum_constraint_use_tda=bool(args.spectrum_use_tda),
        energy_mse_weight=energy_mse_weight,
        energy_mae_weight=energy_mae_weight,
        energy_normalization="none",
        seed=int(args.seed),
        hidden_dims=tuple(int(v) for v in args.hidden_dims),
        semilocal_xc=tuple(str(v) for v in args.semilocal_xc),
        functional_name="water_neural_xc_overfit",
        training_mode=str(args.training_mode),
        scf_max_cycle=int(args.scf_max_cycle),
        scf_damping=float(args.scf_damping),
        scf_conv_tol_density=float(args.scf_conv_tol_density),
        scf_vxc_clip=float(args.scf_vxc_clip),
        scf_iterate_selection=str(args.scf_iterate_selection),
        scf_gradient_mode=str(args.scf_gradient_mode),
        recover_nonfinite_steps=bool(args.recover_nonfinite_steps),
        log_interval=max(1, int(args.steps // 10)),
    )
    simulation = SimulationConfig(
        nstates=int(args.nstates),
        scf_backend="pyscf",
        execution_device="cpu",
        jit_tddft=False,
    )
    spectrum = SpectrumGridConfig(
        eta_ev=spectrum_eta_ev,
        grid_min_ev=float(args.grid_min_ev),
        grid_points=int(args.grid_points),
        zoom_min_ev=float(args.zoom_min_ev),
        zoom_max_ev=float(args.zoom_max_ev),
        compare_states=min(10, int(args.nstates)),
    )

    reference, train_run, neural_run, spectrum_run = run_pipeline_core(
        mf_builder=lambda: _make_water_mf(basis=str(args.basis), xc=str(args.xc)),
        training_config=training,
        simulation_config=simulation,
        spectrum_config=spectrum,
    )

    scf_solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            max_cycle=int(args.scf_max_cycle),
            damping=float(args.scf_damping),
            conv_tol_density=float(args.scf_conv_tol_density),
            vxc_clip=float(args.scf_vxc_clip),
            iterate_selection=str(args.scf_iterate_selection),
        )
    )
    neural_molecule, scf_info = scf_solver.run(
        reference.molecule,
        train_run.functional,
        train_run.params,
    )

    ref_mo_ha = _restricted_vector(np.asarray(reference.molecule.mo_energy, dtype=float))
    neural_mo_ha = _restricted_vector(np.asarray(neural_molecule.mo_energy, dtype=float))
    mo_occ = _restricted_occ(np.asarray(reference.molecule.mo_occ, dtype=float))
    orbital_csv = outdir / "water_orbital_energy_compare.csv"
    orbital_metrics = _write_orbital_csv(
        orbital_csv,
        mo_occ=mo_occ,
        ref_mo_ha=ref_mo_ha,
        neural_mo_ha=neural_mo_ha,
    )

    state_csv = outdir / "water_state_compare.csv"
    state_metrics = _write_state_csv(
        state_csv,
        ref_e_au=np.asarray(reference.energies_au, dtype=float),
        ref_f=np.asarray(reference.oscillator_strengths, dtype=float),
        neural_e_au=np.asarray(neural_run.energies_au, dtype=float),
        neural_f=np.asarray(neural_run.oscillator_strengths, dtype=float),
    )

    spectrum_curve_csv = outdir / "water_spectrum_curve.csv"
    spectrum_png = outdir / "water_spectrum_compare.png"
    _write_spectrum(
        curve_csv=spectrum_curve_csv,
        curve_png=spectrum_png,
        grid_ev=np.asarray(spectrum_run.grid_ev, dtype=float),
        ref_curve=np.asarray(spectrum_run.reference_curve, dtype=float),
        neural_curve=np.asarray(spectrum_run.neural_curve, dtype=float),
        title=f"H2O {args.xc.upper()}/{args.basis}: PySCF vs Neural_xc",
    )

    training_curve_csv = outdir / "water_training_curve.csv"
    training_curve_png = outdir / "water_training_curve.png"
    write_training_curve_csv(training_curve_csv, train_run)
    plot_training_curves(
        training_curve_png,
        train_run,
        title=f"H2O {args.xc.upper()}/{args.basis}: Training Loss",
    )

    grad_norm_vals = np.asarray(train_run.grad_norm_history, dtype=float)
    grad_abs_max_vals = np.asarray(train_run.grad_abs_max_history, dtype=float)
    grad_update_vals = np.asarray(train_run.param_update_norm_history, dtype=float)
    grad_nonfinite_vals = np.asarray(train_run.nonfinite_grad_fraction_history, dtype=float)
    grad_norm_finite = grad_norm_vals[np.isfinite(grad_norm_vals)]
    grad_abs_max_finite = grad_abs_max_vals[np.isfinite(grad_abs_max_vals)]
    grad_update_finite = grad_update_vals[np.isfinite(grad_update_vals)]
    grad_nonfinite_finite = grad_nonfinite_vals[np.isfinite(grad_nonfinite_vals)]

    summary = {
        "basis": str(args.basis),
        "xc": str(args.xc),
        "steps": int(args.steps),
        "learning_rate": float(args.learning_rate),
        "loss_mode": loss_mode,
        "energy_mse_weight": energy_mse_weight,
        "energy_mae_weight": energy_mae_weight,
        "grad_clip_norm": (
            float(args.grad_clip_norm) if args.grad_clip_norm is not None else float("nan")
        ),
        "training_mode": str(args.training_mode),
        "density_weight": density_weight,
        "s1_weight": float(args.s1_weight),
        "s1_use_tda": bool(args.s1_use_tda),
        "excited_weight": float(args.excited_weight),
        "excited_nstates": int(args.excited_nstates),
        "excited_use_tda": bool(args.excited_use_tda),
        "spectrum_weight": spectrum_weight,
        "spectrum_nstates": spectrum_nstates,
        "spectrum_use_tda": bool(args.spectrum_use_tda),
        "spectrum_eta_ev": spectrum_eta_ev,
        "reference_s1_ev": (
            float(np.asarray(reference.energies_au, dtype=float)[0] * HARTREE_TO_EV)
            if np.asarray(reference.energies_au).size > 0
            else float("nan")
        ),
        "neural_s1_ev": (
            float(np.asarray(neural_run.energies_au, dtype=float)[0] * HARTREE_TO_EV)
            if np.asarray(neural_run.energies_au).size > 0
            else float("nan")
        ),
        "initial_loss": float(train_run.initial_loss),
        "final_loss": float(train_run.final_loss),
        "min_loss": float(train_run.min_loss),
        "min_loss_step": int(train_run.min_loss_step),
        "grad_norm_min": (
            float(np.min(grad_norm_finite)) if grad_norm_finite.size > 0 else float("nan")
        ),
        "grad_norm_max": (
            float(np.max(grad_norm_finite)) if grad_norm_finite.size > 0 else float("nan")
        ),
        "grad_norm_last": (
            float(grad_norm_finite[-1]) if grad_norm_finite.size > 0 else float("nan")
        ),
        "grad_abs_max_last": (
            float(grad_abs_max_finite[-1]) if grad_abs_max_finite.size > 0 else float("nan")
        ),
        "param_update_norm_last": (
            float(grad_update_finite[-1]) if grad_update_finite.size > 0 else float("nan")
        ),
        "nonfinite_grad_fraction_max": (
            float(np.max(grad_nonfinite_finite))
            if grad_nonfinite_finite.size > 0
            else float("nan")
        ),
        "trained_energy_ha": float(train_run.trained_energy),
        "trained_hybrid_fraction": float(train_run.trained_hybrid_fraction),
        "scf_converged": bool(scf_info.converged),
        "scf_cycles": int(scf_info.cycles),
        "scf_selected_cycle": int(scf_info.selected_cycle),
        "scf_selected_rms_density": float(scf_info.selected_rms_density),
        **orbital_metrics,
        **state_metrics,
        "orbital_csv": str(orbital_csv),
        "state_csv": str(state_csv),
        "spectrum_curve_csv": str(spectrum_curve_csv),
        "spectrum_png": str(spectrum_png),
        "training_curve_csv": str(training_curve_csv),
        "training_curve_png": str(training_curve_png),
    }

    summary_path = outdir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}={v}\n")

    print(summary)
    print(f"summary={summary_path}")
    print(f"orbital_csv={orbital_csv}")
    print(f"state_csv={state_csv}")
    print(f"spectrum_curve_csv={spectrum_curve_csv}")
    print(f"spectrum_png={spectrum_png}")
    print(f"training_curve_csv={training_curve_csv}")
    print(f"training_curve_png={training_curve_png}")


if __name__ == "__main__":
    main()
