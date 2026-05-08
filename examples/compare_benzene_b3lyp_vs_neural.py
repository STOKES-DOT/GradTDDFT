from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

from pyscf import dft, gto

from td_graddft.neural_xc_presets import (
    DM21_B3LYP_NEURAL_XC_PRESET,
    resolve_coefficient_prior_values,
)
from td_graddft.orbital_compare import (
    plot_orbital_compare_panel,
    render_restricted_orbital_surfaces,
)
from td_graddft.scf import DifferentiableSCF, DifferentiableSCFConfig
from td_graddft.spectra import HARTREE_TO_EV
from td_graddft.workflows import (
    NeuralXCTrainingConfig,
    OutputConfig,
    SimulationConfig,
    SpectrumGridConfig,
    run_and_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Neural_xc on benzene ground state and compare absorption "
            "spectrum against PySCF TDDFT."
        )
    )
    parser.add_argument("--steps", type=int, default=2500, help="training steps")
    parser.add_argument("--lr", type=float, default=0.005, help="Adam learning rate")
    parser.add_argument(
        "--lr-decay-every",
        type=int,
        default=0,
        help="staircase learning-rate decay interval in steps (0 disables)",
    )
    parser.add_argument(
        "--lr-decay-factor",
        type=float,
        default=0.5,
        help="staircase learning-rate decay factor",
    )
    parser.add_argument(
        "--no-jit",
        action="store_true",
        help="disable JIT for the training loop (useful for CPU debugging/overfit runs)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=0,
        help="print training progress every N steps (0 disables)",
    )
    parser.add_argument(
        "--density-weight",
        type=float,
        default=1e-3,
        help="weight for direct grid-density matching loss",
    )
    parser.add_argument(
        "--dm21-scf-weight",
        type=float,
        default=0.0,
        help="weight for DM21-style one-step SCF regularization term",
    )
    parser.add_argument(
        "--dm21-scf-gap-floor",
        type=float,
        default=1e-3,
        help="minimum |epsilon_i - epsilon_j| (Ha) used in DM21 SCF regularizer",
    )
    parser.add_argument(
        "--density-supervision",
        choices=("spin_summed", "spin_resolved"),
        default=DM21_B3LYP_NEURAL_XC_PRESET.density_supervision,
        help="density matching mode when density loss is enabled",
    )
    parser.add_argument(
        "--coefficient-prior-weight",
        type=float,
        default=0.0,
        help="optional DM21-style coefficient prior weight",
    )
    parser.add_argument(
        "--orbital-energy-loss-weight",
        type=float,
        default=0.0,
        help="weight for frontier orbital-energy loss during training",
    )
    parser.add_argument(
        "--orbital-energy-loss-mae-weight",
        type=float,
        default=0.0,
        help="MAE coefficient inside the orbital-energy training loss",
    )
    parser.add_argument(
        "--excitation-weight",
        type=float,
        default=0.0,
        help="weight for low-lying excitation-energy supervision during training",
    )
    parser.add_argument(
        "--excitation-nstates",
        type=int,
        default=3,
        help="number of low-lying excitation energies supervised during training",
    )
    parser.add_argument(
        "--excitation-use-tda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use TDA instead of Casida for excitation-energy supervision",
    )
    parser.add_argument(
        "--excitation-loss-mse-weight",
        type=float,
        default=0.0,
        help="MSE coefficient inside the excitation-energy training loss",
    )
    parser.add_argument(
        "--excitation-loss-mae-weight",
        type=float,
        default=1.0,
        help="MAE coefficient inside the excitation-energy training loss",
    )
    parser.add_argument(
        "--oscillator-strength-weight",
        type=float,
        default=0.0,
        help="weight for direct oscillator-strength supervision during training",
    )
    parser.add_argument(
        "--oscillator-strength-nstates",
        type=int,
        default=3,
        help="number of low-lying oscillator strengths supervised during training",
    )
    parser.add_argument(
        "--oscillator-strength-use-tda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use TDA instead of Casida for oscillator-strength supervision",
    )
    parser.add_argument(
        "--oscillator-strength-loss-mse-weight",
        type=float,
        default=0.0,
        help="MSE coefficient inside the oscillator-strength training loss",
    )
    parser.add_argument(
        "--oscillator-strength-loss-mae-weight",
        type=float,
        default=1.0,
        help="MAE coefficient inside the oscillator-strength training loss",
    )
    parser.add_argument(
        "--spectrum-weight",
        type=float,
        default=0.0,
        help="weight for spectrum/oscillator-strength supervision during training",
    )
    parser.add_argument(
        "--spectrum-nstates",
        type=int,
        default=3,
        help="number of excited states included in spectrum supervision",
    )
    parser.add_argument(
        "--spectrum-use-tda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use TDA instead of Casida for spectrum supervision",
    )
    parser.add_argument(
        "--spectrum-loss-mse-weight",
        type=float,
        default=0.0,
        help="MSE coefficient inside the spectrum training loss",
    )
    parser.add_argument(
        "--spectrum-loss-mae-weight",
        type=float,
        default=1.0,
        help="MAE coefficient inside the spectrum training loss",
    )
    parser.add_argument(
        "--coefficient-prior-values",
        type=float,
        nargs="+",
        default=None,
        help="target Neural_xc channel coefficients matching the basis order",
    )
    parser.add_argument(
        "--coefficient-prior-mode",
        choices=("pointwise", "mean"),
        default=DM21_B3LYP_NEURAL_XC_PRESET.coefficient_prior_mode,
        help="regularize local coefficients or only their grid mean",
    )
    parser.add_argument(
        "--states",
        type=int,
        default=-1,
        help="number of excited states (<=0 means full nocc*nvir)",
    )
    parser.add_argument("--eta-ev", type=float, default=0.20, help="Lorentzian width in eV")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--basis", type=str, default="sto-3g", help="AO basis for PySCF")
    parser.add_argument("--xc", type=str, default="b3lyp", help="reference XC for PySCF")
    parser.add_argument(
        "--grids-level",
        type=int,
        default=0,
        help="PySCF numerical integration grid level",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="benzene_b3lyp_vs_neural_xc",
        help="output file prefix under outputs/",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[64, 64, 64],
        help="MLP hidden dimensions for Neural_xc",
    )
    parser.add_argument(
        "--network-architecture",
        choices=("simple_mlp", "graddft_residual"),
        default="graddft_residual",
        help="mixing-network backbone for Neural_xc",
    )
    parser.add_argument(
        "--strict-graddft-ground-state",
        action="store_true",
        help=(
            "force GradDFT-style ground-state alignment: residual DM21 backbone, "
            "DM21 original inputs, MSE-only energy loss, and no excited-state/auxiliary penalties"
        ),
    )
    parser.add_argument(
        "--skip-orbital-compare",
        action="store_true",
        help="skip HOMO-1/HOMO/LUMO/LUMO+1 real-vs-neural orbital rendering",
    )
    parser.add_argument(
        "--skip-orbital-energy-compare",
        action="store_true",
        help="skip HOMO-k...LUMO+k orbital-energy parity plotting",
    )
    parser.add_argument(
        "--orbital-energy-window",
        type=int,
        default=10,
        help=(
            "frontier orbital-energy window k (compare HOMO-k...HOMO and "
            "LUMO...LUMO+k). Use <=0 to compare all occupied+virtual orbitals."
        ),
    )
    parser.add_argument(
        "--xyzrender-src",
        default="/Volumes/TF/QH9_db/xyzrender/src",
        help="xyzrender source directory",
    )
    parser.add_argument("--orbital-iso", type=float, default=0.05)
    parser.add_argument("--orbital-diff-iso", type=float, default=0.03)
    parser.add_argument("--orbital-mo-blur", type=float, default=1.2)
    parser.add_argument("--orbital-mo-upsample", type=int, default=4)
    parser.add_argument("--orbital-cube-grid", type=int, default=48)
    parser.add_argument("--orbital-canvas-size", type=int, default=620)
    parser.add_argument(
        "--disable-orbital-frontier-match",
        action="store_true",
        help="disable overlap-based HOMO-1/HOMO and LUMO/LUMO+1 orbital matching",
    )
    parser.add_argument(
        "--orbital-frontier-match-window",
        type=int,
        default=6,
        help="number of near-frontier candidate orbitals used for overlap matching",
    )
    parser.add_argument("--orbital-scf-max-cycle", type=int, default=160)
    parser.add_argument("--orbital-scf-damping", type=float, default=0.90)
    parser.add_argument("--orbital-scf-conv-tol-density", type=float, default=1e-6)
    parser.add_argument("--orbital-scf-vxc-clip", type=float, default=20.0)
    parser.add_argument(
        "--orbital-scf-iterate-selection",
        choices=("final", "best_rms", "first_converged"),
        default="first_converged",
    )
    return parser.parse_args()


def make_benzene_mf(*, basis: str, xc: str, grids_level: int):
    mol = gto.Mole()
    mol.atom = """
    C   0.000000   1.396792   0.000000
    C   1.209657   0.698396   0.000000
    C   1.209657  -0.698396   0.000000
    C   0.000000  -1.396792   0.000000
    C  -1.209657  -0.698396   0.000000
    C  -1.209657   0.698396   0.000000
    H   0.000000   2.484212   0.000000
    H   2.151390   1.242106   0.000000
    H   2.151390  -1.242106   0.000000
    H   0.000000  -2.484212   0.000000
    H  -2.151390  -1.242106   0.000000
    H  -2.151390   1.242106   0.000000
    """
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.spin = 0
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = grids_level
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF ground-state SCF did not converge for benzene.")
    return mf


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _restricted_energies_and_occ(mo_energy, mo_occ) -> tuple[np.ndarray, np.ndarray]:
    energies = np.asarray(mo_energy, dtype=float)
    occ = np.asarray(mo_occ, dtype=float)
    if energies.ndim == 2:
        energies = energies[0]
    if occ.ndim == 2:
        occ = occ[0]
    return energies, occ


def _frontier_orbital_label(index: int, homo_idx: int, lumo_idx: int) -> tuple[str, str]:
    if index <= homo_idx:
        offset = homo_idx - index
        label = "HOMO" if offset == 0 else f"HOMO-{offset}"
        return label, "occupied"
    offset = index - lumo_idx
    label = "LUMO" if offset == 0 else f"LUMO+{offset}"
    return label, "virtual"


def _frontier_orbital_energy_rows(
    *,
    reference_mo_energy,
    reference_mo_occ,
    predicted_mo_energy,
    predicted_mo_occ,
    window: int,
) -> list[dict[str, float | int | str]]:
    ref_energies_ha, ref_occ = _restricted_energies_and_occ(reference_mo_energy, reference_mo_occ)
    pred_energies_ha, pred_occ = _restricted_energies_and_occ(predicted_mo_energy, predicted_mo_occ)
    nmo = int(min(ref_energies_ha.size, pred_energies_ha.size, ref_occ.size, pred_occ.size))
    if nmo == 0:
        return []

    occ_idx = np.where(ref_occ[:nmo] > 1e-8)[0]
    vir_idx = np.where(ref_occ[:nmo] <= 1e-8)[0]
    if occ_idx.size == 0 or vir_idx.size == 0:
        return []

    homo_idx = int(occ_idx[-1])
    lumo_idx = int(vir_idx[0])
    ref_zero_ha = 0.5 * float(ref_energies_ha[homo_idx] + ref_energies_ha[lumo_idx])
    pred_zero_ha = 0.5 * float(pred_energies_ha[homo_idx] + pred_energies_ha[lumo_idx])
    ref_zero_ev = ref_zero_ha * HARTREE_TO_EV
    pred_zero_ev = pred_zero_ha * HARTREE_TO_EV
    if int(window) <= 0:
        indices = list(range(0, homo_idx + 1)) + list(range(lumo_idx, nmo))
    else:
        w = max(int(window), 1)
        occ_start = max(homo_idx - w + 1, 0)
        vir_stop = min(lumo_idx + w, nmo)
        indices = list(range(occ_start, homo_idx + 1)) + list(range(lumo_idx, vir_stop))

    rows: list[dict[str, float | int | str]] = []
    for idx in indices:
        label, orbital_type = _frontier_orbital_label(idx, homo_idx, lumo_idx)
        ref_ha = float(ref_energies_ha[idx])
        pred_ha = float(pred_energies_ha[idx])
        ref_ev = ref_ha * HARTREE_TO_EV
        pred_ev = pred_ha * HARTREE_TO_EV
        aligned_ref_ha = ref_ha - ref_zero_ha
        aligned_pred_ha = pred_ha - pred_zero_ha
        aligned_ref_ev = aligned_ref_ha * HARTREE_TO_EV
        aligned_pred_ev = aligned_pred_ha * HARTREE_TO_EV
        rows.append(
            {
                "mo_index": int(idx),
                "label": label,
                "orbital_type": orbital_type,
                "reference_midpoint_ha": ref_zero_ha,
                "predicted_midpoint_ha": pred_zero_ha,
                "reference_midpoint_ev": ref_zero_ev,
                "predicted_midpoint_ev": pred_zero_ev,
                "reference_energy_ha": ref_ha,
                "predicted_energy_ha": pred_ha,
                "reference_energy_ev": ref_ev,
                "predicted_energy_ev": pred_ev,
                "aligned_reference_energy_ha": aligned_ref_ha,
                "aligned_predicted_energy_ha": aligned_pred_ha,
                "aligned_reference_energy_ev": aligned_ref_ev,
                "aligned_predicted_energy_ev": aligned_pred_ev,
                "aligned_abs_error_ev": abs(aligned_pred_ev - aligned_ref_ev),
                "raw_abs_error_ev": abs(pred_ev - ref_ev),
            }
        )
    return rows


def _write_frontier_orbital_energy_parity(
    rows: list[dict[str, float | int | str]],
    *,
    outdir: Path,
    window: int,
) -> tuple[Path, Path, float, float]:
    plt = _plt()
    outdir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"frontier_orbital_energy_parity_homo{window}_lumo{window}"
        if int(window) > 0
        else "frontier_orbital_energy_parity_all"
    )
    csv_path = outdir / f"{stem}.csv"
    png_path = outdir / f"{stem}.png"

    with csv_path.open("w", encoding="utf-8") as f:
        f.write(
            "mo_index,label,orbital_type,reference_midpoint_ha,predicted_midpoint_ha,"
            "reference_midpoint_ev,predicted_midpoint_ev,reference_energy_ha,predicted_energy_ha,"
            "reference_energy_ev,predicted_energy_ev,aligned_reference_energy_ha,"
            "aligned_predicted_energy_ha,aligned_reference_energy_ev,aligned_predicted_energy_ev,"
            "aligned_abs_error_ev,raw_abs_error_ev\n"
        )
        for row in rows:
            f.write(
                f"{int(row['mo_index'])},{row['label']},{row['orbital_type']},"
                f"{float(row['reference_midpoint_ha']):.12f},"
                f"{float(row['predicted_midpoint_ha']):.12f},"
                f"{float(row['reference_midpoint_ev']):.10f},"
                f"{float(row['predicted_midpoint_ev']):.10f},"
                f"{float(row['reference_energy_ha']):.12f},"
                f"{float(row['predicted_energy_ha']):.12f},"
                f"{float(row['reference_energy_ev']):.10f},"
                f"{float(row['predicted_energy_ev']):.10f},"
                f"{float(row['aligned_reference_energy_ha']):.12f},"
                f"{float(row['aligned_predicted_energy_ha']):.12f},"
                f"{float(row['aligned_reference_energy_ev']):.10f},"
                f"{float(row['aligned_predicted_energy_ev']):.10f},"
                f"{float(row['aligned_abs_error_ev']):.10f},"
                f"{float(row['raw_abs_error_ev']):.10f}\n"
            )

    aligned_mae_ev = (
        float(np.mean([float(row["aligned_abs_error_ev"]) for row in rows]))
        if rows
        else float("nan")
    )
    raw_mae_ev = (
        float(np.mean([float(row["raw_abs_error_ev"]) for row in rows]))
        if rows
        else float("nan")
    )
    ref_all = np.asarray(
        [float(row["aligned_reference_energy_ev"]) for row in rows],
        dtype=float,
    )
    pred_all = np.asarray(
        [float(row["aligned_predicted_energy_ev"]) for row in rows],
        dtype=float,
    )
    axis_min = float(min(np.min(ref_all), np.min(pred_all)))
    axis_max = float(max(np.max(ref_all), np.max(pred_all)))
    span = axis_max - axis_min
    margin = 0.05 * (span if span > 1e-10 else 1.0)
    lo = axis_min - margin
    hi = axis_max + margin

    occ_rows = [row for row in rows if row["orbital_type"] == "occupied"]
    vir_rows = [row for row in rows if row["orbital_type"] == "virtual"]

    fig, ax = plt.subplots(figsize=(7.2, 6.8))
    if occ_rows:
        ax.scatter(
            [float(row["aligned_reference_energy_ev"]) for row in occ_rows],
            [float(row["aligned_predicted_energy_ev"]) for row in occ_rows],
            s=30,
            alpha=0.78,
            c="#1f77b4",
            label="Occupied",
        )
    if vir_rows:
        ax.scatter(
            [float(row["aligned_reference_energy_ev"]) for row in vir_rows],
            [float(row["aligned_predicted_energy_ev"]) for row in vir_rows],
            s=34,
            alpha=0.80,
            marker="^",
            c="#ff7f0e",
            label="Virtual",
        )
    for row in rows:
        ax.annotate(
            str(row["label"]),
            (
                float(row["aligned_reference_energy_ev"]),
                float(row["aligned_predicted_energy_ev"]),
            ),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7.2,
            alpha=0.82,
        )
    ax.plot([lo, hi], [lo, hi], "--", color="#555555", linewidth=1.1, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, alpha=0.28)
    ax.set_xlabel("Reference aligned orbital energy (eV)")
    ax.set_ylabel("Predicted aligned orbital energy (eV)")
    window_desc = f"HOMO-{window}...LUMO+{window}" if int(window) > 0 else "All orbitals"
    ax.set_title(
        f"Benzene midpoint-aligned frontier orbital parity ({window_desc})\n"
        f"Aligned MAE={aligned_mae_ev:.4f} eV | Raw MAE={raw_mae_ev:.4f} eV"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=190)
    plt.close(fig)
    return png_path, csv_path, aligned_mae_ev, raw_mae_ev


def main() -> None:
    args = parse_args()
    captured_mf: dict[str, dft.rks.RKS] = {}

    def _mf_builder():
        mf = make_benzene_mf(
            basis=args.basis,
            xc=args.xc,
            grids_level=args.grids_level,
        )
        captured_mf["mf"] = mf
        return mf

    coefficient_prior_values = (
        resolve_coefficient_prior_values(
            DM21_B3LYP_NEURAL_XC_PRESET.semilocal_xc,
            args.coefficient_prior_values,
        )
        if (
            args.coefficient_prior_values is not None
            or float(args.coefficient_prior_weight) != 0.0
        )
        else None
    )
    system_label = f"Benzene (C6H6), {args.xc.upper()}/{args.basis.upper()}"
    run = run_and_report(
        system_label=system_label,
        mf_builder=_mf_builder,
        training_config=NeuralXCTrainingConfig(
            steps=args.steps,
            learning_rate=args.lr,
            lr_decay_every=args.lr_decay_every,
            lr_decay_factor=args.lr_decay_factor,
            jit_train=not args.no_jit,
            log_interval=args.log_interval,
            density_constraint_weight=args.density_weight,
            dm21_scf_regularization_weight=args.dm21_scf_weight,
            dm21_scf_gap_floor=args.dm21_scf_gap_floor,
            orbital_energy_constraint_weight=args.orbital_energy_loss_weight,
            orbital_energy_constraint_window=args.orbital_energy_window,
            density_supervision=args.density_supervision,
            coefficient_prior_weight=args.coefficient_prior_weight,
            coefficient_prior_values=coefficient_prior_values,
            coefficient_prior_mode=args.coefficient_prior_mode,
            orbital_energy_mse_weight=0.0,
            orbital_energy_mae_weight=args.orbital_energy_loss_mae_weight,
            excitation_constraint_weight=args.excitation_weight,
            excitation_constraint_nstates=args.excitation_nstates,
            excitation_constraint_use_tda=bool(args.excitation_use_tda),
            excitation_mse_weight=args.excitation_loss_mse_weight,
            excitation_mae_weight=args.excitation_loss_mae_weight,
            oscillator_strength_constraint_weight=args.oscillator_strength_weight,
            oscillator_strength_constraint_nstates=args.oscillator_strength_nstates,
            oscillator_strength_constraint_use_tda=bool(
                args.oscillator_strength_use_tda
            ),
            oscillator_strength_mse_weight=args.oscillator_strength_loss_mse_weight,
            oscillator_strength_mae_weight=args.oscillator_strength_loss_mae_weight,
            spectrum_constraint_weight=args.spectrum_weight,
            spectrum_constraint_nstates=args.spectrum_nstates,
            spectrum_constraint_use_tda=bool(args.spectrum_use_tda),
            spectrum_mse_weight=args.spectrum_loss_mse_weight,
            spectrum_mae_weight=args.spectrum_loss_mae_weight,
            energy_normalization="none",
            seed=args.seed,
            hidden_dims=tuple(args.hidden_dims),
            network_architecture=args.network_architecture,
            input_feature_mode="dm21_original",
            semilocal_xc=DM21_B3LYP_NEURAL_XC_PRESET.semilocal_xc,
            hf_input_mode=DM21_B3LYP_NEURAL_XC_PRESET.hf_input_mode,
            response_hf_mode=DM21_B3LYP_NEURAL_XC_PRESET.response_hf_mode,
            coefficient_positivity=DM21_B3LYP_NEURAL_XC_PRESET.coefficient_positivity,
            functional_name="benzene_neural_xc_fit",
            strict_graddft_ground_state=bool(args.strict_graddft_ground_state),
        ),
        simulation_config=SimulationConfig(
            nstates=args.states,
            jit_tddft=False,
        ),
        spectrum_config=SpectrumGridConfig(
            eta_ev=args.eta_ev,
            grid_min_ev=0.0,
            grid_points=3500,
            max_padding_ev=2.0,
            zoom_min_ev=3.0,
            zoom_max_ev=12.0,
            compare_states=20,
        ),
        output_config=OutputConfig(
            outdir=Path("outputs"),
            prefix=args.prefix,
            title="Benzene Absorption Spectrum: B3LYP vs Neural_xc",
            reference_label=f"PySCF TDDFT {args.xc.upper()}/{args.basis.upper()}",
            neural_label_template="JAX libxc + Neural_xc TDDFT ({solver})",
            write_training_curves=True,
            training_prefix=f"{args.prefix}_training_curve",
        ),
        print_all_states=True,
    )
    mf = captured_mf.get("mf")
    if mf is None:
        raise RuntimeError("Benzene PySCF reference object was not captured for orbital rendering.")

    scf = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            max_cycle=args.orbital_scf_max_cycle,
            damping=args.orbital_scf_damping,
            conv_tol_density=args.orbital_scf_conv_tol_density,
            vxc_clip=args.orbital_scf_vxc_clip,
            iterate_selection=args.orbital_scf_iterate_selection,
        )
    )
    neural_scf_molecule, scf_info = scf.run(
        run.reference.molecule,
        run.training.functional,
        run.training.params,
    )

    orbital_outdir = Path("outputs") / f"{args.prefix}_orbitals"
    if not args.skip_orbital_energy_compare:
        frontier_rows = _frontier_orbital_energy_rows(
            reference_mo_energy=mf.mo_energy,
            reference_mo_occ=mf.mo_occ,
            predicted_mo_energy=neural_scf_molecule.mo_energy,
            predicted_mo_occ=neural_scf_molecule.mo_occ,
            window=args.orbital_energy_window,
        )
        if frontier_rows:
            (
                orbital_energy_png,
                orbital_energy_csv,
                orbital_energy_aligned_mae,
                orbital_energy_raw_mae,
            ) = _write_frontier_orbital_energy_parity(
                frontier_rows,
                outdir=orbital_outdir,
                window=args.orbital_energy_window,
            )
            energy_summary = orbital_outdir / "orbital_energy_summary.txt"
            with energy_summary.open("w", encoding="utf-8") as handle:
                handle.write("system=Benzene\n")
                handle.write(f"basis={args.basis}\n")
                handle.write(f"xc={args.xc}\n")
                handle.write(f"prefix={args.prefix}\n")
                handle.write(f"orbital_energy_window={int(args.orbital_energy_window)}\n")
                handle.write(f"orbital_energy_count={len(frontier_rows)}\n")
                handle.write(
                    f"orbital_energy_aligned_mae_ev={orbital_energy_aligned_mae:.10f}\n"
                )
                handle.write(f"orbital_energy_raw_mae_ev={orbital_energy_raw_mae:.10f}\n")
                handle.write(
                    f"orbital_energy_mae_ev={orbital_energy_aligned_mae:.10f}\n"
                )
                handle.write(f"orbital_energy_parity_png={orbital_energy_png}\n")
                handle.write(f"orbital_energy_parity_csv={orbital_energy_csv}\n")
            print(f"orbital_energy_parity_png={orbital_energy_png}")
            print(f"orbital_energy_parity_csv={orbital_energy_csv}")
            print(f"orbital_energy_summary={energy_summary}")

    if args.skip_orbital_compare:
        return

    (
        orbital_images,
        orbital_diff_norms,
        orbital_overlaps,
        orbital_diff_isos,
        orbital_diff_scales,
    ) = render_restricted_orbital_surfaces(
        reference_mol=mf.mol,
        reference_mo_coeff=mf.mo_coeff,
        reference_mo_occ=mf.mo_occ,
        neural_molecule=neural_scf_molecule,
        overlap_matrix=run.reference.molecule.overlap_matrix,
        xyzrender_src=args.xyzrender_src,
        outdir=orbital_outdir,
        iso=args.orbital_iso,
        diff_iso=args.orbital_diff_iso,
        mo_blur=args.orbital_mo_blur,
        mo_upsample=args.orbital_mo_upsample,
        cube_grid=args.orbital_cube_grid,
        canvas_size=args.orbital_canvas_size,
        match_frontier_by_overlap=not args.disable_orbital_frontier_match,
        frontier_match_window=args.orbital_frontier_match_window,
    )
    compare_dir = orbital_outdir / "orbital_surfaces" / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    for label, files in orbital_images.items():
        plot_orbital_compare_panel(
            orbital_label=label,
            ref_png=files["reference"],
            neural_png=files["neural"],
            diff_png=files["difference"],
            iso=args.orbital_iso,
            diff_iso=orbital_diff_isos[label],
            overlap_val=orbital_overlaps[label],
            diff_norm=orbital_diff_norms[label],
            diff_scale=orbital_diff_scales[label],
            out_png=compare_dir / f"{label.replace('+', 'p').replace('-', 'm').lower()}_real_vs_neural.png",
        )

    summary_path = orbital_outdir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("system=Benzene\n")
        handle.write(f"basis={args.basis}\n")
        handle.write(f"xc={args.xc}\n")
        handle.write(f"prefix={args.prefix}\n")
        handle.write(f"orbital_scf_converged={bool(scf_info.converged)}\n")
        handle.write(f"orbital_scf_cycles={int(scf_info.cycles)}\n")
        handle.write(f"orbital_scf_selected_cycle={int(scf_info.selected_cycle)}\n")
        handle.write(
            "orbital_scf_selected_rms_density="
            f"{float(scf_info.selected_rms_density):.6e}\n"
        )
        handle.write(
            f"orbital_frontier_match={not bool(args.disable_orbital_frontier_match)}\n"
        )
        handle.write(
            f"orbital_frontier_match_window={int(args.orbital_frontier_match_window)}\n"
        )
        for label in ("HOMO-1", "HOMO", "LUMO", "LUMO+1"):
            handle.write(f"{label}_overlap={orbital_overlaps[label]:.8f}\n")
            handle.write(f"{label}_diff_norm={orbital_diff_norms[label]:.8f}\n")
            handle.write(f"{label}_diff_iso={orbital_diff_isos[label]:.8f}\n")
            handle.write(f"{label}_diff_scale={orbital_diff_scales[label]:.8f}\n")

    print(f"orbital_compare_dir={compare_dir}")
    print(f"orbital_summary={summary_path}")


if __name__ == "__main__":
    main()
