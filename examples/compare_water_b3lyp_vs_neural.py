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
            "Train Neural_xc on water ground state and compare absorption "
            "spectrum and frontier orbital energies against PySCF B3LYP."
        )
    )
    parser.add_argument("--steps", type=int, default=200, help="training steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
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
        help="disable JIT for the training loop",
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
        default=0.0,
        help="weight for direct grid-density matching loss",
    )
    parser.add_argument(
        "--orbital-energy-loss-weight",
        type=float,
        default=1.0,
        help="weight for frontier orbital-energy loss during training",
    )
    parser.add_argument(
        "--orbital-energy-loss-mae-weight",
        type=float,
        default=1.0,
        help="MAE coefficient inside the orbital-energy training loss",
    )
    parser.add_argument(
        "--orbital-energy-window",
        type=int,
        default=1,
        help="number of occupied/virtual frontier orbitals to compare",
    )
    parser.add_argument(
        "--fractional-linearity-weight",
        type=float,
        default=0.0,
        help="weight for the finite-difference fractional-charge linearity penalty",
    )
    parser.add_argument(
        "--fractional-linearity-delta",
        type=float,
        default=0.1,
        help="fractional electron step used in the linearity penalty",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed",
    )
    parser.add_argument(
        "--prefix",
        default="water_graddft_mae_orbital_ep200_w1_aligned",
        help="output file prefix under outputs/",
    )
    parser.add_argument(
        "--basis",
        default="sto-3g",
        help="PySCF basis for the water reference",
    )
    parser.add_argument(
        "--xc",
        default="b3lyp",
        help="PySCF XC functional for the reference",
    )
    parser.add_argument(
        "--grids-level",
        type=int,
        default=0,
        help="PySCF numerical grid level",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=(64, 64, 64),
        help="hidden widths for the Neural_xc MLP",
    )
    parser.add_argument(
        "--nstates",
        type=int,
        default=-1,
        help="number of excited states for TDDFT (-1 means full available space)",
    )
    parser.add_argument(
        "--eta-ev",
        type=float,
        default=0.15,
        help="Lorentzian broadening in eV",
    )
    parser.add_argument(
        "--grid-min-ev",
        type=float,
        default=0.0,
        help="minimum spectrum grid energy in eV",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=2400,
        help="number of spectrum grid points",
    )
    parser.add_argument(
        "--zoom-min-ev",
        type=float,
        default=5.0,
        help="low-energy zoom minimum in eV",
    )
    parser.add_argument(
        "--zoom-max-ev",
        type=float,
        default=20.0,
        help="low-energy zoom maximum in eV",
    )
    parser.add_argument(
        "--compare-states",
        type=int,
        default=8,
        help="number of low-lying states to compare in the report",
    )
    parser.add_argument(
        "--orbital-scf-max-cycle",
        type=int,
        default=80,
        help="SCF max cycles for post-training orbital evaluation",
    )
    parser.add_argument(
        "--orbital-scf-damping",
        type=float,
        default=0.25,
        help="SCF damping for post-training orbital evaluation",
    )
    parser.add_argument(
        "--orbital-scf-conv-tol-density",
        type=float,
        default=1e-8,
        help="density convergence tolerance for post-training orbital evaluation",
    )
    parser.add_argument(
        "--orbital-scf-vxc-clip",
        type=float,
        default=20.0,
        help="v_xc clipping for post-training orbital evaluation",
    )
    parser.add_argument(
        "--orbital-scf-iterate-selection",
        choices=("final", "best_rms", "first_converged"),
        default="final",
        help="which SCF iterate to use for post-training orbital evaluation",
    )
    return parser.parse_args()


def make_water_mf(*, basis: str, xc: str, grids_level: int):
    mol = gto.Mole()
    mol.atom = """
    O  0.000000  0.000000  0.117790
    H  0.000000  0.755453 -0.471161
    H  0.000000 -0.755453 -0.471161
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
        raise RuntimeError("PySCF SCF did not converge for water.")
    return mf


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
) -> tuple[list[dict[str, float | int | str]], float]:
    ref_energies_ha, ref_occ = _restricted_energies_and_occ(reference_mo_energy, reference_mo_occ)
    pred_energies_ha, pred_occ = _restricted_energies_and_occ(predicted_mo_energy, predicted_mo_occ)
    nmo = int(min(ref_energies_ha.size, pred_energies_ha.size, ref_occ.size, pred_occ.size))
    if nmo == 0:
        return [], float("nan")

    occ_idx = np.where(ref_occ[:nmo] > 1e-8)[0]
    vir_idx = np.where(ref_occ[:nmo] <= 1e-8)[0]
    if occ_idx.size == 0 or vir_idx.size == 0:
        return [], float("nan")

    homo_idx = int(occ_idx[-1])
    lumo_idx = int(vir_idx[0])
    ref_zero_ha = 0.5 * float(ref_energies_ha[homo_idx] + ref_energies_ha[lumo_idx])
    pred_zero_ha = 0.5 * float(pred_energies_ha[homo_idx] + pred_energies_ha[lumo_idx])
    ref_zero_ev = ref_zero_ha * HARTREE_TO_EV
    pred_zero_ev = pred_zero_ha * HARTREE_TO_EV
    shift_ev = pred_zero_ev - ref_zero_ev

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
    return rows, shift_ev


def _write_frontier_orbital_energy_parity(
    rows: list[dict[str, float | int | str]],
    *,
    outdir: Path,
    window: int,
) -> tuple[Path, Path, float, float]:
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"frontier_orbital_energy_parity_homo{window}_lumo{window}"
        if int(window) > 0
        else "frontier_orbital_energy_parity_all"
    )
    csv_path = outdir / f"{stem}.csv"
    png_path = outdir / f"{stem}.png"

    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "mo_index,label,orbital_type,reference_midpoint_ha,predicted_midpoint_ha,"
            "reference_midpoint_ev,predicted_midpoint_ev,reference_energy_ha,predicted_energy_ha,"
            "reference_energy_ev,predicted_energy_ev,aligned_reference_energy_ha,"
            "aligned_predicted_energy_ha,aligned_reference_energy_ev,aligned_predicted_energy_ev,"
            "aligned_abs_error_ev,raw_abs_error_ev\n"
        )
        for row in rows:
            handle.write(
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
    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    if occ_rows:
        ax.scatter(
            [float(row["aligned_reference_energy_ev"]) for row in occ_rows],
            [float(row["aligned_predicted_energy_ev"]) for row in occ_rows],
            s=34,
            alpha=0.78,
            c="#1f77b4",
            label="Occupied",
        )
    if vir_rows:
        ax.scatter(
            [float(row["aligned_reference_energy_ev"]) for row in vir_rows],
            [float(row["aligned_predicted_energy_ev"]) for row in vir_rows],
            s=40,
            alpha=0.82,
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
            fontsize=8.0,
            alpha=0.84,
        )
    ax.plot([lo, hi], [lo, hi], "--", color="#555555", linewidth=1.1, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, alpha=0.28)
    ax.set_xlabel("Reference aligned orbital energy (eV)")
    ax.set_ylabel("Predicted aligned orbital energy (eV)")
    window_desc = f"HOMO-{window}...LUMO+{window}" if int(window) > 0 else "All orbitals"
    ax.set_title(
        f"Water midpoint-aligned frontier orbital parity ({window_desc})\n"
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
        mf = make_water_mf(
            basis=args.basis,
            xc=args.xc,
            grids_level=args.grids_level,
        )
        captured_mf["mf"] = mf
        return mf

    run = run_and_report(
        system_label=f"H2O, {args.xc.upper()}/{args.basis.upper()}",
        mf_builder=_mf_builder,
        training_config=NeuralXCTrainingConfig(
            steps=args.steps,
            learning_rate=args.lr,
            lr_decay_every=args.lr_decay_every,
            lr_decay_factor=args.lr_decay_factor,
            jit_train=not args.no_jit,
            log_interval=args.log_interval,
            density_constraint_weight=args.density_weight,
            orbital_energy_constraint_weight=args.orbital_energy_loss_weight,
            orbital_energy_constraint_window=args.orbital_energy_window,
            orbital_energy_mse_weight=0.0,
            orbital_energy_mae_weight=args.orbital_energy_loss_mae_weight,
            fractional_linearity_weight=args.fractional_linearity_weight,
            fractional_linearity_delta=args.fractional_linearity_delta,
            energy_mse_weight=0.0,
            energy_mae_weight=1.0,
            energy_normalization="none",
            seed=args.seed,
            hidden_dims=tuple(args.hidden_dims),
            network_architecture="graddft_residual",
            energy_mode="graddft_coeff_basis",
            input_feature_mode="dm21_original",
            hf_input_mode="spin_resolved",
            response_hf_mode="nonlocal_exchange_only",
            coefficient_positivity="clip",
            density_supervision="spin_resolved",
            strict_dm21_feature_alignment=True,
            functional_name="water_neural_xc_fit",
            strict_graddft_ground_state=False,
        ),
        simulation_config=SimulationConfig(
            nstates=args.nstates,
            jit_tddft=True,
        ),
        spectrum_config=SpectrumGridConfig(
            eta_ev=args.eta_ev,
            grid_min_ev=args.grid_min_ev,
            grid_points=args.grid_points,
            max_padding_ev=2.0,
            zoom_min_ev=args.zoom_min_ev,
            zoom_max_ev=args.zoom_max_ev,
            compare_states=args.compare_states,
        ),
        output_config=OutputConfig(
            outdir=Path("outputs") / args.prefix,
            prefix=args.prefix,
            title="H2O Absorption Spectrum: B3LYP vs Neural_xc",
            reference_label=f"PySCF TDDFT {args.xc.upper()}/{args.basis.upper()}",
            neural_label_template="JAX libxc + Neural_xc TDDFT ({solver})",
            write_training_curves=True,
            training_prefix=f"{args.prefix}_training_curve",
        ),
        print_all_states=True,
    )

    mf = captured_mf.get("mf")
    if mf is None:
        raise RuntimeError("Water PySCF reference object was not captured for orbital evaluation.")

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

    frontier_rows, midpoint_shift_ev = _frontier_orbital_energy_rows(
        reference_mo_energy=mf.mo_energy,
        reference_mo_occ=mf.mo_occ,
        predicted_mo_energy=neural_scf_molecule.mo_energy,
        predicted_mo_occ=neural_scf_molecule.mo_occ,
        window=args.orbital_energy_window,
    )
    if not frontier_rows:
        raise RuntimeError("Failed to build frontier-orbital comparison rows for water.")

    parity_png, parity_csv, aligned_mae_ev, raw_mae_ev = _write_frontier_orbital_energy_parity(
        frontier_rows,
        outdir=Path("outputs") / args.prefix,
        window=args.orbital_energy_window,
    )

    orbital_summary = (Path("outputs") / args.prefix) / "orbital_energy_summary.txt"
    with orbital_summary.open("w", encoding="utf-8") as handle:
        handle.write(f"raw_mae_ev={raw_mae_ev:.10f}\n")
        handle.write(f"aligned_mae_ev={aligned_mae_ev:.10f}\n")
        handle.write(f"midpoint_shift_ev={midpoint_shift_ev:.10f}\n")
        handle.write(f"parity_png={parity_png}\n")
        handle.write(f"parity_csv={parity_csv}\n")

    compare_states = min(4, run.reference.energies_au.size, run.neural.energies_au.size)
    state_mae_ev = (
        float(
            np.mean(
                np.abs(
                    np.asarray(run.reference.energies_au[:compare_states], dtype=float)
                    - np.asarray(run.neural.energies_au[:compare_states], dtype=float)
                )
            )
            * HARTREE_TO_EV
        )
        if compare_states > 0
        else float("nan")
    )

    summary_path = (Path("outputs") / args.prefix) / "run_summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"initial_loss={run.training.initial_loss:.10f}\n")
        handle.write(f"final_loss={run.training.final_loss:.10f}\n")
        handle.write(f"min_loss={run.training.min_loss:.10f}\n")
        handle.write(f"min_loss_step={run.training.min_loss_step}\n")
        handle.write(f"trained_energy_ha={run.training.trained_energy:.10f}\n")
        handle.write(f"solver_label={run.neural.solver_label}\n")
        handle.write(
            f"fractional_linearity_weight={args.fractional_linearity_weight:.10f}\n"
        )
        handle.write(
            f"fractional_linearity_delta={args.fractional_linearity_delta:.10f}\n"
        )
        handle.write(f"first4_state_mae_ev={state_mae_ev:.10f}\n")
        handle.write(f"orbital_scf_converged={bool(getattr(scf_info, 'converged', False))}\n")
        handle.write(f"orbital_scf_cycles={int(getattr(scf_info, 'cycles', -1))}\n")
        handle.write(
            f"orbital_scf_selected_cycle={int(getattr(scf_info, 'selected_cycle', -1))}\n"
        )
        handle.write(
            "orbital_scf_selected_rms_density="
            f"{float(getattr(scf_info, 'selected_rms_density', float('nan'))):.10e}\n"
        )
        handle.write(f"frontier_orbital_mae_raw_ev={raw_mae_ev:.10f}\n")
        handle.write(f"frontier_orbital_mae_aligned_ev={aligned_mae_ev:.10f}\n")
        handle.write(f"frontier_orbital_shift_ev={midpoint_shift_ev:.10f}\n")
        handle.write(f"spectrum_png={run.outputs.spectrum_png}\n")
        handle.write(f"training_png={run.outputs.training_png}\n")
        handle.write(f"orbital_parity_png={parity_png}\n")
        handle.write(f"orbital_parity_csv={parity_csv}\n")
        handle.write(f"orbital_summary={orbital_summary}\n")

    print(f"run_summary={summary_path}")
    print(f"orbital_energy_summary={orbital_summary}")


if __name__ == "__main__":
    main()
