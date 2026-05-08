from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pyscf import ao2mo, dft, fci, gto, scf

HARTREE_TO_EV = 27.211386245988


@dataclass(frozen=True)
class CurvePoint:
    r_angstrom: float
    wb97mv_ground_h: float
    wb97mv_excited_h: float
    wb97mv_excitation_h: float
    fci_ground_h: float
    fci_excited_h: float
    fci_excitation_h: float
    rks_time_s: float
    tddft_time_s: float
    rhf_time_s: float
    fci_time_s: float


def build_h2(r_angstrom: float, basis: str):
    return gto.M(
        atom=f"H 0 0 {-0.5 * r_angstrom}; H 0 0 {0.5 * r_angstrom}",
        unit="Angstrom",
        basis=basis,
        spin=0,
        charge=0,
        verbose=0,
    )


def build_wb97mv_rks(mol, *, xc: str):
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.conv_tol = 1e-11
    mf.max_cycle = 100
    mf.grids.atom_grid = {"H": (99, 590)}
    mf.grids.prune = None
    if mf.nlc != "":
        mf.nlcgrids.atom_grid = {"H": (75, 302)}
        mf.nlcgrids.prune = None
    return mf


def solve_wb97mv_point(mol, *, xc: str, dm0=None) -> tuple[float, float, np.ndarray, float, float]:
    mf = build_wb97mv_rks(mol, xc=xc)
    t0 = time.perf_counter()
    e0 = float(mf.kernel(dm0=dm0))
    rks_elapsed = time.perf_counter() - t0
    if not mf.converged:
        bond_bohr = float(mol.atom_coord(1)[2] - mol.atom_coord(0)[2])
        bond_angstrom = bond_bohr * 0.529177210903
        raise RuntimeError(f"RKS did not converge at R = {bond_angstrom:.6f} Angstrom")

    td = mf.TDDFT()
    td.singlet = True
    td.nstates = 3
    td.conv_tol = 1e-7
    t1 = time.perf_counter()
    exc, _ = td.kernel()
    tddft_elapsed = time.perf_counter() - t1
    if len(exc) < 1:
        raise RuntimeError("PySCF TDDFT did not return a singlet excited state.")

    s1_excitation = float(exc[0])
    total_s1 = e0 + s1_excitation
    return e0, total_s1, mf.make_rdm1(), rks_elapsed, tddft_elapsed


def solve_fci_point(mol, *, dm0=None) -> tuple[float, float, np.ndarray, float, float]:
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 100
    t0 = time.perf_counter()
    mf.kernel(dm0=dm0)
    rhf_elapsed = time.perf_counter() - t0
    if not mf.converged:
        raise RuntimeError("RHF did not converge for FCI reference.")

    h1_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri_mo = ao2mo.kernel(mol, mf.mo_coeff)
    cisolver = fci.direct_spin0.FCI(mol)
    t1 = time.perf_counter()
    roots, _ = cisolver.kernel(h1_mo, eri_mo, h1_mo.shape[0], mol.nelectron, nroots=2)
    fci_elapsed = time.perf_counter() - t1
    if len(roots) < 2:
        raise RuntimeError("FCI did not return two singlet roots.")

    enuc = float(mol.energy_nuc())
    e0 = float(roots[0] + enuc)
    e1 = float(roots[1] + enuc)
    return e0, e1, mf.make_rdm1(), rhf_elapsed, fci_elapsed


def write_csv(path: Path, curve: list[CurvePoint]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "R_Angstrom",
                "E0_wB97M-V_Hartree",
                "E1_wB97M-V_TDDFT_Hartree",
                "Excitation_wB97M-V_eV",
                "E0_FCI_Hartree",
                "E1_FCI_Hartree",
                "Excitation_FCI_eV",
                "Ground_AbsErr_eV",
                "Excited_AbsErr_eV",
                "RKS_Time_s",
                "TDDFT_Time_s",
                "RHF_Time_s",
                "FCI_Time_s",
            ]
        )
        for point in curve:
            writer.writerow(
                [
                    point.r_angstrom,
                    point.wb97mv_ground_h,
                    point.wb97mv_excited_h,
                    point.wb97mv_excitation_h * HARTREE_TO_EV,
                    point.fci_ground_h,
                    point.fci_excited_h,
                    point.fci_excitation_h * HARTREE_TO_EV,
                    abs(point.wb97mv_ground_h - point.fci_ground_h) * HARTREE_TO_EV,
                    abs(point.wb97mv_excited_h - point.fci_excited_h) * HARTREE_TO_EV,
                    point.rks_time_s,
                    point.tddft_time_s,
                    point.rhf_time_s,
                    point.fci_time_s,
                ]
            )


def plot_curves(path: Path, curve: list[CurvePoint], *, xc: str, basis: str) -> None:
    r = np.asarray([point.r_angstrom for point in curve])
    wb97mv_ground = np.asarray([point.wb97mv_ground_h for point in curve])
    wb97mv_excited = np.asarray([point.wb97mv_excited_h for point in curve])
    fci_ground = np.asarray([point.fci_ground_h for point in curve])
    fci_excited = np.asarray([point.fci_excited_h for point in curve])
    wb97mv_gap = np.asarray([point.wb97mv_excitation_h for point in curve]) * HARTREE_TO_EV
    fci_gap = np.asarray([point.fci_excitation_h for point in curve]) * HARTREE_TO_EV

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))

    ax = axes[0, 0]
    ax.plot(r, wb97mv_ground, lw=2.0, label=f"{xc} ground")
    ax.plot(r, fci_ground, lw=2.0, label="FCI ground")
    ax.set_xlabel("H-H distance (Angstrom)")
    ax.set_ylabel("Total energy (Hartree)")
    ax.set_title("Ground-State Dissociation")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(r, wb97mv_excited, lw=2.0, label=f"{xc} S1")
    ax.plot(r, fci_excited, lw=2.0, label="FCI S1")
    ax.set_xlabel("H-H distance (Angstrom)")
    ax.set_ylabel("Total energy (Hartree)")
    ax.set_title("First Singlet Excited State")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(r, wb97mv_gap, lw=2.0, label=f"{xc} TDDFT gap")
    ax.plot(r, fci_gap, lw=2.0, label="FCI gap")
    ax.set_xlabel("H-H distance (Angstrom)")
    ax.set_ylabel("Excitation energy (eV)")
    ax.set_title("Lowest Singlet Excitation")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(r, np.abs(wb97mv_ground - fci_ground) * HARTREE_TO_EV, lw=2.0, label="Ground abs err")
    ax.plot(r, np.abs(wb97mv_excited - fci_excited) * HARTREE_TO_EV, lw=2.0, label="S1 abs err")
    ax.set_xlabel("H-H distance (Angstrom)")
    ax.set_ylabel("Absolute error vs FCI (eV)")
    ax.set_title("Deviation from FCI")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.suptitle(f"H2 dissociation curves | {xc}/{basis}", y=0.98)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_summary(path: Path, curve: list[CurvePoint], *, xc: str, basis: str, elapsed_s: float) -> None:
    wb97mv_ground = np.asarray([point.wb97mv_ground_h for point in curve])
    wb97mv_excited = np.asarray([point.wb97mv_excited_h for point in curve])
    fci_ground = np.asarray([point.fci_ground_h for point in curve])
    fci_excited = np.asarray([point.fci_excited_h for point in curve])
    wb97mv_gap_ev = np.asarray([point.wb97mv_excitation_h for point in curve]) * HARTREE_TO_EV
    fci_gap_ev = np.asarray([point.fci_excitation_h for point in curve]) * HARTREE_TO_EV

    ground_abs_err_ev = np.abs(wb97mv_ground - fci_ground) * HARTREE_TO_EV
    excited_abs_err_ev = np.abs(wb97mv_excited - fci_excited) * HARTREE_TO_EV
    gap_abs_err_ev = np.abs(wb97mv_gap_ev - fci_gap_ev)

    with path.open("w", encoding="utf-8") as handle:
        handle.write("H2 dissociation benchmark\n")
        handle.write(f"xc = {xc}\n")
        handle.write(f"basis = {basis}\n")
        handle.write(f"points = {len(curve)}\n")
        handle.write(f"R_min = {curve[0].r_angstrom:.6f} Angstrom\n")
        handle.write(f"R_max = {curve[-1].r_angstrom:.6f} Angstrom\n")
        handle.write(f"wall_time_s = {elapsed_s:.2f}\n")
        handle.write(f"ground_mae_ev = {ground_abs_err_ev.mean():.6f}\n")
        handle.write(f"ground_max_abs_err_ev = {ground_abs_err_ev.max():.6f}\n")
        handle.write(f"excited_mae_ev = {excited_abs_err_ev.mean():.6f}\n")
        handle.write(f"excited_max_abs_err_ev = {excited_abs_err_ev.max():.6f}\n")
        handle.write(f"gap_mae_ev = {gap_abs_err_ev.mean():.6f}\n")
        handle.write(f"gap_max_abs_err_ev = {gap_abs_err_ev.max():.6f}\n")
        handle.write(f"rks_total_time_s = {sum(point.rks_time_s for point in curve):.2f}\n")
        handle.write(f"tddft_total_time_s = {sum(point.tddft_time_s for point in curve):.2f}\n")
        handle.write(f"rhf_total_time_s = {sum(point.rhf_time_s for point in curve):.2f}\n")
        handle.write(f"fci_total_time_s = {sum(point.fci_time_s for point in curve):.2f}\n")
        handle.write("\n")
        handle.write("Important caveat\n")
        handle.write(
            "PySCF prints that for wB97M-V the VV10/NLC second derivative is unavailable, "
            "so the TDDFT response omits the NLC contribution. The ground-state energies are "
            "wB97M-V total energies, but the excited-state curve is PySCF's current TDDFT "
            "implementation on top of those orbitals rather than a full wB97M-V linear response.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute H2 wB97M-V ground-state and S1 dissociation curves against FCI."
    )
    parser.add_argument("--basis", type=str, default="6-31g")
    parser.add_argument("--xc", type=str, default="wB97M-V")
    parser.add_argument("--r-min", type=float, default=0.45)
    parser.add_argument("--r-max", type=float, default=5.00)
    parser.add_argument("--points", type=int, default=61)
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/h2_wb97mv_631g_vs_fci_dissociation",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    r_grid = np.linspace(args.r_min, args.r_max, args.points)
    curve: list[CurvePoint] = []
    prev_rks_dm = None
    prev_rhf_dm = None

    wall_start = time.perf_counter()
    for idx, r_angstrom in enumerate(r_grid, start=1):
        mol = build_h2(float(r_angstrom), args.basis)
        wb97mv_ground, wb97mv_excited, prev_rks_dm, rks_time_s, tddft_time_s = solve_wb97mv_point(
            mol,
            xc=args.xc,
            dm0=prev_rks_dm,
        )
        fci_ground, fci_excited, prev_rhf_dm, rhf_time_s, fci_time_s = solve_fci_point(
            mol,
            dm0=prev_rhf_dm,
        )
        point = CurvePoint(
            r_angstrom=float(r_angstrom),
            wb97mv_ground_h=wb97mv_ground,
            wb97mv_excited_h=wb97mv_excited,
            wb97mv_excitation_h=wb97mv_excited - wb97mv_ground,
            fci_ground_h=fci_ground,
            fci_excited_h=fci_excited,
            fci_excitation_h=fci_excited - fci_ground,
            rks_time_s=rks_time_s,
            tddft_time_s=tddft_time_s,
            rhf_time_s=rhf_time_s,
            fci_time_s=fci_time_s,
        )
        curve.append(point)
        print(
            f"[{idx:03d}/{len(r_grid):03d}] "
            f"R={point.r_angstrom:.3f} A "
            f"E0({args.xc})={point.wb97mv_ground_h:.10f} "
            f"E1({args.xc})={point.wb97mv_excited_h:.10f} "
            f"E0(FCI)={point.fci_ground_h:.10f} "
            f"E1(FCI)={point.fci_excited_h:.10f}"
        )

    elapsed_s = time.perf_counter() - wall_start

    csv_path = outdir / "h2_wb97mv_vs_fci_curve.csv"
    png_path = outdir / "h2_wb97mv_vs_fci_curve.png"
    summary_path = outdir / "summary.txt"

    write_csv(csv_path, curve)
    plot_curves(png_path, curve, xc=args.xc, basis=args.basis)
    write_summary(summary_path, curve, xc=args.xc, basis=args.basis, elapsed_s=elapsed_s)

    print(f"Wrote csv: {csv_path}")
    print(f"Wrote plot: {png_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
