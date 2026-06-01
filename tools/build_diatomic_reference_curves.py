from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Iterable

import numpy as np
from pyscf import ao2mo, fci, gto, scf, tdscf

HARTREE_TO_EV = 27.211386245988


@dataclass(frozen=True)
class DiatomicSpec:
    name: str
    atom1: str
    atom2: str
    charge: int
    spin: int
    r_min: float
    r_max: float


SYSTEMS: dict[str, DiatomicSpec] = {
    "H2+": DiatomicSpec("H2+", "H", "H", 1, 1, 0.4, 6.0),
    "H2": DiatomicSpec("H2", "H", "H", 0, 0, 0.4, 5.0),
    "LiH": DiatomicSpec("LiH", "Li", "H", 0, 0, 0.9, 6.0),
    "BH": DiatomicSpec("BH", "B", "H", 0, 0, 0.8, 5.0),
    "HF": DiatomicSpec("HF", "H", "F", 0, 0, 0.6, 4.5),
    "F2": DiatomicSpec("F2", "F", "F", 0, 0, 0.9, 4.5),
    "N2": DiatomicSpec("N2", "N", "N", 0, 0, 0.7, 3.5),
    "C2": DiatomicSpec("C2", "C", "C", 0, 0, 0.8, 3.5),
}

PRESETS: dict[str, tuple[str, ...]] = {
    "graddft": ("H2+", "H2", "N2"),
    "small_fci": ("H2+", "H2", "LiH", "BH"),
    "first_wave": ("H2+", "H2", "LiH", "BH", "HF", "F2", "N2", "C2"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build ground/excited-state diatomic reference curves with a common CSV schema. "
            "FCI is intended for small active spaces; TDHF/TDDFT modes are for larger systems."
        )
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["graddft"],
        help=(
            "System names or presets. Presets: "
            + ", ".join(sorted(PRESETS))
            + ". Systems: "
            + ", ".join(sorted(SYSTEMS))
        ),
    )
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument(
        "--method",
        choices=("fci", "tda", "tddft"),
        default="fci",
        help="Excited-state reference method. tda/tddft use PySCF TD response on an SCF/DFT reference.",
    )
    parser.add_argument(
        "--xc",
        default=None,
        help="If set with --method tda/tddft, run RKS/UKS with this XC instead of HF response.",
    )
    parser.add_argument("--points", type=int, default=25)
    parser.add_argument("--r-min", type=float, default=None, help="Override preset minimum R in Angstrom.")
    parser.add_argument("--r-max", type=float, default=None, help="Override preset maximum R in Angstrom.")
    parser.add_argument("--nroots", type=int, default=4, help="Number of FCI roots or TD roots.")
    parser.add_argument(
        "--fci-spin-sector",
        choices=("auto", "singlet", "all"),
        default="auto",
        help="For FCI: auto uses singlet roots for closed-shell singlets and all-spin roots otherwise.",
    )
    parser.add_argument(
        "--max-fci-determinants",
        type=int,
        default=200000,
        help="Skip FCI points whose alpha*beta determinant count exceeds this threshold.",
    )
    parser.add_argument("--scf-conv-tol", type=float, default=1e-10)
    parser.add_argument("--scf-max-cycle", type=int, default=120)
    parser.add_argument(
        "--outdir",
        default="outputs/diatomic_reference_curves",
        help="Output directory containing states.csv, points.csv, manifest.json.",
    )
    return parser.parse_args()


def _expand_systems(tokens: Iterable[str]) -> list[DiatomicSpec]:
    names: list[str] = []
    for token in tokens:
        key = str(token).strip()
        if key in PRESETS:
            names.extend(PRESETS[key])
        elif key in SYSTEMS:
            names.append(key)
        else:
            raise ValueError(f"Unknown system/preset {key!r}.")
    deduped: list[str] = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return [SYSTEMS[name] for name in deduped]


def _build_mol(spec: DiatomicSpec, r_angstrom: float, basis: str) -> gto.Mole:
    mol = gto.Mole()
    mol.atom = f"""
    {spec.atom1} 0.0000000000 0.0000000000 {-0.5 * r_angstrom:.12f}
    {spec.atom2} 0.0000000000 0.0000000000 {+0.5 * r_angstrom:.12f}
    """
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.charge = int(spec.charge)
    mol.spin = int(spec.spin)
    mol.cart = True
    mol.verbose = 0
    mol.build()
    return mol


def _make_scf(mol: gto.Mole, *, xc: str | None, conv_tol: float, max_cycle: int):
    is_open_shell = int(mol.spin) != 0
    if xc:
        mf = scf.UKS(mol) if is_open_shell else scf.RKS(mol)
        mf.xc = str(xc)
    else:
        mf = scf.UHF(mol) if is_open_shell else scf.RHF(mol)
    mf.conv_tol = float(conv_tol)
    mf.max_cycle = int(max_cycle)
    return mf


def _run_scf(mol: gto.Mole, *, xc: str | None, conv_tol: float, max_cycle: int):
    attempts = (
        {"init_guess": "minao", "damping": 0.0, "level_shift": 0.0},
        {"init_guess": "atom", "damping": 0.3, "level_shift": 0.5},
        {"init_guess": "atom", "damping": 0.0, "level_shift": 0.0},
    )
    last_mf = None
    for attempt in attempts:
        mf = _make_scf(mol, xc=xc, conv_tol=conv_tol, max_cycle=max_cycle)
        mf.init_guess = attempt["init_guess"]
        mf.damp = float(attempt["damping"])
        mf.level_shift = float(attempt["level_shift"])
        mf.kernel()
        last_mf = mf
        if bool(mf.converged):
            return mf
    assert last_mf is not None
    return last_mf


def _scf_cycle_count(mf) -> float:
    cycles = getattr(mf, "cycles", None)
    if cycles is None:
        return math.nan
    try:
        return float(cycles)
    except (TypeError, ValueError):
        return math.nan


def _fci_dimension(mol: gto.Mole, norb: int) -> int:
    nalpha, nbeta = mol.nelec
    return int(fci.cistring.num_strings(norb, nalpha) * fci.cistring.num_strings(norb, nbeta))


def _canonical_mo_coeff(mf) -> np.ndarray:
    coeff = mf.mo_coeff
    if isinstance(coeff, (tuple, list)):
        # For ROHF/UHF references, direct_spin1 FCI expects one spatial orbital basis.
        # Use the alpha orbital basis as a deterministic reference for small open-shell cases.
        return np.asarray(coeff[0], dtype=np.float64)
    arr = np.asarray(coeff, dtype=np.float64)
    if arr.ndim == 3:
        return arr[0]
    return arr


def _solve_fci_roots(
    mol: gto.Mole,
    mf,
    *,
    nroots: int,
    max_determinants: int,
    spin_sector: str,
) -> dict[str, object]:
    mo_coeff = _canonical_mo_coeff(mf)
    norb = int(mo_coeff.shape[1])
    ndet = _fci_dimension(mol, norb)
    if ndet > int(max_determinants):
        return {
            "skipped": True,
            "skip_reason": f"fci_determinants={ndet} exceeds max_fci_determinants={max_determinants}",
            "norb": norb,
            "determinants": ndet,
            "energies_elec_h": [],
            "spin_square": [],
        }

    h1_mo = mo_coeff.T @ np.asarray(mf.get_hcore(), dtype=np.float64) @ mo_coeff
    eri_mo = ao2mo.kernel(mol, mo_coeff)
    sector = str(spin_sector)
    if sector == "auto":
        sector = "singlet" if int(mol.spin) == 0 and (int(mol.nelectron) % 2 == 0) else "all"
    solver = fci.direct_spin0.FCI(mol) if sector == "singlet" else fci.direct_spin1.FCI()
    solver.conv_tol = 1e-12
    solver.max_cycle = 200
    root_count = min(max(1, int(nroots)), max(1, ndet))
    e_roots, ci_roots = solver.kernel(h1_mo, eri_mo, norb, mol.nelec, nroots=root_count)
    e_arr = np.asarray(e_roots, dtype=np.float64).reshape(-1)
    ci_list = ci_roots if isinstance(ci_roots, (list, tuple)) else [ci_roots]
    ss_values = [
        float(fci.spin_op.spin_square0(np.asarray(ci), norb, mol.nelec)[0])
        for ci in ci_list
    ]
    return {
        "skipped": False,
        "skip_reason": "",
        "norb": norb,
        "determinants": ndet,
        "fci_spin_sector": sector,
        "energies_elec_h": e_arr.tolist(),
        "spin_square": ss_values,
    }


def _solve_td_roots(mol: gto.Mole, mf, *, method: str, nroots: int) -> dict[str, object]:
    if str(method) == "tda":
        td = tdscf.TDA(mf)
    elif str(method) == "tddft":
        td = tdscf.TDDFT(mf)
    else:
        raise ValueError(f"Unsupported TD method {method!r}")
    td.nstates = int(nroots)
    td.kernel()
    energies = np.asarray(td.e, dtype=np.float64).reshape(-1)
    return {
        "skipped": False,
        "skip_reason": "",
        "norb": int(mol.nao_nr()),
        "determinants": math.nan,
        "excitation_energies_h": energies.tolist(),
        "spin_square": [math.nan] * int(energies.size),
    }


def _point_rows(
    *,
    spec: DiatomicSpec,
    r_angstrom: float,
    basis: str,
    method: str,
    xc: str | None,
    mf,
    elapsed_s: float,
    result: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    point_row = {
        "system": spec.name,
        "atom1": spec.atom1,
        "atom2": spec.atom2,
        "charge": spec.charge,
        "spin": spec.spin,
        "r_angstrom": float(r_angstrom),
        "basis": str(basis),
        "method": str(method),
        "xc": "" if xc is None else str(xc),
        "scf_energy_h": float(mf.e_tot),
        "scf_converged": bool(mf.converged),
        "scf_cycles": _scf_cycle_count(mf),
        "n_electrons": int(mf.mol.nelectron),
        "n_ao": int(mf.mol.nao_nr()),
        "n_orbitals": result.get("norb", math.nan),
        "fci_determinants": result.get("determinants", math.nan),
        "skipped": bool(result.get("skipped", False)),
        "skip_reason": str(result.get("skip_reason", "")),
        "elapsed_s": float(elapsed_s),
    }

    state_rows: list[dict[str, object]] = []
    if bool(result.get("skipped", False)):
        return point_row, state_rows

    if method == "fci":
        elec_energies = np.asarray(result["energies_elec_h"], dtype=np.float64)
        total_energies = elec_energies + float(mf.mol.energy_nuc())
        ground = float(total_energies[0])
        ss_values = list(result.get("spin_square", []))
        for state_index, total_energy in enumerate(total_energies.tolist()):
            gap_h = float(total_energy - ground)
            state_rows.append(
                {
                    **point_row,
                    "state_index": int(state_index),
                    "state_label": "S0" if state_index == 0 else f"E{state_index}",
                    "total_energy_h": float(total_energy),
                    "excitation_energy_h": gap_h,
                    "excitation_energy_ev": gap_h * HARTREE_TO_EV,
                    "spin_square": ss_values[state_index] if state_index < len(ss_values) else math.nan,
                }
            )
    else:
        excitation_energies = np.asarray(result["excitation_energies_h"], dtype=np.float64)
        ground = float(mf.e_tot)
        state_rows.append(
            {
                **point_row,
                "state_index": 0,
                "state_label": "S0",
                "total_energy_h": ground,
                "excitation_energy_h": 0.0,
                "excitation_energy_ev": 0.0,
                "spin_square": math.nan,
            }
        )
        for idx, gap_h in enumerate(excitation_energies.tolist(), start=1):
            state_rows.append(
                {
                    **point_row,
                    "state_index": int(idx),
                    "state_label": f"E{idx}",
                    "total_energy_h": ground + float(gap_h),
                    "excitation_energy_h": float(gap_h),
                    "excitation_energy_ev": float(gap_h) * HARTREE_TO_EV,
                    "spin_square": math.nan,
                }
            )
    return point_row, state_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    specs = _expand_systems(args.systems)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    point_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    manifest = {
        "systems": [spec.name for spec in specs],
        "basis": str(args.basis),
        "method": str(args.method),
        "xc": None if args.xc is None else str(args.xc),
        "points": int(args.points),
        "nroots": int(args.nroots),
        "max_fci_determinants": int(args.max_fci_determinants),
        "fci_spin_sector": str(args.fci_spin_sector),
        "scf_conv_tol": float(args.scf_conv_tol),
        "scf_max_cycle": int(args.scf_max_cycle),
        "system_ranges_angstrom": {},
        "created_files": ["points.csv", "states.csv", "manifest.json", "visualization_manifest.json"],
    }

    for spec in specs:
        r_min = float(spec.r_min if args.r_min is None else args.r_min)
        r_max = float(spec.r_max if args.r_max is None else args.r_max)
        manifest["system_ranges_angstrom"][spec.name] = {"r_min": r_min, "r_max": r_max}
        for r_angstrom in np.linspace(r_min, r_max, int(args.points)):
            t0 = time.perf_counter()
            mol = _build_mol(spec, float(r_angstrom), str(args.basis))
            mf = _run_scf(
                mol,
                xc=args.xc if str(args.method) in {"tda", "tddft"} else None,
                conv_tol=float(args.scf_conv_tol),
                max_cycle=int(args.scf_max_cycle),
            )
            try:
                if str(args.method) == "fci":
                    result = _solve_fci_roots(
                        mol,
                        mf,
                        nroots=int(args.nroots),
                        max_determinants=int(args.max_fci_determinants),
                        spin_sector=str(args.fci_spin_sector),
                    )
                else:
                    result = _solve_td_roots(
                        mol,
                        mf,
                        method=str(args.method),
                        nroots=int(args.nroots),
                    )
            except Exception as exc:
                result = {
                    "skipped": True,
                    "skip_reason": f"{type(exc).__name__}: {exc}",
                    "norb": int(mol.nao_nr()),
                    "determinants": math.nan,
                }
            elapsed_s = time.perf_counter() - t0
            point_row, rows = _point_rows(
                spec=spec,
                r_angstrom=float(r_angstrom),
                basis=str(args.basis),
                method=str(args.method),
                xc=args.xc,
                mf=mf,
                elapsed_s=elapsed_s,
                result=result,
            )
            point_rows.append(point_row)
            state_rows.extend(rows)
            status = "skipped" if point_row["skipped"] else f"{len(rows)} states"
            print(
                f"[{spec.name}] R={float(r_angstrom):.6f} A {status} "
                f"scf_converged={bool(point_row['scf_converged'])} elapsed_s={elapsed_s:.2f}",
                flush=True,
            )

    _write_csv(outdir / "points.csv", point_rows)
    _write_csv(outdir / "states.csv", state_rows)
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    visualization_manifest = {
        "paper_experiment": "Ground-State Potential-Energy Surfaces",
        "description": "Raw tabular data for diatomic reference PES visualizations.",
        "figures": [
            {
                "name": "ground_and_excited_state_reference_curves",
                "data_files": ["states.csv"],
                "x": "r_angstrom",
                "y": ["total_energy_h", "excitation_energy_ev"],
                "group_by": ["system", "state_label"],
            },
            {
                "name": "scf_convergence_by_geometry",
                "data_files": ["points.csv"],
                "x": "r_angstrom",
                "y": ["scf_converged", "scf_cycles"],
                "group_by": ["system"],
            },
        ],
        "metadata_files": ["manifest.json"],
    }
    (outdir / "visualization_manifest.json").write_text(
        json.dumps(visualization_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote {outdir / 'points.csv'}", flush=True)
    print(f"Wrote {outdir / 'states.csv'}", flush=True)
    print(f"Wrote {outdir / 'manifest.json'}", flush=True)
    print(f"Wrote {outdir / 'visualization_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
