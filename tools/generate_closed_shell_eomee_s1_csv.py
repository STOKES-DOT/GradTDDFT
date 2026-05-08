from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from pyscf import cc, gto, scf

from closed_shell_s1_benchmark_common import closed_shell_s1_spec_map


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"[{_timestamp()}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


FIELDNAMES = [
    "system",
    "split",
    "basis",
    "cart",
    "charge",
    "spin",
    "unit",
    "atom",
    "reference_ground_method",
    "reference_excited_method",
    "rhf_energy_h",
    "ccsd_total_energy_h",
    "s1_excitation_h",
    "s1_excitation_ev",
    "singlet_excitation_energies_h_json",
    "singlet_excitation_energies_ev_json",
    "nroots_requested",
    "scf_elapsed_s",
    "ccsd_elapsed_s",
    "eom_elapsed_s",
    "notes",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate reusable closed-shell EOM-EE-CCSD singlet S1 references "
            "for train/validation/test molecules and save them to CSV."
        )
    )
    p.add_argument("--basis", default="6-31g")
    p.add_argument("--systems", nargs="+", default=None)
    p.add_argument("--include-benzene", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--nroots", type=int, default=3)
    p.add_argument("--scf-conv-tol", type=float, default=1e-10)
    p.add_argument("--scf-max-cycle", type=int, default=120)
    p.add_argument("--cc-conv-tol", type=float, default=1e-8)
    p.add_argument("--cc-max-cycle", type=int, default=200)
    p.add_argument("--outcsv", default="outputs/closed_shell_eomee_s1_631g/closed_shell_s1_references.csv")
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def _run_rhf(
    mol: gto.Mole,
    *,
    scf_conv_tol: float,
    scf_max_cycle: int,
) -> scf.hf.RHF:
    attempts = (
        dict(init_guess="minao", damping=0.0, level_shift=0.0, max_cycle=scf_max_cycle, use_newton=False),
        dict(init_guess="atom", damping=0.3, level_shift=0.5, max_cycle=max(scf_max_cycle, 200), use_newton=False),
        dict(init_guess="atom", damping=0.0, level_shift=0.0, max_cycle=max(scf_max_cycle, 80), use_newton=True),
    )
    last_mf = None
    for cfg in attempts:
        mf = scf.RHF(mol)
        mf.conv_tol = float(scf_conv_tol)
        mf.max_cycle = int(cfg["max_cycle"])
        mf.damping = float(cfg["damping"])
        mf.level_shift = float(cfg["level_shift"])
        mf.diis_start_cycle = 1
        mf.init_guess = str(cfg["init_guess"])
        if cfg["use_newton"]:
            mf = mf.newton()
            mf.conv_tol = float(scf_conv_tol)
            mf.max_cycle = int(cfg["max_cycle"])
        mf.kernel()
        last_mf = mf
        if bool(mf.converged):
            return mf
    raise RuntimeError(f"RHF did not converge for {mol.atom}")


def _compute_reference_row(
    spec: Any,
    *,
    basis: str,
    nroots: int,
    scf_conv_tol: float,
    scf_max_cycle: int,
    cc_conv_tol: float,
    cc_max_cycle: int,
    logger: RunLogger,
) -> dict[str, Any]:
    mol = gto.M(
        atom=str(spec.atom).replace(";", "\n"),
        basis=str(basis),
        unit=str(spec.unit),
        charge=int(spec.charge),
        spin=int(spec.spin),
        cart=True,
        verbose=0,
    )
    logger.log(f"[ref] {spec.name}: RHF/{basis} start")
    t0 = time.perf_counter()
    mf = _run_rhf(mol, scf_conv_tol=scf_conv_tol, scf_max_cycle=scf_max_cycle)
    scf_elapsed = time.perf_counter() - t0

    logger.log(f"[ref] {spec.name}: CCSD start")
    t1 = time.perf_counter()
    mycc = cc.CCSD(mf)
    mycc.conv_tol = float(cc_conv_tol)
    mycc.max_cycle = int(cc_max_cycle)
    mycc.kernel()
    ccsd_elapsed = time.perf_counter() - t1
    if not bool(mycc.converged):
        raise RuntimeError(f"CCSD did not converge for {spec.name}.")

    logger.log(f"[ref] {spec.name}: EOM-EE-CCSD singlet start")
    t2 = time.perf_counter()
    singlet_energies, _ = mycc.eomee_ccsd_singlet(nroots=max(1, int(nroots)))
    eom_elapsed = time.perf_counter() - t2
    singlet = np.sort(np.asarray(singlet_energies, dtype=np.float64).reshape(-1))
    singlet = singlet[np.isfinite(singlet)]
    singlet = singlet[singlet > 1e-7]
    if singlet.size < 1:
        raise RuntimeError(f"No positive singlet EOM-EE-CCSD roots found for {spec.name}.")

    hartree_to_ev = 27.211386245988
    row = {
        "system": str(spec.name),
        "split": str(spec.split),
        "basis": str(basis),
        "cart": True,
        "charge": int(spec.charge),
        "spin": int(spec.spin),
        "unit": str(spec.unit),
        "atom": str(spec.atom),
        "reference_ground_method": "RHF/CCSD",
        "reference_excited_method": "EOM-EE-CCSD singlet",
        "rhf_energy_h": float(mf.e_tot),
        "ccsd_total_energy_h": float(mycc.e_tot),
        "s1_excitation_h": float(singlet[0]),
        "s1_excitation_ev": float(singlet[0] * hartree_to_ev),
        "singlet_excitation_energies_h_json": json.dumps([float(x) for x in singlet.tolist()]),
        "singlet_excitation_energies_ev_json": json.dumps([float(x * hartree_to_ev) for x in singlet.tolist()]),
        "nroots_requested": int(nroots),
        "scf_elapsed_s": float(scf_elapsed),
        "ccsd_elapsed_s": float(ccsd_elapsed),
        "eom_elapsed_s": float(eom_elapsed),
        "notes": spec.notes or "",
    }
    logger.log(
        f"[ref] {spec.name}: done S1={row['s1_excitation_ev']:.6f} eV "
        f"CCSD={row['ccsd_total_energy_h']:.10f} Eh"
    )
    return row


def _load_existing_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {str(row["system"]): row for row in reader}


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    outcsv = Path(args.outcsv)
    logger = RunLogger(outcsv.parent / "generate_references.log")
    spec_map = closed_shell_s1_spec_map(include_benzene=bool(args.include_benzene))
    requested = list(spec_map) if args.systems is None else [str(name) for name in args.systems]

    existing = {} if bool(args.overwrite) else _load_existing_rows(outcsv)
    rows_by_name: dict[str, dict[str, Any]] = {
        name: dict(row) for name, row in existing.items() if name in spec_map
    }

    for name in requested:
        if name not in spec_map:
            raise KeyError(f"Unknown system {name!r}. Available: {sorted(spec_map)}")
        if name in rows_by_name and not bool(args.overwrite):
            logger.log(f"[ref] skip existing {name}")
            continue
        rows_by_name[name] = _compute_reference_row(
            spec_map[name],
            basis=str(args.basis),
            nroots=int(args.nroots),
            scf_conv_tol=float(args.scf_conv_tol),
            scf_max_cycle=int(args.scf_max_cycle),
            cc_conv_tol=float(args.cc_conv_tol),
            cc_max_cycle=int(args.cc_max_cycle),
            logger=logger,
        )
        ordered_rows = [rows_by_name[key] for key in requested if key in rows_by_name]
        _write_rows(outcsv, ordered_rows)
        logger.log(f"[ref] wrote {outcsv}")

    ordered_rows = [rows_by_name[key] for key in requested if key in rows_by_name]
    _write_rows(outcsv, ordered_rows)
    summary = {
        "outcsv": str(outcsv),
        "systems": requested,
        "basis": str(args.basis),
        "nroots": int(args.nroots),
        "count": len(ordered_rows),
    }
    (outcsv.parent / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    logger.log(f"[ref] complete count={len(ordered_rows)}")


if __name__ == "__main__":
    main()

