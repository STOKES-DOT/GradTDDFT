from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from pyscf import cc, gto, scf


ATOMIC_SYMBOL = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
}

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sample QH9 molecules and generate EOM-EE-CCSD singlet S1 references."
    )
    p.add_argument("--db-path", default="/home/yjiao/QH9Stable.db")
    p.add_argument("--sample-count", type=int, default=30)
    p.add_argument("--train-count", type=int, default=20)
    p.add_argument("--validation-count", type=int, default=0)
    p.add_argument("--max-atoms", type=int, default=5)
    p.add_argument(
        "--exclude-elements",
        nargs="*",
        default=(),
        help="Atomic symbols or numbers to exclude from sampled molecules, e.g. F or 9.",
    )
    p.add_argument("--seed", type=int, default=20260527)
    p.add_argument(
        "--sample-order",
        choices=("random", "size"),
        default="random",
        help="Candidate order before taking successful references. 'size' sorts by AO count.",
    )
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--nroots", type=int, default=3)
    p.add_argument("--scf-conv-tol", type=float, default=1e-10)
    p.add_argument("--scf-max-cycle", type=int, default=120)
    p.add_argument("--cc-conv-tol", type=float, default=1e-8)
    p.add_argument("--cc-max-cycle", type=int, default=200)
    p.add_argument("--max-candidates", type=int, default=250)
    p.add_argument("--outcsv", default="outputs/qh9_eomee_s1_sto3g/qh9_30_train20_test10.csv")
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def _parse_excluded_atomic_numbers(values: tuple[str, ...] | list[str]) -> set[int]:
    by_symbol = {symbol.upper(): number for number, symbol in ATOMIC_SYMBOL.items()}
    excluded: set[int] = set()
    for raw in values:
        value = str(raw).strip()
        if not value:
            continue
        if value.isdigit():
            number = int(value)
        else:
            number = by_symbol.get(value.upper())
            if number is None:
                raise ValueError(f"Unsupported element in --exclude-elements: {raw!r}")
        if number not in ATOMIC_SYMBOL:
            raise ValueError(f"Unsupported atomic number in --exclude-elements: {number}")
        excluded.add(number)
    return excluded


def _formula_from_z(z: np.ndarray) -> str:
    counts: dict[str, int] = {}
    for zi in z:
        sym = ATOMIC_SYMBOL[int(zi)]
        counts[sym] = counts.get(sym, 0) + 1
    ordered: list[tuple[str, int]] = []
    if "C" in counts:
        ordered.append(("C", counts.pop("C")))
    if "H" in counts:
        ordered.append(("H", counts.pop("H")))
    for sym in sorted(counts):
        ordered.append((sym, counts[sym]))
    return "".join(sym if n == 1 else f"{sym}{n}" for sym, n in ordered)


def _atom_block(z: np.ndarray, pos_ang: np.ndarray) -> str:
    lines = []
    for zi, xyz in zip(z, pos_ang, strict=True):
        lines.append(f"{ATOMIC_SYMBOL[int(zi)]} {xyz[0]: .12f} {xyz[1]: .12f} {xyz[2]: .12f}")
    return "\n".join(lines)


def _candidate_ids(
    conn: sqlite3.Connection,
    *,
    max_atoms: int,
    excluded_atomic_numbers: set[int] | None = None,
) -> list[int]:
    excluded = set() if excluded_atomic_numbers is None else set(excluded_atomic_numbers)
    ids: list[int] = []
    for db_id, z_blob in conn.execute(
        "SELECT id, Z FROM data WHERE N <= ? ORDER BY id",
        (int(max_atoms),),
    ):
        z = np.frombuffer(z_blob, dtype=np.int32)
        if z.size == 0 or any(int(zi) not in ATOMIC_SYMBOL for zi in z):
            continue
        if excluded and any(int(zi) in excluded for zi in z):
            continue
        if int(np.sum(z)) % 2 == 0:
            ids.append(int(db_id))
    return ids


def _split_for_success_index(
    index: int,
    *,
    train_count: int,
    validation_count: int,
) -> str:
    if int(index) < int(train_count):
        return "train"
    if int(index) < int(train_count) + int(validation_count):
        return "validation"
    return "test"


def _fetch_molecule(conn: sqlite3.Connection, db_id: int) -> tuple[np.ndarray, np.ndarray]:
    row = conn.execute("SELECT N, Z, pos FROM data WHERE id = ?", (int(db_id),)).fetchone()
    if row is None:
        raise KeyError(f"QH9 id {db_id} not found")
    n, z_blob, pos_blob = row
    z = np.frombuffer(z_blob, dtype=np.int32).copy()
    pos = np.frombuffer(pos_blob, dtype=np.float64).reshape(int(n), 3).copy()
    return z, pos


def _candidate_size_key(conn: sqlite3.Connection, *, db_id: int, basis: str) -> tuple[int, int, int, int]:
    z, pos = _fetch_molecule(conn, db_id)
    mol = gto.M(
        atom=_atom_block(z, pos),
        basis=str(basis),
        unit="Angstrom",
        charge=0,
        spin=0,
        cart=True,
        verbose=0,
    )
    return (int(mol.nao_nr()), int(z.size), int(np.sum(z)), int(db_id))


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
    raise RuntimeError(f"RHF did not converge for {mol.atom}; last={last_mf}")


def _compute_row(
    *,
    db_id: int,
    z: np.ndarray,
    pos_ang: np.ndarray,
    split: str,
    basis: str,
    nroots: int,
    scf_conv_tol: float,
    scf_max_cycle: int,
    cc_conv_tol: float,
    cc_max_cycle: int,
    logger: RunLogger,
) -> dict[str, Any]:
    atom = _atom_block(z, pos_ang)
    formula = _formula_from_z(z)
    system = f"qh9_{int(db_id)}_{formula}"
    mol = gto.M(
        atom=atom,
        basis=str(basis),
        unit="Angstrom",
        charge=0,
        spin=0,
        cart=True,
        verbose=0,
    )
    logger.log(f"[ref] {system}: RHF/{basis} start")
    t0 = time.perf_counter()
    mf = _run_rhf(mol, scf_conv_tol=scf_conv_tol, scf_max_cycle=scf_max_cycle)
    scf_elapsed = time.perf_counter() - t0

    logger.log(f"[ref] {system}: CCSD start")
    t1 = time.perf_counter()
    mycc = cc.CCSD(mf)
    mycc.conv_tol = float(cc_conv_tol)
    mycc.max_cycle = int(cc_max_cycle)
    mycc.kernel()
    ccsd_elapsed = time.perf_counter() - t1
    if not bool(mycc.converged):
        raise RuntimeError(f"CCSD did not converge for {system}.")

    logger.log(f"[ref] {system}: EOM-EE-CCSD singlet start")
    t2 = time.perf_counter()
    singlet_energies, _ = mycc.eomee_ccsd_singlet(nroots=max(1, int(nroots)))
    eom_elapsed = time.perf_counter() - t2
    singlet = np.sort(np.asarray(singlet_energies, dtype=np.float64).reshape(-1))
    singlet = singlet[np.isfinite(singlet)]
    singlet = singlet[singlet > 1e-7]
    if singlet.size < 1:
        raise RuntimeError(f"No positive singlet EOM-EE-CCSD roots found for {system}.")

    hartree_to_ev = 27.211386245988
    logger.log(
        f"[ref] {system}: done S1={float(singlet[0] * hartree_to_ev):.6f} eV "
        f"CCSD={float(mycc.e_tot):.10f} Eh"
    )
    return {
        "system": system,
        "split": split,
        "basis": str(basis),
        "cart": True,
        "charge": 0,
        "spin": 0,
        "unit": "Angstrom",
        "atom": atom,
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
        "notes": f"db_id={int(db_id)}; formula={formula}; natoms={int(z.size)}",
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    outcsv = Path(args.outcsv)
    if outcsv.exists() and not bool(args.overwrite):
        raise FileExistsError(f"{outcsv} exists; pass --overwrite to replace it.")
    logger = RunLogger(outcsv.parent / "generate_qh9_eomee_s1.log")
    conn = sqlite3.connect(str(args.db_path))
    rng = np.random.default_rng(int(args.seed))
    excluded_atomic_numbers = _parse_excluded_atomic_numbers(tuple(args.exclude_elements))
    candidates = _candidate_ids(
        conn,
        max_atoms=int(args.max_atoms),
        excluded_atomic_numbers=excluded_atomic_numbers,
    )
    if str(args.sample_order) == "size":
        candidates.sort(key=lambda db_id: _candidate_size_key(conn, db_id=int(db_id), basis=str(args.basis)))
    else:
        rng.shuffle(candidates)
    logger.log(
        f"[sample] candidates={len(candidates)} sample_count={args.sample_count} "
        f"train_count={args.train_count} validation_count={args.validation_count} "
        f"max_atoms={args.max_atoms} excluded_atomic_numbers={sorted(excluded_atomic_numbers)} "
        f"sample_order={args.sample_order}"
    )
    if int(args.train_count) > int(args.sample_count):
        raise ValueError("--train-count cannot exceed --sample-count.")
    if int(args.train_count) + int(args.validation_count) > int(args.sample_count):
        raise ValueError("--train-count + --validation-count cannot exceed --sample-count.")

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for db_id in candidates[: max(1, int(args.max_candidates))]:
        split = _split_for_success_index(
            len(rows),
            train_count=int(args.train_count),
            validation_count=int(args.validation_count),
        )
        if len(rows) >= int(args.sample_count):
            break
        try:
            z, pos = _fetch_molecule(conn, db_id)
            row = _compute_row(
                db_id=db_id,
                z=z,
                pos_ang=pos,
                split=split,
                basis=str(args.basis),
                nroots=int(args.nroots),
                scf_conv_tol=float(args.scf_conv_tol),
                scf_max_cycle=int(args.scf_max_cycle),
                cc_conv_tol=float(args.cc_conv_tol),
                cc_max_cycle=int(args.cc_max_cycle),
                logger=logger,
            )
        except Exception as exc:
            failures.append({"db_id": str(db_id), "error": repr(exc)})
            logger.log(f"[ref] skip qh9_{db_id}: {exc!r}")
            continue
        rows.append(row)

    if len(rows) < int(args.sample_count):
        raise RuntimeError(
            f"Only generated {len(rows)} successful references out of {args.sample_count}; "
            f"increase --max-candidates or reduce molecule size."
        )
    _write_rows(outcsv, rows)
    (outcsv.parent / "failures.json").write_text(
        json.dumps(failures, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.log(f"[done] wrote {len(rows)} rows to {outcsv}")


if __name__ == "__main__":
    main()
