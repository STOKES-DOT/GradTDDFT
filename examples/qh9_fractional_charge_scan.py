from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pyscf import dft, gto

from td_graddft_tools.fractional_charge import (
    FractionalChargeAnalysisConfig,
    FractionalChargeOutputConfig,
    run_fractional_charge_workflow,
)

ATOMIC_SYMBOL = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
}


@dataclass(frozen=True)
class FractionalReference:
    mo_coeff: np.ndarray
    mo_occ: np.ndarray
    rdm1: np.ndarray
    electron_count: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a QH9 B3LYP fractional-charge piecewise-linearity scan."
    )
    parser.add_argument("--db-path", default="/Volumes/TF/QH9_db/QH9Stable.db")
    parser.add_argument("--db-id", type=int, default=4, help="QH9 record id")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--xc", default="b3lyp")
    parser.add_argument("--grid-level", type=int, default=5)
    parser.add_argument("--charge-min", type=float, default=-1.0)
    parser.add_argument("--charge-max", type=float, default=1.0)
    parser.add_argument("--num-points", type=int, default=41)
    parser.add_argument(
        "--outdir",
        default="outputs/qh9_fractional_charge_scan",
        help="output directory",
    )
    return parser.parse_args()


def _formula_from_z(z: np.ndarray) -> str:
    counts: dict[str, int] = {}
    for zi in z:
        sym = ATOMIC_SYMBOL[int(zi)]
        counts[sym] = counts.get(sym, 0) + 1

    ordered = []
    if "C" in counts:
        ordered.append(("C", counts.pop("C")))
    if "H" in counts:
        ordered.append(("H", counts.pop("H")))
    for sym in sorted(counts):
        ordered.append((sym, counts[sym]))
    return "".join(sym if n == 1 else f"{sym}{n}" for sym, n in ordered)


def _fetch_molecule(db_path: str, db_id: int) -> tuple[np.ndarray, np.ndarray]:
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT N, Z, pos FROM data WHERE id = ?", (db_id,)).fetchone()
    conn.close()
    if row is None:
        raise KeyError(f"QH9 id {db_id} not found in {db_path}")
    n, z_blob, pos_blob = row
    z = np.frombuffer(z_blob, dtype=np.int32).copy()
    pos = np.frombuffer(pos_blob, dtype=np.float64).reshape(int(n), 3).copy()
    return z, pos


def _build_atom_block(z: np.ndarray, pos_ang: np.ndarray) -> str:
    lines = []
    for zi, xyz in zip(z, pos_ang, strict=True):
        sym = ATOMIC_SYMBOL[int(zi)]
        lines.append(f"{sym} {xyz[0]: .12f} {xyz[1]: .12f} {xyz[2]: .12f}")
    return "\n".join(lines)


def _build_fractional_energy_evaluator(
    mol: gto.Mole,
    *,
    xc: str,
    grid_level: int,
    base_electron_count: float,
) -> tuple[dict[float, float], dict[float, bool], dict[float, int], object]:
    energies_by_delta: dict[float, float] = {}
    converged_by_delta: dict[float, bool] = {}
    cycles_by_delta: dict[float, int] = {}

    def energy_evaluator(molecule_like: object) -> float:
        delta = float(np.sum(np.asarray(molecule_like.mo_occ)) - base_electron_count)
        key = round(delta, 8)
        if key in energies_by_delta:
            return energies_by_delta[key]

        target_nelec = base_electron_count + float(delta)
        n_pairs = int(np.floor(target_nelec / 2.0 + 1e-12))
        frac_occ = float(target_nelec - 2.0 * n_pairs)

        mf = dft.RKS(mol)
        mf.xc = xc
        mf.grids.level = grid_level
        mf.conv_tol = 1e-10
        mf.max_cycle = 200

        def get_occ(mo_energy=None, mo_coeff=None):
            energies = np.asarray(mo_energy)
            occ = np.zeros_like(energies)
            occ[:n_pairs] = 2.0
            if frac_occ > 1e-12:
                if n_pairs >= occ.size:
                    raise ValueError("Fractional occupation exceeds available MO space.")
                occ[n_pairs] = frac_occ
            return occ

        mf.get_occ = get_occ
        e_tot = mf.kernel(dm0=np.asarray(molecule_like.rdm1))
        energies_by_delta[key] = float(e_tot)
        converged_by_delta[key] = bool(mf.converged)
        cycles_by_delta[key] = int(getattr(mf, "cycles", mf.max_cycle))
        return float(e_tot)

    return energies_by_delta, converged_by_delta, cycles_by_delta, energy_evaluator


def main() -> None:
    args = parse_args()
    z, pos_ang = _fetch_molecule(args.db_path, args.db_id)
    formula = _formula_from_z(z)
    atom = _build_atom_block(z, pos_ang)
    mol = gto.M(
        atom=atom,
        basis=args.basis,
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    mf0 = dft.RKS(mol)
    mf0.xc = args.xc
    mf0.grids.level = args.grid_level
    mf0.conv_tol = 1e-10
    mf0.max_cycle = 200
    mf0.kernel()
    if not mf0.converged:
        raise RuntimeError("Neutral SCF did not converge.")

    reference = FractionalReference(
        mo_coeff=np.asarray(mf0.mo_coeff),
        mo_occ=np.asarray(mf0.mo_occ),
        rdm1=np.asarray(mf0.make_rdm1()),
        electron_count=float(np.sum(mf0.mo_occ)),
    )
    base_electron_count = float(reference.electron_count)
    (
        energies_by_delta,
        converged_by_delta,
        cycles_by_delta,
        energy_evaluator,
    ) = _build_fractional_energy_evaluator(
        mol,
        xc=args.xc,
        grid_level=args.grid_level,
        base_electron_count=base_electron_count,
    )

    prefix = f"qh9_id{args.db_id}_{formula}_{args.xc}_{args.basis}".replace("*", "star")
    outdir = Path(args.outdir) / prefix
    result = run_fractional_charge_workflow(
        reference,
        energy_evaluator,
        analysis_config=FractionalChargeAnalysisConfig(
            charge_min=args.charge_min,
            charge_max=args.charge_max,
            num_points=args.num_points,
        ),
        output_config=FractionalChargeOutputConfig(
            outdir=outdir,
            prefix="fractional_charge",
            title=f"QH9 id={args.db_id} {formula} {args.xc.upper()} fractional charge",
            energy_unit="ev",
        ),
    )

    summary = {
        "db_id": int(args.db_id),
        "formula": formula,
        "basis": args.basis,
        "xc": args.xc,
        "neutral_energy_ha": float(mf0.e_tot),
        "max_abs_deviation_ha": float(result.analysis.max_abs_deviation_ha),
        "mean_abs_deviation_ha": float(result.analysis.mean_abs_deviation_ha),
        "rms_deviation_ha": float(result.analysis.rms_deviation_ha),
        "left_endpoint_slope_ha": float(result.analysis.left_endpoint_slope_ha),
        "right_endpoint_slope_ha": float(result.analysis.right_endpoint_slope_ha),
        "energies_by_delta": {f"{k:+.2f}": float(v) for k, v in sorted(energies_by_delta.items())},
        "converged_by_delta": {
            f"{k:+.2f}": bool(v) for k, v in sorted(converged_by_delta.items())
        },
        "cycles_by_delta": {f"{k:+.2f}": int(v) for k, v in sorted(cycles_by_delta.items())},
        "csv_path": str(result.csv_path),
        "png_path": str(result.png_path),
        "summary_path": str(result.summary_path),
    }
    json_path = outdir / "fractional_charge_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(result.png_path)


if __name__ == "__main__":
    main()
