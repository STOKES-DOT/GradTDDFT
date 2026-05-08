from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import numpy as np
from pyscf import dft, gto

from td_graddft import tdscf
from td_graddft.reference import restricted_reference_from_spec_with_jax_rks
from td_graddft.scf import RKSConfig
from td_graddft.spectra import HARTREE_TO_EV


WATER = """
O  0.000000  0.000000  0.117790
H  0.000000  0.755453 -0.471161
H  0.000000 -0.755453 -0.471161
"""


@dataclass(frozen=True)
class BenchmarkRow:
    repeat_index: int
    pyscf_scf_s: float
    jax_reference_s: float
    pyscf_tda_s: float
    jax_tda_s: float
    pyscf_casida_s: float
    jax_casida_s: float
    energy_diff_ha: float
    tda_mae_ev: float
    tda_osc_mae: float
    casida_mae_ev: float
    casida_osc_mae: float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Repeat water/STO-3G PySCF vs strict-JAX DF benchmark with fixed timing/accuracy protocol."
    )
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--xc", default="pbe")
    p.add_argument("--nstates", type=int, default=5)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--grids-level", type=int, default=0)
    p.add_argument("--max-l", type=int, default=3)
    p.add_argument("--jax-precompile-eri", action="store_true")
    p.add_argument("--jax-precompile-eri-chunk-size", type=int, default=512)
    p.add_argument("--outdir", default="outputs/water_strict_jax_vs_pyscf_df_repeats")
    return p.parse_args()


def _time_call(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, float(time.perf_counter() - t0)


def _build_pyscf_reference(*, basis: str, xc: str, grids_level: int):
    mol = gto.M(
        atom=WATER,
        basis=str(basis),
        unit="Angstrom",
        spin=0,
        charge=0,
        cart=True,
        verbose=0,
    )
    mf = dft.RKS(mol).density_fit()
    mf.xc = str(xc)
    mf.grids.level = int(grids_level)
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF SCF did not converge.")
    return mol, mf


def _build_strict_jax_reference(
    *,
    basis: str,
    xc: str,
    grids_level: int,
    max_l: int,
    precompile_eri: bool,
    precompile_eri_chunk_size: int,
):
    cfg = RKSConfig(
        xc_spec=str(xc),
        max_cycle=32,
        conv_tol=1e-9,
        conv_tol_density=1e-7,
        damping=0.05,
        potential_clip=20.0,
        jk_backend="df",
        df_tol=1e-10,
    )
    ref = restricted_reference_from_spec_with_jax_rks(
        atom=WATER,
        basis=str(basis),
        xc_spec=str(xc),
        unit="Angstrom",
        charge=0,
        spin=0,
        cart=True,
        grids_level=int(grids_level),
        max_l=int(max_l),
        rks_config=cfg,
        grid_ao_backend="jax",
        precompile_eri=bool(precompile_eri),
        precompile_eri_chunk_size=int(precompile_eri_chunk_size),
    )
    for name in (
        "ao",
        "ao_deriv1",
        "mo_coeff",
        "mo_occ",
        "mo_energy",
        "rdm1",
        "h1e",
        "overlap_matrix",
        "dipole_integrals",
        "df_factors",
        "eri_ovov",
        "eri_ovvo",
        "eri_oovv",
    ):
        value = getattr(ref, name, None)
        if value is not None:
            jax.block_until_ready(value)
    return ref


def _run_one(*, basis: str, xc: str, nstates: int, grids_level: int, max_l: int, precompile_eri: bool, precompile_eri_chunk_size: int) -> BenchmarkRow:
    (_, mf), pyscf_scf_s = _time_call(
        lambda: _build_pyscf_reference(
            basis=basis,
            xc=xc,
            grids_level=grids_level,
        )
    )

    def _run_pyscf_tda():
        td = mf.TDA()
        td.nstates = int(nstates)
        td.kernel()
        return np.asarray(td.e, dtype=float), np.asarray(td.oscillator_strength(), dtype=float)

    def _run_pyscf_casida():
        td = mf.TDDFT()
        td.nstates = int(nstates)
        td.kernel()
        return np.asarray(td.e, dtype=float), np.asarray(td.oscillator_strength(), dtype=float)

    (pyscf_tda_e, pyscf_tda_f), pyscf_tda_s = _time_call(_run_pyscf_tda)
    (pyscf_casida_e, pyscf_casida_f), pyscf_casida_s = _time_call(_run_pyscf_casida)

    ref, jax_reference_s = _time_call(
        lambda: _build_strict_jax_reference(
            basis=basis,
            xc=xc,
            grids_level=grids_level,
            max_l=max_l,
            precompile_eri=precompile_eri,
            precompile_eri_chunk_size=precompile_eri_chunk_size,
        )
    )

    tda_solver = tdscf.TDA(ref, xc_functional=str(xc))
    solver = tdscf.TDDFT(ref, xc_functional=str(xc))

    def _run_jax_tda():
        result = tda_solver.kernel(nstates=int(nstates))
        strengths = tda_solver.oscillator_strength()
        jax.block_until_ready(result.excitation_energies)
        jax.block_until_ready(strengths)
        return np.asarray(result.excitation_energies, dtype=float), np.asarray(strengths, dtype=float)

    def _run_jax_casida():
        result = solver.kernel(nstates=int(nstates))
        strengths = solver.oscillator_strength()
        jax.block_until_ready(result.excitation_energies)
        jax.block_until_ready(strengths)
        return np.asarray(result.excitation_energies, dtype=float), np.asarray(strengths, dtype=float)

    (jax_tda_e, jax_tda_f), jax_tda_s = _time_call(_run_jax_tda)
    (jax_casida_e, jax_casida_f), jax_casida_s = _time_call(_run_jax_casida)

    nt = min(len(pyscf_tda_e), len(jax_tda_e), len(pyscf_tda_f), len(jax_tda_f))
    nc = min(len(pyscf_casida_e), len(jax_casida_e), len(pyscf_casida_f), len(jax_casida_f))

    return BenchmarkRow(
        repeat_index=-1,
        pyscf_scf_s=float(pyscf_scf_s),
        jax_reference_s=float(jax_reference_s),
        pyscf_tda_s=float(pyscf_tda_s),
        jax_tda_s=float(jax_tda_s),
        pyscf_casida_s=float(pyscf_casida_s),
        jax_casida_s=float(jax_casida_s),
        energy_diff_ha=float(ref.mf_energy - mf.e_tot),
        tda_mae_ev=float(np.mean(np.abs((jax_tda_e[:nt] - pyscf_tda_e[:nt]) * HARTREE_TO_EV))),
        tda_osc_mae=float(np.mean(np.abs(jax_tda_f[:nt] - pyscf_tda_f[:nt]))),
        casida_mae_ev=float(np.mean(np.abs((jax_casida_e[:nc] - pyscf_casida_e[:nc]) * HARTREE_TO_EV))),
        casida_osc_mae=float(np.mean(np.abs(jax_casida_f[:nc] - pyscf_casida_f[:nc]))),
    )


def main() -> None:
    args = _parse_args()
    jax.config.update("jax_enable_x64", True)

    rows: list[BenchmarkRow] = []
    for repeat_idx in range(int(args.repeats)):
        row = _run_one(
            basis=str(args.basis),
            xc=str(args.xc),
            nstates=int(args.nstates),
            grids_level=int(args.grids_level),
            max_l=int(args.max_l),
            precompile_eri=bool(args.jax_precompile_eri),
            precompile_eri_chunk_size=int(args.jax_precompile_eri_chunk_size),
        )
        rows.append(BenchmarkRow(repeat_index=repeat_idx + 1, **{k: v for k, v in asdict(row).items() if k != "repeat_index"}))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"water_{str(args.xc).lower()}_{str(args.basis).lower()}_df_repeats"
    csv_path = outdir / f"{stem}.csv"
    summary_path = outdir / f"{stem}_summary.json"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    def _series(name: str) -> list[float]:
        return [float(getattr(row, name)) for row in rows]

    summary = {
        "basis": str(args.basis),
        "xc": str(args.xc),
        "jk_backend": "df",
        "nstates": int(args.nstates),
        "repeats": int(args.repeats),
        "jax_precompile_eri": bool(args.jax_precompile_eri),
        "jax_precompile_eri_chunk_size": int(args.jax_precompile_eri_chunk_size),
        "pyscf_scf_s": _series("pyscf_scf_s"),
        "jax_reference_s": _series("jax_reference_s"),
        "pyscf_tda_s": _series("pyscf_tda_s"),
        "jax_tda_s": _series("jax_tda_s"),
        "pyscf_casida_s": _series("pyscf_casida_s"),
        "jax_casida_s": _series("jax_casida_s"),
        "energy_diff_ha": _series("energy_diff_ha"),
        "tda_mae_ev": _series("tda_mae_ev"),
        "tda_osc_mae": _series("tda_osc_mae"),
        "casida_mae_ev": _series("casida_mae_ev"),
        "casida_osc_mae": _series("casida_osc_mae"),
        "jax_reference_mean_s": float(np.mean(_series("jax_reference_s"))),
        "jax_reference_warm_mean_s": float(np.mean(_series("jax_reference_s")[1:])) if len(rows) > 1 else float(_series("jax_reference_s")[0]),
        "jax_tda_mean_s": float(np.mean(_series("jax_tda_s"))),
        "jax_tda_warm_mean_s": float(np.mean(_series("jax_tda_s")[1:])) if len(rows) > 1 else float(_series("jax_tda_s")[0]),
        "jax_casida_mean_s": float(np.mean(_series("jax_casida_s"))),
        "jax_casida_warm_mean_s": float(np.mean(_series("jax_casida_s")[1:])) if len(rows) > 1 else float(_series("jax_casida_s")[0]),
        "pyscf_scf_mean_s": float(np.mean(_series("pyscf_scf_s"))),
        "pyscf_tda_mean_s": float(np.mean(_series("pyscf_tda_s"))),
        "pyscf_casida_mean_s": float(np.mean(_series("pyscf_casida_s"))),
    }

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"csv={csv_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
