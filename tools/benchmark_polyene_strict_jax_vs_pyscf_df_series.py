from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/td_graddft_mplconfig")

import jax
import matplotlib
import numpy as np
from pyscf import dft, gto

from td_graddft import tdscf
from td_graddft.scf.builders import restricted_molecule_from_spec_with_jax_rks
from td_graddft.scf import RKSConfig
from td_graddft.spectra import HARTREE_TO_EV

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class BenchmarkRow:
    n_carbon: int
    repeat_index: int
    nao: int
    nocc: int
    nvir: int
    pyscf_scf_s: float
    jax_reference_s: float
    pyscf_tda_s: float
    jax_tda_s: float
    pyscf_casida_s: float
    jax_casida_s: float
    pyscf_total_tda_s: float
    jax_total_tda_s: float
    pyscf_total_casida_s: float
    jax_total_casida_s: float
    pyscf_total_energy_ha: float
    jax_total_energy_ha: float
    energy_diff_ha: float
    tda_mae_ev: float
    tda_osc_mae: float
    casida_mae_ev: float
    casida_osc_mae: float


def _polyene_atoms(n_carbon: int) -> list[tuple[str, tuple[float, float, float]]]:
    if n_carbon < 2:
        raise ValueError("n_carbon must be >= 2")

    c_c_double = 1.34
    c_c_single = 1.46
    c_h = 1.09
    z_wing = 0.90

    carbons: list[tuple[float, float, float]] = []
    x = 0.0
    carbons.append((x, 0.0, 0.0))
    for i in range(1, n_carbon):
        bond = c_c_double if (i - 1) % 2 == 0 else c_c_single
        x += bond
        carbons.append((x, 0.0, 0.0))

    atoms: list[tuple[str, tuple[float, float, float]]] = [("C", c) for c in carbons]
    x0 = carbons[0][0]
    xn = carbons[-1][0]
    atoms.extend(
        [
            ("H", (x0, +c_h, +z_wing)),
            ("H", (x0, +c_h, -z_wing)),
            ("H", (xn, -c_h, +z_wing)),
            ("H", (xn, -c_h, -z_wing)),
        ]
    )
    for i in range(1, n_carbon - 1):
        x_i = carbons[i][0]
        y_i = c_h if (i % 2 == 0) else -c_h
        atoms.append(("H", (x_i, y_i, 0.0)))
    return atoms


def _parse_carbons(text: str) -> list[int]:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("No carbon counts provided.")
    if any(v < 2 for v in values):
        raise ValueError("All carbon counts must be >= 2.")
    return values


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark strict-JAX DF/libcint vs PySCF DF on a polyene series."
    )
    p.add_argument("--carbons", default="2,4,6,8,10")
    p.add_argument("--basis", default="6-31g")
    p.add_argument("--xc", default="pbe0")
    p.add_argument("--nstates", type=int, default=1)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--grids-level", type=int, default=0)
    p.add_argument("--max-l", type=int, default=3)
    p.add_argument("--jax-precompile-eri", action="store_true")
    p.add_argument("--jax-precompile-eri-chunk-size", type=int, default=512)
    p.add_argument("--outdir", default="outputs/polyene_strict_jax_vs_pyscf_df_series")
    return p.parse_args()


def _time_call(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, float(time.perf_counter() - t0)


def _build_pyscf_reference(*, atom, basis: str, xc: str, grids_level: int):
    mol = gto.M(
        atom=atom,
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
    atom,
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
    ref = restricted_molecule_from_spec_with_jax_rks(
        atom=atom,
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


def _run_one_system(
    *,
    n_carbon: int,
    basis: str,
    xc: str,
    nstates: int,
    grids_level: int,
    max_l: int,
    precompile_eri: bool,
    precompile_eri_chunk_size: int,
) -> BenchmarkRow:
    atoms = _polyene_atoms(n_carbon)

    (mol, mf), pyscf_scf_s = _time_call(
        lambda: _build_pyscf_reference(
            atom=atoms,
            basis=basis,
            xc=xc,
            grids_level=grids_level,
        )
    )

    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 1e-8))
    nvir = int(np.asarray(mf.mo_coeff).shape[1] - nocc)
    nstates_eff = min(max(1, int(nstates)), max(1, nocc * nvir))
    compare_nstates = min(max(nstates_eff, 5), max(1, nocc * nvir))

    def _run_pyscf_tda():
        td = mf.TDA()
        td.nstates = int(nstates_eff)
        td.kernel()
        return np.asarray(td.e, dtype=float), np.asarray(td.oscillator_strength(), dtype=float)

    def _run_pyscf_casida():
        td = mf.TDDFT()
        td.nstates = int(nstates_eff)
        td.kernel()
        return np.asarray(td.e, dtype=float), np.asarray(td.oscillator_strength(), dtype=float)

    (pyscf_tda_e, pyscf_tda_f), pyscf_tda_s = _time_call(_run_pyscf_tda)
    (pyscf_casida_e, pyscf_casida_f), pyscf_casida_s = _time_call(_run_pyscf_casida)

    ref, jax_reference_s = _time_call(
        lambda: _build_strict_jax_reference(
            atom=atoms,
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
        result = tda_solver.kernel(nstates=int(nstates_eff))
        strengths = tda_solver.oscillator_strength()
        jax.block_until_ready(result.excitation_energies)
        jax.block_until_ready(strengths)
        return np.asarray(result.excitation_energies, dtype=float), np.asarray(strengths, dtype=float)

    def _run_jax_casida():
        result = solver.kernel(nstates=int(nstates_eff))
        strengths = solver.oscillator_strength()
        jax.block_until_ready(result.excitation_energies)
        jax.block_until_ready(strengths)
        return np.asarray(result.excitation_energies, dtype=float), np.asarray(strengths, dtype=float)

    def _compare_pyscf_tda():
        td = mf.TDA()
        td.nstates = int(compare_nstates)
        td.kernel()
        return np.asarray(td.e, dtype=float), np.asarray(td.oscillator_strength(), dtype=float)

    def _compare_pyscf_casida():
        td = mf.TDDFT()
        td.nstates = int(compare_nstates)
        td.kernel()
        return np.asarray(td.e, dtype=float), np.asarray(td.oscillator_strength(), dtype=float)

    def _compare_jax_tda():
        result = tda_solver.kernel(nstates=int(compare_nstates))
        strengths = tda_solver.oscillator_strength()
        jax.block_until_ready(result.excitation_energies)
        jax.block_until_ready(strengths)
        return np.asarray(result.excitation_energies, dtype=float), np.asarray(strengths, dtype=float)

    def _compare_jax_casida():
        result = solver.kernel(nstates=int(compare_nstates))
        strengths = solver.oscillator_strength()
        jax.block_until_ready(result.excitation_energies)
        jax.block_until_ready(strengths)
        return np.asarray(result.excitation_energies, dtype=float), np.asarray(strengths, dtype=float)

    (jax_tda_e, jax_tda_f), jax_tda_s = _time_call(_run_jax_tda)
    (jax_casida_e, jax_casida_f), jax_casida_s = _time_call(_run_jax_casida)

    compare_pyscf_tda_e, compare_pyscf_tda_f = _compare_pyscf_tda()
    compare_pyscf_casida_e, compare_pyscf_casida_f = _compare_pyscf_casida()
    compare_jax_tda_e, compare_jax_tda_f = _compare_jax_tda()
    compare_jax_casida_e, compare_jax_casida_f = _compare_jax_casida()

    nt = min(
        len(compare_pyscf_tda_e),
        len(compare_jax_tda_e),
        len(compare_pyscf_tda_f),
        len(compare_jax_tda_f),
        nstates_eff,
    )
    nc = min(
        len(compare_pyscf_casida_e),
        len(compare_jax_casida_e),
        len(compare_pyscf_casida_f),
        len(compare_jax_casida_f),
        nstates_eff,
    )

    return BenchmarkRow(
        n_carbon=int(n_carbon),
        repeat_index=-1,
        nao=int(mol.nao_nr()),
        nocc=int(nocc),
        nvir=int(nvir),
        pyscf_scf_s=float(pyscf_scf_s),
        jax_reference_s=float(jax_reference_s),
        pyscf_tda_s=float(pyscf_tda_s),
        jax_tda_s=float(jax_tda_s),
        pyscf_casida_s=float(pyscf_casida_s),
        jax_casida_s=float(jax_casida_s),
        pyscf_total_tda_s=float(pyscf_scf_s + pyscf_tda_s),
        jax_total_tda_s=float(jax_reference_s + jax_tda_s),
        pyscf_total_casida_s=float(pyscf_scf_s + pyscf_casida_s),
        jax_total_casida_s=float(jax_reference_s + jax_casida_s),
        pyscf_total_energy_ha=float(mf.e_tot),
        jax_total_energy_ha=float(ref.mf_energy),
        energy_diff_ha=float(ref.mf_energy - mf.e_tot),
        tda_mae_ev=float(
            np.mean(np.abs((compare_jax_tda_e[:nt] - compare_pyscf_tda_e[:nt]) * HARTREE_TO_EV))
        ),
        tda_osc_mae=float(np.mean(np.abs(compare_jax_tda_f[:nt] - compare_pyscf_tda_f[:nt]))),
        casida_mae_ev=float(
            np.mean(
                np.abs(
                    (compare_jax_casida_e[:nc] - compare_pyscf_casida_e[:nc]) * HARTREE_TO_EV
                )
            )
        ),
        casida_osc_mae=float(
            np.mean(np.abs(compare_jax_casida_f[:nc] - compare_pyscf_casida_f[:nc]))
        ),
    )


def _plot_timing(path: Path, aggregates: list[dict[str, float]]) -> None:
    carbons = np.asarray([row["n_carbon"] for row in aggregates], dtype=int)
    pyscf_tda = np.asarray([row["pyscf_total_tda_mean_s"] for row in aggregates], dtype=float)
    jax_tda = np.asarray([row["jax_total_tda_mean_s"] for row in aggregates], dtype=float)
    pyscf_casida = np.asarray([row["pyscf_total_casida_mean_s"] for row in aggregates], dtype=float)
    jax_casida = np.asarray([row["jax_total_casida_mean_s"] for row in aggregates], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    axes[0].plot(carbons, pyscf_tda, marker="o", lw=2.0, label="PySCF SCF+TDA")
    axes[0].plot(carbons, jax_tda, marker="s", lw=2.0, label="JAX ref+TDA")
    axes[0].set_xlabel("Carbon Count")
    axes[0].set_ylabel("Elapsed Time (s)")
    axes[0].set_title("TDA Pipeline Time")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(carbons, pyscf_casida, marker="o", lw=2.0, label="PySCF SCF+Casida")
    axes[1].plot(carbons, jax_casida, marker="s", lw=2.0, label="JAX ref+Casida")
    axes[1].set_xlabel("Carbon Count")
    axes[1].set_ylabel("Elapsed Time (s)")
    axes[1].set_title("Casida Pipeline Time")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_error(path: Path, aggregates: list[dict[str, float]]) -> None:
    carbons = np.asarray([row["n_carbon"] for row in aggregates], dtype=int)
    energy_diff_mha = np.asarray([abs(row["energy_diff_mean_ha"]) * 1000.0 for row in aggregates], dtype=float)
    tda_mae = np.asarray([row["tda_mae_mean_ev"] for row in aggregates], dtype=float)
    casida_mae = np.asarray([row["casida_mae_mean_ev"] for row in aggregates], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    axes[0].plot(carbons, energy_diff_mha, marker="o", lw=2.0)
    axes[0].set_xlabel("Carbon Count")
    axes[0].set_ylabel("|ΔE_tot| (mHa)")
    axes[0].set_title("Ground-State Energy Error")
    axes[0].grid(alpha=0.25)
    axes[0].set_yscale("log")

    axes[1].plot(carbons, tda_mae, marker="o", lw=2.0, label="TDA MAE")
    axes[1].plot(carbons, casida_mae, marker="s", lw=2.0, label="Casida MAE")
    axes[1].set_xlabel("Carbon Count")
    axes[1].set_ylabel("State MAE (eV)")
    axes[1].set_title("Excitation Energy Error")
    axes[1].grid(alpha=0.25)
    axes[1].set_yscale("log")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    jax.config.update("jax_enable_x64", True)

    carbons = _parse_carbons(args.carbons)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[BenchmarkRow] = []
    for n_carbon in carbons:
        for repeat_idx in range(int(args.repeats)):
            print(
                json.dumps(
                    {
                        "event": "start_system",
                        "n_carbon": int(n_carbon),
                        "repeat_index": int(repeat_idx + 1),
                        "basis": str(args.basis),
                        "xc": str(args.xc),
                        "jax_backend": str(jax.default_backend()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            row = _run_one_system(
                n_carbon=int(n_carbon),
                basis=str(args.basis),
                xc=str(args.xc),
                nstates=int(args.nstates),
                grids_level=int(args.grids_level),
                max_l=int(args.max_l),
                precompile_eri=bool(args.jax_precompile_eri),
                precompile_eri_chunk_size=int(args.jax_precompile_eri_chunk_size),
            )
            row = BenchmarkRow(
                repeat_index=repeat_idx + 1,
                **{k: v for k, v in asdict(row).items() if k != "repeat_index"},
            )
            rows.append(row)
            print(json.dumps(asdict(row), sort_keys=True), flush=True)

    fieldnames = list(asdict(rows[0]).keys())
    stem = f"polyene_series_{str(args.xc).lower()}_{str(args.basis).lower()}".replace("*", "star")
    csv_path = outdir / f"{stem}.csv"
    summary_path = outdir / f"{stem}_summary.json"
    timing_png = outdir / f"{stem}_timing.png"
    error_png = outdir / f"{stem}_error.png"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    def _pick(name: str, n_carbon: int) -> list[float]:
        return [
            float(getattr(row, name))
            for row in rows
            if int(row.n_carbon) == int(n_carbon)
        ]

    aggregates: list[dict[str, float]] = []
    for n_carbon in carbons:
        entry = {
            "n_carbon": int(n_carbon),
            "nao": int(next(row.nao for row in rows if int(row.n_carbon) == int(n_carbon))),
            "nocc": int(next(row.nocc for row in rows if int(row.n_carbon) == int(n_carbon))),
            "nvir": int(next(row.nvir for row in rows if int(row.n_carbon) == int(n_carbon))),
            "pyscf_scf_mean_s": float(np.mean(_pick("pyscf_scf_s", n_carbon))),
            "jax_reference_mean_s": float(np.mean(_pick("jax_reference_s", n_carbon))),
            "pyscf_tda_mean_s": float(np.mean(_pick("pyscf_tda_s", n_carbon))),
            "jax_tda_mean_s": float(np.mean(_pick("jax_tda_s", n_carbon))),
            "pyscf_casida_mean_s": float(np.mean(_pick("pyscf_casida_s", n_carbon))),
            "jax_casida_mean_s": float(np.mean(_pick("jax_casida_s", n_carbon))),
            "pyscf_total_tda_mean_s": float(np.mean(_pick("pyscf_total_tda_s", n_carbon))),
            "jax_total_tda_mean_s": float(np.mean(_pick("jax_total_tda_s", n_carbon))),
            "pyscf_total_casida_mean_s": float(np.mean(_pick("pyscf_total_casida_s", n_carbon))),
            "jax_total_casida_mean_s": float(np.mean(_pick("jax_total_casida_s", n_carbon))),
            "energy_diff_mean_ha": float(np.mean(_pick("energy_diff_ha", n_carbon))),
            "tda_mae_mean_ev": float(np.mean(_pick("tda_mae_ev", n_carbon))),
            "casida_mae_mean_ev": float(np.mean(_pick("casida_mae_ev", n_carbon))),
        }
        entry["tda_speedup_vs_pyscf"] = float(entry["pyscf_total_tda_mean_s"] / entry["jax_total_tda_mean_s"])
        entry["casida_speedup_vs_pyscf"] = float(entry["pyscf_total_casida_mean_s"] / entry["jax_total_casida_mean_s"])
        aggregates.append(entry)

    _plot_timing(timing_png, aggregates)
    _plot_error(error_png, aggregates)

    summary = {
        "basis": str(args.basis),
        "xc": str(args.xc),
        "jk_backend": "df",
        "jax_backend": str(jax.default_backend()),
        "nstates": int(args.nstates),
        "repeats": int(args.repeats),
        "carbons": carbons,
        "jax_precompile_eri": bool(args.jax_precompile_eri),
        "jax_precompile_eri_chunk_size": int(args.jax_precompile_eri_chunk_size),
        "rows": [asdict(row) for row in rows],
        "aggregates": aggregates,
        "csv": str(csv_path),
        "timing_png": str(timing_png),
        "error_png": str(error_png),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"csv={csv_path}", flush=True)
    print(f"summary={summary_path}", flush=True)
    print(f"timing_png={timing_png}", flush=True)
    print(f"error_png={error_png}", flush=True)


if __name__ == "__main__":
    main()
