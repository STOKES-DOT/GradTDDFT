from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pyscf import dft, gto

from td_graddft import tdscf
from td_graddft.scf.builders import restricted_molecule_from_spec_with_jax_rks
from td_graddft.scf import RKSConfig
from td_graddft.spectra import HARTREE_TO_EV, lorentzian_spectrum


MOLECULES = {
    "methane": """
C        0.0000000000      0.0000000000      0.0000000000
H        0.6291180000      0.6291180000      0.6291180000
H       -0.6291180000     -0.6291180000      0.6291180000
H       -0.6291180000      0.6291180000     -0.6291180000
H        0.6291180000     -0.6291180000     -0.6291180000
""",
    "benzene": """
C        0.0000000000      1.3967920000      0.0000000000
C       -1.2096570000      0.6983960000      0.0000000000
C       -1.2096570000     -0.6983960000      0.0000000000
C        0.0000000000     -1.3967920000      0.0000000000
C        1.2096570000     -0.6983960000      0.0000000000
C        1.2096570000      0.6983960000      0.0000000000
H        0.0000000000      2.4842120000      0.0000000000
H       -2.1513900000      1.2421060000      0.0000000000
H       -2.1513900000     -1.2421060000      0.0000000000
H        0.0000000000     -2.4842120000      0.0000000000
H        2.1513900000     -1.2421060000      0.0000000000
H        2.1513900000      1.2421060000      0.0000000000
""",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare strict-JAX full ground/excited-state calculation against PySCF."
    )
    parser.add_argument("--molecule", choices=tuple(MOLECULES.keys()), default="benzene")
    parser.add_argument("--basis", default="6-31g")
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--nstates", type=int, default=8)
    parser.add_argument("--grids-level", type=int, default=0)
    parser.add_argument("--jax-jk-backend", choices=("full", "df", "direct"), default="full")
    parser.add_argument("--integral-backend", choices=("libcint",), default="libcint")
    parser.add_argument(
        "--libcint-geometry-grad-policy",
        choices=("error", "zero"),
        default="error",
    )
    parser.add_argument("--jax-precompile-eri", action="store_true")
    parser.add_argument("--jax-precompile-eri-chunk-size", type=int, default=512)
    parser.add_argument("--eta-ev", type=float, default=0.12)
    parser.add_argument("--grid-points", type=int, default=2200)
    parser.add_argument("--grid-max-ev", type=float, default=12.0)
    parser.add_argument("--outdir", default="outputs/strict_jax_vs_pyscf_full")
    return parser.parse_args()


def _restricted_channel(arr) -> np.ndarray:
    arr_np = np.asarray(arr, dtype=float)
    if arr_np.ndim == 1:
        return arr_np
    if arr_np.ndim == 2 and arr_np.shape[0] in (1, 2):
        return arr_np[0]
    raise ValueError(f"Unsupported restricted array shape: {arr_np.shape}")


def _build_pyscf_reference(
    *,
    atom: str,
    basis: str,
    xc: str,
    grids_level: int,
    jk_backend: str,
):
    mol = gto.M(
        atom=atom,
        basis=basis,
        unit="Angstrom",
        spin=0,
        charge=0,
        cart=True,
        verbose=0,
    )
    mf = dft.RKS(mol)
    if str(jk_backend).lower() == "df":
        mf = mf.density_fit()
    mf.xc = xc
    mf.grids.level = int(grids_level)
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF SCF did not converge.")
    return mol, mf


def _build_strict_jax_reference(
    *,
    atom: str,
    basis: str,
    xc: str,
    grids_level: int,
    jk_backend: str,
    integral_backend: str,
    libcint_geometry_grad_policy: str,
    precompile_eri: bool,
    precompile_eri_chunk_size: int,
):
    config = RKSConfig(
        xc_spec=str(xc),
        max_cycle=32,
        conv_tol=1e-9,
        conv_tol_density=1e-7,
        damping=0.05,
        density_floor=1e-12,
        potential_clip=20.0,
        jk_backend=str(jk_backend),
    )
    return restricted_molecule_from_spec_with_jax_rks(
        atom=atom,
        basis=basis,
        xc_spec=str(xc),
        unit="Angstrom",
        charge=0,
        spin=0,
        cart=True,
        grids_level=int(grids_level),
        max_l=3,
        rks_config=config,
        grid_ao_backend="jax",
        integral_backend=str(integral_backend),
        libcint_geometry_grad_policy=str(libcint_geometry_grad_policy),
        precompile_eri=bool(precompile_eri),
        precompile_eri_chunk_size=int(precompile_eri_chunk_size),
    )


def _block_reference(reference) -> None:
    for name in (
        "ao",
        "ao_deriv1",
        "rep_tensor",
        "mo_coeff",
        "mo_occ",
        "mo_energy",
        "rdm1",
        "h1e",
        "overlap_matrix",
        "dipole_integrals",
    ):
        value = getattr(reference, name, None)
        if value is not None:
            jax.block_until_ready(value)


def _time_call(fn, *, block=None):
    t0 = time.perf_counter()
    out = fn()
    if block is not None:
        block(out)
    return out, float(time.perf_counter() - t0)


def _state_rows(ref_e, pred_e, ref_f, pred_f, n: int) -> list[list[float]]:
    out: list[list[float]] = []
    for idx in range(n):
        out.append(
            [
                idx + 1,
                float(ref_e[idx] * HARTREE_TO_EV),
                float(pred_e[idx] * HARTREE_TO_EV),
                float(abs((pred_e[idx] - ref_e[idx]) * HARTREE_TO_EV)),
                float(ref_f[idx]),
                float(pred_f[idx]),
                float(abs(pred_f[idx] - ref_f[idx])),
            ]
        )
    return out


def _write_state_csv(path: Path, rows: list[list[float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "state",
                "pyscf_energy_ev",
                "jax_energy_ev",
                "abs_diff_ev",
                "pyscf_osc",
                "jax_osc",
                "abs_diff_osc",
            ]
        )
        writer.writerows(rows)


def _spectrum_curve(energies_ha, strengths, grid_ev: np.ndarray, eta_ev: float) -> np.ndarray:
    return np.asarray(
        lorentzian_spectrum(
            jnp.asarray(np.asarray(energies_ha, dtype=float) * HARTREE_TO_EV),
            jnp.asarray(np.asarray(strengths, dtype=float)),
            jnp.asarray(grid_ev, dtype=float),
            eta=float(eta_ev),
        ),
        dtype=float,
    )


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    atom = MOLECULES[str(args.molecule)]
    (_, mf), pyscf_scf_elapsed_s = _time_call(
        lambda: _build_pyscf_reference(
            atom=atom,
            basis=str(args.basis),
            xc=str(args.xc),
            grids_level=int(args.grids_level),
            jk_backend=str(args.jax_jk_backend),
        )
    )
    ref, jax_reference_elapsed_s = _time_call(
        lambda: _build_strict_jax_reference(
            atom=atom,
            basis=str(args.basis),
            xc=str(args.xc),
            grids_level=int(args.grids_level),
            jk_backend=str(args.jax_jk_backend),
            integral_backend=str(args.integral_backend),
            libcint_geometry_grad_policy=str(args.libcint_geometry_grad_policy),
            precompile_eri=bool(args.jax_precompile_eri),
            precompile_eri_chunk_size=int(args.jax_precompile_eri_chunk_size),
        ),
        block=_block_reference,
    )

    pyscf_mo = np.asarray(mf.mo_energy, dtype=float)
    jax_mo = _restricted_channel(ref.mo_energy)
    mo_occ = np.asarray(mf.mo_occ, dtype=float)
    if pyscf_mo.shape != jax_mo.shape:
        raise RuntimeError(f"MO size mismatch: PySCF={pyscf_mo.shape}, JAX={jax_mo.shape}")

    mo_diff = jax_mo - pyscf_mo
    mo_abs = np.abs(mo_diff)
    homo_idx = int(np.where(mo_occ > 1e-8)[0][-1])
    lumo_idx = int(np.where(mo_occ <= 1e-8)[0][0])

    def _run_pyscf_tda():
        td = mf.TDA()
        td.nstates = int(args.nstates)
        td.kernel()
        return np.asarray(td.e, dtype=float), np.asarray(td.oscillator_strength(), dtype=float)

    def _run_pyscf_casida():
        td = mf.TDDFT()
        td.nstates = int(args.nstates)
        td.kernel()
        return np.asarray(td.e, dtype=float), np.asarray(td.oscillator_strength(), dtype=float)

    (pyscf_tda_e, pyscf_tda_f), pyscf_tda_elapsed_s = _time_call(_run_pyscf_tda)
    (pyscf_casida_e, pyscf_casida_f), pyscf_casida_elapsed_s = _time_call(_run_pyscf_casida)

    tda_solver = tdscf.TDA(ref, xc_functional=str(args.xc))
    solver = tdscf.TDDFT(ref, xc_functional=str(args.xc))

    def _run_jax_tda():
        result = tda_solver.kernel(nstates=int(args.nstates))
        strengths = tda_solver.oscillator_strength()
        jax.block_until_ready(result.excitation_energies)
        jax.block_until_ready(strengths)
        return result, strengths

    def _run_jax_casida():
        result = solver.kernel(nstates=int(args.nstates))
        strengths = solver.oscillator_strength()
        jax.block_until_ready(result.excitation_energies)
        jax.block_until_ready(strengths)
        return result, strengths

    (jax_tda, jax_tda_strengths), jax_tda_elapsed_s = _time_call(_run_jax_tda)
    (jax_casida, jax_casida_strengths), jax_casida_elapsed_s = _time_call(_run_jax_casida)
    jax_tda_e = np.asarray(jax_tda.excitation_energies, dtype=float)
    jax_tda_f = np.asarray(jax_tda_strengths, dtype=float)
    jax_casida_e = np.asarray(jax_casida.excitation_energies, dtype=float)
    jax_casida_f = np.asarray(jax_casida_strengths, dtype=float)

    nt = min(len(pyscf_tda_e), len(jax_tda_e), len(pyscf_tda_f), len(jax_tda_f))
    nc = min(len(pyscf_casida_e), len(jax_casida_e), len(pyscf_casida_f), len(jax_casida_f))

    tda_rows = _state_rows(pyscf_tda_e, jax_tda_e, pyscf_tda_f, jax_tda_f, nt)
    casida_rows = _state_rows(pyscf_casida_e, jax_casida_e, pyscf_casida_f, jax_casida_f, nc)

    mo_csv = outdir / f"{args.molecule}_{args.xc}_{args.basis}_mo_compare.csv"
    with mo_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mo_index",
                "occupation",
                "pyscf_mo_ha",
                "jax_mo_ha",
                "diff_ha",
                "abs_diff_ha",
                "pyscf_mo_ev",
                "jax_mo_ev",
                "abs_diff_mev",
            ]
        )
        for idx in range(pyscf_mo.size):
            writer.writerow(
                [
                    idx,
                    float(mo_occ[idx]),
                    float(pyscf_mo[idx]),
                    float(jax_mo[idx]),
                    float(mo_diff[idx]),
                    float(mo_abs[idx]),
                    float(pyscf_mo[idx] * HARTREE_TO_EV),
                    float(jax_mo[idx] * HARTREE_TO_EV),
                    float(mo_abs[idx] * HARTREE_TO_EV * 1000.0),
                ]
            )

    tda_csv = outdir / f"{args.molecule}_{args.xc}_{args.basis}_tda_compare.csv"
    casida_csv = outdir / f"{args.molecule}_{args.xc}_{args.basis}_casida_compare.csv"
    _write_state_csv(tda_csv, tda_rows)
    _write_state_csv(casida_csv, casida_rows)

    grid_ev = np.linspace(0.0, float(args.grid_max_ev), int(args.grid_points), dtype=float)
    pyscf_tda_curve = _spectrum_curve(pyscf_tda_e[:nt], pyscf_tda_f[:nt], grid_ev, float(args.eta_ev))
    jax_tda_curve = _spectrum_curve(jax_tda_e[:nt], jax_tda_f[:nt], grid_ev, float(args.eta_ev))
    pyscf_casida_curve = _spectrum_curve(
        pyscf_casida_e[:nc], pyscf_casida_f[:nc], grid_ev, float(args.eta_ev)
    )
    jax_casida_curve = _spectrum_curve(
        jax_casida_e[:nc], jax_casida_f[:nc], grid_ev, float(args.eta_ev)
    )

    spectrum_csv = outdir / f"{args.molecule}_{args.xc}_{args.basis}_spectrum_compare.csv"
    with spectrum_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "energy_ev",
                "pyscf_tda",
                "jax_tda",
                "pyscf_casida",
                "jax_casida",
            ]
        )
        for i in range(grid_ev.size):
            writer.writerow(
                [
                    float(grid_ev[i]),
                    float(pyscf_tda_curve[i]),
                    float(jax_tda_curve[i]),
                    float(pyscf_casida_curve[i]),
                    float(jax_casida_curve[i]),
                ]
            )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    axes[0].plot(grid_ev, pyscf_tda_curve, lw=2.0, label="PySCF TDA")
    axes[0].plot(grid_ev, jax_tda_curve, lw=2.0, label="JAX TDA")
    axes[0].set_title("TDA Spectrum")
    axes[1].plot(grid_ev, pyscf_casida_curve, lw=2.0, label="PySCF Casida")
    axes[1].plot(grid_ev, jax_casida_curve, lw=2.0, label="JAX Casida")
    axes[1].set_title("Casida Spectrum")
    for ax in axes:
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Absorption (arb. units)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle(f"{args.molecule.title()} {args.xc.upper()}/{args.basis}")
    fig.tight_layout()
    spectrum_png = outdir / f"{args.molecule}_{args.xc}_{args.basis}_spectrum_compare.png"
    fig.savefig(spectrum_png, dpi=170)
    plt.close(fig)

    summary = {
        "molecule": str(args.molecule),
        "xc": str(args.xc),
        "basis": str(args.basis),
        "grids_level": int(args.grids_level),
        "pyscf_jk_backend": str(args.jax_jk_backend),
        "jax_jk_backend": str(args.jax_jk_backend),
        "integral_backend": str(args.integral_backend),
        "libcint_geometry_grad_policy": str(args.libcint_geometry_grad_policy),
        "jax_precompile_eri": bool(args.jax_precompile_eri),
        "jax_precompile_eri_chunk_size": int(args.jax_precompile_eri_chunk_size),
        "nstates": int(args.nstates),
        "pyscf_total_energy_ha": float(mf.e_tot),
        "jax_total_energy_ha": float(ref.mf_energy),
        "total_energy_diff_ha": float(ref.mf_energy - mf.e_tot),
        "pyscf_scf_elapsed_s": float(pyscf_scf_elapsed_s),
        "jax_reference_elapsed_s": float(jax_reference_elapsed_s),
        "pyscf_tda_elapsed_s": float(pyscf_tda_elapsed_s),
        "jax_tda_elapsed_s": float(jax_tda_elapsed_s),
        "pyscf_casida_elapsed_s": float(pyscf_casida_elapsed_s),
        "jax_casida_elapsed_s": float(jax_casida_elapsed_s),
        "mo_mae_ha": float(np.mean(mo_abs)),
        "mo_max_abs_ha": float(np.max(mo_abs)),
        "mo_mae_mev": float(np.mean(mo_abs) * HARTREE_TO_EV * 1000.0),
        "mo_max_abs_mev": float(np.max(mo_abs) * HARTREE_TO_EV * 1000.0),
        "homo_index": homo_idx,
        "lumo_index": lumo_idx,
        "pyscf_homo_ha": float(pyscf_mo[homo_idx]),
        "jax_homo_ha": float(jax_mo[homo_idx]),
        "pyscf_lumo_ha": float(pyscf_mo[lumo_idx]),
        "jax_lumo_ha": float(jax_mo[lumo_idx]),
        "tda_state_mae_ev": float(np.mean(np.abs((jax_tda_e[:nt] - pyscf_tda_e[:nt]) * HARTREE_TO_EV))),
        "tda_state_max_ev": float(np.max(np.abs((jax_tda_e[:nt] - pyscf_tda_e[:nt]) * HARTREE_TO_EV))),
        "tda_osc_mae": float(np.mean(np.abs(jax_tda_f[:nt] - pyscf_tda_f[:nt]))),
        "tda_osc_max": float(np.max(np.abs(jax_tda_f[:nt] - pyscf_tda_f[:nt]))),
        "casida_state_mae_ev": float(
            np.mean(np.abs((jax_casida_e[:nc] - pyscf_casida_e[:nc]) * HARTREE_TO_EV))
        ),
        "casida_state_max_ev": float(
            np.max(np.abs((jax_casida_e[:nc] - pyscf_casida_e[:nc]) * HARTREE_TO_EV))
        ),
        "casida_osc_mae": float(np.mean(np.abs(jax_casida_f[:nc] - pyscf_casida_f[:nc]))),
        "casida_osc_max": float(np.max(np.abs(jax_casida_f[:nc] - pyscf_casida_f[:nc]))),
        "tda_curve_mae": float(np.mean(np.abs(jax_tda_curve - pyscf_tda_curve))),
        "tda_curve_max": float(np.max(np.abs(jax_tda_curve - pyscf_tda_curve))),
        "casida_curve_mae": float(np.mean(np.abs(jax_casida_curve - pyscf_casida_curve))),
        "casida_curve_max": float(np.max(np.abs(jax_casida_curve - pyscf_casida_curve))),
    }

    summary_path = outdir / f"{args.molecule}_{args.xc}_{args.basis}_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"summary={summary_path}")
    print(f"mo_csv={mo_csv}")
    print(f"tda_csv={tda_csv}")
    print(f"casida_csv={casida_csv}")
    print(f"spectrum_csv={spectrum_csv}")
    print(f"spectrum_png={spectrum_png}")


if __name__ == "__main__":
    main()
