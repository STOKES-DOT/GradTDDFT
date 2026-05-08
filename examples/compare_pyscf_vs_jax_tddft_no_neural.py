from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

import jax.numpy as jnp
import numpy as np
from pyscf import dft, gto

from td_graddft import tdscf
from td_graddft.reference_legacy import restricted_reference_from_pyscf
from td_graddft.spectra import HARTREE_TO_EV
from td_graddft.xc import AdiabaticDensityFunctional, lda_from_jax_xc


MOLECULES = {
    "h2o": """
O  0.000000  0.000000  0.117790
H  0.000000  0.755453 -0.471161
H  0.000000 -0.755453 -0.471161
""",
    "ch4": """
C  0.000000  0.000000  0.000000
H  0.629118  0.629118  0.629118
H -0.629118 -0.629118  0.629118
H -0.629118  0.629118 -0.629118
H  0.629118 -0.629118 -0.629118
""",
    "nh3": """
N  0.000000  0.000000  0.116489
H  0.000000  0.939731 -0.271808
H  0.813831 -0.469865 -0.271808
H -0.813831 -0.469865 -0.271808
""",
}


@dataclass(frozen=True)
class CompareRow:
    molecule: str
    xc_mode: str
    basis: str
    nstates: int
    compared_states: int
    mae_energy_ev: float
    max_energy_ev: float
    mae_oscillator: float
    max_oscillator: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PySCF TDDFT and JAX TDDFT without neural XC."
    )
    parser.add_argument(
        "--xc-mode",
        choices=("hf", "lda"),
        default="hf",
        help="reference and JAX adiabatic XC mode (non-neural)",
    )
    parser.add_argument("--basis", default="sto-3g", help="PySCF basis")
    parser.add_argument("--nstates", type=int, default=8, help="max states per molecule")
    parser.add_argument(
        "--molecules",
        nargs="+",
        default=["h2o", "ch4", "nh3"],
        help="molecule keys to include: h2o ch4 nh3",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/pyscf_vs_jax_tddft_no_neural",
        help="output directory",
    )
    return parser.parse_args()


def _build_mf(atom: str, *, basis: str, xc_mode: str):
    mol = gto.M(
        atom=atom,
        basis=basis,
        unit="Angstrom",
        spin=0,
        charge=0,
        verbose=0,
    )
    mf = dft.RKS(mol)
    if xc_mode == "hf":
        mf.xc = "hf"
    elif xc_mode == "lda":
        mf.xc = "lda,vwn"
    else:
        raise ValueError(f"Unsupported xc_mode={xc_mode!r}")
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"SCF did not converge for xc_mode={xc_mode}.")
    return mf


def _jax_xc_for_mode(xc_mode: str):
    if xc_mode == "hf":
        return AdiabaticDensityFunctional(
            name="hf_exact_exchange",
            energy_density_fn=lambda rho: jnp.zeros_like(rho),
            exact_exchange_fraction=1.0,
        )
    if xc_mode == "lda":
        return lda_from_jax_xc("lda")
    raise ValueError(f"Unsupported xc_mode={xc_mode!r}")


def _compare_one(
    *,
    molecule_name: str,
    atom: str,
    basis: str,
    xc_mode: str,
    nstates_cap: int,
) -> tuple[CompareRow, list[tuple[int, float, float, float, float]]]:
    mf = _build_mf(atom, basis=basis, xc_mode=xc_mode)
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 1e-8))
    nvir = int(np.asarray(mf.mo_coeff).shape[-1] - nocc)
    nstates = min(max(1, nstates_cap), nocc * nvir)

    td = mf.TDDFT()
    td.nstates = nstates
    td.kernel()
    ref_energies = np.asarray(td.e, dtype=float)
    ref_osc = np.asarray(td.oscillator_strength(), dtype=float)

    reference = restricted_reference_from_pyscf(mf)
    td_jax = tdscf.TDDFT(
        reference,
        xc_functional=_jax_xc_for_mode(xc_mode),
    )
    result = td_jax.kernel(nstates=nstates)
    pred_energies = np.asarray(result.excitation_energies, dtype=float)
    pred_osc = np.asarray(td_jax.oscillator_strength(), dtype=float)

    n = min(ref_energies.size, pred_energies.size, ref_osc.size, pred_osc.size, nstates)
    e_diff_ev = np.abs((pred_energies[:n] - ref_energies[:n]) * HARTREE_TO_EV)
    f_diff = np.abs(pred_osc[:n] - ref_osc[:n])

    row = CompareRow(
        molecule=molecule_name,
        xc_mode=xc_mode,
        basis=basis,
        nstates=nstates,
        compared_states=n,
        mae_energy_ev=float(np.mean(e_diff_ev)),
        max_energy_ev=float(np.max(e_diff_ev)),
        mae_oscillator=float(np.mean(f_diff)),
        max_oscillator=float(np.max(f_diff)),
    )
    state_rows = []
    for idx in range(n):
        state_rows.append(
            (
                idx + 1,
                float(ref_energies[idx] * HARTREE_TO_EV),
                float(pred_energies[idx] * HARTREE_TO_EV),
                float(ref_osc[idx]),
                float(pred_osc[idx]),
            )
        )
    return row, state_rows


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[CompareRow] = []
    for name in args.molecules:
        key = name.lower()
        if key not in MOLECULES:
            raise KeyError(f"Unknown molecule key: {name!r}")
        row, state_rows = _compare_one(
            molecule_name=key,
            atom=MOLECULES[key],
            basis=args.basis,
            xc_mode=args.xc_mode,
            nstates_cap=args.nstates,
        )
        rows.append(row)

        state_csv = outdir / f"{key}_{args.xc_mode}_{args.basis}_state_compare.csv"
        with state_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "state",
                    "pyscf_energy_ev",
                    "jax_energy_ev",
                    "pyscf_osc",
                    "jax_osc",
                ]
            )
            writer.writerows(state_rows)

    metrics_csv = outdir / f"metrics_{args.xc_mode}_{args.basis}.csv"
    with metrics_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "molecule",
                "xc_mode",
                "basis",
                "nstates",
                "compared_states",
                "mae_energy_ev",
                "max_energy_ev",
                "mae_oscillator",
                "max_oscillator",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.molecule,
                    row.xc_mode,
                    row.basis,
                    row.nstates,
                    row.compared_states,
                    f"{row.mae_energy_ev:.10f}",
                    f"{row.max_energy_ev:.10f}",
                    f"{row.mae_oscillator:.10f}",
                    f"{row.max_oscillator:.10f}",
                ]
            )

    avg_mae_energy = float(np.mean([r.mae_energy_ev for r in rows])) if rows else float("nan")
    avg_mae_osc = float(np.mean([r.mae_oscillator for r in rows])) if rows else float("nan")

    print("molecule,nstates,compared,mae_energy_ev,max_energy_ev,mae_osc,max_osc")
    for row in rows:
        print(
            f"{row.molecule},{row.nstates},{row.compared_states},"
            f"{row.mae_energy_ev:.8f},{row.max_energy_ev:.8f},"
            f"{row.mae_oscillator:.8f},{row.max_oscillator:.8f}"
        )
    print(f"avg_mae_energy_ev={avg_mae_energy:.8f}")
    print(f"avg_mae_osc={avg_mae_osc:.8f}")
    print(f"metrics_csv={metrics_csv}")


if __name__ == "__main__":
    main()
