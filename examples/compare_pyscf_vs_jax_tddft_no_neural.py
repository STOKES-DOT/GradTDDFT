#!/usr/bin/env python3
"""Compare conventional B3LYP TDA/TDDFT against PySCF for one water molecule."""

from __future__ import annotations

import argparse

import jax
import numpy as np
from pyscf import dft, gto

from td_graddft import tdscf
from td_graddft.data.reference import restricted_reference_from_pyscf
from td_graddft.spectra import HARTREE_TO_EV


jax.config.update("jax_enable_x64", True)

WATER = """
O  0.000000  0.000000  0.117790
H  0.000000  0.755453 -0.471161
H  0.000000 -0.755453 -0.471161
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", default="def2-svp")
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument("--nstates", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mol = gto.M(
        atom=WATER,
        basis=args.basis,
        unit="Angstrom",
        spin=0,
        charge=0,
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = "b3lyp"
    mf.grids.level = int(args.grid_level)
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF B3LYP SCF did not converge.")

    reference = restricted_reference_from_pyscf(mf)
    print(f"RKS energy: {mf.e_tot:.12f} Ha")

    for label, pyscf_factory, graddft_factory in (
        ("TDA", mf.TDA, tdscf.TDA),
        ("TDDFT", mf.TDDFT, tdscf.TDDFT),
    ):
        pyscf_td = pyscf_factory()
        pyscf_td.nstates = int(args.nstates)
        pyscf_td.kernel()
        graddft_td = graddft_factory(
            reference,
            xc_functional="b3lyp",
            nstates=int(args.nstates),
        )
        result = graddft_td.kernel()
        if not bool(np.asarray(result.converged)):
            raise RuntimeError(f"GradTDDFT {label} Davidson did not converge.")

        reference_ev = np.asarray(pyscf_td.e) * HARTREE_TO_EV
        predicted_ev = np.asarray(result.excitation_energies) * HARTREE_TO_EV
        reference_f = np.asarray(pyscf_td.oscillator_strength())
        predicted_f = np.asarray(graddft_td.oscillator_strength())
        print(f"\n{label}")
        print("state  PySCF/eV  GradTDDFT/eV  |Delta|/eV  PySCF/f  GradTDDFT/f")
        for state in range(int(args.nstates)):
            print(
                f"{state + 1:5d}  {reference_ev[state]:9.6f}  "
                f"{predicted_ev[state]:12.6f}  "
                f"{abs(predicted_ev[state] - reference_ev[state]):10.3e}  "
                f"{reference_f[state]:8.5f}  {predicted_f[state]:11.5f}"
            )


if __name__ == "__main__":
    main()
