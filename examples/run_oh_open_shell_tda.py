from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

from td_graddft.data.reference import unrestricted_reference_from_pyscf
from td_graddft.spectra import HARTREE_TO_EV
from td_graddft import tdscf


def _build_oh_uks(xc: str, basis: str):
    from pyscf import dft, gto

    mol = gto.Mole()
    mol.atom = """
    O 0.000000 0.000000 0.000000
    H 0.000000 0.000000 0.969700
    """
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.spin = 1
    mol.charge = 0
    mol.cart = True
    mol.verbose = 0
    mol.build()

    mf = dft.UKS(mol)
    mf.xc = xc
    mf.grids.level = 0
    mf.conv_tol = 1e-9
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"PySCF UKS did not converge for {xc}/{basis}.")
    return mf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open-shell OH unrestricted TDA smoke and reference comparison.",
    )
    parser.add_argument("--xc", type=str, default="b3lyp")
    parser.add_argument("--basis", type=str, default="6-31g")
    parser.add_argument("--nstates", type=int, default=6)
    args = parser.parse_args()

    mf = _build_oh_uks(args.xc, args.basis)
    ref = unrestricted_reference_from_pyscf(mf)
    tda = tdscf.TDA(ref)
    result = tda.kernel(nstates=args.nstates)
    osc = tda.oscillator_strength()

    td_ref = mf.TDA()
    td_ref.nstates = args.nstates
    td_ref.kernel()
    ref_energies = np.asarray(td_ref.e)
    ref_osc = np.asarray(td_ref.oscillator_strength())

    ours_energies = np.asarray(result.excitation_energies)
    ours_osc = np.asarray(osc)
    nrows = int(min(ref_energies.size, ours_energies.size))

    print(f"OH open-shell test with {args.xc}/{args.basis} (cartesian basis)")
    print(f"UKS total energy: {float(mf.e_tot):.10f} Ha")
    print("state  E_ref(eV)  E_jax(eV)  f_ref     f_jax")
    for i in range(nrows):
        print(
            f"{i+1:>5d}  "
            f"{ref_energies[i] * HARTREE_TO_EV:>8.4f}  "
            f"{ours_energies[i] * HARTREE_TO_EV:>8.4f}  "
            f"{ref_osc[i]:>7.4f}  "
            f"{ours_osc[i]:>7.4f}"
        )


if __name__ == "__main__":
    main()
