import numpy as np
import pytest

from td_graddft.data.basis import basis_from_pyscf_mol_cart
from td_graddft.scf import RHFConfig, run_rhf


def _pyscf_or_skip():
    try:
        from pyscf import gto  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("PySCF is required for RHF reference tests.")


def test_rhf_h2_sto3g_matches_pyscf():
    _pyscf_or_skip()
    from pyscf import gto, scf

    mol = gto.M(
        atom="""
        H 0.0 0.0 -0.35
        H 0.0 0.0  0.35
        """,
        basis="sto-3g",
        cart=True,
        spin=0,
        verbose=0,
    )
    basis = basis_from_pyscf_mol_cart(mol, max_l=1)
    rhf_res = run_rhf(
        basis=basis,
        nelectron=mol.nelectron,
        nuclear_repulsion=float(mol.energy_nuc()),
        config=RHFConfig(max_cycle=80, conv_tol=1e-11, conv_tol_density=1e-9),
    )

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    assert rhf_res.converged
    assert np.isclose(rhf_res.total_energy, mf.e_tot, atol=2e-7, rtol=2e-7)
