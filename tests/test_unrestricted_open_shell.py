import os

import numpy as np
import pytest

from td_graddft.data.basis import basis_from_pyscf_mol_cart
from td_graddft.data.integrals import eri_element, overlap_element
from td_graddft.reference_legacy import unrestricted_reference_from_pyscf
from td_graddft.spectra import oscillator_strengths
from td_graddft.tddft import UnrestrictedTDA


pytestmark = pytest.mark.skipif(
    os.getenv("TD_GRADDFT_RUN_OPEN_SHELL_TESTS", "0") != "1",
    reason="Open-shell tests are disabled by default; set TD_GRADDFT_RUN_OPEN_SHELL_TESTS=1.",
)


def _pyscf_or_skip():
    try:
        from pyscf import dft, gto  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("PySCF is required for unrestricted/open-shell tests.")


def _make_oh_b3lyp_mf():
    from pyscf import dft, gto

    mol = gto.Mole()
    mol.atom = """
    O 0.000000 0.000000 0.000000
    H 0.000000 0.000000 0.969700
    """
    mol.unit = "Angstrom"
    mol.basis = "6-31g"
    mol.spin = 1
    mol.charge = 0
    mol.cart = True
    mol.verbose = 0
    mol.build()

    mf = dft.UKS(mol)
    mf.xc = "b3lyp"
    mf.grids.level = 0
    mf.conv_tol = 1e-9
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF UKS(B3LYP/6-31G) did not converge for OH test.")
    return mf


def _pyscf_cart_eri_element(mol, i: int, j: int, k: int, l: int) -> float:
    ao_loc = np.asarray(mol.ao_loc_nr(cart=True))
    shell_starts = ao_loc[:-1]
    shell_stops = ao_loc[1:]

    def locate(ao_idx: int) -> tuple[int, int]:
        sh = int(np.searchsorted(shell_stops, ao_idx, side="right"))
        local = int(ao_idx - shell_starts[sh])
        return sh, local

    shi, li = locate(i)
    shj, lj = locate(j)
    shk, lk = locate(k)
    shl, ll = locate(l)
    block = mol.intor_by_shell("int2e_cart", (shi, shj, shk, shl))
    return float(block[li, lj, lk, ll])


def test_oh_b3lyp_basis_cart_integrals_match_pyscf_samples():
    _pyscf_or_skip()
    mf = _make_oh_b3lyp_mf()
    basis = basis_from_pyscf_mol_cart(mf.mol, max_l=3)

    angular_l = [sum(ao.angular) for ao in basis.aos]
    non_s_l = [l for l in angular_l if l > 0]
    assert len(non_s_l) > 0
    sample_l = max(non_s_l)
    sample_idx = [i for i, l in enumerate(angular_l) if l == sample_l]
    assert len(sample_idx) > 0

    s_ref = mf.mol.intor("int1e_ovlp_cart")
    i = sample_idx[0]
    j = 0
    s = float(overlap_element(basis, i, j))
    assert np.isclose(s, float(s_ref[i, j]), atol=8e-6, rtol=8e-6)

    eri = float(eri_element(basis, i, j, j, j))
    eri_ref = _pyscf_cart_eri_element(mf.mol, i, j, j, j)
    assert np.isclose(eri, eri_ref, atol=1.2e-5, rtol=1.2e-5)


def test_oh_open_shell_unrestricted_tda_smoke_with_b3lyp():
    _pyscf_or_skip()
    mf = _make_oh_b3lyp_mf()
    reference = unrestricted_reference_from_pyscf(mf)
    assert reference.mo_coeff.shape[0] == 2
    assert reference.mo_occ.shape[0] == 2
    assert reference.rdm1.shape[0] == 2
    assert reference.exact_exchange_fraction > 0.0

    tda = UnrestrictedTDA(reference)
    result = tda.kernel(nstates=3)
    osc = oscillator_strengths(reference, result)

    assert result.excitation_energies.size > 0
    assert np.all(np.isfinite(np.asarray(result.excitation_energies)))
    assert np.all(np.isfinite(np.asarray(osc)))
    assert osc.shape == result.excitation_energies.shape
    assert np.all(np.asarray(osc) >= -1e-8)
