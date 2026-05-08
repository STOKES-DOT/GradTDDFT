import os

import numpy as np
import pytest

from td_graddft.pyscf_bridge import unrestricted_reference_from_pyscf_with_jax_uks
from td_graddft.scf import UKSConfig


pytestmark = pytest.mark.skipif(
    os.getenv("TD_GRADDFT_RUN_OPEN_SHELL_TESTS", "0") != "1",
    reason="Open-shell tests are disabled by default; set TD_GRADDFT_RUN_OPEN_SHELL_TESTS=1.",
)


def _pyscf_or_skip():
    try:
        from pyscf import dft, gto  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("PySCF is required for PySCF bridge tests.")


def _make_oh_pbe_mf():
    from pyscf import dft, gto

    mol = gto.Mole()
    mol.atom = """
    O 0.000000 0.000000 0.000000
    H 0.000000 0.000000 0.969700
    """
    mol.unit = "Angstrom"
    mol.basis = "sto-3g"
    mol.spin = 1
    mol.charge = 0
    mol.verbose = 0
    mol.build()

    mf = dft.UKS(mol)
    mf.xc = "pbe"
    mf.grids.level = 0
    mf.conv_tol = 1e-9
    mf.max_cycle = 80
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF UKS did not converge for OH test setup.")
    return mf


def test_unrestricted_reference_from_pyscf_with_jax_uks_shapes_and_target_energy():
    _pyscf_or_skip()
    mf = _make_oh_pbe_mf()

    ref = unrestricted_reference_from_pyscf_with_jax_uks(
        mf,
        max_l=1,
        uks_config=UKSConfig(max_cycle=10, conv_tol=1e-7, conv_tol_density=1e-5),
        energy_target=float(mf.e_tot),
    )

    nao = ref.mo_coeff.shape[-1]
    assert ref.ao.shape[1] == nao
    assert ref.h1e.shape == (nao, nao)
    assert ref.rep_tensor.shape == (nao, nao, nao, nao)
    assert ref.mo_coeff.shape[0] == 2
    assert ref.mo_occ.shape[0] == 2
    assert ref.rdm1.shape == (2, nao, nao)
    assert np.isclose(float(ref.mf_energy), float(mf.e_tot), atol=1e-12, rtol=1e-12)
    assert np.isfinite(float(ref.exact_exchange_fraction))
