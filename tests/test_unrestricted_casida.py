import os

import numpy as np
import pytest

from pyscf_reference import unrestricted_reference_from_pyscf
from td_graddft.spectra import oscillator_strengths
from td_graddft.tddft import UnrestrictedCasidaTDDFT


pytestmark = pytest.mark.skipif(
    os.getenv("TD_GRADDFT_RUN_OPEN_SHELL_TESTS", "0") != "1",
    reason="Open-shell tests are disabled by default; set TD_GRADDFT_RUN_OPEN_SHELL_TESTS=1.",
)


def _pyscf_or_skip():
    try:
        from pyscf import dft, gto  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("PySCF is required for unrestricted Casida tests.")


def _make_oh_uks_pbe():
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
    mol.verbose = 0
    mol.build()

    mf = dft.UKS(mol)
    mf.xc = "pbe"
    mf.grids.level = 0
    mf.conv_tol = 1e-9
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF UKS(PBE/6-31G) did not converge for OH test.")
    return mf


def test_unrestricted_casida_gen_tdhf_vind_shapes():
    _pyscf_or_skip()
    mf = _make_oh_uks_pbe()
    reference = unrestricted_reference_from_pyscf(mf)

    solver = UnrestrictedCasidaTDDFT(reference)
    matrices = solver.build_matrices()
    vind, flat_a, flat_b = solver.gen_tdhf_vind()

    assert flat_a.shape == flat_b.shape
    assert flat_a.shape[0] == flat_a.shape[1]
    assert flat_a.shape == matrices.a_matrix.shape
    assert flat_b.shape == matrices.b_matrix.shape

    n = flat_a.shape[0]
    z = np.random.default_rng(0).normal(size=(3, 2 * n))
    out = np.asarray(vind(z))
    assert out.shape == z.shape
    assert np.all(np.isfinite(out))


def test_unrestricted_casida_kernel_and_oscillator_strengths_smoke():
    _pyscf_or_skip()
    mf = _make_oh_uks_pbe()
    reference = unrestricted_reference_from_pyscf(mf)

    solver = UnrestrictedCasidaTDDFT(reference)
    result = solver.kernel(nstates=3)
    osc = oscillator_strengths(reference, result)

    assert result.excitation_energies.size > 0
    assert np.all(np.isfinite(np.asarray(result.excitation_energies)))
    assert result.x_amplitudes_alpha.shape[0] == result.excitation_energies.size
    assert result.y_amplitudes_beta.shape[0] == result.excitation_energies.size
    assert osc.shape == result.excitation_energies.shape
    assert np.all(np.isfinite(np.asarray(osc)))
