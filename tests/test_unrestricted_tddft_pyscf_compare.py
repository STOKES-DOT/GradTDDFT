import os

import jax.numpy as jnp
import numpy as np
import pytest

from pyscf_reference import unrestricted_reference_from_pyscf
from td_graddft.spectra import oscillator_strengths
from td_graddft.tddft import UnrestrictedCasidaTDDFT, UnrestrictedTDA
from td_graddft.tddft._unrestricted_semilocal_response import (
    UnrestrictedSemilocalResponseFunctional,
)
from td_graddft.tddft.unrestricted import (
    build_unrestricted_tda_operator,
    build_unrestricted_tdhf_operator,
)


ENERGY_ATOL = 8e-6
ENERGY_RTOL = 2e-5
OSC_ATOL = 2e-5
OSC_RTOL = 2e-4
MATRIX_ATOL = 2e-5
MATRIX_RTOL = 2e-5


pytestmark = pytest.mark.skipif(
    os.getenv("TD_GRADDFT_RUN_OPEN_SHELL_TESTS", "0") != "1",
    reason="Open-shell PySCF comparisons are disabled by default.",
)


@pytest.fixture(scope="module")
def h2plus_b3lyp_reference():
    bond = 1.06
    return _make_uks_reference(
        atom=f"H 0 0 {-0.5 * bond}; H 0 0 {0.5 * bond}",
        xc="b3lyp",
        charge=1,
        spin=1,
    )


@pytest.fixture(scope="module")
def lihplus_b3lyp_reference():
    return _make_uks_reference(
        atom="Li 0 0 0; H 0 0 1.60",
        xc="b3lyp",
        charge=1,
        spin=1,
    )


@pytest.fixture(scope="module")
def o2_pbe_reference():
    bond = 1.2075
    return _make_uks_reference(
        atom=f"O 0 0 {-0.5 * bond}; O 0 0 {0.5 * bond}",
        xc="pbe",
        charge=0,
        spin=2,
    )


def _make_uks_reference(*, atom, xc, charge, spin):
    pytest.importorskip("jax_xc")
    pytest.importorskip("pyscf")
    from pyscf import dft, gto

    mol = gto.M(
        atom=atom,
        unit="Angstrom",
        basis="def2-svp",
        charge=charge,
        spin=spin,
        cart=True,
        verbose=0,
    )
    mf = dft.UKS(mol)
    mf.xc = xc
    mf.grids.level = 2
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 64
    mf.init_guess = "minao"
    mf.kernel()
    if not mf.converged:
        density = mf.make_rdm1()
        mf = mf.newton()
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-8
        mf.max_cycle = 100
        mf.kernel(dm0=density)
    assert mf.converged
    return mf, unrestricted_reference_from_pyscf(mf)


def _pyscf_tda(mf, nstates):
    solver = mf.TDA()
    solver.nstates = nstates
    solver.conv_tol = 1e-9
    solver.max_cycle = 100
    solver.kernel()
    assert np.all(np.asarray(solver.converged))
    return solver


def _pyscf_tddft(mf, nstates):
    solver = mf.TDDFT()
    solver.nstates = nstates
    solver.conv_tol = 1e-9
    solver.max_cycle = 100
    solver.kernel()
    assert np.all(np.asarray(solver.converged))
    return solver


def _assert_energies_and_oscillator_strengths(
    reference,
    result,
    pyscf_solver,
    *,
    expected_count=4,
):
    predicted_energies = np.asarray(result.excitation_energies, dtype=float)
    predicted_oscillator_strengths = np.asarray(
        oscillator_strengths(reference, result),
        dtype=float,
    )
    reference_energies = np.asarray(pyscf_solver.e, dtype=float)
    reference_oscillator_strengths = np.asarray(
        pyscf_solver.oscillator_strength(),
        dtype=float,
    )
    count = min(
        predicted_energies.size,
        predicted_oscillator_strengths.size,
        reference_energies.size,
        reference_oscillator_strengths.size,
        expected_count,
    )
    assert count == expected_count
    np.testing.assert_allclose(
        predicted_energies[:count],
        reference_energies[:count],
        atol=ENERGY_ATOL,
        rtol=ENERGY_RTOL,
    )
    start = 0
    while start < count:
        stop = start + 1
        while (
            stop < count
            and abs(reference_energies[stop] - reference_energies[start]) <= 1e-5
        ):
            stop += 1
        np.testing.assert_allclose(
            np.sum(predicted_oscillator_strengths[start:stop]),
            np.sum(reference_oscillator_strengths[start:stop]),
            atol=OSC_ATOL,
            rtol=OSC_RTOL,
        )
        start = stop


def _pyscf_spin_block_matrix(blocks):
    aa, ab, bb = (np.asarray(block) for block in blocks)
    naa = int(np.prod(aa.shape[:2]))
    nbb = int(np.prod(bb.shape[:2]))
    aa = aa.reshape(naa, naa)
    ab = ab.reshape(naa, nbb)
    ba = np.asarray(blocks[1]).transpose(2, 3, 0, 1).reshape(nbb, naa)
    bb = bb.reshape(nbb, nbb)
    return np.block([[aa, ab], [ba, bb]])


def _operator_matrices(reference, functional):
    tda_vind, diagonal, _, _ = build_unrestricted_tda_operator(
        reference,
        functional,
    )
    dim = int(diagonal.size)
    identity = jnp.eye(dim, dtype=diagonal.dtype)
    tda_matrix = np.asarray(tda_vind(identity), dtype=float).T

    tddft_vind, _, _ = build_unrestricted_tdhf_operator(reference, functional)
    zeros = jnp.zeros_like(identity)
    a_columns = np.asarray(
        tddft_vind(jnp.concatenate([identity, zeros], axis=-1))[:, :dim],
        dtype=float,
    )
    b_columns = np.asarray(
        tddft_vind(jnp.concatenate([zeros, identity], axis=-1))[:, :dim],
        dtype=float,
    )
    return tda_matrix, a_columns.T, b_columns.T


def test_h2plus_b3lyp_tda_matches_pyscf_energies_and_oscillator_strengths(
    h2plus_b3lyp_reference,
):
    mf, reference = h2plus_b3lyp_reference
    functional = UnrestrictedSemilocalResponseFunctional("b3lyp")
    result = UnrestrictedTDA(
        reference,
        functional,
        davidson_tol=1e-9,
        davidson_max_iter=100,
    ).kernel(nstates=4)

    _assert_energies_and_oscillator_strengths(reference, result, _pyscf_tda(mf, 4))


def test_h2plus_b3lyp_tddft_matches_pyscf_energies_and_oscillator_strengths(
    h2plus_b3lyp_reference,
):
    mf, reference = h2plus_b3lyp_reference
    functional = UnrestrictedSemilocalResponseFunctional("b3lyp")
    result = UnrestrictedCasidaTDDFT(
        reference,
        functional,
        davidson_tol=1e-9,
        davidson_max_iter=100,
    ).kernel(nstates=4)

    _assert_energies_and_oscillator_strengths(reference, result, _pyscf_tddft(mf, 4))


def test_h2plus_b3lyp_response_matrices_match_pyscf(h2plus_b3lyp_reference):
    mf, reference = h2plus_b3lyp_reference
    functional = UnrestrictedSemilocalResponseFunctional("b3lyp")
    predicted_tda, predicted_a, predicted_b = _operator_matrices(
        reference,
        functional,
    )
    reference_a_blocks, reference_b_blocks = mf.TDDFT().get_ab()
    reference_a = _pyscf_spin_block_matrix(reference_a_blocks)
    reference_b = _pyscf_spin_block_matrix(reference_b_blocks)

    np.testing.assert_allclose(
        predicted_tda,
        reference_a,
        atol=MATRIX_ATOL,
        rtol=MATRIX_RTOL,
    )
    np.testing.assert_allclose(
        predicted_a,
        reference_a,
        atol=MATRIX_ATOL,
        rtol=MATRIX_RTOL,
    )
    np.testing.assert_allclose(
        predicted_b,
        reference_b,
        atol=MATRIX_ATOL,
        rtol=MATRIX_RTOL,
    )


@pytest.mark.parametrize("method", ["tda", "tddft"])
def test_lihplus_b3lyp_matches_pyscf_energies_and_oscillator_strengths(
    lihplus_b3lyp_reference,
    method,
):
    mf, reference = lihplus_b3lyp_reference
    functional = UnrestrictedSemilocalResponseFunctional("b3lyp")
    if method == "tda":
        result = UnrestrictedTDA(
            reference,
            functional,
            davidson_tol=1e-9,
            davidson_max_iter=100,
        ).kernel(nstates=4)
        pyscf_result = _pyscf_tda(mf, 4)
    else:
        result = UnrestrictedCasidaTDDFT(
            reference,
            functional,
            davidson_tol=1e-9,
            davidson_max_iter=100,
        ).kernel(nstates=4)
        pyscf_result = _pyscf_tddft(mf, 4)
    _assert_energies_and_oscillator_strengths(reference, result, pyscf_result)


@pytest.mark.parametrize("method", ["tda", "tddft"])
def test_o2_pbe_matches_pyscf_energies_and_oscillator_strengths(
    o2_pbe_reference,
    method,
):
    mf, reference = o2_pbe_reference
    functional = UnrestrictedSemilocalResponseFunctional("pbe")
    if method == "tda":
        result = UnrestrictedTDA(
            reference,
            functional,
            davidson_tol=1e-9,
            davidson_max_iter=100,
        ).kernel(nstates=3)
        pyscf_result = _pyscf_tda(mf, 3)
    else:
        result = UnrestrictedCasidaTDDFT(
            reference,
            functional,
            davidson_tol=1e-9,
            davidson_max_iter=100,
        ).kernel(nstates=3)
        pyscf_result = _pyscf_tddft(mf, 3)
    _assert_energies_and_oscillator_strengths(
        reference,
        result,
        pyscf_result,
        expected_count=3,
    )
