from dataclasses import replace

import numpy as np
import pytest
import jax.numpy as jnp

from pyscf_reference import restricted_reference_from_pyscf
from td_graddft.spectra import oscillator_strengths, transition_dipoles
from td_graddft.tddft._semilocal_response import SemilocalResponseFunctional
from td_graddft.tddft import RestrictedCasidaTDDFT
from td_graddft.tddft.casida import solve_casida
from td_graddft.tddft.types import TDDFTResult
from td_graddft.tddft.types import TDDFTMatrices
from td_graddft.xc import AdiabaticDensityFunctional


def _pyscf_or_skip():
    try:
        from pyscf import dft, gto  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("PySCF is required for TDDFT comparison tests.")


def _make_water_reference(xc: str):
    from pyscf import dft, gto

    mol = gto.Mole()
    mol.atom = """
    O  0.000000  0.000000  0.117790
    H  0.000000  0.755453 -0.471161
    H  0.000000 -0.755453 -0.471161
    """
    mol.unit = "Angstrom"
    mol.basis = "sto-3g"
    mol.spin = 0
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"PySCF RKS({xc}/STO-3G) did not converge for water.")
    return mf


def _make_h2_reference(xc: str):
    from pyscf import dft, gto

    mol = gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 0.74"
    mol.unit = "Angstrom"
    mol.basis = "sto-3g"
    mol.spin = 0
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"PySCF RKS({xc}/STO-3G) did not converge for H2.")
    return mf


def _pyscf_xy_to_td_result(td) -> TDDFTResult:
    x = np.stack([pair[0] for pair in td.xy], axis=0)
    y = np.stack([pair[1] for pair in td.xy], axis=0)
    nocc, nvir = x.shape[1:]
    return TDDFTResult(
        excitation_energies=np.asarray(td.e),
        x_amplitudes=x,
        y_amplitudes=y,
        a_matrix=np.zeros((nocc, nvir, nocc, nvir)),
        b_matrix=np.zeros((nocc, nvir, nocc, nvir)),
        casida_matrix=np.zeros((nocc * nvir, nocc * nvir)),
    )


def test_restricted_tdhf_matches_pyscf_water_reference():
    _pyscf_or_skip()
    mf = _make_water_reference("hf")

    td = mf.TDDFT()
    td.nstates = 4
    td.kernel()
    ref_energies = np.asarray(td.e, dtype=float)
    ref_osc = np.asarray(td.oscillator_strength(), dtype=float)
    ref_a, ref_b = td.get_ab()

    reference = restricted_reference_from_pyscf(mf)
    hf_xc = AdiabaticDensityFunctional(
        name="hf_exact_exchange",
        energy_density_fn=lambda rho: jnp.zeros_like(rho),
        exact_exchange_fraction=1.0,
    )
    solver = RestrictedCasidaTDDFT(
        molecule=reference,
        xc_functional=hf_xc,
    )
    matrices = solver.build_matrices()
    pred_a = np.asarray(matrices.a_matrix, dtype=float)
    pred_b = np.asarray(matrices.b_matrix, dtype=float)
    result = solver.kernel(nstates=4)
    pred_energies = np.asarray(result.excitation_energies, dtype=float)
    pred_osc = np.asarray(oscillator_strengths(reference, result), dtype=float)

    # TDHF response matrices should match PySCF exactly up to floating-point noise.
    np.testing.assert_allclose(pred_a, np.asarray(ref_a, dtype=float), atol=3e-6, rtol=1e-7)
    np.testing.assert_allclose(pred_b, np.asarray(ref_b, dtype=float), atol=3e-6, rtol=1e-7)

    n = min(
        ref_energies.size,
        pred_energies.size,
        ref_osc.size,
        pred_osc.size,
    )
    assert n >= 2
    np.testing.assert_allclose(pred_energies[:n], ref_energies[:n], atol=5e-6, rtol=1e-5)
    np.testing.assert_allclose(pred_osc[:n], ref_osc[:n], atol=5e-5, rtol=5e-4)


def test_cached_mo_response_slices_match_ao_tensor_path():
    _pyscf_or_skip()
    mf = _make_water_reference("hf")
    reference = restricted_reference_from_pyscf(mf)
    uncached = replace(
        reference,
        eri_ovov=None,
        eri_ovvo=None,
        eri_oovv=None,
    )
    hf_xc = AdiabaticDensityFunctional(
        name="hf_exact_exchange",
        energy_density_fn=lambda rho: jnp.zeros_like(rho),
        exact_exchange_fraction=1.0,
    )
    cached_solver = RestrictedCasidaTDDFT(
        molecule=reference,
        xc_functional=hf_xc,
    )
    uncached_solver = RestrictedCasidaTDDFT(
        molecule=uncached,
        xc_functional=hf_xc,
    )

    cached = cached_solver.build_matrices()
    direct = uncached_solver.build_matrices()
    np.testing.assert_allclose(
        np.asarray(cached.a_matrix),
        np.asarray(direct.a_matrix),
        atol=1e-8,
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(cached.b_matrix),
        np.asarray(direct.b_matrix),
        atol=1e-8,
        rtol=1e-8,
    )


def test_restricted_casida_davidson_matches_dense():
    _pyscf_or_skip()
    mf = _make_water_reference("hf")
    reference = restricted_reference_from_pyscf(mf)
    hf_xc = AdiabaticDensityFunctional(
        name="hf_exact_exchange",
        energy_density_fn=lambda rho: jnp.zeros_like(rho),
        exact_exchange_fraction=1.0,
    )
    dense_solver = RestrictedCasidaTDDFT(
        molecule=reference,
        xc_functional=hf_xc,
        eigensolver="dense",
    )
    davidson_solver = RestrictedCasidaTDDFT(
        molecule=reference,
        xc_functional=hf_xc,
        eigensolver="davidson",
        davidson_max_iter=80,
        davidson_max_subspace=24,
    )
    dense = dense_solver.kernel(nstates=4)
    davidson = davidson_solver.kernel(nstates=4)
    np.testing.assert_allclose(
        np.asarray(davidson.excitation_energies),
        np.asarray(dense.excitation_energies),
        atol=1e-6,
        rtol=1e-6,
    )


def test_restricted_tda_davidson_matches_dense():
    _pyscf_or_skip()
    mf = _make_water_reference("hf")
    reference = restricted_reference_from_pyscf(mf)
    hf_xc = AdiabaticDensityFunctional(
        name="hf_exact_exchange",
        energy_density_fn=lambda rho: jnp.zeros_like(rho),
        exact_exchange_fraction=1.0,
    )
    dense_solver = RestrictedCasidaTDDFT(
        molecule=reference,
        xc_functional=hf_xc,
        eigensolver="dense",
    )
    davidson_solver = RestrictedCasidaTDDFT(
        molecule=reference,
        xc_functional=hf_xc,
        eigensolver="davidson",
        davidson_max_iter=80,
        davidson_max_subspace=24,
    )
    dense = dense_solver.tda(nstates=4)
    davidson = davidson_solver.tda(nstates=4)
    np.testing.assert_allclose(
        np.asarray(davidson.excitation_energies),
        np.asarray(dense.excitation_energies),
        atol=1e-6,
        rtol=1e-6,
    )


def test_restricted_b3lyp_h2_pyscf_matrices_are_solved_correctly_by_local_casida():
    _pyscf_or_skip()
    mf = _make_h2_reference("b3lyp")
    td = mf.TDDFT()
    td.nstates = 1
    td.kernel()
    ref_energies = np.asarray(td.e, dtype=float)
    ref_a, ref_b = td.get_ab()

    reference = restricted_reference_from_pyscf(mf)
    mo_energy = np.asarray(reference.mo_energy, dtype=float)
    if mo_energy.ndim == 2:
        mo_energy = mo_energy[0]
    delta_eps = mo_energy[reference.nocc :] - mo_energy[: reference.nocc][:, None]
    matrices = TDDFTMatrices(
        orbital_energy_differences=jnp.asarray(delta_eps),
        a_matrix=jnp.asarray(ref_a),
        b_matrix=jnp.asarray(ref_b),
    )

    result = solve_casida(matrices, nstates=1, eigensolver="dense")
    np.testing.assert_allclose(
        np.asarray(result.excitation_energies),
        ref_energies[:1],
        atol=1e-7,
        rtol=1e-7,
    )


def test_restricted_b3lyp_h2_response_matrices_are_close_to_pyscf_reference():
    _pyscf_or_skip()
    mf = _make_h2_reference("b3lyp")
    td = mf.TDDFT()
    td.nstates = 1
    td.kernel()
    ref_a, ref_b = td.get_ab()

    reference = restricted_reference_from_pyscf(mf)
    solver = RestrictedCasidaTDDFT(
        molecule=reference,
        xc_functional=SemilocalResponseFunctional("b3lyp"),
    )
    matrices = solver.build_matrices()

    np.testing.assert_allclose(
        np.asarray(matrices.a_matrix, dtype=float),
        np.asarray(ref_a, dtype=float),
        atol=2e-3,
        rtol=2e-3,
    )
    np.testing.assert_allclose(
        np.asarray(matrices.b_matrix, dtype=float),
        np.asarray(ref_b, dtype=float),
        atol=2e-3,
        rtol=2e-3,
    )


def test_restricted_pbe0_h2_response_matrices_match_pyscf_reference():
    _pyscf_or_skip()
    mf = _make_h2_reference("pbe0")
    td = mf.TDDFT()
    td.nstates = 1
    td.kernel()
    ref_a, ref_b = td.get_ab()

    reference = restricted_reference_from_pyscf(mf)
    solver = RestrictedCasidaTDDFT(
        molecule=reference,
        xc_functional=SemilocalResponseFunctional("pbe0"),
    )
    matrices = solver.build_matrices()
    result = solver.kernel(nstates=1)

    np.testing.assert_allclose(
        np.asarray(matrices.a_matrix, dtype=float),
        np.asarray(ref_a, dtype=float),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(matrices.b_matrix, dtype=float),
        np.asarray(ref_b, dtype=float),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(result.excitation_energies, dtype=float),
        np.asarray(td.e, dtype=float)[:1],
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("xc", ["hf", "b3lyp"])
def test_transition_dipoles_and_oscillator_strengths_match_pyscf_xy(xc: str):
    _pyscf_or_skip()
    mf = _make_water_reference(xc)
    td = mf.TDDFT()
    td.nstates = 6
    td.kernel()

    reference = restricted_reference_from_pyscf(mf)
    result = _pyscf_xy_to_td_result(td)

    pred_mu = np.asarray(transition_dipoles(reference, result), dtype=float)
    pred_f = np.asarray(oscillator_strengths(reference, result), dtype=float)
    ref_mu = np.asarray(td.transition_dipole(), dtype=float)
    ref_f = np.asarray(td.oscillator_strength(), dtype=float)

    n = min(ref_mu.shape[0], pred_mu.shape[0], ref_f.size, pred_f.size, 6)
    assert n >= 3
    np.testing.assert_allclose(pred_mu[:n], ref_mu[:n], atol=2e-7, rtol=1e-6)
    np.testing.assert_allclose(pred_f[:n], ref_f[:n], atol=2e-7, rtol=1e-6)


def test_restricted_b3lyp_water_excitation_energies_match_pyscf_reference():
    _pyscf_or_skip()
    mf = _make_water_reference("b3lyp")
    td = mf.TDDFT()
    td.nstates = 4
    td.kernel()

    reference = restricted_reference_from_pyscf(mf)
    solver = RestrictedCasidaTDDFT(
        molecule=reference,
        xc_functional=SemilocalResponseFunctional("b3lyp"),
    )
    result = solver.kernel(nstates=4)

    pred_energies = np.asarray(result.excitation_energies, dtype=float)
    pred_osc = np.asarray(oscillator_strengths(reference, result), dtype=float)
    ref_energies = np.asarray(td.e, dtype=float)[:4]
    ref_osc = np.asarray(td.oscillator_strength(), dtype=float)[:4]

    assert pred_energies.shape == (4,)
    np.testing.assert_allclose(pred_energies, ref_energies, atol=8e-4, rtol=2e-3)
    np.testing.assert_allclose(pred_osc, ref_osc, atol=2e-3, rtol=2e-2)
