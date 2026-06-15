from dataclasses import dataclass, replace

import numpy as np
import pytest
import jax.numpy as jnp

from pyscf_reference import restricted_reference_from_pyscf
from td_graddft.spectra import oscillator_strengths, transition_dipoles
from td_graddft.tddft._semilocal_response import SemilocalResponseFunctional
from td_graddft.tddft import RestrictedCasidaTDDFT
from td_graddft.tddft.casida import solve_casida_from_tdhf_operator
from td_graddft.tddft.types import TDDFTResult
from td_graddft.xc_backend.jax_xc_adapter import MissingJAXXCError, load_jax_xc


@dataclass(frozen=True)
class _HFOnlyResponseFunctional:
    name: str
    exact_exchange_fraction: float = 0.0

    def __init__(self, name, energy_density_fn=None, exact_exchange_fraction=0.0):
        del energy_density_fn
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "exact_exchange_fraction", exact_exchange_fraction)

    def local_kernel(self, density):
        return jnp.zeros_like(density)


def _pyscf_or_skip():
    try:
        from pyscf import dft, gto  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("PySCF is required for TDDFT comparison tests.")


def _jax_xc_or_skip():
    try:
        load_jax_xc()
    except MissingJAXXCError:
        pytest.skip("jax_xc is required for semilocal TDDFT comparison tests.")


def _operator_matrices(vind, dim: int) -> tuple[np.ndarray, np.ndarray]:
    eye = jnp.eye(dim)
    zeros = jnp.zeros_like(eye)
    a_cols = vind(jnp.concatenate([eye, zeros], axis=-1))[:, :dim]
    b_cols = vind(jnp.concatenate([zeros, eye], axis=-1))[:, :dim]
    return np.asarray(a_cols.T), np.asarray(b_cols.T)


def _matrix_tdhf_vind(flat_a, flat_b):
    def vind(rows):
        rows = jnp.asarray(rows).reshape(-1, 2 * flat_a.shape[0])
        x = rows[:, : flat_a.shape[0]]
        y = rows[:, flat_a.shape[0] :]
        upper = x @ jnp.asarray(flat_a).T + y @ jnp.asarray(flat_b).T
        lower = -(x @ jnp.asarray(flat_b).T + y @ jnp.asarray(flat_a).T)
        return jnp.concatenate([upper, lower], axis=-1)

    return vind


def _dense_tda_reference(flat_a: np.ndarray, nstates: int) -> np.ndarray:
    flat_a = 0.5 * (flat_a + flat_a.T)
    return np.linalg.eigvalsh(flat_a)[:nstates]


def _dense_casida_reference(
    flat_a: np.ndarray,
    flat_b: np.ndarray,
    nstates: int,
    *,
    matrix_eps: float = 1e-10,
) -> np.ndarray:
    flat_a = 0.5 * (flat_a + flat_a.T)
    flat_b = 0.5 * (flat_b + flat_b.T)
    dim = flat_a.shape[0]
    factor = np.linalg.cholesky(flat_a - flat_b + matrix_eps * np.eye(dim))
    w2 = np.linalg.eigvalsh(factor.T @ (flat_a + flat_b) @ factor)
    return np.sqrt(np.maximum(w2[:nstates], 0.0))


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
    hf_xc = _HFOnlyResponseFunctional(
        name="hf_exact_exchange",
        energy_density_fn=lambda rho: jnp.zeros_like(rho),
        exact_exchange_fraction=1.0,
    )
    solver = RestrictedCasidaTDDFT(
        molecule=reference,
        xc_functional=hf_xc,
    )
    dim = int(np.prod(np.asarray(ref_a).shape[:2]))
    pred_a, pred_b = _operator_matrices(solver.gen_tdhf_vind(), dim)
    pred_a = pred_a.reshape(np.asarray(ref_a).shape)
    pred_b = pred_b.reshape(np.asarray(ref_b).shape)
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
    hf_xc = _HFOnlyResponseFunctional(
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

    dim = int(reference.nocc * (reference.mo_coeff.shape[-1] - reference.nocc))
    cached_a, cached_b = _operator_matrices(cached_solver.gen_tdhf_vind(), dim)
    direct_a, direct_b = _operator_matrices(uncached_solver.gen_tdhf_vind(), dim)
    np.testing.assert_allclose(
        cached_a,
        direct_a,
        atol=1e-8,
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        cached_b,
        direct_b,
        atol=1e-8,
        rtol=1e-8,
    )


def test_restricted_casida_davidson_matches_numpy_reference():
    _pyscf_or_skip()
    mf = _make_water_reference("hf")
    reference = restricted_reference_from_pyscf(mf)
    hf_xc = _HFOnlyResponseFunctional(
        name="hf_exact_exchange",
        energy_density_fn=lambda rho: jnp.zeros_like(rho),
        exact_exchange_fraction=1.0,
    )
    davidson_solver = RestrictedCasidaTDDFT(
        molecule=reference,
        xc_functional=hf_xc,
        eigensolver="davidson",
        davidson_max_iter=80,
        davidson_max_subspace=24,
    )
    dim = int(reference.nocc * (reference.mo_coeff.shape[-1] - reference.nocc))
    flat_a, flat_b = _operator_matrices(davidson_solver.gen_tdhf_vind(), dim)
    ref = _dense_casida_reference(flat_a, flat_b, 4)
    davidson = davidson_solver.kernel(nstates=4)
    np.testing.assert_allclose(
        np.asarray(davidson.excitation_energies),
        ref,
        atol=1e-6,
        rtol=1e-6,
    )


def test_restricted_tda_davidson_matches_numpy_reference():
    _pyscf_or_skip()
    mf = _make_water_reference("hf")
    reference = restricted_reference_from_pyscf(mf)
    hf_xc = _HFOnlyResponseFunctional(
        name="hf_exact_exchange",
        energy_density_fn=lambda rho: jnp.zeros_like(rho),
        exact_exchange_fraction=1.0,
    )
    davidson_solver = RestrictedCasidaTDDFT(
        molecule=reference,
        xc_functional=hf_xc,
        eigensolver="davidson",
        davidson_max_iter=80,
        davidson_max_subspace=24,
    )
    dim = int(reference.nocc * (reference.mo_coeff.shape[-1] - reference.nocc))
    flat_a, _ = _operator_matrices(davidson_solver.gen_tdhf_vind(), dim)
    ref = _dense_tda_reference(flat_a, 4)
    davidson = davidson_solver.tda(nstates=4)
    np.testing.assert_allclose(
        np.asarray(davidson.excitation_energies),
        ref,
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
    flat_a = np.asarray(ref_a).reshape(delta_eps.size, delta_eps.size)
    flat_b = np.asarray(ref_b).reshape(delta_eps.size, delta_eps.size)
    result = solve_casida_from_tdhf_operator(
        jnp.asarray(delta_eps),
        _matrix_tdhf_vind(flat_a, flat_b),
        nstates=1,
        davidson_max_iter=80,
        davidson_max_subspace=24,
    )
    np.testing.assert_allclose(
        np.asarray(result.excitation_energies),
        ref_energies[:1],
        atol=1e-7,
        rtol=1e-7,
    )


def test_restricted_b3lyp_h2_response_matrices_are_close_to_pyscf_reference():
    _pyscf_or_skip()
    _jax_xc_or_skip()
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
    dim = int(reference.nocc * (reference.mo_coeff.shape[-1] - reference.nocc))
    pred_a, pred_b = _operator_matrices(solver.gen_tdhf_vind(), dim)

    np.testing.assert_allclose(
        np.asarray(pred_a.reshape(np.asarray(ref_a).shape), dtype=float),
        np.asarray(ref_a, dtype=float),
        atol=2e-3,
        rtol=2e-3,
    )
    np.testing.assert_allclose(
        np.asarray(pred_b.reshape(np.asarray(ref_b).shape), dtype=float),
        np.asarray(ref_b, dtype=float),
        atol=2e-3,
        rtol=2e-3,
    )


def test_restricted_pbe0_h2_response_matrices_match_pyscf_reference():
    _pyscf_or_skip()
    _jax_xc_or_skip()
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
    dim = int(reference.nocc * (reference.mo_coeff.shape[-1] - reference.nocc))
    pred_a, pred_b = _operator_matrices(solver.gen_tdhf_vind(), dim)
    result = solver.kernel(nstates=1)

    np.testing.assert_allclose(
        np.asarray(pred_a.reshape(np.asarray(ref_a).shape), dtype=float),
        np.asarray(ref_a, dtype=float),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(pred_b.reshape(np.asarray(ref_b).shape), dtype=float),
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
    _jax_xc_or_skip()
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
