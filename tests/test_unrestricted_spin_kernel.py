from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp
import pytest

from td_graddft.tddft._utils import _matrix_power_symmetric, _symmetrize
from td_graddft.tddft.unrestricted import (
    UnrestrictedResponseMatrices,
    build_unrestricted_response_matrices,
    solve_unrestricted_casida,
)


def _toy_unrestricted_reference():
    ao = jnp.asarray(
        [
            [1.0, 0.2],
            [0.8, -0.3],
            [0.6, 0.4],
        ]
    )
    weights = jnp.asarray([0.5, 0.3, 0.2])
    mo_coeff = jnp.stack([jnp.eye(2), jnp.eye(2)], axis=0)
    mo_occ = jnp.asarray([[1.0, 0.0], [1.0, 0.0]])
    mo_energy = jnp.asarray([[-0.60, 0.20], [-0.55, 0.25]])
    dm_a = jnp.asarray([[1.0, 0.0], [0.0, 0.0]])
    dm_b = jnp.asarray([[1.0, 0.0], [0.0, 0.0]])
    return SimpleNamespace(
        ao=ao,
        grid=SimpleNamespace(weights=weights),
        rep_tensor=jnp.zeros((2, 2, 2, 2)),
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        rdm1=jnp.stack([dm_a, dm_b], axis=0),
        exact_exchange_fraction=0.0,
    )


def test_unrestricted_response_uses_spin_resolved_kernel_blocks():
    reference = _toy_unrestricted_reference()

    class SpinResolvedXC:
        exact_exchange_fraction = 0.0

        @staticmethod
        def spin_local_kernel(rho_a, rho_b):
            del rho_b
            return (
                2.0 * jnp.ones_like(rho_a),
                3.0 * jnp.ones_like(rho_a),
                5.0 * jnp.ones_like(rho_a),
            )

    response = build_unrestricted_response_matrices(reference, SpinResolvedXC())
    rho_ov = np.asarray(reference.ao[:, 0] * reference.ao[:, 1])
    w = np.asarray(reference.grid.weights)
    expected_aa = float(np.sum(w * 2.0 * rho_ov * rho_ov))
    expected_ab = float(np.sum(w * 3.0 * rho_ov * rho_ov))
    expected_bb = float(np.sum(w * 5.0 * rho_ov * rho_ov))

    de_a = float(response.orbital_energy_differences_alpha[0, 0])
    de_b = float(response.orbital_energy_differences_beta[0, 0])
    assert np.isclose(float(response.a_aa[0, 0, 0, 0] - de_a), expected_aa, atol=1e-10)
    assert np.isclose(float(response.a_ab[0, 0, 0, 0]), expected_ab, atol=1e-10)
    assert np.isclose(float(response.a_bb[0, 0, 0, 0] - de_b), expected_bb, atol=1e-10)
    assert np.isclose(float(response.b_aa[0, 0, 0, 0]), expected_aa, atol=1e-10)
    assert np.isclose(float(response.b_ab[0, 0, 0, 0]), expected_ab, atol=1e-10)
    assert np.isclose(float(response.b_bb[0, 0, 0, 0]), expected_bb, atol=1e-10)


def test_unrestricted_response_rejects_scalar_kernel_fallback():
    reference = _toy_unrestricted_reference()

    class ScalarKernelXC:
        exact_exchange_fraction = 0.0

        @staticmethod
        def local_kernel(density):
            return 4.0 * jnp.ones_like(density)

    with pytest.raises(ValueError, match="requires spin-resolved XC kernels"):
        build_unrestricted_response_matrices(reference, ScalarKernelXC())


def test_unrestricted_casida_cholesky_metric_matches_symmetric_sqrt_reference():
    de_a = jnp.asarray([[0.8]])
    de_b = jnp.asarray([[1.1]])
    flat_a = jnp.asarray(
        [
            [1.05, 0.08],
            [0.08, 1.32],
        ]
    )
    flat_b = jnp.asarray(
        [
            [0.12, 0.03],
            [0.03, 0.09],
        ]
    )
    zeros = jnp.zeros((1, 1, 1, 1))
    matrices = UnrestrictedResponseMatrices(
        orbital_energy_differences_alpha=de_a,
        orbital_energy_differences_beta=de_b,
        a_aa=zeros,
        a_ab=zeros,
        a_ba=zeros,
        a_bb=zeros,
        b_aa=zeros,
        b_ab=zeros,
        b_ba=zeros,
        b_bb=zeros,
        a_matrix=flat_a,
        b_matrix=flat_b,
    )

    result = solve_unrestricted_casida(matrices, nstates=2, matrix_eps=1e-10)

    a_plus_b = _symmetrize(flat_a + flat_b)
    a_minus_b = _symmetrize(flat_a - flat_b)
    sqrt_a_minus_b = _matrix_power_symmetric(a_minus_b, 0.5, 1e-10)
    ref_casida = _symmetrize(sqrt_a_minus_b @ a_plus_b @ sqrt_a_minus_b)
    ref_w2, _ = jnp.linalg.eigh(ref_casida)
    ref_w = jnp.sqrt(jnp.maximum(ref_w2, 0.0))

    np.testing.assert_allclose(
        np.asarray(result.excitation_energies),
        np.asarray(ref_w),
        atol=1e-6,
        rtol=1e-6,
    )
