from types import SimpleNamespace

import jax
import numpy as np
import jax.numpy as jnp
import pytest

from td_graddft.tddft._utils import _symmetrize
from td_graddft.tddft.unrestricted import (
    UnrestrictedTDA,
    build_unrestricted_tda_operator,
    build_unrestricted_tdhf_operator,
    solve_unrestricted_casida_from_tdhf_operator,
)
import td_graddft.tddft.unrestricted as unrestricted_module


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


def _tdhf_vind(flat_a, flat_b):
    def vind(rows):
        rows = jnp.asarray(rows).reshape(-1, 2 * flat_a.shape[0])
        x = rows[:, : flat_a.shape[0]]
        y = rows[:, flat_a.shape[0] :]
        upper = x @ jnp.asarray(flat_a).T + y @ jnp.asarray(flat_b).T
        lower = -(x @ jnp.asarray(flat_b).T + y @ jnp.asarray(flat_a).T)
        return jnp.concatenate([upper, lower], axis=-1)

    return vind


def _operator_matrix(vind, dim):
    return vind(jnp.eye(dim)).T


def _tdhf_operator_matrices(vind, dim):
    eye = jnp.eye(dim)
    zeros = jnp.zeros_like(eye)
    a_cols = vind(jnp.concatenate([eye, zeros], axis=-1))[:, :dim]
    b_cols = vind(jnp.concatenate([zeros, eye], axis=-1))[:, :dim]
    return a_cols.T, b_cols.T


def test_unrestricted_response_uses_spin_resolved_kernel_actions():
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

    tda_vind, diagonal, de_a, de_b = build_unrestricted_tda_operator(
        reference,
        SpinResolvedXC(),
    )
    tdhf_vind, _, _ = build_unrestricted_tdhf_operator(reference, SpinResolvedXC())
    rho_ov = np.asarray(reference.ao[:, 0] * reference.ao[:, 1])
    weights = np.asarray(reference.grid.weights)
    expected_aa = float(np.sum(weights * 2.0 * rho_ov * rho_ov))
    expected_ab = float(np.sum(weights * 3.0 * rho_ov * rho_ov))
    expected_bb = float(np.sum(weights * 5.0 * rho_ov * rho_ov))

    eye = jnp.eye(2)
    a_columns = tda_vind(eye)
    b_columns = tdhf_vind(jnp.concatenate([jnp.zeros_like(eye), eye], axis=1))[:, :2]

    assert np.isclose(float(diagonal[0]), float(de_a.reshape(-1)[0]), atol=1e-10)
    assert np.isclose(float(diagonal[1]), float(de_b.reshape(-1)[0]), atol=1e-10)
    assert np.isclose(float(a_columns[1, 0]), expected_ab, atol=1e-10)
    assert np.isclose(float(b_columns[0, 0]), expected_aa, atol=1e-10)
    assert np.isclose(float(b_columns[1, 0]), expected_ab, atol=1e-10)
    assert np.isclose(float(b_columns[1, 1]), expected_bb, atol=1e-10)


def test_unrestricted_response_rejects_scalar_kernel_fallback():
    reference = _toy_unrestricted_reference()

    class ScalarKernelXC:
        exact_exchange_fraction = 0.0

        @staticmethod
        def local_kernel(density):
            return 4.0 * jnp.ones_like(density)

    with pytest.raises(ValueError, match="requires spin-resolved XC kernels"):
        build_unrestricted_tda_operator(reference, ScalarKernelXC())


def test_unrestricted_tda_allows_empty_beta_occupied_channel():
    reference = _toy_unrestricted_reference()
    reference = reference.__class__(
        **{
            **reference.__dict__,
            "mo_occ": jnp.asarray([[1.0, 0.0], [0.0, 0.0]]),
            "rdm1": jnp.asarray(
                [
                    [[1.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ]
            ),
        }
    )

    result = UnrestrictedTDA(reference).kernel(nstates=1)

    assert result.excitation_energies.shape == (1,)
    assert result.amplitudes_alpha.shape == (1, 1, 1)
    assert result.amplitudes_beta.shape[0] == 1
    assert result.amplitudes_beta.shape[1] == 0


def test_unrestricted_tda_operator_is_jittable_with_static_spin_counts():
    reference = _toy_unrestricted_reference()
    reference = reference.__class__(
        **{
            **reference.__dict__,
            "mo_occ": jnp.asarray([[1.0, 0.0], [0.0, 0.0]]),
            "rdm1": jnp.asarray(
                [
                    [[1.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ]
            ),
            "nocc_alpha": 1,
            "nocc_beta": 0,
        }
    )

    @jax.jit
    def _build(mo_occ):
        molecule = reference.__class__(**{**reference.__dict__, "mo_occ": mo_occ})
        vind, diagonal, _, _ = build_unrestricted_tda_operator(molecule)
        return diagonal, vind(jnp.ones((1, diagonal.shape[0]), dtype=diagonal.dtype))

    diagonal, action = _build(reference.mo_occ)

    assert diagonal.shape == (1,)
    assert action.shape == (1, 1)
    assert np.all(np.isfinite(np.asarray(action)))


def test_unrestricted_hfx_nu_hybrid_response_uses_standard_ao_exchange(monkeypatch):
    reference = _toy_unrestricted_reference()
    reference = reference.__class__(
        **{
            **reference.__dict__,
            "exact_exchange_fraction": 0.5,
            "hfx_nu": jnp.zeros((1, reference.ao.shape[0], 2, 2), dtype=reference.ao.dtype),
        }
    )

    calls = {"jk": 0}
    original_jk = unrestricted_module._jk_from_full_eri

    def _count_combined_jk(*args, **kwargs):
        calls["jk"] += 1
        return original_jk(*args, **kwargs)

    monkeypatch.setattr(unrestricted_module, "_jk_from_full_eri", _count_combined_jk)

    vind, diagonal, _, _ = build_unrestricted_tda_operator(reference)
    rows = jnp.ones((1, int(diagonal.size)), dtype=diagonal.dtype)

    assert np.all(np.isfinite(np.asarray(vind(rows))))
    assert calls["jk"] > 0


def test_unrestricted_df_hybrid_response_uses_factorized_mo_action(monkeypatch):
    nmo = 4
    naux = 3
    raw = jnp.arange(naux * nmo * nmo, dtype=jnp.float64).reshape(naux, nmo, nmo) / 29.0
    factors = 0.5 * (raw + jnp.swapaxes(raw, -1, -2))
    rep_tensor = jnp.einsum("Qpq,Qrs->pqrs", factors, factors)
    mo_coeff = jnp.stack([jnp.eye(nmo), jnp.eye(nmo)], axis=0)
    mo_occ = jnp.asarray([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    mo_energy = jnp.asarray([[-0.8, -0.3, 0.4, 0.9], [-0.7, 0.2, 0.6, 1.0]])
    rdm1 = jnp.stack([jnp.diag(mo_occ[0]), jnp.diag(mo_occ[1])], axis=0)
    full = SimpleNamespace(
        ao=jnp.eye(nmo),
        grid=SimpleNamespace(weights=jnp.ones((nmo,))),
        rep_tensor=rep_tensor,
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        rdm1=rdm1,
        exact_exchange_fraction=0.25,
        nocc_alpha=2,
        nocc_beta=1,
    )
    df = SimpleNamespace(**{**full.__dict__, "rep_tensor": jnp.zeros((0, 0, 0, 0)), "df_factors": factors})

    full_tda_vind, full_diag, _, _ = build_unrestricted_tda_operator(full)
    full_tdhf_vind, _, _ = build_unrestricted_tdhf_operator(full)
    full_tda_matrix = _operator_matrix(full_tda_vind, int(full_diag.size))
    full_a, full_b = _tdhf_operator_matrices(full_tdhf_vind, int(full_diag.size))

    def fail_transition_density(*args, **kwargs):
        raise AssertionError("DF unrestricted response should not build AO transition density")

    monkeypatch.setattr(unrestricted_module, "_unrestricted_transition_density", fail_transition_density)

    df_tda_vind, df_diag, _, _ = build_unrestricted_tda_operator(df)
    df_tdhf_vind, _, _ = build_unrestricted_tdhf_operator(df)
    df_a, df_b = _tdhf_operator_matrices(df_tdhf_vind, int(df_diag.size))

    assert np.allclose(
        np.asarray(_operator_matrix(df_tda_vind, int(df_diag.size))),
        np.asarray(full_tda_matrix),
        atol=1e-10,
    )
    assert np.allclose(np.asarray(df_a), np.asarray(full_a), atol=1e-10)
    assert np.allclose(np.asarray(df_b), np.asarray(full_b), atol=1e-10)


def test_unrestricted_tda_solver_is_jittable_with_static_nstates():
    reference = _toy_unrestricted_reference()
    reference = reference.__class__(
        **{
            **reference.__dict__,
            "mo_occ": jnp.asarray([[1.0, 0.0], [0.0, 0.0]]),
            "rdm1": jnp.asarray(
                [
                    [[1.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ]
            ),
            "nocc_alpha": 1,
            "nocc_beta": 0,
        }
    )

    @jax.jit
    def _solve(mo_occ):
        molecule = reference.__class__(**{**reference.__dict__, "mo_occ": mo_occ})
        result = UnrestrictedTDA(molecule).kernel(nstates=1)
        return result.excitation_energies, result.amplitudes_alpha, result.amplitudes_beta

    energies, amplitudes_alpha, amplitudes_beta = _solve(reference.mo_occ)

    assert energies.shape == (1,)
    assert amplitudes_alpha.shape == (1, 1, 1)
    assert amplitudes_beta.shape == (1, 0, 2)
    assert np.all(np.isfinite(np.asarray(energies)))


def test_unrestricted_casida_davidson_matches_cholesky_reference():
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

    result = solve_unrestricted_casida_from_tdhf_operator(
        de_a,
        de_b,
        _tdhf_vind(flat_a, flat_b),
        nstates=2,
        matrix_eps=1e-10,
    )

    a_plus_b = _symmetrize(flat_a + flat_b)
    a_minus_b = _symmetrize(flat_a - flat_b)
    factor = jnp.linalg.cholesky(a_minus_b + 1e-10 * jnp.eye(2, dtype=flat_a.dtype))
    ref_casida = _symmetrize(factor.T @ a_plus_b @ factor)
    ref_w2, _ = jnp.linalg.eigh(ref_casida)
    ref_w = jnp.sqrt(jnp.maximum(ref_w2, 0.0))

    np.testing.assert_allclose(
        np.asarray(result.excitation_energies),
        np.asarray(ref_w),
        atol=1e-6,
        rtol=1e-6,
    )
