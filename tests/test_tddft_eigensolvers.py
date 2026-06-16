import numpy as np
import jax
import jax.numpy as jnp
import pytest

from td_graddft.tddft.casida import solve_casida_from_tdhf_operator
from td_graddft.tddft.eigensolvers import implicit_differential_davidson_lowest_symmetric
from td_graddft.tddft.tda import solve_tda_from_operator
from td_graddft.tddft.types import TDAResult, TDDFTResult
from td_graddft.tddft.unrestricted import (
    UnrestrictedTDAResult,
    UnrestrictedTDDFTResult,
    solve_unrestricted_casida_from_tdhf_operator,
    solve_unrestricted_tda_from_operator,
)


def _numpy_tda_reference(flat_a: np.ndarray, nstates: int) -> np.ndarray:
    return np.linalg.eigvalsh(0.5 * (flat_a + flat_a.T))[:nstates]


def _numpy_casida_reference(
    flat_a: np.ndarray,
    flat_b: np.ndarray,
    nstates: int,
    *,
    matrix_eps: float = 1e-10,
) -> np.ndarray:
    a_plus_b = 0.5 * (flat_a + flat_a.T) + 0.5 * (flat_b + flat_b.T)
    a_minus_b = 0.5 * (flat_a + flat_a.T) - 0.5 * (flat_b + flat_b.T)
    factor = np.linalg.cholesky(a_minus_b + matrix_eps * np.eye(a_minus_b.shape[0]))
    w2 = np.linalg.eigvalsh(factor.T @ a_plus_b @ factor)
    return np.sqrt(np.maximum(w2[:nstates], 0.0))


def _tda_vind(flat_a):
    def vind(rows):
        rows = jnp.asarray(rows).reshape(-1, flat_a.shape[0])
        return rows @ jnp.asarray(flat_a).T

    return vind


def _tdhf_vind(flat_a, flat_b):
    def vind(rows):
        rows = jnp.asarray(rows).reshape(-1, 2 * flat_a.shape[0])
        x = rows[:, : flat_a.shape[0]]
        y = rows[:, flat_a.shape[0] :]
        upper = x @ jnp.asarray(flat_a).T + y @ jnp.asarray(flat_b).T
        lower = -(x @ jnp.asarray(flat_b).T + y @ jnp.asarray(flat_a).T)
        return jnp.concatenate([upper, lower], axis=-1)

    return vind


def test_operator_solvers_track_extra_davidson_roots_before_truncation():
    flat_a = np.asarray(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, -9.5, 0.0],
            [0.0, -9.5, 10.0, 0.0],
            [0.0, 0.0, 0.0, 3.0],
        ],
        dtype=np.float64,
    )
    flat_b = np.zeros_like(flat_a)
    delta_eps = jnp.asarray([[2.0, 10.0, 10.0, 3.0]], dtype=jnp.float64)

    tda = solve_tda_from_operator(
        delta_eps,
        _tda_vind(flat_a),
        jnp.diag(jnp.asarray(flat_a)),
        nstates=1,
        davidson_tol=1e-8,
        davidson_max_iter=20,
    )
    casida = solve_casida_from_tdhf_operator(
        delta_eps,
        _tdhf_vind(flat_a, flat_b),
        nstates=1,
        davidson_tol=1e-8,
        davidson_max_iter=20,
    )

    np.testing.assert_allclose(
        np.asarray(tda.excitation_energies),
        np.asarray([0.5]),
        atol=1e-8,
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(casida.excitation_energies),
        np.asarray([0.5]),
        atol=1e-8,
        rtol=1e-8,
    )


def test_restricted_casida_raises_when_davidson_does_not_converge():
    flat_a = np.diag(np.asarray([0.8, 1.2, 1.6, 2.0], dtype=np.float64))
    flat_b = np.zeros_like(flat_a)
    delta_eps = jnp.asarray([[0.8, 1.2], [1.6, 2.0]], dtype=jnp.float64)

    with pytest.raises(RuntimeError, match="Davidson TDDFT solver did not converge"):
        solve_casida_from_tdhf_operator(
            delta_eps,
            _tdhf_vind(flat_a, flat_b),
            nstates=1,
            davidson_max_iter=0,
        )


def test_davidson_restart_matches_numpy_on_random_symmetric_matrix():
    rng = np.random.default_rng(0)
    dim = 40
    matrix = rng.normal(size=(dim, dim)).astype(np.float32)
    matrix = 0.5 * (matrix + matrix.T)

    ref_eigvals, _ = np.linalg.eigh(matrix)
    eigvals, _, converged = implicit_differential_davidson_lowest_symmetric(
        jnp.asarray(matrix),
        nroots=4,
        tol=1e-5,
        max_iter=120,
        max_subspace=12,
        collapse_subspace=8,
    )

    assert converged
    np.testing.assert_allclose(
        np.asarray(eigvals),
        ref_eigvals[:4],
        atol=2e-4,
        rtol=2e-4,
    )


def test_davidson_callable_matches_numpy_on_random_symmetric_matrix():
    rng = np.random.default_rng(1)
    dim = 48
    matrix = rng.normal(size=(dim, dim)).astype(np.float32)
    matrix = 0.5 * (matrix + matrix.T)

    ref_eigvals, _ = np.linalg.eigh(matrix)
    eigvals, _, converged = implicit_differential_davidson_lowest_symmetric(
        lambda vectors: jnp.asarray(matrix) @ vectors,
        nroots=5,
        size=dim,
        diag=jnp.diag(jnp.asarray(matrix)),
        tol=1e-5,
        max_iter=120,
        max_subspace=14,
        collapse_subspace=10,
    )

    assert converged
    np.testing.assert_allclose(
        np.asarray(eigvals),
        ref_eigvals[:5],
        atol=2e-4,
        rtol=2e-4,
    )


def test_restricted_tda_and_casida_operator_match_numpy_reference():
    rng = np.random.default_rng(2)
    nocc, nvir = 8, 9
    dim = nocc * nvir

    delta_eps = np.linspace(0.6, 3.2, dim, dtype=np.float32).reshape(nocc, nvir)
    noise = rng.normal(size=(dim, dim)).astype(np.float32)
    sym = 0.5 * (noise + noise.T)
    flat_b = 0.015 * sym
    flat_a = np.diag(np.ravel(delta_eps)) + 0.05 * sym + 0.25 * np.eye(dim, dtype=np.float32)

    min_eig = np.linalg.eigvalsh(flat_a - flat_b).min()
    if min_eig <= 1e-3:
        flat_a = flat_a + (1e-3 - min_eig + 0.05) * np.eye(dim, dtype=np.float32)

    ref_tda = _numpy_tda_reference(flat_a, 4)
    davidson_tda = solve_tda_from_operator(
        jnp.asarray(delta_eps),
        _tda_vind(flat_a),
        jnp.diag(jnp.asarray(flat_a)),
        nstates=4,
        davidson_tol=1e-5,
        davidson_max_iter=120,
        davidson_max_subspace=24,
    )
    np.testing.assert_allclose(
        np.asarray(davidson_tda.excitation_energies),
        ref_tda,
        atol=2e-5,
        rtol=2e-5,
    )

    ref_casida = _numpy_casida_reference(flat_a, flat_b, 4)
    davidson_casida = solve_casida_from_tdhf_operator(
        jnp.asarray(delta_eps),
        _tdhf_vind(flat_a, flat_b),
        nstates=4,
        davidson_tol=1e-5,
        davidson_max_iter=120,
        davidson_max_subspace=24,
    )
    np.testing.assert_allclose(
        np.asarray(davidson_casida.excitation_energies),
        ref_casida,
        atol=2e-5,
        rtol=2e-5,
    )


def test_unrestricted_tda_and_casida_operator_match_numpy_reference():
    rng = np.random.default_rng(22)
    de_a = np.asarray([[0.7, 1.1], [1.6, 2.0]], dtype=np.float32)
    de_b = np.asarray([[0.9, 1.4]], dtype=np.float32)
    dim = de_a.size + de_b.size
    noise = rng.normal(size=(dim, dim)).astype(np.float32)
    sym = 0.5 * (noise + noise.T)
    flat_b = 0.01 * sym
    flat_a = np.diag(np.concatenate([de_a.ravel(), de_b.ravel()])) + 0.04 * sym
    flat_a = flat_a + 0.2 * np.eye(dim, dtype=np.float32)

    min_eig = np.linalg.eigvalsh(flat_a - flat_b).min()
    if min_eig <= 1e-3:
        flat_a = flat_a + (1e-3 - min_eig + 0.05) * np.eye(dim, dtype=np.float32)

    tda = solve_unrestricted_tda_from_operator(
        jnp.asarray(de_a),
        jnp.asarray(de_b),
        _tda_vind(flat_a),
        jnp.diag(jnp.asarray(flat_a)),
        nstates=3,
        davidson_tol=1e-5,
        davidson_max_iter=120,
        davidson_max_subspace=18,
    )
    np.testing.assert_allclose(
        np.asarray(tda.excitation_energies),
        _numpy_tda_reference(flat_a, 3),
        atol=2e-5,
        rtol=2e-5,
    )

    casida = solve_unrestricted_casida_from_tdhf_operator(
        jnp.asarray(de_a),
        jnp.asarray(de_b),
        _tdhf_vind(flat_a, flat_b),
        nstates=3,
        davidson_tol=1e-5,
        davidson_max_iter=120,
        davidson_max_subspace=18,
    )
    np.testing.assert_allclose(
        np.asarray(casida.excitation_energies),
        _numpy_casida_reference(flat_a, flat_b, 3),
        atol=2e-5,
        rtol=2e-5,
    )


def test_operator_results_are_jittable_pytrees():
    delta_eps = jnp.asarray([[0.8, 1.1], [1.4, 1.9]], dtype=jnp.float32)
    flat_a = jnp.asarray(
        [
            [0.95, 0.02, 0.00, 0.01],
            [0.02, 1.20, 0.03, 0.00],
            [0.00, 0.03, 1.55, 0.02],
            [0.01, 0.00, 0.02, 2.05],
        ],
        dtype=jnp.float32,
    )
    flat_b = 0.02 * jnp.asarray(
        [
            [0.10, 0.01, 0.00, 0.00],
            [0.01, 0.12, 0.01, 0.00],
            [0.00, 0.01, 0.14, 0.01],
            [0.00, 0.00, 0.01, 0.16],
        ],
        dtype=jnp.float32,
    )

    @jax.jit
    def _solve(a, b):
        tda = solve_tda_from_operator(
            delta_eps,
            _tda_vind(a),
            jnp.diag(a),
            nstates=2,
        )
        casida = solve_casida_from_tdhf_operator(
            delta_eps,
            _tdhf_vind(a, b),
            nstates=2,
        )
        return tda, casida

    tda_result, casida_result = _solve(flat_a, flat_b)

    assert isinstance(tda_result, TDAResult)
    assert isinstance(casida_result, TDDFTResult)
    assert tda_result.excitation_energies.shape == (2,)
    assert tda_result.amplitudes.shape == (2, 2, 2)
    assert casida_result.excitation_energies.shape == (2,)
    assert casida_result.x_amplitudes.shape == (2, 2, 2)
    assert casida_result.y_amplitudes.shape == (2, 2, 2)


def test_tda_operator_gradient_matches_dense_eigenvalue_perturbation():
    matrix = jnp.asarray(
        [
            [0.95, 0.02, 0.00, 0.01],
            [0.02, 1.20, 0.03, 0.00],
            [0.00, 0.03, 1.55, 0.02],
            [0.01, 0.00, 0.02, 2.05],
        ],
        dtype=jnp.float32,
    )
    delta_eps = jnp.asarray([[0.8, 1.1], [1.4, 1.9]], dtype=jnp.float32)

    def davidson_energy(raw_matrix):
        sym = 0.5 * (raw_matrix + raw_matrix.T)
        return solve_tda_from_operator(
            delta_eps,
            _tda_vind(sym),
            jnp.diag(sym),
            nstates=1,
        ).excitation_energies[0]

    def dense_energy(raw_matrix):
        sym = 0.5 * (raw_matrix + raw_matrix.T)
        return jnp.linalg.eigvalsh(sym)[0]

    np.testing.assert_allclose(
        np.asarray(jax.grad(davidson_energy)(matrix)),
        np.asarray(jax.grad(dense_energy)(matrix)),
        atol=5e-5,
        rtol=5e-5,
    )


def test_unrestricted_operator_results_are_jittable_pytrees():
    de_a = jnp.asarray([[0.8, 1.1]], dtype=jnp.float32)
    de_b = jnp.asarray([[0.9, 1.2]], dtype=jnp.float32)
    flat_a = jnp.asarray(
        [
            [1.0, 0.02, 0.01, 0.00],
            [0.02, 1.2, 0.00, 0.01],
            [0.01, 0.00, 1.1, 0.02],
            [0.00, 0.01, 0.02, 1.3],
        ],
        dtype=jnp.float32,
    )
    flat_b = 0.01 * jnp.ones_like(flat_a)

    @jax.jit
    def _solve(a, b):
        tda = solve_unrestricted_tda_from_operator(
            de_a,
            de_b,
            _tda_vind(a),
            jnp.diag(a),
            nstates=2,
        )
        casida = solve_unrestricted_casida_from_tdhf_operator(
            de_a,
            de_b,
            _tdhf_vind(a, b),
            nstates=2,
        )
        return tda, casida

    tda_result, casida_result = _solve(flat_a, flat_b)

    assert isinstance(tda_result, UnrestrictedTDAResult)
    assert isinstance(casida_result, UnrestrictedTDDFTResult)
    assert tda_result.excitation_energies.shape == (2,)
    assert tda_result.amplitudes_alpha.shape == (2, 1, 2)
    assert tda_result.amplitudes_beta.shape == (2, 1, 2)
    assert casida_result.excitation_energies.shape == (2,)
    assert casida_result.x_amplitudes_alpha.shape == (2, 1, 2)
    assert casida_result.y_amplitudes_beta.shape == (2, 1, 2)


def test_davidson_callable_is_jittable():
    rng = np.random.default_rng(3)
    dim = 72
    matrix = rng.normal(size=(dim, dim)).astype(np.float32)
    matrix = 0.5 * (matrix + matrix.T) + 0.5 * np.eye(dim, dtype=np.float32)
    diag = jnp.diag(jnp.asarray(matrix))

    compiled = jax.jit(
        lambda d: implicit_differential_davidson_lowest_symmetric(
            lambda vectors: jnp.asarray(matrix) @ vectors,
            nroots=4,
            size=dim,
            diag=d,
            tol=1e-5,
            max_iter=80,
            max_subspace=20,
            collapse_subspace=12,
        )
    )
    eigvals, eigvecs, converged = compiled(diag)

    assert bool(np.asarray(converged))
    assert eigvals.shape == (4,)
    assert eigvecs.shape == (dim, 4)


def test_davidson_callable_is_jittable_with_x64_enabled():
    rng = np.random.default_rng(30)
    dim = 48
    noise = rng.normal(size=(dim, dim)).astype(np.float64)
    sym = 0.5 * (noise + noise.T)
    matrix = np.diag(np.linspace(0.5, 4.0, dim, dtype=np.float64)) + 0.02 * sym
    diag = jnp.diag(jnp.asarray(matrix))

    enable_x64 = getattr(jax, "enable_x64", None)
    if enable_x64 is None:
        enable_x64 = jax.experimental.enable_x64

    with enable_x64(True):
        compiled = jax.jit(
            lambda d: implicit_differential_davidson_lowest_symmetric(
                lambda vectors: jnp.asarray(matrix) @ vectors,
                nroots=4,
                size=dim,
                diag=d,
                tol=1e-8,
                max_iter=80,
                max_subspace=20,
                collapse_subspace=12,
            )
        )
        eigvals, eigvecs, converged = compiled(diag)

    assert eigvals.shape == (4,)
    assert eigvecs.shape == (dim, 4)
    assert np.asarray(converged).shape == ()
    assert np.all(np.isfinite(np.asarray(eigvals)))
    assert np.all(np.isfinite(np.asarray(eigvecs)))
