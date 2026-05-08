import numpy as np
import jax
import jax.numpy as jnp

from td_graddft.tddft.casida import solve_casida
from td_graddft.tddft.eigensolvers import davidson_lowest_symmetric
from td_graddft.tddft.tda import solve_tda
from td_graddft.tddft.types import TDDFTMatrices, TDAResult, TDDFTResult


def test_davidson_restart_matches_dense_on_random_symmetric_matrix():
    rng = np.random.default_rng(0)
    dim = 40
    matrix = rng.normal(size=(dim, dim)).astype(np.float32)
    matrix = 0.5 * (matrix + matrix.T)

    ref_eigvals, _ = np.linalg.eigh(matrix)
    eigvals, _, converged = davidson_lowest_symmetric(
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


def test_davidson_callable_matches_dense_on_random_symmetric_matrix():
    rng = np.random.default_rng(1)
    dim = 48
    matrix = rng.normal(size=(dim, dim)).astype(np.float32)
    matrix = 0.5 * (matrix + matrix.T)

    ref_eigvals, _ = np.linalg.eigh(matrix)
    eigvals, _, converged = davidson_lowest_symmetric(
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


def test_restricted_tda_and_casida_davidson_operator_match_dense():
    rng = np.random.default_rng(2)
    nocc, nvir = 8, 9
    dim = nocc * nvir

    delta_eps = np.linspace(0.6, 3.2, dim, dtype=np.float32).reshape(nocc, nvir)
    noise = rng.normal(size=(dim, dim)).astype(np.float32)
    sym = 0.5 * (noise + noise.T)
    flat_b = 0.015 * sym
    flat_a = np.diag(np.ravel(delta_eps)) + 0.05 * sym + 0.25 * np.eye(dim, dtype=np.float32)

    # Ensure A-B stays positive definite for the Casida square root.
    min_eig = np.linalg.eigvalsh(flat_a - flat_b).min()
    if min_eig <= 1e-3:
        flat_a = flat_a + (1e-3 - min_eig + 0.05) * np.eye(dim, dtype=np.float32)

    matrices = TDDFTMatrices(
        orbital_energy_differences=jnp.asarray(delta_eps),
        a_matrix=jnp.asarray(flat_a.reshape(nocc, nvir, nocc, nvir)),
        b_matrix=jnp.asarray(flat_b.reshape(nocc, nvir, nocc, nvir)),
    )

    dense_tda = solve_tda(matrices, nstates=4, eigensolver="dense")
    davidson_tda = solve_tda(
        matrices,
        nstates=4,
        eigensolver="davidson",
        davidson_tol=1e-5,
        davidson_max_iter=120,
        davidson_max_subspace=24,
    )
    np.testing.assert_allclose(
        np.asarray(davidson_tda.excitation_energies),
        np.asarray(dense_tda.excitation_energies),
        atol=2e-5,
        rtol=2e-5,
    )

    dense_casida = solve_casida(matrices, nstates=4, eigensolver="dense")
    davidson_casida = solve_casida(
        matrices,
        nstates=4,
        eigensolver="davidson",
        davidson_tol=1e-5,
        davidson_max_iter=120,
        davidson_max_subspace=24,
    )
    np.testing.assert_allclose(
        np.asarray(davidson_casida.excitation_energies),
        np.asarray(dense_casida.excitation_energies),
        atol=2e-5,
        rtol=2e-5,
    )
    assert davidson_casida.casida_matrix is None


def test_restricted_tda_and_casida_results_are_jittable_pytrees():
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
    matrices = TDDFTMatrices(
        orbital_energy_differences=delta_eps,
        a_matrix=flat_a.reshape(2, 2, 2, 2),
        b_matrix=flat_b.reshape(2, 2, 2, 2),
    )

    jit_tda = jax.jit(lambda mats: solve_tda(mats, nstates=2, eigensolver="dense"))
    jit_casida = jax.jit(
        lambda mats: solve_casida(mats, nstates=2, eigensolver="dense")
    )

    tda_result = jit_tda(matrices)
    casida_result = jit_casida(matrices)

    assert isinstance(tda_result, TDAResult)
    assert isinstance(casida_result, TDDFTResult)
    assert tda_result.excitation_energies.shape == (2,)
    assert tda_result.amplitudes.shape == (2, 2, 2)
    assert casida_result.excitation_energies.shape == (2,)
    assert casida_result.x_amplitudes.shape == (2, 2, 2)
    assert casida_result.y_amplitudes.shape == (2, 2, 2)


def test_davidson_callable_is_jittable():
    rng = np.random.default_rng(3)
    dim = 72
    matrix = rng.normal(size=(dim, dim)).astype(np.float32)
    matrix = 0.5 * (matrix + matrix.T) + 0.5 * np.eye(dim, dtype=np.float32)
    diag = jnp.diag(jnp.asarray(matrix))

    compiled = jax.jit(
        lambda d: davidson_lowest_symmetric(
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
            lambda d: davidson_lowest_symmetric(
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


def test_davidson_tda_and_casida_are_jittable():
    rng = np.random.default_rng(4)
    nocc, nvir = 8, 9
    dim = nocc * nvir
    delta_eps = np.linspace(0.7, 2.9, dim, dtype=np.float32).reshape(nocc, nvir)
    noise = rng.normal(size=(dim, dim)).astype(np.float32)
    sym = 0.5 * (noise + noise.T)
    flat_b = 0.01 * sym
    flat_a = np.diag(np.ravel(delta_eps)) + 0.03 * sym + 0.2 * np.eye(dim, dtype=np.float32)

    min_eig = np.linalg.eigvalsh(flat_a - flat_b).min()
    if min_eig <= 1e-3:
        flat_a = flat_a + (1e-3 - min_eig + 0.05) * np.eye(dim, dtype=np.float32)

    matrices = TDDFTMatrices(
        orbital_energy_differences=jnp.asarray(delta_eps),
        a_matrix=jnp.asarray(flat_a.reshape(nocc, nvir, nocc, nvir)),
        b_matrix=jnp.asarray(flat_b.reshape(nocc, nvir, nocc, nvir)),
    )

    jit_tda = jax.jit(
        lambda mats: solve_tda(
            mats,
            nstates=4,
            eigensolver="davidson",
            davidson_tol=1e-5,
            davidson_max_iter=80,
            davidson_max_subspace=20,
        )
    )
    jit_casida = jax.jit(
        lambda mats: solve_casida(
            mats,
            nstates=4,
            eigensolver="davidson",
            davidson_tol=1e-5,
            davidson_max_iter=80,
            davidson_max_subspace=20,
        )
    )

    tda_result = jit_tda(matrices)
    casida_result = jit_casida(matrices)

    assert isinstance(tda_result, TDAResult)
    assert isinstance(casida_result, TDDFTResult)
    assert tda_result.excitation_energies.shape == (4,)
    assert tda_result.amplitudes.shape == (4, nocc, nvir)
    assert casida_result.excitation_energies.shape == (4,)
    assert casida_result.x_amplitudes.shape == (4, nocc, nvir)
    assert casida_result.y_amplitudes.shape == (4, nocc, nvir)


def test_auto_tda_and_casida_are_jittable_when_auto_selects_davidson():
    rng = np.random.default_rng(5)
    nocc, nvir = 10, 10
    dim = nocc * nvir
    delta_eps = np.linspace(0.8, 3.4, dim, dtype=np.float32).reshape(nocc, nvir)
    noise = rng.normal(size=(dim, dim)).astype(np.float32)
    sym = 0.5 * (noise + noise.T)
    flat_b = 0.008 * sym
    flat_a = np.diag(np.ravel(delta_eps)) + 0.025 * sym + 0.18 * np.eye(dim, dtype=np.float32)

    min_eig = np.linalg.eigvalsh(flat_a - flat_b).min()
    if min_eig <= 1e-3:
        flat_a = flat_a + (1e-3 - min_eig + 0.05) * np.eye(dim, dtype=np.float32)

    matrices = TDDFTMatrices(
        orbital_energy_differences=jnp.asarray(delta_eps),
        a_matrix=jnp.asarray(flat_a.reshape(nocc, nvir, nocc, nvir)),
        b_matrix=jnp.asarray(flat_b.reshape(nocc, nvir, nocc, nvir)),
    )

    dense_tda = solve_tda(matrices, nstates=4, eigensolver="dense")
    dense_casida = solve_casida(matrices, nstates=4, eigensolver="dense")

    jit_tda = jax.jit(
        lambda mats: solve_tda(
            mats,
            nstates=4,
            eigensolver="auto",
            davidson_tol=1e-5,
            davidson_max_iter=100,
            davidson_max_subspace=24,
        )
    )
    jit_casida = jax.jit(
        lambda mats: solve_casida(
            mats,
            nstates=4,
            eigensolver="auto",
            davidson_tol=1e-5,
            davidson_max_iter=100,
            davidson_max_subspace=24,
        )
    )

    tda_result = jit_tda(matrices)
    casida_result = jit_casida(matrices)

    np.testing.assert_allclose(
        np.asarray(tda_result.excitation_energies),
        np.asarray(dense_tda.excitation_energies),
        atol=2e-5,
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(casida_result.excitation_energies),
        np.asarray(dense_casida.excitation_energies),
        atol=2e-5,
        rtol=2e-5,
    )
