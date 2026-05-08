from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import core as jax_core

from .eigensolvers import davidson_lowest_symmetric
from ._utils import _symmetrize
from .types import TDDFTMatrices, TDAResult


def _prefer_dense_eigensolve(dim: int, nroots: int) -> bool:
    # For small TD spaces dense diagonalization is typically both faster and
    # more accurate than an iterative solve, even if Davidson was requested.
    return dim <= 64 or nroots >= max(8, dim // 3)


def _prefer_dense_auto_eigensolve(dim: int, nroots: int) -> bool:
    # In jit-heavy workflows dense diagonalization remains faster and more
    # reliable than Davidson for moderately sized response spaces.
    return _prefer_dense_eigensolve(dim, nroots) or dim <= 2048


def _is_traced_convergence_flag(value) -> bool:
    return isinstance(value, jax_core.Tracer)


def _lowest_dense_eigenpairs(matrix, *, nroots: int):
    eigvals, eigvecs = jnp.linalg.eigh(matrix)
    return eigvals[:nroots], eigvecs[:, :nroots]


def _finalize_tda_result(
    eigvals,
    eigvecs,
    *,
    nroots: int,
    excitation_threshold: float,
    nocc: int,
    nvir: int,
    a_matrix,
) -> TDAResult:
    valid = eigvals > excitation_threshold
    order = jnp.argsort(jnp.where(valid, eigvals, jnp.inf))
    keep = order[:nroots]
    mask = valid[keep]
    energies = jnp.where(mask, eigvals[keep], 0.0)
    amplitudes = jnp.sqrt(0.5) * eigvecs[:, keep].T.reshape(-1, nocc, nvir)
    amplitudes = amplitudes * mask[:, None, None]
    return TDAResult(
        excitation_energies=energies,
        amplitudes=amplitudes,
        a_matrix=a_matrix,
    )


def solve_tda_from_operator(
    delta_eps,
    vind_rows: Callable,
    diagonal,
    *,
    nstates: int | None = None,
    excitation_threshold: float = 1e-7,
    davidson_tol: float = 1e-6,
    davidson_max_iter: int = 60,
    davidson_max_subspace: int | None = None,
    a_matrix=None,
) -> TDAResult:
    nocc, nvir = delta_eps.shape
    dim = int(nocc * nvir)
    nroots = dim if nstates is None else min(int(nstates), dim)
    eigvals, eigvecs, converged = davidson_lowest_symmetric(
        lambda vectors: vind_rows(jnp.asarray(vectors).T).T,
        nroots=nroots,
        size=dim,
        diag=jnp.asarray(diagonal).reshape(dim),
        tol=davidson_tol,
        max_iter=davidson_max_iter,
        max_subspace=davidson_max_subspace,
    )
    if not _is_traced_convergence_flag(converged) and not bool(converged):
        raise RuntimeError("Davidson TDA solver did not converge.")
    return _finalize_tda_result(
        eigvals,
        eigvecs,
        nroots=nroots,
        excitation_threshold=excitation_threshold,
        nocc=nocc,
        nvir=nvir,
        a_matrix=a_matrix,
    )


def solve_tda_from_a_matrix(
    delta_eps,
    a_matrix,
    *,
    nstates: int | None = None,
    excitation_threshold: float = 1e-7,
) -> TDAResult:
    """Solve TDA directly from a prebuilt A matrix."""

    delta_eps = jnp.asarray(delta_eps)
    nocc, nvir = delta_eps.shape
    dim = int(nocc * nvir)
    nroots = dim if nstates is None else min(int(nstates), dim)
    flat_a = _symmetrize(jnp.asarray(a_matrix).reshape(dim, dim))
    eigvals, eigvecs = jnp.linalg.eigh(flat_a)
    return _finalize_tda_result(
        eigvals,
        eigvecs,
        nroots=nroots,
        excitation_threshold=excitation_threshold,
        nocc=nocc,
        nvir=nvir,
        a_matrix=jnp.asarray(a_matrix),
    )


def solve_tda(
    matrices: TDDFTMatrices,
    *,
    nstates: int | None = None,
    excitation_threshold: float = 1e-7,
    eigensolver: str = "auto",
    davidson_tol: float = 1e-6,
    davidson_max_iter: int = 60,
    davidson_max_subspace: int | None = None,
) -> TDAResult:
    """Solve the Hermitian TDA eigenproblem."""

    delta_eps = matrices.orbital_energy_differences
    nocc, nvir = delta_eps.shape
    flat_a = _symmetrize(matrices.a_matrix.reshape(nocc * nvir, nocc * nvir))
    dim = int(flat_a.shape[0])
    nroots = dim if nstates is None else min(int(nstates), dim)

    mode = str(eigensolver).lower()
    use_davidson = False
    if mode == "davidson":
        use_davidson = not _prefer_dense_eigensolve(dim, nroots)
    elif mode == "dense":
        use_davidson = False
    elif mode == "auto":
        use_davidson = (
            nstates is not None
            and not _prefer_dense_auto_eigensolve(dim, nroots)
            and dim >= 96
            and nroots <= min(24, max(1, dim // 3))
        )
    else:
        raise ValueError(
            f"Unsupported eigensolver={eigensolver!r}. Choose one of {{'auto', 'dense', 'davidson'}}."
        )

    if use_davidson:
        davidson_eigvals, davidson_eigvecs, converged = davidson_lowest_symmetric(
            lambda vectors: flat_a @ vectors,
            nroots=nroots,
            size=dim,
            diag=jnp.diag(flat_a),
            tol=davidson_tol,
            max_iter=davidson_max_iter,
            max_subspace=davidson_max_subspace,
        )
        if mode == "auto":
            if _is_traced_convergence_flag(converged):
                eigvals, eigvecs = jax.lax.cond(
                    converged,
                    lambda _: (davidson_eigvals, davidson_eigvecs),
                    lambda _: _lowest_dense_eigenpairs(flat_a, nroots=nroots),
                    operand=None,
                )
            elif bool(converged):
                eigvals, eigvecs = davidson_eigvals, davidson_eigvecs
            else:
                eigvals, eigvecs = _lowest_dense_eigenpairs(flat_a, nroots=nroots)
        else:
            eigvals, eigvecs = davidson_eigvals, davidson_eigvecs
            if not _is_traced_convergence_flag(converged) and not bool(converged):
                raise RuntimeError("Davidson TDA solver did not converge.")
    else:
        eigvals, eigvecs = jnp.linalg.eigh(flat_a)

    return _finalize_tda_result(
        eigvals,
        eigvecs,
        nroots=nroots,
        excitation_threshold=excitation_threshold,
        nocc=nocc,
        nvir=nvir,
        a_matrix=matrices.a_matrix,
    )
