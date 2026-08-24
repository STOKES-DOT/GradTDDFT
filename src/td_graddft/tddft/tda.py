from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp

from .eigensolvers import PYSCF_TD_DAVIDSON_MAX_CYCLE
from .eigensolvers import PYSCF_TD_DAVIDSON_TOL
from .eigensolvers import PYSCF_TD_POSITIVE_EIG_THRESHOLD
from .eigensolvers import _davidson_search_nroots
from .eigensolvers import implicit_differential_davidson_lowest_symmetric
from .eigenvector_differentiation import (
    TDAGradientMode,
    implicit_differential_davidson_lowest_symmetric_with_eigenvectors,
)
from .types import TDAResult


def _finalize_tda_result(
    eigvals,
    eigvecs,
    *,
    nroots: int,
    excitation_threshold: float,
    nocc: int,
    nvir: int,
    converged=True,
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
        converged=converged,
    )


def solve_tda_from_operator(
    delta_eps,
    vind_rows: Callable,
    diagonal,
    *,
    nstates: int | None = None,
    excitation_threshold: float = PYSCF_TD_POSITIVE_EIG_THRESHOLD,
    davidson_tol: float = PYSCF_TD_DAVIDSON_TOL,
    davidson_max_iter: int = PYSCF_TD_DAVIDSON_MAX_CYCLE,
    davidson_max_subspace: int | None = None,
    davidson_initial_guess_count: int | None = None,
    davidson_max_trial_vectors: int | None = None,
    tda_gradient_mode: TDAGradientMode = "eigenvalue_only",
    eigenvector_adjoint_tol: float = 1e-6,
    eigenvector_adjoint_max_iter: int = 64,
) -> TDAResult:
    nocc, nvir = delta_eps.shape
    dim = int(nocc * nvir)
    nroots = dim if nstates is None else min(int(nstates), dim)
    search_nroots = _davidson_search_nroots(nroots, dim)
    solver_kwargs = {
        "nroots": search_nroots,
        "size": dim,
        "diag": jnp.asarray(diagonal).reshape(dim),
        "tol": davidson_tol,
        "max_iter": davidson_max_iter,
        "max_subspace": davidson_max_subspace,
        "initial_guess_count": davidson_initial_guess_count,
        "max_trial_vectors": davidson_max_trial_vectors,
        "positive_eig_threshold": excitation_threshold,
    }
    def matrix_action(vectors):
        return vind_rows(jnp.asarray(vectors).T).T

    if tda_gradient_mode == "eigenvalue_only":
        eigvals, eigvecs, converged = (
            implicit_differential_davidson_lowest_symmetric(
                matrix_action,
                **solver_kwargs,
            )
        )
    elif tda_gradient_mode == "implicit_eigenvector":
        eigvals, eigvecs, converged = (
            implicit_differential_davidson_lowest_symmetric_with_eigenvectors(
                matrix_action,
                eigenvector_adjoint_tol=eigenvector_adjoint_tol,
                eigenvector_adjoint_max_iter=eigenvector_adjoint_max_iter,
                **solver_kwargs,
            )
        )
    else:
        raise ValueError(f"Unsupported TDA gradient mode {tda_gradient_mode!r}.")
    return _finalize_tda_result(
        eigvals,
        eigvecs,
        nroots=nroots,
        excitation_threshold=excitation_threshold,
        nocc=nocc,
        nvir=nvir,
        converged=converged,
    )
