from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import core as jax_core

from .eigensolvers import PYSCF_TD_DAVIDSON_MAX_CYCLE
from .eigensolvers import PYSCF_TD_DAVIDSON_TOL
from .eigensolvers import PYSCF_TD_POSITIVE_EIG_THRESHOLD
from .eigensolvers import _davidson_search_nroots
from .eigensolvers import implicit_differential_davidson_lowest_symmetric
from .types import TDAResult


def _is_traced_convergence_flag(value) -> bool:
    return isinstance(value, jax_core.Tracer)


def _finalize_tda_result(
    eigvals,
    eigvecs,
    *,
    nroots: int,
    excitation_threshold: float,
    nocc: int,
    nvir: int,
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
) -> TDAResult:
    nocc, nvir = delta_eps.shape
    dim = int(nocc * nvir)
    nroots = dim if nstates is None else min(int(nstates), dim)
    search_nroots = _davidson_search_nroots(nroots, dim)
    eigvals, eigvecs, converged = implicit_differential_davidson_lowest_symmetric(
        lambda vectors: vind_rows(jnp.asarray(vectors).T).T,
        nroots=search_nroots,
        size=dim,
        diag=jnp.asarray(diagonal).reshape(dim),
        tol=davidson_tol,
        max_iter=davidson_max_iter,
        max_subspace=davidson_max_subspace,
        positive_eig_threshold=excitation_threshold,
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
    )
