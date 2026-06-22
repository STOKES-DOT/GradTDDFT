from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jaxtyping import Array
from collections.abc import Callable

from ._utils import _symmetrize


_MATMUL_PRECISION = jax.lax.Precision.HIGHEST
_DEFAULT_MATMUL_PRECISION = "highest"

__all__ = [
    "implicit_differential_davidson_lowest_symmetric",
    "implicit_differential_davidson_lowest_tdhf",
]

PYSCF_TD_DAVIDSON_TOL = 1e-5
PYSCF_TD_DAVIDSON_MAX_CYCLE = 100
PYSCF_TD_POSITIVE_EIG_THRESHOLD = 1e-3


def _solver_dtype(dtype: jnp.dtype) -> jnp.dtype:
    dtype = jnp.dtype(dtype)
    if bool(jax.config.jax_enable_x64):
        if dtype == jnp.dtype(jnp.float32):
            return jnp.dtype(jnp.float64)
        if dtype == jnp.dtype(jnp.complex64):
            return jnp.dtype(jnp.complex128)
    return dtype


def _davidson_search_nroots(nroots: int, dim: int) -> int:
    nroots = max(1, min(int(nroots), int(dim)))
    return nroots


def _davidson_max_subspace(nroots: int, dim: int, max_subspace: int | None) -> int:
    if max_subspace is None:
        return int(dim)
    return min(int(dim), max(int(max_subspace), int(nroots) + 2))


def _residual_tol_with_dtype_slack(tol: float, dtype: jnp.dtype) -> Array:
    tol_arr = jnp.asarray(tol, dtype=dtype)
    return tol_arr + jnp.asarray(100.0, dtype=dtype) * jnp.asarray(
        jnp.finfo(dtype).eps,
        dtype=dtype,
    )


def _matmul(lhs: Array, rhs: Array) -> Array:
    return jnp.matmul(lhs, rhs, precision=_MATMUL_PRECISION)


def _safe_preconditioner_denominator(values: Array, floor: float, level_shift: float) -> Array:
    values = jnp.asarray(values)
    values = values - jnp.asarray(level_shift, dtype=values.dtype)
    sign = jnp.where(values < 0.0, -1.0, 1.0)
    return jnp.where(jnp.abs(values) < floor, sign * floor, values)


def _resolve_symmetric_linear_operator(
    matrix_or_matvec: Array | Callable[[Array], Array],
    *,
    size: int | None,
    diag: Array | None,
) -> tuple[Callable[[Array], Array], Array, int, jnp.dtype]:
    if callable(matrix_or_matvec):
        if size is None:
            raise ValueError("size is required when _davidson_lowest_symmetric receives a matvec.")
        if diag is None:
            raise ValueError("diag is required when _davidson_lowest_symmetric receives a matvec.")
        op_diag = jnp.asarray(diag)
        dim = int(size)
        dtype = _solver_dtype(op_diag.dtype)
        op_diag = op_diag.astype(dtype)

        def apply(vectors: Array) -> Array:
            arr = jnp.asarray(vectors, dtype=dtype)
            squeeze = arr.ndim == 1
            arr = arr.reshape(dim, -1)
            with jax.default_matmul_precision(_DEFAULT_MATMUL_PRECISION):
                out = jnp.asarray(matrix_or_matvec(arr), dtype=dtype).reshape(dim, -1)
            return out[:, 0] if squeeze else out

        return apply, op_diag.reshape(dim), dim, dtype

    matrix = _symmetrize(jnp.asarray(matrix_or_matvec))
    dim = int(matrix.shape[0])
    dtype = _solver_dtype(matrix.dtype)
    matrix = matrix.astype(dtype)

    def apply(vectors: Array) -> Array:
        arr = jnp.asarray(vectors, dtype=dtype)
        squeeze = arr.ndim == 1
        arr = arr.reshape(dim, -1)
        out = _matmul(matrix, arr)
        return out[:, 0] if squeeze else out

    return apply, jnp.diag(matrix), dim, dtype


def _davidson_lowest_symmetric(
    matrix_or_matvec: Array | Callable[[Array], Array],
    *,
    nroots: int,
    size: int | None = None,
    diag: Array | None = None,
    tol: float = PYSCF_TD_DAVIDSON_TOL,
    max_iter: int = PYSCF_TD_DAVIDSON_MAX_CYCLE,
    max_subspace: int | None = None,
    collapse_subspace: int | None = None,
    initial_guess_count: int | None = None,
    max_trial_vectors: int | None = None,
    positive_eig_threshold: float | None = None,
    preconditioner_floor: float = 1e-8,
    preconditioner_level_shift: float = 0.0,
    orth_eps: float = 1e-10,
) -> tuple[Array, Array, Array]:
    """Approximate the lowest eigenpairs of a Hermitian matrix with Davidson."""

    index_dtype = jnp.asarray(0).dtype

    apply, diag, dim, dtype = _resolve_symmetric_linear_operator(
        matrix_or_matvec,
        size=size,
        diag=diag,
    )
    if dim == 0:
        return (
            jnp.zeros((0,), dtype=dtype),
            jnp.zeros((0, 0), dtype=dtype),
            jnp.asarray(True),
        )

    nroots = max(1, min(int(nroots), dim))
    max_subspace = _davidson_max_subspace(nroots, dim, max_subspace)
    if collapse_subspace is None:
        collapse_subspace = min(dim, max(2 * nroots, nroots + 4))
    else:
        collapse_subspace = min(dim, max(int(collapse_subspace), nroots))

    if initial_guess_count is None:
        guess_count = nroots
    else:
        guess_count = max(nroots, int(initial_guess_count))
    if max_trial_vectors is None:
        trial_count = nroots
    else:
        trial_count = max(nroots, int(max_trial_vectors))
    trial_count = min(max_subspace, trial_count)
    guess_dim = min(dim, max_subspace, guess_count)
    guess_idx = jnp.argsort(diag)[:guess_dim]
    guess_basis = jnp.eye(dim, dtype=dtype)[:, guess_idx]
    guess_basis, _ = jnp.linalg.qr(guess_basis, mode="reduced")
    guess_abasis = apply(guess_basis)

    basis = jnp.zeros((dim, max_subspace), dtype=dtype)
    abasis = jnp.zeros((dim, max_subspace), dtype=dtype)
    basis = jax.lax.dynamic_update_slice(basis, guess_basis, (0, 0))
    abasis = jax.lax.dynamic_update_slice(abasis, guess_abasis, (0, 0))
    heff0 = _symmetrize(_matmul(basis.T.conj(), abasis))

    active_mask = jnp.zeros((max_subspace,), dtype=bool)
    active_mask = jax.lax.dynamic_update_slice(
        active_mask,
        jnp.ones((guess_dim,), dtype=bool),
        (0,),
    )
    inactive_shift = jnp.asarray(
        (jnp.maximum(jnp.max(jnp.abs(diag)), 1.0) + 1.0) * 1.0e6,
        dtype=dtype,
    )
    tol_arr = jnp.asarray(tol, dtype=dtype)
    orth_eps_arr = jnp.asarray(orth_eps, dtype=dtype)
    if positive_eig_threshold is None:
        positive_threshold_arr = None
    else:
        positive_threshold_arr = jnp.asarray(positive_eig_threshold, dtype=dtype)
    basis_dim0 = jnp.asarray(guess_dim, dtype=index_dtype)
    best_theta0 = jnp.zeros((nroots,), dtype=dtype)
    best_vecs0 = jnp.zeros((dim, nroots), dtype=dtype)
    best_residual0 = jnp.asarray(jnp.inf, dtype=dtype)
    converged0 = jnp.asarray(False)
    done0 = jnp.asarray(False)

    def _orthogonalize_against(vector: Array, columns: Array) -> Array:
        return vector - _matmul(columns, _matmul(columns.T.conj(), vector))

    def _append_new_columns(
        basis_in: Array,
        abasis_in: Array,
        heff_in: Array,
        active_mask_in: Array,
        basis_dim_in: Array,
        expand_cols: Array,
        expand_mask: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
        def body_fun(col_idx: int, carry):
            appended_cols_cur, appended_mask_cur, appended_count_cur = carry
            accept_seed = jax.lax.dynamic_index_in_dim(
                expand_mask,
                col_idx,
                axis=0,
                keepdims=False,
            )
            raw_col = jax.lax.dynamic_slice(
                expand_cols,
                (0, col_idx),
                (dim, 1),
            ).reshape(dim)
            columns_so_far = jax.lax.cond(
                appended_count_cur > 0,
                lambda cols: cols,
                lambda cols: jnp.zeros_like(cols),
                appended_cols_cur,
            )
            candidate = _orthogonalize_against(raw_col, basis_in)
            candidate = _orthogonalize_against(candidate, columns_so_far)
            candidate = _orthogonalize_against(candidate, basis_in)
            candidate = _orthogonalize_against(candidate, columns_so_far)
            cand_norm = jnp.linalg.norm(candidate)
            accept = accept_seed & (cand_norm > orth_eps_arr)
            safe_norm = jnp.where(cand_norm > orth_eps_arr, cand_norm, 1.0)
            col = (candidate / safe_norm)[:, None]

            def do_update(update_carry):
                cols_upd, mask_upd, count_upd = update_carry
                cols_upd = jax.lax.dynamic_update_slice(cols_upd, col, (0, count_upd))
                mask_upd = jax.lax.dynamic_update_slice(
                    mask_upd,
                    jnp.asarray([True]),
                    (count_upd,),
                )
                return cols_upd, mask_upd, count_upd + jnp.asarray(1, dtype=index_dtype)

            return jax.lax.cond(accept, do_update, lambda x: x, carry)

        init_cols = jnp.zeros((dim, expand_cols.shape[1]), dtype=dtype)
        init_mask = jnp.zeros((expand_cols.shape[1],), dtype=bool)
        appended_cols, appended_mask, appended = jax.lax.fori_loop(
            0,
            expand_cols.shape[1],
            body_fun,
            (init_cols, init_mask, jnp.asarray(0, dtype=index_dtype)),
        )

        appended_abasis = apply(appended_cols)

        def scatter_body(col_idx: int, carry):
            basis_cur, abasis_cur, active_mask_cur, offset = carry
            accept = jax.lax.dynamic_index_in_dim(
                appended_mask,
                col_idx,
                axis=0,
                keepdims=False,
            )
            target = basis_dim_in + offset
            col = jax.lax.dynamic_slice(appended_cols, (0, col_idx), (dim, 1))
            acol = jax.lax.dynamic_slice(appended_abasis, (0, col_idx), (dim, 1))

            def do_update(update_carry):
                basis_upd, abasis_upd, active_upd, offset_upd = update_carry
                basis_upd = jax.lax.dynamic_update_slice(basis_upd, col, (0, target))
                abasis_upd = jax.lax.dynamic_update_slice(abasis_upd, acol, (0, target))
                active_upd = jax.lax.dynamic_update_slice(
                    active_upd,
                    jnp.asarray([True]),
                    (target,),
                )
                return basis_upd, abasis_upd, active_upd, offset_upd + jnp.asarray(1, dtype=index_dtype)

            return jax.lax.cond(accept, do_update, lambda x: x, carry)

        basis_out, abasis_out, active_mask_out, _ = jax.lax.fori_loop(
            0,
            appended_cols.shape[1],
            scatter_body,
            (basis_in, abasis_in, active_mask_in, jnp.asarray(0, dtype=index_dtype)),
        )
        projected = _matmul(basis_out.T.conj(), appended_abasis)

        def heff_body(col_idx: int, carry):
            heff_cur, offset = carry
            accept = jax.lax.dynamic_index_in_dim(
                appended_mask,
                col_idx,
                axis=0,
                keepdims=False,
            )
            target = basis_dim_in + offset
            proj = jax.lax.dynamic_slice(projected, (0, col_idx), (max_subspace, 1))
            proj_row = proj.T.conj()

            def do_update(update_carry):
                heff_upd, offset_upd = update_carry
                heff_upd = jax.lax.dynamic_update_slice(heff_upd, proj, (0, target))
                heff_upd = jax.lax.dynamic_update_slice(heff_upd, proj_row, (target, 0))
                return heff_upd, offset_upd + jnp.asarray(1, dtype=index_dtype)

            return jax.lax.cond(accept, do_update, lambda x: x, carry)

        heff_out, _ = jax.lax.fori_loop(
            0,
            appended_cols.shape[1],
            heff_body,
            (heff_in, jnp.asarray(0, dtype=index_dtype)),
        )
        return basis_out, abasis_out, heff_out, active_mask_out, basis_dim_in + appended

    def _restart_subspace(
        basis_in: Array,
        eigvecs_in: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
        restart_coeff = eigvecs_in[:, :collapse_subspace]
        restarted_basis = _matmul(basis_in, restart_coeff)
        restarted_basis, _ = jnp.linalg.qr(restarted_basis, mode="reduced")
        restarted_abasis = apply(restarted_basis)

        basis_out = jnp.zeros((dim, max_subspace), dtype=dtype)
        abasis_out = jnp.zeros((dim, max_subspace), dtype=dtype)
        basis_out = jax.lax.dynamic_update_slice(basis_out, restarted_basis, (0, 0))
        abasis_out = jax.lax.dynamic_update_slice(abasis_out, restarted_abasis, (0, 0))
        heff_out = _symmetrize(_matmul(basis_out.T.conj(), abasis_out))

        active_mask_out = jnp.zeros((max_subspace,), dtype=bool)
        active_mask_out = jax.lax.dynamic_update_slice(
            active_mask_out,
            jnp.ones((collapse_subspace,), dtype=bool),
            (0,),
        )
        return (
            basis_out,
            abasis_out,
            heff_out,
            active_mask_out,
            jnp.asarray(collapse_subspace, dtype=index_dtype),
        )

    def _step(
        _iter: int,
        state: tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
        (
            basis_cur,
            abasis_cur,
            heff_cur,
            active_mask_cur,
            basis_dim_cur,
            best_theta_cur,
            best_vecs_cur,
            best_residual_cur,
            converged_cur,
            done_cur,
        ) = state

        def _do_iteration(iter_state):
            (
                basis_it,
                abasis_it,
                heff_it,
                active_mask_it,
                basis_dim_it,
                best_theta_it,
                best_vecs_it,
                best_residual_it,
                converged_it,
                _done_it,
            ) = iter_state

            mask_f = active_mask_it.astype(dtype)
            subspace = _symmetrize(heff_it)
            subspace = (
                subspace * mask_f[:, None] * mask_f[None, :]
                + inactive_shift * jnp.diag((~active_mask_it).astype(dtype))
            )
            sub_eigvals, sub_eigvecs = jnp.linalg.eigh(subspace)
            if positive_threshold_arr is None:
                root_valid = jnp.ones_like(sub_eigvals, dtype=bool)
                eig_score = sub_eigvals
            else:
                root_valid = sub_eigvals > positive_threshold_arr
                eig_score = jnp.where(
                    root_valid,
                    sub_eigvals,
                    inactive_shift + jnp.abs(sub_eigvals),
                )
            order = jnp.argsort(eig_score)
            sub_eigvals = sub_eigvals[order]
            sub_eigvecs = sub_eigvecs[:, order]
            root_valid = root_valid[order]

            candidate_theta = sub_eigvals[:trial_count]
            candidate_coeff = sub_eigvecs[:, :trial_count]
            candidate_valid = root_valid[:trial_count]
            theta = candidate_theta[:nroots]
            coeff = candidate_coeff[:, :nroots]
            selected_valid = candidate_valid[:nroots]
            vecs = _matmul(basis_it, coeff)
            avecs = _matmul(abasis_it, coeff)
            residuals = avecs - vecs * theta[None, :]
            residual_norms = jnp.linalg.norm(residuals, axis=0)
            selected_residual_norms = jnp.where(
                selected_valid,
                residual_norms,
                jnp.asarray(jnp.inf, dtype=dtype),
            )
            max_residual = jnp.max(selected_residual_norms)
            full_span = basis_dim_it >= jnp.asarray(dim, dtype=index_dtype)
            enough_roots = jnp.all(selected_valid)
            converged_now = enough_roots & (full_span | (max_residual < tol_arr))
            improve_best = (max_residual < best_residual_it) | converged_now
            best_theta_next = jnp.where(improve_best, theta, best_theta_it)
            best_vecs_next = jnp.where(improve_best, vecs, best_vecs_it)
            best_residual_next = jnp.where(improve_best, max_residual, best_residual_it)

            candidate_vecs = _matmul(basis_it, candidate_coeff)
            candidate_avecs = _matmul(abasis_it, candidate_coeff)
            candidate_residuals = (
                candidate_avecs - candidate_vecs * candidate_theta[None, :]
            )
            candidate_residual_norms = jnp.linalg.norm(candidate_residuals, axis=0)

            def _correction_body(root_idx: int, carry):
                new_cols_cur, new_mask_cur, new_count_cur = carry
                root_residual_norm = jax.lax.dynamic_index_in_dim(
                    candidate_residual_norms,
                    root_idx,
                    axis=0,
                    keepdims=False,
                )
                root_theta = jax.lax.dynamic_index_in_dim(
                    candidate_theta,
                    root_idx,
                    axis=0,
                    keepdims=False,
                )
                root_residual = jax.lax.dynamic_index_in_dim(
                    candidate_residuals,
                    root_idx,
                    axis=1,
                    keepdims=False,
                )
                root_is_valid = jax.lax.dynamic_index_in_dim(
                    candidate_valid,
                    root_idx,
                    axis=0,
                    keepdims=False,
                )
                root_needs_update = root_is_valid & (root_residual_norm > tol_arr)
                denom = _safe_preconditioner_denominator(
                    root_theta - diag,
                    preconditioner_floor,
                    preconditioner_level_shift,
                )
                correction = root_residual / denom
                correction = _orthogonalize_against(correction, basis_it)
                correction = _orthogonalize_against(correction, new_cols_cur)
                corr_norm = jnp.linalg.norm(correction)
                accept = root_needs_update & (corr_norm > orth_eps_arr)
                safe_norm = jnp.where(corr_norm > orth_eps_arr, corr_norm, 1.0)
                correction = correction / safe_norm

                def _accept_update(carry_state):
                    cols_upd, mask_upd, count_upd = carry_state
                    cols_upd = jax.lax.dynamic_update_slice(
                        cols_upd,
                        correction[:, None],
                        (0, count_upd),
                    )
                    mask_upd = jax.lax.dynamic_update_slice(
                        mask_upd,
                        jnp.asarray([True]),
                        (count_upd,),
                    )
                    return cols_upd, mask_upd, count_upd + jnp.asarray(1, dtype=index_dtype)

                return jax.lax.cond(
                    accept,
                    _accept_update,
                    lambda x: x,
                    (new_cols_cur, new_mask_cur, new_count_cur),
                )

            init_new_cols = jnp.zeros((dim, trial_count), dtype=dtype)
            init_new_mask = jnp.zeros((trial_count,), dtype=bool)
            init_new_count = jnp.asarray(0, dtype=index_dtype)
            expand_cols, expand_mask, expand_count = jax.lax.fori_loop(
                0,
                trial_count,
                _correction_body,
                (init_new_cols, init_new_mask, init_new_count),
            )

            no_new = expand_count == 0
            overflow = basis_dim_it + expand_count > jnp.asarray(max_subspace, dtype=index_dtype)

            def _keep_current(_):
                return basis_it, abasis_it, heff_it, active_mask_it, basis_dim_it

            def _grow_or_restart(_):
                return jax.lax.cond(
                    overflow,
                    lambda __: _append_new_columns(
                        *_restart_subspace(basis_it, sub_eigvecs),
                        expand_cols,
                        expand_mask,
                    ),
                    lambda __: _append_new_columns(
                        basis_it,
                        abasis_it,
                        heff_it,
                        active_mask_it,
                        basis_dim_it,
                        expand_cols,
                        expand_mask,
                    ),
                    operand=None,
                )

            basis_next, abasis_next, heff_next, active_mask_next, basis_dim_next = jax.lax.cond(
                converged_now | no_new,
                _keep_current,
                _grow_or_restart,
                operand=None,
            )
            done_next = converged_now | no_new
            converged_flag_next = converged_it | converged_now
            return (
                basis_next,
                abasis_next,
                heff_next,
                active_mask_next,
                basis_dim_next,
                best_theta_next,
                best_vecs_next,
                best_residual_next,
                converged_flag_next,
                done_next,
            )

        return jax.lax.cond(done_cur, lambda s: s, _do_iteration, state)

    final_state = jax.lax.fori_loop(
        0,
        int(max_iter),
        _step,
        (
            basis,
            abasis,
            heff0,
            active_mask,
            basis_dim0,
            best_theta0,
            best_vecs0,
            best_residual0,
            converged0,
            done0,
        ),
    )
    (
        _basis,
        _abasis,
        _heff,
        _active_mask,
        _basis_dim,
        best_theta,
        best_vecs,
        _best_residual,
        converged,
        _done,
    ) = final_state
    return best_theta, best_vecs, converged


def implicit_differential_davidson_lowest_symmetric(
    matrix_or_matvec: Array | Callable[[Array], Array],
    *,
    nroots: int,
    size: int | None = None,
    diag: Array | None = None,
    tol: float = PYSCF_TD_DAVIDSON_TOL,
    max_iter: int = PYSCF_TD_DAVIDSON_MAX_CYCLE,
    max_subspace: int | None = None,
    collapse_subspace: int | None = None,
    initial_guess_count: int | None = None,
    max_trial_vectors: int | None = None,
    positive_eig_threshold: float | None = None,
    preconditioner_floor: float = 1e-8,
    preconditioner_level_shift: float = 0.0,
    orth_eps: float = 1e-10,
) -> tuple[Array, Array, Array]:
    """Return Davidson roots with implicit eigenvalue differentiation.

    Davidson is used only as a numerical eigensolver.  The solver matvec is
    stopped for reverse-mode AD, and the differentiable value is reconstructed
    from the converged Ritz vectors through the Rayleigh quotient.
    """

    apply, diag, dim, _dtype = _resolve_symmetric_linear_operator(
        matrix_or_matvec,
        size=size,
        diag=diag,
    )

    def solver_matvec(vectors: Array) -> Array:
        return jax.lax.stop_gradient(apply(vectors))

    eigvals, eigvecs, converged = _davidson_lowest_symmetric(
        solver_matvec,
        nroots=nroots,
        size=dim,
        diag=diag,
        tol=tol,
        max_iter=max_iter,
        max_subspace=max_subspace,
        collapse_subspace=collapse_subspace,
        initial_guess_count=initial_guess_count,
        max_trial_vectors=max_trial_vectors,
        positive_eig_threshold=positive_eig_threshold,
        preconditioner_floor=preconditioner_floor,
        preconditioner_level_shift=preconditioner_level_shift,
        orth_eps=orth_eps,
    )
    eigvecs = jax.lax.stop_gradient(eigvecs)
    applied = apply(eigvecs)
    denom = jnp.maximum(
        jnp.sum(eigvecs * eigvecs, axis=0),
        jnp.asarray(1e-30, dtype=eigvecs.dtype),
    )
    eigvals = jnp.sum(eigvecs * applied, axis=0) / denom
    return eigvals, eigvecs, converged


def _tdhf_subspace_eigen_solver(
    a: Array,
    b: Array,
    sigma: Array,
    pi: Array,
    *,
    nroots: int,
    eps: Array,
) -> tuple[Array, Array, Array]:
    """JAX counterpart of PySCF's TDDFT_subspace_eigen_solver."""

    dtype = a.dtype
    dim = int(a.shape[0])
    eye = jnp.eye(dim, dtype=dtype)
    d = jnp.maximum(jnp.abs(jnp.diag(sigma)), eps)
    d_mh = d**-0.5
    s_m_p = d_mh[:, None] * (sigma - pi) * d_mh[None, :]
    lu_l, lu_u = jsp_linalg.lu(s_m_p, permute_l=True, check_finite=False)
    l_inv = jnp.linalg.inv(lu_l)
    u_inv = jnp.linalg.inv(lu_u)

    d_amb_d = d_mh[:, None] * (a - b) * d_mh[None, :]
    ggt = _symmetrize(_matmul(_matmul(u_inv.T.conj(), d_amb_d), u_inv))
    g = jnp.linalg.cholesky(ggt + eps * eye)
    g_inv = jnp.linalg.inv(g)

    d_apb_d = d_mh[:, None] * (a + b) * d_mh[None, :]
    m = _symmetrize(_matmul(_matmul(_matmul(_matmul(g.T.conj(), l_inv), d_apb_d), l_inv.T.conj()), g))
    omega2_all, z_all = jnp.linalg.eigh(m)
    order = jnp.argsort(jnp.where(omega2_all > eps, omega2_all, jnp.inf))
    omega2 = jnp.maximum(omega2_all[order][:nroots], eps)
    z = z_all[:, order][:, :nroots]
    omega = jnp.sqrt(omega2)

    x_plus_y = d_mh[:, None] * _matmul(l_inv.T.conj(), _matmul(g, z)) * omega[None, :] ** -0.5
    x_minus_y = d_mh[:, None] * _matmul(u_inv, _matmul(g_inv.T.conj(), z)) * omega[None, :] ** 0.5
    x = 0.5 * (x_plus_y + x_minus_y)
    y = x_plus_y - x
    return omega, x, y


def _davidson_lowest_tdhf(
    vind: Callable[[Array], Array],
    *,
    nroots: int,
    size: int,
    diag: Array,
    tol: float = PYSCF_TD_DAVIDSON_TOL,
    max_iter: int = PYSCF_TD_DAVIDSON_MAX_CYCLE,
    max_subspace: int | None = None,
    matrix_eps: float = 1e-10,
    preconditioner_floor: float = 1e-8,
    preconditioner_level_shift: float = 0.0,
    orth_eps: float = 1e-10,
) -> tuple[Array, Array, Array, Array]:
    """PySCF-style real TDHF/TDDFT Davidson solver in JAX.

    ``vind`` follows PySCF ``gen_tdhf_operation``: input rows are ``[X, Y]`` and
    output rows are ``[AX + BY, -(BX + AY)]``.
    """

    diag = jnp.asarray(diag)
    dim = int(size)
    dtype = _solver_dtype(diag.dtype)
    diag = diag.astype(dtype).reshape(dim)
    index_dtype = jnp.asarray(0).dtype

    if dim == 0:
        empty = jnp.zeros((0, 0), dtype=dtype)
        return jnp.zeros((0,), dtype=dtype), empty, empty, jnp.asarray(True)

    nroots = max(1, min(int(nroots), dim))
    max_subspace = _davidson_max_subspace(nroots, dim, max_subspace)

    def apply_pair(v_cols: Array, w_cols: Array) -> tuple[Array, Array]:
        rows = jnp.concatenate([v_cols.T, w_cols.T], axis=-1)
        with jax.default_matmul_precision(_DEFAULT_MATMUL_PRECISION):
            applied = jnp.asarray(vind(rows), dtype=dtype).reshape(-1, 2 * dim)
        return applied[:, :dim].T, -applied[:, dim:].T

    guess_dim = min(dim, max_subspace, nroots)
    guess_idx = jnp.argsort(diag)[:guess_dim]
    guess_v = jnp.eye(dim, dtype=dtype)[:, guess_idx]
    guess_w = jnp.zeros_like(guess_v)
    guess_u1, guess_u2 = apply_pair(guess_v, guess_w)

    v_basis = jnp.zeros((dim, max_subspace), dtype=dtype)
    w_basis = jnp.zeros((dim, max_subspace), dtype=dtype)
    u1_basis = jnp.zeros((dim, max_subspace), dtype=dtype)
    u2_basis = jnp.zeros((dim, max_subspace), dtype=dtype)
    v_basis = jax.lax.dynamic_update_slice(v_basis, guess_v, (0, 0))
    w_basis = jax.lax.dynamic_update_slice(w_basis, guess_w, (0, 0))
    u1_basis = jax.lax.dynamic_update_slice(u1_basis, guess_u1, (0, 0))
    u2_basis = jax.lax.dynamic_update_slice(u2_basis, guess_u2, (0, 0))
    active_mask = jnp.zeros((max_subspace,), dtype=bool)
    active_mask = jax.lax.dynamic_update_slice(
        active_mask,
        jnp.ones((guess_dim,), dtype=bool),
        (0,),
    )

    inactive_shift = jnp.asarray(
        (jnp.maximum(jnp.max(jnp.abs(diag)), 1.0) + 1.0) * 1.0e6,
        dtype=dtype,
    )
    tol_arr = _residual_tol_with_dtype_slack(tol, dtype)
    eps_arr = jnp.asarray(matrix_eps, dtype=dtype)
    orth_eps_arr = jnp.asarray(orth_eps, dtype=dtype)
    full_diag = jnp.concatenate([diag, -diag])

    basis_dim0 = jnp.asarray(guess_dim, dtype=index_dtype)
    best_w0 = jnp.zeros((nroots,), dtype=dtype)
    best_x0 = jnp.zeros((dim, nroots), dtype=dtype)
    best_y0 = jnp.zeros((dim, nroots), dtype=dtype)
    best_residual0 = jnp.asarray(jnp.inf, dtype=dtype)
    converged0 = jnp.asarray(False)
    done0 = jnp.asarray(False)

    def _solve_subspace(
        v_in: Array,
        w_in: Array,
        u1_in: Array,
        u2_in: Array,
        active_in: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        mask_f = active_in.astype(dtype)
        active_outer = mask_f[:, None] * mask_f[None, :]
        inactive_diag = (~active_in).astype(dtype)
        a = (
            _matmul(v_in.T.conj(), u1_in)
            + _matmul(w_in.T.conj(), u2_in)
        ) * active_outer
        b = (
            _matmul(v_in.T.conj(), u2_in)
            + _matmul(w_in.T.conj(), u1_in)
        ) * active_outer
        sigma = (
            _matmul(v_in.T.conj(), v_in)
            - _matmul(w_in.T.conj(), w_in)
        ) * active_outer
        pi = (
            _matmul(v_in.T.conj(), w_in)
            - _matmul(w_in.T.conj(), v_in)
        ) * active_outer
        a = _symmetrize(a) + inactive_shift * jnp.diag(inactive_diag)
        b = _symmetrize(b)
        sigma = _symmetrize(sigma) + jnp.diag(inactive_diag.astype(dtype))
        pi = 0.5 * (pi - pi.T.conj())
        omega, x_sub, y_sub = _tdhf_subspace_eigen_solver(
            a,
            b,
            sigma,
            pi,
            nroots=nroots,
            eps=eps_arr,
        )
        x_full = _matmul(v_in, x_sub) + _matmul(w_in, y_sub)
        y_full = _matmul(w_in, x_sub) + _matmul(v_in, y_sub)
        r_x = _matmul(u1_in, x_sub) + _matmul(u2_in, y_sub) - x_full * omega[None, :]
        r_y = _matmul(u2_in, x_sub) + _matmul(u1_in, y_sub) + y_full * omega[None, :]
        residual_norms = jnp.sqrt(
            jnp.sum(jnp.abs(r_x) ** 2, axis=0)
            + jnp.sum(jnp.abs(r_y) ** 2, axis=0)
        )
        return omega, x_full, y_full, r_x, r_y, residual_norms

    def _append_new_pairs(
        v_in: Array,
        w_in: Array,
        u1_in: Array,
        u2_in: Array,
        active_in: Array,
        basis_dim_in: Array,
        new_x: Array,
        new_y: Array,
        new_mask: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        def body_fun(col_idx: int, carry):
            x_cols, y_cols, mask_cols, count = carry
            accept_seed = jax.lax.dynamic_index_in_dim(
                new_mask,
                col_idx,
                axis=0,
                keepdims=False,
            )
            x = jax.lax.dynamic_slice(new_x, (0, col_idx), (dim, 1)).reshape(dim)
            y = jax.lax.dynamic_slice(new_y, (0, col_idx), (dim, 1)).reshape(dim)
            x = x - _matmul(v_in, _matmul(v_in.T.conj(), x))
            x = x - _matmul(w_in, _matmul(w_in.T.conj(), x))
            y = y - _matmul(w_in, _matmul(w_in.T.conj(), y))
            y = y - _matmul(v_in, _matmul(v_in.T.conj(), y))
            x = x - _matmul(x_cols, _matmul(x_cols.T.conj(), x))
            y = y - _matmul(y_cols, _matmul(y_cols.T.conj(), y))
            pair_norm = jnp.sqrt(jnp.sum(jnp.abs(x) ** 2) + jnp.sum(jnp.abs(y) ** 2))
            accept = accept_seed & (pair_norm > orth_eps_arr)
            safe_norm = jnp.where(pair_norm > orth_eps_arr, pair_norm, 1.0)
            x = x / safe_norm
            y = y / safe_norm

            def do_update(update_carry):
                x_upd, y_upd, mask_upd, count_upd = update_carry
                x_upd = jax.lax.dynamic_update_slice(x_upd, x[:, None], (0, count_upd))
                y_upd = jax.lax.dynamic_update_slice(y_upd, y[:, None], (0, count_upd))
                mask_upd = jax.lax.dynamic_update_slice(
                    mask_upd,
                    jnp.asarray([True]),
                    (count_upd,),
                )
                return x_upd, y_upd, mask_upd, count_upd + jnp.asarray(1, dtype=index_dtype)

            return jax.lax.cond(accept, do_update, lambda z: z, carry)

        init_x = jnp.zeros_like(new_x)
        init_y = jnp.zeros_like(new_y)
        init_mask = jnp.zeros((new_x.shape[1],), dtype=bool)
        x_pairs, y_pairs, pair_mask, pair_count = jax.lax.fori_loop(
            0,
            new_x.shape[1],
            body_fun,
            (init_x, init_y, init_mask, jnp.asarray(0, dtype=index_dtype)),
        )
        pair_u1, pair_u2 = apply_pair(x_pairs, y_pairs)

        def scatter_body(col_idx: int, carry):
            v_cur, w_cur, u1_cur, u2_cur, active_cur, offset = carry
            accept = jax.lax.dynamic_index_in_dim(
                pair_mask,
                col_idx,
                axis=0,
                keepdims=False,
            )
            target = basis_dim_in + offset
            x = jax.lax.dynamic_slice(x_pairs, (0, col_idx), (dim, 1))
            y = jax.lax.dynamic_slice(y_pairs, (0, col_idx), (dim, 1))
            u1 = jax.lax.dynamic_slice(pair_u1, (0, col_idx), (dim, 1))
            u2 = jax.lax.dynamic_slice(pair_u2, (0, col_idx), (dim, 1))

            def do_update(update_carry):
                v_upd, w_upd, u1_upd, u2_upd, active_upd, offset_upd = update_carry
                v_upd = jax.lax.dynamic_update_slice(v_upd, x, (0, target))
                w_upd = jax.lax.dynamic_update_slice(w_upd, y, (0, target))
                u1_upd = jax.lax.dynamic_update_slice(u1_upd, u1, (0, target))
                u2_upd = jax.lax.dynamic_update_slice(u2_upd, u2, (0, target))
                active_upd = jax.lax.dynamic_update_slice(
                    active_upd,
                    jnp.asarray([True]),
                    (target,),
                )
                return (
                    v_upd,
                    w_upd,
                    u1_upd,
                    u2_upd,
                    active_upd,
                    offset_upd + jnp.asarray(1, dtype=index_dtype),
                )

            return jax.lax.cond(accept, do_update, lambda z: z, carry)

        v_out, w_out, u1_out, u2_out, active_out, _ = jax.lax.fori_loop(
            0,
            x_pairs.shape[1],
            scatter_body,
            (
                v_in,
                w_in,
                u1_in,
                u2_in,
                active_in,
                jnp.asarray(0, dtype=index_dtype),
            ),
        )
        return v_out, w_out, u1_out, u2_out, active_out, basis_dim_in + pair_count

    def _restart_from_roots(x_full: Array, y_full: Array):
        x_seed = x_full[:, :nroots]
        y_seed = y_full[:, :nroots]
        u1_seed, u2_seed = apply_pair(x_seed, y_seed)
        v_out = jnp.zeros((dim, max_subspace), dtype=dtype)
        w_out = jnp.zeros((dim, max_subspace), dtype=dtype)
        u1_out = jnp.zeros((dim, max_subspace), dtype=dtype)
        u2_out = jnp.zeros((dim, max_subspace), dtype=dtype)
        v_out = jax.lax.dynamic_update_slice(v_out, x_seed, (0, 0))
        w_out = jax.lax.dynamic_update_slice(w_out, y_seed, (0, 0))
        u1_out = jax.lax.dynamic_update_slice(u1_out, u1_seed, (0, 0))
        u2_out = jax.lax.dynamic_update_slice(u2_out, u2_seed, (0, 0))
        active_out = jnp.zeros((max_subspace,), dtype=bool)
        active_out = jax.lax.dynamic_update_slice(
            active_out,
            jnp.ones((nroots,), dtype=bool),
            (0,),
        )
        return v_out, w_out, u1_out, u2_out, active_out, jnp.asarray(nroots, dtype=index_dtype)

    def _step(
        _iter: int,
        state: tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
        (
            v_cur,
            w_cur,
            u1_cur,
            u2_cur,
            active_cur,
            basis_dim_cur,
            best_w_cur,
            best_x_cur,
            best_y_cur,
            best_residual_cur,
            converged_cur,
            done_cur,
        ) = state

        def do_iteration(iter_state):
            (
                v_it,
                w_it,
                u1_it,
                u2_it,
                active_it,
                basis_dim_it,
                best_w_it,
                best_x_it,
                best_y_it,
                best_residual_it,
                converged_it,
                _done_it,
            ) = iter_state
            omega, x_full, y_full, r_x, r_y, residual_norms = _solve_subspace(
                v_it,
                w_it,
                u1_it,
                u2_it,
                active_it,
            )
            max_residual = jnp.max(residual_norms)
            full_span = basis_dim_it >= jnp.asarray(dim, dtype=index_dtype)
            converged_now = full_span | (max_residual <= tol_arr)
            improve_best = (max_residual < best_residual_it) | converged_now
            best_w_next = jnp.where(improve_best, omega, best_w_it)
            best_x_next = jnp.where(improve_best, x_full, best_x_it)
            best_y_next = jnp.where(improve_best, y_full, best_y_it)
            best_residual_next = jnp.where(improve_best, max_residual, best_residual_it)

            denom_base = full_diag[:, None] - (
                omega[None, :] - jnp.asarray(preconditioner_level_shift, dtype=dtype)
            )
            denom_sign = jnp.where(denom_base < 0.0, -1.0, 1.0)
            denom = jnp.where(
                jnp.abs(denom_base) < preconditioner_floor,
                denom_sign * preconditioner_floor,
                denom_base,
            )
            residual_stacked = jnp.concatenate([r_x, r_y], axis=0)
            correction = residual_stacked / denom
            new_x = correction[:dim, :]
            new_y = correction[dim:, :]
            new_mask = residual_norms > tol_arr
            no_new = ~jnp.any(new_mask)
            overflow = basis_dim_it + jnp.sum(new_mask.astype(index_dtype)) > jnp.asarray(
                max_subspace,
                dtype=index_dtype,
            )

            def keep_current(_):
                return v_it, w_it, u1_it, u2_it, active_it, basis_dim_it

            def grow_or_restart(_):
                return jax.lax.cond(
                    overflow,
                    lambda __: _append_new_pairs(
                        *_restart_from_roots(x_full, y_full),
                        new_x,
                        new_y,
                        new_mask,
                    ),
                    lambda __: _append_new_pairs(
                        v_it,
                        w_it,
                        u1_it,
                        u2_it,
                        active_it,
                        basis_dim_it,
                        new_x,
                        new_y,
                        new_mask,
                    ),
                    operand=None,
                )

            v_next, w_next, u1_next, u2_next, active_next, basis_dim_next = jax.lax.cond(
                converged_now | no_new,
                keep_current,
                grow_or_restart,
                operand=None,
            )
            done_next = converged_now | no_new
            converged_flag_next = converged_it | converged_now
            return (
                v_next,
                w_next,
                u1_next,
                u2_next,
                active_next,
                basis_dim_next,
                best_w_next,
                best_x_next,
                best_y_next,
                best_residual_next,
                converged_flag_next,
                done_next,
            )

        return jax.lax.cond(done_cur, lambda s: s, do_iteration, state)

    final_state = jax.lax.fori_loop(
        0,
        int(max_iter),
        _step,
        (
            v_basis,
            w_basis,
            u1_basis,
            u2_basis,
            active_mask,
            basis_dim0,
            best_w0,
            best_x0,
            best_y0,
            best_residual0,
            converged0,
            done0,
        ),
    )
    (
        _v_basis,
        _w_basis,
        _u1_basis,
        _u2_basis,
        _active_mask,
        _basis_dim,
        best_w,
        best_x,
        best_y,
        _best_residual,
        converged,
        _done,
    ) = final_state
    return best_w, best_x, best_y, converged


def implicit_differential_davidson_lowest_tdhf(
    vind: Callable[[Array], Array],
    *,
    nroots: int,
    size: int,
    diag: Array,
    tol: float = PYSCF_TD_DAVIDSON_TOL,
    max_iter: int = PYSCF_TD_DAVIDSON_MAX_CYCLE,
    max_subspace: int | None = None,
    matrix_eps: float = 1e-10,
    preconditioner_floor: float = 1e-8,
    preconditioner_level_shift: float = 0.0,
    orth_eps: float = 1e-10,
) -> tuple[Array, Array, Array, Array]:
    """Return TDHF Davidson roots with implicit eigenvalue differentiation."""

    dim = int(size)

    def solver_vind(values: Array) -> Array:
        return jax.lax.stop_gradient(vind(values))

    omega, x_vecs, y_vecs, converged = _davidson_lowest_tdhf(
        solver_vind,
        nroots=nroots,
        size=dim,
        diag=diag,
        tol=tol,
        max_iter=max_iter,
        max_subspace=max_subspace,
        matrix_eps=matrix_eps,
        preconditioner_floor=preconditioner_floor,
        preconditioner_level_shift=preconditioner_level_shift,
        orth_eps=orth_eps,
    )
    x_vecs = jax.lax.stop_gradient(x_vecs)
    y_vecs = jax.lax.stop_gradient(y_vecs)
    applied = vind(jnp.concatenate([x_vecs.T, y_vecs.T], axis=-1))
    top = applied[:, :dim].T
    bottom = -applied[:, dim:].T
    numerator = jnp.sum(x_vecs * top, axis=0) + jnp.sum(y_vecs * bottom, axis=0)
    denominator = jnp.sum(x_vecs * x_vecs, axis=0) - jnp.sum(y_vecs * y_vecs, axis=0)
    denominator = jnp.where(
        jnp.abs(denominator) > jnp.asarray(1e-30, dtype=x_vecs.dtype),
        denominator,
        jnp.asarray(1e-30, dtype=x_vecs.dtype),
    )
    return numerator / denominator, x_vecs, y_vecs, converged
