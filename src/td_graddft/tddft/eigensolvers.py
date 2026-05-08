from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array
from collections.abc import Callable

from ._utils import _symmetrize


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
            raise ValueError("size is required when davidson_lowest_symmetric receives a matvec.")
        if diag is None:
            raise ValueError("diag is required when davidson_lowest_symmetric receives a matvec.")
        op_diag = jnp.asarray(diag)
        dim = int(size)
        dtype = op_diag.dtype

        def apply(vectors: Array) -> Array:
            arr = jnp.asarray(vectors, dtype=dtype)
            squeeze = arr.ndim == 1
            arr = arr.reshape(dim, -1)
            out = jnp.asarray(matrix_or_matvec(arr), dtype=dtype).reshape(dim, -1)
            return out[:, 0] if squeeze else out

        return apply, op_diag.reshape(dim), dim, dtype

    matrix = _symmetrize(jnp.asarray(matrix_or_matvec))
    dim = int(matrix.shape[0])
    dtype = matrix.dtype

    def apply(vectors: Array) -> Array:
        arr = jnp.asarray(vectors, dtype=dtype)
        squeeze = arr.ndim == 1
        arr = arr.reshape(dim, -1)
        out = matrix @ arr
        return out[:, 0] if squeeze else out

    return apply, jnp.diag(matrix), dim, dtype


def davidson_lowest_symmetric(
    matrix_or_matvec: Array | Callable[[Array], Array],
    *,
    nroots: int,
    size: int | None = None,
    diag: Array | None = None,
    tol: float = 1e-6,
    max_iter: int = 60,
    max_subspace: int | None = None,
    collapse_subspace: int | None = None,
    preconditioner_floor: float = 1e-6,
    preconditioner_level_shift: float = 1e-3,
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
    if max_subspace is None:
        max_subspace = min(dim, max(4 * nroots, nroots + 8))
    else:
        max_subspace = min(dim, max(int(max_subspace), nroots + 2))
    if collapse_subspace is None:
        collapse_subspace = min(dim, max(2 * nroots, nroots + 4))
    else:
        collapse_subspace = min(dim, max(int(collapse_subspace), nroots))

    guess_dim = min(dim, max(nroots, min(max_subspace, nroots + 2)))
    guess_idx = jnp.argsort(diag)[:guess_dim]
    guess_basis = jnp.eye(dim, dtype=dtype)[:, guess_idx]
    guess_basis, _ = jnp.linalg.qr(guess_basis, mode="reduced")
    guess_abasis = apply(guess_basis)

    basis = jnp.zeros((dim, max_subspace), dtype=dtype)
    abasis = jnp.zeros((dim, max_subspace), dtype=dtype)
    basis = jax.lax.dynamic_update_slice(basis, guess_basis, (0, 0))
    abasis = jax.lax.dynamic_update_slice(abasis, guess_abasis, (0, 0))
    heff0 = _symmetrize(basis.T.conj() @ abasis)

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
    basis_dim0 = jnp.asarray(guess_dim, dtype=index_dtype)
    best_theta0 = jnp.zeros((nroots,), dtype=dtype)
    best_vecs0 = jnp.zeros((dim, nroots), dtype=dtype)
    best_residual0 = jnp.asarray(jnp.inf, dtype=dtype)
    converged0 = jnp.asarray(False)
    done0 = jnp.asarray(False)

    def _orthogonalize_against(vector: Array, columns: Array) -> Array:
        return vector - columns @ (columns.T.conj() @ vector)

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
        projected = basis_out.T.conj() @ appended_abasis

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
        restarted_basis = basis_in @ restart_coeff
        restarted_basis, _ = jnp.linalg.qr(restarted_basis, mode="reduced")
        restarted_abasis = apply(restarted_basis)

        basis_out = jnp.zeros((dim, max_subspace), dtype=dtype)
        abasis_out = jnp.zeros((dim, max_subspace), dtype=dtype)
        basis_out = jax.lax.dynamic_update_slice(basis_out, restarted_basis, (0, 0))
        abasis_out = jax.lax.dynamic_update_slice(abasis_out, restarted_abasis, (0, 0))
        heff_out = _symmetrize(basis_out.T.conj() @ abasis_out)

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
            order = jnp.argsort(sub_eigvals)
            sub_eigvals = sub_eigvals[order]
            sub_eigvecs = sub_eigvecs[:, order]

            theta = sub_eigvals[:nroots]
            coeff = sub_eigvecs[:, :nroots]
            vecs = basis_it @ coeff
            avecs = abasis_it @ coeff
            residuals = avecs - vecs * theta[None, :]
            residual_norms = jnp.linalg.norm(residuals, axis=0)
            max_residual = jnp.max(residual_norms)
            full_span = basis_dim_it >= jnp.asarray(dim, dtype=index_dtype)
            converged_now = full_span | (max_residual < tol_arr)
            improve_best = (max_residual < best_residual_it) | converged_now
            best_theta_next = jnp.where(improve_best, theta, best_theta_it)
            best_vecs_next = jnp.where(improve_best, vecs, best_vecs_it)
            best_residual_next = jnp.where(improve_best, max_residual, best_residual_it)

            def _correction_body(root_idx: int, carry):
                new_cols_cur, new_mask_cur, new_count_cur = carry
                root_residual_norm = jax.lax.dynamic_index_in_dim(
                    residual_norms,
                    root_idx,
                    axis=0,
                    keepdims=False,
                )
                root_theta = jax.lax.dynamic_index_in_dim(
                    theta,
                    root_idx,
                    axis=0,
                    keepdims=False,
                )
                root_residual = jax.lax.dynamic_index_in_dim(
                    residuals,
                    root_idx,
                    axis=1,
                    keepdims=False,
                )
                root_needs_update = root_residual_norm > tol_arr
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

            init_new_cols = jnp.zeros((dim, nroots), dtype=dtype)
            init_new_mask = jnp.zeros((nroots,), dtype=bool)
            init_new_count = jnp.asarray(0, dtype=index_dtype)
            expand_cols, expand_mask, expand_count = jax.lax.fori_loop(
                0,
                nroots,
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
