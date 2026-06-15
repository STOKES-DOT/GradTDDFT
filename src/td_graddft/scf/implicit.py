from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree


@dataclass(frozen=True)
class ImplicitFixedPointConfig:
    """Controls the adjoint solve for an implicit fixed-point state."""

    solver_name: str = "gmres"
    tolerance: float = 1e-6
    max_iter: int = 6
    restart: int | None = None
    regularization: float = 0.0
    clip: float = 1e4


def implicit_fixed_point_solution(
    params: PyTree,
    *,
    solution: Array,
    fixed_point: Callable[..., Array],
    fixed_point_args: PyTree | None = None,
    config: ImplicitFixedPointConfig | None = None,
    apply_fixed_point_transpose: Callable[..., Array] | None = None,
    apply_fixed_point_transpose_factory: Callable[..., Callable[[Array], Array]] | None = None,
    params_vjp_from_adjoint: Callable[..., PyTree] | None = None,
    callback_aux: PyTree | None = None,
) -> Array:
    """Return a primal fixed-point solution with an implicit VJP w.r.t. params.

    The optimality condition is `fixed_point(solution, params) - solution = 0`.
    The primal `solution` is supplied by the caller, usually from a normal SCF
    loop. Backward solves the transposed fixed-point linear system instead of
    differentiating through that loop.
    """

    cfg = ImplicitFixedPointConfig() if config is None else config
    primal_solution = jnp.asarray(solution)
    has_fixed_point_args = fixed_point_args is not None
    fixed_point_args_tree = () if fixed_point_args is None else fixed_point_args

    @jax.custom_vjp
    def _solution_from_params(params_local: PyTree, solution_local: Array, args_local: PyTree) -> Array:
        del params_local, args_local
        return solution_local

    def _call_fixed_point(solution_value: Array, params_value: PyTree, args_value: PyTree) -> Array:
        if has_fixed_point_args:
            return fixed_point(solution_value, params_value, args_value)
        return fixed_point(solution_value, params_value)

    def _call_with_optional_aux(fn: Callable[..., Any], *args: Any) -> Any:
        if callback_aux is None:
            return fn(*args)
        return fn(*args, callback_aux)

    def _fwd(
        params_local: PyTree,
        solution_local: Array,
        args_local: PyTree,
    ) -> tuple[Array, tuple[PyTree, Array, PyTree]]:
        return solution_local, (params_local, solution_local, args_local)

    def _bwd(
        res: tuple[PyTree, Array, PyTree],
        cotangent_solution: Array,
    ) -> tuple[PyTree, Array, PyTree]:
        params_local, solution_local, args_local = res
        rhs = _clean_and_clip(cotangent_solution, cfg.clip)

        if apply_fixed_point_transpose_factory is not None:
            fixed_point_transpose = apply_fixed_point_transpose_factory(
                solution_local,
                params_local,
            )
        elif apply_fixed_point_transpose is not None:
            fixed_point_transpose = lambda vec: _call_with_optional_aux(
                apply_fixed_point_transpose,
                solution_local,
                params_local,
                vec,
            )
        else:
            _, solution_vjp = jax.vjp(
                lambda solution_var: _call_fixed_point(solution_var, params_local, args_local),
                solution_local,
            )
            fixed_point_transpose = lambda vec: solution_vjp(vec)[0]

        def _optimality_transpose(vec: Array) -> Array:
            cot = fixed_point_transpose(vec) - vec
            return _clean_and_clip(cot, cfg.clip)

        regularization = jnp.asarray(
            max(float(cfg.regularization), 0.0),
            dtype=solution_local.dtype,
        )

        def _adjoint_op(vec_flat: Array) -> Array:
            vec = vec_flat.reshape(solution_local.shape)
            out = _optimality_transpose(vec) - regularization * vec
            return _clean_and_clip(out, cfg.clip).reshape(-1)

        lambda_flat = solve_implicit_linear_system(
            _adjoint_op,
            -jax.lax.stop_gradient(rhs.reshape(-1)),
            solver_name=cfg.solver_name,
            tol=cfg.tolerance,
            max_iter=cfg.max_iter,
            restart=cfg.restart,
        )
        adjoint = jax.lax.stop_gradient(lambda_flat).reshape(solution_local.shape)
        adjoint = _clean_and_clip(adjoint, cfg.clip)

        if params_vjp_from_adjoint is not None:
            grad_params = _call_with_optional_aux(
                params_vjp_from_adjoint,
                solution_local,
                params_local,
                adjoint,
            )
        else:
            _, params_vjp = jax.vjp(
                lambda params_var: _call_fixed_point(solution_local, params_var, args_local),
                params_local,
            )
            grad_params = params_vjp(adjoint)[0]

        grad_params = jax.tree_util.tree_map(
            lambda x: jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
            grad_params,
        )
        return (
            grad_params,
            jnp.zeros_like(solution_local),
            None,
        )

    _solution_from_params.defvjp(_fwd, _bwd)
    return _solution_from_params(params, primal_solution, fixed_point_args_tree)


def solve_implicit_linear_system(
    matvec: Callable[[Array], Array],
    b_flat: Array,
    *,
    solver_name: str = "gmres",
    tol: float,
    max_iter: int,
    restart: int | None = None,
) -> Array:
    if str(solver_name).lower() != "gmres":
        raise ValueError(f"Unsupported implicit linear solver {solver_name!r}.")
    del restart

    def _matvec(vec: Array) -> Array:
        return jnp.nan_to_num(matvec(vec), nan=0.0, posinf=0.0, neginf=0.0)

    sol = _solve_implicit_gmres(
        _matvec,
        b_flat,
        tol=float(tol),
        maxiter=max(1, int(max_iter)),
    )
    return jnp.nan_to_num(sol, nan=0.0, posinf=0.0, neginf=0.0)


def _solve_implicit_gmres(
    matvec: Callable[[Array], Array],
    b_flat: Array,
    *,
    tol: float,
    maxiter: int,
) -> Array:
    maxiter = max(1, int(maxiter))
    x0 = jnp.zeros_like(b_flat)
    dtype = b_flat.dtype
    eps = jnp.asarray(1e-30, dtype=dtype)
    r0 = jnp.nan_to_num(b_flat, nan=0.0, posinf=0.0, neginf=0.0)
    beta = jnp.sqrt(jnp.maximum(jnp.vdot(r0, r0).real, eps))
    b_norm = jnp.sqrt(jnp.maximum(jnp.vdot(b_flat, b_flat).real, eps))
    tol_abs = jnp.asarray(float(tol), dtype=dtype) * b_norm
    v = jnp.zeros((maxiter + 1, b_flat.size), dtype=dtype)
    h = jnp.zeros((maxiter + 1, maxiter), dtype=dtype)
    v = v.at[0].set(jnp.where(beta > eps, r0 / beta, jnp.zeros_like(r0)))
    rhs = jnp.zeros((maxiter + 1,), dtype=dtype).at[0].set(beta)
    done = beta <= tol_abs
    x_best = x0

    for col in range(maxiter):
        w = jnp.nan_to_num(matvec(v[col]), nan=0.0, posinf=0.0, neginf=0.0)
        for row in range(col + 1):
            h_row = jnp.vdot(v[row], w).real
            w = w - h_row * v[row]
            h = h.at[row, col].set(h_row)

        h_next = jnp.sqrt(jnp.maximum(jnp.vdot(w, w).real, eps))
        h = h.at[col + 1, col].set(h_next)
        v_next = jnp.where(h_next > eps, w / h_next, jnp.zeros_like(w))
        v = v.at[col + 1].set(v_next)

        h_sub = h[: col + 2, : col + 1]
        rhs_sub = rhs[: col + 2]
        y, *_ = jnp.linalg.lstsq(h_sub, rhs_sub, rcond=None)
        x_candidate = x0 + v[: col + 1].T @ y
        residual_vec = rhs_sub - h_sub @ y
        residual = jnp.sqrt(jnp.maximum(jnp.vdot(residual_vec, residual_vec).real, eps))
        x_best = jnp.where(done, x_best, x_candidate)
        done = jnp.logical_or(done, residual <= tol_abs)

    return x_best


def _clean_and_clip(value: Any, clip: float) -> Array:
    arr = jnp.nan_to_num(jnp.asarray(value), nan=0.0, posinf=0.0, neginf=0.0)
    return jnp.clip(arr, -float(clip), float(clip))
