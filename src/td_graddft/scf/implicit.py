from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres as jax_gmres
from jaxtyping import Array, PyTree


@dataclass(frozen=True)
class ImplicitFixedPointConfig:
    """Controls the adjoint solve for an implicit fixed-point state."""

    tolerance: float = 1e-6
    max_iter: int = 6
    restart: int | None = 20
    regularization: float = 0.0


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
        rhs = jnp.asarray(cotangent_solution)

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
            return fixed_point_transpose(vec) - vec

        regularization = jnp.asarray(
            max(float(cfg.regularization), 0.0),
            dtype=solution_local.dtype,
        )

        def _adjoint_op(vec_flat: Array) -> Array:
            vec = vec_flat.reshape(solution_local.shape)
            return (_optimality_transpose(vec) - regularization * vec).reshape(-1)

        lambda_flat = solve_implicit_linear_system(
            _adjoint_op,
            -jax.lax.stop_gradient(rhs.reshape(-1)),
            tol=cfg.tolerance,
            max_iter=cfg.max_iter,
            restart=cfg.restart,
        )
        adjoint = jax.lax.stop_gradient(lambda_flat).reshape(solution_local.shape)

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
    tol: float,
    max_iter: int,
    restart: int | None = None,
) -> Array:
    sol, _ = jax_gmres(
        matvec,
        b_flat,
        tol=float(tol),
        atol=0.0,
        restart=20 if restart is None else max(1, int(restart)),
        maxiter=max(1, int(max_iter)),
        solve_method="incremental",
    )
    return sol
