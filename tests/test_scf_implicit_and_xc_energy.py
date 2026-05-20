import jax
import jax.numpy as jnp
import numpy as np

from td_graddft.scf.implicit import (
    ImplicitFixedPointConfig,
    implicit_fixed_point_solution,
    solve_implicit_linear_system,
)
from td_graddft.scf.xc_energy import xc_energy_and_potential_from_density


def test_implicit_fixed_point_solution_matches_scalar_analytic_gradient():
    cfg = ImplicitFixedPointConfig(
        solver_name="gmres",
        tolerance=1e-10,
        max_iter=4,
    )

    def solve(param):
        solution = jax.lax.stop_gradient(1.0 / (1.0 - param))
        return implicit_fixed_point_solution(
            param,
            solution=solution,
            fixed_point=lambda x, p: p * x + 1.0,
            config=cfg,
        )

    param = jnp.asarray(0.2, dtype=jnp.float64)

    assert np.allclose(solve(param), 1.25)
    assert np.allclose(jax.grad(solve)(param), 1.0 / (1.0 - 0.2) ** 2, rtol=1e-6)


def test_scf_package_exports_new_refactor_boundaries():
    from td_graddft.scf import (
        ImplicitFixedPointConfig as ExportedImplicitFixedPointConfig,
        XCEnergyPotentialResult,
        implicit_fixed_point_solution as exported_implicit_fixed_point_solution,
        xc_energy_and_potential_from_density as exported_xc_energy_and_potential,
    )

    assert ExportedImplicitFixedPointConfig is ImplicitFixedPointConfig
    assert exported_implicit_fixed_point_solution is implicit_fixed_point_solution
    assert exported_xc_energy_and_potential is xc_energy_and_potential_from_density
    assert XCEnergyPotentialResult.__name__ == "XCEnergyPotentialResult"


def test_implicit_fixed_point_solution_accepts_custom_transpose_and_param_vjp():
    cfg = ImplicitFixedPointConfig(
        solver_name="gmres",
        tolerance=1e-10,
        max_iter=4,
    )

    def solve(param):
        solution = jax.lax.stop_gradient(1.0 / (1.0 - param))

        def apply_fixed_point_transpose(solution_value, param_value, cotangent):
            del solution_value
            return param_value * cotangent

        def params_vjp_from_adjoint(solution_value, param_value, adjoint):
            del param_value
            return solution_value * adjoint

        return implicit_fixed_point_solution(
            param,
            solution=solution,
            fixed_point=lambda x, p: p * x + 1.0,
            config=cfg,
            apply_fixed_point_transpose=apply_fixed_point_transpose,
            params_vjp_from_adjoint=params_vjp_from_adjoint,
        )

    param = jnp.asarray(0.2, dtype=jnp.float64)

    assert np.allclose(jax.grad(solve)(param), 1.0 / (1.0 - 0.2) ** 2, rtol=1e-6)


def test_implicit_fixed_point_solution_threads_callback_aux():
    cfg = ImplicitFixedPointConfig(
        solver_name="gmres",
        tolerance=1e-10,
        max_iter=4,
    )

    def solve(param):
        solution = jax.lax.stop_gradient(1.0 / (1.0 - 2.0 * param))

        def apply_fixed_point_transpose(solution_value, param_value, cotangent, aux):
            del solution_value
            return aux["scale"] * param_value * cotangent

        def params_vjp_from_adjoint(solution_value, param_value, adjoint, aux):
            del param_value
            return aux["scale"] * solution_value * adjoint

        return implicit_fixed_point_solution(
            param,
            solution=solution,
            fixed_point=lambda x, p: 2.0 * p * x + 1.0,
            config=cfg,
            apply_fixed_point_transpose=apply_fixed_point_transpose,
            params_vjp_from_adjoint=params_vjp_from_adjoint,
            callback_aux={"scale": jnp.asarray(2.0, dtype=param.dtype)},
        )

    param = jnp.asarray(0.2, dtype=jnp.float64)

    assert np.allclose(jax.grad(solve)(param), 2.0 / (1.0 - 2.0 * 0.2) ** 2, rtol=1e-6)


def test_implicit_fixed_point_solution_builds_custom_transpose_once():
    cfg = ImplicitFixedPointConfig(
        solver_name="gmres",
        tolerance=1e-10,
        max_iter=4,
    )
    calls = {"factory": 0, "matvec": 0}

    def solve(param):
        solution = jax.lax.stop_gradient(1.0 / (1.0 - param))

        def apply_fixed_point_transpose_factory(solution_value, param_value):
            del solution_value
            calls["factory"] += 1

            def matvec(cotangent):
                calls["matvec"] += 1
                return param_value * cotangent

            return matvec

        def params_vjp_from_adjoint(solution_value, param_value, adjoint):
            del param_value
            return solution_value * adjoint

        return implicit_fixed_point_solution(
            param,
            solution=solution,
            fixed_point=lambda x, p: p * x + 1.0,
            config=cfg,
            apply_fixed_point_transpose_factory=apply_fixed_point_transpose_factory,
            params_vjp_from_adjoint=params_vjp_from_adjoint,
        )

    param = jnp.asarray(0.2, dtype=jnp.float64)

    assert np.allclose(jax.grad(solve)(param), 1.0 / (1.0 - 0.2) ** 2, rtol=1e-6)
    assert calls["factory"] == 1
    assert calls["matvec"] > 1


def test_implicit_gmres_solves_nonsymmetric_system_with_restart():
    matrix = jnp.asarray(
        [[4.0, 1.0], [2.0, 3.0]],
        dtype=jnp.float64,
    )
    rhs = jnp.asarray([1.0, -1.0], dtype=jnp.float64)

    solution = solve_implicit_linear_system(
        lambda vec: matrix @ vec,
        rhs,
        solver_name="gmres",
        tol=1e-10,
        max_iter=4,
        restart=2,
    )

    assert np.allclose(matrix @ solution, rhs, rtol=1e-8, atol=1e-8)


def test_implicit_gmres_skips_zero_initial_matvec():
    calls = {"matvec": 0}
    rhs = jnp.asarray([1.0, -1.0], dtype=jnp.float64)

    def matvec(vec):
        calls["matvec"] += 1
        return vec

    solution = solve_implicit_linear_system(
        matvec,
        rhs,
        solver_name="gmres",
        tol=1e-10,
        max_iter=1,
        restart=1,
    )

    assert np.allclose(solution, rhs)
    assert calls["matvec"] == 1


def test_xc_energy_and_potential_from_density_uses_energy_gradient():
    density = jnp.asarray(
        [[1.0, 0.2], [0.4, -0.5]],
        dtype=jnp.float64,
    )
    scale = jnp.asarray(1.7, dtype=jnp.float64)

    result = xc_energy_and_potential_from_density(
        scale,
        molecule=None,
        density=density,
        xc_energy_fn=lambda params, _molecule, dm: 0.5 * params * jnp.sum(dm * dm),
    )

    expected_vxc = 0.5 * scale * (density + density.T)
    assert np.allclose(result.xc_energy, 0.5 * scale * jnp.sum(density * density))
    assert np.allclose(result.vxc_matrix, expected_vxc)
    assert np.allclose(result.exact_exchange_fraction, 0.0)


def test_xc_energy_and_potential_from_density_preserves_aux():
    density = jnp.asarray([[1.0, 0.0], [0.0, 0.5]], dtype=jnp.float64)
    scale = jnp.asarray(2.0, dtype=jnp.float64)

    result = xc_energy_and_potential_from_density(
        scale,
        molecule=None,
        density=density,
        xc_energy_fn=lambda params, _molecule, dm: (
            0.5 * params * jnp.sum(dm * dm),
            params + jnp.trace(dm),
        ),
        has_aux=True,
    )

    assert np.allclose(result.vxc_matrix, scale * density)
    assert np.allclose(result.aux, scale + jnp.trace(density))
