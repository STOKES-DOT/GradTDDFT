from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from td_graddft.tddft._unrestricted_semilocal_response import (
    UnrestrictedSemilocalResponseFunctional,
    build_spin_transition_factors,
    project_grid_response_to_spin_transition,
    project_spin_transition_to_grid,
)


@pytest.fixture
def open_shell_molecule():
    ao = jnp.asarray(
        [[1.0, 0.2], [0.8, -0.3], [0.6, 0.4]],
        dtype=jnp.float64,
    )
    ao_deriv1 = jnp.stack([ao, 0.1 * ao, -0.2 * ao, 0.05 * ao], axis=0)
    return SimpleNamespace(
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid=SimpleNamespace(
            weights=jnp.asarray([0.5, 0.3, 0.2], dtype=jnp.float64),
            coords=jnp.zeros((3, 3), dtype=jnp.float64),
        ),
        mo_coeff=jnp.stack([jnp.eye(2), jnp.eye(2)], axis=0),
        mo_occ=jnp.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.float64),
        mo_energy=jnp.asarray([[-0.6, 0.2], [-0.4, 0.3]], dtype=jnp.float64),
        rdm1=jnp.asarray(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
            dtype=jnp.float64,
        ),
        nocc_alpha=1,
        nocc_beta=0,
        exact_exchange_fraction=0.2,
        rep_tensor=jnp.zeros((2, 2, 2, 2), dtype=jnp.float64),
    )


def test_b3lyp_unrestricted_functional_exposes_full_gga_hvp(open_shell_molecule):
    functional = UnrestrictedSemilocalResponseFunctional("b3lyp")
    ngrids = open_shell_molecule.ao.shape[0]
    tangent_a = jnp.zeros((2, 4, ngrids), dtype=jnp.float64)
    tangent_a = tangent_a.at[:, 1, :].set(0.1)
    tangent_b = jnp.zeros_like(tangent_a)

    response_a, response_b = functional.spin_grid_response_hvp(
        open_shell_molecule,
        tangent_a,
        tangent_b,
    )

    assert functional.response_feature_kind == "GGA"
    assert response_a.shape == tangent_a.shape
    assert response_b.shape == tangent_b.shape
    assert jnp.all(jnp.isfinite(response_a))
    assert jnp.all(jnp.isfinite(response_b))
    assert jnp.any(jnp.abs(response_a) > 0.0)


def test_b3lyp_unrestricted_hvp_is_jittable(open_shell_molecule):
    functional = UnrestrictedSemilocalResponseFunctional("b3lyp")
    ngrids = open_shell_molecule.ao.shape[0]
    tangent_a = jnp.ones((1, 4, ngrids), dtype=jnp.float64) * 1e-3
    tangent_b = jnp.zeros_like(tangent_a)

    response_a, response_b = jax.jit(
        lambda alpha, beta: functional.spin_grid_response_hvp(
            open_shell_molecule,
            alpha,
            beta,
        )
    )(tangent_a, tangent_b)

    assert jnp.all(jnp.isfinite(response_a))
    assert jnp.all(jnp.isfinite(response_b))


def test_spin_transition_projection_matches_explicit_ao_derivative_formula(
    open_shell_molecule,
):
    orbo = open_shell_molecule.mo_coeff[0][:, :1]
    orbv = open_shell_molecule.mo_coeff[0][:, 1:]
    factors = build_spin_transition_factors(
        open_shell_molecule,
        orbo,
        orbv,
        feature_kind="GGA",
        dtype=jnp.float64,
    )
    amplitudes = jnp.asarray([[[0.7]]], dtype=jnp.float64)

    projected = project_spin_transition_to_grid(factors, amplitudes)

    ao1 = open_shell_molecule.ao_deriv1[:4]
    occupied = jnp.einsum("xgp,pi->xgi", ao1, orbo)
    virtual = jnp.einsum("xgp,pa->xga", ao1, orbv)
    expected_density = 0.7 * occupied[0, :, 0] * virtual[0, :, 0]
    expected_gradient = 0.7 * (
        occupied[1:4, :, 0] * virtual[0, :, 0]
        + occupied[0, :, 0][None, :] * virtual[1:4, :, 0]
    )

    assert projected.shape == (1, 4, open_shell_molecule.ao.shape[0])
    assert jnp.allclose(projected[0, 0], expected_density, atol=1e-12)
    assert jnp.allclose(projected[0, 1:4], expected_gradient, atol=1e-12)


def test_spin_grid_projection_and_backprojection_are_adjoint(open_shell_molecule):
    orbo = open_shell_molecule.mo_coeff[0][:, :1]
    orbv = open_shell_molecule.mo_coeff[0][:, 1:]
    factors = build_spin_transition_factors(
        open_shell_molecule,
        orbo,
        orbv,
        feature_kind="GGA",
        dtype=jnp.float64,
    )
    amplitudes = jnp.asarray([[[0.37]]], dtype=jnp.float64)
    grid_values = jnp.arange(12, dtype=jnp.float64).reshape(1, 4, 3) * 0.01

    projected = project_spin_transition_to_grid(factors, amplitudes)
    backprojected = project_grid_response_to_spin_transition(factors, grid_values)

    assert jnp.allclose(
        jnp.vdot(projected, grid_values),
        jnp.vdot(amplitudes, backprojected),
        atol=1e-12,
    )
