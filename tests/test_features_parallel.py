from __future__ import annotations

import jax.numpy as jnp

from td_graddft import features as features_mod


def test_restricted_spin_channels_kernel_preserves_values():
    ao = jnp.asarray(
        [
            [1.0, 0.2],
            [0.4, 0.7],
        ],
        dtype=jnp.float64,
    )
    ao_deriv1 = jnp.asarray(
        [
            [[1.0, 0.2], [0.4, 0.7]],
            [[0.1, 0.3], [0.2, 0.5]],
            [[0.2, 0.1], [0.1, 0.4]],
            [[0.3, 0.2], [0.5, 0.1]],
        ],
        dtype=jnp.float64,
    )
    rdm1 = jnp.asarray(
        [
            [[0.8, 0.1], [0.1, 0.6]],
            [[0.7, 0.2], [0.2, 0.5]],
        ],
        dtype=jnp.float64,
    )
    mo_coeff = jnp.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.9, 0.1], [0.1, 0.9]],
        ],
        dtype=jnp.float64,
    )
    mo_occ = jnp.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.2],
        ],
        dtype=jnp.float64,
    )

    bundle, grad = features_mod._restricted_spin_channels_kernel(
        ao,
        ao_deriv1,
        rdm1,
        mo_coeff,
        mo_occ,
    )
    rho_a, grad_a = features_mod._spin_density_and_gradient(ao, ao_deriv1, rdm1[0])
    rho_b, grad_b = features_mod._spin_density_and_gradient(ao, ao_deriv1, rdm1[1])
    tau_a = features_mod._spin_tau(ao_deriv1, mo_coeff[0], mo_occ[0])
    tau_b = features_mod._spin_tau(ao_deriv1, mo_coeff[1], mo_occ[1])

    assert jnp.allclose(bundle.rho_a, rho_a)
    assert jnp.allclose(bundle.rho_b, rho_b)
    assert jnp.allclose(bundle.tau_a, tau_a)
    assert jnp.allclose(bundle.tau_b, tau_b)
    assert jnp.allclose(grad, grad_a + grad_b)
