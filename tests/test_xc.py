import jax.numpy as jnp

from td_graddft.xc import lda_from_callable


def test_quadratic_functional_has_linear_potential():
    functional = lda_from_callable("toy", lambda rho: 0.5 * rho)
    density = jnp.array([0.5, 1.5])

    potential = functional.potential(density)

    assert jnp.allclose(potential, density)


def test_quadratic_functional_has_identity_kernel():
    functional = lda_from_callable("toy", lambda rho: 0.5 * rho)
    density = jnp.array([0.5, 1.5])

    kernel = functional.kernel(density)

    assert jnp.allclose(kernel, jnp.eye(2))
