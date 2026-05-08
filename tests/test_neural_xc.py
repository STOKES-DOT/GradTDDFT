from dataclasses import dataclass

import jax
import jax.numpy as jnp

from td_graddft.neural_xc import (
    DensityNeuralXCFunctional,
    NeuralXCFunctional,
    PointwiseMLP,
    make_neural_lda_functional,
)


def test_make_neural_lda_functional_initializes_and_binds():
    functional = make_neural_lda_functional(hidden_dims=(8,), n_basis=4)
    density = jnp.array([0.2, 0.5, 1.0])
    params = functional.init(jax.random.PRNGKey(0), density)

    bound = functional.bind(params)
    epsilon = bound.energy_density(density)
    kernel = bound.local_kernel(density)

    assert epsilon.shape == density.shape
    assert kernel.shape == density.shape


def test_hybrid_neural_lda_functional_binds_exact_exchange_fraction():
    functional = make_neural_lda_functional(
        hidden_dims=(8,),
        n_basis=4,
        hybrid_fraction_init=0.2,
    )
    density = jnp.array([0.2, 0.5, 1.0])
    params = functional.init(jax.random.PRNGKey(2), density)

    bound = functional.bind(params)
    gradient = jax.grad(functional.hybrid_fraction)(params)

    assert jnp.allclose(bound.exact_exchange_fraction, 0.2, atol=1e-6)
    assert jnp.isfinite(gradient["hybrid_raw"])
    assert gradient["hybrid_raw"] > 0.0


def test_custom_neural_xc_functional_returns_scalar_energy_density():
    functional = NeuralXCFunctional(
        model=PointwiseMLP(hidden_dims=(), output_dim=1, activation=lambda x: x),
        name="toy_neural",
    )
    coefficient_inputs = jnp.ones((3, 1))
    energy_density_channels = jnp.array([[0.2], [0.5], [1.0]])
    params = functional.init(jax.random.PRNGKey(1), coefficient_inputs)

    epsilon = functional.energy_density(params, coefficient_inputs, energy_density_channels)

    assert epsilon.shape == (3,)


def test_density_neural_xc_functional_adapts_density_inputs():
    functional = DensityNeuralXCFunctional(
        model=PointwiseMLP(hidden_dims=(), output_dim=1, activation=lambda x: x),
        coefficient_input_fn=lambda density, density_floor=1e-12: jnp.ones(density.shape + (1,)),
        energy_density_basis_fn=lambda density, density_floor=1e-12: density[..., None],
        name="toy_density_neural",
    )
    density = jnp.array([0.2, 0.5, 1.0])
    params = functional.init(jax.random.PRNGKey(3), density)

    epsilon = functional.energy_density(params, density)

    assert epsilon.shape == density.shape
