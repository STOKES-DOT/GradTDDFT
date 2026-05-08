import jax.numpy as jnp

from td_graddft.jax_libxc import RestrictedFeatureBundle, eval_xc_energy_density
from td_graddft.traditional_xc import (
    TraditionalXCFunctional,
    make_b3lyp_functional,
    make_pbe0_functional,
)


def _toy_features():
    rho_a = jnp.array([0.2, 0.3])
    rho_b = jnp.array([0.2, 0.3])
    sigma = jnp.array([0.01, 0.02])
    tau = jnp.array([0.05, 0.07])
    return RestrictedFeatureBundle(
        rho_a=rho_a,
        rho_b=rho_b,
        sigma_aa=sigma,
        sigma_ab=jnp.zeros_like(sigma),
        sigma_bb=sigma,
        tau_a=tau,
        tau_b=tau,
    )


def test_traditional_xc_functional_matches_jax_libxc_energy_density():
    functional = TraditionalXCFunctional("pbe")
    features = _toy_features()

    assert jnp.allclose(functional.energy_density(features), eval_xc_energy_density("pbe", features))


def test_pbe0_builder_exposes_exact_exchange_fraction():
    functional = make_pbe0_functional()

    assert functional.exact_exchange_fraction == 0.25
    assert functional.response_kind == "GGA"


def test_b3lyp_builder_constructs_supported_functional():
    functional = make_b3lyp_functional()
    features = _toy_features()

    assert functional.name == "b3lyp"
    assert functional.local_energy_density(features).shape == features.rho.shape


def test_local_energy_density_alias_matches_energy_density():
    functional = TraditionalXCFunctional("pbe")
    features = _toy_features()

    assert jnp.allclose(
        functional.local_energy_density(features),
        functional.energy_density(features),
    )
