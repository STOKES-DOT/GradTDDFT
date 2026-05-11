import jax.numpy as jnp
import numpy as np

from td_graddft.dft.rsh import (
    ResolvedRSHParameters,
    SCFXCContributions,
)


def test_resolved_rsh_parameters_expose_paper_and_range_separated_views():
    params = ResolvedRSHParameters(
        sr_hf_fraction=jnp.asarray(0.20),
        lr_hf_fraction=jnp.asarray(0.65),
        omega=jnp.asarray(0.33),
    )

    assert np.isclose(float(params.paper_alpha), 0.20)
    assert np.isclose(float(params.paper_beta), 0.45)
    omega, alpha, beta = params.to_range_separated_coefficients()
    assert np.isclose(float(omega), 0.33)
    assert np.isclose(float(alpha), 0.65)
    assert np.isclose(float(beta), -0.45)
    omega2, alpha2, hyb2 = params.to_range_separated_hybrid_coefficients()
    assert np.isclose(float(omega2), 0.33)
    assert np.isclose(float(alpha2), 0.65)
    assert np.isclose(float(hyb2), 0.20)


def test_range_separated_coefficients_keep_short_range_fraction_explicit():
    params = ResolvedRSHParameters(
        sr_hf_fraction=jnp.asarray(0.15),
        lr_hf_fraction=jnp.asarray(0.70),
        omega=jnp.asarray(0.40),
    )

    assert np.allclose(
        tuple(float(x) for x in params.to_range_separated_coefficients()),
        (0.40, 0.70, -0.55),
    )
    assert np.allclose(
        tuple(float(x) for x in params.to_range_separated_hybrid_coefficients()),
        (0.40, 0.70, 0.15),
    )


def test_scf_xc_contributions_normalize_exchange_channels():
    contributions = SCFXCContributions(
        v_rho=jnp.ones((5,)),
        v_grad=jnp.zeros((5, 3)),
        xc_kind="GGA",
        full_hf_fraction=0.25,
        lr_hf_omegas=jnp.asarray([[0.3, 0.5]]),
        lr_hf_coefficients=jnp.asarray([[0.2, 0.1]]),
    )

    assert contributions.lr_hf_omegas is not None
    assert contributions.lr_hf_coefficients is not None
    assert contributions.lr_hf_omegas.shape == (2,)
    assert contributions.lr_hf_coefficients.shape == (2,)
    assert np.isclose(float(contributions.exact_exchange_fraction), 0.25)
