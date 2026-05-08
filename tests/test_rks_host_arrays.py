from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from td_graddft.scf import rks


def test_host_device_zeros_matches_shape_and_dtype():
    assert hasattr(rks, "_host_device_zeros")
    zeros = rks._host_device_zeros((2, 3), jnp.float64)

    assert zeros.shape == (2, 3)
    assert zeros.dtype == jnp.float64
    np.testing.assert_allclose(np.asarray(jax.device_get(zeros)), np.zeros((2, 3)))


def test_host_device_scalar_matches_dtype():
    assert hasattr(rks, "_host_device_scalar")
    value = rks._host_device_scalar(1.25, jnp.float64)

    assert value.shape == ()
    assert value.dtype == jnp.float64
    assert float(jax.device_get(value)) == 1.25
