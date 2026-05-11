from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from td_graddft.scf import rks


def test_array_xc_value_and_grad_kernel_matches_pointwise_pbe0_gga():
    variables = jnp.asarray(
        [
            [0.25, 0.01, -0.02, 0.03],
            [0.70, -0.04, 0.02, 0.01],
            [1.20, 0.03, 0.05, -0.02],
        ],
        dtype=jnp.float64,
    )

    point_exc, point_grad = rks._point_xc_value_and_grad_kernel("pbe0", "GGA", 1e-12)(
        variables
    )
    array_exc, array_grad = rks._array_xc_value_and_grad_kernel("pbe0", "GGA", 1e-12)(
        variables
    )

    np.testing.assert_allclose(np.asarray(array_exc), np.asarray(point_exc), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(array_grad), np.asarray(point_grad), rtol=1e-12, atol=1e-12)


def test_array_xc_value_and_grad_kernel_can_run_without_jit(monkeypatch):
    rks._array_xc_value_and_grad_kernel.cache_clear()
    jit_calls = []

    def fake_jit(fn):
        jit_calls.append(fn.__name__)
        return fn

    monkeypatch.setattr(rks.jax, "jit", fake_jit)

    variables = jnp.asarray([[0.70, -0.04, 0.02, 0.01]], dtype=jnp.float64)
    point_exc, point_grad = rks._array_xc_value_and_grad_kernel(
        "pbe0",
        "GGA",
        1e-12,
        False,
    )(variables)

    assert jit_calls == []
    assert point_exc.shape == (1,)
    assert point_grad.shape == (1, 4)
