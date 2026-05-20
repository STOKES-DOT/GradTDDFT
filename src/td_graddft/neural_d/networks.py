from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array

from .defaults import (
    DEFAULT_DISPERSION_HIDDEN_DIMS,
    DEFAULT_DISPERSION_R0_FLOOR,
    DEFAULT_DISPERSION_SIGMOID_SCALE_FACTOR,
)


def normalize_hidden_dims(hidden_dims: Sequence[int]) -> tuple[int, ...]:
    dims = tuple(int(width) for width in hidden_dims)
    if not dims:
        raise ValueError("hidden_dims must contain at least one layer width.")
    if any(width <= 0 for width in dims):
        raise ValueError("All hidden_dims entries must be positive integers.")
    return dims


class GradDFTDispersionNetwork(nn.Module):
    """GradDFT-style pair network for damped DFT-D coefficients.

    Input rows are `[R_AB, Z_A, Z_B, n]`.  The network predicts a damped
    positive coefficient `f_theta` that is later multiplied by `R_AB^(-2n)`.
    """

    hidden_dims: Sequence[int] = DEFAULT_DISPERSION_HIDDEN_DIMS
    activation: Callable[[Array], Array] = nn.gelu
    sigmoid_scale_factor: float = DEFAULT_DISPERSION_SIGMOID_SCALE_FACTOR
    r0_floor: float = DEFAULT_DISPERSION_R0_FLOOR

    def _head(self, x: Array, *, prefix: str) -> Array:
        x = nn.Dense(1, name=f"{prefix}HeadDense")(x)
        if self.sigmoid_scale_factor > 0.0:
            scale = jnp.asarray(self.sigmoid_scale_factor, dtype=x.dtype)
            x = scale * jax.nn.sigmoid(x / scale)
        return jnp.squeeze(x, axis=-1)

    def _tower(self, inputs: Array, *, prefix: str) -> Array:
        dims = normalize_hidden_dims(self.hidden_dims)
        x = nn.Dense(dims[0], name=f"{prefix}InitialDense")(inputs)
        x = jnp.tanh(x)
        for index, width in enumerate(dims):
            residual = x
            x = nn.Dense(width, name=f"{prefix}ResidualDense_{index}")(x)
            if residual.shape[-1] != width:
                residual = nn.Dense(
                    width,
                    use_bias=False,
                    name=f"{prefix}ResidualProject_{index}",
                )(residual)
            x = x + residual
            x = nn.LayerNorm(name=f"{prefix}ResidualLayerNorm_{index}")(x)
            x = self.activation(x)
        return self._head(x, prefix=prefix)

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        inputs = jnp.asarray(inputs)
        r_ab = inputs[..., 0]
        pair_inputs = inputs[..., 1:]
        r0 = self._tower(pair_inputs, prefix="R0")
        cab = self._tower(pair_inputs, prefix="C")
        safe_r0 = jnp.maximum(r0, jnp.asarray(self.r0_floor, dtype=r0.dtype))
        damping = jax.nn.sigmoid(r_ab / safe_r0 - 1.0)
        return cab * damping


__all__ = [
    "GradDFTDispersionNetwork",
    "normalize_hidden_dims",
]
