from __future__ import annotations

from typing import Any, Callable, Literal, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array

from .defaults import (
    DEFAULT_NETWORK_ARCHITECTURE,
    DEFAULT_NETWORK_HIDDEN_DIMS,
)


ResolvedArchitecture = Literal["simple_mlp", "graddft_residual"]

_ARCHITECTURE_ALIASES = {
    "residual": "graddft_residual",
    "graddft_residual": "graddft_residual",
    "mlp": "simple_mlp",
    "simple_mlp": "simple_mlp",
}

_ACTIVATION_ALIASES: dict[str, Callable[..., Any]] = {
    "tanh": nn.tanh,
    "elu": nn.elu,
    "relu": nn.relu,
    "gelu": nn.gelu,
    "silu": nn.silu,
}


def normalize_architecture(architecture: str) -> ResolvedArchitecture:
    key = str(architecture).lower()
    try:
        return _ARCHITECTURE_ALIASES[key]  # type: ignore[return-value]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported architecture={architecture!r}. "
            "Expected 'residual', 'graddft_residual', 'mlp', or 'simple_mlp'."
        ) from exc


def resolve_activation(activation: str | Callable[..., Any]) -> Callable[..., Any]:
    if callable(activation):
        return activation
    key = str(activation).lower()
    try:
        return _ACTIVATION_ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(_ACTIVATION_ALIASES))
        raise ValueError(
            f"Unsupported activation={activation!r}. Expected one of: {supported}."
        ) from exc


def normalize_hidden_dims(hidden_dims: Sequence[int]) -> tuple[int, ...]:
    dims = tuple(int(width) for width in hidden_dims)
    if not dims:
        raise ValueError("hidden_dims must contain at least one layer width.")
    if any(width <= 0 for width in dims):
        raise ValueError("All hidden_dims entries must be positive integers.")
    return dims


class SimpleMixingMLP(nn.Module):
    hidden_dims: Sequence[int]
    output_dim: int = 2
    activation: Callable[[Array], Array] = nn.tanh
    squash_offset: float = 1e-4
    sigmoid_scale_factor: float = 2.0

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        # Use a smooth even transform so TDDFT mixed second derivatives stay finite
        # when local response features pass through zero.
        offset = jnp.asarray(self.squash_offset, dtype=jnp.asarray(inputs).dtype)
        x = 0.5 * jnp.log(jnp.square(inputs) + offset * offset)
        for width in self.hidden_dims:
            x = nn.Dense(width)(x)
            x = self.activation(x)
        x = nn.Dense(self.output_dim)(x)
        if self.sigmoid_scale_factor > 0.0:
            scale = jnp.asarray(self.sigmoid_scale_factor, dtype=x.dtype)
            x = scale * jax.nn.sigmoid(x / scale)
        return x


class ResidualMixingMLP(nn.Module):
    """Residual mixing network used by the default neural XC preset."""

    hidden_dims: Sequence[int]
    output_dim: int = 2
    block_activation: Callable[[Array], Array] = nn.elu
    squash_offset: float = 1e-4
    sigmoid_scale_factor: float = 2.0

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        offset = jnp.asarray(self.squash_offset, dtype=jnp.asarray(inputs).dtype)
        x = jnp.log(jnp.abs(inputs) + offset)
        first_width = int(self.hidden_dims[0])
        x = nn.Dense(first_width, name="InitialDense")(x)
        x = jnp.tanh(x)

        for index, width in enumerate(self.hidden_dims):
            residual = x
            x = nn.Dense(int(width), name=f"ResidualDense_{index}")(x)
            if residual.shape[-1] != int(width):
                residual = nn.Dense(
                    int(width),
                    use_bias=False,
                    name=f"ResidualProject_{index}",
                )(residual)
            x = x + residual
            x = nn.LayerNorm(name=f"ResidualLayerNorm_{index}")(x)
            x = self.block_activation(x)

        x = nn.Dense(self.output_dim, name="HeadDense")(x)
        if self.sigmoid_scale_factor > 0.0:
            scale = jnp.asarray(self.sigmoid_scale_factor, dtype=x.dtype)
            x = scale * jax.nn.sigmoid(x / scale)
        return x


NeuralXCMixingMLP = SimpleMixingMLP


__all__ = [
    "DEFAULT_NETWORK_ARCHITECTURE",
    "DEFAULT_NETWORK_HIDDEN_DIMS",
    "NeuralXCMixingMLP",
    "ResolvedArchitecture",
    "ResidualMixingMLP",
    "SimpleMixingMLP",
    "normalize_architecture",
    "normalize_hidden_dims",
    "resolve_activation",
]
