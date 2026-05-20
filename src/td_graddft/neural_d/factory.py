from __future__ import annotations

from typing import Callable, Sequence

from flax import linen as nn
from jaxtyping import Array

from .defaults import (
    DEFAULT_DISPERSION_ARCHITECTURE,
    DEFAULT_DISPERSION_HIDDEN_DIMS,
    DEFAULT_DISPERSION_R0_FLOOR,
    DEFAULT_DISPERSION_SIGMOID_SCALE_FACTOR,
    DispersionArchitecture,
)
from .functional import DispersionFunctional
from .networks import GradDFTDispersionNetwork, normalize_hidden_dims


def make_neural_d_functional(
    *,
    hidden_dims: Sequence[int] = DEFAULT_DISPERSION_HIDDEN_DIMS,
    activation: Callable[[Array], Array] = nn.gelu,
    network_architecture: DispersionArchitecture = DEFAULT_DISPERSION_ARCHITECTURE,
    sigmoid_scale_factor: float = DEFAULT_DISPERSION_SIGMOID_SCALE_FACTOR,
    r0_floor: float = DEFAULT_DISPERSION_R0_FLOOR,
) -> DispersionFunctional:
    if network_architecture != "graddft_residual":
        raise ValueError(
            f"Unsupported network_architecture={network_architecture!r}. "
            "Expected 'graddft_residual'."
        )
    network = GradDFTDispersionNetwork(
        hidden_dims=normalize_hidden_dims(hidden_dims),
        activation=activation,
        sigmoid_scale_factor=sigmoid_scale_factor,
        r0_floor=r0_floor,
    )
    return DispersionFunctional(network=network)


__all__ = [
    "make_neural_d_functional",
]
