from __future__ import annotations

from typing import Callable, Literal, Sequence

import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array

from .binding import BoundNeuralXCFunctional
from .model import NeuralXCFunctional, NeuralXCHybridFunctional
from .components import (
    SemilocalEnergyDensityFn,
    SemilocalEnergyDensityModule,
    normalize_semilocal_xc_names,
)
from .defaults import (
    DEFAULT_NETWORK_ARCHITECTURE,
    DEFAULT_INPUT_FEATURE_MODE,
    DEFAULT_NEURAL_XC_RESPONSE_HF_MODE,
    DEFAULT_NEURAL_XC_SEMILOCAL_XC,
    DEFAULT_NETWORK_HIDDEN_DIMS,
)
from .networks import ResidualMixingMLP, normalize_hidden_dims


def _make_neural_xc_hybrid_functional(
    *,
    non_hf_module: SemilocalEnergyDensityModule | None = None,
    semilocal_xc: str | Sequence[str] = DEFAULT_NEURAL_XC_SEMILOCAL_XC,
    semilocal_energy_density_fn: SemilocalEnergyDensityFn | None = None,
    n_semilocal_channels: int | None = None,
    input_feature_mode: Literal["enhanced", "canonical"] = DEFAULT_INPUT_FEATURE_MODE,
    hf_input_mode: Literal["total_only", "spin_resolved"] = "spin_resolved",
    include_hfx_channel: bool = False,
    ground_state_hf_mode: Literal["off", "nograd", "scf"] | None = None,
    include_pt2_channel: bool = False,
    ground_state_pt2_mode: Literal["off", "nograd", "scf"] | None = None,
    pt2_channel_mode: Literal["scaled_projected", "local_exact"] = "scaled_projected",
    response_hf_mode: Literal["approx", "strict"] = DEFAULT_NEURAL_XC_RESPONSE_HF_MODE,
    response_pt2_mode: Literal["approx", "strict"] = "approx",
    strict_feature_alignment: bool = True,
    allow_experimental_jax_xc: bool = False,
    architecture: str | None = None,
    network_architecture: str | None = None,
    hidden_dims: Sequence[int] = DEFAULT_NETWORK_HIDDEN_DIMS,
    activation: Callable[[Array], Array] = nn.tanh,
    squash_offset: float = 1e-4,
    sigmoid_scale_factor: float = 2.0,
    density_floor: float = 1e-12,
    response_density_floor: float | None = 1e-5,
    kernel_clip: float = 5.0,
    response_kernel_clip: float | None = 5.0,
    hfx_channels: int = 2,
    name: str = "neural_xc",
) -> NeuralXCFunctional:
    if ground_state_hf_mode is not None:
        ground_state_hf_mode = str(ground_state_hf_mode).lower()  # type: ignore[assignment]
        if ground_state_hf_mode == "frozen":
            ground_state_hf_mode = "nograd"  # type: ignore[assignment]
        if ground_state_hf_mode not in {"off", "nograd", "scf"}:
            raise ValueError(
                "ground_state_hf_mode must be 'off', 'nograd', or 'scf'; "
                f"got {ground_state_hf_mode!r}."
            )
        include_hfx_channel = ground_state_hf_mode != "off"
    if ground_state_pt2_mode is not None:
        ground_state_pt2_mode = str(ground_state_pt2_mode).lower()  # type: ignore[assignment]
        if ground_state_pt2_mode == "frozen":
            ground_state_pt2_mode = "nograd"  # type: ignore[assignment]
        if ground_state_pt2_mode not in {"off", "nograd", "scf"}:
            raise ValueError(
                "ground_state_pt2_mode must be 'off', 'nograd', or 'scf'; "
                f"got {ground_state_pt2_mode!r}."
            )
        include_pt2_channel = ground_state_pt2_mode != "off"
    arch = DEFAULT_NETWORK_ARCHITECTURE
    if architecture is not None:
        arch = str(architecture)
    if network_architecture is not None:
        arch = str(network_architecture)
    if arch not in {"graddft_residual", "residual", "simple_mlp", "mlp"}:
        raise ValueError(
            f"Unsupported network architecture={arch!r}. "
            "Expected 'graddft_residual' or 'simple_mlp'."
        )
    if non_hf_module is not None:
        if (
            n_semilocal_channels is not None
            and int(n_semilocal_channels) != int(non_hf_module.n_channels)
        ):
            raise ValueError(
                "n_semilocal_channels must match non_hf_module.n_channels when both are set."
            )
        n_semilocal = int(non_hf_module.n_channels)
    elif semilocal_energy_density_fn is None:
        n_semilocal = len(
            normalize_semilocal_xc_names(
                semilocal_xc,
                allow_experimental_jax_xc=allow_experimental_jax_xc,
            )
        )
    elif n_semilocal_channels is None:
        n_semilocal = 1
    else:
        n_semilocal = int(n_semilocal_channels)
    if n_semilocal <= 0:
        raise ValueError("n_semilocal_channels must be a positive integer.")

    dims = normalize_hidden_dims(hidden_dims)
    output_dim = (
        n_semilocal
        + int(bool(include_pt2_channel))
        + int(bool(include_hfx_channel))
    )
    block_activation = nn.elu if activation is nn.tanh else activation
    model = ResidualMixingMLP(
        hidden_dims=dims,
        output_dim=output_dim,
        block_activation=block_activation,
        squash_offset=squash_offset,
        sigmoid_scale_factor=sigmoid_scale_factor,
    )

    return NeuralXCFunctional(
        model=model,
        non_hf_module=non_hf_module,
        semilocal_xc=semilocal_xc,
        semilocal_energy_density_fn=semilocal_energy_density_fn,
        input_feature_mode=input_feature_mode,
        hf_input_mode=hf_input_mode,
        include_hfx_channel=bool(include_hfx_channel),
        ground_state_hf_mode=ground_state_hf_mode,
        include_pt2_channel=bool(include_pt2_channel),
        ground_state_pt2_mode=ground_state_pt2_mode,
        pt2_channel_mode=pt2_channel_mode,
        response_hf_mode=response_hf_mode,
        response_pt2_mode=response_pt2_mode,
        strict_feature_alignment=bool(strict_feature_alignment),
        allow_experimental_jax_xc=bool(allow_experimental_jax_xc),
        density_floor=density_floor,
        response_density_floor=response_density_floor,
        kernel_clip=kernel_clip,
        response_kernel_clip=response_kernel_clip,
        hfx_channels=max(int(hfx_channels), 1),
        name=name,
    )


NeuralXCMixingMLP = ResidualMixingMLP


def make_neural_xc_functional(
    *,
    non_hf_module: SemilocalEnergyDensityModule | None = None,
    semilocal_xc: str | Sequence[str] = DEFAULT_NEURAL_XC_SEMILOCAL_XC,
    semilocal_energy_density_fn: SemilocalEnergyDensityFn | None = None,
    n_semilocal_channels: int | None = None,
    input_feature_mode: Literal["enhanced", "canonical"] = DEFAULT_INPUT_FEATURE_MODE,
    hf_input_mode: Literal["total_only", "spin_resolved"] = "spin_resolved",
    include_hfx_channel: bool = False,
    ground_state_hf_mode: Literal["off", "nograd", "scf"] | None = None,
    include_pt2_channel: bool = False,
    ground_state_pt2_mode: Literal["off", "nograd", "scf"] | None = None,
    pt2_channel_mode: Literal["scaled_projected", "local_exact"] = "scaled_projected",
    response_hf_mode: Literal["approx", "strict"] = DEFAULT_NEURAL_XC_RESPONSE_HF_MODE,
    response_pt2_mode: Literal["approx", "strict"] = "approx",
    strict_feature_alignment: bool = True,
    allow_experimental_jax_xc: bool = False,
    architecture: str | None = None,
    network_architecture: str | None = None,
    hidden_dims: Sequence[int] = DEFAULT_NETWORK_HIDDEN_DIMS,
    activation: Callable[[Array], Array] = nn.tanh,
    squash_offset: float = 1e-4,
    sigmoid_scale_factor: float = 2.0,
    density_floor: float = 1e-12,
    response_density_floor: float | None = 1e-5,
    kernel_clip: float = 5.0,
    response_kernel_clip: float | None = 5.0,
    hfx_channels: int = 2,
    name: str = "neural_xc",
) -> NeuralXCHybridFunctional:
    return _make_neural_xc_hybrid_functional(
        non_hf_module=non_hf_module,
        semilocal_xc=semilocal_xc,
        semilocal_energy_density_fn=semilocal_energy_density_fn,
        n_semilocal_channels=n_semilocal_channels,
        input_feature_mode=input_feature_mode,
        hf_input_mode=hf_input_mode,
        include_hfx_channel=include_hfx_channel,
        ground_state_hf_mode=ground_state_hf_mode,
        include_pt2_channel=include_pt2_channel,
        ground_state_pt2_mode=ground_state_pt2_mode,
        pt2_channel_mode=pt2_channel_mode,
        response_hf_mode=response_hf_mode,
        response_pt2_mode=response_pt2_mode,
        strict_feature_alignment=strict_feature_alignment,
        allow_experimental_jax_xc=allow_experimental_jax_xc,
        architecture=architecture,
        network_architecture=network_architecture,
        hidden_dims=hidden_dims,
        activation=activation,
        squash_offset=squash_offset,
        sigmoid_scale_factor=sigmoid_scale_factor,
        density_floor=density_floor,
        response_density_floor=response_density_floor,
        kernel_clip=kernel_clip,
        response_kernel_clip=response_kernel_clip,
        hfx_channels=hfx_channels,
        name=name,
    )


__all__ = [
    "BoundNeuralXCFunctional",
    "NeuralXCFunctional",
    "NeuralXCHybridFunctional",
    "NeuralXCMixingMLP",
    "make_neural_xc_functional",
]
