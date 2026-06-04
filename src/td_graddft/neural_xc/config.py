from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

from .defaults import (
    DEFAULT_NETWORK_ARCHITECTURE,
    DEFAULT_NEURAL_XC_HF_INPUT_MODE,
    DEFAULT_NEURAL_XC_RESPONSE_HF_MODE,
    DEFAULT_NEURAL_XC_RESPONSE_PT2_MODE,
    DEFAULT_NEURAL_XC_SEMILOCAL_XC,
    DEFAULT_INPUT_FEATURE_MODE,
    DEFAULT_NETWORK_HIDDEN_DIMS,
)


SemilocalBackend = Literal["jax_libxc"]
HFChannelMode = Literal["total_only", "spin_resolved"]
PT2ChannelMode = Literal["off", "scaled_projected", "local_exact"]
ResponseHFMode = Literal["approx", "strict"]
ResponsePT2Mode = Literal["approx", "strict"]
StrictHFXResponseMode = Literal["dense", "low_memory"]
InputFeatureMode = Literal["enhanced", "canonical"]


@dataclass(frozen=True)
class ComponentSpec:
    backend: SemilocalBackend = "jax_libxc"
    semilocal: str | tuple[str, ...] = field(
        default_factory=lambda: tuple(str(name) for name in DEFAULT_NEURAL_XC_SEMILOCAL_XC)
    )
    module: Any | None = None
    energy_density_channels_fn: Callable[..., Any] | None = None
    n_channels: int | None = None
    channel_names: tuple[str, ...] | None = None
    allow_experimental_jax_xc: bool = False


@dataclass(frozen=True)
class ChannelSpec:
    hf: HFChannelMode = DEFAULT_NEURAL_XC_HF_INPUT_MODE
    pt2: PT2ChannelMode = "off"
    response_hf: ResponseHFMode = DEFAULT_NEURAL_XC_RESPONSE_HF_MODE
    response_pt2: ResponsePT2Mode = DEFAULT_NEURAL_XC_RESPONSE_PT2_MODE
    hfx_channels: int = 2


@dataclass(frozen=True)
class NetworkSpec:
    architecture: str = DEFAULT_NETWORK_ARCHITECTURE
    hidden_dims: tuple[int, ...] = field(
        default_factory=lambda: tuple(int(v) for v in DEFAULT_NETWORK_HIDDEN_DIMS)
    )
    activation: str | Callable[..., Any] = "tanh"
    squash_offset: float = 1e-4
    sigmoid_scale_factor: float = 2.0


@dataclass(frozen=True)
class Config:
    components: ComponentSpec = field(default_factory=ComponentSpec)
    channels: ChannelSpec = field(default_factory=ChannelSpec)
    network: NetworkSpec = field(default_factory=NetworkSpec)
    input_feature_mode: InputFeatureMode = DEFAULT_INPUT_FEATURE_MODE
    strict_feature_alignment: bool = True
    allow_experimental_jax_xc: bool = False
    density_floor: float = 1e-12
    response_density_floor: float | None = 1e-5
    response_grid_chunk_size: int | None = 1024
    strict_hfx_response_mode: StrictHFXResponseMode = "dense"
    kernel_clip: float = 5.0
    response_kernel_clip: float | None = 5.0
    name: str = "neural_xc"


__all__ = [
    "ChannelSpec",
    "ComponentSpec",
    "Config",
    "HFChannelMode",
    "InputFeatureMode",
    "NetworkSpec",
    "PT2ChannelMode",
    "ResponseHFMode",
    "ResponsePT2Mode",
    "SemilocalBackend",
    "StrictHFXResponseMode",
]
