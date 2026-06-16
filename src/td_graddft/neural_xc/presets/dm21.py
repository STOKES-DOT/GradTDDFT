from __future__ import annotations

from dataclasses import replace

from ..config import ChannelSpec, ComponentSpec, Config, NetworkSpec
from ..defaults import (
    DEFAULT_NEURAL_XC_HF_INPUT_MODE,
    DEFAULT_NEURAL_XC_RESPONSE_HF_MODE,
    DEFAULT_NEURAL_XC_RESPONSE_PT2_MODE,
    DEFAULT_NEURAL_XC_SEMILOCAL_XC,
    DEFAULT_INPUT_FEATURE_MODE,
    DEFAULT_NETWORK_HIDDEN_DIMS,
)


def dm21(
    *,
    components: ComponentSpec | None = None,
    channels: ChannelSpec | None = None,
    network: NetworkSpec | None = None,
    **overrides,
) -> Config:
    config = Config(
        components=components
        or ComponentSpec(
            backend="jax_libxc",
            semilocal=tuple(str(name) for name in DEFAULT_NEURAL_XC_SEMILOCAL_XC),
        ),
        channels=channels
        or ChannelSpec(
            hf=DEFAULT_NEURAL_XC_HF_INPUT_MODE,
            pt2="off",
            response_hf=DEFAULT_NEURAL_XC_RESPONSE_HF_MODE,
            response_pt2=DEFAULT_NEURAL_XC_RESPONSE_PT2_MODE,
        ),
        network=network
        or NetworkSpec(
            hidden_dims=tuple(int(v) for v in DEFAULT_NETWORK_HIDDEN_DIMS),
            activation="tanh",
        ),
        input_feature_mode=DEFAULT_INPUT_FEATURE_MODE,
    )
    if not overrides:
        return config
    return replace(config, **overrides)


__all__ = ["dm21"]
