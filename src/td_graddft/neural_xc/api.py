from __future__ import annotations

from typing import Any

from .components import normalize_semilocal_selection, resolve_component_module
from .config import Config
from .factory import make_neural_xc_functional
from .networks import resolve_activation


def _functional_kwargs_from_config(config: Config) -> dict[str, Any]:
    pt2_mode = str(config.channels.pt2).lower()
    include_pt2_channel = pt2_mode != "off"
    allow_experimental_jax_xc = (
        bool(config.allow_experimental_jax_xc)
        or bool(config.components.allow_experimental_jax_xc)
    )
    return {
        "non_hf_module": resolve_component_module(config.components),
        "semilocal_xc": normalize_semilocal_selection(
            config.components.semilocal,
            allow_experimental_jax_xc=allow_experimental_jax_xc,
        ),
        "n_semilocal_channels": config.components.n_channels,
        "input_feature_mode": config.input_feature_mode,
        "hf_input_mode": config.channels.hf,
        "include_pt2_channel": include_pt2_channel,
        "pt2_channel_mode": "scaled_projected" if not include_pt2_channel else pt2_mode,
        "response_hf_mode": config.channels.response_hf,
        "response_pt2_mode": config.channels.response_pt2,
        "strict_feature_alignment": config.strict_feature_alignment,
        "allow_experimental_jax_xc": allow_experimental_jax_xc,
        "network_architecture": config.network.architecture,
        "hidden_dims": tuple(int(value) for value in config.network.hidden_dims),
        "activation": resolve_activation(config.network.activation),
        "squash_offset": float(config.network.squash_offset),
        "sigmoid_scale_factor": float(config.network.sigmoid_scale_factor),
        "density_floor": float(config.density_floor),
        "response_density_floor": config.response_density_floor,
        "response_grid_chunk_size": config.response_grid_chunk_size,
        "strict_hfx_response_mode": config.strict_hfx_response_mode,
        "kernel_clip": float(config.kernel_clip),
        "response_kernel_clip": config.response_kernel_clip,
        "hfx_channels": int(config.channels.hfx_channels),
        "name": config.name,
    }

def make_functional(
    *,
    config: Config | None = None,
    **kwargs: Any,
):
    if config is not None:
        if kwargs:
            merged = _functional_kwargs_from_config(config)
            merged.update(kwargs)
            return make_neural_xc_functional(**merged)
        return make_neural_xc_functional(**_functional_kwargs_from_config(config))
    return make_neural_xc_functional(**kwargs)


def Functional(
    *,
    config: Config | None = None,
    **kwargs: Any,
):
    return make_functional(config=config, **kwargs)


__all__ = [
    "Functional",
    "make_functional",
]
