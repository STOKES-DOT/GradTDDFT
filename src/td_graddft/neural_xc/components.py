from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import jax.numpy as jnp
from jaxtyping import Array
from ..jax_xc_adapter import (
    JAXXCFunctionalInfo,
    JAXXCStatus,
    jax_xc_functional_info,
    list_jax_xc_functionals,
)
from ..jax_libxc import (
    RestrictedFeatureBundle,
    eval_xc_energy_density,
    parse_xc,
    resolve_semilocal_xc_specs,
)
from .config import ComponentSpec


SemilocalEnergyDensityFn = Callable[[RestrictedFeatureBundle], Array]
SemilocalLocalContributionFn = Callable[[RestrictedFeatureBundle, Array, float], Array]

COMMON_SEMILOCAL_COMPONENT_SPECS = {
    "lda_x": "lda_x",
    "gga_x_b88": "gga_x_b88",
    "gga_x_pbe": "gga_x_pbe",
    "lda_c_pw": "lda_c_pw",
    "lda_c_vwn": "lda_c_vwn",
    "lda_c_vwn_rpa": "lda_c_vwn_rpa",
    "gga_c_lyp": "gga_c_lyp",
    "gga_c_pbe": "gga_c_pbe",
}


def normalize_semilocal_xc_names(
    semilocal_xc: str | Sequence[str],
    *,
    allow_experimental_jax_xc: bool = False,
) -> tuple[str, ...]:
    return resolve_semilocal_xc_specs(
        semilocal_xc,
        allow_experimental_jax_xc=allow_experimental_jax_xc,
    )


def normalize_semilocal_selection(
    semilocal_xc: str | Sequence[str],
    *,
    allow_experimental_jax_xc: bool = False,
) -> tuple[str, ...]:
    return tuple(
        str(name)
        for name in normalize_semilocal_xc_names(
            semilocal_xc,
            allow_experimental_jax_xc=allow_experimental_jax_xc,
        )
    )


@dataclass(frozen=True)
class SemilocalEnergyDensityModule:
    """Pluggable semilocal local energy-density module for neural XC."""

    channel_names: tuple[str, ...]
    energy_density_channels_fn: SemilocalEnergyDensityFn
    local_contribution_fn: SemilocalLocalContributionFn | None = None
    name: str = "semilocal_module"

    @property
    def n_channels(self) -> int:
        return len(self.channel_names)

    def energy_density_channels(self, features: RestrictedFeatureBundle) -> Array:
        channels = jnp.asarray(self.energy_density_channels_fn(features))
        channels = jnp.nan_to_num(channels, nan=0.0, posinf=0.0, neginf=0.0)
        if channels.ndim == features.rho.ndim:
            channels = channels[..., None]
        elif channels.ndim != features.rho.ndim + 1:
            raise ValueError(
                "SemilocalEnergyDensityModule must return shape (...,) or (..., n_channels)."
            )
        if channels.shape[-1] != self.n_channels:
            raise ValueError(
                "SemilocalEnergyDensityModule output channel count does not match "
                f"channel_names (got {channels.shape[-1]} vs {self.n_channels})."
            )
        return channels

    def energy_density(self, features: RestrictedFeatureBundle) -> Array:
        return jnp.sum(self.energy_density_channels(features), axis=-1)

    def local_contribution_channels(
        self,
        features: RestrictedFeatureBundle,
        *,
        channels: Array | None = None,
        density_floor: float = 1e-12,
    ) -> Array:
        channel_values = (
            self.energy_density_channels(features) if channels is None else jnp.asarray(channels)
        )
        if self.local_contribution_fn is not None:
            return jnp.asarray(
                self.local_contribution_fn(features, channel_values, float(density_floor))
            )
        del features, density_floor
        return channel_values


SemilocalModule = SemilocalEnergyDensityModule


def semilocal_component_info(name: str) -> JAXXCFunctionalInfo:
    return jax_xc_functional_info(name)


def available_semilocal_component_infos(
    *,
    statuses: Sequence[JAXXCStatus] | None = None,
    include_experimental: bool = False,
) -> tuple[JAXXCFunctionalInfo, ...]:
    if statuses is None:
        selected = {"strict", "wrapped"}
        if include_experimental:
            selected.add("experimental")
    else:
        selected = {str(status) for status in statuses}
        if include_experimental:
            selected.add("experimental")
    infos = list_jax_xc_functionals()
    return tuple(info for info in infos if info.status in selected)


def available_semilocal_components(
    *,
    statuses: Sequence[JAXXCStatus] | None = None,
    include_experimental: bool = False,
) -> tuple[str, ...]:
    infos = available_semilocal_component_infos(
        statuses=statuses,
        include_experimental=include_experimental,
    )
    names = {name for name in COMMON_SEMILOCAL_COMPONENT_SPECS}
    names.update(info.name for info in infos)
    return tuple(sorted(names))


def make_libxc_semilocal_module(
    channel_specs: str | Sequence[str],
    *,
    channel_names: Sequence[str] | None = None,
    name: str = "libxc_semilocal_module",
    allow_experimental_jax_xc: bool = False,
) -> SemilocalEnergyDensityModule:
    specs = normalize_semilocal_xc_names(
        channel_specs,
        allow_experimental_jax_xc=allow_experimental_jax_xc,
    )
    resolved_specs = tuple(COMMON_SEMILOCAL_COMPONENT_SPECS.get(spec, spec) for spec in specs)
    for spec in resolved_specs:
        parse_xc(spec, allow_experimental_jax_xc=allow_experimental_jax_xc)
    names = specs if channel_names is None else tuple(str(label) for label in channel_names)
    if len(names) != len(resolved_specs):
        raise ValueError("channel_names must match the number of semilocal channel specs.")

    def energy_density_channels_fn(features: RestrictedFeatureBundle) -> Array:
        return jnp.stack(
            [
                eval_xc_energy_density(
                    spec,
                    features,
                    allow_experimental_jax_xc=allow_experimental_jax_xc,
                )
                for spec in resolved_specs
            ],
            axis=-1,
        )

    return SemilocalEnergyDensityModule(
        channel_names=tuple(names),
        energy_density_channels_fn=energy_density_channels_fn,
        name=name,
    )


def make_custom_semilocal_module(
    *,
    channel_names: Sequence[str],
    energy_density_channels_fn: SemilocalEnergyDensityFn,
    local_contribution_fn: SemilocalLocalContributionFn | None = None,
    name: str = "custom_semilocal_module",
) -> SemilocalEnergyDensityModule:
    names = tuple(str(label) for label in channel_names)
    if not names:
        raise ValueError("channel_names must contain at least one semilocal component.")
    return SemilocalEnergyDensityModule(
        channel_names=names,
        energy_density_channels_fn=energy_density_channels_fn,
        local_contribution_fn=local_contribution_fn,
        name=name,
    )


def legacy_semilocal_module(
    semilocal_xc: str | Sequence[str],
    semilocal_energy_density_fn: SemilocalEnergyDensityFn | None,
    *,
    n_semilocal_channels: int | None = None,
    allow_experimental_jax_xc: bool = False,
) -> SemilocalEnergyDensityModule:
    if semilocal_energy_density_fn is not None:
        if n_semilocal_channels is None:
            channel_names = ("custom_semilocal",)
        else:
            channel_names = tuple(
                f"custom_semilocal_{idx + 1}" for idx in range(int(n_semilocal_channels))
            )
        return make_custom_semilocal_module(
            channel_names=channel_names,
            energy_density_channels_fn=semilocal_energy_density_fn,
            name="legacy_custom_semilocal_module",
        )
    return make_libxc_semilocal_module(
        semilocal_xc,
        name="legacy_libxc_semilocal_module",
        allow_experimental_jax_xc=allow_experimental_jax_xc,
    )


def resolve_component_module(spec: ComponentSpec) -> SemilocalEnergyDensityModule | None:
    if spec.module is not None:
        return spec.module

    if spec.energy_density_channels_fn is not None:
        if spec.channel_names is not None:
            channel_names = tuple(str(name) for name in spec.channel_names)
        elif spec.n_channels is None:
            channel_names = ("custom_semilocal",)
        else:
            count = int(spec.n_channels)
            if count <= 0:
                raise ValueError("ComponentSpec.n_channels must be positive when provided.")
            channel_names = tuple(f"custom_semilocal_{idx + 1}" for idx in range(count))
        return make_custom_semilocal_module(
            channel_names=channel_names,
            energy_density_channels_fn=spec.energy_density_channels_fn,
            name="config_custom_semilocal_module",
        )

    if spec.backend != "jax_libxc":
        raise ValueError(
            f"Unsupported ComponentSpec.backend={spec.backend!r}. Expected 'jax_libxc'."
        )

    return make_libxc_semilocal_module(
        spec.semilocal,
        channel_names=spec.channel_names,
        name="config_jax_libxc_semilocal_module",
        allow_experimental_jax_xc=bool(spec.allow_experimental_jax_xc),
    )


__all__ = [
    "COMMON_SEMILOCAL_COMPONENT_SPECS",
    "JAXXCFunctionalInfo",
    "JAXXCStatus",
    "SemilocalEnergyDensityFn",
    "SemilocalEnergyDensityModule",
    "SemilocalLocalContributionFn",
    "SemilocalModule",
    "available_semilocal_component_infos",
    "available_semilocal_components",
    "legacy_semilocal_module",
    "make_custom_semilocal_module",
    "make_libxc_semilocal_module",
    "normalize_semilocal_selection",
    "normalize_semilocal_xc_names",
    "resolve_component_module",
    "semilocal_component_info",
]
