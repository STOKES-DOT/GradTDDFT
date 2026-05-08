from __future__ import annotations

from typing import Literal

from ..jax_libxc import (
    RSHFunctionalPreset,
    canonical_rsh_preset_name,
    get_rsh_functional_preset,
    list_rsh_functional_presets,
)
from .schema import RSHFunctionalTemplate, ResolvedRSHParameters


def rsh_preset_default_params(
    name: str,
    *,
    omega_source: Literal["canonical", "optxc"] = "canonical",
) -> ResolvedRSHParameters:
    return get_rsh_functional_preset(name).params_for_omega_source(omega_source)


def make_rsh_template(
    name: str,
    *,
    omega_source: Literal["canonical", "optxc"] = "canonical",
) -> RSHFunctionalTemplate:
    return get_rsh_functional_preset(name).to_template(omega_source)


__all__ = [
    "RSHFunctionalPreset",
    "canonical_rsh_preset_name",
    "get_rsh_functional_preset",
    "list_rsh_functional_presets",
    "make_rsh_template",
    "rsh_preset_default_params",
]
