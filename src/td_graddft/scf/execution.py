from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .rks import RKSConfig


@dataclass(frozen=True)
class ResolvedRKSExecutionPlan:
    mode: Literal["cpu"]
    config: RKSConfig
    integral_backend: Literal["jax", "libcint"]
    grid_ao_backend: Literal["jax"]
    include_dipole_integrals: bool


def resolve_rks_execution_plan(
    config: RKSConfig,
    *,
    integral_backend: Literal["jax", "libcint"],
    grid_ao_backend: Literal["jax"],
    include_dipole_integrals: bool,
    geometry_is_traced: bool,
) -> ResolvedRKSExecutionPlan:
    if geometry_is_traced:
        raise NotImplementedError(
            "Explicit traceable SCF execution has been removed. "
            "Use implicit differential SCF instead."
        )
    return ResolvedRKSExecutionPlan(
        mode="cpu",
        config=config,
        integral_backend=integral_backend,
        grid_ao_backend=grid_ao_backend,
        include_dipole_integrals=include_dipole_integrals,
    )
