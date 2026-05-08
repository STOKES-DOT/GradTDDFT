from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
from jaxtyping import Array


EnergyFunction = Callable[[Array], Array]


@dataclass(frozen=True)
class EnergySurface:
    """Differentiable potential-energy surface for geometry optimization."""

    label: str
    state_kind: str
    energy_fn: EnergyFunction

    def energy(self, coordinates: Array) -> Array:
        coords = jnp.asarray(coordinates)
        return jnp.asarray(self.energy_fn(coords))


def make_ground_state_surface(
    ground_energy_fn: EnergyFunction,
    *,
    label: str = "ground_state",
) -> EnergySurface:
    """Build a ground-state surface wrapper."""

    return EnergySurface(
        label=label,
        state_kind="ground",
        energy_fn=ground_energy_fn,
    )


def make_excited_state_surface(
    ground_energy_fn: EnergyFunction,
    excitation_energy_fn: Callable[[Array], Array],
    *,
    state_index: int = 0,
    label: str | None = None,
) -> EnergySurface:
    """Build an excited-state surface: E_exc = E0 + omega_state."""

    if state_index < 0:
        raise ValueError("state_index must be non-negative.")

    def excited_total_energy(coordinates: Array) -> Array:
        coords = jnp.asarray(coordinates)
        ground = jnp.asarray(ground_energy_fn(coords))
        excitations = jnp.asarray(excitation_energy_fn(coords))
        if excitations.ndim == 0:
            if state_index != 0:
                raise ValueError(
                    "excitation_energy_fn returned a scalar, but state_index != 0."
                )
            omega = excitations
        else:
            omega = excitations[state_index]
        return ground + omega

    tag = f"excited_state_{state_index + 1}" if label is None else label
    return EnergySurface(
        label=tag,
        state_kind="excited",
        energy_fn=excited_total_energy,
    )

