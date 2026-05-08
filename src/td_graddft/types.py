from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from jaxtyping import Array


class HamiltonianBuilder(Protocol):
    """Builds the one-particle Hamiltonian used during propagation."""

    def __call__(self, time: float, density_matrix: Array) -> Array:
        ...


class Observable(Protocol):
    """Computes an observable from a density matrix."""

    def __call__(self, density_matrix: Array) -> Array:
        ...


@dataclass(frozen=True)
class GroundStateReference:
    """Ground-state information needed to initialize TD calculations."""

    density_matrix: Array
    overlap_matrix: Array | None = None
    fock_matrix: Array | None = None
    orbital_coefficients: Array | None = None
    orbital_energies: Array | None = None
    occupations: Array | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RealTimeState:
    """State of a real-time propagation."""

    time: float
    density_matrix: Array
    hamiltonian: Array | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

