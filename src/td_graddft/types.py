from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from jaxtyping import Array


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
