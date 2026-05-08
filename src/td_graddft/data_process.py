from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array

from .features import restricted_grid_features
from .training import GroundStateDatum


@dataclass(frozen=True)
class NeuralXCInput:
    """Prepared grid-local neural-XC inputs for an existing molecule state."""

    molecule: Any
    features: Any
    grid_weights: Array
    coefficient_inputs: Array | None = None
    density_channels: Array | None = None
    target_total_energy: Array | None = None
    target_density_matrix: Array | None = None
    target_orbital_energies: Array | None = None
    target_orbital_occupations: Array | None = None
    density_constraint_weight: float = 0.0
    orbital_energy_constraint_weight: float = 0.0
    janak_frontier_constraint_weight: float = 0.0


def _as_molecule_and_datum(data: Any) -> tuple[Any, GroundStateDatum | None]:
    if isinstance(data, GroundStateDatum):
        return data.molecule, data
    return data, None


def _grid_weights(molecule: Any) -> Array:
    grid = getattr(molecule, "grid", None)
    weights = getattr(grid, "weights", None)
    if weights is None:
        raise AttributeError("Molecule-like object must define grid.weights.")
    return jnp.asarray(weights)


def prepare_neural_xc_input(
    data: Any,
    *,
    functional: Any | None = None,
) -> NeuralXCInput:
    """Prepare neural-XC grid inputs from an existing reference or datum.

    This function does not build a molecule state. It consumes an already
    prepared molecule/reference object and exposes the local feature tensors
    used by neural XC models.
    """

    molecule, datum = _as_molecule_and_datum(data)
    features = restricted_grid_features(molecule)
    coefficient_inputs = None
    density_channels = None

    if functional is not None:
        coefficient_builder = getattr(functional, "compute_coefficient_inputs", None)
        if callable(coefficient_builder):
            coefficient_inputs = coefficient_builder(molecule, features=features)
        density_builder = getattr(functional, "compute_densities", None)
        if callable(density_builder):
            density_channels = density_builder(molecule, features=features)

    return NeuralXCInput(
        molecule=molecule,
        features=features,
        grid_weights=_grid_weights(molecule),
        coefficient_inputs=(
            None if coefficient_inputs is None else jnp.asarray(coefficient_inputs)
        ),
        density_channels=None if density_channels is None else jnp.asarray(density_channels),
        target_total_energy=(
            None if datum is None else jnp.asarray(datum.target_total_energy)
        ),
        target_density_matrix=(
            None if datum is None or datum.target_density_matrix is None else jnp.asarray(datum.target_density_matrix)
        ),
        target_orbital_energies=(
            None if datum is None or datum.target_orbital_energies is None else jnp.asarray(datum.target_orbital_energies)
        ),
        target_orbital_occupations=(
            None if datum is None or datum.target_orbital_occupations is None else jnp.asarray(datum.target_orbital_occupations)
        ),
        density_constraint_weight=0.0 if datum is None else float(datum.density_constraint_weight),
        orbital_energy_constraint_weight=(
            0.0 if datum is None else float(datum.orbital_energy_constraint_weight)
        ),
        janak_frontier_constraint_weight=(
            0.0 if datum is None else float(datum.janak_frontier_constraint_weight)
        ),
    )


__all__ = [
    "NeuralXCInput",
    "prepare_neural_xc_input",
]
