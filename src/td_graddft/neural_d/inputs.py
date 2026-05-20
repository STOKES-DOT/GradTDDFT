from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array


def molecule_positions_and_atoms(molecule: Any) -> tuple[Array, Array]:
    positions = getattr(molecule, "nuclear_pos", None)
    if positions is None:
        positions = getattr(molecule, "atom_coords", None)
    atoms = getattr(molecule, "atom_index", None)
    if atoms is None:
        atoms = getattr(molecule, "atom_charges", None)
    if positions is None:
        raise AttributeError(
            "Molecule-like object must define nuclear_pos or atom_coords for neural_d dispersion."
        )
    if atoms is None:
        raise AttributeError(
            "Molecule-like object must define atom_index or atom_charges for neural_d dispersion."
        )
    positions = jnp.asarray(positions)
    atoms = jnp.asarray(atoms)
    if positions.ndim != 2 or int(positions.shape[-1]) != 3:
        raise ValueError("Nuclear positions must have shape (n_atom, 3).")
    if atoms.ndim != 1 or int(atoms.shape[0]) != int(positions.shape[0]):
        raise ValueError("Atomic index/charge array must have shape (n_atom,).")
    return positions, atoms


def calculate_distances(positions: Array, atoms: Array) -> tuple[Array, Array]:
    """Return GradDFT-style ordered non-self atom-pair distances and atom pairs."""

    positions = jnp.asarray(positions)
    atoms = jnp.asarray(atoms)
    pairwise_distances = jnp.linalg.norm(positions[:, None] - positions, axis=-1)
    n_atoms = int(atoms.shape[0])
    row_idx, col_idx = np.nonzero(~np.eye(n_atoms, dtype=bool))
    row_idx = jnp.asarray(row_idx)
    col_idx = jnp.asarray(col_idx)
    pairwise_distances = pairwise_distances[row_idx, col_idx]
    atom_pairs = jnp.stack((atoms[row_idx], atoms[col_idx]), axis=-1)
    return pairwise_distances[:, None], atom_pairs


def build_dispersion_pair_inputs(molecule: Any, order: int) -> Array:
    """Return pair network features `[R_AB, Z_A, Z_B, n]` for one dispersion order."""

    positions, atoms = molecule_positions_and_atoms(molecule)
    r_ab, atom_pairs = calculate_distances(positions, atoms)
    n_column = jnp.asarray(order, dtype=r_ab.dtype) * jnp.ones_like(r_ab)
    return jnp.concatenate((r_ab, atom_pairs.astype(r_ab.dtype), n_column), axis=-1)


__all__ = [
    "build_dispersion_pair_inputs",
    "calculate_distances",
    "molecule_positions_and_atoms",
]
