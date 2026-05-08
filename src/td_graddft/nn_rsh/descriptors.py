from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

import jax.numpy as jnp
from jaxtyping import Array

from ..features import restricted_grid_features


@dataclass(frozen=True)
class AtomCenteredDensityDescriptorConfig:
    """Atom-centered density power-spectrum descriptor settings.

    This follows the spirit of density-driven atom-centered descriptors used in
    Behler-Parrinello/SOAP-like models: project the self-consistent density onto
    atom-centered radial/angular channels, then contract over magnetic quantum
    numbers to obtain rotational invariants.
    """

    radial_centers: tuple[float, ...] = (0.50, 1.00, 1.80, 3.00)
    radial_width: float = 0.60
    max_angular: int = 2
    density_floor: float = 1e-12
    descriptor_floor: float = 1e-10
    apply_log1p: bool = True


def _angular_basis_components(unit_vectors: Array, max_angular: int) -> list[Array]:
    if max_angular < 0 or max_angular > 2:
        raise NotImplementedError(
            "Atom-centered density descriptors currently support max_angular in {0, 1, 2}."
        )

    x = unit_vectors[..., 0]
    y = unit_vectors[..., 1]
    z = unit_vectors[..., 2]
    components = [jnp.ones(unit_vectors.shape[:-1] + (1,), dtype=unit_vectors.dtype)]
    if max_angular >= 1:
        components.append(jnp.stack([x, y, z], axis=-1))
    if max_angular >= 2:
        components.append(
            jnp.stack(
                [
                    jnp.sqrt(3.0) * x * y,
                    jnp.sqrt(3.0) * y * z,
                    jnp.sqrt(3.0) * z * x,
                    0.5 * (2.0 * z * z - x * x - y * y),
                    0.5 * jnp.sqrt(3.0) * (x * x - y * y),
                ],
                axis=-1,
            )
        )
    return components


def atom_centered_density_power_spectrum(
    molecule: Any,
    *,
    config: AtomCenteredDensityDescriptorConfig | None = None,
) -> Array:
    """Build a rotationally invariant atom-centered density descriptor.

    Returns an array with shape ``(natoms, n_features)``.
    """

    cfg = AtomCenteredDensityDescriptorConfig() if config is None else config
    if getattr(molecule, "atom_coords", None) is None or getattr(molecule, "atom_charges", None) is None:
        raise AttributeError(
            "Atom-centered density descriptor requires molecule.atom_coords and molecule.atom_charges."
        )
    if getattr(molecule, "grid", None) is None or getattr(molecule.grid, "coords", None) is None:
        raise AttributeError(
            "Atom-centered density descriptor requires molecule.grid.coords."
        )

    atom_coords = jnp.asarray(molecule.atom_coords)
    grid_coords = jnp.asarray(molecule.grid.coords)
    weights = jnp.asarray(molecule.grid.weights)
    rho = jnp.maximum(restricted_grid_features(molecule).rho, float(cfg.density_floor))

    if atom_coords.ndim != 2 or atom_coords.shape[-1] != 3:
        raise ValueError(f"atom_coords must have shape (natoms, 3), got {atom_coords.shape}.")
    if grid_coords.ndim != 2 or grid_coords.shape[-1] != 3:
        raise ValueError(f"grid.coords must have shape (ngrids, 3), got {grid_coords.shape}.")

    displacements = grid_coords[None, :, :] - atom_coords[:, None, :]
    radii = jnp.linalg.norm(displacements, axis=-1)
    safe_radii = jnp.maximum(radii, 1e-8)
    unit_vectors = displacements / safe_radii[..., None]

    radial_centers = jnp.asarray(cfg.radial_centers, dtype=grid_coords.dtype)
    if radial_centers.ndim != 1 or radial_centers.shape[0] == 0:
        raise ValueError("radial_centers must be a non-empty 1D sequence.")
    radial_width = jnp.asarray(float(cfg.radial_width), dtype=grid_coords.dtype)
    radial = jnp.exp(-0.5 * ((radii[..., None] - radial_centers[None, None, :]) / radial_width) ** 2)
    weighted_radial = weights[None, :, None] * rho[None, :, None] * radial

    tri_upper = np.triu_indices(int(radial_centers.shape[0]))
    row_idx = jnp.asarray(tri_upper[0], dtype=jnp.int32)
    col_idx = jnp.asarray(tri_upper[1], dtype=jnp.int32)

    descriptors = []
    for block in _angular_basis_components(unit_vectors, int(cfg.max_angular)):
        coeff = jnp.einsum(
            "agn,agm->anm",
            weighted_radial,
            block,
        )
        power = jnp.einsum(
            "anm,apm->anp",
            coeff,
            coeff,
        )
        block_descriptor = power[:, row_idx, col_idx]
        descriptors.append(block_descriptor)

    descriptor = jnp.concatenate(descriptors, axis=-1)
    descriptor = jnp.maximum(descriptor, float(cfg.descriptor_floor))
    if cfg.apply_log1p:
        descriptor = jnp.log1p(descriptor)
    return jnp.nan_to_num(descriptor, nan=0.0, posinf=0.0, neginf=0.0)


def make_atom_centered_density_descriptor_fn(
    config: AtomCenteredDensityDescriptorConfig | None = None,
) -> Callable[[Any | None], dict[str, Array]]:
    """Build a descriptor_fn compatible with ``TrainableRSHFunctional``."""

    cfg = AtomCenteredDensityDescriptorConfig() if config is None else config

    def descriptor_fn(molecule: Any | None) -> dict[str, Array]:
        if molecule is None:
            raise ValueError(
                "Atom-centered density descriptors require a molecule-like input at init/apply time."
            )
        return {
            "atom_descriptors": atom_centered_density_power_spectrum(molecule, config=cfg),
            "atom_charges": jnp.asarray(molecule.atom_charges, dtype=jnp.int32),
            "atom_coords": jnp.asarray(molecule.atom_coords, dtype=jnp.float32),
        }

    return descriptor_fn


__all__ = [
    "AtomCenteredDensityDescriptorConfig",
    "atom_centered_density_power_spectrum",
    "make_atom_centered_density_descriptor_fn",
]
