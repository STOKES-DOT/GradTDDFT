from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
from jaxtyping import Array

from td_graddft.data.integrals.libcint.autodiff import LibcintGeometryGradPolicy
from td_graddft.data.molecule import ANGSTROM_TO_BOHR, MoleculeSpec, parse_molecule_spec
from td_graddft.scf.builders import restricted_molecule_from_spec_with_jax_rks
from td_graddft.scf import RKSConfig

from .objectives import EnergySurface

CoordinateUnit = Literal["angstrom", "bohr"]


def _normalize_coordinate_unit(unit: str) -> CoordinateUnit:
    unit_norm = str(unit).strip().lower()
    if unit_norm.startswith("angs"):
        return "angstrom"
    if unit_norm.startswith("bohr"):
        return "bohr"
    raise ValueError(f"Unsupported coordinate unit {unit!r}.")


def _coordinates_to_bohr(coordinates: Array, unit: str) -> Array:
    coords = jnp.asarray(coordinates)
    unit_norm = _normalize_coordinate_unit(unit)
    if unit_norm == "bohr":
        return coords
    return coords * ANGSTROM_TO_BOHR


def coordinates_from_molecule_spec(
    spec: MoleculeSpec,
    *,
    unit: str = "angstrom",
) -> Array:
    """Return MoleculeSpec nuclear coordinates in the requested unit."""

    molecule = parse_molecule_spec(spec)
    unit_norm = _normalize_coordinate_unit(unit)
    coords_bohr = jnp.asarray(molecule.coords_bohr)
    if unit_norm == "bohr":
        return coords_bohr
    return coords_bohr / ANGSTROM_TO_BOHR


def make_rks_ground_state_surface_from_molecule_spec(
    spec: MoleculeSpec,
    *,
    basis: str,
    xc_spec: str = "pbe",
    coordinate_unit: str = "angstrom",
    cart: bool = True,
    grids_level: int = 0,
    max_l: int = 3,
    rks_config: RKSConfig | None = None,
    grid_ao_backend: Literal["jax"] = "jax",
    integral_backend: Literal["jax", "libcint"] = "libcint",
    libcint_geometry_grad_policy: LibcintGeometryGradPolicy = "analytic",
    label: str = "rks_ground_state",
    verbose: int = 0,
    **mol_kwargs: Any,
) -> EnergySurface:
    """Build an RKS ground-state surface backed by the reference SCF path.

    The returned surface accepts Cartesian coordinates in `coordinate_unit` and
    preserves all static molecular metadata from `spec`.
    """

    template = parse_molecule_spec(spec)
    unit_norm = _normalize_coordinate_unit(coordinate_unit)
    symbols = tuple(template.symbols)
    charges = jnp.asarray(template.charges)
    charge = int(template.charge)
    spin = int(template.spin)

    def energy_fn(coordinates: Array) -> Array:
        coords_bohr = _coordinates_to_bohr(coordinates, unit_norm)
        current = MoleculeSpec(
            symbols=symbols,
            coords_bohr=coords_bohr,
            charges=charges,
            charge=charge,
            spin=spin,
            unit="Bohr",
        )
        reference = restricted_molecule_from_spec_with_jax_rks(
            atom=current,
            basis=basis,
            xc_spec=xc_spec,
            unit="Bohr",
            charge=charge,
            spin=spin,
            cart=cart,
            grids_level=grids_level,
            max_l=max_l,
            rks_config=rks_config,
            grid_ao_backend=grid_ao_backend,
            integral_backend=integral_backend,
            libcint_geometry_grad_policy=libcint_geometry_grad_policy,
            verbose=verbose,
            **mol_kwargs,
        )
        return jnp.asarray(reference.mf_energy)

    return EnergySurface(
        label=label,
        state_kind="ground",
        energy_fn=energy_fn,
    )


__all__ = [
    "coordinates_from_molecule_spec",
    "make_rks_ground_state_surface_from_molecule_spec",
]
