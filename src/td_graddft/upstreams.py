from __future__ import annotations

import importlib
from typing import Any

import jax.numpy as jnp

from .xc_backend.jax_xc_adapter import MissingJAXXCError, load_jax_xc
from .types import GroundStateReference


class MissingDependencyError(ImportError):
    """Raised when an optional upstream package is required but unavailable."""


def _optional_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None


def has_grad_dft() -> bool:
    return _optional_import("grad_dft") is not None


def has_jax_xc() -> bool:
    try:
        module, _ = load_jax_xc()
    except MissingJAXXCError:
        return False
    return module is not None


def spin_summed_density_matrix(density_matrix: Any):
    """Return a spin-summed density matrix when a spin axis is present."""

    density_matrix = jnp.asarray(density_matrix)
    if density_matrix.ndim == 3:
        return density_matrix.sum(axis=0)
    return density_matrix


def ground_state_from_grad_dft_molecule(molecule: Any) -> GroundStateReference:
    """Extract a local ground-state reference from a GradDFT-like molecule object."""

    required_attrs = ("rdm1", "s1e", "fock", "mo_coeff", "mo_energy", "mo_occ")
    missing = [name for name in required_attrs if not hasattr(molecule, name)]
    if missing:
        joined = ", ".join(missing)
        raise AttributeError(
            f"GradDFT-like molecule object is missing required attributes: {joined}"
        )

    metadata = {
        "spin": getattr(molecule, "spin", None),
        "charge": getattr(molecule, "charge", None),
        "name": getattr(molecule, "name", None),
        "basis": getattr(molecule, "basis", None),
    }

    return GroundStateReference(
        density_matrix=jnp.asarray(molecule.rdm1),
        overlap_matrix=None if molecule.s1e is None else jnp.asarray(molecule.s1e),
        fock_matrix=None if molecule.fock is None else jnp.asarray(molecule.fock),
        orbital_coefficients=(
            None if molecule.mo_coeff is None else jnp.asarray(molecule.mo_coeff)
        ),
        orbital_energies=(
            None if molecule.mo_energy is None else jnp.asarray(molecule.mo_energy)
        ),
        occupations=None if molecule.mo_occ is None else jnp.asarray(molecule.mo_occ),
        metadata=metadata,
    )
