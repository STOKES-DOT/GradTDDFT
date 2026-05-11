from __future__ import annotations

from dataclasses import fields, is_dataclass
from dataclasses import replace
from typing import Any, Literal

import jax

from .scf.molecules import QuadratureGrid, RestrictedMolecule


def put_restricted_molecule_on_device(
    molecule: RestrictedMolecule,
    device: Any | None = None,
) -> RestrictedMolecule:
    """Move a restricted molecule dataclass onto an explicit JAX device."""

    if device is None:
        return molecule

    grid = replace(
        molecule.grid,
        weights=jax.device_put(molecule.grid.weights, device),
        coords=(
            None
            if molecule.grid.coords is None
            else jax.device_put(molecule.grid.coords, device)
        ),
    )
    return replace(
        molecule,
        ao=jax.device_put(molecule.ao, device),
        dipole_integrals=jax.device_put(molecule.dipole_integrals, device),
        rep_tensor=jax.device_put(molecule.rep_tensor, device),
        mo_coeff=jax.device_put(molecule.mo_coeff, device),
        mo_occ=jax.device_put(molecule.mo_occ, device),
        mo_energy=jax.device_put(molecule.mo_energy, device),
        rdm1=jax.device_put(molecule.rdm1, device),
        h1e=jax.device_put(molecule.h1e, device),
        atom_coords=(
            None
            if getattr(molecule, "atom_coords", None) is None
            else jax.device_put(molecule.atom_coords, device)
        ),
        atom_charges=(
            None
            if getattr(molecule, "atom_charges", None) is None
            else jax.device_put(molecule.atom_charges, device)
        ),
        ao_deriv1=(
            None if molecule.ao_deriv1 is None else jax.device_put(molecule.ao_deriv1, device)
        ),
        hfx_local=(
            None if molecule.hfx_local is None else jax.device_put(molecule.hfx_local, device)
        ),
        hfx_nu=(
            None if molecule.hfx_nu is None else jax.device_put(molecule.hfx_nu, device)
        ),
        pt2_local=(
            None if getattr(molecule, "pt2_local", None) is None else jax.device_put(molecule.pt2_local, device)
        ),
        df_factors=(
            None if getattr(molecule, "df_factors", None) is None else jax.device_put(molecule.df_factors, device)
        ),
        eri_ovov=(
            None if molecule.eri_ovov is None else jax.device_put(molecule.eri_ovov, device)
        ),
        eri_ovvo=(
            None if molecule.eri_ovvo is None else jax.device_put(molecule.eri_ovvo, device)
        ),
        eri_oovv=(
            None if molecule.eri_oovv is None else jax.device_put(molecule.eri_oovv, device)
        ),
        grid=grid,
    )


def resolve_execution_device(
    preference: Literal["auto", "cpu", "gpu"] = "auto",
) -> Any | None:
    """Resolve a JAX execution device from a high-level preference.

    Returns ``None`` for ``auto`` when no explicit override is required.
    """

    pref = preference.lower()
    if pref == "auto":
        return None
    if pref == "gpu":
        gpus = jax.devices("gpu")
        if not gpus:
            raise RuntimeError("execution_device='gpu' requested but no GPU is visible to JAX.")
        return gpus[0]
    if pref == "cpu":
        cpus = jax.devices("cpu")
        if not cpus:
            raise RuntimeError("execution_device='cpu' requested but no CPU is visible to JAX.")
        return cpus[0]
    raise ValueError(f"Unsupported execution device preference: {preference!r}")


def put_molecule_on_device(
    molecule: Any,
    device: Any | None = None,
) -> Any:
    """Move a molecule-like dataclass onto an explicit JAX device.

    Supports both restricted and unrestricted molecules that share the same
    field names used by TD-GradDFT workflow code.
    """

    if device is None:
        return molecule
    if isinstance(molecule, RestrictedMolecule):
        return put_restricted_molecule_on_device(molecule, device=device)
    if not is_dataclass(molecule):
        return molecule

    updates: dict[str, Any] = {}
    molecule_fields = {f.name for f in fields(molecule)}

    def _put_if_present(name: str) -> None:
        if name in molecule_fields:
            updates[name] = jax.device_put(getattr(molecule, name), device)

    for name in (
        "ao",
        "dipole_integrals",
        "rep_tensor",
        "mo_coeff",
        "mo_occ",
        "mo_energy",
        "rdm1",
        "h1e",
        "atom_coords",
        "atom_charges",
        "overlap_matrix",
        "pt2_local",
        "df_factors",
        "eri_ovov",
        "eri_ovvo",
        "eri_oovv",
    ):
        _put_if_present(name)

    if "ao_deriv1" in molecule_fields and getattr(molecule, "ao_deriv1") is not None:
        updates["ao_deriv1"] = jax.device_put(getattr(molecule, "ao_deriv1"), device)

    if "grid" in molecule_fields and getattr(molecule, "grid") is not None:
        grid = getattr(molecule, "grid")
        if isinstance(grid, QuadratureGrid):
            updates["grid"] = replace(
                grid,
                weights=jax.device_put(grid.weights, device),
                coords=None if grid.coords is None else jax.device_put(grid.coords, device),
            )

    return replace(molecule, **updates)


put_restricted_reference_on_device = put_restricted_molecule_on_device
put_reference_on_device = put_molecule_on_device
