from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from td_graddft.scf.molecules import QuadratureGrid, RestrictedMolecule, UnrestrictedMolecule


_ARRAY_FIELDS = (
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
    "ao_deriv1",
    "ao_laplacian",
    "hfx_omega_values",
    "hfx_local",
    "hfx_nu",
    "pt2_local",
    "scf_initial_density",
    "df_factors",
    "eri_pair_matrix",
    "eri_ovov",
    "eri_ovvo",
    "eri_oovv",
)


def _write_array(group: Any, name: str, value: Any) -> None:
    if value is None:
        return
    array = np.asarray(value)
    if name in group:
        del group[name]
    group.create_dataset(name, data=array, compression="gzip", compression_opts=1)


def _read_array(group: Any, name: str, *, array_backend: str = "jax") -> Any | None:
    if name not in group:
        return None
    array = group[name][()]
    if array_backend == "host":
        return np.asarray(array)
    if array_backend == "jax":
        return jnp.asarray(array)
    raise ValueError(f"Unsupported array_backend {array_backend!r}.")


def write_restricted_molecule(group: Any, molecule: RestrictedMolecule) -> None:
    """Write a restricted molecule's large numerical inputs to an HDF5 group."""

    group.attrs["nuclear_repulsion"] = float(molecule.nuclear_repulsion)
    group.attrs["exact_exchange_fraction"] = float(molecule.exact_exchange_fraction)
    if molecule.mf_energy is not None:
        group.attrs["mf_energy"] = float(molecule.mf_energy)
    if molecule.nocc is not None:
        group.attrs["nocc"] = int(molecule.nocc)
    if molecule.scf_converged is not None:
        group.attrs["scf_converged"] = bool(molecule.scf_converged)
    if molecule.runtime_scf_backend is not None:
        group.attrs["runtime_scf_backend"] = str(molecule.runtime_scf_backend)

    grid_group = group.require_group("grid")
    _write_array(grid_group, "weights", molecule.grid.weights)
    _write_array(grid_group, "coords", molecule.grid.coords)
    for field in _ARRAY_FIELDS:
        _write_array(group, field, getattr(molecule, field))


def write_unrestricted_molecule(group: Any, molecule: UnrestrictedMolecule) -> None:
    """Write an unrestricted molecule's large numerical inputs to an HDF5 group."""

    group.attrs["nuclear_repulsion"] = float(molecule.nuclear_repulsion)
    group.attrs["exact_exchange_fraction"] = float(molecule.exact_exchange_fraction)
    if molecule.mf_energy is not None:
        group.attrs["mf_energy"] = float(molecule.mf_energy)
    if molecule.nocc_alpha is not None:
        group.attrs["nocc_alpha"] = int(molecule.nocc_alpha)
    if molecule.nocc_beta is not None:
        group.attrs["nocc_beta"] = int(molecule.nocc_beta)
    if molecule.runtime_scf_backend is not None:
        group.attrs["runtime_scf_backend"] = str(molecule.runtime_scf_backend)

    grid_group = group.require_group("grid")
    _write_array(grid_group, "weights", molecule.grid.weights)
    _write_array(grid_group, "coords", molecule.grid.coords)
    for field in _ARRAY_FIELDS:
        if hasattr(molecule, field):
            _write_array(group, field, getattr(molecule, field))


def read_restricted_molecule(
    group: Any,
    *,
    array_backend: str = "jax",
) -> RestrictedMolecule:
    """Read a restricted molecule saved by :func:`write_restricted_molecule`."""

    grid_group = group["grid"]
    grid = QuadratureGrid(
        weights=_read_array(grid_group, "weights", array_backend=array_backend),
        coords=_read_array(grid_group, "coords", array_backend=array_backend),
    )
    kwargs = {
        field: _read_array(group, field, array_backend=array_backend)
        for field in _ARRAY_FIELDS
    }
    return RestrictedMolecule(
        ao=kwargs.pop("ao"),
        grid=grid,
        dipole_integrals=kwargs.pop("dipole_integrals"),
        rep_tensor=kwargs.pop("rep_tensor"),
        mo_coeff=kwargs.pop("mo_coeff"),
        mo_occ=kwargs.pop("mo_occ"),
        mo_energy=kwargs.pop("mo_energy"),
        rdm1=kwargs.pop("rdm1"),
        h1e=kwargs.pop("h1e"),
        nuclear_repulsion=float(group.attrs["nuclear_repulsion"]),
        mf_energy=(
            float(group.attrs["mf_energy"])
            if "mf_energy" in group.attrs
            else None
        ),
        exact_exchange_fraction=float(group.attrs.get("exact_exchange_fraction", 0.0)),
        nocc=(int(group.attrs["nocc"]) if "nocc" in group.attrs else None),
        scf_converged=(
            bool(group.attrs["scf_converged"])
            if "scf_converged" in group.attrs
            else None
        ),
        runtime_scf_backend=(
            str(group.attrs["runtime_scf_backend"])
            if "runtime_scf_backend" in group.attrs
            else None
        ),
        **kwargs,
    )


def read_unrestricted_molecule(
    group: Any,
    *,
    array_backend: str = "jax",
) -> UnrestrictedMolecule:
    """Read an unrestricted molecule saved by :func:`write_unrestricted_molecule`."""

    grid_group = group["grid"]
    grid = QuadratureGrid(
        weights=_read_array(grid_group, "weights", array_backend=array_backend),
        coords=_read_array(grid_group, "coords", array_backend=array_backend),
    )
    kwargs = {
        field: _read_array(group, field, array_backend=array_backend)
        for field in _ARRAY_FIELDS
    }
    return UnrestrictedMolecule(
        ao=kwargs.pop("ao"),
        grid=grid,
        dipole_integrals=kwargs.pop("dipole_integrals"),
        rep_tensor=kwargs.pop("rep_tensor"),
        mo_coeff=kwargs.pop("mo_coeff"),
        mo_occ=kwargs.pop("mo_occ"),
        mo_energy=kwargs.pop("mo_energy"),
        rdm1=kwargs.pop("rdm1"),
        h1e=kwargs.pop("h1e"),
        nuclear_repulsion=float(group.attrs["nuclear_repulsion"]),
        mf_energy=(
            float(group.attrs["mf_energy"])
            if "mf_energy" in group.attrs
            else None
        ),
        exact_exchange_fraction=float(group.attrs.get("exact_exchange_fraction", 0.0)),
        nocc_alpha=(int(group.attrs["nocc_alpha"]) if "nocc_alpha" in group.attrs else None),
        nocc_beta=(int(group.attrs["nocc_beta"]) if "nocc_beta" in group.attrs else None),
        runtime_scf_backend=(
            str(group.attrs["runtime_scf_backend"])
            if "runtime_scf_backend" in group.attrs
            else None
        ),
        atom_coords=kwargs.pop("atom_coords"),
        atom_charges=kwargs.pop("atom_charges"),
        overlap_matrix=kwargs.pop("overlap_matrix"),
        ao_deriv1=kwargs.pop("ao_deriv1"),
        ao_laplacian=kwargs.pop("ao_laplacian"),
        hfx_omega_values=kwargs.pop("hfx_omega_values"),
        hfx_local=kwargs.pop("hfx_local"),
        hfx_nu=kwargs.pop("hfx_nu"),
        scf_initial_density=kwargs.pop("scf_initial_density"),
    )


__all__ = [
    "read_restricted_molecule",
    "read_unrestricted_molecule",
    "write_restricted_molecule",
    "write_unrestricted_molecule",
]
