from __future__ import annotations

import json
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
    "hfx_fxx",
    "hfx_nu",
    "pt2_local",
    "scf_initial_density",
    "df_factors",
    "response_df_factors_j",
    "response_df_factors_k",
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


def _write_hfx_nu(group: Any, molecule: RestrictedMolecule | UnrestrictedMolecule) -> None:
    hfx_nu = getattr(molecule, "hfx_nu", None)
    if hfx_nu is not None:
        _write_array(group, "hfx_nu", hfx_nu)
        return

    hfx_nu_api = getattr(molecule, "hfx_nu_api", None)
    grid_chunk = getattr(hfx_nu_api, "grid_chunk", None)
    shape = getattr(hfx_nu_api, "shape", None)
    if hfx_nu_api is None or not callable(grid_chunk) or shape is None:
        return
    shape = tuple(int(dim) for dim in shape)
    if len(shape) != 4:
        raise ValueError(
            "HFX nu API must expose shape (n_omega, ngrids, nao, nao), "
            f"got {shape}."
        )
    if "hfx_nu" in group:
        del group["hfx_nu"]
    ngrid = int(shape[1])
    chunk_size = max(1, int(getattr(hfx_nu_api, "chunk_size", 512)))
    first_stop = min(chunk_size, ngrid)
    first_chunk = (
        np.asarray(grid_chunk(0, first_stop))
        if first_stop > 0
        else np.zeros((shape[0], 0, shape[2], shape[3]), dtype=np.float64)
    )
    bytes_per_grid = max(
        1,
        int(shape[0]) * int(shape[2]) * int(shape[3]) * int(first_chunk.dtype.itemsize),
    )
    target_chunk_bytes = 4 * 1024 * 1024
    hdf5_grid_chunk = max(
        1,
        min(first_stop, max(1, target_chunk_bytes // bytes_per_grid)),
    )
    chunks = (
        (shape[0], hdf5_grid_chunk, shape[2], shape[3])
        if first_stop > 0
        else None
    )
    dataset = group.create_dataset(
        "hfx_nu",
        shape=shape,
        dtype=first_chunk.dtype,
        chunks=chunks,
        compression="gzip",
        compression_opts=1,
    )
    if first_stop > 0:
        dataset[:, 0:first_stop] = first_chunk
    for start in range(first_stop, ngrid, chunk_size):
        stop = min(start + chunk_size, ngrid)
        dataset[:, start:stop] = np.asarray(grid_chunk(start, stop))


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
    if molecule.response_df_metadata is not None:
        group.attrs["response_df_metadata"] = json.dumps(molecule.response_df_metadata)

    grid_group = group.require_group("grid")
    _write_array(grid_group, "weights", molecule.grid.weights)
    _write_array(grid_group, "coords", molecule.grid.coords)
    for field in _ARRAY_FIELDS:
        if field == "hfx_nu":
            continue
        _write_array(group, field, getattr(molecule, field))
    _write_hfx_nu(group, molecule)


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
        if field == "hfx_nu":
            continue
        if hasattr(molecule, field):
            _write_array(group, field, getattr(molecule, field))
    _write_hfx_nu(group, molecule)


def read_restricted_molecule(
    group: Any,
    *,
    array_backend: str = "jax",
    hfx_nu_storage: str = "array",
    hfx_nu_chunk_size: int = 512,
) -> RestrictedMolecule:
    """Read a restricted molecule saved by :func:`write_restricted_molecule`."""

    grid_group = group["grid"]
    grid = QuadratureGrid(
        weights=_read_array(grid_group, "weights", array_backend=array_backend),
        coords=_read_array(grid_group, "coords", array_backend=array_backend),
    )
    kwargs = {}
    for field in _ARRAY_FIELDS:
        if field == "hfx_nu" and hfx_nu_storage == "chunked":
            kwargs[field] = None
        else:
            kwargs[field] = _read_array(group, field, array_backend=array_backend)
    hfx_nu_api = None
    if hfx_nu_storage == "chunked" and "hfx_nu" in group:
        from td_graddft.neural_xc.inputs import ChunkedHFXNu

        hfx_nu_api = ChunkedHFXNu.from_hdf5_dataset(
            str(group.file.filename),
            str(group["hfx_nu"].name),
            chunk_size=int(hfx_nu_chunk_size),
        )
    elif hfx_nu_storage != "array":
        raise ValueError(f"Unsupported hfx_nu_storage={hfx_nu_storage!r}.")
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
        response_df_metadata=(
            json.loads(str(group.attrs["response_df_metadata"]))
            if "response_df_metadata" in group.attrs
            else None
        ),
        hfx_nu_api=hfx_nu_api,
        **kwargs,
    )


def read_unrestricted_molecule(
    group: Any,
    *,
    array_backend: str = "jax",
    hfx_nu_storage: str = "array",
    hfx_nu_chunk_size: int = 512,
) -> UnrestrictedMolecule:
    """Read an unrestricted molecule saved by :func:`write_unrestricted_molecule`."""

    grid_group = group["grid"]
    grid = QuadratureGrid(
        weights=_read_array(grid_group, "weights", array_backend=array_backend),
        coords=_read_array(grid_group, "coords", array_backend=array_backend),
    )
    kwargs = {}
    for field in _ARRAY_FIELDS:
        if field == "hfx_nu" and hfx_nu_storage == "chunked":
            kwargs[field] = None
        else:
            kwargs[field] = _read_array(group, field, array_backend=array_backend)
    hfx_nu_api = None
    if hfx_nu_storage == "chunked" and "hfx_nu" in group:
        from td_graddft.neural_xc.inputs import ChunkedHFXNu

        hfx_nu_api = ChunkedHFXNu.from_hdf5_dataset(
            str(group.file.filename),
            str(group["hfx_nu"].name),
            chunk_size=int(hfx_nu_chunk_size),
        )
    elif hfx_nu_storage != "array":
        raise ValueError(f"Unsupported hfx_nu_storage={hfx_nu_storage!r}.")
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
        hfx_nu_api=hfx_nu_api,
        pt2_local=kwargs.pop("pt2_local"),
        scf_initial_density=kwargs.pop("scf_initial_density"),
    )


__all__ = [
    "read_restricted_molecule",
    "read_unrestricted_molecule",
    "write_restricted_molecule",
    "write_unrestricted_molecule",
]
