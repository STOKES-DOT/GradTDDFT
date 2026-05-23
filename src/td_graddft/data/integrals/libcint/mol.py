from __future__ import annotations

from typing import Any

import jax
import numpy as np

from ...molecule import MoleculeSpec

_LIBCINT_MOL_CACHE_MAXSIZE = 16
_LIBCINT_MOL_CACHE: dict[tuple[Any, ...], Any] = {}


def libcint_intor_name(mol: Any, base: str) -> str:
    suffix = "_cart" if bool(getattr(mol, "cart", False)) else "_sph"
    return f"{base}{suffix}"


def _atom_from_molecule_spec(spec: MoleculeSpec) -> list[tuple[str, tuple[float, float, float]]]:
    coords_bohr = np.asarray(jax.device_get(spec.coords_bohr), dtype=float)
    return [
        (str(symbol), tuple(float(value) for value in coords_bohr[idx]))
        for idx, symbol in enumerate(spec.symbols)
    ]


def _hashable_mol_value(value: Any) -> Any:
    if isinstance(value, MoleculeSpec):
        coords_bohr = np.asarray(jax.device_get(value.coords_bohr), dtype=np.float64)
        return (
            "MoleculeSpec",
            tuple(value.symbols),
            int(value.charge),
            int(value.spin),
            coords_bohr.shape,
            coords_bohr.dtype.str,
            coords_bohr.tobytes(),
        )
    if isinstance(value, dict):
        return tuple(sorted((str(k), _hashable_mol_value(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable_mol_value(v) for v in value)
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        return ("ndarray", arr.shape, arr.dtype.str, arr.tobytes())
    try:
        hash(value)
    except TypeError:
        return ("repr", type(value).__name__, repr(value))
    return value


def _libcint_mol_cache_key(
    *,
    atom: Any,
    basis: Any,
    unit: str,
    charge: int,
    spin: int,
    cart: bool,
    verbose: int,
    mol_kwargs: dict[str, Any],
) -> tuple[Any, ...]:
    return (
        _hashable_mol_value(atom),
        _hashable_mol_value(basis),
        str(unit),
        int(charge),
        int(spin),
        bool(cart),
        int(verbose),
        _hashable_mol_value(mol_kwargs),
    )


def _cache_libcint_mol(key: tuple[Any, ...], mol: Any) -> None:
    if len(_LIBCINT_MOL_CACHE) >= _LIBCINT_MOL_CACHE_MAXSIZE:
        _LIBCINT_MOL_CACHE.pop(next(iter(_LIBCINT_MOL_CACHE)))
    _LIBCINT_MOL_CACHE[key] = mol


def build_libcint_mol(
    *,
    atom: Any,
    basis: Any,
    unit: str,
    charge: int,
    spin: int,
    cart: bool,
    verbose: int,
    **mol_kwargs: Any,
) -> Any:
    """Build the PySCF Mole object used only as a libcint integral handle."""

    try:
        from pyscf import gto
    except ModuleNotFoundError as exc:
        raise ImportError("PySCF/libcint is required when integral_backend='cpu'.") from exc

    cache_key = _libcint_mol_cache_key(
        atom=atom,
        basis=basis,
        unit=unit,
        charge=charge,
        spin=spin,
        cart=cart,
        verbose=verbose,
        mol_kwargs=mol_kwargs,
    )
    cached = _LIBCINT_MOL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if isinstance(atom, MoleculeSpec):
        atom_arg = _atom_from_molecule_spec(atom)
        unit_arg = "Bohr"
        charge_arg = int(atom.charge)
        spin_arg = int(atom.spin)
    else:
        atom_arg = atom
        unit_arg = unit
        charge_arg = int(charge)
        spin_arg = int(spin)

    mol = gto.M(
        atom=atom_arg,
        basis=basis,
        unit=unit_arg,
        charge=charge_arg,
        spin=spin_arg,
        cart=bool(cart),
        verbose=int(verbose),
        **mol_kwargs,
    )
    _cache_libcint_mol(cache_key, mol)
    return mol
