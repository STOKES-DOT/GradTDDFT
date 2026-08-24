from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import numpy as np

ANGSTROM_TO_BOHR = 1.8897261254578281

_SYMBOL_TO_Z = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
}


def _real_dtype():
    return jnp.float64 if jax.config.x64_enabled else jnp.float32


@dataclass(frozen=True)
class MoleculeSpec:
    symbols: tuple[str, ...]
    coords_bohr: jnp.ndarray
    charges: jnp.ndarray
    charge: int = 0
    spin: int = 0
    unit: str = "Angstrom"

    @property
    def nelectron(self) -> int:
        return int(jnp.asarray(self.charges, dtype=jnp.int32).sum()) - int(self.charge)

    @property
    def nuclear_repulsion(self) -> jnp.ndarray:
        return nuclear_repulsion_energy(self)

    @property
    def charge_center(self) -> jnp.ndarray:
        return charge_center(self)


def atomic_number(symbol: str) -> int:
    clean = str(symbol).strip().capitalize()
    if clean not in _SYMBOL_TO_Z:
        raise ValueError(f"Unsupported element symbol {symbol!r} in strict-JAX molecule parser.")
    return _SYMBOL_TO_Z[clean]


def _normalize_atom_records(atom: Any) -> list[tuple[str, Any]]:
    if isinstance(atom, str):
        text = atom.replace(";", "\n")
        records: list[tuple[str, tuple[float, float, float]]] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(
                    "Atom specification lines must have form 'Sym x y z'. "
                    f"Got {line!r}."
                )
            sym = parts[0]
            xyz = (float(parts[1]), float(parts[2]), float(parts[3]))
            records.append((sym, xyz))
        return records
    if isinstance(atom, Iterable):
        records = []
        for item in atom:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError(
                    "Iterable atom specification must contain (symbol, (x,y,z)) pairs."
                )
            sym = str(item[0])
            xyz_raw = item[1]
            xyz = jnp.asarray(xyz_raw)
            if xyz.shape != (3,):
                raise ValueError("Atomic coordinate entries must be length-3 coordinate arrays.")
            records.append((sym, xyz))
        return records
    raise TypeError("Unsupported atom specification type.")


def _is_jax_coordinate(value: Any) -> bool:
    return isinstance(value, jax.Array) or isinstance(value, jax.core.Tracer)


def parse_molecule_spec(
    atom: Any,
    *,
    unit: str = "Angstrom",
    charge: int = 0,
    spin: int = 0,
) -> MoleculeSpec:
    if isinstance(atom, MoleculeSpec):
        return atom

    records = _normalize_atom_records(atom)
    if not records:
        raise ValueError("Atom specification must contain at least one atom.")

    symbols = tuple(str(sym).strip().capitalize() for sym, _ in records)
    has_jax_coordinate = any(_is_jax_coordinate(xyz) for _, xyz in records)
    if has_jax_coordinate:
        coords = jnp.stack([jnp.asarray(xyz, dtype=_real_dtype()) for _, xyz in records], axis=0)
    else:
        coords = np.asarray([xyz for _, xyz in records], dtype=np.float64)
    unit_norm = str(unit).strip().lower()
    if unit_norm.startswith("angs"):
        coords_bohr = coords * ANGSTROM_TO_BOHR
    elif unit_norm.startswith("bohr"):
        coords_bohr = coords
    else:
        raise ValueError(f"Unsupported unit={unit!r}. Expected 'Angstrom' or 'Bohr'.")

    charges_raw = [atomic_number(sym) for sym in symbols]
    charges = (
        jnp.asarray(charges_raw, dtype=_real_dtype())
        if has_jax_coordinate
        else np.asarray(charges_raw, dtype=np.float64)
    )
    return MoleculeSpec(
        symbols=symbols,
        coords_bohr=coords_bohr,
        charges=charges,
        charge=int(charge),
        spin=int(spin),
        unit=unit,
    )


def nuclear_repulsion_energy(spec: MoleculeSpec) -> jnp.ndarray:
    coords = jnp.asarray(spec.coords_bohr, dtype=_real_dtype())
    charges = jnp.asarray(spec.charges, dtype=_real_dtype())
    diffs = coords[:, None, :] - coords[None, :, :]
    squared_distances = jnp.einsum("...r,...r->...", diffs, diffs)
    mask = jnp.triu(jnp.ones_like(squared_distances, dtype=bool), k=1)
    distances = jnp.sqrt(jnp.where(mask, jnp.maximum(squared_distances, 1e-32), 1.0))
    pair_terms = jnp.where(
        mask,
        charges[:, None] * charges[None, :] / jnp.maximum(distances, 1e-16),
        0.0,
    )
    return jnp.sum(pair_terms)


def charge_center(spec: MoleculeSpec) -> jnp.ndarray:
    charges = jnp.asarray(spec.charges, dtype=_real_dtype())
    coords = jnp.asarray(spec.coords_bohr, dtype=_real_dtype())
    return jnp.einsum("a,ar->r", charges, coords) / jnp.sum(charges)


__all__ = [
    "ANGSTROM_TO_BOHR",
    "MoleculeSpec",
    "atomic_number",
    "charge_center",
    "nuclear_repulsion_energy",
    "parse_molecule_spec",
]
