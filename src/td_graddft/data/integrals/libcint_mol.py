from __future__ import annotations

from typing import Any

import jax
import numpy as np

from ..molecule import MoleculeSpec


def libcint_intor_name(mol: Any, base: str) -> str:
    suffix = "_cart" if bool(getattr(mol, "cart", False)) else "_sph"
    return f"{base}{suffix}"


def _atom_from_molecule_spec(spec: MoleculeSpec) -> list[tuple[str, tuple[float, float, float]]]:
    coords_bohr = np.asarray(jax.device_get(spec.coords_bohr), dtype=float)
    return [
        (str(symbol), tuple(float(value) for value in coords_bohr[idx]))
        for idx, symbol in enumerate(spec.symbols)
    ]


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
        raise ImportError("PySCF/libcint is required when integral_backend='libcint'.") from exc

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

    return gto.M(
        atom=atom_arg,
        basis=basis,
        unit=unit_arg,
        charge=charge_arg,
        spin=spin_arg,
        cart=bool(cart),
        verbose=int(verbose),
        **mol_kwargs,
    )
