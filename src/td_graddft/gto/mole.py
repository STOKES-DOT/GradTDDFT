from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..data.molecule import MoleculeSpec, parse_molecule_spec


@dataclass(frozen=True)
class Mole:
    atom: Any
    basis: Any
    unit: str = "Angstrom"
    charge: int = 0
    spin: int = 0
    cart: bool = True
    verbose: int = 0

    def to_spec(self) -> MoleculeSpec:
        return parse_molecule_spec(
            self.atom,
            unit=self.unit,
            charge=self.charge,
            spin=self.spin,
        )

    @property
    def nelectron(self) -> int:
        return self.to_spec().nelectron


def M(*args: Any, **kwargs: Any) -> Mole:
    return Mole(*args, **kwargs)
