from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MoleculeSpec:
    name: str
    split: str
    atom: str
    charge: int = 0
    spin: int = 0
    unit: str = "Angstrom"
    notes: str | None = None


def diatomic_atom(symbol_a: str, symbol_b: str, bond_length_angstrom: float) -> str:
    half = 0.5 * float(bond_length_angstrom)
    return f"{symbol_a} 0.0 0.0 {-half:.12f}; {symbol_b} 0.0 0.0 {half:.12f}"


def benzene_atom_string(
    *,
    carbon_radius_angstrom: float = 1.397,
    hydrogen_radius_angstrom: float = 2.479,
) -> str:
    atoms: list[str] = []
    for idx in range(6):
        theta = idx * math.pi / 3.0
        cx = carbon_radius_angstrom * math.cos(theta)
        cy = carbon_radius_angstrom * math.sin(theta)
        hx = hydrogen_radius_angstrom * math.cos(theta)
        hy = hydrogen_radius_angstrom * math.sin(theta)
        atoms.append(f"C {cx:.12f} {cy:.12f} 0.000000000000")
        atoms.append(f"H {hx:.12f} {hy:.12f} 0.000000000000")
    return "; ".join(atoms)


def closed_shell_s1_specs(*, include_benzene: bool = True) -> tuple[MoleculeSpec, ...]:
    specs: list[MoleculeSpec] = [
        MoleculeSpec(
            name="H2",
            split="train",
            atom=diatomic_atom("H", "H", 0.7414),
            notes="Near-equilibrium closed-shell singlet.",
        ),
        MoleculeSpec(
            name="LiH",
            split="train",
            atom=diatomic_atom("Li", "H", 1.5956),
            notes="Heteronuclear closed-shell singlet.",
        ),
        MoleculeSpec(
            name="CO",
            split="train",
            atom=diatomic_atom("C", "O", 1.1282),
            notes="Closed-shell singlet with strong polarity.",
        ),
        MoleculeSpec(
            name="N2",
            split="validation",
            atom=diatomic_atom("N", "N", 1.0977),
            notes="Validation closed-shell singlet.",
        ),
        MoleculeSpec(
            name="F2",
            split="validation",
            atom=diatomic_atom("F", "F", 1.4119),
            notes="Validation closed-shell singlet.",
        ),
        MoleculeSpec(
            name="HF",
            split="validation",
            atom=diatomic_atom("H", "F", 0.9168),
            notes="Validation closed-shell singlet.",
        ),
    ]
    if include_benzene:
        specs.append(
            MoleculeSpec(
                name="benzene",
                split="test",
                atom=benzene_atom_string(),
                notes="Out-of-distribution aromatic generalization test.",
            )
        )
    return tuple(specs)


def closed_shell_s1_spec_map(*, include_benzene: bool = True) -> dict[str, MoleculeSpec]:
    return {spec.name: spec for spec in closed_shell_s1_specs(include_benzene=include_benzene)}
