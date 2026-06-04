from __future__ import annotations

from dataclasses import dataclass

from pyscf.data.elements import ELEMENTS, charge as atomic_number


@dataclass(frozen=True)
class AtomSpec:
    name: str
    symbol: str
    split: str
    charge: int = 0
    spin: int = 0
    unit: str = "Angstrom"
    notes: str | None = None

    @property
    def atom(self) -> str:
        return f"{self.symbol} 0.0 0.0 0.0"

    @property
    def atomic_number(self) -> int:
        return int(atomic_number(self.symbol))


TRANSITION_METALS_3D = ("Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn")
GRADDFT_TEST_ATOMS = {"K", "Ga", "As", "Br", *TRANSITION_METALS_3D[1::2]}


def _neutral_atom_spin(symbol: str) -> int:
    # Neutral ground-state spin multiplicities in PySCF convention:
    # spin = n_alpha - n_beta = 2S. These are needed only to build reference
    # rows for the GradDFT-style ground-state atom split.
    spins = {
        "H": 1,
        "He": 0,
        "Li": 1,
        "Be": 0,
        "B": 1,
        "C": 2,
        "N": 3,
        "O": 2,
        "F": 1,
        "Ne": 0,
        "Na": 1,
        "Mg": 0,
        "Al": 1,
        "Si": 2,
        "P": 3,
        "S": 2,
        "Cl": 1,
        "Ar": 0,
        "K": 1,
        "Ca": 0,
        "Sc": 1,
        "Ti": 2,
        "V": 3,
        "Cr": 6,
        "Mn": 5,
        "Fe": 4,
        "Co": 3,
        "Ni": 2,
        "Cu": 1,
        "Zn": 0,
        "Ga": 1,
        "Ge": 2,
        "As": 3,
        "Se": 2,
        "Br": 1,
        "Kr": 0,
    }
    if symbol not in spins:
        raise KeyError(f"No neutral atom spin configured for {symbol!r}.")
    return spins[symbol]


def graddft_ground_atom_specs() -> tuple[AtomSpec, ...]:
    specs: list[AtomSpec] = []
    for symbol in ELEMENTS[1:37]:
        split = "test" if symbol in GRADDFT_TEST_ATOMS else "train"
        specs.append(
            AtomSpec(
                name=symbol,
                symbol=symbol,
                split=split,
                spin=_neutral_atom_spin(symbol),
                notes="GradDFT-style atom species extrapolation split; ground-state target.",
            )
        )
    return tuple(specs)


def closed_shell_s1_atom_specs() -> tuple[AtomSpec, ...]:
    # Closed-shell atoms keep the first S1 run restricted and compatible with
    # the existing closed_shell_s1_self_consistent_train.py path.
    layout = (
        ("He", "train"),
        ("Be", "train"),
        ("Ne", "train"),
        ("Mg", "train"),
        ("Ar", "train"),
        ("Ca", "validation"),
        ("Zn", "validation"),
        ("Kr", "test"),
    )
    return tuple(
        AtomSpec(
            name=symbol,
            symbol=symbol,
            split=split,
            spin=0,
            notes="Closed-shell atom species extrapolation split; S1-only excited-state target.",
        )
        for symbol, split in layout
    )


def atom_spec_map(*, preset: str) -> dict[str, AtomSpec]:
    preset_key = str(preset).strip().lower()
    if preset_key == "graddft_ground":
        specs = graddft_ground_atom_specs()
    elif preset_key == "closed_shell_s1":
        specs = closed_shell_s1_atom_specs()
    else:
        raise ValueError("preset must be one of: graddft_ground, closed_shell_s1")
    return {spec.name: spec for spec in specs}
