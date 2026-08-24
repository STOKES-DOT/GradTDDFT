from __future__ import annotations

from typing import Any


_BOHR_PER_ANGSTROM = 1.8897259885789

_ELEMENTS_106 = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr",
)

_GHOSH_RADII = (
    0.5292, 0.3113, 1.6283, 1.0855, 0.8141, 0.6513, 0.5428, 0.4652, 0.4071, 0.3618,
    2.1650, 1.6711, 1.3608, 1.1477, 0.9922, 0.8739, 0.7808, 0.7056, 3.2930, 2.5419,
    2.4149, 2.2998, 2.1953, 2.1000, 2.0124, 1.9319, 1.8575, 1.7888, 1.7250, 1.6654,
    1.4489, 1.2823, 1.1450, 1.0424, 0.9532, 0.8782, 3.8487, 2.9709, 2.8224, 2.6880,
    2.5658, 2.4543, 2.3520, 2.2579, 2.1711, 2.0907, 2.0160, 1.9465, 1.6934, 1.4986,
    1.3440, 1.2183, 1.1141, 1.0263, 4.2433, 3.2753, 2.6673, 2.2494, 1.9447, 1.7129,
    1.5303, 1.3830, 1.2615, 1.1596, 1.0730, 0.9984, 0.9335, 0.8765, 0.8261, 0.7812,
    0.7409, 0.7056, 0.6716, 0.6416, 0.6141, 0.5890, 0.5657, 0.5443, 0.5244, 0.5060,
    1.8670, 1.6523, 1.4818, 1.3431, 1.2283, 1.1315, 4.4479, 3.4332, 3.2615, 3.1061,
    2.2756, 1.9767, 1.7473, 1.4496, 1.2915, 1.2960, 1.1247, 1.0465, 0.9785, 0.9188,
    0.8659, 0.8188, 0.8086,
)

RIS_EXP = {
    element: 1.0 / (radius * _BOHR_PER_ANGSTROM) ** 2
    for element, radius in zip(_ELEMENTS_106, _GHOSH_RADII, strict=True)
}
_RIS_FITS = {"s", "sp", "spd"}


def _element_from_basis_key(key: Any) -> str:
    token = str(key)
    symbol = "".join(char for char in token if char.isalpha())
    if len(symbol) > 1:
        symbol = symbol[0].upper() + symbol[1:].lower()
    else:
        symbol = symbol.upper()
    return symbol


def minimal_ris_auxbasis_for_mol(
    mol: Any,
    *,
    theta: float = 0.2,
    fitting_basis: str = "s",
) -> dict[str, list[list[Any]]]:
    """Return a PySCF-compatible minimal auxiliary basis for TDDFT-RIS factors."""

    fit = str(fitting_basis).lower()
    if fit not in _RIS_FITS:
        raise ValueError(f"fitting_basis must be one of {{'s', 'sp', 'spd'}}, got {fit!r}.")
    if float(theta) <= 0.0:
        raise ValueError("theta must be positive.")

    basis_keys = tuple(getattr(mol, "_basis", {}).keys())
    if not basis_keys:
        atom_symbol = getattr(mol, "atom_symbol", None)
        natm = int(getattr(mol, "natm", 0))
        basis_keys = tuple(atom_symbol(i) for i in range(natm)) if callable(atom_symbol) else ()
    if not basis_keys:
        raise ValueError("Cannot infer atom labels for RIS auxiliary basis construction.")

    auxbasis: dict[str, list[list[Any]]] = {}
    for key in basis_keys:
        element = _element_from_basis_key(key)
        if element not in RIS_EXP:
            raise ValueError(f"No RIS exponent parameter is available for element {element!r}.")
        exponent = float(RIS_EXP[element]) * float(theta)
        shells: list[list[Any]] = [[0, [exponent, 1.0]]]
        if element != "H":
            if "p" in fit:
                shells.append([1, [exponent, 1.0]])
            if "d" in fit:
                shells.append([2, [exponent, 1.0]])
        auxbasis[str(key)] = shells
    return auxbasis


__all__ = ["RIS_EXP", "minimal_ris_auxbasis_for_mol"]
