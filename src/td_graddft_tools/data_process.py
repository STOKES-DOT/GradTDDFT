from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from td_graddft.training.config import MolecularTrainingDatum


_ARRAY_FIELD_NAMES = (
    "target_e0_total_h",
    "target_grid_density",
    "target_density_matrix",
    "target_s1_total_h",
    "target_excitation_gaps_h",
    "target_oscillator_strengths",
    "target_spectrum_grid_ev",
    "target_spectrum_curve",
    "target_xc_potential",
    "target_xc_kernel",
    "target_orbital_energies",
    "target_orbital_occupations",
)

_SCALAR_FIELD_NAMES = (
    "weight",
    "target_xc_kernel_normalization_scale",
)

_METADATA_KEY = "__td_graddft_target_bundle_metadata_json__"


def _to_numpy_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value)


def _to_jax_or_none(value: np.ndarray | None) -> jnp.ndarray | None:
    if value is None:
        return None
    return jnp.asarray(value)


def _bundle_path(path: str | Path) -> Path:
    raw = Path(path)
    if raw.suffix == ".npz":
        return raw
    return raw.with_suffix(".npz")


def _as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(np.asarray(value))


def _as_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _extract_mol_object(molecule: Any) -> Any | None:
    mol = getattr(molecule, "mol", None)
    if mol is not None:
        return mol
    if hasattr(molecule, "atom_coords") and hasattr(molecule, "atom_charges"):
        return molecule
    return None


def _extract_basis_name(molecule: Any) -> str | None:
    if getattr(molecule, "basis", None) is not None:
        return str(getattr(molecule, "basis"))
    mol = _extract_mol_object(molecule)
    if mol is not None and getattr(mol, "basis", None) is not None:
        return str(getattr(mol, "basis"))
    return None


def input_info_atom_rows(
    input_info: InputInfo,
) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    """Return atom symbols and Cartesian coordinates from serialized input metadata."""

    symbols = tuple(str(value) for value in input_info.atom_symbols)
    coords = np.asarray(input_info.coordinates_angstrom, dtype=float).reshape((-1, 3))
    if len(symbols) != int(coords.shape[0]):
        raise ValueError(
            "InputInfo atom_symbols and coordinates_angstrom must have matching lengths "
            f"(got {len(symbols)} symbols and {coords.shape[0]} coordinates)."
        )
    return tuple(
        (
            symbols[index],
            (
                float(coords[index, 0]),
                float(coords[index, 1]),
                float(coords[index, 2]),
            ),
        )
        for index in range(len(symbols))
    )


def input_info_to_geometry_string(input_info: InputInfo) -> str:
    """Serialize stored atom metadata to a PySCF-compatible geometry string in Angstrom."""

    rows = input_info_atom_rows(input_info)
    return "\n".join(
        f"{symbol} {x:.10f} {y:.10f} {z:.10f}"
        for symbol, (x, y, z) in rows
    )


def _extract_atom_symbols_and_coords(molecule: Any) -> tuple[tuple[str, ...], np.ndarray]:
    mol = _extract_mol_object(molecule)
    if mol is not None and hasattr(mol, "natm"):
        natm = int(getattr(mol, "natm"))
        symbols = tuple(str(mol.atom_symbol(i)) for i in range(natm))
        coords = np.asarray(mol.atom_coords(unit="Angstrom"), dtype=float)
        return symbols, coords

    symbols = getattr(molecule, "atom_symbols", None)
    if symbols is None:
        symbols = getattr(molecule, "symbols", None)
    coords = getattr(molecule, "coordinates_angstrom", None)
    if coords is None:
        coords = getattr(molecule, "coordinates", None)
    if coords is None:
        coords = getattr(molecule, "coords", None)
    if symbols is None or coords is None:
        return tuple(), np.zeros((0, 3), dtype=float)
    return tuple(str(v) for v in symbols), np.asarray(coords, dtype=float)


def _extract_atomic_numbers(molecule: Any, n_atoms: int) -> tuple[int, ...]:
    mol = _extract_mol_object(molecule)
    if mol is not None and hasattr(mol, "atom_charges"):
        charges = np.asarray(mol.atom_charges(), dtype=int).reshape(-1)
        return tuple(int(v) for v in charges.tolist())
    z = getattr(molecule, "z", None)
    if z is not None:
        values = np.asarray(z, dtype=int).reshape(-1)
        return tuple(int(v) for v in values.tolist())
    return tuple()


def _extract_charge(molecule: Any) -> int | None:
    if getattr(molecule, "charge", None) is not None:
        return int(getattr(molecule, "charge"))
    mol = _extract_mol_object(molecule)
    if mol is not None and getattr(mol, "charge", None) is not None:
        return int(getattr(mol, "charge"))
    return None


def _extract_spin(molecule: Any) -> int | None:
    if getattr(molecule, "spin", None) is not None:
        return int(getattr(molecule, "spin"))
    mol = _extract_mol_object(molecule)
    if mol is not None and getattr(mol, "spin", None) is not None:
        return int(getattr(mol, "spin"))
    return None


def _extract_electron_count(molecule: Any) -> float | None:
    if getattr(molecule, "electron_count", None) is not None:
        return float(np.asarray(getattr(molecule, "electron_count")))
    if getattr(molecule, "nelectron", None) is not None:
        return float(np.asarray(getattr(molecule, "nelectron")))
    mol = _extract_mol_object(molecule)
    if mol is not None and getattr(mol, "nelectron", None) is not None:
        return float(np.asarray(getattr(mol, "nelectron")))
    return None


def _extract_mo_dimensions(molecule: Any) -> tuple[int | None, int | None, int | None]:
    mo_occ = getattr(molecule, "mo_occ", None)
    mo_coeff = getattr(molecule, "mo_coeff", None)
    if mo_occ is None or mo_coeff is None:
        return None, None, None
    occ = np.asarray(mo_occ)
    coeff = np.asarray(mo_coeff)
    if occ.ndim == 2:
        occ = occ[0]
    if coeff.ndim == 3:
        coeff = coeff[0]
    if occ.ndim != 1 or coeff.ndim != 2:
        return None, None, None
    nmo = int(min(coeff.shape[-1], occ.shape[0]))
    nocc = int(np.sum(occ[:nmo] > 1e-8))
    nvir = int(max(nmo - nocc, 0))
    return nmo, nocc, nvir


@dataclass(frozen=True)
class InputInfo:
    system_label: str
    basis_name: str | None
    charge: int | None
    spin: int | None
    electron_count: float | None
    atom_symbols: tuple[str, ...]
    atomic_numbers: tuple[int, ...]
    coordinates_angstrom: np.ndarray
    n_atoms: int
    n_ao: int | None
    n_mo: int | None
    n_occ: int | None
    n_vir: int | None
    n_grid: int | None
    has_hfx_local: bool

    def to_metadata(self) -> dict[str, Any]:
        return {
            "system_label": self.system_label,
            "basis_name": self.basis_name,
            "charge": self.charge,
            "spin": self.spin,
            "electron_count": self.electron_count,
            "atom_symbols": list(self.atom_symbols),
            "atomic_numbers": list(self.atomic_numbers),
            "coordinates_angstrom": np.asarray(self.coordinates_angstrom, dtype=float).tolist(),
            "n_atoms": self.n_atoms,
            "n_ao": self.n_ao,
            "n_mo": self.n_mo,
            "n_occ": self.n_occ,
            "n_vir": self.n_vir,
            "n_grid": self.n_grid,
            "has_hfx_local": self.has_hfx_local,
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "InputInfo":
        return cls(
            system_label=str(metadata["system_label"]),
            basis_name=metadata.get("basis_name"),
            charge=_as_int_or_none(metadata.get("charge")),
            spin=_as_int_or_none(metadata.get("spin")),
            electron_count=_as_float_or_none(metadata.get("electron_count")),
            atom_symbols=tuple(str(v) for v in metadata.get("atom_symbols", [])),
            atomic_numbers=tuple(int(v) for v in metadata.get("atomic_numbers", [])),
            coordinates_angstrom=np.asarray(
                metadata.get("coordinates_angstrom", []),
                dtype=float,
            ).reshape((-1, 3)),
            n_atoms=int(metadata.get("n_atoms", 0)),
            n_ao=_as_int_or_none(metadata.get("n_ao")),
            n_mo=_as_int_or_none(metadata.get("n_mo")),
            n_occ=_as_int_or_none(metadata.get("n_occ")),
            n_vir=_as_int_or_none(metadata.get("n_vir")),
            n_grid=_as_int_or_none(metadata.get("n_grid")),
            has_hfx_local=bool(metadata.get("has_hfx_local", False)),
        )


@dataclass(frozen=True)
class MolecularTargetBundle:
    input_info: InputInfo
    target_e0_total_h: np.ndarray | None = None
    target_grid_density: np.ndarray | None = None
    target_density_matrix: np.ndarray | None = None
    target_s1_total_h: np.ndarray | None = None
    target_excitation_gaps_h: np.ndarray | None = None
    target_oscillator_strengths: np.ndarray | None = None
    target_spectrum_grid_ev: np.ndarray | None = None
    target_spectrum_curve: np.ndarray | None = None
    target_xc_potential: np.ndarray | None = None
    target_xc_kernel: np.ndarray | None = None
    weight: float = 1.0
    target_xc_kernel_normalization_scale: float | None = None
    target_orbital_energies: np.ndarray | None = None
    target_orbital_occupations: np.ndarray | None = None

    def to_datum(self, molecule: Any) -> MolecularTrainingDatum:
        kwargs = {
            name: _to_jax_or_none(getattr(self, name))
            for name in _ARRAY_FIELD_NAMES
        }
        kwargs.update({name: getattr(self, name) for name in _SCALAR_FIELD_NAMES})
        return MolecularTrainingDatum(molecule=molecule, **kwargs)

    def save(self, path: str | Path) -> Path:
        data_path = _bundle_path(path)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            name: np.asarray(getattr(self, name))
            for name in _ARRAY_FIELD_NAMES
            if getattr(self, name) is not None
        }
        metadata = {
            "input_info": self.input_info.to_metadata(),
            "scalar_fields": {name: getattr(self, name) for name in _SCALAR_FIELD_NAMES},
            "present_array_fields": sorted(arrays.keys()),
        }
        arrays[_METADATA_KEY] = np.asarray(
            json.dumps(metadata, sort_keys=True),
            dtype=np.str_,
        )
        np.savez(data_path, **arrays)
        return data_path

    @classmethod
    def load(cls, path: str | Path) -> "MolecularTargetBundle":
        data_path = _bundle_path(path)
        with np.load(data_path, allow_pickle=False) as handle:
            arrays = {name: np.asarray(handle[name]) for name in handle.files}
        metadata_raw = arrays.pop(_METADATA_KEY, None)
        if metadata_raw is not None:
            metadata = json.loads(str(np.asarray(metadata_raw).item()))
        else:
            legacy_meta_path = data_path.with_suffix(".meta.json")
            metadata = json.loads(legacy_meta_path.read_text(encoding="utf-8"))
        kwargs = {name: arrays.get(name) for name in _ARRAY_FIELD_NAMES}
        kwargs.update(metadata.get("scalar_fields", {}))
        return cls(
            input_info=InputInfo.from_metadata(metadata["input_info"]),
            **kwargs,
        )


def prepare_input_info(
    molecule: Any,
    *,
    system_label: str | None = None,
    basis_name: str | None = None,
) -> InputInfo:
    atom_symbols, coordinates_angstrom = _extract_atom_symbols_and_coords(molecule)
    n_atoms = len(atom_symbols) if atom_symbols else int(coordinates_angstrom.shape[0])
    nmo, nocc, nvir = _extract_mo_dimensions(molecule)
    ao = getattr(molecule, "ao", None)
    n_ao = None if ao is None else int(np.asarray(ao).shape[-1])
    n_grid = None if ao is None else int(np.asarray(ao).shape[0])
    if n_ao is None:
        overlap = getattr(molecule, "overlap_matrix", None)
        if overlap is not None:
            n_ao = int(np.asarray(overlap).shape[0])
    return InputInfo(
        system_label=system_label or getattr(molecule, "name", None) or "system",
        basis_name=basis_name or _extract_basis_name(molecule),
        charge=_extract_charge(molecule),
        spin=_extract_spin(molecule),
        electron_count=_extract_electron_count(molecule),
        atom_symbols=atom_symbols,
        atomic_numbers=_extract_atomic_numbers(molecule, n_atoms),
        coordinates_angstrom=coordinates_angstrom,
        n_atoms=n_atoms,
        n_ao=n_ao,
        n_mo=nmo,
        n_occ=nocc,
        n_vir=nvir,
        n_grid=n_grid,
        has_hfx_local=getattr(molecule, "hfx_local", None) is not None,
    )


def bundle_molecular_datum(
    datum: MolecularTrainingDatum,
    *,
    input_source: Any | None = None,
    system_label: str | None = None,
    basis_name: str | None = None,
) -> MolecularTargetBundle:
    source = datum.molecule if input_source is None else input_source
    kwargs = {
        name: _to_numpy_or_none(getattr(datum, name))
        for name in _ARRAY_FIELD_NAMES
    }
    kwargs.update({name: getattr(datum, name) for name in _SCALAR_FIELD_NAMES})
    return MolecularTargetBundle(
        input_info=prepare_input_info(
            source,
            system_label=system_label,
            basis_name=basis_name,
        ),
        **kwargs,
    )


def build_molecular_target_bundle(
    molecule: Any,
    *,
    system_label: str | None = None,
    basis_name: str | None = None,
    target_e0_total_h: Any | None = None,
    target_grid_density: Any | None = None,
    target_density_matrix: Any | None = None,
    target_s1_total_h: Any | None = None,
    target_excitation_gaps_h: Any | None = None,
    target_oscillator_strengths: Any | None = None,
    target_spectrum_grid_ev: Any | None = None,
    target_spectrum_curve: Any | None = None,
    target_xc_potential: Any | None = None,
    target_xc_kernel: Any | None = None,
    target_xc_kernel_normalization_scale: float | None = None,
    target_orbital_energies: Any | None = None,
    target_orbital_occupations: Any | None = None,
    weight: float = 1.0,
) -> MolecularTargetBundle:
    datum = MolecularTrainingDatum(
        molecule=molecule,
        target_e0_total_h=_to_jax_or_none(_to_numpy_or_none(target_e0_total_h)),
        target_grid_density=_to_jax_or_none(_to_numpy_or_none(target_grid_density)),
        target_density_matrix=_to_jax_or_none(_to_numpy_or_none(target_density_matrix)),
        target_s1_total_h=_to_jax_or_none(_to_numpy_or_none(target_s1_total_h)),
        target_excitation_gaps_h=_to_jax_or_none(
            _to_numpy_or_none(target_excitation_gaps_h)
        ),
        target_oscillator_strengths=_to_jax_or_none(
            _to_numpy_or_none(target_oscillator_strengths)
        ),
        target_spectrum_grid_ev=_to_jax_or_none(_to_numpy_or_none(target_spectrum_grid_ev)),
        target_spectrum_curve=_to_jax_or_none(_to_numpy_or_none(target_spectrum_curve)),
        target_xc_potential=_to_jax_or_none(_to_numpy_or_none(target_xc_potential)),
        target_xc_kernel=_to_jax_or_none(_to_numpy_or_none(target_xc_kernel)),
        target_xc_kernel_normalization_scale=target_xc_kernel_normalization_scale,
        target_orbital_energies=_to_jax_or_none(
            _to_numpy_or_none(target_orbital_energies)
        ),
        target_orbital_occupations=_to_jax_or_none(
            _to_numpy_or_none(target_orbital_occupations)
        ),
        weight=float(weight),
    )
    return bundle_molecular_datum(
        datum,
        input_source=molecule,
        system_label=system_label,
        basis_name=basis_name,
    )


def save_molecular_target_bundle(
    bundle: MolecularTargetBundle,
    path: str | Path,
) -> Path:
    return bundle.save(path)


def load_molecular_target_bundle(path: str | Path) -> MolecularTargetBundle:
    return MolecularTargetBundle.load(path)


def load_molecular_datum(path: str | Path, molecule: Any) -> MolecularTrainingDatum:
    return load_molecular_target_bundle(path).to_datum(molecule)


__all__ = [
    "InputInfo",
    "MolecularTargetBundle",
    "input_info_atom_rows",
    "input_info_to_geometry_string",
    "prepare_input_info",
    "bundle_molecular_datum",
    "build_molecular_target_bundle",
    "save_molecular_target_bundle",
    "load_molecular_target_bundle",
    "load_molecular_datum",
]
