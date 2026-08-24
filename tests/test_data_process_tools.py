from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from td_graddft_tools import (
    build_molecular_target_bundle,
    input_info_atom_rows,
    input_info_to_geometry_string,
    load_molecular_datum,
    load_molecular_target_bundle,
    prepare_input_info,
)


@dataclass
class _DummyMolecule:
    name: str
    basis: str
    charge: int
    spin: int
    electron_count: float
    atom_symbols: tuple[str, ...]
    coordinates_angstrom: jnp.ndarray
    z: jnp.ndarray
    ao: jnp.ndarray
    hfx_local: jnp.ndarray
    mf_energy: float
    mo_occ: jnp.ndarray
    mo_energy: jnp.ndarray
    mo_coeff: jnp.ndarray


def _make_dummy_molecule() -> _DummyMolecule:
    return _DummyMolecule(
        name="h2_dummy",
        basis="sto-3g",
        charge=0,
        spin=0,
        electron_count=2.0,
        atom_symbols=("H", "H"),
        coordinates_angstrom=jnp.asarray([[0.0, 0.0, -0.35], [0.0, 0.0, 0.35]]),
        z=jnp.asarray([1, 1]),
        ao=jnp.ones((7, 2)),
        hfx_local=jnp.zeros((2, 7, 2)),
        mf_energy=-1.2345,
        mo_occ=jnp.asarray([2.0, 0.0]),
        mo_energy=jnp.asarray([-0.55, 0.18]),
        mo_coeff=jnp.eye(2),
    )


def test_prepare_input_info_extracts_basic_metadata():
    molecule = _make_dummy_molecule()
    info = prepare_input_info(molecule)

    assert info.system_label == "h2_dummy"
    assert info.basis_name == "sto-3g"
    assert info.charge == 0
    assert info.spin == 0
    assert np.isclose(info.electron_count, 2.0, atol=0.0, rtol=0.0)
    assert info.atom_symbols == ("H", "H")
    assert info.atomic_numbers == (1, 1)
    assert info.n_atoms == 2
    assert info.n_ao == 2
    assert info.n_mo == 2
    assert info.n_occ == 1
    assert info.n_vir == 1
    assert info.n_grid == 7
    assert info.has_hfx_local is True


def test_input_info_geometry_helpers_roundtrip_coordinates():
    molecule = _make_dummy_molecule()
    info = prepare_input_info(molecule)

    rows = input_info_atom_rows(info)
    geometry = input_info_to_geometry_string(info)

    assert tuple(symbol for symbol, _ in rows) == ("H", "H")
    assert np.allclose(
        np.asarray([coords for _, coords in rows], dtype=float),
        np.asarray([[0.0, 0.0, -0.35], [0.0, 0.0, 0.35]], dtype=float),
        atol=1e-7,
        rtol=0.0,
    )
    geometry_rows = [line.split() for line in geometry.splitlines()]
    assert [row[0] for row in geometry_rows] == ["H", "H"]
    assert np.allclose(
        np.asarray([[float(value) for value in row[1:]] for row in geometry_rows], dtype=float),
        np.asarray([[0.0, 0.0, -0.35], [0.0, 0.0, 0.35]], dtype=float),
        atol=1e-7,
        rtol=0.0,
    )


def test_target_bundle_roundtrip_and_datum_restore(tmp_path: Path):
    molecule = _make_dummy_molecule()
    bundle = build_molecular_target_bundle(
        molecule,
        system_label="dummy_case",
        target_e0_total_h=jnp.asarray(-1.2345),
        target_excitation_gaps_h=jnp.asarray([0.42, 0.77]),
        target_oscillator_strengths=jnp.asarray([0.11, 0.03]),
        target_orbital_energies=jnp.asarray([-0.55, 0.18]),
        target_orbital_occupations=jnp.asarray([2.0, 0.0]),
    )

    bundle_path = bundle.save(tmp_path / "dummy_targets")
    assert bundle_path.exists()
    assert bundle_path.suffix == ".npz"
    assert not bundle_path.with_suffix(".meta.json").exists()

    loaded = load_molecular_target_bundle(bundle_path)
    assert loaded.input_info.system_label == "dummy_case"
    assert np.isclose(float(np.asarray(loaded.target_e0_total_h)), -1.2345, atol=1e-12)
    assert np.allclose(
        np.asarray(loaded.target_excitation_gaps_h),
        np.asarray([0.42, 0.77]),
        atol=1e-6,
        rtol=0.0,
    )
    assert np.allclose(
        np.asarray(loaded.target_oscillator_strengths),
        np.asarray([0.11, 0.03]),
        atol=1e-6,
        rtol=0.0,
    )

    new_molecule = _make_dummy_molecule()
    datum = load_molecular_datum(bundle_path, new_molecule)
    assert datum.molecule is new_molecule
    assert np.isclose(float(np.asarray(datum.target_e0_total_h)), -1.2345, atol=1e-12)
    assert np.allclose(
        np.asarray(datum.target_orbital_energies),
        np.asarray([-0.55, 0.18]),
        atol=1e-6,
        rtol=0.0,
    )
    assert np.allclose(
        np.asarray(datum.target_orbital_occupations),
        np.asarray([2.0, 0.0]),
        atol=1e-6,
        rtol=0.0,
    )
