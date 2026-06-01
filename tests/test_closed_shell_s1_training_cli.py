from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_training_tool():
    path = Path("tools/closed_shell_s1_self_consistent_train.py")
    spec = importlib.util.spec_from_file_location("closed_shell_s1_self_consistent_train", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_closed_shell_s1_training_can_skip_final_evaluation():
    module = _load_training_tool()

    args = module.parse_args(
        [
            "--reference-csv",
            "refs.csv",
            "--skip-final-evaluation",
        ]
    )

    assert args.skip_final_evaluation is True


def test_closed_shell_s1_training_accepts_low_memory_strict_hfx_response_mode():
    module = _load_training_tool()

    args = module.parse_args(
        [
            "--reference-csv",
            "refs.csv",
            "--strict-hfx-response-mode",
            "low_memory",
        ]
    )

    assert args.strict_hfx_response_mode == "low_memory"


def test_stream_train_defaults_to_host_reference_cache():
    module = _load_training_tool()

    args = module.parse_args(["--reference-csv", "refs.csv", "--stream-train"])

    assert module._use_host_reference_cache(args) is True


def test_host_reference_cache_can_be_disabled():
    module = _load_training_tool()

    args = module.parse_args(
        [
            "--reference-csv",
            "refs.csv",
            "--stream-train",
            "--no-host-reference-cache",
        ]
    )

    assert module._use_host_reference_cache(args) is False


def test_reference_cache_defaults_to_hdf5_path():
    module = _load_training_tool()

    args = module.parse_args(["--reference-csv", "refs.csv"])

    assert module._reference_cache_path(args) == Path(
        "outputs/reference_cache/closed_shell_s1_references.h5"
    )


def test_hdf5_cache_can_read_restricted_molecule_on_host(tmp_path):
    h5py = pytest.importorskip("h5py")
    from td_graddft.data.hdf5_cache import (
        read_restricted_molecule,
        write_restricted_molecule,
    )
    from td_graddft.scf.molecules import QuadratureGrid, RestrictedMolecule

    molecule = RestrictedMolecule(
        ao=np.ones((2, 2)),
        grid=QuadratureGrid(weights=np.ones((2,)), coords=np.ones((2, 3))),
        dipole_integrals=np.ones((3, 2, 2)),
        rep_tensor=np.ones((2, 2, 2, 2)),
        mo_coeff=np.ones((2, 2, 2)),
        mo_occ=np.ones((2, 2)),
        mo_energy=np.ones((2, 2)),
        rdm1=np.ones((2, 2, 2)),
        h1e=np.ones((2, 2)),
        nuclear_repulsion=1.0,
        nocc=1,
        hfx_nu=np.ones((2, 2, 2)),
    )
    path = tmp_path / "refs.h5"
    with h5py.File(path, "w") as handle:
        write_restricted_molecule(handle.create_group("molecule"), molecule)
    with h5py.File(path, "r") as handle:
        loaded = read_restricted_molecule(handle["molecule"], array_backend="host")

    assert isinstance(loaded.ao, np.ndarray)
    assert isinstance(loaded.grid.weights, np.ndarray)
    assert isinstance(loaded.hfx_nu, np.ndarray)


def test_hdf5_cache_can_read_unrestricted_molecule_on_host(tmp_path):
    h5py = pytest.importorskip("h5py")
    from td_graddft.data.hdf5_cache import (
        read_unrestricted_molecule,
        write_unrestricted_molecule,
    )
    from td_graddft.scf.molecules import QuadratureGrid, UnrestrictedMolecule

    molecule = UnrestrictedMolecule(
        ao=np.ones((2, 2)),
        grid=QuadratureGrid(weights=np.ones((2,)), coords=np.ones((2, 3))),
        dipole_integrals=np.ones((3, 2, 2)),
        rep_tensor=np.ones((2, 2, 2, 2)),
        mo_coeff=np.ones((2, 2, 2)),
        mo_occ=np.array([[1.0, 0.0], [0.0, 0.0]]),
        mo_energy=np.ones((2, 2)),
        rdm1=np.ones((2, 2, 2)),
        h1e=np.ones((2, 2)),
        nuclear_repulsion=1.0,
        nocc_alpha=1,
        nocc_beta=0,
        hfx_nu=np.ones((2, 2, 2, 2)),
    )
    path = tmp_path / "refs.h5"
    with h5py.File(path, "w") as handle:
        write_unrestricted_molecule(handle.create_group("molecule"), molecule)
    with h5py.File(path, "r") as handle:
        loaded = read_unrestricted_molecule(handle["molecule"], array_backend="host")

    assert isinstance(loaded.ao, np.ndarray)
    assert isinstance(loaded.grid.weights, np.ndarray)
    assert isinstance(loaded.hfx_nu, np.ndarray)
    assert loaded.nocc_alpha == 1
    assert loaded.nocc_beta == 0
