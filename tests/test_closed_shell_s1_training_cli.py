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


def test_stream_train_default_update_mode_accumulates_one_epoch_gradient():
    module = _load_training_tool()

    args = module.parse_args(["--reference-csv", "refs.csv", "--stream-train"])

    assert args.stream_update_mode == "accumulate"
    assert module._lr_transition_steps(args, train_size=35) == args.lr_decay_every


def test_stream_train_per_molecule_update_mode_scales_lr_decay_by_train_size():
    module = _load_training_tool()

    args = module.parse_args(
        [
            "--reference-csv",
            "refs.csv",
            "--stream-train",
            "--stream-update-mode",
            "per_molecule",
            "--lr-decay-every",
            "100",
        ]
    )

    assert args.stream_update_mode == "per_molecule"
    assert module._lr_transition_steps(args, train_size=35) == 3500
    assert module._stream_lr_schedule_index(args, step=101, train_size=35) == 3500


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


def test_hdf5_cache_can_read_restricted_hfx_nu_as_chunked_api(tmp_path):
    h5py = pytest.importorskip("h5py")
    from td_graddft.data.hdf5_cache import (
        read_restricted_molecule,
        write_restricted_molecule,
    )
    from td_graddft.scf.molecules import QuadratureGrid, RestrictedMolecule

    hfx_nu = np.arange(2 * 4 * 2 * 2, dtype=np.float64).reshape(2, 4, 2, 2)
    molecule = RestrictedMolecule(
        ao=np.ones((4, 2)),
        grid=QuadratureGrid(weights=np.ones((4,)), coords=np.ones((4, 3))),
        dipole_integrals=np.ones((3, 2, 2)),
        rep_tensor=np.ones((2, 2, 2, 2)),
        mo_coeff=np.ones((2, 2, 2)),
        mo_occ=np.ones((2, 2)),
        mo_energy=np.ones((2, 2)),
        rdm1=np.ones((2, 2, 2)),
        h1e=np.ones((2, 2)),
        nuclear_repulsion=1.0,
        atom_coords=np.ones((4, 3)),
        nocc=1,
        hfx_nu=hfx_nu,
    )
    path = tmp_path / "refs.h5"
    with h5py.File(path, "w") as handle:
        write_restricted_molecule(handle.create_group("molecule"), molecule)
    with h5py.File(path, "r") as handle:
        loaded = read_restricted_molecule(
            handle["molecule"],
            array_backend="host",
            hfx_nu_storage="chunked",
            hfx_nu_chunk_size=2,
        )

    assert loaded.hfx_nu is None
    assert loaded.hfx_nu_api is not None
    assert loaded.hfx_nu_api.shape == hfx_nu.shape
    assert np.allclose(loaded.hfx_nu_api.grid_chunk(1, 3), hfx_nu[:, 1:3])


def test_hdf5_chunked_hfx_nu_reads_at_runtime_under_jit(tmp_path):
    h5py = pytest.importorskip("h5py")
    import jax
    import jax.numpy as jnp

    from td_graddft.neural_xc.inputs import ChunkedHFXNu

    hfx_nu = np.arange(2 * 5 * 2 * 2, dtype=np.float64).reshape(2, 5, 2, 2)
    path = tmp_path / "refs.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("hfx_nu", data=hfx_nu)

    api = ChunkedHFXNu.from_hdf5_dataset(str(path), "hfx_nu", chunk_size=2)

    def chunk_sum(scale):
        return jnp.sum(api.grid_chunk(1, 3) * scale)

    jitted_chunk_sum = jax.jit(chunk_sum)
    jitted_chunk_grad = jax.jit(jax.grad(chunk_sum))

    first = float(jitted_chunk_sum(jnp.asarray(1.0, dtype=jnp.float32)))
    first_grad = float(jitted_chunk_grad(jnp.asarray(1.0, dtype=jnp.float32)))
    updated = hfx_nu.copy()
    updated[:, 1:3] = updated[:, 1:3] + 100.0
    with h5py.File(path, "r+") as handle:
        handle["hfx_nu"][:, 1:3] = updated[:, 1:3]
        handle.flush()

    second = float(jitted_chunk_sum(jnp.asarray(1.0, dtype=jnp.float32)))
    second_grad = float(jitted_chunk_grad(jnp.asarray(1.0, dtype=jnp.float32)))

    assert first == pytest.approx(float(np.sum(hfx_nu[:, 1:3])))
    assert first_grad == pytest.approx(float(np.sum(hfx_nu[:, 1:3])))
    assert second == pytest.approx(float(np.sum(updated[:, 1:3])))
    assert second_grad == pytest.approx(float(np.sum(updated[:, 1:3])))


def test_hdf5_cache_materializes_chunked_hfx_nu_api(tmp_path):
    h5py = pytest.importorskip("h5py")
    from td_graddft.data.hdf5_cache import (
        read_restricted_molecule,
        write_restricted_molecule,
    )
    from td_graddft.neural_xc.inputs import ChunkedHFXNu
    from td_graddft.scf.molecules import QuadratureGrid, RestrictedMolecule

    hfx_nu = np.arange(2 * 5 * 2 * 2, dtype=np.float64).reshape(2, 5, 2, 2)
    molecule = RestrictedMolecule(
        ao=np.ones((5, 2)),
        grid=QuadratureGrid(weights=np.ones((5,)), coords=np.ones((5, 3))),
        dipole_integrals=np.ones((3, 2, 2)),
        rep_tensor=np.ones((2, 2, 2, 2)),
        mo_coeff=np.ones((2, 2, 2)),
        mo_occ=np.ones((2, 2)),
        mo_energy=np.ones((2, 2)),
        rdm1=np.ones((2, 2, 2)),
        h1e=np.ones((2, 2)),
        nuclear_repulsion=1.0,
        atom_coords=np.ones((4, 3)),
        nocc=1,
        hfx_nu=None,
        hfx_nu_api=ChunkedHFXNu.from_dense(hfx_nu, chunk_size=2),
    )
    path = tmp_path / "refs.h5"
    with h5py.File(path, "w") as handle:
        write_restricted_molecule(handle.create_group("molecule"), molecule)
    with h5py.File(path, "r") as handle:
        dataset = handle["molecule"]["hfx_nu"]
        assert dataset.shape == hfx_nu.shape
        assert np.allclose(dataset[:, 2:5], hfx_nu[:, 2:5])
        loaded = read_restricted_molecule(
            handle["molecule"],
            array_backend="host",
            hfx_nu_storage="chunked",
            hfx_nu_chunk_size=2,
        )

    assert loaded.hfx_nu is None
    assert loaded.hfx_nu_api is not None
    assert np.allclose(loaded.hfx_nu_api.materialize(), hfx_nu)


def test_training_cache_uses_chunked_hfx_nu_only_for_large_low_memory_refs(tmp_path):
    h5py = pytest.importorskip("h5py")
    module = _load_training_tool()

    args = module.parse_args(
        [
            "--reference-csv",
            "refs.csv",
            "--input-feature-mode",
            "canonical",
            "--strict-hfx-response-mode",
            "low_memory",
        ]
    )
    path = tmp_path / "refs.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("molecule")
        group.create_dataset("hfx_nu", data=np.ones((2, 3, 2, 2)))
        group.create_dataset("atom_coords", data=np.ones((4, 3)))
        assert (
            module._cache_hfx_nu_storage(
                group,
                args=args,
                input_feature_mode="canonical",
            )
            == "chunked"
        )

        args.strict_hfx_response_mode = "dense"
        assert (
            module._cache_hfx_nu_storage(
                group,
                args=args,
                input_feature_mode="canonical",
            )
            == "array"
        )

        args.strict_hfx_response_mode = "low_memory"
        del group["atom_coords"]
        group.create_dataset("atom_coords", data=np.ones((3, 3)))
        assert (
            module._cache_hfx_nu_storage(
                group,
                args=args,
                input_feature_mode="canonical",
            )
            == "array"
        )


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
        pt2_local=np.array([0.0, 0.0]),
    )
    path = tmp_path / "refs.h5"
    with h5py.File(path, "w") as handle:
        write_unrestricted_molecule(handle.create_group("molecule"), molecule)
    with h5py.File(path, "r") as handle:
        loaded = read_unrestricted_molecule(handle["molecule"], array_backend="host")

    assert isinstance(loaded.ao, np.ndarray)
    assert isinstance(loaded.grid.weights, np.ndarray)
    assert isinstance(loaded.hfx_nu, np.ndarray)
    assert isinstance(loaded.pt2_local, np.ndarray)
    assert loaded.nocc_alpha == 1
    assert loaded.nocc_beta == 0
