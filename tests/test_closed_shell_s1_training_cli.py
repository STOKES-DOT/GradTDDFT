from __future__ import annotations

import importlib.util
import json
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


def _load_evaluation_tool():
    _load_training_tool()
    path = Path("tools/evaluate_closed_shell_checkpoint.py")
    spec = importlib.util.spec_from_file_location("evaluate_closed_shell_checkpoint", path)
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


def test_streaming_skip_final_evaluation_skips_final_step_eval():
    module = _load_training_tool()

    one_step = module.parse_args(
        [
            "--reference-csv",
            "refs.csv",
            "--stream-train",
            "--steps",
            "1",
            "--skip-final-evaluation",
        ]
    )
    multi_step = module.parse_args(
        [
            "--reference-csv",
            "refs.csv",
            "--stream-train",
            "--steps",
            "5",
            "--eval-interval",
            "2",
            "--skip-final-evaluation",
        ]
    )

    assert module._streaming_should_eval_step(one_step, step=1) is False
    assert module._streaming_should_eval_step(multi_step, step=1) is False
    assert module._streaming_should_eval_step(multi_step, step=2) is True
    assert module._streaming_should_eval_step(multi_step, step=5) is False
    assert module._streaming_should_log_train_step(multi_step, step=1) is True


def test_closed_shell_s1_training_accepts_scf_hfx_grid_block_size():
    module = _load_training_tool()

    args = module.parse_args(
        [
            "--reference-csv",
            "refs.csv",
            "--scf-hfx-grid-block-size",
            "256",
        ]
    )

    assert args.scf_hfx_grid_block_size == 256


def test_closed_shell_s1_training_accepts_functional_hfx_channel_toggle():
    module = _load_training_tool()

    default_args = module.parse_args(["--reference-csv", "refs.csv"])
    hfx_args = module.parse_args(
        [
            "--reference-csv",
            "refs.csv",
            "--include-hfx-channel",
        ]
    )

    assert default_args.include_hfx_channel is False
    assert hfx_args.include_hfx_channel is True


def test_closed_shell_s1_training_rejects_obsolete_response_grid_chunk_size_alias():
    module = _load_training_tool()

    args = module.parse_args(
        [
            "--reference-csv",
            "refs.csv",
            "--scf-hfx-grid-block-size",
            "256",
        ]
    )
    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--reference-csv",
                "refs.csv",
                "--response-grid-chunk-size",
                "128",
            ]
        )

    assert args.scf_hfx_grid_block_size == 256


def test_closed_shell_s1_training_accepts_response_hf_mode_without_none():
    module = _load_training_tool()

    default_args = module.parse_args(["--reference-csv", "refs.csv"])
    approx_args = module.parse_args(
        ["--reference-csv", "refs.csv", "--response-hf-mode", "approx"]
    )
    strict_args = module.parse_args(
        ["--reference-csv", "refs.csv", "--response-hf-mode", "strict"]
    )
    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--reference-csv",
                "refs.csv",
                "--response-hf-mode",
                "none",
            ]
        )

    assert default_args.response_hf_mode == "approx"
    assert approx_args.response_hf_mode == "approx"
    assert strict_args.response_hf_mode == "strict"


def test_closed_shell_checkpoint_eval_restores_hfx_response_metadata_and_rejects_old_chunk_args(tmp_path):
    module = _load_evaluation_tool()
    checkpoint = tmp_path / "params.msgpack"
    checkpoint.write_bytes(b"")
    Path(str(checkpoint) + ".meta.json").write_text(
        json.dumps(
            {
                "include_hfx_channel": True,
                "response_hf_mode": "strict",
                "include_pt2_channel": True,
            }
        ),
        encoding="utf-8",
    )

    args = module.parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--reference-csv",
            "refs.csv",
            "--systems",
            "water",
            "--outdir",
            str(tmp_path / "out"),
        ]
    )
    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--checkpoint",
                str(checkpoint),
                "--reference-csv",
                "refs.csv",
                "--systems",
                "water",
                "--outdir",
                str(tmp_path / "out"),
                "--response-grid-chunk-size",
                "128",
            ]
        )
    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--checkpoint",
                str(checkpoint),
                "--reference-csv",
                "refs.csv",
                "--systems",
                "water",
                "--outdir",
                str(tmp_path / "out"),
                "--strict-hfx-response-mode",
                "low_memory",
            ]
        )

    applied = module._apply_checkpoint_metadata(args)

    assert applied.include_hfx_channel is True
    assert applied.response_hf_mode == "strict"
    assert applied.include_pt2_channel is True


def test_closed_shell_s1_training_accepts_functional_hfx_channel_toggle():
    module = _load_training_tool()

    default_args = module.parse_args(["--reference-csv", "refs.csv"])
    hfx_args = module.parse_args(
        [
            "--reference-csv",
            "refs.csv",
            "--include-hfx-channel",
        ]
    )

    assert default_args.include_hfx_channel is False
    assert hfx_args.include_hfx_channel is True


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


def test_reference_jk_backend_is_switchable_and_part_of_cache_key():
    module = _load_training_tool()

    full_args = module.parse_args(["--reference-csv", "refs.csv"])
    df_args = module.parse_args(
        [
            "--reference-csv",
            "refs.csv",
            "--reference-jk-backend",
            "df",
        ]
    )
    row = module.ReferenceRow(
        system="water",
        split="train",
        atom="O 0 0 0; H 0 0 1; H 0 1 0",
        unit="Angstrom",
        charge=0,
        spin=0,
        basis="sto-3g",
        ccsd_total_energy_h=-75.0,
        s1_excitation_h=0.3,
    )

    assert full_args.reference_jk_backend == "full"
    assert df_args.reference_jk_backend == "df"
    assert module._reference_cache_key(
        row,
        args=full_args,
        input_feature_mode="canonical",
    ) != module._reference_cache_key(
        row,
        args=df_args,
        input_feature_mode="canonical",
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


def test_dense_chunked_hfx_nu_padded_reads_with_dynamic_scan_start_under_jit():
    import jax
    import jax.numpy as jnp

    from td_graddft.neural_xc.inputs import ChunkedHFXNu, hfx_nu_grid_chunk_padded

    hfx_nu = np.arange(2 * 5 * 2 * 2, dtype=np.float64).reshape(2, 5, 2, 2)
    path = tmp_path / "refs.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("hfx_nu", data=hfx_nu)

    api = ChunkedHFXNu.from_hdf5_dataset(str(path), "hfx_nu", chunk_size=2)

    def chunk_sum(scale):
        return jnp.sum(api.grid_chunk_padded(jnp.asarray(1, dtype=jnp.int32), 2) * scale)

    @jax.jit
    def scan_chunk_sums():
        def body(carry, chunk_idx):
            start = chunk_idx * 2
            chunk = hfx_nu_grid_chunk_padded(api, start, 2, n_omega=1)
            return carry + jnp.sum(chunk), None

        total, _ = jax.lax.scan(body, jnp.asarray(0.0, dtype=jnp.float64), jnp.arange(3))
        return total

    assert float(scan_chunk_sums()) == pytest.approx(float(np.sum(hfx_nu[:1])))


def test_dense_chunked_hfx_nu_padded_reads_with_dynamic_scan_start_under_jit():
    import jax
    import jax.numpy as jnp

    from td_graddft.neural_xc.inputs import ChunkedHFXNu, hfx_nu_grid_chunk_padded

    hfx_nu = np.arange(2 * 5 * 2 * 2, dtype=np.float64).reshape(2, 5, 2, 2)
    api = ChunkedHFXNu.from_dense(hfx_nu, chunk_size=2)

    @jax.jit
    def scan_chunk_sums():
        def body(carry, chunk_idx):
            start = chunk_idx * 2
            chunk = hfx_nu_grid_chunk_padded(api, start, 2, n_omega=1)
            return carry + jnp.sum(chunk), None

        total, _ = jax.lax.scan(body, jnp.asarray(0.0, dtype=jnp.float64), jnp.arange(3))
        return total

    assert float(scan_chunk_sums()) == pytest.approx(float(np.sum(hfx_nu[:1])))


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


def test_streaming_preserves_chunked_hfx_nu_api():
    module = _load_training_tool()
    from td_graddft.neural_xc.inputs import ChunkedHFXNu
    from td_graddft.scf.molecules import QuadratureGrid, RestrictedMolecule
    from td_graddft.training import ExcitedStateDatum, GroundStateCoreDatum, GroundStateDatum

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
    datum = GroundStateDatum.from_parts(
        molecule,
        core=GroundStateCoreDatum(
            target_total_energy=np.asarray(-1.0),
            target_density=np.ones((5,)),
            density_constraint_weight=0.25,
        ),
        excited_state=ExcitedStateDatum(
            target_s1_energy=np.asarray(2.0),
            s1_constraint_weight=0.5,
        ),
        weight=3.0,
    )

    seen: list[GroundStateDatum] = []

    def eval_kernel(params, local_datum):
        del params
        seen.append(local_datum)
        return np.asarray(0.0), {
            "s1_mae": np.asarray([0.0]),
            "s1_mse": np.asarray([0.0]),
        }

    loss, metrics = module._streaming_average_eval(None, (datum,), eval_kernel)

    assert datum.molecule.hfx_nu is None
    assert datum.molecule.hfx_nu_api is not None
    assert loss == pytest.approx(0.0)
    assert metrics["s1_mae"] == pytest.approx(0.0)
    assert len(seen) == 1
    streamed = seen[0]
    assert streamed.molecule.hfx_nu is None
    assert streamed.molecule.hfx_nu_api is datum.molecule.hfx_nu_api
    assert np.allclose(streamed.molecule.hfx_nu_api.materialize(), hfx_nu)
    assert streamed.weight == pytest.approx(3.0)
    assert streamed.density_constraint_weight == pytest.approx(0.25)
    assert streamed.s1_constraint_weight == pytest.approx(0.5)
    assert streamed.target_total_energy == pytest.approx(-1.0)
    assert streamed.target_s1_energy == pytest.approx(2.0)
    assert np.allclose(streamed.target_density, np.ones((5,)))


def test_training_cache_uses_chunked_hfx_nu_only_for_large_canonical_refs(tmp_path):
    h5py = pytest.importorskip("h5py")
    module = _load_training_tool()

    args = module.parse_args(
        [
            "--reference-csv",
            "refs.csv",
            "--input-feature-mode",
            "canonical",
        ]
    )
    hfx_args = module.parse_args(
        [
            "--reference-csv",
            "refs.csv",
            "--input-feature-mode",
            "canonical",
            "--include-hfx-channel",
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
            == "array"
        )
        assert (
            module._cache_hfx_nu_storage(
                group,
                args=hfx_args,
                input_feature_mode="canonical",
            )
            == "chunked"
        )

        del group["atom_coords"]
        group.create_dataset("atom_coords", data=np.ones((3, 3)))
        assert (
            module._cache_hfx_nu_storage(
                group,
                args=hfx_args,
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
