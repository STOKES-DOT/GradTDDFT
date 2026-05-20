from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from td_graddft.neural_xc.networks import ResidualMixingMLP


def _load_smoke_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_ccsdt_gpu4pyscf_smoke.py"
    spec = importlib.util.spec_from_file_location("train_ccsdt_gpu4pyscf_smoke", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_epoch_monitor_row_includes_gradient_metrics():
    script = _load_smoke_script()
    metrics = {
        "loss": jnp.asarray(2.0),
        "energy_mae": jnp.asarray([0.75]),
        "normalized_energy_mae": jnp.asarray([0.25]),
        "grad_norm": jnp.asarray([3.0]),
        "raw_grad_norm": jnp.asarray([4.0]),
        "grad_abs_max": jnp.asarray([0.5]),
        "nonfinite_grad_fraction": jnp.asarray([0.25]),
        "param_update_norm": jnp.asarray([0.125]),
        "param_norm": jnp.asarray([8.0]),
        "scf_cycles": jnp.asarray([7.0]),
        "scf_final_rms": jnp.asarray([1e-4]),
    }

    row = script._epoch_monitor_row(
        epoch=2,
        epoch_elapsed=1.5,
        loss=2.0,
        metrics=metrics,
    )
    line = script._format_epoch_log(epoch=2, total_epochs=4, row=row)

    assert row["grad_norm"] == 3.0
    assert row["energy_mae"] == 0.75
    assert row["normalized_energy_mae"] == 0.25
    assert row["raw_grad_norm"] == 4.0
    assert row["grad_abs_max"] == 0.5
    assert row["nonfinite_grad_fraction"] == 0.25
    assert row["param_update_norm"] == 0.125
    assert row["param_norm"] == 8.0
    assert "grad_norm=3.000e+00" in line
    assert "mae_ha=7.500e-01" in line
    assert "raw_grad_norm=4.000e+00" in line
    assert "update_norm=1.250e-01" in line
    assert "nonfinite=2.500e-01" in line


def test_smoke_script_builds_residual_64_network():
    script = _load_smoke_script()

    hidden_dims = script._parse_hidden_dims("64,64,64")
    functional = script._build_functional(
        hidden_dims=hidden_dims,
        architecture="residual",
        sigmoid_scale_factor=1.5,
    )

    assert hidden_dims == (64, 64, 64)
    assert isinstance(functional.model, ResidualMixingMLP)
    assert tuple(functional.model.hidden_dims) == (64, 64, 64)
    assert functional.model.sigmoid_scale_factor == 1.5
    assert functional.input_feature_mode == "canonical"


def test_smoke_script_exposes_hfx_ablation_knobs():
    script = _load_smoke_script()

    assert script._parse_float_tuple("0.0,0.4") == (0.0, 0.4)
    functional = script._build_functional(
        hidden_dims=(64, 64, 64),
        architecture="residual",
        hf_input_mode="total_only",
        hfx_channels=1,
    )

    assert functional.hf_input_mode == "total_only"
    assert functional.hfx_channels == 1
    assert functional.input_feature_mode == "canonical"


def test_smoke_script_optimizer_can_clip_gradients():
    script = _load_smoke_script()

    tx = script._build_optimizer(learning_rate=1e-3, gradient_clip_norm=1e-2)

    assert hasattr(tx, "init")
    assert hasattr(tx, "update")


def test_smoke_script_lr_schedule_decays_by_epoch_boundary():
    script = _load_smoke_script()

    assert script._optimizer_steps_per_epoch("epoch_accum", 80) == 1
    assert script._optimizer_steps_per_epoch("per_batch", 80) == 80

    schedule = script._build_learning_rate_schedule(
        learning_rate=1e-4,
        lr_decay_epochs=500,
        lr_decay_factor=0.5,
        steps_per_epoch=80,
    )

    assert np.isclose(float(schedule(0)), 1e-4)
    assert np.isclose(float(schedule(39999)), 1e-4)
    assert np.isclose(float(schedule(40000)), 5e-5)
    assert np.isclose(float(schedule(80000)), 2.5e-5)


def test_smoke_script_accumulates_weighted_epoch_gradients():
    script = _load_smoke_script()
    grad_sum = None

    grad_sum = script._accumulate_weighted_grads(
        grad_sum,
        {"w": jnp.asarray([1.0, 3.0]), "b": jnp.asarray(2.0)},
        weight=1.0,
    )
    grad_sum = script._accumulate_weighted_grads(
        grad_sum,
        {"w": jnp.asarray([5.0, 7.0]), "b": jnp.asarray(4.0)},
        weight=3.0,
    )
    mean_grads = script._normalize_accumulated_grads(grad_sum, total_weight=4.0)

    np.testing.assert_allclose(np.asarray(mean_grads["w"]), [4.0, 6.0])
    np.testing.assert_allclose(np.asarray(mean_grads["b"]), 3.5)


def test_smoke_script_shuffles_batches_reproducibly_per_epoch():
    script = _load_smoke_script()
    batches = [[0], [1], [2], [3]]

    first = script._batches_for_epoch(batches, epoch=1, shuffle=True, seed=7)
    same = script._batches_for_epoch(batches, epoch=1, shuffle=True, seed=7)
    other = script._batches_for_epoch(batches, epoch=2, shuffle=True, seed=7)

    assert first == same
    assert first != batches
    assert other != first
    assert script._batches_for_epoch(batches, epoch=1, shuffle=False, seed=7) is batches


def test_smoke_script_loads_samples_round_robin_across_h5_groups(tmp_path):
    script = _load_smoke_script()
    h5_path = tmp_path / "samples.h5"
    with h5py.File(h5_path, "w") as f:
        for name, n_conf in (("A", 3), ("B", 2), ("C", 1)):
            grp = f.create_group(name)
            grp.create_dataset("atomic_numbers", data=np.asarray([1, 1], dtype=np.uint8))
            coords = np.zeros((n_conf, 2, 3), dtype=np.float32)
            grp.create_dataset("coordinates", data=coords)
            grp.create_dataset("ccsd(t)_cbs.energy", data=np.arange(n_conf, dtype=np.float64))

    samples = script._load_samples(h5_path, 5, 8, shuffle=False)

    assert [sample["name"] for sample in samples] == ["A", "B", "C", "A", "B"]
    assert [sample["conformer_idx"] for sample in samples] == [0, 0, 0, 1, 1]
    assert script._sample_group_count(samples) == 3


def test_smoke_script_loads_samples_with_seeded_group_shuffle(tmp_path):
    script = _load_smoke_script()
    h5_path = tmp_path / "samples.h5"
    with h5py.File(h5_path, "w") as f:
        for name in ("A", "B", "C", "D"):
            grp = f.create_group(name)
            grp.create_dataset("atomic_numbers", data=np.asarray([1, 1], dtype=np.uint8))
            grp.create_dataset("coordinates", data=np.zeros((2, 2, 3), dtype=np.float32))
            grp.create_dataset("ccsd(t)_cbs.energy", data=np.asarray([0.0, 1.0]))

    first = script._load_samples(h5_path, 4, 8, shuffle=True, seed=11)
    same = script._load_samples(h5_path, 4, 8, shuffle=True, seed=11)
    ordered = script._load_samples(h5_path, 4, 8, shuffle=False, seed=11)

    assert [sample["name"] for sample in first] == [sample["name"] for sample in same]
    assert [sample["name"] for sample in first] != [sample["name"] for sample in ordered]
    assert script._sample_group_count(first) == 4


def test_fixed_density_non_xc_energy_uses_host_pair_matrix():
    script = _load_smoke_script()
    density = np.asarray([[1.0, 0.2], [0.2, 0.7]], dtype=np.float64)
    molecule = SimpleNamespace(
        rdm1=jnp.asarray(np.stack([0.4 * density, 0.6 * density])),
        h1e=jnp.asarray([[0.5, 0.1], [0.1, 0.3]], dtype=jnp.float64),
        eri_pair_matrix=np.eye(3, dtype=np.float64),
        rep_tensor=np.zeros((0,), dtype=np.float64),
        nuclear_repulsion=jnp.asarray(0.4, dtype=jnp.float64),
    )

    energy = script._fixed_density_non_xc_energy(molecule)

    assert np.isclose(energy, 1.975)


def test_smoke_script_marks_build_oom_as_skippable():
    script = _load_smoke_script()

    assert script._is_skippable_build_failure(RuntimeError("RESOURCE_EXHAUSTED: Out of memory"))
    assert script._is_skippable_build_failure(
        RuntimeError("GPU4PySCF exact RKS SCF did not converge")
    )
    assert not script._is_skippable_build_failure(ValueError("bad input"))


def test_final_energy_helper_uses_runtime_predictor(monkeypatch):
    script = _load_smoke_script()
    training_config = object()
    calls = []

    def fake_make_ground_state_predictor(functional, *, training_config=None):
        calls.append((functional, training_config))

        def predictor(params, molecule):
            return jnp.asarray(params["scale"] * molecule.energy), molecule

        return predictor

    monkeypatch.setattr(
        script,
        "make_ground_state_predictor",
        fake_make_ground_state_predictor,
    )
    data = [
        SimpleNamespace(molecule=SimpleNamespace(energy=2.0)),
        SimpleNamespace(molecule=SimpleNamespace(energy=3.0)),
    ]

    predicted = script._predict_final_energies(
        {"scale": 0.5},
        "functional",
        data,
        training_config,
    )

    assert predicted == [1.0, 1.5]
    assert calls == [("functional", training_config)]


def test_smoke_script_resolves_train_test_split_counts():
    script = _load_smoke_script()

    assert script._resolve_split_counts(
        n_samples=100,
        train_samples=80,
        test_samples=20,
    ) == (80, 20)
    assert script._resolve_split_counts(
        n_samples=10,
        train_samples=None,
        test_samples=2,
    ) == (8, 2)

    with pytest.raises(ValueError, match="train/test split"):
        script._resolve_split_counts(
            n_samples=100,
            train_samples=80,
            test_samples=10,
        )


def test_smoke_script_mean_absolute_error_ev_handles_empty_targets():
    script = _load_smoke_script()

    mae = script._mean_absolute_error_ev([1.0, 1.5], [1.25, 1.25])
    empty = script._mean_absolute_error_ev([], [])

    assert np.isclose(mae, 0.25 * script.HARTREE_TO_EV)
    assert np.isnan(empty)


def test_smoke_script_identifies_gpu4pyscf_scf_nonconvergence():
    script = _load_smoke_script()

    assert script._is_gpu4pyscf_scf_nonconvergence(
        RuntimeError("GPU4PySCF exact RKS SCF did not converge.")
    )
    assert not script._is_gpu4pyscf_scf_nonconvergence(RuntimeError("different failure"))


def test_smoke_script_batches_training_data():
    script = _load_smoke_script()

    assert script._make_batches([1, 2, 3], 0) == [[1, 2, 3]]
    assert script._make_batches([1, 2, 3], 2) == [[1, 2], [3]]


def test_smoke_script_batches_training_data_by_shape():
    script = _load_smoke_script()
    items = [
        (jnp.zeros((2,)), 0),
        (jnp.zeros((3,)), 1),
        (jnp.ones((2,)), 2),
    ]

    batches = script._make_batches(items, 2, bucket_by_shape=True)

    assert [[item[1] for item in batch] for batch in batches] == [[0, 2], [1]]


def test_smoke_script_built_data_cache_roundtrip(tmp_path):
    script = _load_smoke_script()
    cache_path = tmp_path / "built.pkl"
    config = script._built_data_cache_config(
        h5_path="dataset.h5",
        n_samples=1,
        sample_buffer=2,
        shuffle_samples=True,
        sample_seed=42,
        max_atoms=8,
        basis="6-31g*",
        ref_xc="b3lyp",
        grids_level=3,
        scf_max_cycle=40,
        scf_conv_tol=1e-9,
        hfx_omega_values=(0.0, 0.4),
        hfx_aux=False,
        compute_response_eri_slices=False,
    )
    datum = SimpleNamespace(molecule=SimpleNamespace(x=jnp.asarray([1.0])), target_total_energy=2.0)

    script._write_built_data_cache(
        cache_path,
        config=config,
        data=[datum],
        samples=[{"name": "sample_0"}],
        built_samples=[{"ccsdt_energy": -1.0}],
        skipped_samples=[],
        reference_energies=[-0.9],
        build_elapsed=1.25,
        fixed_non_xc_offsets=[0.3],
    )
    payload = script._load_built_data_cache(cache_path, config=config)

    assert payload["built_samples"] == [{"ccsdt_energy": -1.0}]
    assert payload["reference_energies"] == [-0.9]
    assert payload["fixed_non_xc_offsets"] == [0.3]
    assert payload["build_elapsed"] == 1.25


def test_smoke_script_built_data_h5_roundtrip(tmp_path):
    script = _load_smoke_script()
    h5_path = tmp_path / "input.hdf5"
    config = script._built_data_cache_config(
        h5_path="dataset.h5",
        n_samples=1,
        sample_buffer=2,
        shuffle_samples=True,
        sample_seed=42,
        max_atoms=8,
        basis="6-31g*",
        ref_xc="b3lyp",
        grids_level=3,
        scf_max_cycle=40,
        scf_conv_tol=1e-9,
        hfx_omega_values=(0.0, 0.4),
        hfx_aux=False,
        compute_response_eri_slices=False,
    )
    sample = {
        "name": "sample_0",
        "conformer_idx": 2,
        "atomic_numbers": np.asarray([1, 1], dtype=np.int32),
        "ccsdt_energy": -1.5,
    }
    datum = script.GroundStateDatum(
        molecule=SimpleNamespace(rdm1=np.asarray([[1.0]])),
        target_total_energy=jnp.asarray(-0.25),
        density_constraint_weight=0.0,
    )
    handle = script._create_built_data_h5(h5_path, config=config, samples=[sample])
    try:
        ref = script._append_built_data_h5_sample(
            handle,
            datum=datum,
            sample=sample,
            reference_energy=-1.4,
            fixed_non_xc_offset=-1.25,
        )
        script._append_built_data_h5_skip(
            handle,
            {"candidate_index": 3, "name": "bad", "conformer_idx": 0, "reason": "OOM"},
        )
    finally:
        handle.close()

    payload = script._load_built_data_h5(h5_path, config=config)
    store = script._H5DatumStore(h5_path)
    try:
        loaded = script._materialize_datum(ref, store)
    finally:
        store.close()

    assert len(payload["refs"]) == 1
    assert payload["built_samples"] == [
        {"name": "sample_0", "conformer_idx": 2, "ccsdt_energy": -1.5}
    ]
    assert payload["skipped_samples"][0]["reason"] == "OOM"
    assert np.isclose(float(loaded.target_total_energy), -0.25)


def test_smoke_script_converts_fixed_density_targets_to_xc_residual():
    script = _load_smoke_script()
    molecule = SimpleNamespace(
        rdm1=jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        h1e=jnp.asarray([[1.0, 0.0], [0.0, 2.0]]),
        rep_tensor=jnp.zeros((2, 2, 2, 2)),
        eri_pair_matrix=jnp.ones((3, 3)),
        df_factors=jnp.ones((1, 2, 2)),
        eri_ovov=jnp.ones((1, 1, 1, 1)),
        eri_ovvo=jnp.ones((1, 1, 1, 1)),
        eri_oovv=jnp.ones((1, 1, 1, 1)),
        nuclear_repulsion=0.5,
        runtime_scf_backend="gpu4pyscf_rks",
        runtime_scf_options=object(),
    )
    datum = script.GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.asarray(5.0),
    )

    residual_data, offsets = script._to_fixed_density_xc_residual_data([datum])

    assert np.allclose(offsets, [3.5])
    assert np.allclose(np.asarray(residual_data[0].target_total_energy), 1.5)
    stripped = residual_data[0].molecule
    assert stripped.runtime_scf_backend is None
    assert stripped.runtime_scf_options is None
    assert stripped.eri_pair_matrix is None
    assert stripped.df_factors is None
    assert stripped.eri_ovov is None
    assert stripped.eri_ovvo is None
    assert stripped.eri_oovv is None
    assert int(jnp.asarray(stripped.rep_tensor).size) == 0


def test_smoke_script_builds_training_config_modes():
    script = _load_smoke_script()

    self_consistent = script._build_training_config(
        training_mode="self_consistent",
        energy_normalization="none",
        energy_loss="mae",
        train_scf_max_cycle=12,
        train_scf_damping=0.25,
        implicit_response_backend="gpu4pyscf_jk",
        implicit_diff_max_iter=6,
    )
    fixed_density = script._build_training_config(
        training_mode="fixed_density",
        energy_normalization="per_atom",
        energy_loss="mae",
        train_scf_max_cycle=12,
        train_scf_damping=0.25,
        implicit_response_backend="gpu4pyscf_jk",
        implicit_diff_max_iter=6,
    )

    assert self_consistent.mode == "self_consistent"
    assert self_consistent.scf_gradient_mode == "impl"
    assert self_consistent.scf_runtime_forward_backend == "gpu4pyscf_rks"
    assert self_consistent.implicit_response_backend == "gpu4pyscf_jk"
    assert self_consistent.energy_mse_weight == 0.0
    assert self_consistent.energy_mae_weight == 1.0
    assert self_consistent.energy_normalization == "none"

    assert fixed_density.mode == "fixed_density"
    assert fixed_density.scf_runtime_forward_backend == "auto"
    assert fixed_density.energy_mse_weight == 0.0
    assert fixed_density.energy_mae_weight == 1.0
    assert fixed_density.energy_normalization == "per_atom"


def test_smoke_script_can_use_graddft_mse_energy_loss():
    script = _load_smoke_script()

    assert script._energy_loss_weights("mae") == (0.0, 1.0)
    assert script._energy_loss_weights("mse") == (1.0, 0.0)

    config = script._build_training_config(
        training_mode="fixed_density",
        energy_normalization="none",
        energy_loss="mse",
        train_scf_max_cycle=12,
        train_scf_damping=0.25,
        implicit_response_backend="jax",
        implicit_diff_max_iter=6,
    )

    assert config.energy_mse_weight == 1.0
    assert config.energy_mae_weight == 0.0
