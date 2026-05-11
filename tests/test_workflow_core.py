import math

import pytest

import jax.numpy as jnp

from td_graddft.workflows.core import (
    _canonicalize_graddft_ground_state_config,
    _resolve_training_scf_gradient_mode,
    build_spectrum,
    run_pipeline_core,
    run_pipeline_core_from_molecule_spec,
    run_pipeline_core_from_spec,
)
from td_graddft.workflows.types import (
    MoleculeRun,
    MoleculeSpecConfig,
    NeuralExcitedStateRun,
    NeuralXCTrainingConfig,
    SimulationConfig,
    SpectrumGridConfig,
)


def test_build_spectrum_handles_empty_neural_states():
    reference = MoleculeRun(
        molecule=object(),
        nocc=1,
        nvir=1,
        nstates=1,
        nstates_full=1,
        energies_au=jnp.asarray([0.4]),
        oscillator_strengths=jnp.asarray([0.2]),
        scf_elapsed_s=0.0,
        tddft_elapsed_s=0.0,
    )
    neural = NeuralExcitedStateRun(
        solver_label="TDA fallback",
        energies_au=jnp.asarray([]),
        oscillator_strengths=jnp.asarray([]),
        elapsed_s=0.0,
    )
    spectrum = build_spectrum(
        reference,
        neural,
        SpectrumGridConfig(grid_min_ev=0.0, grid_points=300),
        SimulationConfig(),
    )

    assert spectrum.grid_ev.shape == (300,)
    assert spectrum.reference_curve.shape == (300,)
    assert spectrum.neural_curve.shape == (300,)
    assert spectrum.compared_states == 0
    assert math.isnan(spectrum.low_energy_mae_ev)


def test_auto_scf_gradient_mode_prefers_unrolled_without_density_constraints():
    config = NeuralXCTrainingConfig(
        scf_gradient_mode="auto",
        density_constraint_weight=0.0,
        stationarity_constraint_weight=0.0,
        training_mode="fixed_density",
    )
    assert _resolve_training_scf_gradient_mode(config) == "unrolled"


def test_auto_scf_gradient_mode_prefers_implicit_with_density_constraints():
    config = NeuralXCTrainingConfig(
        scf_gradient_mode="auto",
        density_constraint_weight=1e-3,
        training_mode="fixed_density",
    )
    assert _resolve_training_scf_gradient_mode(config) == "implicit_commutator"


def test_strict_graddft_ground_state_canonicalizes_network_and_loss_shape():
    config = NeuralXCTrainingConfig(
        strict_graddft_ground_state=True,
        hidden_dims=(32, 32),
        energy_mse_weight=1.0,
        energy_mae_weight=0.0,
        input_feature_mode="enhanced",
        network_architecture="simple_mlp",
        density_supervision="spin_summed",
    )

    aligned = _canonicalize_graddft_ground_state_config(config)

    assert aligned.network_architecture == "graddft_residual"
    assert aligned.input_feature_mode == "canonical"
    assert aligned.energy_mse_weight == 0.0
    assert aligned.energy_mae_weight == 1.0
    assert aligned.orbital_energy_mse_weight == 0.0
    assert aligned.orbital_energy_mae_weight == 1.0
    assert aligned.density_supervision == "spin_resolved"


def test_strict_graddft_ground_state_rejects_excited_state_constraints():
    config = NeuralXCTrainingConfig(
        strict_graddft_ground_state=True,
        s1_constraint_weight=0.1,
    )

    with pytest.raises(ValueError, match="strict_graddft_ground_state"):
        _canonicalize_graddft_ground_state_config(config)


def test_run_pipeline_core_canonicalizes_strict_mode_before_reference_build(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_reference(mf, **kwargs):
        captured["compute_local_hfx_features"] = kwargs["compute_local_hfx_features"]
        captured["compute_local_pt2_features"] = kwargs["compute_local_pt2_features"]
        captured["hfx_omega_values"] = kwargs["hfx_omega_values"]
        captured["hfx_chunk_size"] = kwargs["hfx_chunk_size"]
        return "reference"

    def fake_train_neural_xc(reference, config, spectrum_config):
        captured["train_config"] = config
        return "training"

    monkeypatch.setattr("td_graddft.workflows.core.run_reference", fake_run_reference)
    monkeypatch.setattr("td_graddft.workflows.core.train_neural_xc", fake_train_neural_xc)
    monkeypatch.setattr(
        "td_graddft.workflows.core.run_neural_tddft",
        lambda reference, training, simulation_config: "neural",
    )
    monkeypatch.setattr(
        "td_graddft.workflows.core.build_spectrum",
        lambda reference, neural, spectrum_config, simulation_config: "spectrum",
    )

    training_config = NeuralXCTrainingConfig(
        strict_graddft_ground_state=True,
        input_feature_mode="enhanced",
        network_architecture="simple_mlp",
        dm21_hfx_omega_values=(0.0, 0.4),
        dm21_hfx_chunk_size=128,
    )
    reference, training, neural, spectrum = run_pipeline_core(
        mf_builder=object,
        training_config=training_config,
        simulation_config=SimulationConfig(),
        spectrum_config=SpectrumGridConfig(),
    )

    aligned = captured["train_config"]

    assert reference == "reference"
    assert training == "training"
    assert neural == "neural"
    assert spectrum == "spectrum"
    assert captured["compute_local_hfx_features"] is True
    assert captured["compute_local_pt2_features"] is False
    assert captured["hfx_omega_values"] == (0.0, 0.4)
    assert captured["hfx_chunk_size"] == 128
    assert aligned.input_feature_mode == "canonical"
    assert aligned.network_architecture == "graddft_residual"


def test_run_pipeline_core_from_spec_uses_strict_jax_reference_path(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_molecule_from_spec(
        spec,
        *,
        simulation,
        compute_local_hfx_features=False,
        compute_local_hfx_aux=False,
        compute_local_pt2_features=False,
        hfx_omega_values=(0.0, 0.4),
        hfx_chunk_size=512,
    ):
        captured["spec"] = spec
        captured["simulation"] = simulation
        captured["compute_local_hfx_features"] = compute_local_hfx_features
        captured["compute_local_hfx_aux"] = compute_local_hfx_aux
        captured["compute_local_pt2_features"] = compute_local_pt2_features
        captured["hfx_omega_values"] = hfx_omega_values
        captured["hfx_chunk_size"] = hfx_chunk_size
        return "reference"

    def fake_train_neural_xc(reference, config, spectrum_config):
        captured["train_config"] = config
        return "training"

    monkeypatch.setattr(
        "td_graddft.workflows.core.run_molecule_from_spec",
        fake_run_molecule_from_spec,
    )
    monkeypatch.setattr("td_graddft.workflows.core.train_neural_xc", fake_train_neural_xc)
    monkeypatch.setattr(
        "td_graddft.workflows.core.run_neural_tddft",
        lambda reference, training, simulation_config: "neural",
    )
    monkeypatch.setattr(
        "td_graddft.workflows.core.build_spectrum",
        lambda reference, neural, spectrum_config, simulation_config: "spectrum",
    )

    spec = MoleculeSpecConfig(
        atom="H 0 0 0\nH 0 0 0.74",
        basis="sto-3g",
        xc="pbe",
        unit="Angstrom",
    )
    training_config = NeuralXCTrainingConfig(strict_graddft_ground_state=True)
    reference, training, neural, spectrum = run_pipeline_core_from_molecule_spec(
        molecule_spec=spec,
        training_config=training_config,
        simulation_config=SimulationConfig(scf_backend="jax_rks", jax_grid_ao_backend="jax"),
        spectrum_config=SpectrumGridConfig(),
    )

    aligned = captured["train_config"]
    assert reference == "reference"
    assert training == "training"
    assert neural == "neural"
    assert spectrum == "spectrum"
    assert captured["spec"] == spec
    assert captured["compute_local_hfx_features"] is True
    assert captured["compute_local_hfx_aux"] is False
    assert captured["compute_local_pt2_features"] is False
    assert captured["hfx_omega_values"] == (0.0, 0.4)
    assert aligned.input_feature_mode == "canonical"
    assert aligned.network_architecture == "graddft_residual"


def test_run_pipeline_core_requests_local_pt2_features_when_pt2_channel_enabled(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_molecule_from_spec(
        spec,
        *,
        simulation,
        compute_local_hfx_features=False,
        compute_local_hfx_aux=False,
        compute_local_pt2_features=False,
        hfx_omega_values=(0.0, 0.4),
        hfx_chunk_size=512,
    ):
        captured["compute_local_hfx_features"] = compute_local_hfx_features
        captured["compute_local_pt2_features"] = compute_local_pt2_features
        return "reference"

    monkeypatch.setattr(
        "td_graddft.workflows.core.run_molecule_from_spec",
        fake_run_molecule_from_spec,
    )
    monkeypatch.setattr(
        "td_graddft.workflows.core.train_neural_xc",
        lambda reference, config, spectrum_config: "training",
    )
    monkeypatch.setattr(
        "td_graddft.workflows.core.run_neural_tddft",
        lambda reference, training, simulation_config: "neural",
    )
    monkeypatch.setattr(
        "td_graddft.workflows.core.build_spectrum",
        lambda reference, neural, spectrum_config, simulation_config: "spectrum",
    )

    spec = MoleculeSpecConfig(
        atom="H 0 0 0\nH 0 0 0.74",
        basis="sto-3g",
        xc="pbe",
        unit="Angstrom",
    )
    training_config = NeuralXCTrainingConfig(
        include_pt2_channel=True,
        input_feature_mode="enhanced",
    )
    run_pipeline_core_from_molecule_spec(
        molecule_spec=spec,
        training_config=training_config,
        simulation_config=SimulationConfig(scf_backend="jax_rks", jax_grid_ao_backend="jax"),
        spectrum_config=SpectrumGridConfig(),
    )

    assert captured["compute_local_hfx_features"] is True
    assert captured["compute_local_pt2_features"] is True
