import math
from types import SimpleNamespace

import pytest

import jax.numpy as jnp
import optax

from td_graddft.scf import GPU4PYSCF_RKS_RUNTIME_BACKEND
from td_graddft.scf.molecules import QuadratureGrid, RestrictedMolecule
from td_graddft.workflows.core import (
    _canonicalize_graddft_ground_state_config,
    _resolve_training_scf_gradient_mode,
    build_spectrum,
    run_pipeline_core,
    run_pipeline_core_from_molecule_spec,
    run_pipeline_core_from_spec,
    train_neural_xc,
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


def test_training_scf_gradient_mode_is_always_implicit():
    config = NeuralXCTrainingConfig(
        scf_gradient_mode="impl",
        density_constraint_weight=0.0,
        stationarity_constraint_weight=0.0,
        training_mode="fixed_density",
    )
    assert _resolve_training_scf_gradient_mode(config) == "impl"


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


def test_train_neural_xc_propagates_gpu4pyscf_runtime_forward_backend(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeFunctional:
        model = SimpleNamespace(apply=lambda *args, **kwargs: None)

        def __init__(self, **kwargs):
            captured["functional_kwargs"] = kwargs

        def init_from_molecule(self, rng, molecule):
            del rng, molecule
            return {"strength": jnp.asarray(0.0)}

        def effective_exchange_fraction(self, params, molecule):
            del params, molecule
            return jnp.asarray(0.0)

    def fake_train_step(functional, *, training_config):
        del functional
        captured["train_step_config"] = training_config

        def _step(state, datum):
            del datum
            return state, {
                "loss": jnp.asarray(0.0),
                "grad_norm": jnp.asarray([0.0]),
                "grad_abs_max": jnp.asarray([0.0]),
                "param_update_norm": jnp.asarray([0.0]),
                "nonfinite_grad_fraction": jnp.asarray([0.0]),
                "density_penalty": jnp.asarray([0.0]),
                "stationarity_penalty": jnp.asarray([0.0]),
                "coefficient_prior_penalty": jnp.asarray([0.0]),
            }

        return _step

    def fake_eval(functional, *, training_config):
        del functional
        captured["eval_config"] = training_config
        metrics = {
            "density_penalty": jnp.asarray([0.0]),
            "stationarity_penalty": jnp.asarray([0.0]),
            "coefficient_prior_penalty": jnp.asarray([0.0]),
        }
        return lambda params, datum: (jnp.asarray(0.0), metrics)

    monkeypatch.setattr("td_graddft.workflows.core.neural_xc.Functional", _FakeFunctional)
    monkeypatch.setattr("td_graddft.workflows.core.optax.adam", lambda lr: optax.sgd(lr))
    monkeypatch.setattr(
        "td_graddft.workflows.core.make_ground_state_train_step",
        fake_train_step,
    )
    monkeypatch.setattr("td_graddft.workflows.core.make_ground_state_eval", fake_eval)
    monkeypatch.setattr(
        "td_graddft.workflows.core.make_ground_state_predictor",
        lambda functional, training_config: (
            lambda params, molecule: (jnp.asarray(0.0), molecule)
        ),
    )

    mo_coeff = jnp.eye(2)
    mo_occ = jnp.asarray([[1.0, 0.0], [1.0, 0.0]])
    mo_energy = jnp.asarray([[-0.8, 0.2], [-0.8, 0.2]])
    density_half = jnp.einsum("pi,i,qi->pq", mo_coeff, mo_occ[0], mo_coeff)
    molecule = RestrictedMolecule(
        ao=jnp.ones((3, 2)),
        grid=QuadratureGrid(weights=jnp.ones((3,))),
        dipole_integrals=jnp.zeros((3, 2, 2)),
        rep_tensor=jnp.zeros((2, 2, 2, 2)),
        mo_coeff=jnp.stack([mo_coeff, mo_coeff]),
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        rdm1=jnp.stack([density_half, density_half]),
        h1e=jnp.eye(2),
        nuclear_repulsion=0.0,
        mf_energy=-1.0,
        overlap_matrix=jnp.eye(2),
        runtime_scf_backend=GPU4PYSCF_RKS_RUNTIME_BACKEND,
    )
    reference = MoleculeRun(
        molecule=molecule,
        nocc=1,
        nvir=1,
        nstates=0,
        nstates_full=0,
        energies_au=jnp.asarray([]),
        oscillator_strengths=jnp.asarray([]),
        scf_elapsed_s=0.0,
        tddft_elapsed_s=0.0,
    )

    train_neural_xc(
        reference,
        NeuralXCTrainingConfig(
            steps=0,
            training_mode="self_consistent",
            scf_runtime_forward_backend="gpu4pyscf_rks",
        ),
        SpectrumGridConfig(),
    )

    assert captured["train_step_config"].scf_runtime_forward_backend == "gpu4pyscf_rks"
    assert captured["eval_config"].scf_runtime_forward_backend == "gpu4pyscf_rks"
