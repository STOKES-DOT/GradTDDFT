import pytest

from td_graddft.workflows import (
    ExperimentConfig,
    ExperimentPipeline,
    NeuralXCTrainingConfig,
    ReferenceSpecConfig,
    SystemConfig,
    benzene_experiment_config,
    benzene_legacy_experiment_config,
    benzene_strict_jax_experiment_config,
    legacy_benzene_experiment_config,
    legacy_water_experiment_config,
    water_experiment_config,
    water_legacy_experiment_config,
    water_strict_jax_experiment_config,
)
from td_graddft.neural_xc import (
    DEFAULT_NEURAL_XC_COEFFICIENT_PRIOR_MODE,
    DEFAULT_NEURAL_XC_DENSITY_SUPERVISION,
    DEFAULT_NEURAL_XC_SEMILOCAL_XC,
    DEFAULT_INPUT_FEATURE_MODE,
    DEFAULT_NETWORK_ARCHITECTURE,
    DEFAULT_NETWORK_HIDDEN_DIMS,
)


def _dummy_builder():
    return object()


def test_system_config_generates_default_slug_prefix():
    system = SystemConfig(name="Benzene B3LYP/STO-3G", mf_builder=_dummy_builder)
    prefix = system.resolved_prefix("Neural XC Bench")
    assert prefix == "neural_xc_bench_benzene_b3lyp_sto_3g_b3lyp_vs_neural_xc"


def test_experiment_output_config_uses_system_overrides():
    system = SystemConfig(
        name="Water",
        mf_builder=_dummy_builder,
        output_prefix="custom_prefix",
        plot_title="Custom Title",
    )
    config = ExperimentConfig(experiment_name="exp", systems=[system])
    output = config.output_config_for(system)
    assert output.prefix == "custom_prefix"
    assert output.title == "Custom Title"


def test_system_config_accepts_reference_spec_without_mf_builder():
    system = SystemConfig(
        name="Water strict JAX",
        reference_spec=ReferenceSpecConfig(
            atom="O 0 0 0\nH 0 0 1\nH 0 1 0",
            basis="sto-3g",
            xc="pbe",
        ),
    )

    assert system.reference_spec is not None
    assert system.mf_builder is None
    assert system.source_kind == "reference_spec"
    assert system.uses_legacy_mf_builder is False


def test_system_config_marks_mf_builder_as_legacy_source():
    system = SystemConfig(name="legacy", mf_builder=_dummy_builder)

    assert system.source_kind == "mf_builder"
    assert system.uses_legacy_mf_builder is True


def test_system_config_rejects_ambiguous_or_empty_sources():
    with pytest.raises(ValueError, match="exactly one"):
        SystemConfig(name="bad", mf_builder=_dummy_builder, reference_spec=ReferenceSpecConfig(
            atom="H 0 0 0\nH 0 0 0.74",
            basis="sto-3g",
            xc="pbe",
        ))
    with pytest.raises(ValueError, match="exactly one"):
        SystemConfig(name="bad")


def test_experiment_pipeline_routes_reference_spec_systems(monkeypatch):
    calls: list[tuple[str, str, bool]] = []

    def fake_pipeline(**kwargs):
        calls.append(
            (
                "pipeline",
                kwargs["system_label"],
                kwargs["reference_spec"] is not None,
            )
        )
        return "run"

    monkeypatch.setattr(
        "td_graddft.workflows.pipeline.run_neural_xc_spectrum_pipeline",
        fake_pipeline,
    )
    monkeypatch.setattr(
        "td_graddft.workflows.pipeline.print_run_summary",
        lambda *args, **kwargs: None,
    )

    config = ExperimentConfig(
        experiment_name="exp",
        systems=[
            SystemConfig(
                name="Water strict JAX",
                reference_spec=ReferenceSpecConfig(
                    atom="O 0 0 0\nH 0 0 1\nH 0 1 0",
                    basis="sto-3g",
                    xc="pbe",
                ),
            )
        ],
        training=NeuralXCTrainingConfig(),
    )
    result = ExperimentPipeline(config).run()

    assert result.runs == ["run"]
    assert calls == [("pipeline", "Water strict JAX", True)]


def test_workflow_presets_return_non_empty_systems():
    water = water_experiment_config()
    benzene = benzene_experiment_config()
    assert len(water.systems) == 1
    assert len(benzene.systems) == 1
    assert water.systems[0].reference_spec is not None
    assert benzene.systems[0].reference_spec is not None
    assert water.systems[0].mf_builder is None
    assert benzene.systems[0].mf_builder is None
    assert water.simulation.scf_backend == "jax_rks"
    assert benzene.simulation.scf_backend == "jax_rks"


def test_legacy_workflow_presets_preserve_mf_builder_entrypoints():
    water = legacy_water_experiment_config()
    benzene = legacy_benzene_experiment_config()
    water_alias = water_legacy_experiment_config()
    benzene_alias = benzene_legacy_experiment_config()

    assert len(water.systems) == 1
    assert len(benzene.systems) == 1
    assert callable(water.systems[0].mf_builder)
    assert callable(benzene.systems[0].mf_builder)
    assert water_alias.systems[0].mf_builder is not None
    assert benzene_alias.systems[0].mf_builder is not None


def test_strict_jax_workflow_presets_return_reference_specs():
    water = water_strict_jax_experiment_config()
    benzene = benzene_strict_jax_experiment_config()

    assert len(water.systems) == 1
    assert len(benzene.systems) == 1
    assert water.systems[0].reference_spec is not None
    assert benzene.systems[0].reference_spec is not None
    assert water.systems[0].mf_builder is None
    assert benzene.systems[0].mf_builder is None
    assert water.simulation.scf_backend == "jax_rks"
    assert benzene.simulation.scf_backend == "jax_rks"


def test_neural_xc_training_config_accepts_density_supervision():
    config = NeuralXCTrainingConfig(density_supervision="spin_resolved")

    assert config.density_supervision == "spin_resolved"


def test_neural_xc_training_config_accepts_graddft_alignment_options():
    config = NeuralXCTrainingConfig(
        strict_graddft_ground_state=True,
        network_architecture="graddft_residual",
    )

    assert config.strict_graddft_ground_state is True
    assert config.network_architecture == "graddft_residual"


def test_neural_xc_training_config_defaults_follow_neural_xc_defaults():
    config = NeuralXCTrainingConfig()

    assert config.semilocal_xc == DEFAULT_NEURAL_XC_SEMILOCAL_XC
    assert config.density_supervision == DEFAULT_NEURAL_XC_DENSITY_SUPERVISION
    assert config.coefficient_prior_mode == DEFAULT_NEURAL_XC_COEFFICIENT_PRIOR_MODE
    assert config.energy_mse_weight == 0.0
    assert config.energy_mae_weight == 1.0
    assert config.orbital_energy_mse_weight == 0.0
    assert config.orbital_energy_mae_weight == 1.0
    assert config.hidden_dims == DEFAULT_NETWORK_HIDDEN_DIMS
    assert config.network_architecture == DEFAULT_NETWORK_ARCHITECTURE
    assert config.input_feature_mode == DEFAULT_INPUT_FEATURE_MODE


def test_simulation_config_defaults_to_strict_jax_runtime():
    from td_graddft.workflows import SimulationConfig

    config = SimulationConfig()

    assert config.scf_backend == "jax_rks"
    assert config.jax_grid_ao_backend == "jax"
    assert config.jax_compilation_cache_dir is None
    assert config.jax_persistent_cache_min_compile_time_secs > 0.0
    assert config.jax_persistent_cache_min_entry_size_bytes > 0


def test_neural_xc_training_config_accepts_coefficient_prior():
    config = NeuralXCTrainingConfig(
        coefficient_prior_weight=1.0,
        coefficient_prior_values=(0.08, 0.72, 0.19, 0.81, 0.20),
        coefficient_prior_mode="mean",
    )

    assert config.coefficient_prior_weight == 1.0
    assert config.coefficient_prior_values == (0.08, 0.72, 0.19, 0.81, 0.20)
    assert config.coefficient_prior_mode == "mean"
