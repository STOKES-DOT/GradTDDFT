"""Reusable workflow utilities for training and spectrum benchmarking."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "ExperimentConfig": "config",
    "SystemConfig": "config",
    "ExperimentPipeline": "pipeline",
    "ExperimentRun": "pipeline",
    "run_and_report": "pipeline",
    "run_and_report_from_spec": "pipeline",
    "run_experiment": "pipeline",
    "run_neural_xc_spectrum_pipeline": "pipeline",
    "run_neural_xc_spectrum_pipeline_from_spec": "pipeline",
    "benzene_experiment_config": "presets",
    "benzene_legacy_experiment_config": "presets",
    "benzene_strict_jax_experiment_config": "presets",
    "legacy_benzene_experiment_config": "presets",
    "legacy_water_experiment_config": "presets",
    "water_experiment_config": "presets",
    "water_legacy_experiment_config": "presets",
    "water_strict_jax_experiment_config": "presets",
    "NeuralExcitedStateRun": "types",
    "NeuralXCTrainingConfig": "types",
    "OutputConfig": "types",
    "OutputPaths": "types",
    "PipelineRun": "types",
    "ReferenceSpecConfig": "types",
    "ReferenceRun": "types",
    "SimulationConfig": "types",
    "SpectrumGridConfig": "types",
    "SpectrumRun": "types",
    "TrainingRun": "types",
    "run_pipeline_core_from_spec": "core",
    "run_reference_from_spec": "core",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{_EXPORTS[name]}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
