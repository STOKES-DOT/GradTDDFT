"""Reusable workflow utilities for training and spectrum benchmarking."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_PUBLIC_EXPORTS = {
    "ExperimentConfig": "config",
    "SystemConfig": "config",
    "ExperimentPipeline": "pipeline",
    "ExperimentRun": "pipeline",
    "MoleculeRun": "types",
    "MoleculeSpecConfig": "types",
    "run_molecule_from_spec": "core",
    "run_pipeline_core_from_molecule_spec": "core",
    "run_pipeline_core_from_spec": "core",
    "run_and_report": "pipeline",
    "run_and_report_from_molecule_spec": "pipeline",
    "run_and_report_from_spec": "pipeline",
    "run_experiment": "pipeline",
    "run_neural_xc_spectrum_pipeline": "pipeline",
    "run_neural_xc_spectrum_pipeline_from_molecule_spec": "pipeline",
    "run_neural_xc_spectrum_pipeline_from_spec": "pipeline",
    "benzene_experiment_config": "presets",
    "benzene_strict_jax_experiment_config": "presets",
    "water_experiment_config": "presets",
    "water_strict_jax_experiment_config": "presets",
    "NeuralExcitedStateRun": "types",
    "NeuralXCTrainingConfig": "types",
    "OutputConfig": "types",
    "OutputPaths": "types",
    "PipelineRun": "types",
    "SimulationConfig": "types",
    "SpectrumGridConfig": "types",
    "SpectrumRun": "types",
    "TrainingRun": "types",
}

__all__ = list(_PUBLIC_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _PUBLIC_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{_PUBLIC_EXPORTS[name]}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
