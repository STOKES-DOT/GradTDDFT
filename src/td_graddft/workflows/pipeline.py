from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Any, Callable

from .core import run_pipeline_core
from .config import ExperimentConfig
from .reporting import print_run_summary
from .types import (
    NeuralXCTrainingConfig,
    OutputConfig,
    PipelineRun,
    ReferenceSpecConfig,
    SimulationConfig,
    SpectrumGridConfig,
)


@dataclass(frozen=True)
class ExperimentRun:
    """Collection of per-system runs for an experiment."""

    config: ExperimentConfig
    runs: list[PipelineRun]


def run_neural_xc_spectrum_pipeline(
    *,
    system_label: str,
    training_config: NeuralXCTrainingConfig,
    simulation_config: SimulationConfig,
    spectrum_config: SpectrumGridConfig,
    output_config: OutputConfig,
    mf_builder: Callable[[], Any] | None = None,
    reference_spec: ReferenceSpecConfig | None = None,
) -> PipelineRun:
    """Run reference construction, train Neural_xc, and compare absorption spectra."""

    from .reporting import write_outputs

    reference, training, neural, spectrum = run_pipeline_core(
        training_config=training_config,
        simulation_config=simulation_config,
        spectrum_config=spectrum_config,
        mf_builder=mf_builder,
        reference_spec=reference_spec,
    )
    outputs = write_outputs(
        reference=reference,
        training=training,
        neural=neural,
        spectrum=spectrum,
        output=output_config,
    )
    return PipelineRun(
        system_label=system_label,
        reference=reference,
        training=training,
        neural=neural,
        spectrum=spectrum,
        outputs=outputs,
    )


def run_neural_xc_spectrum_pipeline_from_spec(
    *,
    system_label: str,
    reference_spec: ReferenceSpecConfig,
    training_config: NeuralXCTrainingConfig,
    simulation_config: SimulationConfig,
    spectrum_config: SpectrumGridConfig,
    output_config: OutputConfig,
) -> PipelineRun:
    """Compatibility wrapper around the spec-driven strict-JAX pipeline path."""

    return run_neural_xc_spectrum_pipeline(
        system_label=system_label,
        reference_spec=reference_spec,
        training_config=training_config,
        simulation_config=simulation_config,
        spectrum_config=spectrum_config,
        output_config=output_config,
    )


class ExperimentPipeline:
    """High-level config-driven pipeline runner."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run(self) -> ExperimentRun:
        runs: list[PipelineRun] = []
        for system in self.config.systems:
            output = self.config.output_config_for(system)
            if system.uses_legacy_mf_builder:
                warnings.warn(
                    f"SystemConfig(name={system.name!r}) is using legacy mf_builder. Prefer reference_spec for the strict-JAX runtime.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            run = run_neural_xc_spectrum_pipeline(
                system_label=system.name,
                reference_spec=system.reference_spec,
                mf_builder=system.mf_builder,
                training_config=self.config.training,
                simulation_config=self.config.simulation,
                spectrum_config=self.config.spectrum,
                output_config=output,
            )
            print_run_summary(run, print_all_states=system.print_all_states)
            runs.append(run)
        return ExperimentRun(config=self.config, runs=runs)


def run_experiment(config: ExperimentConfig) -> ExperimentRun:
    """Functional wrapper around the class-based pipeline API."""

    return ExperimentPipeline(config).run()


def run_and_report(
    *,
    system_label: str,
    training_config: NeuralXCTrainingConfig,
    simulation_config: SimulationConfig,
    spectrum_config: SpectrumGridConfig,
    output_config: OutputConfig,
    print_all_states: bool = True,
    mf_builder: Callable[[], Any] | None = None,
    reference_spec: ReferenceSpecConfig | None = None,
) -> PipelineRun:
    """Convenience wrapper: run pipeline and print a terminal summary."""

    run = run_neural_xc_spectrum_pipeline(
        system_label=system_label,
        training_config=training_config,
        simulation_config=simulation_config,
        spectrum_config=spectrum_config,
        output_config=output_config,
        mf_builder=mf_builder,
        reference_spec=reference_spec,
    )
    print_run_summary(run, print_all_states=print_all_states)
    return run


def run_and_report_from_spec(
    *,
    system_label: str,
    reference_spec: ReferenceSpecConfig,
    training_config: NeuralXCTrainingConfig,
    simulation_config: SimulationConfig,
    spectrum_config: SpectrumGridConfig,
    output_config: OutputConfig,
    print_all_states: bool = True,
) -> PipelineRun:
    """Compatibility wrapper around the strict-JAX convenience entrypoint."""

    return run_and_report(
        system_label=system_label,
        reference_spec=reference_spec,
        training_config=training_config,
        simulation_config=simulation_config,
        spectrum_config=spectrum_config,
        output_config=output_config,
        print_all_states=print_all_states,
    )
