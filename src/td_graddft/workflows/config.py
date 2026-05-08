from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .types import (
    NeuralXCTrainingConfig,
    OutputConfig,
    ReferenceSpecConfig,
    SimulationConfig,
    SpectrumGridConfig,
)


def _slugify(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    return lowered or "system"


@dataclass(frozen=True)
class SystemConfig:
    """Single molecular system entry for experiment pipelines.

    `reference_spec` is the preferred strict-JAX source. `mf_builder` is kept as a
    legacy compatibility entrypoint for PySCF-backed workflows.
    """

    name: str
    mf_builder: Callable[[], Any] | None = None
    reference_spec: ReferenceSpecConfig | None = None
    output_prefix: str | None = None
    plot_title: str | None = None
    reference_label: str = "PySCF TDDFT"
    neural_label_template: str = "JAX libxc + Neural_xc TDDFT ({solver})"
    print_all_states: bool = False

    def __post_init__(self) -> None:
        has_mf_builder = self.mf_builder is not None
        has_reference_spec = self.reference_spec is not None
        if has_mf_builder == has_reference_spec:
            raise ValueError(
                "SystemConfig requires exactly one of mf_builder or reference_spec."
            )

    def resolved_prefix(self, experiment_name: str) -> str:
        if self.output_prefix:
            return self.output_prefix
        return f"{_slugify(experiment_name)}_{_slugify(self.name)}_b3lyp_vs_neural_xc"

    @property
    def source_kind(self) -> str:
        if self.reference_spec is not None:
            return "reference_spec"
        return "mf_builder"

    @property
    def uses_legacy_mf_builder(self) -> bool:
        return self.mf_builder is not None

    def resolved_title(self) -> str:
        if self.plot_title:
            return self.plot_title
        return f"{self.name} Absorption Spectrum: B3LYP vs Neural_xc"


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration for multi-system runs."""

    experiment_name: str
    systems: Sequence[SystemConfig]
    training: NeuralXCTrainingConfig = field(default_factory=NeuralXCTrainingConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    spectrum: SpectrumGridConfig = field(default_factory=SpectrumGridConfig)
    output_dir: Path = Path("outputs")
    write_training_curves: bool = True

    def output_config_for(self, system: SystemConfig) -> OutputConfig:
        return OutputConfig(
            outdir=self.output_dir,
            prefix=system.resolved_prefix(self.experiment_name),
            title=system.resolved_title(),
            reference_label=system.reference_label,
            neural_label_template=system.neural_label_template,
            write_training_curves=self.write_training_curves,
        )
