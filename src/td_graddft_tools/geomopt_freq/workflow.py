from __future__ import annotations

from dataclasses import dataclass, field

from jaxtyping import Array

from .frequencies import (
    FrequencyAnalysisConfig,
    FrequencyAnalysisResult,
    run_frequency_analysis,
)
from .objectives import EnergySurface
from .optimization import (
    GeometryOptimizationConfig,
    GeometryOptimizationResult,
    run_geometry_optimization,
)


@dataclass(frozen=True)
class GeometryWorkflowConfig:
    optimization: GeometryOptimizationConfig = field(
        default_factory=GeometryOptimizationConfig
    )
    frequencies: FrequencyAnalysisConfig = field(
        default_factory=FrequencyAnalysisConfig
    )


@dataclass(frozen=True)
class GeometryWorkflowResult:
    optimization: GeometryOptimizationResult
    frequencies: FrequencyAnalysisResult


def run_geometry_workflow(
    surface: EnergySurface,
    initial_coordinates: Array,
    masses_amu: Array,
    config: GeometryWorkflowConfig | None = None,
) -> GeometryWorkflowResult:
    """Run geometry optimization followed by harmonic frequency analysis."""

    cfg = GeometryWorkflowConfig() if config is None else config
    opt_result = run_geometry_optimization(
        surface,
        initial_coordinates,
        config=cfg.optimization,
    )
    freq_result = run_frequency_analysis(
        surface,
        opt_result.optimized_coordinates,
        masses_amu,
        config=cfg.frequencies,
    )
    return GeometryWorkflowResult(
        optimization=opt_result,
        frequencies=freq_result,
    )

