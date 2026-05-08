"""Simplified strict-JAX public API.

This module provides short, stable entry points for the pure-JAX runtime path.
New scripts should prefer these helpers over legacy bridge-style imports.
"""

from __future__ import annotations

from .workflows.types import (
    NeuralXCTrainingConfig,
    OutputConfig,
    ReferenceSpecConfig,
    SimulationConfig,
    SpectrumGridConfig,
)

MoleculeConfig = ReferenceSpecConfig


def build_reference(
    molecule: MoleculeConfig,
    *,
    simulation: SimulationConfig,
):
    """Build a strict-JAX reference from molecule specs."""

    from .workflows.core import run_reference_from_spec

    return run_reference_from_spec(molecule, simulation=simulation)


def run_pipeline(
    molecule: MoleculeConfig,
    *,
    training: NeuralXCTrainingConfig,
    simulation: SimulationConfig,
    spectrum: SpectrumGridConfig,
):
    """Run strict-JAX reference -> training -> TDDFT spectrum core pipeline."""

    from .workflows.core import run_pipeline_core_from_spec

    return run_pipeline_core_from_spec(
        reference_spec=molecule,
        training_config=training,
        simulation_config=simulation,
        spectrum_config=spectrum,
    )


def run_spectrum_pipeline(
    *,
    system_label: str,
    molecule: MoleculeConfig,
    training: NeuralXCTrainingConfig,
    simulation: SimulationConfig,
    spectrum: SpectrumGridConfig,
    output: OutputConfig,
):
    """Run the strict-JAX spectrum pipeline and write outputs."""

    from .workflows.pipeline import run_neural_xc_spectrum_pipeline_from_spec

    return run_neural_xc_spectrum_pipeline_from_spec(
        system_label=system_label,
        reference_spec=molecule,
        training_config=training,
        simulation_config=simulation,
        spectrum_config=spectrum,
        output_config=output,
    )


__all__ = [
    "MoleculeConfig",
    "build_reference",
    "run_pipeline",
    "run_spectrum_pipeline",
]
