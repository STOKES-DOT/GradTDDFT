"""Simplified strict-JAX public API.

This module provides short, stable entry points for the pure-JAX runtime path.
New scripts should prefer these helpers over legacy bridge-style imports.
"""

from __future__ import annotations

from .workflows.types import (
    MoleculeSpecConfig,
    NeuralXCTrainingConfig,
    OutputConfig,
    SimulationConfig,
    SpectrumGridConfig,
)

MoleculeConfig = MoleculeSpecConfig


def build_molecule(
    molecule: MoleculeConfig,
    *,
    simulation: SimulationConfig,
):
    """Build a strict-JAX ground-state molecule from molecule specs."""

    from .workflows.core import run_molecule_from_spec

    return run_molecule_from_spec(molecule, simulation=simulation)


def run_pipeline(
    molecule: MoleculeConfig,
    *,
    training: NeuralXCTrainingConfig,
    simulation: SimulationConfig,
    spectrum: SpectrumGridConfig,
):
    """Run strict-JAX molecule -> training -> TDDFT spectrum core pipeline."""

    from .workflows.core import run_pipeline_core_from_molecule_spec

    return run_pipeline_core_from_molecule_spec(
        molecule_spec=molecule,
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
    """Run the strict-JAX molecule spectrum pipeline and write outputs."""

    from .workflows.pipeline import run_neural_xc_spectrum_pipeline_from_molecule_spec

    return run_neural_xc_spectrum_pipeline_from_molecule_spec(
        system_label=system_label,
        molecule_spec=molecule,
        training_config=training,
        simulation_config=simulation,
        spectrum_config=spectrum,
        output_config=output,
    )


__all__ = [
    "MoleculeConfig",
    "build_molecule",
    "run_pipeline",
    "run_spectrum_pipeline",
]
