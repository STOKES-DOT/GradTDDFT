from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from jaxtyping import Array, PyTree

from .config import GroundStateTrainingConfig
from .targets import (
    _predict_ground_state_total_energy_from_molecule,
    _resolve_training_molecule_with_mode,
    density_on_grid,
    density_on_grid_spin_resolved,
)


def predict_ground_state_molecule(
    params: PyTree,
    functional: Any,
    molecule: Any,
    *,
    training_config: GroundStateTrainingConfig | None = None,
) -> Any:
    """Resolve the molecule used by a ground-state predictor.

    This mirrors GradDFT's explicit predictor separation:
    fixed-density evaluation reuses the provided reference molecule, while
    self-consistent evaluation returns the differentiable SCF-updated one.
    """

    cfg = GroundStateTrainingConfig() if training_config is None else training_config
    return _resolve_training_molecule_with_mode(
        params,
        functional,
        molecule,
        cfg,
    )


def predict_ground_state_density(
    params: PyTree,
    functional: Any,
    molecule: Any,
    *,
    training_config: GroundStateTrainingConfig | None = None,
    spin_resolved: bool = False,
) -> Array:
    """Predict the grid density under fixed-density or self-consistent evaluation."""

    predicted_molecule = predict_ground_state_molecule(
        params,
        functional,
        molecule,
        training_config=training_config,
    )
    if spin_resolved:
        return density_on_grid_spin_resolved(predicted_molecule)
    return density_on_grid(predicted_molecule)


def make_ground_state_predictor(
    functional: Any,
    *,
    training_config: GroundStateTrainingConfig | None = None,
) -> Callable[[PyTree, Any], tuple[Array, Any]]:
    """Create a reusable predictor returning `(energy, evaluated_molecule)`."""

    cfg = GroundStateTrainingConfig() if training_config is None else training_config

    def predictor(params: PyTree, molecule: Any) -> tuple[Array, Any]:
        predicted_molecule = predict_ground_state_molecule(
            params,
            functional,
            molecule,
            training_config=cfg,
        )
        energy = _predict_ground_state_total_energy_from_molecule(
            params,
            functional,
            predicted_molecule,
        )
        return energy, predicted_molecule

    return predictor


def make_fixed_density_predictor(functional: Any) -> Callable[[PyTree, Any], tuple[Array, Any]]:
    """Create a GradDFT-style non-SCF predictor."""

    return make_ground_state_predictor(
        functional,
        training_config=GroundStateTrainingConfig(mode="fixed_density"),
    )


def make_self_consistent_predictor(
    functional: Any,
    *,
    training_config: GroundStateTrainingConfig | None = None,
) -> Callable[[PyTree, Any], tuple[Array, Any]]:
    """Create a GradDFT-style self-consistent predictor."""

    cfg = GroundStateTrainingConfig() if training_config is None else training_config
    if cfg.mode != "self_consistent":
        cfg = replace(cfg, mode="self_consistent")
    return make_ground_state_predictor(functional, training_config=cfg)
