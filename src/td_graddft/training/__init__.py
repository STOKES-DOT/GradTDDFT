"""Differentiable molecular training API."""

from .config import MolecularTrainingConfig, MolecularTrainingDatum
from .checkpoints import load_params_checkpoint, save_params_checkpoint
from .targets import (
    density_on_grid,
    density_on_grid_spin_resolved,
    density_matrix_matching_penalty,
    dm21_scf_regularization_delta_energy,
    dm21_scf_regularization_penalty,
    xc_kernel_matching_penalty,
    density_matching_penalty,
    density_stationarity_penalty,
    molecular_loss,
    predict_excitation_energies,
    predict_oscillator_strengths,
    predict_excitation_spectrum,
    predict_ground_state_total_energy,
)
from .predictors import (
    make_fixed_density_predictor,
    make_ground_state_predictor,
    make_self_consistent_predictor,
    predict_ground_state_density,
    predict_ground_state_molecule,
)
from .trainer import (
    create_train_state,
    create_train_state_from_molecule,
    make_molecular_eval,
    make_molecular_loss_and_grad,
    make_molecular_train_step,
)
from .excited_state_trainer import (
    ExcitedStateFineTuneConfig,
    ExcitedStateFineTuneResult,
    ExcitedStateFineTuner,
)
from .neural_xc_trainer import NeuralXCTrainer
from .results import TrainingResult

__all__ = [
    "MolecularTrainingDatum",
    "MolecularTrainingConfig",
    "load_params_checkpoint",
    "save_params_checkpoint",
    "density_on_grid",
    "density_on_grid_spin_resolved",
    "density_matrix_matching_penalty",
    "dm21_scf_regularization_delta_energy",
    "dm21_scf_regularization_penalty",
    "xc_kernel_matching_penalty",
    "density_matching_penalty",
    "density_stationarity_penalty",
    "molecular_loss",
    "predict_excitation_energies",
    "predict_oscillator_strengths",
    "predict_excitation_spectrum",
    "predict_ground_state_density",
    "predict_ground_state_molecule",
    "predict_ground_state_total_energy",
    "make_fixed_density_predictor",
    "make_ground_state_predictor",
    "make_self_consistent_predictor",
    "create_train_state",
    "create_train_state_from_molecule",
    "make_molecular_eval",
    "make_molecular_loss_and_grad",
    "make_molecular_train_step",
    "ExcitedStateFineTuneConfig",
    "ExcitedStateFineTuneResult",
    "ExcitedStateFineTuner",
    "NeuralXCTrainer",
    "TrainingResult",
]
