"""Training package for ground-state fitting and TDDFT transfer."""

from .config import (
    ExcitedStateDatum,
    ExcitedStateTrainingConfig,
    GroundStateCoreDatum,
    GroundStateCoreTrainingConfig,
    GroundStateDatum,
    GroundStateTrainingConfig,
)
from .checkpoints import load_params_checkpoint, save_params_checkpoint
from .targets import (
    density_on_grid,
    density_on_grid_spin_resolved,
    dm21_scf_regularization_delta_energy,
    dm21_scf_regularization_penalty,
    xc_kernel_matching_penalty,
    density_matching_penalty,
    density_stationarity_penalty,
    ground_state_mse_loss,
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
from .losses import make_ground_state_loss, make_self_supervised_rsh_loss
from .trainer import (
    create_train_state,
    create_train_state_from_molecule,
    make_ground_state_loss_and_grad,
    make_ground_state_train_step,
)
from .excited_state_trainer import (
    ExcitedStateFineTuneConfig,
    ExcitedStateFineTuneResult,
    ExcitedStateFineTuner,
)
from .neural_xc_trainer import NeuralXCTrainer
from .results import TrainingResult
from .rsh_optimizer import RSHOptimizer

__all__ = [
    "GroundStateCoreDatum",
    "ExcitedStateDatum",
    "GroundStateDatum",
    "GroundStateCoreTrainingConfig",
    "ExcitedStateTrainingConfig",
    "GroundStateTrainingConfig",
    "load_params_checkpoint",
    "save_params_checkpoint",
    "density_on_grid",
    "density_on_grid_spin_resolved",
    "dm21_scf_regularization_delta_energy",
    "dm21_scf_regularization_penalty",
    "xc_kernel_matching_penalty",
    "density_matching_penalty",
    "density_stationarity_penalty",
    "ground_state_mse_loss",
    "predict_excitation_energies",
    "predict_oscillator_strengths",
    "predict_excitation_spectrum",
    "predict_ground_state_density",
    "predict_ground_state_molecule",
    "predict_ground_state_total_energy",
    "make_fixed_density_predictor",
    "make_ground_state_predictor",
    "make_ground_state_loss",
    "make_self_supervised_rsh_loss",
    "make_self_consistent_predictor",
    "create_train_state",
    "create_train_state_from_molecule",
    "make_ground_state_loss_and_grad",
    "make_ground_state_train_step",
    "ExcitedStateFineTuneConfig",
    "ExcitedStateFineTuneResult",
    "ExcitedStateFineTuner",
    "NeuralXCTrainer",
    "RSHOptimizer",
    "TrainingResult",
]
