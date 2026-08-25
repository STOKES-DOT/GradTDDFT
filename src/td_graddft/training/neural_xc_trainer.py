from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from .config import MolecularTrainingDatum, MolecularTrainingConfig
from .trainer import create_train_state_from_molecule, make_molecular_train_step
from .results import TrainingResult

_NEURAL_XC_HISTORY_KEYS = (
    "total_loss",
    "e0_total_mae",
    "grid_density_mse",
    "orbital_energy_mae",
    "scf_cycles",
    "scf_converged",
)


def _empty_history(keys: Sequence[str]) -> dict[str, list[Any]]:
    return {key: [] for key in keys}


def _scalar_metric(metrics: dict[str, Any], key: str, *, default: float = float("nan")) -> float:
    value = metrics.get(key)
    if value is None:
        return float(default)
    arr = jnp.asarray(value)
    if arr.size == 0:
        return float(default)
    return float(jnp.ravel(arr)[0])


def _as_training_data(items: Sequence[Any]) -> MolecularTrainingDatum | tuple[MolecularTrainingDatum, ...]:
    data = tuple(items)
    if not data:
        raise ValueError("NeuralXCTrainer positive-step training requires at least one MolecularTrainingDatum.")
    for item in data:
        if not isinstance(item, MolecularTrainingDatum):
            raise TypeError(
                "NeuralXCTrainer positive-step training currently expects molecules to be "
                "MolecularTrainingDatum instances."
            )
    return data[0] if len(data) == 1 else data


@dataclass
class NeuralXCTrainer:
    functional: Any
    molecules: Sequence[Any] = field(default_factory=tuple)
    basis: Any | None = None

    def kernel(
        self,
        *,
        steps: int,
        learning_rate: float = 1e-4,
        scf_gradient_mode: str | None = None,
        training_config: MolecularTrainingConfig | None = None,
        params: Any = None,
        rng: Any | None = None,
    ) -> TrainingResult:
        if int(steps) < 0:
            raise ValueError("steps must be non-negative.")
        history = _empty_history(_NEURAL_XC_HISTORY_KEYS)
        if int(steps) == 0:
            return TrainingResult(
                functional=self.functional,
                params=params,
                history=history,
                final_metrics={},
            )
        data = _as_training_data(self.molecules)
        first_datum = data[0] if isinstance(data, tuple) else data
        tx = optax.adam(float(learning_rate))
        if params is None:
            key = jax.random.PRNGKey(0) if rng is None else rng
            state = create_train_state_from_molecule(
                self.functional,
                key,
                first_datum.molecule,
                tx,
            )
        else:
            state = TrainState.create(
                apply_fn=self.functional.model.apply,
                params=params,
                tx=tx,
            )
        if training_config is None:
            raise ValueError(
                "positive-step training requires an explicit MolecularTrainingConfig."
            )
        elif scf_gradient_mode is None:
            config = training_config
        else:
            config = replace(training_config, scf_gradient_mode=scf_gradient_mode)
        train_step = make_molecular_train_step(
            self.functional,
            training_config=config,
        )
        final_metrics: dict[str, Any] = {}
        for _ in range(int(steps)):
            state, metrics = train_step(state, data)
            history["total_loss"].append(_scalar_metric(metrics, "total_loss"))
            history["e0_total_mae"].append(_scalar_metric(metrics, "e0_total_mae"))
            history["grid_density_mse"].append(_scalar_metric(metrics, "grid_density_mse"))
            history["orbital_energy_mae"].append(_scalar_metric(metrics, "orbital_energy_mae"))
            history["scf_cycles"].append(_scalar_metric(metrics, "scf_cycles_mean"))
            history["scf_converged"].append(_scalar_metric(metrics, "scf_converged_fraction"))
            final_metrics = {
                key: values[-1]
                for key, values in history.items()
                if values
            }
        return TrainingResult(
            functional=self.functional,
            params=state.params,
            history=history,
            final_metrics=final_metrics,
        )


__all__ = ["NeuralXCTrainer"]
