from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from .config import GroundStateTrainingConfig
from .losses import make_self_supervised_rsh_loss
from .neural_xc_trainer import _empty_history
from .neural_xc_trainer import _scalar_metric
from .results import TrainingResult
from .trainer import create_train_state_from_molecule, make_ground_state_train_step

_RSH_HISTORY_KEYS = (
    "loss",
    "omega",
    "alpha",
    "beta",
    "ip_error",
    "ea_error",
)


def _as_rsh_training_data(items: Sequence[Any]) -> Any | tuple[Any, ...]:
    data = tuple(items)
    if not data:
        raise ValueError("RSHOptimizer positive-step training requires at least one molecule.")
    return data[0] if len(data) == 1 else data


def _molecule_for_initialization(data: Any) -> Any:
    first = data[0] if isinstance(data, tuple) else data
    return first.molecule if hasattr(first, "molecule") else first


def _default_rsh_training_config(
    training_config: GroundStateTrainingConfig | None,
) -> GroundStateTrainingConfig:
    if training_config is not None:
        return training_config
    return GroundStateTrainingConfig(
        mode="self_consistent",
        scf_gradient_mode="implicit_commutator",
        scf_require_convergence=False,
    )


def _mean_metrics(metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(metrics) == 1:
        return dict(metrics[0])
    keys = set().union(*(metric.keys() for metric in metrics))
    averaged: dict[str, Any] = {}
    for key in keys:
        values = [metric[key] for metric in metrics if key in metric]
        if values:
            averaged[key] = sum(jnp.asarray(value) for value in values) / len(values)
    return averaged


def _average_loss_over_data(
    loss_fn: Callable[..., tuple[Any, dict[str, Any]]],
) -> Callable[..., tuple[Any, dict[str, Any]]]:
    def _loss(params: Any, functional: Any, data: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        items = data if isinstance(data, tuple) else (data,)
        losses = []
        metric_rows = []
        for item in items:
            value, metrics = loss_fn(params, functional, item, **kwargs)
            losses.append(jnp.asarray(value))
            metric_rows.append(metrics)
        loss_value = sum(losses) / len(losses)
        metrics_out = _mean_metrics(metric_rows)
        metrics_out["loss"] = loss_value
        return loss_value, metrics_out

    return _loss


def _build_rsh_loss(
    functional: Any,
    loss: str | Callable[..., tuple[Any, dict[str, Any]]],
    training_config: GroundStateTrainingConfig,
) -> Callable[..., tuple[Any, dict[str, Any]]]:
    if callable(loss):
        return _average_loss_over_data(loss)
    if loss == "koopmans_ip_ea":
        return _average_loss_over_data(
            make_self_supervised_rsh_loss(
                functional,
                training_config=training_config,
                janak_weight=0.0,
                fractional_weight=0.0,
                koopmans_ip_weight=1.0,
                koopmans_ea_weight=0.0,
                koopmans_lumo_ea_weight=1.0,
                prior_weight=1e-3,
            )
        )
    if loss in {"janak", "self_supervised"}:
        return _average_loss_over_data(
            make_self_supervised_rsh_loss(
                functional,
                training_config=training_config,
                janak_weight=1.0,
                fractional_weight=0.0,
                koopmans_ip_weight=0.0,
                koopmans_ea_weight=0.0,
                koopmans_lumo_ea_weight=0.0,
                prior_weight=1e-3,
            )
        )
    raise ValueError(
        "RSHOptimizer currently supports loss='koopmans_ip_ea', "
        "loss='janak', loss='self_supervised', or a callable loss."
    )


def _append_rsh_history(history: dict[str, list[Any]], metrics: dict[str, Any]) -> None:
    sr = _scalar_metric(metrics, "sr_hf_fraction")
    lr = _scalar_metric(metrics, "lr_hf_fraction")
    history["loss"].append(_scalar_metric(metrics, "loss"))
    history["omega"].append(_scalar_metric(metrics, "omega"))
    history["alpha"].append(sr)
    history["beta"].append(lr - sr)
    history["ip_error"].append(_scalar_metric(metrics, "koopmans_ip_mae"))
    history["ea_error"].append(
        _scalar_metric(
            metrics,
            "koopmans_lumo_ea_mae",
            default=_scalar_metric(metrics, "koopmans_ea_mae"),
        )
    )


@dataclass
class RSHOptimizer:
    functional: Any
    molecules: Sequence[Any] = field(default_factory=tuple)
    basis: Any | None = None

    def kernel(
        self,
        *,
        steps: int,
        learning_rate: float = 1e-3,
        loss: str | Callable[..., tuple[Any, dict[str, Any]]] = "koopmans_ip_ea",
        params: Any = None,
        rng: Any | None = None,
        training_config: GroundStateTrainingConfig | None = None,
    ) -> TrainingResult:
        if int(steps) < 0:
            raise ValueError("steps must be non-negative.")
        history = _empty_history(_RSH_HISTORY_KEYS)
        if int(steps) > 0:
            data = _as_rsh_training_data(self.molecules)
            first_molecule = _molecule_for_initialization(data)
            tx = optax.adam(float(learning_rate))
            if params is None:
                key = jax.random.PRNGKey(0) if rng is None else rng
                state = create_train_state_from_molecule(
                    self.functional,
                    key,
                    first_molecule,
                    tx,
                )
            else:
                state = TrainState.create(
                    apply_fn=self.functional.model.apply,
                    params=params,
                    tx=tx,
                )
            resolved_training_config = _default_rsh_training_config(training_config)
            loss_fn = _build_rsh_loss(
                self.functional,
                loss,
                resolved_training_config,
            )
            train_step = make_ground_state_train_step(
                self.functional,
                training_config=resolved_training_config,
                loss_fn=loss_fn,
            )
            final_metrics: dict[str, Any] = {}
            for _ in range(int(steps)):
                state, metrics = train_step(state, data)
                _append_rsh_history(history, metrics)
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
        return TrainingResult(
            functional=self.functional,
            params=params,
            history=history,
            final_metrics={},
        )


__all__ = ["RSHOptimizer"]
