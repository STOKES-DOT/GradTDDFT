from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import jax
import optax
from flax import traverse_util
from flax.core import FrozenDict, freeze, unfreeze
from jaxtyping import PyTree

from .config import MolecularTrainingConfig, MolecularTrainingDatum
from .targets import molecular_loss
from .trainer import _sanitize_gradients, _tree_abs_max, _tree_l2_norm


def _as_dataset(
    data: MolecularTrainingDatum | Sequence[MolecularTrainingDatum],
) -> list[MolecularTrainingDatum]:
    return [data] if isinstance(data, MolecularTrainingDatum) else list(data)


def _parse_path_prefix(prefix: str) -> tuple[str, ...]:
    return tuple(
        part for part in str(prefix).replace(".", "/").split("/") if part
    )


def _label_tree_for_trainable_prefixes(
    params: PyTree,
    prefixes: Sequence[str],
) -> tuple[PyTree, bool]:
    params_was_frozen = isinstance(params, FrozenDict)
    flat = traverse_util.flatten_dict(unfreeze(params))
    parsed_prefixes = tuple(_parse_path_prefix(prefix) for prefix in prefixes if prefix)
    if not parsed_prefixes:
        raise ValueError("trainable_path_prefixes must contain a non-empty prefix.")

    labels_flat = {}
    matched = False
    for path in flat:
        normalized = tuple(str(part) for part in path)
        trimmed = normalized[1:] if normalized[:1] == ("params",) else normalized
        trainable = any(
            normalized[: len(prefix)] == prefix or trimmed[: len(prefix)] == prefix
            for prefix in parsed_prefixes
        )
        labels_flat[path] = "train" if trainable else "freeze"
        matched = matched or trainable
    labels = traverse_util.unflatten_dict(labels_flat)
    return (freeze(labels) if params_was_frozen else labels), matched


@dataclass(frozen=True)
class ExcitedStateFineTuneConfig:
    """Optimizer policy for a molecular loss used in fine-tuning."""

    steps: int = 500
    learning_rate: float = 1e-3
    gradient_clip_norm: float | None = None
    lr_decay_every: int = 0
    lr_decay_factor: float = 0.5
    freeze_ground_state_params: bool = True
    trainable_path_prefixes: tuple[str, ...] = ("lr_correction",)
    select_params: Literal["best_loss", "final"] = "best_loss"
    log_interval: int = 0

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.select_params not in {"best_loss", "final"}:
            raise ValueError("select_params must be 'best_loss' or 'final'.")


@dataclass(frozen=True)
class ExcitedStateFineTuneResult:
    params: PyTree
    best_params: PyTree
    initial_loss: float
    final_loss: float
    best_loss: float
    best_step: int
    loss_history: tuple[float, ...]
    grad_norm_history: tuple[float, ...]
    grad_abs_max_history: tuple[float, ...]
    param_update_norm_history: tuple[float, ...]


class ExcitedStateFineTuner:
    """Optimize parameters against the canonical molecular loss."""

    def __init__(
        self,
        config: ExcitedStateFineTuneConfig,
        training_config: MolecularTrainingConfig,
        functional: Any,
        initial_params: PyTree,
    ) -> None:
        self.config = config
        self.training_config = training_config
        self.functional = functional
        self.initial_params = initial_params

    def _make_optimizer(self, params: PyTree) -> optax.GradientTransformation:
        if self.config.lr_decay_every > 0:
            learning_rate = optax.exponential_decay(
                init_value=self.config.learning_rate,
                transition_steps=self.config.lr_decay_every,
                decay_rate=self.config.lr_decay_factor,
                staircase=True,
            )
        else:
            learning_rate = self.config.learning_rate
        optimizer = optax.adam(learning_rate)
        if self.config.gradient_clip_norm is not None:
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.config.gradient_clip_norm),
                optimizer,
            )
        if not self.config.freeze_ground_state_params:
            return optimizer

        labels, matched = _label_tree_for_trainable_prefixes(
            params,
            self.config.trainable_path_prefixes,
        )
        if not matched:
            raise ValueError(
                "No parameter leaves matched trainable_path_prefixes="
                f"{self.config.trainable_path_prefixes}."
            )
        return optax.multi_transform(
            {"train": optimizer, "freeze": optax.set_to_zero()},
            labels,
        )

    def fine_tune(
        self,
        data: MolecularTrainingDatum | Sequence[MolecularTrainingDatum],
    ) -> ExcitedStateFineTuneResult:
        dataset = _as_dataset(data)
        if not dataset:
            raise ValueError("fine_tune requires at least one training datum.")

        optimizer = self._make_optimizer(self.initial_params)
        opt_state = optimizer.init(self.initial_params)

        def compute_loss(params: PyTree):
            return molecular_loss(
                params,
                self.functional,
                dataset,
                training_config=self.training_config,
            )

        params = self.initial_params
        initial_loss, _ = compute_loss(params)
        best_params = params
        best_loss = float(initial_loss)
        best_step = 0
        loss_history = [best_loss]
        grad_norm_history = [float("nan")]
        grad_abs_max_history = [float("nan")]
        param_update_norm_history = [float("nan")]

        for step in range(1, self.config.steps + 1):
            (loss, _), grads = jax.value_and_grad(compute_loss, has_aux=True)(params)
            grads, nonfinite_fraction = _sanitize_gradients(grads)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            delta = jax.tree_util.tree_map(
                lambda new, old: new - old,
                new_params,
                params,
            )
            params = new_params

            loss_value = float(loss)
            if loss_value < best_loss:
                best_loss = loss_value
                best_step = step
                best_params = params
            loss_history.append(loss_value)
            grad_norm_history.append(float(_tree_l2_norm(grads, sanitize=True)))
            grad_abs_max_history.append(float(_tree_abs_max(grads, sanitize=True)))
            param_update_norm_history.append(float(_tree_l2_norm(delta, sanitize=True)))

            if self.config.log_interval > 0 and (
                step % self.config.log_interval == 0 or step == self.config.steps
            ):
                print(
                    "[ExcitedStateFineTuner] "
                    f"step={step} loss={loss_value:.8f} best_loss={best_loss:.8f} "
                    f"nonfinite_grad_fraction={float(nonfinite_fraction):.6f}",
                    flush=True,
                )

        final_loss, _ = compute_loss(params)
        selected = best_params if self.config.select_params == "best_loss" else params
        return ExcitedStateFineTuneResult(
            params=selected,
            best_params=best_params,
            initial_loss=float(initial_loss),
            final_loss=float(final_loss),
            best_loss=best_loss,
            best_step=best_step,
            loss_history=tuple(loss_history),
            grad_norm_history=tuple(grad_norm_history),
            grad_abs_max_history=tuple(grad_abs_max_history),
            param_update_norm_history=tuple(param_update_norm_history),
        )
