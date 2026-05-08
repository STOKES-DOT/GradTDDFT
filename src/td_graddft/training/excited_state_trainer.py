from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import optax
from flax import traverse_util
from flax.core import FrozenDict, freeze, unfreeze
from jaxtyping import Array, PyTree

from .config import GroundStateDatum
from .targets import (
    HARTREE_TO_EV,
    lorentzian_spectrum,
    predict_excitation_energies,
    predict_ground_state_total_energy,
    predict_oscillator_strengths,
)


def _as_dataset(data: GroundStateDatum | Sequence[GroundStateDatum]) -> list[GroundStateDatum]:
    if isinstance(data, GroundStateDatum):
        return [data]
    return list(data)


def _tree_l2_norm(tree: Any, *, sanitize: bool = False) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)

    total = jnp.asarray(0.0, dtype=jnp.float32)
    for leaf in leaves:
        arr = jnp.asarray(leaf)
        if sanitize:
            arr = jnp.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        total = total + jnp.sum(jnp.square(arr.astype(jnp.float32)))
    return jnp.sqrt(total)


def _tree_abs_max(tree: Any, *, sanitize: bool = False) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)

    current_max = jnp.asarray(0.0, dtype=jnp.float32)
    for leaf in leaves:
        arr = jnp.asarray(leaf)
        if sanitize:
            arr = jnp.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        current_max = jnp.maximum(current_max, jnp.max(jnp.abs(arr.astype(jnp.float32))))
    return current_max


def _sanitize_gradients(tree: Any) -> tuple[Any, Array]:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    if not leaves:
        return tree, jnp.asarray(0.0, dtype=jnp.float32)

    cleaned_leaves = []
    nonfinite_total = jnp.asarray(0.0, dtype=jnp.float32)
    element_total = jnp.asarray(0.0, dtype=jnp.float32)
    for leaf in leaves:
        arr = jnp.asarray(leaf)
        cleaned_leaves.append(jnp.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
        nonfinite_total = nonfinite_total + jnp.sum((~jnp.isfinite(arr)).astype(jnp.float32))
        element_total = element_total + jnp.asarray(arr.size, dtype=jnp.float32)

    cleaned_tree = jax.tree_util.tree_unflatten(treedef, cleaned_leaves)
    fraction = nonfinite_total / jnp.maximum(element_total, 1.0)
    return cleaned_tree, fraction


def _parse_path_prefix(prefix: str) -> tuple[str, ...]:
    raw = str(prefix).replace(".", "/")
    return tuple(part for part in raw.split("/") if part)


def _label_tree_for_trainable_prefixes(
    params: PyTree,
    prefixes: Sequence[str],
) -> tuple[PyTree, bool]:
    params_was_frozen = isinstance(params, FrozenDict)
    params_dict = unfreeze(params)
    flat = traverse_util.flatten_dict(params_dict)
    parsed_prefixes = tuple(_parse_path_prefix(prefix) for prefix in prefixes if prefix)
    if not parsed_prefixes:
        raise ValueError("trainable_path_prefixes must contain at least one non-empty prefix.")

    labels_flat: dict[tuple[str, ...], str] = {}
    matched = False
    for path, _ in flat.items():
        normalized = tuple(str(part) for part in path)
        trimmed = normalized[1:] if normalized and normalized[0] == "params" else normalized
        trainable = any(
            normalized[: len(prefix)] == prefix or trimmed[: len(prefix)] == prefix
            for prefix in parsed_prefixes
        )
        labels_flat[path] = "train" if trainable else "freeze"
        matched = matched or trainable
    labels = traverse_util.unflatten_dict(labels_flat)
    if params_was_frozen:
        return freeze(labels), matched
    return labels, matched


def _select_state_targets(
    values: Array | None,
    states: tuple[int, ...],
    *,
    label: str,
    fallback_s1: Array | None = None,
) -> Array:
    if values is None:
        if fallback_s1 is not None and states == (1,):
            return jnp.asarray([fallback_s1], dtype=jnp.asarray(fallback_s1).dtype)
        raise ValueError(f"{label} targets are required for states {states}.")

    arr = jnp.asarray(values)
    if arr.ndim != 1:
        arr = jnp.reshape(arr, (-1,))
    state_indices = tuple(int(state) - 1 for state in states)
    max_index = max(state_indices)
    if int(arr.shape[0]) > max_index:
        return arr[jnp.asarray(state_indices, dtype=jnp.int32)]
    if int(arr.shape[0]) == len(state_indices):
        return arr
    raise ValueError(
        f"{label} targets must have at least {max_index + 1} entries or exactly "
        f"{len(state_indices)} selected entries; got shape {arr.shape}."
    )


@dataclass(frozen=True)
class ExcitedStateFineTuneConfig:
    """Configuration for fixed-density excited-state fine-tuning."""

    steps: int = 500
    learning_rate: float = 1e-3
    gradient_clip_norm: float | None = None
    lr_decay_every: int = 0
    lr_decay_factor: float = 0.5
    excited_states: tuple[int, ...] = (1, 2, 3)
    use_tda: bool = True
    weight_energy: float = 1.0
    energy_loss: Literal["mse", "mae"] = "mse"
    weight_oscillator_strength: float = 0.0
    weight_spectrum: float = 0.0
    weight_ground_state_energy: float = 0.0
    spectrum_nstates: int | None = None
    spectrum_eta_ev: float = 0.15
    freeze_ground_state_params: bool = True
    trainable_path_prefixes: tuple[str, ...] = ("lr_correction",)
    select_params: Literal["best_loss", "final"] = "best_loss"
    log_interval: int = 0

    def __post_init__(self) -> None:
        if int(self.steps) <= 0:
            raise ValueError("steps must be positive.")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive.")
        states = tuple(int(state) for state in self.excited_states)
        if not states:
            raise ValueError("excited_states must contain at least one state index.")
        if any(state <= 0 for state in states):
            raise ValueError("excited_states must use 1-based positive indices.")
        if tuple(sorted(states)) != states:
            raise ValueError("excited_states must be sorted in ascending order.")
        if str(self.energy_loss) not in {"mse", "mae"}:
            raise ValueError("energy_loss must be either 'mse' or 'mae'.")
        object.__setattr__(self, "excited_states", states)


@dataclass(frozen=True)
class ExcitedStateFineTuneResult:
    """Outputs from the excited-state fine-tuning loop."""

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
    """Independent fixed-density fine-tuner for excited-state observables."""

    def __init__(
        self,
        config: ExcitedStateFineTuneConfig,
        functional: Any,
        initial_params: PyTree,
    ) -> None:
        self.config = config
        self.functional = functional
        self.initial_params = initial_params

    def _make_optimizer(self, params: PyTree) -> optax.GradientTransformation:
        if self.config.lr_decay_every > 0:
            schedule = optax.exponential_decay(
                init_value=float(self.config.learning_rate),
                transition_steps=int(self.config.lr_decay_every),
                decay_rate=float(self.config.lr_decay_factor),
                staircase=True,
            )
            base_optimizer = optax.adam(schedule)
        else:
            base_optimizer = optax.adam(float(self.config.learning_rate))

        if self.config.gradient_clip_norm is not None and float(self.config.gradient_clip_norm) > 0.0:
            base_optimizer = optax.chain(
                optax.clip_by_global_norm(float(self.config.gradient_clip_norm)),
                base_optimizer,
            )

        if not self.config.freeze_ground_state_params:
            return base_optimizer

        labels, matched = _label_tree_for_trainable_prefixes(
            params,
            self.config.trainable_path_prefixes,
        )
        if not matched:
            prefixes = ", ".join(self.config.trainable_path_prefixes)
            raise ValueError(
                "freeze_ground_state_params=True but no parameter leaves matched "
                f"trainable_path_prefixes=({prefixes})."
            )
        return optax.multi_transform(
            {
                "train": base_optimizer,
                "freeze": optax.set_to_zero(),
            },
            labels,
        )

    def _datum_loss(
        self,
        params: PyTree,
        datum: GroundStateDatum,
    ) -> tuple[Array, dict[str, Array]]:
        dtype = jnp.asarray(datum.target_total_energy).dtype
        zero = jnp.asarray(0.0, dtype=dtype)
        state_indices = tuple(int(state) - 1 for state in self.config.excited_states)
        max_state = max(self.config.excited_states)

        loss = zero
        metrics: dict[str, Array] = {
            "excitation_mse": zero,
            "excitation_mae": zero,
            "oscillator_strength_mse": zero,
            "oscillator_strength_mae": zero,
            "spectrum_mse": zero,
            "spectrum_mae": zero,
            "ground_state_energy_mse": zero,
            "ground_state_energy_mae": zero,
        }

        if float(self.config.weight_energy) != 0.0:
            predicted = predict_excitation_energies(
                params,
                self.functional,
                datum.molecule,
                nstates=max_state,
                use_tda=bool(self.config.use_tda),
            )
            predicted = predicted[jnp.asarray(state_indices, dtype=jnp.int32)]
            target = _select_state_targets(
                datum.target_excitation_energies,
                self.config.excited_states,
                label="excitation",
                fallback_s1=datum.target_s1_energy,
            ).astype(predicted.dtype)
            residual = predicted - target
            metrics["excitation_mse"] = jnp.mean(residual**2)
            metrics["excitation_mae"] = jnp.mean(jnp.abs(residual))
            excitation_loss = (
                metrics["excitation_mae"]
                if self.config.energy_loss == "mae"
                else metrics["excitation_mse"]
            )
            loss = loss + float(self.config.weight_energy) * excitation_loss

        if float(self.config.weight_oscillator_strength) != 0.0:
            predicted = predict_oscillator_strengths(
                params,
                self.functional,
                datum.molecule,
                nstates=max_state,
                use_tda=bool(self.config.use_tda),
            )
            predicted = predicted[jnp.asarray(state_indices, dtype=jnp.int32)]
            target = _select_state_targets(
                datum.target_oscillator_strengths,
                self.config.excited_states,
                label="oscillator-strength",
            ).astype(predicted.dtype)
            residual = predicted - target
            metrics["oscillator_strength_mse"] = jnp.mean(residual**2)
            metrics["oscillator_strength_mae"] = jnp.mean(jnp.abs(residual))
            loss = loss + float(self.config.weight_oscillator_strength) * metrics[
                "oscillator_strength_mse"
            ]

        if float(self.config.weight_spectrum) != 0.0:
            if datum.target_spectrum_grid_ev is None or datum.target_spectrum_curve is None:
                raise ValueError(
                    "Spectrum supervision requires target_spectrum_grid_ev and "
                    "target_spectrum_curve on every datum."
                )
            spectrum_nstates = (
                max_state
                if self.config.spectrum_nstates is None
                else max(int(self.config.spectrum_nstates), max_state)
            )
            excitation_result = predict_excitation_energies(
                params,
                self.functional,
                datum.molecule,
                nstates=spectrum_nstates,
                use_tda=bool(self.config.use_tda),
            )
            strengths = predict_oscillator_strengths(
                params,
                self.functional,
                datum.molecule,
                nstates=spectrum_nstates,
                use_tda=bool(self.config.use_tda),
            )
            target_grid = jnp.asarray(datum.target_spectrum_grid_ev)
            target_curve = jnp.asarray(datum.target_spectrum_curve)
            predicted_curve = lorentzian_spectrum(
                jnp.asarray(excitation_result) * HARTREE_TO_EV,
                jnp.asarray(strengths),
                target_grid,
                eta=float(self.config.spectrum_eta_ev),
            )
            target_rms = jnp.maximum(jnp.sqrt(jnp.mean(target_curve**2)), 1e-8)
            residual = (predicted_curve - target_curve) / target_rms
            metrics["spectrum_mse"] = jnp.mean(residual**2)
            metrics["spectrum_mae"] = jnp.mean(jnp.abs(residual))
            loss = loss + float(self.config.weight_spectrum) * metrics["spectrum_mse"]

        if float(self.config.weight_ground_state_energy) != 0.0:
            predicted = predict_ground_state_total_energy(
                params,
                self.functional,
                datum.molecule,
            )
            target = jnp.asarray(datum.target_total_energy, dtype=predicted.dtype)
            residual = predicted - target
            metrics["ground_state_energy_mse"] = residual**2
            metrics["ground_state_energy_mae"] = jnp.abs(residual)
            loss = loss + float(self.config.weight_ground_state_energy) * metrics[
                "ground_state_energy_mse"
            ]

        weight = jnp.asarray(float(datum.weight), dtype=loss.dtype)
        return weight * loss, {key: weight * value for key, value in metrics.items()}

    def _loss_and_metrics(
        self,
        params: PyTree,
        dataset: Sequence[GroundStateDatum],
    ) -> tuple[Array, dict[str, Array]]:
        total_loss = jnp.asarray(0.0, dtype=jnp.float32)
        total_weight = jnp.asarray(0.0, dtype=jnp.float32)
        metric_totals = {
            "excitation_mse": jnp.asarray(0.0, dtype=jnp.float32),
            "excitation_mae": jnp.asarray(0.0, dtype=jnp.float32),
            "oscillator_strength_mse": jnp.asarray(0.0, dtype=jnp.float32),
            "oscillator_strength_mae": jnp.asarray(0.0, dtype=jnp.float32),
            "spectrum_mse": jnp.asarray(0.0, dtype=jnp.float32),
            "spectrum_mae": jnp.asarray(0.0, dtype=jnp.float32),
            "ground_state_energy_mse": jnp.asarray(0.0, dtype=jnp.float32),
            "ground_state_energy_mae": jnp.asarray(0.0, dtype=jnp.float32),
        }

        for datum in dataset:
            weight = jnp.asarray(float(datum.weight), dtype=jnp.float32)
            datum_loss, datum_metrics = self._datum_loss(params, datum)
            total_loss = total_loss + datum_loss.astype(jnp.float32)
            total_weight = total_weight + weight
            for key, value in datum_metrics.items():
                metric_totals[key] = metric_totals[key] + jnp.asarray(value, dtype=jnp.float32)

        denom = jnp.maximum(total_weight, 1e-8)
        metrics = {key: value / denom for key, value in metric_totals.items()}
        return total_loss / denom, metrics

    def fine_tune(
        self,
        data: GroundStateDatum | Sequence[GroundStateDatum],
    ) -> ExcitedStateFineTuneResult:
        dataset = _as_dataset(data)
        if not dataset:
            raise ValueError("fine_tune requires at least one training datum.")

        optimizer = self._make_optimizer(self.initial_params)
        opt_state = optimizer.init(self.initial_params)

        def compute_loss(params: PyTree) -> tuple[Array, dict[str, Array]]:
            return self._loss_and_metrics(params, dataset)

        params = self.initial_params
        initial_loss, _ = compute_loss(params)
        best_params = params
        best_loss = float(initial_loss)
        best_step = 0

        loss_history = [best_loss]
        grad_norm_history = [float("nan")]
        grad_abs_max_history = [float("nan")]
        param_update_norm_history = [float("nan")]

        for step in range(1, int(self.config.steps) + 1):
            (loss, _), grads = jax.value_and_grad(compute_loss, has_aux=True)(params)
            cleaned_grads, nonfinite_grad_fraction = _sanitize_gradients(grads)
            updates, opt_state = optimizer.update(cleaned_grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            param_delta = jax.tree_util.tree_map(lambda new, old: new - old, new_params, params)
            params = new_params

            loss_value = float(loss)
            if loss_value < best_loss:
                best_loss = loss_value
                best_step = step
                best_params = params

            loss_history.append(loss_value)
            grad_norm_history.append(float(_tree_l2_norm(cleaned_grads, sanitize=True)))
            grad_abs_max_history.append(float(_tree_abs_max(cleaned_grads, sanitize=True)))
            param_update_norm_history.append(float(_tree_l2_norm(param_delta, sanitize=True)))

            if (
                int(self.config.log_interval) > 0
                and (step % int(self.config.log_interval) == 0 or step == int(self.config.steps))
            ):
                print(
                    "[ExcitedStateFineTuner] "
                    f"step={step} loss={loss_value:.8f} "
                    f"best_loss={best_loss:.8f} "
                    f"nonfinite_grad_fraction={float(nonfinite_grad_fraction):.6f}",
                    flush=True,
                )

        final_loss, _ = compute_loss(params)
        selected_params = best_params if self.config.select_params == "best_loss" else params
        return ExcitedStateFineTuneResult(
            params=selected_params,
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
