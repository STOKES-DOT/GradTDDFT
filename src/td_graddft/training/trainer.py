from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from jaxtyping import Array, PRNGKeyArray

from .config import GroundStateDatum, GroundStateTrainingConfig
from .targets import density_on_grid, ground_state_mse_loss


RuntimeForwardStateProvider = Callable[[Any, Any, Any], Any]


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


def _runtime_forward_implicit_config(
    training_config: GroundStateTrainingConfig | None,
) -> GroundStateTrainingConfig:
    cfg = GroundStateTrainingConfig() if training_config is None else training_config
    return replace(
        cfg,
        mode="self_consistent",
        scf_gradient_mode="implicit_commutator",
        scf_implicit_forward_mode="input_state",
    )


def make_self_consistent_runtime_forward_provider(
    training_config: GroundStateTrainingConfig | None = None,
) -> RuntimeForwardStateProvider:
    """Create a Python-controlled SCF forward provider for implicit training.

    The returned provider runs the self-consistent primal state before
    ``jax.value_and_grad``.  Pair it with
    ``make_runtime_forward_implicit_loss_and_grad`` so the backward pass uses
    the existing implicit commutator VJP rather than differentiating through
    SCF iterations.
    """

    from .targets import _make_differentiable_scf

    cfg = replace(
        GroundStateTrainingConfig() if training_config is None else training_config,
        mode="self_consistent",
        scf_gradient_mode="unrolled",
    )
    scf_solver = _make_differentiable_scf(cfg)

    def provider(params: Any, functional: Any, molecule: Any):
        forward_params = jax.tree_util.tree_map(jax.lax.stop_gradient, params)
        return scf_solver.run_runtime_forward(molecule, functional, forward_params)

    return provider


def _runtime_forward_molecule(
    provider: RuntimeForwardStateProvider,
    params: Any,
    functional: Any,
    molecule: Any,
) -> Any:
    out = provider(params, functional, molecule)
    if isinstance(out, tuple) and len(out) == 2:
        return out[0]
    return out


def _with_runtime_forward_state(
    data: GroundStateDatum | Sequence[GroundStateDatum],
    *,
    params: Any,
    functional: Any,
    provider: RuntimeForwardStateProvider | None,
) -> GroundStateDatum | list[GroundStateDatum]:
    if provider is None:
        return data

    def _one(datum: GroundStateDatum) -> GroundStateDatum:
        molecule = _runtime_forward_molecule(
            provider,
            params,
            functional,
            datum.molecule,
        )
        return replace(datum, molecule=molecule)

    if isinstance(data, GroundStateDatum):
        return _one(data)
    return [_one(datum) for datum in data]


def create_train_state(
    functional: Any,
    rng: PRNGKeyArray,
    sample_density: Array,
    tx: optax.GradientTransformation,
) -> TrainState:
    """Initialize a Flax/Optax train state for a neural XC functional."""

    params = functional.init(rng, sample_density)
    return TrainState.create(apply_fn=functional.model.apply, params=params, tx=tx)


def create_train_state_from_molecule(
    functional: Any,
    rng: PRNGKeyArray,
    molecule: Any,
    tx: optax.GradientTransformation,
) -> TrainState:
    """Initialize a train state from a molecule-like object's density."""

    if hasattr(functional, "init_from_molecule"):
        params = functional.init_from_molecule(rng, molecule)
        return TrainState.create(apply_fn=functional.model.apply, params=params, tx=tx)
    sample_density = density_on_grid(molecule)
    return create_train_state(functional, rng, sample_density, tx)


def make_ground_state_loss_and_grad(
    functional: Any,
    training_config: GroundStateTrainingConfig | None = None,
    loss_fn: Callable[..., tuple[Array, dict[str, Array]]] | None = None,
    predictor: Callable[[Any, Any], tuple[Array, Any]] | None = None,
    runtime_forward_state_provider: RuntimeForwardStateProvider | None = None,
):
    """Create a params-only ground-state objective+gradient kernel.

    This is useful when callers want to JIT only the expensive numerical core
    and keep optimizer state updates outside the compiled graph.
    """

    objective = ground_state_mse_loss if loss_fn is None else loss_fn

    def loss_and_grad(
        params: Any,
        data: GroundStateDatum | Sequence[GroundStateDatum],
    ):
        data_for_loss = _with_runtime_forward_state(
            data,
            params=params,
            functional=functional,
            provider=runtime_forward_state_provider,
        )

        def compute_loss(local_params):
            kwargs = {"training_config": training_config}
            if predictor is not None:
                kwargs["predictor"] = predictor
            return objective(
                local_params,
                functional,
                data_for_loss,
                **kwargs,
            )

        (loss, metrics), grads = jax.value_and_grad(compute_loss, has_aux=True)(params)
        cleaned_grads, nonfinite_grad_fraction = _sanitize_gradients(grads)
        metrics = dict(metrics)
        metrics["loss"] = loss
        metrics["raw_grad_norm"] = jnp.asarray([_tree_l2_norm(grads, sanitize=False)], dtype=loss.dtype)
        metrics["grad_norm"] = jnp.asarray([_tree_l2_norm(cleaned_grads, sanitize=True)], dtype=loss.dtype)
        metrics["grad_abs_max"] = jnp.asarray([_tree_abs_max(cleaned_grads, sanitize=True)], dtype=loss.dtype)
        metrics["nonfinite_grad_fraction"] = jnp.asarray([nonfinite_grad_fraction], dtype=loss.dtype)
        return loss, metrics, cleaned_grads

    return loss_and_grad


def make_runtime_forward_implicit_loss_and_grad(
    functional: Any,
    runtime_forward_state_provider: RuntimeForwardStateProvider,
    training_config: GroundStateTrainingConfig | None = None,
    loss_fn: Callable[..., tuple[Array, dict[str, Array]]] | None = None,
    predictor: Callable[[Any, Any], tuple[Array, Any]] | None = None,
):
    """Create a two-stage loss+grad kernel for runtime-forward implicit SCF.

    The runtime provider is called before ``jax.value_and_grad`` and should
    return a molecule-like object containing the converged forward SCF state.
    Gradients are then computed with the implicit commutator custom VJP using
    that state as the primal density.
    """

    return make_ground_state_loss_and_grad(
        functional,
        training_config=_runtime_forward_implicit_config(training_config),
        loss_fn=loss_fn,
        predictor=predictor,
        runtime_forward_state_provider=runtime_forward_state_provider,
    )


def make_ground_state_train_step(
    functional: Any,
    training_config: GroundStateTrainingConfig | None = None,
    loss_fn: Callable[..., tuple[Array, dict[str, Array]]] | None = None,
    predictor: Callable[[Any, Any], tuple[Array, Any]] | None = None,
    runtime_forward_state_provider: RuntimeForwardStateProvider | None = None,
):
    """Create one ground-state training step.

    Passing ``loss_fn`` mirrors GradDFT's explicit ``train_kernel(loss=...)``
    style while keeping the default TD-GradDFT energy+density objective.
    """

    loss_and_grad = make_ground_state_loss_and_grad(
        functional,
        training_config=training_config,
        loss_fn=loss_fn,
        predictor=predictor,
        runtime_forward_state_provider=runtime_forward_state_provider,
    )

    def train_step(
        state: TrainState,
        data: GroundStateDatum | Sequence[GroundStateDatum],
    ):
        loss, metrics, cleaned_grads = loss_and_grad(state.params, data)
        new_state = state.apply_gradients(grads=cleaned_grads)
        param_delta = jax.tree_util.tree_map(lambda new, old: new - old, new_state.params, state.params)
        metrics = dict(metrics)
        metrics["param_update_norm"] = jnp.asarray(
            [_tree_l2_norm(param_delta, sanitize=True)],
            dtype=loss.dtype,
        )
        metrics["param_norm"] = jnp.asarray([_tree_l2_norm(state.params, sanitize=True)], dtype=loss.dtype)
        return new_state, metrics

    return train_step


def make_runtime_forward_implicit_train_step(
    functional: Any,
    runtime_forward_state_provider: RuntimeForwardStateProvider,
    training_config: GroundStateTrainingConfig | None = None,
    loss_fn: Callable[..., tuple[Array, dict[str, Array]]] | None = None,
    predictor: Callable[[Any, Any], tuple[Array, Any]] | None = None,
):
    return make_ground_state_train_step(
        functional,
        training_config=_runtime_forward_implicit_config(training_config),
        loss_fn=loss_fn,
        predictor=predictor,
        runtime_forward_state_provider=runtime_forward_state_provider,
    )
