from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array

from .objectives import EnergySurface


@dataclass(frozen=True)
class GeometryOptimizationConfig:
    """Gradient-based geometry optimization settings."""

    max_steps: int = 300
    learning_rate: float = 3e-2
    grad_clip_norm: float = 5.0
    convergence_grad_norm: float = 1e-5
    convergence_step_norm: float = 1e-6


@dataclass(frozen=True)
class GeometryOptimizationResult:
    converged: bool
    steps: int
    optimized_coordinates: Array
    final_energy: float
    final_gradient: Array
    energy_history: Array
    grad_norm_history: Array


def run_geometry_optimization(
    surface: EnergySurface,
    initial_coordinates: Array,
    config: GeometryOptimizationConfig | None = None,
) -> GeometryOptimizationResult:
    """Optimize Cartesian coordinates with JAX autodiff and Optax Adam."""

    cfg = GeometryOptimizationConfig() if config is None else config
    coords = jnp.asarray(initial_coordinates)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("initial_coordinates must have shape (natom, 3).")

    natom = int(coords.shape[0])
    flat = coords.reshape(-1)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adam(cfg.learning_rate),
    )
    opt_state = optimizer.init(flat)

    def energy_from_flat(flat_coords: Array) -> Array:
        return surface.energy(flat_coords.reshape(natom, 3))

    value_and_grad = jax.value_and_grad(energy_from_flat)
    energy_hist: list[float] = []
    grad_norm_hist: list[float] = []
    converged = False
    steps = 0
    final_grad = jnp.zeros_like(flat)
    final_energy = jnp.asarray(0.0)

    for step in range(1, cfg.max_steps + 1):
        energy, grad = value_and_grad(flat)
        updates, opt_state = optimizer.update(grad, opt_state, flat)
        new_flat = optax.apply_updates(flat, updates)

        grad_norm = jnp.linalg.norm(grad)
        step_norm = jnp.linalg.norm(new_flat - flat)
        energy_hist.append(float(energy))
        grad_norm_hist.append(float(grad_norm))

        flat = new_flat
        steps = step
        final_grad = grad
        final_energy = energy
        if (
            float(grad_norm) < cfg.convergence_grad_norm
            and float(step_norm) < cfg.convergence_step_norm
        ):
            converged = True
            break

    # Report consistent terminal values on the optimized geometry.
    final_energy, final_grad = value_and_grad(flat)
    if not energy_hist:
        energy_hist = [float(final_energy)]
        grad_norm_hist = [float(jnp.linalg.norm(final_grad))]

    return GeometryOptimizationResult(
        converged=converged,
        steps=steps,
        optimized_coordinates=flat.reshape(natom, 3),
        final_energy=float(final_energy),
        final_gradient=final_grad.reshape(natom, 3),
        energy_history=jnp.asarray(energy_hist),
        grad_norm_history=jnp.asarray(grad_norm_hist),
    )

