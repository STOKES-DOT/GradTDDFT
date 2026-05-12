from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array

from .data.molecule import (
    ANGSTROM_TO_BOHR,
)
from .scf import RKSConfig

BOHR_TO_ANGSTROM = 1.0 / ANGSTROM_TO_BOHR

CoordinateUnit = Literal["angstrom", "bohr"]
EnergyFunction = Callable[[Array], Array]


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
    final_forces: Array
    energy_history: Array
    grad_norm_history: Array


def _normalize_coordinate_unit(unit: str) -> CoordinateUnit:
    unit_norm = str(unit).strip().lower()
    if unit_norm in {"angstrom", "ang", "a"}:
        return "angstrom"
    if unit_norm in {"bohr", "au"}:
        return "bohr"
    raise ValueError(f"Unsupported coordinate unit={unit!r}. Expected 'angstrom' or 'bohr'.")


def _coords_to_bohr(coords: Array, unit: CoordinateUnit) -> Array:
    coords_arr = jnp.asarray(coords, dtype=jnp.float64)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError("coordinates must have shape (natom, 3).")
    if unit == "bohr":
        return coords_arr
    return coords_arr * ANGSTROM_TO_BOHR


def compute_forces(
    energy_fn: EnergyFunction,
    coordinates: Array,
) -> Array:
    """Return Cartesian forces: F = -dE/dR."""

    coords = jnp.asarray(coordinates, dtype=jnp.float64)
    grad = jax.jacfwd(lambda x: jnp.asarray(energy_fn(x), dtype=jnp.float64))(coords)
    return -grad


def run_geometry_optimization(
    energy_fn: EnergyFunction,
    initial_coordinates: Array,
    *,
    config: GeometryOptimizationConfig | None = None,
) -> GeometryOptimizationResult:
    """Optimize Cartesian coordinates with JAX autodiff and Optax Adam."""

    cfg = GeometryOptimizationConfig() if config is None else config
    coords = jnp.asarray(initial_coordinates, dtype=jnp.float64)
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
        return jnp.asarray(energy_fn(flat_coords.reshape(natom, 3)), dtype=jnp.float64)

    def value_and_grad(flat_coords: Array) -> tuple[Array, Array]:
        value = energy_from_flat(flat_coords)
        grad = jax.jacfwd(energy_from_flat)(flat_coords)
        return value, grad
    energy_hist: list[float] = []
    grad_norm_hist: list[float] = []
    converged = False
    steps = 0

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
        if (
            float(grad_norm) < cfg.convergence_grad_norm
            and float(step_norm) < cfg.convergence_step_norm
        ):
            converged = True
            break

    final_energy, final_grad = value_and_grad(flat)
    if not energy_hist:
        energy_hist = [float(final_energy)]
        grad_norm_hist = [float(jnp.linalg.norm(final_grad))]

    return GeometryOptimizationResult(
        converged=converged,
        steps=steps,
        optimized_coordinates=flat.reshape(natom, 3),
        final_energy=float(final_energy),
        final_forces=(-final_grad).reshape(natom, 3),
        energy_history=jnp.asarray(energy_hist),
        grad_norm_history=jnp.asarray(grad_norm_hist),
    )


def make_rks_ground_state_energy_fn(
    *,
    symbols: Sequence[str],
    basis: str,
    xc_spec: str = "pbe",
    charge: int = 0,
    spin: int = 0,
    coordinate_unit: str = "angstrom",
    grids_level: int = 0,
    max_l: int = 3,
    rks_config: RKSConfig | None = None,
) -> EnergyFunction:
    """Explicit SCF geometry differentiation has been removed."""

    del (
        symbols,
        basis,
        xc_spec,
        charge,
        spin,
        coordinate_unit,
        grids_level,
        max_l,
        rks_config,
    )
    raise NotImplementedError(
        "Explicit differentiable SCF geometry optimization has been removed. "
        "Use implicit differential SCF workflows instead."
    )


def run_rks_ground_state_geometry_optimization(
    *,
    symbols: Sequence[str],
    initial_coordinates: Array,
    basis: str,
    xc_spec: str = "pbe",
    charge: int = 0,
    spin: int = 0,
    coordinate_unit: str = "angstrom",
    grids_level: int = 0,
    max_l: int = 3,
    rks_config: RKSConfig | None = None,
    optimization_config: GeometryOptimizationConfig | None = None,
) -> GeometryOptimizationResult:
    """High-level RKS geometry optimization API with autodiff forces."""

    energy_fn = make_rks_ground_state_energy_fn(
        symbols=symbols,
        basis=basis,
        xc_spec=xc_spec,
        charge=charge,
        spin=spin,
        coordinate_unit=coordinate_unit,
        grids_level=grids_level,
        max_l=max_l,
        rks_config=rks_config,
    )
    return run_geometry_optimization(
        energy_fn,
        initial_coordinates,
        config=optimization_config,
    )


__all__ = [
    "BOHR_TO_ANGSTROM",
    "CoordinateUnit",
    "EnergyFunction",
    "GeometryOptimizationConfig",
    "GeometryOptimizationResult",
    "compute_forces",
    "make_rks_ground_state_energy_fn",
    "run_geometry_optimization",
    "run_rks_ground_state_geometry_optimization",
]
