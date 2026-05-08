import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from td_graddft.geomopt import (
    GeometryOptimizationConfig,
    compute_forces,
    make_rks_ground_state_energy_fn,
    run_geometry_optimization,
)
from td_graddft.scf import RKSConfig


def _h2_coords(r_angstrom: float) -> jnp.ndarray:
    r = jnp.asarray(r_angstrom, dtype=jnp.float64)
    return jnp.asarray(
        [
            [0.0, 0.0, -0.5 * r],
            [0.0, 0.0, +0.5 * r],
        ],
        dtype=jnp.float64,
    )


def _make_energy_fn():
    return make_rks_ground_state_energy_fn(
        symbols=("H", "H"),
        basis="sto-3g",
        xc_spec="pbe",
        coordinate_unit="angstrom",
        grids_level=0,
        max_l=1,
        rks_config=RKSConfig(
            xc_spec="pbe",
            max_cycle=12,
            conv_tol=1e-10,
            conv_tol_density=1e-8,
            damping=0.2,
            iteration_backend="lax",
        ),
    )


def test_rks_ground_state_force_is_finite_for_h2():
    jax.config.update("jax_enable_x64", True)
    energy_fn = _make_energy_fn()
    coords = _h2_coords(1.60)

    forces = compute_forces(energy_fn, coords)
    assert np.isfinite(np.asarray(forces)).all()
    assert float(jnp.linalg.norm(forces)) > 1e-10
    # Translational invariance: net force should be close to zero.
    assert abs(float(jnp.sum(forces[:, 2]))) < 1e-4


def test_rks_ground_state_geometry_optimization_decreases_energy():
    jax.config.update("jax_enable_x64", True)
    energy_fn = _make_energy_fn()
    coords0 = _h2_coords(1.80)
    e0 = float(energy_fn(coords0))

    result = run_geometry_optimization(
        energy_fn,
        coords0,
        config=GeometryOptimizationConfig(
            max_steps=4,
            learning_rate=0.01,
            grad_clip_norm=2.0,
        ),
    )
    assert np.isfinite(float(result.final_energy))
    assert np.isfinite(np.asarray(result.final_forces)).all()
    assert result.energy_history.size >= 1
    assert float(result.final_energy) <= e0 + 1e-6
