import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from td_graddft.geomopt import GeometryOptimizationConfig, make_rks_ground_state_energy_fn, run_geometry_optimization
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


def test_explicit_rks_geometry_differentiation_is_removed():
    with pytest.raises(NotImplementedError, match="Explicit differentiable SCF geometry optimization has been removed"):
        _make_energy_fn()


def test_generic_geometry_optimizer_still_runs_with_plain_energy_fn():
    jax.config.update("jax_enable_x64", True)

    def energy_fn(coords):
        return jnp.sum(jnp.asarray(coords, dtype=jnp.float64) ** 2)

    result = run_geometry_optimization(
        energy_fn,
        _h2_coords(1.80),
        config=GeometryOptimizationConfig(
            max_steps=4,
            learning_rate=0.05,
            grad_clip_norm=2.0,
        ),
    )
    assert np.isfinite(float(result.final_energy))
    assert np.isfinite(np.asarray(result.final_forces)).all()
    assert result.energy_history.size >= 1
    assert float(result.final_energy) >= 0.0
