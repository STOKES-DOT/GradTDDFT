import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from td_graddft_tools.geomopt_freq import (
    GeometryOptimizationConfig,
    RHFExcitedStateSurfaceConfig,
    make_rhf_excited_state_surface_from_pyscf_mol,
    run_rhf_excited_state_geometry_optimization,
)


def _pyscf_or_skip():
    try:
        from pyscf import gto, scf, tdscf  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("PySCF is required for RHF excited-state geometry tests.")


def _make_h2_mol():
    from pyscf import gto

    return gto.M(
        atom="H 0 0 0; H 0 0 0.90",
        unit="Angstrom",
        basis="sto-3g",
        spin=0,
        verbose=0,
    )


def test_rhf_excited_state_surface_matches_pyscf_tda_energy_for_h2():
    _pyscf_or_skip()
    from pyscf import scf, tdscf

    mol = _make_h2_mol()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    assert mf.converged

    td = tdscf.TDA(mf)
    td.nstates = 1
    td.kernel()

    surface = make_rhf_excited_state_surface_from_pyscf_mol(
        mol,
        config=RHFExcitedStateSurfaceConfig(
            state_index=0,
            response_method="tda",
            coordinate_unit="angstrom",
        ),
    )
    coords = jnp.asarray(mol.atom_coords()) * 0.529177210903
    energy = surface.energy(coords)

    expected = float(mf.e_tot + td.e[0])
    assert np.isfinite(float(energy))
    assert np.isclose(float(energy), expected, atol=5e-4, rtol=0.0)


def test_rhf_excited_state_geometry_api_is_differentiable_and_runs_optimizer():
    _pyscf_or_skip()
    jax.config.update("jax_enable_x64", True)

    mol = _make_h2_mol()
    surface = make_rhf_excited_state_surface_from_pyscf_mol(
        mol,
        config=RHFExcitedStateSurfaceConfig(
            state_index=0,
            response_method="tda",
            coordinate_unit="angstrom",
        ),
    )
    coords0 = jnp.asarray(
        [
            [0.0, 0.0, -0.55],
            [0.0, 0.0, +0.55],
        ]
    )

    grad = jax.grad(lambda coords: surface.energy(coords))(coords0)
    assert np.isfinite(np.asarray(grad)).all()

    result = run_rhf_excited_state_geometry_optimization(
        mol,
        initial_coordinates=coords0,
        surface_config=RHFExcitedStateSurfaceConfig(
            state_index=0,
            response_method="tda",
            coordinate_unit="angstrom",
        ),
        optimization_config=GeometryOptimizationConfig(
            max_steps=3,
            learning_rate=0.02,
        ),
    )
    assert result.energy_history.size >= 1
    assert np.isfinite(float(result.final_energy))
    assert np.isfinite(np.asarray(result.final_gradient)).all()
