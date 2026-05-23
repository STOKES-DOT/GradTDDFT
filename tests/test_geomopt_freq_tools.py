import jax
import jax.numpy as jnp
import numpy as np

from td_graddft.data.molecule import parse_molecule_spec
from td_graddft.scf import RKSConfig
from td_graddft_tools.geomopt_freq import (
    FrequencyAnalysisConfig,
    GeometryOptimizationConfig,
    GeometryWorkflowConfig,
    coordinates_from_molecule_spec,
    make_excited_state_surface,
    make_ground_state_surface,
    make_rks_ground_state_surface_from_molecule_spec,
    run_frequency_analysis,
    run_geometry_optimization,
    run_geometry_workflow,
)


def _pyscf_or_skip():
    try:
        import pyscf  # noqa: F401
    except ModuleNotFoundError:
        import pytest

        pytest.skip("PySCF is required for libcint-backed geometry optimization tests.")


def test_ground_surface_geometry_optimization_converges_for_quadratic():
    eq = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.2],
        ]
    )
    initial = eq + jnp.array(
        [
            [0.25, -0.10, 0.05],
            [-0.10, 0.12, -0.20],
        ]
    )

    def ground_energy(coords):
        return 0.5 * 0.8 * jnp.sum((coords - eq) ** 2)

    surface = make_ground_state_surface(ground_energy)
    result = run_geometry_optimization(
        surface,
        initial,
        GeometryOptimizationConfig(
            max_steps=500,
            learning_rate=0.06,
            convergence_grad_norm=1e-8,
            convergence_step_norm=1e-8,
        ),
    )

    assert result.steps > 0
    assert result.energy_history.size >= 1
    assert np.isfinite(result.final_energy)
    assert np.allclose(
        np.asarray(result.optimized_coordinates),
        np.asarray(eq),
        atol=4e-3,
        rtol=0.0,
    )


def test_excited_surface_and_frequency_analysis_share_same_pipeline():
    eq = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    initial = eq + jnp.array(
        [
            [0.15, 0.00, 0.00],
            [0.00, -0.10, 0.10],
        ]
    )
    masses_amu = jnp.array([1.0, 16.0])

    def ground_energy(coords):
        return 0.5 * 0.4 * jnp.sum((coords - eq) ** 2)

    def excitation_energies(coords):
        delta = jnp.sum((coords - eq) ** 2)
        return jnp.array([0.2 + 0.3 * delta, 0.4 + 0.2 * delta])

    excited_surface = make_excited_state_surface(
        ground_energy_fn=ground_energy,
        excitation_energy_fn=excitation_energies,
        state_index=0,
    )
    workflow = run_geometry_workflow(
        excited_surface,
        initial,
        masses_amu,
        GeometryWorkflowConfig(
            optimization=GeometryOptimizationConfig(
                max_steps=500,
                learning_rate=0.05,
            ),
            frequencies=FrequencyAnalysisConfig(remove_trans_rot=False),
        ),
    )

    assert workflow.optimization.energy_history.size > 0
    assert np.isfinite(workflow.optimization.final_energy)
    freqs = np.asarray(workflow.frequencies.frequencies_cm1)
    assert freqs.shape == (6,)
    assert np.all(np.isfinite(freqs))
    assert np.all(freqs > 0.0)

    # Direct frequency API should be consistent with workflow output.
    freq2 = run_frequency_analysis(
        excited_surface,
        workflow.optimization.optimized_coordinates,
        masses_amu,
        FrequencyAnalysisConfig(remove_trans_rot=False),
    )
    assert np.allclose(
        np.asarray(freq2.frequencies_cm1),
        np.asarray(workflow.frequencies.frequencies_cm1),
        atol=1e-8,
        rtol=1e-8,
    )


def test_rks_libcint_ground_surface_from_molecule_spec_is_differentiable():
    _pyscf_or_skip()
    spec = parse_molecule_spec(
        [
            ("H", (0.0, 0.0, -0.7)),
            ("H", (0.0, 0.0, 0.7)),
        ],
        unit="Bohr",
    )
    coords = coordinates_from_molecule_spec(spec, unit="bohr")
    surface = make_rks_ground_state_surface_from_molecule_spec(
        spec,
        basis="sto-3g",
        xc_spec="hf",
        coordinate_unit="bohr",
        max_l=1,
        integral_backend="cpu",
        grid_ao_backend="jax",
        rks_config=RKSConfig(
            xc_spec="hf",
            max_cycle=16,
            conv_tol=1e-11,
            conv_tol_density=1e-9,
            damping=0.0,
            jk_backend="full",
        ),
    )

    energy, grad = jax.value_and_grad(surface.energy)(coords)
    result = run_geometry_optimization(
        surface,
        coords,
        GeometryOptimizationConfig(max_steps=1, learning_rate=1e-3),
    )

    assert surface.state_kind == "ground"
    assert np.isfinite(float(energy))
    assert np.isfinite(np.asarray(grad)).all()
    assert np.allclose(np.asarray(grad[0]), -np.asarray(grad[1]), atol=1e-8, rtol=1e-8)
    assert result.steps == 1
    assert np.isfinite(result.final_energy)
