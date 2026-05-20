from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from types import SimpleNamespace

import td_graddft.training.targets as training_targets
import td_graddft.training.trainer as training_trainer
import td_graddft.scf.differentiable as scf_differentiable
import td_graddft.scf.rks as scf_rks
from td_graddft.xc_backend.jax_libxc import b3lyp_component_basis
from td_graddft.neural_xc import make_neural_xc_functional
from td_graddft.nn_rsh.functional import BoundTrainableRSHFunctional
from td_graddft.nn_rsh.schema import RSHFunctionalTemplate, ResolvedRSHParameters
from pyscf_reference import restricted_reference_from_pyscf
from td_graddft.scf import DifferentiableSCF, DifferentiableSCFConfig
from td_graddft.scf.molecules import QuadratureGrid, UnrestrictedMolecule
from td_graddft.scf.differentiable import (
    _replace_molecule,
    _restricted_hfx_features_from_nu,
    _restricted_iteration_molecule,
)
from td_graddft.training import (
    GroundStateCoreDatum,
    GroundStateDatum,
    GroundStateTrainingConfig,
    ground_state_mse_loss,
    make_ground_state_eval,
    make_ground_state_loss_and_grad,
    make_ground_state_predictor,
    make_self_consistent_runtime_forward_provider,
    make_runtime_forward_implicit_loss_and_grad,
    predict_ground_state_molecule,
    predict_ground_state_total_energy,
)


def _pyscf_or_skip():
    try:
        from pyscf import dft, gto  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("PySCF is required for differentiable SCF tests.")


def test_replace_molecule_does_not_mutate_namespace_inputs():
    molecule = SimpleNamespace(
        rdm1=jnp.eye(2),
        mo_coeff=jnp.eye(2),
    )

    updated = _replace_molecule(molecule, rdm1=2.0 * jnp.eye(2))

    assert updated is not molecule
    assert np.allclose(np.asarray(molecule.rdm1), np.eye(2))
    assert np.allclose(np.asarray(updated.rdm1), 2.0 * np.eye(2))
    assert updated.mo_coeff is molecule.mo_coeff


def _make_h2_reference(*, half_distance_angstrom: float = 0.35):
    from pyscf import dft, gto

    mol = gto.Mole()
    mol.atom = f"""
    H 0.0 0.0 {-half_distance_angstrom}
    H 0.0 0.0  {half_distance_angstrom}
    """
    mol.unit = "Angstrom"
    mol.basis = "sto-3g"
    mol.spin = 0
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = "b3lyp"
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF SCF did not converge in H2 setup.")
    return restricted_reference_from_pyscf(
        mf,
        compute_local_hfx_features=True,
        compute_local_hfx_aux=True,
    )


def _make_functional_and_params(molecule):
    functional = make_neural_xc_functional(
        semilocal_xc=b3lyp_component_basis(),
        hidden_dims=(16, 16),
        name="test_differentiable_scf",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(0), molecule)
    return functional, params


def _toy_grid():
    ao = jnp.asarray(
        [
            [1.0, 0.2],
            [0.8, -0.1],
            [0.4, 0.9],
        ],
        dtype=jnp.float32,
    )
    ao_deriv1 = jnp.asarray(
        [
            ao,
            [
                [0.10, 0.00],
                [0.00, 0.20],
                [-0.10, 0.05],
            ],
            [
                [0.00, 0.10],
                [0.20, 0.00],
                [0.05, -0.10],
            ],
            [
                [-0.05, 0.00],
                [0.00, -0.05],
                [0.10, 0.10],
            ],
        ],
        dtype=jnp.float32,
    )
    weights = jnp.asarray([0.5, 0.7, 0.6], dtype=jnp.float32)
    return ao, ao_deriv1, weights


def _make_toy_restricted_reference():
    ao, ao_deriv1, weights = _toy_grid()
    mo_coeff = jnp.eye(2, dtype=jnp.float32)
    mo_occ = jnp.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)
    mo_energy = jnp.asarray([[-0.8, 0.2], [-0.8, 0.2]], dtype=jnp.float32)
    density_half = jnp.einsum("pi,i,qi->pq", mo_coeff, mo_occ[0], mo_coeff)
    return SimpleNamespace(
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid=QuadratureGrid(weights=weights),
        rep_tensor=jnp.zeros((2, 2, 2, 2), dtype=jnp.float32),
        mo_coeff=jnp.stack([mo_coeff, mo_coeff], axis=0),
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        rdm1=jnp.stack([density_half, density_half], axis=0),
        h1e=jnp.diag(jnp.asarray([-0.8, 0.2], dtype=jnp.float32)),
        nuclear_repulsion=0.0,
        overlap_matrix=jnp.eye(2, dtype=jnp.float32),
        hfx_omega_values=(0.0,),
    )


def _make_toy_unrestricted_reference():
    ao, ao_deriv1, weights = _toy_grid()
    hfx_nu = jnp.zeros((1, ao.shape[0], ao.shape[1], ao.shape[1]), dtype=jnp.float32)
    mo_coeff = jnp.stack([jnp.eye(2, dtype=jnp.float32), jnp.eye(2, dtype=jnp.float32)], axis=0)
    mo_occ = jnp.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.float32)
    mo_energy = jnp.asarray([[-0.8, 0.2], [-0.8, 0.2]], dtype=jnp.float32)
    rdm1 = jax.vmap(
        lambda coeff_spin, occ_spin: jnp.einsum("pi,i,qi->pq", coeff_spin, occ_spin, coeff_spin)
    )(mo_coeff, mo_occ)
    return UnrestrictedMolecule(
        ao=ao,
        grid=QuadratureGrid(weights=weights),
        dipole_integrals=jnp.zeros((3, 2, 2), dtype=jnp.float32),
        rep_tensor=jnp.zeros((2, 2, 2, 2), dtype=jnp.float32),
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        rdm1=rdm1,
        h1e=jnp.diag(jnp.asarray([-0.8, 0.2], dtype=jnp.float32)),
        nuclear_repulsion=0.0,
        overlap_matrix=jnp.eye(2, dtype=jnp.float32),
        ao_deriv1=ao_deriv1,
        nocc_alpha=1,
        nocc_beta=0,
        hfx_omega_values=(0.0,),
        hfx_nu=hfx_nu,
    )


def _make_toy_unrestricted_bound_rsh():
    template = RSHFunctionalTemplate(
        name="toy_unrestricted_hf",
        local_backend="jax_libxc",
        exchange_backend_id="toy",
        correlation_backend_id="toy",
        default_sr_hf_fraction=1.0,
        default_lr_hf_fraction=1.0,
        default_omega=0.3,
    )
    return BoundTrainableRSHFunctional(
        template=template,
        local_xc_spec="hf",
        resolved_params=ResolvedRSHParameters(
            sr_hf_fraction=1.0,
            lr_hf_fraction=1.0,
            omega=0.3,
        ),
        fallback_omega_values=(0.0,),
    )


class _BoundFunctionalWrapper:
    def __init__(self, bound):
        self.bound = bound

    def bind_to_molecule_for_scf(self, _params, _molecule):
        return self.bound

    def bind_to_molecule(self, _params, _molecule):
        return self.bound


def test_restricted_hfx_features_from_nu_recomputes_local_exchange_density():
    ao = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    density = np.array([[2.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    nu = np.zeros((1, 2, 2, 2), dtype=np.float32)
    nu[0, 0, 0, 0] = 1.0
    nu[0, 1, 1, 1] = 1.0

    hfx_local = _restricted_hfx_features_from_nu(
        ao=ao,
        density=density,
        nu_cache=nu,
    )

    expected = np.array(
        [
            [[-0.5], [0.0]],
            [[-0.5], [0.0]],
        ],
        dtype=np.float32,
    )
    assert np.allclose(np.asarray(hfx_local), expected, atol=1e-12)


def test_restricted_iteration_molecule_prefers_cached_hfx_when_aux_is_present():
    molecule = _make_toy_restricted_reference()
    cached_hfx = jnp.asarray(
        [
            [[-0.11], [-0.07], [-0.03]],
            [[-0.11], [-0.07], [-0.03]],
        ],
        dtype=jnp.float32,
    )
    molecule.hfx_local = cached_hfx
    molecule.hfx_nu = jnp.ones(
        (1, molecule.ao.shape[0], molecule.ao.shape[1], molecule.ao.shape[1]),
        dtype=jnp.float32,
    )
    density = 2.0 * jnp.eye(2, dtype=jnp.float32)
    mo_coeff = jnp.eye(2, dtype=jnp.float32)
    mo_occ = jnp.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)
    mo_energy = jnp.asarray([-0.8, 0.2], dtype=jnp.float32)

    molecule_iter = _restricted_iteration_molecule(
        molecule,
        density=density,
        mo_coeff=mo_coeff,
        mo_occ_stacked=mo_occ,
        mo_energy=mo_energy,
        ao=molecule.ao,
        hfx_nu=molecule.hfx_nu,
        hfx_local=molecule.hfx_local,
        stop_gradient_hfx_local=True,
    )

    assert np.allclose(np.asarray(molecule_iter.hfx_local), np.asarray(cached_hfx))


def test_differentiable_scf_fixed_density_returns_same_density():
    _pyscf_or_skip()
    molecule = _make_h2_reference()
    functional, params = _make_functional_and_params(molecule)

    solver = DifferentiableSCF(DifferentiableSCFConfig(mode="fixed_density"))
    out, info = solver.run(molecule, functional, params)

    assert info.mode == "fixed_density"
    assert bool(info.converged)
    assert np.allclose(np.asarray(out.rdm1), np.asarray(molecule.rdm1))
    assert int(info.selected_cycle) == 0
    assert int(info.best_cycle) == 0


def test_self_consistent_training_mode_produces_finite_loss_and_energy():
    _pyscf_or_skip()
    molecule = _make_h2_reference()
    functional, params = _make_functional_and_params(molecule)

    solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            max_cycle=4,
            damping=0.2,
            conv_tol_density=1e-7,
        )
    )
    molecule_sc, info = solver.run_runtime_forward(molecule, functional, params)
    assert info.mode == "self_consistent_runtime_forward"
    assert int(info.cycles) >= 1
    assert np.isfinite(float(info.final_rms_density))
    assert np.isfinite(np.asarray(molecule_sc.rdm1)).all()

    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=np.asarray(molecule.mf_energy),
        density_constraint_weight=1e-3,
    )
    training_config = GroundStateTrainingConfig(
        mode="self_consistent",
        scf_max_cycle=4,
        scf_damping=0.2,
        scf_conv_tol_density=1e-7,
    )
    loss, metrics = ground_state_mse_loss(
        params,
        functional,
        datum,
        training_config=training_config,
    )
    energy = predict_ground_state_total_energy(
        params,
        functional,
        molecule,
        training_config=training_config,
    )
    assert np.isfinite(float(loss))
    assert metrics["predicted_total_energies"].shape == (1,)
    assert metrics["scf_converged"].shape == (1,)
    assert metrics["scf_cycles"].shape == (1,)
    assert metrics["scf_converged_fraction"].shape == (1,)
    assert metrics["scf_cycles_mean"].shape == (1,)
    assert metrics["scf_selected_rms_max"].shape == (1,)
    assert 0.0 <= float(metrics["scf_converged_fraction"][0]) <= 1.0
    assert float(metrics["scf_cycles_mean"][0]) == 0.0
    assert np.isfinite(float(metrics["scf_selected_rms_max"][0]))
    assert np.isfinite(float(energy))


def test_self_consistent_solver_preserves_fractional_frontier_occupations():
    _pyscf_or_skip()
    molecule = _make_h2_reference()
    functional, params = _make_functional_and_params(molecule)

    mo_occ_frac = jnp.asarray(molecule.mo_occ).at[:, 0].add(-0.05).at[:, 1].add(0.05)
    mo_coeff = jnp.asarray(molecule.mo_coeff)
    rdm1_frac = jax.vmap(
        lambda coeff_spin, occ_spin: jnp.einsum("pi,i,qi->pq", coeff_spin, occ_spin, coeff_spin)
    )(mo_coeff, mo_occ_frac)
    molecule_frac = replace(
        molecule,
        mo_occ=mo_occ_frac,
        rdm1=rdm1_frac,
        scf_initial_density=rdm1_frac.sum(axis=0),
    )

    solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            gradient_mode="impl",
            max_cycle=4,
            damping=0.2,
            conv_tol_density=1e-7,
        )
    )
    molecule_sc, info = solver.run(molecule_frac, functional, params)

    assert info.mode == "self_consistent_implicit_input_state"
    assert np.allclose(np.asarray(molecule_sc.mo_occ), np.asarray(mo_occ_frac), atol=1e-8)
    assert np.isfinite(np.asarray(molecule_sc.rdm1)).all()


def test_unrestricted_self_consistent_solver_runs_for_bound_rsh():
    molecule = _make_toy_unrestricted_reference()
    functional = _BoundFunctionalWrapper(_make_toy_unrestricted_bound_rsh())

    solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            gradient_mode="impl",
            max_cycle=6,
            damping=0.2,
            conv_tol_density=1e-8,
        )
    )
    molecule_sc, info = solver.run(molecule, functional, {})

    assert info.mode == "self_consistent_implicit_input_state"
    assert int(info.cycles) == 0
    assert np.isfinite(float(info.final_rms_density))
    assert np.isfinite(np.asarray(molecule_sc.rdm1)).all()
    assert np.allclose(np.asarray(molecule_sc.mo_occ), np.asarray(molecule.mo_occ), atol=1e-8)


def test_unrestricted_self_consistent_solver_is_differentiable_in_implicit_mode():
    molecule = _make_toy_unrestricted_reference()

    class _ToyUnrestrictedFunctional:
        def bind_to_molecule_for_scf(self, params, _molecule):
            strength = jnp.asarray(params["strength"], dtype=jnp.float32)

            class _Bound:
                exact_exchange_fraction = jnp.asarray(0.0, dtype=jnp.float32)

                def unrestricted_scf_components(self, molecule_in):
                    ngrids = int(molecule_in.ao.shape[0])
                    zeros_grad = jnp.zeros((ngrids, 3), dtype=jnp.float32)
                    zeros_mat = jnp.zeros((molecule_in.ao.shape[1], molecule_in.ao.shape[1]), dtype=jnp.float32)
                    v_alpha = strength * jnp.asarray([1.0, -0.2, 0.4], dtype=jnp.float32)
                    v_beta = -strength * jnp.asarray([0.1, 0.3, -0.5], dtype=jnp.float32)
                    return v_alpha, v_beta, zeros_grad, zeros_grad, "LDA", jnp.asarray(0.0), zeros_mat, zeros_mat

            return _Bound()

    functional = _ToyUnrestrictedFunctional()
    solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            gradient_mode="impl",
            max_cycle=6,
            damping=0.2,
            conv_tol_density=1e-8,
        )
    )

    def _objective(raw_strength):
        params = {"strength": raw_strength}
        out, _ = solver.run(molecule, functional, params)
        return jnp.sum(out.rdm1[0])

    value, grad = jax.value_and_grad(_objective)(jnp.asarray(0.1, dtype=jnp.float32))

    assert jnp.isfinite(value)
    assert jnp.isfinite(grad)


def test_unrestricted_impl_produces_finite_gradient():
    molecule = _make_toy_unrestricted_reference()

    class _ToyUnrestrictedFunctional:
        def bind_to_molecule_for_scf(self, params, _molecule):
            strength = jnp.asarray(params["strength"], dtype=jnp.float32)

            class _Bound:
                exact_exchange_fraction = jnp.asarray(0.0, dtype=jnp.float32)

                def unrestricted_scf_components(self, molecule_in):
                    ngrids = int(molecule_in.ao.shape[0])
                    zeros_grad = jnp.zeros((ngrids, 3), dtype=jnp.float32)
                    zeros_mat = jnp.zeros(
                        (molecule_in.ao.shape[1], molecule_in.ao.shape[1]),
                        dtype=jnp.float32,
                    )
                    v_alpha = strength * jnp.asarray([1.0, -0.2, 0.4], dtype=jnp.float32)
                    v_beta = -strength * jnp.asarray([0.1, 0.3, -0.5], dtype=jnp.float32)
                    return (
                        v_alpha,
                        v_beta,
                        zeros_grad,
                        zeros_grad,
                        "LDA",
                        jnp.asarray(0.0),
                        zeros_mat,
                        zeros_mat,
                    )

            return _Bound()

    functional = _ToyUnrestrictedFunctional()
    solver_implicit = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            gradient_mode="impl",
            max_cycle=6,
            damping=0.2,
            conv_tol_density=1e-8,
            implicit_diff_max_iter=12,
            implicit_diff_regularization=1e-3,
            implicit_diff_tolerance=1e-6,
            implicit_diff_restart=6,
        )
    )
    def _objective_with(solver, raw_strength):
        params = {"strength": raw_strength}
        out, _ = solver.run(molecule, functional, params)
        return jnp.sum(out.rdm1[0])

    raw_strength = jnp.asarray(0.1, dtype=jnp.float32)
    implicit_value, implicit_grad = jax.value_and_grad(
        lambda x: _objective_with(solver_implicit, x)
    )(raw_strength)

    assert jnp.isfinite(implicit_value)
    assert jnp.isfinite(implicit_grad)
    assert jnp.abs(implicit_grad) > 1e-6


def test_unrestricted_impl_delegates_to_generic_fixed_point_wrapper(monkeypatch):
    molecule = _make_toy_unrestricted_reference()
    calls = {"implicit": 0}
    original = scf_differentiable.implicit_fixed_point_solution

    def counted_implicit_fixed_point_solution(*args, **kwargs):
        calls["implicit"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        scf_differentiable,
        "implicit_fixed_point_solution",
        counted_implicit_fixed_point_solution,
    )

    class _ToyUnrestrictedFunctional:
        def bind_to_molecule_for_scf(self, params, _molecule):
            strength = jnp.asarray(params["strength"], dtype=jnp.float32)

            class _Bound:
                exact_exchange_fraction = jnp.asarray(0.0, dtype=jnp.float32)

                def unrestricted_scf_components(self, molecule_in):
                    ngrids = int(molecule_in.ao.shape[0])
                    zeros_grad = jnp.zeros((ngrids, 3), dtype=jnp.float32)
                    zeros_mat = jnp.zeros(
                        (molecule_in.ao.shape[1], molecule_in.ao.shape[1]),
                        dtype=jnp.float32,
                    )
                    v_alpha = strength * jnp.asarray([1.0, -0.2, 0.4], dtype=jnp.float32)
                    v_beta = -strength * jnp.asarray([0.1, 0.3, -0.5], dtype=jnp.float32)
                    return (
                        v_alpha,
                        v_beta,
                        zeros_grad,
                        zeros_grad,
                        "LDA",
                        jnp.asarray(0.0),
                        zeros_mat,
                        zeros_mat,
                    )

            return _Bound()

    solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            gradient_mode="impl",
            max_cycle=6,
            damping=0.2,
            conv_tol_density=1e-8,
            implicit_diff_max_iter=8,
            implicit_diff_regularization=1e-3,
        )
    )

    def _objective(raw_strength):
        params = {"strength": raw_strength}
        out, _ = solver.run(molecule, _ToyUnrestrictedFunctional(), params)
        return jnp.sum(out.rdm1[0])

    _, grad = jax.value_and_grad(_objective)(jnp.asarray(0.1, dtype=jnp.float32))

    assert calls["implicit"] == 1
    assert jnp.isfinite(grad)


def test_impl_can_use_input_state_as_forward_primal():
    molecule = _make_toy_restricted_reference()

    class _ToyRestrictedFunctional:
        def scf_potential_components_and_alpha(self, params, molecule_in):
            strength = jnp.asarray(params["strength"], dtype=jnp.float32)
            v_rho = strength * jnp.asarray([1.0, -0.2, 0.4], dtype=jnp.float32)
            v_grad = jnp.zeros((int(molecule_in.ao.shape[0]), 3), dtype=jnp.float32)
            return v_rho, v_grad, "LDA", jnp.asarray(0.0, dtype=jnp.float32)

    functional = _ToyRestrictedFunctional()
    solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            gradient_mode="impl",
            implicit_forward_mode="input_state",
            implicit_diff_max_iter=12,
            implicit_diff_regularization=1e-3,
            implicit_diff_tolerance=1e-6,
            implicit_diff_restart=6,
        )
    )

    def _objective(raw_strength):
        params = {"strength": raw_strength}
        out, _ = solver.run(molecule, functional, params)
        return jnp.sum(out.rdm1[0])

    value, grad = jax.value_and_grad(_objective)(jnp.asarray(0.1, dtype=jnp.float32))
    out, info = solver.run(molecule, functional, {"strength": jnp.asarray(0.1, dtype=jnp.float32)})

    assert bool(info.converged)
    assert info.mode == "self_consistent_implicit_input_state"
    assert int(info.cycles) == 0
    assert np.allclose(np.asarray(out.rdm1), np.asarray(molecule.rdm1), atol=1e-7)
    assert jnp.isfinite(value)
    assert jnp.isfinite(grad)


def test_restricted_xc_fock_terms_prefers_density_energy_callback():
    molecule = _make_toy_restricted_reference()

    class _EnergyFunctional:
        def scf_xc_energy_for_density(self, params, _molecule, density):
            scale = jnp.asarray(params["scale"], dtype=density.dtype)
            return 0.5 * scale * jnp.sum(density * density)

        def scf_exact_exchange_fraction(self, params, _molecule, density):
            del params, density
            return jnp.asarray(0.25, dtype=jnp.float32)

        def scf_extra_fock_for_density(self, params, _molecule, density):
            del params
            return jnp.eye(density.shape[0], dtype=density.dtype) * 0.3

        def scf_potential_components_and_alpha(self, *_args, **_kwargs):
            raise AssertionError("potential-component path should not be used")

    vxc_matrix, alpha, extra_fock = scf_differentiable._restricted_xc_fock_terms(
        params={"scale": jnp.asarray(1.7, dtype=jnp.float32)},
        functional=_EnergyFunctional(),
        molecule=molecule,
        weights=molecule.grid.weights,
        functional_dtype=jnp.float32,
        vxc_clip=20.0,
    )

    density = jnp.asarray(molecule.rdm1).sum(axis=0)
    assert np.allclose(np.asarray(vxc_matrix), np.asarray(1.7 * density), atol=1e-6)
    assert np.allclose(float(alpha), 0.25)
    assert np.allclose(np.asarray(extra_fock), np.eye(density.shape[0]) * 0.3)


def test_restricted_impl_delegates_to_generic_fixed_point_wrapper(monkeypatch):
    molecule = _make_toy_restricted_reference()
    calls = {"implicit": 0}
    original = scf_differentiable.implicit_fixed_point_solution

    def counted_implicit_fixed_point_solution(*args, **kwargs):
        calls["implicit"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        scf_differentiable,
        "implicit_fixed_point_solution",
        counted_implicit_fixed_point_solution,
    )

    class _ToyRestrictedFunctional:
        def scf_potential_components_and_alpha(self, params, molecule_in):
            strength = jnp.asarray(params["strength"], dtype=jnp.float32)
            v_rho = strength * jnp.asarray([1.0, -0.2, 0.4], dtype=jnp.float32)
            v_grad = jnp.zeros((int(molecule_in.ao.shape[0]), 3), dtype=jnp.float32)
            return v_rho, v_grad, "LDA", jnp.asarray(0.0, dtype=jnp.float32)

    solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            gradient_mode="impl",
            implicit_forward_mode="input_state",
            implicit_response_backend="jax",
            implicit_diff_max_iter=4,
        )
    )

    def objective(raw_strength):
        params = {"strength": raw_strength}
        out, _ = solver.run(molecule, _ToyRestrictedFunctional(), params)
        return jnp.sum(out.rdm1[0])

    _, grad = jax.value_and_grad(objective)(jnp.asarray(0.1, dtype=jnp.float32))

    assert calls["implicit"] == 1
    assert jnp.isfinite(grad)


def test_jax_implicit_response_traces_fixed_point_jk_once(monkeypatch):
    molecule = _make_toy_restricted_reference()
    calls = {"jk": 0}
    original_jk = scf_differentiable._coulomb_exchange_matrices

    def counted_jk(rep_tensor, density):
        calls["jk"] += 1
        return original_jk(rep_tensor, density)

    monkeypatch.setattr(scf_differentiable, "_coulomb_exchange_matrices", counted_jk)

    class _ToyRestrictedFunctional:
        def scf_potential_components_and_alpha(self, params, molecule_in):
            strength = jnp.asarray(params["strength"], dtype=jnp.float32)
            v_rho = strength * jnp.asarray([1.0, -0.2, 0.4], dtype=jnp.float32)
            v_grad = jnp.zeros((int(molecule_in.ao.shape[0]), 3), dtype=jnp.float32)
            return v_rho, v_grad, "LDA", jnp.asarray(0.0, dtype=jnp.float32)

    solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            gradient_mode="impl",
            implicit_forward_mode="input_state",
            implicit_response_backend="jax",
            implicit_diff_max_iter=4,
        )
    )

    def objective(raw_strength):
        params = {"strength": raw_strength}
        out, _ = solver.run(molecule, _ToyRestrictedFunctional(), params)
        return jnp.sum(out.rdm1[0])

    _, grad = jax.value_and_grad(objective)(jnp.asarray(0.1, dtype=jnp.float32))

    assert jnp.isfinite(grad)
    assert calls["jk"] <= 3


def test_gpu4pyscf_implicit_response_uses_transpose_factory(monkeypatch):
    molecule = _make_toy_restricted_reference()
    captured = {}
    original = scf_differentiable.implicit_fixed_point_solution

    def counted_implicit_fixed_point_solution(*args, **kwargs):
        captured["factory"] = kwargs.get("apply_fixed_point_transpose_factory")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        scf_differentiable,
        "implicit_fixed_point_solution",
        counted_implicit_fixed_point_solution,
    )
    monkeypatch.setattr(
        scf_differentiable,
        "_gpu4pyscf_jk_response_options",
        lambda _molecule: object(),
    )
    monkeypatch.setattr(
        scf_differentiable,
        "_gpu4pyscf_jk_response_matrices",
        lambda _molecule, density, *, dtype, with_k=True: (
            jnp.zeros_like(jnp.asarray(density, dtype=dtype)),
            jnp.zeros_like(jnp.asarray(density, dtype=dtype)),
        ),
    )

    class _ToyRestrictedFunctional:
        def scf_potential_components_and_alpha(self, params, molecule_in):
            strength = jnp.asarray(params["strength"], dtype=jnp.float32)
            v_rho = strength * jnp.asarray([1.0, -0.2, 0.4], dtype=jnp.float32)
            v_grad = jnp.zeros((int(molecule_in.ao.shape[0]), 3), dtype=jnp.float32)
            return v_rho, v_grad, "LDA", jnp.asarray(0.0, dtype=jnp.float32)

    solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            gradient_mode="impl",
            implicit_forward_mode="input_state",
            implicit_response_backend="gpu4pyscf_jk",
            implicit_diff_max_iter=4,
        )
    )

    def objective(raw_strength):
        params = {"strength": raw_strength}
        out, _ = solver.run(molecule, _ToyRestrictedFunctional(), params)
        return jnp.sum(out.rdm1[0])

    _, grad = jax.value_and_grad(objective)(jnp.asarray(0.1, dtype=jnp.float32))

    assert jnp.isfinite(grad)
    assert callable(captured["factory"])


def test_training_config_passes_implicit_forward_mode_to_scf():
    import td_graddft.training.targets as targets_mod

    cfg = GroundStateTrainingConfig(
        mode="self_consistent",
        scf_gradient_mode="impl",
        scf_implicit_forward_mode="input_state",
    )

    scf_solver = targets_mod._make_differentiable_scf(cfg)

    assert scf_solver.config.gradient_mode == "impl"
    assert scf_solver.config.implicit_forward_mode == "input_state"


def test_runtime_forward_implicit_loss_runs_provider_before_grad():
    molecule = _make_toy_restricted_reference()

    class _ToyRestrictedFunctional:
        def scf_potential_components_and_alpha(self, params, molecule_in):
            strength = jnp.asarray(params["strength"], dtype=jnp.float32)
            v_rho = strength * jnp.asarray([1.0, -0.2, 0.4], dtype=jnp.float32)
            v_grad = jnp.zeros((int(molecule_in.ao.shape[0]), 3), dtype=jnp.float32)
            return v_rho, v_grad, "LDA", jnp.asarray(0.0, dtype=jnp.float32)

        def energy(self, params, density, weights):
            strength = jnp.asarray(params["strength"], dtype=jnp.float32)
            return strength * jnp.sum(jnp.asarray(density) * jnp.asarray(weights))

    calls = []

    def _runtime_forward(params, functional, molecule_in):
        del params, functional
        calls.append("forward")
        return _replace_molecule(
            molecule_in,
            rdm1=jnp.asarray(molecule_in.rdm1),
            scf_initial_density=jnp.asarray(molecule_in.rdm1).sum(axis=0),
        )

    loss_and_grad = make_runtime_forward_implicit_loss_and_grad(
        _ToyRestrictedFunctional(),
        _runtime_forward,
        training_config=GroundStateTrainingConfig(
            energy_mse_weight=1.0,
            energy_mae_weight=0.0,
            scf_implicit_diff_max_iter=8,
            scf_implicit_diff_regularization=1e-3,
            scf_implicit_diff_tolerance=1e-6,
            scf_implicit_diff_restart=4,
        ),
    )
    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.asarray(0.0, dtype=jnp.float32),
    )

    loss, metrics, grads = loss_and_grad(
        {"strength": jnp.asarray(0.1, dtype=jnp.float32)},
        datum,
    )

    assert calls == ["forward"]
    assert np.isfinite(float(loss))
    assert metrics["scf_cycles"].shape == (1,)
    assert int(metrics["scf_cycles"][0]) == 0
    assert jnp.isfinite(grads["strength"])


def test_self_consistent_runtime_forward_provider_feeds_implicit_loss():
    molecule = _make_toy_restricted_reference()

    class _ToyRestrictedFunctional:
        def scf_potential_components_and_alpha(self, params, molecule_in):
            strength = jnp.asarray(params["strength"], dtype=jnp.float32)
            v_rho = strength * jnp.asarray([1.0, -0.2, 0.4], dtype=jnp.float32)
            v_grad = jnp.zeros((int(molecule_in.ao.shape[0]), 3), dtype=jnp.float32)
            return v_rho, v_grad, "LDA", jnp.asarray(0.0, dtype=jnp.float32)

        def energy(self, params, density, weights):
            strength = jnp.asarray(params["strength"], dtype=jnp.float32)
            return strength * jnp.sum(jnp.asarray(density) * jnp.asarray(weights))

    config = GroundStateTrainingConfig(
        energy_mse_weight=1.0,
        energy_mae_weight=0.0,
        scf_max_cycle=3,
        scf_damping=0.2,
        scf_implicit_diff_max_iter=8,
        scf_implicit_diff_regularization=1e-3,
        scf_implicit_diff_tolerance=1e-6,
        scf_implicit_diff_restart=4,
    )
    functional = _ToyRestrictedFunctional()
    provider = make_self_consistent_runtime_forward_provider(config)
    params = {"strength": jnp.asarray(0.1, dtype=jnp.float32)}

    forward_molecule, forward_info = provider(params, functional, molecule)
    assert forward_info.mode == "self_consistent_runtime_forward"
    assert forward_molecule.rdm1.shape == molecule.rdm1.shape

    loss_and_grad = make_runtime_forward_implicit_loss_and_grad(
        functional,
        provider,
        training_config=config,
    )
    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.asarray(0.0, dtype=jnp.float32),
    )

    loss, metrics, grads = loss_and_grad(params, datum)

    assert np.isfinite(float(loss))
    assert int(metrics["scf_cycles"][0]) == 0
    assert jnp.isfinite(grads["strength"])


def test_ground_state_helpers_auto_use_runtime_forward_for_implicit_self_consistent(
    monkeypatch,
):
    molecule = _make_toy_restricted_reference()

    class _ToyRestrictedFunctional:
        def scf_potential_components_and_alpha(self, params, molecule_in):
            strength = jnp.asarray(params["strength"], dtype=jnp.float32)
            v_rho = strength * jnp.asarray([1.0, -0.2, 0.4], dtype=jnp.float32)
            v_grad = jnp.zeros((int(molecule_in.ao.shape[0]), 3), dtype=jnp.float32)
            return v_rho, v_grad, "LDA", jnp.asarray(0.0, dtype=jnp.float32)

        def energy(self, params, density, weights):
            strength = jnp.asarray(params["strength"], dtype=jnp.float32)
            return strength * jnp.sum(jnp.asarray(density) * jnp.asarray(weights))

    calls: list[str] = []

    def _provider_factory(_cfg):
        def _provider(params, functional, molecule_in):
            del params, functional
            calls.append("forward")
            return _replace_molecule(
                molecule_in,
                rdm1=jnp.asarray(molecule_in.rdm1),
                scf_initial_density=jnp.asarray(molecule_in.rdm1).sum(axis=0),
            )

        return _provider

    monkeypatch.setattr(
        training_trainer,
        "make_self_consistent_runtime_forward_provider",
        _provider_factory,
    )

    cfg = GroundStateTrainingConfig(
        mode="self_consistent",
        scf_gradient_mode="impl",
        energy_mse_weight=1.0,
        energy_mae_weight=0.0,
        scf_implicit_diff_max_iter=8,
        scf_implicit_diff_regularization=1e-3,
        scf_implicit_diff_tolerance=1e-6,
        scf_implicit_diff_restart=4,
    )
    functional = _ToyRestrictedFunctional()
    params = {"strength": jnp.asarray(0.1, dtype=jnp.float32)}
    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.asarray(0.0, dtype=jnp.float32),
    )

    evaluate = make_ground_state_eval(functional, training_config=cfg)
    loss_eval, metrics_eval = evaluate(params, datum)
    assert calls == ["forward"]
    assert np.isfinite(float(loss_eval))
    assert int(metrics_eval["scf_cycles"][0]) == 0

    calls.clear()
    loss_and_grad = make_ground_state_loss_and_grad(functional, training_config=cfg)
    loss_train, metrics_train, grads = loss_and_grad(params, datum)
    assert calls == ["forward"]
    assert np.isfinite(float(loss_train))
    assert int(metrics_train["scf_cycles"][0]) == 0
    assert jnp.isfinite(grads["strength"])

    calls.clear()
    predicted_molecule = predict_ground_state_molecule(
        params,
        functional,
        molecule,
        training_config=cfg,
    )
    assert calls == ["forward"]
    assert predicted_molecule.rdm1.shape == molecule.rdm1.shape

    calls.clear()
    predictor = make_ground_state_predictor(functional, training_config=cfg)
    predicted_energy, predictor_molecule = predictor(params, molecule)
    assert calls == ["forward"]
    assert np.isfinite(float(predicted_energy))
    assert predictor_molecule.rdm1.shape == molecule.rdm1.shape


def test_ground_state_loss_reuses_value_and_grad_transform(monkeypatch):
    molecule = _make_toy_restricted_reference()

    class _ToyRestrictedFunctional:
        def scf_potential_components_and_alpha(self, params, molecule_in):
            strength = jnp.asarray(params["strength"], dtype=jnp.float32)
            v_rho = strength * jnp.asarray([1.0, -0.2, 0.4], dtype=jnp.float32)
            v_grad = jnp.zeros((int(molecule_in.ao.shape[0]), 3), dtype=jnp.float32)
            return v_rho, v_grad, "LDA", jnp.asarray(0.0, dtype=jnp.float32)

        def energy(self, params, density, weights):
            strength = jnp.asarray(params["strength"], dtype=jnp.float32)
            return strength * jnp.sum(jnp.asarray(density) * jnp.asarray(weights))

    transform_calls = []
    real_value_and_grad = training_trainer.jax.value_and_grad

    def _counting_value_and_grad(*args, **kwargs):
        transform_calls.append((args, kwargs))
        return real_value_and_grad(*args, **kwargs)

    monkeypatch.setattr(training_trainer.jax, "value_and_grad", _counting_value_and_grad)

    loss_and_grad = make_ground_state_loss_and_grad(
        _ToyRestrictedFunctional(),
        training_config=GroundStateTrainingConfig(
            mode="fixed_density",
            energy_mse_weight=1.0,
            energy_mae_weight=0.0,
        ),
    )
    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.asarray(0.0, dtype=jnp.float32),
    )
    params = {"strength": jnp.asarray(0.1, dtype=jnp.float32)}

    loss_and_grad(params, datum)
    loss_and_grad(params, datum)

    assert len(transform_calls) == 1


def test_impl_self_consistent_loss_produces_finite_gradient():
    _pyscf_or_skip()
    molecule = _make_h2_reference()
    functional, params = _make_functional_and_params(molecule)

    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=np.asarray(molecule.mf_energy),
        density_constraint_weight=1e-3,
    )
    training_config = GroundStateTrainingConfig(
        mode="self_consistent",
        scf_gradient_mode="impl",
        scf_max_cycle=4,
        scf_damping=0.2,
        scf_conv_tol_density=1e-7,
        scf_implicit_diff_max_iter=8,
        scf_implicit_diff_solver="normal_cg",
        scf_implicit_diff_tolerance=1e-5,
        scf_implicit_diff_regularization=1e-3,
        scf_implicit_diff_restart=4,
    )

    loss, metrics = ground_state_mse_loss(
        params,
        functional,
        datum,
        training_config=training_config,
    )
    grads = jax.grad(
        lambda p: ground_state_mse_loss(
            p,
            functional,
            datum,
            training_config=training_config,
        )[0]
    )(params)

    assert np.isfinite(float(loss))
    assert np.isfinite(float(metrics["scf_selected_rms_max"][0]))
    assert all(
        np.isfinite(np.asarray(leaf)).all()
        for leaf in jax.tree_util.tree_leaves(grads)
    )


def test_density_matching_penalty_is_jittable_in_fixed_density_training():
    _pyscf_or_skip()
    molecule = _make_h2_reference()
    functional, params = _make_functional_and_params(molecule)

    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=np.asarray(molecule.mf_energy),
        density_constraint_weight=1e-3,
    )
    training_config = GroundStateTrainingConfig(
        mode="fixed_density",
        scf_max_cycle=4,
        scf_damping=0.2,
        scf_conv_tol_density=1e-7,
    )

    compiled_loss = jax.jit(
        lambda p: ground_state_mse_loss(
            p,
            functional,
            datum,
            training_config=training_config,
        )
    )
    loss, metrics = compiled_loss(params)
    grads = jax.grad(
        lambda p: ground_state_mse_loss(
            p,
            functional,
            datum,
            training_config=training_config,
        )[0]
    )(params)

    assert np.isfinite(float(loss))
    assert metrics["density_penalty"].shape == (1,)
    assert np.isfinite(float(metrics["density_penalty"][0]))
    if bool(jax.config.read("jax_enable_x64")):
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(grads))


def test_iterate_selection_tracks_best_and_first_converged_cycles():
    _pyscf_or_skip()
    molecule = _make_h2_reference()
    functional, params = _make_functional_and_params(molecule)

    best_solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            max_cycle=8,
            damping=0.2,
            conv_tol_density=1e-12,
            iterate_selection="best_rms",
        )
    )
    _, best_info = best_solver.run_runtime_forward(molecule, functional, params)
    best_history = np.asarray(best_info.rms_density_history)
    assert int(best_info.best_cycle) == int(best_history.argmin() + 1)
    assert np.isclose(float(best_info.best_rms_density), float(best_history.min()))
    assert int(best_info.selected_cycle) == int(best_info.best_cycle)
    assert np.isclose(float(best_info.selected_rms_density), float(best_info.best_rms_density))

    first_solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            max_cycle=8,
            damping=0.2,
            conv_tol_density=1.0,
            iterate_selection="first_converged",
        )
    )
    _, first_info = first_solver.run_runtime_forward(molecule, functional, params)
    assert bool(first_info.converged)
    assert int(first_info.cycles) == 1
    assert int(first_info.selected_cycle) == 1
    assert np.isclose(
        float(first_info.selected_rms_density),
        float(np.asarray(first_info.rms_density_history)[0]),
    )


def test_self_consistent_loss_can_hard_gate_unconverged_scf(monkeypatch):
    _pyscf_or_skip()
    molecule = _make_h2_reference()
    functional, params = _make_functional_and_params(molecule)

    fake_info = SimpleNamespace(
        mode="self_consistent",
        converged=jnp.asarray(False),
        cycles=jnp.asarray(4),
        selected_cycle=jnp.asarray(4),
        best_cycle=jnp.asarray(3),
        final_rms_density=jnp.asarray(1e-2),
        selected_rms_density=jnp.asarray(1e-3),
        best_rms_density=jnp.asarray(1e-3),
    )

    def _fake_run(self, molecule_in, functional_in, params_in):
        del self, functional_in, params_in
        return molecule_in, fake_info

    monkeypatch.setattr(DifferentiableSCF, "run", _fake_run)

    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=np.asarray(molecule.mf_energy + 1.0),
    )
    loose_cfg = GroundStateTrainingConfig(
        mode="self_consistent",
        scf_max_cycle=4,
        scf_require_convergence=False,
    )
    strict_cfg = GroundStateTrainingConfig(
        mode="self_consistent",
        scf_max_cycle=4,
        scf_require_convergence=True,
    )

    loose_loss, loose_metrics = ground_state_mse_loss(
        params,
        functional,
        datum,
        training_config=loose_cfg,
    )
    strict_loss, strict_metrics = ground_state_mse_loss(
        params,
        functional,
        datum,
        training_config=strict_cfg,
    )

    assert float(loose_loss) > 1e-6
    assert np.isclose(float(strict_loss), 0.0, atol=1e-12)
    assert np.isclose(float(loose_metrics["scf_converged_fraction"][0]), 0.0, atol=1e-12)
    assert np.isclose(float(strict_metrics["scf_converged_fraction"][0]), 0.0, atol=1e-12)


def test_self_consistent_loss_can_stop_gradient_on_unconverged_scf(monkeypatch):
    _pyscf_or_skip()
    molecule = _make_h2_reference()
    functional, params = _make_functional_and_params(molecule)

    fake_info = SimpleNamespace(
        mode="self_consistent",
        converged=jnp.asarray(False),
        cycles=jnp.asarray(4),
        selected_cycle=jnp.asarray(4),
        best_cycle=jnp.asarray(3),
        final_rms_density=jnp.asarray(1e-2),
        selected_rms_density=jnp.asarray(1e-3),
        best_rms_density=jnp.asarray(1e-3),
    )

    def _fake_run(self, molecule_in, functional_in, params_in):
        del self, functional_in, params_in
        return molecule_in, fake_info

    monkeypatch.setattr(DifferentiableSCF, "run", _fake_run)

    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=np.asarray(molecule.mf_energy + 1.0),
    )
    loose_cfg = GroundStateTrainingConfig(
        mode="self_consistent",
        scf_max_cycle=4,
        scf_stop_gradient_on_unconverged=False,
    )
    guarded_cfg = GroundStateTrainingConfig(
        mode="self_consistent",
        scf_max_cycle=4,
        scf_stop_gradient_on_unconverged=True,
    )

    def _loss_grad_norm(cfg):
        grad = jax.grad(
            lambda p: ground_state_mse_loss(
                p,
                functional,
                datum,
                training_config=cfg,
            )[0]
        )(params)
        leaves = jax.tree_util.tree_leaves(grad)
        return float(sum(float(jnp.sum(jnp.abs(jnp.asarray(leaf)))) for leaf in leaves))

    loose_norm = _loss_grad_norm(loose_cfg)
    guarded_loss, guarded_metrics = ground_state_mse_loss(
        params,
        functional,
        datum,
        training_config=guarded_cfg,
    )
    guarded_norm = _loss_grad_norm(guarded_cfg)

    assert loose_norm > 1e-10
    assert np.isclose(float(guarded_loss), float(ground_state_mse_loss(params, functional, datum, training_config=loose_cfg)[0]))
    assert guarded_norm < 1e-12
    assert np.isclose(float(guarded_metrics["scf_stop_gradient_fraction"][0]), 1.0, atol=1e-12)


def test_self_consistent_loss_can_stop_gradient_on_large_selected_rms(monkeypatch):
    _pyscf_or_skip()
    molecule = _make_h2_reference()
    functional, params = _make_functional_and_params(molecule)

    fake_info = SimpleNamespace(
        mode="self_consistent",
        converged=jnp.asarray(True),
        cycles=jnp.asarray(4),
        selected_cycle=jnp.asarray(4),
        best_cycle=jnp.asarray(3),
        final_rms_density=jnp.asarray(5e-3),
        selected_rms_density=jnp.asarray(5e-3),
        best_rms_density=jnp.asarray(1e-4),
    )

    def _fake_run(self, molecule_in, functional_in, params_in):
        del self, functional_in, params_in
        return molecule_in, fake_info

    monkeypatch.setattr(DifferentiableSCF, "run", _fake_run)

    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=np.asarray(molecule.mf_energy + 1.0),
    )
    cfg = GroundStateTrainingConfig(
        mode="self_consistent",
        scf_max_cycle=4,
        scf_stop_gradient_rms_threshold=1e-3,
    )

    grad = jax.grad(
        lambda p: ground_state_mse_loss(
            p,
            functional,
            datum,
            training_config=cfg,
        )[0]
    )(params)
    grad_norm = float(
        sum(
            float(jnp.sum(jnp.abs(jnp.asarray(leaf))))
            for leaf in jax.tree_util.tree_leaves(grad)
        )
    )
    _, metrics = ground_state_mse_loss(
        params,
        functional,
        datum,
        training_config=cfg,
    )

    assert grad_norm < 1e-12
    assert np.isclose(float(metrics["scf_stop_gradient_fraction"][0]), 1.0, atol=1e-12)


def test_require_converged_iterates_avoids_best_rms_fallback_when_unconverged():
    _pyscf_or_skip()
    molecule = _make_h2_reference(half_distance_angstrom=2.5)
    functional, params = _make_functional_and_params(molecule)

    solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            max_cycle=1,
            damping=0.25,
            conv_tol_density=1e-8,
            iterate_selection="best_rms",
            require_converged_iterates=True,
        )
    )
    _, info = solver.run_runtime_forward(molecule, functional, params)

    assert not bool(info.converged)
    assert int(info.selected_cycle) == 1
    assert np.isclose(
        float(info.selected_rms_density),
        float(info.final_rms_density),
    )


def test_ground_state_datum_preserves_scf_initial_density_and_stores_target_density():
    _pyscf_or_skip()
    molecule = _make_h2_reference()
    original_rdm1 = np.asarray(molecule.rdm1)
    target_density_matrix = np.asarray(original_rdm1.sum(axis=0)) * 0.9
    datum = GroundStateDatum.from_parts(
        molecule,
        core=GroundStateCoreDatum(
            target_total_energy=np.asarray(molecule.mf_energy),
            target_density_matrix=target_density_matrix,
            density_constraint_weight=1.0,
        ),
    )

    assert np.allclose(np.asarray(datum.molecule.rdm1), original_rdm1)
    assert np.allclose(np.asarray(datum.target_density_matrix), target_density_matrix)


def test_self_consistent_solver_uses_cached_initial_density_when_available():
    _pyscf_or_skip()
    molecule = _make_h2_reference()
    functional, params = _make_functional_and_params(molecule)
    cached_total_density = np.asarray(molecule.rdm1).sum(axis=0) * 0.85
    molecule_cached = replace(molecule, scf_initial_density=jnp.asarray(cached_total_density))

    solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            max_cycle=1,
            damping=1.0,
            conv_tol_density=1e-8,
            iterate_selection="final",
        )
    )
    out, _ = solver.run_runtime_forward(molecule_cached, functional, params)

    assert np.allclose(
        np.asarray(out.rdm1).sum(axis=0),
        cached_total_density,
        atol=1e-8,
    )


def test_batched_self_consistent_ground_state_loss_matches_loop_path(monkeypatch):
    _pyscf_or_skip()
    molecule_a = _make_h2_reference(half_distance_angstrom=0.35)
    molecule_b = _make_h2_reference(half_distance_angstrom=0.70)
    functional, params = _make_functional_and_params(molecule_a)

    dataset = [
        GroundStateDatum(
            molecule=molecule_a,
            target_total_energy=np.asarray(molecule_a.mf_energy),
        ),
        GroundStateDatum(
            molecule=molecule_b,
            target_total_energy=np.asarray(molecule_b.mf_energy),
        ),
    ]
    training_config = GroundStateTrainingConfig(
        mode="self_consistent",
        scf_gradient_mode="impl",
        scf_max_cycle=4,
        scf_damping=0.2,
        scf_conv_tol_density=1e-7,
    )

    loss_batched, metrics_batched = training_targets.ground_state_mse_loss(
        params,
        functional,
        dataset,
        training_config=training_config,
    )

    monkeypatch.setattr(
        training_targets,
        "_can_use_batched_self_consistent_ground_state_path",
        lambda dataset_in, cfg_in, predictor_in: False,
    )
    loss_loop, metrics_loop = training_targets.ground_state_mse_loss(
        params,
        functional,
        dataset,
        training_config=training_config,
    )

    assert np.isclose(float(loss_batched), float(loss_loop), atol=1e-10)
    assert np.allclose(
        np.asarray(metrics_batched["predicted_total_energies"]),
        np.asarray(metrics_loop["predicted_total_energies"]),
        atol=1e-10,
    )
    assert np.allclose(
        np.asarray(metrics_batched["energy_mae"]),
        np.asarray(metrics_loop["energy_mae"]),
        atol=1e-10,
    )
    assert np.allclose(
        np.asarray(metrics_batched["scf_converged"]),
        np.asarray(metrics_loop["scf_converged"]),
        atol=1e-10,
    )


def test_impl_multi_datum_self_consistent_loss_skips_batched_fast_path(monkeypatch):
    _pyscf_or_skip()
    molecule_a = _make_h2_reference(half_distance_angstrom=0.35)
    molecule_b = _make_h2_reference(half_distance_angstrom=0.70)
    functional, params = _make_functional_and_params(molecule_a)

    dataset = [
        GroundStateDatum(
            molecule=molecule_a,
            target_total_energy=np.asarray(molecule_a.mf_energy),
        ),
        GroundStateDatum(
            molecule=molecule_b,
            target_total_energy=np.asarray(molecule_b.mf_energy),
        ),
    ]
    training_config = GroundStateTrainingConfig(
        mode="self_consistent",
        scf_gradient_mode="impl",
        scf_max_cycle=4,
        scf_damping=0.2,
        scf_conv_tol_density=1e-7,
        scf_implicit_diff_max_iter=8,
        scf_implicit_diff_tolerance=1e-5,
        scf_implicit_diff_regularization=1e-3,
        scf_implicit_diff_restart=4,
    )

    monkeypatch.setattr(
        training_targets,
        "_ground_state_mse_loss_batched_self_consistent",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("impl path should not use batched self-consistent fast path")
        ),
    )
    loss_loop, metrics_loop = training_targets.ground_state_mse_loss(
        params,
        functional,
        dataset,
        training_config=training_config,
    )

    assert np.isfinite(float(loss_loop))
    assert np.all(np.isfinite(np.asarray(metrics_loop["predicted_total_energies"])))
    assert np.all(np.isfinite(np.asarray(metrics_loop["energy_mae"])))
    assert np.all(np.isfinite(np.asarray(metrics_loop["scf_cycles"])))
    assert np.isfinite(float(metrics_loop["scf_converged_fraction"][0]))


# ---------------------------------------------------------------------------
# Tests for gradient_mode switching between expl and impl
# ---------------------------------------------------------------------------


def test_expl_gradient_mode_produces_finite_energy():
    """gradient_mode='expl' runs the full SCF loop and returns a converged density."""
    _pyscf_or_skip()
    molecule = _make_h2_reference()
    functional, params = _make_functional_and_params(molecule)

    solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            gradient_mode="expl",
            max_cycle=8,
            damping=0.2,
            conv_tol_density=1e-8,
        )
    )
    molecule_sc, info = solver.run(molecule, functional, params)

    assert info.mode == "self_consistent"
    assert np.isfinite(np.asarray(molecule_sc.rdm1)).all()
    assert np.isfinite(float(info.final_rms_density))


def test_expl_restricted_scf_uses_shared_rks_diis_loop(monkeypatch):
    molecule = _make_toy_restricted_reference()

    class _ToyRestrictedFunctional:
        def scf_potential_components_and_alpha(self, params, molecule_in):
            strength = jnp.asarray(params["strength"], dtype=jnp.float32)
            v_rho = strength * jnp.asarray([1.0, -0.2, 0.4], dtype=jnp.float32)
            v_grad = jnp.zeros((int(molecule_in.ao.shape[0]), 3), dtype=jnp.float32)
            return v_rho, v_grad, "LDA", jnp.asarray(0.0, dtype=jnp.float32)

    diis_calls = []
    original_diis = scf_rks._diis_extrapolate

    def _recording_diis(*args, **kwargs):
        diis_calls.append(True)
        return original_diis(*args, **kwargs)

    monkeypatch.setattr(scf_rks, "_diis_extrapolate", _recording_diis)

    solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            gradient_mode="expl",
            max_cycle=3,
            conv_tol_density=0.0,
        )
    )
    _, info = solver.run(
        molecule,
        _ToyRestrictedFunctional(),
        {"strength": jnp.asarray(0.1, dtype=jnp.float32)},
    )

    assert int(info.cycles) == 3
    assert int(info.selected_cycle) == 3
    assert np.asarray(info.rms_density_history).shape == (3,)
    assert diis_calls


def test_expl_unrestricted_scf_uses_fixed_cycle_loop_without_level_shift(monkeypatch):
    molecule = _make_toy_unrestricted_reference()

    class _ToyUnrestrictedFunctional:
        def bind_to_molecule_for_scf(self, params, _molecule):
            strength = jnp.asarray(params["strength"], dtype=jnp.float32)

            class _Bound:
                exact_exchange_fraction = jnp.asarray(0.0, dtype=jnp.float32)

                def unrestricted_scf_components(self, molecule_in):
                    ngrids = int(molecule_in.ao.shape[0])
                    zeros_grad = jnp.zeros((ngrids, 3), dtype=jnp.float32)
                    zeros_mat = jnp.zeros(
                        (molecule_in.ao.shape[1], molecule_in.ao.shape[1]),
                        dtype=jnp.float32,
                    )
                    v_alpha = strength * jnp.asarray([1.0, -0.2, 0.4], dtype=jnp.float32)
                    v_beta = -strength * jnp.asarray([0.1, 0.3, -0.5], dtype=jnp.float32)
                    return (
                        v_alpha,
                        v_beta,
                        zeros_grad,
                        zeros_grad,
                        "LDA",
                        jnp.asarray(0.0),
                        zeros_mat,
                        zeros_mat,
                    )

            return _Bound()

    def _unexpected_level_shift(*args, **kwargs):
        del args, kwargs
        raise AssertionError("GradDFT-style explicit SCF loop must not use level shift.")

    monkeypatch.setattr(
        scf_differentiable,
        "_apply_level_shift_spin",
        _unexpected_level_shift,
        raising=False,
    )

    solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            gradient_mode="expl",
            max_cycle=3,
            level_shift=0.7,
            conv_tol_density=1e9,
        )
    )
    _, info = solver.run(
        molecule,
        _ToyUnrestrictedFunctional(),
        {"strength": jnp.asarray(0.1, dtype=jnp.float32)},
    )

    assert int(info.cycles) == 3
    assert int(info.selected_cycle) == 3
    assert np.asarray(info.rms_density_history).shape == (3,)


def test_expl_mode_gradient_is_finite():
    """gradient_mode='expl' produces finite gradients through the SCF loop."""
    _pyscf_or_skip()
    molecule = _make_h2_reference()
    functional, params = _make_functional_and_params(molecule)

    solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            gradient_mode="expl",
            max_cycle=8,
            damping=0.25,
            conv_tol_density=1e-8,
        )
    )

    def loss_fn(p):
        mol_sc, _ = solver.run(molecule, functional, p)
        energy = training_targets._predict_ground_state_total_energy_from_molecule(
            p, functional, mol_sc,
        )
        return energy

    loss, grads = jax.value_and_grad(loss_fn)(params)
    assert np.isfinite(float(loss))
    grad_leaves = jax.tree_util.tree_leaves(grads)
    assert len(grad_leaves) > 0
    for g in grad_leaves:
        assert np.isfinite(np.asarray(g)).all(), "gradient contains non-finite values"


def test_implicit_mode_gradient_is_finite():
    """gradient_mode='impl' with input_state produces finite gradients."""
    _pyscf_or_skip()
    molecule = _make_h2_reference()
    functional, params = _make_functional_and_params(molecule)

    solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            gradient_mode="impl",
            implicit_forward_mode="input_state",
            max_cycle=8,
            damping=0.25,
            conv_tol_density=1e-8,
        )
    )

    def loss_fn(p):
        mol_sc, _ = solver.run(molecule, functional, p)
        energy = training_targets._predict_ground_state_total_energy_from_molecule(
            p, functional, mol_sc,
        )
        return energy

    loss, grads = jax.value_and_grad(loss_fn)(params)
    assert np.isfinite(float(loss))
    grad_leaves = jax.tree_util.tree_leaves(grads)
    assert len(grad_leaves) > 0
    for g in grad_leaves:
        assert np.isfinite(np.asarray(g)).all(), "gradient contains non-finite values"


def test_implicit_mode_with_expl_forward_is_finite():
    """gradient_mode='impl' with implicit_forward_mode='expl' works."""
    _pyscf_or_skip()
    molecule = _make_h2_reference()
    functional, params = _make_functional_and_params(molecule)

    solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            gradient_mode="impl",
            implicit_forward_mode="expl",
            max_cycle=8,
            damping=0.25,
            conv_tol_density=1e-8,
        )
    )

    def loss_fn(p):
        mol_sc, info = solver.run(molecule, functional, p)
        # The forward state info should reflect a full SCF run (not input_state).
        assert info.mode == "self_consistent"
        energy = training_targets._predict_ground_state_total_energy_from_molecule(
            p, functional, mol_sc,
        )
        return energy

    loss, grads = jax.value_and_grad(loss_fn)(params)
    assert np.isfinite(float(loss))
    for g in jax.tree_util.tree_leaves(grads):
        assert np.isfinite(np.asarray(g)).all()


def test_expl_and_implicit_gradients_are_consistent():
    """Gradients from expl and implicit modes should be directionally similar."""
    _pyscf_or_skip()
    molecule = _make_h2_reference()
    functional, params = _make_functional_and_params(molecule)

    # Unrolled gradient
    solver_expl = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            gradient_mode="expl",
            max_cycle=8,
            damping=0.25,
            conv_tol_density=1e-8,
        )
    )

    def loss_expl(p):
        mol_sc, _ = solver_expl.run(molecule, functional, p)
        return training_targets._predict_ground_state_total_energy_from_molecule(
            p, functional, mol_sc,
        )

    _, grads_expl = jax.value_and_grad(loss_expl)(params)

    # Implicit gradient (use expl forward for a tighter comparison)
    solver_implicit = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            gradient_mode="impl",
            implicit_forward_mode="expl",
            max_cycle=8,
            damping=0.25,
            conv_tol_density=1e-8,
            implicit_diff_max_iter=24,
            implicit_diff_tolerance=1e-6,
        )
    )

    def loss_implicit(p):
        mol_sc, _ = solver_implicit.run(molecule, functional, p)
        return training_targets._predict_ground_state_total_energy_from_molecule(
            p, functional, mol_sc,
        )

    _, grads_implicit = jax.value_and_grad(loss_implicit)(params)

    # Compare: the two gradient vectors should have positive cosine similarity.
    def _flatten(g):
        leaves = jax.tree_util.tree_leaves(g)
        return jnp.concatenate([jnp.asarray(x).ravel() for x in leaves])

    g_expl = _flatten(grads_expl)
    g_implicit = _flatten(grads_implicit)

    cos_sim = jnp.dot(g_expl, g_implicit) / (
        jnp.linalg.norm(g_expl) * jnp.linalg.norm(g_implicit) + 1e-12
    )
    assert float(cos_sim) > 0.9, (
        f"Gradient cosine similarity {float(cos_sim):.6f} < 0.9, "
        "expl and implicit gradients disagree too much."
    )


def test_config_rejects_invalid_gradient_mode():
    with pytest.raises(ValueError, match="gradient_mode must be one of"):
        DifferentiableSCFConfig(gradient_mode="explicit")


def test_config_rejects_invalid_forward_mode():
    with pytest.raises(ValueError, match="implicit_forward_mode must be one of"):
        DifferentiableSCFConfig(implicit_forward_mode="none")


def test_mode_switch_via_config():
    """Verify that switching between expl and implicit modes is a simple config change."""
    expl_cfg = DifferentiableSCFConfig(gradient_mode="expl")
    assert expl_cfg.gradient_mode == "expl"

    implicit_cfg = replace(expl_cfg, gradient_mode="impl",
                           implicit_forward_mode="input_state")
    assert implicit_cfg.gradient_mode == "impl"
    assert implicit_cfg.implicit_forward_mode == "input_state"

    # Both configs should be usable to create solvers.
    solver_expl = DifferentiableSCF(expl_cfg)
    assert solver_expl.config.gradient_mode == "expl"

    solver_implicit = DifferentiableSCF(implicit_cfg)
    assert solver_implicit.config.gradient_mode == "impl"
