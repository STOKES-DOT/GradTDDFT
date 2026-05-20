from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import td_graddft.training.trainer as training_trainer
from td_graddft.scf import (
    DifferentiableSCF,
    DifferentiableSCFConfig,
    GPU4PYSCF_RKS_RUNTIME_BACKEND,
)
import td_graddft.scf.differentiable as scf_differentiable
from td_graddft.scf.gpu4pyscf import (
    GPU4PySCFRKSForwardOptions,
    GPU4PySCFRKSForwardResult,
)
from td_graddft.scf.molecules import QuadratureGrid, RestrictedMolecule
from td_graddft.training import (
    GroundStateDatum,
    GroundStateTrainingConfig,
    make_ground_state_loss_and_grad,
)


def _make_gpu4pyscf_marked_toy_molecule():
    ao = jnp.asarray(
        [
            [1.0, 0.2],
            [0.8, -0.1],
            [0.4, 0.9],
        ],
        dtype=jnp.float32,
    )
    weights = jnp.asarray([0.5, 0.7, 0.6], dtype=jnp.float32)
    mo_coeff = jnp.eye(2, dtype=jnp.float32)
    mo_occ = jnp.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)
    mo_energy = jnp.asarray([[-0.8, 0.2], [-0.8, 0.2]], dtype=jnp.float32)
    density_half = jnp.einsum("pi,i,qi->pq", mo_coeff, mo_occ[0], mo_coeff)
    return RestrictedMolecule(
        ao=ao,
        grid=QuadratureGrid(weights=weights),
        dipole_integrals=jnp.zeros((3, 2, 2), dtype=jnp.float32),
        rep_tensor=jnp.zeros((2, 2, 2, 2), dtype=jnp.float32),
        mo_coeff=jnp.stack([mo_coeff, mo_coeff], axis=0),
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        rdm1=jnp.stack([density_half, density_half], axis=0),
        h1e=jnp.diag(jnp.asarray([-0.8, 0.2], dtype=jnp.float32)),
        nuclear_repulsion=0.0,
        overlap_matrix=jnp.eye(2, dtype=jnp.float32),
        runtime_scf_backend=GPU4PYSCF_RKS_RUNTIME_BACKEND,
    )


class _ToyRestrictedFunctional:
    def scf_potential_components_and_alpha(self, params, molecule_in):
        strength = jnp.asarray(params["strength"], dtype=jnp.float32)
        v_rho = strength * jnp.asarray([1.0, -0.2, 0.4], dtype=jnp.float32)
        v_grad = jnp.zeros((int(molecule_in.ao.shape[0]), 3), dtype=jnp.float32)
        return v_rho, v_grad, "LDA", jnp.asarray(0.0, dtype=jnp.float32)

    def energy(self, params, density, weights):
        strength = jnp.asarray(params["strength"], dtype=jnp.float32)
        return strength * jnp.sum(jnp.asarray(density) * jnp.asarray(weights))


def test_ground_state_loss_passes_forward_data_as_dynamic_value_and_grad_arg(monkeypatch):
    molecule = _make_gpu4pyscf_marked_toy_molecule()
    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.asarray(0.0, dtype=jnp.float32),
    )
    params = {"strength": jnp.asarray(0.1, dtype=jnp.float32)}
    observed = {}

    def _fake_value_and_grad(fn, *, has_aux=False, argnums=0):
        del fn
        observed["has_aux"] = has_aux
        observed["argnums"] = argnums

        def _wrapped(*args):
            observed["argc"] = len(args)
            observed["data_arg"] = args[1] if len(args) > 1 else None
            return (
                (jnp.asarray(0.0, dtype=jnp.float32), {}),
                {"strength": jnp.asarray(0.0, dtype=jnp.float32)},
            )

        return _wrapped

    monkeypatch.setattr(training_trainer.jax, "value_and_grad", _fake_value_and_grad)

    loss_and_grad = make_ground_state_loss_and_grad(
        _ToyRestrictedFunctional(),
        training_config=GroundStateTrainingConfig(mode="fixed_density"),
    )
    loss, _, _ = loss_and_grad(params, datum)

    assert float(loss) == 0.0
    assert observed["has_aux"] is True
    assert observed["argnums"] == 0
    assert observed["argc"] == 2
    assert observed["data_arg"] is datum


def test_default_implicit_response_backend_keeps_jax_jk_path(monkeypatch):
    molecule = replace(
        _make_gpu4pyscf_marked_toy_molecule(),
        runtime_scf_options=GPU4PySCFRKSForwardOptions(
            atom="H 0 0 0; H 0 0 0.74",
            basis="sto-3g",
        ),
    )
    functional = _ToyRestrictedFunctional()
    params = {"strength": jnp.asarray(0.1, dtype=jnp.float32)}
    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.asarray(0.0, dtype=jnp.float32),
    )

    def _unexpected_gpu4pyscf_jk(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Default implicit response backend must keep JAX JK response.")

    monkeypatch.setattr(
        scf_differentiable,
        "_gpu4pyscf_jk_response_matrices",
        _unexpected_gpu4pyscf_jk,
        raising=False,
    )

    solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            gradient_mode="impl",
            implicit_forward_mode="input_state",
            implicit_diff_max_iter=2,
        )
    )

    def objective(local_params):
        molecule_out, _ = solver.run(molecule, functional, local_params)
        return jnp.sum(jnp.asarray(molecule_out.rdm1))

    loss, grads = jax.value_and_grad(objective)(params)

    assert np.isfinite(float(loss))
    assert jnp.isfinite(grads["strength"])


def test_gpu4pyscf_implicit_response_backend_calls_jk_helper(monkeypatch):
    molecule = replace(
        _make_gpu4pyscf_marked_toy_molecule(),
        runtime_scf_options=GPU4PySCFRKSForwardOptions(
            atom="H 0 0 0; H 0 0 0.74",
            basis="sto-3g",
        ),
    )
    functional = _ToyRestrictedFunctional()
    params = {"strength": jnp.asarray(0.1, dtype=jnp.float32)}
    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.asarray(0.0, dtype=jnp.float32),
    )
    calls = []

    def _fake_gpu4pyscf_jk(molecule_in, density, *, dtype, with_k=True):
        calls.append((molecule_in.runtime_scf_backend, tuple(jnp.asarray(density).shape), bool(with_k)))
        zeros = jnp.zeros_like(jnp.asarray(density, dtype=dtype))
        return zeros, zeros

    monkeypatch.setattr(
        scf_differentiable,
        "_gpu4pyscf_jk_response_matrices",
        _fake_gpu4pyscf_jk,
        raising=False,
    )

    solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            gradient_mode="impl",
            implicit_forward_mode="input_state",
            implicit_response_backend="gpu4pyscf_jk",
            implicit_diff_max_iter=2,
        )
    )

    def objective(local_params):
        molecule_out, _ = solver.run(molecule, functional, local_params)
        return jnp.sum(jnp.asarray(molecule_out.rdm1))

    loss, grads = jax.value_and_grad(objective)(params)

    assert np.isfinite(float(loss))
    assert jnp.isfinite(grads["strength"])
    assert calls
    assert calls[0] == (GPU4PYSCF_RKS_RUNTIME_BACKEND, (2, 2), True)


def test_explicit_hfx_implicit_response_skips_exchange_callback(monkeypatch):
    molecule = replace(
        _make_gpu4pyscf_marked_toy_molecule(),
        runtime_scf_options=GPU4PySCFRKSForwardOptions(
            atom="H 0 0 0; H 0 0 0.74",
            basis="sto-3g",
        ),
        hfx_nu=jnp.zeros((1, 3, 2, 2), dtype=jnp.float32),
    )
    params = {"strength": jnp.asarray(0.1, dtype=jnp.float32)}
    calls = []

    class _ExplicitHFXToyFunctional(_ToyRestrictedFunctional):
        def uses_explicit_hfx_fock_for_scf(self, molecule_in):
            return getattr(molecule_in, "hfx_nu", None) is not None

        def scf_potential_components_and_alpha(self, params_in, molecule_in):
            v_rho, v_grad, xc_kind, _ = super().scf_potential_components_and_alpha(
                params_in,
                molecule_in,
            )
            zeros = jnp.zeros_like(v_rho)
            extra_fock = jnp.zeros((2, 2), dtype=v_rho.dtype)
            return v_rho, v_grad, zeros, zeros, xc_kind, jnp.asarray(0.0, dtype=v_rho.dtype), extra_fock

    def _fake_gpu4pyscf_jk(molecule_in, density, *, dtype, with_k=True):
        calls.append(bool(with_k))
        zeros = jnp.zeros_like(jnp.asarray(density, dtype=dtype))
        return zeros, zeros

    monkeypatch.setattr(
        scf_differentiable,
        "_gpu4pyscf_jk_response_matrices",
        _fake_gpu4pyscf_jk,
        raising=False,
    )

    solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            gradient_mode="impl",
            implicit_forward_mode="input_state",
            implicit_response_backend="gpu4pyscf_jk",
            implicit_diff_max_iter=1,
        )
    )

    def objective(local_params):
        molecule_out, _ = solver.run(molecule, _ExplicitHFXToyFunctional(), local_params)
        return jnp.sum(jnp.asarray(molecule_out.rdm1))

    _, grads = jax.value_and_grad(objective)(params)

    assert jnp.isfinite(grads["strength"])
    assert calls
    assert all(with_k is False for with_k in calls)


def test_implicit_backward_uses_direct_fixed_point_solve(monkeypatch):
    molecule = replace(
        _make_gpu4pyscf_marked_toy_molecule(),
        runtime_scf_options=GPU4PySCFRKSForwardOptions(
            atom="H 0 0 0; H 0 0 0.74",
            basis="sto-3g",
        ),
    )
    functional = _ToyRestrictedFunctional()
    params = {"strength": jnp.asarray(0.1, dtype=jnp.float32)}

    def _fake_gpu4pyscf_jk(molecule_in, density, *, dtype, with_k=True):
        del molecule_in
        zeros = jnp.zeros_like(jnp.asarray(density, dtype=dtype))
        return zeros, zeros

    monkeypatch.setattr(
        scf_differentiable,
        "_gpu4pyscf_jk_response_matrices",
        _fake_gpu4pyscf_jk,
        raising=False,
    )
    assert not hasattr(scf_differentiable, "_solve_implicit_normal_equation")

    solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            gradient_mode="impl",
            implicit_forward_mode="input_state",
            implicit_response_backend="gpu4pyscf_jk",
            implicit_diff_max_iter=1,
        )
    )

    def objective(local_params):
        molecule_out, _ = solver.run(molecule, functional, local_params)
        return jnp.sum(jnp.asarray(molecule_out.rdm1))

    loss, grads = jax.value_and_grad(objective)(params)

    assert np.isfinite(float(loss))
    assert jnp.isfinite(grads["strength"])


def test_gpu4pyscf_implicit_response_reuses_fixed_jk_for_param_vjp(monkeypatch):
    molecule = replace(
        _make_gpu4pyscf_marked_toy_molecule(),
        runtime_scf_options=GPU4PySCFRKSForwardOptions(
            atom="H 0 0 0; H 0 0 0.74",
            basis="sto-3g",
        ),
    )
    functional = _ToyRestrictedFunctional()
    params = {"strength": jnp.asarray(0.1, dtype=jnp.float32)}
    calls = []

    def _fake_gpu4pyscf_jk(molecule_in, density, *, dtype, with_k=True):
        calls.append((molecule_in.runtime_scf_backend, tuple(jnp.asarray(density).shape), bool(with_k)))
        zeros = jnp.zeros_like(jnp.asarray(density, dtype=dtype))
        return zeros, zeros

    monkeypatch.setattr(
        scf_differentiable,
        "_gpu4pyscf_jk_response_matrices",
        _fake_gpu4pyscf_jk,
        raising=False,
    )

    solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            gradient_mode="impl",
            implicit_forward_mode="input_state",
            implicit_response_backend="gpu4pyscf_jk",
            implicit_diff_max_iter=1,
        )
    )

    def objective(local_params):
        molecule_out, _ = solver.run(molecule, functional, local_params)
        return jnp.sum(jnp.asarray(molecule_out.rdm1))

    _, grads = jax.value_and_grad(objective)(params)

    assert jnp.isfinite(grads["strength"])
    assert len(calls) == 2


def test_forced_gpu4pyscf_implicit_response_requires_gpu4pyscf_state():
    molecule = replace(
        _make_gpu4pyscf_marked_toy_molecule(),
        runtime_scf_backend=None,
        runtime_scf_options=None,
    )
    functional = _ToyRestrictedFunctional()
    params = {"strength": jnp.asarray(0.1, dtype=jnp.float32)}
    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.asarray(0.0, dtype=jnp.float32),
    )
    solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            gradient_mode="impl",
            implicit_forward_mode="input_state",
            implicit_response_backend="gpu4pyscf_jk",
        )
    )

    with pytest.raises(ValueError, match="GPU4PySCF RKS-backed molecule"):
        del datum
        solver.run(molecule, functional, params)


def test_gpu4pyscf_marked_forward_state_uses_implicit_backward_without_jax_runtime_forward(
    monkeypatch,
):
    molecule = _make_gpu4pyscf_marked_toy_molecule()
    functional = _ToyRestrictedFunctional()
    params = {"strength": jnp.asarray(0.1, dtype=jnp.float32)}
    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.asarray(0.0, dtype=jnp.float32),
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

    def _unexpected_runtime_forward(self, molecule_in, xc_functional, xc_params):
        del self, molecule_in, xc_functional, xc_params
        raise AssertionError("GPU4PySCF forward branch must not use JAX runtime SCF forward.")

    monkeypatch.setattr(
        DifferentiableSCF,
        "run_runtime_forward",
        _unexpected_runtime_forward,
    )

    loss_and_grad = make_ground_state_loss_and_grad(functional, training_config=cfg)
    loss, metrics, grads = loss_and_grad(params, datum)

    assert np.isfinite(float(loss))
    assert int(metrics["scf_cycles"][0]) == 0
    assert jnp.isfinite(grads["strength"])
    assert jnp.abs(grads["strength"]) > 1e-8
    provider = training_trainer.make_self_consistent_runtime_forward_provider(cfg)
    provider_molecule = provider(params, functional, molecule)
    assert provider_molecule is molecule


def test_gpu4pyscf_runtime_forward_policy_can_force_jax_path(monkeypatch):
    molecule = _make_gpu4pyscf_marked_toy_molecule()
    params = {"strength": jnp.asarray(0.1, dtype=jnp.float32)}
    functional = _ToyRestrictedFunctional()
    cfg = GroundStateTrainingConfig(
        mode="self_consistent",
        scf_gradient_mode="impl",
        scf_runtime_forward_backend="jax",
    )
    calls = []

    def _fake_runtime_forward(self, molecule_in, xc_functional, xc_params):
        del self, xc_functional, xc_params
        calls.append("jax_runtime_forward")
        return molecule_in

    monkeypatch.setattr(DifferentiableSCF, "run_runtime_forward", _fake_runtime_forward)

    provider = training_trainer.make_self_consistent_runtime_forward_provider(cfg)
    provider_molecule = provider(params, functional, molecule)

    assert calls == ["jax_runtime_forward"]
    assert provider_molecule is molecule


def test_gpu4pyscf_runtime_forward_policy_runs_neural_xc_gpu4pyscf_forward(monkeypatch):
    molecule = replace(
        _make_gpu4pyscf_marked_toy_molecule(),
        runtime_scf_options=GPU4PySCFRKSForwardOptions(
            atom="H 0 0 0; H 0 0 0.74",
            basis="sto-3g",
            xc_spec="pbe",
            cart=True,
            max_cycle=99,
        ),
    )
    params = {"strength": jnp.asarray(0.1, dtype=jnp.float32)}
    functional = _ToyRestrictedFunctional()
    cfg = GroundStateTrainingConfig(
        mode="self_consistent",
        scf_gradient_mode="impl",
        scf_runtime_forward_backend="gpu4pyscf_rks",
        scf_vxc_clip=7.5,
    )
    calls = []

    def _unexpected_runtime_forward(self, molecule_in, xc_functional, xc_params):
        del self, molecule_in, xc_functional, xc_params
        raise AssertionError("GPU4PySCF runtime forward must not use JAX runtime SCF.")

    def _fake_gpu4pyscf_forward(**kwargs):
        calls.append(kwargs)
        return GPU4PySCFRKSForwardResult(
            converged=True,
            total_energy=-3.0,
            mo_energy=np.array([-0.9, 0.3]),
            mo_coeff=np.eye(2),
            mo_occ=np.array([2.0, 0.0]),
            density_matrix=np.diag([1.6, 0.4]),
            fock_matrix=np.diag([-0.9, 0.3]),
            cycles=5,
            exact_exchange_fraction=0.2,
        )

    monkeypatch.setattr(
        DifferentiableSCF,
        "run_runtime_forward",
        _unexpected_runtime_forward,
    )
    monkeypatch.setattr(
        training_trainer,
        "run_gpu4pyscf_rks_forward",
        _fake_gpu4pyscf_forward,
    )

    provider = training_trainer.make_self_consistent_runtime_forward_provider(cfg)
    provider_molecule = provider(params, functional, molecule)

    assert len(calls) == 1
    call = calls[0]
    assert call["atom"] == "H 0 0 0; H 0 0 0.74"
    assert call["basis"] == "sto-3g"
    assert call["xc_spec"] == "pbe"
    assert call["max_cycle"] == 99
    assert call["require_convergence"] is False
    assert call["neural_xc_compute_exc"] is False
    assert call["neural_xc_jit_payload"] is True
    assert call["collect_fock"] is False
    assert call["molecule_template"] is molecule
    assert call["xc_functional"] is functional
    assert float(call["neural_vxc_clip"]) == 7.5
    assert np.allclose(
        np.asarray(jax.tree_util.tree_map(np.asarray, call["xc_params"])["strength"]),
        np.asarray(params["strength"]),
    )
    assert provider_molecule is not molecule
    assert provider_molecule.runtime_scf_backend == GPU4PYSCF_RKS_RUNTIME_BACKEND
    assert provider_molecule.mf_energy == -3.0
    assert provider_molecule.exact_exchange_fraction == 0.2
    assert np.allclose(np.asarray(provider_molecule.rdm1).sum(axis=0), np.diag([1.6, 0.4]))


def test_gpu4pyscf_runtime_forward_policy_warm_starts_from_cached_density(monkeypatch):
    molecule = replace(
        _make_gpu4pyscf_marked_toy_molecule(),
        runtime_scf_options=GPU4PySCFRKSForwardOptions(
            atom="H 0 0 0; H 0 0 0.74",
            basis="sto-3g",
            xc_spec="pbe",
            cart=True,
            max_cycle=99,
        ),
    )
    params = {"strength": jnp.asarray(0.1, dtype=jnp.float32)}
    functional = _ToyRestrictedFunctional()
    cfg = GroundStateTrainingConfig(
        mode="self_consistent",
        scf_gradient_mode="impl",
        scf_runtime_forward_backend="gpu4pyscf_rks",
    )
    calls = []
    returned_densities = [
        np.diag([1.7, 0.3]),
        np.diag([1.8, 0.2]),
    ]

    def _fake_gpu4pyscf_forward(**kwargs):
        calls.append(kwargs)
        density_matrix = returned_densities[len(calls) - 1]
        return GPU4PySCFRKSForwardResult(
            converged=True,
            total_energy=-3.0,
            mo_energy=np.array([-0.9, 0.3]),
            mo_coeff=np.eye(2),
            mo_occ=np.array([2.0, 0.0]),
            density_matrix=density_matrix,
            fock_matrix=None,
            cycles=5,
            exact_exchange_fraction=0.2,
        )

    monkeypatch.setattr(
        training_trainer,
        "run_gpu4pyscf_rks_forward",
        _fake_gpu4pyscf_forward,
    )

    provider = training_trainer.make_self_consistent_runtime_forward_provider(cfg)
    provider(params, functional, molecule)
    provider(params, functional, molecule)

    assert len(calls) == 2
    assert np.allclose(calls[0]["initial_density_matrix"], np.diag([2.0, 0.0]))
    assert np.allclose(calls[1]["initial_density_matrix"], returned_densities[0])


def test_gpu4pyscf_runtime_forward_policy_requires_gpu4pyscf_state():
    molecule = replace(
        _make_gpu4pyscf_marked_toy_molecule(),
        runtime_scf_backend=None,
    )
    params = {"strength": jnp.asarray(0.1, dtype=jnp.float32)}
    functional = _ToyRestrictedFunctional()
    cfg = GroundStateTrainingConfig(
        mode="self_consistent",
        scf_gradient_mode="impl",
        scf_runtime_forward_backend="gpu4pyscf_rks",
    )

    provider = training_trainer.make_self_consistent_runtime_forward_provider(cfg)

    with pytest.raises(ValueError, match="gpu4pyscf_rks"):
        provider(params, functional, molecule)
