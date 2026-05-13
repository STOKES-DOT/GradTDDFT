from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import td_graddft.training.trainer as training_trainer
from td_graddft.scf import DifferentiableSCF, GPU4PYSCF_RKS_RUNTIME_BACKEND
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
