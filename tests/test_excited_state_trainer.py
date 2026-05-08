from dataclasses import dataclass

import jax.numpy as jnp
import pytest

from td_graddft.training import (
    ExcitedStateFineTuneConfig,
    ExcitedStateFineTuner,
    GroundStateDatum,
    predict_excitation_energies,
)


@dataclass
class _Grid:
    weights: jnp.ndarray


@dataclass
class _ToyMolecule:
    ao: jnp.ndarray
    grid: _Grid
    rep_tensor: jnp.ndarray
    mo_coeff: jnp.ndarray
    mo_occ: jnp.ndarray
    mo_energy: jnp.ndarray
    rdm1: jnp.ndarray
    dipole_integrals: jnp.ndarray

    def density(self):
        return jnp.einsum("spq,rp,rq->rs", self.rdm1, self.ao, self.ao)


def _make_toy_molecule():
    ao = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    mo_coeff = jnp.stack([jnp.eye(2), jnp.eye(2)], axis=0)
    mo_occ = jnp.array([[1.0, 0.0], [1.0, 0.0]])
    mo_energy = jnp.array([[0.0, 1.0], [0.0, 1.0]])
    rdm1 = jnp.array(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0]],
        ]
    )
    return _ToyMolecule(
        ao=ao,
        grid=_Grid(weights=jnp.array([1.0, 1.0])),
        rep_tensor=jnp.zeros((2, 2, 2, 2)),
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        rdm1=rdm1,
        dipole_integrals=jnp.array(
            [
                [[0.0, 1.0], [1.0, 0.0]],
                [[0.0, 0.2], [0.2, 0.0]],
                [[0.0, 0.1], [0.1, 0.0]],
            ]
        ),
    )


@dataclass(frozen=True)
class _BoundKernelFunctional:
    kernel_value: jnp.ndarray
    exact_exchange_fraction: float = 0.0

    def local_kernel(self, density):
        return jnp.full_like(jnp.asarray(density), self.kernel_value)

    def local_potential(self, density):
        return jnp.zeros_like(jnp.asarray(density))


class _SplitParamFunctional:
    def bind_to_molecule(self, params, molecule):
        del molecule
        return _BoundKernelFunctional(
            kernel_value=jnp.asarray(params["params"]["base"] + params["params"]["lr_correction"])
        )


def test_excited_state_fine_tuner_updates_only_selected_subtree_and_reduces_excitation_error():
    molecule = _make_toy_molecule()
    functional = _SplitParamFunctional()
    initial_params = {
        "params": {
            "base": jnp.asarray(0.15),
            "lr_correction": jnp.asarray(0.0),
        }
    }
    target_params = {
        "params": {
            "base": jnp.asarray(0.15),
            "lr_correction": jnp.asarray(0.45),
        }
    }
    target_excitation = predict_excitation_energies(
        target_params,
        functional,
        molecule,
        nstates=1,
        use_tda=True,
    )
    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.asarray(0.0),
        target_excitation_energies=target_excitation,
    )
    config = ExcitedStateFineTuneConfig(
        steps=60,
        learning_rate=0.15,
        excited_states=(1,),
        use_tda=True,
        weight_energy=1.0,
        energy_loss="mae",
        freeze_ground_state_params=True,
        trainable_path_prefixes=("lr_correction",),
    )

    trainer = ExcitedStateFineTuner(config, functional, initial_params)
    result = trainer.fine_tune(datum)

    final_excitation = predict_excitation_energies(
        result.params,
        functional,
        molecule,
        nstates=1,
        use_tda=True,
    )
    initial_error = jnp.abs(
        predict_excitation_energies(
            initial_params,
            functional,
            molecule,
            nstates=1,
            use_tda=True,
        )[0]
        - target_excitation[0]
    )
    final_error = jnp.abs(final_excitation[0] - target_excitation[0])

    assert result.best_loss < result.initial_loss
    assert final_error < initial_error
    assert jnp.allclose(result.params["params"]["base"], initial_params["params"]["base"])
    assert not jnp.allclose(
        result.params["params"]["lr_correction"],
        initial_params["params"]["lr_correction"],
    )


def test_excited_state_fine_tuner_requires_matching_trainable_prefix():
    molecule = _make_toy_molecule()
    functional = _SplitParamFunctional()
    params = {
        "params": {
            "base": jnp.asarray(0.1),
            "lr_correction": jnp.asarray(0.0),
        }
    }
    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.asarray(0.0),
        target_excitation_energies=jnp.asarray([1.0]),
    )
    trainer = ExcitedStateFineTuner(
        ExcitedStateFineTuneConfig(
            steps=2,
            learning_rate=0.1,
            excited_states=(1,),
            trainable_path_prefixes=("missing_branch",),
        ),
        functional,
        params,
    )

    with pytest.raises(ValueError, match="no parameter leaves matched"):
        trainer.fine_tune(datum)


def test_excited_state_fine_tune_config_rejects_unknown_energy_loss():
    with pytest.raises(ValueError, match="energy_loss"):
        ExcitedStateFineTuneConfig(
            steps=2,
            learning_rate=0.1,
            energy_loss="unknown",
        )
