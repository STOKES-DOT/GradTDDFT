from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn

from td_graddft.tddft import (
    GridPointModeLongRangeCorrectedFunctional,
    LongRangeCorrectedFunctional,
)
from td_graddft.training import (
    ExcitedStateFineTuneConfig,
    ExcitedStateFineTuner,
    GroundStateDatum,
    predict_excitation_energies,
)
from td_graddft.xc import lda_from_callable


@dataclass
class _Grid:
    weights: jnp.ndarray
    coords: jnp.ndarray


@dataclass
class _ToyMolecule:
    ao: jnp.ndarray
    ao_deriv1: jnp.ndarray
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
    ao_deriv1 = jnp.array(
        [
            ao,
            [[0.20, 0.10], [0.10, 0.30]],
            [[0.05, 0.00], [0.00, 0.04]],
            [[0.00, 0.07], [0.08, 0.00]],
        ]
    )
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
        ao_deriv1=ao_deriv1,
        grid=_Grid(
            weights=jnp.array([1.0, 1.0]),
            coords=jnp.array([[0.0, 0.0, 0.0], [1.25, 0.0, 0.0]]),
        ),
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


class _ScalarAlphaKernel(nn.Module):
    gamma: float = 1.5

    @nn.compact
    def __call__(self, pair_features):
        alpha_raw = self.param("alpha_raw", lambda key: jnp.asarray(-4.0))
        alpha_value = jax.nn.softplus(alpha_raw)
        pair_shape = jnp.asarray(pair_features).shape[:-1] + (1,)
        alpha = jnp.full(pair_shape, alpha_value)
        gamma = jnp.full(pair_shape, self.gamma)
        return alpha, gamma


class _ScalarModeCoupling(nn.Module):
    @nn.compact
    def __call__(self, mode_features):
        raw = self.param("raw", lambda key: jnp.asarray(-4.0))
        value = jax.nn.softplus(raw)
        nmode = int(jnp.asarray(mode_features).shape[0])
        vec = jnp.full((nmode,), value)
        return -jnp.outer(vec, vec)


def test_excited_state_fine_tuner_updates_long_range_correction_wrapper():
    molecule = _make_toy_molecule()
    base = lda_from_callable("zero", lambda rho: jnp.zeros_like(rho))
    functional = LongRangeCorrectedFunctional(
        base_functional=base,
        model=_ScalarAlphaKernel(gamma=1.6),
        distance_floor=0.35,
    )

    initial_lr_params = functional.init(jax.random.PRNGKey(0), molecule)
    target_lr_params = {"params": {"alpha_raw": jnp.asarray(-0.8)}}
    initial_params = functional.combine_params(None, initial_lr_params)
    target_params = functional.combine_params(None, target_lr_params)

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
        steps=80,
        learning_rate=0.1,
        excited_states=(1,),
        use_tda=True,
        weight_energy=1.0,
        weight_ground_state_energy=0.0,
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
    initial_excitation = predict_excitation_energies(
        initial_params,
        functional,
        molecule,
        nstates=1,
        use_tda=True,
    )

    assert result.best_loss < result.initial_loss
    assert jnp.abs(final_excitation[0] - target_excitation[0]) < jnp.abs(
        initial_excitation[0] - target_excitation[0]
    )
    assert not jnp.allclose(
        result.params["lr_correction"]["params"]["alpha_raw"],
        initial_params["lr_correction"]["params"]["alpha_raw"],
    )


def test_excited_state_fine_tuner_updates_grid_point_mode_wrapper():
    molecule = _make_toy_molecule()
    base = lda_from_callable("zero", lambda rho: jnp.zeros_like(rho))
    functional = GridPointModeLongRangeCorrectedFunctional(
        base_functional=base,
        model=_ScalarModeCoupling(),
        max_mode_points=2,
    )

    initial_lr_params = functional.init(jax.random.PRNGKey(0), molecule)
    target_lr_params = {"params": {"raw": jnp.asarray(-0.8)}}
    initial_params = functional.combine_params(None, initial_lr_params)
    target_params = functional.combine_params(None, target_lr_params)

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
        steps=80,
        learning_rate=0.1,
        excited_states=(1,),
        use_tda=True,
        weight_energy=1.0,
        weight_ground_state_energy=0.0,
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
    initial_excitation = predict_excitation_energies(
        initial_params,
        functional,
        molecule,
        nstates=1,
        use_tda=True,
    )

    assert result.best_loss < result.initial_loss
    assert jnp.abs(final_excitation[0] - target_excitation[0]) < jnp.abs(
        initial_excitation[0] - target_excitation[0]
    )
    assert not jnp.allclose(
        result.params["lr_correction"]["params"]["raw"],
        initial_params["lr_correction"]["params"]["raw"],
    )
