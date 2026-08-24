from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest


def _patch_excitation(monkeypatch, value: float = 0.4) -> None:
    from td_graddft.training import targets

    monkeypatch.setattr(
        targets,
        "_solve_excited_states",
        lambda *args, **kwargs: SimpleNamespace(
            excitation_energies=jnp.asarray([value], dtype=jnp.float64)
        ),
    )


def _predict_e0(params, molecule):
    del params
    return jnp.asarray(1.0, dtype=jnp.float64), molecule


def test_s1_total_and_excitation_gap_are_distinct_objectives(monkeypatch):
    from td_graddft.training import (
        MolecularTrainingConfig,
        MolecularTrainingDatum,
        molecular_loss,
    )

    _patch_excitation(monkeypatch)
    molecule = SimpleNamespace()
    s1_total = MolecularTrainingDatum(
        molecule=molecule,
        target_s1_total_h=jnp.asarray(1.5, dtype=jnp.float64),
    )
    gap = MolecularTrainingDatum(
        molecule=molecule,
        target_excitation_gaps_h=jnp.asarray([0.5], dtype=jnp.float64),
    )

    total_loss, total_metrics = molecular_loss(
        {},
        object(),
        s1_total,
        training_config=MolecularTrainingConfig(
            s1_total_mse_weight=1.0,
            s1_total_mae_weight=1.0,
        ),
        predictor=_predict_e0,
    )
    gap_loss, gap_metrics = molecular_loss(
        {},
        object(),
        gap,
        training_config=MolecularTrainingConfig(
            excitation_gap_mse_weight=1.0,
            excitation_gap_mae_weight=1.0,
        ),
        predictor=_predict_e0,
    )

    assert float(total_loss) == pytest.approx(0.11)
    assert float(gap_loss) == pytest.approx(0.11)
    assert float(total_metrics["s1_total_predicted_h"][0]) == pytest.approx(1.4)
    assert float(gap_metrics["excitation_gap_predicted_h"][0]) == pytest.approx(0.4)
    assert total_metrics["excitation_gap_predicted_h"].size == 0
    assert gap_metrics["s1_total_predicted_h"].size == 0


def test_total_loss_is_sum_of_named_weighted_components(monkeypatch):
    from td_graddft.training import (
        MolecularTrainingConfig,
        MolecularTrainingDatum,
        molecular_loss,
    )

    _patch_excitation(monkeypatch)
    datum = MolecularTrainingDatum(
        molecule=SimpleNamespace(),
        target_e0_total_h=jnp.asarray(0.8),
        target_s1_total_h=jnp.asarray(1.5),
        target_excitation_gaps_h=jnp.asarray([0.5]),
    )
    config = MolecularTrainingConfig(
        e0_total_mae_weight=1.0,
        s1_total_mae_weight=1.0,
        excitation_gap_mae_weight=1.0,
    )

    loss, metrics = molecular_loss(
        {},
        object(),
        datum,
        training_config=config,
        predictor=_predict_e0,
    )
    component_sum = sum(
        float(metrics[name][0])
        for name in (
            "e0_total_loss",
            "s1_total_loss",
            "excitation_gap_loss",
        )
    )

    assert float(loss) == pytest.approx(0.4)
    assert component_sum == pytest.approx(float(metrics["total_loss"]))


def test_active_objective_requires_matching_target(monkeypatch):
    from td_graddft.training import (
        MolecularTrainingConfig,
        MolecularTrainingDatum,
        molecular_loss,
    )

    _patch_excitation(monkeypatch)
    datum = MolecularTrainingDatum(molecule=SimpleNamespace())

    with pytest.raises(ValueError, match="target_s1_total_h"):
        molecular_loss(
            {},
            object(),
            datum,
            training_config=MolecularTrainingConfig(s1_total_mae_weight=1.0),
            predictor=_predict_e0,
        )


def test_excitation_gaps_require_one_dimensional_target():
    from td_graddft.training import MolecularTrainingDatum

    with pytest.raises(ValueError, match="target_excitation_gaps_h"):
        MolecularTrainingDatum(
            molecule=SimpleNamespace(),
            target_excitation_gaps_h=jnp.asarray(0.5, dtype=jnp.float64),
        )


def test_molecular_datum_is_jittable_with_static_sample_weight(monkeypatch):
    from td_graddft.training import (
        MolecularTrainingConfig,
        MolecularTrainingDatum,
        molecular_loss,
    )

    _patch_excitation(monkeypatch)
    datum = MolecularTrainingDatum(
        molecule={"token": jnp.asarray(0.0)},
        target_excitation_gaps_h=jnp.asarray([0.5]),
        weight=2.0,
    )
    config = MolecularTrainingConfig(excitation_gap_mse_weight=1.0)
    compiled = jax.jit(
        lambda local_datum: molecular_loss(
            {},
            object(),
            local_datum,
            training_config=config,
            predictor=_predict_e0,
        )[0]
    )

    assert float(compiled(datum)) == pytest.approx(0.01)


def test_density_objectives_do_not_fallback_between_grid_and_matrix():
    from td_graddft.training import (
        MolecularTrainingConfig,
        MolecularTrainingDatum,
        molecular_loss,
    )

    datum = MolecularTrainingDatum(
        molecule=SimpleNamespace(),
        target_density_matrix=jnp.eye(1, dtype=jnp.float64),
    )
    with pytest.raises(ValueError, match="target_grid_density"):
        molecular_loss(
            {},
            object(),
            datum,
            training_config=MolecularTrainingConfig(grid_density_mse_weight=1.0),
            predictor=_predict_e0,
        )


def test_empty_dataset_and_empty_objective_are_rejected():
    from td_graddft.training import (
        MolecularTrainingConfig,
        MolecularTrainingDatum,
        molecular_loss,
    )

    with pytest.raises(ValueError, match="dataset must not be empty"):
        molecular_loss(
            {},
            object(),
            (),
            training_config=MolecularTrainingConfig(e0_total_mae_weight=1.0),
            predictor=_predict_e0,
        )

    with pytest.raises(ValueError, match="active loss component"):
        molecular_loss(
            {},
            object(),
            MolecularTrainingDatum(molecule=SimpleNamespace()),
            training_config=MolecularTrainingConfig(),
            predictor=_predict_e0,
        )


def test_legacy_ambiguous_training_api_is_not_public():
    import td_graddft.training as training

    for name in (
        "GroundStateDatum",
        "ExcitedStateDatum",
        "GroundStateCoreDatum",
        "GroundStateTrainingConfig",
        "ground_state_mse_loss",
        "ground_state_mse_loss_pointwise_dataset",
    ):
        assert not hasattr(training, name)
