from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from td_graddft.training import (
    ExcitedStateFineTuneConfig,
    ExcitedStateFineTuner,
    MolecularTrainingConfig,
    MolecularTrainingDatum,
)
from td_graddft.training.excited_state_trainer import (
    _label_tree_for_trainable_prefixes,
)


def test_fine_tune_config_validates_optimizer_policy():
    with pytest.raises(ValueError, match="steps"):
        ExcitedStateFineTuneConfig(steps=0)
    with pytest.raises(ValueError, match="learning_rate"):
        ExcitedStateFineTuneConfig(learning_rate=0.0)
    with pytest.raises(ValueError, match="select_params"):
        ExcitedStateFineTuneConfig(select_params="unknown")


def test_trainable_prefix_labels_only_selected_parameter_subtree():
    params = {
        "params": {
            "ground": {"kernel": jnp.asarray([1.0])},
            "lr_correction": {"kernel": jnp.asarray([2.0])},
        }
    }

    labels, matched = _label_tree_for_trainable_prefixes(
        params,
        ("lr_correction",),
    )

    assert matched is True
    assert labels["params"]["ground"]["kernel"] == "freeze"
    assert labels["params"]["lr_correction"]["kernel"] == "train"


def test_fine_tuner_reuses_molecular_loss_and_freezes_ground_params(monkeypatch):
    import td_graddft.training.excited_state_trainer as module

    calls = []

    def fake_molecular_loss(params, functional, data, *, training_config):
        del functional, data
        calls.append(training_config)
        value = params["params"]["lr_correction"]["kernel"][0]
        loss = (value - 1.0) ** 2
        return loss, {"total_loss": loss}

    monkeypatch.setattr(module, "molecular_loss", fake_molecular_loss)
    params = {
        "params": {
            "ground": {"kernel": jnp.asarray([4.0])},
            "lr_correction": {"kernel": jnp.asarray([3.0])},
        }
    }
    objective = MolecularTrainingConfig(excitation_gap_mse_weight=1.0)
    tuner = ExcitedStateFineTuner(
        ExcitedStateFineTuneConfig(
            steps=20,
            learning_rate=0.1,
            trainable_path_prefixes=("lr_correction",),
        ),
        objective,
        object(),
        params,
    )

    result = tuner.fine_tune(
        MolecularTrainingDatum(
            molecule=SimpleNamespace(),
            target_excitation_gaps_h=jnp.asarray([0.2]),
        )
    )

    assert calls and all(config is objective for config in calls)
    assert result.best_loss < result.initial_loss
    assert result.params["params"]["ground"]["kernel"][0] == pytest.approx(4.0)
    assert result.params["params"]["lr_correction"]["kernel"][0] < 3.0
