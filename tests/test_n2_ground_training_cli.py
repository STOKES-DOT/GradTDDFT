from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_training_tool():
    path = Path("tools/n2_ccsdt_ground_train5.py")
    spec = importlib.util.spec_from_file_location("n2_ccsdt_ground_train5", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_n2_training_accepts_avas_casscf_reference_options():
    module = _load_training_tool()

    args = module.parse_args(
        [
            "--reference-method",
            "casscf",
            "--active-space",
            "avas",
            "--active-labels",
            "N 2s",
            "N 2p",
            "--ncas",
            "8",
            "--nelecas",
            "10",
        ]
    )

    assert args.reference_method == "casscf"
    assert args.active_space == "avas"
    assert args.active_labels == ["N 2s", "N 2p"]
    assert args.ncas == 8
    assert args.nelecas == 10
    assert "ref=casscf" in module._cache_group_name(args)
    assert "active=avas" in module._cache_group_name(args)


def test_n2_training_accepts_graddft_dissociation_reference_options(tmp_path):
    module = _load_training_tool()
    path = tmp_path / "N2_dissociation.xlsx"

    args = module.parse_args(
        [
            "--reference-method",
            "graddft_data",
            "--graddft-dissociation-xlsx",
            str(path),
            "--r-min",
            "0.8",
            "--r-max",
            "3.0",
            "--train-points",
            "5",
            "--r-values",
            "0.8",
            "1.1",
            "1.6",
            "2.2",
            "3.0",
            "--basis",
            "def2-tzvp",
        ]
    )

    assert args.reference_method == "graddft_data"
    assert args.graddft_dissociation_xlsx == str(path)
    assert args.basis == "def2-tzvp"
    assert args.r_values == [0.8, 1.1, 1.6, 2.2, 3.0]
    assert module._reference_label(args) == "GradDFT MR-ccCA"
    assert "ref=graddft_data" in module._cache_group_name(args)


def test_n2_training_accepts_prediction_scf_stability_options():
    module = _load_training_tool()

    args = module.parse_args(
        [
            "--prediction-scf-stability-check",
            "--prediction-scf-stability-tol-ev",
            "0.02",
            "--prediction-scf-stability-init-modes",
            "cached",
            "rdm1",
            "--prediction-scf-stability-selections",
            "best_rms",
            "final",
            "--fail-on-unstable-prediction-scf",
        ]
    )

    assert args.prediction_scf_stability_check is True
    assert args.prediction_scf_stability_tol_ev == 0.02
    assert args.prediction_scf_stability_init_modes == ["cached", "rdm1"]
    assert args.prediction_scf_stability_selections == ["best_rms", "final"]
    assert args.fail_on_unstable_prediction_scf is True


def test_n2_training_accepts_eval_only_checkpoint_option():
    module = _load_training_tool()

    args = module.parse_args(
        [
            "--eval-only-checkpoint",
            "outputs/run/neural_xc_params.msgpack",
            "--reference-method",
            "graddft_data",
        ]
    )

    assert args.eval_only_checkpoint == "outputs/run/neural_xc_params.msgpack"
    assert args.reference_method == "graddft_data"


def test_prediction_rows_to_eval_history_tracks_per_point_errors():
    module = _load_training_tool()

    history = module.prediction_rows_to_eval_history(
        [
            {
                "r_angstrom": 1.1,
                "reference_energy_h": -109.3,
                "predicted_energy_h": -109.31,
                "energy_abs_err_ev": module.HARTREE_TO_EV * 0.01,
            },
            {
                "r_angstrom": 2.2,
                "reference_energy_h": -109.2,
                "predicted_energy_h": -109.18,
                "energy_abs_err_ev": module.HARTREE_TO_EV * 0.02,
            },
        ]
    )

    assert len(history) == 1
    assert history[0]["step"] == 0
    assert history[0]["energy_mae_h"] == pytest.approx(0.015)
    assert history[0]["loss"] == pytest.approx((0.01**2 + 0.02**2) / 2.0)
    assert history[0]["energy_signed_err_h_r_1p1"] == pytest.approx(-0.01)
    assert history[0]["energy_signed_err_h_r_2p2"] == pytest.approx(0.02)


def test_eval_only_main_uses_streaming_path_before_bulk_cache(tmp_path, monkeypatch):
    module = _load_training_tool()
    captured = {}

    def fake_streaming(**kwargs):
        captured.update(kwargs)
        return {"eval_only": True}

    monkeypatch.setattr(module, "run_eval_only_checkpoint_streaming", fake_streaming)
    monkeypatch.setattr(
        module,
        "_has_hdf5_group",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("bulk cache used")),
    )

    result = module.main(
        [
            "--eval-only-checkpoint",
            "params.msgpack",
            "--reference-cache",
            str(tmp_path / "refs.h5"),
            "--r-values",
            "0.8",
            "1.1",
            "--outdir",
            str(tmp_path / "out"),
        ]
    )

    assert result == {"eval_only": True}
    assert captured["args"].eval_only_checkpoint == "params.msgpack"
    assert list(captured["r_values"]) == [0.8, 1.1]


def test_write_prediction_csv_records_unstable_scf_guard(tmp_path, monkeypatch):
    module = _load_training_tool()

    point = module.ReferencePoint(
        r_angstrom=2.2,
        molecule=object(),
        reference_energy_h=-109.244614,
        mean_field_energy_h=-109.067404,
        correlation_energy_h=-0.177210,
        perturbative_corr_h=0.0,
        reference_method="graddft_data",
        target_density_matrix=None,
    )

    monkeypatch.setattr(
        module,
        "predict_ground_state_total_energy",
        lambda *args, **kwargs: -109.24242632111536,
    )

    def fake_stability_evaluator(*args, **kwargs):
        return {
            "scf_stable": 0,
            "scf_energy_spread_ev": 0.508,
            "scf_min_energy_h": -109.2441367175033,
            "scf_max_energy_h": -109.22548270802383,
            "scf_converged_all": 1,
            "scf_max_cycles": 336,
            "scf_max_selected_rms_density": 5.035e-05,
        }

    rows = module.write_prediction_csv(
        tmp_path / "predictions.csv",
        [point],
        params={},
        functional=object(),
        training_config=module.GroundStateTrainingConfig(),
        stability_evaluator=fake_stability_evaluator,
    )

    assert rows[0]["scf_stable"] == 0
    assert rows[0]["scf_energy_spread_ev"] == 0.508
    assert rows[0]["scf_converged_all"] == 1
    assert "scf_energy_spread_ev" in (tmp_path / "predictions.csv").read_text()


def test_load_graddft_dissociation_targets_requires_exact_grid(tmp_path):
    module = _load_training_tool()
    path = tmp_path / "N2_dissociation.csv"
    path.write_text(
        "R,energy (Ha)\n"
        "0.8,-1.0\n"
        "1.0,-2.0\n"
        "1.2,-3.0\n",
        encoding="utf-8",
    )

    targets = module.load_graddft_dissociation_targets(
        path,
        r_values=[0.8, 1.0, 1.2],
    )

    assert targets == {
        0.8: -1.0,
        1.0: -2.0,
        1.2: -3.0,
    }

    with pytest.raises(ValueError, match="No exact GradDFT target"):
        module.load_graddft_dissociation_targets(path, r_values=[0.9])
