from __future__ import annotations

import csv
import json
import importlib.util
from pathlib import Path

import jax.numpy as jnp
import pytest

from td_graddft.data.graddft_dataset import GradDFTGroundAtomRecord
from td_graddft.scf.molecules import QuadratureGrid, RestrictedMolecule
from td_graddft.training import GroundStateCoreDatum, GroundStateDatum


def _load_module():
    path = Path("tools/graddft_atoms_ground_train.py")
    spec = importlib.util.spec_from_file_location("graddft_atoms_ground_train", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _minimal_restricted_molecule(*, mf_energy: float = -1.0) -> RestrictedMolecule:
    return RestrictedMolecule(
        ao=jnp.ones((1, 1)),
        grid=QuadratureGrid(weights=jnp.ones((1,)), coords=jnp.zeros((1, 3))),
        dipole_integrals=jnp.zeros((3, 1, 1)),
        rep_tensor=jnp.zeros((0, 0, 0, 0)),
        mo_coeff=jnp.ones((2, 1, 1)),
        mo_occ=jnp.ones((2, 1)),
        mo_energy=jnp.zeros((2, 1)),
        rdm1=jnp.ones((2, 1, 1)) * 0.5,
        h1e=jnp.zeros((1, 1)),
        nuclear_repulsion=0.0,
        atom_coords=jnp.zeros((1, 3)),
        atom_charges=jnp.ones((1,)),
        overlap_matrix=jnp.eye(1),
        ao_deriv1=jnp.zeros((4, 1, 1)),
        mf_energy=mf_energy,
        exact_exchange_fraction=0.2,
        nocc=1,
        hfx_omega_values=jnp.asarray((0.0, 0.4)),
        hfx_local=jnp.zeros((2, 1, 2)),
        hfx_nu=jnp.zeros((2, 1, 1, 1)),
        eri_pair_matrix=jnp.ones((1, 1)),
    )


def test_graddft_atoms_ground_train_cli_defaults_to_test_train_ratio_2_to_8():
    module = _load_module()

    args = module.parse_args([])

    assert args.test_train_ratio == "2:8"
    assert args.xlsx.endswith("XND_dataset.xlsx")
    assert args.prediction_csv == "graddft_atoms_ground_predictions.csv"
    assert args.eval_prediction_csv == "graddft_atoms_ground_eval_predictions.csv"
    assert args.reference_builder == "pyscf"
    assert args.reference_cache == ""
    assert args.rebuild_reference_cache is False
    assert args.hfx_nu_storage == "array"
    assert args.checkpoint_every == 50


def test_graddft_atoms_ground_train_cli_accepts_remote_smoke_overrides():
    module = _load_module()

    args = module.parse_args(
        [
            "--steps",
            "3",
            "--symbols",
            "H,He",
            "--basis",
            "sto-3g",
            "--integral-backend",
            "jax",
            "--outdir",
            "outputs/demo",
        ]
    )

    assert args.steps == 3
    assert args.symbols == "H,He"
    assert args.basis == "sto-3g"
    assert args.integral_backend == "jax"
    assert args.outdir == "outputs/demo"


def test_persist_training_history_writes_csv_and_json(tmp_path):
    module = _load_module()
    history = [
        {"step": 0, "train_loss": 1.25, "test_loss": 2.5},
        {"step": 1, "train_loss": 0.75, "test_loss": float("nan")},
    ]

    module.persist_training_history(tmp_path, history)

    with (tmp_path / "training_history.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["step"] for row in rows] == ["0", "1"]
    assert rows[0]["train_loss"] == "1.25"

    saved_json = json.loads((tmp_path / "training_history.json").read_text(encoding="utf-8"))
    assert saved_json[0]["test_loss"] == 2.5


def test_history_row_includes_scf_convergence_fields():
    module = _load_module()
    train_eval = module._empty_eval_metrics()
    test_eval = module._empty_eval_metrics()
    train_eval.update(
        {
            "loss": 1.0,
            "energy_mae_h": 0.2,
            "normalized_energy_mae": 0.1,
            "scf_converged_fraction": 1.0,
            "scf_cycles_mean": 3.0,
            "scf_cycles_max": 4.0,
            "scf_selected_rms_max": 1e-9,
        }
    )
    test_eval.update(
        {
            "loss": 2.0,
            "energy_mae_h": 0.4,
            "normalized_energy_mae": 0.2,
            "scf_converged_fraction": 0.5,
            "scf_cycles_mean": 5.0,
            "scf_cycles_max": 6.0,
            "scf_selected_rms_max": 1e-6,
        }
    )

    row = module._history_row(
        step=5,
        batch_loss=0.3,
        batch_energy_mae_h=0.03,
        train_eval=train_eval,
        test_eval=test_eval,
        grad_norm=1.5,
        param_update_norm=0.05,
        lr=1e-3,
        batch_scf_converged_fraction=0.75,
        batch_scf_cycles_mean=7.0,
        batch_scf_selected_rms_max=1e-5,
    )

    assert row["train_scf_converged_fraction"] == 1.0
    assert row["test_scf_converged_fraction"] == 0.5
    assert row["batch_scf_converged_fraction"] == 0.75
    assert row["train_scf_selected_rms_max"] == 1e-9
    assert row["test_scf_selected_rms_max"] == 1e-6


def test_write_rows_csv_appends_with_single_header(tmp_path):
    module = _load_module()
    path = tmp_path / "eval_predictions.csv"

    module._write_rows_csv(path, [{"step": 0, "symbol": "H", "scf_converged": 1.0}])
    module._write_rows_csv(path, [{"step": 1, "symbol": "He", "scf_converged": 0.0}])

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["step"] for row in rows] == ["0", "1"]
    assert [row["scf_converged"] for row in rows] == ["1.0", "0.0"]
    assert path.read_text(encoding="utf-8").count("step,symbol,scf_converged") == 1


def test_build_atom_data_reuses_reference_cache(tmp_path, monkeypatch):
    pytest.importorskip("h5py")
    module = _load_module()
    record = GradDFTGroundAtomRecord(
        symbol="Be",
        split="train",
        target_energy_h=-14.667,
        spin=0,
    )
    cache_path = tmp_path / "refs.h5"
    build_calls = []

    def fake_build(record_arg, *, basis, **kwargs):
        build_calls.append((record_arg.symbol, basis, kwargs))
        return GroundStateDatum.from_parts(
            _minimal_restricted_molecule(mf_energy=-14.6723),
            core=GroundStateCoreDatum(target_total_energy=jnp.asarray(record_arg.target_energy_h)),
        )

    monkeypatch.setattr(module, "build_graddft_ground_atom_datum", fake_build)
    logger = module.RunLogger(tmp_path / "run.log")

    first = module._build_atom_data(
        (record,),
        split="train",
        basis="def2-tzvp",
        molecule_kwargs={"reference_builder": "pyscf", "xc_spec": "b3lyp", "grids_level": 2},
        logger=logger,
        reference_cache_path=cache_path,
        rebuild_reference_cache=False,
        hfx_nu_storage="array",
    )
    second = module._build_atom_data(
        (record,),
        split="train",
        basis="def2-tzvp",
        molecule_kwargs={"reference_builder": "pyscf", "xc_spec": "b3lyp", "grids_level": 2},
        logger=logger,
        reference_cache_path=cache_path,
        rebuild_reference_cache=False,
        hfx_nu_storage="array",
    )

    assert len(first) == 1
    assert len(second) == 1
    assert len(build_calls) == 1
    assert second[0].molecule.mf_energy == -14.6723
    assert float(jnp.asarray(second[0].target_total_energy)) == record.target_energy_h
