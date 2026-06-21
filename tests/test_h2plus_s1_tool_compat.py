from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


def _load_tool_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_h2plus_s1_tool_defaults_to_legacy_s1_only_tda_objective():
    module = _load_tool_module(
        "tools/h2plus_s1_tda_train5_dense100.py",
        "h2plus_s1_tda_train5_dense100_test_default_objective",
    )

    args = module.parse_args([])

    assert args.objective == "auto"
    assert module._resolved_objective_kind(args) == "s1_only"
    assert module._objective_name(args) == "s1_only_tda"
    assert args.reference_excited_method == "tda"


def test_h2plus_s1_tool_joint_objective_backfills_ground_supervision():
    module = _load_tool_module(
        "tools/h2plus_s1_tda_train5_dense100.py",
        "h2plus_s1_tda_train5_dense100_test_joint_objective",
    )

    args = module.parse_args(["--objective", "joint"])

    assert args.s1_weight == 1.0
    assert args.energy_mse_weight == 0.0
    assert args.energy_mae_weight == 1.0
    assert not hasattr(args, "density_matrix_constraint_weight")
    assert module._resolved_objective_kind(args) == "joint"
    assert module._objective_name(args) == "joint_tda"


def test_h2plus_s1_training_data_uses_density_matrix_target():
    module = _load_tool_module(
        "tools/h2plus_s1_tda_train5_dense100.py",
        "h2plus_s1_tda_train5_dense100_test_density_target",
    )
    point = SimpleNamespace(
        molecule=SimpleNamespace(),
        exact_energy_h=-0.5,
        exact_excitation_energies_h=np.asarray([0.1], dtype=np.float64),
        exact_density_grid=np.asarray([0.2, 0.3], dtype=np.float64),
        exact_dm_ao=np.asarray([[1.0]], dtype=np.float64),
    )

    (datum,) = module.build_training_data(
        [point],
        s1_weight=1.0,
        density_constraint_weight=0.4,
        density_matrix_constraint_weight=0.0,
    )

    assert datum.target_density is None
    assert np.allclose(np.asarray(datum.target_density_matrix), point.exact_dm_ao)
    assert datum.density_constraint_weight == 0.4
    assert datum.density_matrix_constraint_weight == 0.0


def test_h2plus_s1_script_keeps_density_matrix_target_internal():
    source = Path("tools/h2plus_s1_tda_train5_dense100.py").read_text(encoding="utf-8")

    assert "target_density_matrix" in source


def test_h2plus_ground_training_data_uses_density_matrix_target():
    module = _load_tool_module(
        "tools/h2plus_fci_ground_train5_dense100.py",
        "h2plus_fci_ground_train5_dense100_test_density_target",
    )
    args = module.parse_args([])
    point = SimpleNamespace(
        molecule=SimpleNamespace(),
        exact_energy_h=-0.5,
        exact_density_grid=np.asarray([0.2, 0.3], dtype=np.float64),
        exact_density_matrix=np.asarray([[1.0]], dtype=np.float64),
    )

    (datum,) = module.build_training_data([point], density_constraint_weight=0.4)

    assert not hasattr(args, "density_matrix_constraint_weight")
    assert datum.target_density is None
    assert np.allclose(np.asarray(datum.target_density_matrix), point.exact_density_matrix)
    assert datum.density_constraint_weight == 0.4
    assert datum.density_matrix_constraint_weight == 0.0


def test_h2plus_ground_script_does_not_emit_dm_constraint_artifacts():
    source = Path("tools/h2plus_fci_ground_train5_dense100.py").read_text(encoding="utf-8")

    for marker in (
        "--density-matrix-constraint-weight",
        "density_matrix_constraint_weight",
        "density_matrix_penalty",
        "density_matrix_mse",
        "AO DM MSE",
        "dm_mse=",
    ):
        assert marker not in source


def test_h2plus_ground_cache_validation_rejects_stale_hcore():
    pytest.importorskip("pyscf")
    from pyscf import gto

    module = _load_tool_module(
        "tools/h2plus_fci_ground_train5_dense100.py",
        "h2plus_fci_ground_train5_dense100_test_cache_validation",
    )
    atom = module.build_h2plus_atom(1.2)
    mol = gto.M(atom=atom, unit="Angstrom", basis="sto-3g", charge=1, spin=1, cart=True, verbose=0)
    hcore = np.asarray(mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc"))
    overlap = np.asarray(mol.intor_symmetric("int1e_ovlp"))
    args = SimpleNamespace(basis="sto-3g")

    def make_point(h1e: np.ndarray):
        return module.ReferencePoint(
            r_angstrom=1.2,
            atom=atom,
            molecule=SimpleNamespace(
                h1e=h1e,
                overlap_matrix=overlap,
                nuclear_repulsion=float(mol.energy_nuc()),
                atom_coords=np.asarray(mol.atom_coords()),
            ),
            exact_energy_h=-0.5,
            exact_total_energies_h=np.asarray([-0.5]),
            exact_density_grid=np.asarray([1.0]),
            exact_density_matrix=np.asarray([[1.0]]),
            exact_electron_count=1.0,
            reference_backend="test",
            reference_converged=True,
        )

    valid, reason = module._validate_cached_reference_point(make_point(hcore), args)
    assert valid, reason

    stale_hcore = hcore.copy()
    stale_hcore[0, 0] -= 1e-3
    valid, reason = module._validate_cached_reference_point(make_point(stale_hcore), args)
    assert not valid
    assert "hcore mismatch" in reason


def test_h2plus_s1_tool_exposes_skip_eval_switches():
    module = _load_tool_module(
        "tools/h2plus_s1_tda_train5_dense100.py",
        "h2plus_s1_tda_train5_dense100_test_skip_eval_switches",
    )

    args = module.parse_args(["--skip-initial-eval", "--skip-final-evaluation"])

    assert args.skip_initial_eval is True
    assert args.skip_final_evaluation is True


def test_h2plus_s1_tool_exposes_pt2_cli_switches():
    module = _load_tool_module(
        "tools/h2plus_s1_tda_train5_dense100.py",
        "h2plus_s1_tda_train5_dense100_test_pt2_cli_switches",
    )

    args = module.parse_args(
        [
            "--include-pt2-channel",
            "--pt2-channel-mode",
            "local_exact",
            "--response-pt2-mode",
            "strict",
        ]
    )

    assert args.include_pt2_channel is True
    assert args.pt2_channel_mode == "local_exact"
    assert args.response_pt2_mode == "strict"
    assert module._response_pt2_mode_label(args) == "strict"


def test_h2plus_reference_builder_requests_pt2_features_when_pt2_channel_enabled(monkeypatch):
    module = _load_tool_module(
        "tools/h2plus_s1_tda_train5_dense100.py",
        "h2plus_s1_tda_train5_dense100_test_reference_pt2",
    )
    captured: list[dict[str, object]] = []

    def fake_solve_h2plus_with_pyscf(*args, **kwargs):
        return (
            -0.5,
            np.asarray([-0.5, 0.1], dtype=np.float64),
            np.asarray([0.6], dtype=np.float64),
            np.asarray([[1.0]], dtype=np.float64),
            "cpu",
            True,
        )

    def fake_unrestricted_molecule_from_spec_with_jax_uks(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(
            ao=np.asarray([[1.0], [1.0]], dtype=np.float64),
            grid=SimpleNamespace(weights=np.asarray([0.5, 0.5], dtype=np.float64)),
        )

    monkeypatch.setattr(module, "solve_h2plus_with_pyscf", fake_solve_h2plus_with_pyscf)
    monkeypatch.setattr(
        module,
        "unrestricted_molecule_from_spec_with_jax_uks",
        fake_unrestricted_molecule_from_spec_with_jax_uks,
    )
    args = SimpleNamespace(
        basis="sto-3g",
        xc="b3lyp",
        grids_level=0,
        max_l=3,
        integral_backend="cpu",
        reference_scf_max_cycle=80,
        reference_scf_conv_tol=1e-10,
        reference_scf_conv_tol_density=1e-8,
        reference_scf_damping=0.15,
        reference_scf_potential_clip=20.0,
        reference_excited_method="orbital",
        reference_excited_xc="",
        reference_scf_device="cpu",
        nroots=4,
        input_feature_mode="canonical",
        include_pt2_channel=True,
    )

    point = module.build_reference_point(1.2, args=args)

    assert captured[0]["compute_local_hfx_features"] is True
    assert captured[0]["compute_local_hfx_aux"] is True
    assert captured[0]["compute_local_pt2_features"] is True
    assert point.reference_backend == "cpu"


def test_h2plus_reference_cache_key_tracks_pt2_feature_toggle():
    module = _load_tool_module(
        "tools/h2plus_s1_tda_train5_dense100.py",
        "h2plus_s1_tda_train5_dense100_test_cache_key_pt2",
    )

    args_no_pt2 = module.parse_args([])
    args_with_pt2 = module.parse_args(["--include-pt2-channel"])

    key_no_pt2 = module._reference_cache_key(1.4, args_no_pt2)
    key_with_pt2 = module._reference_cache_key(1.4, args_with_pt2)

    assert "pt2=off" in key_no_pt2
    assert "pt2=on" in key_with_pt2
    assert key_no_pt2 != key_with_pt2
