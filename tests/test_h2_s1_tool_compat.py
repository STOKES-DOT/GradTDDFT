from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


def _load_tool_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_h2_s1_tool_imports_with_current_public_api():
    module = _load_tool_module(
        "tools/h2_s1_tda_train5_dense100_vs_fci.py",
        "h2_s1_tda_train5_dense100_vs_fci_test_import",
    )

    assert module.DEFAULT_INPUT_FEATURE_MODE == "canonical"
    assert module.DEFAULT_NETWORK_ARCHITECTURE == "graddft_residual"


def test_h2_ground_tool_normalizes_legacy_cli_aliases():
    module = _load_tool_module(
        "tools/h2_self_consistent_ground_train5_dense100_vs_fci.py",
        "h2_self_consistent_ground_train5_dense100_vs_fci_test_import",
    )

    args = module.parse_args(
        [
            "--input-feature-mode",
            "dm21_original",
            "--scf-gradient-mode",
            "implicit_commutator",
        ]
    )

    assert args.input_feature_mode == "canonical"
    assert args.scf_gradient_mode == "impl"


def test_h2_reference_builder_requests_pt2_features_when_pt2_channel_enabled(monkeypatch):
    module = _load_tool_module(
        "tools/h2_self_consistent_ground_train5_dense100_vs_fci.py",
        "h2_self_consistent_ground_train5_dense100_vs_fci_test_pt2",
    )
    captured: list[dict[str, object]] = []

    def fake_build_reference_point(r_angstrom, **kwargs):
        captured.append(kwargs)
        point = SimpleNamespace(
            r_angstrom=float(r_angstrom),
            fci_energy_h=-1.0,
            fci_excitation_energies_h=np.asarray([0.5]),
            molecule=SimpleNamespace(grid=SimpleNamespace(weights=np.ones(1))),
        )
        return point, None

    monkeypatch.setattr(module, "build_reference_point", fake_build_reference_point)
    args = SimpleNamespace(
        basis="sto-3g",
        xc="b3lyp",
        grids_level=0,
        max_l=3,
        grid_ao_backend="jax",
        integral_backend="cpu",
        jk_backend="full",
        df_tol=1e-10,
        df_max_rank=None,
        reference_scf_max_cycle=80,
        reference_scf_conv_tol=1e-10,
        reference_scf_conv_tol_density=1e-8,
        reference_scf_damping=0.15,
        reference_scf_potential_clip=20.0,
        excited_nstates=3,
        input_feature_mode="canonical",
        include_pt2_channel=True,
    )
    logger = SimpleNamespace(log=lambda message: None)

    module.build_reference_curve(np.asarray([0.74]), args=args, logger=logger, label="test")

    assert captured[0]["compute_local_hfx_features"] is True
    assert captured[0]["compute_local_pt2_features"] is True
