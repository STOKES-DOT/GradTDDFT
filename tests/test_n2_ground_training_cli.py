from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


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
