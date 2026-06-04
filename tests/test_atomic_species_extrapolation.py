from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_common_tool():
    path = Path("tools/atomic_species_extrapolation_common.py")
    spec = importlib.util.spec_from_file_location("atomic_species_extrapolation_common", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_graddft_ground_atom_split_matches_paper_script_boundary():
    module = _load_common_tool()

    split_by_symbol = {spec.symbol: spec.split for spec in module.graddft_ground_atom_specs()}

    assert len(split_by_symbol) == 36
    assert split_by_symbol["H"] == "train"
    assert split_by_symbol["Ar"] == "train"
    assert split_by_symbol["Sc"] == "train"
    assert split_by_symbol["K"] == "test"
    assert split_by_symbol["Ti"] == "test"
    assert split_by_symbol["Zn"] == "test"
    assert split_by_symbol["Kr"] == "train"


def test_closed_shell_s1_atom_split_is_restricted_and_has_holdouts():
    module = _load_common_tool()

    specs = module.closed_shell_s1_atom_specs()
    split_by_symbol = {spec.symbol: spec.split for spec in specs}

    assert split_by_symbol == {
        "He": "train",
        "Be": "train",
        "Ne": "train",
        "Mg": "train",
        "Ar": "train",
        "Ca": "validation",
        "Zn": "validation",
        "Kr": "test",
    }
    assert all(spec.spin == 0 for spec in specs)
    assert all(spec.charge == 0 for spec in specs)
