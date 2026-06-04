from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_generator_tool():
    path = Path("tools/generate_qh9_eomee_s1_csv.py")
    spec = importlib.util.spec_from_file_location("generate_qh9_eomee_s1_csv", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_qh9_reference_generator_splits_train_validation_test():
    module = _load_generator_tool()

    splits = [
        module._split_for_success_index(idx, train_count=35, validation_count=5)
        for idx in (0, 34, 35, 39, 40, 49)
    ]

    assert splits == ["train", "train", "validation", "validation", "test", "test"]


def test_qh9_reference_generator_parses_excluded_elements():
    module = _load_generator_tool()

    assert module._parse_excluded_atomic_numbers(["F", "9"]) == {9}
    with pytest.raises(ValueError):
        module._parse_excluded_atomic_numbers(["Cl"])


def test_qh9_reference_generator_accepts_size_ordering(monkeypatch):
    module = _load_generator_tool()

    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_qh9_eomee_s1_csv.py", "--sample-order", "size"],
    )
    args = module.parse_args()

    assert args.sample_order == "size"
