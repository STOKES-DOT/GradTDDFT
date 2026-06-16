from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


_TOOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "search_water_lc_wpbe_omega_td_graddft.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "search_water_lc_wpbe_omega_td_graddft",
    _TOOL_PATH,
)
assert _SPEC is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
_bounded_omega_search = _MODULE._bounded_omega_search


def test_bounded_omega_search_finds_scalar_minimum_and_records_history():
    result, history = _bounded_omega_search(
        lambda omega: (omega - 0.23) ** 2 + 0.1,
        bounds=(0.13, 0.30),
        xatol=1e-4,
        maxiter=12,
    )

    assert result.success
    assert np.isclose(result.x, 0.23, atol=1e-3)
    assert np.isclose(result.fun, 0.1, atol=1e-5)
    assert len(history) >= 3
    assert all("omega" in row and "loss" in row for row in history)
