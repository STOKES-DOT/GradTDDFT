from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "overfit_water_nn_rsh.py"
_SPEC = importlib.util.spec_from_file_location("overfit_water_nn_rsh", _TOOL_PATH)
assert _SPEC is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_free_alpha_keeps_lc_wpbe_short_range_hf_bound_tight():
    args = _MODULE.parse_args(["--free-alpha"])

    template = _MODULE._rsh_template_from_args(args)

    assert template.name == "lc-wpbe"
    assert template.sr_hf_bounds == (0.0, 0.20)
    assert template.supports_trainable_sr_hf is True
