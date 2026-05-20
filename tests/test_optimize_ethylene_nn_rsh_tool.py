from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "optimize_ethylene_nn_rsh.py"
_SPEC = importlib.util.spec_from_file_location("optimize_ethylene_nn_rsh", _TOOL_PATH)
assert _SPEC is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_parse_args_accepts_fixed_density_rsh_training_mode():
    args = _MODULE.parse_args(["--rsh-training-mode", "fixed_density"])

    assert args.rsh_training_mode == "fixed_density"

