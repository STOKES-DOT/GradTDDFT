from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_tool():
    path = Path("tools/qh9_s1_no_pt2_diagnostics.py")
    spec = importlib.util.spec_from_file_location("qh9_s1_no_pt2_diagnostics", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_qh9_no_pt2_diagnostics_parse_args():
    module = _load_tool()

    args = module.parse_args(
        [
            "--reference-csv",
            "refs.csv",
            "--outdir",
            "out",
            "--nstates",
            "4",
        ]
    )

    assert args.reference_csv == "refs.csv"
    assert args.outdir == "out"
    assert args.nstates == 4


def test_qh9_no_pt2_metric_ignores_nonfinite_values():
    module = _load_tool()

    metrics = module.metric([1.0, -2.0, float("nan")])

    assert metrics["count"] == 2
    assert metrics["mae_ev"] == 1.5
