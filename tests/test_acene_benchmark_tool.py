import importlib.util
import sys
from pathlib import Path


_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "benchmark_acene_tddft_gpu_curve.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_acene_tddft_gpu_curve", _TOOL_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TOOL
_SPEC.loader.exec_module(_TOOL)

linear_acene_atoms = _TOOL.linear_acene_atoms
parse_nvidia_smi_sample = _TOOL.parse_nvidia_smi_sample


def test_linear_acene_atom_counts_match_acene_formula():
    expected = {
        1: (6, 6),
        2: (10, 8),
        3: (14, 10),
        4: (18, 12),
        5: (22, 14),
    }

    for rings, (expected_c, expected_h) in expected.items():
        atoms = linear_acene_atoms(rings)
        carbon_count = sum(1 for symbol, _coords in atoms if symbol == "C")
        hydrogen_count = sum(1 for symbol, _coords in atoms if symbol == "H")

        assert carbon_count == expected_c
        assert hydrogen_count == expected_h


def test_parse_nvidia_smi_sample_extracts_utilization_and_memory():
    assert parse_nvidia_smi_sample("87, 24576") == (87.0, 24576.0)
    assert parse_nvidia_smi_sample(" 0 %, 15 MiB ") == (0.0, 15.0)
