from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_plan_tool():
    path = Path("tools/plan_species_extrapolation_s1_only.py")
    spec = importlib.util.spec_from_file_location("plan_species_extrapolation_s1_only", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_s1_species_plan_uses_only_s1_loss_for_training():
    module = _load_plan_tool()

    args = module.parse_args(
        [
            "--molecule-reference-csv",
            "molecule_s1.csv",
            "--atom-reference-csv",
            "atom_s1.csv",
            "--out-root",
            "outputs/species",
        ]
    )
    commands = module.build_commands(args)
    train_commands = [cmd for cmd in commands if "closed_shell_s1_self_consistent_train.py" in cmd]

    assert len(train_commands) == 2
    for command in train_commands:
        assert "conda run -n jax_scf python -u" in command
        assert "--basis def2-svp" in command
        assert "--grids-level 2" in command
        assert "--s1-weight 1.0" in command
        assert "--energy-mse-weight 0.0" in command
        assert "--energy-mae-weight 0.0" in command
        assert "--density-constraint-weight 0.0" in command
        assert "--skip-final-evaluation" not in command


def test_remote_launcher_pins_jax_scf_def2svp_grid2():
    path = Path("tools/launch_species_extrapolation_s1_only_remote.sh")
    text = path.read_text(encoding="utf-8")

    assert 'ROOT="${TDGRADDFT_ROOT:-/home/yjiao/TD-GradDFT}"' in text
    assert 'ENV_NAME="${TDGRADDFT_ENV_NAME:-jax_scf}"' in text
    assert 'BASIS="${TDGRADDFT_BASIS:-def2-svp}"' in text
    assert 'GRIDS_LEVEL="${TDGRADDFT_GRIDS_LEVEL:-2}"' in text
    assert 'conda run -n "$ENV_NAME"' in text
    assert "--energy-mse-weight 0.0" in text
    assert "--energy-mae-weight 0.0" in text
    assert "--density-constraint-weight 0.0" in text
