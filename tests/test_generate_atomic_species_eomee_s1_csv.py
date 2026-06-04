from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_generator_tool():
    common_path = Path("tools/atomic_species_extrapolation_common.py")
    common_spec = importlib.util.spec_from_file_location(
        "atomic_species_extrapolation_common",
        common_path,
    )
    assert common_spec is not None
    assert common_spec.loader is not None
    common = importlib.util.module_from_spec(common_spec)
    sys.modules[common_spec.name] = common
    common_spec.loader.exec_module(common)

    path = Path("tools/generate_atomic_species_eomee_s1_csv.py")
    spec = importlib.util.spec_from_file_location("generate_atomic_species_eomee_s1_csv", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_atomic_reference_generator_accepts_closed_shell_s1_preset():
    module = _load_generator_tool()

    args = module.parse_args(["--preset", "closed_shell_s1", "--systems", "He", "Kr"])

    assert args.preset == "closed_shell_s1"
    assert args.systems == ["He", "Kr"]


def test_atomic_reference_generator_writes_training_compatible_csv(tmp_path, monkeypatch):
    module = _load_generator_tool()
    outcsv = tmp_path / "atomic_refs.csv"

    def fake_compute_reference_row(spec, **kwargs):
        return {
            "system": spec.name,
            "split": spec.split,
            "basis": kwargs["basis"],
            "cart": True,
            "charge": spec.charge,
            "spin": spec.spin,
            "unit": spec.unit,
            "atom": spec.atom,
            "reference_ground_method": "RHF/CCSD",
            "reference_excited_method": "EOM-EE-CCSD singlet",
            "rhf_energy_h": -1.0,
            "ccsd_total_energy_h": -1.1,
            "s1_excitation_h": 0.2,
            "s1_excitation_ev": 5.4422772491976,
            "singlet_excitation_energies_h_json": json.dumps([0.2]),
            "singlet_excitation_energies_ev_json": json.dumps([5.4422772491976]),
            "nroots_requested": kwargs["nroots"],
            "scf_elapsed_s": 0.0,
            "ccsd_elapsed_s": 0.0,
            "eom_elapsed_s": 0.0,
            "atomic_number": spec.atomic_number,
            "symbol": spec.symbol,
            "notes": spec.notes or "",
        }

    monkeypatch.setattr(module, "_compute_reference_row", fake_compute_reference_row)

    module.main(
        [
            "--preset",
            "closed_shell_s1",
            "--systems",
            "He",
            "Kr",
            "--basis",
            "sto-3g",
            "--outcsv",
            str(outcsv),
            "--overwrite",
        ]
    )

    with outcsv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["system"] for row in rows] == ["He", "Kr"]
    assert [row["split"] for row in rows] == ["train", "test"]
    assert set(module.TRAINING_COMPATIBLE_FIELDS).issubset(rows[0])
    assert json.loads((tmp_path / "summary.json").read_text())["preset"] == "closed_shell_s1"
