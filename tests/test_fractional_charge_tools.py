from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from td_graddft_tools.fractional_charge import (
    FractionalChargeAnalysisConfig,
    FractionalChargeOutputConfig,
    analyze_fractional_charge_linearity,
    run_fractional_charge_workflow,
)


@dataclass
class _DummyFractionalMolecule:
    mo_coeff: jnp.ndarray
    mo_occ: jnp.ndarray
    rdm1: jnp.ndarray
    electron_count: float


def _make_dummy_molecule() -> _DummyFractionalMolecule:
    mo_coeff = jnp.eye(2)
    mo_occ = jnp.array([2.0, 0.0])
    rdm1 = jnp.einsum("pi,i,qi->pq", mo_coeff, mo_occ, mo_coeff)
    return _DummyFractionalMolecule(
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        rdm1=rdm1,
        electron_count=2.0,
    )


def test_fractional_charge_analysis_is_exact_for_linear_energy():
    molecule = _make_dummy_molecule()

    def energy_fn(mol: _DummyFractionalMolecule):
        return -1.25 + 0.75 * jnp.asarray(mol.electron_count)

    result = analyze_fractional_charge_linearity(
        molecule,
        energy_fn,
        FractionalChargeAnalysisConfig(
            charge_min=-1.0,
            charge_max=1.0,
            num_points=9,
        ),
    )

    assert result.charge_deltas.shape == (9,)
    assert np.allclose(np.asarray(result.deviation_ha), 0.0, atol=1e-10, rtol=0.0)
    assert result.max_abs_deviation_ha < 1e-10
    assert np.isclose(result.left_endpoint_slope_ha, 0.75, atol=1e-10, rtol=0.0)
    assert np.isclose(result.right_endpoint_slope_ha, 0.75, atol=1e-10, rtol=0.0)


def test_fractional_charge_workflow_writes_outputs(tmp_path: Path):
    molecule = _make_dummy_molecule()

    def energy_fn(mol: _DummyFractionalMolecule):
        charge_delta = jnp.asarray(mol.electron_count) - 2.0
        return -10.0 + 0.3 * charge_delta + 0.2 * charge_delta**2

    result = run_fractional_charge_workflow(
        molecule,
        energy_fn,
        analysis_config=FractionalChargeAnalysisConfig(
            charge_min=-1.0,
            charge_max=1.0,
            num_points=11,
        ),
        output_config=FractionalChargeOutputConfig(
            outdir=tmp_path,
            prefix="dummy_fractional",
            title="Dummy Fractional Charge",
            energy_unit="ev",
        ),
    )

    assert result.csv_path.exists()
    assert result.png_path.exists()
    assert result.summary_path.exists()
    summary = result.summary_path.read_text(encoding="utf-8")
    assert "max_abs_deviation_ha=" in summary
    assert "left_endpoint_slope_ha=" in summary
