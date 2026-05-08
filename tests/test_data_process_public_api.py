from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp


@dataclass(frozen=True)
class _Grid:
    weights: jnp.ndarray


@dataclass(frozen=True)
class _Molecule:
    ao: jnp.ndarray
    ao_deriv1: jnp.ndarray
    rdm1: jnp.ndarray
    mo_coeff: jnp.ndarray
    mo_occ: jnp.ndarray
    grid: _Grid


def _molecule() -> _Molecule:
    ao = jnp.asarray([[1.0], [2.0]])
    ao_deriv1 = jnp.asarray(
        [
            [[1.0], [2.0]],
            [[0.1], [0.2]],
            [[0.0], [0.0]],
            [[0.0], [0.0]],
        ]
    )
    return _Molecule(
        ao=ao,
        ao_deriv1=ao_deriv1,
        rdm1=jnp.asarray([[[0.5]], [[0.5]]]),
        mo_coeff=jnp.asarray([[[1.0]], [[1.0]]]),
        mo_occ=jnp.asarray([[0.5], [0.5]]),
        grid=_Grid(weights=jnp.asarray([0.7, 0.9])),
    )


class _Functional:
    def compute_coefficient_inputs(self, molecule, *, features):
        del molecule
        return jnp.stack([features.rho, features.tau_a + features.tau_b], axis=-1)

    def compute_densities(self, molecule, *, features):
        del molecule
        return features.rho[:, None]


def test_prepare_neural_xc_input_from_reference_builds_grid_features():
    from td_graddft import data_process

    molecule = _molecule()
    prepared = data_process.prepare_neural_xc_input(molecule)

    assert prepared.molecule is molecule
    assert prepared.features.rho.shape == (2,)
    assert jnp.allclose(prepared.grid_weights, molecule.grid.weights)
    assert prepared.coefficient_inputs is None
    assert prepared.density_channels is None
    assert prepared.target_total_energy is None


def test_prepare_neural_xc_input_from_datum_carries_targets():
    from td_graddft import data_process
    from td_graddft.training import GroundStateDatum

    datum = GroundStateDatum(
        molecule=_molecule(),
        target_total_energy=jnp.asarray(-1.25),
        density_constraint_weight=0.5,
    )

    prepared = data_process.prepare_neural_xc_input(datum)

    assert prepared.molecule is datum.molecule
    assert jnp.allclose(prepared.target_total_energy, jnp.asarray(-1.25))
    assert prepared.density_constraint_weight == 0.5


def test_prepare_neural_xc_input_uses_functional_input_builders():
    from td_graddft import data_process

    prepared = data_process.prepare_neural_xc_input(
        _molecule(),
        functional=_Functional(),
    )

    assert prepared.coefficient_inputs.shape == (2, 2)
    assert prepared.density_channels.shape == (2, 1)


def test_data_process_module_is_neural_input_only():
    source = Path("src/td_graddft/data_process.py").read_text()

    forbidden = (
        "reference_legacy",
        "pyscf",
        "prepare_reference",
        "restricted_reference_from_spec",
        "restricted_reference_from_pyscf",
        "run_rks",
        "dft.RKS",
        "mf.kernel",
    )
    assert [token for token in forbidden if token in source] == []


def test_top_level_exposes_data_process_namespace():
    import importlib
    import td_graddft

    assert "data_process" in td_graddft.__all__
    assert td_graddft.data_process is importlib.import_module("td_graddft.data_process")
