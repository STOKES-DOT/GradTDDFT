from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from td_graddft.neural_d import (
    DispersionCorrectedFunctional,
    DispersionFunctional,
    GradDFTDispersionNetwork,
    build_dispersion_pair_inputs,
    calculate_distances,
    make_neural_d_functional,
)
from td_graddft.scf.molecules import QuadratureGrid
from td_graddft.training import predict_ground_state_total_energy


@dataclass(frozen=True)
class PairMolecule:
    atom_coords: jnp.ndarray
    atom_charges: jnp.ndarray


@dataclass(frozen=True)
class MinimalEnergyMolecule:
    grid: QuadratureGrid
    h1e: jnp.ndarray
    rep_tensor: jnp.ndarray
    rdm1: jnp.ndarray
    nuclear_repulsion: float
    atom_coords: jnp.ndarray
    atom_charges: jnp.ndarray


def constant_dispersion(_instance, x, *_, **__):
    return jnp.ones((x.shape[0],), dtype=x.dtype)


def test_calculate_distances_matches_graddft_ordered_nonself_pairs():
    coords = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    charges = jnp.asarray([1, 8])

    distances, atom_pairs = calculate_distances(coords, charges)

    assert distances.shape == (2, 1)
    assert jnp.allclose(distances[:, 0], jnp.asarray([2.0, 2.0]))
    assert jnp.array_equal(atom_pairs, jnp.asarray([[1, 8], [8, 1]]))


def test_calculate_distances_is_jittable_for_fixed_atom_count():
    coords = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    charges = jnp.asarray([1, 8])

    distances, atom_pairs = jax.jit(calculate_distances)(coords, charges)

    assert jnp.allclose(distances[:, 0], jnp.asarray([2.0, 2.0]))
    assert jnp.array_equal(atom_pairs, jnp.asarray([[1, 8], [8, 1]]))


def test_build_dispersion_pair_inputs_returns_graddft_features():
    molecule = PairMolecule(
        atom_coords=jnp.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ),
        atom_charges=jnp.asarray([1, 8]),
    )

    inputs = build_dispersion_pair_inputs(molecule, 4)

    assert inputs.shape == (2, 4)
    assert jnp.allclose(
        inputs,
        jnp.asarray(
            [
                [2.0, 1.0, 8.0, 4.0],
                [2.0, 8.0, 1.0, 4.0],
            ]
        ),
    )


def test_graddft_dispersion_network_has_predefined_output_shape():
    network = GradDFTDispersionNetwork(hidden_dims=(8, 8), sigmoid_scale_factor=2.0)
    sample_inputs = jnp.asarray(
        [
            [2.0, 1.0, 8.0, 3.0],
            [2.0, 8.0, 1.0, 5.0],
        ]
    )

    variables = network.init(jax.random.PRNGKey(0), sample_inputs)
    coefficients = network.apply(variables, sample_inputs)

    assert coefficients.shape == (2,)
    assert jnp.all(jnp.isfinite(coefficients))
    assert jnp.all(coefficients >= 0.0)


def test_make_neural_d_functional_initializes_default_network():
    molecule = PairMolecule(
        atom_coords=jnp.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ),
        atom_charges=jnp.asarray([1, 8]),
    )
    functional = make_neural_d_functional(hidden_dims=(8, 8))

    variables = functional.init_from_molecule(jax.random.PRNGKey(0), molecule)
    energy = functional.energy(variables, molecule)

    assert "params" in variables
    assert energy.shape == ()
    assert jnp.isfinite(energy)


def test_dispersion_functional_uses_graddft_tail_sum():
    molecule = PairMolecule(
        atom_coords=jnp.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ),
        atom_charges=jnp.asarray([1, 8]),
    )
    functional = DispersionFunctional(dispersion=constant_dispersion)

    energy = functional.energy({}, molecule)

    expected = -(2.0**-6 + 2.0**-8 + 2.0**-10)
    assert jnp.allclose(energy, expected)


def test_dispersion_corrected_functional_adds_to_existing_total_energy_path():
    class ConstantXC:
        def energy_from_molecule(self, params, _molecule):
            return jnp.asarray(params["xc"], dtype=jnp.float64)

    molecule = MinimalEnergyMolecule(
        grid=QuadratureGrid(weights=jnp.asarray([1.0])),
        h1e=jnp.zeros((1, 1)),
        rep_tensor=jnp.zeros((1, 1, 1, 1)),
        rdm1=jnp.zeros((1, 1)),
        nuclear_repulsion=0.0,
        atom_coords=jnp.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ),
        atom_charges=jnp.asarray([1, 8]),
    )
    dispersion = DispersionFunctional(dispersion=constant_dispersion)
    corrected = DispersionCorrectedFunctional(ConstantXC(), dispersion)

    energy = predict_ground_state_total_energy(
        {"xc": jnp.asarray(0.25), "dispersion": {}},
        corrected,
        molecule,
    )

    expected_dispersion = -(2.0**-6 + 2.0**-8 + 2.0**-10)
    assert jnp.allclose(energy, 0.25 + expected_dispersion)
