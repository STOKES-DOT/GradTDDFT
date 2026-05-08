from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from td_graddft.upstreams import (
    ground_state_from_grad_dft_molecule,
    spin_summed_density_matrix,
)


def test_spin_summed_density_matrix_reduces_spin_axis():
    density = jnp.array(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 2.0]],
        ]
    )
    reduced = spin_summed_density_matrix(density)
    assert jnp.allclose(reduced, jnp.array([[1.0, 0.0], [0.0, 2.0]]))


def test_ground_state_from_grad_dft_molecule_extracts_expected_fields():
    molecule = SimpleNamespace(
        rdm1=jnp.ones((2, 3, 3)),
        s1e=jnp.eye(3),
        fock=2.0 * jnp.eye(3),
        mo_coeff=jnp.eye(3),
        mo_energy=jnp.array([0.1, 0.2, 0.3]),
        mo_occ=jnp.array([2.0, 0.0, 0.0]),
        spin=0,
        charge=0,
        name="H2",
        basis="sto-3g",
    )

    ground_state = ground_state_from_grad_dft_molecule(molecule)

    assert ground_state.density_matrix.shape == (2, 3, 3)
    assert ground_state.overlap_matrix.shape == (3, 3)
    assert ground_state.metadata["name"] == "H2"


def test_ground_state_from_grad_dft_molecule_validates_shape_sources():
    molecule = SimpleNamespace(rdm1=jnp.eye(2))
    with pytest.raises(AttributeError):
        ground_state_from_grad_dft_molecule(molecule)

