import jax
import jax.numpy as jnp
import pytest

from td_graddft.nn_rsh import (
    AtomCenteredDensityDescriptorConfig,
    RSH,
    RSHGNNHead,
    make_atom_centered_density_descriptor_fn,
    make_gnn_rsh_functional,
)
from pyscf_reference import restricted_reference_from_pyscf


def _small_gnn_head() -> RSHGNNHead:
    return RSHGNNHead(
        node_hidden_dims=(8,),
        num_interaction_blocks=1,
        num_heads=2,
        qkv_features=8,
        ffn_dim=16,
        global_hidden_dims=(8,),
    )


def _make_water_reference():
    pytest.importorskip("pyscf")
    from pyscf import dft, gto

    mol = gto.Mole()
    mol.atom = """
    O  0.000000  0.000000  0.117790
    H  0.000000  0.755453 -0.471161
    H  0.000000 -0.755453 -0.471161
    """
    mol.unit = "Angstrom"
    mol.basis = "sto-3g"
    mol.spin = 0
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = "pbe"
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.kernel()
    return restricted_reference_from_pyscf(
        mf,
        compute_local_hfx_features=True,
        compute_local_hfx_aux=True,
        hfx_omega_values=(0.0, 0.3, 0.6),
    )


def test_rsh_gnn_head_returns_three_raw_parameters():
    head = _small_gnn_head()
    atom_features = jnp.arange(15, dtype=jnp.float32).reshape(1, 3, 5) / 10.0
    atom_coords = jnp.asarray(
        [[[0.0, 0.0, 0.0], [0.0, 1.4, 0.0], [1.1, 0.0, 0.0]]],
        dtype=jnp.float32,
    )

    params = head.init(jax.random.PRNGKey(0), atom_features, atom_coords)
    raw = head.apply(params, atom_features, atom_coords)

    assert raw.shape == (1, 3)
    assert jnp.all(jnp.isfinite(raw))


def test_rsh_gnn_head_is_permutation_invariant_for_molecular_output():
    head = _small_gnn_head()
    atom_features = jnp.asarray(
        [[[0.2, 0.0, 0.4, 0.1], [0.1, 0.3, 0.2, 0.0], [0.8, 0.1, 0.0, 0.5]]],
        dtype=jnp.float32,
    )
    atom_coords = jnp.asarray(
        [[[0.0, 0.0, 0.0], [0.0, 1.4, 0.0], [1.1, 0.0, 0.0]]],
        dtype=jnp.float32,
    )
    params = head.init(jax.random.PRNGKey(1), atom_features, atom_coords)

    raw = head.apply(params, atom_features, atom_coords)
    permutation = jnp.asarray([2, 0, 1], dtype=jnp.int32)
    permuted_raw = head.apply(
        params,
        atom_features[:, permutation, :],
        atom_coords[:, permutation, :],
    )

    assert jnp.allclose(permuted_raw, raw, atol=1e-5)


def test_atom_centered_density_descriptor_includes_coordinates():
    molecule = _make_water_reference()
    descriptor_fn = make_atom_centered_density_descriptor_fn(
        AtomCenteredDensityDescriptorConfig(
            radial_centers=(0.6, 1.4),
            radial_width=0.5,
            max_angular=1,
        )
    )

    descriptor = descriptor_fn(molecule)

    assert set(descriptor) >= {"atom_descriptors", "atom_charges", "atom_coords"}
    assert descriptor["atom_coords"].shape == (3, 3)
    assert descriptor["atom_coords"].dtype == jnp.float32


def test_make_gnn_rsh_functional_initializes_from_water_descriptor():
    molecule = _make_water_reference()
    functional = make_gnn_rsh_functional(
        descriptor_config=AtomCenteredDensityDescriptorConfig(
            radial_centers=(0.6, 1.4),
            radial_width=0.5,
            max_angular=1,
        ),
        node_hidden_dims=(8,),
        num_interaction_blocks=1,
        num_heads=2,
        qkv_features=8,
        ffn_dim=16,
        global_hidden_dims=(8,),
        fallback_omega_values=(0.0, 0.3, 0.6),
    )

    params = functional.init_from_molecule(jax.random.PRNGKey(2), molecule)
    resolved = functional.resolve_parameters(params, molecule)

    assert 0.0 <= float(resolved.sr_hf_fraction) <= float(resolved.lr_hf_fraction) <= 1.0
    assert 0.05 <= float(resolved.omega) <= 0.70
    assert jnp.all(jnp.isfinite(functional._raw_outputs(params, molecule)))


def test_rsh_public_trainable_uses_atom_centered_gnn_workflow():
    molecule = _make_water_reference()
    functional = RSH("lc-wpbe").trainable(
        descriptor_config=AtomCenteredDensityDescriptorConfig(
            radial_centers=(0.6, 1.4),
            radial_width=0.5,
            max_angular=1,
        ),
        node_hidden_dims=(8,),
        global_hidden_dims=(8,),
        num_heads=2,
        num_layers=1,
        qkv_features=8,
        ffn_dim=16,
        fallback_omega_values=(0.0, 0.3, 0.6),
    )

    params = functional.init_from_molecule(jax.random.PRNGKey(7), molecule)
    descriptor = functional.descriptor_fn(molecule)
    resolved = functional.resolve_parameters(params, molecule)

    assert functional.head_type == "gnn"
    assert set(descriptor) >= {"atom_descriptors", "atom_coords", "atom_charges"}
    assert descriptor["atom_descriptors"].shape[0] == molecule.atom_coords.shape[0]
    assert resolved.sr_hf_fraction.shape == ()
    assert resolved.lr_hf_fraction.shape == ()
    assert resolved.omega.shape == ()


def test_hse06_public_trainable_initializes_on_water_descriptor():
    molecule = _make_water_reference()
    functional = RSH("hse06").trainable(
        descriptor_config=AtomCenteredDensityDescriptorConfig(
            radial_centers=(0.6, 1.4),
            radial_width=0.5,
            max_angular=1,
        ),
        node_hidden_dims=(8,),
        global_hidden_dims=(8,),
        num_heads=2,
        num_layers=1,
        qkv_features=8,
        ffn_dim=16,
        fallback_omega_values=(0.0, 0.3, 0.6),
    )

    params = functional.init_from_molecule(jax.random.PRNGKey(9), molecule)
    resolved = functional.resolve_parameters(params, molecule)

    assert functional.head_type == "gnn"
    assert functional.local_term_specs
    assert float(resolved.sr_hf_fraction) == pytest.approx(0.25, abs=1e-6)
    assert float(resolved.lr_hf_fraction) == pytest.approx(0.0, abs=1e-6)
