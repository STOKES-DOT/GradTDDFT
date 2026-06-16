from td_graddft.data import integrals
from td_graddft.data.integrals.jax import screening
from td_graddft import features
from td_graddft.scf import builders, differentiable, facade, inputs, molecules, rhf, rks, uks
from td_graddft.scf import core
import jax
import jax.numpy as jnp


def test_scf_modules_share_core_helper_implementations():
    assert rhf._orthogonalizer is core._orthogonalizer
    assert rks._orthogonalizer is core._orthogonalizer
    assert differentiable._orthogonalizer is core._orthogonalizer

    assert rhf._diagonalize_fock is core._diagonalize_fock
    assert rks._diagonalize_fock is core._diagonalize_fock

    assert rhf._build_density is core._build_density_closed_shell
    assert rks._build_density_from_occ is core._build_density_from_occ
    assert uks._build_density_from_occ is core._build_density_from_occ
    assert rhf._build_jk is rks._build_jk
    assert differentiable._build_jk is rks._build_jk

    assert facade._contains_jax_tracer is core._contains_jax_tracer
    assert builders._contains_jax_tracer is core._contains_jax_tracer
    assert inputs._contains_jax_tracer is core._contains_jax_tracer

    assert uks._host_float_unless_traced is core._host_float_unless_traced


def test_facade_no_longer_exposes_cuda_direct_helpers():
    assert not hasattr(facade, "_make_cuda_direct_reference_solver")
    assert not hasattr(builders, "_make_cuda_direct_reference_solver")
    assert not hasattr(facade.RKS, "cuda_direct_scf")


def test_facade_uses_builder_level_reference_and_result_helpers():
    assert facade.build_restricted_reference_from_facade is builders.build_restricted_reference_from_facade
    assert facade.build_restricted_scf_result_from_facade is builders.build_restricted_scf_result_from_facade
    assert (
        facade.build_unrestricted_reference_from_facade
        is builders.build_unrestricted_reference_from_facade
    )


def test_direct_jk_uses_integrals_screening_helper():
    from td_graddft.data.integrals.jax import direct_jk

    assert direct_jk.shell_pair_schwarz_bounds is screening.shell_pair_schwarz_bounds


def test_integral_layer_exports_packed_eri_helpers():
    assert integrals.build_j_from_eri_pair_matrix is not None
    assert integrals.build_jk_from_eri_pair_matrix is not None
    assert integrals.eri_pair_matrix_to_mo_eri_slices is not None


def test_rks_module_does_not_keep_a_second_traceable_iteration_loop():
    assert not hasattr(rks, "_run_scf_iterations_lax_traceable")


def test_rks_module_exposes_one_shared_integrals_entry():
    assert hasattr(rks, "_run_rks_from_integrals_shared")


def test_spin_density_gradient_helper_is_shared():
    assert rks._spin_density_and_gradient is features._spin_density_and_gradient
    assert uks._spin_density_and_gradient is features._spin_density_and_gradient


def test_rks_integral_path_does_not_build_restricted_spin_view(monkeypatch):
    def _fail_restricted_spin_view(**_kwargs):
        raise AssertionError("RKS fock construction should use direct density features")

    monkeypatch.setattr(rks, "_restricted_spin_view", _fail_restricted_spin_view)

    result = rks.run_rks_from_integrals(
        overlap=jnp.eye(1),
        hcore=jnp.zeros((1, 1)),
        eri=jnp.zeros((1, 1, 1, 1)),
        nelectron=2,
        nuclear_repulsion=0.0,
        ao=jnp.ones((1, 1)),
        ao_deriv1=jnp.zeros((4, 1, 1)),
        grid_weights=jnp.ones((1,)),
        config=rks.RKSConfig(xc_spec="hf", max_cycle=1),
    )

    assert result.density_matrix.shape == (1, 1)


def test_rks_direct_density_features_match_restricted_spin_view():
    nao = 2
    ngrids = 3
    ao = jnp.asarray([[1.0, 0.2], [0.4, 1.2], [0.8, -0.3]])
    ao_deriv1 = jnp.ones((4, ngrids, nao)) * 0.25
    weights = jnp.ones((ngrids,))
    density = jnp.asarray([[1.0, 0.1], [0.1, 0.4]])
    mo_coeff = jnp.eye(nao)
    mo_occ = jnp.asarray([2.0, 0.0])
    mo_energy = jnp.asarray([-0.5, 0.2])

    restricted_state = rks._restricted_spin_view(
        ao=ao,
        ao_deriv1=ao_deriv1,
        weights=weights,
        density=density,
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
    )
    restricted_features, restricted_grad = features.restricted_grid_features_with_gradients(
        restricted_state
    )
    direct_rho, direct_grad = rks._spin_density_and_gradient(ao, ao_deriv1, density)

    assert jnp.allclose(direct_rho, restricted_features.rho)
    assert jnp.allclose(direct_grad, restricted_grad)


def test_rks_and_uks_build_molecule_like_pytree_states():
    nao = 2
    ngrids = 3
    ao = jnp.ones((ngrids, nao))
    ao_deriv1 = jnp.ones((4, ngrids, nao))
    weights = jnp.ones((ngrids,))
    density = jnp.eye(nao)
    mo_coeff = jnp.eye(nao)
    mo_occ = jnp.asarray([2.0, 0.0])
    mo_energy = jnp.asarray([-0.5, 0.2])

    restricted_state = rks._restricted_spin_view(
        ao=ao,
        ao_deriv1=ao_deriv1,
        weights=weights,
        density=density,
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
    )
    assert isinstance(restricted_state, features.MoleculeLikeState)
    assert isinstance(restricted_state.grid, features.MoleculeGridView)
    assert len(jax.tree_util.tree_leaves(restricted_state)) > 0

    template = features.MoleculeLikeState(
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid=features.MoleculeGridView(
            weights=weights,
            coords=jnp.zeros((ngrids, 3)),
            points=jnp.zeros((ngrids, 3)),
        ),
        rdm1=jnp.stack([density, density], axis=0),
        mo_coeff=jnp.stack([mo_coeff, mo_coeff], axis=0),
        mo_occ=jnp.stack([jnp.asarray([1.0, 0.0]), jnp.asarray([1.0, 0.0])], axis=0),
        mo_energy=jnp.stack([mo_energy, mo_energy], axis=0),
        ao_laplacian=jnp.zeros_like(ao),
        atom_coords=jnp.zeros((2, 3)),
        atom_charges=jnp.asarray([1.0, 8.0]),
        hfx_omega_values=(0.0, 0.4),
        hfx_nu=jnp.zeros((2, ngrids, nao, nao)),
    )
    unrestricted_state = uks._molecule_like_state_for_bound_xc(
        density_a=density,
        density_b=density,
        mo_coeff_a=mo_coeff,
        mo_coeff_b=mo_coeff,
        mo_occ_a=jnp.asarray([1.0, 0.0]),
        mo_occ_b=jnp.asarray([1.0, 0.0]),
        mo_energy_a=mo_energy,
        mo_energy_b=mo_energy,
        ao=ao,
        ao_deriv1=ao_deriv1,
        weights=weights,
        h=jnp.eye(nao),
        eri=jnp.ones((nao * (nao + 1) // 2, nao * (nao + 1) // 2)),
        overlap=jnp.eye(nao),
        molecule_template=template,
    )
    assert isinstance(unrestricted_state, features.MoleculeLikeState)
    assert isinstance(unrestricted_state.grid, features.MoleculeGridView)
    assert unrestricted_state.grid.coords is not None
    assert unrestricted_state.hfx_nu is not None
    assert len(jax.tree_util.tree_leaves(unrestricted_state)) > 0


def test_rks_iteration_carry_is_a_pytree_dataclass():
    nao = 2
    carry = rks.RKSIterationCarry(
        cycle=jnp.asarray(0, dtype=jnp.int32),
        converged=jnp.asarray(False),
        density=jnp.eye(nao),
        mo_coeff=jnp.eye(nao),
        mo_energy=jnp.asarray([-0.5, 0.2]),
        energy=jnp.asarray(0.0),
        xc_energy=jnp.asarray(0.0),
        raw_fock=jnp.eye(nao),
        j_mat=jnp.eye(nao),
        k_mat=jnp.eye(nao),
        fock_last=jnp.eye(nao),
        fock_hist=jnp.zeros((8, nao, nao)),
        err_hist=jnp.zeros((8, nao * nao)),
        hist_head=jnp.asarray(0, dtype=jnp.int32),
        hist_count=jnp.asarray(0, dtype=jnp.int32),
    )
    leaves = jax.tree_util.tree_leaves(carry)
    assert len(leaves) == 15
    rebuilt = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(carry), leaves)
    assert isinstance(rebuilt, rks.RKSIterationCarry)


def test_restricted_molecule_tree_flatten_keeps_dynamic_field_slots_stable():
    nao = 2
    ngrids = 3
    base_kwargs = dict(
        ao=jnp.ones((ngrids, nao)),
        grid=molecules.QuadratureGrid(weights=jnp.ones((ngrids,)), coords=None),
        dipole_integrals=jnp.zeros((3, nao, nao)),
        rep_tensor=jnp.zeros((nao, nao, nao, nao)),
        mo_coeff=jnp.eye(nao),
        mo_occ=jnp.asarray([2.0, 0.0]),
        mo_energy=jnp.asarray([-0.5, 0.2]),
        rdm1=jnp.eye(nao),
        h1e=jnp.eye(nao),
        nuclear_repulsion=0.7,
    )
    mol_none = molecules.RestrictedMolecule(
        **base_kwargs,
        ao_laplacian=None,
    )
    mol_array = molecules.RestrictedMolecule(
        **base_kwargs,
        ao_laplacian=jnp.zeros((ngrids, nao)),
    )
    children_none, aux_none = mol_none.tree_flatten()
    children_array, aux_array = mol_array.tree_flatten()
    assert len(children_none) == len(children_array)
    assert len(children_none) > 0
    assert len(aux_none) == len(aux_array)


def test_molecule_hfx_nu_api_is_static_pytree_metadata():
    nao = 2
    ngrids = 3
    hfx_api = object()
    restricted = molecules.RestrictedMolecule(
        ao=jnp.ones((ngrids, nao)),
        grid=molecules.QuadratureGrid(weights=jnp.ones((ngrids,)), coords=None),
        dipole_integrals=jnp.zeros((3, nao, nao)),
        rep_tensor=jnp.zeros((nao, nao, nao, nao)),
        mo_coeff=jnp.eye(nao),
        mo_occ=jnp.asarray([2.0, 0.0]),
        mo_energy=jnp.asarray([-0.5, 0.2]),
        rdm1=jnp.eye(nao),
        h1e=jnp.eye(nao),
        nuclear_repulsion=0.7,
        hfx_nu_api=hfx_api,
    )
    unrestricted = molecules.UnrestrictedMolecule(
        ao=jnp.ones((ngrids, nao)),
        grid=molecules.QuadratureGrid(weights=jnp.ones((ngrids,)), coords=None),
        dipole_integrals=jnp.zeros((3, nao, nao)),
        rep_tensor=jnp.zeros((nao, nao, nao, nao)),
        mo_coeff=jnp.stack([jnp.eye(nao), jnp.eye(nao)], axis=0),
        mo_occ=jnp.stack([jnp.asarray([1.0, 0.0]), jnp.asarray([1.0, 0.0])], axis=0),
        mo_energy=jnp.stack([jnp.asarray([-0.5, 0.2]), jnp.asarray([-0.5, 0.2])], axis=0),
        rdm1=jnp.stack([jnp.eye(nao), jnp.eye(nao)], axis=0),
        h1e=jnp.eye(nao),
        nuclear_repulsion=0.7,
        hfx_nu_api=hfx_api,
    )

    restricted_children, restricted_aux = restricted.tree_flatten()
    unrestricted_children, unrestricted_aux = unrestricted.tree_flatten()

    assert all(child is not hfx_api for child in restricted_children)
    assert all(child is not hfx_api for child in unrestricted_children)
    assert any(item is hfx_api for item in restricted_aux)
    assert any(item is hfx_api for item in unrestricted_aux)


def test_molecule_like_state_tree_flatten_keeps_dynamic_field_slots_stable():
    nao = 2
    ngrids = 3
    base_kwargs = dict(
        ao=jnp.ones((ngrids, nao)),
        ao_deriv1=jnp.ones((4, ngrids, nao)),
        grid=features.MoleculeGridView(weights=jnp.ones((ngrids,))),
        rdm1=jnp.stack([jnp.eye(nao), jnp.eye(nao)], axis=0),
        mo_coeff=jnp.stack([jnp.eye(nao), jnp.eye(nao)], axis=0),
        mo_occ=jnp.stack([jnp.asarray([1.0, 0.0]), jnp.asarray([1.0, 0.0])], axis=0),
        mo_energy=jnp.stack([jnp.asarray([-0.5, 0.2]), jnp.asarray([-0.5, 0.2])], axis=0),
    )
    state_none = features.MoleculeLikeState(
        **base_kwargs,
        hfx_omega_values=None,
    )
    state_array = features.MoleculeLikeState(
        **base_kwargs,
        hfx_omega_values=jnp.asarray([0.0, 0.4]),
    )
    children_none, aux_none = state_none.tree_flatten()
    children_array, aux_array = state_array.tree_flatten()
    assert len(children_none) == len(children_array)
    assert len(children_none) > 0
    assert aux_none == aux_array


def test_restricted_molecule_stores_hfx_omega_values_as_array():
    mol = molecules.RestrictedMolecule(
        ao=jnp.ones((1, 1)),
        grid=molecules.QuadratureGrid(weights=jnp.ones((1,))),
        dipole_integrals=jnp.zeros((3, 1, 1)),
        rep_tensor=jnp.zeros((1, 1, 1, 1)),
        mo_coeff=jnp.ones((1, 1)),
        mo_occ=jnp.ones((1,)),
        mo_energy=jnp.ones((1,)),
        rdm1=jnp.ones((1, 1)),
        h1e=jnp.ones((1, 1)),
        nuclear_repulsion=0.0,
        hfx_omega_values=jnp.asarray([0.0, 0.4]),
    )
    assert isinstance(mol.hfx_omega_values, jnp.ndarray)
