from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

import td_graddft.features as features_module
import td_graddft.tddft.casida as casida_module
import td_graddft.tddft.response as response_module
from td_graddft.tddft import RestrictedCasidaTDDFT, build_restricted_response_matrices
from td_graddft.tddft.response import build_restricted_tda_operator
from td_graddft.xc import AdiabaticDensityFunctional, lda_from_callable


@dataclass
class _Grid:
    weights: jnp.ndarray


@dataclass
class _ToyMolecule:
    ao: jnp.ndarray
    ao_deriv1: jnp.ndarray
    grid: _Grid
    rep_tensor: jnp.ndarray
    mo_coeff: jnp.ndarray
    mo_occ: jnp.ndarray
    mo_energy: jnp.ndarray
    rdm1: jnp.ndarray

    def density(self):
        return jnp.einsum("spq,rp,rq->rs", self.rdm1, self.ao, self.ao)


def _make_toy_molecule(rep_tensor=None):
    ao = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    ao_deriv1 = jnp.array(
        [
            ao,
            [[0.20, 0.10], [0.10, 0.30]],
            [[0.05, 0.00], [0.00, 0.04]],
            [[0.00, 0.07], [0.08, 0.00]],
        ]
    )
    mo_coeff = jnp.stack([jnp.eye(2), jnp.eye(2)], axis=0)
    mo_occ = jnp.array([[1.0, 0.0], [1.0, 0.0]])
    mo_energy = jnp.array([[0.0, 1.0], [0.0, 1.0]])
    rdm1 = jnp.array(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0]],
        ]
    )
    return _ToyMolecule(
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid=_Grid(weights=jnp.array([1.0, 1.0])),
        rep_tensor=(
            jnp.zeros((2, 2, 2, 2)) if rep_tensor is None else jnp.asarray(rep_tensor)
        ),
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        rdm1=rdm1,
    )


def _make_large_diagonal_toy_molecule(nocc=10, nvir=11):
    nmo = nocc + nvir
    ao = jnp.eye(nmo)
    ao_deriv1 = jnp.stack([ao, jnp.zeros_like(ao), jnp.zeros_like(ao), jnp.zeros_like(ao)])
    mo_coeff = jnp.stack([jnp.eye(nmo), jnp.eye(nmo)], axis=0)
    mo_occ_single = jnp.concatenate([jnp.ones((nocc,)), jnp.zeros((nvir,))])
    mo_occ = jnp.stack([mo_occ_single, mo_occ_single], axis=0)
    mo_energy_single = jnp.linspace(-1.0, 3.0, nmo)
    mo_energy = jnp.stack([mo_energy_single, mo_energy_single], axis=0)
    rdm1_single = jnp.diag(mo_occ_single)
    rdm1 = jnp.stack([rdm1_single, rdm1_single], axis=0)
    return _ToyMolecule(
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid=_Grid(weights=jnp.ones((nmo,))),
        rep_tensor=jnp.zeros((nmo, nmo, nmo, nmo)),
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        rdm1=rdm1,
    )


def test_rep_tensor_to_mo_eri_slices_matches_explicit_contractions():
    rep_tensor = jnp.arange(4**4, dtype=jnp.float64).reshape(4, 4, 4, 4) / 100.0
    orbo = jnp.array(
        [
            [1.0, 0.1],
            [0.2, 0.9],
            [0.3, 0.0],
            [0.0, 0.4],
        ]
    )
    orbv = jnp.array(
        [
            [0.0, 0.5],
            [0.3, 0.0],
            [0.8, 0.2],
            [0.1, 1.0],
        ]
    )

    eri_ovov, eri_ovvo, eri_oovv = response_module._rep_tensor_to_mo_eri_slices(
        rep_tensor,
        orbo,
        orbv,
        need_ovvo=True,
        include_oovv=True,
    )
    _, eri_ovvo_skipped, eri_oovv_skipped = response_module._rep_tensor_to_mo_eri_slices(
        rep_tensor,
        orbo,
        orbv,
        need_ovvo=False,
        include_oovv=False,
    )

    expected_ovov = jnp.einsum(
        "pqrs,pi,qa,rj,sb->iajb",
        rep_tensor,
        orbo,
        orbv,
        orbo,
        orbv,
    )
    expected_ovvo = jnp.einsum(
        "pqrs,pi,qa,rb,sj->iabj",
        rep_tensor,
        orbo,
        orbv,
        orbv,
        orbo,
    )
    expected_oovv = jnp.einsum(
        "pqrs,pi,qj,ra,sb->ijab",
        rep_tensor,
        orbo,
        orbo,
        orbv,
        orbv,
    )

    assert jnp.allclose(eri_ovov, expected_ovov, atol=1e-10)
    assert jnp.allclose(eri_ovvo, expected_ovvo, atol=1e-10)
    assert jnp.allclose(eri_oovv, expected_oovv, atol=1e-10)
    assert eri_ovvo_skipped is None
    assert eri_oovv_skipped is None


def test_restricted_response_operator_precomputes_effective_eri_actions():
    rep_tensor = jnp.arange(4**4, dtype=jnp.float64).reshape(4, 4, 4, 4) / 50.0
    nmo = 4
    ao = jnp.eye(nmo)
    mo_coeff = jnp.stack([jnp.eye(nmo), jnp.eye(nmo)], axis=0)
    mo_occ_single = jnp.array([1.0, 1.0, 0.0, 0.0])
    mo_occ = jnp.stack([mo_occ_single, mo_occ_single], axis=0)
    mo_energy_single = jnp.array([-0.8, -0.2, 0.5, 1.1])
    mo_energy = jnp.stack([mo_energy_single, mo_energy_single], axis=0)
    rdm1_single = jnp.diag(mo_occ_single)
    molecule = _ToyMolecule(
        ao=ao,
        ao_deriv1=jnp.stack([ao, jnp.zeros_like(ao), jnp.zeros_like(ao), jnp.zeros_like(ao)]),
        grid=_Grid(weights=jnp.ones((nmo,))),
        rep_tensor=rep_tensor,
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        rdm1=jnp.stack([rdm1_single, rdm1_single], axis=0),
    )
    molecule.nocc = 2
    xc = AdiabaticDensityFunctional(
        name="hybrid_only",
        energy_density_fn=lambda rho: jnp.zeros_like(rho),
        exact_exchange_fraction=0.25,
    )

    data = response_module._build_restricted_response_operator_data(molecule, xc)
    alpha = jnp.asarray(0.25, dtype=data.eri_ovov.dtype)
    expected_tda = 2.0 * data.eri_ovov - alpha * jnp.transpose(data.eri_oovv, (0, 2, 1, 3))
    expected_b = 2.0 * data.eri_ovvo - alpha * jnp.transpose(data.eri_ovvo, (0, 2, 1, 3))

    assert data.effective_tda_eri is not None
    assert data.effective_b_eri is not None
    assert jnp.allclose(data.effective_tda_eri, expected_tda, atol=1e-10)
    assert jnp.allclose(data.effective_b_eri, expected_b, atol=1e-10)


def test_packed_eri_pair_matrix_response_matches_full_tensor_path():
    nmo = 4
    npair = nmo * (nmo + 1) // 2
    pair_values = jnp.arange(npair * npair, dtype=jnp.float64).reshape(npair, npair) / 100.0
    pair_values = 0.5 * (pair_values + pair_values.T)
    rows, cols = jnp.tril_indices(nmo)
    pair_index = jnp.zeros((nmo, nmo), dtype=jnp.int32)
    pair_ids = jnp.arange(npair, dtype=jnp.int32)
    pair_index = pair_index.at[rows, cols].set(pair_ids)
    pair_index = pair_index.at[cols, rows].set(pair_ids)
    ao = jnp.arange(nmo, dtype=jnp.int32)
    rep_tensor = pair_values[
        pair_index[ao[:, None, None, None], ao[None, :, None, None]],
        pair_index[ao[None, None, :, None], ao[None, None, None, :]],
    ]
    mo_coeff = jnp.stack([jnp.eye(nmo), jnp.eye(nmo)], axis=0)
    mo_occ_single = jnp.array([1.0, 1.0, 0.0, 0.0])
    mo_occ = jnp.stack([mo_occ_single, mo_occ_single], axis=0)
    mo_energy_single = jnp.array([-0.8, -0.2, 0.5, 1.1])
    mo_energy = jnp.stack([mo_energy_single, mo_energy_single], axis=0)
    rdm1_single = jnp.diag(mo_occ_single)
    base_molecule = _ToyMolecule(
        ao=jnp.eye(nmo),
        ao_deriv1=jnp.stack(
            [jnp.eye(nmo), jnp.zeros((nmo, nmo)), jnp.zeros((nmo, nmo)), jnp.zeros((nmo, nmo))]
        ),
        grid=_Grid(weights=jnp.ones((nmo,))),
        rep_tensor=rep_tensor,
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        rdm1=jnp.stack([rdm1_single, rdm1_single], axis=0),
    )
    packed_molecule = _ToyMolecule(
        ao=base_molecule.ao,
        ao_deriv1=base_molecule.ao_deriv1,
        grid=base_molecule.grid,
        rep_tensor=jnp.zeros((0, 0, 0, 0)),
        mo_coeff=base_molecule.mo_coeff,
        mo_occ=base_molecule.mo_occ,
        mo_energy=base_molecule.mo_energy,
        rdm1=base_molecule.rdm1,
    )
    packed_molecule.eri_pair_matrix = pair_values
    base_molecule.nocc = 2
    packed_molecule.nocc = 2
    xc = AdiabaticDensityFunctional(
        name="hybrid_only",
        energy_density_fn=lambda rho: jnp.zeros_like(rho),
        exact_exchange_fraction=0.25,
    )

    full = build_restricted_response_matrices(base_molecule, xc)
    packed = build_restricted_response_matrices(packed_molecule, xc)

    assert jnp.allclose(packed.a_matrix, full.a_matrix, atol=1e-10)
    assert jnp.allclose(packed.b_matrix, full.b_matrix, atol=1e-10)


def test_matrix_free_tdhf_matches_materialized_matrix_for_multi_virtual_hybrid():
    rep_tensor = jnp.arange(4**4, dtype=jnp.float64).reshape(4, 4, 4, 4) / 50.0
    nmo = 4
    ao = jnp.eye(nmo)
    mo_coeff = jnp.stack([jnp.eye(nmo), jnp.eye(nmo)], axis=0)
    mo_occ_single = jnp.array([1.0, 1.0, 0.0, 0.0])
    mo_occ = jnp.stack([mo_occ_single, mo_occ_single], axis=0)
    mo_energy_single = jnp.array([-0.8, -0.2, 0.5, 1.1])
    mo_energy = jnp.stack([mo_energy_single, mo_energy_single], axis=0)
    rdm1_single = jnp.diag(mo_occ_single)
    molecule = _ToyMolecule(
        ao=ao,
        ao_deriv1=jnp.stack([ao, jnp.zeros_like(ao), jnp.zeros_like(ao), jnp.zeros_like(ao)]),
        grid=_Grid(weights=jnp.ones((nmo,))),
        rep_tensor=rep_tensor,
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        rdm1=jnp.stack([rdm1_single, rdm1_single], axis=0),
    )
    molecule.nocc = 2
    xc = AdiabaticDensityFunctional(
        name="hybrid_only",
        energy_density_fn=lambda rho: jnp.zeros_like(rho),
        exact_exchange_fraction=0.25,
    )

    vind_dense, flat_a, flat_b = response_module.build_restricted_tdhf_operator(
        molecule,
        xc,
        materialize_matrix=True,
    )
    vind_free, flat_a_free, flat_b_free = response_module.build_restricted_tdhf_operator(
        molecule,
        xc,
        materialize_matrix=False,
    )
    z = jnp.array(
        [
            [0.1, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, -0.8],
            [-0.2, 0.3, 0.6, -0.1, 0.4, 0.9, -0.5, 0.8],
        ]
    )

    x = z[:, :4]
    y = z[:, 4:]
    expected = jnp.concatenate(
        [
            x @ flat_a.T + y @ flat_b.T,
            -(x @ flat_b.T + y @ flat_a.T),
        ],
        axis=-1,
    )

    assert flat_a is not None and flat_b is not None
    assert flat_a_free is None and flat_b_free is None
    assert jnp.allclose(vind_dense(z), expected, atol=1e-9)
    assert jnp.allclose(vind_free(z), expected, atol=1e-9)


def test_response_matrices_match_toy_analytic_values():
    molecule = _make_toy_molecule()
    xc = lda_from_callable("toy", lambda rho: 0.5 * rho)

    matrices = build_restricted_response_matrices(molecule, xc)

    assert matrices.a_matrix.shape == (1, 1, 1, 1)
    assert matrices.b_matrix.shape == (1, 1, 1, 1)
    assert jnp.allclose(matrices.a_matrix[0, 0, 0, 0], 2.0)
    assert jnp.allclose(matrices.b_matrix[0, 0, 0, 0], 1.0)


def test_restricted_casida_tddft_returns_expected_toy_excitation():
    molecule = _make_toy_molecule()
    xc = lda_from_callable("toy", lambda rho: 0.5 * rho)
    solver = RestrictedCasidaTDDFT(molecule, xc)

    result = solver.kernel(nstates=1)
    tda = solver.tda(nstates=1)

    assert jnp.allclose(tda.excitation_energies, jnp.array([2.0]))
    assert jnp.allclose(result.excitation_energies, jnp.array([jnp.sqrt(3.0)]))
    assert result.x_amplitudes.shape == (1, 1, 1)
    assert result.y_amplitudes.shape == (1, 1, 1)


def test_matrix_free_tda_vind_matches_materialized_matrix_action():
    molecule = _make_toy_molecule()
    xc = lda_from_callable("toy", lambda rho: 0.5 * rho)
    solver = RestrictedCasidaTDDFT(molecule, xc)
    matrices = solver.build_matrices()
    vind_dense, flat_a = solver.gen_tda_vind(materialize_matrix=True)
    vind_free, flat_a_free = solver.gen_tda_vind(materialize_matrix=False)

    x = jnp.array([[0.3], [1.1]])
    expected = x @ matrices.a_matrix.reshape(1, 1).T
    assert flat_a is not None
    assert flat_a_free is None
    assert jnp.allclose(vind_dense(x), expected, atol=1e-9)
    assert jnp.allclose(vind_free(x), expected, atol=1e-9)


def test_matrix_free_tdhf_vind_matches_materialized_matrix_action_with_global_hybrid():
    rep_tensor = jnp.zeros((2, 2, 2, 2))
    rep_tensor = rep_tensor.at[0, 1, 1, 0].set(0.4)
    rep_tensor = rep_tensor.at[0, 1, 0, 1].set(0.6)
    rep_tensor = rep_tensor.at[0, 0, 1, 1].set(0.5)
    molecule = _make_toy_molecule(rep_tensor=rep_tensor)

    class _HybridXC:
        exact_exchange_fraction = 0.25

        def local_kernel(self, density):
            return jnp.zeros_like(density)

    solver = RestrictedCasidaTDDFT(molecule, _HybridXC())
    matrices = solver.build_matrices()
    vind_dense, flat_a, flat_b = solver.gen_tdhf_vind(materialize_matrix=True)
    vind_free, flat_a_free, flat_b_free = solver.gen_tdhf_vind(materialize_matrix=False)

    z = jnp.array([[0.4, -0.2], [1.3, 0.7]])
    dense_out = jnp.concatenate(
        [
            z[:, :1] @ matrices.a_matrix.reshape(1, 1).T + z[:, 1:] @ matrices.b_matrix.reshape(1, 1).T,
            -(z[:, :1] @ matrices.b_matrix.reshape(1, 1).T + z[:, 1:] @ matrices.a_matrix.reshape(1, 1).T),
        ],
        axis=-1,
    )
    assert flat_a is not None and flat_b is not None
    assert flat_a_free is None and flat_b_free is None
    assert jnp.allclose(vind_dense(z), dense_out, atol=1e-9)
    assert jnp.allclose(vind_free(z), dense_out, atol=1e-9)


def test_large_toy_tda_davidson_uses_operator_path():
    molecule = _make_large_diagonal_toy_molecule()
    dense_solver = RestrictedCasidaTDDFT(molecule, eigensolver="dense")
    davidson_solver = RestrictedCasidaTDDFT(molecule, eigensolver="davidson")

    dense = dense_solver.tda(nstates=4)
    davidson = davidson_solver.tda(nstates=4)

    assert davidson.a_matrix is None
    assert jnp.allclose(davidson.excitation_energies, dense.excitation_energies, atol=1e-8)


def test_large_toy_casida_davidson_uses_operator_path():
    molecule = _make_large_diagonal_toy_molecule()
    dense_solver = RestrictedCasidaTDDFT(molecule, eigensolver="dense")
    davidson_solver = RestrictedCasidaTDDFT(molecule, eigensolver="davidson")

    dense = dense_solver.kernel(nstates=4)
    davidson = davidson_solver.kernel(nstates=4)

    assert davidson.a_matrix is None
    assert davidson.b_matrix is None
    assert davidson.casida_matrix is None
    assert jnp.allclose(davidson.excitation_energies, dense.excitation_energies, atol=1e-8)


def test_jitted_tda_does_not_cache_traced_matrix_before_jitted_kernel():
    molecule = _make_toy_molecule()
    molecule.nocc = 1
    xc = lda_from_callable("toy", lambda rho: 0.5 * rho)
    solver = RestrictedCasidaTDDFT(molecule, xc, eigensolver="dense")

    with jax.checking_leaks():
        tda = jax.jit(lambda: solver.tda(nstates=1))()
    kernel = jax.jit(lambda: solver.kernel(nstates=1))()

    assert jnp.allclose(tda.excitation_energies, jnp.array([2.0]))
    assert jnp.allclose(kernel.excitation_energies, jnp.array([jnp.sqrt(3.0)]))


def test_jitted_strict_gga_tda_does_not_cache_transition_feature_tracer():
    features_module._TRANSITION_RESPONSE_FEATURE_CACHE.clear()
    molecule = _make_toy_molecule()
    molecule.nocc = 1

    class _StrictGGAXC:
        exact_exchange_fraction = 0.0
        response_feature_kind = "GGA"

        def grid_response_tensor(self, mol):
            del mol
            tensor = jnp.zeros((4, 4, 2))
            tensor = tensor.at[0, 0].set(jnp.array([0.4, 0.6]))
            tensor = tensor.at[1, 1].set(jnp.array([0.1, 0.2]))
            tensor = tensor.at[2, 2].set(jnp.array([0.3, 0.4]))
            tensor = tensor.at[3, 3].set(jnp.array([0.5, 0.7]))
            return tensor

    solver = RestrictedCasidaTDDFT(molecule, _StrictGGAXC(), eigensolver="dense")

    with jax.checking_leaks():
        result = jax.jit(lambda: solver.tda(nstates=1))()

    assert result.excitation_energies.shape == (1,)
    assert features_module._TRANSITION_RESPONSE_FEATURE_CACHE == {}


def test_transition_response_feature_cache_is_bounded():
    features_module._TRANSITION_RESPONSE_FEATURE_CACHE.clear()
    molecules = []

    for _ in range(features_module._MAX_TRANSITION_RESPONSE_FEATURE_CACHE_SIZE + 3):
        molecule = _make_toy_molecule()
        molecule.nocc = 1
        molecules.append(molecule)
        features_module.restricted_transition_response_features(molecule, feature_kind="GGA")

    assert (
        len(features_module._TRANSITION_RESPONSE_FEATURE_CACHE)
        == features_module._MAX_TRANSITION_RESPONSE_FEATURE_CACHE_SIZE
    )


def test_transition_response_mgga_pt2_linearized_path_is_removed():
    molecule = _make_toy_molecule()
    molecule.nocc = 1

    with pytest.raises(ValueError, match="PT2 strict response"):
        features_module.restricted_transition_response_features(
            molecule,
            feature_kind="MGGA_PT2",
        )


def test_davidson_tda_falls_back_to_dense_when_operator_solver_fails(monkeypatch):
    molecule = _make_large_diagonal_toy_molecule()
    dense_solver = RestrictedCasidaTDDFT(molecule, eigensolver="dense")
    davidson_solver = RestrictedCasidaTDDFT(molecule, eigensolver="davidson")

    def _fail_operator(*args, **kwargs):
        raise RuntimeError("forced Davidson failure")

    monkeypatch.setattr(casida_module, "solve_tda_from_operator", _fail_operator)
    dense = dense_solver.tda(nstates=4)
    fallback = davidson_solver.tda(nstates=4)

    assert fallback.a_matrix is not None
    assert jnp.allclose(fallback.excitation_energies, dense.excitation_energies, atol=1e-8)


def test_davidson_casida_falls_back_to_dense_when_operator_solver_fails(monkeypatch):
    molecule = _make_large_diagonal_toy_molecule()
    dense_solver = RestrictedCasidaTDDFT(molecule, eigensolver="dense")
    davidson_solver = RestrictedCasidaTDDFT(molecule, eigensolver="davidson")

    def _fail_operator(*args, **kwargs):
        raise RuntimeError("forced Davidson failure")

    monkeypatch.setattr(casida_module, "solve_casida_from_operator", _fail_operator)
    dense = dense_solver.kernel(nstates=4)
    fallback = davidson_solver.kernel(nstates=4)

    assert fallback.a_matrix is not None
    assert fallback.b_matrix is not None
    assert fallback.casida_matrix is not None
    assert jnp.allclose(fallback.excitation_energies, dense.excitation_energies, atol=1e-8)


def test_nonlocal_response_action_contributes_to_dense_and_operator_paths():
    molecule = _make_toy_molecule()

    class _NonlocalXC:
        exact_exchange_fraction = 0.0

        def local_kernel(self, density):
            return jnp.zeros_like(density)

        def local_potential(self, density):
            return jnp.zeros_like(density)

        def nonlocal_response_action(self, mol, amplitudes, *, occupation_tolerance=1e-8):
            del mol, occupation_tolerance
            return 0.5 * jnp.asarray(amplitudes)

        def nonlocal_response_diagonal(self, mol, *, occupation_tolerance=1e-8):
            del mol, occupation_tolerance
            return jnp.asarray([[0.5]])

    matrices = build_restricted_response_matrices(molecule, _NonlocalXC())
    vind, diagonal, _, _ = build_restricted_tda_operator(
        molecule,
        _NonlocalXC(),
        materialize_matrix=False,
    )
    x = jnp.array([[0.25], [1.1]])

    assert jnp.allclose(matrices.a_matrix[0, 0, 0, 0], 1.5, atol=1e-8)
    assert jnp.allclose(matrices.b_matrix[0, 0, 0, 0], 0.5, atol=1e-8)
    assert jnp.allclose(diagonal, jnp.array([1.5]), atol=1e-8)
    assert jnp.allclose(vind(x), x * 1.5, atol=1e-8)


def test_small_dense_tda_skips_operator_builder(monkeypatch):
    molecule = _make_toy_molecule()
    xc = lda_from_callable("toy", lambda rho: 0.5 * rho)
    solver = RestrictedCasidaTDDFT(molecule, xc, eigensolver="dense")

    def _unexpected(*args, **kwargs):
        raise AssertionError("operator path should not run for small dense TDA")

    monkeypatch.setattr(casida_module, "build_restricted_tda_operator", _unexpected)
    result = solver.tda(nstates=1)
    assert jnp.allclose(result.excitation_energies, jnp.array([2.0]))


def test_small_dense_casida_skips_a_minus_b_builder(monkeypatch):
    molecule = _make_toy_molecule()
    xc = lda_from_callable("toy", lambda rho: 0.5 * rho)
    solver = RestrictedCasidaTDDFT(molecule, xc, eigensolver="dense")

    def _unexpected(*args, **kwargs):
        raise AssertionError("operator path should not run for small dense Casida")

    monkeypatch.setattr(casida_module, "build_restricted_a_minus_b_matrix", _unexpected)
    result = solver.kernel(nstates=1)
    assert jnp.allclose(result.excitation_energies, jnp.array([jnp.sqrt(3.0)]))


def test_hybrid_exchange_contributes_to_restricted_response_matrices():
    rep_tensor = jnp.zeros((2, 2, 2, 2))
    rep_tensor = rep_tensor.at[0, 1, 1, 0].set(0.4)
    rep_tensor = rep_tensor.at[0, 1, 0, 1].set(0.6)
    rep_tensor = rep_tensor.at[0, 0, 1, 1].set(0.5)
    molecule = _make_toy_molecule(rep_tensor=rep_tensor)
    xc = AdiabaticDensityFunctional(
        name="hybrid_only",
        energy_density_fn=lambda rho: jnp.zeros_like(rho),
        exact_exchange_fraction=0.25,
    )

    matrices = build_restricted_response_matrices(molecule, xc)

    # For i=j=0, a=b=1 with alpha=0.25:
    # A = dE + 2(ia|jb) - alpha(ij|ab) = 1 + 2*0.6 - 0.25*0.5 = 2.075
    # B = 2(ia|bj) - alpha(ib|aj) = 2*0.4 - 0.25*0.4 = 0.7
    assert jnp.allclose(matrices.a_matrix[0, 0, 0, 0], 2.075, atol=1e-6)
    assert jnp.allclose(matrices.b_matrix[0, 0, 0, 0], 0.7, atol=1e-6)


def test_spatially_varying_local_hf_fraction_is_rejected_in_strict_response():
    rep_tensor = jnp.zeros((2, 2, 2, 2))
    rep_tensor = rep_tensor.at[0, 1, 1, 0].set(0.4)
    rep_tensor = rep_tensor.at[0, 1, 0, 1].set(0.6)
    rep_tensor = rep_tensor.at[0, 0, 1, 1].set(0.5)
    molecule = _make_toy_molecule(rep_tensor=rep_tensor)

    class _LocalHybridXC:
        exact_exchange_fraction = 0.0

        def local_kernel(self, density):
            return jnp.zeros_like(density)

        def grid_hf_fraction(self, mol):
            del mol
            return jnp.array([0.2, 0.8])

    with pytest.raises(ValueError, match="Spatially varying local HF fractions"):
        build_restricted_response_matrices(molecule, _LocalHybridXC())


def test_response_kernel_rejects_nonfinite_grid_values():
    molecule = _make_toy_molecule()

    class _NaNKernelXC:
        exact_exchange_fraction = 0.0

        def local_kernel(self, density):
            del density
            return jnp.array(jnp.nan)

    with pytest.raises(ValueError, match="non-finite values"):
        build_restricted_response_matrices(molecule, _NaNKernelXC())


def test_scalar_grid_hf_fraction_is_broadcast_in_response():
    rep_tensor = jnp.zeros((2, 2, 2, 2))
    rep_tensor = rep_tensor.at[0, 1, 1, 0].set(0.4)
    rep_tensor = rep_tensor.at[0, 1, 0, 1].set(0.6)
    rep_tensor = rep_tensor.at[0, 0, 1, 1].set(0.5)
    molecule = _make_toy_molecule(rep_tensor=rep_tensor)

    class _ScalarLocalHybridXC:
        exact_exchange_fraction = 0.0

        def local_kernel(self, density):
            return jnp.zeros_like(density)

        def grid_hf_fraction(self, mol):
            del mol
            return jnp.asarray(0.25)

    scalar_local = build_restricted_response_matrices(molecule, _ScalarLocalHybridXC())
    hybrid_ref = build_restricted_response_matrices(
        molecule,
        AdiabaticDensityFunctional(
            name="hybrid_ref",
            energy_density_fn=lambda rho: jnp.zeros_like(rho),
            exact_exchange_fraction=0.25,
        ),
    )
    assert jnp.allclose(scalar_local.a_matrix, hybrid_ref.a_matrix, atol=1e-9)
    assert jnp.allclose(scalar_local.b_matrix, hybrid_ref.b_matrix, atol=1e-9)


def test_gga_without_strict_response_tensor_is_rejected():
    molecule = _make_toy_molecule()

    class _ApproximateGGAXC:
        exact_exchange_fraction = 0.0
        response_feature_kind = "GGA"

        def local_kernel(self, density):
            return jnp.ones_like(density)

    with pytest.raises(ValueError, match="requires grid_response_tensor"):
        build_restricted_response_matrices(molecule, _ApproximateGGAXC())


def test_strict_gga_response_tensor_contracts_gradient_channels():
    molecule = _make_toy_molecule()

    class _StrictGGAXC:
        exact_exchange_fraction = 0.0
        response_feature_kind = "GGA"

        def grid_response_tensor(self, mol):
            del mol
            tensor = jnp.zeros((4, 4, 2))
            tensor = tensor.at[0, 0].set(jnp.array([0.4, 0.6]))
            tensor = tensor.at[1, 1].set(jnp.array([0.1, 0.2]))
            tensor = tensor.at[2, 2].set(jnp.array([0.3, 0.4]))
            tensor = tensor.at[3, 3].set(jnp.array([0.5, 0.7]))
            return tensor

    matrices = build_restricted_response_matrices(molecule, _StrictGGAXC())

    ao = molecule.ao_deriv1[:4]
    orbo = molecule.mo_coeff[0][:, :1]
    orbv = molecule.mo_coeff[0][:, 1:]
    rho_o = jnp.einsum("xrp,pi->xri", ao, orbo)
    rho_v = jnp.einsum("xrp,pa->xra", ao, orbv)
    rho_ov = jnp.einsum("xri,ra->xria", rho_o, rho_v[0])
    rho_ov = rho_ov.at[1:4].add(jnp.einsum("ri,xra->xria", rho_o[0], rho_v[1:4]))
    tensor = _StrictGGAXC().grid_response_tensor(molecule)
    xc_expected = 2.0 * jnp.einsum(
        "xyr,xria,yrjb->iajb",
        tensor * molecule.grid.weights[None, None, :],
        rho_ov,
        rho_ov,
    )

    assert jnp.allclose(matrices.a_matrix, 1.0 + xc_expected, atol=1e-8)
    assert jnp.allclose(matrices.b_matrix, xc_expected, atol=1e-8)


def test_strict_mgga_response_tensor_contracts_tau_channel():
    molecule = _make_toy_molecule()

    class _StrictMGGAXC:
        exact_exchange_fraction = 0.0
        response_feature_kind = "MGGA"

        def grid_response_tensor(self, mol):
            del mol
            tensor = jnp.zeros((5, 5, 2))
            tensor = tensor.at[0, 0].set(jnp.array([0.2, 0.3]))
            tensor = tensor.at[4, 4].set(jnp.array([0.8, 1.1]))
            tensor = tensor.at[0, 4].set(jnp.array([0.1, 0.2]))
            tensor = tensor.at[4, 0].set(jnp.array([0.1, 0.2]))
            return tensor

    matrices = build_restricted_response_matrices(molecule, _StrictMGGAXC())

    ao = molecule.ao_deriv1[:4]
    orbo = molecule.mo_coeff[0][:, :1]
    orbv = molecule.mo_coeff[0][:, 1:]
    rho_o = jnp.einsum("xrp,pi->xri", ao, orbo)
    rho_v = jnp.einsum("xrp,pa->xra", ao, orbv)
    rho_ov = jnp.einsum("xri,ra->xria", rho_o, rho_v[0])
    rho_ov = rho_ov.at[1:4].add(jnp.einsum("ri,xra->xria", rho_o[0], rho_v[1:4]))
    tau_ov = 0.5 * jnp.einsum("xri,xra->ria", rho_o[1:4], rho_v[1:4])
    response_features = jnp.concatenate([rho_ov, tau_ov[None, ...]], axis=0)
    tensor = _StrictMGGAXC().grid_response_tensor(molecule)
    xc_expected = 2.0 * jnp.einsum(
        "xyr,xria,yrjb->iajb",
        tensor * molecule.grid.weights[None, None, :],
        response_features,
        response_features,
    )

    assert jnp.allclose(matrices.a_matrix, 1.0 + xc_expected, atol=1e-8)
    assert jnp.allclose(matrices.b_matrix, xc_expected, atol=1e-8)
