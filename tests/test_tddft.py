from collections.abc import Callable
from dataclasses import dataclass, replace
import math

import jax
import jax.numpy as jnp
import pytest

import td_graddft.features as features_module
import td_graddft.tddft.casida as casida_module
import td_graddft.tddft._semilocal_response as semilocal_response_module
import td_graddft.tddft.response as response_module
from td_graddft.tddft import (
    RestrictedCasidaTDDFT,
    UnrestrictedCasidaTDDFT,
)
from td_graddft.tddft.cisd import (
    restricted_cisd_second_order_correction,
    unrestricted_cisd_second_order_correction,
)
from td_graddft.tddft.response import build_restricted_tda_operator
from td_graddft.tddft.types import TDAResult


def _operator_matrix(vind: Callable[[jnp.ndarray], jnp.ndarray], dim: int) -> jnp.ndarray:
    return vind(jnp.eye(dim)).T


def _tdhf_operator_matrices(
    vind: Callable[[jnp.ndarray], jnp.ndarray],
    dim: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    eye = jnp.eye(dim)
    zeros = jnp.zeros_like(eye)
    a_cols = vind(jnp.concatenate([eye, zeros], axis=-1))[:, :dim]
    b_cols = vind(jnp.concatenate([zeros, eye], axis=-1))[:, :dim]
    return a_cols.T, b_cols.T


def _rep_tensor_from_pair_matrix(pair_values: jnp.ndarray) -> jnp.ndarray:
    npair = int(pair_values.shape[0])
    nmo = (math.isqrt(8 * npair + 1) - 1) // 2
    rows, cols = jnp.tril_indices(nmo)
    pair_index = jnp.zeros((nmo, nmo), dtype=jnp.int32)
    pair_ids = jnp.arange(npair, dtype=jnp.int32)
    pair_index = pair_index.at[rows, cols].set(pair_ids)
    pair_index = pair_index.at[cols, rows].set(pair_ids)
    ao_index = jnp.arange(nmo, dtype=jnp.int32)
    return pair_values[
        pair_index[ao_index[:, None, None, None], ao_index[None, :, None, None]],
        pair_index[ao_index[None, None, :, None], ao_index[None, None, None, :]],
    ]


def _symmetric_rep_tensor(nmo: int, scale: float = 100.0) -> jnp.ndarray:
    npair = nmo * (nmo + 1) // 2
    pair_values = jnp.arange(npair * npair, dtype=jnp.float64).reshape(npair, npair) / scale
    pair_values = 0.5 * (pair_values + pair_values.T)
    return _rep_tensor_from_pair_matrix(pair_values)


@dataclass(frozen=True)
class _ToyAdiabaticFunctional:
    name: str
    energy_density_fn: Callable[[jnp.ndarray], jnp.ndarray]
    exact_exchange_fraction: float = 0.0

    def local_kernel(self, density):
        density = jnp.asarray(density)
        flat = density.reshape(-1)

        def local_energy(value):
            return value * self.energy_density_fn(value)

        return jax.vmap(jax.grad(jax.grad(local_energy)))(flat).reshape(density.shape)


def _lda_from_callable(name, energy_density_fn):
    return _ToyAdiabaticFunctional(name=name, energy_density_fn=energy_density_fn)


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


def _make_open_shell_toy_molecule(rep_tensor=None):
    ao = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    ao_deriv1 = jnp.array(
        [
            ao,
            [[0.20, 0.00], [0.00, 0.20]],
            [[0.00, 0.10], [0.10, 0.00]],
            [[0.10, 0.00], [0.00, 0.10]],
        ]
    )
    mo_coeff = jnp.stack([jnp.eye(2), jnp.eye(2)], axis=0)
    mo_occ = jnp.array([[1.0, 0.0], [0.0, 0.0]])
    mo_energy = jnp.array([[0.0, 2.0], [0.2, 2.2]])
    rdm1 = jnp.array(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
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


def test_restricted_response_operator_matches_explicit_mo_block_reference():
    nmo = 4
    npair = nmo * (nmo + 1) // 2
    pair_values = jnp.arange(npair * npair, dtype=jnp.float64).reshape(npair, npair) / 50.0
    pair_values = 0.5 * (pair_values + pair_values.T)
    rows, cols = jnp.tril_indices(nmo)
    pair_index = jnp.zeros((nmo, nmo), dtype=jnp.int32)
    pair_ids = jnp.arange(npair, dtype=jnp.int32)
    pair_index = pair_index.at[rows, cols].set(pair_ids)
    pair_index = pair_index.at[cols, rows].set(pair_ids)
    ao_index = jnp.arange(nmo, dtype=jnp.int32)
    rep_tensor = pair_values[
        pair_index[ao_index[:, None, None, None], ao_index[None, :, None, None]],
        pair_index[ao_index[None, None, :, None], ao_index[None, None, None, :]],
    ]
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
    xc = _ToyAdiabaticFunctional(
        name="hybrid_only",
        energy_density_fn=lambda rho: jnp.zeros_like(rho),
        exact_exchange_fraction=0.25,
    )

    data = response_module._build_restricted_response_operator_data(molecule, xc)
    assert not hasattr(data, "eri_ovov")
    assert data.ao_response_action_fn is not None

    orbo = mo_coeff[0][:, :2]
    orbv = mo_coeff[0][:, 2:]
    eri_ovov = jnp.einsum("pqrs,pi,qa,rj,sb->iajb", rep_tensor, orbo, orbv, orbo, orbv)
    eri_ovvo = jnp.einsum("pqrs,pi,qa,rb,sj->iabj", rep_tensor, orbo, orbv, orbv, orbo)
    eri_oovv = jnp.einsum("pqrs,pi,qj,ra,sb->ijab", rep_tensor, orbo, orbo, orbv, orbv)
    alpha = jnp.asarray(0.25, dtype=rep_tensor.dtype)
    delta_eps = mo_energy_single[2:][None, :] - mo_energy_single[:2, None]
    dim = int(delta_eps.size)
    expected_a = jnp.diag(delta_eps.reshape(-1)) + (
        2.0 * eri_ovov - alpha * jnp.transpose(eri_oovv, (0, 2, 1, 3))
    ).reshape(dim, dim)
    expected_b = (
        2.0 * eri_ovvo - alpha * jnp.transpose(eri_ovvo, (0, 2, 1, 3))
    ).transpose(0, 1, 3, 2).reshape(dim, dim)

    tda_vind, _, _ = build_restricted_tda_operator(molecule, xc)
    tdhf_vind = response_module.build_restricted_tdhf_operator(molecule, xc)
    actual_a, actual_b = _tdhf_operator_matrices(tdhf_vind, dim)

    assert jnp.allclose(_operator_matrix(tda_vind, dim), expected_a, atol=1e-10)
    assert jnp.allclose(actual_a, expected_a, atol=1e-10)
    assert jnp.allclose(actual_b, expected_b, atol=1e-10)


def test_hfx_nu_hybrid_response_uses_standard_ao_exchange(monkeypatch):
    molecule = _make_toy_molecule()
    molecule.hfx_nu = jnp.zeros((1, molecule.ao.shape[0], 2, 2), dtype=molecule.ao.dtype)
    xc = _ToyAdiabaticFunctional(
        name="hfx_nu_hybrid",
        energy_density_fn=lambda rho: jnp.zeros_like(rho),
        exact_exchange_fraction=0.5,
    )
    calls = {"jk": 0}
    original_jk = response_module._jk_from_full_eri

    def _count_combined_jk(*args, **kwargs):
        calls["jk"] += 1
        return original_jk(*args, **kwargs)

    monkeypatch.setattr(response_module, "_jk_from_full_eri", _count_combined_jk)

    vind, diagonal, _ = build_restricted_tda_operator(molecule, xc)
    amplitudes = jnp.ones((1, int(diagonal.size)), dtype=diagonal.dtype)

    assert jnp.all(jnp.isfinite(vind(amplitudes)))
    assert calls["jk"] > 0


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
            [
                jnp.eye(nmo),
                jnp.zeros((nmo, nmo)),
                jnp.zeros((nmo, nmo)),
                jnp.zeros((nmo, nmo)),
            ]
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
    xc = _ToyAdiabaticFunctional(
        name="hybrid_only",
        energy_density_fn=lambda rho: jnp.zeros_like(rho),
        exact_exchange_fraction=0.25,
    )

    full_vind = response_module.build_restricted_tdhf_operator(base_molecule, xc)
    packed_vind = response_module.build_restricted_tdhf_operator(packed_molecule, xc)
    full_a, full_b = _tdhf_operator_matrices(full_vind, 4)
    packed_a, packed_b = _tdhf_operator_matrices(packed_vind, 4)

    assert jnp.allclose(packed_a, full_a, atol=1e-10)
    assert jnp.allclose(packed_b, full_b, atol=1e-10)


def test_restricted_operator_ignores_stale_response_eri_cache():
    nmo = 4
    rep_tensor = jnp.arange(nmo**4, dtype=jnp.float64).reshape(nmo, nmo, nmo, nmo) / 100.0
    c = 1.0 / jnp.sqrt(2.0)
    mo_single = jnp.array(
        [
            [c, 0.0, c, 0.0],
            [0.0, c, 0.0, c],
            [c, 0.0, -c, 0.0],
            [0.0, c, 0.0, -c],
        ],
        dtype=jnp.float64,
    )
    occ_single = jnp.array([1.0, 1.0, 0.0, 0.0])
    molecule = _ToyMolecule(
        ao=jnp.eye(nmo),
        ao_deriv1=jnp.stack(
            [jnp.eye(nmo), jnp.zeros((nmo, nmo)), jnp.zeros((nmo, nmo)), jnp.zeros((nmo, nmo))]
        ),
        grid=_Grid(weights=jnp.ones((nmo,))),
        rep_tensor=rep_tensor,
        mo_coeff=jnp.stack([mo_single, mo_single], axis=0),
        mo_occ=jnp.stack([occ_single, occ_single], axis=0),
        mo_energy=jnp.stack([jnp.array([-1.0, -0.5, 0.3, 0.8])] * 2, axis=0),
        rdm1=jnp.stack([jnp.diag(occ_single), jnp.diag(occ_single)], axis=0),
    )
    molecule.nocc = 2
    molecule.eri_ovov = jnp.zeros((2, 2, 2, 2))
    molecule.eri_ovvo = jnp.zeros((2, 2, 2, 2))
    molecule.eri_oovv = jnp.zeros((2, 2, 2, 2))

    xc = _ToyAdiabaticFunctional(
        name="hybrid_only",
        energy_density_fn=lambda rho: jnp.zeros_like(rho),
        exact_exchange_fraction=0.25,
    )
    cached_vind, cached_diagonal, _ = build_restricted_tda_operator(molecule, xc)
    fresh = replace(molecule)
    for name in ("eri_ovov", "eri_ovvo", "eri_oovv"):
        if hasattr(fresh, name):
            delattr(fresh, name)
    fresh_vind, fresh_diagonal, _ = build_restricted_tda_operator(fresh, xc)

    probe = jnp.arange(4, dtype=jnp.float64)[None, :] / 7.0
    assert cached_diagonal.shape == (4,)
    assert jnp.allclose(cached_diagonal, fresh_diagonal, atol=1e-10)
    assert jnp.allclose(cached_vind(probe), fresh_vind(probe), atol=1e-10)


def test_matrix_free_tdhf_matches_materialized_matrix_for_multi_virtual_hybrid():
    nmo = 4
    rep_tensor = _symmetric_rep_tensor(nmo, scale=50.0)
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
    xc = _ToyAdiabaticFunctional(
        name="hybrid_only",
        energy_density_fn=lambda rho: jnp.zeros_like(rho),
        exact_exchange_fraction=0.25,
    )

    vind_free = response_module.build_restricted_tdhf_operator(molecule, xc)
    z = jnp.array(
        [
            [0.1, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, -0.8],
            [-0.2, 0.3, 0.6, -0.1, 0.4, 0.9, -0.5, 0.8],
        ]
    )

    x = z[:, :4]
    y = z[:, 4:]
    a_matrix, b_matrix = _tdhf_operator_matrices(vind_free, 4)
    expected = jnp.concatenate(
        [
            x @ a_matrix.T + y @ b_matrix.T,
            -(x @ b_matrix.T + y @ a_matrix.T),
        ],
        axis=-1,
    )

    assert jnp.allclose(vind_free(z), expected, atol=1e-9)


def test_response_matrices_match_toy_analytic_values():
    molecule = _make_toy_molecule()
    xc = _lda_from_callable("toy", lambda rho: 0.5 * rho)

    vind, _, _ = build_restricted_tda_operator(molecule, xc)
    tdhf_vind = response_module.build_restricted_tdhf_operator(molecule, xc)
    a_matrix = _operator_matrix(vind, 1)
    _, b_matrix = _tdhf_operator_matrices(tdhf_vind, 1)

    assert a_matrix.shape == (1, 1)
    assert b_matrix.shape == (1, 1)
    assert jnp.allclose(a_matrix[0, 0], 2.0)
    assert jnp.allclose(b_matrix[0, 0], 1.0)


def test_restricted_casida_tddft_returns_expected_toy_excitation():
    molecule = _make_toy_molecule()
    xc = _lda_from_callable("toy", lambda rho: 0.5 * rho)
    solver = RestrictedCasidaTDDFT(molecule, xc)

    result = solver.kernel(nstates=1)
    tda = solver.tda(nstates=1)

    assert jnp.allclose(tda.excitation_energies, jnp.array([2.0]))
    assert jnp.allclose(result.excitation_energies, jnp.array([jnp.sqrt(3.0)]))
    assert result.x_amplitudes.shape == (1, 1, 1)
    assert result.y_amplitudes.shape == (1, 1, 1)


def test_restricted_solver_applies_posthoc_second_order_corrections():
    molecule = _make_toy_molecule()

    class _PostHocDoubleHybridXC:
        exact_exchange_fraction = 0.0

        def local_kernel(self, density):
            return jnp.zeros_like(density)

        def post_tda_correction(self, mol, result, *, occupation_tolerance=1e-8):
            del mol, occupation_tolerance
            return jnp.full_like(result.excitation_energies, 0.25)

        def post_tddft_correction(self, mol, result, *, occupation_tolerance=1e-8):
            del mol, occupation_tolerance
            return jnp.full_like(result.excitation_energies, -0.125)

    solver = RestrictedCasidaTDDFT(molecule, _PostHocDoubleHybridXC(), eigensolver="davidson")
    tda = solver.tda(nstates=1)
    casida = solver.kernel(nstates=1)

    assert jnp.allclose(tda.excitation_energies, jnp.array([1.25]), atol=1e-10)
    assert jnp.allclose(
        casida.excitation_energies,
        jnp.array([1.0 - 0.125]),
        atol=1e-10,
    )
    assert jnp.allclose(tda.posthoc_correction, jnp.array([0.25]))
    assert jnp.allclose(casida.posthoc_correction, jnp.array([-0.125]))


def test_unrestricted_solver_applies_posthoc_second_order_corrections():
    molecule = _make_open_shell_toy_molecule()

    class _PostHocOpenShellXC:
        exact_exchange_fraction = 0.0

        def spin_local_kernel(self, rho_a, rho_b):
            del rho_a, rho_b
            zeros = jnp.zeros((molecule.grid.weights.shape[0],), dtype=jnp.float64)
            return zeros, zeros, zeros

        def post_tda_correction(self, mol, result, *, occupation_tolerance=1e-8):
            del mol, occupation_tolerance
            return jnp.full_like(result.excitation_energies, 0.125)

        def post_tddft_correction(self, mol, result, *, occupation_tolerance=1e-8):
            del mol, occupation_tolerance
            return jnp.full_like(result.excitation_energies, -0.0625)

    solver = UnrestrictedCasidaTDDFT(molecule, _PostHocOpenShellXC())
    tda = solver.tda(nstates=1)
    casida = solver.kernel(nstates=1)

    assert jnp.allclose(tda.excitation_energies, jnp.array([2.125]), atol=1e-10)
    assert jnp.allclose(casida.excitation_energies, jnp.array([1.9375]), atol=1e-10)
    assert jnp.allclose(tda.posthoc_correction, jnp.array([0.125]))
    assert jnp.allclose(casida.posthoc_correction, jnp.array([-0.0625]))


def test_unrestricted_cisd_correction_is_zero_for_single_electron_reference():
    molecule = _make_open_shell_toy_molecule()
    solver = UnrestrictedCasidaTDDFT(molecule)
    result = solver.tda(nstates=1)
    correction = unrestricted_cisd_second_order_correction(molecule, result, ac=0.4)

    assert correction.shape == (1,)
    assert jnp.allclose(correction, jnp.zeros((1,), dtype=correction.dtype), atol=1e-10)


def test_restricted_cisd_correction_is_root_specific_and_scaled_by_ac():
    nmo = 3
    ao = jnp.eye(nmo, dtype=jnp.float64)
    raw_eri = jnp.arange(nmo**4, dtype=jnp.float64).reshape(nmo, nmo, nmo, nmo) / 100.0
    rep_tensor = 0.25 * (
        raw_eri
        + jnp.transpose(raw_eri, (1, 0, 2, 3))
        + jnp.transpose(raw_eri, (0, 1, 3, 2))
        + jnp.transpose(raw_eri, (2, 3, 0, 1))
    )
    molecule = _ToyMolecule(
        ao=ao,
        ao_deriv1=jnp.stack([ao, jnp.zeros_like(ao), jnp.zeros_like(ao), jnp.zeros_like(ao)]),
        grid=_Grid(weights=jnp.ones((nmo,), dtype=jnp.float64)),
        rep_tensor=rep_tensor,
        mo_coeff=jnp.stack([jnp.eye(nmo, dtype=jnp.float64), jnp.eye(nmo, dtype=jnp.float64)]),
        mo_occ=jnp.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float64),
        mo_energy=jnp.asarray([[0.0, 1.4, 1.9], [0.0, 1.4, 1.9]], dtype=jnp.float64),
        rdm1=jnp.stack(
            [
                jnp.diag(jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64)),
                jnp.diag(jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64)),
            ]
        ),
    )
    amplitudes = jnp.asarray(
        [
            [[1.0 / jnp.sqrt(2.0), 0.0]],
            [[1.0 / jnp.sqrt(2.0), 0.0]],
        ],
        dtype=jnp.float64,
    )
    result = TDAResult(
        excitation_energies=jnp.asarray([0.45, 0.80], dtype=jnp.float64),
        amplitudes=amplitudes,
    )

    unscaled = restricted_cisd_second_order_correction(molecule, result)
    scaled = restricted_cisd_second_order_correction(molecule, result, ac=0.37)

    assert unscaled.shape == (2,)
    assert jnp.all(jnp.isfinite(unscaled))
    assert not jnp.allclose(unscaled[0], unscaled[1], atol=1e-12)
    assert jnp.allclose(scaled, 0.37 * unscaled, atol=1e-12)


def test_matrix_free_tda_vind_matches_response_matrix_reference():
    molecule = _make_toy_molecule()
    xc = _lda_from_callable("toy", lambda rho: 0.5 * rho)
    solver = RestrictedCasidaTDDFT(molecule, xc)
    vind_free = solver.gen_tda_vind()

    x = jnp.array([[0.3], [1.1]])
    expected = 2.0 * x
    assert jnp.allclose(vind_free(x), expected, atol=1e-9)


def test_restricted_vind_defaults_are_matrix_free():
    molecule = _make_toy_molecule()
    xc = _lda_from_callable("toy", lambda rho: 0.5 * rho)
    solver = RestrictedCasidaTDDFT(molecule, xc)

    vind_tda = solver.gen_tda_vind()
    vind_tdhf = solver.gen_tdhf_vind()

    assert vind_tda(jnp.ones((1, 1))).shape == (1, 1)
    assert vind_tdhf(jnp.ones((1, 2))).shape == (1, 2)


def test_matrix_free_tdhf_vind_matches_response_matrix_reference_with_global_hybrid():
    pair_values = jnp.zeros((3, 3), dtype=jnp.float64)
    pair_values = pair_values.at[1, 1].set(0.6)
    pair_values = pair_values.at[0, 2].set(0.5)
    pair_values = pair_values.at[2, 0].set(0.5)
    rep_tensor = _rep_tensor_from_pair_matrix(pair_values)
    molecule = _make_toy_molecule(rep_tensor=rep_tensor)

    class _HybridXC:
        exact_exchange_fraction = 0.25

        def local_kernel(self, density):
            return jnp.zeros_like(density)

    solver = RestrictedCasidaTDDFT(molecule, _HybridXC())
    vind_free = solver.gen_tdhf_vind()

    z = jnp.array([[0.4, -0.2], [1.3, 0.7]])
    a_matrix = jnp.asarray([[2.075]])
    b_matrix = jnp.asarray([[1.05]])
    dense_out = jnp.concatenate(
        [
            z[:, :1] @ a_matrix.T + z[:, 1:] @ b_matrix.T,
            -(z[:, :1] @ b_matrix.T + z[:, 1:] @ a_matrix.T),
        ],
        axis=-1,
    )
    assert jnp.allclose(vind_free(z), dense_out, atol=1e-9)


def test_restricted_hybrid_exchange_ignores_hfx_nu_shortcut():
    rep_tensor = _symmetric_rep_tensor(2, scale=10.0)
    molecule = _make_toy_molecule(rep_tensor=rep_tensor)
    hfx_nu = jnp.asarray(
        [
            [
                [[70.0, 20.0], [20.0, 50.0]],
                [[40.0, 10.0], [10.0, 30.0]],
            ]
        ],
        dtype=jnp.float64,
    )
    molecule.hfx_nu = hfx_nu

    class _HybridXC:
        exact_exchange_fraction = 0.25

        def local_kernel(self, density):
            return jnp.zeros_like(density)

    data = response_module._build_restricted_response_operator_data(molecule, _HybridXC())
    assert not hasattr(data, "eri_oovv")
    assert not hasattr(data, "hybrid_exchange_a_action_fn")
    assert not hasattr(data, "hybrid_exchange_b_action_fn")
    tda_data = response_module._build_restricted_response_operator_data(
        molecule,
        _HybridXC(),
        need_b_terms=False,
    )
    assert not hasattr(tda_data, "hybrid_exchange_a_action_fn")
    assert not hasattr(tda_data, "hybrid_exchange_b_action_fn")

    tda_vind, diagonal, _ = build_restricted_tda_operator(molecule, _HybridXC())
    tdhf_vind = response_module.build_restricted_tdhf_operator(molecule, _HybridXC())
    a_matrix = _operator_matrix(tda_vind, 1)
    _, b_matrix = _tdhf_operator_matrices(tdhf_vind, 1)

    molecule_without_nu = _make_toy_molecule(rep_tensor=rep_tensor)
    reference_tda_vind, reference_diagonal, _ = build_restricted_tda_operator(
        molecule_without_nu,
        _HybridXC(),
    )
    reference_tdhf_vind = response_module.build_restricted_tdhf_operator(
        molecule_without_nu,
        _HybridXC(),
    )
    reference_a = _operator_matrix(reference_tda_vind, 1)
    _, reference_b = _tdhf_operator_matrices(reference_tdhf_vind, 1)

    assert jnp.allclose(diagonal, reference_diagonal, atol=1e-9)
    assert jnp.allclose(a_matrix, reference_a, atol=1e-9)
    assert jnp.allclose(b_matrix, reference_b, atol=1e-9)


def test_large_toy_tda_davidson_uses_operator_path():
    molecule = _make_large_diagonal_toy_molecule()
    davidson_solver = RestrictedCasidaTDDFT(molecule, eigensolver="davidson")

    davidson = davidson_solver.tda(nstates=4)
    nocc = int(jnp.count_nonzero(molecule.mo_occ[0] > 1e-8))
    expected = jnp.sort(
        (
            molecule.mo_energy[0, nocc:]
            - molecule.mo_energy[0, :nocc, None]
        ).reshape(-1)
    )[:4]

    assert jnp.allclose(davidson.excitation_energies, expected, atol=1e-8)


def test_large_toy_casida_davidson_uses_operator_path():
    molecule = _make_large_diagonal_toy_molecule()
    davidson_solver = RestrictedCasidaTDDFT(molecule, eigensolver="davidson")

    davidson = davidson_solver.kernel(nstates=4)
    nocc = int(jnp.count_nonzero(molecule.mo_occ[0] > 1e-8))
    expected = jnp.sort(
        (
            molecule.mo_energy[0, nocc:]
            - molecule.mo_energy[0, :nocc, None]
        ).reshape(-1)
    )[:4]

    assert jnp.allclose(davidson.excitation_energies, expected, atol=1e-8)


def test_jitted_tda_does_not_cache_traced_matrix_before_jitted_kernel():
    molecule = _make_toy_molecule()
    molecule.nocc = 1
    xc = _lda_from_callable("toy", lambda rho: 0.5 * rho)
    solver = RestrictedCasidaTDDFT(molecule, xc, eigensolver="davidson")

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

    solver = RestrictedCasidaTDDFT(molecule, _StrictGGAXC(), eigensolver="davidson")

    with jax.checking_leaks():
        result = jax.jit(lambda: solver.tda(nstates=1))()

    assert result.excitation_energies.shape == (1,)
    assert features_module._TRANSITION_RESPONSE_FEATURE_CACHE == {}


def test_jitted_semilocal_response_does_not_cache_traced_tensor(monkeypatch):
    semilocal_response_module._GRID_RESPONSE_TENSOR_CACHE.clear()
    monkeypatch.setattr(semilocal_response_module, "hybrid_coeff", lambda _spec: 0.0)
    monkeypatch.setattr(semilocal_response_module, "xc_type", lambda _spec: "LDA")

    def fake_grid_response_variables(molecule, *, feature_kind):
        del feature_kind
        rho = jnp.sum(jnp.asarray(molecule.ao), axis=1)
        return rho, None, None, None

    def fake_eval_xc_response_tensor(_spec, rho, *, grad=None, tau=None):
        del grad, tau
        return None, rho[None, None, :]

    monkeypatch.setattr(
        semilocal_response_module,
        "restricted_grid_response_variables",
        fake_grid_response_variables,
    )
    monkeypatch.setattr(
        semilocal_response_module,
        "eval_xc_response_tensor",
        fake_eval_xc_response_tensor,
    )
    molecule = _make_toy_molecule()
    xc = semilocal_response_module.SemilocalResponseFunctional("toy")

    def evaluate(scale):
        traced_molecule = replace(molecule, ao=molecule.ao * scale)
        return jnp.sum(xc.grid_response_tensor(traced_molecule))

    with jax.checking_leaks():
        value = jax.jit(evaluate)(jnp.asarray(1.0))

    assert jnp.allclose(value, jnp.sum(molecule.ao))
    assert semilocal_response_module._GRID_RESPONSE_TENSOR_CACHE == {}


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


def test_restricted_operator_accepts_action_only_nonlocal_response():
    molecule = _make_toy_molecule()

    class _ActionOnlyXC:
        exact_exchange_fraction = 0.0

        def local_kernel(self, density):
            return jnp.zeros_like(density)

        def nonlocal_response_action(self, mol, amplitudes, *, occupation_tolerance=1e-8):
            del mol, occupation_tolerance
            return 0.2 * amplitudes

        def nonlocal_response_b_action(self, mol, amplitudes, *, occupation_tolerance=1e-8):
            del mol, occupation_tolerance
            return 0.1 * amplitudes

        def nonlocal_response_diagonal(self, mol, *, occupation_tolerance=1e-8):
            del mol, occupation_tolerance
            return jnp.asarray([[0.2]])

    xc = _ActionOnlyXC()
    vind, diagonal, _ = build_restricted_tda_operator(
        molecule,
        xc,
    )
    assert diagonal.shape == (1,)
    assert jnp.allclose(diagonal, jnp.asarray([1.0]), atol=1e-8)
    assert jnp.allclose(vind(jnp.ones((1, 1))), jnp.asarray([[1.2]]), atol=1e-8)


def test_restricted_casida_action_only_nonlocal_uses_operator(monkeypatch):
    molecule = _make_large_diagonal_toy_molecule()

    class _ActionOnlyXC:
        exact_exchange_fraction = 0.0

        def local_kernel(self, density):
            return jnp.zeros_like(density)

        def nonlocal_response_action(self, mol, amplitudes, *, occupation_tolerance=1e-8):
            del mol, occupation_tolerance
            return 0.2 * amplitudes

        def nonlocal_response_b_action(self, mol, amplitudes, *, occupation_tolerance=1e-8):
            del mol, occupation_tolerance
            return 0.1 * amplitudes

        def nonlocal_response_diagonal(self, mol, *, occupation_tolerance=1e-8):
            del occupation_tolerance
            nocc = int(jnp.count_nonzero(mol.mo_occ[0] > 1e-8))
            nvir = int(mol.mo_coeff.shape[-1] - nocc)
            return jnp.full((nocc, nvir), 0.2)

    solver = RestrictedCasidaTDDFT(
        molecule,
        _ActionOnlyXC(),
        eigensolver="davidson",
        davidson_tol=1e-8,
        davidson_max_iter=120,
        davidson_max_subspace=24,
    )
    result = solver.kernel(nstates=4)

    nocc = int(jnp.count_nonzero(molecule.mo_occ[0] > 1e-8))
    delta = molecule.mo_energy[0, nocc:][None, :] - molecule.mo_energy[0, :nocc][:, None]
    expected = jnp.sort(jnp.sqrt((delta.reshape(-1) + 0.1) * (delta.reshape(-1) + 0.3)))[:4]
    assert jnp.allclose(result.excitation_energies, expected, atol=1e-7)

    class _ScaledActionOnlyXC(_ActionOnlyXC):
        def __init__(self, scale):
            self.scale = scale

        def nonlocal_response_action(self, mol, amplitudes, *, occupation_tolerance=1e-8):
            del mol, occupation_tolerance
            return self.scale * amplitudes

        def nonlocal_response_diagonal(self, mol, *, occupation_tolerance=1e-8):
            del occupation_tolerance
            nocc = int(jnp.count_nonzero(mol.mo_occ[0] > 1e-8))
            nvir = int(mol.mo_coeff.shape[-1] - nocc)
            return jnp.full((nocc, nvir), self.scale)

    def s1_energy(scale):
        solver_scaled = RestrictedCasidaTDDFT(
            molecule,
            _ScaledActionOnlyXC(scale),
            eigensolver="davidson",
            davidson_tol=1e-8,
            davidson_max_iter=120,
            davidson_max_subspace=24,
        )
        return solver_scaled.kernel(nstates=1).excitation_energies[0]

    scale = jnp.asarray(0.2)
    first_delta = jnp.min(delta)
    expected_grad = (first_delta + scale) / jnp.sqrt((first_delta + scale) ** 2 - 0.1**2)
    assert jnp.allclose(jax.grad(s1_energy)(scale), expected_grad, atol=1e-6)


def test_small_tda_uses_operator_builder(monkeypatch):
    molecule = _make_toy_molecule()
    xc = _lda_from_callable("toy", lambda rho: 0.5 * rho)
    solver = RestrictedCasidaTDDFT(molecule, xc, eigensolver="auto")
    original = casida_module.build_restricted_tda_operator
    calls = 0

    def _counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(casida_module, "build_restricted_tda_operator", _counted)
    result = solver.tda(nstates=1)
    assert calls == 1
    assert jnp.allclose(result.excitation_energies, jnp.array([2.0]))


def test_small_casida_uses_operator_builder(monkeypatch):
    molecule = _make_toy_molecule()
    xc = _lda_from_callable("toy", lambda rho: 0.5 * rho)
    solver = RestrictedCasidaTDDFT(molecule, xc, eigensolver="auto")
    original = casida_module.gen_tdhf_vind
    calls = 0

    def _counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(casida_module, "gen_tdhf_vind", _counted)
    result = solver.kernel(nstates=1)
    assert calls == 1
    assert jnp.allclose(result.excitation_energies, jnp.array([jnp.sqrt(3.0)]))


def test_hybrid_exchange_contributes_to_restricted_response_operator():
    pair_values = jnp.zeros((3, 3), dtype=jnp.float64)
    pair_values = pair_values.at[1, 1].set(0.6)
    pair_values = pair_values.at[0, 2].set(0.5)
    pair_values = pair_values.at[2, 0].set(0.5)
    rep_tensor = _rep_tensor_from_pair_matrix(pair_values)
    molecule = _make_toy_molecule(rep_tensor=rep_tensor)
    xc = _ToyAdiabaticFunctional(
        name="hybrid_only",
        energy_density_fn=lambda rho: jnp.zeros_like(rho),
        exact_exchange_fraction=0.25,
    )

    tda_vind, _, _ = build_restricted_tda_operator(molecule, xc)
    tdhf_vind = response_module.build_restricted_tdhf_operator(molecule, xc)
    a_matrix = _operator_matrix(tda_vind, 1)
    _, b_matrix = _tdhf_operator_matrices(tdhf_vind, 1)

    # For i=j=0, a=b=1 with alpha=0.25:
    # A = dE + 2(ia|jb) - alpha(ij|ab) = 1 + 2*0.6 - 0.25*0.5 = 2.075
    # B = 2(ia|bj) - alpha(ib|aj) = 2*0.6 - 0.25*0.6 = 1.05
    assert jnp.allclose(a_matrix[0, 0], 2.075, atol=1e-6)
    assert jnp.allclose(b_matrix[0, 0], 1.05, atol=1e-6)


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
        build_restricted_tda_operator(molecule, _LocalHybridXC())


def test_response_kernel_rejects_nonfinite_grid_values():
    molecule = _make_toy_molecule()

    class _NaNKernelXC:
        exact_exchange_fraction = 0.0

        def local_kernel(self, density):
            del density
            return jnp.array(jnp.nan)

    with pytest.raises(ValueError, match="non-finite values"):
        build_restricted_tda_operator(molecule, _NaNKernelXC())


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

    scalar_vind = response_module.build_restricted_tdhf_operator(
        molecule,
        _ScalarLocalHybridXC(),
    )
    hybrid_vind = response_module.build_restricted_tdhf_operator(
        molecule,
        _ToyAdiabaticFunctional(
            name="hybrid_ref",
            energy_density_fn=lambda rho: jnp.zeros_like(rho),
            exact_exchange_fraction=0.25,
        ),
    )
    scalar_a, scalar_b = _tdhf_operator_matrices(scalar_vind, 1)
    hybrid_a, hybrid_b = _tdhf_operator_matrices(hybrid_vind, 1)
    assert jnp.allclose(scalar_a, hybrid_a, atol=1e-9)
    assert jnp.allclose(scalar_b, hybrid_b, atol=1e-9)


def test_gga_without_strict_response_tensor_is_rejected():
    molecule = _make_toy_molecule()

    class _ApproximateGGAXC:
        exact_exchange_fraction = 0.0
        response_feature_kind = "GGA"

        def local_kernel(self, density):
            return jnp.ones_like(density)

    with pytest.raises(ValueError, match="requires grid_response_tensor"):
        build_restricted_tda_operator(molecule, _ApproximateGGAXC())


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

    tda_vind, _, _ = build_restricted_tda_operator(molecule, _StrictGGAXC())
    tdhf_vind = response_module.build_restricted_tdhf_operator(molecule, _StrictGGAXC())
    a_matrix = _operator_matrix(tda_vind, 1).reshape(1, 1, 1, 1)
    _, b_matrix = _tdhf_operator_matrices(tdhf_vind, 1)
    b_matrix = b_matrix.reshape(1, 1, 1, 1)

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

    assert jnp.allclose(a_matrix, 1.0 + xc_expected, atol=1e-8)
    assert jnp.allclose(b_matrix, xc_expected, atol=1e-8)


def test_grid_response_hvp_tda_matches_dense_mgga_tensor_action():
    molecule = _make_toy_molecule()
    tensor = jnp.zeros((5, 5, 2), dtype=jnp.float64)
    tensor = tensor.at[0, 0].set(jnp.array([0.2, 0.3]))
    tensor = tensor.at[1, 1].set(jnp.array([0.4, 0.1]))
    tensor = tensor.at[3, 3].set(jnp.array([0.2, 0.5]))
    tensor = tensor.at[4, 4].set(jnp.array([0.8, 1.1]))
    tensor = tensor.at[0, 4].set(jnp.array([0.1, 0.2]))
    tensor = tensor.at[4, 0].set(jnp.array([0.1, 0.2]))

    class _DenseMGGAXC:
        exact_exchange_fraction = 0.0
        response_feature_kind = "MGGA"

        def grid_response_tensor(self, mol):
            del mol
            return tensor

    class _HVPMGGAXC:
        exact_exchange_fraction = 0.0
        response_feature_kind = "MGGA"

        def grid_response_hvp(self, mol, tangent):
            del mol
            return jnp.einsum("xyg,nyg->nxg", tensor, tangent)

    dense_vind, dense_diagonal, _ = build_restricted_tda_operator(
        molecule,
        _DenseMGGAXC(),
    )
    hvp_vind, hvp_diagonal, _ = build_restricted_tda_operator(
        molecule,
        _HVPMGGAXC(),
    )

    assert jnp.allclose(hvp_diagonal, dense_diagonal, atol=1e-10)
    assert jnp.allclose(
        _operator_matrix(hvp_vind, 1),
        _operator_matrix(dense_vind, 1),
        atol=1e-10,
    )


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

    tda_vind, _, _ = build_restricted_tda_operator(molecule, _StrictMGGAXC())
    tdhf_vind = response_module.build_restricted_tdhf_operator(molecule, _StrictMGGAXC())
    a_matrix = _operator_matrix(tda_vind, 1).reshape(1, 1, 1, 1)
    _, b_matrix = _tdhf_operator_matrices(tdhf_vind, 1)
    b_matrix = b_matrix.reshape(1, 1, 1, 1)

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

    assert jnp.allclose(a_matrix, 1.0 + xc_expected, atol=1e-8)
    assert jnp.allclose(b_matrix, xc_expected, atol=1e-8)
