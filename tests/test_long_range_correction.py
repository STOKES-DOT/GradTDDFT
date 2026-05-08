from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze

from td_graddft.tddft import (
    GridPointModeCouplingNet,
    GridPointModeLongRangeCorrectedFunctional,
    build_grid_point_mode_basis,
    build_grid_point_mode_features,
    LongRangeCorrectedFunctional,
    LongRangeXCNet,
    build_long_range_pair_features,
    build_restricted_response_matrices,
)
from td_graddft.xc import lda_from_callable


@dataclass
class _Grid:
    weights: jnp.ndarray
    coords: jnp.ndarray


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


@dataclass
class _MockBoundResponse:
    potential: jnp.ndarray
    grad: jnp.ndarray
    tau: jnp.ndarray
    kernel: jnp.ndarray
    energy_density_values: jnp.ndarray
    hf_fraction: jnp.ndarray
    exact_exchange_fraction: float = 0.0

    def grid_potential_components(self, molecule):
        del molecule
        return self.potential, self.grad, self.tau

    def grid_kernel(self, molecule):
        del molecule
        return self.kernel

    def energy_density(self, density):
        del density
        return self.energy_density_values

    def grid_hf_fraction(self, molecule):
        del molecule
        return self.hf_fraction


def _make_toy_molecule():
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
        grid=_Grid(
            weights=jnp.array([1.0, 1.0]),
            coords=jnp.array([[0.0, 0.0, 0.0], [1.25, 0.0, 0.0]]),
        ),
        rep_tensor=jnp.zeros((2, 2, 2, 2)),
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        rdm1=rdm1,
    )


def _make_toy_molecule_four_grid_points():
    ao = jnp.array(
        [
            [1.0, 0.4],
            [0.8, 0.6],
            [0.6, 0.8],
            [0.4, 1.0],
        ]
    )
    ao_deriv1 = jnp.array(
        [
            ao,
            [[0.20, 0.10], [0.15, 0.12], [0.12, 0.15], [0.10, 0.20]],
            [[0.05, 0.00], [0.04, 0.01], [0.01, 0.04], [0.00, 0.05]],
            [[0.00, 0.07], [0.03, 0.06], [0.06, 0.03], [0.07, 0.00]],
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
        grid=_Grid(
            weights=jnp.array([0.5, 0.7, 0.9, 1.1]),
            coords=jnp.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.2, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.5, -0.1, 0.0],
                ]
            ),
        ),
        rep_tensor=jnp.zeros((2, 2, 2, 2)),
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        rdm1=rdm1,
    )


def _inverse_softplus(value):
    value = jnp.asarray(value)
    return jnp.log(jnp.expm1(value))


def _constant_lr_params(
    functional: LongRangeCorrectedFunctional,
    molecule: _ToyMolecule,
    *,
    alpha: float,
    gamma: float,
):
    params = unfreeze(functional.init(jax.random.PRNGKey(0), molecule))
    params["params"]["AlphaHead"]["kernel"] = jnp.zeros_like(params["params"]["AlphaHead"]["kernel"])
    params["params"]["GammaHead"]["kernel"] = jnp.zeros_like(params["params"]["GammaHead"]["kernel"])
    params["params"]["AlphaHead"]["bias"] = jnp.full_like(
        params["params"]["AlphaHead"]["bias"],
        _inverse_softplus(alpha),
    )
    params["params"]["GammaHead"]["bias"] = jnp.full_like(
        params["params"]["GammaHead"]["bias"],
        _inverse_softplus(gamma - functional.model.gamma_floor),
    )
    return freeze(params)


def test_long_range_pair_features_are_symmetric():
    molecule = _make_toy_molecule()

    features = build_long_range_pair_features(molecule)

    assert features.shape == (2, 2, 30)
    assert jnp.allclose(features, jnp.swapaxes(features, 0, 1), atol=1e-8)


def test_long_range_pair_features_include_optional_local_channels():
    molecule = _make_toy_molecule()
    molecule.hfx_local = jnp.ones((2, 2, 2)) * jnp.array([[[0.2, -0.4], [0.1, -0.3]], [[0.2, -0.4], [0.1, -0.3]]])
    molecule.pt2_local = jnp.array([0.05, -0.08])

    features = build_long_range_pair_features(molecule)

    assert features.shape == (2, 2, 39)
    assert jnp.all(jnp.isfinite(features))


def test_long_range_pair_features_include_base_response_descriptors():
    molecule = _make_toy_molecule()
    base_bound = _MockBoundResponse(
        potential=jnp.array([0.2, -0.1]),
        grad=jnp.array([[0.1, 0.0, 0.2], [0.0, 0.2, 0.1]]),
        tau=jnp.array([0.05, 0.08]),
        kernel=jnp.array([-0.3, -0.25]),
        energy_density_values=jnp.array([-0.4, -0.35]),
        hf_fraction=jnp.array([0.2, 0.3]),
    )

    features = build_long_range_pair_features(molecule, base_bound=base_bound)

    assert features.shape == (2, 2, 30)
    assert jnp.all(jnp.isfinite(features))


def test_grid_point_mode_features_and_basis_have_expected_shapes():
    molecule = _make_toy_molecule_four_grid_points()

    features = build_grid_point_mode_features(molecule, mode_point_indices=jnp.array([0, 2]))
    basis = build_grid_point_mode_basis(molecule, mode_point_indices=jnp.array([0, 2]))

    assert features.shape == (2, 4)
    assert basis.shape == (4, 2)
    norms = jnp.sqrt(jnp.sum(molecule.grid.weights[:, None] * basis * basis, axis=0))
    assert jnp.allclose(norms, jnp.ones_like(norms), atol=1e-6)


def test_grid_point_mode_features_and_basis_handle_signed_weights():
    molecule = _make_toy_molecule_four_grid_points()
    molecule.grid.weights = jnp.array([0.5, -0.7, 0.9, 1.1])

    features = build_grid_point_mode_features(molecule, mode_point_indices=jnp.array([0, 1, 3]))
    basis = build_grid_point_mode_basis(molecule, mode_point_indices=jnp.array([0, 1, 3]))

    assert jnp.all(jnp.isfinite(features))
    assert jnp.all(jnp.isfinite(basis))


def test_grid_point_mode_coupling_net_is_finite_and_bounded_for_many_modes():
    mode_features = jnp.stack(
        [
            jnp.linspace(0.1, 1.3, 128),
            jnp.linspace(-0.4, 0.6, 128),
            jnp.sin(jnp.linspace(0.0, 3.0, 128)),
            jnp.cos(jnp.linspace(0.0, 2.0, 128)),
        ],
        axis=-1,
    )
    model = GridPointModeCouplingNet(
        hidden_dims=(32,),
        latent_dim=16,
        coupling_scale=0.35,
        initial_coupling_strength=0.08,
    )

    params = model.init(jax.random.PRNGKey(7), mode_features)
    coupling = model.apply(params, mode_features)

    assert jnp.all(jnp.isfinite(coupling))
    assert jnp.allclose(coupling, coupling.T, atol=1e-8)
    assert float(jnp.max(jnp.abs(coupling))) <= 0.35 + 1e-6


def test_long_range_corrected_functional_adds_nonlocal_tda_shift():
    molecule = _make_toy_molecule()
    base = lda_from_callable("zero", lambda rho: jnp.zeros_like(rho))
    functional = LongRangeCorrectedFunctional(
        base_functional=base,
        model=LongRangeXCNet(hidden_dims=(), gamma_floor=1e-3),
        distance_floor=0.35,
    )
    lr_params = _constant_lr_params(
        functional,
        molecule,
        alpha=0.08,
        gamma=1.6,
    )
    combined_params = {"lr_correction": lr_params}

    baseline = build_restricted_response_matrices(molecule, base)
    corrected = build_restricted_response_matrices(
        molecule,
        functional,
        xc_params=combined_params,
    )
    bound = functional.bind_to_molecule(combined_params, molecule)
    density = molecule.density().sum(axis=-1)

    assert corrected.a_matrix[0, 0, 0, 0] < baseline.a_matrix[0, 0, 0, 0]
    assert corrected.b_matrix[0, 0, 0, 0] < baseline.b_matrix[0, 0, 0, 0]
    assert jnp.all(bound.nonlocal_response_diagonal(molecule) < 0.0)
    assert jnp.allclose(bound.local_kernel(density), jnp.zeros_like(density), atol=1e-8)
    assert jnp.allclose(bound.local_potential(density), jnp.zeros_like(density), atol=1e-8)


def test_long_range_corrected_functional_can_subsample_response_grid():
    molecule = _make_toy_molecule_four_grid_points()
    base = lda_from_callable("zero", lambda rho: jnp.zeros_like(rho))
    functional = LongRangeCorrectedFunctional(
        base_functional=base,
        model=LongRangeXCNet(hidden_dims=(), gamma_floor=1e-3),
        max_pair_points=2,
    )
    lr_params = _constant_lr_params(functional, molecule, alpha=0.05, gamma=1.4)
    bound = functional.bind_to_molecule({"lr_correction": lr_params}, molecule)

    assert bound.pair_kernel.shape == (2, 2)
    assert bound.transition_densities.shape[0] == 2
    assert bound.weighted_transition_densities.shape[0] == 2
    assert jnp.isclose(jnp.sum(bound.grid_weights), jnp.sum(molecule.grid.weights), atol=1e-8)


def test_grid_point_mode_long_range_functional_adds_nonlocal_tda_shift():
    molecule = _make_toy_molecule_four_grid_points()
    base = lda_from_callable("zero", lambda rho: jnp.zeros_like(rho))
    functional = GridPointModeLongRangeCorrectedFunctional(
        base_functional=base,
        model=GridPointModeCouplingNet(hidden_dims=(), latent_dim=2, coupling_scale=1.0),
        max_mode_points=2,
        mode_width_scale=1.2,
    )
    params = functional.init(jax.random.PRNGKey(0), molecule)
    combined_params = functional.combine_params(None, params)

    baseline = build_restricted_response_matrices(molecule, base)
    corrected = build_restricted_response_matrices(
        molecule,
        functional,
        xc_params=combined_params,
    )
    bound = functional.bind_to_molecule(combined_params, molecule)

    assert corrected.a_matrix[0, 0, 0, 0] <= baseline.a_matrix[0, 0, 0, 0]
    assert bound.coupling_matrix.shape == (2, 2)
    assert bound.mode_projections.shape[0] == 2
    assert bound.nonlocal_response_diagonal(molecule).shape == (1, 1)
