from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from jax.lax import Precision

import td_graddft.training.targets as training_targets
from td_graddft import HARTREE_TO_EV, lorentzian_spectrum
from td_graddft.neural_xc import make_neural_xc_functional
from td_graddft.training import (
    MolecularTrainingDatum,
    MolecularTrainingConfig,
    create_train_state_from_molecule,
    dm21_scf_regularization_delta_energy,
    density_matrix_matching_penalty,
    density_matching_penalty,
    density_on_grid,
    density_on_grid_spin_resolved,
    density_stationarity_penalty,
    molecular_loss,
    make_fixed_density_predictor,
    make_molecular_train_step,
    predict_ground_state_density,
    predict_ground_state_molecule,
    predict_excitation_energies,
    predict_oscillator_strengths,
    predict_excitation_spectrum,
    predict_ground_state_total_energy,
    xc_kernel_matching_penalty,
)
from td_graddft.training.targets import _electron_count, orbital_energy_matching_penalty
from td_graddft.scf.molecules import QuadratureGrid, UnrestrictedMolecule
from td_graddft.workflows.core import run_molecule_from_spec
from td_graddft.workflows.types import MoleculeSpecConfig, SimulationConfig


@dataclass(frozen=True)
class _ToyAdiabaticFunctional:
    name: str
    energy_density_fn: Callable[[jnp.ndarray], jnp.ndarray]
    exact_exchange_fraction: jnp.ndarray | float = 0.0

    def energy_density(self, density):
        return self.energy_density_fn(jnp.asarray(density))

    def local_potential(self, density):
        density = jnp.asarray(density)
        flat = density.reshape(-1)

        def local_energy(value):
            return value * self.energy_density_fn(value)

        return jax.vmap(jax.grad(local_energy))(flat).reshape(density.shape)

    def local_kernel(self, density):
        density = jnp.asarray(density)
        flat = density.reshape(-1)

        def local_energy(value):
            return value * self.energy_density_fn(value)

        return jax.vmap(jax.grad(jax.grad(local_energy)))(flat).reshape(density.shape)


@dataclass
class _Grid:
    weights: jnp.ndarray


@dataclass
class _ToyMolecule:
    ao: jnp.ndarray
    grid: _Grid
    rep_tensor: jnp.ndarray
    mo_coeff: jnp.ndarray
    mo_occ: jnp.ndarray
    mo_energy: jnp.ndarray
    rdm1: jnp.ndarray
    h1e: jnp.ndarray
    nuclear_repulsion: float
    dipole_integrals: jnp.ndarray | None = None
    overlap_matrix: jnp.ndarray | None = None

    def density(self):
        return jnp.einsum("spq,rp,rq->rs", self.rdm1, self.ao, self.ao)


def _make_toy_molecule():
    ao = jnp.array([[1.0, 0.5], [0.5, 1.0]])
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
        grid=_Grid(weights=jnp.array([1.0, 1.0])),
        rep_tensor=jnp.zeros((2, 2, 2, 2)),
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        rdm1=rdm1,
        h1e=jnp.zeros((2, 2)),
        nuclear_repulsion=0.0,
        dipole_integrals=jnp.array(
            [
                [[0.0, 1.0], [1.0, 0.0]],
                [[0.0, 0.2], [0.2, 0.0]],
                [[0.0, 0.1], [0.1, 0.0]],
            ]
        ),
    )


def _make_hybrid_toy_molecule():
    ao = jnp.eye(2)
    mo_coeff = jnp.stack([jnp.eye(2), jnp.eye(2)], axis=0)
    mo_occ = jnp.array([[1.0, 0.0], [1.0, 0.0]])
    mo_energy = jnp.array([[0.0, 1.0], [0.0, 1.0]])
    rdm1 = jnp.array(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0]],
        ]
    )
    rep_tensor = jnp.zeros((2, 2, 2, 2))
    rep_tensor = rep_tensor.at[0, 0, 0, 0].set(4.0)
    rep_tensor = rep_tensor.at[0, 0, 1, 1].set(2.0)
    return _ToyMolecule(
        ao=ao,
        grid=_Grid(weights=jnp.array([1.0, 1.0])),
        rep_tensor=rep_tensor,
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        rdm1=rdm1,
        h1e=jnp.zeros((2, 2)),
        nuclear_repulsion=0.0,
    )


def _make_overlap_toy_molecule():
    return _ToyMolecule(
        ao=jnp.eye(2),
        grid=_Grid(weights=jnp.array([1.0, 1.0])),
        rep_tensor=jnp.zeros((2, 2, 2, 2)),
        mo_coeff=jnp.stack([jnp.eye(2), jnp.eye(2)], axis=0),
        mo_occ=jnp.array([[1.0, 0.0], [1.0, 0.0]]),
        mo_energy=jnp.array([[0.0, 1.0], [0.0, 1.0]]),
        rdm1=jnp.array(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 0.0]],
            ]
        ),
        h1e=jnp.zeros((2, 2)),
        nuclear_repulsion=0.0,
        overlap_matrix=jnp.array([[1.5, 0.0], [0.0, 0.5]]),
    )


def _clip_toy_density(density: jnp.ndarray, density_floor: float) -> jnp.ndarray:
    return jnp.maximum(jnp.asarray(density), density_floor)


class _ToyPointwiseNet(nn.Module):
    hidden_dims: Sequence[int]
    output_dim: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = inputs
        for width in self.hidden_dims:
            x = nn.Dense(width)(x)
            x = self.activation(x)
        return nn.Dense(self.output_dim)(x)


@dataclass(frozen=True)
class _ToyCoefficientCore:
    model: nn.Module
    name: str = "toy_xc"
    hybrid_fraction_init: float | None = None
    hybrid_fraction_bounds: tuple[float, float] = (0.0, 1.0)

    def init(self, rng: jnp.ndarray, coefficient_inputs: jnp.ndarray) -> Any:
        params = self.model.init(rng, jnp.asarray(coefficient_inputs))
        if self.hybrid_fraction_init is None:
            return params
        lower, upper = self.hybrid_fraction_bounds
        scaled = (self.hybrid_fraction_init - lower) / (upper - lower)
        clipped = jnp.clip(scaled, 1e-6, 1.0 - 1e-6)
        raw = jnp.log(clipped / (1.0 - clipped))
        return {"local": params, "hybrid_raw": raw}

    def coefficients(self, params: Any, coefficient_inputs: jnp.ndarray) -> jnp.ndarray:
        local_params = params["local"] if "local" in params else params
        return jnp.asarray(self.model.apply(local_params, jnp.asarray(coefficient_inputs)))

    def energy_density(
        self,
        params: Any,
        coefficient_inputs: jnp.ndarray,
        channels: jnp.ndarray,
    ) -> jnp.ndarray:
        coefficients = self.coefficients(params, coefficient_inputs)
        basis = jnp.asarray(channels)
        if basis.ndim == coefficients.ndim - 1:
            basis = basis[..., None]
        if coefficients.shape != basis.shape:
            raise ValueError(
                "Coefficient/basis channel shape mismatch "
                f"(coefficients={coefficients.shape}, basis={basis.shape})."
            )
        return jnp.einsum("...f,...f->...", coefficients, basis)

    def energy(
        self,
        params: Any,
        coefficient_inputs: jnp.ndarray,
        channels: jnp.ndarray,
        *,
        weights: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        integrand = self.energy_density(params, coefficient_inputs, channels)
        if weights is None:
            return jnp.sum(integrand)
        return jnp.tensordot(jnp.asarray(weights), integrand, axes=(0, 0))

    def hybrid_fraction(self, params: Any) -> jnp.ndarray:
        if self.hybrid_fraction_init is None:
            return jnp.asarray(0.0)
        lower, upper = self.hybrid_fraction_bounds
        return lower + (upper - lower) * jax.nn.sigmoid(params["hybrid_raw"])

    def exact_exchange_energy(self, molecule: Any) -> jnp.ndarray:
        rep_tensor = jnp.asarray(molecule.rep_tensor)
        rdm1 = jnp.asarray(molecule.rdm1)
        if rdm1.ndim == 2:
            rdm1 = jnp.stack([0.5 * rdm1, 0.5 * rdm1], axis=0)

        def spin_exchange(dm_spin):
            exchange_matrix = jnp.einsum(
                "prqs,rs->pq",
                rep_tensor,
                dm_spin,
                precision=Precision.HIGHEST,
            )
            return -0.5 * jnp.einsum(
                "pq,pq->",
                dm_spin,
                exchange_matrix,
                precision=Precision.HIGHEST,
            )

        return jnp.sum(jax.vmap(spin_exchange)(rdm1))


@dataclass(frozen=True)
class _ToyDensityFunctional:
    model: nn.Module
    coefficient_input_fn: Callable[..., jnp.ndarray]
    energy_density_basis_fn: Callable[..., jnp.ndarray]
    density_floor: float = 1e-12
    name: str = "toy_density_xc"
    hybrid_fraction_init: float | None = None
    hybrid_fraction_bounds: tuple[float, float] = (0.0, 1.0)

    def _core(self) -> _ToyCoefficientCore:
        return _ToyCoefficientCore(
            model=self.model,
            name=self.name,
            hybrid_fraction_init=self.hybrid_fraction_init,
            hybrid_fraction_bounds=self.hybrid_fraction_bounds,
        )

    def coefficient_inputs(self, density: jnp.ndarray) -> jnp.ndarray:
        return self.coefficient_input_fn(density, density_floor=self.density_floor)

    def energy_density_basis(self, density: jnp.ndarray) -> jnp.ndarray:
        return self.energy_density_basis_fn(density, density_floor=self.density_floor)

    def init(self, rng: jnp.ndarray, sample_density: jnp.ndarray) -> Any:
        return self._core().init(rng, self.coefficient_inputs(sample_density))

    def init_from_molecule(self, rng: jnp.ndarray, molecule: Any) -> Any:
        return self.init(rng, jnp.asarray(molecule.density()).sum(axis=-1))

    def energy_density(self, params: Any, density: jnp.ndarray) -> jnp.ndarray:
        return self._core().energy_density(
            params,
            self.coefficient_inputs(density),
            self.energy_density_basis(density),
        )

    def energy(self, params: Any, density: jnp.ndarray, weights: jnp.ndarray | None = None) -> jnp.ndarray:
        rho = _clip_toy_density(density, self.density_floor)
        local_channels = rho[..., None] * self.energy_density_basis(rho)
        return self._core().energy(
            params,
            self.coefficient_inputs(rho),
            local_channels,
            weights=weights,
        )

    def hybrid_fraction(self, params: Any) -> jnp.ndarray:
        return self._core().hybrid_fraction(params)

    def exact_exchange_energy(self, molecule: Any) -> jnp.ndarray:
        return self._core().exact_exchange_energy(molecule)

    def energy_from_molecule(self, params: Any, molecule: Any) -> jnp.ndarray:
        total_density = jnp.asarray(molecule.density()).sum(axis=-1)
        return self.energy(params, total_density, molecule.grid.weights) + (
            self.hybrid_fraction(params) * self.exact_exchange_energy(molecule)
        )

    def local_potential(self, params: Any, density: jnp.ndarray) -> jnp.ndarray:
        density = _clip_toy_density(density, self.density_floor)
        flat = density.reshape(-1)

        def local_energy(value):
            return value * self.energy_density(params, value)

        return jax.vmap(jax.grad(local_energy))(flat).reshape(density.shape)

    def local_kernel(self, params: Any, density: jnp.ndarray) -> jnp.ndarray:
        density = _clip_toy_density(density, self.density_floor)
        flat = density.reshape(-1)

        def local_energy(value):
            return value * self.energy_density(params, value)

        return jax.vmap(jax.grad(jax.grad(local_energy)))(flat).reshape(density.shape)

    def bind(self, params: Any) -> _ToyAdiabaticFunctional:
        return _ToyAdiabaticFunctional(
            name=self.name,
            energy_density_fn=lambda density: self.energy_density(params, density),
            exact_exchange_fraction=self.hybrid_fraction(params),
        )


def _make_trainable_functional():
    return _ToyDensityFunctional(
        model=_ToyPointwiseNet(hidden_dims=(), output_dim=1, activation=lambda x: x),
        coefficient_input_fn=lambda density, density_floor=1e-12: jnp.ones(density.shape + (1,)),
        energy_density_basis_fn=lambda density, density_floor=1e-12: density[..., None],
        name="toy_ground_state_xc",
        hybrid_fraction_init=0.25,
    )


def _make_hybrid_only_functional():
    return _ToyDensityFunctional(
        model=_ToyPointwiseNet(hidden_dims=(), output_dim=1, activation=lambda x: x),
        coefficient_input_fn=lambda density, density_floor=1e-12: jnp.ones(
            density.shape + (1,)
        ),
        energy_density_basis_fn=lambda density, density_floor=1e-12: jnp.zeros(
            density.shape + (1,)
        ),
        name="toy_hybrid_xc",
        hybrid_fraction_init=0.25,
    )


def _make_h2_strict_jax_reference():
    return run_molecule_from_spec(
        MoleculeSpecConfig(
            atom="""
            H 0.0 0.0 -0.35
            H 0.0 0.0  0.35
            """,
            basis="sto-3g",
            xc="pbe",
            unit="Angstrom",
            charge=0,
            spin=0,
            cart=True,
            grids_level=0,
        ),
        simulation=SimulationConfig(
            nstates=1,
            scf_backend="jax_rks",
            jax_rks_xc_spec="pbe",
            jax_grid_ao_backend="jax",
            execution_device="cpu",
            jit_tddft=False,
        ),
        compute_local_hfx_features=True,
    )


def test_molecular_e0_training_decreases_loss():
    molecule = _make_toy_molecule()
    functional = _make_trainable_functional()
    datum = MolecularTrainingDatum(
        molecule=molecule,
        target_e0_total_h=jnp.asarray(2.125),
    )
    config = MolecularTrainingConfig(e0_total_mse_weight=1.0)
    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(0),
        molecule,
        optax.adam(0.1),
    )
    train_step = make_molecular_train_step(functional, training_config=config)
    initial_loss, _ = molecular_loss(
        state.params,
        functional,
        datum,
        training_config=config,
    )

    for _ in range(100):
        state, metrics = train_step(state, datum)

    final_loss, _ = molecular_loss(
        state.params,
        functional,
        datum,
        training_config=config,
    )
    assert final_loss < initial_loss
    assert jnp.isfinite(metrics["total_loss"])


def test_grid_density_and_density_matrix_losses_are_explicit():
    molecule = _make_toy_molecule()
    functional = _make_trainable_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(2), molecule)
    grid_datum = MolecularTrainingDatum(
        molecule=molecule,
        target_grid_density=density_on_grid(molecule) * 0.9,
    )
    matrix_datum = MolecularTrainingDatum(
        molecule=molecule,
        target_density_matrix=jnp.asarray(molecule.rdm1).sum(axis=0) * 0.9,
    )

    _, grid_metrics = molecular_loss(
        params,
        functional,
        grid_datum,
        training_config=MolecularTrainingConfig(grid_density_mse_weight=1.0),
    )
    _, matrix_metrics = molecular_loss(
        params,
        functional,
        matrix_datum,
        training_config=MolecularTrainingConfig(density_matrix_mse_weight=1.0),
    )

    assert grid_metrics["grid_density_mse"].shape == (1,)
    assert matrix_metrics["density_matrix_mse"].shape == (1,)
    assert grid_metrics["density_matrix_mse"][0] == 0.0
    assert matrix_metrics["grid_density_mse"][0] == 0.0


def test_density_stationarity_and_dm21_regularizers_use_config_weights():
    molecule = _make_toy_molecule()
    functional = _make_trainable_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(5), molecule)
    datum = MolecularTrainingDatum(molecule=molecule)
    config = MolecularTrainingConfig(
        density_stationarity_weight=0.1,
        dm21_scf_regularization_weight=0.2,
    )

    loss, metrics = molecular_loss(
        params,
        functional,
        datum,
        training_config=config,
    )

    assert jnp.isfinite(loss)
    assert metrics["density_stationarity_loss"][0] >= 0.0
    assert metrics["dm21_scf_loss"][0] >= 0.0


def test_orbital_energy_loss_uses_explicit_target_and_weights():
    molecule = _make_hybrid_toy_molecule()
    functional = _make_hybrid_only_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(15), molecule)
    target = jnp.asarray(molecule.mo_energy) + 0.1
    datum = MolecularTrainingDatum(
        molecule=molecule,
        target_orbital_energies=target,
        target_orbital_occupations=molecule.mo_occ,
    )
    config = MolecularTrainingConfig(
        mode="self_consistent",
        orbital_energy_mse_weight=1.0,
        orbital_energy_window=1,
        scf_max_cycle=12,
        scf_damping=0.5,
    )

    loss, metrics = molecular_loss(
        params,
        functional,
        datum,
        training_config=config,
    )

    assert jnp.isfinite(loss)
    assert metrics["orbital_energy_loss"][0] >= 0.0
    assert metrics["orbital_energy_mse"].shape == (1,)


def test_hybrid_energy_and_excitation_remain_differentiable():
    molecule = _make_hybrid_toy_molecule()
    functional = _make_hybrid_only_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(1), molecule)

    energy_value, energy_grad = jax.value_and_grad(
        lambda p: predict_ground_state_total_energy(p, functional, molecule)
    )(params)
    gap_value, gap_grad = jax.value_and_grad(
        lambda p: predict_excitation_energies(
            p,
            functional,
            molecule,
            nstates=1,
            use_tda=True,
        )[0]
    )(params)

    assert jnp.isfinite(energy_value)
    assert jnp.isfinite(gap_value)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(energy_grad))
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(gap_grad))


def test_e0_per_electron_normalization_changes_only_e0_component():
    molecule = _make_overlap_toy_molecule()
    functional = _make_trainable_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(8), molecule)
    predicted = predict_ground_state_total_energy(params, functional, molecule)
    datum = MolecularTrainingDatum(
        molecule=molecule,
        target_e0_total_h=predicted + 0.4,
    )

    raw_loss, _ = molecular_loss(
        params,
        functional,
        datum,
        training_config=MolecularTrainingConfig(e0_total_mse_weight=1.0),
    )
    normalized_loss, _ = molecular_loss(
        params,
        functional,
        datum,
        training_config=MolecularTrainingConfig(
            e0_total_mse_weight=1.0,
            e0_normalization="per_electron",
        ),
    )

    assert normalized_loss < raw_loss
