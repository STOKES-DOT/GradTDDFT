from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from jax.lax import Precision

import td_graddft.training.targets as training_targets
from td_graddft import HARTREE_TO_EV, lorentzian_spectrum
from td_graddft.neural_xc import make_neural_xc_functional
from td_graddft.training import (
    GroundStateDatum,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    dm21_scf_regularization_delta_energy,
    density_matrix_matching_penalty,
    density_matching_penalty,
    density_on_grid,
    density_on_grid_spin_resolved,
    density_stationarity_penalty,
    ground_state_mse_loss,
    make_fixed_density_predictor,
    make_ground_state_loss,
    make_ground_state_train_step,
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


def test_batched_self_consistent_path_rejects_open_shell_unrestricted_dataset():
    def _molecule(mo_occ, rdm1, *, nocc_alpha, nocc_beta):
        return UnrestrictedMolecule(
            ao=jnp.array([[1.0, 0.5], [0.5, 1.0]]),
            grid=QuadratureGrid(weights=jnp.array([1.0, 1.0])),
            dipole_integrals=jnp.zeros((3, 2, 2)),
            rep_tensor=jnp.zeros((2, 2, 2, 2)),
            mo_coeff=jnp.stack([jnp.eye(2), jnp.eye(2)], axis=0),
            mo_occ=mo_occ,
            mo_energy=jnp.array([[0.0, 1.0], [0.0, 1.0]]),
            rdm1=rdm1,
            h1e=jnp.zeros((2, 2)),
            nuclear_repulsion=jnp.asarray(0.0),
            overlap_matrix=jnp.eye(2),
            nocc_alpha=nocc_alpha,
            nocc_beta=nocc_beta,
        )

    closed_shell = _molecule(
        jnp.array([[1.0, 0.0], [1.0, 0.0]]),
        jnp.array(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 0.0]],
            ]
        ),
        nocc_alpha=1,
        nocc_beta=1,
    )
    open_shell = _molecule(
        jnp.array([[1.0, 0.0], [0.0, 0.0]]),
        jnp.array(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ),
        nocc_alpha=1,
        nocc_beta=0,
    )
    cfg = GroundStateTrainingConfig(mode="self_consistent")

    assert training_targets._can_use_batched_self_consistent_ground_state_path(
        [
            GroundStateDatum(closed_shell, jnp.array(0.0)),
            GroundStateDatum(closed_shell, jnp.array(0.0)),
        ],
        cfg,
        predictor=None,
    )
    assert not training_targets._can_use_batched_self_consistent_ground_state_path(
        [
            GroundStateDatum(open_shell, jnp.array(0.0)),
            GroundStateDatum(open_shell, jnp.array(0.0)),
        ],
        cfg,
        predictor=None,
    )


def test_pointwise_dataset_loss_calls_single_datum_loss(monkeypatch):
    calls = []

    def fake_loss(params, functional, data, *, training_config=None, predictor=None):
        assert isinstance(data, GroundStateDatum)
        calls.append(float(data.target_total_energy))
        value = jnp.asarray(data.target_total_energy + params["offset"])
        return value, {
            "loss": value,
            "energy_mae": jnp.atleast_1d(value),
            "density_mse": jnp.atleast_1d(value + 1.0),
            "density_penalty": jnp.atleast_1d(value + 2.0),
            "density_matrix_mse": jnp.atleast_1d(value + 3.0),
            "density_matrix_penalty": jnp.atleast_1d(value + 4.0),
            "scf_converged": jnp.atleast_1d(1.0),
            "scf_cycles": jnp.atleast_1d(value + 5.0),
            "scf_selected_cycle": jnp.atleast_1d(value + 6.0),
            "scf_best_cycle": jnp.atleast_1d(value + 7.0),
            "scf_final_rms_density": jnp.atleast_1d(value + 8.0),
            "scf_selected_rms_density": jnp.atleast_1d(value + 9.0),
            "scf_best_rms_density": jnp.atleast_1d(value + 10.0),
        }

    monkeypatch.setattr(training_targets, "ground_state_mse_loss", fake_loss)
    dataset = (
        GroundStateDatum(SimpleNamespace(), jnp.asarray(1.0)),
        GroundStateDatum(SimpleNamespace(), jnp.asarray(3.0)),
    )

    loss, metrics = training_targets.ground_state_mse_loss_pointwise_dataset(
        {"offset": jnp.asarray(0.5)},
        object(),
        dataset,
        training_config=GroundStateTrainingConfig(mode="self_consistent"),
    )

    np.testing.assert_allclose(float(loss), 2.5)
    np.testing.assert_allclose(np.asarray(metrics["energy_mae"]), np.asarray([1.5, 3.5]))
    np.testing.assert_allclose(np.asarray(metrics["scf_cycles_mean"]), np.asarray([7.5]))
    np.testing.assert_allclose(np.asarray(metrics["scf_cycles_max"]), np.asarray([8.5]))
    assert calls == [1.0, 3.0]


def test_pointwise_dataset_loss_drops_unconverged_weights(monkeypatch):
    def fake_loss(params, functional, data, *, training_config=None, predictor=None):
        del params, functional, training_config, predictor
        value = jnp.asarray(data.target_total_energy)
        converged = 1.0 if float(value) < 2.0 else 0.0
        return value, {"loss": value, "scf_converged": jnp.atleast_1d(converged)}

    monkeypatch.setattr(training_targets, "ground_state_mse_loss", fake_loss)
    dataset = (
        GroundStateDatum(SimpleNamespace(), jnp.asarray(1.0)),
        GroundStateDatum(SimpleNamespace(), jnp.asarray(3.0)),
    )

    loss, _ = training_targets.ground_state_mse_loss_pointwise_dataset(
        {},
        object(),
        dataset,
        training_config=GroundStateTrainingConfig(
            mode="self_consistent",
            scf_require_convergence=True,
        ),
    )

    np.testing.assert_allclose(float(loss), 1.0)


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


def test_ground_state_training_decreases_loss_and_produces_tddft_energies():
    molecule = _make_toy_molecule()
    functional = _make_trainable_functional()
    datum = GroundStateDatum(molecule=molecule, target_total_energy=jnp.array(2.125))

    tx = optax.adam(0.1)
    state = create_train_state_from_molecule(functional, jax.random.PRNGKey(0), molecule, tx)
    train_step = make_ground_state_train_step(functional)

    initial_loss, _ = ground_state_mse_loss(state.params, functional, datum)

    for _ in range(200):
        state, metrics = train_step(state, datum)

    final_loss, _ = ground_state_mse_loss(state.params, functional, datum)
    predicted_energy = predict_ground_state_total_energy(state.params, functional, molecule)
    excitation = predict_excitation_energies(state.params, functional, molecule, nstates=1)

    assert final_loss < initial_loss
    # With MAE-only supervision, the toy setup still converges but plateaus at
    # a slightly larger finite scale than the previous MSE-dominated default.
    assert final_loss < 7e-2
    assert jnp.allclose(predicted_energy, 2.125, atol=7e-2)
    assert excitation.shape == (1,)
    assert jnp.allclose(excitation[0], jnp.sqrt(3.0), atol=1e-1)


def test_hybrid_exact_exchange_energy_and_excitation_are_differentiable():
    molecule = _make_hybrid_toy_molecule()
    functional = _make_hybrid_only_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(1), molecule)

    exact_exchange = functional.exact_exchange_energy(molecule)
    xc_energy = functional.energy_from_molecule(params, molecule)
    excitation = predict_excitation_energies(
        params,
        functional,
        molecule,
        nstates=1,
        use_tda=False,
    )

    def excitation_energy_from_hybrid(raw):
        varied_params = {
            "local": params["local"],
            "hybrid_raw": raw,
        }
        return predict_excitation_energies(
            varied_params,
            functional,
            molecule,
            nstates=1,
            use_tda=False,
        )[0]

    grad = jax.grad(excitation_energy_from_hybrid)(params["hybrid_raw"])

    assert jnp.allclose(functional.hybrid_fraction(params), 0.25, atol=1e-6)
    assert jnp.allclose(exact_exchange, -4.0, atol=1e-6)
    assert jnp.allclose(xc_energy, -1.0, atol=1e-6)
    assert jnp.allclose(excitation[0], 0.5, atol=1e-6)
    assert jnp.isfinite(grad)
    assert grad < -1e-6


def test_predict_excitation_energies_dispatches_unrestricted_facades(monkeypatch):
    calls: list[tuple[str, Any]] = []

    class FakeTDA:
        def __init__(self, molecule, **kwargs):
            calls.append(("tda_init", {"molecule": molecule, **kwargs}))

        def kernel(self, nstates=None):
            calls.append(("tda_kernel", nstates))
            return SimpleNamespace(excitation_energies=jnp.asarray([0.31]))

    class FakeTDDFT:
        def __init__(self, molecule, **kwargs):
            calls.append(("tddft_init", {"molecule": molecule, **kwargs}))

        def kernel(self, nstates=None):
            calls.append(("tddft_kernel", nstates))
            return SimpleNamespace(excitation_energies=jnp.asarray([0.47]))

    monkeypatch.setattr(training_targets.tdscf, "TDA", FakeTDA)
    monkeypatch.setattr(training_targets.tdscf, "TDDFT", FakeTDDFT)

    molecule = UnrestrictedMolecule(
        ao=jnp.array([[1.0, 0.5], [0.5, 1.0]]),
        grid=QuadratureGrid(weights=jnp.array([1.0, 1.0])),
        dipole_integrals=jnp.zeros((3, 2, 2)),
        rep_tensor=jnp.zeros((2, 2, 2, 2)),
        mo_coeff=jnp.stack([jnp.eye(2), jnp.eye(2)], axis=0),
        mo_occ=jnp.array([[1.0, 0.0], [0.0, 0.0]]),
        mo_energy=jnp.array([[0.0, 1.0], [0.2, 1.2]]),
        rdm1=jnp.array(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ),
        h1e=jnp.zeros((2, 2)),
        nuclear_repulsion=jnp.asarray(0.0),
        overlap_matrix=jnp.eye(2),
        nocc_alpha=1,
        nocc_beta=0,
    )

    tda_excitation = predict_excitation_energies(
        {},
        object(),
        molecule,
        nstates=1,
        use_tda=True,
    )
    tddft_excitation = predict_excitation_energies(
        {},
        object(),
        molecule,
        nstates=1,
        use_tda=False,
    )

    assert jnp.allclose(tda_excitation, jnp.asarray([0.31]))
    assert jnp.allclose(tddft_excitation, jnp.asarray([0.47]))
    assert ("tda_kernel", 1) in calls
    assert ("tddft_kernel", 1) in calls
    assert calls[0][1]["molecule"] is molecule
    assert calls[0][1]["eigensolver"] == "auto"


def test_predict_excitation_energies_uses_davidson_for_traced_params(monkeypatch):
    calls: list[dict[str, Any]] = []

    class FakeTDA:
        def __init__(self, molecule, **kwargs):
            calls.append({"molecule": molecule, **kwargs})

        def kernel(self, nstates=None):
            return SimpleNamespace(excitation_energies=jnp.asarray([0.31]))

    monkeypatch.setattr(training_targets.tdscf, "TDA", FakeTDA)

    molecule = SimpleNamespace()

    @jax.jit
    def traced_excitation(weight):
        return predict_excitation_energies(
            {"weight": weight},
            object(),
            molecule,
            nstates=1,
            use_tda=True,
        )

    excitation = traced_excitation(jnp.asarray(1.0))

    assert jnp.allclose(excitation, jnp.asarray([0.31]))
    assert calls[0]["eigensolver"] == "davidson"
    assert calls[0].get("davidson_max_subspace") is None


def test_strict_jax_self_consistent_losses_have_finite_gradients():
    reference = _make_h2_strict_jax_reference()
    molecule = reference.molecule
    functional = make_neural_xc_functional(
        semilocal_xc=("gga_x_pbe", "gga_c_pbe"),
        hidden_dims=(8, 8),
        name="strict_jax_h2_gradcheck",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(23), molecule)

    common_cfg = dict(
        mode="self_consistent",
        scf_gradient_mode="impl",
        scf_max_cycle=6,
        scf_damping=0.2,
        scf_conv_tol_density=1e-7,
    )
    target_energy = jnp.asarray(molecule.mf_energy)
    target_grid_ev = jnp.linspace(5.0, 25.0, 128)
    target_curve = lorentzian_spectrum(
        reference.energies_au[:1] * HARTREE_TO_EV,
        reference.oscillator_strengths[:1],
        target_grid_ev,
        eta=0.15,
    )

    cases = [
        (
            GroundStateDatum(
                molecule=molecule,
                target_total_energy=target_energy,
            ),
            GroundStateTrainingConfig(
                energy_mse_weight=0.0,
                energy_mae_weight=1.0,
                **common_cfg,
            ),
        ),
        (
            GroundStateDatum(
                molecule=molecule,
                target_total_energy=target_energy,
                target_excitation_energies=reference.energies_au[:1],
                excitation_constraint_weight=1.0,
                excitation_constraint_nstates=1,
            ),
            GroundStateTrainingConfig(
                energy_mse_weight=0.0,
                energy_mae_weight=0.0,
                excitation_constraint_use_tda=True,
                excitation_mse_weight=0.0,
                excitation_mae_weight=1.0,
                **common_cfg,
            ),
        ),
        (
            GroundStateDatum(
                molecule=molecule,
                target_total_energy=target_energy,
                target_spectrum_grid_ev=target_grid_ev,
                target_spectrum_curve=target_curve,
                spectrum_constraint_weight=1.0,
                spectrum_constraint_nstates=1,
            ),
            GroundStateTrainingConfig(
                energy_mse_weight=0.0,
                energy_mae_weight=0.0,
                spectrum_constraint_use_tda=True,
                spectrum_constraint_eta_ev=0.15,
                spectrum_mse_weight=1.0,
                spectrum_mae_weight=0.0,
                **common_cfg,
            ),
        ),
    ]

    for datum, cfg in cases:
        loss_fn = lambda p, _datum=datum, _cfg=cfg: ground_state_mse_loss(  # noqa: E731
            p,
            functional,
            _datum,
            training_config=_cfg,
        )[0]
        value, grads = jax.value_and_grad(loss_fn)(params)
        leaves = jax.tree_util.tree_leaves(grads)
        assert jnp.isfinite(value)
        assert all(jnp.all(jnp.isfinite(jnp.asarray(leaf))) for leaf in leaves)
        assert max(float(jnp.max(jnp.abs(jnp.asarray(leaf)))) for leaf in leaves) > 0.0


def test_density_matching_penalty_is_finite_and_reported_in_loss():
    molecule = _make_toy_molecule()
    functional = _make_trainable_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(2), molecule)
    constrained = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.array(2.125),
        density_constraint_weight=0.1,
    )

    penalty = density_matching_penalty(params, functional, molecule)
    loss_constrained, metrics = ground_state_mse_loss(params, functional, constrained)

    assert jnp.isfinite(penalty)
    assert penalty >= 0.0
    assert metrics["density_penalty"].shape == (1,)
    assert metrics["density_penalty"][0] >= 0.0
    assert metrics["density_mse"].shape == (1,)
    assert metrics["density_mse"][0] >= 0.0


def test_density_matrix_matching_penalty_is_separate_from_grid_density_loss():
    molecule = _make_toy_molecule()
    target_density_matrix = jnp.asarray(molecule.rdm1).sum(axis=0) * 0.75
    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.array(2.125),
        target_density_matrix=target_density_matrix,
        density_constraint_weight=0.1,
        density_matrix_constraint_weight=0.2,
    )

    penalty = density_matrix_matching_penalty(
        molecule,
        target_density_matrix=target_density_matrix,
    )
    _, metrics = ground_state_mse_loss(
        {},
        _make_trainable_functional(),
        datum,
        training_config=GroundStateTrainingConfig(
            mode="self_consistent",
            energy_mse_weight=0.0,
            energy_mae_weight=0.0,
        ),
        predictor=lambda params, mol: (jnp.asarray(2.125), mol),
    )

    assert jnp.isfinite(penalty)
    assert penalty > 0.0
    assert metrics["density_penalty"].shape == (1,)
    assert metrics["density_matrix_penalty"].shape == (1,)
    assert metrics["density_mse"].shape == (1,)
    assert metrics["density_matrix_mse"].shape == (1,)
    assert metrics["density_matrix_penalty"][0] == jnp.asarray(0.2) * metrics["density_matrix_mse"][0]


def test_spin_resolved_density_helpers_and_penalty_are_available():
    molecule = _make_toy_molecule()
    functional = _make_trainable_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(6), molecule)

    spin_density = density_on_grid_spin_resolved(molecule)
    total_density = density_on_grid(molecule)
    predicted_spin_density = predict_ground_state_density(
        params,
        functional,
        molecule,
        spin_resolved=True,
    )
    spin_penalty = density_matching_penalty(
        params,
        functional,
        molecule,
        training_config=GroundStateTrainingConfig(density_supervision="spin_resolved"),
    )

    assert spin_density.shape[-1] == 2
    assert jnp.allclose(spin_density.sum(axis=-1), total_density, atol=1e-6)
    assert predicted_spin_density.shape == spin_density.shape
    assert jnp.isfinite(spin_penalty)
    assert spin_penalty >= 0.0


def test_density_stationarity_penalty_is_finite_and_affects_loss():
    molecule = _make_toy_molecule()
    functional = _make_trainable_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(5), molecule)
    plain = GroundStateDatum(molecule=molecule, target_total_energy=jnp.array(2.125))
    constrained = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.array(2.125),
        stationarity_constraint_weight=0.1,
    )

    penalty = density_stationarity_penalty(params, functional, molecule)
    loss_plain, _ = ground_state_mse_loss(params, functional, plain)
    loss_constrained, metrics = ground_state_mse_loss(params, functional, constrained)

    assert jnp.isfinite(penalty)
    assert penalty >= 0.0
    assert loss_constrained >= loss_plain
    assert metrics["stationarity_penalty"].shape == (1,)


def test_dm21_scf_regularization_penalty_is_finite_and_affects_loss():
    molecule = _make_toy_molecule()
    functional = _make_trainable_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(13), molecule)
    plain = GroundStateDatum(molecule=molecule, target_total_energy=jnp.array(2.125))
    constrained = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.array(2.125),
        dm21_scf_regularization_weight=0.1,
    )

    delta_e = dm21_scf_regularization_delta_energy(params, functional, molecule)
    loss_plain, _ = ground_state_mse_loss(params, functional, plain)
    loss_constrained, metrics = ground_state_mse_loss(params, functional, constrained)

    assert jnp.isfinite(delta_e)
    assert metrics["dm21_scf_delta_energy"].shape == (1,)
    assert metrics["dm21_scf_mse"].shape == (1,)
    assert metrics["dm21_scf_penalty"].shape == (1,)
    assert metrics["dm21_scf_mse"][0] >= 0.0
    assert metrics["dm21_scf_penalty"][0] >= 0.0
    assert loss_constrained >= loss_plain


def test_orbital_energy_matching_penalty_is_finite_and_affects_loss():
    molecule = _make_hybrid_toy_molecule()
    functional = _make_hybrid_only_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(15), molecule)
    target_energies = jnp.asarray(molecule.mo_energy) + jnp.array([[0.0, 0.2], [0.0, 0.2]])

    mse, mae, residual, mask = orbital_energy_matching_penalty(
        molecule,
        target_orbital_energies=target_energies,
        target_orbital_occupations=molecule.mo_occ,
        window=1,
    )
    plain = GroundStateDatum(molecule=molecule, target_total_energy=jnp.array(-1.0))
    constrained = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.array(-1.0),
        target_orbital_energies=target_energies,
        target_orbital_occupations=molecule.mo_occ,
        orbital_energy_constraint_weight=0.5,
        orbital_energy_constraint_window=1,
    )
    loss_plain, _ = ground_state_mse_loss(
        params,
        functional,
        plain,
        training_config=GroundStateTrainingConfig(
            mode="self_consistent",
            scf_max_cycle=12,
            scf_damping=0.5,
        ),
    )
    loss_constrained, metrics = ground_state_mse_loss(
        params,
        functional,
        constrained,
        training_config=GroundStateTrainingConfig(
            mode="self_consistent",
            scf_max_cycle=12,
            scf_damping=0.5,
        ),
    )

    assert jnp.isfinite(mse)
    assert jnp.isfinite(mae)
    assert residual.shape == (2,)
    assert int(jnp.sum(mask)) == 2
    assert metrics["orbital_energy_penalty"].shape == (1,)
    assert metrics["orbital_energy_mse"].shape == (1,)
    assert metrics["orbital_energy_mae"].shape == (1,)
    assert metrics["orbital_energy_penalty"][0] > 0.0
    assert loss_constrained > loss_plain


def test_xc_kernel_matching_penalty_is_finite_and_reported_in_loss():
    molecule = _make_toy_molecule()
    functional = _make_trainable_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(11), molecule)
    target_kernel = functional.local_kernel(params, density_on_grid(molecule)) + 0.25

    plain = GroundStateDatum(molecule=molecule, target_total_energy=jnp.array(2.125))
    constrained = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.array(2.125),
        target_xc_kernel=target_kernel,
        xc_kernel_constraint_weight=0.1,
    )

    penalty = xc_kernel_matching_penalty(
        params,
        functional,
        molecule,
        target_xc_kernel=target_kernel,
    )
    loss_plain, _ = ground_state_mse_loss(params, functional, plain)
    loss_constrained, metrics = ground_state_mse_loss(params, functional, constrained)

    assert jnp.isfinite(penalty)
    assert penalty >= 0.0
    assert metrics["xc_kernel_penalty"].shape == (1,)
    assert metrics["xc_kernel_mse"].shape == (1,)
    assert metrics["xc_kernel_penalty"][0] > 0.0
    assert metrics["xc_kernel_mse"][0] > 0.0
    assert loss_constrained > loss_plain


def test_xc_kernel_penalty_uses_full_bind_response_not_scf_shortcut():
    molecule = _make_toy_molecule()
    ngrids = int(molecule.grid.weights.shape[0])

    class _Bound:
        def __init__(self, kernel):
            self._kernel = kernel

        def grid_kernel(self, _molecule):
            return self._kernel

    class _Functional:
        def bind_to_molecule_for_scf(self, _params, _molecule):
            return _Bound(jnp.zeros((ngrids,)))

        def bind_to_molecule(self, _params, _molecule):
            return _Bound(jnp.ones((ngrids,)))

    params = {}
    functional = _Functional()
    target = jnp.ones((ngrids,))
    penalty = xc_kernel_matching_penalty(
        params,
        functional,
        molecule,
        target_xc_kernel=target,
    )

    assert jnp.isfinite(penalty)
    assert jnp.allclose(penalty, 0.0, atol=1e-10)


def test_s1_constraint_penalty_is_finite_and_affects_loss():
    molecule = _make_toy_molecule()
    functional = _make_trainable_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(8), molecule)

    predicted_s1 = predict_excitation_energies(
        params,
        functional,
        molecule,
        nstates=1,
        use_tda=True,
    )[0]
    plain = GroundStateDatum(molecule=molecule, target_total_energy=jnp.array(2.125))
    constrained = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.array(2.125),
        target_s1_energy=predicted_s1 + 0.1,
        s1_constraint_weight=0.5,
    )
    cfg = GroundStateTrainingConfig(s1_constraint_use_tda=True)

    loss_plain, _ = ground_state_mse_loss(
        params,
        functional,
        plain,
        training_config=cfg,
    )
    loss_constrained, metrics = ground_state_mse_loss(
        params,
        functional,
        constrained,
        training_config=cfg,
    )

    assert metrics["s1_penalty"].shape == (1,)
    assert metrics["s1_mse"].shape == (1,)
    assert metrics["s1_mae"].shape == (1,)
    assert metrics["s1_predicted"].shape == (1,)
    assert metrics["s1_target"].shape == (1,)
    assert metrics["s1_penalty"][0] > 0.0
    assert metrics["s1_mse"][0] > 0.0
    assert metrics["s1_mae"][0] > 0.0
    expected_s1_penalty = 0.5 * (metrics["s1_mse"][0] + metrics["s1_mae"][0])
    assert jnp.allclose(metrics["s1_penalty"][0], expected_s1_penalty, atol=1e-8)
    assert loss_constrained > loss_plain


def test_s1_only_loss_skips_ground_state_total_energy_assembly(monkeypatch):
    molecule = _make_toy_molecule()
    functional = _make_trainable_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(82), molecule)

    predicted_s1 = predict_excitation_energies(
        params,
        functional,
        molecule,
        nstates=1,
        use_tda=True,
    )[0]
    constrained = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.array(2.125),
        target_s1_energy=predicted_s1 + 0.1,
        s1_constraint_weight=0.5,
    )
    cfg = GroundStateTrainingConfig(
        energy_mse_weight=0.0,
        energy_mae_weight=0.0,
        s1_constraint_use_tda=True,
    )

    def _unexpected_energy_call(*_args, **_kwargs):
        raise AssertionError("ground-state total energy should not be assembled for pure S1 loss")

    monkeypatch.setattr(
        training_targets,
        "_predict_ground_state_total_energy_from_molecule",
        _unexpected_energy_call,
    )

    loss, metrics = ground_state_mse_loss(
        params,
        functional,
        constrained,
        training_config=cfg,
    )

    assert jnp.isfinite(loss)
    assert metrics["predicted_total_energies"].shape == (1,)
    assert jnp.isnan(metrics["predicted_total_energies"][0])
    assert metrics["s1_penalty"][0] > 0.0


def test_s1_only_loss_solves_three_roots_but_uses_first(monkeypatch):
    molecule = _make_toy_molecule()
    calls: list[tuple[int, bool]] = []

    def fake_solve_excited_states(
        params,
        functional,
        molecule_arg,
        *,
        nstates: int,
        use_tda: bool,
    ):
        del params, functional, molecule_arg
        calls.append((int(nstates), bool(use_tda)))
        return SimpleNamespace(excitation_energies=jnp.asarray([0.25, 0.50, 0.75]))

    monkeypatch.setattr(
        training_targets,
        "_solve_excited_states",
        fake_solve_excited_states,
    )

    constrained = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.array(2.125),
        target_s1_energy=jnp.asarray(0.35),
        s1_constraint_weight=0.5,
    )
    cfg = GroundStateTrainingConfig(
        energy_mse_weight=0.0,
        energy_mae_weight=0.0,
        s1_constraint_use_tda=True,
    )

    loss, metrics = ground_state_mse_loss(
        {},
        object(),
        constrained,
        training_config=cfg,
    )

    assert calls == [(3, True)]
    assert jnp.isfinite(loss)
    assert metrics["s1_predicted"][0] == 0.25
    assert metrics["s1_penalty"][0] > 0.0


def test_s1_only_loss_skips_oscillator_strength_assembly(monkeypatch):
    molecule = _make_toy_molecule()
    functional = _make_trainable_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(182), molecule)

    predicted_s1 = predict_excitation_energies(
        params,
        functional,
        molecule,
        nstates=1,
        use_tda=True,
    )[0]
    constrained = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.array(2.125),
        target_s1_energy=predicted_s1 + 0.1,
        s1_constraint_weight=0.5,
    )
    cfg = GroundStateTrainingConfig(
        energy_mse_weight=0.0,
        energy_mae_weight=0.0,
        s1_constraint_use_tda=True,
    )

    def _unexpected_oscillator_call(*_args, **_kwargs):
        raise AssertionError("oscillator strengths should not be assembled for pure S1 loss")

    monkeypatch.setattr(
        training_targets,
        "oscillator_strengths",
        _unexpected_oscillator_call,
    )

    loss, metrics = ground_state_mse_loss(
        params,
        functional,
        constrained,
        training_config=cfg,
    )

    assert jnp.isfinite(loss)
    assert metrics["s1_penalty"][0] > 0.0
    assert metrics["oscillator_strength_penalty"].shape == (1,)
    assert metrics["oscillator_strength_penalty"][0] == 0.0


def test_first_excited_total_energy_constraint_penalty_is_finite_and_affects_loss():
    molecule = _make_toy_molecule()
    functional = _make_trainable_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(81), molecule)

    predicted_ground = predict_ground_state_total_energy(params, functional, molecule)
    predicted_s1 = predict_excitation_energies(
        params,
        functional,
        molecule,
        nstates=1,
        use_tda=True,
    )[0]
    predicted_e1_total = predicted_ground + predicted_s1
    cfg = GroundStateTrainingConfig(
        energy_mse_weight=0.0,
        energy_mae_weight=0.0,
        s1_constraint_use_tda=True,
    )
    plain = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.array(-99.0),
    )
    constrained = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.array(12345.0),
        target_first_excited_total_energy=predicted_e1_total + 0.1,
        first_excited_total_energy_constraint_weight=0.5,
    )

    loss_plain, _ = ground_state_mse_loss(
        params,
        functional,
        plain,
        training_config=cfg,
    )
    loss_constrained, metrics = ground_state_mse_loss(
        params,
        functional,
        constrained,
        training_config=cfg,
    )

    assert metrics["first_excited_total_penalty"].shape == (1,)
    assert metrics["first_excited_total_mse"].shape == (1,)
    assert metrics["first_excited_total_predicted"].shape == (1,)
    assert metrics["first_excited_total_target"].shape == (1,)
    assert metrics["first_excited_total_penalty"][0] > 0.0
    assert metrics["first_excited_total_mse"][0] > 0.0
    assert loss_constrained > loss_plain


def test_multistate_excitation_constraint_penalty_is_finite_and_affects_loss():
    molecule = _make_toy_molecule()
    functional = _make_trainable_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(9), molecule)

    predicted_s1 = predict_excitation_energies(
        params,
        functional,
        molecule,
        nstates=1,
        use_tda=True,
    )[0]
    plain = GroundStateDatum(molecule=molecule, target_total_energy=jnp.array(2.125))
    constrained = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.array(2.125),
        target_excitation_energies=jnp.asarray([predicted_s1 + 0.2, predicted_s1 + 0.4]),
        excitation_constraint_weight=0.5,
        excitation_constraint_nstates=2,
    )
    cfg = GroundStateTrainingConfig(
        excitation_constraint_use_tda=True,
        excitation_mse_weight=0.0,
        excitation_mae_weight=1.0,
    )

    loss_plain, _ = ground_state_mse_loss(
        params,
        functional,
        plain,
        training_config=cfg,
    )
    loss_constrained, metrics = ground_state_mse_loss(
        params,
        functional,
        constrained,
        training_config=cfg,
    )

    assert metrics["excitation_penalty"].shape == (1,)
    assert metrics["excitation_mse"].shape == (1,)
    assert metrics["excitation_mae"].shape == (1,)
    assert metrics["excitation_predicted"].shape == (1,)
    assert metrics["excitation_target"].shape == (1,)
    assert metrics["excitation_penalty"][0] > 0.0
    assert metrics["excitation_mse"][0] > 0.0
    assert jnp.allclose(metrics["excitation_penalty"][0], 0.5 * metrics["excitation_mae"][0])
    assert loss_constrained > loss_plain


def test_excitation_only_loss_ignores_ground_state_energy_target():
    molecule = _make_toy_molecule()
    functional = _make_trainable_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(10), molecule)

    predicted_states = predict_excitation_energies(
        params,
        functional,
        molecule,
        nstates=2,
        use_tda=True,
    )
    cfg = GroundStateTrainingConfig(
        energy_mse_weight=0.0,
        energy_mae_weight=0.0,
        excitation_constraint_use_tda=True,
        excitation_mse_weight=0.0,
        excitation_mae_weight=1.0,
    )
    datum_a = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.array(-99.0),
        target_excitation_energies=predicted_states + jnp.asarray([0.2, 0.4]),
        excitation_constraint_weight=0.5,
        excitation_constraint_nstates=2,
    )
    datum_b = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.array(12345.0),
        target_excitation_energies=predicted_states + jnp.asarray([0.2, 0.4]),
        excitation_constraint_weight=0.5,
        excitation_constraint_nstates=2,
    )

    loss_a, metrics_a = ground_state_mse_loss(
        params,
        functional,
        datum_a,
        training_config=cfg,
    )
    loss_b, metrics_b = ground_state_mse_loss(
        params,
        functional,
        datum_b,
        training_config=cfg,
    )

    assert jnp.allclose(loss_a, loss_b, atol=1e-10)
    assert jnp.allclose(metrics_a["excitation_penalty"], metrics_b["excitation_penalty"], atol=1e-10)
    assert metrics_a["excitation_penalty"][0] > 0.0


def test_spectrum_constraint_penalty_is_finite_and_affects_loss():
    molecule = _make_toy_molecule()
    functional = _make_trainable_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(11), molecule)

    grid_ev = jnp.linspace(0.0, 40.0, 64)
    predicted_curve = predict_excitation_spectrum(
        params,
        functional,
        molecule,
        grid_ev=grid_ev,
        nstates=1,
        use_tda=True,
        eta_ev=0.2,
    )
    cfg = GroundStateTrainingConfig(
        energy_mse_weight=0.0,
        energy_mae_weight=0.0,
        spectrum_constraint_use_tda=True,
        spectrum_constraint_eta_ev=0.2,
        spectrum_mse_weight=0.0,
        spectrum_mae_weight=1.0,
    )
    plain = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.array(0.0),
    )
    constrained = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.array(0.0),
        target_spectrum_grid_ev=grid_ev,
        target_spectrum_curve=predicted_curve + 0.1,
        spectrum_constraint_weight=0.5,
        spectrum_constraint_nstates=1,
    )

    loss_plain, _ = ground_state_mse_loss(
        params,
        functional,
        plain,
        training_config=cfg,
    )
    loss_constrained, metrics = ground_state_mse_loss(
        params,
        functional,
        constrained,
        training_config=cfg,
    )

    assert metrics["spectrum_penalty"].shape == (1,)
    assert metrics["spectrum_mse"].shape == (1,)
    assert metrics["spectrum_mae"].shape == (1,)
    assert metrics["spectrum_penalty"][0] > 0.0
    assert metrics["spectrum_mse"][0] > 0.0
    assert jnp.allclose(metrics["spectrum_penalty"][0], 0.5 * metrics["spectrum_mae"][0])
    assert loss_constrained > loss_plain


def test_oscillator_strength_constraint_penalty_is_finite_and_affects_loss():
    molecule = _make_toy_molecule()
    functional = _make_trainable_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(12), molecule)

    predicted_strengths = predict_oscillator_strengths(
        params,
        functional,
        molecule,
        nstates=2,
        use_tda=True,
    )
    cfg = GroundStateTrainingConfig(
        energy_mse_weight=0.0,
        energy_mae_weight=0.0,
        oscillator_strength_constraint_use_tda=True,
        oscillator_strength_mse_weight=0.0,
        oscillator_strength_mae_weight=1.0,
    )
    plain = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.array(0.0),
    )
    constrained = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.array(0.0),
        target_oscillator_strengths=predicted_strengths + jnp.asarray([0.1, 0.2]),
        oscillator_strength_constraint_weight=0.5,
        oscillator_strength_constraint_nstates=2,
    )

    loss_plain, _ = ground_state_mse_loss(
        params,
        functional,
        plain,
        training_config=cfg,
    )
    loss_constrained, metrics = ground_state_mse_loss(
        params,
        functional,
        constrained,
        training_config=cfg,
    )

    assert metrics["oscillator_strength_penalty"].shape == (1,)
    assert metrics["oscillator_strength_mse"].shape == (1,)
    assert metrics["oscillator_strength_mae"].shape == (1,)
    assert metrics["oscillator_strength_predicted"].size > 0
    assert metrics["oscillator_strength_target"].size > 0
    assert metrics["oscillator_strength_penalty"][0] > 0.0
    assert metrics["oscillator_strength_mse"][0] > 0.0
    assert jnp.allclose(
        metrics["oscillator_strength_penalty"][0],
        0.5 * metrics["oscillator_strength_mae"][0],
    )
    assert loss_constrained > loss_plain


def test_ground_state_predictors_return_energy_density_and_molecule():
    molecule = _make_toy_molecule()
    functional = _make_trainable_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(7), molecule)

    predictor = make_fixed_density_predictor(functional)
    predicted_energy, predicted_molecule = predictor(params, molecule)
    explicit_molecule = predict_ground_state_molecule(params, functional, molecule)
    explicit_density = predict_ground_state_density(params, functional, molecule)

    assert jnp.allclose(
        predicted_energy,
        predict_ground_state_total_energy(params, functional, molecule),
        atol=1e-6,
    )
    assert predicted_molecule.rdm1.shape == molecule.rdm1.shape
    assert jnp.allclose(explicit_molecule.rdm1, predicted_molecule.rdm1, atol=1e-6)
    assert explicit_density.shape == density_on_grid(molecule).shape


def test_bound_ground_state_loss_matches_default_and_accepts_predictor():
    molecule = _make_toy_molecule()
    functional = _make_trainable_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(9), molecule)
    datum = GroundStateDatum(molecule=molecule, target_total_energy=jnp.array(2.125))

    predictor = make_fixed_density_predictor(functional)
    bound_loss = make_ground_state_loss(
        functional,
        training_config=GroundStateTrainingConfig(mode="fixed_density"),
        predictor=predictor,
    )

    default_loss, default_metrics = ground_state_mse_loss(params, functional, datum)
    predictor_loss, predictor_metrics = ground_state_mse_loss(
        params,
        functional,
        datum,
        predictor=predictor,
    )
    bound_loss_value, bound_metrics = bound_loss(params, datum)

    assert jnp.allclose(predictor_loss, default_loss, atol=1e-6)
    assert jnp.allclose(bound_loss_value, default_loss, atol=1e-6)
    assert jnp.allclose(
        predictor_metrics["predicted_total_energies"],
        default_metrics["predicted_total_energies"],
        atol=1e-6,
    )
    assert jnp.allclose(
        bound_metrics["predicted_total_energies"],
        default_metrics["predicted_total_energies"],
        atol=1e-6,
    )


def test_fractional_linearity_penalty_is_reported_and_nonnegative():
    molecule = _make_toy_molecule()
    functional = _make_trainable_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(3), molecule)
    datum = GroundStateDatum(molecule=molecule, target_total_energy=jnp.array(2.125))

    plain_loss, _ = ground_state_mse_loss(params, functional, datum)
    constrained_loss, metrics = ground_state_mse_loss(
        params,
        functional,
        datum,
        training_config=GroundStateTrainingConfig(
            fractional_linearity_weight=0.5,
            fractional_linearity_delta=0.1,
        ),
    )

    assert metrics["fractional_penalty"].shape == (1,)
    assert metrics["fractional_penalty"][0] >= 0.0
    assert constrained_loss >= plain_loss


def test_fractional_branch_quality_weight_softly_downweights_bad_scf():
    info = type(
        "_Info",
        (),
        {
            "mode": "self_consistent",
            "selected_rms_density": jnp.asarray(25.0),
        },
    )()
    cfg = GroundStateTrainingConfig(fractional_branch_rms_soft_threshold=1.0)

    weight = training_targets._fractional_branch_quality_weight(
        info,
        cfg,
        dtype=jnp.float32,
    )

    assert jnp.allclose(weight, 0.0016, atol=1e-6)


def test_fractional_branch_training_config_uses_stabilized_overrides():
    cfg = GroundStateTrainingConfig(
        mode="self_consistent",
        scf_max_cycle=4,
        scf_damping=0.1,
        scf_level_shift=0.0,
        scf_iterate_selection="final",
    )

    branch_cfg = training_targets._fractional_branch_training_config(cfg)

    assert branch_cfg.mode == "self_consistent"
    assert branch_cfg.scf_max_cycle == 8
    assert jnp.allclose(branch_cfg.scf_damping, 0.35, atol=1e-6)
    assert jnp.allclose(branch_cfg.scf_level_shift, 0.5, atol=1e-6)
    assert branch_cfg.scf_iterate_selection == "best_rms"


def test_replace_molecule_copy_supports_frozen_dataclass_with_extra_fields():
    @dataclass(frozen=True)
    class _FrozenMol:
        mo_coeff: jnp.ndarray

    molecule = _FrozenMol(mo_coeff=jnp.eye(2))
    updated = training_targets._replace_molecule_copy(
        molecule,
        mo_coeff=2.0 * jnp.eye(2),
        nocc_alpha=1,
        nocc_beta=1,
    )

    assert jnp.allclose(updated.mo_coeff, 2.0 * jnp.eye(2))
    assert getattr(updated, "nocc_alpha") == 1
    assert getattr(updated, "nocc_beta") == 1


def test_self_consistent_energy_penalty_is_reported_and_nonnegative():
    molecule = _make_hybrid_toy_molecule()
    functional = _make_hybrid_only_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(11), molecule)
    datum = GroundStateDatum(molecule=molecule, target_total_energy=jnp.array(-1.0))

    plain_loss, _ = ground_state_mse_loss(params, functional, datum)
    constrained_loss, metrics = ground_state_mse_loss(
        params,
        functional,
        datum,
        training_config=GroundStateTrainingConfig(
            self_consistent_energy_weight=1.0,
            scf_max_cycle=32,
            scf_damping=0.5,
        ),
    )

    assert metrics["self_consistent_energy_penalty"].shape == (1,)
    assert metrics["self_consistent_energy_mse"].shape == (1,)
    assert metrics["self_consistent_energy_mae"].shape == (1,)
    assert metrics["self_consistent_energy_penalty"][0] >= 0.0
    assert constrained_loss >= plain_loss


def test_train_step_accepts_custom_loss_function():
    molecule = _make_toy_molecule()
    functional = _make_trainable_functional()
    datum = GroundStateDatum(molecule=molecule, target_total_energy=jnp.array(2.125))

    tx = optax.adam(0.1)
    state = create_train_state_from_molecule(functional, jax.random.PRNGKey(8), molecule, tx)

    def custom_loss(params, bound_functional, data, *, training_config=None):
        return ground_state_mse_loss(
            params,
            bound_functional,
            data,
            training_config=training_config,
        )

    train_step = make_ground_state_train_step(functional, loss_fn=custom_loss)
    initial_loss, _ = ground_state_mse_loss(state.params, functional, datum)
    for _ in range(50):
        state, _ = train_step(state, datum)
    final_loss, _ = ground_state_mse_loss(state.params, functional, datum)

    assert final_loss < initial_loss


def test_train_step_accepts_explicit_predictor():
    molecule = _make_toy_molecule()
    functional = _make_trainable_functional()
    datum = GroundStateDatum(molecule=molecule, target_total_energy=jnp.array(2.125))

    tx = optax.adam(0.1)
    state = create_train_state_from_molecule(functional, jax.random.PRNGKey(10), molecule, tx)
    predictor = make_fixed_density_predictor(functional)
    train_step = make_ground_state_train_step(functional, predictor=predictor)

    initial_loss, _ = ground_state_mse_loss(state.params, functional, datum, predictor=predictor)
    for _ in range(50):
        state, _ = train_step(state, datum)
    final_loss, _ = ground_state_mse_loss(state.params, functional, datum, predictor=predictor)

    assert final_loss < initial_loss


def test_energy_normalization_per_electron_reduces_scaled_loss():
    molecule = _make_toy_molecule()
    functional = _make_trainable_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(4), molecule)
    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.array(2.125 + 0.2),
    )

    plain_loss, plain_metrics = ground_state_mse_loss(
        params,
        functional,
        datum,
        training_config=GroundStateTrainingConfig(energy_normalization="none"),
    )
    scaled_loss, scaled_metrics = ground_state_mse_loss(
        params,
        functional,
        datum,
        training_config=GroundStateTrainingConfig(energy_normalization="per_electron"),
    )

    assert plain_metrics["energy_mae"].shape == (1,)
    assert scaled_metrics["normalized_energy_mae"].shape == (1,)
    assert scaled_loss <= plain_loss


def test_electron_count_uses_overlap_weighting_when_available():
    molecule = _make_overlap_toy_molecule()

    assert jnp.allclose(_electron_count(molecule), 3.0, atol=1e-6)
