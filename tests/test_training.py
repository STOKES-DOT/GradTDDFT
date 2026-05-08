from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import optax

import td_graddft.training.targets as training_targets
from td_graddft import HARTREE_TO_EV, lorentzian_spectrum
from td_graddft.neural_xc import DensityNeuralXCFunctional, PointwiseMLP
from td_graddft.neural_xc import make_neural_xc_functional
from td_graddft.nn_rsh.functional import BoundTrainableRSHFunctional
from td_graddft.nn_rsh.schema import RSHFunctionalTemplate, ResolvedRSHParameters
from td_graddft.scf import UKSConfig, run_uks_from_integrals
from td_graddft.training import (
    GroundStateDatum,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    dm21_scf_regularization_delta_energy,
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
from td_graddft.training.targets import janak_frontier_finite_difference_penalty
from td_graddft.workflows.core import run_reference_from_spec
from td_graddft.workflows.types import ReferenceSpecConfig, SimulationConfig


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


def _make_trainable_functional():
    return DensityNeuralXCFunctional(
        model=PointwiseMLP(hidden_dims=(), output_dim=1, activation=lambda x: x),
        coefficient_input_fn=lambda density, density_floor=1e-12: jnp.ones(density.shape + (1,)),
        energy_density_basis_fn=lambda density, density_floor=1e-12: density[..., None],
        name="toy_ground_state_xc",
        hybrid_fraction_init=0.25,
    )


def _make_hybrid_only_functional():
    return DensityNeuralXCFunctional(
        model=PointwiseMLP(hidden_dims=(), output_dim=1, activation=lambda x: x),
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
    return run_reference_from_spec(
        ReferenceSpecConfig(
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


def _toy_uks_grid():
    ao = jnp.asarray(
        [
            [1.0, 0.2],
            [0.8, -0.1],
            [0.4, 0.9],
        ],
        dtype=jnp.float32,
    )
    ao_deriv1 = jnp.asarray(
        [
            ao,
            [
                [0.10, 0.00],
                [0.00, 0.20],
                [-0.10, 0.05],
            ],
            [
                [0.00, 0.10],
                [0.20, 0.00],
                [0.05, -0.10],
            ],
            [
                [-0.05, 0.00],
                [0.00, -0.05],
                [0.10, 0.10],
            ],
        ],
        dtype=jnp.float32,
    )
    weights = jnp.asarray([0.5, 0.7, 0.6], dtype=jnp.float32)
    return ao, ao_deriv1, weights


def _toy_bound_rsh(*, local_xc_spec: str, sr: float, lr: float, omega: float = 0.3):
    template = RSHFunctionalTemplate(
        name="toy_rsh",
        local_backend="jax_libxc",
        exchange_backend_id="toy",
        correlation_backend_id="toy",
        default_sr_hf_fraction=sr,
        default_lr_hf_fraction=lr,
        default_omega=omega,
    )
    return BoundTrainableRSHFunctional(
        template=template,
        local_xc_spec=local_xc_spec,
        resolved_params=ResolvedRSHParameters(
            sr_hf_fraction=sr,
            lr_hf_fraction=lr,
            omega=omega,
        ),
        fallback_omega_values=(0.0,),
    )


def _make_toy_uks_neutral_reference():
    ao, ao_deriv1, weights = _toy_uks_grid()
    eri = jnp.zeros((2, 2, 2, 2), dtype=jnp.float32)
    hcore = jnp.diag(jnp.asarray([-1.0, 0.5], dtype=jnp.float32))
    template = SimpleNamespace(
        hfx_omega_values=(0.0,),
        hfx_nu=jnp.zeros((1, weights.shape[0], 2, 2), dtype=jnp.float32),
    )
    bound = _toy_bound_rsh(local_xc_spec="hf", sr=1.0, lr=1.0)
    result = run_uks_from_integrals(
        overlap=jnp.eye(2, dtype=jnp.float32),
        hcore=hcore,
        eri=eri,
        nalpha=1,
        nbeta=1,
        nuclear_repulsion=0.0,
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid_weights=weights,
        config=UKSConfig(
            xc_spec="hf",
            max_cycle=8,
            conv_tol=1e-12,
            conv_tol_density=1e-12,
        ),
        bound_xc=bound,
        molecule_template=template,
    )
    molecule = SimpleNamespace(
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid=_Grid(weights=weights),
        rep_tensor=eri,
        mo_coeff=jnp.stack([result.mo_coeff_alpha, result.mo_coeff_beta], axis=0),
        mo_occ=jnp.stack([result.mo_occ_alpha, result.mo_occ_beta], axis=0),
        mo_energy=jnp.stack([result.mo_energy_alpha, result.mo_energy_beta], axis=0),
        rdm1=jnp.stack([result.density_matrix_alpha, result.density_matrix_beta], axis=0),
        h1e=hcore,
        nuclear_repulsion=0.0,
        overlap_matrix=jnp.eye(2, dtype=jnp.float32),
        hfx_omega_values=(0.0,),
        hfx_nu=jnp.zeros((1, weights.shape[0], 2, 2), dtype=jnp.float32),
    )
    return molecule, bound


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
        scf_gradient_mode="implicit_commutator",
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


def test_koopmans_ip_ea_diagnostic_matches_noninteracting_hf_limit():
    molecule, bound = _make_toy_uks_neutral_reference()

    diagnostic = training_targets.koopmans_ip_ea_diagnostic(
        molecule,
        bound,
    )

    assert diagnostic.cation_converged
    assert diagnostic.anion_converged
    assert jnp.allclose(diagnostic.neutral_energy, -2.0, atol=1e-6)
    assert jnp.allclose(diagnostic.cation_energy, -1.0, atol=1e-6)
    assert jnp.allclose(diagnostic.anion_energy, -1.5, atol=1e-6)
    assert jnp.allclose(diagnostic.ip_delta_scf, 1.0, atol=1e-6)
    assert jnp.allclose(diagnostic.ea_delta_scf, -0.5, atol=1e-6)
    assert jnp.allclose(diagnostic.ip_residual, 0.0, atol=1e-6)
    assert jnp.allclose(diagnostic.ea_residual, 0.0, atol=1e-6)


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
    assert jnp.allclose(metrics["s1_penalty"][0], 0.5 * metrics["s1_mae"][0], atol=1e-8)
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


def test_janak_frontier_penalty_is_reported_and_nonnegative():
    molecule = _make_hybrid_toy_molecule()
    functional = _make_hybrid_only_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(17), molecule)

    janak_mse, janak_mae, residual, finite_difference = (
        janak_frontier_finite_difference_penalty(
            params,
            functional,
            molecule,
            delta=0.1,
        )
    )
    assert residual.shape == (2,)
    assert finite_difference.shape == (2,)
    assert janak_mse >= 0.0
    assert janak_mae >= 0.0

    plain_datum = GroundStateDatum(molecule=molecule, target_total_energy=jnp.array(-1.0))
    constrained_datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.array(-1.0),
        janak_frontier_constraint_weight=0.5,
    )
    training_cfg = GroundStateTrainingConfig(
        mode="self_consistent",
        janak_frontier_delta=0.1,
        scf_max_cycle=32,
        scf_damping=0.5,
    )

    plain_loss, _ = ground_state_mse_loss(
        params,
        functional,
        plain_datum,
        training_config=training_cfg,
    )
    constrained_loss, metrics = ground_state_mse_loss(
        params,
        functional,
        constrained_datum,
        training_config=training_cfg,
    )

    assert metrics["janak_frontier_penalty"].shape == (1,)
    assert metrics["janak_frontier_penalty"][0] >= 0.0
    assert metrics["janak_frontier_mse"].shape == (1,)
    assert metrics["janak_frontier_mse"][0] >= 0.0
    assert metrics["janak_frontier_mae"].shape == (1,)
    assert metrics["janak_frontier_mae"][0] >= 0.0
    assert constrained_loss >= plain_loss


def test_janak_frontier_penalty_reoptimizes_fractional_frontier_states(monkeypatch):
    molecule = _make_hybrid_toy_molecule()
    functional = _make_hybrid_only_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(23), molecule)

    resolve_count = {"value": 0}
    info = type(
        "_Info",
        (),
        {
            "mode": "self_consistent",
            "selected_rms_density": jnp.asarray(0.0),
        },
    )()

    def _fake_resolve(params_in, functional_in, molecule_in, training_config_in):
        del params_in, functional_in, training_config_in
        resolve_count["value"] += 1
        return molecule_in, info

    monkeypatch.setattr(
        training_targets,
        "_resolve_training_molecule_and_info_with_mode",
        _fake_resolve,
    )

    janak_mse, janak_mae, residual, finite_difference = (
        training_targets.janak_frontier_finite_difference_penalty(
            params,
            functional,
            molecule,
            delta=0.1,
            training_config=GroundStateTrainingConfig(mode="self_consistent"),
            assume_self_consistent_input=True,
        )
    )

    assert resolve_count["value"] >= 2
    assert residual.shape == (2,)
    assert finite_difference.shape == (2,)
    assert janak_mse >= 0.0
    assert janak_mae >= 0.0


def test_janak_frontier_penalty_freezes_functional_binding_on_base_state(monkeypatch):
    molecule = _make_hybrid_toy_molecule()
    params = {}
    bound_occupations = []
    info = type(
        "_Info",
        (),
        {
            "mode": "self_consistent",
            "selected_rms_density": jnp.asarray(0.0),
        },
    )()

    class _Bound:
        def __init__(self, scale):
            self.scale = scale

        def energy_from_molecule(self, molecule_in):
            return self.scale * jnp.sum(jnp.asarray(molecule_in.mo_occ))

    class _Functional:
        def bind_to_molecule(self, _params, molecule_in):
            occ = jnp.asarray(molecule_in.mo_occ)
            bound_occupations.append(occ)
            return _Bound(jnp.sum(occ))

    def _fake_resolve(params_in, functional_in, molecule_in, training_config_in):
        del params_in, functional_in, training_config_in
        return molecule_in, info

    monkeypatch.setattr(
        training_targets,
        "_resolve_training_molecule_and_info_with_mode",
        _fake_resolve,
    )

    janak_mse, janak_mae, residual, derivative = (
        training_targets.janak_frontier_finite_difference_penalty(
            params,
            _Functional(),
            molecule,
            delta=0.1,
            training_config=GroundStateTrainingConfig(mode="self_consistent"),
            assume_self_consistent_input=True,
        )
    )

    assert len(bound_occupations) == 1
    assert jnp.allclose(bound_occupations[0], jnp.asarray(molecule.mo_occ))
    assert residual.shape == (2,)
    assert derivative.shape == (2,)
    assert janak_mse >= 0.0
    assert janak_mae >= 0.0


def test_janak_frontier_penalty_tracks_base_orbitals_by_overlap(monkeypatch):
    lumo = jnp.array([0.4, jnp.sqrt(1.0 - 0.4**2)])
    homo = jnp.array([-lumo[1], lumo[0]])
    base_coeff = jnp.stack([homo, lumo], axis=1)
    base_molecule = _ToyMolecule(
        ao=jnp.eye(2),
        grid=_Grid(weights=jnp.array([1.0, 1.0])),
        rep_tensor=jnp.zeros((2, 2, 2, 2)),
        mo_coeff=jnp.stack([base_coeff, base_coeff], axis=0),
        mo_occ=jnp.array([[1.0, 0.0], [1.0, 0.0]]),
        mo_energy=jnp.array([[0.5, 1.5], [0.5, 1.5]]),
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
    target_coeff = jnp.eye(2)
    homo_minus = training_targets._replace_molecule_copy(
        base_molecule,
        mo_coeff=jnp.stack([target_coeff, target_coeff], axis=0),
        mo_energy=jnp.array([[2.0, 6.0], [2.0, 6.0]]),
    )
    lumo_plus = training_targets._replace_molecule_copy(
        base_molecule,
        mo_coeff=jnp.stack([target_coeff, target_coeff], axis=0),
        mo_energy=jnp.array([[3.0, 7.0], [3.0, 7.0]]),
    )
    info = type(
        "_Info",
        (),
        {
            "mode": "self_consistent",
            "selected_rms_density": jnp.asarray(0.0),
        },
    )()

    def _fake_fractional_state(_params, _functional, _molecule, *, homo_delta=0.0, lumo_delta=0.0, **_kwargs):
        if float(jnp.asarray(homo_delta)) < 0.0:
            return homo_minus, info
        if float(jnp.asarray(lumo_delta)) > 0.0:
            return lumo_plus, info
        raise AssertionError("Unexpected fractional branch request.")

    monkeypatch.setattr(
        training_targets,
        "_resolve_variational_frontier_state_and_info",
        _fake_fractional_state,
    )
    monkeypatch.setattr(
        training_targets,
        "_predict_ground_state_total_energy_from_molecule",
        lambda *_args, **_kwargs: jnp.asarray(0.0),
    )

    _, _, residual, derivative = training_targets.janak_frontier_finite_difference_penalty(
        {},
        object(),
        base_molecule,
        delta=0.1,
        training_config=GroundStateTrainingConfig(mode="self_consistent"),
        assume_self_consistent_input=True,
    )

    assert derivative.shape == (2,)
    assert jnp.allclose(residual[1], -3.0, atol=1e-6)


def test_strict_janak_frontier_autodiff_penalty_matches_unrestricted_noninteracting_limit():
    molecule = _ToyMolecule(
        ao=jnp.eye(2),
        grid=_Grid(weights=jnp.array([1.0, 1.0])),
        rep_tensor=jnp.zeros((2, 2, 2, 2)),
        mo_coeff=jnp.stack([jnp.eye(2), jnp.eye(2)], axis=0),
        mo_occ=jnp.array([[1.0, 0.0], [0.0, 0.0]]),
        mo_energy=jnp.array([[-0.8, 0.2], [-0.8, 0.2]]),
        rdm1=jnp.array(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ),
        h1e=jnp.diag(jnp.array([-0.8, 0.2])),
        nuclear_repulsion=0.0,
        overlap_matrix=jnp.eye(2),
    )

    class _Functional:
        def bind_to_molecule_for_scf(self, _params, _molecule):
            class _Bound:
                exact_exchange_fraction = jnp.asarray(0.0)

                def unrestricted_scf_components(self, molecule_in):
                    ngrids = int(molecule_in.ao.shape[0])
                    nao = int(molecule_in.ao.shape[1])
                    zeros_v = jnp.zeros((ngrids,), dtype=jnp.float32)
                    zeros_grad = jnp.zeros((ngrids, 3), dtype=jnp.float32)
                    zeros_mat = jnp.zeros((nao, nao), dtype=jnp.float32)
                    return (
                        zeros_v,
                        zeros_v,
                        zeros_grad,
                        zeros_grad,
                        "LDA",
                        jnp.asarray(0.0, dtype=jnp.float32),
                        zeros_mat,
                        zeros_mat,
                    )

                def energy_from_molecule(self, _molecule_in):
                    return jnp.asarray(0.0, dtype=jnp.float32)

            return _Bound()

        bind_to_molecule = bind_to_molecule_for_scf

    cfg = GroundStateTrainingConfig(
        mode="self_consistent",
        scf_gradient_mode="unrolled",
        scf_max_cycle=4,
        scf_damping=0.0,
        scf_level_shift=0.0,
        scf_iterate_selection="final",
        fractional_branch_scf_max_cycle=4,
        fractional_branch_scf_damping=0.0,
        fractional_branch_scf_level_shift=0.0,
        fractional_branch_scf_iterate_selection="final",
    )

    mse, mae, residual, derivative = training_targets.strict_janak_frontier_autodiff_penalty(
        {},
        _Functional(),
        molecule,
        eta=0.1,
        training_config=cfg,
        assume_self_consistent_input=True,
    )

    assert residual.shape == (2,)
    assert derivative.shape == (2,)
    assert mse >= 0.0
    assert mae >= 0.0
    assert jnp.allclose(derivative, jnp.array([-0.8, -0.8]), atol=1e-5)
    assert jnp.allclose(residual, jnp.zeros((2,)), atol=1e-5)


def test_fixed_orbital_janak_autodiff_penalty_matches_noninteracting_limit():
    molecule = _ToyMolecule(
        ao=jnp.eye(2),
        grid=_Grid(weights=jnp.array([1.0, 1.0])),
        rep_tensor=jnp.zeros((2, 2, 2, 2)),
        mo_coeff=jnp.stack([jnp.eye(2), jnp.eye(2)], axis=0),
        mo_occ=jnp.array([[1.0, 0.0], [1.0, 0.0]]),
        mo_energy=jnp.array([[-0.8, 0.2], [-0.8, 0.2]]),
        rdm1=jnp.array(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 0.0]],
            ]
        ),
        h1e=jnp.diag(jnp.array([-0.8, 0.2])),
        nuclear_repulsion=0.0,
        overlap_matrix=jnp.eye(2),
    )

    class _Functional:
        def bind_to_molecule(self, _params, _molecule):
            class _Bound:
                def energy_from_molecule(self, _molecule_in):
                    return jnp.asarray(0.0, dtype=jnp.float32)

            return _Bound()

    mse, mae, residual, derivative = training_targets.fixed_orbital_janak_autodiff_penalty(
        {},
        _Functional(),
        molecule,
        assume_self_consistent_input=True,
    )

    assert residual.shape == (2,)
    assert derivative.shape == (2,)
    assert jnp.allclose(derivative, jnp.array([-0.8, 0.2]), atol=1e-6)
    assert jnp.allclose(residual, jnp.zeros((2,)), atol=1e-6)
    assert jnp.allclose(mse, 0.0, atol=1e-6)
    assert jnp.allclose(mae, 0.0, atol=1e-6)


def test_half_charge_janak_autodiff_penalty_matches_endpoint_slopes(monkeypatch):
    molecule = _ToyMolecule(
        ao=jnp.eye(2),
        grid=_Grid(weights=jnp.array([1.0, 1.0])),
        rep_tensor=jnp.zeros((2, 2, 2, 2)),
        mo_coeff=jnp.stack([jnp.eye(2), jnp.eye(2)], axis=0),
        mo_occ=jnp.array([[1.0, 0.0], [1.0, 0.0]]),
        mo_energy=jnp.array([[-0.7, 0.3], [-0.7, 0.3]]),
        rdm1=jnp.array(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 0.0]],
            ]
        ),
        h1e=jnp.zeros((2, 2)),
        nuclear_repulsion=0.0,
        overlap_matrix=jnp.eye(2),
    )

    def _state(spin_index, orbital_index, delta, energy):
        state = training_targets._perturb_spin_orbital_occupation(
            molecule,
            spin_index=spin_index,
            orbital_index=orbital_index,
            delta=delta,
        )
        state = training_targets._replace_molecule_copy(
            state,
            total_energy=jnp.asarray(energy),
        )
        return state

    state_energies = {
        -1.0: -9.0,
        -0.5: -9.5,
        0.5: -9.8,
        1.0: -9.4,
    }

    def _fake_branch(
        _params,
        _functional,
        _molecule,
        *,
        spin_index,
        orbital_index,
        delta,
        **_kwargs,
    ):
        return _state(
            spin_index,
            orbital_index,
            delta,
            state_energies[float(delta)],
        ), SimpleNamespace(mode="self_consistent", selected_rms_density=0.0)

    class _Functional:
        def bind_to_molecule(self, _params, _molecule):
            class _Bound:
                def energy_from_molecule(self, molecule_in):
                    occ = jnp.asarray(molecule_in.mo_occ)
                    if hasattr(molecule_in, "total_energy"):
                        return jnp.asarray(molecule_in.total_energy)
                    return (
                        -10.0
                        - 1.0 * (occ[0, 0] - 1.0)
                        + 0.6 * jnp.sum(occ[:, 1])
                    )

            return _Bound()

    monkeypatch.setattr(
        training_targets,
        "_resolve_variational_spin_orbital_state_and_info",
        _fake_branch,
    )

    mse, mae, residual, derivative = training_targets.half_charge_janak_autodiff_penalty(
        {},
        _Functional(),
        molecule,
        training_config=GroundStateTrainingConfig(mode="self_consistent"),
        assume_self_consistent_input=True,
    )

    assert derivative.shape == (2,)
    assert residual.shape == (2,)
    assert jnp.allclose(derivative, jnp.array([-1.0, 0.6]), atol=1e-6)
    assert jnp.allclose(residual, jnp.zeros((2,)), atol=1e-6)
    assert jnp.allclose(mse, 0.0, atol=1e-6)
    assert jnp.allclose(mae, 0.0, atol=1e-6)


def test_strict_janak_frontier_autodiff_penalty_falls_back_when_autodiff_is_nonfinite(monkeypatch):
    molecule = _ToyMolecule(
        ao=jnp.eye(2),
        grid=_Grid(weights=jnp.array([1.0, 1.0])),
        rep_tensor=jnp.zeros((2, 2, 2, 2)),
        mo_coeff=jnp.stack([jnp.eye(2), jnp.eye(2)], axis=0),
        mo_occ=jnp.array([[1.0, 0.0], [0.0, 0.0]]),
        mo_energy=jnp.array([[-0.8, 0.2], [-0.8, 0.2]]),
        rdm1=jnp.array(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ),
        h1e=jnp.diag(jnp.array([-0.8, 0.2])),
        nuclear_repulsion=0.0,
        overlap_matrix=jnp.eye(2),
    )

    class _Functional:
        def bind_to_molecule_for_scf(self, _params, _molecule):
            class _Bound:
                exact_exchange_fraction = jnp.asarray(0.0)

                def unrestricted_scf_components(self, molecule_in):
                    ngrids = int(molecule_in.ao.shape[0])
                    nao = int(molecule_in.ao.shape[1])
                    zeros_v = jnp.zeros((ngrids,), dtype=jnp.float32)
                    zeros_grad = jnp.zeros((ngrids, 3), dtype=jnp.float32)
                    zeros_mat = jnp.zeros((nao, nao), dtype=jnp.float32)
                    return (
                        zeros_v,
                        zeros_v,
                        zeros_grad,
                        zeros_grad,
                        "LDA",
                        jnp.asarray(0.0, dtype=jnp.float32),
                        zeros_mat,
                        zeros_mat,
                    )

                def energy_from_molecule(self, _molecule_in):
                    return jnp.asarray(0.0, dtype=jnp.float32)

            return _Bound()

        bind_to_molecule = bind_to_molecule_for_scf

    real_grad = training_targets.jax.grad

    def _nan_grad(_fn):
        def _wrapped(_eta):
            return jnp.asarray(jnp.nan, dtype=jnp.float32)

        return _wrapped

    monkeypatch.setattr(training_targets.jax, "grad", _nan_grad)
    try:
        mse, mae, residual, derivative = training_targets.strict_janak_frontier_autodiff_penalty(
            {},
            _Functional(),
            molecule,
            eta=0.1,
            training_config=GroundStateTrainingConfig(
                mode="self_consistent",
                janak_frontier_mode="autodiff",
                scf_gradient_mode="unrolled",
                scf_max_cycle=4,
                scf_damping=0.0,
                scf_level_shift=0.0,
                scf_iterate_selection="final",
                fractional_branch_scf_max_cycle=4,
                fractional_branch_scf_damping=0.0,
                fractional_branch_scf_level_shift=0.0,
                fractional_branch_scf_iterate_selection="final",
            ),
            assume_self_consistent_input=True,
        )
    finally:
        monkeypatch.setattr(training_targets.jax, "grad", real_grad)

    assert jnp.isfinite(mse)
    assert jnp.isfinite(mae)
    assert jnp.all(jnp.isfinite(residual))
    assert jnp.all(jnp.isfinite(derivative))
    assert jnp.allclose(derivative, jnp.array([-0.8, -0.8]), atol=1e-4)


def test_strict_janak_frontier_autodiff_penalty_training_path_skips_eta_autodiff(monkeypatch):
    molecule = _ToyMolecule(
        ao=jnp.eye(2),
        grid=_Grid(weights=jnp.array([1.0, 1.0])),
        rep_tensor=jnp.zeros((2, 2, 2, 2)),
        mo_coeff=jnp.stack([jnp.eye(2), jnp.eye(2)], axis=0),
        mo_occ=jnp.array([[1.0, 0.0], [0.0, 0.0]]),
        mo_energy=jnp.array([[-0.8, 0.2], [-0.8, 0.2]]),
        rdm1=jnp.array(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ),
        h1e=jnp.diag(jnp.array([-0.8, 0.2])),
        nuclear_repulsion=0.0,
        overlap_matrix=jnp.eye(2),
    )

    class _ScalarFunctional:
        def bind_to_molecule_for_scf(self, params, _molecule):
            strength = jnp.asarray(params["strength"], dtype=jnp.float32)

            class _Bound:
                exact_exchange_fraction = jnp.asarray(0.0)

                def unrestricted_scf_components(self, molecule_in):
                    ngrids = int(molecule_in.ao.shape[0])
                    nao = int(molecule_in.ao.shape[1])
                    zeros_grad = jnp.zeros((ngrids, 3), dtype=jnp.float32)
                    zeros_mat = jnp.zeros((nao, nao), dtype=jnp.float32)
                    v_alpha = strength * jnp.ones((ngrids,), dtype=jnp.float32)
                    v_beta = -strength * jnp.ones((ngrids,), dtype=jnp.float32)
                    return (
                        v_alpha,
                        v_beta,
                        zeros_grad,
                        zeros_grad,
                        "LDA",
                        jnp.asarray(0.0, dtype=jnp.float32),
                        zeros_mat,
                        zeros_mat,
                    )

                def energy_from_molecule(self, _molecule_in):
                    return jnp.asarray(0.0, dtype=jnp.float32)

            return _Bound()

        bind_to_molecule = bind_to_molecule_for_scf

    monkeypatch.setattr(training_targets, "_tree_contains_jax_tracer", lambda _tree: True)

    def _should_not_be_called(_fn):
        raise AssertionError("strict training path should not call jax.grad on eta branches")

    monkeypatch.setattr(training_targets.jax, "grad", _should_not_be_called)

    cfg = GroundStateTrainingConfig(
        mode="self_consistent",
        janak_frontier_mode="autodiff",
        scf_gradient_mode="unrolled",
        scf_max_cycle=4,
        scf_damping=0.0,
        scf_level_shift=0.0,
        scf_iterate_selection="final",
        fractional_branch_scf_max_cycle=4,
        fractional_branch_scf_damping=0.0,
        fractional_branch_scf_level_shift=0.0,
        fractional_branch_scf_iterate_selection="final",
    )

    def _loss_fn(strength):
        _, mae, _, _ = training_targets.strict_janak_frontier_autodiff_penalty(
            {"strength": strength},
            _ScalarFunctional(),
            molecule,
            eta=0.1,
            training_config=cfg,
            assume_self_consistent_input=True,
        )
        return mae

    value, grad = jax.value_and_grad(_loss_fn)(jnp.asarray(0.05, dtype=jnp.float32))

    assert jnp.isfinite(value)
    assert jnp.isfinite(grad)


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


def test_strict_janak_branch_training_config_forces_unrolled():
    cfg = GroundStateTrainingConfig(
        mode="self_consistent",
        janak_frontier_mode="autodiff",
        scf_gradient_mode="implicit_commutator",
        fractional_branch_scf_max_cycle=6,
        fractional_branch_scf_damping=0.4,
        fractional_branch_scf_level_shift=0.6,
        fractional_branch_scf_iterate_selection="best_rms",
    )

    branch_cfg = training_targets._strict_janak_branch_training_config(cfg)

    assert branch_cfg.scf_gradient_mode == "unrolled"
    assert branch_cfg.scf_max_cycle == 6
    assert jnp.allclose(branch_cfg.scf_damping, 0.4, atol=1e-6)
    assert jnp.allclose(branch_cfg.scf_level_shift, 0.6, atol=1e-6)


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


def test_ground_state_loss_uses_strict_janak_dispatch_when_requested(monkeypatch):
    molecule = _make_hybrid_toy_molecule()
    functional = _make_hybrid_only_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(31), molecule)
    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.array(-1.0),
        janak_frontier_constraint_weight=0.5,
    )

    called = {"strict": 0}

    def _fake_strict(*_args, **_kwargs):
        called["strict"] += 1
        return (
            jnp.asarray(4.0),
            jnp.asarray(2.0),
            jnp.asarray([1.0, -1.0]),
            jnp.asarray([0.5, 0.6]),
        )

    monkeypatch.setattr(
        training_targets,
        "strict_janak_frontier_autodiff_penalty",
        _fake_strict,
    )

    loss, metrics = ground_state_mse_loss(
        params,
        functional,
        datum,
        training_config=GroundStateTrainingConfig(
            mode="self_consistent",
            janak_frontier_mode="autodiff",
            janak_frontier_delta=0.1,
            scf_gradient_mode="unrolled",
            scf_max_cycle=6,
        ),
    )

    assert called["strict"] == 1
    assert jnp.isfinite(loss)
    assert jnp.allclose(metrics["janak_frontier_mse"][0], 4.0, atol=1e-6)
    assert jnp.allclose(metrics["janak_frontier_mae"][0], 2.0, atol=1e-6)


def test_janak_full_scf_ad_dispatch_enables_training_eta_autodiff(monkeypatch):
    molecule = _make_hybrid_toy_molecule()
    functional = _make_hybrid_only_functional()
    params = functional.init_from_molecule(jax.random.PRNGKey(37), molecule)
    called = {"strict": 0}

    def _fake_strict(*_args, **kwargs):
        called["strict"] += 1
        assert kwargs["force_eta_autodiff"] is True
        return (
            jnp.asarray(4.0),
            jnp.asarray(2.0),
            jnp.asarray([1.0, -1.0]),
            jnp.asarray([0.5, 0.6]),
        )

    monkeypatch.setattr(
        training_targets,
        "strict_janak_frontier_autodiff_penalty",
        _fake_strict,
    )

    mse, mae, residual, derivative = training_targets._janak_frontier_penalty_by_mode(
        params,
        functional,
        molecule,
        training_config=GroundStateTrainingConfig(
            mode="self_consistent",
            janak_frontier_mode="full_scf_ad",
            scf_gradient_mode="unrolled",
            scf_max_cycle=3,
        ),
        assume_self_consistent_input=True,
    )

    assert called["strict"] == 1
    assert jnp.allclose(mse, 4.0, atol=1e-6)
    assert jnp.allclose(mae, 2.0, atol=1e-6)
    assert residual.shape == (2,)
    assert derivative.shape == (2,)


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
