import pytest

import jax.numpy as jnp

from td_graddft import neural_xc, nn_rsh, training
from td_graddft.scf.packed_eri import build_j_from_eri_pair_matrix
import td_graddft.training.neural_xc_trainer as neural_xc_trainer_module
import td_graddft.training.rsh_optimizer as rsh_optimizer_module


def test_neural_xc_functional_public_constructor_uses_unified_name():
    functional = neural_xc.Functional(
        hidden_dims=(8,),
        input_feature_mode="dm21_original",
        architecture="residual",
    )
    via_factory = neural_xc.make_functional(
        hidden_dims=(8,),
        input_feature_mode="dm21_original",
        architecture="residual",
    )

    assert isinstance(functional, neural_xc.NeuralXCHybridFunctional)
    assert isinstance(via_factory, neural_xc.NeuralXCHybridFunctional)
    assert functional.name == "neural_xc"
    assert functional.input_feature_mode == "dm21_original"


def test_legacy_neural_xc_public_constructors_are_removed():
    removed = (
        "Density" "NeuralXCFunctional",
        "Neural" "XCFunctional",
        "Pointwise" "MLP",
        "make_neural" "_lda_functional",
        "make_dm21" "_like_functional",
    )

    for name in removed:
        assert not hasattr(neural_xc, name), f"{name} should not be a public Neural XC API"


def test_legacy_neural_xc_subpackage_exports_are_removed():
    import importlib

    base_module = importlib.import_module("td_graddft.neural_xc.base")
    dm21_module = importlib.import_module("td_graddft.neural_xc.dm21")
    base_removed = (
        "Density" "NeuralXCFunctional",
        "Neural" "XCFunctional",
        "Pointwise" "MLP",
        "make_neural" "_lda_functional",
    )

    for name in base_removed:
        assert not hasattr(base_module, name), f"{name} should not be exported from neural_xc.base"
    assert not hasattr(dm21_module, "make_dm21" "_like_functional")


def test_rsh_public_constructor_builds_trainable_functional():
    functional = nn_rsh.RSH("lc-wpbe").trainable(
        params=("omega", "alpha", "beta"),
        hidden_dims=(4,),
    )

    assert isinstance(functional, nn_rsh.TrainableRSHFunctional)
    assert functional.template.name == "lc-wpbe"


def test_rsh_public_constructor_uses_strict_lc_wpbe_local_spec_by_default():
    functional = nn_rsh.RSH("lc-wpbe").trainable()

    assert functional.local_xc_spec == "lc_wpbe_local"


def test_rsh_public_constructor_rejects_unknown_trainable_parameter():
    with pytest.raises(ValueError, match="Unsupported RSH trainable parameter"):
        nn_rsh.RSH("lc-wpbe").trainable(params=("gamma",))


def test_training_result_and_separate_trainers_expose_expected_history_keys():
    neural_functional = neural_xc.Functional(hidden_dims=(8,))
    rsh_functional = nn_rsh.RSH("lc-wpbe").trainable()

    neural_trainer = training.NeuralXCTrainer(
        functional=neural_functional,
        molecules=[],
    )
    rsh_optimizer = training.RSHOptimizer(
        functional=rsh_functional,
        molecules=[],
    )

    assert type(neural_trainer) is not type(rsh_optimizer)

    neural_result = neural_trainer.kernel(steps=0)
    rsh_result = rsh_optimizer.kernel(steps=0)

    assert isinstance(neural_result, training.TrainingResult)
    assert isinstance(rsh_result, training.TrainingResult)
    assert set(neural_result.history) == {
        "loss",
        "energy_mae",
        "density_mse",
        "orbital_energy_mae",
        "scf_cycles",
        "scf_converged",
    }
    assert set(rsh_result.history) == {
        "loss",
        "omega",
        "alpha",
        "beta",
        "ip_error",
        "ea_error",
    }


def test_neural_xc_trainer_positive_steps_require_ground_state_data():
    neural_trainer = training.NeuralXCTrainer(
        functional=neural_xc.Functional(hidden_dims=(8,)),
        molecules=[],
    )

    with pytest.raises(ValueError, match="requires at least one"):
        neural_trainer.kernel(steps=1)


def test_rsh_optimizer_runs_positive_steps_through_self_supervised_loss(monkeypatch):
    build_calls = []
    training_configs = []

    def _fake_self_supervised_loss(functional, **kwargs):
        build_calls.append(kwargs)

        def _loss(params, active_functional, data, *, training_config=None, predictor=None):
            training_configs.append(training_config)
            del data, predictor
            resolved = active_functional.resolve_parameters(params)
            target_omega = jnp.asarray(0.22, dtype=jnp.float32)
            value = (resolved.omega - target_omega) ** 2
            return value, {
                "loss": value,
                "omega": jnp.asarray([resolved.omega]),
                "sr_hf_fraction": jnp.asarray([resolved.sr_hf_fraction]),
                "lr_hf_fraction": jnp.asarray([resolved.lr_hf_fraction]),
                "koopmans_ip_mae": jnp.asarray([value]),
                "koopmans_lumo_ea_mae": jnp.asarray([value + 0.1]),
            }

        return _loss

    monkeypatch.setattr(
        rsh_optimizer_module,
        "make_self_supervised_rsh_loss",
        _fake_self_supervised_loss,
    )
    functional = nn_rsh.RSH("lc-wpbe").trainable()
    molecule = object()
    result = training.RSHOptimizer(
        functional=functional,
        molecules=[molecule],
    ).kernel(
        steps=8,
        learning_rate=0.5,
        loss="koopmans_ip_ea",
    )

    assert len(result.history["loss"]) == 8
    assert result.history["loss"][-1] < result.history["loss"][0]
    assert result.params is not None
    assert result.final_metrics["omega"] == result.history["omega"][-1]
    assert set(result.history) == {
        "loss",
        "omega",
        "alpha",
        "beta",
        "ip_error",
        "ea_error",
    }
    assert build_calls
    assert build_calls[0]["koopmans_ip_weight"] == 1.0
    assert build_calls[0]["koopmans_ea_weight"] == 0.0
    assert build_calls[0]["koopmans_lumo_ea_weight"] == 1.0
    assert training_configs
    assert training_configs[0].mode == "self_consistent"
    assert training_configs[0].scf_gradient_mode == "implicit_commutator"


def test_rsh_optimizer_positive_steps_require_at_least_one_molecule():
    optimizer = training.RSHOptimizer(
        functional=nn_rsh.RSH("lc-wpbe").trainable(),
        molecules=[],
    )

    with pytest.raises(ValueError, match="requires at least one"):
        optimizer.kernel(steps=1)


class _Grid:
    def __init__(self, weights):
        self.weights = weights


class _ToyMolecule:
    def __init__(self):
        self.ao = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        self.grid = _Grid(weights=jnp.array([1.0, 1.0]))
        self.rep_tensor = jnp.zeros((2, 2, 2, 2))
        self.mo_coeff = jnp.stack([jnp.eye(2), jnp.eye(2)], axis=0)
        self.mo_occ = jnp.array([[1.0, 0.0], [1.0, 0.0]])
        self.mo_energy = jnp.array([[0.0, 1.0], [0.0, 1.0]])
        self.rdm1 = jnp.array(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 0.0]],
            ]
        )
        self.h1e = jnp.zeros((2, 2))
        self.nuclear_repulsion = 0.0
        self.dipole_integrals = jnp.zeros((3, 2, 2))

    def density(self):
        return jnp.einsum("spq,rp,rq->rs", self.rdm1, self.ao, self.ao)


class _ToyTrainableModel:
    @staticmethod
    def apply(params, *args, **kwargs):
        del args, kwargs
        return params["scale"]


class _ToyTrainableFunctional:
    model = _ToyTrainableModel()
    name = "toy_public_neural_xc"

    def init_from_molecule(self, rng, molecule):
        del rng, molecule
        return {"scale": jnp.asarray(0.0)}

    def energy_from_molecule(self, params, molecule):
        density = jnp.asarray(molecule.density())
        density_weight = jnp.sum(density)
        return jnp.asarray(params["scale"]) * density_weight


def _toy_trainable_density_functional():
    return _ToyTrainableFunctional()


def test_neural_xc_trainer_runs_50_ground_state_steps_and_lowers_loss():
    molecule = _ToyMolecule()
    datum = training.GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.asarray(2.125),
    )
    trainer = training.NeuralXCTrainer(
        functional=_toy_trainable_density_functional(),
        molecules=[datum],
    )

    result = trainer.kernel(steps=50, learning_rate=0.1)

    assert len(result.history["loss"]) == 50
    assert result.history["loss"][-1] < result.history["loss"][0]
    assert result.final_metrics["loss"] == result.history["loss"][-1]
    assert result.params is not None


def test_neural_xc_trainer_accepts_explicit_training_config(monkeypatch):
    captured = {}

    def _fake_make_train_step(functional, *, training_config=None, **kwargs):
        del functional, kwargs
        captured["training_config"] = training_config

        def _step(state, data):
            del data
            return state, {
                "loss": jnp.asarray([0.0]),
                "energy_mae": jnp.asarray([0.0]),
                "density_mse": jnp.asarray([0.0]),
                "orbital_energy_mae": jnp.asarray([0.0]),
                "scf_cycles_mean": jnp.asarray([1.0]),
                "scf_converged_fraction": jnp.asarray([1.0]),
            }

        return _step

    monkeypatch.setattr(
        neural_xc_trainer_module,
        "make_ground_state_train_step",
        _fake_make_train_step,
    )
    molecule = _ToyMolecule()
    datum = training.GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.asarray(0.0),
    )
    cfg = training.GroundStateTrainingConfig(
        mode="self_consistent",
        scf_gradient_mode="implicit_commutator",
    )
    trainer = training.NeuralXCTrainer(
        functional=_toy_trainable_density_functional(),
        molecules=[datum],
    )

    result = trainer.kernel(steps=1, training_config=cfg)

    assert captured["training_config"] is cfg
    assert result.history["scf_converged"] == [1.0]


def test_ground_state_datum_from_reference_requires_hfx_fields():
    molecule = _ToyMolecule()

    with pytest.raises(ValueError, match="hfx_local"):
        training.GroundStateDatum.from_reference(
            molecule,
            target_total_energy=jnp.asarray(0.0),
            require_hfx=True,
        )

    molecule.hfx_local = jnp.zeros((2, 2, 1))
    molecule.hfx_nu = jnp.zeros((1, 2, 2, 2))
    datum = training.GroundStateDatum.from_reference(
        molecule,
        target_total_energy=jnp.asarray(0.0),
        require_hfx=True,
    )

    assert datum.molecule is molecule


def test_ground_state_datum_from_reference_requires_pt2_fields_when_pt2_enabled():
    from td_graddft.reference import GridReference, RestrictedMoleculeReference

    grid = GridReference(coords=jnp.zeros((2, 3)), weights=jnp.ones((2,)))
    molecule = RestrictedMoleculeReference(
        ao=jnp.ones((2, 2)),
        ao_deriv1=jnp.ones((4, 2, 2)),
        grid=grid,
        dipole_integrals=jnp.zeros((3, 2, 2)),
        rep_tensor=jnp.zeros((2, 2, 2, 2)),
        rdm1=jnp.stack([jnp.eye(2), jnp.eye(2)], axis=0),
        h1e=jnp.eye(2),
        nuclear_repulsion=0.0,
        mo_coeff=jnp.stack([jnp.eye(2), jnp.eye(2)], axis=0),
        mo_occ=jnp.asarray([[1.0, 0.0], [1.0, 0.0]]),
        mo_energy=jnp.asarray([[-0.5, 0.1], [-0.5, 0.1]]),
        mf_energy=-1.0,
        hfx_local=jnp.zeros((2, 2, 1)),
        hfx_nu=jnp.zeros((1, 2, 2, 2)),
    )
    functional = neural_xc.Functional(
        hidden_dims=(8,),
        include_pt2_channel=True,
        pt2_channel_mode="local_exact",
    )

    with pytest.raises(ValueError, match="compute_local_pt2_features=True"):
        training.GroundStateDatum.from_reference(
            molecule,
            target_total_energy=-1.0,
            functional=functional,
        )


def test_training_coulomb_energy_accepts_packed_eri_pair_matrix():
    from td_graddft.training.targets import _coulomb_energy

    rep_tensor = jnp.asarray(
        [
            [1.0, 0.2, 0.3],
            [0.2, 0.4, 0.5],
            [0.3, 0.5, 0.8],
        ]
    )
    density = jnp.asarray([[1.0, 0.2], [0.2, 0.7]])

    packed_energy = _coulomb_energy(density, rep_tensor)
    expected = 0.5 * jnp.einsum(
        "pq,pq->",
        density,
        build_j_from_eri_pair_matrix(rep_tensor, density),
    )

    assert jnp.allclose(packed_energy, expected)
