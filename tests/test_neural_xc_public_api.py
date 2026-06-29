import pytest

import jax.numpy as jnp

from td_graddft import neural_xc, training
from td_graddft.neural_xc.defaults import (
    DEFAULT_NEURAL_XC_HF_INPUT_MODE,
    DEFAULT_NEURAL_XC_RESPONSE_HF_MODE,
    DEFAULT_NEURAL_XC_RESPONSE_PT2_MODE,
    DEFAULT_NEURAL_XC_SEMILOCAL_XC,
)
from td_graddft.data.integrals.jax.packed_eri import build_j_from_eri_pair_matrix
import td_graddft.training.neural_xc_trainer as neural_xc_trainer_module


def test_neural_xc_config_drives_generic_functional_constructor():
    config = neural_xc.Config(
        components=neural_xc.ComponentSpec(
            backend="jax_libxc",
            semilocal=("gga_x_pbe", "gga_c_pbe"),
        ),
        channels=neural_xc.ChannelSpec(
            hf="spin_resolved",
            pt2="off",
        ),
        network=neural_xc.NetworkSpec(
            architecture="residual",
            hidden_dims=(8,),
        ),
        input_feature_mode="canonical",
        name="configured_neural_xc",
    )

    functional = neural_xc.Functional(config=config)

    assert isinstance(functional, neural_xc.NeuralXCHybridFunctional)
    assert functional.name == "configured_neural_xc"
    assert functional.input_feature_mode == "canonical"
    assert functional.hf_input_mode == "spin_resolved"
    assert functional.include_pt2_channel is False
    assert functional.resolved_non_hf_module().channel_names == ("gga_x_pbe", "gga_c_pbe")


def test_neural_xc_dm21_preset_returns_example_config():
    config = neural_xc.presets.dm21()
    functional = neural_xc.Functional(config=config)

    assert isinstance(config, neural_xc.Config)
    assert config.components.backend == "jax_libxc"
    assert config.components.semilocal == tuple(DEFAULT_NEURAL_XC_SEMILOCAL_XC)
    assert config.channels.hf == DEFAULT_NEURAL_XC_HF_INPUT_MODE
    assert config.channels.response_hf == DEFAULT_NEURAL_XC_RESPONSE_HF_MODE
    assert config.channels.response_pt2 == DEFAULT_NEURAL_XC_RESPONSE_PT2_MODE
    assert functional.semilocal_xc == tuple(DEFAULT_NEURAL_XC_SEMILOCAL_XC)


def test_neural_xc_config_exposes_ground_state_pt2_mode():
    config = neural_xc.Config(
        channels=neural_xc.ChannelSpec(
            pt2="local_exact",
            ground_state_pt2="nograd",
        ),
        network=neural_xc.NetworkSpec(hidden_dims=(8,)),
    )

    functional = neural_xc.Functional(config=config)

    assert functional.include_pt2_channel is True
    assert functional.ground_state_pt2_mode == "nograd"
    assert functional.pt2_channel_mode == "local_exact"


def test_neural_xc_lists_wrapped_jax_xc_components_and_exposes_status():
    names = neural_xc.available_semilocal_components()
    infos = neural_xc.available_semilocal_component_infos()
    hse06 = neural_xc.semilocal_component_info("hyb_gga_xc_hse06")

    assert "lda_x" in names
    assert "lyp_c" in names
    assert "hyb_gga_xc_hse06" in names
    assert any(info.name == "hyb_gga_xc_hse06" and info.status == "wrapped" for info in infos)
    assert hse06.status == "wrapped"
    assert "safe semilocal child components" in hse06.reason


def test_neural_xc_config_accepts_wrapped_jax_xc_composite_channel():
    config = neural_xc.Config(
        components=neural_xc.ComponentSpec(
            backend="jax_libxc",
            semilocal=("hyb_gga_xc_hse06",),
        ),
        network=neural_xc.NetworkSpec(
            architecture="residual",
            hidden_dims=(8,),
        ),
        name="wrapped_hse06_component",
    )

    functional = neural_xc.Functional(config=config)

    assert functional.resolved_non_hf_module().channel_names == ("hyb_gga_xc_hse06",)


def test_neural_xc_functional_public_constructor_uses_unified_name():
    functional = neural_xc.Functional(
        hidden_dims=(8,),
        input_feature_mode="canonical",
        architecture="residual",
    )
    via_factory = neural_xc.make_functional(
        hidden_dims=(8,),
        input_feature_mode="canonical",
        architecture="residual",
    )

    assert isinstance(functional, neural_xc.NeuralXCHybridFunctional)
    assert isinstance(via_factory, neural_xc.NeuralXCHybridFunctional)
    assert functional.name == "neural_xc"
    assert functional.input_feature_mode == "canonical"


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

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("td_graddft.neural_xc.base")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("td_graddft.neural_xc.dm21")


def test_neural_xc_factory_stays_as_assembly_layer():
    from pathlib import Path

    factory_text = Path("src/td_graddft/neural_xc/factory.py").read_text()
    forbidden = (
        "class BoundNeuralXCFunctional",
        "class NeuralXCCore",
        "class DLDHMixin",
        "class ProjectedMixin",
        "class ResponseMixin",
        "class BindingMixin",
        "def _dldh_",
        "def projected_hf_grid_contribution_components",
        "def projected_pt2_grid_contribution",
        "def bind_to_molecule(",
        "def bind_to_molecule_for_response(",
        "def bind_to_molecule_for_scf(",
    )

    for pattern in forbidden:
        assert pattern not in factory_text, f"{pattern} should not live in neural_xc/factory.py"


def test_neural_xc_trainer_positive_steps_require_ground_state_data():
    neural_trainer = training.NeuralXCTrainer(
        functional=neural_xc.Functional(hidden_dims=(8,)),
        molecules=[],
    )

    with pytest.raises(ValueError, match="requires at least one"):
        neural_trainer.kernel(steps=1)


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
        scf_gradient_mode="impl",
    )
    trainer = training.NeuralXCTrainer(
        functional=_toy_trainable_density_functional(),
        molecules=[datum],
    )

    result = trainer.kernel(steps=1, training_config=cfg)

    assert captured["training_config"] is cfg
    assert result.history["scf_converged"] == [1.0]


def test_ground_state_datum_from_molecule_requires_cached_hfx_features():
    molecule = _ToyMolecule()

    with pytest.raises(ValueError, match="hfx_local"):
        training.GroundStateDatum.from_molecule(
            molecule,
            target_total_energy=jnp.asarray(0.0),
            require_hfx=True,
        )

    molecule.hfx_local = jnp.zeros((2, 2, 1))
    datum = training.GroundStateDatum.from_molecule(
        molecule,
        target_total_energy=jnp.asarray(0.0),
        require_hfx=True,
    )

    assert datum.molecule is molecule
    assert getattr(datum.molecule, "hfx_nu", None) is None


def test_ground_state_datum_from_molecule_requires_pt2_fields_when_pt2_enabled():
    from td_graddft.scf.molecules import QuadratureGrid, RestrictedMolecule

    grid = QuadratureGrid(coords=jnp.zeros((2, 3)), weights=jnp.ones((2,)))
    molecule = RestrictedMolecule(
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
        training.GroundStateDatum.from_molecule(
            molecule,
            target_total_energy=-1.0,
            functional=functional,
        )


def test_ground_state_datum_from_reference_alias_remains_available():
    molecule = _ToyMolecule()
    datum = training.GroundStateDatum.from_reference(
        molecule,
        target_total_energy=jnp.asarray(0.0),
    )

    assert datum.molecule is molecule


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
