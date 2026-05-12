import jax
import jax.numpy as jnp
import optax
import pytest

pytest.importorskip("td_graddft.nn_rsh.losses", reason="Legacy RSH self-supervised losses were removed.")

import td_graddft.training.targets as training_targets
from td_graddft.nn_rsh import (
    AtomCenteredDensityDescriptorConfig,
    ResolvedRSHParameters,
    atom_centered_density_power_spectrum,
    make_atom_centered_density_rsh_functional,
    make_self_supervised_rsh_loss,
)
from td_graddft.nn_rsh.losses import _neutral_frontier_ip_ea_residuals
from td_graddft.scf import UKSConfig
from pyscf_reference import restricted_reference_from_pyscf
from td_graddft.training import (
    GroundStateDatum,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    make_ground_state_loss_and_grad,
    make_ground_state_train_step,
)


def _make_water_reference(*, with_hfx_aux: bool = False):
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
        compute_local_hfx_features=with_hfx_aux,
        compute_local_hfx_aux=with_hfx_aux,
        hfx_omega_values=(0.0, 0.3, 0.6),
    )


def test_atom_centered_density_power_spectrum_for_water():
    molecule = _make_water_reference()
    config = AtomCenteredDensityDescriptorConfig(
        radial_centers=(0.6, 1.4),
        radial_width=0.5,
        max_angular=2,
    )

    descriptor = atom_centered_density_power_spectrum(molecule, config=config)

    assert descriptor.shape == (3, 9)
    assert jnp.all(jnp.isfinite(descriptor))
    assert jnp.all(descriptor >= 0.0)


def test_atom_centered_density_rsh_functional_initializes_and_trains_one_step():
    molecule = _make_water_reference(with_hfx_aux=True)
    config = AtomCenteredDensityDescriptorConfig(
        radial_centers=(0.6, 1.4),
        radial_width=0.5,
        max_angular=1,
    )
    functional = make_atom_centered_density_rsh_functional(
        descriptor_config=config,
        atom_hidden_dims=(8,),
        pooled_hidden_dims=(),
        embedding_dim=4,
        fallback_omega_values=(0.0, 0.3, 0.6),
    )
    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(0),
        molecule,
        optax.adam(1e-2),
    )
    resolved = functional.resolve_parameters(state.params, molecule)
    assert 0.0 <= float(resolved.sr_hf_fraction) <= float(resolved.lr_hf_fraction) <= 1.0
    assert 0.05 <= float(resolved.omega) <= 0.70

    loss_fn = make_self_supervised_rsh_loss(
        functional,
        training_config=GroundStateTrainingConfig(
            mode="self_consistent",
            scf_gradient_mode="impl",
            scf_max_cycle=3,
            scf_require_convergence=False,
        ),
        janak_weight=1.0,
        fractional_weight=0.0,
        prior_weight=1e-3,
    )
    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.asarray(molecule.mf_energy),
    )
    train_step = make_ground_state_train_step(functional, loss_fn=loss_fn)
    new_state, metrics = train_step(state, datum)

    assert new_state is not None
    assert jnp.isfinite(metrics["loss"])
    assert metrics["janak_frontier_mae"].shape == (1,)
    assert metrics["sr_hf_fraction"].shape == (1,)
    assert metrics["omega"].shape == (1,)
    assert metrics["nonfinite_grad_fraction"][0] == 0.0


def test_params_with_resolved_preserves_hidden_layer_gradients():
    molecule = _make_water_reference(with_hfx_aux=True)
    config = AtomCenteredDensityDescriptorConfig(
        radial_centers=(0.6, 1.4),
        radial_width=0.5,
        max_angular=1,
    )
    functional = make_atom_centered_density_rsh_functional(
        descriptor_config=config,
        atom_hidden_dims=(8,),
        pooled_hidden_dims=(8,),
        embedding_dim=4,
        fallback_omega_values=(0.0, 0.3, 0.6),
    )
    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(0),
        molecule,
        optax.adam(1e-5),
    )
    template = functional.template
    shifted = functional.params_with_resolved(
        state.params,
        ResolvedRSHParameters(
            sr_hf_fraction=template.default_sr_hf_fraction,
            lr_hf_fraction=template.default_lr_hf_fraction,
            omega=template.default_omega,
        ),
        molecule=molecule,
        preserve_network=True,
    )

    loss_fn = make_self_supervised_rsh_loss(
        functional,
        training_config=GroundStateTrainingConfig(
            mode="self_consistent",
            scf_gradient_mode="impl",
            scf_max_cycle=3,
            scf_require_convergence=False,
        ),
        janak_weight=1.0,
        fractional_weight=0.0,
        prior_weight=1e-3,
    )
    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.asarray(molecule.mf_energy),
    )
    loss_and_grad = make_ground_state_loss_and_grad(functional, loss_fn=loss_fn)
    _loss, _metrics, grads = loss_and_grad(shifted, datum)

    atom_hidden_grad = grads["params"]["atom_hidden_0"]
    pooled_hidden_grad = grads["params"]["pooled_hidden_0"]
    embedding_grad = grads["params"]["atomic_number_embedding"]

    def _norm(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return sum(float(jnp.sum(jnp.square(jnp.asarray(leaf)))) for leaf in leaves) ** 0.5

    assert _norm(atom_hidden_grad) > 0.0
    assert _norm(pooled_hidden_grad) > 0.0
    assert _norm(embedding_grad) > 0.0


def test_params_with_raw_output_sets_resolved_rsh_parameters():
    molecule = _make_water_reference(with_hfx_aux=True)
    functional = make_atom_centered_density_rsh_functional(
        descriptor_config=AtomCenteredDensityDescriptorConfig(
            radial_centers=(0.6, 1.4),
            radial_width=0.5,
            max_angular=1,
        ),
        atom_hidden_dims=(8,),
        pooled_hidden_dims=(),
        embedding_dim=4,
        fallback_omega_values=(0.0, 0.3, 0.6),
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(23), molecule)
    raw = jnp.asarray([0.25, -0.75, 0.4], dtype=jnp.float32)

    shifted = functional.params_with_raw_output(params, raw, molecule=molecule)
    resolved = functional.resolve_parameters(shifted, molecule)

    template = functional.template
    expected_sr = template.sr_hf_bounds[0] + (
        template.sr_hf_bounds[1] - template.sr_hf_bounds[0]
    ) * jax.nn.sigmoid(raw[0])
    expected_omega = template.omega_bounds[0] + (
        template.omega_bounds[1] - template.omega_bounds[0]
    ) * jax.nn.sigmoid(raw[2])
    expected_lr = expected_sr + (1.0 - expected_sr) * jax.nn.sigmoid(raw[1])

    assert jnp.allclose(resolved.sr_hf_fraction, expected_sr, atol=1e-6)
    assert jnp.allclose(resolved.lr_hf_fraction, expected_lr, atol=1e-6)
    assert jnp.allclose(resolved.omega, expected_omega, atol=1e-6)
    assert 0.0 <= float(resolved.sr_hf_fraction)
    assert float(resolved.sr_hf_fraction) <= float(resolved.lr_hf_fraction) <= 1.0


def test_params_with_raw_output_can_preserve_or_reset_output_head_kernel():
    molecule = _make_water_reference(with_hfx_aux=True)
    functional = make_atom_centered_density_rsh_functional(
        descriptor_config=AtomCenteredDensityDescriptorConfig(
            radial_centers=(0.6, 1.4),
            radial_width=0.5,
            max_angular=1,
        ),
        atom_hidden_dims=(8,),
        pooled_hidden_dims=(8,),
        embedding_dim=4,
        fallback_omega_values=(0.0, 0.3, 0.6),
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(29), molecule)
    raw = jnp.asarray([0.25, -0.75, 0.4], dtype=jnp.float32)
    original_kernel = params["params"]["output"]["kernel"]

    preserved = functional.params_with_raw_output(
        params,
        raw,
        molecule=molecule,
        preserve_network=True,
    )
    reset = functional.params_with_raw_output(
        params,
        raw,
        molecule=molecule,
        preserve_network=False,
    )

    assert jnp.allclose(preserved["params"]["output"]["kernel"], original_kernel)
    assert jnp.allclose(reset["params"]["output"]["kernel"], jnp.zeros_like(original_kernel))


def test_atom_centered_density_rsh_koopmans_loss_runs_one_step():
    molecule = _make_water_reference(with_hfx_aux=True)
    functional = make_atom_centered_density_rsh_functional(
        descriptor_config=AtomCenteredDensityDescriptorConfig(
            radial_centers=(0.6, 1.4),
            radial_width=0.5,
            max_angular=1,
        ),
        atom_hidden_dims=(8,),
        pooled_hidden_dims=(),
        embedding_dim=4,
        fallback_omega_values=(0.0, 0.3, 0.6),
    )
    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(7),
        molecule,
        optax.adam(1e-3),
    )
    loss_fn = make_self_supervised_rsh_loss(
        functional,
        training_config=GroundStateTrainingConfig(
            mode="self_consistent",
            scf_gradient_mode="impl",
            scf_max_cycle=3,
            scf_require_convergence=False,
        ),
        janak_weight=0.0,
        fractional_weight=0.0,
        koopmans_ip_weight=1.0,
        koopmans_ea_weight=1.0,
        koopmans_lumo_ea_weight=0.5,
        koopmans_cation_config=UKSConfig(
            xc_spec="pbe",
            max_cycle=6,
            conv_tol=1e-8,
            conv_tol_density=1e-7,
            damping=0.35,
            level_shift=0.5,
        ),
        koopmans_anion_config=UKSConfig(
            xc_spec="pbe",
            max_cycle=6,
            conv_tol=1e-8,
            conv_tol_density=1e-7,
            damping=0.35,
            level_shift=0.5,
        ),
        prior_weight=1e-3,
    )
    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.asarray(molecule.mf_energy),
    )
    train_step = make_ground_state_train_step(functional, loss_fn=loss_fn)
    new_state, metrics = train_step(state, datum)

    assert new_state is not None
    assert jnp.isfinite(metrics["loss"])
    assert metrics["koopmans_ip_mae"].shape == (1,)
    assert metrics["koopmans_ea_mae"].shape == (1,)
    assert metrics["koopmans_lumo_ea_mae"].shape == (1,)
    assert metrics["koopmans_neutral_lumo"].shape == (1,)
    assert metrics["koopmans_cation_energy"].shape == (1,)
    assert metrics["koopmans_anion_energy"].shape == (1,)
    assert metrics["nonfinite_grad_fraction"][0] == 0.0


def test_neutral_frontier_ip_ea_residuals_follow_homo_lumo_relations():
    residuals = _neutral_frontier_ip_ea_residuals(
        neutral_homo=jnp.asarray(-0.50),
        neutral_lumo=jnp.asarray(-0.10),
        neutral_energy=jnp.asarray(-10.00),
        cation_energy=jnp.asarray(-9.55),
        anion_energy=jnp.asarray(-10.08),
    )

    assert jnp.allclose(residuals.ip, -0.05, atol=1e-6)
    assert jnp.allclose(residuals.ea, -0.02, atol=1e-6)
    assert jnp.allclose(residuals.gap, 0.03, atol=1e-6)


def test_self_supervised_rsh_loss_penalizes_non_long_range_corrected_limit():
    molecule = _make_water_reference(with_hfx_aux=True)
    functional = make_atom_centered_density_rsh_functional(
        descriptor_config=AtomCenteredDensityDescriptorConfig(
            radial_centers=(0.6, 1.4),
            radial_width=0.5,
            max_angular=1,
        ),
        atom_hidden_dims=(8,),
        pooled_hidden_dims=(),
        embedding_dim=4,
        fallback_omega_values=(0.0, 0.3, 0.6),
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(41), molecule)
    shifted = functional.params_with_resolved(
        params,
        ResolvedRSHParameters(
            sr_hf_fraction=0.20,
            lr_hf_fraction=0.65,
            omega=0.30,
        ),
        molecule=molecule,
        preserve_network=False,
    )
    loss_fn = make_self_supervised_rsh_loss(
        functional,
        training_config=GroundStateTrainingConfig(
            mode="self_consistent",
            scf_gradient_mode="impl",
            scf_max_cycle=1,
            scf_require_convergence=False,
        ),
        janak_weight=0.0,
        fractional_weight=0.0,
        koopmans_ip_weight=0.0,
        koopmans_ea_weight=0.0,
        koopmans_lumo_ea_weight=0.0,
        prior_weight=0.0,
        long_range_correction_weight=2.0,
    )
    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.asarray(molecule.mf_energy),
    )

    loss, metrics = loss_fn(shifted, functional, datum)

    assert jnp.allclose(metrics["long_range_correction_residual"][0], -0.35, atol=1e-6)
    assert jnp.allclose(metrics["long_range_correction_mae"][0], 0.35, atol=1e-6)
    assert jnp.allclose(metrics["long_range_correction_penalty"][0], 0.70, atol=1e-6)
    assert jnp.allclose(loss, 0.70, atol=1e-6)


def test_atom_centered_density_rsh_fixed_orbital_janak_is_finite():
    molecule = _make_water_reference(with_hfx_aux=True)
    functional = make_atom_centered_density_rsh_functional(
        descriptor_config=AtomCenteredDensityDescriptorConfig(
            radial_centers=(0.6, 1.4),
            radial_width=0.5,
            max_angular=1,
        ),
        atom_hidden_dims=(8,),
        pooled_hidden_dims=(),
        embedding_dim=4,
        fallback_omega_values=(0.0, 0.3, 0.6),
    )
    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(31),
        molecule,
        optax.adam(1e-3),
    )
    loss_fn = make_self_supervised_rsh_loss(
        functional,
        training_config=GroundStateTrainingConfig(
            mode="self_consistent",
            janak_frontier_mode="fixed_orbital_ad",
            scf_gradient_mode="impl",
            scf_max_cycle=3,
            scf_require_convergence=False,
        ),
        janak_weight=1.0,
        fractional_weight=0.0,
        koopmans_ip_weight=0.0,
        koopmans_lumo_ea_weight=0.0,
        prior_weight=1e-3,
    )
    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.asarray(molecule.mf_energy),
    )

    loss, metrics = loss_fn(state.params, functional, datum)

    assert jnp.isfinite(loss)
    assert jnp.all(jnp.isfinite(metrics["janak_frontier_mae"]))
    assert jnp.all(jnp.isfinite(metrics["janak_fd_homo"]))
    assert jnp.all(jnp.isfinite(metrics["janak_fd_lumo"]))


def test_self_supervised_rsh_loss_dispatches_to_autodiff_janak(monkeypatch):
    molecule = _make_water_reference(with_hfx_aux=True)
    functional = make_atom_centered_density_rsh_functional(
        descriptor_config=AtomCenteredDensityDescriptorConfig(
            radial_centers=(0.6, 1.4),
            radial_width=0.5,
            max_angular=1,
        ),
        atom_hidden_dims=(8,),
        pooled_hidden_dims=(),
        embedding_dim=4,
        fallback_omega_values=(0.0, 0.3, 0.6),
    )
    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(11),
        molecule,
        optax.adam(1e-3),
    )
    called = {"count": 0}

    def _fake_janak(*_args, **kwargs):
        cfg = kwargs["training_config"]
        assert cfg.janak_frontier_mode == "autodiff"
        called["count"] += 1
        return (
            jnp.asarray(1.5),
            jnp.asarray(0.75),
            jnp.asarray([0.1, -0.2]),
            jnp.asarray([0.3, 0.4]),
        )

    monkeypatch.setattr(
        training_targets,
        "_janak_frontier_penalty_by_mode",
        _fake_janak,
    )

    loss_fn = make_self_supervised_rsh_loss(
        functional,
        training_config=GroundStateTrainingConfig(
            mode="self_consistent",
            janak_frontier_mode="autodiff",
            scf_gradient_mode="impl",
            scf_max_cycle=3,
            scf_require_convergence=False,
        ),
        janak_weight=1.0,
        fractional_weight=0.0,
        prior_weight=1e-3,
    )
    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.asarray(molecule.mf_energy),
    )

    _loss, metrics = loss_fn(state.params, functional, datum)

    assert called["count"] == 1
    assert jnp.isfinite(metrics["loss"])
    assert jnp.allclose(metrics["janak_frontier_mse"][0], 1.5, atol=1e-6)
    assert jnp.allclose(metrics["janak_frontier_mae"][0], 0.75, atol=1e-6)
