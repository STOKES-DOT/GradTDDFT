import copy
from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import optax
import pytest

pytest.importorskip("td_graddft.nn_rsh.losses", reason="Legacy RSH self-supervised losses were removed.")

import td_graddft.training.targets as training_targets
from td_graddft.nn_rsh import (
    AtomCenteredDensityDescriptorConfig,
    RSHParameterHead,
    ResolvedRSHParameters,
    SCFXCContributions,
    atom_centered_density_power_spectrum,
    make_atom_centered_density_rsh_functional,
    make_self_supervised_rsh_loss,
)
from td_graddft.nn_rsh.losses import _neutral_frontier_ip_ea_residuals
from td_graddft.scf import GPU4PYSCF_RKS_RUNTIME_BACKEND, UKSConfig
from td_graddft.scf.gpu4pyscf import GPU4PySCFRKSForwardOptions
from td_graddft.scf.molecules import QuadratureGrid, RestrictedMolecule
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


def test_rsh_parameter_head_uses_two_sigmoid_output_squash():
    head = RSHParameterHead(hidden_dims=())
    params = head.init(jax.random.PRNGKey(19), jnp.ones((4,), dtype=jnp.float32))
    params["params"]["output"]["kernel"] = jnp.zeros_like(
        params["params"]["output"]["kernel"]
    )
    params["params"]["output"]["bias"] = jnp.asarray(
        [100.0, -100.0, 0.0],
        dtype=params["params"]["output"]["bias"].dtype,
    )

    out = head.apply(params, jnp.ones((4,), dtype=jnp.float32))

    assert out.shape == (3,)
    assert jnp.all(out >= 0.0)
    assert jnp.all(out <= 2.0)
    assert jnp.allclose(out, jnp.asarray([2.0, 0.0, 1.0]), atol=1e-6)


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
    raw = jnp.asarray([0.25, 1.25, 0.4], dtype=jnp.float32)

    shifted = functional.params_with_raw_output(params, raw, molecule=molecule)
    resolved = functional.resolve_parameters(shifted, molecule)
    shifted_raw = functional._raw_outputs(shifted, molecule)

    template = functional.template
    expected_sr = template.sr_hf_bounds[0] + (
        template.sr_hf_bounds[1] - template.sr_hf_bounds[0]
    ) * (raw[0] / 2.0)
    expected_omega = template.omega_bounds[0] + (
        template.omega_bounds[1] - template.omega_bounds[0]
    ) * (raw[2] / 2.0)
    expected_lr = expected_sr + (1.0 - expected_sr) * (raw[1] / 2.0)

    assert jnp.allclose(shifted_raw, raw, atol=1e-6)
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
    raw = jnp.asarray([0.25, 1.25, 0.4], dtype=jnp.float32)
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


def test_self_supervised_rsh_prefetches_detached_koopmans_before_value_and_grad(monkeypatch):
    mo_coeff = jnp.eye(2, dtype=jnp.float32)
    mo_occ = jnp.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)
    mo_energy = jnp.asarray([[-0.5, 0.1], [-0.5, 0.1]], dtype=jnp.float32)
    density_half = jnp.einsum("pi,i,qi->pq", mo_coeff, mo_occ[0], mo_coeff)
    molecule = RestrictedMolecule(
        ao=jnp.eye(2, dtype=jnp.float32),
        grid=QuadratureGrid(weights=jnp.ones((2,), dtype=jnp.float32)),
        dipole_integrals=jnp.zeros((3, 2, 2), dtype=jnp.float32),
        rep_tensor=jnp.zeros((2, 2, 2, 2), dtype=jnp.float32),
        mo_coeff=jnp.stack([mo_coeff, mo_coeff], axis=0),
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        rdm1=jnp.stack([density_half, density_half], axis=0),
        h1e=jnp.diag(jnp.asarray([-0.5, 0.1], dtype=jnp.float32)),
        nuclear_repulsion=0.0,
        overlap_matrix=jnp.eye(2, dtype=jnp.float32),
        runtime_scf_backend=GPU4PYSCF_RKS_RUNTIME_BACKEND,
        runtime_scf_options=GPU4PySCFRKSForwardOptions(
            atom="H 0 0 0; H 0 0 0.74",
            basis="sto-3g",
            xc_spec="pbe",
        ),
    )

    @dataclass(frozen=True)
    class FakeResolved:
        sr_hf_fraction: object
        lr_hf_fraction: object
        omega: object

    @dataclass(frozen=True)
    class FakeBound:
        resolved_params: FakeResolved
        local_xc_spec: str = "hf"

        def energy_from_molecule(self, molecule_in):
            del molecule_in
            return jnp.asarray(0.0, dtype=jnp.float32)

    class FakeFunctional:
        def bind_to_molecule(self, params, molecule_in):
            del molecule_in
            return FakeBound(
                FakeResolved(
                    sr_hf_fraction=params["sr"],
                    lr_hf_fraction=params["lr"],
                    omega=params["omega"],
                )
            )

        def resolve_parameters(self, params, molecule_in):
            del molecule_in
            return FakeResolved(
                sr_hf_fraction=params["sr"],
                lr_hf_fraction=params["lr"],
                omega=params["omega"],
            )

    def fake_resolve_training_molecule(params, functional, molecule_in, cfg):
        del params, functional, cfg
        return molecule_in

    def fake_predict_energy(params, functional, molecule_in):
        del functional, molecule_in
        if params is None:
            return jnp.asarray(-1.0, dtype=jnp.float32)
        return jnp.asarray(params["neutral_energy"], dtype=jnp.float32)

    diagnostic_calls = []
    inside_value_and_grad = False

    def fake_koopmans_diagnostic(molecule_in, bound_xc, **kwargs):
        del molecule_in, bound_xc, kwargs
        diagnostic_calls.append(inside_value_and_grad)
        if inside_value_and_grad:
            raise AssertionError("Detached Koopmans charged branch must be prefetched outside AD.")
        return training_targets.KoopmansIPEADiagnostic(
            neutral_energy=jnp.asarray(-1.0, dtype=jnp.float32),
            cation_energy=jnp.asarray(-0.4, dtype=jnp.float32),
            anion_energy=jnp.asarray(-1.2, dtype=jnp.float32),
            ip_delta_scf=jnp.asarray(0.6, dtype=jnp.float32),
            ea_delta_scf=jnp.asarray(0.2, dtype=jnp.float32),
            neutral_homo_energy=jnp.asarray(-0.5, dtype=jnp.float32),
            anion_homo_energy=jnp.asarray(-0.3, dtype=jnp.float32),
            ip_residual=jnp.asarray(0.1, dtype=jnp.float32),
            ea_residual=jnp.asarray(-0.1, dtype=jnp.float32),
            cation_converged=True,
            anion_converged=True,
            cation_result=SimpleNamespace(converged=True),
            anion_result=SimpleNamespace(converged=True),
        )

    monkeypatch.setattr(
        training_targets,
        "_resolve_training_molecule_with_mode",
        fake_resolve_training_molecule,
    )
    monkeypatch.setattr(
        training_targets,
        "_predict_ground_state_total_energy_from_molecule",
        fake_predict_energy,
    )
    monkeypatch.setattr(
        training_targets,
        "koopmans_ip_ea_diagnostic",
        fake_koopmans_diagnostic,
    )
    original_value_and_grad = jax.value_and_grad

    def wrapped_value_and_grad(*args, **kwargs):
        value_and_grad_fn = original_value_and_grad(*args, **kwargs)

        def wrapped(*call_args, **call_kwargs):
            nonlocal inside_value_and_grad
            inside_value_and_grad = True
            try:
                return value_and_grad_fn(*call_args, **call_kwargs)
            finally:
                inside_value_and_grad = False

        return wrapped

    monkeypatch.setattr(jax, "value_and_grad", wrapped_value_and_grad)

    functional = FakeFunctional()
    cfg = GroundStateTrainingConfig(
        mode="self_consistent",
        scf_gradient_mode="impl",
        scf_implicit_forward_mode="input_state",
        scf_require_convergence=False,
    )
    loss_fn = make_self_supervised_rsh_loss(
        functional,
        training_config=cfg,
        janak_weight=0.0,
        fractional_weight=0.0,
        koopmans_ip_weight=1.0,
        koopmans_ea_weight=1.0,
        koopmans_lumo_ea_weight=1.0,
        prior_weight=0.0,
    )

    neutral_provider_calls = []

    def neutral_provider(params, functional_in, molecule_in):
        del params, functional_in
        neutral_provider_calls.append(True)
        out = copy.copy(molecule_in)
        object.__setattr__(out, "_neutral_forward_marker", True)
        return out

    loss_and_grad = make_ground_state_loss_and_grad(
        functional,
        training_config=cfg,
        loss_fn=loss_fn,
        runtime_forward_state_provider=neutral_provider,
    )
    params = {
        "sr": jnp.asarray(0.2, dtype=jnp.float32),
        "lr": jnp.asarray(1.0, dtype=jnp.float32),
        "omega": jnp.asarray(0.3, dtype=jnp.float32),
        "neutral_energy": jnp.asarray(-1.0, dtype=jnp.float32),
    }
    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.asarray(-1.0, dtype=jnp.float32),
    )

    _loss, metrics, grads = loss_and_grad(params, datum)

    assert neutral_provider_calls == [True]
    assert diagnostic_calls == [False]
    assert jnp.isfinite(metrics["loss"])
    assert jnp.isfinite(grads["neutral_energy"])


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


def test_fixed_density_rsh_loss_uses_precomputed_charge_states_without_scf(monkeypatch):
    import td_graddft.nn_rsh.losses as rsh_losses

    assert hasattr(rsh_losses, "FixedDensityRSHDatum")
    assert hasattr(rsh_losses, "make_fixed_density_rsh_loss")

    mo_coeff = jnp.eye(2, dtype=jnp.float32)

    def molecule_with_occ(mo_occ, *, nuclear_repulsion):
        density = jnp.einsum("si,pi,qi->spq", mo_occ, mo_coeff, mo_coeff)
        return RestrictedMolecule(
            ao=jnp.eye(2, dtype=jnp.float32),
            grid=QuadratureGrid(weights=jnp.ones((2,), dtype=jnp.float32)),
            dipole_integrals=jnp.zeros((3, 2, 2), dtype=jnp.float32),
            rep_tensor=jnp.zeros((2, 2, 2, 2), dtype=jnp.float32),
            mo_coeff=jnp.stack([mo_coeff, mo_coeff], axis=0),
            mo_occ=mo_occ,
            mo_energy=jnp.asarray([[-0.5, 0.2], [-0.5, 0.2]], dtype=jnp.float32),
            rdm1=density,
            h1e=jnp.diag(jnp.asarray([-0.5, 0.2], dtype=jnp.float32)),
            nuclear_repulsion=float(nuclear_repulsion),
            overlap_matrix=jnp.eye(2, dtype=jnp.float32),
            hfx_omega_values=jnp.asarray((0.0,), dtype=jnp.float32),
            hfx_nu=jnp.zeros((1, 2, 2), dtype=jnp.float32),
        )

    neutral = molecule_with_occ(
        jnp.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.float32),
        nuclear_repulsion=0.0,
    )
    cation = molecule_with_occ(
        jnp.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.float32),
        nuclear_repulsion=1.0,
    )
    anion = molecule_with_occ(
        jnp.asarray([[1.0, 1.0], [1.0, 0.0]], dtype=jnp.float32),
        nuclear_repulsion=2.0,
    )

    def fail_scf(*_args, **_kwargs):
        raise AssertionError("fixed-density RSH loss must not resolve any SCF state")

    monkeypatch.setattr(training_targets, "_resolve_training_molecule_with_mode", fail_scf)
    monkeypatch.setattr(training_targets, "koopmans_ip_ea_diagnostic", fail_scf)

    @dataclass(frozen=True)
    class FakeResolved:
        sr_hf_fraction: object
        lr_hf_fraction: object
        omega: object

    @dataclass(frozen=True)
    class FakeBound:
        resolved_params: FakeResolved

        def energy_from_molecule(self, molecule_in):
            scale = 1.0 + jnp.asarray(molecule_in.nuclear_repulsion, dtype=jnp.float32)
            return self.resolved_params.omega * scale

        def unrestricted_scf_components(self, molecule_in):
            del molecule_in
            v_rho = jnp.zeros((2,), dtype=jnp.float32)
            v_grad = jnp.zeros((2, 3), dtype=jnp.float32)
            extra = jnp.diag(
                jnp.asarray([0.0, self.resolved_params.sr_hf_fraction], dtype=jnp.float32)
            )
            return (
                v_rho,
                v_rho,
                v_grad,
                v_grad,
                "LDA",
                jnp.asarray(0.0, dtype=jnp.float32),
                extra,
                extra,
            )

    class FakeFunctional:
        def bind_to_molecule(self, params, molecule_in):
            del molecule_in
            return FakeBound(
                FakeResolved(
                    sr_hf_fraction=params["frontier_shift"],
                    lr_hf_fraction=jnp.asarray(1.0, dtype=jnp.float32),
                    omega=params["energy_scale"],
                )
            )

        def resolve_parameters(self, params, molecule_in):
            del molecule_in
            return FakeResolved(
                sr_hf_fraction=params["frontier_shift"],
                lr_hf_fraction=jnp.asarray(1.0, dtype=jnp.float32),
                omega=params["energy_scale"],
            )

    datum = rsh_losses.FixedDensityRSHDatum(
        molecule=neutral,
        cation_molecule=cation,
        anion_molecule=anion,
        target_total_energy=jnp.asarray(0.0, dtype=jnp.float32),
    )
    functional = FakeFunctional()
    loss_fn = rsh_losses.make_fixed_density_rsh_loss(
        functional,
        koopmans_ip_weight=1.0,
        koopmans_ea_weight=1.0,
        koopmans_lumo_ea_weight=1.0,
        koopmans_loss_kind="squared",
        prior_weight=0.0,
    )
    loss_and_grad = make_ground_state_loss_and_grad(
        functional,
        training_config=GroundStateTrainingConfig(mode="fixed_density"),
        loss_fn=loss_fn,
    )
    params = {
        "frontier_shift": jnp.asarray(0.1, dtype=jnp.float32),
        "energy_scale": jnp.asarray(0.2, dtype=jnp.float32),
    }

    loss, metrics, grads = loss_and_grad(params, datum)

    assert jnp.isfinite(loss)
    assert metrics["fixed_density_rsh"][0] == 1.0
    assert metrics["koopmans_cation_energy"].shape == (1,)
    assert metrics["koopmans_anion_energy"].shape == (1,)
    assert jnp.abs(grads["energy_scale"]) > 0.0
    assert jnp.abs(grads["frontier_shift"]) > 0.0


def test_fixed_density_rsh_loss_binds_each_charge_state_density():
    import td_graddft.nn_rsh.losses as rsh_losses

    mo_coeff = jnp.eye(2, dtype=jnp.float32)

    def molecule_with_occ(mo_occ):
        density = jnp.einsum("si,pi,qi->spq", mo_occ, mo_coeff, mo_coeff)
        return RestrictedMolecule(
            ao=jnp.eye(2, dtype=jnp.float32),
            grid=QuadratureGrid(weights=jnp.ones((2,), dtype=jnp.float32)),
            dipole_integrals=jnp.zeros((3, 2, 2), dtype=jnp.float32),
            rep_tensor=jnp.zeros((2, 2, 2, 2), dtype=jnp.float32),
            mo_coeff=jnp.stack([mo_coeff, mo_coeff], axis=0),
            mo_occ=mo_occ,
            mo_energy=jnp.asarray([[-0.5, 0.2], [-0.5, 0.2]], dtype=jnp.float32),
            rdm1=density,
            h1e=jnp.zeros((2, 2), dtype=jnp.float32),
            nuclear_repulsion=0.0,
            overlap_matrix=jnp.eye(2, dtype=jnp.float32),
            hfx_omega_values=jnp.asarray((0.0,), dtype=jnp.float32),
            hfx_nu=jnp.zeros((1, 2, 2), dtype=jnp.float32),
        )

    neutral = molecule_with_occ(jnp.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.float32))
    cation = molecule_with_occ(jnp.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.float32))
    anion = molecule_with_occ(jnp.asarray([[1.0, 1.0], [1.0, 0.0]], dtype=jnp.float32))

    @dataclass(frozen=True)
    class FakeResolved:
        sr_hf_fraction: object
        lr_hf_fraction: object
        omega: object

    @dataclass(frozen=True)
    class FakeBound:
        marker: object
        energy_scale: object
        frontier_shift: object
        resolved_params: FakeResolved

        def energy_from_molecule(self, molecule_in):
            del molecule_in
            return self.energy_scale * self.marker

        def unrestricted_scf_components(self, molecule_in):
            del molecule_in
            v_rho = jnp.zeros((2,), dtype=jnp.float32)
            v_grad = jnp.zeros((2, 3), dtype=jnp.float32)
            extra = jnp.diag(
                jnp.asarray([0.0, self.frontier_shift * self.marker], dtype=jnp.float32)
            )
            return (
                v_rho,
                v_rho,
                v_grad,
                v_grad,
                "LDA",
                jnp.asarray(0.0, dtype=jnp.float32),
                extra,
                extra,
            )

    bind_markers = []

    class FakeFunctional:
        def bind_to_molecule(self, params, molecule_in):
            marker = jnp.trace(jnp.asarray(molecule_in.rdm1).sum(axis=0))
            bind_markers.append(float(marker))
            return FakeBound(
                marker=marker,
                energy_scale=params["energy_scale"],
                frontier_shift=params["frontier_shift"],
                resolved_params=FakeResolved(
                    sr_hf_fraction=params["frontier_shift"],
                    lr_hf_fraction=jnp.asarray(1.0, dtype=jnp.float32),
                    omega=params["energy_scale"],
                ),
            )

        def resolve_parameters(self, params, molecule_in):
            del molecule_in
            return FakeResolved(
                sr_hf_fraction=params["frontier_shift"],
                lr_hf_fraction=jnp.asarray(1.0, dtype=jnp.float32),
                omega=params["energy_scale"],
            )

    functional = FakeFunctional()
    loss_fn = rsh_losses.make_fixed_density_rsh_loss(
        functional,
        koopmans_ip_weight=1.0,
        koopmans_ea_weight=1.0,
        koopmans_lumo_ea_weight=1.0,
        koopmans_loss_kind="squared",
        prior_weight=0.0,
    )
    params = {
        "energy_scale": jnp.asarray(0.5, dtype=jnp.float32),
        "frontier_shift": jnp.asarray(0.1, dtype=jnp.float32),
    }
    datum = rsh_losses.FixedDensityRSHDatum(
        molecule=neutral,
        cation_molecule=cation,
        anion_molecule=anion,
        target_total_energy=jnp.asarray(0.0, dtype=jnp.float32),
    )

    _loss, metrics = loss_fn(params, functional, datum)

    assert bind_markers[:3] == [2.0, 1.0, 3.0]
    assert jnp.allclose(metrics["koopmans_neutral_energy"][0], 1.0, atol=1e-6)
    assert jnp.allclose(metrics["koopmans_cation_energy"][0], 0.5, atol=1e-6)
    assert jnp.allclose(metrics["koopmans_anion_energy"][0], 1.5, atol=1e-6)


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
