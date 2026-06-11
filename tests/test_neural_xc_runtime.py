from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn

from td_graddft.xc_backend.jax_libxc import b3lyp_component_basis
from td_graddft.xc_backend import jax_xc_adapter
from td_graddft.neural_xc import (
    ResidualMixingMLP,
    available_semilocal_components,
    make_custom_semilocal_module,
    make_libxc_semilocal_module,
    make_neural_xc_functional,
)
from td_graddft.neural_xc.factory import NeuralXCFunctional
from td_graddft.neural_xc.inputs import (
    ChunkedHFXNu,
    _local_pt2_feature_from_restricted_orbitals,
    _local_pt2_feature_from_unrestricted_orbitals,
)
import td_graddft.neural_xc.binding as neural_xc_binding
import td_graddft.neural_xc.model as neural_xc_model
from td_graddft.features import restricted_grid_features
import td_graddft.scf.differentiable as scf_differentiable
from td_graddft.scf.xc_energy import xc_energy_and_potential_from_density
from pyscf_reference import restricted_reference_from_pyscf
from td_graddft.spectra import HARTREE_TO_EV, oscillator_strengths
from td_graddft.tddft import (
    RestrictedCasidaTDDFT,
    UnrestrictedCasidaTDDFT,
    build_unrestricted_response_matrices,
)
from td_graddft.tddft.cisd import (
    restricted_cisd_second_order_correction,
    unrestricted_cisd_second_order_correction,
)
import td_graddft.tddft.response as tddft_response
from td_graddft.tddft.response import (
    build_restricted_response_matrices,
    build_restricted_tda_matrix,
    build_restricted_tda_operator,
)
from td_graddft.tddft.cisd import restricted_cisd_second_order_correction
from td_graddft.training import (
    GroundStateDatum,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    ground_state_mse_loss,
    make_ground_state_train_step,
    predict_excitation_energies,
    predict_ground_state_total_energy,
)


@dataclass
class _Grid:
    weights: jnp.ndarray
    coords: jnp.ndarray | None = None


@dataclass
class _ToyMolecule:
    ao: jnp.ndarray
    ao_deriv1: jnp.ndarray
    ao_laplacian: jnp.ndarray | None
    grid: _Grid
    rep_tensor: jnp.ndarray
    mo_coeff: jnp.ndarray
    mo_occ: jnp.ndarray
    mo_energy: jnp.ndarray
    rdm1: jnp.ndarray
    h1e: jnp.ndarray
    nuclear_repulsion: float
    hfx_local: jnp.ndarray | None = None
    hfx_omega_values: tuple[float, ...] | None = None
    hfx_nu: jnp.ndarray | None = None
    pt2_local: jnp.ndarray | None = None

    def density(self):
        return jnp.einsum("spq,rp,rq->rs", self.rdm1, self.ao, self.ao)


def _make_toy_molecule():
    ao = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    ao_deriv1 = jnp.array(
        [
            ao,
            [[0.2, 0.0], [0.0, 0.2]],
            [[0.0, 0.1], [0.1, 0.0]],
            [[0.1, 0.0], [0.0, 0.1]],
        ]
    )
    mo_coeff = jnp.stack([jnp.eye(2), jnp.eye(2)], axis=0)
    mo_occ = jnp.array([[1.0, 0.0], [1.0, 0.0]])
    mo_energy = jnp.array([[0.0, 2.0], [0.0, 2.0]])
    rdm1 = jnp.array(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0]],
        ]
    )
    rep_tensor = jnp.zeros((2, 2, 2, 2))
    rep_tensor = rep_tensor.at[0, 0, 0, 0].set(1.0)
    hfx_local = jnp.array(
        [
            [[-0.30, -0.21], [-0.10, -0.07]],
            [[-0.20, -0.14], [-0.05, -0.035]],
        ]
    )
    return _ToyMolecule(
        ao=ao,
        ao_deriv1=ao_deriv1,
        ao_laplacian=jnp.array([[0.05, -0.02], [-0.01, 0.04]]),
        grid=_Grid(weights=jnp.array([1.0, 1.0])),
        rep_tensor=rep_tensor,
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        rdm1=rdm1,
        h1e=jnp.zeros((2, 2)),
        nuclear_repulsion=0.0,
        hfx_local=hfx_local,
        hfx_omega_values=(0.0, 0.233),
    )


def _make_open_shell_toy_molecule():
    ao = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    ao_deriv1 = jnp.array(
        [
            ao,
            [[0.2, 0.0], [0.0, 0.2]],
            [[0.0, 0.1], [0.1, 0.0]],
            [[0.1, 0.0], [0.0, 0.1]],
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
        ao_laplacian=jnp.array([[0.05, -0.02], [-0.01, 0.04]]),
        grid=_Grid(weights=jnp.array([1.0, 1.0])),
        rep_tensor=jnp.zeros((2, 2, 2, 2)),
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        rdm1=rdm1,
        h1e=jnp.zeros((2, 2)),
        nuclear_repulsion=0.0,
        hfx_local=jnp.array(
            [
                [[-0.30, -0.21], [-0.10, -0.07]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ),
        hfx_omega_values=(0.0, 0.233),
    )


def _toy_hfx_nu_cache():
    nu = jnp.zeros((1, 2, 2, 2), dtype=jnp.float64)
    nu = nu.at[0, 0, 0, 0].set(1.0)
    nu = nu.at[0, 0, 1, 1].set(0.25)
    nu = nu.at[0, 1, 0, 0].set(0.5)
    nu = nu.at[0, 1, 1, 1].set(0.75)
    return nu


def test_chunked_hfx_nu_matches_dense_hf_grid_contribution():
    dense_molecule = _make_toy_molecule()
    dense_nu = _toy_hfx_nu_cache()
    dense_molecule.hfx_local = None
    dense_molecule.hfx_nu = dense_nu

    chunked_molecule = _make_toy_molecule()
    chunked_molecule.hfx_local = None
    chunked_molecule.hfx_nu = None
    chunked_molecule.hfx_nu_api = ChunkedHFXNu.from_dense(dense_nu, chunk_size=1)

    functional = make_neural_xc_functional(
        semilocal_xc=("gga_x_pbe", "gga_c_pbe"),
        hidden_dims=(8,),
    )

    dense = functional.projected_hf_grid_contribution_components(dense_molecule)
    chunked = functional.projected_hf_grid_contribution_components(chunked_molecule)

    for dense_part, chunked_part in zip(dense, chunked, strict=True):
        assert jnp.allclose(chunked_part, dense_part)


def test_hf_grid_contribution_prefers_cached_hfx_local_over_chunked_nu():
    class RaisingChunkedNu:
        shape = (1, 2, 2, 2)
        chunk_size = 1

        def grid_chunk(self, start, stop):
            raise AssertionError("hfx_local should avoid reading hfx_nu chunks")

        def grid_chunk_padded(self, start, chunk_size):
            raise AssertionError("hfx_local should avoid reading padded hfx_nu chunks")

    molecule = _make_toy_molecule()
    molecule.hfx_nu = None
    molecule.hfx_nu_api = RaisingChunkedNu()
    expected_a = molecule.hfx_local[0, :, 0]
    expected_b = molecule.hfx_local[1, :, 0]

    functional = make_neural_xc_functional(
        semilocal_xc=("gga_x_pbe", "gga_c_pbe"),
        hidden_dims=(8,),
    )

    total, hfx_a, hfx_b = functional.projected_hf_grid_contribution_components(molecule)

    assert jnp.allclose(hfx_a, expected_a)
    assert jnp.allclose(hfx_b, expected_b)
    assert jnp.allclose(total, expected_a + expected_b)


def test_chunked_hfx_nu_matches_dense_hfx_fock_contraction():
    dense_molecule = _make_toy_molecule()
    dense_nu = _toy_hfx_nu_cache()
    dense_molecule.hfx_nu = dense_nu

    chunked_molecule = _make_toy_molecule()
    chunked_molecule.hfx_nu = None
    chunked_molecule.hfx_nu_api = ChunkedHFXNu.from_dense(dense_nu, chunk_size=1)

    functional = make_neural_xc_functional(
        semilocal_xc=("gga_x_pbe", "gga_c_pbe"),
        hidden_dims=(8,),
    )
    grad_a = jnp.asarray([[0.2], [0.4]], dtype=dense_nu.dtype)
    grad_b = jnp.asarray([[0.3], [0.1]], dtype=dense_nu.dtype)

    dense_fock, dense_used = functional._contract_hfx_feature_gradients_to_restricted_fock(
        dense_molecule,
        grad_a,
        grad_b,
    )
    chunked_fock, chunked_used = functional._contract_hfx_feature_gradients_to_restricted_fock(
        chunked_molecule,
        grad_a,
        grad_b,
    )

    assert dense_used is True
    assert chunked_used is True
    assert jnp.allclose(chunked_fock, dense_fock)


def _make_pt2_toy_molecule():
    ao = jnp.array(
        [
            [1.0 / jnp.sqrt(2.0), 1.0 / jnp.sqrt(2.0)],
            [1.0 / jnp.sqrt(2.0), -1.0 / jnp.sqrt(2.0)],
        ]
    )
    ao_deriv1 = jnp.array(
        [
            ao,
            [[0.1, 0.0], [0.0, 0.1]],
            [[0.0, 0.1], [0.1, 0.0]],
            [[0.05, 0.0], [0.0, -0.05]],
        ]
    )
    mo_coeff = jnp.stack([jnp.eye(2), jnp.eye(2)], axis=0)
    mo_occ = jnp.array([[1.0, 0.0], [1.0, 0.0]])
    mo_energy = jnp.array([[0.0, 2.0], [0.0, 2.0]])
    rdm1 = jnp.array(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0]],
        ]
    )
    rep_tensor = jnp.zeros((2, 2, 2, 2))
    rep_tensor = rep_tensor.at[0, 1, 0, 1].set(1.0)
    hfx_local = jnp.array(
        [
            [[-0.18, -0.126], [-0.06, -0.042]],
            [[-0.12, -0.084], [-0.03, -0.021]],
        ]
    )
    return _ToyMolecule(
        ao=ao,
        ao_deriv1=ao_deriv1,
        ao_laplacian=jnp.array([[0.04, -0.01], [0.02, -0.03]]),
        grid=_Grid(weights=jnp.array([1.0, 1.0])),
        rep_tensor=rep_tensor,
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        rdm1=rdm1,
        h1e=jnp.zeros((2, 2)),
        nuclear_repulsion=0.0,
        hfx_local=hfx_local,
        hfx_omega_values=(0.0, 0.233),
    )


def _make_three_grid_pt2_toy_molecule():
    molecule = _make_pt2_toy_molecule()
    ao = jnp.concatenate([molecule.ao, molecule.ao[:1]], axis=0)
    ao_deriv1 = jnp.concatenate([molecule.ao_deriv1, molecule.ao_deriv1[:, :1]], axis=1)
    ao_laplacian = jnp.concatenate([molecule.ao_laplacian, molecule.ao_laplacian[:1]], axis=0)
    return replace(
        molecule,
        ao=ao,
        ao_deriv1=ao_deriv1,
        ao_laplacian=ao_laplacian,
        grid=_Grid(weights=jnp.asarray([1.0, 0.8, 0.6])),
    )


def _pyscf_or_skip():
    try:
        from pyscf import dft, gto  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("PySCF is required for response-kernel comparison tests.")


def _make_water_b3lyp_reference():
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
    mf.xc = "b3lyp"
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF RKS(B3LYP/STO-3G) did not converge for water.")
    return mf


def test_neural_xc_functional_trains_and_produces_excitation():
    molecule = _make_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=(8, 8),
        response_hf_mode="approx",
        name="toy_neural_xc",
    )
    datum = GroundStateDatum(molecule=molecule, target_total_energy=jnp.array(0.2))

    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(0),
        molecule,
        optax.adam(0.05),
    )
    train_step = make_ground_state_train_step(functional)
    initial_loss, _ = ground_state_mse_loss(state.params, functional, datum)

    for _ in range(100):
        state, _ = train_step(state, datum)

    final_loss, _ = ground_state_mse_loss(state.params, functional, datum)
    energy = predict_ground_state_total_energy(state.params, functional, molecule)
    solver = RestrictedCasidaTDDFT(
        molecule=molecule,
        xc_functional=functional,
        xc_params=state.params,
    )
    vind, flat_a = solver.gen_tda_vind()
    response = vind(jnp.ones((1, flat_a.shape[0])))
    alpha = functional.effective_exchange_fraction(state.params, molecule)

    assert final_loss < initial_loss
    assert jnp.isfinite(energy)
    assert response.shape == (1, flat_a.shape[0])
    assert jnp.all(jnp.isfinite(response))
    assert 0.0 <= alpha <= 1.0


def test_bounded_sigmoid_coefficients_are_nonnegative_and_bounded():
    molecule = _make_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc=("lda_x", "gga_x_pbe"),
        hidden_dims=(8, 8),
        name="toy_bounded_sigmoid_neural_xc",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(123), molecule)
    features = restricted_grid_features(molecule)
    semilocal = functional.semilocal_energy_density(features)
    hf_total, hf_a, hf_b = functional.projected_hf_grid_contribution_components(
        molecule,
        features=features,
    )
    coefficients = functional.channel_coefficients(
        params,
        features,
        molecule=molecule,
        semilocal_energy_density=semilocal,
        hf_energy_density=hf_total,
        hf_spin_energy_density=(hf_a, hf_b),
    )

    assert coefficients.shape[-1] == 3
    assert jnp.all(jnp.isfinite(coefficients))
    assert jnp.all(coefficients >= 0.0)
    assert jnp.all(coefficients <= functional.kernel_clip + 1e-6)


def test_semilocal_xc_alias_expands_to_component_channels_for_neural_basis():
    molecule = _make_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=(8, 8),
        name="toy_pbe_alias_channel_resolution",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(7), molecule)
    features = restricted_grid_features(molecule)
    channels = functional.semilocal_energy_density_channels(features)
    coefficients = functional.channel_coefficients(
        params,
        features,
        molecule=molecule,
        semilocal_energy_density=jnp.sum(channels, axis=-1),
        hf_energy_density=jnp.zeros(features.rho.shape),
        hf_spin_energy_density=(jnp.zeros(features.rho.shape), jnp.zeros(features.rho.shape)),
    )

    assert channels.shape[-1] == 2
    assert coefficients.shape[-1] >= 2
    assert jnp.all(jnp.isfinite(channels))


def test_neural_xc_rejects_experimental_jax_xc_semilocal_by_default(monkeypatch):
    import td_graddft.xc_backend.jax_xc_adapter as jax_xc_adapter

    class FakeModule:
        __version__ = "fake"

        @staticmethod
        def hyb_gga_xc_b97(*, polarized=False):
            del polarized
            return lambda rho_fn, r, mo_fn=None: rho_fn(r)

    monkeypatch.setattr(
        jax_xc_adapter,
        "load_jax_xc",
        lambda: (jax_xc_adapter._SafeJAXXCModule(FakeModule()), "upstream"),
    )

    with pytest.raises(ValueError, match="allow_experimental_jax_xc=True"):
        make_neural_xc_functional(
            semilocal_xc="hyb_gga_xc_b97",
            hidden_dims=(8,),
        )


def test_neural_xc_accepts_experimental_jax_xc_with_explicit_opt_in(monkeypatch):
    import td_graddft.xc_backend.jax_xc_adapter as jax_xc_adapter

    class FakeModule:
        __version__ = "fake"

        @staticmethod
        def gga_x_rpbe(*, polarized=False):
            del polarized
            return lambda rho_fn, r, mo_fn=None: rho_fn(r)

    monkeypatch.setattr(
        jax_xc_adapter,
        "load_jax_xc",
        lambda: (jax_xc_adapter._SafeJAXXCModule(FakeModule()), "upstream"),
    )

    functional = make_neural_xc_functional(
        semilocal_xc="gga_x_rpbe",
        hidden_dims=(8,),
        allow_experimental_jax_xc=True,
    )

    assert functional.allow_experimental_jax_xc is True
    assert functional.resolved_non_hf_module().channel_names == ("gga_x_rpbe",)


def test_neural_xc_accepts_dynamic_mgga_with_explicit_opt_in(monkeypatch):
    import td_graddft.xc_backend.jax_xc_adapter as jax_xc_adapter

    class FakeModule:
        __version__ = "fake"

        @staticmethod
        def mgga_x_demo(*, polarized=False):
            del polarized

            def functional(rho_fn, r, mo_fn=None):
                if mo_fn is None:
                    raise ValueError("mo_fn is required for MGGA")
                mo_jac = jax.jacfwd(mo_fn)(r)
                tau = 0.5 * jnp.sum(mo_jac * mo_jac)
                return rho_fn(r) + 0.25 * tau

            return functional

    monkeypatch.setattr(
        jax_xc_adapter,
        "load_jax_xc",
        lambda: (jax_xc_adapter._SafeJAXXCModule(FakeModule()), "upstream"),
    )

    with pytest.raises(ValueError, match="allow_experimental_jax_xc=True"):
        make_neural_xc_functional(
            semilocal_xc="mgga_x_demo",
            hidden_dims=(8,),
        )

    functional = make_neural_xc_functional(
        semilocal_xc="mgga_x_demo",
        hidden_dims=(8,),
        allow_experimental_jax_xc=True,
    )
    features = restricted_grid_features(_make_toy_molecule())
    channels = functional.semilocal_energy_density_channels(features)
    kernel = functional._projected_semilocal_kernel(features)

    assert functional.resolved_non_hf_module().channel_names == ("mgga_x_demo",)
    assert channels.shape[-1] == 1
    assert jnp.all(jnp.isfinite(channels))
    assert kernel.shape == features.rho.shape
    assert jnp.all(jnp.isfinite(kernel))


def test_graddft_coeff_basis_hf_pt2_heads_mixing_transform_remains_smooth_above_one():
    functional = make_neural_xc_functional(
        semilocal_xc=("lda_x", "gga_c_pbe"),
        include_pt2_channel=True,
        hidden_dims=(8, 8),
        name="toy_hybrid_head_smooth_mixing_transform",
    )
    coefficients = jnp.asarray([1.0, 1.0, 1.5, 0.5], dtype=jnp.float32)
    transformed = functional._sanitize_coefficients(coefficients)

    assert jnp.allclose(
        transformed[2:],
        jnp.asarray([0.75, 0.25], dtype=jnp.float32),
        atol=1e-6,
    )
    assert jnp.all((transformed[2:] > 0.0) & (transformed[2:] < 1.0))

    hf_grad = jax.grad(
        lambda x: functional._sanitize_coefficients(
            jnp.asarray([1.0, 1.0, x, 0.5], dtype=jnp.float32)
        )[2]
    )(1.5)
    pt2_grad = jax.grad(
        lambda x: functional._sanitize_coefficients(
            jnp.asarray([1.0, 1.0, 1.5, x], dtype=jnp.float32)
        )[3]
    )(0.5)

    assert float(hf_grad) > 0.0
    assert float(pt2_grad) > 0.0


def test_graddft_coeff_basis_hf_pt2_heads_sanitizes_semilocal_and_heads_separately():
    functional = make_neural_xc_functional(
        semilocal_xc=("lda_x", "gga_c_pbe"),
        include_pt2_channel=True,
        hidden_dims=(8, 8),
        name="toy_hybrid_head_sanitize",
    )
    coefficients = jnp.asarray([6.5, 1.5, 1.5, 0.5], dtype=jnp.float32)
    transformed = functional._sanitize_coefficients(coefficients)

    assert jnp.allclose(
        transformed[:2],
        jnp.asarray([functional.kernel_clip, 1.5], dtype=jnp.float32),
    )
    assert jnp.allclose(
        transformed[2:],
        jnp.asarray([0.75, 0.25], dtype=jnp.float32),
        atol=1e-6,
    )

    hf_grad = jax.grad(
        lambda x: functional._sanitize_coefficients(
            jnp.asarray([1.0, 1.0, 0.5, x], dtype=jnp.float32)
        )[-1]
    )(0.5)
    pt2_grad = jax.grad(
        lambda x: functional._sanitize_coefficients(
            jnp.asarray([1.0, 1.0, x, 0.5], dtype=jnp.float32)
        )[-2]
    )(0.5)

    assert float(hf_grad) > 0.0
    assert float(pt2_grad) > 0.0


def test_bind_to_molecule_for_scf_skips_projected_energy_density_assembly():
    molecule = _make_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=(8, 8),
        name="toy_scf_bind_neural_xc",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(7), molecule)

    bound = functional.bind_to_molecule_for_scf(params, molecule)

    assert bound.projected_energy_density_values is None
    assert bound.grid_response_tensor_fn is None
    assert bound.projected_local_potential_values.shape == molecule.grid.weights.shape
    assert jnp.isfinite(bound.exact_exchange_fraction)


def test_bind_to_molecule_for_response_skips_strict_potential_assembly(monkeypatch):
    molecule = _make_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=(8, 8),
        name="toy_response_bind_neural_xc",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(77), molecule)

    def _fail_potential(*args, **kwargs):
        raise AssertionError("strict potential components should not be assembled")

    monkeypatch.setattr(NeuralXCFunctional, "_strict_total_potential_components", _fail_potential)

    bound = functional.bind_to_molecule_for_response(params, molecule)

    assert bound.projected_energy_density_values is None
    assert bound.grid_response_tensor_fn is not None
    tensor = bound.grid_response_tensor(molecule)
    assert tensor.shape[-1] == molecule.grid.weights.shape[0]
    assert jnp.isfinite(bound.exact_exchange_fraction)


def test_bind_to_molecule_for_response_exposes_spin_kernel_for_open_shell():
    molecule = _make_open_shell_toy_molecule()
    non_hf_module = make_custom_semilocal_module(
        channel_names=("alpha_density", "beta_density"),
        energy_density_channels_fn=lambda local_features: jnp.stack(
            [local_features.rho_a, local_features.rho_b],
            axis=-1,
        ),
        name="open_shell_response_module",
    )
    functional = make_neural_xc_functional(
        non_hf_module=non_hf_module,
        hidden_dims=(8, 8),
        name="toy_open_shell_response_bind_neural_xc",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(79), molecule)

    bound = functional.bind_to_molecule_for_response(params, molecule)
    f_aa, f_ab, f_bb = bound.spin_local_kernel(
        jnp.zeros_like(molecule.grid.weights),
        jnp.zeros_like(molecule.grid.weights),
    )

    assert f_aa.shape == molecule.grid.weights.shape
    assert f_ab.shape == molecule.grid.weights.shape
    assert f_bb.shape == molecule.grid.weights.shape
    assert jnp.all(jnp.isfinite(f_aa))
    assert jnp.all(jnp.isfinite(f_ab))
    assert jnp.all(jnp.isfinite(f_bb))


def test_bind_to_molecule_for_response_keeps_closed_shell_with_spin_gauge_difference():
    molecule = _make_toy_molecule()
    molecule = replace(
        molecule,
        mo_coeff=jnp.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, -1.0]],
            ]
        ),
    )
    non_hf_module = make_custom_semilocal_module(
        channel_names=("density",),
        energy_density_channels_fn=lambda local_features: jnp.expand_dims(
            local_features.rho,
            axis=-1,
        ),
        name="closed_shell_spin_gauge_response_module",
    )
    functional = make_neural_xc_functional(
        non_hf_module=non_hf_module,
        hidden_dims=(8, 8),
        name="toy_closed_shell_spin_gauge_response_bind",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(791), molecule)

    bound = functional.bind_to_molecule_for_response(params, molecule)

    assert bound.grid_response_tensor_fn is not None
    assert bound.spin_local_kernel_fn is None
    tensor = bound.grid_response_tensor(molecule)
    assert tensor.shape[-1] == molecule.grid.weights.shape[0]
    assert jnp.all(jnp.isfinite(tensor))


def test_bind_to_molecule_for_response_jittable_for_closed_shell_spin_axis():
    molecule = _make_toy_molecule()
    non_hf_module = make_custom_semilocal_module(
        channel_names=("density",),
        energy_density_channels_fn=lambda local_features: jnp.expand_dims(
            local_features.rho,
            axis=-1,
        ),
        name="closed_shell_spin_axis_jit_response_module",
    )
    functional = make_neural_xc_functional(
        non_hf_module=non_hf_module,
        hidden_dims=(8, 8),
        name="toy_closed_shell_spin_axis_jit_response_bind",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(792), molecule)

    @jax.jit
    def _response_trace(local_params):
        bound = functional.bind_to_molecule_for_response(local_params, molecule)
        return bound.grid_response_tensor(molecule)

    tensor = _response_trace(params)

    assert tensor.shape[-1] == molecule.grid.weights.shape[0]
    assert jnp.all(jnp.isfinite(tensor))


def test_tda_builder_prefers_response_specific_binding(monkeypatch):
    molecule = _make_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=(8, 8),
        response_hf_mode="approx",
        name="toy_response_binding_preference",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(101), molecule)

    def _fail_full_bind(*args, **kwargs):
        raise AssertionError("full bind_to_molecule should not be used for TD response")

    monkeypatch.setattr(NeuralXCFunctional, "bind_to_molecule", _fail_full_bind)

    delta_eps, a_matrix = build_restricted_tda_matrix(
        molecule,
        functional,
        xc_params=params,
    )

    assert delta_eps.shape == (1, 1)
    assert a_matrix.shape == (1, 1, 1, 1)
    assert jnp.all(jnp.isfinite(a_matrix))


def test_scf_potential_components_and_alpha_matches_bound_scf_binding():
    molecule = _make_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=(8, 8),
        name="toy_scf_direct_payload",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(233), molecule)

    bound = functional.bind_to_molecule_for_scf(params, molecule)
    v_rho, v_grad, v_tau, v_lapl, kind, alpha, extra_fock = functional.scf_potential_components_and_alpha(
        params,
        molecule,
    )

    bound_components = bound.grid_potential_components(molecule)
    assert len(bound_components) in (3, 4)
    bound_vrho, bound_vgrad, bound_vtau = bound_components[:3]
    assert kind == "MGGA"
    assert jnp.allclose(v_rho, bound_vrho, atol=1e-8)
    assert jnp.allclose(v_grad, bound_vgrad, atol=1e-8)
    assert jnp.allclose(v_tau, bound_vtau, atol=1e-8)
    assert jnp.allclose(bound_vtau, bound.projected_local_potential_tau_values, atol=1e-8)
    if len(bound_components) == 4:
        assert v_lapl is not None
        assert jnp.allclose(v_lapl, bound_components[3], atol=1e-8)
    else:
        assert v_lapl is None
    assert jnp.allclose(alpha, bound.exact_exchange_fraction, atol=1e-8)
    assert extra_fock.shape == molecule.h1e.shape
    assert jnp.allclose(extra_fock, jnp.zeros_like(molecule.h1e), atol=1e-8)


def test_neural_xc_contracts_hfx_feature_gradients_into_extra_fock():
    molecule = _make_toy_molecule()
    molecule.hfx_nu = jnp.zeros((2, 2, 2, 2), dtype=jnp.float32)
    molecule.hfx_nu = molecule.hfx_nu.at[0, 0, 0, 0].set(1.0)
    molecule.hfx_nu = molecule.hfx_nu.at[0, 1, 1, 1].set(0.8)
    molecule.hfx_nu = molecule.hfx_nu.at[1, 0, 0, 0].set(0.5)
    molecule.hfx_nu = molecule.hfx_nu.at[1, 1, 1, 1].set(0.4)
    functional = make_neural_xc_functional(
        hidden_dims=(8, 8),
        name="toy_scf_explicit_hfx_fock",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(244), molecule)

    components = functional.scf_potential_components_and_alpha(params, molecule)
    _, _, _, _, _, alpha, extra_fock = components
    density = jnp.asarray(molecule.rdm1).sum(axis=0)
    callback_extra = functional.scf_extra_fock_for_density(params, molecule, density)
    _, fock_alpha, fock_extra, _xc_energy = scf_differentiable._restricted_xc_fock_terms(
        params=params,
        functional=functional,
        molecule=molecule,
        weights=molecule.grid.weights,
        functional_dtype=jnp.float32,
        vxc_clip=20.0,
    )

    assert extra_fock.shape == molecule.h1e.shape
    assert jnp.all(jnp.isfinite(extra_fock))
    assert jnp.linalg.norm(extra_fock) > 0.0
    assert jnp.allclose(extra_fock, callback_extra, atol=1e-7)
    assert jnp.allclose(fock_extra, callback_extra, atol=1e-7)
    assert jnp.allclose(alpha, 0.0, atol=1e-8)
    assert jnp.allclose(fock_alpha, 0.0, atol=1e-8)
    assert functional.effective_exchange_fraction(params, molecule) > 0.0


def test_neural_xc_scf_density_energy_matches_rebuilt_molecule_energy():
    molecule = _make_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=(8, 8),
        name="toy_scf_density_energy",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(313), molecule)
    density = 1.1 * jnp.asarray(molecule.rdm1).sum(axis=0)

    density_energy = functional.scf_xc_energy_for_density(params, molecule, density)
    combined_energy, combined_alpha = functional.scf_xc_energy_and_alpha_for_density(
        params,
        molecule,
        density,
    )
    rebuilt = functional.scf_molecule_with_density(molecule, density)
    rebuilt_energy = functional.energy_from_molecule(params, rebuilt)
    rebuilt_alpha = functional.effective_exchange_fraction(params, rebuilt)

    assert jnp.allclose(density_energy, rebuilt_energy, atol=1e-8)
    assert jnp.allclose(combined_energy, rebuilt_energy, atol=1e-8)
    assert jnp.allclose(combined_alpha, rebuilt_alpha, atol=1e-8)


def test_neural_xc_energy_from_molecule_uses_direct_channel_basis(monkeypatch):
    molecule = _make_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=(8, 8),
        name="toy_energy_direct_channel_basis",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(315), molecule)

    def _fail_channel_contributions(*args, **kwargs):
        raise AssertionError("energy_from_molecule should use the direct channel-basis helper")

    monkeypatch.setattr(NeuralXCFunctional, "channel_contributions", _fail_channel_contributions)

    energy = functional.energy_from_molecule(params, molecule)

    assert jnp.isfinite(energy)


def test_neural_xc_scf_fock_terms_use_density_energy_callback(monkeypatch):
    molecule = _make_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=(8, 8),
        name="toy_scf_density_energy_fock",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(317), molecule)

    def _fail_potential(*args, **kwargs):
        raise AssertionError("neural XC SCF should use scf_xc_energy_for_density")

    def _fail_separate_alpha(*args, **kwargs):
        raise AssertionError("neural XC SCF should reuse alpha from the energy callback")

    monkeypatch.setattr(
        NeuralXCFunctional,
        "scf_potential_components_and_alpha",
        _fail_potential,
    )
    monkeypatch.setattr(
        NeuralXCFunctional,
        "scf_exact_exchange_fraction",
        _fail_separate_alpha,
    )

    vxc_matrix, alpha, extra_fock, _xc_energy = scf_differentiable._restricted_xc_fock_terms(
        params=params,
        functional=functional,
        molecule=molecule,
        weights=molecule.grid.weights,
        functional_dtype=jnp.float32,
        vxc_clip=20.0,
    )

    assert vxc_matrix.shape == molecule.h1e.shape
    assert extra_fock.shape == molecule.h1e.shape
    assert jnp.all(jnp.isfinite(vxc_matrix))
    assert jnp.isfinite(alpha)


def test_bind_to_molecule_reuses_precomputed_coefficients_for_energy(monkeypatch):
    molecule = _make_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=(8, 8),
        name="toy_full_bind_energy_reuse",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(234), molecule)

    def _fail_channel_contributions(*args, **kwargs):
        raise AssertionError("bind_to_molecule should not recompute channel contributions")

    monkeypatch.setattr(NeuralXCFunctional, "channel_contributions", _fail_channel_contributions)

    bound = functional.bind_to_molecule(params, molecule)

    assert bound.projected_energy_density_values is not None
    assert bound.projected_energy_density_values.shape == molecule.grid.weights.shape


def test_custom_non_hf_module_is_pluggable_into_neural_xc_functional():
    molecule = _make_toy_molecule()
    non_hf_module = make_custom_semilocal_module(
        channel_names=("toy_exchange", "toy_correlation"),
        energy_density_channels_fn=lambda features: jnp.stack(
            [features.rho, 0.5 * features.rho**2],
            axis=-1,
        ),
        name="toy_non_hf_module",
    )
    functional = make_neural_xc_functional(
        non_hf_module=non_hf_module,
        hidden_dims=(8, 8),
        name="toy_pluggable_neural_xc",
    )

    params = functional.init_from_molecule(jax.random.PRNGKey(11), molecule)
    features = restricted_grid_features(molecule)
    semilocal_channels = functional.semilocal_energy_density_channels(features)
    semilocal_total = functional.semilocal_energy_density(features)
    hf_total, hf_a, hf_b = functional.projected_hf_grid_contribution_components(
        molecule,
        features=features,
    )
    coefficients = functional.channel_coefficients(
        params,
        features,
        molecule=molecule,
        semilocal_energy_density=semilocal_total,
        hf_energy_density=hf_total,
        hf_spin_energy_density=(hf_a, hf_b),
    )

    assert functional.resolved_non_hf_module().name == "toy_non_hf_module"
    assert semilocal_channels.shape[-1] == 2
    assert coefficients.shape[-1] == 3
    assert jnp.all(jnp.isfinite(semilocal_channels))


def test_unrestricted_neural_xc_energy_path_is_spin_resolved():
    from td_graddft.features import unrestricted_grid_features

    molecule = _make_open_shell_toy_molecule()
    features = unrestricted_grid_features(molecule)
    assert jnp.any(features.rho_a > 0.0)
    assert jnp.allclose(features.rho_b, 0.0)

    non_hf_module = make_custom_semilocal_module(
        channel_names=("alpha_density", "beta_density"),
        energy_density_channels_fn=lambda local_features: jnp.stack(
            [local_features.rho_a, local_features.rho_b],
            axis=-1,
        ),
        name="spin_resolved_open_shell_module",
    )
    functional = make_neural_xc_functional(
        non_hf_module=non_hf_module,
        hidden_dims=(4,),
        network_architecture="simple_mlp",
        name="open_shell_neural_xc",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(31), molecule)
    energy = functional.energy_from_molecule(params, molecule)
    grads = jax.grad(lambda local_params: functional.energy_from_molecule(local_params, molecule))(
        params
    )

    assert jnp.isfinite(energy)
    assert all(jnp.all(jnp.isfinite(jnp.asarray(leaf))) for leaf in jax.tree_util.tree_leaves(grads))


def test_custom_non_hf_module_keeps_random_output_head_initialization():
    molecule = _make_toy_molecule()
    non_hf_module = make_custom_semilocal_module(
        channel_names=("toy_1", "toy_2", "toy_3", "toy_4"),
        energy_density_channels_fn=lambda features: jnp.stack(
            [
                features.rho,
                0.5 * features.rho,
                features.sigma,
                0.25 * features.sigma,
            ],
            axis=-1,
        ),
        name="toy_four_channel_non_hf_module",
    )
    functional = make_neural_xc_functional(
        non_hf_module=non_hf_module,
        hidden_dims=(8, 8),
        name="toy_custom_prior_guard",
    )

    params = functional.init_from_molecule(jax.random.PRNGKey(19), molecule)

    assert jnp.any(params["params"]["HeadDense"]["kernel"] != 0.0)


def test_b3lyp_prior_initialization_keeps_body_gradient_live(monkeypatch):
    def fake_eval(name, bundle, *, omega=None, allow_experimental_jax_xc=False):
        factors = {
            "lda_x": 0.5,
            "gga_x_b88": 1.0,
            "lda_c_vwn_rpa": 1.5,
            "gga_c_lyp": 2.0,
        }
        return jnp.full_like(bundle.rho, factors[name])

    monkeypatch.setattr(
        jax_xc_adapter,
        "eval_jax_xc_from_restricted_features",
        fake_eval,
    )

    molecule = _make_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc=b3lyp_component_basis(),
        input_feature_mode="canonical",
        hfx_channels=2,
        hidden_dims=(8, 8),
        name="b3lyp_prior_live_body_gradient",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(23), molecule)
    features = restricted_grid_features(molecule)

    coefficients = functional.channel_coefficients(params, features, molecule=molecule)
    expected = jnp.asarray([0.08, 0.72, 0.19, 0.81, 0.20], dtype=coefficients.dtype)

    def coefficient_objective(local_params):
        local_coefficients = functional.channel_coefficients(
            local_params,
            features,
            molecule=molecule,
        )
        return jnp.sum(local_coefficients)

    grads = jax.grad(coefficient_objective)(params)

    assert jnp.allclose(jnp.mean(coefficients, axis=0), expected, atol=1e-3)
    assert jnp.linalg.norm(params["params"]["HeadDense"]["kernel"]) > 0.0
    assert jnp.linalg.norm(grads["params"]["InitialDense"]["kernel"]) > 0.0


def test_libxc_semilocal_module_supports_common_exchange_and_correlation_components():
    molecule = _make_toy_molecule()
    module = make_libxc_semilocal_module(
        ("lda_x", "gga_x_b88", "lda_c_pw", "gga_c_pbe"),
        name="common_b3lyp_like_semilocal",
    )
    functional = make_neural_xc_functional(
        non_hf_module=module,
        hidden_dims=(8, 8),
        name="toy_common_semilocal_neural_xc",
    )

    params = functional.init_from_molecule(jax.random.PRNGKey(17), molecule)
    features = restricted_grid_features(molecule)
    channels = functional.semilocal_energy_density_channels(features)

    assert "lda_x" in available_semilocal_components()
    assert "gga_c_pbe" in available_semilocal_components()
    assert functional.resolved_non_hf_module().channel_names == (
        "lda_x",
        "gga_x_b88",
        "lda_c_pw",
        "gga_c_pbe",
    )
    assert channels.shape[-1] == 4
    assert jnp.all(jnp.isfinite(channels))
    assert params is not None


def test_canonical_feature_mode_uses_two_hfx_channels():
    _pyscf_or_skip()
    mf = _make_water_b3lyp_reference()
    reference = restricted_reference_from_pyscf(
        mf,
        compute_local_hfx_features=True,
        compute_local_hfx_aux=True,
        hfx_omega_values=(0.0, 0.4),
    )
    assert reference.hfx_local is not None
    assert reference.hfx_local.shape[0] == 2
    assert reference.hfx_local.shape[-1] == 2
    assert reference.hfx_nu is not None
    assert reference.hfx_nu.shape[0] == 2

    functional = make_neural_xc_functional(
        semilocal_xc=("lda_x", "gga_x_b88", "lda_c_pw", "gga_c_pbe"),
        input_feature_mode="canonical",
        hfx_channels=2,
        hidden_dims=(8, 8),
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(7), reference)
    features = restricted_grid_features(reference)
    semilocal = functional.semilocal_energy_density(features)
    hf_total, hf_a, hf_b = functional.projected_hf_energy_density_components(
        reference,
        features=features,
    )
    inputs = functional.coefficient_inputs(
        features,
        semilocal,
        hf_total,
        molecule=reference,
        hf_spin_energy_density=(hf_a, hf_b),
    )
    coefficients = functional.channel_coefficients(
        params,
        features,
        molecule=reference,
        semilocal_energy_density=semilocal,
        hf_energy_density=hf_total,
        hf_spin_energy_density=(hf_a, hf_b),
    )

    # Canonical feature channels: 7 density/gradient/tau + 2 omega(alpha) + 2 omega(beta).
    assert inputs.shape[-1] == 11
    assert coefficients.shape[-1] == 5
    expected_leading = jnp.stack(
        [
            features.rho_a,
            features.rho_b,
            features.sigma,
            features.sigma_aa,
            features.sigma_bb,
            features.tau_a,
            features.tau_b,
        ],
        axis=-1,
    )
    assert jnp.allclose(inputs[:, :7], expected_leading)


def test_canonical_feature_mode_requires_local_hfx_by_default():
    molecule = _make_toy_molecule()
    molecule.hfx_local = None
    functional = make_neural_xc_functional(
        semilocal_xc=("lda_x", "gga_x_b88", "lda_c_pw", "gga_c_pbe"),
        input_feature_mode="canonical",
        hfx_channels=2,
        hidden_dims=(8, 8),
    )

    with pytest.raises((ValueError, AttributeError), match="hfx_local"):
        functional.init_from_molecule(jax.random.PRNGKey(9), molecule)


def test_canonical_feature_mode_can_fallback_when_strictness_disabled():
    molecule = _make_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc=("lda_x", "gga_x_b88", "lda_c_pw", "gga_c_pbe"),
        input_feature_mode="canonical",
        strict_feature_alignment=False,
        hfx_channels=2,
        hidden_dims=(8, 8),
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(10), molecule)
    features = restricted_grid_features(molecule)
    semilocal = functional.semilocal_energy_density(features)
    hf_total, hf_a, hf_b = functional.projected_hf_grid_contribution_components(
        molecule,
        features=features,
    )
    inputs = functional.coefficient_inputs(
        features,
        semilocal,
        hf_total,
        molecule=molecule,
        hf_spin_energy_density=(hf_a, hf_b),
    )
    coefficients = functional.channel_coefficients(
        params,
        features,
        molecule=molecule,
        semilocal_energy_density=semilocal,
        hf_energy_density=hf_total,
        hf_spin_energy_density=(hf_a, hf_b),
    )

    assert inputs.shape[-1] == 11
    assert coefficients.shape[-1] == 5
    assert jnp.all(jnp.isfinite(inputs))


def test_reference_can_cache_local_pt2_feature_and_functional_reuses_it():
    _pyscf_or_skip()
    mf = _make_water_b3lyp_reference()
    reference = restricted_reference_from_pyscf(
        mf,
        compute_local_pt2_features=True,
    )
    assert reference.pt2_local is not None
    assert reference.pt2_local.ndim == 1
    assert reference.pt2_local.shape[0] == reference.ao.shape[0]

    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=(8, 8),
        include_pt2_channel=True,
        pt2_channel_mode="local_exact",
    )
    pt2 = functional.projected_pt2_grid_contribution(reference)
    assert jnp.allclose(pt2, reference.pt2_local, atol=1e-9)


def test_local_pt2_feature_supports_packed_eri_pair_matrix():
    from td_graddft.neural_xc.inputs import _local_pt2_feature_from_restricted_orbitals

    ao = jnp.asarray(
        [
            [0.7, -0.2, 0.4],
            [0.1, 0.8, -0.3],
            [0.5, 0.3, 0.6],
            [-0.4, 0.2, 0.9],
        ],
        dtype=jnp.float64,
    )
    mo_coeff = jnp.eye(3, dtype=jnp.float64)
    mo_occ = jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64)
    mo_energy = jnp.asarray([-0.6, 0.2, 0.55], dtype=jnp.float64)
    metric = jnp.asarray(
        [
            [1.2, 0.3, -0.2],
            [0.3, 0.9, 0.1],
            [-0.2, 0.1, 0.7],
        ],
        dtype=jnp.float64,
    )
    rep_tensor = jnp.einsum("pq,rs->pqrs", metric, metric)

    rows, cols = np.tril_indices(3)
    eri_pair_matrix = rep_tensor[
        rows[:, None],
        cols[:, None],
        rows[None, :],
        cols[None, :],
    ]
    expected = _local_pt2_feature_from_restricted_orbitals(
        ao,
        mo_coeff,
        mo_occ,
        mo_energy,
        rep_tensor=rep_tensor,
        nocc=1,
    )
    actual = _local_pt2_feature_from_restricted_orbitals(
        ao,
        mo_coeff,
        mo_occ,
        mo_energy,
        rep_tensor=jnp.zeros((0,), dtype=jnp.float64),
        eri_pair_matrix=eri_pair_matrix,
        nocc=1,
    )

    assert jnp.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_projected_hf_uses_local_hfx_channel_without_global_rescaling():
    molecule = _make_toy_molecule()
    hfx_local = jnp.array(
        [
            [[-0.30, -0.12], [-0.10, -0.04]],
            [[-0.20, -0.08], [-0.05, -0.02]],
        ]
    )
    molecule = _ToyMolecule(
        ao=molecule.ao,
        ao_deriv1=molecule.ao_deriv1,
        ao_laplacian=molecule.ao_laplacian,
        grid=molecule.grid,
        rep_tensor=molecule.rep_tensor,
        mo_coeff=molecule.mo_coeff,
        mo_occ=molecule.mo_occ,
        mo_energy=molecule.mo_energy,
        rdm1=molecule.rdm1,
        h1e=molecule.h1e,
        nuclear_repulsion=molecule.nuclear_repulsion,
        hfx_local=hfx_local,
    )
    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=(8, 8),
    )
    features = restricted_grid_features(molecule)

    hf_total, hf_a, hf_b = functional.projected_hf_grid_contribution_components(
        molecule,
        features=features,
    )

    assert jnp.allclose(hf_a, hfx_local[0, :, 0], atol=1e-9)
    assert jnp.allclose(hf_b, hfx_local[1, :, 0], atol=1e-9)
    assert jnp.allclose(hf_total, hfx_local[0, :, 0] + hfx_local[1, :, 0], atol=1e-9)


def test_enhanced_feature_mode_uses_semilocal_descriptor_not_local_contribution():
    molecule = _make_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc=("lda_x", "gga_x_b88", "lda_c_pw", "gga_c_pbe"),
        input_feature_mode="enhanced",
        hf_input_mode="spin_resolved",
        hidden_dims=(8, 8),
    )
    features = restricted_grid_features(molecule)
    semilocal_local = functional.semilocal_energy_density(features)
    hf_total, hf_a, hf_b = functional.projected_hf_grid_contribution_components(
        molecule,
        features=features,
    )

    inputs = functional.coefficient_inputs(
        features,
        semilocal_local,
        hf_total,
        molecule=molecule,
        hf_spin_energy_density=(hf_a, hf_b),
    )

    expected_semilocal_descriptor = semilocal_local / jnp.maximum(
        features.rho,
        functional.density_floor,
    )
    assert jnp.allclose(inputs[:, 12], expected_semilocal_descriptor)
    assert jnp.allclose(inputs[:, 13], hf_total)
    assert jnp.allclose(inputs[:, 14], hf_a)
    assert jnp.allclose(inputs[:, 15], hf_b)
    coefficients = functional.channel_coefficients(
        functional.init_from_molecule(jax.random.PRNGKey(23), molecule),
        features,
        molecule=molecule,
        semilocal_energy_density=semilocal_local,
        hf_energy_density=hf_total,
        hf_spin_energy_density=(hf_a, hf_b),
    )
    assert jnp.all(jnp.isfinite(coefficients))


def test_neural_xc_hidden_dims_controls_depth():
    molecule = _make_toy_molecule()
    hidden_dims = (16, 12, 8, 4)
    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=hidden_dims,
        network_architecture="simple_mlp",
        name="depth_check_neural_xc",
    )
    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(0),
        molecule,
        optax.adam(1e-3),
    )
    dense_layers = [
        key for key in state.params["params"].keys() if key.startswith("Dense_")
    ]
    assert len(dense_layers) == len(hidden_dims) + 1
    assert functional.model.hidden_dims == hidden_dims


def test_graddft_residual_architecture_uses_residual_model():
    molecule = _make_toy_molecule()
    hidden_dims = (16, 16, 16)
    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=hidden_dims,
        network_architecture="graddft_residual",
        name="graddft_residual_depth_check",
    )
    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(7),
        molecule,
        optax.adam(1e-3),
    )

    param_keys = tuple(state.params["params"].keys())

    assert isinstance(functional.model, ResidualMixingMLP)
    assert functional.model.hidden_dims == hidden_dims
    assert any(key.startswith("InitialDense") for key in param_keys)
    assert any(key.startswith("ResidualLayerNorm_") for key in param_keys)


def test_neural_xc_hidden_dims_must_be_positive():
    with pytest.raises(ValueError):
        make_neural_xc_functional(hidden_dims=())
    with pytest.raises(ValueError):
        make_neural_xc_functional(hidden_dims=(32, 0, 16))


def test_projected_hf_energy_density_is_finite_and_consistent_with_grid_projection():
    molecule = _make_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=(8, 8),
        name="hf_projection_check",
    )
    features = restricted_grid_features(molecule)
    eps_hf = functional.projected_hf_energy_density(molecule, features=features)
    rho = jnp.maximum(features.rho, functional.density_floor)
    projected_energy = jnp.tensordot(molecule.grid.weights, rho * eps_hf, axes=(0, 0))
    hf_grid = functional.projected_hf_grid_contribution_components(molecule, features=features)[0]

    assert jnp.all(jnp.isfinite(eps_hf))
    assert jnp.isfinite(projected_energy)
    assert jnp.allclose(projected_energy, jnp.tensordot(molecule.grid.weights, hf_grid, axes=(0, 0)), atol=1e-6)


def test_projected_hf_components_are_spin_consistent():
    molecule = _make_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=(8, 8),
        name="hf_projection_components_check",
    )
    features = restricted_grid_features(molecule)
    eps_hf, eps_hf_a, eps_hf_b = functional.projected_hf_energy_density_components(
        molecule,
        features=features,
    )
    rho = jnp.maximum(features.rho, functional.density_floor)
    rho_a = jnp.maximum(features.rho_a, functional.density_floor)
    rho_b = jnp.maximum(features.rho_b, functional.density_floor)
    projected_total = jnp.tensordot(molecule.grid.weights, rho * eps_hf, axes=(0, 0))
    projected_split = jnp.tensordot(
        molecule.grid.weights,
        rho_a * eps_hf_a + rho_b * eps_hf_b,
        axes=(0, 0),
    )
    assert jnp.isfinite(projected_total)
    assert jnp.isfinite(projected_split)
    assert jnp.allclose(projected_total, projected_split, atol=1e-6)
    assert eps_hf_a.shape == eps_hf_b.shape == eps_hf.shape
    assert jnp.all(jnp.isfinite(eps_hf_a))
    assert jnp.all(jnp.isfinite(eps_hf_b))


def test_projected_pt2_grid_contribution_recovers_canonical_mp2_energy():
    molecule = _make_pt2_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=(8, 8),
        include_pt2_channel=True,
        name="pt2_projection_check",
    )
    projected_pt2 = functional.projected_pt2_grid_contribution(molecule)
    projected_energy = jnp.tensordot(
        molecule.grid.weights,
        projected_pt2,
        axes=(0, 0),
    )
    expected_mp2 = jnp.array(-0.25)

    assert jnp.all(jnp.isfinite(projected_pt2))
    assert not jnp.allclose(projected_pt2, 0.0, atol=1e-8)
    assert jnp.allclose(projected_energy, expected_mp2, atol=1e-6)


def test_scaled_pt2_projection_does_not_call_local_exact_wrapper(monkeypatch):
    molecule = _make_pt2_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=(8, 8),
        include_pt2_channel=True,
        pt2_channel_mode="scaled_projected",
        name="pt2_projection_no_wrapper_recompute",
    )

    def _fail_local_exact(*args, **kwargs):
        raise AssertionError("scaled PT2 should share MP2 intermediates directly")

    monkeypatch.setattr(NeuralXCFunctional, "_local_exact_pt2_grid_contribution", _fail_local_exact)

    projected_pt2 = functional.projected_pt2_grid_contribution(molecule)
    projected_energy = jnp.tensordot(molecule.grid.weights, projected_pt2, axes=(0, 0))

    assert jnp.all(jnp.isfinite(projected_pt2))
    assert jnp.allclose(projected_energy, jnp.array(-0.25), atol=1e-6)


def test_include_pt2_channel_adds_feature_and_basis_channels():
    molecule = _make_pt2_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=(8, 8),
        input_feature_mode="enhanced",
        hf_input_mode="spin_resolved",
        include_pt2_channel=True,
        name="pt2_channel_shape_check",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(31), molecule)
    features = restricted_grid_features(molecule)
    semilocal = functional.semilocal_energy_density(features)
    hf_total, hf_a, hf_b = functional.projected_hf_grid_contribution_components(
        molecule,
        features=features,
    )
    pt2 = functional.projected_pt2_grid_contribution(molecule, features=features)
    inputs = functional.coefficient_inputs(
        features,
        semilocal,
        hf_total,
        pt2_energy_density=pt2,
        molecule=molecule,
        hf_spin_energy_density=(hf_a, hf_b),
    )
    basis = functional.compute_densities(molecule, features=features)
    coefficients = functional.channel_coefficients(
        params,
        features,
        molecule=molecule,
        semilocal_energy_density=semilocal,
        hf_energy_density=hf_total,
        pt2_energy_density=pt2,
        hf_spin_energy_density=(hf_a, hf_b),
    )
    channels = functional.channel_contributions(
        params,
        molecule,
        features=features,
        semilocal_energy_density=semilocal,
        hf_energy_density=hf_total,
        pt2_energy_density=pt2,
    )

    assert inputs.shape[-1] == 17
    assert basis.shape[-1] == 4
    assert coefficients.shape[-1] == 4
    assert channels.shape[-1] == 4
    assert jnp.all(jnp.isfinite(inputs))
    assert jnp.all(jnp.isfinite(basis))
    assert jnp.all(jnp.isfinite(coefficients))
    assert jnp.all(jnp.isfinite(channels))


def test_pt2_channel_mode_local_exact_returns_unscaled_pair_gauge():
    molecule = _make_pt2_toy_molecule()
    exact_functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=(8, 8),
        include_pt2_channel=True,
        pt2_channel_mode="local_exact",
    )
    scaled_functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=(8, 8),
        include_pt2_channel=True,
        pt2_channel_mode="scaled_projected",
    )

    exact_pt2 = exact_functional.projected_pt2_grid_contribution(molecule)
    scaled_pt2 = scaled_functional.projected_pt2_grid_contribution(molecule)
    projected_energy = jnp.tensordot(molecule.grid.weights, scaled_pt2, axes=(0, 0))
    raw_energy = jnp.tensordot(molecule.grid.weights, exact_pt2, axes=(0, 0))

    assert jnp.all(jnp.isfinite(exact_pt2))
    assert jnp.all(jnp.isfinite(scaled_pt2))
    assert not jnp.allclose(exact_pt2, scaled_pt2, atol=1e-8)
    assert jnp.allclose(projected_energy, jnp.array(-0.25), atol=1e-6)
    assert not jnp.allclose(raw_energy, jnp.array(-0.25), atol=1e-6)


def test_pt2_channel_mode_local_exact_uses_cached_pt2_local_when_available():
    molecule = _make_pt2_toy_molecule()
    cached_pt2 = jnp.array([-0.7, 0.2])
    molecule = _ToyMolecule(
        ao=molecule.ao,
        ao_deriv1=molecule.ao_deriv1,
        ao_laplacian=molecule.ao_laplacian,
        grid=molecule.grid,
        rep_tensor=molecule.rep_tensor,
        mo_coeff=molecule.mo_coeff,
        mo_occ=molecule.mo_occ,
        mo_energy=molecule.mo_energy,
        rdm1=molecule.rdm1,
        h1e=molecule.h1e,
        nuclear_repulsion=molecule.nuclear_repulsion,
        pt2_local=cached_pt2,
    )
    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=(8, 8),
        include_pt2_channel=True,
        pt2_channel_mode="local_exact",
    )

    pt2 = functional.projected_pt2_grid_contribution(molecule)
    assert jnp.allclose(pt2, cached_pt2, atol=1e-9)


def test_pt2_channel_mode_local_exact_recomputes_open_shell_pt2_when_cache_missing():
    molecule = _make_open_shell_toy_molecule()
    molecule.rep_tensor = jnp.zeros((2, 2, 2, 2), dtype=jnp.float64)
    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=(8, 8),
        include_pt2_channel=True,
        pt2_channel_mode="local_exact",
    )

    pt2 = functional.projected_pt2_grid_contribution(molecule)

    assert pt2.shape == molecule.grid.weights.shape
    assert jnp.all(jnp.isfinite(pt2))
    assert jnp.allclose(pt2, 0.0, atol=1e-10)


def test_pt2_channel_mode_local_exact_matches_unrestricted_open_shell_feature():
    ao = jnp.eye(3, dtype=jnp.float64)
    molecule = _ToyMolecule(
        ao=ao,
        ao_deriv1=jnp.stack([ao, jnp.zeros_like(ao), jnp.zeros_like(ao), jnp.zeros_like(ao)]),
        ao_laplacian=jnp.zeros_like(ao),
        grid=_Grid(weights=jnp.asarray([0.3, 0.4, 0.5], dtype=jnp.float64)),
        rep_tensor=jnp.arange(3**4, dtype=jnp.float64).reshape(3, 3, 3, 3) / 40.0,
        mo_coeff=jnp.stack([jnp.eye(3, dtype=jnp.float64), jnp.eye(3, dtype=jnp.float64)], axis=0),
        mo_occ=jnp.asarray([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float64),
        mo_energy=jnp.asarray([[-0.9, -0.3, 0.5], [-0.7, 0.1, 0.6]], dtype=jnp.float64),
        rdm1=jnp.asarray(
            [
                jnp.diag(jnp.asarray([1.0, 1.0, 0.0], dtype=jnp.float64)),
                jnp.diag(jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64)),
            ]
        ),
        h1e=jnp.diag(jnp.asarray([-0.9, -0.3, 0.5], dtype=jnp.float64)),
        nuclear_repulsion=0.0,
        pt2_local=None,
    )
    molecule.nocc_alpha = 2
    molecule.nocc_beta = 1
    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=(8, 8),
        include_pt2_channel=True,
        pt2_channel_mode="local_exact",
    )

    pt2 = functional.projected_pt2_grid_contribution(molecule)
    expected = _local_pt2_feature_from_unrestricted_orbitals(
        molecule.ao,
        molecule.mo_coeff,
        molecule.mo_occ,
        molecule.mo_energy,
        rep_tensor=molecule.rep_tensor,
    )
    restricted_fallback = _local_pt2_feature_from_restricted_orbitals(
        molecule.ao,
        molecule.mo_coeff,
        molecule.mo_occ,
        molecule.mo_energy,
        rep_tensor=molecule.rep_tensor,
    )

    assert jnp.allclose(pt2, expected, atol=1e-10)
    assert not jnp.allclose(expected, restricted_fallback, atol=1e-8)


def test_graddft_coeff_basis_hf_pt2_heads_assembles_semilocal_channels_with_explicit_heads():
    molecule = _make_pt2_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc=("lda_x", "gga_c_pbe"),
        hidden_dims=(8, 8),
        include_pt2_channel=True,
        name="hybrid_head_assembly_check",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(1331), molecule)
    features = restricted_grid_features(molecule)
    semilocal_channels = functional.semilocal_energy_density_channels(features)
    semilocal_local_channels = functional._semilocal_local_contribution_channels(
        features,
        semilocal_channels,
    )
    hf_grid, hf_a, hf_b = functional.projected_hf_grid_contribution_components(
        molecule, features=features
    )
    pt2_grid = functional.projected_pt2_grid_contribution(molecule, features=features)
    basis = functional.compute_densities(molecule, features=features)
    coefficients = functional.channel_coefficients(
        params,
        features,
        molecule=molecule,
        semilocal_energy_density=jnp.sum(semilocal_channels, axis=-1),
        hf_energy_density=hf_grid,
        pt2_energy_density=pt2_grid,
        hf_spin_energy_density=(hf_a, hf_b),
    )
    channels = functional.channel_contributions(
        params,
        molecule,
        features=features,
        semilocal_energy_density=jnp.sum(semilocal_channels, axis=-1),
        hf_energy_density=hf_grid,
        hf_spin_energy_density=(hf_a, hf_b),
        pt2_energy_density=pt2_grid,
    )

    expected_basis = jnp.concatenate(
        [semilocal_local_channels, pt2_grid[..., None], hf_grid[..., None]],
        axis=-1,
    )
    expected_channels = jnp.concatenate(
        [
            coefficients[..., :2] * semilocal_local_channels,
            coefficients[..., 2:3] * pt2_grid[..., None],
            coefficients[..., 3:4] * hf_grid[..., None],
        ],
        axis=-1,
    )

    assert coefficients.shape[-1] == 4
    assert basis.shape[-1] == 4
    assert jnp.allclose(basis, expected_basis, atol=1e-9)
    assert jnp.allclose(channels, expected_channels, atol=1e-9)


def test_pt2_projection_mode_controls_scaled_vs_local_exact_channel():
    molecule = _make_pt2_toy_molecule()
    scaled_functional = make_neural_xc_functional(
        semilocal_xc=("lda_x", "gga_c_pbe"),
        hidden_dims=(8, 8),
        include_pt2_channel=True,
        pt2_channel_mode="scaled_projected",
        name="scaled_pt2_check",
    )
    local_functional = make_neural_xc_functional(
        semilocal_xc=("lda_x", "gga_c_pbe"),
        hidden_dims=(8, 8),
        include_pt2_channel=True,
        pt2_channel_mode="local_exact",
        name="local_exact_pt2_check",
    )
    scaled_pt2 = scaled_functional.projected_pt2_grid_contribution(molecule)
    local_pt2 = local_functional.projected_pt2_grid_contribution(molecule)
    raw_pt2 = local_functional._local_exact_pt2_grid_contribution(molecule)

    assert not jnp.allclose(scaled_pt2, raw_pt2, atol=1e-9)
    assert jnp.allclose(local_pt2, raw_pt2, atol=1e-9)


def test_custom_semilocal_energy_density_callback_is_used():
    molecule = _make_toy_molecule()

    def custom_eps(features):
        return 0.25 + 0.1 * jnp.log1p(jnp.maximum(features.rho, 1e-12))

    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        semilocal_energy_density_fn=custom_eps,
        hidden_dims=(8, 8),
        name="custom_semilocal_check",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(3), molecule)
    features = restricted_grid_features(molecule)
    semilocal = functional.semilocal_energy_density(features)
    expected = custom_eps(features)
    weights = functional.mixing_weights(
        params,
        features,
        semilocal_energy_density=semilocal,
        hf_energy_density=functional.projected_hf_grid_contribution_components(
            molecule, features=features
        )[0],
    )

    assert jnp.allclose(semilocal, expected, atol=1e-9)
    coefficients = functional.channel_coefficients(
        params,
        features,
        semilocal_energy_density=semilocal,
        hf_energy_density=functional.projected_hf_grid_contribution_components(
            molecule, features=features
        )[0],
    )
    hf_grid = functional.projected_hf_grid_contribution_components(
        molecule, features=features
    )[0]
    channels = functional.channel_contributions(
        params,
        molecule,
        features=features,
        semilocal_energy_density=semilocal,
        hf_energy_density=hf_grid,
    )
    eps = functional.energy_density(
        params,
        molecule,
        features=features,
        semilocal_energy_density=semilocal,
        hf_energy_density=hf_grid,
    )
    hf_grid = functional.projected_hf_grid_contribution_components(
        molecule, features=features
    )[0]
    basis = jnp.stack(
        [
            semilocal,
            hf_grid,
        ],
        axis=-1,
    )
    assert jnp.all(jnp.isfinite(weights))
    assert jnp.allclose(weights, coefficients, atol=1e-9)
    assert jnp.allclose(channels, coefficients * basis, atol=1e-9)
    assert channels.shape[-1] == 2
    assert jnp.allclose(jnp.sum(channels, axis=-1), eps, atol=1e-9)


def test_semilocal_channels_are_freely_combinable_in_energy_density():
    molecule = _make_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc=("lda_x", "gga_c_pbe"),
        hidden_dims=(8, 8),
        name="multi_semilocal_channels_check",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(13), molecule)
    features = restricted_grid_features(molecule)
    semilocal_channels = functional.semilocal_energy_density_channels(features)
    semilocal_total = jnp.sum(semilocal_channels, axis=-1)
    hf_projected = functional.projected_hf_grid_contribution_components(
        molecule, features=features
    )[0]
    coefficients = functional.channel_coefficients(
        params,
        features,
        semilocal_energy_density=semilocal_total,
        hf_energy_density=hf_projected,
    )
    channels = functional.channel_contributions(
        params,
        molecule,
        features=features,
        semilocal_energy_density=semilocal_total,
        hf_energy_density=hf_projected,
    )
    basis = jnp.concatenate([semilocal_channels, hf_projected[..., None]], axis=-1)

    assert semilocal_channels.shape[-1] == 2
    assert coefficients.shape[-1] == 3
    assert channels.shape[-1] == 3
    assert jnp.allclose(channels, coefficients * basis, atol=1e-9)


def test_normalized_mixing_mode_matches_weighted_basis():
    molecule = _make_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=(8, 8),
        name="normalized_mixing_mode_check",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(7), molecule)
    features = restricted_grid_features(molecule)
    semilocal = functional.semilocal_energy_density(features)
    hf_projected = functional.projected_hf_grid_contribution_components(
        molecule, features=features
    )[0]
    weights = functional.mixing_weights(
        params,
        features,
        semilocal_energy_density=semilocal,
        hf_energy_density=hf_projected,
    )
    channels = functional.channel_contributions(
        params,
        molecule,
        features=features,
        semilocal_energy_density=semilocal,
        hf_energy_density=hf_projected,
    )
    semilocal_channels = functional.semilocal_energy_density_channels(features)
    semilocal_local_channels = functional._semilocal_local_contribution_channels(
        features,
        semilocal_channels,
    )
    basis = jnp.concatenate([semilocal_local_channels, hf_projected[..., None]], axis=-1)
    assert jnp.allclose(channels, weights * basis, atol=1e-9)


def test_projected_local_kernel_matches_potential_density_derivative():
    molecule = _make_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=(8, 8),
        response_hf_mode="approx",
        name="kernel_consistency_check",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(11), molecule)
    features = restricted_grid_features(molecule)
    hf_projected = functional.projected_hf_grid_contribution_components(
        molecule, features=features
    )[0]
    potential, kernel = functional._projected_total_potential_kernel(
        params,
        features,
        hf_projected,
    )
    rho0 = jnp.maximum(features.rho, functional.density_floor)
    idx = int(jnp.argmax(rho0))
    rho_point = rho0[idx]
    sigma_point = jnp.maximum(features.sigma[idx], 0.0)
    grad_point = jnp.array(
        [jnp.sqrt(sigma_point), 0.0, 0.0],
        dtype=rho_point.dtype,
    )
    tau_point = jnp.maximum(features.tau_a[idx] + features.tau_b[idx], 0.0)
    hf_point = hf_projected[idx]

    def local_energy(rho_value):
        variables = jnp.array(
            [rho_value, grad_point[0], grad_point[1], grad_point[2], tau_point],
            dtype=rho_point.dtype,
        )
        return functional._total_point_local_energy_from_variables(
            params,
            variables,
            hf_point,
            hf_point,
            hf_point,
            response_hf_mode="approx",
        )

    def vxc(rho_value):
        return jax.grad(local_energy)(rho_value)

    h = 1e-4
    fxc_fd = (vxc(rho_point + h) - vxc(rho_point - h)) / (2.0 * h)
    assert jnp.allclose(kernel[idx], fxc_fd, atol=5e-3)
    assert jnp.isfinite(potential[idx])
    assert jnp.isfinite(kernel[idx])


class _HFOnlyChannelModel(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        coeff = jnp.zeros(inputs.shape[:-1] + (2,), dtype=inputs.dtype)
        return coeff.at[..., 1].set(1.0)


class _HFResponsiveChannelModel(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        coeff = jnp.zeros(inputs.shape[:-1] + (2,), dtype=inputs.dtype)
        hf_coeff = 0.25 + jnp.square(inputs[..., 0])
        return coeff.at[..., 1].set(hf_coeff)


class _DensityResponsivePT2ChannelModel(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        coeff = jnp.zeros(inputs.shape[:-1] + (3,), dtype=inputs.dtype)
        return coeff.at[..., 1].set(jnp.square(inputs[..., 0]))


class _DensityResponsivePT2ChannelModel(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        coeff = jnp.zeros(inputs.shape[:-1] + (3,), dtype=inputs.dtype)
        return coeff.at[..., 1].set(jnp.square(inputs[..., 0]))


class _ConstantChannelModel(nn.Module):
    coeffs: tuple[float, ...]

    @nn.compact
    def __call__(self, inputs):
        coeffs = jnp.asarray(self.coeffs, dtype=inputs.dtype)
        return jnp.broadcast_to(coeffs, inputs.shape[:-1] + (coeffs.shape[0],))


def test_unrestricted_neural_xc_scf_components_keep_spin_potentials():
    molecule = _make_open_shell_toy_molecule()
    molecule = replace(
        molecule,
        rdm1=molecule.rdm1.at[0, 1, 1].set(0.5)
        .at[1, 0, 0]
        .set(0.25)
        .at[1, 1, 1]
        .set(0.25),
        mo_occ=molecule.mo_occ.at[0, 1].set(0.5).at[1, 0].set(0.25).at[1, 1].set(0.25),
    )
    non_hf_module = make_custom_semilocal_module(
        channel_names=("alpha_density", "beta_density"),
        energy_density_channels_fn=lambda features: jnp.stack(
            [features.rho_a, 3.0 * features.rho_b],
            axis=-1,
        ),
        name="spin_potential_module",
    )
    functional = NeuralXCFunctional(
        model=_ConstantChannelModel((1.0, 1.0, 0.0)),
        non_hf_module=non_hf_module,
        response_hf_mode="approx",
        name="spin_potential_functional",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(411), molecule)
    bound = functional.bind_to_molecule_for_scf(params, molecule)

    v_a, v_b, grad_a, grad_b, *_ = bound.unrestricted_scf_components(molecule)

    assert jnp.allclose(v_a, 1.0, atol=1e-10)
    assert jnp.allclose(v_b, 3.0, atol=1e-10)
    assert jnp.allclose(grad_a, 0.0, atol=1e-10)
    assert jnp.allclose(grad_b, 0.0, atol=1e-10)


def test_response_hf_mode_controls_local_hf_kernel_contribution():
    molecule = _make_toy_molecule()
    semilocal_zero = lambda features: jnp.zeros_like(features.rho)

    hf_strict = NeuralXCFunctional(
        model=_HFOnlyChannelModel(),
        semilocal_energy_density_fn=semilocal_zero,
        response_hf_mode="strict",
        name="hf_strict",
    )
    hf_approx = NeuralXCFunctional(
        model=_HFOnlyChannelModel(),
        semilocal_energy_density_fn=semilocal_zero,
        response_hf_mode="approx",
        name="hf_approx",
    )

    params_strict = hf_strict.init_from_molecule(jax.random.PRNGKey(21), molecule)
    params_approx = hf_approx.init_from_molecule(jax.random.PRNGKey(22), molecule)
    bound_strict = hf_strict.bind_to_molecule(params_strict, molecule)
    bound_approx = hf_approx.bind_to_molecule(params_approx, molecule)
    rho = jnp.sum(molecule.density(), axis=-1)

    kernel_strict = bound_strict.local_kernel(rho)
    kernel_approx = bound_approx.local_kernel(rho)
    hf_fraction_strict = bound_strict.local_hf_fraction(rho)
    hf_fraction_approx = bound_approx.local_hf_fraction(rho)

    assert jnp.allclose(kernel_strict, 0.0, atol=1e-8)
    assert jnp.allclose(kernel_approx, kernel_strict, atol=1e-8)
    assert jnp.allclose(hf_fraction_strict, 0.0, atol=1e-8)
    assert jnp.allclose(hf_fraction_approx, 1.0, atol=1e-8)


def test_response_hf_approx_uses_scalar_exchange_kernel():
    molecule = _make_toy_molecule()
    molecule.rep_tensor = molecule.rep_tensor.at[0, 0, 1, 1].set(0.5)
    semilocal_zero = lambda features: jnp.zeros_like(features.rho)
    hf_approx = NeuralXCFunctional(
        model=_HFResponsiveChannelModel(),
        semilocal_energy_density_fn=semilocal_zero,
        response_hf_mode="approx",
        name="hf_approx_response_binding",
    )
    params_approx = hf_approx.init_from_molecule(jax.random.PRNGKey(221), molecule)

    bound_approx = hf_approx.bind_to_molecule_for_response(params_approx, molecule)
    _, a_approx = build_restricted_tda_matrix(
        molecule,
        hf_approx,
        xc_params=params_approx,
    )

    assert bound_approx.response_feature_kind == "MGGA"
    assert bound_approx.local_hf_fraction_values is None
    assert bound_approx.exact_exchange_fraction > 0.0
    assert bound_approx.nonlocal_response_matrices_fn is None
    assert jnp.all(jnp.isfinite(a_approx))


def test_response_hf_strict_builds_hf_channel_kernel_without_scalar_exchange():
    molecule = _make_pt2_toy_molecule()
    molecule.rep_tensor = jnp.zeros_like(molecule.rep_tensor)
    nu = jnp.asarray(
        [
            [
                [[0.7, 0.2], [0.2, 0.5]],
                [[0.4, -0.1], [-0.1, 0.6]],
            ],
        ],
        dtype=jnp.float64,
    )
    molecule.hfx_nu = nu
    molecule.hfx_local = None
    alpha = 0.25
    semilocal_zero = lambda features: jnp.zeros_like(features.rho)
    functional = NeuralXCFunctional(
        model=_ConstantChannelModel((0.0, alpha)),
        semilocal_energy_density_fn=semilocal_zero,
        input_feature_mode="enhanced",
        hf_input_mode="total_only",
        response_hf_mode="strict",
        name="hf_strict_channel_kernel",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(223), molecule)

    bound = functional.bind_to_molecule_for_response(params, molecule)
    hf_a, hf_b = bound.nonlocal_response_matrices(molecule)
    matrices = build_restricted_response_matrices(
        molecule,
        functional,
        xc_params=params,
    )

    ao = jnp.asarray(molecule.ao)
    orbo = jnp.asarray(molecule.mo_coeff[0])[:, :1]
    orbv = jnp.asarray(molecule.mo_coeff[0])[:, 1:]
    rho_o = jnp.einsum("gp,pi->gi", ao, orbo)
    rho_v = jnp.einsum("gp,pa->ga", ao, orbv)
    nu0 = nu[0]
    nu_vv = jnp.einsum("pa,gpq,qb->gab", orbv, nu0, orbv)
    nu_vo = jnp.einsum("pa,gpq,qj->gaj", orbv, nu0, orbo)
    weights = jnp.asarray(molecule.grid.weights)
    expected_a = -alpha * jnp.einsum(
        "g,gi,gj,gab->iajb",
        weights,
        rho_o,
        rho_o,
        nu_vv,
    )
    expected_b = -alpha * jnp.einsum(
        "g,gi,gb,gaj->iajb",
        weights,
        rho_o,
        rho_v,
        nu_vo,
    )
    diagonal = jnp.asarray(molecule.mo_energy[0, 1:] - molecule.mo_energy[0, :1])[0]

    assert jnp.allclose(bound.exact_exchange_fraction, 0.0, atol=1e-12)
    assert jnp.allclose(hf_a.reshape(expected_a.shape), expected_a, atol=1e-10)
    assert jnp.allclose(hf_b.reshape(expected_b.shape), expected_b, atol=1e-10)
    assert jnp.allclose(matrices.a_matrix, expected_a + diagonal, atol=1e-10)
    assert jnp.allclose(matrices.b_matrix, expected_b, atol=1e-10)


def test_response_hf_strict_bound_exposes_matrix_free_nonlocal_action_by_default():
    molecule = _make_pt2_toy_molecule()
    molecule.rep_tensor = jnp.zeros_like(molecule.rep_tensor)
    molecule.hfx_nu = jnp.asarray(
        [
            [
                [[0.7, 0.2], [0.2, 0.5]],
                [[0.4, -0.1], [-0.1, 0.6]],
            ],
        ],
        dtype=jnp.float64,
    )
    molecule.hfx_local = None
    alpha = 0.25
    semilocal_zero = lambda features: jnp.zeros_like(features.rho)
    functional = NeuralXCFunctional(
        model=_ConstantChannelModel((0.0, alpha)),
        semilocal_energy_density_fn=semilocal_zero,
        input_feature_mode="enhanced",
        hf_input_mode="total_only",
        response_hf_mode="strict",
        name="hf_strict_channel_action",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(224), molecule)
    bound = functional.bind_to_molecule_for_response(params, molecule)

    hf_a, _ = bound.nonlocal_response_matrices(molecule)
    amplitudes = jnp.asarray([[[0.4]], [[1.2]]], dtype=jnp.float64)
    action = bound.nonlocal_response_action(molecule, amplitudes)
    b_action = bound.nonlocal_response_b_action(molecule, amplitudes)
    diagonal = bound.nonlocal_response_diagonal(molecule)
    _, hf_b = bound.nonlocal_response_matrices(molecule)

    assert jnp.allclose(action.reshape(2, 1), amplitudes.reshape(2, 1) @ hf_a.T, atol=1e-10)
    assert jnp.allclose(b_action.reshape(2, 1), amplitudes.reshape(2, 1) @ hf_b.T, atol=1e-10)
    assert jnp.allclose(diagonal, jnp.diag(hf_a).reshape(1, 1), atol=1e-10)


@pytest.mark.parametrize("strict_hfx_response_mode", ["dense", "low_memory"])
def test_response_hf_strict_tda_operator_uses_matrix_free_nonlocal_action(
    monkeypatch,
    strict_hfx_response_mode,
):
    molecule = _make_pt2_toy_molecule()
    molecule.rep_tensor = jnp.zeros_like(molecule.rep_tensor)
    molecule.hfx_nu = jnp.asarray(
        [
            [
                [[0.7, 0.2], [0.2, 0.5]],
                [[0.4, -0.1], [-0.1, 0.6]],
            ],
        ],
        dtype=jnp.float64,
    )
    molecule.hfx_local = None
    semilocal_zero = lambda features: jnp.zeros_like(features.rho)
    functional = NeuralXCFunctional(
        model=_ConstantChannelModel((0.0, 0.25)),
        semilocal_energy_density_fn=semilocal_zero,
        input_feature_mode="enhanced",
        hf_input_mode="total_only",
        response_hf_mode="strict",
        strict_hfx_response_mode=strict_hfx_response_mode,
        name="hf_strict_channel_operator_action",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(225), molecule)
    _, dense_a = build_restricted_tda_matrix(molecule, functional, xc_params=params)

    def _fail_matrix_pair(*args, **kwargs):
        del args, kwargs
        raise AssertionError("operator path must not materialize strict HFX matrix pair")

    monkeypatch.setattr(
        NeuralXCFunctional,
        "_strict_hf_nonlocal_response_matrices",
        _fail_matrix_pair,
    )
    vind, diagonal, _, _ = build_restricted_tda_operator(
        molecule,
        functional,
        xc_params=params,
        materialize_matrix=False,
    )
    amplitudes = jnp.asarray([[[0.4]], [[1.2]]], dtype=jnp.float64)

    assert jnp.allclose(vind(amplitudes), amplitudes.reshape(2, 1) @ dense_a.reshape(1, 1).T, atol=1e-10)
    assert jnp.allclose(diagonal, jnp.diag(dense_a.reshape(1, 1)), atol=1e-10)


@pytest.mark.parametrize("strict_hfx_response_mode", ["dense", "low_memory"])
def test_response_hf_strict_tda_operator_backward_uses_matrix_free_nonlocal_action(
    monkeypatch,
    strict_hfx_response_mode,
):
    molecule = _make_pt2_toy_molecule()
    molecule.rep_tensor = jnp.zeros_like(molecule.rep_tensor)
    molecule.hfx_nu = jnp.asarray(
        [
            [
                [[0.7, 0.2], [0.2, 0.5]],
                [[0.4, -0.1], [-0.1, 0.6]],
            ],
        ],
        dtype=jnp.float64,
    )
    molecule.hfx_local = None
    semilocal_zero = lambda features: jnp.zeros_like(features.rho)

    class _ParametricHFChannelModel(nn.Module):
        @nn.compact
        def __call__(self, inputs):
            scale = self.param(
                "scale",
                lambda key, shape: jnp.asarray(0.25, dtype=inputs.dtype),
                (),
            )
            coeff = jnp.zeros(inputs.shape[:-1] + (2,), dtype=inputs.dtype)
            return coeff.at[..., 1].set(scale * (1.0 + 0.1 * inputs[..., 0]))

    functional = NeuralXCFunctional(
        model=_ParametricHFChannelModel(),
        semilocal_energy_density_fn=semilocal_zero,
        input_feature_mode="enhanced",
        hf_input_mode="total_only",
        response_hf_mode="strict",
        strict_hfx_response_mode=strict_hfx_response_mode,
        name="hf_strict_channel_operator_backward_action",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(226), molecule)

    def _fail_matrix_pair(*args, **kwargs):
        del args, kwargs
        raise AssertionError("backward path must not materialize strict HFX matrix pair")

    monkeypatch.setattr(
        NeuralXCFunctional,
        "_strict_hf_nonlocal_response_matrices",
        _fail_matrix_pair,
    )
    amplitudes = jnp.asarray([[[0.4]], [[1.2]]], dtype=jnp.float64)

    def objective(local_params):
        vind, diagonal, _, _ = build_restricted_tda_operator(
            molecule,
            functional,
            xc_params=local_params,
            materialize_matrix=False,
        )
        return jnp.sum(vind(amplitudes)) + 0.01 * jnp.sum(diagonal)

    grads = jax.grad(objective)(params)
    leaves = jax.tree_util.tree_leaves(grads)

    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    assert any(float(jnp.linalg.norm(leaf)) > 0.0 for leaf in leaves)


def test_low_memory_strict_hfx_response_matches_dense_path():
    molecule = _make_pt2_toy_molecule()
    molecule.rep_tensor = jnp.zeros_like(molecule.rep_tensor)
    molecule.hfx_nu = jnp.asarray(
        [
            [
                [[0.7, 0.2], [0.2, 0.5]],
                [[0.4, -0.1], [-0.1, 0.6]],
            ],
        ],
        dtype=jnp.float64,
    )
    molecule.hfx_local = None
    semilocal_zero = lambda features: jnp.zeros_like(features.rho)
    dense_functional = NeuralXCFunctional(
        model=_HFResponsiveChannelModel(),
        semilocal_energy_density_fn=semilocal_zero,
        input_feature_mode="enhanced",
        hf_input_mode="total_only",
        response_hf_mode="strict",
        strict_hfx_response_mode="dense",
        response_grid_chunk_size=1024,
        name="hf_strict_dense_response",
    )
    low_memory_functional = NeuralXCFunctional(
        model=_HFResponsiveChannelModel(),
        semilocal_energy_density_fn=semilocal_zero,
        input_feature_mode="enhanced",
        hf_input_mode="total_only",
        response_hf_mode="strict",
        strict_hfx_response_mode="low_memory",
        response_grid_chunk_size=1,
        name="hf_strict_low_memory_response",
    )
    params_dense = dense_functional.init_from_molecule(jax.random.PRNGKey(225), molecule)
    params_low_memory = low_memory_functional.init_from_molecule(
        jax.random.PRNGKey(226),
        molecule,
    )

    dense_a, dense_b = dense_functional.bind_to_molecule_for_response(
        params_dense,
        molecule,
    ).nonlocal_response_matrices(molecule)
    low_memory_a, low_memory_b = low_memory_functional.bind_to_molecule_for_response(
        params_low_memory,
        molecule,
    ).nonlocal_response_matrices(molecule)

    assert jnp.allclose(low_memory_a, dense_a, atol=1e-10)
    assert jnp.allclose(low_memory_b, dense_b, atol=1e-10)


def test_chunked_low_memory_strict_hfx_response_matches_dense_path():
    molecule = _make_pt2_toy_molecule()
    molecule.rep_tensor = jnp.zeros_like(molecule.rep_tensor)
    dense_nu = jnp.asarray(
        [
            [
                [[0.7, 0.2], [0.2, 0.5]],
                [[0.4, -0.1], [-0.1, 0.6]],
            ],
        ],
        dtype=jnp.float64,
    )
    molecule.hfx_nu = dense_nu
    molecule.hfx_local = None
    chunked_molecule = replace(molecule, hfx_nu=None, hfx_local=None)
    chunked_molecule.hfx_nu_api = ChunkedHFXNu.from_dense(dense_nu, chunk_size=1)
    semilocal_zero = lambda features: jnp.zeros_like(features.rho)
    dense_functional = NeuralXCFunctional(
        model=_HFResponsiveChannelModel(),
        semilocal_energy_density_fn=semilocal_zero,
        input_feature_mode="enhanced",
        hf_input_mode="total_only",
        response_hf_mode="strict",
        strict_hfx_response_mode="dense",
        response_grid_chunk_size=1024,
        name="hf_strict_dense_response_chunk_api_reference",
    )
    chunked_functional = NeuralXCFunctional(
        model=_HFResponsiveChannelModel(),
        semilocal_energy_density_fn=semilocal_zero,
        input_feature_mode="enhanced",
        hf_input_mode="total_only",
        response_hf_mode="strict",
        strict_hfx_response_mode="low_memory",
        response_grid_chunk_size=1,
        name="hf_strict_low_memory_response_chunk_api",
    )
    params_dense = dense_functional.init_from_molecule(jax.random.PRNGKey(228), molecule)
    params_chunked = chunked_functional.init_from_molecule(
        jax.random.PRNGKey(229),
        chunked_molecule,
    )

    dense_bound = dense_functional.bind_to_molecule_for_response(
        params_dense,
        molecule,
    )
    chunked_bound = chunked_functional.bind_to_molecule_for_response(
        params_chunked,
        chunked_molecule,
    )

    dense_a, dense_b = dense_bound.nonlocal_response_matrices(molecule)
    chunked_a, chunked_b = chunked_bound.nonlocal_response_matrices(chunked_molecule)
    amplitudes = jnp.asarray([[0.7]], dtype=jnp.float64)

    assert jnp.allclose(chunked_a, dense_a, atol=1e-10)
    assert jnp.allclose(chunked_b, dense_b, atol=1e-10)
    assert jnp.allclose(
        chunked_bound.nonlocal_response_action(chunked_molecule, amplitudes),
        dense_bound.nonlocal_response_action(molecule, amplitudes),
        atol=1e-10,
    )
    assert jnp.allclose(
        chunked_bound.nonlocal_response_b_action(chunked_molecule, amplitudes),
        dense_bound.nonlocal_response_b_action(molecule, amplitudes),
        atol=1e-10,
    )
    assert jnp.allclose(
        chunked_bound.nonlocal_response_diagonal(chunked_molecule),
        dense_bound.nonlocal_response_diagonal(molecule),
        atol=1e-10,
    )


def test_hdf5_chunked_low_memory_response_supports_jitted_param_grad(tmp_path):
    h5py = pytest.importorskip("h5py")

    molecule = _make_pt2_toy_molecule()
    molecule.rep_tensor = jnp.zeros_like(molecule.rep_tensor)
    molecule.nocc = 1
    dense_nu = np.asarray(
        [
            [
                [[0.7, 0.2], [0.2, 0.5]],
                [[0.4, -0.1], [-0.1, 0.6]],
            ],
        ],
        dtype=np.float64,
    )
    path = tmp_path / "refs.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("hfx_nu", data=dense_nu)
    molecule.hfx_nu = None
    molecule.hfx_local = None
    molecule.hfx_nu_api = ChunkedHFXNu.from_hdf5_dataset(
        str(path),
        "hfx_nu",
        chunk_size=1,
    )
    functional = NeuralXCFunctional(
        model=_HFParametricResponsiveChannelModel(),
        semilocal_energy_density_fn=lambda features: jnp.zeros_like(features.rho),
        input_feature_mode="enhanced",
        hf_input_mode="total_only",
        response_hf_mode="strict",
        strict_hfx_response_mode="low_memory",
        response_grid_chunk_size=1,
        name="hf_strict_low_memory_response_hdf5_chunk_api",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(230), molecule)
    amplitudes = jnp.asarray([[0.7]], dtype=jnp.float64)

    @jax.jit
    def objective(local_params):
        bound = functional.bind_to_molecule_for_response(local_params, molecule)
        return jnp.sum(bound.nonlocal_response_action(molecule, amplitudes))

    value = objective(params)
    grads = jax.jit(jax.grad(objective))(params)
    leaves = jax.tree_util.tree_leaves(grads)

    assert bool(jnp.isfinite(value))
    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


def test_low_memory_strict_hfx_response_does_not_pad_full_hfx_nu(monkeypatch):
    molecule = _make_three_grid_pt2_toy_molecule()
    molecule.rep_tensor = jnp.zeros_like(molecule.rep_tensor)
    molecule.hfx_nu = jnp.asarray(
        [
            [
                [[0.7, 0.2], [0.2, 0.5]],
                [[0.4, -0.1], [-0.1, 0.6]],
                [[0.3, 0.0], [0.0, 0.2]],
            ],
        ],
        dtype=jnp.float64,
    )
    molecule.hfx_local = None
    semilocal_zero = lambda features: jnp.zeros_like(features.rho)
    functional = NeuralXCFunctional(
        model=_HFResponsiveChannelModel(),
        semilocal_energy_density_fn=semilocal_zero,
        input_feature_mode="enhanced",
        hf_input_mode="total_only",
        response_hf_mode="strict",
        strict_hfx_response_mode="low_memory",
        response_grid_chunk_size=2,
        name="hf_strict_low_memory_no_full_nu_pad",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(227), molecule)
    original_pad = neural_xc_model.jnp.pad

    def guard_full_hfx_nu_pad(values, *args, **kwargs):
        arr = jnp.asarray(values)
        if arr.shape == molecule.hfx_nu.shape:
            raise AssertionError("low-memory HFX response must not pad full hfx_nu")
        return original_pad(values, *args, **kwargs)

    monkeypatch.setattr(neural_xc_model.jnp, "pad", guard_full_hfx_nu_pad)

    low_memory_a, low_memory_b = functional.bind_to_molecule_for_response(
        params,
        molecule,
    ).nonlocal_response_matrices(molecule)

    assert jnp.all(jnp.isfinite(low_memory_a))
    assert jnp.all(jnp.isfinite(low_memory_b))


def test_low_memory_scf_fock_terms_match_value_and_grad_reference():
    molecule = _make_three_grid_pt2_toy_molecule()
    molecule.rep_tensor = jnp.zeros_like(molecule.rep_tensor)
    molecule.hfx_local = jnp.asarray(
        [
            [[-0.18, -0.126], [-0.06, -0.042], [-0.12, -0.084]],
            [[-0.12, -0.084], [-0.03, -0.021], [-0.08, -0.056]],
        ],
        dtype=jnp.float64,
    )
    molecule.hfx_nu = jnp.asarray(
        [
            [
                [[0.7, 0.2], [0.2, 0.5]],
                [[0.4, -0.1], [-0.1, 0.6]],
                [[0.3, 0.0], [0.0, 0.2]],
            ],
            [
                [[0.2, 0.1], [0.1, 0.4]],
                [[0.1, 0.0], [0.0, 0.3]],
                [[0.5, -0.2], [-0.2, 0.6]],
            ],
        ],
        dtype=jnp.float64,
    )
    semilocal_zero = lambda features: jnp.zeros_like(features.rho)
    functional = NeuralXCFunctional(
        model=_HFResponsiveChannelModel(),
        semilocal_energy_density_fn=semilocal_zero,
        input_feature_mode="canonical",
        hf_input_mode="spin_resolved",
        response_hf_mode="strict",
        strict_hfx_response_mode="low_memory",
        response_grid_chunk_size=2,
        hfx_channels=2,
        name="hf_strict_low_memory_scf_terms",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(229), molecule)
    density = molecule.rdm1.sum(axis=0)

    extra_fock = functional.scf_extra_fock_for_density(params, molecule, density)
    reference = xc_energy_and_potential_from_density(
        params,
        molecule=molecule,
        density=density,
        xc_energy_fn=functional.scf_xc_energy_and_alpha_for_density,
        extra_fock_matrix=extra_fock,
        has_aux=True,
    )
    vxc_matrix, alpha, direct_extra_fock, energy = functional.scf_xc_fock_terms(
        params,
        functional.scf_molecule_with_density(molecule, density),
        weights=molecule.grid.weights,
        functional_dtype=density.dtype,
        vxc_clip=20.0,
    )

    assert jnp.allclose(vxc_matrix, reference.vxc_matrix, atol=1e-10)
    assert jnp.allclose(direct_extra_fock, reference.extra_fock_matrix, atol=1e-10)
    assert jnp.allclose(energy, reference.xc_energy, atol=1e-10)
    assert jnp.allclose(alpha, reference.aux, atol=1e-10)


def test_low_memory_strict_tda_matrix_builds_transition_features_per_chunk(monkeypatch):
    molecule = _make_three_grid_pt2_toy_molecule()
    molecule.rep_tensor = jnp.zeros_like(molecule.rep_tensor)
    molecule.hfx_nu = jnp.asarray(
        [
            [
                [[0.7, 0.2], [0.2, 0.5]],
                [[0.4, -0.1], [-0.1, 0.6]],
                [[0.3, 0.0], [0.0, 0.2]],
            ],
        ],
        dtype=jnp.float64,
    )
    molecule.hfx_local = None
    semilocal_zero = lambda features: jnp.zeros_like(features.rho)
    functional = NeuralXCFunctional(
        model=_HFResponsiveChannelModel(),
        semilocal_energy_density_fn=semilocal_zero,
        input_feature_mode="enhanced",
        hf_input_mode="total_only",
        response_hf_mode="strict",
        strict_hfx_response_mode="low_memory",
        response_grid_chunk_size=2,
        name="hf_strict_low_memory_chunked_tda_features",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(228), molecule)

    def fail_full_transition_features(*args, **kwargs):
        raise AssertionError("low-memory strict TDA must not build full-grid response features")

    monkeypatch.setattr(
        neural_xc_binding,
        "restricted_transition_response_features",
        fail_full_transition_features,
    )

    matrix = functional.bind_to_molecule_for_response(params, molecule).strict_tda_xc_matrix(
        molecule
    )

    assert jnp.all(jnp.isfinite(matrix))


def test_strict_tda_matrix_floors_singular_response_variables_before_hessian_gradients():
    molecule = _make_toy_molecule()
    molecule.rep_tensor = jnp.zeros_like(molecule.rep_tensor)
    molecule.ao_deriv1 = molecule.ao_deriv1.at[1:, 0, :].set(0.0)
    non_hf_module = make_custom_semilocal_module(
        channel_names=("singular_semilocal_channel",),
        energy_density_channels_fn=lambda features: (
            jnp.sqrt(features.sigma) + jnp.sqrt(features.tau_a + features.tau_b)
        )[..., None],
        name="singular_response_variable_module",
    )
    functional = make_neural_xc_functional(
        non_hf_module=non_hf_module,
        hidden_dims=(4,),
        network_architecture="simple_mlp",
        response_hf_mode="strict",
        strict_hfx_response_mode="low_memory",
        response_grid_chunk_size=1,
        name="inactive_grid_hessian_guard",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(230), molecule)

    def matrix_sum(local_params):
        bound = functional.bind_to_molecule_for_response(local_params, molecule)
        return jnp.sum(bound.strict_tda_xc_matrix(molecule))

    grads = jax.grad(matrix_sum)(params)

    assert all(
        jnp.all(jnp.isfinite(jnp.asarray(leaf)))
        for leaf in jax.tree_util.tree_leaves(grads)
    )


def test_tda_builder_uses_low_memory_strict_tda_matrix_callback(monkeypatch):
    molecule = _make_three_grid_pt2_toy_molecule()
    molecule.rep_tensor = jnp.zeros_like(molecule.rep_tensor)
    molecule.hfx_nu = jnp.asarray(
        [
            [
                [[0.7, 0.2], [0.2, 0.5]],
                [[0.4, -0.1], [-0.1, 0.6]],
                [[0.3, 0.0], [0.0, 0.2]],
            ],
        ],
        dtype=jnp.float64,
    )
    molecule.hfx_local = None
    semilocal_zero = lambda features: jnp.zeros_like(features.rho)
    functional = NeuralXCFunctional(
        model=_HFResponsiveChannelModel(),
        semilocal_energy_density_fn=semilocal_zero,
        input_feature_mode="enhanced",
        hf_input_mode="total_only",
        response_hf_mode="strict",
        strict_hfx_response_mode="low_memory",
        response_grid_chunk_size=2,
        name="hf_strict_low_memory_tda_builder_callback",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(229), molecule)

    def fail_full_transition_features(*args, **kwargs):
        raise AssertionError("TDA builder must use the strict TDA callback in low-memory mode")

    monkeypatch.setattr(
        tddft_response,
        "restricted_transition_response_features",
        fail_full_transition_features,
    )

    _, matrix = build_restricted_tda_matrix(
        molecule,
        functional,
        xc_params=params,
    )

    assert jnp.all(jnp.isfinite(matrix))


def test_tda_operator_uses_low_memory_strict_tda_action_without_matrix(monkeypatch):
    molecule = _make_three_grid_pt2_toy_molecule()
    molecule.rep_tensor = jnp.zeros_like(molecule.rep_tensor)
    molecule.hfx_nu = jnp.asarray(
        [
            [
                [[0.7, 0.2], [0.2, 0.5]],
                [[0.4, -0.1], [-0.1, 0.6]],
                [[0.3, 0.0], [0.0, 0.2]],
            ],
        ],
        dtype=jnp.float64,
    )
    molecule.hfx_local = None
    semilocal_zero = lambda features: jnp.zeros_like(features.rho)
    functional = NeuralXCFunctional(
        model=_HFResponsiveChannelModel(),
        semilocal_energy_density_fn=semilocal_zero,
        input_feature_mode="enhanced",
        hf_input_mode="total_only",
        response_hf_mode="strict",
        strict_hfx_response_mode="low_memory",
        response_grid_chunk_size=2,
        name="hf_strict_low_memory_tda_operator_action",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(232), molecule)

    def fail_full_transition_features(*args, **kwargs):
        raise AssertionError("operator action must not build full-grid response features")

    def fail_strict_tda_matrix(*args, **kwargs):
        raise AssertionError("operator action must not materialize the strict TDA XC matrix")

    monkeypatch.setattr(
        tddft_response,
        "restricted_transition_response_features",
        fail_full_transition_features,
    )
    monkeypatch.setattr(
        neural_xc_binding,
        "restricted_transition_response_features",
        fail_full_transition_features,
    )
    monkeypatch.setattr(
        NeuralXCFunctional,
        "_strict_tda_xc_matrix_chunked",
        fail_strict_tda_matrix,
    )

    vind, diagonal, _, _ = build_restricted_tda_operator(
        molecule,
        functional,
        xc_params=params,
        materialize_matrix=False,
    )
    amplitudes = jnp.ones((1, 1, 2), dtype=jnp.float64)

    assert jnp.all(jnp.isfinite(diagonal))
    assert jnp.all(jnp.isfinite(vind(amplitudes)))


def test_response_pt2_approx_keeps_pt2_as_frozen_basis_channel():
    molecule = _make_pt2_toy_molecule()
    molecule.pt2_local = jnp.asarray([-0.30, 0.40], dtype=jnp.float64)
    semilocal_zero = lambda features: jnp.zeros_like(features.rho)
    functional = NeuralXCFunctional(
        model=_DensityResponsivePT2ChannelModel(),
        semilocal_energy_density_fn=semilocal_zero,
        include_pt2_channel=True,
        pt2_channel_mode="local_exact",
        response_hf_mode="approx",
        response_pt2_mode="approx",
        name="pt2_frozen_basis_response",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(224), molecule)

    bound = functional.bind_to_molecule_for_response(params, molecule)
    tensor = bound.grid_response_tensor(molecule)

    assert jnp.allclose(tensor[0, 0], 0.5 * molecule.pt2_local, atol=1e-10)


def test_response_pt2_strict_uses_no_pt2_response_and_posthoc_correction():
    molecule = _make_pt2_toy_molecule()
    molecule.ao = jnp.asarray([[0.8, 0.3], [0.4, -0.7]], dtype=jnp.float64)
    molecule.ao_deriv1 = jnp.stack(
        [
            molecule.ao,
            jnp.asarray([[0.05, 0.02], [0.01, 0.07]], dtype=jnp.float64),
            jnp.asarray([[0.01, -0.03], [0.04, 0.02]], dtype=jnp.float64),
            jnp.asarray([[0.02, 0.01], [-0.01, 0.03]], dtype=jnp.float64),
        ]
    )
    rep_tensor = jnp.zeros((2, 2, 2, 2), dtype=jnp.float64)
    rep_tensor = rep_tensor.at[0, 1, 0, 1].set(1.0)
    rep_tensor = rep_tensor.at[0, 0, 1, 1].set(0.3)
    rep_tensor = rep_tensor.at[0, 1, 1, 0].set(0.5)
    rep_tensor = rep_tensor.at[1, 0, 0, 1].set(0.2)
    molecule.rep_tensor = rep_tensor
    molecule.mo_energy = jnp.asarray([[0.0, 3.0], [0.0, 3.0]], dtype=jnp.float64)
    molecule.pt2_local = jnp.asarray([-0.30, 0.40], dtype=jnp.float64)
    semilocal_zero = lambda features: jnp.zeros_like(features.rho)

    pt2_strict = NeuralXCFunctional(
        model=_DensityResponsivePT2ChannelModel(),
        semilocal_energy_density_fn=semilocal_zero,
        include_pt2_channel=True,
        pt2_channel_mode="local_exact",
        response_hf_mode="approx",
        response_pt2_mode="strict",
        name="pt2_strict",
    )
    params_strict = pt2_strict.init_from_molecule(jax.random.PRNGKey(122), molecule)

    no_pt2_matrices = build_restricted_response_matrices(molecule, None)
    strict_matrices = build_restricted_response_matrices(
        molecule,
        pt2_strict,
        xc_params=params_strict,
    )
    solver = RestrictedCasidaTDDFT(
        molecule,
        pt2_strict,
        xc_params=params_strict,
        eigensolver="dense",
    )
    result = solver.tda(nstates=1)
    features = restricted_grid_features(molecule)
    hf_projected, hf_projected_a, hf_projected_b = (
        pt2_strict.projected_hf_grid_contribution_components(
            molecule,
            features=features,
        )
    )
    pt2_projected = pt2_strict.projected_pt2_grid_contribution(
        molecule,
        features=features,
    )
    coefficients = pt2_strict.channel_coefficients(
        params_strict,
        features,
        molecule=molecule,
        semilocal_energy_density=semilocal_zero(features),
        hf_energy_density=hf_projected,
        pt2_energy_density=pt2_projected,
        hf_spin_energy_density=(hf_projected_a, hf_projected_b),
    )
    pt2_coefficients = coefficients[..., 1]
    rho = jnp.maximum(features.rho, pt2_strict.density_floor)
    expected_ac = jnp.tensordot(molecule.grid.weights, rho * pt2_coefficients, axes=(0, 0))
    expected_ac = expected_ac / jnp.maximum(
        jnp.tensordot(molecule.grid.weights, rho, axes=(0, 0)),
        pt2_strict.density_floor,
    )
    expected_correction = restricted_cisd_second_order_correction(
        molecule,
        replace(
            result,
            excitation_energies=no_pt2_matrices.a_matrix.reshape(1),
            posthoc_correction=None,
        ),
        ac=expected_ac,
    )

    assert jnp.allclose(strict_matrices.a_matrix, no_pt2_matrices.a_matrix, atol=1e-10)
    assert jnp.allclose(strict_matrices.b_matrix, no_pt2_matrices.b_matrix, atol=1e-10)
    assert jnp.allclose(result.posthoc_correction, expected_correction, atol=1e-10)
    assert jnp.allclose(
        result.excitation_energies,
        no_pt2_matrices.a_matrix.reshape(1) + expected_correction,
        atol=1e-10,
    )


def test_open_shell_response_pt2_strict_uses_no_pt2_response_and_zero_posthoc_correction():
    molecule = _make_open_shell_toy_molecule()
    molecule.pt2_local = jnp.asarray([-0.15, 0.25], dtype=jnp.float64)
    semilocal_zero = lambda features: jnp.zeros_like(features.rho)

    pt2_strict = NeuralXCFunctional(
        model=_DensityResponsivePT2ChannelModel(),
        semilocal_energy_density_fn=semilocal_zero,
        include_pt2_channel=True,
        pt2_channel_mode="local_exact",
        response_hf_mode="approx",
        response_pt2_mode="strict",
        name="open_shell_pt2_strict",
    )
    params_strict = pt2_strict.init_from_molecule(jax.random.PRNGKey(177), molecule)

    no_pt2_matrices = build_unrestricted_response_matrices(molecule, None)
    strict_matrices = build_unrestricted_response_matrices(
        molecule,
        pt2_strict,
        xc_params=params_strict,
    )
    solver = UnrestrictedCasidaTDDFT(
        molecule,
        pt2_strict,
        xc_params=params_strict,
    )
    result = solver.tda(nstates=1)
    expected_correction = unrestricted_cisd_second_order_correction(
        molecule,
        replace(result, excitation_energies=no_pt2_matrices.a_matrix.reshape(1)),
        ac=0.4,
    )

    assert jnp.allclose(strict_matrices.a_matrix, no_pt2_matrices.a_matrix, atol=1e-10)
    assert jnp.allclose(strict_matrices.b_matrix, no_pt2_matrices.b_matrix, atol=1e-10)
    assert jnp.allclose(result.posthoc_correction, expected_correction, atol=1e-10)
    assert jnp.allclose(
        result.excitation_energies,
        no_pt2_matrices.a_matrix.reshape(1) + expected_correction,
        atol=1e-10,
    )


def test_bound_neural_xc_exposes_strict_mgga_response_tensor():
    molecule = _make_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=(8, 8),
        name="strict_response_tensor_check",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(23), molecule)
    bound = functional.bind_to_molecule(params, molecule)
    tensor = bound.grid_response_tensor(molecule)

    assert bound.response_feature_kind == "MGGA"
    assert tensor.shape == (5, 5, molecule.grid.weights.shape[0])
    assert jnp.all(jnp.isfinite(tensor))
    assert jnp.allclose(tensor, jnp.swapaxes(tensor, 0, 1), atol=1e-8)


def test_bound_neural_xc_reuses_precomputed_strict_response_tensor(monkeypatch):
    molecule = _make_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=(8, 8),
        name="strict_response_tensor_cache_check",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(231), molecule)

    original = NeuralXCFunctional._strict_total_response_tensor
    calls = {"count": 0}

    def wrapped(self, *args, **kwargs):
        calls["count"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(NeuralXCFunctional, "_strict_total_response_tensor", wrapped)

    bound = functional.bind_to_molecule(params, molecule)
    tensor = bound.grid_response_tensor(molecule)

    assert tensor.shape == (5, 5, molecule.grid.weights.shape[0])
    assert calls["count"] == 1


def test_gga_neural_xc_tda_gradient_is_finite_for_water():
    _pyscf_or_skip()
    reference = restricted_reference_from_pyscf(
        _make_water_b3lyp_reference(),
        compute_local_hfx_features=True,
        compute_local_hfx_aux=True,
        hfx_omega_values=(0.0, 0.4),
    )
    functional = make_neural_xc_functional(
        semilocal_xc=("gga_x_pbe", "gga_c_pbe"),
        hidden_dims=(8, 8),
        input_feature_mode="enhanced",
        name="gga_tda_grad_check",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(24), reference)

    def s1_energy(p):
        return predict_excitation_energies(
            p,
            functional,
            reference,
            nstates=1,
            use_tda=True,
        )[0]

    grad = jax.grad(s1_energy)(params)
    leaves = jax.tree_util.tree_leaves(grad)
    absmax = max(float(jnp.max(jnp.abs(jnp.asarray(leaf)))) for leaf in leaves)

    assert all(bool(jnp.all(jnp.isfinite(jnp.asarray(leaf)))) for leaf in leaves)
    assert absmax > 0.0


def test_coefficient_prior_penalty_is_reported_and_nonnegative():
    molecule = _make_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc=("lda_x", "gga_c_pbe"),
        hidden_dims=(8, 8),
        name="coefficient_prior_check",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(25), molecule)
    datum = GroundStateDatum(molecule=molecule, target_total_energy=jnp.array(0.2))

    plain_loss, _ = ground_state_mse_loss(params, functional, datum)
    constrained_loss, metrics = ground_state_mse_loss(
        params,
        functional,
        datum,
        training_config=GroundStateTrainingConfig(
            coefficient_prior_weight=1.0,
            coefficient_prior_values=(0.1, 0.2, 0.3),
        ),
    )

    assert metrics["coefficient_prior_penalty"].shape == (1,)
    assert metrics["coefficient_prior_mse"].shape == (1,)
    assert metrics["coefficient_prior_penalty"][0] >= 0.0
    assert constrained_loss >= plain_loss


def test_strict_response_mode_matches_pyscf_b3lyp_tolerance():
    _pyscf_or_skip()
    mf = _make_water_b3lyp_reference()
    reference = restricted_reference_from_pyscf(
        mf,
        compute_local_hfx_features=True,
        compute_local_hfx_aux=True,
    )

    td = mf.TDDFT()
    td.nstates = 6
    td.kernel()
    ref_energies = jnp.asarray(td.e)
    ref_osc = jnp.asarray(td.oscillator_strength())

    def run_mode(mode: str) -> tuple[float, float]:
        functional = NeuralXCFunctional(
            model=_ConstantChannelModel((0.08, 0.72, 0.19, 0.81, 0.20)),
            semilocal_xc=b3lyp_component_basis(),
            hf_input_mode="total_only",
            response_hf_mode=mode,
            name=f"b3lyp_like_{mode}",
        )
        params = functional.init_from_molecule(jax.random.PRNGKey(30), reference)
        bound = functional.bind_to_molecule(params, reference)
        solver = RestrictedCasidaTDDFT(molecule=reference, xc_functional=bound)
        result = solver.kernel(nstates=6)
        pred_energies = jnp.asarray(result.excitation_energies)
        pred_osc = oscillator_strengths(reference, result)
        n = min(ref_energies.size, pred_energies.size, ref_osc.size, pred_osc.size, 6)
        mae_e = float(
            jnp.mean(jnp.abs((pred_energies[:n] - ref_energies[:n]) * HARTREE_TO_EV))
        )
        mae_f = float(jnp.mean(jnp.abs(pred_osc[:n] - ref_osc[:n])))
        return mae_e, mae_f

    strict_mae_e, strict_mae_f = run_mode("strict")
    approx_mae_e, approx_mae_f = run_mode("approx")

    assert strict_mae_e < 0.25
    assert strict_mae_f < 0.02
    assert approx_mae_e < 0.25
    assert approx_mae_f < 0.02
