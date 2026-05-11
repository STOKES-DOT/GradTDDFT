import numpy as np
import pytest
import jax
import jax.numpy as jnp

from td_graddft import neural_xc
from pyscf_reference import restricted_reference_from_pyscf
from td_graddft.tddft.test_module import (
    LocalHFKhhResponseFunctionalWrapper,
    build_restricted_local_hf_khh_tda_matrix,
)
from td_graddft.training.targets import predict_excitation_energies


def _pyscf_or_skip():
    try:
        from pyscf import scf, gto  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("PySCF is required for local HF response tests.")


def _make_h2_hf_reference():
    from pyscf import dft, gto

    mol = gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 0.74"
    mol.unit = "Angstrom"
    mol.basis = "sto-3g"
    mol.spin = 0
    mol.verbose = 0
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = "hf"
    mf.grids.level = 0
    mf.conv_tol = 1e-12
    mf.max_cycle = 80
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RKS(HF/H2/STO-3G) did not converge.")
    return restricted_reference_from_pyscf(
        mf,
        compute_local_hfx_features=True,
        compute_local_hfx_aux=True,
        hfx_omega_values=(0.0,),
    )


def _make_h2_b3lyp_reference():
    from pyscf import dft, gto

    mol = gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 0.74"
    mol.unit = "Angstrom"
    mol.basis = "sto-3g"
    mol.spin = 0
    mol.verbose = 0
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = "b3lyp"
    mf.grids.level = 0
    mf.conv_tol = 1e-12
    mf.max_cycle = 80
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RKS(B3LYP/H2/STO-3G) did not converge.")
    return restricted_reference_from_pyscf(
        mf,
        compute_local_hfx_features=True,
        compute_local_hfx_aux=True,
        hfx_omega_values=(0.0,),
    )


def test_local_hf_khh_tda_matrix_respects_zero_and_scalar_weight_scaling():
    _pyscf_or_skip()
    reference = _make_h2_hf_reference()
    zero = build_restricted_local_hf_khh_tda_matrix(
        reference,
        local_weight=0.0,
        omega_index=0,
    )
    base = build_restricted_local_hf_khh_tda_matrix(
        reference,
        local_weight=1.0,
        omega_index=0,
    )
    scaled = build_restricted_local_hf_khh_tda_matrix(
        reference,
        local_weight=0.25,
        omega_index=0,
    )
    np.testing.assert_allclose(
        np.asarray(zero, dtype=float),
        np.zeros_like(np.asarray(zero, dtype=float)),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(scaled, dtype=float),
        0.25 * np.asarray(base, dtype=float),
        atol=1e-6,
        rtol=1e-6,
    )
    assert np.all(np.isfinite(np.asarray(base, dtype=float)))
    np.testing.assert_allclose(
        np.asarray(base, dtype=float).reshape(1, 1),
        np.asarray(base, dtype=float).reshape(1, 1).T,
        atol=1e-8,
        rtol=1e-8,
    )


def test_local_hf_khh_response_wrapper_preserves_tda_parameter_gradients():
    _pyscf_or_skip()
    reference = _make_h2_b3lyp_reference()
    functional = neural_xc.Functional(
        architecture="residual",
        semilocal_xc=("lda_x", "gga_x_b88", "lda_c_vwn_rpa", "gga_c_lyp"),
        hidden_dims=(8, 8),
        include_pt2_channel=True,
        pt2_channel_mode="scaled_projected",
        response_hf_mode="local_projected",
        name="local_hf_khh_grad_test",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(7), reference)
    wrapped = LocalHFKhhResponseFunctionalWrapper(functional)

    def s1_energy(p):
        return predict_excitation_energies(
            p,
            wrapped,
            reference,
            nstates=1,
            use_tda=True,
        )[0]

    value = s1_energy(params)
    grad = jax.grad(s1_energy)(params)
    leaves = jax.tree_util.tree_leaves(grad)
    absmax = max(float(jnp.max(jnp.abs(jnp.asarray(leaf)))) for leaf in leaves)

    assert jnp.isfinite(value)
    assert all(bool(jnp.all(jnp.isfinite(jnp.asarray(leaf)))) for leaf in leaves)
    assert absmax > 0.0
