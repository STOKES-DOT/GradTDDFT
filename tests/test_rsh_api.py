import jax.numpy as jnp
import numpy as np
import pytest

from td_graddft.dft.rsh import (
    ResolvedRSHParameters,
    SCFXCContributions,
    make_pyscf_rsh_spec,
)


def test_resolved_rsh_parameters_expose_paper_and_pyscf_views():
    params = ResolvedRSHParameters(
        sr_hf_fraction=jnp.asarray(0.20),
        lr_hf_fraction=jnp.asarray(0.65),
        omega=jnp.asarray(0.33),
    )

    assert np.isclose(float(params.paper_alpha), 0.20)
    assert np.isclose(float(params.paper_beta), 0.45)
    omega, alpha, beta = params.to_pyscf_rsh()
    assert np.isclose(float(omega), 0.33)
    assert np.isclose(float(alpha), 0.65)
    assert np.isclose(float(beta), -0.45)
    omega2, alpha2, hyb2 = params.to_pyscf_rsh_and_hybrid()
    assert np.isclose(float(omega2), 0.33)
    assert np.isclose(float(alpha2), 0.65)
    assert np.isclose(float(hyb2), 0.20)


def test_make_pyscf_rsh_spec_uses_short_range_fraction_as_hyb():
    params = ResolvedRSHParameters(
        sr_hf_fraction=jnp.asarray(0.15),
        lr_hf_fraction=jnp.asarray(0.70),
        omega=jnp.asarray(0.40),
    )

    spec = make_pyscf_rsh_spec(
        xc_description="PBE,PBE",
        xctype="GGA",
        resolved_params=params,
    )

    assert spec.xc_description == "PBE,PBE"
    assert spec.xctype == "GGA"
    assert np.isclose(spec.hyb, 0.15)
    assert np.allclose(spec.rsh, (0.40, 0.70, -0.55))
    assert np.allclose(spec.expected_rsh_and_hybrid_coeff(), (0.40, 0.70, 0.15))


def test_scf_xc_contributions_normalize_exchange_channels():
    contributions = SCFXCContributions(
        v_rho=jnp.ones((5,)),
        v_grad=jnp.zeros((5, 3)),
        xc_kind="GGA",
        full_hf_fraction=0.25,
        lr_hf_omegas=jnp.asarray([[0.3, 0.5]]),
        lr_hf_coefficients=jnp.asarray([[0.2, 0.1]]),
    )

    assert contributions.lr_hf_omegas is not None
    assert contributions.lr_hf_coefficients is not None
    assert contributions.lr_hf_omegas.shape == (2,)
    assert contributions.lr_hf_coefficients.shape == (2,)
    assert np.isclose(float(contributions.exact_exchange_fraction), 0.25)


def test_pyscf_rsh_spec_installs_explicit_hyb_and_rsh_into_numint():
    pytest.importorskip("pyscf")
    from pyscf import dft, gto

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = dft.RKS(mol)
    params = ResolvedRSHParameters(
        sr_hf_fraction=jnp.asarray(0.19),
        lr_hf_fraction=jnp.asarray(0.65),
        omega=jnp.asarray(0.33),
    )
    spec = make_pyscf_rsh_spec(
        xc_description="B88,LYP",
        xctype="GGA",
        resolved_params=params,
    )

    spec.install_into_mf(mf)

    assert np.isclose(mf._numint.hybrid_coeff(mf.xc, spin=mol.spin), 0.19)
    assert np.allclose(mf._numint.rsh_coeff(mf.xc), (0.33, 0.65, -0.46))
    assert np.allclose(
        mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin),
        spec.expected_rsh_and_hybrid_coeff(),
    )
    assert mf._numint._xc_type(mf.xc) == "GGA"
