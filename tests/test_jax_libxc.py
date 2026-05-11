import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf.dft import libxc as pyscf_libxc

from td_graddft.jax_libxc import (
    RestrictedFeatureBundle,
    b3lyp_component_basis,
    canonical_rsh_preset_name,
    eval_xc_energy_density,
    eval_xc_response_tensor,
    get_rsh_functional_preset,
    hybrid_coeff,
    list_rsh_functional_presets,
    parse_xc,
    xc_type,
)


def _features():
    return RestrictedFeatureBundle(
        rho_a=jnp.array([0.5, 0.6]),
        rho_b=jnp.array([0.5, 0.6]),
        sigma_aa=jnp.array([0.01, 0.02]),
        sigma_ab=jnp.array([0.01, 0.02]),
        sigma_bb=jnp.array([0.01, 0.02]),
        tau_a=jnp.array([0.1, 0.2]),
        tau_b=jnp.array([0.1, 0.2]),
    )


def test_parse_xc_supports_pyscf_like_aliases():
    terms = parse_xc("pbe0")
    svwn_terms = parse_xc("svwn")

    assert hybrid_coeff("pbe0") == 0.25
    assert xc_type("pbe") == "GGA"
    assert [term.name for term in terms] == ["hf", "gga_x_pbe", "gga_c_pbe"]
    assert xc_type("b3lyp") == "GGA"
    assert [term.name for term in svwn_terms] == ["lda_x", "lda_c_vwn"]


def test_b3lyp_component_basis_returns_explicit_channels():
    assert b3lyp_component_basis() == (
        "lda_x",
        "gga_x_b88",
        "lda_c_vwn_rpa",
        "gga_c_lyp",
    )


def test_jax_libxc_exposes_rsh_functional_presets_directly():
    names = list_rsh_functional_presets()
    lc = get_rsh_functional_preset("LC_WPBE")
    wb = get_rsh_functional_preset("wb97xd")

    assert "lc-wpbe" in names
    assert "wb97x-d" in names
    assert canonical_rsh_preset_name("omega_b97x_d") == "wb97x-d"
    assert lc.canonical_xc_name == "LC_WPBE"
    assert lc.default_sr_hf_fraction == 0.0
    assert lc.default_lr_hf_fraction == 1.0
    assert lc.default_omega == 0.4
    assert wb.canonical_xc_name == "WB97X_D"
    assert wb.default_sr_hf_fraction == 0.222036
    assert wb.default_lr_hf_fraction == 1.0
    assert wb.default_omega == 0.2


def test_lc_wpbe_generated_formula_records_jax_xc_source():
    from td_graddft import _jax_xc_wpbeh

    assert (
        _jax_xc_wpbeh.JAX_XC_WPBEH_SOURCE_PATH
        == "gen_repo/impl/prebuilt/unpol/gga_x_wpbeh.py"
    )
    assert _jax_xc_wpbeh.JAX_XC_WPBEH_SOURCE_COMMIT


def test_lc_wpbe_local_energy_density_matches_pyscf_libxc_restricted():
    pytest.importorskip("pyscf")

    rho = np.array([0.2, 0.4, 0.8])
    dx = np.array([0.03, 0.06, 0.10])
    zeros = np.zeros_like(rho)
    omega = 0.4
    features = RestrictedFeatureBundle(
        rho_a=jnp.asarray(0.5 * rho),
        rho_b=jnp.asarray(0.5 * rho),
        sigma_aa=jnp.asarray(0.25 * dx**2),
        sigma_ab=jnp.asarray(0.25 * dx**2),
        sigma_bb=jnp.asarray(0.25 * dx**2),
        tau_a=jnp.zeros_like(jnp.asarray(rho)),
        tau_b=jnp.zeros_like(jnp.asarray(rho)),
    )
    expected_exchange = rho * pyscf_libxc.eval_xc(
        "GGA_X_WPBEH",
        np.array([rho, dx, zeros, zeros]),
        spin=0,
        deriv=0,
        omega=omega,
    )[0]
    expected_correlation = rho * pyscf_libxc.eval_xc(
        "GGA_C_PBE",
        np.array([rho, dx, zeros, zeros]),
        spin=0,
        deriv=0,
    )[0]
    got = np.asarray(eval_xc_energy_density("lc_wpbe_local", features, omega=omega))

    assert np.allclose(got, expected_exchange + expected_correlation, atol=5e-6, rtol=5e-6)


def test_gga_x_wpbeh_matches_pyscf_libxc_for_large_reduced_gradient_point():
    pytest.importorskip("pyscf")

    rho_a = np.array([6.765271973563358e-05])
    rho_b = np.array([6.765271973563358e-05])
    grad_a = np.array(
        [[0.00014442681276705116, -0.0001162723929155618, -0.00017158087575808167]]
    )
    grad_b = grad_a.copy()
    omega = 0.8105142116546631
    features = RestrictedFeatureBundle(
        rho_a=jnp.asarray(rho_a),
        rho_b=jnp.asarray(rho_b),
        sigma_aa=jnp.asarray(np.sum(grad_a * grad_a, axis=1)),
        sigma_ab=jnp.asarray(np.sum(grad_a * grad_b, axis=1)),
        sigma_bb=jnp.asarray(np.sum(grad_b * grad_b, axis=1)),
        tau_a=jnp.zeros_like(jnp.asarray(rho_a)),
        tau_b=jnp.zeros_like(jnp.asarray(rho_b)),
    )
    expected = (rho_a + rho_b) * pyscf_libxc.eval_xc(
        "GGA_X_WPBEH",
        np.array(
            [
                [rho_a, grad_a[:, 0], grad_a[:, 1], grad_a[:, 2]],
                [rho_b, grad_b[:, 0], grad_b[:, 1], grad_b[:, 2]],
            ]
        ),
        spin=1,
        deriv=0,
        omega=omega,
    )[0]
    got = np.asarray(eval_xc_energy_density("gga_x_wpbeh", features, omega=omega))

    assert np.allclose(got, expected, atol=1e-8, rtol=1e-5)


def test_lc_wpbe_local_exchange_is_omega_dependent_and_differentiable():
    features = _features()
    low_omega = eval_xc_energy_density("gga_x_wpbeh", features, omega=0.2)
    high_omega = eval_xc_energy_density("gga_x_wpbeh", features, omega=0.6)
    grad = jax.grad(
        lambda omega: jnp.sum(eval_xc_energy_density("gga_x_wpbeh", features, omega=omega))
    )(jnp.asarray(0.4))

    assert not jnp.allclose(low_omega, high_omega)
    assert jnp.all(jnp.isfinite(low_omega))
    assert jnp.all(jnp.isfinite(high_omega))
    assert jnp.isfinite(grad)


def test_lc_wpbe_local_potential_kernel_is_differentiable_with_respect_to_omega():
    from td_graddft.nn_rsh.functional import _point_xc_value_and_grad_kernel

    variables = jnp.asarray(
        [
            [1.0, 0.1, 0.2, 0.3],
            [0.8, 0.2, 0.1, 0.0],
        ]
    )
    kernel = _point_xc_value_and_grad_kernel("lc_wpbe_local", "GGA", 1e-12)
    grad = jax.grad(
        lambda omega: jnp.sum(kernel(variables, omega)[0])
        + 0.01 * jnp.sum(kernel(variables, omega)[1])
    )(jnp.asarray(0.4))

    assert jnp.isfinite(grad)


def test_pbe_energy_density_is_finite_and_differentiable():
    features = _features()
    energy_density = eval_xc_energy_density("pbe", features)

    grad = jax.grad(
        lambda rho_a: jnp.sum(
            eval_xc_energy_density(
                "pbe",
                RestrictedFeatureBundle(
                    rho_a=rho_a,
                    rho_b=features.rho_b,
                    sigma_aa=features.sigma_aa,
                    sigma_ab=features.sigma_ab,
                    sigma_bb=features.sigma_bb,
                    tau_a=features.tau_a,
                    tau_b=features.tau_b,
                ),
            )
        )
    )(features.rho_a)

    assert energy_density.shape == (2,)
    assert jnp.all(jnp.isfinite(energy_density))
    assert jnp.all(jnp.isfinite(grad))


def test_dynamic_mgga_jax_xc_energy_and_response_tensor_require_opt_in(monkeypatch):
    from td_graddft import jax_xc_adapter

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
    features = _features()
    total_tau = features.tau_a + features.tau_b
    grad = jnp.stack(
        [
            jnp.sqrt(jnp.maximum(features.sigma, 0.0)),
            jnp.zeros_like(features.rho),
            jnp.zeros_like(features.rho),
        ],
        axis=-1,
    )

    with pytest.raises(ValueError, match="allow_experimental_jax_xc=True"):
        eval_xc_energy_density("mgga_x_demo", features)

    got = eval_xc_energy_density(
        "mgga_x_demo",
        features,
        allow_experimental_jax_xc=True,
    )
    kind, tensor = eval_xc_response_tensor(
        "mgga_x_demo",
        features.rho,
        grad=grad,
        tau=total_tau,
        allow_experimental_jax_xc=True,
    )

    assert jnp.allclose(got, features.rho * (features.rho + 0.25 * total_tau))
    assert kind == "MGGA"
    assert tensor.shape == (5, 5, features.rho.shape[0])
    assert jnp.all(jnp.isfinite(tensor))


def test_b3lyp_energy_density_is_finite():
    features = _features()
    energy_density = eval_xc_energy_density("b3lyp", features)

    assert energy_density.shape == (2,)
    assert jnp.all(jnp.isfinite(energy_density))


def test_lda_c_vwn_energy_density_matches_pyscf_libxc():
    rho = np.array([0.2, 0.4, 0.8])
    features = RestrictedFeatureBundle(
        rho_a=jnp.asarray(0.5 * rho),
        rho_b=jnp.asarray(0.5 * rho),
        sigma_aa=jnp.zeros_like(jnp.asarray(rho)),
        sigma_ab=jnp.zeros_like(jnp.asarray(rho)),
        sigma_bb=jnp.zeros_like(jnp.asarray(rho)),
        tau_a=jnp.zeros_like(jnp.asarray(rho)),
        tau_b=jnp.zeros_like(jnp.asarray(rho)),
    )
    expected = rho * pyscf_libxc.eval_xc("LDA_C_VWN", rho, spin=0, deriv=0)[0]
    got = np.asarray(eval_xc_energy_density("lda_c_vwn", features))

    assert np.allclose(got, expected, atol=1e-7, rtol=1e-7)


def test_lda_c_vwn_rpa_energy_density_matches_pyscf_libxc():
    rho = np.array([0.2, 0.4, 0.8])
    features = RestrictedFeatureBundle(
        rho_a=jnp.asarray(0.5 * rho),
        rho_b=jnp.asarray(0.5 * rho),
        sigma_aa=jnp.zeros_like(jnp.asarray(rho)),
        sigma_ab=jnp.zeros_like(jnp.asarray(rho)),
        sigma_bb=jnp.zeros_like(jnp.asarray(rho)),
        tau_a=jnp.zeros_like(jnp.asarray(rho)),
        tau_b=jnp.zeros_like(jnp.asarray(rho)),
    )
    expected = rho * pyscf_libxc.eval_xc("LDA_C_VWN_RPA", rho, spin=0, deriv=0)[0]
    got = np.asarray(eval_xc_energy_density("lda_c_vwn_rpa", features))

    assert np.allclose(got, expected, atol=1e-7, rtol=1e-7)


def test_gga_c_lyp_energy_density_matches_pyscf_libxc_restricted():
    rho = np.array([0.2, 0.4, 0.8])
    dx = np.array([0.1, 0.12, 0.15])
    zeros = np.zeros_like(rho)
    features = RestrictedFeatureBundle(
        rho_a=jnp.asarray(0.5 * rho),
        rho_b=jnp.asarray(0.5 * rho),
        sigma_aa=jnp.asarray(0.25 * dx**2),
        sigma_ab=jnp.asarray(0.25 * dx**2),
        sigma_bb=jnp.asarray(0.25 * dx**2),
        tau_a=jnp.zeros_like(jnp.asarray(rho)),
        tau_b=jnp.zeros_like(jnp.asarray(rho)),
    )
    expected = rho * pyscf_libxc.eval_xc(
        "GGA_C_LYP",
        np.array([rho, dx, zeros, zeros]),
        spin=0,
        deriv=0,
    )[0]
    got = np.asarray(eval_xc_energy_density("gga_c_lyp", features))

    assert np.allclose(got, expected, atol=1e-7, rtol=1e-7)


def test_gga_c_lyp_energy_density_matches_pyscf_libxc_unrestricted():
    rho_a = np.array([0.10, 0.20, 0.30])
    rho_b = np.array([0.06, 0.12, 0.18])
    dxa = np.array([0.02, 0.03, 0.05])
    dxb = np.array([0.01, 0.025, 0.035])
    zeros = np.zeros_like(rho_a)
    features = RestrictedFeatureBundle(
        rho_a=jnp.asarray(rho_a),
        rho_b=jnp.asarray(rho_b),
        sigma_aa=jnp.asarray(dxa**2),
        sigma_ab=jnp.asarray(dxa * dxb),
        sigma_bb=jnp.asarray(dxb**2),
        tau_a=jnp.zeros_like(jnp.asarray(rho_a)),
        tau_b=jnp.zeros_like(jnp.asarray(rho_b)),
    )
    expected = (rho_a + rho_b) * pyscf_libxc.eval_xc(
        "GGA_C_LYP",
        np.array([[rho_a, dxa, zeros, zeros], [rho_b, dxb, zeros, zeros]]),
        spin=1,
        deriv=0,
    )[0]
    got = np.asarray(eval_xc_energy_density("gga_c_lyp", features))

    assert np.allclose(got, expected, atol=1e-7, rtol=1e-7)


def test_lda_x_energy_density_matches_spin_resolved_local_contribution():
    features = _features()
    energy_density = eval_xc_energy_density("lda_x", features)
    expected = (
        -(3.0 / 2.0)
        * (3.0 / (4.0 * jnp.pi)) ** (1.0 / 3.0)
        * (jnp.power(features.rho_a, 4.0 / 3.0) + jnp.power(features.rho_b, 4.0 / 3.0))
    )

    assert jnp.allclose(energy_density, expected, atol=1e-12, rtol=1e-12)
