import jax
import jax.numpy as jnp

import td_graddft.xc_backend.jax_xc_adapter as jax_xc_adapter
from td_graddft.xc_backend.jax_libxc import (
    LocalXCTermSpec,
    RestrictedFeatureBundle,
    b3lyp_component_basis,
    canonical_rsh_preset_name,
    eval_xc_energy_density,
    eval_xc_response_tensor,
    eval_xc_term_specs_energy_density,
    get_rsh_functional_preset,
    hybrid_coeff,
    jax_xc_backend_info,
    list_rsh_functional_presets,
    parse_xc,
    resolve_xc_component_name,
    resolve_semilocal_xc_specs,
    xc_type,
)


def _features():
    rho = jnp.asarray([0.5, 0.6])
    sigma = jnp.asarray([0.01, 0.02])
    tau = jnp.asarray([0.1, 0.2])
    return RestrictedFeatureBundle(
        rho_a=0.5 * rho,
        rho_b=0.5 * rho,
        sigma_aa=0.25 * sigma,
        sigma_ab=0.25 * sigma,
        sigma_bb=0.25 * sigma,
        tau_a=0.5 * tau,
        tau_b=0.5 * tau,
    )


def test_parse_xc_supports_pyscf_like_aliases():
    terms = parse_xc("pbe0")
    svwn_terms = parse_xc("svwn")

    assert hybrid_coeff("pbe0") == 0.25
    assert xc_type("pbe") == "GGA"
    assert [term.name for term in terms] == ["hf", "gga_x_pbe", "gga_c_pbe"]
    assert xc_type("b3lyp") == "GGA"
    assert [term.name for term in svwn_terms] == ["lda_x", "lda_c_vwn"]


def test_resolve_semilocal_xc_specs_expands_aliases():
    assert resolve_semilocal_xc_specs("pbe") == ("gga_x_pbe", "gga_c_pbe")
    assert resolve_semilocal_xc_specs("hyb_gga_xc_pbeh") == (
        "gga_x_pbe",
        "gga_c_pbe",
    )
    assert resolve_semilocal_xc_specs("hyb_gga_xc_b3lyp") == (
        "lda_x",
        "gga_x_b88",
        "lda_c_vwn_rpa",
        "gga_c_lyp",
    )
    assert resolve_semilocal_xc_specs(("lda_x", "gga_c_pbe")) == ("lda_x", "gga_c_pbe")


def test_resolve_xc_component_name_accepts_friendly_aliases():
    assert resolve_xc_component_name("lyp_c") == "gga_c_lyp"
    assert resolve_xc_component_name("C_LYP") == "gga_c_lyp"
    assert resolve_xc_component_name("pbe_x") == "gga_x_pbe"
    assert resolve_xc_component_name("x_b88") == "gga_x_b88"
    assert resolve_xc_component_name("mgga:scan_c") == "mgga_c_scan"
    assert resolve_semilocal_xc_specs(("lyp_c", "b88_x", "vwn_rpa_c")) == (
        "gga_c_lyp",
        "gga_x_b88",
        "lda_c_vwn_rpa",
    )


def test_resolve_xc_component_name_rejects_ambiguous_or_kinetic_names():
    try:
        resolve_xc_component_name("scan")
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("scan should be ambiguous")

    assert "ambiguous" in message
    assert "scan_x" in message
    assert "scan_c" in message

    try:
        resolve_xc_component_name("gga_k_tfvw")
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("kinetic component should be rejected")

    assert "kinetic" in message
    assert "not an XC" in message


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


def test_eval_xc_energy_density_routes_alias_terms_through_adapter(monkeypatch):
    features = _features()
    calls = []

    def fake_eval(name, bundle, *, omega=None, allow_experimental_jax_xc=False):
        calls.append((name, omega, allow_experimental_jax_xc))
        assert bundle is features
        factors = {
            "lda_x": 0.5,
            "gga_x_b88": 1.0,
            "lda_c_vwn_rpa": 1.5,
            "gga_c_lyp": 2.0,
        }
        return jnp.full_like(bundle.rho, factors[name])

    monkeypatch.setattr(jax_xc_adapter, "eval_jax_xc_from_restricted_features", fake_eval)

    got = eval_xc_energy_density("b3lyp", features)

    assert calls == [
        ("lda_x", None, False),
        ("gga_x_b88", None, False),
        ("lda_c_vwn_rpa", None, False),
        ("gga_c_lyp", None, False),
    ]
    expected_eps = 0.08 * 0.5 + 0.72 * 1.0 + 0.19 * 1.5 + 0.81 * 2.0
    assert jnp.allclose(got, features.rho * expected_eps)


def test_eval_xc_energy_density_accepts_dynamic_installed_component_with_opt_in(monkeypatch):
    class FakeModule:
        __version__ = "fake"

        @staticmethod
        def gga_c_demo(*, polarized=False):
            del polarized
            return lambda rho_fn, r, mo_fn=None: 2.0 * rho_fn(r)

    monkeypatch.setattr(
        jax_xc_adapter,
        "load_jax_xc",
        lambda: (jax_xc_adapter._SafeJAXXCModule(FakeModule()), "upstream"),
    )
    features = _features()

    try:
        eval_xc_energy_density("gga_c_demo", features)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("dynamic component should require experimental opt-in")

    assert "allow_experimental_jax_xc=True" in message

    got = eval_xc_energy_density(
        "gga_c_demo",
        features,
        allow_experimental_jax_xc=True,
    )

    assert jnp.allclose(got, 2.0 * features.rho * features.rho)


def test_eval_xc_term_specs_energy_density_passes_omega(monkeypatch):
    features = _features()
    calls = []

    def fake_eval(name, bundle, *, omega=None, allow_experimental_jax_xc=False):
        calls.append((name, omega, allow_experimental_jax_xc))
        assert bundle is features
        return jnp.full_like(bundle.rho, 1.0 if name == "gga_x_wpbeh" else 2.0)

    monkeypatch.setattr(jax_xc_adapter, "eval_jax_xc_from_restricted_features", fake_eval)

    got = eval_xc_term_specs_energy_density(
        (
            LocalXCTermSpec("gga_x_wpbeh", 1.0, "runtime"),
            LocalXCTermSpec("gga_c_pbe", 1.0, "none"),
        ),
        features,
        omega=0.4,
    )

    assert calls == [
        ("gga_x_wpbeh", 0.4, False),
        ("gga_c_pbe", None, False),
    ]
    assert jnp.allclose(got, 3.0 * features.rho)


def test_energy_density_remains_differentiable_through_adapter(monkeypatch):
    features = _features()

    def fake_eval(name, bundle, *, omega=None, allow_experimental_jax_xc=False):
        del name, omega, allow_experimental_jax_xc
        return bundle.rho + 0.5 * jnp.sqrt(bundle.sigma + 1e-12)

    monkeypatch.setattr(jax_xc_adapter, "eval_jax_xc_from_restricted_features", fake_eval)

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

    assert jnp.all(jnp.isfinite(grad))


def test_eval_xc_response_tensor_uses_adapter_energy(monkeypatch):
    def fake_eval(name, bundle, *, omega=None, allow_experimental_jax_xc=False):
        del name, allow_experimental_jax_xc
        omega_value = 0.0 if omega is None else omega
        return bundle.rho + 0.25 * bundle.sigma + 0.1 * omega_value

    monkeypatch.setattr(jax_xc_adapter, "eval_jax_xc_from_restricted_features", fake_eval)
    rho = jnp.asarray([0.4, 0.8])
    grad = jnp.asarray([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]])

    kind, tensor = eval_xc_response_tensor("gga_x_pbe", rho, grad=grad, omega=0.3)

    assert kind == "GGA"
    assert tensor.shape == (4, 4, rho.shape[0])
    assert jnp.all(jnp.isfinite(tensor))


def test_jax_xc_backend_info_reports_missing_without_raising(monkeypatch):
    monkeypatch.setattr(
        jax_xc_adapter,
        "load_jax_xc",
        lambda: (_ for _ in ()).throw(jax_xc_adapter.MissingJAXXCError("missing")),
    )

    info = jax_xc_backend_info()

    assert info["available"] is False
    assert info["backend"] == "missing"
