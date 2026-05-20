import jax.numpy as jnp
import pytest

import td_graddft.xc_backend.jax_xc_adapter as jax_xc_adapter
from td_graddft.xc_backend.jax_libxc import (
    LocalXCTermSpec,
    RestrictedFeatureBundle,
    eval_xc_energy_density,
    eval_xc_term_specs_energy_density,
)


def _features():
    rho = jnp.asarray([0.2, 0.4])
    sigma = jnp.asarray([0.01, 0.02])
    tau = jnp.asarray([0.05, 0.07])
    return RestrictedFeatureBundle(
        rho_a=0.5 * rho,
        rho_b=0.5 * rho,
        sigma_aa=0.25 * sigma,
        sigma_ab=0.25 * sigma,
        sigma_bb=0.25 * sigma,
        tau_a=0.5 * tau,
        tau_b=0.5 * tau,
    )


def test_load_jax_xc_requires_installed_jax_xc(monkeypatch):
    def missing_import(name):
        if name == "jax_xc":
            raise ModuleNotFoundError("No module named 'jax_xc'")
        raise AssertionError(f"unexpected import {name!r}")

    monkeypatch.setattr(jax_xc_adapter.importlib, "import_module", missing_import)

    with pytest.raises(jax_xc_adapter.MissingJAXXCError):
        jax_xc_adapter.load_jax_xc()


def test_eval_xc_energy_density_routes_builtin_terms_through_adapter(monkeypatch):
    features = _features()
    calls = []

    def fake_eval(name, bundle, *, omega=None, allow_experimental_jax_xc=False):
        calls.append((name, omega, allow_experimental_jax_xc))
        assert bundle is features
        factors = {"gga_x_pbe": 2.0, "gga_c_pbe": 3.0}
        return jnp.full_like(bundle.rho, factors[name])

    monkeypatch.setattr(
        jax_xc_adapter,
        "eval_jax_xc_from_restricted_features",
        fake_eval,
    )

    got = eval_xc_energy_density("pbe0", features)

    assert calls == [
        ("gga_x_pbe", None, False),
        ("gga_c_pbe", None, False),
    ]
    assert jnp.allclose(got, features.rho * (0.75 * 2.0 + 3.0))


def test_eval_xc_term_specs_passes_runtime_omega_to_adapter(monkeypatch):
    features = _features()
    calls = []

    def fake_eval(name, bundle, *, omega=None, allow_experimental_jax_xc=False):
        calls.append((name, omega, allow_experimental_jax_xc))
        assert bundle is features
        return jnp.full_like(bundle.rho, 1.0 if name == "gga_x_wpbeh" else 2.0)

    monkeypatch.setattr(
        jax_xc_adapter,
        "eval_jax_xc_from_restricted_features",
        fake_eval,
    )

    got = eval_xc_term_specs_energy_density(
        (
            LocalXCTermSpec("gga_x_wpbeh", 1.0, "runtime"),
            LocalXCTermSpec("gga_c_pbe", 1.0, "none"),
        ),
        features,
        omega=0.33,
    )

    assert calls == [
        ("gga_x_wpbeh", 0.33, False),
        ("gga_c_pbe", None, False),
    ]
    assert jnp.allclose(got, 3.0 * features.rho)
