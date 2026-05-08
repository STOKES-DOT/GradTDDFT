from pathlib import Path

import jax.numpy as jnp
import pytest

from td_graddft.jax_libxc import (
    RestrictedFeatureBundle,
    eval_xc_energy_density,
    jax_xc_backend_info,
    resolve_semilocal_xc_specs,
)
from td_graddft.jax_xc_adapter import load_jax_xc
from td_graddft.xc_backend.vendor import vendored_jax_xc_info


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


def test_load_jax_xc_reports_external_vendored_or_fallback_backend():
    module, backend = load_jax_xc()

    assert module is not None
    assert backend in {"upstream", "vendored", "fallback"}


def test_load_jax_xc_falls_back_when_installed_backend_import_fails(monkeypatch):
    from td_graddft import jax_xc_adapter

    def broken_import(name):
        if name == "jax_xc":
            raise ImportError("installed jax_xc is not loadable")
        raise AssertionError(f"Unexpected import {name!r}")

    monkeypatch.setattr(jax_xc_adapter.importlib, "import_module", broken_import)

    module, backend = load_jax_xc()

    assert module is not None
    assert backend == "fallback"


def test_vendored_jax_xc_info_has_stable_shape_even_when_missing():
    info = vendored_jax_xc_info()

    assert isinstance(info.root, Path)
    assert isinstance(info.complete, bool)
    assert info.backend_label in {"vendored", "missing"}


def test_public_backend_info_reports_active_backend():
    info = jax_xc_backend_info()

    assert info["backend"] in {"upstream", "vendored", "fallback"}
    assert "module_version" in info
    assert "vendored_complete" in info


def test_resolve_semilocal_xc_specs_expands_alias_and_preserves_tuple_channels():
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
    assert resolve_semilocal_xc_specs("hyb_gga_xc_bhandhlyp") == (
        "gga_x_b88",
        "gga_c_lyp",
    )
    assert resolve_semilocal_xc_specs(("lda_x", "gga_c_pbe")) == ("lda_x", "gga_c_pbe")


def test_resolved_semilocal_specs_are_energy_channels():
    features = _features()
    channels = [
        eval_xc_energy_density(spec, features)
        for spec in resolve_semilocal_xc_specs(("gga_x_pbe", "gga_c_pbe"))
    ]

    assert len(channels) == 2
    assert all(channel.shape == features.rho.shape for channel in channels)


def test_eval_xc_energy_density_routes_experimental_jax_xc_with_opt_in(monkeypatch):
    from td_graddft import jax_xc_adapter

    class FakeModule:
        __version__ = "fake"

        @staticmethod
        def gga_x_rpbe(*, polarized=False):
            del polarized
            return lambda rho_fn, r, mo_fn=None: 3.0 * rho_fn(r)

    monkeypatch.setattr(
        jax_xc_adapter,
        "load_jax_xc",
        lambda: (jax_xc_adapter._SafeJAXXCModule(FakeModule()), "upstream"),
    )
    features = _features()

    with pytest.raises(ValueError, match="allow_experimental_jax_xc=True"):
        eval_xc_energy_density("gga_x_rpbe", features)

    got = eval_xc_energy_density(
        "gga_x_rpbe",
        features,
        allow_experimental_jax_xc=True,
    )

    assert got.shape == features.rho.shape
    assert jnp.allclose(got, 3.0 * features.rho * features.rho)


def test_b97_family_is_experimental_by_default(monkeypatch):
    from td_graddft import jax_xc_adapter

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

    with pytest.raises(ValueError, match="B97-family"):
        eval_xc_energy_density("hyb_gga_xc_b97", _features())
