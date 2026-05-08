import jax
import jax.numpy as jnp
import pytest

from td_graddft.jax_xc_adapter import load_jax_xc
from td_graddft.upstreams import has_jax_xc
from td_graddft.xc import lda_from_jax_xc


def test_load_jax_xc_backend_exposes_lda_factory():
    module, backend = load_jax_xc()
    assert backend in ("upstream", "fallback")
    lda_x = getattr(module, "lda_x")
    functional = lda_x(polarized=False)
    value = functional(lambda _r: jnp.asarray(0.5), jnp.zeros(3))
    assert jnp.isfinite(value)


def test_lda_from_jax_xc_returns_finite_energy_density():
    functional = lda_from_jax_xc("lda_x")
    rho = jnp.asarray([0.05, 0.2, 0.8])
    eps = functional.energy_density(rho)
    assert eps.shape == rho.shape
    assert bool(jnp.all(jnp.isfinite(eps)))


def test_has_jax_xc_enabled_via_adapter():
    assert has_jax_xc()


def test_load_jax_xc_wraps_known_broken_hybrid_composite(monkeypatch):
    from td_graddft import jax_xc_adapter

    def factory(value):
        def _factory(*, polarized=False, **params):
            assert not polarized

            def functional(rho_fn, r, mo_fn=None):
                del rho_fn, r, mo_fn
                if callable(value):
                    return value(params)
                return value

            return functional

        return _factory

    class BrokenHybridModule:
        __version__ = "fake"
        gga_x_pbe = staticmethod(factory(2.0))
        gga_c_pbe = staticmethod(factory(0.5))
        gga_x_wpbeh = staticmethod(
            factory(lambda params: 2.0 if params.get("_omega") == 0.0 else 0.4)
        )
        gga_x_b88 = staticmethod(factory(1.0))
        gga_x_ityh = staticmethod(
            factory(lambda params: 2.0 if params.get("_omega") == 0.33 else -20.0)
        )
        lda_c_vwn = staticmethod(factory(3.0))
        gga_c_lyp = staticmethod(factory(4.0))
        hyb_gga_xc_pbeh = staticmethod(factory(-99.0))
        hyb_gga_xc_hse06 = staticmethod(factory(-99.0))
        hyb_gga_xc_cam_b3lyp = staticmethod(factory(-99.0))

    monkeypatch.setattr(
        jax_xc_adapter.importlib,
        "import_module",
        lambda name: BrokenHybridModule() if name == "jax_xc" else None,
    )

    module, backend = load_jax_xc()
    pbeh = module.hyb_gga_xc_pbeh(polarized=False)
    hse06 = module.hyb_gga_xc_hse06(polarized=False)
    cam_b3lyp = module.hyb_gga_xc_cam_b3lyp(polarized=False)

    assert backend == "upstream"
    assert pbeh(lambda _r: 0.2, None) == 0.75 * 2.0 + 0.5
    assert hse06(lambda _r: 0.2, None) == 2.0 - 0.25 * 0.4 + 0.5
    assert cam_b3lyp(lambda _r: 0.2, None) == 0.35 * 1.0 + 0.46 * 2.0 + 0.19 * 3.0 + 0.81 * 4.0


def test_jax_xc_functional_info_classifies_strict_wrapped_and_experimental(monkeypatch):
    from td_graddft import jax_xc_adapter

    class FakeModule:
        __version__ = "fake"

        @staticmethod
        def gga_x_rpbe(*, polarized=False):
            del polarized
            return lambda rho_fn, r, mo_fn=None: rho_fn(r) * 0.0

        @staticmethod
        def hyb_gga_xc_b97(*, polarized=False):
            del polarized
            return lambda rho_fn, r, mo_fn=None: rho_fn(r) * 0.0

    monkeypatch.setattr(
        jax_xc_adapter,
        "load_jax_xc",
        lambda: (jax_xc_adapter._SafeJAXXCModule(FakeModule()), "upstream"),
    )

    strict = jax_xc_adapter.jax_xc_functional_info("gga_x_pbe")
    wrapped = jax_xc_adapter.jax_xc_functional_info("hyb_gga_xc_pbeh")
    rpbe = jax_xc_adapter.jax_xc_functional_info("gga_x_rpbe")
    b97 = jax_xc_adapter.jax_xc_functional_info("hyb_gga_xc_b97")
    missing = jax_xc_adapter.jax_xc_functional_info("gga_x_not_real")

    assert strict.status == "strict"
    assert wrapped.status == "wrapped"
    assert wrapped.children
    assert rpbe.status == "experimental"
    assert b97.status == "experimental"
    assert "B97" in b97.reason
    assert missing.status == "unavailable"


def test_list_jax_xc_functionals_can_filter_by_status(monkeypatch):
    from td_graddft import jax_xc_adapter

    class FakeModule:
        __version__ = "fake"

        @staticmethod
        def gga_x_rpbe(*, polarized=False):
            del polarized
            return lambda rho_fn, r, mo_fn=None: rho_fn(r) * 0.0

    monkeypatch.setattr(
        jax_xc_adapter,
        "load_jax_xc",
        lambda: (jax_xc_adapter._SafeJAXXCModule(FakeModule()), "upstream"),
    )

    experimental = jax_xc_adapter.list_jax_xc_functionals(status="experimental")

    assert "gga_x_rpbe" in {info.name for info in experimental}
    assert all(info.status == "experimental" for info in experimental)


def test_jax_xc_functional_info_classifies_active_mgga_names_dynamically(monkeypatch):
    from td_graddft import jax_xc_adapter

    class FakeModule:
        __version__ = "fake"

        @staticmethod
        def mgga_x_demo(*, polarized=False):
            del polarized
            return lambda rho_fn, r, mo_fn=None: rho_fn(r) * 0.0

        @staticmethod
        def hyb_mgga_xc_demo(*, polarized=False):
            del polarized
            return lambda rho_fn, r, mo_fn=None: rho_fn(r) * 0.0

    monkeypatch.setattr(
        jax_xc_adapter,
        "load_jax_xc",
        lambda: (jax_xc_adapter._SafeJAXXCModule(FakeModule()), "upstream"),
    )

    mgga = jax_xc_adapter.jax_xc_functional_info("mgga_x_demo")
    hybrid_mgga = jax_xc_adapter.jax_xc_functional_info("hyb_mgga_xc_demo")
    experimental = jax_xc_adapter.list_jax_xc_functionals(status="experimental")

    assert mgga.status == "experimental"
    assert mgga.family == "MGGA"
    assert "MGGA" in mgga.reason
    assert hybrid_mgga.status == "experimental"
    assert hybrid_mgga.family == "MGGA"
    assert {"mgga_x_demo", "hyb_mgga_xc_demo"} <= {info.name for info in experimental}


def test_eval_jax_xc_from_restricted_features_requires_experimental_opt_in(monkeypatch):
    from td_graddft import jax_xc_adapter
    from td_graddft.jax_libxc import RestrictedFeatureBundle

    class FakeModule:
        __version__ = "fake"

        @staticmethod
        def gga_x_rpbe(*, polarized=False):
            del polarized
            return lambda rho_fn, r, mo_fn=None: 2.0 * rho_fn(r)

    monkeypatch.setattr(
        jax_xc_adapter,
        "load_jax_xc",
        lambda: (jax_xc_adapter._SafeJAXXCModule(FakeModule()), "upstream"),
    )
    features = RestrictedFeatureBundle(
        rho_a=jnp.asarray([0.1, 0.2]),
        rho_b=jnp.asarray([0.1, 0.2]),
        sigma_aa=jnp.asarray([0.01, 0.02]),
        sigma_ab=jnp.asarray([0.01, 0.02]),
        sigma_bb=jnp.asarray([0.01, 0.02]),
        tau_a=jnp.zeros((2,)),
        tau_b=jnp.zeros((2,)),
    )

    with pytest.raises(ValueError, match="allow_experimental_jax_xc=True"):
        jax_xc_adapter.eval_jax_xc_from_restricted_features("gga_x_rpbe", features)

    eps = jax_xc_adapter.eval_jax_xc_from_restricted_features(
        "gga_x_rpbe",
        features,
        allow_experimental_jax_xc=True,
    )

    assert eps.shape == features.rho.shape
    assert jnp.allclose(eps, 2.0 * features.rho)


def test_eval_jax_xc_from_restricted_features_passes_mgga_mo_fn_and_tau(monkeypatch):
    from td_graddft import jax_xc_adapter
    from td_graddft.jax_libxc import RestrictedFeatureBundle

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
    features = RestrictedFeatureBundle(
        rho_a=jnp.asarray([0.1, 0.2]),
        rho_b=jnp.asarray([0.1, 0.2]),
        sigma_aa=jnp.asarray([0.01, 0.02]),
        sigma_ab=jnp.asarray([0.01, 0.02]),
        sigma_bb=jnp.asarray([0.01, 0.02]),
        tau_a=jnp.asarray([0.15, 0.25]),
        tau_b=jnp.asarray([0.15, 0.25]),
    )

    with pytest.raises(ValueError, match="allow_experimental_jax_xc=True"):
        jax_xc_adapter.eval_jax_xc_from_restricted_features("mgga_x_demo", features)

    eps = jax_xc_adapter.eval_jax_xc_from_restricted_features(
        "mgga_x_demo",
        features,
        allow_experimental_jax_xc=True,
    )

    assert eps.shape == features.rho.shape
    assert jnp.allclose(eps, features.rho + 0.25 * (features.tau_a + features.tau_b))
