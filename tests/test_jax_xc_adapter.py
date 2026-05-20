import jax
import jax.numpy as jnp
import pytest

from td_graddft import upstreams
from td_graddft.xc_backend.jax_libxc import RestrictedFeatureBundle
from td_graddft.xc_backend.jax_xc_adapter import load_jax_xc
from td_graddft.xc import lda_from_jax_xc


def _features():
    return RestrictedFeatureBundle(
        rho_a=jnp.asarray([0.1, 0.2]),
        rho_b=jnp.asarray([0.1, 0.2]),
        sigma_aa=jnp.asarray([0.01, 0.02]),
        sigma_ab=jnp.asarray([0.01, 0.02]),
        sigma_bb=jnp.asarray([0.01, 0.02]),
        tau_a=jnp.asarray([0.15, 0.25]),
        tau_b=jnp.asarray([0.15, 0.25]),
    )


def _factory(value):
    def factory(*, polarized=False, **params):
        assert not polarized

        def functional(rho_fn, r, mo_fn=None):
            if callable(value):
                return value(rho_fn, r, mo_fn, params)
            return value

        return functional

    return factory


def test_load_jax_xc_backend_exposes_installed_factory(monkeypatch):
    class FakeModule:
        __version__ = "fake"
        lda_x = staticmethod(_factory(1.0))

    monkeypatch.setattr(
        "td_graddft.xc_backend.jax_xc_adapter.importlib.import_module",
        lambda name: FakeModule() if name == "jax_xc" else None,
    )

    module, backend = load_jax_xc()
    lda_x = getattr(module, "lda_x")
    functional = lda_x(polarized=False)
    value = functional(lambda _r: jnp.asarray(0.5), jnp.zeros(3))

    assert backend == "upstream"
    assert jnp.isfinite(value)


def test_load_jax_xc_raises_when_installed_backend_is_missing(monkeypatch):
    import td_graddft.xc_backend.jax_xc_adapter as jax_xc_adapter

    def missing_import(name):
        if name == "jax_xc":
            raise ModuleNotFoundError("No module named 'jax_xc'")
        raise AssertionError(f"Unexpected import {name!r}")

    monkeypatch.setattr(jax_xc_adapter.importlib, "import_module", missing_import)

    with pytest.raises(jax_xc_adapter.MissingJAXXCError):
        load_jax_xc()


def test_lda_from_jax_xc_returns_finite_energy_density(monkeypatch):
    class FakeModule:
        __version__ = "fake"

        @staticmethod
        def lda_x(*, polarized=False):
            assert not polarized

            def functional(rho_fn, r):
                return 2.0 * rho_fn(r)

            return functional

    monkeypatch.setattr(
        "td_graddft.xc_backend.jax_xc_adapter.importlib.import_module",
        lambda name: FakeModule() if name == "jax_xc" else None,
    )

    functional = lda_from_jax_xc("lda_x")
    rho = jnp.asarray([0.05, 0.2, 0.8])
    eps = functional.energy_density(rho)

    assert eps.shape == rho.shape
    assert bool(jnp.all(jnp.isfinite(eps)))


def test_has_jax_xc_false_when_adapter_cannot_load(monkeypatch):
    import td_graddft.xc_backend.jax_xc_adapter as jax_xc_adapter

    monkeypatch.setattr(
        upstreams,
        "load_jax_xc",
        lambda: (_ for _ in ()).throw(jax_xc_adapter.MissingJAXXCError("missing")),
    )

    assert not upstreams.has_jax_xc()


def test_load_jax_xc_wraps_known_hybrid_composites(monkeypatch):
    import td_graddft.xc_backend.jax_xc_adapter as jax_xc_adapter

    class BrokenHybridModule:
        __version__ = "fake"
        gga_x_pbe = staticmethod(_factory(2.0))
        gga_c_pbe = staticmethod(_factory(0.5))
        gga_x_wpbeh = staticmethod(
            _factory(lambda rho_fn, r, mo_fn, params: 2.0 if params.get("_omega") == 0.0 else 0.4)
        )
        hyb_gga_xc_pbeh = staticmethod(_factory(-99.0))
        hyb_gga_xc_hse06 = staticmethod(_factory(-99.0))

    monkeypatch.setattr(
        jax_xc_adapter.importlib,
        "import_module",
        lambda name: BrokenHybridModule() if name == "jax_xc" else None,
    )

    module, backend = load_jax_xc()
    pbeh = module.hyb_gga_xc_pbeh(polarized=False)
    hse06 = module.hyb_gga_xc_hse06(polarized=False)

    assert backend == "upstream"
    assert pbeh(lambda _r: 0.2, None) == 0.75 * 2.0 + 0.5
    assert hse06(lambda _r: 0.2, None) == 2.0 - 0.25 * 0.4 + 0.5


def test_jax_xc_functional_info_classifies_strict_wrapped_and_experimental(monkeypatch):
    import td_graddft.xc_backend.jax_xc_adapter as jax_xc_adapter

    class FakeModule:
        __version__ = "fake"
        gga_x_rpbe = staticmethod(_factory(0.0))
        hyb_gga_xc_b97 = staticmethod(_factory(0.0))

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
    import td_graddft.xc_backend.jax_xc_adapter as jax_xc_adapter

    class FakeModule:
        __version__ = "fake"
        gga_x_rpbe = staticmethod(_factory(0.0))

    monkeypatch.setattr(
        jax_xc_adapter,
        "load_jax_xc",
        lambda: (jax_xc_adapter._SafeJAXXCModule(FakeModule()), "upstream"),
    )

    experimental = jax_xc_adapter.list_jax_xc_functionals(status="experimental")

    assert "gga_x_rpbe" in {info.name for info in experimental}
    assert all(info.status == "experimental" for info in experimental)


def test_jax_xc_functional_info_classifies_active_mgga_names_dynamically(monkeypatch):
    import td_graddft.xc_backend.jax_xc_adapter as jax_xc_adapter

    class FakeModule:
        __version__ = "fake"
        mgga_x_demo = staticmethod(_factory(0.0))
        hyb_mgga_xc_demo = staticmethod(_factory(0.0))

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


def test_jax_xc_functional_info_discovers_active_lda_gga_names_dynamically(monkeypatch):
    import td_graddft.xc_backend.jax_xc_adapter as jax_xc_adapter

    class FakeModule:
        __version__ = "fake"
        lda_c_demo = staticmethod(_factory(0.0))
        gga_c_demo = staticmethod(_factory(0.0))
        gga_k_demo = staticmethod(_factory(0.0))

    monkeypatch.setattr(
        jax_xc_adapter,
        "load_jax_xc",
        lambda: (jax_xc_adapter._SafeJAXXCModule(FakeModule()), "upstream"),
    )

    lda = jax_xc_adapter.jax_xc_functional_info("lda_c_demo")
    gga = jax_xc_adapter.jax_xc_functional_info("gga_c_demo")
    kinetic = jax_xc_adapter.jax_xc_functional_info("gga_k_demo")
    experimental = jax_xc_adapter.list_jax_xc_functionals(status="experimental")
    unavailable = jax_xc_adapter.list_jax_xc_functionals(status="unavailable")

    assert lda.status == "experimental"
    assert lda.family == "LDA"
    assert gga.status == "experimental"
    assert gga.family == "GGA"
    assert kinetic.status == "unavailable"
    assert "kinetic" in kinetic.reason
    assert {"lda_c_demo", "gga_c_demo"} <= {info.name for info in experimental}
    assert "gga_k_demo" in {info.name for info in unavailable}


def test_jax_libxc_and_adapter_share_functional_metadata_surface(monkeypatch):
    import td_graddft.xc_backend.jax_libxc as jax_libxc
    import td_graddft.xc_backend.jax_xc_adapter as jax_xc_adapter

    class FakeModule:
        __version__ = "fake"
        gga_x_rpbe = staticmethod(_factory(0.0))

    monkeypatch.setattr(
        jax_xc_adapter,
        "load_jax_xc",
        lambda: (jax_xc_adapter._SafeJAXXCModule(FakeModule()), "upstream"),
    )

    adapter_info = jax_xc_adapter.jax_xc_functional_info("gga_x_rpbe")
    libxc_info = jax_libxc.jax_xc_functional_info("gga_x_rpbe")
    adapter_names = tuple(
        info.name for info in jax_xc_adapter.list_jax_xc_functionals(status="experimental")
    )
    libxc_names = tuple(
        info.name for info in jax_libxc.list_jax_xc_functionals(status="experimental")
    )

    assert jax_xc_adapter.JAXXCFunctionalInfo is jax_libxc.JAXXCFunctionalInfo
    assert jax_xc_adapter.JAXXCStatus == jax_libxc.JAXXCStatus
    assert adapter_info == libxc_info
    assert adapter_names == libxc_names


def test_eval_jax_xc_from_restricted_features_passes_runtime_omega(monkeypatch):
    import td_graddft.xc_backend.jax_xc_adapter as jax_xc_adapter

    seen = []

    class FakeModule:
        __version__ = "fake"

        @staticmethod
        def gga_x_wpbeh(*, polarized=False, **params):
            assert not polarized
            seen.append(params)

            def functional(rho_fn, r, mo_fn=None):
                del mo_fn
                return rho_fn(r) + params["_omega"]

            return functional

    monkeypatch.setattr(
        jax_xc_adapter,
        "load_jax_xc",
        lambda: (jax_xc_adapter._SafeJAXXCModule(FakeModule()), "upstream"),
    )

    eps = jax_xc_adapter.eval_jax_xc_from_restricted_features(
        "gga_x_wpbeh",
        _features(),
        omega=0.33,
    )

    assert seen == [{"_omega": 0.33}]
    assert jnp.allclose(eps, _features().rho + 0.33)


def test_eval_jax_xc_from_restricted_features_requires_experimental_opt_in(monkeypatch):
    import td_graddft.xc_backend.jax_xc_adapter as jax_xc_adapter

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
    features = _features()

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
    features = _features()

    with pytest.raises(ValueError, match="allow_experimental_jax_xc=True"):
        jax_xc_adapter.eval_jax_xc_from_restricted_features("mgga_x_demo", features)

    eps = jax_xc_adapter.eval_jax_xc_from_restricted_features(
        "mgga_x_demo",
        features,
        allow_experimental_jax_xc=True,
    )

    assert eps.shape == features.rho.shape
    assert jnp.allclose(eps, features.rho + 0.25 * (features.tau_a + features.tau_b))
