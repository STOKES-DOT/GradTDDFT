import jax.numpy as jnp

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
