from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from .jax_libxc import (
    JAXXCFunctionalInfo,
    JAXXCStatus,
    RestrictedFeatureBundle,
    SAFE_JAX_XC_WRAPPED_COMPOSITES,
    _eval_xc_per_particle,
    jax_xc_functional_info,
    list_jax_xc_functionals,
)
from .xc_backend.vendor import vendored_jax_xc_info

_JAX_XC_IMPORT_ERRORS = (ImportError, OSError)


def _raise_if_not_allowed(
    info: JAXXCFunctionalInfo,
    *,
    allow_experimental_jax_xc: bool,
) -> None:
    if info.status == "unavailable":
        raise KeyError(f"jax_xc functional {info.name!r} is unavailable: {info.reason}")
    if info.status == "experimental" and not bool(allow_experimental_jax_xc):
        raise ValueError(
            f"jax_xc functional {info.name!r} is experimental: {info.reason} "
            "Pass allow_experimental_jax_xc=True to evaluate it."
        )


def eval_jax_xc_from_restricted_features(
    name: str,
    features: RestrictedFeatureBundle,
    *,
    allow_experimental_jax_xc: bool = False,
) -> jnp.ndarray:
    info = jax_xc_functional_info(name)
    _raise_if_not_allowed(info, allow_experimental_jax_xc=allow_experimental_jax_xc)
    if info.status == "strict":
        return _eval_xc_per_particle(info.name, features)

    module, _ = load_jax_xc()
    factory = getattr(module, info.name)
    functional = factory(polarized=False)
    rho = jnp.maximum(jnp.asarray(features.rho), 1e-12)
    sigma = jnp.maximum(jnp.asarray(features.sigma), 0.0)
    grad_mag = jnp.sqrt(sigma)
    tau = jnp.maximum(jnp.asarray(features.tau_a + features.tau_b), 0.0)
    origin = jnp.zeros((3,), dtype=rho.dtype)

    def point_eval(rho_value, grad_value, tau_value):
        def rho_fn(r):
            return rho_value + grad_value * r[0]

        def mo_fn(r):
            mo_grad = jnp.sqrt(jnp.maximum(2.0 * tau_value, 1e-30))
            return jnp.asarray([mo_grad * r[0]], dtype=rho.dtype)

        if info.family == "MGGA":
            value = functional(rho_fn, origin, mo_fn)
        else:
            value = functional(rho_fn, origin)
        if isinstance(value, tuple):
            value = value[0]
        return jnp.asarray(value, dtype=rho.dtype)

    if rho.ndim == 0:
        value = point_eval(rho, grad_mag, tau)
    else:
        value = jax.vmap(point_eval)(rho, grad_mag, tau)
    return jnp.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_import_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def _vendored_generated_path() -> Path:
    return vendored_jax_xc_info().root / "generated"


def _coerce_spin_density(rho_value: Any) -> tuple[Any, Any]:
    rho = jnp.asarray(rho_value)
    if rho.ndim > 0 and rho.shape[-1] == 2:
        return rho[..., 0], rho[..., 1]
    return 0.5 * rho, 0.5 * rho


class _FallbackJAXXC:
    """Small functional subset compatible with TD-GradDFT's jax_xc usage."""

    _MAPPING = {
        "lda_x": "lda_x",
        "lda_c_pw": "lda_c_pw",
        "lda_c_vwn": "lda_c_vwn",
        "lda_c_vwn_rpa": "lda_c_vwn_rpa",
        "gga_x_b88": "gga_x_b88",
        "gga_x_pbe": "gga_x_pbe",
        "gga_x_wpbeh": "gga_x_wpbeh",
        "gga_c_lyp": "gga_c_lyp",
        "gga_c_pbe": "gga_c_pbe",
        "lda": "lda",
        "svwn": "svwn",
        "svwn_rpa": "svwn_rpa",
        "pbe": "pbe",
        "pbe0": "pbe0",
        "b3lyp": "b3lyp",
        "lc_wpbe_local": "lc_wpbe_local",
    }

    __version__ = "td_graddft_fallback"

    def __getattr__(self, name: str):
        if name not in self._MAPPING:
            raise AttributeError(f"Fallback jax_xc does not expose functional '{name}'.")
        spec = self._MAPPING[name]

        def factory(*, polarized: bool = False):
            if polarized:
                raise NotImplementedError(
                    "TD-GradDFT fallback jax_xc currently supports only polarized=False."
                )

            def functional(rho_fn, r, mo_fn=None):
                del mo_fn
                rho_a, rho_b = _coerce_spin_density(rho_fn(r))
                zeros = jnp.zeros_like(rho_a)
                features = RestrictedFeatureBundle(
                    rho_a=rho_a,
                    rho_b=rho_b,
                    sigma_aa=zeros,
                    sigma_ab=zeros,
                    sigma_bb=zeros,
                    tau_a=zeros,
                    tau_b=zeros,
                )
                return _eval_xc_per_particle(spec, features)

            return functional

        return factory


class _SafeJAXXCModule:
    """Proxy an upstream jax_xc module while fixing known hybrid mix nodes.

    jax_xc 0.0.9 can expose correct child semilocal functionals while returning
    repeated first coefficients for simple hybrid composite nodes. TD-GradDFT
    only needs the semilocal epsilon_xc part here; exact exchange is handled by
    the SCF/RSH layer.
    """

    def __init__(self, module: Any):
        self._module = module
        self.__version__ = getattr(module, "__version__", None)

    def __getattr__(self, name: str):
        if name in SAFE_JAX_XC_WRAPPED_COMPOSITES:
            return self._hybrid_factory(name)
        return getattr(self._module, name)

    def _hybrid_factory(self, name: str):
        terms = SAFE_JAX_XC_WRAPPED_COMPOSITES[name]
        module = self._module
        raw_factory = getattr(module, name, None)

        def factory(*, polarized: bool = False):
            child_functionals = [
                (
                    coefficient,
                    getattr(module, child_name)(polarized=polarized, **child_params),
                )
                for coefficient, child_name, child_params in terms
            ]
            raw_functional = None
            if raw_factory is not None:
                try:
                    raw_functional = raw_factory(polarized=polarized)
                except Exception:
                    raw_functional = None

            def functional(rho_fn, r, mo_fn=None):
                total = None
                for coefficient, child in child_functionals:
                    value = child(rho_fn, r, mo_fn)
                    contribution = coefficient * value
                    total = contribution if total is None else total + contribution
                if total is None:
                    return jnp.asarray(0.0)
                return total

            if raw_functional is not None:
                for attr in ("cam_alpha", "cam_beta", "cam_omega", "nlc_b", "nlc_C"):
                    if hasattr(raw_functional, attr):
                        setattr(functional, attr, getattr(raw_functional, attr))
            return functional

        return factory


def load_jax_xc() -> tuple[Any, str]:
    """Load jax_xc through external, vendored-generated, then fallback paths."""

    try:
        module = importlib.import_module("jax_xc")
        return _SafeJAXXCModule(module), "upstream"
    except _JAX_XC_IMPORT_ERRORS:
        pass

    generated_path = _vendored_generated_path()
    if generated_path.exists():
        _ensure_import_path(generated_path)
        try:
            module = importlib.import_module("jax_xc")
            return _SafeJAXXCModule(module), "vendored"
        except _JAX_XC_IMPORT_ERRORS:
            pass

    return _FallbackJAXXC(), "fallback"
