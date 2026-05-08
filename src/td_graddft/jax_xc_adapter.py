from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from .jax_libxc import RestrictedFeatureBundle, _eval_xc_per_particle
from .xc_backend.vendor import vendored_jax_xc_info

_JAX_XC_IMPORT_ERRORS = (ImportError, OSError)

_HybridTerm = tuple[float, str, dict[str, float]]

_SAFE_HYBRID_COMPOSITES: dict[str, tuple[_HybridTerm, ...]] = {
    "pbe0": (
        (0.75, "gga_x_pbe", {}),
        (1.0, "gga_c_pbe", {}),
    ),
    "pbeh": (
        (0.75, "gga_x_pbe", {}),
        (1.0, "gga_c_pbe", {}),
    ),
    "hyb_gga_xc_pbeh": (
        (0.75, "gga_x_pbe", {}),
        (1.0, "gga_c_pbe", {}),
    ),
    "hyb_gga_xc_pbe0_13": (
        (0.75, "gga_x_pbe", {}),
        (1.0, "gga_c_pbe", {}),
    ),
    "b3lyp": (
        (0.08, "lda_x", {}),
        (0.72, "gga_x_b88", {}),
        (0.19, "lda_c_vwn_rpa", {}),
        (0.81, "gga_c_lyp", {}),
    ),
    "hyb_gga_xc_b3lyp": (
        (0.08, "lda_x", {}),
        (0.72, "gga_x_b88", {}),
        (0.19, "lda_c_vwn_rpa", {}),
        (0.81, "gga_c_lyp", {}),
    ),
    "b3pw91": (
        (0.08, "lda_x", {}),
        (0.72, "gga_x_b88", {}),
        (0.19, "lda_c_pw", {}),
        (0.81, "gga_c_pw91", {}),
    ),
    "hyb_gga_xc_b3pw91": (
        (0.08, "lda_x", {}),
        (0.72, "gga_x_b88", {}),
        (0.19, "lda_c_pw", {}),
        (0.81, "gga_c_pw91", {}),
    ),
    "bhandhlyp": (
        (0.5, "gga_x_b88", {}),
        (1.0, "gga_c_lyp", {}),
    ),
    "hyb_gga_xc_bhandhlyp": (
        (0.5, "gga_x_b88", {}),
        (1.0, "gga_c_lyp", {}),
    ),
    "hyb_gga_xc_hse03": (
        (1.0, "gga_x_wpbeh", {"_omega": 0.0}),
        (-0.25, "gga_x_wpbeh", {"_omega": 0.18898815748423098}),
        (1.0, "gga_c_pbe", {}),
    ),
    "hyb_gga_xc_hse06": (
        (1.0, "gga_x_wpbeh", {"_omega": 0.0}),
        (-0.25, "gga_x_wpbeh", {"_omega": 0.11}),
        (1.0, "gga_c_pbe", {}),
    ),
    "hyb_gga_xc_cam_b3lyp": (
        (0.35, "gga_x_b88", {}),
        (0.46, "gga_x_ityh", {"_omega": 0.33}),
        (0.19, "lda_c_vwn", {}),
        (0.81, "gga_c_lyp", {}),
    ),
}


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
        if name in _SAFE_HYBRID_COMPOSITES:
            return self._hybrid_factory(name)
        return getattr(self._module, name)

    def _hybrid_factory(self, name: str):
        terms = _SAFE_HYBRID_COMPOSITES[name]
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
