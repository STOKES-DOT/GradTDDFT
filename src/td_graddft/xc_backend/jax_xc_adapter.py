from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp


class MissingJAXXCError(ImportError):
    """Raised when the configured environment does not provide ``jax_xc``."""


JAXXCStatus = Literal["strict", "wrapped", "experimental", "unavailable"]


@dataclass(frozen=True)
class JAXXCFunctionalInfo:
    name: str
    status: JAXXCStatus
    family: str
    reason: str
    children: tuple[str, ...] = ()


_DENSITY_FLOOR = 1e-12

_STRICT_JAX_XC_FUNCTIONALS = {
    "lda_x",
    "lda_c_pw",
    "lda_c_vwn",
    "lda_c_vwn_rpa",
    "gga_x_b88",
    "gga_x_pbe",
    "gga_x_wpbeh",
    "gga_c_lyp",
    "gga_c_pbe",
}

_HybridTerm = tuple[float, str, dict[str, float]]

SAFE_JAX_XC_WRAPPED_COMPOSITES: dict[str, tuple[_HybridTerm, ...]] = {
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
}

_EXPERIMENTAL_JAX_XC_FUNCTIONALS = {
    "gga_x_rpbe",
    "gga_x_wc",
    "gga_x_pw91",
    "hyb_gga_xc_b97",
    "hyb_gga_xc_b97_1",
    "hyb_gga_xc_wb97x",
}

_KNOWN_MISMATCH_JAX_XC_FUNCTIONALS = {
    "hyb_gga_xc_b97",
    "hyb_gga_xc_b97_1",
    "hyb_gga_xc_wb97x",
}

_JAX_XC_FUNCTIONAL_PREFIXES = (
    "lda_",
    "gga_",
    "mgga_",
    "hyb_gga_",
    "hyb_mgga_",
)


def _is_mgga_name(name: str) -> bool:
    return name.startswith("mgga_") or name.startswith("hyb_mgga_")


def _is_kinetic_name(name: str) -> bool:
    return (
        name.startswith("lda_k_")
        or name.startswith("gga_k_")
        or name.startswith("mgga_k_")
    )


def _is_jax_xc_functional_name(name: str) -> bool:
    return name.startswith(_JAX_XC_FUNCTIONAL_PREFIXES)


def _family_from_name(name: str) -> str:
    if name.startswith("lda_"):
        return "LDA"
    if _is_mgga_name(name):
        return "MGGA"
    if name.startswith("gga_"):
        return "GGA"
    if name.startswith("hyb_gga_"):
        return "HYB_GGA"
    return "unknown"


def load_jax_xc() -> tuple[Any, str]:
    """Load the installed ``jax_xc`` package.

    TD-GradDFT no longer carries a generated or local fallback for traditional
    XC formulas. The active runtime environment must provide ``jax_xc``.
    """

    try:
        module = importlib.import_module("jax_xc")
    except (ImportError, OSError) as exc:
        raise MissingJAXXCError(
            "Traditional XC evaluation requires the optional package 'jax_xc' "
            "to be installed in the active Python environment."
        ) from exc
    return _JAXXCModule(module), "upstream"


def _active_jax_xc_has(name: str) -> bool:
    try:
        module, _ = load_jax_xc()
    except MissingJAXXCError:
        return False
    return hasattr(module, name)


def _active_jax_xc_names() -> set[str]:
    try:
        module, _ = load_jax_xc()
    except MissingJAXXCError:
        return set()
    raw_module = getattr(module, "_module", module)
    names = {name for name in dir(raw_module) if not name.startswith("_")}
    names.update(SAFE_JAX_XC_WRAPPED_COMPOSITES)
    return names


def jax_xc_functional_info(name: str) -> JAXXCFunctionalInfo:
    canonical = str(name).strip().lower()
    if canonical in _STRICT_JAX_XC_FUNCTIONALS:
        return JAXXCFunctionalInfo(
            name=canonical,
            status="strict",
            family=_family_from_name(canonical),
            reason="Provided by the installed jax_xc package and allowed by default.",
        )
    if canonical in SAFE_JAX_XC_WRAPPED_COMPOSITES:
        children = tuple(child for _, child, _ in SAFE_JAX_XC_WRAPPED_COMPOSITES[canonical])
        return JAXXCFunctionalInfo(
            name=canonical,
            status="wrapped",
            family=_family_from_name(canonical),
            reason="Resolved into safe semilocal child components from the installed jax_xc package.",
            children=children,
        )
    is_active = _active_jax_xc_has(canonical)
    if is_active and _is_kinetic_name(canonical):
        return JAXXCFunctionalInfo(
            name=canonical,
            status="unavailable",
            family=_family_from_name(canonical),
            reason=(
                "This is a kinetic-energy functional exposed by jax_xc, not an XC "
                "energy-density component."
            ),
        )
    if is_active and (
        canonical in _EXPERIMENTAL_JAX_XC_FUNCTIONALS
        or _is_jax_xc_functional_name(canonical)
    ):
        reason = "Available from installed jax_xc but not validated for Neural XC training."
        if canonical in _KNOWN_MISMATCH_JAX_XC_FUNCTIONALS:
            reason = (
                "B97-family jax_xc output has benchmark mismatches against PySCF/libxc; "
                "explicit experimental opt-in is required."
            )
        elif _is_mgga_name(canonical):
            reason = (
                "MGGA functional is available from jax_xc and requires an orbital-derived "
                "tau/mo_fn bridge; explicit experimental opt-in is required."
            )
            if canonical.startswith("hyb_mgga_"):
                reason = (
                    "Hybrid MGGA functional is available from jax_xc as an experimental "
                    "local channel; TD-GradDFT does not infer exact-exchange fractions "
                    "from this name."
                )
        elif canonical.startswith("hyb_gga_"):
            reason = (
                "Hybrid GGA functional is available from jax_xc as an experimental "
                "local channel; TD-GradDFT does not infer exact-exchange fractions "
                "from this name."
            )
        return JAXXCFunctionalInfo(
            name=canonical,
            status="experimental",
            family=_family_from_name(canonical),
            reason=reason,
        )
    return JAXXCFunctionalInfo(
        name=canonical,
        status="unavailable",
        family=_family_from_name(canonical),
        reason="No allowed jax_xc implementation is available.",
    )


def list_jax_xc_functionals(status: JAXXCStatus | None = None) -> tuple[JAXXCFunctionalInfo, ...]:
    active_functionals = {
        name
        for name in _active_jax_xc_names()
        if _is_jax_xc_functional_name(name)
    }
    names = sorted(
        _STRICT_JAX_XC_FUNCTIONALS
        | set(SAFE_JAX_XC_WRAPPED_COMPOSITES)
        | _EXPERIMENTAL_JAX_XC_FUNCTIONALS
        | active_functionals
    )
    infos = tuple(jax_xc_functional_info(name) for name in names)
    if status is None:
        return infos
    return tuple(info for info in infos if info.status == status)


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


class _JAXXCModule:
    """Proxy an installed jax_xc module and expose TD-GradDFT composites."""

    def __init__(self, module: Any):
        self._module = module
        self.__version__ = getattr(module, "__version__", None)

    def __getattr__(self, name: str):
        if name in SAFE_JAX_XC_WRAPPED_COMPOSITES:
            return self._hybrid_factory(name)
        return getattr(self._module, name)

    def _hybrid_factory(self, name: str):
        terms = SAFE_JAX_XC_WRAPPED_COMPOSITES[name]
        module = self

        def factory(*, polarized: bool = False):
            child_functionals = [
                (
                    coefficient,
                    getattr(module, child_name)(polarized=polarized, **child_params),
                )
                for coefficient, child_name, child_params in terms
            ]

            def functional(rho_fn, r, mo_fn=None):
                total = None
                for coefficient, child in child_functionals:
                    try:
                        value = child(rho_fn, r, mo_fn)
                    except TypeError:
                        value = child(rho_fn, r)
                    contribution = coefficient * _coerce_functional_value(value)
                    total = contribution if total is None else total + contribution
                if total is None:
                    return jnp.asarray(0.0)
                return total

            return functional

        return factory


_SafeJAXXCModule = _JAXXCModule


def _factory_kwargs(name: str, omega: Any | None) -> dict[str, Any]:
    if name == "gga_x_wpbeh" and omega is not None:
        return {"_omega": omega}
    return {}


def _coerce_functional_value(value: Any) -> Any:
    if isinstance(value, tuple):
        value = value[0]
    return value


def _evaluate_factory_from_restricted_features(
    factory: Any,
    features: Any,
    *,
    family: str,
) -> jnp.ndarray:
    rho = jnp.maximum(jnp.asarray(features.rho), _DENSITY_FLOOR)
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

        if family == "MGGA":
            value = factory(rho_fn, origin, mo_fn)
        else:
            value = factory(rho_fn, origin)
        return jnp.asarray(_coerce_functional_value(value), dtype=rho.dtype)

    if rho.ndim == 0:
        value = point_eval(rho, grad_mag, tau)
    else:
        value = jax.vmap(point_eval)(rho, grad_mag, tau)
    return jnp.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def _evaluate_factory_from_unrestricted_features(
    factory: Any,
    features: Any,
    *,
    family: str,
) -> jnp.ndarray:
    rho_a = jnp.maximum(jnp.asarray(features.rho_a), _DENSITY_FLOOR)
    rho_b = jnp.maximum(jnp.asarray(features.rho_b), _DENSITY_FLOOR)
    sigma_aa = jnp.maximum(jnp.asarray(features.sigma_aa), 0.0)
    sigma_bb = jnp.maximum(jnp.asarray(features.sigma_bb), 0.0)
    grad_a = jnp.sqrt(sigma_aa)
    grad_b = jnp.sqrt(sigma_bb)
    tau_a = jnp.maximum(jnp.asarray(features.tau_a), 0.0)
    tau_b = jnp.maximum(jnp.asarray(features.tau_b), 0.0)
    origin = jnp.zeros((3,), dtype=rho_a.dtype)

    def point_eval(rho_a_value, rho_b_value, grad_a_value, grad_b_value, tau_a_value, tau_b_value):
        def rho_fn(r):
            return jnp.asarray(
                [
                    rho_a_value + grad_a_value * r[0],
                    rho_b_value + grad_b_value * r[0],
                ],
                dtype=rho_a.dtype,
            )

        def mo_fn(r):
            mo_grad_a = jnp.sqrt(jnp.maximum(2.0 * tau_a_value, 1e-30))
            mo_grad_b = jnp.sqrt(jnp.maximum(2.0 * tau_b_value, 1e-30))
            return jnp.asarray(
                [
                    [mo_grad_a * r[0]],
                    [mo_grad_b * r[0]],
                ],
                dtype=rho_a.dtype,
            )

        if family == "MGGA":
            value = factory(rho_fn, origin, mo_fn)
        else:
            value = factory(rho_fn, origin)
        return jnp.asarray(_coerce_functional_value(value), dtype=rho_a.dtype)

    if rho_a.ndim == 0:
        value = point_eval(rho_a, rho_b, grad_a, grad_b, tau_a, tau_b)
    else:
        value = jax.vmap(point_eval)(rho_a, rho_b, grad_a, grad_b, tau_a, tau_b)
    return jnp.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def eval_jax_xc_from_restricted_features(
    name: str,
    features: Any,
    *,
    omega: Any | None = None,
    allow_experimental_jax_xc: bool = False,
) -> jnp.ndarray:
    """Evaluate a jax_xc functional as epsilon_xc on restricted grid features."""

    info = jax_xc_functional_info(name)
    _raise_if_not_allowed(info, allow_experimental_jax_xc=allow_experimental_jax_xc)
    module, _ = load_jax_xc()
    try:
        factory = getattr(module, info.name)
    except AttributeError as exc:
        raise KeyError(f"Installed jax_xc does not expose functional {info.name!r}.") from exc
    functional = factory(
        polarized=False,
        **_factory_kwargs(info.name, omega),
    )
    return _evaluate_factory_from_restricted_features(
        functional,
        features,
        family=info.family,
    )


def eval_jax_xc_from_unrestricted_features(
    name: str,
    features: Any,
    *,
    omega: Any | None = None,
    allow_experimental_jax_xc: bool = False,
) -> jnp.ndarray:
    """Evaluate a jax_xc functional as spin-polarized epsilon_xc."""

    info = jax_xc_functional_info(name)
    _raise_if_not_allowed(info, allow_experimental_jax_xc=allow_experimental_jax_xc)
    module, _ = load_jax_xc()
    try:
        factory = getattr(module, info.name)
    except AttributeError as exc:
        raise KeyError(f"Installed jax_xc does not expose functional {info.name!r}.") from exc
    functional = factory(
        polarized=True,
        **_factory_kwargs(info.name, omega),
    )
    return _evaluate_factory_from_unrestricted_features(
        functional,
        features,
        family=info.family,
    )
