from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any, Callable, Literal, Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .jax_xc_adapter import (
    JAXXCFunctionalInfo,
    JAXXCStatus,
    MissingJAXXCError,
    SAFE_JAX_XC_WRAPPED_COMPOSITES,
    jax_xc_functional_info,
    list_jax_xc_functionals,
)


_DENSITY_FLOOR = 1e-12


@dataclass(frozen=True)
class RestrictedFeatureBundle:
    """Restricted semilocal features on an integration grid."""

    rho_a: Array
    rho_b: Array
    sigma_aa: Array
    sigma_ab: Array
    sigma_bb: Array
    tau_a: Array
    tau_b: Array

    @property
    def rho(self) -> Array:
        return self.rho_a + self.rho_b

    @property
    def sigma(self) -> Array:
        return self.sigma_aa + 2.0 * self.sigma_ab + self.sigma_bb


try:
    jax.tree_util.register_dataclass(RestrictedFeatureBundle)
except TypeError:

    def _restricted_feature_bundle_unflatten(aux_data, children):
        del aux_data
        return RestrictedFeatureBundle(*children)

    jax.tree_util.register_pytree_node(
        RestrictedFeatureBundle,
        lambda x: (
            [x.rho_a, x.rho_b, x.sigma_aa, x.sigma_ab, x.sigma_bb, x.tau_a, x.tau_b],
            None,
        ),
        _restricted_feature_bundle_unflatten,
    )


@dataclass(frozen=True)
class XCTerm:
    name: str
    coefficient: float
    family: str
    kind: str = "semilocal"


@dataclass(frozen=True)
class LocalXCTermSpec:
    name: str
    coefficient: float = 1.0
    omega_mode: Literal["none", "fixed", "runtime"] = "none"
    fixed_omega: float | None = None


@dataclass(frozen=True)
class RSHFunctionalPreset:
    """Strict literature/libxc metadata for a named RSH XC functional."""

    name: str
    canonical_xc_name: str
    libxc_id: str
    exchange_form: str
    correlation_form: str
    default_sr_hf_fraction: float
    default_lr_hf_fraction: float
    default_omega: float
    omega_bounds: tuple[float, float]
    sr_hf_bounds: tuple[float, float]
    lr_hf_bounds: tuple[float, float] = (1.0, 1.0)
    optxc_default_omega: float | None = None
    optxc_omega_bounds: tuple[float, float] | None = None
    has_dispersion: bool = False
    dispersion_form: str | None = None
    jax_local_xc_spec: str | None = None
    local_term_specs: tuple[LocalXCTermSpec, ...] = ()
    strict_jax_supported: bool = False
    monotonic_lr_hf: bool = True
    notes: str = ""

    def to_range_separated_coefficients(self) -> tuple[float, float, float]:
        return (
            self.default_omega,
            self.default_lr_hf_fraction,
            self.default_sr_hf_fraction - self.default_lr_hf_fraction,
        )

    def to_range_separated_hybrid_coefficients(self) -> tuple[float, float, float]:
        return (
            self.default_omega,
            self.default_lr_hf_fraction,
            self.default_sr_hf_fraction,
        )

    @property
    def default_params(self):
        from ..nn_rsh.schema import ResolvedRSHParameters

        return ResolvedRSHParameters(
            sr_hf_fraction=jnp.asarray(self.default_sr_hf_fraction),
            lr_hf_fraction=jnp.asarray(self.default_lr_hf_fraction),
            omega=jnp.asarray(self.default_omega),
        )

    def params_for_omega_source(
        self,
        omega_source: Literal["canonical", "optxc"] = "canonical",
    ):
        from ..nn_rsh.schema import ResolvedRSHParameters

        default_omega = self._omega_settings(omega_source)[0]
        return ResolvedRSHParameters(
            sr_hf_fraction=jnp.asarray(self.default_sr_hf_fraction),
            lr_hf_fraction=jnp.asarray(self.default_lr_hf_fraction),
            omega=jnp.asarray(default_omega),
        )

    def _omega_settings(
        self,
        omega_source: Literal["canonical", "optxc"] = "canonical",
    ) -> tuple[float, tuple[float, float]]:
        if omega_source == "canonical":
            return self.default_omega, self.omega_bounds
        if omega_source == "optxc":
            if self.optxc_default_omega is None or self.optxc_omega_bounds is None:
                raise ValueError(f"Preset {self.name!r} does not define OPTXC omega bounds.")
            return self.optxc_default_omega, self.optxc_omega_bounds
        raise ValueError(f"Unsupported omega_source={omega_source!r}.")

    def to_template(
        self,
        omega_source: Literal["canonical", "optxc"] = "canonical",
    ):
        from ..nn_rsh.schema import RSHFunctionalTemplate

        default_omega, omega_bounds = self._omega_settings(omega_source)
        return RSHFunctionalTemplate(
            name=self.name,
            local_backend="libxc_range_separated",
            exchange_backend_id=self.libxc_id,
            correlation_backend_id=self.correlation_form,
            supports_trainable_sr_hf=self.sr_hf_bounds[0] != self.sr_hf_bounds[1],
            supports_trainable_lr_hf=self.lr_hf_bounds[0] != self.lr_hf_bounds[1],
            supports_trainable_omega=omega_bounds[0] != omega_bounds[1],
            has_dispersion=self.has_dispersion,
            monotonic_lr_hf=self.monotonic_lr_hf,
            default_sr_hf_fraction=self.default_sr_hf_fraction,
            default_lr_hf_fraction=self.default_lr_hf_fraction,
            default_omega=default_omega,
            omega_bounds=omega_bounds,
            sr_hf_bounds=self.sr_hf_bounds,
            lr_hf_bounds=self.lr_hf_bounds,
        )


_RSH_PRESETS: dict[str, RSHFunctionalPreset] = {
    "lc-wpbe": RSHFunctionalPreset(
        name="lc-wpbe",
        canonical_xc_name="LC_WPBE",
        libxc_id="HYB_GGA_XC_LC_WPBE",
        exchange_form="SR-PBE exchange plus LR-HF exchange",
        correlation_form="GGA_C_PBE",
        default_sr_hf_fraction=0.0,
        default_lr_hf_fraction=1.0,
        default_omega=0.4,
        omega_bounds=(0.05, 0.80),
        sr_hf_bounds=(0.0, 0.0),
        optxc_default_omega=0.205,
        optxc_omega_bounds=(0.13, 0.30),
        jax_local_xc_spec="lc_wpbe_local",
        local_term_specs=(
            LocalXCTermSpec("gga_x_wpbeh", 1.0, "runtime"),
            LocalXCTermSpec("gga_c_pbe", 1.0, "none"),
        ),
        strict_jax_supported=True,
        notes=(
            "Vydrov-Scuseria LC-wPBE uses a fully long-range-corrected PBE "
            "exchange split. It is not equivalent to full PBE exchange plus LR-HF."
        ),
    ),
    "hse03": RSHFunctionalPreset(
        name="hse03",
        canonical_xc_name="HSE03",
        libxc_id="HYB_GGA_XC_HSE03",
        exchange_form="PBE exchange with 25% short-range HF screening",
        correlation_form="GGA_C_PBE",
        default_sr_hf_fraction=0.25,
        default_lr_hf_fraction=0.0,
        default_omega=0.18898815748423098,
        omega_bounds=(0.05, 0.80),
        sr_hf_bounds=(0.25, 0.25),
        lr_hf_bounds=(0.0, 0.0),
        local_term_specs=(
            LocalXCTermSpec("gga_x_wpbeh", 1.0, "fixed", 0.0),
            LocalXCTermSpec("gga_x_wpbeh", -0.25, "runtime"),
            LocalXCTermSpec("gga_c_pbe", 1.0, "none"),
        ),
        strict_jax_supported=True,
        monotonic_lr_hf=False,
        notes="Screened PBE hybrid with fixed short-range HF fraction and zero long-range HF.",
    ),
    "hse06": RSHFunctionalPreset(
        name="hse06",
        canonical_xc_name="HSE06",
        libxc_id="HYB_GGA_XC_HSE06",
        exchange_form="PBE exchange with 25% short-range HF screening",
        correlation_form="GGA_C_PBE",
        default_sr_hf_fraction=0.25,
        default_lr_hf_fraction=0.0,
        default_omega=0.11,
        omega_bounds=(0.05, 0.80),
        sr_hf_bounds=(0.25, 0.25),
        lr_hf_bounds=(0.0, 0.0),
        local_term_specs=(
            LocalXCTermSpec("gga_x_wpbeh", 1.0, "fixed", 0.0),
            LocalXCTermSpec("gga_x_wpbeh", -0.25, "runtime"),
            LocalXCTermSpec("gga_c_pbe", 1.0, "none"),
        ),
        strict_jax_supported=True,
        monotonic_lr_hf=False,
        notes="Screened PBE hybrid with fixed short-range HF fraction and zero long-range HF.",
    ),
    "wb97x-d": RSHFunctionalPreset(
        name="wb97x-d",
        canonical_xc_name="WB97X_D",
        libxc_id="HYB_GGA_XC_WB97X_D",
        exchange_form="SR-B97 exchange plus 22.2036% SR-HF and 100% LR-HF exchange",
        correlation_form="B97 correlation",
        default_sr_hf_fraction=0.222036,
        default_lr_hf_fraction=1.0,
        default_omega=0.2,
        omega_bounds=(0.05, 0.80),
        sr_hf_bounds=(0.0, 0.60),
        optxc_default_omega=0.164,
        optxc_omega_bounds=(0.10, 0.24),
        has_dispersion=True,
        dispersion_form="Chai-Head-Gordon empirical damped atom-atom dispersion",
        notes=(
            "omegaB97X-D is a re-optimized B97-family RSH with empirical dispersion; "
            "it is not representable by PBE semilocal exchange/correlation."
        ),
    ),
}

_RSH_PRESET_ALIASES = {
    "lc_wpbe": "lc-wpbe",
    "lc-wpbe": "lc-wpbe",
    "lcwpbe": "lc-wpbe",
    "hse03": "hse03",
    "hyb_gga_xc_hse03": "hse03",
    "hse06": "hse06",
    "hyb_gga_xc_hse06": "hse06",
    "wb97x_d": "wb97x-d",
    "wb97x-d": "wb97x-d",
    "wb97xd": "wb97x-d",
    "omega-b97x-d": "wb97x-d",
    "omega_b97x_d": "wb97x-d",
}


def canonical_rsh_preset_name(name: str) -> str:
    key = str(name).strip().lower()
    try:
        return _RSH_PRESET_ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(_RSH_PRESETS))
        raise KeyError(f"Unsupported RSH preset {name!r}. Supported presets: {supported}.") from exc


def get_rsh_functional_preset(name: str) -> RSHFunctionalPreset:
    return _RSH_PRESETS[canonical_rsh_preset_name(name)]


def list_rsh_functional_presets() -> tuple[str, ...]:
    return tuple(sorted(_RSH_PRESETS))


_ALIASES = {
    "lda": "lda_x + lda_c_pw",
    "svwn": "lda_x + lda_c_vwn",
    "svwn_rpa": "lda_x + lda_c_vwn_rpa",
    "pbe": "gga_x_pbe + gga_c_pbe",
    "pbe0": "0.25*hf + 0.75*gga_x_pbe + gga_c_pbe",
    "pbeh": "0.25*hf + 0.75*gga_x_pbe + gga_c_pbe",
    "hyb_gga_xc_pbeh": "0.25*hf + 0.75*gga_x_pbe + gga_c_pbe",
    "b3lyp": "0.20*hf + 0.08*lda_x + 0.72*gga_x_b88 + 0.19*lda_c_vwn_rpa + 0.81*gga_c_lyp",
    "hyb_gga_xc_b3lyp": "0.20*hf + 0.08*lda_x + 0.72*gga_x_b88 + 0.19*lda_c_vwn_rpa + 0.81*gga_c_lyp",
    "bhandhlyp": "0.50*hf + 0.50*gga_x_b88 + gga_c_lyp",
    "hyb_gga_xc_bhandhlyp": "0.50*hf + 0.50*gga_x_b88 + gga_c_lyp",
    "lc_wpbe_local": "gga_x_wpbeh + gga_c_pbe",
    "lc-wpbe-local": "gga_x_wpbeh + gga_c_pbe",
    "lcwpbe_local": "gga_x_wpbeh + gga_c_pbe",
    "lc_wpbe_semilocal": "gga_x_wpbeh + gga_c_pbe",
}

_TERM_FAMILIES = {
    "lda_x": "LDA",
    "lda_c_pw": "LDA",
    "lda_c_vwn": "LDA",
    "lda_c_vwn_rpa": "LDA",
    "gga_x_b88": "GGA",
    "gga_x_pbe": "GGA",
    "gga_x_wpbeh": "GGA",
    "gga_c_lyp": "GGA",
    "gga_c_pbe": "GGA",
    "hf": "HF",
}

_B3LYP_COMPONENT_BASIS = (
    "lda_x",
    "gga_x_b88",
    "lda_c_vwn_rpa",
    "gga_c_lyp",
)
_B3LYP_COMPONENT_COEFFICIENTS = (
    0.08,
    0.72,
    0.19,
    0.81,
    0.20,
)

FRIENDLY_XC_COMPONENT_ALIASES: dict[str, str] = {
    "x_lda": "lda_x",
    "pw_c": "lda_c_pw",
    "c_pw": "lda_c_pw",
    "vwn_c": "lda_c_vwn",
    "c_vwn": "lda_c_vwn",
    "vwn_rpa_c": "lda_c_vwn_rpa",
    "c_vwn_rpa": "lda_c_vwn_rpa",
    "b88_x": "gga_x_b88",
    "x_b88": "gga_x_b88",
    "pbe_x": "gga_x_pbe",
    "x_pbe": "gga_x_pbe",
    "pbe_c": "gga_c_pbe",
    "c_pbe": "gga_c_pbe",
    "lyp_c": "gga_c_lyp",
    "c_lyp": "gga_c_lyp",
    "wpbeh_x": "gga_x_wpbeh",
    "x_wpbeh": "gga_x_wpbeh",
    "rpbe_x": "gga_x_rpbe",
    "x_rpbe": "gga_x_rpbe",
    "pw91_x": "gga_x_pw91",
    "x_pw91": "gga_x_pw91",
    "pw91_c": "gga_c_pw91",
    "c_pw91": "gga_c_pw91",
    "scan_x": "mgga_x_scan",
    "x_scan": "mgga_x_scan",
    "scan_c": "mgga_c_scan",
    "c_scan": "mgga_c_scan",
    "r2scan_x": "mgga_x_r2scan",
    "x_r2scan": "mgga_x_r2scan",
    "r2scan_c": "mgga_c_r2scan",
    "c_r2scan": "mgga_c_r2scan",
    "tpss_x": "mgga_x_tpss",
    "x_tpss": "mgga_x_tpss",
    "tpss_c": "mgga_c_tpss",
    "c_tpss": "mgga_c_tpss",
}

_AMBIGUOUS_XC_COMPONENT_ALIASES: dict[str, tuple[str, ...]] = {
    "pbe": ("pbe_x", "pbe_c", "gga_x_pbe", "gga_c_pbe"),
    "scan": ("scan_x", "scan_c", "mgga_x_scan", "mgga_c_scan"),
    "r2scan": ("r2scan_x", "r2scan_c", "mgga_x_r2scan", "mgga_c_r2scan"),
    "tpss": ("tpss_x", "tpss_c", "mgga_x_tpss", "mgga_c_tpss"),
    "pw91": ("pw91_x", "pw91_c", "gga_x_pw91", "gga_c_pw91"),
}

_CANONICAL_XC_COMPONENT_PREFIXES = (
    "lda_",
    "gga_",
    "mgga_",
    "hyb_gga_",
    "hyb_mgga_",
)


def _canonical_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _parse_coefficient(token: str) -> tuple[float, str]:
    token = token.strip()
    if "*" in token:
        left, right = token.split("*", 1)
        return float(left.strip()), right.strip()
    parts = token.split()
    if len(parts) == 2:
        return float(parts[0]), parts[1]
    return 1.0, token


def _normalize_spec(spec: str) -> str:
    normalized = _canonical_name(spec)
    return _ALIASES.get(normalized, normalized)


def _is_kinetic_component_name(name: str) -> bool:
    return (
        name.startswith("lda_k_")
        or name.startswith("gga_k_")
        or name.startswith("mgga_k_")
    )


def _resolve_family_qualified_component(name: str) -> str | None:
    if ":" not in name:
        return None
    family, component = (part.strip() for part in name.split(":", 1))
    family = _canonical_name(family)
    component = _canonical_name(component)
    valid_families = {"lda", "gga", "mgga", "hyb_gga", "hyb_mgga"}
    if family not in valid_families:
        raise ValueError(
            f"Unsupported XC component family qualifier {family!r}. "
            "Use one of lda, gga, mgga, hyb_gga, hyb_mgga."
        )
    if component in FRIENDLY_XC_COMPONENT_ALIASES:
        resolved = FRIENDLY_XC_COMPONENT_ALIASES[component]
        if resolved.startswith(f"{family}_"):
            return resolved
    if component.endswith(("_x", "_c", "_xc")):
        base, kind = component.rsplit("_", 1)
    elif component.startswith(("x_", "c_", "xc_")):
        kind, base = component.split("_", 1)
    else:
        raise ValueError(
            f"Family-qualified XC component {name!r} must include x/c/xc, "
            "for example 'gga:lyp_c' or 'mgga:scan_x'."
        )
    return f"{family}_{kind}_{base}"


def resolve_xc_component_name(name: str) -> str:
    """Resolve a user-facing XC component name to a canonical jax_xc name."""

    key = _canonical_name(str(name))
    qualified = _resolve_family_qualified_component(key)
    if qualified is not None:
        key = qualified
    elif key in FRIENDLY_XC_COMPONENT_ALIASES:
        key = FRIENDLY_XC_COMPONENT_ALIASES[key]
    elif key in _AMBIGUOUS_XC_COMPONENT_ALIASES:
        suggestions = ", ".join(_AMBIGUOUS_XC_COMPONENT_ALIASES[key])
        raise ValueError(
            f"XC component {name!r} is ambiguous. Use one of: {suggestions}."
        )

    if _is_kinetic_component_name(key):
        raise ValueError(
            f"XC component {name!r} resolves to {key!r}, which is a kinetic-energy "
            "functional exposed by jax_xc, not an XC energy-density component."
        )
    if key == "hf" or key.startswith(_CANONICAL_XC_COMPONENT_PREFIXES):
        return key
    return key


def b3lyp_component_basis() -> tuple[str, ...]:
    """Return the explicit semilocal basis channels of the B3LYP decomposition."""

    return _B3LYP_COMPONENT_BASIS


def b3lyp_component_coefficients() -> tuple[float, ...]:
    """Return B3LYP coefficients in ``b3lyp_component_basis`` order plus HF."""

    return _B3LYP_COMPONENT_COEFFICIENTS


def parse_xc(spec: str, *, allow_experimental_jax_xc: bool = False) -> list[XCTerm]:
    """Parse a small PySCF-like XC specification into weighted terms."""

    normalized = _normalize_spec(spec)
    tokens = [part.strip() for part in re.split(r"[+,]", normalized) if part.strip()]
    terms = []
    for token in tokens:
        coefficient, raw_name = _parse_coefficient(token)
        name = resolve_xc_component_name(raw_name)
        if name in _TERM_FAMILIES:
            family = _TERM_FAMILIES[name]
            terms.append(
                XCTerm(
                    name=name,
                    coefficient=coefficient,
                    family=family,
                    kind="hf" if name == "hf" else "semilocal",
                )
            )
            continue

        info = jax_xc_functional_info(name)
        if info.status == "experimental" and not allow_experimental_jax_xc:
            raise ValueError(
                f"jax_xc functional {name!r} is experimental: {info.reason} "
                "Pass allow_experimental_jax_xc=True to evaluate it."
            )
        if info.status not in {"wrapped", "experimental"}:
            raise KeyError(
                f"Unsupported JAX XC functional {raw_name!r}. Supported names include "
                "TD-GradDFT aliases, jax_xc semilocal names, and explicitly enabled "
                "experimental jax_xc names."
            )
        terms.append(
            XCTerm(
                name=name,
                coefficient=coefficient,
                family=info.family,
                kind="semilocal",
            )
        )
    return terms


def xc_type(spec: str, *, allow_experimental_jax_xc: bool = False) -> str:
    families = {
        term.family
        for term in parse_xc(
            spec,
            allow_experimental_jax_xc=allow_experimental_jax_xc,
        )
        if term.kind == "semilocal"
    }
    if not families:
        return "HF"
    if "MGGA" in families:
        return "MGGA"
    if "GGA" in families or "HYB_GGA" in families:
        return "GGA"
    return "LDA"


def hybrid_coeff(spec: str) -> float:
    return sum(term.coefficient for term in parse_xc(spec) if term.kind == "hf")


def semilocal_terms(spec: str, *, allow_experimental_jax_xc: bool = False) -> list[XCTerm]:
    return [
        term
        for term in parse_xc(spec, allow_experimental_jax_xc=allow_experimental_jax_xc)
        if term.kind == "semilocal"
    ]


def resolve_semilocal_xc_specs(
    spec: str | Sequence[str],
    *,
    allow_experimental_jax_xc: bool = False,
) -> tuple[str, ...]:
    """Resolve aliases/composites into semilocal energy-density channel names."""

    if isinstance(spec, str):
        raw_specs = (spec,)
    else:
        raw_specs = tuple(str(value) for value in spec)
    if not raw_specs:
        raise ValueError("semilocal_xc must contain at least one functional specification.")

    resolved: list[str] = []
    for raw_spec in raw_specs:
        for term in semilocal_terms(
            raw_spec,
            allow_experimental_jax_xc=allow_experimental_jax_xc,
        ):
            resolved.append(term.name)
    if not resolved:
        raise ValueError(f"XC specification {spec!r} contains no semilocal terms.")
    return tuple(resolved)


def jax_xc_backend_info() -> dict[str, Any]:
    """Return diagnostics for the active installed jax_xc backend."""

    from .jax_xc_adapter import load_jax_xc

    try:
        module, backend = load_jax_xc()
    except MissingJAXXCError as exc:
        return {
            "available": False,
            "backend": "missing",
            "module_version": None,
            "reason": str(exc),
        }
    return {
        "available": True,
        "backend": backend,
        "module_version": getattr(module, "__version__", None),
    }


def restricted_feature_bundle_from_rho_grad_tau(
    rho: Array,
    grad: Array | None = None,
    tau: Array | None = None,
    *,
    density_floor: float = _DENSITY_FLOOR,
) -> RestrictedFeatureBundle:
    """Build a restricted-spin semilocal bundle from (rho, grad rho, tau)."""

    rho = jnp.maximum(jnp.asarray(rho), density_floor)
    if grad is None:
        grad = jnp.zeros(rho.shape + (3,), dtype=rho.dtype)
    else:
        grad = jnp.asarray(grad, dtype=rho.dtype)
    if tau is None:
        tau = jnp.zeros_like(rho)
    else:
        tau = jnp.maximum(jnp.asarray(tau, dtype=rho.dtype), 0.0)

    half_rho = 0.5 * rho
    half_grad = 0.5 * grad
    half_sigma = jnp.einsum("...x,...x->...", half_grad, half_grad)
    half_tau = 0.5 * tau
    return RestrictedFeatureBundle(
        rho_a=half_rho,
        rho_b=half_rho,
        sigma_aa=half_sigma,
        sigma_ab=half_sigma,
        sigma_bb=half_sigma,
        tau_a=half_tau,
        tau_b=half_tau,
    )


def _omega_for_term(
    term: LocalXCTermSpec,
    omega: Array | float | None,
) -> Array | float | None:
    if term.omega_mode == "runtime":
        return 0.4 if omega is None else omega
    if term.omega_mode == "fixed":
        if term.fixed_omega is None:
            raise ValueError(f"LocalXCTermSpec {term.name!r} is fixed-omega but fixed_omega is None.")
        return term.fixed_omega
    return None


def xc_type_from_term_specs(
    term_specs: Sequence[LocalXCTermSpec],
    *,
    allow_experimental_jax_xc: bool = False,
) -> str:
    if not term_specs:
        return "HF"
    families = set()
    for term in term_specs:
        term_name = resolve_xc_component_name(term.name)
        if term_name in _TERM_FAMILIES:
            families.add(_TERM_FAMILIES[term_name])
            continue
        info = jax_xc_functional_info(term_name)
        if info.status == "experimental" and not allow_experimental_jax_xc:
            raise ValueError(
                f"jax_xc functional {term_name!r} is experimental: {info.reason} "
                "Pass allow_experimental_jax_xc=True to evaluate it."
            )
        families.add(info.family)
    if not families:
        return "HF"
    if "MGGA" in families:
        return "MGGA"
    if "GGA" in families or "HYB_GGA" in families:
        return "GGA"
    return "LDA"


def eval_xc_term_specs_energy_density(
    term_specs: Sequence[LocalXCTermSpec],
    features: RestrictedFeatureBundle,
    *,
    omega: Array | float | None = None,
    allow_experimental_jax_xc: bool = False,
) -> Array:
    from . import jax_xc_adapter

    values = []
    for term in term_specs:
        term_name = resolve_xc_component_name(term.name)
        eps = jax_xc_adapter.eval_jax_xc_from_restricted_features(
            term_name,
            features,
            omega=_omega_for_term(term, omega),
            allow_experimental_jax_xc=allow_experimental_jax_xc,
        )
        values.append(term.coefficient * features.rho * eps)
    if not values:
        return jnp.zeros_like(features.rho)
    return jnp.sum(jnp.stack(values, axis=0), axis=0)


def eval_xc_energy_density(
    spec: str,
    features: RestrictedFeatureBundle,
    *,
    omega: Array | float | None = None,
    allow_experimental_jax_xc: bool = False,
) -> Array:
    """Return the local XC grid contribution e_xc(r) for direct quadrature."""

    from . import jax_xc_adapter

    values = []
    for term in semilocal_terms(
        spec,
        allow_experimental_jax_xc=allow_experimental_jax_xc,
    ):
        term_omega = omega if term.name == "gga_x_wpbeh" else None
        eps = jax_xc_adapter.eval_jax_xc_from_restricted_features(
            term.name,
            features,
            omega=term_omega,
            allow_experimental_jax_xc=allow_experimental_jax_xc,
        )
        values.append(term.coefficient * features.rho * eps)
    if not values:
        return jnp.zeros_like(features.rho)
    return jnp.sum(jnp.stack(values, axis=0), axis=0)


@lru_cache(maxsize=None)
def _point_xc_response_kernel(
    spec: str,
    kind: str,
    density_floor: float,
    allow_experimental_jax_xc: bool,
) -> Callable[[Array, Array], Array]:
    spec_norm = str(spec)
    kind_norm = str(kind)
    density_floor_value = float(density_floor)
    allow_experimental = bool(allow_experimental_jax_xc)

    def point_energy(variables: Array, omega: Array) -> Array:
        rho_point = jnp.maximum(variables[0], density_floor_value)
        if kind_norm == "LDA":
            grad_point = jnp.zeros((3,), dtype=variables.dtype)
            tau_point = jnp.asarray(0.0, dtype=variables.dtype)
        elif kind_norm == "GGA":
            grad_point = variables[1:4]
            tau_point = jnp.asarray(0.0, dtype=variables.dtype)
        elif kind_norm == "MGGA":
            grad_point = variables[1:4]
            tau_point = jnp.maximum(variables[4], 0.0)
        else:
            raise ValueError(f"Unsupported XC response kind {kind_norm!r}.")
        features = restricted_feature_bundle_from_rho_grad_tau(
            rho_point,
            grad_point,
            tau_point,
            density_floor=density_floor_value,
        )
        return eval_xc_energy_density(
            spec_norm,
            features,
            omega=omega,
            allow_experimental_jax_xc=allow_experimental,
        )

    return jax.jit(jax.vmap(jax.hessian(point_energy, argnums=0), in_axes=(0, None)))


def eval_xc_response_tensor(
    spec: str,
    rho: Array,
    *,
    grad: Array | None = None,
    tau: Array | None = None,
    density_floor: float = _DENSITY_FLOOR,
    omega: Array | float | None = None,
    allow_experimental_jax_xc: bool = False,
) -> tuple[str, Array]:
    """Return the semilocal grid Hessian for a JAX libxc-like spec."""

    kind = xc_type(
        spec,
        allow_experimental_jax_xc=allow_experimental_jax_xc,
    )
    if kind == "HF":
        raise ValueError("Pure HF does not define a semilocal response tensor.")

    rho = jnp.maximum(jnp.asarray(rho), density_floor)
    dtype = rho.dtype
    zeros_grad = jnp.zeros(rho.shape + (3,), dtype=dtype)
    grad = zeros_grad if grad is None else jnp.asarray(grad, dtype=dtype)
    tau = jnp.zeros_like(rho) if tau is None else jnp.asarray(tau, dtype=dtype)

    if kind == "LDA":
        response_variables = rho[..., None]
    elif kind == "GGA":
        response_variables = jnp.concatenate([rho[..., None], grad], axis=-1)
    elif kind == "MGGA":
        response_variables = jnp.concatenate(
            [rho[..., None], grad, tau[..., None]],
            axis=-1,
        )
    else:
        raise ValueError(f"Unsupported XC response kind {kind!r}.")

    tensor = _point_xc_response_kernel(
        spec,
        kind,
        density_floor,
        bool(allow_experimental_jax_xc),
    )(
        response_variables,
        jnp.asarray(0.4 if omega is None else omega, dtype=dtype),
    )
    tensor = jnp.asarray(tensor).transpose(1, 2, 0)
    tensor = jnp.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return kind, tensor
