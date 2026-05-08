from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import re
from typing import Any, Callable, Literal, Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array


_CX = -(3.0 / 4.0) * (3.0 / math.pi) ** (1.0 / 3.0)
_LDA_X_LOCAL_PREFAC = -(3.0 / 2.0) * (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
_PW92_A = 0.0310907
_PW92_ALPHA1 = 0.21370
_PW92_BETA1 = 7.5957
_PW92_BETA2 = 3.5876
_PW92_BETA3 = 1.6382
_PW92_BETA4 = 0.49294
_VWN5_A = 0.0310907
_VWN5_B = 3.72744
_VWN5_C = 12.9352
_VWN5_X0 = -0.10498
_VWNRPA_A = 0.0310907
_VWNRPA_B = 13.0720
_VWNRPA_C = 42.7198
_VWNRPA_X0 = -0.409286
_LYP_A = 0.04918
_LYP_B = 0.132
_LYP_C = 0.2533
_LYP_D = 0.349
_LYP_CF = (3.0 / 10.0) * (3.0 * math.pi**2) ** (2.0 / 3.0)
_PBE_KAPPA = 0.804
_PBE_MU = 0.2195149727645171
_PBE_BETA = 0.06672455060314922
_PBE_GAMMA = (1.0 - math.log(2.0)) / (math.pi**2)
_B88_BETA = 0.0042
_DENSITY_FLOOR = 1e-12
_SIGMA_FLOOR = 1e-18


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


# Register pytree for JAX - compatible with different versions
try:
    # JAX >= 0.4.26
    jax.tree_util.register_dataclass(RestrictedFeatureBundle)
except TypeError:
    # Older JAX versions
    def _restricted_feature_bundle_unflatten(aux_data, children):
        return RestrictedFeatureBundle(*children)
    jax.tree_util.register_pytree_node(
        RestrictedFeatureBundle,
        lambda x: ([x.rho_a, x.rho_b, x.sigma_aa, x.sigma_ab, x.sigma_bb, x.tau_a, x.tau_b], None),
        _restricted_feature_bundle_unflatten
    )


@dataclass(frozen=True)
class XCTerm:
    name: str
    coefficient: float
    family: str
    kind: str = "semilocal"


@dataclass(frozen=True)
class RSHFunctionalPreset:
    """Strict literature/PySCF metadata for a named RSH XC functional."""

    name: str
    pyscf_xc_name: str
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
    strict_jax_supported: bool = False
    notes: str = ""

    def to_pyscf_rsh(self) -> tuple[float, float, float]:
        return (
            self.default_omega,
            self.default_lr_hf_fraction,
            self.default_sr_hf_fraction - self.default_lr_hf_fraction,
        )

    def to_pyscf_rsh_and_hybrid(self) -> tuple[float, float, float]:
        return (
            self.default_omega,
            self.default_lr_hf_fraction,
            self.default_sr_hf_fraction,
        )

    @property
    def default_params(self):
        from .nn_rsh.schema import ResolvedRSHParameters

        return ResolvedRSHParameters(
            sr_hf_fraction=jnp.asarray(self.default_sr_hf_fraction),
            lr_hf_fraction=jnp.asarray(self.default_lr_hf_fraction),
            omega=jnp.asarray(self.default_omega),
        )

    def params_for_omega_source(
        self,
        omega_source: Literal["canonical", "optxc"] = "canonical",
    ):
        from .nn_rsh.schema import ResolvedRSHParameters

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
        from .nn_rsh.schema import RSHFunctionalTemplate

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
            monotonic_lr_hf=True,
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
        pyscf_xc_name="LC_WPBE",
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
        strict_jax_supported=True,
        notes=(
            "Vydrov-Scuseria LC-wPBE uses a fully long-range-corrected PBE "
            "exchange split. It is not equivalent to full PBE exchange plus LR-HF."
        ),
    ),
    "wb97x-d": RSHFunctionalPreset(
        name="wb97x-d",
        pyscf_xc_name="WB97X_D",
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
    "wb97x_d": "wb97x-d",
    "wb97x-d": "wb97x-d",
    "wb97xd": "wb97x-d",
    "omega-b97x-d": "wb97x-d",
    "omega_b97x_d": "wb97x-d",
    "ωb97x-d": "wb97x-d",
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


def _canonical_name(name: str) -> str:
    return name.strip().lower()


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


def _registry() -> dict[str, tuple[str, Callable[[RestrictedFeatureBundle], Array]]]:
    return {
        "lda_x": ("LDA", lda_x_energy_density),
        "lda_c_pw": ("LDA", lda_c_pw92_energy_density),
        "lda_c_vwn": ("LDA", lda_c_vwn_energy_density),
        "lda_c_vwn_rpa": ("LDA", lda_c_vwn_rpa_energy_density),
        "gga_x_b88": ("GGA", gga_x_b88_energy_density),
        "gga_x_pbe": ("GGA", gga_x_pbe_energy_density),
        "gga_x_wpbeh": ("GGA", gga_x_wpbeh_energy_density),
        "gga_c_lyp": ("GGA", gga_c_lyp_energy_density),
        "gga_c_pbe": ("GGA", gga_c_pbe_energy_density),
        "hf": ("HF", lambda features: jnp.zeros_like(features.rho)),
    }


def b3lyp_component_basis() -> tuple[str, ...]:
    """Return the explicit semilocal basis channels of the JAX B3LYP decomposition."""

    return _B3LYP_COMPONENT_BASIS


def b3lyp_component_coefficients() -> tuple[float, ...]:
    """Return B3LYP component coefficients in ``b3lyp_component_basis`` order.

    The returned tuple follows the direct Neural_xc training basis order
    ``(lda_x, gga_x_b88, lda_c_vwn_rpa, gga_c_lyp, hf_projected)``.
    """

    return _B3LYP_COMPONENT_COEFFICIENTS


def parse_xc(spec: str, *, allow_experimental_jax_xc: bool = False) -> list[XCTerm]:
    """Parse a small PySCF-like XC specification into weighted terms."""

    normalized = _normalize_spec(spec)
    tokens = [part.strip() for part in re.split(r"[+,]", normalized) if part.strip()]
    registry = _registry()
    terms = []
    for token in tokens:
        coefficient, raw_name = _parse_coefficient(token)
        name = _canonical_name(raw_name)
        if name not in registry:
            from .jax_xc_adapter import jax_xc_functional_info

            info = jax_xc_functional_info(name)
            if info.status == "experimental" and not allow_experimental_jax_xc:
                raise ValueError(
                    f"jax_xc functional {name!r} is experimental: {info.reason} "
                    "Pass allow_experimental_jax_xc=True to evaluate it."
                )
            if info.status not in {"wrapped", "experimental"}:
                raise KeyError(
                    f"Unsupported JAX XC functional {raw_name!r}. "
                    "Supported names include TD-GradDFT strict local names, safe wrapped "
                    "jax_xc composites, and explicitly enabled experimental jax_xc names."
                )
            family = info.family
            kind = "semilocal"
            terms.append(XCTerm(name=name, coefficient=coefficient, family=family, kind=kind))
            continue
        family, _ = registry[name]
        kind = "hf" if name == "hf" else "semilocal"
        terms.append(XCTerm(name=name, coefficient=coefficient, family=family, kind=kind))
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
    if "GGA" in families:
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
    """Return diagnostics for the active jax_xc-compatible backend."""

    from .jax_xc_adapter import load_jax_xc
    from .xc_backend.vendor import vendored_jax_xc_info

    module, backend = load_jax_xc()
    vendored = vendored_jax_xc_info()
    return {
        "backend": backend,
        "module_version": getattr(module, "__version__", None),
        "vendored_complete": vendored.complete,
        "vendored_root": str(vendored.root),
        "vendored_commit": vendored.commit,
        "vendored_version": vendored.version,
        "vendored_reason": vendored.reason,
    }


def _safe_rho(rho: Array) -> Array:
    return jnp.maximum(jnp.asarray(rho), _DENSITY_FLOOR)


def _safe_sigma(sigma: Array) -> Array:
    # Keep the regularization smooth so higher-order TDDFT response derivatives
    # remain finite at vanishing density gradients.
    return jnp.asarray(sigma) + _SIGMA_FLOOR


def restricted_feature_bundle_from_rho_grad_tau(
    rho: Array,
    grad: Array | None = None,
    tau: Array | None = None,
    *,
    density_floor: float = _DENSITY_FLOOR,
) -> RestrictedFeatureBundle:
    """Build a restricted-spin semilocal bundle from (rho, grad rho, tau).

    The helper follows the closed-shell convention used throughout TD-GradDFT:
    alpha and beta channels share the same density, gradient, and kinetic-energy
    density, each carrying half of the total quantity.
    """

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


def _lda_x_per_particle(features: RestrictedFeatureBundle) -> Array:
    rho = _safe_rho(features.rho)
    return _CX * jnp.cbrt(rho)


def _pw92_unpolarized_correlation(rs: Array) -> Array:
    q1 = 2.0 * _PW92_A * (
        _PW92_BETA1 * jnp.sqrt(rs)
        + _PW92_BETA2 * rs
        + _PW92_BETA3 * rs ** 1.5
        + _PW92_BETA4 * rs**2
    )
    q0 = -2.0 * _PW92_A * (1.0 + _PW92_ALPHA1 * rs)
    return q0 * jnp.log1p(1.0 / q1)


def _vwn_unpolarized_correlation(rs: Array, *, variant: Literal["vwn5", "vwn_rpa"]) -> Array:
    if variant == "vwn5":
        a, b, c, x0 = _VWN5_A, _VWN5_B, _VWN5_C, _VWN5_X0
    elif variant == "vwn_rpa":
        a, b, c, x0 = _VWNRPA_A, _VWNRPA_B, _VWNRPA_C, _VWNRPA_X0
    else:
        raise ValueError(f"Unsupported VWN variant {variant!r}.")

    x = jnp.sqrt(rs)
    x_poly = x**2 + b * x + c
    x0_poly = x0**2 + b * x0 + c
    q = jnp.sqrt(4.0 * c - b**2)
    angle = jnp.arctan(q / (2.0 * x + b))
    log_term = jnp.log((x**2) / x_poly)
    shifted_log_term = jnp.log(((x - x0) ** 2) / x_poly)
    return a * (
        log_term
        + 2.0 * b * angle / q
        - (b * x0 / x0_poly) * (shifted_log_term + 2.0 * (b + 2.0 * x0) * angle / q)
    )


def _lda_c_pw92_per_particle(features: RestrictedFeatureBundle) -> Array:
    rho = _safe_rho(features.rho)
    rs = (3.0 / (4.0 * jnp.pi * rho)) ** (1.0 / 3.0)
    return _pw92_unpolarized_correlation(rs)


def _lda_c_vwn_per_particle(features: RestrictedFeatureBundle) -> Array:
    rho = _safe_rho(features.rho)
    rs = (3.0 / (4.0 * jnp.pi * rho)) ** (1.0 / 3.0)
    return _vwn_unpolarized_correlation(rs, variant="vwn5")


def _lda_c_vwn_rpa_per_particle(features: RestrictedFeatureBundle) -> Array:
    rho = _safe_rho(features.rho)
    rs = (3.0 / (4.0 * jnp.pi * rho)) ** (1.0 / 3.0)
    return _vwn_unpolarized_correlation(rs, variant="vwn_rpa")


def _gga_x_pbe_per_particle(features: RestrictedFeatureBundle) -> Array:
    rho = _safe_rho(features.rho)
    sigma = _safe_sigma(features.sigma)
    kf = (3.0 * jnp.pi**2 * rho) ** (1.0 / 3.0)
    s2 = sigma / (4.0 * kf**2 * rho**2)
    fx = 1.0 + _PBE_KAPPA - _PBE_KAPPA / (1.0 + _PBE_MU * s2 / _PBE_KAPPA)
    return _lda_x_per_particle(features) * fx


def _gga_x_b88_per_particle(features: RestrictedFeatureBundle) -> Array:
    rho = _safe_rho(features.rho)
    sigma = _safe_sigma(features.sigma)
    grad = jnp.sqrt(sigma)
    x = grad / (rho ** (4.0 / 3.0) + _DENSITY_FLOOR)
    correction = _B88_BETA * x**2 / (
        1.0 + 6.0 * _B88_BETA * x * jnp.arcsinh(x + _DENSITY_FLOOR)
    )
    return _CX * jnp.cbrt(rho) - correction * rho ** (1.0 / 3.0)


def _gga_c_pbe_per_particle(features: RestrictedFeatureBundle) -> Array:
    rho = _safe_rho(features.rho)
    sigma = _safe_sigma(features.sigma)
    eps_c_lda = _lda_c_pw92_per_particle(features)
    kf = (3.0 * jnp.pi**2 * rho) ** (1.0 / 3.0)
    ks = jnp.sqrt(4.0 * kf / jnp.pi)
    t2 = sigma / (4.0 * ks**2 * rho**2)
    exp_arg = jnp.clip(-eps_c_lda / _PBE_GAMMA, -60.0, 60.0)
    a = (_PBE_BETA / _PBE_GAMMA) / (jnp.exp(exp_arg) - 1.0 + 1e-12)
    numerator = (_PBE_BETA / _PBE_GAMMA) * t2 * (1.0 + a * t2)
    denominator = 1.0 + a * t2 + a**2 * t2**2
    h = _PBE_GAMMA * jnp.log1p(numerator / denominator)
    return eps_c_lda + h


def gga_c_lyp_energy_density(features: RestrictedFeatureBundle) -> Array:
    rho_a = jnp.maximum(jnp.asarray(features.rho_a), 0.0)
    rho_b = jnp.maximum(jnp.asarray(features.rho_b), 0.0)
    rho = rho_a + rho_b
    rho_safe = _safe_rho(rho)
    inv_cbrt_rho = jnp.power(rho_safe, -1.0 / 3.0)
    denom = 1.0 + _LYP_D * inv_cbrt_rho
    omega = jnp.exp(-_LYP_C * inv_cbrt_rho) * jnp.power(rho_safe, -11.0 / 3.0) / denom
    delta = _LYP_C * inv_cbrt_rho + _LYP_D * inv_cbrt_rho / denom

    sigma_aa = jnp.asarray(features.sigma_aa)
    sigma_bb = jnp.asarray(features.sigma_bb)
    sigma = jnp.asarray(features.sigma)

    spin_prefactor = rho_a * rho_b
    spin_term = (
        8.0
        * jnp.power(2.0, 2.0 / 3.0)
        * _LYP_CF
        * (jnp.power(rho_a, 8.0 / 3.0) + jnp.power(rho_b, 8.0 / 3.0))
        + ((47.0 - 7.0 * delta) / 18.0) * sigma
        - ((45.0 - delta) / 18.0) * (sigma_aa + sigma_bb)
        - ((delta - 11.0) / 9.0)
        * (rho_a * sigma_aa / rho_safe + rho_b * sigma_bb / rho_safe)
    )
    gradient_term = (
        -(2.0 / 3.0) * jnp.square(rho_safe) * sigma
        + ((2.0 / 3.0) * jnp.square(rho_safe) - jnp.square(rho_a)) * sigma_bb
        + ((2.0 / 3.0) * jnp.square(rho_safe) - jnp.square(rho_b)) * sigma_aa
    )
    return (
        -4.0 * _LYP_A * spin_prefactor / (rho_safe * denom)
        - _LYP_A * _LYP_B * omega * (spin_prefactor * spin_term + gradient_term)
    )


def _gga_c_lyp_per_particle(features: RestrictedFeatureBundle) -> Array:
    return gga_c_lyp_energy_density(features) / _safe_rho(features.rho)


def _rho_local_contribution(features: RestrictedFeatureBundle, per_particle: Array) -> Array:
    rho = jnp.maximum(jnp.asarray(features.rho), 0.0)
    return rho * jnp.asarray(per_particle)


def lda_x_energy_density(features: RestrictedFeatureBundle) -> Array:
    rho_a = jnp.maximum(jnp.asarray(features.rho_a), 0.0)
    rho_b = jnp.maximum(jnp.asarray(features.rho_b), 0.0)
    return _LDA_X_LOCAL_PREFAC * (
        jnp.power(rho_a, 4.0 / 3.0) + jnp.power(rho_b, 4.0 / 3.0)
    )


def lda_c_pw92_energy_density(features: RestrictedFeatureBundle) -> Array:
    return _rho_local_contribution(features, _lda_c_pw92_per_particle(features))


def lda_c_vwn_energy_density(features: RestrictedFeatureBundle) -> Array:
    return _rho_local_contribution(features, _lda_c_vwn_per_particle(features))


def lda_c_vwn_rpa_energy_density(features: RestrictedFeatureBundle) -> Array:
    return _rho_local_contribution(features, _lda_c_vwn_rpa_per_particle(features))


def gga_x_pbe_energy_density(features: RestrictedFeatureBundle) -> Array:
    return _rho_local_contribution(features, _gga_x_pbe_per_particle(features))


def gga_x_b88_energy_density(features: RestrictedFeatureBundle) -> Array:
    return _rho_local_contribution(features, _gga_x_b88_per_particle(features))


def gga_x_wpbeh_energy_density(
    features: RestrictedFeatureBundle,
    *,
    omega: Array | float = 0.4,
) -> Array:
    from ._jax_xc_wpbeh import gga_x_wpbeh_energy_density as _impl

    return _impl(features, omega=jnp.asarray(omega))


def gga_c_pbe_energy_density(features: RestrictedFeatureBundle) -> Array:
    return _rho_local_contribution(features, _gga_c_pbe_per_particle(features))


def _omega_or_default(omega: Array | float | None) -> Array:
    if omega is None:
        return jnp.asarray(0.4)
    return jnp.asarray(omega)


def eval_xc_energy_density(
    spec: str,
    features: RestrictedFeatureBundle,
    *,
    omega: Array | float | None = None,
    allow_experimental_jax_xc: bool = False,
) -> Array:
    """Return the local XC grid contribution e_xc(r) for direct quadrature."""

    from .jax_xc_adapter import eval_jax_xc_from_restricted_features

    registry = _registry()
    omega_value = _omega_or_default(omega)
    values = []
    for term in semilocal_terms(
        spec,
        allow_experimental_jax_xc=allow_experimental_jax_xc,
    ):
        if term.name in registry:
            _, evaluator = registry[term.name]
            if term.name == "gga_x_wpbeh":
                values.append(term.coefficient * evaluator(features, omega=omega_value))
            else:
                values.append(term.coefficient * evaluator(features))
        else:
            eps = eval_jax_xc_from_restricted_features(
                term.name,
                features,
                allow_experimental_jax_xc=allow_experimental_jax_xc,
            )
            values.append(term.coefficient * features.rho * eps)
    if not values:
        return jnp.zeros_like(features.rho)
    return jnp.sum(jnp.stack(values, axis=0), axis=0)


def _eval_xc_per_particle(spec: str, features: RestrictedFeatureBundle) -> Array:
    registry = {
        "lda_x": _lda_x_per_particle,
        "lda_c_pw": _lda_c_pw92_per_particle,
        "lda_c_vwn": _lda_c_vwn_per_particle,
        "lda_c_vwn_rpa": _lda_c_vwn_rpa_per_particle,
        "gga_x_b88": _gga_x_b88_per_particle,
        "gga_x_pbe": _gga_x_pbe_per_particle,
        "gga_x_wpbeh": lambda bundle: gga_x_wpbeh_energy_density(bundle) / _safe_rho(bundle.rho),
        "gga_c_lyp": _gga_c_lyp_per_particle,
        "gga_c_pbe": _gga_c_pbe_per_particle,
        "hf": lambda bundle: jnp.zeros_like(bundle.rho),
    }
    values = []
    for term in semilocal_terms(spec):
        evaluator = registry[term.name]
        values.append(term.coefficient * evaluator(features))
    if not values:
        return jnp.zeros_like(features.rho)
    return jnp.sum(jnp.stack(values, axis=0), axis=0)


@lru_cache(maxsize=None)
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
    """Return the strict semilocal grid Hessian for a JAX libxc-like spec.

    The returned tensor follows the same reduced restricted representation used
    by PySCF's singlet TDDFT builder:
    - ``LDA``: ``(1, 1, ngrids)`` with variables ``[rho]``
    - ``GGA``: ``(4, 4, ngrids)`` with variables ``[rho, dx, dy, dz]``
    - ``MGGA``: ``(5, 5, ngrids)`` with variables ``[rho, dx, dy, dz, tau]``
    """

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
        _omega_or_default(omega),
    )
    tensor = jnp.asarray(tensor).transpose(1, 2, 0)
    tensor = jnp.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return kind, tensor
