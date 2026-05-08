from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import jax
import jax.numpy as jnp
from jax.scipy import special as jsp_special
from flax import linen as nn
from jax.lax import Precision
from jaxtyping import Array, PRNGKeyArray, PyTree

from ..base.functional import NeuralXCFunctional
from ...features import (
    dm21_like_input_features,
    dm21_original_input_features,
    restricted_feature_bundle_from_response_variables,
    restricted_grid_features,
    restricted_grid_features_with_gradients,
    restricted_grid_response_variables,
)
from ...jax_libxc import (
    RestrictedFeatureBundle,
    _LDA_X_LOCAL_PREFAC,
    eval_xc_energy_density,
    parse_xc,
    resolve_semilocal_xc_specs,
)
from .defaults import DEFAULT_NEURAL_XC_ENERGY_MODE, DEFAULT_NEURAL_XC_SEMILOCAL_XC


@dataclass(frozen=True)
class BoundDM21LikeFunctional:
    name: str
    projected_local_potential_values: Array
    projected_local_kernel_values: Array
    exact_exchange_fraction: Array
    projected_local_potential_gradient_values: Array | None = None
    projected_local_potential_tau_values: Array | None = None
    projected_local_potential_laplacian_values: Array | None = None
    projected_energy_density_values: Array | None = None
    local_hf_fraction_values: Array | None = None
    response_feature_kind: str | None = None
    grid_response_tensor_fn: Callable[[], Array] | None = None
    grid_hfx_feature_gradients_fn: Callable[[], tuple[Array, Array]] | None = None

    def local_kernel(self, density: Array) -> Array:
        del density
        return self.projected_local_kernel_values

    def local_potential(self, density: Array) -> Array:
        del density
        return self.projected_local_potential_values

    def grid_kernel(self, molecule: Any) -> Array:
        del molecule
        return self.projected_local_kernel_values

    def grid_potential(self, molecule: Any) -> Array:
        del molecule
        return self.projected_local_potential_values

    def grid_potential_components(self, molecule: Any) -> tuple[Array, ...]:
        del molecule
        rho = self.projected_local_potential_values
        grad = (
            self.projected_local_potential_gradient_values
            if self.projected_local_potential_gradient_values is not None
            else jnp.zeros(rho.shape + (3,), dtype=rho.dtype)
        )
        tau = (
            self.projected_local_potential_tau_values
            if self.projected_local_potential_tau_values is not None
            else jnp.zeros_like(rho)
        )
        lapl = self.projected_local_potential_laplacian_values
        if lapl is None:
            return rho, grad, tau
        return rho, grad, tau, lapl

    def energy_density(self, density: Array) -> Array:
        del density
        if self.projected_energy_density_values is None:
            return self.projected_local_potential_values
        return self.projected_energy_density_values

    def local_hf_fraction(self, density: Array) -> Array:
        del density
        if self.local_hf_fraction_values is None:
            return jnp.full_like(
                self.projected_local_potential_values,
                self.exact_exchange_fraction,
            )
        return self.local_hf_fraction_values

    def grid_hf_fraction(self, molecule: Any) -> Array:
        del molecule
        if self.local_hf_fraction_values is None:
            return jnp.full_like(
                self.projected_local_potential_values,
                self.exact_exchange_fraction,
            )
        return self.local_hf_fraction_values

    def grid_response_tensor(self, molecule: Any) -> Array:
        del molecule
        if self.grid_response_tensor_fn is None:
            raise AttributeError("This bound functional does not expose a strict response tensor.")
        return self.grid_response_tensor_fn()

    def grid_hfx_feature_gradients(self, molecule: Any) -> tuple[Array, Array]:
        del molecule
        if self.grid_hfx_feature_gradients_fn is None:
            raise AttributeError(
                "This bound functional does not expose gradients with respect to local HF features."
            )
        return self.grid_hfx_feature_gradients_fn()


class DM21MixingMLP(nn.Module):
    hidden_dims: Sequence[int]
    output_dim: int = 2
    activation: Callable[[Array], Array] = nn.tanh
    squash_offset: float = 1e-4
    sigmoid_scale_factor: float = 2.0

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        # Use a smooth even transform so TDDFT mixed second derivatives stay finite
        # when local response features pass through zero.
        offset = jnp.asarray(self.squash_offset, dtype=jnp.asarray(inputs).dtype)
        x = 0.5 * jnp.log(jnp.square(inputs) + offset * offset)
        for width in self.hidden_dims:
            x = nn.Dense(width)(x)
            x = self.activation(x)
        x = nn.Dense(self.output_dim)(x)
        if self.sigmoid_scale_factor > 0.0:
            scale = jnp.asarray(self.sigmoid_scale_factor, dtype=x.dtype)
            x = scale * jax.nn.sigmoid(x / scale)
        return x


class GradDFTResidualMixingMLP(nn.Module):
    """GradDFT/DM21-style residual mixing network.

    This follows the structure used in GradDFT's DM21 implementation more
    closely than ``DM21MixingMLP``:
    - log-squashed inputs
    - initial dense layer + tanh
    - residual dense blocks with layer normalization
    - sigmoid-scaled output head
    """

    hidden_dims: Sequence[int]
    output_dim: int = 2
    block_activation: Callable[[Array], Array] = nn.elu
    squash_offset: float = 1e-4
    sigmoid_scale_factor: float = 2.0

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        offset = jnp.asarray(self.squash_offset, dtype=jnp.asarray(inputs).dtype)
        x = jnp.log(jnp.abs(inputs) + offset)
        first_width = int(self.hidden_dims[0])
        x = nn.Dense(first_width, name="InitialDense")(x)
        x = jnp.tanh(x)

        for index, width in enumerate(self.hidden_dims):
            residual = x
            x = nn.Dense(int(width), name=f"ResidualDense_{index}")(x)
            if residual.shape[-1] != int(width):
                residual = nn.Dense(
                    int(width),
                    use_bias=False,
                    name=f"ResidualProject_{index}",
                )(residual)
            x = x + residual
            x = nn.LayerNorm(name=f"ResidualLayerNorm_{index}")(x)
            x = self.block_activation(x)

        x = nn.Dense(self.output_dim, name="HeadDense")(x)
        if self.sigmoid_scale_factor > 0.0:
            scale = jnp.asarray(self.sigmoid_scale_factor, dtype=x.dtype)
            x = scale * jax.nn.sigmoid(x / scale)
        return x


def _normalize_hidden_dims(hidden_dims: Sequence[int]) -> tuple[int, ...]:
    dims = tuple(int(width) for width in hidden_dims)
    if not dims:
        raise ValueError("hidden_dims must contain at least one layer width.")
    if any(width <= 0 for width in dims):
        raise ValueError("All hidden_dims entries must be positive integers.")
    return dims


def _normalize_semilocal_xc_names(
    semilocal_xc: str | Sequence[str],
    *,
    allow_experimental_jax_xc: bool = False,
) -> tuple[str, ...]:
    return resolve_semilocal_xc_specs(
        semilocal_xc,
        allow_experimental_jax_xc=allow_experimental_jax_xc,
    )


SemilocalEnergyDensityFn = Callable[[RestrictedFeatureBundle], Array]
SemilocalLocalContributionFn = Callable[[RestrictedFeatureBundle, Array, float], Array]

COMMON_SEMILOCAL_COMPONENT_SPECS = {
    "lda_x": "lda_x",
    "gga_x_b88": "gga_x_b88",
    "gga_x_pbe": "gga_x_pbe",
    "lda_c_pw": "lda_c_pw",
    "lda_c_vwn": "lda_c_vwn",
    "lda_c_vwn_rpa": "lda_c_vwn_rpa",
    "gga_c_lyp": "gga_c_lyp",
    "gga_c_pbe": "gga_c_pbe",
}

_DLDH_RS_OMEGA_DEFAULT = 0.233
_DLDH_QAC_LAMBDA_DEFAULT = 0.878795
_DLDH_QAC_BETA_DEFAULT = 17.388


def _dldh_lrs_lda_f0(x: Array) -> Array:
    srpi = jnp.asarray(1.772453850905516, dtype=jnp.asarray(x).dtype)
    l0 = jnp.asarray(1.0, dtype=jnp.asarray(x).dtype)
    l1 = -4.0 / 3.0 * srpi
    l2 = jnp.asarray(2.0, dtype=jnp.asarray(x).dtype)
    l4 = jnp.asarray(-2.0 / 3.0, dtype=jnp.asarray(x).dtype)
    u1 = jnp.asarray(1.0 / 9.0, dtype=jnp.asarray(x).dtype)
    u2 = jnp.asarray(-1.0 / 60.0, dtype=jnp.asarray(x).dtype)
    u3 = jnp.asarray(1.0 / 420.0, dtype=jnp.asarray(x).dtype)
    u4 = jnp.asarray(-1.0 / 3240.0, dtype=jnp.asarray(x).dtype)
    u5 = jnp.asarray(1.0 / 27720.0, dtype=jnp.asarray(x).dtype)
    u6 = jnp.asarray(-1.0 / 262080.0, dtype=jnp.asarray(x).dtype)
    u7 = jnp.asarray(1.0 / 2721600.0, dtype=jnp.asarray(x).dtype)
    u8 = jnp.asarray(-1.0 / 4626720.0, dtype=jnp.asarray(x).dtype)
    u9 = jnp.asarray(1.0 / 2585520.0, dtype=jnp.asarray(x).dtype)

    x_safe = jnp.maximum(jnp.asarray(x), jnp.asarray(1e-8, dtype=jnp.asarray(x).dtype))
    f0_015 = (((l4 * x_safe) * x_safe + l2) * x_safe + l1) * x_safe + l0
    x2i = x_safe**(-2.0)
    f0_4 = (
        ((((((((u9 * x2i + u8) * x2i + u7) * x2i + u6) * x2i + u5) * x2i + u4) * x2i + u3) * x2i + u2) * x2i + u1)
        * x2i
    )
    inv_x = x_safe**(-1.0)
    f0_else = 1.0 - 2.0 / 3.0 * x_safe * (
        2.0 * srpi * jsp_special.erf(inv_x)
        - 3.0 * x_safe
        + x_safe**3
        + (2.0 * x_safe - x_safe**3) * jnp.exp(-(inv_x**2))
    )
    f0 = jnp.where(x_safe < 0.15, f0_015, f0_else)
    f0 = jnp.where(x_safe > 4.0, f0_4, f0)
    return f0


def _dldh_range_sep_dfa_hirao(
    rho: Array,
    ex_dfa: Array,
    *,
    omega: float,
    density_floor: float,
) -> Array:
    rho_safe = jnp.maximum(jnp.asarray(rho), density_floor)
    ex_dfa_safe = jnp.where(jnp.abs(ex_dfa) > density_floor, ex_dfa, -density_floor)
    ex_lda = _LDA_X_LOCAL_PREFAC * 2.0 * jnp.power(0.5 * rho_safe, 4.0 / 3.0)
    k_f = jnp.power(6.0 * jnp.pi**2 * rho_safe, 1.0 / 3.0)
    ratio = jnp.maximum(ex_lda / ex_dfa_safe, density_floor)
    k = jnp.sqrt(ratio) * k_f
    a = jnp.asarray(omega, dtype=rho_safe.dtype) / jnp.maximum(2.0 * k, density_floor)
    return ex_dfa * _dldh_lrs_lda_f0(2.0 * a)


def _dldh_xuxproynov(y: Array) -> Array:
    y = jnp.asarray(y)
    a1 = 1.5255251812009530
    a2 = 0.4576575543602858
    a3 = 0.4292036732051034
    b = 2.085749716493756
    c0, c1, c2, c3, c4, c5 = (
        0.7566445420735584,
        -2.6363977871370960,
        5.4745159964232880,
        -12.657308127108290,
        4.1250584725121360,
        -30.425133957163840,
    )
    d0, d1, d2, d3, d4, d5 = (
        0.00004435009886795587,
        0.58128653604457910,
        66.742764515940610,
        434.26780897229770,
        824.77657660522390,
        1657.9652731582120,
    )
    b0, b1, b2, b3, b4, b5 = (
        0.4771976183772063,
        -1.7799813494556270,
        3.8433841862302150,
        -9.5912050880518490,
        2.1730180285916720,
        -30.425133851603660,
    )
    e0, e1, e2, e3, e4, e5 = (
        0.00003347285060926091,
        0.47917931023971350,
        62.392268338574240,
        463.14816427938120,
        785.23603501040290,
        1657.9629682232730,
    )
    g = -jnp.arctan(a1 * y + a2) + a3
    p1 = c0 + c1 * y + c2 * y * y + c3 * y**3 + c4 * y**4 + c5 * y**5
    p2 = b0 + b1 * y + b2 * y * y + b3 * y**3 + b4 * y**4 + b5 * y**5
    xx_l0 = g * p1 / p2
    y_abs = jnp.maximum(jnp.abs(y), 1.0e-8)
    g1 = jnp.log(1.0 / (b * y_abs) + jnp.sqrt((1.0 / (b * y_abs)) ** 2 + 1.0)) + 2.0
    p1d = d0 + d1 * y + d2 * y * y + d3 * y**3 + d4 * y**4 + d5 * y**5
    p2e = e0 + e1 * y + e2 * y * y + e3 * y**3 + e4 * y**4 + e5 * y**5
    xx_else = g1 * p1d / p2e
    xx = jnp.where(y < 0.0, xx_l0, xx_else)

    def body(_, xcur):
        f = xcur * jnp.exp(-2.0 * xcur / 3.0) / (xcur - 2.0)
        fp = (
            -2.0 * jnp.exp(-2.0 * xcur / 3.0) * xcur / (3.0 * (xcur - 2.0))
            + jnp.exp(-2.0 * xcur / 3.0) / (xcur - 2.0)
            - jnp.exp(-2.0 * xcur / 3.0) * xcur / ((xcur - 2.0) ** 2)
        )
        return xcur - (f - y) / fp

    return jax.lax.fori_loop(0, 3, body, xx)


def _dldh_mb86x(
    lamb: float,
    bet: float,
    rho: Array,
    gamma: Array,
    tau: Array,
    lapl: Array,
) -> Array:
    rho = jnp.maximum(jnp.asarray(rho), 1.0e-12)
    gamma = jnp.maximum(jnp.asarray(gamma), 1.0e-24)
    tau = jnp.maximum(jnp.asarray(tau), 1.0e-16)
    lapl = jnp.sign(lapl) * jnp.maximum(jnp.abs(lapl), 1.0e-24)

    k_f = jnp.power(6.0 * jnp.pi**2 * rho, 1.0 / 3.0)
    xx = jnp.sqrt(gamma) / jnp.power(rho, 4.0 / 3.0)
    ff = jnp.power(
        1.0
        + 10.0
        * (70.0 / 27.0)
        / (4.0 * jnp.power(6.0 * jnp.pi**2, 2.0 / 3.0))
        * (2.0 * lamb - 1.0) ** 2
        * xx**2
        + bet
        / (16.0 * jnp.power(6.0 * jnp.pi**2, 4.0 / 3.0))
        * (2.0 * lamb - 1.0) ** 4
        * xx**4,
        0.1,
    )
    d = 2.0 * tau - 0.25 * (2.0 * lamb - 1.0) ** 2 * gamma / rho
    q = (
        (1.0 / 6.0)
        * (
            2.0 * (lamb**2 - lamb + 0.5) * lapl
            + 6.0 / 5.0 * k_f**2 * rho * (ff**2 - 1.0)
            - 2.0 * d
        )
    )
    q = jnp.sign(q) * jnp.maximum(jnp.abs(q), 1.0e-60)
    y_br = (2.0 / 3.0) * jnp.power(jnp.pi, 2.0 / 3.0) * jnp.power(rho, 5.0 / 3.0) / q
    x_br = jnp.maximum(_dldh_xuxproynov(y_br), 1.0e-12)
    denom = jnp.power(x_br**3.0 * jnp.exp(-x_br) / (8.0 * jnp.pi * rho), 1.0 / 3.0)
    return -0.5 * rho / denom * (1.0 - jnp.exp(-x_br) * (1.0 + 0.5 * x_br))


def _dldh_qac_pade(
    rho_a: Array,
    rho_b: Array,
    sigma_aa: Array,
    sigma_bb: Array,
    tau_a: Array,
    tau_b: Array,
    lapl_a: Array,
    lapl_b: Array,
    ex_dfa_total: Array,
    exx_a: Array,
    exx_b: Array,
    *,
    p1: float,
    p2: float,
    d: float,
    lamb: float = _DLDH_QAC_LAMBDA_DEFAULT,
    bet: float = _DLDH_QAC_BETA_DEFAULT,
) -> Array:
    ex_dfa_a = 0.5 * ex_dfa_total
    ex_dfa_b = 0.5 * ex_dfa_total
    mb_a = _dldh_mb86x(lamb, bet, rho_a, sigma_aa, tau_a, lapl_a)
    mb_b = _dldh_mb86x(lamb, bet, rho_b, sigma_bb, tau_b, lapl_b)
    z = jnp.maximum((mb_a + mb_b) / (exx_a + exx_b - 1.0e-9) - 1.0, 0.0)
    bb = -4.0 * jnp.log(3.0) / jnp.log(jnp.asarray(p1) / jnp.asarray(p2))
    aa = 9.0 * jnp.asarray(p2) ** (-bb)
    tmp = aa * z**bb / (1.0 + aa * z**bb)
    return 0.5 + jnp.asarray(d, dtype=tmp.dtype) * tmp / 2.0


def _dldh_qac_erf(
    rho_a: Array,
    rho_b: Array,
    sigma_aa: Array,
    sigma_bb: Array,
    tau_a: Array,
    tau_b: Array,
    lapl_a: Array,
    lapl_b: Array,
    ex_dfa_total: Array,
    exx_a: Array,
    exx_b: Array,
    *,
    a: float,
    b: float,
    lamb: float = _DLDH_QAC_LAMBDA_DEFAULT,
    bet: float = _DLDH_QAC_BETA_DEFAULT,
) -> Array:
    ex_dfa_a = 0.5 * ex_dfa_total
    ex_dfa_b = 0.5 * ex_dfa_total
    mb_a = _dldh_mb86x(lamb, bet, rho_a, sigma_aa, tau_a, lapl_a)
    mb_b = _dldh_mb86x(lamb, bet, rho_b, sigma_bb, tau_b, lapl_b)
    zorg = jnp.maximum((mb_a + mb_b) / (exx_a + exx_b - 1.0e-9) - 1.0, 0.0)
    erfz = jsp_special.erf(12.0 * (zorg - jnp.asarray(a, dtype=zorg.dtype)))
    z = zorg * jnp.maximum(erfz, 0.0)
    return 0.5 + 0.5 * jsp_special.erf(jnp.asarray(b, dtype=z.dtype) * z)

# GradDFT-aligned defaults for the DM21-style Neural_xc backbone.
GRADDFT_DEFAULT_INPUT_FEATURE_MODE: Literal["enhanced", "dm21_original"] = "dm21_original"
GRADDFT_DEFAULT_NETWORK_ARCHITECTURE: Literal["simple_mlp", "graddft_residual"] = (
    "graddft_residual"
)
GRADDFT_DEFAULT_DM21_HIDDEN_DIMS: tuple[int, ...] = (
    256,
    256,
    256,
    256,
    256,
    256,
)


@dataclass(frozen=True)
class SemilocalEnergyDensityModule:
    """Pluggable non-HF local energy-density module for Neural_xc.

    The module returns semilocal exchange/correlation local contribution channels
    on the grid. Each channel is already in the form that can be multiplied
    directly by quadrature weights.
    """

    channel_names: tuple[str, ...]
    energy_density_channels_fn: SemilocalEnergyDensityFn
    local_contribution_fn: SemilocalLocalContributionFn | None = None
    name: str = "semilocal_module"

    @property
    def n_channels(self) -> int:
        return len(self.channel_names)

    def energy_density_channels(self, features: RestrictedFeatureBundle) -> Array:
        channels = jnp.asarray(self.energy_density_channels_fn(features))
        channels = jnp.nan_to_num(channels, nan=0.0, posinf=0.0, neginf=0.0)
        if channels.ndim == features.rho.ndim:
            channels = channels[..., None]
        elif channels.ndim != features.rho.ndim + 1:
            raise ValueError(
                "SemilocalEnergyDensityModule must return shape (...,) or (..., n_channels)."
            )
        if channels.shape[-1] != self.n_channels:
            raise ValueError(
                "SemilocalEnergyDensityModule output channel count does not match "
                f"channel_names (got {channels.shape[-1]} vs {self.n_channels})."
            )
        return channels

    def energy_density(self, features: RestrictedFeatureBundle) -> Array:
        return jnp.sum(self.energy_density_channels(features), axis=-1)

    def local_contribution_channels(
        self,
        features: RestrictedFeatureBundle,
        *,
        channels: Array | None = None,
        density_floor: float = 1e-12,
    ) -> Array:
        channel_values = (
            self.energy_density_channels(features) if channels is None else jnp.asarray(channels)
        )
        if self.local_contribution_fn is not None:
            return jnp.asarray(
                self.local_contribution_fn(features, channel_values, float(density_floor))
            )
        del features, density_floor
        return channel_values


def available_semilocal_components() -> tuple[str, ...]:
    """Return built-in exchange/correlation channel names available in pure JAX."""

    return tuple(COMMON_SEMILOCAL_COMPONENT_SPECS.keys())


def make_libxc_semilocal_module(
    channel_specs: str | Sequence[str],
    *,
    channel_names: Sequence[str] | None = None,
    name: str = "libxc_semilocal_module",
    allow_experimental_jax_xc: bool = False,
) -> SemilocalEnergyDensityModule:
    specs = _normalize_semilocal_xc_names(
        channel_specs,
        allow_experimental_jax_xc=allow_experimental_jax_xc,
    )
    resolved_specs = tuple(COMMON_SEMILOCAL_COMPONENT_SPECS.get(spec, spec) for spec in specs)
    for spec in resolved_specs:
        parse_xc(spec, allow_experimental_jax_xc=allow_experimental_jax_xc)
    names = specs if channel_names is None else tuple(str(label) for label in channel_names)
    if len(names) != len(resolved_specs):
        raise ValueError("channel_names must match the number of semilocal channel specs.")

    def energy_density_channels_fn(features: RestrictedFeatureBundle) -> Array:
        return jnp.stack(
            [
                eval_xc_energy_density(
                    spec,
                    features,
                    allow_experimental_jax_xc=allow_experimental_jax_xc,
                )
                for spec in resolved_specs
            ],
            axis=-1,
        )

    return SemilocalEnergyDensityModule(
        channel_names=tuple(names),
        energy_density_channels_fn=energy_density_channels_fn,
        name=name,
    )


def make_custom_semilocal_module(
    *,
    channel_names: Sequence[str],
    energy_density_channels_fn: SemilocalEnergyDensityFn,
    local_contribution_fn: SemilocalLocalContributionFn | None = None,
    name: str = "custom_semilocal_module",
) -> SemilocalEnergyDensityModule:
    names = tuple(str(label) for label in channel_names)
    if not names:
        raise ValueError("channel_names must contain at least one semilocal component.")
    return SemilocalEnergyDensityModule(
        channel_names=names,
        energy_density_channels_fn=energy_density_channels_fn,
        local_contribution_fn=local_contribution_fn,
        name=name,
    )


def _legacy_semilocal_module(
    semilocal_xc: str | Sequence[str],
    semilocal_energy_density_fn: SemilocalEnergyDensityFn | None,
    *,
    n_semilocal_channels: int | None = None,
    allow_experimental_jax_xc: bool = False,
) -> SemilocalEnergyDensityModule:
    if semilocal_energy_density_fn is not None:
        if n_semilocal_channels is None:
            channel_names = ("custom_semilocal",)
        else:
            channel_names = tuple(
                f"custom_semilocal_{idx + 1}" for idx in range(int(n_semilocal_channels))
            )
        return make_custom_semilocal_module(
            channel_names=channel_names,
            energy_density_channels_fn=semilocal_energy_density_fn,
            name="legacy_custom_semilocal_module",
        )
    return make_libxc_semilocal_module(
        semilocal_xc,
        name="legacy_libxc_semilocal_module",
        allow_experimental_jax_xc=allow_experimental_jax_xc,
    )


@dataclass(frozen=True)
class DM21LikeFunctional:
    r"""Strict DM21-style Neural_xc with semilocal + HF local channels.

    The functional uses one shared MLP to produce local mixing coefficients and
    applies them to semilocal, projected-HF, and optional projected-PT2 channels.

    E_xc[n] = \int w(r) * e_xc(r) dr

    where
      e_xc(r) = sum_k c_k(r) * e_k(r)

    and each basis channel e_k(r) is itself a local XC grid contribution
    (energy density per volume element), not a per-particle epsilon_xc term.
    """

    model: nn.Module
    non_hf_module: SemilocalEnergyDensityModule | None = None
    semilocal_xc: str | Sequence[str] = DEFAULT_NEURAL_XC_SEMILOCAL_XC
    semilocal_energy_density_fn: SemilocalEnergyDensityFn | None = None
    energy_mode: Literal[
        "graddft_coeff_basis",
        "normalized_mixing_basis",
        "dldh_two_lmf",
        "graddft_coeff_basis_hf_pt2_heads",
    ] = DEFAULT_NEURAL_XC_ENERGY_MODE
    input_feature_mode: Literal["enhanced", "dm21_original"] = "enhanced"
    hf_input_mode: Literal["total_only", "spin_resolved"] = "spin_resolved"
    hf_fraction_mode: Literal["normalized_weights", "hf_coefficient"] = (
        "normalized_weights"
    )
    include_pt2_channel: bool = False
    pt2_channel_mode: Literal["scaled_projected", "local_exact"] = "scaled_projected"
    dldh_range_separated_exchange: bool = False
    dldh_range_separation_omega: float = _DLDH_RS_OMEGA_DEFAULT
    dldh_qac_mode: Literal["none", "pade", "erf"] = "none"
    dldh_qac_parameters: tuple[float, ...] = ()
    response_hf_mode: Literal["nonlocal_exchange_only", "local_projected"] = (
        "nonlocal_exchange_only"
    )
    response_pt2_mode: Literal["nonlocal_correlation_only", "local_projected"] = (
        "local_projected"
    )
    strict_dm21_feature_alignment: bool = True
    allow_experimental_jax_xc: bool = False
    density_floor: float = 1e-12
    response_density_floor: float | None = None
    kernel_clip: float = 5.0
    response_kernel_clip: float | None = 5.0
    name: str = "neural_xc"
    dm21_hfx_channels: int = 2
    # GradDFT compatibility fields.
    is_xc: bool = True
    exchange_mask: Array | None = None

    def _mlp_functional(self) -> NeuralXCFunctional:
        return NeuralXCFunctional(
            model=self.model,
            coefficient_transform_fn=self._sanitize_coefficients,
            name=self.name,
        )

    def _effective_response_density_floor(self) -> float:
        response_floor = self.density_floor
        if self.response_density_floor is not None:
            response_floor = max(response_floor, float(self.response_density_floor))
        return response_floor

    def resolved_non_hf_module(self) -> SemilocalEnergyDensityModule:
        if self.non_hf_module is not None:
            return self.non_hf_module
        return _legacy_semilocal_module(
            self.semilocal_xc,
            self.semilocal_energy_density_fn,
            allow_experimental_jax_xc=self.allow_experimental_jax_xc,
        )

    def _maybe_clip_response(self, values: Array) -> Array:
        clip = self.response_kernel_clip
        if clip is None:
            return values
        clip_value = float(clip)
        if clip_value <= 0.0:
            return values
        return jnp.clip(values, -clip_value, clip_value)

    def _uses_dldh_aux_exchange_branch(self) -> bool:
        return self.energy_mode == "dldh_two_lmf" and (
            bool(self.dldh_range_separated_exchange) or str(self.dldh_qac_mode).lower() != "none"
        )

    def _uses_dldh_laplacian_branch(self) -> bool:
        return self.energy_mode == "dldh_two_lmf" and str(self.dldh_qac_mode).lower() != "none"

    def _uses_explicit_hf_pt2_heads(self) -> bool:
        return self.energy_mode in {"dldh_two_lmf", "graddft_coeff_basis_hf_pt2_heads"}

    def _response_feature_kind_label(self) -> str:
        return "MGGA_LAPL" if self._uses_dldh_laplacian_branch() else "MGGA"

    def _dldh_long_range_hf_grid_contribution_components(
        self,
        molecule: Any,
        *,
        omega: float,
    ) -> tuple[Array, Array, Array]:
        hfx_local = getattr(molecule, "hfx_local", None)
        omega_values = getattr(molecule, "hfx_omega_values", None)
        if hfx_local is not None and omega_values is not None:
            idx = next(
                (i for i, value in enumerate(tuple(float(v) for v in omega_values)) if abs(value - float(omega)) < 1e-8),
                None,
            )
            if idx is not None:
                hfx_local = jnp.asarray(hfx_local)
                exx_a = jnp.asarray(hfx_local[0, :, idx])
                exx_b = jnp.asarray(hfx_local[1, :, idx])
                return exx_a + exx_b, exx_a, exx_b

        if getattr(molecule, "hfx_nu", None) is None:
            raise AttributeError(
                "Range-separated DLDH branch requires either hfx_local with a matching omega "
                "channel or hfx_nu for interpolation."
            )
        ao = jnp.asarray(molecule.ao)
        nu = jnp.asarray(molecule.hfx_nu)
        omega_values = jnp.asarray(getattr(molecule, "hfx_omega_values"), dtype=nu.dtype)
        target = jnp.clip(jnp.asarray(omega, dtype=nu.dtype), omega_values[0], omega_values[-1])
        if nu.shape[0] == 1:
            nu_interp = nu[0]
        else:
            upper = jnp.clip(jnp.sum(omega_values <= target), 1, omega_values.shape[0] - 1)
            lower = upper - 1
            w0 = omega_values[lower]
            w1 = omega_values[upper]
            frac = (target - w0) / jnp.maximum(w1 - w0, 1e-8)
            nu_interp = nu[lower] + frac * (nu[upper] - nu[lower])
        dm_a, dm_b = self._restricted_spin_density_blocks(molecule)
        e_a = jnp.einsum("gp,pq->gq", ao, dm_a, precision=Precision.HIGHEST)
        e_b = jnp.einsum("gp,pq->gq", ao, dm_b, precision=Precision.HIGHEST)
        fxx_a = jnp.einsum("gbc,gc->gb", nu_interp, e_a, precision=Precision.HIGHEST)
        fxx_b = jnp.einsum("gbc,gc->gb", nu_interp, e_b, precision=Precision.HIGHEST)
        exx_a = -0.5 * jnp.einsum("gq,gq->g", e_a, fxx_a, precision=Precision.HIGHEST)
        exx_b = -0.5 * jnp.einsum("gq,gq->g", e_b, fxx_b, precision=Precision.HIGHEST)
        exx_a = jnp.nan_to_num(exx_a, nan=0.0, posinf=0.0, neginf=0.0)
        exx_b = jnp.nan_to_num(exx_b, nan=0.0, posinf=0.0, neginf=0.0)
        return exx_a + exx_b, exx_a, exx_b

    def _dldh_short_range_hf_grid_contribution_components(
        self,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
    ) -> tuple[Array, Array, Array]:
        full_total, full_a, full_b = self.projected_hf_grid_contribution_components(
            molecule,
            features=features,
        )
        lr_total, lr_a, lr_b = self._dldh_long_range_hf_grid_contribution_components(
            molecule,
            omega=float(self.dldh_range_separation_omega),
        )
        sr_a = full_a - lr_a
        sr_b = full_b - lr_b
        sr_total = full_total - lr_total
        return sr_total, sr_a, sr_b

    def _dldh_qac_factor(
        self,
        *,
        rho: Array,
        grad: Array,
        tau: Array,
        lapl: Array,
        ex_dfa_total: Array,
        exx_a: Array,
        exx_b: Array,
    ) -> Array:
        mode = str(self.dldh_qac_mode).lower()
        if mode == "none":
            return jnp.full_like(rho, 0.5)
        rho_half = 0.5 * rho
        sigma_half = 0.25 * jnp.einsum("...x,...x->...", grad, grad)
        tau_half = 0.5 * tau
        lapl_half = 0.5 * lapl
        if mode == "pade":
            if len(self.dldh_qac_parameters) != 3:
                raise ValueError("dldh_qac_mode='pade' requires three qac parameters (p1, p2, d).")
            p1, p2, d = (float(v) for v in self.dldh_qac_parameters)
            return _dldh_qac_pade(
                rho_half,
                rho_half,
                sigma_half,
                sigma_half,
                tau_half,
                tau_half,
                lapl_half,
                lapl_half,
                ex_dfa_total,
                exx_a,
                exx_b,
                p1=p1,
                p2=p2,
                d=d,
            )
        if mode == "erf":
            if len(self.dldh_qac_parameters) != 2:
                raise ValueError("dldh_qac_mode='erf' requires two qac parameters (a, b).")
            a, b = (float(v) for v in self.dldh_qac_parameters)
            return _dldh_qac_erf(
                rho_half,
                rho_half,
                sigma_half,
                sigma_half,
                tau_half,
                tau_half,
                lapl_half,
                lapl_half,
                ex_dfa_total,
                exx_a,
                exx_b,
                a=a,
                b=b,
            )
        raise ValueError(f"Unsupported dldh_qac_mode={self.dldh_qac_mode!r}.")

    def _restricted_spin_density_blocks(self, molecule: Any) -> Array:
        if getattr(molecule, "rdm1", None) is None:
            raise AttributeError("Molecule-like object must define rdm1.")
        rdm1 = jnp.asarray(molecule.rdm1)
        if rdm1.ndim == 2:
            return jnp.stack([0.5 * rdm1, 0.5 * rdm1], axis=0)
        if rdm1.ndim != 3:
            raise ValueError(
                "Restricted HF/PT2 channels expect rdm1 to have shape "
                "(nao, nao) or (spin, nao, nao)."
            )
        if rdm1.shape[0] == 1:
            return jnp.concatenate([rdm1, rdm1], axis=0)
        if rdm1.shape[0] != 2:
            raise ValueError(
                "Restricted HF/PT2 channels expect one or two spin blocks in rdm1."
            )
        return rdm1

    def _dm21_exact_hf_grid_contribution_components(
        self,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
    ) -> tuple[Array, Array, Array]:
        del features
        hfx_local = getattr(molecule, "hfx_local", None)
        if hfx_local is not None:
            hfx_local = jnp.asarray(hfx_local)
        else:
            if getattr(molecule, "hfx_nu", None) is None:
                raise AttributeError(
                    "DM21 exact HF channel requires molecule.hfx_local or molecule.hfx_nu."
                )
            if getattr(molecule, "ao", None) is None:
                raise AttributeError("Molecule-like object must define ao.")
            ao = jnp.asarray(molecule.ao)
            nu = jnp.asarray(molecule.hfx_nu)
            if nu.ndim != 4:
                raise ValueError(
                    "molecule.hfx_nu must have shape (n_omega, ngrids, nao, nao), "
                    f"got {nu.shape}."
                )
            dm_a, dm_b = self._restricted_spin_density_blocks(molecule)
            e_a = jnp.einsum(
                "rp,pq->rq",
                ao,
                dm_a,
                precision=Precision.HIGHEST,
            )
            e_b = jnp.einsum(
                "rp,pq->rq",
                ao,
                dm_b,
                precision=Precision.HIGHEST,
            )
            fxx_a = jnp.einsum("wgbc,gc->wgb", nu, e_a, precision=Precision.HIGHEST)
            fxx_b = jnp.einsum("wgbc,gc->wgb", nu, e_b, precision=Precision.HIGHEST)
            exx_a = -0.5 * jnp.einsum("gq,wgq->wg", e_a, fxx_a, precision=Precision.HIGHEST)
            exx_b = -0.5 * jnp.einsum("gq,wgq->wg", e_b, fxx_b, precision=Precision.HIGHEST)
            hfx_local = jnp.stack([exx_a.T, exx_b.T], axis=0)

        if hfx_local.ndim != 3 or hfx_local.shape[0] != 2:
            raise ValueError(
                "DM21 exact HF channel expects molecule.hfx_local with shape "
                "(2, ngrids, n_omega)."
            )
        e_hf_a = jnp.asarray(hfx_local[0, :, 0])
        e_hf_b = jnp.asarray(hfx_local[1, :, 0])
        e_hf = e_hf_a + e_hf_b
        e_hf = jnp.nan_to_num(e_hf, nan=0.0, posinf=0.0, neginf=0.0)
        e_hf_a = jnp.nan_to_num(e_hf_a, nan=0.0, posinf=0.0, neginf=0.0)
        e_hf_b = jnp.nan_to_num(e_hf_b, nan=0.0, posinf=0.0, neginf=0.0)
        return e_hf, e_hf_a, e_hf_b

    def _semilocal_local_contribution_channels(
        self,
        features: RestrictedFeatureBundle,
        semilocal_channels: Array,
    ) -> Array:
        return self.resolved_non_hf_module().local_contribution_channels(
            features,
            channels=semilocal_channels,
            density_floor=self.density_floor,
        )

    def _as_descriptor(self, local_contribution: Array, density: Array) -> Array:
        density = jnp.maximum(jnp.asarray(density), self.density_floor)
        return jnp.nan_to_num(
            jnp.asarray(local_contribution) / density,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def _semilocal_input_descriptor(
        self,
        features: RestrictedFeatureBundle,
        semilocal_energy_density: Array,
    ) -> Array:
        return self._as_descriptor(semilocal_energy_density, features.rho)

    def _assemble_basis_channels(
        self,
        semilocal_local_channels: Array,
        *,
        hf_projected: Array,
        pt2_projected: Array | None = None,
    ) -> Array:
        if self.energy_mode == "dldh_two_lmf":
            semilocal_x, semilocal_c = self._split_semilocal_exchange_correlation_local_channels(
                semilocal_local_channels
            )
            pt2_channel = (
                jnp.zeros_like(hf_projected)
                if pt2_projected is None
                else jnp.asarray(pt2_projected)
            )
            return jnp.stack(
                [semilocal_x, semilocal_c, pt2_channel, jnp.asarray(hf_projected)],
                axis=-1,
            )
        channels = [jnp.asarray(semilocal_local_channels)]
        if self.include_pt2_channel:
            if pt2_projected is None:
                raise ValueError("pt2_projected must be provided when include_pt2_channel=True.")
            channels.append(jnp.asarray(pt2_projected)[..., None])
        channels.append(jnp.asarray(hf_projected)[..., None])
        return jnp.concatenate(channels, axis=-1)

    def init(self, rng: PRNGKeyArray, sample_inputs: Array) -> PyTree:
        return self._mlp_functional().init(rng, sample_inputs)

    def init_from_molecule(self, rng: PRNGKeyArray, molecule: Any) -> PyTree:
        features = restricted_grid_features(molecule)
        semilocal = self.semilocal_energy_density(features)
        hf_projected, hf_projected_a, hf_projected_b = self.projected_hf_grid_contribution_components(
            molecule,
            features=features,
        )
        pt2_projected = (
            self.projected_pt2_grid_contribution(molecule, features=features)
            if self.include_pt2_channel
            else None
        )
        hf_spin_inputs: tuple[Array, Array] | None = (hf_projected_a, hf_projected_b)
        if (
            self.input_feature_mode == "dm21_original"
            and self.strict_dm21_feature_alignment
            and getattr(molecule, "hfx_local", None) is None
        ):
            hf_spin_inputs = None
        inputs = self.coefficient_inputs(
            features,
            semilocal,
            hf_projected,
            pt2_energy_density=pt2_projected,
            molecule=molecule,
            hf_spin_energy_density=hf_spin_inputs,
        )
        return self.init(rng, inputs)

    def compute_densities(
        self,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
    ) -> Array:
        """GradDFT-compatible basis-channel builder e_k(r).

        Returns local grid-contribution channels with shape (..., n_channels):
        [semilocal_1, ..., semilocal_n, pt2_projected?, hf_projected].
        """

        if self._uses_dldh_aux_exchange_branch():
            raise ValueError(
                "compute_densities is not defined for dldh_two_lmf with range separation or QAC "
                "because the exchange assembly depends on additional molecule-resolved channels."
            )
        if features is None:
            features = restricted_grid_features(molecule)
        semilocal_channels = self.semilocal_energy_density_channels(features)
        semilocal_local_channels = self._semilocal_local_contribution_channels(
            features,
            semilocal_channels,
        )
        hf_projected, _, _ = self.projected_hf_grid_contribution_components(
            molecule,
            features=features,
        )
        pt2_projected = (
            self.projected_pt2_grid_contribution(molecule, features=features)
            if self.include_pt2_channel
            else None
        )
        return self._assemble_basis_channels(
            semilocal_local_channels,
            hf_projected=hf_projected,
            pt2_projected=pt2_projected,
        )

    def compute_coefficient_inputs(
        self,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
        semilocal_energy_density: Array | None = None,
        hf_energy_density: Array | None = None,
        pt2_energy_density: Array | None = None,
        hf_spin_energy_density: tuple[Array, Array] | None = None,
    ) -> Array:
        """GradDFT-compatible input feature builder for c_theta."""

        if features is None:
            features = restricted_grid_features(molecule)
        semilocal = (
            self.semilocal_energy_density(features)
            if semilocal_energy_density is None
            else jnp.asarray(semilocal_energy_density)
        )
        if hf_energy_density is None:
            hf_projected, hf_projected_a, hf_projected_b = self.projected_hf_grid_contribution_components(
                molecule,
                features=features,
            )
        else:
            hf_projected = jnp.asarray(hf_energy_density)
            if hf_spin_energy_density is None:
                hf_projected_a = hf_projected
                hf_projected_b = hf_projected
            else:
                hf_projected_a, hf_projected_b = hf_spin_energy_density
        spin_inputs = (
            hf_spin_energy_density
            if hf_spin_energy_density is not None
            else (hf_projected_a, hf_projected_b)
        )
        if pt2_energy_density is None and self.include_pt2_channel:
            pt2_energy_density = self.projected_pt2_grid_contribution(
                molecule,
                features=features,
            )
        return self.coefficient_inputs(
            features,
            semilocal,
            hf_projected,
            pt2_energy_density=pt2_energy_density,
            molecule=molecule,
            hf_spin_energy_density=spin_inputs,
        )

    def xc_energy(
        self,
        params: PyTree,
        grid: Any,
        coefficient_inputs: Array,
        densities: Array,
        **_: Any,
    ) -> Array:
        """GradDFT-style XC quadrature from prebuilt inputs/channels."""
        if self._uses_dldh_aux_exchange_branch():
            raise ValueError(
                "xc_energy from prebuilt densities is not defined for dldh_two_lmf with "
                "range separation or QAC because the exchange assembly depends on "
                "molecule-resolved short-range HF/QAC payloads."
            )
        weights = jnp.asarray(getattr(grid, "weights", grid))
        basis = jnp.asarray(densities)
        if basis.ndim == 1:
            basis = basis[:, None]
        local_channels = self._assemble_channel_contributions(
            self.channel_coefficients_from_inputs(params, coefficient_inputs),
            basis,
        )
        return jnp.nan_to_num(
            jnp.tensordot(weights, jnp.sum(local_channels, axis=-1), axes=(0, 0)),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def semilocal_energy_density_channels(self, features: RestrictedFeatureBundle) -> Array:
        return self.resolved_non_hf_module().energy_density_channels(features)

    def semilocal_energy_density(self, features: RestrictedFeatureBundle) -> Array:
        channels = self.semilocal_energy_density_channels(features)
        return jnp.sum(channels, axis=-1)

    def _dm21_hfx_feature_channels(
        self,
        molecule: Any | None,
        features: RestrictedFeatureBundle,
        *,
        hf_energy_density: Array | None = None,
        hf_spin_energy_density: tuple[Array, Array] | None = None,
    ) -> tuple[Array, Array]:
        target_channels = max(int(self.dm21_hfx_channels), 1)
        cached = getattr(molecule, "hfx_local", None) if molecule is not None else None
        if cached is not None:
            cached = jnp.asarray(cached)
            if cached.ndim == 3 and cached.shape[0] == 2:
                if self.strict_dm21_feature_alignment and cached.shape[-1] != target_channels:
                    raise ValueError(
                        "molecule.hfx_local omega-channel count must match dm21_hfx_channels "
                        f"(got {cached.shape[-1]} vs {target_channels})."
                    )
                return cached[0], cached[1]
            raise ValueError(
                "molecule.hfx_local must have shape (2, ngrids, n_omega), "
                f"got {cached.shape}."
            )

        if hf_spin_energy_density is not None:
            hfx_a = jnp.asarray(hf_spin_energy_density[0])
            hfx_b = jnp.asarray(hf_spin_energy_density[1])
            if hfx_a.ndim == features.rho.ndim:
                hfx_a = hfx_a[..., None]
            if hfx_b.ndim == features.rho.ndim:
                hfx_b = hfx_b[..., None]
            if hfx_a.shape[-1] == 1 and target_channels > 1:
                hfx_a = jnp.repeat(hfx_a, target_channels, axis=-1)
            if hfx_b.shape[-1] == 1 and target_channels > 1:
                hfx_b = jnp.repeat(hfx_b, target_channels, axis=-1)
            return hfx_a, hfx_b

        if self.strict_dm21_feature_alignment:
            raise ValueError(
                "DM21 original input mode requires molecule.hfx_local with shape "
                "(2, ngrids, n_omega), or explicit hf_spin_energy_density channels. "
                "Build the reference with compute_local_hfx_features=True "
                "(typically omega values 0.0 and 0.4)."
            )

        # Fallback: build a degenerate DM21-HFX channel stack from the projected HF
        # energy density if explicit multi-omega local HF features are unavailable.
        hf_total = jnp.zeros_like(features.rho) if hf_energy_density is None else jnp.asarray(hf_energy_density)
        local_hfx = hf_total
        n_channels = max(int(self.dm21_hfx_channels), 1)
        local_hfx = jnp.repeat(local_hfx[..., None], n_channels, axis=-1)
        return local_hfx, local_hfx

    def coefficient_inputs(
        self,
        features: RestrictedFeatureBundle,
        semilocal_energy_density: Array,
        hf_energy_density: Array,
        *,
        pt2_energy_density: Array | None = None,
        molecule: Any | None = None,
        hf_spin_energy_density: tuple[Array, Array] | None = None,
    ) -> Array:
        pt2_total = (
            jnp.zeros_like(features.rho)
            if pt2_energy_density is None
            else jnp.asarray(pt2_energy_density)
        )
        if self.input_feature_mode == "dm21_original":
            hfx_a, hfx_b = self._dm21_hfx_feature_channels(
                molecule,
                features,
                hf_energy_density=hf_energy_density,
                hf_spin_energy_density=hf_spin_energy_density,
            )
            base = dm21_original_input_features(
                features,
                hfx_a,
                hfx_b,
                density_floor=self.density_floor,
            )
            if not self.include_pt2_channel:
                return base
            return jnp.concatenate([base, pt2_total[..., None]], axis=-1)
        if self.input_feature_mode != "enhanced":
            raise ValueError(
                f"Unsupported input_feature_mode={self.input_feature_mode!r}. "
                "Expected 'enhanced' or 'dm21_original'."
            )
        semilocal_descriptor = self._semilocal_input_descriptor(
            features,
            semilocal_energy_density,
        )
        base = dm21_like_input_features(
            features,
            semilocal_descriptor,
            density_floor=self.density_floor,
        )
        hf_total = jnp.asarray(hf_energy_density)
        if self.hf_input_mode == "total_only":
            extras = [hf_total[..., None]]
        elif self.hf_input_mode == "spin_resolved":
            if hf_spin_energy_density is None:
                hf_a = hf_total
                hf_b = hf_total
            else:
                hf_a, hf_b = hf_spin_energy_density
            extras = [hf_total[..., None], hf_a[..., None], hf_b[..., None]]
        else:
            raise ValueError(
                f"Unsupported hf_input_mode={self.hf_input_mode!r}. "
                "Expected 'total_only' or 'spin_resolved'."
            )
        if self.include_pt2_channel:
            extras.append(pt2_total[..., None])
        return jnp.concatenate([base, *extras], axis=-1)

    def channel_coefficients(
        self,
        params: PyTree,
        features: RestrictedFeatureBundle,
        *,
        molecule: Any | None = None,
        semilocal_energy_density: Array | None = None,
        hf_energy_density: Array | None = None,
        pt2_energy_density: Array | None = None,
        hf_spin_energy_density: tuple[Array, Array] | None = None,
    ) -> Array:
        semilocal = (
            self.semilocal_energy_density(features)
            if semilocal_energy_density is None
            else semilocal_energy_density
        )
        hf_projected = (
            jnp.zeros_like(semilocal) if hf_energy_density is None else hf_energy_density
        )
        inputs = self.coefficient_inputs(
            features,
            semilocal,
            hf_projected,
            pt2_energy_density=pt2_energy_density,
            molecule=molecule,
            hf_spin_energy_density=hf_spin_energy_density,
        )
        return self.channel_coefficients_from_inputs(params, inputs)

    def channel_coefficients_from_inputs(
        self,
        params: PyTree,
        coefficient_inputs: Array,
    ) -> Array:
        return self._mlp_functional().coefficients(params, coefficient_inputs)

    def _dldh_unit_interval_coefficients(self, coefficients: Array) -> Array:
        """Map DLDH two-LMF channel outputs smoothly into the unit interval.

        The current DLDH heads use a positive sigmoid-scaled output layer
        `scale * sigmoid(raw / scale)`. Re-normalizing by that same `scale`
        keeps the mapping smooth and avoids the hard dead-zone introduced by a
        post-hoc clip to `[0, 1]`.
        """

        scale = float(getattr(self.model, "sigmoid_scale_factor", 0.0))
        if scale > 0.0:
            safe = jnp.nan_to_num(coefficients, nan=0.0, posinf=scale, neginf=0.0)
            return safe / scale
        safe = jnp.nan_to_num(coefficients, nan=0.0, posinf=60.0, neginf=-60.0)
        return jax.nn.sigmoid(safe)

    def _sanitize_coefficients(self, coefficients: Array) -> Array:
        if self.energy_mode == "dldh_two_lmf":
            return self._dldh_unit_interval_coefficients(coefficients)
        if self.energy_mode == "graddft_coeff_basis_hf_pt2_heads":
            safe = jnp.nan_to_num(coefficients, nan=0.0, posinf=0.0, neginf=0.0)
            n_semilocal = int(self.resolved_non_hf_module().n_channels)
            expected = n_semilocal + 1 + int(bool(self.include_pt2_channel))
            if safe.shape[-1] != expected:
                raise ValueError(
                    "graddft_coeff_basis_hf_pt2_heads expects "
                    f"{expected} outputs, got {safe.shape[-1]}."
                )
            semilocal = jnp.clip(safe[..., :n_semilocal], 0.0, self.kernel_clip)
            cursor = n_semilocal
            heads: list[Array] = []
            if self.include_pt2_channel:
                heads.append(
                    self._dldh_unit_interval_coefficients(
                        safe[..., cursor : cursor + 1]
                    )
                )
                cursor += 1
            heads.append(
                self._dldh_unit_interval_coefficients(
                    safe[..., cursor : cursor + 1]
                )
            )
            return jnp.concatenate([semilocal, *heads], axis=-1)
        coefficients = jnp.nan_to_num(coefficients, nan=0.0, posinf=0.0, neginf=0.0)
        return jnp.clip(coefficients, 0.0, self.kernel_clip)

    def mixing_logits(
        self,
        params: PyTree,
        features: RestrictedFeatureBundle,
        *,
        semilocal_energy_density: Array | None = None,
        hf_energy_density: Array | None = None,
    ) -> Array:
        # Backward-compatible alias retained for existing callers/tests.
        return self.channel_coefficients(
            params,
            features,
            semilocal_energy_density=semilocal_energy_density,
            hf_energy_density=hf_energy_density,
        )

    def mixing_weights(
        self,
        params: PyTree,
        features: RestrictedFeatureBundle,
        *,
        semilocal_energy_density: Array | None = None,
        hf_energy_density: Array | None = None,
    ) -> Array:
        coefficients = self.channel_coefficients(
            params,
            features,
            semilocal_energy_density=semilocal_energy_density,
            hf_energy_density=hf_energy_density,
        )
        if self._uses_explicit_hf_pt2_heads():
            return coefficients
        normalizer = jnp.maximum(
            jnp.sum(coefficients, axis=-1, keepdims=True),
            self.density_floor,
        )
        weights = coefficients / normalizer
        weights = jnp.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=0.0)
        weights = jnp.clip(weights, 0.0, 1.0)
        return weights

    def _local_hf_fraction_from_coefficients(self, coefficients: Array) -> Array:
        if self.energy_mode == "dldh_two_lmf":
            return jnp.nan_to_num(coefficients[..., 0], nan=0.0, posinf=1.0, neginf=0.0)
        if self.energy_mode == "graddft_coeff_basis_hf_pt2_heads":
            return jnp.nan_to_num(coefficients[..., -1], nan=0.0, posinf=1.0, neginf=0.0)
        hf_channel = coefficients[..., -1]
        if self.hf_fraction_mode == "normalized_weights":
            normalizer = jnp.maximum(
                jnp.sum(coefficients, axis=-1, keepdims=True),
                self.density_floor,
            )
            hf_field = hf_channel / jnp.squeeze(normalizer, axis=-1)
        elif self.hf_fraction_mode == "hf_coefficient":
            hf_field = hf_channel
        else:
            raise ValueError(
                f"Unsupported hf_fraction_mode={self.hf_fraction_mode!r}. "
                "Expected 'normalized_weights' or 'hf_coefficient'."
            )
        hf_field = jnp.nan_to_num(hf_field, nan=0.0, posinf=1.0, neginf=0.0)
        return jnp.clip(hf_field, 0.0, 1.0)

    def _local_pt2_fraction_from_coefficients(self, coefficients: Array) -> Array:
        if self.energy_mode == "graddft_coeff_basis_hf_pt2_heads" and self.include_pt2_channel:
            return jnp.nan_to_num(coefficients[..., -2], nan=0.0, posinf=1.0, neginf=0.0)
        if self.energy_mode != "dldh_two_lmf" or not self.include_pt2_channel:
            return jnp.zeros(coefficients.shape[:-1], dtype=coefficients.dtype)
        if coefficients.shape[-1] < 2:
            raise ValueError(
                "dldh_two_lmf with include_pt2_channel=True requires two model outputs."
            )
        return jnp.nan_to_num(coefficients[..., 1], nan=0.0, posinf=1.0, neginf=0.0)

    def _resolved_exchange_mask(self) -> Array:
        module = self.resolved_non_hf_module()
        n_channels = int(module.n_channels)
        if self.exchange_mask is not None:
            mask = jnp.asarray(self.exchange_mask, dtype=bool)
            if mask.shape != (n_channels,):
                raise ValueError(
                    "exchange_mask must have shape "
                    f"({n_channels},), got {mask.shape}."
                )
            return mask

        def classify(name: str) -> bool | None:
            label = str(name).strip().lower()
            if label.startswith(("lda_x", "gga_x", "mgga_x", "hyb_x")):
                return True
            if label.startswith(("lda_c", "gga_c", "mgga_c", "hyb_c")):
                return False
            if "exchange" in label and "correlation" not in label:
                return True
            if "correlation" in label and "exchange" not in label:
                return False
            return None

        flags = [classify(name) for name in module.channel_names]
        if any(flag is None for flag in flags):
            unresolved = [
                name for name, flag in zip(module.channel_names, flags, strict=True) if flag is None
            ]
            raise ValueError(
                "dldh_two_lmf requires semilocal channels that can be split into "
                "exchange and correlation parts. Provide component-wise semilocal_xc "
                f"or set exchange_mask explicitly. Unresolved channels: {unresolved!r}."
            )
        if not any(flag is True for flag in flags):
            raise ValueError("dldh_two_lmf requires at least one semilocal exchange channel.")
        if not any(flag is False for flag in flags):
            raise ValueError("dldh_two_lmf requires at least one semilocal correlation channel.")
        return jnp.asarray(flags, dtype=bool)

    def _split_semilocal_exchange_correlation_local_channels(
        self,
        semilocal_local_channels: Array,
    ) -> tuple[Array, Array]:
        channel_values = jnp.asarray(semilocal_local_channels)
        exchange_mask = self._resolved_exchange_mask().astype(channel_values.dtype)
        correlation_mask = 1.0 - exchange_mask
        semilocal_x = jnp.sum(channel_values * exchange_mask, axis=-1)
        semilocal_c = jnp.sum(channel_values * correlation_mask, axis=-1)
        return semilocal_x, semilocal_c

    def _dldh_exchange_components(
        self,
        *,
        molecule: Any | None,
        rho: Array,
        grad: Array,
        tau: Array,
        lapl: Array | None,
        semilocal_x: Array,
        hf_total: Array,
        hf_a: Array,
        hf_b: Array,
        fx: Array,
        exchange_anchor: Array | None = None,
        features: RestrictedFeatureBundle | None = None,
    ) -> tuple[Array, Array]:
        ex_dfa = semilocal_x
        if self.dldh_range_separated_exchange:
            ex_dfa = _dldh_range_sep_dfa_hirao(
                rho,
                ex_dfa,
                omega=float(self.dldh_range_separation_omega),
                density_floor=self.density_floor,
            )
            if exchange_anchor is None:
                if molecule is None:
                    raise ValueError(
                        "Range-separated DLDH point assembly requires an explicit short-range HF anchor."
                    )
                anchor = self._dldh_short_range_hf_grid_contribution_components(
                    molecule,
                    features=features,
                )[0]
            else:
                anchor = jnp.asarray(exchange_anchor)
        else:
            anchor = hf_total
        if self._uses_dldh_laplacian_branch():
            if lapl is None:
                raise ValueError("QAC-enabled DLDH requires laplacian grid values.")
            qac_res = self._dldh_qac_factor(
                rho=rho,
                grad=grad,
                tau=tau,
                lapl=lapl,
                ex_dfa_total=ex_dfa,
                exx_a=hf_a,
                exx_b=hf_b,
            )
        else:
            qac_res = jnp.full_like(ex_dfa, 0.5)
        exchange_residual = 2.0 * qac_res * (1.0 - fx) * (ex_dfa - anchor)
        return exchange_residual, hf_total

    def _assemble_channel_contributions(
        self,
        coefficients: Array,
        basis: Array,
    ) -> Array:
        if self.energy_mode == "dldh_two_lmf":
            if self._uses_dldh_aux_exchange_branch():
                raise ValueError(
                    "dldh_two_lmf with range separation or QAC requires molecule-dependent "
                    "exchange assembly and cannot use the basis-only fast path."
                )
            if coefficients.shape[-1] not in (1, 2):
                raise ValueError(
                    "dldh_two_lmf requires one output (HF only) or two outputs "
                    f"(HF + PT2), got {coefficients.shape[-1]}."
                )
            if basis.shape[-1] != 4:
                raise ValueError(
                    "dldh_two_lmf expects basis channels [x_dfa, c_dfa, pt2, hf], "
                    f"got shape[-1]={basis.shape[-1]}."
                )
            fx = coefficients[..., 0:1]
            if self.include_pt2_channel:
                if coefficients.shape[-1] < 2:
                    raise ValueError(
                        "dldh_two_lmf with include_pt2_channel=True requires two outputs."
                    )
                fpt2 = coefficients[..., 1:2]
            else:
                fpt2 = jnp.zeros_like(fx)
            ex_dfa = basis[..., 0:1]
            ec_dfa = basis[..., 1:2]
            ept2 = basis[..., 2:3]
            ehf = basis[..., 3:4]
            return jnp.concatenate(
                [
                    (1.0 - fx) * ex_dfa,
                    (1.0 - fpt2) * ec_dfa,
                    fpt2 * ept2,
                    fx * ehf,
                ],
                axis=-1,
            )
        if self.energy_mode == "graddft_coeff_basis_hf_pt2_heads":
            n_semilocal = int(self.resolved_non_hf_module().n_channels)
            expected = n_semilocal + 1 + int(bool(self.include_pt2_channel))
            if coefficients.shape[-1] != expected:
                raise ValueError(
                    "graddft_coeff_basis_hf_pt2_heads expects "
                    f"{expected} outputs, got {coefficients.shape[-1]}."
                )
            if basis.shape[-1] != expected:
                raise ValueError(
                    "graddft_coeff_basis_hf_pt2_heads expects basis channels "
                    f"[semilocal..., pt2?, hf], got shape[-1]={basis.shape[-1]}."
                )
            semilocal = coefficients[..., :n_semilocal] * basis[..., :n_semilocal]
            cursor = n_semilocal
            channels = [semilocal]
            if self.include_pt2_channel:
                channels.append(
                    coefficients[..., cursor : cursor + 1] * basis[..., cursor : cursor + 1]
                )
                cursor += 1
            channels.append(
                coefficients[..., cursor : cursor + 1] * basis[..., cursor : cursor + 1]
            )
            return jnp.concatenate(channels, axis=-1)
        if self.energy_mode == "graddft_coeff_basis":
            return coefficients * basis
        if self.energy_mode == "normalized_mixing_basis":
            normalizer = jnp.maximum(
                jnp.sum(coefficients, axis=-1, keepdims=True),
                self.density_floor,
            )
            weights = coefficients / normalizer
            weights = jnp.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=0.0)
            weights = jnp.clip(weights, 0.0, 1.0)
            return weights * basis
        raise ValueError(f"Unsupported energy_mode={self.energy_mode!r}")

    def mixing_fields(
        self,
        params: PyTree,
        features: RestrictedFeatureBundle,
        *,
        semilocal_energy_density: Array | None = None,
        hf_energy_density: Array | None = None,
    ) -> Array:
        # Backward-compatible alias retained for existing callers/tests.
        return self.mixing_weights(
            params,
            features,
            semilocal_energy_density=semilocal_energy_density,
            hf_energy_density=hf_energy_density,
        )

    def projected_hf_grid_contribution_components(
        self,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
    ) -> tuple[Array, Array, Array]:
        return self._dm21_exact_hf_grid_contribution_components(
            molecule,
            features=features,
        )

    def projected_hf_energy_density_components(
        self,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
    ) -> tuple[Array, Array, Array]:
        """Compatibility wrapper returning per-particle HF energy densities.

        DM21 uses local grid contributions directly. This helper exposes the old
        epsilon-style view by dividing out the corresponding spin densities.
        """

        if features is None:
            features = restricted_grid_features(molecule)
        e_hf, e_hf_a, e_hf_b = self.projected_hf_grid_contribution_components(
            molecule,
            features=features,
        )
        rho = jnp.maximum(features.rho, self.density_floor)
        rho_a = jnp.maximum(features.rho_a, self.density_floor)
        rho_b = jnp.maximum(features.rho_b, self.density_floor)
        eps_hf = jnp.nan_to_num(e_hf / rho, nan=0.0, posinf=0.0, neginf=0.0)
        eps_hf_a = jnp.nan_to_num(e_hf_a / rho_a, nan=0.0, posinf=0.0, neginf=0.0)
        eps_hf_b = jnp.nan_to_num(e_hf_b / rho_b, nan=0.0, posinf=0.0, neginf=0.0)
        return eps_hf, eps_hf_a, eps_hf_b

    def projected_hf_energy_density(
        self,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
    ) -> Array:
        eps_hf, _, _ = self.projected_hf_energy_density_components(
            molecule,
            features=features,
        )
        return eps_hf

    def _local_exact_pt2_grid_contribution(
        self,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
        occupation_tolerance: float = 1e-8,
    ) -> Array:
        """Restricted closed-shell MP2 local pair gauge without global rescaling."""
        cached = getattr(molecule, "pt2_local", None)
        if cached is not None:
            cached_arr = jnp.asarray(cached)
            return jnp.nan_to_num(cached_arr, nan=0.0, posinf=0.0, neginf=0.0)

        if getattr(molecule, "rep_tensor", None) is None:
            raise AttributeError("Molecule-like object must define rep_tensor.")
        if getattr(molecule, "mo_coeff", None) is None:
            raise AttributeError("Molecule-like object must define mo_coeff.")
        if getattr(molecule, "mo_occ", None) is None:
            raise AttributeError("Molecule-like object must define mo_occ.")
        if getattr(molecule, "mo_energy", None) is None:
            raise AttributeError("Molecule-like object must define mo_energy.")
        if getattr(molecule, "ao", None) is None:
            raise AttributeError("Molecule-like object must define ao.")
        if getattr(molecule, "grid", None) is None:
            raise AttributeError("Molecule-like object must define grid.weights.")

        if features is None:
            features = restricted_grid_features(molecule)
        del features

        rep_tensor = jnp.asarray(molecule.rep_tensor)
        ao = jnp.asarray(molecule.ao)
        mo_coeff = jnp.asarray(molecule.mo_coeff)
        mo_occ = jnp.asarray(molecule.mo_occ)
        mo_energy = jnp.asarray(molecule.mo_energy)

        if mo_coeff.ndim == 3:
            mo_coeff = mo_coeff[0]
        if mo_occ.ndim == 2:
            mo_occ = mo_occ[0]
        if mo_energy.ndim == 2:
            mo_energy = mo_energy[0]

        nocc = getattr(molecule, "nocc", None)
        if nocc is None:
            nocc = int(jnp.count_nonzero(mo_occ > occupation_tolerance))
        else:
            nocc = int(nocc)
        nmo = int(mo_coeff.shape[1])
        if nocc <= 0 or nocc >= nmo:
            raise ValueError("Restricted MP2 projection requires at least one occupied and one virtual.")

        orbo = mo_coeff[:, :nocc]
        orbv = mo_coeff[:, nocc:]
        eps_occ = mo_energy[:nocc]
        eps_vir = mo_energy[nocc:]

        eri_ovov = getattr(molecule, "eri_ovov", None)
        if eri_ovov is None:
            eri_ovov = jnp.einsum(
                "pqrs,pi,qa,rj,sb->iajb",
                rep_tensor,
                orbo,
                orbv,
                orbo,
                orbv,
                precision=Precision.HIGHEST,
            )
        else:
            eri_ovov = jnp.asarray(eri_ovov)

        denom = (
            eps_occ[:, None, None, None]
            + eps_occ[None, None, :, None]
            - eps_vir[None, :, None, None]
            - eps_vir[None, None, None, :]
        )
        denom = jnp.where(jnp.abs(denom) > self.density_floor, denom, -self.density_floor)
        direct = eri_ovov
        exchange = jnp.transpose(eri_ovov, (0, 3, 2, 1))
        pair_weights = (2.0 * direct - exchange) / denom
        rho_o = jnp.einsum("rp,pi->ri", ao, orbo, precision=Precision.HIGHEST)
        rho_v = jnp.einsum("rp,pa->ra", ao, orbv, precision=Precision.HIGHEST)
        rho_ov = jnp.einsum("ri,ra->ria", rho_o, rho_v, precision=Precision.HIGHEST)
        pair_potential = jnp.einsum(
            "gp,gq,pqrs,rj,sb->gjb",
            ao,
            ao,
            rep_tensor,
            orbo,
            orbv,
            precision=Precision.HIGHEST,
        )
        local_energy = jnp.einsum(
            "ria,rjb,iajb->r",
            rho_ov,
            pair_potential,
            pair_weights,
            precision=Precision.HIGHEST,
        )
        local_energy = jnp.nan_to_num(local_energy, nan=0.0, posinf=0.0, neginf=0.0)
        return local_energy

    def _legacy_projected_pt2_grid_contribution(
        self,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
        occupation_tolerance: float = 1e-8,
    ) -> Array:
        if getattr(molecule, "grid", None) is None:
            raise AttributeError("Molecule-like object must define grid.weights.")
        weights = jnp.asarray(molecule.grid.weights)
        projected = self._local_exact_pt2_grid_contribution(
            molecule,
            features=features,
            occupation_tolerance=occupation_tolerance,
        )
        rep_tensor = jnp.asarray(molecule.rep_tensor)
        mo_coeff = jnp.asarray(molecule.mo_coeff)
        mo_occ = jnp.asarray(molecule.mo_occ)
        mo_energy = jnp.asarray(molecule.mo_energy)
        if mo_coeff.ndim == 3:
            mo_coeff = mo_coeff[0]
        if mo_occ.ndim == 2:
            mo_occ = mo_occ[0]
        if mo_energy.ndim == 2:
            mo_energy = mo_energy[0]
        nocc = getattr(molecule, "nocc", None)
        if nocc is None:
            nocc = int(jnp.count_nonzero(mo_occ > occupation_tolerance))
        else:
            nocc = int(nocc)
        orbo = mo_coeff[:, :nocc]
        orbv = mo_coeff[:, nocc:]
        eps_occ = mo_energy[:nocc]
        eps_vir = mo_energy[nocc:]
        eri_ovov = getattr(molecule, "eri_ovov", None)
        if eri_ovov is None:
            eri_ovov = jnp.einsum(
                "pqrs,pi,qa,rj,sb->iajb",
                rep_tensor,
                orbo,
                orbv,
                orbo,
                orbv,
                precision=Precision.HIGHEST,
            )
        else:
            eri_ovov = jnp.asarray(eri_ovov)
        denom = (
            eps_occ[:, None, None, None]
            + eps_occ[None, None, :, None]
            - eps_vir[None, :, None, None]
            - eps_vir[None, None, None, :]
        )
        denom = jnp.where(jnp.abs(denom) > self.density_floor, denom, -self.density_floor)
        direct = eri_ovov
        exchange = jnp.transpose(eri_ovov, (0, 3, 2, 1))
        pair_weights = (2.0 * direct - exchange) / denom
        total_energy = jnp.sum(direct * pair_weights)
        projected_energy = jnp.tensordot(weights, projected, axes=(0, 0))
        scale = jnp.where(
            jnp.abs(projected_energy) > self.density_floor,
            total_energy / projected_energy,
            0.0,
        )
        projected = scale * projected
        projected = jnp.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)
        return self._maybe_clip_response(projected)

    def projected_pt2_grid_contribution(
        self,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
        occupation_tolerance: float = 1e-8,
    ) -> Array:
        """Return the configured PT2 local channel.

        `scaled_projected` reproduces the legacy behavior: rescale the local
        pair gauge so its weighted grid integral matches the canonical MP2
        correlation energy, then optionally clip it.

        `local_exact` keeps the raw local pair gauge without global rescaling
        or clipping. On finite grids this generally does not integrate exactly
        to the canonical MP2 energy, but it preserves the unprojected spatial
        profile.
        """
        if self.energy_mode in {"dldh_two_lmf", "graddft_coeff_basis_hf_pt2_heads"}:
            return self._local_exact_pt2_grid_contribution(
                molecule,
                features=features,
                occupation_tolerance=occupation_tolerance,
            )
        if self.pt2_channel_mode == "local_exact":
            return self._local_exact_pt2_grid_contribution(
                molecule,
                features=features,
                occupation_tolerance=occupation_tolerance,
            )
        return self._legacy_projected_pt2_grid_contribution(
            molecule,
            features=features,
            occupation_tolerance=occupation_tolerance,
        )

    def energy_density(
        self,
        params: PyTree,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
        semilocal_energy_density: Array | None = None,
        hf_energy_density: Array | None = None,
        hf_spin_energy_density: tuple[Array, Array] | None = None,
        pt2_energy_density: Array | None = None,
    ) -> Array:
        """Return DM21 local XC grid contribution e_xc(r)."""
        channels = self.channel_contributions(
            params,
            molecule,
            features=features,
            semilocal_energy_density=semilocal_energy_density,
            hf_energy_density=hf_energy_density,
            hf_spin_energy_density=hf_spin_energy_density,
            pt2_energy_density=pt2_energy_density,
        )
        return jnp.sum(channels, axis=-1)

    def grid_contribution(
        self,
        params: PyTree,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
        semilocal_energy_density: Array | None = None,
        hf_energy_density: Array | None = None,
        hf_spin_energy_density: tuple[Array, Array] | None = None,
        pt2_energy_density: Array | None = None,
    ) -> Array:
        r"""Return DM21 local XC grid contribution e_xc(r)."""

        return self.energy_density(
            params,
            molecule,
            features=features,
            semilocal_energy_density=semilocal_energy_density,
            hf_energy_density=hf_energy_density,
            hf_spin_energy_density=hf_spin_energy_density,
            pt2_energy_density=pt2_energy_density,
        )

    def channel_contributions(
        self,
        params: PyTree,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
        semilocal_energy_density: Array | None = None,
        hf_energy_density: Array | None = None,
        hf_spin_energy_density: tuple[Array, Array] | None = None,
        pt2_energy_density: Array | None = None,
    ) -> Array:
        r"""Return per-channel DM21 local contributions c_k(r) * e_k(r).

        The returned array has shape (..., n_channels) for either:
        - graddft_coeff_basis / normalized_mixing_basis:
          [semilocal_1, ..., semilocal_n, pt2_projected?, hf_projected]
        - graddft_coeff_basis_hf_pt2_heads:
          [c_1 e_1, ..., c_n e_n, fpt2 e_c^PT2, fx e_x^HF]
        - dldh_two_lmf:
          [(1-fx)e_x^DFA, (1-fpt2)e_c^DFA, fpt2 e_c^PT2, fx e_x^HF]
        """
        total_gradient = None
        lapl = None
        if self._uses_dldh_aux_exchange_branch():
            features_with_grad, total_gradient = restricted_grid_features_with_gradients(molecule)
            if features is None:
                features = features_with_grad
            if self._uses_dldh_laplacian_branch():
                _, _, _, lapl = restricted_grid_response_variables(
                    molecule,
                    feature_kind="MGGA_LAPL",
                )
        elif features is None:
            features = restricted_grid_features(molecule)
        semilocal_channels = self.semilocal_energy_density_channels(features)
        semilocal_total = (
            jnp.sum(semilocal_channels, axis=-1)
            if semilocal_energy_density is None
            else semilocal_energy_density
        )
        if hf_energy_density is None:
            hf_projected, hf_projected_a, hf_projected_b = self.projected_hf_grid_contribution_components(
                molecule,
                features=features,
            )
            hf_spin_inputs: tuple[Array, Array] | None = (hf_projected_a, hf_projected_b)
        else:
            hf_projected = hf_energy_density
            hf_spin_inputs = hf_spin_energy_density
        if pt2_energy_density is None and self.include_pt2_channel:
            pt2_energy_density = self.projected_pt2_grid_contribution(
                molecule,
                features=features,
            )
        coefficients = self.channel_coefficients(
            params,
            features,
            molecule=molecule,
            semilocal_energy_density=semilocal_total,
            hf_energy_density=hf_projected,
            pt2_energy_density=pt2_energy_density,
            hf_spin_energy_density=hf_spin_inputs,
        )
        semilocal_local_channels = self._semilocal_local_contribution_channels(
            features,
            semilocal_channels,
        )
        if self.energy_mode == "dldh_two_lmf" and self._uses_dldh_aux_exchange_branch():
            semilocal_x, semilocal_c = self._split_semilocal_exchange_correlation_local_channels(
                semilocal_local_channels
            )
            fx = coefficients[..., 0]
            if self.include_pt2_channel:
                fpt2 = coefficients[..., 1]
            else:
                fpt2 = jnp.zeros_like(fx)
            if total_gradient is None:
                total_gradient = self._default_total_gradient_from_features(features)
            exchange_residual, hf_term = self._dldh_exchange_components(
                molecule=molecule,
                rho=jnp.maximum(features.rho, self.density_floor),
                grad=jnp.asarray(total_gradient),
                tau=jnp.maximum(features.tau_a + features.tau_b, 0.0),
                lapl=lapl,
                semilocal_x=semilocal_x,
                hf_total=hf_projected,
                hf_a=(hf_projected if hf_spin_inputs is None else hf_spin_inputs[0]),
                hf_b=(hf_projected if hf_spin_inputs is None else hf_spin_inputs[1]),
                fx=fx,
                features=features,
            )
            pt2_value = (
                jnp.zeros_like(hf_projected)
                if pt2_energy_density is None
                else jnp.asarray(pt2_energy_density)
            )
            return jnp.stack(
                [
                    exchange_residual,
                    (1.0 - fpt2) * semilocal_c,
                    fpt2 * pt2_value,
                    hf_term,
                ],
                axis=-1,
            )
        basis = self._assemble_basis_channels(
            semilocal_local_channels,
            hf_projected=hf_projected,
            pt2_projected=pt2_energy_density,
        )
        if self.energy_mode != "dldh_two_lmf" and coefficients.shape[-1] != basis.shape[-1]:
            raise ValueError(
                "Model output_dim must match basis channels "
                f"(got {coefficients.shape[-1]}, expected {basis.shape[-1]})."
            )
        return self._assemble_channel_contributions(coefficients, basis)

    def effective_exchange_fraction(
        self,
        params: PyTree,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
    ) -> Array:
        if features is None:
            features = restricted_grid_features(molecule)
        weights = jnp.asarray(molecule.grid.weights)
        rho = jnp.maximum(features.rho, self.density_floor)
        semilocal = self.semilocal_energy_density(features)
        hf_projected, hf_projected_a, hf_projected_b = self.projected_hf_grid_contribution_components(
            molecule,
            features=features,
        )
        coefficients = self.channel_coefficients(
            params,
            features,
            molecule=molecule,
            semilocal_energy_density=semilocal,
            hf_energy_density=hf_projected,
            pt2_energy_density=(
                self.projected_pt2_grid_contribution(molecule, features=features)
                if self.include_pt2_channel
                else None
            ),
            hf_spin_energy_density=(hf_projected_a, hf_projected_b),
        )
        hf_field = self._local_hf_fraction_from_coefficients(coefficients)
        numerator = jnp.tensordot(weights, rho * hf_field, axes=(0, 0))
        denominator = jnp.tensordot(weights, rho, axes=(0, 0))
        alpha = numerator / jnp.maximum(denominator, self.density_floor)
        alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
        return jnp.clip(alpha, 0.0, 1.0)

    def exact_exchange_energy(self, molecule: Any) -> Array:
        rep_tensor = jnp.asarray(molecule.rep_tensor)
        rdm1 = jnp.asarray(molecule.rdm1)
        if rdm1.ndim == 2:
            rdm1 = jnp.stack([0.5 * rdm1, 0.5 * rdm1], axis=0)

        def spin_exchange(dm_spin):
            exchange_matrix = jnp.einsum(
                "prqs,rs->pq",
                rep_tensor,
                dm_spin,
                precision=Precision.HIGHEST,
            )
            return -0.5 * jnp.einsum(
                "pq,pq->",
                dm_spin,
                exchange_matrix,
                precision=Precision.HIGHEST,
            )

        return jnp.sum(jax.vmap(spin_exchange)(rdm1))

    def semilocal_energy(
        self,
        features: RestrictedFeatureBundle,
        weights: Array,
    ) -> Array:
        semilocal_channels = self.semilocal_energy_density_channels(features)
        local = jnp.sum(
            self._semilocal_local_contribution_channels(features, semilocal_channels),
            axis=-1,
        )
        return jnp.tensordot(jnp.asarray(weights), local, axes=(0, 0))

    def energy_from_molecule(self, params: PyTree, molecule: Any) -> Array:
        features = restricted_grid_features(molecule)
        local_xc = self.grid_contribution(
            params,
            molecule,
            features=features,
        )
        energy = jnp.tensordot(jnp.asarray(molecule.grid.weights), local_xc, axes=(0, 0))
        return jnp.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)

    def energy_xc_only(self, params: PyTree, molecule: Any) -> Array:
        """GradDFT-compatible XC-only energy alias."""

        return self.energy_from_molecule(params, molecule)

    def energy(
        self,
        params: PyTree,
        molecule: Any,
        *,
        include_non_xc: bool = True,
    ) -> Array:
        """GradDFT-compatible total-energy entrypoint.

        When ``include_non_xc`` is true (default), return:
            E_tot = E_one + E_H + E_nuc + E_xc
        otherwise return only ``E_xc``.
        """

        e_xc = self.energy_from_molecule(params, molecule)
        if not include_non_xc or not self.is_xc:
            return e_xc

        if getattr(molecule, "h1e", None) is None:
            raise AttributeError("Molecule-like object must define h1e for total energy.")
        if getattr(molecule, "rep_tensor", None) is None:
            raise AttributeError("Molecule-like object must define rep_tensor for total energy.")
        if getattr(molecule, "rdm1", None) is None:
            raise AttributeError("Molecule-like object must define rdm1 for total energy.")
        if getattr(molecule, "nuclear_repulsion", None) is None:
            raise AttributeError(
                "Molecule-like object must define nuclear_repulsion for total energy."
            )

        density_matrix = jnp.asarray(molecule.rdm1)
        if density_matrix.ndim == 3:
            density_matrix = density_matrix.sum(axis=0)
        h1e = jnp.asarray(molecule.h1e)
        rep_tensor = jnp.asarray(molecule.rep_tensor)

        e_one = jnp.einsum("pq,pq->", density_matrix, h1e, precision=Precision.HIGHEST)
        j_matrix = jnp.einsum(
            "pqrs,rs->pq",
            rep_tensor,
            density_matrix,
            precision=Precision.HIGHEST,
        )
        e_hartree = 0.5 * jnp.einsum(
            "pq,pq->",
            density_matrix,
            j_matrix,
            precision=Precision.HIGHEST,
        )
        e_nuc = jnp.asarray(molecule.nuclear_repulsion)
        return e_one + e_hartree + e_nuc + e_xc

    def _default_total_gradient_from_features(
        self,
        features: RestrictedFeatureBundle,
    ) -> Array:
        sigma = jnp.maximum(features.sigma, 0.0)
        return jnp.stack(
            [jnp.sqrt(sigma), jnp.zeros_like(sigma), jnp.zeros_like(sigma)],
            axis=-1,
        )

    def _response_variables(
        self,
        features: RestrictedFeatureBundle,
        total_gradient: Array | None = None,
        laplacian: Array | None = None,
    ) -> tuple[Array, Array, Array, Array | None, Array]:
        response_floor = self._effective_response_density_floor()
        rho0 = jnp.maximum(features.rho, response_floor)
        tau0 = jnp.maximum(features.tau_a + features.tau_b, 0.0)
        if total_gradient is None:
            grad0 = self._default_total_gradient_from_features(features)
        else:
            grad0 = jnp.asarray(total_gradient, dtype=rho0.dtype)
            if grad0.ndim != rho0.ndim + 1 or grad0.shape[-1] != 3:
                raise ValueError(
                    "total_gradient must have shape (..., 3) matching features.rho."
                )
        if self._uses_dldh_laplacian_branch():
            if laplacian is None:
                raise ValueError("QAC-enabled DLDH response requires a grid laplacian.")
            lapl0 = jnp.asarray(laplacian, dtype=rho0.dtype)
            variables = jnp.concatenate(
                [rho0[..., None], grad0, tau0[..., None], lapl0[..., None]],
                axis=-1,
            )
            return rho0, grad0, tau0, lapl0, variables
        variables = jnp.concatenate(
            [rho0[..., None], grad0, tau0[..., None]],
            axis=-1,
        )
        return rho0, grad0, tau0, None, variables

    def _strict_response_payload(
        self,
        features: RestrictedFeatureBundle,
        total_gradient: Array,
        hf_projected: Array,
        *,
        pt2_projected: Array | None = None,
        hf_spin_energy_density: tuple[Array, Array] | None = None,
        laplacian: Array | None = None,
        exchange_anchor: Array | None = None,
    ) -> tuple[Array, Array, Array, Array, Array, Array | None, Array | None]:
        rho0, _, _, _, response_variables = self._response_variables(
            features,
            total_gradient,
            laplacian=laplacian,
        )
        if hf_spin_energy_density is None:
            hf_feature_a = hf_projected
            hf_feature_b = hf_projected
        else:
            hf_feature_a, hf_feature_b = hf_spin_energy_density
        pt2_feature = (
            jnp.zeros_like(hf_projected)
            if pt2_projected is None
            else jnp.asarray(pt2_projected)
        )
        active = rho0 > self._effective_response_density_floor()
        return (
            response_variables,
            active,
            hf_feature_a,
            hf_feature_b,
            pt2_feature,
            laplacian,
            exchange_anchor,
        )

    def _strict_aux_fields(
        self,
        molecule: Any | None,
        features: RestrictedFeatureBundle,
    ) -> tuple[Array | None, Array | None]:
        laplacian = None
        if self._uses_dldh_laplacian_branch():
            if molecule is None:
                raise ValueError("QAC-enabled DLDH assembly requires molecule grid data.")
            _, _, _, laplacian = restricted_grid_response_variables(
                molecule,
                feature_kind="MGGA_LAPL",
            )
        exchange_anchor = None
        if self.energy_mode == "dldh_two_lmf" and self.dldh_range_separated_exchange:
            if molecule is None:
                raise ValueError("Range-separated DLDH assembly requires molecule data.")
            exchange_anchor = self._dldh_short_range_hf_grid_contribution_components(
                molecule,
                features=features,
            )[0]
        return laplacian, exchange_anchor

    def _semilocal_point_local_energy_from_variables(self, variables: Array) -> Array:
        response_floor = self._effective_response_density_floor()
        rho_point = jnp.maximum(variables[0], response_floor)
        grad_point = variables[1:4]
        tau_point = jnp.maximum(variables[4], 0.0)
        point_features = restricted_feature_bundle_from_response_variables(
            rho_point,
            grad_point,
            tau_point,
            density_floor=response_floor,
        )
        semilocal_channels = self.semilocal_energy_density_channels(point_features)
        semilocal_local_channels = self._semilocal_local_contribution_channels(
            point_features,
            semilocal_channels,
        )
        return jnp.sum(semilocal_local_channels, axis=-1)

    def _total_point_local_energy_from_variables(
        self,
        params: PyTree,
        variables: Array,
        hf_point: Array,
        hf_point_a: Array,
        hf_point_b: Array,
        exchange_anchor_point: Array | None = None,
        *,
        pt2_point: Array | None = None,
        response_hf_mode: Literal["nonlocal_exchange_only", "local_projected"] | None = None,
        response_pt2_mode: Literal["nonlocal_correlation_only", "local_projected"] | None = None,
    ) -> Array:
        hf_mode = self.response_hf_mode if response_hf_mode is None else response_hf_mode
        pt2_mode = self.response_pt2_mode if response_pt2_mode is None else response_pt2_mode
        response_floor = self._effective_response_density_floor()
        rho_point = jnp.maximum(variables[0], response_floor)
        grad_point = variables[1:4]
        tau_point = jnp.maximum(variables[4], 0.0)
        lapl_point = variables[5] if variables.shape[0] > 5 else None
        point_features = restricted_feature_bundle_from_response_variables(
            rho_point,
            grad_point,
            tau_point,
            density_floor=response_floor,
        )
        semilocal_channels = self.semilocal_energy_density_channels(point_features)
        semilocal_local_channels = self._semilocal_local_contribution_channels(
            point_features,
            semilocal_channels,
        )
        semilocal_total = jnp.sum(semilocal_channels, axis=-1)
        if hf_mode == "local_projected":
            hf_input = hf_point
            hf_basis = hf_point
            hf_spin_inputs: tuple[Array, Array] | None = (hf_point_a, hf_point_b)
        elif hf_mode == "nonlocal_exchange_only":
            hf_input = jax.lax.stop_gradient(hf_point)
            hf_basis = jnp.zeros_like(hf_input)
            hf_spin_inputs = (
                jax.lax.stop_gradient(hf_point_a),
                jax.lax.stop_gradient(hf_point_b),
            )
        else:
            raise ValueError(
                f"Unsupported response_hf_mode={hf_mode!r}. "
                "Expected 'nonlocal_exchange_only' or 'local_projected'."
            )
        if pt2_point is None:
            pt2_point = jnp.zeros_like(hf_point)
        if self.include_pt2_channel:
            if pt2_mode == "local_projected":
                pt2_input = pt2_point
                pt2_basis = pt2_point
            elif pt2_mode == "nonlocal_correlation_only":
                pt2_input = jax.lax.stop_gradient(pt2_point)
                pt2_basis = jnp.zeros_like(pt2_input)
            else:
                raise ValueError(
                    f"Unsupported response_pt2_mode={pt2_mode!r}. "
                    "Expected 'nonlocal_correlation_only' or 'local_projected'."
                )
        else:
            pt2_input = None
            pt2_basis = None
        coefficients = self.channel_coefficients(
            params,
            point_features,
            semilocal_energy_density=semilocal_total,
            hf_energy_density=hf_input,
            pt2_energy_density=pt2_input,
            hf_spin_energy_density=hf_spin_inputs,
        )
        if self.energy_mode == "dldh_two_lmf" and self._uses_dldh_aux_exchange_branch():
            semilocal_x, semilocal_c = self._split_semilocal_exchange_correlation_local_channels(
                semilocal_local_channels
            )
            fx = coefficients[..., 0]
            if self.include_pt2_channel:
                fpt2 = coefficients[..., 1]
            else:
                fpt2 = jnp.zeros_like(fx)
            exchange_residual, hf_term = self._dldh_exchange_components(
                molecule=None,
                rho=rho_point,
                grad=grad_point,
                tau=tau_point,
                lapl=lapl_point,
                semilocal_x=semilocal_x,
                hf_total=hf_basis,
                hf_a=hf_point_a,
                hf_b=hf_point_b,
                fx=fx,
                exchange_anchor=exchange_anchor_point,
                features=None,
            )
            correlation_term = (1.0 - fpt2) * semilocal_c
            pt2_term = fpt2 * jnp.zeros_like(correlation_term)
            if self.include_pt2_channel and pt2_basis is not None:
                pt2_term = fpt2 * pt2_basis
            return exchange_residual + correlation_term + pt2_term + hf_term
        basis = self._assemble_basis_channels(
            semilocal_local_channels,
            hf_projected=hf_basis,
            pt2_projected=pt2_basis,
        )
        if self.energy_mode != "dldh_two_lmf" and coefficients.shape[-1] != basis.shape[-1]:
            raise ValueError(
                "Model output_dim must match basis channels "
                f"(got {coefficients.shape[-1]}, expected {basis.shape[-1]})."
            )
        channels = self._assemble_channel_contributions(coefficients, basis)
        return jnp.sum(channels, axis=-1)

    def _strict_total_potential_components(
        self,
        params: PyTree,
        features: RestrictedFeatureBundle,
        total_gradient: Array,
        hf_projected: Array,
        *,
        pt2_projected: Array | None = None,
        hf_spin_energy_density: tuple[Array, Array] | None = None,
        response_hf_mode: Literal["nonlocal_exchange_only", "local_projected"] | None = None,
        response_pt2_mode: Literal["nonlocal_correlation_only", "local_projected"] | None = None,
        reference_coefficient_inputs: Array | None = None,
        return_hf_field: bool = False,
        strict_payload: tuple[Array, Array, Array, Array, Array, Array | None, Array | None] | None = None,
    ) -> tuple[Array, Array, Array, Array] | tuple[Array, Array, Array, Array, Array]:
        if strict_payload is None:
            strict_payload = self._strict_response_payload(
                features,
                total_gradient,
                hf_projected,
                pt2_projected=pt2_projected,
                hf_spin_energy_density=hf_spin_energy_density,
            )
        response_variables, active, hf_feature_a, hf_feature_b, pt2_feature, _, exchange_anchor = strict_payload
        hf_field = None
        if return_hf_field:
            if reference_coefficient_inputs is None:
                raise ValueError(
                    "reference_coefficient_inputs must be provided when return_hf_field=True."
                )
            reference_coefficients = self.channel_coefficients_from_inputs(
                params,
                reference_coefficient_inputs,
            )
            hf_field = self._local_hf_fraction_from_coefficients(reference_coefficients)
        point_gradient_fn = jax.grad(
            self._total_point_local_energy_from_variables,
            argnums=1,
        )

        def point_gradients(
            variables: Array,
            hf_point: Array,
            hf_point_a: Array,
            hf_point_b: Array,
            exchange_anchor_point: Array | None,
            pt2_point: Array,
        ) -> Array:
            return point_gradient_fn(
                params,
                variables,
                hf_point,
                hf_point_a,
                hf_point_b,
                exchange_anchor_point,
                pt2_point=pt2_point,
                response_hf_mode=response_hf_mode,
                response_pt2_mode=response_pt2_mode,
            )

        gradients = jax.vmap(point_gradients)(
            response_variables,
            hf_projected,
            hf_feature_a,
            hf_feature_b,
            (
                jnp.zeros_like(hf_projected)
                if exchange_anchor is None
                else exchange_anchor
            ),
            pt2_feature,
        )
        gradients = jnp.nan_to_num(gradients, nan=0.0, posinf=0.0, neginf=0.0)
        gradients = self._maybe_clip_response(gradients)
        v_rho = jnp.where(active, gradients[:, 0], 0.0)
        v_grad = jnp.where(active[:, None], gradients[:, 1:4], 0.0)
        v_tau = jnp.where(active, gradients[:, 4], 0.0)
        if gradients.shape[1] > 5:
            v_lapl = jnp.where(active, gradients[:, 5], 0.0)
        else:
            v_lapl = jnp.zeros_like(v_rho)
        if return_hf_field:
            return v_rho, v_grad, v_tau, v_lapl, jnp.asarray(hf_field)
        return v_rho, v_grad, v_tau, v_lapl

    def _projected_semilocal_kernel(
        self,
        features: RestrictedFeatureBundle,
    ) -> Array:
        rho0, _, _, _, response_variables = self._response_variables(features)
        point_hessian = jax.vmap(jax.hessian(self._semilocal_point_local_energy_from_variables))(
            response_variables
        )
        kernel = point_hessian[:, 0, 0]
        kernel = jnp.nan_to_num(kernel, nan=0.0, posinf=0.0, neginf=0.0)
        kernel = self._maybe_clip_response(kernel)
        return jnp.where(rho0 <= self._effective_response_density_floor(), 0.0, kernel)

    def _projected_semilocal_potential(
        self,
        features: RestrictedFeatureBundle,
    ) -> Array:
        rho0, _, _, _, response_variables = self._response_variables(features)
        gradients = jax.vmap(jax.grad(self._semilocal_point_local_energy_from_variables))(
            response_variables
        )
        potential = gradients[:, 0]
        potential = jnp.nan_to_num(potential, nan=0.0, posinf=0.0, neginf=0.0)
        potential = self._maybe_clip_response(potential)
        return jnp.where(rho0 <= self._effective_response_density_floor(), 0.0, potential)

    def _projected_total_potential_kernel(
        self,
        params: PyTree,
        features: RestrictedFeatureBundle,
        hf_projected: Array,
        molecule: Any | None = None,
        *,
        pt2_projected: Array | None = None,
        total_gradient: Array | None = None,
        hf_spin_energy_density: tuple[Array, Array] | None = None,
        response_hf_mode: Literal["nonlocal_exchange_only", "local_projected"] | None = None,
        response_pt2_mode: Literal["nonlocal_correlation_only", "local_projected"] | None = None,
    ) -> tuple[Array, Array]:
        grad = (
            self._default_total_gradient_from_features(features)
            if total_gradient is None
            else jnp.asarray(total_gradient)
        )
        laplacian, exchange_anchor = self._strict_aux_fields(molecule, features)
        strict_payload = self._strict_response_payload(
            features,
            grad,
            hf_projected,
            pt2_projected=pt2_projected,
            hf_spin_energy_density=hf_spin_energy_density,
            laplacian=laplacian,
            exchange_anchor=exchange_anchor,
        )
        potential, _, _, _ = self._strict_total_potential_components(
            params,
            features,
            grad,
            hf_projected,
            pt2_projected=pt2_projected,
            hf_spin_energy_density=hf_spin_energy_density,
            response_hf_mode=response_hf_mode,
            response_pt2_mode=response_pt2_mode,
            strict_payload=strict_payload,
        )
        tensor = self._strict_total_response_tensor(
            params,
            features,
            grad,
            hf_projected,
            pt2_projected=pt2_projected,
            hf_spin_energy_density=(
                (hf_projected, hf_projected)
                if hf_spin_energy_density is None
                else hf_spin_energy_density
            ),
            response_hf_mode=response_hf_mode,
            response_pt2_mode=response_pt2_mode,
            strict_payload=strict_payload,
        )
        kernel = tensor[0, 0]
        return potential, kernel

    def _projected_total_potential_only(
        self,
        params: PyTree,
        features: RestrictedFeatureBundle,
        hf_projected: Array,
        molecule: Any | None = None,
        *,
        pt2_projected: Array | None = None,
        total_gradient: Array | None = None,
        hf_spin_energy_density: tuple[Array, Array] | None = None,
        response_hf_mode: Literal["nonlocal_exchange_only", "local_projected"] | None = None,
        response_pt2_mode: Literal["nonlocal_correlation_only", "local_projected"] | None = None,
    ) -> Array:
        grad = (
            self._default_total_gradient_from_features(features)
            if total_gradient is None
            else jnp.asarray(total_gradient)
        )
        laplacian, exchange_anchor = self._strict_aux_fields(molecule, features)
        strict_payload = self._strict_response_payload(
            features,
            grad,
            hf_projected,
            pt2_projected=pt2_projected,
            hf_spin_energy_density=hf_spin_energy_density,
            laplacian=laplacian,
            exchange_anchor=exchange_anchor,
        )
        potential, _, _, _ = self._strict_total_potential_components(
            params,
            features,
            grad,
            hf_projected,
            pt2_projected=pt2_projected,
            hf_spin_energy_density=hf_spin_energy_density,
            response_hf_mode=response_hf_mode,
            response_pt2_mode=response_pt2_mode,
            strict_payload=strict_payload,
        )
        return potential

    def _strict_total_response_tensor(
        self,
        params: PyTree,
        features: RestrictedFeatureBundle,
        total_gradient: Array,
        hf_projected: Array,
        *,
        pt2_projected: Array | None = None,
        hf_spin_energy_density: tuple[Array, Array],
        response_hf_mode: Literal["nonlocal_exchange_only", "local_projected"] | None = None,
        response_pt2_mode: Literal["nonlocal_correlation_only", "local_projected"] | None = None,
        strict_payload: tuple[Array, Array, Array, Array, Array, Array | None, Array | None] | None = None,
    ) -> Array:
        """Return the strict restricted semilocal response tensor on the grid.

        The tensor follows the PySCF reduced MGGA convention with local variables
        ``[rho, d_x rho, d_y rho, d_z rho, tau]``.
        """

        if strict_payload is None:
            strict_payload = self._strict_response_payload(
                features,
                total_gradient,
                hf_projected,
                pt2_projected=pt2_projected,
                hf_spin_energy_density=hf_spin_energy_density,
            )
        response_variables, active, hf_projected_a, hf_projected_b, pt2_feature, _, exchange_anchor = strict_payload
        point_hessian_fn = jax.hessian(
            self._total_point_local_energy_from_variables,
            argnums=1,
        )

        def point_tensor(
            variables: Array,
            hf_point: Array,
            hf_point_a: Array,
            hf_point_b: Array,
            exchange_anchor_point: Array | None,
            pt2_point: Array,
        ) -> Array:
            tensor = point_hessian_fn(
                params,
                variables,
                hf_point,
                hf_point_a,
                hf_point_b,
                exchange_anchor_point,
                pt2_point=pt2_point,
                response_hf_mode=response_hf_mode,
                response_pt2_mode=response_pt2_mode,
            )
            tensor = jnp.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
            tensor = self._maybe_clip_response(tensor)
            return tensor

        tensor = jax.vmap(point_tensor)(
            response_variables,
            hf_projected,
            hf_projected_a,
            hf_projected_b,
            (
                jnp.zeros_like(hf_projected)
                if exchange_anchor is None
                else exchange_anchor
            ),
            pt2_feature,
        )
        tensor = tensor * active[:, None, None].astype(tensor.dtype)
        return jnp.asarray(tensor).transpose(1, 2, 0)

    def _grid_hfx_feature_gradients(
        self,
        params: PyTree,
        features: RestrictedFeatureBundle,
        semilocal_channels: Array,
        hf_projected: Array,
        hf_feature_a: Array,
        hf_feature_b: Array,
        *,
        pt2_projected: Array | None = None,
        grid_weights: Array,
    ) -> tuple[Array, Array]:
        """Gradient of weighted XC energy with respect to local HF input features."""

        semilocal_total = jnp.sum(semilocal_channels, axis=-1)
        semilocal_local_channels = self._semilocal_local_contribution_channels(
            features,
            semilocal_channels,
        )
        basis = self._assemble_basis_channels(
            semilocal_local_channels,
            hf_projected=hf_projected,
            pt2_projected=pt2_projected,
        )
        weights = jnp.asarray(grid_weights)

        def weighted_energy_from_hfx(hfx_a: Array, hfx_b: Array) -> Array:
            coefficients = self.channel_coefficients(
                params,
                features,
                semilocal_energy_density=semilocal_total,
                hf_energy_density=hf_projected,
                pt2_energy_density=pt2_projected,
                hf_spin_energy_density=(hfx_a, hfx_b),
            )
            channels = self._assemble_channel_contributions(coefficients, basis)
            local_xc = jnp.sum(channels, axis=-1)
            return jnp.tensordot(weights, local_xc, axes=(0, 0))

        grad_a, grad_b = jax.grad(weighted_energy_from_hfx, argnums=(0, 1))(
            hf_feature_a,
            hf_feature_b,
        )
        grad_a = jnp.nan_to_num(grad_a, nan=0.0, posinf=0.0, neginf=0.0)
        grad_b = jnp.nan_to_num(grad_b, nan=0.0, posinf=0.0, neginf=0.0)
        grad_a = self._maybe_clip_response(grad_a)
        grad_b = self._maybe_clip_response(grad_b)
        return grad_a, grad_b

    def projected_local_kernel(
        self,
        params: PyTree,
        molecule: Any,
    ) -> Array:
        features, total_gradient = restricted_grid_features_with_gradients(molecule)
        hf_projected = self.projected_hf_grid_contribution_components(
            molecule,
            features=features,
        )[0]
        pt2_projected = (
            self.projected_pt2_grid_contribution(molecule, features=features)
            if self.include_pt2_channel
            else None
        )
        _, kernel = self._projected_total_potential_kernel(
            params,
            features,
            hf_projected,
            molecule,
            pt2_projected=pt2_projected,
            total_gradient=total_gradient,
        )
        return kernel

    def projected_local_potential(
        self,
        params: PyTree,
        molecule: Any,
    ) -> Array:
        features, total_gradient = restricted_grid_features_with_gradients(molecule)
        hf_projected = self.projected_hf_grid_contribution_components(
            molecule,
            features=features,
        )[0]
        pt2_projected = (
            self.projected_pt2_grid_contribution(molecule, features=features)
            if self.include_pt2_channel
            else None
        )
        potential, _ = self._projected_total_potential_kernel(
            params,
            features,
            hf_projected,
            molecule,
            pt2_projected=pt2_projected,
            total_gradient=total_gradient,
        )
        return potential

    def bind_to_molecule(self, params: PyTree, molecule: Any) -> BoundDM21LikeFunctional:
        features, total_gradient = restricted_grid_features_with_gradients(molecule)
        laplacian, exchange_anchor = self._strict_aux_fields(molecule, features)
        semilocal_channels = self.semilocal_energy_density_channels(features)
        semilocal = jnp.sum(semilocal_channels, axis=-1)
        hf_projected, hf_projected_a, hf_projected_b = self.projected_hf_grid_contribution_components(
            molecule,
            features=features,
        )
        if self.input_feature_mode == "dm21_original":
            hfx_feature_a, hfx_feature_b = self._dm21_hfx_feature_channels(
                molecule,
                features,
                hf_energy_density=hf_projected,
                hf_spin_energy_density=(hf_projected_a, hf_projected_b),
            )
        else:
            hfx_feature_a, hfx_feature_b = hf_projected_a, hf_projected_b
        pt2_projected = (
            self.projected_pt2_grid_contribution(molecule, features=features)
            if self.include_pt2_channel
            else None
        )
        coefficient_inputs = self.coefficient_inputs(
            features,
            semilocal,
            hf_projected,
            pt2_energy_density=pt2_projected,
            molecule=molecule,
            hf_spin_energy_density=(hf_projected_a, hf_projected_b),
        )
        coefficients = self.channel_coefficients_from_inputs(
            params,
            coefficient_inputs,
        )
        hf_field = self._local_hf_fraction_from_coefficients(coefficients)
        strict_payload = self._strict_response_payload(
            features,
            total_gradient,
            hf_projected,
            pt2_projected=pt2_projected,
            hf_spin_energy_density=(hfx_feature_a, hfx_feature_b),
            laplacian=laplacian,
            exchange_anchor=exchange_anchor,
        )
        projected_vrho, projected_vgrad, projected_vtau, projected_vlapl = self._strict_total_potential_components(
            params,
            features,
            total_gradient,
            hf_projected,
            pt2_projected=pt2_projected,
            hf_spin_energy_density=(hfx_feature_a, hfx_feature_b),
            strict_payload=strict_payload,
        )
        projected_tensor = self._strict_total_response_tensor(
            params,
            features,
            total_gradient,
            hf_projected,
            pt2_projected=pt2_projected,
            hf_spin_energy_density=(hfx_feature_a, hfx_feature_b),
            strict_payload=strict_payload,
        )
        projected_kernel = projected_tensor[0, 0]
        projected_energy_density = jnp.sum(
            self.channel_contributions(
                params,
                molecule,
                features=features,
                semilocal_energy_density=semilocal,
                hf_energy_density=hf_projected,
                pt2_energy_density=pt2_projected,
                hf_spin_energy_density=(hf_projected_a, hf_projected_b),
            ),
            axis=-1,
        )

        rho = jnp.maximum(features.rho, self.density_floor)
        grid_weights = jnp.asarray(molecule.grid.weights)
        numerator = jnp.tensordot(grid_weights, rho * hf_field, axes=(0, 0))
        denominator = jnp.tensordot(grid_weights, rho, axes=(0, 0))
        alpha = numerator / jnp.maximum(denominator, self.density_floor)
        alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
        alpha = jnp.clip(alpha, 0.0, 1.0)

        def grid_response_tensor_fn() -> Array:
            return projected_tensor

        def grid_hfx_feature_gradients_fn() -> tuple[Array, Array]:
            return self._grid_hfx_feature_gradients(
                params,
                features,
                semilocal_channels,
                hf_projected,
                hfx_feature_a,
                hfx_feature_b,
                pt2_projected=pt2_projected,
                grid_weights=molecule.grid.weights,
            )

        return BoundDM21LikeFunctional(
            name=self.name,
            projected_local_potential_values=projected_vrho,
            projected_local_kernel_values=projected_kernel,
            exact_exchange_fraction=alpha,
            projected_local_potential_gradient_values=projected_vgrad,
            projected_local_potential_tau_values=projected_vtau,
            projected_local_potential_laplacian_values=projected_vlapl,
            projected_energy_density_values=projected_energy_density,
            local_hf_fraction_values=(
                hf_field if self.response_hf_mode == "local_projected" else None
            ),
            response_feature_kind=self._response_feature_kind_label(),
            grid_response_tensor_fn=grid_response_tensor_fn,
            grid_hfx_feature_gradients_fn=grid_hfx_feature_gradients_fn,
        )

    def bind_to_molecule_for_response(
        self,
        params: PyTree,
        molecule: Any,
    ) -> BoundDM21LikeFunctional:
        """TD-response-only binding that avoids assembling strict potential terms."""

        features, total_gradient = restricted_grid_features_with_gradients(molecule)
        laplacian, exchange_anchor = self._strict_aux_fields(molecule, features)
        semilocal_channels = self.semilocal_energy_density_channels(features)
        semilocal = jnp.sum(semilocal_channels, axis=-1)
        hf_projected, hf_projected_a, hf_projected_b = self.projected_hf_grid_contribution_components(
            molecule,
            features=features,
        )
        if self.input_feature_mode == "dm21_original":
            hfx_feature_a, hfx_feature_b = self._dm21_hfx_feature_channels(
                molecule,
                features,
                hf_energy_density=hf_projected,
                hf_spin_energy_density=(hf_projected_a, hf_projected_b),
            )
        else:
            hfx_feature_a, hfx_feature_b = hf_projected_a, hf_projected_b
        pt2_projected = (
            self.projected_pt2_grid_contribution(molecule, features=features)
            if self.include_pt2_channel
            else None
        )
        coefficient_inputs = self.coefficient_inputs(
            features,
            semilocal,
            hf_projected,
            pt2_energy_density=pt2_projected,
            molecule=molecule,
            hf_spin_energy_density=(hf_projected_a, hf_projected_b),
        )
        coefficients = self.channel_coefficients_from_inputs(
            params,
            coefficient_inputs,
        )
        hf_field = self._local_hf_fraction_from_coefficients(coefficients)
        strict_payload = self._strict_response_payload(
            features,
            total_gradient,
            hf_projected,
            pt2_projected=pt2_projected,
            hf_spin_energy_density=(hfx_feature_a, hfx_feature_b),
            laplacian=laplacian,
            exchange_anchor=exchange_anchor,
        )
        projected_tensor = self._strict_total_response_tensor(
            params,
            features,
            total_gradient,
            hf_projected,
            pt2_projected=pt2_projected,
            hf_spin_energy_density=(hfx_feature_a, hfx_feature_b),
            strict_payload=strict_payload,
        )

        rho = jnp.maximum(features.rho, self.density_floor)
        grid_weights = jnp.asarray(molecule.grid.weights)
        numerator = jnp.tensordot(grid_weights, rho * hf_field, axes=(0, 0))
        denominator = jnp.tensordot(grid_weights, rho, axes=(0, 0))
        alpha = numerator / jnp.maximum(denominator, self.density_floor)
        alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
        alpha = jnp.clip(alpha, 0.0, 1.0)

        def grid_response_tensor_fn() -> Array:
            return projected_tensor

        # TD response uses only the strict tensor and scalar HF fraction.
        # Keep the bound object minimal and avoid strict potential/energy assembly.
        return BoundDM21LikeFunctional(
            name=self.name,
            projected_local_potential_values=jnp.zeros_like(features.rho),
            projected_local_kernel_values=projected_tensor[0, 0],
            exact_exchange_fraction=alpha,
            projected_local_potential_gradient_values=None,
            projected_local_potential_tau_values=None,
            projected_local_potential_laplacian_values=None,
            projected_energy_density_values=None,
            local_hf_fraction_values=None,
            response_feature_kind=self._response_feature_kind_label(),
            grid_response_tensor_fn=grid_response_tensor_fn,
            grid_hfx_feature_gradients_fn=None,
        )

    def _scf_binding_payload(
        self,
        params: PyTree,
        molecule: Any,
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Return SCF-only local potential components and scalar HF fraction."""

        features, total_gradient = restricted_grid_features_with_gradients(molecule)
        laplacian, exchange_anchor = self._strict_aux_fields(molecule, features)
        semilocal_channels = self.semilocal_energy_density_channels(features)
        semilocal = jnp.sum(semilocal_channels, axis=-1)
        hf_projected, hf_projected_a, hf_projected_b = self.projected_hf_grid_contribution_components(
            molecule,
            features=features,
        )
        if self.input_feature_mode == "dm21_original":
            hfx_feature_a, hfx_feature_b = self._dm21_hfx_feature_channels(
                molecule,
                features,
                hf_energy_density=hf_projected,
                hf_spin_energy_density=(hf_projected_a, hf_projected_b),
            )
        else:
            hfx_feature_a, hfx_feature_b = hf_projected_a, hf_projected_b
        pt2_projected = (
            self.projected_pt2_grid_contribution(molecule, features=features)
            if self.include_pt2_channel
            else None
        )

        coefficients = self.channel_coefficients(
            params,
            features,
            molecule=molecule,
            semilocal_energy_density=semilocal,
            hf_energy_density=hf_projected,
            pt2_energy_density=pt2_projected,
            hf_spin_energy_density=(hf_projected_a, hf_projected_b),
        )
        hf_field = self._local_hf_fraction_from_coefficients(coefficients)
        strict_payload = self._strict_response_payload(
            features,
            total_gradient,
            hf_projected,
            pt2_projected=pt2_projected,
            hf_spin_energy_density=(hfx_feature_a, hfx_feature_b),
            laplacian=laplacian,
            exchange_anchor=exchange_anchor,
        )
        projected_vrho, projected_vgrad, projected_vtau, projected_vlapl = self._strict_total_potential_components(
            params,
            features,
            total_gradient,
            hf_projected,
            pt2_projected=pt2_projected,
            hf_spin_energy_density=(hfx_feature_a, hfx_feature_b),
            strict_payload=strict_payload,
        )

        rho = jnp.maximum(features.rho, self.density_floor)
        grid_weights = jnp.asarray(molecule.grid.weights)
        numerator = jnp.tensordot(grid_weights, rho * hf_field, axes=(0, 0))
        denominator = jnp.tensordot(grid_weights, rho, axes=(0, 0))
        alpha = numerator / jnp.maximum(denominator, self.density_floor)
        alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
        alpha = jnp.clip(alpha, 0.0, 1.0)

        return projected_vrho, projected_vgrad, projected_vtau, projected_vlapl, alpha

    def scf_potential_components_and_alpha(
        self,
        params: PyTree,
        molecule: Any,
    ) -> tuple[Array, Array, Array, Array, str, Array]:
        """Direct SCF helper avoiding bound-functional construction."""

        projected_vrho, projected_vgrad, projected_vtau, projected_vlapl, alpha = self._scf_binding_payload(params, molecule)
        return projected_vrho, projected_vgrad, projected_vtau, projected_vlapl, self._response_feature_kind_label(), alpha

    def bind_to_molecule_for_scf(self, params: PyTree, molecule: Any) -> BoundDM21LikeFunctional:
        """SCF-only binding that avoids constructing strict f_xc response terms."""
        projected_vrho, projected_vgrad, projected_vtau, projected_vlapl, alpha = self._scf_binding_payload(
            params,
            molecule,
        )
        # SCF uses only the local potential components and the effective HF fraction.
        # Keep the bound object minimal and avoid assembling response/energy terms here.
        projected_kernel = jnp.zeros_like(projected_vrho)

        return BoundDM21LikeFunctional(
            name=self.name,
            projected_local_potential_values=projected_vrho,
            projected_local_kernel_values=projected_kernel,
            exact_exchange_fraction=alpha,
            projected_local_potential_gradient_values=projected_vgrad,
            projected_local_potential_tau_values=projected_vtau,
            projected_local_potential_laplacian_values=projected_vlapl,
            projected_energy_density_values=None,
            local_hf_fraction_values=None,
            response_feature_kind=self._response_feature_kind_label(),
            grid_response_tensor_fn=None,
            grid_hfx_feature_gradients_fn=None,
        )


def _make_neural_xc_hybrid_functional(
    *,
    non_hf_module: SemilocalEnergyDensityModule | None = None,
    semilocal_xc: str | Sequence[str] = DEFAULT_NEURAL_XC_SEMILOCAL_XC,
    semilocal_energy_density_fn: SemilocalEnergyDensityFn | None = None,
    n_semilocal_channels: int | None = None,
    energy_mode: Literal[
        "graddft_coeff_basis",
        "normalized_mixing_basis",
        "dldh_two_lmf",
        "graddft_coeff_basis_hf_pt2_heads",
    ] = DEFAULT_NEURAL_XC_ENERGY_MODE,
    input_feature_mode: Literal["enhanced", "dm21_original"] = "enhanced",
    hf_input_mode: Literal["total_only", "spin_resolved"] = "spin_resolved",
    hf_fraction_mode: Literal["normalized_weights", "hf_coefficient"] = (
        "normalized_weights"
    ),
    include_pt2_channel: bool = False,
    pt2_channel_mode: Literal["scaled_projected", "local_exact"] = "scaled_projected",
    response_hf_mode: Literal["nonlocal_exchange_only", "local_projected"] = (
        "nonlocal_exchange_only"
    ),
    response_pt2_mode: Literal["nonlocal_correlation_only", "local_projected"] = (
        "local_projected"
    ),
    strict_dm21_feature_alignment: bool = True,
    allow_experimental_jax_xc: bool = False,
    hidden_dims: Sequence[int] = GRADDFT_DEFAULT_DM21_HIDDEN_DIMS,
    activation: Callable[[Array], Array] = nn.tanh,
    network_architecture: Literal["simple_mlp", "graddft_residual"] = (
        GRADDFT_DEFAULT_NETWORK_ARCHITECTURE
    ),
    squash_offset: float = 1e-4,
    sigmoid_scale_factor: float = 2.0,
    density_floor: float = 1e-12,
    response_density_floor: float | None = 1e-5,
    kernel_clip: float = 5.0,
    response_kernel_clip: float | None = 5.0,
    dm21_hfx_channels: int = 2,
    dldh_range_separated_exchange: bool = False,
    dldh_range_separation_omega: float = _DLDH_RS_OMEGA_DEFAULT,
    dldh_qac_mode: Literal["none", "pade", "erf"] = "none",
    dldh_qac_parameters: Sequence[float] = (),
    name: str = "neural_xc",
) -> DM21LikeFunctional:
    if non_hf_module is not None:
        if (
            n_semilocal_channels is not None
            and int(n_semilocal_channels) != int(non_hf_module.n_channels)
        ):
            raise ValueError(
                "n_semilocal_channels must match non_hf_module.n_channels when both are set."
            )
        n_semilocal = int(non_hf_module.n_channels)
    elif semilocal_energy_density_fn is None:
        n_semilocal = len(
            _normalize_semilocal_xc_names(
                semilocal_xc,
                allow_experimental_jax_xc=allow_experimental_jax_xc,
            )
        )
    elif n_semilocal_channels is None:
        n_semilocal = 1
    else:
        n_semilocal = int(n_semilocal_channels)
    if n_semilocal <= 0:
        raise ValueError("n_semilocal_channels must be a positive integer.")

    dims = _normalize_hidden_dims(hidden_dims)
    if energy_mode == "dldh_two_lmf":
        output_dim = 1 + int(bool(include_pt2_channel))
    elif energy_mode == "graddft_coeff_basis_hf_pt2_heads":
        output_dim = n_semilocal + 1 + int(bool(include_pt2_channel))
    else:
        output_dim = n_semilocal + 1 + int(bool(include_pt2_channel))
    if network_architecture == "simple_mlp":
        model = DM21MixingMLP(
            hidden_dims=dims,
            output_dim=output_dim,
            activation=activation,
            squash_offset=squash_offset,
            sigmoid_scale_factor=sigmoid_scale_factor,
        )
    elif network_architecture == "graddft_residual":
        block_activation = nn.elu if activation is nn.tanh else activation
        model = GradDFTResidualMixingMLP(
            hidden_dims=dims,
            output_dim=output_dim,
            block_activation=block_activation,
            squash_offset=squash_offset,
            sigmoid_scale_factor=sigmoid_scale_factor,
        )
    else:
        raise ValueError(
            f"Unsupported network_architecture={network_architecture!r}. "
            "Expected 'simple_mlp' or 'graddft_residual'."
        )
    return DM21LikeFunctional(
        model=model,
        non_hf_module=non_hf_module,
        semilocal_xc=semilocal_xc,
        semilocal_energy_density_fn=semilocal_energy_density_fn,
        energy_mode=energy_mode,
        input_feature_mode=input_feature_mode,
        hf_input_mode=hf_input_mode,
        hf_fraction_mode=hf_fraction_mode,
        include_pt2_channel=bool(include_pt2_channel),
        pt2_channel_mode=pt2_channel_mode,
        response_hf_mode=response_hf_mode,
        response_pt2_mode=response_pt2_mode,
        strict_dm21_feature_alignment=bool(strict_dm21_feature_alignment),
        allow_experimental_jax_xc=bool(allow_experimental_jax_xc),
        density_floor=density_floor,
        response_density_floor=response_density_floor,
        kernel_clip=kernel_clip,
        response_kernel_clip=response_kernel_clip,
        dm21_hfx_channels=max(int(dm21_hfx_channels), 1),
        dldh_range_separated_exchange=bool(dldh_range_separated_exchange),
        dldh_range_separation_omega=float(dldh_range_separation_omega),
        dldh_qac_mode=str(dldh_qac_mode).lower(),
        dldh_qac_parameters=tuple(float(v) for v in dldh_qac_parameters),
        name=name,
    )


# Public Neural_xc naming aliases for the semilocal + neural-HF architecture.
BoundNeuralXCFunctional = BoundDM21LikeFunctional
NeuralXCHybridFunctional = DM21LikeFunctional
NeuralXCMixingMLP = DM21MixingMLP


def make_neural_xc_functional(
    *,
    non_hf_module: SemilocalEnergyDensityModule | None = None,
    semilocal_xc: str | Sequence[str] = DEFAULT_NEURAL_XC_SEMILOCAL_XC,
    semilocal_energy_density_fn: SemilocalEnergyDensityFn | None = None,
    n_semilocal_channels: int | None = None,
    energy_mode: Literal[
        "graddft_coeff_basis",
        "normalized_mixing_basis",
        "dldh_two_lmf",
        "graddft_coeff_basis_hf_pt2_heads",
    ] = DEFAULT_NEURAL_XC_ENERGY_MODE,
    input_feature_mode: Literal["enhanced", "dm21_original"] = "enhanced",
    hf_input_mode: Literal["total_only", "spin_resolved"] = "spin_resolved",
    hf_fraction_mode: Literal["normalized_weights", "hf_coefficient"] = (
        "normalized_weights"
    ),
    include_pt2_channel: bool = False,
    pt2_channel_mode: Literal["scaled_projected", "local_exact"] = "scaled_projected",
    response_hf_mode: Literal["nonlocal_exchange_only", "local_projected"] = (
        "nonlocal_exchange_only"
    ),
    response_pt2_mode: Literal["nonlocal_correlation_only", "local_projected"] = (
        "local_projected"
    ),
    strict_dm21_feature_alignment: bool = True,
    allow_experimental_jax_xc: bool = False,
    hidden_dims: Sequence[int] = GRADDFT_DEFAULT_DM21_HIDDEN_DIMS,
    activation: Callable[[Array], Array] = nn.tanh,
    network_architecture: Literal["simple_mlp", "graddft_residual"] = (
        GRADDFT_DEFAULT_NETWORK_ARCHITECTURE
    ),
    squash_offset: float = 1e-4,
    sigmoid_scale_factor: float = 2.0,
    density_floor: float = 1e-12,
    response_density_floor: float | None = 1e-5,
    kernel_clip: float = 5.0,
    response_kernel_clip: float | None = 5.0,
    dm21_hfx_channels: int = 2,
    dldh_range_separated_exchange: bool = False,
    dldh_range_separation_omega: float = _DLDH_RS_OMEGA_DEFAULT,
    dldh_qac_mode: Literal["none", "pade", "erf"] = "none",
    dldh_qac_parameters: Sequence[float] = (),
    name: str = "neural_xc",
) -> NeuralXCHybridFunctional:
    return _make_neural_xc_hybrid_functional(
        non_hf_module=non_hf_module,
        semilocal_xc=semilocal_xc,
        semilocal_energy_density_fn=semilocal_energy_density_fn,
        n_semilocal_channels=n_semilocal_channels,
        energy_mode=energy_mode,
        input_feature_mode=input_feature_mode,
        hf_input_mode=hf_input_mode,
        hf_fraction_mode=hf_fraction_mode,
        include_pt2_channel=include_pt2_channel,
        pt2_channel_mode=pt2_channel_mode,
        response_hf_mode=response_hf_mode,
        response_pt2_mode=response_pt2_mode,
        strict_dm21_feature_alignment=strict_dm21_feature_alignment,
        allow_experimental_jax_xc=allow_experimental_jax_xc,
        hidden_dims=hidden_dims,
        activation=activation,
        network_architecture=network_architecture,
        squash_offset=squash_offset,
        sigmoid_scale_factor=sigmoid_scale_factor,
        density_floor=density_floor,
        response_density_floor=response_density_floor,
        kernel_clip=kernel_clip,
        response_kernel_clip=response_kernel_clip,
        dm21_hfx_channels=dm21_hfx_channels,
        dldh_range_separated_exchange=dldh_range_separated_exchange,
        dldh_range_separation_omega=dldh_range_separation_omega,
        dldh_qac_mode=dldh_qac_mode,
        dldh_qac_parameters=dldh_qac_parameters,
        name=name,
    )
