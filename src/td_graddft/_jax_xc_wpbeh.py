from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp_special
from jaxtyping import Array

from .jax_libxc import RestrictedFeatureBundle


JAX_XC_WPBEH_SOURCE_COMMIT = "60c3eadeca710fede723bd449512fd12c9e9181b"
JAX_XC_WPBEH_SOURCE_PATH = "gen_repo/impl/prebuilt/unpol/gga_x_wpbeh.py"
JAX_XC_WPBEH_HELPER_SOURCE_PATH = "gen_repo/impl/utils.py"

# The generated WPBEH Maple expression is adapted from jax_xc.
# Copyright 2022 Garena Online Private Limited, Apache-2.0.
_WPBEH_ZETA_THRESHOLD = 1e-12
_WPBEH_DENS_THRESHOLD = 1e-24
_WPBEH_DENSITY_FLOOR = 1e-18
_WPBEH_SIGMA_FLOOR = 1e-30


class _JaxMath:
    pi = jnp.pi
    sqrt = staticmethod(jnp.sqrt)
    exp = staticmethod(jnp.exp)
    log = staticmethod(jnp.log)
    erf = staticmethod(jsp_special.erf)


math = _JaxMath()


def lax_cond(predicate: Array, true_value: Array | float, false_value: Array | float) -> Array:
    if isinstance(true_value, int):
        true_value = float(true_value)
    if isinstance(false_value, int):
        false_value = float(false_value)
    return jax.lax.cond(predicate, lambda _: true_value, lambda _: false_value, None)


def _e1_scaled_asymptotic(x: Array) -> Array:
    x_safe = jnp.maximum(jnp.asarray(x), jnp.asarray(1e-30, dtype=jnp.asarray(x).dtype))
    inv = 1.0 / x_safe
    series = (
        1.0
        - inv
        + 2.0 * inv**2
        - 6.0 * inv**3
        + 24.0 * inv**4
        - 120.0 * inv**5
        + 720.0 * inv**6
    )
    return inv * series


def _erfcx_asymptotic(x: Array) -> Array:
    x_safe = jnp.maximum(jnp.asarray(x), jnp.asarray(1e-30, dtype=jnp.asarray(x).dtype))
    inv_x2 = 1.0 / (x_safe * x_safe)
    series = (
        1.0
        - 0.5 * inv_x2
        + 0.75 * inv_x2**2
        - 1.875 * inv_x2**3
        + 6.5625 * inv_x2**4
        - 29.53125 * inv_x2**5
    )
    return series / (jnp.sqrt(jnp.pi) * x_safe)


@jax.custom_jvp
def xc_E1_scaled(x: Array) -> Array:
    x_arr = jnp.asarray(x)
    exact = jnp.exp(x_arr) * jsp_special.exp1(x_arr)
    asymptotic = _e1_scaled_asymptotic(x_arr)
    return jnp.where(x_arr > jnp.asarray(40.0, dtype=x_arr.dtype), asymptotic, exact)


@xc_E1_scaled.defjvp
def _xc_E1_scaled_jvp(primals: tuple[Array], tangents: tuple[Array]) -> tuple[Array, Array]:
    (x,) = primals
    (x_dot,) = tangents
    x_arr = jnp.asarray(x)
    primal = xc_E1_scaled(x_arr)
    x_safe = jnp.maximum(x_arr, jnp.asarray(1e-30, dtype=x_arr.dtype))
    tangent = x_dot * (primal - 1.0 / x_safe)
    return primal, tangent


@jax.custom_jvp
def xc_erfcx(x: Array) -> Array:
    x_arr = jnp.asarray(x)
    exact = jnp.exp(x_arr * x_arr) * jsp_special.erfc(x_arr)
    asymptotic = _erfcx_asymptotic(x_arr)
    return jnp.where(x_arr > jnp.asarray(8.0, dtype=x_arr.dtype), asymptotic, exact)


@xc_erfcx.defjvp
def _xc_erfcx_jvp(primals: tuple[Array], tangents: tuple[Array]) -> tuple[Array, Array]:
    (x,) = primals
    (x_dot,) = tangents
    x_arr = jnp.asarray(x)
    primal = xc_erfcx(x_arr)
    tangent = x_dot * (2.0 * x_arr * primal - 2.0 / jnp.sqrt(jnp.pi))
    return primal, tangent


def _wpbeh_unpolarized_epsilon_scalar(
    r0: Array,
    s0: Array,
    p_a_cam_omega: Array,
) -> Array:
    """Unpolarized GGA_X_WPBEH exchange epsilon from jax_xc/LibXC.

    The expression below is the generated jax_xc prebuilt formula for
    ``gen_repo/impl/prebuilt/unpol/gga_x_wpbeh.py``. It returns exchange
    energy per particle; callers multiply by density or apply exchange spin
    scaling.
    """

    r0_raw = jnp.asarray(r0)
    r0 = jnp.maximum(r0_raw, _WPBEH_DENSITY_FLOOR)
    s0 = jnp.maximum(jnp.asarray(s0), _WPBEH_SIGMA_FLOOR)
    p_a_cam_omega = jnp.asarray(p_a_cam_omega, dtype=r0.dtype)
    p_a_zeta_threshold = jnp.asarray(_WPBEH_ZETA_THRESHOLD, dtype=r0.dtype)
    p_a_dens_threshold = jnp.asarray(_WPBEH_DENS_THRESHOLD, dtype=r0.dtype)

    t4 = 3 ** (0.1e1 / 0.3e1)
    t5 = math.pi ** (0.1e1 / 0.3e1)
    t8 = 0.1e1 <= p_a_zeta_threshold
    t9 = p_a_zeta_threshold - 0.1e1
    t12 = lax_cond(t8, -t9, 0)
    t13 = lax_cond(t8, t9, t12)
    t14 = 0.1e1 + t13
    t15 = t14 <= p_a_zeta_threshold
    t16 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
    t18 = t14 ** (0.1e1 / 0.3e1)
    t20 = lax_cond(t15, t16 * p_a_zeta_threshold, t18 * t14)
    t21 = r0 ** (0.1e1 / 0.3e1)
    t23 = t4 ** 2
    t24 = p_a_cam_omega * t23
    t25 = math.pi ** 2
    t26 = t25 ** (0.1e1 / 0.3e1)
    t27 = 0.1e1 / t26
    t28 = lax_cond(t15, t16, t18)
    t30 = t27 / t28
    t31 = 0.1e1 / t21
    t34 = t24 * t30 * t31 / 0.3e1
    t35 = 0.14e2 < t34
    t36 = 6 ** (0.1e1 / 0.3e1)
    t37 = t36 ** 2
    t39 = math.sqrt(s0)
    t40 = 2 ** (0.1e1 / 0.3e1)
    t46 = t37 * t27 * t39 * t40 / t21 / r0 / 0.12e2
    t48 = 0.15e2 < t46
    t49 = lax_cond(t48, 15, t46)
    t51 = lax_cond(0.1e1 < t49, t49, 1)
    t53 = math.exp(t51 - 0.8572844e1)
    t55 = math.log(0.1e1 + t53)
    t57 = lax_cond(t48, 0.8572844e1, t51 - t55)
    t58 = lax_cond(t46 < 0.1e1, t46, t57)
    t60 = lax_cond(t58 < 0.1e-14, 0.1e-14, t58)
    t61 = t60 ** 2
    t63 = t61 ** 2
    t65 = 0.979681e-2 * t61 + 0.410834e-1 * t63
    t73 = 0.1e1 / (0.1e1 + 0.187440e0 * t63 + 0.120824e-2 * t63 * t60 + 0.347188e-1 * t63 * t61)
    t74 = t61 * t65 * t73
    t75 = 0.22143176004591608976e1 * t74
    t77 = lax_cond(t34 < 0.14e2, 0.1455915450052607e1, 2)
    t78 = p_a_cam_omega ** 2
    t81 = t26 ** 2
    t83 = t28 ** 2
    t86 = t21 ** 2
    t88 = 0.1e1 / t81 / t83 / t86
    t89 = t77 * t78 * t4 * t88
    t92 = xc_E1_scaled(t75 + 0.73810586681972029922e0 * t89)
    t94 = t89 / 0.3e1
    t96 = math.log(0.57786348e0 + t74 + t94)
    t99 = math.log(t74 + t94)
    t102 = lax_cond(t35, 14, t34)
    t104 = t102 ** 2
    t105 = t104 * t102
    t107 = t104 ** 2
    t108 = t107 * t102
    t110 = t107 * t105
    t115 = lax_cond(t102 < 0.14e2, 0.1455915450052607e1, 2)
    t116 = t115 * t104
    t118 = t75 + 0.22143176004591608976e1 * t116
    t119 = math.sqrt(t118)
    t120 = xc_erfcx(t119)
    t125 = t107 * t104
    t127 = t107 ** 2
    t130 = xc_E1_scaled(t118)
    t133 = math.sqrt(math.pi)
    t134 = 0.57786348e0 + t74 + t116
    t135 = math.sqrt(t134)
    t140 = 0.1e1 / t134
    t143 = t74 + t116
    t144 = math.sqrt(t143)
    t156 = t134 ** 2
    t161 = t135 * t156
    t167 = t144 * t143
    t177 = t156 * t134
    t182 = t143 ** 2
    t188 = t144 * t182
    t190 = t135 * t177
    t205 = t156 ** 2
    t209 = t182 * t143
    t225 = math.log(t143 * t140)
    t227 = (0.17059169152930056821e1 * t102 - 0.41622705406440396562e1 * t105 + 0.42174370348694648999e1 * t108 - 0.10676080470633097775e1 * t110) * math.pi * t120 / 0.2e1 - (-0.10161144e1 + 0.32686565979666847500e1 * t104 - 0.48418398881417585092e1 * t107 + 0.27236365685865660550e1 * t125 - 0.20524577845574895866e0 * t127) * t130 / 0.2e1 - 0.57320229933645902590e0 * t133 / t135 * t102 + 0.73807311952199090995e0 * t140 * t104 - 0.1243162299390327e1 * t133 * (-0.9e1 / 0.8e1 / t144 + 0.254028600e0 / t135 / t134) * t105 + (-0.10933029406300511250e1 / t143 + 0.49374260512735112038e0 / t156) * t107 - 0.52484962540331303985e-1 * t133 * (0.3e1 * t161 * (0.9e1 * t74 + 0.9e1 * t116 - 0.20322288e1) + 0.412995389554944e1 * t167) / t161 / t167 * t108 + (0.25085884618821050197e0 / t177 + 0.77150160881310000000e-2 * (-0.36e2 + 0.79715433616529792314e2 * t74) / t182) * t125 + 0.14762353927435135389e-2 * t133 * (-0.41965056246038818960e2 * t188 + 0.9e1 * t190 * (0.27e2 * t182 - 0.60966864e1 * t74 - 0.60966864e1 * t116 + 0.412995389554944e1)) / t190 / t188 * t110 + 0.75666704254679261017e-2 * (0.81278266164980202635e2 * t115 * t205 * t143 + 0.33847844843765416574e1 * t209 + 0.8401793031216e-2 * t205 * (-0.729e3 * t182 + 0.3292210656e3 * t74 + 0.3292210656e3 * t116 - 0.29735668047955968e3)) / t205 / t209 * t127 + 0.50805720000000000000e0 * t225
    t228 = lax_cond(t35, 0.50805720000000000000e0 * t92 - 0.50805720000000000000e0 * t96 + 0.50805720000000000000e0 * t99, t227)
    t230 = 0.57786348e0 + t74
    t231 = t230 ** 2
    t233 = 0.77215461e-1 * t74
    t237 = 0.64753871e1 * t65 * t73 + 0.47965830e0
    t247 = t231 * t230
    t251 = math.sqrt(t230)
    t252 = t251 * t247
    t256 = math.exp(t75)
    t258 = math.sqrt(t74)
    t260 = math.erf(0.14880583323442535321e1 * t258)
    t274 = lax_cond(0.8e-1 < t60, -0.16e2 / 0.15e2 * (0.3e1 / 0.4e1 * math.pi + t133 * (-0.779335965e0 - 0.463292766e0 * (t237 * t61 + 0.1e1) * t230 - 0.148683344e1 * t231 + 0.81289152e1 * t247) / t252 / 0.16e2 - 0.75601874976749088562e0 * math.pi * t256 * (0.1e1 - t260)) / t133 / t61 * t252, -0.2628417880e-1 - 0.7117647788e-1 * t61 + 0.8534541323e-1 * t63)
    t275 = t61 * t274
    t278 = 0.1e1 / t247
    t283 = t78 * t4 * t88
    t285 = 0.57786348e0 + t74 + t283 / 0.3e1
    t286 = t285 ** 2
    t291 = t285 * t61 * t237
    t297 = math.sqrt(t285)
    t299 = 0.1e1 / t297 / t286
    t321 = t78 ** 2
    t326 = t83 ** 2
    t343 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t4 / t5 * t20 * t21 * (-0.8e1 / 0.9e1 * t228 - 0.4e1 / 0.9e1 * (-0.37170836e0 * t231 - 0.14853145700326428e0 - t233 - 0.77215461e-1 * t230 * t61 * t237 + 0.2e1 * t275) * t278 + t24 * t30 * t31 * (-0.148683344e1 * t286 - 0.104705593501958568e1 - 0.463292766e0 * t74 - 0.15443092200000000000e0 * t283 - 0.463292766e0 * t291 + 0.15e2 * t275) / t230 * t299 / 0.27e2 + 0.4e1 / 0.27e2 * t78 * p_a_cam_omega / t25 / t83 / t28 / r0 * (-0.30439865000326428e0 - t233 - 0.25738487000000000000e-1 * t283 - 0.77215461e-1 * t291 + 0.5e1 * t275) / t231 * t299 + 0.8e1 / 0.81e2 * t321 * p_a_cam_omega * t4 / t81 / t25 / t326 / t28 / t86 / r0 * (-0.51955731e-1 + t275) * t278 * t299))
    res = 0.2e1 * t343
    return jnp.where(r0_raw / 0.2e1 <= p_a_dens_threshold, 0.0, res)


def _wpbeh_unpolarized_epsilon(rho: Array, sigma: Array, omega: Array) -> Array:
    rho_arr = jnp.asarray(rho)
    sigma_arr = jnp.asarray(sigma)
    out_shape = jnp.broadcast_shapes(rho_arr.shape, sigma_arr.shape)
    rho_flat = jnp.broadcast_to(rho_arr, out_shape).reshape(-1)
    sigma_flat = jnp.broadcast_to(sigma_arr, out_shape).reshape(-1)
    flat = jax.vmap(
        lambda rho_value, sigma_value: _wpbeh_unpolarized_epsilon_scalar(
            rho_value,
            sigma_value,
            omega,
        )
    )(rho_flat, sigma_flat)
    return flat.reshape(out_shape)


def gga_x_wpbeh_energy_density(features: RestrictedFeatureBundle, *, omega: Array = 0.4) -> Array:
    rho_a = jnp.maximum(jnp.asarray(features.rho_a), 0.0)
    rho_b = jnp.maximum(jnp.asarray(features.rho_b), 0.0)
    sigma_aa = jnp.maximum(jnp.asarray(features.sigma_aa), 0.0)
    sigma_bb = jnp.maximum(jnp.asarray(features.sigma_bb), 0.0)
    eps_a = _wpbeh_unpolarized_epsilon(2.0 * rho_a, 4.0 * sigma_aa, omega)
    eps_b = _wpbeh_unpolarized_epsilon(2.0 * rho_b, 4.0 * sigma_bb, omega)
    density = rho_a * eps_a + rho_b * eps_b
    return jnp.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)


__all__ = [
    "JAX_XC_WPBEH_HELPER_SOURCE_PATH",
    "JAX_XC_WPBEH_SOURCE_COMMIT",
    "JAX_XC_WPBEH_SOURCE_PATH",
    "gga_x_wpbeh_energy_density",
]
