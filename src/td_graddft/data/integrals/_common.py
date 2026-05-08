from __future__ import annotations

import math
from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array

SUPPORTED_CARTESIAN_MAX_L = 3


def _double_factorial(n: int) -> int:
    if n <= 0:
        return 1
    out = 1
    for k in range(n, 0, -2):
        out *= k
    return out


def primitive_cartesian_norm(alpha: Array, angular: tuple[int, int, int]) -> Array:
    """Normalization factor for a primitive Cartesian Gaussian.

    This follows PySCF/libcint cartesian convention (`normalized='sp'`):
    s and p are cartesian-normalized; d/f and above use shell radial norm.
    """

    lx, ly, lz = angular
    ltot = lx + ly + lz
    pref = (2.0 * alpha / jnp.pi) ** 0.75
    if ltot <= 1:
        denom = (
            _double_factorial(2 * lx - 1)
            * _double_factorial(2 * ly - 1)
            * _double_factorial(2 * lz - 1)
        )
        return pref * jnp.sqrt((4.0 * alpha) ** ltot / denom)

    # Radial norm of g(r)=r^l exp(-a r^2), consistent with pyscf.gto.gto_norm.
    # N = sqrt(2^(2l+3) (l+1)! (2a)^(l+1.5) / ((2l+2)! sqrt(pi)))
    l = int(ltot)
    numerator = (2.0 ** (2 * l + 3)) * float(math.factorial(l + 1))
    denominator = float(math.factorial(2 * l + 2)) * jnp.sqrt(jnp.pi)
    return jnp.sqrt(numerator * (2.0 * alpha) ** (l + 1.5) / denominator)


def boys0(t: Array) -> Array:
    """Boys function F0(t) with stable small-t branch."""

    t = jnp.asarray(t)
    tiny = 1e-6
    sqrt_t = jnp.sqrt(jnp.maximum(t, tiny))
    regular = 0.5 * jnp.sqrt(jnp.pi) * jax.scipy.special.erf(sqrt_t) / sqrt_t
    # Maclaurin series: F0(t) = sum_k (-t)^k / (k! (2k+1))
    small = jnp.zeros_like(t)
    term = jnp.ones_like(t)
    factorial = 1.0
    for k in range(0, 11):
        if k > 0:
            factorial *= k
            term = -term * t
        small = small + term / (factorial * (2 * k + 1))
    return jnp.where(t < tiny, small, regular)


def validate_cartesian_angular(
    angular: tuple[int, int, int], *, max_l: int = SUPPORTED_CARTESIAN_MAX_L
) -> None:
    lx, ly, lz = angular
    if min(lx, ly, lz) < 0:
        raise ValueError("Angular momentum components must be non-negative.")
    l = lx + ly + lz
    if l > max_l:
        raise NotImplementedError(
            f"Current JAX integral engine supports only cartesian l<={max_l} AOs."
        )


def _center_derivative_2c(
    fn: Callable[[Array, Array], Array],
    *,
    argnum: int,
    axis: int,
) -> Callable[[Array, Array], Array]:
    if argnum == 0:
        return lambda center_a, center_b: jax.grad(
            lambda x: fn(x, center_b)
        )(center_a)[axis]
    if argnum == 1:
        return lambda center_a, center_b: jax.grad(
            lambda x: fn(center_a, x)
        )(center_b)[axis]
    raise ValueError(f"Invalid argnum for 2-center derivative: {argnum}")


def _raise_axis_2c(
    fn: Callable[[Array, Array], Array],
    *,
    argnum: int,
    axis: int,
    exponent: Array,
    power: int,
) -> Callable[[Array, Array], Array]:
    if power == 0:
        return fn
    if power < 0:
        raise ValueError("Angular power must be non-negative.")

    scale = 2.0 * exponent
    p_nm1 = fn
    d_p0 = _center_derivative_2c(p_nm1, argnum=argnum, axis=axis)
    p_n = lambda center_a, center_b, _d=d_p0, _s=scale: _d(center_a, center_b) / _s
    if power == 1:
        return p_n

    for n in range(1, power):
        prev_nm1 = p_nm1
        prev_n = p_n
        d_prev_n = _center_derivative_2c(prev_n, argnum=argnum, axis=axis)
        p_next = (
            lambda center_a, center_b, _d=d_prev_n, _pnm1=prev_nm1, _n=n, _s=scale: (
                _d(center_a, center_b) + _n * _pnm1(center_a, center_b)
            )
            / _s
        )
        p_nm1, p_n = prev_n, p_next
    return p_n


def apply_cartesian_derivatives_2c(
    base_fn: Callable[[Array, Array], Array],
    *,
    center_a: Array,
    center_b: Array,
    alpha: Array,
    beta: Array,
    ang_a: tuple[int, int, int],
    ang_b: tuple[int, int, int],
) -> Array:
    """Lift an s-s primitive 2-center integral to cartesian AO integrals via Hermite-AD."""

    validate_cartesian_angular(ang_a, max_l=SUPPORTED_CARTESIAN_MAX_L)
    validate_cartesian_angular(ang_b, max_l=SUPPORTED_CARTESIAN_MAX_L)

    f = base_fn
    for axis, power in enumerate(ang_a):
        f = _raise_axis_2c(
            f,
            argnum=0,
            axis=axis,
            exponent=alpha,
            power=power,
        )
    for axis, power in enumerate(ang_b):
        f = _raise_axis_2c(
            f,
            argnum=1,
            axis=axis,
            exponent=beta,
            power=power,
        )
    return f(center_a, center_b)


def _center_derivative_4c(
    fn: Callable[[Array, Array, Array, Array], Array],
    *,
    argnum: int,
    axis: int,
) -> Callable[[Array, Array, Array, Array], Array]:
    if argnum == 0:
        return lambda center_a, center_b, center_c, center_d: jax.grad(
            lambda x: fn(x, center_b, center_c, center_d)
        )(center_a)[axis]
    if argnum == 1:
        return lambda center_a, center_b, center_c, center_d: jax.grad(
            lambda x: fn(center_a, x, center_c, center_d)
        )(center_b)[axis]
    if argnum == 2:
        return lambda center_a, center_b, center_c, center_d: jax.grad(
            lambda x: fn(center_a, center_b, x, center_d)
        )(center_c)[axis]
    if argnum == 3:
        return lambda center_a, center_b, center_c, center_d: jax.grad(
            lambda x: fn(center_a, center_b, center_c, x)
        )(center_d)[axis]
    raise ValueError(f"Invalid argnum for 4-center derivative: {argnum}")


def _raise_axis_4c(
    fn: Callable[[Array, Array, Array, Array], Array],
    *,
    argnum: int,
    axis: int,
    exponent: Array,
    power: int,
) -> Callable[[Array, Array, Array, Array], Array]:
    if power == 0:
        return fn
    if power < 0:
        raise ValueError("Angular power must be non-negative.")

    scale = 2.0 * exponent
    p_nm1 = fn
    d_p0 = _center_derivative_4c(p_nm1, argnum=argnum, axis=axis)
    p_n = (
        lambda center_a, center_b, center_c, center_d, _d=d_p0, _s=scale: _d(
            center_a,
            center_b,
            center_c,
            center_d,
        )
        / _s
    )
    if power == 1:
        return p_n

    for n in range(1, power):
        prev_nm1 = p_nm1
        prev_n = p_n
        d_prev_n = _center_derivative_4c(prev_n, argnum=argnum, axis=axis)
        p_next = (
            lambda center_a, center_b, center_c, center_d, _d=d_prev_n, _pnm1=prev_nm1, _n=n, _s=scale: (
                _d(center_a, center_b, center_c, center_d)
                + _n * _pnm1(center_a, center_b, center_c, center_d)
            )
            / _s
        )
        p_nm1, p_n = prev_n, p_next
    return p_n


def apply_cartesian_derivatives_4c(
    base_fn: Callable[[Array, Array, Array, Array], Array],
    *,
    center_a: Array,
    center_b: Array,
    center_c: Array,
    center_d: Array,
    alpha: Array,
    beta: Array,
    gamma: Array,
    delta: Array,
    ang_a: tuple[int, int, int],
    ang_b: tuple[int, int, int],
    ang_c: tuple[int, int, int],
    ang_d: tuple[int, int, int],
) -> Array:
    """Lift an s-s|s-s primitive ERI to cartesian AO integrals via Hermite-AD."""

    validate_cartesian_angular(ang_a, max_l=SUPPORTED_CARTESIAN_MAX_L)
    validate_cartesian_angular(ang_b, max_l=SUPPORTED_CARTESIAN_MAX_L)
    validate_cartesian_angular(ang_c, max_l=SUPPORTED_CARTESIAN_MAX_L)
    validate_cartesian_angular(ang_d, max_l=SUPPORTED_CARTESIAN_MAX_L)

    f = base_fn
    angulars = [ang_a, ang_b, ang_c, ang_d]
    exponents = [alpha, beta, gamma, delta]
    for argnum, ang in enumerate(angulars):
        for axis, power in enumerate(ang):
            f = _raise_axis_4c(
                f,
                argnum=argnum,
                axis=axis,
                exponent=exponents[argnum],
                power=power,
            )

    return f(center_a, center_b, center_c, center_d)
