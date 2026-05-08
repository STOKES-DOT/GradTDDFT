from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln
from jaxtyping import Array

from .basis import CartesianBasis


_DOUBLE_FACTORIAL_LOOKUP = jnp.asarray([1.0, 1.0, 3.0, 15.0])


def _as_device_array(value, dtype=None):
    if isinstance(value, jax.core.Tracer):
        return jnp.asarray(value, dtype=dtype)
    if isinstance(value, jax.Array):
        return value if dtype is None else value.astype(dtype)
    return jax.device_put(np.asarray(value, dtype=dtype))


def _primitive_cartesian_norm_array(exponents: Array, angulars: Array) -> Array:
    """Vectorized primitive cartesian normalization matching PySCF cart convention."""

    exponents = jnp.asarray(exponents)
    angulars = jnp.asarray(angulars, dtype=jnp.int32)
    lx = angulars[:, 0][:, None]
    ly = angulars[:, 1][:, None]
    lz = angulars[:, 2][:, None]
    ltot = lx + ly + lz

    pref = (2.0 * exponents / jnp.pi) ** 0.75
    denom = (
        _DOUBLE_FACTORIAL_LOOKUP[lx]
        * _DOUBLE_FACTORIAL_LOOKUP[ly]
        * _DOUBLE_FACTORIAL_LOOKUP[lz]
    )
    cart = pref * jnp.sqrt((4.0 * exponents) ** ltot / denom)

    ltot_f = ltot.astype(exponents.dtype)
    numerator = (2.0 ** (2.0 * ltot_f + 3.0)) * jnp.exp(gammaln(ltot_f + 2.0))
    denominator = jnp.exp(gammaln(2.0 * ltot_f + 3.0)) * jnp.sqrt(jnp.pi)
    radial = jnp.sqrt(
        numerator * (2.0 * exponents) ** (ltot_f + 1.5) / denominator
    )
    return jnp.where(ltot <= 1, cart, radial)


@functools.partial(jax.jit, static_argnames=("deriv",))
def _evaluate_cartesian_ao_kernel(
    coords: Array,
    centers: Array,
    exponents: Array,
    coefficients: Array,
    nprims: Array,
    angulars: Array,
    *,
    deriv: int = 0,
) -> Array:
    if deriv not in (0, 1, 2):
        raise NotImplementedError("_evaluate_cartesian_ao_kernel supports deriv=0, 1, or 2.")

    ngrids = int(coords.shape[0])
    nao = int(centers.shape[0])
    if nao == 0:
        if deriv == 0:
            return jnp.zeros((ngrids, 0), dtype=coords.dtype)
        return jnp.zeros((4, ngrids, 0), dtype=coords.dtype)

    dtype = jnp.result_type(coords.dtype, exponents.dtype, coefficients.dtype)
    coords = coords.astype(dtype)
    centers = centers.astype(dtype)
    exponents = exponents.astype(dtype)
    coefficients = coefficients.astype(dtype)

    dx = coords[:, None, 0:1] - centers[None, :, 0:1]
    dy = coords[:, None, 1:2] - centers[None, :, 1:2]
    dz = coords[:, None, 2:3] - centers[None, :, 2:3]
    r2 = dx * dx + dy * dy + dz * dz

    lx = angulars[:, 0].astype(jnp.int32)[None, :, None]
    ly = angulars[:, 1].astype(jnp.int32)[None, :, None]
    lz = angulars[:, 2].astype(jnp.int32)[None, :, None]

    prim_mask = (
        jnp.arange(exponents.shape[1], dtype=jnp.int32)[None, None, :]
        < nprims[None, :, None]
    ).astype(dtype)
    norms = _primitive_cartesian_norm_array(exponents, angulars).astype(dtype)

    exp_term = jnp.exp(-exponents[None, :, :] * r2)
    coeff_pref = coefficients[None, :, :] * norms[None, :, :] * prim_mask

    x_l = dx**lx
    y_l = dy**ly
    z_l = dz**lz
    xyz = x_l * y_l * z_l
    primitive_values = coeff_pref * xyz * exp_term
    ao = jnp.sum(primitive_values, axis=-1)
    if deriv == 0:
        return ao

    x_lm1 = dx ** jnp.maximum(lx - 1, 0)
    y_lm1 = dy ** jnp.maximum(ly - 1, 0)
    z_lm1 = dz ** jnp.maximum(lz - 1, 0)
    x_lp1 = dx ** (lx + 1)
    y_lp1 = dy ** (ly + 1)
    z_lp1 = dz ** (lz + 1)

    d_primitive_x = coeff_pref * exp_term * (
        lx.astype(dtype) * x_lm1 * y_l * z_l
        - 2.0 * exponents[None, :, :] * x_lp1 * y_l * z_l
    )
    d_primitive_y = coeff_pref * exp_term * (
        ly.astype(dtype) * x_l * y_lm1 * z_l
        - 2.0 * exponents[None, :, :] * x_l * y_lp1 * z_l
    )
    d_primitive_z = coeff_pref * exp_term * (
        lz.astype(dtype) * x_l * y_l * z_lm1
        - 2.0 * exponents[None, :, :] * x_l * y_l * z_lp1
    )

    deriv_x = jnp.sum(d_primitive_x, axis=-1)
    deriv_y = jnp.sum(d_primitive_y, axis=-1)
    deriv_z = jnp.sum(d_primitive_z, axis=-1)
    if deriv == 1:
        return jnp.stack([ao, deriv_x, deriv_y, deriv_z], axis=0)

    lx_minus_two = jnp.maximum(lx - 2, 0)
    ly_minus_two = jnp.maximum(ly - 2, 0)
    lz_minus_two = jnp.maximum(lz - 2, 0)
    x_lm2 = dx**lx_minus_two
    y_lm2 = dy**ly_minus_two
    z_lm2 = dz**lz_minus_two

    lx_f = lx.astype(dtype)
    ly_f = ly.astype(dtype)
    lz_f = lz.astype(dtype)
    exponents_b = exponents[None, :, :]
    two_alpha = 2.0 * exponents_b
    four_alpha_sq = 4.0 * exponents_b * exponents_b

    d2_primitive_x = coeff_pref * exp_term * (
        lx_f * jnp.maximum(lx_f - 1.0, 0.0) * x_lm2 * y_l * z_l
        - two_alpha * (2.0 * lx_f + 1.0) * x_l * y_l * z_l
        + four_alpha_sq * x_lp1 * dx * y_l * z_l
    )
    d2_primitive_y = coeff_pref * exp_term * (
        ly_f * jnp.maximum(ly_f - 1.0, 0.0) * x_l * y_lm2 * z_l
        - two_alpha * (2.0 * ly_f + 1.0) * x_l * y_l * z_l
        + four_alpha_sq * x_l * y_lp1 * dy * z_l
    )
    d2_primitive_z = coeff_pref * exp_term * (
        lz_f * jnp.maximum(lz_f - 1.0, 0.0) * x_l * y_l * z_lm2
        - two_alpha * (2.0 * lz_f + 1.0) * x_l * y_l * z_l
        + four_alpha_sq * x_l * y_l * z_lp1 * dz
    )
    laplacian = jnp.sum(d2_primitive_x + d2_primitive_y + d2_primitive_z, axis=-1)
    return jnp.stack([ao, deriv_x, deriv_y, deriv_z, laplacian], axis=0)


@functools.partial(jax.jit, static_argnames=("deriv",))
def _evaluate_cartesian_ao_value_and_derivatives_kernel(
    coords: Array,
    centers: Array,
    exponents: Array,
    coefficients: Array,
    nprims: Array,
    angulars: Array,
    *,
    deriv: int,
) -> tuple[Array, Array]:
    values = _evaluate_cartesian_ao_kernel.__wrapped__(
        coords,
        centers,
        exponents,
        coefficients,
        nprims,
        angulars,
        deriv=deriv,
    )
    return values[0], values


def _basis_kernel_arrays(basis: CartesianBasis, coords: Array):
    return (
        _as_device_array(coords),
        _as_device_array(basis.ao_centers),
        _as_device_array(basis.ao_exponents_padded),
        _as_device_array(basis.ao_coefficients_padded),
        _as_device_array(basis.ao_nprims, dtype=np.int32),
        _as_device_array(basis.ao_angulars, dtype=np.int32),
    )


def evaluate_cartesian_ao(
    basis: CartesianBasis,
    coords: Array,
    *,
    deriv: int = 0,
    chunk_size: int | None = None,
) -> Array:
    """Evaluate contracted cartesian AO values on a quadrature grid.

    Shapes:
    - ``deriv=0``: ``(ngrids, nao)``
    - ``deriv=1``: ``(4, ngrids, nao)`` where axis 0 is ``[value, dx, dy, dz]``
    - ``deriv=2``: ``(5, ngrids, nao)`` where axis 0 is ``[value, dx, dy, dz, laplacian]``
    """

    coords_arr, centers, exponents, coefficients, nprims, angulars = _basis_kernel_arrays(
        basis,
        coords,
    )

    if chunk_size is None or int(chunk_size) <= 0:
        return _evaluate_cartesian_ao_kernel(
            coords_arr,
            centers,
            exponents,
            coefficients,
            nprims,
            angulars,
            deriv=deriv,
        )

    block = int(chunk_size)
    ngrids = int(coords_arr.shape[0])
    if ngrids <= block:
        return _evaluate_cartesian_ao_kernel(
            coords_arr,
            centers,
            exponents,
            coefficients,
            nprims,
            angulars,
            deriv=deriv,
        )

    outputs: list[Array] = []
    for start in range(0, ngrids, block):
        stop = min(start + block, ngrids)
        coords_block = coords_arr[start:stop]
        block_len = stop - start
        if block_len < block:
            coords_block = jnp.pad(coords_block, ((0, block - block_len), (0, 0)))
        block_out = _evaluate_cartesian_ao_kernel(
            coords_block,
            centers,
            exponents,
            coefficients,
            nprims,
            angulars,
            deriv=deriv,
        )
        if block_len < block:
            if deriv == 0:
                block_out = block_out[:block_len, :]
            else:
                block_out = block_out[:, :block_len, :]
        outputs.append(block_out)

    if deriv == 0:
        return jnp.concatenate(outputs, axis=0)
    return jnp.concatenate(outputs, axis=1)


def evaluate_cartesian_ao_with_derivatives(
    basis: CartesianBasis,
    coords: Array,
    *,
    deriv: int = 1,
) -> tuple[Array, Array]:
    if deriv not in (1, 2):
        raise NotImplementedError("evaluate_cartesian_ao_with_derivatives supports deriv=1 or 2.")
    coords_arr, centers, exponents, coefficients, nprims, angulars = _basis_kernel_arrays(
        basis,
        coords,
    )
    return _evaluate_cartesian_ao_value_and_derivatives_kernel(
        coords_arr,
        centers,
        exponents,
        coefficients,
        nprims,
        angulars,
        deriv=deriv,
    )
