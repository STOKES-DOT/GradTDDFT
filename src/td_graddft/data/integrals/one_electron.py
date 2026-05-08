from __future__ import annotations

import functools
import weakref
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..basis import CartesianAO, CartesianBasis, PairBatchGroup
from ._common import (
    SUPPORTED_CARTESIAN_MAX_L,
    apply_cartesian_derivatives_2c,
    boys0,
    primitive_cartesian_norm,
    validate_cartesian_angular,
)

PAIR_BATCH_CHUNK = 256
_RINV_CHUNK_BUILDERS: dict[
    int,
    tuple[weakref.ReferenceType[CartesianBasis], dict[str, Callable[..., Array]]],
] = {}


def _primitive_overlap_ss(alpha: Array, beta: Array, center_a: Array, center_b: Array) -> Array:
    p = alpha + beta
    mu = alpha * beta / p
    rab2 = jnp.dot(center_a - center_b, center_a - center_b)
    return (jnp.pi / p) ** 1.5 * jnp.exp(-mu * rab2)


def _primitive_kinetic_ss(alpha: Array, beta: Array, center_a: Array, center_b: Array) -> Array:
    p = alpha + beta
    mu = alpha * beta / p
    rab2 = jnp.dot(center_a - center_b, center_a - center_b)
    sss = _primitive_overlap_ss(alpha, beta, center_a, center_b)
    return mu * (3.0 - 2.0 * mu * rab2) * sss


def _primitive_nuclear_ss(
    alpha: Array,
    beta: Array,
    center_a: Array,
    center_b: Array,
    atom_coords: Array,
    atom_charges: Array,
) -> Array:
    p = alpha + beta
    mu = alpha * beta / p
    rab2 = jnp.dot(center_a - center_b, center_a - center_b)
    center_p = (alpha * center_a + beta * center_b) / p
    pref = -2.0 * jnp.pi / p * jnp.exp(-mu * rab2)

    diffs = center_p[None, :] - atom_coords
    t = p * jnp.einsum("ni,ni->n", diffs, diffs)
    return pref * jnp.sum(atom_charges * boys0(t))


def _primitive_dipole_ss(
    alpha: Array,
    beta: Array,
    center_a: Array,
    center_b: Array,
    origin: Array,
    axis: int,
) -> Array:
    p = alpha + beta
    mu = alpha * beta / p
    rab2 = jnp.dot(center_a - center_b, center_a - center_b)
    center_p = (alpha * center_a + beta * center_b) / p
    overlap = (jnp.pi / p) ** 1.5 * jnp.exp(-mu * rab2)
    return (center_p[int(axis)] - origin[int(axis)]) * overlap


def _primitive_rinv_ss(
    alpha: Array,
    beta: Array,
    center_a: Array,
    center_b: Array,
    origin: Array,
    zeta: Array,
) -> Array:
    p = alpha + beta
    mu = alpha * beta / p
    rab2 = jnp.dot(center_a - center_b, center_a - center_b)
    center_p = (alpha * center_a + beta * center_b) / p
    rpc2 = jnp.dot(center_p - origin, center_p - origin)

    zeta_arr = jnp.asarray(zeta, dtype=p.dtype)
    is_point = jnp.isinf(zeta_arr)
    positive = zeta_arr > 0.0
    tiny = jnp.finfo(p.dtype).tiny
    safe_zeta = jnp.where(is_point, 1.0, jnp.maximum(zeta_arr, tiny))
    denom_ratio = 1.0 + p / safe_zeta
    pref_scale = jnp.where(
        is_point,
        1.0,
        jnp.where(positive, denom_ratio ** -0.5, 0.0),
    )
    rho = jnp.where(
        is_point,
        p,
        jnp.where(positive, p / denom_ratio, 0.0),
    )
    pref = 2.0 * jnp.pi / p * jnp.exp(-mu * rab2) * pref_scale
    return pref * boys0(rho * rpc2)


def _flatten_primitive_pairs_2c(
    exponents_i: Array,
    coefficients_i: Array,
    exponents_j: Array,
    coefficients_j: Array,
    *,
    angular_i: tuple[int, int, int],
    angular_j: tuple[int, int, int],
) -> tuple[Array, Array, Array]:
    norm_i = primitive_cartesian_norm(exponents_i, angular_i)
    norm_j = primitive_cartesian_norm(exponents_j, angular_j)
    weighted_i = coefficients_i * norm_i
    weighted_j = coefficients_j * norm_j
    alpha, beta = jnp.meshgrid(exponents_i, exponents_j, indexing="ij")
    weights = weighted_i[:, None] * weighted_j[None, :]
    return alpha.reshape(-1), beta.reshape(-1), weights.reshape(-1)


def _contracted_pair_integral_2c(
    ao_i: CartesianAO,
    ao_j: CartesianAO,
    primitive_ss_factory,
) -> Array:
    validate_cartesian_angular(ao_i.angular, max_l=SUPPORTED_CARTESIAN_MAX_L)
    validate_cartesian_angular(ao_j.angular, max_l=SUPPORTED_CARTESIAN_MAX_L)
    value = jnp.asarray(0.0)
    for ip in range(int(ao_i.exponents.shape[0])):
        alpha = ao_i.exponents[ip]
        ci = ao_i.coefficients[ip]
        ni = primitive_cartesian_norm(alpha, ao_i.angular)
        for jp in range(int(ao_j.exponents.shape[0])):
            beta = ao_j.exponents[jp]
            cj = ao_j.coefficients[jp]
            nj = primitive_cartesian_norm(beta, ao_j.angular)

            base_fn = lambda a, b, _alpha=alpha, _beta=beta: primitive_ss_factory(
                _alpha,
                _beta,
                a,
                b,
            )
            prim = apply_cartesian_derivatives_2c(
                base_fn,
                center_a=ao_i.center,
                center_b=ao_j.center,
                alpha=alpha,
                beta=beta,
                ang_a=ao_i.angular,
                ang_b=ao_j.angular,
            )
            value = value + ci * cj * ni * nj * prim
    return value


def _contracted_pair_dipole_integral(
    ao_i: CartesianAO,
    ao_j: CartesianAO,
    *,
    origin: Array,
    axis: int,
) -> Array:
    validate_cartesian_angular(ao_i.angular, max_l=SUPPORTED_CARTESIAN_MAX_L)
    validate_cartesian_angular(ao_j.angular, max_l=SUPPORTED_CARTESIAN_MAX_L)
    value = jnp.asarray(0.0)
    for ip in range(int(ao_i.exponents.shape[0])):
        alpha = ao_i.exponents[ip]
        ci = ao_i.coefficients[ip]
        ni = primitive_cartesian_norm(alpha, ao_i.angular)
        for jp in range(int(ao_j.exponents.shape[0])):
            beta = ao_j.exponents[jp]
            cj = ao_j.coefficients[jp]
            nj = primitive_cartesian_norm(beta, ao_j.angular)

            base_fn = (
                lambda a, b, _alpha=alpha, _beta=beta: _primitive_dipole_ss(
                    _alpha,
                    _beta,
                    a,
                    b,
                    origin,
                    axis,
                )
            )
            prim = apply_cartesian_derivatives_2c(
                base_fn,
                center_a=ao_i.center,
                center_b=ao_j.center,
                alpha=alpha,
                beta=beta,
                ang_a=ao_i.angular,
                ang_b=ao_j.angular,
            )
            value = value + ci * cj * ni * nj * prim
    return value


def _contracted_pair_rinv_integral(
    ao_i: CartesianAO,
    ao_j: CartesianAO,
    *,
    origin: Array,
    zeta: Array,
) -> Array:
    validate_cartesian_angular(ao_i.angular, max_l=SUPPORTED_CARTESIAN_MAX_L)
    validate_cartesian_angular(ao_j.angular, max_l=SUPPORTED_CARTESIAN_MAX_L)
    value = jnp.asarray(0.0)
    for ip in range(int(ao_i.exponents.shape[0])):
        alpha = ao_i.exponents[ip]
        ci = ao_i.coefficients[ip]
        ni = primitive_cartesian_norm(alpha, ao_i.angular)
        for jp in range(int(ao_j.exponents.shape[0])):
            beta = ao_j.exponents[jp]
            cj = ao_j.coefficients[jp]
            nj = primitive_cartesian_norm(beta, ao_j.angular)

            base_fn = (
                lambda a, b, _alpha=alpha, _beta=beta: _primitive_rinv_ss(
                    _alpha,
                    _beta,
                    a,
                    b,
                    origin,
                    zeta,
                )
            )
            prim = apply_cartesian_derivatives_2c(
                base_fn,
                center_a=ao_i.center,
                center_b=ao_j.center,
                alpha=alpha,
                beta=beta,
                ang_a=ao_i.angular,
                ang_b=ao_j.angular,
            )
            value = value + ci * cj * ni * nj * prim
    return value


def _contracted_pair_nuclear_integral(
    ao_i: CartesianAO,
    ao_j: CartesianAO,
    *,
    atom_coords: Array,
    atom_charges: Array,
) -> Array:
    validate_cartesian_angular(ao_i.angular, max_l=SUPPORTED_CARTESIAN_MAX_L)
    validate_cartesian_angular(ao_j.angular, max_l=SUPPORTED_CARTESIAN_MAX_L)
    value = jnp.asarray(0.0)
    for ip in range(int(ao_i.exponents.shape[0])):
        alpha = ao_i.exponents[ip]
        ci = ao_i.coefficients[ip]
        ni = primitive_cartesian_norm(alpha, ao_i.angular)
        for jp in range(int(ao_j.exponents.shape[0])):
            beta = ao_j.exponents[jp]
            cj = ao_j.coefficients[jp]
            nj = primitive_cartesian_norm(beta, ao_j.angular)

            base_fn = (
                lambda a, b, _alpha=alpha, _beta=beta: _primitive_nuclear_ss(
                    _alpha,
                    _beta,
                    a,
                    b,
                    atom_coords,
                    atom_charges,
                )
            )
            prim = apply_cartesian_derivatives_2c(
                base_fn,
                center_a=ao_i.center,
                center_b=ao_j.center,
                alpha=alpha,
                beta=beta,
                ang_a=ao_i.angular,
                ang_b=ao_j.angular,
            )
            value = value + ci * cj * ni * nj * prim
    return value


@functools.lru_cache(maxsize=None)
def _compiled_overlap_pair_kernel(
    angular_i: tuple[int, int, int],
    angular_j: tuple[int, int, int],
    nprim_i: int,
    nprim_j: int,
):
    def kernel(
        exponents_i: Array,
        coefficients_i: Array,
        center_i: Array,
        exponents_j: Array,
        coefficients_j: Array,
        center_j: Array,
    ) -> Array:
        alpha_flat, beta_flat, weight_flat = _flatten_primitive_pairs_2c(
            exponents_i,
            coefficients_i,
            exponents_j,
            coefficients_j,
            angular_i=angular_i,
            angular_j=angular_j,
        )

        def primitive_value(alpha: Array, beta: Array) -> Array:
            base_fn = lambda a, b: _primitive_overlap_ss(alpha, beta, a, b)
            return apply_cartesian_derivatives_2c(
                base_fn,
                center_a=center_i,
                center_b=center_j,
                alpha=alpha,
                beta=beta,
                ang_a=angular_i,
                ang_b=angular_j,
            )

        prim = jax.vmap(primitive_value)(alpha_flat, beta_flat)
        return jnp.dot(weight_flat, prim, precision=jax.lax.Precision.HIGHEST)

    return jax.jit(kernel)


@functools.lru_cache(maxsize=None)
def _compiled_overlap_pair_kernel_batched(
    angular_i: tuple[int, int, int],
    angular_j: tuple[int, int, int],
    nprim_i: int,
    nprim_j: int,
):
    scalar = _compiled_overlap_pair_kernel(angular_i, angular_j, nprim_i, nprim_j)
    return jax.jit(jax.vmap(scalar, in_axes=(0, 0, 0, 0, 0, 0)))


@functools.lru_cache(maxsize=None)
def _compiled_kinetic_pair_kernel(
    angular_i: tuple[int, int, int],
    angular_j: tuple[int, int, int],
    nprim_i: int,
    nprim_j: int,
):
    def kernel(
        exponents_i: Array,
        coefficients_i: Array,
        center_i: Array,
        exponents_j: Array,
        coefficients_j: Array,
        center_j: Array,
    ) -> Array:
        alpha_flat, beta_flat, weight_flat = _flatten_primitive_pairs_2c(
            exponents_i,
            coefficients_i,
            exponents_j,
            coefficients_j,
            angular_i=angular_i,
            angular_j=angular_j,
        )

        def primitive_value(alpha: Array, beta: Array) -> Array:
            base_fn = lambda a, b: _primitive_kinetic_ss(alpha, beta, a, b)
            return apply_cartesian_derivatives_2c(
                base_fn,
                center_a=center_i,
                center_b=center_j,
                alpha=alpha,
                beta=beta,
                ang_a=angular_i,
                ang_b=angular_j,
            )

        prim = jax.vmap(primitive_value)(alpha_flat, beta_flat)
        return jnp.dot(weight_flat, prim, precision=jax.lax.Precision.HIGHEST)

    return jax.jit(kernel)


@functools.lru_cache(maxsize=None)
def _compiled_kinetic_pair_kernel_batched(
    angular_i: tuple[int, int, int],
    angular_j: tuple[int, int, int],
    nprim_i: int,
    nprim_j: int,
):
    scalar = _compiled_kinetic_pair_kernel(angular_i, angular_j, nprim_i, nprim_j)
    return jax.jit(jax.vmap(scalar, in_axes=(0, 0, 0, 0, 0, 0)))


@functools.lru_cache(maxsize=None)
def _compiled_nuclear_pair_kernel(
    angular_i: tuple[int, int, int],
    angular_j: tuple[int, int, int],
    nprim_i: int,
    nprim_j: int,
):
    def kernel(
        exponents_i: Array,
        coefficients_i: Array,
        center_i: Array,
        exponents_j: Array,
        coefficients_j: Array,
        center_j: Array,
        atom_coords: Array,
        atom_charges: Array,
    ) -> Array:
        alpha_flat, beta_flat, weight_flat = _flatten_primitive_pairs_2c(
            exponents_i,
            coefficients_i,
            exponents_j,
            coefficients_j,
            angular_i=angular_i,
            angular_j=angular_j,
        )

        def primitive_value(alpha: Array, beta: Array) -> Array:
            base_fn = lambda a, b: _primitive_nuclear_ss(
                alpha,
                beta,
                a,
                b,
                atom_coords,
                atom_charges,
            )
            return apply_cartesian_derivatives_2c(
                base_fn,
                center_a=center_i,
                center_b=center_j,
                alpha=alpha,
                beta=beta,
                ang_a=angular_i,
                ang_b=angular_j,
            )

        prim = jax.vmap(primitive_value)(alpha_flat, beta_flat)
        return jnp.dot(weight_flat, prim, precision=jax.lax.Precision.HIGHEST)

    return jax.jit(kernel)


@functools.lru_cache(maxsize=None)
def _compiled_nuclear_pair_kernel_batched(
    angular_i: tuple[int, int, int],
    angular_j: tuple[int, int, int],
    nprim_i: int,
    nprim_j: int,
):
    scalar = _compiled_nuclear_pair_kernel(angular_i, angular_j, nprim_i, nprim_j)
    return jax.jit(jax.vmap(scalar, in_axes=(0, 0, 0, 0, 0, 0, None, None)))


@functools.lru_cache(maxsize=None)
def _compiled_hcore_pair_kernel(
    angular_i: tuple[int, int, int],
    angular_j: tuple[int, int, int],
    nprim_i: int,
    nprim_j: int,
):
    def kernel(
        exponents_i: Array,
        coefficients_i: Array,
        center_i: Array,
        exponents_j: Array,
        coefficients_j: Array,
        center_j: Array,
        atom_coords: Array,
        atom_charges: Array,
    ) -> Array:
        alpha_flat, beta_flat, weight_flat = _flatten_primitive_pairs_2c(
            exponents_i,
            coefficients_i,
            exponents_j,
            coefficients_j,
            angular_i=angular_i,
            angular_j=angular_j,
        )

        def primitive_value(alpha: Array, beta: Array) -> Array:
            def base_fn(a, b):
                return _primitive_kinetic_ss(alpha, beta, a, b) + _primitive_nuclear_ss(
                    alpha,
                    beta,
                    a,
                    b,
                    atom_coords,
                    atom_charges,
                )

            return apply_cartesian_derivatives_2c(
                base_fn,
                center_a=center_i,
                center_b=center_j,
                alpha=alpha,
                beta=beta,
                ang_a=angular_i,
                ang_b=angular_j,
            )

        prim = jax.vmap(primitive_value)(alpha_flat, beta_flat)
        return jnp.dot(weight_flat, prim, precision=jax.lax.Precision.HIGHEST)

    return jax.jit(kernel)


@functools.lru_cache(maxsize=None)
def _compiled_hcore_pair_kernel_batched(
    angular_i: tuple[int, int, int],
    angular_j: tuple[int, int, int],
    nprim_i: int,
    nprim_j: int,
):
    scalar = _compiled_hcore_pair_kernel(angular_i, angular_j, nprim_i, nprim_j)
    return jax.jit(jax.vmap(scalar, in_axes=(0, 0, 0, 0, 0, 0, None, None)))


@functools.lru_cache(maxsize=None)
def _compiled_overlap_hcore_pair_kernel(
    angular_i: tuple[int, int, int],
    angular_j: tuple[int, int, int],
    nprim_i: int,
    nprim_j: int,
):
    def kernel(
        exponents_i: Array,
        coefficients_i: Array,
        center_i: Array,
        exponents_j: Array,
        coefficients_j: Array,
        center_j: Array,
        atom_coords: Array,
        atom_charges: Array,
    ) -> Array:
        alpha_flat, beta_flat, weight_flat = _flatten_primitive_pairs_2c(
            exponents_i,
            coefficients_i,
            exponents_j,
            coefficients_j,
            angular_i=angular_i,
            angular_j=angular_j,
        )

        def primitive_value(alpha: Array, beta: Array) -> Array:
            overlap_base = lambda a, b: _primitive_overlap_ss(alpha, beta, a, b)

            def hcore_base(a, b):
                return _primitive_kinetic_ss(alpha, beta, a, b) + _primitive_nuclear_ss(
                    alpha,
                    beta,
                    a,
                    b,
                    atom_coords,
                    atom_charges,
                )

            overlap = apply_cartesian_derivatives_2c(
                overlap_base,
                center_a=center_i,
                center_b=center_j,
                alpha=alpha,
                beta=beta,
                ang_a=angular_i,
                ang_b=angular_j,
            )
            hcore = apply_cartesian_derivatives_2c(
                hcore_base,
                center_a=center_i,
                center_b=center_j,
                alpha=alpha,
                beta=beta,
                ang_a=angular_i,
                ang_b=angular_j,
            )
            return jnp.stack((overlap, hcore))

        prim = jax.vmap(primitive_value)(alpha_flat, beta_flat)
        return jnp.tensordot(weight_flat, prim, axes=(0, 0), precision=jax.lax.Precision.HIGHEST)

    return jax.jit(kernel)


@functools.lru_cache(maxsize=None)
def _compiled_overlap_hcore_pair_kernel_batched(
    angular_i: tuple[int, int, int],
    angular_j: tuple[int, int, int],
    nprim_i: int,
    nprim_j: int,
):
    scalar = _compiled_overlap_hcore_pair_kernel(angular_i, angular_j, nprim_i, nprim_j)
    return jax.jit(jax.vmap(scalar, in_axes=(0, 0, 0, 0, 0, 0, None, None)))


@functools.lru_cache(maxsize=None)
def _compiled_dipole_pair_kernel(
    angular_i: tuple[int, int, int],
    angular_j: tuple[int, int, int],
    nprim_i: int,
    nprim_j: int,
    axis: int,
):
    def kernel(
        exponents_i: Array,
        coefficients_i: Array,
        center_i: Array,
        exponents_j: Array,
        coefficients_j: Array,
        center_j: Array,
        origin: Array,
    ) -> Array:
        alpha_flat, beta_flat, weight_flat = _flatten_primitive_pairs_2c(
            exponents_i,
            coefficients_i,
            exponents_j,
            coefficients_j,
            angular_i=angular_i,
            angular_j=angular_j,
        )

        def primitive_value(alpha: Array, beta: Array) -> Array:
            base_fn = lambda a, b: _primitive_dipole_ss(
                alpha,
                beta,
                a,
                b,
                origin,
                axis,
            )
            return apply_cartesian_derivatives_2c(
                base_fn,
                center_a=center_i,
                center_b=center_j,
                alpha=alpha,
                beta=beta,
                ang_a=angular_i,
                ang_b=angular_j,
            )

        prim = jax.vmap(primitive_value)(alpha_flat, beta_flat)
        return jnp.dot(weight_flat, prim, precision=jax.lax.Precision.HIGHEST)

    return jax.jit(kernel)


@functools.lru_cache(maxsize=None)
def _compiled_dipole_pair_kernel_batched(
    angular_i: tuple[int, int, int],
    angular_j: tuple[int, int, int],
    nprim_i: int,
    nprim_j: int,
    axis: int,
):
    scalar = _compiled_dipole_pair_kernel(angular_i, angular_j, nprim_i, nprim_j, axis)
    return jax.jit(jax.vmap(scalar, in_axes=(0, 0, 0, 0, 0, 0, None)))


@functools.lru_cache(maxsize=None)
def _compiled_rinv_pair_kernel(
    angular_i: tuple[int, int, int],
    angular_j: tuple[int, int, int],
    nprim_i: int,
    nprim_j: int,
):
    def kernel(
        exponents_i: Array,
        coefficients_i: Array,
        center_i: Array,
        exponents_j: Array,
        coefficients_j: Array,
        center_j: Array,
        origin: Array,
        zeta: Array,
    ) -> Array:
        alpha_flat, beta_flat, weight_flat = _flatten_primitive_pairs_2c(
            exponents_i,
            coefficients_i,
            exponents_j,
            coefficients_j,
            angular_i=angular_i,
            angular_j=angular_j,
        )

        def primitive_value(alpha: Array, beta: Array) -> Array:
            base_fn = lambda a, b: _primitive_rinv_ss(
                alpha,
                beta,
                a,
                b,
                origin,
                zeta,
            )
            return apply_cartesian_derivatives_2c(
                base_fn,
                center_a=center_i,
                center_b=center_j,
                alpha=alpha,
                beta=beta,
                ang_a=angular_i,
                ang_b=angular_j,
            )

        prim = jax.vmap(primitive_value)(alpha_flat, beta_flat)
        return jnp.dot(weight_flat, prim, precision=jax.lax.Precision.HIGHEST)

    return jax.jit(kernel)


@functools.lru_cache(maxsize=None)
def _compiled_rinv_pair_kernel_batched(
    angular_i: tuple[int, int, int],
    angular_j: tuple[int, int, int],
    nprim_i: int,
    nprim_j: int,
):
    scalar = _compiled_rinv_pair_kernel(angular_i, angular_j, nprim_i, nprim_j)
    return jax.jit(jax.vmap(scalar, in_axes=(0, 0, 0, 0, 0, 0, None, None)))


def _use_jit_engine(
    engine: str,
    *,
    angulars: tuple[tuple[int, int, int], ...] | None = None,
    nprims: tuple[int, ...] | None = None,
) -> bool:
    mode = str(engine).lower()
    if mode == "jit":
        return True
    if mode == "legacy":
        return False
    if mode == "auto":
        if angulars is None:
            return True
        # Hermite-AD JIT compile cost grows steeply for d/f shells.
        # Keep auto mode GPU-friendly for common s/p workloads and stable on CPU.
        max_l = max(sum(ang) for ang in angulars)
        if max_l > 1:
            return False
        # Large primitive contractions can still explode compile-time memory.
        if nprims is not None and max(nprims) > 3:
            return False
        return True
    raise ValueError(f"Unsupported integral engine mode {engine!r}.")


def _assemble_symmetric_matrix(
    *,
    n: int,
    row_idx,
    col_idx,
    values,
) -> Array:
    vals = jnp.asarray(values)
    if vals.size == 0:
        return jnp.zeros((n, n))
    rows = jnp.asarray(row_idx, dtype=jnp.int32)
    cols = jnp.asarray(col_idx, dtype=jnp.int32)
    mat = jnp.zeros((n, n), dtype=vals.dtype)
    mat = mat.at[rows, cols].set(vals)
    mat = mat.at[cols, rows].set(vals)
    return mat


def _pair_signature_2c(ao_i: CartesianAO, ao_j: CartesianAO):
    return (
        ao_i.angular,
        ao_j.angular,
        int(ao_i.exponents.shape[0]),
        int(ao_j.exponents.shape[0]),
    )


def _group_lower_triangle_pairs(basis: CartesianBasis):
    if basis.pair_groups:
        return {
            group.signature: {
                "row": np.asarray(group.row_idx, dtype=np.int32).tolist(),
                "col": np.asarray(group.col_idx, dtype=np.int32).tolist(),
            }
            for group in basis.pair_groups
        }
    groups: dict[tuple, dict[str, list[int]]] = {}
    for i in range(basis.nao):
        ao_i = basis.aos[i]
        for j in range(i + 1):
            ao_j = basis.aos[j]
            sig = _pair_signature_2c(ao_i, ao_j)
            bucket = groups.setdefault(sig, {"row": [], "col": []})
            bucket["row"].append(i)
            bucket["col"].append(j)
    return groups


def _gather_pair_batch(
    basis: CartesianBasis,
    row_idx,
    col_idx,
    *,
    nprim_i: int,
    nprim_j: int,
):
    row_arr = jnp.asarray(row_idx, dtype=jnp.int32)
    col_arr = jnp.asarray(col_idx, dtype=jnp.int32)
    exponents = jnp.asarray(basis.ao_exponents_padded)
    coefficients = jnp.asarray(basis.ao_coefficients_padded)
    centers = jnp.asarray(basis.ao_centers)
    exp_i = exponents[row_arr, :nprim_i]
    coeff_i = coefficients[row_arr, :nprim_i]
    center_i = centers[row_arr]
    exp_j = exponents[col_arr, :nprim_j]
    coeff_j = coefficients[col_arr, :nprim_j]
    center_j = centers[col_arr]
    return exp_i, coeff_i, center_i, exp_j, coeff_j, center_j


def _run_pair_kernel_chunked(
    kernel,
    batch_inputs: tuple[Array, ...],
    *,
    chunk_size: int = PAIR_BATCH_CHUNK,
    tail_inputs: tuple[Array, ...] = (),
) -> Array:
    n_items = int(batch_inputs[0].shape[0])
    if n_items == 0:
        return jnp.zeros((0,))

    base_chunk = max(int(chunk_size), 1)
    target_size = min(base_chunk, n_items)

    def _pad_chunk_to_fixed_size(chunk: tuple[Array, ...], size: int) -> tuple[Array, ...]:
        cur = int(chunk[0].shape[0])
        if cur >= size:
            return chunk
        pad = size - cur
        return tuple(
            jnp.concatenate(
                (inp, jnp.repeat(inp[cur - 1 : cur], pad, axis=0)),
                axis=0,
            )
            for inp in chunk
        )

    outputs: list[Array] = []
    for start in range(0, n_items, base_chunk):
        end = min(start + base_chunk, n_items)
        chunk = tuple(inp[start:end] for inp in batch_inputs)
        valid = int(chunk[0].shape[0])
        fixed_chunk = _pad_chunk_to_fixed_size(chunk, target_size if start + base_chunk > n_items else base_chunk)
        out = kernel(*fixed_chunk, *tail_inputs)
        outputs.append(out[:valid])
    if len(outputs) == 1:
        return outputs[0]
    return jnp.concatenate(outputs, axis=0)


def overlap_matrix(
    basis: CartesianBasis,
    *,
    engine: str = "auto",
) -> Array:
    """Compute cartesian overlap matrix S in pure JAX."""

    n = basis.nao
    row_chunks: list[Array] = []
    col_chunks: list[Array] = []
    value_chunks: list[Array] = []
    groups = basis.pair_groups or tuple(
        PairBatchGroup(
            signature=signature,
            row_idx=jnp.asarray(bucket["row"], dtype=jnp.int32),
            col_idx=jnp.asarray(bucket["col"], dtype=jnp.int32),
        )
        for signature, bucket in _group_lower_triangle_pairs(basis).items()
    )
    for group in groups:
        signature = group.signature
        row_host = np.asarray(group.row_idx, dtype=np.int32)
        col_host = np.asarray(group.col_idx, dtype=np.int32)
        row_arr = jnp.asarray(row_host, dtype=jnp.int32)
        col_arr = jnp.asarray(col_host, dtype=jnp.int32)
        use_jit = _use_jit_engine(
            engine,
            angulars=(signature[0], signature[1]),
            nprims=(signature[2], signature[3]),
        )
        if use_jit:
            exp_i, coeff_i, center_i, exp_j, coeff_j, center_j = _gather_pair_batch(
                basis,
                row_arr,
                col_arr,
                nprim_i=signature[2],
                nprim_j=signature[3],
            )
            kernel = _compiled_overlap_pair_kernel_batched(*signature)
            vals = _run_pair_kernel_chunked(
                kernel,
                (exp_i, coeff_i, center_i, exp_j, coeff_j, center_j),
            )
        else:
            vals = jnp.asarray(
                [
                    overlap_element(basis, int(i), int(j), engine=engine)
                    for i, j in zip(np.asarray(row_arr), np.asarray(col_arr), strict=True)
                ]
            )
        row_chunks.append(row_arr)
        col_chunks.append(col_arr)
        value_chunks.append(vals)
    row_idx = jnp.concatenate(row_chunks, axis=0) if row_chunks else jnp.zeros((0,), dtype=jnp.int32)
    col_idx = jnp.concatenate(col_chunks, axis=0) if col_chunks else jnp.zeros((0,), dtype=jnp.int32)
    values = jnp.concatenate(value_chunks, axis=0) if value_chunks else jnp.zeros((0,))
    return _assemble_symmetric_matrix(
        n=n,
        row_idx=row_idx,
        col_idx=col_idx,
        values=values,
    )


def kinetic_matrix(
    basis: CartesianBasis,
    *,
    engine: str = "auto",
) -> Array:
    """Compute cartesian kinetic-energy matrix T in pure JAX."""

    n = basis.nao
    row_chunks: list[Array] = []
    col_chunks: list[Array] = []
    value_chunks: list[Array] = []
    groups = basis.pair_groups or tuple(
        PairBatchGroup(
            signature=signature,
            row_idx=jnp.asarray(bucket["row"], dtype=jnp.int32),
            col_idx=jnp.asarray(bucket["col"], dtype=jnp.int32),
        )
        for signature, bucket in _group_lower_triangle_pairs(basis).items()
    )
    for group in groups:
        signature = group.signature
        row_host = np.asarray(group.row_idx, dtype=np.int32)
        col_host = np.asarray(group.col_idx, dtype=np.int32)
        row_arr = jnp.asarray(row_host, dtype=jnp.int32)
        col_arr = jnp.asarray(col_host, dtype=jnp.int32)
        use_jit = _use_jit_engine(
            engine,
            angulars=(signature[0], signature[1]),
            nprims=(signature[2], signature[3]),
        )
        if use_jit:
            exp_i, coeff_i, center_i, exp_j, coeff_j, center_j = _gather_pair_batch(
                basis,
                row_arr,
                col_arr,
                nprim_i=signature[2],
                nprim_j=signature[3],
            )
            kernel = _compiled_kinetic_pair_kernel_batched(*signature)
            vals = _run_pair_kernel_chunked(
                kernel,
                (exp_i, coeff_i, center_i, exp_j, coeff_j, center_j),
            )
        else:
            vals = jnp.asarray(
                [
                    kinetic_element(basis, int(i), int(j), engine=engine)
                    for i, j in zip(np.asarray(row_arr), np.asarray(col_arr), strict=True)
                ]
            )
        row_chunks.append(row_arr)
        col_chunks.append(col_arr)
        value_chunks.append(vals)
    row_idx = jnp.concatenate(row_chunks, axis=0) if row_chunks else jnp.zeros((0,), dtype=jnp.int32)
    col_idx = jnp.concatenate(col_chunks, axis=0) if col_chunks else jnp.zeros((0,), dtype=jnp.int32)
    values = jnp.concatenate(value_chunks, axis=0) if value_chunks else jnp.zeros((0,))
    return _assemble_symmetric_matrix(
        n=n,
        row_idx=row_idx,
        col_idx=col_idx,
        values=values,
    )


def nuclear_attraction_matrix(
    basis: CartesianBasis,
    *,
    atom_coords: Array | None = None,
    atom_charges: Array | None = None,
    engine: str = "auto",
) -> Array:
    """Compute cartesian nuclear attraction matrix V_nuc in pure JAX."""

    coords = basis.atom_coords if atom_coords is None else atom_coords
    charges = basis.atom_charges if atom_charges is None else atom_charges
    if coords is None or charges is None:
        raise ValueError(
            "Nuclear coordinates/charges must be provided either in basis or function args."
        )
    coords_arr = jnp.asarray(coords)
    charges_arr = jnp.asarray(charges)

    n = basis.nao
    row_chunks: list[Array] = []
    col_chunks: list[Array] = []
    value_chunks: list[Array] = []
    groups = basis.pair_groups or tuple(
        PairBatchGroup(
            signature=signature,
            row_idx=jnp.asarray(bucket["row"], dtype=jnp.int32),
            col_idx=jnp.asarray(bucket["col"], dtype=jnp.int32),
        )
        for signature, bucket in _group_lower_triangle_pairs(basis).items()
    )
    for group in groups:
        signature = group.signature
        row_arr = jnp.asarray(group.row_idx, dtype=jnp.int32)
        col_arr = jnp.asarray(group.col_idx, dtype=jnp.int32)
        use_jit = _use_jit_engine(
            engine,
            angulars=(signature[0], signature[1]),
            nprims=(signature[2], signature[3]),
        )
        if use_jit:
            exp_i, coeff_i, center_i, exp_j, coeff_j, center_j = _gather_pair_batch(
                basis,
                row_arr,
                col_arr,
                nprim_i=signature[2],
                nprim_j=signature[3],
            )
            kernel = _compiled_nuclear_pair_kernel_batched(*signature)
            vals = _run_pair_kernel_chunked(
                kernel,
                (exp_i, coeff_i, center_i, exp_j, coeff_j, center_j),
                tail_inputs=(
                    coords_arr,
                    charges_arr,
                ),
            )
        else:
            vals = jnp.asarray(
                [
                    nuclear_attraction_element(
                        basis,
                        int(i),
                        int(j),
                        atom_coords=coords_arr,
                        atom_charges=charges_arr,
                        engine=engine,
                    )
                    for i, j in zip(np.asarray(row_arr), np.asarray(col_arr), strict=True)
                ]
            )
        row_chunks.append(row_arr)
        col_chunks.append(col_arr)
        value_chunks.append(vals)
    row_idx = jnp.concatenate(row_chunks, axis=0) if row_chunks else jnp.zeros((0,), dtype=jnp.int32)
    col_idx = jnp.concatenate(col_chunks, axis=0) if col_chunks else jnp.zeros((0,), dtype=jnp.int32)
    values = jnp.concatenate(value_chunks, axis=0) if value_chunks else jnp.zeros((0,))
    return _assemble_symmetric_matrix(
        n=n,
        row_idx=row_idx,
        col_idx=col_idx,
        values=values,
    )


def build_hcore(
    basis: CartesianBasis,
    *,
    atom_coords: Array | None = None,
    atom_charges: Array | None = None,
    engine: str = "auto",
) -> Array:
    """Build one-electron core Hamiltonian H_core = T + V_nuc."""

    coords = basis.atom_coords if atom_coords is None else atom_coords
    charges = basis.atom_charges if atom_charges is None else atom_charges
    if coords is None or charges is None:
        raise ValueError(
            "Nuclear coordinates/charges must be provided either in basis or function args."
        )
    coords_arr = jnp.asarray(coords)
    charges_arr = jnp.asarray(charges)

    n = basis.nao
    row_chunks: list[Array] = []
    col_chunks: list[Array] = []
    value_chunks: list[Array] = []
    groups = basis.pair_groups or tuple(
        PairBatchGroup(
            signature=signature,
            row_idx=jnp.asarray(bucket["row"], dtype=jnp.int32),
            col_idx=jnp.asarray(bucket["col"], dtype=jnp.int32),
        )
        for signature, bucket in _group_lower_triangle_pairs(basis).items()
    )
    for group in groups:
        signature = group.signature
        row_arr = jnp.asarray(group.row_idx, dtype=jnp.int32)
        col_arr = jnp.asarray(group.col_idx, dtype=jnp.int32)
        use_jit = _use_jit_engine(
            engine,
            angulars=(signature[0], signature[1]),
            nprims=(signature[2], signature[3]),
        )
        if use_jit:
            exp_i, coeff_i, center_i, exp_j, coeff_j, center_j = _gather_pair_batch(
                basis,
                row_arr,
                col_arr,
                nprim_i=signature[2],
                nprim_j=signature[3],
            )
            kernel = _compiled_hcore_pair_kernel_batched(*signature)
            vals = _run_pair_kernel_chunked(
                kernel,
                (exp_i, coeff_i, center_i, exp_j, coeff_j, center_j),
                tail_inputs=(
                    coords_arr,
                    charges_arr,
                ),
            )
        else:
            vals = jnp.asarray(
                [
                    kinetic_element(basis, int(i), int(j), engine=engine)
                    + nuclear_attraction_element(
                        basis,
                        int(i),
                        int(j),
                        atom_coords=coords_arr,
                        atom_charges=charges_arr,
                        engine=engine,
                    )
                    for i, j in zip(np.asarray(row_arr), np.asarray(col_arr), strict=True)
                ]
            )
        row_chunks.append(row_arr)
        col_chunks.append(col_arr)
        value_chunks.append(vals)
    row_idx = jnp.concatenate(row_chunks, axis=0) if row_chunks else jnp.zeros((0,), dtype=jnp.int32)
    col_idx = jnp.concatenate(col_chunks, axis=0) if col_chunks else jnp.zeros((0,), dtype=jnp.int32)
    values = jnp.concatenate(value_chunks, axis=0) if value_chunks else jnp.zeros((0,))
    return _assemble_symmetric_matrix(
        n=n,
        row_idx=row_idx,
        col_idx=col_idx,
        values=values,
    )


def overlap_hcore_matrices(
    basis: CartesianBasis,
    *,
    atom_coords: Array | None = None,
    atom_charges: Array | None = None,
    engine: str = "auto",
) -> tuple[Array, Array]:
    """Build overlap and H_core matrices in one AO-pair pass."""

    coords = basis.atom_coords if atom_coords is None else atom_coords
    charges = basis.atom_charges if atom_charges is None else atom_charges
    if coords is None or charges is None:
        raise ValueError(
            "Nuclear coordinates/charges must be provided either in basis or function args."
        )
    coords_arr = jnp.asarray(coords)
    charges_arr = jnp.asarray(charges)

    n = basis.nao
    row_chunks: list[Array] = []
    col_chunks: list[Array] = []
    overlap_chunks: list[Array] = []
    hcore_chunks: list[Array] = []
    groups = basis.pair_groups or tuple(
        PairBatchGroup(
            signature=signature,
            row_idx=jnp.asarray(bucket["row"], dtype=jnp.int32),
            col_idx=jnp.asarray(bucket["col"], dtype=jnp.int32),
        )
        for signature, bucket in _group_lower_triangle_pairs(basis).items()
    )
    for group in groups:
        signature = group.signature
        row_arr = jnp.asarray(group.row_idx, dtype=jnp.int32)
        col_arr = jnp.asarray(group.col_idx, dtype=jnp.int32)
        use_jit = _use_jit_engine(
            engine,
            angulars=(signature[0], signature[1]),
            nprims=(signature[2], signature[3]),
        )
        if use_jit:
            exp_i, coeff_i, center_i, exp_j, coeff_j, center_j = _gather_pair_batch(
                basis,
                row_arr,
                col_arr,
                nprim_i=signature[2],
                nprim_j=signature[3],
            )
            kernel = _compiled_overlap_hcore_pair_kernel_batched(*signature)
            vals = _run_pair_kernel_chunked(
                kernel,
                (exp_i, coeff_i, center_i, exp_j, coeff_j, center_j),
                tail_inputs=(
                    coords_arr,
                    charges_arr,
                ),
            )
            overlap_vals = vals[:, 0]
            hcore_vals = vals[:, 1]
        else:
            overlap_vals = jnp.asarray(
                [
                    overlap_element(basis, int(i), int(j), engine=engine)
                    for i, j in zip(np.asarray(row_arr), np.asarray(col_arr), strict=True)
                ]
            )
            hcore_vals = jnp.asarray(
                [
                    kinetic_element(basis, int(i), int(j), engine=engine)
                    + nuclear_attraction_element(
                        basis,
                        int(i),
                        int(j),
                        atom_coords=coords_arr,
                        atom_charges=charges_arr,
                        engine=engine,
                    )
                    for i, j in zip(np.asarray(row_arr), np.asarray(col_arr), strict=True)
                ]
            )
        row_chunks.append(row_arr)
        col_chunks.append(col_arr)
        overlap_chunks.append(overlap_vals)
        hcore_chunks.append(hcore_vals)
    row_idx = jnp.concatenate(row_chunks, axis=0) if row_chunks else jnp.zeros((0,), dtype=jnp.int32)
    col_idx = jnp.concatenate(col_chunks, axis=0) if col_chunks else jnp.zeros((0,), dtype=jnp.int32)
    overlap_values = jnp.concatenate(overlap_chunks, axis=0) if overlap_chunks else jnp.zeros((0,))
    hcore_values = jnp.concatenate(hcore_chunks, axis=0) if hcore_chunks else jnp.zeros((0,))
    overlap = _assemble_symmetric_matrix(
        n=n,
        row_idx=row_idx,
        col_idx=col_idx,
        values=overlap_values,
    )
    hcore = _assemble_symmetric_matrix(
        n=n,
        row_idx=row_idx,
        col_idx=col_idx,
        values=hcore_values,
    )
    return overlap, hcore


def dipole_matrix(
    basis: CartesianBasis,
    *,
    origin: Array | None = None,
    engine: str = "auto",
) -> Array:
    """Compute length-gauge AO dipole integrals relative to the given origin."""

    if origin is None:
        if basis.atom_coords is None or basis.atom_charges is None:
            raise ValueError(
                "Dipole origin requires basis.atom_coords and basis.atom_charges when origin is omitted."
            )
        charges = jnp.asarray(basis.atom_charges, dtype=jnp.float64)
        coords = jnp.asarray(basis.atom_coords, dtype=jnp.float64)
        origin_arr = jnp.einsum("a,ar->r", charges, coords) / jnp.sum(charges)
    else:
        origin_arr = jnp.asarray(origin)

    n = basis.nao
    groups = basis.pair_groups or tuple(
        PairBatchGroup(
            signature=signature,
            row_idx=jnp.asarray(bucket["row"], dtype=jnp.int32),
            col_idx=jnp.asarray(bucket["col"], dtype=jnp.int32),
        )
        for signature, bucket in _group_lower_triangle_pairs(basis).items()
    )
    components: list[Array] = []
    for axis in range(3):
        row_chunks: list[Array] = []
        col_chunks: list[Array] = []
        value_chunks: list[Array] = []
        for group in groups:
            signature = group.signature
            row_arr = jnp.asarray(group.row_idx, dtype=jnp.int32)
            col_arr = jnp.asarray(group.col_idx, dtype=jnp.int32)
            use_jit = _use_jit_engine(
                engine,
                angulars=(signature[0], signature[1]),
                nprims=(signature[2], signature[3]),
            )
            if use_jit:
                exp_i, coeff_i, center_i, exp_j, coeff_j, center_j = _gather_pair_batch(
                    basis,
                    row_arr,
                    col_arr,
                    nprim_i=signature[2],
                    nprim_j=signature[3],
                )
                kernel = _compiled_dipole_pair_kernel_batched(*signature, axis)
                vals = _run_pair_kernel_chunked(
                    kernel,
                    (exp_i, coeff_i, center_i, exp_j, coeff_j, center_j),
                    tail_inputs=(origin_arr,),
                )
            else:
                vals = jnp.asarray(
                    [
                        dipole_element(
                            basis,
                            int(i),
                            int(j),
                            origin=origin_arr,
                            axis=axis,
                            engine=engine,
                        )
                        for i, j in zip(np.asarray(row_arr), np.asarray(col_arr), strict=True)
                    ]
                )
            row_chunks.append(row_arr)
            col_chunks.append(col_arr)
            value_chunks.append(vals)
        row_idx = jnp.concatenate(row_chunks, axis=0) if row_chunks else jnp.zeros((0,), dtype=jnp.int32)
        col_idx = jnp.concatenate(col_chunks, axis=0) if col_chunks else jnp.zeros((0,), dtype=jnp.int32)
        values = jnp.concatenate(value_chunks, axis=0) if value_chunks else jnp.zeros((0,))
        components.append(
            _assemble_symmetric_matrix(
                n=n,
                row_idx=row_idx,
                col_idx=col_idx,
                values=values,
            )
        )
    return jnp.stack(components, axis=0)


def rinv_matrix(
    basis: CartesianBasis,
    *,
    origin: Array,
    zeta: float | Array | None = None,
    engine: str = "auto",
) -> Array:
    """Compute AO matrix <i|v(r-origin)|j> where v is 1/r or erf(sqrt(zeta) r)/r."""

    origin_arr = jnp.asarray(origin)
    zeta_arr = jnp.asarray(jnp.inf if zeta is None else zeta, dtype=origin_arr.dtype)

    n = basis.nao
    row_chunks: list[Array] = []
    col_chunks: list[Array] = []
    value_chunks: list[Array] = []
    groups = basis.pair_groups or tuple(
        PairBatchGroup(
            signature=signature,
            row_idx=jnp.asarray(bucket["row"], dtype=jnp.int32),
            col_idx=jnp.asarray(bucket["col"], dtype=jnp.int32),
        )
        for signature, bucket in _group_lower_triangle_pairs(basis).items()
    )
    for group in groups:
        signature = group.signature
        row_host = np.asarray(group.row_idx, dtype=np.int32)
        col_host = np.asarray(group.col_idx, dtype=np.int32)
        row_arr = jnp.asarray(row_host, dtype=jnp.int32)
        col_arr = jnp.asarray(col_host, dtype=jnp.int32)
        use_jit = _use_jit_engine(
            engine,
            angulars=(signature[0], signature[1]),
            nprims=(signature[2], signature[3]),
        )
        if use_jit:
            exp_i, coeff_i, center_i, exp_j, coeff_j, center_j = _gather_pair_batch(
                basis,
                row_arr,
                col_arr,
                nprim_i=signature[2],
                nprim_j=signature[3],
            )
            kernel = _compiled_rinv_pair_kernel_batched(*signature)
            vals = _run_pair_kernel_chunked(
                kernel,
                (exp_i, coeff_i, center_i, exp_j, coeff_j, center_j),
                tail_inputs=(origin_arr, zeta_arr),
            )
        else:
            vals = jnp.asarray(
                [
                    rinv_element(
                        basis,
                        int(i),
                        int(j),
                        origin=origin_arr,
                        zeta=zeta_arr,
                        engine=engine,
                    )
                    for i, j in zip(row_host, col_host, strict=True)
                ]
            )
        row_chunks.append(row_arr)
        col_chunks.append(col_arr)
        value_chunks.append(vals)
    row_idx = jnp.concatenate(row_chunks, axis=0) if row_chunks else jnp.zeros((0,), dtype=jnp.int32)
    col_idx = jnp.concatenate(col_chunks, axis=0) if col_chunks else jnp.zeros((0,), dtype=jnp.int32)
    values = jnp.concatenate(value_chunks, axis=0) if value_chunks else jnp.zeros((0,))
    return _assemble_symmetric_matrix(
        n=n,
        row_idx=row_idx,
        col_idx=col_idx,
        values=values,
    )


def _basis_object_cache(
    basis: CartesianBasis,
) -> dict[str, Callable[..., Array]]:
    key = id(basis)
    entry = _RINV_CHUNK_BUILDERS.get(key)
    if entry is not None:
        basis_ref, bucket = entry
        if basis_ref() is basis:
            return bucket
        if basis_ref() is None:
            _RINV_CHUNK_BUILDERS.pop(key, None)

    bucket: dict[str, Callable[..., Array]] = {}

    def _cleanup(_ref, *, _key=key) -> None:
        current = _RINV_CHUNK_BUILDERS.get(_key)
        if current is not None and current[0]() is None:
            _RINV_CHUNK_BUILDERS.pop(_key, None)

    basis_ref = weakref.ref(basis, _cleanup)
    _RINV_CHUNK_BUILDERS[key] = (basis_ref, bucket)
    return bucket


def _cached_rinv_chunk_builder(
    basis: CartesianBasis,
    *,
    engine: str,
):
    cache = _basis_object_cache(basis)
    engine_key = str(engine)
    builder = cache.get(engine_key)
    if builder is not None:
        return builder

    def _one_origin(one_origin: Array, zeta: Array) -> Array:
        return rinv_matrix(
            basis,
            origin=one_origin,
            zeta=zeta,
            engine=engine_key,
        )

    builder = jax.jit(jax.vmap(_one_origin, in_axes=(0, None)))
    cache[engine_key] = builder
    return builder


def rinv_matrices(
    basis: CartesianBasis,
    origins: Array,
    *,
    zeta: float | Array | None = None,
    engine: str = "auto",
    grid_chunk_size: int = 64,
) -> Array:
    """Batch AO matrices for pointwise/grid Coulomb kernels using vmap+jit."""

    origins_arr = jnp.asarray(origins)
    if origins_arr.ndim != 2 or origins_arr.shape[1] != 3:
        raise ValueError(
            f"origins must have shape (ngrids, 3), got {origins_arr.shape}."
        )
    zeta_arr = jnp.asarray(jnp.inf if zeta is None else zeta, dtype=origins_arr.dtype)
    chunk_builder = _cached_rinv_chunk_builder(basis, engine=engine)
    if int(grid_chunk_size) <= 0 or origins_arr.shape[0] <= int(grid_chunk_size):
        return chunk_builder(origins_arr, zeta_arr)

    outputs: list[Array] = []
    for start in range(0, int(origins_arr.shape[0]), int(grid_chunk_size)):
        end = min(start + int(grid_chunk_size), int(origins_arr.shape[0]))
        outputs.append(chunk_builder(origins_arr[start:end], zeta_arr))
    return jnp.concatenate(outputs, axis=0)


def overlap_element(
    basis: CartesianBasis,
    i: int,
    j: int,
    *,
    engine: str = "auto",
) -> Array:
    """Single overlap element S_ij in cartesian AO basis."""
    ao_i = basis.aos[i]
    ao_j = basis.aos[j]
    if not _use_jit_engine(
        engine,
        angulars=(ao_i.angular, ao_j.angular),
        nprims=(int(ao_i.exponents.shape[0]), int(ao_j.exponents.shape[0])),
    ):
        return _contracted_pair_integral_2c(ao_i, ao_j, _primitive_overlap_ss)
    kernel = _compiled_overlap_pair_kernel(
        ao_i.angular,
        ao_j.angular,
        int(ao_i.exponents.shape[0]),
        int(ao_j.exponents.shape[0]),
    )
    return kernel(
        ao_i.exponents,
        ao_i.coefficients,
        ao_i.center,
        ao_j.exponents,
        ao_j.coefficients,
        ao_j.center,
    )


def kinetic_element(
    basis: CartesianBasis,
    i: int,
    j: int,
    *,
    engine: str = "auto",
) -> Array:
    """Single kinetic element T_ij in cartesian AO basis."""
    ao_i = basis.aos[i]
    ao_j = basis.aos[j]
    if not _use_jit_engine(
        engine,
        angulars=(ao_i.angular, ao_j.angular),
        nprims=(int(ao_i.exponents.shape[0]), int(ao_j.exponents.shape[0])),
    ):
        return _contracted_pair_integral_2c(ao_i, ao_j, _primitive_kinetic_ss)
    kernel = _compiled_kinetic_pair_kernel(
        ao_i.angular,
        ao_j.angular,
        int(ao_i.exponents.shape[0]),
        int(ao_j.exponents.shape[0]),
    )
    return kernel(
        ao_i.exponents,
        ao_i.coefficients,
        ao_i.center,
        ao_j.exponents,
        ao_j.coefficients,
        ao_j.center,
    )


def nuclear_attraction_element(
    basis: CartesianBasis,
    i: int,
    j: int,
    *,
    atom_coords: Array | None = None,
    atom_charges: Array | None = None,
    engine: str = "auto",
) -> Array:
    """Single nuclear-attraction element V_ij in cartesian AO basis."""

    coords = basis.atom_coords if atom_coords is None else atom_coords
    charges = basis.atom_charges if atom_charges is None else atom_charges
    if coords is None or charges is None:
        raise ValueError(
            "Nuclear coordinates/charges must be provided either in basis or function args."
        )
    ao_i = basis.aos[i]
    ao_j = basis.aos[j]
    coords = jnp.asarray(coords)
    charges = jnp.asarray(charges)
    if not _use_jit_engine(
        engine,
        angulars=(ao_i.angular, ao_j.angular),
        nprims=(int(ao_i.exponents.shape[0]), int(ao_j.exponents.shape[0])),
    ):
        return _contracted_pair_nuclear_integral(
            ao_i,
            ao_j,
            atom_coords=coords,
            atom_charges=charges,
        )
    kernel = _compiled_nuclear_pair_kernel(
        ao_i.angular,
        ao_j.angular,
        int(ao_i.exponents.shape[0]),
        int(ao_j.exponents.shape[0]),
    )
    return kernel(
        ao_i.exponents,
        ao_i.coefficients,
        ao_i.center,
        ao_j.exponents,
        ao_j.coefficients,
        ao_j.center,
        coords,
        charges,
    )


def dipole_element(
    basis: CartesianBasis,
    i: int,
    j: int,
    *,
    origin: Array,
    axis: int,
    engine: str = "auto",
) -> Array:
    """Single AO dipole element <i|r_axis-origin_axis|j> in cartesian basis."""

    ao_i = basis.aos[i]
    ao_j = basis.aos[j]
    origin_arr = jnp.asarray(origin)
    if not _use_jit_engine(
        engine,
        angulars=(ao_i.angular, ao_j.angular),
        nprims=(int(ao_i.exponents.shape[0]), int(ao_j.exponents.shape[0])),
    ):
        return _contracted_pair_dipole_integral(
            ao_i,
            ao_j,
            origin=origin_arr,
            axis=axis,
        )
    kernel = _compiled_dipole_pair_kernel(
        ao_i.angular,
        ao_j.angular,
        int(ao_i.exponents.shape[0]),
        int(ao_j.exponents.shape[0]),
        axis,
    )
    return kernel(
        ao_i.exponents,
        ao_i.coefficients,
        ao_i.center,
        ao_j.exponents,
        ao_j.coefficients,
        ao_j.center,
        origin_arr,
    )


def rinv_element(
    basis: CartesianBasis,
    i: int,
    j: int,
    *,
    origin: Array,
    zeta: float | Array | None = None,
    engine: str = "auto",
) -> Array:
    """Single AO matrix element for 1/r or erf(sqrt(zeta) r)/r centered at origin."""

    ao_i = basis.aos[i]
    ao_j = basis.aos[j]
    origin_arr = jnp.asarray(origin)
    zeta_arr = jnp.asarray(jnp.inf if zeta is None else zeta, dtype=origin_arr.dtype)
    if not _use_jit_engine(
        engine,
        angulars=(ao_i.angular, ao_j.angular),
        nprims=(int(ao_i.exponents.shape[0]), int(ao_j.exponents.shape[0])),
    ):
        return _contracted_pair_rinv_integral(
            ao_i,
            ao_j,
            origin=origin_arr,
            zeta=zeta_arr,
        )
    kernel = _compiled_rinv_pair_kernel(
        ao_i.angular,
        ao_j.angular,
        int(ao_i.exponents.shape[0]),
        int(ao_j.exponents.shape[0]),
    )
    return kernel(
        ao_i.exponents,
        ao_i.coefficients,
        ao_i.center,
        ao_j.exponents,
        ao_j.coefficients,
        ao_j.center,
        origin_arr,
        zeta_arr,
    )
