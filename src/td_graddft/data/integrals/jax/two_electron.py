from __future__ import annotations

import functools
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...basis import CartesianAO, CartesianBasis, ContractedShell
from ._common import (
    SUPPORTED_CARTESIAN_MAX_L,
    apply_cartesian_derivatives_4c,
    boys0,
    primitive_cartesian_norm,
    validate_cartesian_angular,
)

QUARTET_BATCH_CHUNK = 512


def _primitive_eri_ssss(
    alpha: Array,
    beta: Array,
    gamma: Array,
    delta: Array,
    center_a: Array,
    center_b: Array,
    center_c: Array,
    center_d: Array,
) -> Array:
    p = alpha + beta
    q = gamma + delta
    mu = alpha * beta / p
    nu = gamma * delta / q

    rab2 = jnp.dot(center_a - center_b, center_a - center_b)
    rcd2 = jnp.dot(center_c - center_d, center_c - center_d)
    center_p = (alpha * center_a + beta * center_b) / p
    center_q = (gamma * center_c + delta * center_d) / q
    rpq2 = jnp.dot(center_p - center_q, center_p - center_q)

    pref = 2.0 * jnp.pi**2.5 / (p * q * jnp.sqrt(p + q))
    t = (p * q / (p + q)) * rpq2
    return pref * jnp.exp(-mu * rab2 - nu * rcd2) * boys0(t)


def _boys_values(max_n: int, t: Array) -> Array:
    t = jnp.asarray(t)
    tiny = 1e-8
    safe_t = jnp.maximum(t, tiny)
    exp_neg_t = jnp.exp(-t)
    regular = [boys0(t)]
    for n in range(1, max_n + 1):
        regular.append(((2 * n - 1) * regular[-1] - exp_neg_t) / (2.0 * safe_t))

    small = []
    for n in range(max_n + 1):
        series = jnp.zeros_like(t)
        term = jnp.ones_like(t)
        factorial = 1.0
        for k in range(0, 16):
            if k > 0:
                factorial *= k
                term = -term * t
            series = series + term / (factorial * (2 * n + 2 * k + 1))
        small.append(series)
    return jnp.stack(
        [jnp.where(t < tiny, small[n], regular[n]) for n in range(max_n + 1)],
        axis=0,
    )


def _tuple_dec(angular: tuple[int, int, int], axis: int) -> tuple[int, int, int]:
    values = list(angular)
    values[axis] -= 1
    return tuple(values)


def _tuple_inc(angular: tuple[int, int, int], axis: int) -> tuple[int, int, int]:
    values = list(angular)
    values[axis] += 1
    return tuple(values)


def _primitive_eri_recurrence(
    alpha: Array,
    beta: Array,
    gamma: Array,
    delta: Array,
    center_a: Array,
    center_b: Array,
    center_c: Array,
    center_d: Array,
    *,
    ang_a: tuple[int, int, int],
    ang_b: tuple[int, int, int],
    ang_c: tuple[int, int, int],
    ang_d: tuple[int, int, int],
) -> Array:
    """Primitive Cartesian ERI via Obara-Saika HRR+VRR recurrence."""

    p = alpha + beta
    q = gamma + delta
    mu = alpha * beta / p
    nu = gamma * delta / q
    rab2 = jnp.dot(center_a - center_b, center_a - center_b)
    rcd2 = jnp.dot(center_c - center_d, center_c - center_d)
    center_p = (alpha * center_a + beta * center_b) / p
    center_q = (gamma * center_c + delta * center_d) / q
    center_w = (p * center_p + q * center_q) / (p + q)
    rpq2 = jnp.dot(center_p - center_q, center_p - center_q)
    pref = 2.0 * jnp.pi**2.5 / (p * q * jnp.sqrt(p + q))
    pref = pref * jnp.exp(-mu * rab2 - nu * rcd2)
    total_l = sum(ang_a) + sum(ang_b) + sum(ang_c) + sum(ang_d)
    boys = pref * _boys_values(total_l, (p * q / (p + q)) * rpq2)
    bra_ratio = q / (p + q)
    ket_ratio = p / (p + q)
    zero = jnp.zeros_like(boys[0])

    @functools.lru_cache(maxsize=None)
    def compute_vrr(
        a: tuple[int, int, int],
        c: tuple[int, int, int],
        m: int,
    ) -> Array:
        if min(a + c) < 0:
            return zero
        if sum(a) + sum(c) == 0:
            return boys[m]

        if sum(a) > 0:
            axis = next(idx for idx, power in enumerate(a) if power > 0)
            a1 = _tuple_dec(a, axis)
            out = (center_p[axis] - center_a[axis]) * compute_vrr(a1, c, m)
            out = out + (center_w[axis] - center_p[axis]) * compute_vrr(a1, c, m + 1)
            if a1[axis] > 0:
                a2 = _tuple_dec(a1, axis)
                coef = a1[axis] / (2.0 * p)
                out = out + coef * (
                    compute_vrr(a2, c, m) - bra_ratio * compute_vrr(a2, c, m + 1)
                )
            if c[axis] > 0:
                c1 = _tuple_dec(c, axis)
                coef = c[axis] / (2.0 * (p + q))
                out = out + coef * compute_vrr(a1, c1, m + 1)
            return out

        axis = next(idx for idx, power in enumerate(c) if power > 0)
        c1 = _tuple_dec(c, axis)
        out = (center_q[axis] - center_c[axis]) * compute_vrr(a, c1, m)
        out = out + (center_w[axis] - center_q[axis]) * compute_vrr(a, c1, m + 1)
        if c1[axis] > 0:
            c2 = _tuple_dec(c1, axis)
            coef = c1[axis] / (2.0 * q)
            out = out + coef * (
                compute_vrr(a, c2, m) - ket_ratio * compute_vrr(a, c2, m + 1)
            )
        if a[axis] > 0:
            a1 = _tuple_dec(a, axis)
            coef = a[axis] / (2.0 * (p + q))
            out = out + coef * compute_vrr(a1, c1, m + 1)
        return out

    @functools.lru_cache(maxsize=None)
    def compute_hrr(
        a: tuple[int, int, int],
        b: tuple[int, int, int],
        c: tuple[int, int, int],
        d: tuple[int, int, int],
    ) -> Array:
        if min(a + b + c + d) < 0:
            return zero
        if sum(b) > 0:
            axis = next(idx for idx, power in enumerate(b) if power > 0)
            b1 = _tuple_dec(b, axis)
            a1 = _tuple_inc(a, axis)
            return compute_hrr(a1, b1, c, d) + (center_a[axis] - center_b[axis]) * compute_hrr(
                a, b1, c, d
            )
        if sum(d) > 0:
            axis = next(idx for idx, power in enumerate(d) if power > 0)
            d1 = _tuple_dec(d, axis)
            c1 = _tuple_inc(c, axis)
            return compute_hrr(a, b, c1, d1) + (center_c[axis] - center_d[axis]) * compute_hrr(
                a, b, c, d1
            )
        return compute_vrr(a, c, 0)

    return compute_hrr(ang_a, ang_b, ang_c, ang_d)


def _flatten_primitive_quartets_4c(
    exponents_i: Array,
    coefficients_i: Array,
    exponents_j: Array,
    coefficients_j: Array,
    exponents_k: Array,
    coefficients_k: Array,
    exponents_l: Array,
    coefficients_l: Array,
    *,
    angular_i: tuple[int, int, int],
    angular_j: tuple[int, int, int],
    angular_k: tuple[int, int, int],
    angular_l: tuple[int, int, int],
) -> tuple[Array, Array, Array, Array, Array]:
    norm_i = primitive_cartesian_norm(exponents_i, angular_i)
    norm_j = primitive_cartesian_norm(exponents_j, angular_j)
    norm_k = primitive_cartesian_norm(exponents_k, angular_k)
    norm_l = primitive_cartesian_norm(exponents_l, angular_l)
    weighted_i = coefficients_i * norm_i
    weighted_j = coefficients_j * norm_j
    weighted_k = coefficients_k * norm_k
    weighted_l = coefficients_l * norm_l
    alpha, beta, gamma, delta = jnp.meshgrid(
        exponents_i,
        exponents_j,
        exponents_k,
        exponents_l,
        indexing="ij",
    )
    weights = (
        weighted_i[:, None, None, None]
        * weighted_j[None, :, None, None]
        * weighted_k[None, None, :, None]
        * weighted_l[None, None, None, :]
    )
    return (
        alpha.reshape(-1),
        beta.reshape(-1),
        gamma.reshape(-1),
        delta.reshape(-1),
        weights.reshape(-1),
    )


@functools.lru_cache(maxsize=None)
def _compiled_shell_block_scatter_kernel(
    block_size: int,
):
    def kernel(
        target: Array,
        scatter_i: Array,
        scatter_j: Array,
        scatter_k: Array,
        scatter_l: Array,
        blocks: Array,
    ) -> Array:
        vv = jnp.concatenate(
            (
                blocks.reshape(-1),
                blocks.transpose(0, 2, 1, 3, 4).reshape(-1),
                blocks.transpose(0, 1, 2, 4, 3).reshape(-1),
                blocks.transpose(0, 2, 1, 4, 3).reshape(-1),
                blocks.transpose(0, 3, 4, 1, 2).reshape(-1),
                blocks.transpose(0, 4, 3, 1, 2).reshape(-1),
                blocks.transpose(0, 3, 4, 2, 1).reshape(-1),
                blocks.transpose(0, 4, 3, 2, 1).reshape(-1),
            ),
            axis=0,
        )
        return target.at[
            scatter_i[: 8 * block_size],
            scatter_j[: 8 * block_size],
            scatter_k[: 8 * block_size],
            scatter_l[: 8 * block_size],
        ].set(vv)

    return jax.jit(kernel)


def _angular_axis(angular: tuple[int, int, int]) -> int:
    for axis, power in enumerate(angular):
        if power:
            return axis
    return -1


def _contracted_eri(
    ao_i: CartesianAO,
    ao_j: CartesianAO,
    ao_k: CartesianAO,
    ao_l: CartesianAO,
) -> Array:
    validate_cartesian_angular(ao_i.angular, max_l=SUPPORTED_CARTESIAN_MAX_L)
    validate_cartesian_angular(ao_j.angular, max_l=SUPPORTED_CARTESIAN_MAX_L)
    validate_cartesian_angular(ao_k.angular, max_l=SUPPORTED_CARTESIAN_MAX_L)
    validate_cartesian_angular(ao_l.angular, max_l=SUPPORTED_CARTESIAN_MAX_L)

    value = jnp.asarray(0.0)
    for ip in range(int(ao_i.exponents.shape[0])):
        alpha = ao_i.exponents[ip]
        ci = ao_i.coefficients[ip]
        ni = primitive_cartesian_norm(alpha, ao_i.angular)
        for jp in range(int(ao_j.exponents.shape[0])):
            beta = ao_j.exponents[jp]
            cj = ao_j.coefficients[jp]
            nj = primitive_cartesian_norm(beta, ao_j.angular)
            for kp in range(int(ao_k.exponents.shape[0])):
                gamma = ao_k.exponents[kp]
                ck = ao_k.coefficients[kp]
                nk = primitive_cartesian_norm(gamma, ao_k.angular)
                for lp in range(int(ao_l.exponents.shape[0])):
                    delta = ao_l.exponents[lp]
                    cl = ao_l.coefficients[lp]
                    nl = primitive_cartesian_norm(delta, ao_l.angular)

                    base_fn = (
                        lambda a, b, c, d, _alpha=alpha, _beta=beta, _gamma=gamma, _delta=delta: _primitive_eri_ssss(
                            _alpha,
                            _beta,
                            _gamma,
                            _delta,
                            a,
                            b,
                            c,
                            d,
                        )
                    )
                    prim = apply_cartesian_derivatives_4c(
                        base_fn,
                        center_a=ao_i.center,
                        center_b=ao_j.center,
                        center_c=ao_k.center,
                        center_d=ao_l.center,
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma,
                        delta=delta,
                        ang_a=ao_i.angular,
                        ang_b=ao_j.angular,
                        ang_c=ao_k.angular,
                        ang_d=ao_l.angular,
                    )
                    value = value + ci * cj * ck * cl * ni * nj * nk * nl * prim
    return value


@functools.lru_cache(maxsize=None)
def _compiled_eri_pair_kernel(
    angular_i: tuple[int, int, int],
    angular_j: tuple[int, int, int],
    angular_k: tuple[int, int, int],
    angular_l: tuple[int, int, int],
    nprim_i: int,
    nprim_j: int,
    nprim_k: int,
    nprim_l: int,
):
    max_l = max(
        sum(angular_i),
        sum(angular_j),
        sum(angular_k),
        sum(angular_l),
    )
    total_l = sum(angular_i) + sum(angular_j) + sum(angular_k) + sum(angular_l)
    use_recurrence_kernel = max_l <= 1 and total_l > 0

    def kernel(
        exponents_i: Array,
        coefficients_i: Array,
        center_i: Array,
        exponents_j: Array,
        coefficients_j: Array,
        center_j: Array,
        exponents_k: Array,
        coefficients_k: Array,
        center_k: Array,
        exponents_l: Array,
        coefficients_l: Array,
        center_l: Array,
    ) -> Array:
        alpha_flat, beta_flat, gamma_flat, delta_flat, weight_flat = _flatten_primitive_quartets_4c(
            exponents_i,
            coefficients_i,
            exponents_j,
            coefficients_j,
            exponents_k,
            coefficients_k,
            exponents_l,
            coefficients_l,
            angular_i=angular_i,
            angular_j=angular_j,
            angular_k=angular_k,
            angular_l=angular_l,
        )

        def primitive_value(alpha: Array, beta: Array, gamma: Array, delta: Array) -> Array:
            if use_recurrence_kernel:
                return _primitive_eri_recurrence(
                    alpha,
                    beta,
                    gamma,
                    delta,
                    center_i,
                    center_j,
                    center_k,
                    center_l,
                    ang_a=angular_i,
                    ang_b=angular_j,
                    ang_c=angular_k,
                    ang_d=angular_l,
                )
            base_fn = lambda a, b, c, d: _primitive_eri_ssss(
                alpha,
                beta,
                gamma,
                delta,
                a,
                b,
                c,
                d,
            )
            return apply_cartesian_derivatives_4c(
                base_fn,
                center_a=center_i,
                center_b=center_j,
                center_c=center_k,
                center_d=center_l,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                delta=delta,
                ang_a=angular_i,
                ang_b=angular_j,
                ang_c=angular_k,
                ang_d=angular_l,
            )

        prim = jax.vmap(primitive_value)(alpha_flat, beta_flat, gamma_flat, delta_flat)
        return jnp.dot(weight_flat, prim, precision=jax.lax.Precision.HIGHEST)

    return jax.jit(kernel)


@functools.lru_cache(maxsize=None)
def _compiled_eri_pair_kernel_batched(
    angular_i: tuple[int, int, int],
    angular_j: tuple[int, int, int],
    angular_k: tuple[int, int, int],
    angular_l: tuple[int, int, int],
    nprim_i: int,
    nprim_j: int,
    nprim_k: int,
    nprim_l: int,
):
    scalar = _compiled_eri_pair_kernel(
        angular_i,
        angular_j,
        angular_k,
        angular_l,
        nprim_i,
        nprim_j,
        nprim_k,
        nprim_l,
    )
    return jax.jit(
        jax.vmap(
            scalar,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        )
    )


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


def _quartet_signature(ao_i: CartesianAO, ao_j: CartesianAO, ao_k: CartesianAO, ao_l: CartesianAO):
    return (
        ao_i.angular,
        ao_j.angular,
        ao_k.angular,
        ao_l.angular,
        int(ao_i.exponents.shape[0]),
        int(ao_j.exponents.shape[0]),
        int(ao_k.exponents.shape[0]),
        int(ao_l.exponents.shape[0]),
    )


@functools.lru_cache(maxsize=None)
def _lower_triangle_pairs(size: int) -> tuple[tuple[int, int], ...]:
    return tuple((i, j) for i in range(size) for j in range(i + 1))


def _lower_triangle_pairs_to_matrix(size: int, *, sentinel: int | None = None) -> np.ndarray:
    fill = -1 if sentinel is None else int(sentinel)
    pair_index = np.full((size, size), fill, dtype=np.int32)
    for pos, (i, j) in enumerate(_lower_triangle_pairs(size)):
        pair_index[i, j] = pos
        pair_index[j, i] = pos
    return pair_index


def _shell_quartet_signature(
    shell_i: ContractedShell,
    shell_j: ContractedShell,
    shell_k: ContractedShell,
    shell_l: ContractedShell,
):
    return (
        shell_i.angulars,
        shell_j.angulars,
        shell_k.angulars,
        shell_l.angulars,
        int(shell_i.exponents.shape[0]),
        int(shell_j.exponents.shape[0]),
        int(shell_k.exponents.shape[0]),
        int(shell_l.exponents.shape[0]),
    )


def _can_use_shell_block_path(
    basis: CartesianBasis,
    *,
    screening_threshold: float | None,
    engine: str,
) -> bool:
    if screening_threshold not in (None, 0.0):
        return False
    if not basis.shells:
        return False
    if not _use_jit_engine(engine):
        return False
    max_l = max((sum(ang) for shell in basis.shells for ang in shell.angulars), default=0)
    if max_l > 1:
        return False
    max_nprim = max((int(shell.exponents.shape[0]) for shell in basis.shells), default=0)
    # Allow larger s/p primitive contractions to stay on the shell-block path.
    # For split-valence bases like 6-31G this avoids falling back to the much
    # more expensive AO-quartet path, while still keeping d/f shells out.
    return max_nprim <= 6


def _pad_array_to_size(arr: Array, target_size: int) -> Array:
    arr = jnp.asarray(arr)
    current = int(arr.shape[0])
    target = int(target_size)
    if current >= target:
        return arr
    if current == 0:
        return jnp.zeros((target,) + arr.shape[1:], dtype=arr.dtype)
    pad = target - current
    return jnp.concatenate((arr, jnp.repeat(arr[current - 1 : current], pad, axis=0)), axis=0)


def _pad_primitive_axis_to_size(arr: Array, target_size: int) -> Array:
    arr = jnp.asarray(arr)
    if arr.ndim < 2:
        return arr
    current = int(arr.shape[1])
    target = int(target_size)
    if current >= target:
        return arr
    pad_width = [(0, 0)] * arr.ndim
    pad_width[1] = (0, target - current)
    return jnp.pad(arr, tuple(pad_width))


def _build_signature_registry(groups: tuple, kernel_builder) -> tuple[dict[tuple, int], tuple]:
    sig_to_id: dict[tuple, int] = {}
    kernels: list = []
    for group in groups:
        signature = group.signature
        if signature in sig_to_id:
            continue
        sig_to_id[signature] = len(kernels)
        kernels.append(kernel_builder(*signature))
    if len(kernels) > 64:
        raise ValueError(f"lax.switch supports at most 64 ERI branches, got {len(kernels)}.")
    return sig_to_id, tuple(kernels)


def _signature_limited_group_chunks(
    groups: tuple,
    *,
    max_signatures: int = 64,
    max_padding_ratio: float | None = 2.0,
) -> tuple[tuple, ...]:
    ordered = tuple(
        sorted(groups, key=lambda group: int(group.idx_i.shape[0]), reverse=True)
    )
    chunks: list[tuple] = []
    current: list = []
    current_signatures: set[tuple] = set()
    current_useful = 0
    current_max_batch = 0
    for group in ordered:
        signature = group.signature
        batch_size = int(group.idx_i.shape[0])
        next_signatures = set(current_signatures)
        next_signatures.add(signature)
        next_useful = current_useful + batch_size
        next_max_batch = max(current_max_batch, batch_size)
        next_padding_ratio = (
            (next_max_batch * (len(current) + 1)) / max(1, next_useful)
        )
        signature_limit_hit = (
            signature not in current_signatures and len(current_signatures) >= max_signatures
        )
        padding_limit_hit = (
            max_padding_ratio is not None
            and bool(current)
            and next_padding_ratio > float(max_padding_ratio)
        )
        if signature_limit_hit or padding_limit_hit:
            chunks.append(tuple(current))
            current = []
            current_signatures = set()
            current_useful = 0
            current_max_batch = 0
        current.append(group)
        current_signatures.add(signature)
        current_useful += batch_size
        current_max_batch = max(current_max_batch, batch_size)
    if current:
        chunks.append(tuple(current))
    return tuple(chunks)


def _shell_group_maxima(groups: tuple) -> tuple[int, int, int, int, int, int]:
    max_batch = max((int(group.idx_i.shape[0]) for group in groups), default=0)
    max_block = max(
        (
            len(group.signature[0])
            * len(group.signature[1])
            * len(group.signature[2])
            * len(group.signature[3])
            for group in groups
        ),
        default=0,
    )
    max_nprim_i = max((int(group.signature[4]) for group in groups), default=0)
    max_nprim_j = max((int(group.signature[5]) for group in groups), default=0)
    max_nprim_k = max((int(group.signature[6]) for group in groups), default=0)
    max_nprim_l = max((int(group.signature[7]) for group in groups), default=0)
    return max_batch, max_block, max_nprim_i, max_nprim_j, max_nprim_k, max_nprim_l


def _pad_and_stack_group_inputs(
    basis: CartesianBasis,
    groups: tuple,
    sig_to_id: dict[tuple, int],
    max_batch: int,
    max_nprims: tuple[int, int, int, int],
) -> tuple[tuple[Array, ...], Array, Array]:
    primitive_targets = {
        0: max_nprims[0],
        1: max_nprims[0],
        3: max_nprims[1],
        4: max_nprims[1],
        6: max_nprims[2],
        7: max_nprims[2],
        9: max_nprims[3],
        10: max_nprims[3],
    }
    all_inputs = []
    sig_ids = []
    n_valid = []
    for group in groups:
        signature = group.signature
        batch = _gather_shell_quartet_batch(
            basis,
            group.idx_i,
            group.idx_j,
            group.idx_k,
            group.idx_l,
            nprim_i=signature[4],
            nprim_j=signature[5],
            nprim_k=signature[6],
            nprim_l=signature[7],
        )
        padded = []
        for pos, arr in enumerate(batch):
            padded_arr = _pad_array_to_size(arr, max_batch)
            if pos in primitive_targets:
                padded_arr = _pad_primitive_axis_to_size(padded_arr, primitive_targets[pos])
            padded.append(padded_arr)
        all_inputs.append(tuple(padded))
        sig_ids.append(sig_to_id[signature])
        n_valid.append(int(group.idx_i.shape[0]))
    stacked = tuple(
        jnp.stack([all_inputs[group_idx][input_idx] for group_idx in range(len(groups))])
        for input_idx in range(12)
    )
    return (
        stacked,
        jnp.asarray(sig_ids, dtype=jnp.int32),
        jnp.asarray(n_valid, dtype=jnp.int32),
    )


def _pad_and_stack_pair_indices(
    basis: CartesianBasis,
    groups: tuple,
    max_batch: int,
    max_block_size: int,
    *,
    sentinel: int,
) -> tuple[Array, Array]:
    pair_index = _lower_triangle_pairs_to_matrix(basis.nao, sentinel=sentinel)
    row_groups: list[np.ndarray] = []
    col_groups: list[np.ndarray] = []
    for group in groups:
        signature = group.signature
        ni = len(signature[0])
        nj = len(signature[1])
        nk = len(signature[2])
        nl = len(signature[3])
        batch_size = int(group.idx_i.shape[0])
        block_size = ni * nj * nk * nl

        ao_i = np.asarray(basis.shell_ao_indices_padded[group.idx_i, :ni], dtype=np.int32)
        ao_j = np.asarray(basis.shell_ao_indices_padded[group.idx_j, :nj], dtype=np.int32)
        ao_k = np.asarray(basis.shell_ao_indices_padded[group.idx_k, :nk], dtype=np.int32)
        ao_l = np.asarray(basis.shell_ao_indices_padded[group.idx_l, :nl], dtype=np.int32)

        rows = pair_index[ao_i[:, :, None], ao_j[:, None, :]]
        cols = pair_index[ao_k[:, :, None], ao_l[:, None, :]]
        rows_flat = np.broadcast_to(
            rows[:, :, :, None, None],
            (batch_size, ni, nj, nk, nl),
        ).reshape(batch_size, block_size)
        cols_flat = np.broadcast_to(
            cols[:, None, None, :, :],
            (batch_size, ni, nj, nk, nl),
        ).reshape(batch_size, block_size)

        rows_pad = np.full((max_batch, max_block_size), sentinel, dtype=np.int32)
        cols_pad = np.full((max_batch, max_block_size), sentinel, dtype=np.int32)
        rows_pad[:batch_size, :block_size] = rows_flat
        cols_pad[:batch_size, :block_size] = cols_flat
        row_groups.append(rows_pad.reshape(-1))
        col_groups.append(cols_pad.reshape(-1))
    return jnp.asarray(np.stack(row_groups, axis=0)), jnp.asarray(np.stack(col_groups, axis=0))


def _pad_and_stack_4index_scatter_indices(
    groups: tuple,
    max_batch: int,
    max_block_size: int,
    *,
    sentinel: int,
) -> tuple[Array, Array, Array, Array]:
    scatters: tuple[list[np.ndarray], ...] = ([], [], [], [])
    for group in groups:
        signature = group.signature
        batch_size = int(group.idx_i.shape[0])
        block_size = (
            len(signature[0])
            * len(signature[1])
            * len(signature[2])
            * len(signature[3])
        )
        target_shape = (8, int(max_batch), int(max_block_size))
        for dest, scatter in zip(
            scatters,
            (group.scatter_i, group.scatter_j, group.scatter_k, group.scatter_l),
        ):
            padded = np.full(target_shape, sentinel, dtype=np.int32)
            arr = np.asarray(scatter, dtype=np.int32).reshape(8, batch_size, block_size)
            padded[:, :batch_size, :block_size] = arr
            dest.append(padded.reshape(-1))
    return tuple(jnp.asarray(np.stack(parts, axis=0)) for parts in scatters)


def _make_shell_pair_branch(
    signature: tuple,
    *,
    max_block_size: int,
):
    kernel = _compiled_eri_shell_block_kernel_batched(*signature)
    block_size = len(signature[0]) * len(signature[1]) * len(signature[2]) * len(signature[3])
    nprim_i, nprim_j, nprim_k, nprim_l = (
        int(signature[4]),
        int(signature[5]),
        int(signature[6]),
        int(signature[7]),
    )

    def branch(
        exp_i,
        coeff_i,
        center_i,
        exp_j,
        coeff_j,
        center_j,
        exp_k,
        coeff_k,
        center_k,
        exp_l,
        coeff_l,
        center_l,
    ):
        blocks = kernel(
            exp_i[:, :nprim_i],
            coeff_i[:, :nprim_i],
            center_i,
            exp_j[:, :nprim_j],
            coeff_j[:, :nprim_j],
            center_j,
            exp_k[:, :nprim_k],
            coeff_k[:, :nprim_k],
            center_k,
            exp_l[:, :nprim_l],
            coeff_l[:, :nprim_l],
            center_l,
        )
        flat = blocks.reshape(blocks.shape[0], block_size)
        if block_size < max_block_size:
            flat = jnp.pad(flat, ((0, 0), (0, max_block_size - block_size)))
        return flat.reshape(-1)

    return branch


def _make_shell_tensor_branch(
    signature: tuple,
    *,
    max_block_size: int,
):
    kernel = _compiled_eri_shell_block_kernel_batched(*signature)
    block_size = len(signature[0]) * len(signature[1]) * len(signature[2]) * len(signature[3])
    nprim_i, nprim_j, nprim_k, nprim_l = (
        int(signature[4]),
        int(signature[5]),
        int(signature[6]),
        int(signature[7]),
    )

    def _flatten_blocks(blocks: Array) -> Array:
        flat = blocks.reshape(blocks.shape[0], block_size)
        if block_size < max_block_size:
            flat = jnp.pad(flat, ((0, 0), (0, max_block_size - block_size)))
        return flat.reshape(-1)

    def branch(
        exp_i,
        coeff_i,
        center_i,
        exp_j,
        coeff_j,
        center_j,
        exp_k,
        coeff_k,
        center_k,
        exp_l,
        coeff_l,
        center_l,
    ):
        blocks = kernel(
            exp_i[:, :nprim_i],
            coeff_i[:, :nprim_i],
            center_i,
            exp_j[:, :nprim_j],
            coeff_j[:, :nprim_j],
            center_j,
            exp_k[:, :nprim_k],
            coeff_k[:, :nprim_k],
            center_k,
            exp_l[:, :nprim_l],
            coeff_l[:, :nprim_l],
            center_l,
        )
        return jnp.concatenate(
            (
                _flatten_blocks(blocks),
                _flatten_blocks(blocks.transpose(0, 2, 1, 3, 4)),
                _flatten_blocks(blocks.transpose(0, 1, 2, 4, 3)),
                _flatten_blocks(blocks.transpose(0, 2, 1, 4, 3)),
                _flatten_blocks(blocks.transpose(0, 3, 4, 1, 2)),
                _flatten_blocks(blocks.transpose(0, 4, 3, 1, 2)),
                _flatten_blocks(blocks.transpose(0, 3, 4, 2, 1)),
                _flatten_blocks(blocks.transpose(0, 4, 3, 2, 1)),
            ),
            axis=0,
        )

    return branch


def _ao_signature_uses_jit(signature: tuple, engine: str) -> bool:
    return _use_jit_engine(
        engine,
        angulars=(signature[0], signature[1], signature[2], signature[3]),
        nprims=(signature[4], signature[5], signature[6], signature[7]),
    )


def _pad_and_stack_ao_group_inputs(
    basis: CartesianBasis,
    groups: tuple,
    sig_to_id: dict[tuple, int],
    max_batch: int,
    max_nprims: tuple[int, int, int, int],
) -> tuple[tuple[Array, ...], Array, Array]:
    primitive_targets = {
        0: max_nprims[0],
        1: max_nprims[0],
        3: max_nprims[1],
        4: max_nprims[1],
        6: max_nprims[2],
        7: max_nprims[2],
        9: max_nprims[3],
        10: max_nprims[3],
    }
    all_inputs = []
    sig_ids = []
    n_valid = []
    for group in groups:
        signature = group.signature
        batch = _gather_quartet_batch(
            basis,
            group.idx_i,
            group.idx_j,
            group.idx_k,
            group.idx_l,
            nprim_i=signature[4],
            nprim_j=signature[5],
            nprim_k=signature[6],
            nprim_l=signature[7],
        )
        padded = []
        for pos, arr in enumerate(batch):
            padded_arr = _pad_array_to_size(arr, max_batch)
            if pos in primitive_targets:
                padded_arr = _pad_primitive_axis_to_size(padded_arr, primitive_targets[pos])
            padded.append(padded_arr)
        all_inputs.append(tuple(padded))
        sig_ids.append(sig_to_id[signature])
        n_valid.append(int(group.idx_i.shape[0]))
    stacked = tuple(
        jnp.stack([all_inputs[group_idx][input_idx] for group_idx in range(len(groups))])
        for input_idx in range(12)
    )
    return (
        stacked,
        jnp.asarray(sig_ids, dtype=jnp.int32),
        jnp.asarray(n_valid, dtype=jnp.int32),
    )


def _pad_and_stack_ao_pair_indices(
    basis: CartesianBasis,
    groups: tuple,
    max_batch: int,
    *,
    sentinel: int,
) -> tuple[Array, Array]:
    pair_index = _lower_triangle_pairs_to_matrix(basis.nao, sentinel=sentinel)
    row_groups: list[np.ndarray] = []
    col_groups: list[np.ndarray] = []
    for group in groups:
        rows = pair_index[np.asarray(group.idx_i), np.asarray(group.idx_j)]
        cols = pair_index[np.asarray(group.idx_k), np.asarray(group.idx_l)]
        batch_size = int(group.idx_i.shape[0])
        rows_pad = np.full((max_batch,), sentinel, dtype=np.int32)
        cols_pad = np.full((max_batch,), sentinel, dtype=np.int32)
        rows_pad[:batch_size] = rows
        cols_pad[:batch_size] = cols
        row_groups.append(rows_pad)
        col_groups.append(cols_pad)
    return jnp.asarray(np.stack(row_groups, axis=0)), jnp.asarray(np.stack(col_groups, axis=0))


def _make_ao_pair_branch(signature: tuple):
    kernel = _compiled_eri_pair_kernel_batched(*signature)
    nprim_i, nprim_j, nprim_k, nprim_l = (
        int(signature[4]),
        int(signature[5]),
        int(signature[6]),
        int(signature[7]),
    )

    def branch(
        exp_i,
        coeff_i,
        center_i,
        exp_j,
        coeff_j,
        center_j,
        exp_k,
        coeff_k,
        center_k,
        exp_l,
        coeff_l,
        center_l,
    ):
        return kernel(
            exp_i[:, :nprim_i],
            coeff_i[:, :nprim_i],
            center_i,
            exp_j[:, :nprim_j],
            coeff_j[:, :nprim_j],
            center_j,
            exp_k[:, :nprim_k],
            coeff_k[:, :nprim_k],
            center_k,
            exp_l[:, :nprim_l],
            coeff_l[:, :nprim_l],
            center_l,
        )

    return branch


@functools.lru_cache(maxsize=None)
def _compiled_ao_pair_scan_executor(
    signatures_by_id: tuple,
    npair: int,
):
    branches = tuple(_make_ao_pair_branch(signature) for signature in signatures_by_id)

    @jax.jit
    def executor(
        stacked_inputs: tuple[Array, ...],
        sig_ids: Array,
        rows: Array,
        cols: Array,
    ) -> Array:
        dtype = stacked_inputs[0].dtype
        pair_init = jnp.zeros((int(npair) + 1, int(npair) + 1), dtype=dtype)

        def scan_body(pair_acc: Array, group_idx: Array):
            batch_inputs = tuple(inp[group_idx] for inp in stacked_inputs)
            vals = jax.lax.switch(sig_ids[group_idx], branches, *batch_inputs)
            group_rows = rows[group_idx]
            group_cols = cols[group_idx]
            pair_acc = pair_acc.at[group_rows, group_cols].set(vals)
            pair_acc = pair_acc.at[group_cols, group_rows].set(vals)
            return pair_acc, None

        pair_with_sentinel, _ = jax.lax.scan(
            scan_body,
            pair_init,
            jnp.arange(sig_ids.shape[0], dtype=jnp.int32),
        )
        pair = pair_with_sentinel[: int(npair), : int(npair)]
        return 0.5 * (pair + pair.T)

    return executor


@functools.lru_cache(maxsize=None)
def _compiled_shell_pair_scan_executor(
    signatures_by_id: tuple,
    npair: int,
    max_block_size: int,
):
    branches = tuple(
        _make_shell_pair_branch(signature, max_block_size=int(max_block_size))
        for signature in signatures_by_id
    )

    @jax.jit
    def executor(
        stacked_inputs: tuple[Array, ...],
        sig_ids: Array,
        rows: Array,
        cols: Array,
    ) -> Array:
        dtype = stacked_inputs[0].dtype
        pair_init = jnp.zeros((int(npair) + 1, int(npair) + 1), dtype=dtype)

        def scan_body(pair_acc: Array, group_idx: Array):
            batch_inputs = tuple(inp[group_idx] for inp in stacked_inputs)
            vals = jax.lax.switch(sig_ids[group_idx], branches, *batch_inputs)
            group_rows = rows[group_idx]
            group_cols = cols[group_idx]
            pair_acc = pair_acc.at[group_rows, group_cols].set(vals)
            pair_acc = pair_acc.at[group_cols, group_rows].set(vals)
            return pair_acc, None

        pair_with_sentinel, _ = jax.lax.scan(
            scan_body,
            pair_init,
            jnp.arange(sig_ids.shape[0], dtype=jnp.int32),
        )
        pair = pair_with_sentinel[: int(npair), : int(npair)]
        return 0.5 * (pair + pair.T)

    return executor


@functools.lru_cache(maxsize=None)
def _compiled_shell_tensor_scan_executor(
    signatures_by_id: tuple,
    nao: int,
    max_block_size: int,
):
    branches = tuple(
        _make_shell_tensor_branch(signature, max_block_size=int(max_block_size))
        for signature in signatures_by_id
    )

    @jax.jit
    def executor(
        stacked_inputs: tuple[Array, ...],
        sig_ids: Array,
        scatter_i: Array,
        scatter_j: Array,
        scatter_k: Array,
        scatter_l: Array,
    ) -> Array:
        dtype = stacked_inputs[0].dtype
        n = int(nao)
        eri_init = jnp.zeros((n + 1, n + 1, n + 1, n + 1), dtype=dtype)

        def scan_body(eri_acc: Array, group_idx: Array):
            batch_inputs = tuple(inp[group_idx] for inp in stacked_inputs)
            vals = jax.lax.switch(sig_ids[group_idx], branches, *batch_inputs)
            eri_acc = eri_acc.at[
                scatter_i[group_idx],
                scatter_j[group_idx],
                scatter_k[group_idx],
                scatter_l[group_idx],
            ].set(vals)
            return eri_acc, None

        eri_with_sentinel, _ = jax.lax.scan(
            scan_body,
            eri_init,
            jnp.arange(sig_ids.shape[0], dtype=jnp.int32),
        )
        return eri_with_sentinel[:n, :n, :n, :n]

    return executor


def _fused_eri_pair_matrix_from_ao_groups(
    basis: CartesianBasis,
    groups: tuple,
    *,
    engine: str,
) -> Array:
    groups = tuple(groups)
    n = basis.nao
    if n == 0:
        return jnp.zeros((0, 0))
    if not groups:
        raise ValueError("Fused AO-pair ERI path requires AO quartet groups.")
    if not all(_ao_signature_uses_jit(group.signature, engine) for group in groups):
        raise ValueError("Fused AO-pair ERI path only accepts JIT-compatible groups.")

    npair = n * (n + 1) // 2
    group_chunks = _signature_limited_group_chunks(groups)
    if len(group_chunks) > 1:
        dtype = jnp.asarray(basis.ao_exponents_padded).dtype
        pair = jnp.zeros((npair, npair), dtype=dtype)
        for chunk in group_chunks:
            pair = pair + _fused_eri_pair_matrix_from_ao_groups(
                basis,
                chunk,
                engine=engine,
            )
        return pair

    sig_to_id, _ = _build_signature_registry(groups, _compiled_eri_pair_kernel_batched)
    max_batch, _, max_nprim_i, max_nprim_j, max_nprim_k, max_nprim_l = _shell_group_maxima(
        groups
    )
    stacked_inputs, sig_ids, _ = _pad_and_stack_ao_group_inputs(
        basis,
        groups,
        sig_to_id,
        max_batch,
        (max_nprim_i, max_nprim_j, max_nprim_k, max_nprim_l),
    )
    rows, cols = _pad_and_stack_ao_pair_indices(
        basis,
        groups,
        max_batch,
        sentinel=npair,
    )
    signatures_by_id = tuple(
        signature for signature, _ in sorted(sig_to_id.items(), key=lambda item: item[1])
    )
    executor = _compiled_ao_pair_scan_executor(signatures_by_id, npair)
    return executor(stacked_inputs, sig_ids, rows, cols)


def _fused_eri_tensor_from_shell_groups(
    basis: CartesianBasis,
    groups: tuple,
) -> Array:
    groups = tuple(groups)
    n = basis.nao
    if n == 0:
        return jnp.zeros((0, 0, 0, 0))
    if not groups:
        raise ValueError("Fused shell-block ERI tensor path requires shell quartet groups.")

    group_chunks = _signature_limited_group_chunks(groups)
    if len(group_chunks) > 1:
        dtype = jnp.asarray(basis.shell_exponents_padded).dtype
        eri = jnp.zeros((n, n, n, n), dtype=dtype)
        for chunk in group_chunks:
            eri = eri + _fused_eri_tensor_from_shell_groups(basis, chunk)
        return eri

    sig_to_id, _ = _build_signature_registry(groups, _compiled_eri_shell_block_kernel_batched)
    max_batch, max_block_size, max_nprim_i, max_nprim_j, max_nprim_k, max_nprim_l = (
        _shell_group_maxima(groups)
    )
    stacked_inputs, sig_ids, _ = _pad_and_stack_group_inputs(
        basis,
        groups,
        sig_to_id,
        max_batch,
        (max_nprim_i, max_nprim_j, max_nprim_k, max_nprim_l),
    )
    scatter_i, scatter_j, scatter_k, scatter_l = _pad_and_stack_4index_scatter_indices(
        groups,
        max_batch,
        max_block_size,
        sentinel=n,
    )
    signatures_by_id = tuple(
        signature for signature, _ in sorted(sig_to_id.items(), key=lambda item: item[1])
    )
    executor = _compiled_shell_tensor_scan_executor(
        signatures_by_id,
        n,
        max_block_size,
    )
    return executor(stacked_inputs, sig_ids, scatter_i, scatter_j, scatter_k, scatter_l)


def _fused_eri_pair_matrix_from_shell_groups(
    basis: CartesianBasis,
    groups: tuple,
) -> Array:
    groups = tuple(groups)
    n = basis.nao
    if n == 0:
        return jnp.zeros((0, 0))
    if not groups:
        raise ValueError("Fused shell-block packed ERI path requires shell quartet groups.")

    npair = n * (n + 1) // 2
    group_chunks = _signature_limited_group_chunks(groups)
    if len(group_chunks) > 1:
        dtype = jnp.asarray(basis.shell_exponents_padded).dtype
        pair = jnp.zeros((npair, npair), dtype=dtype)
        for chunk in group_chunks:
            pair = pair + _fused_eri_pair_matrix_from_shell_groups(basis, chunk)
        return pair

    sig_to_id, _ = _build_signature_registry(groups, _compiled_eri_shell_block_kernel_batched)
    max_batch, max_block_size, max_nprim_i, max_nprim_j, max_nprim_k, max_nprim_l = (
        _shell_group_maxima(groups)
    )
    stacked_inputs, sig_ids, _ = _pad_and_stack_group_inputs(
        basis,
        groups,
        sig_to_id,
        max_batch,
        (max_nprim_i, max_nprim_j, max_nprim_k, max_nprim_l),
    )
    rows, cols = _pad_and_stack_pair_indices(
        basis,
        groups,
        max_batch,
        max_block_size,
        sentinel=npair,
    )
    signatures_by_id = tuple(
        signature for signature, _ in sorted(sig_to_id.items(), key=lambda item: item[1])
    )
    executor = _compiled_shell_pair_scan_executor(
        signatures_by_id,
        npair,
        max_block_size,
    )
    return executor(stacked_inputs, sig_ids, rows, cols)


@functools.lru_cache(maxsize=None)
def _compiled_eri_shell_block_kernel(
    angulars_i: tuple[tuple[int, int, int], ...],
    angulars_j: tuple[tuple[int, int, int], ...],
    angulars_k: tuple[tuple[int, int, int], ...],
    angulars_l: tuple[tuple[int, int, int], ...],
    nprim_i: int,
    nprim_j: int,
    nprim_k: int,
    nprim_l: int,
):
    ni = len(angulars_i)
    nj = len(angulars_j)
    nk = len(angulars_k)
    nl = len(angulars_l)
    axis_j = tuple(_angular_axis(ang) for ang in angulars_j)
    axis_l = tuple(_angular_axis(ang) for ang in angulars_l)
    max_vrr_l = max(
        (
            sum(ang_i) + (1 if ax_j >= 0 else 0)
            + sum(ang_k) + (1 if ax_l >= 0 else 0)
            for ang_i in angulars_i
            for ax_j in axis_j
            for ang_k in angulars_k
            for ax_l in axis_l
        ),
        default=0,
    )

    def primitive_block(
        alpha: Array,
        beta: Array,
        gamma: Array,
        delta: Array,
        center_i: Array,
        center_j: Array,
        center_k: Array,
        center_l: Array,
    ) -> Array:
        p = alpha + beta
        q = gamma + delta
        mu = alpha * beta / p
        nu = gamma * delta / q

        rab2 = jnp.dot(center_i - center_j, center_i - center_j)
        rcd2 = jnp.dot(center_k - center_l, center_k - center_l)
        center_p = (alpha * center_i + beta * center_j) / p
        center_q = (gamma * center_k + delta * center_l) / q
        center_w = (p * center_p + q * center_q) / (p + q)
        rpq2 = jnp.dot(center_p - center_q, center_p - center_q)

        pref = 2.0 * jnp.pi**2.5 / (p * q * jnp.sqrt(p + q))
        pref = pref * jnp.exp(-mu * rab2 - nu * rcd2)
        boys = pref * _boys_values(max_vrr_l, (p * q / (p + q)) * rpq2)
        bra_ratio = q / (p + q)
        ket_ratio = p / (p + q)
        zero = jnp.zeros_like(boys[0])

        @functools.lru_cache(maxsize=None)
        def compute_vrr(
            a: tuple[int, int, int],
            c: tuple[int, int, int],
            m: int,
        ) -> Array:
            if min(a + c) < 0:
                return zero
            if sum(a) + sum(c) == 0:
                return boys[m]

            if sum(a) > 0:
                axis = next(idx for idx, power in enumerate(a) if power > 0)
                a1 = _tuple_dec(a, axis)
                out = (center_p[axis] - center_i[axis]) * compute_vrr(a1, c, m)
                out = out + (center_w[axis] - center_p[axis]) * compute_vrr(a1, c, m + 1)
                if a1[axis] > 0:
                    a2 = _tuple_dec(a1, axis)
                    coef = a1[axis] / (2.0 * p)
                    out = out + coef * (
                        compute_vrr(a2, c, m) - bra_ratio * compute_vrr(a2, c, m + 1)
                    )
                if c[axis] > 0:
                    c1 = _tuple_dec(c, axis)
                    coef = c[axis] / (2.0 * (p + q))
                    out = out + coef * compute_vrr(a1, c1, m + 1)
                return out

            axis = next(idx for idx, power in enumerate(c) if power > 0)
            c1 = _tuple_dec(c, axis)
            out = (center_q[axis] - center_k[axis]) * compute_vrr(a, c1, m)
            out = out + (center_w[axis] - center_q[axis]) * compute_vrr(a, c1, m + 1)
            if c1[axis] > 0:
                c2 = _tuple_dec(c1, axis)
                coef = c1[axis] / (2.0 * q)
                out = out + coef * (
                    compute_vrr(a, c2, m) - ket_ratio * compute_vrr(a, c2, m + 1)
                )
            if a[axis] > 0:
                a1 = _tuple_dec(a, axis)
                coef = a[axis] / (2.0 * (p + q))
                out = out + coef * compute_vrr(a1, c1, m + 1)
            return out

        values: list[Array] = []
        for ang_i in angulars_i:
            for ax_j in axis_j:
                if ax_j >= 0:
                    ang_i_plus = _tuple_inc(ang_i, ax_j)
                    shift_ab = center_i[ax_j] - center_j[ax_j]
                else:
                    ang_i_plus = ang_i
                    shift_ab = jnp.asarray(0.0, dtype=boys.dtype)
                for ang_k in angulars_k:
                    for ax_l in axis_l:
                        if ax_l >= 0:
                            ang_k_plus = _tuple_inc(ang_k, ax_l)
                            shift_cd = center_k[ax_l] - center_l[ax_l]
                        else:
                            ang_k_plus = ang_k
                            shift_cd = jnp.asarray(0.0, dtype=boys.dtype)

                        base = compute_vrr(ang_i, ang_k, 0)
                        if ax_j < 0 and ax_l < 0:
                            val = base
                        elif ax_j >= 0 and ax_l < 0:
                            val = compute_vrr(ang_i_plus, ang_k, 0) + shift_ab * base
                        elif ax_j < 0 and ax_l >= 0:
                            val = compute_vrr(ang_i, ang_k_plus, 0) + shift_cd * base
                        else:
                            val = compute_vrr(ang_i_plus, ang_k_plus, 0)
                            val = val + shift_cd * compute_vrr(ang_i_plus, ang_k, 0)
                            val = val + shift_ab * compute_vrr(ang_i, ang_k_plus, 0)
                            val = val + (shift_ab * shift_cd) * base
                        values.append(val)
        return jnp.asarray(values, dtype=boys.dtype).reshape(ni, nj, nk, nl)

    def kernel(
        exponents_i: Array,
        coefficients_i: Array,
        center_i: Array,
        exponents_j: Array,
        coefficients_j: Array,
        center_j: Array,
        exponents_k: Array,
        coefficients_k: Array,
        center_k: Array,
        exponents_l: Array,
        coefficients_l: Array,
        center_l: Array,
    ) -> Array:
        alpha, beta, gamma, delta = jnp.meshgrid(
            exponents_i,
            exponents_j,
            exponents_k,
            exponents_l,
            indexing="ij",
        )
        alpha_flat = alpha.reshape(-1)
        beta_flat = beta.reshape(-1)
        gamma_flat = gamma.reshape(-1)
        delta_flat = delta.reshape(-1)

        prim_blocks = jax.vmap(
            primitive_block,
            in_axes=(0, 0, 0, 0, None, None, None, None),
        )(
            alpha_flat,
            beta_flat,
            gamma_flat,
            delta_flat,
            center_i,
            center_j,
            center_k,
            center_l,
        )

        norm_i = jnp.stack(
            [primitive_cartesian_norm(exponents_i, ang_i) for ang_i in angulars_i],
            axis=0,
        )
        norm_j = jnp.stack(
            [primitive_cartesian_norm(exponents_j, ang_j) for ang_j in angulars_j],
            axis=0,
        )
        norm_k = jnp.stack(
            [primitive_cartesian_norm(exponents_k, ang_k) for ang_k in angulars_k],
            axis=0,
        )
        norm_l = jnp.stack(
            [primitive_cartesian_norm(exponents_l, ang_l) for ang_l in angulars_l],
            axis=0,
        )

        weighted_i = coefficients_i[None, :] * norm_i
        weighted_j = coefficients_j[None, :] * norm_j
        weighted_k = coefficients_k[None, :] * norm_k
        weighted_l = coefficients_l[None, :] * norm_l
        weight_flat = jnp.einsum(
            "ai,bj,ck,dl->abcdijkl",
            weighted_i,
            weighted_j,
            weighted_k,
            weighted_l,
            precision=jax.lax.Precision.HIGHEST,
        ).reshape(ni, nj, nk, nl, -1)
        return jnp.einsum(
            "abcdq,qabcd->abcd",
            weight_flat,
            prim_blocks,
            precision=jax.lax.Precision.HIGHEST,
        )

    return jax.jit(kernel)


@functools.lru_cache(maxsize=None)
def _compiled_eri_shell_block_kernel_batched(
    angulars_i: tuple[tuple[int, int, int], ...],
    angulars_j: tuple[tuple[int, int, int], ...],
    angulars_k: tuple[tuple[int, int, int], ...],
    angulars_l: tuple[tuple[int, int, int], ...],
    nprim_i: int,
    nprim_j: int,
    nprim_k: int,
    nprim_l: int,
):
    scalar = _compiled_eri_shell_block_kernel(
        angulars_i,
        angulars_j,
        angulars_k,
        angulars_l,
        nprim_i,
        nprim_j,
        nprim_k,
        nprim_l,
    )
    return jax.jit(
        jax.vmap(
            scalar,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        )
    )


def _gather_shell_quartet_batch(
    basis: CartesianBasis,
    shell_i_idx,
    shell_j_idx,
    shell_k_idx,
    shell_l_idx,
    *,
    nprim_i: int,
    nprim_j: int,
    nprim_k: int,
    nprim_l: int,
):
    ii = jnp.asarray(shell_i_idx, dtype=jnp.int32)
    jj = jnp.asarray(shell_j_idx, dtype=jnp.int32)
    kk = jnp.asarray(shell_k_idx, dtype=jnp.int32)
    ll = jnp.asarray(shell_l_idx, dtype=jnp.int32)
    exp_i = basis.shell_exponents_padded[ii, :nprim_i]
    coeff_i = basis.shell_coefficients_padded[ii, :nprim_i]
    center_i = basis.shell_centers[ii]
    exp_j = basis.shell_exponents_padded[jj, :nprim_j]
    coeff_j = basis.shell_coefficients_padded[jj, :nprim_j]
    center_j = basis.shell_centers[jj]
    exp_k = basis.shell_exponents_padded[kk, :nprim_k]
    coeff_k = basis.shell_coefficients_padded[kk, :nprim_k]
    center_k = basis.shell_centers[kk]
    exp_l = basis.shell_exponents_padded[ll, :nprim_l]
    coeff_l = basis.shell_coefficients_padded[ll, :nprim_l]
    center_l = basis.shell_centers[ll]
    return (
        exp_i,
        coeff_i,
        center_i,
        exp_j,
        coeff_j,
        center_j,
        exp_k,
        coeff_k,
        center_k,
        exp_l,
        coeff_l,
        center_l,
    )


def _legacy_eri_tensor_shell_block(
    basis: CartesianBasis,
    *,
    engine: str,
) -> Array:
    n = basis.nao
    nshell = len(basis.shells)
    if n == 0 or nshell == 0:
        return jnp.zeros((n, n, n, n))

    eri = jnp.zeros((n, n, n, n))

    groups = basis.shell_quartet_groups
    if not groups:
        raise ValueError("Shell-block ERI path requires precomputed shell quartet groups.")

    for group in groups:
        signature = group.signature
        idx_i_arr = jnp.asarray(group.idx_i, dtype=jnp.int32)
        idx_j_arr = jnp.asarray(group.idx_j, dtype=jnp.int32)
        idx_k_arr = jnp.asarray(group.idx_k, dtype=jnp.int32)
        idx_l_arr = jnp.asarray(group.idx_l, dtype=jnp.int32)
        kernel = _compiled_eri_shell_block_kernel_batched(*signature)
        blocks = _run_quartet_kernel_chunked(
            kernel,
            group.batch_inputs,
        )
        ni = len(signature[0])
        nj = len(signature[1])
        nk = len(signature[2])
        nl = len(signature[3])
        block_size = int(idx_i_arr.shape[0]) * ni * nj * nk * nl
        scatter_kernel = _compiled_shell_block_scatter_kernel(block_size)
        eri = scatter_kernel(
            eri,
            group.scatter_i,
            group.scatter_j,
            group.scatter_k,
            group.scatter_l,
            blocks,
        )
    return eri


def _eri_tensor_shell_block(
    basis: CartesianBasis,
    *,
    engine: str,
) -> Array:
    n = basis.nao
    nshell = len(basis.shells)
    if n == 0 or nshell == 0:
        return jnp.zeros((n, n, n, n))
    groups = basis.shell_quartet_groups
    if not groups:
        raise ValueError("Shell-block ERI path requires precomputed shell quartet groups.")
    return _fused_eri_tensor_from_shell_groups(basis, groups)


def _legacy_eri_pair_matrix_packed_shell_block(
    basis: CartesianBasis,
    *,
    engine: str,
) -> Array:
    n = basis.nao
    nshell = len(basis.shells)
    if n == 0 or nshell == 0:
        return jnp.zeros((0, 0))

    ao_pairs = _lower_triangle_pairs(n)
    npair = len(ao_pairs)
    pair_index = np.full((n, n), -1, dtype=np.int32)
    for pos, (i, j) in enumerate(ao_pairs):
        pair_index[i, j] = pos
        pair_index[j, i] = pos
    pair_index_arr = jnp.asarray(pair_index, dtype=jnp.int32)

    pair = jnp.zeros((npair, npair))
    groups = basis.shell_quartet_groups
    if not groups:
        raise ValueError("Shell-block packed ERI path requires precomputed shell quartet groups.")

    for group in groups:
        signature = group.signature
        idx_i_arr = jnp.asarray(group.idx_i, dtype=jnp.int32)
        idx_j_arr = jnp.asarray(group.idx_j, dtype=jnp.int32)
        idx_k_arr = jnp.asarray(group.idx_k, dtype=jnp.int32)
        idx_l_arr = jnp.asarray(group.idx_l, dtype=jnp.int32)
        kernel = _compiled_eri_shell_block_kernel_batched(*signature)
        blocks = _run_quartet_kernel_chunked(
            kernel,
            _gather_shell_quartet_batch(
                basis,
                idx_i_arr,
                idx_j_arr,
                idx_k_arr,
                idx_l_arr,
                nprim_i=signature[4],
                nprim_j=signature[5],
                nprim_k=signature[6],
                nprim_l=signature[7],
            ),
        )
        ni = len(signature[0])
        nj = len(signature[1])
        nk = len(signature[2])
        nl = len(signature[3])
        ao_i = basis.shell_ao_indices_padded[idx_i_arr, :ni]
        ao_j = basis.shell_ao_indices_padded[idx_j_arr, :nj]
        ao_k = basis.shell_ao_indices_padded[idx_k_arr, :nk]
        ao_l = basis.shell_ao_indices_padded[idx_l_arr, :nl]

        rows = pair_index_arr[ao_i[:, :, None], ao_j[:, None, :]]
        cols = pair_index_arr[ao_k[:, :, None], ao_l[:, None, :]]
        rows = jnp.broadcast_to(rows[:, :, :, None, None], blocks.shape).reshape(-1)
        cols = jnp.broadcast_to(cols[:, None, None, :, :], blocks.shape).reshape(-1)
        vals = blocks.reshape(-1)
        pair = pair.at[rows, cols].set(vals)
        pair = pair.at[cols, rows].set(vals)
    return 0.5 * (pair + pair.T)


def _eri_pair_matrix_packed_shell_block(
    basis: CartesianBasis,
    *,
    engine: str,
) -> Array:
    n = basis.nao
    nshell = len(basis.shells)
    if n == 0 or nshell == 0:
        return jnp.zeros((0, 0))
    groups = basis.shell_quartet_groups
    if not groups:
        raise ValueError("Shell-block packed ERI path requires precomputed shell quartet groups.")
    return _fused_eri_pair_matrix_from_shell_groups(basis, groups)


def _gather_quartet_batch(
    basis: CartesianBasis,
    idx_i,
    idx_j,
    idx_k,
    idx_l,
    *,
    nprim_i: int,
    nprim_j: int,
    nprim_k: int,
    nprim_l: int,
):
    ii = jnp.asarray(idx_i, dtype=jnp.int32)
    jj = jnp.asarray(idx_j, dtype=jnp.int32)
    kk = jnp.asarray(idx_k, dtype=jnp.int32)
    ll = jnp.asarray(idx_l, dtype=jnp.int32)
    exp_i = basis.ao_exponents_padded[ii, :nprim_i]
    coeff_i = basis.ao_coefficients_padded[ii, :nprim_i]
    center_i = basis.ao_centers[ii]
    exp_j = basis.ao_exponents_padded[jj, :nprim_j]
    coeff_j = basis.ao_coefficients_padded[jj, :nprim_j]
    center_j = basis.ao_centers[jj]
    exp_k = basis.ao_exponents_padded[kk, :nprim_k]
    coeff_k = basis.ao_coefficients_padded[kk, :nprim_k]
    center_k = basis.ao_centers[kk]
    exp_l = basis.ao_exponents_padded[ll, :nprim_l]
    coeff_l = basis.ao_coefficients_padded[ll, :nprim_l]
    center_l = basis.ao_centers[ll]
    return (
        exp_i,
        coeff_i,
        center_i,
        exp_j,
        coeff_j,
        center_j,
        exp_k,
        coeff_k,
        center_k,
        exp_l,
        coeff_l,
        center_l,
    )


def _run_quartet_kernel_chunked(
    kernel,
    batch_inputs: tuple[Array, ...],
    *,
    chunk_size: int = QUARTET_BATCH_CHUNK,
) -> Array:
    n_items = int(batch_inputs[0].shape[0])
    if n_items == 0:
        return jnp.zeros((0,))

    base_chunk = max(int(chunk_size), 1)
    logical_chunk = min(base_chunk, n_items)
    outputs: list[Array] = []
    n_devices = jax.local_device_count()
    mapped_kernel = (
        jax.pmap(lambda *args: kernel(*args))
        if n_devices > 1
        else None
    )

    target_size = logical_chunk
    if mapped_kernel is not None:
        target_size = ((logical_chunk + n_devices - 1) // n_devices) * n_devices

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

    for start in range(0, n_items, base_chunk):
        end = min(start + base_chunk, n_items)
        chunk = tuple(inp[start:end] for inp in batch_inputs)
        valid = int(chunk[0].shape[0])
        fixed_target = target_size if start + base_chunk > n_items else (
            ((base_chunk + n_devices - 1) // n_devices) * n_devices
            if mapped_kernel is not None
            else base_chunk
        )
        fixed_chunk = _pad_chunk_to_fixed_size(chunk, fixed_target)
        if mapped_kernel is not None:
            per_device = int(fixed_chunk[0].shape[0]) // n_devices
            sharded = tuple(
                inp.reshape((n_devices, per_device) + inp.shape[1:])
                for inp in fixed_chunk
            )
            out = mapped_kernel(*sharded)
            out = out.reshape((n_devices * per_device,) + out.shape[2:])
            outputs.append(out[:valid])
        else:
            out = kernel(*fixed_chunk)
            outputs.append(out[:valid])
    if len(outputs) == 1:
        return outputs[0]
    return jnp.concatenate(outputs, axis=0)


def eri_element(
    basis: CartesianBasis,
    i: int,
    j: int,
    k: int,
    l: int,
    *,
    engine: str = "auto",
) -> Array:
    """Single ERI element (ij|kl) in AO basis."""

    ao_i = basis.aos[i]
    ao_j = basis.aos[j]
    ao_k = basis.aos[k]
    ao_l = basis.aos[l]
    if not _use_jit_engine(
        engine,
        angulars=(ao_i.angular, ao_j.angular, ao_k.angular, ao_l.angular),
        nprims=(
            int(ao_i.exponents.shape[0]),
            int(ao_j.exponents.shape[0]),
            int(ao_k.exponents.shape[0]),
            int(ao_l.exponents.shape[0]),
        ),
    ):
        return _contracted_eri(ao_i, ao_j, ao_k, ao_l)
    kernel = _compiled_eri_pair_kernel(
        ao_i.angular,
        ao_j.angular,
        ao_k.angular,
        ao_l.angular,
        int(ao_i.exponents.shape[0]),
        int(ao_j.exponents.shape[0]),
        int(ao_k.exponents.shape[0]),
        int(ao_l.exponents.shape[0]),
    )
    return kernel(
        ao_i.exponents,
        ao_i.coefficients,
        ao_i.center,
        ao_j.exponents,
        ao_j.coefficients,
        ao_j.center,
        ao_k.exponents,
        ao_k.coefficients,
        ao_k.center,
        ao_l.exponents,
        ao_l.coefficients,
        ao_l.center,
    )


def eri_tensor(
    basis: CartesianBasis,
    *,
    engine: str = "auto",
) -> Array:
    """Full ERI tensor (ij|kl) in cartesian AO basis.

    The implementation exploits 8-fold permutation symmetry and supports an
    optional Schwarz screening threshold:

    screening_threshold:
        Skip quartets with `B_ij * B_kl < threshold`, where
        `B_ij = sqrt((ij|ij))`.
    """

    return eri_tensor_screened(basis, engine=engine)


def eri_pair_matrix_packed(
    basis: CartesianBasis,
    *,
    screening_threshold: float | None = None,
    schwarz_bounds: Array | None = None,
    engine: str = "auto",
) -> Array:
    """Packed AO-pair Coulomb matrix over lower-triangle AO pairs.

    Returns a symmetric matrix ``M`` with shape ``(npair, npair)`` where
    ``npair = nao * (nao + 1) // 2`` and
    ``M[pair(i,j), pair(k,l)] = (ij|kl)`` for ``i>=j, k>=l``.
    """

    n = basis.nao
    if n == 0:
        return jnp.zeros((0, 0))

    bounds = None
    if _can_use_shell_block_path(
        basis,
        screening_threshold=screening_threshold,
        engine=engine,
    ):
        return _eri_pair_matrix_packed_shell_block(basis, engine=engine)

    ao_pairs = _lower_triangle_pairs(n)
    npair = len(ao_pairs)
    if screening_threshold is not None:
        if schwarz_bounds is None:
            b_rows: list[int] = []
            b_cols: list[int] = []
            b_vals: list[Array] = []
            for i, j in ao_pairs:
                val = eri_element(basis, i, j, i, j, engine=engine)
                b = jnp.sqrt(jnp.maximum(val, 0.0))
                b_rows.append(i)
                b_cols.append(j)
                b_vals.append(b)
            rows = jnp.asarray(b_rows, dtype=jnp.int32)
            cols = jnp.asarray(b_cols, dtype=jnp.int32)
            vals = jnp.asarray(b_vals)
            bounds = jnp.zeros((n, n), dtype=vals.dtype)
            bounds = bounds.at[rows, cols].set(vals)
            bounds = bounds.at[cols, rows].set(vals)
        else:
            bounds = jnp.asarray(schwarz_bounds)

    threshold = None if screening_threshold is None else float(screening_threshold)
    angulars = basis.ao_angulars
    nprims = basis.ao_nprims_tuple

    if bounds is None and basis.quartet_groups:
        groups = basis.quartet_groups
    else:
        grouped_quartets: dict[tuple, dict[str, list[int]]] = {}
        grouped_rows: dict[tuple, list[int]] = {}
        grouped_cols: dict[tuple, list[int]] = {}
        for ij_pos, (i, j) in enumerate(ao_pairs):
            bij = None if bounds is None else bounds[i, j]
            for kl_pos in range(ij_pos + 1):
                k, l = ao_pairs[kl_pos]
                if bounds is not None:
                    bkl = bounds[k, l]
                    if float(bij * bkl) < threshold:
                        continue
                signature = (
                    angulars[i],
                    angulars[j],
                    angulars[k],
                    angulars[l],
                    nprims[i],
                    nprims[j],
                    nprims[k],
                    nprims[l],
                )
                bucket = grouped_quartets.setdefault(
                    signature,
                    {"i": [], "j": [], "k": [], "l": []},
                )
                bucket["i"].append(i)
                bucket["j"].append(j)
                bucket["k"].append(k)
                bucket["l"].append(l)
                grouped_rows.setdefault(signature, []).append(ij_pos)
                grouped_cols.setdefault(signature, []).append(kl_pos)
        groups = tuple(
            (
                signature,
                bucket["i"],
                bucket["j"],
                bucket["k"],
                bucket["l"],
                grouped_rows[signature],
                grouped_cols[signature],
            )
            for signature, bucket in grouped_quartets.items()
        )
    if not groups:
        return jnp.zeros((npair, npair))

    fused_pair = None
    if bounds is None and basis.quartet_groups:
        jit_groups = tuple(
            group for group in groups if _ao_signature_uses_jit(group.signature, engine)
        )
        if jit_groups:
            fused_pair = _fused_eri_pair_matrix_from_ao_groups(
                basis,
                jit_groups,
                engine=engine,
            )
            if len(jit_groups) == len(groups):
                return fused_pair
            groups = tuple(
                group for group in groups if not _ao_signature_uses_jit(group.signature, engine)
            )

    pair_index = np.full((n, n), -1, dtype=np.int32)
    for pos, (i, j) in enumerate(ao_pairs):
        pair_index[i, j] = pos
        pair_index[j, i] = pos

    row_chunks: list[Array] = []
    col_chunks: list[Array] = []
    vv_chunks: list[Array] = []
    for group in groups:
        if bounds is None and basis.quartet_groups:
            signature = group.signature
            idx_i = jnp.asarray(group.idx_i, dtype=jnp.int32)
            idx_j = jnp.asarray(group.idx_j, dtype=jnp.int32)
            idx_k = jnp.asarray(group.idx_k, dtype=jnp.int32)
            idx_l = jnp.asarray(group.idx_l, dtype=jnp.int32)
            rows = jnp.asarray(
                pair_index[np.asarray(group.idx_i), np.asarray(group.idx_j)],
                dtype=jnp.int32,
            )
            cols = jnp.asarray(
                pair_index[np.asarray(group.idx_k), np.asarray(group.idx_l)],
                dtype=jnp.int32,
            )
        else:
            signature, idx_i, idx_j, idx_k, idx_l, rows, cols = group
            idx_i = jnp.asarray(idx_i, dtype=jnp.int32)
            idx_j = jnp.asarray(idx_j, dtype=jnp.int32)
            idx_k = jnp.asarray(idx_k, dtype=jnp.int32)
            idx_l = jnp.asarray(idx_l, dtype=jnp.int32)
            rows = jnp.asarray(rows, dtype=jnp.int32)
            cols = jnp.asarray(cols, dtype=jnp.int32)
        row_chunks.append(rows)
        col_chunks.append(cols)
        use_jit = _use_jit_engine(
            engine,
            angulars=(signature[0], signature[1], signature[2], signature[3]),
            nprims=(signature[4], signature[5], signature[6], signature[7]),
        )
        if use_jit:
            kernel = _compiled_eri_pair_kernel_batched(*signature)
            vv_chunks.append(
                _run_quartet_kernel_chunked(
                    kernel,
                    _gather_quartet_batch(
                        basis,
                        idx_i,
                        idx_j,
                        idx_k,
                        idx_l,
                        nprim_i=signature[4],
                        nprim_j=signature[5],
                        nprim_k=signature[6],
                        nprim_l=signature[7],
                    ),
                )
            )
        else:
            vv_chunks.append(
                jnp.asarray(
                    [
                        eri_element(basis, i, j, k, l, engine=engine)
                        for i, j, k, l in zip(idx_i, idx_j, idx_k, idx_l)
                    ]
                )
            )

    rows = jnp.concatenate(row_chunks, axis=0)
    cols = jnp.concatenate(col_chunks, axis=0)
    vals = jnp.concatenate(vv_chunks, axis=0)
    pair = (
        fused_pair
        if fused_pair is not None
        else jnp.zeros((npair, npair), dtype=vals.dtype)
    )
    pair = pair.at[rows, cols].set(vals)
    pair = pair.at[cols, rows].set(vals)
    return 0.5 * (pair + pair.T)


def precompile_eri_kernels(
    basis: CartesianBasis,
    *,
    engine: str = "auto",
    chunk_size: int = QUARTET_BATCH_CHUNK,
) -> dict[str, int]:
    """Precompile batched ERI kernels for the quartet signatures present in ``basis``."""

    if basis.nao == 0:
        return {"compiled_pair_signatures": 0, "compiled_batch_shapes": 0}

    compiled: set[tuple] = set()
    n_signatures = 0
    base_chunk = max(int(chunk_size), 1)
    n_devices = jax.local_device_count()
    def _pad_sample_to_fixed_size(sample: tuple[Array, ...], size: int) -> tuple[Array, ...]:
        cur = int(sample[0].shape[0])
        if cur >= size:
            return sample
        pad = size - cur
        return tuple(
            jnp.concatenate(
                (arr, jnp.repeat(arr[cur - 1 : cur], pad, axis=0)),
                axis=0,
            )
            for arr in sample
        )

    if _can_use_shell_block_path(
        basis,
        screening_threshold=None,
        engine=engine,
    ):
        for group in basis.shell_quartet_groups:
            signature = group.signature
            n_signatures += 1
            n_items = int(group.idx_i.shape[0])
            if n_items == 0:
                continue
            kernel = _compiled_eri_shell_block_kernel_batched(*signature)
            key = signature
            if key in compiled:
                continue
            logical_chunk = min(n_items, base_chunk)
            target_size = (
                ((logical_chunk + n_devices - 1) // n_devices) * n_devices
                if n_devices > 1
                else logical_chunk
            )
            take = logical_chunk
            sample = _gather_shell_quartet_batch(
                basis,
                group.idx_i[:take],
                group.idx_j[:take],
                group.idx_k[:take],
                group.idx_l[:take],
                nprim_i=signature[4],
                nprim_j=signature[5],
                nprim_k=signature[6],
                nprim_l=signature[7],
            )
            sample = _pad_sample_to_fixed_size(sample, target_size)
            kernel.lower(*sample).compile()
            compiled.add(key)
        return {
            "compiled_shell_signatures": n_signatures,
            "compiled_batch_shapes": len(compiled),
        }

    for group in basis.quartet_groups:
        signature = group.signature
        if not _use_jit_engine(
            engine,
            angulars=(signature[0], signature[1], signature[2], signature[3]),
            nprims=(signature[4], signature[5], signature[6], signature[7]),
        ):
            continue
        n_signatures += 1
        n_items = int(group.idx_i.shape[0])
        if n_items == 0:
            continue
        kernel = _compiled_eri_pair_kernel_batched(*signature)
        key = signature
        if key in compiled:
            continue
        logical_chunk = min(n_items, base_chunk)
        target_size = (
            ((logical_chunk + n_devices - 1) // n_devices) * n_devices
            if n_devices > 1
            else logical_chunk
        )
        take = logical_chunk
        sample = _gather_quartet_batch(
            basis,
            group.idx_i[:take],
            group.idx_j[:take],
            group.idx_k[:take],
            group.idx_l[:take],
            nprim_i=signature[4],
            nprim_j=signature[5],
            nprim_k=signature[6],
            nprim_l=signature[7],
        )
        sample = _pad_sample_to_fixed_size(sample, target_size)
        kernel.lower(*sample).compile()
        compiled.add(key)
    return {
        "compiled_pair_signatures": n_signatures,
        "compiled_batch_shapes": len(compiled),
    }


def eri_tensor_screened(
    basis: CartesianBasis,
    *,
    screening_threshold: float | None = None,
    schwarz_bounds: Array | None = None,
    engine: str = "auto",
) -> Array:
    """Full ERI tensor with optional Schwarz screening."""

    n = basis.nao
    if n == 0:
        return jnp.zeros((0, 0, 0, 0))
    if _can_use_shell_block_path(
        basis,
        screening_threshold=screening_threshold,
        engine=engine,
    ):
        return _eri_tensor_shell_block(basis, engine=engine)
    pair = eri_pair_matrix_packed(
        basis,
        screening_threshold=screening_threshold,
        schwarz_bounds=schwarz_bounds,
        engine=engine,
    )
    pair_index = np.full((n, n), -1, dtype=np.int32)
    for pos, (i, j) in enumerate(_lower_triangle_pairs(n)):
        pair_index[i, j] = pos
        pair_index[j, i] = pos
    pair_index_arr = jnp.asarray(pair_index, dtype=jnp.int32)
    return pair[
        pair_index_arr[:, :, None, None],
        pair_index_arr[None, None, :, :],
    ]
