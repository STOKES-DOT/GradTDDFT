from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import jax.numpy as jnp
from jaxtyping import Array

from ..data.basis import CartesianBasis
from ..data.integrals.two_electron import (
    _compiled_eri_shell_block_kernel_batched,
    eri_pair_matrix_packed,
    _gather_shell_quartet_batch,
    _run_quartet_kernel_chunked,
)
from .packed_eri import build_jk_from_eri_pair_matrix


_DIRECT_PACKED_JK_MAX_NAO = 64


@dataclass(frozen=True)
class DirectJKResult:
    j: Array
    k: Array


def shell_pair_schwarz_bounds(basis: CartesianBasis) -> np.ndarray:
    """Return shell-pair Schwarz bounds sqrt(max |(ij|ij)|) for screening."""

    nshell = len(basis.shells)
    bounds = np.zeros((nshell, nshell), dtype=float)
    if nshell == 0:
        return bounds
    for i in range(nshell):
        for j in range(i + 1):
            shell_i = basis.shells[i]
            shell_j = basis.shells[j]
            signature = (
                shell_i.angulars,
                shell_j.angulars,
                shell_i.angulars,
                shell_j.angulars,
                basis.shell_nprims_tuple[i],
                basis.shell_nprims_tuple[j],
                basis.shell_nprims_tuple[i],
                basis.shell_nprims_tuple[j],
            )
            kernel = _compiled_eri_shell_block_kernel_batched(*signature)
            blocks = _run_quartet_kernel_chunked(
                kernel,
                _gather_shell_quartet_batch(
                    basis,
                    jnp.asarray([i], dtype=jnp.int32),
                    jnp.asarray([j], dtype=jnp.int32),
                    jnp.asarray([i], dtype=jnp.int32),
                    jnp.asarray([j], dtype=jnp.int32),
                    nprim_i=signature[4],
                    nprim_j=signature[5],
                    nprim_k=signature[6],
                    nprim_l=signature[7],
                ),
            )
            value = float(np.max(np.abs(np.asarray(blocks)))) if blocks.size else 0.0
            bound = float(np.sqrt(max(value, 0.0)))
            bounds[i, j] = bound
            bounds[j, i] = bound
    return bounds


_compute_shell_pair_schwarz_bounds = shell_pair_schwarz_bounds


@lru_cache(maxsize=64)
def _unique_scatter_indices(
    nao: int,
    scatter_i: tuple[int, ...],
    scatter_j: tuple[int, ...],
    scatter_k: tuple[int, ...],
    scatter_l: tuple[int, ...],
) -> np.ndarray:
    i = np.asarray(scatter_i, dtype=np.int64)
    j = np.asarray(scatter_j, dtype=np.int64)
    k = np.asarray(scatter_k, dtype=np.int64)
    l = np.asarray(scatter_l, dtype=np.int64)
    flat = (((i * int(nao) + j) * int(nao) + k) * int(nao) + l)
    _, first = np.unique(flat, return_index=True)
    return np.sort(first).astype(np.int32)


def _symmetrized_block_values(blocks: Array) -> Array:
    return jnp.concatenate(
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


def _screening_active_mask(
    group,
    *,
    screening_threshold: float | None,
    bounds: np.ndarray | None,
) -> np.ndarray | None:
    if screening_threshold is None or screening_threshold <= 0.0:
        return None
    if bounds is None:
        raise ValueError("Shell-pair Schwarz bounds are required when screening_threshold > 0.")
    idx_i = np.asarray(group.idx_i, dtype=np.int32)
    idx_j = np.asarray(group.idx_j, dtype=np.int32)
    idx_k = np.asarray(group.idx_k, dtype=np.int32)
    idx_l = np.asarray(group.idx_l, dtype=np.int32)
    estimates = bounds[idx_i, idx_j] * bounds[idx_k, idx_l]
    return estimates >= float(screening_threshold)


def _filter_scatter_by_active(
    scatter: Array,
    *,
    n_items: int,
    block_size: int,
    active: np.ndarray | None,
) -> Array:
    return jnp.asarray(
        _filter_scatter_by_active_np(
            scatter,
            n_items=n_items,
            block_size=block_size,
            active=active,
        ),
        dtype=jnp.int32,
    )


def _filter_scatter_by_active_np(
    scatter: Array,
    *,
    n_items: int,
    block_size: int,
    active: np.ndarray | None,
) -> np.ndarray:
    arr = np.asarray(scatter, dtype=np.int32)
    if active is None:
        return arr.reshape(-1)
    arr = arr.reshape(8, n_items, block_size)
    return arr[:, active, :].reshape(-1)


def build_direct_jk_from_basis(
    basis: CartesianBasis,
    density: Array,
    *,
    screening_threshold: float | None = None,
    shell_pair_schwarz_bounds: Array | np.ndarray | None = None,
) -> DirectJKResult:
    """Build exact no-DF J/K by generating shell-quartet ERIs on demand."""

    density_arr = 0.5 * (jnp.asarray(density) + jnp.asarray(density).T)
    nao = int(density_arr.shape[0])
    if nao == 0:
        zeros = jnp.zeros_like(density_arr)
        return DirectJKResult(j=zeros, k=zeros)
    if not basis.shell_quartet_groups:
        raise ValueError("Direct J/K requires basis.shell_quartet_groups.")

    threshold = None if screening_threshold is None else float(screening_threshold)
    if (threshold is None or threshold <= 0.0) and nao <= _DIRECT_PACKED_JK_MAX_NAO:
        pair = eri_pair_matrix_packed(basis)
        j_mat, k_mat = build_jk_from_eri_pair_matrix(pair, density_arr)
        return DirectJKResult(j=j_mat, k=k_mat)

    j_mat = jnp.zeros_like(density_arr)
    k_mat = jnp.zeros_like(density_arr)
    bounds = None
    if threshold is not None and threshold > 0.0:
        bounds = (
            shell_pair_schwarz_bounds
            if shell_pair_schwarz_bounds is not None
            else _compute_shell_pair_schwarz_bounds(basis)
        )
        bounds = np.asarray(bounds, dtype=float)

    for group in basis.shell_quartet_groups:
        signature = group.signature
        n_items = int(group.idx_i.shape[0])
        block_size = len(signature[0]) * len(signature[1]) * len(signature[2]) * len(signature[3])
        active = _screening_active_mask(
            group,
            screening_threshold=threshold,
            bounds=bounds,
        )
        if active is not None:
            if not bool(np.any(active)):
                continue
            idx_i = jnp.asarray(np.asarray(group.idx_i, dtype=np.int32)[active], dtype=jnp.int32)
            idx_j = jnp.asarray(np.asarray(group.idx_j, dtype=np.int32)[active], dtype=jnp.int32)
            idx_k = jnp.asarray(np.asarray(group.idx_k, dtype=np.int32)[active], dtype=jnp.int32)
            idx_l = jnp.asarray(np.asarray(group.idx_l, dtype=np.int32)[active], dtype=jnp.int32)
        else:
            idx_i = group.idx_i
            idx_j = group.idx_j
            idx_k = group.idx_k
            idx_l = group.idx_l
        kernel = _compiled_eri_shell_block_kernel_batched(*signature)
        blocks = _run_quartet_kernel_chunked(
            kernel,
            _gather_shell_quartet_batch(
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
        vals = _symmetrized_block_values(blocks)

        scatter_i_np = _filter_scatter_by_active_np(
            group.scatter_i,
            n_items=n_items,
            block_size=block_size,
            active=active,
        )
        scatter_j_np = _filter_scatter_by_active_np(
            group.scatter_j,
            n_items=n_items,
            block_size=block_size,
            active=active,
        )
        scatter_k_np = _filter_scatter_by_active_np(
            group.scatter_k,
            n_items=n_items,
            block_size=block_size,
            active=active,
        )
        scatter_l_np = _filter_scatter_by_active_np(
            group.scatter_l,
            n_items=n_items,
            block_size=block_size,
            active=active,
        )
        unique_idx_np = _unique_scatter_indices(
            nao,
            tuple(scatter_i_np.tolist()),
            tuple(scatter_j_np.tolist()),
            tuple(scatter_k_np.tolist()),
            tuple(scatter_l_np.tolist()),
        )
        unique_idx = jnp.asarray(unique_idx_np, dtype=jnp.int32)
        scatter_i = jnp.asarray(scatter_i_np, dtype=jnp.int32)[unique_idx]
        scatter_j = jnp.asarray(scatter_j_np, dtype=jnp.int32)[unique_idx]
        scatter_k = jnp.asarray(scatter_k_np, dtype=jnp.int32)[unique_idx]
        scatter_l = jnp.asarray(scatter_l_np, dtype=jnp.int32)[unique_idx]
        vals = vals[unique_idx]

        j_mat = j_mat.at[scatter_i, scatter_j].add(vals * density_arr[scatter_k, scatter_l])
        k_mat = k_mat.at[scatter_i, scatter_k].add(vals * density_arr[scatter_j, scatter_l])

    return DirectJKResult(
        j=0.5 * (j_mat + j_mat.T),
        k=0.5 * (k_mat + k_mat.T),
    )


def build_direct_jk_incremental(
    basis: CartesianBasis,
    density: Array,
    *,
    density_last: Array | None = None,
    j_last: Array | None = None,
    k_last: Array | None = None,
    screening_threshold: float | None = None,
    shell_pair_schwarz_bounds: Array | np.ndarray | None = None,
) -> DirectJKResult:
    """Build direct J/K with a PySCF-style density-difference update."""

    if density_last is None:
        return build_direct_jk_from_basis(
            basis,
            density,
            screening_threshold=screening_threshold,
            shell_pair_schwarz_bounds=shell_pair_schwarz_bounds,
        )
    if j_last is None or k_last is None:
        raise ValueError("j_last and k_last are required when density_last is provided.")
    delta = jnp.asarray(density) - jnp.asarray(density_last)
    delta_jk = build_direct_jk_from_basis(
        basis,
        delta,
        screening_threshold=screening_threshold,
        shell_pair_schwarz_bounds=shell_pair_schwarz_bounds,
    )
    return DirectJKResult(
        j=jnp.asarray(j_last) + delta_jk.j,
        k=jnp.asarray(k_last) + delta_jk.k,
    )
