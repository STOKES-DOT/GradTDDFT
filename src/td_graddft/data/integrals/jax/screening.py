from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...basis import CartesianBasis
from .two_electron import (
    _compiled_eri_shell_block_kernel_batched,
    _gather_shell_quartet_batch,
    _run_quartet_kernel_chunked,
    eri_element,
)


def schwarz_bounds(
    basis: CartesianBasis,
    *,
    engine: str = "auto",
) -> Array:
    """Schwarz inequality bounds B_ij = sqrt((ij|ij))."""

    n = basis.nao
    row_idx: list[int] = []
    col_idx: list[int] = []
    values: list[Array] = []
    for i in range(n):
        for j in range(i + 1):
            row_idx.append(i)
            col_idx.append(j)
            val = eri_element(basis, i, j, i, j, engine=engine)
            values.append(jnp.sqrt(jnp.maximum(val, 0.0)))
    if not values:
        return jnp.zeros((n, n))
    rows = jnp.asarray(row_idx, dtype=jnp.int32)
    cols = jnp.asarray(col_idx, dtype=jnp.int32)
    vals = jnp.asarray(values)
    bounds = jnp.zeros((n, n), dtype=vals.dtype)
    bounds = bounds.at[rows, cols].set(vals)
    bounds = bounds.at[cols, rows].set(vals)
    return bounds


def shell_pair_schwarz_bounds(basis: CartesianBasis) -> np.ndarray:
    """Return shell-pair Schwarz bounds sqrt(max |(ij|ij)|) for shell-quartet screening."""

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
