from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

from ..basis import CartesianBasis
from .two_electron import eri_element


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
