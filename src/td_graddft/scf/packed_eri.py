from __future__ import annotations

from functools import lru_cache, partial
import os

import numpy as np

import jax
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array


_PACKED_ERI_JK_BACKEND_ENV = "TD_GRADDFT_PACKED_ERI_JK_BACKEND"
_CUDA_PAIR_BACKENDS = {"cuda", "cuda_pair", "pair_cuda"}


@lru_cache(maxsize=64)
def _pair_metadata(nao: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.tril_indices(int(nao))
    pair_index = np.empty((int(nao), int(nao)), dtype=np.int32)
    pair_ids = np.arange(rows.size, dtype=np.int32)
    pair_index[rows, cols] = pair_ids
    pair_index[cols, rows] = pair_ids
    multiplicity = np.where(rows == cols, 1.0, 2.0)
    return (
        rows.astype(np.int32),
        cols.astype(np.int32),
        pair_index,
        multiplicity.astype(np.float64),
    )


def _metadata_arrays(nao: int, dtype) -> tuple[Array, Array, Array, Array]:
    rows, cols, pair_index, multiplicity = _pair_metadata(int(nao))
    return (
        jnp.asarray(rows, dtype=jnp.int32),
        jnp.asarray(cols, dtype=jnp.int32),
        jnp.asarray(pair_index, dtype=jnp.int32),
        jnp.asarray(multiplicity, dtype=dtype),
    )


def _packed_eri_jk_backend() -> str:
    raw = os.environ.get(_PACKED_ERI_JK_BACKEND_ENV, "").strip().lower()
    return raw or "jax"


@jax.jit
def build_j_from_eri_pair_matrix(eri_pair_matrix: Array, density: Array) -> Array:
    """Build exact Coulomb matrix from an AO-pair packed ERI matrix."""

    pair = jnp.asarray(eri_pair_matrix)
    density = 0.5 * (jnp.asarray(density) + jnp.asarray(density).T)
    nao = int(density.shape[0])
    rows, cols, _, multiplicity = _metadata_arrays(nao, density.dtype)
    density_pair = density[rows, cols] * multiplicity
    j_pair = pair @ density_pair
    j_mat = jnp.zeros_like(density)
    j_mat = j_mat.at[rows, cols].set(j_pair)
    j_mat = j_mat.at[cols, rows].set(j_pair)
    return 0.5 * (j_mat + j_mat.T)


@jax.jit
def build_jk_from_eri_pair_matrix(eri_pair_matrix: Array, density: Array) -> tuple[Array, Array]:
    """Build exact Coulomb and exchange matrices from packed no-DF ERIs.

    ``eri_pair_matrix`` follows PySCF's ``aosym='s4'`` lower-triangle AO-pair
    layout: ``M[pair(p,q), pair(r,s)] = (pq|rs)``.
    """

    pair = jnp.asarray(eri_pair_matrix)
    density = 0.5 * (jnp.asarray(density) + jnp.asarray(density).T)
    nao = int(density.shape[0])
    rows, cols, pair_index, multiplicity = _metadata_arrays(nao, density.dtype)

    backend = _packed_eri_jk_backend()
    if backend in _CUDA_PAIR_BACKENDS or backend == "auto":
        try:
            from .cuda_direct_jk import build_jk_from_eri_pair_matrix_cuda

            j_mat, k_mat = build_jk_from_eri_pair_matrix_cuda(
                pair,
                density,
                rows,
                cols,
            )
            return j_mat, k_mat
        except Exception as exc:
            if backend in _CUDA_PAIR_BACKENDS:
                raise RuntimeError(
                    "TD_GRADDFT_PACKED_ERI_JK_BACKEND requests CUDA, but the "
                    "packed AO-pair J/K FFI path could not be initialized."
                ) from exc

    density_pair = density[rows, cols] * multiplicity
    j_pair = pair @ density_pair
    j_mat = jnp.zeros_like(density)
    j_mat = j_mat.at[rows, cols].set(j_pair)
    j_mat = j_mat.at[cols, rows].set(j_pair)

    ao = jnp.arange(nao, dtype=jnp.int32)
    qs_by_q = pair_index[:, ao]

    def _k_row(p: Array) -> Array:
        pr = pair_index[p, ao]
        blocks = pair[pr[None, :, None], qs_by_q[:, None, :]]
        return jnp.einsum("qrs,rs->q", blocks, density, precision=Precision.HIGHEST)

    k_mat = jax.vmap(_k_row)(ao)
    return 0.5 * (j_mat + j_mat.T), 0.5 * (k_mat + k_mat.T)


def _mo_pair_products(left: Array, right: Array, rows: Array, cols: Array) -> Array:
    left_rows = left[rows]
    left_cols = left[cols]
    right_rows = right[rows]
    right_cols = right[cols]
    products = jnp.einsum(
        "Pi,Pa->iaP",
        left_rows,
        right_cols,
        precision=Precision.HIGHEST,
    )
    swapped = jnp.einsum(
        "Pi,Pa->iaP",
        left_cols,
        right_rows,
        precision=Precision.HIGHEST,
    )
    offdiag = (rows != cols).astype(products.dtype)
    return products + swapped * offdiag[None, None, :]


@partial(jax.jit, static_argnames=("nocc", "include_oovv"))
def eri_pair_matrix_to_mo_eri_slices(
    eri_pair_matrix: Array,
    mo_coeff: Array,
    *,
    nocc: int,
    include_oovv: bool = True,
) -> tuple[Array, Array, Array | None]:
    """Transform packed no-DF AO ERIs into restricted TDDFT MO slices."""

    pair = jnp.asarray(eri_pair_matrix)
    coeff = jnp.asarray(mo_coeff)
    nocc_int = int(nocc)
    rows, cols, _, _ = _metadata_arrays(int(coeff.shape[0]), coeff.dtype)
    orbo = coeff[:, :nocc_int]
    orbv = coeff[:, nocc_int:]
    ov = _mo_pair_products(orbo, orbv, rows, cols)
    vo = _mo_pair_products(orbv, orbo, rows, cols)
    eri_ovov = jnp.einsum("iaP,PQ,jbQ->iajb", ov, pair, ov, precision=Precision.HIGHEST)
    eri_ovvo = jnp.einsum("iaP,PQ,bjQ->iabj", ov, pair, vo, precision=Precision.HIGHEST)
    if not include_oovv:
        return eri_ovov, eri_ovvo, None
    oo = _mo_pair_products(orbo, orbo, rows, cols)
    vv = _mo_pair_products(orbv, orbv, rows, cols)
    eri_oovv = jnp.einsum("ijP,PQ,abQ->ijab", oo, pair, vv, precision=Precision.HIGHEST)
    return eri_ovov, eri_ovvo, eri_oovv
