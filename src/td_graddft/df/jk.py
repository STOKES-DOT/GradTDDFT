from __future__ import annotations

from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array

from ..data.basis import CartesianBasis
from ..data.integrals import eri_pair_matrix_packed


def _packed_pair_indices(nao: int) -> tuple[np.ndarray, np.ndarray]:
    return np.tril_indices(int(nao))


def eri_pair_matrix_to_df_factors(
    pair_matrix: Array,
    *,
    nao: int,
    tol: float = 1e-10,
    max_rank: int | None = None,
    dtype: Array | None = None,
) -> Array:
    pair_np = np.asarray(pair_matrix, dtype=float)
    eigvals, eigvecs = np.linalg.eigh(0.5 * (pair_np + pair_np.T))
    keep = np.where(eigvals > float(tol))[0]
    target_dtype = jnp.asarray(pair_matrix if dtype is None else dtype).dtype
    if keep.size == 0:
        return jnp.zeros((0, int(nao), int(nao)), dtype=target_dtype)
    if max_rank is not None and int(max_rank) > 0:
        keep = keep[-int(max_rank) :]
    vals = eigvals[keep]
    vecs = eigvecs[:, keep]
    packed_factors = (vecs * np.sqrt(vals)[None, :]).T
    rows, cols = _packed_pair_indices(int(nao))
    factors = np.zeros((packed_factors.shape[0], int(nao), int(nao)), dtype=packed_factors.dtype)
    factors[:, rows, cols] = packed_factors
    factors[:, cols, rows] = packed_factors
    return jnp.asarray(factors, dtype=target_dtype)


def eri_pair_matrix_to_df_factors_traceable(
    pair_matrix: Array,
    *,
    nao: int,
    tol: float = 1e-10,
    max_rank: int | None = None,
) -> Array:
    """Traceable AO-pair Cholesky-like factorization for differentiable paths."""

    pair = jnp.asarray(pair_matrix)
    sym_pair = 0.5 * (pair + pair.T)
    eigvals, eigvecs = jnp.linalg.eigh(sym_pair)
    if max_rank is not None and int(max_rank) > 0:
        eigvals = eigvals[-int(max_rank) :]
        eigvecs = eigvecs[:, -int(max_rank) :]
    threshold = jnp.asarray(tol, dtype=eigvals.dtype)
    keep = eigvals > threshold
    scales = jnp.where(
        keep,
        jnp.sqrt(jnp.maximum(eigvals, threshold)),
        jnp.asarray(0.0, dtype=eigvals.dtype),
    )
    packed_factors = (eigvecs * scales[None, :]).T
    rows_np, cols_np = _packed_pair_indices(int(nao))
    rows = jnp.asarray(rows_np)
    cols = jnp.asarray(cols_np)
    factors = jnp.zeros((packed_factors.shape[0], int(nao), int(nao)), dtype=pair.dtype)
    factors = factors.at[:, rows, cols].set(packed_factors)
    factors = factors.at[:, cols, rows].set(packed_factors)
    return factors


def eri_to_df_factors(
    eri: Array,
    *,
    tol: float = 1e-10,
    max_rank: int | None = None,
) -> Array:
    """Build a symmetric DF/Cholesky-like factorization from a full AO ERI tensor.

    The factorization is performed in the AO-pair space:
    (pq|rs) ~= sum_Q B_Q[p,q] B_Q[r,s]
    """

    eri_np = np.asarray(eri, dtype=float)
    nao = int(eri_np.shape[0])
    rows, cols = _packed_pair_indices(nao)
    pair = eri_np[rows, cols][:, rows, cols]
    return eri_pair_matrix_to_df_factors(
        pair,
        nao=nao,
        tol=tol,
        max_rank=max_rank,
        dtype=jnp.asarray(eri),
    )


def eri_to_df_factors_from_basis(
    basis: CartesianBasis,
    *,
    tol: float = 1e-10,
    max_rank: int | None = None,
    engine: str = "auto",
) -> Array:
    pair = eri_pair_matrix_packed(basis, engine=engine)
    return eri_pair_matrix_to_df_factors(
        pair,
        nao=basis.nao,
        tol=tol,
        max_rank=max_rank,
        dtype=pair,
    )


def true_df_factors_from_pyscf_mol(
    mol,
    *,
    auxbasis=None,
) -> Array:
    """Build true density-fitting factors from PySCF/libcint 3c2e/2c2e integrals.

    Returns factors in the existing TD-GradDFT layout:
    ``df_factors[Q, p, q]`` such that
    ``(pq|rs) ~= sum_Q df_factors[Q,p,q] * df_factors[Q,r,s]``.
    """

    try:
        from pyscf import df
    except ModuleNotFoundError as exc:
        raise ImportError(
            "PySCF is required to build true DF factors from a Mole."
        ) from exc

    nao = int(mol.nao_nr())
    cderi = np.asarray(
        df.incore.cholesky_eri(
            mol,
            auxbasis=auxbasis,
            aosym="s1",
        ),
        dtype=float,
    )
    if cderi.ndim != 2:
        raise RuntimeError(
            f"Unexpected cholesky_eri output rank {cderi.ndim}; expected 2."
        )
    if cderi.shape[1] != nao * nao:
        raise RuntimeError(
            "Unexpected cholesky_eri output shape "
            f"{cderi.shape}; expected trailing dimension {nao * nao}."
        )
    return jnp.asarray(cderi.reshape(cderi.shape[0], nao, nao))


def df_factors_to_mo_eri_slices(
    df_factors: Array,
    mo_coeff: Array,
    nocc: int,
    *,
    include_oovv: bool = True,
) -> tuple[Array, Array, Array | None]:
    coeff = jnp.asarray(mo_coeff)
    nocc_int = int(nocc)
    orbo = coeff[:, :nocc_int]
    orbv = coeff[:, nocc_int:]
    factors = jnp.asarray(df_factors)
    b_ov = jnp.einsum("Qpq,pi,qa->Qia", factors, orbo, orbv, precision=Precision.HIGHEST)
    b_vo = jnp.einsum("Qpq,pa,qi->Qai", factors, orbv, orbo, precision=Precision.HIGHEST)
    eri_ovov = jnp.einsum("Qia,Qjb->iajb", b_ov, b_ov, precision=Precision.HIGHEST)
    eri_ovvo = jnp.einsum("Qia,Qbj->iabj", b_ov, b_vo, precision=Precision.HIGHEST)
    if not include_oovv:
        return eri_ovov, eri_ovvo, None
    b_oo = jnp.einsum("Qpq,pi,qj->Qij", factors, orbo, orbo, precision=Precision.HIGHEST)
    b_vv = jnp.einsum("Qpq,pa,qb->Qab", factors, orbv, orbv, precision=Precision.HIGHEST)
    eri_oovv = jnp.einsum("Qij,Qab->ijab", b_oo, b_vv, precision=Precision.HIGHEST)
    return eri_ovov, eri_ovvo, eri_oovv


@jax.jit
def build_jk_from_df(df_factors: Array, density: Array) -> tuple[Array, Array]:
    """Build Coulomb and exchange matrices from DF factors.

    Uses a vmapped contraction over auxiliary channels so the J/K path stays in
    pure JAX and remains jit-friendly on CPU/GPU.
    """

    factors = jnp.asarray(df_factors)
    density = jnp.asarray(density)
    if factors.shape[0] == 0:
        zeros = jnp.zeros_like(density)
        return zeros, zeros

    projected = jnp.matmul(factors, density, precision=Precision.HIGHEST)
    rho_aux = jnp.einsum("Qpq,pq->Q", factors, density, precision=Precision.HIGHEST)
    j_mat = jnp.einsum(
        "Q,Qpq->pq",
        rho_aux,
        factors,
        precision=Precision.HIGHEST,
    )
    k_mat = jnp.einsum(
        "Qps,Qqs->pq",
        projected,
        factors,
        precision=Precision.HIGHEST,
    )
    return 0.5 * (j_mat + j_mat.T), 0.5 * (k_mat + k_mat.T)


@partial(jax.jit, static_argnames=("nocc",))
def build_jk_from_df_orbitals(
    df_factors: Array,
    density: Array,
    mo_coeff: Array,
    mo_occ: Array,
    *,
    nocc: int,
) -> tuple[Array, Array]:
    """Build DF J/K using occupied orbitals for the exchange contraction.

    For a density matrix assembled from orbitals, the K contraction can use
    ``B_Q C_occ`` instead of ``B_Q D``. This follows PySCF's DF-HF path and
    reduces the large exchange intermediate from ``naux * nao * nao`` to
    ``naux * nao * nocc`` for closed-shell RKS.
    """

    factors = jnp.asarray(df_factors)
    density = jnp.asarray(density)
    coeff = jnp.asarray(mo_coeff, dtype=factors.dtype)
    occ = jnp.asarray(mo_occ, dtype=factors.dtype)
    nocc_int = int(nocc)
    if factors.shape[0] == 0:
        zeros = jnp.zeros_like(density)
        return zeros, zeros
    if nocc_int <= 0 or nocc_int > coeff.shape[-1]:
        raise ValueError("nocc must be in the range [1, nmo].")

    rho_aux = jnp.einsum("Qpq,pq->Q", factors, density, precision=Precision.HIGHEST)
    j_mat = jnp.einsum(
        "Q,Qpq->pq",
        rho_aux,
        factors,
        precision=Precision.HIGHEST,
    )
    coeff_occ = coeff[:, :nocc_int]
    occ_scale = jnp.sqrt(jnp.maximum(occ[:nocc_int], 0.0))
    weighted_occ = coeff_occ * occ_scale[None, :]
    b_occ = jnp.einsum(
        "Qpr,ri->Qpi",
        factors,
        weighted_occ,
        precision=Precision.HIGHEST,
    )
    k_mat = jnp.einsum(
        "Qpi,Qqi->pq",
        b_occ,
        b_occ,
        precision=Precision.HIGHEST,
    )
    return 0.5 * (j_mat + j_mat.T), 0.5 * (k_mat + k_mat.T)


@jax.jit
def build_j_from_df(df_factors: Array, density: Array) -> Array:
    """Build Coulomb matrix only from DF factors.

    This mirrors the PySCF `with_j=True, with_k=False` path for pure
    semilocal functionals where HF exchange is absent.
    """

    factors = jnp.asarray(df_factors)
    density = jnp.asarray(density)
    if factors.shape[0] == 0:
        return jnp.zeros_like(density)

    rho_aux = jnp.einsum("Qpq,pq->Q", factors, density, precision=Precision.HIGHEST)
    j_mat = jnp.einsum(
        "Q,Qpq->pq",
        rho_aux,
        factors,
        precision=Precision.HIGHEST,
    )
    return 0.5 * (j_mat + j_mat.T)
