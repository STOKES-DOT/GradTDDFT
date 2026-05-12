from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass, fields, replace
import os
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.lax import Precision
from jaxtyping import Array

from ..data.basis import CartesianBasis
from ..data.integrals import (
    CudaDirectJKBuilder,
    build_j_from_eri_pair_matrix,
    build_jk_from_eri_pair_matrix,
    build_direct_jk_incremental,
    cuda_ffi_available,
    eri_pair_matrix_packed,
    shell_pair_schwarz_bounds,
)
from ..data.integrals.jax.direct_jk import _DIRECT_PACKED_JK_MAX_NAO
from ..df import build_j_from_df, build_jk_from_df, build_jk_from_df_orbitals, eri_to_df_factors
from ..features import (
    MoleculeLikeState,
    molecule_grid_view,
    restricted_grid_features_with_gradients,
)
from ..jax_libxc import (
    eval_xc_energy_density,
    hybrid_coeff,
    parse_xc,
    restricted_feature_bundle_from_rho_grad_tau,
    xc_type,
)
from .core import _build_density_closed_shell as _build_density
from .core import _build_density_from_occ, _diagonalize_fock, _orthogonalizer

_PYSCF_LIKE_DIIS_START_CYCLE = 2
_PYSCF_LIKE_DIIS_SPACE = 8
_DEFAULT_CUDA_FULL_ERI_MAX_MIB = 2048.0
_DEFAULT_CUDA_PAIR_ERI_MAX_MIB = 2048.0
_RUNTIME_RKS_PREWARM_CACHE_MAXSIZE = 64
_RUNTIME_RKS_PREWARMED: set[tuple[Any, ...]] = set()
JKBuilder = Callable[
    [Array, Array | None, Array | None, Array | None, Array | None, Array | None],
    tuple[Array, Array],
]


def _pytree_dataclass(cls):
    def tree_flatten(self):
        children = tuple(getattr(self, field.name) for field in fields(self))
        return children, None

    @classmethod
    def tree_unflatten(cls_, aux_data, children):
        del aux_data
        return cls_(*children)

    cls.tree_flatten = tree_flatten
    cls.tree_unflatten = tree_unflatten
    return jax.tree_util.register_pytree_node_class(cls)


@dataclass(frozen=True)
class RKSConfig:
    """Configuration for restricted Kohn-Sham SCF iterations."""

    xc_spec: str = "pbe"
    max_cycle: int = 80
    conv_tol: float = 1e-10
    conv_tol_density: float = 1e-8
    damping: float = 0.0
    level_shift: float = 0.0
    orthogonalization_eps: float = 1e-10
    density_floor: float = 1e-12
    potential_clip: float | None = None
    iteration_backend: Literal["runtime", "lax"] = "runtime"
    jk_backend: Literal["full", "df", "direct"] = "full"
    direct_jk_engine: Literal["jax", "cuda"] = "jax"
    df_tol: float = 1e-10
    df_max_rank: int | None = None
    direct_scf_tol: float = 0.0
    direct_scf_incremental: bool = True


@dataclass(frozen=True)
class RKSResult:
    """Restricted Kohn-Sham result object."""

    converged: bool
    total_energy: float
    electronic_energy: float
    nuclear_repulsion: float
    xc_energy: float
    exact_exchange_fraction: float
    mo_energy: Array
    mo_coeff: Array
    mo_occ: Array
    density_matrix: Array
    fock_matrix: Array
    overlap_matrix: Array
    hcore_matrix: Array
    cycles: int


def _cuda_full_eri_max_bytes() -> int:
    raw = os.environ.get("TD_GRADDFT_CUDA_FULL_ERI_MAX_MIB")
    if raw is None or str(raw).strip() == "":
        return int(_DEFAULT_CUDA_FULL_ERI_MAX_MIB * 1024.0 * 1024.0)
    try:
        mib = float(raw)
    except ValueError:
        return int(_DEFAULT_CUDA_FULL_ERI_MAX_MIB * 1024.0 * 1024.0)
    return max(0, int(mib * 1024.0 * 1024.0))


def _cuda_pair_eri_max_bytes() -> int:
    raw = os.environ.get("TD_GRADDFT_CUDA_PAIR_ERI_MAX_MIB")
    if raw is None or str(raw).strip() == "":
        return int(_DEFAULT_CUDA_PAIR_ERI_MAX_MIB * 1024.0 * 1024.0)
    try:
        mib = float(raw)
    except ValueError:
        return int(_DEFAULT_CUDA_PAIR_ERI_MAX_MIB * 1024.0 * 1024.0)
    return max(0, int(mib * 1024.0 * 1024.0))


def _should_cache_cuda_full_eri(basis: CartesianBasis, cfg: RKSConfig) -> bool:
    if float(cfg.direct_scf_tol) > 0.0:
        return False
    nao = int(basis.nao)
    estimated_bytes = nao**4 * 8
    return estimated_bytes <= _cuda_full_eri_max_bytes()


def _should_cache_cuda_pair_eri(basis: CartesianBasis, cfg: RKSConfig) -> bool:
    if float(cfg.direct_scf_tol) > 0.0:
        return False
    nao = int(basis.nao)
    npair = nao * (nao + 1) // 2
    estimated_bytes = npair**2 * 8
    return estimated_bytes <= _cuda_pair_eri_max_bytes()


@_pytree_dataclass
@dataclass(frozen=True)
class TraceableRKSResult:
    """Traceable restricted Kohn-Sham result (Array-valued scalars for autodiff)."""

    converged: Array
    total_energy: Array
    electronic_energy: Array
    nuclear_repulsion: Array
    xc_energy: Array
    exact_exchange_fraction: Array
    mo_energy: Array
    mo_coeff: Array
    mo_occ: Array
    density_matrix: Array
    fock_matrix: Array
    overlap_matrix: Array
    hcore_matrix: Array
    cycles: Array


@_pytree_dataclass
@dataclass(frozen=True)
class RKSIterationCarry:
    cycle: Array
    converged: Array
    density: Array
    mo_coeff: Array
    mo_energy: Array
    energy: Array
    xc_energy: Array
    raw_fock: Array
    j_mat: Array
    k_mat: Array
    fock_last: Array
    fock_hist: Array
    err_hist: Array
    hist_head: Array
    hist_count: Array


def _runtime_rks_prewarm_cache_key(
    *,
    overlap: Array,
    ao: Array,
    cfg: RKSConfig,
    eri: Array | None,
    eri_pair_matrix: Array | None,
    df_factors: Array | None,
    direct_basis: CartesianBasis | None,
) -> tuple[Any, ...]:
    overlap_arr = jnp.asarray(overlap)
    ao_arr = jnp.asarray(ao)
    return (
        str(overlap_arr.dtype),
        tuple(int(dim) for dim in overlap_arr.shape),
        tuple(int(dim) for dim in ao_arr.shape),
        str(cfg.xc_spec),
        str(cfg.jk_backend),
        str(cfg.direct_jk_engine),
        bool(eri is not None),
        bool(eri_pair_matrix is not None),
        None if df_factors is None else tuple(int(dim) for dim in jnp.asarray(df_factors).shape),
        None if direct_basis is None else int(direct_basis.nao),
        float(cfg.direct_scf_tol),
        bool(cfg.direct_scf_incremental),
    )


def _closed_shell_mo_occ(nao: int, nocc: int, dtype: Array) -> Array:
    return jnp.zeros((nao,), dtype=dtype).at[:nocc].set(2.0)


def _defer_initial_fock_build(config: RKSConfig) -> bool:
    return (
        config.iteration_backend == "lax"
        and config.jk_backend == "direct"
        and config.direct_jk_engine == "cuda"
    )


def _use_host_initial_orbitals(config: RKSConfig) -> bool:
    return _defer_initial_fock_build(config)


def _numpy_dtype(dtype) -> np.dtype:
    return np.dtype(dtype)


def _host_device_array(value, dtype=None) -> Array:
    np_dtype = None if dtype is None else _numpy_dtype(dtype)
    return jax.device_put(np.asarray(value, dtype=np_dtype))


def _host_device_scalar(value, dtype) -> Array:
    return _host_device_array(value, dtype=dtype)


def _host_device_zeros(shape: tuple[int, ...], dtype) -> Array:
    return jax.device_put(np.zeros(tuple(int(dim) for dim in shape), dtype=_numpy_dtype(dtype)))


def _host_device_zeros_like(value: Array) -> Array:
    return _host_device_zeros(tuple(int(dim) for dim in value.shape), value.dtype)


def _orthogonalizer_host(overlap: Array, eps: float) -> np.ndarray:
    s_np = np.asarray(jax.device_get(overlap), dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(0.5 * (s_np + s_np.T))
    clipped = np.maximum(eigvals, float(eps))
    return eigvecs @ np.diag(clipped ** -0.5) @ eigvecs.T


def _diagonalize_fock_host(fock: Array, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fock_np = np.asarray(jax.device_get(fock), dtype=np.float64)
    f_ortho = x.T @ fock_np @ x
    mo_energy, coeff_ortho = np.linalg.eigh(0.5 * (f_ortho + f_ortho.T))
    return mo_energy, x @ coeff_ortho


def _initial_scf_orbitals_host(
    *,
    overlap: Array,
    hcore: Array,
    init_mo_coeff: Array | None,
    init_mo_occ: Array | None,
    init_mo_energy: Array | None,
    nelectron: int,
    orthogonalization_eps: float,
    dtype: Any,
) -> tuple[Array, Array, Array, Array, int, Array]:
    nao = int(np.asarray(overlap).shape[0])
    x_np = _orthogonalizer_host(overlap, orthogonalization_eps)
    if init_mo_coeff is None:
        mo_energy_np, mo_coeff_np = _diagonalize_fock_host(hcore, x_np)
    else:
        mo_coeff_np = np.asarray(jax.device_get(init_mo_coeff), dtype=np.float64)
        if mo_coeff_np.ndim == 3:
            mo_coeff_np = mo_coeff_np[0]
        if init_mo_energy is None:
            mo_energy_np, _ = _diagonalize_fock_host(hcore, x_np)
        else:
            mo_energy_np = np.asarray(jax.device_get(init_mo_energy), dtype=np.float64)
            if mo_energy_np.ndim == 2:
                mo_energy_np = mo_energy_np[0]
    if init_mo_occ is None:
        if nelectron % 2 != 0:
            raise ValueError("RKS requires an even number of electrons when init_mo_occ is not provided.")
        nocc = int(nelectron) // 2
        if nocc <= 0 or nocc > nao:
            raise ValueError("Invalid occupation count for RKS.")
        mo_occ_np = np.zeros((nao,), dtype=np.float64)
        mo_occ_np[:nocc] = 2.0
    else:
        mo_occ_np = np.asarray(jax.device_get(init_mo_occ), dtype=np.float64)
        if mo_occ_np.ndim == 2:
            mo_occ_np = mo_occ_np[0]
        if mo_occ_np.ndim != 1:
            raise ValueError("init_mo_occ must be a 1D occupation vector for RKS.")
        if float(np.max(mo_occ_np)) <= 1.0 + 1e-6:
            mo_occ_np = mo_occ_np * 2.0
        if int(mo_occ_np.shape[0]) != nao:
            raise ValueError("init_mo_occ must match the AO/MO dimension for RKS.")
        nocc = int(np.count_nonzero(mo_occ_np > 1e-12))
    density_np = np.einsum("pi,i,qi->pq", mo_coeff_np, mo_occ_np, mo_coeff_np)
    return (
        _host_device_array(x_np, dtype=dtype),
        _host_device_array(mo_energy_np, dtype=dtype),
        _host_device_array(mo_coeff_np, dtype=dtype),
        _host_device_array(mo_occ_np, dtype=dtype),
        nocc,
        _host_device_array(density_np, dtype=dtype),
    )


def _validate_initial_density(
    density: Array | None,
    *,
    nao: int,
    dtype: Any,
    label: str = "init_density",
) -> Array | None:
    if density is None:
        return None
    dm = jnp.asarray(density, dtype=dtype)
    if dm.ndim != 2 or int(dm.shape[0]) != nao or int(dm.shape[1]) != nao:
        raise ValueError(f"{label} must be a square ({nao}, {nao}) matrix for RKS.")
    return 0.5 * (dm + dm.T)


def _build_jk(eri: Array, density: Array) -> tuple[Array, Array]:
    j_mat = jnp.einsum("pqrs,rs->pq", eri, density, precision=Precision.HIGHEST)
    k_mat = jnp.einsum("prqs,rs->pq", eri, density, precision=Precision.HIGHEST)
    return j_mat, k_mat


def _build_j(eri: Array, density: Array) -> Array:
    return jnp.einsum("pqrs,rs->pq", eri, density, precision=Precision.HIGHEST)


def _commutator_error(fock: Array, density: Array, overlap: Array) -> Array:
    return fock @ density @ overlap - overlap @ density @ fock


def _orthonormal_diis_error(fock: Array, density: Array, overlap: Array, corth: Array) -> Array:
    error = _commutator_error(fock, density, overlap)
    if jnp.finfo(fock.dtype).bits < 64:
        return error
    return corth.T @ error @ corth


def _orbital_gradient_norm(fock: Array, mo_coeff: Array, nocc: int) -> Array:
    orbo = mo_coeff[:, :nocc]
    orbv = mo_coeff[:, nocc:]
    g = 2.0 * (orbv.T @ fock @ orbo)
    return jnp.linalg.norm(g)


def _scf_residual_norm(fock: Array, density: Array, overlap: Array) -> Array:
    return jnp.linalg.norm(_commutator_error(fock, density, overlap))


def _apply_fock_damping(fock: Array, fock_prev: Array, factor: Array) -> Array:
    return fock * (1.0 - factor) + fock_prev * factor


def _apply_level_shift(fock: Array, overlap: Array, density: Array, factor: Array) -> Array:
    dm_occ = 0.5 * density
    dm_vir = overlap - overlap @ dm_occ @ overlap
    return fock + dm_vir * factor


def _diis_push_lax(
    fock: Array,
    error: Array,
    fock_hist: Array,
    err_hist: Array,
    hist_head: Array,
    hist_count: Array,
) -> tuple[Array, Array, Array, Array]:
    fock_hist = fock_hist.at[hist_head].set(fock)
    err_hist = err_hist.at[hist_head].set(error.reshape(-1))
    hist_head = (hist_head + 1) % jnp.asarray(_PYSCF_LIKE_DIIS_SPACE, dtype=hist_head.dtype)
    hist_count = jnp.minimum(
        hist_count + jnp.asarray(1, dtype=hist_count.dtype),
        jnp.asarray(_PYSCF_LIKE_DIIS_SPACE, dtype=hist_count.dtype),
    )
    return fock_hist, err_hist, hist_head, hist_count


def _diis_extrapolate_lax(
    fock: Array,
    error: Array,
    fock_hist: Array,
    err_hist: Array,
    hist_head: Array,
    hist_count: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    fock_hist, err_hist, hist_head, hist_count = _diis_push_lax(
        fock,
        error,
        fock_hist,
        err_hist,
        hist_head,
        hist_count,
    )

    def _solve_diis(operand: tuple[Array, Array, Array]) -> Array:
        fock_hist_i, err_hist_i, hist_count_i = operand
        valid = (jnp.arange(_PYSCF_LIKE_DIIS_SPACE) < hist_count_i).astype(fock.dtype)
        gram = err_hist_i @ err_hist_i.T
        diag_reg = jnp.asarray(
            1e-14 if jnp.finfo(fock.dtype).bits < 64 else jnp.finfo(fock.dtype).eps * 50.0,
            dtype=fock.dtype,
        )
        top = gram * (valid[:, None] * valid[None, :])
        top = top + jnp.diag(valid * diag_reg + (1.0 - valid))
        b = jnp.zeros(
            (_PYSCF_LIKE_DIIS_SPACE + 1, _PYSCF_LIKE_DIIS_SPACE + 1),
            dtype=fock.dtype,
        )
        b = b.at[:_PYSCF_LIKE_DIIS_SPACE, :_PYSCF_LIKE_DIIS_SPACE].set(top)
        b = b.at[:_PYSCF_LIKE_DIIS_SPACE, _PYSCF_LIKE_DIIS_SPACE].set(-valid)
        b = b.at[_PYSCF_LIKE_DIIS_SPACE, :_PYSCF_LIKE_DIIS_SPACE].set(-valid)
        rhs = jnp.zeros((_PYSCF_LIKE_DIIS_SPACE + 1,), dtype=fock.dtype)
        rhs = rhs.at[_PYSCF_LIKE_DIIS_SPACE].set(-1.0)
        coeff = jnp.linalg.solve(b, rhs)[:_PYSCF_LIKE_DIIS_SPACE]
        coeff = coeff * valid
        return jnp.tensordot(coeff, fock_hist_i, axes=(0, 0))

    fock_eff = jax.lax.cond(
        hist_count >= _PYSCF_LIKE_DIIS_START_CYCLE,
        _solve_diis,
        lambda _: fock,
        operand=(fock_hist, err_hist, hist_count),
    )
    return fock_eff, fock_hist, err_hist, hist_head, hist_count


def _diis_push_runtime(
    fock: Array,
    error: Array,
    fock_hist: Array,
    err_hist: Array,
    hist_head: int,
    hist_count: int,
) -> tuple[Array, Array, int, int]:
    fock_hist = fock_hist.at[hist_head].set(fock)
    err_hist = err_hist.at[hist_head].set(error.reshape(-1))
    hist_head = (hist_head + 1) % _PYSCF_LIKE_DIIS_SPACE
    hist_count = min(hist_count + 1, _PYSCF_LIKE_DIIS_SPACE)
    return fock_hist, err_hist, hist_head, hist_count


def _diis_extrapolate_runtime(
    fock: Array,
    error: Array,
    fock_hist: Array,
    err_hist: Array,
    hist_head: int,
    hist_count: int,
) -> tuple[Array, Array, Array, int, int]:
    fock_hist, err_hist, hist_head, hist_count = _diis_push_runtime(
        fock,
        error,
        fock_hist,
        err_hist,
        hist_head,
        hist_count,
    )
    if hist_count < _PYSCF_LIKE_DIIS_START_CYCLE:
        return fock, fock_hist, err_hist, hist_head, hist_count

    valid = (jnp.arange(_PYSCF_LIKE_DIIS_SPACE) < hist_count).astype(fock.dtype)
    gram = err_hist @ err_hist.T
    diag_reg = jnp.asarray(
        1e-14 if jnp.finfo(fock.dtype).bits < 64 else jnp.finfo(fock.dtype).eps * 50.0,
        dtype=fock.dtype,
    )
    top = gram * (valid[:, None] * valid[None, :])
    top = top + jnp.diag(valid * diag_reg + (1.0 - valid))
    b = jnp.zeros(
        (_PYSCF_LIKE_DIIS_SPACE + 1, _PYSCF_LIKE_DIIS_SPACE + 1),
        dtype=fock.dtype,
    )
    b = b.at[:_PYSCF_LIKE_DIIS_SPACE, :_PYSCF_LIKE_DIIS_SPACE].set(top)
    b = b.at[:_PYSCF_LIKE_DIIS_SPACE, _PYSCF_LIKE_DIIS_SPACE].set(-valid)
    b = b.at[_PYSCF_LIKE_DIIS_SPACE, :_PYSCF_LIKE_DIIS_SPACE].set(-valid)
    rhs = jnp.zeros((_PYSCF_LIKE_DIIS_SPACE + 1,), dtype=fock.dtype)
    rhs = rhs.at[_PYSCF_LIKE_DIIS_SPACE].set(-1.0)
    coeff = jnp.linalg.solve(b, rhs)[:_PYSCF_LIKE_DIIS_SPACE]
    coeff = coeff * valid
    fock_eff = jnp.tensordot(coeff, fock_hist, axes=(0, 0))
    return fock_eff, fock_hist, err_hist, hist_head, hist_count


def _make_jk_builder(
    eri: Array | None,
    cfg: RKSConfig,
    *,
    eri_pair_matrix: Array | None = None,
    df_factors: Array | None = None,
    direct_basis: CartesianBasis | None = None,
    direct_joltqc_basis_data: Array | None = None,
    direct_cuda_jk_builder: Any | None = None,
    with_k: bool = True,
    nocc: int | None = None,
) -> JKBuilder:
    def _direct_builder() -> JKBuilder:
        if direct_basis is None:
            raise ValueError("jk_backend='direct' requires direct_basis.")

        def _accumulate_incremental_jk(
            eval_jk: Callable[[Array], tuple[Array, Array]],
            density: Array,
            *,
            density_last: Array | None,
            j_last: Array | None,
            k_last: Array | None,
            use_incremental: bool,
        ) -> tuple[Array, Array]:
            if use_incremental and density_last is not None:
                if j_last is None or k_last is None:
                    raise ValueError(
                        "j_last and k_last are required when density_last is provided."
                    )
                delta = jnp.asarray(density) - jnp.asarray(density_last)
                delta_j, delta_k = eval_jk(delta)
                return jnp.asarray(j_last) + delta_j, jnp.asarray(k_last) + delta_k
            return eval_jk(density)

        if cfg.direct_jk_engine == "cuda" and cuda_ffi_available():
            cuda_builder = (
                direct_cuda_jk_builder
                if direct_cuda_jk_builder is not None
                else CudaDirectJKBuilder(direct_basis)
            )
            threshold = float(cfg.direct_scf_tol)
            precompute_screening = getattr(cuda_builder, "precompute_screening_metadata", None)
            if threshold > 0.0 and callable(precompute_screening):
                precompute_screening()
            if eri_pair_matrix is not None:
                eri_pair_for_cuda = eri_pair_matrix
                eri_pair_arr = eri_pair_for_cuda

                def _eval_cuda_supplied_pair(density_arg: Array):
                    return cuda_builder.build_jk_from_eri_pair_matrix(
                        eri_pair_arr,
                        density_arg,
                    )

                eval_cuda_supplied_pair = _eval_cuda_supplied_pair

                def _direct_cuda_supplied_pair_jk(
                    density: Array,
                    mo_coeff: Array | None = None,
                    mo_occ: Array | None = None,
                    density_last: Array | None = None,
                    j_last: Array | None = None,
                    k_last: Array | None = None,
                ):
                    del mo_coeff, mo_occ
                    result_j, result_k = _accumulate_incremental_jk(
                        eval_cuda_supplied_pair,
                        density,
                        density_last=density_last,
                        j_last=j_last,
                        k_last=k_last,
                        use_incremental=cfg.direct_scf_incremental,
                    )
                    if with_k:
                        return result_j, result_k
                    return result_j, jnp.zeros_like(density)

                return _direct_cuda_supplied_pair_jk

            def _eval_cuda_direct(density_arg: Array):
                return cuda_builder.build_jk(
                    density_arg,
                    density_cutoff=threshold,
                )

            eval_cuda_direct = _eval_cuda_direct

            def _direct_cuda_jk(
                density: Array,
                mo_coeff: Array | None = None,
                mo_occ: Array | None = None,
                density_last: Array | None = None,
                j_last: Array | None = None,
                k_last: Array | None = None,
            ):
                del mo_coeff, mo_occ
                result_j, result_k = _accumulate_incremental_jk(
                    eval_cuda_direct,
                    density,
                    density_last=density_last,
                    j_last=j_last,
                    k_last=k_last,
                    use_incremental=cfg.direct_scf_incremental,
                )
                if with_k:
                    return result_j, result_k
                return result_j, jnp.zeros_like(density)

            return _direct_cuda_jk
        if cfg.direct_jk_engine not in {"jax", "cuda"}:
            raise ValueError("direct_jk_engine must be 'jax' or 'cuda'.")
        threshold = float(cfg.direct_scf_tol)
        if threshold <= 0.0 and int(direct_basis.nao) <= _DIRECT_PACKED_JK_MAX_NAO:
            pair_arr = jnp.asarray(eri_pair_matrix_packed(direct_basis))

            def _eval_direct_packed(density_arg: Array):
                return build_jk_from_eri_pair_matrix(pair_arr, density_arg)

            def _direct_packed_jk(
                density: Array,
                mo_coeff: Array | None = None,
                mo_occ: Array | None = None,
                density_last: Array | None = None,
                j_last: Array | None = None,
                k_last: Array | None = None,
            ):
                del mo_coeff, mo_occ
                result_j, result_k = _accumulate_incremental_jk(
                    _eval_direct_packed,
                    density,
                    density_last=density_last,
                    j_last=j_last,
                    k_last=k_last,
                    use_incremental=cfg.direct_scf_incremental,
                )
                if with_k:
                    return result_j, result_k
                return result_j, jnp.zeros_like(density)

            return _direct_packed_jk
        bounds = (
            shell_pair_schwarz_bounds(direct_basis)
            if threshold > 0.0
            else None
        )

        def _direct_jk(
            density: Array,
            mo_coeff: Array | None = None,
            mo_occ: Array | None = None,
            density_last: Array | None = None,
            j_last: Array | None = None,
            k_last: Array | None = None,
        ):
            del mo_coeff, mo_occ
            result = build_direct_jk_incremental(
                direct_basis,
                density,
                density_last=density_last if cfg.direct_scf_incremental else None,
                j_last=j_last,
                k_last=k_last,
                screening_threshold=threshold,
                shell_pair_schwarz_bounds=bounds,
            )
            if with_k:
                return result.j, result.k
            return result.j, jnp.zeros_like(density)

        return _direct_jk

    if cfg.jk_backend == "full":
        pair_arr = None if eri_pair_matrix is None else jnp.asarray(eri_pair_matrix)
        eri_arr = None if eri is None else jnp.asarray(eri)
        if eri_arr is None and pair_arr is None:
            if direct_basis is not None:
                return _direct_builder()
            raise ValueError("jk_backend='full' requires full AO ERI, packed AO-pair ERI, or direct_basis.")
        if not with_k:
            def _full_j(
                density: Array,
                mo_coeff: Array | None = None,
                mo_occ: Array | None = None,
                density_last: Array | None = None,
                j_last: Array | None = None,
                k_last: Array | None = None,
            ):
                del mo_coeff, mo_occ, density_last, j_last, k_last
                if pair_arr is not None:
                    return build_j_from_eri_pair_matrix(pair_arr, density), jnp.zeros_like(density)
                if eri_arr is None:
                    raise ValueError("jk_backend='full' requires full AO ERI or packed AO-pair ERI.")
                return _build_j(eri_arr, density), jnp.zeros_like(density)

            return _full_j

        def _full_jk(
            density: Array,
            mo_coeff: Array | None = None,
            mo_occ: Array | None = None,
            density_last: Array | None = None,
            j_last: Array | None = None,
            k_last: Array | None = None,
        ):
            del mo_coeff, mo_occ, density_last, j_last, k_last
            if pair_arr is not None:
                return build_jk_from_eri_pair_matrix(pair_arr, density)
            if eri_arr is None:
                raise ValueError("jk_backend='full' requires full AO ERI or packed AO-pair ERI.")
            return _build_jk(eri_arr, density)

        return _full_jk
    if cfg.jk_backend == "df":
        if df_factors is None:
            if eri is None:
                raise ValueError("jk_backend='df' requires either eri or df_factors.")
            df_factors = eri_to_df_factors(
                eri,
                tol=cfg.df_tol,
                max_rank=cfg.df_max_rank,
            )
        factors = jnp.asarray(df_factors)
        if not with_k:
            def _df_j(
                density: Array,
                mo_coeff: Array | None = None,
                mo_occ: Array | None = None,
                density_last: Array | None = None,
                j_last: Array | None = None,
                k_last: Array | None = None,
            ):
                del mo_coeff, mo_occ, density_last, j_last, k_last
                return build_j_from_df(factors, density), jnp.zeros_like(density)

            return _df_j

        use_orbital_k = nocc is not None and jnp.finfo(factors.dtype).bits >= 64

        def _df_jk(
            density: Array,
            mo_coeff: Array | None = None,
            mo_occ: Array | None = None,
            density_last: Array | None = None,
            j_last: Array | None = None,
            k_last: Array | None = None,
        ):
            del density_last, j_last, k_last
            if use_orbital_k and mo_coeff is not None and mo_occ is not None:
                return build_jk_from_df_orbitals(
                    factors,
                    density,
                    mo_coeff,
                    mo_occ,
                    nocc=int(nocc),
                )
            return build_jk_from_df(factors, density)

        return _df_jk
    if cfg.jk_backend == "direct":
        return _direct_builder()
    raise ValueError(
        f"Unsupported jk_backend={cfg.jk_backend!r}. Choose one of {{'full', 'df', 'direct'}}."
    )


def _restricted_spin_view(
    *,
    ao: Array,
    ao_deriv1: Array,
    weights: Array,
    density: Array,
    mo_coeff: Array,
    mo_occ: Array,
    mo_energy: Array,
):
    density_spin = jnp.stack([0.5 * density, 0.5 * density], axis=0)
    coeff_spin = jnp.stack([mo_coeff, mo_coeff], axis=0)
    occ_spin = jnp.stack([0.5 * mo_occ, 0.5 * mo_occ], axis=0)
    energy_spin = jnp.stack([mo_energy, mo_energy], axis=0)
    return MoleculeLikeState(
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid=molecule_grid_view(weights),
        rdm1=density_spin,
        mo_coeff=coeff_spin,
        mo_occ=occ_spin,
        mo_energy=energy_spin,
    )


@lru_cache(maxsize=64)
def _point_xc_value_and_grad_kernel(
    xc_spec: str,
    xc_kind: str,
    density_floor: float,
) -> Callable[[Array], tuple[Array, Array]]:
    xc_spec_norm = str(xc_spec)
    xc_kind_norm = str(xc_kind)
    density_floor_value = float(density_floor)

    def point_energy(variables: Array) -> Array:
        rho_point = jnp.maximum(variables[0], density_floor_value)
        if xc_kind_norm == "LDA":
            grad_point = jnp.zeros((3,), dtype=variables.dtype)
            tau_point = jnp.asarray(0.0, dtype=variables.dtype)
        elif xc_kind_norm == "GGA":
            grad_point = variables[1:4]
            tau_point = jnp.asarray(0.0, dtype=variables.dtype)
        elif xc_kind_norm == "MGGA":
            grad_point = variables[1:4]
            tau_point = jnp.maximum(variables[4], 0.0)
        else:
            raise ValueError(f"Unsupported XC kind={xc_kind_norm!r}.")
        point_features = restricted_feature_bundle_from_rho_grad_tau(
            rho_point,
            grad_point,
            tau_point,
            density_floor=density_floor_value,
        )
        return eval_xc_energy_density(xc_spec_norm, point_features)

    return jax.jit(jax.vmap(jax.value_and_grad(point_energy)))


@lru_cache(maxsize=64)
def _array_xc_value_and_grad_kernel(
    xc_spec: str,
    xc_kind: str,
    density_floor: float,
    use_jit: bool = True,
) -> Callable[[Array], tuple[Array, Array]]:
    xc_spec_norm = str(xc_spec)
    xc_kind_norm = str(xc_kind)
    density_floor_value = float(density_floor)

    def point_exc_array(variables: Array) -> Array:
        rho = jnp.maximum(variables[:, 0], density_floor_value)
        if xc_kind_norm == "LDA":
            grad = jnp.zeros((variables.shape[0], 3), dtype=variables.dtype)
            tau = jnp.zeros_like(rho)
        elif xc_kind_norm == "GGA":
            grad = variables[:, 1:4]
            tau = jnp.zeros_like(rho)
        elif xc_kind_norm == "MGGA":
            grad = variables[:, 1:4]
            tau = jnp.maximum(variables[:, 4], 0.0)
        else:
            raise ValueError(f"Unsupported XC kind={xc_kind_norm!r}.")
        features = restricted_feature_bundle_from_rho_grad_tau(
            rho,
            grad,
            tau,
            density_floor=density_floor_value,
        )
        return eval_xc_energy_density(xc_spec_norm, features)

    def unweighted_sum(variables: Array) -> Array:
        return jnp.sum(point_exc_array(variables))

    value_and_grad = jax.value_and_grad(unweighted_sum)

    def kernel(variables: Array) -> tuple[Array, Array]:
        _, grad = value_and_grad(variables)
        return point_exc_array(variables), grad

    if bool(use_jit):
        return jax.jit(kernel)
    return kernel


def _xc_energy_and_potential_on_grid(
    *,
    molecule,
    xc_spec: str,
    density_floor: float,
    potential_clip: float | None,
    xc_kind: str,
    jit_xc: bool,
) -> tuple[Array, Array, Array]:
    features, grad = restricted_grid_features_with_gradients(molecule)
    rho = jnp.maximum(features.rho, density_floor)
    tau = jnp.maximum(features.tau_a + features.tau_b, 0.0)
    weights = jnp.asarray(molecule.grid.weights)

    if xc_kind == "HF":
        zeros = jnp.zeros_like(rho)
        return jnp.asarray(0.0, dtype=rho.dtype), zeros, jnp.zeros((rho.shape[0], 3), dtype=rho.dtype)

    if xc_kind == "LDA":
        response_variables = rho[..., None]
    elif xc_kind == "GGA":
        response_variables = jnp.concatenate([rho[..., None], grad], axis=-1)
    elif xc_kind == "MGGA":
        response_variables = jnp.concatenate([rho[..., None], grad, tau[..., None]], axis=-1)
    else:
        raise ValueError(f"Unsupported XC kind={xc_kind!r}.")

    point_exc, point_grad = _array_xc_value_and_grad_kernel(
        xc_spec,
        xc_kind,
        density_floor,
        bool(jit_xc),
    )(response_variables)
    point_exc = jnp.nan_to_num(point_exc, nan=0.0, posinf=0.0, neginf=0.0)
    point_grad = jnp.nan_to_num(point_grad, nan=0.0, posinf=0.0, neginf=0.0)
    exc = jnp.tensordot(weights, point_exc, axes=(0, 0))

    mask = rho > density_floor
    vxc_rho = jnp.where(mask, point_grad[:, 0], 0.0)
    if xc_kind in {"GGA", "MGGA"}:
        vxc_grad = jnp.where(mask[:, None], point_grad[:, 1:4], 0.0)
    else:
        vxc_grad = jnp.zeros((rho.shape[0], 3), dtype=rho.dtype)

    if potential_clip is not None:
        clip = jnp.asarray(potential_clip, dtype=rho.dtype)
        vxc_rho = jnp.clip(vxc_rho, -clip, clip)
        vxc_grad = jnp.clip(vxc_grad, -clip, clip)
    return exc, vxc_rho, vxc_grad


def _vxc_matrix_from_grid_potential(
    *,
    ao: Array,
    ao_deriv1: Array,
    ao_laplacian: Array,
    weights: Array,
    vxc_rho: Array,
    vxc_grad: Array,
    vxc_tau: Array,
    vxc_lapl: Array,
    xc_kind: str,
) -> Array:
    vxc_matrix = jnp.einsum(
        "r,rp,rq->pq",
        weights * vxc_rho,
        ao,
        ao,
        precision=Precision.HIGHEST,
    )
    if xc_kind in {"GGA", "MGGA", "MGGA_LAPL"}:
        grad_term = jnp.einsum(
            "rx,xrp,rq->pq",
            weights[:, None] * vxc_grad,
            ao_deriv1[1:4],
            ao,
            precision=Precision.HIGHEST,
        )
        vxc_matrix = vxc_matrix + grad_term + grad_term.T
    if xc_kind in {"MGGA", "MGGA_LAPL"}:
        tau_term = 0.5 * jnp.einsum(
            "r,xrp,xrq->pq",
            weights * vxc_tau,
            ao_deriv1[1:4],
            ao_deriv1[1:4],
            precision=Precision.HIGHEST,
        )
        vxc_matrix = vxc_matrix + tau_term
    if xc_kind == "MGGA_LAPL":
        lapl_left = jnp.einsum(
            "r,rp,rq->pq",
            weights * vxc_lapl,
            ao_laplacian,
            ao,
            precision=Precision.HIGHEST,
        )
        lapl_grad = 2.0 * jnp.einsum(
            "r,xrp,xrq->pq",
            weights * vxc_lapl,
            ao_deriv1[1:4],
            ao_deriv1[1:4],
            precision=Precision.HIGHEST,
        )
        vxc_matrix = vxc_matrix + lapl_left + lapl_left.T + lapl_grad
    return 0.5 * (vxc_matrix + vxc_matrix.T)


def _fock_components_for_density(
    *,
    density: Array,
    mo_coeff: Array,
    mo_occ: Array,
    mo_energy: Array,
    jk_builder: JKBuilder,
    density_last: Array | None = None,
    j_last: Array | None = None,
    k_last: Array | None = None,
    ao: Array,
    ao_deriv1: Array,
    weights: Array,
    h: Array,
    alpha: Array,
    cfg: RKSConfig,
    xc_kind: str,
    jit_xc: bool,
) -> tuple[Array, Array, Array, Array, Array]:
    j_mat, k_mat = jk_builder(density, mo_coeff, mo_occ, density_last, j_last, k_last)
    molecule = _restricted_spin_view(
        ao=ao,
        ao_deriv1=ao_deriv1,
        weights=weights,
        density=density,
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
    )
    xc_energy, vxc_rho, vxc_grad = _xc_energy_and_potential_on_grid(
        molecule=molecule,
        xc_spec=cfg.xc_spec,
        density_floor=cfg.density_floor,
        potential_clip=cfg.potential_clip,
        xc_kind=xc_kind,
        jit_xc=jit_xc,
    )
    ao_laplacian = getattr(molecule, "ao_laplacian", None)
    if ao_laplacian is None:
        ao_laplacian = jnp.zeros_like(ao)
    vxc_matrix = _vxc_matrix_from_grid_potential(
        ao=ao,
        ao_deriv1=ao_deriv1,
        ao_laplacian=ao_laplacian,
        weights=weights,
        vxc_rho=vxc_rho,
        vxc_grad=vxc_grad,
        vxc_tau=jnp.zeros_like(vxc_rho),
        vxc_lapl=jnp.zeros_like(vxc_rho),
        xc_kind=xc_kind,
    )
    fock = h + j_mat - 0.5 * alpha * k_mat + vxc_matrix
    return j_mat, k_mat, xc_energy, vxc_matrix, fock


def _energy_and_raw_fock_for_density(
    *,
    density: Array,
    mo_coeff: Array,
    mo_occ: Array,
    mo_energy: Array,
    jk_builder: JKBuilder,
    density_last: Array | None = None,
    j_last: Array | None = None,
    k_last: Array | None = None,
    ao: Array,
    ao_deriv1: Array,
    weights: Array,
    h: Array,
    s: Array,
    enuc: Array,
    alpha: Array,
    cfg: RKSConfig,
    xc_kind: str,
    jit_xc: bool,
) -> tuple[Array, Array, Array, Array, Array]:
    j_mat, k_mat, xc_energy, _vxc_matrix, fock = _fock_components_for_density(
        density=density,
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        jk_builder=jk_builder,
        density_last=density_last,
        j_last=j_last,
        k_last=k_last,
        ao=ao,
        ao_deriv1=ao_deriv1,
        weights=weights,
        h=h,
        alpha=alpha,
        cfg=cfg,
        xc_kind=xc_kind,
        jit_xc=jit_xc,
    )

    e_one = jnp.einsum("ij,ij->", density, h, precision=Precision.HIGHEST)
    e_coul = 0.5 * jnp.einsum("ij,ij->", density, j_mat, precision=Precision.HIGHEST)
    e_x_hf = -0.25 * alpha * jnp.einsum(
        "ij,ij->",
        density,
        k_mat,
        precision=Precision.HIGHEST,
    )
    total = e_one + e_coul + e_x_hf + xc_energy + enuc
    return total, xc_energy, fock, j_mat, k_mat


def _raw_fock_for_density(
    *,
    density: Array,
    mo_coeff: Array,
    mo_occ: Array,
    mo_energy: Array,
    jk_builder: JKBuilder,
    density_last: Array | None = None,
    j_last: Array | None = None,
    k_last: Array | None = None,
    ao: Array,
    ao_deriv1: Array,
    weights: Array,
    h: Array,
    s: Array,
    alpha: Array,
    cfg: RKSConfig,
    xc_kind: str,
    jit_xc: bool,
) -> tuple[Array, Array, Array]:
    j_mat, k_mat, _xc_energy, _vxc_matrix, fock = _fock_components_for_density(
        density=density,
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        jk_builder=jk_builder,
        density_last=density_last,
        j_last=j_last,
        k_last=k_last,
        ao=ao,
        ao_deriv1=ao_deriv1,
        weights=weights,
        h=h,
        alpha=alpha,
        cfg=cfg,
        xc_kind=xc_kind,
        jit_xc=jit_xc,
    )
    return fock, j_mat, k_mat


def _prewarm_runtime_rks_primitives(
    *,
    overlap: Array,
    hcore: Array,
    eri: Array | None,
    eri_pair_matrix: Array | None,
    nelectron: int,
    ao: Array,
    ao_deriv1: Array,
    grid_weights: Array,
    df_factors: Array | None,
    direct_basis: CartesianBasis | None,
    config: RKSConfig,
) -> None:
    if config.iteration_backend != "runtime":
        return
    if nelectron % 2 != 0:
        return
    if (
        config.jk_backend == "direct"
        and direct_basis is not None
        and eri_pair_matrix is None
        and not bool(getattr(direct_basis, "shell_quartet_groups", ()))
    ):
        return
    xc_kind = xc_type(config.xc_spec)
    if xc_kind == "MGGA":
        return
    key = _runtime_rks_prewarm_cache_key(
        overlap=overlap,
        ao=ao,
        cfg=config,
        eri=eri,
        eri_pair_matrix=eri_pair_matrix,
        df_factors=df_factors,
        direct_basis=direct_basis,
    )
    if key in _RUNTIME_RKS_PREWARMED:
        return
    if len(_RUNTIME_RKS_PREWARMED) >= _RUNTIME_RKS_PREWARM_CACHE_MAXSIZE:
        _RUNTIME_RKS_PREWARMED.clear()

    s = jnp.asarray(overlap)
    h = jnp.asarray(hcore)
    eri_arr = None if eri is None else jnp.asarray(eri)
    eri_pair_arr = None if eri_pair_matrix is None else jnp.asarray(eri_pair_matrix)
    ao_arr = jnp.asarray(ao)
    ao_deriv1_arr = jnp.asarray(ao_deriv1)
    weights_arr = jnp.asarray(grid_weights)
    nao = int(s.shape[0])
    nocc = int(nelectron) // 2
    if nocc <= 0 or nocc > nao:
        return
    x = _orthogonalizer(s, config.orthogonalization_eps)
    mo_energy, mo_coeff = _diagonalize_fock(h, x)
    mo_occ = _closed_shell_mo_occ(nao, nocc, h.dtype)
    density = _build_density_from_occ(mo_coeff, mo_occ)
    alpha_scalar = float(hybrid_coeff(config.xc_spec))
    alpha = jnp.asarray(alpha_scalar, dtype=h.dtype)
    jk_builder = _make_jk_builder(
        eri_arr,
        config,
        eri_pair_matrix=eri_pair_arr,
        df_factors=df_factors,
        direct_basis=direct_basis,
        with_k=bool(abs(alpha_scalar) > 1e-14),
        nocc=nocc,
    )
    total, xc_energy, fock, j_mat, k_mat = _energy_and_raw_fock_for_density(
        density=density,
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        jk_builder=jk_builder,
        ao=ao_arr,
        ao_deriv1=ao_deriv1_arr,
        weights=weights_arr,
        h=h,
        s=s,
        enuc=jnp.asarray(0.0, dtype=h.dtype),
        alpha=alpha,
        cfg=config,
        xc_kind=xc_kind,
        jit_xc=False,
    )
    error = _orthonormal_diis_error(fock, density, s, mo_coeff)
    fock_hist = jnp.zeros((_PYSCF_LIKE_DIIS_SPACE, nao, nao), dtype=h.dtype).at[0].set(fock)
    err_hist = jnp.zeros((_PYSCF_LIKE_DIIS_SPACE, nao * nao), dtype=h.dtype).at[0].set(error.reshape(-1))
    fock_diis, fock_hist, err_hist, hist_head, hist_count = _diis_extrapolate_runtime(
        fock,
        error,
        fock_hist,
        err_hist,
        1,
        1,
    )
    residual = _scf_residual_norm(fock, density, s)
    for value in (
        x,
        mo_energy,
        mo_coeff,
        density,
        total,
        xc_energy,
        fock,
        j_mat,
        k_mat,
        fock_diis,
        fock_hist,
        err_hist,
        residual,
        jnp.asarray(hist_head),
        jnp.asarray(hist_count),
    ):
        jax.block_until_ready(value)
    _RUNTIME_RKS_PREWARMED.add(key)


def _run_extra_final_cycle(
    *,
    density: Array,
    mo_coeff: Array,
    mo_occ: Array,
    mo_energy: Array,
    raw_fock: Array,
    x: Array,
    jk_builder: JKBuilder,
    ao: Array,
    ao_deriv1: Array,
    weights: Array,
    h: Array,
    s: Array,
    enuc: Array,
    cfg: RKSConfig,
    alpha: Array,
    xc_kind: str,
    mo_occ_fixed: Array,
    energy: Array,
    j_mat: Array,
    k_mat: Array,
    jit_xc: bool,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    mo_energy_final, mo_coeff_final = _diagonalize_fock(raw_fock, x)
    density_final = _build_density_from_occ(mo_coeff_final, mo_occ_fixed)
    total_final, xc_energy_final, raw_fock_final, _, _ = _energy_and_raw_fock_for_density(
        density=density_final,
        mo_coeff=mo_coeff_final,
        mo_occ=mo_occ,
        mo_energy=mo_energy_final,
        jk_builder=jk_builder,
        density_last=density,
        j_last=j_mat,
        k_last=k_mat,
        ao=ao,
        ao_deriv1=ao_deriv1,
        weights=weights,
        h=h,
        s=s,
        enuc=enuc,
        alpha=alpha,
        cfg=cfg,
        xc_kind=xc_kind,
        jit_xc=jit_xc,
    )
    tol_e = jnp.asarray(cfg.conv_tol, dtype=h.dtype) * jnp.asarray(10.0, dtype=h.dtype)
    grad_tol = jnp.sqrt(jnp.asarray(cfg.conv_tol, dtype=h.dtype)) * jnp.asarray(3.0, dtype=h.dtype)
    delta_e = jnp.abs(total_final - energy)
    grad_rms = _scf_residual_norm(raw_fock_final, density_final, s)
    converged_final = jnp.logical_or(delta_e < tol_e, grad_rms < grad_tol)
    return (
        density_final,
        mo_coeff_final,
        mo_energy_final,
        total_final,
        xc_energy_final,
        raw_fock_final,
        converged_final,
    )


def _maybe_run_extra_final_cycle(
    *,
    density: Array,
    mo_coeff: Array,
    mo_energy: Array,
    energy: Array,
    xc_energy: Array,
    raw_fock: Array,
    converged,
    x: Array,
    jk_builder: JKBuilder,
    ao: Array,
    ao_deriv1: Array,
    weights: Array,
    h: Array,
    s: Array,
    enuc: Array,
    cfg: RKSConfig,
    alpha: Array,
    xc_kind: str,
    mo_occ_fixed: Array,
    j_mat: Array,
    k_mat: Array,
    traceable: bool,
    jit_xc: bool,
) -> tuple[Array, Array, Array, Array, Array, Array, Any]:
    if cfg.level_shift == 0.0:
        return density, mo_coeff, mo_energy, energy, xc_energy, raw_fock, converged

    def _run_cycle(_: None):
        return _run_extra_final_cycle(
            density=density,
            mo_coeff=mo_coeff,
            mo_occ=mo_occ_fixed,
            mo_energy=mo_energy,
            raw_fock=raw_fock,
            x=x,
            jk_builder=jk_builder,
            ao=ao,
            ao_deriv1=ao_deriv1,
            weights=weights,
            h=h,
            s=s,
            enuc=enuc,
            cfg=cfg,
            alpha=alpha,
            xc_kind=xc_kind,
            mo_occ_fixed=mo_occ_fixed,
            energy=energy,
            j_mat=j_mat,
            k_mat=k_mat,
            jit_xc=jit_xc,
        )

    if traceable:
        return jax.lax.cond(
            converged,
            _run_cycle,
            lambda _: (
                density,
                mo_coeff,
                mo_energy,
                energy,
                xc_energy,
                raw_fock,
                converged,
            ),
            operand=None,
        )

    if converged:
        (
            density,
            mo_coeff,
            mo_energy,
            energy,
            xc_energy,
            raw_fock,
            extra_converged,
        ) = _run_cycle(None)
        converged = bool(extra_converged)
    return density, mo_coeff, mo_energy, energy, xc_energy, raw_fock, converged


def _run_scf_iterations_lax(
    *,
    h: Array,
    s: Array,
    x: Array,
    jk_builder: JKBuilder,
    ao: Array,
    ao_deriv1: Array,
    weights: Array,
    enuc: Array,
    cfg: RKSConfig,
    alpha: Array,
    xc_kind: str,
    mo_occ_fixed: Array,
    diis_basis: Array,
    skip_first_fock_damping: bool,
    density: Array,
    mo_coeff: Array,
    mo_occ: Array,
    mo_energy: Array,
    raw_fock: Array,
    j_mat: Array,
    k_mat: Array,
    jit_xc: bool,
) -> tuple[bool, int, Array, Array, Array, Array, Array, Array, Array, Array]:
    """SCF loop wrapper returning Python scalars on the non-traced path."""

    converged, cycles, density, mo_coeff, mo_energy, energy, xc_energy, raw_fock, j_mat, k_mat = (
        _run_scf_iterations_lax_arrays(
            h=h,
            s=s,
            x=x,
            jk_builder=jk_builder,
            ao=ao,
            ao_deriv1=ao_deriv1,
            weights=weights,
            enuc=enuc,
            cfg=cfg,
            alpha=alpha,
            xc_kind=xc_kind,
            mo_occ_fixed=mo_occ_fixed,
            diis_basis=diis_basis,
            skip_first_fock_damping=skip_first_fock_damping,
            density=density,
            mo_coeff=mo_coeff,
            mo_occ=mo_occ,
            mo_energy=mo_energy,
            raw_fock=raw_fock,
            j_mat=j_mat,
            k_mat=k_mat,
            jit_xc=jit_xc,
        )
    )
    if isinstance(converged, jax.core.Tracer) or isinstance(cycles, jax.core.Tracer):
        return converged, cycles, density, mo_coeff, mo_energy, energy, xc_energy, raw_fock, j_mat, k_mat
    return bool(converged), int(cycles), density, mo_coeff, mo_energy, energy, xc_energy, raw_fock, j_mat, k_mat


def _run_scf_iterations_lax_arrays(
    *,
    h: Array,
    s: Array,
    x: Array,
    jk_builder: JKBuilder,
    ao: Array,
    ao_deriv1: Array,
    weights: Array,
    enuc: Array,
    cfg: RKSConfig,
    alpha: Array,
    xc_kind: str,
    mo_occ_fixed: Array,
    diis_basis: Array,
    skip_first_fock_damping: bool,
    density: Array,
    mo_coeff: Array,
    mo_occ: Array,
    mo_energy: Array,
    raw_fock: Array,
    j_mat: Array,
    k_mat: Array,
    jit_xc: bool,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
    """Array-valued SCF loop shared by regular and traceable RKS execution."""

    tol_e = jnp.asarray(cfg.conv_tol, dtype=h.dtype)
    grad_tol = jnp.sqrt(tol_e)
    damping = jnp.asarray(cfg.damping, dtype=h.dtype)
    has_damping = cfg.damping != 0.0
    use_pyscf_like_damping = jnp.finfo(h.dtype).bits >= 64
    level_shift = jnp.asarray(cfg.level_shift, dtype=h.dtype)
    has_level_shift = cfg.level_shift != 0.0

    def body_fn(carry: RKSIterationCarry) -> RKSIterationCarry:
        cycle = carry.cycle
        converged = carry.converged
        density_i = carry.density
        mo_coeff_i = carry.mo_coeff
        mo_energy_i = carry.mo_energy
        energy_i = carry.energy
        raw_fock_i = carry.raw_fock
        j_i = carry.j_mat
        k_i = carry.k_mat
        fock_last = carry.fock_last
        fock_hist = carry.fock_hist
        err_hist = carry.err_hist
        hist_head = carry.hist_head
        hist_count = carry.hist_count

        use_fock_damping = jnp.logical_and(
            jnp.asarray(has_damping and use_pyscf_like_damping),
            jnp.logical_and(
                jnp.logical_or(
                    jnp.asarray(not skip_first_fock_damping),
                    cycle > jnp.asarray(0, dtype=cycle.dtype),
                ),
                cycle < jnp.asarray(_PYSCF_LIKE_DIIS_START_CYCLE - 1, dtype=cycle.dtype),
            ),
        )
        fock_pre_diis = jax.lax.cond(
            use_fock_damping,
            lambda operand: _apply_fock_damping(*operand),
            lambda operand: operand[0],
            operand=(raw_fock_i, fock_last, damping),
        )
        diis_active = cycle >= jnp.asarray(1, dtype=cycle.dtype)
        fock_eff, fock_hist, err_hist, hist_head, hist_count = jax.lax.cond(
            diis_active,
            lambda operand: _diis_extrapolate_lax(
                operand[0],
                _orthonormal_diis_error(operand[0], operand[1], operand[2], operand[3]),
                operand[4],
                operand[5],
                operand[6],
                operand[7],
            ),
            lambda operand: (
                operand[0],
                operand[4],
                operand[5],
                operand[6],
                operand[7],
            ),
            operand=(fock_pre_diis, density_i, s, diis_basis, fock_hist, err_hist, hist_head, hist_count),
        )
        fock_eff = jax.lax.cond(
            jnp.asarray(has_level_shift),
            lambda operand: _apply_level_shift(*operand),
            lambda operand: operand[0],
            operand=(fock_eff, s, density_i, level_shift),
        )

        mo_energy_new, mo_coeff_new = _diagonalize_fock(fock_eff, x)
        density_new = _build_density_from_occ(mo_coeff_new, mo_occ_fixed)
        use_density_damping = jnp.logical_and(
            jnp.asarray(has_damping and (not use_pyscf_like_damping)),
            hist_count < jnp.asarray(_PYSCF_LIKE_DIIS_START_CYCLE, dtype=hist_count.dtype),
        )
        density_new = jax.lax.cond(
            use_density_damping,
            lambda pair: (1.0 - damping) * pair[0] + damping * pair[1],
            lambda pair: pair[0],
            operand=(density_new, density_i),
        )

        total_new, xc_energy_new, raw_fock_new, j_new, k_new = _energy_and_raw_fock_for_density(
            density=density_new,
            mo_coeff=mo_coeff_new,
            mo_occ=mo_occ,
            mo_energy=mo_energy_new,
            jk_builder=jk_builder,
            density_last=density_i,
            j_last=j_i,
            k_last=k_i,
            ao=ao,
            ao_deriv1=ao_deriv1,
            weights=weights,
            h=h,
            s=s,
            enuc=enuc,
            alpha=alpha,
            cfg=cfg,
            xc_kind=xc_kind,
            jit_xc=jit_xc,
        )

        delta_e = jnp.abs(total_new - energy_i)
        grad_rms = _scf_residual_norm(raw_fock_new, density_new, s)
        converged_new = jnp.logical_or(
            converged,
            jnp.logical_and(delta_e < tol_e, grad_rms < grad_tol),
        )
        return RKSIterationCarry(
            cycle=cycle + 1,
            converged=converged_new,
            density=density_new,
            mo_coeff=mo_coeff_new,
            mo_energy=mo_energy_new,
            energy=total_new,
            xc_energy=xc_energy_new,
            raw_fock=raw_fock_new,
            j_mat=j_new,
            k_mat=k_new,
            fock_last=fock_eff,
            fock_hist=fock_hist,
            err_hist=err_hist,
            hist_head=hist_head,
            hist_count=hist_count,
        )

    nao = h.shape[0]
    init = RKSIterationCarry(
        cycle=jnp.asarray(0, dtype=jnp.int32),
        converged=jnp.asarray(False),
        density=density,
        mo_coeff=mo_coeff,
        mo_energy=mo_energy,
        energy=jnp.asarray(0.0, dtype=h.dtype),
        xc_energy=jnp.asarray(0.0, dtype=h.dtype),
        raw_fock=raw_fock,
        j_mat=j_mat,
        k_mat=k_mat,
        fock_last=h,
        fock_hist=jnp.zeros((_PYSCF_LIKE_DIIS_SPACE, nao, nao), dtype=h.dtype),
        err_hist=jnp.zeros((_PYSCF_LIKE_DIIS_SPACE, nao * nao), dtype=h.dtype),
        hist_head=jnp.asarray(0, dtype=jnp.int32),
        hist_count=jnp.asarray(0, dtype=jnp.int32),
    )

    def scan_step(carry: RKSIterationCarry, _):
        converged = carry.converged
        next_carry = jax.lax.cond(
            converged,
            lambda current: current,
            body_fn,
            carry,
        )
        return next_carry, None

    final_carry, _ = jax.lax.scan(
        scan_step,
        init,
        xs=None,
        length=int(cfg.max_cycle),
    )
    return (
        final_carry.converged,
        final_carry.cycle,
        final_carry.density,
        final_carry.mo_coeff,
        final_carry.mo_energy,
        final_carry.energy,
        final_carry.xc_energy,
        final_carry.raw_fock,
        final_carry.j_mat,
        final_carry.k_mat,
    )


def _run_scf_iterations_runtime(
    *,
    h: Array,
    s: Array,
    x: Array,
    jk_builder: JKBuilder,
    ao: Array,
    ao_deriv1: Array,
    weights: Array,
    enuc: Array,
    cfg: RKSConfig,
    alpha: Array,
    xc_kind: str,
    mo_occ_fixed: Array,
    diis_basis: Array,
    skip_first_fock_damping: bool,
    density: Array,
    mo_coeff: Array,
    mo_occ: Array,
    mo_energy: Array,
    raw_fock: Array,
    j_mat: Array,
    k_mat: Array,
    jit_xc: bool,
) -> tuple[bool, int, Array, Array, Array, Array, Array, Array, Array, Array]:
    tol_e = float(cfg.conv_tol)
    grad_tol = float(np.sqrt(cfg.conv_tol))
    damping = float(cfg.damping)
    has_damping = damping != 0.0
    use_pyscf_like_damping = jnp.finfo(h.dtype).bits >= 64
    level_shift = jnp.asarray(cfg.level_shift, dtype=h.dtype)
    has_level_shift = cfg.level_shift != 0.0
    nao = int(h.shape[0])
    energy = jnp.asarray(0.0, dtype=h.dtype)
    xc_energy = jnp.asarray(0.0, dtype=h.dtype)
    fock_last = h
    fock_hist = jnp.zeros((_PYSCF_LIKE_DIIS_SPACE, nao, nao), dtype=h.dtype)
    err_hist = jnp.zeros((_PYSCF_LIKE_DIIS_SPACE, nao * nao), dtype=h.dtype)
    hist_head = 0
    hist_count = 0
    converged = False
    cycles = 0

    for cycle in range(int(cfg.max_cycle)):
        if converged:
            break
        use_fock_damping = (
            has_damping
            and use_pyscf_like_damping
            and ((not skip_first_fock_damping or cycle > 0) and cycle < (_PYSCF_LIKE_DIIS_START_CYCLE - 1))
        )
        if use_fock_damping:
            fock_pre_diis = _apply_fock_damping(raw_fock, fock_last, jnp.asarray(damping, dtype=h.dtype))
        else:
            fock_pre_diis = raw_fock
        if cycle >= 1:
            fock_eff, fock_hist, err_hist, hist_head, hist_count = _diis_extrapolate_runtime(
                fock_pre_diis,
                _orthonormal_diis_error(fock_pre_diis, density, s, diis_basis),
                fock_hist,
                err_hist,
                hist_head,
                hist_count,
            )
        else:
            fock_eff = fock_pre_diis
        if has_level_shift:
            fock_eff = _apply_level_shift(fock_eff, s, density, level_shift)

        mo_energy_new, mo_coeff_new = _diagonalize_fock(fock_eff, x)
        density_new = _build_density_from_occ(mo_coeff_new, mo_occ_fixed)
        if has_damping and (not use_pyscf_like_damping) and hist_count < _PYSCF_LIKE_DIIS_START_CYCLE:
            density_new = (1.0 - damping) * density_new + damping * density

        total_new, xc_energy_new, raw_fock_new, j_new, k_new = _energy_and_raw_fock_for_density(
            density=density_new,
            mo_coeff=mo_coeff_new,
            mo_occ=mo_occ,
            mo_energy=mo_energy_new,
            jk_builder=jk_builder,
            density_last=density,
            j_last=j_mat,
            k_last=k_mat,
            ao=ao,
            ao_deriv1=ao_deriv1,
            weights=weights,
            h=h,
            s=s,
            enuc=enuc,
            alpha=alpha,
            cfg=cfg,
            xc_kind=xc_kind,
            jit_xc=jit_xc,
        )

        delta_e = float(np.asarray(jax.device_get(jnp.abs(total_new - energy))))
        grad_rms = float(np.asarray(jax.device_get(_scf_residual_norm(raw_fock_new, density_new, s))))
        converged = delta_e < tol_e and grad_rms < grad_tol
        cycles = cycle + 1
        density = density_new
        mo_coeff = mo_coeff_new
        mo_energy = mo_energy_new
        energy = total_new
        xc_energy = xc_energy_new
        raw_fock = raw_fock_new
        j_mat = j_new
        k_mat = k_new
        fock_last = fock_eff

    return converged, cycles, density, mo_coeff, mo_energy, energy, xc_energy, raw_fock, j_mat, k_mat


def _run_rks_from_integrals_shared(
    *,
    overlap: Array,
    hcore: Array,
    eri: Array | None,
    eri_pair_matrix: Array | None = None,
    nelectron: int,
    nuclear_repulsion: float | Array,
    ao: Array,
    ao_deriv1: Array,
    grid_weights: Array,
    df_factors: Array | None = None,
    direct_basis: CartesianBasis | None = None,
    direct_joltqc_basis_data: Array | None = None,
    direct_cuda_jk_builder: Any | None = None,
    init_density: Array | None = None,
    init_mo_coeff: Array | None = None,
    init_mo_occ: Array | None = None,
    init_mo_energy: Array | None = None,
    config: RKSConfig | None = None,
    traceable: bool,
) -> tuple[
    Array,
    float,
    Array,
    Array,
    Array,
    Any,
    Any,
    Any,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
]:
    cfg = RKSConfig() if config is None else config
    if cfg.iteration_backend not in {"runtime", "lax"}:
        raise ValueError(
            f"Unsupported iteration_backend={cfg.iteration_backend!r}. "
            "RKS SCF supports {'runtime', 'lax'}."
        )
    if traceable and cfg.iteration_backend != "lax":
        raise ValueError(
            "run_rks_from_integrals_traceable requires config.iteration_backend='lax' "
            "to remain JAX-traceable."
        )
    parse_xc(cfg.xc_spec)
    xc_kind = xc_type(cfg.xc_spec)
    if xc_kind == "MGGA":
        if traceable:
            raise NotImplementedError(
                "Traceable RKS matrix assembly currently supports LDA/GGA/HF semilocal terms. "
                "MGGA requires tau-dependent AO Hessian terms."
            )
        raise NotImplementedError(
            "RKS SCF matrix assembly currently supports LDA/GGA/HF semilocal terms. "
            "MGGA requires tau-dependent AO Hessian terms."
        )

    s = jnp.asarray(overlap)
    h = jnp.asarray(hcore)
    eri_arr = None if eri is None else jnp.asarray(eri)
    eri_pair_arr = None if eri_pair_matrix is None else jnp.asarray(eri_pair_matrix)
    ao = jnp.asarray(ao)
    ao_deriv1 = jnp.asarray(ao_deriv1)
    weights = jnp.asarray(grid_weights)
    enuc = (
        jnp.asarray(nuclear_repulsion)
        if traceable
        else _host_device_scalar(float(np.asarray(jax.device_get(nuclear_repulsion))), h.dtype)
    )
    nao = int(s.shape[0])

    if (not traceable) and _use_host_initial_orbitals(cfg):
        x, mo_energy, mo_coeff, mo_occ_fixed, nocc, density = _initial_scf_orbitals_host(
            overlap=overlap,
            hcore=hcore,
            init_mo_coeff=init_mo_coeff,
            init_mo_occ=init_mo_occ,
            init_mo_energy=init_mo_energy,
            nelectron=nelectron,
            orthogonalization_eps=cfg.orthogonalization_eps,
            dtype=h.dtype,
        )
    else:
        x = _orthogonalizer(s, cfg.orthogonalization_eps)
        if init_mo_coeff is None:
            mo_energy, mo_coeff = _diagonalize_fock(h, x)
        else:
            mo_coeff = jnp.asarray(init_mo_coeff)
            if mo_coeff.ndim == 3:
                mo_coeff = mo_coeff[0]
            if init_mo_energy is None:
                mo_energy, _ = _diagonalize_fock(h, x)
            else:
                mo_energy = jnp.asarray(init_mo_energy)
                if mo_energy.ndim == 2:
                    mo_energy = mo_energy[0]
        if init_mo_occ is None:
            if nelectron % 2 != 0:
                raise ValueError("RKS requires an even number of electrons when init_mo_occ is not provided.")
            nocc = nelectron // 2
            if nocc <= 0 or nocc > nao:
                raise ValueError("Invalid occupation count for RKS.")
            mo_occ_fixed = _closed_shell_mo_occ(nao, nocc, h.dtype)
        else:
            mo_occ_fixed = jnp.asarray(init_mo_occ, dtype=h.dtype)
            if mo_occ_fixed.ndim == 2:
                mo_occ_fixed = mo_occ_fixed[0]
            if mo_occ_fixed.ndim != 1:
                raise ValueError("init_mo_occ must be a 1D occupation vector for RKS.")
            if float(jnp.max(mo_occ_fixed)) <= 1.0 + 1e-6:
                mo_occ_fixed = mo_occ_fixed * 2.0
            if int(mo_occ_fixed.shape[0]) != nao:
                raise ValueError("init_mo_occ must match the AO/MO dimension for RKS.")
            nocc = int(jnp.count_nonzero(mo_occ_fixed > jnp.asarray(1e-12, dtype=h.dtype)))
        density = _build_density_from_occ(mo_coeff, mo_occ_fixed)

    init_density_matrix = _validate_initial_density(
        init_density,
        nao=nao,
        dtype=h.dtype,
        label="init_density",
    )
    if init_density_matrix is not None:
        density = init_density_matrix

    skip_first_fock_damping = (init_mo_coeff is not None) or (init_density_matrix is not None)
    alpha_scalar = float(hybrid_coeff(cfg.xc_spec))
    alpha = jnp.asarray(alpha_scalar, dtype=h.dtype) if traceable else _host_device_scalar(alpha_scalar, h.dtype)
    jk_builder = _make_jk_builder(
        eri_arr,
        cfg,
        eri_pair_matrix=eri_pair_arr,
        df_factors=df_factors,
        direct_basis=direct_basis,
        direct_joltqc_basis_data=direct_joltqc_basis_data,
        direct_cuda_jk_builder=direct_cuda_jk_builder,
        with_k=bool(abs(alpha_scalar) > 1e-14),
        nocc=nocc,
    )
    if (not traceable) and _defer_initial_fock_build(cfg):
        fock = h
        j_mat = _host_device_zeros_like(h)
        k_mat = _host_device_zeros_like(h)
    else:
        fock, j_mat, k_mat = _raw_fock_for_density(
            density=density,
            mo_coeff=mo_coeff,
            mo_occ=mo_occ_fixed,
            mo_energy=mo_energy,
            jk_builder=jk_builder,
            ao=ao,
            ao_deriv1=ao_deriv1,
            weights=weights,
            h=h,
            s=s,
            alpha=alpha,
            cfg=cfg,
            xc_kind=xc_kind,
            jit_xc=cfg.iteration_backend == "lax",
        )
    scf_kwargs = dict(
        h=h,
        s=s,
        x=x,
        jk_builder=jk_builder,
        ao=ao,
        ao_deriv1=ao_deriv1,
        weights=weights,
        enuc=enuc,
        cfg=cfg,
        alpha=alpha,
        xc_kind=xc_kind,
        mo_occ_fixed=mo_occ_fixed,
        diis_basis=mo_coeff,
        skip_first_fock_damping=skip_first_fock_damping,
        density=density,
        mo_coeff=mo_coeff,
        mo_occ=mo_occ_fixed,
        mo_energy=mo_energy,
        raw_fock=fock,
        j_mat=j_mat,
        k_mat=k_mat,
    )
    scf_iteration_runner = (
        _run_scf_iterations_lax_arrays
        if traceable
        else (_run_scf_iterations_lax if cfg.iteration_backend == "lax" else _run_scf_iterations_runtime)
    )
    converged, cycles, density, mo_coeff, mo_energy, energy, xc_energy, fock, j_mat, k_mat = scf_iteration_runner(
        **scf_kwargs,
        jit_xc=cfg.iteration_backend == "lax",
    )
    density, mo_coeff, mo_energy, energy, xc_energy, fock, converged = _maybe_run_extra_final_cycle(
        density=density,
        mo_coeff=mo_coeff,
        mo_energy=mo_energy,
        energy=energy,
        xc_energy=xc_energy,
        raw_fock=fock,
        converged=converged,
        x=x,
        jk_builder=jk_builder,
        ao=ao,
        ao_deriv1=ao_deriv1,
        weights=weights,
        h=h,
        s=s,
        enuc=enuc,
        cfg=cfg,
        alpha=alpha,
        xc_kind=xc_kind,
        mo_occ_fixed=mo_occ_fixed,
        j_mat=j_mat,
        k_mat=k_mat,
        traceable=traceable,
        jit_xc=cfg.iteration_backend == "lax",
    )
    return (
        enuc,
        alpha_scalar,
        alpha,
        s,
        h,
        mo_occ_fixed,
        converged,
        cycles,
        density,
        mo_coeff,
        mo_energy,
        energy,
        xc_energy,
        fock,
    )


def run_rks_from_integrals(
    *,
    overlap: Array,
    hcore: Array,
    eri: Array | None,
    eri_pair_matrix: Array | None = None,
    nelectron: int,
    nuclear_repulsion: float | Array,
    ao: Array,
    ao_deriv1: Array,
    grid_weights: Array,
    df_factors: Array | None = None,
    direct_basis: CartesianBasis | None = None,
    direct_joltqc_basis_data: Array | None = None,
    direct_cuda_jk_builder: Any | None = None,
    init_density: Array | None = None,
    init_mo_coeff: Array | None = None,
    init_mo_occ: Array | None = None,
    init_mo_energy: Array | None = None,
    config: RKSConfig | None = None,
) -> RKSResult:
    """Run restricted Kohn-Sham SCF from AO integrals and numerical grid data."""
    (
        enuc,
        alpha_scalar,
        _alpha,
        s,
        h,
        mo_occ_fixed,
        converged,
        cycles,
        density,
        mo_coeff,
        mo_energy,
        energy,
        xc_energy,
        fock,
    ) = _run_rks_from_integrals_shared(
        overlap=overlap,
        hcore=hcore,
        eri=eri,
        eri_pair_matrix=eri_pair_matrix,
        nelectron=nelectron,
        nuclear_repulsion=nuclear_repulsion,
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid_weights=grid_weights,
        df_factors=df_factors,
        direct_basis=direct_basis,
        direct_joltqc_basis_data=direct_joltqc_basis_data,
        direct_cuda_jk_builder=direct_cuda_jk_builder,
        init_density=init_density,
        init_mo_coeff=init_mo_coeff,
        init_mo_occ=init_mo_occ,
        init_mo_energy=init_mo_energy,
        config=config,
        traceable=False,
    )

    return RKSResult(
        converged=converged,
        total_energy=float(energy),
        electronic_energy=float(energy) - float(enuc),
        nuclear_repulsion=float(enuc),
        xc_energy=float(xc_energy),
        exact_exchange_fraction=float(alpha_scalar),
        mo_energy=mo_energy,
        mo_coeff=mo_coeff,
        mo_occ=mo_occ_fixed,
        density_matrix=density,
        fock_matrix=fock,
        overlap_matrix=s,
        hcore_matrix=h,
        cycles=cycles,
    )


def run_rks_from_integrals_traceable(
    *,
    overlap: Array,
    hcore: Array,
    eri: Array | None,
    eri_pair_matrix: Array | None = None,
    nelectron: int,
    nuclear_repulsion: float | Array,
    ao: Array,
    ao_deriv1: Array,
    grid_weights: Array,
    df_factors: Array | None = None,
    direct_basis: CartesianBasis | None = None,
    direct_joltqc_basis_data: Array | None = None,
    direct_cuda_jk_builder: Any | None = None,
    init_density: Array | None = None,
    init_mo_coeff: Array | None = None,
    init_mo_occ: Array | None = None,
    init_mo_energy: Array | None = None,
    config: RKSConfig | None = None,
) -> TraceableRKSResult:
    """Traceable RKS SCF from AO integrals (array-valued outputs for autodiff)."""
    cfg = replace(RKSConfig() if config is None else config, iteration_backend="lax")
    (
        enuc,
        _alpha_scalar,
        alpha,
        s,
        h,
        mo_occ_fixed,
        converged,
        cycles,
        density,
        mo_coeff,
        mo_energy,
        energy,
        xc_energy,
        fock,
    ) = _run_rks_from_integrals_shared(
        overlap=overlap,
        hcore=hcore,
        eri=eri,
        eri_pair_matrix=eri_pair_matrix,
        nelectron=nelectron,
        nuclear_repulsion=nuclear_repulsion,
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid_weights=grid_weights,
        df_factors=df_factors,
        direct_basis=direct_basis,
        direct_joltqc_basis_data=direct_joltqc_basis_data,
        direct_cuda_jk_builder=direct_cuda_jk_builder,
        init_density=init_density,
        init_mo_coeff=init_mo_coeff,
        init_mo_occ=init_mo_occ,
        init_mo_energy=init_mo_energy,
        config=cfg,
        traceable=True,
    )

    return TraceableRKSResult(
        converged=converged,
        total_energy=energy,
        electronic_energy=energy - enuc,
        nuclear_repulsion=enuc,
        xc_energy=xc_energy,
        exact_exchange_fraction=alpha,
        mo_energy=mo_energy,
        mo_coeff=mo_coeff,
        mo_occ=mo_occ_fixed,
        density_matrix=density,
        fock_matrix=fock,
        overlap_matrix=s,
        hcore_matrix=h,
        cycles=cycles,
    )
