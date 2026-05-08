from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import ctypes
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import inspect
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..data.basis import CartesianBasis


_FFI_TARGET_NAME = "td_graddft_cuda_direct_jk"
_JOLTQC_DIRECT_JK_TARGET_NAME = "td_graddft_cuda_joltqc_direct_jk"
_SCREENED_FFI_TARGET_NAME = "td_graddft_cuda_screened_direct_jk"
_PAIR_SCHWARZ_TARGET_NAME = "td_graddft_cuda_pair_schwarz"
_ERI_TARGET_NAME = "td_graddft_cuda_eri_tensor"
_ERI_PAIR_TARGET_NAME = "td_graddft_cuda_eri_pair_matrix"
_PAIR_JK_TARGET_NAME = "td_graddft_cuda_pair_matrix_jk"
_REGISTERED_FFI_LIBS: dict[Path, ctypes.CDLL] = {}
_REGISTERED_FFI_TARGETS: set[str] = set()
_PI = math.pi
_PREBUILT_LIBRARY_ENV = "TD_GRADDFT_CUDA_JK_LIBRARY"
_PAIR_ERI_BUILD_CUTOFF_ENV = "TD_GRADDFT_CUDA_PAIR_ERI_BUILD_CUTOFF"
_NVCC_SPLIT_COMPILE_ENV = "TD_GRADDFT_NVCC_SPLIT_COMPILE"
_NVCC_COMPILE_JOBS_ENV = "TD_GRADDFT_NVCC_COMPILE_JOBS"
_JOLTQC_SPLIT_THRESHOLD_ENV = "TD_GRADDFT_JOLTQC_SPLIT_THRESHOLD"
_JOLTQC_DISPATCH_ENV = "TD_GRADDFT_CUDA_JOLTQC_DISPATCH"
_JOLTQC_FIXED_UNIVERSE_ENV = "TD_GRADDFT_CUDA_JOLTQC_FIXED_UNIVERSE"
_JOLTQC_FIXED_MAX_L_ENV = "TD_GRADDFT_CUDA_JOLTQC_FIXED_MAX_L"
_JOLTQC_FIXED_NPRIM_MAX_ENV = "TD_GRADDFT_CUDA_JOLTQC_FIXED_NPRIM_MAX"
_GPU4PYSCF_RYS_TARGET_NAME = "td_graddft_cuda_gpu4pyscf_rys_direct_jk"
_GPU4PYSCF_RYS_ENV = "TD_GRADDFT_ENABLE_GPU4PYSCF_RYS"
_DEFAULT_PAIR_ERI_BUILD_CUTOFF = 1.0e-8
_JOLTQC_GROUP_ALIGNMENT = 4
_JOLTQC_NPRIM_MAX = 3
_JOLTQC_BASIS_STRIDE = 12
_GPU4PYSCF_PAIR_MAPPING_TILE = 6
_RYS_CHARGE_OF = 0
_RYS_PTR_COORD = 1
_RYS_ATM_SLOTS = 6
_RYS_ATOM_OF = 0
_RYS_ANG_OF = 1
_RYS_NPRIM_OF = 2
_RYS_NCTR_OF = 3
_RYS_KAPPA_OF = 4
_RYS_PTR_EXP = 5
_RYS_PTR_COEFF = 6
_RYS_PTR_BAS_COORD = 7
_RYS_BAS_SLOTS = 8
_RYS_PTR_RANGE_OMEGA = 8


def cuda_ffi_available() -> bool:
    """Return whether CUDA FFI kernels can be used in this process."""

    if os.environ.get("TD_GRADDFT_DISABLE_CUDA_FFI", "").lower() in {"1", "true", "yes", "on"}:
        return False
    prebuilt = os.environ.get(_PREBUILT_LIBRARY_ENV)
    has_prebuilt = bool(prebuilt) and Path(prebuilt).expanduser().exists()
    if prebuilt and not has_prebuilt:
        return False
    has_packaged = _any_packaged_prebuilt_library() is not None
    if (
        not has_prebuilt
        and not has_packaged
        and os.environ.get("TD_GRADDFT_NVCC") is None
        and shutil.which("nvcc") is None
    ):
        return False
    try:
        return bool(jax.devices("gpu"))
    except Exception:
        return False


def _cuda_pair_eri_build_cutoff() -> float:
    raw = os.environ.get(_PAIR_ERI_BUILD_CUTOFF_ENV, "")
    if str(raw).strip() == "":
        return _DEFAULT_PAIR_ERI_BUILD_CUTOFF
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def _ffi_module():
    try:
        from jax import ffi
    except ImportError:
        from jax.extend import ffi
    return ffi


def _pool_pair_schwarz_by_screen_group(
    pair_schwarz: Array,
    group_ids: Array,
    num_groups: int,
) -> Array:
    """Use shell-pair/block max Schwarz bounds for AO-pair screening."""

    values = jnp.asarray(pair_schwarz, dtype=jnp.float64)
    if int(num_groups) <= 0:
        return values
    groups = jnp.asarray(group_ids, dtype=jnp.int32)
    group_max = jax.ops.segment_max(values, groups, num_segments=int(num_groups))
    return group_max[groups]


def _symmetrize_density_like(value: Array) -> Array:
    arr = jnp.asarray(value, dtype=jnp.float64)
    return 0.5 * (arr + arr.T)


def _ffi_call(target_name, result_shape_dtypes, *args):
    ffi = _ffi_module()
    parameters = inspect.signature(ffi.ffi_call).parameters
    positional_args = parameters.get("args")
    if positional_args is not None and positional_args.kind is inspect.Parameter.VAR_POSITIONAL:
        return ffi.ffi_call(
            target_name,
            result_shape_dtypes,
            *args,
            vectorized=False,
            has_side_effect=False,
        )
    call = ffi.ffi_call(
        target_name,
        result_shape_dtypes,
        has_side_effect=False,
        vmap_method="sequential",
    )
    return call(*args)


def _cuda_pair_matrix_jk_primitive(
    eri_pair_matrix: Array,
    density: Array,
    pair_rows: Array,
    pair_cols: Array,
) -> tuple[Array, Array]:
    density_arr = _symmetrize_density_like(density)
    pair_arr = jnp.asarray(eri_pair_matrix, dtype=jnp.float64)
    shape = jax.ShapeDtypeStruct(density_arr.shape, jnp.float64)
    j_mat, k_mat = _ffi_call(
        _PAIR_JK_TARGET_NAME,
        (shape, shape),
        pair_arr,
        density_arr,
        pair_rows,
        pair_cols,
    )
    dtype = jnp.asarray(density).dtype
    return jnp.asarray(j_mat, dtype=dtype), jnp.asarray(k_mat, dtype=dtype)


@jax.custom_vjp
def _cuda_pair_matrix_jk_with_density_vjp(
    eri_pair_matrix: Array,
    density: Array,
    pair_rows: Array,
    pair_cols: Array,
) -> tuple[Array, Array]:
    return _cuda_pair_matrix_jk_primitive(
        eri_pair_matrix,
        density,
        pair_rows,
        pair_cols,
    )


def _cuda_pair_matrix_jk_fwd(
    eri_pair_matrix: Array,
    density: Array,
    pair_rows: Array,
    pair_cols: Array,
):
    out = _cuda_pair_matrix_jk_primitive(
        eri_pair_matrix,
        density,
        pair_rows,
        pair_cols,
    )
    density_template = jnp.zeros_like(jnp.asarray(density))
    return out, (jnp.asarray(eri_pair_matrix), pair_rows, pair_cols, density_template)


def _cuda_pair_matrix_jk_bwd(res, cotangents):
    eri_pair_matrix, pair_rows, pair_cols, density_template = res
    j_cotangent, k_cotangent = cotangents
    if j_cotangent is None:
        j_cotangent = density_template
    if k_cotangent is None:
        k_cotangent = density_template
    grad_j, _ = _cuda_pair_matrix_jk_primitive(
        eri_pair_matrix,
        _symmetrize_density_like(j_cotangent),
        pair_rows,
        pair_cols,
    )
    _, grad_k = _cuda_pair_matrix_jk_primitive(
        eri_pair_matrix,
        _symmetrize_density_like(k_cotangent),
        pair_rows,
        pair_cols,
    )
    grad_density = _symmetrize_density_like(grad_j + grad_k).astype(
        jnp.asarray(density_template).dtype
    )
    return (
        jnp.zeros_like(eri_pair_matrix),
        grad_density,
        None,
        None,
    )


_cuda_pair_matrix_jk_with_density_vjp.defvjp(
    _cuda_pair_matrix_jk_fwd,
    _cuda_pair_matrix_jk_bwd,
)


def _pair_matrix_jk_jax_reference(
    eri_pair_matrix: Array,
    density: Array,
    pair_rows: Array,
    pair_cols: Array,
) -> tuple[Array, Array]:
    pair = jnp.asarray(eri_pair_matrix)
    density_arr = _symmetrize_density_like(density)
    nao = int(density_arr.shape[0])
    pair_ids = jnp.arange(pair_rows.shape[0], dtype=jnp.int32)
    pair_index = jnp.zeros((nao, nao), dtype=jnp.int32)
    pair_index = pair_index.at[pair_rows, pair_cols].set(pair_ids, unique_indices=True)
    pair_index = pair_index.at[pair_cols, pair_rows].set(pair_ids, unique_indices=True)
    multiplicity = jnp.where(pair_rows == pair_cols, 1.0, 2.0).astype(density_arr.dtype)

    density_pair = density_arr[pair_rows, pair_cols] * multiplicity
    j_pair = pair @ density_pair
    j_mat = jnp.zeros_like(density_arr)
    j_mat = j_mat.at[pair_rows, pair_cols].set(j_pair, unique_indices=True)
    j_mat = j_mat.at[pair_cols, pair_rows].set(j_pair, unique_indices=True)

    ao = jnp.arange(nao, dtype=jnp.int32)
    qs_by_q = pair_index[:, ao]

    def _k_row(p: Array) -> Array:
        pr = pair_index[p, ao]
        blocks = pair[pr[None, :, None], qs_by_q[:, None, :]]
        return jnp.einsum("qrs,rs->q", blocks, density_arr)

    k_mat = jax.vmap(_k_row)(ao)
    return 0.5 * (j_mat + j_mat.T), 0.5 * (k_mat + k_mat.T)


@jax.custom_jvp
def _cuda_pair_matrix_jk_with_jax_jvp(
    eri_pair_matrix: Array,
    density: Array,
    pair_rows: Array,
    pair_cols: Array,
) -> tuple[Array, Array]:
    return _cuda_pair_matrix_jk_primitive(
        eri_pair_matrix,
        density,
        pair_rows,
        pair_cols,
    )


@_cuda_pair_matrix_jk_with_jax_jvp.defjvp
def _cuda_pair_matrix_jk_with_jax_jvp_rule(primals, tangents):
    eri_pair_matrix, density, pair_rows, pair_cols = primals
    _, density_dot, _, _ = tangents
    primal_out = _cuda_pair_matrix_jk_primitive(
        eri_pair_matrix,
        density,
        pair_rows,
        pair_cols,
    )
    tangent_out = _pair_matrix_jk_jax_reference(
        eri_pair_matrix,
        density_dot,
        pair_rows,
        pair_cols,
    )
    return primal_out, tangent_out


def _register_cuda_pair_matrix_jk_library(library: str | os.PathLike[str]) -> None:
    """Register only the packed AO-pair J/K FFI target.

    This target depends only on the AO-pair matrix and the lower-triangle
    pair metadata, so it can be used by packed-ERI code paths that do not
    carry a full ``CudaDirectJKBuilder`` or basis object.
    """

    library_path = Path(library).expanduser().resolve()
    lib = ctypes.CDLL(str(library_path))
    ffi = _ffi_module()
    if _PAIR_JK_TARGET_NAME not in _REGISTERED_FFI_TARGETS:
        ffi.register_ffi_target(
            _PAIR_JK_TARGET_NAME,
            ffi.pycapsule(getattr(lib, "TdGraddftCudaPairMatrixJkFfi")),
            platform="CUDA",
            api_version=1,
        )
        _REGISTERED_FFI_TARGETS.add(_PAIR_JK_TARGET_NAME)
    _REGISTERED_FFI_LIBS[library_path] = lib


def ensure_cuda_pair_matrix_jk_ffi_registered(
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    nvcc: str | None = None,
    arch: str = "native",
) -> bool:
    """Ensure the packed AO-pair J/K CUDA FFI target is available."""

    if _PAIR_JK_TARGET_NAME in _REGISTERED_FFI_TARGETS:
        return True
    if not cuda_ffi_available():
        return False

    prebuilt = os.environ.get(_PREBUILT_LIBRARY_ENV)
    if prebuilt:
        library = Path(prebuilt).expanduser()
        if not library.exists():
            return False
        _register_cuda_pair_matrix_jk_library(library)
        return True

    compiler = nvcc or os.environ.get("TD_GRADDFT_NVCC") or shutil.which("nvcc")
    arch_name = _detect_cuda_arch() if str(arch) == "native" else str(arch)
    packaged = _packaged_prebuilt_library_path(arch_name)
    if packaged is None:
        packaged = _any_packaged_prebuilt_library()
    if packaged is not None and compiler is None:
        _register_cuda_pair_matrix_jk_library(packaged)
        return True

    if compiler is None:
        return False

    library = build_prebuilt_cuda_direct_jk_library(
        cache_dir
        or os.environ.get("TD_GRADDFT_CUDA_JK_CACHE", "")
        or (Path(tempfile.gettempdir()) / "td_graddft_cuda_direct_jk"),
        nvcc=compiler,
        arch=arch_name,
    )
    _register_cuda_pair_matrix_jk_library(library)
    return True


def build_jk_from_eri_pair_matrix_cuda(
    eri_pair_matrix: Array,
    density: Array,
    pair_rows: Array,
    pair_cols: Array,
) -> tuple[Array, Array]:
    """Build J/K from packed AO-pair ERIs using the CUDA FFI contraction."""

    if not ensure_cuda_pair_matrix_jk_ffi_registered():
        raise RuntimeError("CUDA pair-matrix J/K FFI target is not available.")
    return _cuda_pair_matrix_jk_with_jax_jvp(
        eri_pair_matrix,
        density,
        pair_rows,
        pair_cols,
    )


def _cuda_direct_jk_primitive(
    density: Array,
    centers: Array,
    angulars: Array,
    pair_exponents: Array,
    pair_centers: Array,
    pair_prefactors: Array,
    pair_nprims: Array,
    pair_rows: Array,
    pair_cols: Array,
    pair_schwarz: Array,
    cutoff_arr: Array,
) -> tuple[Array, Array]:
    density_arr = _symmetrize_density_like(density)
    shape = jax.ShapeDtypeStruct(density_arr.shape, jnp.float64)
    j_mat, k_mat = _ffi_call(
        _FFI_TARGET_NAME,
        (shape, shape),
        density_arr,
        centers,
        angulars,
        pair_exponents,
        pair_centers,
        pair_prefactors,
        pair_nprims,
        pair_rows,
        pair_cols,
        pair_schwarz,
        cutoff_arr,
    )
    dtype = jnp.asarray(density).dtype
    return jnp.asarray(j_mat, dtype=dtype), jnp.asarray(k_mat, dtype=dtype)


@jax.custom_vjp
def _cuda_direct_jk_with_density_vjp(
    density: Array,
    centers: Array,
    angulars: Array,
    pair_exponents: Array,
    pair_centers: Array,
    pair_prefactors: Array,
    pair_nprims: Array,
    pair_rows: Array,
    pair_cols: Array,
    pair_schwarz: Array,
    cutoff_arr: Array,
) -> tuple[Array, Array]:
    return _cuda_direct_jk_primitive(
        density,
        centers,
        angulars,
        pair_exponents,
        pair_centers,
        pair_prefactors,
        pair_nprims,
        pair_rows,
        pair_cols,
        pair_schwarz,
        cutoff_arr,
    )


def _cuda_direct_jk_fwd(
    density: Array,
    centers: Array,
    angulars: Array,
    pair_exponents: Array,
    pair_centers: Array,
    pair_prefactors: Array,
    pair_nprims: Array,
    pair_rows: Array,
    pair_cols: Array,
    pair_schwarz: Array,
    cutoff_arr: Array,
):
    out = _cuda_direct_jk_primitive(
        density,
        centers,
        angulars,
        pair_exponents,
        pair_centers,
        pair_prefactors,
        pair_nprims,
        pair_rows,
        pair_cols,
        pair_schwarz,
        cutoff_arr,
    )
    density_template = jnp.zeros_like(jnp.asarray(density))
    return (
        out,
        (
            density_template,
            centers,
            angulars,
            pair_exponents,
            pair_centers,
            pair_prefactors,
            pair_nprims,
            pair_rows,
            pair_cols,
            pair_schwarz,
            cutoff_arr,
        ),
    )


def _cuda_direct_jk_bwd(res, cotangents):
    (
        density_template,
        centers,
        angulars,
        pair_exponents,
        pair_centers,
        pair_prefactors,
        pair_nprims,
        pair_rows,
        pair_cols,
        pair_schwarz,
        cutoff_arr,
    ) = res
    j_cotangent, k_cotangent = cotangents
    if j_cotangent is None:
        j_cotangent = density_template
    if k_cotangent is None:
        k_cotangent = density_template
    grad_j, _ = _cuda_direct_jk_primitive(
        _symmetrize_density_like(j_cotangent),
        centers,
        angulars,
        pair_exponents,
        pair_centers,
        pair_prefactors,
        pair_nprims,
        pair_rows,
        pair_cols,
        pair_schwarz,
        cutoff_arr,
    )
    _, grad_k = _cuda_direct_jk_primitive(
        _symmetrize_density_like(k_cotangent),
        centers,
        angulars,
        pair_exponents,
        pair_centers,
        pair_prefactors,
        pair_nprims,
        pair_rows,
        pair_cols,
        pair_schwarz,
        cutoff_arr,
    )
    grad_density = _symmetrize_density_like(grad_j + grad_k).astype(
        jnp.asarray(density_template).dtype
    )
    return (
        grad_density,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_cuda_direct_jk_with_density_vjp.defvjp(
    _cuda_direct_jk_fwd,
    _cuda_direct_jk_bwd,
)


def _cuda_gpu4pyscf_rys_direct_jk_primitive(
    density: Array,
    rys_atm: Array,
    rys_bas: Array,
    rys_env: Array,
    rys_ao_loc: Array,
    rys_ao_to_parent_ao: Array,
    rys_group_offsets: Array,
    rys_q_cond: Array,
    rys_dm_cond: Array,
    rys_pair_offsets: Array,
    rys_pair_ids: Array,
    log_cutoff: Array,
) -> tuple[Array, Array]:
    density_arr = _symmetrize_density_like(density)
    shape = jax.ShapeDtypeStruct(density_arr.shape, jnp.float64)
    j_mat, k_mat = _ffi_call(
        _GPU4PYSCF_RYS_TARGET_NAME,
        (shape, shape),
        density_arr,
        rys_atm,
        rys_bas,
        rys_env,
        rys_ao_loc,
        rys_ao_to_parent_ao,
        rys_group_offsets,
        rys_q_cond,
        rys_dm_cond,
        rys_pair_offsets,
        rys_pair_ids,
        log_cutoff,
    )
    dtype = jnp.asarray(density).dtype
    return jnp.asarray(j_mat, dtype=dtype), jnp.asarray(k_mat, dtype=dtype)


@jax.custom_vjp
def _cuda_gpu4pyscf_rys_direct_jk_with_density_vjp(
    density: Array,
    rys_atm: Array,
    rys_bas: Array,
    rys_env: Array,
    rys_ao_loc: Array,
    rys_ao_to_parent_ao: Array,
    rys_group_offsets: Array,
    rys_q_cond: Array,
    rys_dm_cond: Array,
    rys_pair_offsets: Array,
    rys_pair_ids: Array,
    log_cutoff: Array,
) -> tuple[Array, Array]:
    return _cuda_gpu4pyscf_rys_direct_jk_primitive(
        density,
        rys_atm,
        rys_bas,
        rys_env,
        rys_ao_loc,
        rys_ao_to_parent_ao,
        rys_group_offsets,
        rys_q_cond,
        rys_dm_cond,
        rys_pair_offsets,
        rys_pair_ids,
        log_cutoff,
    )


def _cuda_gpu4pyscf_rys_direct_jk_fwd(
    density: Array,
    rys_atm: Array,
    rys_bas: Array,
    rys_env: Array,
    rys_ao_loc: Array,
    rys_ao_to_parent_ao: Array,
    rys_group_offsets: Array,
    rys_q_cond: Array,
    rys_dm_cond: Array,
    rys_pair_offsets: Array,
    rys_pair_ids: Array,
    log_cutoff: Array,
):
    out = _cuda_gpu4pyscf_rys_direct_jk_primitive(
        density,
        rys_atm,
        rys_bas,
        rys_env,
        rys_ao_loc,
        rys_ao_to_parent_ao,
        rys_group_offsets,
        rys_q_cond,
        rys_dm_cond,
        rys_pair_offsets,
        rys_pair_ids,
        log_cutoff,
    )
    density_template = jnp.zeros_like(jnp.asarray(density))
    return (
        out,
        (
            density_template,
            rys_atm,
            rys_bas,
            rys_env,
            rys_ao_loc,
            rys_ao_to_parent_ao,
            rys_group_offsets,
            rys_q_cond,
            rys_dm_cond,
            rys_pair_offsets,
            rys_pair_ids,
            log_cutoff,
        ),
    )


def _cuda_gpu4pyscf_rys_direct_jk_bwd(res, cotangents):
    (
        density_template,
        rys_atm,
        rys_bas,
        rys_env,
        rys_ao_loc,
        rys_ao_to_parent_ao,
        rys_group_offsets,
        rys_q_cond,
        rys_dm_cond,
        rys_pair_offsets,
        rys_pair_ids,
        log_cutoff,
    ) = res
    j_cotangent, k_cotangent = cotangents
    if j_cotangent is None:
        j_cotangent = density_template
    if k_cotangent is None:
        k_cotangent = density_template
    grad_j, _ = _cuda_gpu4pyscf_rys_direct_jk_primitive(
        _symmetrize_density_like(j_cotangent),
        rys_atm,
        rys_bas,
        rys_env,
        rys_ao_loc,
        rys_ao_to_parent_ao,
        rys_group_offsets,
        rys_q_cond,
        rys_dm_cond,
        rys_pair_offsets,
        rys_pair_ids,
        log_cutoff,
    )
    _, grad_k = _cuda_gpu4pyscf_rys_direct_jk_primitive(
        _symmetrize_density_like(k_cotangent),
        rys_atm,
        rys_bas,
        rys_env,
        rys_ao_loc,
        rys_ao_to_parent_ao,
        rys_group_offsets,
        rys_q_cond,
        rys_dm_cond,
        rys_pair_offsets,
        rys_pair_ids,
        log_cutoff,
    )
    grad_density = _symmetrize_density_like(grad_j + grad_k).astype(
        jnp.asarray(density_template).dtype
    )
    return (
        grad_density,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_cuda_gpu4pyscf_rys_direct_jk_with_density_vjp.defvjp(
    _cuda_gpu4pyscf_rys_direct_jk_fwd,
    _cuda_gpu4pyscf_rys_direct_jk_bwd,
)


def _cuda_screened_direct_jk_primitive(
    density: Array,
    centers: Array,
    angulars: Array,
    pair_exponents: Array,
    pair_centers: Array,
    pair_prefactors: Array,
    pair_nprims: Array,
    pair_rows: Array,
    pair_cols: Array,
    pair_schwarz: Array,
    cutoff_arr: Array,
    shell_log_q_matrix: Array,
    shell_dm_cond: Array,
    shell_ao_indices_padded: Array,
    shell_ao_sizes: Array,
    tile_shell_indices: Array,
    tile_shell_pad_mask: Array,
    tile_pair_ids: Array,
) -> tuple[Array, Array]:
    density_arr = _symmetrize_density_like(density)
    shape = jax.ShapeDtypeStruct(density_arr.shape, jnp.float64)
    j_mat, k_mat = _ffi_call(
        _SCREENED_FFI_TARGET_NAME,
        (shape, shape),
        density_arr,
        centers,
        angulars,
        pair_exponents,
        pair_centers,
        pair_prefactors,
        pair_nprims,
        pair_rows,
        pair_cols,
        pair_schwarz,
        cutoff_arr,
        shell_log_q_matrix,
        shell_dm_cond,
        shell_ao_indices_padded,
        shell_ao_sizes,
        tile_shell_indices,
        tile_shell_pad_mask,
        tile_pair_ids,
    )
    dtype = jnp.asarray(density).dtype
    return jnp.asarray(j_mat, dtype=dtype), jnp.asarray(k_mat, dtype=dtype)


@jax.custom_vjp
def _cuda_screened_direct_jk_with_density_vjp(
    density: Array,
    centers: Array,
    angulars: Array,
    pair_exponents: Array,
    pair_centers: Array,
    pair_prefactors: Array,
    pair_nprims: Array,
    pair_rows: Array,
    pair_cols: Array,
    pair_schwarz: Array,
    cutoff_arr: Array,
    shell_log_q_matrix: Array,
    shell_dm_cond: Array,
    shell_ao_indices_padded: Array,
    shell_ao_sizes: Array,
    tile_shell_indices: Array,
    tile_shell_pad_mask: Array,
    tile_pair_ids: Array,
) -> tuple[Array, Array]:
    return _cuda_screened_direct_jk_primitive(
        density,
        centers,
        angulars,
        pair_exponents,
        pair_centers,
        pair_prefactors,
        pair_nprims,
        pair_rows,
        pair_cols,
        pair_schwarz,
        cutoff_arr,
        shell_log_q_matrix,
        shell_dm_cond,
        shell_ao_indices_padded,
        shell_ao_sizes,
        tile_shell_indices,
        tile_shell_pad_mask,
        tile_pair_ids,
    )


def _cuda_screened_direct_jk_fwd(
    density: Array,
    centers: Array,
    angulars: Array,
    pair_exponents: Array,
    pair_centers: Array,
    pair_prefactors: Array,
    pair_nprims: Array,
    pair_rows: Array,
    pair_cols: Array,
    pair_schwarz: Array,
    cutoff_arr: Array,
    shell_log_q_matrix: Array,
    shell_dm_cond: Array,
    shell_ao_indices_padded: Array,
    shell_ao_sizes: Array,
    tile_shell_indices: Array,
    tile_shell_pad_mask: Array,
    tile_pair_ids: Array,
):
    out = _cuda_screened_direct_jk_primitive(
        density,
        centers,
        angulars,
        pair_exponents,
        pair_centers,
        pair_prefactors,
        pair_nprims,
        pair_rows,
        pair_cols,
        pair_schwarz,
        cutoff_arr,
        shell_log_q_matrix,
        shell_dm_cond,
        shell_ao_indices_padded,
        shell_ao_sizes,
        tile_shell_indices,
        tile_shell_pad_mask,
        tile_pair_ids,
    )
    density_template = jnp.zeros_like(jnp.asarray(density))
    return (
        out,
        (
            density_template,
            centers,
            angulars,
            pair_exponents,
            pair_centers,
            pair_prefactors,
            pair_nprims,
            pair_rows,
            pair_cols,
            pair_schwarz,
            cutoff_arr,
            shell_log_q_matrix,
            shell_dm_cond,
            shell_ao_indices_padded,
            shell_ao_sizes,
            tile_shell_indices,
            tile_shell_pad_mask,
            tile_pair_ids,
        ),
    )


def _cuda_screened_direct_jk_bwd(res, cotangents):
    (
        density_template,
        centers,
        angulars,
        pair_exponents,
        pair_centers,
        pair_prefactors,
        pair_nprims,
        pair_rows,
        pair_cols,
        pair_schwarz,
        cutoff_arr,
        shell_log_q_matrix,
        shell_dm_cond,
        shell_ao_indices_padded,
        shell_ao_sizes,
        tile_shell_indices,
        tile_shell_pad_mask,
        tile_pair_ids,
    ) = res
    j_cotangent, k_cotangent = cotangents
    if j_cotangent is None:
        j_cotangent = density_template
    if k_cotangent is None:
        k_cotangent = density_template
    grad_j, _ = _cuda_screened_direct_jk_primitive(
        _symmetrize_density_like(j_cotangent),
        centers,
        angulars,
        pair_exponents,
        pair_centers,
        pair_prefactors,
        pair_nprims,
        pair_rows,
        pair_cols,
        pair_schwarz,
        cutoff_arr,
        shell_log_q_matrix,
        shell_dm_cond,
        shell_ao_indices_padded,
        shell_ao_sizes,
        tile_shell_indices,
        tile_shell_pad_mask,
        tile_pair_ids,
    )
    _, grad_k = _cuda_screened_direct_jk_primitive(
        _symmetrize_density_like(k_cotangent),
        centers,
        angulars,
        pair_exponents,
        pair_centers,
        pair_prefactors,
        pair_nprims,
        pair_rows,
        pair_cols,
        pair_schwarz,
        cutoff_arr,
        shell_log_q_matrix,
        shell_dm_cond,
        shell_ao_indices_padded,
        shell_ao_sizes,
        tile_shell_indices,
        tile_shell_pad_mask,
        tile_pair_ids,
    )
    grad_density = _symmetrize_density_like(grad_j + grad_k).astype(
        jnp.asarray(density_template).dtype
    )
    return (
        grad_density,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_cuda_screened_direct_jk_with_density_vjp.defvjp(
    _cuda_screened_direct_jk_fwd,
    _cuda_screened_direct_jk_bwd,
)


def _cuda_joltqc_direct_jk_primitive(
    density: Array,
    basis_data: Array,
    shell_l: Array,
    shell_nprims: Array,
    ao_to_parent_ao: Array,
    group_keys: Array,
    group_quartet_keys: Array,
    group_quartet_offsets: Array,
    shell_quartets: Array,
) -> tuple[Array, Array]:
    density_arr = _symmetrize_density_like(density)
    shape = jax.ShapeDtypeStruct(density_arr.shape, jnp.float64)
    j_mat, k_mat = _ffi_call(
        _JOLTQC_DIRECT_JK_TARGET_NAME,
        (shape, shape),
        density_arr,
        basis_data,
        shell_l,
        shell_nprims,
        ao_to_parent_ao,
        group_keys,
        group_quartet_keys,
        group_quartet_offsets,
        shell_quartets,
    )
    dtype = jnp.asarray(density).dtype
    return jnp.asarray(j_mat, dtype=dtype), jnp.asarray(k_mat, dtype=dtype)


@jax.custom_vjp
def _cuda_joltqc_direct_jk_with_density_vjp(
    density: Array,
    basis_data: Array,
    shell_l: Array,
    shell_nprims: Array,
    ao_to_parent_ao: Array,
    group_keys: Array,
    group_quartet_keys: Array,
    group_quartet_offsets: Array,
    shell_quartets: Array,
) -> tuple[Array, Array]:
    return _cuda_joltqc_direct_jk_primitive(
        density,
        basis_data,
        shell_l,
        shell_nprims,
        ao_to_parent_ao,
        group_keys,
        group_quartet_keys,
        group_quartet_offsets,
        shell_quartets,
    )


def _cuda_joltqc_direct_jk_fwd(
    density: Array,
    basis_data: Array,
    shell_l: Array,
    shell_nprims: Array,
    ao_to_parent_ao: Array,
    group_keys: Array,
    group_quartet_keys: Array,
    group_quartet_offsets: Array,
    shell_quartets: Array,
):
    out = _cuda_joltqc_direct_jk_primitive(
        density,
        basis_data,
        shell_l,
        shell_nprims,
        ao_to_parent_ao,
        group_keys,
        group_quartet_keys,
        group_quartet_offsets,
        shell_quartets,
    )
    density_template = jnp.zeros_like(jnp.asarray(density))
    return (
        out,
        (
            density_template,
            basis_data,
            shell_l,
            shell_nprims,
            ao_to_parent_ao,
            group_keys,
            group_quartet_keys,
            group_quartet_offsets,
            shell_quartets,
        ),
    )


def _cuda_joltqc_direct_jk_bwd(res, cotangents):
    (
        density_template,
        basis_data,
        shell_l,
        shell_nprims,
        ao_to_parent_ao,
        group_keys,
        group_quartet_keys,
        group_quartet_offsets,
        shell_quartets,
    ) = res
    j_cotangent, k_cotangent = cotangents
    if j_cotangent is None:
        j_cotangent = density_template
    if k_cotangent is None:
        k_cotangent = density_template
    grad_j, _ = _cuda_joltqc_direct_jk_primitive(
        _symmetrize_density_like(j_cotangent),
        basis_data,
        shell_l,
        shell_nprims,
        ao_to_parent_ao,
        group_keys,
        group_quartet_keys,
        group_quartet_offsets,
        shell_quartets,
    )
    _, grad_k = _cuda_joltqc_direct_jk_primitive(
        _symmetrize_density_like(k_cotangent),
        basis_data,
        shell_l,
        shell_nprims,
        ao_to_parent_ao,
        group_keys,
        group_quartet_keys,
        group_quartet_offsets,
        shell_quartets,
    )
    grad_density = _symmetrize_density_like(grad_j + grad_k).astype(
        jnp.asarray(density_template).dtype
    )
    return (
        grad_density,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_cuda_joltqc_direct_jk_with_density_vjp.defvjp(
    _cuda_joltqc_direct_jk_fwd,
    _cuda_joltqc_direct_jk_bwd,
)


@dataclass(frozen=True)
class CudaShellLayout:
    shell_l: np.ndarray
    shell_nprims: np.ndarray
    sorted_shell_indices: np.ndarray
    group_keys: np.ndarray
    group_offsets: np.ndarray
    padded_sorted_shell_indices: np.ndarray
    padded_shell_pad_mask: np.ndarray
    padded_group_offsets: np.ndarray
    tile_size: int
    tile_shell_indices: np.ndarray
    tile_shell_pad_mask: np.ndarray
    shell_pair_group_matrix: np.ndarray


@dataclass(frozen=True)
class CudaJoltQCBasisLayout:
    basis_data: np.ndarray
    basis_data_fp32: np.ndarray
    shell_l: np.ndarray
    shell_nprims: np.ndarray
    to_parent_shell: np.ndarray
    primitive_starts: np.ndarray
    pad_mask: np.ndarray
    group_keys: np.ndarray
    group_offsets: np.ndarray
    ao_loc: np.ndarray
    ao_to_parent_ao: np.ndarray


@dataclass(frozen=True)
class CudaJoltQCQuartetLayout:
    group_quartet_keys: np.ndarray
    group_quartet_offsets: np.ndarray
    shell_quartets: np.ndarray


@dataclass(frozen=True)
class CudaRysEnvLayout:
    atm: np.ndarray
    bas: np.ndarray
    env: np.ndarray
    ao_loc: np.ndarray
    sorted_shell_indices: np.ndarray
    shell_to_rys_shell: np.ndarray
    ao_to_parent_ao: np.ndarray
    parent_ao_to_ao: np.ndarray
    group_keys: np.ndarray
    group_offsets: np.ndarray


@dataclass(frozen=True)
class CudaAOSystem:
    centers: np.ndarray
    angulars: np.ndarray
    exponents: np.ndarray
    coefficients: np.ndarray
    nprims: np.ndarray
    pair_exponents: np.ndarray
    pair_centers: np.ndarray
    pair_prefactors: np.ndarray
    pair_nprims: np.ndarray
    pair_rows: np.ndarray
    pair_cols: np.ndarray
    pair_screen_group_ids: np.ndarray
    pair_screen_representative_ids: np.ndarray
    n_pair_screen_groups: int
    shell_ao_indices_padded: np.ndarray
    shell_ao_sizes: np.ndarray
    shell_layout: CudaShellLayout
    joltqc_basis_layout: CudaJoltQCBasisLayout
    joltqc_quartet_layout: CudaJoltQCQuartetLayout

    @property
    def nao(self) -> int:
        return int(self.centers.shape[0])

    @property
    def max_nprim(self) -> int:
        return int(self.exponents.shape[1])


@dataclass(frozen=True)
class CudaTilePairLayout:
    group_pair_keys: np.ndarray
    group_pair_offsets: np.ndarray
    tile_pair_ids: np.ndarray


@dataclass(frozen=True)
class CudaRysPairMappingLayout:
    group_pair_keys: np.ndarray
    group_pair_offsets: np.ndarray
    pair_ids: np.ndarray


def _primitive_cart_norm(exponents: np.ndarray, angular: tuple[int, int, int]) -> np.ndarray:
    alpha = np.asarray(exponents, dtype=np.float64)
    ltot = int(sum(int(power) for power in angular))
    prefactor = (2.0 * alpha / _PI) ** 0.75
    if ltot == 0:
        return prefactor
    if ltot == 1:
        return prefactor * np.sqrt(4.0 * alpha)
    factorial_l_plus_1 = math.factorial(ltot + 1)
    factorial_2l_plus_2 = math.factorial(2 * ltot + 2)
    numerator = (2.0 ** (2 * ltot + 3)) * float(factorial_l_plus_1)
    denominator = float(factorial_2l_plus_2) * math.sqrt(_PI)
    return np.sqrt(numerator * (2.0 * alpha) ** (ltot + 1.5) / denominator)


def _rys_radial_gto_norm(l: int, exponents: np.ndarray) -> np.ndarray:
    """Reconstruct PySCF's mol._env contraction coefficient convention."""

    alpha = np.asarray(exponents, dtype=np.float64)
    n1 = float(2 * int(l) + 3) * 0.5
    gaussian_int = math.gamma(n1) / (2.0 * (2.0 * alpha) ** n1)
    return 1.0 / np.sqrt(gaussian_int)


def _rys_env_coefficients(
    l: int,
    exponents: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Match GPU4PySCF's Rys env coefficients from public bas_ctr_coeff values."""

    values = (
        np.asarray(coefficients, dtype=np.float64).reshape(-1)
        * _rys_radial_gto_norm(int(l), np.asarray(exponents, dtype=np.float64))
    )
    if int(l) < 2:
        values = values * math.sqrt((2 * int(l) + 1) / (4.0 * _PI))
    return values


def _build_joltqc_shell_layout(basis: CartesianBasis) -> CudaShellLayout:
    shells = tuple(basis.shells)
    nshell = len(shells)
    if nshell == 0:
        return CudaShellLayout(
            shell_l=np.zeros((0,), dtype=np.int32),
            shell_nprims=np.zeros((0,), dtype=np.int32),
            sorted_shell_indices=np.zeros((0,), dtype=np.int32),
            group_keys=np.zeros((0, 2), dtype=np.int32),
            group_offsets=np.zeros((1,), dtype=np.int32),
            padded_sorted_shell_indices=np.zeros((0,), dtype=np.int32),
            padded_shell_pad_mask=np.zeros((0,), dtype=bool),
            padded_group_offsets=np.zeros((1,), dtype=np.int32),
            tile_size=_JOLTQC_GROUP_ALIGNMENT,
            tile_shell_indices=np.zeros((0, _JOLTQC_GROUP_ALIGNMENT), dtype=np.int32),
            tile_shell_pad_mask=np.zeros((0, _JOLTQC_GROUP_ALIGNMENT), dtype=bool),
            shell_pair_group_matrix=np.zeros((0, 0), dtype=np.int32),
        )

    shell_l = np.asarray(
        [
            sum(int(power) for power in shell.angulars[0]) if shell.angulars else 0
            for shell in shells
        ],
        dtype=np.int32,
    )
    shell_nprims = np.asarray(
        [int(np.asarray(shell.exponents).shape[0]) for shell in shells],
        dtype=np.int32,
    )
    sorted_shell_indices = np.asarray(
        sorted(
            range(nshell),
            key=lambda idx: (int(shell_l[idx]), -int(shell_nprims[idx]), int(idx)),
        ),
        dtype=np.int32,
    )

    group_keys: list[tuple[int, int]] = []
    group_offsets: list[int] = []
    last_key: tuple[int, int] | None = None
    for offset, shell_idx in enumerate(sorted_shell_indices.tolist()):
        key = (int(shell_l[shell_idx]), int(shell_nprims[shell_idx]))
        if key != last_key:
            group_keys.append(key)
            group_offsets.append(offset)
            last_key = key
    group_offsets.append(nshell)

    padded_sorted_shell_indices: list[int] = []
    padded_shell_pad_mask: list[bool] = []
    padded_group_offsets: list[int] = [0]
    for group_id in range(len(group_keys)):
        start = group_offsets[group_id]
        stop = group_offsets[group_id + 1]
        segment = sorted_shell_indices[start:stop].tolist()
        padded_sorted_shell_indices.extend(segment)
        padded_shell_pad_mask.extend([False] * len(segment))
        padding = (-len(segment)) % _JOLTQC_GROUP_ALIGNMENT
        if padding > 0 and segment:
            padded_sorted_shell_indices.extend([segment[0]] * padding)
            padded_shell_pad_mask.extend([True] * padding)
        padded_group_offsets.append(len(padded_sorted_shell_indices))

    padded_sorted_shell_indices_arr = np.asarray(padded_sorted_shell_indices, dtype=np.int32)
    padded_shell_pad_mask_arr = np.asarray(padded_shell_pad_mask, dtype=bool)
    tile_size = _JOLTQC_GROUP_ALIGNMENT
    ntile = len(padded_sorted_shell_indices) // tile_size

    return CudaShellLayout(
        shell_l=shell_l,
        shell_nprims=shell_nprims,
        sorted_shell_indices=sorted_shell_indices,
        group_keys=np.asarray(group_keys, dtype=np.int32),
        group_offsets=np.asarray(group_offsets, dtype=np.int32),
        padded_sorted_shell_indices=padded_sorted_shell_indices_arr,
        padded_shell_pad_mask=padded_shell_pad_mask_arr,
        padded_group_offsets=np.asarray(padded_group_offsets, dtype=np.int32),
        tile_size=tile_size,
        tile_shell_indices=padded_sorted_shell_indices_arr.reshape(ntile, tile_size),
        tile_shell_pad_mask=padded_shell_pad_mask_arr.reshape(ntile, tile_size),
        shell_pair_group_matrix=np.zeros((nshell, nshell), dtype=np.int32),
    )


def _build_joltqc_basis_layout(basis: CartesianBasis) -> CudaJoltQCBasisLayout:
    shells = tuple(basis.shells)
    if not shells:
        return CudaJoltQCBasisLayout(
            basis_data=np.zeros((0, _JOLTQC_BASIS_STRIDE), dtype=np.float64),
            basis_data_fp32=np.zeros((0, _JOLTQC_BASIS_STRIDE), dtype=np.float32),
            shell_l=np.zeros((0,), dtype=np.int32),
            shell_nprims=np.zeros((0,), dtype=np.int32),
            to_parent_shell=np.zeros((0,), dtype=np.int32),
            primitive_starts=np.zeros((0,), dtype=np.int32),
            pad_mask=np.zeros((0,), dtype=bool),
            group_keys=np.zeros((0, 2), dtype=np.int32),
            group_offsets=np.zeros((1,), dtype=np.int32),
            ao_loc=np.zeros((1,), dtype=np.int32),
            ao_to_parent_ao=np.zeros((0,), dtype=np.int32),
        )

    split_records: list[tuple[int, int, int, int]] = []
    for shell_id, shell in enumerate(shells):
        nprim = int(np.asarray(shell.exponents).shape[0])
        ltot = sum(int(power) for power in shell.angulars[0]) if shell.angulars else 0
        for start in range(0, nprim, _JOLTQC_NPRIM_MAX):
            split_records.append(
                (
                    int(ltot),
                    min(_JOLTQC_NPRIM_MAX, nprim - start),
                    int(shell_id),
                    int(start),
                )
            )

    pattern_records: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for record in split_records:
        key = (record[0], record[1])
        pattern_records.setdefault(key, []).append(record)

    sorted_keys = sorted(pattern_records, key=lambda key: (int(key[0]), -int(key[1])))
    padded_records: list[tuple[int, int, int, int]] = []
    pad_mask: list[bool] = []
    group_offsets: list[int] = [0]
    for key in sorted_keys:
        records = pattern_records[key]
        padded_records.extend(records)
        pad_mask.extend([False] * len(records))
        padding = (-len(records)) % _JOLTQC_GROUP_ALIGNMENT
        if padding:
            padded_records.extend([records[0]] * padding)
            pad_mask.extend([True] * padding)
        group_offsets.append(len(padded_records))

    nbas = len(padded_records)
    shell_l = np.asarray([record[0] for record in padded_records], dtype=np.int32)
    shell_nprims = np.asarray([record[1] for record in padded_records], dtype=np.int32)
    to_parent_shell = np.asarray([record[2] for record in padded_records], dtype=np.int32)
    primitive_starts = np.asarray([record[3] for record in padded_records], dtype=np.int32)
    pad_mask_arr = np.asarray(pad_mask, dtype=bool)
    group_keys = np.asarray(sorted_keys, dtype=np.int32)

    ao_loc = np.zeros((nbas + 1,), dtype=np.int32)
    for row, ltot in enumerate(shell_l.tolist()):
        if pad_mask_arr[row]:
            ao_loc[row + 1] = ao_loc[row]
        else:
            ao_loc[row + 1] = ao_loc[row] + ((int(ltot) + 1) * (int(ltot) + 2)) // 2

    basis_data = np.zeros((nbas, _JOLTQC_BASIS_STRIDE), dtype=np.float64)
    ao_to_parent_ao: list[int] = []
    for row, record in enumerate(padded_records):
        ltot, nprim, shell_id, start = record
        shell = shells[shell_id]
        exponents = np.asarray(shell.exponents, dtype=np.float64)[start : start + nprim]
        coefficients = np.asarray(shell.coefficients, dtype=np.float64)[start : start + nprim]
        coefficients = coefficients * _primitive_cart_norm(
            exponents,
            tuple(int(power) for power in shell.angulars[0]),
        )
        basis_data[row, 0:3] = np.asarray(shell.center, dtype=np.float64)
        basis_data[row, 3] = float(ao_loc[row])
        basis_data[row, 4 : 4 + 2 * nprim : 2] = coefficients
        basis_data[row, 5 : 5 + 2 * nprim : 2] = exponents
        if not pad_mask_arr[row]:
            ao_to_parent_ao.extend(int(value) for value in np.asarray(shell.ao_indices, dtype=np.int32))

    return CudaJoltQCBasisLayout(
        basis_data=np.ascontiguousarray(basis_data, dtype=np.float64),
        basis_data_fp32=np.ascontiguousarray(basis_data, dtype=np.float32),
        shell_l=shell_l,
        shell_nprims=shell_nprims,
        to_parent_shell=to_parent_shell,
        primitive_starts=primitive_starts,
        pad_mask=pad_mask_arr,
        group_keys=group_keys,
        group_offsets=np.asarray(group_offsets, dtype=np.int32),
        ao_loc=ao_loc,
        ao_to_parent_ao=np.asarray(ao_to_parent_ao, dtype=np.int32),
    )


def joltqc_basis_data_from_basis(basis: CartesianBasis) -> np.ndarray:
    """Return JoltQC basis rows for the current geometry."""

    return _build_joltqc_basis_layout(basis).basis_data


def _build_joltqc_quartet_layout(
    basis_layout: CudaJoltQCBasisLayout,
) -> CudaJoltQCQuartetLayout:
    nbas = int(basis_layout.basis_data.shape[0])
    if nbas == 0:
        return CudaJoltQCQuartetLayout(
            group_quartet_keys=np.zeros((0, 4), dtype=np.int32),
            group_quartet_offsets=np.zeros((1,), dtype=np.int32),
            shell_quartets=np.zeros((0, 4), dtype=np.int32),
        )

    shell_to_group = np.zeros((nbas,), dtype=np.int32)
    for group_id in range(int(basis_layout.group_keys.shape[0])):
        start = int(basis_layout.group_offsets[group_id])
        stop = int(basis_layout.group_offsets[group_id + 1])
        shell_to_group[start:stop] = group_id

    nonpad_shells = np.flatnonzero(~np.asarray(basis_layout.pad_mask, dtype=bool)).astype(
        np.int32,
        copy=False,
    )
    if nonpad_shells.size == 0:
        return CudaJoltQCQuartetLayout(
            group_quartet_keys=np.zeros((0, 4), dtype=np.int32),
            group_quartet_offsets=np.zeros((1,), dtype=np.int32),
            shell_quartets=np.zeros((0, 4), dtype=np.int32),
        )

    pair_row_pos, pair_col_pos = np.tril_indices(int(nonpad_shells.size))
    pair_i = nonpad_shells[pair_row_pos]
    pair_j = nonpad_shells[pair_col_pos]
    pair_p_pos, pair_q_pos = np.tril_indices(int(pair_i.size))
    quartets = np.stack(
        (
            pair_i[pair_p_pos],
            pair_j[pair_p_pos],
            pair_i[pair_q_pos],
            pair_j[pair_q_pos],
        ),
        axis=1,
    ).astype(np.int32, copy=False)
    quartet_keys = np.stack(
        (
            shell_to_group[quartets[:, 0]],
            shell_to_group[quartets[:, 1]],
            shell_to_group[quartets[:, 2]],
            shell_to_group[quartets[:, 3]],
        ),
        axis=1,
    ).astype(np.int32, copy=False)
    structured_dtype = np.dtype(
        [
            ("g0", np.int32),
            ("g1", np.int32),
            ("g2", np.int32),
            ("g3", np.int32),
        ]
    )
    structured_keys = np.ascontiguousarray(quartet_keys).view(structured_dtype).reshape(-1)
    order = np.argsort(structured_keys, kind="stable")
    sorted_keys_per_quartet = quartet_keys[order]
    quartets = quartets[order]
    key_starts = np.ones((sorted_keys_per_quartet.shape[0],), dtype=bool)
    key_starts[1:] = np.any(
        sorted_keys_per_quartet[1:] != sorted_keys_per_quartet[:-1],
        axis=1,
    )
    offsets = np.flatnonzero(key_starts).astype(np.int32)
    keys = sorted_keys_per_quartet[offsets]
    offsets = np.concatenate(
        [offsets, np.asarray([quartets.shape[0]], dtype=np.int32)],
        axis=0,
    )
    return CudaJoltQCQuartetLayout(
        group_quartet_keys=np.asarray(keys, dtype=np.int32).reshape(-1, 4),
        group_quartet_offsets=np.asarray(offsets, dtype=np.int32),
        shell_quartets=np.asarray(quartets, dtype=np.int32),
    )


def _build_rys_env_layout(
    basis: CartesianBasis,
    shell_layout: CudaShellLayout | None = None,
) -> CudaRysEnvLayout:
    """Build GPU4PySCF-compatible runtime atm/bas/env/ao_loc buffers."""

    shells = tuple(basis.shells)
    nshell = len(shells)
    if shell_layout is None:
        shell_layout = _build_joltqc_shell_layout(basis)
    sorted_shell_indices = np.asarray(shell_layout.sorted_shell_indices, dtype=np.int32)
    if sorted_shell_indices.size == 0 and nshell > 0:
        sorted_shell_indices = np.arange(nshell, dtype=np.int32)
    if sorted_shell_indices.size != nshell:
        raise ValueError("Rys env shell order must contain each physical shell exactly once.")

    atom_coords = (
        np.asarray(jax.device_get(basis.atom_coords), dtype=np.float64)
        if basis.atom_coords is not None
        else np.zeros((0, 3), dtype=np.float64)
    )
    atom_charges = (
        np.asarray(jax.device_get(basis.atom_charges), dtype=np.float64)
        if basis.atom_charges is not None
        else np.zeros((atom_coords.shape[0],), dtype=np.float64)
    )
    natm = int(atom_coords.shape[0])
    atm = np.zeros((natm, _RYS_ATM_SLOTS), dtype=np.int32)
    env_values: list[float] = [0.0] * (_RYS_PTR_RANGE_OMEGA + 1)
    for atom_idx in range(natm):
        coord_ptr = len(env_values)
        env_values.extend(float(value) for value in atom_coords[atom_idx])
        atm[atom_idx, _RYS_CHARGE_OF] = int(round(float(atom_charges[atom_idx])))
        atm[atom_idx, _RYS_PTR_COORD] = coord_ptr

    bas = np.zeros((nshell, _RYS_BAS_SLOTS), dtype=np.int32)
    ao_loc = np.zeros((nshell + 1,), dtype=np.int32)
    shell_to_rys_shell = np.full((nshell,), -1, dtype=np.int32)
    ao_to_parent_ao: list[int] = []
    parent_ao_to_ao = np.full((int(basis.nao),), -1, dtype=np.int32)

    for rys_shell_id, shell_id in enumerate(sorted_shell_indices.tolist()):
        shell = shells[int(shell_id)]
        center = np.asarray(jax.device_get(shell.center), dtype=np.float64)
        if natm:
            distances = np.linalg.norm(atom_coords - center[None, :], axis=1)
            atom_idx = int(np.argmin(distances))
        else:
            atom_idx = 0
        ltot = sum(int(power) for power in shell.angulars[0]) if shell.angulars else 0
        exponents = np.asarray(jax.device_get(shell.exponents), dtype=np.float64).reshape(-1)
        coefficients = _rys_env_coefficients(
            int(ltot),
            exponents,
            np.asarray(jax.device_get(shell.coefficients), dtype=np.float64).reshape(-1),
        )
        exp_ptr = len(env_values)
        env_values.extend(float(value) for value in exponents)
        coeff_ptr = len(env_values)
        env_values.extend(float(value) for value in coefficients)

        bas[rys_shell_id, _RYS_ATOM_OF] = atom_idx
        bas[rys_shell_id, _RYS_ANG_OF] = int(ltot)
        bas[rys_shell_id, _RYS_NPRIM_OF] = int(exponents.shape[0])
        bas[rys_shell_id, _RYS_NCTR_OF] = 1
        bas[rys_shell_id, _RYS_KAPPA_OF] = 0
        bas[rys_shell_id, _RYS_PTR_EXP] = exp_ptr
        bas[rys_shell_id, _RYS_PTR_COEFF] = coeff_ptr
        bas[rys_shell_id, _RYS_PTR_BAS_COORD] = (
            int(atm[atom_idx, _RYS_PTR_COORD]) if natm else 0
        )
        ao_loc[rys_shell_id + 1] = ao_loc[rys_shell_id] + (
            ((int(ltot) + 1) * (int(ltot) + 2)) // 2
        )
        shell_to_rys_shell[int(shell_id)] = int(rys_shell_id)
        for local_ao_offset, parent_ao in enumerate(
            np.asarray(shell.ao_indices, dtype=np.int32).tolist()
        ):
            sorted_ao = int(ao_loc[rys_shell_id]) + int(local_ao_offset)
            ao_to_parent_ao.append(int(parent_ao))
            if 0 <= int(parent_ao) < parent_ao_to_ao.shape[0]:
                parent_ao_to_ao[int(parent_ao)] = sorted_ao

    return CudaRysEnvLayout(
        atm=np.asarray(atm, dtype=np.int32),
        bas=np.asarray(bas, dtype=np.int32),
        env=np.asarray(env_values, dtype=np.float64),
        ao_loc=np.asarray(ao_loc, dtype=np.int32),
        sorted_shell_indices=np.asarray(sorted_shell_indices, dtype=np.int32),
        shell_to_rys_shell=np.asarray(shell_to_rys_shell, dtype=np.int32),
        ao_to_parent_ao=np.asarray(ao_to_parent_ao, dtype=np.int32),
        parent_ao_to_ao=np.asarray(parent_ao_to_ao, dtype=np.int32),
        group_keys=np.asarray(shell_layout.group_keys, dtype=np.int32),
        group_offsets=np.asarray(shell_layout.group_offsets, dtype=np.int32),
    )


def _build_shell_log_q_matrix(
    shell_layout: CudaShellLayout,
    pair_screen_representative_ids: np.ndarray,
    pair_schwarz: Array,
) -> np.ndarray:
    nshell = int(shell_layout.shell_l.shape[0])
    if nshell == 0:
        return np.zeros((0, 0), dtype=np.float32)
    rep_ids = np.asarray(pair_screen_representative_ids, dtype=np.int32)
    values = np.asarray(pair_schwarz, dtype=np.float64)
    shell_pair_values = values[rep_ids]
    shell_pair_group_matrix = np.asarray(shell_layout.shell_pair_group_matrix, dtype=np.int32)
    q_matrix = shell_pair_values[shell_pair_group_matrix]
    return np.log(np.maximum(q_matrix, 1.0e-300)).astype(np.float32)


def _build_padded_log_q_matrix(
    shell_layout: CudaShellLayout,
    shell_log_q_matrix: np.ndarray,
    *,
    pad_value: float = -100.0,
) -> np.ndarray:
    sorted_idx = np.asarray(shell_layout.padded_sorted_shell_indices, dtype=np.int32)
    if sorted_idx.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    padded = np.asarray(shell_log_q_matrix, dtype=np.float32)[np.ix_(sorted_idx, sorted_idx)]
    pad_mask = np.asarray(shell_layout.padded_shell_pad_mask, dtype=bool)
    if np.any(pad_mask):
        padded = np.array(padded, copy=True)
        padded[pad_mask, :] = np.float32(pad_value)
        padded[:, pad_mask] = np.float32(pad_value)
    return padded


def _make_joltqc_tile_pairs(
    group_offsets: np.ndarray,
    log_q_matrix: np.ndarray,
    *,
    cutoff: float,
    tile_size: int = _JOLTQC_GROUP_ALIGNMENT,
) -> dict[tuple[int, int], np.ndarray]:
    q_matrix = np.asarray(log_q_matrix, dtype=np.float32)
    offsets = np.asarray(group_offsets, dtype=np.int32)
    tile = int(tile_size)
    if q_matrix.shape[0] == 0:
        return {}
    if q_matrix.shape[0] != q_matrix.shape[1]:
        raise ValueError("log_q_matrix must be square.")
    if q_matrix.shape[0] % tile != 0:
        raise ValueError("log_q_matrix size must be divisible by tile_size.")
    if offsets.ndim != 1 or offsets.shape[0] < 1:
        raise ValueError("group_offsets must be a 1D offset array.")
    if int(offsets[-1]) != q_matrix.shape[0]:
        raise ValueError("group_offsets must end at the padded shell dimension.")

    ntiles = q_matrix.shape[0] // tile
    tile_loc = offsets // tile
    tiled_q = q_matrix.reshape(ntiles, tile, ntiles, tile).max(axis=(1, 3))
    tile_pairs: dict[tuple[int, int], np.ndarray] = {}
    n_groups = offsets.shape[0] - 1
    for i in range(n_groups):
        i0 = int(tile_loc[i])
        i1 = int(tile_loc[i + 1])
        if i1 <= i0:
            continue
        i_range = np.arange(i0, i1, dtype=np.int32)
        for j in range(i + 1):
            j0 = int(tile_loc[j])
            j1 = int(tile_loc[j + 1])
            if j1 <= j0:
                continue
            j_range = np.arange(j0, j1, dtype=np.int32)
            sub_q = tiled_q[i0:i1, j0:j1]
            mask = sub_q > np.float32(cutoff)
            if i == j:
                mask = np.tril(mask)
            if not np.any(mask):
                continue
            tile_ids = i_range[:, None] * np.int32(ntiles) + j_range[None, :]
            valid = tile_ids[mask]
            order = np.argsort(sub_q[mask], kind="stable")
            tile_pairs[(i, j)] = np.asarray(valid[order], dtype=np.int32)
    return tile_pairs


def _pack_joltqc_tile_pairs(
    tile_pairs: dict[tuple[int, int], np.ndarray],
) -> CudaTilePairLayout:
    if not tile_pairs:
        return CudaTilePairLayout(
            group_pair_keys=np.zeros((0, 2), dtype=np.int32),
            group_pair_offsets=np.zeros((1,), dtype=np.int32),
            tile_pair_ids=np.zeros((0,), dtype=np.int32),
        )

    keys = sorted(tile_pairs)
    offsets = [0]
    chunks: list[np.ndarray] = []
    for key in keys:
        values = np.asarray(tile_pairs[key], dtype=np.int32).reshape(-1)
        chunks.append(values)
        offsets.append(offsets[-1] + int(values.shape[0]))
    tile_pair_ids = (
        np.concatenate(chunks, axis=0) if chunks else np.zeros((0,), dtype=np.int32)
    )
    return CudaTilePairLayout(
        group_pair_keys=np.asarray(keys, dtype=np.int32),
        group_pair_offsets=np.asarray(offsets, dtype=np.int32),
        tile_pair_ids=np.asarray(tile_pair_ids, dtype=np.int32),
    )


def _build_rys_log_q_matrix(
    shell_layout: CudaShellLayout,
    shell_log_q_matrix: np.ndarray,
) -> np.ndarray:
    sorted_idx = np.asarray(shell_layout.sorted_shell_indices, dtype=np.int32)
    if sorted_idx.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray(shell_log_q_matrix, dtype=np.float32)[np.ix_(sorted_idx, sorted_idx)]


def _make_rys_tril_pair_mappings(
    group_offsets: np.ndarray,
    log_q_matrix: np.ndarray,
    *,
    cutoff: float,
    tile: int = _GPU4PYSCF_PAIR_MAPPING_TILE,
) -> dict[tuple[int, int], np.ndarray]:
    q_matrix = np.asarray(log_q_matrix, dtype=np.float32)
    if q_matrix.ndim != 2 or q_matrix.shape[0] != q_matrix.shape[1]:
        raise ValueError("log_q_matrix must be a square shell matrix.")
    offsets = np.asarray(group_offsets, dtype=np.int32)
    if offsets.ndim != 1 or offsets.shape[0] < 1:
        raise ValueError("group_offsets must be a 1D offset array.")
    nbas = int(q_matrix.shape[0])
    if int(offsets[-1]) != nbas:
        raise ValueError("group_offsets must end at the sorted shell dimension.")
    if nbas == 0:
        return {}

    q_flat = q_matrix.reshape(-1)
    pair_mappings: dict[tuple[int, int], np.ndarray] = {}
    n_groups = int(offsets.shape[0] - 1)
    tile_size = int(tile)
    if tile_size <= 0:
        raise ValueError("tile must be positive.")
    for i in range(n_groups):
        ish0 = int(offsets[i])
        ish1 = int(offsets[i + 1])
        nish = ish1 - ish0
        if nish <= 0:
            continue
        ntiles_i = (nish + tile_size - 1) // tile_size
        ish = np.arange(
            ish0,
            ish0 + ntiles_i * tile_size,
            dtype=np.int32,
        ).reshape(ntiles_i, tile_size)
        for j in range(i + 1):
            jsh0 = int(offsets[j])
            jsh1 = int(offsets[j + 1])
            njsh = jsh1 - jsh0
            if njsh <= 0:
                continue
            ntiles_j = (njsh + tile_size - 1) // tile_size
            jsh = np.arange(
                jsh0,
                jsh0 + ntiles_j * tile_size,
                dtype=np.int32,
            ).reshape(ntiles_j, tile_size)
            ish_grid = ish[:, None, :, None]
            jsh_grid = jsh[None, :, None, :]
            pair_ids = ish_grid * np.int32(nbas) + jsh_grid
            mask = (ish_grid < np.int32(ish1)) & (jsh_grid < np.int32(jsh1))
            if i == j:
                mask = mask & (ish_grid >= jsh_grid)
            valid = pair_ids[mask]
            if valid.size:
                valid = valid[q_flat[valid] > np.float32(cutoff)]
            pair_mappings[(i, j)] = np.asarray(valid, dtype=np.int32)
    return pair_mappings


def _pack_rys_pair_mappings(
    pair_mappings: dict[tuple[int, int], np.ndarray],
) -> CudaRysPairMappingLayout:
    if not pair_mappings:
        return CudaRysPairMappingLayout(
            group_pair_keys=np.zeros((0, 2), dtype=np.int32),
            group_pair_offsets=np.zeros((1,), dtype=np.int32),
            pair_ids=np.zeros((0,), dtype=np.int32),
        )

    keys = sorted(pair_mappings)
    offsets = [0]
    chunks: list[np.ndarray] = []
    for key in keys:
        values = np.asarray(pair_mappings[key], dtype=np.int32).reshape(-1)
        chunks.append(values)
        offsets.append(offsets[-1] + int(values.shape[0]))
    pair_ids = np.concatenate(chunks, axis=0) if chunks else np.zeros((0,), dtype=np.int32)
    return CudaRysPairMappingLayout(
        group_pair_keys=np.asarray(keys, dtype=np.int32),
        group_pair_offsets=np.asarray(offsets, dtype=np.int32),
        pair_ids=np.asarray(pair_ids, dtype=np.int32),
    )


def _make_full_joltqc_tile_pairs(
    group_offsets: np.ndarray,
    *,
    tile_size: int = _JOLTQC_GROUP_ALIGNMENT,
) -> dict[tuple[int, int], np.ndarray]:
    offsets = np.asarray(group_offsets, dtype=np.int32)
    tile = int(tile_size)
    if offsets.ndim != 1 or offsets.shape[0] < 1:
        raise ValueError("group_offsets must be a 1D offset array.")
    if int(offsets[-1]) == 0:
        return {}
    if int(offsets[-1]) % tile != 0:
        raise ValueError("group_offsets must end at a tile-aligned shell dimension.")

    ntiles = int(offsets[-1]) // tile
    tile_loc = offsets // tile
    tile_pairs: dict[tuple[int, int], np.ndarray] = {}
    n_groups = offsets.shape[0] - 1
    for i in range(n_groups):
        i0 = int(tile_loc[i])
        i1 = int(tile_loc[i + 1])
        if i1 <= i0:
            continue
        i_range = np.arange(i0, i1, dtype=np.int32)
        for j in range(i + 1):
            j0 = int(tile_loc[j])
            j1 = int(tile_loc[j + 1])
            if j1 <= j0:
                continue
            j_range = np.arange(j0, j1, dtype=np.int32)
            tile_ids = i_range[:, None] * np.int32(ntiles) + j_range[None, :]
            if i == j:
                tile_ids = tile_ids[np.tril(np.ones(tile_ids.shape, dtype=bool))]
            tile_pairs[(i, j)] = np.asarray(tile_ids.reshape(-1), dtype=np.int32)
    return tile_pairs


def _build_shell_density_condition(
    density: Array,
    shell_ao_indices_padded: Array,
    shell_ao_sizes: Array,
) -> Array:
    shell_indices = jnp.asarray(shell_ao_indices_padded, dtype=jnp.int32)
    shell_sizes_arr = jnp.asarray(shell_ao_sizes, dtype=jnp.int32)
    nshell = int(shell_indices.shape[0])
    max_shell_ao = int(shell_indices.shape[1]) if shell_indices.ndim == 2 else 0
    if nshell == 0 or max_shell_ao == 0:
        return jnp.zeros((nshell, nshell), dtype=jnp.float64)
    density_abs = jnp.abs(jnp.asarray(density, dtype=jnp.float64))
    ao_mask = jnp.arange(max_shell_ao, dtype=jnp.int32)[None, :] < shell_sizes_arr[:, None]

    def _block_row(i: Array) -> Array:
        ao_i = shell_indices[i]
        mask_i = ao_mask[i]

        def _block_col(j: Array) -> Array:
            ao_j = shell_indices[j]
            mask_j = ao_mask[j]
            block = density_abs[ao_i[:, None], ao_j[None, :]]
            valid = mask_i[:, None] & mask_j[None, :]
            return jnp.max(jnp.where(valid, block, 0.0))

        return jax.vmap(_block_col)(jnp.arange(nshell, dtype=jnp.int32))

    return jax.vmap(_block_row)(jnp.arange(nshell, dtype=jnp.int32))


def extract_cuda_ao_system(
    basis: CartesianBasis,
    *,
    max_l: int = 2,
    include_pair_metadata: bool = True,
) -> CudaAOSystem:
    """Extract dense Cartesian AO arrays accepted by the CUDA direct-J/K kernel."""

    aos = tuple(basis.aos)
    shell_layout = _build_joltqc_shell_layout(basis)
    joltqc_basis_layout = _build_joltqc_basis_layout(basis)
    joltqc_quartet_layout = _build_joltqc_quartet_layout(joltqc_basis_layout)
    if not aos:
        return CudaAOSystem(
            centers=np.zeros((0, 3), dtype=np.float64),
            angulars=np.zeros((0, 3), dtype=np.int32),
            exponents=np.zeros((0, 0), dtype=np.float64),
            coefficients=np.zeros((0, 0), dtype=np.float64),
            nprims=np.zeros((0,), dtype=np.int32),
            pair_exponents=np.zeros((0, 0), dtype=np.float64),
            pair_centers=np.zeros((0, 0, 3), dtype=np.float64),
            pair_prefactors=np.zeros((0, 0), dtype=np.float64),
            pair_nprims=np.zeros((0,), dtype=np.int32),
            pair_rows=np.zeros((0,), dtype=np.int32),
            pair_cols=np.zeros((0,), dtype=np.int32),
            pair_screen_group_ids=np.zeros((0,), dtype=np.int32),
            pair_screen_representative_ids=np.zeros((0,), dtype=np.int32),
            n_pair_screen_groups=0,
            shell_ao_indices_padded=np.asarray(basis.shell_ao_indices_padded, dtype=np.int32),
            shell_ao_sizes=np.asarray(basis.shell_ao_sizes, dtype=np.int32),
            shell_layout=shell_layout,
            joltqc_basis_layout=joltqc_basis_layout,
            joltqc_quartet_layout=joltqc_quartet_layout,
        )
    for ao in aos:
        if sum(int(power) for power in ao.angular) > int(max_l):
            raise NotImplementedError(
                f"CUDA direct J/K currently supports Cartesian AOs up to l={int(max_l)}."
            )

    nprims = np.asarray([int(ao.exponents.shape[0]) for ao in aos], dtype=np.int32)
    max_nprim = int(np.max(nprims))
    centers = np.asarray([np.asarray(ao.center, dtype=np.float64) for ao in aos], dtype=np.float64)
    angulars = np.asarray([ao.angular for ao in aos], dtype=np.int32)
    exponents = np.zeros((len(aos), max_nprim), dtype=np.float64)
    coefficients = np.zeros((len(aos), max_nprim), dtype=np.float64)
    for idx, ao in enumerate(aos):
        nprim = int(nprims[idx])
        exponents[idx, :nprim] = np.asarray(ao.exponents, dtype=np.float64)
        coefficients[idx, :nprim] = np.asarray(ao.coefficients, dtype=np.float64) * _primitive_cart_norm(
            np.asarray(ao.exponents, dtype=np.float64),
            tuple(int(power) for power in ao.angular),
        )
    if not bool(include_pair_metadata):
        return CudaAOSystem(
            centers=centers,
            angulars=angulars,
            exponents=exponents,
            coefficients=coefficients,
            nprims=nprims,
            pair_exponents=np.zeros((0, 0), dtype=np.float64),
            pair_centers=np.zeros((0, 0, 3), dtype=np.float64),
            pair_prefactors=np.zeros((0, 0), dtype=np.float64),
            pair_nprims=np.zeros((0,), dtype=np.int32),
            pair_rows=np.zeros((0,), dtype=np.int32),
            pair_cols=np.zeros((0,), dtype=np.int32),
            pair_screen_group_ids=np.zeros((0,), dtype=np.int32),
            pair_screen_representative_ids=np.zeros((0,), dtype=np.int32),
            n_pair_screen_groups=0,
            shell_ao_indices_padded=np.asarray(basis.shell_ao_indices_padded, dtype=np.int32),
            shell_ao_sizes=np.asarray(basis.shell_ao_sizes, dtype=np.int32),
            shell_layout=shell_layout,
            joltqc_basis_layout=joltqc_basis_layout,
            joltqc_quartet_layout=joltqc_quartet_layout,
        )
    npair = len(aos) * (len(aos) + 1) // 2
    pair_nprims = np.zeros((npair,), dtype=np.int32)
    max_pair_nprim = int(max_nprim * max_nprim)
    pair_exponents = np.zeros((npair, max_pair_nprim), dtype=np.float64)
    pair_centers = np.zeros((npair, max_pair_nprim, 3), dtype=np.float64)
    pair_prefactors = np.zeros((npair, max_pair_nprim), dtype=np.float64)
    pair_rows = np.zeros((npair,), dtype=np.int32)
    pair_cols = np.zeros((npair,), dtype=np.int32)
    ao_shell_ids = np.arange(len(aos), dtype=np.int32)
    if basis.shells:
        ao_shell_ids = np.full((len(aos),), -1, dtype=np.int32)
        for shell_id, shell in enumerate(basis.shells):
            for ao_idx in np.asarray(shell.ao_indices, dtype=np.int32):
                ao_shell_ids[int(ao_idx)] = int(shell_id)
        if np.any(ao_shell_ids < 0):
            ao_shell_ids = np.arange(len(aos), dtype=np.int32)
    pair_screen_group_ids = np.zeros((npair,), dtype=np.int32)
    shell_pair_group_ids: dict[tuple[int, int], int] = {}
    pair_screen_representative_ids: list[int] = []
    shell_pair_group_matrix = np.zeros((len(basis.shells), len(basis.shells)), dtype=np.int32)
    for i, ao_i in enumerate(aos):
        center_i = centers[i]
        nprim_i = int(nprims[i])
        for j in range(i + 1):
            center_j = centers[j]
            nprim_j = int(nprims[j])
            rab2 = float(np.sum((center_i - center_j) ** 2))
            pair_id = i * (i + 1) // 2 + j
            pair_rows[pair_id] = i
            pair_cols[pair_id] = j
            shell_i = int(ao_shell_ids[i])
            shell_j = int(ao_shell_ids[j])
            if shell_i < shell_j:
                shell_i, shell_j = shell_j, shell_i
            shell_pair = (shell_i, shell_j)
            if shell_pair not in shell_pair_group_ids:
                shell_pair_group_ids[shell_pair] = len(shell_pair_group_ids)
                pair_screen_representative_ids.append(pair_id)
            group_id = shell_pair_group_ids[shell_pair]
            if shell_pair_group_matrix.size:
                shell_pair_group_matrix[shell_i, shell_j] = group_id
                shell_pair_group_matrix[shell_j, shell_i] = group_id
            pair_screen_group_ids[pair_id] = group_id
            cursor = 0
            for ip in range(nprim_i):
                alpha = float(exponents[i, ip])
                weight_i = float(coefficients[i, ip])
                for jp in range(nprim_j):
                    beta = float(exponents[j, jp])
                    p = alpha + beta
                    mu = alpha * beta / p
                    pair_exponents[pair_id, cursor] = p
                    pair_centers[pair_id, cursor] = (alpha * center_i + beta * center_j) / p
                    pair_prefactors[pair_id, cursor] = (
                        weight_i
                        * float(coefficients[j, jp])
                        * math.exp(-mu * rab2)
                    )
                    cursor += 1
            pair_nprims[pair_id] = cursor
    shell_layout = CudaShellLayout(
        shell_l=shell_layout.shell_l,
        shell_nprims=shell_layout.shell_nprims,
        sorted_shell_indices=shell_layout.sorted_shell_indices,
        group_keys=shell_layout.group_keys,
        group_offsets=shell_layout.group_offsets,
        padded_sorted_shell_indices=shell_layout.padded_sorted_shell_indices,
        padded_shell_pad_mask=shell_layout.padded_shell_pad_mask,
        padded_group_offsets=shell_layout.padded_group_offsets,
        tile_size=shell_layout.tile_size,
        tile_shell_indices=shell_layout.tile_shell_indices,
        tile_shell_pad_mask=shell_layout.tile_shell_pad_mask,
        shell_pair_group_matrix=shell_pair_group_matrix,
    )
    return CudaAOSystem(
        centers=centers,
        angulars=angulars,
        exponents=exponents,
        coefficients=coefficients,
        nprims=nprims,
        pair_exponents=pair_exponents,
        pair_centers=pair_centers,
        pair_prefactors=pair_prefactors,
        pair_nprims=pair_nprims,
        pair_rows=pair_rows,
        pair_cols=pair_cols,
        pair_screen_group_ids=pair_screen_group_ids,
        pair_screen_representative_ids=np.asarray(pair_screen_representative_ids, dtype=np.int32),
        n_pair_screen_groups=len(shell_pair_group_ids),
        shell_ao_indices_padded=np.asarray(basis.shell_ao_indices_padded, dtype=np.int32),
        shell_ao_sizes=np.asarray(basis.shell_ao_sizes, dtype=np.int32),
        shell_layout=shell_layout,
        joltqc_basis_layout=joltqc_basis_layout,
        joltqc_quartet_layout=joltqc_quartet_layout,
    )


def _kernel_source_path() -> Path:
    source = Path(__file__).with_name("cuda_direct_jk_kernel.cu")
    if source.exists():
        return source
    raise FileNotFoundError("Could not locate cuda_direct_jk_kernel.cu.")


def _gpu4pyscf_rys_source_root() -> Path:
    source = Path(__file__).with_name("gpu4pyscf_gvhf_rys")
    if source.exists():
        return source
    raise FileNotFoundError("Could not locate vendored GPU4PySCF gvhf-rys sources.")


def _gpu4pyscf_rys_source_files() -> tuple[Path, ...]:
    root = _gpu4pyscf_rys_source_root()
    names = (
        "gvhf-rys/rys_jk_driver.cu",
        "gvhf-rys/rys_roots_dat.cu",
        "gvhf-rys/rys_constant.cu",
        "gvhf-rys/rys_contract_k.cu",
        "gvhf-rys/unrolled_rys_k.cu",
        "gvhf-rys/rys_contract_jk.cu",
        "gvhf-rys/unrolled_rys_jk.cu",
    )
    return tuple(root / name for name in names)


def _library_name_for_arch(
    arch: str,
    *,
    source_path: str | os.PathLike[str] | None = None,
    extra_source: str | None = None,
    extra_source_key: str | None = None,
) -> str:
    ffi = _ffi_module()
    source = Path(source_path) if source_path is not None else _kernel_source_path()
    digest = hashlib.sha256()
    digest.update(source.read_bytes())
    if extra_source_key is not None:
        digest.update(str(extra_source_key).encode())
    if extra_source is not None:
        digest.update(extra_source.encode())
    digest.update(str(arch).encode())
    digest.update(str(ffi.include_dir()).encode())
    return f"libtd_graddft_cuda_direct_jk_{digest.hexdigest()[:16]}.so"


def _gpu4pyscf_rys_library_name_for_arch(arch: str) -> str:
    digest = hashlib.sha256()
    for source in _gpu4pyscf_rys_source_files():
        digest.update(source.read_bytes())
    root = _gpu4pyscf_rys_source_root()
    for header in (
        root / "gvhf-rys/vhf.cuh",
        root / "gvhf-rys/rys_contract_k.cuh",
        root / "gvhf-rys/rys_roots.cuh",
        root / "gvhf-rys/rys_roots.cu",
        root / "gvhf-rys/rys_roots_for_k.cu",
        root / "gvhf-rys/create_tasks.cu",
        root / "gint/cuda_alloc.cuh",
    ):
        digest.update(header.read_bytes())
    digest.update(str(arch).encode())
    digest.update(" ".join(_nvcc_split_compile_flags()).encode())
    return f"libtd_graddft_gpu4pyscf_gvhf_rys_{digest.hexdigest()[:16]}.so"


def _nvcc_split_compile_flags() -> list[str]:
    raw = os.environ.get(_NVCC_SPLIT_COMPILE_ENV, "0").strip()
    if raw.lower() in {"", "none", "off", "false", "no"}:
        return []
    try:
        threads = int(raw)
    except ValueError:
        threads = 0
    if threads < 0:
        threads = 0
    return [f"--split-compile={threads}"]


def _nvcc_compile_jobs(n_sources: int) -> int:
    raw = os.environ.get(_NVCC_COMPILE_JOBS_ENV, "").strip()
    if raw:
        try:
            jobs = int(raw)
        except ValueError:
            jobs = 1
    else:
        jobs = min(max(1, os.cpu_count() or 1), 8)
    return max(1, min(int(n_sources), int(jobs)))


def _joltqc_split_threshold() -> int:
    raw = os.environ.get(_JOLTQC_SPLIT_THRESHOLD_ENV, "").strip()
    if not raw:
        return 8
    try:
        return max(0, int(raw))
    except ValueError:
        return 8


def _truthy_env(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _gpu4pyscf_rys_enabled() -> bool:
    return _truthy_env(_GPU4PYSCF_RYS_ENV, default=True)


def _joltqc_fixed_universe_config(
    *,
    max_l: int | None = None,
    nprim_max: int | None = None,
) -> tuple[int, int]:
    if max_l is None:
        raw_max_l = os.environ.get(_JOLTQC_FIXED_MAX_L_ENV, "").strip()
        max_l = int(raw_max_l) if raw_max_l else 2
    if nprim_max is None:
        raw_nprim = os.environ.get(_JOLTQC_FIXED_NPRIM_MAX_ENV, "").strip()
        nprim_max = int(raw_nprim) if raw_nprim else _JOLTQC_NPRIM_MAX
    return int(max_l), int(nprim_max)


def _joltqc_fixed_universe_arrays(
    *,
    max_l: int | None = None,
    nprim_max: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from .joltqc_port import codegen as joltqc_codegen

    fixed_max_l, fixed_nprim_max = _joltqc_fixed_universe_config(
        max_l=max_l,
        nprim_max=nprim_max,
    )
    return joltqc_codegen.build_fixed_1qnt_dispatch_arrays(
        max_l=fixed_max_l,
        nprim_max=fixed_nprim_max,
    )


def _joltqc_fixed_universe_source_key(
    *,
    max_l: int | None = None,
    nprim_max: int | None = None,
) -> str:
    from .joltqc_port import codegen as joltqc_codegen

    group_keys, group_quartet_keys, group_quartet_offsets = _joltqc_fixed_universe_arrays(
        max_l=max_l,
        nprim_max=nprim_max,
    )
    return joltqc_codegen.build_1qnt_dispatch_source_key(
        group_keys,
        group_quartet_keys,
        group_quartet_offsets,
    )


def _packaged_prebuilt_library_path(
    arch: str,
    *,
    extra_source_key: str | None = None,
) -> Path | None:
    path = Path(__file__).with_name(
        _library_name_for_arch(str(arch), extra_source_key=extra_source_key)
    )
    return path if path.exists() else None


def _any_packaged_prebuilt_library() -> Path | None:
    candidates = sorted(Path(__file__).parent.glob("libtd_graddft_cuda_direct_jk_*.so"))
    return candidates[0] if candidates else None


def build_prebuilt_cuda_direct_jk_library(
    output_dir: str | os.PathLike[str],
    *,
    source_path: str | os.PathLike[str] | None = None,
    nvcc: str | None = None,
    arch: str = "native",
    force: bool = False,
    joltqc_group_keys: np.ndarray | None = None,
    joltqc_group_quartet_keys: np.ndarray | None = None,
    joltqc_group_quartet_offsets: np.ndarray | None = None,
    joltqc_dispatch: str | None = None,
    joltqc_fixed_universe: bool = False,
    joltqc_fixed_max_l: int | None = None,
    joltqc_fixed_nprim_max: int | None = None,
    include_gpu4pyscf_rys: bool | None = None,
) -> Path:
    """Build the CUDA direct-J/K FFI shared library outside SCF runtime."""

    compiler = nvcc or os.environ.get("TD_GRADDFT_NVCC") or shutil.which("nvcc")
    if compiler is None:
        raise RuntimeError(
            "CUDA direct J/K prebuild requires nvcc. Set TD_GRADDFT_NVCC or put nvcc on PATH."
        )
    arch_name = _detect_cuda_arch() if str(arch) == "native" else str(arch)
    source = Path(source_path) if source_path is not None else _kernel_source_path()
    use_gpu4pyscf_rys = (
        _gpu4pyscf_rys_enabled()
        if include_gpu4pyscf_rys is None
        else bool(include_gpu4pyscf_rys)
    )
    joltqc_source_key = None
    joltqc_source_units: list[tuple[str, str]] = []
    joltqc_codegen = None
    group_keys_arr = None
    group_quartet_keys_arr = None
    group_quartet_offsets_arr = None
    if bool(joltqc_fixed_universe):
        (
            joltqc_group_keys,
            joltqc_group_quartet_keys,
            joltqc_group_quartet_offsets,
        ) = _joltqc_fixed_universe_arrays(
            max_l=joltqc_fixed_max_l,
            nprim_max=joltqc_fixed_nprim_max,
        )
    if joltqc_group_keys is not None and joltqc_group_quartet_keys is not None:
        dispatch_mode = str(
            joltqc_dispatch or os.environ.get(_JOLTQC_DISPATCH_ENV) or "signature"
        ).strip().lower().replace("-", "_")
        if dispatch_mode in {"runtime_signature", "device_signature"}:
            raise NotImplementedError(
                "Runtime JoltQC signature dispatch cannot be used with GPU FFI device "
                "metadata buffers. Use signature_cached/basis_specific host dispatch."
            )
        if dispatch_mode not in {"signature", "signature_cached", "basis_specific"}:
            raise ValueError(
                "joltqc_dispatch must be 'signature', 'signature_cached', or 'basis_specific', "
                f"got {joltqc_dispatch!r}"
            )
        from .joltqc_port import codegen as joltqc_codegen

        group_keys_arr = np.asarray(joltqc_group_keys, dtype=np.int32)
        group_quartet_keys_arr = np.asarray(joltqc_group_quartet_keys, dtype=np.int32)
        if joltqc_group_quartet_offsets is None:
            raise ValueError("joltqc_group_quartet_offsets is required for JoltQC host dispatch")
        group_quartet_offsets_arr = np.asarray(joltqc_group_quartet_offsets, dtype=np.int32)
        joltqc_source_key = joltqc_codegen.build_1qnt_dispatch_source_key(
            group_keys_arr,
            group_quartet_keys_arr,
            group_quartet_offsets_arr,
        )
    ffi = _ffi_module()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_key_parts = []
    if joltqc_source_key is not None:
        source_key_parts.append(str(joltqc_source_key))
    if use_gpu4pyscf_rys:
        source_key_parts.append(_gpu4pyscf_rys_library_name_for_arch(arch_name))
    combined_source_key = "|".join(source_key_parts) if source_key_parts else None
    library = out_dir / _library_name_for_arch(
        arch_name,
        source_path=source,
        extra_source_key=combined_source_key,
    )
    if library.exists() and not bool(force):
        return library
    source_files = [str(source)]
    if use_gpu4pyscf_rys:
        source_files.extend(str(path) for path in _gpu4pyscf_rys_source_files())
    if joltqc_source_key is not None:
        if (
            joltqc_codegen is None
            or group_keys_arr is None
            or group_quartet_keys_arr is None
            or group_quartet_offsets_arr is None
        ):
            raise RuntimeError("JoltQC source key was created without dispatch metadata.")
        joltqc_source_units = joltqc_codegen.build_1qnt_dispatch_source_units(
            group_keys_arr,
            group_quartet_keys_arr,
            group_quartet_offsets_arr,
        )
        generated_digest = hashlib.sha256(str(joltqc_source_key).encode()).hexdigest()[:16]
        for unit_name, source_text in joltqc_source_units:
            generated_source = out_dir / f"joltqc_1qnt_{generated_digest}_{unit_name}"
            generated_source.write_text(source_text)
            source_files.append(str(generated_source))

    compile_base = [
        compiler,
        "-O3",
        "--std=c++17",
        "-Xcompiler",
        "-fPIC",
        *(["-rdc=true"] if use_gpu4pyscf_rys else []),
        f"-arch={arch_name}",
        *_nvcc_split_compile_flags(),
        *(["-DTD_GRADDFT_ENABLE_GPU4PYSCF_RYS=1"] if use_gpu4pyscf_rys else []),
        "-I",
        ffi.include_dir(),
        *(["-I", str(_gpu4pyscf_rys_source_root()), "-I", str(_gpu4pyscf_rys_source_root() / "gvhf-rys")] if use_gpu4pyscf_rys else []),
    ]
    if not joltqc_source_units:
        command = [
            compiler,
            "-O3",
            "--std=c++17",
            "-shared",
            "-Xcompiler",
            "-fPIC",
            *(["-rdc=true"] if use_gpu4pyscf_rys else []),
            f"-arch={arch_name}",
            *_nvcc_split_compile_flags(),
            *(["-DTD_GRADDFT_ENABLE_GPU4PYSCF_RYS=1"] if use_gpu4pyscf_rys else []),
            "-I",
            ffi.include_dir(),
            *(["-I", str(_gpu4pyscf_rys_source_root()), "-I", str(_gpu4pyscf_rys_source_root() / "gvhf-rys")] if use_gpu4pyscf_rys else []),
            *source_files,
            "-o",
            str(library),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Failed to prebuild CUDA direct J/K FFI library with command:\n"
                + " ".join(command)
                + "\nstdout:\n"
                + (exc.stdout or "")
                + "\nstderr:\n"
                + (exc.stderr or "")
            ) from exc
        return library

    object_dir = out_dir / "_objects" / str(arch_name)
    object_dir.mkdir(parents=True, exist_ok=True)

    def _object_file_for_source(source_file: str) -> Path:
        source_path_obj = Path(source_file)
        digest = hashlib.sha256()
        digest.update(source_path_obj.read_bytes())
        digest.update(str(arch_name).encode())
        digest.update(str(ffi.include_dir()).encode())
        digest.update(" ".join(_nvcc_split_compile_flags()).encode())
        return object_dir / f"obj_{digest.hexdigest()[:16]}.o"

    object_files = [_object_file_for_source(source_file) for source_file in source_files]

    def _compile_object(source_file: str, object_file: Path) -> None:
        if object_file.exists() and not bool(force):
            return
        command = [
            *compile_base,
            "-c",
            source_file,
            "-o",
            str(object_file),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Failed to compile CUDA direct J/K object with command:\n"
                + " ".join(command)
                + "\nstdout:\n"
                + (exc.stdout or "")
                + "\nstderr:\n"
                + (exc.stderr or "")
            ) from exc

    jobs = _nvcc_compile_jobs(len(source_files))
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(_compile_object, source_file, object_file)
            for source_file, object_file in zip(source_files, object_files, strict=True)
        ]
        for future in as_completed(futures):
            future.result()

    link_command = [
        compiler,
        "-shared",
        *(["-rdc=true", f"-arch={arch_name}"] if use_gpu4pyscf_rys else []),
        *[str(object_file) for object_file in object_files],
        "-o",
        str(library),
    ]
    try:
        subprocess.run(link_command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to link CUDA direct J/K FFI library with command:\n"
            + " ".join(link_command)
            + "\nstdout:\n"
            + (exc.stdout or "")
            + "\nstderr:\n"
            + (exc.stderr or "")
        ) from exc
    return library


def build_gpu4pyscf_gvhf_rys_library(
    output_dir: str | os.PathLike[str],
    *,
    nvcc: str | None = None,
    arch: str = "native",
    force: bool = False,
) -> Path:
    """Build vendored GPU4PySCF gvhf-rys as a fixed CUDA shared library."""

    compiler = nvcc or os.environ.get("TD_GRADDFT_NVCC") or shutil.which("nvcc")
    if compiler is None:
        raise RuntimeError(
            "GPU4PySCF Rys prebuild requires nvcc. Set TD_GRADDFT_NVCC or put nvcc on PATH."
        )
    arch_name = _detect_cuda_arch() if str(arch) == "native" else str(arch)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    library = out_dir / _gpu4pyscf_rys_library_name_for_arch(arch_name)
    if library.exists() and not bool(force):
        return library

    root = _gpu4pyscf_rys_source_root()
    source_files = [str(source) for source in _gpu4pyscf_rys_source_files()]
    command = [
        compiler,
        "-O3",
        "--std=c++17",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-rdc=true",
        f"-arch={arch_name}",
        *_nvcc_split_compile_flags(),
        "-I",
        str(root),
        "-I",
        str(root / "gvhf-rys"),
        *source_files,
        "-o",
        str(library),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to prebuild GPU4PySCF gvhf-rys library with command:\n"
            + " ".join(command)
            + "\nstdout:\n"
            + (exc.stdout or "")
            + "\nstderr:\n"
            + (exc.stderr or "")
        ) from exc
    return library


@lru_cache(maxsize=16)
def _detect_cuda_arch_for_env(env_arch: str | None, visible_devices_raw: str) -> str:
    if env_arch:
        return env_arch
    visible_devices = str(visible_devices_raw).split(",")
    first_visible = visible_devices[0].strip() if visible_devices else ""
    capability_pattern = re.compile(r"^\s*(\d+)\.(\d+)\s*$")

    def _query_arch(command: list[str]) -> str | None:
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        for line in result.stdout.splitlines():
            match = capability_pattern.match(line)
            if match:
                return f"sm_{match.group(1)}{match.group(2)}"
        return None

    command = ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
    if first_visible and first_visible not in {"-1", "none", "None"}:
        arch = _query_arch(
            [
                "nvidia-smi",
                f"--id={first_visible}",
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            ]
        )
        if arch is not None:
            return arch
    arch = _query_arch(command)
    if arch is not None:
        return arch
    return "sm_80"


def _clear_cuda_arch_cache() -> None:
    _detect_cuda_arch_for_env.cache_clear()


def _detect_cuda_arch() -> str:
    return _detect_cuda_arch_for_env(
        os.environ.get("TD_GRADDFT_CUDA_ARCH"),
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )


class CudaDirectJKBuilder:
    """JAX FFI CUDA direct J/K builder for no-DF SCF."""

    def __init__(
        self,
        basis: CartesianBasis,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        nvcc: str | None = None,
        arch: str = "native",
        max_l: int = 2,
        include_pair_metadata: bool = True,
    ) -> None:
        self.system = extract_cuda_ao_system(
            basis,
            max_l=max_l,
            include_pair_metadata=include_pair_metadata,
        )
        self.rys_env_layout = _build_rys_env_layout(
            basis,
            shell_layout=self.system.shell_layout,
        )
        self.cache_dir = Path(
            cache_dir
            or os.environ.get("TD_GRADDFT_CUDA_JK_CACHE", "")
            or (Path(tempfile.gettempdir()) / "td_graddft_cuda_direct_jk")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.nvcc = nvcc or os.environ.get("TD_GRADDFT_NVCC") or shutil.which("nvcc")
        self.arch = None if str(arch) == "native" else str(arch)
        self.max_l = int(max_l)
        self.library = self._compile_library()
        self.has_gpu4pyscf_rys = False
        self._compile_and_register()
        self.last_kernel_avg_ms: float | None = None
        self.centers = jnp.asarray(self.system.centers, dtype=jnp.float64)
        self.angulars = jnp.asarray(self.system.angulars, dtype=jnp.int32)
        self.exponents = jnp.asarray(self.system.exponents, dtype=jnp.float64)
        self.coefficients = jnp.asarray(self.system.coefficients, dtype=jnp.float64)
        self.nprims = jnp.asarray(self.system.nprims, dtype=jnp.int32)
        self.pair_exponents = jnp.asarray(self.system.pair_exponents, dtype=jnp.float64)
        self.pair_centers = jnp.asarray(self.system.pair_centers, dtype=jnp.float64)
        self.pair_prefactors = jnp.asarray(self.system.pair_prefactors, dtype=jnp.float64)
        self.pair_nprims = jnp.asarray(self.system.pair_nprims, dtype=jnp.int32)
        self.pair_rows = jnp.asarray(self.system.pair_rows, dtype=jnp.int32)
        self.pair_cols = jnp.asarray(self.system.pair_cols, dtype=jnp.int32)
        self.pair_screen_group_ids = jnp.asarray(
            self.system.pair_screen_group_ids,
            dtype=jnp.int32,
        )
        self.shell_ao_indices_padded = jnp.asarray(
            self.system.shell_ao_indices_padded,
            dtype=jnp.int32,
        )
        self.shell_ao_sizes = jnp.asarray(
            self.system.shell_ao_sizes,
            dtype=jnp.int32,
        )
        self.tile_shell_indices = jnp.asarray(
            self.system.shell_layout.tile_shell_indices,
            dtype=jnp.int32,
        )
        self.tile_shell_pad_mask = jnp.asarray(
            self.system.shell_layout.tile_shell_pad_mask,
            dtype=jnp.int32,
        )
        self.joltqc_basis_data = jnp.asarray(
            self.system.joltqc_basis_layout.basis_data,
            dtype=jnp.float64,
        )
        self.joltqc_basis_data_fp32 = jnp.asarray(
            self.system.joltqc_basis_layout.basis_data_fp32,
            dtype=jnp.float32,
        )
        self.joltqc_ao_to_parent_ao = jnp.asarray(
            self.system.joltqc_basis_layout.ao_to_parent_ao,
            dtype=jnp.int32,
        )
        self.joltqc_shell_l = jnp.asarray(
            self.system.joltqc_basis_layout.shell_l,
            dtype=jnp.int32,
        )
        self.joltqc_shell_nprims = jnp.asarray(
            self.system.joltqc_basis_layout.shell_nprims,
            dtype=jnp.int32,
        )
        self.joltqc_group_keys = jnp.asarray(
            self.system.joltqc_basis_layout.group_keys,
            dtype=jnp.int32,
        )
        self.joltqc_group_quartet_keys = jnp.asarray(
            self.system.joltqc_quartet_layout.group_quartet_keys,
            dtype=jnp.int32,
        )
        self.joltqc_group_quartet_offsets = jnp.asarray(
            self.system.joltqc_quartet_layout.group_quartet_offsets,
            dtype=jnp.int32,
        )
        self.joltqc_shell_quartets = jnp.asarray(
            self.system.joltqc_quartet_layout.shell_quartets,
            dtype=jnp.int32,
        )
        self.rys_atm = jnp.asarray(self.rys_env_layout.atm, dtype=jnp.int32)
        self.rys_bas = jnp.asarray(self.rys_env_layout.bas, dtype=jnp.int32)
        self.rys_env = jnp.asarray(self.rys_env_layout.env, dtype=jnp.float64)
        self.rys_ao_loc = jnp.asarray(self.rys_env_layout.ao_loc, dtype=jnp.int32)
        self.rys_ao_to_parent_ao = jnp.asarray(
            self.rys_env_layout.ao_to_parent_ao,
            dtype=jnp.int32,
        )
        self.rys_parent_ao_to_ao = jnp.asarray(
            self.rys_env_layout.parent_ao_to_ao,
            dtype=jnp.int32,
        )
        self.rys_group_offsets = jnp.asarray(
            self.rys_env_layout.group_offsets,
            dtype=jnp.int32,
        )
        self._pair_schwarz: Array | None = None
        self._shell_log_q_matrix: np.ndarray | None = None
        self._full_tile_pair_layout: CudaTilePairLayout | None = None
        self._rys_log_q_matrix: np.ndarray | None = None
        self._full_rys_pair_mapping_layout: CudaRysPairMappingLayout | None = None

    def _compile_library(self) -> Path:
        prebuilt = os.environ.get(_PREBUILT_LIBRARY_ENV)
        if prebuilt:
            library = Path(prebuilt).expanduser()
            if not library.exists():
                raise FileNotFoundError(
                    f"{_PREBUILT_LIBRARY_ENV} points to a missing CUDA FFI library: {library}"
                )
            return library
        arch = self.arch
        if arch is None and self.nvcc is not None:
            arch = _detect_cuda_arch()
        if arch is not None:
            packaged = _packaged_prebuilt_library_path(arch)
            if packaged is not None:
                return packaged
            if _truthy_env(_JOLTQC_FIXED_UNIVERSE_ENV):
                fixed_source_key = _joltqc_fixed_universe_source_key(max_l=self.max_l)
                packaged_fixed = _packaged_prebuilt_library_path(
                    arch,
                    extra_source_key=fixed_source_key,
                )
                if packaged_fixed is not None:
                    return packaged_fixed
        if self.nvcc is not None:
            if arch is None:
                arch = _detect_cuda_arch()
            if _truthy_env(_JOLTQC_FIXED_UNIVERSE_ENV):
                return build_prebuilt_cuda_direct_jk_library(
                    self.cache_dir,
                    nvcc=self.nvcc,
                    arch=arch,
                    joltqc_fixed_universe=True,
                    joltqc_fixed_max_l=self.max_l,
                )
            return build_prebuilt_cuda_direct_jk_library(
                self.cache_dir,
                nvcc=self.nvcc,
                arch=arch,
            )
        if self.arch is not None:
            packaged = _packaged_prebuilt_library_path(self.arch)
            if packaged is not None:
                return packaged
        any_packaged = _any_packaged_prebuilt_library()
        if any_packaged is not None:
            return any_packaged
        raise RuntimeError(
            "CUDA direct J/K requires nvcc. Set TD_GRADDFT_NVCC or put nvcc on PATH."
        )

    def _compile_and_register(self) -> None:
        library = self.library.resolve()
        lib = ctypes.CDLL(str(library))
        ffi = _ffi_module()
        if _FFI_TARGET_NAME not in _REGISTERED_FFI_TARGETS:
            ffi.register_ffi_target(
                _FFI_TARGET_NAME,
                ffi.pycapsule(getattr(lib, "TdGraddftCudaDirectJkFfi")),
                platform="CUDA",
                api_version=1,
            )
            _REGISTERED_FFI_TARGETS.add(_FFI_TARGET_NAME)
        if _JOLTQC_DIRECT_JK_TARGET_NAME not in _REGISTERED_FFI_TARGETS:
            ffi.register_ffi_target(
                _JOLTQC_DIRECT_JK_TARGET_NAME,
                ffi.pycapsule(getattr(lib, "TdGraddftCudaJoltQCDirectJkFfi")),
                platform="CUDA",
                api_version=1,
            )
            _REGISTERED_FFI_TARGETS.add(_JOLTQC_DIRECT_JK_TARGET_NAME)
        if _SCREENED_FFI_TARGET_NAME not in _REGISTERED_FFI_TARGETS:
            ffi.register_ffi_target(
                _SCREENED_FFI_TARGET_NAME,
                ffi.pycapsule(getattr(lib, "TdGraddftCudaScreenedDirectJkFfi")),
                platform="CUDA",
                api_version=1,
            )
            _REGISTERED_FFI_TARGETS.add(_SCREENED_FFI_TARGET_NAME)
        rys_symbol = getattr(lib, "TdGraddftCudaGpu4PyScfRysDirectJkFfi", None)
        self.has_gpu4pyscf_rys = (
            rys_symbol is not None or _GPU4PYSCF_RYS_TARGET_NAME in _REGISTERED_FFI_TARGETS
        )
        if rys_symbol is not None and _GPU4PYSCF_RYS_TARGET_NAME not in _REGISTERED_FFI_TARGETS:
            ffi.register_ffi_target(
                _GPU4PYSCF_RYS_TARGET_NAME,
                ffi.pycapsule(rys_symbol),
                platform="CUDA",
                api_version=1,
            )
            _REGISTERED_FFI_TARGETS.add(_GPU4PYSCF_RYS_TARGET_NAME)
        if _PAIR_SCHWARZ_TARGET_NAME not in _REGISTERED_FFI_TARGETS:
            ffi.register_ffi_target(
                _PAIR_SCHWARZ_TARGET_NAME,
                ffi.pycapsule(getattr(lib, "TdGraddftCudaPairSchwarzFfi")),
                platform="CUDA",
                api_version=1,
            )
            _REGISTERED_FFI_TARGETS.add(_PAIR_SCHWARZ_TARGET_NAME)
        if _ERI_TARGET_NAME not in _REGISTERED_FFI_TARGETS:
            ffi.register_ffi_target(
                _ERI_TARGET_NAME,
                ffi.pycapsule(getattr(lib, "TdGraddftCudaEriTensorFfi")),
                platform="CUDA",
                api_version=1,
            )
            _REGISTERED_FFI_TARGETS.add(_ERI_TARGET_NAME)
        if _ERI_PAIR_TARGET_NAME not in _REGISTERED_FFI_TARGETS:
            ffi.register_ffi_target(
                _ERI_PAIR_TARGET_NAME,
                ffi.pycapsule(getattr(lib, "TdGraddftCudaEriPairMatrixFfi")),
                platform="CUDA",
                api_version=1,
            )
            _REGISTERED_FFI_TARGETS.add(_ERI_PAIR_TARGET_NAME)
        if _PAIR_JK_TARGET_NAME not in _REGISTERED_FFI_TARGETS:
            ffi.register_ffi_target(
                _PAIR_JK_TARGET_NAME,
                ffi.pycapsule(getattr(lib, "TdGraddftCudaPairMatrixJkFfi")),
                platform="CUDA",
                api_version=1,
            )
            _REGISTERED_FFI_TARGETS.add(_PAIR_JK_TARGET_NAME)
        _REGISTERED_FFI_LIBS[library] = lib

    def _require_pair_metadata(self) -> None:
        if self.system.nao > 0 and self.system.pair_rows.size == 0:
            raise RuntimeError(
                "This CUDA direct J/K builder was constructed without AO-pair metadata. "
                "Use include_pair_metadata=True for ERI, pair-Schwarz, or screened direct-SCF paths."
            )

    def build_pair_schwarz(self) -> Array:
        self._require_pair_metadata()
        if self._pair_schwarz is not None:
            return self._pair_schwarz
        npair = self.system.nao * (self.system.nao + 1) // 2
        shape = jax.ShapeDtypeStruct((npair,), jnp.float64)
        bounds = _ffi_call(
            _PAIR_SCHWARZ_TARGET_NAME,
            shape,
            self.centers,
            self.angulars,
            self.pair_exponents,
            self.pair_centers,
            self.pair_prefactors,
            self.pair_nprims,
            self.pair_rows,
            self.pair_cols,
        )
        self._pair_schwarz = _pool_pair_schwarz_by_screen_group(
            jnp.asarray(bounds, dtype=jnp.float64),
            self.pair_screen_group_ids,
            self.system.n_pair_screen_groups,
        )
        return self._pair_schwarz

    def build_shell_log_q_matrix(self) -> np.ndarray:
        if self._shell_log_q_matrix is None:
            self._shell_log_q_matrix = _build_shell_log_q_matrix(
                self.system.shell_layout,
                self.system.pair_screen_representative_ids,
                self.build_pair_schwarz(),
            )
        return self._shell_log_q_matrix

    def build_padded_log_q_matrix(self) -> np.ndarray:
        return _build_padded_log_q_matrix(
            self.system.shell_layout,
            self.build_shell_log_q_matrix(),
        )

    def build_rys_log_q_matrix(self) -> np.ndarray:
        if self._rys_log_q_matrix is None:
            self._rys_log_q_matrix = _build_rys_log_q_matrix(
                self.system.shell_layout,
                self.build_shell_log_q_matrix(),
            )
        return self._rys_log_q_matrix

    def build_shell_density_condition(self, density: Array) -> Array:
        return _build_shell_density_condition(
            density,
            self.shell_ao_indices_padded,
            self.shell_ao_sizes,
        )

    def build_joltqc_tile_pairs(self, *, cutoff: float) -> dict[tuple[int, int], np.ndarray]:
        return _make_joltqc_tile_pairs(
            self.system.shell_layout.padded_group_offsets,
            self.build_padded_log_q_matrix(),
            cutoff=float(cutoff),
            tile_size=int(self.system.shell_layout.tile_size),
        )

    def build_joltqc_tile_pair_layout(self, *, cutoff: float) -> CudaTilePairLayout:
        return _pack_joltqc_tile_pairs(self.build_joltqc_tile_pairs(cutoff=cutoff))

    def build_full_joltqc_tile_pair_layout(self) -> CudaTilePairLayout:
        if self._full_tile_pair_layout is None:
            self._full_tile_pair_layout = _pack_joltqc_tile_pairs(
                _make_full_joltqc_tile_pairs(
                    self.system.shell_layout.padded_group_offsets,
                    tile_size=int(self.system.shell_layout.tile_size),
                )
            )
        return self._full_tile_pair_layout

    def build_rys_pair_mapping_layout(self, *, cutoff: float) -> CudaRysPairMappingLayout:
        return _pack_rys_pair_mappings(
            _make_rys_tril_pair_mappings(
                self.rys_env_layout.group_offsets,
                self.build_rys_log_q_matrix(),
                cutoff=float(cutoff),
            )
        )

    def build_full_rys_pair_mapping_layout(self) -> CudaRysPairMappingLayout:
        if self._full_rys_pair_mapping_layout is None:
            nshell = int(self.rys_env_layout.ao_loc.shape[0] - 1)
            full_q = np.zeros((nshell, nshell), dtype=np.float32)
            self._full_rys_pair_mapping_layout = _pack_rys_pair_mappings(
                _make_rys_tril_pair_mappings(
                    self.rys_env_layout.group_offsets,
                    full_q,
                    cutoff=-np.inf,
                )
            )
        return self._full_rys_pair_mapping_layout

    def precompute_screening_metadata(self) -> None:
        self.build_pair_schwarz()

    def build_eri_pair_matrix(self) -> Array:
        self._require_pair_metadata()
        npair = self.system.nao * (self.system.nao + 1) // 2
        shape = jax.ShapeDtypeStruct((npair, npair), jnp.float64)
        cutoff = _cuda_pair_eri_build_cutoff()
        pair_schwarz = (
            self.build_pair_schwarz()
            if cutoff > 0.0
            else jnp.ones((npair,), dtype=jnp.float64)
        )
        cutoff_arr = jnp.asarray([cutoff], dtype=jnp.float64)
        pair = _ffi_call(
            _ERI_PAIR_TARGET_NAME,
            shape,
            self.centers,
            self.angulars,
            self.pair_exponents,
            self.pair_centers,
            self.pair_prefactors,
            self.pair_nprims,
            self.pair_rows,
            self.pair_cols,
            pair_schwarz,
            cutoff_arr,
        )
        return jnp.asarray(pair, dtype=jnp.float64)

    def build_jk_from_eri_pair_matrix(
        self,
        eri_pair_matrix: Array,
        density: Array,
    ) -> tuple[Array, Array]:
        return build_jk_from_eri_pair_matrix_cuda(
            eri_pair_matrix,
            density,
            self.pair_rows,
            self.pair_cols,
        )

    def build_jk_with_joltqc_basis_data(
        self,
        density: Array,
        basis_data: Array,
    ) -> tuple[Array, Array]:
        return _cuda_joltqc_direct_jk_with_density_vjp(
            density,
            jnp.asarray(basis_data, dtype=jnp.float64),
            self.joltqc_shell_l,
            self.joltqc_shell_nprims,
            self.joltqc_ao_to_parent_ao,
            self.joltqc_group_keys,
            self.joltqc_group_quartet_keys,
            self.joltqc_group_quartet_offsets,
            self.joltqc_shell_quartets,
        )

    def build_jk_with_gpu4pyscf_rys(self, density: Array) -> tuple[Array, Array]:
        pair_layout = self.build_full_rys_pair_mapping_layout()
        nshell = int(self.rys_env_layout.ao_loc.shape[0] - 1)
        zeros = jnp.zeros((nshell, nshell), dtype=jnp.float32)
        return _cuda_gpu4pyscf_rys_direct_jk_with_density_vjp(
            density,
            self.rys_atm,
            self.rys_bas,
            self.rys_env,
            self.rys_ao_loc,
            self.rys_ao_to_parent_ao,
            self.rys_group_offsets,
            zeros,
            zeros,
            jnp.asarray(pair_layout.group_pair_offsets, dtype=jnp.int32),
            jnp.asarray(pair_layout.pair_ids, dtype=jnp.int32),
            jnp.asarray([-np.inf], dtype=jnp.float32),
        )

    def build_eri_tensor(self) -> Array:
        self._require_pair_metadata()
        shape = jax.ShapeDtypeStruct(
            (self.system.nao, self.system.nao, self.system.nao, self.system.nao),
            jnp.float64,
        )
        eri = _ffi_call(
            _ERI_TARGET_NAME,
            shape,
            self.centers,
            self.angulars,
            self.pair_exponents,
            self.pair_centers,
            self.pair_prefactors,
            self.pair_nprims,
            self.pair_rows,
            self.pair_cols,
        )
        return jnp.asarray(eri, dtype=jnp.float64)

    def build_jk(self, density: Array, *, density_cutoff: float = 0.0) -> tuple[Array, Array]:
        self._require_pair_metadata()
        npair = self.system.nao * (self.system.nao + 1) // 2
        cutoff = float(density_cutoff)
        if cutoff <= 0.0:
            if bool(self.has_gpu4pyscf_rys):
                return self.build_jk_with_gpu4pyscf_rys(density)
            return _cuda_direct_jk_with_density_vjp(
                density,
                self.centers,
                self.angulars,
                self.pair_exponents,
                self.pair_centers,
                self.pair_prefactors,
                self.pair_nprims,
                self.pair_rows,
                self.pair_cols,
                jnp.ones((npair,), dtype=jnp.float64),
                jnp.asarray([0.0], dtype=jnp.float64),
            )

        pair_schwarz = self.build_pair_schwarz()
        cutoff_arr = jnp.asarray([math.log(cutoff)], dtype=jnp.float64)
        shell_log_q_matrix = jnp.asarray(self.build_shell_log_q_matrix(), dtype=jnp.float64)
        shell_dm_cond = self.build_shell_density_condition(_symmetrize_density_like(density))
        tile_layout = self.build_full_joltqc_tile_pair_layout()
        return _cuda_screened_direct_jk_with_density_vjp(
            density,
            self.centers,
            self.angulars,
            self.pair_exponents,
            self.pair_centers,
            self.pair_prefactors,
            self.pair_nprims,
            self.pair_rows,
            self.pair_cols,
            pair_schwarz,
            cutoff_arr,
            shell_log_q_matrix,
            shell_dm_cond,
            self.shell_ao_indices_padded,
            self.shell_ao_sizes,
            self.tile_shell_indices,
            self.tile_shell_pad_mask,
            jnp.asarray(tile_layout.tile_pair_ids, dtype=jnp.int32),
        )
