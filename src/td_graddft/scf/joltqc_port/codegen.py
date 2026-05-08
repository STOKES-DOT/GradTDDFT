from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from importlib import resources

import numpy as np

from .util import generate_lookup_table


NPRIM_MAX = 3
BASIS_STRIDE = 12
MAX_SMEM = 48 * 1024


def _padded_stride(n: int) -> int:
    if n <= 0:
        return 1
    if n % 32 == 0:
        return n + 1
    if n % 16 == 0:
        return n + 1
    if n % 8 == 0 and n >= 8:
        return n + 1
    return n


def _create_scheme(
    ang: tuple[int, int, int, int],
    *,
    frags: np.ndarray | None = None,
    max_shared_memory: int = MAX_SMEM,
    max_gout: int = 128,
    max_threads: int = 256,
    dtype: np.dtype = np.dtype(np.float64),
) -> tuple[int, int, np.ndarray, int]:
    li, lj, lk, ll = (int(value) for value in ang)
    nroots = (li + lj + lk + ll) // 2 + 1
    nf = np.asarray(
        [
            (li + 1) * (li + 2) // 2,
            (lj + 1) * (lj + 2) // 2,
            (lk + 1) * (lk + 2) // 2,
            (ll + 1) * (ll + 2) // 2,
        ],
        dtype=np.int32,
    )
    if frags is None:
        frags = np.ones(4, dtype=np.int32)
        nthreads = (nf + frags - 1) // frags
        frags1 = frags.copy()
        while int(np.prod(frags1)) < max_gout:
            frags = frags1.copy()
            nthreads = (nf + frags - 1) // frags
            if np.all(nthreads == 1):
                break
            frags1[int(np.argmax(nthreads))] += 1
        while int(np.prod(nthreads)) > max_threads:
            frags[int(np.argmax(nthreads))] += 1
            nthreads = (nf + frags - 1) // frags
    else:
        frags = np.asarray(frags, dtype=np.int32)
        if frags.shape != (4,):
            raise ValueError("JoltQC 1qnt fragments must have shape (4,)")
        if np.any(frags <= 0):
            raise ValueError("JoltQC 1qnt fragments must be positive")
        nthreads = (nf + frags - 1) // frags

    g_size = (li + 1) * (lj + 1) * (lk + 1) * (ll + 1)
    dtype_size = np.dtype(dtype).itemsize
    smem_per_quartet = g_size * 3 + nroots * 2
    nti, ntj, ntk, ntl = (int(value) for value in nthreads)
    nfi, nfj, nfk, nfl = (int(value) for value in nf)

    if nti * ntj > 1:
        smem_per_quartet = max(smem_per_quartet, nti * ntj * nfk * nfl)
    if ntk * ntl > 1:
        smem_per_quartet = max(smem_per_quartet, ntk * ntl * nfi * nfj)
    if nti * ntk > 1:
        smem_per_quartet = max(smem_per_quartet, nti * ntk * nfj * nfl)
    if ntj * ntl > 1:
        smem_per_quartet = max(smem_per_quartet, ntj * ntl * nfi * nfk)
    if nti * ntl > 1:
        smem_per_quartet = max(smem_per_quartet, nti * ntl * nfj * nfk)
    if ntj * ntk > 1:
        smem_per_quartet = max(smem_per_quartet, ntj * ntk * nfi * nfl)

    nt = int(np.prod(nthreads))
    nthreads_per_sq = 1 << (nt - 1).bit_length()
    nsq_per_block = max_threads // nthreads_per_sq
    static_sm = nfk * nfl * 3 * 4
    max_dynamic_sm = max_shared_memory - static_sm
    smem_stride = _padded_stride(nsq_per_block)
    while smem_per_quartet * smem_stride * dtype_size > max_dynamic_sm:
        nsq_per_block >>= 1
        smem_stride = _padded_stride(nsq_per_block)
        if nsq_per_block == 0:
            raise RuntimeError("Shared memory is not enough for JoltQC 1qnt kernel")
    nthreads_per_sq = max_threads // nsq_per_block
    if nthreads_per_sq < 3:
        nthreads_per_sq = 4
        nsq_per_block = max_threads // nthreads_per_sq
        smem_stride = _padded_stride(nsq_per_block)
        while smem_per_quartet * smem_stride * dtype_size > max_dynamic_sm:
            nsq_per_block >>= 1
            smem_stride = _padded_stride(nsq_per_block)
            if nsq_per_block == 0:
                raise RuntimeError("Shared memory is not enough for JoltQC 1qnt kernel")
        nthreads_per_sq = max_threads // nsq_per_block
    dynamic_shared_memory = smem_per_quartet * smem_stride * dtype_size
    return int(nsq_per_block), int(nthreads_per_sq), frags, int(dynamic_shared_memory)


@lru_cache(maxsize=None)
def _read_port_source(name: str) -> str:
    return resources.files(__package__).joinpath(name).read_text()


@lru_cache(maxsize=1)
def _optimal_fp64_fragments() -> dict[str, list[int]]:
    return json.loads(_read_port_source("optimal_scheme_fp64.json"))


def _optimal_fragment_for_angular(ang: tuple[int, int, int, int]) -> np.ndarray | None:
    li, lj, lk, ll = (int(value) for value in ang)
    key = str(li * 1000 + lj * 100 + lk * 10 + ll)
    values = _optimal_fp64_fragments().get(key)
    if values is None:
        return None
    return np.asarray(values, dtype=np.int32)


def build_1qnt_source(
    ang: tuple[int, int, int, int],
    nprim: tuple[int, int, int, int],
    *,
    frags: np.ndarray | None = None,
    dtype: np.dtype = np.dtype(np.float64),
    n_dm: int = 1,
    do_j: bool = True,
    do_k: bool = True,
    max_shared_memory: int = MAX_SMEM,
) -> str:
    """Build static CUDA source for one JoltQC 1qnt angular/primitive signature."""

    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.float64):
        dtype_cuda = "double"
    elif dtype == np.dtype(np.float32):
        dtype_cuda = "float"
    else:
        raise TypeError("JoltQC 1qnt source supports only float64 or float32")

    li, lj, lk, ll = (int(value) for value in ang)
    npi, npj, npk, npl = (int(value) for value in nprim)
    nroots = (li + lj + lk + ll) // 2 + 1
    if not 1 <= nroots <= 9:
        raise ValueError(f"Unsupported JoltQC Rys root count: {nroots}")
    nsq_per_block, nthreads_per_sq, frags, _ = _create_scheme(
        (li, lj, lk, ll),
        frags=frags,
        dtype=dtype,
        max_shared_memory=max_shared_memory,
    )
    fragi, fragj, fragk, fragl = (int(value) for value in frags)
    const = f"""
typedef unsigned int uint32_t;
using DataType = {dtype_cuda};
constexpr int li = {li};
constexpr int lj = {lj};
constexpr int lk = {lk};
constexpr int ll = {ll};
constexpr int npi = {npi};
constexpr int npj = {npj};
constexpr int npk = {npk};
constexpr int npl = {npl};
constexpr int fragi = {fragi};
constexpr int fragj = {fragj};
constexpr int fragk = {fragk};
constexpr int fragl = {fragl};
constexpr int n_dm = {int(n_dm)};
constexpr int rys_type = 0;
constexpr int nsq_per_block = {nsq_per_block};
constexpr int nthreads_per_sq = {nthreads_per_sq};
constexpr int threads = nsq_per_block * nthreads_per_sq;
constexpr int do_j = {int(bool(do_j))};
constexpr int do_k = {int(bool(do_k))};
constexpr int smem_stride = {_padded_stride(nsq_per_block)};
#define NPRIM_MAX {NPRIM_MAX}
#define BASIS_STRIDE {BASIS_STRIDE}
constexpr int nroots = ((li+lj+lk+ll)/2+1);
"""
    return (
        const
        + _read_port_source(f"rys_root{nroots}.cu")
        + _read_port_source("rys_roots_parallel.cu")
        + generate_lookup_table(li, lj, lk, ll)
        + _read_port_source("1qnt.cu")
    )


def build_1q1t_source(
    ang: tuple[int, int, int, int],
    nprim: tuple[int, int, int, int],
    *,
    dtype: np.dtype = np.dtype(np.float64),
    n_dm: int = 1,
    do_j: bool = True,
    do_k: bool = True,
) -> str:
    """Build static CUDA source for one JoltQC 1q1t angular/primitive signature."""

    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.float64):
        dtype_cuda = "double"
    elif dtype == np.dtype(np.float32):
        dtype_cuda = "float"
    else:
        raise TypeError("JoltQC 1q1t source supports only float64 or float32")

    li, lj, lk, ll = (int(value) for value in ang)
    npi, npj, npk, npl = (int(value) for value in nprim)
    nroots = (li + lj + lk + ll) // 2 + 1
    if not 1 <= nroots <= 9:
        raise ValueError(f"Unsupported JoltQC Rys root count: {nroots}")
    const = f"""
typedef unsigned int uint32_t;
using DataType = {dtype_cuda};
constexpr int li = {li};
constexpr int lj = {lj};
constexpr int lk = {lk};
constexpr int ll = {ll};
constexpr int npi = {npi};
constexpr int npj = {npj};
constexpr int npk = {npk};
constexpr int npl = {npl};
constexpr int n_dm = {int(n_dm)};
constexpr int rys_type = 0;
constexpr int do_j = {int(bool(do_j))};
constexpr int do_k = {int(bool(do_k))};
constexpr int nsq_per_block = 256;
#define NPRIM_MAX {NPRIM_MAX}
#define BASIS_STRIDE {BASIS_STRIDE}
constexpr int nroots = ((li+lj+lk+ll)/2+1);
"""
    return (
        const
        + _read_port_source(f"rys_root{nroots}.cu")
        + _read_port_source("rys_roots.cu")
        + generate_lookup_table(li, lj, lk, ll)
        + _read_port_source("1q1t.cu")
    )


def _mangle_signature_name(index: int | str, algorithm: str) -> str:
    return f"tdg_joltqc_{algorithm}_{index}"


def _signature_token(
    cache_key: tuple[tuple[int, int, int, int], tuple[int, int, int, int]],
) -> str:
    ang, nprim = cache_key
    ang_key = "_".join(str(int(value)) for value in ang)
    nprim_key = "_".join(str(int(value)) for value in nprim)
    return f"l_{ang_key}_p_{nprim_key}"


def _kernel_launch_parameters(
    ang: tuple[int, int, int, int],
    *,
    frags: np.ndarray | None = None,
) -> tuple[int, int, int]:
    nsq_per_block, nthreads_per_sq, _, dynamic_shared_memory = _create_scheme(
        tuple(int(value) for value in ang),
        frags=frags,
        dtype=np.dtype(np.float64),
    )
    return nsq_per_block, nthreads_per_sq, dynamic_shared_memory


def _rename_1qnt_kernel(source: str, kernel_name: str) -> str:
    renamed = source.replace(
        'extern "C" __global__\nvoid rys_1qnt_vjk',
        f"__global__\nvoid {kernel_name}",
    )
    renamed = renamed.replace(
        "const ushort4* __restrict__ shl_quartet_idx",
        "const int4* __restrict__ shl_quartet_idx",
    )
    renamed = renamed.replace("ushort4 sq = {0,0,0,0};", "int4 sq = {0,0,0,0};")
    return renamed


def _rename_1q1t_kernel(source: str, kernel_name: str) -> str:
    renamed = source.replace(
        'extern "C" __global__\nvoid rys_1q1t_vjk',
        f"__global__\nvoid {kernel_name}",
    )
    renamed = renamed.replace(
        "const ushort4* __restrict__ shl_quartet_idx",
        "const int4* __restrict__ shl_quartet_idx",
    )
    renamed = renamed.replace("ushort4 sq = {0,0,0,0};", "int4 sq = {0,0,0,0};")
    return renamed


def _signature_key_from_group_ids(
    group_keys: np.ndarray,
    group_ids: list[int],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    signature = [tuple(int(x) for x in group_keys[group_id]) for group_id in group_ids]
    ang = tuple(item[0] for item in signature)
    nprim = tuple(item[1] for item in signature)
    return ang, nprim


def _dispatch_arrays(
    group_keys: np.ndarray,
    group_quartet_keys: np.ndarray,
    group_quartet_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group_keys_arr = np.asarray(group_keys, dtype=np.int32)
    quartet_keys_arr = np.asarray(group_quartet_keys, dtype=np.int32)
    quartet_offsets_arr = np.asarray(group_quartet_offsets, dtype=np.int32)
    if group_keys_arr.ndim != 2 or group_keys_arr.shape[1] != 2:
        raise ValueError("group_keys must have shape (ngroups, 2)")
    if quartet_keys_arr.ndim != 2 or quartet_keys_arr.shape[1] != 4:
        raise ValueError("group_quartet_keys must have shape (nquartet_groups, 4)")
    if quartet_offsets_arr.shape != (quartet_keys_arr.shape[0] + 1,):
        raise ValueError("group_quartet_offsets must have shape (nquartet_groups + 1,)")
    return group_keys_arr, quartet_keys_arr, quartet_offsets_arr


def build_fixed_1qnt_dispatch_arrays(
    *,
    max_l: int,
    nprim_max: int = NPRIM_MAX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a signature-complete dispatch metadata set.

    The arrays are a compile-time signature universe, not a molecule layout.
    Runtime molecule layouts still provide their own quartet offsets and shell
    quartets through the FFI call, mirroring GPU4PySCF's fixed-kernel/runtime
    metadata split.
    """

    if int(max_l) < 0:
        raise ValueError("max_l must be non-negative")
    if int(nprim_max) <= 0:
        raise ValueError("nprim_max must be positive")
    if int(nprim_max) > NPRIM_MAX:
        raise ValueError(f"nprim_max cannot exceed JoltQC NPRIM_MAX={NPRIM_MAX}")
    group_keys = np.asarray(
        [
            (l_value, nprim)
            for l_value in range(int(max_l) + 1)
            for nprim in range(int(nprim_max), 0, -1)
        ],
        dtype=np.int32,
    )
    if group_keys.size == 0:
        return (
            group_keys.reshape(0, 2),
            np.zeros((0, 4), dtype=np.int32),
            np.zeros((1,), dtype=np.int32),
        )
    pair_i, pair_j = np.tril_indices(int(group_keys.shape[0]))
    pair_p, pair_q = np.tril_indices(int(pair_i.shape[0]))
    group_quartet_keys = np.stack(
        (
            pair_i[pair_p],
            pair_j[pair_p],
            pair_i[pair_q],
            pair_j[pair_q],
        ),
        axis=1,
    ).astype(np.int32, copy=False)
    group_quartet_offsets = np.arange(
        int(group_quartet_keys.shape[0]) + 1,
        dtype=np.int32,
    )
    return group_keys, group_quartet_keys, group_quartet_offsets


@lru_cache(maxsize=1)
def _joltqc_dispatch_source_fingerprint() -> str:
    digest = hashlib.sha256()
    digest.update(b"td_graddft_joltqc_dispatch_source_key_v1")
    for name in (
        "codegen.py",
        "util.py",
        "optimal_scheme_fp64.json",
        "1qnt.cu",
        "1q1t.cu",
        "rys_roots.cu",
        "rys_roots_parallel.cu",
        "rys_root1.cu",
        "rys_root2.cu",
        "rys_root3.cu",
        "rys_root4.cu",
        "rys_root5.cu",
        "rys_root6.cu",
        "rys_root7.cu",
        "rys_root8.cu",
        "rys_root9.cu",
    ):
        digest.update(name.encode())
        digest.update(_read_port_source(name).encode())
    return digest.hexdigest()


def _digest_dispatch_array(digest: "hashlib._Hash", name: str, value: np.ndarray) -> None:
    arr = np.ascontiguousarray(np.asarray(value, dtype=np.int32))
    digest.update(name.encode())
    digest.update(str(tuple(int(dim) for dim in arr.shape)).encode())
    digest.update(arr.tobytes())


def build_1qnt_dispatch_source_key(
    group_keys: np.ndarray,
    group_quartet_keys: np.ndarray,
    group_quartet_offsets: np.ndarray,
) -> str:
    """Return a cache key for JoltQC signature kernels.

    The generated CUDA must not depend on molecule-specific quartet offsets.
    Offsets and quartet ordering are runtime dispatch metadata; only the unique
    angular/primitive signatures determine which JoltQC kernels need nvcc.
    """

    group_keys_arr, quartet_keys_arr, _ = _dispatch_arrays(
        group_keys,
        group_quartet_keys,
        group_quartet_offsets,
    )
    digest = hashlib.sha256()
    digest.update(_joltqc_dispatch_source_fingerprint().encode())
    for cache_key in _unique_signature_keys(group_keys_arr, quartet_keys_arr):
        digest.update(_signature_token(cache_key).encode())
    return digest.hexdigest()


def _unique_signature_keys(
    group_keys: np.ndarray,
    group_quartet_keys: np.ndarray,
) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    signatures = {
        _signature_key_from_group_ids(
            group_keys,
            [int(value) for value in key],
        )
        for key in np.asarray(group_quartet_keys, dtype=np.int32).tolist()
    }
    return sorted(signatures, key=_signature_token)


def _kernel_source_for_signature(
    cache_index: int | str,
    cache_key: tuple[tuple[int, int, int, int], tuple[int, int, int, int]],
) -> tuple[str, str, str, int, int, int, str]:
    ang, nprim = cache_key
    optimal_frags = _optimal_fragment_for_angular(ang)
    use_1q1t = (
        optimal_frags is not None
        and optimal_frags.shape == (1,)
        and int(optimal_frags[0]) == -1
    )
    if use_1q1t:
        algorithm = "1q1t"
        kernel_name = _mangle_signature_name(cache_index, algorithm)
        kernel_source = _rename_1q1t_kernel(
            build_1q1t_source(ang, nprim),
            kernel_name,
        )
        nsq_per_block, nthreads_per_sq, dynamic_shared_memory = 256, 1, 256 * 4
    else:
        algorithm = "1qnt"
        kernel_name = _mangle_signature_name(cache_index, algorithm)
        kernel_source = _rename_1qnt_kernel(
            build_1qnt_source(ang, nprim, frags=optimal_frags),
            kernel_name,
        )
        nsq_per_block, nthreads_per_sq, dynamic_shared_memory = _kernel_launch_parameters(
            ang,
            frags=optimal_frags,
        )
    namespace = f"tdg_joltqc_{algorithm}_sig_{cache_index}"
    return (
        algorithm,
        namespace,
        kernel_name,
        nsq_per_block,
        nthreads_per_sq,
        dynamic_shared_memory,
        kernel_source,
    )


def build_1qnt_dispatch_source(
    group_keys: np.ndarray,
    group_quartet_keys: np.ndarray,
    group_quartet_offsets: np.ndarray,
) -> str:
    """Build a single CUDA source file dispatching JoltQC signature kernels."""

    return "\n".join(
        source
        for _, source in build_1qnt_dispatch_source_units(
            group_keys,
            group_quartet_keys,
            group_quartet_offsets,
        )
    )


def build_1qnt_dispatch_source_units(
    group_keys: np.ndarray,
    group_quartet_keys: np.ndarray,
    group_quartet_offsets: np.ndarray,
) -> list[tuple[str, str]]:
    """Build split CUDA source units for basis-specific JoltQC 1qnt dispatch."""

    group_keys_arr, quartet_keys_arr, _ = _dispatch_arrays(
        group_keys,
        group_quartet_keys,
        group_quartet_offsets,
    )

    units: list[tuple[str, str]] = []
    launcher_decls: list[str] = []
    support_terms: list[str] = []
    launch_cases: list[str] = []
    for cache_key in _unique_signature_keys(group_keys_arr, quartet_keys_arr):
        ang, nprim = cache_key
        signature_token = _signature_token(cache_key)
        (
            algorithm,
            namespace,
            kernel_name,
            nsq_per_block,
            nthreads_per_sq,
            dynamic_shared_memory,
            kernel_source,
        ) = _kernel_source_for_signature(signature_token, cache_key)
        launcher_name = f"TdGraddftLaunchJoltQC1qntSignature_{signature_token}"
        source = f"""
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace {namespace} {{
{kernel_source}
}}  // namespace {namespace}

extern "C" cudaError_t {launcher_name}(
    cudaStream_t stream,
    int nao,
    const double* basis_data,
    double* dm,
    double* vj,
    double* vk,
    const int* shell_quartets,
    int start,
    int stop
) {{
    const int ntasks = stop - start;
    if (ntasks <= 0) {{
        return cudaSuccess;
    }}
    dim3 block({nsq_per_block}, {nthreads_per_sq}, 1);
    const int grid = (ntasks + {nsq_per_block} - 1) / {nsq_per_block};
    {namespace}::{kernel_name}<<<grid, block, {dynamic_shared_memory}, stream>>>(
        nao,
        basis_data,
        dm,
        vj,
        vk,
        0.0,
        reinterpret_cast<const int4*>(shell_quartets + 4 * start),
        ntasks);
    return cudaGetLastError();
}}
// angular key: ({", ".join(str(value) for value in ang)})
// primitive key: ({", ".join(str(value) for value in nprim)})
"""
        units.append((f"signature_{signature_token}.cu", source))
        launcher_decls.append(
            f"""
extern "C" cudaError_t {launcher_name}(
    cudaStream_t stream,
    int nao,
    const double* basis_data,
    double* dm,
    double* vj,
    double* vk,
    const int* shell_quartets,
    int start,
    int stop);
"""
        )
        l0, l1, l2, l3 = (int(value) for value in ang)
        p0, p1, p2, p3 = (int(value) for value in nprim)
        condition = (
            f"l0 == {l0} && l1 == {l1} && l2 == {l2} && l3 == {l3} && "
            f"p0 == {p0} && p1 == {p1} && p2 == {p2} && p3 == {p3}"
        )
        support_terms.append(f"({condition})")
        launch_cases.append(
            f"""
    if ({condition}) {{
        return {launcher_name}(
            stream,
            nao,
            basis_data,
            dm,
            vj,
            vk,
            shell_quartets,
            start,
            stop);
    }}
"""
        )

    support_expr = " ||\n           ".join(support_terms) if support_terms else "false"
    main_source = (
        "#include <cuda_runtime.h>\n"
        "#include <cstddef>\n"
        "#include <vector>\n\n"
        + "".join(launcher_decls)
        + f"""
namespace {{

bool TdGraddftJoltQCSignatureSupported(
    int l0,
    int l1,
    int l2,
    int l3,
    int p0,
    int p1,
    int p2,
    int p3
) {{
    return {support_expr};
}}

cudaError_t TdGraddftLaunchJoltQCSignature(
    cudaStream_t stream,
    int nao,
    const double* basis_data,
    double* dm,
    double* vj,
    double* vk,
    const int* shell_quartets,
    int start,
    int stop,
    int l0,
    int l1,
    int l2,
    int l3,
    int p0,
    int p1,
    int p2,
    int p3
) {{
{''.join(launch_cases)}
    return cudaErrorNotSupported;
}}

}}  // namespace

extern "C" cudaError_t TdGraddftLaunchJoltQC1qnt(
    cudaStream_t stream,
    int nao,
    const double* basis_data,
    double* dm,
    double* vj,
    double* vk,
    const int* group_keys,
    int n_groups,
    const int* group_quartet_keys,
    const int* group_quartet_offsets,
    const int* shell_quartets,
    int n_shell_quartets,
    int n_group_quartets
) {{
    if (n_groups < 0 || n_group_quartets < 0 || n_shell_quartets < 0) {{
        return cudaErrorInvalidValue;
    }}
    if (n_group_quartets == 0) {{
        return cudaSuccess;
    }}
    if (n_groups == 0) {{
        return cudaErrorInvalidValue;
    }}

    std::vector<int> host_group_keys(static_cast<std::size_t>(n_groups) * 2);
    std::vector<int> host_group_quartet_keys(static_cast<std::size_t>(n_group_quartets) * 4);
    std::vector<int> host_group_quartet_offsets(static_cast<std::size_t>(n_group_quartets) + 1);

    cudaError_t err = cudaMemcpyAsync(
        host_group_keys.data(),
        group_keys,
        host_group_keys.size() * sizeof(int),
        cudaMemcpyDeviceToHost,
        stream);
    if (err != cudaSuccess) {{
        return err;
    }}
    err = cudaMemcpyAsync(
        host_group_quartet_keys.data(),
        group_quartet_keys,
        host_group_quartet_keys.size() * sizeof(int),
        cudaMemcpyDeviceToHost,
        stream);
    if (err != cudaSuccess) {{
        return err;
    }}
    err = cudaMemcpyAsync(
        host_group_quartet_offsets.data(),
        group_quartet_offsets,
        host_group_quartet_offsets.size() * sizeof(int),
        cudaMemcpyDeviceToHost,
        stream);
    if (err != cudaSuccess) {{
        return err;
    }}
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {{
        return err;
    }}

    if (host_group_quartet_offsets[0] != 0) {{
        return cudaErrorInvalidValue;
    }}
    for (int index = 0; index < n_group_quartets; ++index) {{
        const int start = host_group_quartet_offsets[index];
        const int stop = host_group_quartet_offsets[index + 1];
        if (start < 0 || stop < start || stop > n_shell_quartets) {{
            return cudaErrorInvalidValue;
        }}
        const int g0 = host_group_quartet_keys[4 * index + 0];
        const int g1 = host_group_quartet_keys[4 * index + 1];
        const int g2 = host_group_quartet_keys[4 * index + 2];
        const int g3 = host_group_quartet_keys[4 * index + 3];
        if (g0 < 0 || g0 >= n_groups ||
            g1 < 0 || g1 >= n_groups ||
            g2 < 0 || g2 >= n_groups ||
            g3 < 0 || g3 >= n_groups) {{
            return cudaErrorInvalidValue;
        }}
        const int l0 = host_group_keys[2 * g0 + 0];
        const int l1 = host_group_keys[2 * g1 + 0];
        const int l2 = host_group_keys[2 * g2 + 0];
        const int l3 = host_group_keys[2 * g3 + 0];
        const int p0 = host_group_keys[2 * g0 + 1];
        const int p1 = host_group_keys[2 * g1 + 1];
        const int p2 = host_group_keys[2 * g2 + 1];
        const int p3 = host_group_keys[2 * g3 + 1];
        if (!TdGraddftJoltQCSignatureSupported(l0, l1, l2, l3, p0, p1, p2, p3)) {{
            return cudaErrorNotSupported;
        }}
    }}

    for (int index = 0; index < n_group_quartets; ++index) {{
        const int start = host_group_quartet_offsets[index];
        const int stop = host_group_quartet_offsets[index + 1];
        const int g0 = host_group_quartet_keys[4 * index + 0];
        const int g1 = host_group_quartet_keys[4 * index + 1];
        const int g2 = host_group_quartet_keys[4 * index + 2];
        const int g3 = host_group_quartet_keys[4 * index + 3];
        const int l0 = host_group_keys[2 * g0 + 0];
        const int l1 = host_group_keys[2 * g1 + 0];
        const int l2 = host_group_keys[2 * g2 + 0];
        const int l3 = host_group_keys[2 * g3 + 0];
        const int p0 = host_group_keys[2 * g0 + 1];
        const int p1 = host_group_keys[2 * g1 + 1];
        const int p2 = host_group_keys[2 * g2 + 1];
        const int p3 = host_group_keys[2 * g3 + 1];
        err = TdGraddftLaunchJoltQCSignature(
            stream,
            nao,
            basis_data,
            dm,
            vj,
            vk,
            shell_quartets,
            start,
            stop,
            l0,
            l1,
            l2,
            l3,
            p0,
            p1,
            p2,
            p3);
        if (err != cudaSuccess) {{
            return err;
        }}
    }}
    return cudaSuccess;
}}
"""
    )
    return [("dispatch.cu", main_source), *units]
