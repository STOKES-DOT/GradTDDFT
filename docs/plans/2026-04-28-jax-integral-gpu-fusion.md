# JAX 积分引擎 GPU 融合优化方案

**日期:** 2026-04-28
**范围:** `src/td_graddft/data/integrals/two_electron.py`
**约束:** 仅针对 `integral_backend="jax"` 路径，不新增后端分支

> **Implementation status (2026-04-29):** 核心融合逻辑已实现。`_fused_eri_pair_matrix_from_shell_groups` 和 `_fused_eri_pair_matrix_from_ao_groups` 已通过 `lax.scan` + `lax.switch` 替换了 for-loop。额外实现了 `_signature_limited_group_chunks` 用于智能分组（处理 >64 signatures 和 padding ratio 控制）。**d 轨道 shell-block 扩展未实现**——`_can_use_shell_block_path` 仍为 `max_l > 1`，含 d 轨道的基组仍回退到 AO-pair 路径。

---

## 1. 问题定位

### 1.1 核心瓶颈：Python for-loop 导致 O(N_groups) 次 GPU kernel launch

积分引擎中 **3 个函数** 包含 `for group in groups` 的 Python 循环，每次迭代触发至少 1 次 GPU kernel launch（ERI 计算），对 s/p shell-block 路径还额外触发 1 次 scatter kernel：

| 函数 | 文件行号 | 触发路径 |
|------|----------|----------|
| `_eri_tensor_shell_block` | `two_electron.py:852` | s/p shell-block，`can_use_shell_block_path=True` |
| `_eri_pair_matrix_packed_shell_block` | `two_electron.py:903` | s/p shell-block，`can_use_shell_block_path=True` |
| `eri_pair_matrix_packed` 后半段 | `two_electron.py:1230` | AO-pair 回退路径，`can_use_shell_block_path=False` |

以 6-31G* 苯分子为例：~20-30 个 shell quartet group × 每个 group 1-2 次 kernel launch = **40-60 次独立 GPU kernel launch**，仅用于构建单个 ERI 矩阵。

### 1.2 瓶颈放大：d 轨道回退到 AO-pair 路径

`_can_use_shell_block_path`（line 527-546）在 `max_l > 1` 时返回 `False`：

```python
def _can_use_shell_block_path(basis, screening_threshold, engine):
    max_l = max((sum(ang) for shell in basis.shells for ang in shell.angulars))
    if max_l > 1:   # d 轨道及以上直接拒绝
        return False
```

这导致所有含 d 轨道的基组（def2-SVP、6-31G*、cc-pVDZ 等）回退到 AO-pair 路径。AO-pair 路径同样是 Python for-loop + per-group kernel launch，且以单个 AO（而非 shell）为粒度分组，group 数量是 shell-block 路径的 3-10 倍。

### 1.3 辅助瓶颈

- **Scatter kernel 的 8 倍内存膨胀**（`two_electron.py:249-260`）：为利用 8-fold ERI 对称性，将 `blocks` 显式 concatenate 为 8 份 permutation copies，导致 8x 内存膨胀 + 8x scatter 写入
- **Weight contraction 的中间张量膨胀**（`two_electron.py:745-758`）：`einsum("ai,bj,ck,dl->abcdijkl", ...)` 产生大小为 `ni×nj×nk×nl×nprim^4` 的稠密中间张量
- **`_run_quartet_kernel_chunked` 的 pmap padding 浪费**（`two_electron.py:1011-1022`）：多 GPU 下 padding 到 device 倍数，浪费 15-50% 计算

---

## 2. 技术方案

### 2.1 总策略：`lax.scan` + `lax.switch` 融合全部 group 为一次 XLA computation

**原理：** 预处理阶段将所有 shell quartet group 的 batch inputs 对齐到统一大小，用一个 `jax.lax.scan` 遍历所有 group。scan 的 body 内用 `jax.lax.switch` 分派到对应 signature 的预编译 kernel；scan 结束后，一次性 scatter 写入结果矩阵。

### 2.2 实施步骤

#### 步骤 A：预注册所有 signature 的 kernel 并分配 ID

```python
# two_electron.py 新增函数
def _build_signature_registry(groups, kernel_builder):
    """为所有 group 的唯一 signature 分配整数 ID，并预编译对应 kernel。

    Args:
        groups: basis.shell_quartet_groups 或 basis.quartet_groups
        kernel_builder: _compiled_eri_shell_block_kernel_batched
                        或 _compiled_eri_pair_kernel_batched

    Returns:
        sig_to_id: dict mapping signature tuple -> int
        kernels:   tuple of JIT-compiled kernel functions, indexable by int ID
    """
    sig_to_id = {}
    kernels = []
    for group in groups:
        sig = group.signature
        if sig not in sig_to_id:
            sig_to_id[sig] = len(kernels)
            kernels.append(kernel_builder(*sig))
    assert len(kernels) <= 64, (
        f"lax.switch supports at most 64 branches, got {len(kernels)}"
    )
    return sig_to_id, tuple(kernels)
```

#### 步骤 B：将所有 group 的输入对齐为统一形状

```python
# two_electron.py 新增函数
def _pad_and_stack_group_inputs(basis, groups, sig_to_id, max_batch):
    """全部 group 的 batch_inputs pad 到 max_batch 后 stack 为统一张量。

    Returns:
        stacked_inputs:  12 元组，每个元素 shape = (n_groups, max_batch, ...)
        sig_ids:          shape (n_groups,), 每个 group 对应的 kernel ID
        n_valid_list:     shape (n_groups,), 每个 group 的有效 batch size
    """
    n_groups = len(groups)
    all_inputs = []
    sig_ids_list = []
    n_valid_list = []

    for group in groups:
        n = int(group.idx_i.shape[0])
        sig = group.signature
        inputs = _gather_shell_quartet_batch(
            basis,
            group.idx_i, group.idx_j, group.idx_k, group.idx_l,
            nprim_i=sig[4], nprim_j=sig[5], nprim_k=sig[6], nprim_l=sig[7],
        )
        # pad 到 max_batch
        padded = tuple(
            _pad_array_to_size(arr, max_batch) for arr in inputs
        )
        all_inputs.append(padded)
        sig_ids_list.append(sig_to_id[sig])
        n_valid_list.append(n)

    # transpose: 12 个 (n_groups, max_batch, ...)
    stacked = tuple(
        jnp.stack([all_inputs[g][i] for g in range(n_groups)])
        for i in range(12)
    )
    return (
        stacked,
        jnp.array(sig_ids_list, dtype=jnp.int32),
        jnp.array(n_valid_list, dtype=jnp.int32),
    )
```

#### 步骤 C：预计算所有 group 的 pair indices

```python
# two_electron.py 新增函数
def _pad_and_stack_pair_indices(basis, groups, max_batch):
    """为每个 group 预计算 packed ERI 的 (row, col) pair indices。

    Returns:
        all_rows: shape (n_groups, max_batch * ni * nj * nk * nl)
        all_cols: shape (n_groups, max_batch * ni * nj * nk * nl)
    """
    n = basis.nao
    ao_pairs = tuple((i, j) for i in range(n) for j in range(i + 1))
    pair_index = np.full((n, n), -1, dtype=np.int32)
    for pos, (i, j) in enumerate(ao_pairs):
        pair_index[i, j] = pos
        pair_index[j, i] = pos
    pair_index_arr = jnp.asarray(pair_index, dtype=jnp.int32)

    all_rows_list, all_cols_list = [], []
    for group in groups:
        sig = group.signature
        ni, nj, nk, nl = len(sig[0]), len(sig[1]), len(sig[2]), len(sig[3])
        batch_size = int(group.idx_i.shape[0])
        ao_i = basis.shell_ao_indices_padded[group.idx_i, :ni]  # (batch, ni)
        ao_j = basis.shell_ao_indices_padded[group.idx_j, :nj]
        ao_k = basis.shell_ao_indices_padded[group.idx_k, :nk]
        ao_l = basis.shell_ao_indices_padded[group.idx_l, :nl]

        rows = pair_index_arr[ao_i[:, :, None], ao_j[:, None, :]]  # (batch, ni, nj)
        cols = pair_index_arr[ao_k[:, :, None], ao_l[:, None, :]]

        # broadcast 到 (batch, ni, nj, nk, nl) → flatten
        block_size = ni * nj * nk * nl
        rows_flat = jnp.broadcast_to(
            rows[:, :, :, None, None], (batch_size, ni, nj, nk, nl)
        ).reshape(batch_size, block_size)
        cols_flat = jnp.broadcast_to(
            cols[:, None, None, :, :], (batch_size, ni, nj, nk, nl)
        ).reshape(batch_size, block_size)

        # pad
        padded_size = max_batch * block_size
        rows_pad = jnp.pad(rows_flat.reshape(-1), (0, padded_size - rows_flat.size))
        cols_pad = jnp.pad(cols_flat.reshape(-1), (0, padded_size - cols_flat.size))
        all_rows_list.append(rows_pad)
        all_cols_list.append(cols_pad)

    return jnp.stack(all_rows_list), jnp.stack(all_cols_list)
```

#### 步骤 D：核心——用 `lax.scan` + `lax.switch` 替换 Python for-loop

```python
# two_electron.py 新增函数：统一的融合 ERI 构建
def _fused_eri_pair_matrix_from_groups(basis, groups):
    """单次 XLA launch 构建完整 packed ERI 矩阵。

    合并原来 _eri_pair_matrix_packed_shell_block 的 for-loop 逻辑
    和 eri_pair_matrix_packed 后半段 group 迭代逻辑。
    """
    n = basis.nao
    npair = n * (n + 1) // 2

    # 1) 注册 signature → kernel ID
    sig_to_id, kernels = _build_signature_registry(
        groups, _compiled_eri_shell_block_kernel_batched
    )

    # 2) 确定 max_batch
    max_batch = max(int(g.idx_i.shape[0]) for g in groups)

    # 3) 对齐输入
    stacked_inputs, sig_ids, n_valid = _pad_and_stack_group_inputs(
        basis, groups, sig_to_id, max_batch
    )

    # 4) 预计算 pair indices
    all_rows, all_cols = _pad_and_stack_pair_indices(
        basis, groups, max_batch
    )

    # 5) 单次 scan 遍历全部 group
    n_groups = len(groups)
    pair_init = jnp.zeros((npair, npair))

    def _scan_body(carry, idx):
        pair = carry
        sig_id = sig_ids[idx]
        nv = n_valid[idx]

        # 取出该 group 的 12 个输入
        batch_12 = tuple(inp[idx] for inp in stacked_inputs)

        # lax.switch 分派到对应 kernel
        # kernel 签名: (exp_i, coeff_i, center_i, ..., center_l) → (max_batch, ni, nj, nk, nl)
        sig = groups[idx].signature
        ni, nj, nk, nl = len(sig[0]), len(sig[1]), len(sig[2]), len(sig[3])
        blocks = jax.lax.switch(sig_id, kernels, *batch_12)
        block_size = ni * nj * nk * nl

        # 只取有效 batch 条目，flatten
        vals = blocks.reshape(max_batch, block_size)[:nv].reshape(-1)

        # 取出对应的 pair indices（同样只取有效部分）
        rows = all_rows[idx, :nv * block_size]
        cols = all_cols[idx, :nv * block_size]

        pair = pair.at[rows, cols].set(vals)
        pair = pair.at[cols, rows].set(vals)
        return pair, None

    pair, _ = jax.lax.scan(_scan_body, pair_init, jnp.arange(n_groups))
    return 0.5 * (pair + pair.T)
```

#### 步骤 E：同样处理 4-index full ERI tensor

```python
# two_electron.py 新增函数
def _fused_eri_tensor_from_groups(basis, groups):
    """单次 XLA launch 构建完整 4-index ERI tensor。

    合并原来 _eri_tensor_shell_block 的 for-loop 逻辑。
    """
    n = basis.nao
    sig_to_id, kernels = _build_signature_registry(
        groups, _compiled_eri_shell_block_kernel_batched
    )
    max_batch = max(int(g.idx_i.shape[0]) for g in groups)
    stacked_inputs, sig_ids, n_valid = _pad_and_stack_group_inputs(
        basis, groups, sig_to_id, max_batch
    )

    # 预计算 4-index scatter indices
    all_scatter = _pad_and_stack_4index_scatter(basis, groups, max_batch)

    eri_init = jnp.zeros((n, n, n, n))
    n_groups = len(groups)

    def _scan_body(carry, idx):
        eri = carry
        sig_id = sig_ids[idx]
        nv = n_valid[idx]
        batch_12 = tuple(inp[idx] for inp in stacked_inputs)
        blocks = jax.lax.switch(sig_id, kernels, *batch_12)

        scatter_i, scatter_j, scatter_k, scatter_l = all_scatter  # each (n_groups, padded_flat_size)
        sig = groups[idx].signature
        block_size = len(sig[0]) * len(sig[1]) * len(sig[2]) * len(sig[3])

        # 将 blocks 的 8-fold 对称性 flatten（不复制，按需 scatter）
        vals = _symmetrized_block_values(blocks.reshape(max_batch, -1).reshape(-1))
        # 每个 block 对应 8 倍 scatter，这里取对应 slice
        flat_size = max_batch * block_size * 8
        si = scatter_i[idx, :flat_size][: nv * block_size * 8]
        sj = scatter_j[idx, :flat_size][: nv * block_size * 8]
        sk = scatter_k[idx, :flat_size][: nv * block_size * 8]
        sl = scatter_l[idx, :flat_size][: nv * block_size * 8]
        vv = vals[: nv * block_size * 8]

        eri = eri.at[si, sj, sk, sl].set(vv)
        return eri, None

    eri, _ = jax.lax.scan(_scan_body, eri_init, jnp.arange(n_groups))
    return eri
```

> **注意：** 步骤 E 中 4-index ERI tensor 的 scatter indices 预处理函数 `_pad_and_stack_4index_scatter` 可以利用已有 `group.scatter_i/j/k/l` 属性，这些在 `CartesianBasis` 构建时已预计算。

#### 步骤 F：修改调用点

**修改 `_eri_pair_matrix_packed_shell_block`**（line 880-940）：

```python
def _eri_pair_matrix_packed_shell_block(basis, *, engine):
    n = basis.nao
    nshell = len(basis.shells)
    if n == 0 or nshell == 0:
        return jnp.zeros((0, 0))
    groups = basis.shell_quartet_groups
    if not groups:
        raise ValueError(
            "Shell-block packed ERI path requires precomputed shell quartet groups."
        )
    # 原: for group in groups: ...  →  改为一次调用
    return _fused_eri_pair_matrix_from_groups(basis, groups)
```

**修改 `_eri_tensor_shell_block`**（line 836-877）：

```python
def _eri_tensor_shell_block(basis, *, engine):
    n = basis.nao
    nshell = len(basis.shells)
    if n == 0 or nshell == 0:
        return jnp.zeros((n, n, n, n))
    groups = basis.shell_quartet_groups
    if not groups:
        raise ValueError(
            "Shell-block ERI path requires precomputed shell quartet groups."
        )
    return _fused_eri_tensor_from_groups(basis, groups)
```

**修改 `eri_pair_matrix_packed`**（line 1230 附近的 for-loop 段落）：

AO-pair 回退路径的 for-loop 段落（当前为 `for group in groups: ...`，约 line 1230-1280）改为类似的融合调用。此时 kernel builder 使用 `_compiled_eri_pair_kernel_batched` 而非 `_compiled_eri_shell_block_kernel_batched`。

### 2.3 扩展 d 轨道走 shell-block 路径

`_compiled_eri_shell_block_kernel` 内部的 VRR 递推（`compute_vrr`）本身已支持任意 angular momentum——递推终止条件 `sum(a) + sum(c) == 0` 不依赖 `max_l`。唯一的限制来自外层 `_can_use_shell_block_path`。

**修改：** 将 `max_l > 1` 放宽为 `max_l > 2`：

```python
def _can_use_shell_block_path(basis, screening_threshold, engine):
    if screening_threshold not in (None, 0.0):
        return False
    if not basis.shells:
        return False
    if not _use_jit_engine(engine):
        return False
    max_l = max((sum(ang) for shell in basis.shells for ang in shell.angulars), default=0)
    if max_l > 2:       # 允许 s/p/d，排除 f 及以上
        return False
    max_nprim = max((int(shell.exponents.shape[0]) for shell in basis.shells), default=0)
    return max_nprim <= 6
```

**同时需要确认 `CartesianBasis` 已正确生成 d 壳层的 `shell_ao_indices_padded` 和 `shell_quartet_groups`。** 这些在 `data/basis.py` 中构建，如果当前限制了 d 壳层的 shell quartet group 生成，需要同步放宽。

---

## 3. 改动清单

| 文件 | 改动类型 | 内容 |
|------|----------|------|
| `two_electron.py` | 新增 4 个函数 | `_build_signature_registry`、`_pad_and_stack_group_inputs`、`_pad_and_stack_pair_indices`、`_fused_eri_pair_matrix_from_groups` |
| `two_electron.py` | 新增 2 个函数 | `_pad_and_stack_4index_scatter`、`_fused_eri_tensor_from_groups` |
| `two_electron.py` | 修改 2 个函数 | `_eri_pair_matrix_packed_shell_block`、`_eri_tensor_shell_block`：移除 for-loop，改为调用融合版本 |
| `two_electron.py` | 修改 1 个函数 | `eri_pair_matrix_packed`：后半段 AO-pair for-loop 改为融合调用 |
| `two_electron.py` | 修改 1 行 | `_can_use_shell_block_path`：`max_l > 1` → `max_l > 2` |
| `two_electron.py` | 删除 | `_compiled_shell_block_scatter_kernel`（不再需要 per-group scatter） |
| `data/basis.py` | 确认/修改 | d 壳层的 `shell_quartet_groups` 是否正确生成 |

**不涉及的文件：** `one_electron.py`、`_common.py`、`screening.py`、`libcint_autodiff.py`、SCF 模块、facade 层。

---

## 4. 回退策略

如果融合版本在特定基组/分子下出现 OOM 或编译超时：

1. 按 `max_batch` 上限拆分：如果 `max_batch > 2048`，拆为多个 scan（类似当前 chunk 逻辑，但 chunk 边界在 Python 层仅决定拆分为几个 scan，每个 scan 内部仍然是单次 XLA computation）
2. 混合路径阈值：`nao > 500` 时回退到原有的 per-group 循环（加 `if nao > 500: return _legacy_path(...)`）

---

## 5. 预期收益

| 场景 | 改动前 | 改动后 |
|------|--------|--------|
| 水 / STO-3G (28 AO, s/p) | ~10 groups × 1-2 launches = 10-20 次 | **1 次** |
| 苯 / 6-31G (72 AO, s/p) | ~20 groups × 1-2 launches = 20-40 次 | **1 次** |
| 苯 / 6-31G* (108 AO, sp+d) | AO-pair 回退，~100+ groups × 1-2 = 100-200+ 次 | **1 次**（d 轨道走 shell-block） |
| 水 / def2-SVP (24 AO, sp+d) | AO-pair 回退，~15 groups | **1 次**（d 轨道走 shell-block） |

**保守估计加速比 3-10x**，取决于分子大小和基组。对小分子和 s/p 基组约 3-5x；对 def2-SVP / 6-31G* 级别的含 d 轨道计算约 5-10x（因为额外避免了 AO-pair 回退路径）。

---

## 6. 验证方式

### 6.1 数值正确性

```python
# 改动前后对同一体系比较 ERI 矩阵的 Frobenius norm 差值
import jax.numpy as jnp
eri_before = eri_pair_matrix_packed_legacy(basis)  # 保留原实现做 reference
eri_after  = eri_pair_matrix_packed(basis)
assert jnp.allclose(eri_before, eri_after, atol=1e-10)
```

### 6.2 性能回归测试

- 使用 `tools/benchmark_integrals_vs_pyscf.py` 做 before/after 计时对比
- 对 STO-3G、6-31G、6-31G*、def2-SVP 四组基组 + water/benzene 两个分子，各测 5 次取中位数
- 同时记录首次 JIT 编译时间和后续调用时间

### 6.3 端到端集成测试

- `pytest tests/` 中所有涉及 `integral_backend="jax"` 的测试必须通过
- 特别关注 `test_density_fitting_rks.py`（大量使用 DF，依赖 ERI 构建）和 `test_differentiable_scf.py`（依赖可微 ERI 路径）

---

## 7. 关键风险

| 风险 | 缓解措施 |
|------|----------|
| `lax.switch` 分支数超过 64 | 按 signature 分组做多个 scan，每组 ≤ 64 个分支 |
| 首次 JIT 编译时间增加 | 保留 `precompile_eri_kernels` 预热逻辑，增加 scan 版本的预热 |
| 全量 pad 到 max_batch 导致显存峰值 | 设置 `max_batch` 上限 2048，超出则拆分为多个 scan |

---

## 8. Implementation Specification

### 8.1 File modifications

**Single file: `src/td_graddft/data/integrals/two_electron.py`**

### 8.2 New functions to add (in implementation order)

**8.2.1 `_build_signature_registry(groups, kernel_builder)`**

```python
@functools.lru_cache(maxsize=None)
def _build_signature_registry(
    groups: tuple,  # tuple of ShellQuartetGroup
    kernel_builder: Callable,  # _compiled_eri_shell_block_kernel_batched
) -> tuple[dict, tuple]:
    """
    Returns:
      sig_to_id: dict mapping signature (8-tuple) -> int (0..k-1)
      kernels:   tuple of k JIT-compiled functions, indexable by int
    """
    sig_to_id = {}
    kernels = []
    for group in groups:
        sig = group.signature
        if sig not in sig_to_id:
            sig_to_id[sig] = len(kernels)
            kernels.append(kernel_builder(*sig))
    if len(kernels) > 64:
        raise ValueError(f"lax.switch supports ≤64 branches, got {len(kernels)}")
    return sig_to_id, tuple(kernels)
```

**8.2.2 `_pad_array_to_size(arr, target_size)`**

```python
def _pad_array_to_size(arr: jnp.ndarray, target_size: int) -> jnp.ndarray:
    """Pad leading axis of arr to target_size by repeating last element."""
    n = int(arr.shape[0])
    if n >= target_size:
        return arr
    pad = target_size - n
    last = arr[n - 1 : n]
    return jnp.concatenate([arr, jnp.repeat(last, pad, axis=0)], axis=0)
```

**8.2.3 `_pad_and_stack_group_inputs(basis, groups, sig_to_id, max_batch)`**

```python
def _pad_and_stack_group_inputs(
    basis: CartesianBasis,
    groups: tuple,  # tuple of ShellQuartetGroup
    sig_to_id: dict,
    max_batch: int,
) -> tuple[tuple[jnp.ndarray, ...], jnp.ndarray, jnp.ndarray]:
    """
    Returns:
      stacked: 12-tuple of (n_groups, max_batch, ...) arrays
      sig_ids: (n_groups,) int32, kernel ID per group
      n_valid: (n_groups,) int32, valid batch size per group
    """
    n_groups = len(groups)
    all_inputs = []
    sig_ids_list = []
    n_valid_list = []

    for group in groups:
        sig = group.signature
        batch = _gather_shell_quartet_batch(
            basis,
            group.idx_i, group.idx_j, group.idx_k, group.idx_l,
            nprim_i=sig[4], nprim_j=sig[5], nprim_k=sig[6], nprim_l=sig[7],
        )
        # batch is 12-tuple of (n_items, ...) arrays
        n = int(batch[0].shape[0])
        padded = tuple(_pad_array_to_size(a, max_batch) for a in batch)
        all_inputs.append(padded)
        sig_ids_list.append(sig_to_id[sig])
        n_valid_list.append(n)

    # transpose: 12 groups of (n_groups, max_batch, ...)
    stacked = tuple(
        jnp.stack([all_inputs[g][i] for g in range(n_groups)])
        for i in range(12)
    )
    return stacked, jnp.array(sig_ids_list, dtype=jnp.int32), jnp.array(n_valid_list, dtype=jnp.int32)
```

**8.2.4 `_pad_and_stack_pair_indices(basis, groups, max_batch)`**

```python
def _pad_and_stack_pair_indices(
    basis: CartesianBasis,
    groups: tuple,  # tuple of ShellQuartetGroup
    max_batch: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns:
      all_rows: (n_groups, max_batch * block_size_flat) int32, packed ERI row indices
      all_cols: (n_groups, max_batch * block_size_flat) int32, packed ERI col indices
    """
    n = basis.nao
    pair_index = jnp.asarray(
        _lower_triangle_pairs_to_matrix(n),  # (nao, nao) -> pair index
        dtype=jnp.int32,
    )

    all_rows_list, all_cols_list = [], []
    for group in groups:
        sig = group.signature
        ni, nj, nk, nl = len(sig[0]), len(sig[1]), len(sig[2]), len(sig[3])
        batch_size = int(group.idx_i.shape[0])
        ao_i = basis.shell_ao_indices_padded[group.idx_i, :ni]
        ao_j = basis.shell_ao_indices_padded[group.idx_j, :nj]
        ao_k = basis.shell_ao_indices_padded[group.idx_k, :nk]
        ao_l = basis.shell_ao_indices_padded[group.idx_l, :nl]

        rows = pair_index[ao_i[:, :, None], ao_j[:, None, :]]  # (batch, ni, nj)
        cols = pair_index[ao_k[:, :, None], ao_l[:, None, :]]  # (batch, nk, nl)

        block_size = ni * nj * nk * nl
        rows_flat = jnp.broadcast_to(
            rows[:, :, :, None, None], (batch_size, ni, nj, nk, nl)
        ).reshape(batch_size, block_size)
        cols_flat = jnp.broadcast_to(
            cols[:, None, None, :, :], (batch_size, ni, nj, nk, nl)
        ).reshape(batch_size, block_size)

        padded_size = max_batch * block_size
        rows_pad = jnp.pad(rows_flat.reshape(-1), (0, padded_size - rows_flat.size))
        cols_pad = jnp.pad(cols_flat.reshape(-1), (0, padded_size - cols_flat.size))
        all_rows_list.append(rows_pad)
        all_cols_list.append(cols_pad)

    return jnp.stack(all_rows_list), jnp.stack(all_cols_list)
```

**Helper: `_lower_triangle_pairs_to_matrix(n)`**

```python
def _lower_triangle_pairs_to_matrix(n: int) -> np.ndarray:
    """Build (nao, nao) matrix mapping AO indices to packed pair index."""
    pairs = _lower_triangle_pairs(n)  # existing function
    mat = np.full((n, n), -1, dtype=np.int32)
    for pos, (i, j) in enumerate(pairs):
        mat[i, j] = pos
        mat[j, i] = pos
    return mat
```

**8.2.5 `_fused_eri_pair_matrix_from_shell_groups(basis, groups)`**

```python
def _fused_eri_pair_matrix_from_shell_groups(
    basis: CartesianBasis,
    groups: tuple,  # tuple of ShellQuartetGroup
) -> jnp.ndarray:
    """Single XLA launch builds packed ERI matrix from shell-quartet groups.

    shape: (npair, npair) where npair = nao * (nao + 1) // 2
    dtype: float64 or float32 depending on basis.exponents dtype
    """
    n = basis.nao
    npair = n * (n + 1) // 2
    sig_to_id, kernels = _build_signature_registry(
        groups, _compiled_eri_shell_block_kernel_batched
    )
    max_batch = max(int(g.idx_i.shape[0]) for g in groups)
    stacked_inputs, sig_ids, n_valid = _pad_and_stack_group_inputs(
        basis, groups, sig_to_id, max_batch
    )
    all_rows, all_cols = _pad_and_stack_pair_indices(basis, groups, max_batch)
    n_groups = len(groups)
    pair_init = jnp.zeros((npair, npair), dtype=stacked_inputs[0].dtype)

    def _scan_body(carry, idx):
        pair_acc = carry
        sig_id = sig_ids[idx]
        nv = n_valid[idx]
        batch_12 = tuple(inp[idx] for inp in stacked_inputs)
        blocks = jax.lax.switch(sig_id, kernels, *batch_12)
        sig = groups[idx].signature
        block_size = len(sig[0]) * len(sig[1]) * len(sig[2]) * len(sig[3])
        vals = blocks.reshape(max_batch, block_size)[:nv].reshape(-1)
        rows = all_rows[idx, :nv * block_size]
        cols = all_cols[idx, :nv * block_size]
        pair_acc = pair_acc.at[rows, cols].set(vals)
        pair_acc = pair_acc.at[cols, rows].set(vals)
        return pair_acc, None

    pair, _ = jax.lax.scan(_scan_body, pair_init, jnp.arange(n_groups))
    return 0.5 * (pair + pair.T)
```

### 8.3 Functions to modify

**8.3.1 `_eri_pair_matrix_packed_shell_block` (line 880-940)**

Replace entire body of for-loop with single call:
```python
def _eri_pair_matrix_packed_shell_block(basis, *, engine):
    n = basis.nao
    nshell = len(basis.shells)
    if n == 0 or nshell == 0:
        return jnp.zeros((0, 0))
    groups = basis.shell_quartet_groups
    if not groups:
        raise ValueError("...")
    return _fused_eri_pair_matrix_from_shell_groups(basis, groups)
```

**8.3.2 `_can_use_shell_block_path` (line 527-546)**

```
❌ NOT IMPLEMENTED. Code still has max_l > 1 at line 549.
   Plan: max_l > 1  →  max_l > 2  (allow s, p, d; exclude f and above)
```

**8.3.3 `_compiled_shell_block_scatter_kernel` (line 237-269)**

Remove entirely — no longer needed.

### 8.4 Implementation order

1. Add `_pad_array_to_size` and `_lower_triangle_pairs_to_matrix` (simple utilities)
2. Add `_build_signature_registry`
3. Add `_pad_and_stack_group_inputs` and `_pad_and_stack_pair_indices`
4. Add `_fused_eri_pair_matrix_from_shell_groups`
5. Modify `_eri_pair_matrix_packed_shell_block` to call it
6. Modify `_can_use_shell_block_path` to allow d orbitals (max_l > 1 → max_l > 2)
7. Modify `_eri_tensor_shell_block` similarly (or defer — s/p 4-index is less used)
8. Remove `_compiled_shell_block_scatter_kernel`
9. Run validation tests before removing old code paths

### 8.5 Test specification

```python
# test_integral_fusion.py
import jax
import jax.numpy as jnp
from td_graddft.data.basis import CartesianBasis, basis_from_molecule_spec
from td_graddft.data.integrals.two_electron import (
    eri_pair_matrix_packed,
    _eri_pair_matrix_packed_shell_block,
    _fused_eri_pair_matrix_from_shell_groups,
)

def test_fused_vs_legacy_water_sto3g():
    """Fused path matches legacy path for H2O/STO-3G."""
    spec = MoleculeSpec(
        symbols=("O", "H", "H"),
        coords_bohr=jnp.array([...]),
        charges=jnp.array([8, 1, 1]),
    )
    basis = basis_from_molecule_spec(spec, basis="sto-3g", max_l=3)
    
    fused = _fused_eri_pair_matrix_from_shell_groups(basis, basis.shell_quartet_groups)
    ref = _eri_pair_matrix_packed_shell_block_legacy(basis, engine="jit")
    
    assert jnp.allclose(fused, ref, atol=1e-10)

def test_fused_vs_legacy_benzene_def2svp():
    """Fused path with d orbitals matches legacy AO-pair path."""
    # benzene def2-SVP has d orbitals on C
    # fused should use shell-block path with max_l=2
    # ref should use AO-pair path
    ...

def test_switch_branch_limit():
    """Ensure signature count ≤ 64 for 6-31G* benzene."""
    # Build basis, verify _build_signature_registry doesn't raise
    ...
```
| d 轨道 VRR 的数值精度 | 参考 PySCF/libcint 结果做 cross-validation，保留 atol=1e-10 |
| d 轨道 shell quartet groups 未生成 | 在 `data/basis.py` 中确认或补充 d 壳层的 group 生成逻辑 |
