# TDDFT 计算 GPU 优化方案

**日期:** 2026-04-28
**范围:** `src/td_graddft/tddft/`、`src/td_graddft/tdscf/`
**约束:** 仅针对 `integral_backend="jax"` 路径，不新增后端分支

> **Implementation status (2026-04-29):** 已实现。`_rep_tensor_to_mo_eri_slices` 带 `@jax.jit`，`effective_tda_eri` 预合并已就绪。

---

## 1. 问题定位

### 1.1 TDDFT 计算流程

```
基态 molecule
  (mo_coeff, mo_energy, ao, grid_weights,
   eri_ovov/eri_ovvo/eri_oovv 或 df_factors 或 rep_tensor)
        │
        ▼
┌──────────────────────────────────────────────────┐
│ build_restricted_response_matrices  [response.py] │
│  ├─ 提取 occ/vir → delta_eps = ε_a - ε_i          │
│  ├─ ERI → MO 切片: eri_ovov, eri_ovvo, eri_oovv  │
│  ├─ Hartree:   A += 2*(ia|jb),  B += 2*(ia|bj)   │
│  ├─ Exchange:  A -= α*(ij|ab), B -= α*(ib|aj)    │
│  └─ XC kernel: f_xc on grid → contraction        │
│     → TDDFTMatrices(A, B, delta_eps)              │
└──────────────────┬───────────────────────────────┘
                   │
           ┌───────┴───────┐
           ▼               ▼
      TDA (tda.py)    Casida (casida.py)
      eigh(A) 或      Ω = (A-B)^{1/2}(A+B)(A-B)^{1/2}
      Davidson(A)     → eigh(Ω) 或 Davidson(Ω)
           │               │
           ▼               ▼
      TDAResult        TDDFTResult
```

整个流程中计算密集的部分集中在 `response.py`（矩阵构建）和 `casida.py`（Casida 对角化）。

### 1.2 核心瓶颈

| 瓶颈 | 位置 | 触发条件 | 影响 |
|------|------|----------|------|
| rep_tensor → MO ERI 切片无 JIT | `response.py:423-451` | 无 DF 且无预计算 MO ERI 时 | 3 次 O(N⁵) einsum + 独立 JIT 编译 |
| Casida `(A-B)^{1/2}` 的 dense eigh | `casida.py:246` | dense Casida 路径 | 完整 dim×dim 对角化 |
| Davidson 迭代的重复 einsum | `response.py:977-1010` | Davidson 路径 | 每轮多次小 einsum |
| Dense eigensolve 阈值 | `tda.py:20-23` | `dim <= 2048` | 训练循环中重复 dense eigh |

---

## 2. 技术方案

### 2.1 rep_tensor → MO ERI 切片：加 JIT + 融合为单次 kernel

**问题：** `response.py:423-451` 和 `_restricted_eri_slices`（line 612-649）中，当 `df_factors` 和预计算的 `eri_ovov/eri_ovvo/eri_oovv` 都不可用时，从 `rep_tensor` 做 MO 变换：

```python
eri_ovov = jnp.einsum("pqrs,pi,qa,rj,sb->iajb", rep_tensor, orbo, orbv, orbo, orbv)
eri_ovvo = jnp.einsum("pqrs,pi,qa,rb,sj->iabj", rep_tensor, orbo, orbv, orbv, orbo)
eri_oovv = jnp.einsum("pqrs,pi,qj,ra,sb->ijab", rep_tensor, orbo, orbo, orbv, orbv)
```

三个独立的 `einsum` 分别触发 JIT 编译和 GPU kernel launch。`rep_tensor` 是 `(nao, nao, nao, nao)` 的 4-index ERI 张量。

**修改：** 将三个 einsum 合并到一个 `@jax.jit` 函数中，利用 XLA 自动融合中间计算：

```python
@jax.jit
def _rep_tensor_to_mo_eri_slices(rep_tensor, orbo, orbv, *, include_oovv=True):
    """单次 XLA launch 完成 rep_tensor → MO ERI 切片变换。"""
    eri_ovov = jnp.einsum(
        "pqrs,pi,qa,rj,sb->iajb",
        rep_tensor, orbo, orbv, orbo, orbv,
        precision=jax.lax.Precision.HIGHEST,
    )
    eri_ovvo = jnp.einsum(
        "pqrs,pi,qa,rb,sj->iabj",
        rep_tensor, orbo, orbv, orbv, orbo,
        precision=jax.lax.Precision.HIGHEST,
    )
    if include_oovv:
        eri_oovv = jnp.einsum(
            "pqrs,pi,qj,ra,sb->ijab",
            rep_tensor, orbo, orbo, orbv, orbv,
            precision=jax.lax.Precision.HIGHEST,
        )
    else:
        eri_oovv = None
    return eri_ovov, eri_ovvo, eri_oovv
```

**修改点：** `response.py` 的 `_restricted_eri_slices` 中 line 612-649 的 rep_tensor 分支，替换为调用上述函数。

**注意：** 此瓶颈仅在**无 DF 且无预计算 MO ERI** 时触发。当前 `build_rks_integral_inputs` 中 `jk_backend="df"` 和 `jk_backend="full"` 都会预计算 `df_factors` 或 `eri_pair_matrix`，response 构建会优先走 `df_factors_to_mo_eri_slices` 路径（response.py line 410-414）。`rep_tensor` 回退路径在实际使用中较少触发，但对 legacy 兼容性仍有价值。

### 2.2 Casida `(A-B)^{1/2}` 计算优化

**问题：** `_matrix_power_symmetric`（`_utils.py`）的实现是：

```python
eigvals, eigvecs = jnp.linalg.eigh(matrix)
return eigvecs @ jnp.diag(eigvals ** power) @ eigvecs.T.conj()
```

一次 `eigh` + 两次矩阵乘法。对于 dim=2000，约 0.3-0.5s。

**修改策略：** 

当走 dense 路径时，合并 `(A-B)^{1/2}` 和后续的 `Ω = metric^T @ (A+B) @ metric` 为一次完整的 Casida 矩阵构建，避免显式存储 metric_factor 的中间稠密矩阵。当前 `casida.py:246-248` 已经做了 `casida_matrix = metric^T @ (A+B) @ metric`，此步骤无法进一步简化。

当走 Davidson 路径时（`dim > 2048` 或 `nroots <= 24`），使用 operator action 避免构建 `(A-B)^{1/2}` 和 `Ω`。当前代码（`casida.py:412-468`）已实现此逻辑。**无需额外修改。**

### 2.3 Davidson A/B action 的 einsum 合并

**问题：** `_restricted_a_action`（response.py:977-993）每轮做 3 次 einsum：

```python
out = x * delta_eps                         # element-wise，无开销
out += 2.0 * einsum("iajb,njb->nia", eri_ovov, x)
out -= alpha * einsum("ijab,njb->nia", eri_oovv, x)  # exchange 项
out += _restricted_xc_action(data, x)       # XC 项
```

`_restricted_b_action` 类似，做 2-3 次 einsum。

**修改：** 对于 Davidson 的 TDA 路径（无 B matrix action），将 Hartree + exchange 预合并。`eri_ovov` 的 exchange 对应项是先 transpose `eri_oovv` 再 contract。ErIR 切片可以预组合：

```python
# 在 _build_restricted_response_operator_data 中预计算
def _build_restricted_tda_effective_eri(data):
    """预合并 TDA A action 的 ERI 贡献，避免每轮 Davidson 重复 transpose。"""
    alpha = jnp.asarray(data.hybrid_fraction)
    # Hartree: 2*(ia|jb), Exchange: -α*(ij|ab) → transpose to (ia|jb) layout
    hartree = 2.0 * data.eri_ovov
    if data.eri_oovv is not None:
        exchange = -alpha * jnp.transpose(data.eri_oovv, (0, 2, 1, 3))
        return hartree + exchange
    return hartree

# 然后在 _restricted_a_action 中：
def _restricted_a_action(data, x):
    effective_eri = data.effective_tda_eri  # 预计算
    out = x * data.delta_eps[None, :, :]
    out = out + jnp.einsum("iajb,njb->nia", effective_eri, x, precision=Precision.HIGHEST)
    return out + _restricted_xc_action(data, x)
```

这避免了每次 Davidson 迭代中对 `eri_oovv` 重复做 transpose，并将 2-3 次 einsum 减为 2 次。

**修改范围：**
- `_RestrictedResponseOperatorData` 新增字段 `effective_tda_eri: Array | None`
- `_build_restricted_response_operator_data` 末尾计算此合并项
- `_restricted_a_action` 用合并项替代原来的两部操作
- `_restricted_b_action` 做类似合并

### 2.4 Dense eigensolve 阈值调整为训练感知

**问题：** `tda.py:20-23` 的 `_prefer_dense_auto_eigensolve` 对 `dim <= 2048` 一律走 dense eigh。对于训练循环（单次 TDDFT 调用花费 < 0.1s），dense eigh 是合理的选择。对于大体系（dim > 2048）需要用 Davidson。

当前阈值合适，无需修改。但需要确保 Davidson 在 auto 模式下的 fallback 逻辑可靠。`tda.py:158-186` 的 Davidson → dense fallback 在 `mode="auto"` 时已实现。

---

## 3. 改动清单

| 文件 | 改动类型 | 内容 |
|------|----------|------|
| `response.py` | 新增 1 个函数 | `_rep_tensor_to_mo_eri_slices`：加 `@jax.jit` 融合 3 个 einsum |
| `response.py` | 修改 `_restricted_eri_slices` | rep_tensor 分支调用新函数 |
| `response.py` | 修改 `_RestrictedResponseOperatorData` | 新增 `effective_tda_eri` 和 `effective_b_eri` 字段 |
| `response.py` | 修改 `_build_restricted_response_operator_data` | 末尾预计算合并后的 ERI action 张量 |
| `response.py` | 修改 `_restricted_a_action` | 用预合并张量替代分步 einsum |
| `response.py` | 修改 `_restricted_b_action` | 同上 |
| `tda.py` | 无需修改 | 当前阈值和 fallback 逻辑已合理 |
| `casida.py` | 无需修改 | Davidson 路径已正确实现 |
| `eigensolvers.py` | 无需修改 | 已是纯 JAX `lax.fori_loop` |

---

## 4. 验证方式

### 4.1 数值正确性

```python
# 改动前后对同一体系比较 TDA/Casida 激发能和振幅
td_before = TDA(mf).kernel(nstates=5)
# ... 修改后 ...
td_after = TDA(mf).kernel(nstates=5)
assert jnp.allclose(td_before.e, td_after.e, atol=1e-8)
assert jnp.allclose(td_before.xy, td_after.xy, atol=1e-8)
```

### 4.2 性能测试

- 使用 `tools/benchmark_restricted_tddft_pyscf_vs_jax_devices.py` 做 before/after 计时对比
- 测试分子：water (STO-3G, 6-31G*)、benzene (6-31G*)、H₂ (def2-SVP)
- 分别测 TDA dense、TDA Davidson、Casida dense、Casida Davidson 四条路径

### 4.3 集成测试

- `pytest tests/test_pyscf_style_excited_state_api.py` 全部通过
- `pytest tests/test_excited_state_trainer.py` 全部通过

---

## 5. 预期收益

| 改动项 | 场景 | 预期加速 |
|--------|------|----------|
| rep_tensor MO 切片 JIT 融合 | 无 DF 的大分子 TDA/Casida 矩阵构建 | 1.5-2x（消除 3 次独立 kernel launch） |
| Davidson A/B action einsum 合并 | Davidson TDA 迭代（大体系） | 1.2-1.5x（每轮减少 1-2 次 einsum） |
| 其他 | - | 架构已良好，改动边际收益有限 |

---

## 6. Implementation Specification

### 6.1 P0: rep_tensor → MO ERI transformation JIT fusion

**File: `src/td_graddft/tddft/response.py`**

**Add function** (after `_restricted_eri_slices`, line 657):

```python
@jax.jit
def _rep_tensor_to_mo_eri_slices(
    rep_tensor: Array,    # (nao, nao, nao, nao)
    orbo: Array,          # (nao, nocc)
    orbv: Array,          # (nao, nvir)
    *,
    include_oovv: bool = True,
) -> tuple[Array, Array, Array | None]:
    """Single XLA launch: full 4-index AO ERI → MO-basis ov slices.
    
    Returns: (eri_ovov, eri_ovvo, eri_oovv | None)
      eri_ovov: (nocc, nvir, nocc, nvir) for TDA A matrix
      eri_ovvo: (nocc, nvir, nvir, nocc) for TDA B matrix
      eri_oovv: (nocc, nocc, nvir, nvir) for exchange terms
    """
    prec = jax.lax.Precision.HIGHEST
    
    eri_ovov = jnp.einsum("pqrs,pi,qa,rj,sb->iajb", rep_tensor, orbo, orbv, orbo, orbv, precision=prec)
    eri_ovvo = jnp.einsum("pqrs,pi,qa,rb,sj->iabj", rep_tensor, orbo, orbv, orbv, orbo, precision=prec)
    
    if include_oovv:
        eri_oovv = jnp.einsum("pqrs,pi,qj,ra,sb->ijab", rep_tensor, orbo, orbo, orbv, orbv, precision=prec)
    else:
        eri_oovv = None
        
    return eri_ovov, eri_ovvo, eri_oovv
```

**Modify `_restricted_eri_slices`** (line 612-649): replace the three separate einsum calls with single call to `_rep_tensor_to_mo_eri_slices`.

### 6.2 P1: Davidson A/B action einsum pre-merge

**File: `src/td_graddft/tddft/response.py`**

**Modify `_RestrictedResponseOperatorData`** (line 25): add field
```python
effective_tda_eri: Array | None = None  # 2*eri_ovov - α*transpose(eri_oovv)
```

**Modify `_build_restricted_response_operator_data`** (line 905): after building ERI slices, compute:
```python
if eri_oovv is not None:
    data.effective_tda_eri = 2.0 * eri_ovov - hybrid_fraction * jnp.transpose(eri_oovv, (0, 2, 1, 3))
else:
    data.effective_tda_eri = 2.0 * eri_ovov
```

**Modify `_restricted_a_action`** (line 977): replace two einsum calls with one:
```python
# Before (2 einsums):
# out += 2.0 * einsum("iajb,njb->nia", eri_ovov, x)
# out -= alpha * einsum("ijab,njb->nia", eri_oovv, x)
# After (1 einsum):
out += einsum("iajb,njb->nia", data.effective_tda_eri, x)
```

### 6.3 Implementation order

1. Add `_rep_tensor_to_mo_eri_slices` with `@jax.jit`
2. Modify `_restricted_eri_slices` rep_tensor branch to call it
3. Add `effective_tda_eri` field to `_RestrictedResponseOperatorData`
4. Compute `effective_tda_eri` in `_build_restricted_response_operator_data`
5. Simplify `_restricted_a_action` and `_restricted_b_action`
6. Run `test_pyscf_style_excited_state_api.py` to verify correctness
7. Benchmark with `benchmark_restricted_tddft_*.py`

**总体评估：** TDDFT 计算部分的 GPU 架构已较为成熟。DF 路径已是最优方案（两次 GEMM 完成 Hartree），Davidson 已用 `lax.fori_loop` 融合。主要收益来自对边缘路径（rep_tensor 回退、training 中重复调用）的修整，预期整体加速 1.2-2x。
