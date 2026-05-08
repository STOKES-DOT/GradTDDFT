# NN-RSH 泛函：架构升级与训练方案

**日期:** 2026-04-28
**范围:** `src/td_graddft/nn_rsh/`、`src/td_graddft/training/rsh_optimizer.py`

> **Implementation status (2026-04-29):** GNN head 已实现。`nn_rsh/gnn.py` (183 行) 包含 `RSHGNNHead`、`DistanceGatedAttention`、`AttentionReadout`。`functional.py` 已支持 `head_type="gnn"`。`make_gnn_rsh_functional` 构造函数已就绪。训练阶段的 Janak/Koopmans 多阶段 schedule 未实现。

---

## 1. 当前架构回顾

### 1.1 泛函形式（Type B）

基于可训练范围分离杂化泛函，NN 从分子描述符预测 $(\alpha_{\text{SR}}, \alpha_{\text{LR}}, \omega)$：

$$F = h + J + (1-\alpha_{\text{SR}})v_x^{\text{DFA}} + v_c^{\text{DFA}} + \alpha_{\text{SR}}K_{\text{full}} + (\alpha_{\text{LR}}-\alpha_{\text{SR}})K_{\text{LR}}(\omega)$$

参数通过 sigmoid-to-interval 映射到物理范围：

$$\alpha_{\text{SR}} = \text{sigmoid\_to\_interval}(z_1, [\alpha_{\text{SR}}^{\min}, \alpha_{\text{SR}}^{\max}])$$
$$\alpha_{\text{LR}} = \alpha_{\text{SR}} + (1-\alpha_{\text{SR}}) \cdot \sigma(z_2) \quad \text{(monotonic 模式)}$$
$$\omega = \text{sigmoid\_to\_interval}(z_3, [\omega^{\min}, \omega^{\max}])$$

### 1.2 前向链路

```
atom_descriptors (N_atoms × d_features)
    │
    ▼
Per-atom MLP [32, 32] → (N_atoms, 32)
    │
    ▼
Mean pooling → (32,)
    │
    ▼
Global MLP [32] → Dense(3) → (α_SR, α_LR, ω)
    │
    ▼
ResolvedRSHParameters → BoundTrainableRSHFunctional → SCF
```

### 1.3 训练方式：理论自洽的无监督学习

**核心立场：** 不依赖任何参考方法（FCI、CCSD(T)、实验值）提供能量标签。损失信号完全来自量子力学基本定理——这些定理对任何分子、任何电子态都成立，不局限于训练集覆盖的化学空间。

$$\mathcal{L} = w_{\text{janak}} \cdot \mathcal{L}_{\text{janak}} + w_{\text{koopmans}} \cdot \mathcal{L}_{\text{koopmans}} + w_{\text{frac}} \cdot \mathcal{L}_{\text{frac}} + w_{\text{prior}} \cdot \mathcal{L}_{\text{prior}}$$

**每个约束的理论来源：**

| 约束 | 量子力学定理 | 成立条件 |
|------|-------------|---------|
| $\varepsilon_{\text{HOMO}} = \partial E / \partial f_{\text{HOMO}}$ | Janak 定理 | 精确 XC 泛函 |
| $\varepsilon_{\text{HOMO}} = -IP$ | Koopmans 定理（KS 版本）| 精确 XC 泛函 |
| $E(N+\delta)$ 分段线性 | 密度泛函理论的精确条件 | 精确 XC 泛函 |

**关键推论：** 泛函越接近精确，这些残差越趋于零。因此 Loss → 0 的过程即泛函逼近精确泛函的过程。无需任何外部参考值——物理定理本身就是 ground truth。

**与监督路线的根本分歧：**

| | 能量监督（aPBE0, LH24n） | 物理自监督（TD-GradDFT nn_rsh） |
|---|---|---|
| 训练目标来源 | CCSD(T) / 实验 | 量子力学定理 |
| 泛化保证 | 在训练集覆盖的化学空间内有效 | 在任何满足定理前提的体系上有效 |
| 对参考方法的依赖 | 完全依赖——参考方法的系统误差进入泛函 | 零依赖 |
| 跨化学空间迁移 | 需要不确定性感知的 fallback | 定理普适，自然迁移 |
| 离线数据需求 | 需要预先计算大量参考值 | 无需任何离线数据 |

这就是为什么描述符必须密度感知：**密度是定理作用的对象**。描述符从密度构建，密度来自泛函自身的 SCF，形成闭环——没有外部参考方法介入。

其中：
- Janak 约束：$\varepsilon_{\text{HOMO}} \approx \partial E / \partial f_{\text{HOMO}}$（有限差分验证）
- Koopmans IP/EA：$\varepsilon_{\text{HOMO}} + E_{\text{cation}} - E_{\text{neutral}} \approx 0$
- 分数电荷线性度：$E(N+\delta) \approx (1-\delta)E(N) + \delta E(N+1)$
- 先验惩罚：参数偏离模板默认值的 L2 正则化

### 1.4 当前架构的瓶颈

- **Per-atom MLP 无原子间交互：** 每个原子的描述符被独立处理，模型无法感知 donor 和 acceptor 原子之间的关系
- **Mean pooling 丢失异质性：** donor 和 acceptor 描述符的反差信号被平均化抹平
- **无距离信息：** CT 激发中 donor-acceptor 距离是 $\omega$ 选择的关键物理量，当前架构无法利用
- **Global MLP 容量有限：** 从 32 维 pooled 向量到 3 个参数，信息压缩过于剧烈

---

## 2. 文献定位

### 2.1 全局标量预测路线的全景

| 工作 | 年份 | 期刊 | 预测目标 | ML 方法 | 描述符 | 数据集 |
|------|------|------|---------|---------|--------|--------|
| ML-ωPBE (Ju/Lin) | 2021 | *JPCL* | $\omega$ | Stacked ensemble | 半经验 QM | ~4,000 |
| ωGDDML (Villot/Lao) | 2023 | *JCP* | $\omega$ | XGBoost | 笛卡尔坐标 | 11,466 |
| ML-ωPBE 自由基 (Ju/Lin) | 2024 | *JPCA* | $\omega$ | Stacked ensemble | 半经验 QM | ~4,000 |
| aPBE0 (Khan/von Lilienfeld) | 2025 | *Sci. Adv.* | $a_{\text{opt}}$ | KRR | cMBDF（几何）| ~1,169 |
| ML-LC-PBE0\* (Liu/Cui) | 2025 | *JPCL* | $\omega$ (α 扫描) | SVM | 62K 维多指纹 | 4,380 |

**关键发现：所有已发表工作只预测单个参数（$\omega$ 或 $a$），无人同时预测 $\alpha_{\text{SR}}$、$\alpha_{\text{LR}}$、$\omega$ 三个参数。** 三参数 per-molecule 预测在文献中是空白。

### 2.2 与本文方案的关系

每条路线都依赖外部参考（CCSD(T)、实验、传统 OT 计算）。**本文方案是唯一不依赖任何外部参考值的三参数泛函预测方法。**

| | 已有工作 | TD-GradDFT nn_rsh |
|---|---|---|
| 预测目标 | 单参数（$\omega$ 或 $a$）| 三参数（$\alpha_{\text{SR}}$, $\alpha_{\text{LR}}$, $\omega$）|
| 训练信号 | **外部参考**：CCSD(T) 原子化能 / 传统 OT 结果 | **内部物理**：Koopmans / Janak / 分数电荷定理 |
| 描述符 | 几何 / 半经验 QM / cMBDF | Atom-centered density power spectrum（密度感知，自洽闭环） |
| 架构 | Ensemble / KRR / SVM / XGBoost | GNN + Global MLP |
| 泛化机制 | 描述符空间插值 + 不确定性 fallback | 物理定理的普适性 |
| 离线计算需求 | 需要预先计算大量参考值 | **零** |

---

## 3. 升级后的 GNN 架构

### 3.1 总架构

```
atom_descriptors (N_atoms × d_features)          atom_coords (N_atoms × 3)
    │                                                    │
    ▼                                                    ▼
Node Encoder MLP [d_features → h → h]              R_ij = ||r_i - r_j||
    │                                                    │
    ▼                                                    ▼
┌──────────────────────────────────────────────────────────────┐
│ Cross-Atom Interaction Block × L (推荐 L=1 或 2)             │
│                                                              │
│  Q, K, V = Linear_Q(node), Linear_K(node), Linear_V(node)    │
│                                                              │
│  A_raw_ij = Q_i · K_j^T / √d                                 │
│  gate_ij = exp(-R_ij / λ)    ← 可学习衰减长度 λ              │
│  A_ij = softmax_j(A_raw_ij ⊙ gate_ij)                        │
│                                                              │
│  msg_i = Σ_j A_ij · V_j                                       │
│  node_i' = node_i + msg_i          ← 残差连接                │
│  node_i' = LayerNorm(node_i')                                 │
│  node_i' = node_i' + FFN(node_i')   ← FFN: Dense → GELU →   │
│  node_i' = LayerNorm(node_i')               Dense            │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
Attention-Weighted Readout
    │
    q_readout = learnable query vector (h,)
    α_i = softmax(q_readout^T · W_readout · node_i)
    pooled = Σ_i α_i · node_i
    │
    ▼
Global MLP [h → h/2 → h/4] → Dense(3) → (α_SR_raw, α_LR_raw, ω_raw)
    │
    ▼
sigmoid_to_interval → (α_SR, α_LR, ω) → SCF
```

### 3.2 关键设计选择

**全连接图：** CT 体系的 donor 和 acceptor 可相隔 10-20Å，边不加空间截断。

**Distance-Gated Attention：** 距离作为门控而非 bias。$\lambda$ 初始化为 ~5Å，可学习。
- $\lambda$ 小 → 模型偏好局域交互（LE 场景）
- $\lambda$ 大 → 模型允许远距离交互（CT 场景）
- 训练中 $\lambda$ 自适应调整，无需人工指定

**L=1 层：** 5-50 原子的全连接图上，单层 attention 足够让每个原子感知全局信息。更多层有过度平滑风险。

**Attention-Weighted Readout：** 替代 mean pooling。模型自动学习哪些原子对 RSH 参数重要——CT 体系中 donor/acceptor 区域的原子应获得更高权重。

**残差 + LayerNorm：** 保证梯度流畅，训练稳定。

### 3.3 参数量

| 组件 | 参数 |
|------|------|
| Node Encoder | d_features × h + h × h ≈ 5k-20k |
| Multi-Head Attention (4 heads) | 4·h² ≈ 4k-16k |
| FFN | 2·h·h_ffn ≈ 8k-32k |
| Attention Readout | h × h ≈ 1k-4k |
| Global MLP | h × h/2 + h/2 × h/4 + h/4 × 3 ≈ 3k |
| **总计** | **~20k-75k** |

与当前 `AtomwiseRSHParameterHead`（无隐藏层时仅 bias 3 个参数，有 [32,32] 时约 5k）相比参数量增加 4-15 倍，但仍然极小——传统 GNN 通常有 100k-1M 参数。

### 3.4 描述符的自洽闭环

Atom-centered density power spectrum 编码的是每个原子周围的局域密度形状，从自洽密度 $\rho(\mathbf{r})$ 投影得到。这构成了一个**无外部参考介入的自洽闭环**：

```
泛函参数 (α_SR, α_LR, ω)
        │
        ▼
      SCF → ρ(r)
        │
        ▼
  Atom-centered density descriptor
        │
        ▼
      GNN → (α_SR, α_LR, ω)
        │
        └──────────────┘  ← 闭环，无外部参考
```

这与所有已有工作形成鲜明对照：

| 方法 | 描述符来源 | 是否自洽 |
|------|-----------|---------|
| aPBE0 (cMBDF) | 分子几何 | 否——描述符在 SCF 之前计算，不改 |
| ML-ωPBE | 半经验 QM | 否——外部方法提供，与泛函无关 |
| ωGDDML | 笛卡尔坐标 | 否——纯几何，完全独立 |
| **本文** | **自洽密度** | **是——泛函参数 → 密度 → 描述符 → 泛函参数** |

这个闭环的意义：当 GNN 给出更好的 $(\alpha, \omega)$ 时，SCF 给出更好的密度，描述符随之更新，GNN 在新的描述符上做更精准的预测——这是所有离线描述符无法实现的迭代优化。

用纯几何描述符做不到这一点，因为描述符在 SCF 前后不变。

---

## 4. 训练方案

### 4.1 损失函数

保留无能量监督的核心优势：

$$\mathcal{L}_{\text{total}} = w_J \cdot \mathcal{L}_{\text{janak}} + w_K \cdot \mathcal{L}_{\text{koopmans}} + w_F \cdot \mathcal{L}_{\text{frac}} + w_P \cdot \mathcal{L}_{\text{prior}}$$

推荐默认权重配置：

| 约束 | 权重 | 理由 |
|------|------|------|
| $\mathcal{L}_{\text{janak}}$ | 1.0 | 核心约束，强制 $\varepsilon_{\text{HOMO}} \approx \partial E / \partial f$ |
| $\mathcal{L}_{\text{koopmans}}$ | 0.3 | IP/EA 约束，需 3 次 SCF/step，与 Janak 部分冗余 |
| $\mathcal{L}_{\text{frac}}$ | 0.5 | 分数电荷线性度，消除离域误差的关键 |
| $\mathcal{L}_{\text{prior}}$ | 1e-3 | 防参数漂移，保持与模板泛函的合理距离 |

### 4.2 训练阶段

**阶段 1：单一分子 overfit 验证**

在 H₂O 或 benzene 上以 `loss="janak"` 训练 200 steps。验证：
- $\omega$ 收敛到合理范围（LC-$\omega$PBE 默认 0.3 附近）
- SCF 收敛稳定（converged=true in ≥90% steps）
- 参数在训练过程中平滑变化而非剧烈震荡

**阶段 2：多分子泛化训练**

在 5-20 个不同分子（包含不同共轭长度、donor-acceptor 距离）上训练。验证不同分子得到不同参数——确保 GNN 确实学到了分子特异性。

**阶段 3：激发态验证**

对训练好的泛函做 TDDFT 计算，与 OT-RSH 或 CCSD 参考比较激发能。

### 4.3 训练配置

```python
config = GroundStateTrainingConfig(
    mode="self_consistent",       # 固定自洽模式
    scf_max_cycle=12,             # 避免过长 SCF
    scf_damping=0.3,              # 训练初期需要 damping
    scf_level_shift=0.5,          # 分数占据时尤其重要
    scf_require_convergence=False, # 允许部分未收敛
    scf_gradient_mode="unrolled",  # 或 implicit_commutator（更快）
    janak_frontier_weight=1.0,
    fractional_linearity_weight=0.5,
    koopmans_ip_weight=0.3,
    koopmans_detach_charged_states=True,  # 阶段 1 用 detach 保稳定
    prior_weight=1e-3,
)
```

### 4.4 训练稳定性策略

- **阶段 1 用 `koopmans_detach_charged_states=True`：** 带电态 SCF 做 stop_gradient，仅中性态轨道能量驱动梯度。稳定性优先
- **阶段 2 可尝试 `koopmans_detach_charged_states=False`：** 完整梯度流，但需要更小的 learning rate (1e-4) 和更多的 damping
- **梯度裁剪 + nan 保护：** `trainer.py` 已有 `_sanitize_gradients`，训练前确认开启
- **参数变化监控：** 每 20 step 记录 $(\alpha_{\text{SR}}, \alpha_{\text{LR}}, \omega)$ 的变化曲线，异常时调整 learning rate

---

## 5. 实施清单

### 5.1 新增模块

| 文件 | 内容 |
|------|------|
| `nn_rsh/gnn.py` | `RSHGNNHead`：NodeEncoder + DistanceGatedAttention + AttentionReadout + GlobalMLP |
| `nn_rsh/gnn.py` | `DistanceGatedAttention`：QKV + softmax(dot + distance_gate) |

### 5.2 修改模块

| 文件 | 改动 |
|------|------|
| `nn_rsh/functional.py` | `TrainableRSHFunctional` 支持 GNN head（新增 `head_type="gnn"` 参数） |
| `nn_rsh/descriptors.py` | 描述符输出增加 `atom_coords` 字段（供 attention 距离计算） |
| `training/rsh_optimizer.py` | `RSHOptimizer.kernel` 支持 `koopmans_detach_charged_states` 参数透传 |
| `nn_rsh/__init__.py` | 导出 `RSHGNNHead` |

### 5.3 验证脚本

- 在 `tests/` 中新增 `test_nn_rsh_gnn_overfit.py`：单分子 overfit 测试
- 使用 `tools/overfit_water_nn_rsh.py` 做 before/after（GNN vs 当前 MLP）性能对比

---

## 6. 预期效果

| 指标 | 当前 MLP (constant desc) | 当前 MLP (atom desc) | 升级 GNN |
|------|--------------------------|---------------------|----------|
| 模型容量 | ~3 params (bias only) | ~5k params | ~20k-75k params |
| 原子间交互 | 无 | 无（per-atom MLP + mean pool）| 有（attention + dist gate） |
| 每分子特异性 | 无（退化到全局常数）| 部分（per-atom 模式不同）| 充分（atom + pairwise） |
| CT 体系泛化 | 差 | 中等 | 预期好 |
| 训练稳定性 | 稳定（参数少）| 稳定 | 需验证（残差 + LayerNorm 应确保稳定）|

---

## 7. Implementation Specification

### 7.1 File to create: `src/td_graddft/nn_rsh/gnn.py`

**Full Flax module:**

```python
from typing import Sequence
import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array


class DistanceGatedAttention(nn.Module):
    """Multi-head self-attention with learnable distance gate.

    Attention: A_ij = softmax_j(Q_i·K_j^T / √d_head ⊙ exp(-R_ij / λ))
    """
    num_heads: int = 4
    qkv_features: int | None = None  # defaults to input features
    lambda_init: float = 5.0  # Angstrom, initial decay length
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self,
        node_features: Array,   # (batch, N_atoms, d_model)
        atom_coords: Array,     # (batch, N_atoms, 3)
        *,
        deterministic: bool = True,
    ) -> Array:
        d_model = node_features.shape[-1]
        d_head = d_model // self.num_heads
        qkv_features = self.qkv_features or d_model

        # QKV projections
        q = nn.Dense(qkv_features, name="q_proj")(node_features)
        k = nn.Dense(qkv_features, name="k_proj")(node_features)
        v = nn.Dense(qkv_features, name="v_proj")(node_features)

        # Reshape for multi-head
        # (batch, N, heads, d_head)
        q = q.reshape(*q.shape[:-1], self.num_heads, d_head)
        k = k.reshape(*k.shape[:-1], self.num_heads, d_head)
        v = v.reshape(*v.shape[:-1], self.num_heads, d_head)

        # Scaled dot-product attention logits: (batch, heads, N, N)
        attn_logits = jnp.einsum("bihd,bjhd->bhij", q, k) / jnp.sqrt(d_head)

        # Distance gate: R_ij = ||r_i - r_j||
        r_diff = atom_coords[:, :, None, :] - atom_coords[:, None, :, :]
        r_ij = jnp.sqrt(jnp.sum(r_diff ** 2, axis=-1) + 1e-8)  # (batch, N, N)

        # Learnable decay parameter λ
        log_lambda = self.param(
            "log_lambda",
            nn.initializers.constant(jnp.log(self.lambda_init)),
            (),
        )
        lam = jnp.exp(log_lambda)
        gate = jnp.exp(-r_ij / lam)  # (batch, N, N)

        # Apply gate and softmax
        attn_logits = attn_logits + jnp.log(gate[:, None, :, :] + 1e-8)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)

        if self.dropout_rate > 0.0:
            attn_weights = nn.Dropout(self.dropout_rate, deterministic=deterministic)(
                attn_weights
            )

        # Weighted sum: (batch, heads, N, d_head)
        out = jnp.einsum("bhij,bjhd->bihd", attn_weights, v)

        # Concatenate heads: (batch, N, d_model)
        out = out.reshape(*out.shape[:-2], d_model)
        return nn.Dense(d_model, name="out_proj")(out)


class AttentionReadout(nn.Module):
    """Learnable query vector for attention-weighted pooling over atoms."""
    d_model: int

    @nn.compact
    def __call__(self, node_features: Array) -> Array:
        # node_features: (batch, N_atoms, d_model)
        query = self.param(
            "readout_query",
            nn.initializers.normal(stddev=0.02),
            (1, self.d_model),
        )
        key = nn.Dense(self.d_model, name="readout_key")(node_features)
        # (batch, 1, d_model) @ (batch, d_model, N) -> (batch, 1, N)
        attn = jnp.einsum("bqd,bnd->bqn", query[None, :, :], key)
        attn = jax.nn.softmax(attn / jnp.sqrt(self.d_model), axis=-1)
        # (batch, 1, N) @ (batch, N, d_model) -> (batch, d_model)
        return jnp.einsum("bqn,bnd->bd", attn, node_features)


class RSHGNNHead(nn.Module):
    """GNN head predicting RSH parameters from atom-centered density descriptors.

    Architecture:
      NodeEncoder (MLP) → DistanceGatedAttention → LayerNorm → FFN → LayerNorm
      → AttentionReadout → GlobalMLP → Dense(3)

    Output: raw (α_SR, α_LR, ω) in logit space
    """
    node_hidden_dims: Sequence[int] = (32, 32)
    global_hidden_dims: Sequence[int] = (32, 16)
    num_heads: int = 4
    num_layers: int = 1  # number of attention + FFN blocks
    ffn_expansion: int = 4
    dropout_rate: float = 0.0
    activation: callable = nn.gelu

    @nn.compact
    def __call__(
        self,
        atom_descriptors: Array,  # (batch, N_atoms, d_features)
        atom_coords: Array,       # (batch, N_atoms, 3)
        *,
        deterministic: bool = True,
    ) -> Array:  # (batch, 3)
        d_model = self.node_hidden_dims[-1]

        # Node encoder
        x = atom_descriptors
        for i, width in enumerate(self.node_hidden_dims):
            x = nn.Dense(width, name=f"node_encoder_{i}")(x)
            x = self.activation(x)

        # Cross-atom interaction blocks
        for layer_idx in range(self.num_layers):
            residual = x
            x = DistanceGatedAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                name=f"attn_{layer_idx}",
            )(x, atom_coords, deterministic=deterministic)
            x = x + residual
            x = nn.LayerNorm(name=f"ln1_{layer_idx}")(x)

            ffn_residual = x
            x = nn.Dense(d_model * self.ffn_expansion, name=f"ffn1_{layer_idx}")(x)
            x = self.activation(x)
            x = nn.Dense(d_model, name=f"ffn2_{layer_idx}")(x)
            x = x + ffn_residual
            x = nn.LayerNorm(name=f"ln2_{layer_idx}")(x)

        # Attention-weighted readout
        pooled = AttentionReadout(d_model, name="readout")(x)

        # Global MLP
        y = pooled
        for i, width in enumerate(self.global_hidden_dims):
            y = nn.Dense(width, name=f"global_mlp_{i}")(y)
            y = self.activation(y)

        # Output head: 3 raw parameters
        return nn.Dense(3, name="output")(y)
```

### 7.2 File to modify: `src/td_graddft/nn_rsh/descriptors.py`

Add `atom_coords` to descriptor output:

```python
# In make_atom_centered_density_descriptor_fn:
def descriptor_fn(molecule: Any | None) -> dict[str, Array]:
    ...
    return {
        "atom_descriptors": atom_centered_density_power_spectrum(molecule, config=cfg),
        "atom_charges": jnp.asarray(molecule.atom_charges, dtype=jnp.int32),
        "atom_coords": jnp.asarray(molecule.atom_coords, dtype=jnp.float32),  # NEW
    }
```

### 7.3 File to modify: `src/td_graddft/nn_rsh/functional.py`

**7.3.1 `TrainableRSHFunctional` — add `head_type` parameter:**

```python
@dataclass(frozen=True)
class TrainableRSHFunctional:
    model: nn.Module
    template: RSHFunctionalTemplate
    local_xc_spec: str = "pbe"
    descriptor_fn: Callable = _constant_rsh_descriptor
    head_type: str = "mlp"  # "mlp" | "gnn"  ← NEW
    ...
```

**7.3.2 Modify `init_from_molecule` and `_raw_outputs`:**

When `head_type == "gnn"`, pass `atom_coords` as additional input:

```python
def _raw_outputs(self, params, molecule=None):
    descriptor = self.descriptor_fn(molecule)
    if self.head_type == "gnn":
        return self.model.apply(
            params,
            descriptor["atom_descriptors"],
            descriptor["atom_coords"],
        )
    else:
        return self.model.apply(params, descriptor)
```

### 7.4 File to modify: `src/td_graddft/nn_rsh/__init__.py`

```python
from .gnn import RSHGNNHead, DistanceGatedAttention, AttentionReadout
```

### 7.5 Construction helper (in `functional.py` or new `gnn.py`)

```python
def make_gnn_rsh_functional(
    template_name: str = "lc-wpbe",
    local_xc_spec: str = "pbe",
    node_hidden_dims: tuple[int, ...] = (32, 32),
    global_hidden_dims: tuple[int, ...] = (32, 16),
    num_heads: int = 4,
    num_layers: int = 1,
    density_floor: float = 1e-12,
    potential_clip: float = 20.0,
) -> TrainableRSHFunctional:
    template = make_rsh_template(template_name)
    descriptor_fn = make_atom_centered_density_descriptor_fn()
    model = RSHGNNHead(
        node_hidden_dims=node_hidden_dims,
        global_hidden_dims=global_hidden_dims,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    return TrainableRSHFunctional(
        model=model,
        template=template,
        local_xc_spec=local_xc_spec,
        descriptor_fn=descriptor_fn,
        head_type="gnn",
        density_floor=density_floor,
        potential_clip=potential_clip,
    )
```

### 7.6 Implementation order

1. Create `gnn.py` with `DistanceGatedAttention`, `AttentionReadout`, `RSHGNNHead`
2. Modify `descriptors.py` to include `atom_coords` in output dict
3. Modify `functional.py` `TrainableRSHFunctional` to support `head_type`
4. Add `make_gnn_rsh_functional` constructor
5. Update `__init__.py` exports
6. Write single-molecule overfit test
7. Train on H₂O, verify convergence and parameter stability

### 7.7 Test specification

```python
def test_gnn_head_forward():
    """RSHGNNHead runs forward without error."""
    key = jax.random.PRNGKey(0)
    head = RSHGNNHead()
    dummy_descriptors = jnp.ones((1, 5, 10))  # batch=1, 5 atoms, 10 features
    dummy_coords = jnp.zeros((1, 5, 3))
    params = head.init(key, dummy_descriptors, dummy_coords)
    out = head.apply(params, dummy_descriptors, dummy_coords)
    assert out.shape == (1, 3)

def test_gnn_permutation_equivariance():
    """Permuting atoms permutes output labels but preserves global prediction."""
    ...

def test_gnn_vs_mlp_single_molecule():
    """On a single molecule, both heads should converge to similar params."""
    ...
```
