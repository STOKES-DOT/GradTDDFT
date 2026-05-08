# 神经网络泛函形式与优化分析

**日期:** 2026-04-28
**范围:** `src/td_graddft/neural_xc/dm21/functional.py`、`src/td_graddft/training/`

> **Implementation status (2026-04-29):** 未实现。semilocal 系数 softmax 统一、分数自旋约束、多阶段 training schedule 均为计划状态。

---

## 1. 保留的两种泛函形式

### Mode 3: `graddft_coeff_basis_hf_pt2_heads`

**基组通道:**

$$\{e_k\} = \{e_1^{\text{semilocal}}, \ldots, e_n^{\text{semilocal}}, e_{\text{PT2}}, e_{\text{HF}}\}$$

其中 $n = |\text{semilocal\_xc}|$（如 B3LYP 基组：$n=4$），PT2 通道可选。

**MLP 输出维度:** $K = n + 1(\text{HF}) + 1(\text{PT2, optional})$

**系数变换（per-channel activation）:**

$$c_k = \text{clip}(c_k^{\text{raw}}, 0, \lambda), \quad k \in [1, n] \qquad \text{(semilocal)}$$
$$c_{\text{PT2}} = \sigma(c_{\text{PT2}}^{\text{raw}}) \in [0, 1]$$
$$c_{\text{HF}} = \sigma(c_{\text{HF}}^{\text{raw}}) \in [0, 1]$$

其中 $\lambda$ = `kernel_clip`（默认 5.0），$\sigma(x) = 1/(1+e^{-x})$。

**能量密度:**

$$\boxed{e_{xc}(\mathbf{r}) = \sum_{k=1}^{n} c_k(\mathbf{r}) \cdot e_k^{\text{semilocal}}(\mathbf{r}) + \sigma(c_{\text{PT2}}) \cdot e_{\text{PT2}}(\mathbf{r}) + \sigma(c_{\text{HF}}) \cdot e_{\text{HF}}(\mathbf{r})}$$

**关键性质:**
- Semilocal 通道系数可以独立大于 1（clip 上限 $\lambda=5$）
- HF 和 PT2 系数始终在 $[0,1]$ 内
- 所有通道之间无约束关系——semilocal 可以很大而 HF 可以很小，总能量密度无上界

---

### Mode 4: `dldh_two_lmf`

**基组通道（固定 4 通道）:**

$$\{e_k\} = \{e_x^{\text{DFA}}, e_c^{\text{DFA}}, e_{\text{PT2}}, e_{\text{HF}}\}$$

其中 $e_x^{\text{DFA}} = \sum_{k \in \text{exchange}} e_k^{\text{semilocal}}$，$e_c^{\text{DFA}} = \sum_{k \in \text{correlation}} e_k^{\text{semilocal}}$，通过 `exchange_mask` 分离。

**MLP 输出维度:** $G = 1(\text{HF mixing } f_x) + 1(\text{PT2 mixing } f_{\text{pt2}}, \text{optional})$

**系数变换（pairwise complementarity）:**

$$f_x = \sigma(c_1^{\text{raw}}) \in [0, 1]$$
$$f_{\text{pt2}} = \sigma(c_2^{\text{raw}}) \in [0, 1] \quad \text{or} \quad f_{\text{pt2}} = 0$$

**能量密度（基础形式，无 RS/QAC）:**

$$\boxed{e_{xc}(\mathbf{r}) = \underbrace{(1-f_x) e_x^{\text{DFA}} + f_x e_{\text{HF}}}_{\text{交换分区：系数和为 1}} + \underbrace{(1-f_{\text{pt2}}) e_c^{\text{DFA}} + f_{\text{pt2}} e_{\text{PT2}}}_{\text{相关分区：系数和为 1}}}$$

**能量密度（含 RS + QAC 的完整形式）:**

当 `dldh_range_separated_exchange=True` 且 `dldh_qac_mode != "none"` 时，交换部分变为：

**Step 1 — Range-separated DFA exchange（Hirao 方案）:**

$$e_x^{\text{DFA,RS}}(\mathbf{r}) = e_x^{\text{DFA}}(\mathbf{r}) \cdot F_0(2a), \quad a = \frac{\omega}{2k_F\sqrt{e_x^{\text{LDA}}/e_x^{\text{DFA}}}}$$

其中 $F_0$ 是 DLDH LRS-LDA 衰减函数，$k_F = (6\pi^2\rho)^{1/3}$。

**Step 2 — Short-range HF anchor（SR-HF exchange at ω）:**

$$e_{\text{HF}}^{\text{SR}}(\mathbf{r}) = e_{\text{HF}}(\mathbf{r}) - e_{\text{HF}}^{\text{LR}}(\mathbf{r}; \omega)$$

通过 `hfx_nu` 在 $\omega$ 上的插值得到（格点投影 $\text{erfc}(\omega r_{12})/r_{12}$ 的贡献）。

**Step 3 — QAC（二次绝热连接）因子:**

$$q(\mathbf{r}) = \text{QAC}(z(\mathbf{r})), \quad z = \max\left(\frac{e_x^{\text{MB86x}}}{e_x^{\text{HF}}} - 1, 0\right)$$

其中 $e_x^{\text{MB86x}}$ 是 meta-GGA 交换描述符（依赖 $\rho, \nabla\rho, \tau, \nabla^2\rho$，使用 $\lambda, \beta$ 参数）。

**Pade 模式:** $q = 0.5 + d \cdot \frac{aa \cdot z^{bb}}{2(1 + aa \cdot z^{bb})}$，其中 $aa, bb$ 由 $(p_1, p_2)$ 确定

**Erf 模式:** $q = 0.5 + 0.5 \cdot \text{erf}(b \cdot z_{\text{erf}})$，其中 $z_{\text{erf}} = z \cdot \max(\text{erf}(12(z-a)), 0)$

**Step 4 — 最终交换贡献:**

$$e_x^{\text{DLDH,RS+QAC}} = 2q \cdot (1-f_x) \cdot (e_x^{\text{DFA,RS}} - e_{\text{HF}}^{\text{SR}}) + e_{\text{HF}}$$

注意：最后一项是完整的 $e_{\text{HF}}$（全 HF 交换），不是 $e_{\text{HF}}^{\text{SR}}$。

---

## 2. 两种形式的本质对比

| 维度 | Mode 3 | Mode 4 |
|------|--------|--------|
| 通道数 | $n+1+1$（可扩展） | 固定 4 |
| MLP 输出自由度 | $n+1+1$ 个独立系数 | 1-2 个混合分数 |
| 半局域通道 | 独立系数，无约束 | 预分为 x/c，受互补约束 |
| HF 系数范围 | $\sigma \in [0,1]$（独立） | $f_x \in [0,1]$，与 DFA-x 互补 |
| PT2 系数范围 | $\sigma \in [0,1]$（独立） | $f_{\text{pt2}} \in [0,1]$，与 DFA-c 互补 |
| 物理约束 | **无**（能量密度无上界） | **分区 unity**（$c_x^{\text{DFA}}+c_{\text{HF}}=1$） |
| Range separation | 不支持 | 支持（Hirao + QAC） |
| 额外物理参数 | 无 | $\omega, \lambda, \beta, p_1, p_2, d$（或 $a, b$） |

---

## 3. 优化模式分析

### 3.1 训练流程

```
Input: 分子几何 {R_A}
    │
    ▼
┌────────────────────────────────────────┐
│ 1. 格点构建 + 密度特征                  │
│    {r_g, w_g}, ρ(r_g), ∇ρ(r_g), τ(r_g)│
├────────────────────────────────────────┤
│ 2. 基组通道计算                         │
│    e_k(r_g) = semilocal + HF + PT2     │
├────────────────────────────────────────┤
│ 3. MLP 前向                             │
│    c(r_g) = MLP_θ(x(r_g))             │
├────────────────────────────────────────┤
│ 4. 能量积分                             │
│    E_xc = Σ_g w_g · Σ_k a_k · e_k     │
│    E_tot = E_one + E_H + E_nuc + E_xc │
├────────────────────────────────────────┤
│ 5. 损失 + 反向传播                       │
│    L = |E_tot - E_ref|²               │
│    ∂L/∂θ = ∂L/∂E · Σ_g ∂E/∂a · ∂a/∂θ │
└────────────────────────────────────────┘
```

### 3.2 两种优化模式

**Fixed-Density 模式:** 密度 $\rho$ 由参考方法（如 PBE/PBE0）预收敛，MLP 仅拟合能量。梯度：

$$\frac{\partial \mathcal{L}}{\partial \theta} = 2(E^{\text{pred}} - E^{\text{ref}}) \cdot \sum_g w_g \sum_k e_k(\mathbf{r}_g) \cdot \frac{\partial a_k(\mathbf{r}_g)}{\partial \theta}$$

**Self-Consistent 模式:** 泛函嵌入 SCF 循环，梯度通过隐函数定理传播。在 SCF 收敛点：

$$\frac{\partial E^*}{\partial \theta} = \left.\frac{\partial E}{\partial \theta}\right|_{\rho^*}$$

两种模式可以组合：fixed-density 做预训练（快速收敛），self-consistent 做精调（物理一致性）。

---

## 4. 可改进之处

### 4.1 Mode 3 的改进点

**问题 1: semilocal 系数无上界约束**

当前 `clip(raw, 0, 5.0)` 允许 semilocal 通道被放大到 5 倍。这可以拟合任意能量尺度但对泛函的转移性不利——同样的系数在 H₂O 和 benzene 上不应有完全不同的尺度。

**改进:** 对所有通道统一使用 softmax/sigmoid，通过 temperature 控制混合锐度：

$$a_k = \frac{\exp(c_k^{\text{raw}} / T)}{\sum_{j} \exp(c_j^{\text{raw}} / T)}$$

- $T=1$: 标准 softmax（Mode 2 行为）
- $T \to \infty$: 趋近均匀混合
- $T \to 0$: 趋近 one-hot

保留 HF/PT2 有独立 temperature 的自由度。

**问题 2: UEG 极限约束缺失**

均匀电子气下 $\rho = \text{const}, \nabla\rho = 0, \tau = \tau_{\text{UEG}}$，MLP 输入为常数，输出系数应为常数。但当前没有机制保证 UEG 极限下的系数接近物理值。

**改进:** 加入 UEG 正则化项：

$$\mathcal{L}_{\text{UEG}} = \left\| \mathbf{c}_{\text{MLP}}(\mathbf{x}_{\text{UEG}}) - \mathbf{c}_{\text{UEG}}^{\text{ref}} \right\|^2$$

其中 $\mathbf{c}_{\text{UEG}}^{\text{ref}}$ 可以通过分析得到（如 LDA 交换在 UEG 下的精确行为）。

**问题 3: 空间平滑性缺失**

MLP 逐格点独立计算系数，相邻格点的系数可能剧烈跳变。虽然能量积分对小幅跳变不敏感，但 $v_{xc}$ 势（涉及系数对密度的导数）可能产生噪声。

**改进:** 加系数梯度惩罚：

$$\mathcal{L}_{\text{smooth}} = \frac{1}{N_{\text{grid}}} \sum_g \|\nabla \mathbf{c}(\mathbf{r}_g)\|^2$$

实现上可通过 `jax.vjp` 对系数场关于格点坐标求导。

### 4.2 Mode 4 的改进点

**问题 4: QAC 参数不可训练**

`dldh_qac_parameters`（p1, p2, d 或 a, b）在构造时固定，不参与梯度更新。但 QAC 因子对泛函精度敏感——不同分子类型可能需要不同的绝热连接参数。

**改进:** 将 QAC 参数作为可训练变量，通过 `self.put_variable` 存储，与 MLP 权重一起优化。或通过另一个小型 MLP 从分子描述符预测 QAC 参数：

$$(p_1, p_2, d) = \text{MLP}_{\text{QAC}}(\text{global\_descriptor})$$

**问题 5: RS-ω 固定**

`dldh_range_separation_omega` 固定为默认值 0.233。这与 RSH 泛函中 ω 是核心可优化参数形成对比。DLDH 的交换部分对 ω 敏感——ω 决定了短程/长程 HF 的分界点。

**改进:** 将 ω 也作为 MLP 的额外输出头（类似 RSH 的 atomwise head）：

$$(\omega, f_x, f_{\text{pt2}}) = \text{MLP}_{\text{DLDH}}(\text{descriptor})$$

或者保持 f_x 为局域量，ω 为全局量（分子级描述符 → ω，格点级描述符 → f_x）。

**问题 6: 交换-相关分离依赖 exchange_mask**

`_split_semilocal_exchange_correlation_local_channels` 使用 `exchange_mask` 来区分哪些 libxc 组分是交换、哪些是相关。对自定义 `SemilocalEnergyDensityModule`，此 mask 可能不准确（如果用户混合了交换和相关通道）。

**改进:** 将 exchange_mask 作为 `SemilocalEnergyDensityModule` 的属性，由模块构造时显式声明，而非从 libxc 的 `xc_type` 推断。

### 4.3 两模式共有的改进点

**问题 7: HF 格点投影的冗余计算**

`projected_hf_grid_contribution_components` 在每次 functional binding 时都被调用。但 HF 格点投影（$e_{\text{HF}}(\mathbf{r}_g)$）在 SCF 的密度不变期间是常数——应该在 SCF 循环外部计算一次并缓存。

**当前状态:** `molecule.hfx_local` 已经做了预计算缓存，但 `hfx_nu` 路径需要在每次 binding 时从 `ao, nu, dm` 重新投影。对 RSH 训练（频繁 bind），这引入重复计算。

**改进:** SCF 循环中 HF 格点投影与 Fock 矩阵构建解耦——在每轮 SCF 开始时一次性计算 $e_{\text{HF}}(\mathbf{r}_g)$ 和 $e_{\text{HF}}^{\text{LR}}(\mathbf{r}_g; \omega)$，供后续 functional binding 复用。

**问题 8: 输入特征冗余**

`coefficient_inputs` 构建中，"enhanced" 模式的特征维度约 15-20，"dm21_original" 模式约 10。两者都包含 hand-crafted 的密度缩放特征（`dm21_like_input_features` 中的 $u(r_s), w(|\nabla\rho|/\rho^{4/3})$ 等）。这些手工特征可能不是最优的。

**改进:** 引入 learned feature embedding——在 MLP 的第一层前加一个可训练的 feature embedding 层，让网络学习最优的密度表示：

$$\mathbf{x}_{\text{learned}} = \text{Embed}(\rho, |\nabla\rho|, \tau, e_{\text{semilocal}}/\rho, e_{\text{HF}}/\rho, \ldots)$$

替代当前的手工缩放。

**问题 9: 格点数量对 GPU 吞吐的影响**

Mode 3 的 MLP 是逐格点应用的，GPU 上表现为 `vmap` over $N_{\text{grid}}$ 的矩阵乘法。对于精细格点（grid_level ≥ 3），$N_{\text{grid}}$ 可超过 10⁴，MLP 的 `Dense` 层 FLOPs 为 $O(N_{\text{grid}} \cdot d_{\text{in}} \cdot d_{\text{hidden}})$，其中间激活张量为 $N_{\text{grid}} \times d_{\text{hidden}}$，对显存和带宽构成压力。

**改进:** 使用低精度推理（bf16/fp16）进行 MLP 前向，仅在能量积分（`tensordot`）时还原为 fp32。JAX 的 `jax.lax.convert_element_type` 可实现 per-layer precision control。

---

## 5. 训练方法论

### 5.1 两阶段训练协议

当前实际采用的训练流程：

```
阶段 1: 基态自洽训练（大量 steps）
        mode = "self_consistent"
        ├── 能量损失: |E_tot^pred - E_tot^ref|
        ├── 密度约束:  ||ρ_pred - ρ_ref|| (可选)
        └── 分数电荷约束 + Janak 定理约束 (可选)
            │
            ▼  (SCF 收敛 → 轨道能量 + 密度)
            │
阶段 2: 激发态 fine-tune
        mode = "fixed_density"
        ├── 基态密度冻结（不再跑 SCF）
        ├── TDDFT 激发能: |Ω_pred - Ω_ref|
        ├── 振子强度: |f_pred - f_ref|
        └── 光谱曲线匹配 (可选)
```

**关键设计决策：** 阶段 2 切换到 fixed-density 模式后，基态 SCF 不再执行。这避免了激发态训练中的 SCF 收敛不稳定问题，但也意味着激发态梯度不会流回基态密度优化。

### 5.2 数据集

当前 `GroundStateDatum` 承载了基态 + 激发态的全部监督信号：

```
GroundStateDatum
├── molecule:             JAX 分子对象（ao, grid, mo_coeff, rep_tensor, hfx_nu, ...）
├── target_total_energy:  参考总能 (FCI/CCSD(T)/... )
├── target_density_matrix: 参考密度矩阵 (可选，用于密度约束)
│
├── 激发态监督:
│   ├── target_s1_energy / target_excitation_energies
│   ├── target_oscillator_strengths
│   └── target_spectrum_grid_ev + target_spectrum_curve
│
├── 约束权重 (均为 0.0 默认):
│   ├── density_constraint_weight
│   ├── janak_frontier_constraint_weight
│   ├── fractional_linearity_weight (配置级)
│   ├── s1_constraint_weight / excitation_constraint_weight
│   └── oscillator_strength_constraint_weight / spectrum_constraint_weight
│
└── weight: 样本权重 (默认 1.0)
```

**实际使用的典型数据集：**

| 体系 | 参考方法 | 用途 |
|------|----------|------|
| H₂ 解离曲线（FCI/aug-cc-pVDZ） | FCI | 基态 + 激发态训练，分数自旋约束验证 |
| QH9 子集（闭壳层单重态 S₁） | EOM-CCSD/def2-TZVP | 激发态 fine-tune |
| 水分子（PBE/def2-SVP） | PBE | overfit 测试 |
| 苯分子（B3LYP/6-31G*） | B3LYP | 激发态对比 benchmark |

### 5.3 约束实现方式

DM21 风格的核心约束在 `targets.py` 中已实现，通过 `GroundStateTrainingConfig` 的权重参数激活：

```python
# 全套约束训练配置（DM21 风格）
config = GroundStateTrainingConfig(
    mode="self_consistent",           # 阶段 1: 自洽
    energy_mae_weight=1.0,            # 能量 MAE
    density_constraint_weight=0.1,    # 密度约束 (需 datum 级设置)
    fractional_linearity_weight=0.5,  # 分数电荷线性度
    janak_frontier_weight=0.3,        # Janak 定理约束
    coefficient_prior_weight=1e-3,    # B3LYP 先验正则化
    scf_max_cycle=12,
    scf_damping=0.25,
    scf_require_convergence=False,    # 允许未收敛时 soft penalty
)
```

**Janak 定理约束** 有五种计算模式（`janak_frontier_mode`）：

| mode | 方法 | 精度 | 开销 |
|------|------|------|------|
| `finite_difference` | $\partial E/\partial f \approx (E(f+\delta)-E(f-\delta))/2\delta$ | 高 | 2 次额外 SCF |
| `autodiff` | JAX 自动微分 $\partial E/\partial f$；训练时使用安全 fallback，避免二阶 AD 不稳定 | 高 | 分数占据 SCF + AD/fallback |
| `full_scf_ad` | 对分数占据自洽能量 $E^*_{\text{SCF}}(f)$ 直接做 AD，并允许训练梯度穿过该导数 | 最严格 | 很高 |
| `fixed_orbital_ad` | 轨道固定时的 AD | 近似 | 无额外 SCF |
| `half_charge_ad` | 半占据 AD | 中 | 1 次分数占据 SCF |

### 5.4 参考 DM21 的训练策略差距

| DM21 做法 | 当前项目 | 差距 |
|-----------|----------|------|
| 三阶段：pretrain → fractional → SCF finetune | 两阶段：基态 SCF → 激发态 fixed-density | 缺少独立的分数电荷预训练阶段 |
| 分数电荷数据增强（对每个分子计算 N±δ 电子） | `fractional_linearity_penalty` 存在但未集成到 trainer | 需要在 `NeuralXCTrainer` 中串入 |
| 分数自旋约束（stretched H₂） | 无 | 需新增 |
| 所有约束同时启用 | 约束权重默认全为 0 | trainer 需暴露约束开关 |
| SCF 收敛后密度约束 | 阶段 2 用 fixed-density 绕开 SCF | 阶段 2 无密度反馈 |

### 5.5 建议的训练流程改进

**P0: Trainer 支持分阶段训练计划**

```python
# 当前: NeuralXCTrainer.kernel(steps=50, loss="ground_state")
# 建议: NeuralXCTrainer.kernel(schedule=[
#     {"mode": "self_consistent", "steps": 500, 
#      "energy_weight": 1.0, "density_weight": 0.1, "fractional_weight": 0.3},
#     {"mode": "fixed_density", "steps": 200,
#      "energy_weight": 0.0, "excitation_weight": 1.0, "oscillator_weight": 0.5},
# ])
```

`TrainingSchedule` 作为 `list[PhaseConfig]`，每 phase 有不同的 `mode`、`steps`、`learning_rate` 和约束权重。

**P1: 分数自旋约束**

对 stretched H₂（R > 4Å），计算 unrestricted 和 restricted 的总能，差值应 → 0：

$$\mathcal{L}_{\text{spin}} = \lambda_{\text{spin}} \cdot \left(E_{\text{unrestricted}}^{\text{pred}} - E_{\text{restricted}}^{\text{pred}}\right)^2$$

`DM21LikeFunctional` 内部已有 `_restricted_spin_density_blocks` 和 unrestricted 路径，可以直接构造此约束。

**P2: 激发态 fine-tune 中的密度一致性**

阶段 2 用 fixed-density 意味着激发态梯度只优化 MLP 在固定格点特征上的系数，不优化密度。这可能导致泛函在激发态 fine-tune 后基态性能退化。

改进：阶段 2 中周期性插入基态 SCF 验证（不参与训练），监控基态能量漂移：

$$\text{drift} = \left|E_{\text{GS}}^{\text{pred}}(\theta_{\text{step}=t}) - E_{\text{GS}}^{\text{ref}}\right|$$

如果漂移超过阈值，降低激发态约束权重或临时切回 self_consistent 模式。

---

## 6. 优先级总结

| 优先级 | 改进 | 范围 | 类型 |
|--------|------|------|------|
| **P0** | Trainer 支持分阶段训练计划（schedule） | `training/` | 训练流程 |
| **P0** | 全部约束默认启用 + 可配置权重（非全部 0） | `training/config.py` | 训练流程 |
| **P0** | Mode 3 semilocal 系数约束（softmax 统一） | `neural_xc/dm21` | 泛函形式 |
| **P1** | 分数自旋约束（stretched H₂） | `training/targets.py` | 训练约束 |
| **P1** | 激发态 fine-tune 中基态能量漂移监控 | `training/` | 训练流程 |
| **P1** | QAC 参数可训练化 | `neural_xc/dm21` | 泛函形式 |
| **P1** | HF 格点投影缓存（bind 时复用） | `neural_xc/dm21` | 性能 |
| **P2** | UEG 极限正则化 | `neural_xc/dm21` | 训练约束 |
| **P2** | RS-ω 可预测（atomwise head） | `nn_rsh` | 泛函形式 |
| **P2** | 输入特征可学习化 | `neural_xc/dm21` | 模型架构 |
| **P3** | 空间平滑性约束 | `training/` | 训练约束 |
| **P3** | 低精度 MLP 推理 | `neural_xc/dm21` | 性能 |

---

## 7. Implementation Specification

### 7.1 P0: Semilocal 系数 softmax 统一 (Mode 3)

**File: `src/td_graddft/neural_xc/dm21/functional.py`**

**7.1.1 Modify `_sanitize_coefficients` (line 1265)**

Current (Mode 3 path, line 1277):
```python
semilocal = jnp.clip(safe[..., :n_semilocal], 0.0, self.kernel_clip)
```

Replace with temperature-controlled softmax:
```python
# New parameter: self.coefficient_temperature (float, default 1.0)
T = jnp.asarray(self.coefficient_temperature, dtype=safe.dtype)
semilocal_raw = safe[..., :n_semilocal]
semilocal = jax.nn.softmax(semilocal_raw / T, axis=-1)
```

Add `coefficient_temperature` to `DM21LikeFunctional.__init__` defaults (line 585-636):
```python
coefficient_temperature: float = 1.0  # T→∞ = uniform; T→0 = one-hot
```

**7.1.2 Modify `_assemble_channel_contributions` for Mode 3 (line 1531)**

Current:
```python
semilocal = coefficients[..., :n_semilocal] * basis[..., :n_semilocal]
```

No change needed if softmax is applied in `_sanitize_coefficients` — the coefficients already arrive softmax-normalized.

### 7.2 P1: Fractional spin constraint

**File: `src/td_graddft/training/targets.py`**

Add function after `fractional_charge_linearity_penalty` (line 2596):

```python
def fractional_spin_penalty(
    params: PyTree,
    functional: Any,
    neutral_molecule: Any,
    *,
    training_config: GroundStateTrainingConfig | None = None,
    r_cutoff: float = 4.0,  # H-H distance in Angstrom for stretched H2
) -> Array:
    """Fractional spin constraint: stretched H2 restricted vs unrestricted energy gap.
    
    E_restricted(R→∞) - E_unrestricted(R→∞) should → 0 for exact functional.
    """
    cfg = GroundStateTrainingConfig() if training_config is None else training_config
    
    # Use existing infrastructure: build a stretched H2 molecule
    from td_graddft.gto import M
    mol_stretched = M(
        atom=f"H 0 0 0; H 0 0 {r_cutoff}",
        basis="def2-svp",
        unit="Angstrom",
    )
    
    # Restricted SCF
    restr_mol = _resolve_training_molecule_with_mode(params, functional, mol_stretched, cfg)
    e_restricted = _predict_ground_state_total_energy_from_molecule(
        params, functional, restr_mol
    )
    
    # Unrestricted SCF (via DifferentiableSCF with unrestricted mode)
    uks_mol = _resolve_unrestricted_training_molecule(params, functional, mol_stretched, cfg)
    e_unrestricted = _predict_ground_state_total_energy_from_molecule(
        params, functional, uks_mol
    )
    
    return jnp.abs(e_restricted - e_unrestricted)
```

### 7.3 P1: Multi-phase training schedule

**File: `src/td_graddft/training/neural_xc_trainer.py`**

Add `TrainingPhase` dataclass and modify `NeuralXCTrainer.kernel`:

```python
@dataclass(frozen=True)
class TrainingPhase:
    steps: int
    mode: str = "self_consistent"
    energy_weight: float = 1.0
    density_weight: float = 0.0
    fractional_weight: float = 0.0
    janak_weight: float = 0.0
    excitation_weight: float = 0.0
    learning_rate: float = 1e-3

@dataclass
class NeuralXCTrainer:
    functional: Any
    molecules: Sequence[Any] = field(default_factory=tuple)
    
    def kernel(self, *, phases: Sequence[TrainingPhase], ...) -> TrainingResult:
        """Execute multi-phase training. Each phase has its own config and step count."""
        ...
```

### 7.4 Implementation order

1. Add `coefficient_temperature` parameter → `DM21LikeFunctional`
2. Modify `_sanitize_coefficients` softmax path
3. Verify Mode 3 backward compatibility (temperature=0 → old behavior with clip)
4. Add `fractional_spin_penalty` to `targets.py`
5. Add `TrainingPhase` to `neural_xc_trainer.py`
6. Wire multi-phase into `NeuralXCTrainer.kernel`
