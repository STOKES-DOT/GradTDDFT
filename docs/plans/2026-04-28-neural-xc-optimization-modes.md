# 神经网络泛函架构与优化模式

**日期:** 2026-04-28
**范围:** `src/td_graddft/neural_xc/`、`src/td_graddft/nn_rsh/`、`src/td_graddft/training/`

---

## 1. 架构总览

### 1.1 两类神经网络泛函

```
                              ┌──────────────────────────┐
                              │    Neural XC Framework     │
                              └──────────┬───────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
          ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐
          │  NeuralXCFunctional │  │ DM21LikeFunctional│  │ TrainableRSHFunctional│
          │  (neural_xc/base)  │  │ (neural_xc/dm21)  │  │   (nn_rsh)          │
          └────────┬──────────┘  └────────┬──────────┘  └─────────┬───────────┘
                   │                      │                        │
                   ▼                      ▼                        ▼
            点态 MLP 系数           DM21 风格混合系数           可训练 RSH 参数
            c_θ: R^d → R^n         c_θ(r): R^d(r) → R^n      (ω, α, β) → global scalars
                                                                 或 atomwise head
```

- **NeuralXCFunctional** (`neural_xc/base/functional.py`)：最简形式，点态 MLP 从 LDA 输入特征预测能量密度基组系数
- **DM21LikeFunctional** (`neural_xc/dm21/functional.py`)：完整 DM21 风格泛函，支持 semilocal + HF + PT2 通道 + 严格 TDDFT 响应核
- **TrainableRSHFunctional** (`nn_rsh/functional.py`)：可训练的范围分离杂化泛函，预测 ω、α、β 参数

### 1.2 泛函的数学形式

**DM21 风格泛函（主要的 neural_xc 路径）**

基态 XC 能量写为格点上的基组展开：

$$E_{xc}[n] = \int w(\mathbf{r}) \cdot e_{xc}(\mathbf{r}) \, d\mathbf{r}$$

其中局部能量密度展开为：

$$e_{xc}(\mathbf{r}) = \sum_k c_k(\mathbf{r}; \theta) \cdot e_k(\mathbf{r})$$

- $e_k(\mathbf{r})$：基组通道，包括半局域交换-相关（B3LYP 组分：lda_x, gga_x_b88, lda_c_vwn, gga_c_lyp 等）、投影 HF、投影 PT2
- $c_k(\mathbf{r}; \theta) = \text{MLP}_\theta(\mathbf{x}(\mathbf{r}))$：神经网络预测的点态混合系数
- $\mathbf{x}(\mathbf{r})$：局部密度特征（ρ, |∇ρ|, τ, ∇²ρ + semilocal/HF/PT2 能量密度描述符）

**范围分离杂化泛函（nn_rsh 路径）**

Fock 算符中的 HF 交换项按 ω 分离：

$$\frac{1}{r_{12}} = \underbrace{\frac{\text{erfc}(\omega r_{12})}{r_{12}}}_{\text{短程 HF}} + \underbrace{\frac{\text{erf}(\omega r_{12})}{r_{12}}}_{\text{长程 HF}}$$

XC 能量：

$$E_{xc}^{\text{RSH}} = \alpha_{\text{SR}} E_{x}^{\text{HF,SR}}(\omega) + \alpha_{\text{LR}} E_{x}^{\text{HF,LR}}(\omega) + (1-\alpha_{\text{SR}})E_{x}^{\text{DFA,SR}}(\omega) + (1-\alpha_{\text{LR}})E_{x}^{\text{DFA,LR}}(\omega) + E_{c}^{\text{DFA}}$$

其中 $(\omega, \alpha_{\text{SR}}, \alpha_{\text{LR}})$ 可由神经网络从分子描述符预测。

---

## 2. 优化模式一：能量监督训练（Energy Optimization）

### 2.1 数学框架

**目标函数：**

$$\mathcal{L}(\theta) = \frac{1}{N_{\text{mol}}} \sum_{m=1}^{N_{\text{mol}}} w_m \cdot \ell(E_{m}^{\text{pred}}, E_{m}^{\text{ref}})$$

**前向传播：**

给定分子几何 $\{\mathbf{R}_A\}$，构建格点 $\{\mathbf{r}_g\}_{g=1}^{N_{\text{grid}}}$，计算：

**Step 1 — 密度特征提取：**

$$\rho(\mathbf{r}) = 2\sum_{i}^{\text{occ}} |\phi_i(\mathbf{r})|^2, \quad \nabla\rho(\mathbf{r}), \quad \tau(\mathbf{r}) = \frac{1}{2}\sum_{i}^{\text{occ}} |\nabla\phi_i(\mathbf{r})|^2$$

**Step 2 — 基组通道构建：**

半局域通道（以 B3LYP 组分为例）：

$$e_1^{\text{lda}_x}(\mathbf{r}) = -\frac{3}{4}\left(\frac{3}{\pi}\right)^{1/3} \rho^{4/3}(\mathbf{r})$$
$$e_2^{\text{gga}_x^{\text{b88}}}(\mathbf{r}) = -\beta \rho^{4/3} \frac{x^2}{1+6\beta x \sinh^{-1}x}, \quad x = \frac{|\nabla\rho|}{\rho^{4/3}}$$
$$e_3^{\text{lda}_c^{\text{vwn}}}(\mathbf{r}) = \rho \cdot \varepsilon_c^{\text{VWN}}(r_s), \quad r_s = (3/4\pi\rho)^{1/3}$$

投影 HF 通道（从 AO 密度矩阵构建）：

$$e_{\text{HF}}(\mathbf{r}) = -\frac{1}{2}\sum_{\mu\nu\lambda\sigma} D_{\mu\nu} D_{\lambda\sigma} \iint \frac{\phi_\mu(\mathbf{r})\phi_\lambda(\mathbf{r}')\phi_\nu(\mathbf{r})\phi_\sigma(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|} d\mathbf{r}'$$

**Step 3 — 输入特征：**

$$\mathbf{x}(\mathbf{r}) = \left[\rho, |\nabla\rho|, \tau, \frac{e_{\text{semilocal}}}{\rho}, \frac{e_{\text{HF}}}{\rho}, \ldots \right] \in \mathbb{R}^{d_{\text{in}}}$$

输入经过 log-squash 预处理：

$$x_i' = \log(|x_i| + \varepsilon) \quad \text{或} \quad x_i' = \frac{1}{2}\log(x_i^2 + \varepsilon^2)$$

**Step 4 — MLP 前向：**

$$\mathbf{h}^{(0)} = \mathbf{W}^{(0)}\mathbf{x}' + \mathbf{b}^{(0)}, \quad \mathbf{h}^{(0)} = \tanh(\mathbf{h}^{(0)})$$

$$\mathbf{h}^{(l+1)} = \mathbf{h}^{(l)} + \text{LayerNorm}\left(\mathbf{W}^{(l)}\mathbf{h}^{(l)} + \mathbf{b}^{(l)}\right), \quad \mathbf{h}^{(l+1)} = \text{ELU}(\mathbf{h}^{(l+1)})$$

$$\mathbf{c}(\mathbf{r}) = \sigma_{\text{scale}} \cdot \text{sigmoid}\left(\frac{\mathbf{W}^{\text{head}}\mathbf{h}^{(L)} + \mathbf{b}^{\text{head}}}{\sigma_{\text{scale}}}\right)$$

**Step 5 — 能量积分：**

$$E_{xc}^{\text{pred}}[\rho] = \sum_{g} w_g \sum_k c_k(\mathbf{r}_g) \cdot e_k(\mathbf{r}_g)$$

$$E_{\text{tot}}^{\text{pred}} = E_{\text{kin}} + E_{\text{ext}} + E_{\text{H}} + E_{xc}^{\text{pred}}$$

**Step 6 — 损失函数（能量匹配）：**

$$\ell_{\text{MSE}}(E^{\text{pred}}, E^{\text{ref}}) = (E^{\text{pred}} - E^{\text{ref}})^2$$

$$\ell_{\text{MAE}}(E^{\text{pred}}, E^{\text{ref}}) = |E^{\text{pred}} - E^{\text{ref}}|$$

### 2.2 梯度反向传播

**∂L/∂θ 的计算路径：**

$$\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial E^{\text{pred}}} \cdot \sum_g \sum_k \frac{\partial E^{\text{pred}}}{\partial c_k(\mathbf{r}_g)} \cdot \frac{\partial c_k(\mathbf{r}_g)}{\partial \theta}$$

其中：

$$\frac{\partial E^{\text{pred}}}{\partial c_k(\mathbf{r}_g)} = w_g \cdot e_k(\mathbf{r}_g)$$

$$\frac{\partial c_k(\mathbf{r}_g)}{\partial \theta} = \frac{\partial \text{MLP}_\theta(\mathbf{x}(\mathbf{r}_g))}{\partial \theta}$$

当使用**固定密度**（fixed-density）模式时，$\mathbf{x}(\mathbf{r}_g)$ 和 $e_k(\mathbf{r}_g)$ 都是常数（由参考基态提供），梯度只流经 MLP 参数 θ。

### 2.3 密度约束扩展

可额外加入密度矩阵匹配：

$$\mathcal{L}_{\text{density}} = \ell_{\text{energy}} + \lambda_D \cdot \|D^{\text{pred}} - D^{\text{ref}}\|_F^2$$

XC 势匹配：

$$\mathcal{L}_{\text{potential}} = \ell_{\text{energy}} + \lambda_V \cdot \|v_{xc}^{\text{pred}} - v_{xc}^{\text{ref}}\|^2$$

其中 $v_{xc}^{\text{pred}}$ 通过格点投影得到：

$$v_{xc}^{\text{pred}}(\mathbf{r}) = \frac{\delta E_{xc}^{\text{pred}}}{\delta \rho(\mathbf{r})} = \sum_k \left[c_k \frac{\partial e_k}{\partial \rho} + e_k \frac{\partial c_k}{\partial \rho}\right]$$

### 2.4 关键实现细节

**系数先验（Coefficient Prior）：**

对于 B3LYP 默认基组，初始系数设为 B3LYP 的解析值：
$$\mathbf{c}^{\text{prior}} = (0.20, 0.72, 1.0, 0.81)$$

先验模式为 `"mean"` 时，MLP 输出与先验做加权平均：
$$\mathbf{c}^{\text{eff}} = \lambda \cdot \mathbf{c}_{\text{MLP}} + (1-\lambda) \cdot \mathbf{c}^{\text{prior}}$$

**HF 通道模式：**

- `"total_only"`：输入只用总 HF 能量密度 $e_{\text{HF}} = e_{\text{HF}}^\alpha + e_{\text{HF}}^\beta$
- `"spin_resolved"`：分别使用 $e_{\text{HF}}^\alpha, e_{\text{HF}}^\beta$

**响应 HF 模式（用于 TDDFT）：**

- `"nonlocal_exchange_only"`：HF 交换直接通过非局域 Fock 算符贡献到响应矩阵
- `"local_projected"`：HF 交换的 f_xc 核通过局部投影近似，$\alpha$ 从 MLP 输出的 HF 场积分得到：

$$\alpha = \frac{\int w(\mathbf{r}) \rho(\mathbf{r}) \cdot h_{\text{HF}}(\mathbf{r}) d\mathbf{r}}{\int w(\mathbf{r}) \rho(\mathbf{r}) d\mathbf{r}}$$

---

## 3. 优化模式二：自洽监督训练（Self-Consistent Optimization）

### 3.1 数学框架

自洽模式不走固定密度路径，而是将泛函插入可微 SCF 循环中，从轨道能量和总能构建物理约束损失。

**前向传播（SCF 嵌入）：**

$$\rho^{(0)} \xrightarrow{\text{build } v_{xc}[\rho^{(0)}]} \rho^{(1)} \xrightarrow{\text{build } v_{xc}[\rho^{(1)}]} \cdots \xrightarrow{\text{converged}} \rho^*$$

收敛后得到 $\{\varepsilon_i^*, \phi_i^*, E_{\text{tot}}^*\}$。

### 3.2 Janak 定理约束

Janak 定理将轨道能量与总能对占据数的导数联系起来：

$$\frac{\partial E}{\partial f_i} = \varepsilon_i$$

其中 $f_i$ 是 i 轨道的占据数。对于分数占据，可以通过有限差分验证：

**Janak 残差（HOMO）：**

$$R_{\text{Janak}} = \varepsilon_{\text{HOMO}} - \frac{E[f_{\text{HOMO}} + \delta] - E[f_{\text{HOMO}} - \delta]}{2\delta}$$

当泛函精确时，$R_{\text{Janak}} \rightarrow 0$。

**实际实现：**

在分数占据的 SCF 计算中，构造 $f_i^{\pm} = f_i \pm \delta \cdot \delta_{i,\text{HOMO}}$，分别做 SCF，计算有限差分：

$$\Delta E_{\text{FD}} = \frac{E(f_{\text{HOMO}}+\delta) - E(f_{\text{HOMO}}-\delta)}{2\delta}$$

**Janak 损失：**

$$\mathcal{L}_{\text{Janak}} = |\varepsilon_{\text{HOMO}}(f) - \Delta E_{\text{FD}}|$$

### 3.3 Koopmans IP/EA 约束

严格的 Koopmans 定理要求：

$$\varepsilon_{\text{HOMO}}(N) = -IP(N), \quad \varepsilon_{\text{LUMO}}(N) = -EA(N)$$

其中：

$$IP(N) = E(N-1) - E(N), \quad EA(N) = E(N) - E(N+1)$$

**三态 Koopmans 损失：**

$$\mathcal{L}_{\text{Koopmans}} = w_{\text{IP}} \cdot |\varepsilon_{\text{HOMO}} + E_{\text{cation}} - E_{\text{neutral}}| + w_{\text{EA}} \cdot |\varepsilon_{\text{LUMO}} + E_{\text{neutral}} - E_{\text{anion}}|$$

这需要对中性分子、阳离子（N-1 电子）、阴离子（N+1 电子）分别做 SCF 计算。

**Koopmans 带隙约束：**

$$\mathcal{L}_{\text{gap}} = |(\varepsilon_{\text{LUMO}} - \varepsilon_{\text{HOMO}}) - (IP - EA)|$$

$$= |(\varepsilon_{\text{LUMO}} - \varepsilon_{\text{HOMO}}) - (E_{\text{cation}} + E_{\text{anion}} - 2E_{\text{neutral}})|$$

### 3.4 分数电荷线性约束

精确泛函的 $E(N)$ 应在整数 N 之间呈分段线性。对分数电子数 $N+\delta$：

$$E(N+\delta) = (1-\delta)E(N) + \delta E(N+1) \quad \forall \delta \in [0,1]$$

**分数电荷线性度损失：**

$$\mathcal{L}_{\text{frac}} = |E(N+\delta) - [(1-\delta)E(N) + \delta E(N+1)]|$$

即对 $\delta=0.5$ 做一次分数占据 SCF，计算偏离线性的程度：

$$\mathcal{L}_{\text{frac}} = |E(N+0.5) - \frac{1}{2}[E(N) + E(N+1)]|$$

### 3.5 RSH 参数优化

对于 `TrainableRSHFunctional`，优化变量是 $(\omega, \alpha_{\text{SR}}, \alpha_{\text{LR}})$。

**前向（参数 → SCF）：**

神经网络（可选 atomwise head）从分子描述符预测 RSH 参数，然后用这些参数做 SCF 得到轨道能量和总能。

**参数边界处理：**

每个参数被限制在预设边界 $[p_{\min}, p_{\max}]$ 中，通过 sigmoid 映射：

$$p = p_{\min} + (p_{\max} - p_{\min}) \cdot \sigma(p_{\text{raw}})$$

其中 $\sigma(x) = 1/(1+e^{-x})$。

**先验惩罚（Prior Penalty）：**

$$\mathcal{L}_{\text{prior}} = \frac{1}{3}\left[\left(\frac{\alpha_{\text{SR}}-\alpha_{\text{SR}}^0}{\Delta\alpha_{\text{SR}}}\right)^2 + \left(\frac{\alpha_{\text{LR}}-\alpha_{\text{LR}}^0}{\Delta\alpha_{\text{LR}}}\right)^2 + \left(\frac{\omega-\omega^0}{\Delta\omega}\right)^2\right]$$

### 3.6 梯度流分析（自洽模式）

自洽模式下的梯度需要流经整个 SCF 循环。利用隐函数定理：

$$\frac{\partial E^*}{\partial \theta} = \frac{\partial E}{\partial \theta}\bigg|_{\rho^*} + \int \frac{\delta E}{\delta \rho(\mathbf{r})}\bigg|_{\rho^*} \cdot \frac{\partial \rho^*(\mathbf{r})}{\partial \theta} d\mathbf{r}$$

在 SCF 收敛点 $\partial E/\partial \rho = 0$（稳定点条件），因此：

$$\frac{\partial E^*}{\partial \theta} = \frac{\partial E}{\partial \theta}\bigg|_{\rho^*}$$

这简化了梯度计算：只需对收敛密度处的显式泛函参数求导，无需反传通过整个 SCF 迭代。

**对于 Koopmans IP/EA 损失：**

当 `koopmans_detach_charged_states=True` 时，带电态的 SCF 对泛函参数做 stop_gradient，梯度只流经中性态的轨道能量：

$$\frac{\partial \mathcal{L}_{\text{Koopmans}}}{\partial \theta} \approx \frac{\partial \varepsilon_{\text{HOMO}}^{\text{neutral}}}{\partial \theta}$$

当 `koopmans_detach_charged_states=False` 时，梯度通过 JAX 的自动微分流经全部三态 SCF：

$$\frac{\partial \mathcal{L}_{\text{Koopmans}}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial \varepsilon_{\text{HOMO}}} \frac{\partial \varepsilon_{\text{HOMO}}}{\partial \theta} + \frac{\partial \mathcal{L}}{\partial E_{\text{cation}}} \frac{\partial E_{\text{cation}}}{\partial \theta} + \frac{\partial \mathcal{L}}{\partial E_{\text{anion}}} \frac{\partial E_{\text{anion}}}{\partial \theta}$$

---

## 4. 两种模式的对比

| 特性 | 模式一：能量监督 | 模式二：自洽监督 |
|------|----------------|----------------|
| **前向路径** | 固定密度 → MLP 预测系数 → 能量积分 | SCF → 收敛密度 → 轨道能量 + 总能 |
| **损失信号** | $E_{\text{pred}} - E_{\text{ref}}$ | $\varepsilon_{\text{HOMO}} + E_{\text{cation}} - E_{\text{neutral}}$ 等 |
| **梯度复杂度** | 仅 MLP 参数 | MLP/RSH 参数 + 隐式 SCF 梯度 |
| **是否需要 SCF** | 否（fixed-density）或 1 次 SCF（single-shot） | 是（self-consistent），每步 3+ 次 SCF |
| **物理约束** | 能量 → 间接约束泛函形式 | Janak / Koopmans / 分数电荷 → 直接约束泛函性质 |
| **训练稳定性** | 高（监督学习） | 中（需要 damping + level_shift） |
| **适用场景** | 基态能量预测、TDDFT 激发能 | 前线轨道能量、带隙、RSH 参数调优 |

---

## 5. GPU 性能分析

### 5.1 计算热点

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| 格点特征提取 | $O(N_{\text{grid}} \cdot N_{\text{ao}}^2)$ | AO 密度投影到格点 |
| MLP 前向 | $O(N_{\text{grid}} \cdot \sum_l d_l d_{l+1})$ | 逐格点的 MLP |
| 能量积分 | $O(N_{\text{grid}} \cdot N_{\text{channels}})$ | `tensordot(weights, e_xc)` |
| SCF 循环（自洽模式） | $O(N_{\text{iter}} \cdot [N_{\text{ao}}^4 + N_{\text{grid}}])$ | 每轮 Fock 构建 + 对角化 |
| 三态 SCF（Koopmans） | $3\times$ 以上 | 中性 + 阳离子 + 阴离子 |

### 5.2 格点 MLP 的 GPU 特性

`DM21MixingMLP` 和 `GradDFTResidualMixingMLP` 在 Flax 中实现，JAX 的 `vmap` 将 MLP 自动批量化到所有格点：

```python
# Flax NN 天然支持 per-point 批量化：
# inputs 形状: (n_grid_points, n_input_features)
# 输出形状: (n_grid_points, n_output_channels)
c = MLP(x)  # 一次 XLA launch 处理所有格点
```

这已经是 GPU 最优的实现方式（单次 matrix-matrix multiply 完成所有格点上的全连接层）。

**主要瓶颈在格点数量**：对精细格点（grids_level=3+），$N_{\text{grid}}$ 可达 5000-20000，MLP 的 `Dense` 层计算量线性增长。

### 5.3 自洽模式的 SCF 开销

自洽模式下每步训练需要 1-3 次完整 SCF（取决于损失配置）。SCF 开销在积分引擎优化方案（`2026-04-28-jax-integral-gpu-fusion.md`）中已覆盖。

---
