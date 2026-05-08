# Unrestricted Differentiable Charged-State SCF 与严格 Janak 实施文档

**日期**: 2026-04-23  
**状态**: 执行稿  
**目标**: 在 TD-GradDFT 中补齐 `unrestricted differentiable SCF`，并以“固定 functional、对轨道占据数变分”的严格定义实现 Janak reward，同时把当前半可微的 Koopmans/IP-EA 路径升级为未来可全可微的统一框架。

**状态更新（2026-04-23）**

- `unrestricted differentiable SCF` 的 `unrolled` 主路径已完成。
- `strict Janak autodiff` 已接入训练分发，但训练态目前仍保守地走 strict-branch semidiff / `unrolled` 路径。
- `unrestricted implicit commutator` 已在 solver 层落地，采用 spin-stacked residual 的 `jvp/vjp` fixed-point adjoint。
- 上层 strict Janak / Koopmans 还没有切到这条 full-gradient unrestricted implicit path，这仍是后续工作边界。

---

## 1. 背景与结论

当前 `nn_rsh` 主线已经具备三块基础：

1. restricted 的 differentiable self-consistent SCF
2. 独立可运行的 pure-JAX `UKS`
3. 基于 charged-state `UKS` 的 Koopmans/IP-EA 诊断与半可微 loss

但这三块还没有被统一成一个理论与实现上自洽的框架。

当前最核心的缺口不是单个 loss，而是：

**缺少一个支持固定自旋分辨占据的 unrestricted differentiable SCF 主干。**

一旦这层补齐，下面三件事会自然统一：

- 整数带电态 `N-1 / N / N+1`
- 分数占据态 `f_{pσ}`
- 严格 Janak 与 Koopmans/IP-EA

本轮实现的主线结论如下：

1. 不再把“charged-state SCF”和“严格 Janak”视为两条独立路线。
2. 先补一个统一的 `unrestricted differentiable SCF` 底座。
3. 严格 Janak 不是“最小化总能量”，而是最小化  
   `∂E*/∂f_{pσ} - ε_{pσ}` 的残差。
4. 对 `nn_rsh` 而言，严格 Janak 必须在**冻结 functional 参数**的前提下定义。
5. 当前 Koopmans loss 只是过渡版，后续要迁移到同一个 unrestricted differentiable SCF 上。

---

## 2. 当前代码状态

### 2.1 已有能力

- [src/td_graddft/scf/uks.py](/Users/jiaoyuan/Documents/GitHub/TD-GradDFT/src/td_graddft/scf/uks.py)
  - 已是独立的 unrestricted SCF
  - 不依赖 `pyscf_bridge`
  - 已支持 `bound_xc`
  - 已支持 `nn_rsh` 的 spin-resolved local XC 与 spin-specific LR exchange

- [src/td_graddft/training/targets.py](/Users/jiaoyuan/Documents/GitHub/TD-GradDFT/src/td_graddft/training/targets.py)
  - 已有 `charged_state_uks_from_reference(...)`
  - 已有 `koopmans_ip_ea_diagnostic(...)`
  - 已有 restricted 的 Janak / fractional 训练路径

- [src/td_graddft/nn_rsh/losses.py](/Users/jiaoyuan/Documents/GitHub/TD-GradDFT/src/td_graddft/nn_rsh/losses.py)
  - 已接入半可微 `koopmans_ip_weight / koopmans_ea_weight`
  - 当前 charged-state 分支在反传中显式 `detach`

### 2.2 当前真正缺失的能力

- [src/td_graddft/scf/differentiable.py](/Users/jiaoyuan/Documents/GitHub/TD-GradDFT/src/td_graddft/scf/differentiable.py)
  - 仍然是 restricted-only
  - 关键假设包括：
    - `_restricted_channel(...)`
    - `_restricted_total_occupations(...)`
    - `0.5 * density` 的闭壳层 spin 包装
    - 单 density 矩阵 `D` 的 self-consistent residual

- 严格 Janak 目前尚未实现
  - 当前训练中的 Janak 仍是 restricted frontier finite-difference 路径
  - 还不是一般的 `∂E*/∂f_{pσ} = ε_{pσ}`

- charged-state Koopmans 还不是全可微
  - 当前 `N±1` 分支用的是独立 `UKS`
  - 可前向评估，但反传不穿过 charged-state SCF

### 2.3 当前已验证基线

已验证通过的当前基线：

- `pytest -q tests/test_scf_uks.py tests/test_reference_uks.py`
  - `8 passed`
- `pytest -q tests/test_nn_rsh_density_descriptor.py -k koopmans tests/test_training.py -k koopmans`
  - `2 passed`

这说明：

- unrestricted `UKS + bound_xc` 已可用
- Koopmans 诊断与半可微 loss 已接通
- 下一步可以安全转向 differentiable unrestricted 主干

---

## 3. 理论定义

### 3.1 固定占据下的变分能量

对给定 functional 参数 `θ`，给定自旋分辨占据向量 `f = {f_{pσ}}`，定义：

```text
E*θ(f) = min_C Eθ[C, f]
subject to  C† S C = I
```

其中：

- `C` 是自旋轨道系数
- `f_{pσ}` 是自旋分辨轨道占据
- `E*θ(f)` 是该固定占据约束下完全变分优化后的自洽总能

### 3.2 严格 Janak 定理

严格 Janak 定义为：

```text
∂E*θ(f) / ∂f_{pσ} = ε_{pσ}(f)
```

也就是说，外层 reward 不应是“能量越低越好”，而应是 Janak 残差：

```text
L_J = Σ_{pσ} ρ( ∂E*θ(f)/∂f_{pσ} - ε_{pσ}(f) )
```

第一阶段不需要对所有轨道做这一约束，只需要对前沿相关轨道做：

- 中性态去电子方向的 HOMO 分支
- 中性态加电子方向的 LUMO 分支

### 3.3 Janak 与 Koopmans 的关系

两者不应被看作竞争关系，而是同一框架下的不同观测：

- Janak:
  - 局部的占据导数关系
  - 依赖分数占据态
- Koopmans/IP-EA:
  - 整数粒子数差分关系
  - 依赖 `N-1` 与 `N+1` 带电态

统一底座就是：

- 同一个 unrestricted differentiable SCF
- 允许固定整数/分数占据

### 3.4 对 `nn_rsh` 的关键限制

`nn_rsh` 当前是 density-driven 参数化。  
如果分数占据改变后，network 又重新预测了一套 `(sr, lr, omega)`，则占据导数不再是固定 functional 下的 Janak。

因此严格 Janak 必须满足：

```text
先在参考态上解析出一套 RSH 参数
然后在占据变分路径上冻结该 functional
只允许轨道与密度变分
```

这条规则同样适用于严格 Koopmans/IP-EA。

---

## 4. 总体架构目标

目标不是再做一个“特化的 differentiable charged-state solver”，而是把现有 `DifferentiableSCF` 泛化成：

**一个支持 restricted / unrestricted、整数 / 分数占据、固定 functional / 自洽 functional 的统一可微 SCF 主干。**

统一后的能力应覆盖：

1. restricted neutral self-consistent ground state
2. unrestricted charged open-shell self-consistent state
3. unrestricted fractional-occupation self-consistent state
4. strict Janak reward
5. full-gradient Koopmans/IP-EA loss

---

## 5. 实现原则

### 5.1 先做一般底座，不做特例拼补

不建议：

- 在当前 restricted `DifferentiableSCF` 上继续加更多 if/else 特判
- 单独复制一份 `DifferentiableUKS` 后面再维护两套逻辑

建议：

- 抽象出 spin-aware 的 occupation / density / Fock / residual 主干
- 在同一文件内保留 restricted 与 unrestricted 两条前向路径
- 优先共享：
  - DIIS 逻辑
  - implicit commutator 逻辑
  - grid XC potential bridge

### 5.2 先 unrolled，再 implicit

执行顺序必须是：

1. unrestricted `unrolled` self-consistent SCF 跑通
2. 数值与测试稳定
3. 再补 unrestricted `implicit_commutator`

不要一开始就同时改两条梯度路径。

### 5.3 先支持固定占据，再谈 Janak

严格 Janak 的前提是：

- SCF 支持外部传入固定 `mo_occ_alpha`
- SCF 支持外部传入固定 `mo_occ_beta`
- 并在整个自洽迭代中保持不变

如果 solver 自己会重新按能级重填充占据，就不是 Janak 路径。

### 5.4 先冻结 functional，再对占据求导

严格 Janak 与 strict Koopmans 在 `nn_rsh` 上都必须遵守：

- 先 `bind_to_molecule(...)`
- 冻结解析后的 `(sr, lr, omega)`
- 再进入整数/分数占据分支

---

## 6. 工作包拆解

## WP1. 把 `DifferentiableSCF` 泛化为 spin-aware 主干

### 目标

让 [src/td_graddft/scf/differentiable.py](/Users/jiaoyuan/Documents/GitHub/TD-GradDFT/src/td_graddft/scf/differentiable.py) 支持 unrestricted self-consistent SCF。

### 必须完成的子任务

1. 替换 restricted-only 入口
   - 抽出通用 orbital/occupation 解析函数
   - 支持：
     - `(nao, nmo)` / `(nmo,)`
     - `(2, nao, nmo)` / `(2, nmo)`

2. 替换单 density 状态变量
   - restricted:
     - `D`
   - unrestricted:
     - `Dα, Dβ`

3. 抽象通用 Fock builder
   - unrestricted 目标形式：

```text
Fα = h + J[Dα + Dβ] - a_full K[Dα] + Vxc,α + Vextra,α
Fβ = h + J[Dα + Dβ] - a_full K[Dβ] + Vxc,β + Vextra,β
```

4. 升级 SCF-XC 接口
   - 当前已有 restricted `SCFXCContributions`
   - unrestricted 路径需要支持：
     - `v_rho_alpha / v_rho_beta`
     - `v_grad_alpha / v_grad_beta`
     - `extra_fock_alpha / extra_fock_beta`

5. 升级 iterate state
   - `density_history`
   - `mo_coeff_history`
   - `mo_energy_history`
   - `rms_history`
   都要支持 unrestricted 形状

### 文件范围

- [src/td_graddft/scf/differentiable.py](/Users/jiaoyuan/Documents/GitHub/TD-GradDFT/src/td_graddft/scf/differentiable.py)

### 验收标准

- 中性闭壳层 reference 走 unrestricted 路径时能退化回 restricted 结果
- 开壳层 `N±1` toy 体系能 self-consistent 收敛
- forward 与 gradient 都有限

---

## WP2. 支持固定自旋分辨占据的 differentiable SCF

### 目标

让 solver 支持显式传入固定 occupation specification，而不是每次按本征值自动回填。

### 新能力

- `mo_occ_alpha_fixed`
- `mo_occ_beta_fixed`

这些量一旦传入，在整个 SCF 迭代中必须保持不变。

### 为什么这是核心

这一步完成后，同一个 solver 就可以统一支持：

- 中性整数占据
- `N-1`
- `N+1`
- fractional HOMO/LUMO
- 一般开壳层分数占据

### 文件范围

- [src/td_graddft/scf/differentiable.py](/Users/jiaoyuan/Documents/GitHub/TD-GradDFT/src/td_graddft/scf/differentiable.py)
- 可能需要同步整理 [src/td_graddft/training/config.py](/Users/jiaoyuan/Documents/GitHub/TD-GradDFT/src/td_graddft/training/config.py) 中的 SCF 配置语义

### 验收标准

- 对固定 occupation 的 unrestricted toy 系统，SCF 前后 occupation 不发生漂移
- 分数占据下 density/Fock/energy 均有限

---

## WP3. 先做 unrestricted unrolled gradient

### 目标

先让 unrestricted self-consistent path 具备最直接的反传能力。

### 范围

- 只做 `gradient_mode="unrolled"`
- 不先动 implicit VJP

### 验收标准

- `jax.value_and_grad(...)` 能对 unrestricted self-consistent total energy 反传
- 所有梯度 finite
- 不出现 tracer leak / Python bool on tracer / 形状分支错误

---

## WP4. 补 unrestricted implicit commutator

### 目标

把当前 restricted 的 `implicit_commutator` 推广到 unrestricted。

### unrestricted residual

建议用联合 residual：

```text
Rα = Fα Dα S - S Dα Fα
Rβ = Fβ Dβ S - S Dβ Fβ
```

再把两个 residual 组合成一个线性系统。

### 说明

这一步比 WP1-WP3 更难，不应提前做。

### 验收标准

- unrestricted implicit path 和 unrolled path 在小体系梯度方向上大体一致
- 可用于训练而不产生系统性 NaN

---

## WP5. 严格 Janak reward

### 目标

在固定 bound functional 的前提下，实现严格占据变分 Janak reward。

### 推荐定义

对某个前沿分支定义分数参数 `η`：

```text
f_H,σ(η): HOMOσ 占据减少 η
f_L,σ(η): LUMOσ 占据增加 η
```

相应的 reward 写成：

```text
L_H = ρ( - dE*θ(f_H,σ(η))/dη - ε_H,σ(η) )
L_L = ρ(   dE*θ(f_L,σ(η))/dη - ε_L,σ(η) )
```

总 Janak reward：

```text
L_J = mean_η [L_H(η) + L_L(η)]
```

### 实现策略

第一版建议：

- 只做前沿轨道
- 用多点 `η` 平均
  - 例如 `{0.05, 0.10, 0.15}`
- functional 冻结在参考中性态

### 技术选择

有两种实现顺序：

1. 过渡版
   - 多点 finite difference
   - 先提升稳定性
2. 严格版
   - 对 `η` 直接 `jax.grad`
   - 真正匹配 `∂E*/∂f`

建议顺序：

- 先做可微的 occupation-parameterized energy path
- 再在同一接口上把 finite difference 过渡到 `jax.grad`

### 文件范围

- [src/td_graddft/training/targets.py](/Users/jiaoyuan/Documents/GitHub/TD-GradDFT/src/td_graddft/training/targets.py)
- [src/td_graddft/nn_rsh/losses.py](/Users/jiaoyuan/Documents/GitHub/TD-GradDFT/src/td_graddft/nn_rsh/losses.py)

### 验收标准

- 分数占据态完全通过 differentiable unrestricted SCF 求出
- Janak reward 不再依赖 restricted-only frontier helper
- water / aniline 等小体系上 reward 与参数更新方向稳定

---

## WP6. 把 Koopmans/IP-EA 从半可微升级为全可微

### 当前状态

当前 [src/td_graddft/nn_rsh/losses.py](/Users/jiaoyuan/Documents/GitHub/TD-GradDFT/src/td_graddft/nn_rsh/losses.py) 中：

- `koopmans_ip_weight`
- `koopmans_ea_weight`

已经可用，但 charged-state 分支显式 `detach`。

### 目标形式

```text
L_K = ρ( ε_HOMO(N)   + E(N-1) - E(N) )
    + ρ( ε_HOMO(N+1) + E(N)   - E(N+1) )
```

一旦 unrestricted differentiable charged-state SCF 完成：

- 去掉 charged-state detach
- 直接对 `E(N-1)` / `E(N+1)` 反传

### 文件范围

- [src/td_graddft/nn_rsh/losses.py](/Users/jiaoyuan/Documents/GitHub/TD-GradDFT/src/td_graddft/nn_rsh/losses.py)
- [src/td_graddft/training/targets.py](/Users/jiaoyuan/Documents/GitHub/TD-GradDFT/src/td_graddft/training/targets.py)

### 验收标准

- charged-state branch 不再 detach
- forward/backward 稳定
- Koopmans loss 能显著推动参数，而不是只靠 neutral Janak

---

## 7. 关键实现注意事项

### 7.1 不要把严格 Janak 误写成“最小化总能量”

错误思路：

```text
reward = -E
```

这不是 Janak。

正确思路：

```text
reward = - |∂E*/∂f - ε|
```

### 7.2 不要让 `nn_rsh` 在分数占据或 charged-state 分支重新预测参数

这是严格定义上的红线。

应该始终：

1. 在参考态解析出一套 RSH 参数
2. 冻结 bound functional
3. 再做 fractional / charged branches

### 7.3 先修 solver，再修 loss

当前最大的技术债在 solver，不在 reward 公式。

如果 solver 仍然 restricted-only，那么：

- 严格 Janak 只能继续是 proxy
- Koopmans 只能继续是半可微

### 7.4 不要同时修改 unrolled 与 implicit 两条路径

这会极大放大调试难度。

推荐顺序：

1. unrestricted unrolled
2. unrestricted tests
3. unrestricted implicit

---

## 8. 测试计划

## 8.1 solver 层

新增/扩展测试文件建议：

- `tests/test_differentiable_scf_unrestricted.py`
- `tests/test_differentiable_scf_fractional_occ.py`

测试内容：

- unrestricted neutral 闭壳层退化到 restricted
- unrestricted open-shell toy 系统 forward 有限
- fixed fractional occupations 保持不变
- unrolled gradient 有限

## 8.2 target/loss 层

- strict Janak reward 对 toy system 可计算
- Koopmans full-gradient path 对 toy system 可计算
- gradients finite

## 8.3 regression 层

必须保留现有通过项：

- `tests/test_scf_uks.py`
- `tests/test_reference_uks.py`
- `tests/test_training.py -k koopmans`
- `tests/test_nn_rsh_density_descriptor.py -k koopmans`

---

## 9. 里程碑与交付标准

### M1. unrestricted differentiable unrolled SCF

交付标准：

- unrestricted self-consistent path 跑通
- fixed occupations supported
- toy gradients finite

### M2. strict Janak occupation path

交付标准：

- frozen-functional fractional self-consistent path 跑通
- Janak reward 可以稳定前向与反向

### M3. unrestricted implicit commutator

交付标准：

- implicit path 可用
- unrestricted training 不依赖 unrolled 才能稳定

### M4. full-gradient Koopmans/IP-EA

交付标准：

- charged-state branches 不再 detach
- `koopmans_ip_weight / koopmans_ea_weight` 真正全可微

---

## 10. 下一步编码顺序

本轮上下文结束后，下一位实现者应直接按以下顺序执行：

1. 从 [src/td_graddft/scf/differentiable.py](/Users/jiaoyuan/Documents/GitHub/TD-GradDFT/src/td_graddft/scf/differentiable.py) 开始
2. 把 restricted-only 的 occupation / density / Fock 逻辑拆成 spin-aware 主干
3. 先落 unrestricted `unrolled` self-consistent path
4. 加最小 toy tests
5. 通过后再把 strict Janak target 改到该路径

禁止顺序：

- 先写 strict Janak loss 再去补 unrestricted differentiable SCF
- 先写 unrestricted implicit 再写 unrestricted unrolled
- 让 charged/fractional 分支重新预测 `nn_rsh` 参数

---

## 11. 当前冻结决策

以下决策在本阶段视为已经拍板：

1. 先补 `unrestricted differentiable SCF`
2. 严格 Janak 建立在固定 functional、对占据数变分的定义上
3. `nn_rsh` 的 strict Janak / strict Koopmans 路径都必须冻结 functional
4. 当前 Koopmans 半可微实现只是过渡态，不是最终设计
5. 下一阶段的主文件是 `src/td_graddft/scf/differentiable.py`

---

## 12. 相关文档

- [2026-04-16-general-rsh-groundstate-plan.md](/Users/jiaoyuan/Documents/GitHub/TD-GradDFT/docs/plans/2026-04-16-general-rsh-groundstate-plan.md)

本文件是上面总方案的后续执行文档，专门处理：

- unrestricted differentiable SCF
- strict Janak
- full-gradient Koopmans/IP-EA
