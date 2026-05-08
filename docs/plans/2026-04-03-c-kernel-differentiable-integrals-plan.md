# 用 C Kernel 重构电子积分并保持 JAX 可微的方案

**日期**: 2026-04-03  
**状态**: 阶段 1 已落地；阶段 1.5 已落地（`libcint_geometry_grad_policy = error/zero`）  
**目标**: 在不破坏当前 JAX SCF/TDDFT/训练链路可微性的前提下，引入接近 PySCF/libcint 的 C kernel 电子积分后端，解决 `1e/2e` 积分在 CPU 上明显慢于 PySCF 的问题，并为后续 GPU benchmark 建立清晰边界。

---

## 0. 最新进展（2026-04-03）

- 已在 strict-JAX reference 构建入口新增 `integral_backend` 选项：
  - `jax`：现有纯 JAX 积分路径
  - `libcint`：通过 PySCF/libcint 提供 AO 值积分（`S/T/Vnuc/ERI/dipole`），SCF/TDDFT 仍在 JAX 中运行
- 已在 workflow `SimulationConfig` 新增 `jax_integral_backend` 并完成透传。
- 本地 CPU smoke（H2O / STO-3G / PBE）结果：
  - `jax` 与 `libcint` 的 SCF 能量、激发能、振子强度对齐到 `~1e-9 ~ 1e-12` 量级。
  - 参考构建时间（含积分+SCF，单次冷启动）`libcint` 显著快于当前 JAX 积分实现。
- 当前仍是“值积分前向加速”阶段：
  - 已支持 `libcint_geometry_grad_policy`：
    - `error`（默认）：请求几何梯度时显式报错，防止 silent failure；
    - `zero`：将 libcint 积分视作几何常数，返回零几何梯度以保持训练链路运行。
  - 几何梯度相关的导数积分 kernel 与严格 VJP 仍未接入。

## 1. 当前结论

### 1.1 现在和 PySCF 已经对齐的部分

- 计算对象一致：
  - 当前 `two_electron.py` 计算的仍然是化学家记号 `(ij|kl)`。
- 基组和归一化逻辑一致：
  - strict-JAX basis loader 已按 PySCF `make_env / _normalize_contracted_ao / bas_ctr_coeff` 对齐。
- 数值结果一致：
  - `water / STO-3G / 6-31G` 等测试上，`ERI max_abs_diff` 已压到 `1e-10` 量级。
- 下游可微链路一致：
  - SCF、XC、TDA/Casida、spectrum、training 都在 JAX 图内。

### 1.2 现在和 PySCF 仍然不一致的部分

- 最底层 primitive/shell kernel 逻辑不同。
  - `PySCF` 走 `libcint` 的 shell-level C kernel。
  - 当前实现是：
    - `ssss` 闭式表达式
    - `s/p` 的 JAX recurrence
    - 更高角动量很多时候仍依赖 `apply_cartesian_derivatives_4c(...)` 从 `ssss` 抬上去
- 对称性输出路径不同。
  - `PySCF` 原生支持 `s8/s4/s1` 等格式直接写出。
  - 当前实现通常是先算 batch，再自己做对称 scatter 或 pair-restore。
- 优化器/预筛逻辑不同。
  - `PySCF` 有 `cintopt`、`GTOnr2e_fill_drv`、原生 shell slice 驱动。
  - 当前实现主要依赖 JAX 分组、`jit`、显式 Schwarz screening。

### 1.3 为什么当前方法会慢

当前瓶颈不在 TDDFT，也不主要在 `restore/scatter`，而在 primitive/shell kernel 本身：

- `PySCF/libcint`：
  - shell-level C recurrence
  - `ao_loc + shls_slice`
  - `cintopt`
  - 无 JIT 编译开销
- 当前 JAX ERI：
  - per-signature `jit`
  - `ssss + recurrence/导数抬角动量`
  - 首次调用有显著 XLA compile cost
  - 高角动量和较大 primitive contraction 时 kernel 复杂度迅速上升

当前最好结果说明了这一点：

- `water / STO-3G / full ERI / CPU`
  - `PySCF ≈ 0.00146 s`
  - `JAX warm ≈ 0.00204 s`
- `water / 6-31G / full ERI / CPU`
  - `PySCF ≈ 0.00529 s`
  - `JAX warm ≈ 0.0475 s`

结论：

- 小基组上外层数据流已经被压到接近 PySCF。
- 更大的 primitive contraction 下，差距主要来自 shell kernel，不是外围组织。

---

## 2. 为什么可以考虑 C kernel，同时保持可微

### 2.1 对神经网络泛函参数 `theta` 的训练

当前训练里真正需要反传的是 `theta`，不是 AO 积分本身。

数据流是：

```text
R, basis
  -> electronic integrals
  -> JAX SCF
  -> JAX XC / response kernel
  -> JAX TDA/Casida
  -> loss(theta)
```

如果积分层对 `theta` 不显式依赖，那么：

- `h/S/ERI` 对 `theta` 是常数
- `dL/dtheta` 不需要穿过积分 kernel
- 积分层即便由 C 实现，也不会破坏对 `theta` 的训练可微性

这意味着：

- **C kernel 不会破坏当前以 XC 参数为核心的训练链路**

### 2.2 如果将来要对几何 `R` 也可微

如果将来目标扩展到几何优化、力或核坐标梯度，那么仅有积分值不够，还需要导数积分：

- `dS/dR`
- `dh/dR`
- `d(ij|kl)/dR`

这时正确方案不是让 JAX 自动微分穿过 C 代码，而是：

- forward:
  - C kernel 返回积分值
- backward:
  - 通过 `custom_vjp`/`custom_jvp`
  - 调用对应的导数积分 kernel

结论：

- **对 `theta` 可微**：C kernel 可以直接接入
- **对 `R` 可微**：需要值积分 + 导数积分 + 手工 VJP

---

## 3. 建议的后端架构

### 3.1 增加积分 backend 抽象

对 `1e/2e` 积分统一抽象成 backend：

- `backend="jax"`
- `backend="libcint"`

其中：

- `jax`：
  - 保留当前 strict-JAX 全流程能力
  - 便于 GPU 原生探索
- `cint`：
  - 目标是 CPU 上对齐 PySCF/libcint 的速度
  - 不再依赖 PySCF Python 层，但可以参考其 `libcint` 调用方式

### 3.2 backend 责任边界

`cint` backend 负责：

- `int1e_ovlp`
- `int1e_kin`
- `int1e_nuc`
- `int2e`
- 后续可扩展：
  - `int1e_r`
  - `int1e_rinv`
  - `int2e_ip1` 等导数积分

JAX 侧继续负责：

- SCF 迭代
- XC 能量与势
- 响应核
- TDA/Casida
- 光谱
- 训练 loss 与梯度

### 3.3 推荐的数据流

```text
MoleculeSpec / BasisSpec
  -> basis loader
  -> integral backend (jax or cint)
  -> h, S, ERI, dipole, optional derivative integrals
  -> JAX SCF
  -> JAX TDDFT / training
```

---

## 4. 具体实现建议

### 4.1 第一阶段：只解决 CPU 上的积分速度

先不动训练主链路，只替换积分来源。

做法：

1. 在 `data/integrals` 层引入 backend dispatcher
2. 让 `reference.py` / strict-JAX reference builder 可以配置：
   - `integral_backend="jax"`
   - `integral_backend="libcint"`
3. 先只实现值积分，不做几何导数

这一步的目标是：

- 在 CPU 上使 `1e/2e` 接近 PySCF/libcint 的 wall-clock
- 不影响现有基于 `theta` 的可微训练

### 4.2 第二阶段：把 C kernel 接入 JAX 图

如果要保持统一 API，可在 JAX 侧包一层 primitive：

- forward:
  - 调 C kernel，返回 `jnp.asarray(...)`
- backward:
  - 对 `theta` 无需额外处理
  - 对 `R` 以后再补 `custom_vjp`

### 4.3 第三阶段：补几何可微

需要补：

- 一电子导数积分
- 双电子导数积分
- 对应 `VJP/JVP`

这一步才真正实现：

- `R -> integrals -> SCF -> TDDFT -> loss`

的全链路几何可微

---

## 5. 为什么这条路合理

### 5.1 它解决的是当前真正的瓶颈

当前瓶颈是电子积分，尤其是 full ERI 无 DF 路径，而不是：

- SCF
- TDA/Casida
- 光谱展宽

所以继续在 TDDFT 层优化，不能解决 reference build 明显慢于 PySCF 的问题。

### 5.2 它不会推翻当前 JAX 训练架构

当前项目最大的价值是：

- JAX SCF
- JAX TDDFT
- JAX training
- 可微 Neural XC

C kernel 只替换上游积分值生成，不会要求重写这些部分。

### 5.3 它符合分阶段演进

先达成：

- CPU 积分像 PySCF 一样快

再考虑：

- 几何导数
- GPU 原生积分
- 真正的 3-index DF / RI

这比继续在当前 primitive JAX ERI kernel 上做小修补更现实。

---

## 6. 风险与边界

### 6.1 这不能直接解决 GPU 上的积分加速

C kernel 的自然运行位置仍然是 CPU。

所以它能解决的是：

- CPU 上的积分吞吐

但不能直接保证：

- GPU 上 full pipeline 从积分到 TDDFT 都快

如果目标是 GPU 真正受益，后续仍要继续做：

- 3-index / DF
- 避免 full ERI materialization
- GPU 原生积分 kernel 或混合策略

### 6.2 它会带来额外的工程复杂度

需要维护：

- C kernel wrapper
- backend dispatch
- 值积分与导数积分接口
- JAX primitive/custom_vjp 封装

所以必须分阶段推进，先做最小闭环。

---

## 7. 当前不修改代码时的 GPU 测试建议

在真正引入 C kernel 之前，当前方法的 GPU 测试应该只回答两个问题：

1. 当前 strict-JAX 积分/SCF/TDDFT 哪些阶段能真正利用 GPU
2. 当前方法的 wall-clock 主要浪费在：
   - compile
   - CPU 积分
   - host/device 数据搬运
   - 还是 TDDFT eigensolve

建议 benchmark 拆成：

- `basis/grid prep`
- `1e integrals`
- `2e integrals`
- `SCF`
- `TDA/Casida`
- `spectrum`

并同时记录：

- `first`
- `warm`
- `CPU vs GPU`

这样后面再切 C kernel 或 DF 时，才能看清每一步是否真的收益。

---

## 8. 后续执行顺序

建议顺序：

1. 先在 GPU 上 benchmark 当前 strict-JAX 方法
2. 确认瓶颈仍主要在 `1e/2e`
3. 再开始引入 `cint` backend skeleton
4. 先只替换值积分
5. 之后再补几何导数和更深层可微

---

## 9. 最终判断

当前最合理的技术判断是：

- 继续在现有 JAX primitive ERI kernel 上做局部修补，边际收益已经明显下降。
- 如果目标是 **CPU 上积分速度接近 PySCF**，应引入接近 `libcint` 的 C kernel backend。
- 如果目标是 **保持当前 Neural XC 训练可微**，这条路不会构成障碍，因为当前训练主要对 `theta` 求导，而不是对积分本身求导。
- 如果目标最终还包括 **几何可微**，则必须在 C kernel 基础上补导数积分与手工 VJP。
