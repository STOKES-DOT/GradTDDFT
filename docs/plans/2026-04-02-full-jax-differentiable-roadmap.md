# TD-GradDFT 全流程 JAX 可微与加速路线图

**日期**: 2026-04-02  
**状态**: 执行中  
**目标**: 在保持与 PySCF 数值对齐的前提下，逐步摆脱 PySCF 参考构建依赖，最终实现可微、可 `jit`、可并行的全流程 JAX SCF + TDA/Casida + 光谱计算。

补充原则：

- 项目默认主干应当是 `GradDFT ground-state core + differentiable excited-state extension`
- 基态 neural XC / SCF 抽象保持 GradDFT-first
- 激发态训练允许直接对激发能、振子强度、光谱等量做监督
- 任何架构重构前，先维持基本 DFT/TDDFT 与 PySCF 的数值对齐基线

---

## 1. 当前状态

### 1.0 补充文档

- 关于“引入 C kernel 以替代当前 CPU 侧积分，并保持 JAX 训练/TDDFT 链路可微”的方案整理，见：
  - [2026-04-03-c-kernel-differentiable-integrals-plan.md](/Users/jiaoyuan/Documents/GitHub/TD-GradDFT/docs/plans/2026-04-03-c-kernel-differentiable-integrals-plan.md)
- 关于“对照 GPU4PySCF 审计当前 GPU 利用率瓶颈，并规划向 GPU-first 数据流靠拢”的方案整理，见：
  - [2026-04-08-gpu4pyscf-alignment.md](/Users/jiaoyuan/Documents/GitHub/TD-GradDFT/docs/plans/2026-04-08-gpu4pyscf-alignment.md)

### 1.1 已经完成的部分

- `basis`:
  - strict-JAX basis loader 已能直接读取 vendored PySCF basis snapshot。
  - shell contraction normalization 已按 PySCF `make_env/_normalize_contracted_ao/bas_ctr_coeff` 逻辑对齐。
- `AO on grid`:
  - 已有 JAX 实现，支持 `deriv=0/1`。
  - `grid_ao_backend="jax"` 时不依赖 `pyscf.dft.numint.eval_ao`。
- `one-electron / two-electron integrals`:
  - 已有纯 JAX `S/T/V/ERI` 实现。
  - 已新增纯 JAX `int1e_rinv` / grid-batched `nu_cache`，支持 point-Coulomb 与 `erf(omega r)/r`。
  - 已使用 `jit`、`vmap`、`fori_loop`，并在部分批处理路径接入 `pmap`。
- `strict-JAX reference`:
  - closed-shell spec-driven `JAX-RKS` 主路径已不再依赖 `gto.M / gen_grid / numint.eval_ao / int1e_r / int1e_rinv`。
  - `local_hfx_features` / `hfx_nu` 已可在 strict-JAX path 中直接构建。
- `workflow/preset 入口`:
  - 默认 `water_experiment_config(...)` / `benzene_experiment_config(...)` 已切到 strict-JAX spec-driven path。
  - `SimulationConfig` 包级默认已切到 `scf_backend="jax_rks"` 与 `jax_grid_ao_backend="jax"`。
  - 旧的 PySCF-mf preset 仅保留为 `legacy_*_experiment_config(...)` 兼容入口。
  - `reference.py` 已收缩为 strict-JAX 主模块；`*_from_pyscf*` legacy builder 实现已外移到 `reference_legacy.py`。
- `SCF`:
  - RHF/RKS/UKS 已有 JAX 实现。
  - `self_consistent + implicit_commutator` 路径可微，训练梯度有限。
  - 在当前 strict-JAX `water / PBE / STO-3G / grids_level=0` 审计中，
    `self_consistent + unrolled` 对 ground / excitation / spectrum 也给出全有限梯度；
    但默认训练路径仍优先使用 `implicit_commutator`，因为它在更广配置上的稳健性更好。
- `TDDFT/TDA/Casida`:
  - 激发态层已与 PySCF 高精度对齐。
  - `jit warm` 后在 QH9 和 anthracene 上相对 PySCF 有明显加速。

### 1.2 仍然依赖 PySCF 的部分

- `run_reference(...)` 的 reference/benchmark 路径仍以 PySCF `mf` 为主要上游对象。
- `restricted_reference_from_pyscf*` / `unrestricted_reference_from_pyscf*` 系列仍以 PySCF `mol/mf/grids` 为输入。
- `grid_ao_backend="pyscf"` fallback 仍依赖 `numint.eval_ao`。
- unrestricted spec-driven strict-JAX reference 仍未建立，当前 spec path 只覆盖 restricted closed-shell。
- PySCF 仍是当前数值对照、benchmark 和部分旧脚本的运行依赖。

### 1.3 现有 benchmark 结论

- 激发态层 `jit warm` 后已快于 PySCF：
  - `QH9 id=104 / PBE / STO-3G / 8 states`
    - `auto jit TDA` 相对 PySCF 约 `15x`
    - `auto jit spectrum` 相对 PySCF 约 `9.4x`
  - `anthracene / PBE0 / STO-3G / 5 states`
    - `auto jit TDA` 相对 PySCF 约 `4.9x`
    - `auto jit Casida` 相对 PySCF 约 `8.4x`
- 但全流程仍然慢：
  - `water / PBE / STO-3G / grid_ao_backend=jax`
  - `grid_prep ≈ 0.85 s`
  - `integral_prep ≈ 61.38 s`
  - `jax_cpu_full_tda_s ≈ 65.97 s`
  - `pyscf_cpu_full_tda_s ≈ 0.148 s`

**结论**:
- 当前“激发态层加速”已经成立。
- 当前“全流程无额外成本”尚不成立。
- 瓶颈明确集中在 `reference build / grid / integrals`。

### 1.4 2026-04-02 积分层优化尝试记录

已完成并验证的几轮积分优化：

1. `primitive contraction` 从多层 `fori_loop` 改为 `vmap + dot(highest precision)`  
   - 精度保持与 PySCF 对齐。
   - `water / STO-3G` warm：
     - `overlap ≈ 13 ms`
     - `hcore ≈ 31 ms`
     - `eri ≈ 228 ms`
   - 对 `PySCF intor` 仍明显慢，但数值误差已压到 `1e-9 ~ 1e-10` 量级。

2. 放宽 shell-block 自动启用阈值  
   - 结论：**小体系明显变慢**。
   - `water / STO-3G` 的 `eri first/warm` 都退化，已回退该策略。

3. 预打包 shell 元数据与 quartet 分组  
   - `CartesianBasis` 现在预计算：
     - shell padded exponents / coefficients
     - shell AO index padding
     - AO quartet groups
     - shell quartet groups
   - 目的：把 Python 分桶/stack 开销从 warm 路径中移除。
   - 当前 `water / STO-3G` 已恢复到较优水平：
     - `eri first ≈ 52.35 s`
     - `eri warm ≈ 0.228 s`
   - `ethylene / STO-3G` 上，shell-block scatter 向量化后的 `eri warm`
     从约 `5.23 s` 压到约 `0.158 s`，数值误差仍维持在 `~3e-10`。

4. 2026-04-03 五轮小步调优（固定口径：`ethylene / STO-3G / PySCF vs JAX integrals`）
   - `iter1`: 去掉 quartet 索引的 host `list` 往返
     - `eri warm ≈ 0.1540 s`
     - 结论：基本持平
   - `iter2`: 预构造 shell-block 内部 `scalar_kernel` 列表，避免在 block kernel 内重复临时构造
     - `eri warm ≈ 0.1574 s`
     - 结论：无收益
   - `iter3`: 将 8 次对称 shell-block scatter 收到缓存的 `jit` helper
     - `eri warm ≈ 0.0517 s`
     - 结论：显著有效，约 `3x` 提升
   - `iter4`: `QUARTET_BATCH_CHUNK = 256`
     - `eri warm ≈ 0.0483 s`
   - `iter5`: `QUARTET_BATCH_CHUNK = 512`
     - `eri warm ≈ 0.0482 s`
   - 当前最优配置：
     - `QUARTET_BATCH_CHUNK = 512`
     - `ethylene / STO-3G`：
       - `PySCF eri ≈ 0.0201 s`
       - `JAX eri warm ≈ 0.0482 s`
       - `max_abs_diff ≈ 2.85e-10`
   - 额外回归试验：精简双电子积分热点路径的 `jit` 边界
     - 做法：去掉 `2e` 热点路径里内层 `scalar kernel` 的 `jit`，只保留最外层 batched kernel 的 `jit`
     - 结果：
       - `ethylene / STO-3G / ERI first ≈ 194.24 s`
       - `ethylene / STO-3G / ERI warm ≈ 0.5623 s`
       - `max_abs_diff ≈ 2.85e-10`
     - 对比当前最优实现：
       - `ERI first`: `115.31 s -> 194.24 s`
       - `ERI warm`: `0.0482 s -> 0.5623 s`
     - 结论：对双电子积分热点路径，不能直接删除内层 `jit`；当前 nested-`jit` 边界虽然更重，但实际吞吐更好。

当前判断：

- `1e` 路径已经相对稳定，但在 CPU 上仍远慢于 PySCF 的 C 积分实现。
- `2e` 路径仍是决定性瓶颈。
- 想达到“CPU 上 warm 也快过 PySCF full intor”的目标，单纯继续打磨当前 AO 级 full-ERI materialization 可能不够，需要进一步评估：
  - 更激进的 shell-block scatter 向量化
  - 3-index / density-fitting 路径
  - 避免 full `n^4` tensor materialization 的 JK/response 设计

### 1.5 2026-04-03 density fitting 五轮优化记录

说明：

- 当前 `df` 路径还不是 auxiliary-basis 3-index DF，而是对 full AO ERI 的 AO-pair 矩阵做对称低秩分解：
  `(pq|rs) ~= sum_Q B_Q[p,q] B_Q[r,s]`
- 这与 PySCF 的主流 `density fitting / RI` 逻辑不同。PySCF 走的是：
  - auxiliary basis
  - 3-center integrals `(P|pq)`
  - 2-center metric `(P|Q)` 及其逆或 Cholesky
  - 再由 `B_{P,pq}` 直接构造 `J/K` 与 MO 响应块
- 当前实现虽然在数值上可以逼近 dense AO ERI contraction，但在复杂度和数据流上没有真正绕开
  `AO-pair` 空间，也没有真正对齐 PySCF 的 DF 逻辑。
- 这正是 benzene/6-31G strict-JAX 全链路明显卡在 reference build 的根因之一。当前 `jk_backend='df'`
  仍然会先走重的 AO-pair / ERI-pair 构造与分解，而不是 PySCF 那种 3c2e/2c metric 路线。
- 因此本轮 benchmark 的目标是：
  - 在保持 `J/K` 数值严格对齐的前提下
  - 优化 `eri_to_df_factors(...)` 和 `build_jk_from_df(...)`
  - 重点观察 `df_factor_s` 与 `df_jk_warm_s`

固定口径：

- `ethylene / STO-3G / CPU / JAX vs dense JK`

基线：

- `dense_jk_s ≈ 0.02159 s`
- `df_factor_s ≈ 0.01225 s`
- `df_jk_warm_s ≈ 1.82e-05 s`
- `J/K max_abs_diff ≈ 1e-13`

### 1.6 2026-04-03 strict-JAX ERI 预编译入口

- 新增 `precompile_eri_kernels(...)`：
  - 位置：[src/td_graddft/data/integrals/two_electron.py](/Users/jiaoyuan/Documents/GitHub/TD-GradDFT/src/td_graddft/data/integrals/two_electron.py)
  - 作用：将 `2e` kernel 的 compile 成本显式前置，避免把 compile 时间混入第一次 reference / SCF / TDDFT 计算。
- strict-JAX workflow 配置新增：
  - `SimulationConfig.jax_precompile_eri`
  - `SimulationConfig.jax_precompile_eri_chunk_size`
- strict-JAX spec-driven reference 构建已接线：
  - `run_reference_from_spec(...)` 会把上述配置透传到
    `restricted_reference_from_spec_with_jax_rks(...)`
- 当前策略：
  - 不改变默认物理结果
  - 不默认开启预编译
  - 由 workflow/benchmark 显式选择是否把 compile 成本前置

五轮尝试：

1. `iter1`: `build_jk_from_df(...)` 改为 fused-einsum 内核
   - `df_factor_s ≈ 2.06e-03 s`
   - `df_jk_warm_s ≈ 1.47e-05 s`
   - 结论：有效，warm 提升约 `19%`

2. `iter2`: `eri_to_df_factors(...)` 改为对称 packed-pair 空间分解，再恢复为 full factors
   - `df_factor_s ≈ 9.61e-04 s`
   - `df_jk_warm_s ≈ 1.82e-05 s`
   - 结论：factorization 明显下降，但 warm 回到接近基线

3. `iter3`: `build_jk_from_df(...)` 改为 batched `matmul + trace` 复用
   - `df_factor_s ≈ 8.99e-04 s`
   - `df_jk_warm_s ≈ 1.72e-05 s`
   - 结论：虽然 warm 不如 `iter1`，但结合 factorization 一次成本后，总成本最低

4. `iter4`: 回退 fused-einsum 内核，并用 single-gather 恢复 full factors
   - `df_factor_s ≈ 1.57e-03 s`
   - `df_jk_warm_s ≈ 1.50e-05 s`
   - 结论：warm 接近最优，但 factorization 不如 `iter2`

5. `iter5`: `rho_aux` 标量收缩改为显式 elementwise multiply-reduce
   - `df_factor_s ≈ 1.04e-03 s`
   - `df_jk_warm_s ≈ 1.61e-05 s`
   - 结论：整体仍不如 `iter2`

当前保留版本：

- 保留 `iter3`
- 也就是：
  - `packed-pair` factorization
  - `batched matmul + trace` 的 `DF J/K` 内核

原因：

- 对真实 SCF 使用场景，更重要的是一次 factorization 之后的多次 `J/K` 调用总成本。
- 以几十到上百次 `J/K` 评估计，`iter3` 的总成本低于其余四轮，同时精度保持在
  `J/K max_abs_diff ≈ 1e-13`。

---

## 2. 目标状态

最终目标不是“局部 JAX”，而是以下链路全部由 JAX 主导：

```text
atom/basis spec
  -> strict-JAX basis loader
  -> strict-JAX molecular grid
  -> strict-JAX AO / AO-grad on grid
  -> strict-JAX one-electron integrals
  -> strict-JAX ERI / JK
  -> strict-JAX RHF/RKS/UKS SCF
  -> strict-JAX TDA / Casida / spectrum
  -> strict-JAX differentiable training
```

并满足：

- 数值上与 PySCF 严格对齐。
- 激发态层与训练链路保持可微。
- 热启动场景下通过 `jit` 和批处理向量化得到实质加速。

---

## 3. 分阶段执行计划

### Phase A: 去掉 spec-based reference 对 PySCF 的依赖

目标：

- 让 `restricted_reference_from_*_spec_with_jax_rks(...)` 不再需要：
  - `gto.M`
  - `gen_grid.Grids`
  - `int1e_r`
  - `numint.eval_ao`

需要完成：

1. strict-JAX `basis_from_spec`
2. strict-JAX `build_molecular_grid`
3. strict-JAX `dipole_integrals`
4. strict-JAX `nuclear_repulsion`
5. strict-JAX spec-based JAX-RKS reference builder
6. strict-JAX workflow entry: `run_reference_from_spec(...)` / `run_pipeline_core_from_spec(...)`

验收：

- 水分子 `STO-3G / PBE / level=0`
  - `|ΔE_tot| < 2e-5 Ha`
  - `TDA(3 states) MAE < 2e-3 eV`
  - `dipole_integrals` 与 PySCF 对齐

### Phase B: 把 workflow 主入口从 PySCF-mf 转向 spec-driven JAX

目标：

- workflow 可以直接吃 `atom/basis/xc` 配置。
- PySCF 只保留为 benchmark/reference backend，不再是运行前提。

需要完成：

1. 新增 strict-JAX reference workflow path
2. 将 `SimulationConfig`/`run_reference` 的 spec path 抽象出来
3. 保留 PySCF 对照入口用于 benchmark

当前已完成的第一步：

- 已新增 spec-driven core API：
  - `run_reference_from_spec(...)`
  - `run_pipeline_core_from_spec(...)`
- 已新增 experiment/pipeline 双轨配置：
  - `SystemConfig(mf_builder=...)`
  - `SystemConfig(reference_spec=...)`
- 已新增 strict-JAX presets：
  - `water_strict_jax_experiment_config(...)`
  - `benzene_strict_jax_experiment_config(...)`
- 已新增简化 public API：
  - `td_graddft.api.MoleculeConfig`
  - `td_graddft.api.build_reference(...)`
  - `td_graddft.api.run_pipeline(...)`
  - `td_graddft.api.run_spectrum_pipeline(...)`
- `pyscf_bridge.py` 现已明确标记为 deprecated compatibility shim。
- 这两条路径的目标是让新脚本不再需要先构造 PySCF `mf`。
- 下一步再把 experiment/preset 层从 `mf_builder` 模式逐步迁移到 spec-driven 模式。

验收：

- 本地 water 和 QH9 单分子能直接走 strict-JAX spec path。
- 不引入新的 PySCF hard dependency 到运行主路径。

### Phase C: 清理 residual PySCF kernels

目标：

- 清理剩余 `mol/grids/mf` 输入型 PySCF 依赖，并把主入口彻底切到 strict-JAX spec path。

需要完成：

1. 让 experiment / preset / benchmark 默认改走 `reference_spec`
2. 为 unrestricted 路径补 strict-JAX reference builder
3. 保留 PySCF 只作为对照 backend，不再参与默认运行主路径

验收：

- `reference.py` 运行主路径不再 import PySCF。

### Phase D: 性能优化

目标：

- 将上游 `reference build` 成本压下来，让全流程 benchmark 有实际竞争力。

重点：

1. 积分层减少 Python loop 与碎片化 JIT
2. 统一 batch shape，增加可复用 compiled kernel
3. 更细的 shell-block / pair-block 调度
4. 合理使用 `jit` / `vmap` / `pmap`
5. 按阶段输出 `grid / integrals / scf / tda / casida / spectrum` 分项计时

验收：

- 至少在中等体系上，全流程 JAX 相对 PySCF 不再被上游 prep 完全拖垮。

---

## 4. 当前优先级排序

按紧急程度，当前只做下面三件事：

1. 清理 `run_reference/run_pipeline` 默认入口中的 `mf_builder` 依赖
2. 将 benchmark / preset / example 逐步切换到 `reference_spec`
3. 继续对 water 与代表性分子维持 PySCF 对照

原因：

- strict-JAX closed-shell reference 主路径已经建立，下一步要解决的是“默认入口仍旧绕回 PySCF”。
- 只有把实验入口迁走，`pyscf_bridge.py` 才能真正逐步消失。
- water 对照仍是最低成本且最敏感的回归基准。

---

## 5. 对齐与验收策略

每一次改动都必须同时给出两类结果：

### 5.1 数值对齐

- basis shell coefficients
- overlap / hcore / dipole / ERI
- SCF total energy
- TDA / Casida excitation energies
- oscillator strengths / broadened spectrum

### 5.2 性能代价

- cold compile
- jit warm
- grid prep
- integral prep
- scf
- tda
- casida
- spectrum

禁止只报告“更 JAX”而不报告：

- 是否还依赖 PySCF
- 是否引入额外 wall-clock 开销
- 是否破坏与 PySCF 的数值一致性

---

## 6. 严格 JAX 全链路可微性审计

当前已经固化的审计入口：

- [tools/check_strict_jax_full_chain_differentiability.py](/Users/jiaoyuan/Documents/GitHub/TD-GradDFT/tools/check_strict_jax_full_chain_differentiability.py)
- [tests/test_training.py](/Users/jiaoyuan/Documents/GitHub/TD-GradDFT/tests/test_training.py)

审计范围：

- strict-JAX `run_reference_from_spec(...)`
- `self_consistent`
- `ground-state energy loss`
- `excitation loss`
- `spectrum loss`
- `implicit_commutator` 与 `unrolled` 两种 SCF 反传模式

当前 water 审计结论：

- `implicit_commutator`:
  - ground / excitation / spectrum 梯度均全有限
- `unrolled`:
  - ground / excitation / spectrum 梯度在当前 water 配置下也全有限

当前 H2 回归结论：

- strict-JAX `self_consistent + implicit_commutator`
  - ground / excitation / spectrum 梯度全有限且非零

因此，当前仓库中更准确的结论是：

- `implicit_commutator` 是推荐的默认全链路训练路径
- `unrolled` 不是“一定不可微”，而是“配置相关、鲁棒性较弱、不应作为默认稳定路径来承诺”

---

## 7. 当前风险

- strict-JAX molecular grid 目前只实现 `level=0`，且 Lebedev 表仍是有限快照。
- 积分层对更大基组仍有明显首次编译与 warm 执行成本。
- `self_consistent + unrolled` 在当前 water 审计中可微，但其稳健性仍未在更大体系、更多基组和更多训练目标上系统确认，训练默认仍应继续倾向 `implicit_commutator`。
- 局域 HFX 辅助量仍是 PySCF 残留重区，需要单独处理。
- `int1e_rinv/local_hfx` 已 JAX 化，但 `rinv_matrices` 当前通过 `vmap(rinv_matrix)` 生成，首次编译成本仍偏高，需要后续专门优化。

---

## 8. 本轮修改开始点

本轮直接从 **Phase A** 开始：

- 新增 strict-JAX `dipole_integrals`
- 新增 strict-JAX `nuclear_repulsion` helper
- 将 spec-based JAX-RKS reference builder 改成不依赖 PySCF 的实现
- 用水分子与 PySCF 做数值对照
