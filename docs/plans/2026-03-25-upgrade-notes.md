# TD-GradDFT 架构升级落地说明

**日期**: 2026-03-25  
**关联文档**: `docs/plans/2026-03-25-architecture.md`

## 1. 本次已完成的升级

### 1.1 协议驱动接口（基础设施层）

新增文件：
- `src/td_graddft/protocols.py`

新增协议：
- `XCFunctionalProtocol`
- `BoundXCFunctionalProtocol`
- `MoleculeReferenceProtocol`

目标：
- 统一 XC、分子引用对象的最小行为接口，降低后续模块替换成本。

### 1.2 配置驱动工作流（用户 API 层）

新增文件：
- `src/td_graddft/workflows/config.py`
- `src/td_graddft/workflows/presets.py`

新增能力：
- `SystemConfig` / `ExperimentConfig`
- 多系统实验配置组织
- 基于系统自动生成输出前缀和标题
- 水分子、苯环预设配置（可切基组）

### 1.3 高层 Pipeline 类（用户 API 层）

更新文件：
- `src/td_graddft/workflows/pipeline.py`

新增能力：
- `ExperimentPipeline`
- `ExperimentRun`
- `run_experiment(config)`

目标：
- 对齐架构文档中的 `ExperimentConfig + ExperimentPipeline` 推荐入口。

### 1.4 顶层 API 整理

更新文件：
- `src/td_graddft/workflows/__init__.py`
- `src/td_graddft/__init__.py`

新增导出：
- 实验配置与 pipeline API
- presets API
- 协议接口导出

### 1.5 示例与测试

新增文件：
- `examples/run_workflow_experiment.py`
- `tests/test_workflows_config.py`

验证：
- 新增配置层测试通过
- workflow 入口可直接运行（water smoke）

### 1.6 振子强度归一化修正

更新文件：
- `src/td_graddft/tddft/casida.py`
- `src/td_graddft/tddft/tda.py`

修正内容：
- TDA 与 Casida 振幅归一化改为与 PySCF restricted-spin 一致（alpha block 归一到 0.5），
  消除了振子强度系统性约 2 倍偏差。

### 1.7 TDDFT 子包化重构

新增文件：
- `src/td_graddft/tddft/__init__.py`
- `src/td_graddft/tddft/types.py`
- `src/td_graddft/tddft/response.py`
- `src/td_graddft/tddft/tda.py`
- `src/td_graddft/tddft/casida.py`
- `src/td_graddft/tddft/_utils.py`

迁移结果：
- 原 `src/td_graddft/tddft.py` 已拆分为分模块实现；
- 外部导入路径 `from td_graddft.tddft import ...` 保持兼容。

### 1.8 纯 JAX 基组积分引擎（阶段性）

新增文件：
- `src/td_graddft/data/basis.py`
- `src/td_graddft/data/integrals/_common.py`
- `src/td_graddft/data/integrals/one_electron.py`
- `src/td_graddft/data/integrals/two_electron.py`
- `src/td_graddft/data/integrals/screening.py`

新增能力：
- Cartesian AO 基组容器与 PySCF cart basis 解析器
- 一电子积分（`S`, `T`, `V_nuc`, `H_core`）纯 JAX 实现
- 双电子积分（`(ij|kl)`）纯 JAX 实现
- Schwarz bound 计算
- 针对 PySCF `int1e_*_cart` / `int2e_cart` 的数值对齐测试

当前范围：
- 已支持 `l <= 3`（s/p/d/f）Cartesian AO；
- 与 PySCF `cart=True` 的 libcint `normalized='sp'` 约定对齐（s/p 归一化，d/f 采用 shell-radial norm）；
- `Boys F0` 小 `t` 分支扩展为高阶级数，提高高角动量核吸引积分稳定性；
- 新增 one-electron 单元素接口（`overlap_element/kinetic_element/nuclear_attraction_element`）用于高效抽样校验。

### 1.9 纯 JAX RHF 接入 workflow

更新文件：
- `src/td_graddft/scf/rhf.py`
- `src/td_graddft/pyscf_bridge.py`
- `src/td_graddft/workflows/core.py`
- `src/td_graddft/workflows/types.py`

新增能力：
- 新增 `SimulationConfig.scf_backend`，支持 `"pyscf"` 与 `"jax_rhf"` 切换
- 在 `jax_rhf` 模式下，AO 积分 (`S/T/V/ERI`) 与 RHF 轨道由纯 JAX 计算
- 训练目标能量默认保持对齐 PySCF 参考（例如 B3LYP 总能），便于现有监督流程迁移

说明：
- 参考激发谱仍使用 PySCF TDDFT 输出（用于对比曲线）；
- Neural_xc 激发态分支可切到 pure-JAX RHF 轨道输入，完成“先扩展积分，再接入纯 JAX SCF”的链路闭环。

### 1.10 可微分 SCF 训练模式（fixed_density / self_consistent）

新增文件：
- `src/td_graddft/scf/differentiable.py`
- `tests/test_differentiable_scf.py`

更新文件：
- `src/td_graddft/scf/__init__.py`
- `src/td_graddft/training/config.py`
- `src/td_graddft/training/targets.py`
- `src/td_graddft/training/trainer.py`
- `src/td_graddft/workflows/core.py`
- `src/td_graddft/workflows/types.py`
- `src/td_graddft/pyscf_bridge.py`

新增能力：
- 新增 `DifferentiableSCF`，支持：
  - `fixed_density`：对参考密度做 `stop_gradient`
  - `self_consistent`：在训练图中执行可微 restricted SCF 迭代
- `GroundStateTrainingConfig` 增加 SCF 超参数（循环数、阻尼、收敛阈值）
- `ground_state_mse_loss` / `predict_ground_state_total_energy` 支持按训练模式切换
- workflow 训练配置新增 `training_mode` 与 self-consistent 参数入口
- `RestrictedMoleculeReference` 新增 `overlap_matrix` 字段，支撑 generalized eigenproblem

`fixed_density` 说明：

- `fixed_density` 的含义不是“忽略密度”，而是训练时固定使用参考分子对象中已经给定的
  `rdm1 / mo_coeff / mo_occ / ao / grid.weights` 来构造格点特征与能量项，不在每一步参数更新后
  再执行新的自洽 SCF。
- 在这个模式下，优化目标写成
  \[
  L(\theta; n_{\mathrm{ref}})
  \]
  而不是
  \[
  L(\theta; n_{\theta})
  \]
  其中 \(n_{\mathrm{ref}}\) 是参考泛函或参考计算给出的密度，\(n_{\theta}\) 是当前神经网络泛函
  自己自洽得到的密度。
- 这类训练在工程上是合理的，因为它把“泛函参数误差”和“SCF 固定点误差”分开处理：
  训练图只需要对固定参考密度上的 XC 能量、轨道相关约束或响应约束求导，因此更稳定、更快。
- 它本质上对应一种 post-SCF / orbital-frozen 拟合近似。训练过程中被监督的是参考密度附近的
  泛函行为，而不是神经网络泛函完全自洽后的变分解。
- 这个近似成立的前提是：参考密度与目标泛函产生的自洽密度差异不能过大。若两者偏差显著，
  则总能量、轨道能、响应核和激发态之间可能出现不一致。
- 因此，`fixed_density` 更适合解释为“在参考密度上拟合泛函”，它对基态能量监督通常是有效的，
  但并不等价于对自洽轨道、化学势或 TDDFT 激发态的完整监督。

### 1.11 开壳层数据桥接与 TDA（OH / B3LYP/6-31G）

新增文件：
- `src/td_graddft/tddft/unrestricted.py`
- `tests/test_unrestricted_open_shell.py`
- `examples/run_oh_open_shell_tda.py`

更新文件：
- `src/td_graddft/pyscf_bridge.py`
- `src/td_graddft/spectra.py`
- `src/td_graddft/tddft/__init__.py`
- `src/td_graddft/__init__.py`

新增能力：
- 新增 `UnrestrictedMoleculeReference` 与 `unrestricted_reference_from_pyscf`
- 新增 `UnrestrictedTDA` 与 spin-block 响应矩阵构建/求解（纯 JAX 线性代数）
- 振子强度计算扩展到开壳层振幅（alpha/beta 通道）
- 新增 OH 自由基 `B3LYP/6-31G` 测试：
  - 开壳层参考桥接与 TDA smoke
  - d 壳层积分样本与 PySCF 对齐

### 1.12 开壳层 Casida/TDDFT（A/B）最小可用实现

新增/更新文件：
- `src/td_graddft/tddft/unrestricted.py`
- `src/td_graddft/tddft/__init__.py`
- `src/td_graddft/spectra.py`
- `src/td_graddft/__init__.py`
- `tests/test_unrestricted_casida.py`

新增能力：
- 新增 `UnrestrictedResponseMatrices`（A/B 全块）
- 新增 `solve_unrestricted_casida` 与 `UnrestrictedCasidaTDDFT`
- 新增开壳层 `gen_tdhf_vind` 接口（与 restricted 风格一致）
- 振子强度计算支持 `UnrestrictedTDDFTResult`（使用 `(X+Y)` 的 alpha/beta 通道）
- 新增 OH 开壳层 Casida smoke 测试与 `vind` 维度测试

### 1.13 纯 JAX RKS 后端（workflow 可选）

新增/更新文件：
- `src/td_graddft/scf/rks.py`
- `src/td_graddft/scf/__init__.py`
- `src/td_graddft/pyscf_bridge.py`
- `src/td_graddft/workflows/core.py`
- `src/td_graddft/workflows/types.py`
- `tests/test_pyscf_bridge_jax_rks.py`

新增能力：
- 新增 `RKSConfig` / `RKSResult` / `run_rks_from_integrals`
- 新增 `restricted_reference_from_pyscf_with_jax_rks`
- workflow `SimulationConfig.scf_backend` 新增 `"jax_rks"` 路径
- 支持从 PySCF XC 名称推断到当前 jax_libxc 可用表达式（含 B3LYP 近似表达）
- 增加 jax_rks backend 的桥接与 workflow 回归测试

### 1.14 开壳层自旋分辨 f_xc 内核接口（`f_xc^{σσ'}`）

新增/更新文件：
- `src/td_graddft/xc.py`
- `src/td_graddft/tddft/unrestricted.py`
- `tests/test_unrestricted_spin_kernel.py`

新增能力：
- `AdiabaticDensityFunctional` 增加 `spin_local_potential` / `spin_local_kernel` 接口
- 开壳层响应矩阵构建支持 `f_aa / f_ab / f_bb` 三通道注入
- 兼容多种 kernel 形态：
  - 标量 `f_xc(r)`（自动广播到三通道）
  - `(f_aa, f_ab, f_bb)` 元组
  - `(ngrids, 3)` 或 `(ngrids, 2, 2)` 表示
- 新增单测覆盖：
  - 自旋分辨 kernel 被正确写入 `aa/ab/bb` 响应块
  - 标量 kernel 回退路径保持旧行为

### 1.15 workflow 开壳层路径（`jax_uks + UnrestrictedCasidaTDDFT`）

新增/更新文件：
- `src/td_graddft/workflows/types.py`
- `src/td_graddft/workflows/core.py`
- `tests/test_workflow_open_shell.py`

新增能力：
- `SimulationConfig` 新增 `jax_uks_*` 参数组
- workflow 参考构建新增 `scf_backend="jax_uks"` 路径
- `run_reference` 支持 unrestricted 维度推断（`nocc_a*nvir_a + nocc_b*nvir_b`）
- `run_neural_tddft` 自动在 restricted/unrestricted 间切换：
  - restricted: `RestrictedCasidaTDDFT`
  - unrestricted: `UnrestrictedCasidaTDDFT`

### 1.16 测试分层（先闭壳层，后开壳层）

新增/更新文件：
- `tests/test_pyscf_bridge_jax_uks.py`
- `tests/test_unrestricted_open_shell.py`
- `tests/test_unrestricted_casida.py`
- `tests/test_workflow_open_shell.py`

新增能力：
- 开壳层测试默认关闭，避免阻塞闭壳层主线迭代速度
- 通过环境变量显式开启：
  - `TD_GRADDFT_RUN_OPEN_SHELL_TESTS=1 pytest -q ...`

### 1.17 jax_xc 引入（vendoring + 兼容适配）

新增/更新文件：
- `third_party/jax_xc/`（上游仓库镜像）
- `src/td_graddft/jax_xc_adapter.py`
- `src/td_graddft/upstreams.py`
- `src/td_graddft/xc.py`
- `src/td_graddft/__init__.py`
- `tests/test_jax_xc_adapter.py`

新增能力：
- `jax_xc` 加载策略升级为：
  - 优先加载上游 `jax_xc`
  - 不可用时自动回退到本地 `jax_libxc` 兼容层
- `lda_from_jax_xc` 改为统一走 adapter，不再硬绑定系统环境
- `has_jax_xc()` 反映“可用后端”而非仅系统 pip 包

### 1.18 双电子积分优化（闭壳层优先）

新增/更新文件：
- `src/td_graddft/data/integrals/two_electron.py`
- `src/td_graddft/data/integrals/__init__.py`
- `src/td_graddft/__init__.py`
- `tests/test_integrals_jax.py`

新增能力：
- `eri_tensor` 接入 8 重对称填充（减少重复 quartet 计算）
- 新增 `eri_tensor_screened(...)`，支持 Schwarz 阈值筛选
- 新增积分筛选回归测试：
  - 阈值 0 与未筛选一致
  - 超大阈值可全部跳过

### 1.19 独立 tools 包：可微几何优化 + 频率分析（基态/激发态通用）

新增/更新文件：
- `src/td_graddft_tools/__init__.py`
- `src/td_graddft_tools/README.md`
- `src/td_graddft_tools/geomopt_freq/__init__.py`
- `src/td_graddft_tools/geomopt_freq/objectives.py`
- `src/td_graddft_tools/geomopt_freq/optimization.py`
- `src/td_graddft_tools/geomopt_freq/frequencies.py`
- `src/td_graddft_tools/geomopt_freq/workflow.py`
- `tests/test_geomopt_freq_tools.py`

新增能力：
- 在 `td_graddft` 核心包之外新增独立 `tools` 子包（但位于 `src/` 内可直接导入）
- 面向可微分工作流的模块化分层：
  - `objectives`: 基态/激发态能量面构建
  - `optimization`: 基于 `jax.value_and_grad` + Optax 的几何优化
  - `frequencies`: Hessian、质量加权、谐振频率与模向量
  - `workflow`: 一键 optimize + frequency 管道
- 同一 API 同时适用于：
  - 基态能量面 `E0(R)`
  - 激发态能量面 `E_exc(R)=E0(R)+ω_k(R)`
- 参考 `pyscfad` 的结构优化思路，对优化与振动后处理职责进行解耦

---

## 2. 与目标架构的对齐情况

已对齐：
- 协议驱动（核心接口）
- 配置驱动（实验级）
- workflow 高层入口
- 训练/光谱输出流程化组织
- TDDFT 模块化拆分（response/casida/tda）
- 纯 JAX 积分引擎（s/p/d/f）
- 可微分 SCF 训练模式（fixed_density / self_consistent）
- 开壳层桥接 + TDA 最小可用链路
- 开壳层 Casida/TDDFT（A/B）最小可用链路
- 纯 JAX RKS backend 最小可用链路
- 纯 JAX UKS backend 最小可用链路
- 开壳层自旋分辨 `f_xc^{σσ'}` 响应接口

未完全对齐（后续阶段）：
- 纯 JAX Kohn-Sham 全链路（当前网格 AO 与参考谱仍部分依赖 PySCF）
- 自旋极化 semilocal（LDA/GGA）能量密度本体（当前 jax_libxc 仍以 spin-summed 近似为主）
- workflow 端到端开壳层训练/激发态管线（当前以 restricted 主流程为主）

---

## 3. 建议下一步（按优先级）

1. 为 `jax_libxc` 增加自旋极化 LDA/GGA（`rho_a/rho_b`）能量密度与势/核。  
2. 在 workflow 层打通开壳层主流程（`jax_uks + UnrestrictedCasidaTDDFT`）。  
3. 继续替换 PySCF 依赖（网格、参考谱）以实现更完整的 pure-JAX 链路。  
