# 基态 Neural_xc 与 GradDFT 差异对比（2026-04-04）

**范围**：只比较“基态神经网络交换相关泛函 + 可微 SCF 训练核心”，不展开激发态监督细节。  
**对照对象**：

- 本仓当前实现：`src/td_graddft/neural_xc.py`、`src/td_graddft/scf/differentiable.py`、`src/td_graddft/training/targets.py`、`src/td_graddft/workflows/core.py`
- 上游 GradDFT：`/tmp/GradDFT_ref/grad_dft/functional.py`、`/tmp/GradDFT_ref/grad_dft/train.py`、`/tmp/GradDFT_ref/grad_dft/evaluate.py`、`/tmp/GradDFT_ref/README.md`

---

## 1. 结论摘要

当前 TD-GradDFT 的基态 Neural_xc 核心已经与 GradDFT 在以下主线思想上对齐：

- 采用 `E_{xc,\theta}[\rho] = \int c_\theta[\rho](r)\cdot e[\rho](r)\,dr` 的系数-基函数形式；
- 使用 JAX/Flax/Optax 的可微训练主栈；
- 提供 `fixed_density` 与 `self_consistent` 两类训练/求解路径。

但实现形态已明显分叉：

- **GradDFT 更“通用 functional 框架”**（`Functional/NeuralFunctional` + 可插拔梯度通道）；
- **本仓更“TDDFT 绑定式 functional 框架”**（`bind_to_molecule*` 返回可直接驱动 SCF/TDDFT 的 bound 对象）。

这不是错误分叉，而是目标导向差异：本仓为了激发态链路，牺牲了一部分通用性，换取了 response/TDDFT 侧接口统一。

---

## 2. 对齐与差异矩阵

| 维度 | GradDFT | 当前 TD-GradDFT | 结论 |
|---|---|---|---|
| XC 总体表达 | README 明确系数-基函数形式 | `NeuralXCFunctional` 与 `DM21LikeFunctional` 都沿用该形式 | 已对齐 |
| Functional 抽象层 | `Functional/NeuralFunctional`，`compute_densities`、`compute_coefficient_inputs`、`nograd_*`、`*_grads` 可插拔 | 以 `DM21LikeFunctional` 为主，靠 `bind_to_molecule`/`bind_to_molecule_for_scf` 产出 bound XC 对象 | 部分对齐（接口风格不同） |
| DM21 HF 局域特征 | `HF_energy_density(omega=[0,0.4])` + `HF_*_grad_2_Fock` 显式梯度项 | 支持 `dm21_original` 输入、`hfx_local` 通道和 `grid_hfx_feature_gradients_fn`；但 SCF 快路径默认关闭 strict response/hfx 梯度回传 | 部分对齐 |
| SCF 可微路径 | `non_scf_predictor`、`diff_scf_loop`（JIT/fori_loop/DIIS） | `DifferentiableSCF`：`fixed_density` / `self_consistent` + `unrolled` / `implicit_commutator` | 已对齐（策略实现不同） |
| fixed-density 梯度语义 | 非 SCF 预测器语义 | `_single_step` 对 `rdm1` 使用 `stop_gradient` | 已对齐 |
| ground-state loss 形态 | `simple_energy_loss`、`mse_energy_loss`、`mse_density_loss`、`mse_energy_and_density_loss`（较轻量） | `ground_state_mse_loss` 单函数集成能量/密度/势/核/轨道/Janak/SCF 正则等约束 | 本仓更重、更可配置 |
| 严格 GradDFT 基态模式 | 上游原生范式 | `strict_graddft_ground_state` 已提供约束门控 | 已实现兼容层 |
| 默认网络结构 | DM21 默认较宽残差网络（256 宽，多残差层） | 默认 `simple_mlp` (64,64,64)；可切 `graddft_residual` | 默认值不同（可配置对齐） |
| TensorFlow 依赖 | DM21 权重导入函数含 TF 依赖 | 本仓主流程不依赖 TF | 本仓更“纯 JAX” |

---

## 3. 关键差异细化

### 3.1 Functional API 设计哲学

- GradDFT：核心是“通用 local functional 运行时”。
  - 通过 `energy_densities / coefficient_inputs / nograd_* / *_grads / combine_*` 组合梯度路径。
  - 优点是高度可组合，缺点是接口面较大。
- 本仓：核心是“可直接服务 SCF + TDDFT response 的 bound functional”。
  - `bind_to_molecule*` 输出 `grid_potential/grid_kernel/exact_exchange_fraction/...`。
  - 优点是 TDDFT 端接线直接，缺点是与上游通用 API 不完全同构。

### 3.2 HF 成分处理

- GradDFT DM21 通过 `nograd_densities/nograd_coefficient_inputs` 引入 HF 局域特征，再通过显式 `densitygrads/coefficient_input_grads` 把对 Fock 的贡献补回去。
- 本仓既支持 `dm21_original_input_features`（含 `hfx_a/hfx_b`），也支持 enhanced 特征；并区分：
  - `bind_to_molecule`：构造 strict response（含 `grid_response_tensor_fn` 与 `grid_hfx_feature_gradients_fn`）；
  - `bind_to_molecule_for_scf`：为 SCF 速度关闭 strict response/hfx 梯度接口。

### 3.3 SCF 反传策略

- 本仓显式给出两类梯度策略：
  - `unrolled`：直接穿透迭代；
  - `implicit_commutator`：前向用 `stop_gradient(xc_params)` 固定点，再做隐式修正，稳定性更好。
- 这比上游“单一 diff_scf_loop 入口”更工程化，但也增加了配置复杂度。

### 3.4 训练目标复杂度

- 上游基态训练目标相对简洁（能量、密度及其组合）。
- 本仓在同一损失中叠加了更多约束项（势、核、轨道能、Janak、fractional linearity、DM21-SCF 正则等），对研究更灵活，但默认设置更容易“偏离纯 GradDFT 基态”。
- 为控制偏离，本仓提供了 `graddft_core_defaults` 与 `strict_graddft_ground_state` 两级收敛到 GradDFT 语义的开关。

---

## 4. 当前状态判断（基态部分）

按“GradDFT ground-state core + TD extension”原则，当前状态是：

- **核心形式已对齐**：系数-基函数 Neural_xc、可微 SCF、JAX 训练栈。
- **接口层有意分叉**：本仓为 TDDFT 链路优化了 bound functional API。
- **训练目标更重**：需要通过 strict 配置回到上游最小基态语义。

简化结论：当前不是“与 GradDFT 不一致”，而是“GradDFT 核心兼容 + TD 场景扩展”。

---

## 5. 建议的后续整理动作（仅基态）

1. 固化一个 `graddft_core_profile`（代码级 preset）  
   将 `strict_graddft_ground_state` 的隐式组合参数显式化为可复用 profile，避免脚本层重复手工拼配置。

2. 在 `neural_xc.py` 增加“接口映射注释”  
   逐项标注本仓 `bind_to_molecule*` 与上游 `Functional` 插件点的对应关系，降低维护成本。

3. 把 SCF 快路径与 strict 响应路径分离成文档化双轨  
   明确“训练/速度默认轨”与“严格响应/理论对齐轨”的适用场景，减少实验配置误用。

4. 建一个最小对齐回归用例  
   固定一个水分子基态脚本，同时输出：
   - strict profile 配置快照
   - 能量/密度损失曲线
   - 与上游等价目标下的误差统计  
   用作后续重构回归基线。
