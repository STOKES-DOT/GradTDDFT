# 通用 `(α, β, ω)` RSH 框架接入 TD-GradDFT 执行方案

**日期**: 2026-04-16  
**状态**: 讨论稿，可直接按阶段执行  
**目标**: 在现有 TD-GradDFT 可微 SCF/训练框架中接入通用可训练 `(α, β, ω)` range-separated hybrid (RSH) ground-state functional，并优先支持基于 Janak 定理与分数电子线性的自洽优化。

---

## 1. 结论先行

本项目不应复刻 `OPTXC` 这类“先离线调 `ω_OT`，再训练 surrogate 预测 `ω`”的路线。  
TD-GradDFT 当前已经具备更强的基础条件：

- 已有可微自洽 SCF。
- 已有 `janak_frontier_finite_difference_penalty(...)`。
- 已有 `fractional_charge_linearity_penalty(...)`。
- 已有多 `ω` 的 local-HF / `hfx_nu` 数据通道。

因此推荐路线是：

1. 在 ground-state functional 层引入通用可训练 `(α, β, ω)` RSH 对象。
2. 在 SCF Fock 构造中原生支持 RSH 非局域 HF 项。
3. 直接用自洽 Janak / 分数电子 / frontier 相关目标反传优化参数。
4. 待地面态稳定后，再决定是否把该 RSH functional 进一步接入 TD 响应层。

---

## 2. 当前仓库状态与约束

### 2.1 已经存在、可直接复用的部分

- Janak 前沿轨道有限差分约束：
  - `src/td_graddft/training/targets.py`
- 分数电子线性约束：
  - `src/td_graddft/training/targets.py`
- ground-state 自洽训练配置：
  - `src/td_graddft/training/config.py`
- 多 `ω` 的 `hfx_local` / `hfx_nu` 参考构造：
  - `src/td_graddft/reference.py`
- 响应层 long-range wrapper，但它当前是 response-only，不改 ground-state：
  - `src/td_graddft/tddft/long_range_correction.py`

### 2.2 当前真正缺失的部分

- SCF 主路径目前本质上只支持标量全局 hybrid `alpha`。
- 还没有 ground-state 级别的通用 RSH functional 抽象。
- 还没有把 `(α, β, ω)` 直接接到 Fock builder。
- 当前自写 `jax_libxc` 只覆盖少量 LDA/GGA/hybrid alias，不足以表达标准 RSH 的短程/长程 semilocal 分拆。

### 2.3 一个关键可行性判断

标准 RSH 的 HF 部分可写成：

```text
α K_SR(ω) + (α + β) K_LR(ω)
= α K_full + β K_LR(ω)
```

其中：

- `K_SR(ω) = K_full - K_LR(ω)`
- `K_LR(ω)` 由 `erf(ω r12) / r12` 构造

这个重写非常重要，因为它意味着 SCF 侧只需要支持：

- 一个全局 full-range HF 系数 `α`
- 若干个带 `ω` 的 long-range HF 通道系数 `β_i`

换句话说，SCF Fock 侧的最小泛化是可控的，不需要一开始就把整个 SCF 重写成完全新框架。

---

## 3. 一阶段范围

### 3.1 一阶段必须完成

- restricted closed-shell ground-state `(α, β, ω)` RSH functional
- self-consistent SCF 可微训练
- Janak frontier loss
- fractional-charge linearity loss
- coefficient prior / parameter bound
- 至少一个标准 RSH 家族模板可用
  - 首选：`LC-ωPBE`

### 3.2 一阶段明确不做

- unrestricted / open-shell 全覆盖
- TDDFT/Casida 响应层同步改造
- `ωB97X-D` 里的经验色散完整还原
- surrogate ML 预测 `ω`
- 大规模 benchmark 自动化

---

## 4. 核心设计决策

### 4.1 参数化方式

为了避免训练时进入非法参数区，同时避免和 `PySCF` 的 `rsh=(omega, alpha, beta)` 记号冲突，内部建议不要把论文记号 `(α, β, ω)` 直接作为主 API。

内部主语义应统一为：

- `sr_hf_fraction = c_SR`
- `lr_hf_fraction = c_LR`
- `omega`

其中论文常用记号可视为：

```text
paper_alpha = c_SR
paper_beta  = c_LR - c_SR
```

而 `PySCF` 的 `rsh=(omega, alpha, beta)` 语义是：

```text
pyscf_alpha = c_LR
pyscf_beta  = c_SR - c_LR
```

这一步很关键，因为如果内部直接把字段命名为 `alpha/beta`，后续导出到 `PySCF` 很容易把符号意义写反。

对于常见“长程 HF 不小于短程 HF”的家族，一阶段推荐参数化为：

```text
sr_hf = sigmoid(sr_hf_raw)
lr_hf = sr_hf + (1 - sr_hf) * sigmoid(lr_delta_raw)
omega = omega_min + (omega_max - omega_min) * sigmoid(omega_raw)
```

性质：

- `0 <= sr_hf <= 1`
- `0 <= lr_hf <= 1`
- `lr_hf >= sr_hf`
- `omega` 自动限制在稳定区间

默认建议：

- `omega_min = 0.05 bohr^-1`
- `omega_max = 0.80 bohr^-1`

### 4.2 局域 semilocal 部分的实现口

这部分不能继续只靠当前 `jax_libxc` 的简化 alias。

推荐优先级：

1. 首选：扩展 `jax_xc_adapter`，使用 vendored `third_party/jax_xc` 中带 `cam_omega`/hybrid metadata 的实现，正确评估 RSH 对应的局域 semilocal 部分。
2. 次选：若 `jax_xc` 对目标 functional 的接口不够稳定，则新增一个独立的 `rsh_local_backend` 适配层，只先支持 `LC-ωPBE`。
3. 禁止：用“普通 PBE + 手工加 LR-HF”去冒充标准 `LC-ωPBE`，除非明确标注为近似实验路径。

### 4.3 SCF 接口的最小泛化

当前 `DifferentiableSCF` 只认：

- `v_rho`
- `v_grad`
- `xc_kind`
- `alpha`

需要升级为显式支持非局域 HF 通道。推荐新增统一 dataclass：

```python
@dataclass(frozen=True)
class SCFXCContributions:
    v_rho: Array
    v_grad: Array
    xc_kind: str
    full_hf_fraction: Array
    lr_hf_omegas: Array | None = None
    lr_hf_coefficients: Array | None = None
    extra_fock_matrix: Array | None = None
    resolved_xc: Any | None = None
```

SCF Fock 组装目标形式：

```text
F = h + J
  - 0.5 * α * K_full
  - 0.5 * Σ_i β_i * K_LR(ω_i)
  + V_xc_local
  + V_extra
```

这样可以：

- 完整兼容现有 global hybrid
- 兼容单 `ω` 的标准 RSH
- 兼容未来多 `ω` 或离散 `ω` 混合实验

### 4.4 连续 `ω` 与离散 `ω` 的关系

推荐主路线：连续 `ω`。

原因：

- 仓库已有 `rinv_matrices(..., zeta=omega^2)` 的 JAX 路径。
- Janak / 分数电子自洽调参的核心价值就在于直接优化 `ω`，不是做格点插值。

过渡路线允许保留：

- 用已有 `hfx_omega_values` + `hfx_nu` 先做离散 `ω` 通道混合
- 只作为 smoke / fallback，不作为主方案

### 4.5 通用 RSH API 草案

推荐把 API 分成四层：

1. **模板层**
   描述“这个 functional 家族是什么”，不含具体 trainable 参数值。
2. **参数层**
   描述 `(α, β, ω)` 的受约束训练参数与解析后的物理参数。
3. **bound functional 层**
   把 functional 与具体 molecule/grid 绑定，产出可直接用于 SCF/训练的对象。
4. **SCF contributions 层**
   明确告诉 SCF 需要哪些局域势、哪些 HF 通道、以及是否有额外 Fock 修正。

推荐 API 如下。

#### 4.5.1 模板层

```python
@dataclass(frozen=True)
class RSHFunctionalTemplate:
    name: str
    local_backend: str
    exchange_backend_id: str
    correlation_backend_id: str
    supports_trainable_sr_hf: bool = True
    supports_trainable_lr_hf: bool = True
    supports_trainable_omega: bool = True
    has_dispersion: bool = False
    monotonic_lr_hf: bool = True
    default_sr_hf_fraction: float = 0.0
    default_lr_hf_fraction: float = 1.0
    default_omega: float = 0.30
    omega_bounds: tuple[float, float] = (0.05, 0.80)
    sr_hf_bounds: tuple[float, float] = (0.0, 1.0)
    lr_hf_bounds: tuple[float, float] = (0.0, 1.0)
```

职责：

- 定义 functional family 的身份与默认参数
- 指向 local semilocal backend
- 指定哪些参数可训练
- 规定参数边界

示例：

- `LC-ωPBE`
- `CAM-B3LYP`
- `ωB97X`

#### 4.5.2 参数层

```python
@dataclass(frozen=True)
class RSHParameterBounds:
    sr_hf_bounds: tuple[float, float] = (0.0, 1.0)
    lr_hf_bounds: tuple[float, float] = (0.0, 1.0)
    omega_bounds: tuple[float, float] = (0.05, 0.80)


@dataclass(frozen=True)
class ResolvedRSHParameters:
    sr_hf_fraction: Array
    lr_hf_fraction: Array
    omega: Array

    @property
    def paper_alpha(self) -> Array: ...

    @property
    def paper_beta(self) -> Array: ...

    def to_pyscf_rsh(self) -> tuple[Array, Array, Array]: ...


@dataclass(frozen=True)
class TrainableRSHParametrization:
    bounds: RSHParameterBounds
    monotonic_lr_hf: bool = True

    def init(
        self,
        *,
        sr_hf_fraction: float,
        lr_hf_fraction: float,
        omega: float,
    ) -> PyTree: ...

    def resolve(self, params: PyTree) -> ResolvedRSHParameters: ...
```

职责：

- 存储 raw 参数
- 统一做合法区间映射
- 内部统一输出 `sr_hf_fraction / lr_hf_fraction / omega`
- 同时提供：
  - 论文记号视图
  - `PySCF` 记号视图

这样日志、训练、导出都不需要再靠临时换元。

#### 4.5.3 SCF contributions 层

```python
@dataclass(frozen=True)
class SCFXCContributions:
    v_rho: Array
    v_grad: Array
    xc_kind: str
    full_hf_fraction: Array
    lr_hf_omegas: Array | None = None
    lr_hf_coefficients: Array | None = None
    extra_fock_matrix: Array | None = None
    exact_exchange_fraction: Array | None = None
    resolved_xc: Any | None = None
```

字段语义：

- `full_hf_fraction`
  full-range `K_full` 的系数，即 `c_SR`
- `lr_hf_omegas`
  long-range HF 通道的 `ω`
- `lr_hf_coefficients`
  对应 each `ω` 的附加 long-range 系数，标准单通道情形就是
  `[c_LR - c_SR]`
- `exact_exchange_fraction`
  兼容旧路径，等价于 `full_hf_fraction`
- `extra_fock_matrix`
  保留给 DM21/额外投影修正等非标准项

要求：

- `lr_hf_omegas is None` 与 `lr_hf_coefficients is None` 必须同时为空
- 若不为空，两者长度必须相同
- 当前一阶段默认只支持标量或一维通道数组

#### 4.5.4 bound functional 层

```python
@dataclass(frozen=True)
class BoundRSHFunctional:
    template: RSHFunctionalTemplate
    resolved_params: ResolvedRSHParameters
    response_feature_kind: str
    exact_exchange_fraction: Array

    def grid_potential_components(
        self,
        molecule: Any,
    ) -> tuple[Array, Array, Array | None]: ...

    def local_potential(self, density: Array) -> Array: ...

    def local_kernel(self, density: Array) -> Array: ...

    def scf_contributions(self, molecule: Any) -> SCFXCContributions: ...
```

职责：

- 对具体 `molecule` 评估 local semilocal potential
- 暴露 SCF 所需的完整贡献
- 兼容现有 training/response 代码对 `exact_exchange_fraction`、`local_potential` 的读取习惯

其中最关键的方法是：

```python
def scf_contributions(self, molecule: Any) -> SCFXCContributions
```

SCF 后续应优先读这个方法，而不是继续手工拆 tuple。

#### 4.5.5 PySCF 导出层

训练好的模型如果要直接接入 `PySCF`，推荐把这件事做成正式 API，而不是让脚本手工拼接。

```python
@dataclass(frozen=True)
class PySCFRSHSpec:
    xc_description: str | Callable[..., Any]
    xctype: str
    hyb: float
    rsh: tuple[float, float, float]


class BoundRSHFunctional:
    ...

    def to_pyscf_spec(self) -> PySCFRSHSpec: ...

    def install_into_pyscf(self, mf: Any) -> Any: ...
```

推荐语义：

- `hyb = sr_hf_fraction`
- `rsh = (omega, lr_hf_fraction, sr_hf_fraction - lr_hf_fraction)`
- `rsh_and_hybrid = (omega, lr_hf_fraction, sr_hf_fraction)`

原因：

- `PySCF` 的 `rsh=(omega, alpha, beta)` 语义是：
  - `alpha = c_LR`
  - `beta = c_SR - c_LR`
- 这和论文里的 `(α, β, ω)` 记号不是同一套定义

实现注意：

- `pyscf.dft.libxc.define_xc_` 在 `description` 是原始字符串时，会重新从该字符串解析 hybrid/RSH 系数。
- 因此只要目标是“导出任意训练后的参数”，就不能依赖 `description + hyb + rsh` 的字符串路径。
- 正确做法是把 local XC 部分包装成 callable `eval_xc(...)`，同时把显式 `hyb/rsh` 交给 `define_xc_`。
- 只有在刻意复用 `PySCF` 内建 functional 默认参数时，纯字符串 symbolic export 才是稳妥的。

#### 4.5.6 PySCF 接入的两种模式

建议区分下面两种模式：

1. **symbolic export**
   只适用于复用 `PySCF` 内建 functional 默认参数的场景。
   直接导出 `xc_description` 字符串给 `pyscf.dft.libxc.define_xc_`。

2. **callable local export**
   适用于 trainable RSH 参数，或未来 local 部分不再是标准 libxc 组合的情形。
   通过自定义 `eval_xc(...)` 提供局域部分，同时仍让 `PySCF` 根据 `hyb/rsh` 处理 HF 部分。

一阶段优先做 `callable local export`，因为这是支持任意训练后参数的必要条件。

#### 4.5.7 trainable functional 层

```python
@dataclass(frozen=True)
class TrainableRSHFunctional:
    template: RSHFunctionalTemplate
    parametrization: TrainableRSHParametrization

    def init(self, rng: PRNGKeyArray, molecule: Any) -> PyTree: ...

    def resolve_parameters(self, params: PyTree) -> ResolvedRSHParameters: ...

    def bind_to_molecule(self, params: PyTree, molecule: Any) -> BoundRSHFunctional: ...

    def bind_to_molecule_for_scf(
        self,
        params: PyTree,
        molecule: Any,
    ) -> BoundRSHFunctional: ...

    def energy_from_molecule(self, params: PyTree, molecule: Any) -> Array: ...
```

职责：

- 管理 trainable params
- 负责把 params 解析成物理参数
- 生成 `BoundRSHFunctional`
- 对 training targets 暴露 `energy_from_molecule(...)`

#### 4.5.8 兼容策略

为了不同时重写所有 functional，SCF 侧建议按下面顺序探测：

1. 若 `resolved_xc` 暴露 `scf_contributions(...)`，优先使用
2. 否则回退到现有：
   - `grid_potential_components(...)`
   - `exact_exchange_fraction`
   - `extra_fock_matrix` 或 DM21 修正
3. 旧 tuple 返回逻辑只作为兼容层保留

这样旧功能不需要一次性迁移完。

#### 4.5.9 一阶段 API 冻结建议

一阶段先冻结以下名字，不再轻易变动：

- `RSHFunctionalTemplate`
- `ResolvedRSHParameters`
- `SCFXCContributions`
- `PySCFRSHSpec`
- `BoundRSHFunctional`
- `TrainableRSHFunctional`
- `scf_contributions(...)`
- `to_pyscf_spec(...)`

后续 family 扩展只新增模板和 backend，不改主协议。

---

## 5. 推荐代码落点

### 5.1 新增模块

- `src/td_graddft/nn_rsh/`
  - RSH 专用命名空间，和通用 `dft/`、`training/` 逻辑解耦
  - 当前建议子模块：
    - `schema.py`
      - `RSHFunctionalTemplate`
      - `ResolvedRSHParameters`
      - `SCFXCContributions`
      - `PySCFRSHSpec`
    - `functional.py`
      - `TrainableRSHFunctional`
      - `BoundRSHFunctional`
      - descriptor / parameter head / PySCF bridge
    - `descriptors.py`
      - atom-centered density power-spectrum descriptor
      - BP-style molecule input builder
    - `losses.py`
      - `make_self_supervised_rsh_loss(...)`
      - Janak / fractional / prior 组合
  - `dft/rsh.py`、`dft/trainable_rsh.py` 只保留兼容导出层

- `src/td_graddft/nn_rsh/rsh_backends.py`
  - `jax_xc` 适配
  - functional metadata 解析
  - local semilocal energy density / potential / kernel 接口

- `tools/tune_general_rsh_ground_state.py`
  - 最小训练入口
  - 支持 Janak / fractional / prior 组合

- `tools/overfit_water_nn_rsh.py`
  - 单水分子无标签自洽训练脚本
  - 使用 atom-centered density descriptor
  - 输出 Janak MAE 与 `(sr_hf_fraction, lr_hf_fraction, omega)` 轨迹

### 5.2 需要修改的现有模块

- `src/td_graddft/scf/differentiable.py`
  - 从 `alpha` 扩展到 `SCFXCContributions`
  - Fock 组装支持 `K_LR(ω)`

- `src/td_graddft/scf/rks.py`
  - strict-JAX RKS 路径对齐同样的 HF 通道语义

- `src/td_graddft/reference.py`
  - 补齐 continuous `ω` 下的 LR exchange cache 复用接口

- `src/td_graddft/training/targets.py`
  - Janak / fractional 路径原则上无需改公式
  - 只需保证 `energy_from_molecule(...)` 与 SCF 路径能调用新 functional

- `src/td_graddft/__init__.py`
  - 暴露新 functional 与模板

---

## 6. 分阶段执行

### Phase 0: 方案冻结与接口勘测

**目标**: 在不写大量业务代码前，把接口边界定死。

**任务**

- [ ] 确认 `third_party/jax_xc` 中 `LC-ωPBE` / `ωB97X` / CAM 类 functional 的可调用接口与 metadata。
- [ ] 明确 `BoundRSHFunctional` 对外最小方法集：
  - `energy_from_molecule(...)`
  - `grid_potential_components(...)`
  - `scf_contributions(...)`
- [ ] 明确 SCF 侧只认 `SCFXCContributions`，旧接口通过兼容层映射。

**完成标准**

- 形成固定接口签名，不再在实现中临时改协议。

### Phase 1: SCF 非局域 HF 通道泛化

**目标**: 让 SCF 原生支持 `α K_full + β K_LR(ω)`。

**任务**

- [ ] 引入 `SCFXCContributions`。
- [ ] 将 `_scf_xc_components(...)` 改为统一返回该 dataclass。
- [ ] 在 `DifferentiableSCF` 中加入 `K_LR(ω)` 构造。
- [ ] 兼容旧 functional：
  - 无 LR 通道时退化为当前逻辑
  - 现有测试不回归

**完成标准**

- 现有 LDA/PBE/PBE0/B3LYP 路径保持通过。
- 单元测试中可构造“只加 LR-HF 通道”的 toy functional。

### Phase 2: `LC-ωPBE` 模板接入

**目标**: 第一类标准 RSH functional 落地。

**任务**

- [ ] 实现 `TrainableRSHParameters` 与受约束投影。
- [ ] 实现 `RSHFunctionalTemplate`。
- [ ] 接入 `LC-ωPBE` 的 local backend。
- [ ] 实现 `BoundRSHFunctional.energy_from_molecule(...)`。
- [ ] 实现 `BoundRSHFunctional.grid_potential_components(...)`。

**完成标准**

- fixed-parameter `LC-ωPBE` 可以跑通 restricted self-consistent SCF。
- 参数固定时，能给出稳定总能与轨道能。
- 默认参数下能和 `PySCF` 参考 functional 做一次系数级对照：
  - `hybrid_coeff`
  - `rsh_coeff`
  - `rsh_and_hybrid_coeff`

### Phase 3: Janak / 分数电子训练回路

**目标**: 用现有 training target 直接优化 `(α, β, ω)`。

**任务**

- [ ] 新增 ground-state tuning 工具脚本。
- [ ] 支持配置：
  - `janak_frontier_constraint_weight`
  - `fractional_linearity_weight`
  - `coefficient_prior_weight`
  - self-consistent mode
- [ ] 输出：
  - learned `(α, β, ω)`
  - frontier residual
  - fractional curvature
  - 训练曲线

**完成标准**

- 在小体系 smoke set 上，Janak residual 有可见下降。
- fractional linearity curvature 有可见下降。
- SCF 训练中梯度有限，参数不会频繁撞边界。

### Phase 4: 扩展到通用 RSH 家族

**目标**: 从单模板扩到通用模板。

**候选**

- [ ] `ωB97X`
- [ ] `CAM-B3LYP`
- [ ] 其他 `jax_xc` 已支持的 CAM/RSH functional

**完成标准**

- functional family 切换只改模板，不改 SCF 主干。

---

## 7. 风险与规避

### 风险 1: local RSH semilocal 部分实现不完整

这是首要风险。

如果 `jax_xc` 不能稳定提供目标 functional 的局域部分与导数：

- 一阶段只承诺 `LC-ωPBE`
- 不宣称“通用 family 已可用”
- 禁止偷偷退回近似 PBE 替代

### 风险 2: continuous `ω` 下 LR exchange 构造太慢

规避方式：

- 先做小体系 restricted smoke
- 优先复用 `rinv_matrices(..., zeta=omega^2)` 路径
- 保留离散 `ω` cache fallback 作为 benchmark 对照

### 风险 3: 自洽训练不稳定

规避方式：

- 一开始固定 `α`，只调 `(β, ω)` 做 warm start
- 或固定 `ω`，只调 `(α, β)` 做 warm start
- 默认启用 parameter prior
- 默认保守 `omega` 范围

### 风险 4: `ωB97X-D` 的 `-D` 色散项不在当前框架中

规避方式：

- 一阶段只做 `ωB97X` 主体，不把 `-D` 当完成标准
- 如需 `-D`，单独作为附加模块处理

---

## 8. 一阶段验收测试

### 8.1 单元测试

- [ ] `tests/test_rsh_parameterization.py`
  - 参数投影满足 `0 <= alpha <= alpha+beta <= 1`
  - `omega` 始终在配置范围

- [ ] `tests/test_rsh_scf_contributions.py`
  - 旧 functional 走兼容层不回归
  - LR 通道为空时等价当前逻辑

- [ ] `tests/test_rsh_lr_exchange_matrix.py`
  - `ω -> 0` 时 LR exchange 极限行为合理
  - 系数为零时不贡献 Fock

- [ ] `tests/test_rsh_janak_training_smoke.py`
  - toy molecule 上一步或数步训练后 Janak loss 降低

### 8.2 集成 smoke

- [ ] `H2 / STO-3G`
- [ ] `H2 / 6-31G`
- [ ] `H2O / STO-3G`

建议记录：

- SCF 是否收敛
- `HOMO/LUMO` residual
- fractional curvature
- 参数轨迹 `(α, β, ω)`

---

## 9. 第一批 PR 切分建议

### PR 1: SCF 接口泛化

- `SCFXCContributions`
- `DifferentiableSCF` 兼容层
- 不引入真正的 RSH functional

### PR 2: `LC-ωPBE` functional 最小落地

- `dft/rsh.py`
- `dft/rsh_backends.py`
- fixed-parameter SCF smoke

### PR 3: Janak / fractional tuning 工具

- CLI 脚本
- smoke tests
- 输出与日志

### PR 4: 扩展到通用模板

- `CAM-B3LYP`
- `ωB97X`
- family registry

---

## 10. 文件级改动清单

下面的清单是按“先能编译、再能跑 smoke、最后再补 family 扩展”的顺序组织的。

### 10.1 PR 1 需要新增

- `src/td_graddft/dft/rsh.py`
  - `RSHFunctionalTemplate`
  - `RSHParameterBounds`
  - `ResolvedRSHParameters`
  - `SCFXCContributions`
  - `PySCFRSHSpec`
  - 参数换元/导出 helper

### 10.2 PR 1 需要修改

- `src/td_graddft/protocols.py`
  - 补 `BoundRSHFunctionalProtocol`
  - 补 `TrainableRSHFunctionalProtocol`
  - 补 `scf_contributions(...)` 协议

- `src/td_graddft/scf/differentiable.py`
  - 优先探测 `resolved_xc.scf_contributions(...)`
  - 在 legacy path 缺省时回退到旧逻辑
  - 当前 PR 先支持：
    - `full_hf_fraction`
    - `lr_hf_omegas`
    - `lr_hf_coefficients`
  - 允许 `extra_fock_matrix`

- `src/td_graddft/__init__.py`
  - 暴露上面新增 dataclass / helper

### 10.3 PR 1 测试

- `tests/test_rsh_api.py`
  - `ResolvedRSHParameters` 的论文视图 / PySCF 视图
  - `PySCFRSHSpec` 的构造
  - `SCFXCContributions` 的合法性检查

- `tests/test_scf_contributions_bridge.py`
  - fake functional 通过 `scf_contributions(...)` 接入 `DifferentiableSCF`
  - legacy fake functional 仍然走旧路径

### 10.4 PR 2 需要新增

- `src/td_graddft/dft/rsh_backends.py`
  - `jax_xc` local backend 适配层
  - `LC-ωPBE` 模板 helper

- `tools/tune_general_rsh_ground_state.py`
  - 最小训练入口
  - 保存 learned params
  - 输出 Janak / fractional 指标

### 10.5 PR 2 需要修改

- `src/td_graddft/scf/rks.py`
  - strict-JAX RKS 语义对齐

- `src/td_graddft/reference.py`
  - continuous `ω` 的 LR exchange cache 复用

- `src/td_graddft/training/targets.py`
  - 不改公式
  - 只确保新 functional 可直接走 `energy_from_molecule(...)`

---

## 11. Immediate Next Steps

这是建议立刻执行的开发顺序。

### Step 1: 固化协议

- [ ] 在代码层新增 `SCFXCContributions`
- [ ] 在代码层新增 `ResolvedRSHParameters`
- [ ] 在代码层新增 `PySCFRSHSpec`
- [ ] 在 `protocols.py` 中把 `scf_contributions(...)` 变成显式协议

### Step 2: 先让 SCF 读新协议

- [ ] `DifferentiableSCF` 优先从 `resolved_xc.scf_contributions(...)` 取值
- [ ] 若没有新协议，则自动走旧 tuple 逻辑
- [ ] 对现有 test suite 不引入行为变化

### Step 3: 再做最小 RSH functional

- [ ] 实现一个最小但不完整 family-agnostic 的 `TrainableRSHFunctional`
- [ ] 先只提供参数解析、导出、PySCF spec 生成
- [ ] 不在这一步承诺 local semilocal backend 完整可用
- [ ] 固化一条“默认参数 vs `PySCF` built-in”的对照测试路径

### Step 4: 再接 `LC-ωPBE`

- [ ] 接 `jax_xc` local backend
- [ ] 跑 fixed-parameter smoke
- [ ] 再跑 self-consistent Janak smoke

这个顺序的原因是：

- SCF 接口是所有后续工作的主干。
- `LC-ωPBE` 是最干净的验证模板。
- 先打通 ground-state Janak 优化，能最快验证这条路线是否值得继续加大投入。

---

## 12. 编程前的最小接口草案

下面的草案不是最终实现，只是为了保证 PR 1 编码时不再继续摇摆。

```python
@dataclass(frozen=True)
class ResolvedRSHParameters:
    sr_hf_fraction: Array
    lr_hf_fraction: Array
    omega: Array

    @property
    def paper_alpha(self) -> Array:
        return self.sr_hf_fraction

    @property
    def paper_beta(self) -> Array:
        return self.lr_hf_fraction - self.sr_hf_fraction

    def to_pyscf_rsh(self) -> tuple[Array, Array, Array]:
        return (
            self.omega,
            self.lr_hf_fraction,
            self.sr_hf_fraction - self.lr_hf_fraction,
        )

    def to_pyscf_rsh_and_hybrid(self) -> tuple[Array, Array, Array]:
        return (
            self.omega,
            self.lr_hf_fraction,
            self.sr_hf_fraction,
        )


@dataclass(frozen=True)
class SCFXCContributions:
    v_rho: Array
    v_grad: Array
    xc_kind: str
    full_hf_fraction: Array
    lr_hf_omegas: Array | None = None
    lr_hf_coefficients: Array | None = None
    extra_fock_matrix: Array | None = None
    exact_exchange_fraction: Array | None = None
    resolved_xc: Any | None = None


@dataclass(frozen=True)
class PySCFRSHSpec:
    xc_description: str | Callable[..., Any]
    xctype: str
    hyb: float
    rsh: tuple[float, float, float]

    def expected_rsh_and_hybrid_coeff(self) -> tuple[float, float, float]:
        ...
```

PR 1 的 coding target：

- 文档里的这个接口必须在代码里真实出现。
- `DifferentiableSCF` 至少能消费 `SCFXCContributions`。
- 测试覆盖 `to_pyscf_rsh()` / `to_pyscf_rsh_and_hybrid()` 的语义转换。

---

## 13. 当前尚未冻结的问题

- `jax_xc` 对目标 RSH functional 的 Python API 具体长什么样，是否足够稳定。
- local RSH 部分是否能直接从 `jax_xc` 取得一阶导数与需要的响应信息。
- `ωB97X-D` 是否在一阶段就需要带色散，还是先只做不含 `-D` 的主骨架。
- 是否需要在第一版就让 `reference.py` 支持 continuous `ω` cache，而不是直接在 functional 内即时构造。

这些问题不影响先启动 PR 1。

---

## 14. Endpoint Koopmans/IP-EA 自监督目标更新

**日期**: 2026-04-24  
**状态**: 已进入代码实验路径  
**动机**: 大分子和开壳层带电态下，严格分数占据 Janak loss 的 SCF 固定点会产生较强数值噪声。先退一步使用端点 `N-1 / N / N+1` 的离散 Koopmans 近似，可以更稳定地定义优化方向。

### 14.1 当前推荐 loss

对同一个 RSH functional 的自洽端点能量定义：

```text
IP = E(N-1) - E(N)
EA = E(N) - E(N+1)

r_IP       = eps_HOMO(N) + IP
r_EA_HOMO  = eps_HOMO(N+1) + EA
r_EA_LUMO  = eps_LUMO(N) + EA
```

第一版训练推荐只打开：

```text
L = w_IP * |r_IP| + w_LUMO_EA * |r_EA_LUMO| + w_prior * prior
```

默认：

- `w_IP = 1`
- `w_LUMO_EA = 1`
- `w_EA_HOMO = 0`
- `w_prior = 1e-3`

理由：

- `r_IP` 对应中性 HOMO 与 `DeltaSCF` 电离能。
- `r_EA_LUMO` 对应中性 LUMO 与 `DeltaSCF` 电子亲和能，是调中性前沿轨道最直接的端点约束。
- `r_EA_HOMO` 依赖阴离子 HOMO，当前可以作为诊断项，但不建议第一版强行训练。

### 14.2 梯度路径

当前有三种路径：

1. **detached charged states**
   带电态总能完全 stop-gradient。这个路径稳定，但梯度不代表端点能量对 RSH 参数的真实响应，训练中容易 loss 反向漂移。

2. **full-gradient charged SCF**
   对带电态 fixed-point SCF 全量反传。当前在 water smoke 中会出现 nonfinite gradient，不作为默认路径。

3. **endpoint envelope-gradient**
   中性/带电态 SCF 轨道固定在当前自洽点，能量对 RSH 参数的显式依赖保留梯度。这是目前默认推荐路径：

```text
SCF density/orbitals: stop-gradient
RSH explicit parameters in E_xc/K terms: differentiable
```

这个策略符合 stationary endpoint energy 的 envelope-theorem 近似，避免了不稳定的 charged-state fixed-point 梯度。

### 14.3 当前代码入口

- loss API:
  - `src/td_graddft/nn_rsh/losses.py`
  - `make_self_supervised_rsh_loss(...)`
- 带电态 differentiable SCF 构造:
  - `src/td_graddft/training/targets.py`
  - `charged_state_differentiable_scf_from_reference(...)`
- 直接三参数端点调谐工具:
  - `tools/tune_water_rsh_endpoint_koopmans.py`
  - 默认 `--preserve-network=True`，只移动输出 bias 来实现目标 raw RSH 参数，保留 descriptor、hidden layers 与 output kernel。

典型命令：

```bash
env JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mplconfig-tdgraddft \
python tools/tune_water_rsh_endpoint_koopmans.py \
  --steps 50 \
  --optimizer coordinate \
  --coordinate-step-size 0.05 \
  --basis sto-3g \
  --grid-level 0 \
  --scf-max-cycle 4 \
  --outdir outputs/water_nn_rsh_endpoint_koopmans_output_head_coordinate_ep50
```

### 14.4 Water 50-step 控制实验

输出目录：

```text
outputs/water_nn_rsh_endpoint_koopmans_output_head_coordinate_ep50
```

结果：

- initial loss: `0.461137 Ha`
- final/best loss: `0.361951 Ha`
- best step: `50`
- final `sr_hf_fraction`: `0.531687`
- final `lr_hf_fraction`: `0.667311`
- final paper `alpha`: `0.531687`
- final paper `beta`: `0.135623`
- final `omega`: `0.355936`
- `preserve_network`: `true`

解释：

- 端点 Koopmans 目标本身可优化，且在 NN output-head coordinate-search 下稳定下降。
- 下降主要来自提高 `sr_hf_fraction` 和 `lr_hf_fraction`；`omega` 在这 50 步中没有被选中移动，说明当前 water/STO-3G 端点目标对 HF fraction 更敏感。
- 下一步应从单分子 output-head tuning 扩展到多分子 batch：保持 hidden layers 可学习，但优先对 output head 使用坐标搜索/line-search 或较强 trust-region，再逐步释放隐藏层。

---

## 15. 半电荷处 fixed-orbital Janak AD 约束

**日期**: 2026-04-24  
**状态**: 已实现 smoke test  
**动机**: 端点 Koopmans 只约束 `N-1 / N / N+1` 的整数态关系，不能直接看到整数点之间的能量曲率。半电荷约束在 `N-0.5` 和 `N+0.5` 的自洽分数态上检查轨道占据数微分，可以作为更接近 piecewise-linearity 的辅助项。

### 15.1 定义

先用同一组冻结的 RSH 参数计算：

```text
s_- = E(N)   - E(N-1)
s_+ = E(N+1) - E(N)
```

然后在自洽分数态上固定轨道，使用 AD 对 frontier occupation 求偏导：

```text
d_- = partial E(C_{N-0.5}, f; theta) / partial f_HOMO
d_+ = partial E(C_{N+0.5}, f; theta) / partial f_LUMO

r_- = d_- - s_-
r_+ = d_+ - s_+

L_half = mean(|r_-|, |r_+|)
```

这里的 `C_{N±0.5}` 来自分数占据 SCF，但求导时固定轨道；这避免了 full fixed-point differentiation 的不稳定性。

### 15.2 代码入口

- `src/td_graddft/training/targets.py`
  - `half_charge_janak_autodiff_penalty(...)`
  - `_janak_frontier_penalty_by_mode(..., mode="half_charge_ad")`
- `src/td_graddft/training/config.py`
  - `janak_frontier_mode="half_charge_ad"`
- RSH 自监督 loss:
  - `make_self_supervised_rsh_loss(..., janak_weight=..., training_config=...)`
- CLI:
  - `--janak-mode half_charge_ad`

推荐先作为 endpoint Koopmans 的辅助项：

```text
L = |r_IP| + |r_LUMO_EA| + 0.1~0.3 * L_half + 1e-3 * prior
```

### 15.3 Water 10-step smoke test

命令：

```bash
env JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mplconfig-tdgraddft \
python tools/tune_water_rsh_endpoint_koopmans.py \
  --steps 10 \
  --optimizer coordinate \
  --coordinate-step-size 0.05 \
  --basis sto-3g \
  --grid-level 0 \
  --scf-max-cycle 4 \
  --janak-weight 0.2 \
  --janak-mode half_charge_ad \
  --koopmans-ip-weight 1.0 \
  --koopmans-ea-weight 0.0 \
  --koopmans-lumo-ea-weight 1.0 \
  --prior-weight 1e-3 \
  --outdir outputs/water_nn_rsh_half_charge_ad_endpoint_coordinate_ep10
```

结果：

- initial loss: `0.461378`
- final/best loss: `0.436561`
- half-charge Janak MAE: `1.208e-3 -> 1.042e-3 Ha`
- Koopmans IP MAE: `0.2563 -> 0.2421 Ha`
- Koopmans LUMO-EA MAE: `0.2048 -> 0.1943 Ha`
- final `sr_hf_fraction`: `0.307657`
- final `lr_hf_fraction`: `0.507915`
- final paper `alpha`: `0.307657`
- final paper `beta`: `0.200258`
- final `omega`: `0.356035`

解释：

- 新约束数值稳定，10 步内没有 nonfinite loss。
- 半电荷 Janak residual 随 endpoint loss 同时下降，说明它可以作为辅助曲率约束参与训练。
- 当前 loss 仍主要由端点 Koopmans 项控制；后续可以增加 `janak_weight` 或加入显式 fractional curvature 项，测试对 `N-1..N..N+1` 曲线的改善幅度。

---

## 16. Autschbach/Srebro 风格 2D Tuning 策略

**日期**: 2026-04-24  
**状态**: 已实现初版 loss/API 与训练入口  
**动机**: 两篇参考文献的关键结论是，单纯满足 `-epsilon_HOMO = IP` 往往只给出一条近似等价的参数谷。若目标是获得更可靠的密度与响应性质，需要在满足 Koopmans/IP tuning 与正确长程渐近行为的参数谷上，再用 `E(N)` 分数电荷曲率选择参数。

### 16.1 文献映射到本框架

文献 RSH 记号：

```text
alpha = short-range exact exchange
alpha + beta = long-range exact exchange
gamma = range-separation parameter
```

本框架内部记号：

```text
sr_hf_fraction = alpha
lr_hf_fraction = alpha + beta
omega = gamma
```

fully long-range corrected 约束：

```text
alpha + beta = 1  <=>  lr_hf_fraction = 1
```

### 16.2 Loss 变更

`make_self_supervised_rsh_loss(...)` 新增：

```text
long_range_correction_weight
```

对应项：

```text
residual_lc = lr_hf_fraction - 1
L_lc = long_range_correction_weight * |residual_lc|
```

新增 metrics：

```text
long_range_correction_residual
long_range_correction_mae
long_range_correction_penalty
```

### 16.3 默认 Koopmans 目标更新

当前训练默认采用文献中 Janak/piecewise-linearity 对中性 frontier 轨道给出的关系：

```text
IP = E(N-1) - E(N)
EA = E(N) - E(N+1)

r_H = epsilon_HOMO(N) + IP
r_L = epsilon_LUMO(N) + EA
```

因此 `tools/tune_water_rsh_endpoint_koopmans.py` 的默认权重调整为中性 HOMO/LUMO 约束：

```text
koopmans_ip_weight = 1.0
koopmans_ea_weight = 0.0          # optional HOMO(N+1) legacy/refined-tuning target
koopmans_lumo_ea_weight = 1.0     # primary LUMO(N) = -EA target
```

默认 frontier loss:

```text
L_frontier = |r_H| + |r_L|
```

同时记录 gap 诊断：

```text
r_gap = [epsilon_LUMO(N) - epsilon_HOMO(N)] - [IP - EA]
```

`r_gap` 默认不单独进入 loss，因为当 `r_H` 和 `r_L` 同时满足时它是冗余约束；它用于判断 HOMO/LUMO 与整数能量差的整体一致性。`HOMO(N+1)+EA` 保留为可选训练项和诊断项。

### 16.3.1 optDFTw-style traditional tuning

Sobereva 的 optDFTw 工具采用传统 tuned-RSH 目标：

```text
r_N   = epsilon_HOMO(N)   + E(N-1) - E(N)
r_N+1 = epsilon_HOMO(N+1) + E(N)   - E(N+1)

J   = |r_N| + |r_N+1|
J^2 = r_N^2 + r_N+1^2
```

optDFTw 默认用 Brent 算法最小化 `J^2`，默认搜索范围是 `w=0.05..0.6 Bohr^-1`，初猜为区间中点；Gaussian 通过 `IOp(3/107,3/108)` 写入 `w`。

当前工具中可用以下参数复现这个目标形式：

```bash
python tools/tune_water_rsh_endpoint_koopmans.py \
  --rsh-preset lc-wpbe \
  --rsh-omega-source canonical \
  --strategy single \
  --koopmans-ip-weight 1.0 \
  --koopmans-ea-weight 1.0 \
  --koopmans-lumo-ea-weight 0.0 \
  --koopmans-loss-kind squared \
  --fractional-weight 0.0 \
  --long-range-correction-weight 0.0
```

注意：`canonical` preset 当前上界是 `0.8`，比 optDFTw 默认 `0.6` 更宽；若目标函数在 `0.6` 以上仍下降，训练会继续推向更大 `omega`。严格对标 optDFTw 时应额外固定搜索上界为 `0.6`，或做显式扫描。

### 16.4 两阶段策略

新增 CLI：

```text
--strategy single|2dt
--stage-a-steps
--stage-b-steps
--fractional-weight
--long-range-correction-weight
```

`--strategy 2dt` 默认执行：

Stage A: `koopmans_lc`

```text
L_A = |r_IP| + |r_EA| + L_lc + prior
```

目的：先找到满足 Koopmans 与 `lr -> 1` 的参数谷。

Stage B: `curvature_selection`

```text
L_B = L_A + fractional_weight * fractional_charge_linearity_penalty
```

目的：在参数谷上进一步降低 `E(N)` 分数电荷曲率，作为密度质量代理。

### 16.5 典型命令

```bash
env JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mplconfig-tdgraddft \
python tools/tune_water_rsh_endpoint_koopmans.py \
  --strategy 2dt \
  --steps 20 \
  --optimizer coordinate \
  --coordinate-step-size 0.05 \
  --basis sto-3g \
  --grid-level 0 \
  --scf-max-cycle 3 \
  --prior-weight 1e-3 \
  --long-range-correction-weight 1.0 \
  --fractional-weight 1.0 \
  --outdir outputs/water_nn_rsh_2dt_endpoint_curvature_ep20
```

如果大分子分数态 SCF 不稳定，可先只跑 Stage A：

```bash
python tools/tune_water_rsh_endpoint_koopmans.py --strategy 2dt --steps 100 --stage-b-steps 0
```

然后用离线 fractional scan 选择曲率最小的参数。

## 17. 严格泛函形式：LC-wPBE 与 omegaB97X-D

本节固定命名泛函的文献/PySCF 定义，避免把 generic `(sr_hf_fraction, lr_hf_fraction, omega)` RSH 壳误称为具体泛函。

### 17.1 通用 RSH 交换分解

库内部使用：

```text
sr_hf_fraction = a
lr_hf_fraction = a + b
omega = range-separation parameter
```

PySCF/LibXC 的 RSH 约定为：

```text
rsh = (omega, lr_hf_fraction, sr_hf_fraction - lr_hf_fraction)
rsh_and_hybrid = (omega, lr_hf_fraction, sr_hf_fraction)
```

严格 RSH 不是简单的：

```text
full_semilocal_x + lr_hf * long_range_hf
```

而是需要互补的短程/长程半局域交换：

```text
E_x = a E_x,HF^SR(omega)
    + (1-a) E_x,DFT^SR(omega)
    + (a+b) E_x,HF^LR(omega)
    + (1-a-b) E_x,DFT^LR(omega)
```

因此，JAX generic RSH 路径若只用 `local_xc_spec="pbe"`，不能被称作严格 LC-wPBE。
严格 LC-wPBE 应使用 `local_xc_spec="lc_wpbe_local"`，也就是
`GGA_X_WPBEH + GGA_C_PBE`。

### 17.2 LC-wPBE

采用 LibXC/PySCF 名称：

```text
LC_WPBE / HYB_GGA_XC_LC_WPBE
```

默认参数：

```text
sr_hf_fraction = 0.0
lr_hf_fraction = 1.0
paper beta     = 1.0
omega          = 0.4 bohr^-1
PySCF rsh      = (0.4, 1.0, -1.0)
```

严格形式：

```text
E_xc^LC-wPBE = E_x,PBE^SR(omega=0.4) + E_x,HF^LR(omega=0.4) + E_c,PBE
```

无经验色散项。

当前 JAX 实现：

```text
local_xc_spec = "lc_wpbe_local"
local term    = GGA_X_WPBEH(omega) + GGA_C_PBE
RSH term      = LR-HF(omega), sr_hf_fraction=0, lr_hf_fraction=1
```

`GGA_X_WPBEH` 的局域交换项使用 `jax_xc`/LibXC 生成表达式的 unpolarized
形式，并通过交换自旋标度构造 unrestricted 形式。该表达式返回每粒子交换能，
代码中显式乘以自旋密度得到能量密度。回归测试已和 PySCF/LibXC 的
`GGA_X_WPBEH` restricted grid energy density 对齐，并验证 `omega` 可导。

### 17.3 omegaB97X-D

采用 LibXC/PySCF 名称：

```text
WB97X_D / HYB_GGA_XC_WB97X_D
```

默认参数：

```text
sr_hf_fraction = 0.222036
lr_hf_fraction = 1.0
paper beta     = 0.777964
omega          = 0.2 bohr^-1
PySCF rsh      = (0.2, 1.0, -0.777964)
```

严格形式：

```text
E_xc^omegaB97X-D =
    E_x,B97^SR(omega=0.2)
  + 0.222036 E_x,HF^SR(omega=0.2)
  + E_x,HF^LR(omega=0.2)
  + E_c,B97
  + E_disp
```

这里 `E_disp` 是 Chai-Head-Gordon 的经验阻尼原子对色散项。该泛函不是 PBE 型，也不能用 `pbe + RSH` 近似后仍称为 omegaB97X-D。

### 17.4 当前代码约束

底层 preset 放在 `td_graddft.jax_libxc`，可直接调用：

```python
from td_graddft.jax_libxc import get_rsh_functional_preset

get_rsh_functional_preset("lc-wpbe")
get_rsh_functional_preset("wb97x-d")
```

`td_graddft.nn_rsh.presets` 只保留 RSH schema/template 适配：

```python
make_rsh_template("lc-wpbe")
make_rsh_template("wb97x-d")
rsh_preset_default_params("lc-wpbe")
```

这些 preset 固定两类信息：

- `omega_source="canonical"`：严格文献/PySCF 默认值与宽松安全边界。
- `omega_source="optxc"`：参考 OPTXC 训练集 `best_omega` 分布的分子特异调参范围。

当前 OPTXC 参考范围采用 IQR 过滤后再保守取整：

```text
LC-wPBE:
  canonical omega = 0.4
  optxc tuning omega center = 0.205
  optxc tuning omega bounds = (0.13, 0.30)

omegaB97X-D:
  canonical omega = 0.2
  optxc tuning omega center = 0.164
  optxc tuning omega bounds = (0.10, 0.24)
```

训练脚本默认使用 `--rsh-omega-source optxc`，避免端点/曲率 loss 把水分子过拟合到不合理的大 `omega`；若要和 PySCF 默认参数直接对比，应显式使用 `--rsh-omega-source canonical`。

`strict_jax_supported=True` 目前只适用于 LC-wPBE。omegaB97X-D 仍为
`strict_jax_supported=False`，因为当前 JAX 训练路径还没有实现 B97
range-separated semilocal exchange/correlation 与经验色散项。

下一步若要严格训练：

```text
LC-wPBE:
  使用 preset 默认的 lc_wpbe_local；
  不再使用 pbe local + LR-HF 近似命名为 LC-wPBE。

omegaB97X-D:
  实现 B97 range-separated exchange/correlation 参数化；
  再决定训练中是否纳入 E_disp。色散对固定几何的参数梯度通常为零，但对几何和光谱比较不可忽略。
```
