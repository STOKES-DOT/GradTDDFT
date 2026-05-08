# TD-GradDFT 架构设计文档

**日期**: 2026-03-25
**版本**: 0.2.0
**状态**: 设计阶段

---

## 1. 项目概述

TD-GradDFT 是一个纯 JAX 实现的含时密度泛函理论（TDDFT）库，专注于神经网络交换相关泛函的训练和激发态计算。

### 1.1 核心目标

- **纯 JAX 实现**：所有核心计算（积分、SCF、TDDFT）使用 JAX，不依赖 PySCF/GradDFT 的核心代码
- **可微分训练**：支持固定密度和自洽两种训练模式
- **开壳层支持**：为 unrestricted 计算预留接口
- **模块化设计**：XC 泛函、特征提取、混合模块可自由组合
- **可发布库**：目标发布到 PyPI

### 1.2 设计原则

1. **分层架构**：清晰的依赖方向（基础设施 → 数据 → 核心计算 → 用户 API）
2. **协议驱动**：使用 `Protocol` 定义接口，便于扩展和替换
3. **可微分优先**：核心函数支持 `jax.grad` 和 `jax.vmap`
4. **配置驱动**：通过配置类控制行为，减少代码修改

---

## 2. 目录结构

```
src/td_graddft/
├── __init__.py                    # 顶层 API 导出
│
├── types/                         # 【基础设施层】
│   ├── __init__.py
│   ├── protocols.py               # Protocol 接口定义
│   ├── molecules.py               # 分子数据容器
│   └── results.py                 # SCF/TDDFT 结果容器
│
├── data/                          # 【数据层】
│   ├── __init__.py
│   ├── molecule.py                # Molecule 类
│   ├── basis.py                   # 基组解析与构建
│   ├── integrals/                 # 积分计算（纯 JAX）
│   │   ├── __init__.py
│   │   ├── one_electron.py        # S, T, V 积分
│   │   ├── two_electron.py        # ERI 积分
│   │   └── screening.py           # Schwarz 筛选
│   ├── grid/                      # 数值网格
│   │   ├── __init__.py
│   │   ├── atomic_grid.py         # 原子网格
│   │   └── molecular_grid.py      # 分子网格
│   ├── cache.py                   # 分层缓存
│   └── io.py                      # 文件读写
│
├── scf/                           # 【核心计算层】
│   ├── __init__.py
│   ├── rhf.py                     # Restricted HF/DFT
│   ├── uhf.py                     # Unrestricted HF/DFT
│   ├── diis.py                    # DIIS 加速
│   ├── convergence.py             # 收敛判断
│   └── differentiable.py          # 可微分 SCF 包装
│
├── xc/                            # 【核心计算层】
│   ├── __init__.py
│   ├── base.py                    # XC Protocol
│   ├── features.py                # 特征提取器
│   ├── energy_basis.py            # 能量基函数
│   ├── mixing.py                  # HF 混合模块
│   ├── lda.py                     # LDA 泛函
│   ├── gga.py                     # GGA 泛函（预留）
│   ├── neural/                    # 神经网络 XC
│   │   ├── __init__.py
│   │   ├── networks.py            # MLP 等网络
│   │   ├── functional.py          # NeuralXCFunctional
│   │   └── dm21_like.py           # DM21 风格实现
│   └── libxc_bridge.py            # jax_xc 桥接（可选）
│
├── tddft/                         # 【核心计算层】
│   ├── __init__.py
│   ├── response.py                # 响应矩阵构建
│   ├── casida.py                  # Casida 方程
│   ├── tda.py                     # TDA 近似
│   ├── unrestricted.py            # 开壳层 TDDFT（预留）
│   └── spectra.py                 # 光谱计算
│
├── training/                      # 【核心计算层】
│   ├── __init__.py
│   ├── config.py                  # 训练配置
│   ├── targets.py                 # 训练目标
│   ├── trainer.py                 # 训练器
│   └── checkpoint.py              # 检查点
│
├── adapters/                      # 【适配层】
│   ├── __init__.py
│   ├── pyscf.py                   # PySCF 数据导入
│   └── graddft.py                 # GradDFT 数据导入
│
└── workflows/                     # 【用户 API 层】
    ├── __init__.py
    ├── config.py                  # 实验配置
    ├── pipeline.py                # 端到端管道
    ├── reporting.py               # 结果报告
    └── presets.py                 # 预设配置模板
```

---

## 3. 核心设计

### 3.1 分层洋葱架构

```
┌─────────────────────────────────────────────────┐
│                   用户 API 层                     │
│  workflows / high_level_api.py                  │
├─────────────────────────────────────────────────┤
│                   核心计算层                     │
│  scf / xc / tddft / training / spectra          │
├─────────────────────────────────────────────────┤
│                   数据层                         │
│  molecule / integrals / grid / cache            │
├─────────────────────────────────────────────────┤
│                   基础设施层                     │
│  types / utils / io                             │
└─────────────────────────────────────────────────┘
```

**依赖方向**：外层依赖内层，内层不依赖外层

---

### 3.2 协议定义（`types/protocols.py`）

```python
from typing import Protocol, runtime_checkable
from jaxtyping import Array, PyTree

@runtime_checkable
class XCFunctional(Protocol):
    """交换相关泛函接口"""

    def energy(self, params: PyTree, density: Array, weights: Array) -> Array:
        """计算 XC 能量"""
        ...

    def potential(self, params: PyTree, density: Array) -> Array:
        """计算 XC 势"""
        ...

    def kernel(self, params: PyTree, density: Array) -> Array:
        """计算 f_xc 核（TDDFT 响应）"""
        ...

    def init_params(self, rng: Array, sample_density: Array) -> PyTree:
        """初始化参数"""
        ...
```

---

### 3.3 可微分 SCF 插口（`scf/differentiable.py`）

```python
class DifferentiableSCF:
    """SCF 求解器，支持两种模式"""

    def __init__(self, mode: str = "fixed_density"):
        # mode: "fixed_density" | "self_consistent"
        self.mode = mode

    def __call__(self, molecule, xc_functional, xc_params):
        if self.mode == "fixed_density":
            # 停止梯度，SCF 不进入计算图
            density = jax.lax.stop_gradient(molecule.density)
            return self._single_step(molecule, xc_functional, xc_params, density)
        else:
            # 完整 SCF 循环，梯度回传
            return self._full_scf(molecule, xc_functional, xc_params)
```

**模式说明**：
- `fixed_density`：固定密度训练，梯度只传到 XC 参数（训练快）
- `self_consistent`：自洽训练，SCF 进入计算图（端到端可微）

---

### 3.4 纯 JAX 积分计算（`data/integrals/`）

```python
@jax.jit
def compute_overlap(basis: BasisSet) -> Array:
    """计算重叠矩阵 S"""
    # 使用 Obara-Saika 或 McMurchie-Davidson 递推
    ...

@jax.jit
def compute_eri(basis: BasisSet) -> Array:
    """计算双电子积分 (mu nu | lambda sigma)"""
    # 使用 Rys 求积或 HGP 算法
    # 大基组需要 Schwarz 筛选
    ...
```

**积分引擎**：完全使用 JAX 实现，可 JIT 编译，不依赖 PySCF

---

### 3.5 模块化神经 XC 泛函（`xc/neural/functional.py`）

```python
@dataclass(frozen=True)
class NeuralXCFunctional:
    """模块化神经网络 XC 泛函

    组合方式: Features → CoefficientNetwork + EnergyBasis → ε_xc
    可选: HFMixer → α_eff
    """

    # 必需组件
    feature_extractor: FeatureExtractor
    coefficient_network: nn.Module
    energy_basis: EnergyBasis

    # 可选组件
    hf_mixer: HFMixer | None = None

    def energy(self, params, density, weights):
        # E_xc = ∫ ρ(r) · c(r) · e(r) dr
        ...
```

**模块组合示例**：
```python
# DM21 风格
dm21_like = NeuralXCFunctional(
    feature_extractor=GGAFeatures(),
    coefficient_network=MLP((64, 64, 64)),
    energy_basis=LDAEnergyBasis(),
    hf_mixer=NeuralHFMixing(),
)

# 简单 LDA
simple_lda = NeuralXCFunctional(
    feature_extractor=LDAFeatures(),
    coefficient_network=MLP((32, 32)),
    energy_basis=LDAEnergyBasis(),
)
```

---

### 3.6 开壳层支持（预留接口）

```python
# tddft/unrestricted.py
@dataclass(frozen=True)
class UnrestrictedSCFResult:
    """开壳层 SCF 结果"""
    mo_coeff_alpha: Array
    mo_coeff_beta: Array
    density_alpha: Array
    density_beta: Array
    spin_density: Array  # α - β

class UnrestrictedCasidaTDDFT:
    """开壳层 TDDFT 求解器（预留接口）"""
    ...
```

---

### 3.7 用户 API（`workflows/`）

```python
# 高层 API - 推荐入口
from td_graddft.workflows import ExperimentConfig, ExperimentPipeline

config = ExperimentConfig(
    experiment_name="benzene_neural_xc",
    systems=[...],
    train_indices=[0],
    xc_type="neural_lda",
    scf_mode="fixed_density",  # 关键配置
    max_epochs=500,
)

pipeline = ExperimentPipeline(config)
result = pipeline.run()
```

---

## 4. 数据流

```
用户输入（坐标 + 基组名）
        ↓
    parse_basis() → BasisSet（JAX 兼容）
        ↓
    compute_integrals()（纯 JAX）
        ↓
    Integrals（S, T, V, ERI, S^-1/2）
        ↓
    Molecule（Integrals + Grid）
        ↓
    SCF 求解器（纯 JAX）
        ↓
    SCFResult（能量、轨道、密度）
        ↓
    训练循环 / TDDFT 计算
```

---

## 5. 公开 API（`__init__.py`）

```python
# 核心
from .types import Molecule, SCFResult, TDDFTResult
from .types.protocols import XCFunctional

# XC 泛函
from .xc import (
    NeuralXCFunctional,
    make_neural_lda_functional,
    make_dm21_like_functional,
)

# SCF
from .scf import DifferentiableSCF, SCFConfig

# TDDFT
from .tddft import RestrictedCasidaTDDFT, solve_casida

# 训练
from .training import Trainer, TrainingConfig, EnergyTarget

# 工作流（推荐入口）
from .workflows import (
    ExperimentConfig,
    ExperimentPipeline,
    SystemConfig,
    run_experiment,
)
```

---

## 6. 实现优先级

### Phase 1: 核心基础
1. 基础类型和协议
2. 纯 JAX 积分计算（S, T, V, ERI）
3. Restricted SCF 求解器
4. LDA 级神经 XC 泛函
5. 固定密度训练

### Phase 2: TDDFT
1. Restricted Casida/TDA
2. 响应矩阵构建
3. 光谱计算

### Phase 3: 扩展
1. GGA 特征和泛函
2. 开壳层 SCF/TDDFT
3. 自洽训练模式

### Phase 4: 优化
1. 积分筛选和分块
2. 并行化
3. 检查点和断点续训

---

## 7. 技术决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| 上游依赖 | 完全独立 JAX | 可移植性强，可微性好 |
| SCF 模式 | 可配置插口 | 灵活性，支持两种训练方式 |
| XC 架构 | 模块化组合 | 易于扩展和实验 |
| 开壳层 | 预留接口 | 当前聚焦 closed-shell |
| 积分计算 | 纯 JAX | 不依赖 PySCF 核心代码 |
| 数据组织 | 分层缓存 | 支持断点续训 |

---

## 8. 待讨论问题

1. **双电子积分存储**：大基组时 ERI 占用大量内存，是否需要稀疏/分块？
2. **网格生成**：是否自己实现还是调用外部库（如 DFTGrid）？
3. **基组库**：如何存储和加载基组数据（内置 vs 外部文件）？

---

## 附录 A: 术语表

- **SCF**: Self-Consistent Field，自洽场
- **TDDFT**: Time-Dependent DFT，含时密度泛函理论
- **LDA**: Local Density Approximation，局域密度近似
- **GGA**: Generalized Gradient Approximation，广义梯度近似
- **ERI**: Electron Repulsion Integral，双电子积分
- **TDA**: Tamm-Dancoff Approximation
- **RHF/RKS**: Restricted HF/KS
- **UHF/UKS**: Unrestricted HF/KS

---

## 附录 B: 参考实现

- **GradDFT**: https://github.com/XanaduAI/GradDFT
- **jax_xc**: https://github.com/sail-sg/jax_xc
- **PySCF**: https://github.com/pyscf/pyscf
- **DM21**: DeepMind 21 (Nature 603, 2022)
