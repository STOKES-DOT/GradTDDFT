# TD-GradDFT 与 GPU4PySCF 对齐的 GPU 路径审计

**日期**: 2026-04-08  
**状态**: 执行中  
**目标**: 识别当前 TD-GradDFT 中仍停留在 CPU 的关键环节，并给出一条向 GPU4PySCF 数据流靠拢、同时保留 JAX 可微训练能力的重构路线。

---

## 1. GPU4PySCF 提供了什么

`gpu4pyscf` 已经把这些核心阶段放到 GPU：

- density fitting 与 direct SCF
- HF/DFT 的 SCF、梯度、Hessian
- 通过 libXC 的 LDA / GGA / mGGA / hybrid / RSH
- spin-conserved / spin-flip TDA 与 TDDFT
- `PySCF >= 2.5` 时支持 `mf.density_fit().to_gpu()`

参考：

- GitHub README: <https://github.com/pyscf/gpu4pyscf>
- 最新 release: <https://github.com/pyscf/gpu4pyscf/releases>

这说明它的主设计是：

1. 用 GPU 原生积分/JK/DF 路径构造 ground-state reference  
2. 在 GPU 常驻的轨道/DF 张量上继续做 TDA / TDDFT  
3. 尽量避免把 full ERI、grid、中间张量频繁搬回 CPU

---

## 2. 当前 TD-GradDFT 链路里仍然停在 CPU 的部分

### 2.1 reference 构建阶段仍然最重

在 [reference.py](../../src/td_graddft/reference.py)：

- `integral_backend='libcint'` 时，`mol.intor('int2e')` / `mol.intor_symmetric(...)` 仍由 PySCF/libcint 在 CPU 上生成，再绑定成常量进入 JAX。
- `grid_ao_backend!='jax'` 时，仍会走 `pyscf.dft.gen_grid` 和 `numint.eval_ao`。
- 即使走 strict-JAX grid，`build_molecular_grid_from_spec(...)` 目前仍有 Python 层原子循环和原子对双循环。
- `jk_backend='df'` 不是 PySCF/GPU4PySCF 风格的 3c2e/metric density fitting，而是先构造 AO-pair Coulomb 矩阵再做低秩分解。

结论：

- 当前 reference build 的主要瓶颈仍在 CPU。
- 这也是远程训练里 GPU 长时间 `0% util` 的第一来源。

### 2.2 grid 构造和 AO-on-grid 还不够 GPU-first

在 [grid.py](../../src/td_graddft/data/grid.py)：

- `build_molecular_grid_from_spec(...)` 对每个 atom 做局域 grid，再做 Becke partition。
- 当前实现包含：
  - `for ia, sym in enumerate(spec.symbols)`
  - `for i in range(natm):`
  - `for j in range(i):`

这意味着：

- 小体系上问题不大
- 但大体系时，grid 准备不是单个大张量程序，而是 host 驱动的分段拼装

在 [grid_ao.py](../../src/td_graddft/data/grid_ao.py)：

- `evaluate_cartesian_ao(...)` 的主 kernel 是 JIT 的
- 但大 grid 仍可能走 Python chunk loop

结论：

- grid/AO 已经是 JAX 形式
- 但还没有完全变成 GPU 友好的“大批量单图计算”

### 2.3 双电子积分与当前 DF 路径没有对齐 GPU4PySCF 的主思路

在 [two_electron.py](../../src/td_graddft/data/integrals/two_electron.py)：

- `eri_tensor(...)` / `eri_tensor_screened(...)` 仍然以 full ERI materialization 为主线
- 即使 shell-block path 已做了大量 `jit/vmap` 优化，full ERI 仍是 `O(n^4)` 张量路径

在 [df/jk.py](../../src/td_graddft/df/jk.py)：

- `eri_to_df_factors(...)` / `eri_pair_matrix_to_df_factors(...)` 目前核心还是：
  - 先构造 AO-pair matrix
  - 再用 `np.linalg.eigh(...)` 在 CPU 上分解

这和 GPU4PySCF 的主路径不同：

- GPU4PySCF 主路径是 auxiliary-basis / 3-center integral / metric
- 当前 TD-GradDFT 的 DF 路径本质上还是 full-ERI 的后处理近似

结论：

- 当前 J/K 的运行时 contraction 已经可以留在 JAX/GPU
- 但 DF factorization 的入口仍然是 CPU-heavy
- 这是当前“SCF 不容易把 GPU 喂满”的第二来源

### 2.4 SCF 迭代本身已经是设备可执行，但 workload 形状还不对

在 [differentiable.py](../../src/td_graddft/scf/differentiable.py)：

- self-consistent SCF 主循环已经是 `jax.lax.scan`
- 每步 J/K、`V_xc`、对角化都可以在设备上执行

在 [rks.py](../../src/td_graddft/scf/rks.py)：

- `build_jk_from_df(...)` / `build_j_from_df(...)` 已经是 `@jax.jit`
- `_xc_energy_and_potential_on_grid(...)` 也是纯 JAX

问题不在“SCF 没上 GPU”，而在：

- reference/integral/grid 之前的 CPU 阶段太重
- 单个分子、单个几何点、很小 basis 时，单步 SCF kernel 规模太小
- 之前训练数据还是 Python datum loop，现在只在一个窄场景下加了 batched fast path

结论：

- SCF 主循环结构本身已经接近 GPU 可执行
- 但数据准备和 batch 形状还没有达到 GPU4PySCF 那种吞吐风格

### 2.5 TDDFT/TDA/Casida 求解器本身已经在 JAX 设备侧

在 [response.py](../../src/td_graddft/tddft/response.py)：

- A/B 矩阵装配是纯 JAX `einsum`
- 如果 `eri_ovov/eri_ovvo/eri_oovv` 已经准备好，这部分可以直接在 GPU 上跑

在 [tda.py](../../src/td_graddft/tddft/tda.py) 和 [casida.py](../../src/td_graddft/tddft/casida.py)：

- dense 求解走 `jnp.linalg.eigh`
- Davidson 求解也是 JAX matvec 路线

所以：

- TDDFT 末端求解器不是当前主要 CPU 瓶颈
- 真正限制 GPU 利用率的是上游 reference / ERI / DF / grid

---

## 3. 当前“GPU 利用率低”的直接原因

按优先级排序：

1. reference build 仍然大量依赖 CPU  
2. DF 不是 true auxiliary-basis DF，而是 AO-pair matrix 分解  
3. grid partition 仍有 Python 循环  
4. 小体系/小 batch 时，SCF 与 TDDFT 的设备 kernel 太小  
5. 训练数据与几何点批处理仍不够彻底

所以不能把问题理解成“JAX 没有把 SCF 放上 GPU”。更准确的说法是：

- SCF/TDDFT 主算子已经可以上 GPU
- 但 GPU 前面的数据准备和张量形状还没有变成 GPU-first

---

## 4. 向 GPU4PySCF 靠拢，但保持 JAX 训练能力的重构路线

### Phase A: 把 GPU4PySCF 作为 GPU 参考后端，而不是最终训练主干

先引入一个明确的工程边界：

- `gpu4pyscf` 用于 benchmark / correctness / 数据流对照
- 先不要把它直接塞进可微训练主链路

原因：

- `gpu4pyscf` 是 `PySCF + CuPy/CUDA` 路线
- 它不是 JAX 计算图，不能天然提供 JAX 反传

但它很适合回答三个问题：

- GPU 上的 SCF 极限吞吐大概是多少
- GPU 上的 DF/JK/TDDFT 应该怎样组织数据流
- 我们当前的慢点到底在 reference、SCF、还是 TDDFT

### Phase B: 将当前 DF 改成真正的 3c2e / metric 路径

这是最关键的一步。

目标替换：

- 当前：
  - full ERI -> AO-pair matrix -> eig/low-rank factor
- 目标：
  - auxiliary basis
  - `(P|pq)` 3-center integral
  - `(P|Q)` metric
  - `B_{P,pq}` 常驻设备

这样改完后：

- SCF 的 J/K 可以像 GPU4PySCF 一样走 DF-first
- TDDFT 的 `ovov/ovvo/oovv` 也可以从 DF tensors 直接构造
- 不需要频繁 materialize full `rep_tensor`

### Phase C: 把 grid preparation 改成单图张量程序

目标：

- 去掉 `build_molecular_grid_from_spec(...)` 里的 Python 双循环
- 把 atom-local grid、Becke partition、拼接都改成张量化批处理

至少要做到：

- grid 构造输入是 `coords[natm,3]`, `charges[natm]`
- 输出是单个设备数组 `grid_coords[ngrids,3]`, `weights[ngrids]`
- 不再依赖 host 侧 `append + concatenate`

### Phase D: 扩大 SCF 设备工作负载

目标：

- 批处理几何点
- 批处理训练样本
- 尽量让 `SCF cycle` 内部只看设备常驻张量

已完成的第一步：

- `ground_state_mse_loss(...)` 已为 `self_consistent + 多 datum + energy-only` 加了 batched fast path

后续要继续：

- 扩到更多训练配置
- 扩到 reference evaluation / dense inference
- 逐步把当前 workflow 里仍然按 Python datum 迭代的地方收掉

### Phase E: TDDFT 主路径优先使用 DF 响应构造

当前 response builder 已经是 JAX 的，但它高度依赖上游提供：

- `eri_ovov`
- `eri_ovvo`
- `eri_oovv`

下一步应改成：

- 默认优先从 DF tensors 构造这些响应块
- 大体系优先走 matvec/operator 形式，而不是一上来 materialize dense Casida matrix

这一步和 GPU4PySCF 的路线是一致的：

- GPU 上优先避免 full ERI 与 full dense response matrix

### Phase F: 明确可微边界

如果当前只要求：

- **对神经网络参数可微**

那么：

- 外部 GPU 积分 kernel
- 外部 GPU SCF reference
- 外部 GPU DF tensor 构造

都可以作为“常量上游”，不阻碍对 `xc_params` 的训练。

如果后续要求：

- 对几何坐标也严格可微

那么就需要：

- 自己的 JAX kernel
- 或者 `custom_vjp/custom_jvp`
- 或者 JAX FFI 接 CUDA kernel 并补导数

这两条路线要分开看，不应混成一个任务。

---

## 5. 结论

当前最该做的，不是继续单独优化末端 TDA/Casida 求解器，而是：

1. 用 GPU4PySCF 建立 GPU 参考基线  
2. 把当前 pseudo-DF 改成 true auxiliary-basis DF  
3. 把 grid/reference 准备改成 GPU-first 的张量数据流  
4. 扩大 SCF/训练的 batched device workload

如果只盯着“SCF 主循环是否在 GPU 上”，会误判问题。当前真正限制 GPU 利用率的，是 **GPU 前面的 CPU-heavy 准备阶段** 和 **不够 GPU-friendly 的数据形状**。

---

## 6. 下一步执行顺序

建议严格按这个顺序推进：

1. 引入 `gpu4pyscf` benchmark/reference 脚本，测同一分子上的 GPU SCF/TDA/TDDFT 基线
   - 已新增脚本：
     - [benchmark_gpu4pyscf_vs_strict_jax_full.py](/Users/jiaoyuan/Documents/GitHub/TD-GradDFT/tools/benchmark_gpu4pyscf_vs_strict_jax_full.py)
2. 新建 true-DF 设计文档：aux basis、3c2e、metric、J/K、TDDFT 响应块
3. 改 `build_molecular_grid_from_spec(...)` 为张量化版本
4. 将 batched `self_consistent` 训练路径扩展到 workflow 主线
5. 再做大体系 GPU profiling，对比：
   - reference prep
   - SCF
   - TDDFT matrix build
   - eigensolver
