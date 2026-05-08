# JoltQC Integral Port Comparison

Reference clone:

- Repository: `/private/tmp/tdg_refs/JoltQC`
- Commit: `fd55a75e12887b60dc3a22645a589716c415e9a3`
- Remote HEAD verified with `git ls-remote https://github.com/ByteDance-Seed/JoltQC HEAD`

## Line-Level Mapping

| JoltQC source | Meaning | TD-GradDFT status |
| --- | --- | --- |
| `jqc/constants.py:25-36` | `NPRIM_MAX=3`, `BASIS_STRIDE=12`, `TILE=4` | Mirrored as `_JOLTQC_NPRIM_MAX`, `_JOLTQC_BASIS_STRIDE`, `_JOLTQC_GROUP_ALIGNMENT` in `cuda_direct_jk.py`. |
| `jqc/pyscf/basis.py:374-417` | `BasisLayout.from_mol`: split basis, sort/group, keep mapping | Partially mirrored for our native `CartesianBasis`; no PySCF dependency added. |
| `jqc/pyscf/basis.py:483-650` | Sort by `(l, -nprim)`, pad each group to alignment 4, pack `coords` and `ce` | Implemented as `CudaJoltQCBasisLayout` with packed `basis_data`, fp32 copy, pad mask, group offsets. |
| `jqc/pyscf/basis.py:680-837` | Split `nprim > NPRIM_MAX`, decontract multi-contraction shells | `nprim > 3` split is implemented for our already single-contraction `ContractedShell` objects. Multi-contraction decontracting remains handled upstream by the basis loader. |
| `jqc/backend/cart2sph.py:245-335` and `common/cart2cart.cu:18-35` | Duplicate/sum matrices between molecular AO order and split internal AO order | Added `ao_to_parent_ao` and CUDA bridge kernels `expand_joltqc_density_kernel` / `contract_joltqc_potential_kernel`; wired into the default no-cutoff J/K FFI path. |
| `jqc/pyscf/jk.py:170-187` | Build density screening condition and tile pairs | Existing TD-GradDFT has shell/tile screening for original shell layout; split-basis version still needs wiring. |
| `jqc/pyscf/jk.py:209-258` | Iterate group quartets and generate one `(l,nprim)` kernel per group quartet | Added `CudaJoltQCQuartetLayout` with `group_quartet_keys`, offsets, and shell quartet batches; default no-cutoff CUDA J/K now dispatches through `td_graddft_cuda_joltqc_direct_jk`. |
| `jqc/backend/jk_tasks.py:40-109` and `jk/screen_jk_tasks.cu:75-330` | GPU task screening into FP32/FP64 queues | TD-GradDFT has an older shell task queue; FP32/FP64 split is not ported yet. |
| `jqc/backend/jk_1qnt.py:173-317` | Generate specialized 1qnt kernel constants and launch shape | `joltqc_port.codegen` now emits basis-specific CUDA companion sources and dispatches JoltQC-style 1qnt kernels for optimal-scheme 1qnt signatures. |
| `jqc/backend/jk_1q1t.py:33-151` and `jk/1q1t.cu` | Use one-quartet/one-thread kernels for signatures whose optimal scheme is `[-1]` | Vendored and wired into the same basis-specific launcher. This is required for low-angular signatures such as `ssss`, where forcing 1qnt caused launch failure. |
| `jqc/backend/jk/1qnt.cu:28-55` | Packed basis data contract and kernel signature | Device-side packed basis helpers and grouped FFI signature are wired. |
| `jqc/backend/jk/1qnt.cu:143-284` | Load shell quartet, primitive loops, Rys roots | Ported through generated static CUDA source. Runtime CuPy/JoltQC imports are not required. |
| `jqc/backend/jk/1qnt.cu:301-478` | TRR/HRR unrolled recurrence and integral fragment accumulation | Ported through the generated companion source. |
| `jqc/backend/jk/1qnt.cu:480-862` | Six J/K contraction/scatter paths | Ported for the no-cutoff fast path. Internal J/K matrices now apply the same JoltQC post-processing (`J *= 2; A += A.T`) before mapping back to the public AO shape. |

## Current Boundary

The Python/JAX side now produces JoltQC-compatible split basis and grouped quartet metadata. The CUDA file has the bridge kernels needed to preserve the public SCF density shape while running an internal split-basis kernel, and the default no-cutoff CUDA J/K route now calls the grouped JoltQC-style FFI target.

The direct kernel route is now performance-positive for real molecule sizes. The remaining bottleneck is cold cost: basis-specific CUDA source generation plus nvcc compilation is still roughly one minute for a 6-31G* carbon/hydrogen basis.

Remote verification on an idle GPU1-equivalent visible device:

- H2/STO-3G J/K vs pair-matrix reference: `max_j = 0.0`, `max_k = 2.22e-16`.
- C-H/6-31g* split-basis path: `nao = 17`, `joltqc_nao = 18`, `max_j = 8.88e-15`, `max_k = 1.07e-14`.
- H2/6-31G*: JoltQC direct `first_jk_s = 0.2146`, pair-reference `0.4315`, `max_j = 2.22e-16`, `max_k = 1.67e-16`.
- Ethylene/6-31G*: JoltQC direct vs generic fallback on the same FFI path: first J/K `0.2981s` vs `0.9735s`; steady-min J/K `0.0339s` vs `0.7087s`; checksum agreed to about `1e-8`.
- Ethylene/6-31G*: PySCF CPU direct J/K on 192 threads was still competitive for this small system: first `0.0877s`, steady-min `0.0301s`.
- Benzene/6-31G*: JoltQC direct beats PySCF CPU direct J/K: JoltQC steady-min `0.0373s` vs PySCF steady-min `0.6350s`, with `max_j = 3.02e-8`, `max_k = 2.12e-8`.
