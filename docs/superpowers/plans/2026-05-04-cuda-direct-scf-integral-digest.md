# CUDA Direct SCF Integral Digest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing `direct_jk_engine="cuda"` no-DF SCF path default to a JoltQC-style CUDA integral engine: `.cu` task screening plus specialized direct J/K kernels, without constructing full ERI or pair-ERI matrices.

**Architecture:** Keep the public API unchanged: `RKSConfig(jk_backend="direct", direct_jk_engine="cuda")` still selects the GPU no-DF path. Internally `_make_jk_builder` instantiates `CudaDirectJKBuilder` and calls `build_jk()` for each density or density difference; explicit supplied `eri_pair_matrix` remains a compatibility path, but automatic full/pair ERI cache construction is removed from the default CUDA route. The CUDA implementation will be reorganized toward JoltQC's model: basis shell/pair layout on the host, `.cu` screening kernel that produces a compact shell-quartet task queue, then specialized J/K digestion kernels keyed by angular momentum and primitive pattern. JAX sees a single differentiable FFI operation with a custom VJP.

**Tech Stack:** Python, JAX, CUDA FFI, pytest, remote `jax_scf` GPU environment.

---

### Task 1: Make CUDA Direct J/K The Default RKS Path

**Files:**
- Modify: `src/td_graddft/scf/rks.py`
- Test: `tests/test_density_fitting_rks.py`

- [ ] **Step 1: Write failing tests**

Add tests asserting that automatic CUDA direct SCF does not call `build_eri_pair_matrix()` or `build_eri_tensor()`, and does call `build_jk()`:

```python
def test_rks_direct_cuda_engine_uses_direct_digest_without_pair_or_full_cache(monkeypatch):
    import td_graddft.scf.rks as rks_mod

    basis = basis_from_pyscf_spec(
        "H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=1,
    )
    density = np.asarray([[0.83, 0.21], [0.21, 0.71]], dtype=np.float64)
    captured = {}

    class FakeCudaDirectJKBuilder:
        def __init__(self, direct_basis):
            captured["basis"] = direct_basis

        def build_eri_tensor(self):
            raise AssertionError("CUDA direct SCF default must not construct full ERI.")

        def build_eri_pair_matrix(self):
            raise AssertionError("CUDA direct SCF default must not construct pair-ERI matrix.")

        def build_jk_from_eri_pair_matrix(self, pair_arg, density_arg):
            raise AssertionError("CUDA direct SCF default must not use pair-ERI contraction.")

        def build_jk(self, density_arg, **kwargs):
            captured["density"] = density_arg
            captured["kwargs"] = kwargs
            return np.ones_like(np.asarray(density_arg)), 2.0 * np.ones_like(np.asarray(density_arg))

    monkeypatch.setattr(rks_mod, "CudaDirectJKBuilder", FakeCudaDirectJKBuilder)
    monkeypatch.setattr(rks_mod, "cuda_ffi_available", lambda: True)

    builder = rks_mod._make_jk_builder(
        None,
        RKSConfig(jk_backend="direct", direct_jk_engine="cuda"),
        direct_basis=basis,
        with_k=True,
    )
    j_mat, k_mat = builder(density)

    assert captured["basis"] is basis
    assert np.allclose(np.asarray(captured["density"]), density)
    assert captured["kwargs"]["density_cutoff"] == 0.0
    assert np.allclose(np.asarray(j_mat), 1.0)
    assert np.allclose(np.asarray(k_mat), 2.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
JAX_PLATFORMS=cpu pytest -q tests/test_density_fitting_rks.py::test_rks_direct_cuda_engine_uses_direct_digest_without_pair_or_full_cache
```

Expected: fail because `_make_jk_builder` currently prebuilds pair/full ERI caches when thresholds allow them.

- [ ] **Step 3: Implement default direct digest**

In `src/td_graddft/scf/rks.py`, inside `_make_jk_builder`, remove automatic `_should_cache_cuda_pair_eri()` and `_should_cache_cuda_full_eri()` branches from the CUDA default path. Keep the explicit `eri_pair_matrix is not None` compatibility branch. The default returned builder should call:

```python
result_j, result_k = cuda_builder.build_jk(density, density_cutoff=threshold)
```

and incremental mode should call:

```python
delta_j, delta_k = cuda_builder.build_jk(delta, density_cutoff=threshold)
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
JAX_PLATFORMS=cpu pytest -q tests/test_density_fitting_rks.py::test_rks_direct_cuda_engine_uses_direct_digest_without_pair_or_full_cache tests/test_density_fitting_rks.py::test_direct_cuda_jk_builder_uses_incremental_density_difference tests/test_density_fitting_rks.py::test_direct_cuda_jk_builder_applies_screening_to_initial_density
```

Expected: all pass.

### Task 2: Keep Explicit Pair-ERI Compatibility But Mark It Non-Default

**Files:**
- Modify: `tests/test_density_fitting_rks.py`

- [ ] **Step 1: Update old cache-preference tests**

Replace tests named around "prefers pair cache" and "uses full ERI cache" with direct-digest expectations for the automatic path. Keep `test_rks_direct_cuda_engine_reuses_supplied_pair_eri_cache` unchanged because explicit supplied pair data is a compatibility contract.

- [ ] **Step 2: Run direct CUDA RKS tests**

Run:

```bash
JAX_PLATFORMS=cpu pytest -q tests/test_density_fitting_rks.py::test_rks_direct_cuda_engine_uses_direct_digest_without_pair_or_full_cache tests/test_density_fitting_rks.py::test_rks_direct_cuda_engine_reuses_supplied_pair_eri_cache tests/test_density_fitting_rks.py::test_direct_cuda_jk_builder_falls_back_to_jax_when_cuda_unavailable
```

Expected: all pass.

### Task 3: Verify CUDA FFI Direct J/K Coverage

**Files:**
- Test: `tests/test_cuda_direct_jk_ffi.py`
- Test: `tests/test_gpu_s_shell_direct_jk.py`

- [ ] **Step 1: Run FFI interface and s/p/d extraction tests**

Run:

```bash
JAX_PLATFORMS=cpu pytest -q tests/test_cuda_direct_jk_ffi.py tests/test_gpu_s_shell_direct_jk.py
```

Expected: pass, including `extract_cartesian_ao_system_can_allow_d_shells`.

### Task 4: Remote Fresh-Cache Benchmark

**Files:**
- Use existing `/tmp/tdg_true_cold_scf_compare.py` on the remote server.

- [ ] **Step 1: Sync changed files**

Run:

```bash
rsync -avR -e 'ssh -p 60001' src/td_graddft/scf/rks.py tests/test_density_fitting_rks.py yjiao@8.218.101.131:/home/yjiao/TD-GradDFT/
```

- [ ] **Step 2: Run water/benzene/naphthalene fresh-cache SCF on GPU**

Use `conda activate jax_scf`, set `CUDA_VISIBLE_DEVICES` to an idle GPU, set `TD_GRADDFT_NVCC=/nonexistent` after selecting the prebuilt library, and run the fresh-cache benchmark script for at least water, benzene, and naphthalene.

Expected report fields: `td_graddft_input_plus_scf_s`, `pyscf_full_scf_s`, `energy_diff_ha`, GPU utilization, and memory.

### Task 5: Decide Next Kernel Work From Measurements

**Files:**
- Modify only after benchmark identifies the bottleneck.

- [ ] **Step 1: If direct digest is slower than pair cache**

Inspect whether time is in `pair_schwarz`, direct quartet digestion, or repeated SCF iterations. The next code task should be shell-level direct J/K task batching rather than pair-ERI matrix rebuilds.

- [ ] **Step 2: If energy diff exceeds `1e-7 Ha`**

Lower density screening or disable density screening while keeping direct digest. Do not re-enable pair/full ERI cache as the default.

### Task 6: Add JoltQC-Style Basis Layout Metadata

**Files:**
- Modify: `src/td_graddft/scf/cuda_direct_jk.py`
- Test: `tests/test_cuda_direct_jk_ffi.py`

- [ ] **Step 1: Write failing tests**

Add tests showing that `extract_cuda_ao_system()` records shell-pair screen group ids and that multiple AO pairs inside one shell pair share a screen group:

```python
def test_cuda_ao_system_records_shell_pair_screen_groups():
    basis = basis_from_spec(
        "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
    )
    builder = CudaDirectJKBuilder.__new__(CudaDirectJKBuilder)
    system = cuda_direct_jk.extract_cuda_ao_system(basis)
    group_ids = np.asarray(system.pair_screen_group_ids)
    assert group_ids.shape == (basis.nao * (basis.nao + 1) // 2,)
    assert system.n_pair_screen_groups < group_ids.size
```

- [ ] **Step 2: Implement shell-pair metadata**

In `CudaAOSystem`, add:

```python
pair_screen_group_ids: np.ndarray
n_pair_screen_groups: int
```

Populate these from `basis.shells[*].ao_indices`, falling back to one AO per shell when shell metadata is absent.

- [ ] **Step 3: Run tests**

Run:

```bash
JAX_PLATFORMS=cpu pytest -q tests/test_cuda_direct_jk_ffi.py::test_cuda_ao_system_records_shell_pair_screen_groups
```

Expected: pass.

### Task 7: Introduce Conservative Shell-Pair Schwarz Pooling

**Files:**
- Modify: `src/td_graddft/scf/cuda_direct_jk.py`
- Test: `tests/test_cuda_direct_jk_ffi.py`

- [ ] **Step 1: Write failing tests**

Add a test that fake AO-pair Schwarz values are max-pooled within each shell-pair screen group before being passed to the CUDA direct J/K kernel:

```python
def test_cuda_direct_jk_builder_pools_pair_schwarz_by_shell_pair(monkeypatch, tmp_path):
    basis = basis_from_spec(
        "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
    )
    npair = basis.nao * (basis.nao + 1) // 2
    raw_bounds = np.arange(npair, dtype=np.float64) + 1.0

    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_library", lambda self: tmp_path / "libfake.so")
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_and_register", lambda self: None)
    monkeypatch.setattr("td_graddft.scf.cuda_direct_jk._ffi_call", lambda *args, **kwargs: raw_bounds)

    builder = CudaDirectJKBuilder(basis, cache_dir=tmp_path)
    pooled = np.asarray(builder.build_pair_schwarz())
    group_ids = np.asarray(builder.system.pair_screen_group_ids)
    for group_id in np.unique(group_ids):
        mask = group_ids == group_id
        assert np.allclose(pooled[mask], raw_bounds[mask].max())
```

- [ ] **Step 2: Implement pooling**

Use `jax.ops.segment_max(pair_schwarz, pair_screen_group_ids, num_segments=n_pair_screen_groups)` and scatter back to AO pairs.

- [ ] **Step 3: Verify**

Run:

```bash
JAX_PLATFORMS=cpu pytest -q tests/test_cuda_direct_jk_ffi.py tests/test_gpu_s_shell_direct_jk.py
```

Expected: pass.

### Task 8: Replace Positive-Cutoff Branch With Task Queue Screening

**Files:**
- Modify: `src/td_graddft/scf/cuda_direct_jk.py`
- Modify: `src/td_graddft/scf/cuda_direct_jk_kernel.cu`
- Test: `tests/test_cuda_direct_jk_ffi.py`

- [ ] **Step 1: Add source-level RED tests**

The `.cu` source must contain:

```python
assert "screen_jk_tasks" in source
assert "log_dm_cond" in source
assert "shell_quartet_tasks" in source
assert "q_ij + q_kl + d_large" in source
```

- [ ] **Step 2: Implement the JoltQC-style screening kernel**

Add a CUDA kernel modeled on JoltQC's `screen_jk_tasks.cu`: it receives shell/tile mappings, log Schwarz matrix, and log density block maxima, and outputs compact shell quartet tasks. It must not use the current AO-pair `pair_schwarz[pair_p] * pair_schwarz[pair_q] * abs_density < cutoff` branch for positive cutoff.

- [ ] **Step 3: Route `build_jk(... density_cutoff>0)` through task queue**

The JAX FFI wrapper should call task screening first, then launch the direct J/K digestion kernel over the generated task list.

- [ ] **Step 4: Remote validation**

Run benzene and naphthalene with `direct_scf_tol=1e-13` and require:

```text
abs(energy_diff_ha) < 1e-7
converged == true
```

If this fails, disable positive-cutoff screening by default and keep task queue work behind tests until fixed.

### Task 9: Add JAX Custom VJP Wrapper

**Files:**
- Modify: `src/td_graddft/scf/cuda_direct_jk.py`
- Test: `tests/test_cuda_direct_jk_ffi.py`

- [ ] **Step 1: Add differentiability tests**

Test that a scalar depending on `J(D)` differentiates with respect to `D`:

```python
def test_cuda_direct_jk_custom_vjp_density_contracts_with_cotangent(monkeypatch):
    # Use fake primal and transpose calls to verify the VJP invokes the same linear JK operation.
```

- [ ] **Step 2: Implement custom VJP**

Wrap FFI direct J/K as:

```python
@jax.custom_vjp
def _cuda_direct_jk_vjp_call(density, static_basis_arrays, density_cutoff):
    ...
```

The backward rule should exploit linearity: cotangent wrt density is the adjoint J/K contraction. For symmetric density and symmetric J/K, reuse the same direct J/K kernel with the cotangent matrix, applying the same hybrid coefficients at the SCF layer.

- [ ] **Step 3: Verify local autodiff**

Run the targeted custom VJP tests on CPU with fake FFI calls.
