# Remove Legacy Training APIs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove legacy public Neural XC and PySCF bridge APIs without breaking the supported `hfx_local` / `hfx_nu` / PT2 training path.

**Architecture:** Keep `neural_xc.Functional(...)` and explicit reference builders as the supported route. Remove old public exports first, migrate tests and docs to the current route, then add guards that fail before optimization when local-HF or PT2 payloads are missing. This workspace is not a Git checkout, so verification checkpoints replace commit steps.

**Tech Stack:** Python, JAX, Flax, PySCF, pytest, TD-GradDFT Neural XC / reference / training modules.

---

## File Map

- `src/td_graddft/neural_xc/__init__.py`: public Neural XC export surface.
- `src/td_graddft/neural_xc/api.py`: public Neural XC facade; remove deprecated `make_dm21_like_functional`.
- `src/td_graddft/__init__.py`: top-level lazy exports; remove legacy Neural XC names.
- `src/td_graddft/pyscf_bridge.py`: deprecated shim to remove or turn into hard import failure.
- `tests/test_neural_xc_public_api.py`: public API expectations and trainer validation tests.
- `tests/test_pyscf_style_namespace.py`: top-level namespace expectations.
- `tests/test_tools_public_api_usage.py`: deprecated import checks.
- `tests/test_water_smoke.py`: migrate from removed LDA toy functional to current Neural XC route.
- `README.md`: remove public references to the deprecated wrapper and compatibility bridge.
- `examples/` and `tools/`: replace `td_graddft.pyscf_bridge` imports if any remain; leave `reference_legacy` imports explicit.
- `src/td_graddft/training/config.py` and/or `src/td_graddft/training/neural_xc_trainer.py`: add fail-fast validation for local-HF/PT2 training payloads if not already complete.

---

### Task 1: Add Public API Removal Tests

**Files:**
- Modify: `tests/test_neural_xc_public_api.py`
- Modify: `tests/test_pyscf_style_namespace.py`
- Modify: `tests/test_tools_public_api_usage.py`

- [ ] **Step 1: Write failing public Neural XC removal test**

Add this test to `tests/test_neural_xc_public_api.py` near the existing public constructor tests:

```python
def test_legacy_neural_xc_public_constructors_are_removed():
    removed = (
        "DensityNeuralXCFunctional",
        "NeuralXCFunctional",
        "PointwiseMLP",
        "make_neural_lda_functional",
        "make_dm21_like_functional",
    )

    for name in removed:
        assert not hasattr(neural_xc, name), f"{name} should not be a public Neural XC API"
```

Remove or replace the existing deprecated-wrapper expectation:

```python
def test_dm21_like_public_wrapper_is_deprecated():
    ...
```

with the absence assertion above.

- [ ] **Step 2: Write failing top-level removal test**

Add this to `tests/test_pyscf_style_namespace.py`:

```python
def test_top_level_removes_legacy_neural_xc_exports():
    import td_graddft

    removed = (
        "DensityNeuralXCFunctional",
        "NeuralXCFunctional",
        "PointwiseMLP",
        "make_neural_lda_functional",
        "make_dm21_like_functional",
    )

    for name in removed:
        assert not hasattr(td_graddft, name), f"{name} should not be exported at top level"
```

- [ ] **Step 3: Write failing PySCF bridge removal test**

Add this to `tests/test_tools_public_api_usage.py`:

```python
def test_pyscf_bridge_module_is_removed_from_public_api():
    import importlib

    try:
        importlib.import_module("td_graddft.pyscf_bridge")
    except ModuleNotFoundError:
        return
    except ImportError as exc:
        assert "td_graddft.reference_legacy" in str(exc)
        return

    raise AssertionError("td_graddft.pyscf_bridge should no longer import successfully")
```

- [ ] **Step 4: Verify RED**

Run:

```bash
pytest tests/test_neural_xc_public_api.py::test_legacy_neural_xc_public_constructors_are_removed tests/test_pyscf_style_namespace.py::test_top_level_removes_legacy_neural_xc_exports tests/test_tools_public_api_usage.py::test_pyscf_bridge_module_is_removed_from_public_api -q
```

Expected: FAIL because legacy names still exist and `td_graddft.pyscf_bridge` still imports.

---

### Task 2: Remove Legacy Public Exports

**Files:**
- Modify: `src/td_graddft/neural_xc/__init__.py`
- Modify: `src/td_graddft/neural_xc/api.py`
- Modify: `src/td_graddft/__init__.py`
- Delete or hard-fail: `src/td_graddft/pyscf_bridge.py`

- [ ] **Step 1: Remove deprecated Neural XC wrapper from facade**

In `src/td_graddft/neural_xc/api.py`, remove:

```python
import warnings
from .dm21.functional import make_dm21_like_functional as _make_dm21_like_functional
```

and delete:

```python
def make_dm21_like_functional(*args: Any, **kwargs: Any):
    warnings.warn(
        "neural_xc.make_dm21_like_functional is deprecated; "
        "use neural_xc.make_functional instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _make_dm21_like_functional(*args, **kwargs)
```

Remove `"make_dm21_like_functional"` from `__all__`.

- [ ] **Step 2: Remove legacy names from Neural XC package exports**

In `src/td_graddft/neural_xc/__init__.py`, remove imports from `.base`:

```python
DensityNeuralXCFunctional,
NeuralXCFunctional,
PointwiseMLP,
default_lda_coefficient_inputs,
default_lda_energy_density_basis,
make_neural_lda_functional,
```

Keep `default_lda_*` only if another public test still requires them; otherwise remove them with the legacy constructors. Remove these strings from `__all__`:

```python
"DensityNeuralXCFunctional",
"NeuralXCFunctional",
"PointwiseMLP",
"default_lda_coefficient_inputs",
"default_lda_energy_density_basis",
"make_dm21_like_functional",
"make_neural_lda_functional",
```

- [ ] **Step 3: Remove top-level lazy exports**

In `src/td_graddft/__init__.py`, remove these keys from `_EXPORTS`:

```python
"DensityNeuralXCFunctional": "neural_xc",
"NeuralXCFunctional": "neural_xc",
"PointwiseMLP": "neural_xc",
```

If `make_dm21_like_functional` or `make_neural_lda_functional` appears in `_EXPORTS`, remove those keys too.

- [ ] **Step 4: Make `td_graddft.pyscf_bridge` fail hard**

Prefer deleting `src/td_graddft/pyscf_bridge.py`. If package metadata or tests assume the file exists during this phase, replace its body with:

```python
raise ImportError(
    "td_graddft.pyscf_bridge has been removed. "
    "Use td_graddft.reference_legacy for explicit PySCF-backed builders "
    "or td_graddft.reference for strict reference builders."
)
```

- [ ] **Step 5: Verify GREEN for public removal**

Run:

```bash
pytest tests/test_neural_xc_public_api.py::test_legacy_neural_xc_public_constructors_are_removed tests/test_pyscf_style_namespace.py::test_top_level_removes_legacy_neural_xc_exports tests/test_tools_public_api_usage.py::test_pyscf_bridge_module_is_removed_from_public_api -q
```

Expected: PASS.

- [ ] **Step 6: Verification checkpoint**

Run:

```bash
pytest tests/test_neural_xc_public_api.py tests/test_pyscf_style_namespace.py tests/test_tools_public_api_usage.py -q
```

Expected: PASS or only failures that point to remaining old-name assertions/imports. Fix those assertions/imports in the same files before moving to Task 3.

No commit is possible in this workspace because `git status` reports `fatal: not a git repository`.

---

### Task 3: Migrate Water Smoke Test To Supported Training Route

**Files:**
- Modify: `tests/test_water_smoke.py`

- [ ] **Step 1: Write/replace smoke test using current API**

Replace imports:

```python
from td_graddft.neural_xc import DensityNeuralXCFunctional, PointwiseMLP
from td_graddft.pyscf_bridge import restricted_reference_from_pyscf
```

with:

```python
from td_graddft import neural_xc
from td_graddft.reference_legacy import restricted_reference_from_pyscf
```

Change `_make_water_reference()` to request local-HF auxiliary data:

```python
return restricted_reference_from_pyscf(
    mf,
    compute_local_hfx_features=True,
    compute_local_hfx_aux=True,
    hfx_omega_values=(0.0, 0.4),
)
```

Replace `_make_trainable_functional()` with:

```python
def _make_trainable_functional():
    return neural_xc.Functional(
        architecture="residual",
        semilocal_xc=("gga_x_pbe", "gga_c_pbe"),
        hidden_dims=(8, 8),
        energy_mode="graddft_coeff_basis_hf_pt2_heads",
        include_pt2_channel=False,
        name="water_smoke_xc",
    )
```

Reduce the optimization assertion to finite loss/energy behavior rather than exact overfit of the old one-parameter toy model:

```python
initial_energy = predict_ground_state_total_energy(state.params, functional, molecule)
for _ in range(5):
    state, metrics = train_step(state, datum)

predicted_energy = predict_ground_state_total_energy(state.params, functional, molecule)
excitations = predict_excitation_energies(
    state.params,
    functional,
    molecule,
    nstates=3,
)

assert jnp.isfinite(initial_energy)
assert jnp.isfinite(predicted_energy)
assert jnp.isfinite(metrics["loss"])
assert excitations.shape == (3,)
assert jnp.all(jnp.isfinite(excitations))
assert jnp.all(excitations > 0.0)
```

- [ ] **Step 2: Verify RED or migration failure**

Run:

```bash
pytest tests/test_water_smoke.py -q
```

Expected before Task 2 cleanup: old imports may still pass; after Task 2, this test must fail if any removed API remains referenced.

- [ ] **Step 3: Verify GREEN**

Run:

```bash
pytest tests/test_water_smoke.py tests/test_dm21_like.py::test_dm21_like_functional_trains_and_produces_excitation -q
```

Expected: PASS.

---

### Task 4: Add Training Payload Guards

**Files:**
- Modify: `src/td_graddft/training/config.py`
- Modify: `tests/test_neural_xc_public_api.py`

- [ ] **Step 1: Add failing PT2 guard test**

In `tests/test_neural_xc_public_api.py`, add a test next to `test_ground_state_datum_from_reference_requires_hfx_fields`:

```python
def test_ground_state_datum_from_reference_requires_pt2_fields_when_pt2_enabled():
    import jax.numpy as jnp
    from td_graddft import neural_xc
    from td_graddft.reference import GridReference, RestrictedMoleculeReference
    from td_graddft.training import GroundStateDatum

    grid = GridReference(coords=jnp.zeros((2, 3)), weights=jnp.ones((2,)))
    molecule = RestrictedMoleculeReference(
        ao=jnp.ones((2, 2)),
        ao_deriv1=jnp.ones((4, 2, 2)),
        grid=grid,
        density_matrix=jnp.eye(2),
        rdm1=jnp.eye(2),
        h1e=jnp.eye(2),
        rep_tensor=jnp.zeros((2, 2, 2, 2)),
        nuclear_repulsion=jnp.asarray(0.0),
        mo_coeff=jnp.eye(2),
        mo_occ=jnp.asarray([2.0, 0.0]),
        mo_energy=jnp.asarray([-0.5, 0.1]),
        mf_energy=jnp.asarray(-1.0),
        hfx_local=jnp.zeros((2, 2, 1)),
        hfx_nu=jnp.zeros((1, 2, 2, 2)),
    )
    functional = neural_xc.Functional(
        hidden_dims=(8,),
        include_pt2_channel=True,
        pt2_channel_mode="local_exact",
    )

    with pytest.raises(ValueError, match="compute_local_pt2_features=True"):
        GroundStateDatum.from_reference(
            molecule,
            target_total_energy=-1.0,
            functional=functional,
        )
```

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest tests/test_neural_xc_public_api.py::test_ground_state_datum_from_reference_requires_pt2_fields_when_pt2_enabled -q
```

Expected: FAIL because PT2 field validation is missing or the error message does not mention `compute_local_pt2_features=True`.

- [ ] **Step 3: Implement PT2 validation**

In `src/td_graddft/training/config.py`, update the reference validation used by `GroundStateDatum.from_reference(...)`. Add logic equivalent to:

```python
include_pt2 = bool(getattr(functional, "include_pt2_channel", False))
if include_pt2:
    has_local_pt2 = (
        getattr(reference, "pt2_local", None) is not None
        or getattr(reference, "pt2_energy_density", None) is not None
        or getattr(reference, "local_pt2", None) is not None
    )
    if not has_local_pt2:
        raise ValueError(
            "Neural XC training with include_pt2_channel=True requires local PT2 "
            "features. Build the reference with compute_local_pt2_features=True."
        )
```

Use the actual PT2 field names present on `RestrictedMoleculeReference`; inspect `src/td_graddft/reference.py` before editing and keep the guard limited to those concrete attributes.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
pytest tests/test_neural_xc_public_api.py::test_ground_state_datum_from_reference_requires_hfx_fields tests/test_neural_xc_public_api.py::test_ground_state_datum_from_reference_requires_pt2_fields_when_pt2_enabled -q
```

Expected: PASS.

---

### Task 5: Remove Deprecated Imports From Tests, Examples, And Docs

**Files:**
- Modify: `README.md`
- Modify tests/examples/tools that still import `td_graddft.pyscf_bridge`
- Modify public API usage tests if their assertions name removed APIs

- [ ] **Step 1: Find remaining deprecated imports**

Run:

```bash
rg -n "td_graddft\\.pyscf_bridge|from td_graddft\\.pyscf_bridge|make_dm21_like_functional|DensityNeuralXCFunctional|make_neural_lda_functional|PointwiseMLP" README.md src tests examples tools
```

Expected: hits in docs/tests/examples before cleanup.

- [ ] **Step 2: Replace PySCF bridge imports**

For each Python file importing from `td_graddft.pyscf_bridge`, replace:

```python
from td_graddft.pyscf_bridge import restricted_reference_from_pyscf
```

with:

```python
from td_graddft.reference_legacy import restricted_reference_from_pyscf
```

and replace:

```python
from td_graddft.pyscf_bridge import unrestricted_reference_from_pyscf
```

with:

```python
from td_graddft.reference_legacy import unrestricted_reference_from_pyscf
```

- [ ] **Step 3: Update README Neural XC section**

In `README.md`, remove the paragraph:

```markdown
The older `make_dm21_like_functional(...)` name remains as a deprecated
compatibility wrapper; new code should use `neural_xc.Functional(...)` or
`neural_xc.make_functional(...)`.
```

Add a supported-reference note:

```markdown
Neural XC training data that uses HF or PT2 channels should be built with
`compute_local_hfx_features=True`, `compute_local_hfx_aux=True`, and
`compute_local_pt2_features=True` when `include_pt2_channel=True`.
```

- [ ] **Step 4: Verify usage tests**

Run:

```bash
pytest tests/test_examples_public_api_usage.py tests/test_workflows_public_api_usage.py tests/test_tools_public_api_usage.py -q
```

Expected: PASS.

- [ ] **Step 5: Verify no public deprecated hits remain**

Run:

```bash
rg -n "td_graddft\\.pyscf_bridge|from td_graddft\\.pyscf_bridge|make_dm21_like_functional|DensityNeuralXCFunctional|make_neural_lda_functional|PointwiseMLP" README.md src/td_graddft/neural_xc src/td_graddft/__init__.py tests examples tools
```

Expected: no hits except private base-module definitions if `src/td_graddft/neural_xc/base/functional.py` is still intentionally retained.

---

### Task 6: End-To-End Training Verification

**Files:**
- No edits unless verification exposes a regression.

- [ ] **Step 1: Run jax_libxc and preset checks**

Run:

```bash
pytest tests/test_jax_libxc.py tests/test_vendored_jax_xc_backend.py tests/test_neural_xc_presets.py -q
```

Expected: PASS.

- [ ] **Step 2: Run focused Neural XC training checks**

Run:

```bash
pytest tests/test_dm21_like.py::test_dm21_like_functional_trains_and_produces_excitation tests/test_dm21_like.py::test_libxc_semilocal_module_supports_common_exchange_and_correlation_components tests/test_dm21_like.py::test_coefficient_prior_penalty_is_reported_and_nonnegative -q
```

Expected: PASS.

- [ ] **Step 3: Run local-HF response gradient check**

Run:

```bash
pytest tests/test_local_hf_test_module.py::test_local_hf_khh_response_wrapper_preserves_tda_parameter_gradients -q
```

Expected: PASS.

- [ ] **Step 4: Run self-consistent training check**

Run:

```bash
pytest tests/test_differentiable_scf.py::test_self_consistent_training_mode_produces_finite_loss_and_energy -q
```

Expected: PASS.

- [ ] **Step 5: Final grep check**

Run:

```bash
rg -n "td_graddft\\.pyscf_bridge|make_dm21_like_functional|DensityNeuralXCFunctional|make_neural_lda_functional" README.md src tests examples tools
```

Expected: no public-surface hits; private retained implementation files must be explicitly reviewed before final report.

No commit is possible in this workspace because `git status` reports `fatal: not a git repository`.

---

## Self-Review

- Spec coverage: Tasks 1-2 remove public APIs, Task 3 migrates the training smoke path, Task 4 adds fail-fast guards, Task 5 updates docs/examples/tools, and Task 6 verifies supported training paths.
- Placeholder scan: no TBD/TODO placeholders are present.
- Type consistency: the plan uses `neural_xc.Functional(...)`, `restricted_reference_from_pyscf(... compute_local_hfx_features=True, compute_local_hfx_aux=True ...)`, `include_pt2_channel`, and `pt2_channel_mode` consistently with the current codebase.
