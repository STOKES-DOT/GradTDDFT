# Neural XC and RSH Training API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add thin public API facades for unified `neural_xc` naming and separate RSH/NeuralXC training routes.

**Architecture:** Keep the numerical implementation in existing modules. Add small facade modules that normalize names, construct existing functional objects, expose separate trainer classes, and return a shared result container.

**Tech Stack:** Python dataclasses, pytest, existing `neural_xc.dm21`, `nn_rsh`, and `training` modules.

---

### Task 1: Unified Neural XC Public Construction

**Files:**
- Test: `tests/test_neural_xc_public_api.py`
- Create: `src/td_graddft/neural_xc/api.py`
- Modify: `src/td_graddft/neural_xc/__init__.py`

- [ ] **Step 1: Write failing tests**

Test that `neural_xc.Functional(...)` and `neural_xc.make_functional(...)` create the existing neural XC functional type, with `architecture="residual"` mapped to `network_architecture="graddft_residual"`.

- [ ] **Step 2: Verify the tests fail**

Run: `pytest tests/test_neural_xc_public_api.py::test_neural_xc_functional_public_constructor_uses_unified_name -q`

Expected: FAIL because the new public constructor does not exist.

- [ ] **Step 3: Implement the facade**

Create `neural_xc/api.py` with:

- `Functional(...)`
- `make_functional(...)`
- `make_dm21_like_functional(...)` compatibility wrapper with `DeprecationWarning`

- [ ] **Step 4: Verify Neural XC public API tests pass**

Run: `pytest tests/test_neural_xc_public_api.py -q`

Expected: PASS.

### Task 2: RSH Public Construction

**Files:**
- Test: `tests/test_neural_xc_public_api.py`
- Create: `src/td_graddft/nn_rsh/api.py`
- Modify: `src/td_graddft/nn_rsh/__init__.py`

- [ ] **Step 1: Write failing tests**

Test that `nn_rsh.RSH("lc-wpbe").trainable(params=("omega", "alpha", "beta"))` returns an existing `TrainableRSHFunctional` with a preset-derived template.

- [ ] **Step 2: Verify the tests fail**

Run: `pytest tests/test_neural_xc_public_api.py::test_rsh_public_constructor_builds_trainable_functional -q`

Expected: FAIL because `nn_rsh.RSH` does not exist.

- [ ] **Step 3: Implement the facade**

Create `nn_rsh/api.py` with:

- `RSH(name, omega_source="canonical")`
- `RSH.trainable(params=("omega", "alpha", "beta"), local_xc_spec="pbe", hidden_dims=())`

Reject parameter names outside `omega`, `alpha`, and `beta`.

- [ ] **Step 4: Verify RSH public API tests pass**

Run: `pytest tests/test_neural_xc_public_api.py::test_rsh_public_constructor_builds_trainable_functional -q`

Expected: PASS.

### Task 3: Shared Training Result and Separate Trainer Classes

**Files:**
- Test: `tests/test_neural_xc_public_api.py`
- Create: `src/td_graddft/training/results.py`
- Create: `src/td_graddft/training/neural_xc_trainer.py`
- Create: `src/td_graddft/training/rsh_optimizer.py`
- Modify: `src/td_graddft/training/__init__.py`

- [ ] **Step 1: Write failing tests**

Test:

- `training.TrainingResult` exposes `functional`, `params`, `history`, and `final_metrics`.
- `training.NeuralXCTrainer(...).kernel(steps=0)` returns Neural XC history keys.
- `training.RSHOptimizer(...).kernel(steps=0)` returns RSH history keys.
- The two trainer classes are distinct.

- [ ] **Step 2: Verify the tests fail**

Run: `pytest tests/test_neural_xc_public_api.py::test_training_result_and_separate_trainers_expose_expected_history_keys -q`

Expected: FAIL because the result and trainer classes do not exist.

- [ ] **Step 3: Implement the trainer facades**

Implement:

- `TrainingResult`
- `NeuralXCTrainer`
- `RSHOptimizer`

For this API pass, `kernel(steps=0)` returns an initialized empty history. `kernel(steps>0)` raises a clear `NotImplementedError` until the public trainer is wired to concrete training data conversion.

- [ ] **Step 4: Verify trainer facade tests pass**

Run: `pytest tests/test_neural_xc_public_api.py -q`

Expected: PASS.

### Task 4: Focused Regression

**Files:**
- Test: existing namespace and functional tests.

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest tests/test_neural_xc_public_api.py tests/test_neural_xc.py tests/test_nn_rsh_namespace.py tests/test_training.py::test_train_step_accepts_custom_loss_function -q
```

Expected: PASS.

