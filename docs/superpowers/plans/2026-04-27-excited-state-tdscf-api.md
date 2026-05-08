# Excited-State TD-SCF API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a concise PySCF-style excited-state API on top of the existing TDDFT/TDA solvers.

**Architecture:** Keep `src/td_graddft/tddft/` as the numerical implementation layer. Add `src/td_graddft/tdscf/api.py` as a user-facing facade that accepts either a ground-state SCF facade object or a raw molecule reference, dispatches restricted/unrestricted solvers, stores PySCF-like result fields, and exposes spectra helpers. Add `mf.TDA()` and `mf.TDDFT()` shortcuts to the existing SCF facade.

**Tech Stack:** Python dataclasses/classes, JAX arrays, existing `td_graddft.tddft` solvers, existing `td_graddft.spectra` helpers, pytest.

---

### Task 1: Public TD-SCF Facade Tests

**Files:**
- Create: `tests/test_pyscf_style_excited_state_api.py`
- Modify: none

- [ ] **Step 1: Write failing tests**

Create tests that import `tdscf.TDA` and `tdscf.TDDFT`, monkeypatch the low-level solver classes, and verify:

```python
from td_graddft import tdscf

td = tdscf.TDA(mf)
td.nstates = 2
result = td.kernel()
assert td.e is result.excitation_energies
assert td.e_ev.shape == result.excitation_energies.shape
assert td.oscillator_strength().shape == result.excitation_energies.shape
```

Also verify unrestricted references dispatch to `UnrestrictedTDA` and `UnrestrictedCasidaTDDFT`, and `mf.TDA()` / `mf.TDDFT()` return facade objects.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
pytest tests/test_pyscf_style_excited_state_api.py -q
```

Expected: fail because `td_graddft.tdscf` does not yet define `TDA` or `TDDFT`.

### Task 2: TD-SCF Facade Implementation

**Files:**
- Create: `src/td_graddft/tdscf/api.py`
- Modify: `src/td_graddft/tdscf/__init__.py`

- [ ] **Step 1: Implement facade classes**

Add `_BaseTD`, `TDA`, and `TDDFT`. `_BaseTD` resolves the reference, chooses default XC from a ground-state object when possible, stores `result`, `e`, `e_ev`, `xy`, and forwards `oscillator_strength()` / `transition_dipole()` to `td_graddft.spectra`.

- [ ] **Step 2: Export public names**

Export `TDA` and `TDDFT` from `tdscf/__init__.py` while keeping the existing low-level re-exports.

- [ ] **Step 3: Run tests to verify GREEN**

Run:

```bash
pytest tests/test_pyscf_style_excited_state_api.py -q
```

Expected: pass.

### Task 3: SCF Shortcut Methods

**Files:**
- Modify: `src/td_graddft/scf/facade.py`
- Test: `tests/test_pyscf_style_excited_state_api.py`

- [ ] **Step 1: Add shortcut tests**

Verify:

```python
td = mf.TDA()
assert isinstance(td, tdscf.TDA)
td = mf.TDDFT()
assert isinstance(td, tdscf.TDDFT)
```

- [ ] **Step 2: Add `_BaseKS.TDA()` and `_BaseKS.TDDFT()`**

Methods import `td_graddft.tdscf` inside the method body and return the matching facade bound to `self`.

- [ ] **Step 3: Run focused regression**

Run:

```bash
pytest tests/test_pyscf_style_excited_state_api.py tests/test_pyscf_style_ground_state_api.py tests/test_tddft.py -q
```

Expected: pass.
