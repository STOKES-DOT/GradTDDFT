# PySCF-Style Ground-State API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a concise PySCF-style ground-state API for `td_graddft.gto.M`, `td_graddft.scf.RKS`, and `td_graddft.scf.UKS`.

**Architecture:** Implement a thin facade over the existing molecule parser and RKS/UKS reference builders. Do not add a new SCF numerical route. Keep defaults aligned with the production path: `libcint`, `full` no-DF J/K, and analytic geometry gradients.

**Tech Stack:** Python dataclasses, JAX arrays, pytest, existing `td_graddft.reference` builders.

---

### Task 1: Molecule Facade

**Files:**
- Test: `tests/test_pyscf_style_ground_state_api.py`
- Create: `src/td_graddft/gto/mole.py`
- Modify: `src/td_graddft/gto/__init__.py`

- [ ] **Step 1: Write the failing molecule API test**

```python
from td_graddft import gto


def test_gto_m_stores_pyscf_style_molecule_fields():
    mol = gto.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis="sto-3g",
        unit="Angstrom",
        charge=0,
        spin=0,
        cart=True,
        verbose=0,
    )

    assert mol.atom.startswith("O")
    assert mol.basis == "sto-3g"
    assert mol.unit == "Angstrom"
    assert mol.charge == 0
    assert mol.spin == 0
    assert mol.cart is True
    assert mol.verbose == 0
    assert mol.nelectron == 10
    assert mol.to_spec().symbols == ("O", "H", "H")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_pyscf_style_ground_state_api.py::test_gto_m_stores_pyscf_style_molecule_fields -q`

Expected: FAIL because `td_graddft.gto.M` is not defined.

- [ ] **Step 3: Implement `gto.M`**

Create `src/td_graddft/gto/mole.py` with a lightweight frozen dataclass:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..data.molecule import MoleculeSpec, parse_molecule_spec


@dataclass(frozen=True)
class Mole:
    atom: Any
    basis: Any
    unit: str = "Angstrom"
    charge: int = 0
    spin: int = 0
    cart: bool = True
    verbose: int = 0

    def to_spec(self) -> MoleculeSpec:
        return parse_molecule_spec(
            self.atom,
            unit=self.unit,
            charge=self.charge,
            spin=self.spin,
        )

    @property
    def nelectron(self) -> int:
        return self.to_spec().nelectron


def M(*args: Any, **kwargs: Any) -> Mole:
    return Mole(*args, **kwargs)
```

Modify `src/td_graddft/gto/__init__.py` to export `M` and `Mole`.

- [ ] **Step 4: Run the molecule API test**

Run: `pytest tests/test_pyscf_style_ground_state_api.py::test_gto_m_stores_pyscf_style_molecule_fields -q`

Expected: PASS.

### Task 2: SCF Facade Defaults and Builder Wiring

**Files:**
- Test: `tests/test_pyscf_style_ground_state_api.py`
- Create: `src/td_graddft/scf/facade.py`
- Modify: `src/td_graddft/scf/__init__.py`

- [ ] **Step 1: Write failing tests for RKS/UKS defaults and reference builder calls**

```python
import types

from td_graddft import gto, scf


def test_rks_kernel_calls_existing_restricted_reference_builder(monkeypatch):
    captured = {}

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            mf_energy=-76.0,
            mo_energy="mo_energy",
            mo_coeff="mo_coeff",
            mo_occ="mo_occ",
        )

    monkeypatch.setattr("td_graddft.scf.facade.restricted_reference_from_spec_with_jax_rks", fake_builder)

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    mf = scf.RKS(mol, xc="pbe")
    energy = mf.kernel()

    assert energy == -76.0
    assert mf.e_tot == -76.0
    assert mf.reference.mf_energy == -76.0
    assert mf.mo_energy == "mo_energy"
    assert mf.mo_coeff == "mo_coeff"
    assert mf.mo_occ == "mo_occ"
    assert mf.converged is True
    assert captured["atom"].symbols == ("H", "H")
    assert captured["basis"] == "sto-3g"
    assert captured["xc_spec"] == "pbe"
    assert captured["integral_backend"] == "libcint"
    assert captured["libcint_geometry_grad_policy"] == "analytic"
    assert captured["rks_config"].jk_backend == "full"


def test_uks_kernel_calls_existing_unrestricted_reference_builder(monkeypatch):
    captured = {}

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            mf_energy=-39.0,
            mo_energy="mo_energy",
            mo_coeff="mo_coeff",
            mo_occ="mo_occ",
        )

    monkeypatch.setattr("td_graddft.scf.facade.unrestricted_reference_from_spec_with_jax_uks", fake_builder)

    mol = gto.M(atom="O 0 0 0", basis="sto-3g", spin=2)
    mf = scf.UKS(mol, xc="pbe")
    energy = mf.kernel()

    assert energy == -39.0
    assert captured["atom"].symbols == ("O",)
    assert captured["basis"] == "sto-3g"
    assert captured["xc_spec"] == "pbe"
    assert captured["integral_backend"] == "libcint"
    assert captured["libcint_geometry_grad_policy"] == "analytic"
    assert captured["uks_config"].max_cycle == mf.max_cycle
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_pyscf_style_ground_state_api.py::test_rks_kernel_calls_existing_restricted_reference_builder tests/test_pyscf_style_ground_state_api.py::test_uks_kernel_calls_existing_unrestricted_reference_builder -q`

Expected: FAIL because `td_graddft.scf.RKS` and `td_graddft.scf.UKS` are not facade classes yet.

- [ ] **Step 3: Implement the SCF facade**

Create `src/td_graddft/scf/facade.py` with:

- `RKS(mol, xc="pbe")`
- `UKS(mol, xc="pbe")`
- shared fields: `xc`, `conv_tol`, `max_cycle`, `damp`, `level_shift`, `integral_backend`, `geometry_grad_policy`, `grid_ao_backend`, `execution_device`
- RKS field: `jk_backend`
- `kernel()` calls the matching existing reference builder.
- `run()` calls `kernel()` and returns `self`.
- `_sync_from_reference()` copies `e_tot`, `mo_energy`, `mo_coeff`, `mo_occ`, `reference`, `converged`.

Modify `src/td_graddft/scf/__init__.py` to export `RKS` and `UKS` in addition to existing low-level symbols.

- [ ] **Step 4: Run the SCF facade tests**

Run: `pytest tests/test_pyscf_style_ground_state_api.py::test_rks_kernel_calls_existing_restricted_reference_builder tests/test_pyscf_style_ground_state_api.py::test_uks_kernel_calls_existing_unrestricted_reference_builder -q`

Expected: PASS.

### Task 3: Convenience Methods and Gradients

**Files:**
- Test: `tests/test_pyscf_style_ground_state_api.py`
- Modify: `src/td_graddft/scf/facade.py`

- [ ] **Step 1: Write failing tests for `run`, `density_fit`, `direct_scf`, and gradients**

```python
import jax.numpy as jnp

from td_graddft import gto, scf


def test_rks_run_and_backend_helpers(monkeypatch):
    monkeypatch.setattr(
        "td_graddft.scf.facade.restricted_reference_from_spec_with_jax_rks",
        lambda **kwargs: type("Ref", (), {
            "mf_energy": -1.0,
            "mo_energy": None,
            "mo_coeff": None,
            "mo_occ": None,
        })(),
    )

    mf = scf.RKS(gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"))

    assert mf.density_fit() is mf
    assert mf.jk_backend == "df"
    assert mf.direct_scf() is mf
    assert mf.jk_backend == "direct"
    assert mf.run() is mf
    assert mf.e_tot == -1.0


def test_nuc_grad_method_returns_geometry_gradient(monkeypatch):
    calls = {"count": 0}

    def fake_energy(mf, coords_bohr):
        calls["count"] += 1
        return jnp.sum(coords_bohr * coords_bohr)

    monkeypatch.setattr("td_graddft.scf.facade._energy_for_coords", fake_energy)

    mol = gto.M(atom=[("H", jnp.array([0.0, 0.0, 0.0])), ("H", jnp.array([0.0, 0.0, 1.0]))], basis="sto-3g")
    mf = scf.RKS(mol)
    grad = mf.nuc_grad_method().kernel()

    assert grad.shape == (2, 3)
    assert calls["count"] >= 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_pyscf_style_ground_state_api.py::test_rks_run_and_backend_helpers tests/test_pyscf_style_ground_state_api.py::test_nuc_grad_method_returns_geometry_gradient -q`

Expected: FAIL because helper methods and gradient facade are missing.

- [ ] **Step 3: Implement helper methods and gradient object**

Add:

- `density_fit()` on RKS sets `jk_backend = "df"` and returns `self`.
- `direct_scf()` on RKS sets `jk_backend = "direct"` and returns `self`.
- `density_fit()` and `direct_scf()` on UKS raise `NotImplementedError` until UKS DF/direct exists in the core solver.
- `_NuclearGradient.kernel()` computes `jax.grad` over `coords_bohr` using a new `MoleculeSpec` with the same symbols, charges, charge, spin, and unit.

- [ ] **Step 4: Run convenience and gradient tests**

Run: `pytest tests/test_pyscf_style_ground_state_api.py::test_rks_run_and_backend_helpers tests/test_pyscf_style_ground_state_api.py::test_nuc_grad_method_returns_geometry_gradient -q`

Expected: PASS.

### Task 4: Smoke Test and Namespace Regression

**Files:**
- Test: `tests/test_pyscf_style_ground_state_api.py`
- Modify: `src/td_graddft/__init__.py`

- [ ] **Step 1: Write smoke tests**

```python
import importlib

from td_graddft import gto, scf


def test_top_level_import_exposes_gto_and_scf_modules():
    assert importlib.import_module("td_graddft.gto") is gto
    assert importlib.import_module("td_graddft.scf") is scf


def test_real_rks_kernel_smoke_sto3g_h2():
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    mf = scf.RKS(mol, xc="pbe")
    mf.max_cycle = 4
    energy = mf.kernel()

    assert energy < 0.0
    assert mf.reference is not None
    assert mf.mo_coeff is not None
```

- [ ] **Step 2: Run the smoke tests**

Run: `pytest tests/test_pyscf_style_ground_state_api.py::test_top_level_import_exposes_gto_and_scf_modules tests/test_pyscf_style_ground_state_api.py::test_real_rks_kernel_smoke_sto3g_h2 -q`

Expected: PASS.

- [ ] **Step 3: Run focused regression tests**

Run: `pytest tests/test_pyscf_style_ground_state_api.py tests/test_pyscf_style_namespace.py tests/test_reference_uks.py tests/test_density_fitting_rks.py -q`

Expected: PASS.

