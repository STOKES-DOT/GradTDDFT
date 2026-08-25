# Unrestricted TDDFT/TDA Semilocal Response Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add complete matrix-free spin-polarized LDA/GGA response to unrestricted TDA and full TDDFT, including neural-functional differentiation, PySCF-level excitation energies, and oscillator strengths.

**Architecture:** A new unrestricted-only module projects alpha/beta occupied-virtual amplitudes to spin-grid response variables and applies a pointwise JAX HVP. Traditional and neural functionals expose the same `spin_grid_response_hvp` contract. `unrestricted.py` retains solver assembly while all restricted response source remains unchanged.

**Tech Stack:** Python, JAX, jax-xc, PySCF, NumPy, pytest, Davidson/Casida solvers.

---

## File Structure

Create:

- `src/td_graddft/tddft/_unrestricted_semilocal_response.py`: unrestricted point HVP, transition projection, and traditional functional wrapper.
- `tests/test_unrestricted_semilocal_response.py`: contract, projection, JIT, and edge-case tests.
- `tests/test_unrestricted_tddft_pyscf_compare.py`: H2+, OH, and O2 accuracy plus oscillator strengths.

Modify only unrestricted behavior in:

- `src/td_graddft/tddft/unrestricted.py`: resolve unrestricted XC and use the spin-grid HVP action.
- `src/td_graddft/tdscf/api.py`: choose the unrestricted wrapper only for unrestricted molecules.
- `src/td_graddft/neural_xc/components.py`: provide polarized semilocal channels.
- `src/td_graddft/neural_xc/model.py`: evaluate unrestricted channels and expose a full unrestricted point HVP.
- `src/td_graddft/neural_xc/binding.py`: bind `spin_grid_response_hvp` and remove the density-only unrestricted Hessian path.
- `tests/test_unrestricted_spin_kernel.py`: replace scalar GGA expectations with HVP expectations while preserving LDA compatibility.
- `tests/test_neural_xc_runtime.py`: validate the neural spin HVP and parameter gradients.
- `tests/test_pyscf_style_excited_state_api.py`: validate string-XC dispatch for unrestricted sources.

Frozen files that must remain byte-for-byte unchanged:

- `src/td_graddft/tddft/response.py`
- `src/td_graddft/scf/rks.py`
- restricted solver source and restricted tests

### Task 1: Establish the unrestricted spin-grid HVP contract

**Files:**
- Create: `tests/test_unrestricted_semilocal_response.py`
- Create: `src/td_graddft/tddft/_unrestricted_semilocal_response.py`

- [ ] **Step 1: Write the failing traditional-functional contract test**

```python
import jax
import jax.numpy as jnp
import pytest
from types import SimpleNamespace

from td_graddft.tddft._unrestricted_semilocal_response import (
    UnrestrictedSemilocalResponseFunctional,
)


@pytest.fixture
def open_shell_molecule():
    ao = jnp.asarray([[1.0, 0.2], [0.8, -0.3], [0.6, 0.4]], dtype=jnp.float64)
    ao_deriv1 = jnp.stack([ao, 0.1 * ao, -0.2 * ao, 0.05 * ao], axis=0)
    mo_coeff = jnp.stack([jnp.eye(2), jnp.eye(2)], axis=0)
    mo_occ = jnp.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.float64)
    rdm1 = jnp.asarray(
        [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
        dtype=jnp.float64,
    )
    return SimpleNamespace(
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid=SimpleNamespace(
            weights=jnp.asarray([0.5, 0.3, 0.2], dtype=jnp.float64),
            coords=jnp.zeros((3, 3), dtype=jnp.float64),
        ),
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=jnp.asarray([[-0.6, 0.2], [-0.4, 0.3]], dtype=jnp.float64),
        rdm1=rdm1,
        nocc_alpha=1,
        nocc_beta=0,
        exact_exchange_fraction=0.2,
        rep_tensor=jnp.zeros((2, 2, 2, 2), dtype=jnp.float64),
    )


def test_b3lyp_unrestricted_functional_exposes_full_gga_hvp(open_shell_molecule):
    functional = UnrestrictedSemilocalResponseFunctional("b3lyp")
    ngrids = open_shell_molecule.ao.shape[0]
    tangent_a = jnp.zeros((2, 4, ngrids)).at[:, 1, :].set(0.1)
    tangent_b = jnp.zeros((2, 4, ngrids))

    response_a, response_b = functional.spin_grid_response_hvp(
        open_shell_molecule,
        tangent_a,
        tangent_b,
    )

    assert functional.response_feature_kind == "GGA"
    assert response_a.shape == tangent_a.shape
    assert response_b.shape == tangent_b.shape
    assert jnp.all(jnp.isfinite(response_a))
    assert jnp.any(jnp.abs(response_a) > 0.0)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
pytest -q tests/test_unrestricted_semilocal_response.py::test_b3lyp_unrestricted_functional_exposes_full_gga_hvp
```

Expected: collection fails because `_unrestricted_semilocal_response` does not exist.

- [ ] **Step 3: Implement the pointwise traditional HVP**

Create the new module with this public contract and variable order:

```python
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..features import grid_features_with_spin_gradients_for_molecule
from ..xc_backend.jax_libxc import (
    eval_xc_energy_density_unrestricted_from_density_gradients,
    hybrid_coeff,
    parse_xc,
    xc_type,
)


SpinGridHVP = Callable[[Any, Array, Array], tuple[Array, Array]]


@lru_cache(maxsize=64)
def _point_spin_hvp(spec: str, feature_kind: str):
    kind = str(feature_kind).upper()

    def point_energy(values):
        rho_a = values[0]
        rho_b = values[1]
        zero = jnp.zeros((3,), dtype=values.dtype)
        grad_a = zero if kind == "LDA" else values[2:5]
        grad_b = zero if kind == "LDA" else values[5:8]
        return eval_xc_energy_density_unrestricted_from_density_gradients(
            spec, rho_a, rho_b, grad_a, grad_b
        )

    gradient = jax.grad(point_energy)

    def point_hvp(values, tangent):
        return jax.jvp(gradient, (values,), (tangent,))[1]

    return jax.jit(jax.vmap(jax.vmap(point_hvp)))


@dataclass(frozen=True)
class UnrestrictedSemilocalResponseFunctional:
    xc_spec: str

    def __post_init__(self):
        parse_xc(self.xc_spec)
        kind = str(xc_type(self.xc_spec)).upper()
        if kind not in {"LDA", "GGA"}:
            raise NotImplementedError(
                "Unrestricted semilocal response supports LDA/GGA only."
            )
        object.__setattr__(self, "exact_exchange_fraction", float(hybrid_coeff(self.xc_spec)))
        object.__setattr__(self, "response_feature_kind", kind)

    def spin_grid_response_hvp(self, molecule, tangent_a, tangent_b):
        features, grad_a, grad_b = grid_features_with_spin_gradients_for_molecule(molecule)
        tangent_a = jnp.asarray(tangent_a)
        tangent_b = jnp.asarray(tangent_b)
        if self.response_feature_kind == "LDA":
            base = jnp.stack([features.rho_a, features.rho_b], axis=-1)
            tangent = jnp.stack([tangent_a[:, 0], tangent_b[:, 0]], axis=-1)
        else:
            base = jnp.concatenate(
                [features.rho_a[:, None], features.rho_b[:, None], grad_a, grad_b],
                axis=-1,
            )
            tangent = jnp.concatenate(
                [
                    tangent_a[:, 0:1, :],
                    tangent_b[:, 0:1, :],
                    tangent_a[:, 1:4, :],
                    tangent_b[:, 1:4, :],
                ],
                axis=1,
            ).transpose(0, 2, 1)
        base_batch = jnp.broadcast_to(base, (tangent.shape[0],) + base.shape)
        response = _point_spin_hvp(self.xc_spec, self.response_feature_kind)(
            base_batch, tangent
        ).transpose(0, 2, 1)
        if self.response_feature_kind == "LDA":
            return response[:, 0:1], response[:, 1:2]
        return (
            jnp.concatenate([response[:, 0:1], response[:, 2:5]], axis=1),
            jnp.concatenate([response[:, 1:2], response[:, 5:8]], axis=1),
        )
```

Keep raw HVP values visible to tests; do not add `nan_to_num` here.

- [ ] **Step 4: Run the contract test and the empty-spin boundary test**

Run:

```bash
pytest -q tests/test_unrestricted_semilocal_response.py
```

Expected: contract and finite-HVP tests pass for B88, PBE, LYP, and B3LYP.

- [ ] **Step 5: Commit**

```bash
git add src/td_graddft/tddft/_unrestricted_semilocal_response.py tests/test_unrestricted_semilocal_response.py
git commit -m "feat: add unrestricted semilocal grid HVP"
```

### Task 2: Add alpha/beta MO-to-grid projection without dense Hessians

**Files:**
- Modify: `src/td_graddft/tddft/_unrestricted_semilocal_response.py`
- Modify: `tests/test_unrestricted_semilocal_response.py`

- [ ] **Step 1: Write failing projection tests**

Add tests that compare the factorized projection to an explicit transition-density calculation and prove a pure gradient tangent affects a GGA response:

```python
def test_spin_transition_projection_matches_explicit_ao_derivative_formula(open_shell_molecule):
    factors = build_spin_transition_factors(
        open_shell_molecule,
        open_shell_molecule.mo_coeff[0][:, :1],
        open_shell_molecule.mo_coeff[0][:, 1:],
        feature_kind="GGA",
        dtype=jnp.float64,
    )
    amplitudes = jnp.ones((1, 1, open_shell_molecule.mo_coeff.shape[-1] - 1))
    projected = project_spin_transition_to_grid(factors, amplitudes)
    assert projected.shape[1] == 4
    assert jnp.any(jnp.abs(projected[:, 1:]) > 0.0)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest -q tests/test_unrestricted_semilocal_response.py -k projection
```

Expected: fails because projection builders are missing.

- [ ] **Step 3: Reuse frozen restricted factor helpers read-only**

In the new module import, but do not modify, the existing helpers:

```python
from .response import (
    _project_grid_response_to_restricted_transition,
    _project_restricted_transition_to_grid,
    _restricted_response_factors,
)


def build_spin_transition_factors(molecule, orbo, orbv, *, feature_kind, dtype):
    return _restricted_response_factors(
        molecule, orbo, orbv, feature_kind=feature_kind, dtype=dtype
    )


def project_spin_transition_to_grid(factors, values):
    return _project_restricted_transition_to_grid(factors, values)


def project_grid_response_to_spin_transition(factors, values):
    return _project_grid_response_to_restricted_transition(factors, values)
```

Then add the action builder:

```python
def build_unrestricted_semilocal_response_action(
    molecule,
    orbo_a,
    orbv_a,
    orbo_b,
    orbv_b,
    response_hvp,
    *,
    feature_kind,
    dtype,
):
    weights = jnp.asarray(molecule.grid.weights, dtype=dtype)
    factors_a = build_spin_transition_factors(
        molecule, orbo_a, orbv_a, feature_kind=feature_kind, dtype=dtype
    )
    factors_b = build_spin_transition_factors(
        molecule, orbo_b, orbv_b, feature_kind=feature_kind, dtype=dtype
    )

    def action(alpha, beta):
        tangent_a = project_spin_transition_to_grid(factors_a, alpha)
        tangent_b = project_spin_transition_to_grid(factors_b, beta)
        response_a, response_b = response_hvp(molecule, tangent_a, tangent_b)
        weighted_a = jnp.asarray(response_a, dtype=dtype) * weights[None, None, :]
        weighted_b = jnp.asarray(response_b, dtype=dtype) * weights[None, None, :]
        return (
            project_grid_response_to_spin_transition(factors_a, weighted_a),
            project_grid_response_to_spin_transition(factors_b, weighted_b),
        )

    return action
```

Do not multiply by the restricted closed-shell factor of two; alpha and beta
transition densities are explicit separate channels.

- [ ] **Step 4: Verify projection, empty beta, JIT, and no dense Hessian**

Run:

```bash
pytest -q tests/test_unrestricted_semilocal_response.py
```

Expected: all tests pass and source inspection finds no `(ngrids, 8, 8)` allocation.

- [ ] **Step 5: Commit**

```bash
git add src/td_graddft/tddft/_unrestricted_semilocal_response.py tests/test_unrestricted_semilocal_response.py
git commit -m "feat: project unrestricted GGA response matrix free"
```

### Task 3: Wire the HVP into unrestricted TDA and full TDDFT

**Files:**
- Modify: `src/td_graddft/tddft/unrestricted.py`
- Modify: `tests/test_unrestricted_spin_kernel.py`

- [ ] **Step 1: Write RED tests for string XC and GGA action**

```python
def test_unrestricted_tda_accepts_b3lyp_string(reference):
    vind, diagonal, _, _ = build_unrestricted_tda_operator(reference, "b3lyp")
    result = vind(jnp.eye(diagonal.size))
    assert jnp.all(jnp.isfinite(result))


def test_unrestricted_gga_rejects_scalar_spin_kernel(reference):
    class InvalidGGA:
        response_feature_kind = "GGA"
        exact_exchange_fraction = 0.0

        def spin_local_kernel(self, rho_a, rho_b):
            return rho_a, rho_a * 0.0, rho_b

    with pytest.raises(ValueError, match="spin_grid_response_hvp"):
        build_unrestricted_tda_operator(reference, InvalidGGA())
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest -q tests/test_unrestricted_spin_kernel.py -k "b3lyp_string or rejects_scalar"
```

Expected: B3LYP fails with the current missing spin-kernel error.

- [ ] **Step 3: Resolve strings and prefer spin HVP**

In `unrestricted.py`, add imports from the new module and change only
`_build_unrestricted_response_operator_data`:

```python
if isinstance(resolved_xc, str):
    resolved_xc = UnrestrictedSemilocalResponseFunctional(resolved_xc)

spin_grid_response_hvp = getattr(resolved_xc, "spin_grid_response_hvp", None)
if callable(spin_grid_response_hvp):
    feature_kind = str(getattr(resolved_xc, "response_feature_kind", "LDA")).upper()
    xc_response_action_fn = build_unrestricted_semilocal_response_action(
        molecule,
        orbo_a,
        orbv_a,
        orbo_b,
        orbv_b,
        spin_grid_response_hvp,
        feature_kind=feature_kind,
        dtype=jnp.result_type(de_a, de_b),
    )
else:
    xc_response_action_fn = _unrestricted_grid_xc_response_action(
        molecule, resolved_xc, orbo_a, orbv_a, orbo_b, orbv_b,
        dtype=jnp.result_type(de_a, de_b),
    )
```

Update `_spin_resolved_kernel_on_grid` so scalar compatibility is accepted only
when `response_feature_kind == "LDA"`. GGA without HVP raises explicitly.

- [ ] **Step 4: Verify both TDA and full TDDFT operator paths**

Run:

```bash
pytest -q tests/test_unrestricted_spin_kernel.py tests/test_unrestricted_casida.py
```

Expected: existing LDA custom-kernel tests and new GGA HVP tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/td_graddft/tddft/unrestricted.py tests/test_unrestricted_spin_kernel.py
git commit -m "feat: use spin GGA HVP in unrestricted TDDFT"
```

### Task 4: Dispatch traditional XC correctly through the PySCF-style API

**Files:**
- Modify: `src/td_graddft/tdscf/api.py`
- Modify: `tests/test_pyscf_style_excited_state_api.py`

- [ ] **Step 1: Write a failing unrestricted API dispatch test**

```python
def test_tda_dispatches_b3lyp_to_unrestricted_response(uks_source):
    driver = TDA(uks_source, xc_functional="b3lyp", nstates=1)
    solver = driver._build_solver()
    assert isinstance(
        solver.xc_functional,
        UnrestrictedSemilocalResponseFunctional,
    )
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
pytest -q tests/test_pyscf_style_excited_state_api.py -k unrestricted_response
```

Expected: receives the restricted `SemilocalResponseFunctional`.

- [ ] **Step 3: Add unrestricted-only resolution**

Add a private method without changing `_resolved_xc_functional`:

```python
def _resolved_unrestricted_xc_functional(self):
    source = self.xc_functional
    if source is None:
        source = getattr(self.mf, "xc", None)
    if isinstance(source, str):
        return UnrestrictedSemilocalResponseFunctional(source)
    return source
```

Set `kwargs["xc_functional"]` to this value only inside
`_unrestricted_solver_kwargs`. Restricted kwargs continue using the existing
resolver unchanged.

- [ ] **Step 4: Verify dispatch and restricted regression**

Run:

```bash
pytest -q tests/test_pyscf_style_excited_state_api.py
```

Expected: unrestricted and all existing restricted dispatch tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/td_graddft/tdscf/api.py tests/test_pyscf_style_excited_state_api.py
git commit -m "feat: dispatch unrestricted semilocal response"
```

### Task 5: Make neural semilocal channels spin-polarized

**Files:**
- Modify: `src/td_graddft/neural_xc/components.py`
- Modify: `src/td_graddft/neural_xc/model.py`
- Modify: `tests/test_neural_xc_runtime.py`

- [ ] **Step 1: Write the failing polarized-channel test**

```python
def test_libxc_module_uses_polarized_channels_for_unrestricted_features(monkeypatch):
    calls = []

    def fake_unrestricted(spec, features, **kwargs):
        calls.append(spec)
        return features.rho_a - features.rho_b

    monkeypatch.setattr(components, "eval_xc_energy_density_unrestricted", fake_unrestricted)
    module = make_libxc_semilocal_module(("gga_x_b88", "gga_c_lyp"))
    values = module.unrestricted_energy_density_channels(open_shell_features)
    assert values.shape[-1] == 2
    assert calls == ["gga_x_b88", "gga_c_lyp"]
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
pytest -q tests/test_neural_xc_runtime.py -k polarized_channels
```

Expected: `unrestricted_energy_density_channels` is missing.

- [ ] **Step 3: Add the optional polarized callback**

Extend `SemilocalEnergyDensityModule`:

```python
unrestricted_energy_density_channels_fn: SemilocalEnergyDensityFn | None = None

def unrestricted_energy_density_channels(self, features):
    callback = self.unrestricted_energy_density_channels_fn
    if callback is None:
        callback = self.energy_density_channels_fn
    return self._normalize_channels(callback(features), features)
```

Extract the existing shape/finite normalization into `_normalize_channels` so
restricted and unrestricted methods do not duplicate validation.

In `make_libxc_semilocal_module`, supply:

```python
def unrestricted_energy_density_channels_fn(features):
    return jnp.stack(
        [
            eval_xc_energy_density_unrestricted(
                spec,
                features,
                allow_experimental_jax_xc=allow_experimental_jax_xc,
            )
            for spec in specs
        ],
        axis=-1,
    )
```

In `NeuralXCModel`, add
`unrestricted_semilocal_energy_density_channels` and use it only in
`_total_point_local_energy_from_unrestricted_variables` and unrestricted SCF
binding paths.

- [ ] **Step 4: Verify neural restricted behavior is unchanged**

Run:

```bash
pytest -q tests/test_neural_xc_runtime.py tests/test_neural_xc_presets.py
```

Expected: new polarized test and all existing restricted neural tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/td_graddft/neural_xc/components.py src/td_graddft/neural_xc/model.py tests/test_neural_xc_runtime.py
git commit -m "fix: evaluate neural UKS channels spin polarized"
```

### Task 6: Expose full neural unrestricted response HVP

**Files:**
- Modify: `src/td_graddft/neural_xc/model.py`
- Modify: `src/td_graddft/neural_xc/binding.py`
- Modify: `tests/test_neural_xc_runtime.py`

- [ ] **Step 1: Write failing HVP and parameter-gradient tests**

```python
def test_open_shell_neural_binding_exposes_full_spin_gga_hvp():
    molecule = _make_open_shell_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc=("gga_x_b88", "gga_c_lyp"),
        hidden_dims=(8, 8),
        name="open_shell_full_hvp",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(79), molecule)
    bound = functional.bind_to_molecule_for_response(params, molecule)
    tangent_a = jnp.ones((1, 4, molecule.ao.shape[0]))
    tangent_b = jnp.zeros_like(tangent_a)
    response_a, response_b = bound.spin_grid_response_hvp(
        molecule, tangent_a, tangent_b
    )
    assert response_a.shape == tangent_a.shape
    assert response_b.shape == tangent_b.shape
    assert jnp.all(jnp.isfinite(response_a))


def test_unrestricted_tda_gradient_through_neural_hvp_is_finite():
    molecule = _make_open_shell_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc=("gga_x_b88", "gga_c_lyp"),
        hidden_dims=(8, 8),
        name="open_shell_tda_gradient",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(81), molecule)

    def s1(local_params):
        return UnrestrictedTDA(
            molecule,
            functional,
            xc_params=local_params,
        ).kernel(nstates=1).excitation_energies[0]
    gradient = jax.grad(s1)(params)
    leaves = jax.tree_util.tree_leaves(gradient)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    assert sum(float(jnp.vdot(leaf, leaf).real) for leaf in leaves) > 0.0
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
pytest -q tests/test_neural_xc_runtime.py -k "full_spin_gga_hvp or tda_gradient_through_neural_hvp"
```

Expected: bound functional lacks `spin_grid_response_hvp`.

- [ ] **Step 3: Implement matrix-free neural spin HVP**

Add `spin_grid_response_hvp_fn` to `BoundNeuralXCFunctional` and a forwarding
method matching the traditional contract.

In `NeuralXCModel`, implement `_unrestricted_total_response_hvp` by applying
`jax.jvp` to the gradient of
`_total_point_local_energy_from_unrestricted_variables`. Pack and unpack the
same `[rho_a, rho_b, grad_a_xyz, grad_b_xyz]` variable order used by the
traditional wrapper.

In unrestricted response binding, provide a closure:

```python
def spin_grid_response_hvp_fn(response_molecule, tangent_a, tangent_b):
    del response_molecule
    return self._unrestricted_total_response_hvp(
        params,
        features,
        grad_a,
        grad_b,
        hf_projected,
        tangent_a,
        tangent_b,
        pt2_projected=pt2_projected,
        hf_spin_energy_density=(hfx_feature_a, hfx_feature_b),
        response_pt2_mode=self.response_pt2_mode,
    )
```

Remove `_unrestricted_spin_local_kernel_components` and its neural-binding
closure. Retain the dataclass's existing `spin_local_kernel_fn` field and
forwarding method solely for LDA-compatible custom functionals; never populate
or consume it for GGA.

- [ ] **Step 4: Verify neural JIT and differentiation**

Run:

```bash
pytest -q tests/test_neural_xc_runtime.py tests/test_unrestricted_spin_kernel.py
```

Expected: HVP, JIT, empty-beta, and parameter-gradient tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/td_graddft/neural_xc/model.py src/td_graddft/neural_xc/binding.py tests/test_neural_xc_runtime.py
git commit -m "feat: differentiate neural unrestricted TD response"
```

### Task 7: Match PySCF excitation energies and oscillator strengths

**Files:**
- Create: `tests/test_unrestricted_tddft_pyscf_compare.py`

- [ ] **Step 1: Add H2+ B3LYP RED comparison using identical orbitals**

Build PySCF UKS references and convert them with
`unrestricted_reference_from_pyscf`. For TDA and full TDDFT compare the first
four roots and oscillator strengths:

```python
ENERGY_ATOL = 8e-6
ENERGY_RTOL = 2e-5
OSC_ATOL = 2e-5
OSC_RTOL = 2e-4
MATRIX_ATOL = 2e-5
MATRIX_RTOL = 2e-5

np.testing.assert_allclose(pred_tda.e, ref_tda.e, atol=ENERGY_ATOL, rtol=ENERGY_RTOL)
np.testing.assert_allclose(
    oscillator_strengths(reference, pred_tda),
    ref_tda.oscillator_strength(),
    atol=OSC_ATOL,
    rtol=OSC_RTOL,
)
```

- [ ] **Step 2: Run H2+ tests and verify RED against the old response**

Run remotely in `jax_scf`:

```bash
TD_GRADDFT_RUN_OPEN_SHELL_TESTS=1 JAX_ENABLE_X64=1 \
pytest -q tests/test_unrestricted_tddft_pyscf_compare.py -k h2plus
```

Expected before implementation: missing spin-kernel error or 1-3 eV energy mismatch.

- [ ] **Step 3: Compare unrestricted A/B operator matrices**

Build dense matrices only inside the test by applying the local matrix-free
operator to identity columns. Reshape PySCF `get_ab()` blocks into the same
alpha-then-beta ordering and assert:

```python
np.testing.assert_allclose(
    predicted_a,
    reference_a,
    atol=MATRIX_ATOL,
    rtol=MATRIX_RTOL,
)
np.testing.assert_allclose(
    predicted_b,
    reference_b,
    atol=MATRIX_ATOL,
    rtol=MATRIX_RTOL,
)
```

Expected: both matrices pass after Tasks 1-6. A failure is a root-cause signal
to return to the corresponding projection or point-HVP task; tolerances are not
changed.

- [ ] **Step 4: Add OH and O2 tests**

Use:

- OH: B3LYP/def2-SVP, doublet, grid level 2;
- O2: PBE/def2-SVP, triplet, grid level 2.

Compare isolated roots directly. Group roots whose PySCF energies differ by at
most `1e-5 Hartree` and compare the sum of oscillator strengths within each
group.

- [ ] **Step 5: Run all open-shell numerical tests**

Run:

```bash
TD_GRADDFT_RUN_OPEN_SHELL_TESTS=1 JAX_ENABLE_X64=1 \
pytest -q tests/test_unrestricted_tddft_pyscf_compare.py
```

Expected: H2+, OH, and O2 pass the specified energy and oscillator-strength tolerances.

- [ ] **Step 6: Commit**

```bash
git add tests/test_unrestricted_tddft_pyscf_compare.py \
  src/td_graddft/tddft/_unrestricted_semilocal_response.py \
  src/td_graddft/tddft/unrestricted.py
git commit -m "test: validate unrestricted TDDFT against PySCF"
```

### Task 8: Regression, source freeze, and final remote H2+ validation

**Files:**
- No new production files.

- [ ] **Step 1: Run unrestricted unit and integration tests**

```bash
pytest -q \
  tests/test_unrestricted_semilocal_response.py \
  tests/test_unrestricted_spin_kernel.py \
  tests/test_unrestricted_casida.py \
  tests/test_unrestricted_open_shell.py \
  tests/test_neural_xc_runtime.py \
  tests/test_pyscf_style_excited_state_api.py
```

Expected: all pass.

- [ ] **Step 2: Run frozen restricted regression tests**

```bash
pytest -q tests/test_tddft.py tests/test_tddft_pyscf_compare.py
```

Expected: all pass without editing either test file or restricted source.

- [ ] **Step 3: Enforce the restricted source freeze**

```bash
git diff c195992 --name-only | rg \
  'src/td_graddft/tddft/response.py|src/td_graddft/scf/rks.py'
```

Expected: no output.

- [ ] **Step 4: Repeat the H2+ B3LYP/def2-SVP diagnostic on GPU4**

Run the four-root TDA/full-TDDFT comparison and save JSON/CSV containing:

- PySCF and GradTDDFT energies;
- absolute root errors;
- oscillator strengths and absolute errors;
- ground-state energy and density-matrix differences;
- hardware, backend, grid level, and elapsed time.

Expected: all root and oscillator-strength errors satisfy the acceptance thresholds.

- [ ] **Step 5: Final commit**

```bash
git add src tests docs/superpowers
git commit -m "feat: complete unrestricted TDDFT response"
```

If no cleanup changes remain after prior commits, skip this commit and report
the existing commit sequence.
