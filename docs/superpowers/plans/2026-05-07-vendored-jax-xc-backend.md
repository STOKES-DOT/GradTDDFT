# Vendored jax_xc Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Vendor the complete `sail-sg/jax_xc` repository and route TD-GradDFT's neural XC and RSH training paths through a stable backend adapter.

**Architecture:** Keep `td_graddft.jax_libxc` as the compatibility facade. Add vendored-source discovery to `td_graddft.jax_xc_adapter`, backend metadata helpers in `td_graddft.xc_backend`, and semilocal channel resolution that can use existing local guaranteed functionals now while exposing vendored availability for broader functional experiments.

**Tech Stack:** Python 3.10/3.11, JAX, Flax, PySCF/libxc for comparisons, vendored `sail-sg/jax_xc`, pytest.

---

## File Structure

- Create `src/td_graddft/xc_backend/__init__.py`: internal backend namespace.
- Create `src/td_graddft/xc_backend/vendor.py`: detect complete vendored `third_party/jax_xc`, read commit/source metadata, and report availability.
- Modify `src/td_graddft/jax_xc_adapter.py`: search external package first, vendored generated package second, fallback last; expose backend label and diagnostic metadata.
- Modify `src/td_graddft/jax_libxc.py`: add public helper functions for backend metadata and semilocal channel resolution without breaking existing imports.
- Modify `src/td_graddft/neural_xc/dm21/functional.py`: route `semilocal_xc` names through the backend resolver so single aliases, tuples, and vendored names share one validation path.
- Add `tests/test_vendored_jax_xc_backend.py`: verify vendored discovery, fallback behavior, and semilocal channel resolution.
- Update `tests/test_dm21_like.py`: add a focused neural XC test for non-default vendored-capable channel names.
- Update remote `/home/yjiao/TD-GradDFT/third_party/jax_xc`: replace incomplete directory with a complete clone and write TD-GradDFT metadata.

## Task 1: Write Vendored Backend Discovery Tests

**Files:**
- Create: `tests/test_vendored_jax_xc_backend.py`
- Test: `tests/test_vendored_jax_xc_backend.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path

import jax.numpy as jnp

from td_graddft.jax_libxc import (
    RestrictedFeatureBundle,
    eval_xc_energy_density,
    jax_xc_backend_info,
    resolve_semilocal_xc_specs,
)
from td_graddft.jax_xc_adapter import load_jax_xc
from td_graddft.xc_backend.vendor import vendored_jax_xc_info


def _features():
    rho = jnp.asarray([0.2, 0.4])
    sigma = jnp.asarray([0.01, 0.02])
    tau = jnp.asarray([0.05, 0.07])
    return RestrictedFeatureBundle(
        rho_a=0.5 * rho,
        rho_b=0.5 * rho,
        sigma_aa=0.25 * sigma,
        sigma_ab=0.25 * sigma,
        sigma_bb=0.25 * sigma,
        tau_a=0.5 * tau,
        tau_b=0.5 * tau,
    )


def test_load_jax_xc_reports_external_vendored_or_fallback_backend():
    module, backend = load_jax_xc()

    assert module is not None
    assert backend in {"upstream", "vendored", "fallback"}


def test_vendored_jax_xc_info_has_stable_shape_even_when_missing():
    info = vendored_jax_xc_info()

    assert isinstance(info.root, Path)
    assert isinstance(info.complete, bool)
    assert info.backend_label in {"vendored", "missing"}


def test_public_backend_info_reports_active_backend():
    info = jax_xc_backend_info()

    assert info["backend"] in {"upstream", "vendored", "fallback"}
    assert "module_version" in info
    assert "vendored_complete" in info


def test_resolve_semilocal_xc_specs_expands_alias_and_preserves_tuple_channels():
    assert resolve_semilocal_xc_specs("pbe") == ("gga_x_pbe", "gga_c_pbe")
    assert resolve_semilocal_xc_specs(("lda_x", "gga_c_pbe")) == ("lda_x", "gga_c_pbe")


def test_resolved_semilocal_specs_are_energy_channels():
    features = _features()
    channels = [
        eval_xc_energy_density(spec, features)
        for spec in resolve_semilocal_xc_specs(("gga_x_pbe", "gga_c_pbe"))
    ]

    assert len(channels) == 2
    assert all(channel.shape == features.rho.shape for channel in channels)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vendored_jax_xc_backend.py -q`

Expected: FAIL because `td_graddft.xc_backend.vendor`, `jax_xc_backend_info`, and `resolve_semilocal_xc_specs` do not exist yet.

## Task 2: Implement Vendored Backend Discovery

**Files:**
- Create: `src/td_graddft/xc_backend/__init__.py`
- Create: `src/td_graddft/xc_backend/vendor.py`
- Modify: `src/td_graddft/jax_xc_adapter.py`
- Test: `tests/test_vendored_jax_xc_backend.py`

- [ ] **Step 1: Add backend namespace**

```python
"""Internal exchange-correlation backend helpers."""

from .vendor import VendoredJAXXCInfo, vendored_jax_xc_info

__all__ = ["VendoredJAXXCInfo", "vendored_jax_xc_info"]
```

- [ ] **Step 2: Add vendored source metadata helper**

Implement `VendoredJAXXCInfo` with fields `root`, `complete`, `backend_label`, `commit`, `version`, and `reason`. `vendored_jax_xc_info()` should consider a vendored tree complete when these files exist:

```text
third_party/jax_xc/LICENSE
third_party/jax_xc/README.rst
third_party/jax_xc/gen_repo/__init__.py
third_party/jax_xc/gen_repo/wheel.BUILD
```

It should read optional metadata from `third_party/jax_xc/TD_GRADDFT_VENDOR.json`.

- [ ] **Step 3: Update adapter search order**

In `jax_xc_adapter.py`, keep external `import jax_xc` first. If that fails, add a vendored generated-package candidate path only when it exists. The initial candidate is `third_party/jax_xc/generated`; this keeps clone source and generated importable package separate. Fall back to `_FallbackJAXXC` when neither import path works.

- [ ] **Step 4: Run discovery test**

Run: `pytest tests/test_vendored_jax_xc_backend.py::test_load_jax_xc_reports_external_vendored_or_fallback_backend tests/test_vendored_jax_xc_backend.py::test_vendored_jax_xc_info_has_stable_shape_even_when_missing -q`

Expected: PASS locally with fallback or vendored status.

## Task 3: Add Public Backend Info and Semilocal Spec Resolution

**Files:**
- Modify: `src/td_graddft/jax_libxc.py`
- Test: `tests/test_vendored_jax_xc_backend.py`

- [ ] **Step 1: Write public helpers**

Add:

```python
def resolve_semilocal_xc_specs(spec: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(spec, str):
        raw_specs = (spec,)
    else:
        raw_specs = tuple(str(value) for value in spec)
    if not raw_specs:
        raise ValueError("semilocal_xc must contain at least one functional specification.")
    resolved = []
    for raw_spec in raw_specs:
        for term in semilocal_terms(raw_spec):
            resolved.append(term.name)
    if not resolved:
        raise ValueError(f"XC specification {spec!r} contains no semilocal terms.")
    return tuple(resolved)


def jax_xc_backend_info() -> dict[str, Any]:
    from .jax_xc_adapter import load_jax_xc
    from .xc_backend.vendor import vendored_jax_xc_info

    module, backend = load_jax_xc()
    vendored = vendored_jax_xc_info()
    return {
        "backend": backend,
        "module_version": getattr(module, "__version__", None),
        "vendored_complete": vendored.complete,
        "vendored_root": str(vendored.root),
        "vendored_commit": vendored.commit,
        "vendored_reason": vendored.reason,
    }
```

`resolve_semilocal_xc_specs("pbe")` returns `("gga_x_pbe", "gga_c_pbe")`. A single non-composite atomic term returns itself. A tuple/list resolves each element and flattens aliases.

- [ ] **Step 2: Preserve current parser errors**

`resolve_semilocal_xc_specs` should call `parse_xc` for validation, reject `hf`-only specs for semilocal channel construction, and keep exact unsupported-name messages from `parse_xc`.

- [ ] **Step 3: Run resolver tests**

Run: `pytest tests/test_vendored_jax_xc_backend.py::test_public_backend_info_reports_active_backend tests/test_vendored_jax_xc_backend.py::test_resolve_semilocal_xc_specs_expands_alias_and_preserves_tuple_channels tests/test_vendored_jax_xc_backend.py::test_resolved_semilocal_specs_are_energy_channels -q`

Expected: PASS.

## Task 4: Route Neural semilocal_xc Through Resolver

**Files:**
- Modify: `src/td_graddft/neural_xc/dm21/functional.py`
- Modify: `tests/test_dm21_like.py`
- Test: `tests/test_dm21_like.py`

- [ ] **Step 1: Write focused failing test**

Append:

```python
def test_semilocal_xc_alias_expands_to_component_channels_for_neural_basis():
    molecule = _make_toy_molecule()
    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        energy_mode="graddft_coeff_basis",
        hidden_dims=(8, 8),
        name="toy_pbe_alias_channel_resolution",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(7), molecule)
    features = restricted_grid_features(molecule)
    channels = functional.semilocal_energy_density_channels(features)
    coefficients = functional.channel_coefficients(
        params,
        features,
        molecule=molecule,
        semilocal_energy_density=jnp.sum(channels, axis=-1),
        hf_energy_density=jnp.zeros(features.rho.shape),
        hf_spin_energy_density=(jnp.zeros(features.rho.shape), jnp.zeros(features.rho.shape)),
    )

    assert channels.shape[-1] == 2
    assert coefficients.shape[-1] >= 2
    assert jnp.all(jnp.isfinite(channels))
```

- [ ] **Step 2: Run it to verify failure if current alias remains a single channel**

Run: `pytest tests/test_dm21_like.py::test_semilocal_xc_alias_expands_to_component_channels_for_neural_basis -q`

Expected before implementation: FAIL with `channels.shape[-1] == 1` if aliases are not expanded.

- [ ] **Step 3: Update `_normalize_semilocal_xc_names`**

Import `resolve_semilocal_xc_specs` from `td_graddft.jax_libxc` and make `_normalize_semilocal_xc_names` return the resolved flattened tuple.

- [ ] **Step 4: Run focused neural tests**

Run: `pytest tests/test_dm21_like.py::test_semilocal_xc_alias_expands_to_component_channels_for_neural_basis tests/test_dm21_like.py::test_bounded_sigmoid_coefficients_are_nonnegative_and_bounded -q`

Expected: PASS.

## Task 5: Vendor Complete jax_xc on Remote

**Files:**
- Remote create/update: `/home/yjiao/TD-GradDFT/third_party/jax_xc`
- Remote create: `/home/yjiao/TD-GradDFT/third_party/jax_xc/TD_GRADDFT_VENDOR.json`

- [ ] **Step 1: Back up incomplete directory**

Run on remote:

```bash
cd /home/yjiao/TD-GradDFT
if [ -d third_party/jax_xc ] && [ ! -f third_party/jax_xc/README.rst ]; then
  mv third_party/jax_xc third_party/jax_xc.incomplete.$(date +%Y%m%d%H%M%S)
fi
```

Expected: incomplete directory is preserved under a timestamped backup path.

- [ ] **Step 2: Clone upstream**

Run on remote:

```bash
cd /home/yjiao/TD-GradDFT
git clone https://github.com/sail-sg/jax_xc third_party/jax_xc
```

Expected: `third_party/jax_xc/README.rst`, `LICENSE`, and `gen_repo/wheel.BUILD` exist.

- [ ] **Step 3: Write metadata**

Run on remote:

```bash
cd /home/yjiao/TD-GradDFT/third_party/jax_xc
commit="$(git rev-parse HEAD)"
cat > TD_GRADDFT_VENDOR.json <<EOF
{
  "upstream": "https://github.com/sail-sg/jax_xc",
  "commit": "${commit}",
  "license": "MPL-2.0",
  "vendored_for": "TD-GradDFT XC backend"
}
EOF
```

Expected: metadata file records upstream commit.

## Task 6: Sync Code to Remote and Verify

**Files:**
- Sync modified local files from `/Users/jiaoyuan/Documents/GitHub/TD-GradDFT` to `/home/yjiao/TD-GradDFT`

- [ ] **Step 1: Sync source and tests**

Use `rsync -av` for:

```text
src/td_graddft/jax_xc_adapter.py
src/td_graddft/jax_libxc.py
src/td_graddft/xc_backend/
src/td_graddft/neural_xc/dm21/functional.py
tests/test_vendored_jax_xc_backend.py
tests/test_dm21_like.py
docs/superpowers/specs/2026-05-07-jax-libxc-jax-xc-refactor-design.md
docs/superpowers/plans/2026-05-07-vendored-jax-xc-backend.md
```

- [ ] **Step 2: Run focused remote tests**

Run:

```bash
cd /home/yjiao/TD-GradDFT
PYTHONPATH=/home/yjiao/TD-GradDFT/src /home/yjiao/opt/miniconda3/envs/grad/bin/python -m pytest \
  tests/test_vendored_jax_xc_backend.py \
  tests/test_jax_xc_adapter.py \
  tests/test_jax_libxc.py::test_lc_wpbe_local_exchange_is_omega_dependent_and_differentiable \
  tests/test_dm21_like.py::test_semilocal_xc_alias_expands_to_component_channels_for_neural_basis \
  tests/test_neural_xc_public_api.py::test_rsh_public_constructor_uses_strict_lc_wpbe_local_spec_by_default \
  -q
```

Expected: PASS.

- [ ] **Step 3: Run remote backend probe**

Run:

```bash
cd /home/yjiao/TD-GradDFT
PYTHONPATH=/home/yjiao/TD-GradDFT/src /home/yjiao/opt/miniconda3/envs/grad/bin/python - <<'PY'
from td_graddft.jax_libxc import jax_xc_backend_info, resolve_semilocal_xc_specs
print(jax_xc_backend_info())
print(resolve_semilocal_xc_specs("pbe"))
print(resolve_semilocal_xc_specs(("lda_x", "gga_c_pbe")))
PY
```

Expected: backend info reports `vendored_complete=True`; `pbe` resolves to `("gga_x_pbe", "gga_c_pbe")`.

## Task 7: Completion Review

**Files:**
- No new files unless verification reveals a defect.

- [ ] **Step 1: Check source for stale direct assumptions**

Run:

```bash
rg -n "semilocal_xc=\"pbe\"|_normalize_semilocal_xc_names|load_jax_xc|third_party/jax_xc" src tests docs
```

Expected: references are intentional and route through the new resolver/adapter.

- [ ] **Step 2: Check no broad generated code was copied into Apache modules**

Run:

```bash
find src/td_graddft -maxdepth 3 -type f | sort
```

Expected: no full generated `jax_xc` formula catalogue under `src/td_graddft`; complete upstream source lives under `third_party/jax_xc`.

- [ ] **Step 3: Record final status**

Summarize changed files, remote vendored commit, tests run, and any remaining limitation such as generated wheel/package import not yet built from vendored source.
