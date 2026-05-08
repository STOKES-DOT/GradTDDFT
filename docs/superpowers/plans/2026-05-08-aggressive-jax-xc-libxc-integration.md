# Aggressive JAX-XC LibXC Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route selected `jax_xc` functionals through `jax_libxc` and Neural XC semilocal-channel construction while guarding experimental B97-family paths by default.

**Architecture:** Keep the existing local strict evaluator as the default for verified names. Add adapter metadata and a feature-bundle evaluator for active upstream/vendored `jax_xc`, then extend `jax_libxc` parsing/evaluation with an explicit `allow_experimental_jax_xc` opt-in. Propagate the same guard to Neural XC semilocal modules so training never silently accepts known mismatched functionals.

**Tech Stack:** Python, JAX, Flax, PySCF comparison utilities, pytest, TD-GradDFT `jax_xc_adapter`, `jax_libxc`, and Neural XC DM21 modules.

---

## File Map

- `src/td_graddft/jax_xc_adapter.py`: add metadata, status classification, listing, and feature-bundle evaluation against active `jax_xc`.
- `src/td_graddft/jax_libxc.py`: extend parsing, semilocal resolution, and energy-density evaluation with `allow_experimental_jax_xc`.
- `src/td_graddft/neural_xc/dm21/functional.py`: propagate the experimental opt-in to semilocal module construction and functional factories.
- `tests/test_jax_xc_adapter.py`: adapter metadata and safe/evaluation unit tests.
- `tests/test_vendored_jax_xc_backend.py`: `jax_libxc` opt-in and guard behavior tests.
- `tests/test_dm21_like.py`: Neural XC default guard and explicit opt-in tests.
- `README.md`: document strict, wrapped, and experimental `jax_xc` routes.
- `src/td_graddft.egg-info/PKG-INFO`: keep packaging metadata text aligned with README if this repository continues tracking generated egg-info.

No commit is possible in this workspace because `git status` reports `fatal: not a git repository`.

---

### Task 1: Add Adapter Metadata And Experimental Classification Tests

**Files:**
- Modify: `tests/test_jax_xc_adapter.py`
- Modify: `src/td_graddft/jax_xc_adapter.py`

- [ ] **Step 1: Write failing metadata tests**

Add these tests to `tests/test_jax_xc_adapter.py`:

```python
def test_jax_xc_functional_info_classifies_strict_wrapped_and_experimental(monkeypatch):
    from td_graddft import jax_xc_adapter

    class FakeModule:
        __version__ = "fake"

        @staticmethod
        def gga_x_rpbe(*, polarized=False):
            del polarized
            return lambda rho_fn, r, mo_fn=None: rho_fn(r) * 0.0

        @staticmethod
        def hyb_gga_xc_b97(*, polarized=False):
            del polarized
            return lambda rho_fn, r, mo_fn=None: rho_fn(r) * 0.0

    monkeypatch.setattr(
        jax_xc_adapter,
        "load_jax_xc",
        lambda: (jax_xc_adapter._SafeJAXXCModule(FakeModule()), "upstream"),
    )

    strict = jax_xc_adapter.jax_xc_functional_info("gga_x_pbe")
    wrapped = jax_xc_adapter.jax_xc_functional_info("hyb_gga_xc_pbeh")
    rpbe = jax_xc_adapter.jax_xc_functional_info("gga_x_rpbe")
    b97 = jax_xc_adapter.jax_xc_functional_info("hyb_gga_xc_b97")
    missing = jax_xc_adapter.jax_xc_functional_info("gga_x_not_real")

    assert strict.status == "strict"
    assert wrapped.status == "wrapped"
    assert wrapped.children
    assert rpbe.status == "experimental"
    assert b97.status == "experimental"
    assert "B97" in b97.reason
    assert missing.status == "unavailable"
```

Add a listing test:

```python
def test_list_jax_xc_functionals_can_filter_by_status(monkeypatch):
    from td_graddft import jax_xc_adapter

    class FakeModule:
        __version__ = "fake"

        @staticmethod
        def gga_x_rpbe(*, polarized=False):
            del polarized
            return lambda rho_fn, r, mo_fn=None: rho_fn(r) * 0.0

    monkeypatch.setattr(
        jax_xc_adapter,
        "load_jax_xc",
        lambda: (jax_xc_adapter._SafeJAXXCModule(FakeModule()), "upstream"),
    )

    experimental = jax_xc_adapter.list_jax_xc_functionals(status="experimental")

    assert "gga_x_rpbe" in {info.name for info in experimental}
    assert all(info.status == "experimental" for info in experimental)
```

- [ ] **Step 2: Run RED**

Run:

```bash
pytest tests/test_jax_xc_adapter.py::test_jax_xc_functional_info_classifies_strict_wrapped_and_experimental tests/test_jax_xc_adapter.py::test_list_jax_xc_functionals_can_filter_by_status -q
```

Expected: FAIL because `jax_xc_functional_info` and `list_jax_xc_functionals` do not exist yet.

- [ ] **Step 3: Implement metadata API**

In `src/td_graddft/jax_xc_adapter.py`, add:

```python
from dataclasses import dataclass
from typing import Literal

JAXXCStatus = Literal["strict", "wrapped", "experimental", "unavailable"]

@dataclass(frozen=True)
class JAXXCFunctionalInfo:
    name: str
    status: JAXXCStatus
    family: str
    reason: str
    children: tuple[str, ...] = ()
```

Add constants:

```python
_STRICT_FUNCTIONALS = {
    "lda_x",
    "lda_c_pw",
    "lda_c_vwn",
    "lda_c_vwn_rpa",
    "gga_x_b88",
    "gga_x_pbe",
    "gga_x_wpbeh",
    "gga_c_lyp",
    "gga_c_pbe",
}

_EXPERIMENTAL_FUNCTIONALS = {
    "gga_x_rpbe",
    "gga_x_wc",
    "gga_x_pw91",
    "hyb_gga_xc_b97",
    "hyb_gga_xc_b97_1",
    "hyb_gga_xc_wb97x",
}

_KNOWN_MISMATCH_FUNCTIONALS = {
    "hyb_gga_xc_b97",
    "hyb_gga_xc_b97_1",
    "hyb_gga_xc_wb97x",
}
```

Add helper functions:

```python
def _family_from_name(name: str) -> str:
    if name.startswith("lda_"):
        return "LDA"
    if name.startswith("gga_"):
        return "GGA"
    if name.startswith("hyb_gga_"):
        return "HYB_GGA"
    return "unknown"


def _active_jax_xc_has(name: str) -> bool:
    module, _ = load_jax_xc()
    return hasattr(module, name)


def jax_xc_functional_info(name: str) -> JAXXCFunctionalInfo:
    canonical = str(name).strip().lower()
    if canonical in _STRICT_FUNCTIONALS:
        return JAXXCFunctionalInfo(
            name=canonical,
            status="strict",
            family=_family_from_name(canonical),
            reason="Implemented by TD-GradDFT's strict local JAX evaluator.",
        )
    if canonical in _SAFE_HYBRID_COMPOSITES:
        children = tuple(child for _, child, _ in _SAFE_HYBRID_COMPOSITES[canonical])
        return JAXXCFunctionalInfo(
            name=canonical,
            status="wrapped",
            family=_family_from_name(canonical),
            reason="Reconstructed by TD-GradDFT from safe semilocal child components.",
            children=children,
        )
    if canonical in _EXPERIMENTAL_FUNCTIONALS and _active_jax_xc_has(canonical):
        reason = "Available from jax_xc but not validated for Neural XC training."
        if canonical in _KNOWN_MISMATCH_FUNCTIONALS:
            reason = (
                "B97-family jax_xc output has benchmark mismatches against PySCF/libxc; "
                "explicit experimental opt-in is required."
            )
        return JAXXCFunctionalInfo(
            name=canonical,
            status="experimental",
            family=_family_from_name(canonical),
            reason=reason,
        )
    return JAXXCFunctionalInfo(
        name=canonical,
        status="unavailable",
        family=_family_from_name(canonical),
        reason="No strict, wrapped, or active jax_xc implementation is available.",
    )


def list_jax_xc_functionals(status: JAXXCStatus | None = None) -> tuple[JAXXCFunctionalInfo, ...]:
    names = sorted(_STRICT_FUNCTIONALS | set(_SAFE_HYBRID_COMPOSITES) | _EXPERIMENTAL_FUNCTIONALS)
    infos = tuple(jax_xc_functional_info(name) for name in names)
    if status is None:
        return infos
    return tuple(info for info in infos if info.status == status)
```

- [ ] **Step 4: Run GREEN**

Run:

```bash
pytest tests/test_jax_xc_adapter.py::test_jax_xc_functional_info_classifies_strict_wrapped_and_experimental tests/test_jax_xc_adapter.py::test_list_jax_xc_functionals_can_filter_by_status -q
```

Expected: PASS.

---

### Task 2: Add Adapter Feature-Bundle Evaluation

**Files:**
- Modify: `tests/test_jax_xc_adapter.py`
- Modify: `src/td_graddft/jax_xc_adapter.py`

- [ ] **Step 1: Write failing feature-bundle evaluation tests**

Add:

```python
def test_eval_jax_xc_from_restricted_features_requires_experimental_opt_in(monkeypatch):
    from td_graddft import jax_xc_adapter
    from td_graddft.jax_libxc import RestrictedFeatureBundle

    class FakeModule:
        __version__ = "fake"

        @staticmethod
        def gga_x_rpbe(*, polarized=False):
            del polarized
            return lambda rho_fn, r, mo_fn=None: 2.0 * rho_fn(r)

    monkeypatch.setattr(
        jax_xc_adapter,
        "load_jax_xc",
        lambda: (jax_xc_adapter._SafeJAXXCModule(FakeModule()), "upstream"),
    )
    features = RestrictedFeatureBundle(
        rho_a=jnp.asarray([0.1, 0.2]),
        rho_b=jnp.asarray([0.1, 0.2]),
        sigma_aa=jnp.asarray([0.01, 0.02]),
        sigma_ab=jnp.asarray([0.01, 0.02]),
        sigma_bb=jnp.asarray([0.01, 0.02]),
        tau_a=jnp.zeros((2,)),
        tau_b=jnp.zeros((2,)),
    )

    with pytest.raises(ValueError, match="allow_experimental_jax_xc=True"):
        jax_xc_adapter.eval_jax_xc_from_restricted_features("gga_x_rpbe", features)

    eps = jax_xc_adapter.eval_jax_xc_from_restricted_features(
        "gga_x_rpbe",
        features,
        allow_experimental_jax_xc=True,
    )

    assert eps.shape == features.rho.shape
    assert jnp.allclose(eps, 2.0 * features.rho)
```

Add `import pytest` and `import jax.numpy as jnp` at the top if missing.

- [ ] **Step 2: Run RED**

Run:

```bash
pytest tests/test_jax_xc_adapter.py::test_eval_jax_xc_from_restricted_features_requires_experimental_opt_in -q
```

Expected: FAIL because `eval_jax_xc_from_restricted_features` does not exist.

- [ ] **Step 3: Implement feature-bundle evaluator**

Add to `src/td_graddft/jax_xc_adapter.py`:

```python
def _raise_if_not_allowed(info: JAXXCFunctionalInfo, *, allow_experimental_jax_xc: bool) -> None:
    if info.status == "unavailable":
        raise KeyError(f"jax_xc functional {info.name!r} is unavailable: {info.reason}")
    if info.status == "experimental" and not bool(allow_experimental_jax_xc):
        raise ValueError(
            f"jax_xc functional {info.name!r} is experimental: {info.reason} "
            "Pass allow_experimental_jax_xc=True to evaluate it."
        )


def eval_jax_xc_from_restricted_features(
    name: str,
    features: RestrictedFeatureBundle,
    *,
    allow_experimental_jax_xc: bool = False,
) -> jnp.ndarray:
    info = jax_xc_functional_info(name)
    _raise_if_not_allowed(info, allow_experimental_jax_xc=allow_experimental_jax_xc)
    if info.status == "strict":
        return _eval_xc_per_particle(info.name, features)

    module, _ = load_jax_xc()
    factory = getattr(module, info.name)
    functional = factory(polarized=False)
    rho = jnp.maximum(jnp.asarray(features.rho), 1e-12)
    sigma = jnp.maximum(jnp.asarray(features.sigma), 0.0)
    grad_mag = jnp.sqrt(sigma)
    origin = jnp.zeros((3,), dtype=rho.dtype)

    def point_eval(rho_value, grad_value):
        def rho_fn(r):
            return rho_value + grad_value * r[0]

        value = functional(rho_fn, origin)
        if isinstance(value, tuple):
            value = value[0]
        return jnp.asarray(value, dtype=rho.dtype)

    return jnp.nan_to_num(jax.vmap(point_eval)(rho, grad_mag), nan=0.0, posinf=0.0, neginf=0.0)
```

Add `import jax` if missing.

- [ ] **Step 4: Run GREEN**

Run:

```bash
pytest tests/test_jax_xc_adapter.py::test_eval_jax_xc_from_restricted_features_requires_experimental_opt_in tests/test_jax_xc_adapter.py -q
```

Expected: PASS.

---

### Task 3: Integrate Experimental Opt-In Into `jax_libxc`

**Files:**
- Modify: `tests/test_vendored_jax_xc_backend.py`
- Modify: `src/td_graddft/jax_libxc.py`

- [ ] **Step 1: Write failing `jax_libxc` opt-in tests**

Add:

```python
def test_eval_xc_energy_density_routes_experimental_jax_xc_with_opt_in(monkeypatch):
    from td_graddft import jax_xc_adapter

    class FakeModule:
        __version__ = "fake"

        @staticmethod
        def gga_x_rpbe(*, polarized=False):
            del polarized
            return lambda rho_fn, r, mo_fn=None: 3.0 * rho_fn(r)

    monkeypatch.setattr(
        jax_xc_adapter,
        "load_jax_xc",
        lambda: (jax_xc_adapter._SafeJAXXCModule(FakeModule()), "upstream"),
    )
    features = _features()

    with pytest.raises(ValueError, match="allow_experimental_jax_xc=True"):
        eval_xc_energy_density("gga_x_rpbe", features)

    got = eval_xc_energy_density(
        "gga_x_rpbe",
        features,
        allow_experimental_jax_xc=True,
    )

    assert got.shape == features.rho.shape
    assert jnp.allclose(got, 3.0 * features.rho * features.rho)
```

Add `import pytest` at the top if missing.

Add B97 guard:

```python
def test_b97_family_is_experimental_by_default(monkeypatch):
    from td_graddft import jax_xc_adapter

    class FakeModule:
        __version__ = "fake"

        @staticmethod
        def hyb_gga_xc_b97(*, polarized=False):
            del polarized
            return lambda rho_fn, r, mo_fn=None: rho_fn(r)

    monkeypatch.setattr(
        jax_xc_adapter,
        "load_jax_xc",
        lambda: (jax_xc_adapter._SafeJAXXCModule(FakeModule()), "upstream"),
    )

    with pytest.raises(ValueError, match="B97-family"):
        eval_xc_energy_density("hyb_gga_xc_b97", _features())
```

- [ ] **Step 2: Run RED**

Run:

```bash
pytest tests/test_vendored_jax_xc_backend.py::test_eval_xc_energy_density_routes_experimental_jax_xc_with_opt_in tests/test_vendored_jax_xc_backend.py::test_b97_family_is_experimental_by_default -q
```

Expected: FAIL because `eval_xc_energy_density` has no `allow_experimental_jax_xc` argument and `parse_xc` rejects unknown names before adapter routing.

- [ ] **Step 3: Extend parser helpers**

In `src/td_graddft/jax_libxc.py`, import adapter metadata inside helper functions to avoid import cycles. Change signatures:

```python
def parse_xc(spec: str, *, allow_experimental_jax_xc: bool = False) -> list[XCTerm]:
```

Inside the unknown-name branch, replace the unconditional `KeyError` with:

```python
        from .jax_xc_adapter import jax_xc_functional_info

        info = jax_xc_functional_info(name)
        if info.status == "experimental" and not allow_experimental_jax_xc:
            raise ValueError(
                f"jax_xc functional {name!r} is experimental: {info.reason} "
                "Pass allow_experimental_jax_xc=True to evaluate it."
            )
        if info.status not in {"wrapped", "experimental"}:
            raise KeyError(
                f"Unsupported JAX XC functional {raw_name!r}. "
                "Supported names include TD-GradDFT strict local names, safe wrapped "
                "jax_xc composites, and explicitly enabled experimental jax_xc names."
            )
        family = info.family
        kind = "semilocal"
        terms.append(XCTerm(name=name, coefficient=coefficient, family=family, kind=kind))
        continue
```

Update:

```python
def semilocal_terms(spec: str, *, allow_experimental_jax_xc: bool = False) -> list[XCTerm]:
    return [
        term
        for term in parse_xc(spec, allow_experimental_jax_xc=allow_experimental_jax_xc)
        if term.kind == "semilocal"
    ]
```

Update `resolve_semilocal_xc_specs` to accept and forward `allow_experimental_jax_xc`.

- [ ] **Step 4: Extend evaluator**

Change signature:

```python
def eval_xc_energy_density(
    spec: str,
    features: RestrictedFeatureBundle,
    *,
    omega: Array | float | None = None,
    allow_experimental_jax_xc: bool = False,
) -> Array:
```

Inside the term loop:

```python
    from .jax_xc_adapter import eval_jax_xc_from_restricted_features

    for term in semilocal_terms(spec, allow_experimental_jax_xc=allow_experimental_jax_xc):
        if term.name in registry:
            _, evaluator = registry[term.name]
            if term.name == "gga_x_wpbeh":
                values.append(term.coefficient * evaluator(features, omega=omega_value))
            else:
                values.append(term.coefficient * evaluator(features))
        else:
            eps = eval_jax_xc_from_restricted_features(
                term.name,
                features,
                allow_experimental_jax_xc=allow_experimental_jax_xc,
            )
            values.append(term.coefficient * features.rho * eps)
```

Leave `_eval_xc_per_particle` strict-only for fallback adapter use.

- [ ] **Step 5: Run GREEN**

Run:

```bash
pytest tests/test_vendored_jax_xc_backend.py::test_eval_xc_energy_density_routes_experimental_jax_xc_with_opt_in tests/test_vendored_jax_xc_backend.py::test_b97_family_is_experimental_by_default tests/test_vendored_jax_xc_backend.py tests/test_jax_libxc.py -q
```

Expected: PASS.

---

### Task 4: Propagate Experimental Guard Into Neural XC

**Files:**
- Modify: `tests/test_dm21_like.py`
- Modify: `src/td_graddft/neural_xc/dm21/functional.py`

- [ ] **Step 1: Write failing Neural XC guard tests**

Add to `tests/test_dm21_like.py`:

```python
def test_neural_xc_rejects_experimental_jax_xc_semilocal_by_default(monkeypatch):
    from td_graddft import jax_xc_adapter

    class FakeModule:
        __version__ = "fake"

        @staticmethod
        def hyb_gga_xc_b97(*, polarized=False):
            del polarized
            return lambda rho_fn, r, mo_fn=None: rho_fn(r)

    monkeypatch.setattr(
        jax_xc_adapter,
        "load_jax_xc",
        lambda: (jax_xc_adapter._SafeJAXXCModule(FakeModule()), "upstream"),
    )

    with pytest.raises(ValueError, match="allow_experimental_jax_xc=True"):
        make_neural_xc_functional(
            semilocal_xc="hyb_gga_xc_b97",
            hidden_dims=(8,),
        )
```

Add explicit opt-in test:

```python
def test_neural_xc_accepts_experimental_jax_xc_with_explicit_opt_in(monkeypatch):
    from td_graddft import jax_xc_adapter

    class FakeModule:
        __version__ = "fake"

        @staticmethod
        def gga_x_rpbe(*, polarized=False):
            del polarized
            return lambda rho_fn, r, mo_fn=None: rho_fn(r)

    monkeypatch.setattr(
        jax_xc_adapter,
        "load_jax_xc",
        lambda: (jax_xc_adapter._SafeJAXXCModule(FakeModule()), "upstream"),
    )

    functional = make_neural_xc_functional(
        semilocal_xc="gga_x_rpbe",
        hidden_dims=(8,),
        allow_experimental_jax_xc=True,
    )

    assert functional.allow_experimental_jax_xc is True
    assert functional.resolved_non_hf_module().channel_names == ("gga_x_rpbe",)
```

- [ ] **Step 2: Run RED**

Run:

```bash
pytest tests/test_dm21_like.py::test_neural_xc_rejects_experimental_jax_xc_semilocal_by_default tests/test_dm21_like.py::test_neural_xc_accepts_experimental_jax_xc_with_explicit_opt_in -q
```

Expected: FAIL because `make_neural_xc_functional` does not accept `allow_experimental_jax_xc`.

- [ ] **Step 3: Add guard field and propagation**

In `src/td_graddft/neural_xc/dm21/functional.py`:

Change `_normalize_semilocal_xc_names`:

```python
def _normalize_semilocal_xc_names(
    semilocal_xc: str | Sequence[str],
    *,
    allow_experimental_jax_xc: bool = False,
) -> tuple[str, ...]:
    return resolve_semilocal_xc_specs(
        semilocal_xc,
        allow_experimental_jax_xc=allow_experimental_jax_xc,
    )
```

Change `make_libxc_semilocal_module`:

```python
def make_libxc_semilocal_module(
    channel_specs: str | Sequence[str],
    *,
    channel_names: Sequence[str] | None = None,
    name: str = "libxc_semilocal_module",
    allow_experimental_jax_xc: bool = False,
) -> SemilocalEnergyDensityModule:
    specs = _normalize_semilocal_xc_names(
        channel_specs,
        allow_experimental_jax_xc=allow_experimental_jax_xc,
    )
```

Change its evaluator:

```python
        return jnp.stack(
            [
                eval_xc_energy_density(
                    spec,
                    features,
                    allow_experimental_jax_xc=allow_experimental_jax_xc,
                )
                for spec in resolved_specs
            ],
            axis=-1,
        )
```

Change `_legacy_semilocal_module` to accept and pass `allow_experimental_jax_xc`.

Add field to `DM21LikeFunctional`:

```python
    allow_experimental_jax_xc: bool = False
```

Use it in `resolved_non_hf_module()`.

Add keyword to `_make_neural_xc_hybrid_functional(...)` and `make_neural_xc_functional(...)`:

```python
    allow_experimental_jax_xc: bool = False,
```

Use it for `n_semilocal` resolution and store it on `DM21LikeFunctional`.

- [ ] **Step 4: Run GREEN**

Run:

```bash
pytest tests/test_dm21_like.py::test_neural_xc_rejects_experimental_jax_xc_semilocal_by_default tests/test_dm21_like.py::test_neural_xc_accepts_experimental_jax_xc_with_explicit_opt_in tests/test_dm21_like.py::test_semilocal_xc_alias_expands_to_component_channels_for_neural_basis -q
```

Expected: PASS.

---

### Task 5: Documentation And Packaging Metadata

**Files:**
- Modify: `README.md`
- Modify: `src/td_graddft.egg-info/PKG-INFO`

- [ ] **Step 1: Update README `jax_xc` support section**

In `README.md`, replace the current `jax_xc` backend bullets with:

```markdown
`jax_xc` backend support:

- `td_graddft.jax_xc_adapter.load_jax_xc()` resolves backends in this order:
  external `jax_xc`, vendored generated `third_party/jax_xc/generated`, then the
  TD-GradDFT fallback subset.
- `jax_libxc.eval_xc_energy_density(...)` uses the local strict JAX
  implementations first, then safe wrapped `jax_xc` composites, then explicitly
  enabled experimental `jax_xc` functionals.
- Safe wrapped composites include PBE0/PBEH, B3LYP, B3PW91, BHandHLYP, HSE03,
  HSE06, and CAM-B3LYP.
- B97-family and other unvalidated generated functionals require
  `allow_experimental_jax_xc=True`; Neural XC training rejects them by default.
```

- [ ] **Step 2: Mirror the same text in `src/td_graddft.egg-info/PKG-INFO`**

Apply the same text block to the generated package metadata if it contains the README content.

- [ ] **Step 3: Verify docs grep**

Run:

```bash
rg -n "allow_experimental_jax_xc|B97-family|safe wrapped" README.md src/td_graddft.egg-info/PKG-INFO
```

Expected: finds the new documentation lines in both files.

---

### Task 6: Local And Remote Verification

**Files:**
- No edits unless verification exposes a regression.

- [ ] **Step 1: Run local focused tests**

Run:

```bash
pytest tests/test_jax_xc_adapter.py tests/test_vendored_jax_xc_backend.py tests/test_jax_libxc.py tests/test_dm21_like.py::test_neural_xc_rejects_experimental_jax_xc_semilocal_by_default tests/test_dm21_like.py::test_neural_xc_accepts_experimental_jax_xc_with_explicit_opt_in -q
```

Expected: PASS.

- [ ] **Step 2: Run local training smoke**

Run:

```bash
pytest tests/test_water_smoke.py tests/test_neural_xc_public_api.py tests/test_training.py -q
```

Expected: PASS.

- [ ] **Step 3: Run final grep**

Run:

```bash
rg -n "allow_experimental_jax_xc" src/td_graddft tests README.md
```

Expected: shows adapter, `jax_libxc`, Neural XC propagation, tests, and docs.

- [ ] **Step 4: Sync changed files to remote**

Run:

```bash
rsync -av --relative \
  README.md \
  src/td_graddft/jax_xc_adapter.py \
  src/td_graddft/jax_libxc.py \
  src/td_graddft/neural_xc/dm21/functional.py \
  src/td_graddft.egg-info/PKG-INFO \
  tests/test_jax_xc_adapter.py \
  tests/test_vendored_jax_xc_backend.py \
  tests/test_dm21_like.py \
  yjiao@8.218.101.131:/home/yjiao/TD-GradDFT/ \
  -e 'ssh -p 60001'
```

Expected: only the listed files transfer.

- [ ] **Step 5: Run remote smoke in `jax_scf`**

Run:

```bash
ssh -p 60001 yjiao@8.218.101.131 "cd /home/yjiao/TD-GradDFT && conda run -n jax_scf pytest tests/test_vendored_jax_xc_backend.py::test_eval_xc_energy_density_routes_experimental_jax_xc_with_opt_in tests/test_vendored_jax_xc_backend.py::test_b97_family_is_experimental_by_default tests/test_dm21_like.py::test_neural_xc_accepts_experimental_jax_xc_with_explicit_opt_in -q"
```

Expected: PASS.

- [ ] **Step 6: Run remote benchmark smoke for RPBE/WC/PW91**

Run:

```bash
ssh -p 60001 yjiao@8.218.101.131 "cd /home/yjiao/TD-GradDFT && conda run -n jax_scf python scripts/benchmark_jax_xc_pyscf_water.py --functionals gga_x_rpbe,gga_x_wc,gga_x_pw91 --grid-level 0 --max-points 128 --point-selection even --chunk-size 128 --output artifacts/jax_xc_water_aggressive_smoke_20260508.json"
```

Expected: command exits 0 and writes the JSON artifact. If one functional is unavailable from upstream `jax_xc`, report that exact status instead of treating it as a training regression.

---

## Self-Review

- Spec coverage: Tasks 1-2 add adapter metadata and evaluation; Task 3 integrates `jax_libxc`; Task 4 adds Neural XC training guard; Task 5 updates docs; Task 6 covers local and remote verification.
- Placeholder scan: no placeholder steps are intentionally left for future work.
- Type consistency: `allow_experimental_jax_xc` is used consistently in adapter evaluation, `jax_libxc`, semilocal module construction, and Neural XC factories.
- Git status: this workspace is not a Git checkout, so the plan omits commit steps and uses verification checkpoints instead.
