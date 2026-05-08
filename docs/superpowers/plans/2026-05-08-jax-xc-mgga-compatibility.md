# JAX-XC MGGA Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route active `jax_xc` MGGA functionals through the experimental adapter path while preserving Neural XC and `f_xc` response behavior.

**Architecture:** Extend adapter metadata to classify active `mgga_*` and `hyb_mgga_*` names dynamically. Add an MGGA point evaluator that supplies `mo_fn` derived from `RestrictedFeatureBundle.tau`, then forward the explicit opt-in through `jax_libxc.eval_xc_response_tensor`. Keep all MGGA names experimental by default.

**Tech Stack:** Python, JAX, pytest, TD-GradDFT `jax_xc_adapter`, `jax_libxc`, Neural XC DM21.

---

## File Map

- `src/td_graddft/jax_xc_adapter.py`: dynamic MGGA classification and MGGA evaluator with `mo_fn`.
- `src/td_graddft/jax_libxc.py`: opt-in aware `xc_type` and `eval_xc_response_tensor`.
- `src/td_graddft/neural_xc/dm21/functional.py`: already forwards `allow_experimental_jax_xc`; no signature changes expected.
- `tests/test_jax_xc_adapter.py`: dynamic MGGA metadata and adapter evaluation tests.
- `tests/test_jax_libxc.py`: MGGA response tensor opt-in test.
- `tests/test_dm21_like.py`: Neural XC MGGA opt-in smoke.
- `README.md` and `src/td_graddft.egg-info/PKG-INFO`: document experimental MGGA support.

## Tasks

- [ ] Add failing adapter tests for dynamic MGGA classification, listing, and `mo_fn`/`tau` evaluation.
- [ ] Implement dynamic MGGA metadata and MGGA `mo_fn` construction in `jax_xc_adapter.py`.
- [ ] Add failing `jax_libxc` test for experimental MGGA energy and `(5, 5, ngrids)` response tensor.
- [ ] Forward `allow_experimental_jax_xc` through `xc_type`, `_point_xc_response_kernel`, and `eval_xc_response_tensor`.
- [ ] Add failing Neural XC smoke for dynamic MGGA opt-in and finite semilocal kernel.
- [ ] Update docs and run local focused tests.
- [ ] Sync changed files to remote and run `jax_scf` focused validation against active upstream `jax_xc`.

## Verification

Run:

```bash
pytest tests/test_jax_xc_adapter.py tests/test_jax_libxc.py tests/test_vendored_jax_xc_backend.py tests/test_dm21_like.py::test_neural_xc_accepts_dynamic_mgga_with_explicit_opt_in -q
```

Then sync and run remote focused tests in `jax_scf`.
