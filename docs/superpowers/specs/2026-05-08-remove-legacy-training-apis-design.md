# Remove Legacy Training APIs Design

## Goal

Remove public legacy APIs that can accidentally bypass the current Neural XC training path while keeping the supported `hfx_local` / `hfx_nu` / PT2 hybrid route working.

The supported training route is:

1. Build references with local HF auxiliary data when Neural XC training needs HF channels:
   - `compute_local_hfx_features=True`
   - `compute_local_hfx_aux=True`
   - `hfx_omega_values=(0.0, 0.4)` unless a narrower experiment requires otherwise.
2. Build references with local PT2 data when PT2 channels are enabled:
   - `compute_local_pt2_features=True` whenever `include_pt2_channel=True`.
3. Construct neural functionals through:
   - `td_graddft.neural_xc.Functional(...)`
   - `td_graddft.neural_xc.make_functional(...)`

## Non-Goals

- Do not remove the implemented `hfx_local`, `hfx_nu`, local-HF response, RSH interpolation, or PT2 channel machinery.
- Do not remove `td_graddft.reference_legacy` in this first cleanup. It remains the explicit PySCF-backed compatibility layer used by research scripts.
- Do not change the `td_graddft.jax_libxc` functional formulas or add new XC channels in this cleanup.
- Do not silently auto-generate expensive local-HF/PT2 fields inside trainers. Missing required fields should fail with clear errors.

## APIs To Remove From Public Surface

Remove these from `td_graddft.neural_xc` public exports and top-level lazy exports:

- `DensityNeuralXCFunctional`
- `NeuralXCFunctional`
- `PointwiseMLP`
- `make_neural_lda_functional`
- `make_dm21_like_functional`

Remove the deprecated compatibility module:

- `td_graddft.pyscf_bridge`

The internal modules under `td_graddft.neural_xc.base` may remain temporarily if current tests or experiments still need implementation pieces during migration, but they should no longer be public recommended APIs. If no internal code imports them after migration, they can be deleted in a later cleanup.

## Migration Rules

### Neural XC Construction

Old LDA toy training constructors should migrate to the supported facade:

```python
from td_graddft import neural_xc

functional = neural_xc.Functional(
    architecture="residual",
    semilocal_xc=("lda_x", "gga_x_b88", "lda_c_vwn_rpa", "gga_c_lyp"),
    include_pt2_channel=True,
    pt2_channel_mode="scaled_projected",
    energy_mode="graddft_coeff_basis_hf_pt2_heads",
)
```

Toy tests that intentionally do not need HF/PT2 channels can still construct a small private test double rather than using removed public training APIs.

### Reference Construction

Training examples and tools must import explicit builders from `td_graddft.reference_legacy` or the strict reference API, not from `td_graddft.pyscf_bridge`.

Neural XC training references that use the supported DM21-like path must request local HF features:

```python
reference = restricted_reference_from_pyscf(
    mf,
    compute_local_hfx_features=True,
    compute_local_hfx_aux=True,
    hfx_omega_values=(0.0, 0.4),
    compute_local_pt2_features=include_pt2_channel,
)
```

If `include_pt2_channel=True` and `compute_local_pt2_features=False`, training setup should fail before optimization starts.

## Phased Removal

### Phase 1: Public Export Cleanup

- Remove legacy neural constructors from `td_graddft.neural_xc.__all__`.
- Remove matching top-level entries from `td_graddft.__init__`.
- Delete `td_graddft.pyscf_bridge` or replace it with an import-time `ImportError` that points users to `td_graddft.reference_legacy`.
- Update public API tests so they assert the old names are gone.

Verification gate:

```bash
pytest tests/test_neural_xc_public_api.py tests/test_pyscf_style_namespace.py tests/test_tools_public_api_usage.py -q
```

### Phase 2: Training Test Migration

- Replace `tests/test_water_smoke.py` with a current Neural XC smoke path using `neural_xc.Functional(...)`.
- Ensure reference construction includes `hfx_local` and `hfx_nu`.
- Keep a small training assertion, but avoid relying on the removed LDA toy API.

Verification gate:

```bash
pytest tests/test_water_smoke.py tests/test_dm21_like.py::test_dm21_like_functional_trains_and_produces_excitation -q
```

### Phase 3: Reference/Trainer Guards

- Add explicit validation for Neural XC training data:
  - DM21-like HF channels require `reference.hfx_local` or `reference.hfx_nu`.
  - PT2-enabled functionals require local PT2 payloads.
- Make errors point to the required reference builder flags.

Verification gate:

```bash
pytest tests/test_neural_xc_public_api.py::test_ground_state_datum_from_reference_requires_hfx_fields tests/test_dm21_like.py::test_coefficient_prior_penalty_is_reported_and_nonnegative -q
```

### Phase 4: Examples And Docs

- Update README and examples to use `neural_xc.Functional(...)`.
- Replace `td_graddft.pyscf_bridge` imports with `td_graddft.reference_legacy` or strict reference builders.
- Keep research tools that intentionally use `reference_legacy`, but make their local-HF/PT2 flags explicit when they construct Neural XC training data.

Verification gate:

```bash
pytest tests/test_examples_public_api_usage.py tests/test_workflows_public_api_usage.py tests/test_tools_public_api_usage.py -q
```

## End-To-End Verification

Run these after each phase if the edited files affect training behavior, and run all of them before considering the cleanup complete:

```bash
pytest tests/test_jax_libxc.py tests/test_vendored_jax_xc_backend.py tests/test_neural_xc_presets.py -q
pytest tests/test_dm21_like.py::test_dm21_like_functional_trains_and_produces_excitation tests/test_dm21_like.py::test_libxc_semilocal_module_supports_common_exchange_and_correlation_components tests/test_dm21_like.py::test_coefficient_prior_penalty_is_reported_and_nonnegative -q
pytest tests/test_local_hf_test_module.py::test_local_hf_khh_response_wrapper_preserves_tda_parameter_gradients -q
pytest tests/test_differentiable_scf.py::test_self_consistent_training_mode_produces_finite_loss_and_energy -q
```

## Risks

- Removing `td_graddft.pyscf_bridge` will break old tests and external scripts that import it directly. This is intentional for the public API cleanup.
- Some research tools still use `reference_legacy`; this first cleanup should not force those tools onto strict builders unless they are already in the training path being edited.
- Full-suite failures unrelated to these APIs should be triaged separately, not hidden by compatibility shims.

## Success Criteria

- Old public Neural XC constructors and `td_graddft.pyscf_bridge` are no longer importable from the public API surface.
- Supported Neural XC training uses `neural_xc.Functional(...)`.
- Training references used by supported Neural XC paths include `hfx_local` / `hfx_nu`, and PT2 data when PT2 channels are enabled.
- The focused training, local-HF response, PT2, and `jax_libxc` tests pass after each deletion phase.
