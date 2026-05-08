# Aggressive JAX-XC LibXC Integration Design

## Goal

Integrate `jax_xc` more directly into TD-GradDFT's `jax_libxc` parsing and evaluation layer so additional libxc-translated semilocal and hybrid functional names can be used through the normal `eval_xc_energy_density(...)` and Neural XC semilocal-channel paths.

The integration must preserve the current strict local subset, keep training from silently accepting known mismatched functionals, and provide diagnostics that distinguish verified, wrapped, and experimental `jax_xc` routes.

## Non-Goals

- Do not replace the existing local strict implementations for `lda_x`, `lda_c_pw`, `lda_c_vwn`, `lda_c_vwn_rpa`, `gga_x_b88`, `gga_x_pbe`, `gga_x_wpbeh`, `gga_c_lyp`, and `gga_c_pbe`.
- Do not make B97-family functionals training-safe by default. Prior benchmarks showed `hyb_gga_xc_b97` as a numerical outlier against PySCF/libxc.
- Do not implement full spin-polarized or MGGA training support in this pass.
- Do not change exact-exchange ownership: TD-GradDFT's SCF/RSH layer still handles HF exchange; `jax_xc` supplies semilocal energy-density terms.

## Current State

- `jax_xc_adapter.load_jax_xc()` already resolves external `jax_xc`, vendored generated source, then fallback.
- `_SafeJAXXCModule` already reconstructs selected hybrid composite nodes such as PBE0, B3LYP, B3PW91, BHandHLYP, HSE03, HSE06, and CAM-B3LYP.
- `jax_libxc.parse_xc(...)` currently only recognizes the local strict subset and a few aliases.
- `eval_xc_energy_density(...)` currently evaluates only local strict terms.
- Benchmark evidence on water density shows common non-RSH names mostly match PySCF/libxc, with B97-family names requiring caution.

## Public API Shape

Add adapter-level metadata:

- `JAXXCFunctionalInfo`
  - `name`
  - `status`: `strict`, `wrapped`, `experimental`, or `unavailable`
  - `family`: `LDA`, `GGA`, `HYB_GGA`, or `unknown`
  - `reason`
  - `children` for wrapped composites
- `jax_xc_functional_info(name) -> JAXXCFunctionalInfo`
- `list_jax_xc_functionals(status: str | None = None) -> tuple[JAXXCFunctionalInfo, ...]`

Extend evaluation:

- `eval_xc_energy_density(spec, features, *, omega=None, allow_experimental_jax_xc=False)`
- `resolve_semilocal_xc_specs(..., allow_experimental_jax_xc=False)`

Default behavior remains conservative:

- strict local names work unchanged.
- wrapped names such as `hyb_gga_xc_pbeh`, `hyb_gga_xc_b3lyp`, `hyb_gga_xc_b3pw91`, and `hyb_gga_xc_bhandhlyp` work through explicit safe decomposition.
- experimental names raise unless `allow_experimental_jax_xc=True`.

## Functional Classification

`strict`:

- Local implementations already in `jax_libxc`.

`wrapped`:

- Safe composite hybrids that TD-GradDFT reconstructs from child semilocal components and validated mixing coefficients.
- Initial set: PBE0/PBEH, B3LYP, B3PW91, BHandHLYP, HSE03, HSE06, CAM-B3LYP.

`experimental`:

- Functionals exposed by installed or vendored `jax_xc` but not yet validated for training.
- Initial examples: `gga_x_rpbe`, `gga_x_wc`, `gga_x_pw91`, `hyb_gga_xc_b97`, `hyb_gga_xc_b97_1`, `hyb_gga_xc_wb97x`.
- Experimental functionals can be evaluated for diagnostics and benchmark scripts when explicitly enabled.

`unavailable`:

- Names not present in local strict registry, safe wrapper registry, or active `jax_xc` backend.

## Evaluation Flow

1. Normalize aliases and parse weighted XC terms.
2. For each semilocal term:
   - If local strict implementation exists, use it.
   - Else ask `jax_xc_adapter` for metadata.
   - If metadata is `wrapped`, evaluate using safe wrapper.
   - If metadata is `experimental`, require `allow_experimental_jax_xc=True`.
   - If metadata is `unavailable`, raise a clear `KeyError`.
3. Convert `jax_xc` per-particle epsilon to local grid contribution by multiplying by total density.
4. Keep `hf` terms parsed for hybrid coefficient accounting but excluded from semilocal quadrature.

## Neural XC Training Guard

`neural_xc.Functional(... semilocal_xc=...)` should reject experimental `jax_xc` channels by default.

Add an explicit opt-in field only where needed:

- `allow_experimental_jax_xc: bool = False`

This flag should propagate to semilocal module construction and channel evaluation. It should not alter local-HF/PT2 guards.

## Error Handling

Errors should name the functional and the remediation:

- unavailable backend: explain that no external or vendored `jax_xc` exposes the name.
- experimental functional: explain that the user must pass `allow_experimental_jax_xc=True`.
- unsupported polarized route: keep the current `polarized=False` limitation explicit.
- known mismatch: B97-family messages should reference benchmark validation rather than pretending support is complete.

## Testing

TDD sequence:

1. RED: `eval_xc_energy_density("gga_x_rpbe", features)` currently raises.
2. GREEN: with active upstream/vendored `jax_xc`, `eval_xc_energy_density("gga_x_rpbe", features, allow_experimental_jax_xc=True)` returns finite grid contributions.
3. RED/GREEN: default evaluation of `hyb_gga_xc_b97` raises an experimental-functional error.
4. GREEN: explicit experimental opt-in evaluates B97-family names without marking them training-safe.
5. GREEN: safe wrapped hybrids keep existing results for PBE0/B3LYP/B3PW91/BHandHLYP/HSE/CAM.
6. GREEN: `neural_xc.Functional(semilocal_xc="hyb_gga_xc_b97")` rejects by default and accepts only with the explicit experimental flag.
7. Remote smoke on `jax_scf`: run a small PySCF comparison for RPBE/WC/PW91 and a short Neural XC construction test.

## Rollout

Phase 1: adapter metadata and evaluation opt-in.

Phase 2: parser/evaluator integration in `jax_libxc`.

Phase 3: Neural XC semilocal module propagation and training guard.

Phase 4: remote benchmark smoke and documentation update.

## Risks

- `jax_xc` function signatures may vary between upstream versions. The adapter must inspect through `load_jax_xc()` and fail clearly.
- Some generated functionals are slow or hard to JIT. Evaluation should allow chunked benchmark use without forcing training defaults to expand.
- Hybrid composites may contain exact-exchange pieces. TD-GradDFT must continue to drop semilocal-only `jax_xc` contributions into Neural XC channels and leave HF exchange to the SCF layer.
- B97-family mismatches can contaminate training if allowed silently. The default guard is mandatory.

## Success Criteria

- Existing strict local tests continue to pass.
- Current `jax_xc` fallback behavior remains available when upstream import fails.
- Additional `jax_xc` names can be evaluated through `eval_xc_energy_density(...)` with explicit experimental opt-in.
- Known safe wrapped hybrids remain default-usable.
- B97-family names are never accepted into training unless explicitly opted in as experimental.
- Remote `jax_scf` smoke passes after synchronization.
