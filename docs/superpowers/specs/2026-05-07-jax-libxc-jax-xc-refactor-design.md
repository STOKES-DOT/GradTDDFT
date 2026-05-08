# jax_libxc and Vendored jax_xc Refactor Design

## Goal

Refine TD-GradDFT's local XC backend so it is compatible with a complete vendored `sail-sg/jax_xc` source tree while preserving both active training paths:

- `training.NeuralXCTrainer`, where `semilocal_xc` defines the energy-density basis used by the neural coefficient model.
- `training.RSHOptimizer`, where `nn_rsh.RSH("lc-wpbe").trainable()` uses `lc_wpbe_local` with differentiable `omega`, `alpha`, and `beta`.

The design vendors the complete upstream `jax_xc` repository under `third_party/jax_xc` so TD-GradDFT can draw from the full functional catalogue. TD-GradDFT still owns the training-facing adapter layer, parser, feature bundle conversion, and compatibility facade.

## Non-Goals

- Do not manually copy every generated formula into `src/td_graddft`.
- Do not make the external pip package `jax-xc` a hard runtime dependency for training.
- Do not change the public training APIs.
- Do not change the public names exported from `td_graddft.jax_libxc`.
- Do not replace the existing SCF, TDDFT, or neural XC training loops.

## Current Constraints

The remote project at `/home/yjiao/TD-GradDFT` is not a Git checkout. The remote `third_party/jax_xc` directory appears incomplete and cannot be used as a normal Git repository. The remote `grad` conda environment does not currently provide an importable external `jax_xc`, so TD-GradDFT falls back to its local `td_graddft_fallback` adapter.

Because of this, implementation must replace the incomplete `third_party/jax_xc` with a complete upstream checkout or subtree. Training must be able to import the vendored package through TD-GradDFT's adapter without requiring `pip install jax-xc`.

`jax_xc` is MPL-2.0. The vendored tree must keep its original license and notices. TD-GradDFT's Apache-2.0 code should import or adapt the vendored package through a narrow boundary rather than mixing generated MPL files directly into unrelated modules.

## Architecture

Keep `td_graddft.jax_libxc` as the stable public facade. Add a vendored upstream source tree and split the TD-GradDFT adapter responsibilities that are currently concentrated in one file:

```text
third_party/jax_xc/
  -> complete upstream sail-sg/jax_xc checkout or subtree, including LICENSE

td_graddft/jax_libxc.py
  -> public compatibility facade

td_graddft/jax_xc_adapter.py
  -> resolves external jax_xc, vendored jax_xc, or local fallback

td_graddft/xc_backend/features.py
  -> RestrictedFeatureBundle and feature construction helpers

td_graddft/xc_backend/registry.py
  -> aliases, parser, vendored/upstream functional discovery, hybrid coefficients, xc_type

td_graddft/xc_backend/semilocal.py
  -> grid-feature-to-jax_xc evaluators plus selected local fallbacks

td_graddft/xc_backend/response.py
  -> value/gradient/Hessian kernels for SCF and TDDFT response

td_graddft/xc_backend/rsh_presets.py
  -> LC-wPBE and wB97X-D metadata
```

`jax_libxc.py` should re-export the existing names so current imports continue to work.

## Functional Registry

The registry remains PySCF-like at the TD-GradDFT boundary, but it can resolve many more functionals through vendored `jax_xc`.

The guaranteed compatibility set remains:

- Atomic semilocal terms: `lda_x`, `lda_c_pw`, `lda_c_vwn`, `lda_c_vwn_rpa`, `gga_x_b88`, `gga_x_pbe`, `gga_x_wpbeh`, `gga_c_lyp`, `gga_c_pbe`.
- Hybrid marker: `hf`.
- Aliases: `lda`, `svwn`, `svwn_rpa`, `pbe`, `pbe0`, `b3lyp`, `lc_wpbe_local`, plus the existing LC-wPBE local spelling variants.

Additional vendored `jax_xc` functionals should be accepted when their inputs can be represented from TD-GradDFT grid features:

- LDA: density only.
- GGA: density and `sigma`.
- mGGA: density, `sigma`, Laplacian when required, and `tau` when available.
- Hybrid metadata: exact-exchange and range-separation attributes are exposed when `jax_xc` provides them.

Each registered term exposes:

- name
- family: `LDA`, `GGA`, `MGGA`, or `HF`
- coefficient
- kind: `semilocal` or `hf`
- evaluator
- whether it accepts `omega`
- source: local fallback, external `jax_xc`, or vendored `jax_xc`

This avoids ad hoc branching in every caller.

## NeuralXCTrainer Compatibility

`training.NeuralXCTrainer` remains a neural coefficient-basis route. The important contract is:

```text
semilocal_xc -> energy-density channels -> neural coefficients -> local XC energy
```

`semilocal_xc` is the source of truth for the energy-density basis. It may be:

- a single alias, such as `pbe` or `b3lyp`
- a vendored `jax_xc` functional name, such as a supported LDA/GGA/mGGA name
- a comma/plus expression, such as `gga_x_pbe + gga_c_pbe`
- a tuple/list of channels, such as `("lda_x", "gga_x_b88", "lda_c_vwn_rpa", "gga_c_lyp")`

For the default neural XC path, `b3lyp_component_basis()` continues to return:

```python
("lda_x", "gga_x_b88", "lda_c_vwn_rpa", "gga_c_lyp")
```

and `b3lyp_component_coefficients()` continues to return the default prior coefficients in that channel order, including the HF projected channel where the current neural XC code expects it.

The refactor should make it straightforward for training tools to select a different `semilocal_xc` without hard-coding B3LYP in the backend. This is the main reason for keeping the complete vendored `jax_xc` source tree: users can expand the neural basis to PBE-like, B3LYP-like, SCAN-like, or other supported semilocal channel sets as soon as the required grid features are available.

## RSHOptimizer Compatibility

`training.RSHOptimizer` continues to use `TrainableRSHFunctional.local_xc_spec`. For `nn_rsh.RSH("lc-wpbe").trainable()`, the default local spec remains:

```text
lc_wpbe_local = gga_x_wpbeh + gga_c_pbe
```

The `gga_x_wpbeh` evaluator must remain differentiable with respect to `omega` for:

- local XC energy
- SCF potential construction
- unrestricted energy JVP used by the self-supervised RSH loss
- TDDFT response Hessians where applicable

The range-separated exact-exchange part remains outside `jax_libxc`; it stays in `nn_rsh.functional` through `hfx_nu` interpolation and exact-exchange matrix assembly.

For future RSH experiments, `local_xc_spec` may point to other vendored GGA/mGGA semilocal parts. The RSH optimizer can diversify over local functional forms only when exact-exchange metadata, range-separation parameters, and required grid features are explicit. Unsupported hybrid/RSH forms must fail early with a clear message instead of silently treating them as pure semilocal functionals.

## Vendored jax_xc Usage

Use `sail-sg/jax_xc` as a complete vendored source dependency and correctness reference. The vendored tree must preserve:

- MPL-2.0 license file and notices.
- upstream remote URL and commit hash in a TD-GradDFT metadata file.
- generated formula source paths for any local fallback or patched implementation.
- notes for any intentional numerical stabilization or JAX-version compatibility patch, such as custom JVPs for `E1_scaled` or `erfcx`.

The current `td_graddft._jax_xc_wpbeh` pattern is acceptable as a local fallback for LC-wPBE, but the preferred primary source should be vendored `jax_xc` when available and performant. If vendored `jax_xc` cannot JIT a known expensive formula, TD-GradDFT may route that specific functional to a patched local fallback while recording the source and reason.

## Error Handling

Unsupported functionals should fail early in `parse_xc` with a supported-name list and the reason for rejection, such as missing `tau`, missing Laplacian, unsupported range-separated hybrid metadata, or unavailable vendored source.

A missing external pip `jax_xc` installation must not break local training. A missing or incomplete vendored `third_party/jax_xc` should produce a clear diagnostic and fall back only to the guaranteed compatibility set.

Low-density and low-gradient regularization stays in the backend layer so all callers share the same finite-gradient behavior.

## Tests

The refactor must preserve and extend focused tests:

- Parser aliases and hybrid coefficients: `lda`, `pbe`, `pbe0`, `b3lyp`, `lc_wpbe_local`.
- Vendored `jax_xc` discovery finds the complete local source tree and records its commit or source metadata.
- `NeuralXCTrainer` default basis compatibility through `b3lyp_component_basis()` and prior coefficients.
- `semilocal_xc` can select `pbe`, `b3lyp`, explicit channel tuples, and at least one non-default vendored GGA channel set for neural XC energy channels.
- `lc_wpbe_local` matches PySCF/libxc at representative restricted points.
- `gga_x_wpbeh` is finite and differentiable with respect to `omega`.
- SCF/TDDFT response tensor callers still receive the same shape contract for LDA and GGA.
- `load_jax_xc()` resolves external upstream, vendored upstream, or fallback in that order without making training depend on external pip installation.

## Remote Implementation Notes

Implementation should happen on `/home/yjiao/TD-GradDFT` after either:

- converting the main project into a real Git checkout, or
- accepting that changes are direct file edits without Git safety.

The incomplete `/home/yjiao/TD-GradDFT/third_party/jax_xc` directory should be replaced by a complete `sail-sg/jax_xc` clone or subtree. Runtime training should import it through `td_graddft.jax_xc_adapter`, not by requiring users to manually edit `PYTHONPATH`.

## Success Criteria

- Existing imports from `td_graddft.jax_libxc` continue to work.
- `third_party/jax_xc` is a complete vendored upstream source tree with license and source metadata preserved.
- Both `training.NeuralXCTrainer` and `training.RSHOptimizer` run their existing smoke tests.
- Neural XC tools can control the semilocal energy-density basis through `semilocal_xc`, including at least one vendored non-default functional family.
- LC-wPBE `omega` gradients remain finite in the RSH training path.
- The backend is split into smaller modules without broad behavioral changes.
