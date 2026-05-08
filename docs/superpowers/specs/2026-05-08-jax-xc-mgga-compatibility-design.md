# JAX-XC MGGA Compatibility Design

## Goal

Make active `jax_xc` MGGA functionals usable as experimental semilocal channels in TD-GradDFT, including Neural XC and the reduced MGGA `f_xc` tensor path.

## Scope

- Dynamically recognize active `jax_xc` names beginning with `mgga_` or `hyb_mgga_`.
- Keep all MGGA names experimental and rejected by default.
- Require `allow_experimental_jax_xc=True` for energy, Neural XC, and response tensor paths.
- Build a local `mo_fn` from `RestrictedFeatureBundle.tau_a + tau_b` so `jax_xc` MGGA factories receive the required orbital callback.
- Preserve the existing reduced response variables `[rho, grad_x, grad_y, grad_z, tau]` and return `(5, 5, ngrids)` tensors for MGGA `f_xc`.

## Non-Goals

- Do not infer exact-exchange fractions from `hyb_mgga_*` names.
- Do not make traditional hybrid MGGA nodes PySCF-equivalent in this pass.
- Do not add Laplacian-carrying `MGGA_LAPL` support to `RestrictedFeatureBundle`; the local density function used for `jax_xc` MGGA has zero Laplacian in this pass.

## Design

`jax_xc_adapter.jax_xc_functional_info()` will classify active `mgga_*` and `hyb_mgga_*` names as `experimental` with family `MGGA`. `list_jax_xc_functionals()` will include active dynamic MGGA names without hard-coding the full upstream list.

`eval_jax_xc_from_restricted_features()` will keep the current LDA/GGA call shape. For MGGA names it will additionally pass an unpolarized `mo_fn(r)` whose Jacobian produces the requested total kinetic-energy density:

```text
tau = sum(|d mo / d r|^2) / 2
```

`jax_libxc.eval_xc_response_tensor()` will accept and forward `allow_experimental_jax_xc`. Its MGGA Hessian already differentiates over `[rho, grad, tau]`; the new adapter dependency on `tau` makes the tensor path consistent with Neural XC response assembly.
