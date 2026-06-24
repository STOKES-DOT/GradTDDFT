# PySCF TDDFT Functional Support Scan

Remote environment:

- PySCF 2.13.0
- Python 3.11.15
- Host output directory: `/home/yjiao/TD-GradDFT/benchmark/pyscf_correctness/runs/remote_pyscf_tddft_functional_support_alias_common_20260528`

Command:

```bash
python benchmark/pyscf_correctness/scan_pyscf_tddft_functional_support.py \
  --output-dir benchmark/pyscf_correctness/runs/remote_pyscf_tddft_functional_support_alias_common_20260528 \
  --smoke-source aliases-and-common \
  --timeout-s 15
```

Scope:

- `pyscf_xc_deriv2_support.csv`: derivative-order check for 981 LibXC codes, 70 PySCF aliases, and 26 common extra names.
- `pyscf_tddft_smoke_support.csv`: H2/STO-3G RKS smoke test for 70 PySCF aliases plus 26 common extra names; each functional is run in a separate process with a 15 s timeout.
- `pyscf_tddft_unsupported_smoke.csv`: smoke failures or derivative-check failures.

Summary:

- All 981 LibXC codes and all 70 PySCF aliases report support for `deriv=2`, which is the formal `f_xc` requirement used in PySCF TDDFT/TDA.
- Among the 96 smoke-tested alias/common names, 87 completed SCF, TDA, and full TDDFT.
- The true PySCF calculation-chain failures were:
  - `R2SCANL`: `NotImplementedError: laplacian in meta-GGA method`
  - `SCANL`: `NotImplementedError: laplacian in meta-GGA method`
  - `wb97x-d`: `NotImplementedError: wb97x-d is not supported yet.`
  - `wb97x_d`: `NotImplementedError: wb97x_d is not supported yet.`
- The remaining failures are unresolved or noncanonical names in this PySCF/LibXC entry point:
  - `bhhlyp`: `KeyError: "LibXCFunctional: name 'BHHLYP' not found."`
  - `b2plyp`: `KeyError: "LibXCFunctional: name 'B2PLYP' not found."`
  - `b2gpplyp`: `KeyError: "LibXCFunctional: name 'B2GPPLYP' not found."`
  - `dsd-blyp`: `KeyError: "LibXCFunctional: name 'DSD' not found."`
  - `dsd_blyp`: `KeyError: "LibXCFunctional: name 'DSD_BLYP' not found."`

Interpretation:

For the paper benchmark scope, PBE, B3LYP, PBE0, CAM-B3LYP, SCAN, RSCAN, R2SCAN, M06-family, wB97X, wB97X-V, wB97M-V, B97M-V, and B97-D passed this small PySCF TDA/full-TDDFT smoke. The unsupported set to avoid or explicitly caveat in PySCF baselines is `SCANL`, `R2SCANL`, and `wB97X-D`; double-hybrid labels should not be treated as ordinary PySCF TDDFT functionals in this scan.
