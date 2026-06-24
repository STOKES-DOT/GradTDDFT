# Step 1: PySCF Correctness

This benchmark implements the first manuscript evaluation:
GradTDDFT in the conventional-functional limit is compared against
matched PySCF TDDFT calculations.

Planned paper matrix:

- Molecules: H2O, CO, N2, C2H4, formaldehyde, benzene, OH
- Functionals: PBE, B3LYP, PBE0
- Bases: def2-SVP, aug-cc-pVDZ
- Closed-shell response: RKS plus TDA/full TDDFT, singlet and triplet
  PySCF references
- Open-shell response: UKS spin-conserving response for OH
- States: first 5 roots whenever available

Current support boundary:

- GradTDDFT restricted singlet TDA/full TDDFT is compared directly
  against PySCF for closed-shell systems.
- Closed-shell triplet and unrestricted semilocal response rows are
  recorded as PySCF-only / pending GradTDDFT support unless a matching
  GradTDDFT response path is added.

Primary outputs:

- `task_manifest.csv`: selected task matrix and support status.
- `visualization_data.csv`: per-state values for plots.
- `summary.csv`: per-task errors, timings, and status.
- `progress.jsonl`: append-only run events and failures.
- `environment.json`: runtime metadata.
- `runs/`: archived remote/local runs and full-matrix manifests.

Current visualization entry point:

- `visualization_data.csv`: remote PBE/def2-SVP first batch, copied from
  `runs/remote_pbe_def2svp_allmol_20260527/visualization_data.csv`.
- `runs/remote_full_matrix_manifest_20260527/task_manifest.csv`: full
  planned Step 1 matrix with 156 tasks.

Smoke run used to validate the pipeline:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python benchmark/pyscf_correctness/run_pyscf_correctness.py --preset smoke
```

Full selected matrix:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python benchmark/pyscf_correctness/run_pyscf_correctness.py --preset paper
```

Remote first-batch command used for the current CSV:

```bash
CUDA_VISIBLE_DEVICES=GPU-488dbae2-96e3-2698-66f8-d1928726bef8 \
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
python benchmark/pyscf_correctness/run_pyscf_correctness.py \
  --preset paper \
  --molecules water co n2 ethylene formaldehyde benzene oh \
  --xcs pbe \
  --bases def2-svp \
  --solvers tda tddft \
  --outdir benchmark/pyscf_correctness/runs/remote_pbe_def2svp_allmol_20260527
```
