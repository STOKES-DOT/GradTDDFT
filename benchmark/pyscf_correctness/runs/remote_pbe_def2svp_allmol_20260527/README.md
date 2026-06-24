# Remote PBE/def2-SVP First Batch

Host: `c20`

Environment: `jax_scf`

Backend: JAX GPU, `CUDA_VISIBLE_DEVICES=GPU-488dbae2-96e3-2698-66f8-d1928726bef8`

Command:

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

Completed tasks: 26/26.

Per-state visualization rows: 130.

Support split:

- GradTDDFT restricted singlet comparisons: 12 tasks.
- PySCF-only restricted triplet references: 12 tasks.
- PySCF-only unrestricted OH references: 2 tasks.

Observed GradTDDFT-vs-PySCF restricted singlet energy MAE range:
`5.861977570020827e-14` to `2.871072268817443e-10` eV.
