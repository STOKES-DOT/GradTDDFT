# GradTDDFT v1.0.0 Reproducibility Artifacts

This directory contains the compact, source-controlled artifact set for the
numerical tests reported in the GradTDDFT manuscript. It intentionally excludes
HDF5 integral caches, raw QM9 archives, temporary logs, profiling runs, and
intermediate checkpoints.

## Contents

- `figures/`: final manuscript figures and QM9 structure sheets.
- `checkpoints/`: one selected checkpoint per reported Neural XC model.
- `results/`: final inference CSV/JSON files used to validate each task.
- `references/`: compact external reference tables needed by the training and
  inference scripts.
- `SHA256SUMS`: content hashes for every artifact in this directory.

## Manuscript Task Map

| Task | Training/evaluation entry point | Release artifacts |
| --- | --- | --- |
| PySCF numerical validation | `tools/compare_pyscf_vs_jax_same_xc.py`, `tools/trace_benzene_pbe_tddft_davidson.py` | `results/validation/`, Davidson figure |
| H2+ ground-state dissociation | `tools/h2plus_fci_ground_train5_dense100.py` | Two historical manuscript figures; v1.0.0 checkpoint pending nograd retraining |
| H2 ground-state dissociation | `tools/h2_self_consistent_ground_train5_dense100_vs_fci.py` | `checkpoints/h2_ground_*`, `results/h2_ground/`, H2 ground figures |
| N2 ground-state dissociation | `tools/n2_ccsdt_ground_train5.py` | `checkpoints/n2_ground_*`, `results/n2_ground/`, N2 ground figures |
| H2 first-excited-state dissociation | `tools/h2_s1_tda_train5_dense100_vs_fci.py` | `checkpoints/h2_s1_*`, `results/h2_s1/`, H2 S1 figures |
| N2 first-excited-state dissociation | `tools/h2_s1_tda_train5_dense100_vs_fci.py` with the external N2 reference CSV | `checkpoints/n2_s1_*`, `results/n2_s1/`, N2 S1 figures |
| QM9 ground-state training | `tools/closed_shell_s1_self_consistent_train.py` with zero excitation-gap weights | `checkpoints/qm9_ground_step580.msgpack`, `results/qm9_ground/` |
| QM9GWBSE S1/TDA training | `tools/closed_shell_s1_self_consistent_train.py` | `checkpoints/qm9_s1_def2svp_scf32_step1150.msgpack`, `results/qm9_s1/` |
| Checkpoint inference | `tools/evaluate_closed_shell_checkpoint.py` | QM9 validation predictions and compatibility summaries |

## Selected Reported Metrics

- H2 ground-state MAE: 0.0100 eV with PT2 and 0.1112 eV without PT2.
- N2 ground-state MAE: 0.0095 eV with PT2 and 0.2437 eV without PT2.
- H2 S1-total MAE: 0.0182 eV with PT2 and 0.0701 eV without PT2.
- N2 S1-total MAE: 0.161 eV with PT2 and 0.871 eV without PT2.
- QM9 ground-state validation MAE: 0.033 eV.

The authoritative numerical values are the CSV/JSON files in `results/`; the
rounded values above match the manuscript presentation.

## Provenance Notes

- QM9 ground-state inference uses the 40/10 split with
  CCSD(T)/aug-cc-pVTZ targets and the def2-TZVP/grid-level-2 model checkpoint at
  step 580.
- QM9GWBSE inference uses the 40/10 split with qsGW-BSE/TZ3P excitation targets
  and the def2-SVP/grid-level-1 SCF-32 checkpoint at step 1150.
- The H2 and N2 PT2/no-PT2 checkpoints are the checkpoint files used by the
  corresponding saved dense-curve inference runs.
- The original H2+ manuscript models used the unfinished HFX-SCF channel path.
  v1.0.0 supports only fixed-cache `nograd` HFX/PT2 channels, so no H2+
  checkpoint or numerical CSV is shipped. The two manuscript PDFs are retained
  as historical figure-level results pending a nograd retraining run.
- QM9GWBSE reference CSVs use `ccsd_total_energy_h = 0` as a placeholder for
  S1-only records. Ground-state error columns computed from those placeholders
  are not physical; use excitation-gap metrics for the S1 task.

## Scope

These files support result inspection and checkpoint inference. Raw datasets,
large reference caches, and full remote training logs are excluded from the
v1.0.0 source release.
