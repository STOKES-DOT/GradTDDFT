# Plot Data Index

This directory contains normalized, manuscript-facing plotting data.
It is a copied data index: original run directories under `outputs/`
and legacy `benchmark/` folders are preserved.

## Layout

- `validation/`: PySCF/GradTDDFT correctness and response-kernel
  validation tables.
- `dissociation/ground_state/`: H2+, H2, and N2 ground-state
  dissociation curves.
- `dissociation/excited_state/`: H2 and N2 first-excited-state
  dissociation curves.
- `loss_curves/`: training loss histories and loss-curve PNGs for the
  dissociation models.
- `reference_curves/`: external/reference curves used by dissociation
  plots.
- `semilocal_channel/`: protocol for the N2 noPT2+HFX implicit-SCF
  semi-local channel-composition tests.

## Naming

Directories follow:

`<category>/<state>/<system>/<mode>/`

where `<mode>` records the SCF treatment, HFX/PT2 setting, and training
grid when needed, for example `implicit_scf_hfx_pt2_train7`.

Each leaf directory uses standardized filenames:

- `dense_curve.csv`, `predictions.csv`, or `visualization_data.csv`
  for plot-ready tables.
- `training_points.csv` for training geometries.
- `metrics.json` and `summary.json` for provenance and numerical
  summaries.

## Manifest

`MANIFEST.csv` and `MANIFEST.json` record source path, destination path,
row count, file size, SHA256 checksum, category, system, state, task,
variant, and role.

Copied/written entries: 97
Missing entries: 0
