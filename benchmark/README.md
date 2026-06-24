# GradTDDFT Benchmarks

This directory stores manuscript-facing benchmark runs and their
machine-readable outputs. Each subdirectory should contain the run
script, task manifest, progress logs, per-state CSV data for plotting,
and per-task summary tables.

For manuscript plotting, use `plot_data/` first. It is a normalized
index that copies the plot-ready CSV/JSON files from `outputs/` and
legacy benchmark run folders into task-based paths:

- `plot_data/validation/`
- `plot_data/dissociation/ground_state/`
- `plot_data/dissociation/excited_state/`
- `plot_data/loss_curves/`
- `plot_data/reference_curves/`
- `plot_data/semilocal_channel/`

`plot_data/MANIFEST.csv` and `plot_data/MANIFEST.json` record the
source path, destination path, row count, file size, checksum, task,
system, state, variant, and file role for each copied artifact.

Do not treat terminal output as the source of record. Keep the CSV and
JSONL artifacts together with the exact run command and environment
metadata.
