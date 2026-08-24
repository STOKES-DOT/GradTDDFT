# Changelog

## 1.0.0 - 2026-08-24

### Features

- Provide differentiable restricted and unrestricted SCF with explicit and
  implicit gradient modes.
- Provide restricted and unrestricted TDA/Casida response solvers, Davidson
  eigensolvers, and differentiable Neural XC response paths.
- Support semilocal channels plus fixed-cache `nograd` local-HF and optional
  PT2/CIS(D)-style Neural XC channels. Ground-state HFX/PT2-SCF recomputation is
  not part of v1.0.0.
- Support CPU/libcint, JAX, density-fitting, and GPU4PySCF integral backends.
- Provide the manuscript training and evaluation entry points for H2+, H2, N2,
  QM9 ground-state, and QM9GWBSE S1/TDA tasks.

### Reproducibility

- Add selected manuscript checkpoints, final inference CSV/JSON files, compact
  reference tables, final figures, and a SHA-256 manifest under
  `reproducibility/v1.0.0/`.
- Validate the published QM9 ground and QM9GWBSE S1 checkpoints with the v1.0.0
  inference API.
- Keep only manuscript-facing examples and tools, and document conventional
  DFT/TDDFT plus Neural XC training, checkpoint, and inference workflows.
- Normalize public QM9 naming and remove machine-specific paths from released
  reproducibility metadata.

### Fixes

- Align the matrix-free full-TDDFT Davidson solver with PySCF-style dual
  subspaces, residual-only convergence, and differentiable symplectic Rayleigh
  reconstruction.
- Use canonical `excitation_gap_*` metric names in closed-shell checkpoint
  evaluation summaries.
- Keep restricted two-AO references out of unrestricted spin-axis dispatch and
  keep occupation conversion JIT-safe.
- Remove obsolete training loss/RSH modules and legacy research drivers from
  the public release tree.
