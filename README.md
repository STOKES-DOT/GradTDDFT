# GradTDDFT / TD-GradDFT

`TD-GradDFT` is a research codebase for time-dependent differentiable density
functional theory in JAX. It provides a PySCF-style molecular API, strict-JAX
SCF and TDDFT/TDA building blocks, Neural XC and neural range-separated hybrid
training utilities, and benchmark workflows for spectra, fractional charge, and
geometry-response experiments.

The package name is `td-graddft`; the import namespace is `td_graddft`.

## Repository Description

GitHub description:

> JAX research toolkit for differentiable TDDFT: strict-JAX SCF, TDA/Casida TDDFT, Neural XC/RSH training, spectra, integrals, and CUDA J/K experiments.

## Project Status

This repository is an active research prototype. The stable public entry points
are the compact namespaces below:

```python
from td_graddft import gto, scf, dft, tdscf, neural_xc, nn_rsh, training, workflows
```

Lower-level modules such as `td_graddft.tddft`, `td_graddft.data.integrals`,
`td_graddft.reference`, and `td_graddft.scf.run_*_from_integrals` are available
for experiments and regression tests, but new scripts should prefer the public
facades unless they need direct solver internals.

## Features

- PySCF-style molecule and solver facades:
  `gto.M`, `scf.RKS`, `scf.UKS`, `dft.RKS`, `dft.UKS`, `mf.TDA()`, and
  `mf.TDDFT()`.
- Strict-JAX ground-state paths for restricted and unrestricted Kohn-Sham
  calculations, with differentiable SCF machinery for geometry and training
  experiments.
- Restricted and unrestricted excited-state solvers for TDA and Casida TDDFT,
  including excitation energies, transition dipoles, oscillator strengths, and
  broadened spectra.
- Neural XC functionals in the GradDFT/DM21 style, implemented with
  JAX/Flax/Optax and exposed through `neural_xc.Functional(...)`.
- Neural range-separated hybrid functionals through `nn_rsh.RSH(...)`, including
  trainable `omega`, `alpha`, and `beta` parameters plus atom-centered and GNN
  heads.
- Multiple integral and J/K paths: PySCF/libcint references, pure-JAX Cartesian
  Gaussian one- and two-electron integrals, density fitting, direct J/K, and
  optional CUDA FFI acceleration.
- Training utilities for ground-state energy/density matching, self-consistent
  losses, excited-state fine tuning, Koopmans/Janak-style RSH objectives, and
  fractional-charge analysis.
- Reusable workflows for water, benzene, QH9-style benchmarks, TDDFT spectrum
  comparisons, geometry optimization, frequency analysis, and plotting.

## Repository Layout

```text
src/td_graddft/
  gto/                 PySCF-style molecule input
  scf/                 RHF/RKS/UKS solvers, facades, J/K backends, CUDA FFI
  dft/                 XC parsing, RSH presets, trainable RSH helpers
  tdscf/               PySCF-style TDA and TDDFT facades
  tddft/               Response matrices, TDA/Casida solvers, eigensolvers
  neural_xc/           Neural XC public API and DM21-like functionals
  nn_rsh/              Neural range-separated hybrid package
  training/            Ground-state, excited-state, and RSH training utilities
  workflows/           Config-driven training and spectrum pipelines
  data/                Molecules, basis data, grids, integral engines
  df/                  Density-fitting J/K helpers
  traditional_xc/      Classic XC functional wrappers
  realtime.py          Density-matrix propagation utilities
  spectra.py           Spectra, oscillator strengths, transition dipoles

src/td_graddft_tools/
  fractional_charge/   Piecewise-linearity and fractional-charge workflows
  geomopt_freq/        Geometry optimization and harmonic frequency tools

examples/              End-to-end scripts and small benchmark drivers
tools/                 Research and benchmark utilities
docs/                  Architecture notes and implementation plans
tests/                 Unit and regression tests
```

Local run products such as `outputs/`, `tmp/`, `artifacts/`, `build/`, caches,
and `.tmp_*` files are not part of the source release.

## Installation

### Requirements

- Python 3.10 or newer.
- A working C/C++ build toolchain for packages that compile native extensions.
- JAX and `jaxlib`; use a CPU or CUDA build that matches your hardware.
- Optional: PySCF for reference calculations and PySCF/libcint-backed chemistry
  workflows.
- Optional: CUDA, `nvcc`, and a GPU-enabled JAX install for CUDA FFI direct J/K.

### CPU Development Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

This installs the core JAX/Flax/Optax stack and test tooling.

### Install With Optional Upstream Chemistry Dependencies

```bash
python -m pip install -e ".[dev,upstreams]"
```

The `upstreams` extra installs:

- `pyscf`, used for reference calculations and optional comparison tests.
- `jax-xc`, used for JAX-native libxc-style XC functionals where available.

### GPU And CUDA FFI Notes

Install the CUDA-enabled `jaxlib` build that matches your CUDA runtime first.
Then install this package as usual:

```bash
python -m pip install -e ".[dev,upstreams]"
```

Optional CUDA FFI kernels can be built during packaging:

```bash
TD_GRADDFT_BUILD_CUDA_FFI=1 \
TD_GRADDFT_CUDA_ARCH=native \
python -m pip install -e .
```

Useful environment variables:

```bash
export TD_GRADDFT_NVCC=/usr/local/cuda/bin/nvcc
export TD_GRADDFT_CUDA_ARCH=sm_90
export TD_GRADDFT_JAX_CACHE_DIR=.jax_cache
export TD_GRADDFT_CUDA_JK_LIBRARY=/path/to/libtd_graddft_cuda_direct_jk.so
```

If `TD_GRADDFT_CUDA_JK_LIBRARY` points to a prebuilt library, runtime compilation
is skipped.

## Supported Methods

| Area | Current support |
| --- | --- |
| Molecules | Cartesian AO molecular input via `gto.M`; charge, spin, Angstrom/Bohr style specs. |
| Ground state | RHF kernels, RKS and UKS facades, differentiable SCF, PySCF/libcint reference bridges. |
| SCF backends | Full J/K, density fitting for RKS, direct J/K, optional CUDA direct J/K for supported Cartesian basis sets. |
| XC models | Local JAX parser for core LDA/GGA/hybrid channels, optional `jax_xc` bridge for libxc-translated functionals, classic helpers for LDA/PBE/PBE0/B3LYP-style workflows, and RSH presets such as LC-wPBE and wB97X-D. |
| TDDFT | Restricted and unrestricted TDA, restricted and unrestricted Casida TDDFT, dense and Davidson-style eigensolver paths. |
| Observables | Excitation energies, eV conversion, transition dipoles, oscillator strengths, Lorentzian spectra. |
| Neural XC | Residual and MLP Neural XC functionals, DM21-like feature modes, HF/PT2 channels, long-range correction heads. |
| Neural RSH | Trainable `omega`, `alpha`, `beta`; minimal parameter heads, atom-centered density descriptors, GNN parameter heads. |
| Training | Ground-state energy/density/orbital losses, fixed-density and self-consistent modes, excited-state constraints, RSH Koopmans/Janak objectives. |
| Analysis tools | Fractional-charge scans, geometry optimization, harmonic frequencies, benchmark and plotting scripts. |

Integral-engine notes:

- Pure-JAX Cartesian Gaussian integral validation supports angular momentum up
  to `l <= 3`.
- Auto-JIT paths are optimized for common `s`/`p` workloads; higher angular
  momentum or large contractions can fall back to safer non-JIT paths.
- CUDA direct J/K is optional and depends on GPU-enabled JAX, `nvcc` or a
  prebuilt shared library, and the supported CUDA basis/kernel configuration.

## Supported Basis Sets

Basis names follow PySCF-style spelling and are normalized internally, so common
forms such as `sto-3g`, `6-31g*`, `def2-svp`, and `cc-pvdz` are accepted. The
strict-JAX basis loader reads the bundled `.dat` snapshot in
`src/td_graddft/data/pyscf_basis_snapshot`; PySCF-backed reference paths can also
use basis names that an installed PySCF environment can resolve.

Representative bundled orbital basis families:

- Minimal and compact bases: `sto-3g`, `sto-6g`, `dz`, `dzp`, `dzvp`, `dzvp2`,
  `tzv`, `tzp`, `qzp`, `ano`, `roos-dz`, `roos-tz`, `adzp`, `atzp`, `aqzp`.
- Pople bases: `3-21g`, `3-21g*`, `3-21++g`, `3-21++g*`, `4-31g`, `6-31g`,
  `6-31g*`, `6-31g**`, `6-31+g`, `6-31+g*`, `6-31+g**`, `6-31++g`,
  `6-31++g*`, `6-31++g**`, `6-311g`, `6-311g*`, `6-311g**`, `6-311+g`,
  `6-311+g*`, `6-311+g**`, `6-311++g`, `6-311++g*`, and `6-311++g**`.
- Dunning correlation-consistent bases: `cc-pvdz`, `cc-pvtz`, `cc-pvqz`,
  `cc-pv5z`, `aug-cc-pvdz`, `aug-cc-pvtz`, `aug-cc-pvqz`, `aug-cc-pv5z`,
  `cc-pcvdz`, `cc-pcvtz`, `cc-pcvqz`, `cc-pcv5z`, `cc-pcv6z`, `cc-pwcvdz`,
  `cc-pwcvtz`, `cc-pwcvqz`, `cc-pwcv5z`, plus DK, DK3, PP, and PP-NR variants.
- F12 and auxiliary Dunning variants: `cc-pvdz-f12`, `cc-pvtz-f12`,
  `cc-pvqz-f12`, `cc-pv5z-f12`, `cc-pvdz-f12rev2`, `cc-pvtz-f12rev2`,
  `cc-pvqz-f12rev2`, `cc-pv5z-f12rev2`, and OptRI companion bases.
- Karlsruhe/def2 bases: `def2-svp`, `def2-svpd`, `def2-tzvp`, `def2-tzvpd`,
  `def2-tzvpp`, `def2-tzvppd`, `def2-qzvp`, `def2-qzvpd`, `def2-qzvpp`,
  `def2-qzvppd`, `def2-mtzvp`, `def2-mtzvpp`, `ma-def2-svp`, `ma-def2-svpp`,
  `ma-def2-tzvp`, `ma-def2-tzvpp`, `ma-def2-qzvp`, and `ma-def2-qzvpp`.
- Jensen polarization-consistent bases: `pc-0` through `pc-4`, `aug-pc-0`
  through `aug-pc-4`, `pcseg-0` through `pcseg-4`, and `aug-pcseg-0` through
  `aug-pcseg-4`.
- Relativistic, ECP, and heavy-element families: `lanl2dz`, `lanl2tz`,
  `lanl08`, `stuttgart_dz`, `stuttgart_rsc`, `bfd_vdz`, `bfd_vtz`, `bfd_vqz`,
  `bfd_v5z`, `bfd_pp`, `Burkatzi-Filippi-Dolg-PP`, `crenbl`, `crenbs`,
  `sbkjc`, `sarc-dkh2`, `ccECP`, `ccECP_cc-pVDZ` through `ccECP_cc-pV6Z`,
  `ccECP_aug-cc-pVDZ` through `ccECP_aug-cc-pV6Z`, and spin-orbit ECP data.
- Fitting and auxiliary bases: `def2-*-ri`, `def2-universal-jfit`,
  `def2-universal-jkfit`, `cc-pv*z-ri`, `cc-pv*z-jkfit`, `cc-pV*Z_MP2FIT`,
  `weigend_cfit`, `ahlrichs_cfit`, `demon_cfit`, `DgaussA1_dft_cfit`,
  `DgaussA2_dft_cfit`, `DgaussA1_dft_xfit`, and `DgaussA2_dft_xfit`.

Backend caveats:

- Strict-JAX molecule construction currently accepts named basis strings from
  the bundled `.dat` snapshot and converts them to Cartesian AOs.
- The pure-JAX integral implementation validates angular momentum up to `l <= 3`.
  Higher angular momentum basis functions should use a PySCF/libcint-backed path
  unless the target JAX integral path explicitly supports them.
- Auxiliary RI/JFIT/JKFIT/MP2FIT files are included for density-fitting and
  reference workflows; they are not always appropriate as primary orbital bases.

## Supported Exchange-Correlation Functionals

TD-GradDFT has three XC layers: a guaranteed local JAX compatibility layer, an
optional `jax_xc` backend, and neural/trainable functional wrappers.

Guaranteed local JAX parser support:

- Primitive channels: `lda_x`, `lda_c_pw`, `lda_c_vwn`, `lda_c_vwn_rpa`,
  `gga_x_b88`, `gga_x_pbe`, `gga_x_wpbeh`, `gga_c_lyp`, `gga_c_pbe`, and `hf`.
- Aliases and composites: `lda`, `svwn`, `svwn_rpa`, `pbe`, `pbe0`, `pbeh`,
  `hyb_gga_xc_pbeh`, `b3lyp`, `hyb_gga_xc_b3lyp`, `bhandhlyp`,
  `hyb_gga_xc_bhandhlyp`, `lc_wpbe_local`, `lc-wpbe-local`,
  `lcwpbe_local`, and `lc_wpbe_semilocal`.
- Classic helper constructors: `make_lda_functional()`, `make_pbe_functional()`,
  `make_pbe0_functional()`, and `make_b3lyp_functional()`.

`jax_xc` backend support:

- Installing with `python -m pip install -e ".[upstreams]"` adds
  `jax-xc>=0.0.12`.
- `td_graddft.jax_xc_adapter.load_jax_xc()` resolves backends in this order:
  external `jax_xc`, vendored generated `third_party/jax_xc/generated`, then the
  local fallback subset above.
- The adapter exposes `jax_xc` factories for LDA-like adiabatic wrappers through
  `lda_from_jax_xc(...)` and wraps selected hybrid composite nodes so TD-GradDFT
  handles exact exchange in the SCF/RSH layer while using semilocal JAX
  components from `jax_xc`.
- When a complete upstream or vendored `jax_xc` backend is present, additional
  libxc-translated semilocal names can be used experimentally if their required
  grid features are supplied by the calling path. The strict training and TDDFT
  paths still validate against the guaranteed local subset unless explicitly
  routed through the adapter.

Range-separated and neural XC support:

- RSH presets: `lc-wpbe` and `wb97x-d`, with aliases such as `LC_WPBE`,
  `lc_wpbe`, `wb97xd`, `omega-b97x-d`, and `omega_b97x_d`.
- `lc-wpbe` uses SR-PBE semilocal exchange plus LR-HF exchange and PBE
  correlation. Defaults: `omega=0.4`, SR-HF fraction `0.0`, LR-HF fraction `1.0`.
- `wb97x-d` uses a B97-family range-separated hybrid form with dispersion
  metadata. Defaults: `omega=0.2`, SR-HF fraction `0.222036`, LR-HF fraction
  `1.0`.
- Neural XC supports residual and MLP architectures, DM21-style feature modes,
  HF/PT2 channels, semilocal channel bases from the supported XC specs, and
  trainable long-range correction heads.
- Neural RSH supports trainable `omega`, `alpha`, and `beta` parameters, plus
  atom-centered and GNN parameter heads.

## Quick Start

### 1. Closed-Shell RKS Ground State

```python
from td_graddft import gto, scf

mol = gto.M(
    atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
    basis="sto-3g",
    unit="Angstrom",
    charge=0,
    spin=0,
)

mf = scf.RKS(mol, xc="pbe")
mf.grids_level = 0
mf.max_cycle = 80
energy = mf.kernel()

print("E_tot =", energy)
print("MO energies =", mf.mo_energy)
```

`td_graddft.dft.RKS` is an alias for the same facade:

```python
from td_graddft import dft

mf = dft.RKS(mol, xc="b3lyp").run()
```

### 2. TDA And Full TDDFT Excited States

```python
from td_graddft import tdscf

td = tdscf.TDA(mf)
td.nstates = 5
td.kernel()

print("TDA energies / Ha:", td.e)
print("TDA energies / eV:", td.e_ev)
print("Oscillator strengths:", td.oscillator_strength())

td_full = mf.TDDFT()
td_full.nstates = 5
td_full.kernel()
print("Casida TDDFT energies / eV:", td_full.e_ev)
```

### 3. Open-Shell UKS And Unrestricted TDA

```python
from td_graddft import gto, scf

oh = gto.M(
    atom="O 0 0 0; H 0 0 0.9697",
    basis="6-31g",
    unit="Angstrom",
    charge=0,
    spin=1,
)

mf = scf.UKS(oh, xc="b3lyp")
mf.max_cycle = 120
mf.kernel()

tda = mf.TDA()
tda.kernel(nstates=6)

print(tda.e_ev)
print(tda.oscillator_strength())
```

### 4. Select SCF And J/K Backends

```python
from td_graddft import scf

# Dense/full J/K, the default RKS path.
mf_full = scf.RKS(mol, xc="pbe0")
mf_full.kernel()

# Density-fitting RKS.
mf_df = scf.RKS(mol, xc="pbe0").density_fit()
mf_df.kernel()

# Direct J/K RKS on the JAX/libcint path.
mf_direct = scf.RKS(mol, xc="pbe0").direct_scf()
mf_direct.kernel()

# CUDA direct J/K when CUDA FFI is available.
mf_cuda = scf.RKS(mol, xc="pbe0")
mf_cuda.cuda_direct_scf(execution_device="gpu", precompile=True)
mf_cuda.kernel()
```

If `execution_device="gpu"` is requested and CUDA FFI is unavailable, the CUDA
path raises an explicit error. With `execution_device="auto"`, it falls back to
the non-CUDA path.

### 5. Neural XC Ground-State Training

```python
from td_graddft import neural_xc, training

# Use a converged reference from an SCF run.
datum = training.GroundStateDatum.from_reference(mf.reference)

functional = neural_xc.Functional(
    architecture="residual",
    hidden_dims=(64, 64),
    input_feature_mode="dm21_original",
)

trainer = training.NeuralXCTrainer(
    functional=functional,
    molecules=[datum],
)

result = trainer.kernel(
    steps=50,
    learning_rate=1e-4,
    loss="ground_state",
)

print(result.history["loss"][-1])
print(result.params)
```

### 6. Neural RSH Parameter Optimization

```python
from td_graddft import nn_rsh, training

rsh_functional = nn_rsh.RSH("lc-wpbe").trainable(
    params=("omega", "alpha", "beta"),
    hidden_dims=(32,),
)

optimizer = training.RSHOptimizer(
    functional=rsh_functional,
    molecules=[datum],
)

rsh_result = optimizer.kernel(
    steps=50,
    learning_rate=1e-3,
    loss="koopmans_ip_ea",
)

print("omega history:", rsh_result.history["omega"])
print("alpha history:", rsh_result.history["alpha"])
print("beta history:", rsh_result.history["beta"])
```

### 7. Config-Driven Spectrum Workflow

```python
from dataclasses import replace

from td_graddft.workflows import (
    run_experiment,
    water_experiment_config,
)

config = water_experiment_config(
    basis="sto-3g",
    xc="b3lyp",
    steps=200,
)

config = replace(
    config,
    simulation=replace(config.simulation, nstates=8, execution_device="auto"),
)

run = run_experiment(config)
print(run.runs[0].outputs.spectrum_csv)
print(run.runs[0].outputs.spectrum_png)
```

For command-line use, see:

```bash
python examples/run_workflow_experiment.py --system water --basis sto-3g --steps 200 --states 8
python examples/compare_pyscf_vs_jax_tddft_no_neural.py --xc-mode hf --basis sto-3g
python examples/run_oh_open_shell_tda.py --xc b3lyp --basis 6-31g --nstates 6
```

## Tests

Install development dependencies, then run:

```bash
pytest -q
```

Useful focused checks:

```bash
pytest -q tests/test_pyscf_style_ground_state_api.py
pytest -q tests/test_pyscf_style_excited_state_api.py
pytest -q tests/test_neural_xc_public_api.py
pytest -q tests/test_integrals_jax.py
```

Some tests require optional dependencies such as PySCF, CUDA, or a visible GPU;
those tests skip or fail explicitly when the required backend is unavailable.

## Development Notes

- Prefer the public namespaces for scripts:
  `gto`, `scf`, `dft`, `tdscf`, `neural_xc`, `nn_rsh`, `training`, `workflows`.
- Keep generated results in `outputs/`, `tmp/`, or `artifacts/`; these paths are
  ignored for source publishing.
- Use `src/td_graddft_tools/` for reusable analysis helpers and `tools/` for
  one-off benchmark drivers.
- Keep PySCF-dependent paths optional where possible; strict-JAX workflows should
  continue to run without importing PySCF at runtime.
- CUDA FFI code should always have a CPU/JAX fallback or a clear availability
  error.

## Related Upstreams

This project is designed to interoperate with:

- GradDFT-style differentiable ground-state abstractions.
- PySCF for reference chemistry calculations and basis/integral validation.
- `jax_xc` for JAX-native XC components translated from libxc.

The repository does not vendor GradDFT or PySCF. If third-party code is copied
into the tree in the future, preserve the original license notices next to the
copied files.
