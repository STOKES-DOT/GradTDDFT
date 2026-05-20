# TD-GradDFT/GradTDDFT

TD-GradDFT/GradTDDFT is a JAX research codebase for differentiable ground-state DFT,
TDDFT/TDA response calculations, and Neural XC/RSH training. The Python package
name is `td-graddft`; the import namespace is `td_graddft`.

This repository is an active research prototype. APIs are kept practical for
experiments, but the most stable entry points are:

```python
from td_graddft import gto, scf, dft, tdscf, neural_xc, nn_rsh, training, workflows
```

## What The Project Does

- PySCF-style molecule and solver facades: `gto.M`, `scf.RKS`, `scf.UKS`,
  `dft.RKS`, `dft.UKS`, `mf.TDA()`, and `mf.TDDFT()`.
- Restricted and unrestricted SCF paths with differentiable JAX components.
- Restricted and unrestricted TDA and Casida TDDFT response solvers.
- Neural XC functionals with semilocal, HF, and optional PT2 local channels.
- Strict and approximate response-kernel paths for HF/PT2 Neural XC channels.
- Neural range-separated hybrid functionals with trainable RSH parameters.
- GPU4PySCF-backed reference and training workflows for GPU SCF work.
- Research scripts for H2 dissociation, S1 training, QH9-style benchmarks, and
  closed-shell excited-state training.

## Current Backend Model

TD-GradDFT separates three concerns:

1. Molecule, grid, integral, SCF, and TDDFT data structures live in
   `td_graddft.gto`, `td_graddft.scf`, `td_graddft.dft`, and
   `td_graddft.tddft`.
2. Traditional XC energy-density components are routed through
   `td_graddft.xc_backend`.
3. Trainable Neural XC functionals are constructed through
   `td_graddft.neural_xc`.

Traditional XC evaluation now expects the active Python environment to provide
`jax_xc`. TD-GradDFT no longer relies on a vendored generated `jax_xc` fallback.
Install the `upstreams` extra for the standard development environment:

```bash
python -m pip install -e ".[dev,upstreams]"
```

The `upstreams` extra installs `jax-xc` and `pyscf`.

## Repository Layout

```text
src/td_graddft/
  gto/                 PySCF-style molecule input
  scf/                 RHF/RKS/UKS solvers, facades, GPU4PySCF bridges
  dft/                 XC/RSH helpers exposed through DFT-style facades
  tdscf/               PySCF-style TDA and TDDFT facades
  tddft/               Response matrices, TDA/Casida solvers, response kernels
  xc_backend/          jax_xc adapter, XC parser, RSH preset metadata
  neural_xc/           Neural XC construction, channels, response logic
  nn_rsh/              Neural range-separated hybrid models(under coding!)
  training/            Ground-state and excited-state training utilities
  workflows/           Config-driven workflow helpers
  data/                Basis data, grids, integral helpers, references

examples/              End-to-end examples and small comparisons
tools/                 Research, benchmark, and training drivers
scripts/               Shell/Python workflow entry points
tests/                 Unit and regression tests
```

Generated outputs, checkpoints, plots, remote run logs, and temporary artifacts
should stay outside source control unless a script explicitly needs them as a
small test fixture.

## Installation

### CPU Development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,upstreams]"
```

### GPU Workflows

Install a CUDA-enabled JAX build and a GPU4PySCF environment that matches the
target machine. TD-GradDFT GPU SCF training workflows use GPU4PySCF rather than
project-owned CUDA kernels.

Useful runtime checks:

```bash
python - <<'PY'
import jax
import td_graddft
from td_graddft.xc_backend import jax_xc_backend_info

print(jax.devices())
print(jax_xc_backend_info())
print(td_graddft.__name__)
PY
```

## Traditional XC Support

Traditional XC names are parsed by `td_graddft.xc_backend.jax_libxc`. The parser
accepts PySCF-like composite strings, canonical `jax_xc` names, and a small set
of user-friendly component aliases.

### Strict Default Components

These components are allowed by default when `jax_xc` is installed:

```text
lda_x
lda_c_pw
lda_c_vwn
lda_c_vwn_rpa
gga_x_b88
gga_x_pbe
gga_x_wpbeh
gga_c_lyp
gga_c_pbe
```

`hf` is parsed as an exact-exchange term, but it is not a semilocal
`jax_xc` energy-density component.

### Composite Aliases

Common aliases expand into explicit components:

```text
lda
svwn
svwn_rpa
pbe
pbe0 / pbeh / hyb_gga_xc_pbeh
b3lyp / hyb_gga_xc_b3lyp
bhandhlyp / hyb_gga_xc_bhandhlyp
lc_wpbe_local / lc-wpbe-local / lcwpbe_local / lc_wpbe_semilocal
```

Safe wrapped composites include PBE0/PBEH, B3LYP, BHandHLYP, HSE03, and HSE06.
They are resolved into explicit local child channels where possible.

### Friendly Component Names

Neural XC component lists can use canonical `jax_xc` names or friendlier names:

```text
lyp_c      -> gga_c_lyp
b88_x      -> gga_x_b88
pbe_x      -> gga_x_pbe
pbe_c      -> gga_c_pbe
scan_x     -> mgga_x_scan
scan_c     -> mgga_c_scan
r2scan_x   -> mgga_x_r2scan
tpss_c     -> mgga_c_tpss
```

Family-qualified names are also accepted:

```text
gga:lyp_c
mgga:scan_x
hyb_gga:some_xc_name
```

Ambiguous names such as `scan`, `pbe`, or `pw91` are rejected as individual
components. Use `scan_x`/`scan_c`, `pbe_x`/`pbe_c`, or a full composite XC spec
instead.

`jax_xc` kinetic-energy functionals such as `gga_k_*` are rejected for XC
component lists by default.

### Experimental jax_xc Components

Installed `jax_xc` LDA/GGA/MGGA/hybrid names that are not in the strict default
set are discovered dynamically. They are marked experimental and require:

```python
allow_experimental_jax_xc=True
```

This is intentional. MGGA components need a tau/mo_fn bridge, and some generated
hybrid or B97-family forms have not been validated against the training paths.

Inspect support status programmatically:

```python
from td_graddft import neural_xc

for info in neural_xc.available_semilocal_component_infos(include_experimental=True):
    print(info.name, info.status, info.family, info.reason)
```

## Neural XC Construction

The main user entry point is:

```python
from td_graddft import neural_xc

functional = neural_xc.Functional(
    architecture="residual",
    semilocal_xc=("lda_x", "gga_x_b88", "lyp_c"),
    hidden_dims=(256, 256, 256),
    input_feature_mode="canonical",
    include_pt2_channel=True,
    pt2_channel_mode="scaled_projected",
    response_hf_mode="strict",
    response_pt2_mode="strict",
)
```

The local Neural XC form is:

```text
e_xc^NN(r) =
    sum_k c_k^theta(r) e_k^semilocal(r)
  + c_pt2^theta(r) e_pt2(r)       # optional
  + c_hf^theta(r) e_hf(r)
```

The basis-channel order is fixed:

```text
[semilocal_1, ..., semilocal_n, pt2?, hf]
```

Important construction fields:

- `semilocal_xc`: semilocal basis components or composite specs. Examples:
  `("lda_x", "gga_x_b88", "gga_c_lyp")`, `"pbe"`, or `"b3lyp"`.
- `include_pt2_channel`: adds a PT2 local channel before the HF channel.
- `pt2_channel_mode`: currently `scaled_projected` or `local_exact`.
- `response_hf_mode`: `approx` or `strict`.
- `response_pt2_mode`: `approx` or `strict`.
- `input_feature_mode`: `canonical` or `enhanced`.
- `architecture`: `residual`/`graddft_residual` or `simple_mlp`.
- `allow_experimental_jax_xc`: opt in to unvalidated installed `jax_xc`
  components.

The default Neural XC semilocal basis is the B3LYP local decomposition:

```text
("lda_x", "gga_x_b88", "lda_c_vwn_rpa", "gga_c_lyp")
```

with default coefficient priors:

```text
(0.08, 0.72, 0.19, 0.81, 0.20)
```

The final value is the HF coefficient prior.

### Config-Style Construction

The same functional can be specified through dataclasses:

```python
from td_graddft import neural_xc

config = neural_xc.Config(
    components=neural_xc.ComponentSpec(
        semilocal=("pbe_x", "pbe_c"),
        allow_experimental_jax_xc=False,
    ),
    channels=neural_xc.ChannelSpec(
        hf="spin_resolved",
        pt2="scaled_projected",
        response_hf="strict",
        response_pt2="strict",
    ),
    network=neural_xc.NetworkSpec(
        architecture="residual",
        hidden_dims=(256, 256, 256),
    ),
)

functional = neural_xc.Functional(config=config)
```

### Custom Non-HF Channels

Advanced users can provide a `SemilocalEnergyDensityModule` or a custom
`energy_density_channels_fn`. This is useful when the semilocal part is not a
plain `jax_xc` component list.

```python
module = neural_xc.make_libxc_semilocal_module(("gga_x_pbe", "gga_c_pbe"))

functional = neural_xc.Functional(
    non_hf_module=module,
    include_pt2_channel=False,
)
```

## Ground-State And Excited-State Examples

### RKS Ground State

```python
from td_graddft import gto, scf

mol = gto.M(
    atom="H 0 0 0; H 0 0 0.74",
    basis="sto-3g",
    unit="Angstrom",
)

mf = scf.RKS(mol, xc="pbe")
mf.grids_level = 0
energy = mf.kernel()

print(energy)
print(mf.mo_energy)
```

### TDA And Casida TDDFT

```python
td = mf.TDA()
td.nstates = 3
td.kernel()
print(td.e_ev)

td_full = mf.TDDFT()
td_full.nstates = 3
td_full.kernel()
print(td_full.e_ev)
```

### H2 Neural XC Training Scripts

Representative command-line entry points:

```bash
python tools/h2_self_consistent_ground_train5_dense100_vs_fci.py \
  --basis sto-3g \
  --semilocal-xc lda_x gga_x_b88 lyp_c \
  --include-pt2-channel \
  --steps 100

python tools/h2_s1_tda_train5_dense100_vs_fci.py \
  --basis sto-3g \
  --semilocal-xc lda_x gga_x_b88 lyp_c \
  --include-pt2-channel \
  --response-pt2-mode strict \
  --steps 100
```

For production H2 S1 dissociation experiments we usually test with `sto-3g` and
then train with `6-31+g*`, using GPU4PySCF for ground-state SCF work.

## Basis Sets

Basis names use PySCF-style spelling and are normalized internally. Common
examples include:

```text
sto-3g
6-31g
6-31g*
6-31+g*
6-31++g**
def2-svp
def2-tzvp
cc-pvdz
aug-cc-pvdz
```

Strict-JAX basis loading reads the bundled PySCF basis snapshot. PySCF-backed
reference paths can use any basis available in the active PySCF installation.

## Testing

Install development dependencies:

```bash
python -m pip install -e ".[dev,upstreams]"
```

Run focused checks:

```bash
pytest -q tests/test_jax_xc_adapter.py
pytest -q tests/test_jax_libxc.py
pytest -q tests/test_neural_xc_public_api.py
pytest -q tests/test_neural_xc_runtime.py
pytest -q tests/test_tddft.py
```

Some tests require PySCF, GPU4PySCF, CUDA-visible devices, or generated
reference data. Those tests should either skip cleanly or fail with an explicit
backend availability error.

## Development Notes

- Prefer public namespaces in scripts: `gto`, `scf`, `dft`, `tdscf`,
  `neural_xc`, `nn_rsh`, `training`, and `workflows`.
- Keep XC component parsing centralized in `td_graddft.xc_backend.jax_libxc`.
- Keep installed `jax_xc` discovery centralized in
  `td_graddft.xc_backend.jax_xc_adapter`.
- Do not add a second top-level `td_graddft.jax_libxc` or
  `td_graddft.jax_xc_adapter` module.
- Keep GPU SCF paths routed through GPU4PySCF unless a new backend has a clear
  CPU/JAX fallback and tests.
- Keep run outputs in ignored directories such as `outputs/`, `tmp/`, or
  `artifacts/`.

## Upstreams

TD-GradDFT interoperates with:

- JAX, Flax, and Optax for differentiable numerical work.
- `jax_xc` for JAX-native XC components translated from libxc.
- PySCF for reference calculations, basis validation, and GPU4PySCF workflows.

Third-party source snapshots should keep their original license notices next to
the copied files.
