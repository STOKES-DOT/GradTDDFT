# GradTDDFT

GradTDDFT is a JAX toolkit for differentiable Kohn-Sham DFT, TDA/full-TDDFT,
and Neural XC training. The Python package is `td-graddft`; the import namespace
is `td_graddft`.

```python
from td_graddft import dft, gto, neural_xc, tdscf, training
```

## v1.0.0 Scope

The first release contains:

- restricted and unrestricted JAX SCF;
- differentiable explicit (`expl`) and implicit (`impl`) SCF modes;
- matrix-free TDA and full Casida TDDFT with Davidson solvers;
- strict conventional XC response through `jax-xc`;
- residual Neural XC models with semilocal, fixed-cache HFX, and optional
  fixed-cache PT2 channels;
- the training and evaluation drivers used in the GradTDDFT manuscript;
- selected checkpoints, final inference tables, references, and figures under
  `reproducibility/v1.0.0/`.

Raw datasets, HDF5 integral caches, profiling runs, temporary remote scripts,
and unrelated research drivers are not part of the release.

## Installation

Create an environment with Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,upstreams]"
```

The `upstreams` extra installs PySCF and `jax-xc`. For manuscript scripts and
checkpoint evaluation, install:

```bash
python -m pip install -e ".[dev,reproducibility]"
```

GPU runs require a CUDA-enabled JAX build and a GPU4PySCF installation matched
to the CUDA environment. Confirm the active backend before a long run:

```bash
python - <<'PY'
import jax
from td_graddft.xc_backend import jax_xc_backend_info

print(jax.devices())
print(jax_xc_backend_info())
PY
```

Scientific reference calculations should enable JAX float64 before arrays are
constructed:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

## Conventional DFT, TDA, and full TDDFT

The public facade follows the PySCF workflow. This example runs a conventional
B3LYP calculation entirely through the GradTDDFT API:

```python
import jax
jax.config.update("jax_enable_x64", True)

from td_graddft import dft, gto

mol = gto.M(
    atom="""
    O  0.000000  0.000000  0.117790
    H  0.000000  0.755453 -0.471161
    H  0.000000 -0.755453 -0.471161
    """,
    basis="def2-svp",
    unit="Angstrom",
    spin=0,
    charge=0,
)

mf = dft.RKS(
    mol,
    xc="b3lyp",
    grids_level=0,
    integral_backend="cpu",
    execution_device="cpu",
)
energy_h = mf.kernel()
if not mf.converged:
    raise RuntimeError("RKS did not converge")
print("E0 / Ha:", energy_h)

tda = mf.TDA(nstates=3)
tda_result = tda.kernel()
print("TDA / eV:", tda.e_ev)
print("TDA oscillator strengths:", tda.oscillator_strength())

full = mf.TDDFT(nstates=3)
full_result = full.kernel()
if not bool(full_result.converged):
    raise RuntimeError("full-TDDFT Davidson did not converge")
print("full-TDDFT / eV:", full.e_ev)
```

Use `integral_backend="gpu"` and `execution_device="gpu"` in a configured
GPU4PySCF environment. `integral_backend="cpu"` uses the CPU integral path;
the resulting fixed molecular integrals are reused by SCF and response calls.

A complete PySCF-versus-GradTDDFT B3LYP comparison is provided in
`examples/compare_pyscf_vs_jax_tddft_no_neural.py`.

## Build a Neural XC Functional

The manuscript model predicts local mixing coefficients for the B3LYP
semilocal decomposition and optional HFX/PT2 channels:

```text
e_xc^NN(r) = sum_k c_k(r) e_k^semilocal(r)
             + c_HF(r) e_HF(r)
             + c_PT2(r) e_PT2(r)       # optional
```

Construct the model through the public `neural_xc` namespace:

```python
from td_graddft import neural_xc
from td_graddft.xc_backend import b3lyp_component_basis

functional = neural_xc.Functional(
    architecture="graddft_residual",
    hidden_dims=(128, 128, 128, 128),
    semilocal_xc=b3lyp_component_basis(),
    input_feature_mode="canonical",
    include_hfx_channel=True,
    ground_state_hf_mode="nograd",
    include_pt2_channel=False,
    ground_state_pt2_mode="off",
    response_hf_mode="approx",
    response_pt2_mode="approx",
    name="my_neural_xc",
)
```

The channel order is fixed:

```text
[semilocal_1, ..., semilocal_n, pt2?, hf]
```

In v1.0.0, ground-state nonlocal channels support only `off` and `nograd`:

- HFX `nograd` requires a fixed `hfx_fxx` cache.
- PT2 `nograd` requires fixed `pt2_local`; self-consistent Fock construction
  additionally requires `pt2_fock_response`.
- Density-updated ground-state HFX/PT2 `scf` modes are not public in v1.0.0.

Build these caches once when the molecular reference is prepared. They are not
recomputed after each parameter update:

```python
from pyscf import dft as pyscf_dft, gto as pyscf_gto
from td_graddft.data.reference import restricted_reference_from_pyscf

pyscf_mol = pyscf_gto.M(
    atom="H 0 0 0; H 0 0 0.74",
    basis="def2-svp",
    unit="Angstrom",
    verbose=0,
)
pyscf_mf = pyscf_dft.RKS(pyscf_mol)
pyscf_mf.xc = "b3lyp"
pyscf_mf.grids.level = 2
pyscf_mf.kernel()

reference = restricted_reference_from_pyscf(
    pyscf_mf,
    compute_local_hfx_features=True,
    compute_local_hfx_aux=False,
    compute_local_pt2_features=False,
    jk_backend="full",
)
```

Set `compute_local_pt2_features=True`, `include_pt2_channel=True`, and
`ground_state_pt2_mode="nograd"` to train with the fixed PT2 channel.

## Train the Neural XC Model

One API handles fixed-density, explicit-SCF, and implicit-SCF objectives. Target
names include units and physical meaning: `target_e0_total_h` is a ground-state
total energy in hartree; `target_s1_total_h` is an S1 total energy; and
`target_excitation_gaps_h` contains excitation gaps.

```python
import jax
import jax.numpy as jnp
import optax

from td_graddft import training

datum = training.MolecularTrainingDatum(
    molecule=reference,
    target_e0_total_h=jnp.asarray(-1.1372838345),
)
config = training.MolecularTrainingConfig(
    mode="self_consistent",
    scf_gradient_mode="impl",       # use "expl" for unrolled SCF
    e0_total_mse_weight=1.0,
    e0_total_mae_weight=1.0,
    scf_max_cycle=32,
    scf_convergence_metric="energy",
    scf_conv_tol_energy=1e-8,
)

state = training.create_train_state_from_molecule(
    functional,
    jax.random.PRNGKey(0),
    reference,
    optax.adam(1e-3),
)
train_step = training.make_molecular_train_step(
    functional,
    training_config=config,
)

for step in range(100):
    state, metrics = train_step(state, (datum,))
    print(step, float(metrics["total_loss"]))
```

For excited-state training, activate explicit loss components rather than an
ambiguous generic S1 weight:

```python
excited_config = training.MolecularTrainingConfig(
    mode="self_consistent",
    scf_gradient_mode="impl",
    excited_state_solver="tda",
    excitation_gap_mse_weight=1.0,
    excitation_gap_mae_weight=1.0,
    excitation_gap_nstates=1,
)
excited_datum = training.MolecularTrainingDatum(
    molecule=reference,
    target_excitation_gaps_h=jnp.asarray([0.40]),
)
```

The full paper-scale H2 example is
`examples/h2_fci_self_consistent_train.py`. Production H2, H2+, N2, QM9, and
QM9GWBSE commands are listed below.

## Save, Restore, and Infer

Save model parameters together with the architecture and channel configuration:

```python
checkpoint = "outputs/my_neural_xc.msgpack"
training.save_params_checkpoint(
    checkpoint,
    state.params,
    metadata={
        "architecture": "graddft_residual",
        "hidden_dims": [128, 128, 128, 128],
        "ground_state_hf_mode": "nograd",
        "ground_state_pt2_mode": "off",
    },
)

params = training.load_params_checkpoint(checkpoint, template=state.params)
```

Ground-state inference must use the same fixed-density or self-consistent policy
as training:

```python
energy_h = training.predict_ground_state_total_energy(
    params,
    functional,
    reference,
    training_config=config,
)
converged_molecule = training.predict_ground_state_molecule(
    params,
    functional,
    reference,
    training_config=config,
)
```

Run TDA or full-TDDFT from the converged molecule:

```python
from td_graddft.spectra import HARTREE_TO_EV

gaps_h = training.predict_excitation_energies(
    params,
    functional,
    converged_molecule,
    nstates=3,
    use_tda=True,
)
strengths = training.predict_oscillator_strengths(
    params,
    functional,
    converged_molecule,
    nstates=3,
    use_tda=True,
)
print("gaps / eV:", gaps_h * HARTREE_TO_EV)
print("oscillator strengths:", strengths)
```

Set `use_tda=False` for full Casida TDDFT. Excitation-energy derivatives use a
converged-vector implicit differential and do not backpropagate through the
Davidson iteration history. TDA also provides opt-in implicit eigenvector
gradients for oscillator-strength objectives.

## Manuscript Workflows

Only manuscript-facing drivers are included in `tools/`:

| Manuscript task | Entry point |
| --- | --- |
| Conventional PySCF validation | `tools/compare_pyscf_vs_jax_same_xc.py` |
| Benzene Davidson convergence | `tools/trace_benzene_pbe_tddft_davidson.py` |
| H2+ ground-state dissociation | `tools/h2plus_fci_ground_train5_dense100.py` |
| H2 ground-state dissociation | `tools/h2_self_consistent_ground_train5_dense100_vs_fci.py` |
| N2 ground-state dissociation | `tools/n2_ccsdt_ground_train5.py` |
| H2/N2 S1 TDA dissociation | `tools/h2_s1_tda_train5_dense100_vs_fci.py` |
| QM9 ground and QM9GWBSE S1 training | `tools/closed_shell_s1_self_consistent_train.py` |
| Released checkpoint inference | `tools/evaluate_closed_shell_checkpoint.py` |
| QM9/QM9GWBSE baselines and figures | `tools/compute_qm9_ground_classic_baselines.py`, `tools/compare_qm9_pyscf_vs_jax_tda.py`, `tools/plot_qm9_reference_structures.py`, `tools/plot_qm9_val_bars_with_structures.py` |

Example paper commands:

```bash
python tools/h2_self_consistent_ground_train5_dense100_vs_fci.py \
  --basis def2-tzvp \
  --grids-level 2 \
  --ground-state-hf-mode nograd \
  --ground-state-pt2-mode off \
  --steps 2000

python tools/h2_s1_tda_train5_dense100_vs_fci.py \
  --basis def2-tzvp \
  --grids-level 2 \
  --include-pt2-channel \
  --response-pt2-mode strict \
  --steps 2000
```

Selected checkpoints, compact reference CSVs, final inference results, figures,
and provenance are documented in
[`reproducibility/v1.0.0/README.md`](reproducibility/v1.0.0/README.md). Verify
the artifact manifest from that directory with:

```bash
shasum -a 256 -c SHA256SUMS
```

## Traditional XC Support

Conventional XC labels are parsed by `td_graddft.xc_backend.jax_libxc` and
evaluated with `jax-xc`. Strict default components include:

```text
lda_x, lda_c_pw, lda_c_vwn, lda_c_vwn_rpa
gga_x_b88, gga_x_pbe, gga_x_wpbeh
gga_c_lyp, gga_c_pbe
```

Common composites include `lda`, `svwn`, `pbe`, `pbe0`, `b3lyp`, BHandHLYP,
HSE03, and HSE06. B3LYP is resolved as:

```text
0.20*hf + 0.08*lda_x + 0.72*gga_x_b88
        + 0.19*lda_c_vwn_rpa + 0.81*gga_c_lyp
```

Installed functionals outside the validated set require
`allow_experimental_jax_xc=True`.

## Repository Layout

```text
src/td_graddft/       DFT, SCF, TDDFT, Neural XC, training, and data APIs
src/td_graddft_tools/ Small supporting analysis utilities
examples/             Two runnable manuscript-oriented examples
tools/                Manuscript training, validation, and plotting drivers
tests/                Focused unit and regression tests
reproducibility/      Versioned checkpoints, results, references, and figures
```

## Testing

Run the focused release checks:

```bash
pytest -q tests/test_neural_xc_public_api.py
pytest -q tests/test_tddft_eigensolvers.py
pytest -q tests/test_molecular_training_api.py
pytest -q tests/test_workflows_config.py
```

Some reference comparisons require PySCF, GPU4PySCF, CUDA, or generated input
data. Large integral/basis sweeps are not part of the compact release gate;
the manuscript dissociation and QM9 artifacts provide the corresponding
end-to-end evidence.

## v1.0.0 Limitations

- Ground-state HFX/PT2 supports `off` and fixed-cache `nograd`; density-updated
  HFX/PT2 `scf` modes are not released.
- Neural local-HF strict response remains fail-fast; released Neural XC
  checkpoints use the validated approximate HFX response path.
- PT2 strict response is a post-hoc CIS(D)-type correction; SCS/SOS scaling is
  not included.
- Full-TDDFT excitation energies are differentiable through the converged
  symplectic Rayleigh quotient. Full-TDDFT X/Y eigenvector gradients are not
  exposed in v1.0.0.
- Geometry optimization and analytical nuclear gradients are outside the
  v1.0.0 release scope.

## License and Upstreams

GradTDDFT is released under the MIT License. It interoperates with JAX, Flax,
Optax, `jax-xc`, PySCF, and GPU4PySCF. Third-party data or source snapshots keep
their original licenses and notices.
