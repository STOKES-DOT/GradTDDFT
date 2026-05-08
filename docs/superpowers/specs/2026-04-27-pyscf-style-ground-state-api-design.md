# PySCF-Style Ground-State API Design

## Goal

Add a small PySCF-style public API for ground-state calculations without adding a new SCF implementation route.

The API should make common calculations look familiar:

```python
from td_graddft import gto, scf

mol = gto.M(
    atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
    basis="6-31g*",
    charge=0,
    spin=0,
)

mf = scf.RKS(mol, xc="pbe")
mf.jk_backend = "full"
mf.integral_backend = "libcint"

e = mf.kernel()
g = mf.nuc_grad_method().kernel()
```

## Public Surface

Expose:

- `td_graddft.gto.M(...)`
- `td_graddft.scf.RKS(mol, xc="pbe")`
- `td_graddft.scf.UKS(mol, xc="pbe")`

`gto.M(...)` stores molecule input and converts to the existing internal molecule specification.

`scf.RKS` and `scf.UKS` are thin facades over the existing reference builders:

- `restricted_reference_from_spec_with_jax_rks`
- `unrestricted_reference_from_spec_with_jax_uks`

## Defaults

Use the current preferred production route:

- `integral_backend = "libcint"`
- `jk_backend = "full"`
- `geometry_grad_policy = "analytic"`
- no density fitting by default

## Method Object Fields

Expose PySCF-like fields:

- `xc`
- `conv_tol`
- `max_cycle`
- `damp`
- `level_shift`

Expose TD-GradDFT-specific fields:

- `integral_backend`
- `jk_backend`
- `geometry_grad_policy`
- `grid_ao_backend`
- `execution_device`

## Methods

Required methods:

- `kernel()` runs SCF and returns total energy.
- `run()` runs SCF and returns `self`.
- `nuc_grad_method().kernel()` returns nuclear gradients.
- `density_fit()` sets `jk_backend = "df"` and returns `self`.
- `direct_scf()` sets `jk_backend = "direct"` and returns `self`.

After `kernel()`, the method object should expose:

- `e_tot`
- `converged`
- `mo_energy`
- `mo_coeff`
- `mo_occ`
- `reference`

## Scope

This design intentionally does not pursue full PySCF compatibility. It only adopts the familiar object style and names where they make the ground-state workflow clearer.

TDDFT workflow APIs remain separate for now.

