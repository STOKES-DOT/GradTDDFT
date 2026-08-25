# Unrestricted TDDFT/TDA Semilocal Response Design

## Objective

Complete the unrestricted TDA and full TDDFT response path for the LDA, GGA,
and global-hybrid functionals already supported by the UKS ground-state code.
The implementation must remain JAX-native, JIT-compatible, differentiable
with respect to neural-functional parameters, and matrix-free on the grid.

## Scope

The supported first release covers:

- spin-polarized LDA and GGA semilocal response;
- global hybrids whose exact-exchange response is represented by the existing
  Coulomb/exchange action;
- traditional functionals, including PBE, PBE0, and B3LYP;
- bound neural functionals using the same unrestricted grid-response contract;
- unrestricted TDA and full Casida TDDFT;
- empty minority-spin occupied spaces, as in H2+.

The following are out of scope:

- MGGA response, because the current UKS ground-state matrix assembly does not
  provide a complete tau-dependent unrestricted implementation;
- range-separated exchange with independent short- and long-range response
  contractions;
- strict local-HF second response beyond the existing approximate global
  exchange treatment;
- changes to any restricted RKS/TDA/TDDFT behavior.

## Restricted-Code Freeze

The implementation must not modify the restricted response implementation or
its numerical behavior. In particular, the following files and code paths are
frozen:

- `src/td_graddft/tddft/response.py`;
- restricted Casida/TDA solver code;
- RKS SCF code;
- restricted response tensor and HVP contracts;
- existing restricted tests, except for running them as regression checks.

Unrestricted code may import existing restricted transition-factor helpers as
read-only utilities. It must not move, rename, or change them.

## Current Failure

`SemilocalResponseFunctional` exposes only a restricted
`grid_response_tensor`. The unrestricted operator requires
`spin_grid_kernel` or `spin_local_kernel`, so an explicit B3LYP functional is
rejected. Omitting the functional allows the solver to run but drops the
semilocal XC response and retains only

\[
\delta J - \alpha\,\delta K.
\]

For H2+ at 1.06 Angstrom with B3LYP/def2-SVP, this omission shifts the first
four TDA roots by approximately 1.2 to 3.1 eV relative to PySCF, even when both
solvers use identical PySCF orbitals.

The existing neural unrestricted response is also incomplete: it extracts
only the density-density 2 by 2 block of a point Hessian and therefore omits
the density-gradient and gradient-gradient terms required by GGA.

## Mathematical Contract

For each spin channel, define the LDA/GGA response variables

\[
q_\sigma =
(\rho_\sigma,
  \partial_x\rho_\sigma,
  \partial_y\rho_\sigma,
  \partial_z\rho_\sigma).
\]

For LDA only the density component is active. For GGA, concatenate both spin
channels into

\[
q = (q_\alpha, q_\beta) \in \mathbb{R}^{8}.
\]

The semilocal response is the Hessian-vector product

\[
\delta v_{xc} =
\frac{\partial^2 e_{xc}}{\partial q^2}\,\delta q.
\]

The implementation computes this action with JAX JVP/HVP operations. It does
not materialize a `(ngrids, 8, 8)` tensor. The existing one-sided
minority-spin regularization remains part of the traditional point-energy
evaluation.

The complete response operator is

\[
\delta F_\sigma =
\delta J[\delta P_\alpha + \delta P_\beta]
- \alpha_{HF}\,\delta K[\delta P_\sigma]
+ \delta V_{xc,\sigma}.
\]

TDA and full TDDFT use this same action. They differ only in how the action is
embedded into their A and B operator products.

## Architecture

### New unrestricted semilocal module

Add `src/td_graddft/tddft/_unrestricted_semilocal_response.py` with four
responsibilities:

1. Resolve traditional unrestricted LDA/GGA point energies.
2. Evaluate spin-grid Hessian-vector products.
3. Project alpha and beta occupied-virtual amplitudes to spin-grid transition
   variables.
4. Project weighted spin-grid response variables back to occupied-virtual
   amplitudes.

The module exposes an `UnrestrictedSemilocalResponseFunctional` for traditional
XC strings and a builder that converts a functional HVP callback into the
operator action expected by `unrestricted.py`.

### Unrestricted solver assembly

`src/td_graddft/tddft/unrestricted.py` remains responsible for:

- occupied/virtual partitioning;
- Hartree and exact-exchange response;
- assembling TDA and TDDFT operator products;
- Davidson and Casida solver dispatch.

It delegates semilocal projection and HVP evaluation to the new module. The
old scalar `f_aa/f_ab/f_bb` path is retained only for existing LDA-compatible
custom functionals; GGA must use the full HVP path and must never silently fall
back to a scalar kernel.

### Public TD dispatch

`src/td_graddft/tdscf/api.py` selects
`UnrestrictedSemilocalResponseFunctional` only when the source molecule is
unrestricted and the XC source is a string. Restricted dispatch remains
unchanged.

Direct use of `UnrestrictedTDA` or `UnrestrictedCasidaTDDFT` with an XC string
must resolve the same unrestricted functional. Passing `None` continues to
mean Hartree/exchange-only response and is not labeled as DFT response.

### Neural functional binding

`BoundNeuralXCFunctional` gains a `spin_grid_response_hvp` callback. The
unrestricted neural binding computes a full LDA/GGA HVP from
`_total_point_local_energy_from_unrestricted_variables` and exposes it through
that callback.

The existing density-only `spin_local_kernel_fn` construction is removed once
all unrestricted callers use the HVP path. Semilocal energy-density modules
gain a spin-polarized evaluation callback so unrestricted neural channels call
the polarized jax-xc implementation. Restricted channel evaluation remains
unchanged.

## Data Flow

For one TDA/TDDFT operator application:

1. Split alpha and beta trial amplitudes.
2. Project each spin's amplitudes to `delta rho` and, for GGA, three gradient
   components on the molecular grid.
3. Call `spin_grid_response_hvp` with both spin tangents together so cross-spin
   derivatives are retained.
4. Multiply response components by quadrature weights.
5. Project the alpha and beta response components back to their corresponding
   occupied-virtual spaces.
6. Add the existing Hartree and exact-exchange response contributions.
7. Feed the resulting action to the existing TDA or full-TDDFT eigensolver.

The ground-state density, AO values, AO derivatives, quadrature weights, and
electron-repulsion integrals are immutable inputs during this process.

## Error Handling

- GGA response without `ao_deriv1` raises a clear error.
- MGGA and unsupported range-separated specifications raise before solver
  iteration.
- A GGA functional exposing only a scalar spin kernel raises instead of
  silently using an LDA approximation.
- Empty alpha or beta occupied-virtual channels produce correctly shaped empty
  arrays and do not trigger invalid contractions.
- Non-finite point derivatives are not silently accepted as correctness. Tests
  inspect raw HVP finiteness before any final numerical sanitization.

## Testing Strategy

Implementation follows red-green-refactor order.

### Contract tests

- An unrestricted B3LYP string resolves to a spin-grid HVP functional.
- The unrestricted GGA operator rejects scalar-kernel fallback.
- LDA compatibility reproduces the existing density-only action.
- GGA gradient tangents change the response, proving gradient blocks are used.
- Empty beta occupied spaces remain JIT-compatible.

### Traditional numerical tests

- H2+ B3LYP/def2-SVP, grid level 2, using identical PySCF orbitals, covers an
  empty beta occupied space.
- OH B3LYP/def2-SVP covers a conventional doublet with alpha and beta
  occupied-virtual channels.
- O2 PBE/def2-SVP covers a triplet and a non-hybrid GGA.
- Compare the first four TDA and full-TDDFT roots and oscillator strengths
  against PySCF for H2+ and the lowest stable matched roots for OH and O2.
- Compare raw operator columns where root matching alone is insufficient.
- Compare oscillator strengths root by root for isolated roots. For degenerate
  or numerically near-degenerate manifolds, compare the sum of oscillator
  strengths over the matched manifold because eigenvectors may rotate within
  that subspace.

### Neural tests

- The unrestricted neural HVP is finite at an empty beta density.
- Parameter gradients through unrestricted TDA remain finite and nonzero.
- The HVP is JIT-compatible and does not materialize a grid Hessian.

### Regression tests

- Run all unrestricted SCF, TDA, TDDFT, and neural-XC tests.
- Run the restricted TDDFT/PySCF comparison tests without modifying their
  source.
- Verify `git diff` contains no changes to frozen restricted files.

## Acceptance Criteria

The change is complete when:

1. Explicit B3LYP and PBE unrestricted TDA/full-TDDFT no longer raise for a
   missing spin kernel.
2. Traditional unrestricted excitation energies use the same tolerance as the
   existing restricted B3LYP comparison: `atol=8e-4 Hartree` and `rtol=2e-3`.
3. Unrestricted oscillator strengths use the same tolerance as the existing
   restricted B3LYP comparison: `atol=2e-3` and `rtol=2e-2`, with summed
   strengths used for matched degenerate manifolds.
4. H2+, OH, and O2 cover empty-beta, doublet, and triplet response behavior,
   respectively; the applicable TDA and full-TDDFT comparisons pass on
   identical PySCF orbitals.
5. JAX UKS orbitals give consistent excitation energies within the known SCF
   orbital tolerance.
6. Neural unrestricted response is fully differentiable and uses all GGA
   response variables.
7. No restricted source file is modified.
8. No dense `(ngrids, 8, 8)` Hessian is retained or cached.
