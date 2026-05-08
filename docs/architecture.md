# Architecture Notes

## Project Principle

`TD-GradDFT` should be organized as:

```text
GradDFT ground-state core
  + differentiable excited-state extension
```

That means:

- the default neural XC and SCF abstractions should stay GradDFT-first
- DFT correctness remains the base contract
- TDDFT/TDA/Casida sits on top of that base as a differentiable extension
- excited-state losses are allowed, but they should not redefine the ground-state core API

## Upstream Roles

### `GradDFT`

`GradDFT` already provides the pieces that are hardest to rebuild correctly:

- `Molecule` and `Solid` containers
- differentiable XC functional abstractions
- SCF and differentiable SCF loops
- conversion from PySCF mean-field objects into JAX-friendly structures

That makes it the natural ground-state substrate for this project.

### `jax_xc`

`jax_xc` is valuable for a different reason: it exposes a large library of JAX-native exchange-correlation functionals translated from `libxc`. It is a good source for:

- baseline adiabatic LDA/GGA/mGGA functionals
- derivative-based XC potentials
- future XC kernel experiments

## Proposed Data Flow

```text
ground-state reference build
  -> GradDFT-style neural XC / differentiable SCF core
  -> converged KS orbitals, energies, density
  -> TD Hamiltonian / linear-response builder
  -> TDA or Casida solver
  -> excited-state observables and spectra
  -> optional excited-state supervision back to XC parameters
```

## Module Boundaries

### `td_graddft.upstreams`

Small compatibility layer around optional dependencies. This is where environment-sensitive code should live:

- importing `grad_dft`
- importing `jax_xc`
- converting upstream objects into local dataclasses

### `td_graddft.xc`

Owns adiabatic XC wrappers that are easy to differentiate with JAX. The first implementation only bridges local-density style functionals from `jax_xc`, which keeps the API honest while leaving room for a richer grid-feature pipeline later.

### `td_graddft.neural_xc`

Owns the pure-JAX neural functional layer. The structure is borrowed from the
core GradDFT idea:

- generate local coefficient inputs from the density
- predict coefficients with a Flax pointwise network
- contract them with a small local XC basis

This keeps the learning stack in JAX/Flax/Optax and avoids TensorFlow entirely.
The default abstraction should stay close to the GradDFT coefficient-basis
interface, with DM21-like variants treated as concrete model implementations
rather than the project-wide default.

### `td_graddft.realtime`

Owns propagation utilities. The current version is intentionally generic:

- commutator utilities
- Liouville-von Neumann right-hand side
- unitary matrix-exponential stepper
- trajectory generation

This lets us plug in a chemistry-aware Hamiltonian builder later without changing the propagation API.

## Scope Of The First Prototype

Included now:

- real-time density-matrix propagation core
- optional upstream adapters
- local adiabatic XC helper objects
- pure-JAX neural XC functionals inspired by GradDFT
- fixed-density and self-consistent ground-state training utilities
- restricted closed-shell Casida/TDA TDDFT matrices and solvers
- differentiable excited-state loss plumbing for excitation energies,
  oscillator strengths, and broadened spectra

Deferred deliberately:

- AO-overlap-aware propagators
- self-consistent ground-state training
- unrestricted TDDFT
- large-system `gen_vind` operator path
- GGA and mGGA feature plumbing from `GradDFT` grids
- spectrum post-processing
