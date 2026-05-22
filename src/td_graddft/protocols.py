from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from jaxtyping import Array, PyTree

@runtime_checkable
class XCFunctionalProtocol(Protocol):
    """Minimal trainable XC functional protocol used across TD-GradDFT."""

    def energy(self, params: PyTree, density: Array, weights: Array) -> Array:
        """Return integrated XC energy for a density sample."""

    def init(self, rng: Array, sample_density: Array) -> PyTree:
        """Initialize trainable parameters from a sample density."""


@runtime_checkable
class BoundXCFunctionalProtocol(Protocol):
    """Molecule-bound XC object protocol used by TDDFT response builders."""

    exact_exchange_fraction: float

    def local_potential(self, density: Array) -> Array:
        """Return local v_xc[n](r) on the given grid density."""

    def local_kernel(self, density: Array) -> Array:
        """Return local f_xc[n](r) on the given grid density."""


@runtime_checkable
class MoleculeReferenceProtocol(Protocol):
    """Minimal molecule-like container required by training/TDDFT routines."""

    ao: Array
    mo_coeff: Array
    mo_occ: Array
    mo_energy: Array
    rdm1: Array
    rep_tensor: Array
    dipole_integrals: Array
    h1e: Array
    nuclear_repulsion: float
    grid: Any

    def density(self) -> Array:
        """Return density on quadrature grid (or spin-resolved grid density)."""
