from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from jaxtyping import Array, PyTree

from .nn_rsh.schema import PySCFRSHSpec, ResolvedRSHParameters, SCFXCContributions


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
class BoundRSHFunctionalProtocol(Protocol):
    """Molecule-bound RSH object protocol used by SCF/training integration."""

    resolved_params: ResolvedRSHParameters
    exact_exchange_fraction: Array | float

    def local_potential(self, density: Array) -> Array:
        """Return local v_xc[n](r) on the given grid density."""

    def local_kernel(self, density: Array) -> Array:
        """Return local f_xc[n](r) on the given grid density."""

    def scf_contributions(self, molecule: Any) -> SCFXCContributions:
        """Return all SCF-facing XC contributions for the bound molecule."""

    def to_pyscf_spec(self) -> PySCFRSHSpec:
        """Return a PySCF-installable XC spec for the resolved parameters."""


@runtime_checkable
class TrainableRSHFunctionalProtocol(Protocol):
    """Trainable RSH functional protocol."""

    def init(self, rng: Array, molecule: Any) -> PyTree:
        """Initialize trainable parameters from a molecule sample."""

    def resolve_parameters(self, params: PyTree) -> ResolvedRSHParameters:
        """Convert raw trainable parameters to physical RSH parameters."""

    def bind_to_molecule(self, params: PyTree, molecule: Any) -> BoundRSHFunctionalProtocol:
        """Bind the trainable functional to a molecule."""

    def bind_to_molecule_for_scf(
        self,
        params: PyTree,
        molecule: Any,
    ) -> BoundRSHFunctionalProtocol:
        """Bind the functional for SCF usage."""

    def energy_from_molecule(self, params: PyTree, molecule: Any) -> Array:
        """Return the XC energy for the given molecule."""


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
