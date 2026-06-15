from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from ._pytree import pytree_dataclass


@pytree_dataclass()
@dataclass(frozen=True)
class QuadratureGrid:
    """Minimal quadrature grid container used by the TDDFT modules."""

    weights: jnp.ndarray
    coords: jnp.ndarray | None = None


@pytree_dataclass(
    static_fields=(
        "nocc",
        "scf_converged",
        "runtime_scf_backend",
        "runtime_scf_options",
        "hfx_nu_api",
    )
)
@dataclass(frozen=True)
class RestrictedMolecule:
    """Minimal restricted molecule container used across TD-GradDFT."""

    ao: jnp.ndarray
    grid: QuadratureGrid
    dipole_integrals: jnp.ndarray
    rep_tensor: jnp.ndarray
    mo_coeff: jnp.ndarray
    mo_occ: jnp.ndarray
    mo_energy: jnp.ndarray
    rdm1: jnp.ndarray
    h1e: jnp.ndarray
    nuclear_repulsion: float
    atom_coords: jnp.ndarray | None = None
    atom_charges: jnp.ndarray | None = None
    overlap_matrix: jnp.ndarray | None = None
    ao_deriv1: jnp.ndarray | None = None
    ao_laplacian: jnp.ndarray | None = None
    mf_energy: float | None = None
    exact_exchange_fraction: float = 0.0
    nocc: int | None = None
    hfx_omega_values: jnp.ndarray | None = None
    hfx_local: jnp.ndarray | None = None
    hfx_nu: jnp.ndarray | None = None
    hfx_nu_api: Any | None = None
    pt2_local: jnp.ndarray | None = None
    neural_xc_grid_payload: Any | None = None
    scf_initial_density: jnp.ndarray | None = None
    df_factors: jnp.ndarray | None = None
    eri_pair_matrix: jnp.ndarray | None = None
    eri_ovov: jnp.ndarray | None = None
    eri_ovvo: jnp.ndarray | None = None
    eri_oovv: jnp.ndarray | None = None
    scf_converged: bool | None = None
    runtime_scf_backend: str | None = None
    runtime_scf_options: Any | None = None

    def density(self) -> jnp.ndarray:
        return jnp.einsum("spq,rp,rq->rs", self.rdm1, self.ao, self.ao)


@pytree_dataclass(
    static_fields=(
        "nocc_alpha",
        "nocc_beta",
        "runtime_scf_backend",
        "runtime_scf_options",
        "hfx_nu_api",
    )
)
@dataclass(frozen=True)
class UnrestrictedMolecule:
    """Minimal unrestricted molecule container used across TD-GradDFT."""

    ao: jnp.ndarray
    grid: QuadratureGrid
    dipole_integrals: jnp.ndarray
    rep_tensor: jnp.ndarray
    mo_coeff: jnp.ndarray
    mo_occ: jnp.ndarray
    mo_energy: jnp.ndarray
    rdm1: jnp.ndarray
    h1e: jnp.ndarray
    nuclear_repulsion: float
    atom_coords: jnp.ndarray | None = None
    atom_charges: jnp.ndarray | None = None
    overlap_matrix: jnp.ndarray | None = None
    ao_deriv1: jnp.ndarray | None = None
    ao_laplacian: jnp.ndarray | None = None
    mf_energy: float | None = None
    exact_exchange_fraction: float = 0.0
    nocc_alpha: int | None = None
    nocc_beta: int | None = None
    hfx_omega_values: jnp.ndarray | None = None
    hfx_local: jnp.ndarray | None = None
    hfx_nu: jnp.ndarray | None = None
    hfx_nu_api: Any | None = None
    pt2_local: jnp.ndarray | None = None
    scf_initial_density: jnp.ndarray | None = None
    runtime_scf_backend: str | None = None
    runtime_scf_options: Any | None = None

    def density(self) -> jnp.ndarray:
        return jnp.einsum("spq,rp,rq->r", self.rdm1, self.ao, self.ao)


__all__ = [
    "QuadratureGrid",
    "RestrictedMolecule",
    "UnrestrictedMolecule",
]
