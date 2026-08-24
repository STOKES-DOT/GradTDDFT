from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array

from ..data.basis import CartesianBasis
from ..data.integrals import build_hcore, eri_tensor, overlap_matrix
from .rks import RKSConfig, run_rks_from_integrals


@dataclass(frozen=True)
class RHFConfig:
    """Configuration for restricted Hartree-Fock SCF iterations."""

    max_cycle: int = 800
    conv_tol: float = 1e-10
    conv_tol_density: float = 1e-8
    diis_start_cycle: int = 2
    diis_space: int = 8
    damping: float = 0.0
    level_shift: float = 0.0
    orthogonalization_eps: float = 1e-10


@dataclass(frozen=True)
class RHFResult:
    """Restricted Hartree-Fock result object."""

    converged: bool
    total_energy: float
    electronic_energy: float
    nuclear_repulsion: float
    mo_energy: Array
    mo_coeff: Array
    mo_occ: Array
    density_matrix: Array
    fock_matrix: Array
    overlap_matrix: Array
    hcore_matrix: Array
    cycles: int


def nuclear_repulsion_energy(atom_coords: Array, atom_charges: Array) -> Array:
    """Compute classical nuclear repulsion energy."""

    coords = jnp.asarray(atom_coords)
    charges = jnp.asarray(atom_charges)
    enuc = jnp.asarray(0.0)
    natm = int(coords.shape[0])
    for i in range(natm):
        for j in range(i):
            rij = jnp.linalg.norm(coords[i] - coords[j])
            enuc = enuc + charges[i] * charges[j] / rij
    return enuc


def run_rhf_from_integrals(
    *,
    overlap: Array,
    hcore: Array,
    eri: Array,
    nelectron: int,
    nuclear_repulsion: float | Array,
    config: RHFConfig | None = None,
) -> RHFResult:
    """Run restricted Hartree-Fock from precomputed AO integrals."""

    cfg = RHFConfig() if config is None else config
    s = jnp.asarray(overlap)
    h = jnp.asarray(hcore)
    nao = int(s.shape[0])
    if cfg.diis_start_cycle != 2 or cfg.diis_space != 8:
        raise ValueError(
            "run_rhf_from_integrals uses the shared RKS DIIS schedule "
            "(diis_start_cycle=2, diis_space=8)."
        )
    result = run_rks_from_integrals(
        overlap=s,
        hcore=h,
        eri=jnp.asarray(eri),
        nelectron=nelectron,
        nuclear_repulsion=nuclear_repulsion,
        ao=jnp.zeros((0, nao), dtype=h.dtype),
        ao_deriv1=jnp.zeros((4, 0, nao), dtype=h.dtype),
        grid_weights=jnp.zeros((0,), dtype=h.dtype),
        config=RKSConfig(
            xc_spec="hf",
            max_cycle=cfg.max_cycle,
            conv_tol=cfg.conv_tol,
            conv_tol_density=cfg.conv_tol_density,
            damping=cfg.damping,
            level_shift=cfg.level_shift,
            orthogonalization_eps=cfg.orthogonalization_eps,
        ),
    )
    return RHFResult(
        converged=result.converged,
        total_energy=result.total_energy,
        electronic_energy=result.electronic_energy,
        nuclear_repulsion=result.nuclear_repulsion,
        mo_energy=result.mo_energy,
        mo_coeff=result.mo_coeff,
        mo_occ=result.mo_occ,
        density_matrix=result.density_matrix,
        fock_matrix=result.fock_matrix,
        overlap_matrix=result.overlap_matrix,
        hcore_matrix=result.hcore_matrix,
        cycles=result.cycles,
    )


def run_rhf(
    *,
    basis: CartesianBasis,
    nelectron: int,
    nuclear_repulsion: float | Array | None = None,
    config: RHFConfig | None = None,
) -> RHFResult:
    """Run RHF from a Cartesian basis and pure-JAX integral tensors."""

    s = overlap_matrix(basis)
    h = build_hcore(basis)
    eri = eri_tensor(basis)
    enuc = (
        nuclear_repulsion_energy(basis.atom_coords, basis.atom_charges)
        if nuclear_repulsion is None
        else jnp.asarray(nuclear_repulsion)
    )
    return run_rhf_from_integrals(
        overlap=s,
        hcore=h,
        eri=eri,
        nelectron=nelectron,
        nuclear_repulsion=enuc,
        config=config,
    )
