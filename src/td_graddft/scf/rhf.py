from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array

from ..data.basis import CartesianBasis
from ..data.integrals import build_hcore, eri_tensor, overlap_matrix


@dataclass(frozen=True)
class RHFConfig:
    """Configuration for restricted Hartree-Fock SCF iterations."""

    max_cycle: int = 80
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


def _orthogonalizer(s: Array, eps: float) -> Array:
    eigvals, eigvecs = jnp.linalg.eigh(s)
    clipped = jnp.maximum(eigvals, eps)
    return eigvecs @ jnp.diag(clipped ** -0.5) @ eigvecs.T


def _diagonalize_fock(fock: Array, x: Array) -> tuple[Array, Array]:
    f_ortho = x.T @ fock @ x
    mo_energy, coeff_ortho = jnp.linalg.eigh(0.5 * (f_ortho + f_ortho.T))
    mo_coeff = x @ coeff_ortho
    return mo_energy, mo_coeff


def _build_density(mo_coeff: Array, nocc: int) -> Array:
    occ = mo_coeff[:, :nocc]
    return 2.0 * (occ @ occ.T)


def _build_fock(hcore: Array, eri: Array, density: Array) -> Array:
    j_mat = jnp.einsum("pqrs,rs->pq", eri, density)
    k_mat = jnp.einsum("prqs,rs->pq", eri, density)
    return hcore + j_mat - 0.5 * k_mat


def _electronic_energy(density: Array, hcore: Array, fock: Array) -> Array:
    return 0.5 * jnp.einsum("ij,ij->", density, hcore + fock)


def _diis_extrapolate(
    fock: Array,
    error: Array,
    fock_hist: list[Array],
    err_hist: list[Array],
    max_space: int,
) -> Array:
    fock_hist.append(fock)
    err_hist.append(error.reshape(-1))
    if len(fock_hist) > max_space:
        del fock_hist[0]
        del err_hist[0]
    if len(fock_hist) < 2:
        return fock

    m = len(fock_hist)
    b = jnp.empty((m + 1, m + 1), dtype=fock.dtype)
    b = b.at[:, :].set(0.0)
    for i in range(m):
        for j in range(m):
            b = b.at[i, j].set(jnp.dot(err_hist[i], err_hist[j]))
    b = b.at[jnp.arange(m), jnp.arange(m)].add(1e-14)
    b = b.at[:m, m].set(-1.0)
    b = b.at[m, :m].set(-1.0)
    rhs = jnp.zeros((m + 1,), dtype=fock.dtype)
    rhs = rhs.at[m].set(-1.0)

    coeff = jnp.linalg.solve(b, rhs)[:m]
    out = jnp.zeros_like(fock)
    for c, fm in zip(coeff, fock_hist, strict=True):
        out = out + c * fm
    return out


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
    if nelectron % 2 != 0:
        raise ValueError("RHF requires an even number of electrons.")

    s = jnp.asarray(overlap)
    h = jnp.asarray(hcore)
    eri = jnp.asarray(eri)
    enuc = jnp.asarray(nuclear_repulsion)
    nao = int(s.shape[0])
    nocc = nelectron // 2
    if nocc <= 0 or nocc > nao:
        raise ValueError("Invalid occupation count for RHF.")

    x = _orthogonalizer(s, cfg.orthogonalization_eps)
    mo_energy, mo_coeff = _diagonalize_fock(h, x)
    density = _build_density(mo_coeff, nocc)

    energy = jnp.asarray(0.0)
    converged = False
    fock_hist: list[Array] = []
    err_hist: list[Array] = []
    fock = h
    cycles = 0

    for cycle in range(1, cfg.max_cycle + 1):
        fock = _build_fock(h, eri, density)
        if cfg.level_shift != 0.0:
            fock = fock + cfg.level_shift * s

        error = fock @ density @ s - s @ density @ fock
        if cycle >= cfg.diis_start_cycle and cfg.diis_space > 1:
            fock_eff = _diis_extrapolate(
                fock,
                error,
                fock_hist,
                err_hist,
                cfg.diis_space,
            )
        else:
            fock_eff = fock

        mo_energy, mo_coeff = _diagonalize_fock(fock_eff, x)
        density_new = _build_density(mo_coeff, nocc)
        if cfg.damping != 0.0:
            density_new = (1.0 - cfg.damping) * density_new + cfg.damping * density

        elec = _electronic_energy(density_new, h, fock)
        total = elec + enuc
        delta_e = jnp.abs(total - energy)
        rms_d = jnp.sqrt(jnp.mean((density_new - density) ** 2))
        density = density_new
        energy = total
        cycles = cycle

        if float(delta_e) < cfg.conv_tol and float(rms_d) < cfg.conv_tol_density:
            converged = True
            break

    mo_occ = jnp.zeros((nao,), dtype=h.dtype).at[:nocc].set(2.0)
    return RHFResult(
        converged=converged,
        total_energy=float(energy),
        electronic_energy=float(energy - enuc),
        nuclear_repulsion=float(enuc),
        mo_energy=mo_energy,
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        density_matrix=density,
        fock_matrix=fock,
        overlap_matrix=s,
        hcore_matrix=h,
        cycles=cycles,
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
