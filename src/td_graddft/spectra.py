from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array

from .tddft import (
    TDAResult,
    TDDFTResult,
    UnrestrictedTDAResult,
    UnrestrictedTDDFTResult,
)

HARTREE_TO_EV = 27.211386245988


def _restricted_channel(molecule: Any) -> tuple[Array, Array]:
    mo_coeff = jnp.asarray(molecule.mo_coeff)
    mo_occ = jnp.asarray(molecule.mo_occ)

    if mo_coeff.ndim == 2:
        return mo_coeff, mo_occ
    if mo_coeff.ndim != 3 or mo_coeff.shape[0] not in (1, 2):
        raise NotImplementedError("Only restricted closed-shell references are supported.")
    return mo_coeff[0], mo_occ[0]


def occupied_virtual_orbitals(
    molecule: Any,
    occupation_tolerance: float = 1e-8,
) -> tuple[Array, Array]:
    """Extract occupied and virtual MO blocks for a restricted reference."""

    mo_coeff, mo_occ = _restricted_channel(molecule)
    nocc = getattr(molecule, "nocc", None)
    if nocc is not None:
        nocc_int = int(nocc)
        return mo_coeff[:, :nocc_int], mo_coeff[:, nocc_int:]
    occidx = jnp.where(mo_occ > occupation_tolerance)[0]
    viridx = jnp.where(mo_occ <= occupation_tolerance)[0]
    return mo_coeff[:, occidx], mo_coeff[:, viridx]


def occupied_virtual_orbitals_unrestricted(
    molecule: Any,
    occupation_tolerance: float = 1e-8,
) -> tuple[Array, Array, Array, Array]:
    """Extract occupied/virtual MO blocks for an unrestricted reference."""

    mo_coeff = jnp.asarray(molecule.mo_coeff)
    mo_occ = jnp.asarray(molecule.mo_occ)
    if mo_coeff.ndim != 3 or mo_coeff.shape[0] != 2:
        raise NotImplementedError("Expected unrestricted orbitals with shape (2, nao, nmo).")

    nocc_a = getattr(molecule, "nocc_alpha", None)
    nocc_b = getattr(molecule, "nocc_beta", None)
    if nocc_a is not None and nocc_b is not None:
        nocc_a_int = int(nocc_a)
        nocc_b_int = int(nocc_b)
        return (
            mo_coeff[0][:, :nocc_a_int],
            mo_coeff[0][:, nocc_a_int:],
            mo_coeff[1][:, :nocc_b_int],
            mo_coeff[1][:, nocc_b_int:],
        )

    occ_a = jnp.where(mo_occ[0] > occupation_tolerance)[0]
    vir_a = jnp.where(mo_occ[0] <= occupation_tolerance)[0]
    occ_b = jnp.where(mo_occ[1] > occupation_tolerance)[0]
    vir_b = jnp.where(mo_occ[1] <= occupation_tolerance)[0]
    return (
        mo_coeff[0][:, occ_a],
        mo_coeff[0][:, vir_a],
        mo_coeff[1][:, occ_b],
        mo_coeff[1][:, vir_b],
    )


def transition_dipoles(
    molecule: Any,
    result: TDAResult | TDDFTResult | UnrestrictedTDAResult | UnrestrictedTDDFTResult,
    *,
    occupation_tolerance: float = 1e-8,
) -> Array:
    """Compute length-gauge transition dipoles."""

    if getattr(molecule, "dipole_integrals", None) is None:
        raise AttributeError("Molecule-like object must define dipole_integrals.")

    dipole_ao = jnp.asarray(molecule.dipole_integrals)

    if isinstance(result, (UnrestrictedTDAResult, UnrestrictedTDDFTResult)):
        orbo_a, orbv_a, orbo_b, orbv_b = occupied_virtual_orbitals_unrestricted(
            molecule,
            occupation_tolerance,
        )
        dipole_mo_a = jnp.einsum("xpq,pi,qa->xia", dipole_ao, orbo_a.conj(), orbv_a)
        dipole_mo_b = jnp.einsum("xpq,pi,qa->xia", dipole_ao, orbo_b.conj(), orbv_b)
        if isinstance(result, UnrestrictedTDDFTResult):
            amp_a = result.x_amplitudes_alpha + result.y_amplitudes_alpha
            amp_b = result.x_amplitudes_beta + result.y_amplitudes_beta
        else:
            amp_a = result.amplitudes_alpha
            amp_b = result.amplitudes_beta
        mu_a = jnp.einsum("xia,sia->sx", dipole_mo_a, amp_a)
        mu_b = jnp.einsum("xia,sia->sx", dipole_mo_b, amp_b)
        return mu_a + mu_b

    orbo, orbv = occupied_virtual_orbitals(molecule, occupation_tolerance)
    dipole_mo = jnp.einsum("xpq,pi,qa->xia", dipole_ao, orbo.conj(), orbv)
    if isinstance(result, TDDFTResult):
        amplitudes = result.x_amplitudes + result.y_amplitudes
    else:
        amplitudes = result.amplitudes
    return 2.0 * jnp.einsum("xia,sia->sx", dipole_mo, amplitudes)


def oscillator_strengths(
    molecule: Any,
    result: TDAResult | TDDFTResult | UnrestrictedTDAResult | UnrestrictedTDDFTResult,
    *,
    occupation_tolerance: float = 1e-8,
) -> Array:
    """Length-gauge oscillator strengths for the restricted solver."""

    dipoles = transition_dipoles(
        molecule,
        result,
        occupation_tolerance=occupation_tolerance,
    )
    energies = result.excitation_energies
    return (2.0 / 3.0) * energies * jnp.einsum("sx,sx->s", dipoles, dipoles)


def lorentzian_spectrum(
    energies: Array,
    strengths: Array,
    grid: Array,
    *,
    eta: float = 0.1,
) -> Array:
    """Broaden stick spectra with Lorentzians on an energy grid."""

    energies = jnp.asarray(energies)
    strengths = jnp.asarray(strengths)
    grid = jnp.asarray(grid)
    diffs = grid[:, None] - energies[None, :]
    broadened = eta / (jnp.pi * (diffs**2 + eta**2))
    return jnp.sum(strengths[None, :] * broadened, axis=1)
