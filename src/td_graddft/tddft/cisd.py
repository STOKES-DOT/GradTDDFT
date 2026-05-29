from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array

from ..data.integrals.jax.packed_eri import _metadata_arrays, _mo_pair_products
from ._utils import _restricted_channel
from .types import TDDFTResult, TDAResult


def _restricted_nocc(molecule: Any, mo_occ: Array, occupation_tolerance: float) -> int:
    nocc = getattr(molecule, "nocc", None)
    if nocc is not None:
        return int(nocc)
    return int(jnp.count_nonzero(mo_occ > occupation_tolerance))


def _restricted_mo_eri_tensor(molecule: Any, mo_coeff: Array) -> Array:
    explicit_mo_eri = getattr(molecule, "mo_eri", None)
    if explicit_mo_eri is not None:
        return jnp.asarray(explicit_mo_eri)

    rep_tensor = getattr(molecule, "rep_tensor", None)
    if rep_tensor is not None and int(jnp.asarray(rep_tensor).size) > 0:
        eri_ao = jnp.asarray(rep_tensor)
        return jnp.einsum(
            "pqrs,pP,qQ,rR,sS->PQRS",
            eri_ao,
            mo_coeff,
            mo_coeff,
            mo_coeff,
            mo_coeff,
            precision=Precision.HIGHEST,
        )

    eri_pair_matrix = getattr(molecule, "eri_pair_matrix", None)
    if eri_pair_matrix is not None and int(jnp.asarray(eri_pair_matrix).size) > 0:
        pair = jnp.asarray(eri_pair_matrix)
        rows, cols, _, _ = _metadata_arrays(int(mo_coeff.shape[0]), mo_coeff.dtype)
        mo_pairs = _mo_pair_products(mo_coeff, mo_coeff, rows, cols)
        return jnp.einsum(
            "pqP,PQ,rsQ->pqrs",
            mo_pairs,
            pair,
            mo_pairs,
            precision=Precision.HIGHEST,
        )

    df_factors = getattr(molecule, "df_factors", None)
    if df_factors is not None and int(jnp.asarray(df_factors).size) > 0:
        factors = jnp.asarray(df_factors)
        b_mo = jnp.einsum(
            "Lpq,pP,qR->LPR",
            factors,
            mo_coeff,
            mo_coeff,
            precision=Precision.HIGHEST,
        )
        return jnp.einsum("Lpq,Lrs->pqrs", b_mo, b_mo, precision=Precision.HIGHEST)

    raise ValueError(
        "CIS(D) correction requires mo_eri, rep_tensor, eri_pair_matrix, or df_factors."
    )


def _spin_orbital_labels(spatial: Array) -> tuple[Array, Array]:
    spatial = jnp.asarray(spatial, dtype=jnp.int32)
    return (
        jnp.repeat(spatial, 2),
        jnp.tile(jnp.asarray([0, 1], dtype=jnp.int32), int(spatial.shape[0])),
    )


def _chemist_block(
    mo_eri: Array,
    p_spatial: Array,
    p_spin: Array,
    q_spatial: Array,
    q_spin: Array,
    r_spatial: Array,
    r_spin: Array,
    s_spatial: Array,
    s_spin: Array,
) -> Array:
    values = mo_eri[
        p_spatial[:, None, None, None],
        q_spatial[None, :, None, None],
        r_spatial[None, None, :, None],
        s_spatial[None, None, None, :],
    ]
    spin_mask = (
        (p_spin[:, None, None, None] == q_spin[None, :, None, None])
        & (r_spin[None, None, :, None] == s_spin[None, None, None, :])
    )
    return values * spin_mask.astype(values.dtype)


def _antisymmetrized_block(
    mo_eri: Array,
    p_spatial: Array,
    p_spin: Array,
    q_spatial: Array,
    q_spin: Array,
    r_spatial: Array,
    r_spin: Array,
    s_spatial: Array,
    s_spin: Array,
) -> Array:
    # <pq||rs> = (pr|qs) - (ps|qr), with (..|..) in chemists' notation.
    direct = _chemist_block(
        mo_eri,
        p_spatial,
        p_spin,
        r_spatial,
        r_spin,
        q_spatial,
        q_spin,
        s_spatial,
        s_spin,
    )
    exchange = _chemist_block(
        mo_eri,
        p_spatial,
        p_spin,
        s_spatial,
        s_spin,
        q_spatial,
        q_spin,
        r_spatial,
        r_spin,
    )
    return jnp.transpose(direct, (0, 2, 1, 3)) - jnp.transpose(
        exchange,
        (0, 2, 3, 1),
    )


def _result_singles_amplitudes(result: TDAResult | TDDFTResult) -> Array:
    if isinstance(result, TDDFTResult):
        return jnp.asarray(result.x_amplitudes + result.y_amplitudes)
    return jnp.asarray(result.amplitudes)


def _spatial_to_spin_singlet_amplitudes(amplitudes: Array) -> Array:
    nstates, nocc, nvir = amplitudes.shape
    spin_amplitudes = jnp.zeros((nstates, 2 * nocc, 2 * nvir), dtype=amplitudes.dtype)
    alpha = 2 * jnp.arange(nocc)
    beta = alpha + 1
    vir_alpha = 2 * jnp.arange(nvir)
    vir_beta = vir_alpha + 1
    spin_amplitudes = spin_amplitudes.at[:, alpha[:, None], vir_alpha[None, :]].set(
        amplitudes
    )
    spin_amplitudes = spin_amplitudes.at[:, beta[:, None], vir_beta[None, :]].set(
        amplitudes
    )
    return spin_amplitudes


def restricted_cisd_second_order_correction(
    molecule: Any,
    result: TDAResult | TDDFTResult,
    *,
    ac: Array | float = 1.0,
    occupation_tolerance: float = 1e-8,
) -> Array:
    """Return the unscaled/SCS-free CIS(D) doubles correction for restricted roots.

    This follows the Head-Gordon CIS(D) form used by ORCA double hybrids:
    solve the singles-only state first, then add a root-specific second-order
    doubles correction. No SCS/SOS spin-component factors or damping are used.
    The caller supplies the double-hybrid PT2 coefficient ``ac``.
    """

    mo_coeff, mo_occ, mo_energy = _restricted_channel(molecule)
    nocc = _restricted_nocc(molecule, mo_occ, occupation_tolerance)
    nmo = int(mo_coeff.shape[1])
    nvir = nmo - nocc
    if nocc <= 0 or nvir <= 0:
        raise ValueError("CIS(D) correction requires at least one occupied and one virtual orbital.")

    mo_eri = _restricted_mo_eri_tensor(molecule, mo_coeff)
    occ_spatial, occ_spin = _spin_orbital_labels(jnp.arange(nocc, dtype=jnp.int32))
    vir_spatial, vir_spin = _spin_orbital_labels(
        jnp.arange(nocc, nmo, dtype=jnp.int32)
    )
    occ_eps = jnp.repeat(jnp.asarray(mo_energy[:nocc]), 2)
    vir_eps = jnp.repeat(jnp.asarray(mo_energy[nocc:]), 2)

    ij_ab = _antisymmetrized_block(
        mo_eri,
        occ_spatial,
        occ_spin,
        occ_spatial,
        occ_spin,
        vir_spatial,
        vir_spin,
        vir_spatial,
        vir_spin,
    )

    denom_mp2 = (
        vir_eps[None, None, :, None]
        + vir_eps[None, None, None, :]
        - occ_eps[:, None, None, None]
        - occ_eps[None, :, None, None]
    )
    mp2_amplitudes = -ij_ab / denom_mp2
    ab_cj = _antisymmetrized_block(
        mo_eri,
        vir_spatial,
        vir_spin,
        vir_spatial,
        vir_spin,
        vir_spatial,
        vir_spin,
        occ_spatial,
        occ_spin,
    )
    ka_ij = _antisymmetrized_block(
        mo_eri,
        occ_spatial,
        occ_spin,
        vir_spatial,
        vir_spin,
        occ_spatial,
        occ_spin,
        occ_spatial,
        occ_spin,
    )
    jc_kb = _chemist_block(
        mo_eri,
        occ_spatial,
        occ_spin,
        vir_spatial,
        vir_spin,
        occ_spatial,
        occ_spin,
        vir_spatial,
        vir_spin,
    )
    jk_bc = ij_ab
    r_ab = -jnp.einsum(
        "jckb,jkca->ab",
        jc_kb,
        mp2_amplitudes,
        precision=Precision.HIGHEST,
    )
    r_ij = -jnp.einsum(
        "jakb,ikab->ij",
        jc_kb,
        mp2_amplitudes,
        precision=Precision.HIGHEST,
    )

    spatial_amplitudes = _result_singles_amplitudes(result)
    spin_amplitudes = _spatial_to_spin_singlet_amplitudes(spatial_amplitudes)
    omegas = jnp.asarray(result.excitation_energies, dtype=jnp.asarray(ij_ab).dtype)

    def root_correction(b: Array, omega: Array) -> Array:
        u = (
            jnp.einsum("abcj,ic->ijab", ab_cj, b, precision=Precision.HIGHEST)
            - jnp.einsum("abci,jc->ijab", ab_cj, b, precision=Precision.HIGHEST)
            + jnp.einsum("kaij,kb->ijab", ka_ij, b, precision=Precision.HIGHEST)
            - jnp.einsum("kbij,ka->ijab", ka_ij, b, precision=Precision.HIGHEST)
        )
        denom_excited = denom_mp2 - omega
        direct = -0.25 * jnp.sum(u * u / denom_excited)

        w_ia = jnp.einsum(
            "jkbc,ikac,jb->ia",
            jk_bc,
            mp2_amplitudes,
            b,
            precision=Precision.HIGHEST,
        )
        indirect = (
            jnp.einsum("ia,ib,ab->", b, b, r_ab, precision=Precision.HIGHEST)
            + jnp.einsum("ic,jc,ij->", b, b, r_ij, precision=Precision.HIGHEST)
            + jnp.einsum("ia,ia->", b, w_ia, precision=Precision.HIGHEST)
        )
        return jnp.real(direct + indirect)

    correction = jax.vmap(root_correction)(spin_amplitudes, omegas)
    return jnp.asarray(ac, dtype=correction.dtype) * correction
