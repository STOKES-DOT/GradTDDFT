from __future__ import annotations

from typing import Any

import numpy as np
import jax.numpy as jnp

from td_graddft.scf.features import (
    _charge_center,
    _restricted_response_eri_slices_from_mo_tensor,
)
from td_graddft.neural_xc.inputs import _local_hfx_features_from_dm
from td_graddft.scf.molecules import QuadratureGrid, RestrictedMolecule


def _hybrid_fraction_from_mf(mf: Any) -> float:
    numint = getattr(mf, "_numint", None)
    xc = getattr(mf, "xc", None)
    if numint is None or xc is None:
        return 0.0
    rsh_hyb = getattr(numint, "rsh_and_hybrid_coeff", None)
    if rsh_hyb is None:
        return 0.0
    try:
        _, _, hyb = rsh_hyb(xc, int(getattr(mf.mol, "spin", 0)))
    except Exception:
        return 0.0
    return float(hyb)


def restricted_reference_from_pyscf(
    mf: Any,
    *,
    compute_local_hfx_features: bool = False,
    compute_local_hfx_aux: bool = False,
    compute_local_pt2_features: bool = False,
    hfx_omega_values: tuple[float, ...] = (0.0, 0.4),
    hfx_chunk_size: int = 512,
) -> RestrictedMolecule:
    """Convert a restricted PySCF SCF/DFT object to a TD-GradDFT-ready reference."""

    try:
        from pyscf.dft import numint
    except ModuleNotFoundError as exc:
        raise ImportError("PySCF is required for restricted_reference_from_pyscf.") from exc

    if getattr(mf.mol, "spin", 0) != 0:
        raise NotImplementedError("Only restricted closed-shell PySCF references are supported.")
    if getattr(mf, "mo_coeff", None) is None:
        raise ValueError("PySCF mean-field object is not converged; run mf.kernel() first.")

    if getattr(mf.grids, "coords", None) is None:
        mf.grids.build()

    ao_np = np.asarray(numint.eval_ao(mf.mol, mf.grids.coords, deriv=0))
    ao = jnp.asarray(ao_np)
    ao_deriv1 = jnp.asarray(numint.eval_ao(mf.mol, mf.grids.coords, deriv=1))
    weights = jnp.asarray(mf.grids.weights)
    dm_total = jnp.asarray(mf.make_rdm1())
    half_dm = dm_total / 2.0
    mo_coeff = jnp.asarray(mf.mo_coeff)
    mo_occ = jnp.asarray(mf.mo_occ) / 2.0
    mo_energy = jnp.asarray(mf.mo_energy)
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 1e-8))
    rep_tensor = jnp.asarray(mf.mol.intor("int2e"))
    eri_ovov, eri_ovvo, eri_oovv = _restricted_response_eri_slices_from_mo_tensor(
        rep_tensor,
        mo_coeff,
        nocc,
    )
    with mf.mol.with_common_orig(_charge_center(mf.mol)):
        dipole_integrals = jnp.asarray(mf.mol.intor_symmetric("int1e_r", comp=3))

    hfx_local = None
    hfx_nu = None
    pt2_local = None
    if compute_local_hfx_features:
        dm_half_np = np.asarray(half_dm)
        hfx_result = _local_hfx_features_from_dm(
            mf.mol,
            ao_np,
            (dm_half_np, dm_half_np),
            np.asarray(mf.grids.coords),
            omega_values=tuple(float(omega) for omega in hfx_omega_values),
            chunk_size=hfx_chunk_size,
            return_nu=bool(compute_local_hfx_aux),
        )
        if compute_local_hfx_aux:
            hfx_local_np, hfx_nu_np = hfx_result
            hfx_nu = jnp.asarray(hfx_nu_np)
        else:
            hfx_local_np = hfx_result
        hfx_local = jnp.asarray(hfx_local_np)
    if compute_local_pt2_features:
        from td_graddft.neural_xc.inputs import _local_pt2_feature_from_restricted_orbitals

        pt2_local = _local_pt2_feature_from_restricted_orbitals(
            ao,
            mo_coeff,
            mo_occ,
            mo_energy,
            rep_tensor=rep_tensor,
            eri_ovov=eri_ovov,
            nocc=nocc,
        )

    return RestrictedMolecule(
        ao=ao,
        grid=QuadratureGrid(weights=weights, coords=jnp.asarray(mf.grids.coords)),
        dipole_integrals=dipole_integrals,
        rep_tensor=rep_tensor,
        mo_coeff=jnp.stack([mo_coeff, mo_coeff], axis=0),
        mo_occ=jnp.stack([mo_occ, mo_occ], axis=0),
        mo_energy=jnp.stack([mo_energy, mo_energy], axis=0),
        rdm1=jnp.stack([half_dm, half_dm], axis=0),
        h1e=jnp.asarray(mf.get_hcore()),
        nuclear_repulsion=float(mf.mol.energy_nuc()),
        atom_coords=jnp.asarray(mf.mol.atom_coords()),
        atom_charges=jnp.asarray(mf.mol.atom_charges()),
        overlap_matrix=jnp.asarray(mf.get_ovlp()),
        ao_deriv1=ao_deriv1,
        mf_energy=float(getattr(mf, "e_tot", jnp.nan)),
        exact_exchange_fraction=_hybrid_fraction_from_mf(mf),
        nocc=nocc,
        hfx_omega_values=(
            tuple(float(omega) for omega in hfx_omega_values)
            if compute_local_hfx_features
            else None
        ),
        hfx_local=hfx_local,
        hfx_nu=hfx_nu,
        pt2_local=pt2_local,
        eri_ovov=eri_ovov,
        eri_ovvo=eri_ovvo,
        eri_oovv=eri_oovv,
    )


__all__ = ["restricted_reference_from_pyscf"]
