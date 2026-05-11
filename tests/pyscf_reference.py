from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import numpy as np
import jax.numpy as jnp

from td_graddft.data.basis import basis_from_pyscf_mol_cart
from td_graddft.data.integrals import (
    build_hcore,
    dipole_matrix,
    eri_tensor,
    overlap_hcore_matrices,
    overlap_matrix,
)
from td_graddft.jax_libxc import parse_xc
from td_graddft.neural_xc.inputs import (
    _local_hfx_features_from_basis_dm,
    _local_hfx_features_from_dm,
    _local_pt2_feature_from_restricted_orbitals,
)
from td_graddft.scf.builders import restricted_reference_from_spec_with_jax_rks
from td_graddft.scf.features import (
    _charge_center,
    _eval_grid_ao,
    _restricted_response_eri_slices_from_mo_tensor,
)
from td_graddft.scf.molecules import (
    GridReference,
    RestrictedMoleculeReference,
    UnrestrictedMoleculeReference,
)
from td_graddft.scf import (
    RHFConfig,
    RKSConfig,
    UKSConfig,
    run_rhf_from_integrals,
    run_rks_from_integrals,
    run_uks_from_integrals,
)
from td_graddft.scf.packed_eri import eri_pair_matrix_to_mo_eri_slices


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


def _uses_cuda_direct_jk(cfg: RKSConfig) -> bool:
    return cfg.jk_backend == "direct" and cfg.direct_jk_engine == "cuda"


def _overlap_hcore_for_jax_rks_reference(basis: Any, cfg: RKSConfig) -> tuple[Any, Any]:
    if _uses_cuda_direct_jk(cfg):
        from td_graddft.scf.cuda_direct_jk import cuda_ffi_available
        from td_graddft.scf.cuda_one_electron import CudaOneElectronBuilder

        if cuda_ffi_available():
            return CudaOneElectronBuilder(basis).build_overlap_hcore()
    return overlap_hcore_matrices(basis)


def _infer_jax_xc_spec_from_pyscf_label(xc_label: str | None) -> str:
    raw = "pbe" if xc_label is None else str(xc_label).strip().lower()
    mapping = {
        "lda": "lda",
        "svwn": "svwn",
        "pbe": "pbe",
        "pbe0": "pbe0",
        "b3lyp": "0.20*hf + 0.08*lda_x + 0.72*gga_x_b88 + 0.19*lda_c_vwn_rpa + 0.81*gga_c_lyp",
    }
    spec = mapping.get(raw, raw)
    parse_xc(spec)
    return spec


def _clone_pyscf_mol_cart(mol: Any) -> Any:
    """Create a cartesian-AO PySCF Mole clone from an existing Mole."""

    try:
        from pyscf import gto
    except ModuleNotFoundError as exc:
        raise ImportError("PySCF is required for cartesian Mole cloning.") from exc

    atom_bohr = [
        (mol.atom_symbol(i), tuple(float(x) for x in mol.atom_coord(i)))
        for i in range(mol.natm)
    ]
    return gto.M(
        atom=atom_bohr,
        unit="Bohr",
        basis=mol.basis,
        ecp=getattr(mol, "ecp", None),
        charge=int(getattr(mol, "charge", 0)),
        spin=int(getattr(mol, "spin", 0)),
        cart=True,
        verbose=0,
    )


def restricted_reference_from_pyscf(
    mf: Any,
    *,
    compute_local_hfx_features: bool = False,
    compute_local_hfx_aux: bool = False,
    compute_local_pt2_features: bool = False,
    hfx_omega_values: tuple[float, ...] = (0.0, 0.4),
    hfx_chunk_size: int = 512,
) -> RestrictedMoleculeReference:
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
        pt2_local = _local_pt2_feature_from_restricted_orbitals(
            ao,
            mo_coeff,
            mo_occ,
            mo_energy,
            rep_tensor=rep_tensor,
            eri_ovov=eri_ovov,
            nocc=nocc,
        )

    return RestrictedMoleculeReference(
        ao=ao,
        grid=GridReference(weights=weights, coords=jnp.asarray(mf.grids.coords)),
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


def unrestricted_reference_from_pyscf(mf: Any) -> UnrestrictedMoleculeReference:
    """Convert an unrestricted PySCF SCF/DFT object to a TD-GradDFT-ready reference."""

    try:
        from pyscf.dft import numint
    except ModuleNotFoundError as exc:
        raise ImportError("PySCF is required for unrestricted_reference_from_pyscf.") from exc

    if getattr(mf, "mo_coeff", None) is None:
        raise ValueError("PySCF mean-field object is not converged; run mf.kernel() first.")
    if getattr(mf.grids, "coords", None) is None:
        mf.grids.build()

    mo_coeff = jnp.asarray(mf.mo_coeff)
    mo_occ = jnp.asarray(mf.mo_occ)
    mo_energy = jnp.asarray(mf.mo_energy)
    nocc_alpha = int(np.count_nonzero(np.asarray(mf.mo_occ)[0] > 1e-8))
    nocc_beta = int(np.count_nonzero(np.asarray(mf.mo_occ)[1] > 1e-8))
    dm_spin = jnp.asarray(mf.make_rdm1())
    if mo_coeff.ndim != 3 or mo_coeff.shape[0] != 2:
        raise NotImplementedError(
            "unrestricted_reference_from_pyscf expects unrestricted orbitals with spin axis size 2."
        )
    if dm_spin.ndim == 2:
        dm_spin = jnp.stack([0.5 * dm_spin, 0.5 * dm_spin], axis=0)
    if dm_spin.ndim != 3 or dm_spin.shape[0] != 2:
        raise NotImplementedError("Expected unrestricted density matrix shape (2, nao, nao).")

    ao = jnp.asarray(numint.eval_ao(mf.mol, mf.grids.coords, deriv=0))
    ao_deriv1 = jnp.asarray(numint.eval_ao(mf.mol, mf.grids.coords, deriv=1))
    weights = jnp.asarray(mf.grids.weights)
    with mf.mol.with_common_orig(_charge_center(mf.mol)):
        dipole_integrals = jnp.asarray(mf.mol.intor_symmetric("int1e_r", comp=3))

    return UnrestrictedMoleculeReference(
        ao=ao,
        grid=GridReference(weights=weights, coords=jnp.asarray(mf.grids.coords)),
        dipole_integrals=dipole_integrals,
        rep_tensor=jnp.asarray(mf.mol.intor("int2e")),
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        rdm1=dm_spin,
        h1e=jnp.asarray(mf.get_hcore()),
        nuclear_repulsion=float(mf.mol.energy_nuc()),
        atom_coords=jnp.asarray(mf.mol.atom_coords()),
        atom_charges=jnp.asarray(mf.mol.atom_charges()),
        overlap_matrix=jnp.asarray(mf.get_ovlp()),
        ao_deriv1=ao_deriv1,
        mf_energy=float(getattr(mf, "e_tot", jnp.nan)),
        exact_exchange_fraction=_hybrid_fraction_from_mf(mf),
        nocc_alpha=nocc_alpha,
        nocc_beta=nocc_beta,
        hfx_omega_values=None,
    )


def restricted_reference_from_pyscf_with_jax_rhf(
    mf: Any,
    *,
    max_l: int = 1,
    rhf_config: RHFConfig | None = None,
    energy_target: float | None = None,
    grid_ao_backend: Literal["jax"] = "jax",
    compute_local_hfx_features: bool = False,
    compute_local_hfx_aux: bool = False,
    compute_local_pt2_features: bool = False,
    hfx_omega_values: tuple[float, ...] = (0.0, 0.4),
    hfx_chunk_size: int = 512,
) -> RestrictedMoleculeReference:
    """Build a restricted molecule reference with pure-JAX RHF orbitals/integrals."""

    if getattr(mf.mol, "spin", 0) != 0:
        raise NotImplementedError("Only restricted closed-shell PySCF references are supported.")
    if getattr(mf, "grids", None) is None:
        raise ValueError(
            "The PySCF mean-field object must provide numerical grids for Neural_xc features."
        )
    if getattr(mf.grids, "coords", None) is None:
        mf.grids.build()

    mol_cart = mf.mol if bool(getattr(mf.mol, "cart", False)) else _clone_pyscf_mol_cart(mf.mol)
    basis = basis_from_pyscf_mol_cart(mol_cart, max_l=max_l)

    s = overlap_matrix(basis)
    h1e = build_hcore(basis)
    eri = eri_tensor(basis)
    rhf = run_rhf_from_integrals(
        overlap=s,
        hcore=h1e,
        eri=eri,
        nelectron=mol_cart.nelectron,
        nuclear_repulsion=float(mol_cart.energy_nuc()),
        config=rhf_config,
    )
    if not rhf.converged:
        raise RuntimeError("Pure JAX RHF did not converge.")

    coords = jnp.asarray(mf.grids.coords)
    ao, ao_deriv1 = _eval_grid_ao(
        mol_cart,
        basis,
        coords,
        backend=grid_ao_backend,
    )
    weights = jnp.asarray(mf.grids.weights)

    dipole_integrals = dipole_matrix(basis)

    dm_total = jnp.asarray(rhf.density_matrix)
    half_dm = dm_total / 2.0
    mo_coeff = jnp.asarray(rhf.mo_coeff)
    mo_occ = jnp.asarray(rhf.mo_occ) / 2.0
    mo_energy = jnp.asarray(rhf.mo_energy)
    hfx_local = None
    hfx_nu = None
    pt2_local = None
    if compute_local_hfx_features:
        hfx_result = _local_hfx_features_from_basis_dm(
            basis,
            ao,
            (half_dm, half_dm),
            coords,
            omega_values=tuple(float(omega) for omega in hfx_omega_values),
            chunk_size=hfx_chunk_size,
            return_nu=bool(compute_local_hfx_aux),
        )
        if compute_local_hfx_aux:
            hfx_local, hfx_nu = hfx_result
        else:
            hfx_local = hfx_result
    nocc = int(np.count_nonzero(np.asarray(rhf.mo_occ) > 1e-8))
    eri_ovov, eri_ovvo, eri_oovv = _restricted_response_eri_slices_from_mo_tensor(
        np.asarray(eri),
        np.asarray(rhf.mo_coeff),
        nocc,
    )
    if compute_local_pt2_features:
        pt2_local = _local_pt2_feature_from_restricted_orbitals(
            ao,
            mo_coeff,
            mo_occ,
            mo_energy,
            rep_tensor=jnp.asarray(eri),
            eri_ovov=eri_ovov,
            nocc=nocc,
            density_floor=1e-12,
        )

    mf_energy = float(rhf.total_energy) if energy_target is None else float(energy_target)

    return RestrictedMoleculeReference(
        ao=ao,
        grid=GridReference(weights=weights, coords=coords),
        dipole_integrals=dipole_integrals,
        rep_tensor=jnp.asarray(eri),
        mo_coeff=jnp.stack([mo_coeff, mo_coeff], axis=0),
        mo_occ=jnp.stack([mo_occ, mo_occ], axis=0),
        mo_energy=jnp.stack([mo_energy, mo_energy], axis=0),
        rdm1=jnp.stack([half_dm, half_dm], axis=0),
        h1e=jnp.asarray(h1e),
        nuclear_repulsion=float(rhf.nuclear_repulsion),
        atom_coords=jnp.asarray(basis.atom_coords),
        atom_charges=jnp.asarray(basis.atom_charges),
        overlap_matrix=jnp.asarray(s),
        ao_deriv1=ao_deriv1,
        mf_energy=mf_energy,
        exact_exchange_fraction=_hybrid_fraction_from_mf(mf),
        nocc=nocc,
        hfx_local=hfx_local,
        hfx_nu=hfx_nu,
        pt2_local=pt2_local,
        eri_ovov=eri_ovov,
        eri_ovvo=eri_ovvo,
        eri_oovv=eri_oovv,
    )


def restricted_reference_from_pyscf_with_jax_rks(
    mf: Any,
    *,
    max_l: int = 3,
    rks_config: RKSConfig | None = None,
    xc_spec: str | None = None,
    energy_target: float | None = None,
    grid_ao_backend: Literal["jax"] = "jax",
    compute_local_hfx_features: bool = False,
    compute_local_hfx_aux: bool = False,
    compute_local_pt2_features: bool = False,
    hfx_omega_values: tuple[float, ...] = (0.0, 0.4),
    hfx_chunk_size: int = 512,
) -> RestrictedMoleculeReference:
    """Build a restricted molecule reference with pure-JAX RKS orbitals/integrals."""

    if getattr(mf.mol, "spin", 0) != 0:
        raise NotImplementedError("Only restricted closed-shell PySCF references are supported.")
    if getattr(mf, "grids", None) is None:
        raise ValueError(
            "The PySCF mean-field object must provide numerical grids for JAX-RKS features."
        )
    if getattr(mf.grids, "coords", None) is None:
        mf.grids.build()

    xc_spec_resolved = (
        _infer_jax_xc_spec_from_pyscf_label(getattr(mf, "xc", None))
        if xc_spec is None
        else str(xc_spec)
    )
    parse_xc(xc_spec_resolved)
    cfg = RKSConfig(xc_spec=xc_spec_resolved) if rks_config is None else rks_config
    if cfg.xc_spec != xc_spec_resolved:
        cfg = replace(cfg, xc_spec=xc_spec_resolved)

    mol_cart = mf.mol if bool(getattr(mf.mol, "cart", False)) else _clone_pyscf_mol_cart(mf.mol)
    basis = basis_from_pyscf_mol_cart(
        mol_cart,
        max_l=max_l,
        precompute_eri_groups=not _uses_cuda_direct_jk(cfg),
    )
    s, h1e = _overlap_hcore_for_jax_rks_reference(basis, cfg)
    eri_pair_matrix = None
    if cfg.jk_backend != "direct":
        eri_pair_matrix = jnp.asarray(
            mol_cart.intor(
                "int2e_cart" if bool(getattr(mol_cart, "cart", False)) else "int2e_sph",
                aosym="s4",
            )
        )

    coords = jnp.asarray(mf.grids.coords)
    ao, ao_deriv1 = _eval_grid_ao(
        mol_cart,
        basis,
        coords,
        backend=grid_ao_backend,
    )
    weights = jnp.asarray(mf.grids.weights)

    rks = run_rks_from_integrals(
        overlap=s,
        hcore=h1e,
        eri=None,
        eri_pair_matrix=eri_pair_matrix,
        nelectron=mol_cart.nelectron,
        nuclear_repulsion=float(mol_cart.energy_nuc()),
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid_weights=weights,
        direct_basis=basis if cfg.jk_backend == "direct" else None,
        init_mo_coeff=jnp.asarray(getattr(mf, "mo_coeff")),
        init_mo_occ=jnp.asarray(getattr(mf, "mo_occ")),
        init_mo_energy=jnp.asarray(getattr(mf, "mo_energy")),
        config=cfg,
    )
    if not rks.converged:
        if not (
            jnp.all(jnp.isfinite(rks.mo_coeff))
            and jnp.all(jnp.isfinite(rks.mo_energy))
            and jnp.all(jnp.isfinite(rks.density_matrix))
        ):
            raise RuntimeError("Pure JAX RKS did not converge to a finite solution.")

    dipole_integrals = dipole_matrix(basis)

    dm_total = jnp.asarray(rks.density_matrix)
    half_dm = dm_total / 2.0
    mo_coeff = jnp.asarray(rks.mo_coeff)
    mo_occ = jnp.asarray(rks.mo_occ) / 2.0
    mo_energy = jnp.asarray(rks.mo_energy)
    hfx_local = None
    hfx_nu = None
    pt2_local = None
    if compute_local_hfx_features:
        hfx_result = _local_hfx_features_from_basis_dm(
            basis,
            ao,
            (half_dm, half_dm),
            coords,
            omega_values=tuple(float(omega) for omega in hfx_omega_values),
            chunk_size=hfx_chunk_size,
            return_nu=bool(compute_local_hfx_aux),
        )
        if compute_local_hfx_aux:
            hfx_local, hfx_nu = hfx_result
        else:
            hfx_local = hfx_result
    mf_energy = float(rks.total_energy) if energy_target is None else float(energy_target)
    nocc = int(np.count_nonzero(np.asarray(rks.mo_occ) > 1e-8))
    eri_ovov, eri_ovvo, eri_oovv = eri_pair_matrix_to_mo_eri_slices(
        eri_pair_matrix,
        rks.mo_coeff,
        nocc=nocc,
    )
    if compute_local_pt2_features:
        pt2_local = _local_pt2_feature_from_restricted_orbitals(
            ao,
            mo_coeff,
            mo_occ,
            mo_energy,
            rep_tensor=jnp.zeros((0, 0, 0, 0), dtype=jnp.asarray(s).dtype),
            eri_ovov=eri_ovov,
            nocc=nocc,
            density_floor=cfg.density_floor,
        )

    return RestrictedMoleculeReference(
        ao=ao,
        grid=GridReference(weights=weights, coords=coords),
        dipole_integrals=dipole_integrals,
        rep_tensor=jnp.zeros((0, 0, 0, 0), dtype=jnp.asarray(s).dtype),
        mo_coeff=jnp.stack([mo_coeff, mo_coeff], axis=0),
        mo_occ=jnp.stack([mo_occ, mo_occ], axis=0),
        mo_energy=jnp.stack([mo_energy, mo_energy], axis=0),
        rdm1=jnp.stack([half_dm, half_dm], axis=0),
        h1e=jnp.asarray(h1e),
        nuclear_repulsion=float(rks.nuclear_repulsion),
        atom_coords=jnp.asarray(basis.atom_coords),
        atom_charges=jnp.asarray(basis.atom_charges),
        overlap_matrix=jnp.asarray(s),
        ao_deriv1=ao_deriv1,
        mf_energy=mf_energy,
        exact_exchange_fraction=float(rks.exact_exchange_fraction),
        nocc=nocc,
        hfx_local=hfx_local,
        hfx_nu=hfx_nu,
        pt2_local=pt2_local,
        eri_ovov=eri_ovov,
        eri_ovvo=eri_ovvo,
        eri_oovv=eri_oovv,
    )


def restricted_reference_from_pyscf_spec_with_jax_rks(
    *,
    atom: Any,
    basis: Any,
    xc_spec: str = "pbe",
    unit: str = "Angstrom",
    charge: int = 0,
    spin: int = 0,
    cart: bool = True,
    grids_level: int = 0,
    max_l: int = 3,
    rks_config: RKSConfig | None = None,
    grid_ao_backend: Literal["jax"] = "jax",
    energy_target: float | None = None,
    compute_local_hfx_features: bool = False,
    compute_local_hfx_aux: bool = False,
    compute_local_pt2_features: bool = False,
    hfx_omega_values: tuple[float, ...] = (0.0, 0.4),
    hfx_chunk_size: int = 512,
    verbose: int = 0,
    **mol_kwargs: Any,
) -> RestrictedMoleculeReference:
    """Legacy compatibility alias for the strict-JAX spec-driven reference builder."""

    return restricted_reference_from_spec_with_jax_rks(
        atom=atom,
        basis=basis,
        xc_spec=xc_spec,
        unit=unit,
        charge=charge,
        spin=spin,
        cart=cart,
        grids_level=grids_level,
        max_l=max_l,
        rks_config=rks_config,
        grid_ao_backend=grid_ao_backend,
        energy_target=energy_target,
        compute_local_hfx_features=compute_local_hfx_features,
        compute_local_hfx_aux=compute_local_hfx_aux,
        compute_local_pt2_features=compute_local_pt2_features,
        hfx_omega_values=hfx_omega_values,
        hfx_chunk_size=hfx_chunk_size,
        verbose=verbose,
        **mol_kwargs,
    )


def unrestricted_reference_from_pyscf_with_jax_uks(
    mf: Any,
    *,
    max_l: int = 3,
    uks_config: UKSConfig | None = None,
    xc_spec: str | None = None,
    energy_target: float | None = None,
    grid_ao_backend: Literal["jax"] = "jax",
    compute_local_hfx_features: bool = False,
    compute_local_hfx_aux: bool = False,
    hfx_omega_values: tuple[float, ...] = (0.0, 0.4),
    hfx_chunk_size: int = 512,
) -> UnrestrictedMoleculeReference:
    """Build an unrestricted molecule reference with pure-JAX UKS orbitals/integrals."""

    if getattr(mf, "grids", None) is None:
        raise ValueError(
            "The PySCF mean-field object must provide numerical grids for JAX-UKS features."
        )
    if getattr(mf.grids, "coords", None) is None:
        mf.grids.build()

    xc_spec_resolved = (
        _infer_jax_xc_spec_from_pyscf_label(getattr(mf, "xc", None))
        if xc_spec is None
        else str(xc_spec)
    )
    parse_xc(xc_spec_resolved)
    cfg = UKSConfig(xc_spec=xc_spec_resolved) if uks_config is None else uks_config
    if cfg.xc_spec != xc_spec_resolved:
        cfg = UKSConfig(
            xc_spec=xc_spec_resolved,
            max_cycle=cfg.max_cycle,
            conv_tol=cfg.conv_tol,
            conv_tol_density=cfg.conv_tol_density,
            damping=cfg.damping,
            level_shift=cfg.level_shift,
            orthogonalization_eps=cfg.orthogonalization_eps,
            density_floor=cfg.density_floor,
            potential_clip=cfg.potential_clip,
        )

    mol_cart = mf.mol if bool(getattr(mf.mol, "cart", False)) else _clone_pyscf_mol_cart(mf.mol)
    basis = basis_from_pyscf_mol_cart(mol_cart, max_l=max_l)
    s = overlap_matrix(basis)
    h1e = build_hcore(basis)
    eri = eri_tensor(basis)

    coords = jnp.asarray(mf.grids.coords)
    ao, ao_deriv1 = _eval_grid_ao(
        mol_cart,
        basis,
        coords,
        backend=grid_ao_backend,
    )
    weights = jnp.asarray(mf.grids.weights)

    mo_coeff_init = jnp.asarray(getattr(mf, "mo_coeff"))
    mo_occ_init = jnp.asarray(getattr(mf, "mo_occ"))
    mo_energy_init = jnp.asarray(getattr(mf, "mo_energy"))
    if mo_coeff_init.ndim != 3 or mo_coeff_init.shape[0] != 2:
        raise NotImplementedError(
            "unrestricted_reference_from_pyscf_with_jax_uks expects unrestricted PySCF orbitals."
        )
    nelec = getattr(mol_cart, "nelec", None)
    if nelec is None or len(nelec) != 2:
        raise ValueError("PySCF Mole must expose spin-resolved electron counts for UKS.")
    nalpha, nbeta = int(nelec[0]), int(nelec[1])

    uks = run_uks_from_integrals(
        overlap=s,
        hcore=h1e,
        eri=eri,
        nalpha=nalpha,
        nbeta=nbeta,
        nuclear_repulsion=float(mol_cart.energy_nuc()),
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid_weights=weights,
        init_mo_coeff_alpha=mo_coeff_init[0],
        init_mo_coeff_beta=mo_coeff_init[1],
        init_mo_occ_alpha=mo_occ_init[0],
        init_mo_occ_beta=mo_occ_init[1],
        init_mo_energy_alpha=mo_energy_init[0],
        init_mo_energy_beta=mo_energy_init[1],
        config=cfg,
    )
    if not uks.converged:
        if not (
            jnp.all(jnp.isfinite(uks.mo_coeff_alpha))
            and jnp.all(jnp.isfinite(uks.mo_coeff_beta))
            and jnp.all(jnp.isfinite(uks.mo_energy_alpha))
            and jnp.all(jnp.isfinite(uks.mo_energy_beta))
        ):
            raise RuntimeError("Pure JAX UKS did not converge to a finite solution.")

    dipole_integrals = dipole_matrix(basis)

    mf_energy = float(uks.total_energy) if energy_target is None else float(energy_target)
    nocc_alpha = int(np.count_nonzero(np.asarray(uks.mo_occ_alpha) > 1e-8))
    nocc_beta = int(np.count_nonzero(np.asarray(uks.mo_occ_beta) > 1e-8))
    hfx_local = None
    hfx_nu = None
    if compute_local_hfx_features:
        hfx_result = _local_hfx_features_from_basis_dm(
            basis,
            ao,
            (uks.density_matrix_alpha, uks.density_matrix_beta),
            coords,
            omega_values=tuple(float(omega) for omega in hfx_omega_values),
            chunk_size=hfx_chunk_size,
            return_nu=bool(compute_local_hfx_aux),
        )
        if compute_local_hfx_aux:
            hfx_local, hfx_nu = hfx_result
        else:
            hfx_local = hfx_result
    return UnrestrictedMoleculeReference(
        ao=ao,
        grid=GridReference(weights=weights, coords=coords),
        dipole_integrals=dipole_integrals,
        rep_tensor=jnp.asarray(eri),
        mo_coeff=jnp.stack([uks.mo_coeff_alpha, uks.mo_coeff_beta], axis=0),
        mo_occ=jnp.stack([uks.mo_occ_alpha, uks.mo_occ_beta], axis=0),
        mo_energy=jnp.stack([uks.mo_energy_alpha, uks.mo_energy_beta], axis=0),
        rdm1=jnp.stack([uks.density_matrix_alpha, uks.density_matrix_beta], axis=0),
        h1e=jnp.asarray(h1e),
        nuclear_repulsion=float(uks.nuclear_repulsion),
        atom_coords=jnp.asarray(basis.atom_coords),
        atom_charges=jnp.asarray(basis.atom_charges),
        overlap_matrix=jnp.asarray(s),
        ao_deriv1=ao_deriv1,
        mf_energy=mf_energy,
        exact_exchange_fraction=float(uks.exact_exchange_fraction),
        nocc_alpha=nocc_alpha,
        nocc_beta=nocc_beta,
        hfx_omega_values=(
            tuple(float(omega) for omega in hfx_omega_values)
            if compute_local_hfx_features
            else None
        ),
        hfx_local=hfx_local,
        hfx_nu=hfx_nu,
    )


__all__ = [
    "restricted_reference_from_pyscf",
    "unrestricted_reference_from_pyscf",
    "restricted_reference_from_pyscf_with_jax_rhf",
    "restricted_reference_from_pyscf_with_jax_rks",
    "restricted_reference_from_pyscf_spec_with_jax_rks",
    "unrestricted_reference_from_pyscf_with_jax_uks",
]
