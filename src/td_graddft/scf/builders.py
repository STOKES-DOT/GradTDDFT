from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from ..data.integrals import eri_pair_matrix_to_mo_eri_slices, precompile_eri_kernels
from ..data.integrals.libcint import LibcintGeometryGradPolicy
from ..data.molecule import MoleculeSpec
from ..xc_backend.jax_libxc import hybrid_coeff, parse_xc
from ..neural_xc.inputs import (
    _local_hfx_features_from_basis_dm,
    _local_pt2_feature_from_restricted_orbitals,
    _local_pt2_feature_from_unrestricted_orbitals,
)
from .core import _contains_jax_tracer, _host_float_unless_traced
from .features import _restricted_response_eri_slices_from_mo_tensor
from .inputs import build_rks_integral_inputs, build_uks_integral_inputs
from .molecules import QuadratureGrid, RestrictedMolecule, UnrestrictedMolecule
from .rks import RKSConfig, run_rks_from_integrals
from .uks import UKSConfig, run_uks_from_integrals


def _host_array(value: Any, dtype: Any | None = None) -> np.ndarray:
    return np.asarray(jax.device_get(value), dtype=dtype)


def _restricted_reference_array_packaging(
    *,
    mo_coeff: Any,
    mo_occ: Any,
    mo_energy: Any,
    half_dm: Any,
    h1e: Any,
    atom_coords: Any,
    atom_charges: Any,
    overlap: Any,
    df_factors: Any | None,
    dtype: Any,
    traced: bool,
) -> dict[str, Any]:
    if traced:
        return {
            "mo_coeff": jnp.stack([jnp.asarray(mo_coeff, dtype=dtype)] * 2, axis=0),
            "mo_occ": jnp.stack([jnp.asarray(mo_occ, dtype=dtype)] * 2, axis=0),
            "mo_energy": jnp.stack([jnp.asarray(mo_energy, dtype=dtype)] * 2, axis=0),
            "rdm1": jnp.stack([jnp.asarray(half_dm, dtype=dtype)] * 2, axis=0),
            "h1e": jnp.asarray(h1e, dtype=dtype),
            "atom_coords": jnp.asarray(atom_coords, dtype=dtype),
            "atom_charges": jnp.asarray(atom_charges, dtype=dtype),
            "overlap_matrix": jnp.asarray(overlap, dtype=dtype),
            "df_factors": (
                jnp.asarray(df_factors, dtype=dtype) if df_factors is not None else None
            ),
        }

    host_dtype = np.dtype(dtype)
    mo_coeff_arr = _host_array(mo_coeff, host_dtype)
    mo_occ_arr = _host_array(mo_occ, host_dtype)
    mo_energy_arr = _host_array(mo_energy, host_dtype)
    half_dm_arr = _host_array(half_dm, host_dtype)
    return {
        "mo_coeff": np.stack([mo_coeff_arr, mo_coeff_arr], axis=0),
        "mo_occ": np.stack([mo_occ_arr, mo_occ_arr], axis=0),
        "mo_energy": np.stack([mo_energy_arr, mo_energy_arr], axis=0),
        "rdm1": np.stack([half_dm_arr, half_dm_arr], axis=0),
        "h1e": _host_array(h1e, host_dtype),
        "atom_coords": _host_array(atom_coords, host_dtype),
        "atom_charges": _host_array(atom_charges, host_dtype),
        "overlap_matrix": _host_array(overlap, host_dtype),
        "df_factors": (
            _host_array(df_factors, host_dtype) if df_factors is not None else None
        ),
    }


def _empty_rep_tensor_like(overlap: Any, *, traced: bool) -> Any:
    dtype = jnp.asarray(overlap).dtype if traced else np.asarray(overlap).dtype
    if traced:
        return jnp.zeros((0, 0, 0, 0), dtype=dtype)
    return np.zeros((0, 0, 0, 0), dtype=dtype)


def restricted_molecule_from_spec_with_jax_rks(
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
    integral_backend: Literal["jax", "cpu", "gpu", "libcint"] = "cpu",
    libcint_geometry_grad_policy: LibcintGeometryGradPolicy = "analytic",
    energy_target: float | None = None,
    compute_local_hfx_features: bool = False,
    compute_local_hfx_aux: bool = False,
    compute_local_pt2_features: bool = False,
    hfx_omega_values: tuple[float, ...] = (0.0, 0.4),
    hfx_chunk_size: int = 512,
    include_dipole_integrals: bool = True,
    init_guess: Any = "minao",
    chkfile: str | None = None,
    init_guess_sap_basis: Any | None = None,
    init_guess_chkfile_project: bool | None = None,
    precompile_eri: bool = False,
    precompile_eri_chunk_size: int = 512,
    verbose: int = 0,
    **mol_kwargs: Any,
) -> RestrictedMolecule:
    """Build a restricted strict-JAX RKS reference directly from molecule specs."""

    if isinstance(atom, MoleculeSpec):
        charge = int(atom.charge)
        spin = int(atom.spin)
    if int(spin) != 0:
        raise NotImplementedError(
            "restricted_molecule_from_spec_with_jax_rks only supports closed-shell systems."
        )
    if not bool(cart):
        raise NotImplementedError(
            "restricted_molecule_from_spec_with_jax_rks currently supports cart=True only."
        )

    xc_spec_resolved = str(xc_spec)
    parse_xc(xc_spec_resolved)
    cfg = RKSConfig(xc_spec=xc_spec_resolved) if rks_config is None else rks_config
    if cfg.xc_spec != xc_spec_resolved:
        cfg = replace(cfg, xc_spec=xc_spec_resolved)

    libcint_grad_policy_mode = str(libcint_geometry_grad_policy).lower()
    if libcint_grad_policy_mode not in {"analytic", "error", "zero"}:
        raise ValueError(
            f"Unsupported libcint_geometry_grad_policy={libcint_geometry_grad_policy!r}. "
            "Expected 'analytic', 'error', or 'zero'."
        )

    exact_exchange_fraction = float(hybrid_coeff(xc_spec_resolved))
    scf_inputs = build_rks_integral_inputs(
        atom=atom,
        basis=basis,
        config=cfg,
        xc_spec=xc_spec_resolved,
        unit=unit,
        charge=charge,
        spin=spin,
        cart=cart,
        grids_level=grids_level,
        max_l=max_l,
        grid_ao_backend=grid_ao_backend,
        integral_backend=integral_backend,
        libcint_geometry_grad_policy=libcint_geometry_grad_policy,
        include_dipole_integrals=include_dipole_integrals,
        init_guess=init_guess,
        chkfile=chkfile,
        init_guess_sap_basis=init_guess_sap_basis,
        init_guess_chkfile_project=init_guess_chkfile_project,
        precompile_eri=precompile_eri,
        precompile_eri_chunk_size=precompile_eri_chunk_size,
        _precompile_eri_kernels=precompile_eri_kernels,
        verbose=verbose,
        **mol_kwargs,
    )
    basis_cart = scf_inputs.basis
    s = scf_inputs.overlap
    h1e = scf_inputs.hcore
    eri = scf_inputs.eri
    eri_pair_matrix = scf_inputs.eri_pair_matrix
    df_factors = scf_inputs.df_factors
    coords = scf_inputs.coords
    weights = scf_inputs.grid_weights
    ao = scf_inputs.ao
    ao_deriv1 = scf_inputs.ao_deriv1
    ao_laplacian = scf_inputs.ao_laplacian
    dipole_integrals = scf_inputs.dipole_integrals
    if scf_inputs.geometry_is_traced:
        raise NotImplementedError(
            "Explicit traceable SCF execution has been removed. "
            "Use implicit differential SCF instead."
        )
    nelectron = scf_inputs.nelectron
    rks = run_rks_from_integrals(
        **scf_inputs.as_rks_kwargs(),
        config=cfg,
    )
    if not rks.converged:
        if not (
            jnp.all(jnp.isfinite(rks.mo_coeff))
            and jnp.all(jnp.isfinite(rks.mo_energy))
            and jnp.all(jnp.isfinite(rks.density_matrix))
        ):
            raise RuntimeError(
                "Pure JAX RKS from molecule specs did not converge to a finite solution."
            )

    dm_total = _host_array(rks.density_matrix)
    half_dm = dm_total * 0.5
    mo_coeff = _host_array(rks.mo_coeff)
    mo_occ = _host_array(rks.mo_occ) * 0.5
    mo_energy = _host_array(rks.mo_energy)
    hfx_local = None
    hfx_nu = None
    pt2_local = None
    reference_eri_pair_matrix = None
    if compute_local_hfx_features:
        hfx_result = _local_hfx_features_from_basis_dm(
            basis_cart,
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
    mf_energy = (
        _host_float_unless_traced(rks.total_energy)
        if energy_target is None
        else _host_float_unless_traced(energy_target)
    )
    nocc = int(np.count_nonzero(np.asarray(rks.mo_occ) > 1e-8))
    if cfg.jk_backend == "df":
        if df_factors is None:
            raise RuntimeError("DF backend requested but df_factors were not constructed.")
        eri_ovov = None
        eri_ovvo = None
        eri_oovv = None
        rep_tensor = _empty_rep_tensor_like(s, traced=False)
    else:
        if eri is None and eri_pair_matrix is None:
            if cfg.jk_backend == "direct":
                eri_pair_matrix = scf_inputs.response_eri_pair_matrix()
            else:
                raise RuntimeError("Full ERI backend requested but exact ERI data is missing.")
        needs_exchange_slices = abs(exact_exchange_fraction) > 1e-14
        if eri_pair_matrix is not None:
            reference_eri_pair_matrix = _host_array(eri_pair_matrix, np.asarray(s).dtype)
            eri_ovov, eri_ovvo, eri_oovv = eri_pair_matrix_to_mo_eri_slices(
                eri_pair_matrix,
                rks.mo_coeff,
                nocc=nocc,
                include_oovv=needs_exchange_slices,
            )
            rep_tensor = _empty_rep_tensor_like(s, traced=False)
        else:
            eri_ovov, eri_ovvo, eri_oovv = _restricted_response_eri_slices_from_mo_tensor(
                np.asarray(eri),
                np.asarray(rks.mo_coeff),
                nocc,
                include_oovv=needs_exchange_slices,
            )
            rep_tensor = jnp.asarray(eri)

    if compute_local_pt2_features:
        pt2_local = _local_pt2_feature_from_restricted_orbitals(
            ao,
            mo_coeff,
            mo_occ,
            mo_energy,
            rep_tensor=rep_tensor,
            eri_ovov=eri_ovov,
            eri_pair_matrix=eri_pair_matrix,
            df_factors=df_factors,
            nocc=nocc,
            density_floor=cfg.density_floor,
        )

    reference_arrays = _restricted_reference_array_packaging(
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        half_dm=half_dm,
        h1e=h1e,
        atom_coords=basis_cart.atom_coords,
        atom_charges=basis_cart.atom_charges,
        overlap=s,
        df_factors=df_factors,
        dtype=jnp.asarray(s).dtype,
        traced=False,
    )
    return RestrictedMolecule(
        ao=ao,
        grid=QuadratureGrid(weights=weights, coords=coords),
        dipole_integrals=dipole_integrals,
        rep_tensor=rep_tensor,
        mo_coeff=reference_arrays["mo_coeff"],
        mo_occ=reference_arrays["mo_occ"],
        mo_energy=reference_arrays["mo_energy"],
        rdm1=reference_arrays["rdm1"],
        h1e=reference_arrays["h1e"],
        nuclear_repulsion=_host_float_unless_traced(rks.nuclear_repulsion),
        atom_coords=reference_arrays["atom_coords"],
        atom_charges=reference_arrays["atom_charges"],
        overlap_matrix=reference_arrays["overlap_matrix"],
        ao_deriv1=ao_deriv1,
        ao_laplacian=ao_laplacian,
        mf_energy=mf_energy,
        exact_exchange_fraction=exact_exchange_fraction,
        nocc=nocc,
        hfx_omega_values=(
            jnp.asarray(hfx_omega_values, dtype=reference_arrays["mo_coeff"].dtype)
            if compute_local_hfx_features
            else None
        ),
        hfx_local=hfx_local,
        hfx_nu=hfx_nu,
        pt2_local=pt2_local,
        df_factors=reference_arrays["df_factors"],
        eri_pair_matrix=reference_eri_pair_matrix,
        eri_ovov=eri_ovov,
        eri_ovvo=eri_ovvo,
        eri_oovv=eri_oovv,
        scf_converged=bool(rks.converged),
    )


def build_restricted_reference_from_facade(
    spec: MoleculeSpec,
    *,
    mol: Any,
    xc: str,
    grids_level: int,
    max_l: int,
    integral_backend: str,
    geometry_grad_policy: str,
    grid_ao_backend: str,
    rks_config: RKSConfig,
    init_guess: Any,
    chkfile: str | None,
    sap_basis: Any | None,
    init_guess_chkfile_project: bool | None,
    compute_local_hfx_features: bool,
    compute_local_hfx_aux: bool,
    hfx_omega_values: tuple[float, ...],
    hfx_chunk_size: int,
    include_dipole_integrals: bool,
    geometry_is_traced: bool,
    reference_builder: Any,
) -> Any:
    if geometry_is_traced:
        raise NotImplementedError(
            "Explicit traceable SCF execution has been removed. "
            "Use implicit differential SCF instead."
        )
    return reference_builder(
        atom=spec,
        basis=mol.basis,
        xc_spec=xc,
        unit=mol.unit,
        charge=mol.charge,
        spin=mol.spin,
        cart=mol.cart,
        grids_level=grids_level,
        max_l=max_l,
        rks_config=rks_config,
        grid_ao_backend=grid_ao_backend,
        integral_backend=integral_backend,
        libcint_geometry_grad_policy=geometry_grad_policy,
        init_guess=init_guess,
        chkfile=chkfile,
        init_guess_sap_basis=sap_basis,
        init_guess_chkfile_project=init_guess_chkfile_project,
        compute_local_hfx_features=compute_local_hfx_features,
        compute_local_hfx_aux=compute_local_hfx_aux,
        hfx_omega_values=hfx_omega_values,
        hfx_chunk_size=hfx_chunk_size,
        include_dipole_integrals=include_dipole_integrals,
        verbose=mol.verbose,
    )


def build_restricted_scf_result_from_facade(
    spec: MoleculeSpec,
    *,
    mol: Any,
    xc: str,
    grids_level: int,
    max_l: int,
    integral_backend: str,
    geometry_grad_policy: str,
    grid_ao_backend: str,
    rks_config: RKSConfig,
    init_guess: Any,
    chkfile: str | None,
    sap_basis: Any | None,
    init_guess_chkfile_project: bool | None,
    geometry_is_traced: bool,
    build_inputs_fn: Any,
    run_rks_fn: Any,
) -> Any:
    if geometry_is_traced:
        raise NotImplementedError(
            "Explicit traceable SCF execution has been removed. "
            "Use implicit differential SCF instead."
        )
    scf_inputs = build_inputs_fn(
        atom=spec,
        basis=mol.basis,
        config=rks_config,
        xc_spec=xc,
        unit=mol.unit,
        charge=mol.charge,
        spin=mol.spin,
        cart=mol.cart,
        grids_level=grids_level,
        max_l=max_l,
        grid_ao_backend=grid_ao_backend,
        integral_backend=integral_backend,
        libcint_geometry_grad_policy=geometry_grad_policy,
        init_guess=init_guess,
        chkfile=chkfile,
        init_guess_sap_basis=sap_basis,
        init_guess_chkfile_project=init_guess_chkfile_project,
        include_dipole_integrals=False,
        verbose=mol.verbose,
    )
    return run_rks_fn(
        **scf_inputs.as_rks_kwargs(),
        config=rks_config,
    )


def unrestricted_molecule_from_spec_with_jax_uks(
    *,
    atom: Any,
    basis: Any,
    xc_spec: str = "pbe",
    unit: str = "Angstrom",
    charge: int = 0,
    spin: int = 1,
    cart: bool = True,
    grids_level: int = 0,
    max_l: int = 3,
    uks_config: UKSConfig | None = None,
    grid_ao_backend: Literal["jax"] = "jax",
    integral_backend: Literal["jax", "cpu", "gpu", "libcint"] = "cpu",
    libcint_geometry_grad_policy: LibcintGeometryGradPolicy = "error",
    energy_target: float | None = None,
    compute_local_hfx_features: bool = False,
    compute_local_hfx_aux: bool = False,
    compute_local_pt2_features: bool = False,
    hfx_omega_values: tuple[float, ...] = (0.0, 0.4),
    hfx_chunk_size: int = 512,
    init_guess: Any = "minao",
    chkfile: str | None = None,
    init_guess_sap_basis: Any | None = None,
    init_guess_chkfile_project: bool | None = None,
    precompile_eri: bool = False,
    precompile_eri_chunk_size: int = 512,
    verbose: int = 0,
    **mol_kwargs: Any,
) -> UnrestrictedMolecule:
    """Build an unrestricted strict-JAX UKS reference directly from molecule specs."""

    if not bool(cart):
        raise NotImplementedError(
            "unrestricted_molecule_from_spec_with_jax_uks currently supports cart=True only."
        )
    if isinstance(atom, MoleculeSpec):
        charge = int(atom.charge)
        spin = int(atom.spin)

    xc_spec_resolved = str(xc_spec)
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

    scf_inputs = build_uks_integral_inputs(
        atom=atom,
        basis=basis,
        config=cfg,
        xc_spec=xc_spec_resolved,
        unit=unit,
        charge=charge,
        spin=spin,
        cart=cart,
        grids_level=grids_level,
        max_l=max_l,
        grid_ao_backend=grid_ao_backend,
        integral_backend=integral_backend,
        libcint_geometry_grad_policy=libcint_geometry_grad_policy,
        precompile_eri=precompile_eri,
        precompile_eri_chunk_size=precompile_eri_chunk_size,
        _precompile_eri_kernels=precompile_eri_kernels,
        init_guess=init_guess,
        chkfile=chkfile,
        init_guess_sap_basis=init_guess_sap_basis,
        init_guess_chkfile_project=init_guess_chkfile_project,
        verbose=verbose,
        **mol_kwargs,
    )
    basis_cart = scf_inputs.basis
    s = scf_inputs.overlap
    h1e = scf_inputs.hcore
    eri = scf_inputs.eri
    coords = scf_inputs.coords
    weights = scf_inputs.grid_weights
    ao = scf_inputs.ao
    ao_deriv1 = scf_inputs.ao_deriv1
    ao_laplacian = scf_inputs.ao_laplacian
    dipole_integrals = scf_inputs.dipole_integrals

    uks = run_uks_from_integrals(
        **scf_inputs.as_uks_kwargs(),
        config=cfg,
    )
    uks_is_traceable = _contains_jax_tracer(uks.total_energy)
    if not uks_is_traceable and not uks.converged:
        if not (
            jnp.all(jnp.isfinite(uks.mo_coeff_alpha))
            and jnp.all(jnp.isfinite(uks.mo_coeff_beta))
            and jnp.all(jnp.isfinite(uks.mo_energy_alpha))
            and jnp.all(jnp.isfinite(uks.mo_energy_beta))
            and jnp.all(jnp.isfinite(uks.density_matrix_alpha))
            and jnp.all(jnp.isfinite(uks.density_matrix_beta))
        ):
            raise RuntimeError(
                "Pure JAX UKS from molecule specs did not converge to a finite solution."
            )

    hfx_local = None
    hfx_nu = None
    pt2_local = None
    if compute_local_hfx_features:
        hfx_result = _local_hfx_features_from_basis_dm(
            basis_cart,
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
    if compute_local_pt2_features:
        pt2_local = _local_pt2_feature_from_unrestricted_orbitals(
            ao,
            jnp.stack([uks.mo_coeff_alpha, uks.mo_coeff_beta], axis=0),
            jnp.stack([uks.mo_occ_alpha, uks.mo_occ_beta], axis=0),
            jnp.stack([uks.mo_energy_alpha, uks.mo_energy_beta], axis=0),
            rep_tensor=jnp.asarray(eri),
            density_floor=cfg.density_floor,
        )

    mf_energy = (
        _host_float_unless_traced(uks.total_energy)
        if energy_target is None
        else _host_float_unless_traced(energy_target)
    )
    nocc_alpha = int(np.count_nonzero(np.asarray(uks.mo_occ_alpha) > 1e-8))
    nocc_beta = int(np.count_nonzero(np.asarray(uks.mo_occ_beta) > 1e-8))
    return UnrestrictedMolecule(
        ao=ao,
        grid=QuadratureGrid(weights=weights, coords=coords),
        dipole_integrals=dipole_integrals,
        rep_tensor=jnp.asarray(eri),
        mo_coeff=jnp.stack([uks.mo_coeff_alpha, uks.mo_coeff_beta], axis=0),
        mo_occ=jnp.stack([uks.mo_occ_alpha, uks.mo_occ_beta], axis=0),
        mo_energy=jnp.stack([uks.mo_energy_alpha, uks.mo_energy_beta], axis=0),
        rdm1=jnp.stack([uks.density_matrix_alpha, uks.density_matrix_beta], axis=0),
        h1e=jnp.asarray(h1e),
        nuclear_repulsion=_host_float_unless_traced(uks.nuclear_repulsion),
        atom_coords=jnp.asarray(basis_cart.atom_coords),
        atom_charges=jnp.asarray(basis_cart.atom_charges),
        overlap_matrix=jnp.asarray(s),
        ao_deriv1=ao_deriv1,
        ao_laplacian=ao_laplacian,
        mf_energy=mf_energy,
        exact_exchange_fraction=float(uks.exact_exchange_fraction),
        nocc_alpha=nocc_alpha,
        nocc_beta=nocc_beta,
        hfx_omega_values=(
            jnp.asarray(hfx_omega_values, dtype=jnp.asarray(uks.mo_coeff_alpha).dtype)
            if compute_local_hfx_features
            else None
        ),
        hfx_local=hfx_local,
        hfx_nu=hfx_nu,
        pt2_local=pt2_local,
    )


def build_unrestricted_reference_from_facade(
    spec: MoleculeSpec,
    *,
    mol: Any,
    xc: str,
    grids_level: int,
    max_l: int,
    integral_backend: str,
    geometry_grad_policy: str,
    grid_ao_backend: str,
    uks_config: UKSConfig,
    init_guess: Any,
    chkfile: str | None,
    sap_basis: Any | None,
    init_guess_chkfile_project: bool | None,
    compute_local_hfx_features: bool = False,
    compute_local_hfx_aux: bool = False,
    compute_local_pt2_features: bool = False,
    hfx_omega_values: tuple[float, ...] = (0.0, 0.4),
    hfx_chunk_size: int = 512,
    reference_builder: Any | None = None,
) -> Any:
    if reference_builder is None:
        raise ValueError("reference_builder must be provided for unrestricted facade references.")
    return reference_builder(
        atom=spec,
        basis=mol.basis,
        xc_spec=xc,
        unit=mol.unit,
        charge=mol.charge,
        spin=mol.spin,
        cart=mol.cart,
        grids_level=grids_level,
        max_l=max_l,
        uks_config=uks_config,
        grid_ao_backend=grid_ao_backend,
        integral_backend=integral_backend,
        libcint_geometry_grad_policy=geometry_grad_policy,
        compute_local_hfx_features=compute_local_hfx_features,
        compute_local_hfx_aux=compute_local_hfx_aux,
        compute_local_pt2_features=compute_local_pt2_features,
        hfx_omega_values=hfx_omega_values,
        hfx_chunk_size=hfx_chunk_size,
        init_guess=init_guess,
        chkfile=chkfile,
        init_guess_sap_basis=sap_basis,
        init_guess_chkfile_project=init_guess_chkfile_project,
        verbose=mol.verbose,
    )


__all__ = [
    "_restricted_reference_array_packaging",
    "build_restricted_reference_from_facade",
    "build_restricted_scf_result_from_facade",
    "build_unrestricted_reference_from_facade",
    "restricted_molecule_from_spec_with_jax_rks",
    "unrestricted_molecule_from_spec_with_jax_uks",
]
