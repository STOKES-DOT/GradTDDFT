from __future__ import annotations

from typing import Any, Literal

import numpy as np
import jax.numpy as jnp

from td_graddft.df import true_df_factors_from_libcint_mol
from td_graddft.data.ris_auxbasis import minimal_ris_auxbasis_for_mol
from td_graddft.scf.features import (
    _charge_center,
)
from td_graddft.neural_xc.inputs import (
    ChunkedHFXNu,
    _local_hfx_features_from_dm,
    _local_pt2_feature_and_fock_response_from_restricted_orbitals,
    _local_pt2_feature_from_unrestricted_orbitals,
)
from td_graddft.scf.molecules import QuadratureGrid, RestrictedMolecule, UnrestrictedMolecule


ArrayBackend = Literal["jax", "host"]
HFXNuStorage = Literal["auto", "dense", "chunked"]
ReferenceJKBackend = Literal["full", "df"]
ResponseDFMode = Literal["none", "df", "ris"]


def _backend_array(value: Any, *, array_backend: ArrayBackend) -> Any:
    if array_backend == "jax":
        return jnp.asarray(value)
    if array_backend == "host":
        return np.asarray(value)
    raise ValueError(f"Unsupported array_backend {array_backend!r}.")


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


def _molecule_atom_count(mol: Any) -> int:
    natm = getattr(mol, "natm", None)
    if natm is not None:
        return int(natm)
    return int(np.asarray(mol.atom_coords()).shape[0])


def _use_chunked_hfx_nu(mol: Any, storage: HFXNuStorage) -> bool:
    if storage == "dense":
        return False
    if storage == "chunked":
        return True
    if storage == "auto":
        return _molecule_atom_count(mol) > 3
    raise ValueError(f"Unsupported hfx_nu_storage={storage!r}.")


def _build_response_df_factors(
    mol: Any,
    *,
    response_df_mode: ResponseDFMode,
    response_ris_theta: float,
    response_ris_j_fit: str,
    response_ris_k_fit: str,
    existing_df_factors: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any] | None]:
    mode = str(response_df_mode).lower()
    if mode == "none":
        return None, None, None
    if mode == "df":
        factors = (
            np.asarray(existing_df_factors)
            if existing_df_factors is not None
            else np.asarray(true_df_factors_from_libcint_mol(mol))
        )
        return factors, factors, {"response_df_mode": "df"}
    if mode == "ris":
        auxbasis_j = minimal_ris_auxbasis_for_mol(
            mol,
            theta=float(response_ris_theta),
            fitting_basis=str(response_ris_j_fit),
        )
        factors_j = np.asarray(true_df_factors_from_libcint_mol(mol, auxbasis=auxbasis_j))
        if str(response_ris_k_fit).lower() == str(response_ris_j_fit).lower():
            factors_k = factors_j
        else:
            auxbasis_k = minimal_ris_auxbasis_for_mol(
                mol,
                theta=float(response_ris_theta),
                fitting_basis=str(response_ris_k_fit),
            )
            factors_k = np.asarray(true_df_factors_from_libcint_mol(mol, auxbasis=auxbasis_k))
        return (
            factors_j,
            factors_k,
            {
                "response_df_mode": "ris",
                "ris_theta": float(response_ris_theta),
                "ris_j_fit": str(response_ris_j_fit).lower(),
                "ris_k_fit": str(response_ris_k_fit).lower(),
            },
        )
    raise ValueError("response_df_mode must be one of {'none', 'df', 'ris'}.")


def restricted_reference_from_pyscf(
    mf: Any,
    *,
    compute_local_hfx_features: bool = False,
    compute_local_hfx_aux: bool = False,
    compute_local_pt2_features: bool = False,
    hfx_omega_values: tuple[float, ...] = (0.0, 0.4),
    hfx_chunk_size: int = 512,
    array_backend: ArrayBackend = "jax",
    hfx_nu_storage: HFXNuStorage = "auto",
    jk_backend: ReferenceJKBackend = "full",
    response_df_mode: ResponseDFMode = "none",
    response_ris_theta: float = 0.2,
    response_ris_j_fit: Literal["s", "sp", "spd"] = "sp",
    response_ris_k_fit: Literal["s", "sp", "spd"] = "s",
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
    ao_deriv1_np = np.asarray(numint.eval_ao(mf.mol, mf.grids.coords, deriv=1))
    weights_np = np.asarray(mf.grids.weights)
    dm_total_np = np.asarray(mf.make_rdm1())
    half_dm_np = dm_total_np / 2.0
    mo_coeff_np = np.asarray(mf.mo_coeff)
    mo_occ_np = np.asarray(mf.mo_occ) / 2.0
    mo_energy_np = np.asarray(mf.mo_energy)
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 1e-8))
    jk_backend_norm = str(jk_backend).lower()
    if jk_backend_norm == "df":
        df_factors_np = np.asarray(true_df_factors_from_libcint_mol(mf.mol))
        eri_pair_matrix_np = None
        integral_dtype = df_factors_np.dtype
    elif jk_backend_norm == "full":
        eri_pair_matrix_np = np.asarray(mf.mol.intor("int2e", aosym="s4"))
        df_factors_np = None
        integral_dtype = eri_pair_matrix_np.dtype
    else:
        raise ValueError("jk_backend must be one of {'full', 'df'}.")
    response_df_factors_j_np, response_df_factors_k_np, response_df_metadata = (
        _build_response_df_factors(
            mf.mol,
            response_df_mode=response_df_mode,
            response_ris_theta=response_ris_theta,
            response_ris_j_fit=response_ris_j_fit,
            response_ris_k_fit=response_ris_k_fit,
            existing_df_factors=df_factors_np,
        )
    )
    with mf.mol.with_common_orig(_charge_center(mf.mol)):
        dipole_integrals_np = np.asarray(mf.mol.intor_symmetric("int1e_r", comp=3))

    hfx_local = None
    hfx_fxx = None
    hfx_nu = None
    hfx_nu_api = None
    pt2_local = None
    pt2_fock_response = None
    if compute_local_hfx_features:
        use_chunked_hfx_nu = bool(compute_local_hfx_aux) and _use_chunked_hfx_nu(
            mf.mol,
            hfx_nu_storage,
        )
        hfx_result = _local_hfx_features_from_dm(
            mf.mol,
            ao_np,
            (half_dm_np, half_dm_np),
            np.asarray(mf.grids.coords),
            omega_values=tuple(float(omega) for omega in hfx_omega_values),
            chunk_size=hfx_chunk_size,
            return_nu=bool(compute_local_hfx_aux) and not use_chunked_hfx_nu,
            return_fxx=True,
        )
        if compute_local_hfx_aux and not use_chunked_hfx_nu:
            hfx_local_np, hfx_nu_np, hfx_fxx_np = hfx_result
            hfx_nu = _backend_array(hfx_nu_np, array_backend=array_backend)
        else:
            hfx_local_np, hfx_fxx_np = hfx_result
            if use_chunked_hfx_nu:
                hfx_nu_api = ChunkedHFXNu.from_pyscf_mol(
                    mf.mol,
                    np.asarray(mf.grids.coords),
                    omega_values=tuple(float(omega) for omega in hfx_omega_values),
                    nao=int(ao_np.shape[1]),
                    chunk_size=int(hfx_chunk_size),
                )
        hfx_local = _backend_array(hfx_local_np, array_backend=array_backend)
        hfx_fxx = _backend_array(hfx_fxx_np, array_backend=array_backend)
    if compute_local_pt2_features:
        pt2_local_jax, pt2_fock_response_jax = (
            _local_pt2_feature_and_fock_response_from_restricted_orbitals(
                jnp.asarray(ao_np),
                jnp.asarray(mo_coeff_np),
                jnp.asarray(mo_occ_np),
                jnp.asarray(mo_energy_np),
                rep_tensor=jnp.zeros((0, 0, 0, 0), dtype=integral_dtype),
                eri_ovov=None,
                eri_pair_matrix=(
                    None if eri_pair_matrix_np is None else jnp.asarray(eri_pair_matrix_np)
                ),
                df_factors=None if df_factors_np is None else jnp.asarray(df_factors_np),
                nocc=nocc,
            )
        )
        pt2_local = _backend_array(pt2_local_jax, array_backend=array_backend)
        pt2_fock_response = _backend_array(
            pt2_fock_response_jax,
            array_backend=array_backend,
        )

    ao = _backend_array(ao_np, array_backend=array_backend)
    ao_deriv1 = _backend_array(ao_deriv1_np, array_backend=array_backend)
    weights = _backend_array(weights_np, array_backend=array_backend)
    half_dm = _backend_array(half_dm_np, array_backend=array_backend)
    mo_coeff = _backend_array(mo_coeff_np, array_backend=array_backend)
    mo_occ = _backend_array(mo_occ_np, array_backend=array_backend)
    mo_energy = _backend_array(mo_energy_np, array_backend=array_backend)
    rep_tensor = _backend_array(
        np.zeros((0, 0, 0, 0), dtype=integral_dtype),
        array_backend=array_backend,
    )
    eri_pair_matrix = (
        None
        if eri_pair_matrix_np is None
        else _backend_array(eri_pair_matrix_np, array_backend=array_backend)
    )
    df_factors = (
        None
        if df_factors_np is None
        else _backend_array(df_factors_np, array_backend=array_backend)
    )
    response_df_factors_j = (
        None
        if response_df_factors_j_np is None
        else _backend_array(response_df_factors_j_np, array_backend=array_backend)
    )
    response_df_factors_k = (
        None
        if response_df_factors_k_np is None
        else _backend_array(response_df_factors_k_np, array_backend=array_backend)
    )

    return RestrictedMolecule(
        ao=ao,
        grid=QuadratureGrid(
            weights=weights,
            coords=_backend_array(mf.grids.coords, array_backend=array_backend),
        ),
        dipole_integrals=_backend_array(dipole_integrals_np, array_backend=array_backend),
        rep_tensor=rep_tensor,
        mo_coeff=_backend_array(
            np.stack([mo_coeff_np, mo_coeff_np], axis=0),
            array_backend=array_backend,
        ),
        mo_occ=_backend_array(
            np.stack([mo_occ_np, mo_occ_np], axis=0),
            array_backend=array_backend,
        ),
        mo_energy=_backend_array(
            np.stack([mo_energy_np, mo_energy_np], axis=0),
            array_backend=array_backend,
        ),
        rdm1=_backend_array(
            np.stack([half_dm_np, half_dm_np], axis=0),
            array_backend=array_backend,
        ),
        h1e=_backend_array(mf.get_hcore(), array_backend=array_backend),
        nuclear_repulsion=float(mf.mol.energy_nuc()),
        atom_coords=_backend_array(mf.mol.atom_coords(), array_backend=array_backend),
        atom_charges=_backend_array(mf.mol.atom_charges(), array_backend=array_backend),
        overlap_matrix=_backend_array(mf.get_ovlp(), array_backend=array_backend),
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
        hfx_fxx=hfx_fxx,
        hfx_nu=hfx_nu,
        hfx_nu_api=hfx_nu_api,
        pt2_local=pt2_local,
        pt2_fock_response=pt2_fock_response,
        df_factors=df_factors,
        response_df_factors_j=response_df_factors_j,
        response_df_factors_k=response_df_factors_k,
        response_df_metadata=response_df_metadata,
        eri_pair_matrix=eri_pair_matrix,
        eri_ovov=None,
        eri_ovvo=None,
        eri_oovv=None,
    )


def unrestricted_reference_from_pyscf(
    mf: Any,
    *,
    compute_local_hfx_features: bool = False,
    compute_local_hfx_aux: bool = False,
    compute_local_pt2_features: bool = False,
    hfx_omega_values: tuple[float, ...] = (0.0, 0.4),
    hfx_chunk_size: int = 512,
    array_backend: ArrayBackend = "jax",
    hfx_nu_storage: HFXNuStorage = "dense",
    jk_backend: ReferenceJKBackend = "full",
) -> UnrestrictedMolecule:
    """Convert an unrestricted PySCF SCF/DFT object to a TD-GradDFT-ready reference."""

    try:
        from pyscf.dft import numint
    except ModuleNotFoundError as exc:
        raise ImportError("PySCF is required for unrestricted_reference_from_pyscf.") from exc

    if getattr(mf, "mo_coeff", None) is None:
        raise ValueError("PySCF mean-field object is not converged; run mf.kernel() first.")
    if getattr(mf.grids, "coords", None) is None:
        mf.grids.build()

    ao_np = np.asarray(numint.eval_ao(mf.mol, mf.grids.coords, deriv=0))
    ao_deriv1_np = np.asarray(numint.eval_ao(mf.mol, mf.grids.coords, deriv=1))
    weights_np = np.asarray(mf.grids.weights)
    dm_spin_np = np.asarray(mf.make_rdm1())
    if dm_spin_np.ndim == 2:
        dm_spin_np = np.stack([0.5 * dm_spin_np, 0.5 * dm_spin_np], axis=0)
    if dm_spin_np.ndim != 3 or dm_spin_np.shape[0] != 2:
        raise NotImplementedError("Expected unrestricted density matrix shape (2, nao, nao).")

    mo_coeff_np = np.asarray(mf.mo_coeff)
    mo_occ_np = np.asarray(mf.mo_occ)
    mo_energy_np = np.asarray(mf.mo_energy)
    if mo_coeff_np.ndim != 3 or mo_coeff_np.shape[0] != 2:
        raise NotImplementedError(
            "unrestricted_reference_from_pyscf expects unrestricted orbitals with spin axis size 2."
        )
    if mo_occ_np.ndim != 2 or mo_occ_np.shape[0] != 2:
        raise NotImplementedError("Expected unrestricted occupations with shape (2, nmo).")
    if mo_energy_np.ndim != 2 or mo_energy_np.shape[0] != 2:
        raise NotImplementedError("Expected unrestricted orbital energies with shape (2, nmo).")

    nocc_alpha = int(np.count_nonzero(mo_occ_np[0] > 1e-8))
    nocc_beta = int(np.count_nonzero(mo_occ_np[1] > 1e-8))
    jk_backend_norm = str(jk_backend).lower()
    if jk_backend_norm == "df":
        df_factors_np = np.asarray(true_df_factors_from_libcint_mol(mf.mol))
        eri_pair_matrix_np = None
        integral_dtype = df_factors_np.dtype
    elif jk_backend_norm == "full":
        eri_pair_matrix_np = np.asarray(mf.mol.intor("int2e", aosym="s4"))
        df_factors_np = None
        integral_dtype = eri_pair_matrix_np.dtype
    else:
        raise ValueError("jk_backend must be one of {'full', 'df'}.")
    with mf.mol.with_common_orig(_charge_center(mf.mol)):
        dipole_integrals_np = np.asarray(mf.mol.intor_symmetric("int1e_r", comp=3))

    hfx_local = None
    hfx_fxx = None
    hfx_nu = None
    hfx_nu_api = None
    pt2_local = None
    pt2_fock_response = None
    if compute_local_hfx_features:
        use_chunked_hfx_nu = bool(compute_local_hfx_aux) and _use_chunked_hfx_nu(
            mf.mol,
            hfx_nu_storage,
        )
        hfx_result = _local_hfx_features_from_dm(
            mf.mol,
            ao_np,
            (dm_spin_np[0], dm_spin_np[1]),
            np.asarray(mf.grids.coords),
            omega_values=tuple(float(omega) for omega in hfx_omega_values),
            chunk_size=hfx_chunk_size,
            return_nu=bool(compute_local_hfx_aux) and not use_chunked_hfx_nu,
            return_fxx=True,
        )
        if compute_local_hfx_aux and not use_chunked_hfx_nu:
            hfx_local_np, hfx_nu_np, hfx_fxx_np = hfx_result
            hfx_nu = _backend_array(hfx_nu_np, array_backend=array_backend)
        else:
            hfx_local_np, hfx_fxx_np = hfx_result
            if use_chunked_hfx_nu:
                hfx_nu_api = ChunkedHFXNu.from_pyscf_mol(
                    mf.mol,
                    np.asarray(mf.grids.coords),
                    omega_values=tuple(float(omega) for omega in hfx_omega_values),
                    nao=int(ao_np.shape[1]),
                    chunk_size=int(hfx_chunk_size),
                )
        hfx_local = _backend_array(hfx_local_np, array_backend=array_backend)
        hfx_fxx = _backend_array(hfx_fxx_np, array_backend=array_backend)
    if compute_local_pt2_features:
        pt2_local_jax, pt2_fock_response_jax = _local_pt2_feature_from_unrestricted_orbitals(
            jnp.asarray(ao_np),
            jnp.asarray(mo_coeff_np),
            jnp.asarray(mo_occ_np),
            jnp.asarray(mo_energy_np),
            rep_tensor=jnp.zeros((0, 0, 0, 0), dtype=integral_dtype),
            eri_pair_matrix=(
                None if eri_pair_matrix_np is None else jnp.asarray(eri_pair_matrix_np)
            ),
            df_factors=None if df_factors_np is None else jnp.asarray(df_factors_np),
            return_fock_response=True,
        )
        pt2_local = _backend_array(pt2_local_jax, array_backend=array_backend)
        pt2_fock_response = _backend_array(
            pt2_fock_response_jax,
            array_backend=array_backend,
        )

    return UnrestrictedMolecule(
        ao=_backend_array(ao_np, array_backend=array_backend),
        grid=QuadratureGrid(
            weights=_backend_array(weights_np, array_backend=array_backend),
            coords=_backend_array(mf.grids.coords, array_backend=array_backend),
        ),
        dipole_integrals=_backend_array(dipole_integrals_np, array_backend=array_backend),
        rep_tensor=_backend_array(
            np.zeros((0, 0, 0, 0), dtype=integral_dtype)
            if eri_pair_matrix_np is None
            else eri_pair_matrix_np,
            array_backend=array_backend,
        ),
        mo_coeff=_backend_array(mo_coeff_np, array_backend=array_backend),
        mo_occ=_backend_array(mo_occ_np, array_backend=array_backend),
        mo_energy=_backend_array(mo_energy_np, array_backend=array_backend),
        rdm1=_backend_array(dm_spin_np, array_backend=array_backend),
        h1e=_backend_array(mf.get_hcore(), array_backend=array_backend),
        nuclear_repulsion=float(mf.mol.energy_nuc()),
        atom_coords=_backend_array(mf.mol.atom_coords(), array_backend=array_backend),
        atom_charges=_backend_array(mf.mol.atom_charges(), array_backend=array_backend),
        overlap_matrix=_backend_array(mf.get_ovlp(), array_backend=array_backend),
        ao_deriv1=_backend_array(ao_deriv1_np, array_backend=array_backend),
        mf_energy=float(getattr(mf, "e_tot", jnp.nan)),
        exact_exchange_fraction=_hybrid_fraction_from_mf(mf),
        nocc_alpha=nocc_alpha,
        nocc_beta=nocc_beta,
        hfx_omega_values=(
            tuple(float(omega) for omega in hfx_omega_values)
            if compute_local_hfx_features
            else None
        ),
        hfx_local=hfx_local,
        hfx_fxx=hfx_fxx,
        hfx_nu=hfx_nu,
        hfx_nu_api=hfx_nu_api,
        pt2_local=pt2_local,
        pt2_fock_response=pt2_fock_response,
        df_factors=(
            None
            if df_factors_np is None
            else _backend_array(df_factors_np, array_backend=array_backend)
        ),
    )


__all__ = ["restricted_reference_from_pyscf", "unrestricted_reference_from_pyscf"]
