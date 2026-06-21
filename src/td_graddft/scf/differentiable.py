from __future__ import annotations

import copy
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import core as jax_core
from jax.lax import Precision
from jaxtyping import Array, PyTree

from ..data.integrals import build_j_from_eri_pair_matrix, build_jk_from_eri_pair_matrix
from ..df import build_j_from_df, build_jk_from_df
from ..neural_xc.inputs import (
    hfx_nu_source,
)
from .core import _build_density_from_occ, _diagonalize_fock, _orthogonalizer
from .implicit import (
    ImplicitFixedPointConfig,
    implicit_fixed_point_solution,
)
from .xc_energy import xc_energy_and_potential_from_density
from .rks import (
    RKSConfig,
    _build_jk,
    _run_scf_iterations_lax_core,
    _vxc_matrix_from_grid_potential,
)
from .uks import run_unrestricted_scf_scan


_GRID_PAYLOAD_DEPENDENCY_FIELDS = frozenset(
    (
        "ao",
        "ao_deriv1",
        "ao_laplacian",
        "grid",
        "rdm1",
        "mo_coeff",
        "mo_occ",
        "mo_energy",
        "hfx_local",
        "hfx_fxx",
        "hfx_nu",
        "hfx_nu_api",
        "pt2_local",
    )
)


def _supports_neural_xc_grid_payload(molecule: Any) -> bool:
    if is_dataclass(molecule):
        return any(field.name == "neural_xc_grid_payload" for field in fields(molecule))
    return hasattr(molecule, "neural_xc_grid_payload")


def _replace_molecule(molecule: Any, **updates: Any) -> Any:
    if (
        "neural_xc_grid_payload" not in updates
        and _supports_neural_xc_grid_payload(molecule)
        and _GRID_PAYLOAD_DEPENDENCY_FIELDS.intersection(updates)
    ):
        updates = dict(updates)
        updates["neural_xc_grid_payload"] = None
    if is_dataclass(molecule):
        return replace(molecule, **updates)
    molecule_out = copy.copy(molecule)
    for key, value in updates.items():
        setattr(molecule_out, key, value)
    return molecule_out


def _spin_summed_density_matrix(molecule: Any) -> Array:
    density_matrix = jnp.asarray(molecule.rdm1)
    if density_matrix.ndim == 3:
        return density_matrix.sum(axis=0)
    return density_matrix


def _initial_density_matrix(molecule: Any) -> Array:
    cached = getattr(molecule, "scf_initial_density", None)
    if cached is not None:
        return jnp.asarray(cached)
    return _spin_summed_density_matrix(molecule)


def _spin_resolved_density_matrix(molecule: Any) -> Array:
    density_matrix = jnp.asarray(molecule.rdm1)
    if density_matrix.ndim == 2:
        return jnp.stack([0.5 * density_matrix, 0.5 * density_matrix], axis=0)
    if density_matrix.ndim == 3 and density_matrix.shape[0] == 2:
        return density_matrix
    raise ValueError("Expected rdm1 to have shape (nao, nao) or (2, nao, nao).")


def _host_array_if_concrete(value: Any) -> np.ndarray | None:
    arr = jnp.asarray(value)
    if isinstance(arr, jax_core.Tracer):
        return None
    return np.asarray(jax.device_get(arr))


def _is_unrestricted_reference(molecule: Any) -> bool:
    if getattr(molecule, "nocc_alpha", None) is not None or getattr(molecule, "nocc_beta", None) is not None:
        return True

    mo_occ = _host_array_if_concrete(molecule.mo_occ)
    if mo_occ is not None and mo_occ.ndim == 2 and mo_occ.shape[0] == 2 and not np.allclose(mo_occ[0], mo_occ[1]):
        return True
    density = _host_array_if_concrete(molecule.rdm1)
    if density is not None and density.ndim == 3 and density.shape[0] == 2 and not np.allclose(density[0], density[1]):
        return True
    mo_coeff = _host_array_if_concrete(molecule.mo_coeff)
    if mo_coeff is not None and mo_coeff.ndim == 3 and mo_coeff.shape[0] == 2 and not np.allclose(mo_coeff[0], mo_coeff[1]):
        return True
    if getattr(molecule, "nocc", None) is not None:
        return False
    return False


def _is_traceable_pytree(tree: Any) -> bool:
    try:
        jax.tree_util.tree_map(jnp.asarray, tree)
    except (TypeError, ValueError):
        return False
    return True


def _unrestricted_channel(molecule: Any) -> tuple[Array, Array, Array]:
    mo_coeff = jnp.asarray(molecule.mo_coeff)
    mo_occ = jnp.asarray(molecule.mo_occ)
    mo_energy = jnp.asarray(molecule.mo_energy)

    if mo_coeff.ndim == 2:
        mo_coeff = jnp.stack([mo_coeff, mo_coeff], axis=0)
    elif mo_coeff.ndim != 3 or mo_coeff.shape[0] != 2:
        raise ValueError("Expected mo_coeff to have shape (nao, nmo) or (2, nao, nmo).")

    if mo_occ.ndim == 1:
        if float(jnp.max(mo_occ)) <= 1.0 + 1e-6:
            mo_occ = jnp.stack([mo_occ, mo_occ], axis=0)
        else:
            mo_occ = jnp.stack([0.5 * mo_occ, 0.5 * mo_occ], axis=0)
    elif mo_occ.ndim != 2 or mo_occ.shape[0] != 2:
        raise ValueError("Expected mo_occ to have shape (nmo,) or (2, nmo).")

    if mo_energy.ndim == 1:
        mo_energy = jnp.stack([mo_energy, mo_energy], axis=0)
    elif mo_energy.ndim != 2 or mo_energy.shape[0] != 2:
        raise ValueError("Expected mo_energy to have shape (nmo,) or (2, nmo).")

    return mo_coeff, mo_occ, mo_energy


def _initial_density_matrix_spin(molecule: Any) -> Array:
    cached = getattr(molecule, "scf_initial_density", None)
    if cached is not None:
        cached_arr = jnp.asarray(cached)
        if cached_arr.ndim == 3 and cached_arr.shape[0] == 2:
            return cached_arr
    return _spin_resolved_density_matrix(molecule)


def _restricted_channel(molecule: Any) -> tuple[Array, Array]:
    mo_coeff = jnp.asarray(molecule.mo_coeff)
    mo_occ = jnp.asarray(molecule.mo_occ)

    if mo_coeff.ndim == 2:
        return mo_coeff, mo_occ
    if mo_coeff.ndim != 3:
        raise ValueError("Expected mo_coeff to have shape (nao, nmo) or (spin, nao, nmo).")
    if mo_coeff.shape[0] == 1:
        return mo_coeff[0], mo_occ[0]
    if mo_coeff.shape[0] != 2:
        raise NotImplementedError("DifferentiableSCF currently supports restricted references only.")
    return mo_coeff[0], mo_occ[0]


def _target_electron_count(molecule: Any) -> float | None:
    electron_count = getattr(molecule, "electron_count", None)
    if electron_count is not None:
        return float(np.asarray(electron_count))
    nelectron = getattr(molecule, "nelectron", None)
    if nelectron is not None:
        return float(np.asarray(nelectron))
    return None


def _restricted_total_occupations(
    molecule: Any,
    *,
    occupation_tolerance: float,
) -> Array:
    del occupation_tolerance
    mo_occ = jnp.asarray(molecule.mo_occ)
    if mo_occ.ndim == 2:
        if mo_occ.shape[0] == 1:
            occ_total = 2.0 * mo_occ[0]
        elif mo_occ.shape[0] == 2:
            occ_total = mo_occ.sum(axis=0)
        else:
            raise NotImplementedError("DifferentiableSCF currently supports restricted references only.")
    elif mo_occ.ndim == 1:
        occ_total = mo_occ
        target_electrons = _target_electron_count(molecule)
        if (
            target_electrons is not None
            and float(jnp.max(occ_total)) <= 1.0 + 1e-6
            and abs(2.0 * float(jnp.sum(occ_total)) - target_electrons)
            < abs(float(jnp.sum(occ_total)) - target_electrons)
        ):
            occ_total = 2.0 * occ_total
    else:
        raise ValueError("Expected mo_occ to have shape (nmo,) or (spin, nmo).")
    return occ_total


def _restricted_stacked_occupations_from_total(mo_occ_total: Array) -> Array:
    occ = jnp.asarray(mo_occ_total)
    return jnp.stack([0.5 * occ, 0.5 * occ], axis=0)


def _mo_coeff_guess_from_density_matrix(
    density_matrix: Any,
    overlap: Any,
    *,
    orthogonalization_eps: float,
) -> Array:
    dm = jnp.asarray(density_matrix)
    s = jnp.asarray(overlap)
    eigvals, eigvecs = jnp.linalg.eigh(s)
    clipped = jnp.maximum(eigvals, orthogonalization_eps)
    x = eigvecs @ jnp.diag(clipped ** -0.5) @ eigvecs.T
    dm_ortho = 0.5 * (x.T @ dm @ x + (x.T @ dm @ x).T)
    occ_vals, coeff_ortho = jnp.linalg.eigh(dm_ortho)
    order = jnp.argsort(occ_vals)[::-1]
    coeff_ortho = coeff_ortho[:, order]
    return x @ coeff_ortho


def _resolved_xc_object(
    params: PyTree,
    functional: Any,
    molecule: Any,
) -> Any:
    scf_molecule_binder = getattr(functional, "bind_to_molecule_for_scf", None)
    if scf_molecule_binder is not None:
        return scf_molecule_binder(params, molecule)
    molecule_binder = getattr(functional, "bind_to_molecule", None)
    if molecule_binder is not None:
        return molecule_binder(params, molecule)
    binder = getattr(functional, "bind", None)
    if binder is not None:
        return binder(params)
    return functional


def _scf_xc_components(
    params: PyTree,
    functional: Any,
    molecule: Any,
    *,
    functional_dtype: Any,
) -> tuple[Array, Array, Array, Array, str, Array, Array]:
    direct = getattr(functional, "scf_potential_components_and_alpha", None)
    if callable(direct):
        direct_components = direct(params, molecule)
        extra_fock = None
        if len(direct_components) == 4:
            v_rho, v_grad, xc_kind, alpha = direct_components
            v_tau = jnp.zeros_like(jnp.asarray(v_rho))
            v_lapl = jnp.zeros_like(jnp.asarray(v_rho))
        elif len(direct_components) == 6:
            v_rho, v_grad, v_tau, v_lapl, xc_kind, alpha = direct_components
        elif len(direct_components) == 7:
            v_rho, v_grad, v_tau, v_lapl, xc_kind, alpha, extra_fock = direct_components
        else:
            raise ValueError(
                "scf_potential_components_and_alpha must return "
                "(v_rho, v_grad, xc_kind, alpha) or "
                "(v_rho, v_grad, v_tau, v_lapl, xc_kind, alpha) or "
                "(v_rho, v_grad, v_tau, v_lapl, xc_kind, alpha, extra_fock)."
            )
        if extra_fock is None:
            extra_fock_matrix = jnp.zeros(
                (molecule.ao.shape[1], molecule.ao.shape[1]),
                dtype=functional_dtype,
            )
        else:
            extra_fock_matrix = jnp.asarray(extra_fock, dtype=functional_dtype)
            extra_fock_matrix = jnp.nan_to_num(
                extra_fock_matrix,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            extra_fock_matrix = 0.5 * (extra_fock_matrix + extra_fock_matrix.T)
        return (
            jnp.asarray(v_rho, dtype=functional_dtype),
            jnp.asarray(v_grad, dtype=functional_dtype),
            jnp.asarray(v_tau, dtype=functional_dtype),
            jnp.asarray(v_lapl, dtype=functional_dtype),
            str(xc_kind),
            jnp.asarray(alpha, dtype=functional_dtype),
            extra_fock_matrix,
        )

    resolved = _resolved_xc_object(params, functional, molecule)
    v_rho, v_grad, v_tau, v_lapl, xc_kind = _grid_xc_potential_components_from_resolved(
        resolved,
        functional=functional,
        params=params,
        molecule=molecule,
    )
    alpha = jnp.asarray(getattr(resolved, "exact_exchange_fraction", 0.0))
    return (
        jnp.asarray(v_rho, dtype=functional_dtype),
        jnp.asarray(v_grad, dtype=functional_dtype),
        jnp.asarray(v_tau, dtype=functional_dtype),
        jnp.asarray(v_lapl, dtype=functional_dtype),
        str(xc_kind),
        jnp.asarray(alpha, dtype=functional_dtype),
        jnp.zeros((molecule.ao.shape[1], molecule.ao.shape[1]), dtype=functional_dtype),
    )


def _unrestricted_scf_xc_components(
    params: PyTree,
    functional: Any,
    molecule: Any,
    *,
    functional_dtype: Any,
) -> tuple[Array, Array, Array, Array, str, Array, Array, Array]:
    direct = getattr(functional, "unrestricted_scf_potential_components_and_alpha", None)
    if callable(direct):
        components = direct(params, molecule)
    else:
        resolved = _resolved_xc_object(params, functional, molecule)
        contributions_getter = getattr(resolved, "unrestricted_scf_components", None)
        if not callable(contributions_getter):
            raise NotImplementedError(
                "Unrestricted differentiable SCF currently requires the resolved XC object "
                "to expose unrestricted_scf_components(molecule)."
            )
        components = contributions_getter(molecule)
    if len(components) != 8:
        raise ValueError(
            "unrestricted_scf_components must return "
            "(v_rho_a, v_rho_b, v_grad_a, v_grad_b, xc_kind, alpha, extra_fock_a, extra_fock_b)."
        )
    (
        v_rho_a,
        v_rho_b,
        v_grad_a,
        v_grad_b,
        xc_kind,
        alpha,
        extra_fock_a,
        extra_fock_b,
    ) = components
    return (
        jnp.asarray(v_rho_a, dtype=functional_dtype),
        jnp.asarray(v_rho_b, dtype=functional_dtype),
        jnp.asarray(v_grad_a, dtype=functional_dtype),
        jnp.asarray(v_grad_b, dtype=functional_dtype),
        str(xc_kind),
        jnp.asarray(alpha, dtype=functional_dtype),
        jnp.asarray(extra_fock_a, dtype=functional_dtype),
        jnp.asarray(extra_fock_b, dtype=functional_dtype),
    )


def _coulomb_exchange_matrices(
    rep_tensor: Array,
    density: Array,
) -> tuple[Array, Array]:
    rep = jnp.asarray(rep_tensor)
    if rep.ndim == 2 and int(rep.size) > 0:
        return build_jk_from_eri_pair_matrix(rep, density)
    if int(rep.size) == 0:
        raise ValueError(
            "DifferentiableSCF requires full AO ERI or packed AO-pair ERI data "
            "to build Coulomb/exchange matrices."
        )
    return _build_jk(rep, density)


def _coulomb_matrix(
    rep_tensor: Array,
    density: Array,
) -> Array:
    rep = jnp.asarray(rep_tensor)
    if rep.ndim == 2 and int(rep.size) > 0:
        return build_j_from_eri_pair_matrix(rep, density)
    if int(rep.size) == 0:
        raise ValueError(
            "DifferentiableSCF requires full AO ERI or packed AO-pair ERI data "
            "to build Coulomb matrices."
        )
    return jnp.einsum("pqrs,rs->pq", rep, density, precision=Precision.HIGHEST)


def _repulsion_integrals_from_molecule(molecule: Any) -> Array:
    rep_tensor = getattr(molecule, "rep_tensor", None)
    if rep_tensor is not None:
        rep = jnp.asarray(rep_tensor)
        if int(rep.size) > 0:
            return rep
    pair = getattr(molecule, "eri_pair_matrix", None)
    if pair is not None:
        pair_arr = jnp.asarray(pair)
        if int(pair_arr.size) > 0:
            return pair_arr
    if rep_tensor is None:
        raise AttributeError("Molecule-like object must define rep_tensor or eri_pair_matrix.")
    return jnp.asarray(rep_tensor)


def _df_factors_from_molecule(molecule: Any) -> Array | None:
    df_factors = getattr(molecule, "df_factors", None)
    if df_factors is None:
        return None
    factors = jnp.asarray(df_factors)
    if int(factors.size) == 0:
        return None
    return factors


def _scf_problem_arrays(molecule: Any) -> tuple[Array, Array, Array, Array, Array]:
    if getattr(molecule, "h1e", None) is None:
        raise AttributeError("Molecule-like object must define h1e for self-consistent mode.")
    if getattr(molecule, "ao", None) is None or getattr(molecule, "grid", None) is None:
        raise AttributeError(
            "Molecule-like object must define ao and grid.weights for self-consistent mode."
        )
    h1e = jnp.asarray(molecule.h1e)
    overlap = getattr(molecule, "overlap_matrix", None)
    if overlap is None:
        overlap = jnp.eye(h1e.shape[0], dtype=h1e.dtype)
    else:
        overlap = jnp.asarray(overlap)
    return (
        h1e,
        _repulsion_integrals_from_molecule(molecule),
        jnp.asarray(molecule.ao),
        jnp.asarray(molecule.grid.weights),
        overlap,
    )


def _normalize_response_feature_kind(value: Any) -> str:
    if value is None:
        return "LDA"
    kind = str(value).upper()
    if kind in {"LDA", "GGA", "MGGA", "MGGA_LAPL"}:
        return kind
    return "LDA"


def _clip_grid_potential_components(v_rho: Array, v_grad: Array, clip: float) -> tuple[Array, Array]:
    v_rho = jnp.nan_to_num(v_rho, nan=0.0, posinf=clip, neginf=-clip)
    v_grad = jnp.nan_to_num(v_grad, nan=0.0, posinf=clip, neginf=-clip)
    return jnp.clip(v_rho, -clip, clip), jnp.clip(v_grad, -clip, clip)


def _clip_hybrid_alpha(alpha: Array) -> Array:
    return jnp.clip(jnp.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _grid_xc_potential_components_from_resolved(
    resolved: Any,
    *,
    functional: Any,
    params: PyTree,
    molecule: Any,
) -> tuple[Array, Array, Array, Array, str]:
    component_getter = getattr(resolved, "grid_potential_components", None)
    response_kind = _normalize_response_feature_kind(
        getattr(resolved, "response_feature_kind", None)
    )
    if callable(component_getter):
        components = component_getter(molecule)
        if len(components) == 2:
            v_rho, v_grad = components
            v_tau = jnp.zeros_like(jnp.asarray(v_rho))
            v_lapl = jnp.zeros_like(jnp.asarray(v_rho))
        elif len(components) == 3:
            v_rho, v_grad, v_tau = components
            v_lapl = jnp.zeros_like(jnp.asarray(v_rho))
        elif len(components) == 4:
            v_rho, v_grad, v_tau, v_lapl = components
        else:
            raise ValueError(
                "grid_potential_components must return (v_rho, v_grad) or "
                "(v_rho, v_grad, v_tau) or (v_rho, v_grad, v_tau, v_lapl)."
            )
        v_rho = jnp.asarray(v_rho)
        v_grad = jnp.asarray(v_grad, dtype=v_rho.dtype)
        v_tau = jnp.asarray(v_tau, dtype=v_rho.dtype)
        v_lapl = jnp.asarray(v_lapl, dtype=v_rho.dtype)
        if v_grad.ndim == 2 and v_grad.shape == (3, v_rho.shape[0]):
            v_grad = v_grad.T
        if v_grad.ndim != 2 or v_grad.shape[0] != v_rho.shape[0] or v_grad.shape[1] != 3:
            raise ValueError(
                "v_grad must have shape (ngrids, 3) compatible with v_rho."
            )
        if v_tau.shape != v_rho.shape:
            raise ValueError("v_tau must have the same shape as v_rho.")
        if v_lapl.shape != v_rho.shape:
            raise ValueError("v_lapl must have the same shape as v_rho.")
        return v_rho, v_grad, v_tau, v_lapl, response_kind

    grid_potential = getattr(resolved, "grid_potential", None)
    if callable(grid_potential):
        v_rho = jnp.asarray(grid_potential(molecule))
    else:
        density_matrix = _spin_summed_density_matrix(molecule)
        ao = jnp.asarray(molecule.ao)
        total_density = jnp.einsum(
            "rp,pq,rq->r",
            ao,
            density_matrix,
            ao,
            precision=Precision.HIGHEST,
        )
        local_potential = getattr(resolved, "local_potential", None)
        if callable(local_potential):
            v_rho = jnp.asarray(local_potential(total_density))
        else:
            functional_local_potential = getattr(functional, "local_potential", None)
            if functional_local_potential is None:
                raise AttributeError(
                    "The XC functional must expose local_potential(...) or grid_potential(...)."
                )
            v_rho = jnp.asarray(functional_local_potential(params, total_density))
    v_rho = jnp.asarray(v_rho)
    v_grad = jnp.zeros(v_rho.shape + (3,), dtype=v_rho.dtype)
    v_tau = jnp.zeros_like(v_rho)
    v_lapl = jnp.zeros_like(v_rho)
    return v_rho, v_grad, v_tau, v_lapl, "LDA"


def _build_vxc_matrix_from_components(
    *,
    molecule: Any,
    weights: Array,
    v_rho: Array,
    v_grad: Array,
    v_tau: Array,
    v_lapl: Array,
    xc_kind: str,
) -> Array:
    ao = jnp.asarray(molecule.ao)
    ao_deriv1 = getattr(molecule, "ao_deriv1", None)
    ao_laplacian = getattr(molecule, "ao_laplacian", None)
    kind = _normalize_response_feature_kind(xc_kind)
    if ao_deriv1 is None:
        kind = "LDA"
        ao_deriv1_arr = jnp.zeros((4, ao.shape[0], ao.shape[1]), dtype=ao.dtype)
    else:
        ao_deriv1_arr = jnp.asarray(ao_deriv1)
        if ao_deriv1_arr.shape[0] < 4 and kind in {"GGA", "MGGA", "MGGA_LAPL"}:
            kind = "LDA"
    if ao_laplacian is None:
        ao_laplacian_arr = jnp.zeros_like(ao)
        if kind == "MGGA_LAPL":
            kind = "MGGA"
    else:
        ao_laplacian_arr = jnp.asarray(ao_laplacian)
    return _vxc_matrix_from_grid_potential(
        ao=ao,
        ao_deriv1=ao_deriv1_arr,
        ao_laplacian=ao_laplacian_arr,
        weights=jnp.asarray(weights),
        vxc_rho=jnp.asarray(v_rho),
        vxc_grad=jnp.asarray(v_grad),
        vxc_tau=jnp.asarray(v_tau),
        vxc_lapl=jnp.asarray(v_lapl),
        xc_kind=kind,
    )

def _restricted_iteration_molecule(
    molecule: Any,
    *,
    density: Array,
    mo_coeff: Array,
    mo_occ_stacked: Array,
    mo_energy: Array,
    ao: Array,
    hfx_nu: Any = None,
) -> Any:
    del ao
    updates = dict(
        rdm1=jnp.stack([0.5 * density, 0.5 * density], axis=0),
        mo_coeff=jnp.stack([mo_coeff, mo_coeff], axis=0),
        mo_occ=mo_occ_stacked,
        mo_energy=jnp.stack([mo_energy, mo_energy], axis=0),
    )
    nu_source = hfx_nu if hfx_nu is not None else hfx_nu_source(molecule)
    if hasattr(molecule, "hfx_local"):
        if nu_source is not None:
            updates["hfx_local"] = None
        else:
            local_ref = getattr(molecule, "hfx_local", None)
            if local_ref is not None:
                updates["hfx_local"] = jax.lax.stop_gradient(jnp.asarray(local_ref))
    if hasattr(molecule, "hfx_fxx"):
        if nu_source is not None:
            updates["hfx_fxx"] = None
        else:
            fxx_ref = getattr(molecule, "hfx_fxx", None)
            if fxx_ref is not None:
                updates["hfx_fxx"] = jax.lax.stop_gradient(jnp.asarray(fxx_ref))
    return _replace_molecule(molecule, **updates)


def _safe_symmetric_matrix(matrix: Array) -> Array:
    matrix = jnp.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return 0.5 * (matrix + matrix.T)


def _restricted_xc_fock_terms(
    *,
    params: PyTree,
    functional: Any,
    molecule: Any,
    weights: Array,
    functional_dtype: Any,
    vxc_clip: float,
) -> tuple[Array, Array, Array, Array]:
    direct_terms_preference = getattr(functional, "prefer_direct_scf_fock_terms", False)
    if callable(direct_terms_preference):
        direct_terms_preference = direct_terms_preference()
    direct_terms_callback = getattr(functional, "scf_xc_fock_terms", None)
    if bool(direct_terms_preference) and callable(direct_terms_callback):
        vxc_matrix, alpha, vhf_matrix, xc_energy = direct_terms_callback(
            params,
            molecule,
            weights=weights,
            functional_dtype=functional_dtype,
            vxc_clip=vxc_clip,
        )
        return (
            jnp.asarray(vxc_matrix, dtype=functional_dtype),
            _clip_hybrid_alpha(jnp.asarray(alpha, dtype=functional_dtype)),
            jnp.asarray(vhf_matrix, dtype=functional_dtype),
            jnp.asarray(xc_energy, dtype=functional_dtype),
        )

    energy_alpha_callback = getattr(functional, "scf_xc_energy_and_alpha_for_density", None)
    energy_callback = getattr(functional, "scf_xc_energy_for_density", None)
    if callable(energy_alpha_callback) or callable(energy_callback):
        density = _spin_summed_density_matrix(molecule)
        has_aux = callable(energy_alpha_callback)
        extra_fock_getter = getattr(functional, "scf_extra_fock_for_density", None)
        extra_fock = (
            extra_fock_getter(params, molecule, density)
            if callable(extra_fock_getter)
            else None
        )
        result = xc_energy_and_potential_from_density(
            params,
            molecule=molecule,
            density=density,
            xc_energy_fn=energy_alpha_callback if has_aux else energy_callback,
            exact_exchange_fraction=0.0,
            extra_fock_matrix=extra_fock,
            has_aux=has_aux,
        )
        if has_aux:
            alpha = result.aux
        else:
            alpha_getter = getattr(functional, "scf_exact_exchange_fraction", None)
            alpha = (
                alpha_getter(params, molecule, density)
                if callable(alpha_getter)
                else jnp.asarray(0.0, dtype=functional_dtype)
            )
        return (
            jnp.asarray(result.vxc_matrix, dtype=functional_dtype),
            _clip_hybrid_alpha(jnp.asarray(alpha, dtype=functional_dtype)),
            jnp.asarray(result.extra_fock_matrix, dtype=functional_dtype),
            jnp.asarray(result.xc_energy, dtype=functional_dtype),
        )

    vxc_rho, vxc_grad, vxc_tau, vxc_lapl, xc_kind, alpha, vhf_matrix = _scf_xc_components(
        params,
        functional,
        molecule,
        functional_dtype=functional_dtype,
    )
    vxc_rho, vxc_grad = _clip_grid_potential_components(vxc_rho, vxc_grad, vxc_clip)
    vxc_matrix = _build_vxc_matrix_from_components(
        molecule=molecule,
        weights=weights,
        v_rho=vxc_rho,
        v_grad=vxc_grad,
        v_tau=vxc_tau,
        v_lapl=vxc_lapl,
        xc_kind=xc_kind,
    )
    energy_from_molecule = getattr(functional, "energy_from_molecule", None)
    if callable(energy_from_molecule):
        xc_energy = energy_from_molecule(params, molecule)
    else:
        xc_energy = jnp.asarray(0.0, dtype=functional_dtype)
    return vxc_matrix, _clip_hybrid_alpha(alpha), vhf_matrix, jnp.asarray(xc_energy, dtype=functional_dtype)


@dataclass(frozen=True)
class DifferentiableSCFConfig:
    """Configuration for fixed-density / self-consistent differentiable SCF."""

    mode: Literal["fixed_density", "self_consistent"] = "fixed_density"
    gradient_mode: Literal["expl", "impl"] = "expl"
    max_cycle: int = 12
    damping: float = 0.25
    level_shift: float = 0.0
    conv_tol_energy: float | None = None
    convergence_metric: Literal["energy_and_residual", "energy"] = "energy_and_residual"
    occupation_tolerance: float = 1e-8
    conv_tol_density: float = 1e-8
    orthogonalization_eps: float = 1e-10
    eigenvalue_jitter: float = 1e-8
    vxc_clip: float = 20.0
    iterate_selection: Literal["final", "best_rms", "first_converged"] = "final"
    require_converged_iterates: bool = False
    implicit_diff_max_iter: int = 6
    implicit_diff_step_size: float = 0.2
    implicit_diff_clip: float = 1e4
    implicit_diff_solver: Literal["normal_cg", "gmres", "bicgstab"] = "gmres"
    implicit_diff_tolerance: float = 1e-6
    implicit_diff_regularization: float = 0.0
    implicit_diff_restart: int = 12

    def __post_init__(self) -> None:
        _valid_gradient_modes = {"expl", "impl"}
        if self.gradient_mode not in _valid_gradient_modes:
            raise ValueError(
                f"gradient_mode must be one of {_valid_gradient_modes}, "
                f"got {self.gradient_mode!r}."
            )
        _valid_convergence_metrics = {"energy_and_residual", "energy"}
        if self.convergence_metric not in _valid_convergence_metrics:
            raise ValueError(
                "convergence_metric must be one of "
                f"{_valid_convergence_metrics}, got {self.convergence_metric!r}."
            )

    def energy_convergence_tolerance(self) -> float:
        if self.conv_tol_energy is not None:
            return float(self.conv_tol_energy)
        return float(self.conv_tol_density) ** 2

@dataclass(frozen=True)
class DifferentiableSCFInfo:
    mode: str
    converged: Array
    cycles: Array
    final_rms_density: Array
    rms_density_history: Array
    selected_cycle: Array
    selected_rms_density: Array
    best_cycle: Array
    best_rms_density: Array


@dataclass(frozen=True)
class _RestrictedSCFContext:
    molecule: Any
    xc_functional: Any
    h1e: Array
    rep_tensor: Array
    df_factors: Array | None
    ao: Array
    weights: Array
    overlap: Array
    x: Array
    mo_occ_total: Array
    mo_occ_stacked: Array
    hfx_nu: Any
    uses_explicit_hfx_fock: bool


@dataclass(frozen=True)
class _RestrictedSCFProblem:
    ctx: _RestrictedSCFContext
    density0: Array
    mo_coeff0: Array
    mo_energy0: Array
    energy_and_fock_builder: Any
    fock0: Array
    j0: Array
    k0: Array


class DifferentiableSCF:
    """Differentiable SCF wrapper with fixed-density and self-consistent modes."""

    def __init__(self, config: DifferentiableSCFConfig | None = None):
        self.config = DifferentiableSCFConfig() if config is None else config

    def _with_neural_xc_grid_payload(self, molecule: Any, xc_functional: Any) -> Any:
        if not _supports_neural_xc_grid_payload(molecule):
            return molecule
        payload_builder = getattr(xc_functional, "restricted_grid_payload_for_molecule", None)
        if not callable(payload_builder):
            return molecule
        return _replace_molecule(molecule, neural_xc_grid_payload=payload_builder(molecule))

    def _fixed_cycle_info(self, mode: str, rms_history: Array) -> DifferentiableSCFInfo:
        best_idx = jnp.argmin(rms_history)
        selected_idx = jnp.asarray(int(self.config.max_cycle) - 1, dtype=jnp.int32)
        cycles = jnp.asarray(int(self.config.max_cycle), dtype=jnp.int32)
        converged = jnp.any(rms_history < self.config.conv_tol_density)
        return DifferentiableSCFInfo(
            mode=mode,
            converged=converged,
            cycles=cycles,
            final_rms_density=rms_history[-1],
            rms_density_history=rms_history,
            selected_cycle=cycles,
            selected_rms_density=rms_history[selected_idx],
            best_cycle=best_idx + 1,
            best_rms_density=rms_history[best_idx],
        )

    def _rks_loop_info(
        self,
        mode: str,
        *,
        converged: Array,
        cycles: Array,
        rms_history: Array,
    ) -> DifferentiableSCFInfo:
        max_idx = jnp.asarray(int(self.config.max_cycle) - 1, dtype=jnp.int32)
        final_idx = jnp.clip(jnp.asarray(cycles, dtype=jnp.int32) - 1, 0, max_idx)
        selected_idx = final_idx
        valid_history = jnp.arange(int(self.config.max_cycle), dtype=jnp.int32) <= final_idx
        masked_history = jnp.where(
            valid_history,
            rms_history,
            jnp.asarray(jnp.inf, dtype=rms_history.dtype),
        )
        best_idx = jnp.argmin(masked_history)
        first_converged_history = masked_history < jnp.asarray(
            self.config.conv_tol_density,
            dtype=rms_history.dtype,
        )
        first_converged_idx = jnp.argmax(first_converged_history.astype(jnp.int32))
        has_density_converged = jnp.any(first_converged_history)
        if self.config.iterate_selection == "best_rms":
            selected_idx = best_idx
        elif self.config.iterate_selection == "first_converged":
            selected_idx = jnp.where(has_density_converged, first_converged_idx, best_idx)
        selected_idx = jnp.where(jnp.asarray(converged), selected_idx, final_idx)
        return DifferentiableSCFInfo(
            mode=mode,
            converged=converged,
            cycles=cycles,
            final_rms_density=rms_history[final_idx],
            rms_density_history=rms_history,
            selected_cycle=selected_idx + 1,
            selected_rms_density=rms_history[selected_idx],
            best_cycle=best_idx + 1,
            best_rms_density=rms_history[best_idx],
        )

    def _select_restricted_scf_iterate(
        self,
        *,
        converged: Array,
        cycles: Array,
        density_final: Array,
        mo_coeff_final: Array,
        mo_energy_final: Array,
        density_history: Array,
        mo_coeff_history: Array,
        mo_energy_history: Array,
        rms_history: Array,
    ) -> tuple[Array, Array, Array]:
        if self.config.iterate_selection == "final":
            return density_final, mo_coeff_final, mo_energy_final
        info = self._rks_loop_info(
            "self_consistent",
            converged=converged,
            cycles=cycles,
            rms_history=rms_history,
        )
        selected_idx = jnp.clip(
            jnp.asarray(info.selected_cycle, dtype=jnp.int32) - 1,
            0,
            int(self.config.max_cycle) - 1,
        )
        return (
            density_history[selected_idx],
            mo_coeff_history[selected_idx],
            mo_energy_history[selected_idx],
        )

    def __call__(
        self,
        molecule: Any,
        xc_functional: Any,
        xc_params: PyTree,
    ) -> Any:
        molecule_out, _ = self.run(molecule, xc_functional, xc_params)
        return molecule_out

    def run(
        self,
        molecule: Any,
        xc_functional: Any,
        xc_params: PyTree,
    ) -> tuple[Any, DifferentiableSCFInfo]:
        if self.config.mode == "fixed_density":
            return self._single_step(molecule)
        if self.config.mode == "self_consistent":
            return self._full_scf(molecule, xc_functional, xc_params)
        raise ValueError(f"Unsupported DifferentiableSCF mode: {self.config.mode!r}")

    def _single_step(self, molecule: Any) -> tuple[Any, DifferentiableSCFInfo]:
        rdm1 = jax.lax.stop_gradient(jnp.asarray(molecule.rdm1))
        fixed = _replace_molecule(molecule, rdm1=rdm1)
        info = DifferentiableSCFInfo(
            mode="fixed_density",
            converged=jnp.asarray(True),
            cycles=jnp.asarray(0),
            final_rms_density=jnp.asarray(0.0),
            rms_density_history=jnp.zeros((0,)),
            selected_cycle=jnp.asarray(0),
            selected_rms_density=jnp.asarray(0.0),
            best_cycle=jnp.asarray(0),
            best_rms_density=jnp.asarray(0.0),
        )
        return fixed, info

    def _implicit_input_state_info(
        self,
        density: Array,
    ) -> DifferentiableSCFInfo:
        dtype = jnp.asarray(density).dtype
        zero_float = jnp.asarray(0.0, dtype=dtype)
        zero_int = jnp.asarray(0, dtype=jnp.int32)
        return DifferentiableSCFInfo(
            mode="self_consistent_implicit_input_state",
            converged=jnp.asarray(True),
            cycles=zero_int,
            final_rms_density=zero_float,
            rms_density_history=jnp.zeros((0,), dtype=dtype),
            selected_cycle=zero_int,
            selected_rms_density=zero_float,
            best_cycle=zero_int,
            best_rms_density=zero_float,
        )

    def _restricted_scf_context(
        self,
        molecule: Any,
        xc_functional: Any,
    ) -> _RestrictedSCFContext:
        h1e, rep_tensor, ao, weights, overlap = _scf_problem_arrays(molecule)
        x = _orthogonalizer(overlap, self.config.orthogonalization_eps)
        mo_occ_total = _restricted_total_occupations(
            molecule,
            occupation_tolerance=self.config.occupation_tolerance,
        )
        mo_occ_total = jnp.asarray(mo_occ_total, dtype=h1e.dtype)
        explicit_hfx_probe = getattr(xc_functional, "uses_explicit_hfx_fock_for_scf", None)
        uses_explicit_hfx_fock = (
            bool(explicit_hfx_probe(molecule))
            if callable(explicit_hfx_probe)
            else False
        )
        return _RestrictedSCFContext(
            molecule=molecule,
            xc_functional=xc_functional,
            h1e=h1e,
            rep_tensor=rep_tensor,
            df_factors=_df_factors_from_molecule(molecule),
            ao=ao,
            weights=weights,
            overlap=overlap,
            x=x,
            mo_occ_total=mo_occ_total,
            mo_occ_stacked=_restricted_stacked_occupations_from_total(mo_occ_total),
            hfx_nu=hfx_nu_source(molecule),
            uses_explicit_hfx_fock=uses_explicit_hfx_fock,
        )

    def _restricted_response_jk(
        self,
        ctx: _RestrictedSCFContext,
        density: Array,
        *,
        with_k: bool = True,
    ) -> tuple[Array, Array]:
        if ctx.df_factors is not None:
            if not bool(with_k):
                j_mat = build_j_from_df(ctx.df_factors, density)
                return j_mat, jnp.zeros_like(j_mat)
            return build_jk_from_df(ctx.df_factors, density)
        if not bool(with_k):
            j_mat = _coulomb_matrix(ctx.rep_tensor, density)
            return j_mat, jnp.zeros_like(j_mat)
        return _coulomb_exchange_matrices(ctx.rep_tensor, density)

    def _restricted_xc_terms_from_density(
        self,
        ctx: _RestrictedSCFContext,
        params: PyTree,
        density: Array,
        mo_coeff: Array,
        mo_energy: Array,
    ) -> tuple[Array, Array, Array]:
        molecule_iter = _restricted_iteration_molecule(
            molecule=ctx.molecule,
            density=density,
            mo_coeff=mo_coeff,
            mo_occ_stacked=ctx.mo_occ_stacked,
            mo_energy=mo_energy,
            ao=ctx.ao,
            hfx_nu=ctx.hfx_nu,
        )
        return _restricted_xc_fock_terms(
            params=params,
            functional=ctx.xc_functional,
            molecule=molecule_iter,
            weights=ctx.weights,
            functional_dtype=ctx.h1e.dtype,
            vxc_clip=self.config.vxc_clip,
        )

    def _restricted_energy_and_fock_from_density(
        self,
        ctx: _RestrictedSCFContext,
        params: PyTree,
        density: Array,
        mo_coeff: Array,
        mo_energy: Array,
        *,
        with_k: bool = True,
    ) -> tuple[Array, Array, Array, Array]:
        j_mat, k_mat = self._restricted_response_jk(
            ctx,
            density,
            with_k=with_k,
        )
        vxc_matrix, alpha, vhf_matrix, xc_energy = self._restricted_xc_terms_from_density(
            ctx,
            params,
            density,
            mo_coeff,
            mo_energy,
        )
        fock = ctx.h1e + j_mat - 0.5 * alpha * k_mat + vxc_matrix + vhf_matrix
        one_body = jnp.einsum("ij,ij->", density, ctx.h1e, precision=Precision.HIGHEST)
        coulomb = 0.5 * jnp.einsum("ij,ij->", density, j_mat, precision=Precision.HIGHEST)
        exact_exchange = -0.25 * alpha * jnp.einsum(
            "ij,ij->",
            density,
            k_mat,
            precision=Precision.HIGHEST,
        )
        total = one_body + coulomb + exact_exchange + xc_energy + jnp.asarray(
            getattr(ctx.molecule, "nuclear_repulsion", 0.0),
            dtype=ctx.h1e.dtype,
        )
        return total, xc_energy, _safe_symmetric_matrix(fock), alpha, j_mat, k_mat

    def _restricted_fock_from_density(
        self,
        ctx: _RestrictedSCFContext,
        params: PyTree,
        density: Array,
        mo_coeff: Array,
        mo_energy: Array,
        *,
        with_k: bool = True,
    ) -> tuple[Array, Array, Array, Array]:
        _total, _xc_energy, fock, alpha, j_mat, k_mat = self._restricted_energy_and_fock_from_density(
            ctx,
            params,
            density,
            mo_coeff,
            mo_energy,
            with_k=with_k,
        )
        return fock, alpha, j_mat, k_mat

    def _restricted_scf_problem(
        self,
        molecule: Any,
        xc_functional: Any,
        xc_params: PyTree,
    ) -> _RestrictedSCFProblem:
        ctx = self._restricted_scf_context(molecule, xc_functional)
        density0 = _initial_density_matrix(molecule)
        cached_initial_density = getattr(molecule, "scf_initial_density", None)
        if cached_initial_density is None:
            mo_coeff0, _ = _restricted_channel(molecule)
        else:
            mo_coeff0 = _mo_coeff_guess_from_density_matrix(
                density0,
                ctx.overlap,
                orthogonalization_eps=self.config.orthogonalization_eps,
            )
        mo_energy_raw = jnp.asarray(molecule.mo_energy)
        mo_energy0 = mo_energy_raw[0] if mo_energy_raw.ndim == 2 else mo_energy_raw
        nmo = int(mo_coeff0.shape[-1])
        if ctx.mo_occ_total.ndim != 1 or int(ctx.mo_occ_total.shape[0]) != nmo:
            raise ValueError("Restricted occupation vector must have shape (nmo,) in self-consistent mode.")

        def _energy_and_fock_builder(
            density: Array,
            mo_coeff_ref: Array,
            _mo_occ_ref: Array,
            mo_energy_ref: Array,
            _density_last: Array | None,
            _j_last: Array | None,
            _k_last: Array | None,
        ) -> tuple[Array, Array, Array, Array, Array]:
            total, xc_energy, fock, _alpha, j_mat, k_mat = self._restricted_energy_and_fock_from_density(
                ctx,
                xc_params,
                density,
                mo_coeff_ref,
                mo_energy_ref,
            )
            return total, xc_energy, fock, j_mat, k_mat

        _, _, fock0, j0, k0 = _energy_and_fock_builder(
            density0,
            mo_coeff0,
            ctx.mo_occ_total,
            mo_energy0,
            None,
            None,
            None,
        )
        return _RestrictedSCFProblem(
            ctx=ctx,
            density0=density0,
            mo_coeff0=mo_coeff0,
            mo_energy0=mo_energy0,
            energy_and_fock_builder=_energy_and_fock_builder,
            fock0=fock0,
            j0=j0,
            k0=k0,
        )

    def _restricted_molecule_from_total_density(
        self,
        molecule: Any,
        *,
        density: Array,
        mo_coeff: Array,
        mo_energy: Array,
        mo_occ_stacked: Array,
    ) -> Any:
        updates = {
            "rdm1": jnp.stack([0.5 * density, 0.5 * density], axis=0),
            "mo_coeff": jnp.stack([mo_coeff, mo_coeff], axis=0),
            "mo_occ": mo_occ_stacked,
            "mo_energy": jnp.stack([mo_energy, mo_energy], axis=0),
        }
        if is_dataclass(molecule):
            field_names = getattr(molecule, "__dataclass_fields__", {})
            updates.update(
                {
                    name: None
                    for name in (
                        "eri_ovov",
                        "eri_ovvo",
                        "eri_oovv",
                        "pt2_local",
                    )
                    if name in field_names
                }
            )
        else:
            updates.update(
                {
                    "eri_ovov": None,
                    "eri_ovvo": None,
                    "eri_oovv": None,
                    "pt2_local": None,
                }
            )
        return _replace_molecule(molecule, **updates)

    def _run_restricted_scf_core(
        self,
        problem: _RestrictedSCFProblem,
        *,
        conv_tol: float,
        force_density_damping: bool = False,
        density_convergence_tol: float | None = None,
    ) -> tuple[Any, ...]:
        rks_cfg = RKSConfig(
            xc_spec="hf",
            max_cycle=self.config.max_cycle,
            conv_tol=conv_tol,
            conv_tol_density=self.config.conv_tol_density,
            damping=self.config.damping,
            level_shift=self.config.level_shift,
            orthogonalization_eps=self.config.orthogonalization_eps,
            convergence_metric=self.config.convergence_metric,
        )
        return _run_scf_iterations_lax_core(
            h=problem.ctx.h1e,
            s=problem.ctx.overlap,
            x=problem.ctx.x,
            energy_and_fock_builder=problem.energy_and_fock_builder,
            cfg=rks_cfg,
            mo_occ_fixed=problem.ctx.mo_occ_total,
            diis_basis=problem.mo_coeff0,
            skip_first_fock_damping=True,
            density=problem.density0,
            mo_coeff=problem.mo_coeff0,
            mo_occ=problem.ctx.mo_occ_total,
            mo_energy=problem.mo_energy0,
            raw_fock=problem.fock0,
            j_mat=problem.j0,
            k_mat=problem.k0,
            force_density_damping=force_density_damping,
            density_convergence_tol=density_convergence_tol,
            eigenvalue_jitter=self.config.eigenvalue_jitter,
        )

    def _full_scf(
        self,
        molecule: Any,
        xc_functional: Any,
        xc_params: PyTree,
    ) -> tuple[Any, DifferentiableSCFInfo]:
        if _is_unrestricted_reference(molecule):
            if self.config.gradient_mode == "impl":
                return self._full_scf_implicit_commutator_unrestricted(
                    molecule,
                    xc_functional,
                    xc_params,
                )
            return self._full_scf_unrestricted(molecule, xc_functional, xc_params)
        if self.config.gradient_mode == "impl":
            return self._full_scf_implicit_fixed_point(molecule, xc_functional, xc_params)
        problem = self._restricted_scf_problem(molecule, xc_functional, xc_params)
        (
            converged,
            cycles,
            density_final,
            mo_coeff_final,
            mo_energy_final,
            _energy,
            _xc_energy,
            _fock_final,
            _j_final,
            _k_final,
            density_history,
            mo_coeff_history,
            mo_energy_history,
            rms_history,
        ) = self._run_restricted_scf_core(
            problem,
            conv_tol=self.config.energy_convergence_tolerance(),
        )
        density_final, mo_coeff_final, mo_energy_final = self._select_restricted_scf_iterate(
            converged=converged,
            cycles=cycles,
            density_final=density_final,
            mo_coeff_final=mo_coeff_final,
            mo_energy_final=mo_energy_final,
            density_history=density_history,
            mo_coeff_history=mo_coeff_history,
            mo_energy_history=mo_energy_history,
            rms_history=rms_history,
        )

        molecule_final = self._restricted_molecule_from_total_density(
            molecule,
            density=density_final,
            mo_coeff=mo_coeff_final,
            mo_energy=mo_energy_final,
            mo_occ_stacked=problem.ctx.mo_occ_stacked,
        )
        molecule_final = self._with_neural_xc_grid_payload(molecule_final, xc_functional)
        return molecule_final, self._rks_loop_info(
            "self_consistent",
            converged=converged,
            cycles=cycles,
            rms_history=rms_history,
        )

    def _full_scf_unrestricted(
        self,
        molecule: Any,
        xc_functional: Any,
        xc_params: PyTree,
    ) -> tuple[Any, DifferentiableSCFInfo]:
        h1e, rep_tensor, ao, weights, overlap = _scf_problem_arrays(molecule)

        density_spin0 = _initial_density_matrix_spin(molecule)
        cached_initial_density = getattr(molecule, "scf_initial_density", None)
        mo_coeff_spin0, mo_occ_spin_fixed, mo_energy_spin0 = _unrestricted_channel(molecule)
        if cached_initial_density is not None:
            cached_initial_density_arr = jnp.asarray(cached_initial_density)
            if cached_initial_density_arr.ndim == 3 and cached_initial_density_arr.shape[0] == 2:
                mo_coeff_spin0 = jax.vmap(
                    lambda density_spin: _mo_coeff_guess_from_density_matrix(
                        density_spin,
                        overlap,
                        orthogonalization_eps=self.config.orthogonalization_eps,
                    )
                )(cached_initial_density_arr)

        if mo_occ_spin_fixed.ndim != 2 or int(mo_occ_spin_fixed.shape[0]) != 2:
            raise ValueError(
                "Unrestricted self-consistent mode expects spin-resolved occupations with shape (2, nmo)."
            )
        mo_occ_spin_fixed = jnp.asarray(mo_occ_spin_fixed, dtype=h1e.dtype)

        def fock_builder(
            density_spin: Array,
            mo_coeff_ref_spin: Array,
            mo_energy_ref_spin: Array,
        ) -> tuple[Array, Array, Array]:
            updates = dict(
                rdm1=density_spin,
                mo_coeff=mo_coeff_ref_spin,
                mo_occ=mo_occ_spin_fixed,
                mo_energy=mo_energy_ref_spin,
            )
            molecule_iter = _replace_molecule(molecule, **updates)

            density_total = density_spin.sum(axis=0)
            j_mat, _ = _coulomb_exchange_matrices(rep_tensor, density_total)
            _, k_alpha = _coulomb_exchange_matrices(rep_tensor, density_spin[0])
            _, k_beta = _coulomb_exchange_matrices(rep_tensor, density_spin[1])
            (
                vxc_rho_a,
                vxc_rho_b,
                vxc_grad_a,
                vxc_grad_b,
                xc_kind,
                alpha,
                extra_fock_a,
                extra_fock_b,
            ) = _unrestricted_scf_xc_components(
                xc_params,
                xc_functional,
                molecule_iter,
                functional_dtype=h1e.dtype,
            )
            vxc_rho_a, vxc_grad_a = _clip_grid_potential_components(
                vxc_rho_a,
                vxc_grad_a,
                self.config.vxc_clip,
            )
            vxc_rho_b, vxc_grad_b = _clip_grid_potential_components(
                vxc_rho_b,
                vxc_grad_b,
                self.config.vxc_clip,
            )

            zero_aux_a = jnp.zeros_like(vxc_rho_a)
            zero_aux_b = jnp.zeros_like(vxc_rho_b)
            vxc_matrix_a = _build_vxc_matrix_from_components(
                molecule=molecule_iter,
                weights=weights,
                v_rho=vxc_rho_a,
                v_grad=vxc_grad_a,
                v_tau=zero_aux_a,
                v_lapl=zero_aux_a,
                xc_kind=xc_kind,
            )
            vxc_matrix_b = _build_vxc_matrix_from_components(
                molecule=molecule_iter,
                weights=weights,
                v_rho=vxc_rho_b,
                v_grad=vxc_grad_b,
                v_tau=zero_aux_b,
                v_lapl=zero_aux_b,
                xc_kind=xc_kind,
            )
            alpha = _clip_hybrid_alpha(alpha)
            fock_alpha = h1e + j_mat - alpha * k_alpha + extra_fock_a + vxc_matrix_a
            fock_beta = h1e + j_mat - alpha * k_beta + extra_fock_b + vxc_matrix_b
            fock_alpha = jnp.nan_to_num(fock_alpha, nan=0.0, posinf=0.0, neginf=0.0)
            fock_beta = jnp.nan_to_num(fock_beta, nan=0.0, posinf=0.0, neginf=0.0)
            fock_spin = jnp.stack(
                [
                    0.5 * (fock_alpha + fock_alpha.T),
                    0.5 * (fock_beta + fock_beta.T),
                ],
                axis=0,
            )
            one_body = jnp.einsum("spq,pq->", density_spin, h1e, optimize=True)
            coulomb = 0.5 * jnp.einsum("pq,pq->", density_total, j_mat, optimize=True)
            exchange = -0.5 * alpha * (
                jnp.einsum("pq,pq->", density_spin[0], k_alpha, optimize=True)
                + jnp.einsum("pq,pq->", density_spin[1], k_beta, optimize=True)
            )
            energy_from_molecule = getattr(xc_functional, "energy_from_molecule", None)
            xc_energy = (
                energy_from_molecule(xc_params, molecule_iter)
                if callable(energy_from_molecule)
                else jnp.asarray(0.0, dtype=h1e.dtype)
            )
            total_energy = (
                one_body
                + coulomb
                + exchange
                + jnp.asarray(xc_energy, dtype=h1e.dtype)
                + jnp.asarray(getattr(molecule, "nuclear_repulsion", 0.0), dtype=h1e.dtype)
            )
            return fock_spin, fock_spin, total_energy

        (
            density_spin_final,
            mo_coeff_spin_final,
            mo_energy_spin_final,
            _raw_fock_spin_final,
            converged,
            cycles,
            rms_history,
            selected_cycle,
            best_cycle,
            selected_rms,
            best_rms,
        ) = run_unrestricted_scf_scan(
            fock_builder=fock_builder,
            density_spin=density_spin0,
            mo_coeff_spin=mo_coeff_spin0,
            mo_occ_spin=mo_occ_spin_fixed,
            mo_energy_spin=mo_energy_spin0,
            overlap=overlap,
            max_cycle=int(self.config.max_cycle),
            damping=float(self.config.damping),
            conv_tol=float(self.config.energy_convergence_tolerance()),
            conv_tol_density=float(self.config.conv_tol_density),
            orthogonalization_eps=float(self.config.orthogonalization_eps),
            convergence_metric=str(self.config.convergence_metric),
            eigenvalue_jitter=float(self.config.eigenvalue_jitter),
            iterate_selection=str(self.config.iterate_selection),
        )
        molecule_final = _replace_molecule(
            molecule,
            rdm1=density_spin_final,
            mo_coeff=mo_coeff_spin_final,
            mo_occ=mo_occ_spin_fixed,
            mo_energy=mo_energy_spin_final,
        )
        return molecule_final, DifferentiableSCFInfo(
            mode="self_consistent",
            converged=converged,
            cycles=cycles,
            final_rms_density=rms_history[-1],
            rms_density_history=rms_history,
            selected_cycle=selected_cycle,
            selected_rms_density=selected_rms,
            best_cycle=best_cycle,
            best_rms_density=best_rms,
        )

    def _full_scf_implicit_commutator_unrestricted(
        self,
        molecule: Any,
        xc_functional: Any,
        xc_params: PyTree,
    ) -> tuple[Any, DifferentiableSCFInfo]:
        forward_params = jax.tree_util.tree_map(jax.lax.stop_gradient, xc_params)
        forward_molecule, forward_info = self._full_scf_unrestricted(
            molecule,
            xc_functional,
            forward_params,
        )

        density_star_spin = _spin_resolved_density_matrix(forward_molecule)
        density_star_spin = jnp.nan_to_num(
            density_star_spin,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        mo_coeff_spin_ref, mo_occ_spin_fixed, mo_energy_spin_ref = _unrestricted_channel(
            forward_molecule
        )

        h1e, rep_tensor, _ao, weights, overlap = _scf_problem_arrays(molecule)
        fixed_point_args = None
        if _is_traceable_pytree(molecule):
            fixed_point_args = (
                molecule,
                mo_coeff_spin_ref,
                mo_occ_spin_fixed,
                mo_energy_spin_ref,
            )

        def _unrestricted_args(
            args: tuple[Any, ...] | None,
        ) -> tuple[Any, Array, Array, Array, Array, Array, Array, Array]:
            if args is None:
                return (
                    molecule,
                    h1e,
                    rep_tensor,
                    weights,
                    overlap,
                    mo_coeff_spin_ref,
                    mo_occ_spin_fixed,
                    mo_energy_spin_ref,
                )
            molecule_arg, coeff_ref_arg, occ_fixed_arg, energy_ref_arg = args
            h1e_arg, rep_tensor_arg, _ao_arg, weights_arg, overlap_arg = _scf_problem_arrays(
                molecule_arg
            )
            return (
                molecule_arg,
                h1e_arg,
                rep_tensor_arg,
                weights_arg,
                overlap_arg,
                coeff_ref_arg,
                occ_fixed_arg,
                energy_ref_arg,
            )

        def _raw_fock_from_density(
            density_spin: Array,
            params_local: PyTree,
            args: tuple[Any, ...] | None = fixed_point_args,
        ) -> Array:
            (
                molecule_base,
                h1e_local,
                rep_tensor_local,
                weights_local,
                _overlap_local,
                mo_coeff_ref_local,
                mo_occ_fixed_local,
                mo_energy_ref_local,
            ) = _unrestricted_args(args)
            density_spin = jnp.nan_to_num(density_spin, nan=0.0, posinf=0.0, neginf=0.0)
            updates = dict(
                rdm1=density_spin,
                mo_coeff=mo_coeff_ref_local,
                mo_occ=mo_occ_fixed_local,
                mo_energy=mo_energy_ref_local,
            )
            molecule_iter = _replace_molecule(molecule_base, **updates)
            density_total = density_spin.sum(axis=0)
            j_mat, _ = _coulomb_exchange_matrices(rep_tensor_local, density_total)
            _, k_alpha = _coulomb_exchange_matrices(rep_tensor_local, density_spin[0])
            _, k_beta = _coulomb_exchange_matrices(rep_tensor_local, density_spin[1])
            (
                vxc_rho_a,
                vxc_rho_b,
                vxc_grad_a,
                vxc_grad_b,
                xc_kind,
                alpha,
                extra_fock_a,
                extra_fock_b,
            ) = _unrestricted_scf_xc_components(
                params_local,
                xc_functional,
                molecule_iter,
                functional_dtype=h1e_local.dtype,
            )
            vxc_rho_a, vxc_grad_a = _clip_grid_potential_components(
                vxc_rho_a,
                vxc_grad_a,
                self.config.vxc_clip,
            )
            vxc_rho_b, vxc_grad_b = _clip_grid_potential_components(
                vxc_rho_b,
                vxc_grad_b,
                self.config.vxc_clip,
            )

            zero_aux_a = jnp.zeros_like(vxc_rho_a)
            zero_aux_b = jnp.zeros_like(vxc_rho_b)
            vxc_matrix_a = _build_vxc_matrix_from_components(
                molecule=molecule_iter,
                weights=weights_local,
                v_rho=vxc_rho_a,
                v_grad=vxc_grad_a,
                v_tau=zero_aux_a,
                v_lapl=zero_aux_a,
                xc_kind=xc_kind,
            )
            vxc_matrix_b = _build_vxc_matrix_from_components(
                molecule=molecule_iter,
                weights=weights_local,
                v_rho=vxc_rho_b,
                v_grad=vxc_grad_b,
                v_tau=zero_aux_b,
                v_lapl=zero_aux_b,
                xc_kind=xc_kind,
            )
            alpha = _clip_hybrid_alpha(alpha)
            fock_alpha = h1e_local + j_mat - alpha * k_alpha + extra_fock_a + vxc_matrix_a
            fock_beta = h1e_local + j_mat - alpha * k_beta + extra_fock_b + vxc_matrix_b
            fock_spin = jnp.stack(
                [
                    0.5
                    * (
                        jnp.nan_to_num(fock_alpha, nan=0.0, posinf=0.0, neginf=0.0)
                        + jnp.nan_to_num(fock_alpha.T, nan=0.0, posinf=0.0, neginf=0.0)
                    ),
                    0.5
                    * (
                        jnp.nan_to_num(fock_beta, nan=0.0, posinf=0.0, neginf=0.0)
                        + jnp.nan_to_num(fock_beta.T, nan=0.0, posinf=0.0, neginf=0.0)
                    ),
                ],
                axis=0,
            )
            return fock_spin

        def _residual(
            density_spin: Array,
            params_local: PyTree,
            args: tuple[Any, ...] | None = fixed_point_args,
        ) -> Array:
            _molecule_base, _h1e_local, _rep_tensor_local, _weights_local, overlap_local, *_ = (
                _unrestricted_args(args)
            )
            density_spin = jnp.nan_to_num(density_spin, nan=0.0, posinf=0.0, neginf=0.0)
            fock_spin = _raw_fock_from_density(density_spin, params_local, args)
            residual_spin = jnp.stack(
                [
                    fock_spin[0] @ density_spin[0] @ overlap_local
                    - overlap_local @ density_spin[0] @ fock_spin[0],
                    fock_spin[1] @ density_spin[1] @ overlap_local
                    - overlap_local @ density_spin[1] @ fock_spin[1],
                ],
                axis=0,
            )
            residual_spin = jnp.nan_to_num(
                residual_spin,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            residual_spin = jnp.clip(
                residual_spin,
                -self.config.implicit_diff_clip,
                self.config.implicit_diff_clip,
            )
            return residual_spin

        implicit_cfg = ImplicitFixedPointConfig(
            tolerance=float(self.config.implicit_diff_tolerance),
            max_iter=int(self.config.implicit_diff_max_iter),
            regularization=float(self.config.implicit_diff_regularization),
            clip=float(self.config.implicit_diff_clip),
        )

        if fixed_point_args is None:
            def _fixed_point_from_residual(
                density_spin: Array,
                params_local: PyTree,
            ) -> Array:
                return density_spin + _residual(density_spin, params_local)
        else:
            def _fixed_point_from_residual(
                density_spin: Array,
                params_local: PyTree,
                args: tuple[Any, ...],
            ) -> Array:
                return density_spin + _residual(density_spin, params_local, args)

        density_spin_implicit = implicit_fixed_point_solution(
            xc_params,
            solution=density_star_spin,
            fixed_point=_fixed_point_from_residual,
            fixed_point_args=fixed_point_args,
            config=implicit_cfg,
        )
        molecule_final = _replace_molecule(
            forward_molecule,
            rdm1=density_spin_implicit,
        )
        return molecule_final, replace(forward_info, mode="self_consistent_implicit")

    def _full_scf_implicit_fixed_point(
        self,
        molecule: Any,
        xc_functional: Any,
        xc_params: PyTree,
    ) -> tuple[Any, DifferentiableSCFInfo]:
        forward_params = jax.tree_util.tree_map(jax.lax.stop_gradient, xc_params)
        problem = self._restricted_scf_problem(molecule, xc_functional, forward_params)
        (
            converged,
            cycles,
            density_star,
            mo_coeff_star,
            mo_energy_star,
            _energy,
            _xc_energy,
            _fock_final,
            _j_final,
            _k_final,
            density_history,
            mo_coeff_history,
            mo_energy_history,
            rms_history,
        ) = self._run_restricted_scf_core(
            problem,
            conv_tol=self.config.energy_convergence_tolerance(),
        )
        ctx = problem.ctx
        density_star, mo_coeff_star, mo_energy_star = self._select_restricted_scf_iterate(
            converged=converged,
            cycles=cycles,
            density_final=density_star,
            mo_coeff_final=mo_coeff_star,
            mo_energy_final=mo_energy_star,
            density_history=density_history,
            mo_coeff_history=mo_coeff_history,
            mo_energy_history=mo_energy_history,
            rms_history=rms_history,
        )
        density_star = jax.lax.stop_gradient(
            jnp.nan_to_num(density_star, nan=0.0, posinf=0.0, neginf=0.0)
        )
        mo_coeff_ref = jax.lax.stop_gradient(mo_coeff_star)
        mo_energy_ref = jax.lax.stop_gradient(mo_energy_star)
        response_with_k = not ctx.uses_explicit_hfx_fock
        fixed_point_args = None
        if _is_traceable_pytree(ctx.molecule):
            fixed_point_args = (
                ctx.molecule,
                ctx.x,
                ctx.mo_occ_total,
                ctx.mo_occ_stacked,
                mo_coeff_ref,
                mo_energy_ref,
            )

        def _context_from_args(args: tuple[Any, ...] | None) -> _RestrictedSCFContext:
            if args is None:
                return ctx
            molecule_arg, x_arg, mo_occ_total_arg, mo_occ_stacked_arg, _, _ = args
            h1e, rep_tensor, ao, weights, overlap = _scf_problem_arrays(molecule_arg)
            return _RestrictedSCFContext(
                molecule=molecule_arg,
                xc_functional=xc_functional,
                h1e=h1e,
                rep_tensor=rep_tensor,
                df_factors=_df_factors_from_molecule(molecule_arg),
                ao=ao,
                weights=weights,
                overlap=overlap,
                x=x_arg,
                mo_occ_total=mo_occ_total_arg,
                mo_occ_stacked=mo_occ_stacked_arg,
                hfx_nu=hfx_nu_source(molecule_arg),
                uses_explicit_hfx_fock=ctx.uses_explicit_hfx_fock,
            )

        def _density_from_fock(fock: Array, args: tuple[Any, ...] | None) -> Array:
            if args is None:
                x_arg, mo_occ_total_arg = ctx.x, ctx.mo_occ_total
            else:
                _molecule_arg, x_arg, mo_occ_total_arg, _mo_occ_stacked_arg, _coeff_ref, _energy_ref = args
            mo_energy_new, mo_coeff_new = _diagonalize_fock(
                _safe_symmetric_matrix(fock),
                x_arg,
                eigenvalue_jitter=self.config.eigenvalue_jitter,
            )
            density_new = _build_density_from_occ(
                jnp.nan_to_num(mo_coeff_new, nan=0.0, posinf=0.0, neginf=0.0),
                mo_occ_total_arg,
            )
            return jnp.nan_to_num(density_new, nan=0.0, posinf=0.0, neginf=0.0)

        def _full_fock_from_density(
            density_var: Array,
            params_var: PyTree,
            args: tuple[Any, ...] | None = fixed_point_args,
        ) -> tuple[Array, Array, Array, Array]:
            if args is None:
                coeff_ref, energy_ref = mo_coeff_ref, mo_energy_ref
            else:
                _molecule_arg, _x_arg, _mo_occ_total_arg, _mo_occ_stacked_arg, coeff_ref, energy_ref = args
            return self._restricted_fock_from_density(
                _context_from_args(args),
                params_var,
                density_var,
                coeff_ref,
                energy_ref,
                with_k=response_with_k,
            )

        if fixed_point_args is None:
            def _fixed_point_density(density_var: Array, params_var: PyTree) -> Array:
                fock, _alpha, _j_mat, _k_mat = _full_fock_from_density(density_var, params_var)
                return _density_from_fock(fock, None)
        else:
            def _fixed_point_density(density_var: Array, params_var: PyTree, args: tuple[Any, ...]) -> Array:
                fock, _alpha, _j_mat, _k_mat = _full_fock_from_density(density_var, params_var, args)
                return _density_from_fock(fock, args)

        implicit_cfg = ImplicitFixedPointConfig(
            tolerance=float(self.config.implicit_diff_tolerance),
            max_iter=int(self.config.implicit_diff_max_iter),
            regularization=float(self.config.implicit_diff_regularization),
            clip=float(self.config.implicit_diff_clip),
        )

        density_implicit = implicit_fixed_point_solution(
            xc_params,
            solution=density_star,
            fixed_point=_fixed_point_density,
            fixed_point_args=fixed_point_args,
            config=implicit_cfg,
        )

        fock_implicit, _alpha, _j_mat, _k_mat = _full_fock_from_density(
            density_implicit,
            xc_params,
        )
        mo_energy_implicit, mo_coeff_implicit = _diagonalize_fock(
            fock_implicit,
            ctx.x,
            eigenvalue_jitter=self.config.eigenvalue_jitter,
        )
        molecule_final = self._restricted_molecule_from_total_density(
            molecule,
            density=density_implicit,
            mo_coeff=mo_coeff_implicit,
            mo_energy=mo_energy_implicit,
            mo_occ_stacked=ctx.mo_occ_stacked,
        )
        molecule_final = self._with_neural_xc_grid_payload(molecule_final, xc_functional)
        return molecule_final, self._rks_loop_info(
            "self_consistent_implicit",
            converged=converged,
            cycles=cycles,
            rms_history=rms_history,
        )
