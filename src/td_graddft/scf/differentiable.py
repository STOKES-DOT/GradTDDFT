from __future__ import annotations

import copy
from dataclasses import dataclass, is_dataclass, replace
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import core as jax_core
from jax.lax import Precision
from jaxtyping import Array, PyTree

from ..data.integrals import build_direct_jk_from_basis, build_jk_from_eri_pair_matrix, shell_pair_schwarz_bounds
from ..nn_rsh.schema import SCFXCContributions
from .core import _build_density_from_occ, _diagonalize_fock, _orthogonalizer
from .rks import (
    _PYSCF_LIKE_DIIS_SPACE,
    _apply_level_shift,
    _build_jk,
    _commutator_error,
    _diis_extrapolate_lax,
    _vxc_matrix_from_grid_potential,
)


def _replace_molecule(molecule: Any, **updates: Any) -> Any:
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


def _build_density_spin(mo_coeff_spin: Array, mo_occ_spin: Array) -> Array:
    return jax.vmap(_build_density_from_occ)(mo_coeff_spin, mo_occ_spin)


def _apply_level_shift_spin(fock: Array, overlap: Array, density_spin: Array, factor: Array) -> Array:
    dm_vir = overlap - overlap @ density_spin @ overlap
    return fock + dm_vir * factor


def _spin_density_rms(density_new: Array, density_old: Array) -> Array:
    diff = jnp.asarray(density_new) - jnp.asarray(density_old)
    return jnp.sqrt(jnp.mean(diff**2))


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


def _restricted_channel_static(molecule: Any) -> tuple[np.ndarray, np.ndarray]:
    mo_coeff = np.asarray(molecule.mo_coeff)
    mo_occ = np.asarray(molecule.mo_occ)

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
) -> tuple[Array, Array, Array, Array, str, Array, Any | None, Array]:
    direct = getattr(functional, "scf_potential_components_and_alpha", None)
    if callable(direct):
        direct_components = direct(params, molecule)
        if len(direct_components) == 4:
            v_rho, v_grad, xc_kind, alpha = direct_components
            v_tau = jnp.zeros_like(jnp.asarray(v_rho))
            v_lapl = jnp.zeros_like(jnp.asarray(v_rho))
        elif len(direct_components) == 6:
            v_rho, v_grad, v_tau, v_lapl, xc_kind, alpha = direct_components
        else:
            raise ValueError(
                "scf_potential_components_and_alpha must return "
                "(v_rho, v_grad, xc_kind, alpha) or "
                "(v_rho, v_grad, v_tau, v_lapl, xc_kind, alpha)."
            )
        zero_vhf = jnp.zeros(
            (molecule.ao.shape[1], molecule.ao.shape[1]),
            dtype=functional_dtype,
        )
        return (
            jnp.asarray(v_rho, dtype=functional_dtype),
            jnp.asarray(v_grad, dtype=functional_dtype),
            jnp.asarray(v_tau, dtype=functional_dtype),
            jnp.asarray(v_lapl, dtype=functional_dtype),
            str(xc_kind),
            jnp.asarray(alpha, dtype=functional_dtype),
            None,
            zero_vhf,
        )

    resolved = _resolved_xc_object(params, functional, molecule)
    contributions_getter = getattr(resolved, "scf_contributions", None)
    if callable(contributions_getter):
        contributions = contributions_getter(molecule)
        if not isinstance(contributions, SCFXCContributions):
            raise TypeError(
                "scf_contributions(...) must return SCFXCContributions, "
                f"got {type(contributions)!r}."
            )
        if contributions.lr_hf_omegas is not None:
            raise NotImplementedError(
                "SCFXCContributions exposes long-range HF channels, but direct "
                "SCF assembly from (omega, coefficient) is not wired yet. "
                "Use extra_fock_matrix as the interim bridge in this PR."
            )
        extra_fock = contributions.extra_fock_matrix
        if extra_fock is None:
            extra_fock = jnp.zeros(
                (molecule.ao.shape[1], molecule.ao.shape[1]),
                dtype=functional_dtype,
            )
        resolved_xc = contributions.resolved_xc if contributions.resolved_xc is not None else resolved
        return (
            jnp.asarray(contributions.v_rho, dtype=functional_dtype),
            jnp.asarray(contributions.v_grad, dtype=functional_dtype),
            jnp.zeros_like(jnp.asarray(contributions.v_rho, dtype=functional_dtype)),
            jnp.zeros_like(jnp.asarray(contributions.v_rho, dtype=functional_dtype)),
            str(contributions.xc_kind),
            jnp.asarray(contributions.full_hf_fraction, dtype=functional_dtype),
            resolved_xc,
            jnp.asarray(extra_fock, dtype=functional_dtype),
        )

    v_rho, v_grad, v_tau, v_lapl, xc_kind = _grid_xc_potential_components_from_resolved(
        resolved,
        functional=functional,
        params=params,
        molecule=molecule,
    )
    alpha = _effective_exact_exchange_fraction_from_resolved(resolved)
    return (
        jnp.asarray(v_rho, dtype=functional_dtype),
        jnp.asarray(v_grad, dtype=functional_dtype),
        jnp.asarray(v_tau, dtype=functional_dtype),
        jnp.asarray(v_lapl, dtype=functional_dtype),
        str(xc_kind),
        jnp.asarray(alpha, dtype=functional_dtype),
        resolved,
        jnp.zeros((molecule.ao.shape[1], molecule.ao.shape[1]), dtype=functional_dtype),
    )


def _unrestricted_scf_xc_components(
    params: PyTree,
    functional: Any,
    molecule: Any,
    *,
    functional_dtype: Any,
) -> tuple[Array, Array, Array, Array, str, Array, Any | None, Array, Array]:
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
        resolved,
        jnp.asarray(extra_fock_a, dtype=functional_dtype),
        jnp.asarray(extra_fock_b, dtype=functional_dtype),
    )


def _grid_xc_potential_from_resolved(
    resolved: Any,
    *,
    functional: Any,
    params: PyTree,
    molecule: Any,
) -> Array:
    grid_potential = getattr(resolved, "grid_potential", None)
    if grid_potential is not None:
        return jnp.asarray(grid_potential(molecule))

    total_density = _density_on_grid(molecule)
    local_potential = getattr(resolved, "local_potential", None)
    if local_potential is not None:
        return jnp.asarray(local_potential(total_density))

    functional_local_potential = getattr(functional, "local_potential", None)
    if functional_local_potential is None:
        raise AttributeError(
            "The XC functional must expose local_potential(...) or grid_potential(...)."
        )
    return jnp.asarray(functional_local_potential(params, total_density))


def _density_on_grid(molecule: Any) -> Array:
    density_matrix = _spin_summed_density_matrix(molecule)
    ao = jnp.asarray(molecule.ao)
    return jnp.einsum(
        "rp,pq,rq->r",
        ao,
        density_matrix,
        ao,
        precision=Precision.HIGHEST,
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


def _direct_cuda_coulomb_exchange_matrices(
    direct_cuda_jk_builder: Any,
    direct_basis: Any,
    density: Array,
    *,
    direct_scf_tol: float = 0.0,
) -> tuple[Array, Array]:
    threshold = max(float(direct_scf_tol), 0.0)
    bounds = shell_pair_schwarz_bounds(direct_basis) if threshold > 0.0 else None

    @jax.custom_jvp
    def _eval(density_arg: Array) -> tuple[Array, Array]:
        return direct_cuda_jk_builder.build_jk(
            density_arg,
            density_cutoff=threshold,
        )

    @_eval.defjvp
    def _eval_jvp(primals, tangents):
        (density_arg,) = primals
        (density_dot,) = tangents
        density_dot = jnp.zeros_like(density_arg) if density_dot is None else density_dot
        primal_out = direct_cuda_jk_builder.build_jk(
            density_arg,
            density_cutoff=threshold,
        )
        tangent_out = build_direct_jk_from_basis(
            direct_basis,
            density_dot,
            screening_threshold=threshold,
            shell_pair_schwarz_bounds=bounds,
        )
        return primal_out, (tangent_out.j, tangent_out.k)

    return _eval(density)


def _jk_source_from_molecule(molecule: Any) -> tuple[Any, ...]:
    direct_cuda_jk_builder = getattr(molecule, "direct_cuda_jk_builder", None)
    direct_basis = getattr(molecule, "direct_basis", None)
    direct_jk_engine = getattr(molecule, "direct_jk_engine", None)
    if (
        direct_cuda_jk_builder is not None
        and direct_basis is not None
        and str(direct_jk_engine) == "cuda"
    ):
        return (
            "direct_cuda",
            direct_cuda_jk_builder,
            direct_basis,
            float(getattr(molecule, "direct_scf_tol", 0.0) or 0.0),
        )
    return ("eri", _repulsion_integrals_from_molecule(molecule))


def _coulomb_exchange_matrices_from_source(
    jk_source: tuple[Any, ...],
    density: Array,
) -> tuple[Array, Array]:
    if jk_source[0] == "direct_cuda":
        _, direct_cuda_jk_builder, direct_basis, direct_scf_tol = jk_source
        return _direct_cuda_coulomb_exchange_matrices(
            direct_cuda_jk_builder,
            direct_basis,
            density,
            direct_scf_tol=direct_scf_tol,
        )
    return _coulomb_exchange_matrices(jk_source[1], density)


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


def _grid_xc_potential(
    params: PyTree,
    functional: Any,
    molecule: Any,
) -> Array:
    resolved = _resolved_xc_object(params, functional, molecule)
    grid_potential = getattr(resolved, "grid_potential", None)
    if grid_potential is not None:
        return jnp.asarray(grid_potential(molecule))

    total_density = _density_on_grid(molecule)
    local_potential = getattr(resolved, "local_potential", None)
    if local_potential is not None:
        return jnp.asarray(local_potential(total_density))

    functional_local_potential = getattr(functional, "local_potential", None)
    if functional_local_potential is None:
        raise AttributeError(
            "The XC functional must expose local_potential(...) or grid_potential(...)."
        )
    return jnp.asarray(functional_local_potential(params, total_density))


def _normalize_response_feature_kind(value: Any) -> str:
    if value is None:
        return "LDA"
    kind = str(value).upper()
    if kind in {"LDA", "GGA", "MGGA", "MGGA_LAPL"}:
        return kind
    return "LDA"


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

    v_rho = _grid_xc_potential_from_resolved(
        resolved,
        functional=functional,
        params=params,
        molecule=molecule,
    )
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


def _effective_exact_exchange_fraction(
    params: PyTree,
    functional: Any,
    molecule: Any,
) -> Array:
    resolved = _resolved_xc_object(params, functional, molecule)
    exact_exchange_fraction = getattr(resolved, "exact_exchange_fraction", 0.0)
    return jnp.asarray(exact_exchange_fraction)


def _effective_exact_exchange_fraction_from_resolved(resolved: Any) -> Array:
    exact_exchange_fraction = getattr(resolved, "exact_exchange_fraction", 0.0)
    return jnp.asarray(exact_exchange_fraction)


def _dm21_local_hfx_fock_correction(
    *,
    resolved_xc: Any,
    molecule: Any,
    ao: Array,
    density: Array,
) -> Array:
    """DM21-style local-HF derivative correction: sum_i vhf_i * d(hfx_i)/dD."""

    nu_cache = getattr(molecule, "hfx_nu", None)
    if nu_cache is None:
        return jnp.zeros((ao.shape[1], ao.shape[1]), dtype=ao.dtype)
    if (
        hasattr(resolved_xc, "grid_hfx_feature_gradients_fn")
        and getattr(resolved_xc, "grid_hfx_feature_gradients_fn", None) is None
    ):
        return jnp.zeros((ao.shape[1], ao.shape[1]), dtype=ao.dtype)

    gradient_getter = getattr(resolved_xc, "grid_hfx_feature_gradients", None)
    if gradient_getter is None:
        return jnp.zeros((ao.shape[1], ao.shape[1]), dtype=ao.dtype)

    try:
        vhf_pair = gradient_getter(molecule)
    except TypeError:
        try:
            vhf_pair = gradient_getter()
        except AttributeError:
            return jnp.zeros((ao.shape[1], ao.shape[1]), dtype=ao.dtype)
    except AttributeError:
        return jnp.zeros((ao.shape[1], ao.shape[1]), dtype=ao.dtype)

    if vhf_pair is None:
        return jnp.zeros((ao.shape[1], ao.shape[1]), dtype=ao.dtype)
    vhf_a_raw, vhf_b_raw = vhf_pair

    vhf_a = jnp.asarray(vhf_a_raw, dtype=ao.dtype)
    vhf_b = jnp.asarray(vhf_b_raw, dtype=ao.dtype)
    if vhf_a.ndim == 1:
        vhf_a = vhf_a[..., None]
    if vhf_b.ndim == 1:
        vhf_b = vhf_b[..., None]
    vhf = 0.5 * (vhf_a + vhf_b)
    vhf = jnp.nan_to_num(vhf, nan=0.0, posinf=0.0, neginf=0.0)

    nu = jnp.asarray(nu_cache, dtype=ao.dtype)
    if nu.ndim != 4:
        raise ValueError(
            "molecule.hfx_nu must have shape (n_omega, ngrids, nao, nao), "
            f"got {nu.shape}."
        )
    n_omega, ngrid, nao, nao2 = nu.shape
    if nao != ao.shape[1] or nao2 != ao.shape[1]:
        raise ValueError(
            "molecule.hfx_nu AO dimensions must match molecule.ao second axis "
            f"(got {nu.shape[-2:]} vs {(ao.shape[1], ao.shape[1])})."
        )
    if vhf.shape[0] != ngrid:
        raise ValueError(
            "grid_hfx_feature_gradients grid axis must match hfx_nu grid axis "
            f"(got {vhf.shape[0]} vs {ngrid})."
        )
    if vhf.shape[1] != n_omega:
        if vhf.shape[1] == 1 and n_omega > 1:
            vhf = jnp.repeat(vhf, n_omega, axis=1)
        else:
            raise ValueError(
                "grid_hfx_feature_gradients omega axis must match hfx_nu omega axis "
                f"(got {vhf.shape[1]} vs {n_omega})."
            )

    density_half = 0.5 * density
    e = jnp.einsum(
        "rp,pq->rq",
        ao,
        density_half,
        precision=Precision.HIGHEST,
    )
    fxx = jnp.einsum(
        "wgbc,gc->wgb",
        nu,
        e,
        precision=Precision.HIGHEST,
    )
    aow = -0.5 * fxx * vhf.T[:, :, None]
    vmat = jnp.einsum(
        "rp,wrq->wpq",
        ao,
        aow,
        precision=Precision.HIGHEST,
    ).sum(axis=0)
    correction = vmat + vmat.T
    correction = jnp.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0)
    return 0.5 * (correction + correction.T)


def _uses_dm21_local_hfx_correction(resolved_xc: Any) -> bool:
    """Return True only for DM21-like functionals with explicit HFX feature grads."""

    gradient_getter = getattr(resolved_xc, "grid_hfx_feature_gradients", None)
    if gradient_getter is None or not callable(gradient_getter):
        return False
    if (
        hasattr(resolved_xc, "grid_hfx_feature_gradients_fn")
        and getattr(resolved_xc, "grid_hfx_feature_gradients_fn", None) is None
    ):
        return False
    return True


def _restricted_hfx_features_from_nu(
    *,
    ao: Array,
    density: Array,
    nu_cache: Array,
) -> Array:
    """Recompute Neural XC local-HF density features hfx_local from current density."""

    nu = jnp.asarray(nu_cache, dtype=ao.dtype)
    if nu.ndim != 4:
        raise ValueError(
            "hfx_nu must have shape (n_omega, ngrids, nao, nao), "
            f"got {nu.shape}."
        )
    density_half = 0.5 * density
    e = jnp.einsum(
        "rp,pq->rq",
        ao,
        density_half,
        precision=Precision.HIGHEST,
    )
    fxx = jnp.einsum(
        "wgbc,gc->wgb",
        nu,
        e,
        precision=Precision.HIGHEST,
    )
    exx = -0.5 * jnp.einsum(
        "gb,wgb->wg",
        e,
        fxx,
        precision=Precision.HIGHEST,
    )
    exx = jnp.nan_to_num(exx, nan=0.0, posinf=0.0, neginf=0.0)
    # Restricted reference: alpha and beta channels share the same exx profile.
    exx_grid = exx.T
    return jnp.stack([exx_grid, exx_grid], axis=0)

_build_density = _build_density_from_occ


@dataclass(frozen=True)
class DifferentiableSCFConfig:
    """Configuration for fixed-density / self-consistent differentiable SCF."""

    mode: Literal["fixed_density", "self_consistent"] = "fixed_density"
    gradient_mode: Literal["expl", "impl"] = "expl"
    implicit_forward_mode: Literal["expl", "input_state"] = "input_state"
    max_cycle: int = 12
    damping: float = 0.25
    level_shift: float = 0.0
    occupation_tolerance: float = 1e-8
    conv_tol_density: float = 1e-8
    orthogonalization_eps: float = 1e-10
    eigenvalue_jitter: float = 1e-8
    vxc_clip: float = 20.0
    iterate_selection: Literal["final", "best_rms", "first_converged"] = "final"
    require_converged_iterates: bool = False
    implicit_diff_max_iter: int = 24
    implicit_diff_step_size: float = 0.2
    implicit_diff_clip: float = 1e4
    implicit_diff_solver: Literal["normal_cg", "gmres", "bicgstab"] = "normal_cg"
    implicit_diff_tolerance: float = 1e-6
    implicit_diff_regularization: float = 1e-3
    implicit_diff_restart: int = 12

    def __post_init__(self) -> None:
        _valid_gradient_modes = {"expl", "impl"}
        if self.gradient_mode not in _valid_gradient_modes:
            raise ValueError(
                f"gradient_mode must be one of {_valid_gradient_modes}, "
                f"got {self.gradient_mode!r}."
            )
        _valid_forward_modes = {"expl", "input_state"}
        if self.implicit_forward_mode not in _valid_forward_modes:
            raise ValueError(
                f"implicit_forward_mode must be one of {_valid_forward_modes}, "
                f"got {self.implicit_forward_mode!r}."
            )


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


class DifferentiableSCF:
    """Differentiable SCF wrapper with fixed-density and self-consistent modes."""

    def __init__(self, config: DifferentiableSCFConfig | None = None):
        self.config = DifferentiableSCFConfig() if config is None else config

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

    def _implicit_forward_state_restricted(
        self,
        molecule: Any,
        xc_functional: Any,
        xc_params: PyTree,
    ) -> tuple[Any, DifferentiableSCFInfo]:
        mode = str(self.config.implicit_forward_mode)
        if mode == "expl":
            expl_cfg = replace(
                self.config,
                gradient_mode="expl",
                implicit_forward_mode="expl",
            )
            expl_solver = DifferentiableSCF(expl_cfg)
            return expl_solver._full_scf(
                molecule,
                xc_functional,
                jax.lax.stop_gradient(xc_params),
            )
        if mode == "input_state":
            density = _spin_summed_density_matrix(molecule)
            return molecule, self._implicit_input_state_info(density)
        raise ValueError(
            "implicit_forward_mode must be 'expl' or 'input_state', "
            f"got {self.config.implicit_forward_mode!r}."
        )

    def _implicit_forward_state_unrestricted(
        self,
        molecule: Any,
        xc_functional: Any,
        xc_params: PyTree,
    ) -> tuple[Any, DifferentiableSCFInfo]:
        mode = str(self.config.implicit_forward_mode)
        if mode == "expl":
            expl_cfg = replace(
                self.config,
                gradient_mode="expl",
                implicit_forward_mode="expl",
            )
            expl_solver = DifferentiableSCF(expl_cfg)
            return expl_solver._full_scf_unrestricted(
                molecule,
                xc_functional,
                jax.lax.stop_gradient(xc_params),
            )
        if mode == "input_state":
            density = _spin_resolved_density_matrix(molecule)
            return molecule, self._implicit_input_state_info(density)
        raise ValueError(
            "implicit_forward_mode must be 'expl' or 'input_state', "
            f"got {self.config.implicit_forward_mode!r}."
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
            return self._full_scf_implicit_commutator(molecule, xc_functional, xc_params)
        if getattr(molecule, "h1e", None) is None:
            raise AttributeError("Molecule-like object must define h1e for self-consistent mode.")
        if getattr(molecule, "rep_tensor", None) is None:
            raise AttributeError(
                "Molecule-like object must define rep_tensor for self-consistent mode."
            )
        if getattr(molecule, "ao", None) is None or getattr(molecule, "grid", None) is None:
            raise AttributeError(
                "Molecule-like object must define ao and grid.weights for self-consistent mode."
            )

        h1e = jnp.asarray(molecule.h1e)
        jk_source = _jk_source_from_molecule(molecule)
        ao = jnp.asarray(molecule.ao)
        weights = jnp.asarray(molecule.grid.weights)

        overlap = getattr(molecule, "overlap_matrix", None)
        if overlap is None:
            overlap = jnp.eye(h1e.shape[0], dtype=h1e.dtype)
        else:
            overlap = jnp.asarray(overlap)
        x = _orthogonalizer(overlap, self.config.orthogonalization_eps)

        density0 = _initial_density_matrix(molecule)
        cached_initial_density = getattr(molecule, "scf_initial_density", None)
        if cached_initial_density is None:
            mo_coeff0, mo_occ0 = _restricted_channel(molecule)
        else:
            mo_coeff0 = _mo_coeff_guess_from_density_matrix(
                density0,
                overlap,
                orthogonalization_eps=self.config.orthogonalization_eps,
            )
            _, mo_occ0 = _restricted_channel(molecule)
        mo_energy_raw = jnp.asarray(molecule.mo_energy)
        mo_energy0 = mo_energy_raw[0] if mo_energy_raw.ndim == 2 else mo_energy_raw

        mo_occ_total = _restricted_total_occupations(
            molecule,
            occupation_tolerance=self.config.occupation_tolerance,
        )
        nmo = int(mo_coeff0.shape[-1])
        if mo_occ_total.ndim != 1 or int(mo_occ_total.shape[0]) != nmo:
            raise ValueError("Restricted occupation vector must have shape (nmo,) in self-consistent mode.")
        mo_occ_total = jnp.asarray(mo_occ_total, dtype=h1e.dtype)

        mo_occ_stacked = _restricted_stacked_occupations_from_total(mo_occ_total)
        level_shift = jnp.asarray(self.config.level_shift, dtype=h1e.dtype)
        has_level_shift = self.config.level_shift != 0.0

        def _raw_fock_from_density(
            density: Array,
            mo_coeff_ref: Array,
            mo_energy_ref: Array,
        ) -> tuple[Array, Any]:
            density_spin = jnp.stack([0.5 * density, 0.5 * density], axis=0)
            mo_coeff_spin = jnp.stack([mo_coeff_ref, mo_coeff_ref], axis=0)
            mo_energy_spin = jnp.stack([mo_energy_ref, mo_energy_ref], axis=0)
            hfx_local_iter = getattr(molecule, "hfx_local", None)
            hfx_nu = getattr(molecule, "hfx_nu", None)
            if hfx_nu is not None:
                hfx_local_iter = _restricted_hfx_features_from_nu(
                    ao=ao,
                    density=density,
                    nu_cache=hfx_nu,
                )
            updates = dict(
                rdm1=density_spin,
                mo_coeff=mo_coeff_spin,
                mo_occ=mo_occ_stacked,
                mo_energy=mo_energy_spin,
            )
            if hasattr(molecule, "hfx_local"):
                updates["hfx_local"] = hfx_local_iter
            molecule_iter = _replace_molecule(molecule, **updates)
            j_mat, k_mat = _coulomb_exchange_matrices_from_source(jk_source, density)
            vxc_rho, vxc_grad, vxc_tau, vxc_lapl, xc_kind, alpha, resolved_xc, vhf_matrix = _scf_xc_components(
                xc_params,
                xc_functional,
                molecule_iter,
                functional_dtype=h1e.dtype,
            )
            vxc_rho = jnp.nan_to_num(
                vxc_rho,
                nan=0.0,
                posinf=self.config.vxc_clip,
                neginf=-self.config.vxc_clip,
            )
            vxc_grad = jnp.nan_to_num(
                vxc_grad,
                nan=0.0,
                posinf=self.config.vxc_clip,
                neginf=-self.config.vxc_clip,
            )
            vxc_rho = jnp.clip(vxc_rho, -self.config.vxc_clip, self.config.vxc_clip)
            vxc_grad = jnp.clip(vxc_grad, -self.config.vxc_clip, self.config.vxc_clip)
            vxc_matrix = _build_vxc_matrix_from_components(
                molecule=molecule_iter,
                weights=weights,
                v_rho=vxc_rho,
                v_grad=vxc_grad,
                v_tau=vxc_tau,
                v_lapl=vxc_lapl,
                xc_kind=xc_kind,
            )
            alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
            alpha = jnp.clip(alpha, 0.0, 1.0)
            if resolved_xc is not None and _uses_dm21_local_hfx_correction(resolved_xc):
                vhf_matrix = vhf_matrix + _dm21_local_hfx_fock_correction(
                    resolved_xc=resolved_xc,
                    molecule=molecule_iter,
                    ao=ao,
                    density=density,
                )
            fock = h1e + j_mat - 0.5 * alpha * k_mat + vxc_matrix + vhf_matrix
            fock = jnp.nan_to_num(fock, nan=0.0, posinf=0.0, neginf=0.0)
            return 0.5 * (fock + fock.T), molecule_iter

        def body(
            carry: tuple[Array, Array, Array, Array, Array, Array, Array],
            _,
        ) -> tuple[
            tuple[Array, Array, Array, Array, Array, Array, Array],
            tuple[Array, Array, Array, Array],
        ]:
            density, mo_coeff, mo_energy, fock_hist, err_hist, hist_head, hist_count = carry
            fock, _ = _raw_fock_from_density(density, mo_coeff, mo_energy)
            error = _commutator_error(fock, density, overlap)
            fock_eff, fock_hist, err_hist, hist_head, hist_count = _diis_extrapolate_lax(
                fock,
                error,
                fock_hist,
                err_hist,
                hist_head,
                hist_count,
            )
            fock_diag = jax.lax.cond(
                jnp.asarray(has_level_shift),
                lambda operand: _apply_level_shift(*operand),
                lambda operand: operand[0],
                operand=(fock_eff, overlap, density, level_shift),
            )

            mo_energy_new, mo_coeff_new = _diagonalize_fock(
                fock_diag,
                x,
                eigenvalue_jitter=self.config.eigenvalue_jitter,
            )
            mo_energy_new = jnp.nan_to_num(mo_energy_new, nan=0.0, posinf=0.0, neginf=0.0)
            mo_coeff_new = jnp.nan_to_num(mo_coeff_new, nan=0.0, posinf=0.0, neginf=0.0)
            density_new = _build_density(mo_coeff_new, mo_occ_total)
            density_new = jnp.nan_to_num(density_new, nan=0.0, posinf=0.0, neginf=0.0)
            density_next = (1.0 - self.config.damping) * density_new + self.config.damping * density
            rms_density = jnp.sqrt(jnp.mean((density_next - density) ** 2))
            return (
                density_next,
                mo_coeff_new,
                mo_energy_new,
                fock_hist,
                err_hist,
                hist_head,
                hist_count,
            ), (
                density_next,
                mo_coeff_new,
                mo_energy_new,
                rms_density,
            )

        nao = h1e.shape[0]
        carry0 = (
            density0,
            mo_coeff0,
            mo_energy0,
            jnp.zeros((_PYSCF_LIKE_DIIS_SPACE, nao, nao), dtype=h1e.dtype),
            jnp.zeros((_PYSCF_LIKE_DIIS_SPACE, nao * nao), dtype=h1e.dtype),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )
        (_, _, _, _, _, _, _), (
            density_history,
            mo_coeff_history,
            mo_energy_history,
            rms_history,
        ) = jax.lax.scan(
            body,
            carry0,
            xs=None,
            length=self.config.max_cycle,
        )

        best_idx = jnp.argmin(rms_history)
        converged_mask = rms_history < self.config.conv_tol_density
        converged = jnp.any(converged_mask)
        first_conv = jnp.argmax(converged_mask) + 1
        cycles = jnp.where(converged, first_conv, self.config.max_cycle)
        best_cycle = best_idx + 1
        converged_best_idx = jnp.argmin(
            jnp.where(
                converged_mask,
                rms_history,
                jnp.asarray(jnp.inf, dtype=rms_history.dtype),
            )
        )
        fallback_idx = jnp.asarray(self.config.max_cycle - 1)

        if self.config.iterate_selection == "best_rms":
            selected_idx = jnp.where(
                converged & bool(self.config.require_converged_iterates),
                converged_best_idx,
                best_idx,
            )
        elif self.config.iterate_selection == "first_converged":
            selected_idx = jnp.where(
                converged,
                first_conv - 1,
                fallback_idx if bool(self.config.require_converged_iterates) else best_idx,
            )
        else:
            selected_idx = fallback_idx

        if bool(self.config.require_converged_iterates):
            selected_idx = jnp.where(converged, selected_idx, fallback_idx)

        density_final = density_history[selected_idx]
        mo_coeff_selected = mo_coeff_history[selected_idx]
        mo_energy_selected = mo_energy_history[selected_idx]
        fock_final, _ = _raw_fock_from_density(
            density_final,
            mo_coeff_selected,
            mo_energy_selected,
        )
        mo_energy_final, mo_coeff_final = _diagonalize_fock(
            fock_final,
            x,
            eigenvalue_jitter=self.config.eigenvalue_jitter,
        )
        mo_energy_final = jnp.nan_to_num(mo_energy_final, nan=0.0, posinf=0.0, neginf=0.0)
        mo_coeff_final = jnp.nan_to_num(mo_coeff_final, nan=0.0, posinf=0.0, neginf=0.0)

        density_spin_final = jnp.stack([0.5 * density_final, 0.5 * density_final], axis=0)
        mo_coeff_spin_final = jnp.stack([mo_coeff_final, mo_coeff_final], axis=0)
        mo_energy_spin_final = jnp.stack([mo_energy_final, mo_energy_final], axis=0)
        molecule_final = _replace_molecule(
            molecule,
            rdm1=density_spin_final,
            mo_coeff=mo_coeff_spin_final,
            mo_occ=mo_occ_stacked,
            mo_energy=mo_energy_spin_final,
        )

        info = DifferentiableSCFInfo(
            mode="self_consistent",
            converged=converged,
            cycles=cycles,
            final_rms_density=rms_history[-1],
            rms_density_history=rms_history,
            selected_cycle=selected_idx + 1,
            selected_rms_density=rms_history[selected_idx],
            best_cycle=best_cycle,
            best_rms_density=rms_history[best_idx],
        )
        return molecule_final, info

    def run_runtime_forward(
        self,
        molecule: Any,
        xc_functional: Any,
        xc_params: PyTree,
    ) -> tuple[Any, DifferentiableSCFInfo]:
        """Run the SCF primal state with Python control flow.

        This is intended for two-stage training: the forward SCF state is built
        outside ``jax.value_and_grad``, then the implicit commutator VJP consumes
        that state as ``implicit_forward_mode='input_state'`` (``gradient_mode='impl'``).  The current
        runtime forward path is restricted closed-shell only.
        """

        if self.config.mode == "fixed_density":
            return self._single_step(molecule)
        if self.config.mode != "self_consistent":
            raise ValueError(f"Unsupported DifferentiableSCF mode: {self.config.mode!r}")
        if _is_unrestricted_reference(molecule):
            raise NotImplementedError(
                "Runtime-forward implicit SCF currently supports restricted references only."
            )
        return self._full_scf_runtime_restricted(molecule, xc_functional, xc_params)

    def _full_scf_runtime_restricted(
        self,
        molecule: Any,
        xc_functional: Any,
        xc_params: PyTree,
    ) -> tuple[Any, DifferentiableSCFInfo]:
        if getattr(molecule, "h1e", None) is None:
            raise AttributeError("Molecule-like object must define h1e for self-consistent mode.")
        if getattr(molecule, "rep_tensor", None) is None:
            raise AttributeError(
                "Molecule-like object must define rep_tensor for self-consistent mode."
            )
        if getattr(molecule, "ao", None) is None or getattr(molecule, "grid", None) is None:
            raise AttributeError(
                "Molecule-like object must define ao and grid.weights for self-consistent mode."
            )

        h1e = jnp.asarray(molecule.h1e)
        jk_source = _jk_source_from_molecule(molecule)
        ao = jnp.asarray(molecule.ao)
        weights = jnp.asarray(molecule.grid.weights)

        overlap = getattr(molecule, "overlap_matrix", None)
        if overlap is None:
            overlap = jnp.eye(h1e.shape[0], dtype=h1e.dtype)
        else:
            overlap = jnp.asarray(overlap)
        x = _orthogonalizer(overlap, self.config.orthogonalization_eps)

        density0 = _initial_density_matrix(molecule)
        cached_initial_density = getattr(molecule, "scf_initial_density", None)
        if cached_initial_density is None:
            mo_coeff0, mo_occ0 = _restricted_channel(molecule)
        else:
            mo_coeff0 = _mo_coeff_guess_from_density_matrix(
                density0,
                overlap,
                orthogonalization_eps=self.config.orthogonalization_eps,
            )
            _, mo_occ0 = _restricted_channel(molecule)
        mo_energy_raw = jnp.asarray(molecule.mo_energy)
        mo_energy0 = mo_energy_raw[0] if mo_energy_raw.ndim == 2 else mo_energy_raw

        mo_occ_total = _restricted_total_occupations(
            molecule,
            occupation_tolerance=self.config.occupation_tolerance,
        )
        nmo = int(mo_coeff0.shape[-1])
        if mo_occ_total.ndim != 1 or int(mo_occ_total.shape[0]) != nmo:
            raise ValueError("Restricted occupation vector must have shape (nmo,) in self-consistent mode.")
        mo_occ_total = jnp.asarray(mo_occ_total, dtype=h1e.dtype)

        mo_occ_stacked = _restricted_stacked_occupations_from_total(mo_occ_total)
        level_shift = jnp.asarray(self.config.level_shift, dtype=h1e.dtype)
        has_level_shift = self.config.level_shift != 0.0
        hfx_nu = getattr(molecule, "hfx_nu", None)

        def _raw_fock_from_density(
            density: Array,
            mo_coeff_ref: Array,
            mo_energy_ref: Array,
        ) -> tuple[Array, Any]:
            density_spin = jnp.stack([0.5 * density, 0.5 * density], axis=0)
            mo_coeff_spin = jnp.stack([mo_coeff_ref, mo_coeff_ref], axis=0)
            mo_energy_spin = jnp.stack([mo_energy_ref, mo_energy_ref], axis=0)
            hfx_local_iter = getattr(molecule, "hfx_local", None)
            if hfx_nu is not None:
                hfx_local_iter = _restricted_hfx_features_from_nu(
                    ao=ao,
                    density=density,
                    nu_cache=hfx_nu,
                )
            updates = dict(
                rdm1=density_spin,
                mo_coeff=mo_coeff_spin,
                mo_occ=mo_occ_stacked,
                mo_energy=mo_energy_spin,
            )
            if hasattr(molecule, "hfx_local"):
                updates["hfx_local"] = hfx_local_iter
            molecule_iter = _replace_molecule(molecule, **updates)
            j_mat, k_mat = _coulomb_exchange_matrices_from_source(jk_source, density)
            vxc_rho, vxc_grad, vxc_tau, vxc_lapl, xc_kind, alpha, resolved_xc, vhf_matrix = _scf_xc_components(
                xc_params,
                xc_functional,
                molecule_iter,
                functional_dtype=h1e.dtype,
            )
            vxc_rho = jnp.nan_to_num(
                vxc_rho,
                nan=0.0,
                posinf=self.config.vxc_clip,
                neginf=-self.config.vxc_clip,
            )
            vxc_grad = jnp.nan_to_num(
                vxc_grad,
                nan=0.0,
                posinf=self.config.vxc_clip,
                neginf=-self.config.vxc_clip,
            )
            vxc_rho = jnp.clip(vxc_rho, -self.config.vxc_clip, self.config.vxc_clip)
            vxc_grad = jnp.clip(vxc_grad, -self.config.vxc_clip, self.config.vxc_clip)
            vxc_matrix = _build_vxc_matrix_from_components(
                molecule=molecule_iter,
                weights=weights,
                v_rho=vxc_rho,
                v_grad=vxc_grad,
                v_tau=vxc_tau,
                v_lapl=vxc_lapl,
                xc_kind=xc_kind,
            )
            alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
            alpha = jnp.clip(alpha, 0.0, 1.0)
            if resolved_xc is not None and _uses_dm21_local_hfx_correction(resolved_xc):
                vhf_matrix = vhf_matrix + _dm21_local_hfx_fock_correction(
                    resolved_xc=resolved_xc,
                    molecule=molecule_iter,
                    ao=ao,
                    density=density,
                )
            fock = h1e + j_mat - 0.5 * alpha * k_mat + vxc_matrix + vhf_matrix
            fock = jnp.nan_to_num(fock, nan=0.0, posinf=0.0, neginf=0.0)
            return 0.5 * (fock + fock.T), molecule_iter

        nao = h1e.shape[0]
        density = density0
        mo_coeff = mo_coeff0
        mo_energy = mo_energy0
        fock_hist = jnp.zeros((_PYSCF_LIKE_DIIS_SPACE, nao, nao), dtype=h1e.dtype)
        err_hist = jnp.zeros((_PYSCF_LIKE_DIIS_SPACE, nao * nao), dtype=h1e.dtype)
        hist_head = jnp.asarray(0, dtype=jnp.int32)
        hist_count = jnp.asarray(0, dtype=jnp.int32)
        density_history: list[Array] = []
        mo_coeff_history: list[Array] = []
        mo_energy_history: list[Array] = []
        rms_values: list[Array] = []

        for _cycle in range(int(self.config.max_cycle)):
            fock, _ = _raw_fock_from_density(density, mo_coeff, mo_energy)
            error = _commutator_error(fock, density, overlap)
            fock_eff, fock_hist, err_hist, hist_head, hist_count = _diis_extrapolate_lax(
                fock,
                error,
                fock_hist,
                err_hist,
                hist_head,
                hist_count,
            )
            if has_level_shift:
                fock_diag = _apply_level_shift(fock_eff, overlap, density, level_shift)
            else:
                fock_diag = fock_eff

            mo_energy_new, mo_coeff_new = _diagonalize_fock(
                fock_diag,
                x,
                eigenvalue_jitter=self.config.eigenvalue_jitter,
            )
            mo_energy_new = jnp.nan_to_num(mo_energy_new, nan=0.0, posinf=0.0, neginf=0.0)
            mo_coeff_new = jnp.nan_to_num(mo_coeff_new, nan=0.0, posinf=0.0, neginf=0.0)
            density_new = _build_density(mo_coeff_new, mo_occ_total)
            density_new = jnp.nan_to_num(density_new, nan=0.0, posinf=0.0, neginf=0.0)
            density_next = (1.0 - self.config.damping) * density_new + self.config.damping * density
            rms_density = jnp.sqrt(jnp.mean((density_next - density) ** 2))

            density_history.append(density_next)
            mo_coeff_history.append(mo_coeff_new)
            mo_energy_history.append(mo_energy_new)
            rms_values.append(rms_density)

            density = density_next
            mo_coeff = mo_coeff_new
            mo_energy = mo_energy_new
            if float(jax.device_get(rms_density)) < float(self.config.conv_tol_density):
                break

        if not rms_values:
            density_history.append(density)
            mo_coeff_history.append(mo_coeff)
            mo_energy_history.append(mo_energy)
            rms_values.append(jnp.asarray(0.0, dtype=h1e.dtype))

        density_history_arr = jnp.stack(density_history, axis=0)
        mo_coeff_history_arr = jnp.stack(mo_coeff_history, axis=0)
        mo_energy_history_arr = jnp.stack(mo_energy_history, axis=0)
        rms_history = jnp.stack(rms_values, axis=0)

        rms_host = np.asarray(jax.device_get(rms_history))
        converged_mask_host = rms_host < float(self.config.conv_tol_density)
        converged_host = bool(np.any(converged_mask_host))
        best_idx_host = int(np.argmin(rms_host))
        first_conv_idx_host = int(np.argmax(converged_mask_host)) if converged_host else len(rms_host) - 1
        fallback_idx_host = len(rms_host) - 1
        if self.config.iterate_selection == "best_rms":
            selected_idx_host = (
                first_conv_idx_host
                if converged_host and bool(self.config.require_converged_iterates)
                else best_idx_host
            )
        elif self.config.iterate_selection == "first_converged":
            selected_idx_host = (
                first_conv_idx_host
                if converged_host
                else fallback_idx_host if bool(self.config.require_converged_iterates) else best_idx_host
            )
        else:
            selected_idx_host = fallback_idx_host
        if bool(self.config.require_converged_iterates) and not converged_host:
            selected_idx_host = fallback_idx_host

        selected_idx = jnp.asarray(selected_idx_host, dtype=jnp.int32)
        density_final = density_history_arr[selected_idx_host]
        mo_coeff_selected = mo_coeff_history_arr[selected_idx_host]
        mo_energy_selected = mo_energy_history_arr[selected_idx_host]
        fock_final, _ = _raw_fock_from_density(
            density_final,
            mo_coeff_selected,
            mo_energy_selected,
        )
        mo_energy_final, mo_coeff_final = _diagonalize_fock(
            fock_final,
            x,
            eigenvalue_jitter=self.config.eigenvalue_jitter,
        )
        mo_energy_final = jnp.nan_to_num(mo_energy_final, nan=0.0, posinf=0.0, neginf=0.0)
        mo_coeff_final = jnp.nan_to_num(mo_coeff_final, nan=0.0, posinf=0.0, neginf=0.0)

        density_spin_final = jnp.stack([0.5 * density_final, 0.5 * density_final], axis=0)
        mo_coeff_spin_final = jnp.stack([mo_coeff_final, mo_coeff_final], axis=0)
        mo_energy_spin_final = jnp.stack([mo_energy_final, mo_energy_final], axis=0)
        molecule_final = _replace_molecule(
            molecule,
            rdm1=density_spin_final,
            mo_coeff=mo_coeff_spin_final,
            mo_occ=mo_occ_stacked,
            mo_energy=mo_energy_spin_final,
        )

        best_idx = jnp.asarray(best_idx_host, dtype=jnp.int32)
        info = DifferentiableSCFInfo(
            mode="self_consistent_runtime_forward",
            converged=jnp.asarray(converged_host),
            cycles=jnp.asarray(first_conv_idx_host + 1 if converged_host else len(rms_values), dtype=jnp.int32),
            final_rms_density=rms_history[-1],
            rms_density_history=rms_history,
            selected_cycle=selected_idx + 1,
            selected_rms_density=rms_history[selected_idx_host],
            best_cycle=best_idx + 1,
            best_rms_density=rms_history[best_idx_host],
        )
        return molecule_final, info

    def _full_scf_unrestricted(
        self,
        molecule: Any,
        xc_functional: Any,
        xc_params: PyTree,
    ) -> tuple[Any, DifferentiableSCFInfo]:
        if getattr(molecule, "h1e", None) is None:
            raise AttributeError("Molecule-like object must define h1e for self-consistent mode.")
        if getattr(molecule, "rep_tensor", None) is None:
            raise AttributeError(
                "Molecule-like object must define rep_tensor for self-consistent mode."
            )
        if getattr(molecule, "ao", None) is None or getattr(molecule, "grid", None) is None:
            raise AttributeError(
                "Molecule-like object must define ao and grid.weights for self-consistent mode."
            )

        h1e = jnp.asarray(molecule.h1e)
        rep_tensor = _repulsion_integrals_from_molecule(molecule)
        ao = jnp.asarray(molecule.ao)
        weights = jnp.asarray(molecule.grid.weights)

        overlap = getattr(molecule, "overlap_matrix", None)
        if overlap is None:
            overlap = jnp.eye(h1e.shape[0], dtype=h1e.dtype)
        else:
            overlap = jnp.asarray(overlap)
        x = _orthogonalizer(overlap, self.config.orthogonalization_eps)

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

        level_shift = jnp.asarray(self.config.level_shift, dtype=h1e.dtype)
        has_level_shift = self.config.level_shift != 0.0

        def _raw_fock_from_density(
            density_spin: Array,
            mo_coeff_ref_spin: Array,
            mo_energy_ref_spin: Array,
        ) -> tuple[Array, Any]:
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
                resolved_xc,
                extra_fock_a,
                extra_fock_b,
            ) = _unrestricted_scf_xc_components(
                xc_params,
                xc_functional,
                molecule_iter,
                functional_dtype=h1e.dtype,
            )
            if resolved_xc is not None and _uses_dm21_local_hfx_correction(resolved_xc):
                raise NotImplementedError(
                    "Unrestricted differentiable SCF does not yet support Neural XC local-HFX corrections."
                )

            vxc_rho_a = jnp.nan_to_num(
                vxc_rho_a,
                nan=0.0,
                posinf=self.config.vxc_clip,
                neginf=-self.config.vxc_clip,
            )
            vxc_rho_b = jnp.nan_to_num(
                vxc_rho_b,
                nan=0.0,
                posinf=self.config.vxc_clip,
                neginf=-self.config.vxc_clip,
            )
            vxc_grad_a = jnp.nan_to_num(
                vxc_grad_a,
                nan=0.0,
                posinf=self.config.vxc_clip,
                neginf=-self.config.vxc_clip,
            )
            vxc_grad_b = jnp.nan_to_num(
                vxc_grad_b,
                nan=0.0,
                posinf=self.config.vxc_clip,
                neginf=-self.config.vxc_clip,
            )
            vxc_rho_a = jnp.clip(vxc_rho_a, -self.config.vxc_clip, self.config.vxc_clip)
            vxc_rho_b = jnp.clip(vxc_rho_b, -self.config.vxc_clip, self.config.vxc_clip)
            vxc_grad_a = jnp.clip(vxc_grad_a, -self.config.vxc_clip, self.config.vxc_clip)
            vxc_grad_b = jnp.clip(vxc_grad_b, -self.config.vxc_clip, self.config.vxc_clip)

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
            alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
            alpha = jnp.clip(alpha, 0.0, 1.0)
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
            return fock_spin, molecule_iter

        def body(
            carry: tuple[Array, Array, Array],
            _,
        ) -> tuple[tuple[Array, Array, Array], tuple[Array, Array, Array, Array]]:
            density_spin, mo_coeff_spin, mo_energy_spin = carry
            raw_fock_spin, _ = _raw_fock_from_density(density_spin, mo_coeff_spin, mo_energy_spin)
            fock_diag_spin = jax.lax.cond(
                jnp.asarray(has_level_shift),
                lambda operand: jax.vmap(
                    lambda fock, density: _apply_level_shift_spin(
                        fock,
                        overlap,
                        density,
                        level_shift,
                    )
                )(
                    operand[0],
                    operand[1],
                ),
                lambda operand: operand[0],
                operand=(raw_fock_spin, density_spin),
            )
            mo_energy_spin_new, mo_coeff_spin_new = jax.vmap(
                lambda fock_diag: _diagonalize_fock(
                    fock_diag,
                    x,
                    eigenvalue_jitter=self.config.eigenvalue_jitter,
                )
            )(fock_diag_spin)
            mo_energy_spin_new = jnp.nan_to_num(
                mo_energy_spin_new,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            mo_coeff_spin_new = jnp.nan_to_num(
                mo_coeff_spin_new,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            density_spin_new = _build_density_spin(mo_coeff_spin_new, mo_occ_spin_fixed)
            density_spin_new = jnp.nan_to_num(
                density_spin_new,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            density_spin_next = (
                (1.0 - self.config.damping) * density_spin_new + self.config.damping * density_spin
            )
            rms_density = _spin_density_rms(density_spin_next, density_spin)
            return (
                density_spin_next,
                mo_coeff_spin_new,
                mo_energy_spin_new,
            ), (
                density_spin_next,
                mo_coeff_spin_new,
                mo_energy_spin_new,
                rms_density,
            )

        carry0 = (
            density_spin0,
            mo_coeff_spin0,
            mo_energy_spin0,
        )
        (_, _, _), (
            density_history,
            mo_coeff_history,
            mo_energy_history,
            rms_history,
        ) = jax.lax.scan(
            body,
            carry0,
            xs=None,
            length=self.config.max_cycle,
        )

        best_idx = jnp.argmin(rms_history)
        converged_mask = rms_history < self.config.conv_tol_density
        converged = jnp.any(converged_mask)
        first_conv = jnp.argmax(converged_mask) + 1
        cycles = jnp.where(converged, first_conv, self.config.max_cycle)
        best_cycle = best_idx + 1
        converged_best_idx = jnp.argmin(
            jnp.where(
                converged_mask,
                rms_history,
                jnp.asarray(jnp.inf, dtype=rms_history.dtype),
            )
        )
        fallback_idx = jnp.asarray(self.config.max_cycle - 1)

        if self.config.iterate_selection == "best_rms":
            selected_idx = jnp.where(
                converged & bool(self.config.require_converged_iterates),
                converged_best_idx,
                best_idx,
            )
        elif self.config.iterate_selection == "first_converged":
            selected_idx = jnp.where(
                converged,
                first_conv - 1,
                fallback_idx if bool(self.config.require_converged_iterates) else best_idx,
            )
        else:
            selected_idx = fallback_idx

        if bool(self.config.require_converged_iterates):
            selected_idx = jnp.where(converged, selected_idx, fallback_idx)

        density_spin_selected = density_history[selected_idx]
        mo_coeff_spin_selected = mo_coeff_history[selected_idx]
        mo_energy_spin_selected = mo_energy_history[selected_idx]
        fock_spin_final, _ = _raw_fock_from_density(
            density_spin_selected,
            mo_coeff_spin_selected,
            mo_energy_spin_selected,
        )
        mo_energy_spin_final, mo_coeff_spin_final = jax.vmap(
            lambda fock_spin: _diagonalize_fock(
                fock_spin,
                x,
                eigenvalue_jitter=self.config.eigenvalue_jitter,
            )
        )(fock_spin_final)
        mo_energy_spin_final = jnp.nan_to_num(
            mo_energy_spin_final,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        mo_coeff_spin_final = jnp.nan_to_num(
            mo_coeff_spin_final,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        density_spin_final = _build_density_spin(mo_coeff_spin_final, mo_occ_spin_fixed)
        density_spin_final = jnp.nan_to_num(
            density_spin_final,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        molecule_final = _replace_molecule(
            molecule,
            rdm1=density_spin_final,
            mo_coeff=mo_coeff_spin_final,
            mo_occ=mo_occ_spin_fixed,
            mo_energy=mo_energy_spin_final,
        )

        info = DifferentiableSCFInfo(
            mode="self_consistent",
            converged=converged,
            cycles=cycles,
            final_rms_density=rms_history[-1],
            rms_density_history=rms_history,
            selected_cycle=selected_idx + 1,
            selected_rms_density=rms_history[selected_idx],
            best_cycle=best_cycle,
            best_rms_density=rms_history[best_idx],
        )
        return molecule_final, info

    def _full_scf_implicit_commutator_unrestricted(
        self,
        molecule: Any,
        xc_functional: Any,
        xc_params: PyTree,
    ) -> tuple[Any, DifferentiableSCFInfo]:
        forward_molecule, info = self._implicit_forward_state_unrestricted(
            molecule,
            xc_functional,
            xc_params,
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

        h1e = jnp.asarray(molecule.h1e)
        rep_tensor = _repulsion_integrals_from_molecule(molecule)
        ao = jnp.asarray(molecule.ao)
        weights = jnp.asarray(molecule.grid.weights)
        overlap = getattr(molecule, "overlap_matrix", None)
        if overlap is None:
            overlap = jnp.eye(h1e.shape[0], dtype=h1e.dtype)
        else:
            overlap = jnp.asarray(overlap)

        def _raw_fock_from_density(
            density_spin: Array,
            params_local: PyTree,
        ) -> Array:
            density_spin = jnp.nan_to_num(density_spin, nan=0.0, posinf=0.0, neginf=0.0)
            updates = dict(
                rdm1=density_spin,
                mo_coeff=mo_coeff_spin_ref,
                mo_occ=mo_occ_spin_fixed,
                mo_energy=mo_energy_spin_ref,
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
                resolved_xc,
                extra_fock_a,
                extra_fock_b,
            ) = _unrestricted_scf_xc_components(
                params_local,
                xc_functional,
                molecule_iter,
                functional_dtype=h1e.dtype,
            )
            if resolved_xc is not None and _uses_dm21_local_hfx_correction(resolved_xc):
                raise NotImplementedError(
                    "Unrestricted differentiable SCF does not yet support Neural XC local-HFX corrections."
                )

            vxc_rho_a = jnp.nan_to_num(
                vxc_rho_a,
                nan=0.0,
                posinf=self.config.vxc_clip,
                neginf=-self.config.vxc_clip,
            )
            vxc_rho_b = jnp.nan_to_num(
                vxc_rho_b,
                nan=0.0,
                posinf=self.config.vxc_clip,
                neginf=-self.config.vxc_clip,
            )
            vxc_grad_a = jnp.nan_to_num(
                vxc_grad_a,
                nan=0.0,
                posinf=self.config.vxc_clip,
                neginf=-self.config.vxc_clip,
            )
            vxc_grad_b = jnp.nan_to_num(
                vxc_grad_b,
                nan=0.0,
                posinf=self.config.vxc_clip,
                neginf=-self.config.vxc_clip,
            )
            vxc_rho_a = jnp.clip(vxc_rho_a, -self.config.vxc_clip, self.config.vxc_clip)
            vxc_rho_b = jnp.clip(vxc_rho_b, -self.config.vxc_clip, self.config.vxc_clip)
            vxc_grad_a = jnp.clip(vxc_grad_a, -self.config.vxc_clip, self.config.vxc_clip)
            vxc_grad_b = jnp.clip(vxc_grad_b, -self.config.vxc_clip, self.config.vxc_clip)

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
            alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
            alpha = jnp.clip(alpha, 0.0, 1.0)
            fock_alpha = h1e + j_mat - alpha * k_alpha + extra_fock_a + vxc_matrix_a
            fock_beta = h1e + j_mat - alpha * k_beta + extra_fock_b + vxc_matrix_b
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
        ) -> Array:
            density_spin = jnp.nan_to_num(density_spin, nan=0.0, posinf=0.0, neginf=0.0)
            fock_spin = _raw_fock_from_density(density_spin, params_local)
            residual_spin = jnp.stack(
                [
                    fock_spin[0] @ density_spin[0] @ overlap
                    - overlap @ density_spin[0] @ fock_spin[0],
                    fock_spin[1] @ density_spin[1] @ overlap
                    - overlap @ density_spin[1] @ fock_spin[1],
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

        @jax.custom_vjp
        def _density_from_params(params_local: PyTree) -> Array:
            return density_star_spin

        def _density_from_params_fwd(
            params_local: PyTree,
        ) -> tuple[Array, tuple[PyTree, Array]]:
            return density_star_spin, (params_local, density_star_spin)

        def _density_from_params_bwd(
            res: tuple[PyTree, Array],
            cotangent_density_spin: Array,
        ) -> tuple[PyTree]:
            params_local, density_fixed = res
            cotangent_density_spin = jnp.nan_to_num(
                cotangent_density_spin,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            cotangent_density_spin = jnp.clip(
                cotangent_density_spin,
                -self.config.implicit_diff_clip,
                self.config.implicit_diff_clip,
            )

            def _residual_wrt_density(density_var: Array) -> Array:
                return _residual(density_var, params_local)

            _, residual_density_vjp = jax.vjp(_residual_wrt_density, density_fixed)

            def _apply_a(vec: Array) -> Array:
                _, tangent = jax.jvp(
                    _residual_wrt_density,
                    (density_fixed,),
                    (vec,),
                )
                tangent = jnp.nan_to_num(tangent, nan=0.0, posinf=0.0, neginf=0.0)
                tangent = jnp.clip(
                    tangent,
                    -self.config.implicit_diff_clip,
                    self.config.implicit_diff_clip,
                )
                return tangent

            def _apply_at(vec: Array) -> Array:
                cot = residual_density_vjp(vec)[0]
                cot = jnp.nan_to_num(cot, nan=0.0, posinf=0.0, neginf=0.0)
                cot = jnp.clip(
                    cot,
                    -self.config.implicit_diff_clip,
                    self.config.implicit_diff_clip,
                )
                return cot

            reg = jnp.asarray(
                max(float(self.config.implicit_diff_regularization), 0.0),
                dtype=density_fixed.dtype,
            )
            tol = float(self.config.implicit_diff_tolerance)
            max_iter = max(1, int(self.config.implicit_diff_max_iter))
            restart = max(1, min(int(self.config.implicit_diff_restart), max_iter))
            restart_enabled = restart > 0
            rhs_density = jax.lax.stop_gradient(cotangent_density_spin)
            rhs_density = jnp.nan_to_num(rhs_density, nan=0.0, posinf=0.0, neginf=0.0)
            rhs_density = jnp.clip(
                rhs_density,
                -self.config.implicit_diff_clip,
                self.config.implicit_diff_clip,
            )

            def _normal_op(vec_flat: Array) -> Array:
                vec = vec_flat.reshape(density_fixed.shape)
                at_vec = _apply_at(vec)
                aat_vec = _apply_a(at_vec)
                out = aat_vec + reg * vec
                out = jnp.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
                out = jnp.clip(
                    out,
                    -self.config.implicit_diff_clip,
                    self.config.implicit_diff_clip,
                )
                return out.reshape(-1)

            rhs_normal = _apply_a(rhs_density).reshape(-1)
            rhs_normal = jax.lax.stop_gradient(rhs_normal)

            def _solve_normal_eq_cg(b_flat: Array) -> Array:
                x0 = jnp.zeros_like(b_flat)
                r0 = b_flat - _normal_op(x0)
                r0 = jnp.nan_to_num(r0, nan=0.0, posinf=0.0, neginf=0.0)
                p0 = r0
                rr0 = jnp.vdot(r0, r0).real
                b_norm = jnp.sqrt(jnp.maximum(jnp.vdot(b_flat, b_flat).real, 1e-30))
                tol_sq = jnp.asarray(tol, dtype=b_flat.dtype) ** 2 * jnp.maximum(
                    b_norm**2,
                    jnp.asarray(1.0, dtype=b_flat.dtype),
                )
                eps = jnp.asarray(1e-30, dtype=b_flat.dtype)

                def _body(
                    iter_idx: int,
                    carry: tuple[Array, Array, Array, Array, Array],
                ) -> tuple[Array, Array, Array, Array, Array]:
                    x, r, p, rr, done = carry

                    def _no_update(_: None) -> tuple[Array, Array, Array, Array, Array]:
                        return x, r, p, rr, done

                    def _do_update(_: None) -> tuple[Array, Array, Array, Array, Array]:
                        use_restart = jnp.asarray(restart_enabled) & (iter_idx > 0) & (
                            iter_idx % restart == 0
                        )
                        p_work = jnp.where(use_restart, r, p)
                        ap = _normal_op(p_work)
                        denom = jnp.vdot(p_work, ap).real
                        denom_safe = jnp.where(jnp.abs(denom) > eps, denom, eps)
                        alpha = rr / denom_safe
                        x_new = x + alpha * p_work
                        r_new = r - alpha * ap
                        r_new = jnp.nan_to_num(r_new, nan=0.0, posinf=0.0, neginf=0.0)
                        rr_new = jnp.vdot(r_new, r_new).real
                        beta = jnp.where(rr > eps, rr_new / jnp.maximum(rr, eps), 0.0)
                        p_new = r_new + beta * p_work
                        done_new = rr_new <= tol_sq
                        return x_new, r_new, p_new, rr_new, done_new

                    return jax.lax.cond(done, _no_update, _do_update, operand=None)

                x_final, _, _, _, _ = jax.lax.fori_loop(
                    0,
                    max_iter,
                    _body,
                    (x0, r0, p0, rr0, rr0 <= tol_sq),
                )
                return x_final

            solver_name = str(self.config.implicit_diff_solver)
            if solver_name not in {"normal_cg", "gmres", "bicgstab"}:
                raise ValueError(f"Unsupported implicit_diff_solver: {solver_name}")
            lambda_flat = _solve_normal_eq_cg(rhs_normal)
            lambda_flat = jax.lax.stop_gradient(lambda_flat)
            lambda_var = lambda_flat.reshape(density_fixed.shape)
            lambda_var = jnp.nan_to_num(lambda_var, nan=0.0, posinf=0.0, neginf=0.0)
            lambda_var = jnp.clip(
                lambda_var,
                -self.config.implicit_diff_clip,
                self.config.implicit_diff_clip,
            )

            def _residual_wrt_params(params_var: PyTree) -> Array:
                return _residual(density_fixed, params_var)

            _, vjp_params = jax.vjp(_residual_wrt_params, params_local)
            grad_params = vjp_params(lambda_var)[0]
            grad_params = jax.tree_util.tree_map(
                lambda x: -jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
                grad_params,
            )
            return (grad_params,)

        _density_from_params.defvjp(_density_from_params_fwd, _density_from_params_bwd)
        density_spin_implicit = _density_from_params(xc_params)
        molecule_final = _replace_molecule(
            forward_molecule,
            rdm1=density_spin_implicit,
        )
        return molecule_final, info

    def _full_scf_implicit_commutator(
        self,
        molecule: Any,
        xc_functional: Any,
        xc_params: PyTree,
    ) -> tuple[Any, DifferentiableSCFInfo]:
        forward_molecule, info = self._implicit_forward_state_restricted(
            molecule,
            xc_functional,
            xc_params,
        )

        density_star = _spin_summed_density_matrix(forward_molecule)
        density_star = jnp.nan_to_num(density_star, nan=0.0, posinf=0.0, neginf=0.0)
        mo_coeff_ref, _ = _restricted_channel(forward_molecule)
        mo_energy_raw = jnp.asarray(forward_molecule.mo_energy)
        mo_energy_ref = mo_energy_raw[0] if mo_energy_raw.ndim == 2 else mo_energy_raw
        mo_occ_total = _restricted_total_occupations(
            forward_molecule,
            occupation_tolerance=self.config.occupation_tolerance,
        )
        mo_occ_stacked = _restricted_stacked_occupations_from_total(mo_occ_total)

        h1e = jnp.asarray(molecule.h1e)
        jk_source = _jk_source_from_molecule(molecule)
        ao = jnp.asarray(molecule.ao)
        weights = jnp.asarray(molecule.grid.weights)
        overlap = getattr(molecule, "overlap_matrix", None)
        if overlap is None:
            overlap = jnp.eye(h1e.shape[0], dtype=h1e.dtype)
        else:
            overlap = jnp.asarray(overlap)

        hfx_nu = getattr(molecule, "hfx_nu", None)
        hfx_local_ref = getattr(forward_molecule, "hfx_local", None)
        mo_coeff_spin_ref = jnp.stack([mo_coeff_ref, mo_coeff_ref], axis=0)
        mo_energy_spin_ref = jnp.stack([mo_energy_ref, mo_energy_ref], axis=0)

        def _residual(density: Array, params_local: PyTree) -> Array:
            density = jnp.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
            density_spin = jnp.stack([0.5 * density, 0.5 * density], axis=0)
            updates = dict(
                rdm1=density_spin,
                mo_coeff=mo_coeff_spin_ref,
                mo_occ=mo_occ_stacked,
                mo_energy=mo_energy_spin_ref,
            )
            if hasattr(molecule, "hfx_local"):
                if hfx_nu is not None:
                    updates["hfx_local"] = _restricted_hfx_features_from_nu(
                        ao=ao,
                        density=density,
                        nu_cache=hfx_nu,
                    )
                elif hfx_local_ref is not None:
                    updates["hfx_local"] = jax.lax.stop_gradient(jnp.asarray(hfx_local_ref))
            molecule_iter = _replace_molecule(molecule, **updates)
            j_mat, k_mat = _coulomb_exchange_matrices_from_source(jk_source, density)
            vxc_rho, vxc_grad, vxc_tau, vxc_lapl, xc_kind, alpha, resolved_xc, vhf_matrix = _scf_xc_components(
                params_local,
                xc_functional,
                molecule_iter,
                functional_dtype=h1e.dtype,
            )
            vxc_rho = jnp.nan_to_num(
                vxc_rho,
                nan=0.0,
                posinf=self.config.vxc_clip,
                neginf=-self.config.vxc_clip,
            )
            vxc_grad = jnp.nan_to_num(
                vxc_grad,
                nan=0.0,
                posinf=self.config.vxc_clip,
                neginf=-self.config.vxc_clip,
            )
            vxc_rho = jnp.clip(vxc_rho, -self.config.vxc_clip, self.config.vxc_clip)
            vxc_grad = jnp.clip(vxc_grad, -self.config.vxc_clip, self.config.vxc_clip)
            vxc_matrix = _build_vxc_matrix_from_components(
                molecule=molecule_iter,
                weights=weights,
                v_rho=vxc_rho,
                v_grad=vxc_grad,
                v_tau=vxc_tau,
                v_lapl=vxc_lapl,
                xc_kind=xc_kind,
            )
            alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
            alpha = jnp.clip(alpha, 0.0, 1.0)
            if resolved_xc is not None and _uses_dm21_local_hfx_correction(resolved_xc):
                vhf_matrix = vhf_matrix + _dm21_local_hfx_fock_correction(
                    resolved_xc=resolved_xc,
                    molecule=molecule_iter,
                    ao=ao,
                    density=density,
                )
            fock = h1e + j_mat - 0.5 * alpha * k_mat + vxc_matrix + vhf_matrix
            fock = jnp.nan_to_num(fock, nan=0.0, posinf=0.0, neginf=0.0)
            fock = 0.5 * (fock + fock.T)
            residual = fock @ density @ overlap - overlap @ density @ fock
            residual = jnp.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
            residual = jnp.clip(
                residual,
                -self.config.implicit_diff_clip,
                self.config.implicit_diff_clip,
            )
            return residual

        @jax.custom_vjp
        def _density_from_params(params_local: PyTree) -> Array:
            return density_star

        def _density_from_params_fwd(params_local: PyTree) -> tuple[Array, tuple[PyTree, Array]]:
            return density_star, (params_local, density_star)

        def _density_from_params_bwd(
            res: tuple[PyTree, Array],
            cotangent_density: Array,
        ) -> tuple[PyTree]:
            params_local, density_fixed = res
            cotangent_density = jnp.nan_to_num(
                cotangent_density,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            cotangent_density = jnp.clip(
                cotangent_density,
                -self.config.implicit_diff_clip,
                self.config.implicit_diff_clip,
            )

            def _iterate_molecule_and_resolved(
                density_var: Array,
                params_var: PyTree,
            ) -> tuple[Any, Any | None]:
                density_var = jnp.nan_to_num(density_var, nan=0.0, posinf=0.0, neginf=0.0)
                density_spin = jnp.stack([0.5 * density_var, 0.5 * density_var], axis=0)
                updates = dict(
                    rdm1=density_spin,
                    mo_coeff=mo_coeff_spin_ref,
                    mo_occ=mo_occ_stacked,
                    mo_energy=mo_energy_spin_ref,
                )
                if hasattr(molecule, "hfx_local"):
                    if hfx_nu is not None:
                        updates["hfx_local"] = _restricted_hfx_features_from_nu(
                            ao=ao,
                            density=density_var,
                            nu_cache=hfx_nu,
                        )
                    elif hfx_local_ref is not None:
                        updates["hfx_local"] = jax.lax.stop_gradient(jnp.asarray(hfx_local_ref))
                molecule_iter = _replace_molecule(molecule, **updates)
                return molecule_iter, None

            def _full_fock_from_density(
                density_var: Array,
                params_var: PyTree,
            ) -> tuple[Array, Array, Array]:
                molecule_iter, _ = _iterate_molecule_and_resolved(
                    density_var,
                    params_var,
                )
                j_mat, k_mat = _coulomb_exchange_matrices_from_source(jk_source, density_var)
                vxc_rho, vxc_grad, vxc_tau, vxc_lapl, xc_kind, alpha, resolved_xc, vhf_matrix = _scf_xc_components(
                    params_var,
                    xc_functional,
                    molecule_iter,
                    functional_dtype=h1e.dtype,
                )
                vxc_rho = jnp.nan_to_num(
                    vxc_rho,
                    nan=0.0,
                    posinf=self.config.vxc_clip,
                    neginf=-self.config.vxc_clip,
                )
                vxc_grad = jnp.nan_to_num(
                    vxc_grad,
                    nan=0.0,
                    posinf=self.config.vxc_clip,
                    neginf=-self.config.vxc_clip,
                )
                vxc_rho = jnp.clip(vxc_rho, -self.config.vxc_clip, self.config.vxc_clip)
                vxc_grad = jnp.clip(vxc_grad, -self.config.vxc_clip, self.config.vxc_clip)
                vxc_matrix = _build_vxc_matrix_from_components(
                    molecule=molecule_iter,
                    weights=weights,
                    v_rho=vxc_rho,
                    v_grad=vxc_grad,
                    v_tau=vxc_tau,
                    v_lapl=vxc_lapl,
                    xc_kind=xc_kind,
                )
                alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
                alpha = jnp.clip(alpha, 0.0, 1.0)
                if resolved_xc is not None and _uses_dm21_local_hfx_correction(resolved_xc):
                    vhf_matrix = vhf_matrix + _dm21_local_hfx_fock_correction(
                        resolved_xc=resolved_xc,
                        molecule=molecule_iter,
                        ao=ao,
                        density=density_var,
                    )
                fock = h1e + j_mat - 0.5 * alpha * k_mat + vxc_matrix + vhf_matrix
                fock = jnp.nan_to_num(fock, nan=0.0, posinf=0.0, neginf=0.0)
                fock = 0.5 * (fock + fock.T)
                return fock, alpha, k_mat

            fock_fixed, alpha_fixed, k_fixed = _full_fock_from_density(density_fixed, params_local)
            k_fixed = jax.lax.stop_gradient(k_fixed)
            density_ds_fixed = density_fixed @ overlap
            sd_density_fixed = overlap @ density_fixed

            def _linear_fock_from_density(density_var: Array) -> Array:
                j_mat, k_mat = _coulomb_exchange_matrices_from_source(jk_source, density_var)
                fock = j_mat - 0.5 * alpha_fixed * k_mat
                fock = jnp.nan_to_num(fock, nan=0.0, posinf=0.0, neginf=0.0)
                return 0.5 * (fock + fock.T)

            def _linear_fock_adjoint(cotangent_fock: Array) -> Array:
                cotangent_fock = 0.5 * (cotangent_fock + cotangent_fock.T)
                return _linear_fock_from_density(cotangent_fock)

            def _nonlinear_fock_from_density(density_var: Array) -> Array:
                molecule_iter, _ = _iterate_molecule_and_resolved(
                    density_var,
                    params_local,
                )
                vxc_rho, vxc_grad, vxc_tau, vxc_lapl, xc_kind, alpha, resolved_xc, vhf_matrix = _scf_xc_components(
                    params_local,
                    xc_functional,
                    molecule_iter,
                    functional_dtype=h1e.dtype,
                )
                vxc_rho = jnp.nan_to_num(
                    vxc_rho,
                    nan=0.0,
                    posinf=self.config.vxc_clip,
                    neginf=-self.config.vxc_clip,
                )
                vxc_grad = jnp.nan_to_num(
                    vxc_grad,
                    nan=0.0,
                    posinf=self.config.vxc_clip,
                    neginf=-self.config.vxc_clip,
                )
                vxc_rho = jnp.clip(vxc_rho, -self.config.vxc_clip, self.config.vxc_clip)
                vxc_grad = jnp.clip(vxc_grad, -self.config.vxc_clip, self.config.vxc_clip)
                vxc_matrix = _build_vxc_matrix_from_components(
                    molecule=molecule_iter,
                    weights=weights,
                    v_rho=vxc_rho,
                    v_grad=vxc_grad,
                    v_tau=vxc_tau,
                    v_lapl=vxc_lapl,
                    xc_kind=xc_kind,
                )
                alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
                alpha = jnp.clip(alpha, 0.0, 1.0)
                if resolved_xc is not None and _uses_dm21_local_hfx_correction(resolved_xc):
                    vhf_matrix = vhf_matrix + _dm21_local_hfx_fock_correction(
                        resolved_xc=resolved_xc,
                        molecule=molecule_iter,
                        ao=ao,
                        density=density_var,
                    )
                fock = h1e - 0.5 * (alpha - alpha_fixed) * k_fixed + vxc_matrix + vhf_matrix
                fock = jnp.nan_to_num(fock, nan=0.0, posinf=0.0, neginf=0.0)
                return 0.5 * (fock + fock.T)

            _, nonlinear_fock_vjp = jax.vjp(_nonlinear_fock_from_density, density_fixed)

            def _apply_a(vec: Array) -> Array:
                linear_fock_tangent = _linear_fock_from_density(vec)
                _, nonlinear_fock_tangent = jax.jvp(
                    _nonlinear_fock_from_density,
                    (density_fixed,),
                    (vec,),
                )
                delta_fock = linear_fock_tangent + nonlinear_fock_tangent
                tangent = (
                    fock_fixed @ vec @ overlap
                    - overlap @ vec @ fock_fixed
                    + delta_fock @ density_ds_fixed
                    - sd_density_fixed @ delta_fock
                )
                tangent = jnp.nan_to_num(tangent, nan=0.0, posinf=0.0, neginf=0.0)
                tangent = jnp.clip(
                    tangent,
                    -self.config.implicit_diff_clip,
                    self.config.implicit_diff_clip,
                )
                return tangent

            def _apply_at(vec: Array) -> Array:
                cot_explicit = (
                    fock_fixed.T @ vec @ overlap.T
                    - overlap.T @ vec @ fock_fixed.T
                )
                cot_fock = vec @ sd_density_fixed - density_ds_fixed @ vec
                cot_linear = _linear_fock_adjoint(cot_fock)
                cot_nonlinear = nonlinear_fock_vjp(cot_fock)[0]
                cot = cot_explicit + cot_linear + cot_nonlinear
                cot = jnp.nan_to_num(cot, nan=0.0, posinf=0.0, neginf=0.0)
                cot = jnp.clip(
                    cot,
                    -self.config.implicit_diff_clip,
                    self.config.implicit_diff_clip,
                )
                return cot

            reg = jnp.asarray(
                max(float(self.config.implicit_diff_regularization), 0.0),
                dtype=density_fixed.dtype,
            )
            tol = float(self.config.implicit_diff_tolerance)
            max_iter = max(1, int(self.config.implicit_diff_max_iter))
            restart = max(1, min(int(self.config.implicit_diff_restart), max_iter))
            restart_enabled = restart > 0
            rhs_density = jax.lax.stop_gradient(cotangent_density)
            rhs_density = jnp.nan_to_num(rhs_density, nan=0.0, posinf=0.0, neginf=0.0)
            rhs_density = jnp.clip(
                rhs_density,
                -self.config.implicit_diff_clip,
                self.config.implicit_diff_clip,
            )

            def _normal_op(vec_flat: Array) -> Array:
                vec = vec_flat.reshape(density_fixed.shape)
                at_vec = _apply_at(vec)
                aat_vec = _apply_a(at_vec)
                out = aat_vec + reg * vec
                out = jnp.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
                out = jnp.clip(
                    out,
                    -self.config.implicit_diff_clip,
                    self.config.implicit_diff_clip,
                )
                return out.reshape(-1)

            rhs_normal = _apply_a(rhs_density).reshape(-1)
            rhs_normal = jax.lax.stop_gradient(rhs_normal)

            def _solve_normal_eq_cg(b_flat: Array) -> Array:
                x0 = jnp.zeros_like(b_flat)
                r0 = b_flat - _normal_op(x0)
                r0 = jnp.nan_to_num(r0, nan=0.0, posinf=0.0, neginf=0.0)
                p0 = r0
                rr0 = jnp.vdot(r0, r0).real
                b_norm = jnp.sqrt(jnp.maximum(jnp.vdot(b_flat, b_flat).real, 1e-30))
                tol_sq = jnp.asarray(tol, dtype=b_flat.dtype) ** 2 * jnp.maximum(
                    b_norm**2,
                    jnp.asarray(1.0, dtype=b_flat.dtype),
                )
                eps = jnp.asarray(1e-30, dtype=b_flat.dtype)

                def _body(iter_idx: int, carry: tuple[Array, Array, Array, Array, Array]) -> tuple[Array, Array, Array, Array, Array]:
                    x, r, p, rr, done = carry

                    def _no_update(_: None) -> tuple[Array, Array, Array, Array, Array]:
                        return x, r, p, rr, done

                    def _do_update(_: None) -> tuple[Array, Array, Array, Array, Array]:
                        use_restart = jnp.asarray(restart_enabled) & (iter_idx > 0) & (
                            iter_idx % restart == 0
                        )
                        p_work = jnp.where(use_restart, r, p)
                        ap = _normal_op(p_work)
                        denom = jnp.vdot(p_work, ap).real
                        denom_safe = jnp.where(jnp.abs(denom) > eps, denom, eps)
                        alpha = rr / denom_safe
                        x_new = x + alpha * p_work
                        r_new = r - alpha * ap
                        r_new = jnp.nan_to_num(r_new, nan=0.0, posinf=0.0, neginf=0.0)
                        rr_new = jnp.vdot(r_new, r_new).real
                        beta = jnp.where(rr > eps, rr_new / jnp.maximum(rr, eps), 0.0)
                        p_new = r_new + beta * p_work
                        done_new = rr_new <= tol_sq
                        return x_new, r_new, p_new, rr_new, done_new

                    return jax.lax.cond(done, _no_update, _do_update, operand=None)

                x_final, _, _, _, _ = jax.lax.fori_loop(
                    0,
                    max_iter,
                    _body,
                    (x0, r0, p0, rr0, rr0 <= tol_sq),
                )
                return x_final

            solver_name = str(self.config.implicit_diff_solver)
            if solver_name not in {"normal_cg", "gmres", "bicgstab"}:
                raise ValueError(f"Unsupported implicit_diff_solver: {solver_name}")
            lambda_flat = _solve_normal_eq_cg(rhs_normal)
            lambda_flat = jax.lax.stop_gradient(lambda_flat)
            lambda_var = lambda_flat.reshape(density_fixed.shape)
            lambda_var = jnp.nan_to_num(lambda_var, nan=0.0, posinf=0.0, neginf=0.0)
            lambda_var = jnp.clip(
                lambda_var,
                -self.config.implicit_diff_clip,
                self.config.implicit_diff_clip,
            )

            def _residual_wrt_params(params_var: PyTree) -> Array:
                return _residual(density_fixed, params_var)

            _, vjp_params = jax.vjp(_residual_wrt_params, params_local)
            grad_params = vjp_params(lambda_var)[0]
            grad_params = jax.tree_util.tree_map(
                lambda x: -jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
                grad_params,
            )
            return (grad_params,)

        _density_from_params.defvjp(_density_from_params_fwd, _density_from_params_bwd)
        density_implicit = _density_from_params(xc_params)

        density_spin_final = jnp.stack([0.5 * density_implicit, 0.5 * density_implicit], axis=0)
        molecule_final = _replace_molecule(
            forward_molecule,
            rdm1=density_spin_final,
        )
        return molecule_final, info
