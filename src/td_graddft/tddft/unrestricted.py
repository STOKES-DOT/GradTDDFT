from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Any
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.lax import Precision
from jaxtyping import Array

from .cisd import unrestricted_cisd_second_order_correction
from .eigensolvers import (
    davidson_lowest_tdhf,
    davidson_lowest_symmetric,
    _solver_dtype,
)
from ._utils import (
    _resolve_xc_functional,
    _transition_densities_on_grid,
)
from .response import _needs_exchange_terms


def _pytree_dataclass(cls):
    def tree_flatten(self):
        children = tuple(getattr(self, field.name) for field in fields(self))
        return children, None

    @classmethod
    def tree_unflatten(cls_, aux_data, children):
        del aux_data
        return cls_(*children)

    cls.tree_flatten = tree_flatten
    cls.tree_unflatten = tree_unflatten
    return jax.tree_util.register_pytree_node_class(cls)


@_pytree_dataclass
@dataclass(frozen=True)
class UnrestrictedTDAResult:
    """Excitation energies and alpha/beta amplitudes from unrestricted TDA."""

    excitation_energies: Array
    amplitudes_alpha: Array
    amplitudes_beta: Array
    posthoc_correction: Array | None = None


@_pytree_dataclass
@dataclass(frozen=True)
class UnrestrictedTDDFTResult:
    """Excitation energies and (X, Y) amplitudes from unrestricted Casida TDDFT."""

    excitation_energies: Array
    x_amplitudes_alpha: Array
    x_amplitudes_beta: Array
    y_amplitudes_alpha: Array
    y_amplitudes_beta: Array
    posthoc_correction: Array | None = None


@dataclass(frozen=True)
class _UnrestrictedResponseOperatorData:
    orbital_energy_differences_alpha: Array
    orbital_energy_differences_beta: Array
    orbo_alpha: Array
    orbv_alpha: Array
    orbo_beta: Array
    orbv_beta: Array
    ao_response_action_fn: Callable[[Array, Array], tuple[Array, Array]]
    xc_response_action_fn: Callable[[Array, Array], tuple[Array, Array]] | None = None


def _unrestricted_orbital_data(
    molecule: Any,
    occupation_tolerance: float,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    mo_coeff = jnp.asarray(molecule.mo_coeff)
    mo_occ = jnp.asarray(molecule.mo_occ)
    mo_energy = jnp.asarray(molecule.mo_energy)
    if mo_coeff.ndim != 3 or mo_coeff.shape[0] != 2:
        raise NotImplementedError(
            "Expected unrestricted orbitals with shape (2, nao, nmo)."
        )

    nmo = int(mo_coeff.shape[-1])

    def _channel_partition(
        spin: int,
        *,
        nocc_hint: Any | None,
    ) -> tuple[Array, Array, Array]:
        if nocc_hint is not None:
            nocc = int(nocc_hint)
            nocc = max(0, min(nocc, nmo))
            occ_idx = jnp.arange(nocc)
            vir_idx = jnp.arange(nocc, nmo)
        else:
            occ_values = mo_occ[spin]
            if isinstance(occ_values, jax.core.Tracer):
                raise ValueError(
                    "JIT-traced unrestricted TDDFT/TDA requires static "
                    "nocc_alpha/nocc_beta on the molecule."
                )
            host_occ = np.asarray(jax.device_get(occ_values))
            occ_idx = jnp.asarray(np.where(host_occ > occupation_tolerance)[0], dtype=jnp.int32)
            vir_idx = jnp.asarray(np.where(host_occ <= occupation_tolerance)[0], dtype=jnp.int32)
        orbo = mo_coeff[spin][:, occ_idx]
        orbv = mo_coeff[spin][:, vir_idx]
        de = mo_energy[spin][vir_idx][None, :] - mo_energy[spin][occ_idx][:, None]
        return orbo, orbv, de

    orbo_a, orbv_a, de_a = _channel_partition(
        0,
        nocc_hint=getattr(molecule, "nocc_alpha", None),
    )
    orbo_b, orbv_b, de_b = _channel_partition(
        1,
        nocc_hint=getattr(molecule, "nocc_beta", None),
    )
    has_alpha_channel = de_a.shape[0] > 0 and de_a.shape[1] > 0
    has_beta_channel = de_b.shape[0] > 0 and de_b.shape[1] > 0
    if not (has_alpha_channel or has_beta_channel):
        raise ValueError(
            "Need at least one occupied-virtual excitation channel across alpha/beta spins."
        )
    return orbo_a, orbv_a, orbo_b, orbv_b, de_a, de_b


def _spin_densities_on_grid(molecule: Any) -> tuple[Array, Array]:
    ao = jnp.asarray(molecule.ao)
    rdm1 = jnp.asarray(molecule.rdm1)
    if rdm1.ndim != 3 or rdm1.shape[0] != 2:
        raise ValueError("Unrestricted response requires spin-resolved rdm1 with shape (2, nao, nao).")
    rho_a = jnp.einsum("pq,rp,rq->r", rdm1[0], ao, ao, precision=Precision.HIGHEST)
    rho_b = jnp.einsum("pq,rp,rq->r", rdm1[1], ao, ao, precision=Precision.HIGHEST)
    return rho_a, rho_b


def _normalize_spin_kernel_values(raw_kernel: Any, dtype: Any) -> tuple[Array, Array, Array]:
    if isinstance(raw_kernel, (tuple, list)):
        if len(raw_kernel) != 3:
            raise ValueError("spin kernel tuple/list must contain (f_aa, f_ab, f_bb).")
        f_aa, f_ab, f_bb = raw_kernel
    else:
        kernel = jnp.asarray(raw_kernel, dtype=dtype)
        if kernel.ndim == 1:
            return kernel, kernel, kernel
        if kernel.ndim == 2 and kernel.shape[-1] == 3:
            return kernel[:, 0], kernel[:, 1], kernel[:, 2]
        if kernel.ndim == 3 and kernel.shape[-2:] == (2, 2):
            f_aa = kernel[:, 0, 0]
            f_ab = 0.5 * (kernel[:, 0, 1] + kernel[:, 1, 0])
            f_bb = kernel[:, 1, 1]
            return f_aa, f_ab, f_bb
        raise ValueError(
            "Unsupported spin-kernel format. Expected 1D scalar kernel, "
            "(ngrids, 3), (ngrids, 2, 2), or tuple/list (f_aa, f_ab, f_bb)."
        )
    return (
        jnp.asarray(f_aa, dtype=dtype),
        jnp.asarray(f_ab, dtype=dtype),
        jnp.asarray(f_bb, dtype=dtype),
    )


def _spin_resolved_kernel_on_grid(
    molecule: Any,
    resolved_xc: Any,
    density_alpha: Array,
    density_beta: Array,
    dtype: Any,
) -> tuple[Array, Array, Array]:
    spin_grid_kernel = getattr(resolved_xc, "spin_grid_kernel", None)
    if callable(spin_grid_kernel):
        raw_kernel = spin_grid_kernel(molecule)
    else:
        spin_local_kernel = getattr(resolved_xc, "spin_local_kernel", None)
        if callable(spin_local_kernel):
            raw_kernel = spin_local_kernel(density_alpha, density_beta)
        else:
            raise ValueError(
                "Unrestricted strict TDDFT response requires spin-resolved XC kernels "
                "(spin_grid_kernel or spin_local_kernel). Scalar local_kernel/grid_kernel "
                "fallback is an approximation and is disabled."
            )
    return _normalize_spin_kernel_values(raw_kernel, dtype=dtype)


def _unrestricted_transition_density(orbo: Array, orbv: Array, values: Array, *, bottom: bool) -> Array:
    values = jnp.asarray(values)
    if bottom:
        return jnp.einsum(
            "nia,pi,qa->npq",
            values,
            orbo,
            orbv,
            precision=Precision.HIGHEST,
        )
    return jnp.einsum(
        "nia,pa,qi->npq",
        values,
        orbv,
        orbo,
        precision=Precision.HIGHEST,
    )


def _unrestricted_project_response(
    response_ao: Array,
    orbo: Array,
    orbv: Array,
    *,
    bottom: bool,
) -> Array:
    if bottom:
        return jnp.einsum(
            "npq,pi,qa->nia",
            response_ao,
            orbo,
            orbv,
            precision=Precision.HIGHEST,
        )
    return jnp.einsum(
        "npq,qi,pa->nia",
        response_ao,
        orbo,
        orbv,
        precision=Precision.HIGHEST,
    )


def _jk_from_full_eri(eri: Array, density: Array) -> tuple[Array, Array]:
    j_mat = jnp.einsum("pqrs,nrs->npq", eri, density, precision=Precision.HIGHEST)
    k_mat = jnp.einsum("prqs,nrs->npq", eri, density, precision=Precision.HIGHEST)
    return j_mat, k_mat


def _j_from_full_eri(eri: Array, density: Array) -> Array:
    return jnp.einsum("pqrs,nrs->npq", eri, density, precision=Precision.HIGHEST)


def _unrestricted_ao_response_action(
    molecule: Any,
    hybrid_fraction: Array,
    *,
    include_exchange: bool,
    dtype: Any,
) -> Callable[[Array, Array], tuple[Array, Array]]:
    if getattr(molecule, "rep_tensor", None) is None:
        raise ValueError("The molecule must provide rep_tensor for Hartree/exchange response.")
    eri = jnp.asarray(molecule.rep_tensor, dtype=dtype)
    alpha = jnp.asarray(hybrid_fraction, dtype=dtype)

    def action(density_alpha: Array, density_beta: Array) -> tuple[Array, Array]:
        density_alpha = jnp.asarray(density_alpha, dtype=dtype)
        density_beta = jnp.asarray(density_beta, dtype=dtype)
        if not include_exchange:
            j_total = _j_from_full_eri(eri, density_alpha) + _j_from_full_eri(
                eri,
                density_beta,
            )
            return j_total, j_total
        j_alpha, k_alpha = _jk_from_full_eri(eri, density_alpha)
        j_beta, k_beta = _jk_from_full_eri(eri, density_beta)
        j_total = j_alpha + j_beta
        return j_total - alpha * k_alpha, j_total - alpha * k_beta

    return action


def _unrestricted_grid_xc_response_action(
    molecule: Any,
    resolved_xc: Any | None,
    orbo_a: Array,
    orbv_a: Array,
    orbo_b: Array,
    orbv_b: Array,
    *,
    dtype: Any,
) -> Callable[[Array, Array], tuple[Array, Array]] | None:
    if resolved_xc is None:
        return None
    ao = jnp.asarray(molecule.ao)
    weights = jnp.asarray(molecule.grid.weights, dtype=dtype)
    rho_a, rho_b = _spin_densities_on_grid(molecule)
    f_aa, f_ab, f_bb = _spin_resolved_kernel_on_grid(
        molecule,
        resolved_xc,
        rho_a,
        rho_b,
        dtype,
    )
    weighted_f_aa = weights * f_aa
    weighted_f_ab = weights * f_ab
    weighted_f_bb = weights * f_bb
    rho_ov_a = _transition_densities_on_grid(ao, orbo_a, orbv_a)
    rho_ov_b = _transition_densities_on_grid(ao, orbo_b, orbv_b)

    def action(alpha: Array, beta: Array) -> tuple[Array, Array]:
        proj_a = jnp.einsum("ria,nia->nr", rho_ov_a, alpha, precision=Precision.HIGHEST)
        proj_b = jnp.einsum("ria,nia->nr", rho_ov_b, beta, precision=Precision.HIGHEST)
        pot_a = weighted_f_aa[None, :] * proj_a + weighted_f_ab[None, :] * proj_b
        pot_b = weighted_f_ab[None, :] * proj_a + weighted_f_bb[None, :] * proj_b
        out_a = jnp.einsum("ria,nr->nia", rho_ov_a, pot_a, precision=Precision.HIGHEST)
        out_b = jnp.einsum("ria,nr->nia", rho_ov_b, pot_b, precision=Precision.HIGHEST)
        return out_a, out_b

    return action


def _build_unrestricted_response_operator_data(
    molecule: Any,
    xc_functional: Any | None = None,
    *,
    xc_params: Any | None = None,
    occupation_tolerance: float = 1e-8,
) -> _UnrestrictedResponseOperatorData:
    resolved_xc = _resolve_xc_functional(molecule, xc_functional, xc_params)
    orbo_a, orbv_a, orbo_b, orbv_b, de_a, de_b = _unrestricted_orbital_data(
        molecule,
        occupation_tolerance,
    )
    hybrid_fraction_raw = getattr(molecule, "exact_exchange_fraction", 0.0)
    hybrid_fraction: Any = (
        float(hybrid_fraction_raw)
        if isinstance(hybrid_fraction_raw, (int, float, np.number))
        else jnp.asarray(hybrid_fraction_raw, dtype=jnp.result_type(de_a, de_b))
    )
    if resolved_xc is not None:
        hybrid_fraction_raw = getattr(resolved_xc, "exact_exchange_fraction", hybrid_fraction)
        hybrid_fraction = (
            float(hybrid_fraction_raw)
            if isinstance(hybrid_fraction_raw, (int, float, np.number))
            else jnp.asarray(hybrid_fraction_raw, dtype=jnp.result_type(de_a, de_b))
        )
    ao_response_action_fn = _unrestricted_ao_response_action(
        molecule,
        hybrid_fraction,
        include_exchange=_needs_exchange_terms(hybrid_fraction),
        dtype=jnp.result_type(de_a, de_b),
    )
    xc_response_action_fn = _unrestricted_grid_xc_response_action(
        molecule,
        resolved_xc,
        orbo_a,
        orbv_a,
        orbo_b,
        orbv_b,
        dtype=jnp.result_type(de_a, de_b),
    )
    return _UnrestrictedResponseOperatorData(
        orbital_energy_differences_alpha=de_a,
        orbital_energy_differences_beta=de_b,
        orbo_alpha=orbo_a,
        orbv_alpha=orbv_a,
        orbo_beta=orbo_b,
        orbv_beta=orbv_b,
        ao_response_action_fn=ao_response_action_fn,
        xc_response_action_fn=xc_response_action_fn,
    )


def _unrestricted_dimensions(
    data: _UnrestrictedResponseOperatorData,
) -> tuple[int, int, int, int, int]:
    nocca, nvira = data.orbital_energy_differences_alpha.shape
    noccb, nvirb = data.orbital_energy_differences_beta.shape
    naa = int(nocca * nvira)
    nbb = int(noccb * nvirb)
    return int(nocca), int(nvira), int(noccb), int(nvirb), int(naa + nbb)


def _split_unrestricted_rows(
    data: _UnrestrictedResponseOperatorData,
    rows: Array,
) -> tuple[Array, Array]:
    nocca, nvira, noccb, nvirb, dim = _unrestricted_dimensions(data)
    rows = jnp.asarray(rows).reshape(-1, dim)
    batch = int(rows.shape[0])
    naa = int(nocca * nvira)
    alpha = rows[:, :naa].reshape(batch, nocca, nvira)
    beta = rows[:, naa:].reshape(batch, noccb, nvirb)
    return alpha, beta


def _join_unrestricted_rows(alpha: Array, beta: Array) -> Array:
    batch = int(alpha.shape[0])
    return jnp.concatenate(
        [
            alpha.reshape(batch, -1),
            beta.reshape(batch, -1),
        ],
        axis=-1,
    )


def _unrestricted_xc_action(
    data: _UnrestrictedResponseOperatorData,
    alpha: Array,
    beta: Array,
) -> tuple[Array, Array]:
    if data.xc_response_action_fn is None:
        return jnp.zeros_like(alpha), jnp.zeros_like(beta)
    return data.xc_response_action_fn(alpha, beta)


def _unrestricted_ao_mo_action(
    data: _UnrestrictedResponseOperatorData,
    alpha: Array,
    beta: Array,
    *,
    bottom_density: bool,
    bottom_projection: bool,
) -> tuple[Array, Array]:
    density_alpha = _unrestricted_transition_density(
        data.orbo_alpha,
        data.orbv_alpha,
        alpha,
        bottom=bottom_density,
    )
    density_beta = _unrestricted_transition_density(
        data.orbo_beta,
        data.orbv_beta,
        beta,
        bottom=bottom_density,
    )
    response_alpha, response_beta = data.ao_response_action_fn(density_alpha, density_beta)
    out_alpha = _unrestricted_project_response(
        response_alpha,
        data.orbo_alpha,
        data.orbv_alpha,
        bottom=bottom_projection,
    )
    out_beta = _unrestricted_project_response(
        response_beta,
        data.orbo_beta,
        data.orbv_beta,
        bottom=bottom_projection,
    )
    return out_alpha, out_beta


def _unrestricted_a_action(
    data: _UnrestrictedResponseOperatorData,
    rows: Array,
) -> Array:
    alpha, beta = _split_unrestricted_rows(data, rows)
    out_alpha, out_beta = _unrestricted_ao_mo_action(
        data,
        alpha,
        beta,
        bottom_density=False,
        bottom_projection=False,
    )
    xc_alpha, xc_beta = _unrestricted_xc_action(data, alpha, beta)
    out_alpha = out_alpha + xc_alpha + alpha * data.orbital_energy_differences_alpha[None, :, :]
    out_beta = out_beta + xc_beta + beta * data.orbital_energy_differences_beta[None, :, :]
    return _join_unrestricted_rows(out_alpha, out_beta)


def _unrestricted_b_action(
    data: _UnrestrictedResponseOperatorData,
    rows: Array,
) -> Array:
    alpha, beta = _split_unrestricted_rows(data, rows)
    out_alpha, out_beta = _unrestricted_ao_mo_action(
        data,
        alpha,
        beta,
        bottom_density=True,
        bottom_projection=False,
    )
    xc_alpha, xc_beta = _unrestricted_xc_action(data, alpha, beta)
    out_alpha = out_alpha + xc_alpha
    out_beta = out_beta + xc_beta
    return _join_unrestricted_rows(out_alpha, out_beta)


def _unrestricted_tda_diagonal(data: _UnrestrictedResponseOperatorData) -> Array:
    return jnp.concatenate(
        [
            data.orbital_energy_differences_alpha.reshape(-1),
            data.orbital_energy_differences_beta.reshape(-1),
        ],
        axis=0,
    )


def _unrestricted_tdhf_action(
    data: _UnrestrictedResponseOperatorData,
    x_rows: Array,
    y_rows: Array,
) -> tuple[Array, Array]:
    x_alpha, x_beta = _split_unrestricted_rows(data, x_rows)
    y_alpha, y_beta = _split_unrestricted_rows(data, y_rows)
    density_alpha = _unrestricted_transition_density(
        data.orbo_alpha,
        data.orbv_alpha,
        x_alpha,
        bottom=False,
    ) + _unrestricted_transition_density(
        data.orbo_alpha,
        data.orbv_alpha,
        y_alpha,
        bottom=True,
    )
    density_beta = _unrestricted_transition_density(
        data.orbo_beta,
        data.orbv_beta,
        x_beta,
        bottom=False,
    ) + _unrestricted_transition_density(
        data.orbo_beta,
        data.orbv_beta,
        y_beta,
        bottom=True,
    )
    response_alpha, response_beta = data.ao_response_action_fn(density_alpha, density_beta)
    upper_alpha = _unrestricted_project_response(
        response_alpha,
        data.orbo_alpha,
        data.orbv_alpha,
        bottom=False,
    )
    upper_beta = _unrestricted_project_response(
        response_beta,
        data.orbo_beta,
        data.orbv_beta,
        bottom=False,
    )
    lower_alpha = _unrestricted_project_response(
        response_alpha,
        data.orbo_alpha,
        data.orbv_alpha,
        bottom=True,
    )
    lower_beta = _unrestricted_project_response(
        response_beta,
        data.orbo_beta,
        data.orbv_beta,
        bottom=True,
    )
    xc_alpha, xc_beta = _unrestricted_xc_action(
        data,
        x_alpha + y_alpha,
        x_beta + y_beta,
    )
    upper_alpha = upper_alpha + xc_alpha + x_alpha * data.orbital_energy_differences_alpha[None, :, :]
    upper_beta = upper_beta + xc_beta + x_beta * data.orbital_energy_differences_beta[None, :, :]
    lower_alpha = lower_alpha + xc_alpha + y_alpha * data.orbital_energy_differences_alpha[None, :, :]
    lower_beta = lower_beta + xc_beta + y_beta * data.orbital_energy_differences_beta[None, :, :]
    return _join_unrestricted_rows(upper_alpha, upper_beta), _join_unrestricted_rows(
        lower_alpha,
        lower_beta,
    )


def build_unrestricted_tda_operator(
    molecule: Any,
    xc_functional: Any | None = None,
    *,
    xc_params: Any | None = None,
    occupation_tolerance: float = 1e-8,
) -> tuple[Callable[[Array], Array], Array, Array, Array]:
    data = _build_unrestricted_response_operator_data(
        molecule,
        xc_functional,
        xc_params=xc_params,
        occupation_tolerance=occupation_tolerance,
    )
    diagonal = _unrestricted_tda_diagonal(data)

    def vind(rows: Array) -> Array:
        return _unrestricted_a_action(data, rows)

    return (
        vind,
        diagonal,
        data.orbital_energy_differences_alpha,
        data.orbital_energy_differences_beta,
    )


def build_unrestricted_tdhf_operator(
    molecule: Any,
    xc_functional: Any | None = None,
    *,
    xc_params: Any | None = None,
    occupation_tolerance: float = 1e-8,
) -> tuple[Callable[[Array], Array], Array, Array]:
    data = _build_unrestricted_response_operator_data(
        molecule,
        xc_functional,
        xc_params=xc_params,
        occupation_tolerance=occupation_tolerance,
    )
    _, _, _, _, dim = _unrestricted_dimensions(data)

    def vind(rows: Array) -> Array:
        rows = jnp.asarray(rows).reshape(-1, 2 * dim)
        x = rows[:, :dim]
        y = rows[:, dim:]
        upper, lower = _unrestricted_tdhf_action(data, x, y)
        return jnp.concatenate([upper, -lower], axis=-1)

    return (
        vind,
        data.orbital_energy_differences_alpha,
        data.orbital_energy_differences_beta,
    )


def _is_traced_convergence_flag(value: Any) -> bool:
    return isinstance(value, jax.core.Tracer)


def _finalize_unrestricted_tda_result(
    eigvals: Array,
    eigvecs: Array,
    *,
    nroots: int,
    excitation_threshold: float,
    de_a: Array,
    de_b: Array,
) -> UnrestrictedTDAResult:
    nocca, nvira = de_a.shape
    noccb, nvirb = de_b.shape
    naa = int(nocca * nvira)
    valid = eigvals > excitation_threshold
    order = jnp.argsort(jnp.where(valid, eigvals, jnp.inf))
    keep = order[:nroots]
    mask = valid[keep]
    energies = jnp.where(mask, eigvals[keep], 0.0)
    amps = eigvecs[:, keep].T * mask[:, None]
    amplitudes_alpha = amps[:, :naa].reshape(nroots, nocca, nvira)
    amplitudes_beta = amps[:, naa:].reshape(nroots, noccb, nvirb)
    return UnrestrictedTDAResult(
        excitation_energies=energies,
        amplitudes_alpha=amplitudes_alpha,
        amplitudes_beta=amplitudes_beta,
    )


def solve_unrestricted_tda_from_operator(
    de_a: Array,
    de_b: Array,
    vind_rows: Callable[[Array], Array],
    diagonal: Array,
    *,
    nstates: int | None = None,
    excitation_threshold: float = 1e-7,
    davidson_tol: float = 1e-6,
    davidson_max_iter: int = 60,
    davidson_max_subspace: int | None = None,
) -> UnrestrictedTDAResult:
    dim = int(jnp.asarray(diagonal).size)
    nroots = dim if nstates is None else min(int(nstates), dim)
    eigvals, eigvecs, converged = davidson_lowest_symmetric(
        lambda vectors: vind_rows(jnp.asarray(vectors).T).T,
        nroots=nroots,
        size=dim,
        diag=jnp.asarray(diagonal).reshape(dim),
        tol=davidson_tol,
        max_iter=davidson_max_iter,
        max_subspace=davidson_max_subspace,
    )
    if not _is_traced_convergence_flag(converged) and not bool(converged):
        raise RuntimeError("Davidson unrestricted TDA solver did not converge.")
    eigvecs = jax.lax.stop_gradient(eigvecs)
    av = vind_rows(eigvecs.T).T
    eigvals = jnp.sum(eigvecs * av, axis=0) / jnp.maximum(
        jnp.sum(eigvecs * eigvecs, axis=0),
        jnp.asarray(1e-30, dtype=eigvecs.dtype),
    )
    return _finalize_unrestricted_tda_result(
        eigvals,
        eigvecs,
        nroots=nroots,
        excitation_threshold=excitation_threshold,
        de_a=de_a,
        de_b=de_b,
    )


def _finalize_unrestricted_casida_result(
    w: Array,
    x_vecs: Array,
    y_vecs: Array,
    *,
    nroots: int,
    excitation_threshold: float,
    matrix_eps: float,
    de_a: Array,
    de_b: Array,
) -> UnrestrictedTDDFTResult:
    nocca, nvira = de_a.shape
    noccb, nvirb = de_b.shape
    naa = int(nocca * nvira)
    valid = w > excitation_threshold
    order = jnp.argsort(jnp.where(valid, w, jnp.inf))
    keep = order[:nroots]
    keep_mask = valid[keep]

    energies = jnp.where(keep_mask, w[keep], 0.0)
    x = x_vecs[:, keep]
    y = y_vecs[:, keep]
    x = x * keep_mask[jnp.newaxis, :]
    y = y * keep_mask[jnp.newaxis, :]
    norm = jnp.sum(jnp.abs(x) ** 2, axis=0) - jnp.sum(jnp.abs(y) ** 2, axis=0)
    scale = 1.0 / jnp.sqrt(jnp.maximum(jnp.abs(norm), matrix_eps))
    x = x * scale[jnp.newaxis, :]
    y = y * scale[jnp.newaxis, :]

    x_alpha = x[:naa, :].T.reshape(nroots, nocca, nvira)
    x_beta = x[naa:, :].T.reshape(nroots, noccb, nvirb)
    y_alpha = y[:naa, :].T.reshape(nroots, nocca, nvira)
    y_beta = y[naa:, :].T.reshape(nroots, noccb, nvirb)
    return UnrestrictedTDDFTResult(
        excitation_energies=energies,
        x_amplitudes_alpha=x_alpha,
        x_amplitudes_beta=x_beta,
        y_amplitudes_alpha=y_alpha,
        y_amplitudes_beta=y_beta,
    )


def solve_unrestricted_casida_from_tdhf_operator(
    de_a: Array,
    de_b: Array,
    tdhf_vind_rows: Callable[[Array], Array],
    *,
    nstates: int | None = None,
    excitation_threshold: float = 1e-7,
    matrix_eps: float = 1e-10,
    davidson_tol: float = 1e-6,
    davidson_max_iter: int = 60,
    davidson_max_subspace: int | None = None,
) -> UnrestrictedTDDFTResult:
    dim = int(jnp.asarray(de_a).size + jnp.asarray(de_b).size)
    nroots = dim if nstates is None else min(int(nstates), dim)
    dtype = _solver_dtype(jnp.result_type(de_a, de_b))

    def tdhf_vind(values: Array) -> Array:
        values = jnp.asarray(values, dtype=dtype).reshape(-1, 2 * dim)
        return tdhf_vind_rows(values)

    diagonal = jnp.concatenate([jnp.ravel(de_a), jnp.ravel(de_b)]).astype(dtype)
    w, x_vecs, y_vecs, converged = davidson_lowest_tdhf(
        lambda values: jax.lax.stop_gradient(tdhf_vind(values)),
        nroots=nroots,
        size=dim,
        diag=diagonal,
        tol=davidson_tol,
        max_iter=davidson_max_iter,
        max_subspace=davidson_max_subspace,
        matrix_eps=matrix_eps,
    )
    del converged
    x_vecs = jax.lax.stop_gradient(x_vecs)
    y_vecs = jax.lax.stop_gradient(y_vecs)
    applied = tdhf_vind(jnp.concatenate([x_vecs.T, y_vecs.T], axis=-1))
    top = applied[:, :dim].T
    bottom = -applied[:, dim:].T
    numerator = jnp.sum(x_vecs * top, axis=0) + jnp.sum(
        y_vecs * bottom,
        axis=0,
    )
    denominator = jnp.sum(x_vecs * x_vecs, axis=0) - jnp.sum(y_vecs * y_vecs, axis=0)
    denominator = jnp.where(
        jnp.abs(denominator) > jnp.asarray(1e-30, dtype=dtype),
        denominator,
        jnp.asarray(1e-30, dtype=dtype),
    )
    return _finalize_unrestricted_casida_result(
        numerator / denominator,
        x_vecs,
        y_vecs,
        nroots=nroots,
        excitation_threshold=excitation_threshold,
        matrix_eps=matrix_eps,
        de_a=de_a,
        de_b=de_b,
    )


@dataclass(frozen=True)
class UnrestrictedTDA:
    """PySCF-like unrestricted TDA driver."""

    molecule: Any
    xc_functional: Any | None = None
    xc_params: Any | None = None
    occupation_tolerance: float = 1e-8
    excitation_threshold: float = 1e-7
    eigensolver: Literal["auto", "davidson"] = "auto"
    davidson_tol: float = 1e-6
    davidson_max_iter: int = 60
    davidson_max_subspace: int | None = None

    def gen_tda_vind(self):
        vind, _, _, _ = build_unrestricted_tda_operator(
            self.molecule,
            self.xc_functional,
            xc_params=self.xc_params,
            occupation_tolerance=self.occupation_tolerance,
        )
        return vind

    def _posthoc_correction(self, result: UnrestrictedTDAResult) -> Array | None:
        resolved_xc = _resolve_xc_functional(
            self.molecule,
            self.xc_functional,
            self.xc_params,
        )
        if resolved_xc is None:
            return None
        correction_fn = getattr(resolved_xc, "post_tda_correction", None)
        if not callable(correction_fn):
            return None
        try:
            correction = correction_fn(
                self.molecule,
                result,
                occupation_tolerance=self.occupation_tolerance,
            )
        except AttributeError as exc:
            if "does not expose" not in str(exc):
                raise
            return None
        correction = jnp.asarray(correction, dtype=result.excitation_energies.dtype)
        if correction.ndim == 0:
            correction = jnp.full_like(result.excitation_energies, correction)
        elif correction.shape != result.excitation_energies.shape:
            raise ValueError(
                "post_tda_correction must return a scalar or shape "
                f"{result.excitation_energies.shape}, got {correction.shape}."
            )
        return correction

    def _apply_posthoc_correction(
        self,
        result: UnrestrictedTDAResult,
    ) -> UnrestrictedTDAResult:
        correction = self._posthoc_correction(result)
        if correction is None:
            return result
        return replace(
            result,
            excitation_energies=result.excitation_energies + correction,
            posthoc_correction=correction,
        )

    def kernel(self, nstates: int | None = None) -> UnrestrictedTDAResult:
        mode = str(self.eigensolver).lower()
        if mode not in {"auto", "davidson"}:
            raise ValueError(
                f"Unsupported eigensolver={self.eigensolver!r}. Choose one of {{'auto', 'davidson'}}."
            )
        vind, diagonal, de_a, de_b = build_unrestricted_tda_operator(
            self.molecule,
            self.xc_functional,
            xc_params=self.xc_params,
            occupation_tolerance=self.occupation_tolerance,
        )
        result = solve_unrestricted_tda_from_operator(
            de_a,
            de_b,
            vind,
            diagonal,
            nstates=nstates,
            excitation_threshold=self.excitation_threshold,
            davidson_tol=self.davidson_tol,
            davidson_max_iter=self.davidson_max_iter,
            davidson_max_subspace=self.davidson_max_subspace,
        )
        return self._apply_posthoc_correction(result)


@dataclass(frozen=True)
class UnrestrictedCasidaTDDFT:
    """PySCF-like unrestricted Casida TDDFT driver."""

    molecule: Any
    xc_functional: Any | None = None
    xc_params: Any | None = None
    occupation_tolerance: float = 1e-8
    excitation_threshold: float = 1e-7
    matrix_eps: float = 1e-10
    eigensolver: Literal["auto", "davidson"] = "auto"
    davidson_tol: float = 1e-6
    davidson_max_iter: int = 60
    davidson_max_subspace: int | None = None

    def tda(self, nstates: int | None = None) -> UnrestrictedTDAResult:
        mode = str(self.eigensolver).lower()
        if mode not in {"auto", "davidson"}:
            raise ValueError(
                f"Unsupported eigensolver={self.eigensolver!r}. Choose one of {{'auto', 'davidson'}}."
            )
        vind, diagonal, de_a, de_b = build_unrestricted_tda_operator(
            self.molecule,
            self.xc_functional,
            xc_params=self.xc_params,
            occupation_tolerance=self.occupation_tolerance,
        )
        result = solve_unrestricted_tda_from_operator(
            de_a,
            de_b,
            vind,
            diagonal,
            nstates=nstates,
            excitation_threshold=self.excitation_threshold,
            davidson_tol=self.davidson_tol,
            davidson_max_iter=self.davidson_max_iter,
            davidson_max_subspace=self.davidson_max_subspace,
        )
        return self._apply_posthoc_correction(result, use_tda=True)

    def gen_tda_vind(self):
        vind, _, _, _ = build_unrestricted_tda_operator(
            self.molecule,
            self.xc_functional,
            xc_params=self.xc_params,
            occupation_tolerance=self.occupation_tolerance,
        )
        return vind

    def gen_tdhf_vind(self):
        vind, _, _ = build_unrestricted_tdhf_operator(
            self.molecule,
            self.xc_functional,
            xc_params=self.xc_params,
            occupation_tolerance=self.occupation_tolerance,
        )
        return vind

    def _posthoc_correction(
        self,
        result: UnrestrictedTDAResult | UnrestrictedTDDFTResult,
        *,
        use_tda: bool,
    ) -> Array | None:
        resolved_xc = _resolve_xc_functional(
            self.molecule,
            self.xc_functional,
            self.xc_params,
        )
        if resolved_xc is None:
            return None
        method_name = "post_tda_correction" if use_tda else "post_tddft_correction"
        correction_fn = getattr(resolved_xc, method_name, None)
        if not callable(correction_fn):
            return None
        try:
            correction = correction_fn(
                self.molecule,
                result,
                occupation_tolerance=self.occupation_tolerance,
            )
        except AttributeError as exc:
            if "does not expose" not in str(exc):
                raise
            return None
        correction = jnp.asarray(correction, dtype=result.excitation_energies.dtype)
        if correction.ndim == 0:
            correction = jnp.full_like(result.excitation_energies, correction)
        elif correction.shape != result.excitation_energies.shape:
            raise ValueError(
                f"{method_name} must return a scalar or shape "
                f"{result.excitation_energies.shape}, got {correction.shape}."
            )
        return correction

    def _apply_posthoc_correction(
        self,
        result: UnrestrictedTDAResult | UnrestrictedTDDFTResult,
        *,
        use_tda: bool,
    ) -> UnrestrictedTDAResult | UnrestrictedTDDFTResult:
        correction = self._posthoc_correction(result, use_tda=use_tda)
        if correction is None:
            return result
        return replace(
            result,
            excitation_energies=result.excitation_energies + correction,
            posthoc_correction=correction,
        )

    def kernel(self, nstates: int | None = None) -> UnrestrictedTDDFTResult:
        mode = str(self.eigensolver).lower()
        if mode not in {"auto", "davidson"}:
            raise ValueError(
                f"Unsupported eigensolver={self.eigensolver!r}. Choose one of {{'auto', 'davidson'}}."
            )
        vind_tdhf, de_a, de_b = build_unrestricted_tdhf_operator(
            self.molecule,
            self.xc_functional,
            xc_params=self.xc_params,
            occupation_tolerance=self.occupation_tolerance,
        )
        result = solve_unrestricted_casida_from_tdhf_operator(
            de_a,
            de_b,
            vind_tdhf,
            nstates=nstates,
            excitation_threshold=self.excitation_threshold,
            matrix_eps=self.matrix_eps,
            davidson_tol=self.davidson_tol,
            davidson_max_iter=self.davidson_max_iter,
            davidson_max_subspace=self.davidson_max_subspace,
        )
        return self._apply_posthoc_correction(result, use_tda=False)
