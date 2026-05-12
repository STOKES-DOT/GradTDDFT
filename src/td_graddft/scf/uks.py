from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import jax
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array

from ..features import (
    MoleculeLikeState,
    _spin_density_and_gradient,
    molecule_grid_view,
)
from .core import _build_density_from_occ, _contains_jax_tracer, _host_float_unless_traced
from .rks import (
    _build_jk,
    _diagonalize_fock,
    _orthogonalizer,
    _vxc_matrix_from_grid_potential,
)
from ..jax_libxc import RestrictedFeatureBundle, eval_xc_energy_density, hybrid_coeff, parse_xc, xc_type


@dataclass(frozen=True)
class UKSConfig:
    """Configuration for unrestricted Kohn-Sham SCF iterations."""

    xc_spec: str = "pbe"
    max_cycle: int = 80
    conv_tol: float = 1e-10
    conv_tol_density: float = 1e-8
    damping: float = 0.0
    level_shift: float = 0.0
    orthogonalization_eps: float = 1e-10
    density_floor: float = 1e-12
    potential_clip: float | None = None


@dataclass(frozen=True)
class UKSResult:
    """Unrestricted Kohn-Sham result object."""

    converged: bool
    total_energy: float
    electronic_energy: float
    nuclear_repulsion: float
    xc_energy: float
    exact_exchange_fraction: float
    mo_energy_alpha: Array
    mo_energy_beta: Array
    mo_coeff_alpha: Array
    mo_coeff_beta: Array
    mo_occ_alpha: Array
    mo_occ_beta: Array
    density_matrix_alpha: Array
    density_matrix_beta: Array
    fock_matrix_alpha: Array
    fock_matrix_beta: Array
    overlap_matrix: Array
    hcore_matrix: Array
    cycles: int

def _default_spin_mo_occ(nao: int, nelec: int, dtype) -> Array:
    return jnp.zeros((nao,), dtype=dtype).at[:nelec].set(1.0)


def _validate_spin_occ(mo_occ: Array, *, nao: int, nelec: int, label: str) -> Array:
    occ = jnp.asarray(mo_occ)
    if occ.ndim != 1 or int(occ.shape[0]) != nao:
        raise ValueError(f"{label} occupations must be a 1D vector with length nao.")
    if float(jnp.min(occ)) < -1e-8 or float(jnp.max(occ)) > 1.0 + 1e-8:
        raise ValueError(f"{label} occupations must lie in [0, 1].")
    if abs(float(jnp.sum(occ)) - float(nelec)) > 1e-6:
        raise ValueError(f"{label} occupations must sum to the requested electron count.")
    return occ


def _validate_initial_spin_density(
    density: Array | None,
    *,
    nao: int,
    dtype: Any,
    label: str,
) -> Array | None:
    if density is None:
        return None
    dm = jnp.asarray(density, dtype=dtype)
    if dm.ndim != 2 or int(dm.shape[0]) != nao or int(dm.shape[1]) != nao:
        raise ValueError(f"{label} must be a square ({nao}, {nao}) matrix for UKS.")
    return 0.5 * (dm + dm.T)


@lru_cache(maxsize=64)
def _point_unrestricted_xc_value_and_grad_kernel(
    xc_spec: str,
    xc_kind: str,
):
    xc_spec_norm = str(xc_spec)
    xc_kind_norm = str(xc_kind).upper()

    def point_energy(variables: Array) -> Array:
        rho_a = jnp.maximum(variables[0], 0.0)
        rho_b = jnp.maximum(variables[1], 0.0)
        zero_grad = jnp.zeros((3,), dtype=variables.dtype)
        if xc_kind_norm == "LDA":
            grad_a = zero_grad
            grad_b = zero_grad
        elif xc_kind_norm == "GGA":
            grad_a = variables[2:5]
            grad_b = variables[5:8]
        else:
            raise ValueError(f"Unsupported XC kind={xc_kind_norm!r}.")
        features = RestrictedFeatureBundle(
            rho_a=rho_a,
            rho_b=rho_b,
            sigma_aa=jnp.dot(grad_a, grad_a),
            sigma_ab=jnp.dot(grad_a, grad_b),
            sigma_bb=jnp.dot(grad_b, grad_b),
            tau_a=jnp.asarray(0.0, dtype=variables.dtype),
            tau_b=jnp.asarray(0.0, dtype=variables.dtype),
        )
        return eval_xc_energy_density(xc_spec_norm, features)

    return jax.jit(jax.vmap(jax.value_and_grad(point_energy)))


def _unrestricted_xc_energy_and_potential_on_grid(
    *,
    ao: Array,
    ao_deriv1: Array,
    weights: Array,
    density_a: Array,
    density_b: Array,
    xc_spec: str,
    density_floor: float,
    potential_clip: float | None,
    xc_kind: str,
) -> tuple[Array, Array, Array, Array, Array]:
    rho_a, grad_a = _spin_density_and_gradient(ao, ao_deriv1, density_a)
    rho_b, grad_b = _spin_density_and_gradient(ao, ao_deriv1, density_b)
    rho_total = rho_a + rho_b
    if xc_kind == "HF":
        zeros = jnp.zeros_like(rho_total)
        zero_grads = jnp.zeros((rho_total.shape[0], 3), dtype=rho_total.dtype)
        return jnp.asarray(0.0, dtype=rho_total.dtype), zeros, zeros, zero_grads, zero_grads

    if xc_kind == "LDA":
        response_variables = jnp.stack([rho_a, rho_b], axis=-1)
    elif xc_kind == "GGA":
        response_variables = jnp.concatenate(
            [
                rho_a[..., None],
                rho_b[..., None],
                grad_a,
                grad_b,
            ],
            axis=-1,
        )
    else:
        raise ValueError(f"Unsupported XC kind={xc_kind!r}.")

    point_exc, point_grad = _point_unrestricted_xc_value_and_grad_kernel(
        xc_spec,
        xc_kind,
    )(response_variables)
    point_exc = jnp.nan_to_num(point_exc, nan=0.0, posinf=0.0, neginf=0.0)
    point_grad = jnp.nan_to_num(point_grad, nan=0.0, posinf=0.0, neginf=0.0)
    exc = jnp.tensordot(weights, point_exc, axes=(0, 0))

    mask = rho_total > density_floor
    vxc_rho_a = jnp.where(mask, point_grad[:, 0], 0.0)
    vxc_rho_b = jnp.where(mask, point_grad[:, 1], 0.0)
    if xc_kind == "GGA":
        vxc_grad_a = jnp.where(mask[:, None], point_grad[:, 2:5], 0.0)
        vxc_grad_b = jnp.where(mask[:, None], point_grad[:, 5:8], 0.0)
    else:
        vxc_grad_a = jnp.zeros((rho_total.shape[0], 3), dtype=rho_total.dtype)
        vxc_grad_b = jnp.zeros((rho_total.shape[0], 3), dtype=rho_total.dtype)

    if potential_clip is not None:
        clip = jnp.asarray(potential_clip, dtype=rho_total.dtype)
        vxc_rho_a = jnp.clip(vxc_rho_a, -clip, clip)
        vxc_rho_b = jnp.clip(vxc_rho_b, -clip, clip)
        vxc_grad_a = jnp.clip(vxc_grad_a, -clip, clip)
        vxc_grad_b = jnp.clip(vxc_grad_b, -clip, clip)
    return exc, vxc_rho_a, vxc_rho_b, vxc_grad_a, vxc_grad_b


def _raw_fock_and_energy_for_state(
    *,
    density_a: Array,
    density_b: Array,
    mo_coeff_a: Array,
    mo_coeff_b: Array,
    mo_occ_a: Array,
    mo_occ_b: Array,
    ao: Array,
    ao_deriv1: Array,
    weights: Array,
    h: Array,
    eri: Array,
    enuc: Array,
    alpha: Array,
    cfg: UKSConfig,
    xc_kind: str,
) -> tuple[Array, Array, Array, Array]:
    density_tot = density_a + density_b
    j_tot, _ = _build_jk(eri, density_tot)
    _, k_a = _build_jk(eri, density_a)
    _, k_b = _build_jk(eri, density_b)
    xc_energy, vxc_rho_a, vxc_rho_b, vxc_grad_a, vxc_grad_b = _unrestricted_xc_energy_and_potential_on_grid(
        ao=ao,
        ao_deriv1=ao_deriv1,
        weights=weights,
        density_a=density_a,
        density_b=density_b,
        xc_spec=cfg.xc_spec,
        density_floor=cfg.density_floor,
        potential_clip=cfg.potential_clip,
        xc_kind=xc_kind,
    )
    ao_laplacian = jnp.zeros_like(ao)
    zeros = jnp.zeros_like(vxc_rho_a)
    vxc_matrix_a = _vxc_matrix_from_grid_potential(
        ao=ao,
        ao_deriv1=ao_deriv1,
        ao_laplacian=ao_laplacian,
        weights=weights,
        vxc_rho=vxc_rho_a,
        vxc_grad=vxc_grad_a,
        vxc_tau=zeros,
        vxc_lapl=zeros,
        xc_kind=xc_kind,
    )
    vxc_matrix_b = _vxc_matrix_from_grid_potential(
        ao=ao,
        ao_deriv1=ao_deriv1,
        ao_laplacian=ao_laplacian,
        weights=weights,
        vxc_rho=vxc_rho_b,
        vxc_grad=vxc_grad_b,
        vxc_tau=zeros,
        vxc_lapl=zeros,
        xc_kind=xc_kind,
    )
    fock_a = h + j_tot - alpha * k_a + vxc_matrix_a
    fock_b = h + j_tot - alpha * k_b + vxc_matrix_b

    e_one = jnp.einsum("ij,ij->", density_tot, h, precision=Precision.HIGHEST)
    e_coul = 0.5 * jnp.einsum("ij,ij->", density_tot, j_tot, precision=Precision.HIGHEST)
    e_x_hf = -0.5 * alpha * (
        jnp.einsum("ij,ij->", density_a, k_a, precision=Precision.HIGHEST)
        + jnp.einsum("ij,ij->", density_b, k_b, precision=Precision.HIGHEST)
    )
    total = e_one + e_coul + e_x_hf + xc_energy + enuc
    del mo_coeff_a, mo_coeff_b, mo_occ_a, mo_occ_b
    return total, xc_energy, fock_a, fock_b


def _molecule_like_state_for_bound_xc(
    *,
    density_a: Array,
    density_b: Array,
    mo_coeff_a: Array,
    mo_coeff_b: Array,
    mo_occ_a: Array,
    mo_occ_b: Array,
    mo_energy_a: Array,
    mo_energy_b: Array,
    ao: Array,
    ao_deriv1: Array,
    weights: Array,
    h: Array,
    eri: Array,
    overlap: Array,
    molecule_template: Any | None,
) -> Any:
    return MoleculeLikeState(
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid=molecule_grid_view(weights, template=getattr(molecule_template, "grid", None)),
        rdm1=jnp.stack([density_a, density_b], axis=0),
        mo_coeff=jnp.stack([mo_coeff_a, mo_coeff_b], axis=0),
        mo_occ=jnp.stack([mo_occ_a, mo_occ_b], axis=0),
        mo_energy=jnp.stack([mo_energy_a, mo_energy_b], axis=0),
        rep_tensor=eri,
        h1e=h,
        overlap_matrix=overlap,
        ao_laplacian=getattr(molecule_template, "ao_laplacian", None),
        atom_coords=getattr(molecule_template, "atom_coords", None),
        atom_charges=getattr(molecule_template, "atom_charges", None),
        hfx_omega_values=getattr(molecule_template, "hfx_omega_values", None),
        hfx_nu=getattr(molecule_template, "hfx_nu", None),
    )


def _raw_fock_and_energy_for_bound_xc_state(
    *,
    density_a: Array,
    density_b: Array,
    mo_coeff_a: Array,
    mo_coeff_b: Array,
    mo_occ_a: Array,
    mo_occ_b: Array,
    mo_energy_a: Array,
    mo_energy_b: Array,
    ao: Array,
    ao_deriv1: Array,
    weights: Array,
    h: Array,
    eri: Array,
    enuc: Array,
    overlap: Array,
    bound_xc: Any,
    molecule_template: Any | None,
) -> tuple[Array, Array, Array, Array]:
    molecule_state = _molecule_like_state_for_bound_xc(
        density_a=density_a,
        density_b=density_b,
        mo_coeff_a=mo_coeff_a,
        mo_coeff_b=mo_coeff_b,
        mo_occ_a=mo_occ_a,
        mo_occ_b=mo_occ_b,
        mo_energy_a=mo_energy_a,
        mo_energy_b=mo_energy_b,
        ao=ao,
        ao_deriv1=ao_deriv1,
        weights=weights,
        h=h,
        eri=eri,
        overlap=overlap,
        molecule_template=molecule_template,
    )
    density_tot = density_a + density_b
    j_tot, _ = _build_jk(eri, density_tot)
    _, k_a = _build_jk(eri, density_a)
    _, k_b = _build_jk(eri, density_b)
    (
        vxc_rho_a,
        vxc_rho_b,
        vxc_grad_a,
        vxc_grad_b,
        xc_kind,
        alpha,
        extra_fock_a,
        extra_fock_b,
    ) = bound_xc.unrestricted_scf_components(molecule_state)
    ao_laplacian = getattr(molecule_state, "ao_laplacian", None)
    if ao_laplacian is None:
        ao_laplacian = jnp.zeros_like(ao)
    zeros_a = jnp.zeros_like(vxc_rho_a)
    zeros_b = jnp.zeros_like(vxc_rho_b)
    vxc_matrix_a = _vxc_matrix_from_grid_potential(
        ao=ao,
        ao_deriv1=ao_deriv1,
        ao_laplacian=ao_laplacian,
        weights=weights,
        vxc_rho=vxc_rho_a,
        vxc_grad=vxc_grad_a,
        vxc_tau=zeros_a,
        vxc_lapl=zeros_a,
        xc_kind=xc_kind,
    )
    vxc_matrix_b = _vxc_matrix_from_grid_potential(
        ao=ao,
        ao_deriv1=ao_deriv1,
        ao_laplacian=ao_laplacian,
        weights=weights,
        vxc_rho=vxc_rho_b,
        vxc_grad=vxc_grad_b,
        vxc_tau=zeros_b,
        vxc_lapl=zeros_b,
        xc_kind=xc_kind,
    )
    fock_a = h + j_tot - alpha * k_a + vxc_matrix_a + extra_fock_a
    fock_b = h + j_tot - alpha * k_b + vxc_matrix_b + extra_fock_b
    xc_energy = jnp.asarray(bound_xc.energy_from_molecule(molecule_state), dtype=h.dtype)
    e_one = jnp.einsum("ij,ij->", density_tot, h, precision=Precision.HIGHEST)
    e_coul = 0.5 * jnp.einsum("ij,ij->", density_tot, j_tot, precision=Precision.HIGHEST)
    total = e_one + e_coul + xc_energy + enuc
    return total, xc_energy, fock_a, fock_b


def _density_rms(density_a_new: Array, density_a_old: Array, density_b_new: Array, density_b_old: Array) -> Array:
    return jnp.sqrt(
        0.5
        * (
            jnp.mean((density_a_new - density_a_old) ** 2)
            + jnp.mean((density_b_new - density_b_old) ** 2)
        )
    )


def run_uks_from_integrals(
    *,
    overlap: Array,
    hcore: Array,
    eri: Array,
    nalpha: int,
    nbeta: int,
    nuclear_repulsion: float | Array,
    ao: Array,
    ao_deriv1: Array,
    grid_weights: Array,
    init_density_alpha: Array | None = None,
    init_density_beta: Array | None = None,
    init_mo_coeff_alpha: Array | None = None,
    init_mo_coeff_beta: Array | None = None,
    init_mo_occ_alpha: Array | None = None,
    init_mo_occ_beta: Array | None = None,
    init_mo_energy_alpha: Array | None = None,
    init_mo_energy_beta: Array | None = None,
    config: UKSConfig | None = None,
    bound_xc: Any | None = None,
    molecule_template: Any | None = None,
) -> UKSResult:
    """Run unrestricted Kohn-Sham SCF from AO integrals and numerical grid data."""

    cfg = UKSConfig() if config is None else config
    xc_kind = None
    if bound_xc is None:
        parse_xc(cfg.xc_spec)
        xc_kind = xc_type(cfg.xc_spec)
        if xc_kind == "MGGA":
            raise NotImplementedError(
                "UKS SCF matrix assembly currently supports LDA/GGA/HF semilocal terms. "
                "MGGA requires tau-dependent AO Hessian terms."
            )
    s = jnp.asarray(overlap)
    h = jnp.asarray(hcore)
    eri = jnp.asarray(eri)
    ao = jnp.asarray(ao)
    ao_deriv1 = jnp.asarray(ao_deriv1)
    weights = jnp.asarray(grid_weights)
    enuc = jnp.asarray(nuclear_repulsion)
    traceable_inputs = _contains_jax_tracer((s, h, eri, ao, ao_deriv1, weights, enuc))
    nao = int(s.shape[0])
    if nalpha < 0 or nalpha > nao or nbeta < 0 or nbeta > nao or (nalpha + nbeta) <= 0:
        raise ValueError("Invalid occupation counts for UKS.")
    if xc_kind == "GGA" and ao_deriv1.shape[0] < 4:
        raise ValueError("GGA UKS requires ao_deriv1 to include AO values plus first derivatives.")

    x = _orthogonalizer(s, cfg.orthogonalization_eps)
    core_mo_energy, core_mo_coeff = _diagonalize_fock(h, x)
    mo_coeff_a = core_mo_coeff if init_mo_coeff_alpha is None else jnp.asarray(init_mo_coeff_alpha)
    mo_coeff_b = core_mo_coeff if init_mo_coeff_beta is None else jnp.asarray(init_mo_coeff_beta)
    mo_energy_a = core_mo_energy if init_mo_energy_alpha is None else jnp.asarray(init_mo_energy_alpha)
    mo_energy_b = core_mo_energy if init_mo_energy_beta is None else jnp.asarray(init_mo_energy_beta)

    mo_occ_a = _default_spin_mo_occ(nao, nalpha, h.dtype) if init_mo_occ_alpha is None else init_mo_occ_alpha
    mo_occ_b = _default_spin_mo_occ(nao, nbeta, h.dtype) if init_mo_occ_beta is None else init_mo_occ_beta
    mo_occ_a = _validate_spin_occ(mo_occ_a, nao=nao, nelec=nalpha, label="alpha")
    mo_occ_b = _validate_spin_occ(mo_occ_b, nao=nao, nelec=nbeta, label="beta")

    density_a = _build_density_from_occ(mo_coeff_a, mo_occ_a)
    density_b = _build_density_from_occ(mo_coeff_b, mo_occ_b)
    init_density_a = _validate_initial_spin_density(
        init_density_alpha,
        nao=nao,
        dtype=h.dtype,
        label="init_density_alpha",
    )
    init_density_b = _validate_initial_spin_density(
        init_density_beta,
        nao=nao,
        dtype=h.dtype,
        label="init_density_beta",
    )
    if init_density_a is not None:
        density_a = init_density_a
    if init_density_b is not None:
        density_b = init_density_b

    energy = jnp.asarray(0.0, dtype=h.dtype)
    xc_energy = jnp.asarray(0.0, dtype=h.dtype)
    converged = False
    cycles = 0
    fock_a = h
    fock_b = h
    if bound_xc is None:
        alpha_scalar = float(hybrid_coeff(cfg.xc_spec))
        alpha = jnp.asarray(alpha_scalar, dtype=h.dtype)
    else:
        alpha = None
        alpha_scalar = float(jnp.asarray(getattr(bound_xc, "exact_exchange_fraction", 0.0)))

    for cycle in range(1, cfg.max_cycle + 1):
        if bound_xc is None:
            _, _, raw_fock_a, raw_fock_b = _raw_fock_and_energy_for_state(
                density_a=density_a,
                density_b=density_b,
                mo_coeff_a=mo_coeff_a,
                mo_coeff_b=mo_coeff_b,
                mo_occ_a=mo_occ_a,
                mo_occ_b=mo_occ_b,
                ao=ao,
                ao_deriv1=ao_deriv1,
                weights=weights,
                h=h,
                eri=eri,
                enuc=enuc,
                alpha=alpha,
                cfg=cfg,
                xc_kind=xc_kind,
            )
        else:
            _, _, raw_fock_a, raw_fock_b = _raw_fock_and_energy_for_bound_xc_state(
                density_a=density_a,
                density_b=density_b,
                mo_coeff_a=mo_coeff_a,
                mo_coeff_b=mo_coeff_b,
                mo_occ_a=mo_occ_a,
                mo_occ_b=mo_occ_b,
                mo_energy_a=mo_energy_a,
                mo_energy_b=mo_energy_b,
                ao=ao,
                ao_deriv1=ao_deriv1,
                weights=weights,
                h=h,
                eri=eri,
                enuc=enuc,
                overlap=s,
                bound_xc=bound_xc,
                molecule_template=molecule_template,
            )
        fock_a_eff = raw_fock_a
        fock_b_eff = raw_fock_b
        if cfg.level_shift != 0.0:
            shift = jnp.asarray(cfg.level_shift, dtype=h.dtype)
            fock_a_eff = fock_a_eff + shift * s
            fock_b_eff = fock_b_eff + shift * s

        mo_energy_a_new, mo_coeff_a_new = _diagonalize_fock(fock_a_eff, x)
        mo_energy_b_new, mo_coeff_b_new = _diagonalize_fock(fock_b_eff, x)
        density_a_new = _build_density_from_occ(mo_coeff_a_new, mo_occ_a)
        density_b_new = _build_density_from_occ(mo_coeff_b_new, mo_occ_b)
        if cfg.damping != 0.0:
            damping = jnp.asarray(cfg.damping, dtype=h.dtype)
            density_a_new = (1.0 - damping) * density_a_new + damping * density_a
            density_b_new = (1.0 - damping) * density_b_new + damping * density_b

        if bound_xc is None:
            total_new, xc_energy_new, fock_a_new, fock_b_new = _raw_fock_and_energy_for_state(
                density_a=density_a_new,
                density_b=density_b_new,
                mo_coeff_a=mo_coeff_a_new,
                mo_coeff_b=mo_coeff_b_new,
                mo_occ_a=mo_occ_a,
                mo_occ_b=mo_occ_b,
                ao=ao,
                ao_deriv1=ao_deriv1,
                weights=weights,
                h=h,
                eri=eri,
                enuc=enuc,
                alpha=alpha,
                cfg=cfg,
                xc_kind=xc_kind,
            )
        else:
            total_new, xc_energy_new, fock_a_new, fock_b_new = _raw_fock_and_energy_for_bound_xc_state(
                density_a=density_a_new,
                density_b=density_b_new,
                mo_coeff_a=mo_coeff_a_new,
                mo_coeff_b=mo_coeff_b_new,
                mo_occ_a=mo_occ_a,
                mo_occ_b=mo_occ_b,
                mo_energy_a=mo_energy_a_new,
                mo_energy_b=mo_energy_b_new,
                ao=ao,
                ao_deriv1=ao_deriv1,
                weights=weights,
                h=h,
                eri=eri,
                enuc=enuc,
                overlap=s,
                bound_xc=bound_xc,
                molecule_template=molecule_template,
            )

        delta_e = jnp.abs(total_new - energy)
        rms_d = _density_rms(density_a_new, density_a, density_b_new, density_b)
        density_a = density_a_new
        density_b = density_b_new
        mo_coeff_a = mo_coeff_a_new
        mo_coeff_b = mo_coeff_b_new
        mo_energy_a = mo_energy_a_new
        mo_energy_b = mo_energy_b_new
        energy = total_new
        xc_energy = xc_energy_new
        fock_a = fock_a_new
        fock_b = fock_b_new
        cycles = cycle
        if (
            not traceable_inputs
            and float(delta_e) < cfg.conv_tol
            and float(rms_d) < cfg.conv_tol_density
        ):
            converged = True
            break

    if not traceable_inputs and cfg.level_shift != 0.0:
        mo_energy_a_final, mo_coeff_a_final = _diagonalize_fock(fock_a, x)
        mo_energy_b_final, mo_coeff_b_final = _diagonalize_fock(fock_b, x)
        density_a_final = _build_density_from_occ(mo_coeff_a_final, mo_occ_a)
        density_b_final = _build_density_from_occ(mo_coeff_b_final, mo_occ_b)
        if bound_xc is None:
            total_final, xc_energy_final, fock_a_final, fock_b_final = _raw_fock_and_energy_for_state(
                density_a=density_a_final,
                density_b=density_b_final,
                mo_coeff_a=mo_coeff_a_final,
                mo_coeff_b=mo_coeff_b_final,
                mo_occ_a=mo_occ_a,
                mo_occ_b=mo_occ_b,
                ao=ao,
                ao_deriv1=ao_deriv1,
                weights=weights,
                h=h,
                eri=eri,
                enuc=enuc,
                alpha=alpha,
                cfg=cfg,
                xc_kind=xc_kind,
            )
        else:
            total_final, xc_energy_final, fock_a_final, fock_b_final = _raw_fock_and_energy_for_bound_xc_state(
                density_a=density_a_final,
                density_b=density_b_final,
                mo_coeff_a=mo_coeff_a_final,
                mo_coeff_b=mo_coeff_b_final,
                mo_occ_a=mo_occ_a,
                mo_occ_b=mo_occ_b,
                mo_energy_a=mo_energy_a_final,
                mo_energy_b=mo_energy_b_final,
                ao=ao,
                ao_deriv1=ao_deriv1,
                weights=weights,
                h=h,
                eri=eri,
                enuc=enuc,
                overlap=s,
                bound_xc=bound_xc,
                molecule_template=molecule_template,
            )
        tol_e = jnp.asarray(cfg.conv_tol, dtype=h.dtype) * jnp.asarray(10.0, dtype=h.dtype)
        density_tol = jnp.sqrt(jnp.asarray(cfg.conv_tol, dtype=h.dtype)) * jnp.asarray(3.0, dtype=h.dtype)
        delta_e_final = jnp.abs(total_final - energy)
        rms_d_final = _density_rms(density_a_final, density_a, density_b_final, density_b)
        converged = bool((delta_e_final < tol_e) | (rms_d_final < density_tol))
        density_a = density_a_final
        density_b = density_b_final
        mo_coeff_a = mo_coeff_a_final
        mo_coeff_b = mo_coeff_b_final
        mo_energy_a = mo_energy_a_final
        mo_energy_b = mo_energy_b_final
        energy = total_final
        xc_energy = xc_energy_final
        fock_a = fock_a_final
        fock_b = fock_b_final

    return UKSResult(
        converged=converged,
        total_energy=_host_float_unless_traced(energy),
        electronic_energy=_host_float_unless_traced(energy - enuc),
        nuclear_repulsion=_host_float_unless_traced(enuc),
        xc_energy=_host_float_unless_traced(xc_energy),
        exact_exchange_fraction=float(alpha_scalar),
        mo_energy_alpha=mo_energy_a,
        mo_energy_beta=mo_energy_b,
        mo_coeff_alpha=mo_coeff_a,
        mo_coeff_beta=mo_coeff_b,
        mo_occ_alpha=mo_occ_a,
        mo_occ_beta=mo_occ_b,
        density_matrix_alpha=density_a,
        density_matrix_beta=density_b,
        fock_matrix_alpha=fock_a,
        fock_matrix_beta=fock_b,
        overlap_matrix=s,
        hcore_matrix=h,
        cycles=cycles,
    )
