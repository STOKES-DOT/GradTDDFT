from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.lax import Precision
from jaxtyping import Array, PRNGKeyArray, PyTree

from .descriptors import (
    AtomCenteredDensityDescriptorConfig,
    make_atom_centered_density_descriptor_fn,
)
from .gnn import RSHGNNHead
from ..features import restricted_grid_features, restricted_grid_features_with_gradients
from ..jax_libxc import (
    RestrictedFeatureBundle,
    eval_xc_energy_density,
    restricted_feature_bundle_from_rho_grad_tau,
    semilocal_terms,
    xc_type,
)
from .schema import (
    RSHFunctionalTemplate,
    ResolvedRSHParameters,
    SCFXCContributions,
    make_pyscf_rsh_spec,
)


def _constant_rsh_descriptor(_molecule: Any | None = None) -> Array:
    return jnp.ones((1,), dtype=jnp.float32)


def _logit_from_interval(value: Array, bounds: tuple[float, float]) -> Array:
    lower, upper = bounds
    value_arr = jnp.asarray(value, dtype=jnp.float32)
    scaled = (value_arr - lower) / jnp.maximum(upper - lower, 1e-8)
    clipped = jnp.clip(scaled, 1e-6, 1.0 - 1e-6)
    return jnp.log(clipped / (1.0 - clipped))


def _sigmoid_to_interval(raw: Array, bounds: tuple[float, float]) -> Array:
    lower, upper = bounds
    raw_arr = jnp.asarray(raw, dtype=jnp.float32)
    return lower + (upper - lower) * jax.nn.sigmoid(raw_arr)


def _restricted_spin_density_blocks(molecule: Any) -> tuple[Array, Array]:
    rdm1 = jnp.asarray(molecule.rdm1)
    if rdm1.ndim == 2:
        half = 0.5 * rdm1
        return half, half
    if rdm1.ndim != 3 or rdm1.shape[0] != 2:
        raise ValueError(
            "Restricted RSH functional expects rdm1 with shape (nao, nao) or (2, nao, nao)."
        )
    return rdm1[0], rdm1[1]


def _range_separated_aux_error() -> AttributeError:
    return AttributeError(
        "Range-separated Fock assembly requires molecule.hfx_nu. "
        "Build the reference with compute_local_hfx_aux=True."
    )


def _has_range_separated_aux(molecule: Any) -> bool:
    return getattr(molecule, "hfx_nu", None) is not None


def _is_concrete_scalar_near_zero(value: Any, *, tol: float = 1e-12) -> bool:
    try:
        return abs(float(jnp.asarray(value))) <= tol
    except Exception:
        return False


def _xc_spec_uses_wpbeh(xc_spec: str) -> bool:
    return any(term.name == "gga_x_wpbeh" for term in semilocal_terms(str(xc_spec)))


def _exact_exchange_energy(molecule: Any) -> Array:
    rep_tensor = jnp.asarray(molecule.rep_tensor)
    dm_a, dm_b = _restricted_spin_density_blocks(molecule)

    def spin_exchange(dm_spin: Array) -> Array:
        exchange_matrix = jnp.einsum(
            "prqs,rs->pq",
            rep_tensor,
            dm_spin,
            precision=Precision.HIGHEST,
        )
        return -0.5 * jnp.einsum(
            "pq,pq->",
            dm_spin,
            exchange_matrix,
            precision=Precision.HIGHEST,
        )

    return jnp.sum(jax.vmap(spin_exchange)(jnp.stack([dm_a, dm_b], axis=0)))


def _molecule_hfx_omega_values(
    molecule: Any,
    fallback: tuple[float, ...] | None,
    *,
    count: int,
) -> Array:
    values = getattr(molecule, "hfx_omega_values", None)
    if values is None:
        if fallback is None:
            raise AttributeError(
                "RSH omega interpolation requires molecule.hfx_omega_values or a functional fallback grid."
            )
        values = fallback
    omega_values = jnp.asarray(values, dtype=jnp.float32)
    if omega_values.ndim != 1 or omega_values.shape[0] != count:
        raise ValueError(
            "hfx_omega_values must be 1D and match the omega-channel count "
            f"(got {omega_values.shape}, expected ({count},))."
        )
    return omega_values


def _interpolate_over_omega_axis(
    values: Array,
    omega_values: Array,
    omega: Array,
) -> Array:
    values_arr = jnp.asarray(values)
    omega_grid = jnp.asarray(omega_values, dtype=values_arr.dtype)
    omega_scalar = jnp.asarray(omega, dtype=values_arr.dtype)
    if values_arr.shape[0] != omega_grid.shape[0]:
        raise ValueError(
            "Leading omega axis of values must match omega_values "
            f"(got {values_arr.shape[0]} vs {omega_grid.shape[0]})."
        )
    if values_arr.shape[0] == 1:
        return values_arr[0]

    omega_clipped = jnp.clip(omega_scalar, omega_grid[0], omega_grid[-1])
    upper = jnp.clip(jnp.sum(omega_grid <= omega_clipped), 1, omega_grid.shape[0] - 1)
    lower = upper - 1
    omega_lower = omega_grid[lower]
    omega_upper = omega_grid[upper]
    fraction = (omega_clipped - omega_lower) / jnp.maximum(omega_upper - omega_lower, 1e-8)
    left = values_arr[lower]
    right = values_arr[upper]
    return left + fraction * (right - left)


def _range_separated_exchange_grid_components(
    molecule: Any,
    *,
    omega: Array,
    fallback_omega_values: tuple[float, ...] | None,
) -> tuple[Array, Array, Array]:
    nu_cache = getattr(molecule, "hfx_nu", None)
    if nu_cache is None:
        raise AttributeError(
            "Range-separated exchange evaluation requires molecule.hfx_nu. "
            "Build the reference with compute_local_hfx_aux=True."
        )
    ao = jnp.asarray(molecule.ao)
    nu = jnp.asarray(nu_cache)
    if nu.ndim != 4:
        raise ValueError(
            "molecule.hfx_nu must have shape (n_omega, ngrids, nao, nao), "
            f"got {nu.shape}."
        )
    omega_values = _molecule_hfx_omega_values(
        molecule,
        fallback_omega_values,
        count=int(nu.shape[0]),
    )
    nu_interp = _interpolate_over_omega_axis(nu, omega_values, omega)
    dm_a, dm_b = _restricted_spin_density_blocks(molecule)
    e_a = jnp.einsum("gp,pq->gq", ao, dm_a, precision=Precision.HIGHEST)
    e_b = jnp.einsum("gp,pq->gq", ao, dm_b, precision=Precision.HIGHEST)
    fxx_a = jnp.einsum("gbc,gc->gb", nu_interp, e_a, precision=Precision.HIGHEST)
    fxx_b = jnp.einsum("gbc,gc->gb", nu_interp, e_b, precision=Precision.HIGHEST)
    exx_a = -0.5 * jnp.einsum("gq,gq->g", e_a, fxx_a, precision=Precision.HIGHEST)
    exx_b = -0.5 * jnp.einsum("gq,gq->g", e_b, fxx_b, precision=Precision.HIGHEST)
    total = jnp.nan_to_num(exx_a + exx_b, nan=0.0, posinf=0.0, neginf=0.0)
    return total, jnp.nan_to_num(exx_a), jnp.nan_to_num(exx_b)


def _range_separated_exchange_energy(
    molecule: Any,
    *,
    omega: Array,
    fallback_omega_values: tuple[float, ...] | None,
) -> Array:
    total, _, _ = _range_separated_exchange_grid_components(
        molecule,
        omega=omega,
        fallback_omega_values=fallback_omega_values,
    )
    return jnp.tensordot(jnp.asarray(molecule.grid.weights), total, axes=(0, 0))


def _range_separated_exchange_matrix(
    molecule: Any,
    *,
    omega: Array,
    fallback_omega_values: tuple[float, ...] | None,
) -> Array:
    nu_cache = getattr(molecule, "hfx_nu", None)
    if nu_cache is None:
        raise _range_separated_aux_error()
    ao = jnp.asarray(molecule.ao)
    weights = jnp.asarray(molecule.grid.weights)
    nu = jnp.asarray(nu_cache)
    omega_values = _molecule_hfx_omega_values(
        molecule,
        fallback_omega_values,
        count=int(nu.shape[0]),
    )
    nu_interp = _interpolate_over_omega_axis(nu, omega_values, omega)
    density_total = jnp.asarray(molecule.rdm1)
    if density_total.ndim == 3:
        density_total = density_total.sum(axis=0)
    e = jnp.einsum("gp,pq->gq", ao, density_total, precision=Precision.HIGHEST)
    fxx = jnp.einsum("gbc,gc->gb", nu_interp, e, precision=Precision.HIGHEST)
    k_mat = jnp.einsum(
        "g,gp,gq->pq",
        weights,
        ao,
        fxx,
        precision=Precision.HIGHEST,
    )
    k_mat = jnp.nan_to_num(k_mat, nan=0.0, posinf=0.0, neginf=0.0)
    return 0.5 * (k_mat + k_mat.T)


def _range_separated_exchange_matrix_for_density(
    molecule: Any,
    density_spin: Array,
    *,
    omega: Array,
    fallback_omega_values: tuple[float, ...] | None,
) -> Array:
    nu_cache = getattr(molecule, "hfx_nu", None)
    if nu_cache is None:
        raise _range_separated_aux_error()
    ao = jnp.asarray(molecule.ao)
    weights = jnp.asarray(molecule.grid.weights)
    nu = jnp.asarray(nu_cache)
    omega_values = _molecule_hfx_omega_values(
        molecule,
        fallback_omega_values,
        count=int(nu.shape[0]),
    )
    nu_interp = _interpolate_over_omega_axis(nu, omega_values, omega)
    density_spin = jnp.asarray(density_spin)
    e = jnp.einsum("gp,pq->gq", ao, density_spin, precision=Precision.HIGHEST)
    fxx = jnp.einsum("gbc,gc->gb", nu_interp, e, precision=Precision.HIGHEST)
    k_mat = jnp.einsum(
        "g,gp,gq->pq",
        weights,
        ao,
        fxx,
        precision=Precision.HIGHEST,
    )
    k_mat = jnp.nan_to_num(k_mat, nan=0.0, posinf=0.0, neginf=0.0)
    return 0.5 * (k_mat + k_mat.T)


@lru_cache(maxsize=64)
def _point_xc_value_and_grad_kernel(
    xc_spec: str,
    xc_kind: str,
    density_floor: float,
) -> Callable[[Array], tuple[Array, Array]]:
    xc_spec_norm = str(xc_spec)
    xc_kind_norm = str(xc_kind)
    density_floor_value = float(density_floor)

    def point_energy(variables: Array, omega: Array) -> Array:
        rho_point = jnp.maximum(variables[0], density_floor_value)
        if xc_kind_norm == "LDA":
            grad_point = jnp.zeros((3,), dtype=variables.dtype)
            tau_point = jnp.asarray(0.0, dtype=variables.dtype)
        elif xc_kind_norm == "GGA":
            grad_point = variables[1:4]
            tau_point = jnp.asarray(0.0, dtype=variables.dtype)
        elif xc_kind_norm == "MGGA":
            grad_point = variables[1:4]
            tau_point = jnp.maximum(variables[4], 0.0)
        else:
            raise ValueError(f"Unsupported XC kind={xc_kind_norm!r}.")
        features = restricted_feature_bundle_from_rho_grad_tau(
            rho_point,
            grad_point,
            tau_point,
            density_floor=density_floor_value,
        )
        return eval_xc_energy_density(xc_spec_norm, features, omega=omega)

    mapped = jax.vmap(jax.value_and_grad(point_energy, argnums=0), in_axes=(0, None))
    if _xc_spec_uses_wpbeh(xc_spec_norm):
        # jax_xc marks GGA_X_WPBEH as too expensive to JIT because of E1_scaled.
        return mapped
    return jax.jit(mapped)


def _spin_density_and_gradient(
    ao: Array,
    ao_deriv1: Array,
    density_spin: Array,
) -> tuple[Array, Array]:
    rho = jnp.einsum(
        "rp,pq,rq->r",
        ao,
        density_spin,
        ao,
        precision=Precision.HIGHEST,
    )
    grad = 2.0 * jnp.einsum(
        "xrp,pq,rq->rx",
        ao_deriv1[1:4],
        density_spin,
        ao,
        precision=Precision.HIGHEST,
    )
    return rho, grad


@lru_cache(maxsize=64)
def _point_unrestricted_xc_value_and_grad_kernel(
    xc_spec: str,
    xc_kind: str,
    density_floor: float,
) -> Callable[[Array], tuple[Array, Array]]:
    xc_spec_norm = str(xc_spec)
    xc_kind_norm = str(xc_kind).upper()
    density_floor_value = float(density_floor)

    def point_energy(variables: Array, omega: Array) -> Array:
        rho_a = jnp.maximum(variables[0], density_floor_value)
        rho_b = jnp.maximum(variables[1], density_floor_value)
        if xc_kind_norm == "LDA":
            grad_a = jnp.zeros((3,), dtype=variables.dtype)
            grad_b = jnp.zeros((3,), dtype=variables.dtype)
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
        return eval_xc_energy_density(xc_spec_norm, features, omega=omega)

    return jax.jit(jax.vmap(jax.value_and_grad(point_energy, argnums=0), in_axes=(0, None)))


@lru_cache(maxsize=64)
def _point_unrestricted_xc_value_and_grads_kernel(
    xc_spec: str,
    xc_kind: str,
    density_floor: float,
) -> Callable[[Array, Array], tuple[Array, Array, Array]]:
    xc_spec_norm = str(xc_spec)
    xc_kind_norm = str(xc_kind).upper()
    density_floor_value = float(density_floor)

    def point_energy(variables: Array, omega: Array) -> Array:
        rho_a = jnp.maximum(variables[0], density_floor_value)
        rho_b = jnp.maximum(variables[1], density_floor_value)
        if xc_kind_norm == "LDA":
            grad_a = jnp.zeros((3,), dtype=variables.dtype)
            grad_b = jnp.zeros((3,), dtype=variables.dtype)
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
        return eval_xc_energy_density(xc_spec_norm, features, omega=omega)

    point_value_and_grads = jax.value_and_grad(point_energy, argnums=(0, 1))

    def mapped(variables: Array, omega: Array) -> tuple[Array, Array, Array]:
        point_exc, (point_grad, point_omega_grad) = jax.vmap(
            point_value_and_grads,
            in_axes=(0, None),
        )(variables, omega)
        return point_exc, point_grad, point_omega_grad

    if _xc_spec_uses_wpbeh(xc_spec_norm):
        # Avoid multi-minute XLA compilation for the generated WPBEH expression.
        return mapped
    return jax.jit(mapped)


@lru_cache(maxsize=64)
def _point_unrestricted_xc_energy_kernel(
    xc_spec: str,
    xc_kind: str,
    density_floor: float,
) -> Callable[[Array], Array]:
    xc_spec_norm = str(xc_spec)
    xc_kind_norm = str(xc_kind).upper()
    density_floor_value = float(density_floor)

    def point_energy(variables: Array, omega: Array) -> Array:
        rho_a = jnp.maximum(variables[0], density_floor_value)
        rho_b = jnp.maximum(variables[1], density_floor_value)
        if xc_kind_norm == "LDA":
            grad_a = jnp.zeros((3,), dtype=variables.dtype)
            grad_b = jnp.zeros((3,), dtype=variables.dtype)
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
        return eval_xc_energy_density(xc_spec_norm, features, omega=omega)

    mapped = jax.vmap(point_energy, in_axes=(0, None))
    if _xc_spec_uses_wpbeh(xc_spec_norm):
        return mapped
    return jax.jit(mapped)


def _local_xc_energy_and_components(
    molecule: Any,
    *,
    xc_spec: str,
    density_floor: float,
    potential_clip: float | None,
    omega: Array | float | None = None,
) -> tuple[Array, Array, Array, str]:
    features, total_gradient = restricted_grid_features_with_gradients(molecule)
    rho = jnp.maximum(features.rho, density_floor)
    tau = jnp.maximum(features.tau_a + features.tau_b, 0.0)
    weights = jnp.asarray(molecule.grid.weights)
    kind = str(xc_type(xc_spec)).upper()
    if kind == "HF":
        zeros = jnp.zeros_like(rho)
        return jnp.asarray(0.0, dtype=rho.dtype), zeros, jnp.zeros((rho.shape[0], 3), dtype=rho.dtype), "LDA"

    if kind == "LDA":
        response_variables = rho[..., None]
    elif kind == "GGA":
        response_variables = jnp.concatenate([rho[..., None], total_gradient], axis=-1)
    elif kind == "MGGA":
        response_variables = jnp.concatenate(
            [rho[..., None], total_gradient, tau[..., None]],
            axis=-1,
        )
    else:
        raise ValueError(f"Unsupported XC kind={kind!r}.")

    point_exc, point_grad = _point_xc_value_and_grad_kernel(
        xc_spec,
        kind,
        density_floor,
    )(response_variables, jnp.asarray(0.4 if omega is None else omega, dtype=rho.dtype))
    point_exc = jnp.nan_to_num(point_exc, nan=0.0, posinf=0.0, neginf=0.0)
    point_grad = jnp.nan_to_num(point_grad, nan=0.0, posinf=0.0, neginf=0.0)
    exc = jnp.tensordot(weights, point_exc, axes=(0, 0))

    mask = rho > density_floor
    vxc_rho = jnp.where(mask, point_grad[:, 0], 0.0)
    if kind in {"GGA", "MGGA"}:
        vxc_grad = jnp.where(mask[:, None], point_grad[:, 1:4], 0.0)
    else:
        vxc_grad = jnp.zeros((rho.shape[0], 3), dtype=rho.dtype)

    if potential_clip is not None:
        clip = jnp.asarray(potential_clip, dtype=rho.dtype)
        vxc_rho = jnp.clip(vxc_rho, -clip, clip)
        vxc_grad = jnp.clip(vxc_grad, -clip, clip)
    return exc, vxc_rho, vxc_grad, kind


def _local_xc_energy_and_components_unrestricted(
    molecule: Any,
    *,
    xc_spec: str,
    density_floor: float,
    potential_clip: float | None,
    omega: Array | float | None = None,
) -> tuple[Array, Array, Array, Array, Array, str]:
    kind = str(xc_type(xc_spec)).upper()
    ao = jnp.asarray(molecule.ao)
    ao_deriv1 = getattr(molecule, "ao_deriv1", None)
    if ao_deriv1 is None:
        raise AttributeError("Unrestricted local XC requires molecule.ao_deriv1.")
    ao_deriv1 = jnp.asarray(ao_deriv1)
    dm_a, dm_b = _restricted_spin_density_blocks(molecule)
    rho_a, grad_a = _spin_density_and_gradient(ao, ao_deriv1, dm_a)
    rho_b, grad_b = _spin_density_and_gradient(ao, ao_deriv1, dm_b)
    rho_total = rho_a + rho_b
    weights = jnp.asarray(molecule.grid.weights)
    if kind == "HF":
        zeros = jnp.zeros_like(rho_total)
        zero_grads = jnp.zeros((rho_total.shape[0], 3), dtype=rho_total.dtype)
        return jnp.asarray(0.0, dtype=rho_total.dtype), zeros, zeros, zero_grads, zero_grads, "LDA"
    if kind == "MGGA":
        raise NotImplementedError(
            "Unrestricted local XC components currently support LDA/GGA/HF only."
        )

    if kind == "LDA":
        response_variables = jnp.stack([rho_a, rho_b], axis=-1)
    elif kind == "GGA":
        response_variables = jnp.concatenate(
            [rho_a[..., None], rho_b[..., None], grad_a, grad_b],
            axis=-1,
        )
    else:
        raise ValueError(f"Unsupported XC kind={kind!r}.")

    point_exc, point_grad = _point_unrestricted_xc_value_and_grad_kernel(
        xc_spec,
        kind,
        density_floor,
    )(response_variables, jnp.asarray(0.4 if omega is None else omega, dtype=rho_total.dtype))
    point_exc = jnp.nan_to_num(point_exc, nan=0.0, posinf=0.0, neginf=0.0)
    point_grad = jnp.nan_to_num(point_grad, nan=0.0, posinf=0.0, neginf=0.0)
    exc = jnp.tensordot(weights, point_exc, axes=(0, 0))

    mask = rho_total > density_floor
    v_rho_a = jnp.where(mask, point_grad[:, 0], 0.0)
    v_rho_b = jnp.where(mask, point_grad[:, 1], 0.0)
    if kind == "GGA":
        v_grad_a = jnp.where(mask[:, None], point_grad[:, 2:5], 0.0)
        v_grad_b = jnp.where(mask[:, None], point_grad[:, 5:8], 0.0)
    else:
        v_grad_a = jnp.zeros((rho_total.shape[0], 3), dtype=rho_total.dtype)
        v_grad_b = jnp.zeros((rho_total.shape[0], 3), dtype=rho_total.dtype)

    if potential_clip is not None:
        clip = jnp.asarray(potential_clip, dtype=rho_total.dtype)
        v_rho_a = jnp.clip(v_rho_a, -clip, clip)
        v_rho_b = jnp.clip(v_rho_b, -clip, clip)
        v_grad_a = jnp.clip(v_grad_a, -clip, clip)
        v_grad_b = jnp.clip(v_grad_b, -clip, clip)
    return exc, v_rho_a, v_rho_b, v_grad_a, v_grad_b, kind


def _local_xc_energy_unrestricted(
    molecule: Any,
    *,
    xc_spec: str,
    density_floor: float,
    omega: Array | float | None = None,
) -> Array:
    kind = str(xc_type(xc_spec)).upper()
    ao = jnp.asarray(molecule.ao)
    ao_deriv1 = getattr(molecule, "ao_deriv1", None)
    if ao_deriv1 is None:
        raise AttributeError("Unrestricted local XC requires molecule.ao_deriv1.")
    ao_deriv1 = jnp.asarray(ao_deriv1)
    dm_a, dm_b = _restricted_spin_density_blocks(molecule)
    rho_a, grad_a = _spin_density_and_gradient(ao, ao_deriv1, dm_a)
    rho_b, grad_b = _spin_density_and_gradient(ao, ao_deriv1, dm_b)
    rho_total = rho_a + rho_b
    weights = jnp.asarray(molecule.grid.weights)
    if kind == "HF":
        return jnp.asarray(0.0, dtype=rho_total.dtype)
    if kind == "MGGA":
        raise NotImplementedError(
            "Unrestricted local XC energy currently supports LDA/GGA/HF only."
        )
    if kind == "LDA":
        response_variables = jnp.stack([rho_a, rho_b], axis=-1)
    elif kind == "GGA":
        response_variables = jnp.concatenate(
            [rho_a[..., None], rho_b[..., None], grad_a, grad_b],
            axis=-1,
        )
    else:
        raise ValueError(f"Unsupported XC kind={kind!r}.")
    energy_kernel = _point_unrestricted_xc_energy_kernel(
        xc_spec,
        kind,
        density_floor,
    )
    value_and_grads_kernel = _point_unrestricted_xc_value_and_grads_kernel(
        xc_spec,
        kind,
        density_floor,
    )
    omega_value = jnp.asarray(0.4 if omega is None else omega, dtype=rho_total.dtype)

    @jax.custom_jvp
    def _energy_from_response(variables: Array, omega_arg: Array) -> Array:
        point_exc = energy_kernel(variables, omega_arg)
        point_exc = jnp.nan_to_num(point_exc, nan=0.0, posinf=0.0, neginf=0.0)
        return jnp.tensordot(weights, point_exc, axes=(0, 0))

    @_energy_from_response.defjvp
    def _energy_from_response_jvp(primals: tuple[Array], tangents: tuple[Array]):
        variables, omega_arg = primals
        variables_dot, omega_dot = tangents
        point_exc, point_grad, point_omega_grad = value_and_grads_kernel(variables, omega_arg)
        point_exc = jnp.nan_to_num(point_exc, nan=0.0, posinf=0.0, neginf=0.0)
        point_grad = jnp.nan_to_num(point_grad, nan=0.0, posinf=0.0, neginf=0.0)
        point_omega_grad = jnp.nan_to_num(
            point_omega_grad,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        primal = jnp.tensordot(weights, point_exc, axes=(0, 0))
        point_dot = jnp.sum(point_grad * variables_dot, axis=-1) + point_omega_grad * omega_dot
        tangent = jnp.tensordot(weights, point_dot, axes=(0, 0))
        return primal, tangent

    return _energy_from_response(response_variables, omega_value)


class RSHParameterHead(nn.Module):
    hidden_dims: Sequence[int] = ()
    activation: Callable[[Array], Array] = nn.tanh

    @nn.compact
    def __call__(self, descriptor: Array) -> Array:
        x = jnp.asarray(descriptor)
        if x.ndim == 0:
            x = x[None]
        for index, width in enumerate(self.hidden_dims):
            x = nn.Dense(width, name=f"hidden_{index}")(x)
            x = self.activation(x)
        return nn.Dense(3, name="output")(x)


class AtomwiseRSHParameterHead(nn.Module):
    atom_hidden_dims: Sequence[int] = (32, 32)
    pooled_hidden_dims: Sequence[int] = (32,)
    activation: Callable[[Array], Array] = nn.tanh
    embedding_dim: int = 8
    max_atomic_number: int = 100

    @nn.compact
    def __call__(self, descriptor_inputs: Any) -> Array:
        if not isinstance(descriptor_inputs, dict):
            raise TypeError(
                "AtomwiseRSHParameterHead expects a dict with atom_descriptors and atom_charges."
            )
        atom_descriptors = jnp.asarray(descriptor_inputs["atom_descriptors"])
        atom_charges = jnp.asarray(descriptor_inputs["atom_charges"], dtype=jnp.int32)
        if atom_descriptors.ndim != 2:
            raise ValueError(
                f"atom_descriptors must have shape (natoms, n_features), got {atom_descriptors.shape}."
            )
        if atom_charges.ndim != 1 or atom_charges.shape[0] != atom_descriptors.shape[0]:
            raise ValueError(
                "atom_charges must have shape (natoms,) matching atom_descriptors."
            )

        x = atom_descriptors
        if self.embedding_dim > 0:
            charges = jnp.clip(atom_charges, 0, int(self.max_atomic_number))
            embedding = nn.Embed(
                num_embeddings=int(self.max_atomic_number) + 1,
                features=int(self.embedding_dim),
                name="atomic_number_embedding",
            )(charges)
            x = jnp.concatenate([x, embedding], axis=-1)

        for index, width in enumerate(self.atom_hidden_dims):
            x = nn.Dense(width, name=f"atom_hidden_{index}")(x)
            x = self.activation(x)

        pooled = jnp.mean(x, axis=0)
        for index, width in enumerate(self.pooled_hidden_dims):
            pooled = nn.Dense(width, name=f"pooled_hidden_{index}")(pooled)
            pooled = self.activation(pooled)
        return nn.Dense(3, name="output")(pooled)


@dataclass(frozen=True)
class BoundTrainableRSHFunctional:
    template: RSHFunctionalTemplate
    local_xc_spec: str
    resolved_params: ResolvedRSHParameters
    density_floor: float = 1e-12
    potential_clip: float | None = None
    fallback_omega_values: tuple[float, ...] | None = None

    @property
    def exact_exchange_fraction(self) -> Array:
        return self.resolved_params.exact_exchange_fraction

    def to_pyscf_spec(self):
        return make_pyscf_rsh_spec(
            xc_description=self.local_xc_spec,
            xctype=str(xc_type(self.local_xc_spec)).upper(),
            resolved_params=self.resolved_params,
        )

    def local_potential(self, density: Array) -> Array:
        del density
        raise AttributeError(
            "BoundTrainableRSHFunctional.local_potential depends on the molecule grid. "
            "Use grid_potential_components(molecule) or scf_contributions(molecule)."
        )

    def local_kernel(self, density: Array) -> Array:
        del density
        return jnp.asarray(0.0)

    def grid_potential_components(self, molecule: Any) -> tuple[Array, Array, Array]:
        _, v_rho, v_grad, _ = _local_xc_energy_and_components(
            molecule,
            xc_spec=self.local_xc_spec,
            density_floor=self.density_floor,
            potential_clip=self.potential_clip,
            omega=self.resolved_params.omega,
        )
        return v_rho, v_grad, jnp.zeros_like(v_rho)

    def energy_from_molecule(self, molecule: Any) -> Array:
        omega = jnp.asarray(self.resolved_params.omega)
        local_energy = _local_xc_energy_unrestricted(
            molecule,
            xc_spec=self.local_xc_spec,
            density_floor=self.density_floor,
            omega=omega,
        )
        sr = jnp.asarray(self.resolved_params.sr_hf_fraction)
        lr = jnp.asarray(self.resolved_params.lr_hf_fraction)
        full_exchange = _exact_exchange_energy(molecule)
        beta = lr - sr
        if _has_range_separated_aux(molecule):
            long_range_exchange = _range_separated_exchange_energy(
                molecule,
                omega=omega,
                fallback_omega_values=self.fallback_omega_values,
            )
        elif _is_concrete_scalar_near_zero(beta):
            long_range_exchange = jnp.asarray(0.0, dtype=full_exchange.dtype)
        else:
            raise _range_separated_aux_error()
        exchange_correction = sr * full_exchange + beta * long_range_exchange
        return local_energy + exchange_correction

    def unrestricted_scf_components(
        self,
        molecule: Any,
    ) -> tuple[Array, Array, Array, Array, str, Array, Array, Array]:
        omega = jnp.asarray(self.resolved_params.omega)
        _, v_rho_a, v_rho_b, v_grad_a, v_grad_b, kind = _local_xc_energy_and_components_unrestricted(
            molecule,
            xc_spec=self.local_xc_spec,
            density_floor=self.density_floor,
            potential_clip=self.potential_clip,
            omega=omega,
        )
        dm_a, dm_b = _restricted_spin_density_blocks(molecule)
        sr = jnp.asarray(self.resolved_params.sr_hf_fraction)
        lr = jnp.asarray(self.resolved_params.lr_hf_fraction)
        # UKS spin blocks use D_alpha/D_beta directly, so dE_x[D_spin]/dD_spin
        # gives -K[D_spin]. The restricted total-density path keeps the 1/2
        # factor because D_total = D_alpha + D_beta for a closed shell.
        extra_factor = -(lr - sr)
        has_range_aux = _has_range_separated_aux(molecule)
        if not has_range_aux:
            if _is_concrete_scalar_near_zero(extra_factor):
                return (
                    v_rho_a,
                    v_rho_b,
                    v_grad_a,
                    v_grad_b,
                    kind,
                    sr,
                    jnp.zeros_like(dm_a),
                    jnp.zeros_like(dm_b),
                )
            raise _range_separated_aux_error()
        use_extra = jnp.abs(extra_factor) > 1e-12

        def _zero_extra(_: None) -> tuple[Array, Array]:
            return jnp.zeros_like(dm_a), jnp.zeros_like(dm_b)

        def _build_extra(_: None) -> tuple[Array, Array]:
            extra_fock_a = extra_factor * _range_separated_exchange_matrix_for_density(
                molecule,
                dm_a,
                omega=omega,
                fallback_omega_values=self.fallback_omega_values,
            )
            extra_fock_b = extra_factor * _range_separated_exchange_matrix_for_density(
                molecule,
                dm_b,
                omega=omega,
                fallback_omega_values=self.fallback_omega_values,
            )
            return extra_fock_a, extra_fock_b

        extra_fock_a, extra_fock_b = jax.lax.cond(
            use_extra,
            _build_extra,
            _zero_extra,
            operand=None,
        )
        return v_rho_a, v_rho_b, v_grad_a, v_grad_b, kind, sr, extra_fock_a, extra_fock_b

    def scf_contributions(self, molecule: Any) -> SCFXCContributions:
        omega = jnp.asarray(self.resolved_params.omega)
        _, v_rho, v_grad, kind = _local_xc_energy_and_components(
            molecule,
            xc_spec=self.local_xc_spec,
            density_floor=self.density_floor,
            potential_clip=self.potential_clip,
            omega=omega,
        )
        sr = jnp.asarray(self.resolved_params.sr_hf_fraction)
        lr = jnp.asarray(self.resolved_params.lr_hf_fraction)
        extra_factor = -0.5 * (lr - sr)
        if _has_range_separated_aux(molecule):
            extra_fock = extra_factor * _range_separated_exchange_matrix(
                molecule,
                omega=omega,
                fallback_omega_values=self.fallback_omega_values,
            )
        elif _is_concrete_scalar_near_zero(extra_factor):
            density_total = jnp.asarray(molecule.rdm1)
            if density_total.ndim == 3:
                density_total = density_total.sum(axis=0)
            extra_fock = jnp.zeros_like(density_total)
        else:
            raise _range_separated_aux_error()
        return SCFXCContributions(
            v_rho=v_rho,
            v_grad=v_grad,
            xc_kind=kind,
            full_hf_fraction=sr,
            extra_fock_matrix=extra_fock,
            exact_exchange_fraction=sr,
            resolved_xc=self,
        )


@dataclass(frozen=True)
class TrainableRSHFunctional:
    model: nn.Module
    template: RSHFunctionalTemplate
    local_xc_spec: str = "pbe"
    descriptor_fn: Callable[[Any | None], Any] = _constant_rsh_descriptor
    head_type: str = "mlp"
    density_floor: float = 1e-12
    potential_clip: float | None = 20.0
    fallback_omega_values: tuple[float, ...] | None = None

    def _model_args(self, molecule: Any | None = None) -> tuple[Any, ...]:
        descriptor = self.descriptor_fn(molecule)
        if self.head_type == "mlp":
            return (descriptor,)
        if self.head_type != "gnn":
            raise ValueError(f"Unsupported RSH head_type {self.head_type!r}.")
        if not isinstance(descriptor, dict):
            raise TypeError("GNN RSH head expects descriptor_fn to return a dict.")
        missing = {"atom_descriptors", "atom_coords"} - set(descriptor)
        if missing:
            raise KeyError(f"GNN RSH descriptor is missing required keys: {sorted(missing)}.")

        atom_descriptors = jnp.asarray(descriptor["atom_descriptors"])
        atom_coords = jnp.asarray(descriptor["atom_coords"], dtype=jnp.float32)
        if atom_descriptors.ndim == 2:
            atom_descriptors = atom_descriptors[None, :, :]
        elif atom_descriptors.ndim != 3:
            raise ValueError(
                "GNN atom_descriptors must have shape (natoms, features) or "
                f"(batch, natoms, features), got {atom_descriptors.shape}."
            )
        if atom_coords.ndim == 2:
            atom_coords = atom_coords[None, :, :]
        elif atom_coords.ndim != 3:
            raise ValueError(
                "GNN atom_coords must have shape (natoms, 3) or "
                f"(batch, natoms, 3), got {atom_coords.shape}."
            )
        if atom_coords.shape[-1] != 3:
            raise ValueError(
                f"GNN atom_coords last dimension must be 3, got {atom_coords.shape}."
            )
        if atom_coords.shape[:2] != atom_descriptors.shape[:2]:
            raise ValueError("GNN atom_coords must match atom_descriptors batch and atom dimensions.")
        return atom_descriptors, atom_coords

    def init(self, rng: PRNGKeyArray, molecule: Any | None = None) -> PyTree:
        return self.model.init(rng, *self._model_args(molecule))

    def init_from_molecule(self, rng: PRNGKeyArray, molecule: Any) -> PyTree:
        return self.init(rng, molecule)

    def _raw_outputs(self, params: PyTree, molecule: Any | None = None) -> Array:
        raw = self.model.apply(params, *self._model_args(molecule))
        raw = jnp.asarray(raw)
        if raw.ndim == 2 and raw.shape[0] == 1 and raw.shape[1] == 3:
            raw = raw[0]
        if raw.ndim != 1 or raw.shape[0] != 3:
            raise ValueError(f"RSH parameter head must return shape (3,), got {raw.shape}.")
        return raw

    def resolve_parameters(
        self,
        params: PyTree,
        molecule: Any | None = None,
    ) -> ResolvedRSHParameters:
        raw = self._raw_outputs(params, molecule)
        sr = _sigmoid_to_interval(raw[0], self.template.sr_hf_bounds)
        omega = _sigmoid_to_interval(raw[2], self.template.omega_bounds)
        if self.template.monotonic_lr_hf:
            lr_delta = jax.nn.sigmoid(raw[1])
            lr = sr + (1.0 - sr) * lr_delta
        else:
            lr = _sigmoid_to_interval(raw[1], self.template.lr_hf_bounds)
        lr = jnp.clip(lr, self.template.lr_hf_bounds[0], self.template.lr_hf_bounds[1])
        return ResolvedRSHParameters(
            sr_hf_fraction=sr,
            lr_hf_fraction=lr,
            omega=omega,
        )

    def raw_parameters_from_resolved(self, resolved: ResolvedRSHParameters) -> Array:
        sr = _logit_from_interval(resolved.sr_hf_fraction, self.template.sr_hf_bounds)
        omega = _logit_from_interval(resolved.omega, self.template.omega_bounds)
        if self.template.monotonic_lr_hf:
            sr_value = jnp.asarray(resolved.sr_hf_fraction, dtype=jnp.float32)
            lr_value = jnp.asarray(resolved.lr_hf_fraction, dtype=jnp.float32)
            ratio = (lr_value - sr_value) / jnp.maximum(1.0 - sr_value, 1e-6)
            ratio = jnp.clip(ratio, 1e-6, 1.0 - 1e-6)
            lr_raw = jnp.log(ratio / (1.0 - ratio))
        else:
            lr_raw = _logit_from_interval(resolved.lr_hf_fraction, self.template.lr_hf_bounds)
        return jnp.asarray([sr, lr_raw, omega], dtype=jnp.float32)

    def params_with_raw_output(
        self,
        params: PyTree,
        raw_output: Array,
        molecule: Any | None = None,
        *,
        preserve_network: bool = True,
    ) -> PyTree:
        target_raw = jnp.asarray(raw_output, dtype=jnp.float32)
        if target_raw.shape != (3,):
            raise ValueError(f"raw_output must have shape (3,), got {target_raw.shape}.")
        params_out = params
        if "params" not in params_out or "output" not in params_out["params"]:
            raise ValueError(
                "params_with_raw_output currently expects a flax params tree with params['output']."
            )
        params_out = jax.tree_util.tree_map(lambda x: x, params_out)
        if preserve_network:
            current_raw = self._raw_outputs(params_out, molecule)
            bias_delta = target_raw - current_raw
            params_out["params"]["output"]["bias"] = (
                params_out["params"]["output"]["bias"] + bias_delta.astype(
                    params_out["params"]["output"]["bias"].dtype
                )
            )
        else:
            params_out["params"]["output"]["bias"] = target_raw.astype(
                params_out["params"]["output"]["bias"].dtype
            )
            params_out["params"]["output"]["kernel"] = jnp.zeros_like(
                params_out["params"]["output"]["kernel"]
            )
        return params_out

    def params_with_resolved(
        self,
        params: PyTree,
        resolved: ResolvedRSHParameters,
        molecule: Any | None = None,
        *,
        preserve_network: bool = True,
    ) -> PyTree:
        target_raw = self.raw_parameters_from_resolved(resolved)
        return self.params_with_raw_output(
            params,
            target_raw,
            molecule=molecule,
            preserve_network=preserve_network,
        )

    def bind_to_molecule(self, params: PyTree, molecule: Any) -> BoundTrainableRSHFunctional:
        return BoundTrainableRSHFunctional(
            template=self.template,
            local_xc_spec=self.local_xc_spec,
            resolved_params=self.resolve_parameters(params, molecule),
            density_floor=self.density_floor,
            potential_clip=self.potential_clip,
            fallback_omega_values=self.fallback_omega_values,
        )

    def bind_to_molecule_for_scf(
        self,
        params: PyTree,
        molecule: Any,
    ) -> BoundTrainableRSHFunctional:
        return self.bind_to_molecule(params, molecule)

    def energy_from_molecule(self, params: PyTree, molecule: Any) -> Array:
        return self.bind_to_molecule(params, molecule).energy_from_molecule(molecule)


def make_minimal_trainable_rsh_functional(
    *,
    local_xc_spec: str = "pbe",
    hidden_dims: Sequence[int] = (),
    descriptor_fn: Callable[[Any | None], Any] = _constant_rsh_descriptor,
    template: RSHFunctionalTemplate | None = None,
    fallback_omega_values: tuple[float, ...] | None = None,
    name: str = "minimal_trainable_rsh",
) -> TrainableRSHFunctional:
    resolved_template = template or RSHFunctionalTemplate(
        name=name,
        local_backend="jax_libxc",
        exchange_backend_id=local_xc_spec,
        correlation_backend_id=local_xc_spec,
        default_sr_hf_fraction=0.20,
        default_lr_hf_fraction=0.65,
        default_omega=0.30,
        omega_bounds=(0.05, 0.70),
        sr_hf_bounds=(0.0, 0.60),
        lr_hf_bounds=(0.0, 1.0),
    )
    return TrainableRSHFunctional(
        model=RSHParameterHead(hidden_dims=hidden_dims),
        template=resolved_template,
        local_xc_spec=local_xc_spec,
        descriptor_fn=descriptor_fn,
        fallback_omega_values=fallback_omega_values,
    )


def make_atom_centered_density_rsh_functional(
    *,
    local_xc_spec: str = "pbe",
    descriptor_config: AtomCenteredDensityDescriptorConfig | None = None,
    atom_hidden_dims: Sequence[int] = (32, 32),
    pooled_hidden_dims: Sequence[int] = (32,),
    embedding_dim: int = 8,
    max_atomic_number: int = 100,
    template: RSHFunctionalTemplate | None = None,
    fallback_omega_values: tuple[float, ...] | None = None,
    name: str = "atom_centered_density_rsh",
) -> TrainableRSHFunctional:
    resolved_template = template or RSHFunctionalTemplate(
        name=name,
        local_backend="jax_libxc",
        exchange_backend_id=local_xc_spec,
        correlation_backend_id=local_xc_spec,
        default_sr_hf_fraction=0.20,
        default_lr_hf_fraction=0.65,
        default_omega=0.30,
        omega_bounds=(0.05, 0.70),
        sr_hf_bounds=(0.0, 0.60),
        lr_hf_bounds=(0.0, 1.0),
    )
    return TrainableRSHFunctional(
        model=AtomwiseRSHParameterHead(
            atom_hidden_dims=atom_hidden_dims,
            pooled_hidden_dims=pooled_hidden_dims,
            embedding_dim=embedding_dim,
            max_atomic_number=max_atomic_number,
        ),
        template=resolved_template,
        local_xc_spec=local_xc_spec,
        descriptor_fn=make_atom_centered_density_descriptor_fn(descriptor_config),
        fallback_omega_values=fallback_omega_values,
    )


def make_gnn_rsh_functional(
    *,
    local_xc_spec: str = "pbe",
    descriptor_config: AtomCenteredDensityDescriptorConfig | None = None,
    node_hidden_dims: Sequence[int] = (32, 32),
    global_hidden_dims: Sequence[int] = (32, 16),
    num_heads: int = 4,
    num_layers: int | None = None,
    num_interaction_blocks: int | None = None,
    qkv_features: int | None = None,
    ffn_dim: int | None = None,
    ffn_expansion: int = 4,
    lambda_init: float = 5.0,
    dropout_rate: float = 0.0,
    template: RSHFunctionalTemplate | None = None,
    density_floor: float = 1e-12,
    potential_clip: float | None = 20.0,
    fallback_omega_values: tuple[float, ...] | None = None,
    name: str = "gnn_atom_centered_density_rsh",
) -> TrainableRSHFunctional:
    if num_layers is not None and num_interaction_blocks is not None:
        if int(num_layers) != int(num_interaction_blocks):
            raise ValueError("num_layers and num_interaction_blocks disagree.")
    resolved_num_layers = (
        num_layers
        if num_layers is not None
        else (num_interaction_blocks if num_interaction_blocks is not None else 1)
    )
    resolved_template = template or RSHFunctionalTemplate(
        name=name,
        local_backend="jax_libxc",
        exchange_backend_id=local_xc_spec,
        correlation_backend_id=local_xc_spec,
        default_sr_hf_fraction=0.20,
        default_lr_hf_fraction=0.65,
        default_omega=0.30,
        omega_bounds=(0.05, 0.70),
        sr_hf_bounds=(0.0, 0.60),
        lr_hf_bounds=(0.0, 1.0),
    )
    return TrainableRSHFunctional(
        model=RSHGNNHead(
            node_hidden_dims=node_hidden_dims,
            global_hidden_dims=global_hidden_dims,
            num_heads=num_heads,
            num_layers=int(resolved_num_layers),
            qkv_features=qkv_features,
            ffn_dim=ffn_dim,
            ffn_expansion=ffn_expansion,
            lambda_init=lambda_init,
            dropout_rate=dropout_rate,
        ),
        template=resolved_template,
        local_xc_spec=local_xc_spec,
        descriptor_fn=make_atom_centered_density_descriptor_fn(descriptor_config),
        head_type="gnn",
        density_floor=density_floor,
        potential_clip=potential_clip,
        fallback_omega_values=fallback_omega_values,
    )


__all__ = [
    "AtomwiseRSHParameterHead",
    "BoundTrainableRSHFunctional",
    "RSHParameterHead",
    "TrainableRSHFunctional",
    "make_atom_centered_density_rsh_functional",
    "make_gnn_rsh_functional",
    "make_minimal_trainable_rsh_functional",
]
