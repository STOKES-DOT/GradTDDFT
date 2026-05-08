from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jax.scipy.linalg import expm

from ._utils import _restricted_orbital_data


def _restricted_channel_arrays(molecule: Any) -> tuple[Array, Array]:
    mo_coeff = jnp.asarray(molecule.mo_coeff)
    mo_occ = jnp.asarray(molecule.mo_occ)
    if mo_coeff.ndim == 3:
        mo_coeff = mo_coeff[0]
    if mo_occ.ndim == 2:
        mo_occ = mo_occ[0]
    return mo_coeff, mo_occ


def _as_grid_weight(local_weight: Any, weights: Array) -> Array:
    arr = jnp.asarray(local_weight, dtype=weights.dtype)
    if arr.ndim == 0:
        return weights * float(arr)
    if arr.shape != weights.shape:
        raise ValueError(
            f"local_weight must be scalar or have shape {weights.shape}, got {arr.shape}."
        )
    return weights * arr


def _rotate_restricted_mo_coeff(
    mo_coeff: Array,
    *,
    nocc: int,
    kappa_flat: Array,
) -> Array:
    nmo = int(mo_coeff.shape[1])
    nvir = int(nmo - nocc)
    kappa = jnp.asarray(kappa_flat, dtype=mo_coeff.dtype).reshape(nocc, nvir)
    generator = jnp.zeros((nmo, nmo), dtype=mo_coeff.dtype)
    generator = generator.at[:nocc, nocc:].set(kappa)
    generator = generator.at[nocc:, :nocc].set(-kappa.T)
    return mo_coeff @ expm(generator)


def _local_hf_weighted_energy(
    molecule: Any,
    *,
    local_weight: Any,
    omega_index: int,
    occupation_tolerance: float,
    rotated_mo_coeff: Array | None = None,
) -> Array:
    hfx_nu = getattr(molecule, "hfx_nu", None)
    if hfx_nu is None:
        raise ValueError(
            "_local_hf_weighted_energy requires molecule.hfx_nu. "
            "Build the reference with compute_local_hfx_features=True and compute_local_hfx_aux=True."
        )
    ao = jnp.asarray(molecule.ao, dtype=jnp.float64)
    weights = jnp.asarray(molecule.grid.weights, dtype=jnp.float64)
    weighted_grid = _as_grid_weight(local_weight, weights)
    nu = jnp.asarray(hfx_nu, dtype=jnp.float64)[int(omega_index)]

    mo_coeff_ref, mo_occ_ref = _restricted_channel_arrays(molecule)
    nocc = getattr(molecule, "nocc", None)
    if nocc is None:
        nocc = int(np.count_nonzero(np.asarray(mo_occ_ref > occupation_tolerance)))
    nocc = int(nocc)

    if rotated_mo_coeff is None:
        mo_coeff = mo_coeff_ref
    else:
        mo_coeff = jnp.asarray(rotated_mo_coeff, dtype=jnp.float64)

    occ = jnp.asarray(mo_occ_ref[:nocc], dtype=jnp.float64)
    coeff_occ = mo_coeff[:, :nocc]
    dm_spin = (coeff_occ * occ[None, :]) @ coeff_occ.T
    e = ao @ dm_spin
    fxx = jnp.einsum("gbc,gc->gb", nu, e, optimize=True)
    h_spin = -0.5 * jnp.einsum("gb,gb->g", e, fxx, optimize=True)
    h_total = 2.0 * h_spin
    return jnp.dot(weighted_grid, h_total)


def build_restricted_local_hf_khh_tda_matrix(
    molecule: Any,
    *,
    local_weight: Any,
    omega_index: int = 0,
    occupation_tolerance: float = 1e-8,
    fd_step: float = 1e-4,
) -> Array:
    """Experimental TDA-only K_hh block from finite differences of the local HF surrogate.

    This is intentionally an experimental test module:

    - it targets the current DM21-style local HF surrogate in this repository
    - it computes the occupied-virtual orbital-rotation Hessian with JAX autodiff
    - it is suitable for very small systems such as H2/STO-3G
    """

    mo_coeff, mo_occ = _restricted_channel_arrays(molecule)
    nocc = getattr(molecule, "nocc", None)
    if nocc is None:
        nocc = int(np.count_nonzero(np.asarray(mo_occ > occupation_tolerance)))
    nocc = int(nocc)
    nmo = int(mo_coeff.shape[1])
    nvir = int(nmo - nocc)
    dim = int(nocc * nvir)

    kappa0 = jnp.zeros((dim,), dtype=mo_coeff.dtype)
    local_weight_arr = jnp.asarray(local_weight, dtype=jnp.float64)

    def energy_from_kappa(kappa_flat: Array, current_local_weight: Array) -> Array:
        rotated_mo_coeff = _rotate_restricted_mo_coeff(
            mo_coeff,
            nocc=nocc,
            kappa_flat=kappa_flat,
        )
        return _local_hf_weighted_energy(
            molecule,
            local_weight=current_local_weight,
            omega_index=omega_index,
            occupation_tolerance=occupation_tolerance,
            rotated_mo_coeff=rotated_mo_coeff,
        )

    hessian = jax.hessian(energy_from_kappa, argnums=0)(kappa0, local_weight_arr)
    return jnp.asarray(hessian).reshape(nocc, nvir, nocc, nvir)


@dataclass(frozen=True)
class RestrictedLocalHFKhhTDAWrapper:
    """TDA-only wrapper that replaces scalar HF exchange with experimental K_hh."""

    base_xc: Any
    local_weight: Any
    omega_index: int = 0
    fd_step: float = 1e-4
    exact_exchange_fraction: float = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_xc, name)

    def local_hf_fraction(self, density: Any) -> Array:
        density_arr = jnp.asarray(density)
        return jnp.full_like(density_arr, self.exact_exchange_fraction, dtype=density_arr.dtype)

    def grid_hf_fraction(self, molecule: Any) -> Array:
        weights = jnp.asarray(molecule.grid.weights)
        return jnp.full_like(weights, self.exact_exchange_fraction, dtype=weights.dtype)

    def nonlocal_response_matrix(
        self,
        molecule: Any,
        *,
        occupation_tolerance: float = 1e-8,
    ) -> Array:
        return build_restricted_local_hf_khh_tda_matrix(
            molecule,
            local_weight=self.local_weight,
            omega_index=self.omega_index,
            occupation_tolerance=occupation_tolerance,
            fd_step=self.fd_step,
        )


@dataclass(frozen=True)
class LocalHFKhhResponseFunctionalWrapper:
    """Functional wrapper that injects the experimental local-HF K_hh block into TD response."""

    base_functional: Any
    omega_index: int = 0
    fd_step: float = 1e-4
    exact_exchange_fraction: float = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_functional, name)

    def bind_to_molecule(self, params: Any, molecule: Any) -> Any:
        return self.base_functional.bind_to_molecule(params, molecule)

    def bind_to_molecule_for_response(self, params: Any, molecule: Any) -> Any:
        bound_full = self.base_functional.bind_to_molecule(params, molecule)
        local_weight = getattr(bound_full, "local_hf_fraction_values", None)
        if local_weight is None:
            raise ValueError(
                "LocalHFKhhResponseFunctionalWrapper requires base_functional.bind_to_molecule(...) "
                "to expose local_hf_fraction_values. Use response_hf_mode='local_projected'."
            )
        response_binder = getattr(self.base_functional, "bind_to_molecule_for_response", None)
        bound_response = (
            response_binder(params, molecule)
            if callable(response_binder)
            else bound_full
        )
        return RestrictedLocalHFKhhTDAWrapper(
            base_xc=bound_response,
            local_weight=local_weight,
            omega_index=self.omega_index,
            fd_step=self.fd_step,
            exact_exchange_fraction=self.exact_exchange_fraction,
        )
