from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree


XCEnergyFn = Callable[[PyTree, Any, Array], Array]


@dataclass(frozen=True)
class XCEnergyPotentialResult:
    """XC energy and density-matrix derivative for SCF Fock construction."""

    xc_energy: Array
    vxc_matrix: Array
    exact_exchange_fraction: Array
    extra_fock_matrix: Array
    aux: Any | None = None


def xc_energy_and_potential_from_density(
    params: PyTree,
    *,
    molecule: Any,
    density: Array,
    xc_energy_fn: XCEnergyFn,
    exact_exchange_fraction: Array | float = 0.0,
    extra_fock_matrix: Array | None = None,
    symmetrize: bool = True,
    has_aux: bool = False,
) -> XCEnergyPotentialResult:
    """Build `Vxc = d Exc / d density` from a GradDFT-style energy callback."""

    density_arr = jnp.asarray(density)

    aux = None
    if has_aux:

        def _energy_for_density(density_var: Array) -> tuple[Array, Any]:
            energy, local_aux = xc_energy_fn(params, molecule, density_var)
            return jnp.asarray(energy), local_aux

        (xc_energy, aux), vxc_matrix = jax.value_and_grad(
            _energy_for_density,
            has_aux=True,
        )(density_arr)
    else:

        def _energy_for_density(density_var: Array) -> Array:
            return jnp.asarray(xc_energy_fn(params, molecule, density_var))

        xc_energy, vxc_matrix = jax.value_and_grad(_energy_for_density)(density_arr)
    vxc_matrix = jnp.nan_to_num(vxc_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if symmetrize:
        vxc_matrix = 0.5 * (vxc_matrix + vxc_matrix.T)

    if extra_fock_matrix is None:
        extra_fock = jnp.zeros_like(vxc_matrix)
    else:
        extra_fock = jnp.asarray(extra_fock_matrix, dtype=vxc_matrix.dtype)
        extra_fock = jnp.nan_to_num(extra_fock, nan=0.0, posinf=0.0, neginf=0.0)
        if symmetrize:
            extra_fock = 0.5 * (extra_fock + extra_fock.T)

    alpha = jnp.asarray(exact_exchange_fraction, dtype=vxc_matrix.dtype)
    alpha = jnp.clip(jnp.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    return XCEnergyPotentialResult(
        xc_energy=jnp.asarray(xc_energy, dtype=vxc_matrix.dtype),
        vxc_matrix=vxc_matrix,
        exact_exchange_fraction=alpha,
        extra_fock_matrix=extra_fock,
        aux=aux,
    )
