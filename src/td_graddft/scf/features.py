from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
from jax.lax import Precision

from ..data.grid_ao import evaluate_cartesian_ao, evaluate_cartesian_ao_with_derivatives


def _charge_center(mol: Any) -> jnp.ndarray:
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    return jnp.asarray(jnp.einsum("z,zr->r", charges, coords) / charges.sum())
def _eval_grid_ao(
    mol: Any,
    basis: Any,
    coords: Any,
    *,
    backend: Literal["jax"] = "jax",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    del mol
    backend = str(backend).lower()
    coords_arr = jnp.asarray(coords)
    if backend == "jax":
        return evaluate_cartesian_ao_with_derivatives(basis, coords_arr, deriv=1)
    raise ValueError(
        f"Unsupported grid_ao_backend={backend!r}. Only grid_ao_backend='jax' is supported."
    )


def _eval_grid_ao_laplacian(
    mol: Any,
    basis: Any,
    coords: Any,
    *,
    backend: Literal["jax"] = "jax",
) -> jnp.ndarray:
    del mol
    backend = str(backend).lower()
    coords_arr = jnp.asarray(coords)
    if backend == "jax":
        ao_deriv2 = evaluate_cartesian_ao(basis, coords_arr, deriv=2)
        return ao_deriv2[4]
    raise ValueError(
        f"Unsupported grid_ao_backend={backend!r}. Only grid_ao_backend='jax' is supported."
    )


def _restricted_response_eri_slices_from_mo_tensor(
    rep_tensor: Any,
    mo_coeff: Any,
    nocc: int,
    *,
    include_oovv: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
    nocc = int(nocc)
    coeff = jnp.asarray(mo_coeff)
    orbo = coeff[:, :nocc]
    orbv = coeff[:, nocc:]
    rep = jnp.asarray(rep_tensor)
    eri_ovov = jnp.einsum(
        "pqrs,pi,qa,rj,sb->iajb",
        rep,
        orbo,
        orbv,
        orbo,
        orbv,
        precision=Precision.HIGHEST,
    )
    eri_ovvo = jnp.einsum(
        "pqrs,pi,qa,rb,sj->iabj",
        rep,
        orbo,
        orbv,
        orbv,
        orbo,
        precision=Precision.HIGHEST,
    )
    if not include_oovv:
        return eri_ovov, eri_ovvo, None
    eri_oovv = jnp.einsum(
        "pqrs,pi,qj,ra,sb->ijab",
        rep,
        orbo,
        orbo,
        orbv,
        orbv,
        precision=Precision.HIGHEST,
    )
    return eri_ovov, eri_ovvo, eri_oovv


__all__ = [
    "_charge_center",
    "_eval_grid_ao",
    "_eval_grid_ao_laplacian",
    "_restricted_response_eri_slices_from_mo_tensor",
]
