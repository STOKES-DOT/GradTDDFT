from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array


def build_restricted_lowrank_mo_response_action(
    j_factors: Array,
    k_factors: Array | None,
    orbo: Array,
    orbv: Array,
    hybrid_fraction: Any,
    *,
    include_exchange: bool,
    dtype: Any,
) -> Callable[..., Array]:
    """Build a restricted low-rank two-electron response action.

    The same contraction covers full DF and RIS.  DF passes the same factor tensor
    for J and K, while RIS can pass separate minimal auxiliary factors for each
    term.
    """

    alpha = jnp.asarray(hybrid_fraction, dtype=dtype)
    j_source = jnp.asarray(j_factors, dtype=dtype)
    orbo = jnp.asarray(orbo, dtype=dtype)
    orbv = jnp.asarray(orbv, dtype=dtype)
    j_ov = jnp.einsum("Qpq,pi,qa->Qia", j_source, orbo, orbv, precision=Precision.HIGHEST)
    j_vo = jnp.einsum("Qpq,pa,qi->Qai", j_source, orbv, orbo, precision=Precision.HIGHEST)

    if include_exchange:
        if k_factors is None:
            raise ValueError("K low-rank factors are required for hybrid response.")
        k_source = jnp.asarray(k_factors, dtype=dtype)
        k_ov = jnp.einsum("Qpq,pi,qa->Qia", k_source, orbo, orbv, precision=Precision.HIGHEST)
        k_vo = jnp.einsum("Qpq,pa,qi->Qai", k_source, orbv, orbo, precision=Precision.HIGHEST)
        k_oo = jnp.einsum("Qpq,pi,qj->Qij", k_source, orbo, orbo, precision=Precision.HIGHEST)
        k_vv = jnp.einsum("Qpq,pa,qb->Qab", k_source, orbv, orbv, precision=Precision.HIGHEST)
    else:
        k_ov = None
        k_vo = None
        k_oo = None
        k_vv = None

    nocc = int(orbo.shape[1])
    nvir = int(orbv.shape[1])

    def action(values: Array, *, bottom_density: bool, bottom_projection: bool) -> Array:
        original_shape = jnp.asarray(values).shape
        x = jnp.asarray(values, dtype=dtype).reshape(-1, nocc, nvir)
        density_factor = j_ov if bottom_density else j_vo
        if bottom_density:
            rho_aux = 2.0 * jnp.einsum(
                "Qia,nia->nQ",
                density_factor,
                x,
                precision=Precision.HIGHEST,
            )
        else:
            rho_aux = 2.0 * jnp.einsum(
                "Qai,nia->nQ",
                density_factor,
                x,
                precision=Precision.HIGHEST,
            )
        if bottom_projection:
            out = jnp.einsum("Qia,nQ->nia", j_ov, rho_aux, precision=Precision.HIGHEST)
        else:
            out = jnp.einsum("Qai,nQ->nia", j_vo, rho_aux, precision=Precision.HIGHEST)

        if include_exchange:
            if bottom_density == bottom_projection:
                assert k_oo is not None and k_vv is not None
                k_out = 2.0 * jnp.einsum(
                    "Qij,Qab,njb->nia",
                    k_oo,
                    k_vv,
                    x,
                    precision=Precision.HIGHEST,
                )
            else:
                assert k_ov is not None and k_vo is not None
                k_out = 2.0 * jnp.einsum(
                    "Qaj,Qib,njb->nia",
                    k_vo,
                    k_ov,
                    x,
                    precision=Precision.HIGHEST,
                )
            out = out - 0.5 * alpha * k_out
        return out.reshape(original_shape)

    return action
