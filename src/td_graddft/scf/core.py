from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array

from ..data.molecule import MoleculeSpec


def _contains_jax_tracer(value: Any) -> bool:
    if isinstance(value, jax.core.Tracer):
        return True
    if isinstance(value, MoleculeSpec):
        return _contains_jax_tracer((value.coords_bohr, value.charges))
    if isinstance(value, dict):
        return any(_contains_jax_tracer(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_jax_tracer(item) for item in value)
    leaves = jax.tree_util.tree_leaves(value)
    return any(isinstance(leaf, jax.core.Tracer) for leaf in leaves)


def _host_float_unless_traced(value: Any) -> Any:
    return value if _contains_jax_tracer(value) else float(value)


def _orthogonalizer(overlap: Array, eps: float) -> Array:
    eigvals, eigvecs = jnp.linalg.eigh(overlap)
    clipped = jnp.maximum(eigvals, eps)
    return eigvecs @ jnp.diag(clipped ** -0.5) @ eigvecs.T


def _diagonalize_fock(
    fock: Array,
    x: Array,
    eigenvalue_jitter: float = 0.0,
) -> tuple[Array, Array]:
    f_ortho = x.T @ fock @ x
    f_ortho = 0.5 * (f_ortho + f_ortho.T)
    if eigenvalue_jitter != 0.0:
        shift = jnp.arange(f_ortho.shape[0], dtype=f_ortho.dtype) * eigenvalue_jitter
        f_ortho = f_ortho + jnp.diag(shift)
    mo_energy, coeff_ortho = jnp.linalg.eigh(f_ortho)
    mo_coeff = x @ coeff_ortho
    return mo_energy, mo_coeff


def _build_density_closed_shell(mo_coeff: Array, nocc: int) -> Array:
    occ = mo_coeff[:, :nocc]
    return 2.0 * (occ @ occ.T)


def _build_density_from_occ(mo_coeff: Array, mo_occ: Array) -> Array:
    occ = jnp.asarray(mo_occ, dtype=mo_coeff.dtype)
    return jnp.einsum("pi,i,qi->pq", mo_coeff, occ, mo_coeff, precision=Precision.HIGHEST)
