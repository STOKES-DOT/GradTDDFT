from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array

from ..data.molecule import MoleculeSpec


_EIGH_DEGENERACY_TOL = 1e-6
_EIGH_BROADENING = 1e-10


@jax.custom_vjp
def _safe_symmetric_eigh(matrix: Array) -> tuple[Array, Array]:
    values, vectors = jnp.linalg.eigh(matrix)
    return values, vectors


def _safe_symmetric_eigh_fwd(matrix: Array) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
    values, vectors = _safe_symmetric_eigh(matrix)
    return (values, vectors), (values, vectors)


def _safe_symmetric_eigh_bwd(res: tuple[Array, Array], cotangent: tuple[Array, Array]) -> tuple[Array]:
    values, vectors = res
    grad_values, grad_vectors = cotangent
    value_diff = values[None, :] - values[:, None]
    nondegenerate = jnp.abs(value_diff) >= _EIGH_DEGENERACY_TOL
    regular_gap = jnp.nan_to_num(1.0 / value_diff, nan=0.0, posinf=0.0, neginf=0.0)
    broadened_gap = value_diff / (value_diff * value_diff + _EIGH_BROADENING)
    response = jnp.where(nondegenerate, regular_gap, broadened_gap)
    response = response.at[jnp.diag_indices_from(response)].set(0.0)
    inner = jnp.diag(grad_values) + response * (vectors.T @ grad_vectors)
    grad_matrix = vectors @ inner @ vectors.T
    return (0.5 * (grad_matrix + grad_matrix.T),)


_safe_symmetric_eigh.defvjp(_safe_symmetric_eigh_fwd, _safe_symmetric_eigh_bwd)


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
    eigvals, eigvecs = _safe_symmetric_eigh(overlap)
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
    mo_energy, coeff_ortho = _safe_symmetric_eigh(f_ortho)
    mo_coeff = x @ coeff_ortho
    return mo_energy, mo_coeff


def _build_density_closed_shell(mo_coeff: Array, nocc: int) -> Array:
    occ = mo_coeff[:, :nocc]
    return 2.0 * (occ @ occ.T)


def _build_density_from_occ(mo_coeff: Array, mo_occ: Array) -> Array:
    occ = jnp.asarray(mo_occ, dtype=mo_coeff.dtype)
    return jnp.einsum("pi,i,qi->pq", mo_coeff, occ, mo_coeff, precision=Precision.HIGHEST)
