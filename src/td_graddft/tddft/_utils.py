from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array



def _symmetrize(matrix: Array) -> Array:
    return 0.5 * (matrix + matrix.T.conj())


def _matrix_power_symmetric(matrix: Array, power: float, eps: float) -> Array:
    eigvals, eigvecs = jnp.linalg.eigh(_symmetrize(matrix))
    clipped = jnp.maximum(eigvals, eps)
    return (eigvecs * (clipped**power)) @ eigvecs.T.conj()


def _casida_metric_factor(matrix: Array, eps: float) -> Array:
    """Lower-triangular factor L with L L^T ~= matrix for Casida transforms."""

    sym = _symmetrize(matrix)
    eye = jnp.eye(sym.shape[0], dtype=sym.dtype)
    return jnp.linalg.cholesky(sym + eps * eye)


def _restricted_channel(molecule: Any) -> tuple[Array, Array, Array]:
    mo_coeff = jnp.asarray(molecule.mo_coeff)
    mo_occ = jnp.asarray(molecule.mo_occ)
    mo_energy = jnp.asarray(molecule.mo_energy)

    if mo_coeff.ndim == 2:
        return mo_coeff, mo_occ, mo_energy
    if mo_coeff.ndim != 3:
        raise ValueError(
            "Expected mo_coeff to have shape (nao, nmo) or (spin, nao, nmo)."
        )
    if mo_coeff.shape[0] == 1:
        return mo_coeff[0], mo_occ[0], mo_energy[0]
    if mo_coeff.shape[0] != 2:
        raise NotImplementedError("Only restricted closed-shell references are supported.")
    # Avoid Python boolean conversions on traced arrays under JIT by assuming
    # restricted callers provide spin-identical channels when shape is (2,...).
    return mo_coeff[0], mo_occ[0], mo_energy[0]


def _density_on_grid(molecule: Any) -> Array:
    if hasattr(molecule, "density"):
        density = molecule.density()
    else:
        density = jnp.einsum("spq,rp,rq->rs", molecule.rdm1, molecule.ao, molecule.ao)
    density = jnp.asarray(density)
    if density.ndim == 1:
        return density
    return density.sum(axis=-1)


def _restricted_orbital_data(
    molecule: Any,
    occupation_tolerance: float,
) -> tuple[Array, Array, Array, Array]:
    mo_coeff, mo_occ, mo_energy = _restricted_channel(molecule)

    nocc = getattr(molecule, "nocc", None)
    if nocc is None:
        # Fallback for ad-hoc molecule-like inputs that do not expose nocc.
        # Keep this outside traced/JIT paths by requiring a Python int.
        nocc = int(jnp.count_nonzero(mo_occ > occupation_tolerance))
    else:
        nocc = int(nocc)

    nmo = int(mo_coeff.shape[1])
    if nocc <= 0 or nocc >= nmo:
        raise ValueError("Need at least one occupied and one virtual orbital.")

    orbo = mo_coeff[:, :nocc]
    orbv = mo_coeff[:, nocc:]
    delta_eps = mo_energy[nocc:][None, :] - mo_energy[:nocc][:, None]
    return orbo, orbv, delta_eps, mo_coeff


def _transition_densities_on_grid(ao: Array, orbo: Array, orbv: Array) -> Array:
    rho_o = jnp.einsum("rp,pi->ri", ao, orbo, precision=Precision.HIGHEST)
    rho_v = jnp.einsum("rp,pa->ra", ao, orbv, precision=Precision.HIGHEST)
    return jnp.einsum("ri,ra->ria", rho_o, rho_v, precision=Precision.HIGHEST)


def _resolve_xc_functional(
    molecule: Any,
    xc_functional: Any,
    xc_params: Any | None,
) -> Any | None:
    if xc_functional is None:
        return None
    if xc_params is None:
        return xc_functional
    response_molecule_binder = getattr(xc_functional, "bind_to_molecule_for_response", None)
    if response_molecule_binder is not None:
        return response_molecule_binder(xc_params, molecule)
    molecule_binder = getattr(xc_functional, "bind_to_molecule", None)
    if molecule_binder is not None:
        return molecule_binder(xc_params, molecule)
    binder = getattr(xc_functional, "bind", None)
    if binder is None:
        raise TypeError(
            "xc_params were provided, but the XC functional does not implement bind(params)."
        )
    return binder(xc_params)
