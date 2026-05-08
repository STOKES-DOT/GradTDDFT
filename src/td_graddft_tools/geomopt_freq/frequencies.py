from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .objectives import EnergySurface

AMU_TO_ELECTRON_MASS = 1822.888486209
HARTREE_TO_WAVENUMBER_CM1 = 219474.6313705


@dataclass(frozen=True)
class FrequencyAnalysisConfig:
    """Settings for harmonic frequency analysis from Cartesian Hessians."""

    remove_trans_rot: bool = True
    linear_molecule: bool = False
    imaginary_mode_epsilon: float = 1e-16


@dataclass(frozen=True)
class FrequencyAnalysisResult:
    frequencies_cm1: Array
    hessian_cartesian: Array
    hessian_mass_weighted: Array
    mode_vectors_cartesian: Array
    raw_eigenvalues: Array
    removed_mode_count: int


def _mass_vector_au(masses_amu: Array) -> Array:
    masses = jnp.asarray(masses_amu)
    if masses.ndim != 1:
        raise ValueError("masses_amu must have shape (natom,).")
    return jnp.repeat(masses * AMU_TO_ELECTRON_MASS, repeats=3)


def _symmetric(matrix: Array) -> Array:
    return 0.5 * (matrix + matrix.T)


def _removed_mode_count(natom: int, config: FrequencyAnalysisConfig) -> int:
    ncart = 3 * natom
    if not config.remove_trans_rot:
        return 0
    if natom <= 1:
        return min(3, max(0, ncart - 1))
    target = 5 if config.linear_molecule else 6
    return min(target, max(0, ncart - 1))


def run_frequency_analysis(
    surface: EnergySurface,
    coordinates: Array,
    masses_amu: Array,
    config: FrequencyAnalysisConfig | None = None,
) -> FrequencyAnalysisResult:
    """Compute harmonic frequencies via JAX Hessian on Cartesian coordinates."""

    cfg = FrequencyAnalysisConfig() if config is None else config
    coords = jnp.asarray(coordinates)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coordinates must have shape (natom, 3).")

    natom = int(coords.shape[0])
    flat_coords = coords.reshape(-1)
    mass_vec = _mass_vector_au(masses_amu)
    if mass_vec.shape[0] != flat_coords.shape[0]:
        raise ValueError("masses_amu size must match coordinates natom.")

    def energy_from_flat(flat: Array) -> Array:
        return surface.energy(flat.reshape(natom, 3))

    hessian = jax.hessian(energy_from_flat)(flat_coords)
    hessian = _symmetric(hessian)
    denom = jnp.sqrt(jnp.outer(mass_vec, mass_vec))
    hessian_mw = _symmetric(hessian / denom)

    eigvals, eigvecs = jnp.linalg.eigh(hessian_mw)
    remove_n = _removed_mode_count(natom, cfg)
    order = jnp.argsort(jnp.abs(eigvals))
    keep = order[remove_n:]
    kept_vals = eigvals[keep]
    kept_vecs = eigvecs[:, keep]

    omega = jnp.sign(kept_vals) * jnp.sqrt(jnp.abs(kept_vals) + cfg.imaginary_mode_epsilon)
    frequencies_cm1 = omega * HARTREE_TO_WAVENUMBER_CM1

    # Convert mass-weighted eigenvectors to Cartesian displacement patterns.
    cart_modes = kept_vecs / jnp.sqrt(mass_vec)[:, None]
    mode_norm = jnp.linalg.norm(cart_modes, axis=0)
    cart_modes = cart_modes / jnp.maximum(mode_norm, 1e-20)
    mode_vectors = cart_modes.T.reshape(-1, natom, 3)

    return FrequencyAnalysisResult(
        frequencies_cm1=frequencies_cm1,
        hessian_cartesian=hessian.reshape(natom, 3, natom, 3),
        hessian_mass_weighted=hessian_mw,
        mode_vectors_cartesian=mode_vectors,
        raw_eigenvalues=eigvals,
        removed_mode_count=remove_n,
    )

