"""Pure-JAX Gaussian integral engines (cartesian s/p/d/f)."""

from .jax.direct_jk import DirectJKResult, build_direct_jk_from_basis, build_direct_jk_incremental
from .jax.one_electron import (
    build_hcore,
    dipole_element,
    dipole_matrix,
    kinetic_element,
    kinetic_matrix,
    nuclear_attraction_element,
    nuclear_attraction_matrix,
    overlap_hcore_matrices,
    overlap_element,
    overlap_matrix,
    rinv_element,
    rinv_matrices,
    rinv_matrix,
)
from .jax.packed_eri import build_j_from_eri_pair_matrix, build_jk_from_eri_pair_matrix, eri_pair_matrix_to_mo_eri_slices
from .jax.screening import schwarz_bounds, shell_pair_schwarz_bounds
from .jax.two_electron import eri_element, eri_pair_matrix_packed, eri_tensor, eri_tensor_screened, precompile_eri_kernels
from .libcint.mol import build_libcint_mol, libcint_intor_name

__all__ = [
    "build_libcint_mol",
    "build_direct_jk_from_basis",
    "build_direct_jk_incremental",
    "overlap_matrix",
    "overlap_hcore_matrices",
    "overlap_element",
    "kinetic_matrix",
    "kinetic_element",
    "dipole_matrix",
    "dipole_element",
    "rinv_matrix",
    "rinv_matrices",
    "rinv_element",
    "nuclear_attraction_matrix",
    "nuclear_attraction_element",
    "build_hcore",
    "build_j_from_eri_pair_matrix",
    "build_jk_from_eri_pair_matrix",
    "DirectJKResult",
    "eri_element",
    "eri_pair_matrix_packed",
    "eri_pair_matrix_to_mo_eri_slices",
    "precompile_eri_kernels",
    "eri_tensor",
    "eri_tensor_screened",
    "schwarz_bounds",
    "shell_pair_schwarz_bounds",
    "libcint_intor_name",
]
