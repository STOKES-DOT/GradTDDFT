"""Pure-JAX Gaussian integral engines (cartesian s/p/d/f)."""

from .one_electron import (
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
from .screening import schwarz_bounds
from .two_electron import eri_element, eri_pair_matrix_packed, eri_tensor, eri_tensor_screened, precompile_eri_kernels

__all__ = [
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
    "eri_element",
    "eri_pair_matrix_packed",
    "precompile_eri_kernels",
    "eri_tensor",
    "eri_tensor_screened",
    "schwarz_bounds",
]
