"""Restricted TDDFT solvers and response builders."""

from .casida import RestrictedCasidaTDDFT, solve_casida
from .long_range_correction import (
    BoundLongRangeCorrectedFunctional,
    BoundGridPointModeLongRangeCorrectedFunctional,
    GridPointModeCouplingNet,
    GridPointModeLongRangeCorrectedFunctional,
    build_grid_point_mode_basis,
    build_grid_point_mode_features,
    LongRangeCorrectedFunctional,
    LongRangeXCNet,
    build_long_range_pair_features,
    compute_long_range_kernel,
    pairwise_grid_distances,
)
from .response import (
    build_restricted_response_matrices,
    gen_tda_vind,
    gen_tdhf_vind,
)
from .tda import solve_tda
from .types import TDDFTMatrices, TDDFTResult, TDAResult
from .unrestricted import (
    UnrestrictedCasidaTDDFT,
    UnrestrictedResponseMatrices,
    UnrestrictedTDA,
    UnrestrictedTDAMatrices,
    UnrestrictedTDDFTResult,
    UnrestrictedTDAResult,
    build_unrestricted_response_matrices,
    solve_unrestricted_casida,
    build_unrestricted_tda_matrices,
    solve_unrestricted_tda,
)

__all__ = [
    "TDDFTMatrices",
    "TDAResult",
    "TDDFTResult",
    "build_restricted_response_matrices",
    "gen_tda_vind",
    "gen_tdhf_vind",
    "solve_tda",
    "solve_casida",
    "RestrictedCasidaTDDFT",
    "LongRangeXCNet",
    "LongRangeCorrectedFunctional",
    "BoundLongRangeCorrectedFunctional",
    "GridPointModeCouplingNet",
    "GridPointModeLongRangeCorrectedFunctional",
    "BoundGridPointModeLongRangeCorrectedFunctional",
    "pairwise_grid_distances",
    "build_long_range_pair_features",
    "build_grid_point_mode_features",
    "build_grid_point_mode_basis",
    "compute_long_range_kernel",
    "UnrestrictedResponseMatrices",
    "UnrestrictedTDAMatrices",
    "UnrestrictedTDAResult",
    "UnrestrictedTDDFTResult",
    "build_unrestricted_response_matrices",
    "build_unrestricted_tda_matrices",
    "solve_unrestricted_tda",
    "solve_unrestricted_casida",
    "UnrestrictedTDA",
    "UnrestrictedCasidaTDDFT",
]
