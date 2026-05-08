"""Density-fitting helpers inspired by PySCF's df namespace."""

from .jk import (
    build_j_from_df,
    build_jk_from_df,
    build_jk_from_df_orbitals,
    df_factors_to_mo_eri_slices,
    eri_pair_matrix_to_df_factors,
    eri_pair_matrix_to_df_factors_traceable,
    eri_to_df_factors,
    eri_to_df_factors_from_basis,
    true_df_factors_from_pyscf_mol,
)

__all__ = [
    "build_j_from_df",
    "build_jk_from_df",
    "build_jk_from_df_orbitals",
    "df_factors_to_mo_eri_slices",
    "eri_pair_matrix_to_df_factors",
    "eri_pair_matrix_to_df_factors_traceable",
    "eri_to_df_factors",
    "eri_to_df_factors_from_basis",
    "true_df_factors_from_pyscf_mol",
]
