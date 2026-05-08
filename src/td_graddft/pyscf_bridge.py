"""Backward-compatible shim for legacy PySCF bridge imports.

This module is deprecated. New code should prefer:

- ``td_graddft.api`` for simplified strict-JAX runtime entry points
- ``td_graddft.reference`` for strict-JAX low-level reference builders
- ``td_graddft.reference_legacy`` for explicit PySCF-backed compatibility builders
"""

from __future__ import annotations

import warnings

warnings.warn(
    "td_graddft.pyscf_bridge is deprecated. Use td_graddft.api, td_graddft.reference, or td_graddft.reference_legacy instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .reference import (
    GridReference,
    RestrictedMoleculeReference,
    UnrestrictedMoleculeReference,
    restricted_reference_from_spec_with_jax_rks,
)
from .reference_legacy import (
    restricted_reference_from_pyscf,
    restricted_reference_from_pyscf_spec_with_jax_rks,
    restricted_reference_from_pyscf_with_jax_rhf,
    restricted_reference_from_pyscf_with_jax_rks,
    unrestricted_reference_from_pyscf,
    unrestricted_reference_from_pyscf_with_jax_uks,
)

__all__ = [
    "GridReference",
    "RestrictedMoleculeReference",
    "UnrestrictedMoleculeReference",
    "restricted_reference_from_pyscf",
    "unrestricted_reference_from_pyscf",
    "restricted_reference_from_pyscf_with_jax_rhf",
    "restricted_reference_from_pyscf_with_jax_rks",
    "restricted_reference_from_spec_with_jax_rks",
    "restricted_reference_from_pyscf_spec_with_jax_rks",
    "unrestricted_reference_from_pyscf_with_jax_uks",
]
