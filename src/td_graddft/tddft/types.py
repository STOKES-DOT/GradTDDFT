from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import jax
from jaxtyping import Array


def _pytree_dataclass(cls):
    def tree_flatten(self):
        children = tuple(getattr(self, field.name) for field in fields(self))
        return children, None

    @classmethod
    def tree_unflatten(cls_, aux_data, children):
        del aux_data
        return cls_(*children)

    cls.tree_flatten = tree_flatten
    cls.tree_unflatten = tree_unflatten
    return jax.tree_util.register_pytree_node_class(cls)


@_pytree_dataclass
@dataclass(frozen=True)
class TDDFTMatrices:
    """Response matrices for a restricted closed-shell reference."""

    orbital_energy_differences: Array
    a_matrix: Array
    b_matrix: Array


@_pytree_dataclass
@dataclass(frozen=True)
class TDAResult:
    """Excitation energies and amplitudes from TDA."""

    excitation_energies: Array
    amplitudes: Array
    a_matrix: Optional[Array]
    posthoc_correction: Optional[Array] = None


@_pytree_dataclass
@dataclass(frozen=True)
class TDDFTResult:
    """Excitation energies and (X, Y) amplitudes from Casida TDDFT."""

    excitation_energies: Array
    x_amplitudes: Array
    y_amplitudes: Array
    a_matrix: Optional[Array]
    b_matrix: Optional[Array]
    casida_matrix: Optional[Array]
    posthoc_correction: Optional[Array] = None
