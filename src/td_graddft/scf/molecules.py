from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import jax
import jax.numpy as jnp


def _pytree_dataclass(*, static_fields: tuple[str, ...] = ()):
    static_field_names = frozenset(static_fields)

    def decorator(cls):
        all_field_names = tuple(field.name for field in fields(cls))
        dynamic_field_names = tuple(
            field_name for field_name in all_field_names if field_name not in static_field_names
        )
        static_field_names_ordered = tuple(
            field_name for field_name in all_field_names if field_name in static_field_names
        )

        def tree_flatten(self):
            children = tuple(getattr(self, field_name) for field_name in dynamic_field_names)
            static_values = tuple(getattr(self, field_name) for field_name in static_field_names_ordered)
            return children, static_values

        @classmethod
        def tree_unflatten(cls_, aux_data, children):
            kwargs = {
                name: value for name, value in zip(dynamic_field_names, children, strict=True)
            }
            kwargs.update(
                {
                    name: value
                    for name, value in zip(static_field_names_ordered, aux_data, strict=True)
                }
            )
            return cls_(**kwargs)

        cls.tree_flatten = tree_flatten
        cls.tree_unflatten = tree_unflatten
        return jax.tree_util.register_pytree_node_class(cls)

    return decorator


@_pytree_dataclass()
@dataclass(frozen=True)
class QuadratureGrid:
    """Minimal quadrature grid container used by the TDDFT modules."""

    weights: jnp.ndarray
    coords: jnp.ndarray | None = None


@_pytree_dataclass(
    static_fields=(
        "nocc",
        "scf_converged",
        "direct_jk_engine",
        "direct_scf_tol",
        "direct_basis",
        "direct_cuda_jk_builder",
    )
)
@dataclass(frozen=True)
class RestrictedMolecule:
    """Minimal restricted molecule container used across TD-GradDFT."""

    ao: jnp.ndarray
    grid: QuadratureGrid
    dipole_integrals: jnp.ndarray
    rep_tensor: jnp.ndarray
    mo_coeff: jnp.ndarray
    mo_occ: jnp.ndarray
    mo_energy: jnp.ndarray
    rdm1: jnp.ndarray
    h1e: jnp.ndarray
    nuclear_repulsion: float
    atom_coords: jnp.ndarray | None = None
    atom_charges: jnp.ndarray | None = None
    overlap_matrix: jnp.ndarray | None = None
    ao_deriv1: jnp.ndarray | None = None
    ao_laplacian: jnp.ndarray | None = None
    mf_energy: float | None = None
    exact_exchange_fraction: float = 0.0
    nocc: int | None = None
    hfx_omega_values: jnp.ndarray | None = None
    hfx_local: jnp.ndarray | None = None
    hfx_nu: jnp.ndarray | None = None
    pt2_local: jnp.ndarray | None = None
    scf_initial_density: jnp.ndarray | None = None
    df_factors: jnp.ndarray | None = None
    eri_pair_matrix: jnp.ndarray | None = None
    eri_ovov: jnp.ndarray | None = None
    eri_ovvo: jnp.ndarray | None = None
    eri_oovv: jnp.ndarray | None = None
    scf_converged: bool | None = None
    direct_jk_engine: str | None = None
    direct_scf_tol: float | None = None
    direct_basis: Any | None = None
    direct_cuda_jk_builder: Any | None = None

    def density(self) -> jnp.ndarray:
        return jnp.einsum("spq,rp,rq->rs", self.rdm1, self.ao, self.ao)


@_pytree_dataclass(static_fields=("nocc_alpha", "nocc_beta"))
@dataclass(frozen=True)
class UnrestrictedMolecule:
    """Minimal unrestricted molecule container used across TD-GradDFT."""

    ao: jnp.ndarray
    grid: QuadratureGrid
    dipole_integrals: jnp.ndarray
    rep_tensor: jnp.ndarray
    mo_coeff: jnp.ndarray
    mo_occ: jnp.ndarray
    mo_energy: jnp.ndarray
    rdm1: jnp.ndarray
    h1e: jnp.ndarray
    nuclear_repulsion: float
    atom_coords: jnp.ndarray | None = None
    atom_charges: jnp.ndarray | None = None
    overlap_matrix: jnp.ndarray | None = None
    ao_deriv1: jnp.ndarray | None = None
    ao_laplacian: jnp.ndarray | None = None
    mf_energy: float | None = None
    exact_exchange_fraction: float = 0.0
    nocc_alpha: int | None = None
    nocc_beta: int | None = None
    hfx_omega_values: jnp.ndarray | None = None
    hfx_local: jnp.ndarray | None = None
    hfx_nu: jnp.ndarray | None = None
    scf_initial_density: jnp.ndarray | None = None

    def density(self) -> jnp.ndarray:
        return jnp.einsum("spq,rp,rq->r", self.rdm1, self.ao, self.ao)


GridReference = QuadratureGrid
RestrictedMoleculeReference = RestrictedMolecule
UnrestrictedMoleculeReference = UnrestrictedMolecule

__all__ = [
    "QuadratureGrid",
    "RestrictedMolecule",
    "UnrestrictedMolecule",
]
