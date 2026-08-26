from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..features import grid_features_with_spin_gradients_for_molecule
from ..xc_backend.jax_libxc import (
    eval_xc_energy_density_unrestricted_from_density_gradients,
    hybrid_coeff,
    parse_xc,
    xc_type,
)
from .response import (
    _project_grid_response_to_restricted_transition,
    _project_restricted_transition_to_grid,
    _restricted_response_factors,
)


SpinGridHVP = Callable[[Any, Array, Array], tuple[Array, Array]]


@lru_cache(maxsize=64)
def _point_spin_hvp(spec: str, feature_kind: str):
    kind = str(feature_kind).upper()

    def point_energy(values: Array) -> Array:
        zero = jnp.zeros((3,), dtype=values.dtype)
        grad_a = zero if kind == "LDA" else values[2:5]
        grad_b = zero if kind == "LDA" else values[5:8]
        return eval_xc_energy_density_unrestricted_from_density_gradients(
            spec,
            values[0],
            values[1],
            grad_a,
            grad_b,
        )

    gradient = jax.grad(point_energy)

    def point_hvp(values: Array, tangent: Array) -> Array:
        return jax.jvp(gradient, (values,), (tangent,))[1]

    return jax.jit(jax.vmap(jax.vmap(point_hvp)))


def _validate_spin_tangent(
    tangent: Array,
    *,
    nfeatures: int,
    ngrids: int,
    label: str,
) -> Array:
    array = jnp.asarray(tangent)
    expected_tail = (int(nfeatures), int(ngrids))
    if array.ndim != 3 or array.shape[1:] != expected_tail:
        raise ValueError(
            f"{label} must have shape (batch, {nfeatures}, {ngrids}), "
            f"got {array.shape}."
        )
    return array


def build_spin_transition_factors(
    molecule: Any,
    orbo: Array,
    orbv: Array,
    *,
    feature_kind: str,
    dtype: Any,
):
    return _restricted_response_factors(
        molecule,
        orbo,
        orbv,
        feature_kind=str(feature_kind).upper(),
        dtype=dtype,
    )


def project_spin_transition_to_grid(factors: Any, values: Array) -> Array:
    return _project_restricted_transition_to_grid(factors, values)


def project_grid_response_to_spin_transition(factors: Any, values: Array) -> Array:
    return _project_grid_response_to_restricted_transition(factors, values)


def build_unrestricted_semilocal_response_action(
    molecule: Any,
    orbo_a: Array,
    orbv_a: Array,
    orbo_b: Array,
    orbv_b: Array,
    response_hvp: SpinGridHVP,
    *,
    feature_kind: str,
    dtype: Any,
) -> Callable[[Array, Array], tuple[Array, Array]]:
    kind = str(feature_kind).upper()
    if kind not in {"LDA", "GGA"}:
        raise NotImplementedError(
            "Unrestricted semilocal response projection supports LDA/GGA only."
        )
    factors_a = build_spin_transition_factors(
        molecule,
        orbo_a,
        orbv_a,
        feature_kind=kind,
        dtype=dtype,
    )
    factors_b = build_spin_transition_factors(
        molecule,
        orbo_b,
        orbv_b,
        feature_kind=kind,
        dtype=dtype,
    )
    weights = jnp.asarray(molecule.grid.weights, dtype=dtype)

    def action(alpha: Array, beta: Array) -> tuple[Array, Array]:
        tangent_a = project_spin_transition_to_grid(factors_a, alpha)
        tangent_b = project_spin_transition_to_grid(factors_b, beta)
        response_a, response_b = response_hvp(molecule, tangent_a, tangent_b)
        weighted_a = jnp.asarray(response_a, dtype=dtype) * weights[None, None, :]
        weighted_b = jnp.asarray(response_b, dtype=dtype) * weights[None, None, :]
        return (
            project_grid_response_to_spin_transition(factors_a, weighted_a),
            project_grid_response_to_spin_transition(factors_b, weighted_b),
        )

    return action


@dataclass(frozen=True)
class UnrestrictedSemilocalResponseFunctional:
    """Traditional spin-polarized LDA/GGA response represented as grid HVPs."""

    xc_spec: str

    def __post_init__(self) -> None:
        spec = str(self.xc_spec).lower()
        parse_xc(spec)
        kind = str(xc_type(spec)).upper()
        if kind not in {"LDA", "GGA"}:
            raise NotImplementedError(
                "Unrestricted semilocal response supports LDA/GGA only."
            )
        object.__setattr__(self, "xc_spec", spec)
        object.__setattr__(self, "exact_exchange_fraction", float(hybrid_coeff(spec)))
        object.__setattr__(self, "response_feature_kind", kind)

    def spin_grid_response_hvp(
        self,
        molecule: Any,
        tangent_a: Array,
        tangent_b: Array,
    ) -> tuple[Array, Array]:
        features, grad_a, grad_b = grid_features_with_spin_gradients_for_molecule(
            molecule
        )
        ngrids = int(features.rho.shape[0])
        nfeatures = 1 if self.response_feature_kind == "LDA" else 4
        tangent_a = _validate_spin_tangent(
            tangent_a,
            nfeatures=nfeatures,
            ngrids=ngrids,
            label="tangent_a",
        )
        tangent_b = _validate_spin_tangent(
            tangent_b,
            nfeatures=nfeatures,
            ngrids=ngrids,
            label="tangent_b",
        )
        if tangent_a.shape[0] != tangent_b.shape[0]:
            raise ValueError("Alpha and beta spin-grid tangents must share a batch size.")

        if self.response_feature_kind == "LDA":
            base = jnp.stack([features.rho_a, features.rho_b], axis=-1)
            tangent = jnp.stack([tangent_a[:, 0], tangent_b[:, 0]], axis=-1)
        else:
            base = jnp.concatenate(
                [
                    features.rho_a[:, None],
                    features.rho_b[:, None],
                    grad_a,
                    grad_b,
                ],
                axis=-1,
            )
            tangent = jnp.concatenate(
                [
                    tangent_a[:, 0:1],
                    tangent_b[:, 0:1],
                    tangent_a[:, 1:4],
                    tangent_b[:, 1:4],
                ],
                axis=1,
            ).transpose(0, 2, 1)

        base_batch = jnp.broadcast_to(base, (tangent.shape[0],) + base.shape)
        response = _point_spin_hvp(
            self.xc_spec,
            self.response_feature_kind,
        )(base_batch, tangent).transpose(0, 2, 1)
        if self.response_feature_kind == "LDA":
            return response[:, 0:1], response[:, 1:2]
        return (
            jnp.concatenate([response[:, 0:1], response[:, 2:5]], axis=1),
            jnp.concatenate([response[:, 1:2], response[:, 5:8]], axis=1),
        )


__all__ = [
    "SpinGridHVP",
    "UnrestrictedSemilocalResponseFunctional",
    "build_spin_transition_factors",
    "build_unrestricted_semilocal_response_action",
    "project_grid_response_to_spin_transition",
    "project_spin_transition_to_grid",
]
