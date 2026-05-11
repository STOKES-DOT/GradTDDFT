from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array


def _as_array(value: Any) -> Array:
    return jnp.asarray(value)


def _as_optional_array(value: Any | None) -> Array | None:
    if value is None:
        return None
    return jnp.asarray(value)


@dataclass(frozen=True)
class RSHFunctionalTemplate:
    """Metadata for a range-separated hybrid XC family."""

    name: str
    local_backend: str
    exchange_backend_id: str
    correlation_backend_id: str
    supports_trainable_sr_hf: bool = True
    supports_trainable_lr_hf: bool = True
    supports_trainable_omega: bool = True
    has_dispersion: bool = False
    monotonic_lr_hf: bool = True
    default_sr_hf_fraction: float = 0.0
    default_lr_hf_fraction: float = 1.0
    default_omega: float = 0.30
    omega_bounds: tuple[float, float] = (0.05, 0.80)
    sr_hf_bounds: tuple[float, float] = (0.0, 1.0)
    lr_hf_bounds: tuple[float, float] = (0.0, 1.0)


@dataclass(frozen=True)
class RSHParameterBounds:
    sr_hf_bounds: tuple[float, float] = (0.0, 1.0)
    lr_hf_bounds: tuple[float, float] = (0.0, 1.0)
    omega_bounds: tuple[float, float] = (0.05, 0.80)


@dataclass(frozen=True)
class ResolvedRSHParameters:
    """Canonical internal RSH parameters.

    Internally we prefer `(sr_hf_fraction, lr_hf_fraction, omega)` because it
    maps cleanly to both the paper notation and the conventional
    `(omega, long_range_hf, short_minus_long_hf)` range-separation tuple
    without overloading the meaning of `alpha/beta`.
    """

    sr_hf_fraction: Array
    lr_hf_fraction: Array
    omega: Array

    def __post_init__(self) -> None:
        object.__setattr__(self, "sr_hf_fraction", _as_array(self.sr_hf_fraction))
        object.__setattr__(self, "lr_hf_fraction", _as_array(self.lr_hf_fraction))
        object.__setattr__(self, "omega", _as_array(self.omega))

    @property
    def paper_alpha(self) -> Array:
        """Short-range HF fraction in the manuscript convention."""

        return self.sr_hf_fraction

    @property
    def paper_beta(self) -> Array:
        """Increment from short-range to long-range HF in the manuscript convention."""

        return self.lr_hf_fraction - self.sr_hf_fraction

    @property
    def exact_exchange_fraction(self) -> Array:
        """Compatibility view for legacy global-hybrid-style code paths."""

        return self.sr_hf_fraction

    def to_range_separated_coefficients(self) -> tuple[Array, Array, Array]:
        """Return `(omega, long_range_hf, short_minus_long_hf)` coefficients."""

        return (
            self.omega,
            self.lr_hf_fraction,
            self.sr_hf_fraction - self.lr_hf_fraction,
        )

    def to_range_separated_hybrid_coefficients(self) -> tuple[Array, Array, Array]:
        """Return `(omega, long_range_hf, short_range_hf)` coefficients."""

        return (
            self.omega,
            self.lr_hf_fraction,
            self.sr_hf_fraction,
        )


@dataclass(frozen=True)
class SCFXCContributions:
    """Unified SCF-facing XC contribution bundle."""

    v_rho: Array
    v_grad: Array
    xc_kind: str
    full_hf_fraction: Array
    lr_hf_omegas: Array | None = None
    lr_hf_coefficients: Array | None = None
    extra_fock_matrix: Array | None = None
    exact_exchange_fraction: Array | None = None
    resolved_xc: Any | None = None

    def __post_init__(self) -> None:
        v_rho = _as_array(self.v_rho)
        v_grad = _as_array(self.v_grad)
        full_hf_fraction = _as_array(self.full_hf_fraction)
        lr_hf_omegas = _as_optional_array(self.lr_hf_omegas)
        lr_hf_coefficients = _as_optional_array(self.lr_hf_coefficients)
        extra_fock_matrix = _as_optional_array(self.extra_fock_matrix)
        exact_exchange_fraction = _as_optional_array(self.exact_exchange_fraction)
        if exact_exchange_fraction is None:
            exact_exchange_fraction = full_hf_fraction

        if (lr_hf_omegas is None) != (lr_hf_coefficients is None):
            raise ValueError(
                "lr_hf_omegas and lr_hf_coefficients must both be provided or both be None."
            )
        if lr_hf_omegas is not None and lr_hf_coefficients is not None:
            lr_hf_omegas = jnp.reshape(lr_hf_omegas, (-1,))
            lr_hf_coefficients = jnp.reshape(lr_hf_coefficients, (-1,))
            if lr_hf_omegas.shape != lr_hf_coefficients.shape:
                raise ValueError(
                    "lr_hf_omegas and lr_hf_coefficients must have the same shape "
                    f"(got {lr_hf_omegas.shape} vs {lr_hf_coefficients.shape})."
                )

        object.__setattr__(self, "v_rho", v_rho)
        object.__setattr__(self, "v_grad", v_grad)
        object.__setattr__(self, "full_hf_fraction", full_hf_fraction)
        object.__setattr__(self, "lr_hf_omegas", lr_hf_omegas)
        object.__setattr__(self, "lr_hf_coefficients", lr_hf_coefficients)
        object.__setattr__(self, "extra_fock_matrix", extra_fock_matrix)
        object.__setattr__(self, "exact_exchange_fraction", exact_exchange_fraction)

__all__ = [
    "RSHFunctionalTemplate",
    "RSHParameterBounds",
    "ResolvedRSHParameters",
    "SCFXCContributions",
]
