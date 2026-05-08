from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp
from jaxtyping import Array


def _as_array(value: Any) -> Array:
    return jnp.asarray(value)


def _as_optional_array(value: Any | None) -> Array | None:
    if value is None:
        return None
    return jnp.asarray(value)


def _make_pyscf_eval_xc_callable(xc_description: str) -> Callable[..., Any]:
    from pyscf.dft.libxc import eval_xc

    def _eval_xc(xc_code: Any, rho: Any, *args: Any, **kwargs: Any) -> Any:
        del xc_code
        return eval_xc(xc_description, rho, *args, **kwargs)

    return _eval_xc


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
    maps cleanly to both the paper notation and the PySCF `rsh=(omega, alpha, beta)`
    convention without overloading the meaning of `alpha/beta`.
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

    def to_pyscf_rsh(self) -> tuple[Array, Array, Array]:
        """Return `PySCF`'s `(omega, alpha, beta)` tuple.

        PySCF interprets:
        - `alpha` as the long-range HF fraction
        - `beta` as the short-range minus long-range HF increment
        """

        return (
            self.omega,
            self.lr_hf_fraction,
            self.sr_hf_fraction - self.lr_hf_fraction,
        )

    def to_pyscf_rsh_and_hybrid(self) -> tuple[Array, Array, Array]:
        """Return `PySCF numint.rsh_and_hybrid_coeff` semantics."""

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


@dataclass(frozen=True)
class PySCFRSHSpec:
    """Portable PySCF-installable XC spec."""

    xc_description: str | Callable[..., Any]
    xctype: str
    hyb: float
    rsh: tuple[float, float, float]

    def expected_rsh_and_hybrid_coeff(self) -> tuple[float, float, float]:
        omega, alpha, _beta = self.rsh
        return (float(omega), float(alpha), float(self.hyb))

    def install_into_numint(self, ni: Any) -> Any:
        from pyscf.dft.libxc import define_xc_

        description = self.xc_description
        if isinstance(description, str):
            # PySCF ignores explicit hyb/rsh when `description` is a raw string.
            # Wrap the local XC description as a callable so the supplied
            # range-separation coefficients remain authoritative.
            description = _make_pyscf_eval_xc_callable(description)

        return define_xc_(
            ni,
            description,
            xctype=str(self.xctype),
            hyb=float(self.hyb),
            rsh=tuple(float(value) for value in self.rsh),
        )

    def install_into_mf(self, mf: Any) -> Any:
        numint = getattr(mf, "_numint", None)
        if numint is None:
            raise AttributeError("PySCF mean-field object must define _numint.")
        mf._numint = self.install_into_numint(numint)
        return mf


def make_pyscf_rsh_spec(
    *,
    xc_description: str | Callable[..., Any],
    xctype: str,
    resolved_params: ResolvedRSHParameters,
) -> PySCFRSHSpec:
    omega, alpha, beta = resolved_params.to_pyscf_rsh()
    return PySCFRSHSpec(
        xc_description=xc_description,
        xctype=str(xctype),
        hyb=float(jnp.asarray(resolved_params.sr_hf_fraction)),
        rsh=(
            float(jnp.asarray(omega)),
            float(jnp.asarray(alpha)),
            float(jnp.asarray(beta)),
        ),
    )


__all__ = [
    "PySCFRSHSpec",
    "RSHFunctionalTemplate",
    "RSHParameterBounds",
    "ResolvedRSHParameters",
    "SCFXCContributions",
    "make_pyscf_rsh_spec",
]
