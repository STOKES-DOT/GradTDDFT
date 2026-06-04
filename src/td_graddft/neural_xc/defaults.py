from __future__ import annotations

from typing import Literal, Sequence

from ..xc_backend.jax_libxc import b3lyp_component_basis, b3lyp_component_coefficients


DEFAULT_NEURAL_XC_SEMILOCAL_XC = b3lyp_component_basis()
DEFAULT_NEURAL_XC_COEFFICIENT_PRIOR_VALUES = b3lyp_component_coefficients()
DEFAULT_NEURAL_XC_DENSITY_SUPERVISION = "spin_resolved"
DEFAULT_NEURAL_XC_COEFFICIENT_PRIOR_MODE = "mean"
DEFAULT_NEURAL_XC_HF_INPUT_MODE = "spin_resolved"
DEFAULT_NEURAL_XC_HF_CHANNEL_MODE = "auto"
DEFAULT_NEURAL_XC_RESPONSE_HF_MODE = "strict"
DEFAULT_NEURAL_XC_RESPONSE_PT2_MODE = "approx"

DEFAULT_INPUT_FEATURE_MODE: Literal["enhanced", "canonical"] = "canonical"
DEFAULT_NETWORK_ARCHITECTURE = "graddft_residual"
DEFAULT_NETWORK_HIDDEN_DIMS: tuple[int, ...] = (
    192,
    192,
    192,
    192,
)

def _normalize_semilocal_xc(semilocal_xc: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(semilocal_xc, str):
        return (semilocal_xc,)
    return tuple(str(name) for name in semilocal_xc)


def resolve_coefficient_prior_values(
    semilocal_xc: str | Sequence[str],
    explicit_values: Sequence[float] | None = None,
) -> tuple[float, ...] | None:
    """Resolve default coefficient priors for the default Neural XC basis."""

    if explicit_values is not None:
        return tuple(float(value) for value in explicit_values)

    if _normalize_semilocal_xc(semilocal_xc) == _normalize_semilocal_xc(
        DEFAULT_NEURAL_XC_SEMILOCAL_XC
    ):
        return DEFAULT_NEURAL_XC_COEFFICIENT_PRIOR_VALUES
    return None


__all__ = [
    "DEFAULT_INPUT_FEATURE_MODE",
    "DEFAULT_NETWORK_ARCHITECTURE",
    "DEFAULT_NETWORK_HIDDEN_DIMS",
    "DEFAULT_NEURAL_XC_COEFFICIENT_PRIOR_MODE",
    "DEFAULT_NEURAL_XC_COEFFICIENT_PRIOR_VALUES",
    "DEFAULT_NEURAL_XC_DENSITY_SUPERVISION",
    "DEFAULT_NEURAL_XC_HF_CHANNEL_MODE",
    "DEFAULT_NEURAL_XC_HF_INPUT_MODE",
    "DEFAULT_NEURAL_XC_RESPONSE_HF_MODE",
    "DEFAULT_NEURAL_XC_RESPONSE_PT2_MODE",
    "DEFAULT_NEURAL_XC_SEMILOCAL_XC",
    "resolve_coefficient_prior_values",
]
