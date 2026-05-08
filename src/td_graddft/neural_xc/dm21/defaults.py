from __future__ import annotations

from typing import Sequence

from ...jax_libxc import b3lyp_component_basis, b3lyp_component_coefficients

DEFAULT_NEURAL_XC_SEMILOCAL_XC = b3lyp_component_basis()
DEFAULT_NEURAL_XC_COEFFICIENT_PRIOR_VALUES = b3lyp_component_coefficients()
DEFAULT_NEURAL_XC_DENSITY_SUPERVISION = "spin_resolved"
DEFAULT_NEURAL_XC_COEFFICIENT_PRIOR_MODE = "mean"
DEFAULT_NEURAL_XC_ENERGY_MODE = "graddft_coeff_basis_hf_pt2_heads"
DEFAULT_NEURAL_XC_HF_INPUT_MODE = "spin_resolved"
DEFAULT_NEURAL_XC_HF_CHANNEL_MODE = "auto"
DEFAULT_NEURAL_XC_RESPONSE_HF_MODE = "nonlocal_exchange_only"
DEFAULT_NEURAL_XC_RESPONSE_PT2_MODE = "local_projected"


def _normalize_semilocal_xc(semilocal_xc: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(semilocal_xc, str):
        return (semilocal_xc,)
    return tuple(str(name) for name in semilocal_xc)


def resolve_coefficient_prior_values(
    semilocal_xc: str | Sequence[str],
    explicit_values: Sequence[float] | None = None,
    *,
    energy_mode: str | None = None,
) -> tuple[float, ...] | None:
    """Resolve default coefficient priors for the default Neural_xc basis."""

    if explicit_values is not None:
        return tuple(float(value) for value in explicit_values)

    if energy_mode in {"dldh_two_lmf", "graddft_coeff_basis_hf_pt2_heads"}:
        return None

    if _normalize_semilocal_xc(semilocal_xc) == _normalize_semilocal_xc(
        DEFAULT_NEURAL_XC_SEMILOCAL_XC
    ):
        return DEFAULT_NEURAL_XC_COEFFICIENT_PRIOR_VALUES
    return None
