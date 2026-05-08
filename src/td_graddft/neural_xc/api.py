from __future__ import annotations

import warnings
from typing import Any, Literal

from .dm21.functional import make_dm21_like_functional as _make_dm21_like_functional
from .dm21.functional import make_neural_xc_functional

_ARCHITECTURE_ALIASES = {
    "residual": "graddft_residual",
    "graddft_residual": "graddft_residual",
    "mlp": "simple_mlp",
    "simple_mlp": "simple_mlp",
}


def _normalize_architecture(architecture: str) -> Literal["simple_mlp", "graddft_residual"]:
    key = str(architecture).lower()
    try:
        return _ARCHITECTURE_ALIASES[key]  # type: ignore[return-value]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported architecture={architecture!r}. "
            "Expected 'residual', 'graddft_residual', 'mlp', or 'simple_mlp'."
        ) from exc


def make_functional(
    *,
    architecture: str = "residual",
    **kwargs: Any,
):
    return make_neural_xc_functional(
        network_architecture=_normalize_architecture(architecture),
        **kwargs,
    )


def Functional(
    *,
    architecture: str = "residual",
    **kwargs: Any,
):
    return make_functional(architecture=architecture, **kwargs)


def make_long_range_correction(
    *,
    base_functional: Any | None = None,
    hidden_dims: tuple[int, ...] = (64, 64, 32),
    alpha_scale: float = 1.0,
    gamma_floor: float = 1e-3,
    **kwargs: Any,
):
    from ..tddft.long_range_correction import LongRangeCorrectedFunctional, LongRangeXCNet

    return LongRangeCorrectedFunctional(
        base_functional=base_functional,
        model=LongRangeXCNet(
            hidden_dims=tuple(int(value) for value in hidden_dims),
            alpha_scale=float(alpha_scale),
            gamma_floor=float(gamma_floor),
        ),
        **kwargs,
    )


def LongRangeCorrection(
    *,
    base_functional: Any | None = None,
    hidden_dims: tuple[int, ...] = (64, 64, 32),
    alpha_scale: float = 1.0,
    gamma_floor: float = 1e-3,
    **kwargs: Any,
):
    return make_long_range_correction(
        base_functional=base_functional,
        hidden_dims=hidden_dims,
        alpha_scale=alpha_scale,
        gamma_floor=gamma_floor,
        **kwargs,
    )


def make_dm21_like_functional(*args: Any, **kwargs: Any):
    warnings.warn(
        "neural_xc.make_dm21_like_functional is deprecated; "
        "use neural_xc.make_functional instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _make_dm21_like_functional(*args, **kwargs)


__all__ = [
    "Functional",
    "LongRangeCorrection",
    "make_functional",
    "make_long_range_correction",
    "make_dm21_like_functional",
]
