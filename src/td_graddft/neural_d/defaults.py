from __future__ import annotations

from typing import Literal


DispersionArchitecture = Literal["graddft_residual"]

DEFAULT_DISPERSION_ARCHITECTURE: DispersionArchitecture = "graddft_residual"
DEFAULT_DISPERSION_HIDDEN_DIMS: tuple[int, ...] = (128, 128, 128, 128, 128)
DEFAULT_DISPERSION_SIGMOID_SCALE_FACTOR: float = 2.0
DEFAULT_DISPERSION_R0_FLOOR: float = 1e-12


__all__ = [
    "DEFAULT_DISPERSION_ARCHITECTURE",
    "DEFAULT_DISPERSION_HIDDEN_DIMS",
    "DEFAULT_DISPERSION_R0_FLOOR",
    "DEFAULT_DISPERSION_SIGMOID_SCALE_FACTOR",
    "DispersionArchitecture",
]
