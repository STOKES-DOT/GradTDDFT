"""GradDFT-style neural dispersion corrections."""

from .defaults import (
    DEFAULT_DISPERSION_ARCHITECTURE,
    DEFAULT_DISPERSION_HIDDEN_DIMS,
    DEFAULT_DISPERSION_R0_FLOOR,
    DEFAULT_DISPERSION_SIGMOID_SCALE_FACTOR,
    DispersionArchitecture,
)
from .factory import make_neural_d_functional
from .functional import (
    DispersionCorrectedFunctional,
    DispersionFunctional,
    calculate_distances,
    make_dispersion_corrected_functional,
)
from .inputs import build_dispersion_pair_inputs
from .networks import GradDFTDispersionNetwork

__all__ = [
    "DEFAULT_DISPERSION_ARCHITECTURE",
    "DEFAULT_DISPERSION_HIDDEN_DIMS",
    "DEFAULT_DISPERSION_R0_FLOOR",
    "DEFAULT_DISPERSION_SIGMOID_SCALE_FACTOR",
    "DispersionCorrectedFunctional",
    "DispersionArchitecture",
    "DispersionFunctional",
    "GradDFTDispersionNetwork",
    "build_dispersion_pair_inputs",
    "calculate_distances",
    "make_dispersion_corrected_functional",
    "make_neural_d_functional",
]
