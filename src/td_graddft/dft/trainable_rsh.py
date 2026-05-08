"""Compatibility wrapper for neural RSH functionals.

Prefer importing from ``td_graddft.nn_rsh`` or ``td_graddft.nn_rsh.functional``.
"""

from ..nn_rsh.functional import (
    BoundTrainableRSHFunctional,
    RSHParameterHead,
    TrainableRSHFunctional,
    make_minimal_trainable_rsh_functional,
)

__all__ = [
    "BoundTrainableRSHFunctional",
    "RSHParameterHead",
    "TrainableRSHFunctional",
    "make_minimal_trainable_rsh_functional",
]
