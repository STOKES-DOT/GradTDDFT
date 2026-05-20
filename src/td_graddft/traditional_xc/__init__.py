"""Compatibility namespace for classic XC functional wrappers.

Prefer importing new code from ``td_graddft.dft.xc``.
"""

from ..dft.xc import (
    CLASSIC_XC_SPECS,
    TraditionalXCFunctional,
    make_b3lyp_functional,
    make_classic_xc_functional,
    make_lda_functional,
    make_pbe0_functional,
    make_pbe_functional,
)

__all__ = [
    "CLASSIC_XC_SPECS",
    "TraditionalXCFunctional",
    "make_b3lyp_functional",
    "make_classic_xc_functional",
    "make_lda_functional",
    "make_pbe0_functional",
    "make_pbe_functional",
]
