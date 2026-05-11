"""Compatibility wrapper for neural RSH schema.

Prefer importing from ``td_graddft.nn_rsh`` or ``td_graddft.nn_rsh.schema``.
"""

from ..nn_rsh.schema import (
    RSHFunctionalTemplate,
    RSHParameterBounds,
    ResolvedRSHParameters,
    SCFXCContributions,
)
from ..nn_rsh.presets import (
    RSHFunctionalPreset,
    canonical_rsh_preset_name,
    get_rsh_functional_preset,
    list_rsh_functional_presets,
    make_rsh_template,
)

__all__ = [
    "RSHFunctionalTemplate",
    "RSHParameterBounds",
    "ResolvedRSHParameters",
    "SCFXCContributions",
    "RSHFunctionalPreset",
    "canonical_rsh_preset_name",
    "get_rsh_functional_preset",
    "list_rsh_functional_presets",
    "make_rsh_template",
]
