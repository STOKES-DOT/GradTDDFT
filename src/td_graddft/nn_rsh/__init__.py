"""Dedicated neural RSH package.

This namespace isolates range-separated-hybrid-specific schemas, trainable
functionals, and self-supervised objectives from the generic DFT/training
stacks.
"""

from .schema import (
    RSHFunctionalTemplate,
    RSHParameterBounds,
    ResolvedRSHParameters,
    SCFXCContributions,
)
from .presets import (
    RSHFunctionalPreset,
    canonical_rsh_preset_name,
    get_rsh_functional_preset,
    list_rsh_functional_presets,
    make_rsh_template,
    rsh_preset_default_params,
)
from .descriptors import (
    AtomCenteredDensityDescriptorConfig,
    atom_centered_density_power_spectrum,
    make_atom_centered_density_descriptor_fn,
)
from .gnn import AttentionReadout, DistanceGatedAttention, RSHGNNHead
from .functional import (
    AtomwiseRSHParameterHead,
    BoundTrainableRSHFunctional,
    RSHParameterHead,
    TrainableRSHFunctional,
    make_atom_centered_density_rsh_functional,
    make_gnn_rsh_functional,
    make_minimal_trainable_rsh_functional,
)
from .api import RSH


def __getattr__(name: str):
    if name == "make_self_supervised_rsh_loss":
        from .losses import make_self_supervised_rsh_loss

        return make_self_supervised_rsh_loss
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
    "rsh_preset_default_params",
    "AtomCenteredDensityDescriptorConfig",
    "atom_centered_density_power_spectrum",
    "make_atom_centered_density_descriptor_fn",
    "AttentionReadout",
    "DistanceGatedAttention",
    "RSHGNNHead",
    "AtomwiseRSHParameterHead",
    "BoundTrainableRSHFunctional",
    "RSHParameterHead",
    "RSH",
    "TrainableRSHFunctional",
    "make_atom_centered_density_rsh_functional",
    "make_gnn_rsh_functional",
    "make_minimal_trainable_rsh_functional",
    "make_self_supervised_rsh_loss",
]
