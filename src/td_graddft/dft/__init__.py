"""PySCF-style DFT namespace for TD-GradDFT."""

from ..scf import RKS, UKS
from .rks import (
    RKSConfig,
    RKSResult,
    restricted_molecule_from_spec_with_jax_rks,
    run_rks_from_integrals,
)
from .uks import UKSConfig, UKSResult, run_uks_from_integrals
from .xc import (
    eval_xc_energy_density,
    eval_xc_response_tensor,
    hybrid_coeff,
    parse_xc,
    semilocal_terms,
    xc_type,
)
from .rsh import (
    RSHFunctionalTemplate,
    RSHParameterBounds,
    RSHFunctionalPreset,
    ResolvedRSHParameters,
    SCFXCContributions,
    canonical_rsh_preset_name,
    get_rsh_functional_preset,
    list_rsh_functional_presets,
    make_rsh_template,
)
from .trainable_rsh import (
    BoundTrainableRSHFunctional,
    RSHParameterHead,
    TrainableRSHFunctional,
    make_minimal_trainable_rsh_functional,
)

__all__ = [
    "RKS",
    "UKS",
    "RKSConfig",
    "RKSResult",
    "UKSConfig",
    "UKSResult",
    "run_rks_from_integrals",
    "run_uks_from_integrals",
    "restricted_molecule_from_spec_with_jax_rks",
    "eval_xc_energy_density",
    "eval_xc_response_tensor",
    "hybrid_coeff",
    "parse_xc",
    "semilocal_terms",
    "xc_type",
    "RSHFunctionalTemplate",
    "RSHParameterBounds",
    "RSHFunctionalPreset",
    "ResolvedRSHParameters",
    "SCFXCContributions",
    "canonical_rsh_preset_name",
    "get_rsh_functional_preset",
    "list_rsh_functional_presets",
    "make_rsh_template",
    "BoundTrainableRSHFunctional",
    "RSHParameterHead",
    "TrainableRSHFunctional",
    "make_minimal_trainable_rsh_functional",
]
