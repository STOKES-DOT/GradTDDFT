"""Pure-JAX SCF solvers."""

from .differentiable import (
    DifferentiableSCF,
    DifferentiableSCFConfig,
    DifferentiableSCFInfo,
)
from .rhf import (
    RHFConfig,
    RHFResult,
    nuclear_repulsion_energy,
    run_rhf,
    run_rhf_from_integrals,
)
from .rks import (
    RKSConfig,
    RKSResult,
    TraceableRKSResult,
    run_rks_from_integrals,
    run_rks_from_integrals_traceable,
)
from .uks import (
    UKSConfig,
    UKSResult,
    run_uks_from_integrals,
)
from .facade import RKS, UKS
from .inputs import (
    RKSIntegralInputs,
    UKSIntegralInputs,
    build_rks_integral_inputs,
    build_uks_integral_inputs,
)

__all__ = [
    "DifferentiableSCF",
    "DifferentiableSCFConfig",
    "DifferentiableSCFInfo",
    "RHFConfig",
    "RHFResult",
    "nuclear_repulsion_energy",
    "run_rhf",
    "run_rhf_from_integrals",
    "RKSConfig",
    "RKSResult",
    "RKS",
    "RKSIntegralInputs",
    "TraceableRKSResult",
    "UKSIntegralInputs",
    "build_rks_integral_inputs",
    "build_uks_integral_inputs",
    "run_rks_from_integrals",
    "run_rks_from_integrals_traceable",
    "UKSConfig",
    "UKSResult",
    "UKS",
    "run_uks_from_integrals",
]
