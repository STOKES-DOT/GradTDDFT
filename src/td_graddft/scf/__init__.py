"""Pure-JAX SCF solvers."""

from . import core
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
    run_rks_from_integrals,
)
from .uks import (
    UKSConfig,
    UKSResult,
    run_uks_from_integrals,
)
from .facade import RKS, UKS
from .builders import (
    precompile_restricted_cuda_direct_rks_solver,
    restricted_molecule_from_spec_with_gpu4pyscf_rks,
    restricted_molecule_from_spec_with_jax_rks,
    unrestricted_molecule_from_spec_with_jax_uks,
)
from .gpu4pyscf import (
    GPU4PYSCF_RKS_RUNTIME_BACKEND,
    GPU4PySCFRKSForwardOptions,
    GPU4PySCFRKSForwardResult,
    molecule_from_gpu4pyscf_rks_forward_result,
    run_gpu4pyscf_rks_forward,
)
from .molecules import QuadratureGrid, RestrictedMolecule, UnrestrictedMolecule
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
    "QuadratureGrid",
    "RestrictedMolecule",
    "UnrestrictedMolecule",
    "RKSIntegralInputs",
    "UKSIntegralInputs",
    "build_rks_integral_inputs",
    "build_uks_integral_inputs",
    "core",
    "precompile_restricted_cuda_direct_rks_solver",
    "GPU4PYSCF_RKS_RUNTIME_BACKEND",
    "GPU4PySCFRKSForwardOptions",
    "GPU4PySCFRKSForwardResult",
    "molecule_from_gpu4pyscf_rks_forward_result",
    "restricted_molecule_from_spec_with_gpu4pyscf_rks",
    "restricted_molecule_from_spec_with_jax_rks",
    "run_gpu4pyscf_rks_forward",
    "run_rks_from_integrals",
    "UKSConfig",
    "UKSResult",
    "UKS",
    "unrestricted_molecule_from_spec_with_jax_uks",
    "run_uks_from_integrals",
]
