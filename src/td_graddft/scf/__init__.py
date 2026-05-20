"""Pure-JAX SCF solvers."""

from . import core
from .differentiable import (
    DifferentiableSCF,
    DifferentiableSCFConfig,
    DifferentiableSCFInfo,
)
from .implicit import (
    ImplicitFixedPointConfig,
    implicit_fixed_point_solution,
)
from .xc_energy import (
    XCEnergyPotentialResult,
    xc_energy_and_potential_from_density,
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
    restricted_molecule_from_spec_with_gpu4pyscf_rks,
    restricted_molecule_from_spec_with_jax_rks,
    unrestricted_molecule_from_spec_with_gpu4pyscf_uks,
    unrestricted_molecule_from_spec_with_jax_uks,
)
from .gpu4pyscf import (
    GPU4PYSCF_RKS_RUNTIME_BACKEND,
    GPU4PYSCF_UKS_RUNTIME_BACKEND,
    GPU4PySCFRKSForwardOptions,
    GPU4PySCFRKSForwardResult,
    GPU4PySCFUKSForwardOptions,
    GPU4PySCFUKSForwardResult,
    compute_gpu4pyscf_direct_jk_response,
    compute_gpu4pyscf_direct_jk_response_from_options,
    molecule_from_gpu4pyscf_rks_forward_result,
    run_gpu4pyscf_rks_forward,
    run_gpu4pyscf_uks_forward,
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
    "ImplicitFixedPointConfig",
    "implicit_fixed_point_solution",
    "XCEnergyPotentialResult",
    "xc_energy_and_potential_from_density",
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
    "GPU4PYSCF_RKS_RUNTIME_BACKEND",
    "GPU4PYSCF_UKS_RUNTIME_BACKEND",
    "GPU4PySCFRKSForwardOptions",
    "GPU4PySCFRKSForwardResult",
    "GPU4PySCFUKSForwardOptions",
    "GPU4PySCFUKSForwardResult",
    "compute_gpu4pyscf_direct_jk_response",
    "compute_gpu4pyscf_direct_jk_response_from_options",
    "molecule_from_gpu4pyscf_rks_forward_result",
    "restricted_molecule_from_spec_with_gpu4pyscf_rks",
    "unrestricted_molecule_from_spec_with_gpu4pyscf_uks",
    "restricted_molecule_from_spec_with_jax_rks",
    "run_gpu4pyscf_rks_forward",
    "run_gpu4pyscf_uks_forward",
    "run_rks_from_integrals",
    "UKSConfig",
    "UKSResult",
    "UKS",
    "unrestricted_molecule_from_spec_with_jax_uks",
    "run_uks_from_integrals",
]
