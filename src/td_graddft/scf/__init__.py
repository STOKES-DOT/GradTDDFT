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
    restricted_molecule_from_spec_with_jax_rks,
    unrestricted_molecule_from_spec_with_jax_uks,
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
    "restricted_molecule_from_spec_with_jax_rks",
    "run_rks_from_integrals",
    "UKSConfig",
    "UKSResult",
    "UKS",
    "unrestricted_molecule_from_spec_with_jax_uks",
    "run_uks_from_integrals",
]
