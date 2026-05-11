"""Grid/AO evaluation helpers under a PySCF-style gto namespace."""

from ..data.grid import build_molecular_grid
from ..data.grid_ao import evaluate_cartesian_ao
from ..scf.molecules import QuadratureGrid

GridReference = QuadratureGrid

__all__ = [
    "QuadratureGrid",
    "build_molecular_grid",
    "evaluate_cartesian_ao",
]
