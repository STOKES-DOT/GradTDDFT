from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
from pyscf import dft, gto

from td_graddft import neural_xc
from td_graddft.jax_libxc import b3lyp_component_basis
from td_graddft.tddft.response import build_restricted_response_matrices
from td_graddft.training.targets import predict_excitation_energies
from td_graddft.workflows.core import run_reference
from td_graddft.workflows.types import SimulationConfig


WATER_GEOM = """
O  0.000000  0.000000  0.117790
H  0.000000  0.755453 -0.471161
H  0.000000 -0.755453 -0.471161
"""


@dataclass(frozen=True)
class GradientStats:
    value_finite: bool
    finite_count: int
    total_count: int
    finite_fraction: float
    absmax: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose TDDFT differentiability for Neural_xc on a water molecule."
    )
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--xc", default="b3lyp")
    parser.add_argument("--nstates", type=int, default=3)
    parser.add_argument(
        "--semilocal-case",
        choices=("lda_only", "gga_pbe", "b3lyp_basis", "all"),
        default="all",
        help="Which semilocal backbone case to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _make_water_mf(*, basis: str, xc: str):
    mol = gto.Mole()
    mol.atom = WATER_GEOM
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.spin = 0
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF reference SCF did not converge.")
    return mf


def _gradient_stats(value: jax.Array, grad_tree) -> GradientStats:
    leaves = jax.tree_util.tree_leaves(grad_tree)
    total_count = 0
    finite_count = 0
    absmax = 0.0
    for leaf in leaves:
        arr = jnp.asarray(leaf)
        total_count += int(arr.size)
        finite_count += int(jnp.sum(jnp.isfinite(arr)))
        safe_abs = jnp.nan_to_num(jnp.abs(arr), nan=0.0, posinf=0.0, neginf=0.0)
        absmax = max(absmax, float(jnp.max(safe_abs)))
    return GradientStats(
        value_finite=bool(jnp.isfinite(value)),
        finite_count=finite_count,
        total_count=total_count,
        finite_fraction=float(finite_count / max(total_count, 1)),
        absmax=absmax,
    )


def _case_specs(selected: str) -> list[tuple[str, tuple[str, ...]]]:
    cases = {
        "lda_only": ("lda_x", "lda_c_pw"),
        "gga_pbe": ("gga_x_pbe", "gga_c_pbe"),
        "b3lyp_basis": b3lyp_component_basis(),
    }
    if selected == "all":
        return list(cases.items())
    return [(selected, cases[selected])]


def main() -> None:
    args = _parse_args()
    mf = _make_water_mf(basis=str(args.basis), xc=str(args.xc))
    reference = run_reference(
        mf,
        scf_elapsed_s=0.0,
        simulation=SimulationConfig(
            nstates=int(args.nstates),
            scf_backend="pyscf",
            execution_device="cpu",
            jit_tddft=False,
        ),
    )
    molecule = reference.molecule

    report: dict[str, dict[str, dict[str, float | bool | int]]] = {}
    for label, semilocal_xc in _case_specs(str(args.semilocal_case)):
        functional = neural_xc.Functional(
            semilocal_xc=semilocal_xc,
            hidden_dims=(32, 32),
            name=f"diag_{label}",
        )
        params = functional.init_from_molecule(jax.random.PRNGKey(int(args.seed)), molecule)

        def gs_energy(p):
            return functional.energy_from_molecule(p, molecule)

        def gs_bound_potential_sum(p):
            bound = functional.bind_to_molecule_for_scf(p, molecule)
            return jnp.sum(bound.grid_potential(molecule))

        def es_response_tensor_sum(p):
            bound = functional.bind_to_molecule(p, molecule)
            return jnp.sum(bound.grid_response_tensor(molecule))

        def a_matrix_sum(p):
            matrices = build_restricted_response_matrices(
                molecule,
                functional,
                xc_params=p,
            )
            return jnp.sum(matrices.a_matrix)

        def tda_s1(p):
            return predict_excitation_energies(
                p,
                functional,
                molecule,
                nstates=1,
                use_tda=True,
            )[0]

        case_report = {}
        for metric_name, fn in (
            ("gs_energy", gs_energy),
            ("gs_bound_potential_sum", gs_bound_potential_sum),
            ("es_response_tensor_sum", es_response_tensor_sum),
            ("a_matrix_sum", a_matrix_sum),
            ("tda_s1", tda_s1),
        ):
            value, grad = jax.value_and_grad(fn)(params)
            case_report[metric_name] = asdict(_gradient_stats(value, grad))
        report[label] = case_report

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
