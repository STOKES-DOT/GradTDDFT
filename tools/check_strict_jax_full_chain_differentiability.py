from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp

from td_graddft import HARTREE_TO_EV, lorentzian_spectrum, neural_xc
from td_graddft.training import (
    ExcitedStateDatum,
    ExcitedStateTrainingConfig,
    GroundStateCoreDatum,
    GroundStateCoreTrainingConfig,
    GroundStateDatum,
    GroundStateTrainingConfig,
    ground_state_mse_loss,
)
from td_graddft.workflows.core import run_molecule_from_spec
from td_graddft.workflows.types import MoleculeSpecConfig, SimulationConfig


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


def _gradient_stats(value, grad_tree) -> GradientStats:
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check full-chain NeuralXC self-consistent differentiability."
    )
    parser.add_argument(
        "--include-unrolled",
        action="store_true",
        help="Also run unrolled-SCF diagnostics. The stable training default is implicit_commutator.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    reference = run_molecule_from_spec(
        MoleculeSpecConfig(
            atom=WATER_GEOM,
            basis="sto-3g",
            xc="pbe",
            unit="Angstrom",
            charge=0,
            spin=0,
            cart=True,
            grids_level=0,
        ),
        simulation=SimulationConfig(
            nstates=1,
            scf_backend="jax_rks",
            jax_rks_xc_spec="pbe",
            jax_grid_ao_backend="jax",
            execution_device="cpu",
            jax_compilation_cache_dir=None,
            jit_tddft=False,
        ),
        compute_local_hfx_features=True,
        compute_local_hfx_aux=True,
    )
    molecule = reference.molecule
    functional = neural_xc.Functional(
        semilocal_xc="pbe",
        hidden_dims=(16, 16),
        name="strict_jax_full_chain_diag",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(0), molecule)

    common_kwargs = dict(
        mode="self_consistent",
        scf_max_cycle=8,
        scf_damping=0.2,
        scf_conv_tol_density=1e-7,
    )
    target_energy = jnp.asarray(molecule.mf_energy)
    grid_ev = jnp.linspace(5.0, 20.0, 256)
    target_curve = lorentzian_spectrum(
        reference.energies_au[:1] * HARTREE_TO_EV,
        reference.oscillator_strengths[:1],
        grid_ev,
        eta=0.15,
    )

    checks: dict[str, tuple[GroundStateDatum, GroundStateTrainingConfig]] = {}
    gradient_modes = ["implicit_commutator"]
    if args.include_unrolled:
        gradient_modes.append("unrolled")
    for gradient_mode in gradient_modes:
        suffix = "implicit" if gradient_mode == "implicit_commutator" else "unrolled"
        checks[f"ground_self_consistent_{suffix}"] = (
            GroundStateDatum.from_parts(
                molecule,
                core=GroundStateCoreDatum(
                    target_total_energy=target_energy,
                ),
            ),
            GroundStateTrainingConfig.from_parts(
                core=GroundStateCoreTrainingConfig(
                    energy_mse_weight=0.0,
                    energy_mae_weight=1.0,
                    scf_gradient_mode=gradient_mode,
                    **common_kwargs,
                ),
            ),
        )
        checks[f"excitation_self_consistent_{suffix}"] = (
            GroundStateDatum.from_parts(
                molecule,
                core=GroundStateCoreDatum(
                    target_total_energy=target_energy,
                ),
                excited_state=ExcitedStateDatum(
                    target_excitation_energies=reference.energies_au[:1],
                    excitation_constraint_weight=1.0,
                    excitation_constraint_nstates=1,
                ),
            ),
            GroundStateTrainingConfig.from_parts(
                core=GroundStateCoreTrainingConfig(
                    energy_mse_weight=0.0,
                    energy_mae_weight=0.0,
                    scf_gradient_mode=gradient_mode,
                    **common_kwargs,
                ),
                excited_state=ExcitedStateTrainingConfig(
                    excitation_constraint_use_tda=True,
                    excitation_mse_weight=0.0,
                    excitation_mae_weight=1.0,
                ),
            ),
        )
        checks[f"spectrum_self_consistent_{suffix}"] = (
            GroundStateDatum.from_parts(
                molecule,
                core=GroundStateCoreDatum(
                    target_total_energy=target_energy,
                ),
                excited_state=ExcitedStateDatum(
                    target_spectrum_grid_ev=grid_ev,
                    target_spectrum_curve=target_curve,
                    spectrum_constraint_weight=1.0,
                    spectrum_constraint_nstates=1,
                ),
            ),
            GroundStateTrainingConfig.from_parts(
                core=GroundStateCoreTrainingConfig(
                    energy_mse_weight=0.0,
                    energy_mae_weight=0.0,
                    scf_gradient_mode=gradient_mode,
                    **common_kwargs,
                ),
                excited_state=ExcitedStateTrainingConfig(
                    spectrum_constraint_use_tda=True,
                    spectrum_constraint_eta_ev=0.15,
                    spectrum_mse_weight=1.0,
                    spectrum_mae_weight=0.0,
                ),
            ),
        )

    report: dict[str, dict[str, float | bool | int]] = {}
    for label, (datum, cfg) in checks.items():
        fn = lambda p, _datum=datum, _cfg=cfg: ground_state_mse_loss(  # noqa: E731
            p,
            functional,
            _datum,
            training_config=_cfg,
        )[0]
        value, grad = jax.value_and_grad(fn)(params)
        report[label] = asdict(_gradient_stats(value, grad))

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
