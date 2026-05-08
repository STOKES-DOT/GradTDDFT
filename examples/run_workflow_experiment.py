from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

from td_graddft.workflows import (
    benzene_experiment_config,
    run_experiment,
    water_experiment_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run config-driven Neural_xc training + TDDFT spectrum workflow."
    )
    parser.add_argument(
        "--system",
        choices=("water", "benzene"),
        default="benzene",
        help="molecule preset",
    )
    parser.add_argument("--basis", default="sto-3g", help="basis for the default strict-JAX reference")
    parser.add_argument("--steps", type=int, default=1200, help="training steps")
    parser.add_argument(
        "--states",
        type=int,
        default=-1,
        help="number of TDDFT states (<=0 means full nocc*nvir)",
    )
    parser.add_argument(
        "--scf-backend",
        choices=("pyscf", "jax_rhf", "jax_rks", "jax_uks"),
        default="jax_rks",
        help="ground-state orbital backend used by Neural_xc TDDFT",
    )
    parser.add_argument(
        "--jax-basis-max-l",
        type=int,
        default=3,
        help="maximum AO angular momentum for pure-JAX integral engine",
    )
    parser.add_argument(
        "--execution-device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="execution target for JAX TDDFT workflow",
    )
    parser.add_argument(
        "--no-jit-tddft",
        action="store_true",
        help="disable JIT for TDDFT solver path",
    )
    parser.add_argument(
        "--no-jit-spectrum",
        action="store_true",
        help="disable JIT for spectrum broadening path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.system == "water":
        config = water_experiment_config(basis=args.basis, steps=args.steps)
    else:
        config = benzene_experiment_config(basis=args.basis, steps=args.steps)

    # Keep existing preset knobs, only override the CLI-selected fields.
    config = replace(
        config,
        training=replace(config.training, steps=args.steps),
        simulation=replace(
            config.simulation,
            nstates=args.states,
            scf_backend=args.scf_backend,
            jax_basis_max_l=args.jax_basis_max_l,
            execution_device=args.execution_device,
            jit_tddft=not args.no_jit_tddft,
            jit_spectrum=not args.no_jit_spectrum,
        ),
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
