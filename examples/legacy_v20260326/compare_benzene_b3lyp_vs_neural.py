from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

from pyscf import dft, gto

from td_graddft_legacy.workflows import (
    NeuralXCTrainingConfig,
    OutputConfig,
    SimulationConfig,
    SpectrumGridConfig,
    run_and_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Neural_xc on benzene ground state and compare absorption "
            "spectrum against PySCF TDDFT."
        )
    )
    parser.add_argument("--steps", type=int, default=2500, help="training steps")
    parser.add_argument("--lr", type=float, default=0.005, help="Adam learning rate")
    parser.add_argument(
        "--density-weight",
        type=float,
        default=1e-3,
        help="weight for density stationarity penalty",
    )
    parser.add_argument(
        "--states",
        type=int,
        default=-1,
        help="number of excited states (<=0 means full nocc*nvir)",
    )
    parser.add_argument("--eta-ev", type=float, default=0.20, help="Lorentzian width in eV")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--basis", type=str, default="sto-3g", help="AO basis for PySCF")
    parser.add_argument("--xc", type=str, default="b3lyp", help="reference XC for PySCF")
    parser.add_argument(
        "--grids-level",
        type=int,
        default=0,
        help="PySCF numerical integration grid level",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="benzene_b3lyp_vs_neural_xc",
        help="output file prefix under outputs/",
    )
    return parser.parse_args()


def make_benzene_mf(*, basis: str, xc: str, grids_level: int):
    mol = gto.Mole()
    mol.atom = """
    C   0.000000   1.396792   0.000000
    C   1.209657   0.698396   0.000000
    C   1.209657  -0.698396   0.000000
    C   0.000000  -1.396792   0.000000
    C  -1.209657  -0.698396   0.000000
    C  -1.209657   0.698396   0.000000
    H   0.000000   2.484212   0.000000
    H   2.151390   1.242106   0.000000
    H   2.151390  -1.242106   0.000000
    H   0.000000  -2.484212   0.000000
    H  -2.151390  -1.242106   0.000000
    H  -2.151390   1.242106   0.000000
    """
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.spin = 0
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = grids_level
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF ground-state SCF did not converge for benzene.")
    return mf


def main() -> None:
    args = parse_args()
    system_label = f"Benzene (C6H6), {args.xc.upper()}/{args.basis.upper()}"
    run_and_report(
        system_label=system_label,
        mf_builder=lambda: make_benzene_mf(
            basis=args.basis,
            xc=args.xc,
            grids_level=args.grids_level,
        ),
        training_config=NeuralXCTrainingConfig(
            steps=args.steps,
            learning_rate=args.lr,
            density_constraint_weight=args.density_weight,
            seed=args.seed,
            hidden_dims=(96, 96, 96),
            semilocal_xc="b3lyp_sl_approx",
            functional_name="benzene_neural_xc_fit",
        ),
        simulation_config=SimulationConfig(nstates=args.states),
        spectrum_config=SpectrumGridConfig(
            eta_ev=args.eta_ev,
            grid_min_ev=0.0,
            grid_points=3500,
            max_padding_ev=2.0,
            zoom_min_ev=3.0,
            zoom_max_ev=12.0,
            compare_states=20,
        ),
        output_config=OutputConfig(
            outdir=Path("outputs"),
            prefix=args.prefix,
            title="Benzene Absorption Spectrum: B3LYP vs Neural_xc",
            reference_label=f"PySCF TDDFT {args.xc.upper()}/{args.basis.upper()}",
            neural_label_template="JAX libxc + Neural_xc TDDFT ({solver})",
            write_training_curves=True,
            training_prefix="benzene_neural_xc_training_curve",
        ),
        print_all_states=True,
    )


if __name__ == "__main__":
    main()
