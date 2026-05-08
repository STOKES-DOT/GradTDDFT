from __future__ import annotations

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


def make_water_mf():
    mol = gto.Mole()
    mol.atom = """
    O  0.000000  0.000000  0.117790
    H  0.000000  0.755453 -0.471161
    H  0.000000 -0.755453 -0.471161
    """
    mol.unit = "Angstrom"
    mol.basis = "sto-3g"
    mol.spin = 0
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = "b3lyp"
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.kernel()
    return mf


def main() -> None:
    run_and_report(
        system_label="H2O, B3LYP/STO-3G",
        mf_builder=make_water_mf,
        training_config=NeuralXCTrainingConfig(
            steps=2000,
            learning_rate=0.01,
            density_constraint_weight=1e-3,
            seed=0,
            hidden_dims=(64, 64, 64),
            semilocal_xc="b3lyp_sl_approx",
            functional_name="water_neural_xc_fit",
        ),
        simulation_config=SimulationConfig(nstates=-1),
        spectrum_config=SpectrumGridConfig(
            eta_ev=0.15,
            grid_min_ev=5.0,
            grid_points=2200,
            max_padding_ev=2.0,
            zoom_min_ev=5.0,
            zoom_max_ev=45.0,
            compare_states=8,
        ),
        output_config=OutputConfig(
            outdir=Path("outputs"),
            prefix="water_b3lyp_vs_neural_xc",
            title="H2O Absorption Spectrum: B3LYP vs Neural_xc",
            reference_label="PySCF TDDFT B3LYP/STO-3G",
            neural_label_template="JAX libxc + Neural_xc TDDFT ({solver})",
            write_training_curves=False,
        ),
        print_all_states=True,
    )


if __name__ == "__main__":
    main()
