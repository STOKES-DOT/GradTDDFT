from __future__ import annotations

from ..jax_libxc import b3lyp_component_basis
from .config import ExperimentConfig, SystemConfig
from .types import (
    NeuralXCTrainingConfig,
    MoleculeSpecConfig,
    SimulationConfig,
    SpectrumGridConfig,
)


def _water_atom_block() -> str:
    return """
    O  0.000000  0.000000  0.117790
    H  0.000000  0.755453 -0.471161
    H  0.000000 -0.755453 -0.471161
    """


def _benzene_atom_block() -> str:
    return """
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


def _make_water_mf(*, basis: str = "sto-3g", xc: str = "b3lyp", grids_level: int = 0):
    del basis, xc, grids_level
    raise RuntimeError(
        "Legacy external mean-field presets were removed from the TD-GradDFT runtime. "
        "Use water_strict_jax_experiment_config instead."
    )


def _make_benzene_mf(*, basis: str = "sto-3g", xc: str = "b3lyp", grids_level: int = 0):
    del basis, xc, grids_level
    raise RuntimeError(
        "Legacy external mean-field presets were removed from the TD-GradDFT runtime. "
        "Use benzene_strict_jax_experiment_config instead."
    )


def legacy_water_experiment_config(
    *,
    basis: str = "sto-3g",
    xc: str = "b3lyp",
    steps: int = 2000,
) -> ExperimentConfig:
    """Legacy PySCF-mf preset for H2O ground-state fit + excited-state comparison."""

    system = SystemConfig(
        name=f"H2O {xc.upper()}/{basis.upper()}",
        mf_builder=lambda: _make_water_mf(basis=basis, xc=xc),
        output_prefix="water_b3lyp_vs_neural_xc",
        plot_title=f"H2O Absorption Spectrum: {xc.upper()} vs Neural_xc",
        print_all_states=True,
    )
    return ExperimentConfig(
        experiment_name="water_neural_xc",
        systems=[system],
        training=NeuralXCTrainingConfig(
            steps=steps,
            semilocal_xc=b3lyp_component_basis(),
            functional_name="water_neural_xc_fit",
        ),
        simulation=SimulationConfig(nstates=-1),
        spectrum=SpectrumGridConfig(
            eta_ev=0.15,
            grid_min_ev=0.0,
            zoom_min_ev=5.0,
            zoom_max_ev=45.0,
            compare_states=8,
        ),
    )


def legacy_benzene_experiment_config(
    *,
    basis: str = "sto-3g",
    xc: str = "b3lyp",
    steps: int = 1200,
) -> ExperimentConfig:
    """Legacy PySCF-mf preset for C6H6 benchmark training and spectrum comparison."""

    system = SystemConfig(
        name=f"Benzene {xc.upper()}/{basis.upper()}",
        mf_builder=lambda: _make_benzene_mf(basis=basis, xc=xc),
        output_prefix="benzene_b3lyp_vs_neural_xc",
        plot_title=f"Benzene Absorption Spectrum: {xc.upper()} vs Neural_xc",
        print_all_states=True,
    )
    return ExperimentConfig(
        experiment_name="benzene_neural_xc",
        systems=[system],
        training=NeuralXCTrainingConfig(
            steps=steps,
            learning_rate=0.005,
            semilocal_xc=b3lyp_component_basis(),
            hidden_dims=(96, 96, 96),
            functional_name="benzene_neural_xc_fit",
        ),
        simulation=SimulationConfig(nstates=-1),
        spectrum=SpectrumGridConfig(
            eta_ev=0.20,
            grid_min_ev=0.0,
            grid_points=3500,
            zoom_min_ev=3.0,
            zoom_max_ev=12.0,
            compare_states=20,
        ),
    )


def water_strict_jax_experiment_config(
    *,
    basis: str = "sto-3g",
    xc: str = "b3lyp",
    steps: int = 2000,
) -> ExperimentConfig:
    """Strict-JAX preset for H2O without PySCF participation in the runtime path."""

    system = SystemConfig(
        name=f"H2O {xc.upper()}/{basis.upper()}",
        reference_spec=MoleculeSpecConfig(
            atom=_water_atom_block(),
            basis=basis,
            xc=xc,
            unit="Angstrom",
            charge=0,
            spin=0,
            cart=True,
            grids_level=0,
        ),
        output_prefix="water_jax_vs_neural_xc",
        plot_title=f"H2O Absorption Spectrum: JAX {xc.upper()} vs Neural_xc",
        reference_label=f"JAX {xc.upper()} TDDFT",
        print_all_states=True,
    )
    return ExperimentConfig(
        experiment_name="water_neural_xc_strict_jax",
        systems=[system],
        training=NeuralXCTrainingConfig(
            steps=steps,
            semilocal_xc=b3lyp_component_basis(),
            functional_name="water_neural_xc_fit",
        ),
        simulation=SimulationConfig(
            nstates=-1,
            scf_backend="jax_rks",
            jax_rks_xc_spec=xc,
            jax_grid_ao_backend="jax",
        ),
        spectrum=SpectrumGridConfig(
            eta_ev=0.15,
            grid_min_ev=0.0,
            zoom_min_ev=5.0,
            zoom_max_ev=45.0,
            compare_states=8,
        ),
    )


def benzene_strict_jax_experiment_config(
    *,
    basis: str = "sto-3g",
    xc: str = "b3lyp",
    steps: int = 1200,
) -> ExperimentConfig:
    """Strict-JAX preset for benzene without PySCF participation in the runtime path."""

    system = SystemConfig(
        name=f"Benzene {xc.upper()}/{basis.upper()}",
        reference_spec=MoleculeSpecConfig(
            atom=_benzene_atom_block(),
            basis=basis,
            xc=xc,
            unit="Angstrom",
            charge=0,
            spin=0,
            cart=True,
            grids_level=0,
        ),
        output_prefix="benzene_jax_vs_neural_xc",
        plot_title=f"Benzene Absorption Spectrum: JAX {xc.upper()} vs Neural_xc",
        reference_label=f"JAX {xc.upper()} TDDFT",
        print_all_states=True,
    )
    return ExperimentConfig(
        experiment_name="benzene_neural_xc_strict_jax",
        systems=[system],
        training=NeuralXCTrainingConfig(
            steps=steps,
            learning_rate=0.005,
            semilocal_xc=b3lyp_component_basis(),
            hidden_dims=(96, 96, 96),
            functional_name="benzene_neural_xc_fit",
        ),
        simulation=SimulationConfig(
            nstates=-1,
            scf_backend="jax_rks",
            jax_rks_xc_spec=xc,
            jax_grid_ao_backend="jax",
        ),
        spectrum=SpectrumGridConfig(
            eta_ev=0.20,
            grid_min_ev=0.0,
            grid_points=3500,
            zoom_min_ev=3.0,
            zoom_max_ev=12.0,
            compare_states=20,
        ),
    )


def water_experiment_config(
    *,
    basis: str = "sto-3g",
    xc: str = "b3lyp",
    steps: int = 2000,
) -> ExperimentConfig:
    """Default H2O preset. This now routes to the strict-JAX spec-driven pipeline."""

    return water_strict_jax_experiment_config(basis=basis, xc=xc, steps=steps)


def benzene_experiment_config(
    *,
    basis: str = "sto-3g",
    xc: str = "b3lyp",
    steps: int = 1200,
) -> ExperimentConfig:
    """Default benzene preset. This now routes to the strict-JAX spec-driven pipeline."""

    return benzene_strict_jax_experiment_config(basis=basis, xc=xc, steps=steps)


def water_legacy_experiment_config(
    *,
    basis: str = "sto-3g",
    xc: str = "b3lyp",
    steps: int = 2000,
) -> ExperimentConfig:
    """Compatibility alias for the legacy PySCF-based water preset."""

    return legacy_water_experiment_config(basis=basis, xc=xc, steps=steps)


def benzene_legacy_experiment_config(
    *,
    basis: str = "sto-3g",
    xc: str = "b3lyp",
    steps: int = 1200,
) -> ExperimentConfig:
    """Compatibility alias for the legacy PySCF-based benzene preset."""

    return legacy_benzene_experiment_config(basis=basis, xc=xc, steps=steps)
