from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jaxtyping import Array

from td_graddft.jax_runtime import (
    DEFAULT_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS,
    DEFAULT_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES,
)
from td_graddft.neural_xc import (
    DEFAULT_NEURAL_XC_COEFFICIENT_PRIOR_MODE,
    DEFAULT_NEURAL_XC_DENSITY_SUPERVISION,
    DEFAULT_NEURAL_XC_HF_INPUT_MODE,
    DEFAULT_NEURAL_XC_RESPONSE_PT2_MODE,
    DEFAULT_NEURAL_XC_SEMILOCAL_XC,
    DEFAULT_INPUT_FEATURE_MODE,
    DEFAULT_NETWORK_HIDDEN_DIMS,
)


@dataclass(frozen=True)
class NeuralXCTrainingConfig:
    """Configuration for fixed-density or implicit self-consistent Neural_xc fitting."""

    steps: int = 2000
    learning_rate: float = 5e-5
    gradient_clip_norm: float | None = None
    lr_decay_every: int = 0
    lr_decay_factor: float = 0.5
    jit_train: bool = True
    log_interval: int = 0
    density_constraint_weight: float = 0.0
    stationarity_constraint_weight: float = 0.0
    dm21_scf_regularization_weight: float = 0.0
    orbital_energy_constraint_weight: float = 0.0
    orbital_energy_constraint_window: int = 10
    density_supervision: Literal["spin_summed", "spin_resolved"] = DEFAULT_NEURAL_XC_DENSITY_SUPERVISION
    coefficient_prior_weight: float = 0.0
    coefficient_prior_values: tuple[float, ...] | None = None
    coefficient_prior_mode: Literal["pointwise", "mean"] = DEFAULT_NEURAL_XC_COEFFICIENT_PRIOR_MODE
    energy_mse_weight: float = 0.0
    energy_mae_weight: float = 1.0
    orbital_energy_mse_weight: float = 0.0
    orbital_energy_mae_weight: float = 1.0
    energy_normalization: Literal["none", "per_electron", "per_atom"] = "per_electron"
    s1_constraint_weight: float = 0.0
    s1_constraint_use_tda: bool = True
    s1_target_energy_au: float | None = None
    excitation_constraint_weight: float = 0.0
    excitation_constraint_nstates: int = 3
    excitation_constraint_use_tda: bool = True
    excitation_mse_weight: float = 0.0
    excitation_mae_weight: float = 1.0
    oscillator_strength_constraint_weight: float = 0.0
    oscillator_strength_constraint_nstates: int = 3
    oscillator_strength_constraint_use_tda: bool = True
    oscillator_strength_mse_weight: float = 0.0
    oscillator_strength_mae_weight: float = 1.0
    excitation_target_energies_au: tuple[float, ...] | None = None
    spectrum_constraint_weight: float = 0.0
    spectrum_constraint_nstates: int = 5
    spectrum_constraint_use_tda: bool = True
    spectrum_mse_weight: float = 0.0
    spectrum_mae_weight: float = 1.0
    seed: int = 0
    hidden_dims: tuple[int, ...] = DEFAULT_NETWORK_HIDDEN_DIMS
    semilocal_xc: str | tuple[str, ...] = DEFAULT_NEURAL_XC_SEMILOCAL_XC
    n_semilocal_channels: int | None = None
    input_feature_mode: Literal["enhanced", "canonical"] = DEFAULT_INPUT_FEATURE_MODE
    hf_input_mode: Literal["total_only", "spin_resolved"] = DEFAULT_NEURAL_XC_HF_INPUT_MODE
    include_pt2_channel: bool = False
    ground_state_pt2_mode: Literal["off", "nograd", "scf"] | None = None
    pt2_channel_mode: Literal["scaled_projected", "local_exact"] = "scaled_projected"
    response_pt2_mode: Literal["approx", "strict"] = (
        DEFAULT_NEURAL_XC_RESPONSE_PT2_MODE
    )
    strict_feature_alignment: bool = True
    hfx_channels: int = 2
    dm21_hfx_omega_values: tuple[float, ...] = (0.0, 0.4)
    dm21_hfx_chunk_size: int = 512
    functional_name: str = "neural_xc_fit"
    squash_offset: float = 1e-4
    sigmoid_scale_factor: float = 2.0
    training_mode: str = "fixed_density"
    graddft_core_defaults: bool = False
    strict_graddft_ground_state: bool = False
    fractional_linearity_weight: float = 0.0
    fractional_linearity_delta: float = 0.1
    dm21_scf_gap_floor: float = 1e-3
    scf_max_cycle: int = 12
    scf_damping: float = 0.25
    scf_conv_tol_density: float = 1e-8
    scf_orthogonalization_eps: float = 1e-10
    scf_vxc_clip: float = 20.0
    scf_iterate_selection: Literal["final", "best_rms", "first_converged"] = "final"
    scf_require_convergence: bool = False
    scf_gradient_mode: Literal["impl"] = "impl"
    scf_implicit_diff_max_iter: int = 24
    scf_implicit_diff_clip: float = 1e4
    scf_implicit_diff_tolerance: float = 1e-6
    scf_implicit_diff_regularization: float = 0.0
    recover_nonfinite_steps: bool = True


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for excited-state simulation dimensions."""

    nstates: int = -1
    occupation_tolerance: float = 1e-8
    scf_backend: str = "jax_rks"
    jax_basis_max_l: int = 3
    jax_grid_ao_backend: Literal["jax"] = "jax"
    jax_rhf_max_cycle: int = 80
    jax_rhf_conv_tol: float = 1e-10
    jax_rhf_conv_tol_density: float = 1e-8
    jax_rks_xc_spec: str | None = None
    jax_rks_max_cycle: int = 50
    jax_rks_conv_tol: float = 1e-9
    jax_rks_conv_tol_density: float = 1e-7
    jax_rks_damping: float = 0.15
    jax_rks_density_floor: float = 1e-12
    jax_rks_potential_clip: float = 20.0
    jax_rks_jk_backend: Literal["full", "df", "direct"] = "full"
    jax_rks_df_tol: float = 1e-10
    jax_rks_df_max_rank: int | None = None
    jax_precompile_eri: bool = False
    jax_precompile_eri_chunk_size: int = 512
    jax_compilation_cache_dir: str | None = None
    jax_persistent_cache_min_compile_time_secs: float = (
        DEFAULT_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS
    )
    jax_persistent_cache_min_entry_size_bytes: int = (
        DEFAULT_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES
    )
    jax_uks_xc_spec: str | None = None
    jax_uks_max_cycle: int = 50
    jax_uks_conv_tol: float = 1e-9
    jax_uks_conv_tol_density: float = 1e-7
    jax_uks_damping: float = 0.15
    jax_uks_density_floor: float = 1e-12
    jax_uks_potential_clip: float = 20.0
    execution_device: Literal["auto", "cpu", "gpu"] = "auto"
    move_reference_to_device: bool = True
    jax_integral_backend: Literal["jax", "cpu", "gpu", "libcint"] = "cpu"
    jax_libcint_geometry_grad_policy: Literal["analytic", "error", "zero"] = "analytic"
    jit_tddft: bool = True
    jit_spectrum: bool = True


@dataclass(frozen=True)
class MoleculeSpecConfig:
    """Spec-driven system definition for strict-JAX molecule construction."""

    atom: Any
    basis: str
    xc: str = "pbe"
    unit: str = "Angstrom"
    charge: int = 0
    spin: int = 0
    cart: bool = True
    grids_level: int = 0
    verbose: int = 0


@dataclass(frozen=True)
class SpectrumGridConfig:
    """Configuration for stick-spectrum broadening and plotting ranges."""

    eta_ev: float = 0.15
    grid_min_ev: float = 0.0
    grid_points: int = 2200
    max_padding_ev: float = 2.0
    zoom_min_ev: float = 5.0
    zoom_max_ev: float = 45.0
    compare_states: int = 8


@dataclass(frozen=True)
class OutputConfig:
    """Configuration for filenames, labels, and output paths."""

    outdir: Path = Path("outputs")
    prefix: str = "system_b3lyp_vs_neural_xc"
    title: str = "Absorption Spectrum: B3LYP vs Neural_xc"
    reference_label: str = "PySCF TDDFT"
    neural_label_template: str = "JAX libxc + Neural_xc TDDFT ({solver})"
    write_training_curves: bool = True
    training_prefix: str | None = None


@dataclass(frozen=True)
class OutputPaths:
    spectrum_csv: Path
    spectrum_png: Path
    training_csv: Path | None = None
    training_png: Path | None = None


@dataclass(frozen=True)
class MoleculeRun:
    molecule: Any
    nocc: int
    nvir: int
    nstates: int
    nstates_full: int
    energies_au: Array
    oscillator_strengths: Array
    scf_elapsed_s: float
    tddft_elapsed_s: float

@dataclass(frozen=True)
class TrainingRun:
    functional: Any
    params: Any
    initial_loss: float
    final_loss: float
    min_loss: float
    min_loss_step: int
    initial_density_penalty: float
    final_density_penalty: float
    initial_stationarity_penalty: float
    final_stationarity_penalty: float
    initial_coefficient_prior_penalty: float
    final_coefficient_prior_penalty: float
    loss_history: list[float]
    density_penalty_history: list[float]
    stationarity_penalty_history: list[float]
    coefficient_prior_penalty_history: list[float]
    grad_norm_history: list[float]
    grad_abs_max_history: list[float]
    param_update_norm_history: list[float]
    nonfinite_grad_fraction_history: list[float]
    trained_energy: float
    trained_hybrid_fraction: float
    elapsed_s: float


@dataclass(frozen=True)
class NeuralExcitedStateRun:
    solver_label: str
    energies_au: Array
    oscillator_strengths: Array
    elapsed_s: float


@dataclass(frozen=True)
class SpectrumRun:
    grid_ev: Array
    reference_curve: Array
    neural_curve: Array
    low_energy_mask: Array
    low_energy_mae_ev: float
    compared_states: int


@dataclass(frozen=True)
class PipelineRun:
    system_label: str
    reference: MoleculeRun
    training: TrainingRun
    neural: NeuralExcitedStateRun
    spectrum: SpectrumRun
    outputs: OutputPaths
