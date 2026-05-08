from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from jaxtyping import Array


@dataclass(frozen=True)
class GroundStateCoreDatum:
    """Ground-state supervision and regularization targets for one molecule."""

    target_total_energy: Array
    target_density_matrix: Array | None = None
    target_xc_potential: Array | None = None
    target_xc_kernel: Array | None = None
    density_constraint_weight: float = 0.0
    xc_potential_constraint_weight: float = 0.0
    xc_kernel_constraint_weight: float = 0.0
    xc_kernel_normalization_scale: float | None = None
    stationarity_constraint_weight: float = 0.0
    dm21_scf_regularization_weight: float = 0.0
    target_orbital_energies: Array | None = None
    target_orbital_occupations: Array | None = None
    orbital_energy_constraint_weight: float = 0.0
    orbital_energy_constraint_window: int | None = None
    janak_frontier_constraint_weight: float = 0.0


@dataclass(frozen=True)
class ExcitedStateDatum:
    """Excited-state supervision targets attached to one training molecule."""

    target_s1_energy: Array | None = None
    target_first_excited_total_energy: Array | None = None
    target_excitation_energies: Array | None = None
    target_oscillator_strengths: Array | None = None
    target_spectrum_grid_ev: Array | None = None
    target_spectrum_curve: Array | None = None
    s1_constraint_weight: float = 0.0
    first_excited_total_energy_constraint_weight: float = 0.0
    excitation_constraint_weight: float = 0.0
    excitation_constraint_nstates: int | None = None
    oscillator_strength_constraint_weight: float = 0.0
    oscillator_strength_constraint_nstates: int | None = None
    spectrum_constraint_weight: float = 0.0
    spectrum_constraint_nstates: int | None = None


@dataclass(frozen=True)
class GroundStateDatum:
    """Single training example with a GradDFT core and optional TD extension."""

    molecule: Any
    target_total_energy: Array
    target_density_matrix: Array | None = None
    target_s1_energy: Array | None = None
    target_first_excited_total_energy: Array | None = None
    target_excitation_energies: Array | None = None
    target_oscillator_strengths: Array | None = None
    target_spectrum_grid_ev: Array | None = None
    target_spectrum_curve: Array | None = None
    target_xc_potential: Array | None = None
    target_xc_kernel: Array | None = None
    weight: float = 1.0
    density_constraint_weight: float = 0.0
    xc_potential_constraint_weight: float = 0.0
    xc_kernel_constraint_weight: float = 0.0
    xc_kernel_normalization_scale: float | None = None
    stationarity_constraint_weight: float = 0.0
    dm21_scf_regularization_weight: float = 0.0
    target_orbital_energies: Array | None = None
    target_orbital_occupations: Array | None = None
    orbital_energy_constraint_weight: float = 0.0
    orbital_energy_constraint_window: int | None = None
    janak_frontier_constraint_weight: float = 0.0
    s1_constraint_weight: float = 0.0
    first_excited_total_energy_constraint_weight: float = 0.0
    excitation_constraint_weight: float = 0.0
    excitation_constraint_nstates: int | None = None
    oscillator_strength_constraint_weight: float = 0.0
    oscillator_strength_constraint_nstates: int | None = None
    spectrum_constraint_weight: float = 0.0
    spectrum_constraint_nstates: int | None = None

    @classmethod
    def from_reference(
        cls,
        reference: Any,
        *,
        target_total_energy: Array | None = None,
        require_hfx: bool = False,
        functional: Any | None = None,
        **kwargs: Any,
    ) -> "GroundStateDatum":
        if require_hfx:
            if getattr(reference, "hfx_local", None) is None:
                raise ValueError(
                    "GroundStateDatum.from_reference(require_hfx=True) requires "
                    "reference.hfx_local."
                )
            if getattr(reference, "hfx_nu", None) is None:
                raise ValueError(
                    "GroundStateDatum.from_reference(require_hfx=True) requires "
                    "reference.hfx_nu."
                )
        if functional is not None and bool(getattr(functional, "include_pt2_channel", False)):
            if getattr(reference, "pt2_local", None) is None:
                raise ValueError(
                    "Neural XC training with include_pt2_channel=True requires local PT2 "
                    "features. Build the reference with compute_local_pt2_features=True."
                )
        target = (
            getattr(reference, "mf_energy", None)
            if target_total_energy is None
            else target_total_energy
        )
        if target is None:
            raise ValueError(
                "target_total_energy must be provided when reference.mf_energy is missing."
            )
        return cls(
            molecule=reference,
            target_total_energy=target,
            **kwargs,
        )

    @classmethod
    def from_parts(
        cls,
        molecule: Any,
        *,
        core: GroundStateCoreDatum,
        excited_state: ExcitedStateDatum | None = None,
        weight: float = 1.0,
    ) -> "GroundStateDatum":
        extension = ExcitedStateDatum() if excited_state is None else excited_state
        return cls(
            molecule=molecule,
            target_total_energy=core.target_total_energy,
            target_density_matrix=core.target_density_matrix,
            target_xc_potential=core.target_xc_potential,
            target_xc_kernel=core.target_xc_kernel,
            weight=weight,
            density_constraint_weight=core.density_constraint_weight,
            xc_potential_constraint_weight=core.xc_potential_constraint_weight,
            xc_kernel_constraint_weight=core.xc_kernel_constraint_weight,
            xc_kernel_normalization_scale=core.xc_kernel_normalization_scale,
            stationarity_constraint_weight=core.stationarity_constraint_weight,
            dm21_scf_regularization_weight=core.dm21_scf_regularization_weight,
            target_orbital_energies=core.target_orbital_energies,
            target_orbital_occupations=core.target_orbital_occupations,
            orbital_energy_constraint_weight=core.orbital_energy_constraint_weight,
            orbital_energy_constraint_window=core.orbital_energy_constraint_window,
            janak_frontier_constraint_weight=core.janak_frontier_constraint_weight,
            target_s1_energy=extension.target_s1_energy,
            target_first_excited_total_energy=extension.target_first_excited_total_energy,
            target_excitation_energies=extension.target_excitation_energies,
            target_oscillator_strengths=extension.target_oscillator_strengths,
            target_spectrum_grid_ev=extension.target_spectrum_grid_ev,
            target_spectrum_curve=extension.target_spectrum_curve,
            s1_constraint_weight=extension.s1_constraint_weight,
            first_excited_total_energy_constraint_weight=(
                extension.first_excited_total_energy_constraint_weight
            ),
            excitation_constraint_weight=extension.excitation_constraint_weight,
            excitation_constraint_nstates=extension.excitation_constraint_nstates,
            oscillator_strength_constraint_weight=(
                extension.oscillator_strength_constraint_weight
            ),
            oscillator_strength_constraint_nstates=(
                extension.oscillator_strength_constraint_nstates
            ),
            spectrum_constraint_weight=extension.spectrum_constraint_weight,
            spectrum_constraint_nstates=extension.spectrum_constraint_nstates,
        )

    def ground_state_core(self) -> GroundStateCoreDatum:
        return GroundStateCoreDatum(
            target_total_energy=self.target_total_energy,
            target_density_matrix=self.target_density_matrix,
            target_xc_potential=self.target_xc_potential,
            target_xc_kernel=self.target_xc_kernel,
            density_constraint_weight=self.density_constraint_weight,
            xc_potential_constraint_weight=self.xc_potential_constraint_weight,
            xc_kernel_constraint_weight=self.xc_kernel_constraint_weight,
            xc_kernel_normalization_scale=self.xc_kernel_normalization_scale,
            stationarity_constraint_weight=self.stationarity_constraint_weight,
            dm21_scf_regularization_weight=self.dm21_scf_regularization_weight,
            target_orbital_energies=self.target_orbital_energies,
            target_orbital_occupations=self.target_orbital_occupations,
            orbital_energy_constraint_weight=self.orbital_energy_constraint_weight,
            orbital_energy_constraint_window=self.orbital_energy_constraint_window,
            janak_frontier_constraint_weight=self.janak_frontier_constraint_weight,
        )

    def excited_state_extension(self) -> ExcitedStateDatum:
        return ExcitedStateDatum(
            target_s1_energy=self.target_s1_energy,
            target_first_excited_total_energy=self.target_first_excited_total_energy,
            target_excitation_energies=self.target_excitation_energies,
            target_oscillator_strengths=self.target_oscillator_strengths,
            target_spectrum_grid_ev=self.target_spectrum_grid_ev,
            target_spectrum_curve=self.target_spectrum_curve,
            s1_constraint_weight=self.s1_constraint_weight,
            first_excited_total_energy_constraint_weight=(
                self.first_excited_total_energy_constraint_weight
            ),
            excitation_constraint_weight=self.excitation_constraint_weight,
            excitation_constraint_nstates=self.excitation_constraint_nstates,
            oscillator_strength_constraint_weight=self.oscillator_strength_constraint_weight,
            oscillator_strength_constraint_nstates=self.oscillator_strength_constraint_nstates,
            spectrum_constraint_weight=self.spectrum_constraint_weight,
            spectrum_constraint_nstates=self.spectrum_constraint_nstates,
        )


@dataclass(frozen=True)
class GroundStateCoreTrainingConfig:
    """Ground-state training configuration for the GradDFT core."""

    mode: Literal["fixed_density", "self_consistent"] = "fixed_density"
    energy_mse_weight: float = 0.0
    energy_mae_weight: float = 1.0
    energy_normalization: Literal["none", "per_electron", "per_atom"] = "none"
    energy_normalization_eps: float = 1e-8
    density_supervision: Literal["spin_summed", "spin_resolved"] = "spin_summed"
    self_consistent_energy_weight: float = 0.0
    orbital_energy_mse_weight: float = 0.0
    orbital_energy_mae_weight: float = 1.0
    coefficient_prior_weight: float = 0.0
    coefficient_prior_values: tuple[float, ...] | None = None
    coefficient_prior_mode: Literal["pointwise", "mean"] = "pointwise"
    fractional_linearity_weight: float = 0.0
    fractional_linearity_delta: float = 0.1
    janak_frontier_mode: Literal[
        "finite_difference",
        "autodiff",
        "full_scf_ad",
        "fixed_orbital_ad",
        "half_charge_ad",
    ] = "finite_difference"
    janak_frontier_delta: float = 0.1
    fractional_branch_rms_soft_threshold: float | None = 1.0
    occupation_tolerance: float = 1e-8
    dm21_scf_gap_floor: float = 1e-3
    scf_max_cycle: int = 12
    scf_damping: float = 0.25
    scf_level_shift: float = 0.0
    scf_conv_tol_density: float = 1e-8
    scf_orthogonalization_eps: float = 1e-10
    scf_eigenvalue_jitter: float = 1e-8
    scf_vxc_clip: float = 20.0
    scf_iterate_selection: Literal["final", "best_rms", "first_converged"] = "final"
    fractional_branch_scf_max_cycle: int | None = None
    fractional_branch_scf_damping: float | None = None
    fractional_branch_scf_level_shift: float | None = None
    fractional_branch_scf_iterate_selection: (
        Literal["final", "best_rms", "first_converged"] | None
    ) = None
    scf_require_convergence: bool = False
    scf_stop_gradient_on_unconverged: bool = False
    scf_stop_gradient_rms_threshold: float | None = None
    scf_gradient_mode: Literal["unrolled", "implicit_commutator"] = "unrolled"
    scf_implicit_forward_mode: Literal["unrolled", "input_state"] = "unrolled"
    scf_implicit_diff_max_iter: int = 24
    scf_implicit_diff_step_size: float = 0.2
    scf_implicit_diff_clip: float = 1e4
    scf_implicit_diff_solver: Literal["normal_cg", "gmres", "bicgstab"] = "normal_cg"
    scf_implicit_diff_tolerance: float = 1e-6
    scf_implicit_diff_regularization: float = 1e-3
    scf_implicit_diff_restart: int = 12


@dataclass(frozen=True)
class ExcitedStateTrainingConfig:
    """Excited-state loss configuration layered on top of the ground-state core."""

    s1_constraint_use_tda: bool = False
    excitation_constraint_use_tda: bool = True
    excitation_mse_weight: float = 0.0
    excitation_mae_weight: float = 1.0
    oscillator_strength_constraint_use_tda: bool = True
    oscillator_strength_mse_weight: float = 0.0
    oscillator_strength_mae_weight: float = 1.0
    spectrum_constraint_use_tda: bool = True
    spectrum_constraint_eta_ev: float = 0.15
    spectrum_mse_weight: float = 0.0
    spectrum_mae_weight: float = 1.0


@dataclass(frozen=True)
class GroundStateTrainingConfig:
    """Flat compatibility wrapper around core DFT + excited-state TD settings."""

    mode: Literal["fixed_density", "self_consistent"] = "fixed_density"
    energy_mse_weight: float = 0.0
    energy_mae_weight: float = 1.0
    energy_normalization: Literal["none", "per_electron", "per_atom"] = "none"
    energy_normalization_eps: float = 1e-8
    density_supervision: Literal["spin_summed", "spin_resolved"] = "spin_summed"
    self_consistent_energy_weight: float = 0.0
    orbital_energy_mse_weight: float = 0.0
    orbital_energy_mae_weight: float = 1.0
    coefficient_prior_weight: float = 0.0
    coefficient_prior_values: tuple[float, ...] | None = None
    coefficient_prior_mode: Literal["pointwise", "mean"] = "pointwise"
    fractional_linearity_weight: float = 0.0
    fractional_linearity_delta: float = 0.1
    janak_frontier_mode: Literal[
        "finite_difference",
        "autodiff",
        "full_scf_ad",
        "fixed_orbital_ad",
        "half_charge_ad",
    ] = "finite_difference"
    janak_frontier_delta: float = 0.1
    fractional_branch_rms_soft_threshold: float | None = 1.0
    s1_constraint_use_tda: bool = False
    excitation_constraint_use_tda: bool = True
    excitation_mse_weight: float = 0.0
    excitation_mae_weight: float = 1.0
    oscillator_strength_constraint_use_tda: bool = True
    oscillator_strength_mse_weight: float = 0.0
    oscillator_strength_mae_weight: float = 1.0
    spectrum_constraint_use_tda: bool = True
    spectrum_constraint_eta_ev: float = 0.15
    spectrum_mse_weight: float = 0.0
    spectrum_mae_weight: float = 1.0
    occupation_tolerance: float = 1e-8
    dm21_scf_gap_floor: float = 1e-3
    scf_max_cycle: int = 12
    scf_damping: float = 0.25
    scf_level_shift: float = 0.0
    scf_conv_tol_density: float = 1e-8
    scf_orthogonalization_eps: float = 1e-10
    scf_eigenvalue_jitter: float = 1e-8
    scf_vxc_clip: float = 20.0
    scf_iterate_selection: Literal["final", "best_rms", "first_converged"] = "final"
    fractional_branch_scf_max_cycle: int | None = None
    fractional_branch_scf_damping: float | None = None
    fractional_branch_scf_level_shift: float | None = None
    fractional_branch_scf_iterate_selection: (
        Literal["final", "best_rms", "first_converged"] | None
    ) = None
    scf_require_convergence: bool = False
    scf_stop_gradient_on_unconverged: bool = False
    scf_stop_gradient_rms_threshold: float | None = None
    scf_gradient_mode: Literal["unrolled", "implicit_commutator"] = "unrolled"
    scf_implicit_forward_mode: Literal["unrolled", "input_state"] = "unrolled"
    scf_implicit_diff_max_iter: int = 24
    scf_implicit_diff_step_size: float = 0.2
    scf_implicit_diff_clip: float = 1e4
    scf_implicit_diff_solver: Literal["normal_cg", "gmres", "bicgstab"] = "normal_cg"
    scf_implicit_diff_tolerance: float = 1e-6
    scf_implicit_diff_regularization: float = 1e-3
    scf_implicit_diff_restart: int = 12

    @classmethod
    def from_parts(
        cls,
        *,
        core: GroundStateCoreTrainingConfig,
        excited_state: ExcitedStateTrainingConfig | None = None,
    ) -> "GroundStateTrainingConfig":
        extension = ExcitedStateTrainingConfig() if excited_state is None else excited_state
        return cls(
            mode=core.mode,
            energy_mse_weight=core.energy_mse_weight,
            energy_mae_weight=core.energy_mae_weight,
            energy_normalization=core.energy_normalization,
            energy_normalization_eps=core.energy_normalization_eps,
            density_supervision=core.density_supervision,
            self_consistent_energy_weight=core.self_consistent_energy_weight,
            orbital_energy_mse_weight=core.orbital_energy_mse_weight,
            orbital_energy_mae_weight=core.orbital_energy_mae_weight,
            coefficient_prior_weight=core.coefficient_prior_weight,
            coefficient_prior_values=core.coefficient_prior_values,
            coefficient_prior_mode=core.coefficient_prior_mode,
            fractional_linearity_weight=core.fractional_linearity_weight,
            fractional_linearity_delta=core.fractional_linearity_delta,
            janak_frontier_mode=core.janak_frontier_mode,
            janak_frontier_delta=core.janak_frontier_delta,
            fractional_branch_rms_soft_threshold=core.fractional_branch_rms_soft_threshold,
            s1_constraint_use_tda=extension.s1_constraint_use_tda,
            excitation_constraint_use_tda=extension.excitation_constraint_use_tda,
            excitation_mse_weight=extension.excitation_mse_weight,
            excitation_mae_weight=extension.excitation_mae_weight,
            oscillator_strength_constraint_use_tda=(
                extension.oscillator_strength_constraint_use_tda
            ),
            oscillator_strength_mse_weight=extension.oscillator_strength_mse_weight,
            oscillator_strength_mae_weight=extension.oscillator_strength_mae_weight,
            spectrum_constraint_use_tda=extension.spectrum_constraint_use_tda,
            spectrum_constraint_eta_ev=extension.spectrum_constraint_eta_ev,
            spectrum_mse_weight=extension.spectrum_mse_weight,
            spectrum_mae_weight=extension.spectrum_mae_weight,
            occupation_tolerance=core.occupation_tolerance,
            dm21_scf_gap_floor=core.dm21_scf_gap_floor,
            scf_max_cycle=core.scf_max_cycle,
            scf_damping=core.scf_damping,
            scf_level_shift=core.scf_level_shift,
            scf_conv_tol_density=core.scf_conv_tol_density,
            scf_orthogonalization_eps=core.scf_orthogonalization_eps,
            scf_eigenvalue_jitter=core.scf_eigenvalue_jitter,
            scf_vxc_clip=core.scf_vxc_clip,
            scf_iterate_selection=core.scf_iterate_selection,
            fractional_branch_scf_max_cycle=core.fractional_branch_scf_max_cycle,
            fractional_branch_scf_damping=core.fractional_branch_scf_damping,
            fractional_branch_scf_level_shift=core.fractional_branch_scf_level_shift,
            fractional_branch_scf_iterate_selection=(
                core.fractional_branch_scf_iterate_selection
            ),
            scf_require_convergence=core.scf_require_convergence,
            scf_stop_gradient_on_unconverged=core.scf_stop_gradient_on_unconverged,
            scf_stop_gradient_rms_threshold=core.scf_stop_gradient_rms_threshold,
            scf_gradient_mode=core.scf_gradient_mode,
            scf_implicit_forward_mode=core.scf_implicit_forward_mode,
            scf_implicit_diff_max_iter=core.scf_implicit_diff_max_iter,
            scf_implicit_diff_step_size=core.scf_implicit_diff_step_size,
            scf_implicit_diff_clip=core.scf_implicit_diff_clip,
            scf_implicit_diff_solver=core.scf_implicit_diff_solver,
            scf_implicit_diff_tolerance=core.scf_implicit_diff_tolerance,
            scf_implicit_diff_regularization=core.scf_implicit_diff_regularization,
            scf_implicit_diff_restart=core.scf_implicit_diff_restart,
        )

    def ground_state_core_config(self) -> GroundStateCoreTrainingConfig:
        return GroundStateCoreTrainingConfig(
            mode=self.mode,
            energy_mse_weight=self.energy_mse_weight,
            energy_mae_weight=self.energy_mae_weight,
            energy_normalization=self.energy_normalization,
            energy_normalization_eps=self.energy_normalization_eps,
            density_supervision=self.density_supervision,
            self_consistent_energy_weight=self.self_consistent_energy_weight,
            orbital_energy_mse_weight=self.orbital_energy_mse_weight,
            orbital_energy_mae_weight=self.orbital_energy_mae_weight,
            coefficient_prior_weight=self.coefficient_prior_weight,
            coefficient_prior_values=self.coefficient_prior_values,
            coefficient_prior_mode=self.coefficient_prior_mode,
            fractional_linearity_weight=self.fractional_linearity_weight,
            fractional_linearity_delta=self.fractional_linearity_delta,
            janak_frontier_mode=self.janak_frontier_mode,
            janak_frontier_delta=self.janak_frontier_delta,
            fractional_branch_rms_soft_threshold=self.fractional_branch_rms_soft_threshold,
            occupation_tolerance=self.occupation_tolerance,
            dm21_scf_gap_floor=self.dm21_scf_gap_floor,
            scf_max_cycle=self.scf_max_cycle,
            scf_damping=self.scf_damping,
            scf_level_shift=self.scf_level_shift,
            scf_conv_tol_density=self.scf_conv_tol_density,
            scf_orthogonalization_eps=self.scf_orthogonalization_eps,
            scf_eigenvalue_jitter=self.scf_eigenvalue_jitter,
            scf_vxc_clip=self.scf_vxc_clip,
            scf_iterate_selection=self.scf_iterate_selection,
            fractional_branch_scf_max_cycle=self.fractional_branch_scf_max_cycle,
            fractional_branch_scf_damping=self.fractional_branch_scf_damping,
            fractional_branch_scf_level_shift=self.fractional_branch_scf_level_shift,
            fractional_branch_scf_iterate_selection=(
                self.fractional_branch_scf_iterate_selection
            ),
            scf_require_convergence=self.scf_require_convergence,
            scf_stop_gradient_on_unconverged=self.scf_stop_gradient_on_unconverged,
            scf_stop_gradient_rms_threshold=self.scf_stop_gradient_rms_threshold,
            scf_gradient_mode=self.scf_gradient_mode,
            scf_implicit_forward_mode=self.scf_implicit_forward_mode,
            scf_implicit_diff_max_iter=self.scf_implicit_diff_max_iter,
            scf_implicit_diff_step_size=self.scf_implicit_diff_step_size,
            scf_implicit_diff_clip=self.scf_implicit_diff_clip,
            scf_implicit_diff_solver=self.scf_implicit_diff_solver,
            scf_implicit_diff_tolerance=self.scf_implicit_diff_tolerance,
            scf_implicit_diff_regularization=self.scf_implicit_diff_regularization,
            scf_implicit_diff_restart=self.scf_implicit_diff_restart,
        )

    def excited_state_training_config(self) -> ExcitedStateTrainingConfig:
        return ExcitedStateTrainingConfig(
            s1_constraint_use_tda=self.s1_constraint_use_tda,
            excitation_constraint_use_tda=self.excitation_constraint_use_tda,
            excitation_mse_weight=self.excitation_mse_weight,
            excitation_mae_weight=self.excitation_mae_weight,
            oscillator_strength_constraint_use_tda=self.oscillator_strength_constraint_use_tda,
            oscillator_strength_mse_weight=self.oscillator_strength_mse_weight,
            oscillator_strength_mae_weight=self.oscillator_strength_mae_weight,
            spectrum_constraint_use_tda=self.spectrum_constraint_use_tda,
            spectrum_constraint_eta_ev=self.spectrum_constraint_eta_ev,
            spectrum_mse_weight=self.spectrum_mse_weight,
            spectrum_mae_weight=self.spectrum_mae_weight,
        )
