from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array


def _pytree_dataclass(*, static_fields: tuple[str, ...] = ()):
    static_field_names = frozenset(static_fields)

    def decorator(cls):
        all_field_names = tuple(field.name for field in fields(cls))
        dynamic_names = tuple(
            name for name in all_field_names if name not in static_field_names
        )
        static_names = tuple(
            name for name in all_field_names if name in static_field_names
        )

        def tree_flatten(self):
            return (
                tuple(getattr(self, name) for name in dynamic_names),
                tuple(getattr(self, name) for name in static_names),
            )

        @classmethod
        def tree_unflatten(cls_, aux_data, children):
            values = dict(zip(dynamic_names, children, strict=True))
            values.update(zip(static_names, aux_data, strict=True))
            return cls_(**values)

        cls.tree_flatten = tree_flatten
        cls.tree_unflatten = tree_unflatten
        return jax.tree_util.register_pytree_node_class(cls)

    return decorator


def _require_scalar(name: str, value: Array | None) -> None:
    if value is not None and jnp.asarray(value).ndim != 0:
        raise ValueError(f"{name} must be a scalar in hartree.")


def _require_vector(name: str, value: Array | None) -> None:
    if value is None:
        return
    array = jnp.asarray(value)
    if array.ndim != 1 or int(array.size) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")


@_pytree_dataclass(static_fields=("weight",))
@dataclass(frozen=True)
class MolecularTrainingDatum:
    """Physical reference targets for one molecular training example."""

    molecule: Any
    target_e0_total_h: Array | None = None
    target_grid_density: Array | None = None
    target_s1_total_h: Array | None = None
    target_excitation_gaps_h: Array | None = None
    target_oscillator_strengths: Array | None = None
    target_spectrum_grid_ev: Array | None = None
    target_spectrum_curve: Array | None = None
    target_xc_potential: Array | None = None
    target_xc_kernel: Array | None = None
    target_xc_kernel_normalization_scale: float | None = None
    target_orbital_energies: Array | None = None
    target_orbital_occupations: Array | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        _require_scalar("target_e0_total_h", self.target_e0_total_h)
        _require_scalar("target_s1_total_h", self.target_s1_total_h)
        _require_vector("target_excitation_gaps_h", self.target_excitation_gaps_h)
        _require_vector("target_oscillator_strengths", self.target_oscillator_strengths)
        _require_vector("target_spectrum_grid_ev", self.target_spectrum_grid_ev)
        _require_vector("target_spectrum_curve", self.target_spectrum_curve)
        if (
            self.target_spectrum_grid_ev is not None
            and self.target_spectrum_curve is not None
            and jnp.asarray(self.target_spectrum_grid_ev).shape
            != jnp.asarray(self.target_spectrum_curve).shape
        ):
            raise ValueError(
                "target_spectrum_grid_ev and target_spectrum_curve must have equal shape."
            )
        if float(self.weight) < 0.0:
            raise ValueError("weight must be non-negative.")


@dataclass(frozen=True)
class MolecularTrainingConfig:
    """Loss weights and numerical settings for molecular training."""

    mode: Literal["fixed_density", "self_consistent"] = "fixed_density"
    e0_total_mse_weight: float = 0.0
    e0_total_mae_weight: float = 0.0
    e0_normalization: Literal["none", "per_electron", "per_atom"] = "none"
    e0_normalization_eps: float = 1e-8
    grid_density_mse_weight: float = 0.0
    xc_potential_mse_weight: float = 0.0
    xc_kernel_mse_weight: float = 0.0
    dm21_scf_regularization_weight: float = 0.0
    self_consistent_e0_weight: float = 0.0
    orbital_energy_mse_weight: float = 0.0
    orbital_energy_mae_weight: float = 0.0
    orbital_energy_window: int | None = None
    coefficient_prior_weight: float = 0.0
    coefficient_prior_values: tuple[float, ...] | None = None
    coefficient_prior_mode: Literal["pointwise", "mean"] = "pointwise"
    fractional_linearity_weight: float = 0.0
    fractional_linearity_delta: float = 0.1
    fractional_branch_rms_soft_threshold: float | None = 1.0
    s1_total_mse_weight: float = 0.0
    s1_total_mae_weight: float = 0.0
    excitation_gap_mse_weight: float = 0.0
    excitation_gap_mae_weight: float = 0.0
    excitation_gap_nstates: int | None = None
    oscillator_strength_mse_weight: float = 0.0
    oscillator_strength_mae_weight: float = 0.0
    oscillator_strength_nstates: int | None = None
    spectrum_mse_weight: float = 0.0
    spectrum_mae_weight: float = 0.0
    spectrum_nstates: int | None = None
    spectrum_eta_ev: float = 0.15
    excited_state_solver: Literal["tda", "casida"] = "tda"
    tda_gradient_mode: Literal["eigenvalue_only", "implicit_eigenvector"] = (
        "eigenvalue_only"
    )
    tda_eigenvector_adjoint_tolerance: float = 1e-6
    tda_eigenvector_adjoint_max_iter: int = 64
    response_two_electron_mode: Literal["auto", "direct", "df", "ris"] = "auto"
    response_ris_theta: float = 0.2
    response_ris_j_fit: Literal["s", "sp", "spd"] = "sp"
    response_ris_k_fit: Literal["s", "sp", "spd"] = "s"
    response_ris_aux_chunk_size: int = 256
    occupation_tolerance: float = 1e-8
    dm21_scf_gap_floor: float = 1e-3
    scf_max_cycle: int = 12
    scf_damping: float = 0.25
    scf_level_shift: float = 0.0
    scf_conv_tol_energy: float | None = None
    scf_convergence_metric: Literal["energy_and_residual", "energy"] = (
        "energy_and_residual"
    )
    scf_conv_tol_density: float = 1e-8
    scf_orthogonalization_eps: float = 1e-10
    scf_eigenvalue_jitter: float = 1e-8
    scf_vxc_clip: float = 20.0
    scf_iterate_selection: Literal["final", "best_rms", "first_converged"] = (
        "final"
    )
    fractional_branch_scf_max_cycle: int | None = None
    fractional_branch_scf_damping: float | None = None
    fractional_branch_scf_level_shift: float | None = None
    fractional_branch_scf_iterate_selection: (
        Literal["final", "best_rms", "first_converged"] | None
    ) = None
    scf_gradient_mode: Literal["expl", "impl"] = "impl"
    scf_implicit_diff_max_iter: int = 6
    scf_implicit_diff_tolerance: float = 1e-6
    scf_implicit_diff_regularization: float = 0.0

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name.endswith("_weight") and float(getattr(self, field.name)) < 0.0:
                raise ValueError(f"{field.name} must be non-negative.")
        if (
            self.tda_gradient_mode == "implicit_eigenvector"
            and self.excited_state_solver != "tda"
        ):
            raise ValueError(
                "implicit_eigenvector gradients require excited_state_solver='tda'."
            )
        if self.scf_gradient_mode not in {"impl", "expl"}:
            raise ValueError("scf_gradient_mode must be 'impl' or 'expl'.")
        if self.mode not in {"fixed_density", "self_consistent"}:
            raise ValueError("mode must be 'fixed_density' or 'self_consistent'.")
        if self.e0_normalization not in {"none", "per_electron", "per_atom"}:
            raise ValueError(
                "e0_normalization must be 'none', 'per_electron', or 'per_atom'."
            )
        for name in (
            "excitation_gap_nstates",
            "oscillator_strength_nstates",
            "spectrum_nstates",
        ):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive when provided.")
