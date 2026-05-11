from __future__ import annotations

import copy
from dataclasses import dataclass, is_dataclass, replace
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree


def _molecule_from_data(data: Any) -> Any:
    return data.molecule if hasattr(data, "molecule") else data


def _resolve_rsh_parameters(functional: Any, params: PyTree, molecule: Any) -> Any | None:
    resolver = getattr(functional, "resolve_parameters", None)
    if resolver is None:
        return None
    try:
        return resolver(params, molecule)
    except TypeError:
        return resolver(params)


def _default_rsh_prior_penalty(functional: Any, resolved: Any) -> Array:
    template = getattr(functional, "template", None)
    if template is None or resolved is None:
        return jnp.asarray(0.0, dtype=jnp.float32)

    sr = jnp.asarray(resolved.sr_hf_fraction, dtype=jnp.float32)
    lr = jnp.asarray(resolved.lr_hf_fraction, dtype=jnp.float32)
    omega = jnp.asarray(resolved.omega, dtype=jnp.float32)
    sr_scale = max(float(template.sr_hf_bounds[1] - template.sr_hf_bounds[0]), 1e-6)
    lr_scale = max(float(template.lr_hf_bounds[1] - template.lr_hf_bounds[0]), 1e-6)
    omega_scale = max(float(template.omega_bounds[1] - template.omega_bounds[0]), 1e-6)
    sr_term = ((sr - float(template.default_sr_hf_fraction)) / sr_scale) ** 2
    lr_term = ((lr - float(template.default_lr_hf_fraction)) / lr_scale) ** 2
    omega_term = ((omega - float(template.default_omega)) / omega_scale) ** 2
    return (sr_term + lr_term + omega_term) / 3.0


def _lumo_energy_from_spin_orbitals(
    mo_energy_spin: Array,
    mo_occ_spin: Array,
    *,
    occupation_tolerance: float = 1e-8,
) -> Array:
    vir_mask = jnp.asarray(mo_occ_spin) <= occupation_tolerance
    energies = jnp.asarray(mo_energy_spin)
    masked = jnp.where(vir_mask, energies, jnp.asarray(1.0e6, dtype=energies.dtype))
    return jnp.min(masked)


@dataclass(frozen=True)
class NeutralFrontierIPEAResiduals:
    ip: Array
    ea: Array
    gap: Array


def _neutral_frontier_ip_ea_residuals(
    *,
    neutral_homo: Array,
    neutral_lumo: Array,
    neutral_energy: Array,
    cation_energy: Array,
    anion_energy: Array,
) -> NeutralFrontierIPEAResiduals:
    """Return neutral-orbital Koopmans residuals.

    The literature frontier constraints are:
    - eps_HOMO(N) = -IP(N), with IP(N) = E(N-1) - E(N)
    - eps_LUMO(N) = -EA(N), with EA(N) = E(N) - E(N+1)
    """

    ip_residual = neutral_homo + cation_energy - neutral_energy
    ea_residual = neutral_lumo + neutral_energy - anion_energy
    gap_residual = (neutral_lumo - neutral_homo) - (
        cation_energy + anion_energy - 2.0 * neutral_energy
    )
    return NeutralFrontierIPEAResiduals(
        ip=ip_residual,
        ea=ea_residual,
        gap=gap_residual,
    )


def _bind_rsh_functional_to_molecule(
    functional: Any,
    params: PyTree,
    molecule: Any,
) -> Any:
    binder = getattr(functional, "bind_to_molecule", None)
    if callable(binder):
        return binder(params, molecule)
    scf_binder = getattr(functional, "bind_to_molecule_for_scf", None)
    if callable(scf_binder):
        return scf_binder(params, molecule)
    raise AttributeError("RSH functional must expose bind_to_molecule(...) for Koopmans loss.")


def _detach_bound_rsh_functional(bound: Any) -> Any:
    resolved = getattr(bound, "resolved_params", None)
    if resolved is None:
        return bound
    detached_resolved = type(resolved)(
        sr_hf_fraction=jax.lax.stop_gradient(jnp.asarray(resolved.sr_hf_fraction)),
        lr_hf_fraction=jax.lax.stop_gradient(jnp.asarray(resolved.lr_hf_fraction)),
        omega=jax.lax.stop_gradient(jnp.asarray(resolved.omega)),
    )
    if is_dataclass(bound):
        return replace(bound, resolved_params=detached_resolved)
    try:
        bound_out = type(bound)(**{**bound.__dict__, "resolved_params": detached_resolved})
    except Exception:
        bound_out = bound
        setattr(bound_out, "resolved_params", detached_resolved)
    return bound_out


def _detach_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        arr = jnp.asarray(value)
    except Exception:
        return value
    return jax.lax.stop_gradient(arr)


def _detach_molecule_for_charged_states(molecule: Any) -> Any:
    grid = getattr(molecule, "grid", None)
    grid_out = grid
    if grid is not None:
        grid_updates = {}
        for attr in ("weights", "coords", "points"):
            if hasattr(grid, attr):
                value = getattr(grid, attr)
                if value is not None:
                    grid_updates[attr] = _detach_value(value)
        if grid_updates:
            if is_dataclass(grid):
                grid_out = replace(grid, **grid_updates)
            else:
                grid_out = copy.copy(grid)
                for key, value in grid_updates.items():
                    setattr(grid_out, key, value)

    updates = {}
    for attr in (
        "ao",
        "ao_deriv1",
        "ao_laplacian",
        "atom_coords",
        "atom_charges",
        "rep_tensor",
        "mo_coeff",
        "mo_occ",
        "mo_energy",
        "rdm1",
        "h1e",
        "overlap_matrix",
        "hfx_nu",
    ):
        if hasattr(molecule, attr):
            value = getattr(molecule, attr)
            if value is not None:
                updates[attr] = _detach_value(value)
    if hasattr(molecule, "nuclear_repulsion"):
        updates["nuclear_repulsion"] = getattr(molecule, "nuclear_repulsion")
    if hasattr(molecule, "hfx_omega_values"):
        updates["hfx_omega_values"] = getattr(molecule, "hfx_omega_values")
    if grid_out is not None:
        updates["grid"] = grid_out

    if is_dataclass(molecule):
        return replace(molecule, **updates)
    out = copy.copy(molecule)
    for key, value in updates.items():
        setattr(out, key, value)
    return out


def _charged_branch_training_config(base_cfg: Any, branch_config: Any | None) -> Any:
    if branch_config is None:
        return replace(
            base_cfg,
            mode="self_consistent",
            scf_max_cycle=max(int(base_cfg.scf_max_cycle), 8),
            scf_damping=max(float(base_cfg.scf_damping), 0.35),
            scf_level_shift=max(float(base_cfg.scf_level_shift), 0.5),
            scf_iterate_selection="final",
            scf_require_convergence=False,
        )
    return replace(
        base_cfg,
        mode="self_consistent",
        scf_max_cycle=int(getattr(branch_config, "max_cycle", base_cfg.scf_max_cycle)),
        scf_damping=float(getattr(branch_config, "damping", base_cfg.scf_damping)),
        scf_level_shift=float(getattr(branch_config, "level_shift", base_cfg.scf_level_shift)),
        scf_conv_tol_density=float(
            getattr(branch_config, "conv_tol_density", base_cfg.scf_conv_tol_density)
        ),
        scf_orthogonalization_eps=float(
            getattr(branch_config, "orthogonalization_eps", base_cfg.scf_orthogonalization_eps)
        ),
        scf_vxc_clip=(
            float(getattr(branch_config, "potential_clip"))
            if getattr(branch_config, "potential_clip", None) is not None
            else base_cfg.scf_vxc_clip
        ),
        scf_iterate_selection="final",
        scf_require_convergence=False,
    )


def make_self_supervised_rsh_loss(
    functional: Any,
    *,
    training_config: Any | None = None,
    janak_weight: float = 1.0,
    fractional_weight: float = 0.0,
    koopmans_ip_weight: float = 0.0,
    koopmans_ea_weight: float = 0.0,
    koopmans_lumo_ea_weight: float = 0.0,
    koopmans_loss_kind: str = "absolute",
    koopmans_detach_charged_states: bool = True,
    koopmans_differentiate_charged_orbitals: bool = False,
    koopmans_cation_config: Any | None = None,
    koopmans_anion_config: Any | None = None,
    long_range_correction_weight: float = 0.0,
    prior_weight: float = 1e-3,
    prior_penalty_fn: Callable[[Any, Any], Array] | None = None,
) -> Callable[[PyTree, Any], tuple[Array, dict[str, Array]]]:
    """Build a label-free self-consistent RSH objective.

    Koopmans/IP-EA terms currently use detached charged-state UKS branches.
    This keeps the training path stable until unrestricted differentiable SCF
    is implemented.
    """

    from ..training.config import GroundStateTrainingConfig
    from ..training.targets import (
        _homo_energy_from_spin_orbitals,
        _janak_frontier_penalty_by_mode,
        _FrozenFunctionalAdapter,
        _predict_ground_state_total_energy_from_molecule,
        _resolve_training_molecule_with_mode,
        _spin_resolved_orbital_blocks,
        charged_state_differentiable_scf_from_molecule,
        fractional_charge_linearity_penalty,
        koopmans_ip_ea_diagnostic,
    )

    cfg = GroundStateTrainingConfig() if training_config is None else training_config
    scf_cfg = cfg if cfg.mode == "self_consistent" else replace(cfg, mode="self_consistent")
    prior_fn = _default_rsh_prior_penalty if prior_penalty_fn is None else prior_penalty_fn
    if koopmans_loss_kind not in {"absolute", "squared"}:
        raise ValueError(
            "koopmans_loss_kind must be 'absolute' or 'squared', "
            f"got {koopmans_loss_kind!r}."
        )

    def loss(
        params: PyTree,
        bound_functional_or_data: Any,
        data: Any | None = None,
        *,
        training_config: Any | None = None,
        predictor: Any | None = None,
    ) -> tuple[Array, dict[str, Array]]:
        del predictor
        if data is None:
            active_functional = functional
            datum = bound_functional_or_data
        else:
            active_functional = bound_functional_or_data
            datum = data

        molecule = _molecule_from_data(datum)
        active_cfg = cfg if training_config is None else training_config
        active_scf_cfg = (
            active_cfg
            if active_cfg.mode == "self_consistent"
            else replace(active_cfg, mode="self_consistent")
        )
        use_koopmans = (
            koopmans_ip_weight != 0.0
            or koopmans_ea_weight != 0.0
            or koopmans_lumo_ea_weight != 0.0
        )
        scf_molecule = _resolve_training_molecule_with_mode(
            params,
            active_functional,
            molecule,
            active_scf_cfg,
        )
        endpoint_envelope = (
            use_koopmans
            and not koopmans_detach_charged_states
            and not koopmans_differentiate_charged_orbitals
        )
        parameter_molecule = (
            _detach_molecule_for_charged_states(scf_molecule)
            if endpoint_envelope
            else scf_molecule
        )
        if janak_weight != 0.0:
            janak_mse, janak_mae, janak_residual, janak_fd = _janak_frontier_penalty_by_mode(
                params,
                active_functional,
                scf_molecule,
                occupation_tolerance=active_scf_cfg.occupation_tolerance,
                training_config=active_scf_cfg,
                assume_self_consistent_input=True,
            )
        else:
            janak_mse = jnp.asarray(0.0, dtype=jnp.asarray(scf_molecule.h1e).dtype)
            janak_mae = jnp.asarray(0.0, dtype=janak_mse.dtype)
            janak_residual = jnp.zeros((2,), dtype=janak_mse.dtype)
            janak_fd = jnp.zeros((2,), dtype=janak_mse.dtype)
        if fractional_weight != 0.0:
            fractional = fractional_charge_linearity_penalty(
                params,
                active_functional,
                scf_molecule,
                delta=active_scf_cfg.fractional_linearity_delta,
                training_config=active_scf_cfg,
                assume_self_consistent_input=True,
            )
        else:
            fractional = jnp.asarray(0.0, dtype=janak_mae.dtype)

        if use_koopmans:
            bound = _bind_rsh_functional_to_molecule(active_functional, params, parameter_molecule)
            charged_bound = (
                _detach_bound_rsh_functional(bound)
                if koopmans_detach_charged_states
                else bound
            )
            charged_reference = (
                _detach_molecule_for_charged_states(scf_molecule)
                if koopmans_detach_charged_states
                else scf_molecule
            )
            _mo_coeff_spin, neutral_occ_spin, neutral_energy_spin = _spin_resolved_orbital_blocks(
                scf_molecule,
                occupation_tolerance=active_scf_cfg.occupation_tolerance,
            )
            neutral_homo = _homo_energy_from_spin_orbitals(
                neutral_energy_spin,
                neutral_occ_spin,
                occupation_tolerance=active_scf_cfg.occupation_tolerance,
            )
            neutral_lumo = _lumo_energy_from_spin_orbitals(
                neutral_energy_spin,
                neutral_occ_spin,
                occupation_tolerance=active_scf_cfg.occupation_tolerance,
            )
            if (
                not koopmans_detach_charged_states
                and not koopmans_differentiate_charged_orbitals
            ):
                neutral_homo = jax.lax.stop_gradient(neutral_homo)
                neutral_lumo = jax.lax.stop_gradient(neutral_lumo)
            if koopmans_detach_charged_states:
                neutral_energy = _predict_ground_state_total_energy_from_molecule(
                    params,
                    active_functional,
                    scf_molecule,
                )
                diagnostic = koopmans_ip_ea_diagnostic(
                    charged_reference,
                    charged_bound,
                    cation_config=koopmans_cation_config,
                    anion_config=koopmans_anion_config,
                    occupation_tolerance=active_scf_cfg.occupation_tolerance,
                )
                cation_energy = jax.lax.stop_gradient(
                    jnp.asarray(diagnostic.cation_energy, dtype=janak_mae.dtype)
                )
                anion_energy = jax.lax.stop_gradient(
                    jnp.asarray(diagnostic.anion_energy, dtype=janak_mae.dtype)
                )
                anion_homo = jax.lax.stop_gradient(
                    jnp.asarray(diagnostic.anion_homo_energy, dtype=janak_mae.dtype)
                )
            else:
                frozen_bound_functional = _FrozenFunctionalAdapter(bound)
                neutral_energy_molecule = (
                    parameter_molecule
                    if koopmans_differentiate_charged_orbitals
                    else parameter_molecule
                )
                neutral_energy = _predict_ground_state_total_energy_from_molecule(
                    None,
                    frozen_bound_functional,
                    neutral_energy_molecule,
                )
                cation_molecule, _ = charged_state_differentiable_scf_from_molecule(
                    parameter_molecule,
                    bound,
                    charge_delta=1,
                    training_config=_charged_branch_training_config(
                        active_scf_cfg,
                        koopmans_cation_config,
                    ),
                    occupation_tolerance=active_scf_cfg.occupation_tolerance,
                )
                anion_molecule, _ = charged_state_differentiable_scf_from_molecule(
                    parameter_molecule,
                    bound,
                    charge_delta=-1,
                    training_config=_charged_branch_training_config(
                        active_scf_cfg,
                        koopmans_anion_config,
                    ),
                    occupation_tolerance=active_scf_cfg.occupation_tolerance,
                )
                if not koopmans_differentiate_charged_orbitals:
                    cation_molecule = _detach_molecule_for_charged_states(cation_molecule)
                    anion_molecule = _detach_molecule_for_charged_states(anion_molecule)
                cation_energy = _predict_ground_state_total_energy_from_molecule(
                    None,
                    frozen_bound_functional,
                    cation_molecule,
                )
                anion_energy = _predict_ground_state_total_energy_from_molecule(
                    None,
                    frozen_bound_functional,
                    anion_molecule,
                )
                _anion_coeff_spin, anion_occ_spin, anion_energy_spin = _spin_resolved_orbital_blocks(
                    anion_molecule,
                    occupation_tolerance=active_scf_cfg.occupation_tolerance,
                )
                anion_homo = _homo_energy_from_spin_orbitals(
                    anion_energy_spin,
                    anion_occ_spin,
                    occupation_tolerance=active_scf_cfg.occupation_tolerance,
                )
                if not koopmans_differentiate_charged_orbitals:
                    anion_homo = jax.lax.stop_gradient(anion_homo)
            frontier_residuals = _neutral_frontier_ip_ea_residuals(
                neutral_homo=neutral_homo,
                neutral_lumo=neutral_lumo,
                neutral_energy=neutral_energy,
                cation_energy=cation_energy,
                anion_energy=anion_energy,
            )
            koopmans_ip_residual = frontier_residuals.ip
            koopmans_ea_residual = anion_homo + neutral_energy - anion_energy
            koopmans_lumo_ea_residual = frontier_residuals.ea
            koopmans_gap_residual = frontier_residuals.gap
            koopmans_ip_mse = koopmans_ip_residual**2
            koopmans_ea_mse = koopmans_ea_residual**2
            koopmans_lumo_ea_mse = koopmans_lumo_ea_residual**2
            koopmans_gap_mse = koopmans_gap_residual**2
            koopmans_ip_mae = jnp.abs(koopmans_ip_residual)
            koopmans_ea_mae = jnp.abs(koopmans_ea_residual)
            koopmans_lumo_ea_mae = jnp.abs(koopmans_lumo_ea_residual)
            koopmans_gap_mae = jnp.abs(koopmans_gap_residual)
        else:
            neutral_energy = _predict_ground_state_total_energy_from_molecule(
                params,
                active_functional,
                scf_molecule,
            )
            cation_energy = jnp.asarray(0.0, dtype=janak_mae.dtype)
            anion_energy = jnp.asarray(0.0, dtype=janak_mae.dtype)
            anion_homo = jnp.asarray(0.0, dtype=janak_mae.dtype)
            neutral_lumo = jnp.asarray(0.0, dtype=janak_mae.dtype)
            koopmans_ip_residual = jnp.asarray(0.0, dtype=janak_mae.dtype)
            koopmans_ea_residual = jnp.asarray(0.0, dtype=janak_mae.dtype)
            koopmans_lumo_ea_residual = jnp.asarray(0.0, dtype=janak_mae.dtype)
            koopmans_gap_residual = jnp.asarray(0.0, dtype=janak_mae.dtype)
            koopmans_ip_mse = jnp.asarray(0.0, dtype=janak_mae.dtype)
            koopmans_ea_mse = jnp.asarray(0.0, dtype=janak_mae.dtype)
            koopmans_lumo_ea_mse = jnp.asarray(0.0, dtype=janak_mae.dtype)
            koopmans_gap_mse = jnp.asarray(0.0, dtype=janak_mae.dtype)
            koopmans_ip_mae = jnp.asarray(0.0, dtype=janak_mae.dtype)
            koopmans_ea_mae = jnp.asarray(0.0, dtype=janak_mae.dtype)
            koopmans_lumo_ea_mae = jnp.asarray(0.0, dtype=janak_mae.dtype)
            koopmans_gap_mae = jnp.asarray(0.0, dtype=janak_mae.dtype)

        resolved = _resolve_rsh_parameters(active_functional, params, parameter_molecule)
        if resolved is not None:
            long_range_correction_residual = (
                jnp.asarray(resolved.lr_hf_fraction, dtype=janak_mae.dtype) - 1.0
            )
            long_range_correction_mae = jnp.abs(long_range_correction_residual)
        else:
            long_range_correction_residual = jnp.asarray(0.0, dtype=janak_mae.dtype)
            long_range_correction_mae = jnp.asarray(0.0, dtype=janak_mae.dtype)
        prior = (
            jnp.asarray(prior_fn(active_functional, resolved), dtype=janak_mae.dtype)
            if prior_weight != 0.0
            else jnp.asarray(0.0, dtype=janak_mae.dtype)
        )
        total = (
            janak_weight * janak_mae
            + fractional_weight * fractional
            + koopmans_ip_weight
            * (koopmans_ip_mse if koopmans_loss_kind == "squared" else koopmans_ip_mae)
            + koopmans_ea_weight
            * (koopmans_ea_mse if koopmans_loss_kind == "squared" else koopmans_ea_mae)
            + koopmans_lumo_ea_weight
            * (
                koopmans_lumo_ea_mse
                if koopmans_loss_kind == "squared"
                else koopmans_lumo_ea_mae
            )
            + long_range_correction_weight * long_range_correction_mae
            + prior_weight * prior
        )

        sr = (
            jnp.asarray(resolved.sr_hf_fraction, dtype=janak_mae.dtype)
            if resolved is not None
            else jnp.asarray(0.0, dtype=janak_mae.dtype)
        )
        lr = (
            jnp.asarray(resolved.lr_hf_fraction, dtype=janak_mae.dtype)
            if resolved is not None
            else jnp.asarray(0.0, dtype=janak_mae.dtype)
        )
        omega = (
            jnp.asarray(resolved.omega, dtype=janak_mae.dtype)
            if resolved is not None
            else jnp.asarray(0.0, dtype=janak_mae.dtype)
        )
        metrics = {
            "loss": total,
            "janak_frontier_penalty": jnp.asarray([janak_weight * janak_mae], dtype=janak_mae.dtype),
            "janak_frontier_mse": jnp.asarray([janak_mse], dtype=janak_mae.dtype),
            "janak_frontier_mae": jnp.asarray([janak_mae], dtype=janak_mae.dtype),
            "fractional_linearity_penalty": jnp.asarray(
                [fractional_weight * fractional],
                dtype=janak_mae.dtype,
            ),
            "fractional_linearity_raw": jnp.asarray([fractional], dtype=janak_mae.dtype),
            "koopmans_ip_penalty": jnp.asarray(
                [
                    koopmans_ip_weight
                    * (
                        koopmans_ip_mse
                        if koopmans_loss_kind == "squared"
                        else koopmans_ip_mae
                    )
                ],
                dtype=janak_mae.dtype,
            ),
            "koopmans_ip_mse": jnp.asarray([koopmans_ip_mse], dtype=janak_mae.dtype),
            "koopmans_ip_mae": jnp.asarray([koopmans_ip_mae], dtype=janak_mae.dtype),
            "koopmans_ip_residual": jnp.asarray([koopmans_ip_residual], dtype=janak_mae.dtype),
            "koopmans_ea_penalty": jnp.asarray(
                [
                    koopmans_ea_weight
                    * (
                        koopmans_ea_mse
                        if koopmans_loss_kind == "squared"
                        else koopmans_ea_mae
                    )
                ],
                dtype=janak_mae.dtype,
            ),
            "koopmans_ea_mse": jnp.asarray([koopmans_ea_mse], dtype=janak_mae.dtype),
            "koopmans_ea_mae": jnp.asarray([koopmans_ea_mae], dtype=janak_mae.dtype),
            "koopmans_ea_residual": jnp.asarray([koopmans_ea_residual], dtype=janak_mae.dtype),
            "koopmans_lumo_ea_penalty": jnp.asarray(
                [
                    koopmans_lumo_ea_weight
                    * (
                        koopmans_lumo_ea_mse
                        if koopmans_loss_kind == "squared"
                        else koopmans_lumo_ea_mae
                    )
                ],
                dtype=janak_mae.dtype,
            ),
            "koopmans_lumo_ea_mse": jnp.asarray([koopmans_lumo_ea_mse], dtype=janak_mae.dtype),
            "koopmans_lumo_ea_mae": jnp.asarray([koopmans_lumo_ea_mae], dtype=janak_mae.dtype),
            "koopmans_lumo_ea_residual": jnp.asarray(
                [koopmans_lumo_ea_residual],
                dtype=janak_mae.dtype,
            ),
            "koopmans_gap_mse": jnp.asarray([koopmans_gap_mse], dtype=janak_mae.dtype),
            "koopmans_gap_mae": jnp.asarray([koopmans_gap_mae], dtype=janak_mae.dtype),
            "koopmans_gap_residual": jnp.asarray(
                [koopmans_gap_residual],
                dtype=janak_mae.dtype,
            ),
            "koopmans_neutral_energy": jnp.asarray([neutral_energy], dtype=janak_mae.dtype),
            "koopmans_cation_energy": jnp.asarray([cation_energy], dtype=janak_mae.dtype),
            "koopmans_anion_energy": jnp.asarray([anion_energy], dtype=janak_mae.dtype),
            "koopmans_anion_homo": jnp.asarray([anion_homo], dtype=janak_mae.dtype),
            "koopmans_neutral_lumo": jnp.asarray([neutral_lumo], dtype=janak_mae.dtype),
            "long_range_correction_penalty": jnp.asarray(
                [long_range_correction_weight * long_range_correction_mae],
                dtype=janak_mae.dtype,
            ),
            "long_range_correction_mae": jnp.asarray(
                [long_range_correction_mae],
                dtype=janak_mae.dtype,
            ),
            "long_range_correction_residual": jnp.asarray(
                [long_range_correction_residual],
                dtype=janak_mae.dtype,
            ),
            "rsh_prior_penalty": jnp.asarray([prior_weight * prior], dtype=janak_mae.dtype),
            "sr_hf_fraction": jnp.asarray([sr], dtype=janak_mae.dtype),
            "lr_hf_fraction": jnp.asarray([lr], dtype=janak_mae.dtype),
            "omega": jnp.asarray([omega], dtype=janak_mae.dtype),
            "janak_residual_homo": jnp.asarray([janak_residual[0]], dtype=janak_mae.dtype),
            "janak_residual_lumo": jnp.asarray([janak_residual[1]], dtype=janak_mae.dtype),
            "janak_fd_homo": jnp.asarray([janak_fd[0]], dtype=janak_mae.dtype),
            "janak_fd_lumo": jnp.asarray([janak_fd[1]], dtype=janak_mae.dtype),
        }
        return total, metrics

    return loss


__all__ = ["make_self_supervised_rsh_loss"]
