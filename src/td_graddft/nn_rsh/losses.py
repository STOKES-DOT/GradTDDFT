from __future__ import annotations

import copy
from dataclasses import dataclass, is_dataclass, replace
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree


_CACHED_KOOPMANS_DIAGNOSTIC_ATTR = "_td_graddft_cached_koopmans_ip_ea_diagnostic"
_FIXED_DENSITY_CATION_ATTR = "_td_graddft_fixed_density_cation_molecule"
_FIXED_DENSITY_ANION_ATTR = "_td_graddft_fixed_density_anion_molecule"


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FixedDensityRSHDatum:
    """Fixed-density RSH datum carrying precomputed neutral and charged states."""

    molecule: Any
    cation_molecule: Any | None = None
    anion_molecule: Any | None = None
    target_total_energy: Any | None = None
    weight: float = 1.0

    def tree_flatten(self):
        children = (
            self.molecule,
            self.cation_molecule,
            self.anion_molecule,
            self.target_total_energy,
        )
        return children, (float(self.weight),)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        molecule, cation_molecule, anion_molecule, target_total_energy = children
        (weight,) = aux_data
        return cls(
            molecule=molecule,
            cation_molecule=cation_molecule,
            anion_molecule=anion_molecule,
            target_total_energy=target_total_energy,
            weight=weight,
        )


def _molecule_from_data(data: Any) -> Any:
    return data.molecule if hasattr(data, "molecule") else data


def _copy_with_attrs(value: Any, **attrs: Any) -> Any:
    out = copy.copy(value)
    for key, attr_value in attrs.items():
        try:
            object.__setattr__(out, key, attr_value)
        except TypeError:
            setattr(out, key, attr_value)
    return out


def with_fixed_density_koopmans_states(
    molecule: Any,
    *,
    cation_molecule: Any,
    anion_molecule: Any,
) -> Any:
    """Attach fixed cation/anion states to a molecule-like object."""

    return _copy_with_attrs(
        molecule,
        **{
            _FIXED_DENSITY_CATION_ATTR: cation_molecule,
            _FIXED_DENSITY_ANION_ATTR: anion_molecule,
        },
    )


def _fixed_density_charge_state_from_data(data: Any, molecule: Any, name: str) -> Any:
    direct_attr = f"{name}_molecule"
    direct = getattr(data, direct_attr, None)
    if direct is not None:
        return direct
    cached_attr = (
        _FIXED_DENSITY_CATION_ATTR
        if name == "cation"
        else _FIXED_DENSITY_ANION_ATTR
    )
    cached = getattr(molecule, cached_attr, None)
    if cached is not None:
        return cached
    raise ValueError(
        "Fixed-density RSH Koopmans loss requires precomputed "
        f"{name}_molecule on the datum or molecule."
    )


def _cached_koopmans_diagnostic(molecule: Any) -> Any | None:
    return getattr(molecule, _CACHED_KOOPMANS_DIAGNOSTIC_ATTR, None)


def _with_cached_koopmans_diagnostic(molecule: Any, diagnostic: Any) -> Any:
    return _copy_with_attrs(molecule, **{_CACHED_KOOPMANS_DIAGNOSTIC_ATTR: diagnostic})


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


def _replace_molecule_fields(molecule: Any, **updates: Any) -> Any:
    if is_dataclass(molecule):
        return replace(molecule, **updates)
    out = copy.copy(molecule)
    for key, value in updates.items():
        setattr(out, key, value)
    return out


def _fixed_density_one_shot_spin_orbital_energies(
    params: PyTree,
    functional: Any,
    molecule: Any,
    *,
    occupation_tolerance: float,
    vxc_clip: float,
    orthogonalization_eps: float,
    eigenvalue_jitter: float,
) -> tuple[Array, Array, Array]:
    from ..scf.core import _diagonalize_fock, _orthogonalizer
    from ..scf.differentiable import (
        _build_vxc_matrix_from_components,
        _clip_grid_potential_components,
        _clip_hybrid_alpha,
        _coulomb_exchange_matrices,
        _repulsion_integrals_from_molecule,
        _safe_symmetric_matrix,
        _spin_resolved_density_matrix,
        _unrestricted_channel,
        _unrestricted_scf_xc_components,
    )

    h1e = jnp.asarray(molecule.h1e)
    dtype = h1e.dtype
    density_spin = jnp.asarray(_spin_resolved_density_matrix(molecule), dtype=dtype)
    mo_coeff_spin, mo_occ_spin, mo_energy_spin = _unrestricted_channel(molecule)
    mo_coeff_spin = jnp.asarray(mo_coeff_spin, dtype=dtype)
    mo_occ_spin = jnp.asarray(mo_occ_spin, dtype=dtype)
    mo_energy_spin = jnp.asarray(mo_energy_spin, dtype=dtype)
    molecule_iter = _replace_molecule_fields(
        molecule,
        rdm1=density_spin,
        mo_coeff=mo_coeff_spin,
        mo_occ=mo_occ_spin,
        mo_energy=mo_energy_spin,
    )

    density_total = density_spin.sum(axis=0)
    repulsion = _repulsion_integrals_from_molecule(molecule)
    j_mat, _ = _coulomb_exchange_matrices(repulsion, density_total)
    _, k_alpha = _coulomb_exchange_matrices(repulsion, density_spin[0])
    _, k_beta = _coulomb_exchange_matrices(repulsion, density_spin[1])
    (
        vxc_rho_a,
        vxc_rho_b,
        vxc_grad_a,
        vxc_grad_b,
        xc_kind,
        alpha,
        extra_fock_a,
        extra_fock_b,
    ) = _unrestricted_scf_xc_components(
        params,
        functional,
        molecule_iter,
        functional_dtype=dtype,
    )
    vxc_rho_a, vxc_grad_a = _clip_grid_potential_components(
        vxc_rho_a,
        vxc_grad_a,
        float(vxc_clip),
    )
    vxc_rho_b, vxc_grad_b = _clip_grid_potential_components(
        vxc_rho_b,
        vxc_grad_b,
        float(vxc_clip),
    )
    zero_a = jnp.zeros_like(vxc_rho_a)
    zero_b = jnp.zeros_like(vxc_rho_b)
    weights = jnp.asarray(molecule_iter.grid.weights, dtype=dtype)
    vxc_matrix_a = _build_vxc_matrix_from_components(
        molecule=molecule_iter,
        weights=weights,
        v_rho=vxc_rho_a,
        v_grad=vxc_grad_a,
        v_tau=zero_a,
        v_lapl=zero_a,
        xc_kind=xc_kind,
    )
    vxc_matrix_b = _build_vxc_matrix_from_components(
        molecule=molecule_iter,
        weights=weights,
        v_rho=vxc_rho_b,
        v_grad=vxc_grad_b,
        v_tau=zero_b,
        v_lapl=zero_b,
        xc_kind=xc_kind,
    )
    alpha = _clip_hybrid_alpha(jnp.asarray(alpha, dtype=dtype))
    fock_alpha = _safe_symmetric_matrix(
        h1e + j_mat - alpha * k_alpha + jnp.asarray(extra_fock_a, dtype=dtype) + vxc_matrix_a
    )
    fock_beta = _safe_symmetric_matrix(
        h1e + j_mat - alpha * k_beta + jnp.asarray(extra_fock_b, dtype=dtype) + vxc_matrix_b
    )
    overlap = getattr(molecule, "overlap_matrix", None)
    if overlap is None:
        overlap = jnp.eye(h1e.shape[0], dtype=dtype)
    x = _orthogonalizer(jnp.asarray(overlap, dtype=dtype), float(orthogonalization_eps))
    mo_energy_alpha, _ = _diagonalize_fock(
        fock_alpha,
        x,
        eigenvalue_jitter=float(eigenvalue_jitter),
    )
    mo_energy_beta, _ = _diagonalize_fock(
        fock_beta,
        x,
        eigenvalue_jitter=float(eigenvalue_jitter),
    )
    del occupation_tolerance
    return (
        jnp.stack([mo_energy_alpha, mo_energy_beta], axis=0),
        mo_occ_spin,
        jnp.stack([fock_alpha, fock_beta], axis=0),
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


def make_fixed_density_rsh_loss(
    functional: Any,
    *,
    training_config: Any | None = None,
    koopmans_ip_weight: float = 0.0,
    koopmans_ea_weight: float = 0.0,
    koopmans_lumo_ea_weight: float = 0.0,
    koopmans_loss_kind: str = "absolute",
    long_range_correction_weight: float = 0.0,
    prior_weight: float = 1e-3,
    prior_penalty_fn: Callable[[Any, Any], Array] | None = None,
) -> Callable[[PyTree, Any], tuple[Array, dict[str, Array]]]:
    """Build a frozen-density RSH Koopmans objective.

    The datum must carry precomputed neutral, cation, and anion density states.
    No SCF solve or runtime-forward provider is invoked in this loss.
    """

    from ..training.config import GroundStateTrainingConfig
    from ..training.targets import (
        _FrozenFunctionalAdapter,
        _homo_energy_from_spin_orbitals,
        _predict_ground_state_total_energy_from_molecule,
    )

    cfg = (
        replace(training_config, mode="fixed_density")
        if training_config is not None
        else GroundStateTrainingConfig(mode="fixed_density")
    )
    prior_fn = _default_rsh_prior_penalty if prior_penalty_fn is None else prior_penalty_fn
    if koopmans_loss_kind not in {"absolute", "squared"}:
        raise ValueError(
            "koopmans_loss_kind must be 'absolute' or 'squared', "
            f"got {koopmans_loss_kind!r}."
        )
    has_koopmans_terms = (
        koopmans_ip_weight != 0.0
        or koopmans_ea_weight != 0.0
        or koopmans_lumo_ea_weight != 0.0
    )

    def _frontiers(active_functional: Any, molecule: Any, active_cfg: Any):
        mo_energy_spin, mo_occ_spin, _fock_spin = _fixed_density_one_shot_spin_orbital_energies(
            None,
            active_functional,
            molecule,
            occupation_tolerance=active_cfg.occupation_tolerance,
            vxc_clip=active_cfg.scf_vxc_clip,
            orthogonalization_eps=active_cfg.scf_orthogonalization_eps,
            eigenvalue_jitter=active_cfg.scf_eigenvalue_jitter,
        )
        homo = _homo_energy_from_spin_orbitals(
            mo_energy_spin,
            mo_occ_spin,
            occupation_tolerance=active_cfg.occupation_tolerance,
        )
        lumo = _lumo_energy_from_spin_orbitals(
            mo_energy_spin,
            mo_occ_spin,
            occupation_tolerance=active_cfg.occupation_tolerance,
        )
        return homo, lumo

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
        active_cfg = (
            replace(training_config, mode="fixed_density")
            if training_config is not None
            else cfg
        )
        neutral_bound = _bind_rsh_functional_to_molecule(active_functional, params, molecule)
        neutral_functional = _FrozenFunctionalAdapter(neutral_bound)
        neutral_energy = _predict_ground_state_total_energy_from_molecule(
            None,
            neutral_functional,
            molecule,
        )
        dtype = jnp.asarray(neutral_energy).dtype
        neutral_homo, neutral_lumo = _frontiers(neutral_functional, molecule, active_cfg)

        if has_koopmans_terms:
            cation_molecule = _fixed_density_charge_state_from_data(datum, molecule, "cation")
            anion_molecule = _fixed_density_charge_state_from_data(datum, molecule, "anion")
            cation_bound = _bind_rsh_functional_to_molecule(
                active_functional,
                params,
                cation_molecule,
            )
            anion_bound = _bind_rsh_functional_to_molecule(
                active_functional,
                params,
                anion_molecule,
            )
            cation_functional = _FrozenFunctionalAdapter(cation_bound)
            anion_functional = _FrozenFunctionalAdapter(anion_bound)
            cation_energy = _predict_ground_state_total_energy_from_molecule(
                None,
                cation_functional,
                cation_molecule,
            )
            anion_energy = _predict_ground_state_total_energy_from_molecule(
                None,
                anion_functional,
                anion_molecule,
            )
            anion_homo, _anion_lumo = _frontiers(anion_functional, anion_molecule, active_cfg)
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
            cation_energy = jnp.asarray(0.0, dtype=dtype)
            anion_energy = jnp.asarray(0.0, dtype=dtype)
            anion_homo = jnp.asarray(0.0, dtype=dtype)
            koopmans_ip_residual = jnp.asarray(0.0, dtype=dtype)
            koopmans_ea_residual = jnp.asarray(0.0, dtype=dtype)
            koopmans_lumo_ea_residual = jnp.asarray(0.0, dtype=dtype)
            koopmans_gap_residual = jnp.asarray(0.0, dtype=dtype)
            koopmans_ip_mse = jnp.asarray(0.0, dtype=dtype)
            koopmans_ea_mse = jnp.asarray(0.0, dtype=dtype)
            koopmans_lumo_ea_mse = jnp.asarray(0.0, dtype=dtype)
            koopmans_gap_mse = jnp.asarray(0.0, dtype=dtype)
            koopmans_ip_mae = jnp.asarray(0.0, dtype=dtype)
            koopmans_ea_mae = jnp.asarray(0.0, dtype=dtype)
            koopmans_lumo_ea_mae = jnp.asarray(0.0, dtype=dtype)
            koopmans_gap_mae = jnp.asarray(0.0, dtype=dtype)

        resolved = _resolve_rsh_parameters(active_functional, params, molecule)
        if resolved is not None:
            long_range_correction_residual = (
                jnp.asarray(resolved.lr_hf_fraction, dtype=dtype) - 1.0
            )
            long_range_correction_mae = jnp.abs(long_range_correction_residual)
            sr = jnp.asarray(resolved.sr_hf_fraction, dtype=dtype)
            lr = jnp.asarray(resolved.lr_hf_fraction, dtype=dtype)
            omega = jnp.asarray(resolved.omega, dtype=dtype)
        else:
            long_range_correction_residual = jnp.asarray(0.0, dtype=dtype)
            long_range_correction_mae = jnp.asarray(0.0, dtype=dtype)
            sr = jnp.asarray(0.0, dtype=dtype)
            lr = jnp.asarray(0.0, dtype=dtype)
            omega = jnp.asarray(0.0, dtype=dtype)
        prior = (
            jnp.asarray(prior_fn(active_functional, resolved), dtype=dtype)
            if prior_weight != 0.0
            else jnp.asarray(0.0, dtype=dtype)
        )
        total = (
            koopmans_ip_weight
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
        metrics = {
            "loss": total,
            "fixed_density_rsh": jnp.asarray([1.0], dtype=dtype),
            "koopmans_ip_penalty": jnp.asarray(
                [
                    koopmans_ip_weight
                    * (
                        koopmans_ip_mse
                        if koopmans_loss_kind == "squared"
                        else koopmans_ip_mae
                    )
                ],
                dtype=dtype,
            ),
            "koopmans_ip_mse": jnp.asarray([koopmans_ip_mse], dtype=dtype),
            "koopmans_ip_mae": jnp.asarray([koopmans_ip_mae], dtype=dtype),
            "koopmans_ip_residual": jnp.asarray([koopmans_ip_residual], dtype=dtype),
            "koopmans_ea_penalty": jnp.asarray(
                [
                    koopmans_ea_weight
                    * (
                        koopmans_ea_mse
                        if koopmans_loss_kind == "squared"
                        else koopmans_ea_mae
                    )
                ],
                dtype=dtype,
            ),
            "koopmans_ea_mse": jnp.asarray([koopmans_ea_mse], dtype=dtype),
            "koopmans_ea_mae": jnp.asarray([koopmans_ea_mae], dtype=dtype),
            "koopmans_ea_residual": jnp.asarray([koopmans_ea_residual], dtype=dtype),
            "koopmans_lumo_ea_penalty": jnp.asarray(
                [
                    koopmans_lumo_ea_weight
                    * (
                        koopmans_lumo_ea_mse
                        if koopmans_loss_kind == "squared"
                        else koopmans_lumo_ea_mae
                    )
                ],
                dtype=dtype,
            ),
            "koopmans_lumo_ea_mse": jnp.asarray([koopmans_lumo_ea_mse], dtype=dtype),
            "koopmans_lumo_ea_mae": jnp.asarray([koopmans_lumo_ea_mae], dtype=dtype),
            "koopmans_lumo_ea_residual": jnp.asarray(
                [koopmans_lumo_ea_residual],
                dtype=dtype,
            ),
            "koopmans_gap_mse": jnp.asarray([koopmans_gap_mse], dtype=dtype),
            "koopmans_gap_mae": jnp.asarray([koopmans_gap_mae], dtype=dtype),
            "koopmans_gap_residual": jnp.asarray([koopmans_gap_residual], dtype=dtype),
            "koopmans_neutral_energy": jnp.asarray([neutral_energy], dtype=dtype),
            "koopmans_cation_energy": jnp.asarray([cation_energy], dtype=dtype),
            "koopmans_anion_energy": jnp.asarray([anion_energy], dtype=dtype),
            "koopmans_anion_homo": jnp.asarray([anion_homo], dtype=dtype),
            "koopmans_neutral_lumo": jnp.asarray([neutral_lumo], dtype=dtype),
            "long_range_correction_penalty": jnp.asarray(
                [long_range_correction_weight * long_range_correction_mae],
                dtype=dtype,
            ),
            "long_range_correction_mae": jnp.asarray([long_range_correction_mae], dtype=dtype),
            "long_range_correction_residual": jnp.asarray(
                [long_range_correction_residual],
                dtype=dtype,
            ),
            "rsh_prior_penalty": jnp.asarray([prior_weight * prior], dtype=dtype),
            "sr_hf_fraction": jnp.asarray([sr], dtype=dtype),
            "lr_hf_fraction": jnp.asarray([lr], dtype=dtype),
            "omega": jnp.asarray([omega], dtype=dtype),
        }
        return total, metrics

    return loss


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
    has_koopmans_terms = (
        koopmans_ip_weight != 0.0
        or koopmans_ea_weight != 0.0
        or koopmans_lumo_ea_weight != 0.0
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
        use_koopmans = has_koopmans_terms
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
            charged_diagnostic_cached = False
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
                diagnostic = _cached_koopmans_diagnostic(molecule)
                if diagnostic is None:
                    diagnostic = koopmans_ip_ea_diagnostic(
                        charged_reference,
                        charged_bound,
                        cation_config=koopmans_cation_config,
                        anion_config=koopmans_anion_config,
                        occupation_tolerance=active_scf_cfg.occupation_tolerance,
                    )
                else:
                    charged_diagnostic_cached = True
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
            charged_diagnostic_cached = False
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
            "koopmans_charged_cached": jnp.asarray(
                [1.0 if charged_diagnostic_cached else 0.0],
                dtype=janak_mae.dtype,
            ),
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

    def runtime_forward_state_provider(provider_training_config: Any | None = None):
        if not has_koopmans_terms or not koopmans_detach_charged_states:
            return None
        provider_cfg = cfg if provider_training_config is None else provider_training_config
        provider_scf_cfg = (
            provider_cfg
            if getattr(provider_cfg, "mode", None) == "self_consistent"
            else replace(provider_cfg, mode="self_consistent")
        )

        def provider(params: PyTree, active_functional: Any, molecule: Any) -> Any:
            frozen_params = jax.tree_util.tree_map(jax.lax.stop_gradient, params)
            bound = _bind_rsh_functional_to_molecule(
                active_functional,
                frozen_params,
                molecule,
            )
            charged_bound = _detach_bound_rsh_functional(bound)
            charged_reference = _detach_molecule_for_charged_states(molecule)
            diagnostic = koopmans_ip_ea_diagnostic(
                charged_reference,
                charged_bound,
                cation_config=koopmans_cation_config,
                anion_config=koopmans_anion_config,
                occupation_tolerance=provider_scf_cfg.occupation_tolerance,
            )
            return _with_cached_koopmans_diagnostic(molecule, diagnostic)

        return provider

    setattr(loss, "runtime_forward_state_provider", runtime_forward_state_provider)
    return loss


__all__ = [
    "FixedDensityRSHDatum",
    "make_fixed_density_rsh_loss",
    "make_self_supervised_rsh_loss",
    "with_fixed_density_koopmans_states",
]
