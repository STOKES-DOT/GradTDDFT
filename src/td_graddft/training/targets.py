from __future__ import annotations

import copy
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from ..features import grid_features_for_molecule
from .. import tdscf
from ..scf import (
    DifferentiableSCF,
    DifferentiableSCFConfig,
    UKSConfig,
    run_uks_from_integrals,
)
from ..data.integrals import build_j_from_eri_pair_matrix, build_jk_from_eri_pair_matrix
from ..df import build_j_from_df, build_jk_from_df
from ..scf.rks import _vxc_matrix_from_grid_potential
from ..spectra import HARTREE_TO_EV, lorentzian_spectrum, oscillator_strengths
from ..tddft.response_options import (
    ResponseKernelOptions,
    normalize_response_kernel_options,
)
from .config import MolecularTrainingConfig, MolecularTrainingDatum

_S1_SOLVER_NSTATES = 3


def _response_kernel_options_from_excited_config(
    config: MolecularTrainingConfig,
) -> ResponseKernelOptions:
    return normalize_response_kernel_options(
        {
            "two_electron_mode": config.response_two_electron_mode,
            "ris_theta": config.response_ris_theta,
            "ris_j_fit": config.response_ris_j_fit,
            "ris_k_fit": config.response_ris_k_fit,
            "ris_aux_chunk_size": config.response_ris_aux_chunk_size,
        }
    )


def _as_dataset(data: MolecularTrainingDatum | Sequence[MolecularTrainingDatum]) -> list[MolecularTrainingDatum]:
    if isinstance(data, MolecularTrainingDatum):
        return [data]
    return list(data)


def _excited_state_terms(
    *,
    predicted_e0_total_h: Array | None,
    datum: MolecularTrainingDatum,
    config: MolecularTrainingConfig,
    get_excited_state_observables: Callable[..., tuple[Array, Array]],
) -> dict[str, Array]:
    dtype_source = predicted_e0_total_h
    if dtype_source is None:
        dtype_source = (
            datum.target_excitation_gaps_h
            if datum.target_excitation_gaps_h is not None
            else datum.target_s1_total_h
        )
    dtype = _datum_dtype(datum) if dtype_source is None else jnp.asarray(dtype_source).dtype
    empty = jnp.array([], dtype=dtype)
    terms = {
        "s1_total_loss": jnp.asarray(0.0, dtype=dtype),
        "s1_total_mse": jnp.asarray(0.0, dtype=dtype),
        "s1_total_mae": jnp.asarray(0.0, dtype=dtype),
        "s1_total_predicted_h": empty,
        "s1_total_target_h": empty,
        "excitation_gap_loss": jnp.asarray(0.0, dtype=dtype),
        "excitation_gap_mse": jnp.asarray(0.0, dtype=dtype),
        "excitation_gap_mae": jnp.asarray(0.0, dtype=dtype),
        "excitation_gap_predicted_h": empty,
        "excitation_gap_target_h": empty,
        "oscillator_strength_loss": jnp.asarray(0.0, dtype=dtype),
        "oscillator_strength_mse": jnp.asarray(0.0, dtype=dtype),
        "oscillator_strength_mae": jnp.asarray(0.0, dtype=dtype),
        "oscillator_strength_predicted": empty,
        "oscillator_strength_target": empty,
        "spectrum_loss": jnp.asarray(0.0, dtype=dtype),
        "spectrum_mse": jnp.asarray(0.0, dtype=dtype),
        "spectrum_mae": jnp.asarray(0.0, dtype=dtype),
    }

    use_tda = config.excited_state_solver == "tda"
    s1_total_active = (
        config.s1_total_mse_weight != 0.0
        or config.s1_total_mae_weight != 0.0
    )
    if s1_total_active:
        if datum.target_s1_total_h is None:
            raise ValueError(
                "target_s1_total_h is required when an S1-total loss weight is nonzero."
            )
        if predicted_e0_total_h is None:
            raise ValueError("S1-total supervision requires a predicted E0 total energy.")
        s1_gap_h = get_excited_state_observables(
            _S1_SOLVER_NSTATES,
            use_tda,
        )[0][0]
        predicted_h = jnp.asarray(predicted_e0_total_h, dtype=dtype) + s1_gap_h
        target_h = jnp.asarray(datum.target_s1_total_h, dtype=dtype)
        residual = predicted_h - target_h
        terms["s1_total_predicted_h"] = jnp.atleast_1d(predicted_h)
        terms["s1_total_target_h"] = jnp.atleast_1d(target_h)
        terms["s1_total_mse"] = residual**2
        terms["s1_total_mae"] = jnp.abs(residual)
        terms["s1_total_loss"] = (
            config.s1_total_mse_weight * terms["s1_total_mse"]
            + config.s1_total_mae_weight * terms["s1_total_mae"]
        )

    excitation_gap_active = (
        config.excitation_gap_mse_weight != 0.0
        or config.excitation_gap_mae_weight != 0.0
    )
    if excitation_gap_active:
        if datum.target_excitation_gaps_h is None:
            raise ValueError(
                "target_excitation_gaps_h is required when an excitation-gap loss weight is nonzero."
            )
        target_h = jnp.asarray(datum.target_excitation_gaps_h, dtype=dtype)
        requested_nstates = (
            int(config.excitation_gap_nstates)
            if config.excitation_gap_nstates is not None
            else int(target_h.shape[0])
        )
        requested_nstates = max(1, requested_nstates)
        predicted_h, _ = get_excited_state_observables(
            requested_nstates,
            use_tda,
        )
        n_compare = min(
            int(predicted_h.shape[0]),
            int(target_h.shape[0]),
            requested_nstates,
        )
        if n_compare <= 0:
            raise ValueError("No comparable excitation gaps were produced.")
        predicted_h = jnp.asarray(predicted_h[:n_compare], dtype=dtype)
        target_h = target_h[:n_compare]
        residual = predicted_h - target_h
        terms["excitation_gap_predicted_h"] = predicted_h
        terms["excitation_gap_target_h"] = target_h
        terms["excitation_gap_mse"] = jnp.mean(residual**2)
        terms["excitation_gap_mae"] = jnp.mean(jnp.abs(residual))
        terms["excitation_gap_loss"] = (
            config.excitation_gap_mse_weight * terms["excitation_gap_mse"]
            + config.excitation_gap_mae_weight * terms["excitation_gap_mae"]
        )

    oscillator_active = (
        config.oscillator_strength_mse_weight != 0.0
        or config.oscillator_strength_mae_weight != 0.0
    )
    if oscillator_active:
        if datum.target_oscillator_strengths is None:
            raise ValueError(
                "target_oscillator_strengths is required when an oscillator-strength loss weight is nonzero."
            )
        strength_target = jnp.asarray(datum.target_oscillator_strengths, dtype=dtype)
        requested_nstates = (
            int(config.oscillator_strength_nstates)
            if config.oscillator_strength_nstates is not None
            else int(strength_target.shape[0])
        )
        requested_nstates = max(1, requested_nstates)
        _, predicted_strengths = get_excited_state_observables(
            requested_nstates,
            use_tda,
            need_strengths=True,
            tda_gradient_mode=config.tda_gradient_mode,
        )
        strength_predicted = jnp.asarray(predicted_strengths, dtype=dtype)
        if strength_predicted.ndim != 1:
            strength_predicted = jnp.reshape(strength_predicted, (-1,))
        n_compare = min(
            int(strength_predicted.shape[0]),
            int(strength_target.shape[0]),
            requested_nstates,
        )
        if n_compare <= 0:
            raise ValueError(
                "oscillator-strength constraint requested but no comparable states "
                "were produced."
            )
        strength_predicted = strength_predicted[:n_compare]
        strength_target = strength_target[:n_compare]
        strength_residual = strength_predicted - strength_target
        terms["oscillator_strength_predicted"] = strength_predicted
        terms["oscillator_strength_target"] = strength_target
        terms["oscillator_strength_mse"] = jnp.mean(strength_residual**2)
        terms["oscillator_strength_mae"] = jnp.mean(jnp.abs(strength_residual))
        terms["oscillator_strength_loss"] = (
            config.oscillator_strength_mse_weight * terms["oscillator_strength_mse"]
            + config.oscillator_strength_mae_weight * terms["oscillator_strength_mae"]
        )

    spectrum_active = config.spectrum_mse_weight != 0.0 or config.spectrum_mae_weight != 0.0
    if spectrum_active:
        if datum.target_spectrum_grid_ev is None:
            raise ValueError(
                "target_spectrum_grid_ev is required when a spectrum loss weight is nonzero."
            )
        if datum.target_spectrum_curve is None:
            raise ValueError(
                "target_spectrum_curve is required when a spectrum loss weight is nonzero."
            )
        target_grid_ev = jnp.asarray(datum.target_spectrum_grid_ev, dtype=dtype)
        target_curve = jnp.asarray(datum.target_spectrum_curve, dtype=dtype)
        requested_nstates = config.spectrum_nstates
        if requested_nstates is None:
            if datum.target_excitation_gaps_h is not None:
                requested_nstates = int(jnp.asarray(datum.target_excitation_gaps_h).shape[0])
            else:
                requested_nstates = 1
        requested_nstates = max(1, int(requested_nstates))
        predicted_energies, predicted_strengths = get_excited_state_observables(
            requested_nstates,
            use_tda,
            need_strengths=True,
        )
        predicted_curve = lorentzian_spectrum(
            jnp.asarray(predicted_energies, dtype=dtype) * HARTREE_TO_EV,
            jnp.asarray(predicted_strengths, dtype=dtype),
            target_grid_ev,
            eta=config.spectrum_eta_ev,
        )
        predicted_curve = jnp.nan_to_num(
            predicted_curve,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )
        target_curve = jnp.nan_to_num(
            target_curve,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )
        target_rms = jnp.maximum(jnp.sqrt(jnp.mean(target_curve**2)), 1e-8)
        spectrum_residual = (predicted_curve - target_curve) / target_rms
        terms["spectrum_mse"] = jnp.mean(spectrum_residual**2)
        terms["spectrum_mae"] = jnp.mean(jnp.abs(spectrum_residual))
        terms["spectrum_loss"] = (
            config.spectrum_mse_weight * terms["spectrum_mse"]
            + config.spectrum_mae_weight * terms["spectrum_mae"]
        )

    return terms


def _needs_predicted_ground_state_energy(
    config: MolecularTrainingConfig,
) -> bool:
    return bool(
        config.e0_total_mse_weight != 0.0
        or config.e0_total_mae_weight != 0.0
        or config.self_consistent_e0_weight != 0.0
        or config.s1_total_mse_weight != 0.0
        or config.s1_total_mae_weight != 0.0
    )


_LOSS_COMPONENT_KEYS = (
    "e0_total_loss",
    "grid_density_loss",
    "density_matrix_loss",
    "xc_potential_loss",
    "xc_kernel_loss",
    "self_consistent_e0_loss",
    "orbital_energy_loss",
    "coefficient_prior_loss",
    "density_stationarity_loss",
    "dm21_scf_loss",
    "fractional_linearity_loss",
    "s1_total_loss",
    "excitation_gap_loss",
    "oscillator_strength_loss",
    "spectrum_loss",
)

_MOLECULAR_METRIC_KEYS = (
    "e0_total_mse",
    "e0_total_mae",
    "normalized_e0_total_mse",
    "normalized_e0_total_mae",
    "grid_density_mse",
    "density_matrix_mse",
    "xc_potential_mse",
    "xc_kernel_mse",
    "self_consistent_e0_mse",
    "self_consistent_e0_mae",
    "orbital_energy_mse",
    "orbital_energy_mae",
    "coefficient_prior_mse",
    "dm21_scf_mse",
    "dm21_scf_delta_energy",
    "s1_total_mse",
    "s1_total_mae",
    "s1_total_predicted_h",
    "s1_total_target_h",
    "excitation_gap_mse",
    "excitation_gap_mae",
    "excitation_gap_predicted_h",
    "excitation_gap_target_h",
    "oscillator_strength_mse",
    "oscillator_strength_mae",
    "oscillator_strength_predicted",
    "oscillator_strength_target",
    "spectrum_mse",
    "spectrum_mae",
    "predicted_e0_total_h",
)

_SCF_METRIC_ATTRS = (
    ("scf_converged", "converged"),
    ("scf_cycles", "cycles"),
    ("scf_selected_cycle", "selected_cycle"),
    ("scf_best_cycle", "best_cycle"),
    ("scf_final_rms_density", "final_rms_density"),
    ("scf_selected_rms_density", "selected_rms_density"),
    ("scf_best_rms_density", "best_rms_density"),
)
_SCF_METRIC_KEYS = tuple(key for key, _ in _SCF_METRIC_ATTRS)
_SCF_SUMMARY_METRICS = (
    ("scf_converged_fraction", "scf_converged", "mean"),
    ("scf_cycles_mean", "scf_cycles", "mean"),
    ("scf_cycles_max", "scf_cycles", "max"),
    ("scf_selected_cycle_mean", "scf_selected_cycle", "mean"),
    ("scf_best_cycle_mean", "scf_best_cycle", "mean"),
    ("scf_final_rms_mean", "scf_final_rms_density", "mean"),
    ("scf_final_rms_max", "scf_final_rms_density", "max"),
    ("scf_selected_rms_mean", "scf_selected_rms_density", "mean"),
    ("scf_selected_rms_max", "scf_selected_rms_density", "max"),
    ("scf_best_rms_mean", "scf_best_rms_density", "mean"),
    ("scf_best_rms_max", "scf_best_rms_density", "max"),
)


def _new_metric_terms(keys: Sequence[str]) -> dict[str, list[Array]]:
    return {key: [] for key in keys}


def _append_metric_term(
    terms: dict[str, list[Array]],
    key: str,
    value: Any,
) -> None:
    terms[key].append(jnp.atleast_1d(value))


def _append_metric_terms(
    terms: dict[str, list[Array]],
    values: dict[str, Any],
) -> None:
    for key, value in values.items():
        _append_metric_term(terms, key, value)


def _concat_metric_terms(terms: Sequence[Array], *, empty_dtype: Any) -> Array:
    if not terms:
        return jnp.array([], dtype=empty_dtype)
    return jnp.concatenate(terms)


def _mean_or_nan(values: Array, *, dtype: Any) -> Array:
    if int(values.size) <= 0:
        return jnp.asarray([jnp.nan], dtype=dtype)
    return jnp.asarray([jnp.mean(values)], dtype=dtype)


def _max_or_nan(values: Array, *, dtype: Any) -> Array:
    if int(values.size) <= 0:
        return jnp.asarray([jnp.nan], dtype=dtype)
    return jnp.asarray([jnp.max(values)], dtype=dtype)


def density_on_grid(molecule: Any) -> Array:
    """Return spin-summed density sampled on the integration grid."""

    density = density_on_grid_spin_resolved(molecule)
    if density.ndim == 1:
        return density
    return density.sum(axis=-1)


def density_on_grid_spin_resolved(molecule: Any) -> Array:
    """Return spin-resolved density sampled on the integration grid."""

    if hasattr(molecule, "density"):
        density = molecule.density()
        density = jnp.asarray(density)
        if density.ndim == 2:
            return density
        if density.ndim == 1:
            density_matrix = getattr(molecule, "rdm1", None)
            if density_matrix is None or jnp.asarray(density_matrix).ndim != 3:
                return density[:, None]

    if getattr(molecule, "rdm1", None) is None or getattr(molecule, "ao", None) is None:
        raise AttributeError("Molecule-like object must define density() or both rdm1 and ao.")

    density_matrix = jnp.asarray(molecule.rdm1)
    ao = jnp.asarray(molecule.ao)
    if density_matrix.ndim == 2:
        density = jnp.einsum("pq,rp,rq->r", density_matrix, ao, ao)
        return density[:, None]
    if density_matrix.ndim == 3:
        return jnp.einsum("spq,rp,rq->rs", density_matrix, ao, ao)
    raise ValueError("Expected rdm1 to have shape (nao, nao) or (spin, nao, nao).")


def _density_on_grid_from_density_matrix(molecule: Any, density_matrix: Any) -> Array:
    if getattr(molecule, "ao", None) is None:
        raise AttributeError("Molecule-like object must define ao for density projection.")
    ao = jnp.asarray(molecule.ao)
    density_matrix = jnp.asarray(density_matrix)
    if density_matrix.ndim == 2:
        return jnp.einsum("pq,rp,rq->r", density_matrix, ao, ao)
    if density_matrix.ndim == 3:
        spin_density = jnp.einsum("spq,rp,rq->rs", density_matrix, ao, ao)
        return spin_density.sum(axis=-1)
    raise ValueError("Expected density_matrix to have shape (nao, nao) or (spin, nao, nao).")


def _spin_summed_density_matrix(molecule: Any) -> Array:
    density_matrix = jnp.asarray(molecule.rdm1)
    if density_matrix.ndim == 3:
        return density_matrix.sum(axis=0)
    return density_matrix


def _one_body_energy(density_matrix: Array, h1e: Array) -> Array:
    return jnp.einsum("ij,ij->", density_matrix, jnp.asarray(h1e))


def _coulomb_potential(density_matrix: Array, rep_tensor: Array) -> Array:
    rep = jnp.asarray(rep_tensor)
    if rep.ndim == 2:
        return build_j_from_eri_pair_matrix(rep, density_matrix)
    if int(rep.size) == 0:
        raise ValueError("Coulomb potential requires full AO ERI or packed AO-pair ERI data.")
    return jnp.einsum("pqrt,rt->pq", rep, density_matrix)


def _exchange_potential(density_matrix: Array, rep_tensor: Array) -> Array:
    rep = jnp.asarray(rep_tensor)
    if rep.ndim == 2:
        _, k_matrix = build_jk_from_eri_pair_matrix(rep, density_matrix)
        return k_matrix
    if int(rep.size) == 0:
        raise ValueError("Exchange potential requires full AO ERI or packed AO-pair ERI data.")
    return jnp.einsum("prqs,rs->pq", rep, density_matrix)


def _coulomb_energy(density_matrix: Array, rep_tensor: Array) -> Array:
    potential = _coulomb_potential(density_matrix, rep_tensor)
    return 0.5 * jnp.einsum("ij,ij->", density_matrix, potential)


def _coulomb_potential_from_molecule(molecule: Any, density_matrix: Array) -> Array:
    df_factors = getattr(molecule, "df_factors", None)
    if df_factors is not None:
        factors = jnp.asarray(df_factors)
        if int(factors.size) > 0:
            return build_j_from_df(factors, density_matrix)
    return _coulomb_potential(density_matrix, _repulsion_integrals_from_molecule(molecule))


def _exchange_potential_from_molecule(molecule: Any, density_matrix: Array) -> Array:
    df_factors = getattr(molecule, "df_factors", None)
    if df_factors is not None:
        factors = jnp.asarray(df_factors)
        if int(factors.size) > 0:
            _, k_matrix = build_jk_from_df(factors, density_matrix)
            return k_matrix
    return _exchange_potential(density_matrix, _repulsion_integrals_from_molecule(molecule))


def _coulomb_energy_from_molecule(molecule: Any, density_matrix: Array) -> Array:
    potential = _coulomb_potential_from_molecule(molecule, density_matrix)
    return 0.5 * jnp.einsum("ij,ij->", density_matrix, potential)


def _repulsion_integrals_from_molecule(molecule: Any) -> Array:
    rep_tensor = getattr(molecule, "rep_tensor", None)
    if rep_tensor is not None:
        rep = jnp.asarray(rep_tensor)
        if int(rep.size) > 0:
            return rep
    pair = getattr(molecule, "eri_pair_matrix", None)
    if pair is not None:
        pair_arr = jnp.asarray(pair)
        if int(pair_arr.size) > 0:
            return pair_arr
    if rep_tensor is None:
        raise AttributeError("Molecule-like object must define rep_tensor or eri_pair_matrix.")
    return jnp.asarray(rep_tensor)


def _replace_molecule_copy(molecule: Any, **updates: Any) -> Any:
    if is_dataclass(molecule):
        field_names = {field.name for field in fields(molecule)}
        dataclass_updates = {key: value for key, value in updates.items() if key in field_names}
        extra_updates = {key: value for key, value in updates.items() if key not in field_names}
        cloned = replace(molecule, **dataclass_updates)
        for key, value in extra_updates.items():
            object.__setattr__(cloned, key, value)
        return cloned
    cloned = copy.copy(molecule)
    for key, value in updates.items():
        setattr(cloned, key, value)
    return cloned


def _electron_count(molecule: Any) -> Array:
    if hasattr(molecule, "electron_count") and getattr(molecule, "electron_count") is not None:
        return jnp.asarray(getattr(molecule, "electron_count"))
    if hasattr(molecule, "nelectron") and getattr(molecule, "nelectron") is not None:
        return jnp.asarray(getattr(molecule, "nelectron"))
    density_matrix = _spin_summed_density_matrix(molecule)
    overlap = getattr(molecule, "overlap_matrix", None)
    if overlap is not None:
        return jnp.trace(jnp.asarray(density_matrix) @ jnp.asarray(overlap))
    return jnp.trace(jnp.asarray(density_matrix))


def _atom_count(molecule: Any) -> Array:
    if hasattr(molecule, "natm") and getattr(molecule, "natm") is not None:
        return jnp.asarray(getattr(molecule, "natm"))
    z = getattr(molecule, "z", None)
    if z is not None:
        return jnp.asarray(jnp.asarray(z).shape[0])
    mol = getattr(molecule, "mol", None)
    if mol is not None and hasattr(mol, "natm"):
        return jnp.asarray(getattr(mol, "natm"))
    return jnp.asarray(1.0)


def _energy_normalization_scale(
    molecule: Any,
    cfg: MolecularTrainingConfig,
) -> Array:
    if cfg.e0_normalization == "none":
        return jnp.asarray(1.0)
    if cfg.e0_normalization == "per_electron":
        scale = _electron_count(molecule)
    elif cfg.e0_normalization == "per_atom":
        scale = _atom_count(molecule)
    else:
        raise ValueError(f"Unsupported e0_normalization={cfg.e0_normalization!r}")
    return jnp.maximum(jnp.asarray(scale), cfg.e0_normalization_eps)


def _tree_contains_jax_tracer(tree: Any) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(tree))


def _detach_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        arr = jnp.asarray(value)
    except Exception:
        return value
    return jax.lax.stop_gradient(arr)


def _detach_molecule_state(molecule: Any) -> Any:
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
            grid_out = _replace_molecule_copy(grid, **grid_updates)

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
        "scf_initial_density",
    ):
        if hasattr(molecule, attr):
            value = getattr(molecule, attr)
            if value is not None:
                updates[attr] = _detach_value(value)
    if hasattr(molecule, "nuclear_repulsion"):
        updates["nuclear_repulsion"] = getattr(molecule, "nuclear_repulsion")
    if hasattr(molecule, "hfx_omega_values"):
        updates["hfx_omega_values"] = getattr(molecule, "hfx_omega_values")
    if hasattr(molecule, "nocc_alpha"):
        updates["nocc_alpha"] = getattr(molecule, "nocc_alpha")
    if hasattr(molecule, "nocc_beta"):
        updates["nocc_beta"] = getattr(molecule, "nocc_beta")
    if grid_out is not None:
        updates["grid"] = grid_out
    return _replace_molecule_copy(molecule, **updates)


def _restricted_channel(molecule: Any) -> tuple[Array, Array]:
    mo_coeff = jnp.asarray(molecule.mo_coeff)
    mo_occ = jnp.asarray(molecule.mo_occ)

    if mo_coeff.ndim == 2:
        return mo_coeff, mo_occ
    if mo_coeff.ndim != 3:
        raise ValueError(
            "Expected mo_coeff to have shape (nao, nmo) or (spin, nao, nmo)."
        )
    if mo_coeff.shape[0] == 1:
        return mo_coeff[0], mo_occ[0]
    if mo_coeff.shape[0] != 2:
        raise NotImplementedError(
            "Density-constrained training currently supports restricted references only."
        )
    return mo_coeff[0], mo_occ[0]


def _restricted_channel_with_energies(molecule: Any) -> tuple[Array, Array, Array]:
    mo_coeff, mo_occ = _restricted_channel(molecule)
    mo_energy = jnp.asarray(molecule.mo_energy)
    if mo_energy.ndim == 2:
        mo_energy = mo_energy[0]
    if mo_energy.ndim != 1:
        raise ValueError(
            "Expected mo_energy to have shape (nmo,) or (spin, nmo)."
        )
    nmo = min(int(mo_coeff.shape[-1]), int(mo_occ.shape[-1]), int(mo_energy.shape[-1]))
    return mo_coeff[:, :nmo], mo_occ[:nmo], mo_energy[:nmo]


def _restricted_energies_and_occ(values: Any, occupations: Any) -> tuple[Array, Array]:
    energies = jnp.asarray(values)
    occ = jnp.asarray(occupations)
    if energies.ndim == 2:
        energies = energies[0]
    if occ.ndim == 2:
        occ = occ[0]
    if energies.ndim != 1 or occ.ndim != 1:
        raise ValueError("Expected 1D or restricted spin-stacked orbital energies/occupations.")
    nmo = min(int(energies.shape[0]), int(occ.shape[0]))
    return energies[:nmo], occ[:nmo]


def _orbital_window_mask(
    occupations: Array,
    *,
    window: int | None,
    occupation_tolerance: float,
) -> tuple[Array, Array, Array]:
    occupations = jnp.asarray(occupations)
    nmo = occupations.shape[0]
    idx = jnp.arange(nmo, dtype=jnp.int32)
    occ_mask = occupations > occupation_tolerance
    vir_mask = occupations <= occupation_tolerance
    homo_idx = jnp.max(jnp.where(occ_mask, idx, jnp.asarray(-1, dtype=idx.dtype)))
    lumo_idx = jnp.min(jnp.where(vir_mask, idx, jnp.asarray(nmo, dtype=idx.dtype)))

    if window is None or int(window) <= 0:
        select_mask = jnp.ones((nmo,), dtype=bool)
    else:
        w = max(int(window), 1)
        occ_select = occ_mask & (idx >= homo_idx - w + 1) & (idx <= homo_idx)
        vir_select = vir_mask & (idx >= lumo_idx) & (idx < lumo_idx + w)
        select_mask = occ_select | vir_select
    return select_mask, homo_idx, lumo_idx


def orbital_energy_matching_penalty(
    predicted_molecule: Any,
    *,
    target_orbital_energies: Array,
    target_orbital_occupations: Array,
    window: int | None,
    occupation_tolerance: float = 1e-8,
) -> tuple[Array, Array, Array, Array]:
    """Compare frontier orbital energies after removing the common zero-point shift.

    We align both spectra by the midpoint of the HOMO/LUMO pair and then compare
    individual orbital energies in the requested window.
    """

    pred_energies, pred_occ = _restricted_energies_and_occ(
        predicted_molecule.mo_energy,
        predicted_molecule.mo_occ,
    )
    target_energies, target_occ = _restricted_energies_and_occ(
        target_orbital_energies,
        target_orbital_occupations,
    )
    nmo = min(
        int(pred_energies.shape[0]),
        int(pred_occ.shape[0]),
        int(target_energies.shape[0]),
        int(target_occ.shape[0]),
    )
    pred_energies = pred_energies[:nmo]
    pred_occ = pred_occ[:nmo]
    target_energies = target_energies[:nmo]
    target_occ = target_occ[:nmo]

    select_mask, homo_idx, lumo_idx = _orbital_window_mask(
        target_occ,
        window=window,
        occupation_tolerance=occupation_tolerance,
    )
    pred_zero = 0.5 * (pred_energies[homo_idx] + pred_energies[lumo_idx])
    target_zero = 0.5 * (target_energies[homo_idx] + target_energies[lumo_idx])

    residual = (pred_energies - pred_zero) - (target_energies - target_zero)
    select = select_mask.astype(residual.dtype)
    normalization = jnp.maximum(jnp.sum(select), 1.0)
    mse = jnp.sum(select * (residual**2)) / normalization
    mae = jnp.sum(select * jnp.abs(residual)) / normalization
    return mse, mae, residual, select_mask


def _restricted_frontier_indices(
    occupations: Array,
    *,
    occupation_tolerance: float,
) -> tuple[Array, Array]:
    _, homo_idx, lumo_idx = _orbital_window_mask(
        occupations,
        window=1,
        occupation_tolerance=occupation_tolerance,
    )
    return homo_idx, lumo_idx


def _rebuild_density_matrix_from_orbitals(
    mo_coeff: Array,
    mo_occ: Array,
) -> Array:
    mo_coeff = jnp.asarray(mo_coeff)
    mo_occ = jnp.asarray(mo_occ)
    if mo_coeff.ndim == 2:
        if mo_occ.ndim != 1:
            raise ValueError("Expected 1D occupations for restricted orbital coefficients.")
        return jnp.einsum("pi,i,qi->pq", mo_coeff, mo_occ, mo_coeff)
    if mo_coeff.ndim == 3:
        if mo_occ.ndim == 1:
            mo_occ = jnp.stack([0.5 * mo_occ, 0.5 * mo_occ], axis=0)
        if mo_occ.ndim != 2 or mo_occ.shape[0] != mo_coeff.shape[0]:
            raise ValueError("Spin-resolved orbital coefficients require spin-resolved occupations.")
        return jax.vmap(
            lambda coeff_spin, occ_spin: jnp.einsum(
                "pi,i,qi->pq",
                coeff_spin,
                occ_spin,
                coeff_spin,
            )
        )(mo_coeff, mo_occ)
    raise ValueError("Expected mo_coeff to have shape (nao, nmo) or (spin, nao, nmo).")


def _perturb_restricted_frontier_occupations(
    molecule: Any,
    *,
    homo_delta: float = 0.0,
    lumo_delta: float = 0.0,
    occupation_tolerance: float = 1e-8,
) -> Any:
    mo_coeff = jnp.asarray(molecule.mo_coeff)
    mo_occ = jnp.asarray(molecule.mo_occ)
    restricted_occ = (
        mo_occ if mo_occ.ndim == 1 else jnp.asarray(mo_occ[0])
    )
    homo_idx, lumo_idx = _restricted_frontier_indices(
        restricted_occ,
        occupation_tolerance=occupation_tolerance,
    )

    if mo_occ.ndim == 1:
        updated_occ = mo_occ
        max_occ = jnp.asarray(2.0, dtype=updated_occ.dtype)
        updated_occ = updated_occ.at[homo_idx].set(
            jnp.clip(updated_occ[homo_idx] + homo_delta, 0.0, max_occ)
        )
        updated_occ = updated_occ.at[lumo_idx].set(
            jnp.clip(updated_occ[lumo_idx] + lumo_delta, 0.0, max_occ)
        )
        delta_total = (updated_occ[homo_idx] - mo_occ[homo_idx]) + (
            updated_occ[lumo_idx] - mo_occ[lumo_idx]
        )
    elif mo_occ.ndim == 2 and mo_occ.shape[0] == 2:
        updated_occ = mo_occ
        max_occ = jnp.asarray(1.0, dtype=updated_occ.dtype)
        half_homo_delta = 0.5 * jnp.asarray(homo_delta, dtype=updated_occ.dtype)
        half_lumo_delta = 0.5 * jnp.asarray(lumo_delta, dtype=updated_occ.dtype)
        updated_occ = updated_occ.at[:, homo_idx].set(
            jnp.clip(updated_occ[:, homo_idx] + half_homo_delta, 0.0, max_occ)
        )
        updated_occ = updated_occ.at[:, lumo_idx].set(
            jnp.clip(updated_occ[:, lumo_idx] + half_lumo_delta, 0.0, max_occ)
        )
        delta_total = jnp.sum(updated_occ[:, homo_idx] - mo_occ[:, homo_idx]) + jnp.sum(
            updated_occ[:, lumo_idx] - mo_occ[:, lumo_idx]
        )
    else:
        raise ValueError("Frontier occupation perturbation currently supports restricted orbitals only.")

    updates = {
        "mo_occ": updated_occ,
        "rdm1": _rebuild_density_matrix_from_orbitals(mo_coeff, updated_occ),
    }
    if hasattr(molecule, "electron_count"):
        current = getattr(molecule, "electron_count")
        if current is not None:
            updates["electron_count"] = jnp.asarray(current) + delta_total
    if hasattr(molecule, "nelectron"):
        current = getattr(molecule, "nelectron")
        if current is not None:
            updates["nelectron"] = jnp.asarray(current) + delta_total
    return _replace_molecule_copy(molecule, **updates)


@dataclass(frozen=True)
class _FrozenFunctionalAdapter:
    """Freeze a molecule-bound functional for fractional-state diagnostics.

    For nn-RSH, the learned `(sr, lr, omega)` depend on the input density
    descriptor. Fractional-state probes should instead keep the
    functional fixed at the base state and only vary occupations/orbitals.
    """

    bound: Any

    def bind_to_molecule(self, _params: PyTree, _molecule: Any) -> Any:
        return self.bound

    def bind_to_molecule_for_scf(self, _params: PyTree, _molecule: Any) -> Any:
        return self.bound

    def energy_from_molecule(self, _params: PyTree, molecule: Any) -> Array:
        energy_fn = getattr(self.bound, "energy_from_molecule", None)
        if energy_fn is None:
            raise AttributeError("Frozen bound functional must expose energy_from_molecule(molecule).")
        return energy_fn(molecule)


def _freeze_functional_for_fractional_path(
    params: PyTree,
    functional: Any,
    molecule: Any,
) -> tuple[Any, PyTree]:
    resolver = getattr(functional, "resolve_parameters", None)
    params_with_resolved = getattr(functional, "params_with_resolved", None)
    if callable(resolver) and callable(params_with_resolved):
        try:
            resolved = resolver(params, molecule)
        except TypeError:
            resolved = resolver(params)
        fixed_params = params_with_resolved(
            params,
            resolved,
            molecule=molecule,
            preserve_network=False,
        )
        return functional, fixed_params
    scf_binder = getattr(functional, "bind_to_molecule_for_scf", None)
    if callable(scf_binder):
        return _FrozenFunctionalAdapter(scf_binder(params, molecule)), params
    binder = getattr(functional, "bind_to_molecule", None)
    if callable(binder):
        return _FrozenFunctionalAdapter(binder(params, molecule)), params
    return functional, params


def _fractional_branch_quality_weight(
    scf_info: Any | None,
    training_config: MolecularTrainingConfig | None,
    *,
    dtype: Any | None = None,
) -> Array:
    dtype = jnp.float32 if dtype is None else dtype
    cfg = MolecularTrainingConfig() if training_config is None else training_config
    threshold = getattr(cfg, "fractional_branch_rms_soft_threshold", None)
    if threshold is None or scf_info is None or getattr(scf_info, "mode", None) != "self_consistent":
        return jnp.asarray(1.0, dtype=dtype)
    threshold_arr = jnp.maximum(jnp.asarray(float(threshold), dtype=dtype), 1e-12)
    rms = jnp.nan_to_num(
        jnp.asarray(getattr(scf_info, "selected_rms_density", 0.0), dtype=dtype),
        nan=jnp.inf,
        posinf=jnp.inf,
        neginf=jnp.inf,
    )
    weight = jnp.where(
        rms <= threshold_arr,
        jnp.asarray(1.0, dtype=dtype),
        (threshold_arr / jnp.maximum(rms, threshold_arr)) ** 2,
    )
    return jnp.clip(weight, 0.0, 1.0)


def _as_self_consistent_training_config(
    training_config: MolecularTrainingConfig | None,
) -> MolecularTrainingConfig:
    cfg = MolecularTrainingConfig() if training_config is None else training_config
    return cfg if cfg.mode == "self_consistent" else replace(cfg, mode="self_consistent")


def _fractional_branch_training_config(
    training_config: MolecularTrainingConfig | None,
) -> MolecularTrainingConfig:
    cfg = _as_self_consistent_training_config(training_config)
    branch_max_cycle = cfg.fractional_branch_scf_max_cycle
    if branch_max_cycle is None:
        branch_max_cycle = max(int(cfg.scf_max_cycle), 8)
    branch_damping = cfg.fractional_branch_scf_damping
    if branch_damping is None:
        branch_damping = max(float(cfg.scf_damping), 0.35)
    branch_level_shift = cfg.fractional_branch_scf_level_shift
    if branch_level_shift is None:
        branch_level_shift = max(float(cfg.scf_level_shift), 0.5)
    branch_iterate_selection = cfg.fractional_branch_scf_iterate_selection
    if branch_iterate_selection is None:
        branch_iterate_selection = "best_rms"
    return replace(
        cfg,
        scf_max_cycle=int(branch_max_cycle),
        scf_damping=float(branch_damping),
        scf_level_shift=float(branch_level_shift),
        scf_iterate_selection=branch_iterate_selection,
    )


@dataclass(frozen=True)
class KoopmansIPEADiagnostic:
    neutral_energy: Array
    cation_energy: Array
    anion_energy: Array
    ip_delta_scf: Array
    ea_delta_scf: Array
    neutral_homo_energy: Array
    anion_homo_energy: Array
    ip_residual: Array
    ea_residual: Array
    cation_converged: bool
    anion_converged: bool
    cation_result: Any
    anion_result: Any


def _spin_resolved_orbital_blocks(
    molecule: Any,
    *,
    occupation_tolerance: float = 1e-8,
) -> tuple[Array, Array, Array]:
    mo_coeff = jnp.asarray(molecule.mo_coeff)
    mo_occ = jnp.asarray(molecule.mo_occ)
    mo_energy = jnp.asarray(molecule.mo_energy)

    if mo_coeff.ndim == 2:
        mo_coeff = jnp.stack([mo_coeff, mo_coeff], axis=0)
    elif mo_coeff.ndim != 3 or int(mo_coeff.shape[0]) != 2:
        raise ValueError(
            "Charged-state Koopmans diagnostics require mo_coeff with shape "
            "(nao, nmo) or (2, nao, nmo)."
        )

    if mo_occ.ndim == 1:
        if float(jnp.max(mo_occ)) > 1.0 + occupation_tolerance:
            mo_occ = jnp.stack([0.5 * mo_occ, 0.5 * mo_occ], axis=0)
        else:
            mo_occ = jnp.stack([mo_occ, mo_occ], axis=0)
    elif mo_occ.ndim != 2 or int(mo_occ.shape[0]) != 2:
        raise ValueError(
            "Charged-state Koopmans diagnostics require mo_occ with shape "
            "(nmo,) or (2, nmo)."
        )

    if mo_energy.ndim == 1:
        mo_energy = jnp.stack([mo_energy, mo_energy], axis=0)
    elif mo_energy.ndim != 2 or int(mo_energy.shape[0]) != 2:
        raise ValueError(
            "Charged-state Koopmans diagnostics require mo_energy with shape "
            "(nmo,) or (2, nmo)."
        )

    nmo = min(
        int(mo_coeff.shape[-1]),
        int(mo_occ.shape[-1]),
        int(mo_energy.shape[-1]),
    )
    return mo_coeff[:, :, :nmo], mo_occ[:, :nmo], mo_energy[:, :nmo]


def _as_spin_resolved_molecule(
    molecule: Any,
    *,
    occupation_tolerance: float = 1e-8,
) -> Any:
    mo_coeff_spin, mo_occ_spin, mo_energy_spin = _spin_resolved_orbital_blocks(
        molecule,
        occupation_tolerance=occupation_tolerance,
    )
    updates = {
        "mo_coeff": mo_coeff_spin,
        "mo_occ": mo_occ_spin,
        "mo_energy": mo_energy_spin,
        "rdm1": _rebuild_density_matrix_from_orbitals(mo_coeff_spin, mo_occ_spin),
        "nocc_alpha": int(jnp.sum(mo_occ_spin[0] > occupation_tolerance)),
        "nocc_beta": int(jnp.sum(mo_occ_spin[1] > occupation_tolerance)),
    }
    if hasattr(molecule, "scf_initial_density"):
        updates["scf_initial_density"] = updates["rdm1"].sum(axis=0)
    return _replace_molecule_copy(molecule, **updates)


def _spin_orbital_frontier_indices(
    molecule: Any,
    *,
    occupation_tolerance: float = 1e-8,
) -> tuple[Array, Array, Array, Array]:
    _, mo_occ_spin, mo_energy_spin = _spin_resolved_orbital_blocks(
        molecule,
        occupation_tolerance=occupation_tolerance,
    )
    occ_mask = mo_occ_spin > occupation_tolerance
    vir_mask = mo_occ_spin <= occupation_tolerance
    flat_occ = jnp.where(occ_mask, mo_energy_spin, -jnp.inf).reshape(-1)
    flat_vir = jnp.where(vir_mask, mo_energy_spin, jnp.inf).reshape(-1)
    nspin, nmo = mo_occ_spin.shape
    del nspin
    homo_flat = jnp.argmax(flat_occ)
    lumo_flat = jnp.argmin(flat_vir)
    return (
        homo_flat // nmo,
        homo_flat % nmo,
        lumo_flat // nmo,
        lumo_flat % nmo,
    )


def _homo_energy_from_spin_orbitals(
    mo_energy_spin: Array,
    mo_occ_spin: Array,
    *,
    occupation_tolerance: float = 1e-8,
) -> Array:
    occ_mask = jnp.asarray(mo_occ_spin) > occupation_tolerance
    energies = jnp.asarray(mo_energy_spin)
    masked = jnp.where(occ_mask, energies, jnp.asarray(-1.0e6, dtype=energies.dtype))
    return jnp.max(masked)


def _minimal_spin_for_electron_count(
    total_electrons: int,
    *,
    sign_hint: int = 1,
) -> tuple[int, int]:
    if total_electrons <= 0:
        raise ValueError("Charged-state Koopmans diagnostics require a positive electron count.")
    if total_electrons % 2 == 0:
        spin = 0
    else:
        spin = 1 if sign_hint >= 0 else -1
    nalpha = (total_electrons + spin) // 2
    nbeta = total_electrons - nalpha
    return int(nalpha), int(nbeta)


def _default_charged_state_uks_config(
    bound_xc: Any,
    override: UKSConfig | None,
) -> UKSConfig:
    if override is not None:
        return override
    return UKSConfig(
        xc_spec=str(getattr(bound_xc, "local_xc_spec", "hf")),
        max_cycle=32,
        conv_tol=1e-10,
        conv_tol_density=1e-8,
        damping=0.35,
        level_shift=0.5,
        orthogonalization_eps=1e-10,
        density_floor=1e-12,
        potential_clip=20.0,
    )


def charged_state_uks_from_molecule(
    molecule: Any,
    bound_xc: Any,
    *,
    charge_delta: int,
    config: UKSConfig | None = None,
    occupation_tolerance: float = 1e-8,
) -> Any:
    if int(charge_delta) not in (-1, 1):
        raise ValueError("charge_delta must be +1 (cation) or -1 (anion).")
    mo_coeff_spin, mo_occ_spin, mo_energy_spin = _spin_resolved_orbital_blocks(
        molecule,
        occupation_tolerance=occupation_tolerance,
    )
    neutral_total = int(round(float(jnp.sum(mo_occ_spin))))
    neutral_spin = int(round(float(jnp.sum(mo_occ_spin[0]) - jnp.sum(mo_occ_spin[1]))))
    target_total = neutral_total - int(charge_delta)
    nalpha, nbeta = _minimal_spin_for_electron_count(
        target_total,
        sign_hint=neutral_spin if neutral_spin != 0 else 1,
    )
    charged_cfg = _default_charged_state_uks_config(bound_xc, config)
    return run_uks_from_integrals(
        overlap=jnp.asarray(molecule.overlap_matrix),
        hcore=jnp.asarray(molecule.h1e),
        eri=_repulsion_integrals_from_molecule(molecule),
        nalpha=nalpha,
        nbeta=nbeta,
        nuclear_repulsion=jnp.asarray(molecule.nuclear_repulsion),
        ao=jnp.asarray(molecule.ao),
        ao_deriv1=jnp.asarray(molecule.ao_deriv1),
        grid_weights=jnp.asarray(molecule.grid.weights),
        init_mo_coeff_alpha=mo_coeff_spin[0],
        init_mo_coeff_beta=mo_coeff_spin[1],
        init_mo_occ_alpha=jnp.zeros((mo_occ_spin.shape[-1],), dtype=mo_occ_spin.dtype).at[:nalpha].set(1.0),
        init_mo_occ_beta=jnp.zeros((mo_occ_spin.shape[-1],), dtype=mo_occ_spin.dtype).at[:nbeta].set(1.0),
        init_mo_energy_alpha=mo_energy_spin[0],
        init_mo_energy_beta=mo_energy_spin[1],
        config=charged_cfg,
        bound_xc=bound_xc,
        molecule_template=molecule,
    )


def _charged_spin_molecule_from_molecule(
    molecule: Any,
    *,
    charge_delta: int,
    occupation_tolerance: float = 1e-8,
) -> Any:
    if int(charge_delta) not in (-1, 1):
        raise ValueError("charge_delta must be +1 (cation) or -1 (anion).")
    mo_coeff_spin, mo_occ_spin, mo_energy_spin = _spin_resolved_orbital_blocks(
        molecule,
        occupation_tolerance=occupation_tolerance,
    )
    neutral_total = int(round(float(jnp.sum(mo_occ_spin))))
    neutral_spin = int(round(float(jnp.sum(mo_occ_spin[0]) - jnp.sum(mo_occ_spin[1]))))
    target_total = neutral_total - int(charge_delta)
    nalpha, nbeta = _minimal_spin_for_electron_count(
        target_total,
        sign_hint=neutral_spin if neutral_spin != 0 else 1,
    )
    mo_occ_a = jnp.zeros((mo_occ_spin.shape[-1],), dtype=mo_occ_spin.dtype).at[:nalpha].set(1.0)
    mo_occ_b = jnp.zeros((mo_occ_spin.shape[-1],), dtype=mo_occ_spin.dtype).at[:nbeta].set(1.0)
    updated_occ = jnp.stack([mo_occ_a, mo_occ_b], axis=0)
    updated_rdm1 = _rebuild_density_matrix_from_orbitals(mo_coeff_spin, updated_occ)
    updates = {
        "mo_coeff": mo_coeff_spin,
        "mo_occ": updated_occ,
        "mo_energy": mo_energy_spin,
        "rdm1": updated_rdm1,
        "nocc_alpha": nalpha,
        "nocc_beta": nbeta,
    }
    if hasattr(molecule, "electron_count"):
        updates["electron_count"] = jnp.asarray(target_total, dtype=updated_occ.dtype)
    if hasattr(molecule, "nelectron"):
        updates["nelectron"] = jnp.asarray(target_total, dtype=updated_occ.dtype)
    if hasattr(molecule, "scf_initial_density"):
        updates["scf_initial_density"] = updated_rdm1.sum(axis=0)
    return _replace_molecule_copy(molecule, **updates)


def charged_state_differentiable_scf_from_molecule(
    molecule: Any,
    bound_xc: Any,
    *,
    charge_delta: int,
    training_config: MolecularTrainingConfig | None = None,
    occupation_tolerance: float = 1e-8,
) -> tuple[Any, Any]:
    charged_initial = _charged_spin_molecule_from_molecule(
        molecule,
        charge_delta=charge_delta,
        occupation_tolerance=occupation_tolerance,
    )
    return _resolve_training_molecule_and_info_with_mode(
        None,
        _FrozenFunctionalAdapter(bound_xc),
        charged_initial,
        _as_self_consistent_training_config(training_config),
    )


def koopmans_ip_ea_diagnostic(
    molecule: Any,
    bound_xc: Any,
    *,
    cation_config: UKSConfig | None = None,
    anion_config: UKSConfig | None = None,
    occupation_tolerance: float = 1e-8,
) -> KoopmansIPEADiagnostic:
    frozen_functional = _FrozenFunctionalAdapter(bound_xc)
    neutral_energy = _predict_ground_state_total_energy_from_molecule(
        None,
        frozen_functional,
        molecule,
    )
    _, neutral_occ_spin, neutral_energy_spin = _spin_resolved_orbital_blocks(
        molecule,
        occupation_tolerance=occupation_tolerance,
    )
    neutral_homo = _homo_energy_from_spin_orbitals(
        neutral_energy_spin,
        neutral_occ_spin,
        occupation_tolerance=occupation_tolerance,
    )

    cation = charged_state_uks_from_molecule(
        molecule,
        bound_xc,
        charge_delta=1,
        config=cation_config,
        occupation_tolerance=occupation_tolerance,
    )
    anion = charged_state_uks_from_molecule(
        molecule,
        bound_xc,
        charge_delta=-1,
        config=anion_config,
        occupation_tolerance=occupation_tolerance,
    )
    cation_energy = jnp.asarray(cation.total_energy)
    anion_energy = jnp.asarray(anion.total_energy)
    ip_delta_scf = cation_energy - neutral_energy
    ea_delta_scf = neutral_energy - anion_energy
    anion_homo = _homo_energy_from_spin_orbitals(
        jnp.stack([anion.mo_energy_alpha, anion.mo_energy_beta], axis=0),
        jnp.stack([anion.mo_occ_alpha, anion.mo_occ_beta], axis=0),
        occupation_tolerance=occupation_tolerance,
    )
    return KoopmansIPEADiagnostic(
        neutral_energy=neutral_energy,
        cation_energy=cation_energy,
        anion_energy=anion_energy,
        ip_delta_scf=ip_delta_scf,
        ea_delta_scf=ea_delta_scf,
        neutral_homo_energy=neutral_homo,
        anion_homo_energy=anion_homo,
        ip_residual=neutral_homo + ip_delta_scf,
        ea_residual=anion_homo + ea_delta_scf,
        cation_converged=bool(cation.converged),
        anion_converged=bool(anion.converged),
        cation_result=cation,
        anion_result=anion,
    )


charged_state_uks_from_reference = charged_state_uks_from_molecule
_charged_spin_molecule_from_reference = _charged_spin_molecule_from_molecule
charged_state_differentiable_scf_from_reference = (
    charged_state_differentiable_scf_from_molecule
)


def _resolve_variational_frontier_state_and_info(
    params: PyTree,
    functional: Any,
    molecule: Any,
    *,
    homo_delta: float = 0.0,
    lumo_delta: float = 0.0,
    training_config: MolecularTrainingConfig | None = None,
    occupation_tolerance: float = 1e-8,
) -> Any:
    perturbed = _perturb_restricted_frontier_occupations(
        molecule,
        homo_delta=homo_delta,
        lumo_delta=lumo_delta,
        occupation_tolerance=occupation_tolerance,
    )
    if hasattr(perturbed, "scf_initial_density"):
        perturbed = _replace_molecule_copy(
            perturbed,
            scf_initial_density=_spin_summed_density_matrix(perturbed),
        )
    return _resolve_training_molecule_and_info_with_mode(
        params,
        functional,
        perturbed,
        _fractional_branch_training_config(training_config),
    )


def _resolve_variational_frontier_state(
    params: PyTree,
    functional: Any,
    molecule: Any,
    *,
    homo_delta: float = 0.0,
    lumo_delta: float = 0.0,
    training_config: MolecularTrainingConfig | None = None,
    occupation_tolerance: float = 1e-8,
) -> Any:
    molecule_out, _ = _resolve_variational_frontier_state_and_info(
        params,
        functional,
        molecule,
        homo_delta=homo_delta,
        lumo_delta=lumo_delta,
        training_config=training_config,
        occupation_tolerance=occupation_tolerance,
    )
    return molecule_out


def _perturb_spin_orbital_occupation(
    molecule: Any,
    *,
    spin_index: Any,
    orbital_index: Any,
    delta: Any,
) -> Any:
    mo_coeff_spin, mo_occ_spin, mo_energy_spin = _spin_resolved_orbital_blocks(molecule)
    updated_occ = mo_occ_spin.at[spin_index, orbital_index].set(
        jnp.clip(
            mo_occ_spin[spin_index, orbital_index] + jnp.asarray(delta, dtype=mo_occ_spin.dtype),
            0.0,
            1.0,
        )
    )
    delta_total = updated_occ[spin_index, orbital_index] - mo_occ_spin[spin_index, orbital_index]
    updates = {
        "mo_coeff": mo_coeff_spin,
        "mo_occ": updated_occ,
        "mo_energy": mo_energy_spin,
        "rdm1": _rebuild_density_matrix_from_orbitals(mo_coeff_spin, updated_occ),
    }
    if hasattr(molecule, "nocc_alpha"):
        updates["nocc_alpha"] = getattr(molecule, "nocc_alpha")
    if hasattr(molecule, "nocc_beta"):
        updates["nocc_beta"] = getattr(molecule, "nocc_beta")
    if hasattr(molecule, "electron_count"):
        current = getattr(molecule, "electron_count")
        if current is not None:
            updates["electron_count"] = jnp.asarray(current) + delta_total
    if hasattr(molecule, "nelectron"):
        current = getattr(molecule, "nelectron")
        if current is not None:
            updates["nelectron"] = jnp.asarray(current) + delta_total
    if hasattr(molecule, "scf_initial_density"):
        updates["scf_initial_density"] = updates["rdm1"].sum(axis=0)
    return _replace_molecule_copy(molecule, **updates)


def _resolved_xc_object(
    params: PyTree,
    functional: Any,
    molecule: Any,
) -> Any:
    scf_molecule_binder = getattr(functional, "bind_to_molecule_for_scf", None)
    if scf_molecule_binder is not None:
        return scf_molecule_binder(params, molecule)
    molecule_binder = getattr(functional, "bind_to_molecule", None)
    if molecule_binder is not None:
        return molecule_binder(params, molecule)
    binder = getattr(functional, "bind", None)
    if binder is not None:
        return binder(params)
    return functional


def _grid_xc_potential(
    params: PyTree,
    functional: Any,
    molecule: Any,
) -> Array:
    resolved = _resolved_xc_object(params, functional, molecule)
    return _grid_xc_potential_from_resolved(
        resolved,
        functional=functional,
        params=params,
        molecule=molecule,
    )


def _grid_xc_potential_from_resolved(
    resolved: Any,
    *,
    functional: Any,
    params: PyTree,
    molecule: Any,
) -> Array:
    grid_potential = getattr(resolved, "grid_potential", None)
    if grid_potential is not None:
        return jnp.asarray(grid_potential(molecule))

    total_density = density_on_grid(molecule)
    local_potential = getattr(resolved, "local_potential", None)
    if local_potential is not None:
        return jnp.asarray(local_potential(total_density))

    functional_local_potential = getattr(functional, "local_potential", None)
    if functional_local_potential is None:
        raise AttributeError(
            "The XC functional must expose local_potential(...) or grid_potential(...)."
        )
    return jnp.asarray(functional_local_potential(params, total_density))


def _grid_xc_kernel(
    params: PyTree,
    functional: Any,
    molecule: Any,
) -> Array:
    # Kernel supervision requires the full response-enabled binding.
    # Some SCF-only binders intentionally expose a zero/placeholder kernel.
    full_binder = getattr(functional, "bind_to_molecule", None)
    if full_binder is not None:
        resolved = full_binder(params, molecule)
    else:
        resolved = _resolved_xc_object(params, functional, molecule)
    return _grid_xc_kernel_from_resolved(
        resolved,
        functional=functional,
        params=params,
        molecule=molecule,
    )


def _grid_xc_kernel_from_resolved(
    resolved: Any,
    *,
    functional: Any,
    params: PyTree,
    molecule: Any,
) -> Array:
    grid_kernel = getattr(resolved, "grid_kernel", None)
    if grid_kernel is not None:
        return jnp.asarray(grid_kernel(molecule))

    total_density = density_on_grid(molecule)
    local_kernel = getattr(resolved, "local_kernel", None)
    if local_kernel is not None:
        return jnp.asarray(local_kernel(total_density))

    functional_local_kernel = getattr(functional, "local_kernel", None)
    if functional_local_kernel is None:
        raise AttributeError(
            "The XC functional must expose local_kernel(...) or grid_kernel(...)."
        )
    return jnp.asarray(functional_local_kernel(params, total_density))


def _as_grid_density_density_component(values: Array, ngrids: int, *, name: str) -> Array:
    """Normalize flexible kernel-like grid arrays to a scalar (d/d rho) channel."""

    arr = jnp.asarray(values)
    if arr.ndim == 1:
        if arr.shape[0] != ngrids:
            raise ValueError(
                f"{name} must have grid dimension {ngrids}, got {arr.shape[0]}."
            )
        return arr
    if arr.ndim == 2:
        if arr.shape == (2, ngrids):
            return jnp.mean(arr, axis=0)
        if arr.shape == (ngrids, 2):
            return jnp.mean(arr, axis=-1)
        if arr.shape[0] == ngrids:
            return arr[:, 0]
        if arr.shape[1] == ngrids:
            return arr[0, :]
        raise ValueError(
            f"{name} must be (ngrids,), (2, ngrids), (ngrids, 2), "
            f"(ngrids, ncomp), or (ncomp, ngrids). Got {arr.shape}."
        )
    if arr.ndim >= 3:
        if arr.shape[-1] == ngrids:
            return arr.reshape((-1, ngrids))[0]
        if arr.shape[0] == ngrids:
            return arr.reshape((ngrids, -1))[:, 0]
    raise ValueError(
        f"{name} must expose a readable density-density kernel component on {ngrids} grids. "
        f"Got shape {arr.shape}."
    )


def _normalize_response_feature_kind(value: Any) -> str:
    if value is None:
        return "LDA"
    kind = str(value).upper()
    if kind in {"LDA", "GGA", "MGGA", "MGGA_LAPL"}:
        return kind
    return "LDA"


def _grid_xc_potential_components(
    params: PyTree,
    functional: Any,
    molecule: Any,
) -> tuple[Array, Array, Array, Array, str]:
    resolved = _resolved_xc_object(params, functional, molecule)
    component_getter = getattr(resolved, "grid_potential_components", None)
    response_kind = _normalize_response_feature_kind(
        getattr(resolved, "response_feature_kind", None)
    )
    if callable(component_getter):
        components = component_getter(molecule)
        if len(components) == 2:
            v_rho, v_grad = components
            v_tau = jnp.zeros_like(jnp.asarray(v_rho))
            v_lapl = jnp.zeros_like(jnp.asarray(v_rho))
        elif len(components) == 3:
            v_rho, v_grad, v_tau = components
            v_lapl = jnp.zeros_like(jnp.asarray(v_rho))
        elif len(components) == 4:
            v_rho, v_grad, v_tau, v_lapl = components
        else:
            raise ValueError(
                "grid_potential_components must return (v_rho, v_grad) or "
                "(v_rho, v_grad, v_tau) or (v_rho, v_grad, v_tau, v_lapl)."
            )
        v_rho = jnp.asarray(v_rho)
        v_grad = jnp.asarray(v_grad, dtype=v_rho.dtype)
        v_tau = jnp.asarray(v_tau, dtype=v_rho.dtype)
        v_lapl = jnp.asarray(v_lapl, dtype=v_rho.dtype)
        if v_grad.ndim == 2 and v_grad.shape == (3, v_rho.shape[0]):
            v_grad = v_grad.T
        if v_grad.ndim != 2 or v_grad.shape[0] != v_rho.shape[0] or v_grad.shape[1] != 3:
            raise ValueError("v_grad must have shape (ngrids, 3) compatible with v_rho.")
        if v_tau.shape != v_rho.shape:
            raise ValueError("v_tau must have the same shape as v_rho.")
        if v_lapl.shape != v_rho.shape:
            raise ValueError("v_lapl must have the same shape as v_rho.")
        return v_rho, v_grad, v_tau, v_lapl, response_kind

    v_rho = _grid_xc_potential_from_resolved(
        resolved,
        functional=functional,
        params=params,
        molecule=molecule,
    )
    v_rho = jnp.asarray(v_rho)
    v_grad = jnp.zeros(v_rho.shape + (3,), dtype=v_rho.dtype)
    v_tau = jnp.zeros_like(v_rho)
    v_lapl = jnp.zeros_like(v_rho)
    return v_rho, v_grad, v_tau, v_lapl, "LDA"


def _effective_exact_exchange_fraction(
    params: PyTree,
    functional: Any,
    molecule: Any,
) -> Array:
    resolved = _resolved_xc_object(params, functional, molecule)
    exact_exchange_fraction = getattr(resolved, "exact_exchange_fraction", 0.0)
    return jnp.asarray(exact_exchange_fraction)


def density_stationarity_penalty(
    params: PyTree,
    functional: Any,
    molecule: Any,
    *,
    occupation_tolerance: float = 1e-8,
) -> Array:
    """Differentiable density consistency penalty for fixed-density training."""

    if getattr(molecule, "ao", None) is None:
        raise AttributeError("Molecule-like object must define ao.")
    if getattr(molecule, "grid", None) is None:
        raise AttributeError("Molecule-like object must define grid.weights.")
    if getattr(molecule, "h1e", None) is None:
        raise AttributeError("Molecule-like object must define h1e.")
    if getattr(molecule, "rep_tensor", None) is None:
        raise AttributeError("Molecule-like object must define rep_tensor.")

    density_matrix = _spin_summed_density_matrix(molecule)
    ao = jnp.asarray(molecule.ao)
    ao_deriv1 = getattr(molecule, "ao_deriv1", None)
    weights = jnp.asarray(molecule.grid.weights)
    h1e = jnp.asarray(molecule.h1e)
    v_rho, v_grad, v_tau, v_lapl, xc_kind = _grid_xc_potential_components(
        params,
        functional,
        molecule,
    )
    v_rho = jnp.nan_to_num(v_rho, nan=0.0, posinf=0.0, neginf=0.0)
    v_grad = jnp.nan_to_num(v_grad, nan=0.0, posinf=0.0, neginf=0.0)
    kind = _normalize_response_feature_kind(xc_kind)
    if ao_deriv1 is None:
        kind = "LDA"
        ao_deriv1_arr = jnp.zeros((4, ao.shape[0], ao.shape[1]), dtype=ao.dtype)
    else:
        ao_deriv1_arr = jnp.asarray(ao_deriv1)
        if ao_deriv1_arr.shape[0] < 4 and kind in {"GGA", "MGGA", "MGGA_LAPL"}:
            kind = "LDA"
    ao_laplacian = getattr(molecule, "ao_laplacian", None)
    if ao_laplacian is None:
        ao_laplacian_arr = jnp.zeros_like(ao)
        if kind == "MGGA_LAPL":
            kind = "MGGA"
    else:
        ao_laplacian_arr = jnp.asarray(ao_laplacian)
    vxc_matrix = _vxc_matrix_from_grid_potential(
        ao=ao,
        ao_deriv1=ao_deriv1_arr,
        ao_laplacian=ao_laplacian_arr,
        weights=weights,
        vxc_rho=v_rho,
        vxc_grad=v_grad,
        vxc_tau=v_tau,
        vxc_lapl=v_lapl,
        xc_kind=kind,
    )

    j_matrix = _coulomb_potential_from_molecule(molecule, density_matrix)
    k_matrix = _exchange_potential_from_molecule(molecule, density_matrix)
    alpha = _effective_exact_exchange_fraction(params, functional, molecule)
    fock = h1e + j_matrix - 0.5 * alpha * k_matrix + vxc_matrix

    mo_coeff, mo_occ = _restricted_channel(molecule)
    fock_mo = mo_coeff.T.conj() @ fock @ mo_coeff
    occ_mask = (mo_occ > occupation_tolerance).astype(fock_mo.dtype)
    vir_mask = (mo_occ <= occupation_tolerance).astype(fock_mo.dtype)
    ov_mask = occ_mask[:, None] * vir_mask[None, :]
    denominator = jnp.maximum(jnp.sum(ov_mask), 1.0)
    return jnp.sum(jnp.abs(fock_mo * ov_mask) ** 2) / denominator


def dm21_scf_regularization_delta_energy(
    params: PyTree,
    functional: Any,
    molecule: Any,
    *,
    occupation_tolerance: float = 1e-8,
    gap_floor: float = 1e-3,
) -> Array:
    """DM21-style one-step SCF energy change proxy (S6, signed).

    δE_SCF = 1/2 Σ_{i != j} ((n_i - n_j)/(ε_i - ε_j)) |F_ij|^2
    where F is built from the model XC potential but projected in the
    reference molecular-orbital basis C from ``molecule``.
    """

    if getattr(molecule, "ao", None) is None:
        raise AttributeError("Molecule-like object must define ao.")
    if getattr(molecule, "grid", None) is None:
        raise AttributeError("Molecule-like object must define grid.weights.")
    if getattr(molecule, "h1e", None) is None:
        raise AttributeError("Molecule-like object must define h1e.")
    if getattr(molecule, "rep_tensor", None) is None:
        raise AttributeError("Molecule-like object must define rep_tensor.")
    if getattr(molecule, "mo_coeff", None) is None or getattr(molecule, "mo_occ", None) is None:
        raise AttributeError("Molecule-like object must define mo_coeff and mo_occ.")
    if getattr(molecule, "mo_energy", None) is None:
        raise AttributeError("Molecule-like object must define mo_energy.")

    density_matrix = _spin_summed_density_matrix(molecule)
    ao = jnp.asarray(molecule.ao)
    ao_deriv1 = getattr(molecule, "ao_deriv1", None)
    weights = jnp.asarray(molecule.grid.weights)
    h1e = jnp.asarray(molecule.h1e)
    v_rho, v_grad, v_tau, v_lapl, xc_kind = _grid_xc_potential_components(
        params,
        functional,
        molecule,
    )
    v_rho = jnp.nan_to_num(v_rho, nan=0.0, posinf=0.0, neginf=0.0)
    v_grad = jnp.nan_to_num(v_grad, nan=0.0, posinf=0.0, neginf=0.0)
    kind = _normalize_response_feature_kind(xc_kind)
    if ao_deriv1 is None:
        kind = "LDA"
        ao_deriv1_arr = jnp.zeros((4, ao.shape[0], ao.shape[1]), dtype=ao.dtype)
    else:
        ao_deriv1_arr = jnp.asarray(ao_deriv1)
        if ao_deriv1_arr.shape[0] < 4 and kind in {"GGA", "MGGA", "MGGA_LAPL"}:
            kind = "LDA"
    ao_laplacian = getattr(molecule, "ao_laplacian", None)
    if ao_laplacian is None:
        ao_laplacian_arr = jnp.zeros_like(ao)
        if kind == "MGGA_LAPL":
            kind = "MGGA"
    else:
        ao_laplacian_arr = jnp.asarray(ao_laplacian)
    vxc_matrix = _vxc_matrix_from_grid_potential(
        ao=ao,
        ao_deriv1=ao_deriv1_arr,
        ao_laplacian=ao_laplacian_arr,
        weights=weights,
        vxc_rho=v_rho,
        vxc_grad=v_grad,
        vxc_tau=v_tau,
        vxc_lapl=v_lapl,
        xc_kind=kind,
    )

    j_matrix = _coulomb_potential_from_molecule(molecule, density_matrix)
    k_matrix = _exchange_potential_from_molecule(molecule, density_matrix)
    alpha = _effective_exact_exchange_fraction(params, functional, molecule)
    fock = h1e + j_matrix - 0.5 * alpha * k_matrix + vxc_matrix

    mo_coeff, mo_occ, mo_energy = _restricted_channel_with_energies(molecule)
    fock_mo = mo_coeff.T.conj() @ fock @ mo_coeff

    mo_occ = jnp.asarray(mo_occ, dtype=jnp.asarray(fock_mo).real.dtype)
    mo_energy = jnp.asarray(mo_energy, dtype=jnp.asarray(fock_mo).real.dtype)
    delta_n = mo_occ[:, None] - mo_occ[None, :]
    delta_eps = mo_energy[:, None] - mo_energy[None, :]

    gap_floor = jnp.asarray(gap_floor, dtype=delta_eps.dtype)
    gap_floor = jnp.maximum(jnp.abs(gap_floor), 1e-12)
    abs_delta_eps = jnp.abs(delta_eps)
    safe_delta_eps = jnp.where(
        delta_eps >= 0.0,
        jnp.maximum(abs_delta_eps, gap_floor),
        -jnp.maximum(abs_delta_eps, gap_floor),
    )

    offdiag = 1.0 - jnp.eye(fock_mo.shape[0], dtype=delta_eps.dtype)
    occ_mask = (jnp.abs(delta_n) > occupation_tolerance).astype(delta_eps.dtype)
    active = offdiag * occ_mask

    ratio = jnp.where(active > 0.0, delta_n / safe_delta_eps, 0.0)
    fij2 = jnp.abs(fock_mo) ** 2
    delta_e = 0.5 * jnp.sum(active * ratio * fij2)
    return jnp.real(delta_e)


def dm21_scf_regularization_penalty(
    params: PyTree,
    functional: Any,
    molecule: Any,
    *,
    occupation_tolerance: float = 1e-8,
    gap_floor: float = 1e-3,
) -> Array:
    """DM21-style SCF proxy contribution used in the training loss."""

    delta_e = dm21_scf_regularization_delta_energy(
        params,
        functional,
        molecule,
        occupation_tolerance=occupation_tolerance,
        gap_floor=gap_floor,
    )
    return delta_e**2


def density_matching_penalty(
    params: PyTree,
    functional: Any,
    molecule: Any,
    *,
    training_config: MolecularTrainingConfig | None = None,
    self_consistent_molecule: Any | None = None,
    spin_resolved: bool | None = None,
    target_density: Array | None = None,
    target_density_matrix: Array | None = None,
) -> Array:
    """Weighted grid-density MSE between the reference and model self-consistent densities."""

    if getattr(molecule, "grid", None) is None:
        raise AttributeError("Molecule-like object must define grid.weights.")

    cfg = MolecularTrainingConfig() if training_config is None else training_config
    use_spin_resolved = (
        cfg.grid_density_spin == "spin_resolved"
        if spin_resolved is None
        else bool(spin_resolved)
    )
    if target_density is not None:
        reference_density = jnp.asarray(target_density)
    elif target_density_matrix is None:
        reference_density = density_on_grid(molecule)
    else:
        reference_density = _density_on_grid_from_density_matrix(molecule, target_density_matrix)
    model_molecule = self_consistent_molecule
    if model_molecule is None:
        scf_cfg = cfg if cfg.mode == "self_consistent" else replace(cfg, mode="self_consistent")
        model_molecule = _resolve_training_molecule_with_mode(params, functional, molecule, scf_cfg)
    if use_spin_resolved:
        if target_density is not None:
            reference_density = jnp.asarray(target_density)
            if reference_density.ndim == 1:
                reference_density = reference_density[:, None]
        elif target_density_matrix is None:
            reference_density = density_on_grid_spin_resolved(molecule)
        else:
            density_matrix = jnp.asarray(target_density_matrix)
            if density_matrix.ndim == 2:
                reference_density = _density_on_grid_from_density_matrix(
                    molecule,
                    density_matrix,
                )[:, None]
            elif density_matrix.ndim == 3:
                ao = jnp.asarray(molecule.ao)
                reference_density = jnp.einsum("spq,rp,rq->rs", density_matrix, ao, ao)
            else:
                raise ValueError(
                    "Expected target_density_matrix to have shape (nao, nao) or (spin, nao, nao)."
                )
        model_density = density_on_grid_spin_resolved(model_molecule)
        channel_count = max(reference_density.shape[-1], 1)
        residual = jnp.sum((model_density - reference_density) ** 2, axis=-1) / channel_count
    else:
        model_density = density_on_grid(model_molecule)
        residual = (model_density - reference_density) ** 2
    weights = jnp.asarray(molecule.grid.weights)
    normalization = jnp.maximum(jnp.sum(weights), 1e-12)
    return jnp.sum(weights * residual) / normalization


def density_matrix_matching_penalty(
    molecule: Any,
    *,
    self_consistent_molecule: Any | None = None,
    target_density_matrix: Array | None = None,
) -> Array:
    """Mean-squared AO density-matrix error using spin-summed matrices."""

    reference = (
        _spin_summed_density_matrix(molecule)
        if target_density_matrix is None
        else jnp.asarray(target_density_matrix)
    )
    if reference.ndim == 3:
        reference = reference.sum(axis=0)
    model_molecule = molecule if self_consistent_molecule is None else self_consistent_molecule
    model = _spin_summed_density_matrix(model_molecule)
    return jnp.mean((model - reference) ** 2)


def xc_potential_matching_penalty(
    params: PyTree,
    functional: Any,
    molecule: Any,
    *,
    target_xc_potential: Array,
) -> Array:
    """Weighted grid MSE between model v_xc(r) and reference target v_xc(r)."""

    if getattr(molecule, "grid", None) is None:
        raise AttributeError("Molecule-like object must define grid.weights.")

    predicted = jnp.asarray(_grid_xc_potential(params, functional, molecule))
    predicted = jnp.nan_to_num(predicted, nan=0.0, posinf=1e6, neginf=-1e6)
    target = jnp.asarray(target_xc_potential, dtype=predicted.dtype)
    target = jnp.nan_to_num(target, nan=0.0, posinf=1e6, neginf=-1e6)

    if target.ndim == 1:
        if target.shape[0] != predicted.shape[0]:
            raise ValueError(
                "target_xc_potential grid dimension must match predicted v_xc grid dimension "
                f"(got {target.shape[0]} vs {predicted.shape[0]})."
            )
    elif target.ndim == 2:
        # Accept spin-resolved targets as either (2, ngrids) or (ngrids, 2).
        if target.shape == (2, predicted.shape[0]):
            target = jnp.mean(target, axis=0)
        elif target.shape == (predicted.shape[0], 2):
            target = jnp.mean(target, axis=-1)
        else:
            raise ValueError(
                "2D target_xc_potential must be shaped (2, ngrids) or (ngrids, 2). "
                f"Got {target.shape} with ngrids={predicted.shape[0]}."
            )
    else:
        raise ValueError(
            "target_xc_potential must have shape (ngrids,) or spin-resolved (2, ngrids)/(ngrids, 2)."
        )

    residual = (predicted - target) ** 2
    weights = jnp.asarray(molecule.grid.weights)
    normalization = jnp.maximum(jnp.sum(weights), 1e-12)
    return jnp.sum(weights * residual) / normalization


def xc_kernel_matching_penalty(
    params: PyTree,
    functional: Any,
    molecule: Any,
    *,
    target_xc_kernel: Array,
    normalization_scale: float | Array | None = None,
) -> Array:
    """Weighted grid MSE between model f_xc(r) and reference target f_xc(r).

    The residual is normalized by the reference RMS magnitude so this term
    remains numerically trainable when reference kernels span very large scales.
    """

    if getattr(molecule, "grid", None) is None:
        raise AttributeError("Molecule-like object must define grid.weights.")

    predicted_raw = _grid_xc_kernel(params, functional, molecule)
    predicted = _as_grid_density_density_component(
        predicted_raw,
        int(jnp.asarray(molecule.grid.weights).shape[0]),
        name="predicted f_xc",
    )
    predicted = jnp.nan_to_num(predicted, nan=0.0, posinf=1e12, neginf=-1e12)

    target_raw = jnp.asarray(target_xc_kernel, dtype=predicted.dtype)
    target = _as_grid_density_density_component(
        target_raw,
        int(predicted.shape[0]),
        name="target_xc_kernel",
    )
    target = jnp.nan_to_num(target, nan=0.0, posinf=1e12, neginf=-1e12)

    weights = jnp.asarray(molecule.grid.weights)
    normalization = jnp.maximum(jnp.sum(weights), 1e-12)
    if normalization_scale is None:
        scale = jnp.sqrt(jnp.sum(weights * (target**2)) / normalization)
    else:
        scale = jnp.asarray(normalization_scale, dtype=predicted.dtype)
    scale = jnp.maximum(jnp.abs(scale), 1e-8)
    residual = ((predicted - target) / scale) ** 2
    return jnp.sum(weights * residual) / normalization


def coefficient_prior_penalty(
    params: PyTree,
    functional: Any,
    molecule: Any,
    *,
    prior_values: Sequence[float],
    mode: str = "pointwise",
) -> Array:
    """Weighted grid MSE that keeps Neural_xc channel coefficients near a prior."""

    if getattr(molecule, "grid", None) is None:
        raise AttributeError("Molecule-like object must define grid.weights.")
    if not hasattr(functional, "channel_coefficients"):
        raise AttributeError("XC functional must expose channel_coefficients(...) for a coefficient prior.")
    if not hasattr(functional, "semilocal_energy_density_channels"):
        raise AttributeError(
            "XC functional must expose semilocal_energy_density_channels(...) for a coefficient prior."
        )
    if not hasattr(functional, "projected_hf_energy_density_components"):
        raise AttributeError(
            "XC functional must expose projected_hf_energy_density_components(...) for a coefficient prior."
        )

    features = grid_features_for_molecule(molecule)
    coefficient_inputs = functional.compute_coefficient_inputs(
        molecule,
        features=features,
    )
    coefficients = functional.channel_coefficients_from_inputs(params, coefficient_inputs)
    prior = jnp.asarray(tuple(prior_values), dtype=coefficients.dtype)
    if prior.ndim != 1 or prior.shape[0] != coefficients.shape[-1]:
        raise ValueError(
            "coefficient_prior_values must match the Neural_xc channel dimension "
            f"(got {prior.shape}, expected ({coefficients.shape[-1]},))."
        )
    weights = jnp.asarray(molecule.grid.weights)
    normalization = jnp.maximum(jnp.sum(weights), 1e-12)
    if mode == "pointwise":
        residual = jnp.mean((coefficients - prior) ** 2, axis=-1)
        return jnp.sum(weights * residual) / normalization
    if mode == "mean":
        coeff_mean = jnp.sum(weights[:, None] * coefficients, axis=0) / normalization
        return jnp.mean((coeff_mean - prior) ** 2)
    raise ValueError(
        f"Unsupported coefficient_prior_mode={mode!r}. Expected 'pointwise' or 'mean'."
    )


def fractional_charge_linearity_penalty(
    params: PyTree,
    functional: Any,
    molecule: Any,
    *,
    delta: float = 0.1,
    training_config: MolecularTrainingConfig | None = None,
    assume_self_consistent_input: bool = False,
) -> Array:
    """Piecewise-linearity proxy from variational fractional frontier states."""

    clipped_delta = jnp.clip(jnp.asarray(delta), 1e-3, 0.49)
    base_molecule = (
        molecule
        if assume_self_consistent_input
        else _resolve_training_molecule_with_mode(
            params,
            functional,
            molecule,
            _as_self_consistent_training_config(training_config),
        )
    )
    frozen_functional, frozen_params = _freeze_functional_for_fractional_path(
        params,
        functional,
        base_molecule,
    )

    mol_m2, info_m2 = _resolve_variational_frontier_state_and_info(
        frozen_params,
        frozen_functional,
        base_molecule,
        homo_delta=-2.0 * clipped_delta,
        training_config=training_config,
    )
    mol_m1, info_m1 = _resolve_variational_frontier_state_and_info(
        frozen_params,
        frozen_functional,
        base_molecule,
        homo_delta=-clipped_delta,
        training_config=training_config,
    )
    mol_p1, info_p1 = _resolve_variational_frontier_state_and_info(
        frozen_params,
        frozen_functional,
        base_molecule,
        lumo_delta=clipped_delta,
        training_config=training_config,
    )
    mol_p2, info_p2 = _resolve_variational_frontier_state_and_info(
        frozen_params,
        frozen_functional,
        base_molecule,
        lumo_delta=2.0 * clipped_delta,
        training_config=training_config,
    )

    e_m2 = _predict_ground_state_total_energy_from_molecule(frozen_params, frozen_functional, mol_m2)
    e_m1 = _predict_ground_state_total_energy_from_molecule(frozen_params, frozen_functional, mol_m1)
    e_0 = _predict_ground_state_total_energy_from_molecule(frozen_params, frozen_functional, base_molecule)
    e_p1 = _predict_ground_state_total_energy_from_molecule(frozen_params, frozen_functional, mol_p1)
    e_p2 = _predict_ground_state_total_energy_from_molecule(frozen_params, frozen_functional, mol_p2)

    remove_curvature = e_0 - 2.0 * e_m1 + e_m2
    add_curvature = e_p2 - 2.0 * e_p1 + e_0
    remove_weight = jnp.minimum(
        _fractional_branch_quality_weight(info_m2, training_config, dtype=remove_curvature.dtype),
        _fractional_branch_quality_weight(info_m1, training_config, dtype=remove_curvature.dtype),
    )
    add_weight = jnp.minimum(
        _fractional_branch_quality_weight(info_p1, training_config, dtype=add_curvature.dtype),
        _fractional_branch_quality_weight(info_p2, training_config, dtype=add_curvature.dtype),
    )
    weights = jnp.stack([remove_weight, add_weight], axis=0)
    curvatures = jnp.stack([remove_curvature**2, add_curvature**2], axis=0)
    normalization = jnp.maximum(jnp.sum(weights), 1e-8)
    return jnp.sum(weights * curvatures) / normalization


def _resolve_training_molecule_with_mode(
    params: PyTree,
    functional: Any,
    molecule: Any,
    training_config: MolecularTrainingConfig | None,
) -> Any:
    molecule_out, _ = _resolve_training_molecule_and_info_with_mode(
        params,
        functional,
        molecule,
        training_config,
    )
    return molecule_out


def _resolve_training_molecule_and_info_with_mode(
    params: PyTree,
    functional: Any,
    molecule: Any,
    training_config: MolecularTrainingConfig | None,
) -> tuple[Any, Any]:
    scf = _make_differentiable_scf(training_config)
    return scf.run(molecule, functional, params)


def _make_differentiable_scf(
    training_config: MolecularTrainingConfig | None,
) -> DifferentiableSCF:
    cfg = MolecularTrainingConfig() if training_config is None else training_config
    return DifferentiableSCF(
        DifferentiableSCFConfig(
            mode=cfg.mode,
            gradient_mode=cfg.scf_gradient_mode,
            max_cycle=cfg.scf_max_cycle,
            damping=cfg.scf_damping,
            level_shift=cfg.scf_level_shift,
            conv_tol_energy=cfg.scf_conv_tol_energy,
            convergence_metric=cfg.scf_convergence_metric,
            occupation_tolerance=cfg.occupation_tolerance,
            conv_tol_density=cfg.scf_conv_tol_density,
            orthogonalization_eps=cfg.scf_orthogonalization_eps,
            eigenvalue_jitter=cfg.scf_eigenvalue_jitter,
            vxc_clip=cfg.scf_vxc_clip,
            iterate_selection=cfg.scf_iterate_selection,
            implicit_diff_max_iter=cfg.scf_implicit_diff_max_iter,
            implicit_diff_clip=cfg.scf_implicit_diff_clip,
            implicit_diff_tolerance=cfg.scf_implicit_diff_tolerance,
            implicit_diff_regularization=cfg.scf_implicit_diff_regularization,
        )
    )


def _predict_ground_state_total_energy_from_molecule(
    params: PyTree,
    functional: Any,
    molecule: Any,
) -> Array:
    if getattr(molecule, "grid", None) is None:
        raise AttributeError("Molecule-like object must define grid.weights.")
    if getattr(molecule, "h1e", None) is None:
        raise AttributeError("Molecule-like object must define h1e.")
    if getattr(molecule, "rep_tensor", None) is None:
        raise AttributeError("Molecule-like object must define rep_tensor.")
    if getattr(molecule, "nuclear_repulsion", None) is None:
        raise AttributeError("Molecule-like object must define nuclear_repulsion.")

    density_matrix = _spin_summed_density_matrix(molecule)
    non_xc = (
        _one_body_energy(density_matrix, molecule.h1e)
        + _coulomb_energy_from_molecule(molecule, density_matrix)
        + jnp.asarray(molecule.nuclear_repulsion)
    )
    if hasattr(functional, "energy_from_molecule"):
        xc = functional.energy_from_molecule(params, molecule)
    else:
        total_density = density_on_grid(molecule)
        xc = functional.energy(params, total_density, molecule.grid.weights)
    return non_xc + xc


def predict_ground_state_total_energy(
    params: PyTree,
    functional: Any,
    molecule: Any,
    *,
    training_config: MolecularTrainingConfig | None = None,
) -> Array:
    """Predict ground-state total energy with fixed-density or self-consistent mode."""

    eval_molecule = _resolve_training_molecule_with_mode(
        params,
        functional,
        molecule,
        training_config,
    )
    return _predict_ground_state_total_energy_from_molecule(params, functional, eval_molecule)


def _has_weight(config: MolecularTrainingConfig, *names: str) -> bool:
    return any(float(getattr(config, name)) != 0.0 for name in names)


def _validate_training_datum(
    datum: MolecularTrainingDatum,
    config: MolecularTrainingConfig,
) -> None:
    requirements = (
        (
            _has_weight(config, "e0_total_mse_weight", "e0_total_mae_weight")
            or config.self_consistent_e0_weight != 0.0,
            datum.target_e0_total_h,
            "target_e0_total_h",
        ),
        (
            config.grid_density_mse_weight != 0.0,
            datum.target_grid_density,
            "target_grid_density",
        ),
        (
            config.density_matrix_mse_weight != 0.0,
            datum.target_density_matrix,
            "target_density_matrix",
        ),
        (
            config.xc_potential_mse_weight != 0.0,
            datum.target_xc_potential,
            "target_xc_potential",
        ),
        (
            config.xc_kernel_mse_weight != 0.0,
            datum.target_xc_kernel,
            "target_xc_kernel",
        ),
        (
            _has_weight(config, "orbital_energy_mse_weight", "orbital_energy_mae_weight"),
            datum.target_orbital_energies,
            "target_orbital_energies",
        ),
        (
            _has_weight(config, "s1_total_mse_weight", "s1_total_mae_weight"),
            datum.target_s1_total_h,
            "target_s1_total_h",
        ),
        (
            _has_weight(
                config,
                "excitation_gap_mse_weight",
                "excitation_gap_mae_weight",
            ),
            datum.target_excitation_gaps_h,
            "target_excitation_gaps_h",
        ),
        (
            _has_weight(
                config,
                "oscillator_strength_mse_weight",
                "oscillator_strength_mae_weight",
            ),
            datum.target_oscillator_strengths,
            "target_oscillator_strengths",
        ),
        (
            _has_weight(config, "spectrum_mse_weight", "spectrum_mae_weight"),
            datum.target_spectrum_curve,
            "target_spectrum_curve",
        ),
    )
    for active, target, name in requirements:
        if active and target is None:
            raise ValueError(f"{name} is required by the active loss weights.")
    if _has_weight(config, "spectrum_mse_weight", "spectrum_mae_weight"):
        if datum.target_spectrum_grid_ev is None:
            raise ValueError(
                "target_spectrum_grid_ev is required by the active spectrum loss weights."
            )


def _has_active_objective(config: MolecularTrainingConfig) -> bool:
    names = (
        "e0_total_mse_weight",
        "e0_total_mae_weight",
        "grid_density_mse_weight",
        "density_matrix_mse_weight",
        "xc_potential_mse_weight",
        "xc_kernel_mse_weight",
        "density_stationarity_weight",
        "dm21_scf_regularization_weight",
        "orbital_energy_mse_weight",
        "orbital_energy_mae_weight",
        "coefficient_prior_weight",
        "fractional_linearity_weight",
        "s1_total_mse_weight",
        "s1_total_mae_weight",
        "excitation_gap_mse_weight",
        "excitation_gap_mae_weight",
        "oscillator_strength_mse_weight",
        "oscillator_strength_mae_weight",
        "spectrum_mse_weight",
        "spectrum_mae_weight",
    )
    return _has_weight(config, *names)


def _datum_dtype(datum: MolecularTrainingDatum) -> Any:
    for name in (
        "target_e0_total_h",
        "target_s1_total_h",
        "target_excitation_gaps_h",
        "target_grid_density",
        "target_density_matrix",
        "target_xc_potential",
        "target_xc_kernel",
        "target_orbital_energies",
        "target_oscillator_strengths",
        "target_spectrum_curve",
    ):
        value = getattr(datum, name)
        if value is not None:
            return jnp.asarray(value).dtype
    return jnp.float64


def molecular_loss(
    params: PyTree,
    functional: Any,
    data: MolecularTrainingDatum | Sequence[MolecularTrainingDatum],
    *,
    training_config: MolecularTrainingConfig,
    predictor: Callable[[PyTree, Any], tuple[Array, Any]] | None = None,
) -> tuple[Array, dict[str, Array]]:
    """Evaluate one explicit molecular objective through a single SCF/TD path."""

    dataset = _as_dataset(data)
    if not dataset:
        raise ValueError("dataset must not be empty.")
    if not _has_active_objective(training_config):
        raise ValueError("MolecularTrainingConfig has no active loss component.")
    if sum(float(datum.weight) for datum in dataset) <= 0.0:
        raise ValueError("dataset must contain positive total sample weight.")

    config = training_config
    response_options = _response_kernel_options_from_excited_config(config)
    metric_terms = _new_metric_terms((*_MOLECULAR_METRIC_KEYS, *_SCF_METRIC_KEYS))
    component_totals = dict.fromkeys(_LOSS_COMPONENT_KEYS, 0.0)
    total_weight = 0.0

    for datum in dataset:
        _validate_training_datum(datum, config)
        dtype = _datum_dtype(datum)
        zero = jnp.asarray(0.0, dtype=dtype)
        needs_e0 = _needs_predicted_ground_state_energy(config)
        scf_info = None

        if predictor is None:
            eval_molecule, eval_info = _resolve_training_molecule_and_info_with_mode(
                params,
                functional,
                datum.molecule,
                config,
            )
            if str(eval_info.mode).startswith("self_consistent"):
                scf_info = eval_info
            predicted_e0_h = (
                _predict_ground_state_total_energy_from_molecule(
                    params,
                    functional,
                    eval_molecule,
                )
                if needs_e0
                else None
            )
        else:
            predicted_value, eval_molecule = predictor(params, datum.molecule)
            predicted_e0_h = jnp.asarray(predicted_value) if needs_e0 else None

        self_consistent_molecule = eval_molecule if config.mode == "self_consistent" else None
        needs_self_consistent = (
            config.grid_density_mse_weight != 0.0
            or config.density_matrix_mse_weight != 0.0
            or config.self_consistent_e0_weight != 0.0
            or _has_weight(
                config,
                "orbital_energy_mse_weight",
                "orbital_energy_mae_weight",
            )
        )
        if self_consistent_molecule is None and needs_self_consistent:
            self_consistent_molecule, scf_info = (
                _resolve_training_molecule_and_info_with_mode(
                    params,
                    functional,
                    datum.molecule,
                    replace(config, mode="self_consistent"),
                )
            )

        e0_mse = e0_mae = normalized_e0_mse = normalized_e0_mae = zero
        component_losses = dict.fromkeys(_LOSS_COMPONENT_KEYS, zero)
        if _has_weight(config, "e0_total_mse_weight", "e0_total_mae_weight"):
            target_e0_h = jnp.asarray(datum.target_e0_total_h, dtype=dtype)
            residual = jnp.asarray(predicted_e0_h, dtype=dtype) - target_e0_h
            scale = _energy_normalization_scale(eval_molecule, config)
            normalized_residual = residual / scale
            e0_mse = residual**2
            e0_mae = jnp.abs(residual)
            normalized_e0_mse = normalized_residual**2
            normalized_e0_mae = jnp.abs(normalized_residual)
            component_losses["e0_total_loss"] = (
                config.e0_total_mse_weight * normalized_e0_mse
                + config.e0_total_mae_weight * normalized_e0_mae
            )

        grid_density_mse = zero
        if config.grid_density_mse_weight != 0.0:
            grid_density_mse = density_matching_penalty(
                params,
                functional,
                datum.molecule,
                training_config=config,
                self_consistent_molecule=self_consistent_molecule,
                target_density=datum.target_grid_density,
            )
            component_losses["grid_density_loss"] = (
                config.grid_density_mse_weight * grid_density_mse
            )

        density_matrix_mse = zero
        if config.density_matrix_mse_weight != 0.0:
            density_matrix_mse = density_matrix_matching_penalty(
                datum.molecule,
                self_consistent_molecule=self_consistent_molecule,
                target_density_matrix=datum.target_density_matrix,
            )
            component_losses["density_matrix_loss"] = (
                config.density_matrix_mse_weight * density_matrix_mse
            )

        xc_potential_mse = zero
        if config.xc_potential_mse_weight != 0.0:
            xc_potential_mse = xc_potential_matching_penalty(
                params,
                functional,
                datum.molecule,
                target_xc_potential=datum.target_xc_potential,
            )
            component_losses["xc_potential_loss"] = (
                config.xc_potential_mse_weight * xc_potential_mse
            )

        xc_kernel_mse = zero
        if config.xc_kernel_mse_weight != 0.0:
            xc_kernel_mse = xc_kernel_matching_penalty(
                params,
                functional,
                datum.molecule,
                target_xc_kernel=datum.target_xc_kernel,
                normalization_scale=datum.target_xc_kernel_normalization_scale,
            )
            component_losses["xc_kernel_loss"] = (
                config.xc_kernel_mse_weight * xc_kernel_mse
            )

        self_consistent_e0_mse = self_consistent_e0_mae = zero
        if config.self_consistent_e0_weight != 0.0:
            self_consistent_e0_h = _predict_ground_state_total_energy_from_molecule(
                params,
                functional,
                self_consistent_molecule,
            )
            residual = self_consistent_e0_h - jnp.asarray(
                datum.target_e0_total_h,
                dtype=dtype,
            )
            scale = _energy_normalization_scale(self_consistent_molecule, config)
            residual = residual / scale
            self_consistent_e0_mse = residual**2
            self_consistent_e0_mae = jnp.abs(residual)
            component_losses["self_consistent_e0_loss"] = (
                config.self_consistent_e0_weight
                * (
                    config.e0_total_mse_weight * self_consistent_e0_mse
                    + config.e0_total_mae_weight * self_consistent_e0_mae
                )
            )

        orbital_energy_mse = orbital_energy_mae = zero
        if _has_weight(config, "orbital_energy_mse_weight", "orbital_energy_mae_weight"):
            target_occ = (
                datum.target_orbital_occupations
                if datum.target_orbital_occupations is not None
                else datum.molecule.mo_occ
            )
            orbital_energy_mse, orbital_energy_mae, _, _ = (
                orbital_energy_matching_penalty(
                    self_consistent_molecule,
                    target_orbital_energies=datum.target_orbital_energies,
                    target_orbital_occupations=target_occ,
                    window=config.orbital_energy_window,
                    occupation_tolerance=config.occupation_tolerance,
                )
            )
            component_losses["orbital_energy_loss"] = (
                config.orbital_energy_mse_weight * orbital_energy_mse
                + config.orbital_energy_mae_weight * orbital_energy_mae
            )

        coefficient_prior_mse = zero
        if config.coefficient_prior_weight != 0.0:
            if config.coefficient_prior_values is None:
                raise ValueError(
                    "coefficient_prior_values is required when coefficient_prior_weight is nonzero."
                )
            coefficient_prior_mse = coefficient_prior_penalty(
                params,
                functional,
                datum.molecule,
                prior_values=config.coefficient_prior_values,
                mode=config.coefficient_prior_mode,
            )
            component_losses["coefficient_prior_loss"] = (
                config.coefficient_prior_weight * coefficient_prior_mse
            )

        if config.density_stationarity_weight != 0.0:
            component_losses["density_stationarity_loss"] = (
                config.density_stationarity_weight
                * density_stationarity_penalty(params, functional, datum.molecule)
            )

        dm21_scf_delta = dm21_scf_mse = zero
        if config.dm21_scf_regularization_weight != 0.0:
            dm21_scf_delta = dm21_scf_regularization_delta_energy(
                params,
                functional,
                datum.molecule,
                occupation_tolerance=config.occupation_tolerance,
                gap_floor=config.dm21_scf_gap_floor,
            )
            dm21_scf_mse = dm21_scf_delta**2
            component_losses["dm21_scf_loss"] = (
                config.dm21_scf_regularization_weight * dm21_scf_mse
            )

        if config.fractional_linearity_weight != 0.0:
            component_losses["fractional_linearity_loss"] = (
                config.fractional_linearity_weight
                * fractional_charge_linearity_penalty(
                    params,
                    functional,
                    eval_molecule,
                    delta=config.fractional_linearity_delta,
                    training_config=config,
                    assume_self_consistent_input=(config.mode == "self_consistent"),
                )
            )

        excited_cache: dict[
            tuple[int, bool, ResponseKernelOptions, str],
            dict[str, Any],
        ] = {}

        def get_excited_state_observables(
            requested_nstates: int,
            use_tda: bool,
            *,
            need_strengths: bool = False,
            tda_gradient_mode: str = "eigenvalue_only",
        ) -> tuple[Array, Array]:
            key = (
                int(requested_nstates),
                bool(use_tda),
                response_options,
                tda_gradient_mode,
            )
            entry = excited_cache.get(key)
            if entry is None:
                gradient_kwargs = {}
                if tda_gradient_mode == "implicit_eigenvector":
                    gradient_kwargs = {
                        "tda_gradient_mode": tda_gradient_mode,
                        "tda_eigenvector_adjoint_tolerance": (
                            config.tda_eigenvector_adjoint_tolerance
                        ),
                        "tda_eigenvector_adjoint_max_iter": (
                            config.tda_eigenvector_adjoint_max_iter
                        ),
                    }
                result = _solve_excited_states(
                    params,
                    functional,
                    eval_molecule,
                    nstates=requested_nstates,
                    use_tda=use_tda,
                    response_kernel_options=response_options,
                    **gradient_kwargs,
                )
                entry = {
                    "result": result,
                    "energies": jnp.asarray(result.excitation_energies),
                    "strengths": None,
                }
                excited_cache[key] = entry
            energies = jnp.asarray(entry["energies"])
            if not need_strengths:
                return energies, jnp.array([], dtype=energies.dtype)
            if entry["strengths"] is None:
                entry["strengths"] = jnp.asarray(
                    oscillator_strengths(
                        eval_molecule,
                        entry["result"],
                        occupation_tolerance=config.occupation_tolerance,
                    )
                )
            return energies, jnp.asarray(entry["strengths"])

        excited_terms = _excited_state_terms(
            predicted_e0_total_h=predicted_e0_h,
            datum=datum,
            config=config,
            get_excited_state_observables=get_excited_state_observables,
        )
        for key in (
            "s1_total_loss",
            "excitation_gap_loss",
            "oscillator_strength_loss",
            "spectrum_loss",
        ):
            component_losses[key] = excited_terms[key]

        weight = jnp.asarray(datum.weight, dtype=dtype)
        total_weight += weight
        for key, value in component_losses.items():
            component_totals[key] = component_totals[key] + weight * value

        metric_values = {
            "e0_total_mse": e0_mse,
            "e0_total_mae": e0_mae,
            "normalized_e0_total_mse": normalized_e0_mse,
            "normalized_e0_total_mae": normalized_e0_mae,
            "grid_density_mse": grid_density_mse,
            "density_matrix_mse": density_matrix_mse,
            "xc_potential_mse": xc_potential_mse,
            "xc_kernel_mse": xc_kernel_mse,
            "self_consistent_e0_mse": self_consistent_e0_mse,
            "self_consistent_e0_mae": self_consistent_e0_mae,
            "orbital_energy_mse": orbital_energy_mse,
            "orbital_energy_mae": orbital_energy_mae,
            "coefficient_prior_mse": coefficient_prior_mse,
            "dm21_scf_mse": dm21_scf_mse,
            "dm21_scf_delta_energy": dm21_scf_delta,
            "predicted_e0_total_h": (
                jnp.atleast_1d(predicted_e0_h)
                if predicted_e0_h is not None
                else jnp.array([], dtype=dtype)
            ),
        }
        metric_values.update(
            {key: value for key, value in excited_terms.items() if key not in _LOSS_COMPONENT_KEYS}
        )
        _append_metric_terms(metric_terms, metric_values)

        if scf_info is not None and str(scf_info.mode).startswith("self_consistent"):
            for key, attr in _SCF_METRIC_ATTRS:
                _append_metric_term(
                    metric_terms,
                    key,
                    jnp.asarray(getattr(scf_info, attr), dtype=dtype),
                )

    denominator = jnp.asarray(total_weight)
    component_means = {
        key: jnp.asarray(value) / denominator for key, value in component_totals.items()
    }
    total_loss = sum(component_means.values(), start=jnp.asarray(0.0))
    metrics = {"total_loss": total_loss}
    metrics.update(
        {key: jnp.atleast_1d(value) for key, value in component_means.items()}
    )
    for key in (*_MOLECULAR_METRIC_KEYS, *_SCF_METRIC_KEYS):
        metrics[key] = _concat_metric_terms(metric_terms[key], empty_dtype=total_loss.dtype)
    for summary_key, source_key, reducer in _SCF_SUMMARY_METRICS:
        values = metrics[source_key]
        metrics[summary_key] = (
            _mean_or_nan(values, dtype=total_loss.dtype)
            if reducer == "mean"
            else _max_or_nan(values, dtype=total_loss.dtype)
        )
    return total_loss, metrics


def _build_excited_state_solver(
    params: PyTree,
    functional: Any,
    molecule: Any,
    *,
    use_tda: bool,
    response_kernel_options: ResponseKernelOptions | dict[str, Any] | None = None,
    tda_gradient_mode: str = "eigenvalue_only",
    tda_eigenvector_adjoint_tolerance: float = 1e-6,
    tda_eigenvector_adjoint_max_iter: int = 64,
) -> Any:
    if tda_gradient_mode not in {"eigenvalue_only", "implicit_eigenvector"}:
        raise ValueError(f"Unsupported TDA gradient mode {tda_gradient_mode!r}.")
    if tda_gradient_mode == "implicit_eigenvector" and not bool(use_tda):
        raise ValueError(
            "implicit_eigenvector is currently supported only for TDA "
            "oscillator-strength training."
        )
    solver_cls = tdscf.TDA if bool(use_tda) else tdscf.TDDFT
    eigensolver = "davidson" if _tree_contains_jax_tracer(params) else "auto"
    solver_kwargs = dict(
        xc_functional=functional,
        xc_params=params,
        eigensolver=eigensolver,
        response_kernel_options=response_kernel_options,
    )
    if tda_gradient_mode == "implicit_eigenvector":
        solver_kwargs.update(
            tda_gradient_mode=tda_gradient_mode,
            eigenvector_adjoint_tol=tda_eigenvector_adjoint_tolerance,
            eigenvector_adjoint_max_iter=tda_eigenvector_adjoint_max_iter,
        )
    return solver_cls(molecule, **solver_kwargs)


def _solve_excited_states(
    params: PyTree,
    functional: Any,
    molecule: Any,
    *,
    nstates: int,
    use_tda: bool,
    response_kernel_options: ResponseKernelOptions | dict[str, Any] | None = None,
    tda_gradient_mode: str = "eigenvalue_only",
    tda_eigenvector_adjoint_tolerance: float = 1e-6,
    tda_eigenvector_adjoint_max_iter: int = 64,
) -> Any:
    solver = _build_excited_state_solver(
        params,
        functional,
        molecule,
        use_tda=use_tda,
        response_kernel_options=response_kernel_options,
        tda_gradient_mode=tda_gradient_mode,
        tda_eigenvector_adjoint_tolerance=tda_eigenvector_adjoint_tolerance,
        tda_eigenvector_adjoint_max_iter=tda_eigenvector_adjoint_max_iter,
    )
    return solver.kernel(nstates=nstates)


def predict_excitation_energies(
    params: PyTree,
    functional: Any,
    molecule: Any,
    *,
    nstates: int = 1,
    use_tda: bool = False,
    response_kernel_options: ResponseKernelOptions | dict[str, Any] | None = None,
) -> Array:
    """Use the trained ground-state XC functional for excited-state TDDFT."""

    result = _solve_excited_states(
        params,
        functional,
        molecule,
        nstates=nstates,
        use_tda=use_tda,
        response_kernel_options=response_kernel_options,
    )
    return result.excitation_energies


def predict_oscillator_strengths(
    params: PyTree,
    functional: Any,
    molecule: Any,
    *,
    nstates: int = 1,
    use_tda: bool = True,
    occupation_tolerance: float = 1e-8,
    response_kernel_options: ResponseKernelOptions | dict[str, Any] | None = None,
    tda_gradient_mode: str = "eigenvalue_only",
    tda_eigenvector_adjoint_tolerance: float = 1e-6,
    tda_eigenvector_adjoint_max_iter: int = 64,
) -> Array:
    """Predict oscillator strengths for the lowest excited states."""

    result = _solve_excited_states(
        params,
        functional,
        molecule,
        nstates=nstates,
        use_tda=use_tda,
        response_kernel_options=response_kernel_options,
        tda_gradient_mode=tda_gradient_mode,
        tda_eigenvector_adjoint_tolerance=tda_eigenvector_adjoint_tolerance,
        tda_eigenvector_adjoint_max_iter=tda_eigenvector_adjoint_max_iter,
    )
    return oscillator_strengths(
        molecule,
        result,
        occupation_tolerance=occupation_tolerance,
    )


def predict_excitation_spectrum(
    params: PyTree,
    functional: Any,
    molecule: Any,
    *,
    grid_ev: Array,
    nstates: int = 1,
    use_tda: bool = True,
    eta_ev: float = 0.15,
    occupation_tolerance: float = 1e-8,
    response_kernel_options: ResponseKernelOptions | dict[str, Any] | None = None,
) -> Array:
    """Predict a broadened absorption spectrum on a fixed energy grid."""

    result = _solve_excited_states(
        params,
        functional,
        molecule,
        nstates=nstates,
        use_tda=use_tda,
        response_kernel_options=response_kernel_options,
    )
    energies_ev = jnp.asarray(result.excitation_energies) * HARTREE_TO_EV
    strengths = oscillator_strengths(
        molecule,
        result,
        occupation_tolerance=occupation_tolerance,
    )
    return lorentzian_spectrum(
        energies_ev,
        strengths,
        jnp.asarray(grid_ev),
        eta=eta_ev,
    )
