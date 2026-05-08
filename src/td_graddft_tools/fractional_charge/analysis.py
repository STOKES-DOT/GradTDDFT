from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array

from td_graddft.training import GroundStateTrainingConfig, predict_ground_state_total_energy


FractionalChargeEnergyEvaluator = Callable[[Any], Array]
_ANALYSIS_DTYPE = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


@dataclass(frozen=True)
class FractionalChargeAnalysisConfig:
    """Configuration for piecewise-linearity scans over fractional electron number."""

    charge_min: float = -1.0
    charge_max: float = 1.0
    num_points: int = 21
    occupation_tolerance: float = 1e-8


@dataclass(frozen=True)
class FractionalChargeAnalysisResult:
    charge_deltas: Array
    electron_counts: Array
    energies_ha: Array
    piecewise_linear_energies_ha: Array
    deviation_ha: Array
    homo_occupations: Array
    lumo_occupations: Array
    max_abs_deviation_ha: float
    mean_abs_deviation_ha: float
    rms_deviation_ha: float
    left_endpoint_slope_ha: float
    right_endpoint_slope_ha: float


def _replace_object(obj: Any, **updates: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return replace(obj, **updates)
    cloned = copy.copy(obj)
    for key, value in updates.items():
        setattr(cloned, key, value)
    return cloned


def _restricted_occ_vector(molecule: Any) -> Array:
    mo_occ = jnp.asarray(molecule.mo_occ)
    if mo_occ.ndim == 1:
        return mo_occ
    if mo_occ.ndim == 2 and mo_occ.shape[0] == 2:
        return mo_occ.sum(axis=0)
    raise ValueError("Fractional-charge analysis currently supports restricted orbitals only.")


def _frontier_indices(
    occupations: Array,
    *,
    occupation_tolerance: float,
) -> tuple[int, int]:
    occ = jnp.asarray(occupations)
    idx = jnp.arange(occ.shape[0], dtype=jnp.int32)
    occ_mask = occ > occupation_tolerance
    vir_mask = occ <= occupation_tolerance
    if not bool(jnp.any(occ_mask)):
        raise ValueError("Need at least one occupied orbital for fractional-charge analysis.")
    if not bool(jnp.any(vir_mask)):
        raise ValueError("Need at least one virtual orbital for fractional-charge analysis.")
    homo_idx = int(jnp.max(jnp.where(occ_mask, idx, jnp.asarray(-1, dtype=idx.dtype))))
    lumo_idx = int(jnp.min(jnp.where(vir_mask, idx, jnp.asarray(occ.shape[0], dtype=idx.dtype))))
    return homo_idx, lumo_idx


def _rebuild_density_matrix(molecule: Any, mo_occ: Array) -> Array:
    mo_coeff = jnp.asarray(molecule.mo_coeff)
    occ = jnp.asarray(mo_occ)
    if mo_coeff.ndim == 2:
        if occ.ndim != 1:
            raise ValueError("Restricted orbital coefficients expect 1D occupations.")
        return jnp.einsum("pi,i,qi->pq", mo_coeff, occ, mo_coeff)
    if mo_coeff.ndim == 3 and mo_coeff.shape[0] == 2:
        if occ.ndim == 1:
            occ = jnp.stack([0.5 * occ, 0.5 * occ], axis=0)
        if occ.ndim != 2 or occ.shape[0] != 2:
            raise ValueError("Spin-resolved restricted orbitals expect shape (2, nmo).")
        return jnp.stack(
            [
                jnp.einsum("pi,i,qi->pq", mo_coeff[0], occ[0], mo_coeff[0]),
                jnp.einsum("pi,i,qi->pq", mo_coeff[1], occ[1], mo_coeff[1]),
            ],
            axis=0,
        )
    raise ValueError("Unsupported mo_coeff shape for fractional-charge analysis.")


def make_fractional_frontier_molecule(
    molecule: Any,
    charge_delta: float,
    *,
    occupation_tolerance: float = 1e-8,
) -> Any:
    """Return a shallow molecule copy with fractional charge on HOMO/LUMO only.

    Negative `charge_delta` removes electrons from the HOMO; positive values add
    electrons to the LUMO. This follows the usual piecewise-linearity analysis
    setup more closely than uniformly scaling all occupations.
    """

    occ_total = _restricted_occ_vector(molecule)
    homo_idx, lumo_idx = _frontier_indices(
        occ_total,
        occupation_tolerance=occupation_tolerance,
    )
    delta = jnp.asarray(charge_delta, dtype=occ_total.dtype)

    if jnp.asarray(molecule.mo_occ).ndim == 1:
        updated_occ = occ_total
        if float(delta) < 0.0:
            updated_occ = updated_occ.at[homo_idx].set(occ_total[homo_idx] + delta)
        else:
            updated_occ = updated_occ.at[lumo_idx].set(occ_total[lumo_idx] + delta)
        if float(updated_occ[homo_idx]) < -1e-8 or float(updated_occ[lumo_idx]) > 2.0 + 1e-8:
            raise ValueError("Requested fractional charge exceeds restricted HOMO/LUMO capacity.")
        density_occ = updated_occ
    else:
        mo_occ = jnp.asarray(molecule.mo_occ)
        updated_occ = mo_occ
        half_delta = 0.5 * delta
        if float(delta) < 0.0:
            updated_occ = updated_occ.at[:, homo_idx].set(mo_occ[:, homo_idx] + half_delta)
        else:
            updated_occ = updated_occ.at[:, lumo_idx].set(mo_occ[:, lumo_idx] + half_delta)
        if (
            float(jnp.min(updated_occ[:, homo_idx])) < -1e-8
            or float(jnp.max(updated_occ[:, lumo_idx])) > 1.0 + 1e-8
        ):
            raise ValueError("Requested fractional charge exceeds spin-channel HOMO/LUMO capacity.")
        density_occ = updated_occ

    rdm1 = _rebuild_density_matrix(molecule, density_occ)
    updates: dict[str, Any] = {
        "mo_occ": density_occ,
        "rdm1": rdm1,
    }
    for attr in ("electron_count", "nelectron"):
        if hasattr(molecule, attr):
            current = getattr(molecule, attr)
            if current is not None:
                updates[attr] = jnp.asarray(current) + delta
    return _replace_object(molecule, **updates)


def make_neural_xc_energy_evaluator(
    params: Any,
    functional: Any,
    *,
    training_config: GroundStateTrainingConfig | None = None,
) -> FractionalChargeEnergyEvaluator:
    """Wrap the standard ground-state predictor as a molecule -> energy callable."""

    def evaluate(molecule: Any) -> Array:
        return predict_ground_state_total_energy(
            params,
            functional,
            molecule,
            training_config=training_config,
        )

    return evaluate


def _charge_grid(config: FractionalChargeAnalysisConfig) -> Array:
    if int(config.num_points) < 3:
        raise ValueError("num_points must be at least 3.")
    charge_min = float(config.charge_min)
    charge_max = float(config.charge_max)
    if charge_min >= 0.0 or charge_max <= 0.0:
        raise ValueError("charge_min must be < 0 and charge_max must be > 0.")
    charges = jnp.linspace(charge_min, charge_max, int(config.num_points))
    if not bool(jnp.any(jnp.isclose(charges, 0.0, atol=1e-12))):
        charges = jnp.sort(jnp.concatenate([charges, jnp.array([0.0])]))
    return charges


def analyze_fractional_charge_linearity(
    molecule: Any,
    energy_evaluator: FractionalChargeEnergyEvaluator,
    config: FractionalChargeAnalysisConfig | None = None,
) -> FractionalChargeAnalysisResult:
    """Evaluate a piecewise-linearity curve along fractional HOMO/LUMO charging."""

    cfg = FractionalChargeAnalysisConfig() if config is None else config
    charge_deltas = _charge_grid(cfg)
    base_occ = _restricted_occ_vector(molecule)
    homo_idx, lumo_idx = _frontier_indices(
        base_occ,
        occupation_tolerance=cfg.occupation_tolerance,
    )
    base_electron_count = float(jnp.sum(base_occ))

    scan_molecules = [
        make_fractional_frontier_molecule(
            molecule,
            float(delta),
            occupation_tolerance=cfg.occupation_tolerance,
        )
        for delta in charge_deltas
    ]
    energies = jnp.asarray([energy_evaluator(mol) for mol in scan_molecules], dtype=_ANALYSIS_DTYPE)
    electron_counts = jnp.asarray(
        [base_electron_count + float(delta) for delta in charge_deltas],
        dtype=_ANALYSIS_DTYPE,
    )
    homo_occ = jnp.asarray(
        [_restricted_occ_vector(mol)[homo_idx] for mol in scan_molecules],
        dtype=_ANALYSIS_DTYPE,
    )
    lumo_occ = jnp.asarray(
        [_restricted_occ_vector(mol)[lumo_idx] for mol in scan_molecules],
        dtype=_ANALYSIS_DTYPE,
    )

    zero_idx = int(jnp.argmin(jnp.abs(charge_deltas)))
    left_idx = 0
    right_idx = int(charge_deltas.shape[0] - 1)
    q_left = charge_deltas[left_idx]
    q_zero = charge_deltas[zero_idx]
    q_right = charge_deltas[right_idx]
    e_left = energies[left_idx]
    e_zero = energies[zero_idx]
    e_right = energies[right_idx]

    left_denom = jnp.where(jnp.abs(q_left - q_zero) < 1e-12, -1.0, q_left - q_zero)
    right_denom = jnp.where(jnp.abs(q_right - q_zero) < 1e-12, 1.0, q_right - q_zero)
    piecewise = jnp.where(
        charge_deltas <= q_zero,
        e_zero + (charge_deltas - q_zero) * (e_left - e_zero) / left_denom,
        e_zero + (charge_deltas - q_zero) * (e_right - e_zero) / right_denom,
    )
    deviation = energies - piecewise
    abs_deviation = jnp.abs(deviation)

    left_slope = float((e_zero - e_left) / (q_zero - q_left))
    right_slope = float((e_right - e_zero) / (q_right - q_zero))
    return FractionalChargeAnalysisResult(
        charge_deltas=charge_deltas,
        electron_counts=electron_counts,
        energies_ha=energies,
        piecewise_linear_energies_ha=piecewise,
        deviation_ha=deviation,
        homo_occupations=homo_occ,
        lumo_occupations=lumo_occ,
        max_abs_deviation_ha=float(jnp.max(abs_deviation)),
        mean_abs_deviation_ha=float(jnp.mean(abs_deviation)),
        rms_deviation_ha=float(jnp.sqrt(jnp.mean(deviation**2))),
        left_endpoint_slope_ha=left_slope,
        right_endpoint_slope_ha=right_slope,
    )
