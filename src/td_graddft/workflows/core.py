from __future__ import annotations

import functools
import time
import warnings
from contextlib import nullcontext
from dataclasses import replace
from dataclasses import dataclass
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
import optax

from td_graddft import neural_xc, tdscf
from td_graddft.device import put_reference_on_device, resolve_execution_device
from td_graddft.jax_runtime import configure_jax_persistent_cache
from td_graddft.reference import (
    restricted_reference_from_spec_with_jax_rks,
    unrestricted_reference_from_spec_with_jax_uks,
)
from td_graddft.reference_legacy import (
    restricted_reference_from_pyscf,
    restricted_reference_from_pyscf_with_jax_rhf,
    restricted_reference_from_pyscf_with_jax_rks,
    unrestricted_reference_from_pyscf,
    unrestricted_reference_from_pyscf_with_jax_uks,
)
from td_graddft.scf import RHFConfig, RKSConfig, UKSConfig
from td_graddft.spectra import HARTREE_TO_EV, lorentzian_spectrum, oscillator_strengths
from td_graddft.training import (
    ExcitedStateDatum,
    ExcitedStateTrainingConfig,
    GroundStateCoreDatum,
    GroundStateCoreTrainingConfig,
    GroundStateDatum,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    ground_state_mse_loss,
    make_ground_state_train_step,
    predict_ground_state_total_energy,
)

from .types import (
    NeuralExcitedStateRun,
    NeuralXCTrainingConfig,
    ReferenceSpecConfig,
    ReferenceRun,
    SimulationConfig,
    SpectrumGridConfig,
    SpectrumRun,
    TrainingRun,
)


def _tree_all_finite(tree: Any) -> bool:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return True
    return all(bool(jnp.all(jnp.isfinite(jnp.asarray(leaf)))) for leaf in leaves)


def _loss_and_metrics_all_finite(loss: Any, metrics: dict[str, Any]) -> bool:
    return bool(jnp.all(jnp.isfinite(jnp.asarray(loss)))) and _tree_all_finite(metrics)


@functools.lru_cache(maxsize=1)
def _compiled_lorentzian_spectrum():
    return jax.jit(lorentzian_spectrum, static_argnames=("eta",))


def _resolve_training_scf_gradient_mode(
    config: NeuralXCTrainingConfig,
) -> Literal["unrolled", "implicit_commutator"]:
    if config.scf_gradient_mode in {"unrolled", "implicit_commutator"}:
        return config.scf_gradient_mode
    requires_stable_scf_gradients = (
        float(config.density_constraint_weight) != 0.0
        or float(config.stationarity_constraint_weight) != 0.0
        or str(config.training_mode) == "self_consistent"
    )
    return "implicit_commutator" if requires_stable_scf_gradients else "unrolled"


def _canonicalize_graddft_ground_state_config(
    config: NeuralXCTrainingConfig,
) -> NeuralXCTrainingConfig:
    """Return a GradDFT-aligned training configuration.

    `graddft_core_defaults=True` applies the GradDFT-style neural-functional and
    training defaults while still allowing excited-state supervision on top of
    the same differentiable DFT/TDDFT backbone.

    `strict_graddft_ground_state=True` further restricts the workflow to the
    original ground-state-only GradDFT envelope. The strict path mirrors
    GradDFT's ground-state setup while keeping the current project-wide
    MAE-only training objective:
    - local coefficient-basis XC ansatz
    - DM21-style original feature inputs
    - residual DM21 mixing network
    - MAE-only energy objective with optional density supervision
    - no excited-state / orbital / auxiliary regularization terms
    """

    use_graddft_core = bool(config.graddft_core_defaults) or bool(
        config.strict_graddft_ground_state
    )
    if not use_graddft_core:
        return config

    aligned = replace(
        config,
        network_architecture="graddft_residual",
        energy_mode="graddft_coeff_basis",
        input_feature_mode="dm21_original",
        hf_input_mode="spin_resolved",
        response_hf_mode="nonlocal_exchange_only",
        response_pt2_mode="local_projected",
        density_supervision="spin_resolved",
        strict_dm21_feature_alignment=True,
        dm21_hfx_channels=max(int(config.dm21_hfx_channels), 2),
        energy_mse_weight=0.0,
        energy_mae_weight=1.0,
        orbital_energy_mse_weight=0.0,
        orbital_energy_mae_weight=1.0,
    )

    if not bool(config.strict_graddft_ground_state):
        return aligned

    forbidden_weights = {
        "stationarity_constraint_weight": float(aligned.stationarity_constraint_weight),
        "dm21_scf_regularization_weight": float(aligned.dm21_scf_regularization_weight),
        "orbital_energy_constraint_weight": float(aligned.orbital_energy_constraint_weight),
        "janak_frontier_constraint_weight": float(aligned.janak_frontier_constraint_weight),
        "coefficient_prior_weight": float(aligned.coefficient_prior_weight),
        "s1_constraint_weight": float(aligned.s1_constraint_weight),
        "excitation_constraint_weight": float(aligned.excitation_constraint_weight),
        "oscillator_strength_constraint_weight": float(
            aligned.oscillator_strength_constraint_weight
        ),
        "spectrum_constraint_weight": float(aligned.spectrum_constraint_weight),
        "fractional_linearity_weight": float(aligned.fractional_linearity_weight),
    }
    active_forbidden = {name: value for name, value in forbidden_weights.items() if value != 0.0}
    if active_forbidden:
        details = ", ".join(f"{name}={value}" for name, value in active_forbidden.items())
        raise ValueError(
            "strict_graddft_ground_state only supports ground-state energy/density "
            f"training. Disable the following options first: {details}."
        )
    return aligned


def _is_unrestricted_reference(reference: Any) -> bool:
    mo_coeff = jnp.asarray(reference.mo_coeff)
    if mo_coeff.ndim != 3 or mo_coeff.shape[0] != 2:
        return False
    mo_occ = jnp.asarray(reference.mo_occ)
    mo_energy = jnp.asarray(reference.mo_energy)
    return not (
        bool(jnp.allclose(mo_coeff[0], mo_coeff[1]))
        and bool(jnp.allclose(mo_occ[0], mo_occ[1]))
        and bool(jnp.allclose(mo_energy[0], mo_energy[1]))
    )


def _reference_state_dimensions(reference: Any, occupation_tolerance: float) -> tuple[int, int, int]:
    mo_occ = jnp.asarray(reference.mo_occ)
    mo_coeff = jnp.asarray(reference.mo_coeff)
    if mo_occ.ndim == 1:
        nocc = int(jnp.count_nonzero(mo_occ > occupation_tolerance))
        nvir = int(mo_coeff.shape[-1] - nocc)
        return nocc, nvir, nocc * nvir
    if mo_occ.ndim == 2 and mo_occ.shape[0] == 2 and _is_unrestricted_reference(reference):
        nocc_a = int(jnp.count_nonzero(mo_occ[0] > occupation_tolerance))
        nvir_a = int(mo_coeff.shape[-1] - nocc_a)
        nocc_b = int(jnp.count_nonzero(mo_occ[1] > occupation_tolerance))
        nvir_b = int(mo_coeff.shape[-1] - nocc_b)
        return nocc_a + nocc_b, nvir_a + nvir_b, nocc_a * nvir_a + nocc_b * nvir_b
    nocc = int(jnp.count_nonzero(mo_occ[0] > occupation_tolerance))
    nvir = int(mo_coeff.shape[-1] - nocc)
    return nocc, nvir, nocc * nvir


def _mf_state_capacity(mf: Any, occupation_tolerance: float) -> int:
    mf_mo_occ = jnp.asarray(getattr(mf, "mo_occ"))
    mf_mo_coeff = jnp.asarray(getattr(mf, "mo_coeff"))
    if mf_mo_occ.ndim == 1:
        mf_nocc = int(jnp.count_nonzero(mf_mo_occ > occupation_tolerance))
        mf_nvir = int(mf_mo_coeff.shape[-1] - mf_nocc)
        return mf_nocc * mf_nvir
    if mf_mo_occ.ndim == 2 and mf_mo_occ.shape[0] == 2:
        mf_nocc_a = int(jnp.count_nonzero(mf_mo_occ[0] > occupation_tolerance))
        mf_nvir_a = int(mf_mo_coeff.shape[-1] - mf_nocc_a)
        mf_nocc_b = int(jnp.count_nonzero(mf_mo_occ[1] > occupation_tolerance))
        mf_nvir_b = int(mf_mo_coeff.shape[-1] - mf_nocc_b)
        return mf_nocc_a * mf_nvir_a + mf_nocc_b * mf_nvir_b
    raise ValueError("Unsupported PySCF orbital occupation shape for TDDFT capacity inference.")


def _build_reference(
    mf: Any,
    simulation: SimulationConfig,
    *,
    compute_local_hfx_features: bool = False,
    compute_local_hfx_aux: bool = False,
    compute_local_pt2_features: bool = False,
    hfx_omega_values: tuple[float, ...] = (0.0, 0.4),
    hfx_chunk_size: int = 512,
):
    backend = simulation.scf_backend.lower()
    if backend == "pyscf":
        if int(getattr(mf.mol, "spin", 0)) != 0:
            return unrestricted_reference_from_pyscf(mf)
        return restricted_reference_from_pyscf(
            mf,
            compute_local_hfx_features=compute_local_hfx_features,
            compute_local_hfx_aux=compute_local_hfx_aux,
            compute_local_pt2_features=compute_local_pt2_features,
            hfx_omega_values=hfx_omega_values,
            hfx_chunk_size=hfx_chunk_size,
        )
    if backend == "jax_rhf":
        if int(getattr(mf.mol, "spin", 0)) != 0:
            raise ValueError("jax_rhf backend only supports closed-shell references.")
        rhf_config = RHFConfig(
            max_cycle=simulation.jax_rhf_max_cycle,
            conv_tol=simulation.jax_rhf_conv_tol,
            conv_tol_density=simulation.jax_rhf_conv_tol_density,
        )
        return restricted_reference_from_pyscf_with_jax_rhf(
            mf,
            max_l=simulation.jax_basis_max_l,
            rhf_config=rhf_config,
            grid_ao_backend=simulation.jax_grid_ao_backend,
            compute_local_hfx_features=compute_local_hfx_features,
            compute_local_hfx_aux=compute_local_hfx_aux,
            compute_local_pt2_features=compute_local_pt2_features,
            hfx_omega_values=hfx_omega_values,
            hfx_chunk_size=hfx_chunk_size,
            # Keep fixed-density training target aligned with the PySCF reference run.
            energy_target=float(getattr(mf, "e_tot", jnp.nan)),
        )
    if backend == "jax_rks":
        if int(getattr(mf.mol, "spin", 0)) != 0:
            raise ValueError("jax_rks backend only supports closed-shell references.")
        rks_config = RKSConfig(
            xc_spec=simulation.jax_rks_xc_spec or str(getattr(mf, "xc", "pbe")),
            max_cycle=simulation.jax_rks_max_cycle,
            conv_tol=simulation.jax_rks_conv_tol,
            conv_tol_density=simulation.jax_rks_conv_tol_density,
            damping=simulation.jax_rks_damping,
            density_floor=simulation.jax_rks_density_floor,
            potential_clip=simulation.jax_rks_potential_clip,
            jk_backend=simulation.jax_rks_jk_backend,
            df_tol=simulation.jax_rks_df_tol,
            df_max_rank=simulation.jax_rks_df_max_rank,
        )
        return restricted_reference_from_pyscf_with_jax_rks(
            mf,
            max_l=simulation.jax_basis_max_l,
            rks_config=rks_config,
            xc_spec=simulation.jax_rks_xc_spec,
            grid_ao_backend=simulation.jax_grid_ao_backend,
            compute_local_hfx_features=compute_local_hfx_features,
            compute_local_hfx_aux=compute_local_hfx_aux,
            compute_local_pt2_features=compute_local_pt2_features,
            hfx_omega_values=hfx_omega_values,
            hfx_chunk_size=hfx_chunk_size,
            # Keep fixed-density training target aligned with the PySCF reference run.
            energy_target=float(getattr(mf, "e_tot", jnp.nan)),
        )
    if backend == "jax_uks":
        uks_config = UKSConfig(
            xc_spec=simulation.jax_uks_xc_spec or str(getattr(mf, "xc", "pbe")),
            max_cycle=simulation.jax_uks_max_cycle,
            conv_tol=simulation.jax_uks_conv_tol,
            conv_tol_density=simulation.jax_uks_conv_tol_density,
            damping=simulation.jax_uks_damping,
            density_floor=simulation.jax_uks_density_floor,
            potential_clip=simulation.jax_uks_potential_clip,
        )
        return unrestricted_reference_from_pyscf_with_jax_uks(
            mf,
            max_l=simulation.jax_basis_max_l,
            uks_config=uks_config,
            xc_spec=simulation.jax_uks_xc_spec,
            grid_ao_backend=simulation.jax_grid_ao_backend,
            compute_local_hfx_features=compute_local_hfx_features,
            compute_local_hfx_aux=compute_local_hfx_aux,
            hfx_omega_values=hfx_omega_values,
            hfx_chunk_size=hfx_chunk_size,
            # Keep fixed-density training target aligned with the PySCF reference run.
            energy_target=float(getattr(mf, "e_tot", jnp.nan)),
        )
    raise ValueError(
        f"Unsupported scf_backend={simulation.scf_backend!r}. "
        "Expected 'pyscf', 'jax_rhf', 'jax_rks', or 'jax_uks'."
    )


def run_reference(
    mf: Any,
    *,
    scf_elapsed_s: float,
    simulation: SimulationConfig,
    compute_local_hfx_features: bool = False,
    compute_local_hfx_aux: bool = False,
    compute_local_pt2_features: bool = False,
    hfx_omega_values: tuple[float, ...] = (0.0, 0.4),
    hfx_chunk_size: int = 512,
) -> ReferenceRun:
    """Run PySCF reference TDDFT and prepare a device-ready molecule reference."""

    t0_ref = time.perf_counter()
    exec_device = resolve_execution_device(simulation.execution_device)
    device_context = jax.default_device(exec_device) if exec_device is not None else nullcontext()
    with device_context:
        reference = _build_reference(
            mf,
            simulation,
            compute_local_hfx_features=compute_local_hfx_features,
            compute_local_hfx_aux=compute_local_hfx_aux,
            compute_local_pt2_features=compute_local_pt2_features,
            hfx_omega_values=hfx_omega_values,
            hfx_chunk_size=hfx_chunk_size,
        )
    if simulation.move_reference_to_device and exec_device is not None:
        reference = put_reference_on_device(reference, device=exec_device)
    reference_build_elapsed_s = time.perf_counter() - t0_ref
    nocc, nvir, nstates_full = _reference_state_dimensions(
        reference,
        simulation.occupation_tolerance,
    )
    mf_nstates_full = _mf_state_capacity(mf, simulation.occupation_tolerance)
    nstates_cap = min(nstates_full, mf_nstates_full)
    nstates = nstates_cap if simulation.nstates <= 0 else min(simulation.nstates, nstates_cap)

    if nstates <= 0:
        energies = jnp.array([])
        strengths = jnp.array([])
        elapsed = 0.0
    else:
        t0 = time.perf_counter()
        td = mf.TDDFT()
        td.nstates = nstates
        try:
            td.kernel()
        except Exception:
            td = mf.TDA()
            td.nstates = nstates
            td.kernel()
        elapsed = time.perf_counter() - t0
        energies = jnp.asarray(td.e)
        strengths = jnp.asarray(td.oscillator_strength())

    return ReferenceRun(
        molecule=reference,
        nocc=nocc,
        nvir=nvir,
        nstates=nstates,
        nstates_full=nstates_full,
        energies_au=energies,
        oscillator_strengths=strengths,
        scf_elapsed_s=scf_elapsed_s + reference_build_elapsed_s,
        tddft_elapsed_s=elapsed,
    )


def run_reference_from_spec(
    spec: ReferenceSpecConfig,
    *,
    simulation: SimulationConfig,
    compute_local_hfx_features: bool = False,
    compute_local_hfx_aux: bool = False,
    compute_local_pt2_features: bool = False,
    hfx_omega_values: tuple[float, ...] = (0.0, 0.4),
    hfx_chunk_size: int = 512,
) -> ReferenceRun:
    """Build a strict-JAX reference and excited states directly from molecule specs."""

    backend = str(simulation.scf_backend).lower()
    if backend not in {"jax_rks", "jax_uks"}:
        raise NotImplementedError(
            "run_reference_from_spec currently supports simulation.scf_backend in "
            "{'jax_rks', 'jax_uks'} only."
        )

    configure_jax_persistent_cache(
        cache_dir=simulation.jax_compilation_cache_dir,
        min_compile_time_secs=simulation.jax_persistent_cache_min_compile_time_secs,
        min_entry_size_bytes=simulation.jax_persistent_cache_min_entry_size_bytes,
    )

    t0_ref = time.perf_counter()
    exec_device = resolve_execution_device(simulation.execution_device)
    device_context = jax.default_device(exec_device) if exec_device is not None else nullcontext()
    with device_context:
        if backend == "jax_rks":
            rks_xc = simulation.jax_rks_xc_spec or str(spec.xc)
            rks_config = RKSConfig(
                xc_spec=rks_xc,
                max_cycle=simulation.jax_rks_max_cycle,
                conv_tol=simulation.jax_rks_conv_tol,
                conv_tol_density=simulation.jax_rks_conv_tol_density,
                damping=simulation.jax_rks_damping,
                density_floor=simulation.jax_rks_density_floor,
                potential_clip=simulation.jax_rks_potential_clip,
                jk_backend=simulation.jax_rks_jk_backend,
                df_tol=simulation.jax_rks_df_tol,
                df_max_rank=simulation.jax_rks_df_max_rank,
            )
            reference_kwargs = dict(
                atom=spec.atom,
                basis=spec.basis,
                xc_spec=rks_xc,
                unit=spec.unit,
                charge=spec.charge,
                spin=spec.spin,
                cart=spec.cart,
                grids_level=spec.grids_level,
                max_l=simulation.jax_basis_max_l,
                rks_config=rks_config,
                grid_ao_backend=simulation.jax_grid_ao_backend,
                integral_backend=simulation.jax_integral_backend,
                libcint_geometry_grad_policy=simulation.jax_libcint_geometry_grad_policy,
                compute_local_hfx_features=compute_local_hfx_features,
                compute_local_hfx_aux=compute_local_hfx_aux,
                compute_local_pt2_features=compute_local_pt2_features,
                hfx_omega_values=hfx_omega_values,
                hfx_chunk_size=hfx_chunk_size,
                precompile_eri=simulation.jax_precompile_eri,
                precompile_eri_chunk_size=simulation.jax_precompile_eri_chunk_size,
                verbose=spec.verbose,
            )
            reference = restricted_reference_from_spec_with_jax_rks(**reference_kwargs)
            xc_label = rks_xc
        else:
            uks_xc = simulation.jax_uks_xc_spec or str(spec.xc)
            uks_config = UKSConfig(
                xc_spec=uks_xc,
                max_cycle=simulation.jax_uks_max_cycle,
                conv_tol=simulation.jax_uks_conv_tol,
                conv_tol_density=simulation.jax_uks_conv_tol_density,
                damping=simulation.jax_uks_damping,
                density_floor=simulation.jax_uks_density_floor,
                potential_clip=simulation.jax_uks_potential_clip,
            )
            reference = unrestricted_reference_from_spec_with_jax_uks(
                atom=spec.atom,
                basis=spec.basis,
                xc_spec=uks_xc,
                unit=spec.unit,
                charge=spec.charge,
                spin=spec.spin,
                cart=spec.cart,
                grids_level=spec.grids_level,
                max_l=simulation.jax_basis_max_l,
                uks_config=uks_config,
                grid_ao_backend=simulation.jax_grid_ao_backend,
                integral_backend=simulation.jax_integral_backend,
                libcint_geometry_grad_policy=simulation.jax_libcint_geometry_grad_policy,
                compute_local_hfx_features=compute_local_hfx_features,
                compute_local_hfx_aux=compute_local_hfx_aux,
                hfx_omega_values=hfx_omega_values,
                hfx_chunk_size=hfx_chunk_size,
                precompile_eri=simulation.jax_precompile_eri,
                precompile_eri_chunk_size=simulation.jax_precompile_eri_chunk_size,
                verbose=spec.verbose,
            )
            xc_label = uks_xc
    if simulation.move_reference_to_device and exec_device is not None:
        reference = put_reference_on_device(reference, device=exec_device)
    scf_elapsed = time.perf_counter() - t0_ref

    nocc, nvir, nstates_full = _reference_state_dimensions(
        reference,
        simulation.occupation_tolerance,
    )
    nstates = nstates_full if simulation.nstates <= 0 else min(simulation.nstates, nstates_full)

    if nstates <= 0:
        energies = jnp.array([])
        strengths = jnp.array([])
        elapsed = 0.0
    else:
        td = tdscf.TDDFT(reference, xc_functional=xc_label)
        tda = tdscf.TDA(reference, xc_functional=xc_label)
        kernel_fn = td.kernel
        tda_fn = tda.kernel
        if simulation.jit_tddft:
            kernel_fn = jax.jit(kernel_fn, static_argnames=("nstates",))
            tda_fn = jax.jit(tda_fn, static_argnames=("nstates",))

        t0_td = time.perf_counter()
        with device_context:
            try:
                result = kernel_fn(nstates=nstates)
            except Exception:
                result = tda_fn(nstates=nstates)
            energies = jnp.asarray(result.excitation_energies)
            strengths = jnp.asarray(
                oscillator_strengths(
                    reference,
                    result,
                    occupation_tolerance=simulation.occupation_tolerance,
                )
            )
        valid = jnp.isfinite(energies) & jnp.isfinite(strengths) & (energies > 0.0)
        energies = energies[valid]
        strengths = strengths[valid]
        elapsed = time.perf_counter() - t0_td

    return ReferenceRun(
        molecule=reference,
        nocc=nocc,
        nvir=nvir,
        nstates=nstates,
        nstates_full=nstates_full,
        energies_au=energies,
        oscillator_strengths=strengths,
        scf_elapsed_s=scf_elapsed,
        tddft_elapsed_s=elapsed,
    )


def train_neural_xc(
    reference: ReferenceRun,
    config: NeuralXCTrainingConfig,
    spectrum_config: SpectrumGridConfig,
) -> TrainingRun:
    """Train Neural_xc against fixed-density ground-state total energy."""

    config = _canonicalize_graddft_ground_state_config(config)

    coefficient_prior_values = (
        neural_xc.resolve_coefficient_prior_values(
            config.semilocal_xc,
            config.coefficient_prior_values,
            energy_mode=config.energy_mode,
        )
        if (
            config.coefficient_prior_values is not None
            or float(config.coefficient_prior_weight) != 0.0
        )
        else None
    )
    functional = neural_xc.Functional(
        architecture=config.network_architecture,
        semilocal_xc=config.semilocal_xc,
        n_semilocal_channels=config.n_semilocal_channels,
        energy_mode=config.energy_mode,
        input_feature_mode=config.input_feature_mode,
        hf_input_mode=config.hf_input_mode,
        hf_fraction_mode=config.hf_fraction_mode,
        include_pt2_channel=config.include_pt2_channel,
        pt2_channel_mode=config.pt2_channel_mode,
        response_hf_mode=config.response_hf_mode,
        response_pt2_mode=config.response_pt2_mode,
        strict_dm21_feature_alignment=config.strict_dm21_feature_alignment,
        hidden_dims=config.hidden_dims,
        squash_offset=config.squash_offset,
        sigmoid_scale_factor=config.sigmoid_scale_factor,
        dm21_hfx_channels=config.dm21_hfx_channels,
        name=config.functional_name,
    )
    s1_target = None
    if float(config.s1_constraint_weight) != 0.0:
        if config.s1_target_energy_au is not None:
            s1_target = jnp.asarray(float(config.s1_target_energy_au))
        elif int(reference.energies_au.size) > 0:
            s1_target = jnp.asarray(reference.energies_au[0])
        else:
            raise ValueError(
                "s1_constraint_weight != 0 but no reference excited states are available. "
                "Set simulation.nstates >= 1 or provide s1_target_energy_au explicitly."
            )
    excitation_targets = None
    if float(config.excitation_constraint_weight) != 0.0:
        if config.excitation_target_energies_au is not None:
            excitation_targets = jnp.asarray(
                tuple(float(value) for value in config.excitation_target_energies_au)
            )
        else:
            ref_energies = jnp.asarray(reference.energies_au)
            if int(ref_energies.size) <= 0:
                raise ValueError(
                    "excitation_constraint_weight != 0 but no reference excited states are "
                    "available. Set simulation.nstates >= 1 or provide "
                    "excitation_target_energies_au explicitly."
                )
            requested_excited_states = max(1, int(config.excitation_constraint_nstates))
            n_take = min(requested_excited_states, int(ref_energies.size))
            excitation_targets = ref_energies[:n_take]
    oscillator_strength_targets = None
    oscillator_strength_nstates = 0
    if float(config.oscillator_strength_constraint_weight) != 0.0:
        ref_energies = jnp.asarray(reference.energies_au)
        ref_strengths = jnp.asarray(reference.oscillator_strengths)
        if int(ref_energies.size) <= 0 or int(ref_strengths.size) <= 0:
            raise ValueError(
                "oscillator_strength_constraint_weight != 0 but no reference oscillator "
                "strengths are available. Set simulation.nstates >= 1 so PySCF returns "
                "excitation energies and oscillator strengths."
            )
        oscillator_strength_nstates = min(
            max(1, int(config.oscillator_strength_constraint_nstates)),
            int(ref_energies.size),
            int(ref_strengths.size),
        )
        oscillator_strength_targets = ref_strengths[:oscillator_strength_nstates]
    spectrum_target_grid_ev = None
    spectrum_target_curve = None
    spectrum_nstates = 0
    if float(config.spectrum_constraint_weight) != 0.0:
        ref_energies = jnp.asarray(reference.energies_au)
        ref_strengths = jnp.asarray(reference.oscillator_strengths)
        if int(ref_energies.size) <= 0 or int(ref_strengths.size) <= 0:
            raise ValueError(
                "spectrum_constraint_weight != 0 but no reference spectrum is available. "
                "Set simulation.nstates >= 1 so PySCF returns excitation energies and oscillator strengths."
            )
        spectrum_nstates = min(
            max(1, int(config.spectrum_constraint_nstates)),
            int(ref_energies.size),
            int(ref_strengths.size),
        )
        spectrum_target_grid_ev = jnp.linspace(
            float(spectrum_config.grid_min_ev),
            float(
                max(
                    spectrum_config.grid_min_ev,
                    spectrum_config.grid_min_ev
                    + spectrum_config.max_padding_ev
                    + float(jnp.max(ref_energies * HARTREE_TO_EV)),
                )
            ),
            int(spectrum_config.grid_points),
        )
        spectrum_target_curve = lorentzian_spectrum(
            ref_energies[:spectrum_nstates] * HARTREE_TO_EV,
            ref_strengths[:spectrum_nstates],
            spectrum_target_grid_ev,
            eta=float(spectrum_config.eta_ev),
        )

    datum_core = GroundStateCoreDatum(
        target_total_energy=jnp.asarray(reference.molecule.mf_energy),
        density_constraint_weight=config.density_constraint_weight,
        stationarity_constraint_weight=config.stationarity_constraint_weight,
        dm21_scf_regularization_weight=config.dm21_scf_regularization_weight,
        target_orbital_energies=jnp.asarray(reference.molecule.mo_energy),
        target_orbital_occupations=jnp.asarray(reference.molecule.mo_occ),
        orbital_energy_constraint_weight=config.orbital_energy_constraint_weight,
        orbital_energy_constraint_window=int(config.orbital_energy_constraint_window),
        janak_frontier_constraint_weight=config.janak_frontier_constraint_weight,
    )
    datum_excited = ExcitedStateDatum(
        target_s1_energy=s1_target,
        s1_constraint_weight=float(config.s1_constraint_weight),
        target_excitation_energies=excitation_targets,
        excitation_constraint_weight=float(config.excitation_constraint_weight),
        excitation_constraint_nstates=int(config.excitation_constraint_nstates),
        target_oscillator_strengths=oscillator_strength_targets,
        oscillator_strength_constraint_weight=float(
            config.oscillator_strength_constraint_weight
        ),
        oscillator_strength_constraint_nstates=(
            oscillator_strength_nstates if oscillator_strength_nstates > 0 else None
        ),
        target_spectrum_grid_ev=spectrum_target_grid_ev,
        target_spectrum_curve=spectrum_target_curve,
        spectrum_constraint_weight=float(config.spectrum_constraint_weight),
        spectrum_constraint_nstates=(spectrum_nstates if spectrum_nstates > 0 else None),
    )
    datum = GroundStateDatum.from_parts(
        reference.molecule,
        core=datum_core,
        excited_state=datum_excited,
    )
    selected_scf_gradient_mode = _resolve_training_scf_gradient_mode(config)
    gs_core_training = GroundStateCoreTrainingConfig(
        mode=config.training_mode,
        energy_mse_weight=config.energy_mse_weight,
        energy_mae_weight=config.energy_mae_weight,
        orbital_energy_mse_weight=config.orbital_energy_mse_weight,
        orbital_energy_mae_weight=config.orbital_energy_mae_weight,
        energy_normalization=config.energy_normalization,
        density_supervision=config.density_supervision,
        coefficient_prior_weight=config.coefficient_prior_weight,
        coefficient_prior_values=coefficient_prior_values,
        coefficient_prior_mode=config.coefficient_prior_mode,
        fractional_linearity_weight=config.fractional_linearity_weight,
        fractional_linearity_delta=config.fractional_linearity_delta,
        janak_frontier_mode=config.janak_frontier_mode,
        janak_frontier_delta=config.janak_frontier_delta,
        dm21_scf_gap_floor=config.dm21_scf_gap_floor,
        occupation_tolerance=1e-8,
        scf_max_cycle=config.scf_max_cycle,
        scf_damping=config.scf_damping,
        scf_conv_tol_density=config.scf_conv_tol_density,
        scf_orthogonalization_eps=config.scf_orthogonalization_eps,
        scf_vxc_clip=config.scf_vxc_clip,
        scf_iterate_selection=config.scf_iterate_selection,
        scf_require_convergence=config.scf_require_convergence,
        scf_stop_gradient_on_unconverged=config.scf_stop_gradient_on_unconverged,
        scf_stop_gradient_rms_threshold=config.scf_stop_gradient_rms_threshold,
        scf_gradient_mode=selected_scf_gradient_mode,
        scf_implicit_diff_max_iter=config.scf_implicit_diff_max_iter,
        scf_implicit_diff_step_size=config.scf_implicit_diff_step_size,
        scf_implicit_diff_clip=config.scf_implicit_diff_clip,
        scf_implicit_diff_solver=config.scf_implicit_diff_solver,
        scf_implicit_diff_tolerance=config.scf_implicit_diff_tolerance,
        scf_implicit_diff_regularization=config.scf_implicit_diff_regularization,
        scf_implicit_diff_restart=config.scf_implicit_diff_restart,
    )
    gs_excited_training = ExcitedStateTrainingConfig(
        s1_constraint_use_tda=bool(config.s1_constraint_use_tda),
        excitation_constraint_use_tda=bool(config.excitation_constraint_use_tda),
        excitation_mse_weight=config.excitation_mse_weight,
        excitation_mae_weight=config.excitation_mae_weight,
        oscillator_strength_constraint_use_tda=bool(
            config.oscillator_strength_constraint_use_tda
        ),
        oscillator_strength_mse_weight=config.oscillator_strength_mse_weight,
        oscillator_strength_mae_weight=config.oscillator_strength_mae_weight,
        spectrum_constraint_use_tda=bool(config.spectrum_constraint_use_tda),
        spectrum_constraint_eta_ev=float(spectrum_config.eta_ev),
        spectrum_mse_weight=config.spectrum_mse_weight,
        spectrum_mae_weight=config.spectrum_mae_weight,
    )
    gs_training = GroundStateTrainingConfig.from_parts(
        core=gs_core_training,
        excited_state=gs_excited_training,
    )
    if config.lr_decay_every > 0:
        lr_schedule = optax.exponential_decay(
            init_value=float(config.learning_rate),
            transition_steps=int(config.lr_decay_every),
            decay_rate=float(config.lr_decay_factor),
            staircase=True,
        )
        base_optimizer = optax.adam(lr_schedule)
    else:
        base_optimizer = optax.adam(config.learning_rate)
    if config.gradient_clip_norm is not None and float(config.gradient_clip_norm) > 0.0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(float(config.gradient_clip_norm)),
            base_optimizer,
        )
    else:
        optimizer = base_optimizer
    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(config.seed),
        reference.molecule,
        optimizer,
    )
    train_step = make_ground_state_train_step(functional, training_config=gs_training)
    fallback_train_step = None
    if config.recover_nonfinite_steps and selected_scf_gradient_mode != "implicit_commutator":
        fallback_training = replace(gs_training, scf_gradient_mode="implicit_commutator")
        fallback_train_step = make_ground_state_train_step(
            functional,
            training_config=fallback_training,
        )
    # Keep per-step TDDFT-supervised overfit stable by default: S1 loss introduces
    # an internal eigen-solve in the objective and is not worth JIT-compiling
    # during rapid prototyping loops.
    use_jit_train = (
        gs_training.mode == "fixed_density"
        and config.jit_train
        and float(config.s1_constraint_weight) == 0.0
        and float(config.excitation_constraint_weight) == 0.0
        and float(config.oscillator_strength_constraint_weight) == 0.0
        and float(config.spectrum_constraint_weight) == 0.0
    )
    if use_jit_train:
        compiled_train_step = jax.jit(lambda current_state: train_step(current_state, datum))
        compiled_eval = jax.jit(
            lambda params: ground_state_mse_loss(
                params,
                functional,
                datum,
                training_config=gs_training,
            )
        )
    else:
        compiled_train_step = lambda current_state: train_step(current_state, datum)
        compiled_eval = lambda params: ground_state_mse_loss(
            params,
            functional,
            datum,
            training_config=gs_training,
        )

    initial_loss, initial_metrics = compiled_eval(state.params)
    initial_loss = float(initial_loss)
    initial_density_penalty = float(initial_metrics["density_penalty"][0])
    initial_stationarity_penalty = float(initial_metrics["stationarity_penalty"][0])
    initial_coefficient_prior_penalty = float(initial_metrics["coefficient_prior_penalty"][0])
    min_loss = initial_loss
    min_loss_step = 0
    loss_history = [initial_loss]
    density_penalty_history = [initial_density_penalty]
    stationarity_penalty_history = [initial_stationarity_penalty]
    coefficient_prior_penalty_history = [initial_coefficient_prior_penalty]
    grad_norm_history = [float("nan")]
    grad_abs_max_history = [float("nan")]
    param_update_norm_history = [float("nan")]
    nonfinite_grad_fraction_history = [0.0]
    best_params = state.params

    def _metric_scalar(metrics: dict[str, Any], key: str, default: float = float("nan")) -> float:
        if key not in metrics:
            return default
        arr = jnp.asarray(metrics[key])
        if int(arr.size) <= 0:
            return default
        return float(arr.reshape(-1)[0])

    t0 = time.perf_counter()
    fallback_recoveries = 0
    guard_post_update = (
        gs_training.mode == "self_consistent"
        and selected_scf_gradient_mode == "implicit_commutator"
    )
    for step in range(1, config.steps + 1):
        prev_state = state
        state, train_metrics = compiled_train_step(state)
        reverted_step = False
        if config.recover_nonfinite_steps and not _tree_all_finite(state.params):
            recovered = False
            if fallback_train_step is not None:
                candidate_state, candidate_metrics = fallback_train_step(prev_state, datum)
                if _tree_all_finite(candidate_state.params):
                    state = candidate_state
                    train_metrics = candidate_metrics
                    recovered = True
            if not recovered:
                state = prev_state
                reverted_step = True
            fallback_recoveries += 1
        if config.recover_nonfinite_steps and guard_post_update and not reverted_step:
            guarded_loss, guarded_metrics = compiled_eval(state.params)
            if not _loss_and_metrics_all_finite(guarded_loss, guarded_metrics):
                state = prev_state
                reverted_step = True
                fallback_recoveries += 1
        grad_norm_val = _metric_scalar(train_metrics, "grad_norm")
        grad_abs_max_val = _metric_scalar(train_metrics, "grad_abs_max")
        param_update_norm_val = (
            0.0 if reverted_step else _metric_scalar(train_metrics, "param_update_norm")
        )
        nonfinite_grad_fraction_val = _metric_scalar(train_metrics, "nonfinite_grad_fraction", 0.0)
        train_loss_val = _metric_scalar(train_metrics, "loss")
        train_density_penalty_val = _metric_scalar(train_metrics, "density_penalty", 0.0)
        train_stationarity_penalty_val = _metric_scalar(train_metrics, "stationarity_penalty", 0.0)
        train_coefficient_prior_penalty_val = _metric_scalar(
            train_metrics,
            "coefficient_prior_penalty",
            0.0,
        )
        grad_norm_history.append(grad_norm_val)
        grad_abs_max_history.append(grad_abs_max_val)
        param_update_norm_history.append(param_update_norm_val)
        nonfinite_grad_fraction_history.append(nonfinite_grad_fraction_val)
        # ``train_metrics`` is evaluated on ``prev_state.params`` before the optimizer
        # update. That corresponds to the post-update state from the previous step, so
        # we record it with a one-step lag and only use a single explicit eval at the
        # very end to cover the final state.
        if step >= 2:
            tracked_step = step - 1
            loss_history.append(train_loss_val)
            density_penalty_history.append(train_density_penalty_val)
            stationarity_penalty_history.append(train_stationarity_penalty_val)
            coefficient_prior_penalty_history.append(train_coefficient_prior_penalty_val)
            if train_loss_val < min_loss:
                min_loss = train_loss_val
                min_loss_step = tracked_step
                best_params = prev_state.params
        if config.log_interval > 0 and (
            step == 1 or step % config.log_interval == 0 or step == config.steps
        ):
            step_loss, step_metrics = compiled_eval(state.params)
            loss_val = float(step_loss)
            density_penalty_val = float(step_metrics["density_penalty"][0])
            stationarity_penalty_val = float(step_metrics["stationarity_penalty"][0])
            dm21_scf_penalty_val = float(step_metrics["dm21_scf_penalty"][0])
            orbital_energy_penalty_val = float(step_metrics["orbital_energy_penalty"][0])
            janak_frontier_penalty_val = float(step_metrics["janak_frontier_penalty"][0])
            coefficient_prior_penalty_val = float(step_metrics["coefficient_prior_penalty"][0])
            s1_penalty_val = float(step_metrics["s1_penalty"][0])
            excitation_penalty_val = float(step_metrics["excitation_penalty"][0])
            oscillator_strength_penalty_val = float(
                step_metrics["oscillator_strength_penalty"][0]
            )
            spectrum_penalty_val = float(step_metrics["spectrum_penalty"][0])
            scf_converged_fraction_val = _metric_scalar(
                step_metrics,
                "scf_converged_fraction",
            )
            scf_cycles_mean_val = _metric_scalar(step_metrics, "scf_cycles_mean")
            scf_cycles_max_val = _metric_scalar(step_metrics, "scf_cycles_max")
            scf_selected_rms_max_val = _metric_scalar(
                step_metrics,
                "scf_selected_rms_max",
            )
            print(
                "[NeuralXCTrain] "
                f"step={step}/{config.steps} "
                f"loss={loss_val:.6e} "
                f"density_penalty={density_penalty_val:.6e} "
                f"stationarity_penalty={stationarity_penalty_val:.6e} "
                f"dm21_scf_penalty={dm21_scf_penalty_val:.6e} "
                f"orbital_energy_penalty={orbital_energy_penalty_val:.6e} "
                f"janak_frontier_penalty={janak_frontier_penalty_val:.6e} "
                f"coefficient_prior_penalty={coefficient_prior_penalty_val:.6e} "
                f"s1_penalty={s1_penalty_val:.6e} "
                f"excitation_penalty={excitation_penalty_val:.6e} "
                f"oscillator_strength_penalty={oscillator_strength_penalty_val:.6e} "
                f"spectrum_penalty={spectrum_penalty_val:.6e} "
                f"scf_conv_frac={scf_converged_fraction_val:.6e} "
                f"scf_cycles_mean={scf_cycles_mean_val:.6e} "
                f"scf_cycles_max={scf_cycles_max_val:.6e} "
                f"scf_selected_rms_max={scf_selected_rms_max_val:.6e} "
                f"grad_norm={grad_norm_val:.6e} "
                f"grad_abs_max={grad_abs_max_val:.6e} "
                f"update_norm={param_update_norm_val:.6e} "
                f"nonfinite_grad_frac={nonfinite_grad_fraction_val:.6e} "
                f"scf_grad_mode={selected_scf_gradient_mode} "
                f"recoveries={fallback_recoveries}"
            )
    elapsed = time.perf_counter() - t0

    final_loss, final_metrics = compiled_eval(state.params)
    final_loss = float(final_loss)
    final_density_penalty = float(final_metrics["density_penalty"][0])
    final_stationarity_penalty = float(final_metrics["stationarity_penalty"][0])
    final_coefficient_prior_penalty = float(final_metrics["coefficient_prior_penalty"][0])
    loss_history.append(final_loss)
    density_penalty_history.append(final_density_penalty)
    stationarity_penalty_history.append(final_stationarity_penalty)
    coefficient_prior_penalty_history.append(final_coefficient_prior_penalty)
    if final_loss < min_loss:
        min_loss = final_loss
        min_loss_step = config.steps
        best_params = state.params

    trained_energy = float(
        predict_ground_state_total_energy(
            best_params,
            functional,
            reference.molecule,
            training_config=gs_training,
        )
    )
    trained_hybrid_fraction = float(
        functional.effective_exchange_fraction(best_params, reference.molecule)
    )

    return TrainingRun(
        functional=functional,
        params=best_params,
        initial_loss=initial_loss,
        final_loss=final_loss,
        min_loss=min_loss,
        min_loss_step=min_loss_step,
        initial_density_penalty=initial_density_penalty,
        final_density_penalty=final_density_penalty,
        initial_stationarity_penalty=initial_stationarity_penalty,
        final_stationarity_penalty=final_stationarity_penalty,
        initial_coefficient_prior_penalty=initial_coefficient_prior_penalty,
        final_coefficient_prior_penalty=final_coefficient_prior_penalty,
        loss_history=loss_history,
        density_penalty_history=density_penalty_history,
        stationarity_penalty_history=stationarity_penalty_history,
        coefficient_prior_penalty_history=coefficient_prior_penalty_history,
        grad_norm_history=grad_norm_history,
        grad_abs_max_history=grad_abs_max_history,
        param_update_norm_history=param_update_norm_history,
        nonfinite_grad_fraction_history=nonfinite_grad_fraction_history,
        trained_energy=trained_energy,
        trained_hybrid_fraction=trained_hybrid_fraction,
        elapsed_s=elapsed,
    )


def run_neural_tddft(
    reference: ReferenceRun,
    training: TrainingRun,
    simulation: SimulationConfig,
) -> NeuralExcitedStateRun:
    """Evaluate excited states with the trained Neural_xc functional."""

    t0 = time.perf_counter()
    exec_device = resolve_execution_device(simulation.execution_device)
    device_context = jax.default_device(exec_device) if exec_device is not None else nullcontext()
    td = tdscf.TDDFT(
        reference.molecule,
        xc_functional=training.functional,
        xc_params=training.params,
    )
    tda = tdscf.TDA(
        reference.molecule,
        xc_functional=training.functional,
        xc_params=training.params,
    )
    kernel_fn = td.kernel
    tda_fn = tda.kernel
    if simulation.jit_tddft:
        kernel_fn = jax.jit(kernel_fn, static_argnames=("nstates",))
        tda_fn = jax.jit(tda_fn, static_argnames=("nstates",))

    def _sanitize_states(energies: Any, strengths: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
        energy_arr = jnp.asarray(energies)
        strength_arr = jnp.asarray(strengths)
        valid = jnp.isfinite(energy_arr) & jnp.isfinite(strength_arr) & (energy_arr > 0.0)
        return energy_arr[valid], strength_arr[valid]

    requested_states = int(reference.nstates)
    solver_label = "Casida"
    try:
        with device_context:
            result = kernel_fn(nstates=requested_states)
            active_td = td
    except Exception:
        with device_context:
            result = tda_fn(nstates=requested_states)
            active_td = tda
        solver_label = "TDA fallback"

    with device_context:
        raw_energies = jnp.asarray(result.excitation_energies)
        raw_strengths = jnp.asarray(active_td.oscillator_strength())
    energies, strengths = _sanitize_states(raw_energies, raw_strengths)
    has_nonfinite = bool(jnp.any(~jnp.isfinite(raw_energies))) or bool(
        jnp.any(~jnp.isfinite(raw_strengths))
    )
    needs_tda = (
        solver_label != "TDA fallback"
        and (
            energies.size == 0
            or has_nonfinite
            or (requested_states > 0 and energies.size < requested_states)
        )
    )
    if needs_tda:
        try:
            with device_context:
                tda_result = tda_fn(nstates=requested_states)
                tda_raw_energies = jnp.asarray(tda_result.excitation_energies)
                tda_raw_strengths = jnp.asarray(tda.oscillator_strength())
            tda_energies, tda_strengths = _sanitize_states(tda_raw_energies, tda_raw_strengths)
            if tda_energies.size > 0 and tda_energies.size >= energies.size:
                result = tda_result
                energies = tda_energies
                strengths = tda_strengths
                solver_label = "TDA fallback"
        except Exception:
            pass

    elapsed = time.perf_counter() - t0

    return NeuralExcitedStateRun(
        solver_label=solver_label,
        energies_au=energies,
        oscillator_strengths=strengths,
        elapsed_s=elapsed,
    )


def build_spectrum(
    reference: ReferenceRun,
    neural: NeuralExcitedStateRun,
    grid: SpectrumGridConfig,
    simulation: SimulationConfig,
) -> SpectrumRun:
    """Build broadened absorption curves and low-energy MAE summary."""

    ref_max_ev = (
        float(jnp.max(reference.energies_au * HARTREE_TO_EV))
        if reference.energies_au.size > 0
        else grid.grid_min_ev
    )
    neural_max_ev = (
        float(jnp.max(neural.energies_au * HARTREE_TO_EV))
        if neural.energies_au.size > 0
        else grid.grid_min_ev
    )
    energy_max_ev = max(ref_max_ev, neural_max_ev) + grid.max_padding_ev
    grid_ev = jnp.linspace(grid.grid_min_ev, energy_max_ev, grid.grid_points)
    broaden_fn = (
        _compiled_lorentzian_spectrum()
        if simulation.jit_spectrum
        else lorentzian_spectrum
    )
    ref_curve = broaden_fn(
        reference.energies_au * HARTREE_TO_EV,
        reference.oscillator_strengths,
        grid_ev,
        eta=grid.eta_ev,
    )
    neural_curve = broaden_fn(
        neural.energies_au * HARTREE_TO_EV,
        neural.oscillator_strengths,
        grid_ev,
        eta=grid.eta_ev,
    )
    low_energy_mask = (grid_ev >= grid.zoom_min_ev) & (grid_ev <= grid.zoom_max_ev)

    ncompare = min(grid.compare_states, reference.energies_au.size, neural.energies_au.size)
    if ncompare == 0:
        low_energy_mae = float("nan")
    else:
        low_energy_mae = float(
            jnp.mean(
                jnp.abs(
                    reference.energies_au[:ncompare] * HARTREE_TO_EV
                    - neural.energies_au[:ncompare] * HARTREE_TO_EV
                )
            )
        )

    return SpectrumRun(
        grid_ev=grid_ev,
        reference_curve=ref_curve,
        neural_curve=neural_curve,
        low_energy_mask=low_energy_mask,
        low_energy_mae_ev=low_energy_mae,
        compared_states=ncompare,
    )


def run_pipeline_core(
    *,
    training_config: NeuralXCTrainingConfig,
    simulation_config: SimulationConfig,
    spectrum_config: SpectrumGridConfig,
    mf_builder: Callable[[], Any] | None = None,
    reference_spec: ReferenceSpecConfig | None = None,
) -> tuple[ReferenceRun, TrainingRun, NeuralExcitedStateRun, SpectrumRun]:
    """Run all compute steps from reference construction to Neural_xc spectrum.

    `reference_spec` is the preferred strict-JAX entrypoint. `mf_builder` is kept
    as a legacy compatibility path.
    """

    aligned_training_config = _canonicalize_graddft_ground_state_config(training_config)
    has_mf_builder = mf_builder is not None
    has_reference_spec = reference_spec is not None
    if has_mf_builder == has_reference_spec:
        raise ValueError(
            "run_pipeline_core requires exactly one of mf_builder or reference_spec."
        )

    compute_local_hfx_features = True
    compute_local_hfx_aux = False
    compute_local_pt2_features = bool(aligned_training_config.include_pt2_channel)
    hfx_omega_values = aligned_training_config.dm21_hfx_omega_values
    hfx_chunk_size = aligned_training_config.dm21_hfx_chunk_size

    if reference_spec is not None:
        reference = run_reference_from_spec(
            reference_spec,
            simulation=simulation_config,
            compute_local_hfx_features=compute_local_hfx_features,
            compute_local_hfx_aux=compute_local_hfx_aux,
            compute_local_pt2_features=compute_local_pt2_features,
            hfx_omega_values=hfx_omega_values,
            hfx_chunk_size=hfx_chunk_size,
        )
    else:
        warnings.warn(
            "mf_builder is a legacy compatibility path. Prefer reference_spec for the strict-JAX runtime.",
            DeprecationWarning,
            stacklevel=2,
        )
        t0 = time.perf_counter()
        mf = mf_builder()
        scf_elapsed = time.perf_counter() - t0

        reference = run_reference(
            mf,
            scf_elapsed_s=scf_elapsed,
            simulation=simulation_config,
            compute_local_hfx_features=compute_local_hfx_features,
            compute_local_hfx_aux=compute_local_hfx_aux,
            compute_local_pt2_features=compute_local_pt2_features,
            hfx_omega_values=hfx_omega_values,
            hfx_chunk_size=hfx_chunk_size,
        )
    training = train_neural_xc(reference, aligned_training_config, spectrum_config)
    neural = run_neural_tddft(reference, training, simulation_config)
    spectrum = build_spectrum(reference, neural, spectrum_config, simulation_config)
    return reference, training, neural, spectrum


def run_pipeline_core_from_spec(
    *,
    reference_spec: ReferenceSpecConfig,
    training_config: NeuralXCTrainingConfig,
    simulation_config: SimulationConfig,
    spectrum_config: SpectrumGridConfig,
) -> tuple[ReferenceRun, TrainingRun, NeuralExcitedStateRun, SpectrumRun]:
    """Compatibility wrapper around the spec-driven strict-JAX pipeline path."""

    return run_pipeline_core(
        reference_spec=reference_spec,
        training_config=training_config,
        simulation_config=simulation_config,
        spectrum_config=spectrum_config,
    )
