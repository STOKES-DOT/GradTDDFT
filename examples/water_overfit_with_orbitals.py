from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from pyscf import dft, gto

from td_graddft import neural_xc, tdscf
from td_graddft.device import put_restricted_molecule_on_device
from td_graddft.neural_xc_presets import (
    DM21_B3LYP_NEURAL_XC_PRESET,
    resolve_coefficient_prior_values,
)
from td_graddft.orbital_compare import (
    plot_orbital_compare_panel,
    render_restricted_orbital_surfaces,
)
from td_graddft.reference_legacy import restricted_reference_from_pyscf
from td_graddft.scf import DifferentiableSCF, DifferentiableSCFConfig
from td_graddft.spectra import HARTREE_TO_EV, lorentzian_spectrum, oscillator_strengths
from td_graddft.training import (
    GroundStateDatum,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    ground_state_mse_loss,
    predict_ground_state_total_energy,
    save_params_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overfit Neural_xc on H2O and compare spectra plus frontier orbitals."
    )
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--xc", default="b3lyp")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lr-decay-every", type=int, default=0)
    parser.add_argument("--lr-decay-factor", type=float, default=0.5)
    parser.add_argument("--density-weight", type=float, default=0.0)
    parser.add_argument(
        "--xc-potential-weight",
        type=float,
        default=0.0,
        help="weight for reference-v_xc grid matching penalty",
    )
    parser.add_argument(
        "--xc-kernel-weight",
        type=float,
        default=0.0,
        help="weight for reference-f_xc grid matching penalty (d v_xc / d rho)",
    )
    parser.add_argument(
        "--xc-kernel-target-clip",
        type=float,
        default=0.0,
        help="absolute clip applied to reference f_xc target before supervision (<=0 disables; default off)",
    )
    parser.add_argument(
        "--response-kernel-clip",
        type=float,
        default=0.0,
        help="internal response-tensor clipping bound; <=0 disables response clipping",
    )
    parser.add_argument(
        "--xc-kernel-normalize",
        choices=("none", "weighted_rms", "p95_abs"),
        default="weighted_rms",
        help="normalization mode for f_xc loss scaling",
    )
    parser.add_argument(
        "--xc-kernel-normalize-eps",
        type=float,
        default=1e-8,
        help="minimum scale used for f_xc normalization",
    )
    parser.add_argument(
        "--total-derivative-delta",
        type=float,
        default=1e-2,
        help="finite-difference step for total-energy density-scaling derivatives",
    )
    parser.add_argument(
        "--orbital-loss-weight",
        type=float,
        default=0.0,
        help="weight for frontier orbital energy-window MSE loss",
    )
    parser.add_argument(
        "--orbital-loss-window",
        type=int,
        default=2,
        help="frontier window k for HOMO-k...LUMO+k orbital-energy loss",
    )
    parser.add_argument(
        "--orbital-loss-mode",
        choices=("projected_fock", "self_consistent"),
        default="self_consistent",
        help="how orbital-window loss is computed",
    )
    parser.add_argument(
        "--orbital-loss-gradient",
        choices=("enabled", "detached"),
        default="enabled",
        help="whether orbital-window loss participates in backprop",
    )
    parser.add_argument("--orbital-loss-scf-max-cycle", type=int, default=16)
    parser.add_argument("--orbital-loss-scf-damping", type=float, default=0.35)
    parser.add_argument("--orbital-loss-scf-conv-tol-density", type=float, default=1e-7)
    parser.add_argument("--orbital-loss-scf-vxc-clip", type=float, default=20.0)
    parser.add_argument(
        "--orbital-loss-scf-gradient-mode",
        choices=("unrolled", "implicit_commutator"),
        default="implicit_commutator",
    )
    parser.add_argument("--orbital-loss-scf-implicit-max-iter", type=int, default=24)
    parser.add_argument("--orbital-loss-scf-implicit-step-size", type=float, default=0.2)
    parser.add_argument("--orbital-loss-scf-implicit-clip", type=float, default=1e4)
    parser.add_argument(
        "--orbital-loss-scf-iterate-selection",
        choices=("final", "best_rms", "first_converged"),
        default="best_rms",
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--s1-weight",
        type=float,
        default=0.0,
        help="Penalty weight on (predicted S1 - reference S1)^2 in Hartree^2.",
    )
    parser.add_argument(
        "--s1-use-tda",
        action="store_true",
        help="Use TDA (instead of Casida) when evaluating the S1 training constraint.",
    )
    parser.add_argument(
        "--train-scf-mode",
        choices=("fixed_density", "self_consistent"),
        default="fixed_density",
        help="Whether to include SCF iterations in the training computation graph.",
    )
    parser.add_argument("--train-scf-max-cycle", type=int, default=12)
    parser.add_argument("--train-scf-damping", type=float, default=0.25)
    parser.add_argument("--train-scf-conv-tol-density", type=float, default=1e-8)
    parser.add_argument("--train-scf-vxc-clip", type=float, default=20.0)
    parser.add_argument(
        "--train-scf-gradient-mode",
        choices=("unrolled", "implicit_commutator"),
        default="implicit_commutator",
        help="Gradient mode for self-consistent training.",
    )
    parser.add_argument("--train-scf-implicit-max-iter", type=int, default=24)
    parser.add_argument("--train-scf-implicit-step-size", type=float, default=0.2)
    parser.add_argument("--train-scf-implicit-clip", type=float, default=1e4)
    parser.add_argument(
        "--train-scf-iterate-selection",
        choices=("final", "best_rms", "first_converged"),
        default="final",
    )
    parser.add_argument(
        "--density-supervision",
        choices=("spin_summed", "spin_resolved"),
        default=DM21_B3LYP_NEURAL_XC_PRESET.density_supervision,
    )
    parser.add_argument("--stationarity-weight", type=float, default=0.0)
    parser.add_argument("--self-consistent-energy-weight", type=float, default=0.0)
    parser.add_argument("--coefficient-prior-weight", type=float, default=0.0)
    parser.add_argument("--coefficient-prior-values", type=float, nargs="+", default=None)
    parser.add_argument(
        "--coefficient-prior-mode",
        choices=("pointwise", "mean"),
        default=DM21_B3LYP_NEURAL_XC_PRESET.coefficient_prior_mode,
    )
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 64, 64])
    parser.add_argument(
        "--semilocal-xc",
        nargs="+",
        default=list(DM21_B3LYP_NEURAL_XC_PRESET.semilocal_xc),
        help="one or more semilocal basis channels for Neural_xc",
    )
    parser.add_argument(
        "--n-semilocal-channels",
        type=int,
        default=None,
        help="required only for custom multi-channel semilocal callbacks",
    )
    parser.add_argument(
        "--hf-input-mode",
        choices=("total_only", "spin_resolved"),
        default=DM21_B3LYP_NEURAL_XC_PRESET.hf_input_mode,
    )
    parser.add_argument(
        "--coefficient-positivity",
        choices=("clip", "softplus"),
        default=DM21_B3LYP_NEURAL_XC_PRESET.coefficient_positivity,
    )
    parser.add_argument(
        "--response-hf-mode",
        choices=("nonlocal_exchange_only", "local_projected"),
        default=DM21_B3LYP_NEURAL_XC_PRESET.response_hf_mode,
    )
    parser.add_argument(
        "--feature-mode",
        choices=("enhanced", "dm21_original"),
        default="dm21_original",
        help="Neural_xc input feature mode; dm21_original mirrors DM21 NeuralNumInt inputs",
    )
    parser.add_argument(
        "--dm21-omega-values",
        type=float,
        nargs="+",
        default=[0.0, 0.4],
        help="Local HF omega values when --feature-mode=dm21_original",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nstates", type=int, default=8)
    parser.add_argument("--eta-ev", type=float, default=0.15)
    parser.add_argument("--grid-min-ev", type=float, default=5.0)
    parser.add_argument("--grid-max-ev", type=float, default=35.0)
    parser.add_argument("--grid-points", type=int, default=2200)
    parser.add_argument(
        "--xyzrender-src",
        default="/Volumes/TF/QH9_db/xyzrender/src",
        help="xyzrender source directory",
    )
    parser.add_argument("--orbital-iso", type=float, default=0.05)
    parser.add_argument("--orbital-diff-iso", type=float, default=0.03)
    parser.add_argument("--orbital-mo-blur", type=float, default=1.2)
    parser.add_argument("--orbital-mo-upsample", type=int, default=4)
    parser.add_argument("--orbital-cube-grid", type=int, default=48)
    parser.add_argument("--orbital-canvas-size", type=int, default=620)
    parser.add_argument(
        "--disable-orbital-frontier-match",
        action="store_true",
        help="disable overlap-based HOMO-1/HOMO and LUMO/LUMO+1 orbital matching",
    )
    parser.add_argument(
        "--orbital-frontier-match-window",
        type=int,
        default=6,
        help="number of near-frontier candidate orbitals used for overlap matching",
    )
    parser.add_argument("--orbital-scf-max-cycle", type=int, default=256)
    parser.add_argument("--orbital-scf-damping", type=float, default=0.96)
    parser.add_argument("--orbital-scf-conv-tol-density", type=float, default=1e-6)
    parser.add_argument("--orbital-scf-vxc-clip", type=float, default=20.0)
    parser.add_argument(
        "--orbital-scf-iterate-selection",
        choices=("final", "best_rms", "first_converged"),
        default="first_converged",
    )
    parser.add_argument("--outdir", default="outputs/water_overfit_strict_mgga")
    return parser.parse_args()


def _normalize_semilocal_arg(values: list[str] | str) -> str | tuple[str, ...]:
    if isinstance(values, str):
        return values
    if len(values) == 1:
        return values[0]
    return tuple(values)


def _make_water_mf(*, basis: str, xc: str):
    mol = gto.Mole()
    mol.atom = """
    O  0.000000  0.000000  0.117790
    H  0.000000  0.755453 -0.471161
    H  0.000000 -0.755453 -0.471161
    """
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.spin = 0
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"PySCF SCF did not converge for H2O {xc}/{basis}.")
    return mol, mf


def _reference_xc_potential_on_grid(mf: Any) -> jnp.ndarray:
    """Evaluate reference scalar v_xc(r) on the PySCF integration grid."""

    ni = mf._numint
    mol = mf.mol
    grids = mf.grids
    dm = mf.make_rdm1()
    xctype = ni._xc_type(mf.xc)
    ao = ni.eval_ao(mol, grids.coords, deriv=0 if xctype == "LDA" else 1)
    rho = ni.eval_rho(mol, ao, dm, xctype=xctype)
    _, vxc, _, _ = ni.eval_xc(mf.xc, rho, spin=0, relativity=0, deriv=1)
    vrho = np.asarray(vxc[0], dtype=float)
    if vrho.ndim == 2:
        if vrho.shape[1] == 1:
            vrho = vrho[:, 0]
        elif vrho.shape[0] == 1:
            vrho = vrho[0]
        else:
            vrho = np.mean(vrho, axis=-1)
    return jnp.asarray(vrho.reshape(-1))


def _reference_xc_kernel_on_grid(mf: Any, *, clip_abs: float | None = None) -> jnp.ndarray:
    """Evaluate reference scalar f_xc(r)=d(v_rho)/d(rho) on the PySCF grid."""

    ni = mf._numint
    mol = mf.mol
    grids = mf.grids
    dm = mf.make_rdm1()
    xctype = ni._xc_type(mf.xc)
    ao = ni.eval_ao(mol, grids.coords, deriv=0 if xctype == "LDA" else 1)
    rho = ni.eval_rho(mol, ao, dm, xctype=xctype)
    _, _, fxc, _ = ni.eval_xc(mf.xc, rho, spin=0, relativity=0, deriv=2)

    if isinstance(fxc, (list, tuple)):
        if len(fxc) == 0:
            raise RuntimeError("PySCF eval_xc returned an empty f_xc container.")
        frr = np.asarray(fxc[0], dtype=float)
    else:
        arr = np.asarray(fxc, dtype=float)
        if arr.ndim == 1:
            frr = arr
        elif arr.ndim == 2:
            ngrids = int(np.asarray(grids.weights).shape[0])
            if arr.shape[0] == ngrids:
                frr = arr[:, 0]
            elif arr.shape[1] == ngrids:
                frr = arr[0, :]
            else:
                frr = arr.reshape((-1,))[0:ngrids]
        else:
            ngrids = int(np.asarray(grids.weights).shape[0])
            frr = arr.reshape((-1, ngrids))[0]
    values = np.nan_to_num(frr.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if clip_abs is not None and clip_abs > 0.0:
        values = np.clip(values, -clip_abs, clip_abs)
    return jnp.asarray(values)


def _replace_molecule_copy(molecule: Any, **updates: Any) -> Any:
    if is_dataclass(molecule):
        return replace(molecule, **updates)
    cloned = copy.copy(molecule)
    for key, value in updates.items():
        setattr(cloned, key, value)
    return cloned


def _scaled_molecule_density(molecule: Any, scale: float) -> Any:
    updates = {"rdm1": jnp.asarray(molecule.rdm1) * float(scale)}
    if getattr(molecule, "mo_occ", None) is not None:
        updates["mo_occ"] = jnp.asarray(molecule.mo_occ) * float(scale)
    return _replace_molecule_copy(molecule, **updates)


def _reference_total_energy_derivatives_from_mf(
    mf: Any,
    *,
    delta: float,
) -> tuple[float, float]:
    dm0 = np.asarray(mf.make_rdm1(), dtype=float)
    enuc = float(mf.energy_nuc())
    h = float(delta)
    if h <= 0.0:
        raise ValueError("total-derivative-delta must be positive.")
    if h >= 1.0:
        raise ValueError("total-derivative-delta must be < 1.0 for density scaling.")

    def total_energy(scale: float) -> float:
        e_elec, _ = mf.energy_elec(dm=dm0 * float(scale))
        return float(e_elec) + enuc

    e_m = total_energy(1.0 - h)
    e_0 = total_energy(1.0)
    e_p = total_energy(1.0 + h)
    d1 = (e_p - e_m) / (2.0 * h)
    d2 = (e_p - 2.0 * e_0 + e_m) / (h * h)
    return float(d1), float(d2)


def _kernel_normalization_scale(
    values: jnp.ndarray,
    weights: jnp.ndarray,
    *,
    mode: str,
    eps: float,
) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape[0] != arr.shape[0]:
        raise ValueError(
            f"Kernel normalization expects matching shapes, got values={arr.shape} weights={w.shape}."
        )
    w = np.clip(w, 0.0, None)
    wsum = max(float(np.sum(w)), float(eps))
    if mode == "none":
        scale = 1.0
    elif mode == "weighted_rms":
        scale = float(np.sqrt(np.sum(w * arr * arr) / wsum))
    elif mode == "p95_abs":
        scale = float(np.percentile(np.abs(arr), 95.0))
    else:
        raise ValueError(f"Unsupported xc-kernel-normalize mode: {mode!r}")
    return float(max(abs(scale), float(eps)))


def _write_training_curve(
    path: Path,
    total_history: list[float],
    ground_history: list[float],
    orbital_mse_history: list[float],
    orbital_weighted_history: list[float],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("step,total_loss,ground_loss,orbital_mse,orbital_weighted\n")
        for step in range(len(total_history)):
            handle.write(
                f"{step},"
                f"{total_history[step]:.16e},"
                f"{ground_history[step]:.16e},"
                f"{orbital_mse_history[step]:.16e},"
                f"{orbital_weighted_history[step]:.16e}\n"
            )


def _plot_training_curve(
    path: Path,
    total_history: list[float],
    ground_history: list[float],
    orbital_weighted_history: list[float],
    *,
    orbital_loss_weight: float,
) -> None:
    total_values = np.maximum(np.asarray(total_history, dtype=float), 1e-16)
    ground_values = np.maximum(np.asarray(ground_history, dtype=float), 1e-16)
    orbital_values = np.maximum(np.asarray(orbital_weighted_history, dtype=float), 1e-16)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    x = np.arange(total_values.size)
    ax.plot(x, total_values, lw=2.2, label="total")
    ax.plot(x, ground_values, lw=1.8, alpha=0.95, label="ground")
    if orbital_loss_weight != 0.0:
        ax.plot(x, orbital_values, lw=1.6, alpha=0.9, label="orbital(weighted)")
    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss (log scale)")
    ax.set_title("H2O single-molecule overfit loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _restricted_mo_occ_vector(mo_occ: Any) -> np.ndarray:
    occ = np.asarray(mo_occ, dtype=float)
    if occ.ndim == 1:
        return occ
    if occ.ndim == 2:
        return np.mean(occ, axis=0)
    raise ValueError("Expected mo_occ to have shape (nmo,) or (spin, nmo).")


def _restricted_mo_energy_vector(mo_energy: Any) -> jnp.ndarray:
    energy = jnp.asarray(mo_energy)
    if energy.ndim == 1:
        return energy
    if energy.ndim == 2:
        return jnp.mean(energy, axis=0)
    raise ValueError("Expected mo_energy to have shape (nmo,) or (spin, nmo).")


def _frontier_orbital_window_from_occ(
    mo_occ: np.ndarray,
    *,
    window: int,
) -> tuple[int, int, int, int]:
    occ = np.where(mo_occ > 1e-8)[0]
    vir = np.where(mo_occ <= 1e-8)[0]
    if occ.size == 0:
        raise RuntimeError("No occupied orbitals found for orbital-loss window.")
    if vir.size == 0:
        raise RuntimeError("No virtual orbitals found for orbital-loss window.")

    homo = int(occ[-1])
    lumo = int(vir[0])
    nmo = int(mo_occ.shape[0])
    k = max(int(window), 0)
    start = max(0, homo - k)
    stop = min(nmo, lumo + k + 1)
    if stop <= start:
        stop = min(nmo, start + 1)
    return start, stop, homo, lumo


def _spin_summed_density_matrix(molecule: Any) -> jnp.ndarray:
    density = jnp.asarray(molecule.rdm1)
    if density.ndim == 3:
        return density.sum(axis=0)
    return density


def _restricted_mo_coeff_matrix(mo_coeff: Any) -> jnp.ndarray:
    coeff = jnp.asarray(mo_coeff)
    if coeff.ndim == 2:
        return coeff
    if coeff.ndim == 3:
        if coeff.shape[0] == 1:
            return coeff[0]
        if coeff.shape[0] == 2:
            return coeff[0]
    raise ValueError("Expected mo_coeff to have shape (nao,nmo) or (spin,nao,nmo).")


def _resolved_xc_object(params: Any, functional: Any, molecule: Any) -> Any:
    binder = getattr(functional, "bind_to_molecule", None)
    if binder is not None:
        return binder(params, molecule)
    binder = getattr(functional, "bind", None)
    if binder is not None:
        return binder(params)
    return functional


def _grid_xc_potential_from_resolved(
    resolved: Any,
    *,
    functional: Any,
    params: Any,
    molecule: Any,
) -> jnp.ndarray:
    grid_potential = getattr(resolved, "grid_potential", None)
    if grid_potential is not None:
        return jnp.asarray(grid_potential(molecule))

    local_potential = getattr(resolved, "local_potential", None)
    if local_potential is not None:
        density = jnp.asarray(molecule.density()).sum(axis=-1)
        return jnp.asarray(local_potential(density))

    functional_local_potential = getattr(functional, "local_potential", None)
    if functional_local_potential is None:
        raise AttributeError(
            "The XC functional must expose local_potential(...) or grid_potential(...)."
        )
    density = jnp.asarray(molecule.density()).sum(axis=-1)
    return jnp.asarray(functional_local_potential(params, density))


def _effective_exact_exchange_fraction_from_resolved(resolved: Any) -> jnp.ndarray:
    alpha = getattr(resolved, "exact_exchange_fraction", 0.0)
    return jnp.asarray(alpha)


def _projected_orbital_window_mse(
    *,
    params: Any,
    functional: Any,
    molecule: Any,
    reference_mo_energy_window: jnp.ndarray,
    window_start: int,
    window_stop: int,
    vxc_clip: float,
) -> jnp.ndarray:
    density = _spin_summed_density_matrix(molecule)
    h1e = jnp.asarray(molecule.h1e)
    rep_tensor = jnp.asarray(molecule.rep_tensor)
    ao = jnp.asarray(molecule.ao)
    weights = jnp.asarray(molecule.grid.weights)
    mo_coeff = _restricted_mo_coeff_matrix(molecule.mo_coeff)

    resolved_xc = _resolved_xc_object(params, functional, molecule)
    vxc_grid = _grid_xc_potential_from_resolved(
        resolved_xc,
        functional=functional,
        params=params,
        molecule=molecule,
    )
    vxc_grid = jnp.nan_to_num(vxc_grid, nan=0.0, posinf=vxc_clip, neginf=-vxc_clip)
    vxc_grid = jnp.clip(vxc_grid, -vxc_clip, vxc_clip)

    j_mat = jnp.einsum("pqrs,rs->pq", rep_tensor, density)
    k_mat = jnp.einsum("prqs,rs->pq", rep_tensor, density)
    vxc_matrix = jnp.einsum("r,rp,rq->pq", weights * vxc_grid, ao, ao)
    alpha = _effective_exact_exchange_fraction_from_resolved(resolved_xc)
    alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
    alpha = jnp.clip(alpha, 0.0, 1.0)

    fock = h1e + j_mat - 0.5 * alpha * k_mat + vxc_matrix
    fock = 0.5 * (fock + fock.T)
    fock = jnp.nan_to_num(fock, nan=0.0, posinf=0.0, neginf=0.0)

    projected_mo_energy = jnp.einsum("pi,pq,qi->i", mo_coeff, fock, mo_coeff)
    pred_window = projected_mo_energy[window_start:window_stop]
    mse = jnp.mean((pred_window - reference_mo_energy_window) ** 2)
    return jnp.nan_to_num(mse, nan=0.0, posinf=1e6, neginf=1e6)


def _write_spectrum_csv(
    path: Path,
    grid_ev: jnp.ndarray,
    ref_curve: jnp.ndarray,
    neural_curve: jnp.ndarray,
) -> None:
    np.savetxt(
        path,
        np.column_stack([np.asarray(grid_ev), np.asarray(ref_curve), np.asarray(neural_curve)]),
        delimiter=",",
        header="energy_ev,reference_curve,neural_curve",
        comments="",
    )


def _write_state_csv(
    path: Path,
    ref_energies_au: jnp.ndarray,
    ref_osc: jnp.ndarray,
    neural_energies_au: jnp.ndarray,
    neural_osc: jnp.ndarray,
) -> None:
    nrows = max(
        int(ref_energies_au.size),
        int(ref_osc.size),
        int(neural_energies_au.size),
        int(neural_osc.size),
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write("state,b3lyp_energy_ev,b3lyp_oscillator_strength,neural_energy_ev,neural_oscillator_strength\n")
        for idx in range(nrows):
            ref_e = (
                f"{float(ref_energies_au[idx] * HARTREE_TO_EV):.10f}"
                if idx < ref_energies_au.size
                else ""
            )
            ref_f = f"{float(ref_osc[idx]):.10f}" if idx < ref_osc.size else ""
            neu_e = (
                f"{float(neural_energies_au[idx] * HARTREE_TO_EV):.10f}"
                if idx < neural_energies_au.size
                else ""
            )
            neu_f = f"{float(neural_osc[idx]):.10f}" if idx < neural_osc.size else ""
            handle.write(f"{idx + 1},{ref_e},{ref_f},{neu_e},{neu_f}\n")


def _plot_spectrum(
    path: Path,
    *,
    grid_ev: jnp.ndarray,
    ref_curve: jnp.ndarray,
    neural_curve: jnp.ndarray,
    excitation_mae_ev: float,
    solver_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    ax.plot(np.asarray(grid_ev), np.asarray(ref_curve), lw=2.0, label="PySCF TDDFT B3LYP/STO-3G")
    ax.plot(
        np.asarray(grid_ev),
        np.asarray(neural_curve),
        lw=2.0,
        label=f"Neural_xc TDDFT ({solver_label})",
    )
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Absorption (a.u.)")
    ax.set_title(f"H2O absorption spectrum | strict kernel | MAE={excitation_mae_ev:.3f} eV")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_excitation_energy_compare(
    path: Path,
    *,
    ref_energies_au: jnp.ndarray,
    neural_energies_au: jnp.ndarray,
) -> None:
    ncompare = int(min(ref_energies_au.size, neural_energies_au.size))
    if ncompare <= 0:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        ax.text(0.5, 0.5, "No comparable excited states", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return

    ref_ev = np.asarray(ref_energies_au[:ncompare] * HARTREE_TO_EV, dtype=float)
    neural_ev = np.asarray(neural_energies_au[:ncompare] * HARTREE_TO_EV, dtype=float)
    state_idx = np.arange(1, ncompare + 1)
    abs_err = np.abs(neural_ev - ref_ev)
    mae = float(np.mean(abs_err))

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10.2, 4.2))
    ax_left.plot(state_idx, ref_ev, "o-", lw=2.0, ms=4.5, label="PySCF")
    ax_left.plot(state_idx, neural_ev, "s--", lw=1.8, ms=4.2, label="Neural_xc")
    ax_left.set_xlabel("Excited state index")
    ax_left.set_ylabel("Excitation energy (eV)")
    ax_left.set_title(f"State-by-state energy comparison (MAE={mae:.3f} eV)")
    ax_left.grid(True, alpha=0.3)
    ax_left.legend()

    lo = float(min(np.min(ref_ev), np.min(neural_ev)))
    hi = float(max(np.max(ref_ev), np.max(neural_ev)))
    pad = 0.06 * max(hi - lo, 1.0)
    line_lo = lo - pad
    line_hi = hi + pad
    scatter = ax_right.scatter(ref_ev, neural_ev, c=state_idx, cmap="viridis", s=32)
    ax_right.plot([line_lo, line_hi], [line_lo, line_hi], "k--", lw=1.2, label="y = x")
    ax_right.set_xlim(line_lo, line_hi)
    ax_right.set_ylim(line_lo, line_hi)
    ax_right.set_xlabel("Reference excitation energy (eV)")
    ax_right.set_ylabel("Neural_xc excitation energy (eV)")
    ax_right.set_title("Reference vs Neural_xc")
    ax_right.grid(True, alpha=0.3)
    ax_right.legend(loc="upper left")
    cbar = fig.colorbar(scatter, ax=ax_right, fraction=0.046, pad=0.04)
    cbar.set_label("State index")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    semilocal_xc = _normalize_semilocal_arg(args.semilocal_xc)
    coefficient_prior_values = (
        resolve_coefficient_prior_values(semilocal_xc, args.coefficient_prior_values)
        if (
            args.coefficient_prior_values is not None
            or float(args.coefficient_prior_weight) != 0.0
        )
        else None
    )

    t0 = time.perf_counter()
    mol, mf = _make_water_mf(basis=args.basis, xc=args.xc)
    reference = put_restricted_molecule_on_device(
        restricted_reference_from_pyscf(
            mf,
            compute_local_hfx_features=(args.feature_mode == "dm21_original"),
            compute_local_hfx_aux=(args.feature_mode == "dm21_original"),
            hfx_omega_values=tuple(float(x) for x in args.dm21_omega_values),
        )
    )
    td = mf.TDDFT()
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 1e-8))
    nvir = int(np.asarray(mf.mo_coeff).shape[-1] - nocc)
    nstates = min(max(args.nstates, 1), nocc * nvir)
    td.nstates = nstates
    td.kernel()
    ref_energies_au = jnp.asarray(td.e)
    ref_osc = jnp.asarray(td.oscillator_strength())
    reference_elapsed = time.perf_counter() - t0

    target_xc_potential = (
        _reference_xc_potential_on_grid(mf) if float(args.xc_potential_weight) != 0.0 else None
    )
    target_xc_kernel = (
        _reference_xc_kernel_on_grid(
            mf,
            clip_abs=(
                float(args.xc_kernel_target_clip)
                if float(args.xc_kernel_target_clip) > 0.0
                else None
            ),
        )
        if float(args.xc_kernel_weight) != 0.0
        else None
    )
    xc_kernel_normalization_scale = (
        _kernel_normalization_scale(
            target_xc_kernel,
            jnp.asarray(reference.grid.weights),
            mode=str(args.xc_kernel_normalize),
            eps=float(args.xc_kernel_normalize_eps),
        )
        if target_xc_kernel is not None
        else 1.0
    )
    target_total_d1, target_total_d2 = _reference_total_energy_derivatives_from_mf(
        mf,
        delta=float(args.total_derivative_delta),
    )

    datum = GroundStateDatum(
        molecule=reference,
        target_total_energy=jnp.asarray(reference.mf_energy),
        target_s1_energy=(
            jnp.asarray(ref_energies_au[0]) if int(ref_energies_au.size) > 0 else None
        ),
        target_xc_potential=None,
        target_xc_kernel=None,
        density_constraint_weight=float(args.density_weight),
        xc_potential_constraint_weight=0.0,
        xc_kernel_constraint_weight=0.0,
        xc_kernel_normalization_scale=1.0,
        stationarity_constraint_weight=float(args.stationarity_weight),
        s1_constraint_weight=float(args.s1_weight),
    )
    functional = neural_xc.Functional(
        semilocal_xc=semilocal_xc,
        n_semilocal_channels=args.n_semilocal_channels,
        input_feature_mode=args.feature_mode,
        dm21_hfx_channels=max(len(args.dm21_omega_values), 1),
        hf_input_mode=args.hf_input_mode,
        response_hf_mode=args.response_hf_mode,
        coefficient_positivity=args.coefficient_positivity,
        hidden_dims=tuple(args.hidden_dims),
        response_kernel_clip=(
            None
            if float(args.response_kernel_clip) <= 0.0
            else float(args.response_kernel_clip)
        ),
        name="water_overfit_neural_xc",
    )
    training_config = GroundStateTrainingConfig(
        mode=args.train_scf_mode,
        energy_mse_weight=1.0,
        energy_mae_weight=1.0,
        energy_normalization="none",
        density_supervision=args.density_supervision,
        self_consistent_energy_weight=float(args.self_consistent_energy_weight),
        coefficient_prior_weight=float(args.coefficient_prior_weight),
        coefficient_prior_values=coefficient_prior_values,
        coefficient_prior_mode=args.coefficient_prior_mode,
        s1_constraint_use_tda=bool(args.s1_use_tda),
        scf_max_cycle=int(args.train_scf_max_cycle),
        scf_damping=float(args.train_scf_damping),
        scf_conv_tol_density=float(args.train_scf_conv_tol_density),
        scf_vxc_clip=float(args.train_scf_vxc_clip),
        scf_gradient_mode=args.train_scf_gradient_mode,
        scf_implicit_diff_max_iter=int(args.train_scf_implicit_max_iter),
        scf_implicit_diff_step_size=float(args.train_scf_implicit_step_size),
        scf_implicit_diff_clip=float(args.train_scf_implicit_clip),
        scf_iterate_selection=args.train_scf_iterate_selection,
    )
    deriv_training_config = replace(training_config, mode="fixed_density")
    if args.lr_decay_every > 0:
        lr_schedule = optax.exponential_decay(
            init_value=float(args.learning_rate),
            transition_steps=int(args.lr_decay_every),
            decay_rate=float(args.lr_decay_factor),
            staircase=True,
        )
        optimizer = optax.adam(lr_schedule)
    else:
        optimizer = optax.adam(args.learning_rate)
    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(args.seed),
        reference,
        optimizer,
    )
    ref_mo_occ = _restricted_mo_occ_vector(mf.mo_occ)
    window_start, window_stop, window_homo, window_lumo = _frontier_orbital_window_from_occ(
        ref_mo_occ,
        window=args.orbital_loss_window,
    )
    ref_mo_energy_window = _restricted_mo_energy_vector(mf.mo_energy)[window_start:window_stop]
    orbital_loss_weight = float(args.orbital_loss_weight)
    use_orbital_loss = orbital_loss_weight != 0.0
    use_orbital_self_consistent = use_orbital_loss and args.orbital_loss_mode == "self_consistent"
    if use_orbital_self_consistent:
        orbital_loss_scf = DifferentiableSCF(
            DifferentiableSCFConfig(
                mode="self_consistent",
                gradient_mode=args.orbital_loss_scf_gradient_mode,
                max_cycle=args.orbital_loss_scf_max_cycle,
                damping=args.orbital_loss_scf_damping,
                conv_tol_density=args.orbital_loss_scf_conv_tol_density,
                vxc_clip=args.orbital_loss_scf_vxc_clip,
                iterate_selection=args.orbital_loss_scf_iterate_selection,
                implicit_diff_max_iter=args.orbital_loss_scf_implicit_max_iter,
                implicit_diff_step_size=args.orbital_loss_scf_implicit_step_size,
                implicit_diff_clip=args.orbital_loss_scf_implicit_clip,
            )
        )
    else:
        orbital_loss_scf = None

    def _compute_total_loss(params: Any) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        ground_loss, ground_metrics = ground_state_mse_loss(
            params,
            functional,
            datum,
            training_config=training_config,
        )
        h = float(args.total_derivative_delta)
        if h <= 0.0:
            raise ValueError("total-derivative-delta must be positive.")

        mol_m = _scaled_molecule_density(reference, 1.0 - h)
        mol_0 = _scaled_molecule_density(reference, 1.0)
        mol_p = _scaled_molecule_density(reference, 1.0 + h)
        e_m = predict_ground_state_total_energy(
            params,
            functional,
            mol_m,
            training_config=deriv_training_config,
        )
        e_0 = predict_ground_state_total_energy(
            params,
            functional,
            mol_0,
            training_config=deriv_training_config,
        )
        e_p = predict_ground_state_total_energy(
            params,
            functional,
            mol_p,
            training_config=deriv_training_config,
        )
        pred_d1 = (e_p - e_m) / (2.0 * h)
        pred_d2 = (e_p - 2.0 * e_0 + e_m) / (h * h)
        d1_mse = (pred_d1 - jnp.asarray(target_total_d1, dtype=ground_loss.dtype)) ** 2
        d2_mse = (pred_d2 - jnp.asarray(target_total_d2, dtype=ground_loss.dtype)) ** 2
        d1_penalty = jnp.asarray(args.xc_potential_weight, dtype=ground_loss.dtype) * d1_mse
        d2_penalty = jnp.asarray(args.xc_kernel_weight, dtype=ground_loss.dtype) * d2_mse

        orbital_mse = jnp.asarray(0.0, dtype=ground_loss.dtype)
        if use_orbital_loss:
            if args.orbital_loss_mode == "projected_fock":
                orbital_mse = _projected_orbital_window_mse(
                    params=params,
                    functional=functional,
                    molecule=reference,
                    reference_mo_energy_window=ref_mo_energy_window,
                    window_start=window_start,
                    window_stop=window_stop,
                    vxc_clip=float(args.orbital_loss_scf_vxc_clip),
                )
            else:
                assert orbital_loss_scf is not None
                scf_mol, _ = orbital_loss_scf.run(reference, functional, params)
                pred_mo_energy = _restricted_mo_energy_vector(scf_mol.mo_energy)
                pred_window = pred_mo_energy[window_start:window_stop]
                orbital_mse = jnp.mean((pred_window - ref_mo_energy_window) ** 2)
                orbital_mse = jnp.nan_to_num(orbital_mse, nan=0.0, posinf=1e6, neginf=1e6)
        orbital_grad_value = (
            orbital_mse
            if args.orbital_loss_gradient == "enabled"
            else jax.lax.stop_gradient(orbital_mse)
        )
        orbital_weighted = (
            jnp.asarray(orbital_loss_weight, dtype=ground_loss.dtype) * orbital_grad_value
        )
        total_loss = ground_loss + orbital_weighted + d1_penalty + d2_penalty
        metrics = dict(ground_metrics)
        metrics["ground_loss"] = ground_loss
        metrics["orbital_mse_loss"] = orbital_mse
        metrics["orbital_weighted_loss"] = orbital_weighted
        metrics["xc_potential_penalty"] = jnp.atleast_1d(d1_penalty)
        metrics["xc_potential_mse"] = jnp.atleast_1d(d1_mse)
        metrics["xc_kernel_penalty"] = jnp.atleast_1d(d2_penalty)
        metrics["xc_kernel_mse"] = jnp.atleast_1d(d2_mse)
        metrics["total_energy_d1_pred"] = jnp.atleast_1d(pred_d1)
        metrics["total_energy_d1_target"] = jnp.atleast_1d(
            jnp.asarray(target_total_d1, dtype=ground_loss.dtype)
        )
        metrics["total_energy_d2_pred"] = jnp.atleast_1d(pred_d2)
        metrics["total_energy_d2_target"] = jnp.atleast_1d(
            jnp.asarray(target_total_d2, dtype=ground_loss.dtype)
        )
        metrics["total_loss"] = total_loss
        return total_loss, metrics

    initial_loss, initial_metrics = _compute_total_loss(state.params)
    initial_loss = float(initial_loss)
    initial_ground_loss = float(initial_metrics["ground_loss"])
    initial_orbital_mse = float(initial_metrics["orbital_mse_loss"])
    initial_orbital_weighted = float(initial_metrics["orbital_weighted_loss"])
    initial_density_penalty = float(initial_metrics["density_penalty"][0])
    initial_xc_potential_penalty = float(initial_metrics["xc_potential_penalty"][0])
    initial_xc_potential_mse = float(initial_metrics["xc_potential_mse"][0])
    initial_xc_kernel_penalty = float(initial_metrics["xc_kernel_penalty"][0])
    initial_xc_kernel_mse = float(initial_metrics["xc_kernel_mse"][0])
    initial_self_consistent_energy_penalty = float(
        initial_metrics["self_consistent_energy_penalty"][0]
    )
    initial_coefficient_prior_penalty = float(initial_metrics["coefficient_prior_penalty"][0])
    initial_stationarity_penalty = float(initial_metrics["stationarity_penalty"][0])
    initial_s1_penalty = float(initial_metrics["s1_penalty"][0])
    initial_s1_mse = float(initial_metrics["s1_mse"][0])
    initial_s1_predicted = float(initial_metrics["s1_predicted"][0])
    initial_s1_target = float(initial_metrics["s1_target"][0])
    total_history = [initial_loss]
    ground_history = [initial_ground_loss]
    orbital_mse_history = [initial_orbital_mse]
    orbital_weighted_history = [initial_orbital_weighted]
    best_loss = initial_loss
    best_step = 0
    best_params = state.params
    nan_grad_fallback_steps = 0

    train_t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        params_before_update = state.params
        (train_loss, train_metrics), grads = jax.value_and_grad(
            _compute_total_loss,
            has_aux=True,
        )(params_before_update)
        grad_leaves = jax.tree_util.tree_leaves(grads)
        grad_has_nan = any(bool(jnp.isnan(g).any()) for g in grad_leaves)
        grad_has_inf = any(bool(jnp.isinf(g).any()) for g in grad_leaves)
        if (
            (grad_has_nan or grad_has_inf)
            and use_orbital_loss
            and args.orbital_loss_gradient == "enabled"
        ):
            _, grads_ground = jax.value_and_grad(
                lambda p: ground_state_mse_loss(
                    p,
                    functional,
                    datum,
                    training_config=training_config,
                )[0]
            )(params_before_update)
            grads = grads_ground
            nan_grad_fallback_steps += 1
        grads = jax.tree_util.tree_map(
            lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0),
            grads,
        )
        train_loss_val = float(train_loss)
        total_history.append(train_loss_val)
        ground_history.append(float(train_metrics["ground_loss"]))
        orbital_mse_history.append(float(train_metrics["orbital_mse_loss"]))
        orbital_weighted_history.append(float(train_metrics["orbital_weighted_loss"]))
        state = state.apply_gradients(grads=grads)
        if train_loss_val < best_loss:
            best_loss = train_loss_val
            best_step = step
            best_params = params_before_update
        if args.log_every > 0 and (step == 1 or step == args.steps or step % args.log_every == 0):
            print(
                f"[Water][step {step}/{args.steps}] "
                f"total={train_loss_val:.6e} "
                f"ground={float(train_metrics['ground_loss']):.6e} "
                f"orb_mse={float(train_metrics['orbital_mse_loss']):.6e} "
                f"orb_weighted={float(train_metrics['orbital_weighted_loss']):.6e} "
                f"xc_pot={float(train_metrics['xc_potential_penalty'][0]):.6e} "
                f"fxc={float(train_metrics['xc_kernel_penalty'][0]):.6e} "
                f"grad_nan={grad_has_nan} grad_inf={grad_has_inf}",
                flush=True,
            )
    train_elapsed = time.perf_counter() - train_t0

    final_loss, final_metrics = _compute_total_loss(state.params)
    final_loss = float(final_loss)
    final_ground_loss = float(final_metrics["ground_loss"])
    final_orbital_mse = float(final_metrics["orbital_mse_loss"])
    final_orbital_weighted = float(final_metrics["orbital_weighted_loss"])
    final_density_penalty = float(final_metrics["density_penalty"][0])
    final_xc_potential_penalty = float(final_metrics["xc_potential_penalty"][0])
    final_xc_potential_mse = float(final_metrics["xc_potential_mse"][0])
    final_xc_kernel_penalty = float(final_metrics["xc_kernel_penalty"][0])
    final_xc_kernel_mse = float(final_metrics["xc_kernel_mse"][0])
    final_self_consistent_energy_penalty = float(
        final_metrics["self_consistent_energy_penalty"][0]
    )
    final_coefficient_prior_penalty = float(final_metrics["coefficient_prior_penalty"][0])
    final_stationarity_penalty = float(final_metrics["stationarity_penalty"][0])
    final_s1_penalty = float(final_metrics["s1_penalty"][0])
    final_s1_mse = float(final_metrics["s1_mse"][0])
    final_s1_predicted = float(final_metrics["s1_predicted"][0])
    final_s1_target = float(final_metrics["s1_target"][0])

    orbital_eval_config = GroundStateTrainingConfig(
        mode="self_consistent",
        density_supervision=args.density_supervision,
        scf_max_cycle=args.orbital_scf_max_cycle,
        scf_damping=args.orbital_scf_damping,
        scf_conv_tol_density=args.orbital_scf_conv_tol_density,
        scf_vxc_clip=args.orbital_scf_vxc_clip,
        scf_iterate_selection=args.orbital_scf_iterate_selection,
    )
    scf = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            max_cycle=args.orbital_scf_max_cycle,
            damping=args.orbital_scf_damping,
            conv_tol_density=args.orbital_scf_conv_tol_density,
            vxc_clip=args.orbital_scf_vxc_clip,
            iterate_selection=args.orbital_scf_iterate_selection,
        )
    )
    neural_scf_molecule, scf_info = scf.run(reference, functional, best_params)

    predicted_total_traincfg = float(
        predict_ground_state_total_energy(
            best_params,
            functional,
            reference,
            training_config=training_config,
        )
    )
    predicted_total_orbitalcfg = float(
        predict_ground_state_total_energy(
            best_params,
            functional,
            reference,
            training_config=orbital_eval_config,
        )
    )

    td = tdscf.TDDFT(
        neural_scf_molecule,
        xc_functional=functional,
        xc_params=best_params,
    )
    try:
        result = td.kernel(nstates=nstates)
        solver_label = "Casida"
    except Exception:
        result = tdscf.TDA(
            neural_scf_molecule,
            xc_functional=functional,
            xc_params=best_params,
        ).kernel(nstates=nstates)
        solver_label = "TDA fallback"
    if result.excitation_energies.size == 0:
        result = tdscf.TDA(
            neural_scf_molecule,
            xc_functional=functional,
            xc_params=best_params,
        ).kernel(nstates=nstates)
        solver_label = "TDA fallback"

    neural_energies_au = jnp.asarray(result.excitation_energies)
    neural_osc = oscillator_strengths(neural_scf_molecule, result)
    ncompare = int(min(ref_energies_au.size, neural_energies_au.size, nstates))
    excitation_mae_ev = float(
        jnp.mean(
            jnp.abs(
                ref_energies_au[:ncompare] * HARTREE_TO_EV
                - neural_energies_au[:ncompare] * HARTREE_TO_EV
            )
        )
    )

    grid_ev = jnp.linspace(args.grid_min_ev, args.grid_max_ev, args.grid_points)
    ref_curve = lorentzian_spectrum(
        ref_energies_au * HARTREE_TO_EV,
        ref_osc,
        grid_ev,
        eta=args.eta_ev,
    )
    neural_curve = lorentzian_spectrum(
        neural_energies_au * HARTREE_TO_EV,
        neural_osc,
        grid_ev,
        eta=args.eta_ev,
    )

    params_ckpt, params_meta = save_params_checkpoint(
        outdir / "neural_xc_params.msgpack",
        best_params,
        metadata={
            "basis": args.basis,
            "xc": args.xc,
            "steps": int(args.steps),
            "learning_rate": float(args.learning_rate),
            "lr_decay_every": int(args.lr_decay_every),
            "lr_decay_factor": float(args.lr_decay_factor),
            "hidden_dims": list(args.hidden_dims),
            "semilocal_xc": semilocal_xc,
            "response_hf_mode": args.response_hf_mode,
            "feature_mode": args.feature_mode,
            "dm21_omega_values": list(float(x) for x in args.dm21_omega_values),
            "density_weight": float(args.density_weight),
            "xc_potential_weight": float(args.xc_potential_weight),
            "xc_kernel_weight": float(args.xc_kernel_weight),
            "xc_kernel_target_clip": float(args.xc_kernel_target_clip),
            "xc_kernel_normalize": str(args.xc_kernel_normalize),
            "xc_kernel_normalize_eps": float(args.xc_kernel_normalize_eps),
            "xc_kernel_normalization_scale": float(xc_kernel_normalization_scale),
            "total_derivative_delta": float(args.total_derivative_delta),
            "target_total_d1": float(target_total_d1),
            "target_total_d2": float(target_total_d2),
            "response_kernel_clip": float(args.response_kernel_clip),
            "orbital_loss_weight": float(args.orbital_loss_weight),
            "orbital_loss_window": int(args.orbital_loss_window),
            "orbital_loss_mode": args.orbital_loss_mode,
            "orbital_loss_gradient": args.orbital_loss_gradient,
            "orbital_loss_scf_max_cycle": int(args.orbital_loss_scf_max_cycle),
            "orbital_loss_scf_damping": float(args.orbital_loss_scf_damping),
            "orbital_loss_scf_conv_tol_density": float(args.orbital_loss_scf_conv_tol_density),
            "orbital_loss_scf_vxc_clip": float(args.orbital_loss_scf_vxc_clip),
            "orbital_loss_scf_gradient_mode": args.orbital_loss_scf_gradient_mode,
            "orbital_loss_scf_implicit_max_iter": int(args.orbital_loss_scf_implicit_max_iter),
            "orbital_loss_scf_implicit_step_size": float(args.orbital_loss_scf_implicit_step_size),
            "orbital_loss_scf_implicit_clip": float(args.orbital_loss_scf_implicit_clip),
            "orbital_loss_scf_iterate_selection": args.orbital_loss_scf_iterate_selection,
            "s1_weight": float(args.s1_weight),
            "s1_use_tda": bool(args.s1_use_tda),
            "train_scf_mode": args.train_scf_mode,
            "train_scf_max_cycle": int(args.train_scf_max_cycle),
            "train_scf_damping": float(args.train_scf_damping),
            "train_scf_conv_tol_density": float(args.train_scf_conv_tol_density),
            "train_scf_vxc_clip": float(args.train_scf_vxc_clip),
            "train_scf_gradient_mode": args.train_scf_gradient_mode,
            "train_scf_implicit_max_iter": int(args.train_scf_implicit_max_iter),
            "train_scf_implicit_step_size": float(args.train_scf_implicit_step_size),
            "train_scf_implicit_clip": float(args.train_scf_implicit_clip),
            "train_scf_iterate_selection": args.train_scf_iterate_selection,
            "stationarity_weight": float(args.stationarity_weight),
            "self_consistent_energy_weight": float(args.self_consistent_energy_weight),
            "coefficient_prior_weight": float(args.coefficient_prior_weight),
            "coefficient_prior_values": coefficient_prior_values,
            "coefficient_prior_mode": args.coefficient_prior_mode,
            "best_step": int(best_step),
            "best_loss": float(best_loss),
            "nan_grad_fallback_steps": int(nan_grad_fallback_steps),
        },
    )

    training_csv = outdir / "training_loss.csv"
    training_png = outdir / "training_loss.png"
    spectrum_csv = outdir / "spectrum_compare.csv"
    spectrum_png = outdir / "spectrum_compare.png"
    state_csv = outdir / "state_compare.csv"
    state_energy_png = outdir / "state_energy_compare.png"
    summary_path = outdir / "summary.txt"

    _write_training_curve(
        training_csv,
        total_history,
        ground_history,
        orbital_mse_history,
        orbital_weighted_history,
    )
    _plot_training_curve(
        training_png,
        total_history,
        ground_history,
        orbital_weighted_history,
        orbital_loss_weight=orbital_loss_weight,
    )
    _write_spectrum_csv(spectrum_csv, grid_ev, ref_curve, neural_curve)
    _write_state_csv(state_csv, ref_energies_au, ref_osc, neural_energies_au, neural_osc)
    _plot_excitation_energy_compare(
        state_energy_png,
        ref_energies_au=ref_energies_au,
        neural_energies_au=neural_energies_au,
    )
    _plot_spectrum(
        spectrum_png,
        grid_ev=grid_ev,
        ref_curve=ref_curve,
        neural_curve=neural_curve,
        excitation_mae_ev=excitation_mae_ev,
        solver_label=solver_label,
    )

    (
        orbital_images,
        orbital_diff_norms,
        orbital_overlaps,
        orbital_diff_isos,
        orbital_diff_scales,
    ) = render_restricted_orbital_surfaces(
        reference_mol=mol,
        reference_mo_coeff=mf.mo_coeff,
        reference_mo_occ=mf.mo_occ,
        neural_molecule=neural_scf_molecule,
        overlap_matrix=reference.overlap_matrix,
        xyzrender_src=args.xyzrender_src,
        outdir=outdir,
        iso=args.orbital_iso,
        diff_iso=args.orbital_diff_iso,
        mo_blur=args.orbital_mo_blur,
        mo_upsample=args.orbital_mo_upsample,
        cube_grid=args.orbital_cube_grid,
        canvas_size=args.orbital_canvas_size,
        match_frontier_by_overlap=not args.disable_orbital_frontier_match,
        frontier_match_window=args.orbital_frontier_match_window,
    )
    compare_dir = outdir / "orbital_surfaces" / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    for label, files in orbital_images.items():
        plot_orbital_compare_panel(
            orbital_label=label,
            ref_png=files["reference"],
            neural_png=files["neural"],
            diff_png=files["difference"],
            iso=args.orbital_iso,
            diff_iso=orbital_diff_isos[label],
            overlap_val=orbital_overlaps[label],
            diff_norm=orbital_diff_norms[label],
            diff_scale=orbital_diff_scales[label],
            out_png=compare_dir / f"{label.replace('+', 'p').replace('-', 'm').lower()}_real_vs_neural.png",
        )

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("system=H2O\n")
        handle.write(f"basis={args.basis}\n")
        handle.write(f"xc={args.xc}\n")
        handle.write(f"semilocal_xc={semilocal_xc}\n")
        handle.write(f"feature_mode={args.feature_mode}\n")
        handle.write(f"dm21_omega_values={tuple(float(x) for x in args.dm21_omega_values)}\n")
        handle.write(f"hidden_dims={tuple(args.hidden_dims)}\n")
        handle.write(f"steps={args.steps}\n")
        handle.write(f"learning_rate={args.learning_rate}\n")
        handle.write(f"lr_decay_every={args.lr_decay_every}\n")
        handle.write(f"lr_decay_factor={args.lr_decay_factor}\n")
        handle.write(f"jax_enable_x64={os.environ.get('JAX_ENABLE_X64', '')}\n")
        handle.write(f"density_weight={args.density_weight}\n")
        handle.write(f"xc_potential_weight={args.xc_potential_weight}\n")
        handle.write(f"xc_kernel_weight={args.xc_kernel_weight}\n")
        handle.write(f"xc_kernel_target_clip={args.xc_kernel_target_clip}\n")
        handle.write(f"xc_kernel_normalize={args.xc_kernel_normalize}\n")
        handle.write(f"xc_kernel_normalize_eps={args.xc_kernel_normalize_eps}\n")
        handle.write(f"xc_kernel_normalization_scale={xc_kernel_normalization_scale}\n")
        handle.write(f"total_derivative_delta={args.total_derivative_delta}\n")
        handle.write(f"target_total_d1={target_total_d1:.16e}\n")
        handle.write(f"target_total_d2={target_total_d2:.16e}\n")
        handle.write(f"response_kernel_clip={args.response_kernel_clip}\n")
        handle.write(f"orbital_loss_weight={args.orbital_loss_weight}\n")
        handle.write(f"orbital_loss_window={args.orbital_loss_window}\n")
        handle.write(f"orbital_loss_mode={args.orbital_loss_mode}\n")
        handle.write(f"orbital_loss_gradient={args.orbital_loss_gradient}\n")
        handle.write(f"orbital_loss_window_start={window_start}\n")
        handle.write(f"orbital_loss_window_stop_exclusive={window_stop}\n")
        handle.write(f"orbital_loss_window_size={window_stop - window_start}\n")
        handle.write(f"orbital_loss_ref_homo_index={window_homo}\n")
        handle.write(f"orbital_loss_ref_lumo_index={window_lumo}\n")
        handle.write(f"orbital_loss_scf_max_cycle={args.orbital_loss_scf_max_cycle}\n")
        handle.write(f"orbital_loss_scf_damping={args.orbital_loss_scf_damping}\n")
        handle.write(
            "orbital_loss_scf_conv_tol_density="
            f"{float(args.orbital_loss_scf_conv_tol_density):.6e}\n"
        )
        handle.write(f"orbital_loss_scf_vxc_clip={args.orbital_loss_scf_vxc_clip}\n")
        handle.write(f"orbital_loss_scf_gradient_mode={args.orbital_loss_scf_gradient_mode}\n")
        handle.write(
            f"orbital_loss_scf_implicit_max_iter={args.orbital_loss_scf_implicit_max_iter}\n"
        )
        handle.write(
            f"orbital_loss_scf_implicit_step_size={args.orbital_loss_scf_implicit_step_size}\n"
        )
        handle.write(f"orbital_loss_scf_implicit_clip={args.orbital_loss_scf_implicit_clip}\n")
        handle.write(
            f"orbital_loss_scf_iterate_selection={args.orbital_loss_scf_iterate_selection}\n"
        )
        handle.write(f"s1_weight={args.s1_weight}\n")
        handle.write(f"s1_use_tda={bool(args.s1_use_tda)}\n")
        handle.write(f"train_scf_mode={args.train_scf_mode}\n")
        handle.write(f"train_scf_max_cycle={int(args.train_scf_max_cycle)}\n")
        handle.write(f"train_scf_damping={float(args.train_scf_damping)}\n")
        handle.write(
            "train_scf_conv_tol_density="
            f"{float(args.train_scf_conv_tol_density):.6e}\n"
        )
        handle.write(f"train_scf_vxc_clip={float(args.train_scf_vxc_clip)}\n")
        handle.write(f"train_scf_gradient_mode={args.train_scf_gradient_mode}\n")
        handle.write(f"train_scf_implicit_max_iter={int(args.train_scf_implicit_max_iter)}\n")
        handle.write(f"train_scf_implicit_step_size={float(args.train_scf_implicit_step_size)}\n")
        handle.write(f"train_scf_implicit_clip={float(args.train_scf_implicit_clip)}\n")
        handle.write(f"train_scf_iterate_selection={args.train_scf_iterate_selection}\n")
        handle.write(
            f"orbital_frontier_match={not bool(args.disable_orbital_frontier_match)}\n"
        )
        handle.write(
            f"orbital_frontier_match_window={int(args.orbital_frontier_match_window)}\n"
        )
        handle.write(f"stationarity_weight={args.stationarity_weight}\n")
        handle.write(f"self_consistent_energy_weight={args.self_consistent_energy_weight}\n")
        handle.write(f"coefficient_prior_weight={args.coefficient_prior_weight}\n")
        handle.write(f"coefficient_prior_values={coefficient_prior_values}\n")
        handle.write(f"coefficient_prior_mode={args.coefficient_prior_mode}\n")
        handle.write(f"response_hf_mode={args.response_hf_mode}\n")
        handle.write(f"initial_loss={initial_loss:.16e}\n")
        handle.write(f"initial_ground_loss={initial_ground_loss:.16e}\n")
        handle.write(f"initial_orbital_mse={initial_orbital_mse:.16e}\n")
        handle.write(f"initial_orbital_weighted={initial_orbital_weighted:.16e}\n")
        handle.write(f"initial_density_penalty={initial_density_penalty:.16e}\n")
        handle.write(f"initial_xc_potential_penalty={initial_xc_potential_penalty:.16e}\n")
        handle.write(f"initial_xc_potential_mse={initial_xc_potential_mse:.16e}\n")
        handle.write(f"initial_xc_kernel_penalty={initial_xc_kernel_penalty:.16e}\n")
        handle.write(f"initial_xc_kernel_mse={initial_xc_kernel_mse:.16e}\n")
        handle.write(f"initial_total_d1_pred={float(initial_metrics['total_energy_d1_pred'][0]):.16e}\n")
        handle.write(f"initial_total_d1_target={float(initial_metrics['total_energy_d1_target'][0]):.16e}\n")
        handle.write(f"initial_total_d2_pred={float(initial_metrics['total_energy_d2_pred'][0]):.16e}\n")
        handle.write(f"initial_total_d2_target={float(initial_metrics['total_energy_d2_target'][0]):.16e}\n")
        handle.write(
            "initial_self_consistent_energy_penalty="
            f"{initial_self_consistent_energy_penalty:.16e}\n"
        )
        handle.write(
            "initial_coefficient_prior_penalty="
            f"{initial_coefficient_prior_penalty:.16e}\n"
        )
        handle.write(f"initial_stationarity_penalty={initial_stationarity_penalty:.16e}\n")
        handle.write(f"initial_s1_penalty={initial_s1_penalty:.16e}\n")
        handle.write(f"initial_s1_mse={initial_s1_mse:.16e}\n")
        handle.write(f"initial_s1_predicted_au={initial_s1_predicted:.16e}\n")
        handle.write(f"initial_s1_target_au={initial_s1_target:.16e}\n")
        handle.write(f"best_loss={best_loss:.16e}\n")
        handle.write(f"best_step={best_step}\n")
        handle.write(f"final_loss={final_loss:.16e}\n")
        handle.write(f"final_ground_loss={final_ground_loss:.16e}\n")
        handle.write(f"final_orbital_mse={final_orbital_mse:.16e}\n")
        handle.write(f"final_orbital_weighted={final_orbital_weighted:.16e}\n")
        handle.write(f"final_density_penalty={final_density_penalty:.16e}\n")
        handle.write(f"final_xc_potential_penalty={final_xc_potential_penalty:.16e}\n")
        handle.write(f"final_xc_potential_mse={final_xc_potential_mse:.16e}\n")
        handle.write(f"final_xc_kernel_penalty={final_xc_kernel_penalty:.16e}\n")
        handle.write(f"final_xc_kernel_mse={final_xc_kernel_mse:.16e}\n")
        handle.write(f"final_total_d1_pred={float(final_metrics['total_energy_d1_pred'][0]):.16e}\n")
        handle.write(f"final_total_d1_target={float(final_metrics['total_energy_d1_target'][0]):.16e}\n")
        handle.write(f"final_total_d2_pred={float(final_metrics['total_energy_d2_pred'][0]):.16e}\n")
        handle.write(f"final_total_d2_target={float(final_metrics['total_energy_d2_target'][0]):.16e}\n")
        handle.write(
            "final_self_consistent_energy_penalty="
            f"{final_self_consistent_energy_penalty:.16e}\n"
        )
        handle.write(
            "final_coefficient_prior_penalty="
            f"{final_coefficient_prior_penalty:.16e}\n"
        )
        handle.write(f"final_stationarity_penalty={final_stationarity_penalty:.16e}\n")
        handle.write(f"final_s1_penalty={final_s1_penalty:.16e}\n")
        handle.write(f"final_s1_mse={final_s1_mse:.16e}\n")
        handle.write(f"final_s1_predicted_au={final_s1_predicted:.16e}\n")
        handle.write(f"final_s1_target_au={final_s1_target:.16e}\n")
        handle.write(
            "final_s1_abs_err_ev="
            f"{abs(final_s1_predicted - final_s1_target) * HARTREE_TO_EV:.12f}\n"
        )
        reference_energy = float(reference.mf_energy)
        handle.write(f"reference_energy_ha={reference_energy:.12f}\n")
        handle.write(f"predicted_energy_traincfg_ha={predicted_total_traincfg:.12f}\n")
        handle.write(f"predicted_energy_orbitaleval_ha={predicted_total_orbitalcfg:.12f}\n")
        handle.write(
            f"energy_abs_err_traincfg_ha={abs(predicted_total_traincfg - reference_energy):.12f}\n"
        )
        handle.write(
            f"energy_abs_err_orbitaleval_ha={abs(predicted_total_orbitalcfg - reference_energy):.12f}\n"
        )
        # Backward-compatible aliases used by earlier analysis scripts.
        handle.write(f"predicted_energy_fixed_ha={predicted_total_traincfg:.12f}\n")
        handle.write(f"predicted_energy_self_consistent_ha={predicted_total_orbitalcfg:.12f}\n")
        handle.write(
            f"energy_abs_err_fixed_ha={abs(predicted_total_traincfg - reference_energy):.12f}\n"
        )
        handle.write(
            f"energy_abs_err_self_consistent_ha={abs(predicted_total_orbitalcfg - reference_energy):.12f}\n"
        )
        handle.write(f"solver={solver_label}\n")
        handle.write(f"nstates={nstates}\n")
        handle.write(f"excitation_mae_ev={excitation_mae_ev:.12f}\n")
        handle.write(f"state_energy_compare_png={state_energy_png}\n")
        handle.write(f"reference_elapsed_s={reference_elapsed:.2f}\n")
        handle.write(f"training_elapsed_s={train_elapsed:.2f}\n")
        handle.write(f"nan_grad_fallback_steps={nan_grad_fallback_steps}\n")
        handle.write(f"orbital_scf_converged={bool(np.asarray(scf_info.converged))}\n")
        handle.write(f"orbital_scf_cycles={int(np.asarray(scf_info.cycles))}\n")
        handle.write(f"orbital_scf_selected_cycle={int(np.asarray(scf_info.selected_cycle))}\n")
        handle.write(
            "orbital_scf_selected_rms_density="
            f"{float(np.asarray(scf_info.selected_rms_density)):.6e}\n"
        )
        handle.write(f"params_checkpoint={params_ckpt}\n")
        handle.write(f"params_meta={params_meta}\n")
        for label in ("HOMO-1", "HOMO", "LUMO", "LUMO+1"):
            handle.write(f"{label}_overlap={orbital_overlaps[label]:.8f}\n")
            handle.write(f"{label}_diff_norm={orbital_diff_norms[label]:.8f}\n")
            handle.write(f"{label}_diff_iso={orbital_diff_isos[label]:.8f}\n")
            handle.write(f"{label}_diff_scale={orbital_diff_scales[label]:.8f}\n")

    print(f"summary={summary_path}")
    print(f"spectrum_png={spectrum_png}")
    print(f"state_energy_png={state_energy_png}")
    print(f"training_png={training_png}")
    print(f"orbital_compare_dir={compare_dir}")


if __name__ == "__main__":
    main()
