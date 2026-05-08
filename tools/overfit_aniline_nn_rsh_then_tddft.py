from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from pyscf import dft, gto

from td_graddft.nn_rsh import (
    AtomCenteredDensityDescriptorConfig,
    ResolvedRSHParameters,
    make_atom_centered_density_rsh_functional,
    make_self_supervised_rsh_loss,
)
from td_graddft.reference_legacy import restricted_reference_from_pyscf
from td_graddft.spectra import HARTREE_TO_EV, lorentzian_spectrum
from td_graddft.training import (
    GroundStateDatum,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    load_params_checkpoint,
    make_ground_state_train_step,
    save_params_checkpoint,
)


ANILINE_GEOMETRY = """
N   -2.4046    0.0000    0.0005
C   -0.9941   -0.0002   -0.0003
C   -0.2969    1.2079   -0.0003
C   -0.2965   -1.2080   -0.0003
C    1.0980    1.2080    0.0001
C    1.0984   -1.2078    0.0001
C    1.7957    0.0002    0.0003
H   -0.8289    2.1558   -0.0003
H   -0.8283   -2.1561   -0.0002
H    1.6411    2.1486    0.0002
H    1.6417   -2.1482    0.0002
H    2.8818    0.0004    0.0006
H   -2.9109   -0.8755   -0.0005
H   -2.9107    0.8756   -0.0006
"""

EV_TO_NM = 1239.8419843320026
NIST_WEBBOOK_URL = "https://webbook.nist.gov/cgi/cbook.cgi?ID=C62533&Mask=484"
NIST_JCAMP_URL = "https://webbook.nist.gov/cgi/cbook.cgi?Index=0&JCAMP=C62533&Type=UVVis"
PUBCHEM_SDF_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/6115/record/SDF/?record_type=3d"
DEFAULT_EXPERIMENT_CSV = (
    Path(__file__).resolve().parent / "benchmark_data" / "aniline_nist_uvvis_1949.csv"
)
PEAK_WINDOWS_NM: dict[str, tuple[float, float]] = {
    "short_wave_band": (215.0, 235.0),
    "long_wave_band": (300.0, 340.0),
}


@dataclass(frozen=True)
class SpectrumResult:
    label: str
    scf_energy_au: float
    excitation_energies_ev: np.ndarray
    oscillator_strengths: np.ndarray
    normalized_curve: np.ndarray
    peaks_nm: dict[str, float]
    peaks_ev: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-system aniline benchmark: self-supervised ground-state nn-RSH overfit "
            "followed by PySCF TDDFT/TDA excited-state evaluation and experiment comparison."
        ),
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Global gradient clipping norm applied before Adam updates.",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine", "exponential"),
        default="cosine",
    )
    parser.add_argument("--final-learning-rate-scale", type=float, default=0.05)
    parser.add_argument("--basis", default="6-31+g*")
    parser.add_argument("--training-grid-level", type=int, default=0)
    parser.add_argument("--excited-grid-level", type=int, default=1)
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--scf-max-cycle", type=int, default=4)
    parser.add_argument("--scf-level-shift", type=float, default=0.0)
    parser.add_argument(
        "--scf-gradient-mode",
        choices=("unrolled", "implicit_commutator"),
        default="implicit_commutator",
    )
    parser.add_argument(
        "--janak-mode",
        choices=(
            "finite_difference",
            "autodiff",
            "full_scf_ad",
            "fixed_orbital_ad",
            "half_charge_ad",
        ),
        default="finite_difference",
    )
    parser.add_argument("--janak-delta", type=float, default=0.1)
    parser.add_argument("--fractional-delta", type=float, default=0.1)
    parser.add_argument("--fractional-branch-scf-max-cycle", type=int, default=None)
    parser.add_argument("--fractional-branch-scf-damping", type=float, default=None)
    parser.add_argument("--fractional-branch-scf-level-shift", type=float, default=None)
    parser.add_argument(
        "--fractional-branch-scf-iterate-selection",
        choices=("final", "best_rms", "first_converged"),
        default=None,
    )
    parser.add_argument("--janak-weight", type=float, default=1.0)
    parser.add_argument("--fractional-weight", type=float, default=0.0)
    parser.add_argument("--koopmans-ip-weight", type=float, default=0.0)
    parser.add_argument("--koopmans-ea-weight", type=float, default=0.0)
    parser.add_argument("--koopmans-lumo-ea-weight", type=float, default=0.0)
    parser.add_argument(
        "--koopmans-detach-charged-states",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--koopmans-differentiate-charged-orbitals",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--prior-weight", type=float, default=1e-3)
    parser.add_argument("--omega-grid", default="0.0,0.3,0.6")
    parser.add_argument("--radial-centers", default="0.6,1.4,2.6")
    parser.add_argument("--radial-width", type=float, default=0.5)
    parser.add_argument("--max-angular", type=int, default=1)
    parser.add_argument("--atom-hidden-dims", default="16,16")
    parser.add_argument("--pooled-hidden-dims", default="16")
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--initialize-from-template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start training from the functional template defaults instead of a random RSH point.",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--nstates", type=int, default=16)
    parser.add_argument("--eta-ev", type=float, default=0.12)
    parser.add_argument(
        "--solver",
        choices=("tddft", "tda"),
        default="tddft",
    )
    parser.add_argument(
        "--baseline-xc",
        default="cam-b3lyp",
        help="Optional PySCF baseline XC. Use 'none' to disable.",
    )
    parser.add_argument(
        "--experiment-csv",
        default=str(DEFAULT_EXPERIMENT_CSV),
    )
    parser.add_argument(
        "--skip-excited-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only run ground-state training and checkpointing; skip post-training TDDFT/TDA evaluation.",
    )
    parser.add_argument("--outdir", default="outputs/aniline_nn_rsh_ground_then_excited")
    return parser.parse_args()


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _make_learning_rate_schedule(args: argparse.Namespace):
    base_lr = float(args.learning_rate)
    final_scale = float(args.final_learning_rate_scale)
    if args.lr_schedule == "constant":
        return optax.constant_schedule(base_lr)
    if args.lr_schedule == "cosine":
        return optax.cosine_decay_schedule(
            init_value=base_lr,
            decay_steps=max(int(args.steps), 1),
            alpha=max(min(final_scale, 1.0), 0.0),
        )
    if args.lr_schedule == "exponential":
        return optax.exponential_decay(
            init_value=base_lr,
            transition_steps=max(int(args.steps), 1),
            decay_rate=max(final_scale, 1e-8),
            staircase=False,
        )
    raise ValueError(f"Unsupported lr_schedule={args.lr_schedule!r}.")


def _make_optimizer(args: argparse.Namespace):
    lr_schedule = _make_learning_rate_schedule(args)
    transforms = []
    if args.grad_clip_norm is not None and float(args.grad_clip_norm) > 0.0:
        transforms.append(optax.clip_by_global_norm(float(args.grad_clip_norm)))
    transforms.append(optax.adam(lr_schedule))
    return optax.chain(*transforms), lr_schedule


def _metric_scalar(metrics: dict[str, jnp.ndarray], key: str) -> float:
    arr = jnp.asarray(metrics[key])
    if arr.ndim == 0:
        return float(arr)
    return float(arr.reshape(-1)[0])


def _build_aniline_mf(*, basis: str, xc: str, grid_level: int) -> Any:
    mol = gto.Mole()
    mol.atom = ANILINE_GEOMETRY
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.spin = 0
    mol.charge = 0
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = int(grid_level)
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    return mf


def _build_aniline_reference(
    *,
    basis: str,
    xc: str,
    grid_level: int,
    omega_grid: tuple[float, ...],
):
    mf = _build_aniline_mf(basis=basis, xc=xc, grid_level=grid_level)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"PySCF SCF did not converge for aniline {xc}/{basis}.")
    return restricted_reference_from_pyscf(
        mf,
        compute_local_hfx_features=True,
        compute_local_hfx_aux=True,
        hfx_omega_values=omega_grid,
    )


def _default_resolved(functional: Any) -> ResolvedRSHParameters:
    template = functional.template
    return ResolvedRSHParameters(
        sr_hf_fraction=template.default_sr_hf_fraction,
        lr_hf_fraction=template.default_lr_hf_fraction,
        omega=template.default_omega,
    )


def _write_training_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def _load_experiment_curve(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    wavelengths_nm = np.asarray(data["wavelength_nm"], dtype=float)
    log_epsilon = np.asarray(data["log_epsilon"], dtype=float)
    linear_epsilon = np.power(10.0, log_epsilon)
    normalized = linear_epsilon / np.maximum(np.max(linear_epsilon), 1e-12)
    return wavelengths_nm, log_epsilon, normalized


def _peak_in_window(
    wavelengths_nm: np.ndarray,
    values: np.ndarray,
    *,
    window_nm: tuple[float, float],
) -> tuple[float, float]:
    lo, hi = window_nm
    mask = (wavelengths_nm >= lo) & (wavelengths_nm <= hi)
    if not np.any(mask):
        return float("nan"), float("nan")
    local_wavelengths = wavelengths_nm[mask]
    local_values = values[mask]
    idx = int(np.argmax(local_values))
    peak_nm = float(local_wavelengths[idx])
    peak_ev = float(EV_TO_NM / peak_nm)
    return peak_nm, peak_ev


def _evaluate_curve_on_wavelength_grid(
    energies_ev: np.ndarray,
    strengths: np.ndarray,
    wavelengths_nm: np.ndarray,
    *,
    eta_ev: float,
) -> np.ndarray:
    energy_grid_ev = EV_TO_NM / wavelengths_nm
    curve = np.asarray(
        lorentzian_spectrum(
            jnp.asarray(energies_ev),
            jnp.asarray(strengths),
            jnp.asarray(energy_grid_ev),
            eta=float(eta_ev),
        ),
        dtype=float,
    )
    return curve / np.maximum(np.max(curve), 1e-12)


def _evaluate_pyscf_excited_states(
    *,
    label: str,
    basis: str,
    grid_level: int,
    solver: str,
    nstates: int,
    eta_ev: float,
    wavelengths_nm: np.ndarray,
    xc: str | None = None,
    pyscf_spec: Any | None = None,
) -> SpectrumResult:
    if (xc is None) == (pyscf_spec is None):
        raise ValueError("Provide exactly one of xc or pyscf_spec.")

    mf = _build_aniline_mf(
        basis=basis,
        xc="pbe" if pyscf_spec is not None else str(xc),
        grid_level=grid_level,
    )
    if pyscf_spec is not None:
        pyscf_spec.install_into_mf(mf)

    total_energy = float(mf.kernel())
    if not mf.converged:
        raise RuntimeError(f"PySCF SCF did not converge for spectrum model {label}.")

    td = mf.TDA() if solver == "tda" else mf.TDDFT()
    td.nstates = int(nstates)
    td.conv_tol = 1e-7
    td.kernel()
    energies_ev = np.asarray(td.e, dtype=float) * HARTREE_TO_EV
    strengths = np.asarray(td.oscillator_strength(), dtype=float)
    curve = _evaluate_curve_on_wavelength_grid(
        energies_ev,
        strengths,
        wavelengths_nm,
        eta_ev=eta_ev,
    )
    peaks_nm: dict[str, float] = {}
    peaks_ev: dict[str, float] = {}
    for name, window in PEAK_WINDOWS_NM.items():
        peak_nm, peak_ev = _peak_in_window(wavelengths_nm, curve, window_nm=window)
        peaks_nm[name] = peak_nm
        peaks_ev[name] = peak_ev

    return SpectrumResult(
        label=label,
        scf_energy_au=total_energy,
        excitation_energies_ev=energies_ev,
        oscillator_strengths=strengths,
        normalized_curve=curve,
        peaks_nm=peaks_nm,
        peaks_ev=peaks_ev,
    )


def _write_state_lines_csv(path: Path, results: list[SpectrumResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "state",
                "energy_ev",
                "wavelength_nm",
                "oscillator_strength",
            ]
        )
        for result in results:
            for idx, (energy_ev, osc) in enumerate(
                zip(result.excitation_energies_ev, result.oscillator_strengths, strict=False),
                start=1,
            ):
                writer.writerow(
                    [
                        result.label,
                        idx,
                        float(energy_ev),
                        float(EV_TO_NM / energy_ev),
                        float(osc),
                    ]
                )


def _write_curve_csv(
    path: Path,
    wavelengths_nm: np.ndarray,
    log_epsilon: np.ndarray,
    exp_curve: np.ndarray,
    results: list[SpectrumResult],
) -> None:
    headers = ["wavelength_nm", "experiment_log_epsilon", "experiment_normalized"]
    headers.extend(f"{result.label}_normalized" for result in results)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for idx in range(int(wavelengths_nm.size)):
            row = [
                float(wavelengths_nm[idx]),
                float(log_epsilon[idx]),
                float(exp_curve[idx]),
            ]
            row.extend(float(result.normalized_curve[idx]) for result in results)
            writer.writerow(row)


def _write_peak_csv(
    path: Path,
    exp_peaks_nm: dict[str, float],
    exp_peaks_ev: dict[str, float],
    results: list[SpectrumResult],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "band",
                "experiment_peak_nm",
                "experiment_peak_ev",
                "model",
                "predicted_peak_nm",
                "predicted_peak_ev",
                "abs_error_nm",
                "abs_error_ev",
            ]
        )
        for band in PEAK_WINDOWS_NM:
            for result in results:
                pred_nm = result.peaks_nm[band]
                pred_ev = result.peaks_ev[band]
                exp_nm = exp_peaks_nm[band]
                exp_ev = exp_peaks_ev[band]
                writer.writerow(
                    [
                        band,
                        exp_nm,
                        exp_ev,
                        result.label,
                        pred_nm,
                        pred_ev,
                        abs(pred_nm - exp_nm),
                        abs(pred_ev - exp_ev),
                    ]
                )


def _plot_spectra(
    path: Path,
    wavelengths_nm: np.ndarray,
    exp_curve: np.ndarray,
    results: list[SpectrumResult],
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.plot(wavelengths_nm, exp_curve, color="black", lw=2.4, label="Experiment (NIST, normalized)")
    for result in results:
        ax.plot(wavelengths_nm, result.normalized_curve, lw=2.0, label=result.label)
    for band, (lo, hi) in PEAK_WINDOWS_NM.items():
        ax.axvspan(lo, hi, color="#d9d9d9", alpha=0.15)
        ax.text(
            0.5 * (lo + hi),
            1.02,
            band,
            ha="center",
            va="bottom",
            fontsize=9,
            transform=ax.get_xaxis_transform(),
        )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized intensity")
    ax.set_title("Aniline Absorption Spectrum")
    ax.set_xlim(float(np.min(wavelengths_nm)), float(np.max(wavelengths_nm)))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    omega_grid = _parse_float_tuple(args.omega_grid)
    descriptor_config = AtomCenteredDensityDescriptorConfig(
        radial_centers=_parse_float_tuple(args.radial_centers),
        radial_width=float(args.radial_width),
        max_angular=int(args.max_angular),
    )
    atom_hidden_dims = _parse_int_tuple(args.atom_hidden_dims)
    pooled_hidden_dims = _parse_int_tuple(args.pooled_hidden_dims)
    wavelengths_nm, log_epsilon, exp_curve = _load_experiment_curve(Path(args.experiment_csv))

    molecule = _build_aniline_reference(
        basis=args.basis,
        xc=args.xc,
        grid_level=args.training_grid_level,
        omega_grid=omega_grid,
    )
    functional = make_atom_centered_density_rsh_functional(
        local_xc_spec=args.xc,
        descriptor_config=descriptor_config,
        atom_hidden_dims=atom_hidden_dims,
        pooled_hidden_dims=pooled_hidden_dims,
        embedding_dim=int(args.embedding_dim),
        fallback_omega_values=omega_grid,
    )

    optimizer, lr_schedule = _make_optimizer(args)
    if args.checkpoint is not None:
        template_state = create_train_state_from_molecule(
            functional,
            jax.random.PRNGKey(int(args.seed)),
            molecule,
            optimizer,
        )
        loaded = load_params_checkpoint(args.checkpoint, template=template_state.params)
        state = template_state.replace(params=loaded)
    else:
        state = create_train_state_from_molecule(
            functional,
            jax.random.PRNGKey(int(args.seed)),
            molecule,
            optimizer,
        )

    default_resolved = _default_resolved(functional)
    default_params = functional.params_with_resolved(
        state.params,
        default_resolved,
        molecule=molecule,
        preserve_network=True,
    )
    if bool(args.initialize_from_template) and args.checkpoint is None:
        state = state.replace(params=default_params)
    default_sr = float(default_resolved.sr_hf_fraction)
    default_lr = float(default_resolved.lr_hf_fraction)
    default_omega = float(default_resolved.omega)

    loss_fn = make_self_supervised_rsh_loss(
        functional,
        training_config=GroundStateTrainingConfig(
            mode="self_consistent",
            scf_gradient_mode=str(args.scf_gradient_mode),
            scf_max_cycle=int(args.scf_max_cycle),
            scf_level_shift=float(args.scf_level_shift),
            janak_frontier_mode=str(args.janak_mode),
            janak_frontier_delta=float(args.janak_delta),
            fractional_linearity_delta=float(args.fractional_delta),
            fractional_branch_scf_max_cycle=(
                None
                if args.fractional_branch_scf_max_cycle is None
                else int(args.fractional_branch_scf_max_cycle)
            ),
            fractional_branch_scf_damping=(
                None
                if args.fractional_branch_scf_damping is None
                else float(args.fractional_branch_scf_damping)
            ),
            fractional_branch_scf_level_shift=(
                None
                if args.fractional_branch_scf_level_shift is None
                else float(args.fractional_branch_scf_level_shift)
            ),
            fractional_branch_scf_iterate_selection=(
                None
                if args.fractional_branch_scf_iterate_selection is None
                else str(args.fractional_branch_scf_iterate_selection)
            ),
            scf_require_convergence=False,
        ),
        janak_weight=float(args.janak_weight),
        fractional_weight=float(args.fractional_weight),
        koopmans_ip_weight=float(args.koopmans_ip_weight),
        koopmans_ea_weight=float(args.koopmans_ea_weight),
        koopmans_lumo_ea_weight=float(args.koopmans_lumo_ea_weight),
        koopmans_detach_charged_states=bool(args.koopmans_detach_charged_states),
        koopmans_differentiate_charged_orbitals=bool(
            args.koopmans_differentiate_charged_orbitals
        ),
        prior_weight=float(args.prior_weight),
    )
    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.asarray(molecule.mf_energy),
    )
    train_step = make_ground_state_train_step(functional, loss_fn=loss_fn)

    initial_loss, initial_metrics = loss_fn(state.params, functional, datum)
    best_params = state.params
    best_record = {
        "step": 0,
        "loss": float(initial_loss),
        "janak_frontier_mae": _metric_scalar(initial_metrics, "janak_frontier_mae"),
        "koopmans_ip_mae": _metric_scalar(initial_metrics, "koopmans_ip_mae"),
        "koopmans_ea_mae": _metric_scalar(initial_metrics, "koopmans_ea_mae"),
        "koopmans_lumo_ea_mae": _metric_scalar(initial_metrics, "koopmans_lumo_ea_mae"),
        "sr_hf_fraction": float(functional.resolve_parameters(state.params, molecule).sr_hf_fraction),
        "lr_hf_fraction": float(functional.resolve_parameters(state.params, molecule).lr_hf_fraction),
        "omega": float(functional.resolve_parameters(state.params, molecule).omega),
        "learning_rate": float(lr_schedule(0)),
    }
    history: list[dict[str, float]] = []
    best_metric_key = (
        "loss"
        if float(args.janak_weight) == 0.0
        else "janak_frontier_mae"
    )

    for step in range(int(args.steps)):
        params_before = state.params
        state, metrics = train_step(state, datum)
        record = {
            "step": float(step + 1),
            "loss": _metric_scalar(metrics, "loss"),
            "janak_frontier_mae": _metric_scalar(metrics, "janak_frontier_mae"),
            "koopmans_ip_mae": _metric_scalar(metrics, "koopmans_ip_mae"),
            "koopmans_ea_mae": _metric_scalar(metrics, "koopmans_ea_mae"),
            "koopmans_lumo_ea_mae": _metric_scalar(metrics, "koopmans_lumo_ea_mae"),
            "sr_hf_fraction": _metric_scalar(metrics, "sr_hf_fraction"),
            "lr_hf_fraction": _metric_scalar(metrics, "lr_hf_fraction"),
            "omega": _metric_scalar(metrics, "omega"),
            "grad_norm": _metric_scalar(metrics, "grad_norm"),
            "nonfinite_grad_fraction": _metric_scalar(metrics, "nonfinite_grad_fraction"),
            "learning_rate": float(lr_schedule(step)),
        }
        history.append(record)
        if record[best_metric_key] < best_record[best_metric_key]:
            best_record = dict(record)
            # train_step metrics are evaluated on params_before, then the optimizer
            # update is applied. Keep the checkpoint aligned with the reported metrics.
            best_params = params_before
        print(
            f"step={step + 1:03d} "
            f"loss={record['loss']:.6e} "
            f"janak_mae={record['janak_frontier_mae']:.6e} "
            f"kip={record['koopmans_ip_mae']:.3e} "
            f"kea={record['koopmans_ea_mae']:.3e} "
            f"klumo={record['koopmans_lumo_ea_mae']:.3e} "
            f"sr={record['sr_hf_fraction']:.8f} "
            f"lr={record['lr_hf_fraction']:.8f} "
            f"omega={record['omega']:.8f} "
            f"d_sr={record['sr_hf_fraction'] - default_sr:+.3e} "
            f"d_lr={record['lr_hf_fraction'] - default_lr:+.3e} "
            f"d_omega={record['omega'] - default_omega:+.3e} "
            f"grad={record['grad_norm']:.3e}",
            flush=True,
        )

    final_loss, final_metrics = loss_fn(state.params, functional, datum)
    final_resolved = functional.resolve_parameters(state.params, molecule)
    trained_resolved = functional.resolve_parameters(best_params, molecule)

    training_summary = {
        "initial_loss": float(initial_loss),
        "initial_janak_frontier_mae": _metric_scalar(initial_metrics, "janak_frontier_mae"),
        "final_loss": float(final_loss),
        "final_janak_frontier_mae": _metric_scalar(final_metrics, "janak_frontier_mae"),
        "final_koopmans_ip_mae": _metric_scalar(final_metrics, "koopmans_ip_mae"),
        "final_koopmans_ea_mae": _metric_scalar(final_metrics, "koopmans_ea_mae"),
        "final_koopmans_lumo_ea_mae": _metric_scalar(final_metrics, "koopmans_lumo_ea_mae"),
        "final_sr_hf_fraction": float(final_resolved.sr_hf_fraction),
        "final_lr_hf_fraction": float(final_resolved.lr_hf_fraction),
        "final_omega": float(final_resolved.omega),
        "best_step": int(best_record["step"]),
        "best_loss": float(best_record["loss"]),
        "best_metric": best_metric_key,
        "best_janak_frontier_mae": float(best_record["janak_frontier_mae"]),
        "best_koopmans_ip_mae": float(best_record["koopmans_ip_mae"]),
        "best_koopmans_ea_mae": float(best_record["koopmans_ea_mae"]),
        "best_koopmans_lumo_ea_mae": float(best_record["koopmans_lumo_ea_mae"]),
        "best_sr_hf_fraction": float(trained_resolved.sr_hf_fraction),
        "best_lr_hf_fraction": float(trained_resolved.lr_hf_fraction),
        "best_omega": float(trained_resolved.omega),
        "steps": int(args.steps),
        "basis": str(args.basis),
        "training_grid_level": int(args.training_grid_level),
        "scf_gradient_mode": str(args.scf_gradient_mode),
        "koopmans_detach_charged_states": bool(args.koopmans_detach_charged_states),
        "koopmans_differentiate_charged_orbitals": bool(
            args.koopmans_differentiate_charged_orbitals
        ),
        "scf_level_shift": float(args.scf_level_shift),
        "janak_delta": float(args.janak_delta),
        "fractional_delta": float(args.fractional_delta),
        "fractional_branch_scf_max_cycle": (
            None
            if args.fractional_branch_scf_max_cycle is None
            else int(args.fractional_branch_scf_max_cycle)
        ),
        "fractional_branch_scf_damping": (
            None
            if args.fractional_branch_scf_damping is None
            else float(args.fractional_branch_scf_damping)
        ),
        "fractional_branch_scf_level_shift": (
            None
            if args.fractional_branch_scf_level_shift is None
            else float(args.fractional_branch_scf_level_shift)
        ),
        "fractional_branch_scf_iterate_selection": (
            None
            if args.fractional_branch_scf_iterate_selection is None
            else str(args.fractional_branch_scf_iterate_selection)
        ),
        "lr_schedule": str(args.lr_schedule),
        "learning_rate": float(args.learning_rate),
        "grad_clip_norm": (
            None if args.grad_clip_norm is None else float(args.grad_clip_norm)
        ),
        "final_learning_rate_scale": float(args.final_learning_rate_scale),
        "skip_excited_eval": bool(args.skip_excited_eval),
    }
    (outdir / "training_summary.json").write_text(
        json.dumps(training_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_training_history_csv(outdir / "training_history.csv", history)
    save_params_checkpoint(
        outdir / "best_params.msgpack",
        best_params,
        metadata=training_summary,
    )

    if bool(args.skip_excited_eval):
        summary_path = outdir / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "molecule": "aniline",
                    "training": training_summary,
                    "excited_state": None,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"wrote {summary_path}")
        return

    exp_peaks_nm: dict[str, float] = {}
    exp_peaks_ev: dict[str, float] = {}
    for name, window in PEAK_WINDOWS_NM.items():
        peak_nm, peak_ev = _peak_in_window(
            wavelengths_nm,
            exp_curve,
            window_nm=window,
        )
        exp_peaks_nm[name] = peak_nm
        exp_peaks_ev[name] = peak_ev

    spectrum_results: list[SpectrumResult] = []
    baseline_xc = str(args.baseline_xc).strip().lower()
    if baseline_xc and baseline_xc != "none":
        spectrum_results.append(
            _evaluate_pyscf_excited_states(
                label=f"baseline_{baseline_xc}",
                basis=args.basis,
                grid_level=int(args.excited_grid_level),
                solver=str(args.solver),
                nstates=int(args.nstates),
                eta_ev=float(args.eta_ev),
                wavelengths_nm=wavelengths_nm,
                xc=baseline_xc,
            )
        )

    default_bound = functional.bind_to_molecule(default_params, molecule)
    spectrum_results.append(
        _evaluate_pyscf_excited_states(
            label="default_nn_rsh",
            basis=args.basis,
            grid_level=int(args.excited_grid_level),
            solver=str(args.solver),
            nstates=int(args.nstates),
            eta_ev=float(args.eta_ev),
            wavelengths_nm=wavelengths_nm,
            pyscf_spec=default_bound.to_pyscf_spec(),
        )
    )

    trained_bound = functional.bind_to_molecule(best_params, molecule)
    spectrum_results.append(
        _evaluate_pyscf_excited_states(
            label="trained_nn_rsh",
            basis=args.basis,
            grid_level=int(args.excited_grid_level),
            solver=str(args.solver),
            nstates=int(args.nstates),
            eta_ev=float(args.eta_ev),
            wavelengths_nm=wavelengths_nm,
            pyscf_spec=trained_bound.to_pyscf_spec(),
        )
    )

    _write_state_lines_csv(outdir / "state_lines.csv", spectrum_results)
    _write_curve_csv(
        outdir / "spectrum_curves.csv",
        wavelengths_nm,
        log_epsilon,
        exp_curve,
        spectrum_results,
    )
    _write_peak_csv(
        outdir / "peak_summary.csv",
        exp_peaks_nm,
        exp_peaks_ev,
        spectrum_results,
    )
    _plot_spectra(
        outdir / "spectrum_compare.png",
        wavelengths_nm,
        exp_curve,
        spectrum_results,
    )

    spectrum_summary: dict[str, Any] = {}
    for result in spectrum_results:
        band_metrics = {}
        for band in PEAK_WINDOWS_NM:
            band_metrics[band] = {
                "predicted_peak_nm": float(result.peaks_nm[band]),
                "predicted_peak_ev": float(result.peaks_ev[band]),
                "abs_error_nm": float(abs(result.peaks_nm[band] - exp_peaks_nm[band])),
                "abs_error_ev": float(abs(result.peaks_ev[band] - exp_peaks_ev[band])),
            }
        spectrum_summary[result.label] = {
            "scf_energy_au": float(result.scf_energy_au),
            "excitation_energies_ev": [float(x) for x in result.excitation_energies_ev.tolist()],
            "oscillator_strengths": [float(x) for x in result.oscillator_strengths.tolist()],
            "bands": band_metrics,
        }

    overall_summary = {
        "molecule": "aniline",
        "geometry_source": {
            "provider": "PubChem 3D conformer",
            "url": PUBCHEM_SDF_URL,
        },
        "experiment_source": {
            "provider": "NIST Chemistry WebBook",
            "webbook_url": NIST_WEBBOOK_URL,
            "jcamp_url": NIST_JCAMP_URL,
            "reference": (
                "Ramart-Lucas, M.; Hoch, J.; Grumez, M. "
                "Bull. Soc. Chim. Fr. 16, 447-454 (1949)"
            ),
            "x_units": "wavelength_nm",
            "y_units": "log10_epsilon",
            "local_csv": str(Path(args.experiment_csv)),
        },
        "experimental_peaks": {
            band: {
                "peak_nm": float(exp_peaks_nm[band]),
                "peak_ev": float(exp_peaks_ev[band]),
            }
            for band in PEAK_WINDOWS_NM
        },
        "training": training_summary,
        "excited_state": {
            "solver": str(args.solver),
            "nstates": int(args.nstates),
            "eta_ev": float(args.eta_ev),
            "basis": str(args.basis),
            "excited_grid_level": int(args.excited_grid_level),
            "models": spectrum_summary,
        },
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(
        json.dumps(overall_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
