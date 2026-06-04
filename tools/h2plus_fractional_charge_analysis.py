from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax

from td_graddft import neural_xc
from td_graddft.neural_xc import (
    DEFAULT_INPUT_FEATURE_MODE,
    DEFAULT_NETWORK_ARCHITECTURE,
    DEFAULT_NETWORK_HIDDEN_DIMS,
)
from td_graddft.spectra import HARTREE_TO_EV
from td_graddft.training import (
    GroundStateCoreTrainingConfig,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    load_params_checkpoint,
    make_ground_state_predictor,
)
from td_graddft.training.targets import (
    _as_spin_resolved_molecule,
    _freeze_functional_for_fractional_path,
    _predict_ground_state_total_energy_from_molecule,
    _perturb_spin_orbital_occupation,
    _spin_orbital_frontier_indices,
    _spin_resolved_orbital_blocks,
)
from tools.h2plus_fci_ground_train5_dense100 import (
    _DEFAULT_SEMILOCAL_XC,
    RunLogger,
    get_or_build_reference_point,
    parse_args as parse_h2plus_args,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-functional H2+ fractional-charge piecewise-linearity diagnostics."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="H2+ Neural_xc msgpack checkpoint.",
    )
    parser.add_argument(
        "--reference-cache",
        required=True,
        help="H2+ HDF5 reference cache used by the ground-state run.",
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--basis", default="def2-svp")
    parser.add_argument("--xc", default="b3lyp")
    parser.add_argument("--grids-level", type=int, default=2)
    parser.add_argument("--max-l", type=int, default=3)
    parser.add_argument("--integral-backend", choices=("jax", "cpu", "gpu", "libcint"), default="gpu")
    parser.add_argument("--r-values", type=float, nargs="+", default=[0.4, 1.8, 2.3232323232, 3.2, 4.6, 6.0])
    parser.add_argument("--charge-min", type=float, default=-1.0)
    parser.add_argument("--charge-max", type=float, default=1.0)
    parser.add_argument("--num-points", type=int, default=41)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=list(DEFAULT_NETWORK_HIDDEN_DIMS))
    parser.add_argument("--network-architecture", choices=("simple_mlp", "graddft_residual"), default=DEFAULT_NETWORK_ARCHITECTURE)
    parser.add_argument("--input-feature-mode", choices=("enhanced", "canonical", "dm21_original"), default=DEFAULT_INPUT_FEATURE_MODE)
    parser.add_argument("--semilocal-xc", nargs="+", default=list(_DEFAULT_SEMILOCAL_XC))
    parser.add_argument("--energy-mse-weight", type=float, default=1.0)
    parser.add_argument("--energy-mae-weight", type=float, default=1.0)
    parser.add_argument("--energy-normalization", choices=("none", "per_electron", "per_atom"), default="none")
    parser.add_argument("--coefficient-prior-weight", type=float, default=0.0)
    parser.add_argument("--train-scf-max-cycle", type=int, default=0)
    parser.add_argument("--train-scf-damping", type=float, default=0.25)
    parser.add_argument("--train-scf-conv-tol-energy", type=float, default=1e-6)
    parser.add_argument("--train-scf-convergence-metric", choices=("energy_and_residual", "energy"), default="energy")
    parser.add_argument("--train-scf-conv-tol-density", type=float, default=1e-8)
    parser.add_argument("--train-scf-vxc-clip", type=float, default=20.0)
    parser.add_argument("--scf-iterate-selection", choices=("final", "best_rms", "first_converged"), default="best_rms")
    parser.add_argument("--scf-gradient-mode", choices=("expl", "impl"), default="impl")
    parser.add_argument("--scf-implicit-diff-tolerance", type=float, default=1e-6)
    parser.add_argument("--scf-implicit-diff-regularization", type=float, default=1e-3)
    return parser.parse_args(argv)


def _make_h2plus_reference_args(args: argparse.Namespace) -> argparse.Namespace:
    h2plus_args = parse_h2plus_args([])
    h2plus_args.basis = str(args.basis)
    h2plus_args.xc = str(args.xc)
    h2plus_args.grids_level = int(args.grids_level)
    h2plus_args.max_l = int(args.max_l)
    h2plus_args.integral_backend = str(args.integral_backend)
    h2plus_args.reference_cache = str(args.reference_cache)
    h2plus_args.rebuild_reference_cache = False
    h2plus_args.input_feature_mode = str(args.input_feature_mode)
    return h2plus_args


def _make_functional(args: argparse.Namespace) -> Any:
    return neural_xc.Functional(
        semilocal_xc=tuple(str(name) for name in args.semilocal_xc),
        hidden_dims=tuple(int(value) for value in args.hidden_dims),
        architecture=str(args.network_architecture),
        input_feature_mode=str(args.input_feature_mode),
        include_pt2_channel=False,
        name="neural_xc_h2plus_fci_ground",
    )


def _make_training_config(args: argparse.Namespace) -> GroundStateTrainingConfig:
    coefficient_prior = neural_xc.resolve_coefficient_prior_values(
        tuple(str(name) for name in args.semilocal_xc)
    )
    return GroundStateTrainingConfig.from_parts(
        core=GroundStateCoreTrainingConfig(
            mode="self_consistent",
            energy_mse_weight=float(args.energy_mse_weight),
            energy_mae_weight=float(args.energy_mae_weight),
            energy_normalization=str(args.energy_normalization),
            coefficient_prior_weight=float(args.coefficient_prior_weight),
            coefficient_prior_values=coefficient_prior,
            scf_max_cycle=32 if int(args.train_scf_max_cycle) <= 0 else int(args.train_scf_max_cycle),
            scf_damping=float(args.train_scf_damping),
            scf_conv_tol_energy=float(args.train_scf_conv_tol_energy),
            scf_convergence_metric=str(args.train_scf_convergence_metric),
            scf_conv_tol_density=float(args.train_scf_conv_tol_density),
            scf_vxc_clip=float(args.train_scf_vxc_clip),
            scf_iterate_selection=str(args.scf_iterate_selection),
            scf_gradient_mode=str(args.scf_gradient_mode),
            scf_implicit_diff_tolerance=float(args.scf_implicit_diff_tolerance),
            scf_implicit_diff_regularization=float(args.scf_implicit_diff_regularization),
        ),
    )


def _charge_grid(args: argparse.Namespace) -> np.ndarray:
    if int(args.num_points) < 3:
        raise ValueError("--num-points must be at least 3.")
    charges = np.linspace(float(args.charge_min), float(args.charge_max), int(args.num_points))
    if not np.any(np.isclose(charges, 0.0, atol=1e-12)):
        charges = np.sort(np.concatenate([charges, np.asarray([0.0])]))
    if charges[0] > -1e-12 or charges[-1] < 1e-12:
        raise ValueError("charge grid must span 0.")
    return charges


def _scan_one_geometry(
    *,
    r_angstrom: float,
    reference_molecule: Any,
    params: Any,
    functional: Any,
    training_config: GroundStateTrainingConfig,
    charge_deltas: np.ndarray,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int]]:
    predictor = make_ground_state_predictor(functional, training_config=training_config)
    neutral_energy_arr, neutral_molecule_raw = predictor(params, reference_molecule)
    neutral_molecule = _as_spin_resolved_molecule(neutral_molecule_raw)
    frozen_functional, frozen_params = _freeze_functional_for_fractional_path(
        params,
        functional,
        neutral_molecule,
    )
    freeze_mode = "frozen_descriptor"
    try:
        _ = _predict_ground_state_total_energy_from_molecule(
            frozen_params,
            frozen_functional,
            neutral_molecule,
        )
    except AttributeError:
        frozen_functional = functional
        frozen_params = params
        freeze_mode = "unfrozen_descriptor_fallback"
    homo_spin, homo_idx, lumo_spin, lumo_idx = _spin_orbital_frontier_indices(neutral_molecule)
    _, base_occ, base_mo_energy = _spin_resolved_orbital_blocks(neutral_molecule)

    rows: list[dict[str, float | int | str]] = []
    for delta in charge_deltas:
        if float(delta) < 0.0:
            branch = "remove_homo"
            spin_idx = int(homo_spin)
            orb_idx = int(homo_idx)
        elif float(delta) > 0.0:
            branch = "add_lumo"
            spin_idx = int(lumo_spin)
            orb_idx = int(lumo_idx)
        else:
            branch = "neutral"
            spin_idx = int(homo_spin)
            orb_idx = int(homo_idx)
        fractional_molecule = (
            neutral_molecule
            if float(delta) == 0.0
            else _perturb_spin_orbital_occupation(
                neutral_molecule,
                spin_index=spin_idx,
                orbital_index=orb_idx,
                delta=float(delta),
            )
        )
        _, occ, _ = _spin_resolved_orbital_blocks(fractional_molecule)
        energy = float(
            _predict_ground_state_total_energy_from_molecule(
                frozen_params,
                frozen_functional,
                fractional_molecule,
            )
        )
        rows.append(
            {
                "r_angstrom": float(r_angstrom),
                "charge_delta": float(delta),
                "electron_count": float(jnp.sum(occ)),
                "energy_ha": energy,
                "energy_ev": energy * HARTREE_TO_EV,
                "alpha_homo_occ": float(occ[int(homo_spin), int(homo_idx)]),
                "beta_lumo_occ": float(occ[int(lumo_spin), int(lumo_idx)]),
                "branch": branch,
                "freeze_mode": freeze_mode,
                "homo_spin": int(homo_spin),
                "homo_index": int(homo_idx),
                "lumo_spin": int(lumo_spin),
                "lumo_index": int(lumo_idx),
                "homo_energy_ha": float(base_mo_energy[int(homo_spin), int(homo_idx)]),
                "lumo_energy_ha": float(base_mo_energy[int(lumo_spin), int(lumo_idx)]),
            }
        )

    zero_idx = int(np.argmin(np.abs(charge_deltas)))
    left_idx = 0
    right_idx = len(charge_deltas) - 1
    q0 = float(charge_deltas[zero_idx])
    q_left = float(charge_deltas[left_idx])
    q_right = float(charge_deltas[right_idx])
    e0 = float(rows[zero_idx]["energy_ha"])
    e_left = float(rows[left_idx]["energy_ha"])
    e_right = float(rows[right_idx]["energy_ha"])
    left_slope = (e0 - e_left) / (q0 - q_left)
    right_slope = (e_right - e0) / (q_right - q0)
    for row in rows:
        q = float(row["charge_delta"])
        if q <= q0:
            piecewise = e0 + (q - q0) * (e_left - e0) / (q_left - q0)
        else:
            piecewise = e0 + (q - q0) * (e_right - e0) / (q_right - q0)
        deviation = float(row["energy_ha"]) - float(piecewise)
        row["piecewise_linear_energy_ha"] = float(piecewise)
        row["piecewise_linear_energy_ev"] = float(piecewise) * HARTREE_TO_EV
        row["deviation_ha"] = deviation
        row["deviation_ev"] = deviation * HARTREE_TO_EV
        row["abs_deviation_ev"] = abs(deviation) * HARTREE_TO_EV

    deviations = np.asarray([float(row["deviation_ha"]) for row in rows], dtype=float)
    summary = {
        "r_angstrom": float(r_angstrom),
        "neutral_energy_ha": float(neutral_energy_arr),
        "fixed_fractional_neutral_energy_ha": e0,
        "max_abs_deviation_ha": float(np.max(np.abs(deviations))),
        "max_abs_deviation_ev": float(np.max(np.abs(deviations)) * HARTREE_TO_EV),
        "mean_abs_deviation_ha": float(np.mean(np.abs(deviations))),
        "mean_abs_deviation_ev": float(np.mean(np.abs(deviations)) * HARTREE_TO_EV),
        "rms_deviation_ha": float(np.sqrt(np.mean(deviations * deviations))),
        "rms_deviation_ev": float(np.sqrt(np.mean(deviations * deviations)) * HARTREE_TO_EV),
        "left_endpoint_slope_ha": float(left_slope),
        "right_endpoint_slope_ha": float(right_slope),
        "homo_spin": int(homo_spin),
        "homo_index": int(homo_idx),
        "lumo_spin": int(lumo_spin),
        "lumo_index": int(lumo_idx),
        "freeze_mode": freeze_mode,
        "base_alpha_homo_occ": float(base_occ[int(homo_spin), int(homo_idx)]),
        "base_beta_lumo_occ": float(base_occ[int(lumo_spin), int(lumo_idx)]),
        "homo_energy_ha": float(base_mo_energy[int(homo_spin), int(homo_idx)]),
        "lumo_energy_ha": float(base_mo_energy[int(lumo_spin), int(lumo_idx)]),
    }
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_outputs(outdir: Path, curve_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for r_value in sorted({float(row["r_angstrom"]) for row in curve_rows}):
        rows = [row for row in curve_rows if abs(float(row["r_angstrom"]) - r_value) < 1e-10]
        x = np.asarray([float(row["charge_delta"]) for row in rows], dtype=float)
        y = np.asarray([float(row["deviation_ev"]) for row in rows], dtype=float)
        axes[0].plot(x, y, lw=2.0, label=f"R={r_value:.3g} A")
    axes[0].axhline(0.0, color="black", lw=0.9, ls="--", alpha=0.7)
    axes[0].set_xlabel("Fractional electron delta")
    axes[0].set_ylabel("Deviation from piecewise line (eV)")
    axes[0].set_title("H2+ Fractional-Charge Curvature")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    r = np.asarray([float(row["r_angstrom"]) for row in summary_rows], dtype=float)
    max_dev = np.asarray([float(row["max_abs_deviation_ev"]) for row in summary_rows], dtype=float)
    rms_dev = np.asarray([float(row["rms_deviation_ev"]) for row in summary_rows], dtype=float)
    axes[1].plot(r, max_dev, marker="o", lw=2.0, label="max |dev|")
    axes[1].plot(r, rms_dev, marker="s", lw=2.0, label="RMS dev")
    axes[1].set_xlabel("R (Angstrom)")
    axes[1].set_ylabel("Deviation (eV)")
    axes[1].set_title("Curvature Summary")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    path = outdir / "h2plus_fractional_charge_deviation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(outdir / "fractional_charge.log")
    reference_args = _make_h2plus_reference_args(args)
    charge_deltas = _charge_grid(args)

    first_point = get_or_build_reference_point(float(args.r_values[0]), args=reference_args, logger=logger)
    functional = _make_functional(args)
    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(int(args.seed)),
        first_point.molecule,
        optax.adam(1e-5),
    )
    params = load_params_checkpoint(Path(args.checkpoint), template=state.params)
    training_config = _make_training_config(args)

    all_curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for idx, r_value in enumerate(args.r_values, start=1):
        point = first_point if idx == 1 else get_or_build_reference_point(float(r_value), args=reference_args, logger=logger)
        logger.log(f"[fractional] {idx}/{len(args.r_values)} R={float(r_value):.6f}")
        curve_rows, summary = _scan_one_geometry(
            r_angstrom=float(r_value),
            reference_molecule=point.molecule,
            params=params,
            functional=functional,
            training_config=training_config,
            charge_deltas=charge_deltas,
        )
        all_curve_rows.extend(curve_rows)
        summary_rows.append(summary)
        logger.log(
            "[fractional] "
            f"R={float(r_value):.6f} max_abs_dev={summary['max_abs_deviation_ev']:.6e} eV "
            f"rms_dev={summary['rms_deviation_ev']:.6e} eV"
        )

    curves_csv = outdir / "h2plus_fractional_charge_curves.csv"
    summary_csv = outdir / "h2plus_fractional_charge_summary.csv"
    _write_csv(curves_csv, all_curve_rows)
    _write_csv(summary_csv, summary_rows)
    png_path = _plot_outputs(outdir, all_curve_rows, summary_rows)
    summary_json = {
        "system": "H2+",
        "checkpoint": str(args.checkpoint),
        "reference_cache": str(args.reference_cache),
        "r_values": [float(value) for value in args.r_values],
        "charge_min": float(args.charge_min),
        "charge_max": float(args.charge_max),
        "num_points": int(args.num_points),
        "analysis_boundary": (
            "Spin-resolved HOMO-removal / beta-LUMO-addition piecewise-linearity diagnostic "
            "from each neutral neural self-consistent H2+ state. The tool attempts frozen "
            "descriptor evaluation first; if the current bound functional does not expose a "
            "molecule-energy method, rows are marked freeze_mode=unfrozen_descriptor_fallback."
        ),
        "curves_csv": str(curves_csv),
        "summary_csv": str(summary_csv),
        "figure_png": str(png_path),
        "summary_rows": summary_rows,
    }
    summary_path = outdir / "h2plus_fractional_charge_summary.json"
    summary_path.write_text(json.dumps(summary_json, indent=2, sort_keys=True), encoding="utf-8")
    logger.log(f"[summary] wrote {curves_csv} {summary_csv} {png_path} {summary_path}")
    return summary_json


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
