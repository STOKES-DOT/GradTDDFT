from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze, unfreeze
from pyscf import dft, gto

from td_graddft import neural_xc
from td_graddft.reference_legacy import restricted_reference_from_pyscf
from td_graddft.spectra import HARTREE_TO_EV
from td_graddft.training import (
    ExcitedStateFineTuneConfig,
    ExcitedStateFineTuner,
    GroundStateTrainingConfig,
    load_params_checkpoint,
    predict_excitation_energies,
    predict_ground_state_total_energy,
    predict_oscillator_strengths,
    save_params_checkpoint,
)
from td_graddft_tools import (
    GroundStateTargetBundle,
    InputInfo,
    input_info_to_geometry_string,
    load_ground_state_target_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a long-range TD-GradDFT response correction from serialized target bundles "
            "and a stage-1 neural XC checkpoint."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="stage-1 base functional checkpoint")
    parser.add_argument("--target-bundles", nargs="+", required=True, help="one or more .npz bundles")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--xc-ref", default="", help="PySCF orbital XC used to rebuild references")
    parser.add_argument(
        "--semilocal-xc",
        nargs="+",
        default=None,
        help="override semilocal basis used by the stage-1 neural XC",
    )
    parser.add_argument("--base-hidden-dims", type=int, nargs="+", default=None)
    parser.add_argument(
        "--network-architecture",
        choices=("simple_mlp", "graddft_residual"),
        default=None,
    )
    parser.add_argument(
        "--input-feature-mode",
        choices=("enhanced", "dm21_original"),
        default=None,
    )
    parser.add_argument(
        "--hf-input-mode",
        choices=("total_only", "spin_resolved"),
        default=None,
    )
    parser.add_argument("--include-pt2-channel", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--pt2-channel-mode",
        choices=("scaled_projected", "local_exact"),
        default=None,
    )
    parser.add_argument(
        "--states",
        type=int,
        default=0,
        help="number of states to fine-tune; 0 infers the largest common state count from the bundles",
    )
    parser.add_argument("--use-tda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference-grids-level", type=int, default=0)
    parser.add_argument("--reference-scf-max-cycle", type=int, default=120)
    parser.add_argument("--reference-scf-conv-tol", type=float, default=1e-10)
    parser.add_argument("--lr-steps", type=int, default=200)
    parser.add_argument("--lr-learning-rate", type=float, default=5e-2)
    parser.add_argument("--lr-hidden-dims", type=int, nargs="+", default=[64, 64, 32])
    parser.add_argument("--lr-alpha-scale", type=float, default=0.2)
    parser.add_argument("--weight-energy", type=float, default=1.0)
    parser.add_argument(
        "--energy-loss",
        choices=("mse", "mae"),
        default="mse",
        help="loss used for excited-state energy supervision",
    )
    parser.add_argument("--weight-oscillator-strength", type=float, default=0.0)
    parser.add_argument("--weight-ground-state-energy", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=20)
    return parser.parse_args()


def _load_checkpoint_metadata(checkpoint_path: str | Path) -> dict[str, Any]:
    ckpt_path = Path(checkpoint_path)
    meta_path = ckpt_path.with_suffix(ckpt_path.suffix + ".meta.json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _normalize_semilocal_xc(value: Any) -> str | tuple[str, ...]:
    if value is None:
        return "b3lyp"
    if isinstance(value, str):
        return str(value)
    values = tuple(str(item) for item in value)
    if len(values) == 1:
        return values[0]
    return values


def _resolve_base_functional_config(
    args: argparse.Namespace,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    semilocal_xc = (
        _normalize_semilocal_xc(args.semilocal_xc)
        if args.semilocal_xc is not None
        else _normalize_semilocal_xc(metadata.get("semilocal_xc", "b3lyp"))
    )
    hidden_dims_raw = (
        args.base_hidden_dims
        if args.base_hidden_dims is not None
        else metadata.get("base_hidden_dims", metadata.get("hidden_dims", (64, 64)))
    )
    hidden_dims = tuple(int(value) for value in hidden_dims_raw)
    if not hidden_dims:
        raise ValueError("base hidden dimensions must contain at least one width.")

    return {
        "xc_ref": str(args.xc_ref or metadata.get("xc_ref") or metadata.get("xc") or "b3lyp"),
        "semilocal_xc": semilocal_xc,
        "hidden_dims": hidden_dims,
        "network_architecture": str(
            args.network_architecture
            or metadata.get("network_architecture")
            or "graddft_residual"
        ),
        "input_feature_mode": str(
            args.input_feature_mode
            or metadata.get("input_feature_mode")
            or "dm21_original"
        ),
        "hf_input_mode": str(
            args.hf_input_mode
            or metadata.get("hf_input_mode")
            or "spin_resolved"
        ),
        "include_pt2_channel": bool(
            metadata.get("include_pt2_channel", False)
            if args.include_pt2_channel is None
            else args.include_pt2_channel
        ),
        "pt2_channel_mode": str(
            args.pt2_channel_mode
            or metadata.get("pt2_channel_mode")
            or "scaled_projected"
        ),
    }


def _build_reference_from_input_info(
    input_info: InputInfo,
    *,
    basis_override: str | None,
    xc_ref: str,
    grids_level: int,
    scf_max_cycle: int,
    scf_conv_tol: float,
    compute_local_pt2_features: bool,
):
    basis_name = str(basis_override or input_info.basis_name or "")
    if not basis_name:
        raise ValueError(
            f"Missing basis information for bundle system {input_info.system_label!r}."
        )

    mol = gto.Mole()
    mol.atom = input_info_to_geometry_string(input_info)
    mol.unit = "Angstrom"
    mol.basis = basis_name
    mol.charge = int(input_info.charge or 0)
    mol.spin = int(input_info.spin or 0)
    mol.verbose = 0
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = str(xc_ref)
    mf.grids.level = int(grids_level)
    mf.conv_tol = float(scf_conv_tol)
    mf.max_cycle = int(scf_max_cycle)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(
            f"RKS did not converge for {input_info.system_label!r} with {xc_ref}/{basis_name}."
        )

    reference = restricted_reference_from_pyscf(
        mf,
        compute_local_hfx_features=True,
        compute_local_pt2_features=bool(compute_local_pt2_features),
        hfx_omega_values=(0.0, 0.4),
        hfx_chunk_size=256,
    )
    return reference


def _infer_state_count(bundles: list[GroundStateTargetBundle], requested: int) -> int:
    if int(requested) > 0:
        return int(requested)

    lengths: list[int] = []
    for bundle in bundles:
        if bundle.target_excitation_energies is not None:
            lengths.append(int(np.asarray(bundle.target_excitation_energies).shape[0]))
        if bundle.target_oscillator_strengths is not None:
            lengths.append(int(np.asarray(bundle.target_oscillator_strengths).shape[0]))
    if not lengths:
        raise ValueError("Could not infer excited-state count from the provided bundles.")
    inferred = min(lengths)
    if inferred <= 0:
        raise ValueError("Inferred excited-state count is zero.")
    return inferred


def _initialize_near_zero_lr_params(
    functional,
    molecule,
    *,
    seed: int,
    alpha_value: float = 5e-3,
    gamma_value: float = 1.0,
    random_scale: float = 1e-2,
):
    params = unfreeze(functional.init(jax.random.PRNGKey(int(seed)), molecule))
    params = jax.tree_util.tree_map(
        lambda value: jnp.asarray(value) * jnp.asarray(random_scale, dtype=jnp.asarray(value).dtype),
        params,
    )

    alpha_scale = float(getattr(functional.model, "alpha_scale", 1.0))
    alpha_target = max(float(alpha_value) / max(alpha_scale, 1e-8), 1e-6)
    gamma_floor = float(getattr(functional.model, "gamma_floor", 1e-3))
    gamma_target = max(float(gamma_value) - gamma_floor, 1e-6)
    params["params"]["AlphaHead"]["bias"] = jnp.full_like(
        params["params"]["AlphaHead"]["bias"],
        jnp.log(jnp.expm1(jnp.asarray(alpha_target))),
    )
    params["params"]["GammaHead"]["bias"] = jnp.full_like(
        params["params"]["GammaHead"]["bias"],
        jnp.log(jnp.expm1(jnp.asarray(gamma_target))),
    )
    return freeze(params)


def _evaluate_dataset(
    *,
    entries: list[tuple[GroundStateTargetBundle, Any]],
    params,
    functional,
    nstates: int,
    use_tda: bool,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    excitation_errors_ev: list[float] = []
    oscillator_errors: list[float] = []
    ground_errors_ev: list[float] = []

    for bundle, datum in entries:
        system = str(bundle.input_info.system_label)
        predicted_ground = float(
            predict_ground_state_total_energy(
                params,
                functional,
                datum.molecule,
                training_config=GroundStateTrainingConfig(mode="fixed_density"),
            )
        )
        predicted_energies = np.asarray(
            predict_excitation_energies(
                params,
                functional,
                datum.molecule,
                nstates=int(nstates),
                use_tda=bool(use_tda),
            ),
            dtype=float,
        )
        predicted_oscillator_strengths = np.asarray(
            predict_oscillator_strengths(
                params,
                functional,
                datum.molecule,
                nstates=int(nstates),
                use_tda=bool(use_tda),
            ),
            dtype=float,
        )
        reference_energies = (
            np.asarray(bundle.target_excitation_energies, dtype=float)
            if bundle.target_excitation_energies is not None
            else np.zeros((0,), dtype=float)
        )
        reference_oscillator_strengths = (
            np.asarray(bundle.target_oscillator_strengths, dtype=float)
            if bundle.target_oscillator_strengths is not None
            else np.zeros((0,), dtype=float)
        )

        ground_errors_ev.append(
            abs(predicted_ground - float(np.asarray(bundle.target_total_energy))) * HARTREE_TO_EV
        )
        energy_compare = min(int(reference_energies.size), int(predicted_energies.size))
        osc_compare = min(
            int(reference_oscillator_strengths.size),
            int(predicted_oscillator_strengths.size),
        )
        if energy_compare > 0:
            excitation_errors_ev.extend(
                np.abs(predicted_energies[:energy_compare] - reference_energies[:energy_compare])
                * HARTREE_TO_EV
            )
        if osc_compare > 0:
            oscillator_errors.extend(
                np.abs(
                    predicted_oscillator_strengths[:osc_compare]
                    - reference_oscillator_strengths[:osc_compare]
                )
            )

        nrows = max(
            int(predicted_energies.size),
            int(reference_energies.size),
            int(predicted_oscillator_strengths.size),
            int(reference_oscillator_strengths.size),
        )
        for state_idx in range(nrows):
            rows.append(
                {
                    "system": system,
                    "state": int(state_idx + 1),
                    "reference_ground_total_energy_hartree": float(
                        np.asarray(bundle.target_total_energy)
                    ),
                    "predicted_ground_total_energy_hartree": float(predicted_ground),
                    "reference_excitation_energy_eV": (
                        float(reference_energies[state_idx] * HARTREE_TO_EV)
                        if state_idx < int(reference_energies.size)
                        else None
                    ),
                    "predicted_excitation_energy_eV": (
                        float(predicted_energies[state_idx] * HARTREE_TO_EV)
                        if state_idx < int(predicted_energies.size)
                        else None
                    ),
                    "reference_oscillator_strength": (
                        float(reference_oscillator_strengths[state_idx])
                        if state_idx < int(reference_oscillator_strengths.size)
                        else None
                    ),
                    "predicted_oscillator_strength": (
                        float(predicted_oscillator_strengths[state_idx])
                        if state_idx < int(predicted_oscillator_strengths.size)
                        else None
                    ),
                }
            )

    metrics = {
        "ground_mae_ev": float(np.mean(ground_errors_ev)) if ground_errors_ev else float("nan"),
        "excitation_mae_ev": (
            float(np.mean(excitation_errors_ev)) if excitation_errors_ev else float("nan")
        ),
        "oscillator_strength_mae": (
            float(np.mean(oscillator_errors)) if oscillator_errors else float("nan")
        ),
    }
    return rows, metrics


def _write_prediction_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bundles = [
        load_ground_state_target_bundle(path)
        for path in [Path(value) for value in args.target_bundles]
    ]
    if not bundles:
        raise ValueError("At least one target bundle is required.")

    if float(args.weight_energy) > 0.0 and any(
        bundle.target_excitation_energies is None for bundle in bundles
    ):
        raise ValueError("All bundles must provide target_excitation_energies when weight_energy > 0.")
    if float(args.weight_oscillator_strength) > 0.0 and any(
        bundle.target_oscillator_strengths is None for bundle in bundles
    ):
        raise ValueError(
            "All bundles must provide target_oscillator_strengths when weight_oscillator_strength > 0."
        )

    checkpoint_metadata = _load_checkpoint_metadata(args.checkpoint)
    base_config = _resolve_base_functional_config(args, checkpoint_metadata)
    nstates = _infer_state_count(bundles, int(args.states))

    dataset_entries: list[tuple[GroundStateTargetBundle, Any]] = []
    for bundle in bundles:
        reference = _build_reference_from_input_info(
            bundle.input_info,
            basis_override=None,
            xc_ref=base_config["xc_ref"],
            grids_level=int(args.reference_grids_level),
            scf_max_cycle=int(args.reference_scf_max_cycle),
            scf_conv_tol=float(args.reference_scf_conv_tol),
            compute_local_pt2_features=bool(base_config["include_pt2_channel"]),
        )
        datum = bundle.to_datum(reference)
        dataset_entries.append((bundle, datum))

    base_functional = neural_xc.Functional(
        semilocal_xc=base_config["semilocal_xc"],
        hidden_dims=base_config["hidden_dims"],
        architecture=base_config["network_architecture"],
        input_feature_mode=base_config["input_feature_mode"],
        hf_input_mode=base_config["hf_input_mode"],
        include_pt2_channel=base_config["include_pt2_channel"],
        pt2_channel_mode=base_config["pt2_channel_mode"],
        name="long_range_stage1_base",
    )
    template_params = base_functional.init_from_molecule(
        jax.random.PRNGKey(int(args.seed)),
        dataset_entries[0][1].molecule,
    )
    base_params = load_params_checkpoint(args.checkpoint, template=template_params)

    base_rows, base_metrics = _evaluate_dataset(
        entries=dataset_entries,
        params=base_params,
        functional=base_functional,
        nstates=nstates,
        use_tda=bool(args.use_tda),
    )
    for row in base_rows:
        row["stage"] = "base"

    lr_functional = neural_xc.LongRangeCorrection(
        base_functional=base_functional,
        hidden_dims=tuple(int(value) for value in args.lr_hidden_dims),
        alpha_scale=float(args.lr_alpha_scale),
        name="long_range_stage2_wrapper",
    )
    lr_params = _initialize_near_zero_lr_params(
        lr_functional,
        dataset_entries[0][1].molecule,
        seed=int(args.seed) + 1,
    )
    combined_params = lr_functional.combine_params(base_params, lr_params)

    fine_tune_dataset = [datum for _, datum in dataset_entries]
    config = ExcitedStateFineTuneConfig(
        steps=int(args.lr_steps),
        learning_rate=float(args.lr_learning_rate),
        excited_states=tuple(range(1, int(nstates) + 1)),
        use_tda=bool(args.use_tda),
        weight_energy=float(args.weight_energy),
        energy_loss=str(args.energy_loss),
        weight_oscillator_strength=float(args.weight_oscillator_strength),
        weight_ground_state_energy=float(args.weight_ground_state_energy),
        freeze_ground_state_params=True,
        trainable_path_prefixes=("lr_correction",),
        log_interval=int(args.log_interval),
    )

    start_time = time.perf_counter()
    result = ExcitedStateFineTuner(config, lr_functional, combined_params).fine_tune(
        fine_tune_dataset
    )
    wall_time_s = float(time.perf_counter() - start_time)

    corrected_rows, corrected_metrics = _evaluate_dataset(
        entries=dataset_entries,
        params=result.params,
        functional=lr_functional,
        nstates=nstates,
        use_tda=bool(args.use_tda),
    )
    for row in corrected_rows:
        row["stage"] = "long_range_corrected"

    predictions_csv = outdir / "predictions.csv"
    _write_prediction_csv(predictions_csv, base_rows + corrected_rows)

    stage2_checkpoint, stage2_meta = save_params_checkpoint(
        outdir / "long_range_corrected_params.msgpack",
        result.params,
        metadata={
            "source_checkpoint": str(args.checkpoint),
            "target_bundles": [str(Path(path)) for path in args.target_bundles],
            "xc_ref": base_config["xc_ref"],
            "semilocal_xc": base_config["semilocal_xc"],
            "base_hidden_dims": list(base_config["hidden_dims"]),
            "network_architecture": base_config["network_architecture"],
            "input_feature_mode": base_config["input_feature_mode"],
            "hf_input_mode": base_config["hf_input_mode"],
            "include_pt2_channel": bool(base_config["include_pt2_channel"]),
            "pt2_channel_mode": base_config["pt2_channel_mode"],
            "states": int(nstates),
            "use_tda": bool(args.use_tda),
            "lr_hidden_dims": [int(value) for value in args.lr_hidden_dims],
            "lr_alpha_scale": float(args.lr_alpha_scale),
            "lr_steps": int(args.lr_steps),
            "lr_learning_rate": float(args.lr_learning_rate),
            "weight_energy": float(args.weight_energy),
            "weight_oscillator_strength": float(args.weight_oscillator_strength),
            "weight_ground_state_energy": float(args.weight_ground_state_energy),
        },
    )

    summary = {
        "source_checkpoint": str(args.checkpoint),
        "target_bundles": [str(Path(path)) for path in args.target_bundles],
        "systems": [bundle.input_info.system_label for bundle, _ in dataset_entries],
        "xc_ref": base_config["xc_ref"],
        "semilocal_xc": base_config["semilocal_xc"],
        "base_hidden_dims": list(base_config["hidden_dims"]),
        "network_architecture": base_config["network_architecture"],
        "input_feature_mode": base_config["input_feature_mode"],
        "hf_input_mode": base_config["hf_input_mode"],
        "include_pt2_channel": bool(base_config["include_pt2_channel"]),
        "pt2_channel_mode": base_config["pt2_channel_mode"],
        "states": int(nstates),
        "solver": "tda" if bool(args.use_tda) else "casida",
        "stage1_metrics": base_metrics,
        "stage2_metrics": corrected_metrics,
        "fine_tune": {
            "initial_loss": float(result.initial_loss),
            "final_loss": float(result.final_loss),
            "best_loss": float(result.best_loss),
            "best_step": int(result.best_step),
            "steps": int(args.lr_steps),
            "learning_rate": float(args.lr_learning_rate),
            "weight_energy": float(args.weight_energy),
            "weight_oscillator_strength": float(args.weight_oscillator_strength),
            "weight_ground_state_energy": float(args.weight_ground_state_energy),
        },
        "predictions_csv": str(predictions_csv),
        "stage2_checkpoint": str(stage2_checkpoint),
        "stage2_checkpoint_meta": str(stage2_meta) if stage2_meta is not None else None,
        "wall_time_s": wall_time_s,
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
