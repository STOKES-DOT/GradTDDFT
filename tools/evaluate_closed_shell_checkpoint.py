from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from td_graddft import neural_xc
from td_graddft.xc_backend.jax_libxc import b3lyp_component_basis
from td_graddft.neural_xc import (
    DEFAULT_INPUT_FEATURE_MODE,
    DEFAULT_NETWORK_ARCHITECTURE,
    DEFAULT_NETWORK_HIDDEN_DIMS,
)
from td_graddft.training import GroundStateTrainingConfig, load_params_checkpoint

from closed_shell_s1_self_consistent_train import (
    _evaluate_dataset,
    _load_reference_rows,
    _prepare_references,
    _normalize_scf_gradient_mode,
)


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"[{_timestamp()}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained closed-shell neural XC checkpoint on selected molecules."
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--reference-csv", required=True)
    p.add_argument("--basis", default="6-31g")
    p.add_argument("--systems", nargs="+", required=True)
    p.add_argument("--xc", default="b3lyp")
    p.add_argument("--training-mode", choices=("fixed_density", "self_consistent"), default="self_consistent")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=list(DEFAULT_NETWORK_HIDDEN_DIMS))
    p.add_argument(
        "--network-architecture",
        choices=("simple_mlp", "graddft_residual"),
        default=DEFAULT_NETWORK_ARCHITECTURE,
    )
    p.add_argument(
        "--input-feature-mode",
        choices=("enhanced", "dm21_original"),
        default=DEFAULT_INPUT_FEATURE_MODE,
    )
    p.add_argument("--include-pt2-channel", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--pt2-channel-mode",
        choices=("scaled_projected", "local_exact"),
        default="scaled_projected",
    )
    p.add_argument("--response-grid-chunk-size", type=int, default=1024)
    p.add_argument(
        "--strict-hfx-response-mode",
        choices=("dense", "low_memory"),
        default="dense",
    )
    p.add_argument("--semilocal-xc", nargs="+", default=list(b3lyp_component_basis()))
    p.add_argument("--eval-use-tda", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--grids-level", type=int, default=0)
    p.add_argument("--reference-scf-max-cycle", type=int, default=100)
    p.add_argument("--reference-scf-conv-tol", type=float, default=1e-10)
    p.add_argument("--train-scf-max-cycle", type=int, default=16)
    p.add_argument("--train-scf-damping", type=float, default=0.25)
    p.add_argument("--train-scf-conv-tol-density", type=float, default=1e-8)
    p.add_argument("--train-scf-vxc-clip", type=float, default=20.0)
    p.add_argument(
        "--scf-iterate-selection",
        choices=("final", "best_rms", "first_converged"),
        default="best_rms",
    )
    p.add_argument(
        "--scf-gradient-mode",
        choices=("unrolled", "implicit_commutator"),
        default="unrolled",
    )
    p.add_argument(
        "--scf-implicit-diff-solver",
        choices=("normal_cg", "gmres", "bicgstab"),
        default="normal_cg",
    )
    p.add_argument("--scf-implicit-diff-tolerance", type=float, default=1e-6)
    p.add_argument("--scf-implicit-diff-regularization", type=float, default=1e-3)
    p.add_argument("--scf-implicit-diff-restart", type=int, default=12)
    p.add_argument("--scf-require-convergence", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--scf-stop-gradient-on-unconverged", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--scf-stop-gradient-rms-threshold", type=float, default=None)
    p.add_argument("--scf-warm-start", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--scf-warm-start-update-interval", type=int, default=1)
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def _apply_checkpoint_metadata(args: argparse.Namespace) -> argparse.Namespace:
    meta_path = Path(str(args.checkpoint) + ".meta.json")
    if not meta_path.exists():
        return args
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    defaults = {
        "training_mode": "self_consistent",
        "include_pt2_channel": False,
        "pt2_channel_mode": "scaled_projected",
        "response_grid_chunk_size": 1024,
        "strict_hfx_response_mode": "dense",
        "scf_gradient_mode": "unrolled",
        "scf_implicit_diff_solver": "normal_cg",
        "scf_implicit_diff_tolerance": 1e-6,
        "scf_implicit_diff_regularization": 1e-3,
        "scf_implicit_diff_restart": 12,
        "scf_stop_gradient_on_unconverged": False,
        "scf_stop_gradient_rms_threshold": None,
        "scf_warm_start": False,
        "scf_warm_start_update_interval": 1,
    }
    for key in (
        "training_mode",
        "include_pt2_channel",
        "pt2_channel_mode",
        "response_grid_chunk_size",
        "strict_hfx_response_mode",
        "scf_gradient_mode",
        "scf_implicit_diff_solver",
        "scf_implicit_diff_tolerance",
        "scf_implicit_diff_regularization",
        "scf_implicit_diff_restart",
        "scf_stop_gradient_on_unconverged",
        "scf_stop_gradient_rms_threshold",
        "scf_warm_start",
        "scf_warm_start_update_interval",
    ):
        if key in meta and getattr(args, key, None) == defaults[key]:
            setattr(args, key, meta[key])
    return args


def main() -> None:
    args = parse_args()
    args = _apply_checkpoint_metadata(args)
    args.scf_gradient_mode = _normalize_scf_gradient_mode(str(args.scf_gradient_mode))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(outdir / "run.log")

    logger.log(
        "Config: "
        f"checkpoint={args.checkpoint}, reference_csv={args.reference_csv}, basis={args.basis}, "
        f"systems={list(args.systems)}, mode={args.training_mode}, "
        f"grid={args.grids_level}, scf_grad_mode={args.scf_gradient_mode}, "
        f"include_pt2_channel={bool(args.include_pt2_channel)}, "
        f"pt2_channel_mode={args.pt2_channel_mode if bool(args.include_pt2_channel) else 'none'}, "
        f"strict_hfx_response_mode={args.strict_hfx_response_mode}"
    )

    rows = _load_reference_rows(Path(args.reference_csv), basis=str(args.basis))
    requested = {str(name) for name in args.systems}
    selected_rows = [row for row in rows if row.system in requested]
    missing = sorted(requested - {row.system for row in selected_rows})
    if missing:
        raise ValueError(f"Missing requested systems in reference CSV: {missing}")

    prepared = _prepare_references(selected_rows, args=args, logger=logger)
    if not prepared:
        raise ValueError("No prepared references selected.")

    functional = neural_xc.Functional(
        architecture=str(args.network_architecture),
        semilocal_xc=tuple(str(name) for name in args.semilocal_xc),
        hidden_dims=tuple(int(value) for value in args.hidden_dims),
        input_feature_mode=str(args.input_feature_mode),
        include_pt2_channel=bool(args.include_pt2_channel),
        pt2_channel_mode=str(args.pt2_channel_mode),
        response_grid_chunk_size=int(args.response_grid_chunk_size),
        strict_hfx_response_mode=str(args.strict_hfx_response_mode),
        name="neural_xc_closed_shell_eval",
    )
    template = functional.init_from_molecule(jax.random.PRNGKey(int(args.seed)), prepared[0].molecule)
    params = load_params_checkpoint(args.checkpoint, template=template)

    training_config = GroundStateTrainingConfig(
        mode=str(args.training_mode),
        energy_mse_weight=0.0,
        energy_mae_weight=0.0,
        s1_constraint_use_tda=bool(args.eval_use_tda),
        scf_max_cycle=int(args.train_scf_max_cycle),
        scf_damping=float(args.train_scf_damping),
        scf_conv_tol_density=float(args.train_scf_conv_tol_density),
        scf_vxc_clip=float(args.train_scf_vxc_clip),
        scf_iterate_selection=str(args.scf_iterate_selection),
        scf_require_convergence=bool(args.scf_require_convergence),
        scf_stop_gradient_on_unconverged=bool(args.scf_stop_gradient_on_unconverged),
        scf_stop_gradient_rms_threshold=args.scf_stop_gradient_rms_threshold,
        scf_gradient_mode=str(args.scf_gradient_mode),
        scf_implicit_diff_solver=str(args.scf_implicit_diff_solver),
        scf_implicit_diff_tolerance=float(args.scf_implicit_diff_tolerance),
        scf_implicit_diff_regularization=float(args.scf_implicit_diff_regularization),
        scf_implicit_diff_restart=int(args.scf_implicit_diff_restart),
    )

    pred_rows, metrics = _evaluate_dataset(
        prepared,
        params=params,
        functional=functional,
        training_config=training_config,
        use_tda=bool(args.eval_use_tda),
    )

    predictions_csv = outdir / "predictions.csv"
    with predictions_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pred_rows[0].keys()))
        writer.writeheader()
        for row in pred_rows:
            writer.writerow(row)

    summary = {
        "checkpoint": str(args.checkpoint),
        "reference_csv": str(args.reference_csv),
        "basis": str(args.basis),
        "xc": str(args.xc),
        "systems": [row["system"] for row in pred_rows],
        "evaluation_solver": "tda" if bool(args.eval_use_tda) else "casida",
        "training_mode": str(args.training_mode),
        "include_pt2_channel": bool(args.include_pt2_channel),
        "pt2_channel_mode": str(args.pt2_channel_mode) if bool(args.include_pt2_channel) else None,
        "s1_mae_ev": float(metrics["s1_mae_ev"]),
        "s1_max_ev": float(metrics["s1_max_ev"]),
        "total_mae_ev": float(metrics["total_mae_ev"]),
        "predictions_csv": str(predictions_csv),
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    logger.log(f"Wrote predictions: {predictions_csv}")
    logger.log(f"Wrote summary   : {summary_path}")


if __name__ == "__main__":
    main()
