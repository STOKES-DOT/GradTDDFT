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
    DEFAULT_NEURAL_XC_RESPONSE_HF_MODE,
)
from td_graddft.training import (
    MolecularTrainingConfig,
    load_params_checkpoint,
    predict_ground_state_total_energy,
)

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
        choices=("graddft_residual",),
        default=DEFAULT_NETWORK_ARCHITECTURE,
    )
    p.add_argument(
        "--input-feature-mode",
        choices=("enhanced", "canonical", "dm21_original"),
        default=DEFAULT_INPUT_FEATURE_MODE,
    )
    p.add_argument("--include-pt2-channel", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--include-hfx-channel", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--response-hf-mode",
        choices=("approx", "strict"),
        default=DEFAULT_NEURAL_XC_RESPONSE_HF_MODE,
        help=(
            "Excited-state handling of the neural local-HF channel. 'approx' uses a "
            "scalar averaged hybrid fraction; 'strict' is gated until chi/fxx "
            "second-response contractions are implemented."
        ),
    )
    p.add_argument(
        "--pt2-channel-mode",
        choices=("scaled_projected", "local_exact"),
        default="scaled_projected",
    )
    p.add_argument("--semilocal-xc", nargs="+", default=list(b3lyp_component_basis()))
    p.add_argument("--eval-use-tda", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--skip-excitation-prediction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only write ground-state energy predictions; skip TDA/Casida excitation inference.",
    )
    p.add_argument("--grids-level", type=int, default=0)
    p.add_argument("--reference-scf-max-cycle", type=int, default=100)
    p.add_argument("--reference-scf-conv-tol", type=float, default=1e-10)
    p.add_argument("--reference-jk-backend", choices=("full", "df"), default="full")
    p.add_argument(
        "--response-df-mode",
        choices=("none", "df", "ris"),
        default="none",
        help="Two-electron integral representation stored in prepared references.",
    )
    p.add_argument("--response-ris-theta", type=float, default=0.2)
    p.add_argument("--response-ris-j-fit", choices=("s", "sp", "spd"), default="sp")
    p.add_argument("--response-ris-k-fit", choices=("s", "sp", "spd"), default="s")
    p.add_argument(
        "--reference-cache",
        default="outputs/reference_cache/closed_shell_s1_references.h5",
        help=(
            "HDF5 cache for prepared RKS/HFX reference molecules. Pass an empty "
            "string to disable."
        ),
    )
    p.add_argument("--rebuild-reference-cache", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--host-reference-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep prepared reference arrays on host memory during evaluation.",
    )
    p.add_argument("--train-scf-max-cycle", type=int, default=128)
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
        choices=("expl", "impl"),
        default="unrolled",
    )
    p.add_argument("--scf-implicit-diff-max-iter", type=int, default=6)
    p.add_argument("--scf-implicit-diff-tolerance", type=float, default=1e-6)
    p.add_argument("--scf-implicit-diff-regularization", type=float, default=1e-3)
    p.add_argument("--scf-warm-start", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--scf-warm-start-update-interval", type=int, default=1)
    p.add_argument("--outdir", required=True)
    return p.parse_args(argv)


def _apply_checkpoint_metadata(args: argparse.Namespace) -> argparse.Namespace:
    meta_path = Path(str(args.checkpoint) + ".meta.json")
    if not meta_path.exists():
        return args
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    defaults = {
        "training_mode": "self_consistent",
        "include_hfx_channel": False,
        "response_hf_mode": DEFAULT_NEURAL_XC_RESPONSE_HF_MODE,
        "include_pt2_channel": False,
        "pt2_channel_mode": "scaled_projected",
        "scf_gradient_mode": "unrolled",
        "scf_implicit_diff_max_iter": 6,
        "scf_implicit_diff_tolerance": 1e-6,
        "scf_implicit_diff_regularization": 1e-3,
        "scf_warm_start": False,
        "scf_warm_start_update_interval": 1,
    }
    for key in (
        "training_mode",
        "include_hfx_channel",
        "response_hf_mode",
        "include_pt2_channel",
        "pt2_channel_mode",
        "scf_gradient_mode",
        "scf_implicit_diff_max_iter",
        "scf_implicit_diff_tolerance",
        "scf_implicit_diff_regularization",
        "scf_warm_start",
        "scf_warm_start_update_interval",
    ):
        if key in meta and getattr(args, key, None) == defaults[key]:
            setattr(args, key, meta[key])
    return args


def _evaluation_metric_summary(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "excitation_gap_mae_ev": float(metrics["excitation_gap_mae_ev"]),
        "excitation_gap_max_ev": float(metrics["excitation_gap_max_ev"]),
        "total_mae_ev": float(metrics["total_mae_ev"]),
    }


def _evaluate_ground_only(
    prepared: list[Any],
    *,
    params: Any,
    functional: Any,
    training_config: MolecularTrainingConfig,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    energy_err_ev: list[float] = []
    for ref in prepared:
        predicted_total_h = float(
            predict_ground_state_total_energy(
                params,
                functional,
                ref.molecule,
                training_config=training_config,
            )
        )
        total_abs_err_ev = abs(predicted_total_h - ref.row.ccsd_total_energy_h) * 27.211386245988
        energy_err_ev.append(float(total_abs_err_ev))
        rows.append(
            {
                "system": ref.row.system,
                "split": ref.row.split,
                "target_total_energy_h": float(ref.row.ccsd_total_energy_h),
                "predicted_total_energy_h": float(predicted_total_h),
                "target_s1_h": float(ref.row.s1_excitation_h),
                "predicted_s1_h": float("nan"),
                "target_s1_ev": float(ref.row.s1_excitation_h * 27.211386245988),
                "predicted_s1_ev": float("nan"),
                "s1_abs_err_ev": float("nan"),
                "total_abs_err_ev": float(total_abs_err_ev),
            }
        )
    return (
        rows,
        {
            "excitation_gap_mae_ev": float("nan"),
            "excitation_gap_max_ev": float("nan"),
            "total_mae_ev": float(sum(energy_err_ev) / len(energy_err_ev))
            if energy_err_ev
            else float("nan"),
        },
    )


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
        f"include_hfx_channel={bool(args.include_hfx_channel)}, "
        f"response_hf_mode={args.response_hf_mode}, "
        f"include_pt2_channel={bool(args.include_pt2_channel)}, "
        f"pt2_channel_mode={args.pt2_channel_mode if bool(args.include_pt2_channel) else 'none'}"
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
        include_hfx_channel=bool(args.include_hfx_channel),
        response_hf_mode=str(args.response_hf_mode),
        include_pt2_channel=bool(args.include_pt2_channel),
        pt2_channel_mode=str(args.pt2_channel_mode),
        name="neural_xc_closed_shell_eval",
    )
    template = functional.init_from_molecule(jax.random.PRNGKey(int(args.seed)), prepared[0].molecule)
    params = load_params_checkpoint(args.checkpoint, template=template)

    training_config = MolecularTrainingConfig(
        mode=str(args.training_mode),
        excited_state_solver="tda" if bool(args.eval_use_tda) else "casida",
        scf_max_cycle=int(args.train_scf_max_cycle),
        scf_damping=float(args.train_scf_damping),
        scf_conv_tol_density=float(args.train_scf_conv_tol_density),
        scf_vxc_clip=float(args.train_scf_vxc_clip),
        scf_iterate_selection=str(args.scf_iterate_selection),
        scf_gradient_mode=str(args.scf_gradient_mode),
        scf_implicit_diff_max_iter=int(args.scf_implicit_diff_max_iter),
        scf_implicit_diff_tolerance=float(args.scf_implicit_diff_tolerance),
        scf_implicit_diff_regularization=float(args.scf_implicit_diff_regularization),
    )

    if bool(args.skip_excitation_prediction):
        pred_rows, metrics = _evaluate_ground_only(
            prepared,
            params=params,
            functional=functional,
            training_config=training_config,
        )
    else:
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
        **_evaluation_metric_summary(metrics),
        "predictions_csv": str(predictions_csv),
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    logger.log(f"Wrote predictions: {predictions_csv}")
    logger.log(f"Wrote summary   : {summary_path}")


if __name__ == "__main__":
    main()
