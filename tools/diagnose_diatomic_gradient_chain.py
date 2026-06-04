from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax

from td_graddft import neural_xc
from td_graddft.training import (
    ExcitedStateTrainingConfig,
    GroundStateCoreTrainingConfig,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    ground_state_mse_loss_pointwise_dataset,
    load_params_checkpoint,
    make_ground_state_loss_and_grad,
)


def _load_tool_module(name: str, relative_path: str) -> Any:
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


H2 = _load_tool_module("_h2_s1_gradient_diagnostic", "tools/h2_s1_tda_train5_dense100_vs_fci.py")
H2PLUS = _load_tool_module("_h2plus_s1_gradient_diagnostic", "tools/h2plus_s1_tda_train5_dense100.py")


def _path_to_str(path: Any) -> str:
    parts: list[str] = []
    for entry in path:
        key = getattr(entry, "key", None)
        idx = getattr(entry, "idx", None)
        name = getattr(entry, "name", None)
        if key is not None:
            parts.append(str(key))
        elif idx is not None:
            parts.append(str(idx))
        elif name is not None:
            parts.append(str(name))
        else:
            parts.append(str(entry))
    return "/".join(parts)


def _as_array(value: Any) -> jnp.ndarray:
    return jnp.asarray(value)


def _leaf_stats(params: Any, grads: Any) -> list[dict[str, Any]]:
    param_items = jax.tree_util.tree_flatten_with_path(params)[0]
    grad_items = jax.tree_util.tree_flatten_with_path(grads)[0]
    rows: list[dict[str, Any]] = []
    for (param_path, param_leaf), (grad_path, grad_leaf) in zip(param_items, grad_items, strict=True):
        path = _path_to_str(param_path)
        grad_path_str = _path_to_str(grad_path)
        if path != grad_path_str:
            raise RuntimeError(f"Parameter/gradient tree mismatch: {path} != {grad_path_str}")
        p = np.asarray(param_leaf)
        g = np.asarray(grad_leaf)
        finite = np.isfinite(g)
        nonzero = np.abs(g) > 0.0
        rows.append(
            {
                "path": path,
                "shape": "x".join(str(dim) for dim in g.shape),
                "size": int(g.size),
                "param_l2": float(np.linalg.norm(p.reshape(-1))) if p.size else 0.0,
                "param_abs_max": float(np.max(np.abs(p))) if p.size else 0.0,
                "grad_l2": float(np.linalg.norm(g.reshape(-1))) if g.size else 0.0,
                "grad_abs_max": float(np.max(np.abs(g))) if g.size else 0.0,
                "grad_abs_mean": float(np.mean(np.abs(g))) if g.size else 0.0,
                "grad_nonzero_fraction": float(np.mean(nonzero)) if g.size else 0.0,
                "grad_finite_fraction": float(np.mean(finite)) if g.size else 1.0,
            }
        )
    return rows


def _tree_l2(tree: Any) -> float:
    leaves = jax.tree_util.tree_leaves(tree)
    total = 0.0
    for leaf in leaves:
        arr = np.asarray(leaf)
        total += float(np.sum(arr * arr))
    return float(np.sqrt(total))


def _tree_abs_max(tree: Any) -> float:
    values = [float(np.max(np.abs(np.asarray(leaf)))) for leaf in jax.tree_util.tree_leaves(tree) if np.asarray(leaf).size]
    return max(values) if values else 0.0


def _metric_scalar(metrics: dict[str, Any], key: str) -> float | None:
    if key not in metrics:
        return None
    arr = np.asarray(metrics[key])
    if arr.size == 0:
        return None
    return float(np.nanmean(arr.astype(np.float64)))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _finite_difference_check(loss_fn: Any, params: Any, grads: Any, *, eps: float) -> dict[str, Any]:
    grad_leaves, treedef = jax.tree_util.tree_flatten(grads)
    param_leaves = jax.tree_util.tree_leaves(params)
    path_items = jax.tree_util.tree_flatten_with_path(grads)[0]
    grad_abs_max = [
        float(np.max(np.abs(np.asarray(leaf)))) if np.asarray(leaf).size else 0.0
        for leaf in grad_leaves
    ]
    leaf_index = int(np.argmax(np.asarray(grad_abs_max)))
    grad_leaf = jnp.asarray(grad_leaves[leaf_index])
    grad_norm = jnp.linalg.norm(grad_leaf.reshape(-1))
    direction = jnp.where(grad_norm > 0.0, grad_leaf / grad_norm, jnp.zeros_like(grad_leaf))
    plus_leaves = list(param_leaves)
    minus_leaves = list(param_leaves)
    plus_leaves[leaf_index] = jnp.asarray(plus_leaves[leaf_index]) + float(eps) * direction
    minus_leaves[leaf_index] = jnp.asarray(minus_leaves[leaf_index]) - float(eps) * direction
    params_plus = treedef.unflatten(plus_leaves)
    params_minus = treedef.unflatten(minus_leaves)
    loss_plus, _ = loss_fn(params_plus)
    loss_minus, _ = loss_fn(params_minus)
    fd = (float(loss_plus) - float(loss_minus)) / (2.0 * float(eps))
    ad = float(jnp.vdot(grad_leaf, direction))
    rel_err = abs(fd - ad) / max(1.0, abs(fd), abs(ad))
    return {
        "leaf_path": _path_to_str(path_items[leaf_index][0]),
        "eps": float(eps),
        "loss_plus": float(loss_plus),
        "loss_minus": float(loss_minus),
        "finite_difference_directional_derivative": fd,
        "ad_directional_derivative": ad,
        "relative_error_scaled": float(rel_err),
    }


def _h2_args(case: str, cache: Path, outdir: Path) -> argparse.Namespace:
    argv = [
        "--basis",
        "def2-svp",
        "--xc",
        "b3lyp",
        "--r-min",
        "0.4",
        "--r-max",
        "6.0",
        "--train-points",
        "5",
        "--dense-points",
        "100",
        "--steps",
        "1",
        "--learning-rate",
        "1e-4",
        "--lr-decay-every",
        "400",
        "--lr-decay-factor",
        "0.5",
        "--training-mode",
        "self_consistent",
        "--objective",
        "joint",
        "--grids-level",
        "2",
        "--integral-backend",
        "gpu",
        "--reference-scf-backend",
        "jax_rks",
        "--train-scf-convergence-metric",
        "energy",
        "--scf-gradient-mode",
        "impl",
        "--reference-cache",
        str(cache),
        "--outdir",
        str(outdir),
    ]
    if case == "h2-nopt2":
        argv.append("--no-include-pt2-channel")
    elif case == "h2-strictpt2":
        argv.extend(
            [
                "--include-pt2-channel",
                "--pt2-channel-mode",
                "scaled_projected",
                "--response-pt2-mode",
                "strict",
            ]
        )
    else:
        raise ValueError(f"Unsupported H2 case: {case}")
    return H2.parse_args(argv)


def _h2plus_args(cache: Path, outdir: Path) -> argparse.Namespace:
    return H2PLUS.parse_args(
        [
            "--basis",
            "def2-svp",
            "--xc",
            "b3lyp",
            "--r-min",
            "0.4",
            "--r-max",
            "6.0",
            "--train-points",
            "5",
            "--dense-points",
            "100",
            "--steps",
            "1",
            "--learning-rate",
            "1e-4",
            "--lr-decay-every",
            "400",
            "--lr-decay-factor",
            "0.5",
            "--objective",
            "joint",
            "--reference-excited-method",
            "orbital",
            "--grids-level",
            "2",
            "--integral-backend",
            "gpu",
            "--reference-scf-device",
            "cpu",
            "--train-scf-convergence-metric",
            "energy",
            "--scf-gradient-mode",
            "impl",
            "--reference-cache",
            str(cache),
            "--outdir",
            str(outdir),
        ]
    )


def _select_points(points: list[Any], indices: str) -> list[Any]:
    if indices == "all":
        return points
    selected: list[Any] = []
    for raw in indices.split(","):
        idx = int(raw.strip())
        selected.append(points[idx])
    return selected


def _build_h2(case: str, cache: Path, outdir: Path, indices: str) -> tuple[Any, Any, Any, tuple[Any, ...], dict[str, Any]]:
    args = _h2_args(case, cache, outdir)
    logger = H2.RunLogger(outdir / "reference_load.log")
    r_values = np.linspace(float(args.r_min), float(args.r_max), int(args.train_points))
    points = H2._get_or_build_reference_curve(r_values, args=args, logger=logger, label="train_ref")
    points = _select_points(points, indices)
    data = H2.build_s1_training_data(
        points,
        s1_weight=float(args.s1_weight),
        density_constraint_weight=float(args.density_constraint_weight),
    )
    functional = H2._make_s1_functional(args)
    config = H2.GroundStateTrainingConfig(
        mode=str(args.training_mode),
        energy_mse_weight=float(args.energy_mse_weight),
        energy_mae_weight=float(args.energy_mae_weight),
        s1_constraint_use_tda=bool(args.s1_use_tda),
        scf_max_cycle=H2._HELPERS._resolve_train_scf_max_cycle(args.train_scf_max_cycle),
        scf_damping=float(args.train_scf_damping),
        scf_conv_tol_energy=args.train_scf_conv_tol_energy,
        scf_convergence_metric=str(args.train_scf_convergence_metric),
        scf_conv_tol_density=float(args.train_scf_conv_tol_density),
        scf_vxc_clip=float(args.train_scf_vxc_clip),
        scf_iterate_selection=str(args.scf_iterate_selection),
        scf_require_convergence=bool(args.scf_require_convergence),
        scf_gradient_mode=str(args.scf_gradient_mode),
        scf_implicit_diff_tolerance=float(args.scf_implicit_diff_tolerance),
        scf_implicit_diff_regularization=float(args.scf_implicit_diff_regularization),
    )
    return args, functional, config, tuple(data), {"n_points": len(points)}


def _build_h2plus(cache: Path, outdir: Path, indices: str) -> tuple[Any, Any, Any, tuple[Any, ...], dict[str, Any]]:
    args = _h2plus_args(cache, outdir)
    logger = H2PLUS.RunLogger(outdir / "reference_load.log")
    r_values = np.linspace(float(args.r_min), float(args.r_max), int(args.train_points))
    points = [
        H2PLUS.get_or_build_reference_point(float(r_value), args=args, logger=logger)
        for r_value in r_values
    ]
    points = _select_points(points, indices)
    data = H2PLUS.build_training_data(
        points,
        s1_weight=float(args.s1_weight),
        density_constraint_weight=float(args.density_constraint_weight),
        density_matrix_constraint_weight=float(args.density_matrix_constraint_weight),
    )
    functional = neural_xc.Functional(
        semilocal_xc=tuple(str(name) for name in args.semilocal_xc),
        hidden_dims=tuple(int(value) for value in args.hidden_dims),
        architecture=str(args.network_architecture),
        input_feature_mode=str(args.input_feature_mode),
        include_pt2_channel=bool(getattr(args, "include_pt2_channel", False)),
        pt2_channel_mode=str(getattr(args, "pt2_channel_mode", "scaled_projected")),
        response_hf_mode="approx",
        response_pt2_mode=str(getattr(args, "response_pt2_mode", "approx")),
        name="neural_xc_h2plus_s1_tda",
    )
    coefficient_prior = neural_xc.resolve_coefficient_prior_values(
        tuple(str(name) for name in args.semilocal_xc)
    )
    config = GroundStateTrainingConfig.from_parts(
        core=GroundStateCoreTrainingConfig(
            mode="self_consistent",
            energy_mse_weight=float(args.energy_mse_weight),
            energy_mae_weight=float(args.energy_mae_weight),
            energy_normalization=str(args.energy_normalization),
            coefficient_prior_weight=float(args.coefficient_prior_weight),
            coefficient_prior_values=coefficient_prior,
            scf_max_cycle=(
                H2PLUS._TRAIN_SCF_SAFETY_MAX_CYCLE
                if int(args.train_scf_max_cycle) <= 0
                else int(args.train_scf_max_cycle)
            ),
            scf_damping=float(args.train_scf_damping),
            scf_level_shift=float(args.train_scf_level_shift),
            scf_conv_tol_energy=args.train_scf_conv_tol_energy,
            scf_convergence_metric=str(args.train_scf_convergence_metric),
            scf_conv_tol_density=float(args.train_scf_conv_tol_density),
            scf_vxc_clip=float(args.train_scf_vxc_clip),
            scf_iterate_selection=str(args.scf_iterate_selection),
            scf_gradient_mode=str(args.scf_gradient_mode),
            scf_implicit_diff_tolerance=float(args.scf_implicit_diff_tolerance),
            scf_implicit_diff_regularization=float(args.scf_implicit_diff_regularization),
        ),
        excited_state=ExcitedStateTrainingConfig(
            s1_constraint_use_tda=bool(args.s1_use_tda),
        ),
    )
    return args, functional, config, tuple(data), {"n_points": len(points)}


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.case in {"h2-nopt2", "h2-strictpt2"}:
        run_args, functional, config, data, build_info = _build_h2(
            args.case,
            Path(args.reference_cache),
            outdir,
            args.point_indices,
        )
        loss_fn_name = None
    elif args.case == "h2plus-nopt2":
        run_args, functional, config, data, build_info = _build_h2plus(
            Path(args.reference_cache),
            outdir,
            args.point_indices,
        )
        loss_fn_name = ground_state_mse_loss_pointwise_dataset
    else:
        raise ValueError(f"Unsupported case: {args.case}")

    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(int(run_args.seed)),
        data[0].molecule,
        optax.adam(float(run_args.learning_rate)),
    )
    params = state.params
    if args.checkpoint:
        params = load_params_checkpoint(Path(args.checkpoint), template=params)

    loss_and_grad = make_ground_state_loss_and_grad(
        functional,
        training_config=config,
        loss_fn=loss_fn_name,
    )
    objective = (
        ground_state_mse_loss_pointwise_dataset
        if loss_fn_name is not None
        else H2.ground_state_mse_loss
    )

    def loss_only(local_params: Any) -> tuple[Any, dict[str, Any]]:
        return objective(
            local_params,
            functional,
            data,
            training_config=config,
        )

    t0 = time.perf_counter()
    loss, metrics, grads = loss_and_grad(params, data)
    elapsed = time.perf_counter() - t0
    rows = _leaf_stats(params, grads)
    rows_sorted = sorted(rows, key=lambda row: float(row["grad_l2"]), reverse=True)
    _write_csv(outdir / "gradient_leaf_stats.csv", rows_sorted)

    fd = None
    if not bool(args.skip_finite_difference):
        fd = _finite_difference_check(loss_only, params, grads, eps=float(args.fd_eps))
        _write_csv(outdir / "finite_difference_check.csv", [fd])

    summary = {
        "case": args.case,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "reference_cache": str(args.reference_cache),
        "point_indices": args.point_indices,
        "n_points": int(build_info["n_points"]),
        "elapsed_s": float(elapsed),
        "loss": float(loss),
        "raw_grad_norm": _metric_scalar(metrics, "raw_grad_norm"),
        "grad_norm": _metric_scalar(metrics, "grad_norm"),
        "grad_abs_max": _metric_scalar(metrics, "grad_abs_max"),
        "tree_grad_norm_recomputed": _tree_l2(grads),
        "tree_grad_abs_max_recomputed": _tree_abs_max(grads),
        "nonfinite_grad_fraction": _metric_scalar(metrics, "nonfinite_grad_fraction"),
        "scf_converged_fraction": _metric_scalar(metrics, "scf_converged_fraction"),
        "scf_cycles_mean": _metric_scalar(metrics, "scf_cycles_mean"),
        "scf_cycles_max": _metric_scalar(metrics, "scf_cycles_max"),
        "s1_mae_h": _metric_scalar(metrics, "s1_mae"),
        "s1_predicted_h": _metric_scalar(metrics, "s1_predicted"),
        "s1_target_h": _metric_scalar(metrics, "s1_target"),
        "top_gradient_leaves": rows_sorted[:10],
        "finite_difference": fd,
        "outputs": {
            "summary_json": str(outdir / "gradient_chain_summary.json"),
            "leaf_stats_csv": str(outdir / "gradient_leaf_stats.csv"),
            "finite_difference_csv": str(outdir / "finite_difference_check.csv"),
        },
    }
    (outdir / "gradient_chain_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose full-chain S1 training gradients.")
    parser.add_argument(
        "--case",
        choices=("h2-nopt2", "h2-strictpt2", "h2plus-nopt2"),
        required=True,
    )
    parser.add_argument("--reference-cache", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--point-indices", default="0", help="Comma-separated train point indices, or all.")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--fd-eps", type=float, default=1e-4)
    parser.add_argument("--skip-finite-difference", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    diagnose(parse_args(argv))


if __name__ == "__main__":
    main()
