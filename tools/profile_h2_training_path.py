from __future__ import annotations

import argparse
import atexit
import json
import os
import runpy
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile high-level TD-GradDFT training path segments by monkeypatching "
            "selected functions and then running a target script in-process."
        )
    )
    parser.add_argument(
        "--target",
        default="tools/h2_s1_tda_train5_dense100_vs_fci.py",
        help="Path to the Python script to run under profiling.",
    )
    parser.add_argument(
        "--profile-json",
        default=None,
        help="Optional JSON output path for aggregated timing statistics.",
    )
    parser.add_argument(
        "--profile-txt",
        default=None,
        help="Optional text output path for aggregated timing statistics.",
    )
    parser.add_argument(
        "target_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the target script. Prefix with -- to separate.",
    )
    return parser.parse_args()


def _strip_double_dash(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def _detect_outdir(forwarded_args: list[str]) -> Path | None:
    for idx, item in enumerate(forwarded_args[:-1]):
        if item == "--outdir":
            return Path(forwarded_args[idx + 1])
    return None


def _install_wrappers(stats: dict[str, dict[str, float]]) -> None:
    from td_graddft.features import (
        restricted_grid_features_with_gradients,
        restricted_transition_response_features,
    )
    from td_graddft.neural_xc import NeuralXCHybridFunctional
    from td_graddft.scf import differentiable as differentiable_module
    from td_graddft.scf.differentiable import DifferentiableSCF
    from td_graddft.tddft import response as response_module
    from td_graddft.training import targets as training_targets

    def wrap(obj: Any, attr: str, label: str | None = None) -> None:
        original = getattr(obj, attr)
        key = label or f"{obj.__name__}.{attr}" if hasattr(obj, "__name__") else attr

        def wrapped(*args: Any, **kwargs: Any):
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                entry = stats[key]
                entry["count"] += 1.0
                entry["total_s"] += dt
                entry["max_s"] = max(entry["max_s"], dt)

        setattr(obj, attr, wrapped)

    wrap(DifferentiableSCF, "run", "DifferentiableSCF.run")
    wrap(DifferentiableSCF, "_full_scf", "DifferentiableSCF._full_scf")
    wrap(
        DifferentiableSCF,
        "_full_scf_implicit_commutator",
        "DifferentiableSCF._full_scf_implicit_commutator",
    )
    wrap(training_targets, "ground_state_mse_loss", "targets.ground_state_mse_loss")
    wrap(training_targets, "_solve_excited_states", "targets._solve_excited_states")
    wrap(
        NeuralXCHybridFunctional,
        "bind_to_molecule_for_scf",
        "NeuralXCHybridFunctional.bind_to_molecule_for_scf",
    )
    wrap(
        NeuralXCHybridFunctional,
        "projected_hf_grid_contribution_components",
        "NeuralXCHybridFunctional.projected_hf_grid_contribution_components",
    )
    wrap(
        NeuralXCHybridFunctional,
        "channel_coefficients",
        "NeuralXCHybridFunctional.channel_coefficients",
    )
    wrap(
        NeuralXCHybridFunctional,
        "_strict_total_potential_components",
        "NeuralXCHybridFunctional._strict_total_potential_components",
    )
    wrap(
        NeuralXCHybridFunctional,
        "_strict_total_response_tensor",
        "NeuralXCHybridFunctional._strict_total_response_tensor",
    )
    wrap(
        differentiable_module,
        "_resolved_xc_object",
        "differentiable._resolved_xc_object",
    )
    wrap(
        differentiable_module,
        "_grid_xc_potential_components_from_resolved",
        "differentiable._grid_xc_potential_components_from_resolved",
    )
    wrap(
        differentiable_module,
        "_build_vxc_matrix_from_components",
        "differentiable._build_vxc_matrix_from_components",
    )
    # `restricted_transition_response_features` is imported directly in response.py,
    # so patch both the feature module symbol and the response-module alias.
    def wrapped_features(*args: Any, **kwargs: Any):
        t0 = time.perf_counter()
        try:
            return restricted_transition_response_features(*args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            entry = stats["features.restricted_transition_response_features"]
            entry["count"] += 1.0
            entry["total_s"] += dt
            entry["max_s"] = max(entry["max_s"], dt)

    import td_graddft.features as features_module

    features_module.restricted_transition_response_features = wrapped_features
    response_module.restricted_transition_response_features = wrapped_features

    def wrapped_grid_features(*args: Any, **kwargs: Any):
        t0 = time.perf_counter()
        try:
            return restricted_grid_features_with_gradients(*args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            entry = stats["features.restricted_grid_features_with_gradients"]
            entry["count"] += 1.0
            entry["total_s"] += dt
            entry["max_s"] = max(entry["max_s"], dt)

    features_module.restricted_grid_features_with_gradients = wrapped_grid_features
    differentiable_module.restricted_grid_features_with_gradients = wrapped_grid_features


def _write_outputs(
    stats: dict[str, dict[str, float]],
    *,
    profile_json: Path | None,
    profile_txt: Path | None,
) -> None:
    rows = []
    for key, entry in stats.items():
        count = int(entry["count"])
        total = float(entry["total_s"])
        rows.append(
            {
                "label": key,
                "count": count,
                "total_s": total,
                "avg_s": total / max(count, 1),
                "max_s": float(entry["max_s"]),
            }
        )
    rows.sort(key=lambda item: item["total_s"], reverse=True)

    if profile_json is not None:
        profile_json.parent.mkdir(parents=True, exist_ok=True)
        profile_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if profile_txt is not None:
        profile_txt.parent.mkdir(parents=True, exist_ok=True)
        lines = ["label,count,total_s,avg_s,max_s"]
        lines.extend(
            f"{row['label']},{row['count']},{row['total_s']:.6f},{row['avg_s']:.6f},{row['max_s']:.6f}"
            for row in rows
        )
        profile_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n[profile] aggregated timings")
    for row in rows:
        print(
            f"[profile] {row['label']}: count={row['count']} "
            f"total={row['total_s']:.3f}s avg={row['avg_s']:.3f}s max={row['max_s']:.3f}s"
        )


def main() -> None:
    args = _parse_args()
    forwarded_args = _strip_double_dash(list(args.target_args))
    outdir = _detect_outdir(forwarded_args)

    profile_json = Path(args.profile_json) if args.profile_json is not None else None
    profile_txt = Path(args.profile_txt) if args.profile_txt is not None else None
    if outdir is not None:
        if profile_json is None:
            profile_json = outdir / "profile_summary.json"
        if profile_txt is None:
            profile_txt = outdir / "profile_summary.txt"

    stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "total_s": 0.0, "max_s": 0.0}
    )
    _install_wrappers(stats)
    atexit.register(
        _write_outputs,
        stats,
        profile_json=profile_json,
        profile_txt=profile_txt,
    )

    target = Path(args.target)
    sys.argv = [str(target)] + forwarded_args
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
