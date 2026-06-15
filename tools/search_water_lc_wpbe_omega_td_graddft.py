from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/td_graddft_matplotlib")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search LC-wPBE omega on water using TD-GradDFT's JAX "
            "lc_wpbe_local evaluator inside PySCF SCF."
        )
    )
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument("--omega-bounds", default="0.05,1.20")
    parser.add_argument("--baseline-omegas", default="0.4,0.8,0.8458180876")
    parser.add_argument("--xatol", type=float, default=5e-3)
    parser.add_argument("--maxiter", type=int, default=12)
    parser.add_argument("--scf-max-cycle", type=int, default=100)
    parser.add_argument("--conv-tol", type=float, default=1e-6)
    parser.add_argument("--occupation-tol", type=float, default=1e-8)
    parser.add_argument("--density-floor", type=float, default=1e-12)
    parser.add_argument("--no-jit", action="store_true")
    parser.add_argument("--koopmans-ip-weight", type=float, default=1.0)
    parser.add_argument("--koopmans-ea-weight", type=float, default=1.0)
    parser.add_argument("--koopmans-lumo-ea-weight", type=float, default=0.0)
    parser.add_argument(
        "--koopmans-loss-kind",
        choices=("absolute", "squared"),
        default="squared",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/water_lc_wpbe_omega_td_graddft_search",
    )
    return parser.parse_args()


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in str(raw).split(",") if part.strip())


def _bounded_omega_search(
    objective: Callable[[float], float],
    *,
    bounds: tuple[float, float],
    xatol: float,
    maxiter: int,
) -> tuple[Any, list[dict[str, float]]]:
    from scipy.optimize import minimize_scalar

    history: list[dict[str, float]] = []

    def wrapped(omega: float) -> float:
        omega_value = float(omega)
        loss = float(objective(omega_value))
        history.append(
            {
                "eval": float(len(history)),
                "omega": omega_value,
                "loss": loss,
            }
        )
        return loss

    result = minimize_scalar(
        wrapped,
        bounds=(float(bounds[0]), float(bounds[1])),
        method="bounded",
        options={"xatol": float(xatol), "maxiter": int(maxiter)},
    )
    return result, history


def _evaluate_td_graddft_omega(*, vector: list[float], args: argparse.Namespace) -> dict[str, Any]:
    try:
        from tools.search_water_rsh_params_pyscf import _evaluate
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tools.search_water_lc_wpbe_omega_td_graddft requires "
            "tools.search_water_rsh_params_pyscf for full CLI evaluation."
        ) from exc
    return _evaluate(vector=vector, args=args)


def _write_plot(history: list[dict[str, Any]], outdir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    evals = [int(row["eval"]) for row in history]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
    axes[0].plot(evals, [float(row["loss"]) for row in history], "-o", label="loss")
    axes[0].plot(
        evals,
        [float(row["koopmans_ip_mae"]) for row in history],
        "--",
        label="|HOMO(N)+IP|",
    )
    axes[0].plot(
        evals,
        [float(row["koopmans_ea_mae"]) for row in history],
        "--",
        label="|HOMO(N+1)+EA|",
    )
    axes[0].set_ylabel("Hartree")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(evals, [float(row["omega"]) for row in history], "-o", color="#b23a48")
    axes[1].set_xlabel("scipy objective evaluation")
    axes[1].set_ylabel("omega")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    path = outdir / "omega_td_graddft_search.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    bounds_raw = _parse_float_tuple(args.omega_bounds)
    if len(bounds_raw) != 2:
        raise ValueError("--omega-bounds must contain exactly lower,upper.")
    bounds = (float(bounds_raw[0]), float(bounds_raw[1]))
    cache: dict[float, dict[str, Any]] = {}

    eval_args = argparse.Namespace(
        basis=str(args.basis),
        grid_level=int(args.grid_level),
        local_xc="td_graddft_lc_wpbe",
        density_floor=float(args.density_floor),
        no_jit=bool(args.no_jit),
        scf_max_cycle=int(args.scf_max_cycle),
        conv_tol=float(args.conv_tol),
        occupation_tol=float(args.occupation_tol),
        koopmans_ip_weight=float(args.koopmans_ip_weight),
        koopmans_ea_weight=float(args.koopmans_ea_weight),
        koopmans_lumo_ea_weight=float(args.koopmans_lumo_ea_weight),
        koopmans_loss_kind=str(args.koopmans_loss_kind),
        nonconverged_penalty=100.0,
    )

    def evaluate(omega: float) -> dict[str, Any]:
        key = round(float(omega), 10)
        if key not in cache:
            row = _evaluate_td_graddft_omega(vector=[0.0, 1.0, float(omega)], args=eval_args)
            cache[key] = row
            print(
                f"eval={len(cache):03d} omega={row['omega']:.6f} "
                f"loss={row['loss']:.6e} "
                f"kip={row['koopmans_ip_mae']:.3e} "
                f"kea={row['koopmans_ea_mae']:.3e} "
                f"conv={row['all_converged']}",
                flush=True,
            )
        return cache[key]

    def objective(omega: float) -> float:
        return float(evaluate(float(omega))["loss"])

    baselines: list[dict[str, Any]] = []
    for omega in _parse_float_tuple(str(args.baseline_omegas)):
        row = dict(evaluate(float(omega)))
        row["label"] = f"baseline_omega_{omega:g}"
        baselines.append(row)

    result, scipy_history = _bounded_omega_search(
        objective,
        bounds=bounds,
        xatol=float(args.xatol),
        maxiter=int(args.maxiter),
    )
    history: list[dict[str, Any]] = []
    for scipy_row in scipy_history:
        key = round(float(scipy_row["omega"]), 10)
        row = dict(cache[key])
        row["eval"] = int(scipy_row["eval"])
        history.append(row)

    best_key = min(cache, key=lambda key: float(cache[key]["loss"]))
    best_row = cache[best_key]
    plot_path = _write_plot(history, outdir)
    summary = {
        "backend": "td_graddft_jax_eval_xc_in_pyscf_scf",
        "local_xc": "lc_wpbe_local",
        "rsh_form": "sr=0, lr=1, omega searched",
        "basis": str(args.basis),
        "grid_level": int(args.grid_level),
        "bounds": [float(bounds[0]), float(bounds[1])],
        "xatol": float(args.xatol),
        "maxiter": int(args.maxiter),
        "scf_max_cycle": int(args.scf_max_cycle),
        "conv_tol": float(args.conv_tol),
        "td_graddft_jit": not bool(args.no_jit),
        "scipy_success": bool(result.success),
        "scipy_message": str(result.message),
        "scipy_nfev": int(result.nfev),
        "scipy_x": float(result.x),
        "scipy_fun": float(result.fun),
        "best_loss": float(best_row["loss"]),
        "best_sr_hf_fraction": 0.0,
        "best_lr_hf_fraction": 1.0,
        "best_omega": float(best_row["omega"]),
        "best_row": best_row,
        "baselines": baselines,
        "history": history,
        "koopmans_ip_weight": float(args.koopmans_ip_weight),
        "koopmans_ea_weight": float(args.koopmans_ea_weight),
        "koopmans_lumo_ea_weight": float(args.koopmans_lumo_ea_weight),
        "koopmans_loss_kind": str(args.koopmans_loss_kind),
        "plot": str(plot_path) if plot_path is not None else None,
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
