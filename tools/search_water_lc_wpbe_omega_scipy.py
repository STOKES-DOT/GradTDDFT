from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/td_graddft_matplotlib")

import jax
import jax.numpy as jnp

from td_graddft.nn_rsh import (
    AtomCenteredDensityDescriptorConfig,
    ResolvedRSHParameters,
    get_rsh_functional_preset,
    make_atom_centered_density_rsh_functional,
    make_rsh_template,
    make_self_supervised_rsh_loss,
    rsh_preset_default_params,
)
from td_graddft.data.reference import restricted_reference_from_pyscf
from td_graddft.training import (
    GroundStateDatum,
    GroundStateTrainingConfig,
    save_params_checkpoint,
)


WATER_GEOMETRY = """
O  0.000000  0.000000  0.117790
H  0.000000  0.755453 -0.471161
H  0.000000 -0.755453 -0.471161
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derivative-free scipy search of LC-wPBE omega on water using the "
            "same self-supervised Koopmans IP/EA loss as NN-RSH training."
        ),
    )
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument("--reference-xc", default="pbe")
    parser.add_argument("--rsh-preset", default="lc-wpbe")
    parser.add_argument(
        "--rsh-omega-source",
        choices=("canonical", "optxc"),
        default="optxc",
    )
    parser.add_argument(
        "--omega-bounds",
        default="",
        help="Optional explicit lower,upper bounds. Defaults to preset bounds.",
    )
    parser.add_argument("--xatol", type=float, default=5e-3)
    parser.add_argument("--maxiter", type=int, default=8)
    parser.add_argument("--omega-grid", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6")
    parser.add_argument("--scf-max-cycle", type=int, default=1)
    parser.add_argument(
        "--scf-gradient-mode",
        choices=("unrolled", "implicit_commutator"),
        default="implicit_commutator",
    )
    parser.add_argument("--koopmans-ip-weight", type=float, default=1.0)
    parser.add_argument("--koopmans-ea-weight", type=float, default=1.0)
    parser.add_argument("--koopmans-lumo-ea-weight", type=float, default=0.0)
    parser.add_argument(
        "--koopmans-detach-charged-states",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the non-differentiable JAX UKS charged-state diagnostic path. "
            "Useful for PySCF-like black-box omega searches and finite-difference training."
        ),
    )
    parser.add_argument(
        "--koopmans-loss-kind",
        choices=("absolute", "squared"),
        default="squared",
    )
    parser.add_argument("--janak-weight", type=float, default=0.0)
    parser.add_argument("--fractional-weight", type=float, default=0.0)
    parser.add_argument("--long-range-correction-weight", type=float, default=0.0)
    parser.add_argument("--prior-weight", type=float, default=0.0)
    parser.add_argument("--radial-centers", default="0.6,1.4,2.6")
    parser.add_argument("--radial-width", type=float, default=0.5)
    parser.add_argument("--max-angular", type=int, default=1)
    parser.add_argument("--atom-hidden-dims", default="16,16")
    parser.add_argument("--pooled-hidden-dims", default="16")
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--outdir",
        default="outputs/water_lc_wpbe_omega_scipy_search",
    )
    return parser.parse_args()


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    cleaned = [part.strip() for part in raw.split(",") if part.strip()]
    return tuple(int(part) for part in cleaned)


def _omega_grid_covering_bounds(
    omega_grid: tuple[float, ...],
    bounds: tuple[float, float],
) -> tuple[float, ...]:
    values = [float(value) for value in omega_grid]
    lower, upper = (float(bounds[0]), float(bounds[1]))
    if not values:
        values = [lower, upper]
    if min(values) > lower:
        values.append(lower)
    if max(values) < upper:
        values.append(upper)
    return tuple(sorted(set(round(value, 12) for value in values)))


def _metric_scalar(metrics: dict[str, jnp.ndarray], key: str) -> float:
    value = jnp.asarray(metrics[key])
    if value.ndim == 0:
        return float(value)
    return float(value.reshape(-1)[0])


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
        options={
            "xatol": float(xatol),
            "maxiter": int(maxiter),
        },
    )
    return result, history


def _build_water_reference(
    *,
    basis: str,
    xc: str,
    grid_level: int,
    omega_grid: tuple[float, ...],
) -> Any:
    from pyscf import dft, gto

    mol = gto.M(
        atom=WATER_GEOMETRY,
        unit="Angstrom",
        basis=str(basis),
        spin=0,
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = str(xc)
    mf.grids.level = int(grid_level)
    mf.conv_tol = 1e-10
    mf.kernel()
    return restricted_reference_from_pyscf(
        mf,
        compute_local_hfx_features=True,
        compute_local_hfx_aux=True,
        hfx_omega_values=omega_grid,
    )


def _write_plot(history: list[dict[str, Any]], outdir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    evals = [int(row["eval"]) for row in history]
    losses = [float(row["loss"]) for row in history]
    omegas = [float(row["omega"]) for row in history]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
    axes[0].plot(evals, losses, "-o", label="loss")
    axes[0].set_ylabel("Koopmans loss")
    axes[0].grid(alpha=0.25, linewidth=0.6)
    axes[0].legend(frameon=False)
    axes[1].plot(evals, omegas, "-o", color="#b23a48", label="omega")
    axes[1].set_xlabel("scipy objective evaluation")
    axes[1].set_ylabel("omega")
    axes[1].grid(alpha=0.25, linewidth=0.6)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    path = outdir / "omega_scipy_search.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    preset = get_rsh_functional_preset(str(args.rsh_preset))
    template = make_rsh_template(
        str(args.rsh_preset),
        omega_source=str(args.rsh_omega_source),
    )
    if str(args.omega_bounds).strip():
        omega_bounds = _parse_float_tuple(str(args.omega_bounds))
        if len(omega_bounds) != 2:
            raise ValueError("--omega-bounds must contain exactly lower,upper.")
        bounds = (float(omega_bounds[0]), float(omega_bounds[1]))
        template = replace(
            template,
            omega_bounds=bounds,
            default_omega=min(max(float(template.default_omega), bounds[0]), bounds[1]),
        )
    else:
        bounds = tuple(float(x) for x in template.omega_bounds)
    omega_grid = _omega_grid_covering_bounds(_parse_float_tuple(args.omega_grid), bounds)
    molecule = _build_water_reference(
        basis=str(args.basis),
        xc=str(args.reference_xc),
        grid_level=int(args.grid_level),
        omega_grid=omega_grid,
    )

    descriptor_config = AtomCenteredDensityDescriptorConfig(
        radial_centers=_parse_float_tuple(args.radial_centers),
        radial_width=float(args.radial_width),
        max_angular=int(args.max_angular),
    )
    local_xc_spec = preset.jax_local_xc_spec or "pbe"
    functional = make_atom_centered_density_rsh_functional(
        local_xc_spec=local_xc_spec,
        descriptor_config=descriptor_config,
        atom_hidden_dims=_parse_int_tuple(args.atom_hidden_dims),
        pooled_hidden_dims=_parse_int_tuple(args.pooled_hidden_dims),
        embedding_dim=int(args.embedding_dim),
        template=template,
        fallback_omega_values=omega_grid,
    )
    base_params = functional.init_from_molecule(jax.random.PRNGKey(int(args.seed)), molecule)
    initial_resolved = rsh_preset_default_params(
        str(args.rsh_preset),
        omega_source=str(args.rsh_omega_source),
    )
    base_params = functional.params_with_resolved(
        base_params,
        initial_resolved,
        molecule=molecule,
        preserve_network=True,
    )
    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.asarray(molecule.mf_energy),
    )
    training_config = GroundStateTrainingConfig(
        mode="self_consistent",
        scf_gradient_mode=str(args.scf_gradient_mode),
        scf_max_cycle=int(args.scf_max_cycle),
        scf_require_convergence=False,
    )
    loss_fn = make_self_supervised_rsh_loss(
        functional,
        training_config=training_config,
        janak_weight=float(args.janak_weight),
        fractional_weight=float(args.fractional_weight),
        koopmans_ip_weight=float(args.koopmans_ip_weight),
        koopmans_ea_weight=float(args.koopmans_ea_weight),
        koopmans_lumo_ea_weight=float(args.koopmans_lumo_ea_weight),
        koopmans_loss_kind=str(args.koopmans_loss_kind),
        koopmans_detach_charged_states=bool(args.koopmans_detach_charged_states),
        koopmans_differentiate_charged_orbitals=False,
        long_range_correction_weight=float(args.long_range_correction_weight),
        prior_weight=float(args.prior_weight),
    )
    cache: dict[float, tuple[float, dict[str, Any], Any]] = {}

    def params_for_omega(omega: float) -> Any:
        resolved = ResolvedRSHParameters(
            sr_hf_fraction=jnp.asarray(float(preset.default_sr_hf_fraction)),
            lr_hf_fraction=jnp.asarray(float(preset.default_lr_hf_fraction)),
            omega=jnp.asarray(float(omega)),
        )
        return functional.params_with_resolved(
            base_params,
            resolved,
            molecule=molecule,
            preserve_network=True,
        )

    def objective(omega: float) -> float:
        key = round(float(omega), 10)
        if key in cache:
            return cache[key][0]
        params = params_for_omega(float(omega))
        loss, metrics = loss_fn(params, functional, datum)
        row = {
            "omega": float(_metric_scalar(metrics, "omega")),
            "loss": float(loss),
            "koopmans_ip_mae": _metric_scalar(metrics, "koopmans_ip_mae"),
            "koopmans_ea_mae": _metric_scalar(metrics, "koopmans_ea_mae"),
            "koopmans_lumo_ea_mae": _metric_scalar(metrics, "koopmans_lumo_ea_mae"),
            "koopmans_ip_residual": _metric_scalar(metrics, "koopmans_ip_residual"),
            "koopmans_ea_residual": _metric_scalar(metrics, "koopmans_ea_residual"),
            "neutral_energy": _metric_scalar(metrics, "koopmans_neutral_energy"),
            "cation_energy": _metric_scalar(metrics, "koopmans_cation_energy"),
            "anion_energy": _metric_scalar(metrics, "koopmans_anion_energy"),
            "sr_hf_fraction": _metric_scalar(metrics, "sr_hf_fraction"),
            "lr_hf_fraction": _metric_scalar(metrics, "lr_hf_fraction"),
        }
        cache[key] = (row["loss"], row, params)
        print(
            f"eval={len(cache):03d} "
            f"omega={row['omega']:.6f} "
            f"loss={row['loss']:.6e} "
            f"kip={row['koopmans_ip_mae']:.3e} "
            f"kea={row['koopmans_ea_mae']:.3e}",
            flush=True,
        )
        return row["loss"]

    result, scipy_history = _bounded_omega_search(
        objective,
        bounds=bounds,
        xatol=float(args.xatol),
        maxiter=int(args.maxiter),
    )

    rows: list[dict[str, Any]] = []
    for scipy_row in scipy_history:
        key = round(float(scipy_row["omega"]), 10)
        loss, metrics_row, _params = cache[key]
        row = {
            "eval": int(scipy_row["eval"]),
            **metrics_row,
        }
        row["loss"] = float(loss)
        rows.append(row)

    best_key = min(cache, key=lambda omega_key: cache[omega_key][0])
    best_loss, best_row, best_params = cache[best_key]
    plot_path = _write_plot(rows, outdir)
    summary = {
        "rsh_preset": str(args.rsh_preset),
        "rsh_omega_source": str(args.rsh_omega_source),
        "reference_xc": str(args.reference_xc),
        "local_xc_spec": local_xc_spec,
        "basis": str(args.basis),
        "grid_level": int(args.grid_level),
        "bounds": [float(bounds[0]), float(bounds[1])],
        "template_omega_bounds": [
            float(template.omega_bounds[0]),
            float(template.omega_bounds[1]),
        ],
        "omega_grid": [float(value) for value in omega_grid],
        "xatol": float(args.xatol),
        "maxiter": int(args.maxiter),
        "scipy_success": bool(result.success),
        "scipy_message": str(result.message),
        "scipy_nfev": int(result.nfev),
        "scipy_x": float(result.x),
        "scipy_fun": float(result.fun),
        "best_loss": float(best_loss),
        "best_omega": float(best_row["omega"]),
        "best_row": best_row,
        "history": rows,
        "koopmans_ip_weight": float(args.koopmans_ip_weight),
        "koopmans_ea_weight": float(args.koopmans_ea_weight),
        "koopmans_lumo_ea_weight": float(args.koopmans_lumo_ea_weight),
        "koopmans_detach_charged_states": bool(args.koopmans_detach_charged_states),
        "koopmans_loss_kind": str(args.koopmans_loss_kind),
        "scf_gradient_mode": str(args.scf_gradient_mode),
        "scf_max_cycle": int(args.scf_max_cycle),
    }
    if plot_path is not None:
        summary["plot"] = str(plot_path)

    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_params_checkpoint(
        outdir / "best_params.msgpack",
        best_params,
        metadata={
            "loss": float(best_loss),
            "omega": float(best_row["omega"]),
            "optimizer": "scipy.optimize.minimize_scalar(bounded)",
        },
    )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
