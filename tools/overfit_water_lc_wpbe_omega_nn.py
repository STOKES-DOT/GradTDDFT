from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax

from td_graddft.nn_rsh import (
    AtomCenteredDensityDescriptorConfig,
    ResolvedRSHParameters,
    get_rsh_functional_preset,
    make_atom_centered_density_rsh_functional,
    make_rsh_template,
    make_self_supervised_rsh_loss,
    rsh_preset_default_params,
)
from td_graddft.reference_legacy import restricted_reference_from_pyscf
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
            "True neural overfit of LC-wPBE omega on one water molecule using "
            "self-supervised Koopmans IP/EA constraints."
        ),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument(
        "--train-scope",
        choices=("output_bias_omega", "output_bias", "output_head", "all"),
        default="output_bias_omega",
        help=(
            "output_bias_omega updates only the NN output bias component that "
            "controls omega. This is the stable single-molecule default."
        ),
    )
    parser.add_argument(
        "--gradient-source",
        choices=("finite_difference", "autodiff"),
        default="autodiff",
        help=(
            "autodiff uses the full differentiable SCF/RSH path. finite_difference "
            "is retained as a debugging fallback for the omega output bias."
        ),
    )
    parser.add_argument("--finite-diff-step", type=float, default=0.25)
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
        help=(
            "Optional explicit lower,upper trainable omega interval. "
            "Use this to decouple water overfitting from OPTXC/TADF-style bounds."
        ),
    )
    parser.add_argument(
        "--initial-omega",
        type=float,
        default=None,
        help=(
            "Optional omega initialization override before NN overfitting. "
            "This changes only the initial NN output bias, not the loss."
        ),
    )
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
            "Pair with --gradient-source finite_difference for PySCF-like omega tuning."
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
    parser.add_argument("--line-search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--line-search-shrink", type=float, default=0.5)
    parser.add_argument("--line-search-attempts", type=int, default=6)
    parser.add_argument("--accept-tolerance", type=float, default=1e-10)
    parser.add_argument("--radial-centers", default="0.6,1.4,2.6")
    parser.add_argument("--radial-width", type=float, default=0.5)
    parser.add_argument("--max-angular", type=int, default=1)
    parser.add_argument("--atom-hidden-dims", default="16,16")
    parser.add_argument("--pooled-hidden-dims", default="16")
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--outdir",
        default="outputs/water_lc_wpbe_omega_nn_overfit",
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


def _tree_l2_norm(tree: Any) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    total = jnp.asarray(0.0, dtype=jnp.float32)
    for leaf in leaves:
        arr = jnp.nan_to_num(jnp.asarray(leaf), nan=0.0, posinf=0.0, neginf=0.0)
        total = total + jnp.sum(jnp.square(arr.astype(jnp.float32)))
    return jnp.sqrt(total)


def _zero_like_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), tree)


def _filter_nn_grads(grads: Any, scope: str) -> Any:
    """Keep only the NN parameters selected for the single-molecule overfit."""

    if scope == "all":
        return grads

    filtered = _zero_like_tree(grads)
    try:
        output_grads = grads["params"]["output"]
        filtered_output = filtered["params"]["output"]
    except Exception as exc:
        raise ValueError(
            "Expected Flax params tree with grads['params']['output'] for NN RSH head."
        ) from exc

    filtered = copy.deepcopy(filtered)
    if scope == "output_head":
        filtered["params"]["output"] = output_grads
        return filtered

    if scope in {"output_bias", "output_bias_omega"}:
        if "bias" not in output_grads:
            raise ValueError("Expected output head to expose a bias parameter.")
        bias_grad = jnp.asarray(output_grads["bias"])
        if scope == "output_bias":
            filtered_output["bias"] = bias_grad
        else:
            if bias_grad.shape != (3,):
                raise ValueError(
                    "output_bias_omega expects a three-component RSH output bias, "
                    f"got shape {bias_grad.shape}."
                )
            filtered_output["bias"] = jnp.zeros_like(bias_grad).at[2].set(bias_grad[2])
        filtered["params"]["output"] = filtered_output
        return filtered

    raise ValueError(f"Unsupported train scope {scope!r}.")


def _with_output_bias_omega_delta(params: Any, delta: float | jnp.ndarray) -> Any:
    updated = copy.deepcopy(params)
    try:
        bias = jnp.asarray(updated["params"]["output"]["bias"])
    except Exception as exc:
        raise ValueError(
            "Expected params['params']['output']['bias'] for omega-bias updates."
        ) from exc
    if bias.shape != (3,):
        raise ValueError(
            "omega-bias update expects a three-component RSH output bias, "
            f"got shape {bias.shape}."
        )
    updated["params"]["output"]["bias"] = bias.at[2].add(
        jnp.asarray(delta, dtype=bias.dtype)
    )
    return updated


def _omega_bias_only_grads_like(params: Any, omega_bias_grad: float | jnp.ndarray) -> Any:
    grads = _zero_like_tree(params)
    try:
        bias = jnp.asarray(grads["params"]["output"]["bias"])
    except Exception as exc:
        raise ValueError(
            "Expected params['params']['output']['bias'] for omega-bias gradients."
        ) from exc
    grads = copy.deepcopy(grads)
    grads["params"]["output"]["bias"] = bias.at[2].set(
        jnp.asarray(omega_bias_grad, dtype=bias.dtype)
    )
    return grads


def _clean_grads(grads: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda x: jnp.nan_to_num(
            jnp.asarray(x),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        grads,
    )


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


def _record_from_state(
    *,
    step: int,
    params: Any,
    functional: Any,
    molecule: Any,
    loss: jnp.ndarray,
    metrics: dict[str, jnp.ndarray],
    grad_norm: float | None,
    accepted_scale: float | None,
    accepted: bool | None,
) -> dict[str, Any]:
    raw = jnp.asarray(functional._raw_outputs(params, molecule))
    sr = _metric_scalar(metrics, "sr_hf_fraction")
    lr = _metric_scalar(metrics, "lr_hf_fraction")
    omega = _metric_scalar(metrics, "omega")
    return {
        "step": int(step),
        "loss": float(loss),
        "koopmans_ip_mae": _metric_scalar(metrics, "koopmans_ip_mae"),
        "koopmans_ea_mae": _metric_scalar(metrics, "koopmans_ea_mae"),
        "koopmans_lumo_ea_mae": _metric_scalar(metrics, "koopmans_lumo_ea_mae"),
        "koopmans_ip_residual": _metric_scalar(metrics, "koopmans_ip_residual"),
        "koopmans_ea_residual": _metric_scalar(metrics, "koopmans_ea_residual"),
        "koopmans_lumo_ea_residual": _metric_scalar(
            metrics,
            "koopmans_lumo_ea_residual",
        ),
        "fractional_linearity_raw": _metric_scalar(metrics, "fractional_linearity_raw"),
        "janak_frontier_mae": _metric_scalar(metrics, "janak_frontier_mae"),
        "neutral_energy": _metric_scalar(metrics, "koopmans_neutral_energy"),
        "cation_energy": _metric_scalar(metrics, "koopmans_cation_energy"),
        "anion_energy": _metric_scalar(metrics, "koopmans_anion_energy"),
        "sr_hf_fraction": sr,
        "lr_hf_fraction": lr,
        "paper_alpha": sr,
        "paper_beta": lr - sr,
        "omega": omega,
        "raw_parameters": [float(x) for x in raw.reshape(-1)],
        "grad_norm": grad_norm,
        "accepted_scale": accepted_scale,
        "accepted": accepted,
    }


def _write_training_plot(history: list[dict[str, Any]], outdir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    steps = [int(row["step"]) for row in history]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
    axes[0].plot(steps, [float(row["loss"]) for row in history], "-o", label="loss")
    axes[0].plot(
        steps,
        [float(row["koopmans_ip_mae"]) for row in history],
        "--",
        label="|HOMO(N)+IP|",
    )
    axes[0].plot(
        steps,
        [float(row["koopmans_ea_mae"]) for row in history],
        "--",
        label="|HOMO(N+1)+EA|",
    )
    axes[0].set_ylabel("Hartree")
    axes[0].grid(alpha=0.25, linewidth=0.6)
    axes[0].legend(frameon=False)

    axes[1].plot(steps, [float(row["omega"]) for row in history], "-o", label="omega")
    axes[1].set_xlabel("NN training epoch")
    axes[1].set_ylabel("omega")
    axes[1].grid(alpha=0.25, linewidth=0.6)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    path = outdir / "omega_nn_overfit.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    descriptor_config = AtomCenteredDensityDescriptorConfig(
        radial_centers=_parse_float_tuple(args.radial_centers),
        radial_width=float(args.radial_width),
        max_angular=int(args.max_angular),
    )
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
        bounds = tuple(float(value) for value in template.omega_bounds)
    omega_grid = _omega_grid_covering_bounds(_parse_float_tuple(args.omega_grid), bounds)
    molecule = _build_water_reference(
        basis=str(args.basis),
        xc=str(args.reference_xc),
        grid_level=int(args.grid_level),
        omega_grid=omega_grid,
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
    params = functional.init_from_molecule(jax.random.PRNGKey(int(args.seed)), molecule)
    initial_resolved = rsh_preset_default_params(
        str(args.rsh_preset),
        omega_source=str(args.rsh_omega_source),
    )
    if args.initial_omega is not None:
        initial_omega = float(args.initial_omega)
        if not (float(bounds[0]) <= initial_omega <= float(bounds[1])):
            raise ValueError(
                "--initial-omega must lie inside the active omega bounds "
                f"{bounds}, got {initial_omega}."
            )
        initial_resolved = ResolvedRSHParameters(
            sr_hf_fraction=initial_resolved.sr_hf_fraction,
            lr_hf_fraction=initial_resolved.lr_hf_fraction,
            omega=jnp.asarray(initial_omega),
        )
    params = functional.params_with_resolved(
        params,
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

    def objective(train_params: Any) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        return loss_fn(train_params, functional, datum)

    value_and_grad = jax.value_and_grad(objective, has_aux=True)
    tx = optax.adam(float(args.learning_rate))
    opt_state = tx.init(params)

    current_loss, current_metrics = objective(params)
    history = [
        _record_from_state(
            step=0,
            params=params,
            functional=functional,
            molecule=molecule,
            loss=current_loss,
            metrics=current_metrics,
            grad_norm=None,
            accepted_scale=None,
            accepted=None,
        )
    ]
    best_params = params
    best_record = history[0]

    for epoch in range(1, max(int(args.epochs), 0) + 1):
        baseline_loss = current_loss
        baseline_metrics = current_metrics
        if str(args.gradient_source) == "autodiff":
            (loss, metrics), grads = value_and_grad(params)
            filtered_grads = _clean_grads(
                _filter_nn_grads(grads, str(args.train_scope))
            )
        else:
            if str(args.train_scope) != "output_bias_omega":
                raise ValueError(
                    "finite_difference gradient source currently supports only "
                    "--train-scope output_bias_omega."
                )
            fd_step = max(float(args.finite_diff_step), 1e-8)
            plus_loss, _ = objective(_with_output_bias_omega_delta(params, fd_step))
            minus_loss, _ = objective(_with_output_bias_omega_delta(params, -fd_step))
            omega_bias_grad = (plus_loss - minus_loss) / (2.0 * fd_step)
            loss = baseline_loss
            metrics = baseline_metrics
            filtered_grads = _clean_grads(
                _omega_bias_only_grads_like(params, omega_bias_grad)
            )
        grad_norm = float(_tree_l2_norm(filtered_grads))
        updates, proposed_opt_state = tx.update(filtered_grads, opt_state, params)

        shrink = float(args.line_search_shrink)
        attempts = max(int(args.line_search_attempts), 1)
        scales = (
            [shrink**idx for idx in range(attempts)]
            if bool(args.line_search)
            else [1.0]
        )
        accepted = False
        accepted_scale = 0.0
        accepted_params = params
        accepted_loss = loss
        accepted_metrics = metrics
        tolerance = float(args.accept_tolerance)
        for scale in scales:
            scaled_updates = jax.tree_util.tree_map(lambda x: scale * x, updates)
            candidate_params = optax.apply_updates(params, scaled_updates)
            candidate_loss, candidate_metrics = objective(candidate_params)
            if jnp.isfinite(candidate_loss) and (
                not bool(args.line_search)
                or float(candidate_loss) <= float(baseline_loss) + tolerance
            ):
                accepted = True
                accepted_scale = float(scale)
                accepted_params = candidate_params
                accepted_loss = candidate_loss
                accepted_metrics = candidate_metrics
                break

        if accepted:
            params = accepted_params
            opt_state = proposed_opt_state
            current_loss = accepted_loss
            current_metrics = accepted_metrics
        else:
            current_loss = baseline_loss
            current_metrics = baseline_metrics

        record = _record_from_state(
            step=epoch,
            params=params,
            functional=functional,
            molecule=molecule,
            loss=current_loss,
            metrics=current_metrics,
            grad_norm=grad_norm,
            accepted_scale=accepted_scale,
            accepted=accepted,
        )
        history.append(record)
        if float(record["loss"]) < float(best_record["loss"]):
            best_record = record
            best_params = params

        print(
            f"epoch={epoch:03d} "
            f"loss={record['loss']:.6e} "
            f"kip={record['koopmans_ip_mae']:.3e} "
            f"kea={record['koopmans_ea_mae']:.3e} "
            f"omega={record['omega']:.4f} "
            f"grad={grad_norm:.3e} "
            f"scale={accepted_scale:.3g} "
            f"accepted={accepted}",
            flush=True,
        )

    plot_path = _write_training_plot(history, outdir)
    final_resolved = functional.resolve_parameters(params, molecule)
    best_resolved = functional.resolve_parameters(best_params, molecule)
    summary = {
        "rsh_preset": str(args.rsh_preset),
        "rsh_omega_source": str(args.rsh_omega_source),
        "reference_xc": str(args.reference_xc),
        "local_xc_spec": local_xc_spec,
        "template_omega_bounds": [
            float(template.omega_bounds[0]),
            float(template.omega_bounds[1]),
        ],
        "omega_grid": [float(value) for value in omega_grid],
        "train_scope": str(args.train_scope),
        "gradient_source": str(args.gradient_source),
        "finite_diff_step": float(args.finite_diff_step),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "initial_loss": float(history[0]["loss"]),
        "final_loss": float(history[-1]["loss"]),
        "best_loss": float(best_record["loss"]),
        "best_step": int(best_record["step"]),
        "initial_omega_override": (
            None if args.initial_omega is None else float(args.initial_omega)
        ),
        "initial_omega": float(history[0]["omega"]),
        "final_omega": float(final_resolved.omega),
        "best_omega": float(best_resolved.omega),
        "final_sr_hf_fraction": float(final_resolved.sr_hf_fraction),
        "final_lr_hf_fraction": float(final_resolved.lr_hf_fraction),
        "best_sr_hf_fraction": float(best_resolved.sr_hf_fraction),
        "best_lr_hf_fraction": float(best_resolved.lr_hf_fraction),
        "koopmans_ip_weight": float(args.koopmans_ip_weight),
        "koopmans_ea_weight": float(args.koopmans_ea_weight),
        "koopmans_lumo_ea_weight": float(args.koopmans_lumo_ea_weight),
        "koopmans_detach_charged_states": bool(args.koopmans_detach_charged_states),
        "koopmans_loss_kind": str(args.koopmans_loss_kind),
        "scf_gradient_mode": str(args.scf_gradient_mode),
        "scf_max_cycle": int(args.scf_max_cycle),
        "history": history,
    }
    if plot_path is not None:
        summary["plot"] = str(plot_path)

    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_params_checkpoint(
        outdir / "final_params.msgpack",
        params,
        metadata={
            "step": int(history[-1]["step"]),
            "loss": float(history[-1]["loss"]),
            "omega": float(final_resolved.omega),
            "train_scope": str(args.train_scope),
            "gradient_source": str(args.gradient_source),
        },
    )
    save_params_checkpoint(
        outdir / "best_params.msgpack",
        best_params,
        metadata={
            "step": int(best_record["step"]),
            "loss": float(best_record["loss"]),
            "omega": float(best_resolved.omega),
            "train_scope": str(args.train_scope),
            "gradient_source": str(args.gradient_source),
        },
    )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
