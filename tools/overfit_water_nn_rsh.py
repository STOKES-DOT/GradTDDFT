from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax

from td_graddft.nn_rsh import (
    AtomCenteredDensityDescriptorConfig,
    get_rsh_functional_preset,
    make_atom_centered_density_rsh_functional,
    make_gnn_rsh_functional,
    make_rsh_template,
    make_self_supervised_rsh_loss,
    rsh_preset_default_params,
)
from td_graddft.data.reference import restricted_reference_from_pyscf
from td_graddft.training import save_params_checkpoint
from td_graddft.training import (
    GroundStateDatum,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    make_ground_state_train_step,
)
from td_graddft.training.targets import (
    _freeze_functional_for_fractional_path,
    _predict_ground_state_total_energy_from_molecule,
    _resolve_training_molecule_and_info_with_mode,
    _resolve_variational_frontier_state_and_info,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-supervised single-water neural RSH overfit driven by Janak/fractional constraints.",
    )
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine", "exponential"),
        default="cosine",
    )
    parser.add_argument(
        "--final-learning-rate-scale",
        type=float,
        default=0.1,
        help="Final LR / initial LR for cosine or exponential schedules.",
    )
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--rsh-preset", default="lc-wpbe")
    parser.add_argument(
        "--free-alpha",
        action="store_true",
        default=False,
        help="Free sr_hf_fraction (alpha) to be trainable in [0, --free-alpha-max].",
    )
    parser.add_argument(
        "--free-alpha-max",
        type=float,
        default=0.20,
        help="Upper bound for sr_hf_fraction when --free-alpha is set.",
    )
    parser.add_argument("--scf-max-cycle", type=int, default=4)
    parser.add_argument(
        "--scf-gradient-mode",
        choices=("unrolled", "implicit_commutator", "impl", "expl"),
        default="impl",
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
    parser.add_argument("--janak-weight", type=float, default=1.0)
    parser.add_argument("--fractional-weight", type=float, default=0.0)
    parser.add_argument("--koopmans-ip-weight", type=float, default=0.0)
    parser.add_argument("--koopmans-ea-weight", type=float, default=0.0)
    parser.add_argument("--koopmans-lumo-ea-weight", type=float, default=0.0)
    parser.add_argument(
        "--koopmans-loss-kind",
        choices=("absolute", "squared"),
        default="absolute",
    )
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
    parser.add_argument(
        "--analyze-fractional-profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute self-consistent fractional-occupation energy curves at every step.",
    )
    parser.add_argument(
        "--fractional-profile-every",
        type=int,
        default=1,
        help=(
            "When fractional profile analysis is enabled, compute it every N "
            "training steps plus the first/final steps."
        ),
    )
    parser.add_argument(
        "--fractional-analysis-delta",
        type=float,
        default=None,
        help="Fractional occupation step for diagnostic curves; defaults to janak-delta.",
    )
    parser.add_argument("--omega-grid", default="0.0,0.3,0.6")
    parser.add_argument("--radial-centers", default="0.6,1.4,2.6")
    parser.add_argument("--radial-width", type=float, default=0.5)
    parser.add_argument("--max-angular", type=int, default=1)
    parser.add_argument(
        "--head",
        choices=("atomwise", "gnn"),
        default="atomwise",
        help="Neural RSH parameter head. atomwise keeps the legacy per-atom MLP.",
    )
    parser.add_argument("--atom-hidden-dims", default="16,16")
    parser.add_argument("--pooled-hidden-dims", default="16")
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--gnn-node-hidden-dims", default="16,16")
    parser.add_argument("--gnn-global-hidden-dims", default="16")
    parser.add_argument("--gnn-num-heads", type=int, default=4)
    parser.add_argument("--gnn-num-layers", type=int, default=1)
    parser.add_argument("--gnn-qkv-features", type=int, default=None)
    parser.add_argument("--gnn-ffn-dim", type=int, default=None)
    parser.add_argument("--gnn-lambda-init", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-best",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--outdir", default="outputs/water_nn_rsh_overfit")
    return parser.parse_args(argv)


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    cleaned = [part.strip() for part in raw.split(",") if part.strip()]
    return tuple(int(part) for part in cleaned)


def _rsh_template_from_args(args: argparse.Namespace):
    template = make_rsh_template(str(args.rsh_preset))
    if bool(args.free_alpha):
        free_alpha_max = float(args.free_alpha_max)
        if not 0.0 < free_alpha_max <= 1.0:
            raise ValueError("--free-alpha-max must be in (0, 1].")
        template = replace(
            template,
            sr_hf_bounds=(0.0, free_alpha_max),
            supports_trainable_sr_hf=True,
        )
    return template


def _build_water_reference(
    *,
    basis: str,
    xc: str,
    grid_level: int,
    omega_grid: tuple[float, ...],
):
    from pyscf import dft, gto

    mol = gto.Mole()
    mol.atom = """
    O  0.000000  0.000000  0.117790
    H  0.000000  0.755453 -0.471161
    H  0.000000 -0.755453 -0.471161
    """
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.spin = 0
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = int(grid_level)
    mf.conv_tol = 1e-10
    mf.kernel()
    return restricted_reference_from_pyscf(
        mf,
        compute_local_hfx_features=True,
        compute_local_hfx_aux=True,
        hfx_omega_values=omega_grid,
    )


def _metric_scalar(metrics: dict[str, jnp.ndarray], key: str) -> float:
    value = metrics[key]
    arr = jnp.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    return float(arr.reshape(-1)[0])


def _make_learning_rate_schedule(args: argparse.Namespace):
    base_lr = float(args.learning_rate)
    final_scale = float(args.final_learning_rate_scale)
    if args.lr_schedule == "constant":
        return optax.constant_schedule(base_lr)
    if args.lr_schedule == "cosine":
        alpha = max(min(final_scale, 1.0), 0.0)
        return optax.cosine_decay_schedule(
            init_value=base_lr,
            decay_steps=max(int(args.steps), 1),
            alpha=alpha,
        )
    if args.lr_schedule == "exponential":
        transition_steps = max(int(args.steps), 1)
        return optax.exponential_decay(
            init_value=base_lr,
            transition_steps=transition_steps,
            decay_rate=max(final_scale, 1e-8),
            staircase=False,
        )
    raise ValueError(f"Unsupported lr_schedule={args.lr_schedule!r}.")


def _fractional_profile_record(
    params: Any,
    functional: Any,
    molecule: Any,
    *,
    training_config: GroundStateTrainingConfig,
    delta: float,
) -> dict[str, Any]:
    clipped_delta = float(jnp.clip(jnp.asarray(delta), 1e-3, 0.49))
    base_molecule, base_info = _resolve_training_molecule_and_info_with_mode(
        params,
        functional,
        molecule,
        training_config,
    )
    frozen_functional, frozen_params = _freeze_functional_for_fractional_path(
        params,
        functional,
        base_molecule,
    )
    mol_m2, info_m2 = _resolve_variational_frontier_state_and_info(
        frozen_params,
        frozen_functional,
        base_molecule,
        homo_delta=-2.0 * clipped_delta,
        training_config=training_config,
    )
    mol_m1, info_m1 = _resolve_variational_frontier_state_and_info(
        frozen_params,
        frozen_functional,
        base_molecule,
        homo_delta=-clipped_delta,
        training_config=training_config,
    )
    mol_p1, info_p1 = _resolve_variational_frontier_state_and_info(
        frozen_params,
        frozen_functional,
        base_molecule,
        lumo_delta=clipped_delta,
        training_config=training_config,
    )
    mol_p2, info_p2 = _resolve_variational_frontier_state_and_info(
        frozen_params,
        frozen_functional,
        base_molecule,
        lumo_delta=2.0 * clipped_delta,
        training_config=training_config,
    )
    e_m2 = float(_predict_ground_state_total_energy_from_molecule(frozen_params, frozen_functional, mol_m2))
    e_m1 = float(_predict_ground_state_total_energy_from_molecule(frozen_params, frozen_functional, mol_m1))
    e_0 = float(_predict_ground_state_total_energy_from_molecule(frozen_params, frozen_functional, base_molecule))
    e_p1 = float(_predict_ground_state_total_energy_from_molecule(frozen_params, frozen_functional, mol_p1))
    e_p2 = float(_predict_ground_state_total_energy_from_molecule(frozen_params, frozen_functional, mol_p2))

    remove_line_mid = 0.5 * (e_0 + e_m2)
    add_line_mid = 0.5 * (e_0 + e_p2)
    remove_deviation = e_m1 - remove_line_mid
    add_deviation = e_p1 - add_line_mid
    charges = [-2.0 * clipped_delta, -clipped_delta, 0.0, clipped_delta, 2.0 * clipped_delta]
    energies = [e_m2, e_m1, e_0, e_p1, e_p2]
    relative_energies = [energy - e_0 for energy in energies]

    def _selected_rms(info: Any) -> float | None:
        if info is None or getattr(info, "mode", None) != "self_consistent":
            return None
        return float(jnp.asarray(getattr(info, "selected_rms_density", 0.0)))

    return {
        "delta": clipped_delta,
        "charges": charges,
        "energies_hartree": energies,
        "relative_energies_hartree": relative_energies,
        "remove_curvature_hartree": float(e_0 - 2.0 * e_m1 + e_m2),
        "add_curvature_hartree": float(e_p2 - 2.0 * e_p1 + e_0),
        "remove_line_mid_deviation_hartree": float(remove_deviation),
        "add_line_mid_deviation_hartree": float(add_deviation),
        "base_selected_rms_density": _selected_rms(base_info),
        "minus_selected_rms_density": [_selected_rms(info_m2), _selected_rms(info_m1)],
        "plus_selected_rms_density": [_selected_rms(info_p1), _selected_rms(info_p2)],
    }


def _write_fractional_profile_plot(
    history: list[dict[str, Any]],
    outdir: Path,
) -> Path | None:
    profiles = [record.get("fractional_profile") for record in history if record.get("fractional_profile") is not None]
    if not profiles:
        return None
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    cmap = plt.get_cmap("viridis")
    denom = max(len(history) - 1, 1)
    for idx, record in enumerate(history):
        profile = record.get("fractional_profile")
        if profile is None:
            continue
        charges = jnp.asarray(profile["charges"], dtype=jnp.float32)
        rel_mhartree = 1000.0 * jnp.asarray(profile["relative_energies_hartree"], dtype=jnp.float32)
        color = cmap(idx / denom)
        linewidth = 2.2 if idx in (0, len(history) - 1) else 1.1
        alpha = 1.0 if idx in (0, len(history) - 1) else 0.35
        label = None
        if idx == 0:
            label = f"step {int(record['step'])}"
        elif idx == len(history) - 1:
            label = f"step {int(record['step'])}"
        ax.plot(charges, rel_mhartree, "-o", color=color, linewidth=linewidth, alpha=alpha, label=label)

    ax.axvline(0.0, color="0.8", linewidth=0.8)
    ax.set_xlabel("Fractional charge displacement")
    ax.set_ylabel(r"$E(N+\Delta q)-E(N)$ (mHa)")
    ax.set_title("Water fractional-occupation energy curves during training")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    plot_path = outdir / "fractional_profiles.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return plot_path


_GRAD_MODE_MAP = {"implicit_commutator": "impl", "unrolled": "expl", "impl": "impl", "expl": "expl"}


def main() -> None:
    args = parse_args()
    args.scf_gradient_mode = _GRAD_MODE_MAP[str(args.scf_gradient_mode)]
    omega_grid = _parse_float_tuple(args.omega_grid)
    descriptor_config = AtomCenteredDensityDescriptorConfig(
        radial_centers=_parse_float_tuple(args.radial_centers),
        radial_width=float(args.radial_width),
        max_angular=int(args.max_angular),
    )
    atom_hidden_dims = _parse_int_tuple(args.atom_hidden_dims)
    pooled_hidden_dims = _parse_int_tuple(args.pooled_hidden_dims)
    gnn_node_hidden_dims = _parse_int_tuple(args.gnn_node_hidden_dims)
    gnn_global_hidden_dims = _parse_int_tuple(args.gnn_global_hidden_dims)

    molecule = _build_water_reference(
        basis=args.basis,
        xc=args.xc,
        grid_level=args.grid_level,
        omega_grid=omega_grid,
    )
    preset = get_rsh_functional_preset(str(args.rsh_preset))
    template = _rsh_template_from_args(args)
    local_xc_spec = preset.jax_local_xc_spec or str(args.xc)
    local_term_specs = tuple(preset.local_term_specs)
    if args.head == "gnn":
        functional = make_gnn_rsh_functional(
            local_xc_spec=local_xc_spec,
            local_term_specs=local_term_specs,
            descriptor_config=descriptor_config,
            node_hidden_dims=gnn_node_hidden_dims,
            global_hidden_dims=gnn_global_hidden_dims,
            num_heads=int(args.gnn_num_heads),
            num_layers=int(args.gnn_num_layers),
            qkv_features=args.gnn_qkv_features,
            ffn_dim=args.gnn_ffn_dim,
            lambda_init=float(args.gnn_lambda_init),
            template=template,
            fallback_omega_values=omega_grid,
        )
    else:
        functional = make_atom_centered_density_rsh_functional(
            local_xc_spec=local_xc_spec,
            local_term_specs=local_term_specs,
            descriptor_config=descriptor_config,
            atom_hidden_dims=atom_hidden_dims,
            pooled_hidden_dims=pooled_hidden_dims,
            embedding_dim=int(args.embedding_dim),
            template=template,
            fallback_omega_values=omega_grid,
        )
    loss_fn = make_self_supervised_rsh_loss(
        functional,
        training_config=GroundStateTrainingConfig(
            mode="self_consistent",
            scf_gradient_mode=str(args.scf_gradient_mode),
            scf_max_cycle=int(args.scf_max_cycle),
            janak_frontier_mode=str(args.janak_mode),
            janak_frontier_delta=float(args.janak_delta),
            scf_require_convergence=False,
        ),
        janak_weight=float(args.janak_weight),
        fractional_weight=float(args.fractional_weight),
        koopmans_ip_weight=float(args.koopmans_ip_weight),
        koopmans_ea_weight=float(args.koopmans_ea_weight),
        koopmans_lumo_ea_weight=float(args.koopmans_lumo_ea_weight),
        koopmans_loss_kind=str(args.koopmans_loss_kind),
        koopmans_detach_charged_states=bool(args.koopmans_detach_charged_states),
        koopmans_differentiate_charged_orbitals=bool(
            args.koopmans_differentiate_charged_orbitals
        ),
        prior_weight=float(args.prior_weight),
    )
    profile_training_config = GroundStateTrainingConfig(
        mode="self_consistent",
        scf_gradient_mode=str(args.scf_gradient_mode),
        scf_max_cycle=int(args.scf_max_cycle),
        janak_frontier_mode=str(args.janak_mode),
        janak_frontier_delta=float(args.janak_delta),
        scf_require_convergence=False,
    )
    fractional_analysis_delta = (
        float(args.janak_delta)
        if args.fractional_analysis_delta is None
        else float(args.fractional_analysis_delta)
    )
    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.asarray(molecule.mf_energy),
    )
    lr_schedule = _make_learning_rate_schedule(args)
    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(int(args.seed)),
        molecule,
        optax.adam(lr_schedule),
    )
    train_step = make_ground_state_train_step(functional, loss_fn=loss_fn)

    initial_loss, initial_metrics = loss_fn(state.params, functional, datum)
    history: list[dict[str, float]] = []
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
    best_metric_key = (
        "loss"
        if float(args.janak_weight) == 0.0
        else "janak_frontier_mae"
    )
    for step in range(int(args.steps)):
        params_before = state.params
        state, metrics = train_step(state, datum)
        learning_rate = float(lr_schedule(step))
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
            "learning_rate": learning_rate,
        }
        profile_every = max(int(args.fractional_profile_every), 1)
        should_analyze_profile = (
            bool(args.analyze_fractional_profile)
            and (
                step == 0
                or step + 1 == int(args.steps)
                or (step + 1) % profile_every == 0
            )
        )
        if should_analyze_profile:
            record["fractional_profile"] = _fractional_profile_record(
                params_before,
                functional,
                molecule,
                training_config=profile_training_config,
                delta=fractional_analysis_delta,
            )
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
            f"sr={record['sr_hf_fraction']:.4f} "
            f"lr={record['lr_hf_fraction']:.4f} "
            f"omega={record['omega']:.4f} "
            f"lrn={record['learning_rate']:.3e}",
            flush=True,
        )

    final_loss, final_metrics = loss_fn(state.params, functional, datum)
    final_resolved = functional.resolve_parameters(state.params, molecule)
    summary = {
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
        "best_janak_frontier_mae": float(best_record["janak_frontier_mae"]),
        "best_koopmans_ip_mae": float(best_record["koopmans_ip_mae"]),
        "best_koopmans_ea_mae": float(best_record["koopmans_ea_mae"]),
        "best_koopmans_lumo_ea_mae": float(best_record["koopmans_lumo_ea_mae"]),
        "best_sr_hf_fraction": float(best_record["sr_hf_fraction"]),
        "best_lr_hf_fraction": float(best_record["lr_hf_fraction"]),
        "best_omega": float(best_record["omega"]),
        "best_learning_rate": float(best_record["learning_rate"]),
        "best_metric": best_metric_key,
        "steps": int(args.steps),
        "learning_rate": float(args.learning_rate),
        "head": str(args.head),
        "free_alpha": bool(args.free_alpha),
        "free_alpha_max": float(args.free_alpha_max),
        "lr_schedule": str(args.lr_schedule),
        "final_learning_rate_scale": float(args.final_learning_rate_scale),
        "scf_gradient_mode": str(args.scf_gradient_mode),
        "koopmans_detach_charged_states": bool(args.koopmans_detach_charged_states),
        "koopmans_differentiate_charged_orbitals": bool(
            args.koopmans_differentiate_charged_orbitals
        ),
        "koopmans_loss_kind": str(args.koopmans_loss_kind),
        "analyze_fractional_profile": bool(args.analyze_fractional_profile),
        "fractional_profile_every": int(args.fractional_profile_every),
        "history": history,
    }
    if args.head == "gnn":
        summary.update(
            {
                "gnn_node_hidden_dims": list(gnn_node_hidden_dims),
                "gnn_global_hidden_dims": list(gnn_global_hidden_dims),
                "gnn_num_heads": int(args.gnn_num_heads),
                "gnn_num_layers": int(args.gnn_num_layers),
                "gnn_qkv_features": args.gnn_qkv_features,
                "gnn_ffn_dim": args.gnn_ffn_dim,
                "gnn_lambda_init": float(args.gnn_lambda_init),
            }
        )
    else:
        summary.update(
            {
                "atom_hidden_dims": list(atom_hidden_dims),
                "pooled_hidden_dims": list(pooled_hidden_dims),
                "embedding_dim": int(args.embedding_dim),
            }
        )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_path = _write_fractional_profile_plot(history, outdir)
    if plot_path is not None:
        summary["fractional_profile_plot"] = str(plot_path)
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    if bool(args.save_best):
        save_params_checkpoint(
            outdir / "best_params.msgpack",
            best_params,
            metadata={
                "best_step": int(best_record["step"]),
                "best_janak_frontier_mae": float(best_record["janak_frontier_mae"]),
                "best_loss": float(best_record["loss"]),
                "best_metric": best_metric_key,
                "best_sr_hf_fraction": float(best_record["sr_hf_fraction"]),
                "best_lr_hf_fraction": float(best_record["lr_hf_fraction"]),
                "best_omega": float(best_record["omega"]),
                "head": str(args.head),
                "scf_gradient_mode": str(args.scf_gradient_mode),
                "lr_schedule": str(args.lr_schedule),
            },
        )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
