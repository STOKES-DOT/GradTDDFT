from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax

from td_graddft.nn_rsh import (
    AtomCenteredDensityDescriptorConfig,
    get_rsh_functional_preset,
    make_atom_centered_density_rsh_functional,
    make_self_supervised_rsh_loss,
    make_rsh_template,
    rsh_preset_default_params,
)
from td_graddft.data.reference import restricted_reference_from_pyscf
from td_graddft.training import (
    GroundStateDatum,
    GroundStateTrainingConfig,
    save_params_checkpoint,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Output-head RSH endpoint tuning on water using Koopmans IP/EA constraints."
        ),
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--optimizer",
        choices=("coordinate", "sgd", "adam"),
        default="coordinate",
    )
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument("--xc", default="pbe")
    parser.add_argument(
        "--rsh-preset",
        default="",
        help=(
            "Optional named RSH preset, e.g. lc-wpbe or wb97x-d. The preset "
            "sets the RSH parameter bounds/defaults. For presets with a strict "
            "JAX local backend, the default --xc=pbe is used only for the "
            "PySCF reference guess while the trainable functional uses the "
            "preset local XC."
        ),
    )
    parser.add_argument(
        "--rsh-omega-source",
        choices=("canonical", "optxc"),
        default="optxc",
        help=(
            "For named presets, canonical keeps the literature/PySCF default "
            "omega and broad safety bounds. optxc uses the OPTXC training-data "
            "omega range and center for molecule-specific tuning."
        ),
    )
    parser.add_argument("--scf-max-cycle", type=int, default=4)
    parser.add_argument(
        "--scf-gradient-mode",
        choices=("unrolled", "implicit_commutator"),
        default="implicit_commutator",
    )
    parser.add_argument("--koopmans-ip-weight", type=float, default=1.0)
    parser.add_argument(
        "--koopmans-ea-weight",
        type=float,
        default=0.0,
        help="Weight for the optional anion HOMO relation eps_HOMO(N+1)=-EA.",
    )
    parser.add_argument(
        "--koopmans-lumo-ea-weight",
        type=float,
        default=1.0,
        help="Weight for the neutral LUMO relation eps_LUMO(N)=-EA.",
    )
    parser.add_argument(
        "--koopmans-loss-kind",
        choices=("absolute", "squared"),
        default="absolute",
        help=(
            "absolute minimizes weighted absolute residuals. squared minimizes "
            "weighted squared residuals, matching optDFTw-style J^2."
        ),
    )
    parser.add_argument("--janak-weight", type=float, default=0.0)
    parser.add_argument("--fractional-weight", type=float, default=1.0)
    parser.add_argument("--long-range-correction-weight", type=float, default=1.0)
    parser.add_argument(
        "--strategy",
        choices=("single", "2dt"),
        default="2dt",
        help=(
            "single uses the supplied weights for all steps. 2dt first optimizes "
            "Koopmans(N,N+1)+long-range-correction, then adds fractional curvature "
            "to select along the Koopmans valley."
        ),
    )
    parser.add_argument(
        "--stage-a-steps",
        type=int,
        default=None,
        help="Number of 2dt Koopmans+LC steps. Defaults to floor(steps/2).",
    )
    parser.add_argument(
        "--stage-b-steps",
        type=int,
        default=None,
        help="Number of 2dt curvature-selection steps. Defaults to remaining steps.",
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
        default="fixed_orbital_ad",
    )
    parser.add_argument("--janak-delta", type=float, default=0.1)
    parser.add_argument("--prior-weight", type=float, default=1e-3)
    parser.add_argument("--line-search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--line-search-shrink", type=float, default=0.5)
    parser.add_argument("--line-search-attempts", type=int, default=8)
    parser.add_argument("--accept-tolerance", type=float, default=1e-10)
    parser.add_argument("--coordinate-step-size", type=float, default=0.02)
    parser.add_argument("--coordinate-shrink", type=float, default=0.5)
    parser.add_argument("--coordinate-min-step-size", type=float, default=1e-5)
    parser.add_argument("--omega-grid", default="0.0,0.3,0.6")
    parser.add_argument("--radial-centers", default="0.6,1.4,2.6")
    parser.add_argument("--radial-width", type=float, default=0.5)
    parser.add_argument("--max-angular", type=int, default=1)
    parser.add_argument("--atom-hidden-dims", default="16,16")
    parser.add_argument("--pooled-hidden-dims", default="16")
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--preserve-network",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Preserve descriptor/hidden layers and output kernel, and tune only the "
            "output bias required to realize the target raw RSH parameters. Disable "
            "to reset the output head to a molecule-independent three-parameter model."
        ),
    )
    parser.add_argument("--outdir", default="outputs/water_rsh_endpoint_koopmans_raw")
    return parser.parse_args(argv)


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    cleaned = [part.strip() for part in raw.split(",") if part.strip()]
    return tuple(int(part) for part in cleaned)


def _normalized_rsh_preset_name(args: argparse.Namespace) -> str | None:
    raw = str(getattr(args, "rsh_preset", "") or "").strip()
    return raw or None


def _rsh_omega_source_from_args(args: argparse.Namespace) -> str:
    return str(getattr(args, "rsh_omega_source", "canonical") or "canonical")


def _rsh_template_from_args(args: argparse.Namespace):
    preset_name = _normalized_rsh_preset_name(args)
    if preset_name is None:
        return None
    return make_rsh_template(
        preset_name,
        omega_source=_rsh_omega_source_from_args(args),
    )


def _initial_resolved_from_args(args: argparse.Namespace):
    preset_name = _normalized_rsh_preset_name(args)
    if preset_name is None:
        return None
    return rsh_preset_default_params(
        preset_name,
        omega_source=_rsh_omega_source_from_args(args),
    )


def _local_xc_spec_from_args(args: argparse.Namespace) -> str:
    xc = str(getattr(args, "xc", "pbe") or "pbe")
    preset_name = _normalized_rsh_preset_name(args)
    if preset_name is None:
        return xc
    preset = get_rsh_functional_preset(preset_name)
    if xc.strip().lower() == "pbe" and preset.jax_local_xc_spec:
        return preset.jax_local_xc_spec
    return xc


def _coordinate_active_raw_dims_from_template(template: Any | None) -> tuple[int, ...]:
    if template is None:
        return (0, 1, 2)

    active_dims: list[int] = []
    if bool(template.supports_trainable_sr_hf) and (
        float(template.sr_hf_bounds[0]) < float(template.sr_hf_bounds[1])
    ):
        active_dims.append(0)
    if bool(template.supports_trainable_lr_hf) and (
        float(template.lr_hf_bounds[0]) < float(template.lr_hf_bounds[1])
    ):
        active_dims.append(1)
    if bool(template.supports_trainable_omega) and (
        float(template.omega_bounds[0]) < float(template.omega_bounds[1])
    ):
        active_dims.append(2)
    return tuple(active_dims)


@dataclass(frozen=True)
class StageSpec:
    name: str
    steps: int
    janak_weight: float
    fractional_weight: float
    koopmans_ip_weight: float
    koopmans_ea_weight: float
    koopmans_lumo_ea_weight: float
    koopmans_loss_kind: str
    long_range_correction_weight: float


def _build_stage_specs(args: argparse.Namespace) -> list[StageSpec]:
    if str(args.strategy) == "single":
        return [
            StageSpec(
                name="single",
                steps=max(int(args.steps), 0),
                janak_weight=float(args.janak_weight),
                fractional_weight=float(args.fractional_weight),
                koopmans_ip_weight=float(args.koopmans_ip_weight),
                koopmans_ea_weight=float(args.koopmans_ea_weight),
                koopmans_lumo_ea_weight=float(args.koopmans_lumo_ea_weight),
                koopmans_loss_kind=str(args.koopmans_loss_kind),
                long_range_correction_weight=float(args.long_range_correction_weight),
            )
        ]

    total_steps = max(int(args.steps), 0)
    stage_a_steps = getattr(args, "stage_a_steps", None)
    stage_b_steps = getattr(args, "stage_b_steps", None)
    if stage_a_steps is None and stage_b_steps is None:
        stage_a = total_steps // 2
        stage_b = total_steps - stage_a
    elif stage_a_steps is None:
        stage_b = max(int(stage_b_steps), 0)
        stage_a = max(total_steps - stage_b, 0)
    elif stage_b_steps is None:
        stage_a = max(int(stage_a_steps), 0)
        stage_b = max(total_steps - stage_a, 0)
    else:
        stage_a = max(int(stage_a_steps), 0)
        stage_b = max(int(stage_b_steps), 0)

    return [
        StageSpec(
            name="koopmans_lc",
            steps=stage_a,
            janak_weight=float(args.janak_weight),
            fractional_weight=0.0,
            koopmans_ip_weight=float(args.koopmans_ip_weight),
            koopmans_ea_weight=float(args.koopmans_ea_weight),
            koopmans_lumo_ea_weight=float(args.koopmans_lumo_ea_weight),
            koopmans_loss_kind=str(args.koopmans_loss_kind),
            long_range_correction_weight=float(args.long_range_correction_weight),
        ),
        StageSpec(
            name="curvature_selection",
            steps=stage_b,
            janak_weight=float(args.janak_weight),
            fractional_weight=float(args.fractional_weight),
            koopmans_ip_weight=float(args.koopmans_ip_weight),
            koopmans_ea_weight=float(args.koopmans_ea_weight),
            koopmans_lumo_ea_weight=float(args.koopmans_lumo_ea_weight),
            koopmans_loss_kind=str(args.koopmans_loss_kind),
            long_range_correction_weight=float(args.long_range_correction_weight),
        ),
    ]


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


def _record_from_state(
    *,
    step: int,
    raw: jnp.ndarray,
    loss: jnp.ndarray,
    metrics: dict[str, jnp.ndarray],
    grad_norm: float | None,
    accepted_scale: float | None,
    accepted: bool | None,
) -> dict[str, float | int | bool | None | list[float]]:
    sr = _metric_scalar(metrics, "sr_hf_fraction")
    lr = _metric_scalar(metrics, "lr_hf_fraction")
    omega = _metric_scalar(metrics, "omega")
    return {
        "step": int(step),
        "loss": float(loss),
        "koopmans_ip_mae": _metric_scalar(metrics, "koopmans_ip_mae"),
        "koopmans_ea_mae": _metric_scalar(metrics, "koopmans_ea_mae"),
        "koopmans_lumo_ea_mae": _metric_scalar(metrics, "koopmans_lumo_ea_mae"),
        "fractional_linearity_raw": _metric_scalar(metrics, "fractional_linearity_raw"),
        "long_range_correction_mae": _metric_scalar(metrics, "long_range_correction_mae"),
        "long_range_correction_residual": _metric_scalar(
            metrics,
            "long_range_correction_residual",
        ),
        "long_range_correction_penalty": _metric_scalar(
            metrics,
            "long_range_correction_penalty",
        ),
        "janak_frontier_mae": _metric_scalar(metrics, "janak_frontier_mae"),
        "janak_residual_homo": _metric_scalar(metrics, "janak_residual_homo"),
        "janak_residual_lumo": _metric_scalar(metrics, "janak_residual_lumo"),
        "janak_fd_homo": _metric_scalar(metrics, "janak_fd_homo"),
        "janak_fd_lumo": _metric_scalar(metrics, "janak_fd_lumo"),
        "koopmans_ip_residual": _metric_scalar(metrics, "koopmans_ip_residual"),
        "koopmans_ea_residual": _metric_scalar(metrics, "koopmans_ea_residual"),
        "koopmans_lumo_ea_residual": _metric_scalar(metrics, "koopmans_lumo_ea_residual"),
        "koopmans_gap_mae": _metric_scalar(metrics, "koopmans_gap_mae"),
        "koopmans_gap_residual": _metric_scalar(metrics, "koopmans_gap_residual"),
        "neutral_energy": _metric_scalar(metrics, "koopmans_neutral_energy"),
        "cation_energy": _metric_scalar(metrics, "koopmans_cation_energy"),
        "anion_energy": _metric_scalar(metrics, "koopmans_anion_energy"),
        "sr_hf_fraction": sr,
        "lr_hf_fraction": lr,
        "paper_alpha": sr,
        "paper_beta": lr - sr,
        "omega": omega,
        "raw_parameters": [float(x) for x in jnp.asarray(raw).reshape(-1)],
        "grad_norm": grad_norm,
        "accepted_scale": accepted_scale,
        "accepted": accepted,
    }


def _write_trajectory_plot(history: list[dict[str, Any]], outdir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    steps = [int(row["step"]) for row in history]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
    axes[0].plot(steps, [float(row["loss"]) for row in history], "-o", label="loss")
    axes[0].plot(
        steps,
        [float(row["janak_frontier_mae"]) for row in history],
        "-.",
        label="fixed-orbital Janak MAE",
    )
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
    axes[0].plot(
        steps,
        [float(row["koopmans_lumo_ea_mae"]) for row in history],
        "--",
        label="|LUMO(N)+EA|",
    )
    axes[0].plot(
        steps,
        [float(row["koopmans_gap_mae"]) for row in history],
        ":",
        label="|gap-(IP-EA)|",
    )
    axes[0].plot(
        steps,
        [float(row["fractional_linearity_raw"]) for row in history],
        ":",
        label="fractional curvature",
    )
    axes[0].plot(
        steps,
        [float(row["long_range_correction_mae"]) for row in history],
        ":",
        label="|lr-1|",
    )
    axes[0].set_ylabel("Hartree")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25, linewidth=0.6)

    axes[1].plot(steps, [float(row["sr_hf_fraction"]) for row in history], "-o", label="sr")
    axes[1].plot(steps, [float(row["lr_hf_fraction"]) for row in history], "-o", label="lr")
    axes[1].plot(steps, [float(row["omega"]) for row in history], "-o", label="omega")
    axes[1].set_xlabel("Optimization step")
    axes[1].set_ylabel("RSH parameter")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25, linewidth=0.6)

    fig.tight_layout()
    path = outdir / "raw_endpoint_tuning.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    omega_grid = _parse_float_tuple(args.omega_grid)
    descriptor_config = AtomCenteredDensityDescriptorConfig(
        radial_centers=_parse_float_tuple(args.radial_centers),
        radial_width=float(args.radial_width),
        max_angular=int(args.max_angular),
    )
    molecule = _build_water_reference(
        basis=str(args.basis),
        xc=str(args.xc),
        grid_level=int(args.grid_level),
        omega_grid=omega_grid,
    )
    local_xc_spec = _local_xc_spec_from_args(args)
    rsh_template = _rsh_template_from_args(args)
    coordinate_active_raw_dims = _coordinate_active_raw_dims_from_template(rsh_template)
    functional = make_atom_centered_density_rsh_functional(
        local_xc_spec=local_xc_spec,
        descriptor_config=descriptor_config,
        atom_hidden_dims=_parse_int_tuple(args.atom_hidden_dims),
        pooled_hidden_dims=_parse_int_tuple(args.pooled_hidden_dims),
        embedding_dim=int(args.embedding_dim),
        template=rsh_template,
        fallback_omega_values=omega_grid,
    )
    base_params = functional.init_from_molecule(jax.random.PRNGKey(int(args.seed)), molecule)
    initial_resolved = _initial_resolved_from_args(args)
    if initial_resolved is None:
        initial_resolved = functional.resolve_parameters(base_params, molecule)
    else:
        base_params = functional.params_with_resolved(
            base_params,
            initial_resolved,
            molecule=molecule,
            preserve_network=bool(args.preserve_network),
        )
    raw = functional.raw_parameters_from_resolved(initial_resolved)
    datum = GroundStateDatum(
        molecule=molecule,
        target_total_energy=jnp.asarray(molecule.mf_energy),
    )
    training_config = GroundStateTrainingConfig(
        mode="self_consistent",
        scf_gradient_mode=str(args.scf_gradient_mode),
        scf_max_cycle=int(args.scf_max_cycle),
        janak_frontier_mode=str(args.janak_mode),
        janak_frontier_delta=float(args.janak_delta),
        scf_require_convergence=False,
    )
    def params_from_raw(raw_parameters: jnp.ndarray):
        return functional.params_with_raw_output(
            base_params,
            raw_parameters,
            molecule=molecule,
            preserve_network=bool(args.preserve_network),
        )

    def stage_loss_fn(stage: StageSpec):
        return make_self_supervised_rsh_loss(
            functional,
            training_config=training_config,
            janak_weight=float(stage.janak_weight),
            fractional_weight=float(stage.fractional_weight),
            koopmans_ip_weight=float(stage.koopmans_ip_weight),
            koopmans_ea_weight=float(stage.koopmans_ea_weight),
            koopmans_lumo_ea_weight=float(stage.koopmans_lumo_ea_weight),
            koopmans_loss_kind=str(stage.koopmans_loss_kind),
            koopmans_detach_charged_states=False,
            koopmans_differentiate_charged_orbitals=False,
            long_range_correction_weight=float(stage.long_range_correction_weight),
            prior_weight=float(args.prior_weight),
        )

    def annotate_record(
        record: dict[str, Any],
        *,
        stage: StageSpec,
        stage_index: int,
        stage_local_step: int,
        stage_start: bool = False,
    ) -> dict[str, Any]:
        record["stage"] = stage.name
        record["stage_index"] = int(stage_index)
        record["stage_local_step"] = int(stage_local_step)
        record["stage_start"] = bool(stage_start)
        record["stage_fractional_weight"] = float(stage.fractional_weight)
        record["stage_koopmans_ip_weight"] = float(stage.koopmans_ip_weight)
        record["stage_koopmans_ea_weight"] = float(stage.koopmans_ea_weight)
        record["stage_koopmans_lumo_ea_weight"] = float(stage.koopmans_lumo_ea_weight)
        record["stage_koopmans_loss_kind"] = str(stage.koopmans_loss_kind)
        record["stage_long_range_correction_weight"] = float(
            stage.long_range_correction_weight
        )
        return record

    stages = _build_stage_specs(args)
    history: list[dict[str, Any]] = []
    best_raw = raw
    best_record: dict[str, Any] | None = None
    best_stage_index = max(len(stages) - 1, 0)
    coordinate_step_size = float(args.coordinate_step_size)
    global_step = 0

    for stage_index, stage in enumerate(stages):
        loss_fn = stage_loss_fn(stage)

        def raw_objective(raw_parameters: jnp.ndarray):
            return loss_fn(params_from_raw(raw_parameters), functional, datum)

        loss_and_grad = jax.value_and_grad(raw_objective, has_aux=True)
        tx = None
        opt_state = None
        if args.optimizer == "adam":
            tx = optax.adam(float(args.learning_rate))
            opt_state = tx.init(raw)
        elif args.optimizer == "sgd":
            tx = optax.sgd(float(args.learning_rate))
            opt_state = tx.init(raw)

        current_loss, current_metrics = raw_objective(raw)
        start_record = annotate_record(
            _record_from_state(
                step=global_step,
                raw=raw,
                loss=current_loss,
                metrics=current_metrics,
                grad_norm=None,
                accepted_scale=None,
                accepted=None,
            ),
            stage=stage,
            stage_index=stage_index,
            stage_local_step=0,
            stage_start=bool(history),
        )
        history.append(start_record)
        if stage_index == best_stage_index:
            best_raw = raw
            best_record = start_record

        print(
            f"stage={stage.name} "
            f"steps={stage.steps} "
            f"w_kip={stage.koopmans_ip_weight:g} "
            f"w_kea={stage.koopmans_ea_weight:g} "
            f"w_klumo={stage.koopmans_lumo_ea_weight:g} "
            f"koopmans_loss={stage.koopmans_loss_kind} "
            f"w_frac={stage.fractional_weight:g} "
            f"w_lc={stage.long_range_correction_weight:g}",
            flush=True,
        )

        for local_step in range(1, int(stage.steps) + 1):
            global_step += 1
            baseline_loss = current_loss
            baseline_metrics = current_metrics
            if args.optimizer == "coordinate":
                accepted = False
                accepted_scale = 0.0
                accepted_raw = raw
                accepted_loss = baseline_loss
                accepted_metrics = baseline_metrics
                best_candidate_loss = baseline_loss
                for dim in coordinate_active_raw_dims:
                    for sign in (-1.0, 1.0):
                        delta = jnp.zeros_like(raw).at[dim].set(
                            sign * coordinate_step_size
                        )
                        candidate_raw = raw + delta
                        candidate_loss, candidate_metrics = raw_objective(candidate_raw)
                        if (
                            jnp.isfinite(candidate_loss)
                            and float(candidate_loss) < float(best_candidate_loss)
                        ):
                            best_candidate_loss = candidate_loss
                            accepted_raw = candidate_raw
                            accepted_loss = candidate_loss
                            accepted_metrics = candidate_metrics
                            accepted = True
                            accepted_scale = coordinate_step_size

                if accepted and float(accepted_loss) <= (
                    float(baseline_loss) + float(args.accept_tolerance)
                ):
                    raw = accepted_raw
                    current_loss = accepted_loss
                    current_metrics = accepted_metrics
                else:
                    accepted = False
                    coordinate_step_size = max(
                        coordinate_step_size * float(args.coordinate_shrink),
                        float(args.coordinate_min_step_size),
                    )
                    current_loss = baseline_loss
                    current_metrics = baseline_metrics

                record = annotate_record(
                    _record_from_state(
                        step=global_step,
                        raw=raw,
                        loss=current_loss,
                        metrics=current_metrics,
                        grad_norm=None,
                        accepted_scale=accepted_scale,
                        accepted=accepted,
                    ),
                    stage=stage,
                    stage_index=stage_index,
                    stage_local_step=local_step,
                )
                record["coordinate_step_size"] = coordinate_step_size
                history.append(record)
                if stage_index == best_stage_index and (
                    best_record is None or float(record["loss"]) < float(best_record["loss"])
                ):
                    best_record = record
                    best_raw = raw

                print(
                    f"step={global_step:03d} "
                    f"stage={stage.name} "
                    f"loss={record['loss']:.6e} "
                    f"janak={record['janak_frontier_mae']:.3e} "
                    f"kip={record['koopmans_ip_mae']:.3e} "
                    f"kea={record['koopmans_ea_mae']:.3e} "
                    f"klumo={record['koopmans_lumo_ea_mae']:.3e} "
                    f"frac={record['fractional_linearity_raw']:.3e} "
                    f"lc={record['long_range_correction_mae']:.3e} "
                    f"sr={record['sr_hf_fraction']:.4f} "
                    f"lr={record['lr_hf_fraction']:.4f} "
                    f"omega={record['omega']:.4f} "
                    f"coord_step={coordinate_step_size:.3e} "
                    f"accepted={accepted}",
                    flush=True,
                )
                continue

            if tx is None or opt_state is None:
                raise RuntimeError(f"Unsupported optimizer={args.optimizer!r}.")
            (loss, metrics), grads = loss_and_grad(raw)
            clean_grads = jax.tree_util.tree_map(
                lambda x: jnp.nan_to_num(
                    jnp.asarray(x),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                grads,
            )
            grad_norm = float(_tree_l2_norm(clean_grads))
            updates, proposed_opt_state = tx.update(clean_grads, opt_state, raw)
            shrink = float(args.line_search_shrink)
            attempts = max(int(args.line_search_attempts), 1)
            scales = (
                [shrink**idx for idx in range(attempts)]
                if bool(args.line_search)
                else [1.0]
            )
            accepted = False
            accepted_scale = 0.0
            accepted_raw = raw
            accepted_loss = loss
            accepted_metrics = metrics
            tolerance = float(args.accept_tolerance)
            for scale in scales:
                scaled_updates = jax.tree_util.tree_map(lambda x: scale * x, updates)
                candidate_raw = optax.apply_updates(raw, scaled_updates)
                candidate_loss, candidate_metrics = raw_objective(candidate_raw)
                candidate_loss_value = float(candidate_loss)
                if jnp.isfinite(candidate_loss) and (
                    not bool(args.line_search)
                    or candidate_loss_value <= float(baseline_loss) + tolerance
                ):
                    accepted = True
                    accepted_scale = float(scale)
                    accepted_raw = candidate_raw
                    accepted_loss = candidate_loss
                    accepted_metrics = candidate_metrics
                    break

            if accepted:
                raw = accepted_raw
                opt_state = proposed_opt_state
                current_loss = accepted_loss
                current_metrics = accepted_metrics
            else:
                current_loss = baseline_loss
                current_metrics = baseline_metrics

            record = annotate_record(
                _record_from_state(
                    step=global_step,
                    raw=raw,
                    loss=current_loss,
                    metrics=current_metrics,
                    grad_norm=grad_norm,
                    accepted_scale=accepted_scale,
                    accepted=accepted,
                ),
                stage=stage,
                stage_index=stage_index,
                stage_local_step=local_step,
            )
            history.append(record)
            if stage_index == best_stage_index and (
                best_record is None or float(record["loss"]) < float(best_record["loss"])
            ):
                best_record = record
                best_raw = raw

            print(
                f"step={global_step:03d} "
                f"stage={stage.name} "
                f"loss={record['loss']:.6e} "
                f"janak={record['janak_frontier_mae']:.3e} "
                f"kip={record['koopmans_ip_mae']:.3e} "
                f"kea={record['koopmans_ea_mae']:.3e} "
                f"klumo={record['koopmans_lumo_ea_mae']:.3e} "
                f"frac={record['fractional_linearity_raw']:.3e} "
                f"lc={record['long_range_correction_mae']:.3e} "
                f"sr={record['sr_hf_fraction']:.4f} "
                f"lr={record['lr_hf_fraction']:.4f} "
                f"omega={record['omega']:.4f} "
                f"grad={grad_norm:.3e} "
                f"scale={accepted_scale:.3g} "
                f"accepted={accepted}",
                flush=True,
            )

    if best_record is None:
        raise RuntimeError("No optimization records were produced.")

    final_params = params_from_raw(raw)
    best_params = params_from_raw(best_raw)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_path = _write_trajectory_plot(history, outdir)
    summary = {
        "rsh_preset": _normalized_rsh_preset_name(args),
        "rsh_omega_source": _rsh_omega_source_from_args(args),
        "reference_xc": str(args.xc),
        "local_xc_spec": local_xc_spec,
        "initial_loss": float(history[0]["loss"]),
        "final_loss": float(history[-1]["loss"]),
        "best_loss": float(best_record["loss"]),
        "best_step": int(best_record["step"]),
        "final_janak_frontier_mae": float(history[-1]["janak_frontier_mae"]),
        "best_janak_frontier_mae": float(best_record["janak_frontier_mae"]),
        "final_sr_hf_fraction": float(history[-1]["sr_hf_fraction"]),
        "final_lr_hf_fraction": float(history[-1]["lr_hf_fraction"]),
        "final_paper_alpha": float(history[-1]["paper_alpha"]),
        "final_paper_beta": float(history[-1]["paper_beta"]),
        "final_omega": float(history[-1]["omega"]),
        "best_sr_hf_fraction": float(best_record["sr_hf_fraction"]),
        "best_lr_hf_fraction": float(best_record["lr_hf_fraction"]),
        "best_paper_alpha": float(best_record["paper_alpha"]),
        "best_paper_beta": float(best_record["paper_beta"]),
        "best_omega": float(best_record["omega"]),
        "learning_rate": float(args.learning_rate),
        "optimizer": str(args.optimizer),
        "line_search": bool(args.line_search),
        "coordinate_step_size": float(args.coordinate_step_size),
        "coordinate_shrink": float(args.coordinate_shrink),
        "coordinate_min_step_size": float(args.coordinate_min_step_size),
        "coordinate_active_raw_dims": [int(dim) for dim in coordinate_active_raw_dims],
        "janak_weight": float(args.janak_weight),
        "janak_mode": str(args.janak_mode),
        "janak_delta": float(args.janak_delta),
        "strategy": str(args.strategy),
        "stage_specs": [stage.__dict__ for stage in stages],
        "fractional_weight": float(args.fractional_weight),
        "long_range_correction_weight": float(args.long_range_correction_weight),
        "koopmans_ip_weight": float(args.koopmans_ip_weight),
        "koopmans_ea_weight": float(args.koopmans_ea_weight),
        "koopmans_lumo_ea_weight": float(args.koopmans_lumo_ea_weight),
        "koopmans_loss_kind": str(args.koopmans_loss_kind),
        "prior_weight": float(args.prior_weight),
        "scf_gradient_mode": str(args.scf_gradient_mode),
        "scf_max_cycle": int(args.scf_max_cycle),
        "preserve_network": bool(args.preserve_network),
        "history": history,
    }
    if _normalized_rsh_preset_name(args) is not None:
        preset = get_rsh_functional_preset(str(args.rsh_preset))
        summary["rsh_preset_pyscf_xc_name"] = preset.pyscf_xc_name
        summary["rsh_preset_strict_jax_supported"] = bool(preset.strict_jax_supported)
        summary["rsh_preset_note"] = preset.notes
        summary["rsh_preset_canonical_default_omega"] = float(preset.default_omega)
        summary["rsh_preset_canonical_omega_bounds"] = [
            float(preset.omega_bounds[0]),
            float(preset.omega_bounds[1]),
        ]
        if preset.optxc_default_omega is not None:
            summary["rsh_preset_optxc_default_omega"] = float(preset.optxc_default_omega)
        if preset.optxc_omega_bounds is not None:
            summary["rsh_preset_optxc_omega_bounds"] = [
                float(preset.optxc_omega_bounds[0]),
                float(preset.optxc_omega_bounds[1]),
            ]
    if plot_path is not None:
        summary["trajectory_plot"] = str(plot_path)

    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    save_params_checkpoint(
        outdir / "final_params.msgpack",
        final_params,
        metadata={
            "step": int(history[-1]["step"]),
            "loss": float(history[-1]["loss"]),
            "sr_hf_fraction": float(history[-1]["sr_hf_fraction"]),
            "lr_hf_fraction": float(history[-1]["lr_hf_fraction"]),
            "omega": float(history[-1]["omega"]),
            "preserve_network": bool(args.preserve_network),
        },
    )
    save_params_checkpoint(
        outdir / "best_params.msgpack",
        best_params,
        metadata={
            "step": int(best_record["step"]),
            "loss": float(best_record["loss"]),
            "sr_hf_fraction": float(best_record["sr_hf_fraction"]),
            "lr_hf_fraction": float(best_record["lr_hf_fraction"]),
            "omega": float(best_record["omega"]),
            "preserve_network": bool(args.preserve_network),
        },
    )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
