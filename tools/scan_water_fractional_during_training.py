from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax

from scan_water_fractional_occupation import (
    _build_water_reference,
    _parse_float_tuple,
    _parse_int_tuple,
    _piecewise_linear_energy,
    _scan_one_point,
)
from td_graddft.nn_rsh import (
    AtomCenteredDensityDescriptorConfig,
    make_atom_centered_density_rsh_functional,
    make_self_supervised_rsh_loss,
)
from td_graddft.training import (
    GroundStateDatum,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    make_ground_state_train_step,
)
from td_graddft.training.targets import (
    _freeze_functional_for_fractional_path,
    _resolve_training_molecule_and_info_with_mode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay water nn-RSH training and scan N-1..N..N+1 fractional occupations at every step.",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--scan-every", type=int, default=1)
    parser.add_argument("--points", type=int, default=21)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine", "exponential"),
        default="cosine",
    )
    parser.add_argument("--final-learning-rate-scale", type=float, default=0.1)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--scf-max-cycle", type=int, default=4)
    parser.add_argument(
        "--scf-gradient-mode",
        choices=("unrolled", "implicit_commutator"),
        default="implicit_commutator",
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
        default="autodiff",
    )
    parser.add_argument("--janak-delta", type=float, default=0.1)
    parser.add_argument("--janak-weight", type=float, default=1.0)
    parser.add_argument("--fractional-weight", type=float, default=0.0)
    parser.add_argument("--koopmans-ip-weight", type=float, default=0.0)
    parser.add_argument("--koopmans-ea-weight", type=float, default=0.0)
    parser.add_argument("--koopmans-lumo-ea-weight", type=float, default=0.0)
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
    parser.add_argument("--omega-grid", default="0.0,0.3,0.6")
    parser.add_argument("--radial-centers", default="0.6,1.4,2.6")
    parser.add_argument("--radial-width", type=float, default=0.5)
    parser.add_argument("--max-angular", type=int, default=1)
    parser.add_argument("--atom-hidden-dims", default="16,16")
    parser.add_argument("--pooled-hidden-dims", default="16")
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--outdir",
        default="outputs/water_nn_rsh_fractional_scan_during_training_ep10",
    )
    return parser.parse_args()


def _make_learning_rate_schedule(args: argparse.Namespace):
    base_lr = float(args.learning_rate)
    final_scale = float(args.final_learning_rate_scale)
    if args.lr_schedule == "constant":
        return optax.constant_schedule(base_lr)
    if args.lr_schedule == "cosine":
        return optax.cosine_decay_schedule(
            init_value=base_lr,
            decay_steps=max(int(args.steps), 1),
            alpha=max(min(final_scale, 1.0), 0.0),
        )
    if args.lr_schedule == "exponential":
        return optax.exponential_decay(
            init_value=base_lr,
            transition_steps=max(int(args.steps), 1),
            decay_rate=max(final_scale, 1e-8),
            staircase=False,
        )
    raise ValueError(f"Unsupported lr_schedule={args.lr_schedule!r}.")


def _metric_scalar(metrics: dict[str, jnp.ndarray], key: str) -> float:
    arr = jnp.asarray(metrics[key])
    if arr.ndim == 0:
        return float(arr)
    return float(arr.reshape(-1)[0])


def _scan_params(
    *,
    step: int,
    params: Any,
    functional: Any,
    molecule: Any,
    loss_fn: Any,
    datum: GroundStateDatum,
    training_config: GroundStateTrainingConfig,
    q_values: list[float],
) -> dict[str, Any]:
    loss, metrics = loss_fn(params, functional, datum)
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
    resolved = functional.resolve_parameters(params, base_molecule)

    energies: dict[float, float] = {}
    rms_values: dict[float, float | None] = {}
    for q in q_values:
        energy, rms = _scan_one_point(
            frozen_params,
            frozen_functional,
            base_molecule,
            q=q,
            training_config=training_config,
        )
        energies[q] = energy
        rms_values[q] = rms

    q_minus = min(q_values, key=lambda value: abs(value + 1.0))
    q_zero = min(q_values, key=lambda value: abs(value))
    q_plus = min(q_values, key=lambda value: abs(value - 1.0))
    e_minus = energies[q_minus]
    e_zero = energies[q_zero]
    e_plus = energies[q_plus]

    rows: list[dict[str, float | None]] = []
    for q in q_values:
        energy = energies[q]
        linear = _piecewise_linear_energy(q, e_minus, e_zero, e_plus)
        rows.append(
            {
                "step": float(step),
                "q": q,
                "energy_hartree": energy,
                "relative_energy_hartree": energy - e_zero,
                "linear_energy_hartree": linear,
                "linear_relative_energy_hartree": linear - e_zero,
                "linearity_deviation_hartree": energy - linear,
                "selected_rms_density": rms_values[q],
            }
        )

    max_abs_dev = max(abs(float(row["linearity_deviation_hartree"])) for row in rows)
    return {
        "step": int(step),
        "loss": float(loss),
        "janak_frontier_mae": _metric_scalar(metrics, "janak_frontier_mae"),
        "koopmans_ip_mae": _metric_scalar(metrics, "koopmans_ip_mae"),
        "koopmans_ea_mae": _metric_scalar(metrics, "koopmans_ea_mae"),
        "koopmans_lumo_ea_mae": _metric_scalar(metrics, "koopmans_lumo_ea_mae"),
        "sr_hf_fraction": float(resolved.sr_hf_fraction),
        "lr_hf_fraction": float(resolved.lr_hf_fraction),
        "omega": float(resolved.omega),
        "base_selected_rms_density": (
            float(jnp.asarray(getattr(base_info, "selected_rms_density", 0.0)))
            if getattr(base_info, "mode", None) == "self_consistent"
            else None
        ),
        "max_abs_linearity_deviation_hartree": max_abs_dev,
        "max_abs_linearity_deviation_mhartree": 1000.0 * max_abs_dev,
        "rows": rows,
    }


def _write_outputs(
    snapshots: list[dict[str, Any]],
    *,
    outdir: Path,
) -> tuple[Path, Path | None]:
    rows = [row for snapshot in snapshots for row in snapshot["rows"]]
    csv_path = outdir / "fractional_scan_during_training.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    os.environ.setdefault("MPLCONFIGDIR", str(outdir / ".mplconfig"))
    try:
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
    except ModuleNotFoundError:
        return csv_path, None

    steps = [int(snapshot["step"]) for snapshot in snapshots]
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=min(steps), vmax=max(steps) if max(steps) > min(steps) else min(steps) + 1)

    fig, (ax_energy, ax_dev) = plt.subplots(
        2,
        1,
        figsize=(8.0, 6.6),
        sharex=False,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    for snapshot in snapshots:
        step = int(snapshot["step"])
        q = [float(row["q"]) for row in snapshot["rows"]]
        rel_mha = [
            1000.0 * float(row["relative_energy_hartree"])
            for row in snapshot["rows"]
        ]
        color = cmap(norm(step))
        linewidth = 2.4 if step in (min(steps), max(steps)) else 1.2
        alpha = 1.0 if step in (min(steps), max(steps)) else 0.35
        label = f"step {step}" if step in (min(steps), max(steps)) else None
        ax_energy.plot(q, rel_mha, "-o", color=color, linewidth=linewidth, markersize=3.5, alpha=alpha, label=label)

    final_rows = snapshots[-1]["rows"]
    q_final = [float(row["q"]) for row in final_rows]
    linear_final_mha = [
        1000.0 * float(row["linear_relative_energy_hartree"])
        for row in final_rows
    ]
    ax_energy.plot(
        q_final,
        linear_final_mha,
        "--",
        color="0.2",
        linewidth=1.8,
        label="final PL reference",
    )
    ax_energy.axvline(0.0, color="0.75", linewidth=0.8)
    ax_energy.set_title("Water nn-RSH fractional scan during training")
    ax_energy.set_xlabel("Electron displacement q")
    ax_energy.set_ylabel(r"$E(N+q)-E(N)$ (mHa)")
    ax_energy.grid(alpha=0.25, linewidth=0.6)
    ax_energy.legend(frameon=False)
    fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax_energy,
        pad=0.015,
        label="training step",
    )

    ax_dev.plot(
        steps,
        [float(snapshot["max_abs_linearity_deviation_mhartree"]) for snapshot in snapshots],
        "-o",
        color="#b23a48",
        linewidth=2.0,
        markersize=4.5,
    )
    ax_dev.set_xlabel("Training step")
    ax_dev.set_ylabel("max |deviation| (mHa)")
    ax_dev.grid(alpha=0.25, linewidth=0.6)

    fig.tight_layout()
    plot_path = outdir / "fractional_scan_during_training.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return csv_path, plot_path


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    omega_grid = _parse_float_tuple(args.omega_grid)
    molecule = _build_water_reference(
        basis=str(args.basis),
        xc=str(args.xc),
        grid_level=int(args.grid_level),
        omega_grid=omega_grid,
    )
    descriptor_config = AtomCenteredDensityDescriptorConfig(
        radial_centers=_parse_float_tuple(args.radial_centers),
        radial_width=float(args.radial_width),
        max_angular=int(args.max_angular),
    )
    functional = make_atom_centered_density_rsh_functional(
        local_xc_spec=str(args.xc),
        descriptor_config=descriptor_config,
        atom_hidden_dims=_parse_int_tuple(args.atom_hidden_dims),
        pooled_hidden_dims=_parse_int_tuple(args.pooled_hidden_dims),
        embedding_dim=int(args.embedding_dim),
        fallback_omega_values=omega_grid,
    )
    training_config = GroundStateTrainingConfig(
        mode="self_consistent",
        scf_gradient_mode=str(args.scf_gradient_mode),
        scf_max_cycle=int(args.scf_max_cycle),
        janak_frontier_mode=str(args.janak_mode),
        janak_frontier_delta=float(args.janak_delta),
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
        koopmans_detach_charged_states=bool(args.koopmans_detach_charged_states),
        koopmans_differentiate_charged_orbitals=bool(
            args.koopmans_differentiate_charged_orbitals
        ),
        prior_weight=float(args.prior_weight),
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
    q_values = [float(x) for x in jnp.linspace(-1.0, 1.0, max(3, int(args.points)))]

    snapshots: list[dict[str, Any]] = []
    scan_every = max(1, int(args.scan_every))
    for step in range(0, int(args.steps) + 1):
        if step % scan_every == 0 or step == int(args.steps):
            snapshot = _scan_params(
                step=step,
                params=state.params,
                functional=functional,
                molecule=molecule,
                loss_fn=loss_fn,
                datum=datum,
                training_config=training_config,
                q_values=q_values,
            )
            snapshots.append(snapshot)
            print(
                f"scan step={step:03d} janak={snapshot['janak_frontier_mae']:.6e} "
                f"kip={snapshot['koopmans_ip_mae']:.3e} "
                f"kea={snapshot['koopmans_ea_mae']:.3e} "
                f"klumo={snapshot['koopmans_lumo_ea_mae']:.3e} "
                f"max_dev={snapshot['max_abs_linearity_deviation_mhartree']:.3f} mHa "
                f"sr={snapshot['sr_hf_fraction']:.4f} lr={snapshot['lr_hf_fraction']:.4f} "
                f"omega={snapshot['omega']:.4f}",
                flush=True,
            )
        if step == int(args.steps):
            break
        state, _metrics = train_step(state, datum)

    csv_path, plot_path = _write_outputs(snapshots, outdir=outdir)
    summary = {
        "csv": str(csv_path),
        "plot": str(plot_path) if plot_path is not None else None,
        "steps": int(args.steps),
        "scan_every": scan_every,
        "points": len(q_values),
        "learning_rate": float(args.learning_rate),
        "lr_schedule": str(args.lr_schedule),
        "janak_mode": str(args.janak_mode),
        "scf_gradient_mode": str(args.scf_gradient_mode),
        "koopmans_detach_charged_states": bool(args.koopmans_detach_charged_states),
        "koopmans_differentiate_charged_orbitals": bool(
            args.koopmans_differentiate_charged_orbitals
        ),
        "snapshots": [
            {key: value for key, value in snapshot.items() if key != "rows"}
            for snapshot in snapshots
        ],
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")
    if plot_path is not None:
        print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
