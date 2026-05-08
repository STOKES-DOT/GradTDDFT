from __future__ import annotations

import argparse
import csv
import json
import os
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
    make_rsh_template,
)
from td_graddft.reference_legacy import restricted_reference_from_pyscf
from td_graddft.training import (
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    load_params_checkpoint,
)
from td_graddft.training.targets import (
    _freeze_functional_for_fractional_path,
    _predict_ground_state_total_energy_from_molecule,
    _resolve_training_molecule_and_info_with_mode,
    _resolve_variational_frontier_state_and_info,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan water fractional occupations from N-1 to N to N+1 for a trained nn-RSH checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/water_nn_rsh_autodiff_implicit_ep10_fractional/best_params.msgpack",
    )
    parser.add_argument("--outdir", default="outputs/water_nn_rsh_fractional_scan_npm1")
    parser.add_argument("--points", type=int, default=21)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument("--xc", default="pbe")
    parser.add_argument(
        "--rsh-preset",
        default="",
        help="Optional named RSH preset used to reconstruct the checkpoint template.",
    )
    parser.add_argument(
        "--rsh-omega-source",
        choices=("canonical", "optxc"),
        default="canonical",
        help=(
            "Omega-bounds source used when reconstructing a named-preset checkpoint. "
            "Use the same value used during training."
        ),
    )
    parser.add_argument(
        "--omega-bounds",
        default="",
        help=(
            "Optional lower,upper omega interval used to decode trained RSH "
            "parameters. Use the same bounds used during training."
        ),
    )
    parser.add_argument("--omega-grid", default="0.0,0.3,0.6")
    parser.add_argument("--radial-centers", default="0.6,1.4,2.6")
    parser.add_argument("--radial-width", type=float, default=0.5)
    parser.add_argument("--max-angular", type=int, default=1)
    parser.add_argument("--atom-hidden-dims", default="16,16")
    parser.add_argument("--pooled-hidden-dims", default="16")
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scf-max-cycle", type=int, default=12)
    parser.add_argument("--scf-damping", type=float, default=0.35)
    parser.add_argument("--scf-level-shift", type=float, default=0.5)
    parser.add_argument(
        "--scf-iterate-selection",
        choices=("final", "best_rms", "first_converged"),
        default="best_rms",
    )
    parser.add_argument(
        "--scf-gradient-mode",
        choices=("unrolled", "implicit_commutator"),
        default="implicit_commutator",
    )
    return parser.parse_args()


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _rsh_template_from_name(name: str, *, omega_source: str = "canonical"):
    clean = str(name or "").strip()
    if not clean:
        return None
    return make_rsh_template(clean, omega_source=omega_source)


def _omega_grid_covering_bounds(
    omega_grid: tuple[float, ...],
    bounds: tuple[float, float] | None,
) -> tuple[float, ...]:
    values = [float(value) for value in omega_grid]
    if bounds is not None:
        lower, upper = (float(bounds[0]), float(bounds[1]))
        if not values:
            values = [lower, upper]
        if min(values) > lower:
            values.append(lower)
        if max(values) < upper:
            values.append(upper)
    return tuple(sorted(set(round(value, 12) for value in values)))


def _local_xc_spec_from_args(args: argparse.Namespace) -> str:
    xc = str(getattr(args, "xc", "pbe") or "pbe")
    preset_name = str(getattr(args, "rsh_preset", "") or "").strip()
    if not preset_name:
        return xc
    preset = get_rsh_functional_preset(preset_name)
    if xc.strip().lower() == "pbe" and preset.jax_local_xc_spec:
        return preset.jax_local_xc_spec
    return xc


def _build_water_reference(
    *,
    basis: str,
    xc: str,
    grid_level: int,
    omega_grid: tuple[float, ...],
) -> Any:
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


def _scan_one_point(
    frozen_params: Any,
    frozen_functional: Any,
    base_molecule: Any,
    *,
    q: float,
    training_config: GroundStateTrainingConfig,
) -> tuple[float, float | None]:
    if q < 0.0:
        molecule_q, info_q = _resolve_variational_frontier_state_and_info(
            frozen_params,
            frozen_functional,
            base_molecule,
            homo_delta=float(q),
            training_config=training_config,
        )
    elif q > 0.0:
        molecule_q, info_q = _resolve_variational_frontier_state_and_info(
            frozen_params,
            frozen_functional,
            base_molecule,
            lumo_delta=float(q),
            training_config=training_config,
        )
    else:
        molecule_q = base_molecule
        info_q = None

    energy = float(
        _predict_ground_state_total_energy_from_molecule(
            frozen_params,
            frozen_functional,
            molecule_q,
        )
    )
    if info_q is None or getattr(info_q, "mode", None) != "self_consistent":
        return energy, None
    return energy, float(jnp.asarray(getattr(info_q, "selected_rms_density", 0.0)))


def _piecewise_linear_energy(q: float, e_minus: float, e_zero: float, e_plus: float) -> float:
    if q <= 0.0:
        t = q + 1.0
        return (1.0 - t) * e_minus + t * e_zero
    t = q
    return (1.0 - t) * e_zero + t * e_plus


def _write_plot(rows: list[dict[str, float | None]], outdir: Path) -> Path | None:
    os.environ.setdefault("MPLCONFIGDIR", str(outdir / ".mplconfig"))
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    q = [float(row["q"]) for row in rows]
    rel_mha = [1000.0 * float(row["relative_energy_hartree"]) for row in rows]
    lin_rel_mha = [1000.0 * float(row["linear_relative_energy_hartree"]) for row in rows]
    dev_mha = [1000.0 * float(row["linearity_deviation_hartree"]) for row in rows]

    fig, (ax_energy, ax_dev) = plt.subplots(
        2,
        1,
        figsize=(7.4, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    ax_energy.plot(q, rel_mha, "-o", linewidth=2.2, markersize=4.5, label="SCF fractional energy")
    ax_energy.plot(q, lin_rel_mha, "--", linewidth=1.8, label="piecewise-linear reference")
    ax_energy.axvline(0.0, color="0.75", linewidth=0.8)
    ax_energy.set_ylabel(r"$E(N+q)-E(N)$ (mHa)")
    ax_energy.set_title("Water nn-RSH fractional occupation scan")
    ax_energy.grid(alpha=0.25, linewidth=0.6)
    ax_energy.legend(frameon=False)

    ax_dev.plot(q, dev_mha, "-o", color="#b23a48", linewidth=1.8, markersize=4)
    ax_dev.axhline(0.0, color="0.2", linewidth=0.8)
    ax_dev.axvline(0.0, color="0.75", linewidth=0.8)
    ax_dev.set_xlabel("Electron displacement q")
    ax_dev.set_ylabel("deviation (mHa)")
    ax_dev.grid(alpha=0.25, linewidth=0.6)

    fig.tight_layout()
    plot_path = outdir / "fractional_scan_nminus1_to_nplus1.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    template = _rsh_template_from_name(
        str(args.rsh_preset),
        omega_source=str(args.rsh_omega_source),
    )
    explicit_bounds = None
    if str(args.omega_bounds).strip():
        if template is None:
            raise ValueError("--omega-bounds requires --rsh-preset.")
        parsed_bounds = _parse_float_tuple(str(args.omega_bounds))
        if len(parsed_bounds) != 2:
            raise ValueError("--omega-bounds must contain exactly lower,upper.")
        explicit_bounds = (float(parsed_bounds[0]), float(parsed_bounds[1]))
        template = replace(
            template,
            omega_bounds=explicit_bounds,
            default_omega=min(
                max(float(template.default_omega), explicit_bounds[0]),
                explicit_bounds[1],
            ),
        )
    omega_grid = _omega_grid_covering_bounds(
        _parse_float_tuple(args.omega_grid),
        explicit_bounds if explicit_bounds is not None else (
            tuple(float(value) for value in template.omega_bounds)
            if template is not None
            else None
        ),
    )
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
    local_xc_spec = _local_xc_spec_from_args(args)
    functional = make_atom_centered_density_rsh_functional(
        local_xc_spec=local_xc_spec,
        descriptor_config=descriptor_config,
        atom_hidden_dims=_parse_int_tuple(args.atom_hidden_dims),
        pooled_hidden_dims=_parse_int_tuple(args.pooled_hidden_dims),
        embedding_dim=int(args.embedding_dim),
        template=template,
        fallback_omega_values=omega_grid,
    )
    state_template = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(int(args.seed)),
        molecule,
        optax.adam(1e-3),
    )
    params = load_params_checkpoint(args.checkpoint, template=state_template.params)
    training_config = GroundStateTrainingConfig(
        mode="self_consistent",
        scf_gradient_mode=str(args.scf_gradient_mode),
        scf_max_cycle=int(args.scf_max_cycle),
        scf_damping=float(args.scf_damping),
        scf_level_shift=float(args.scf_level_shift),
        scf_iterate_selection=str(args.scf_iterate_selection),
        scf_require_convergence=False,
        fractional_branch_scf_max_cycle=int(args.scf_max_cycle),
        fractional_branch_scf_damping=float(args.scf_damping),
        fractional_branch_scf_level_shift=float(args.scf_level_shift),
        fractional_branch_scf_iterate_selection=str(args.scf_iterate_selection),
    )
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

    npoints = max(3, int(args.points))
    q_values = [float(x) for x in jnp.linspace(-1.0, 1.0, npoints)]
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
        print(f"q={q:+.3f} E={energy:.10f} rms={rms}", flush=True)

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
                "q": q,
                "electron_count_relative_to_neutral": q,
                "energy_hartree": energy,
                "relative_energy_hartree": energy - e_zero,
                "linear_energy_hartree": linear,
                "linear_relative_energy_hartree": linear - e_zero,
                "linearity_deviation_hartree": energy - linear,
                "selected_rms_density": rms_values[q],
            }
        )

    csv_path = outdir / "fractional_scan.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_path = _write_plot(rows, outdir)
    max_abs_dev = max(abs(float(row["linearity_deviation_hartree"])) for row in rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "rsh_preset": str(args.rsh_preset).strip() or None,
        "rsh_omega_source": str(args.rsh_omega_source),
        "reference_xc": str(args.xc),
        "local_xc_spec": local_xc_spec,
        "template_omega_bounds": (
            [float(template.omega_bounds[0]), float(template.omega_bounds[1])]
            if template is not None
            else None
        ),
        "omega_grid": [float(value) for value in omega_grid],
        "csv": str(csv_path),
        "plot": str(plot_path) if plot_path is not None else None,
        "points": npoints,
        "sr_hf_fraction": float(resolved.sr_hf_fraction),
        "lr_hf_fraction": float(resolved.lr_hf_fraction),
        "omega": float(resolved.omega),
        "base_selected_rms_density": (
            float(jnp.asarray(getattr(base_info, "selected_rms_density", 0.0)))
            if getattr(base_info, "mode", None) == "self_consistent"
            else None
        ),
        "e_nminus1_hartree": e_minus,
        "e_n_hartree": e_zero,
        "e_nplus1_hartree": e_plus,
        "max_abs_linearity_deviation_hartree": max_abs_dev,
        "max_abs_linearity_deviation_mhartree": 1000.0 * max_abs_dev,
        "rows": rows,
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")
    if plot_path is not None:
        print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
