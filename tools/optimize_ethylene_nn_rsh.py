from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax

from td_graddft.nn_rsh import (
    AtomCenteredDensityDescriptorConfig,
    FixedDensityRSHDatum,
    get_rsh_functional_preset,
    make_atom_centered_density_rsh_functional,
    make_fixed_density_rsh_loss,
    make_rsh_template,
    make_self_supervised_rsh_loss,
    rsh_preset_default_params,
)
from td_graddft.training import (
    GroundStateDatum,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    make_ground_state_eval,
    make_ground_state_train_step,
    save_params_checkpoint,
)
from td_graddft.scf import UKSConfig
from td_graddft.scf.builders import (
    unrestricted_molecule_from_spec_with_gpu4pyscf_uks,
    unrestricted_molecule_from_spec_with_jax_uks,
)
from td_graddft.workflows.core import run_molecule_from_spec
from td_graddft.workflows.types import MoleculeSpecConfig, SimulationConfig


ETHYLENE_GEOMETRY = """
C -0.6695  0.0000  0.0000
C  0.6695  0.0000  0.0000
H -1.2321  0.9289  0.0000
H -1.2321 -0.9289  0.0000
H  1.2321  0.9289  0.0000
H  1.2321 -0.9289  0.0000
"""


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in str(raw).split(",") if part.strip())


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())


def _metric_scalar(metrics: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = metrics.get(key)
    if value is None:
        return float(default)
    arr = jnp.asarray(value)
    if int(arr.size) == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def _stage_log(label: str, start: float | None = None) -> float:
    now = time.perf_counter()
    if start is None:
        print(f"[stage] {label}", flush=True)
    else:
        print(f"[stage] {label} done in {now - start:.3f}s", flush=True)
    return now


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-supervised LC-wPBE RSH parameter optimization for ethylene."
    )
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument("--reference-xc", default="pbe")
    parser.add_argument("--rsh-preset", default="lc-wpbe")
    parser.add_argument("--rsh-omega-source", choices=("canonical", "optxc"), default="optxc")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--scf-max-cycle", type=int, default=6)
    parser.add_argument("--scf-conv-tol-density", type=float, default=1e-7)
    parser.add_argument("--scf-damping", type=float, default=0.35)
    parser.add_argument("--scf-level-shift", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rsh-training-mode",
        choices=("self_consistent", "fixed_density"),
        default="self_consistent",
    )
    parser.add_argument(
        "--scf-backend",
        choices=("jax_rks", "gpu4pyscf_rks"),
        default="gpu4pyscf_rks",
    )
    parser.add_argument(
        "--execution-device",
        choices=("auto", "cpu", "gpu"),
        default="gpu",
    )
    parser.add_argument(
        "--runtime-forward-backend",
        choices=("auto", "jax", "gpu4pyscf_rks"),
        default="gpu4pyscf_rks",
    )
    parser.add_argument(
        "--implicit-response-backend",
        choices=("jax", "gpu4pyscf_jk"),
        default="gpu4pyscf_jk",
    )
    parser.add_argument("--omega-grid", default="0.0,0.13,0.205,0.30,0.50")
    parser.add_argument("--radial-centers", default="0.6,1.4,2.6")
    parser.add_argument("--radial-width", type=float, default=0.5)
    parser.add_argument("--max-angular", type=int, default=1)
    parser.add_argument("--atom-hidden-dims", default="16,16")
    parser.add_argument("--pooled-hidden-dims", default="16")
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--koopmans-ip-weight", type=float, default=1.0)
    parser.add_argument("--koopmans-ea-weight", type=float, default=0.0)
    parser.add_argument("--koopmans-lumo-ea-weight", type=float, default=1.0)
    parser.add_argument("--koopmans-loss-kind", choices=("absolute", "squared"), default="squared")
    parser.add_argument("--janak-weight", type=float, default=0.0)
    parser.add_argument("--fractional-weight", type=float, default=0.0)
    parser.add_argument("--prior-weight", type=float, default=0.0)
    parser.add_argument("--outdir", default="outputs/ethylene_nn_rsh_opt")
    return parser.parse_args(argv)


def _charged_reference_backend(neutral_backend: str) -> str:
    return "gpu4pyscf_uks" if str(neutral_backend) == "gpu4pyscf_rks" else "jax_uks"


def _build_fixed_density_charged_reference(
    args: argparse.Namespace,
    *,
    charge: int,
    spin: int,
    omega_grid: tuple[float, ...],
):
    backend = _charged_reference_backend(str(args.scf_backend))
    builder = (
        unrestricted_molecule_from_spec_with_gpu4pyscf_uks
        if backend == "gpu4pyscf_uks"
        else unrestricted_molecule_from_spec_with_jax_uks
    )
    return builder(
        atom=ETHYLENE_GEOMETRY,
        basis=str(args.basis),
        xc_spec=str(args.reference_xc),
        charge=int(charge),
        spin=int(spin),
        grids_level=int(args.grid_level),
        uks_config=UKSConfig(
            xc_spec=str(args.reference_xc),
            max_cycle=80,
            conv_tol=1e-10,
        ),
        integral_backend="libcint",
        compute_local_hfx_features=True,
        compute_local_hfx_aux=True,
        hfx_omega_values=omega_grid,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    omega_grid = _parse_float_tuple(args.omega_grid)
    stage = _stage_log("build reference molecule")
    reference = run_molecule_from_spec(
        MoleculeSpecConfig(
            atom=ETHYLENE_GEOMETRY,
            basis=str(args.basis),
            xc=str(args.reference_xc),
            charge=0,
            spin=0,
            grids_level=int(args.grid_level),
        ),
        simulation=SimulationConfig(
            scf_backend=str(args.scf_backend),
            nstates=0,
            execution_device=str(args.execution_device),
            move_reference_to_device=True,
            jax_rks_max_cycle=80,
            jax_rks_conv_tol=1e-10,
            jax_integral_backend="libcint",
        ),
        compute_local_hfx_features=True,
        compute_local_hfx_aux=True,
        hfx_omega_values=omega_grid,
    )
    stage = _stage_log("build reference molecule", stage)
    molecule = reference.molecule
    cation_molecule = None
    anion_molecule = None
    if str(args.rsh_training_mode) == "fixed_density":
        stage = _stage_log("build fixed-density charged references")
        cation_molecule = _build_fixed_density_charged_reference(
            args,
            charge=1,
            spin=1,
            omega_grid=omega_grid,
        )
        anion_molecule = _build_fixed_density_charged_reference(
            args,
            charge=-1,
            spin=1,
            omega_grid=omega_grid,
        )
        stage = _stage_log("build fixed-density charged references", stage)

    stage = _stage_log("initialize RSH functional")
    preset = get_rsh_functional_preset(str(args.rsh_preset))
    template = make_rsh_template(
        str(args.rsh_preset),
        omega_source=str(args.rsh_omega_source),
    )
    descriptor_config = AtomCenteredDensityDescriptorConfig(
        radial_centers=_parse_float_tuple(args.radial_centers),
        radial_width=float(args.radial_width),
        max_angular=int(args.max_angular),
    )
    functional = make_atom_centered_density_rsh_functional(
        local_xc_spec=preset.jax_local_xc_spec or str(args.reference_xc),
        local_term_specs=tuple(preset.local_term_specs),
        descriptor_config=descriptor_config,
        atom_hidden_dims=_parse_int_tuple(args.atom_hidden_dims),
        pooled_hidden_dims=_parse_int_tuple(args.pooled_hidden_dims),
        embedding_dim=int(args.embedding_dim),
        template=template,
        fallback_omega_values=omega_grid,
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(int(args.seed)), molecule)
    params = functional.params_with_resolved(
        params,
        rsh_preset_default_params(
            str(args.rsh_preset),
            omega_source=str(args.rsh_omega_source),
        ),
        molecule=molecule,
        preserve_network=True,
    )
    stage = _stage_log("initialize RSH functional", stage)

    stage = _stage_log("build loss/training kernels")
    training_config = GroundStateTrainingConfig(
        mode=str(args.rsh_training_mode),
        scf_gradient_mode="impl",
        scf_runtime_forward_backend=str(args.runtime_forward_backend),
        implicit_response_backend=str(args.implicit_response_backend),
        scf_implicit_forward_mode="input_state",
        scf_max_cycle=int(args.scf_max_cycle),
        scf_damping=float(args.scf_damping),
        scf_level_shift=float(args.scf_level_shift),
        scf_conv_tol_density=float(args.scf_conv_tol_density),
        scf_require_convergence=False,
    )
    if str(args.rsh_training_mode) == "fixed_density":
        if cation_molecule is None or anion_molecule is None:
            raise RuntimeError("fixed_density RSH training requires charged references.")
        loss_fn = make_fixed_density_rsh_loss(
            functional,
            training_config=training_config,
            koopmans_ip_weight=float(args.koopmans_ip_weight),
            koopmans_ea_weight=float(args.koopmans_ea_weight),
            koopmans_lumo_ea_weight=float(args.koopmans_lumo_ea_weight),
            koopmans_loss_kind=str(args.koopmans_loss_kind),
            prior_weight=float(args.prior_weight),
        )
        datum = FixedDensityRSHDatum(
            molecule=molecule,
            cation_molecule=cation_molecule,
            anion_molecule=anion_molecule,
            target_total_energy=jnp.asarray(molecule.mf_energy),
        )
    else:
        loss_fn = make_self_supervised_rsh_loss(
            functional,
            training_config=training_config,
            janak_weight=float(args.janak_weight),
            fractional_weight=float(args.fractional_weight),
            koopmans_ip_weight=float(args.koopmans_ip_weight),
            koopmans_ea_weight=float(args.koopmans_ea_weight),
            koopmans_lumo_ea_weight=float(args.koopmans_lumo_ea_weight),
            koopmans_loss_kind=str(args.koopmans_loss_kind),
            koopmans_detach_charged_states=True,
            koopmans_differentiate_charged_orbitals=False,
            prior_weight=float(args.prior_weight),
        )
        datum = GroundStateDatum(
            molecule=molecule,
            target_total_energy=jnp.asarray(molecule.mf_energy),
        )
    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(int(args.seed)),
        molecule,
        optax.adam(float(args.learning_rate)),
    )
    state = state.replace(params=params)
    train_step = make_ground_state_train_step(
        functional,
        training_config=training_config,
        loss_fn=loss_fn,
    )
    eval_loss = make_ground_state_eval(
        functional,
        training_config=training_config,
        loss_fn=loss_fn,
    )
    stage = _stage_log("build loss/training kernels", stage)

    history: list[dict[str, float]] = []
    stage = _stage_log("initial eval_loss")
    initial_loss, initial_metrics = eval_loss(state.params, datum)
    stage = _stage_log("initial eval_loss", stage)
    initial_row = {
        "step": 0.0,
        "loss": float(initial_loss),
        "omega": _metric_scalar(initial_metrics, "omega"),
        "sr_hf_fraction": _metric_scalar(initial_metrics, "sr_hf_fraction"),
        "lr_hf_fraction": _metric_scalar(initial_metrics, "lr_hf_fraction"),
        "koopmans_ip_mae": _metric_scalar(initial_metrics, "koopmans_ip_mae"),
        "koopmans_lumo_ea_mae": _metric_scalar(initial_metrics, "koopmans_lumo_ea_mae"),
        "koopmans_ea_mae": _metric_scalar(initial_metrics, "koopmans_ea_mae"),
    }
    history.append(initial_row)
    print(
        "step=000 "
        f"loss={initial_row['loss']:.6e} "
        f"omega={initial_row['omega']:.6f} "
        f"kip={initial_row['koopmans_ip_mae']:.3e} "
        f"klumo={initial_row['koopmans_lumo_ea_mae']:.3e}",
        flush=True,
    )

    best_row = dict(initial_row)
    best_params = state.params
    for step in range(1, int(args.steps) + 1):
        stage = _stage_log(f"train step {step:03d}")
        params_before = state.params
        state, metrics = train_step(state, datum)
        stage = _stage_log(f"train step {step:03d}", stage)
        row = {
            "step": float(step),
            "loss": _metric_scalar(metrics, "loss"),
            "omega": _metric_scalar(metrics, "omega"),
            "sr_hf_fraction": _metric_scalar(metrics, "sr_hf_fraction"),
            "lr_hf_fraction": _metric_scalar(metrics, "lr_hf_fraction"),
            "koopmans_ip_mae": _metric_scalar(metrics, "koopmans_ip_mae"),
            "koopmans_lumo_ea_mae": _metric_scalar(metrics, "koopmans_lumo_ea_mae"),
            "koopmans_ea_mae": _metric_scalar(metrics, "koopmans_ea_mae"),
            "grad_norm": _metric_scalar(metrics, "grad_norm"),
            "nonfinite_grad_fraction": _metric_scalar(metrics, "nonfinite_grad_fraction"),
        }
        history.append(row)
        if row["loss"] < best_row["loss"]:
            best_row = dict(row)
            best_params = params_before
        print(
            f"step={step:03d} "
            f"loss={row['loss']:.6e} "
            f"omega={row['omega']:.6f} "
            f"kip={row['koopmans_ip_mae']:.3e} "
            f"klumo={row['koopmans_lumo_ea_mae']:.3e} "
            f"grad={row['grad_norm']:.3e}",
            flush=True,
        )

    stage = _stage_log("final eval_loss")
    final_loss, final_metrics = eval_loss(state.params, datum)
    stage = _stage_log("final eval_loss", stage)
    final_resolved = functional.resolve_parameters(state.params, molecule)
    summary = {
        "system": "ethylene",
        "geometry": ETHYLENE_GEOMETRY.strip(),
        "basis": str(args.basis),
        "grid_level": int(args.grid_level),
        "reference_xc": str(args.reference_xc),
        "reference_backend": str(getattr(molecule, "runtime_scf_backend", None)),
        "rsh_preset": str(args.rsh_preset),
        "rsh_omega_source": str(args.rsh_omega_source),
        "rsh_training_mode": str(args.rsh_training_mode),
        "charged_reference_backend": (
            _charged_reference_backend(str(args.scf_backend))
            if str(args.rsh_training_mode) == "fixed_density"
            else None
        ),
        "omega_grid": [float(value) for value in omega_grid],
        "steps": int(args.steps),
        "learning_rate": float(args.learning_rate),
        "training_config": {
            "scf_runtime_forward_backend": str(args.runtime_forward_backend),
            "implicit_response_backend": str(args.implicit_response_backend),
            "scf_max_cycle": int(args.scf_max_cycle),
            "scf_damping": float(args.scf_damping),
            "scf_level_shift": float(args.scf_level_shift),
            "scf_conv_tol_density": float(args.scf_conv_tol_density),
        },
        "initial": initial_row,
        "best": best_row,
        "final": {
            "loss": float(final_loss),
            "omega": float(final_resolved.omega),
            "sr_hf_fraction": float(final_resolved.sr_hf_fraction),
            "lr_hf_fraction": float(final_resolved.lr_hf_fraction),
            "koopmans_ip_mae": _metric_scalar(final_metrics, "koopmans_ip_mae"),
            "koopmans_lumo_ea_mae": _metric_scalar(final_metrics, "koopmans_lumo_ea_mae"),
            "koopmans_ea_mae": _metric_scalar(final_metrics, "koopmans_ea_mae"),
        },
        "history": history,
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_params_checkpoint(
        outdir / "best_params.msgpack",
        best_params,
        metadata={
            "system": "ethylene",
            "loss": float(best_row["loss"]),
            "omega": float(best_row["omega"]),
        },
    )
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
