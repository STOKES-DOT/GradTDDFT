from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (str(_SRC_ROOT), str(_REPO_ROOT), str(_REPO_ROOT / "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax

import ani1ccx_ground_state_smoke as smoke
from td_graddft import neural_xc
from td_graddft.training import (
    GroundStateDatum,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    make_ground_state_train_step,
)
from td_graddft.training.targets import ground_state_mse_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-reference-build ANI-1ccx ground-state implicit-SCF efficiency scan.",
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--outdir", default="outputs/ani1ccx_ground_state_efficiency_scan")
    parser.add_argument("--num-molecules", type=int, default=30)
    parser.add_argument("--max-atoms", type=int, default=8)
    parser.add_argument("--max-conformers-per-formula", type=int, default=1)
    parser.add_argument("--basis", default="6-31+g*")
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument("--pyscf-conv-tol", type=float, default=1e-9)
    parser.add_argument("--pyscf-max-cycle", type=int, default=80)
    parser.add_argument("--training-eri-storage", choices=("packed", "full"), default="packed")
    parser.add_argument("--semilocal-xc", default="lda_x,lda_c_pw")
    parser.add_argument("--hidden-dims", default="16,16")
    parser.add_argument("--input-feature-mode", choices=("enhanced", "dm21_original"), default="enhanced")
    parser.add_argument("--hf-input-mode", choices=("total_only", "spin_resolved"), default="spin_resolved")
    parser.add_argument(
        "--response-hf-mode",
        choices=("approx", "strict"),
        default="strict",
    )
    parser.add_argument(
        "--response-pt2-mode",
        choices=("approx", "strict"),
        default="approx",
    )
    parser.add_argument("--dm21-hfx-channels", type=int, default=2)
    parser.add_argument("--omega-grid", default="0.0,0.4")
    parser.add_argument("--hfx-chunk-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scf-damping", type=float, default=0.35)
    parser.add_argument("--scf-conv-tol-density", type=float, default=1e-8)
    parser.add_argument(
        "--scf-iterate-selection",
        choices=("final", "best_rms", "first_converged"),
        default="best_rms",
    )
    parser.add_argument("--scf-implicit-diff-tolerance", type=float, default=1e-6)
    parser.add_argument("--scf-implicit-diff-regularization", type=float, default=1e-3)
    parser.add_argument(
        "--scan-configs",
        default="c32_i32,c24_i16,c20_i12",
        help="Comma-separated configs of form c<scf_max_cycle>_i<implicit_diff_max_iter>.",
    )
    return parser.parse_args()


def _parse_scan_config(raw: str) -> dict[str, int | str]:
    name = raw.strip()
    parts = name.split("_")
    if len(parts) != 2 or not parts[0].startswith("c") or not parts[1].startswith("i"):
        raise ValueError(f"Invalid scan config {raw!r}; expected c32_i32.")
    return {
        "name": name,
        "scf_max_cycle": int(parts[0][1:]),
        "implicit_diff_max_iter": int(parts[1][1:]),
    }


def _metric_record(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "loss": smoke._metric_first(metrics, "loss"),
        "energy_mae_hartree_mean": smoke._metric_mean(metrics, "energy_mae"),
        "normalized_energy_mae_mean": smoke._metric_mean(metrics, "normalized_energy_mae"),
        "grad_norm": smoke._metric_first(metrics, "grad_norm"),
        "grad_abs_max": smoke._metric_first(metrics, "grad_abs_max"),
        "param_update_norm": smoke._metric_first(metrics, "param_update_norm"),
        "nonfinite_grad_fraction": smoke._metric_first(metrics, "nonfinite_grad_fraction"),
        "scf_converged_fraction": smoke._metric_first(metrics, "scf_converged_fraction"),
        "scf_cycles_mean": smoke._metric_first(metrics, "scf_cycles_mean"),
        "scf_selected_rms_max": smoke._metric_first(metrics, "scf_selected_rms_max"),
    }


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    omega_grid = smoke._parse_float_tuple(args.omega_grid)
    semilocal_xc = smoke._parse_str_tuple(args.semilocal_xc)
    hidden_dims = smoke._parse_int_tuple(args.hidden_dims)
    scan_configs = [_parse_scan_config(part) for part in str(args.scan_configs).split(",") if part.strip()]

    print(f"[load] data={data_path}", flush=True)
    samples = smoke.read_ani_samples(
        data_path,
        limit=int(args.num_molecules),
        max_atoms=int(args.max_atoms),
        max_conformers_per_formula=int(args.max_conformers_per_formula),
    )
    if len(samples) != int(args.num_molecules):
        raise RuntimeError(f"Requested {args.num_molecules} samples but found {len(samples)}.")
    smoke._write_json(outdir / "samples.json", [sample.metadata() for sample in samples])

    references = []
    reference_records = []
    t_ref = time.time()
    for index, sample in enumerate(samples, start=1):
        ref, converged, pyscf_energy = smoke.build_reference(
            sample,
            basis=str(args.basis),
            xc=str(args.xc),
            grid_level=int(args.grid_level),
            pyscf_conv_tol=float(args.pyscf_conv_tol),
            pyscf_max_cycle=int(args.pyscf_max_cycle),
            omega_grid=omega_grid,
            hfx_chunk_size=int(args.hfx_chunk_size),
            training_eri_storage=str(args.training_eri_storage),
        )
        references.append(ref)
        reference_records.append(
            {
                **sample.metadata(),
                "pyscf_converged": bool(converged),
                "pyscf_energy_hartree": float(pyscf_energy),
                "pyscf_minus_ccsd_t_cbs_hartree": float(pyscf_energy - sample.ccsdt_cbs_energy),
                "grid_points": int(ref.grid.weights.shape[0]),
                "nao": int(ref.h1e.shape[0]),
                "training_eri_storage": str(args.training_eri_storage),
                "has_eri_pair_matrix": getattr(ref, "eri_pair_matrix", None) is not None,
                "rep_tensor_size": int(np.asarray(jax.device_get(ref.rep_tensor)).size),
            }
        )
        print(
            f"[ref] {index:02d}/{len(samples):02d} {sample.name} "
            f"nao={ref.h1e.shape[0]} conv={int(converged)}",
            flush=True,
        )
    ref_elapsed_s = time.time() - t_ref
    smoke._write_json(outdir / "references.json", reference_records)
    print(f"[ref] done elapsed_s={ref_elapsed_s:.2f}", flush=True)

    data = [
        GroundStateDatum(
            molecule=ref,
            target_total_energy=jnp.asarray(sample.ccsdt_cbs_energy, dtype=jnp.float64),
        )
        for ref, sample in zip(references, samples, strict=True)
    ]

    functional = neural_xc.Functional(
        architecture="simple_mlp",
        semilocal_xc=semilocal_xc,
        energy_mode="graddft_coeff_basis_hf_pt2_heads",
        input_feature_mode=str(args.input_feature_mode),
        hf_input_mode=str(args.hf_input_mode),
        include_pt2_channel=True,
        pt2_channel_mode="local_exact",
        response_hf_mode=str(args.response_hf_mode),
        response_pt2_mode=str(args.response_pt2_mode),
        strict_dm21_feature_alignment=True,
        hidden_dims=hidden_dims,
        dm21_hfx_channels=int(args.dm21_hfx_channels),
        name="ani1ccx_efficiency_scan_local_density_hf_pt2_xc",
    )

    results_path = outdir / "scan_results.jsonl"
    if results_path.exists():
        results_path.unlink()
    results = []
    for cfg in scan_configs:
        state = create_train_state_from_molecule(
            functional,
            jax.random.PRNGKey(int(args.seed)),
            references[0],
            optax.adam(float(args.learning_rate)),
        )
        training_config = GroundStateTrainingConfig(
            mode="self_consistent",
            energy_mse_weight=0.0,
            energy_mae_weight=1.0,
            energy_normalization="none",
            scf_max_cycle=int(cfg["scf_max_cycle"]),
            scf_damping=float(args.scf_damping),
            scf_conv_tol_density=float(args.scf_conv_tol_density),
            scf_iterate_selection=str(args.scf_iterate_selection),
            scf_gradient_mode="implicit_commutator",
            scf_require_convergence=False,
            scf_implicit_diff_max_iter=int(cfg["implicit_diff_max_iter"]),
            scf_implicit_diff_tolerance=float(args.scf_implicit_diff_tolerance),
            scf_implicit_diff_regularization=float(args.scf_implicit_diff_regularization),
        )
        train_step = make_ground_state_train_step(functional, training_config=training_config)

        t_initial = time.time()
        initial_loss, initial_metrics = ground_state_mse_loss(
            state.params,
            functional,
            data,
            training_config=training_config,
        )
        initial_elapsed_s = time.time() - t_initial

        t_step = time.time()
        state, metrics = train_step(state, data)
        step_elapsed_s = time.time() - t_step

        record = {
            "config": str(cfg["name"]),
            "scf_max_cycle": int(cfg["scf_max_cycle"]),
            "scf_implicit_diff_max_iter": int(cfg["implicit_diff_max_iter"]),
            "initial_loss": float(jax.device_get(initial_loss)),
            "initial_energy_mae_hartree_mean": smoke._metric_mean(initial_metrics, "energy_mae"),
            "initial_scf_converged_fraction": smoke._metric_first(initial_metrics, "scf_converged_fraction"),
            "initial_scf_cycles_mean": smoke._metric_first(initial_metrics, "scf_cycles_mean"),
            "initial_scf_selected_rms_max": smoke._metric_first(initial_metrics, "scf_selected_rms_max"),
            "initial_eval_elapsed_s": float(initial_elapsed_s),
            "step_elapsed_s": float(step_elapsed_s),
            **_metric_record(metrics),
        }
        results.append(record)
        with results_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            f"[scan] {record['config']} init={record['initial_eval_elapsed_s']:.1f}s "
            f"step={record['step_elapsed_s']:.1f}s loss={record['loss']:.8e} "
            f"grad={record['grad_norm']:.3e} scf={record['scf_converged_fraction']:.2f} "
            f"cyc={record['scf_cycles_mean']:.1f} rms={record['scf_selected_rms_max']:.3e}",
            flush=True,
        )

    summary = {
        "data_path": str(data_path),
        "outdir": str(outdir),
        "num_molecules": len(samples),
        "basis": str(args.basis),
        "xc": str(args.xc),
        "grid_level": int(args.grid_level),
        "training_eri_storage": str(args.training_eri_storage),
        "scf_gradient_mode": "implicit_commutator",
        "reference_build_elapsed_s": float(ref_elapsed_s),
        "pyscf_converged_fraction": float(np.mean([r["pyscf_converged"] for r in reference_records])),
        "scan_configs": scan_configs,
        "results": results,
        "jax_devices": [str(device) for device in jax.devices()],
    }
    smoke._write_json(outdir / "summary.json", summary)
    print(f"[done] summary={outdir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
