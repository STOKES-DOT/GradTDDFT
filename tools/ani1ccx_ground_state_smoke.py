from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (str(_REPO_ROOT), str(_SRC_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import h5py
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax

from td_graddft import neural_xc
from td_graddft.data.reference import restricted_reference_from_pyscf
from td_graddft.training import (
    GroundStateDatum,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    make_ground_state_train_step,
    save_params_checkpoint,
)
from td_graddft.training.targets import ground_state_mse_loss


Z_TO_SYMBOL = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
}


@dataclass(frozen=True)
class AniSample:
    name: str
    source_formula: str
    source_index: int
    atomic_numbers: np.ndarray
    coordinates_angstrom: np.ndarray
    ccsdt_cbs_energy: float

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_formula": self.source_formula,
            "source_index": int(self.source_index),
            "atomic_numbers": [int(z) for z in self.atomic_numbers.tolist()],
            "natoms": int(self.atomic_numbers.shape[0]),
            "nelectrons": int(np.sum(self.atomic_numbers)),
            "ccsd_t_cbs_energy_hartree": float(self.ccsdt_cbs_energy),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small ANI-1ccx ground-state training smoke test.",
    )
    parser.add_argument("--data", required=True, help="ANI-1x/ANI-1ccx HDF5 or small subset HDF5.")
    parser.add_argument("--outdir", default="outputs/ani1ccx_ground_state_smoke30")
    parser.add_argument("--num-molecules", type=int, default=30)
    parser.add_argument("--max-atoms", type=int, default=8)
    parser.add_argument("--max-conformers-per-formula", type=int, default=1)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--xc", default="pbe", help="PySCF reference XC used to build the starting density.")
    parser.add_argument(
        "--semilocal-xc",
        default="lda_x,lda_c_pw",
        help="Comma-separated local/semilocal basis channels for the neural XC functional.",
    )
    parser.add_argument(
        "--network-architecture",
        choices=("simple_mlp", "residual", "graddft_residual", "mlp"),
        default="simple_mlp",
    )
    parser.add_argument("--hidden-dims", default="16,16")
    parser.add_argument(
        "--input-feature-mode",
        choices=("enhanced", "dm21_original"),
        default="enhanced",
    )
    parser.add_argument(
        "--hf-input-mode",
        choices=("total_only", "spin_resolved"),
        default="spin_resolved",
    )
    parser.add_argument(
        "--response-hf-mode",
        choices=("approx", "strict"),
        default=neural_xc.DEFAULT_NEURAL_XC_RESPONSE_HF_MODE,
    )
    parser.add_argument(
        "--response-pt2-mode",
        choices=("approx", "strict"),
        default="approx",
    )
    parser.add_argument("--dm21-hfx-channels", type=int, default=2)
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument(
        "--reference-backend",
        choices=("pyscf",),
        default="pyscf",
        help="Reference SCF path.",
    )
    parser.add_argument("--pyscf-conv-tol", type=float, default=1e-9)
    parser.add_argument("--pyscf-max-cycle", type=int, default=80)
    parser.add_argument("--jax-rks-conv-tol-density", type=float, default=1e-8)
    parser.add_argument("--jax-rks-damping", type=float, default=0.0)
    parser.add_argument("--jax-rks-direct-scf-tol", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--energy-normalization",
        choices=("none", "per_atom", "per_electron"),
        default="per_atom",
    )
    parser.add_argument(
        "--training-mode",
        choices=("fixed_density", "self_consistent"),
        default="self_consistent",
    )
    parser.add_argument("--scf-max-cycle", type=int, default=4)
    parser.add_argument("--scf-damping", type=float, default=0.25)
    parser.add_argument("--scf-conv-tol-density", type=float, default=1e-8)
    parser.add_argument(
        "--scf-iterate-selection",
        choices=("final", "best_rms", "first_converged"),
        default="first_converged",
    )
    parser.add_argument(
        "--scf-require-convergence",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--scf-gradient-mode",
        choices=("unrolled", "implicit_commutator"),
        default="implicit_commutator",
    )
    parser.add_argument("--scf-implicit-diff-max-iter", type=int, default=24)
    parser.add_argument("--scf-implicit-diff-tolerance", type=float, default=1e-6)
    parser.add_argument("--scf-implicit-diff-regularization", type=float, default=1e-3)
    parser.add_argument("--omega-grid", default="0.0,0.4")
    parser.add_argument("--hfx-chunk-size", type=int, default=512)
    parser.add_argument(
        "--training-eri-storage",
        choices=("packed", "full"),
        default="packed",
        help=(
            "How AO ERIs are stored on training references. 'packed' uses "
            "aosym='s4' AO-pair ERIs so differentiable SCF takes the packed "
            "GPU JK path; 'full' preserves the legacy 4D tensor."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one omega value.")
    return values


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _parse_str_tuple(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _formula_sort_key(name: str) -> tuple[int, str]:
    def _count_after(symbol: str) -> int:
        idx = name.find(symbol)
        if idx < 0:
            return 0
        idx += len(symbol)
        digits = []
        while idx < len(name) and name[idx].isdigit():
            digits.append(name[idx])
            idx += 1
        return int("".join(digits) or "1")

    natoms = _count_after("C") + _count_after("H") + _count_after("N") + _count_after("O")
    return natoms, name


def _valid_energy_indices(energies: np.ndarray) -> np.ndarray:
    flat = np.asarray(energies, dtype=np.float64).reshape(-1)
    return np.flatnonzero(np.isfinite(flat))


def _read_flat_subset(handle: h5py.File, *, limit: int, max_atoms: int) -> list[AniSample]:
    samples: list[AniSample] = []
    root = handle["samples"]
    for name in sorted(root.keys()):
        group = root[name]
        atomic_numbers = np.asarray(group["atomic_numbers"], dtype=np.int64)
        if atomic_numbers.shape[0] > max_atoms:
            continue
        if any(int(z) not in Z_TO_SYMBOL for z in atomic_numbers):
            continue
        if int(np.sum(atomic_numbers)) % 2 != 0:
            continue
        coordinates = np.asarray(group["coordinates"], dtype=np.float64)
        if coordinates.ndim == 3:
            coordinates = coordinates[0]
        energy = float(np.asarray(group["ccsd(t)_cbs.energy"], dtype=np.float64).reshape(-1)[0])
        if not math.isfinite(energy):
            continue
        samples.append(
            AniSample(
                name=str(name),
                source_formula=str(group.attrs.get("source_formula", name)),
                source_index=int(group.attrs.get("source_index", 0)),
                atomic_numbers=atomic_numbers,
                coordinates_angstrom=coordinates,
                ccsdt_cbs_energy=energy,
            )
        )
        if len(samples) >= limit:
            break
    return samples


def read_ani_samples(
    path: Path,
    *,
    limit: int,
    max_atoms: int,
    max_conformers_per_formula: int,
) -> list[AniSample]:
    samples: list[AniSample] = []
    with h5py.File(path, "r") as handle:
        if "samples" in handle:
            return _read_flat_subset(handle, limit=limit, max_atoms=max_atoms)
        for formula in sorted(handle.keys(), key=_formula_sort_key):
            group = handle[formula]
            required = {"atomic_numbers", "coordinates", "ccsd(t)_cbs.energy"}
            if not required.issubset(group.keys()):
                continue
            atomic_numbers = np.asarray(group["atomic_numbers"], dtype=np.int64)
            if atomic_numbers.shape[0] > max_atoms:
                continue
            if any(int(z) not in Z_TO_SYMBOL for z in atomic_numbers):
                continue
            if int(np.sum(atomic_numbers)) % 2 != 0:
                continue
            valid_indices = _valid_energy_indices(np.asarray(group["ccsd(t)_cbs.energy"]))
            for source_index in valid_indices[:max_conformers_per_formula]:
                energy = float(group["ccsd(t)_cbs.energy"][source_index])
                coordinates = np.asarray(group["coordinates"][source_index], dtype=np.float64)
                samples.append(
                    AniSample(
                        name=f"{formula}_{int(source_index):06d}",
                        source_formula=str(formula),
                        source_index=int(source_index),
                        atomic_numbers=atomic_numbers,
                        coordinates_angstrom=coordinates,
                        ccsdt_cbs_energy=energy,
                    )
                )
                if len(samples) >= limit:
                    return samples
    return samples


def build_reference(
    sample: AniSample,
    *,
    basis: str,
    xc: str,
    grid_level: int,
    pyscf_conv_tol: float,
    pyscf_max_cycle: int,
    omega_grid: tuple[float, ...],
    hfx_chunk_size: int,
    training_eri_storage: str,
    reference_backend: str = "pyscf",
    jax_rks_conv_tol_density: float = 1e-8,
    jax_rks_damping: float = 0.0,
    jax_rks_direct_scf_tol: float = 0.0,
):
    atom = [
        (Z_TO_SYMBOL[int(z)], tuple(float(x) for x in xyz))
        for z, xyz in zip(sample.atomic_numbers, sample.coordinates_angstrom, strict=True)
    ]
    from pyscf import dft, gto

    reference_backend = str(reference_backend)
    mol = gto.M(
        atom=atom,
        unit="Angstrom",
        basis=basis,
        charge=0,
        spin=0,
        cart=False,
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = int(grid_level)
    mf.conv_tol = float(pyscf_conv_tol)
    mf.max_cycle = int(pyscf_max_cycle)
    energy = mf.kernel()
    if not np.isfinite(float(energy)):
        raise RuntimeError(f"PySCF returned non-finite energy for {sample.name}.")
    ref = restricted_reference_from_pyscf(
        mf,
        compute_local_hfx_features=True,
        compute_local_hfx_aux=True,
        compute_local_pt2_features=True,
        hfx_omega_values=omega_grid,
        hfx_chunk_size=int(hfx_chunk_size),
    )
    if str(training_eri_storage) == "packed":
        eri_pair_matrix = jnp.asarray(mf.mol.intor("int2e", aosym="s4"), dtype=ref.h1e.dtype)
        ref = replace(
            ref,
            rep_tensor=jnp.zeros((0, 0, 0, 0), dtype=ref.h1e.dtype),
            eri_pair_matrix=eri_pair_matrix,
        )
    return ref, bool(mf.converged), float(energy)


def _metric_mean(metrics: dict[str, Any], key: str) -> float:
    if key not in metrics:
        return float("nan")
    arr = np.asarray(jax.device_get(metrics[key]), dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def _metric_first(metrics: dict[str, Any], key: str) -> float:
    if key not in metrics:
        return float("nan")
    arr = np.asarray(jax.device_get(metrics[key]), dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float("nan")
    return float(arr[0])


def _functional_record(functional: Any, params: Any, molecule: Any) -> dict[str, float]:
    try:
        effective_hf = float(jax.device_get(functional.effective_exchange_fraction(params, molecule)))
    except Exception:
        effective_hf = float("nan")
    try:
        e_xc = float(jax.device_get(functional.energy_from_molecule(params, molecule)))
    except Exception:
        e_xc = float("nan")
    return {
        "effective_exchange_fraction_first": effective_hf,
        "xc_energy_first_hartree": e_xc,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    omega_grid = _parse_float_tuple(args.omega_grid)
    semilocal_xc = _parse_str_tuple(str(args.semilocal_xc))
    hidden_dims = _parse_int_tuple(str(args.hidden_dims))

    print(f"[load] data={data_path}", flush=True)
    samples = read_ani_samples(
        data_path,
        limit=int(args.num_molecules),
        max_atoms=int(args.max_atoms),
        max_conformers_per_formula=int(args.max_conformers_per_formula),
    )
    if len(samples) != int(args.num_molecules):
        raise RuntimeError(
            f"Requested {args.num_molecules} samples but found {len(samples)} "
            f"with max_atoms={args.max_atoms}."
        )
    _write_json(outdir / "samples.json", [sample.metadata() for sample in samples])
    print(
        f"[load] selected={len(samples)} max_atoms={max(s.atomic_numbers.shape[0] for s in samples)}",
        flush=True,
    )

    references = []
    reference_records = []
    t0 = time.time()
    for index, sample in enumerate(samples, start=1):
        ref, converged, pyscf_energy = build_reference(
            sample,
            basis=str(args.basis),
            xc=str(args.xc),
            grid_level=int(args.grid_level),
            pyscf_conv_tol=float(args.pyscf_conv_tol),
            pyscf_max_cycle=int(args.pyscf_max_cycle),
            omega_grid=omega_grid,
            hfx_chunk_size=int(args.hfx_chunk_size),
            training_eri_storage=str(args.training_eri_storage),
            reference_backend=str(args.reference_backend),
            jax_rks_conv_tol_density=float(args.jax_rks_conv_tol_density),
            jax_rks_damping=float(args.jax_rks_damping),
            jax_rks_direct_scf_tol=float(args.jax_rks_direct_scf_tol),
        )
        references.append(ref)
        reference_records.append(
            {
                **sample.metadata(),
                "pyscf_converged": bool(converged),
                "pyscf_energy_hartree": float(pyscf_energy),
                "pyscf_minus_ccsd_t_cbs_hartree": float(pyscf_energy - sample.ccsdt_cbs_energy),
                "reference_backend": str(args.reference_backend),
                "reference_energy_hartree": float(pyscf_energy),
                "reference_minus_ccsd_t_cbs_hartree": float(pyscf_energy - sample.ccsdt_cbs_energy),
                "grid_points": int(ref.grid.weights.shape[0]),
                "nao": int(ref.h1e.shape[0]),
                "has_hfx_local": getattr(ref, "hfx_local", None) is not None,
                "has_pt2_local": getattr(ref, "pt2_local", None) is not None,
                "training_eri_storage": str(args.training_eri_storage),
                "has_eri_pair_matrix": getattr(ref, "eri_pair_matrix", None) is not None,
                "rep_tensor_size": int(np.asarray(jax.device_get(ref.rep_tensor)).size),
            }
        )
        print(
            f"[ref] {index:02d}/{len(samples):02d} {sample.name} "
            f"natoms={sample.atomic_numbers.shape[0]} nao={ref.h1e.shape[0]} "
            f"grid={ref.grid.weights.shape[0]} conv={int(converged)} "
            f"ani={sample.ccsdt_cbs_energy:.8f} pyscf={pyscf_energy:.8f}",
            flush=True,
        )
    _write_json(outdir / "references.json", reference_records)
    ref_elapsed_s = time.time() - t0
    print(f"[ref] done elapsed_s={ref_elapsed_s:.2f}", flush=True)

    data = [
        GroundStateDatum(
            molecule=ref,
            target_total_energy=jnp.asarray(sample.ccsdt_cbs_energy, dtype=jnp.float64),
        )
        for ref, sample in zip(references, samples, strict=True)
    ]
    functional = neural_xc.Functional(
        architecture=str(args.network_architecture),
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
        name="ani1ccx_local_density_hf_pt2_xc",
    )
    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(int(args.seed)),
        references[0],
        optax.adam(float(args.learning_rate)),
    )
    training_config = GroundStateTrainingConfig(
        mode=str(args.training_mode),
        energy_mse_weight=0.0,
        energy_mae_weight=1.0,
        energy_normalization=str(args.energy_normalization),
        scf_max_cycle=int(args.scf_max_cycle),
        scf_damping=float(args.scf_damping),
        scf_conv_tol_density=float(args.scf_conv_tol_density),
        scf_iterate_selection=str(args.scf_iterate_selection),
        scf_gradient_mode=str(args.scf_gradient_mode),
        scf_require_convergence=bool(args.scf_require_convergence),
        scf_implicit_diff_max_iter=int(args.scf_implicit_diff_max_iter),
        scf_implicit_diff_tolerance=float(args.scf_implicit_diff_tolerance),
        scf_implicit_diff_regularization=float(args.scf_implicit_diff_regularization),
    )
    train_step = make_ground_state_train_step(
        functional,
        training_config=training_config,
    )
    history_path = outdir / "history.jsonl"
    if history_path.exists():
        history_path.unlink()

    t_initial_eval = time.time()
    initial_loss, initial_metrics = ground_state_mse_loss(
        state.params,
        functional,
        data,
        training_config=training_config,
    )
    initial_eval_elapsed_s = time.time() - t_initial_eval
    initial_record = {
        "step": 0,
        "loss": float(jax.device_get(initial_loss)),
        "eval_elapsed_s": float(initial_eval_elapsed_s),
        "energy_mae_hartree_mean": _metric_mean(initial_metrics, "energy_mae"),
        "normalized_energy_mae_mean": _metric_mean(initial_metrics, "normalized_energy_mae"),
        **_functional_record(functional, state.params, references[0]),
    }
    with history_path.open("a") as handle:
        handle.write(json.dumps(initial_record, sort_keys=True) + "\n")
    print(
        f"[train] step=000 loss={initial_record['loss']:.8e} "
        f"energy_mae={initial_record['energy_mae_hartree_mean']:.8e} "
        f"norm_mae={initial_record['normalized_energy_mae_mean']:.8e} "
        f"eff_hf={initial_record['effective_exchange_fraction_first']:.4f}",
        flush=True,
    )

    last_record = initial_record
    t_train = time.time()
    for step in range(1, int(args.steps) + 1):
        t_step = time.time()
        state, metrics = train_step(state, data)
        step_elapsed_s = time.time() - t_step
        record = {
            "step": int(step),
            "loss": _metric_first(metrics, "loss"),
            "step_elapsed_s": float(step_elapsed_s),
            "energy_mae_hartree_mean": _metric_mean(metrics, "energy_mae"),
            "normalized_energy_mae_mean": _metric_mean(metrics, "normalized_energy_mae"),
            "grad_norm": _metric_first(metrics, "grad_norm"),
            "grad_abs_max": _metric_first(metrics, "grad_abs_max"),
            "param_update_norm": _metric_first(metrics, "param_update_norm"),
            "nonfinite_grad_fraction": _metric_first(metrics, "nonfinite_grad_fraction"),
            "scf_converged_fraction": _metric_first(metrics, "scf_converged_fraction"),
            "scf_cycles_mean": _metric_first(metrics, "scf_cycles_mean"),
            "scf_selected_rms_max": _metric_first(metrics, "scf_selected_rms_max"),
            **_functional_record(functional, state.params, references[0]),
        }
        last_record = record
        with history_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if step == 1 or step == int(args.steps) or step % max(int(args.log_every), 1) == 0:
            print(
                f"[train] step={step:03d} loss={record['loss']:.8e} "
                f"energy_mae={record['energy_mae_hartree_mean']:.8e} "
                f"norm_mae={record['normalized_energy_mae_mean']:.8e} "
                f"grad={record['grad_norm']:.3e} "
                f"scf={record['scf_converged_fraction']:.2f} "
                f"cyc={record['scf_cycles_mean']:.1f} "
                f"dt={record['step_elapsed_s']:.1f}s "
                f"eff_hf={record['effective_exchange_fraction_first']:.4f}",
                flush=True,
            )

    t_final_eval = time.time()
    final_loss, final_metrics = ground_state_mse_loss(
        state.params,
        functional,
        data,
        training_config=training_config,
    )
    final_eval_elapsed_s = time.time() - t_final_eval
    final_record = {
        "loss": float(jax.device_get(final_loss)),
        "eval_elapsed_s": float(final_eval_elapsed_s),
        "energy_mae_hartree_mean": _metric_mean(final_metrics, "energy_mae"),
        "normalized_energy_mae_mean": _metric_mean(final_metrics, "normalized_energy_mae"),
        "scf_converged_fraction": _metric_first(final_metrics, "scf_converged_fraction"),
        "scf_cycles_mean": _metric_first(final_metrics, "scf_cycles_mean"),
        "scf_selected_rms_max": _metric_first(final_metrics, "scf_selected_rms_max"),
        **_functional_record(functional, state.params, references[0]),
    }
    summary = {
        "data_path": str(data_path),
        "outdir": str(outdir),
        "num_molecules": len(samples),
        "basis": str(args.basis),
        "xc": str(args.xc),
        "semilocal_xc": list(semilocal_xc),
        "network_architecture": str(args.network_architecture),
        "hidden_dims": list(hidden_dims),
        "energy_mode": "graddft_coeff_basis_hf_pt2_heads",
        "include_pt2_channel": True,
        "input_feature_mode": str(args.input_feature_mode),
        "hf_input_mode": str(args.hf_input_mode),
        "response_hf_mode": str(args.response_hf_mode),
        "response_pt2_mode": str(args.response_pt2_mode),
        "omega_grid": list(omega_grid),
        "training_eri_storage": str(args.training_eri_storage),
        "reference_backend": str(args.reference_backend),
        "jax_rks_conv_tol_density": float(args.jax_rks_conv_tol_density),
        "jax_rks_damping": float(args.jax_rks_damping),
        "jax_rks_direct_scf_tol": float(args.jax_rks_direct_scf_tol),
        "grid_level": int(args.grid_level),
        "training_mode": str(args.training_mode),
        "scf_gradient_mode": str(args.scf_gradient_mode),
        "scf_max_cycle": int(args.scf_max_cycle),
        "scf_damping": float(args.scf_damping),
        "scf_conv_tol_density": float(args.scf_conv_tol_density),
        "scf_iterate_selection": str(args.scf_iterate_selection),
        "scf_require_convergence": bool(args.scf_require_convergence),
        "scf_implicit_diff_max_iter": int(args.scf_implicit_diff_max_iter),
        "scf_implicit_diff_tolerance": float(args.scf_implicit_diff_tolerance),
        "scf_implicit_diff_regularization": float(args.scf_implicit_diff_regularization),
        "steps": int(args.steps),
        "learning_rate": float(args.learning_rate),
        "energy_normalization": str(args.energy_normalization),
        "initial_eval_elapsed_s": float(initial_eval_elapsed_s),
        "final_eval_elapsed_s": float(final_eval_elapsed_s),
        "initial": initial_record,
        "last_step_metrics": last_record,
        "final_eval": final_record,
        "reference_build_elapsed_s": float(ref_elapsed_s),
        "training_elapsed_s": float(time.time() - t_train),
        "pyscf_converged_fraction": float(
            np.mean([record["pyscf_converged"] for record in reference_records])
        ),
        "jax_devices": [str(device) for device in jax.devices()],
    }
    _write_json(outdir / "summary.json", summary)
    if bool(args.save_checkpoint):
        save_params_checkpoint(
            outdir / "final_params.msgpack",
            state.params,
            metadata={
                "script": Path(__file__).name,
                "num_molecules": len(samples),
                "basis": str(args.basis),
                "xc": str(args.xc),
                "semilocal_xc": list(semilocal_xc),
                "energy_mode": "graddft_coeff_basis_hf_pt2_heads",
                "include_pt2_channel": True,
                "scf_gradient_mode": str(args.scf_gradient_mode),
                "steps": int(args.steps),
            },
        )
    print(
        f"[done] final_loss={final_record['loss']:.8e} "
        f"final_energy_mae={final_record['energy_mae_hartree_mean']:.8e} "
        f"summary={outdir / 'summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
