from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import time
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from pyscf import dft, gto

from td_graddft import neural_xc
from td_graddft.data.hdf5_cache import read_restricted_molecule, write_restricted_molecule
from td_graddft.xc_backend.jax_libxc import b3lyp_component_basis
from td_graddft.neural_xc import (
    DEFAULT_INPUT_FEATURE_MODE,
    DEFAULT_NETWORK_ARCHITECTURE,
    DEFAULT_NETWORK_HIDDEN_DIMS,
    DEFAULT_NEURAL_XC_RESPONSE_HF_MODE,
)
from td_graddft.data.reference import restricted_reference_from_pyscf
from td_graddft.neural_xc.inputs import ChunkedHFXNu
from td_graddft.spectra import HARTREE_TO_EV
from td_graddft.training import (
    ExcitedStateDatum,
    GroundStateCoreDatum,
    GroundStateDatum,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    ground_state_mse_loss,
    make_ground_state_loss_and_grad,
    make_ground_state_train_step,
    predict_excitation_energies,
    predict_ground_state_molecule,
    predict_ground_state_total_energy,
    load_params_checkpoint,
    save_params_checkpoint,
)


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"[{_timestamp()}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


@dataclass(frozen=True)
class ReferenceRow:
    system: str
    split: str
    atom: str
    unit: str
    charge: int
    spin: int
    basis: str
    ccsd_total_energy_h: float
    s1_excitation_h: float


@dataclass(frozen=True)
class PreparedReference:
    row: ReferenceRow
    molecule: Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train the neural functional on a reusable closed-shell EOM-EE-CCSD S1 CSV, "
            "using self-consistent or fixed-density mode, then report train/validation "
            "and benzene generalization errors."
        )
    )
    p.add_argument("--reference-csv", required=True)
    p.add_argument("--basis", default="6-31g")
    p.add_argument("--xc", default="b3lyp")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--lr-decay-every", type=int, default=500)
    p.add_argument("--lr-decay-factor", type=float, default=0.5)
    p.add_argument("--training-mode", choices=("fixed_density", "self_consistent"), default="self_consistent")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=list(DEFAULT_NETWORK_HIDDEN_DIMS))
    p.add_argument(
        "--network-architecture",
        choices=("simple_mlp", "graddft_residual"),
        default=DEFAULT_NETWORK_ARCHITECTURE,
    )
    p.add_argument(
        "--input-feature-mode",
        choices=("enhanced", "canonical", "dm21_original"),
        default=DEFAULT_INPUT_FEATURE_MODE,
    )
    p.add_argument("--include-pt2-channel", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--include-hfx-channel", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--response-hf-mode",
        choices=("approx", "strict"),
        default=DEFAULT_NEURAL_XC_RESPONSE_HF_MODE,
        help=(
            "Excited-state handling of the neural local-HF channel. 'approx' averages "
            "the grid HF coefficient into a scalar hybrid fraction; 'strict' is gated "
            "until chi/fxx second-response contractions are implemented."
        ),
    )
    p.add_argument(
        "--pt2-channel-mode",
        choices=("scaled_projected", "local_exact"),
        default="scaled_projected",
    )
    p.add_argument(
        "--scf-hfx-grid-block-size",
        dest="scf_hfx_grid_block_size",
        type=int,
        default=1024,
        help="Grid block size for SCF local-HF HFX nu cache reads and Fock contractions.",
    )
    p.add_argument("--semilocal-xc", nargs="+", default=list(b3lyp_component_basis()))
    p.add_argument("--s1-weight", type=float, default=1.0)
    p.add_argument("--s1-use-tda", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-use-tda", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--energy-mse-weight", type=float, default=0.0)
    p.add_argument("--energy-mae-weight", type=float, default=0.0)
    p.add_argument("--density-constraint-weight", type=float, default=0.0)
    p.add_argument("--grids-level", type=int, default=0)
    p.add_argument("--reference-scf-max-cycle", type=int, default=100)
    p.add_argument("--reference-scf-conv-tol", type=float, default=1e-10)
    p.add_argument("--reference-jk-backend", choices=("full", "df"), default="full")
    p.add_argument(
        "--response-df-mode",
        choices=("none", "df", "ris"),
        default="none",
        help="Optional response-factor cache content generated with each reference molecule.",
    )
    p.add_argument(
        "--response-two-electron-mode",
        choices=("auto", "direct", "df", "ris"),
        default="auto",
        help="Two-electron backend used by the TDDFT/TDA response kernel during training.",
    )
    p.add_argument("--response-ris-theta", type=float, default=0.2)
    p.add_argument("--response-ris-j-fit", choices=("s", "sp", "spd"), default="sp")
    p.add_argument("--response-ris-k-fit", choices=("s", "sp", "spd"), default="s")
    p.add_argument("--response-ris-aux-chunk-size", type=int, default=256)
    p.add_argument(
        "--reference-cache",
        default="outputs/reference_cache/closed_shell_s1_references.h5",
        help=(
            "HDF5 cache for prepared RKS/HFX reference molecules. Pass an empty "
            "string to disable."
        ),
    )
    p.add_argument("--rebuild-reference-cache", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--train-scf-max-cycle", type=int, default=16)
    p.add_argument("--train-scf-damping", type=float, default=0.25)
    p.add_argument("--train-scf-conv-tol-density", type=float, default=1e-8)
    p.add_argument("--train-scf-vxc-clip", type=float, default=20.0)
    p.add_argument(
        "--scf-iterate-selection",
        choices=("final", "best_rms", "first_converged"),
        default="best_rms",
    )
    p.add_argument(
        "--scf-gradient-mode",
        choices=("unrolled", "implicit_commutator"),
        default="implicit_commutator",
    )
    p.add_argument(
        "--scf-implicit-diff-solver",
        choices=("normal_cg", "gmres", "bicgstab"),
        default="normal_cg",
    )
    p.add_argument("--scf-implicit-diff-tolerance", type=float, default=1e-6)
    p.add_argument("--scf-implicit-diff-regularization", type=float, default=1e-3)
    p.add_argument("--scf-implicit-diff-restart", type=int, default=12)
    p.add_argument("--scf-require-convergence", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--grad-clip-norm", type=float, default=None)
    p.add_argument("--scf-warm-start", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--scf-warm-start-update-interval", type=int, default=1)
    p.add_argument(
        "--recover-nonfinite-steps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After each implicit self-consistent update, run an extra full loss "
            "evaluation and revert the step if it becomes non-finite."
        ),
    )
    p.add_argument("--jit-eval", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--jit-train", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--stream-train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Average one-molecule gradients instead of tracing the whole dataset as one batch.",
    )
    p.add_argument(
        "--stream-update-mode",
        choices=("accumulate", "per_molecule"),
        default="accumulate",
        help=(
            "Optimizer update policy for --stream-train. 'accumulate' averages "
            "one-molecule gradients over the training split before one update; "
            "'per_molecule' applies one optimizer update per molecule, i.e. "
            "traditional batch_size=1 while still evaluating per epoch."
        ),
    )
    p.add_argument(
        "--host-reference-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Keep prepared reference arrays on host memory. By default this is enabled "
            "for --stream-train so multiple molecules do not all keep hfx_nu on GPU."
        ),
    )
    p.add_argument(
        "--skip-initial-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="In streaming mode, start step 1 without a pre-training full-dataset eval.",
    )
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional parameter checkpoint used to initialize training.",
    )
    p.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="Global training step already completed by --init-checkpoint.",
    )
    p.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="Write streaming training parameter checkpoints every N steps; 0 disables periodic checkpoints.",
    )
    p.add_argument(
        "--skip-final-evaluation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Write training-only artifacts and checkpoint immediately after training, "
            "skipping the per-molecule final train/validation/test prediction pass."
        ),
    )
    p.add_argument("--outdir", default="outputs/closed_shell_s1_self_consistent_train")
    return p.parse_args(argv)


def _metric_scalar(metrics: dict[str, Any], key: str, default: float = float("nan")) -> float:
    if key not in metrics:
        return default
    arr = jnp.asarray(metrics[key])
    if int(arr.size) <= 0:
        return default
    return float(arr.reshape(-1)[0])


def _metric_mean(metrics: dict[str, Any], key: str, default: float = float("nan")) -> float:
    if key not in metrics:
        return default
    arr = jnp.asarray(metrics[key])
    if int(arr.size) <= 0:
        return default
    return float(jnp.mean(arr))


def _ground_only_best_score(
    args: argparse.Namespace,
    *,
    train_loss: float,
    train_metrics: dict[str, Any],
    val_loss: float | None,
    val_metrics: dict[str, Any] | None,
    has_validation: bool,
) -> float:
    if float(getattr(args, "s1_weight", 1.0)) == 0.0:
        if has_validation and val_loss is not None:
            return float(val_loss)
        return float(train_loss)
    if has_validation and val_metrics is not None:
        fallback = float(val_loss) if val_loss is not None else float("inf")
        return _metric_mean(val_metrics, "s1_mae", fallback)
    return _metric_mean(train_metrics, "s1_mae", float(train_loss))


def _normalize_input_feature_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode == "dm21_original":
        return "canonical"
    if mode in {"canonical", "enhanced"}:
        return mode
    raise ValueError(f"Unsupported input feature mode {value!r}.")


def _normalize_scf_gradient_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode in {"impl", "implicit_commutator"}:
        return "impl"
    if mode in {"expl", "unrolled"}:
        return "expl"
    raise ValueError(f"Unsupported SCF gradient mode {value!r}.")


def _ground_state_training_config(**kwargs: Any) -> GroundStateTrainingConfig:
    allowed = {field.name for field in fields(GroundStateTrainingConfig)}
    return GroundStateTrainingConfig(
        **{name: value for name, value in kwargs.items() if name in allowed}
    )


def _with_scf_initial_density(molecule: Any, density: Any) -> Any:
    density_arr = jnp.asarray(density)
    if is_dataclass(molecule):
        if "scf_initial_density" in getattr(molecule, "__dataclass_fields__", {}):
            return replace(molecule, scf_initial_density=density_arr)
        return molecule
    setattr(molecule, "scf_initial_density", density_arr)
    return molecule


def _spin_summed_density_matrix(molecule: Any) -> jnp.ndarray:
    density = jnp.asarray(molecule.rdm1)
    if density.ndim == 3:
        return density.sum(axis=0)
    return density


def _refresh_dataset_scf_warm_start_cache(
    dataset: tuple[GroundStateDatum, ...],
    *,
    params: Any,
    functional: Any,
    training_config: GroundStateTrainingConfig,
) -> tuple[GroundStateDatum, ...]:
    if training_config.mode != "self_consistent":
        return dataset
    refreshed: list[GroundStateDatum] = []
    for datum in dataset:
        predicted_molecule = predict_ground_state_molecule(
            params,
            functional,
            datum.molecule,
            training_config=training_config,
        )
        refreshed.append(
            replace(
                datum,
                molecule=_with_scf_initial_density(
                    datum.molecule,
                    jax.lax.stop_gradient(_spin_summed_density_matrix(predicted_molecule)),
                ),
            )
        )
    return tuple(refreshed)


def _tree_all_finite(tree: Any) -> bool:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return True
    return all(bool(jnp.all(jnp.isfinite(jnp.asarray(leaf)))) for leaf in leaves)


def _tree_l2_norm(tree: Any) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    total = jnp.asarray(0.0, dtype=jnp.float32)
    for leaf in leaves:
        arr = jnp.asarray(leaf)
        arr = jnp.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        total = total + jnp.sum(jnp.square(arr.astype(jnp.float32)))
    return jnp.sqrt(total)


def _tree_add(left: Any | None, right: Any) -> Any:
    if left is None:
        return right
    return jax.tree_util.tree_map(lambda a, b: a + b, left, right)


def _tree_scale(tree: Any, scale: float) -> Any:
    return jax.tree_util.tree_map(lambda value: value * scale, tree)


def _loss_and_metrics_all_finite(loss: Any, metrics: dict[str, Any]) -> bool:
    return bool(jnp.all(jnp.isfinite(jnp.asarray(loss)))) and _tree_all_finite(metrics)


def _use_host_reference_cache(args: argparse.Namespace) -> bool:
    if args.host_reference_cache is not None:
        return bool(args.host_reference_cache)
    return bool(args.stream_train)


def _scf_hfx_grid_block_size(args: argparse.Namespace, *, default: int = 1024) -> int:
    return int(getattr(args, "scf_hfx_grid_block_size", default))


def _lr_transition_steps(args: argparse.Namespace, *, train_size: int) -> int:
    transition_steps = int(args.lr_decay_every)
    if (
        bool(args.stream_train)
        and str(getattr(args, "stream_update_mode", "accumulate")) == "per_molecule"
    ):
        transition_steps *= max(1, int(train_size))
    return transition_steps


def _stream_lr_schedule_index(
    args: argparse.Namespace,
    *,
    step: int,
    train_size: int,
) -> int:
    if str(getattr(args, "stream_update_mode", "accumulate")) == "per_molecule":
        return max(0, int(step) - 1) * max(1, int(train_size))
    return max(0, int(step) - 1)


def _streaming_should_eval_step(args: argparse.Namespace, *, step: int) -> bool:
    final_step = int(step) == int(args.steps)
    if final_step and bool(args.skip_final_evaluation):
        return False
    return final_step or int(step) % max(1, int(args.eval_interval)) == 0


def _streaming_should_log_train_step(args: argparse.Namespace, *, step: int) -> bool:
    return (
        int(step) == 1
        or int(step) == int(args.steps)
        or int(step) % max(1, int(args.eval_interval)) == 0
    )


def _host_cache_pytree(tree: Any) -> Any:
    return jax.device_get(tree)


def _reference_cache_path(args: argparse.Namespace) -> Path | None:
    value = str(getattr(args, "reference_cache", "") or "").strip()
    if not value or value.lower() in {"none", "off", "false"}:
        return None
    return Path(value)


def _reference_cache_key(
    row: ReferenceRow,
    *,
    args: argparse.Namespace,
    input_feature_mode: str,
) -> str:
    payload = {
        "version": 1,
        "system": row.system,
        "atom": row.atom,
        "basis": row.basis,
        "unit": row.unit,
        "charge": int(row.charge),
        "spin": int(row.spin),
        "cart": True,
        "xc": str(args.xc),
        "grids_level": int(args.grids_level),
        "reference_jk_backend": str(args.reference_jk_backend),
        "response_df_mode": str(args.response_df_mode),
        "response_ris_theta": float(args.response_ris_theta),
        "response_ris_j_fit": str(args.response_ris_j_fit),
        "response_ris_k_fit": str(args.response_ris_k_fit),
        "input_feature_mode": str(input_feature_mode),
        "include_hfx_channel": bool(args.include_hfx_channel),
        "include_pt2_channel": bool(args.include_pt2_channel),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"closed_shell_s1/v1/{digest[:2]}/{digest}"


def _build_pyscf_mol_from_row(row: ReferenceRow) -> Any:
    return gto.M(
        atom=row.atom.replace(";", "\n"),
        unit=row.unit,
        basis=row.basis,
        charge=row.charge,
        spin=row.spin,
        cart=True,
        verbose=0,
    )


def _maybe_reattach_chunked_hfx_nu_api(
    row: ReferenceRow,
    molecule: Any,
    *,
    input_feature_mode: str,
    logger: RunLogger,
) -> Any:
    if str(input_feature_mode) != "canonical":
        return molecule
    if getattr(molecule, "hfx_nu_api", None) is not None:
        return molecule
    mol = _build_pyscf_mol_from_row(row)
    if int(getattr(mol, "natm", 0)) <= 3:
        return molecule
    omega_values = getattr(molecule, "hfx_omega_values", None)
    coords = getattr(getattr(molecule, "grid", None), "coords", None)
    ao = getattr(molecule, "ao", None)
    if omega_values is None or coords is None or ao is None:
        return molecule
    omega_tuple = tuple(
        float(value)
        for value in np.asarray(jax.device_get(omega_values)).reshape(-1)
    )
    if not omega_tuple:
        return molecule
    api = ChunkedHFXNu.from_pyscf_mol(
        mol,
        np.asarray(jax.device_get(coords)),
        omega_values=omega_tuple,
        nao=int(np.asarray(jax.device_get(ao)).shape[1]),
    )
    logger.log(f"[ref_cache] reattached chunked hfx_nu_api for {row.system}")
    return replace(molecule, hfx_nu=None, hfx_nu_api=api)


def _cache_hfx_nu_storage(
    molecule_group: Any,
    *,
    args: argparse.Namespace,
    input_feature_mode: str,
) -> str:
    if not bool(args.include_hfx_channel):
        return "array"
    if str(input_feature_mode) != "canonical":
        return "array"
    if "hfx_nu" not in molecule_group:
        return "array"
    if "atom_coords" not in molecule_group:
        return "array"
    if int(molecule_group["atom_coords"].shape[0]) <= 3:
        return "array"
    return "chunked"


def _load_reference_from_cache(
    row: ReferenceRow,
    *,
    args: argparse.Namespace,
    input_feature_mode: str,
    host_reference_cache: bool,
    logger: RunLogger,
    ignore_rebuild: bool = False,
) -> Any | None:
    cache_path = _reference_cache_path(args)
    if (
        cache_path is None
        or (bool(args.rebuild_reference_cache) and not bool(ignore_rebuild))
        or not cache_path.exists()
    ):
        return None
    key = _reference_cache_key(row, args=args, input_feature_mode=input_feature_mode)
    try:
        import h5py

        with h5py.File(cache_path, "r") as handle:
            if key not in handle:
                return None
            molecule_group = handle[key]["molecule"]
            hfx_nu_storage = _cache_hfx_nu_storage(
                molecule_group,
                args=args,
                input_feature_mode=input_feature_mode,
            )
            molecule = read_restricted_molecule(
                molecule_group,
                array_backend="host" if host_reference_cache else "jax",
                hfx_nu_storage=hfx_nu_storage,
                hfx_nu_chunk_size=_scf_hfx_grid_block_size(args, default=512),
            )
            molecule = _maybe_reattach_chunked_hfx_nu_api(
                row,
                molecule,
                input_feature_mode=input_feature_mode,
                logger=logger,
            )
    except Exception as exc:
        logger.log(f"[ref_cache] miss/error {row.system}: {exc!r}; rebuilding")
        return None
    logger.log(f"[ref_cache] hit {row.system}: {cache_path}::{key}")
    return molecule


def _save_reference_to_cache(
    row: ReferenceRow,
    molecule: Any,
    *,
    args: argparse.Namespace,
    input_feature_mode: str,
    logger: RunLogger,
) -> None:
    cache_path = _reference_cache_path(args)
    if cache_path is None:
        return
    key = _reference_cache_key(row, args=args, input_feature_mode=input_feature_mode)
    try:
        import h5py

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(cache_path, "a") as handle:
            parent_name, leaf_name = key.rsplit("/", 1)
            parent = handle.require_group(parent_name)
            if leaf_name in parent:
                del parent[leaf_name]
            group = parent.create_group(leaf_name)
            group.attrs["system"] = row.system
            group.attrs["basis"] = row.basis
            group.attrs["xc"] = str(args.xc)
            group.attrs["grids_level"] = int(args.grids_level)
            group.attrs["reference_jk_backend"] = str(args.reference_jk_backend)
            group.attrs["input_feature_mode"] = str(input_feature_mode)
            group.attrs["include_hfx_channel"] = bool(args.include_hfx_channel)
            group.attrs["include_pt2_channel"] = bool(args.include_pt2_channel)
            write_restricted_molecule(group.require_group("molecule"), molecule)
    except Exception as exc:
        logger.log(f"[ref_cache] write failed {row.system}: {exc!r}")
        return
    logger.log(f"[ref_cache] wrote {row.system}: {cache_path}::{key}")


def _streaming_average_eval(
    params: Any,
    data: tuple[GroundStateDatum, ...],
    eval_kernel: Any,
) -> tuple[float, dict[str, float]]:
    if not data:
        return float("nan"), {"s1_mae": float("nan"), "s1_mse": float("nan")}
    losses: list[float] = []
    s1_maes: list[float] = []
    s1_mses: list[float] = []
    for datum in data:
        loss, metrics = eval_kernel(params, datum)
        losses.append(float(loss))
        s1_maes.append(_metric_mean(metrics, "s1_mae", 0.0))
        s1_mses.append(_metric_mean(metrics, "s1_mse", 0.0))
    return float(np.mean(losses)), {
        "s1_mae": float(np.mean(s1_maes)),
        "s1_mse": float(np.mean(s1_mses)),
    }

def _load_reference_rows(path: Path, *, basis: str) -> list[ReferenceRow]:
    rows: list[ReferenceRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row["basis"]).lower() != str(basis).lower():
                continue
            rows.append(
                ReferenceRow(
                    system=str(row["system"]),
                    split=str(row["split"]),
                    atom=str(row["atom"]),
                    unit=str(row["unit"]),
                    charge=int(row["charge"]),
                    spin=int(row["spin"]),
                    basis=str(row["basis"]),
                    ccsd_total_energy_h=float(row["ccsd_total_energy_h"]),
                    s1_excitation_h=float(row["s1_excitation_h"]),
                )
            )
    if not rows:
        raise ValueError(f"No rows with basis={basis!r} found in {path}.")
    return rows


def _run_rks_reference(
    row: ReferenceRow,
    *,
    xc: str,
    grids_level: int,
    scf_conv_tol: float,
    scf_max_cycle: int,
) -> Any:
    mol = _build_pyscf_mol_from_row(row)
    attempts = (
        dict(init_guess="minao", damping=0.15, level_shift=0.0, max_cycle=scf_max_cycle, use_newton=False),
        dict(init_guess="atom", damping=0.3, level_shift=0.5, max_cycle=max(scf_max_cycle, 180), use_newton=False),
        dict(init_guess="atom", damping=0.0, level_shift=0.0, max_cycle=max(scf_max_cycle, 80), use_newton=True),
    )
    last_mf = None
    for cfg in attempts:
        mf = dft.RKS(mol)
        mf.xc = str(xc)
        mf.grids.level = int(grids_level)
        mf.conv_tol = float(scf_conv_tol)
        mf.max_cycle = int(cfg["max_cycle"])
        mf.damping = float(cfg["damping"])
        mf.level_shift = float(cfg["level_shift"])
        mf.diis_start_cycle = 1
        mf.init_guess = str(cfg["init_guess"])
        if cfg["use_newton"]:
            mf = mf.newton()
            mf.xc = str(xc)
            mf.grids.level = int(grids_level)
            mf.conv_tol = float(scf_conv_tol)
            mf.max_cycle = int(cfg["max_cycle"])
        mf.kernel()
        last_mf = mf
        if bool(mf.converged):
            return mf
    raise RuntimeError(f"PySCF RKS did not converge for {row.system}.")


def _prepare_references(
    rows: list[ReferenceRow],
    *,
    args: argparse.Namespace,
    logger: RunLogger,
) -> list[PreparedReference]:
    prepared: list[PreparedReference] = []
    host_reference_cache = _use_host_reference_cache(args)
    for idx, row in enumerate(rows, start=1):
        input_feature_mode = _normalize_input_feature_mode(str(args.input_feature_mode))
        reference = _load_reference_from_cache(
            row,
            args=args,
            input_feature_mode=input_feature_mode,
            host_reference_cache=host_reference_cache,
            logger=logger,
        )
        if reference is not None:
            prepared.append(PreparedReference(row=row, molecule=reference))
            continue

        logger.log(f"[ref] {idx}/{len(rows)} build {row.system} ({row.split})")
        mf = _run_rks_reference(
            row,
            xc=str(args.xc),
            grids_level=int(args.grids_level),
            scf_conv_tol=float(args.reference_scf_conv_tol),
            scf_max_cycle=int(args.reference_scf_max_cycle),
        )
        reference = restricted_reference_from_pyscf(
            mf,
            compute_local_hfx_features=(
                bool(args.include_hfx_channel) and input_feature_mode == "canonical"
            ),
            compute_local_hfx_aux=(
                bool(args.include_hfx_channel) and input_feature_mode == "canonical"
            ),
            compute_local_pt2_features=bool(args.include_pt2_channel),
            array_backend="host" if host_reference_cache else "jax",
            jk_backend=str(args.reference_jk_backend),
            response_df_mode=str(args.response_df_mode),
            response_ris_theta=float(args.response_ris_theta),
            response_ris_j_fit=str(args.response_ris_j_fit),
            response_ris_k_fit=str(args.response_ris_k_fit),
        )
        if host_reference_cache:
            reference = _host_cache_pytree(reference)
            gc.collect()
            logger.log(f"[ref] {idx}/{len(rows)} host-cached {row.system} ({row.split})")
        _save_reference_to_cache(
            row,
            reference,
            args=args,
            input_feature_mode=input_feature_mode,
            logger=logger,
        )
        cached_reference = _load_reference_from_cache(
            row,
            args=args,
            input_feature_mode=input_feature_mode,
            host_reference_cache=host_reference_cache,
            logger=logger,
            ignore_rebuild=True,
        )
        if cached_reference is not None:
            reference = cached_reference
        prepared.append(PreparedReference(row=row, molecule=reference))
    return prepared


def _build_dataset(
    prepared: list[PreparedReference],
    *,
    s1_weight: float,
    density_constraint_weight: float,
) -> tuple[GroundStateDatum, ...]:
    return tuple(
        GroundStateDatum.from_parts(
            ref.molecule,
            core=GroundStateCoreDatum(
                target_total_energy=jnp.asarray(ref.row.ccsd_total_energy_h, dtype=jnp.float64),
                density_constraint_weight=float(density_constraint_weight),
            ),
            excited_state=ExcitedStateDatum(
                target_s1_energy=jnp.asarray(ref.row.s1_excitation_h, dtype=jnp.float64),
                s1_constraint_weight=float(s1_weight),
            ),
        )
        for ref in prepared
    )


def _evaluate_dataset(
    prepared: list[PreparedReference],
    *,
    params: Any,
    functional: Any,
    training_config: GroundStateTrainingConfig,
    use_tda: bool,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    s1_err_ev: list[float] = []
    energy_err_ev: list[float] = []
    for ref in prepared:
        predicted_molecule = predict_ground_state_molecule(
            params,
            functional,
            ref.molecule,
            training_config=training_config,
        )
        predicted_total_h = float(
            predict_ground_state_total_energy(
                params,
                functional,
                ref.molecule,
                training_config=training_config,
            )
        )
        predicted_s1_h = float(
            np.asarray(
                predict_excitation_energies(
                    params,
                    functional,
                    predicted_molecule,
                    nstates=1,
                    use_tda=bool(use_tda),
                )
            ).reshape(-1)[0]
        )
        s1_abs_err_ev = abs(predicted_s1_h - ref.row.s1_excitation_h) * HARTREE_TO_EV
        total_abs_err_ev = abs(predicted_total_h - ref.row.ccsd_total_energy_h) * HARTREE_TO_EV
        s1_err_ev.append(float(s1_abs_err_ev))
        energy_err_ev.append(float(total_abs_err_ev))
        rows.append(
            {
                "system": ref.row.system,
                "split": ref.row.split,
                "target_total_energy_h": float(ref.row.ccsd_total_energy_h),
                "predicted_total_energy_h": float(predicted_total_h),
                "target_s1_h": float(ref.row.s1_excitation_h),
                "predicted_s1_h": float(predicted_s1_h),
                "target_s1_ev": float(ref.row.s1_excitation_h * HARTREE_TO_EV),
                "predicted_s1_ev": float(predicted_s1_h * HARTREE_TO_EV),
                "s1_abs_err_ev": float(s1_abs_err_ev),
                "total_abs_err_ev": float(total_abs_err_ev),
            }
        )
    metrics = {
        "s1_mae_ev": float(np.mean(s1_err_ev)) if s1_err_ev else float("nan"),
        "s1_max_ev": float(np.max(s1_err_ev)) if s1_err_ev else float("nan"),
        "total_mae_ev": float(np.mean(energy_err_ev)) if energy_err_ev else float("nan"),
    }
    return rows, metrics


def _write_prediction_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_training_history(path: Path, training: dict[str, Any], *, title: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2))
    steps = np.asarray(training["history_steps"], dtype=np.int64)
    loss = np.asarray(training["loss_history"], dtype=np.float64)
    ax.plot(steps, np.maximum(loss, 1e-16), lw=1.6, label="train loss")
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_training_history_csv(path: Path, training: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "train_loss",
                "train_s1_mae",
                "grad_norm",
                "grad_abs_max",
                "param_update_norm",
                "nonfinite_grad_fraction",
                "train_loss_reevaluated",
                "train_s1_mae_reevaluated",
                "val_loss_reevaluated",
                "val_s1_mae_reevaluated",
            ]
        )
        eval_map = {
            int(step): (
                float(train_loss),
                float(train_s1),
                float(val_loss),
                float(val_s1),
            )
            for step, train_loss, train_s1, val_loss, val_s1 in zip(
                training["eval_steps"],
                training["eval_train_loss_history"],
                training["eval_train_s1_mae_history"],
                training["eval_val_loss_history"],
                training["eval_val_s1_mae_history"],
                strict=False,
            )
        }
        for idx, step in enumerate(training["history_steps"]):
            train_loss, train_s1, val_loss, val_s1 = eval_map.get(
                int(step),
                (float("nan"), float("nan"), float("nan"), float("nan")),
            )
            writer.writerow(
                [
                    int(step),
                    float(training["loss_history"][idx]),
                    float(training["s1_mae_history"][idx]),
                    float(training["grad_norm_history"][idx]),
                    float(training["grad_abs_max_history"][idx]),
                    float(training["param_update_norm_history"][idx]),
                    float(training["nonfinite_grad_fraction_history"][idx]),
                    train_loss,
                    train_s1,
                    val_loss,
                    val_s1,
                ]
            )


def _train_streaming(
    state: Any,
    train_dataset: tuple[GroundStateDatum, ...],
    val_dataset: tuple[GroundStateDatum, ...],
    *,
    functional: Any,
    training_config: GroundStateTrainingConfig,
    lr_schedule: Any,
    args: argparse.Namespace,
    logger: RunLogger,
) -> dict[str, Any]:
    loss_and_grad_kernel = make_ground_state_loss_and_grad(
        functional,
        training_config=training_config,
    )
    eval_kernel = lambda params, datum: ground_state_mse_loss(  # noqa: E731
        params,
        functional,
        datum,
        training_config=training_config,
    )
    if bool(args.jit_train):
        loss_and_grad_kernel = jax.jit(loss_and_grad_kernel)
    if bool(args.jit_eval):
        eval_kernel = jax.jit(eval_kernel)

    stream_update_mode = str(getattr(args, "stream_update_mode", "accumulate"))
    if stream_update_mode == "per_molecule":
        logger.log(
            "[train_init] streaming mode: one molecule per JIT call; "
            "applying per-molecule optimizer updates"
        )
    else:
        logger.log("[train_init] streaming mode: one molecule per JIT call; averaging grads")
    if bool(args.skip_initial_eval):
        initial_train_loss = float("nan")
        initial_train_metrics = {"s1_mae": float("nan"), "s1_mse": float("nan")}
        initial_val_loss = float("nan")
        initial_val_metrics = {"s1_mae": float("nan"), "s1_mse": float("nan")}
        best_score = float("inf")
        logger.log("[train_init] skipped initial eval")
    else:
        initial_train_loss, initial_train_metrics = _streaming_average_eval(
            state.params,
            train_dataset,
            eval_kernel,
        )
        initial_val_loss, initial_val_metrics = _streaming_average_eval(
            state.params,
            val_dataset,
            eval_kernel,
        )
        best_score = _ground_only_best_score(
            args,
            train_loss=float(initial_train_loss),
            train_metrics=initial_train_metrics,
            val_loss=float(initial_val_loss),
            val_metrics=initial_val_metrics,
            has_validation=bool(val_dataset),
        )
    best_params = state.params
    start_step = max(0, int(getattr(args, "start_step", 0)))
    best_step = start_step

    history_steps = [start_step]
    loss_history = [initial_train_loss]
    s1_mae_history = [initial_train_metrics["s1_mae"]]
    grad_norm_history = [float("nan")]
    grad_abs_max_history = [float("nan")]
    param_update_norm_history = [float("nan")]
    nonfinite_grad_fraction_history = [0.0]
    eval_steps = [start_step]
    eval_train_loss_history = [initial_train_loss]
    eval_train_s1_mae_history = [initial_train_metrics["s1_mae"]]
    eval_val_loss_history = [initial_val_loss]
    eval_val_s1_mae_history = [initial_val_metrics["s1_mae"]]

    logger.log(
        "[train] "
        f"steps={int(args.steps)} mode={str(args.training_mode)} "
        f"objective=s1_only_{'tda' if bool(args.s1_use_tda) else 'casida'} "
        f"start_step={start_step} "
        f"train_step_mode=stream_single_molecule update_mode={stream_update_mode} "
        f"train_size={len(train_dataset)} "
        f"val_size={len(val_dataset)}"
    )

    t0 = time.perf_counter()
    for step in range(start_step + 1, int(args.steps) + 1):
        losses: list[float] = []
        s1_maes: list[float] = []
        grad_norms: list[float] = []
        grad_abs_maxes: list[float] = []
        nonfinite_fracs: list[float] = []
        update_norms: list[float] = []
        if stream_update_mode == "per_molecule":
            for datum in train_dataset:
                loss, metrics, grads = loss_and_grad_kernel(state.params, datum)
                losses.append(float(loss))
                s1_maes.append(_metric_mean(metrics, "s1_mae", 0.0))
                grad_norms.append(_metric_scalar(metrics, "grad_norm"))
                grad_abs_maxes.append(_metric_scalar(metrics, "grad_abs_max"))
                nonfinite_fracs.append(_metric_scalar(metrics, "nonfinite_grad_fraction", 0.0))
                prev_state = state
                state = state.apply_gradients(grads=grads)
                param_delta = jax.tree_util.tree_map(
                    lambda new, old: new - old,
                    state.params,
                    prev_state.params,
                )
                if not _tree_all_finite(state.params):
                    state = prev_state
                    update_norms.append(0.0)
                    logger.log(
                        f"[train] non-finite params at step {step}; "
                        "reverted per-molecule update"
                    )
                else:
                    update_norms.append(float(_tree_l2_norm(param_delta)))
        else:
            params_for_loss = state.params
            grad_sum = None
            for datum in train_dataset:
                loss, metrics, grads = loss_and_grad_kernel(params_for_loss, datum)
                losses.append(float(loss))
                s1_maes.append(_metric_mean(metrics, "s1_mae", 0.0))
                grad_norms.append(_metric_scalar(metrics, "grad_norm"))
                grad_abs_maxes.append(_metric_scalar(metrics, "grad_abs_max"))
                nonfinite_fracs.append(_metric_scalar(metrics, "nonfinite_grad_fraction", 0.0))
                grad_sum = _tree_add(grad_sum, grads)
            grads_avg = _tree_scale(grad_sum, 1.0 / max(1, len(train_dataset)))
            prev_state = state
            state = state.apply_gradients(grads=grads_avg)
            param_delta = jax.tree_util.tree_map(
                lambda new, old: new - old,
                state.params,
                prev_state.params,
            )
            if not _tree_all_finite(state.params):
                state = prev_state
                update_norms.append(0.0)
                logger.log(f"[train] non-finite params at step {step}; reverted update")
            else:
                update_norms.append(float(_tree_l2_norm(param_delta)))
        train_loss_val = float(np.mean(losses))
        train_s1_mae_val = float(np.mean(s1_maes))
        grad_norm_val = float(np.mean(grad_norms))
        grad_abs_max_val = float(np.max(grad_abs_maxes))
        nonfinite_grad_fraction_val = float(np.mean(nonfinite_fracs))
        update_norm_val = float(np.mean(update_norms))
        checkpoint_train_loss = train_loss_val
        checkpoint_train_s1_mae = train_s1_mae_val
        checkpoint_val_loss = float("nan")
        checkpoint_val_s1_mae = float("nan")

        history_steps.append(step)
        loss_history.append(train_loss_val)
        s1_mae_history.append(train_s1_mae_val)
        grad_norm_history.append(grad_norm_val)
        grad_abs_max_history.append(grad_abs_max_val)
        param_update_norm_history.append(update_norm_val)
        nonfinite_grad_fraction_history.append(nonfinite_grad_fraction_val)

        current_lr = (
            float(
                lr_schedule(
                    _stream_lr_schedule_index(
                        args,
                        step=step,
                        train_size=len(train_dataset),
                    )
                )
            )
            if lr_schedule is not None
            else float(args.learning_rate)
        )
        if _streaming_should_eval_step(args, step=step):
            eval_train_loss, eval_train_metrics = _streaming_average_eval(
                state.params,
                train_dataset,
                eval_kernel,
            )
            eval_val_loss, eval_val_metrics = _streaming_average_eval(
                state.params,
                val_dataset,
                eval_kernel,
            )
            eval_steps.append(step)
            eval_train_loss_history.append(eval_train_loss)
            eval_train_s1_mae_history.append(eval_train_metrics["s1_mae"])
            eval_val_loss_history.append(eval_val_loss)
            eval_val_s1_mae_history.append(eval_val_metrics["s1_mae"])
            checkpoint_train_loss = float(eval_train_loss)
            checkpoint_train_s1_mae = float(eval_train_metrics["s1_mae"])
            checkpoint_val_loss = float(eval_val_loss)
            checkpoint_val_s1_mae = float(eval_val_metrics["s1_mae"])
            score = _ground_only_best_score(
                args,
                train_loss=float(eval_train_loss),
                train_metrics=eval_train_metrics,
                val_loss=float(eval_val_loss),
                val_metrics=eval_val_metrics,
                has_validation=bool(val_dataset),
            )
            if score < best_score:
                best_score = score
                best_params = state.params
                best_step = step
            logger.log(
                "[train] "
                f"step={step:4d}/{int(args.steps):4d} "
                f"train_loss={eval_train_loss:.8e} "
                f"train_s1_mae={eval_train_metrics['s1_mae']:.8e} "
                f"val_loss={eval_val_loss:.8e} "
                f"val_s1_mae={eval_val_metrics['s1_mae']:.8e} "
                f"grad_norm={grad_norm_val:.8e} "
                f"grad_abs_max={grad_abs_max_val:.8e} "
                f"update_norm={update_norm_val:.8e} "
                f"lr={current_lr:.8e}"
            )
        elif _streaming_should_log_train_step(args, step=step):
            logger.log(
                "[train] "
                f"step={step:4d}/{int(args.steps):4d} "
                f"train_loss={train_loss_val:.8e} "
                f"train_s1_mae={train_s1_mae_val:.8e} "
                "val_loss=nan val_s1_mae=nan "
                f"grad_norm={grad_norm_val:.8e} "
                f"grad_abs_max={grad_abs_max_val:.8e} "
                f"update_norm={update_norm_val:.8e} "
                f"lr={current_lr:.8e} "
                "eval=skipped"
            )
        checkpoint_interval = max(0, int(getattr(args, "checkpoint_interval", 0)))
        if checkpoint_interval > 0 and step % checkpoint_interval == 0:
            checkpoint_dir = Path(str(args.outdir)) / "checkpoints"
            checkpoint_path = checkpoint_dir / f"neural_xc_params_step{step:06d}.msgpack"
            checkpoint_path, checkpoint_meta_path = save_params_checkpoint(
                checkpoint_path,
                state.params,
                metadata={
                    "step": int(step),
                    "steps": int(args.steps),
                    "reference_csv": str(args.reference_csv),
                    "basis": str(args.basis),
                    "xc": str(args.xc),
                    "training_mode": str(args.training_mode),
                    "include_hfx_channel": bool(args.include_hfx_channel),
                    "response_hf_mode": str(args.response_hf_mode),
                    "stream_train": bool(args.stream_train),
                    "stream_update_mode": str(args.stream_update_mode),
                    "learning_rate": float(current_lr),
                    "train_loss": float(checkpoint_train_loss),
                    "train_s1_mae": float(checkpoint_train_s1_mae),
                    "val_loss": float(checkpoint_val_loss),
                    "val_s1_mae": float(checkpoint_val_s1_mae),
                    "grad_norm": float(grad_norm_val),
                    "update_norm": float(update_norm_val),
                },
            )
            progress = {
                "step": int(step),
                "steps": int(args.steps),
                "checkpoint": str(checkpoint_path),
                "checkpoint_meta": str(checkpoint_meta_path)
                if checkpoint_meta_path is not None
                else None,
                "train_loss": float(checkpoint_train_loss),
                "train_s1_mae": float(checkpoint_train_s1_mae),
                "val_loss": float(checkpoint_val_loss),
                "val_s1_mae": float(checkpoint_val_s1_mae),
                "grad_norm": float(grad_norm_val),
                "update_norm": float(update_norm_val),
                "learning_rate": float(current_lr),
            }
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "latest_checkpoint.txt").write_text(
                str(checkpoint_path) + "\n",
                encoding="utf-8",
            )
            (checkpoint_dir / "latest_progress.json").write_text(
                json.dumps(progress, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            logger.log(f"[checkpoint] step={step:4d} wrote {checkpoint_path}")

    elapsed_s = time.perf_counter() - t0
    if bool(args.skip_final_evaluation):
        final_train_loss = float(loss_history[-1])
        final_train_metrics = {
            "s1_mae": float(s1_mae_history[-1]),
            "s1_mse": float("nan"),
        }
        final_val_loss = float("nan")
        final_val_metrics = {"s1_mae": float("nan"), "s1_mse": float("nan")}
        if not np.isfinite(best_score):
            best_params = state.params
            best_step = int(args.steps)
            best_score = (
                float(final_train_loss)
                if float(getattr(args, "s1_weight", 1.0)) == 0.0
                else float(final_train_metrics["s1_mae"])
            )
    else:
        final_train_loss, final_train_metrics = _streaming_average_eval(
            state.params,
            train_dataset,
            eval_kernel,
        )
        final_val_loss, final_val_metrics = _streaming_average_eval(
            state.params,
            val_dataset,
            eval_kernel,
        )
    logger.log(
        "[train] done "
        f"final_train_loss={final_train_loss:.8e} "
        f"best_s1_mae={best_score:.8e}@{best_step} "
        f"elapsed_s={elapsed_s:.2f}"
    )
    return {
        "functional": functional,
        "training_config": training_config,
        "best_params": best_params,
        "final_params": state.params,
        "best_step": int(best_step),
        "best_score": float(best_score),
        "final_train_loss": float(final_train_loss),
        "final_train_s1_mae": float(final_train_metrics["s1_mae"]),
        "final_val_loss": float(final_val_loss),
        "final_val_s1_mae": float(final_val_metrics["s1_mae"]),
        "elapsed_s": float(elapsed_s),
        "history_steps": history_steps,
        "loss_history": loss_history,
        "s1_mae_history": s1_mae_history,
        "grad_norm_history": grad_norm_history,
        "grad_abs_max_history": grad_abs_max_history,
        "param_update_norm_history": param_update_norm_history,
        "nonfinite_grad_fraction_history": nonfinite_grad_fraction_history,
        "eval_steps": eval_steps,
        "eval_train_loss_history": eval_train_loss_history,
        "eval_train_s1_mae_history": eval_train_s1_mae_history,
        "eval_val_loss_history": eval_val_loss_history,
        "eval_val_s1_mae_history": eval_val_s1_mae_history,
        "post_update_recoveries": 0,
    }


def _train(
    train_dataset: tuple[GroundStateDatum, ...],
    val_dataset: tuple[GroundStateDatum, ...],
    *,
    init_molecule: Any,
    args: argparse.Namespace,
    logger: RunLogger,
) -> dict[str, Any]:
    functional = neural_xc.Functional(
        architecture=str(args.network_architecture),
        semilocal_xc=tuple(str(name) for name in args.semilocal_xc),
        hidden_dims=tuple(int(value) for value in args.hidden_dims),
        input_feature_mode=_normalize_input_feature_mode(str(args.input_feature_mode)),
        include_hfx_channel=bool(args.include_hfx_channel),
        response_hf_mode=str(args.response_hf_mode),
        include_pt2_channel=bool(args.include_pt2_channel),
        pt2_channel_mode=str(args.pt2_channel_mode),
        name=f"neural_xc_closed_shell_{str(args.training_mode)}",
    )
    training_config = _ground_state_training_config(
        mode=str(args.training_mode),
        energy_mse_weight=float(args.energy_mse_weight),
        energy_mae_weight=float(args.energy_mae_weight),
        s1_constraint_use_tda=bool(args.s1_use_tda),
        scf_max_cycle=int(args.train_scf_max_cycle),
        scf_damping=float(args.train_scf_damping),
        scf_conv_tol_density=float(args.train_scf_conv_tol_density),
        scf_vxc_clip=float(args.train_scf_vxc_clip),
        scf_iterate_selection=str(args.scf_iterate_selection),
        scf_require_convergence=bool(args.scf_require_convergence),
        scf_gradient_mode=_normalize_scf_gradient_mode(str(args.scf_gradient_mode)),
        scf_implicit_diff_solver=str(args.scf_implicit_diff_solver),
        scf_implicit_diff_tolerance=float(args.scf_implicit_diff_tolerance),
        scf_implicit_diff_regularization=float(args.scf_implicit_diff_regularization),
        scf_implicit_diff_restart=int(args.scf_implicit_diff_restart),
        response_two_electron_mode=str(args.response_two_electron_mode),
        response_ris_theta=float(args.response_ris_theta),
        response_ris_j_fit=str(args.response_ris_j_fit),
        response_ris_k_fit=str(args.response_ris_k_fit),
        response_ris_aux_chunk_size=int(args.response_ris_aux_chunk_size),
    )
    if int(args.lr_decay_every) > 0:
        transition_steps = _lr_transition_steps(args, train_size=len(train_dataset))
        lr_schedule = optax.exponential_decay(
            init_value=float(args.learning_rate),
            transition_steps=transition_steps,
            decay_rate=float(args.lr_decay_factor),
            staircase=True,
        )
        optimizer_lr_schedule = lr_schedule
        start_step = max(0, int(getattr(args, "start_step", 0)))
        if start_step > 0:
            schedule_offset = _stream_lr_schedule_index(
                args,
                step=start_step + 1,
                train_size=len(train_dataset),
            )
            optimizer_lr_schedule = lambda count: lr_schedule(count + schedule_offset)
        base_optimizer = optax.adam(optimizer_lr_schedule)
    else:
        lr_schedule = None
        base_optimizer = optax.adam(float(args.learning_rate))
    optimizer = (
        optax.chain(optax.clip_by_global_norm(float(args.grad_clip_norm)), base_optimizer)
        if args.grad_clip_norm is not None and float(args.grad_clip_norm) > 0.0
        else base_optimizer
    )
    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(int(args.seed)),
        init_molecule,
        optimizer,
    )
    if args.init_checkpoint is not None:
        checkpoint_params = load_params_checkpoint(args.init_checkpoint, template=state.params)
        state = state.replace(params=checkpoint_params)
        logger.log(
            "[train_init] loaded initial params checkpoint "
            f"{args.init_checkpoint} at start_step={int(args.start_step)}"
        )
    if bool(args.stream_train):
        return _train_streaming(
            state,
            train_dataset,
            val_dataset,
            functional=functional,
            training_config=training_config,
            lr_schedule=lr_schedule,
            args=args,
            logger=logger,
        )
    train_step = make_ground_state_train_step(functional, training_config=training_config)
    use_warm_start = bool(args.scf_warm_start) and training_config.mode == "self_consistent"
    train_dataset_current = train_dataset
    val_dataset_current = val_dataset
    if use_warm_start:
        train_eval_fn = lambda params, dataset: ground_state_mse_loss(  # noqa: E731
            params,
            functional,
            dataset,
            training_config=training_config,
        )
        val_eval_fn = (
            (lambda params, dataset: ground_state_mse_loss(  # noqa: E731
                params,
                functional,
                dataset,
                training_config=training_config,
            ))
            if val_dataset
            else None
        )
        eager_train_step = lambda current_state, dataset: train_step(current_state, dataset)  # noqa: E731
        compiled_train_eval = train_eval_fn
        compiled_val_eval = val_eval_fn
        compiled_train_step = eager_train_step
        train_step_mode = "eager"
        if bool(args.jit_train) or bool(args.jit_eval):
            logger.log("[train] disabled JIT because scf_warm_start requires dynamic dataset updates")
        initial_train_loss, initial_train_metrics = compiled_train_eval(state.params, train_dataset_current)
    else:
        train_eval_fn = lambda params: ground_state_mse_loss(  # noqa: E731
            params,
            functional,
            train_dataset,
            training_config=training_config,
        )
        val_eval_fn = (
            (lambda params: ground_state_mse_loss(  # noqa: E731
                params,
                functional,
                val_dataset,
                training_config=training_config,
            ))
            if val_dataset
            else None
        )
        eager_train_step = lambda current_state: train_step(current_state, train_dataset)  # noqa: E731
        compiled_train_eval = jax.jit(train_eval_fn) if bool(args.jit_eval) else train_eval_fn
        compiled_val_eval = (
            jax.jit(val_eval_fn) if (val_eval_fn is not None and bool(args.jit_eval)) else val_eval_fn
        )
        compiled_train_step = eager_train_step
        train_step_mode = "eager"
        if bool(args.jit_train):
            candidate = jax.jit(eager_train_step)
            try:
                _ = candidate.lower(state).compile()
                compiled_train_step = candidate
                train_step_mode = "jit"
            except Exception as exc:
                logger.log(f"[train] jit compilation failed for multi-molecule train step: {exc!r}")
        initial_train_loss, initial_train_metrics = compiled_train_eval(state.params)
    if bool(args.scf_warm_start):
        train_dataset_current = _refresh_dataset_scf_warm_start_cache(
            train_dataset_current,
            params=state.params,
            functional=functional,
            training_config=training_config,
        )
    if compiled_val_eval is not None:
        initial_val_loss, initial_val_metrics = (
            compiled_val_eval(state.params, val_dataset_current)
            if use_warm_start
            else compiled_val_eval(state.params)
        )
        if bool(args.scf_warm_start):
            val_dataset_current = _refresh_dataset_scf_warm_start_cache(
                val_dataset_current,
                params=state.params,
                functional=functional,
                training_config=training_config,
            )
        best_score = _ground_only_best_score(
            args,
            train_loss=float(initial_train_loss),
            train_metrics=initial_train_metrics,
            val_loss=float(initial_val_loss),
            val_metrics=initial_val_metrics,
            has_validation=True,
        )
    else:
        initial_val_loss, initial_val_metrics = None, None
        best_score = _ground_only_best_score(
            args,
            train_loss=float(initial_train_loss),
            train_metrics=initial_train_metrics,
            val_loss=None,
            val_metrics=None,
            has_validation=False,
        )
    best_params = state.params
    best_step = 0

    history_steps = [0]
    loss_history = [float(initial_train_loss)]
    s1_mae_history = [_metric_mean(initial_train_metrics, "s1_mae", 0.0)]
    grad_norm_history = [float("nan")]
    grad_abs_max_history = [float("nan")]
    param_update_norm_history = [float("nan")]
    nonfinite_grad_fraction_history = [0.0]
    eval_steps = [0]
    eval_train_loss_history = [float(initial_train_loss)]
    eval_train_s1_mae_history = [_metric_mean(initial_train_metrics, "s1_mae", float("nan"))]
    eval_val_loss_history = [float(initial_val_loss) if initial_val_loss is not None else float("nan")]
    eval_val_s1_mae_history = [_metric_mean(initial_val_metrics, "s1_mae", float("nan")) if initial_val_metrics is not None else float("nan")]

    logger.log(
        "[train] "
        f"steps={int(args.steps)} mode={str(args.training_mode)} objective=s1_only_{'tda' if bool(args.s1_use_tda) else 'casida'} "
        f"train_step_mode={train_step_mode} train_size={len(train_dataset)} val_size={len(val_dataset)}"
    )

    t0 = time.perf_counter()
    post_update_recoveries = 0
    guard_post_update = (
        bool(args.recover_nonfinite_steps)
        and training_config.mode == "self_consistent"
        and str(args.scf_gradient_mode) == "implicit_commutator"
    )
    for step in range(1, int(args.steps) + 1):
        prev_state = state
        prev_train_dataset_current = train_dataset_current
        state, train_metrics = (
            compiled_train_step(state, train_dataset_current)
            if use_warm_start
            else compiled_train_step(state)
        )
        reverted_step = False
        if not _tree_all_finite(state.params):
            state = prev_state
            train_dataset_current = prev_train_dataset_current
            reverted_step = True
            logger.log(f"[train] non-finite params at step {step}; reverted update")
        elif (
            bool(args.scf_warm_start)
            and training_config.mode == "self_consistent"
            and step % max(1, int(args.scf_warm_start_update_interval)) == 0
        ):
            train_dataset_current = _refresh_dataset_scf_warm_start_cache(
                train_dataset_current,
                params=state.params,
                functional=functional,
                training_config=training_config,
            )
        if guard_post_update and not reverted_step:
            guarded_train_loss, guarded_train_metrics = (
                compiled_train_eval(state.params, train_dataset_current)
                if use_warm_start
                else compiled_train_eval(state.params)
            )
            if not _loss_and_metrics_all_finite(guarded_train_loss, guarded_train_metrics):
                state = prev_state
                train_dataset_current = prev_train_dataset_current
                reverted_step = True
                post_update_recoveries += 1
                logger.log(
                    f"[train] non-finite post-update train eval at step {step}; reverted update"
                )

        train_loss_val = _metric_scalar(train_metrics, "loss")
        train_s1_mae_val = _metric_mean(train_metrics, "s1_mae", float("nan"))
        grad_norm_val = _metric_scalar(train_metrics, "grad_norm")
        grad_abs_max_val = _metric_scalar(train_metrics, "grad_abs_max")
        update_norm_val = (
            0.0 if reverted_step else _metric_scalar(train_metrics, "param_update_norm")
        )
        nonfinite_grad_fraction_val = _metric_scalar(train_metrics, "nonfinite_grad_fraction", 0.0)
        history_steps.append(step)
        loss_history.append(train_loss_val)
        s1_mae_history.append(train_s1_mae_val)
        grad_norm_history.append(grad_norm_val)
        grad_abs_max_history.append(grad_abs_max_val)
        param_update_norm_history.append(update_norm_val)
        nonfinite_grad_fraction_history.append(nonfinite_grad_fraction_val)

        should_eval = (
            step == 1
            or step == int(args.steps)
            or step % max(1, int(args.eval_interval)) == 0
        )
        if should_eval:
            eval_train_loss, eval_train_metrics = (
                compiled_train_eval(state.params, train_dataset_current)
                if use_warm_start
                else compiled_train_eval(state.params)
            )
            eval_steps.append(step)
            eval_train_loss_history.append(float(eval_train_loss))
            eval_train_s1_mae_history.append(_metric_mean(eval_train_metrics, "s1_mae", float("nan")))
            if compiled_val_eval is not None:
                eval_val_loss, eval_val_metrics = (
                    compiled_val_eval(state.params, val_dataset_current)
                    if use_warm_start
                    else compiled_val_eval(state.params)
                )
                if bool(args.scf_warm_start):
                    val_dataset_current = _refresh_dataset_scf_warm_start_cache(
                        val_dataset_current,
                        params=state.params,
                        functional=functional,
                        training_config=training_config,
                    )
                val_loss_val = float(eval_val_loss)
                val_s1_mae_val = _metric_mean(eval_val_metrics, "s1_mae", float("nan"))
                eval_val_loss_history.append(val_loss_val)
                eval_val_s1_mae_history.append(val_s1_mae_val)
                score = _ground_only_best_score(
                    args,
                    train_loss=float(eval_train_loss),
                    train_metrics=eval_train_metrics,
                    val_loss=float(eval_val_loss),
                    val_metrics=eval_val_metrics,
                    has_validation=True,
                )
            else:
                eval_val_loss_history.append(float("nan"))
                eval_val_s1_mae_history.append(float("nan"))
                score = _ground_only_best_score(
                    args,
                    train_loss=float(eval_train_loss),
                    train_metrics=eval_train_metrics,
                    val_loss=None,
                    val_metrics=None,
                    has_validation=False,
                )
            if score < best_score:
                best_score = score
                best_params = state.params
                best_step = step
            current_lr = float(lr_schedule(step - 1)) if lr_schedule is not None else float(args.learning_rate)
            logger.log(
                "[train] "
                f"step={step:4d}/{int(args.steps):4d} "
                f"train_loss={float(eval_train_loss):.8e} "
                f"train_s1_mae={_metric_mean(eval_train_metrics, 's1_mae', float('nan')):.8e} "
                f"val_loss={eval_val_loss_history[-1]:.8e} "
                f"val_s1_mae={eval_val_s1_mae_history[-1]:.8e} "
                f"grad_norm={grad_norm_val:.8e} "
                f"grad_abs_max={grad_abs_max_val:.8e} "
                f"update_norm={update_norm_val:.8e} "
                f"lr={current_lr:.8e} "
                f"recoveries={post_update_recoveries:d}"
            )

    elapsed_s = time.perf_counter() - t0
    final_train_loss, final_train_metrics = (
        compiled_train_eval(state.params, train_dataset_current)
        if use_warm_start
        else compiled_train_eval(state.params)
    )
    final_val_loss = None
    final_val_metrics = None
    if compiled_val_eval is not None:
        final_val_loss, final_val_metrics = (
            compiled_val_eval(state.params, val_dataset_current)
            if use_warm_start
            else compiled_val_eval(state.params)
        )
    logger.log(
        "[train] done "
        f"final_train_loss={float(final_train_loss):.8e} "
        f"best_val_s1_mae={best_score:.8e}@{best_step} "
        f"elapsed_s={elapsed_s:.2f}"
    )
    return {
        "functional": functional,
        "training_config": training_config,
        "best_params": best_params,
        "final_params": state.params,
        "best_step": int(best_step),
        "best_score": float(best_score),
        "final_train_loss": float(final_train_loss),
        "final_train_s1_mae": _metric_mean(final_train_metrics, "s1_mae", float("nan")),
        "final_val_loss": float(final_val_loss) if final_val_loss is not None else float("nan"),
        "final_val_s1_mae": _metric_mean(final_val_metrics, "s1_mae", float("nan")) if final_val_metrics is not None else float("nan"),
        "elapsed_s": float(elapsed_s),
        "history_steps": history_steps,
        "loss_history": loss_history,
        "s1_mae_history": s1_mae_history,
        "grad_norm_history": grad_norm_history,
        "grad_abs_max_history": grad_abs_max_history,
        "param_update_norm_history": param_update_norm_history,
        "nonfinite_grad_fraction_history": nonfinite_grad_fraction_history,
        "eval_steps": eval_steps,
        "eval_train_loss_history": eval_train_loss_history,
        "eval_train_s1_mae_history": eval_train_s1_mae_history,
        "eval_val_loss_history": eval_val_loss_history,
        "eval_val_s1_mae_history": eval_val_s1_mae_history,
        "post_update_recoveries": int(post_update_recoveries),
    }


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(outdir / "run.log")

    rows = _load_reference_rows(Path(args.reference_csv), basis=str(args.basis))
    train_rows = [row for row in rows if row.split == "train"]
    val_rows = [row for row in rows if row.split == "validation"]
    test_rows = [row for row in rows if row.split == "test"]
    if not train_rows:
        raise ValueError("Training split is empty.")
    logger.log(
        "Config: "
        f"reference_csv={args.reference_csv}, basis={args.basis}, xc={args.xc}, "
        f"steps={args.steps}, mode={args.training_mode}, include_hfx_channel={bool(args.include_hfx_channel)}, "
        f"response_hf_mode={args.response_hf_mode}, "
        f"include_pt2_channel={bool(args.include_pt2_channel)}, "
        f"pt2_channel_mode={args.pt2_channel_mode if bool(args.include_pt2_channel) else 'none'}, "
        f"response_df_mode={args.response_df_mode}, "
        f"response_two_electron_mode={args.response_two_electron_mode}, "
        f"response_ris=theta:{float(args.response_ris_theta):.6g}/J:{args.response_ris_j_fit}/K:{args.response_ris_k_fit}, "
        f"stream_train={bool(args.stream_train)}, stream_update_mode={args.stream_update_mode}, "
        f"train={len(train_rows)}, validation={len(val_rows)}, test={len(test_rows)}"
    )

    prepared_train = _prepare_references(train_rows, args=args, logger=logger)
    prepared_val = _prepare_references(val_rows, args=args, logger=logger)
    if bool(args.skip_final_evaluation):
        logger.log("[ref] skipped test split preparation because final evaluation is disabled")
        prepared_test = []
    else:
        prepared_test = _prepare_references(test_rows, args=args, logger=logger)

    train_dataset = _build_dataset(
        prepared_train,
        s1_weight=float(args.s1_weight),
        density_constraint_weight=float(args.density_constraint_weight),
    )
    val_dataset = _build_dataset(
        prepared_val,
        s1_weight=float(args.s1_weight),
        density_constraint_weight=float(args.density_constraint_weight),
    )

    training = _train(
        train_dataset,
        val_dataset,
        init_molecule=prepared_train[0].molecule,
        args=args,
        logger=logger,
    )

    params = training["best_params"]
    functional = training["functional"]
    training_config = training["training_config"]
    if bool(args.skip_final_evaluation):
        training_png = outdir / "training_loss.png"
        _plot_training_history(
            training_png,
            training,
            title=f"Closed-shell S1 {'TDA' if bool(args.s1_use_tda) else 'Casida'} training",
        )
        training_history_csv = outdir / "training_history.csv"
        _write_training_history_csv(training_history_csv, training)
        checkpoint_path = outdir / "neural_xc_params.msgpack"
        checkpoint_path, checkpoint_meta_path = save_params_checkpoint(
            checkpoint_path,
            params,
            metadata={
                "reference_csv": str(args.reference_csv),
                "basis": str(args.basis),
                "xc": str(args.xc),
                "training_mode": str(args.training_mode),
                "skip_final_evaluation": True,
                "include_hfx_channel": bool(args.include_hfx_channel),
                "response_hf_mode": str(args.response_hf_mode),
                "include_pt2_channel": bool(args.include_pt2_channel),
                "pt2_channel_mode": str(args.pt2_channel_mode) if bool(args.include_pt2_channel) else None,
                "scf_hfx_grid_block_size": _scf_hfx_grid_block_size(args),
                "stream_train": bool(args.stream_train),
                "stream_update_mode": str(args.stream_update_mode),
                "lr_decay_every": int(args.lr_decay_every),
                "lr_decay_factor": float(args.lr_decay_factor),
                "lr_transition_steps": _lr_transition_steps(args, train_size=len(train_dataset)),
                "s1_use_tda": bool(args.s1_use_tda),
                "eval_use_tda": bool(args.eval_use_tda),
                "steps": int(args.steps),
                "best_step": int(training["best_step"]),
                "train_systems": [row.system for row in train_rows],
                "validation_systems": [row.system for row in val_rows],
                "test_systems": [row.system for row in test_rows],
            },
        )
        summary = {
            "reference_csv": str(args.reference_csv),
            "basis": str(args.basis),
            "xc": str(args.xc),
            "training_mode": str(args.training_mode),
            "objective": f"s1_only_{'tda' if bool(args.s1_use_tda) else 'casida'}",
            "evaluation_solver": None,
            "skip_final_evaluation": True,
            "include_hfx_channel": bool(args.include_hfx_channel),
            "response_hf_mode": str(args.response_hf_mode),
            "include_pt2_channel": bool(args.include_pt2_channel),
            "pt2_channel_mode": str(args.pt2_channel_mode) if bool(args.include_pt2_channel) else None,
            "scf_hfx_grid_block_size": _scf_hfx_grid_block_size(args),
            "stream_train": bool(args.stream_train),
            "stream_update_mode": str(args.stream_update_mode),
            "lr_decay_every": int(args.lr_decay_every),
            "lr_decay_factor": float(args.lr_decay_factor),
            "lr_transition_steps": _lr_transition_steps(args, train_size=len(train_dataset)),
            "steps": int(args.steps),
            "best_step": int(training["best_step"]),
            "best_validation_s1_mae_h": float(training["best_score"]),
            "final_train_loss": float(training["final_train_loss"]),
            "final_train_s1_mae_h": float(training["final_train_s1_mae"]),
            "final_val_loss": float(training["final_val_loss"]),
            "final_val_s1_mae_h": float(training["final_val_s1_mae"]),
            "train_s1_mae_ev": None,
            "validation_s1_mae_ev": None,
            "test_s1_mae_ev": None,
            "predictions_csv": None,
            "training_curve_png": str(training_png),
            "training_history_csv": str(training_history_csv),
            "checkpoint": str(checkpoint_path),
            "checkpoint_meta": str(checkpoint_meta_path) if checkpoint_meta_path is not None else None,
            "train_systems": [row.system for row in train_rows],
            "validation_systems": [row.system for row in val_rows],
            "test_systems": [row.system for row in test_rows],
        }
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        logger.log("[eval] skipped final per-molecule prediction pass")
        logger.log(f"Wrote summary   : {outdir / 'summary.json'}")
        logger.log(f"Wrote checkpoint: {checkpoint_path}")
        return

    use_tda = bool(args.eval_use_tda)
    train_pred_rows, train_metrics = _evaluate_dataset(
        prepared_train,
        params=params,
        functional=functional,
        training_config=training_config,
        use_tda=use_tda,
    )
    val_pred_rows, val_metrics = _evaluate_dataset(
        prepared_val,
        params=params,
        functional=functional,
        training_config=training_config,
        use_tda=use_tda,
    )
    test_pred_rows, test_metrics = _evaluate_dataset(
        prepared_test,
        params=params,
        functional=functional,
        training_config=training_config,
        use_tda=use_tda,
    )

    predictions_csv = outdir / "predictions.csv"
    _write_prediction_csv(predictions_csv, train_pred_rows + val_pred_rows + test_pred_rows)
    training_png = outdir / "training_loss.png"
    _plot_training_history(
        training_png,
        training,
        title=f"Closed-shell S1 {'TDA' if bool(args.s1_use_tda) else 'Casida'} self-consistent training",
    )
    training_history_csv = outdir / "training_history.csv"
    _write_training_history_csv(training_history_csv, training)

    checkpoint_path = outdir / "neural_xc_params.msgpack"
    checkpoint_path, checkpoint_meta_path = save_params_checkpoint(
        checkpoint_path,
        params,
        metadata={
            "reference_csv": str(args.reference_csv),
            "basis": str(args.basis),
            "xc": str(args.xc),
            "training_mode": str(args.training_mode),
            "scf_warm_start": bool(args.scf_warm_start),
            "scf_warm_start_update_interval": int(args.scf_warm_start_update_interval),
            "recover_nonfinite_steps": bool(args.recover_nonfinite_steps),
            "scf_gradient_mode": str(args.scf_gradient_mode),
            "scf_implicit_diff_solver": str(args.scf_implicit_diff_solver),
            "scf_implicit_diff_tolerance": float(args.scf_implicit_diff_tolerance),
            "scf_implicit_diff_regularization": float(args.scf_implicit_diff_regularization),
            "scf_implicit_diff_restart": int(args.scf_implicit_diff_restart),
            "include_hfx_channel": bool(args.include_hfx_channel),
            "response_hf_mode": str(args.response_hf_mode),
            "include_pt2_channel": bool(args.include_pt2_channel),
            "pt2_channel_mode": str(args.pt2_channel_mode) if bool(args.include_pt2_channel) else None,
            "scf_hfx_grid_block_size": _scf_hfx_grid_block_size(args),
            "stream_train": bool(args.stream_train),
            "stream_update_mode": str(args.stream_update_mode),
            "lr_decay_every": int(args.lr_decay_every),
            "lr_decay_factor": float(args.lr_decay_factor),
            "lr_transition_steps": _lr_transition_steps(args, train_size=len(train_dataset)),
            "s1_use_tda": bool(args.s1_use_tda),
            "eval_use_tda": bool(args.eval_use_tda),
            "steps": int(args.steps),
            "best_step": int(training["best_step"]),
            "train_systems": [row.system for row in train_rows],
            "validation_systems": [row.system for row in val_rows],
            "test_systems": [row.system for row in test_rows],
        },
    )

    benzene_row = next((row for row in test_pred_rows if row["system"] == "benzene"), None)
    summary = {
        "reference_csv": str(args.reference_csv),
        "basis": str(args.basis),
        "xc": str(args.xc),
        "reference_jk_backend": str(args.reference_jk_backend),
        "training_mode": str(args.training_mode),
        "objective": f"s1_only_{'tda' if bool(args.s1_use_tda) else 'casida'}",
        "evaluation_solver": "tda" if bool(args.eval_use_tda) else "casida",
        "scf_warm_start": bool(args.scf_warm_start),
        "scf_warm_start_update_interval": int(args.scf_warm_start_update_interval),
        "recover_nonfinite_steps": bool(args.recover_nonfinite_steps),
        "scf_gradient_mode": str(args.scf_gradient_mode),
        "scf_implicit_diff_solver": str(args.scf_implicit_diff_solver),
        "scf_implicit_diff_tolerance": float(args.scf_implicit_diff_tolerance),
        "scf_implicit_diff_regularization": float(args.scf_implicit_diff_regularization),
        "scf_implicit_diff_restart": int(args.scf_implicit_diff_restart),
        "include_hfx_channel": bool(args.include_hfx_channel),
        "response_hf_mode": str(args.response_hf_mode),
        "include_pt2_channel": bool(args.include_pt2_channel),
        "pt2_channel_mode": str(args.pt2_channel_mode) if bool(args.include_pt2_channel) else None,
        "scf_hfx_grid_block_size": _scf_hfx_grid_block_size(args),
        "stream_train": bool(args.stream_train),
        "stream_update_mode": str(args.stream_update_mode),
        "lr_decay_every": int(args.lr_decay_every),
        "lr_decay_factor": float(args.lr_decay_factor),
        "lr_transition_steps": _lr_transition_steps(args, train_size=len(train_dataset)),
        "steps": int(args.steps),
        "best_step": int(training["best_step"]),
        "best_validation_s1_mae_h": float(training["best_score"]),
        "final_train_loss": float(training["final_train_loss"]),
        "final_train_s1_mae_h": float(training["final_train_s1_mae"]),
        "final_val_loss": float(training["final_val_loss"]),
        "final_val_s1_mae_h": float(training["final_val_s1_mae"]),
        "train_s1_mae_ev": float(train_metrics["s1_mae_ev"]),
        "validation_s1_mae_ev": float(val_metrics["s1_mae_ev"]),
        "test_s1_mae_ev": float(test_metrics["s1_mae_ev"]),
        "benzene_s1_predicted_ev": float(benzene_row["predicted_s1_ev"]) if benzene_row is not None else None,
        "benzene_s1_target_ev": float(benzene_row["target_s1_ev"]) if benzene_row is not None else None,
        "benzene_s1_abs_err_ev": float(benzene_row["s1_abs_err_ev"]) if benzene_row is not None else None,
        "predictions_csv": str(predictions_csv),
        "training_curve_png": str(training_png),
        "training_history_csv": str(training_history_csv),
        "checkpoint": str(checkpoint_path),
        "checkpoint_meta": str(checkpoint_meta_path) if checkpoint_meta_path is not None else None,
        "train_systems": [row.system for row in train_rows],
        "validation_systems": [row.system for row in val_rows],
        "test_systems": [row.system for row in test_rows],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    logger.log(f"Wrote predictions: {predictions_csv}")
    logger.log(f"Wrote summary   : {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
