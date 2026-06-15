from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax

from td_graddft import neural_xc
from td_graddft.data import (
    build_graddft_ground_atom_datum,
    load_graddft_ground_atom_records,
    split_graddft_ground_atom_records,
)
from td_graddft.data.hdf5_cache import (
    read_restricted_molecule,
    read_unrestricted_molecule,
    write_restricted_molecule,
    write_unrestricted_molecule,
)
from td_graddft.neural_xc import (
    DEFAULT_INPUT_FEATURE_MODE,
    DEFAULT_NETWORK_ARCHITECTURE,
    DEFAULT_NETWORK_HIDDEN_DIMS,
)
from td_graddft.training import (
    GroundStateCoreDatum,
    GroundStateCoreTrainingConfig,
    GroundStateDatum,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    ground_state_mse_loss,
    make_ground_state_loss_and_grad,
    save_params_checkpoint,
)

HARTREE_TO_EV = 27.211386245988
_DEFAULT_SEMILOCAL_XC = ("lda_x", "gga_x_b88", "lda_c_vwn_rpa", "gga_c_lyp")
_TRAIN_SCF_SAFETY_MAX_CYCLE = 512
_SCF_SUMMARY_KEYS = (
    "scf_converged_fraction",
    "scf_cycles_mean",
    "scf_cycles_max",
    "scf_selected_cycle_mean",
    "scf_best_cycle_mean",
    "scf_final_rms_mean",
    "scf_final_rms_max",
    "scf_selected_rms_mean",
    "scf_selected_rms_max",
    "scf_best_rms_mean",
    "scf_best_rms_max",
)
_SCF_DETAIL_KEYS = (
    "scf_converged",
    "scf_cycles",
    "scf_selected_cycle",
    "scf_best_cycle",
    "scf_final_rms_density",
    "scf_selected_rms_density",
    "scf_best_rms_density",
)


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class StopController:
    def __init__(self) -> None:
        self.requested = False
        self.signum: int | None = None

    def request(self, signum: int) -> None:
        self.requested = True
        self.signum = int(signum)


def _install_stop_controller() -> StopController:
    controller = StopController()

    def _handler(signum: int, _frame: Any) -> None:
        controller.request(signum)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
    return controller


def _parse_symbols(raw: str | None) -> tuple[str, ...] | None:
    if raw is None or not str(raw).strip():
        return None
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def _metric_scalar(metrics: dict[str, Any], key: str, default: float = float("nan")) -> float:
    if key not in metrics:
        return default
    arr = jnp.asarray(metrics[key])
    if int(arr.size) <= 0:
        return default
    return float(jnp.mean(arr))


def _tree_add(left: Any | None, right: Any) -> Any:
    if left is None:
        return right
    return jax.tree_util.tree_map(lambda a, b: a + b, left, right)


def _tree_scale(tree: Any, scale: float) -> Any:
    return jax.tree_util.tree_map(lambda value: value * scale, tree)


def _tree_l2_norm(tree: Any) -> float:
    leaves = [jnp.asarray(leaf) for leaf in jax.tree_util.tree_leaves(tree)]
    if not leaves:
        return 0.0
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)))


def _tree_all_finite(tree: Any) -> bool:
    return all(
        bool(jnp.all(jnp.isfinite(jnp.asarray(leaf))))
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def _finite_mean(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _finite_max(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.max(finite)) if finite else float("nan")


def _write_split_csv(path: Path, train_records: tuple[Any, ...], test_records: tuple[Any, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "split",
                "symbol",
                "spin",
                "charge",
                "target_energy_h",
                "source_row",
                "source_path",
                "energy_column",
            ),
        )
        writer.writeheader()
        for split, records in (("train", train_records), ("test", test_records)):
            for record in records:
                writer.writerow(
                    {
                        "split": split,
                        "symbol": record.symbol,
                        "spin": int(record.spin),
                        "charge": int(record.charge),
                        "target_energy_h": float(record.target_energy_h),
                        "source_row": record.source_row,
                        "source_path": record.source_path,
                        "energy_column": record.energy_column,
                    }
                )


def _write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def persist_training_history(outdir: str | Path, history: list[dict[str, Any]]) -> None:
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    _write_history_csv(path / "training_history.csv", history)
    (path / "training_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )


def _write_rows_csv(path: Path, rows: list[dict[str, Any]], *, append: bool = True) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)


def _json_cache_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_json_cache_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_cache_value(val) for key, val in sorted(value.items())}
    return repr(value)


def _atom_cache_key(
    record: Any,
    *,
    basis: str,
    molecule_kwargs: dict[str, Any],
) -> str:
    payload = {
        "version": 1,
        "symbol": str(record.symbol),
        "atom": str(record.atom),
        "unit": str(record.unit),
        "charge": int(record.charge),
        "spin": int(record.spin),
        "basis": str(basis),
        "molecule_kwargs": _json_cache_value(molecule_kwargs),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    symbol = str(record.symbol).replace("/", "_")
    return f"graddft_atoms_ground/v1/{symbol}/{digest[:2]}/{digest}"


def _reference_cache_path(value: str | Path | None, *, outdir: Path | None = None) -> Path | None:
    raw = "" if value is None else str(value).strip()
    if raw.lower() in {"none", "off", "false"}:
        return None
    if raw:
        return Path(raw)
    if outdir is None:
        return None
    return outdir / "reference_cache.h5"


def _create_or_replace_group(handle: Any, key: str) -> Any:
    parent_name, leaf_name = key.rsplit("/", 1)
    parent = handle.require_group(parent_name)
    if leaf_name in parent:
        del parent[leaf_name]
    return parent.create_group(leaf_name)


def _datum_from_cached_molecule(record: Any, molecule: Any) -> GroundStateDatum:
    return GroundStateDatum.from_parts(
        molecule,
        core=GroundStateCoreDatum(
            target_total_energy=jnp.asarray(record.target_energy_h, dtype=jnp.float64),
        ),
    )


def _load_cached_atom_datum(
    *,
    cache_path: Path | None,
    key: str,
    record: Any,
    hfx_nu_storage: str,
    hfx_nu_chunk_size: int,
    logger: RunLogger,
) -> GroundStateDatum | None:
    if cache_path is None or not cache_path.exists():
        return None
    try:
        import h5py

        with h5py.File(cache_path, "r") as handle:
            if key not in handle:
                return None
            molecule_group = handle[key]["molecule"]
            if int(record.spin) == 0:
                molecule = read_restricted_molecule(
                    molecule_group,
                    array_backend="jax",
                    hfx_nu_storage=str(hfx_nu_storage),
                    hfx_nu_chunk_size=int(hfx_nu_chunk_size),
                )
            else:
                molecule = read_unrestricted_molecule(
                    molecule_group,
                    array_backend="jax",
                    hfx_nu_storage=str(hfx_nu_storage),
                    hfx_nu_chunk_size=int(hfx_nu_chunk_size),
                )
    except Exception as exc:
        logger.log(
            f"[ref_cache] read error symbol={record.symbol} key={key}: {exc!r}; rebuilding"
        )
        return None
    logger.log(f"[ref_cache] hit symbol={record.symbol}: {cache_path}::{key}")
    return _datum_from_cached_molecule(record, molecule)


def _save_cached_atom_molecule(
    *,
    cache_path: Path | None,
    key: str,
    record: Any,
    basis: str,
    molecule_kwargs: dict[str, Any],
    molecule: Any,
    logger: RunLogger,
) -> None:
    if cache_path is None:
        return
    try:
        import h5py

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(cache_path, "a") as handle:
            group = _create_or_replace_group(handle, key)
            group.attrs["symbol"] = str(record.symbol)
            group.attrs["charge"] = int(record.charge)
            group.attrs["spin"] = int(record.spin)
            group.attrs["basis"] = str(basis)
            group.attrs["molecule_kwargs_json"] = json.dumps(
                _json_cache_value(molecule_kwargs),
                sort_keys=True,
            )
            molecule_group = group.require_group("molecule")
            if int(record.spin) == 0:
                write_restricted_molecule(molecule_group, molecule)
            else:
                write_unrestricted_molecule(molecule_group, molecule)
    except Exception as exc:
        logger.log(f"[ref_cache] write failed symbol={record.symbol}: {exc!r}")
        return
    logger.log(f"[ref_cache] wrote symbol={record.symbol}: {cache_path}::{key}")


def _build_atom_data(
    records: tuple[Any, ...],
    *,
    split: str,
    basis: str,
    molecule_kwargs: dict[str, Any],
    logger: RunLogger,
    reference_cache_path: str | Path | None = None,
    rebuild_reference_cache: bool = False,
    hfx_nu_storage: str = "array",
    hfx_nu_chunk_size: int = 512,
) -> tuple[GroundStateDatum, ...]:
    data = []
    total = len(records)
    cache_path = _reference_cache_path(reference_cache_path)
    for index, record in enumerate(records, start=1):
        cache_key = _atom_cache_key(
            record,
            basis=str(basis),
            molecule_kwargs=molecule_kwargs,
        )
        if not bool(rebuild_reference_cache):
            cached = _load_cached_atom_datum(
                cache_path=cache_path,
                key=cache_key,
                record=record,
                hfx_nu_storage=str(hfx_nu_storage),
                hfx_nu_chunk_size=int(hfx_nu_chunk_size),
                logger=logger,
            )
            if cached is not None:
                data.append(cached)
                continue
        logger.log(
            "[build] "
            f"{split} {index}/{total} symbol={record.symbol} spin={int(record.spin)} start"
        )
        t0 = time.perf_counter()
        datum = build_graddft_ground_atom_datum(
            record,
            basis=basis,
            **molecule_kwargs,
        )
        molecule = datum.molecule
        ao = getattr(molecule, "ao", None)
        ngrid = int(ao.shape[0]) if ao is not None and len(ao.shape) >= 1 else -1
        nao = int(ao.shape[-1]) if ao is not None and len(ao.shape) >= 1 else -1
        logger.log(
            "[build] "
            f"{split} {index}/{total} symbol={record.symbol} done "
            f"dt={time.perf_counter() - t0:.2f}s ngrid={ngrid} nao={nao} "
            f"mf_energy={float(getattr(molecule, 'mf_energy', float('nan'))):.10f}"
        )
        _save_cached_atom_molecule(
            cache_path=cache_path,
            key=cache_key,
            record=record,
            basis=str(basis),
            molecule_kwargs=molecule_kwargs,
            molecule=molecule,
            logger=logger,
        )
        data.append(datum)
    return tuple(data)


def _make_functional(args: argparse.Namespace) -> Any:
    return neural_xc.Functional(
        semilocal_xc=tuple(str(name) for name in args.semilocal_xc),
        hidden_dims=tuple(int(value) for value in args.hidden_dims),
        architecture=str(args.network_architecture),
        input_feature_mode=(
            "canonical"
            if str(args.input_feature_mode) == "dm21_original"
            else str(args.input_feature_mode)
        ),
        include_pt2_channel=False,
        name="neural_xc_graddft_atoms_ground",
    )


def _make_training_config(args: argparse.Namespace) -> GroundStateTrainingConfig:
    coefficient_prior = neural_xc.resolve_coefficient_prior_values(
        tuple(str(name) for name in args.semilocal_xc)
    )
    return GroundStateTrainingConfig.from_parts(
        core=GroundStateCoreTrainingConfig(
            mode=str(args.training_mode),
            energy_mse_weight=float(args.energy_mse_weight),
            energy_mae_weight=float(args.energy_mae_weight),
            energy_normalization=str(args.energy_normalization),
            coefficient_prior_weight=float(args.coefficient_prior_weight),
            coefficient_prior_values=coefficient_prior,
            scf_max_cycle=(
                _TRAIN_SCF_SAFETY_MAX_CYCLE
                if int(args.train_scf_max_cycle) <= 0
                else int(args.train_scf_max_cycle)
            ),
            scf_damping=float(args.train_scf_damping),
            scf_level_shift=float(args.train_scf_level_shift),
            scf_conv_tol_energy=args.train_scf_conv_tol_energy,
            scf_convergence_metric=str(args.train_scf_convergence_metric),
            scf_conv_tol_density=float(args.train_scf_conv_tol_density),
            scf_vxc_clip=float(args.train_scf_vxc_clip),
            scf_iterate_selection=str(args.scf_iterate_selection),
            scf_require_convergence=bool(args.scf_require_convergence),
            scf_gradient_mode=str(args.scf_gradient_mode),
            scf_implicit_diff_tolerance=float(args.scf_implicit_diff_tolerance),
            scf_implicit_diff_regularization=float(args.scf_implicit_diff_regularization),
        )
    )


def _empty_eval_metrics() -> dict[str, float]:
    values = {
        "loss": float("nan"),
        "energy_mae_h": float("nan"),
        "normalized_energy_mae": float("nan"),
    }
    values.update({key: float("nan") for key in _SCF_SUMMARY_KEYS})
    return values


def _eval_dataset_with_predictions(
    *,
    params: Any,
    data: tuple[GroundStateDatum, ...],
    eval_single: Any,
    records: tuple[Any, ...] | None = None,
    split: str | None = None,
    step: int | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    losses = []
    raw_maes = []
    norm_maes = []
    scf_summary_values = {key: [] for key in _SCF_SUMMARY_KEYS}
    rows = []
    if records is not None and len(records) != len(data):
        raise ValueError("records and data must have the same length")
    for index, datum in enumerate(data):
        loss_value, metrics = eval_single(params, datum)
        losses.append(float(loss_value))
        raw_maes.append(_metric_scalar(metrics, "energy_mae"))
        norm_maes.append(_metric_scalar(metrics, "normalized_energy_mae"))
        for key in _SCF_SUMMARY_KEYS:
            scf_summary_values[key].append(_metric_scalar(metrics, key))
        if records is not None and split is not None and step is not None:
            record = records[index]
            predicted = _metric_scalar(metrics, "predicted_total_energies")
            target = float(record.target_energy_h)
            row = {
                "step": int(step),
                "split": split,
                "symbol": record.symbol,
                "spin": int(record.spin),
                "charge": int(record.charge),
                "target_energy_h": target,
                "predicted_energy_h": predicted,
                "energy_error_h": predicted - target,
                "energy_abs_err_ev": abs(predicted - target) * HARTREE_TO_EV,
                "loss": float(loss_value),
                "energy_mae_h": _metric_scalar(metrics, "energy_mae"),
                "normalized_energy_mae": _metric_scalar(metrics, "normalized_energy_mae"),
            }
            for key in _SCF_DETAIL_KEYS:
                row[key] = _metric_scalar(metrics, key)
            rows.append(row)
    summary = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "energy_mae_h": float(np.mean(raw_maes)) if raw_maes else float("nan"),
        "normalized_energy_mae": float(np.mean(norm_maes)) if norm_maes else float("nan"),
    }
    summary.update(
        {
            key: _finite_mean(values)
            for key, values in scf_summary_values.items()
        }
    )
    return summary, rows


def _eval_dataset(
    *,
    params: Any,
    data: tuple[GroundStateDatum, ...],
    eval_single: Any,
) -> dict[str, float]:
    summary, _ = _eval_dataset_with_predictions(
        params=params,
        data=data,
        eval_single=eval_single,
    )
    return summary


def _history_row(
    *,
    step: int,
    batch_loss: float,
    batch_energy_mae_h: float,
    train_eval: dict[str, float],
    test_eval: dict[str, float],
    grad_norm: float,
    param_update_norm: float,
    lr: float,
    batch_scf_converged_fraction: float,
    batch_scf_cycles_mean: float,
    batch_scf_selected_rms_max: float,
) -> dict[str, float]:
    row = {
        "step": int(step),
        "batch_loss": float(batch_loss),
        "batch_energy_mae_h": float(batch_energy_mae_h),
        "batch_scf_converged_fraction": float(batch_scf_converged_fraction),
        "batch_scf_cycles_mean": float(batch_scf_cycles_mean),
        "batch_scf_selected_rms_max": float(batch_scf_selected_rms_max),
        "train_loss": float(train_eval["loss"]),
        "test_loss": float(test_eval["loss"]),
        "train_energy_mae_h": float(train_eval["energy_mae_h"]),
        "test_energy_mae_h": float(test_eval["energy_mae_h"]),
        "train_normalized_energy_mae": float(train_eval["normalized_energy_mae"]),
        "test_normalized_energy_mae": float(test_eval["normalized_energy_mae"]),
        "grad_norm": float(grad_norm),
        "param_update_norm": float(param_update_norm),
        "lr": float(lr),
    }
    for key in _SCF_SUMMARY_KEYS:
        row[f"train_{key}"] = float(train_eval[key])
        row[f"test_{key}"] = float(test_eval[key])
    return row


def train(
    *,
    train_records: tuple[Any, ...],
    train_data: tuple[GroundStateDatum, ...],
    test_records: tuple[Any, ...],
    test_data: tuple[GroundStateDatum, ...],
    args: argparse.Namespace,
    logger: RunLogger,
) -> dict[str, Any]:
    functional = _make_functional(args)
    training_config = _make_training_config(args)
    lr_schedule = optax.exponential_decay(
        init_value=float(args.learning_rate),
        transition_steps=max(1, int(args.lr_decay_every)),
        decay_rate=float(args.lr_decay_factor),
        staircase=True,
    )
    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(int(args.seed)),
        train_data[0].molecule,
        optax.adam(lr_schedule),
    )
    loss_and_grad = make_ground_state_loss_and_grad(
        functional,
        training_config=training_config,
    )
    eval_single = lambda params, datum: ground_state_mse_loss(  # noqa: E731
        params,
        functional,
        datum,
        training_config=training_config,
    )
    if bool(args.jit_train):
        loss_and_grad = jax.jit(loss_and_grad)
    if bool(args.jit_eval):
        eval_single = jax.jit(eval_single)

    outdir = Path(args.outdir)
    latest_checkpoint_path = outdir / "neural_xc_params_latest.msgpack"
    best_checkpoint_path = outdir / "neural_xc_params_best.msgpack"
    eval_prediction_path = outdir / str(args.eval_prediction_csv)
    if str(args.eval_prediction_csv).strip() and eval_prediction_path.exists():
        eval_prediction_path.unlink()
    save_params_checkpoint(latest_checkpoint_path, state.params)
    save_params_checkpoint(best_checkpoint_path, state.params)

    history: list[dict[str, float]] = []
    initial_train, initial_train_rows = _eval_dataset_with_predictions(
        params=state.params,
        data=train_data,
        eval_single=eval_single,
        records=train_records,
        split="train",
        step=0,
    )
    initial_test, initial_test_rows = _eval_dataset_with_predictions(
        params=state.params,
        data=test_data,
        eval_single=eval_single,
        records=test_records,
        split="test",
        step=0,
    )
    if str(args.eval_prediction_csv).strip():
        _write_rows_csv(eval_prediction_path, initial_train_rows + initial_test_rows)
    history.append(
        _history_row(
            step=0,
            batch_loss=float("nan"),
            batch_energy_mae_h=float("nan"),
            train_eval=initial_train,
            test_eval=initial_test,
            grad_norm=float("nan"),
            param_update_norm=float("nan"),
            lr=float(args.learning_rate),
            batch_scf_converged_fraction=float("nan"),
            batch_scf_cycles_mean=float("nan"),
            batch_scf_selected_rms_max=float("nan"),
        )
    )
    persist_training_history(args.outdir, history)
    best_params = state.params
    best_test_loss = initial_test["loss"]
    best_step = 0
    last_train_eval = initial_train
    last_test_eval = initial_test
    rng = np.random.default_rng(int(args.seed))
    train_indices = np.arange(len(train_data))
    rng.shuffle(train_indices)
    cursor = 0
    stopped_early = False
    stop_controller = getattr(args, "stop_controller", None)
    logger.log(
        "[train] "
        f"mode={args.training_mode} steps={int(args.steps)} batch_size={int(args.batch_size)} "
        f"train={len(train_data)} test={len(test_data)} lr={float(args.learning_rate):.6g}"
    )
    t0 = time.perf_counter()
    for step in range(1, int(args.steps) + 1):
        if cursor + int(args.batch_size) > len(train_indices):
            rng.shuffle(train_indices)
            cursor = 0
        batch_ids = train_indices[cursor : cursor + int(args.batch_size)]
        cursor += int(args.batch_size)
        prev_state = state
        grad_sum = None
        losses = []
        maes = []
        grad_norms = []
        batch_scf_converged = []
        batch_scf_cycles = []
        batch_scf_selected_rms = []
        for idx in batch_ids:
            loss_value, metrics, grads = loss_and_grad(state.params, train_data[int(idx)])
            losses.append(float(loss_value))
            maes.append(_metric_scalar(metrics, "energy_mae"))
            grad_norms.append(_metric_scalar(metrics, "grad_norm"))
            batch_scf_converged.append(_metric_scalar(metrics, "scf_converged"))
            batch_scf_cycles.append(_metric_scalar(metrics, "scf_cycles"))
            batch_scf_selected_rms.append(_metric_scalar(metrics, "scf_selected_rms_density"))
            grad_sum = _tree_add(grad_sum, grads)
        grads_avg = _tree_scale(grad_sum, 1.0 / max(1, len(batch_ids)))
        state = state.apply_gradients(grads=grads_avg)
        param_delta = jax.tree_util.tree_map(
            lambda new, old: new - old,
            state.params,
            prev_state.params,
        )
        if not _tree_all_finite(state.params):
            state = prev_state
            logger.log(f"[train] non-finite params at step {step}; reverted update")
        train_eval = _empty_eval_metrics()
        test_eval = _empty_eval_metrics()
        if step % int(args.eval_every) == 0 or step == int(args.steps):
            train_eval, train_rows = _eval_dataset_with_predictions(
                params=state.params,
                data=train_data,
                eval_single=eval_single,
                records=train_records,
                split="train",
                step=step,
            )
            test_eval, test_rows = _eval_dataset_with_predictions(
                params=state.params,
                data=test_data,
                eval_single=eval_single,
                records=test_records,
                split="test",
                step=step,
            )
            last_train_eval = train_eval
            last_test_eval = test_eval
            if str(args.eval_prediction_csv).strip():
                _write_rows_csv(eval_prediction_path, train_rows + test_rows)
            if test_eval["loss"] < best_test_loss:
                best_test_loss = test_eval["loss"]
                best_step = step
                best_params = state.params
                save_params_checkpoint(best_checkpoint_path, best_params)
        row = _history_row(
            step=step,
            batch_loss=float(np.mean(losses)),
            batch_energy_mae_h=float(np.mean(maes)),
            train_eval=train_eval,
            test_eval=test_eval,
            grad_norm=float(np.mean(grad_norms)),
            param_update_norm=float(_tree_l2_norm(param_delta)),
            lr=float(lr_schedule(step - 1)),
            batch_scf_converged_fraction=_finite_mean(batch_scf_converged),
            batch_scf_cycles_mean=_finite_mean(batch_scf_cycles),
            batch_scf_selected_rms_max=_finite_max(batch_scf_selected_rms),
        )
        history.append(row)
        persist_training_history(args.outdir, history)
        should_checkpoint = (
            int(args.checkpoint_every) > 0
            and (step % int(args.checkpoint_every) == 0 or step == int(args.steps))
        )
        if should_checkpoint:
            save_params_checkpoint(latest_checkpoint_path, state.params)
        if step == 1 or step % int(args.log_every) == 0 or step == int(args.steps):
            logger.log(
                "[train] "
                f"step={step:5d}/{int(args.steps):5d} "
                f"batch_loss={row['batch_loss']:.8e} "
                f"batch_energy_mae={row['batch_energy_mae_h']:.8e} "
                f"train_loss={row['train_loss']:.8e} test_loss={row['test_loss']:.8e} "
                f"train_scf_conv={row['train_scf_converged_fraction']:.8e} "
                f"test_scf_conv={row['test_scf_converged_fraction']:.8e} "
                f"batch_scf_conv={row['batch_scf_converged_fraction']:.8e} "
                f"grad_norm={row['grad_norm']:.8e} lr={row['lr']:.8e}"
            )
        if stop_controller is not None and bool(getattr(stop_controller, "requested", False)):
            stopped_early = True
            save_params_checkpoint(latest_checkpoint_path, state.params)
            logger.log(
                "[train] stop requested "
                f"signum={getattr(stop_controller, 'signum', None)} after_step={step}; "
                "saved latest checkpoint and ending without extra final eval"
            )
            break
    elapsed = time.perf_counter() - t0
    if stopped_early:
        final_train = last_train_eval
        final_test = last_test_eval
    else:
        final_train = _eval_dataset(params=state.params, data=train_data, eval_single=eval_single)
        final_test = _eval_dataset(params=state.params, data=test_data, eval_single=eval_single)
        if final_test["loss"] < best_test_loss:
            best_test_loss = final_test["loss"]
            best_step = int(history[-1]["step"])
            best_params = state.params
            save_params_checkpoint(best_checkpoint_path, best_params)
    save_params_checkpoint(latest_checkpoint_path, state.params)
    logger.log(
        "[train] done "
        f"final_train_loss={final_train['loss']:.8e} final_test_loss={final_test['loss']:.8e} "
        f"final_train_scf_conv={final_train['scf_converged_fraction']:.8e} "
        f"final_test_scf_conv={final_test['scf_converged_fraction']:.8e} "
        f"best_test_loss={best_test_loss:.8e}@{best_step} "
        f"stopped_early={stopped_early} elapsed_s={elapsed:.2f}"
    )
    return {
        "functional": functional,
        "training_config": training_config,
        "params": state.params,
        "best_params": best_params,
        "history": history,
        "elapsed_s": elapsed,
        "final_train": final_train,
        "final_test": final_test,
        "best_test_loss": best_test_loss,
        "best_step": best_step,
        "eval_single": eval_single,
        "stopped_early": stopped_early,
        "latest_checkpoint": str(latest_checkpoint_path),
        "best_checkpoint": str(best_checkpoint_path),
        "eval_prediction_csv": str(eval_prediction_path),
    }


def write_predictions(
    path: Path,
    *,
    records: tuple[Any, ...],
    data: tuple[GroundStateDatum, ...],
    split: str,
    params: Any,
    eval_single: Any,
) -> list[dict[str, Any]]:
    rows = []
    for record, datum in zip(records, data, strict=True):
        _, metrics = eval_single(params, datum)
        predicted = _metric_scalar(metrics, "predicted_total_energies")
        target = float(record.target_energy_h)
        rows.append(
            {
                "split": split,
                "symbol": record.symbol,
                "spin": int(record.spin),
                "charge": int(record.charge),
                "target_energy_h": target,
                "predicted_energy_h": predicted,
                "energy_error_h": predicted - target,
                "energy_abs_err_ev": abs(predicted - target) * HARTREE_TO_EV,
                "scf_converged": _metric_scalar(metrics, "scf_converged"),
                "scf_cycles": _metric_scalar(metrics, "scf_cycles"),
                "scf_selected_cycle": _metric_scalar(metrics, "scf_selected_cycle"),
                "scf_best_cycle": _metric_scalar(metrics, "scf_best_cycle"),
                "scf_final_rms_density": _metric_scalar(metrics, "scf_final_rms_density"),
                "scf_selected_rms_density": _metric_scalar(metrics, "scf_selected_rms_density"),
                "scf_best_rms_density": _metric_scalar(metrics, "scf_best_rms_density"),
            }
        )
    _write_rows_csv(path, rows)
    return rows


def plot_outputs(outdir: Path, history: list[dict[str, Any]], pred_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.asarray([row["step"] for row in history], dtype=float)
    batch_loss = np.asarray([row["batch_loss"] for row in history], dtype=float)
    train_loss = np.asarray([row["train_loss"] for row in history], dtype=float)
    test_loss = np.asarray([row["test_loss"] for row in history], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(
        steps[np.isfinite(batch_loss)],
        batch_loss[np.isfinite(batch_loss)],
        lw=1.0,
        alpha=0.55,
        label="batch",
    )
    ax.plot(
        steps[np.isfinite(train_loss)],
        train_loss[np.isfinite(train_loss)],
        marker="o",
        lw=1.5,
        label="train eval",
    )
    ax.plot(
        steps[np.isfinite(test_loss)],
        test_loss[np.isfinite(test_loss)],
        marker="s",
        lw=1.5,
        label="test eval",
    )
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "training_loss.png", dpi=220)
    plt.close(fig)

    train = [row for row in pred_rows if row["split"] == "train"]
    test = [row for row in pred_rows if row["split"] == "test"]
    if not pred_rows:
        return
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    for label, rows, marker in (("train", train, "o"), ("test", test, "s")):
        if not rows:
            continue
        target = np.asarray([row["target_energy_h"] for row in rows], dtype=float)
        pred = np.asarray([row["predicted_energy_h"] for row in rows], dtype=float)
        ax.scatter(target, pred, s=18, alpha=0.75, marker=marker, label=label)
    all_e = np.asarray(
        [row["target_energy_h"] for row in pred_rows]
        + [row["predicted_energy_h"] for row in pred_rows],
        dtype=float,
    )
    lo, hi = float(np.min(all_e)), float(np.max(all_e))
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.0)
    ax.set_xlabel("target energy (Ha)")
    ax.set_ylabel("predicted energy (Ha)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "graddft_atoms_ground_energy_scatter.png", dpi=220)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Neural XC on GradDFT XND ground-state atoms.")
    p.add_argument("--xlsx", default="data/raw/XND_dataset.xlsx")
    p.add_argument("--outdir", default="outputs/graddft_atoms_ground")
    p.add_argument("--prediction-csv", default="graddft_atoms_ground_predictions.csv")
    p.add_argument("--eval-prediction-csv", default="graddft_atoms_ground_eval_predictions.csv")
    p.add_argument("--symbols", default=None, help="Optional comma-separated atom symbols, e.g. H,He,Li.")
    p.add_argument("--test-train-ratio", default="2:8")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--basis", default="def2-svp")
    p.add_argument("--xc", default="b3lyp")
    p.add_argument("--grids-level", type=int, default=2)
    p.add_argument("--max-l", type=int, default=3)
    p.add_argument(
        "--reference-builder",
        choices=("pyscf", "jax"),
        default="pyscf",
        help="Build cached reference inputs with PySCF or the legacy JAX molecule builder.",
    )
    p.add_argument(
        "--reference-cache",
        default="",
        help=(
            "HDF5 cache for per-atom reference inputs. Empty defaults to "
            "<outdir>/reference_cache.h5; use off/none/false to disable."
        ),
    )
    p.add_argument("--rebuild-reference-cache", action="store_true")
    p.add_argument("--hfx-nu-storage", choices=("array", "chunked"), default="array")
    p.add_argument("--hfx-nu-chunk-size", type=int, default=512)
    p.add_argument("--integral-backend", choices=("cpu", "gpu", "jax", "libcint"), default="gpu")
    p.add_argument("--reference-scf-max-cycle", type=int, default=512)
    p.add_argument("--reference-scf-conv-tol", type=float, default=1e-10)
    p.add_argument("--reference-scf-conv-tol-density", type=float, default=1e-8)
    p.add_argument("--reference-scf-damping", type=float, default=0.25)
    p.add_argument("--reference-scf-level-shift", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--lr-decay-every", type=int, default=200)
    p.add_argument("--lr-decay-factor", type=float, default=0.5)
    p.add_argument("--training-mode", choices=("fixed_density", "self_consistent"), default="fixed_density")
    p.add_argument("--energy-mse-weight", type=float, default=1.0)
    p.add_argument("--energy-mae-weight", type=float, default=1.0)
    p.add_argument("--energy-normalization", choices=("none", "per_electron", "per_atom"), default="per_electron")
    p.add_argument("--coefficient-prior-weight", type=float, default=0.0)
    p.add_argument("--train-scf-max-cycle", type=int, default=0)
    p.add_argument("--train-scf-damping", type=float, default=0.25)
    p.add_argument("--train-scf-level-shift", type=float, default=0.0)
    p.add_argument("--train-scf-conv-tol-energy", type=float, default=1e-10)
    p.add_argument("--train-scf-convergence-metric", choices=("energy_and_residual", "energy"), default="energy")
    p.add_argument("--train-scf-conv-tol-density", type=float, default=1e-8)
    p.add_argument("--train-scf-vxc-clip", type=float, default=20.0)
    p.add_argument("--scf-iterate-selection", choices=("final", "best_rms", "first_converged"), default="final")
    p.add_argument("--scf-require-convergence", action="store_true")
    p.add_argument("--scf-gradient-mode", choices=("impl", "expl"), default="impl")
    p.add_argument("--scf-implicit-diff-tolerance", type=float, default=1e-6)
    p.add_argument("--scf-implicit-diff-regularization", type=float, default=0.0)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=list(DEFAULT_NETWORK_HIDDEN_DIMS))
    p.add_argument("--network-architecture", choices=("simple_mlp", "graddft_residual"), default=DEFAULT_NETWORK_ARCHITECTURE)
    p.add_argument("--input-feature-mode", choices=("enhanced", "canonical", "dm21_original"), default=DEFAULT_INPUT_FEATURE_MODE)
    p.add_argument("--semilocal-xc", nargs="+", default=list(_DEFAULT_SEMILOCAL_XC))
    p.add_argument("--jit-train", action="store_true")
    p.add_argument("--jit-eval", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=50, help="Save latest params every N steps; <=0 disables periodic checkpoints.")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--verbose", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(outdir / "run.log")
    args.stop_controller = _install_stop_controller()
    symbols = _parse_symbols(args.symbols)
    records = load_graddft_ground_atom_records(Path(args.xlsx), symbols=symbols)
    reference_cache_path = _reference_cache_path(args.reference_cache, outdir=outdir)
    compute_hfx = str(args.input_feature_mode) in {"canonical", "dm21_original"}
    logger.log(
        "[setup] "
        f"xlsx={args.xlsx} records={len(records)} ratio={args.test_train_ratio} "
        f"basis={args.basis} xc={args.xc} grid={args.grids_level} "
        f"reference_builder={args.reference_builder} reference_cache={reference_cache_path} "
        f"integral={args.integral_backend} compute_hfx={compute_hfx} "
        f"hfx_nu_storage={args.hfx_nu_storage}"
    )
    split = split_graddft_ground_atom_records(
        records,
        test_train_ratio=str(args.test_train_ratio),
        seed=int(args.seed),
    )
    _write_split_csv(outdir / "split.csv", split.train_records, split.test_records)
    molecule_kwargs = {
        "reference_builder": str(args.reference_builder),
        "xc_spec": str(args.xc),
        "grids_level": int(args.grids_level),
        "max_l": int(args.max_l),
        "integral_backend": str(args.integral_backend),
        "init_guess": "1e",
        "scf_max_cycle": int(args.reference_scf_max_cycle),
        "scf_conv_tol": float(args.reference_scf_conv_tol),
        "scf_conv_tol_density": float(args.reference_scf_conv_tol_density),
        "scf_damping": float(args.reference_scf_damping),
        "scf_level_shift": float(args.reference_scf_level_shift),
        "compute_local_hfx_features": compute_hfx,
        "compute_local_hfx_aux": compute_hfx,
        "hfx_nu_storage": str(args.hfx_nu_storage),
        "hfx_chunk_size": int(args.hfx_nu_chunk_size),
        "verbose": int(args.verbose),
    }
    logger.log(
        "[setup] split ready "
        f"train={len(split.train_records)} test={len(split.test_records)} "
        f"train_symbols={' '.join(record.symbol for record in split.train_records)} "
        f"test_symbols={' '.join(record.symbol for record in split.test_records)}"
    )
    train_data = _build_atom_data(
        split.train_records,
        split="train",
        basis=str(args.basis),
        molecule_kwargs=molecule_kwargs,
        logger=logger,
        reference_cache_path=reference_cache_path,
        rebuild_reference_cache=bool(args.rebuild_reference_cache),
        hfx_nu_storage=str(args.hfx_nu_storage),
        hfx_nu_chunk_size=int(args.hfx_nu_chunk_size),
    )
    test_data = _build_atom_data(
        split.test_records,
        split="test",
        basis=str(args.basis),
        molecule_kwargs=molecule_kwargs,
        logger=logger,
        reference_cache_path=reference_cache_path,
        rebuild_reference_cache=bool(args.rebuild_reference_cache),
        hfx_nu_storage=str(args.hfx_nu_storage),
        hfx_nu_chunk_size=int(args.hfx_nu_chunk_size),
    )
    logger.log(f"[setup] data ready train={len(train_data)} test={len(test_data)}")
    result = train(
        train_records=split.train_records,
        train_data=train_data,
        test_records=split.test_records,
        test_data=test_data,
        args=args,
        logger=logger,
    )
    persist_training_history(outdir, result["history"])
    checkpoint_path = outdir / "neural_xc_params.msgpack"
    save_params_checkpoint(checkpoint_path, result["best_params"])
    pred_path = outdir / str(args.prediction_csv)
    if pred_path.exists():
        pred_path.unlink()
    pred_rows = []
    if not bool(result["stopped_early"]):
        pred_rows.extend(
            write_predictions(
                pred_path,
                records=split.train_records,
                data=train_data,
                split="train",
                params=result["best_params"],
                eval_single=result["eval_single"],
            )
        )
        pred_rows.extend(
            write_predictions(
                pred_path,
                records=split.test_records,
                data=test_data,
                split="test",
                params=result["best_params"],
                eval_single=result["eval_single"],
            )
        )
    plot_outputs(outdir, result["history"], pred_rows)
    summary = {
        "xlsx": str(args.xlsx),
        "basis": str(args.basis),
        "xc": str(args.xc),
        "grids_level": int(args.grids_level),
        "reference_builder": str(args.reference_builder),
        "reference_cache": str(reference_cache_path) if reference_cache_path is not None else None,
        "rebuild_reference_cache": bool(args.rebuild_reference_cache),
        "hfx_nu_storage": str(args.hfx_nu_storage),
        "hfx_nu_chunk_size": int(args.hfx_nu_chunk_size),
        "integral_backend": str(args.integral_backend),
        "input_feature_mode": str(args.input_feature_mode),
        "training_mode": str(args.training_mode),
        "energy_normalization": str(args.energy_normalization),
        "seed": int(args.seed),
        "test_train_ratio": str(args.test_train_ratio),
        "symbols": [record.symbol for record in records],
        "train_symbols": [record.symbol for record in split.train_records],
        "test_symbols": [record.symbol for record in split.test_records],
        "train_count": len(split.train_records),
        "test_count": len(split.test_records),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "lr_decay_every": int(args.lr_decay_every),
        "lr_decay_factor": float(args.lr_decay_factor),
        "checkpoint_every": int(args.checkpoint_every),
        "stopped_early": bool(result["stopped_early"]),
        "final_train_loss": result["final_train"]["loss"],
        "final_test_loss": result["final_test"]["loss"],
        "final_train_energy_mae_ev": result["final_train"]["energy_mae_h"] * HARTREE_TO_EV,
        "final_test_energy_mae_ev": result["final_test"]["energy_mae_h"] * HARTREE_TO_EV,
        "final_train_scf_converged_fraction": result["final_train"]["scf_converged_fraction"],
        "final_test_scf_converged_fraction": result["final_test"]["scf_converged_fraction"],
        "final_train_scf_cycles_mean": result["final_train"]["scf_cycles_mean"],
        "final_test_scf_cycles_mean": result["final_test"]["scf_cycles_mean"],
        "final_train_scf_selected_rms_max": result["final_train"]["scf_selected_rms_max"],
        "final_test_scf_selected_rms_max": result["final_test"]["scf_selected_rms_max"],
        "best_test_loss": result["best_test_loss"],
        "best_step": int(result["best_step"]),
        "elapsed_s": float(result["elapsed_s"]),
        "history_csv": str(outdir / "training_history.csv"),
        "history_json": str(outdir / "training_history.json"),
        "prediction_csv": str(pred_path) if pred_rows else None,
        "eval_prediction_csv": result["eval_prediction_csv"],
        "split_csv": str(outdir / "split.csv"),
        "training_curve_png": str(outdir / "training_loss.png"),
        "prediction_scatter_png": (
            str(outdir / "graddft_atoms_ground_energy_scatter.png") if pred_rows else None
        ),
        "checkpoint": str(checkpoint_path),
        "latest_checkpoint": result["latest_checkpoint"],
        "best_checkpoint": result["best_checkpoint"],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.log(
        "[summary] "
        f"final_train_mae={summary['final_train_energy_mae_ev']:.8e} eV "
        f"final_test_mae={summary['final_test_energy_mae_ev']:.8e} eV "
        f"outdir={outdir}"
    )
    return summary


if __name__ == "__main__":
    main()
