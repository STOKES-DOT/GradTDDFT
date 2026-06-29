from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import h5py
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax

from td_graddft import neural_xc
from td_graddft.data.hdf5_cache import read_unrestricted_molecule, write_unrestricted_molecule
from td_graddft.data.reference import unrestricted_reference_from_pyscf
from td_graddft.neural_xc import (
    DEFAULT_INPUT_FEATURE_MODE,
    DEFAULT_NETWORK_ARCHITECTURE,
    DEFAULT_NETWORK_HIDDEN_DIMS,
)
from td_graddft.scf import UKSConfig, unrestricted_molecule_from_spec_with_jax_uks
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


@dataclass(frozen=True)
class XNDDimerRow:
    row_index: int
    atom1: str
    atom2: str
    bond_distance_angstrom: float
    multiplicity: int
    spin: int
    target_energy_h: float


@dataclass(frozen=True)
class ReferencePoint:
    row: XNDDimerRow
    molecule: Any


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


def _read_xnd_dimers_csv(path: Path) -> list[XNDDimerRow]:
    rows: list[XNDDimerRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for idx, row in enumerate(csv.DictReader(handle)):
            rows.append(
                XNDDimerRow(
                    row_index=idx,
                    atom1=str(row["atom1"]),
                    atom2=str(row["atom2"]),
                    bond_distance_angstrom=float(row["bond_distance_angstrom"]),
                    multiplicity=int(row["multiplicity"]),
                    spin=int(row["spin"]),
                    target_energy_h=float(row["target_energy_h"]),
                )
            )
    return rows


def _split_rows(
    rows: list[XNDDimerRow],
    *,
    seed: int,
    test_fraction: float,
    max_points: int | None,
) -> tuple[list[XNDDimerRow], list[XNDDimerRow]]:
    rng = np.random.default_rng(int(seed))
    indices = np.arange(len(rows))
    rng.shuffle(indices)
    if max_points is not None:
        indices = indices[: int(max_points)]
    n_test = max(1, int(round(float(test_fraction) * len(indices))))
    test_ids = set(int(i) for i in indices[:n_test])
    train_rows = [row for row in rows if row.row_index in set(int(i) for i in indices[n_test:])]
    test_rows = [row for row in rows if row.row_index in test_ids]
    return train_rows, test_rows


def _atom_string(row: XNDDimerRow) -> str:
    r = float(row.bond_distance_angstrom)
    return f"{row.atom1} 0.0 0.0 0.0; {row.atom2} {r:.12f} 0.0 0.0"


def _cache_key(row: XNDDimerRow, args: argparse.Namespace) -> str:
    basis = str(args.basis).replace("/", "_")
    xc = str(args.xc).replace("/", "_")
    feature_mode = "canonical" if str(args.input_feature_mode) == "dm21_original" else str(args.input_feature_mode)
    pt2 = "pt2" if bool(args.include_pt2_channel) else "nopt2"
    pair = f"{row.atom1}-{row.atom2}".replace("/", "_")
    return (
        f"xnd_dimers/basis={basis}/xc={xc}/grid={int(args.grids_level)}/"
        f"max_l={int(args.max_l)}/integral={str(args.integral_backend)}/"
        f"reference={str(args.reference_builder)}/"
        f"jk={str(args.jk_backend)}/df_tol={float(args.df_tol):.1e}/df_rank={args.df_max_rank}/"
        f"mode={str(args.training_mode)}/{pt2}_{str(args.pt2_channel_mode)}/"
        f"features={feature_mode}/row={row.row_index:04d}_{pair}_spin={row.spin}"
    )


def _write_reference(group: Any, point: ReferencePoint) -> None:
    row = point.row
    group.attrs["row_index"] = int(row.row_index)
    group.attrs["atom1"] = str(row.atom1)
    group.attrs["atom2"] = str(row.atom2)
    group.attrs["bond_distance_angstrom"] = float(row.bond_distance_angstrom)
    group.attrs["multiplicity"] = int(row.multiplicity)
    group.attrs["spin"] = int(row.spin)
    group.attrs["target_energy_h"] = float(row.target_energy_h)
    write_unrestricted_molecule(group.require_group("molecule"), point.molecule)


def _read_reference(
    group: Any,
    *,
    array_backend: str = "jax",
    hfx_nu_storage: str = "array",
) -> ReferencePoint:
    row = XNDDimerRow(
        row_index=int(group.attrs["row_index"]),
        atom1=str(group.attrs["atom1"]),
        atom2=str(group.attrs["atom2"]),
        bond_distance_angstrom=float(group.attrs["bond_distance_angstrom"]),
        multiplicity=int(group.attrs["multiplicity"]),
        spin=int(group.attrs["spin"]),
        target_energy_h=float(group.attrs["target_energy_h"]),
    )
    return ReferencePoint(
        row=row,
        molecule=read_unrestricted_molecule(
            group["molecule"],
            array_backend=array_backend,
            hfx_nu_storage=hfx_nu_storage,
        ),
    )


def build_reference(row: XNDDimerRow, *, args: argparse.Namespace) -> ReferencePoint:
    feature_mode = str(args.input_feature_mode)
    compute_hfx = feature_mode in {"canonical", "dm21_original"} or bool(args.include_hfx_channel)
    compute_pt2 = bool(args.include_pt2_channel)
    if str(args.reference_builder) == "pyscf":
        from pyscf import dft, gto

        mol = gto.M(
            atom=_atom_string(row),
            basis=str(args.basis),
            unit="Angstrom",
            charge=0,
            spin=int(row.spin),
            cart=True,
            verbose=int(args.verbose),
        )
        mf = dft.UKS(mol)
        mf.xc = str(args.xc)
        if str(args.jk_backend) == "df":
            mf = mf.density_fit()
        mf.grids.level = int(args.grids_level)
        mf.max_cycle = int(args.reference_scf_max_cycle)
        mf.conv_tol = float(args.reference_scf_conv_tol)
        if hasattr(mf, "conv_tol_grad"):
            mf.conv_tol_grad = float(args.reference_scf_conv_tol_density)
        if hasattr(mf, "damp"):
            mf.damp = float(args.reference_scf_damping)
        if hasattr(mf, "level_shift"):
            mf.level_shift = float(args.reference_scf_level_shift)
        mf.kernel()
        molecule = unrestricted_reference_from_pyscf(
            mf,
            compute_local_hfx_features=compute_hfx,
            compute_local_hfx_aux=compute_hfx,
            compute_local_pt2_features=compute_pt2,
            hfx_nu_storage="chunked",
            jk_backend=str(args.jk_backend),
            array_backend=("host" if bool(args.reference_only) else "jax"),
        )
        return ReferencePoint(row=row, molecule=molecule)
    molecule = unrestricted_molecule_from_spec_with_jax_uks(
        atom=_atom_string(row),
        basis=str(args.basis),
        xc_spec=str(args.xc),
        unit="Angstrom",
        charge=0,
        spin=int(row.spin),
        cart=True,
        grids_level=int(args.grids_level),
        max_l=int(args.max_l),
        uks_config=UKSConfig(
            xc_spec=str(args.xc),
            max_cycle=int(args.reference_scf_max_cycle),
            conv_tol=float(args.reference_scf_conv_tol),
            conv_tol_density=float(args.reference_scf_conv_tol_density),
            damping=float(args.reference_scf_damping),
            level_shift=float(args.reference_scf_level_shift),
            potential_clip=float(args.reference_scf_potential_clip),
            jk_backend=str(args.jk_backend),
            df_tol=float(args.df_tol),
            df_max_rank=args.df_max_rank,
        ),
        grid_ao_backend="jax",
        integral_backend=str(args.integral_backend),
        compute_local_hfx_features=compute_hfx,
        compute_local_hfx_aux=compute_hfx,
        compute_local_pt2_features=compute_pt2,
        verbose=int(args.verbose),
    )
    return ReferencePoint(row=row, molecule=molecule)


def get_or_build_reference(
    row: XNDDimerRow,
    *,
    args: argparse.Namespace,
    logger: RunLogger,
) -> ReferencePoint | None:
    cache_path = Path(args.reference_cache)
    key = _cache_key(row, args)
    if cache_path.exists() and not bool(args.rebuild_reference_cache):
        try:
            with h5py.File(cache_path, "r") as handle:
                if key in handle:
                    logger.log(f"[ref_cache] hit row={row.row_index} {row.atom1}-{row.atom2}")
                    return _read_reference(
                        handle[key],
                        array_backend="host",
                        hfx_nu_storage=("chunked" if bool(args.include_hfx_channel) else "array"),
                    )
        except Exception as exc:
            logger.log(f"[ref_cache] read error row={row.row_index}: {exc!r}; rebuilding")
    try:
        point = build_reference(row, args=args)
    except Exception as exc:
        logger.log(
            f"[ref] failed row={row.row_index} {row.atom1}-{row.atom2} "
            f"R={row.bond_distance_angstrom:.6f} spin={row.spin}: {exc!r}"
        )
        if bool(args.fail_on_build_error):
            raise
        return None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(cache_path, "a") as handle:
        if key in handle:
            del handle[key]
        _write_reference(handle.create_group(key), point)
    logger.log(
        f"[ref] built row={row.row_index} {row.atom1}-{row.atom2} "
        f"R={row.bond_distance_angstrom:.6f} spin={row.spin} "
        f"target={row.target_energy_h:.10f} Eh"
    )
    return point


def reference_exists(row: XNDDimerRow, *, args: argparse.Namespace) -> bool:
    cache_path = Path(args.reference_cache)
    if not cache_path.exists() or bool(args.rebuild_reference_cache):
        return True
    with h5py.File(cache_path, "r") as handle:
        return _cache_key(row, args) in handle


def build_reference_cache_only(
    rows: list[XNDDimerRow],
    *,
    args: argparse.Namespace,
    logger: RunLogger,
) -> int:
    count = 0
    for row in rows:
        point = get_or_build_reference(row, args=args, logger=logger)
        if point is not None:
            count += 1
        del point
        gc.collect()
        try:
            jax.clear_caches()
        except Exception:
            pass
    return count


def build_datum(point: ReferencePoint) -> GroundStateDatum:
    return GroundStateDatum.from_parts(
        point.molecule,
        core=GroundStateCoreDatum(
            target_total_energy=jnp.asarray(point.row.target_energy_h, dtype=jnp.float64),
        ),
    )


def load_reference_datum(
    row: XNDDimerRow,
    *,
    args: argparse.Namespace,
    logger: RunLogger,
) -> tuple[ReferencePoint, GroundStateDatum] | None:
    point = get_or_build_reference(row, args=args, logger=logger)
    if point is None:
        return None
    return point, build_datum(point)


def _write_split_csv(path: Path, train_rows: list[XNDDimerRow], test_rows: list[XNDDimerRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "split",
                "row_index",
                "atom1",
                "atom2",
                "bond_distance_angstrom",
                "multiplicity",
                "spin",
                "target_energy_h",
            ),
        )
        writer.writeheader()
        for split, rows in (("train", train_rows), ("test", test_rows)):
            for row in rows:
                writer.writerow({"split": split, **row.__dict__})


def train(
    train_rows: list[XNDDimerRow],
    test_rows: list[XNDDimerRow],
    *,
    args: argparse.Namespace,
    logger: RunLogger,
) -> dict[str, Any]:
    functional = neural_xc.Functional(
        semilocal_xc=tuple(str(name) for name in args.semilocal_xc),
        hidden_dims=tuple(int(value) for value in args.hidden_dims),
        architecture=str(args.network_architecture),
        input_feature_mode=(
            "canonical" if str(args.input_feature_mode) == "dm21_original" else str(args.input_feature_mode)
        ),
        include_hfx_channel=bool(args.include_hfx_channel),
        include_pt2_channel=bool(args.include_pt2_channel),
        pt2_channel_mode=str(args.pt2_channel_mode),
        name="neural_xc_xnd_dimers_ground",
    )
    coefficient_prior = neural_xc.resolve_coefficient_prior_values(
        tuple(str(name) for name in args.semilocal_xc)
    )
    training_config = GroundStateTrainingConfig.from_parts(
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
    lr_schedule = optax.exponential_decay(
        init_value=float(args.learning_rate),
        transition_steps=max(1, int(args.lr_decay_every)),
        decay_rate=float(args.lr_decay_factor),
        staircase=True,
    )
    init_loaded = None
    for row in train_rows:
        init_loaded = load_reference_datum(row, args=args, logger=logger)
        if init_loaded is not None:
            break
    if init_loaded is None:
        raise RuntimeError("Need at least one train reference to initialize the model.")
    init_point, init_datum = init_loaded
    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(int(args.seed)),
        init_point.molecule,
        optax.adam(lr_schedule),
    )
    del init_loaded, init_point, init_datum
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

    def eval_data(rows: list[XNDDimerRow], *, label: str) -> dict[str, float]:
        logger.log(f"[eval] {label} start n={len(rows)}")
        losses = []
        raw_maes = []
        norm_maes = []
        conv = []
        progress_every = max(1, min(25, len(rows)))
        for row_index, row in enumerate(rows, start=1):
            loaded = load_reference_datum(row, args=args, logger=logger)
            if loaded is None:
                continue
            point, datum = loaded
            loss_val, metrics = eval_single(state.params, datum)
            losses.append(float(loss_val))
            raw_maes.append(_metric_scalar(metrics, "energy_mae"))
            norm_maes.append(_metric_scalar(metrics, "normalized_energy_mae"))
            conv.append(_metric_scalar(metrics, "scf_converged", 1.0))
            del loaded, point, datum
            if row_index % progress_every == 0 or row_index == len(rows):
                logger.log(f"[eval] {label} progress {row_index}/{len(rows)}")
        result = {
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "energy_mae_h": float(np.mean(raw_maes)) if raw_maes else float("nan"),
            "normalized_energy_mae": float(np.mean(norm_maes)) if norm_maes else float("nan"),
            "scf_converged_fraction": float(np.mean(conv)) if conv else float("nan"),
        }
        logger.log(
            f"[eval] {label} done loss={result['loss']:.8e} "
            f"energy_mae={result['energy_mae_h']:.8e} "
            f"scf_converged={result['scf_converged_fraction']:.6f}"
        )
        return result

    rng = np.random.default_rng(int(args.seed))
    train_indices = np.arange(len(train_rows))
    rng.shuffle(train_indices)
    cursor = 0
    history: list[dict[str, float]] = []
    history_path = Path(args.outdir) / "training_history.csv"

    def flush_history() -> None:
        write_history(history_path, history)

    best_params = state.params
    best_test_loss = float("inf")
    best_step = 0

    if bool(args.skip_initial_eval):
        initial_train = {"loss": float("nan"), "energy_mae_h": float("nan")}
        initial_test = {"loss": float("nan"), "energy_mae_h": float("nan")}
        logger.log("[train_init] skipped initial eval")
    else:
        initial_train = eval_data(train_rows, label="initial_train")
        initial_test = eval_data(test_rows, label="initial_test")
        logger.log(
            "[train_init] "
            f"train_loss={initial_train['loss']:.8e} "
            f"test_loss={initial_test['loss']:.8e} "
            f"train_energy_mae={initial_train['energy_mae_h']:.8e} "
            f"test_energy_mae={initial_test['energy_mae_h']:.8e}"
        )
    history.append(
        {
            "step": 0,
            "train_loss": initial_train["loss"],
            "train_energy_mae_h": initial_train["energy_mae_h"],
            "train_loss_reevaluated": initial_train["loss"],
            "train_energy_mae_reevaluated_h": initial_train["energy_mae_h"],
            "test_loss": initial_test["loss"],
            "test_energy_mae_h": initial_test["energy_mae_h"],
            "grad_norm": float("nan"),
            "param_update_norm": float("nan"),
            "lr": float(args.learning_rate),
        }
    )
    flush_history()
    logger.log(
        "[train] "
        f"steps={int(args.steps)} train_step_mode=stream_single_molecule "
        f"batch_size={int(args.batch_size)} "
        f"jit_train={bool(args.jit_train)} jit_eval={bool(args.jit_eval)} "
        f"train={len(train_rows)} test={len(test_rows)} lr={float(args.learning_rate):.6g} "
        f"lr_decay_every={int(args.lr_decay_every)} lr_decay_factor={float(args.lr_decay_factor):.6g} "
        f"energy_normalization={args.energy_normalization}"
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
        for idx in batch_ids:
            loaded = load_reference_datum(train_rows[int(idx)], args=args, logger=logger)
            if loaded is None:
                continue
            point, datum = loaded
            loss_val, metrics, grads = loss_and_grad(state.params, datum)
            losses.append(float(loss_val))
            maes.append(_metric_scalar(metrics, "energy_mae"))
            grad_norms.append(_metric_scalar(metrics, "grad_norm"))
            grad_sum = _tree_add(grad_sum, grads)
            del loaded, point, datum
        if grad_sum is None:
            logger.log(f"[train] no valid references in batch at step {step}; skipped update")
            continue
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
        train_loss = float(np.mean(losses))
        train_mae = float(np.mean(maes))
        train_eval = {"loss": float("nan"), "energy_mae_h": float("nan")}
        test_eval = {"loss": float("nan"), "energy_mae_h": float("nan")}
        final_step = step == int(args.steps)
        if (step % int(args.eval_every) == 0 or final_step) and not (
            final_step and bool(args.skip_final_evaluation)
        ):
            train_eval = eval_data(train_rows, label=f"step={step} train")
            test_eval = eval_data(test_rows, label=f"step={step} test")
            if test_eval["loss"] < best_test_loss:
                best_test_loss = test_eval["loss"]
                best_step = step
                best_params = state.params
        row = {
            "step": step,
            "train_loss": train_loss,
            "train_energy_mae_h": train_mae,
            "train_loss_reevaluated": train_eval["loss"],
            "train_energy_mae_reevaluated_h": train_eval["energy_mae_h"],
            "test_loss": test_eval["loss"],
            "test_energy_mae_h": test_eval["energy_mae_h"],
            "grad_norm": float(np.mean(grad_norms)),
            "param_update_norm": float(_tree_l2_norm(param_delta)),
            "lr": float(lr_schedule(step - 1)),
        }
        history.append(row)
        flush_history()
        if step == 1 or step % int(args.log_every) == 0 or step == int(args.steps):
            logger.log(
                "[train] "
                f"step={step:5d}/{int(args.steps):5d} "
                f"train_loss={row['train_loss']:.8e} "
                f"train_energy_mae={row['train_energy_mae_h']:.8e} "
                f"train_loss_reevaluated={row['train_loss_reevaluated']:.8e} "
                f"test_loss={row['test_loss']:.8e} "
                f"grad_norm={row['grad_norm']:.8e} lr={row['lr']:.8e}"
            )
    elapsed = time.perf_counter() - t0
    if bool(args.skip_final_evaluation):
        final_train = {"loss": float("nan"), "energy_mae_h": float("nan")}
        final_test = {"loss": float("nan"), "energy_mae_h": float("nan")}
        if best_test_loss == float("inf"):
            best_params = state.params
            best_step = int(args.steps)
        logger.log("[train] skipped final eval")
    else:
        final_train = eval_data(train_rows, label="final_train")
        final_test = eval_data(test_rows, label="final_test")
    if final_test["loss"] < best_test_loss:
        best_test_loss = final_test["loss"]
        best_step = int(args.steps)
        best_params = state.params
    logger.log(
        "[train] done "
        f"final_train_loss={final_train['loss']:.8e} final_test_loss={final_test['loss']:.8e} "
        f"best_test_loss={best_test_loss:.8e}@{best_step} elapsed_s={elapsed:.2f}"
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
    }


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def write_predictions(
    path: Path,
    *,
    rows: list[XNDDimerRow],
    split: str,
    params: Any,
    eval_single: Any,
    args: argparse.Namespace,
    logger: RunLogger,
) -> list[dict[str, Any]]:
    pred_rows = []
    for row in rows:
        loaded = load_reference_datum(row, args=args, logger=logger)
        if loaded is None:
            continue
        point, datum = loaded
        _, metrics = eval_single(params, datum)
        predicted = _metric_scalar(metrics, "predicted_total_energies")
        target = float(point.row.target_energy_h)
        pred_rows.append(
            {
                "split": split,
                "row_index": int(point.row.row_index),
                "atom1": point.row.atom1,
                "atom2": point.row.atom2,
                "bond_distance_angstrom": point.row.bond_distance_angstrom,
                "multiplicity": int(point.row.multiplicity),
                "spin": int(point.row.spin),
                "target_energy_h": target,
                "predicted_energy_h": predicted,
                "energy_error_h": predicted - target,
                "energy_abs_err_ev": abs(predicted - target) * HARTREE_TO_EV,
            }
        )
        del loaded, point, datum
    if not pred_rows:
        return pred_rows
    mode = "a" if path.exists() else "w"
    with path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pred_rows[0].keys()))
        if mode == "w":
            writer.writeheader()
        writer.writerows(pred_rows)
    return pred_rows


def plot_outputs(outdir: Path, history: list[dict[str, Any]], pred_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.asarray([row["step"] for row in history], dtype=float)
    train_loss = np.asarray([row["train_loss_reevaluated"] for row in history], dtype=float)
    test_loss = np.asarray([row["test_loss"] for row in history], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(steps[np.isfinite(train_loss)], train_loss[np.isfinite(train_loss)], marker="o", lw=1.5, label="train eval")
    ax.plot(steps[np.isfinite(test_loss)], test_loss[np.isfinite(test_loss)], marker="s", lw=1.5, label="test eval")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "training_loss.png", dpi=220)
    plt.close(fig)

    if pred_rows:
        train = [row for row in pred_rows if row["split"] == "train"]
        test = [row for row in pred_rows if row["split"] == "test"]
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        for label, rows, marker in (("train", train, "o"), ("test", test, "s")):
            if not rows:
                continue
            target = np.asarray([row["target_energy_h"] for row in rows], dtype=float)
            pred = np.asarray([row["predicted_energy_h"] for row in rows], dtype=float)
            ax.scatter(target, pred, s=16, alpha=0.75, marker=marker, label=label)
        all_e = np.asarray(
            [row["target_energy_h"] for row in pred_rows] + [row["predicted_energy_h"] for row in pred_rows],
            dtype=float,
        )
        lo, hi = float(np.min(all_e)), float(np.max(all_e))
        ax.plot([lo, hi], [lo, hi], color="black", lw=1.0)
        ax.set_xlabel("target energy (Ha)")
        ax.set_ylabel("predicted energy (Ha)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "xnd_dimers_ground_energy_scatter.png", dpi=220)
        plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train self-consistent Neural XC on GradDFT XND dimers.")
    p.add_argument("--csv", default="datasets/graddft_public/processed/XND_dimers.csv")
    p.add_argument("--outdir", default="outputs/xnd_dimers_ground")
    p.add_argument("--reference-cache", default="outputs/xnd_dimers_ground/reference_cache.h5")
    p.add_argument("--rebuild-reference-cache", action="store_true")
    p.add_argument("--reference-only", action="store_true")
    p.add_argument("--fail-on-build-error", action="store_true")
    p.add_argument("--max-points", type=int, default=None)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--basis", default="def2-svp")
    p.add_argument("--xc", default="b3lyp")
    p.add_argument("--grids-level", type=int, default=2)
    p.add_argument("--max-l", type=int, default=3)
    p.add_argument("--integral-backend", choices=("cpu", "gpu", "jax", "libcint"), default="gpu")
    p.add_argument("--jk-backend", choices=("full", "df"), default="full")
    p.add_argument("--reference-builder", choices=("jax", "pyscf"), default="jax")
    p.add_argument("--df-tol", type=float, default=1e-10)
    p.add_argument("--df-max-rank", type=int, default=None)
    p.add_argument("--reference-scf-max-cycle", type=int, default=512)
    p.add_argument("--reference-scf-conv-tol", type=float, default=1e-10)
    p.add_argument("--reference-scf-conv-tol-density", type=float, default=1e-8)
    p.add_argument("--reference-scf-damping", type=float, default=0.25)
    p.add_argument("--reference-scf-level-shift", type=float, default=0.0)
    p.add_argument("--reference-scf-potential-clip", type=float, default=20.0)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--lr-decay-every", type=int, default=200)
    p.add_argument("--lr-decay-factor", type=float, default=0.5)
    p.add_argument("--energy-mse-weight", type=float, default=1.0)
    p.add_argument("--energy-mae-weight", type=float, default=1.0)
    p.add_argument("--energy-normalization", choices=("none", "per_electron", "per_atom"), default="per_electron")
    p.add_argument("--coefficient-prior-weight", type=float, default=0.0)
    p.add_argument("--training-mode", choices=("fixed_density", "self_consistent"), default="self_consistent")
    p.add_argument("--include-hfx-channel", action="store_true")
    p.add_argument("--include-pt2-channel", action="store_true")
    p.add_argument("--pt2-channel-mode", choices=("scaled_projected", "local_exact"), default="scaled_projected")
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
    p.add_argument("--skip-initial-eval", action="store_true")
    p.add_argument("--skip-final-evaluation", action="store_true")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--verbose", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(outdir / "run.log")
    rows = _read_xnd_dimers_csv(Path(args.csv))
    train_rows, test_rows = _split_rows(
        rows,
        seed=int(args.seed),
        test_fraction=float(args.test_fraction),
        max_points=args.max_points,
    )
    _write_split_csv(outdir / "split.csv", train_rows, test_rows)
    logger.log(
        "[setup] "
        f"csv={args.csv} rows={len(rows)} train={len(train_rows)} test={len(test_rows)} "
        f"basis={args.basis} grid={args.grids_level} integral={args.integral_backend} "
        f"reference={args.reference_builder} jk={args.jk_backend} "
        f"include_hfx_channel={bool(args.include_hfx_channel)}"
    )
    if bool(args.reference_only):
        train_references = build_reference_cache_only(train_rows, args=args, logger=logger)
        test_references = build_reference_cache_only(test_rows, args=args, logger=logger)
        if not train_references or not test_references:
            raise RuntimeError(f"Need non-empty train/test references; got {train_references}/{test_references}")
        logger.log(f"[setup] references cached train={train_references} test={test_references}")
        summary = {
            "csv": str(args.csv),
            "basis": str(args.basis),
            "xc": str(args.xc),
            "grids_level": int(args.grids_level),
            "integral_backend": str(args.integral_backend),
            "reference_builder": str(args.reference_builder),
            "jk_backend": str(args.jk_backend),
            "include_hfx_channel": bool(args.include_hfx_channel),
            "reference_only": True,
            "train_rows_requested": len(train_rows),
            "test_rows_requested": len(test_rows),
            "train_references": train_references,
            "test_references": test_references,
            "reference_cache": str(args.reference_cache),
        }
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.log(f"[summary] reference_only outdir={outdir}")
        return summary
    train_rows_ready = [row for row in train_rows if reference_exists(row, args=args)]
    test_rows_ready = [row for row in test_rows if reference_exists(row, args=args)]
    if not train_rows_ready or not test_rows_ready:
        raise RuntimeError(f"Need non-empty train/test references; got {len(train_rows_ready)}/{len(test_rows_ready)}")
    logger.log(f"[setup] reference rows ready train={len(train_rows_ready)} test={len(test_rows_ready)}")
    result = train(train_rows_ready, test_rows_ready, args=args, logger=logger)
    write_history(outdir / "training_history.csv", result["history"])
    checkpoint_path = outdir / "neural_xc_params.msgpack"
    save_params_checkpoint(checkpoint_path, result["best_params"])
    pred_path = outdir / "xnd_dimers_ground_predictions.csv"
    if pred_path.exists():
        pred_path.unlink()
    pred_rows = []
    if bool(args.skip_final_evaluation):
        logger.log("[summary] skipped final predictions")
    else:
        pred_rows.extend(
            write_predictions(
                pred_path,
                rows=train_rows_ready,
                split="train",
                params=result["best_params"],
                eval_single=result["eval_single"],
                args=args,
                logger=logger,
            )
        )
        pred_rows.extend(
            write_predictions(
                pred_path,
                rows=test_rows_ready,
                split="test",
                params=result["best_params"],
                eval_single=result["eval_single"],
                args=args,
                logger=logger,
            )
        )
    plot_outputs(outdir, result["history"], pred_rows)
    summary = {
        "csv": str(args.csv),
        "basis": str(args.basis),
        "xc": str(args.xc),
        "grids_level": int(args.grids_level),
        "integral_backend": str(args.integral_backend),
        "reference_builder": str(args.reference_builder),
        "jk_backend": str(args.jk_backend),
        "df_tol": float(args.df_tol),
        "df_max_rank": args.df_max_rank,
        "input_feature_mode": str(args.input_feature_mode),
        "training_mode": str(args.training_mode),
        "include_hfx_channel": bool(args.include_hfx_channel),
        "include_pt2_channel": bool(args.include_pt2_channel),
        "pt2_channel_mode": str(args.pt2_channel_mode),
        "skip_initial_eval": bool(args.skip_initial_eval),
        "skip_final_evaluation": bool(args.skip_final_evaluation),
        "energy_normalization": str(args.energy_normalization),
        "seed": int(args.seed),
        "test_fraction": float(args.test_fraction),
        "train_rows_requested": len(train_rows),
        "test_rows_requested": len(test_rows),
        "train_references": len(train_rows_ready),
        "test_references": len(test_rows_ready),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "lr_decay_every": int(args.lr_decay_every),
        "lr_decay_factor": float(args.lr_decay_factor),
        "final_train_loss": result["final_train"]["loss"],
        "final_test_loss": result["final_test"]["loss"],
        "final_train_energy_mae_ev": result["final_train"]["energy_mae_h"] * HARTREE_TO_EV,
        "final_test_energy_mae_ev": result["final_test"]["energy_mae_h"] * HARTREE_TO_EV,
        "best_test_loss": result["best_test_loss"],
        "best_step": int(result["best_step"]),
        "elapsed_s": float(result["elapsed_s"]),
        "history_csv": str(outdir / "training_history.csv"),
        "prediction_csv": (None if bool(args.skip_final_evaluation) else str(pred_path)),
        "training_curve_png": str(outdir / "training_loss.png"),
        "prediction_scatter_png": (
            None
            if bool(args.skip_final_evaluation)
            else str(outdir / "xnd_dimers_ground_energy_scatter.png")
        ),
        "checkpoint": str(checkpoint_path),
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
