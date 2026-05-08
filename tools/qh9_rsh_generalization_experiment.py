from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from td_graddft.device import put_restricted_reference_on_device
from td_graddft.nn_rsh import (
    AtomCenteredDensityDescriptorConfig,
    ResolvedRSHParameters,
    make_atom_centered_density_rsh_functional,
    make_gnn_rsh_functional,
    make_self_supervised_rsh_loss,
)
from td_graddft.reference_legacy import restricted_reference_from_pyscf
from td_graddft.training import GroundStateDatum, GroundStateTrainingConfig
from td_graddft.training.targets import (
    _freeze_functional_for_fractional_path,
    _perturb_restricted_frontier_occupations,
    _resolve_training_molecule_and_info_with_mode,
)

from scan_water_fractional_occupation import _piecewise_linear_energy, _scan_one_point


ATOMIC_SYMBOL = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
}


@dataclass(frozen=True)
class QH9Entry:
    db_id: int
    formula: str
    natoms: int
    electron_count: int
    z: np.ndarray
    pos_ang: np.ndarray
    molecule: Any

    def to_datum(self) -> GroundStateDatum:
        return GroundStateDatum(
            molecule=self.molecule,
            target_total_energy=jnp.asarray(self.molecule.mf_energy),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "QH9 nn-RSH generalization experiment: sample 30 molecules, train on 24 "
            "with Koopmans IP/LUMO-EA, and evaluate fractional-charge improvement on 6."
        ),
    )
    parser.add_argument("--db-path", default="/home/yjiao/QH9Stable.db")
    parser.add_argument("--outdir", default="outputs/qh9_rsh_generalization_30_train24_test6_ep200")
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--train-size", type=int, default=24)
    parser.add_argument("--max-atoms", type=int, default=8)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=0,
        help=(
            "Number of training molecules per optimizer step. 0 means full "
            "batch. For strict full_scf training, use 1-2 to keep each step "
            "self-consistent but computationally tractable."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate-scale", type=float, default=0.2)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--line-search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--line-search-shrink", type=float, default=0.5)
    parser.add_argument("--line-search-attempts", type=int, default=4)
    parser.add_argument("--accept-tolerance", type=float, default=1e-10)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument(
        "--head",
        choices=("gnn", "atomwise"),
        default="gnn",
        help="Neural head used to predict molecule-specific RSH parameters.",
    )
    parser.add_argument("--omega-grid", default="0.0,0.3,0.6")
    parser.add_argument("--radial-centers", default="0.6,1.4,2.6")
    parser.add_argument("--radial-width", type=float, default=0.5)
    parser.add_argument("--max-angular", type=int, default=1)
    parser.add_argument("--atom-hidden-dims", default="16,16")
    parser.add_argument("--pooled-hidden-dims", default="16")
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--gnn-node-hidden-dims", default="16,16")
    parser.add_argument("--gnn-global-hidden-dims", default="16")
    parser.add_argument("--gnn-num-heads", type=int, default=4)
    parser.add_argument("--gnn-qkv-features", type=int, default=16)
    parser.add_argument("--gnn-ffn-dim", type=int, default=64)
    parser.add_argument("--gnn-num-layers", type=int, default=1)
    parser.add_argument("--scf-max-cycle", type=int, default=4)
    parser.add_argument(
        "--loss-impl",
        choices=("fixed_density", "full_scf"),
        default="full_scf",
        help=(
            "full_scf is the physically consistent default: each Koopmans "
            "endpoint is evaluated through the differentiable self-consistent "
            "SCF path. fixed_density is only a fast debug/negative-control "
            "proxy and must not be used for final RSH tuning conclusions."
        ),
    )
    parser.add_argument(
        "--scf-gradient-mode",
        choices=("unrolled", "implicit_commutator"),
        default="implicit_commutator",
    )
    parser.add_argument("--fractional-points", type=int, default=21)
    parser.add_argument("--fractional-scf-max-cycle", type=int, default=12)
    parser.add_argument("--fractional-scf-damping", type=float, default=0.35)
    parser.add_argument("--fractional-scf-level-shift", type=float, default=0.5)
    parser.add_argument(
        "--koopmans-detach-charged-states",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Detach charged N-1/N+1 branches from the gradient. Defaults to "
            "False for strict self-consistent Koopmans tuning; enable only as "
            "a cheaper envelope/debug approximation."
        ),
    )
    parser.add_argument(
        "--koopmans-differentiate-charged-orbitals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Differentiate through the charged-state SCF orbital response. "
            "Defaults to True for strict full-gradient training."
        ),
    )
    parser.add_argument("--prior-weight", type=float, default=0.0)
    parser.add_argument(
        "--reset-output-head-to-default",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Initialize all molecules at the same generic RSH default "
            "(sr=0.20, lr=0.65, omega=0.30) before neural training."
        ),
    )
    return parser.parse_args()


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _formula_from_z(z: np.ndarray) -> str:
    counts: dict[str, int] = {}
    for zi in z:
        sym = ATOMIC_SYMBOL[int(zi)]
        counts[sym] = counts.get(sym, 0) + 1

    ordered: list[tuple[str, int]] = []
    if "C" in counts:
        ordered.append(("C", counts.pop("C")))
    if "H" in counts:
        ordered.append(("H", counts.pop("H")))
    for sym in sorted(counts):
        ordered.append((sym, counts[sym]))
    return "".join(sym if n == 1 else f"{sym}{n}" for sym, n in ordered)


def _build_atom_block(z: np.ndarray, pos_ang: np.ndarray) -> str:
    lines = []
    for zi, xyz in zip(z, pos_ang, strict=True):
        lines.append(f"{ATOMIC_SYMBOL[int(zi)]} {xyz[0]: .12f} {xyz[1]: .12f} {xyz[2]: .12f}")
    return "\n".join(lines)


def _candidate_ids(conn: sqlite3.Connection, *, max_atoms: int) -> list[int]:
    ids: list[int] = []
    for db_id, z_blob in conn.execute("SELECT id, Z FROM data WHERE N <= ?", (int(max_atoms),)):
        z = np.frombuffer(z_blob, dtype=np.int32)
        if int(np.sum(z)) % 2 == 0:
            ids.append(int(db_id))
    return ids


def _fetch_molecule(conn: sqlite3.Connection, db_id: int) -> tuple[np.ndarray, np.ndarray]:
    row = conn.execute("SELECT N, Z, pos FROM data WHERE id = ?", (int(db_id),)).fetchone()
    if row is None:
        raise KeyError(f"QH9 id {db_id} not found")
    n, z_blob, pos_blob = row
    z = np.frombuffer(z_blob, dtype=np.int32).copy()
    pos = np.frombuffer(pos_blob, dtype=np.float64).reshape(int(n), 3).copy()
    return z, pos


def _build_entry(
    db_id: int,
    z: np.ndarray,
    pos_ang: np.ndarray,
    *,
    basis: str,
    xc: str,
    grid_level: int,
    omega_grid: tuple[float, ...],
) -> QH9Entry:
    from pyscf import dft, gto

    mol = gto.M(
        atom=_build_atom_block(z, pos_ang),
        basis=basis,
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = int(grid_level)
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"RKS did not converge for QH9 id={db_id}")

    reference = restricted_reference_from_pyscf(
        mf,
        compute_local_hfx_features=True,
        compute_local_hfx_aux=True,
        hfx_omega_values=omega_grid,
    )
    reference = put_restricted_reference_on_device(reference)
    return QH9Entry(
        db_id=int(db_id),
        formula=_formula_from_z(z),
        natoms=int(len(z)),
        electron_count=int(np.sum(z)),
        z=z,
        pos_ang=pos_ang,
        molecule=reference,
    )


def _sample_entries(args: argparse.Namespace, omega_grid: tuple[float, ...]) -> list[QH9Entry]:
    db_path = Path(args.db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"QH9 db not found: {db_path}")

    rng = np.random.default_rng(int(args.seed))
    conn = sqlite3.connect(db_path)
    try:
        ids = _candidate_ids(conn, max_atoms=int(args.max_atoms))
        if len(ids) < int(args.sample_size):
            raise RuntimeError(
                f"Only {len(ids)} closed-shell candidates with N <= {args.max_atoms}; "
                f"need {args.sample_size}."
            )
        rng.shuffle(ids)
        selected: list[QH9Entry] = []
        failures: list[dict[str, str | int]] = []
        for db_id in ids:
            try:
                z, pos = _fetch_molecule(conn, db_id)
                entry = _build_entry(
                    db_id,
                    z,
                    pos,
                    basis=str(args.basis),
                    xc=str(args.xc),
                    grid_level=int(args.grid_level),
                    omega_grid=omega_grid,
                )
            except Exception as exc:
                failures.append({"db_id": int(db_id), "error": repr(exc)})
                continue
            selected.append(entry)
            print(
                f"selected {len(selected):02d}/{args.sample_size}: "
                f"id={entry.db_id} formula={entry.formula} "
                f"natoms={entry.natoms} nelec={entry.electron_count}",
                flush=True,
            )
            if len(selected) >= int(args.sample_size):
                break
    finally:
        conn.close()

    if len(selected) < int(args.sample_size):
        raise RuntimeError(
            f"Could only build {len(selected)} converged QH9 entries; "
            f"last failures={failures[-5:]}"
        )
    return selected


def _make_functional(args: argparse.Namespace):
    descriptor_config = AtomCenteredDensityDescriptorConfig(
        radial_centers=_parse_float_tuple(str(args.radial_centers)),
        radial_width=float(args.radial_width),
        max_angular=int(args.max_angular),
    )
    omega_grid = _parse_float_tuple(str(args.omega_grid))
    if str(args.head) == "gnn":
        return make_gnn_rsh_functional(
            local_xc_spec=str(args.xc),
            descriptor_config=descriptor_config,
            node_hidden_dims=_parse_int_tuple(str(args.gnn_node_hidden_dims)),
            global_hidden_dims=_parse_int_tuple(str(args.gnn_global_hidden_dims)),
            num_heads=int(args.gnn_num_heads),
            num_layers=int(args.gnn_num_layers),
            qkv_features=int(args.gnn_qkv_features),
            ffn_dim=int(args.gnn_ffn_dim),
            fallback_omega_values=omega_grid,
        )
    return make_atom_centered_density_rsh_functional(
        local_xc_spec=str(args.xc),
        descriptor_config=descriptor_config,
        atom_hidden_dims=_parse_int_tuple(str(args.atom_hidden_dims)),
        pooled_hidden_dims=_parse_int_tuple(str(args.pooled_hidden_dims)),
        embedding_dim=int(args.embedding_dim),
        fallback_omega_values=omega_grid,
    )


def _scalar(value: Any) -> float:
    arr = jnp.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    return float(arr.reshape(-1)[0])


def _mean_metrics(metric_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    keys = set().union(*(row.keys() for row in metric_rows))
    out: dict[str, Any] = {}
    for key in keys:
        values = [jnp.asarray(row[key]) for row in metric_rows if key in row]
        if values:
            out[key] = sum(values) / len(values)
    return out


def _average_loss_fn(single_loss_fn):
    def loss(params: Any, functional: Any, data: Sequence[GroundStateDatum]):
        losses = []
        metric_rows = []
        for datum in data:
            value, metrics = single_loss_fn(params, functional, datum)
            losses.append(jnp.asarray(value))
            metric_rows.append(metrics)
        mean_loss = sum(losses) / len(losses)
        metrics_out = _mean_metrics(metric_rows)
        metrics_out["loss"] = mean_loss
        return mean_loss, metrics_out

    return loss


def _frontier_energies(molecule: Any, *, occupation_tolerance: float = 1e-8) -> tuple[Any, Any]:
    mo_occ = jnp.asarray(molecule.mo_occ)
    mo_energy = jnp.asarray(molecule.mo_energy)
    restricted_occ = mo_occ if mo_occ.ndim == 1 else mo_occ[0]
    restricted_energy = mo_energy if mo_energy.ndim == 1 else mo_energy[0]
    occ_mask = restricted_occ > float(occupation_tolerance)
    vir_mask = restricted_occ <= float(occupation_tolerance)
    nmo = int(restricted_occ.shape[0])
    homo_idx = jnp.max(jnp.where(occ_mask, jnp.arange(nmo), -1))
    lumo_idx = jnp.min(jnp.where(vir_mask, jnp.arange(nmo), nmo))
    return restricted_energy[homo_idx], restricted_energy[lumo_idx]


def _make_fixed_density_koopmans_loss(
    functional: Any,
    *,
    occupation_tolerance: float = 1e-8,
    prior_weight: float = 0.0,
):
    """Fast endpoint Koopmans proxy for debugging only.

    The neural model predicts RSH parameters from the neutral PBE reference.
    Those parameters are then frozen while N, N-1 and N+1 endpoint energies are
    evaluated on fixed frontier occupations. This is not a variational RSH
    tuning objective because the density/orbitals do not respond to the
    functional. Use it only as a negative control or wiring test.
    """

    def loss(params: Any, active_functional: Any, datum: GroundStateDatum):
        del active_functional
        molecule = datum.molecule
        bound = functional.bind_to_molecule(params, molecule)
        neutral_energy = bound.energy_from_molecule(molecule)
        cation = _perturb_restricted_frontier_occupations(
            molecule,
            homo_delta=-1.0,
            occupation_tolerance=float(occupation_tolerance),
        )
        anion = _perturb_restricted_frontier_occupations(
            molecule,
            lumo_delta=1.0,
            occupation_tolerance=float(occupation_tolerance),
        )
        cation_energy = bound.energy_from_molecule(cation)
        anion_energy = bound.energy_from_molecule(anion)
        neutral_homo, neutral_lumo = _frontier_energies(
            molecule,
            occupation_tolerance=float(occupation_tolerance),
        )
        ip_residual = neutral_homo + cation_energy - neutral_energy
        lumo_ea_residual = neutral_lumo + neutral_energy - anion_energy
        total = ip_residual**2 + lumo_ea_residual**2
        resolved = functional.resolve_parameters(params, molecule)
        prior = (
            (resolved.sr_hf_fraction - 0.20) ** 2
            + (resolved.lr_hf_fraction - 0.65) ** 2
            + (resolved.omega - 0.30) ** 2
        )
        total = total + float(prior_weight) * prior
        gap_residual = (neutral_lumo - neutral_homo) - (
            cation_energy + anion_energy - 2.0 * neutral_energy
        )
        zeros = jnp.asarray([0.0], dtype=jnp.asarray(total).dtype)
        return total, {
            "loss": total,
            "koopmans_ip_mae": jnp.asarray([jnp.abs(ip_residual)], dtype=jnp.asarray(total).dtype),
            "koopmans_lumo_ea_mae": jnp.asarray(
                [jnp.abs(lumo_ea_residual)],
                dtype=jnp.asarray(total).dtype,
            ),
            "koopmans_gap_mae": jnp.asarray([jnp.abs(gap_residual)], dtype=jnp.asarray(total).dtype),
            "koopmans_ip_residual": jnp.asarray([ip_residual], dtype=jnp.asarray(total).dtype),
            "koopmans_lumo_ea_residual": jnp.asarray(
                [lumo_ea_residual],
                dtype=jnp.asarray(total).dtype,
            ),
            "koopmans_ea_mae": zeros,
            "sr_hf_fraction": jnp.asarray(
                [resolved.sr_hf_fraction],
                dtype=jnp.asarray(total).dtype,
            ),
            "lr_hf_fraction": jnp.asarray(
                [resolved.lr_hf_fraction],
                dtype=jnp.asarray(total).dtype,
            ),
            "omega": jnp.asarray([resolved.omega], dtype=jnp.asarray(total).dtype),
        }

    return loss


def _loss_metrics_for_dataset(
    params: Any,
    functional: Any,
    data: Sequence[GroundStateDatum],
    loss_fn,
) -> dict[str, float]:
    value, metrics = loss_fn(params, functional, data)
    jax.block_until_ready(value)
    return {
        "loss": float(value),
        "koopmans_ip_mae": _scalar(metrics["koopmans_ip_mae"]),
        "koopmans_lumo_ea_mae": _scalar(metrics["koopmans_lumo_ea_mae"]),
        "koopmans_gap_mae": _scalar(metrics["koopmans_gap_mae"]),
        "sr_hf_fraction": _scalar(metrics["sr_hf_fraction"]),
        "lr_hf_fraction": _scalar(metrics["lr_hf_fraction"]),
        "omega": _scalar(metrics["omega"]),
    }


def _tree_l2_norm(tree: Any) -> float:
    total = 0.0
    for leaf in jax.tree_util.tree_leaves(tree):
        arr = np.asarray(jax.device_get(leaf), dtype=float)
        total += float(np.sum(arr * arr))
    return float(np.sqrt(total))


def _scaled_updates(updates: Any, scale: float) -> Any:
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x) * float(scale), updates)


def _train(
    *,
    args: argparse.Namespace,
    functional: Any,
    initial_params: Any,
    train_data: Sequence[GroundStateDatum],
    test_data: Sequence[GroundStateDatum],
    loss_fn,
) -> tuple[Any, list[dict[str, Any]]]:
    schedule = optax.cosine_decay_schedule(
        init_value=float(args.learning_rate),
        decay_steps=max(int(args.steps), 1),
        alpha=float(args.min_learning_rate_scale),
    )
    tx = optax.chain(
        optax.clip_by_global_norm(float(args.gradient_clip_norm)),
        optax.adam(schedule),
    )
    state = TrainState.create(
        apply_fn=functional.model.apply,
        params=initial_params,
        tx=tx,
    )

    train_items = list(train_data)
    n_train = len(train_items)
    batch_size = int(args.train_batch_size)
    use_minibatch = batch_size > 0 and batch_size < n_train
    rng = np.random.default_rng(int(args.seed) + 7919)

    def step_batch() -> tuple[Sequence[GroundStateDatum], list[int] | None]:
        if not use_minibatch:
            return train_items, None
        indices = sorted(
            int(i)
            for i in rng.choice(n_train, size=batch_size, replace=False).tolist()
        )
        return [train_items[i] for i in indices], indices

    def value_and_grad(params, batch_data: Sequence[GroundStateDatum]):
        return jax.value_and_grad(
            lambda p: loss_fn(p, functional, batch_data),
            has_aux=True,
        )(params)

    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    best_params = state.params
    best_loss = float("inf")
    for step in range(0, int(args.steps) + 1):
        do_eval = step == 0 or step == int(args.steps) or step % max(1, int(args.eval_every)) == 0
        if do_eval:
            train_metrics = _loss_metrics_for_dataset(state.params, functional, train_data, loss_fn)
            test_metrics = _loss_metrics_for_dataset(state.params, functional, test_data, loss_fn)
            row = {
                "step": step,
                "elapsed_s": float(time.perf_counter() - start),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
                "accepted": None,
                "accepted_scale": None,
                "grad_norm": None,
            }
            history.append(row)
            print(
                f"eval step={step:03d} "
                f"train_loss={train_metrics['loss']:.6e} "
                f"test_loss={test_metrics['loss']:.6e} "
                f"train_kip={train_metrics['koopmans_ip_mae']:.3e} "
                f"test_kip={test_metrics['koopmans_ip_mae']:.3e} "
                f"test_klumo={test_metrics['koopmans_lumo_ea_mae']:.3e}",
                flush=True,
            )
            if train_metrics["loss"] < best_loss:
                best_loss = train_metrics["loss"]
                best_params = state.params
        if step == int(args.steps):
            break

        batch_data, batch_indices = step_batch()
        (baseline_loss, baseline_metrics), grads = value_and_grad(state.params, batch_data)
        baseline_loss_value = float(baseline_loss)
        clean_grads = jax.tree_util.tree_map(
            lambda x: jnp.nan_to_num(jnp.asarray(x), nan=0.0, posinf=0.0, neginf=0.0),
            grads,
        )
        updates, next_opt_state = tx.update(clean_grads, state.opt_state, state.params)

        accepted = False
        accepted_scale = 0.0
        next_params = state.params
        next_loss_value = baseline_loss_value
        if not bool(args.line_search):
            accepted = True
            accepted_scale = 1.0
            next_params = optax.apply_updates(state.params, updates)
        else:
            scales = [
                float(args.line_search_shrink) ** i
                for i in range(max(1, int(args.line_search_attempts)))
            ]
            for scale in scales:
                candidate_params = optax.apply_updates(state.params, _scaled_updates(updates, scale))
                candidate_loss, _candidate_metrics = loss_fn(candidate_params, functional, batch_data)
                candidate_loss_value = float(candidate_loss)
                if np.isfinite(candidate_loss_value) and (
                    candidate_loss_value <= baseline_loss_value + float(args.accept_tolerance)
                ):
                    accepted = True
                    accepted_scale = float(scale)
                    next_params = candidate_params
                    next_loss_value = candidate_loss_value
                    break

        if accepted:
            state = state.replace(params=next_params, opt_state=next_opt_state)
        if step % max(1, int(args.log_every)) == 0 or not accepted:
            print(
                f"train step={step + 1:03d} "
                f"baseline={baseline_loss_value:.6e} next={next_loss_value:.6e} "
                f"accepted={accepted} scale={accepted_scale:.3g} "
                f"batch={batch_indices if batch_indices is not None else 'full'} "
                f"grad={_tree_l2_norm(clean_grads):.3e} "
                f"sr={_scalar(baseline_metrics['sr_hf_fraction']):.4f} "
                f"lr={_scalar(baseline_metrics['lr_hf_fraction']):.4f} "
                f"omega={_scalar(baseline_metrics['omega']):.4f}",
                flush=True,
            )

    return best_params, history


def _metric_row_for_entry(
    *,
    split: str,
    stage: str,
    entry: QH9Entry,
    params: Any,
    functional: Any,
    datum: GroundStateDatum,
    single_loss_fn,
) -> dict[str, Any]:
    loss, metrics = single_loss_fn(params, functional, datum)
    resolved = functional.resolve_parameters(params, entry.molecule)
    return {
        "split": split,
        "stage": stage,
        "db_id": entry.db_id,
        "formula": entry.formula,
        "natoms": entry.natoms,
        "electron_count": entry.electron_count,
        "loss": float(loss),
        "koopmans_ip_mae": _scalar(metrics["koopmans_ip_mae"]),
        "koopmans_lumo_ea_mae": _scalar(metrics["koopmans_lumo_ea_mae"]),
        "koopmans_gap_mae": _scalar(metrics["koopmans_gap_mae"]),
        "sr_hf_fraction": float(resolved.sr_hf_fraction),
        "lr_hf_fraction": float(resolved.lr_hf_fraction),
        "paper_beta": float(resolved.lr_hf_fraction - resolved.sr_hf_fraction),
        "omega": float(resolved.omega),
    }


def _fractional_scan_entry(
    *,
    stage: str,
    entry: QH9Entry,
    params: Any,
    functional: Any,
    training_config: GroundStateTrainingConfig,
    q_values: Sequence[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base_molecule, base_info = _resolve_training_molecule_and_info_with_mode(
        params,
        functional,
        entry.molecule,
        training_config,
    )
    frozen_functional, frozen_params = _freeze_functional_for_fractional_path(
        params,
        functional,
        base_molecule,
    )
    resolved = functional.resolve_parameters(params, entry.molecule)
    energies: dict[float, float] = {}
    rms_values: dict[float, float | None] = {}
    for q in q_values:
        energy, rms = _scan_one_point(
            frozen_params,
            frozen_functional,
            base_molecule,
            q=float(q),
            training_config=training_config,
        )
        energies[float(q)] = energy
        rms_values[float(q)] = rms
    q_minus = min(q_values, key=lambda value: abs(float(value) + 1.0))
    q_zero = min(q_values, key=lambda value: abs(float(value)))
    q_plus = min(q_values, key=lambda value: abs(float(value) - 1.0))
    e_minus = energies[float(q_minus)]
    e_zero = energies[float(q_zero)]
    e_plus = energies[float(q_plus)]

    rows: list[dict[str, Any]] = []
    for q in q_values:
        q_float = float(q)
        energy = energies[q_float]
        linear = _piecewise_linear_energy(q_float, e_minus, e_zero, e_plus)
        rows.append(
            {
                "stage": stage,
                "db_id": entry.db_id,
                "formula": entry.formula,
                "q": q_float,
                "energy_hartree": energy,
                "relative_energy_hartree": energy - e_zero,
                "linear_energy_hartree": linear,
                "linear_relative_energy_hartree": linear - e_zero,
                "linearity_deviation_hartree": energy - linear,
                "selected_rms_density": rms_values[q_float],
            }
        )
    max_abs_dev = max(abs(float(row["linearity_deviation_hartree"])) for row in rows)
    summary = {
        "stage": stage,
        "db_id": entry.db_id,
        "formula": entry.formula,
        "natoms": entry.natoms,
        "electron_count": entry.electron_count,
        "sr_hf_fraction": float(resolved.sr_hf_fraction),
        "lr_hf_fraction": float(resolved.lr_hf_fraction),
        "paper_beta": float(resolved.lr_hf_fraction - resolved.sr_hf_fraction),
        "omega": float(resolved.omega),
        "e_nminus1_hartree": e_minus,
        "e_n_hartree": e_zero,
        "e_nplus1_hartree": e_plus,
        "max_abs_linearity_deviation_hartree": max_abs_dev,
        "max_abs_linearity_deviation_mhartree": 1000.0 * max_abs_dev,
        "base_selected_rms_density": (
            float(jnp.asarray(getattr(base_info, "selected_rms_density", 0.0)))
            if getattr(base_info, "mode", None) == "self_consistent"
            else None
        ),
    }
    return summary, rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_fractional_improvement(path: Path, rows: Sequence[dict[str, Any]]) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    labels = [f"{row['db_id']} {row['formula']}" for row in rows]
    before = [float(row["initial_max_abs_dev_mhartree"]) for row in rows]
    after = [float(row["final_max_abs_dev_mhartree"]) for row in rows]
    x = np.arange(len(labels))

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.linewidth": 1.2,
        }
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    width = 0.38
    ax.bar(x - width / 2, before, width=width, label="initial", color="#4c78a8")
    ax.bar(x + width / 2, after, width=width, label="trained", color="#f58518")
    ax.set_title("QH9 test molecules: fractional-linearity deviation")
    ax.set_ylabel("max |deviation| (mHa)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_training_history(path: Path, history: Sequence[dict[str, Any]]) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None
    if not history:
        return None
    steps = [int(row["step"]) for row in history]
    fig, (ax_loss, ax_err) = plt.subplots(2, 1, figsize=(8.4, 6.6), sharex=True)
    ax_loss.plot(steps, [float(row["train_loss"]) for row in history], "-o", label="train")
    ax_loss.plot(steps, [float(row["test_loss"]) for row in history], "-o", label="test")
    ax_loss.set_ylabel("Koopmans loss")
    ax_loss.set_yscale("log")
    ax_loss.grid(alpha=0.25, linewidth=0.8)
    ax_loss.legend(frameon=False)
    ax_err.plot(steps, [float(row["test_koopmans_ip_mae"]) for row in history], "-o", label="test IP")
    ax_err.plot(
        steps,
        [float(row["test_koopmans_lumo_ea_mae"]) for row in history],
        "-o",
        label="test LUMO-EA",
    )
    ax_err.set_xlabel("epoch")
    ax_err.set_ylabel("MAE (Ha)")
    ax_err.grid(alpha=0.25, linewidth=0.8)
    ax_err.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    omega_grid = _parse_float_tuple(str(args.omega_grid))
    entries = _sample_entries(args, omega_grid)
    if int(args.train_size) <= 0 or int(args.train_size) >= len(entries):
        raise ValueError("--train-size must be between 1 and sample-size - 1.")
    train_entries = entries[: int(args.train_size)]
    test_entries = entries[int(args.train_size) :]
    train_data = [entry.to_datum() for entry in train_entries]
    test_data = [entry.to_datum() for entry in test_entries]

    functional = _make_functional(args)
    params = functional.init_from_molecule(jax.random.PRNGKey(int(args.seed)), train_entries[0].molecule)
    if bool(args.reset_output_head_to_default):
        params = functional.params_with_resolved(
            params,
            ResolvedRSHParameters(
                sr_hf_fraction=0.20,
                lr_hf_fraction=0.65,
                omega=0.30,
            ),
            molecule=train_entries[0].molecule,
            preserve_network=False,
        )
    initial_params = params

    training_config = GroundStateTrainingConfig(
        mode="self_consistent",
        scf_gradient_mode=str(args.scf_gradient_mode),
        scf_max_cycle=int(args.scf_max_cycle),
        scf_require_convergence=False,
        fractional_branch_scf_max_cycle=int(args.fractional_scf_max_cycle),
        fractional_branch_scf_damping=float(args.fractional_scf_damping),
        fractional_branch_scf_level_shift=float(args.fractional_scf_level_shift),
        fractional_branch_scf_iterate_selection="best_rms",
    )
    single_loss_fn = make_self_supervised_rsh_loss(
        functional,
        training_config=training_config,
        janak_weight=0.0,
        fractional_weight=0.0,
        koopmans_ip_weight=1.0,
        koopmans_ea_weight=0.0,
        koopmans_lumo_ea_weight=1.0,
        koopmans_loss_kind="squared",
        koopmans_detach_charged_states=bool(args.koopmans_detach_charged_states),
        koopmans_differentiate_charged_orbitals=bool(args.koopmans_differentiate_charged_orbitals),
        prior_weight=float(args.prior_weight),
    )
    if str(args.loss_impl) == "fixed_density":
        print(
            "WARNING: --loss-impl fixed_density is a debug/negative-control "
            "proxy. It is not a self-consistent RSH tuning objective.",
            flush=True,
        )
        single_loss_fn = _make_fixed_density_koopmans_loss(
            functional,
            occupation_tolerance=training_config.occupation_tolerance,
            prior_weight=float(args.prior_weight),
        )
    avg_loss_fn = _average_loss_fn(single_loss_fn)

    split = {
        "seed": int(args.seed),
        "sample_size": int(args.sample_size),
        "train_size": int(args.train_size),
        "test_size": len(test_entries),
        "max_atoms": int(args.max_atoms),
        "basis": str(args.basis),
        "xc": str(args.xc),
        "head": str(args.head),
        "loss_impl": str(args.loss_impl),
        "train_batch_size": int(args.train_batch_size),
        "scf_gradient_mode": str(args.scf_gradient_mode),
        "scf_max_cycle": int(args.scf_max_cycle),
        "train": [
            {
                "db_id": entry.db_id,
                "formula": entry.formula,
                "natoms": entry.natoms,
                "electron_count": entry.electron_count,
            }
            for entry in train_entries
        ],
        "test": [
            {
                "db_id": entry.db_id,
                "formula": entry.formula,
                "natoms": entry.natoms,
                "electron_count": entry.electron_count,
            }
            for entry in test_entries
        ],
    }
    (outdir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    final_params, history = _train(
        args=args,
        functional=functional,
        initial_params=initial_params,
        train_data=train_data,
        test_data=test_data,
        loss_fn=avg_loss_fn,
    )
    _write_csv(outdir / "history.csv", history)
    _plot_training_history(outdir / "training_history.png", history)

    metric_rows = []
    for split_name, split_entries, split_data in (
        ("train", train_entries, train_data),
        ("test", test_entries, test_data),
    ):
        for stage, stage_params in (("initial", initial_params), ("final", final_params)):
            for entry, datum in zip(split_entries, split_data, strict=True):
                metric_rows.append(
                    _metric_row_for_entry(
                        split=split_name,
                        stage=stage,
                        entry=entry,
                        params=stage_params,
                        functional=functional,
                        datum=datum,
                        single_loss_fn=single_loss_fn,
                    )
                )
    _write_csv(outdir / "per_molecule_koopmans_metrics.csv", metric_rows)

    q_values = [float(x) for x in jnp.linspace(-1.0, 1.0, max(3, int(args.fractional_points)))]
    fractional_rows = []
    fractional_summaries = []
    for entry in test_entries:
        initial_summary, initial_rows = _fractional_scan_entry(
            stage="initial",
            entry=entry,
            params=initial_params,
            functional=functional,
            training_config=training_config,
            q_values=q_values,
        )
        final_summary, final_rows = _fractional_scan_entry(
            stage="final",
            entry=entry,
            params=final_params,
            functional=functional,
            training_config=training_config,
            q_values=q_values,
        )
        fractional_rows.extend(initial_rows)
        fractional_rows.extend(final_rows)
        improvement = (
            float(initial_summary["max_abs_linearity_deviation_mhartree"])
            - float(final_summary["max_abs_linearity_deviation_mhartree"])
        )
        fractional_summaries.append(
            {
                "db_id": entry.db_id,
                "formula": entry.formula,
                "natoms": entry.natoms,
                "electron_count": entry.electron_count,
                "initial_max_abs_dev_mhartree": initial_summary[
                    "max_abs_linearity_deviation_mhartree"
                ],
                "final_max_abs_dev_mhartree": final_summary[
                    "max_abs_linearity_deviation_mhartree"
                ],
                "improvement_mhartree": improvement,
                "improved": improvement > 0.0,
                "initial_sr_hf_fraction": initial_summary["sr_hf_fraction"],
                "initial_lr_hf_fraction": initial_summary["lr_hf_fraction"],
                "initial_omega": initial_summary["omega"],
                "final_sr_hf_fraction": final_summary["sr_hf_fraction"],
                "final_lr_hf_fraction": final_summary["lr_hf_fraction"],
                "final_omega": final_summary["omega"],
            }
        )
        print(
            f"fractional test id={entry.db_id} {entry.formula}: "
            f"dev {initial_summary['max_abs_linearity_deviation_mhartree']:.3f} -> "
            f"{final_summary['max_abs_linearity_deviation_mhartree']:.3f} mHa "
            f"improvement={improvement:.3f}",
            flush=True,
        )

    _write_csv(outdir / "test_fractional_profiles.csv", fractional_rows)
    _write_csv(outdir / "test_fractional_improvement.csv", fractional_summaries)
    _plot_fractional_improvement(outdir / "test_fractional_improvement.png", fractional_summaries)

    improved_count = sum(1 for row in fractional_summaries if bool(row["improved"]))
    train_final = _loss_metrics_for_dataset(final_params, functional, train_data, avg_loss_fn)
    test_final = _loss_metrics_for_dataset(final_params, functional, test_data, avg_loss_fn)
    train_initial = _loss_metrics_for_dataset(initial_params, functional, train_data, avg_loss_fn)
    test_initial = _loss_metrics_for_dataset(initial_params, functional, test_data, avg_loss_fn)
    summary = {
        "args": vars(args),
        "split": split,
        "initial_train_metrics": train_initial,
        "final_train_metrics": train_final,
        "initial_test_metrics": test_initial,
        "final_test_metrics": test_final,
        "test_fractional_improved_count": int(improved_count),
        "test_fractional_total": len(fractional_summaries),
        "test_fractional_mean_initial_max_dev_mhartree": float(
            np.mean([row["initial_max_abs_dev_mhartree"] for row in fractional_summaries])
        ),
        "test_fractional_mean_final_max_dev_mhartree": float(
            np.mean([row["final_max_abs_dev_mhartree"] for row in fractional_summaries])
        ),
        "test_fractional_mean_improvement_mhartree": float(
            np.mean([row["improvement_mhartree"] for row in fractional_summaries])
        ),
        "files": {
            "split": str(outdir / "split.json"),
            "history": str(outdir / "history.csv"),
            "training_plot": str(outdir / "training_history.png"),
            "per_molecule_koopmans_metrics": str(outdir / "per_molecule_koopmans_metrics.csv"),
            "test_fractional_profiles": str(outdir / "test_fractional_profiles.csv"),
            "test_fractional_improvement": str(outdir / "test_fractional_improvement.csv"),
            "test_fractional_improvement_plot": str(outdir / "test_fractional_improvement.png"),
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
