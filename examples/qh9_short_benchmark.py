from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from td_graddft import neural_xc, tdscf
from td_graddft.device import put_restricted_molecule_on_device
from td_graddft.neural_xc_presets import (
    DM21_B3LYP_NEURAL_XC_PRESET,
    resolve_coefficient_prior_values,
)
from td_graddft.data.reference import restricted_reference_from_pyscf
from td_graddft.scf import DifferentiableSCF, DifferentiableSCFConfig
from td_graddft.spectra import HARTREE_TO_EV, lorentzian_spectrum, oscillator_strengths
from td_graddft.training import (
    GroundStateDatum,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    ground_state_mse_loss,
    make_ground_state_train_step,
    save_params_checkpoint,
)
from td_graddft.training.targets import _predict_ground_state_total_energy_from_molecule

ATOMIC_SYMBOL = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
}


def _plt():
    import matplotlib.pyplot as plt

    return plt


@dataclass(frozen=True)
class MoleculeEntry:
    db_id: int
    z: np.ndarray
    pos_ang: np.ndarray
    formula: str
    mf: Any
    reference: Any
    ref_energies_au: jnp.ndarray
    ref_osc: jnp.ndarray


@dataclass(frozen=True)
class EvalRow:
    split: str
    db_id: int
    formula: str
    solver: str
    n_ref_states: int
    n_neural_states: int
    compare_states: int
    excitation_mae_ev: float
    reference_total_energy_ha: float
    predicted_total_energy_ha: float
    energy_abs_error_ha: float


@dataclass(frozen=True)
class OrbitalEvalRow:
    split: str
    db_id: int
    formula: str
    orbital: str
    status: str
    overlap: float
    diff_norm: float
    diff_iso_used: float
    diff_scale_used: float
    scf_converged: bool
    scf_cycles: int
    scf_final_rms_density: float
    compare_png: str
    error: str


@dataclass(frozen=True)
class OrbitalEnergyEvalRow:
    split: str
    db_id: int
    formula: str
    mo_index: int
    label: str
    orbital_type: str
    reference_midpoint_ha: float
    predicted_midpoint_ha: float
    reference_midpoint_ev: float
    predicted_midpoint_ev: float
    reference_energy_ha: float
    predicted_energy_ha: float
    reference_energy_ev: float
    predicted_energy_ev: float
    aligned_reference_energy_ha: float
    aligned_predicted_energy_ha: float
    aligned_reference_energy_ev: float
    aligned_predicted_energy_ev: float
    aligned_abs_error_ev: float
    raw_abs_error_ev: float


@dataclass(frozen=True)
class EntryEvaluation:
    row: EvalRow
    ref_curve: jnp.ndarray
    neural_curve: jnp.ndarray
    neural_scf_molecule: Any
    scf_info: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QH9 Neural_xc benchmark on train/test splits."
    )
    parser.add_argument(
        "--db-path",
        default="/Volumes/TF/QH9_db/QH9Stable.db",
        help="path to QH9 sqlite database",
    )
    parser.add_argument("--sample-count", type=int, default=10, help="total molecules")
    parser.add_argument("--train-count", type=int, default=8, help="train molecules")
    parser.add_argument("--seed", type=int, default=20260325, help="sampling seed")
    parser.add_argument("--max-atoms", type=int, default=8, help="sampling atom-count cap")
    parser.add_argument("--basis", default="6-31g", help="PySCF basis for references")
    parser.add_argument("--xc", default="b3lyp", help="PySCF XC for references")
    parser.add_argument("--steps", type=int, default=800, help="training steps")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument(
        "--lr-schedule-boundaries",
        type=int,
        nargs="+",
        default=None,
        help="learning rate decay boundaries (e.g., --lr-schedule-boundaries 500 1000 1500)",
    )
    parser.add_argument(
        "--lr-schedule-decay",
        type=float,
        default=0.1,
        help="learning rate decay factor at each boundary (default: 0.1 = one order of magnitude)",
    )
    parser.add_argument(
        "--lr-decay-every",
        type=int,
        default=0,
        help="staircase decay interval in steps (0 disables)",
    )
    parser.add_argument(
        "--lr-decay-factor",
        type=float,
        default=0.5,
        help="staircase decay factor applied every --lr-decay-every steps",
    )
    parser.add_argument("--density-weight", type=float, default=0.0, help="density penalty weight")
    parser.add_argument(
        "--density-supervision",
        choices=("spin_summed", "spin_resolved"),
        default=DM21_B3LYP_NEURAL_XC_PRESET.density_supervision,
        help="density matching mode used when --density-weight is nonzero",
    )
    parser.add_argument(
        "--stationarity-weight",
        type=float,
        default=0.0,
        help="fixed-density stationarity penalty weight",
    )
    parser.add_argument(
        "--coefficient-prior-weight",
        type=float,
        default=0.0,
        help="optional DM21-style coefficient prior weight",
    )
    parser.add_argument(
        "--coefficient-prior-values",
        type=float,
        nargs="+",
        default=None,
        help="target channel coefficients matching the Neural_xc basis order",
    )
    parser.add_argument(
        "--coefficient-prior-mode",
        choices=("pointwise", "mean"),
        default=DM21_B3LYP_NEURAL_XC_PRESET.coefficient_prior_mode,
        help="whether to regularize local coefficients or their grid mean",
    )
    parser.add_argument(
        "--energy-mse-weight",
        type=float,
        default=1.0,
        help="weight for the MSE part of the ground-state energy loss",
    )
    parser.add_argument(
        "--energy-mae-weight",
        type=float,
        default=1.0,
        help="weight for the MAE part of the ground-state energy loss",
    )
    parser.add_argument(
        "--energy-normalization",
        choices=("none", "per_electron", "per_atom"),
        default="per_electron",
        help="normalization mode applied to energy error before MAE/MSE",
    )
    parser.add_argument(
        "--jit-train",
        action="store_true",
        help="JIT compile training/eval steps (faster, but may use more memory)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="print training progress every N steps (0 disables)",
    )
    parser.add_argument(
        "--semilocal-xc",
        nargs="+",
        default=list(DM21_B3LYP_NEURAL_XC_PRESET.semilocal_xc),
        help="one or more jax_libxc semilocal specs used as Neural_xc basis channels",
    )
    parser.add_argument(
        "--n-semilocal-channels",
        type=int,
        default=None,
        help="required only when a custom semilocal callback returns multiple channels",
    )
    parser.add_argument(
        "--hf-input-mode",
        choices=("total_only", "spin_resolved"),
        default=DM21_B3LYP_NEURAL_XC_PRESET.hf_input_mode,
        help="how projected HF energy density is exposed to the MLP",
    )
    parser.add_argument(
        "--coefficient-positivity",
        choices=("clip", "softplus"),
        default=DM21_B3LYP_NEURAL_XC_PRESET.coefficient_positivity,
        help="how nonnegative channel coefficients are enforced",
    )
    parser.add_argument("--nstates", type=int, default=8, help="states per molecule")
    parser.add_argument("--eta-ev", type=float, default=0.20, help="Lorentzian broadening (eV)")
    parser.add_argument("--grid-min-ev", type=float, default=0.0, help="spectrum min (eV)")
    parser.add_argument("--grid-max-ev", type=float, default=20.0, help="spectrum max (eV)")
    parser.add_argument("--grid-points", type=int, default=1400, help="spectrum grid points")
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[64, 64],
        help="MLP hidden dims",
    )
    parser.add_argument(
        "--xyzrender-src",
        default="/Volumes/TF/QH9_db/xyzrender/src",
        help="xyzrender source directory (contains xyzrender package)",
    )
    parser.add_argument(
        "--skip-orbital-compare",
        action="store_true",
        help="skip HOMO-1/HOMO/LUMO/LUMO+1 orbital comparison rendering",
    )
    parser.add_argument(
        "--skip-orbital-energy-compare",
        action="store_true",
        help="skip frontier orbital energy parity plotting",
    )
    parser.add_argument(
        "--orbital-energy-window",
        type=int,
        default=10,
        help="number of frontier occupied orbitals including HOMO and virtual orbitals including LUMO",
    )
    parser.add_argument("--orbital-iso", type=float, default=0.05)
    parser.add_argument("--orbital-diff-iso", type=float, default=0.03)
    parser.add_argument("--orbital-mo-blur", type=float, default=1.2)
    parser.add_argument("--orbital-mo-upsample", type=int, default=4)
    parser.add_argument("--orbital-cube-grid", type=int, default=48)
    parser.add_argument("--orbital-canvas-size", type=int, default=620)
    parser.add_argument(
        "--disable-orbital-frontier-match",
        action="store_true",
        help="disable overlap-based HOMO-1/HOMO and LUMO/LUMO+1 orbital matching",
    )
    parser.add_argument(
        "--orbital-frontier-match-window",
        type=int,
        default=6,
        help="number of near-frontier candidate orbitals used for overlap matching",
    )
    parser.add_argument("--orbital-scf-max-cycle", type=int, default=96)
    parser.add_argument("--orbital-scf-damping", type=float, default=0.90)
    parser.add_argument("--orbital-scf-conv-tol-density", type=float, default=1e-6)
    parser.add_argument("--orbital-scf-vxc-clip", type=float, default=20.0)
    parser.add_argument(
        "--orbital-scf-iterate-selection",
        choices=("final", "best_rms", "first_converged"),
        default="first_converged",
    )
    parser.add_argument("--outdir", default="outputs/qh9_short_benchmark", help="output folder")
    return parser.parse_args()


def _normalize_semilocal_arg(values: list[str] | str) -> str | tuple[str, ...]:
    if isinstance(values, str):
        return values
    if len(values) == 1:
        return values[0]
    return tuple(values)


def _formula_from_z(z: np.ndarray) -> str:
    counts: dict[str, int] = {}
    for zi in z:
        sym = ATOMIC_SYMBOL[int(zi)]
        counts[sym] = counts.get(sym, 0) + 1

    ordered = []
    if "C" in counts:
        ordered.append(("C", counts.pop("C")))
    if "H" in counts:
        ordered.append(("H", counts.pop("H")))
    for sym in sorted(counts):
        ordered.append((sym, counts[sym]))
    return "".join(sym if n == 1 else f"{sym}{n}" for sym, n in ordered)


def _iter_small_even_ids(conn: sqlite3.Connection, max_atoms: int) -> list[int]:
    ids: list[int] = []
    cur = conn.execute("SELECT id, Z FROM data WHERE N <= ?", (max_atoms,))
    for db_id, z_blob in cur:
        z = np.frombuffer(z_blob, dtype=np.int32)
        if int(np.sum(z)) % 2 == 0:
            ids.append(int(db_id))
    return ids


def _fetch_molecule(conn: sqlite3.Connection, db_id: int) -> tuple[np.ndarray, np.ndarray]:
    row = conn.execute("SELECT N, Z, pos FROM data WHERE id = ?", (db_id,)).fetchone()
    if row is None:
        raise KeyError(f"QH9 id {db_id} not found")
    n, z_blob, pos_blob = row
    z = np.frombuffer(z_blob, dtype=np.int32).copy()
    pos = np.frombuffer(pos_blob, dtype=np.float64).reshape(int(n), 3).copy()
    return z, pos


def _build_atom_block(z: np.ndarray, pos_ang: np.ndarray) -> str:
    lines = []
    for zi, xyz in zip(z, pos_ang, strict=True):
        sym = ATOMIC_SYMBOL[int(zi)]
        lines.append(f"{sym} {xyz[0]: .12f} {xyz[1]: .12f} {xyz[2]: .12f}")
    return "\n".join(lines)


def _build_entry(
    *,
    db_id: int,
    z: np.ndarray,
    pos_ang: np.ndarray,
    basis: str,
    xc: str,
    nstates: int,
) -> MoleculeEntry:
    from pyscf import dft, gto

    atom = _build_atom_block(z, pos_ang)
    mol = gto.M(
        atom=atom,
        basis=basis,
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF not converged")

    td = mf.TDDFT()
    nocc = int(np.count_nonzero(mf.mo_occ > 1e-8))
    nvir = int(mf.mo_coeff.shape[-1] - nocc)
    nstates_eff = min(max(nstates, 1), nocc * nvir)
    td.nstates = nstates_eff
    td.kernel()

    reference = put_restricted_molecule_on_device(restricted_reference_from_pyscf(mf))
    return MoleculeEntry(
        db_id=db_id,
        z=z,
        pos_ang=pos_ang,
        formula=_formula_from_z(z),
        mf=mf,
        reference=reference,
        ref_energies_au=jnp.asarray(td.e),
        ref_osc=jnp.asarray(td.oscillator_strength()),
    )


def _sample_entries(args: argparse.Namespace) -> list[MoleculeEntry]:
    rng = np.random.default_rng(args.seed)
    conn = sqlite3.connect(args.db_path)

    even_ids = _iter_small_even_ids(conn, args.max_atoms)
    if len(even_ids) < args.sample_count:
        raise RuntimeError(
            f"Not enough small/even-electron molecules: found {len(even_ids)}, "
            f"need {args.sample_count}."
        )

    rng.shuffle(even_ids)
    selected: list[MoleculeEntry] = []
    tried = 0
    for db_id in even_ids:
        tried += 1
        if len(selected) >= args.sample_count:
            break
        z, pos = _fetch_molecule(conn, db_id)
        try:
            entry = _build_entry(
                db_id=db_id,
                z=z,
                pos_ang=pos,
                basis=args.basis,
                xc=args.xc,
                nstates=args.nstates,
            )
        except Exception:
            continue
        selected.append(entry)

    conn.close()

    if len(selected) < args.sample_count:
        raise RuntimeError(
            f"Only collected {len(selected)} molecules after trying {tried}; "
            f"need {args.sample_count}."
        )
    return selected


def _train_functional(
    train_entries: list[MoleculeEntry],
    test_entries: list[MoleculeEntry],
    *,
    steps: int,
    learning_rate: float,
    lr_schedule_boundaries: list[int] | None,
    lr_schedule_decay: float,
    lr_decay_every: int,
    lr_decay_factor: float,
    density_weight: float,
    stationarity_weight: float,
    coefficient_prior_weight: float,
    coefficient_prior_values: tuple[float, ...] | None,
    coefficient_prior_mode: str,
    energy_mse_weight: float,
    energy_mae_weight: float,
    energy_normalization: str,
    density_supervision: str,
    hidden_dims: tuple[int, ...],
    semilocal_xc: str | tuple[str, ...],
    n_semilocal_channels: int | None,
    hf_input_mode: str,
    coefficient_positivity: str,
    jit_train: bool,
    log_interval: int,
):
    functional = neural_xc.Functional(
        semilocal_xc=semilocal_xc,
        n_semilocal_channels=n_semilocal_channels,
        hf_input_mode=hf_input_mode,
        coefficient_positivity=coefficient_positivity,
        hidden_dims=hidden_dims,
        name="qh9_short_neural_xc",
    )
    train_data = [
        GroundStateDatum(
            molecule=e.reference,
            target_total_energy=jnp.asarray(e.reference.mf_energy),
            density_constraint_weight=density_weight,
            stationarity_constraint_weight=stationarity_weight,
        )
        for e in train_entries
    ]
    test_data = [
        GroundStateDatum(
            molecule=e.reference,
            target_total_energy=jnp.asarray(e.reference.mf_energy),
            density_constraint_weight=density_weight,
            stationarity_constraint_weight=stationarity_weight,
        )
        for e in test_entries
    ]

    if lr_schedule_boundaries:
        boundaries_and_scales = {
            boundary: lr_schedule_decay
            for boundary in lr_schedule_boundaries
        }
        lr_schedule = optax.piecewise_constant_schedule(
            init_value=learning_rate,
            boundaries_and_scales=boundaries_and_scales,
        )
        optimizer = optax.adam(learning_rate=lr_schedule)
    elif lr_decay_every > 0:
        lr_schedule = optax.exponential_decay(
            init_value=float(learning_rate),
            transition_steps=int(lr_decay_every),
            decay_rate=float(lr_decay_factor),
            staircase=True,
        )
        optimizer = optax.adam(learning_rate=lr_schedule)
    else:
        lr_schedule = lambda step: jnp.asarray(learning_rate, dtype=jnp.float64)
        optimizer = optax.adam(learning_rate)

    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(0),
        train_entries[0].reference,
        optimizer,
    )
    gs_training = GroundStateTrainingConfig(
        mode="fixed_density",
        energy_mse_weight=energy_mse_weight,
        energy_mae_weight=energy_mae_weight,
        energy_normalization=energy_normalization,
        density_supervision=density_supervision,
        coefficient_prior_weight=coefficient_prior_weight,
        coefficient_prior_values=coefficient_prior_values,
        coefficient_prior_mode=coefficient_prior_mode,
    )
    train_step = make_ground_state_train_step(functional, training_config=gs_training)
    if jit_train:
        step_fn = jax.jit(lambda st: train_step(st, train_data))
        eval_train_fn = jax.jit(
            lambda params: ground_state_mse_loss(
                params,
                functional,
                train_data,
                training_config=gs_training,
            )
        )
        if test_data:
            eval_test_fn = jax.jit(
                lambda params: ground_state_mse_loss(
                    params,
                    functional,
                    test_data,
                    training_config=gs_training,
                )
            )
        else:
            eval_test_fn = None
    else:
        step_fn = lambda st: train_step(st, train_data)
        eval_train_fn = lambda params: ground_state_mse_loss(
            params,
            functional,
            train_data,
            training_config=gs_training,
        )
        if test_data:
            eval_test_fn = lambda params: ground_state_mse_loss(
                params,
                functional,
                test_data,
                training_config=gs_training,
            )
        else:
            eval_test_fn = None

    initial_train_loss, _ = eval_train_fn(state.params)
    if eval_test_fn is not None:
        initial_test_loss, _ = eval_test_fn(state.params)
        test_loss_history = [float(initial_test_loss)]
    else:
        test_loss_history = [float("nan")]

    train_loss_history = [float(initial_train_loss)]
    min_train_loss = float(initial_train_loss)
    min_train_loss_step = 0
    best_params = state.params
    min_test_loss = float(test_loss_history[0]) if test_data else float("nan")
    min_test_loss_step = 0

    for step in range(1, steps + 1):
        state, _ = step_fn(state)
        train_loss, _ = eval_train_fn(state.params)
        train_loss = float(train_loss)
        train_loss_history.append(train_loss)
        if train_loss < min_train_loss:
            min_train_loss = train_loss
            min_train_loss_step = step
            best_params = state.params

        if eval_test_fn is not None:
            test_loss, _ = eval_test_fn(state.params)
            test_loss = float(test_loss)
            test_loss_history.append(test_loss)
            if test_loss < min_test_loss:
                min_test_loss = test_loss
                min_test_loss_step = step
        else:
            test_loss_history.append(float("nan"))

        if log_interval > 0 and (step == 1 or step == steps or step % log_interval == 0):
            current_test = test_loss_history[-1]
            current_lr = float(lr_schedule(step))
            print(
                "[QH9][train] "
                f"step={step}/{steps} "
                f"lr={current_lr:.6e} "
                f"train_loss={train_loss:.6e} "
                f"test_loss={current_test:.6e}",
                flush=True,
            )

    final_train_loss, _ = eval_train_fn(state.params)
    if eval_test_fn is not None:
        final_test_loss, _ = eval_test_fn(state.params)
    else:
        final_test_loss = jnp.asarray(jnp.nan)
    return (
        functional,
        best_params,
        state.params,
        train_loss_history,
        test_loss_history,
        min_train_loss,
        min_train_loss_step,
        float(final_train_loss),
        min_test_loss,
        min_test_loss_step,
        float(final_test_loss),
    )


def _evaluate_entry(
    entry: MoleculeEntry,
    *,
    split: str,
    functional: Any,
    params: Any,
    density_supervision: str,
    nstates: int,
    eta_ev: float,
    grid_ev: jnp.ndarray,
    scf_max_cycle: int,
    scf_damping: float,
    scf_conv_tol_density: float,
    scf_vxc_clip: float,
    scf_iterate_selection: str,
) -> EntryEvaluation:
    scf_training_config = GroundStateTrainingConfig(
        mode="self_consistent",
        density_supervision=density_supervision,
        scf_max_cycle=scf_max_cycle,
        scf_damping=scf_damping,
        scf_conv_tol_density=scf_conv_tol_density,
        scf_vxc_clip=scf_vxc_clip,
        scf_iterate_selection=scf_iterate_selection,
    )
    scf = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            max_cycle=scf_max_cycle,
            damping=scf_damping,
            conv_tol_density=scf_conv_tol_density,
            vxc_clip=scf_vxc_clip,
            iterate_selection=scf_iterate_selection,
        )
    )
    neural_scf_molecule, scf_info = scf.run(entry.reference, functional, params)

    td = tdscf.TDDFT(
        neural_scf_molecule,
        xc_functional=functional,
        xc_params=params,
    )
    result = td.kernel(nstates=nstates)
    solver_label = "Casida"
    if result.excitation_energies.size == 0:
        result = tdscf.TDA(
            neural_scf_molecule,
            xc_functional=functional,
            xc_params=params,
        ).kernel(nstates=nstates)
        solver_label = "TDA fallback"

    neural_energies_au = result.excitation_energies
    neural_osc = oscillator_strengths(neural_scf_molecule, result)

    ncompare = int(min(entry.ref_energies_au.size, neural_energies_au.size, nstates))
    mae_ev = float(
        jnp.mean(
            jnp.abs(
                entry.ref_energies_au[:ncompare] * HARTREE_TO_EV
                - neural_energies_au[:ncompare] * HARTREE_TO_EV
            )
        )
    )
    predicted_total = float(
        _predict_ground_state_total_energy_from_molecule(
            params,
            functional,
            neural_scf_molecule,
        )
    )
    target_total = float(entry.reference.mf_energy)

    ref_curve = lorentzian_spectrum(
        entry.ref_energies_au * HARTREE_TO_EV,
        entry.ref_osc,
        grid_ev,
        eta=eta_ev,
    )
    neural_curve = lorentzian_spectrum(
        neural_energies_au * HARTREE_TO_EV,
        neural_osc,
        grid_ev,
        eta=eta_ev,
    )

    row = EvalRow(
        split=split,
        db_id=entry.db_id,
        formula=entry.formula,
        solver=solver_label,
        n_ref_states=int(entry.ref_energies_au.size),
        n_neural_states=int(neural_energies_au.size),
        compare_states=ncompare,
        excitation_mae_ev=mae_ev,
        reference_total_energy_ha=target_total,
        predicted_total_energy_ha=predicted_total,
        energy_abs_error_ha=abs(predicted_total - target_total),
    )
    return EntryEvaluation(
        row=row,
        ref_curve=ref_curve,
        neural_curve=neural_curve,
        neural_scf_molecule=neural_scf_molecule,
        scf_info=scf_info,
    )


def _write_selected(entries: list[MoleculeEntry], train_count: int, outdir: Path) -> None:
    path = outdir / "selected_molecules.csv"
    with path.open("w", encoding="utf-8") as f:
        f.write("split,index,db_id,formula,natoms,reference_energy_ha\n")
        for idx, entry in enumerate(entries):
            split = "train" if idx < train_count else "test"
            f.write(
                f"{split},{idx},{entry.db_id},{entry.formula},{entry.z.size},"
                f"{float(entry.reference.mf_energy):.12f}\n"
            )


def _write_training_curve(
    train_loss_history: list[float],
    test_loss_history: list[float],
    outdir: Path,
) -> None:
    plt = _plt()
    csv_path = outdir / "training_loss.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("step,train_loss,test_loss\n")
        for step, (train_loss, test_loss) in enumerate(
            zip(train_loss_history, test_loss_history, strict=True)
        ):
            f.write(f"{step},{train_loss:.16e},{test_loss:.16e}\n")

    fig, ax = plt.subplots(figsize=(7, 4))
    train_values = np.maximum(np.asarray(train_loss_history, dtype=float), 1e-16)
    test_values = np.maximum(np.asarray(test_loss_history, dtype=float), 1e-16)
    steps = np.arange(len(train_values))
    ax.plot(steps, train_values, lw=2.0, label="Train loss")
    ax.plot(steps, test_values, lw=2.0, label="Test loss")
    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Ground-state loss (log scale)")
    ax.set_title("QH9 short benchmark train/test loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "training_loss.png", dpi=180)
    plt.close(fig)


def _write_eval_rows(rows: list[EvalRow], outdir: Path) -> None:
    path = outdir / "evaluation_metrics.csv"
    with path.open("w", encoding="utf-8") as f:
        f.write(
            "split,db_id,formula,solver,n_ref_states,n_neural_states,compare_states,"
            "excitation_mae_ev,reference_total_energy_ha,predicted_total_energy_ha,"
            "energy_abs_error_ha\n"
        )
        for row in rows:
            f.write(
                f"{row.split},{row.db_id},{row.formula},{row.solver},"
                f"{row.n_ref_states},{row.n_neural_states},{row.compare_states},"
                f"{row.excitation_mae_ev:.10f},{row.reference_total_energy_ha:.12f},"
                f"{row.predicted_total_energy_ha:.12f},{row.energy_abs_error_ha:.12f}\n"
            )


def _write_ground_state_parity_plot(rows: list[EvalRow], outdir: Path) -> tuple[Path, Path]:
    plt = _plt()
    png_path = outdir / "ground_state_energy_parity.png"
    csv_path = outdir / "ground_state_energy_parity.csv"

    train_rows = [row for row in rows if row.split == "train"]
    test_rows = [row for row in rows if row.split == "test"]
    all_rows = train_rows + test_rows
    if not all_rows:
        raise ValueError("No evaluation rows available for ground-state parity plotting.")

    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "split,db_id,formula,reference_total_energy_ha,predicted_total_energy_ha,energy_abs_error_ha\n"
        )
        for row in all_rows:
            handle.write(
                f"{row.split},{row.db_id},{row.formula},{row.reference_total_energy_ha:.12f},"
                f"{row.predicted_total_energy_ha:.12f},{row.energy_abs_error_ha:.12f}\n"
            )

    ref_all = np.asarray([row.reference_total_energy_ha for row in all_rows], dtype=float)
    pred_all = np.asarray([row.predicted_total_energy_ha for row in all_rows], dtype=float)
    axis_min = float(min(np.min(ref_all), np.min(pred_all)))
    axis_max = float(max(np.max(ref_all), np.max(pred_all)))
    span = axis_max - axis_min
    margin = 0.04 * (span if span > 1e-10 else 1.0)
    lo = axis_min - margin
    hi = axis_max + margin

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    if train_rows:
        ax.scatter(
            [row.reference_total_energy_ha for row in train_rows],
            [row.predicted_total_energy_ha for row in train_rows],
            s=54,
            c="#1f77b4",
            alpha=0.88,
            edgecolors="white",
            linewidths=0.6,
            label=f"Train ({len(train_rows)})",
        )
    if test_rows:
        ax.scatter(
            [row.reference_total_energy_ha for row in test_rows],
            [row.predicted_total_energy_ha for row in test_rows],
            s=58,
            c="#d62728",
            alpha=0.92,
            marker="s",
            edgecolors="white",
            linewidths=0.6,
            label=f"Test ({len(test_rows)})",
        )

    ax.plot([lo, hi], [lo, hi], "--", color="#555555", linewidth=1.2, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Reference ground-state energy (Ha)")
    ax.set_ylabel("Predicted ground-state energy (Ha)")
    ax.set_title("Ground-State Energy Parity (Train/Test)")
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return png_path, csv_path


def _safe_text(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def _prepare_cairo_runtime() -> None:
    """Ensure cairocffi can find cairo on this macOS environment."""

    homebrew_lib = "/opt/homebrew/lib"
    current = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    parts = [p for p in current.split(":") if p]
    if homebrew_lib not in parts:
        parts.insert(0, homebrew_lib)
    if "/usr/lib" not in parts:
        parts.append("/usr/lib")
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(parts)


def _orbital_indices(mo_occ: np.ndarray) -> dict[str, int]:
    occ = np.where(mo_occ > 1e-8)[0]
    vir = np.where(mo_occ <= 1e-8)[0]
    if occ.size < 2:
        raise RuntimeError("Need at least 2 occupied orbitals for HOMO-1/HOMO rendering.")
    if vir.size < 2:
        raise RuntimeError("Need at least 2 virtual orbitals for LUMO/LUMO+1 rendering.")
    return {
        "HOMO-1": int(occ[-2]),
        "HOMO": int(occ[-1]),
        "LUMO": int(vir[0]),
        "LUMO+1": int(vir[1]),
    }


def _restricted_channel(mo_coeff: Any, mo_occ: Any) -> tuple[np.ndarray, np.ndarray]:
    coeff = np.asarray(mo_coeff, dtype=float)
    occ = np.asarray(mo_occ, dtype=float)
    if coeff.ndim == 3:
        coeff = coeff[0]
    if occ.ndim == 2:
        occ = occ[0]
    return coeff, occ


def _restricted_energies_and_occ(mo_energy: Any, mo_occ: Any) -> tuple[np.ndarray, np.ndarray]:
    energies = np.asarray(mo_energy, dtype=float)
    occ = np.asarray(mo_occ, dtype=float)
    if energies.ndim == 2:
        energies = energies[0]
    if occ.ndim == 2:
        occ = occ[0]
    return energies, occ


def _safe_label(label: str) -> str:
    return label.replace("+", "p").replace("-", "m").replace(" ", "_").lower()


def _frontier_orbital_label(index: int, homo_idx: int, lumo_idx: int) -> tuple[str, str]:
    if index <= homo_idx:
        offset = homo_idx - index
        label = "HOMO" if offset == 0 else f"HOMO-{offset}"
        return label, "occupied"
    offset = index - lumo_idx
    label = "LUMO" if offset == 0 else f"LUMO+{offset}"
    return label, "virtual"


def _orbital_energy_rows_for_entry(
    *,
    split: str,
    entry: MoleculeEntry,
    neural_scf_molecule: Any,
    window: int,
) -> list[OrbitalEnergyEvalRow]:
    ref_energies_ha, ref_occ = _restricted_energies_and_occ(entry.mf.mo_energy, entry.mf.mo_occ)
    pred_energies_ha, pred_occ = _restricted_energies_and_occ(
        neural_scf_molecule.mo_energy,
        neural_scf_molecule.mo_occ,
    )
    nmo = int(min(ref_energies_ha.size, pred_energies_ha.size, ref_occ.size, pred_occ.size))
    if nmo == 0:
        return []

    occ_idx = np.where(ref_occ[:nmo] > 1e-8)[0]
    vir_idx = np.where(ref_occ[:nmo] <= 1e-8)[0]
    if occ_idx.size == 0 or vir_idx.size == 0:
        return []

    window = max(int(window), 1)
    homo_idx = int(occ_idx[-1])
    lumo_idx = int(vir_idx[0])
    ref_zero_ha = 0.5 * float(ref_energies_ha[homo_idx] + ref_energies_ha[lumo_idx])
    pred_zero_ha = 0.5 * float(pred_energies_ha[homo_idx] + pred_energies_ha[lumo_idx])
    ref_zero_ev = ref_zero_ha * HARTREE_TO_EV
    pred_zero_ev = pred_zero_ha * HARTREE_TO_EV
    occ_start = max(homo_idx - window + 1, 0)
    vir_stop = min(lumo_idx + window, nmo)
    indices = list(range(occ_start, homo_idx + 1)) + list(range(lumo_idx, vir_stop))

    rows: list[OrbitalEnergyEvalRow] = []
    for idx in indices:
        label, orbital_type = _frontier_orbital_label(idx, homo_idx, lumo_idx)
        ref_ha = float(ref_energies_ha[idx])
        pred_ha = float(pred_energies_ha[idx])
        ref_ev = ref_ha * HARTREE_TO_EV
        pred_ev = pred_ha * HARTREE_TO_EV
        aligned_ref_ha = ref_ha - ref_zero_ha
        aligned_pred_ha = pred_ha - pred_zero_ha
        aligned_ref_ev = aligned_ref_ha * HARTREE_TO_EV
        aligned_pred_ev = aligned_pred_ha * HARTREE_TO_EV
        rows.append(
            OrbitalEnergyEvalRow(
                split=split,
                db_id=entry.db_id,
                formula=entry.formula,
                mo_index=int(idx),
                label=label,
                orbital_type=orbital_type,
                reference_midpoint_ha=ref_zero_ha,
                predicted_midpoint_ha=pred_zero_ha,
                reference_midpoint_ev=ref_zero_ev,
                predicted_midpoint_ev=pred_zero_ev,
                reference_energy_ha=ref_ha,
                predicted_energy_ha=pred_ha,
                reference_energy_ev=ref_ev,
                predicted_energy_ev=pred_ev,
                aligned_reference_energy_ha=aligned_ref_ha,
                aligned_predicted_energy_ha=aligned_pred_ha,
                aligned_reference_energy_ev=aligned_ref_ev,
                aligned_predicted_energy_ev=aligned_pred_ev,
                aligned_abs_error_ev=abs(aligned_pred_ev - aligned_ref_ev),
                raw_abs_error_ev=abs(pred_ev - ref_ev),
            )
        )
    return rows


def _match_frontier_pair(
    *,
    ref_coeff: np.ndarray,
    neural_coeff: np.ndarray,
    overlap: np.ndarray,
    ref_pair: tuple[int, int],
    candidate_pool: np.ndarray,
) -> tuple[int, int]:
    if candidate_pool.size < 2:
        raise RuntimeError("Need at least two candidate orbitals for frontier matching.")

    ref_pair_vec = np.asarray(ref_coeff[:, [ref_pair[0], ref_pair[1]]], dtype=float)
    cand_pair_vec = np.asarray(neural_coeff[:, candidate_pool], dtype=float)
    score = np.abs(ref_pair_vec.T @ overlap @ cand_pair_vec)

    best_score = -1.0
    best_0 = 0
    best_1 = 1
    for j0 in range(candidate_pool.size):
        for j1 in range(candidate_pool.size):
            if j0 == j1:
                continue
            current = float(score[0, j0] + score[1, j1])
            if current > best_score:
                best_score = current
                best_0 = j0
                best_1 = j1

    return int(candidate_pool[best_0]), int(candidate_pool[best_1])


def _render_orbital_surfaces(
    *,
    entry: MoleculeEntry,
    split: str,
    neural_molecule: Any,
    xyzrender_src: str,
    outdir: Path,
    iso: float,
    diff_iso: float,
    mo_blur: float,
    mo_upsample: int,
    cube_grid: int,
    canvas_size: int,
    match_frontier_by_overlap: bool,
    frontier_match_window: int,
) -> tuple[
    dict[str, dict[str, Path]],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    if not Path(xyzrender_src).exists():
        raise FileNotFoundError(f"xyzrender source not found: {xyzrender_src}")

    _prepare_cairo_runtime()
    if xyzrender_src not in sys.path:
        sys.path.insert(0, xyzrender_src)
    from xyzrender import load as xr_load, render as xr_render
    from pyscf.tools import cubegen

    stem = f"{split}_{entry.db_id}_{_safe_text(entry.formula)}"
    orbital_dir = outdir / "orbital_surfaces" / stem
    cube_ref_dir = orbital_dir / "cubes" / "reference"
    cube_neural_dir = orbital_dir / "cubes" / "neural"
    cube_diff_dir = orbital_dir / "cubes" / "difference"
    png_ref_dir = orbital_dir / "png" / "reference"
    png_neural_dir = orbital_dir / "png" / "neural"
    png_diff_dir = orbital_dir / "png" / "difference"
    for p in (
        cube_ref_dir,
        cube_neural_dir,
        cube_diff_dir,
        png_ref_dir,
        png_neural_dir,
        png_diff_dir,
    ):
        p.mkdir(parents=True, exist_ok=True)

    ref_coeff, ref_occ = _restricted_channel(entry.mf.mo_coeff, entry.mf.mo_occ)
    neural_coeff, neural_occ = _restricted_channel(neural_molecule.mo_coeff, neural_molecule.mo_occ)
    ref_idx = _orbital_indices(ref_occ)
    neural_idx = _orbital_indices(neural_occ)
    if getattr(entry.reference, "overlap_matrix", None) is None:
        overlap = np.eye(ref_coeff.shape[0], dtype=float)
    else:
        overlap = np.asarray(entry.reference.overlap_matrix, dtype=float)
    labels = ("HOMO-1", "HOMO", "LUMO", "LUMO+1")

    if match_frontier_by_overlap:
        occupied_ref = np.where(ref_occ > 1e-8)[0]
        virtual_ref = np.where(ref_occ <= 1e-8)[0]
        occupied_neural = np.where(neural_occ > 1e-8)[0]
        virtual_neural = np.where(neural_occ <= 1e-8)[0]

        occ_window = int(np.clip(frontier_match_window, 2, occupied_neural.size))
        vir_window = int(np.clip(frontier_match_window, 2, virtual_neural.size))
        occ_pool = occupied_neural[-occ_window:]
        vir_pool = virtual_neural[:vir_window]

        matched_homo_m1, matched_homo = _match_frontier_pair(
            ref_coeff=ref_coeff,
            neural_coeff=neural_coeff,
            overlap=overlap,
            ref_pair=(int(occupied_ref[-2]), int(occupied_ref[-1])),
            candidate_pool=occ_pool,
        )
        matched_lumo, matched_lumo_p1 = _match_frontier_pair(
            ref_coeff=ref_coeff,
            neural_coeff=neural_coeff,
            overlap=overlap,
            ref_pair=(int(virtual_ref[0]), int(virtual_ref[1])),
            candidate_pool=vir_pool,
        )
        neural_idx["HOMO-1"] = matched_homo_m1
        neural_idx["HOMO"] = matched_homo
        neural_idx["LUMO"] = matched_lumo
        neural_idx["LUMO+1"] = matched_lumo_p1

    outputs: dict[str, dict[str, Path]] = {}
    diff_norms: dict[str, float] = {}
    aligned_overlaps: dict[str, float] = {}
    diff_iso_used: dict[str, float] = {}
    diff_scale_used: dict[str, float] = {}
    for label in labels:
        ref_vec = np.asarray(ref_coeff[:, ref_idx[label]], dtype=float)
        neural_vec = np.asarray(neural_coeff[:, neural_idx[label]], dtype=float)
        phase = float(ref_vec.T @ overlap @ neural_vec)
        if phase < 0.0:
            neural_vec = -neural_vec
            phase = -phase
        diff_vec = neural_vec - ref_vec

        diff_norm = float(np.sqrt(np.maximum(diff_vec.T @ overlap @ diff_vec, 0.0)))
        diff_norms[label] = diff_norm
        aligned_overlaps[label] = phase

        label_stem = _safe_label(label)
        cube_ref = cube_ref_dir / f"{label_stem}.cube"
        cube_neural = cube_neural_dir / f"{label_stem}.cube"
        cube_diff = cube_diff_dir / f"{label_stem}.cube"
        png_ref = png_ref_dir / f"{label_stem}.png"
        png_neural = png_neural_dir / f"{label_stem}.png"
        png_diff = png_diff_dir / f"{label_stem}.png"

        cubegen.orbital(
            entry.mf.mol,
            str(cube_ref),
            ref_vec,
            nx=cube_grid,
            ny=cube_grid,
            nz=cube_grid,
        )
        cubegen.orbital(
            entry.mf.mol,
            str(cube_neural),
            neural_vec,
            nx=cube_grid,
            ny=cube_grid,
            nz=cube_grid,
        )
        cubegen.orbital(
            entry.mf.mol,
            str(cube_diff),
            diff_vec,
            nx=cube_grid,
            ny=cube_grid,
            nz=cube_grid,
        )

        diff_scale = 1.0
        diff_mol = xr_load(str(cube_diff))
        grid_data = getattr(getattr(diff_mol, "cube_data", None), "grid_data", None)
        max_abs = float(np.max(np.abs(np.asarray(grid_data)))) if grid_data is not None else 0.0
        target_peak = max(diff_iso * 2.5, 2e-2)
        if 0.0 < max_abs < target_peak:
            diff_scale = target_peak / max_abs
            cubegen.orbital(
                entry.mf.mol,
                str(cube_diff),
                diff_vec * diff_scale,
                nx=cube_grid,
                ny=cube_grid,
                nz=cube_grid,
            )
            diff_mol = xr_load(str(cube_diff))
            grid_data = getattr(getattr(diff_mol, "cube_data", None), "grid_data", None)
            max_abs = float(np.max(np.abs(np.asarray(grid_data)))) if grid_data is not None else 0.0

        if max_abs > 1e-8:
            cur_diff_iso = min(diff_iso, 0.5 * max_abs)
            cur_diff_iso = max(cur_diff_iso, 0.1 * max_abs)
        else:
            cur_diff_iso = diff_iso
        diff_iso_used[label] = float(cur_diff_iso)
        diff_scale_used[label] = float(diff_scale)

        for cube_path, png_path, cur_iso in (
            (cube_ref, png_ref, iso),
            (cube_neural, png_neural, iso),
            (cube_diff, png_diff, cur_diff_iso),
        ):
            cube_mol = xr_load(str(cube_path))
            xr_render(
                cube_mol,
                output=str(png_path),
                config="flat",
                hy=True,
                mo=True,
                iso=cur_iso,
                mo_blur=mo_blur,
                mo_upsample=mo_upsample,
                transparent=True,
                canvas_size=canvas_size,
                mo_pos_color="#2F80ED",
                mo_neg_color="#C0392B",
            )

        outputs[label] = {
            "reference": png_ref,
            "neural": png_neural,
            "difference": png_diff,
        }

    return outputs, diff_norms, aligned_overlaps, diff_iso_used, diff_scale_used


def _plot_orbital_compare(
    *,
    orbital_label: str,
    ref_png: Path,
    neural_png: Path,
    diff_png: Path,
    iso: float,
    diff_iso: float,
    overlap_val: float,
    diff_norm: float,
    diff_scale: float,
    out_png: Path,
) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.8))
    for ax, title, path in (
        (axes[0], f"Reference {orbital_label}\niso=±{iso:.3f}", ref_png),
        (axes[1], f"Neural_xc {orbital_label}\niso=±{iso:.3f}", neural_png),
        (axes[2], f"Difference Δ{orbital_label}\niso=±{diff_iso:.4f}", diff_png),
    ):
        img = plt.imread(path)
        ax.imshow(img)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"{orbital_label} real vs neural | overlap={overlap_val:.4f} | ||Δψ||_S={diff_norm:.4f} | Δscale={diff_scale:.2f}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _write_orbital_eval_rows(rows: list[OrbitalEvalRow], outdir: Path) -> None:
    path = outdir / "orbital_metrics.csv"
    with path.open("w", encoding="utf-8") as f:
        f.write(
            "split,db_id,formula,orbital,status,overlap,diff_norm,diff_iso_used,"
            "diff_scale_used,scf_converged,scf_cycles,scf_final_rms_density,compare_png,error\n"
        )
        for row in rows:
            f.write(
                f"{row.split},{row.db_id},{row.formula},{row.orbital},{row.status},"
                f"{row.overlap:.10f},{row.diff_norm:.10f},{row.diff_iso_used:.10f},"
                f"{row.diff_scale_used:.10f},{int(row.scf_converged)},{row.scf_cycles},"
                f"{row.scf_final_rms_density:.12e},{row.compare_png},{row.error}\n"
            )


def _write_orbital_energy_parity(
    rows: list[OrbitalEnergyEvalRow],
    *,
    window: int,
    outdir: Path,
) -> tuple[Path, Path, float, float, float, float]:
    plt = _plt()
    stem = f"orbital_energy_parity_homo{window}_lumo{window}"
    csv_path = outdir / f"{stem}.csv"
    png_path = outdir / f"{stem}.png"

    with csv_path.open("w", encoding="utf-8") as f:
        f.write(
            "split,db_id,formula,mo_index,label,orbital_type,reference_midpoint_ha,"
            "predicted_midpoint_ha,reference_midpoint_ev,predicted_midpoint_ev,"
            "reference_energy_ha,predicted_energy_ha,reference_energy_ev,predicted_energy_ev,"
            "aligned_reference_energy_ha,aligned_predicted_energy_ha,aligned_reference_energy_ev,"
            "aligned_predicted_energy_ev,aligned_abs_error_ev,raw_abs_error_ev\n"
        )
        for row in rows:
            f.write(
                f"{row.split},{row.db_id},{row.formula},{row.mo_index},{row.label},"
                f"{row.orbital_type},{row.reference_midpoint_ha:.12f},"
                f"{row.predicted_midpoint_ha:.12f},{row.reference_midpoint_ev:.10f},"
                f"{row.predicted_midpoint_ev:.10f},{row.reference_energy_ha:.12f},"
                f"{row.predicted_energy_ha:.12f},{row.reference_energy_ev:.10f},"
                f"{row.predicted_energy_ev:.10f},{row.aligned_reference_energy_ha:.12f},"
                f"{row.aligned_predicted_energy_ha:.12f},{row.aligned_reference_energy_ev:.10f},"
                f"{row.aligned_predicted_energy_ev:.10f},{row.aligned_abs_error_ev:.10f},"
                f"{row.raw_abs_error_ev:.10f}\n"
            )

    train_rows = [row for row in rows if row.split == "train"]
    test_rows = [row for row in rows if row.split == "test"]
    train_aligned_mae = (
        float(np.mean([row.aligned_abs_error_ev for row in train_rows]))
        if train_rows
        else float("nan")
    )
    test_aligned_mae = (
        float(np.mean([row.aligned_abs_error_ev for row in test_rows]))
        if test_rows
        else float("nan")
    )
    train_raw_mae = (
        float(np.mean([row.raw_abs_error_ev for row in train_rows]))
        if train_rows
        else float("nan")
    )
    test_raw_mae = (
        float(np.mean([row.raw_abs_error_ev for row in test_rows]))
        if test_rows
        else float("nan")
    )

    ref_all = np.asarray([row.aligned_reference_energy_ev for row in rows], dtype=float)
    pred_all = np.asarray([row.aligned_predicted_energy_ev for row in rows], dtype=float)
    axis_min = float(min(np.min(ref_all), np.min(pred_all)))
    axis_max = float(max(np.max(ref_all), np.max(pred_all)))
    span = axis_max - axis_min
    margin = 0.04 * (span if span > 1e-10 else 1.0)
    lo = axis_min - margin
    hi = axis_max + margin

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), sharex=True, sharey=True)
    split_specs = (
        ("train", train_rows, axes[0], "#1f77b4", "#ff7f0e"),
        ("test", test_rows, axes[1], "#2a9d8f", "#e76f51"),
    )
    for split_name, split_rows, ax, occ_color, vir_color in split_specs:
        occ_rows = [row for row in split_rows if row.orbital_type == "occupied"]
        vir_rows = [row for row in split_rows if row.orbital_type == "virtual"]
        if occ_rows:
            ax.scatter(
                [row.aligned_reference_energy_ev for row in occ_rows],
                [row.aligned_predicted_energy_ev for row in occ_rows],
                s=26,
                c=occ_color,
                alpha=0.72,
                label="Occupied",
            )
        if vir_rows:
            ax.scatter(
                [row.aligned_reference_energy_ev for row in vir_rows],
                [row.aligned_predicted_energy_ev for row in vir_rows],
                s=30,
                c=vir_color,
                alpha=0.74,
                marker="^",
                label="Virtual",
            )
        ax.plot([lo, hi], [lo, hi], "--", color="#555555", linewidth=1.1, label="y = x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Reference aligned orbital energy (eV)")
        ax.grid(True, alpha=0.28)
        split_aligned_mae = (
            float(np.mean([row.aligned_abs_error_ev for row in split_rows]))
            if split_rows
            else float("nan")
        )
        split_raw_mae = (
            float(np.mean([row.raw_abs_error_ev for row in split_rows]))
            if split_rows
            else float("nan")
        )
        n_molecules = len({row.db_id for row in split_rows})
        ax.set_title(
            f"{split_name.title()} frontier orbitals ({n_molecules} molecules)\n"
            f"Aligned MAE={split_aligned_mae:.3f} eV | Raw MAE={split_raw_mae:.3f} eV"
        )
        ax.legend()
    axes[0].set_ylabel("Predicted aligned orbital energy (eV)")
    fig.suptitle(
        f"Midpoint-Aligned Frontier Orbital Energy Parity ({window} occ incl. HOMO, {window} vir incl. LUMO)",
        fontsize=12.5,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return (
        png_path,
        csv_path,
        train_aligned_mae,
        test_aligned_mae,
        train_raw_mae,
        test_raw_mae,
    )


def _evaluate_orbitals_for_entry(
    *,
    split: str,
    entry: MoleculeEntry,
    neural_scf_molecule: Any,
    scf_info: Any,
    xyzrender_src: str,
    outdir: Path,
    iso: float,
    diff_iso: float,
    mo_blur: float,
    mo_upsample: int,
    cube_grid: int,
    canvas_size: int,
    match_frontier_by_overlap: bool,
    frontier_match_window: int,
) -> list[OrbitalEvalRow]:
    scf_converged = bool(np.asarray(getattr(scf_info, "converged", False)))
    scf_cycles = int(np.asarray(getattr(scf_info, "cycles", -1)))
    scf_final_rms_density = float(
        np.asarray(getattr(scf_info, "final_rms_density", float("nan")))
    )
    try:
        (
            orbital_images,
            orbital_diff_norms,
            orbital_overlaps,
            orbital_diff_isos,
            orbital_diff_scales,
        ) = _render_orbital_surfaces(
            entry=entry,
            split=split,
            neural_molecule=neural_scf_molecule,
            xyzrender_src=xyzrender_src,
            outdir=outdir,
            iso=iso,
            diff_iso=diff_iso,
            mo_blur=mo_blur,
            mo_upsample=mo_upsample,
            cube_grid=cube_grid,
            canvas_size=canvas_size,
            match_frontier_by_overlap=match_frontier_by_overlap,
            frontier_match_window=frontier_match_window,
        )
    except Exception as exc:
        return [
            OrbitalEvalRow(
                split=split,
                db_id=entry.db_id,
                formula=entry.formula,
                orbital=label,
                status="failed",
                overlap=float("nan"),
                diff_norm=float("nan"),
                diff_iso_used=float("nan"),
                diff_scale_used=float("nan"),
                scf_converged=scf_converged,
                scf_cycles=scf_cycles,
                scf_final_rms_density=scf_final_rms_density,
                compare_png="",
                error=str(exc).replace(",", ";"),
            )
            for label in ("HOMO-1", "HOMO", "LUMO", "LUMO+1")
        ]

    compare_dir = outdir / "orbital_surfaces" / f"{split}_{entry.db_id}_{_safe_text(entry.formula)}" / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    rows: list[OrbitalEvalRow] = []
    for label, files in orbital_images.items():
        panel_path = compare_dir / f"{_safe_label(label)}_real_vs_neural.png"
        _plot_orbital_compare(
            orbital_label=label,
            ref_png=files["reference"],
            neural_png=files["neural"],
            diff_png=files["difference"],
            iso=iso,
            diff_iso=orbital_diff_isos[label],
            overlap_val=orbital_overlaps[label],
            diff_norm=orbital_diff_norms[label],
            diff_scale=orbital_diff_scales[label],
            out_png=panel_path,
        )
        rows.append(
            OrbitalEvalRow(
                split=split,
                db_id=entry.db_id,
                formula=entry.formula,
                orbital=label,
                status="ok",
                overlap=orbital_overlaps[label],
                diff_norm=orbital_diff_norms[label],
                diff_iso_used=orbital_diff_isos[label],
                diff_scale_used=orbital_diff_scales[label],
                scf_converged=scf_converged,
                scf_cycles=scf_cycles,
                scf_final_rms_density=scf_final_rms_density,
                compare_png=str(panel_path),
                error="",
            )
        )
    return rows


def _write_entry_xyz(entry: MoleculeEntry, xyz_path: Path) -> None:
    with xyz_path.open("w", encoding="utf-8") as f:
        f.write(f"{entry.z.size}\n")
        f.write(f"QH9 id={entry.db_id} formula={entry.formula}\n")
        for zi, xyz in zip(entry.z, entry.pos_ang, strict=True):
            sym = ATOMIC_SYMBOL[int(zi)]
            f.write(f"{sym:2s} {xyz[0]: .12f} {xyz[1]: .12f} {xyz[2]: .12f}\n")


def _render_structure_png_with_xyzrender(
    entry: MoleculeEntry,
    *,
    outdir: Path,
    prefix: str,
    xyzrender_src: str,
) -> Path | None:
    """Render a molecule structure PNG via xyzrender for spectrum insets."""

    if not Path(xyzrender_src).exists():
        print(f"[QH9] xyzrender source not found: {xyzrender_src}")
        return None

    struct_dir = outdir / "structure_insets"
    struct_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_{entry.db_id}_{_safe_text(entry.formula)}"
    xyz_path = struct_dir / f"{stem}.xyz"
    png_path = struct_dir / f"{stem}.png"
    _write_entry_xyz(entry, xyz_path)

    try:
        _prepare_cairo_runtime()
        if xyzrender_src not in sys.path:
            sys.path.insert(0, xyzrender_src)
        from xyzrender import load, render

        mol = load(str(xyz_path))
        render(
            mol,
            output=str(png_path),
            config="flat",
            hy=True,
            canvas_size=520,
            transparent=True,
        )
    except Exception as exc:  # pragma: no cover - environment-dependent rendering
        print(f"[QH9] xyzrender failed for {entry.db_id}: {exc}")
        return None

    if not png_path.exists():
        return None
    return png_path


def _add_structure_inset(
    ax: Any,
    *,
    image_path: Path | None,
    title: str,
    loc: tuple[float, float, float, float],
) -> None:
    plt = _plt()
    if image_path is None or not image_path.exists():
        return
    img = plt.imread(image_path)
    inset = ax.inset_axes(loc)
    inset.imshow(img)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_facecolor("none")
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#777777")
    inset.text(
        0.03,
        0.98,
        title,
        transform=inset.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="black",
        bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.8},
    )


def _write_split_spectra(
    *,
    grid_ev: jnp.ndarray,
    train_ref_curves: list[jnp.ndarray],
    train_neural_curves: list[jnp.ndarray],
    test_ref_curves: list[jnp.ndarray],
    test_neural_curves: list[jnp.ndarray],
    train_repr: MoleculeEntry | None,
    test_repr: MoleculeEntry | None,
    train_repr_png: Path | None,
    test_repr_png: Path | None,
    reference_label: str,
    train_count: int,
    test_count: int,
    outdir: Path,
) -> None:
    plt = _plt()
    grid_np = np.asarray(grid_ev)
    train_ref_mean = np.mean(np.stack([np.asarray(x) for x in train_ref_curves], axis=0), axis=0)
    train_neural_mean = np.mean(
        np.stack([np.asarray(x) for x in train_neural_curves], axis=0), axis=0
    )
    test_ref_mean = np.mean(np.stack([np.asarray(x) for x in test_ref_curves], axis=0), axis=0)
    test_neural_mean = np.mean(
        np.stack([np.asarray(x) for x in test_neural_curves], axis=0), axis=0
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8), sharey=True)

    axes[0].plot(grid_np, train_ref_mean, lw=2.0, label=reference_label)
    axes[0].plot(grid_np, train_neural_mean, lw=2.0, label="Neural_xc TDDFT")
    axes[0].set_title(f"Train split mean spectrum ({train_count} molecules)")
    axes[0].set_xlabel("Energy (eV)")
    axes[0].set_ylabel("Absorption (a.u.)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(grid_np, test_ref_mean, lw=2.0, label=reference_label)
    axes[1].plot(grid_np, test_neural_mean, lw=2.0, label="Neural_xc TDDFT")
    axes[1].set_title(f"Test split mean spectrum ({test_count} molecules)")
    axes[1].set_xlabel("Energy (eV)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    _add_structure_inset(
        axes[0],
        image_path=train_repr_png,
        title=(
            f"Train repr: {train_repr.formula}\nid={train_repr.db_id}"
            if train_repr is not None
            else "Train representative"
        ),
        loc=(0.68, 0.62, 0.28, 0.32),
    )
    _add_structure_inset(
        axes[1],
        image_path=test_repr_png,
        title=(
            f"Test repr: {test_repr.formula}\nid={test_repr.db_id}"
            if test_repr is not None
            else "Test representative"
        ),
        loc=(0.68, 0.62, 0.28, 0.32),
    )

    fig.tight_layout()
    fig.savefig(outdir / "split_mean_absorption_spectra_raw.png", dpi=180)
    fig.savefig(outdir / "split_mean_absorption_spectra.png", dpi=180)
    plt.close(fig)

    np.savetxt(
        outdir / "split_mean_train_spectrum.csv",
        np.column_stack([grid_np, train_ref_mean, train_neural_mean]),
        delimiter=",",
        header="energy_ev,reference_mean,neural_mean",
        comments="",
    )
    np.savetxt(
        outdir / "split_mean_test_spectrum.csv",
        np.column_stack([grid_np, test_ref_mean, test_neural_mean]),
        delimiter=",",
        header="energy_ev,reference_mean,neural_mean",
        comments="",
    )


def _write_per_molecule_spectrum(
    *,
    split: str,
    entry: MoleculeEntry,
    grid_ev: jnp.ndarray,
    ref_curve: jnp.ndarray,
    neural_curve: jnp.ndarray,
    row: EvalRow,
    reference_label: str,
    image_path: Path | None,
    outdir: Path,
) -> None:
    plt = _plt()
    per_dir = outdir / "per_molecule_spectra"
    per_dir.mkdir(parents=True, exist_ok=True)
    grid_np = np.asarray(grid_ev)
    ref_np = np.asarray(ref_curve)
    neural_np = np.asarray(neural_curve)
    stem = f"{split}_{entry.db_id}_{_safe_text(entry.formula)}"

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(grid_np, ref_np, lw=2.0, label=reference_label)
    ax.plot(grid_np, neural_np, lw=2.0, label="Neural_xc TDDFT")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Absorption (a.u.)")
    ax.set_title(
        f"{split.upper()} | id={entry.db_id} | {entry.formula} | "
        f"MAE={row.excitation_mae_ev:.3f} eV"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    _add_structure_inset(
        ax,
        image_path=image_path,
        title=f"{entry.formula}\nid={entry.db_id}",
        loc=(0.73, 0.60, 0.25, 0.33),
    )
    fig.tight_layout()
    fig.savefig(per_dir / f"{stem}.png", dpi=180)
    plt.close(fig)

    np.savetxt(
        per_dir / f"{stem}.csv",
        np.column_stack([grid_np, ref_np, neural_np]),
        delimiter=",",
        header="energy_ev,reference_curve,neural_curve",
        comments="",
    )


def main() -> None:
    args = parse_args()
    print("[QH9] Entering main()", flush=True)
    if args.train_count < 1:
        raise ValueError("train_count must be >= 1.")
    if args.train_count > args.sample_count:
        raise ValueError("train_count must be <= sample_count.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[QH9] Output directory: {outdir.resolve()}", flush=True)
    semilocal_xc = _normalize_semilocal_arg(args.semilocal_xc)
    coefficient_prior_values = (
        resolve_coefficient_prior_values(semilocal_xc, args.coefficient_prior_values)
        if (
            args.coefficient_prior_values is not None
            or float(args.coefficient_prior_weight) != 0.0
        )
        else None
    )

    t0 = time.perf_counter()
    print(
        f"[QH9] Sampling {args.sample_count} molecules "
        f"({args.train_count} train / {args.sample_count - args.train_count} test)",
        flush=True,
    )
    entries = _sample_entries(args)
    sample_elapsed = time.perf_counter() - t0
    train_entries = entries[: args.train_count]
    test_entries = entries[args.train_count :]
    _write_selected(entries, args.train_count, outdir)

    t1 = time.perf_counter()
    print("[QH9] Starting Neural_xc training", flush=True)
    (
        functional,
        params,
        final_params,
        train_loss_history,
        test_loss_history,
        min_train_loss,
        min_train_loss_step,
        final_train_loss,
        min_test_loss,
        min_test_loss_step,
        final_test_loss,
    ) = _train_functional(
        train_entries,
        test_entries,
        steps=args.steps,
        learning_rate=args.learning_rate,
        lr_schedule_boundaries=args.lr_schedule_boundaries,
        lr_schedule_decay=args.lr_schedule_decay,
        lr_decay_every=args.lr_decay_every,
        lr_decay_factor=args.lr_decay_factor,
        density_weight=args.density_weight,
        stationarity_weight=args.stationarity_weight,
        coefficient_prior_weight=args.coefficient_prior_weight,
        coefficient_prior_values=coefficient_prior_values,
        coefficient_prior_mode=args.coefficient_prior_mode,
        energy_mse_weight=args.energy_mse_weight,
        energy_mae_weight=args.energy_mae_weight,
        energy_normalization=args.energy_normalization,
        density_supervision=args.density_supervision,
        hidden_dims=tuple(args.hidden_dims),
        semilocal_xc=semilocal_xc,
        n_semilocal_channels=args.n_semilocal_channels,
        hf_input_mode=args.hf_input_mode,
        coefficient_positivity=args.coefficient_positivity,
        jit_train=args.jit_train,
        log_interval=args.log_interval,
    )
    train_elapsed = time.perf_counter() - t1
    params_ckpt, params_meta = save_params_checkpoint(
        outdir / "neural_xc_params.msgpack",
        params,
        metadata={
            "sample_count": int(args.sample_count),
            "train_count": int(args.train_count),
            "basis": args.basis,
            "xc": args.xc,
            "semilocal_xc": semilocal_xc,
            "n_semilocal_channels": args.n_semilocal_channels,
            "hf_input_mode": args.hf_input_mode,
            "coefficient_positivity": args.coefficient_positivity,
            "hidden_dims": list(args.hidden_dims),
            "steps": int(args.steps),
            "learning_rate": float(args.learning_rate),
            "lr_schedule_boundaries": args.lr_schedule_boundaries,
            "lr_schedule_decay": float(args.lr_schedule_decay),
            "lr_decay_every": int(args.lr_decay_every),
            "lr_decay_factor": float(args.lr_decay_factor),
            "density_weight": float(args.density_weight),
            "stationarity_weight": float(args.stationarity_weight),
            "coefficient_prior_weight": float(args.coefficient_prior_weight),
            "coefficient_prior_values": coefficient_prior_values,
            "coefficient_prior_mode": args.coefficient_prior_mode,
            "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
            "selected_params": "best_train_loss",
            "min_train_loss": float(min_train_loss),
            "min_train_loss_step": int(min_train_loss_step),
            "final_train_loss": float(final_train_loss),
        },
    )
    final_params_ckpt, final_params_meta = save_params_checkpoint(
        outdir / "neural_xc_params_final.msgpack",
        final_params,
        metadata={
            "sample_count": int(args.sample_count),
            "train_count": int(args.train_count),
            "basis": args.basis,
            "xc": args.xc,
            "semilocal_xc": semilocal_xc,
            "n_semilocal_channels": args.n_semilocal_channels,
            "hf_input_mode": args.hf_input_mode,
            "coefficient_positivity": args.coefficient_positivity,
            "hidden_dims": list(args.hidden_dims),
            "steps": int(args.steps),
            "learning_rate": float(args.learning_rate),
            "lr_schedule_boundaries": args.lr_schedule_boundaries,
            "lr_schedule_decay": float(args.lr_schedule_decay),
            "lr_decay_every": int(args.lr_decay_every),
            "lr_decay_factor": float(args.lr_decay_factor),
            "density_weight": float(args.density_weight),
            "stationarity_weight": float(args.stationarity_weight),
            "coefficient_prior_weight": float(args.coefficient_prior_weight),
            "coefficient_prior_values": coefficient_prior_values,
            "coefficient_prior_mode": args.coefficient_prior_mode,
            "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
            "selected_params": "final",
            "final_train_loss": float(final_train_loss),
        },
    )
    _write_training_curve(train_loss_history, test_loss_history, outdir)

    grid_ev = jnp.linspace(args.grid_min_ev, args.grid_max_ev, args.grid_points)
    rows: list[EvalRow] = []
    orbital_rows: list[OrbitalEvalRow] = []
    orbital_energy_rows: list[OrbitalEnergyEvalRow] = []
    train_ref_curves: list[jnp.ndarray] = []
    train_neural_curves: list[jnp.ndarray] = []
    test_ref_curves: list[jnp.ndarray] = []
    test_neural_curves: list[jnp.ndarray] = []

    for split, split_entries in (("train", train_entries), ("test", test_entries)):
        for entry in split_entries:
            evaluation = _evaluate_entry(
                entry,
                split=split,
                functional=functional,
                params=params,
                density_supervision=args.density_supervision,
                nstates=args.nstates,
                eta_ev=args.eta_ev,
                grid_ev=grid_ev,
                scf_max_cycle=args.orbital_scf_max_cycle,
                scf_damping=args.orbital_scf_damping,
                scf_conv_tol_density=args.orbital_scf_conv_tol_density,
                scf_vxc_clip=args.orbital_scf_vxc_clip,
                scf_iterate_selection=args.orbital_scf_iterate_selection,
            )
            rows.append(evaluation.row)
            struct_png = _render_structure_png_with_xyzrender(
                entry,
                outdir=outdir,
                prefix=f"{split}_mol",
                xyzrender_src=args.xyzrender_src,
            )
            _write_per_molecule_spectrum(
                split=split,
                entry=entry,
                grid_ev=grid_ev,
                ref_curve=evaluation.ref_curve,
                neural_curve=evaluation.neural_curve,
                row=evaluation.row,
                reference_label=f"{args.xc.upper()}/{args.basis.upper()} TDDFT",
                image_path=struct_png,
                outdir=outdir,
            )
            if not args.skip_orbital_compare:
                orbital_rows.extend(
                    _evaluate_orbitals_for_entry(
                        split=split,
                        entry=entry,
                        neural_scf_molecule=evaluation.neural_scf_molecule,
                        scf_info=evaluation.scf_info,
                        xyzrender_src=args.xyzrender_src,
                        outdir=outdir,
                        iso=args.orbital_iso,
                        diff_iso=args.orbital_diff_iso,
                        mo_blur=args.orbital_mo_blur,
                        mo_upsample=args.orbital_mo_upsample,
                        cube_grid=args.orbital_cube_grid,
                        canvas_size=args.orbital_canvas_size,
                        match_frontier_by_overlap=not args.disable_orbital_frontier_match,
                        frontier_match_window=args.orbital_frontier_match_window,
                    )
                )
            if not args.skip_orbital_energy_compare:
                orbital_energy_rows.extend(
                    _orbital_energy_rows_for_entry(
                        split=split,
                        entry=entry,
                        neural_scf_molecule=evaluation.neural_scf_molecule,
                        window=args.orbital_energy_window,
                    )
                )
            if split == "train":
                train_ref_curves.append(evaluation.ref_curve)
                train_neural_curves.append(evaluation.neural_curve)
            else:
                test_ref_curves.append(evaluation.ref_curve)
                test_neural_curves.append(evaluation.neural_curve)

    _write_eval_rows(rows, outdir)
    parity_png, parity_csv = _write_ground_state_parity_plot(rows, outdir)
    if orbital_rows:
        _write_orbital_eval_rows(orbital_rows, outdir)
    orbital_energy_png: Path | None = None
    orbital_energy_csv: Path | None = None
    train_orbital_energy_mae = float("nan")
    test_orbital_energy_mae = float("nan")
    train_orbital_energy_raw_mae = float("nan")
    test_orbital_energy_raw_mae = float("nan")
    if orbital_energy_rows:
        (
            orbital_energy_png,
            orbital_energy_csv,
            train_orbital_energy_mae,
            test_orbital_energy_mae,
            train_orbital_energy_raw_mae,
            test_orbital_energy_raw_mae,
        ) = _write_orbital_energy_parity(
            orbital_energy_rows,
            window=args.orbital_energy_window,
            outdir=outdir,
        )
    train_repr = train_entries[0] if train_entries else None
    test_repr = test_entries[0] if test_entries else None
    train_repr_png = (
        _render_structure_png_with_xyzrender(
            train_repr,
            outdir=outdir,
            prefix="train_repr",
            xyzrender_src=args.xyzrender_src,
        )
        if train_repr is not None
        else None
    )
    test_repr_png = (
        _render_structure_png_with_xyzrender(
            test_repr,
            outdir=outdir,
            prefix="test_repr",
            xyzrender_src=args.xyzrender_src,
        )
        if test_repr is not None
        else None
    )
    if test_ref_curves and test_neural_curves:
        _write_split_spectra(
            grid_ev=grid_ev,
            train_ref_curves=train_ref_curves,
            train_neural_curves=train_neural_curves,
            test_ref_curves=test_ref_curves,
            test_neural_curves=test_neural_curves,
            train_repr=train_repr,
            test_repr=test_repr,
            train_repr_png=train_repr_png,
            test_repr_png=test_repr_png,
            reference_label=f"{args.xc.upper()}/{args.basis.upper()} TDDFT",
            train_count=len(train_entries),
            test_count=len(test_entries),
            outdir=outdir,
        )
    else:
        plt = _plt()
        grid_np = np.asarray(grid_ev)
        train_ref_mean = np.mean(
            np.stack([np.asarray(x) for x in train_ref_curves], axis=0), axis=0
        )
        train_neural_mean = np.mean(
            np.stack([np.asarray(x) for x in train_neural_curves], axis=0), axis=0
        )
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        ax.plot(grid_np, train_ref_mean, lw=2.0, label=f"{args.xc.upper()}/{args.basis.upper()} TDDFT")
        ax.plot(grid_np, train_neural_mean, lw=2.0, label="Neural_xc TDDFT")
        ax.set_title(f"Train split mean spectrum ({len(train_entries)} molecules)")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Absorption (a.u.)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _add_structure_inset(
            ax,
            image_path=train_repr_png,
            title=(
                f"Train repr: {train_repr.formula}\nid={train_repr.db_id}"
                if train_repr is not None
                else "Train representative"
            ),
            loc=(0.68, 0.62, 0.28, 0.32),
        )
        fig.tight_layout()
        fig.savefig(outdir / "split_mean_absorption_spectra_raw.png", dpi=180)
        fig.savefig(outdir / "split_mean_absorption_spectra.png", dpi=180)
        plt.close(fig)
        np.savetxt(
            outdir / "split_mean_train_spectrum.csv",
            np.column_stack([grid_np, train_ref_mean, train_neural_mean]),
            delimiter=",",
            header="energy_ev,reference_mean,neural_mean",
            comments="",
        )
        np.savetxt(
            outdir / "split_mean_test_spectrum.csv",
            np.column_stack([grid_np, np.full_like(grid_np, np.nan), np.full_like(grid_np, np.nan)]),
            delimiter=",",
            header="energy_ev,reference_mean,neural_mean",
            comments="",
        )

    train_rows = [r for r in rows if r.split == "train"]
    test_rows = [r for r in rows if r.split == "test"]
    train_mae = float(np.mean([r.excitation_mae_ev for r in train_rows])) if train_rows else float("nan")
    test_mae = float(np.mean([r.excitation_mae_ev for r in test_rows])) if test_rows else float("nan")
    train_e_mae = float(np.mean([r.energy_abs_error_ha for r in train_rows])) if train_rows else float("nan")
    test_e_mae = float(np.mean([r.energy_abs_error_ha for r in test_rows])) if test_rows else float("nan")
    train_data_for_report = [
        GroundStateDatum(
            molecule=e.reference,
            target_total_energy=jnp.asarray(e.reference.mf_energy),
            density_constraint_weight=args.density_weight,
            stationarity_constraint_weight=args.stationarity_weight,
        )
        for e in train_entries
    ]
    report_training_cfg = GroundStateTrainingConfig(
        mode="fixed_density",
        energy_mse_weight=args.energy_mse_weight,
        energy_mae_weight=args.energy_mae_weight,
        energy_normalization=args.energy_normalization,
        density_supervision=args.density_supervision,
    )
    selected_param_train_loss, _ = ground_state_mse_loss(
        params,
        functional,
        train_data_for_report,
        training_config=report_training_cfg,
    )
    final_param_train_loss, _ = ground_state_mse_loss(
        final_params,
        functional,
        train_data_for_report,
        training_config=report_training_cfg,
    )

    summary_path = outdir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"db_path={args.db_path}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"sample_count={args.sample_count}\n")
        f.write(f"train_count={args.train_count}\n")
        f.write(f"basis={args.basis}\n")
        f.write(f"xc={args.xc}\n")
        f.write(f"steps={args.steps}\n")
        f.write(f"learning_rate={args.learning_rate}\n")
        f.write(f"lr_decay_every={args.lr_decay_every}\n")
        f.write(f"lr_decay_factor={args.lr_decay_factor}\n")
        if args.lr_schedule_boundaries:
            f.write(f"lr_schedule_boundaries={args.lr_schedule_boundaries}\n")
            f.write(f"lr_schedule_decay={args.lr_schedule_decay}\n")
        f.write(f"density_weight={args.density_weight}\n")
        f.write(f"stationarity_weight={args.stationarity_weight}\n")
        f.write(f"energy_mse_weight={args.energy_mse_weight}\n")
        f.write(f"energy_mae_weight={args.energy_mae_weight}\n")
        f.write(f"energy_normalization={args.energy_normalization}\n")
        f.write(f"jit_train={args.jit_train}\n")
        f.write(f"log_interval={args.log_interval}\n")
        f.write(f"jax_enable_x64={bool(jax.config.read('jax_enable_x64'))}\n")
        f.write(f"nstates={args.nstates}\n")
        f.write(f"semilocal_xc={semilocal_xc}\n")
        f.write(f"n_semilocal_channels={args.n_semilocal_channels}\n")
        f.write(f"hf_input_mode={args.hf_input_mode}\n")
        f.write(f"coefficient_positivity={args.coefficient_positivity}\n")
        f.write(f"hidden_dims={tuple(args.hidden_dims)}\n")
        f.write(f"params_ckpt={params_ckpt}\n")
        f.write(f"params_meta={params_meta}\n")
        f.write(f"params_ckpt_final={final_params_ckpt}\n")
        f.write(f"params_meta_final={final_params_meta}\n")
        f.write("selected_params=best_train_loss\n")
        f.write(f"skip_orbital_compare={args.skip_orbital_compare}\n")
        f.write(f"skip_orbital_energy_compare={args.skip_orbital_energy_compare}\n")
        f.write(f"orbital_energy_window={args.orbital_energy_window}\n")
        f.write(f"orbital_frontier_match={not bool(args.disable_orbital_frontier_match)}\n")
        f.write(f"orbital_frontier_match_window={args.orbital_frontier_match_window}\n")
        f.write(f"orbital_rows={len(orbital_rows)}\n")
        f.write(
            f"orbital_success_rows={sum(1 for row in orbital_rows if row.status == 'ok')}\n"
        )
        f.write(f"orbital_energy_rows={len(orbital_energy_rows)}\n")
        f.write(f"sampling_elapsed_s={sample_elapsed:.2f}\n")
        f.write(f"training_elapsed_s={train_elapsed:.2f}\n")
        f.write(f"initial_train_loss={train_loss_history[0]:.12e}\n")
        f.write(f"initial_test_loss={test_loss_history[0]:.12e}\n")
        f.write(f"min_train_loss={min_train_loss:.12e}\n")
        f.write(f"min_train_loss_step={min_train_loss_step}\n")
        f.write(f"final_train_loss={final_train_loss:.12e}\n")
        f.write(f"min_test_loss={min_test_loss:.12e}\n")
        f.write(f"min_test_loss_step={min_test_loss_step}\n")
        f.write(f"final_test_loss={final_test_loss:.12e}\n")
        f.write(f"selected_param_train_loss={float(selected_param_train_loss):.12e}\n")
        f.write(f"final_param_train_loss={float(final_param_train_loss):.12e}\n")
        f.write(f"train_excitation_mae_ev={train_mae:.6f}\n")
        f.write(f"test_excitation_mae_ev={test_mae:.6f}\n")
        f.write(f"train_energy_abs_error_ha={train_e_mae:.6f}\n")
        f.write(f"test_energy_abs_error_ha={test_e_mae:.6f}\n")
        f.write(f"train_orbital_energy_aligned_mae_ev={train_orbital_energy_mae:.6f}\n")
        f.write(f"test_orbital_energy_aligned_mae_ev={test_orbital_energy_mae:.6f}\n")
        f.write(f"train_orbital_energy_raw_mae_ev={train_orbital_energy_raw_mae:.6f}\n")
        f.write(f"test_orbital_energy_raw_mae_ev={test_orbital_energy_raw_mae:.6f}\n")
        f.write(f"train_orbital_energy_mae_ev={train_orbital_energy_mae:.6f}\n")
        f.write(f"test_orbital_energy_mae_ev={test_orbital_energy_mae:.6f}\n")
        f.write(f"ground_state_parity_plot={parity_png}\n")
        f.write(f"ground_state_parity_csv={parity_csv}\n")
        f.write(
            f"orbital_energy_parity_plot={str(orbital_energy_png) if orbital_energy_png is not None else 'N/A'}\n"
        )
        f.write(
            f"orbital_energy_parity_csv={str(orbital_energy_csv) if orbital_energy_csv is not None else 'N/A'}\n"
        )
        f.write(
            f"train_structure_png={str(train_repr_png) if train_repr_png is not None else 'N/A'}\n"
        )
        f.write(
            f"test_structure_png={str(test_repr_png) if test_repr_png is not None else 'N/A'}\n"
        )

    print(f"[QH9] Sampled {args.sample_count} molecules ({args.train_count} train / {args.sample_count - args.train_count} test)")
    print(f"[QH9] Sampling elapsed: {sample_elapsed:.2f}s")
    print(f"[QH9] Training elapsed: {train_elapsed:.2f}s")
    print(
        f"[QH9] Initial train/test loss: "
        f"{train_loss_history[0]:.6e} / {test_loss_history[0]:.6e}"
    )
    print(
        f"[QH9] Min train/test loss: "
        f"{min_train_loss:.6e} (step {min_train_loss_step}) / "
        f"{min_test_loss:.6e} (step {min_test_loss_step})"
    )
    print(
        f"[QH9] Final train/test loss: "
        f"{final_train_loss:.6e} / {final_test_loss:.6e}"
    )
    print("[QH9] Excited-state evaluation uses params from minimum train loss.")
    print(f"[QH9] Train excitation MAE (eV): {train_mae:.6f}")
    print(f"[QH9] Test excitation MAE  (eV): {test_mae:.6f}")
    print(f"[QH9] Train |E| MAE (Ha): {train_e_mae:.6f}")
    print(f"[QH9] Test  |E| MAE (Ha): {test_e_mae:.6f}")
    if orbital_energy_rows:
        print(f"[QH9] Train orbital-energy MAE (eV): {train_orbital_energy_mae:.6f}")
        print(f"[QH9] Test  orbital-energy MAE (eV): {test_orbital_energy_mae:.6f}")
    print(f"[QH9] Outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
