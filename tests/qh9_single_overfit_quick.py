from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from td_graddft.device import put_restricted_molecule_on_device
from td_graddft.xc_backend.jax_libxc import b3lyp_component_basis
from td_graddft.neural_xc import make_neural_xc_functional
from pyscf_reference import restricted_reference_from_pyscf
from td_graddft.scf import DifferentiableSCF, DifferentiableSCFConfig
from td_graddft.scf.rks import _vxc_matrix_from_grid_potential
from td_graddft.spectra import HARTREE_TO_EV, lorentzian_spectrum, oscillator_strengths
from td_graddft.tddft import RestrictedCasidaTDDFT
from td_graddft.training import (
    GroundStateDatum,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    ground_state_mse_loss,
    predict_excitation_energies,
    predict_excitation_spectrum,
    predict_ground_state_total_energy,
    save_params_checkpoint,
)

ATOMIC_SYMBOL = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
}


@dataclass(frozen=True)
class Entry:
    db_id: int
    formula: str
    z: np.ndarray
    pos_ang: np.ndarray
    mol: Any
    mo_energy: np.ndarray
    mo_coeff: np.ndarray
    mo_occ: np.ndarray
    reference: object
    ref_energies_au: jnp.ndarray
    ref_osc: jnp.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quick single-sample QH9 overfit for Neural_xc training + spectrum prediction."
        )
    )
    parser.add_argument("--db-path", default="/Volumes/TF/QH9_db/QH9Stable.db")
    parser.add_argument("--db-id", type=int, default=None, help="specific QH9 id")
    parser.add_argument("--seed", type=int, default=20260326)
    parser.add_argument("--max-atoms", type=int, default=8)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--xc", default="b3lyp")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument(
        "--loss-mode",
        choices=("mixed", "excited_only"),
        default="mixed",
        help="Use the default mixed objective or supervise excited states only.",
    )
    parser.add_argument(
        "--energy-mse-weight",
        type=float,
        default=1.0,
        help="Ground-state energy MSE weight when --loss-mode=mixed.",
    )
    parser.add_argument(
        "--energy-mae-weight",
        type=float,
        default=1.0,
        help="Ground-state energy MAE weight when --loss-mode=mixed.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=5,
        help="print optimization progress every N steps",
    )
    parser.add_argument(
        "--lr-decay-steps",
        type=int,
        default=0,
        help="if >0, use cosine decay schedule over this many steps",
    )
    parser.add_argument(
        "--density-weight",
        type=float,
        default=0.0,
        help=(
            "density matching penalty weight inside ground-state loss; "
            "set >0 to enable density constraint"
        ),
    )
    parser.add_argument(
        "--excited-weight",
        type=float,
        default=0.0,
        help="Weight for multistate excitation-energy supervision.",
    )
    parser.add_argument(
        "--excited-nstates",
        type=int,
        default=3,
        help="Number of lowest excited states supervised during training.",
    )
    parser.add_argument(
        "--excited-use-tda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use TDA for excitation supervision during training.",
    )
    parser.add_argument(
        "--spectrum-weight",
        type=float,
        default=0.0,
        help="Weight for broadened absorption-spectrum supervision.",
    )
    parser.add_argument(
        "--spectrum-nstates",
        type=int,
        default=0,
        help="Number of lowest excited states used to build the training spectrum; 0 means use --nstates.",
    )
    parser.add_argument(
        "--spectrum-use-tda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use TDA for spectrum supervision during training.",
    )
    parser.add_argument(
        "--spectrum-eta-ev",
        type=float,
        default=None,
        help="Lorentzian broadening used inside spectrum supervision; defaults to --eta-ev.",
    )
    parser.add_argument(
        "--orbital-loss-weight",
        type=float,
        default=0.0,
        help=(
            "weight for frontier-orbital energy loss on HOMO-k...LUMO+k "
            "(k set by --orbital-loss-window)"
        ),
    )
    parser.add_argument(
        "--orbital-loss-window",
        type=int,
        default=10,
        help="frontier orbital window size k for HOMO-k...LUMO+k",
    )
    parser.add_argument(
        "--orbital-loss-mode",
        choices=("projected_fock", "self_consistent"),
        default="projected_fock",
        help=(
            "how orbital-window loss is differentiated: "
            "projected_fock is stable; self_consistent backprop can be unstable"
        ),
    )
    parser.add_argument(
        "--orbital-loss-gradient",
        choices=("enabled", "detached"),
        default="detached",
        help=(
            "whether orbital loss participates in backprop; detached avoids NaN "
            "gradients from SCF/eigen backprop on many molecules"
        ),
    )
    parser.add_argument("--orbital-loss-scf-max-cycle", type=int, default=32)
    parser.add_argument("--orbital-loss-scf-damping", type=float, default=0.85)
    parser.add_argument("--orbital-loss-scf-conv-tol-density", type=float, default=1e-5)
    parser.add_argument("--orbital-loss-scf-vxc-clip", type=float, default=20.0)
    parser.add_argument(
        "--orbital-loss-scf-gradient-mode",
        choices=("expl", "impl"),
        default="impl",
    )
    parser.add_argument("--orbital-loss-scf-implicit-max-iter", type=int, default=24)
    parser.add_argument("--orbital-loss-scf-implicit-step-size", type=float, default=0.2)
    parser.add_argument("--orbital-loss-scf-implicit-clip", type=float, default=1e4)
    parser.add_argument(
        "--orbital-loss-scf-iterate-selection",
        choices=("final", "best_rms", "first_converged"),
        default="final",
    )
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[32, 16])
    parser.add_argument(
        "--semilocal-xc",
        nargs="+",
        default=list(b3lyp_component_basis()),
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
        default="spin_resolved",
        help="how projected HF energy density is exposed to the MLP",
    )
    parser.add_argument("--nstates", type=int, default=6)
    parser.add_argument("--eta-ev", type=float, default=0.20)
    parser.add_argument("--grid-min-ev", type=float, default=0.0)
    parser.add_argument("--grid-max-ev", type=float, default=16.0)
    parser.add_argument("--grid-points", type=int, default=1200)
    parser.add_argument(
        "--xyzrender-src",
        default="/Volumes/TF/QH9_db/xyzrender/src",
        help="xyzrender source directory for orbital rendering",
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
    parser.add_argument("--orbital-scf-max-cycle", type=int, default=256)
    parser.add_argument("--orbital-scf-damping", type=float, default=0.96)
    parser.add_argument("--orbital-scf-conv-tol-density", type=float, default=1e-6)
    parser.add_argument("--orbital-scf-vxc-clip", type=float, default=20.0)
    parser.add_argument(
        "--orbital-scf-iterate-selection",
        choices=("final", "best_rms", "first_converged"),
        default="first_converged",
    )
    parser.add_argument(
        "--eval-params",
        choices=("best", "final"),
        default="best",
        help="choose which checkpoint is used for spectrum/orbital evaluation",
    )
    parser.add_argument(
        "--skip-orbital-render",
        action="store_true",
        help="skip HOMO/LUMO cube rendering to shorten post-processing.",
    )
    parser.add_argument("--outdir", default="outputs/qh9_single_overfit_quick")
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


def _build_atom_block(z: np.ndarray, pos_ang: np.ndarray) -> str:
    lines = []
    for zi, xyz in zip(z, pos_ang, strict=True):
        sym = ATOMIC_SYMBOL[int(zi)]
        lines.append(f"{sym} {xyz[0]: .12f} {xyz[1]: .12f} {xyz[2]: .12f}")
    return "\n".join(lines)


def _fetch_molecule(conn: sqlite3.Connection, db_id: int) -> tuple[np.ndarray, np.ndarray]:
    row = conn.execute("SELECT N, Z, pos FROM data WHERE id = ?", (db_id,)).fetchone()
    if row is None:
        raise KeyError(f"QH9 id {db_id} not found")
    n, z_blob, pos_blob = row
    z = np.frombuffer(z_blob, dtype=np.int32).copy()
    pos = np.frombuffer(pos_blob, dtype=np.float64).reshape(int(n), 3).copy()
    return z, pos


def _iter_even_ids(conn: sqlite3.Connection, max_atoms: int) -> list[int]:
    ids: list[int] = []
    cur = conn.execute("SELECT id, Z FROM data WHERE N <= ?", (max_atoms,))
    for db_id, z_blob in cur:
        z = np.frombuffer(z_blob, dtype=np.int32)
        if int(np.sum(z)) % 2 == 0:
            ids.append(int(db_id))
    return ids


def _build_entry(
    db_id: int,
    z: np.ndarray,
    pos_ang: np.ndarray,
    *,
    basis: str,
    xc: str,
    nstates: int,
) -> Entry:
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
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"SCF not converged for id={db_id}")

    td = mf.TDDFT()
    nocc = int(np.count_nonzero(mf.mo_occ > 1e-8))
    nvir = int(mf.mo_coeff.shape[-1] - nocc)
    nstates_eff = min(max(nstates, 1), nocc * nvir)
    td.nstates = nstates_eff
    td.kernel()

    reference = put_restricted_molecule_on_device(restricted_reference_from_pyscf(mf))
    return Entry(
        db_id=db_id,
        formula=_formula_from_z(z),
        z=z,
        pos_ang=pos_ang,
        mol=mol,
        mo_energy=np.asarray(mf.mo_energy, dtype=float),
        mo_coeff=np.asarray(mf.mo_coeff, dtype=float),
        mo_occ=np.asarray(mf.mo_occ, dtype=float),
        reference=reference,
        ref_energies_au=jnp.asarray(td.e),
        ref_osc=jnp.asarray(td.oscillator_strength()),
    )


def _select_entry(args: argparse.Namespace) -> Entry:
    conn = sqlite3.connect(args.db_path)
    rng = np.random.default_rng(args.seed)
    try:
        if args.db_id is not None:
            z, pos = _fetch_molecule(conn, args.db_id)
            return _build_entry(
                args.db_id,
                z,
                pos,
                basis=args.basis,
                xc=args.xc,
                nstates=args.nstates,
            )

        ids = _iter_even_ids(conn, args.max_atoms)
        if not ids:
            raise RuntimeError("No small even-electron molecules found in DB.")
        rng.shuffle(ids)
        last_error: Exception | None = None
        for db_id in ids:
            z, pos = _fetch_molecule(conn, db_id)
            try:
                return _build_entry(
                    db_id,
                    z,
                    pos,
                    basis=args.basis,
                    xc=args.xc,
                    nstates=args.nstates,
                )
            except Exception as exc:  # pragma: no cover - depends on runtime molecule
                last_error = exc
                continue
        if last_error is not None:
            raise RuntimeError(f"Failed to build any candidate molecule: {last_error}")
        raise RuntimeError("Failed to build any candidate molecule.")
    finally:
        conn.close()


def _write_training_curve(
    path: Path,
    total_history: list[float],
    ground_history: list[float],
    excitation_penalty_history: list[float],
    spectrum_penalty_history: list[float],
    orbital_mse_history: list[float],
    orbital_weighted_history: list[float],
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(
            "step,total_loss,ground_loss,excitation_penalty,spectrum_penalty,"
            "orbital_mse,orbital_weighted\n"
        )
        for step in range(len(total_history)):
            f.write(
                f"{step},"
                f"{total_history[step]:.16e},"
                f"{ground_history[step]:.16e},"
                f"{excitation_penalty_history[step]:.16e},"
                f"{spectrum_penalty_history[step]:.16e},"
                f"{orbital_mse_history[step]:.16e},"
                f"{orbital_weighted_history[step]:.16e}\n"
            )


def _plot_training_curve(
    path: Path,
    total_history: list[float],
    ground_history: list[float],
    excitation_penalty_history: list[float],
    spectrum_penalty_history: list[float],
    orbital_weighted_history: list[float],
    *,
    excited_weight: float,
    spectrum_weight: float,
    orbital_loss_weight: float,
) -> None:
    total_values = np.maximum(np.asarray(total_history, dtype=float), 1e-16)
    ground_values = np.maximum(np.asarray(ground_history, dtype=float), 1e-16)
    excitation_values = np.maximum(np.asarray(excitation_penalty_history, dtype=float), 1e-16)
    spectrum_values = np.maximum(np.asarray(spectrum_penalty_history, dtype=float), 1e-16)
    orbital_weighted_values = np.maximum(np.asarray(orbital_weighted_history, dtype=float), 1e-16)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    x = np.arange(total_values.size)
    ax.plot(x, total_values, lw=2.2, label="total")
    ax.plot(x, ground_values, lw=1.8, alpha=0.95, label="ground")
    if excited_weight != 0.0:
        ax.plot(x, excitation_values, lw=1.6, alpha=0.9, label="excitation")
    if spectrum_weight != 0.0:
        ax.plot(x, spectrum_values, lw=1.6, alpha=0.9, label="spectrum")
    if orbital_loss_weight != 0.0:
        ax.plot(x, orbital_weighted_values, lw=1.6, alpha=0.9, label="orbital(weighted)")
    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss (log scale)")
    ax.set_title("QH9 single-sample overfit loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_spectrum_csv(
    path: Path,
    grid_ev: jnp.ndarray,
    ref_curve: jnp.ndarray,
    neural_curve: jnp.ndarray,
) -> None:
    np.savetxt(
        path,
        np.column_stack([np.asarray(grid_ev), np.asarray(ref_curve), np.asarray(neural_curve)]),
        delimiter=",",
        header="energy_ev,reference_curve,neural_curve",
        comments="",
    )


def _plot_spectrum(
    path: Path,
    *,
    entry: Entry,
    grid_ev: jnp.ndarray,
    ref_curve: jnp.ndarray,
    neural_curve: jnp.ndarray,
    ref_label: str,
    excitation_mae_ev: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(np.asarray(grid_ev), np.asarray(ref_curve), lw=2.0, label=ref_label)
    ax.plot(np.asarray(grid_ev), np.asarray(neural_curve), lw=2.0, label="Neural_xc TDDFT")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Absorption (a.u.)")
    ax.set_title(
        f"QH9 single overfit | id={entry.db_id} | {entry.formula} | "
        f"MAE={excitation_mae_ev:.3f} eV"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


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


def _restricted_mo_occ_vector(mo_occ: Any) -> np.ndarray:
    occ = np.asarray(mo_occ, dtype=float)
    if occ.ndim == 1:
        return occ
    if occ.ndim == 2:
        return np.mean(occ, axis=0)
    raise ValueError("Expected mo_occ to have shape (nmo,) or (spin, nmo).")


def _restricted_mo_energy_vector(mo_energy: Any) -> jnp.ndarray:
    energy = jnp.asarray(mo_energy)
    if energy.ndim == 1:
        return energy
    if energy.ndim == 2:
        return jnp.mean(energy, axis=0)
    raise ValueError("Expected mo_energy to have shape (nmo,) or (spin, nmo).")


def _frontier_orbital_window_from_occ(
    mo_occ: np.ndarray,
    *,
    window: int,
) -> tuple[int, int, int, int]:
    occ = np.where(mo_occ > 1e-8)[0]
    vir = np.where(mo_occ <= 1e-8)[0]
    if occ.size == 0:
        raise RuntimeError("No occupied orbitals found for orbital-loss window.")
    if vir.size == 0:
        raise RuntimeError("No virtual orbitals found for orbital-loss window.")

    homo = int(occ[-1])
    lumo = int(vir[0])
    nmo = int(mo_occ.shape[0])
    k = max(int(window), 0)
    start = max(0, homo - k)
    stop = min(nmo, lumo + k + 1)
    if stop <= start:
        stop = min(nmo, start + 1)
    return start, stop, homo, lumo


def _spin_summed_density_matrix(molecule: Any) -> jnp.ndarray:
    density = jnp.asarray(molecule.rdm1)
    if density.ndim == 3:
        return density.sum(axis=0)
    return density


def _restricted_mo_coeff_matrix(mo_coeff: Any) -> jnp.ndarray:
    coeff = jnp.asarray(mo_coeff)
    if coeff.ndim == 2:
        return coeff
    if coeff.ndim == 3:
        if coeff.shape[0] == 1:
            return coeff[0]
        if coeff.shape[0] == 2:
            return coeff[0]
    raise ValueError("Expected mo_coeff to have shape (nao,nmo) or (spin,nao,nmo).")


def _resolved_xc_object(params: Any, functional: Any, molecule: Any) -> Any:
    binder = getattr(functional, "bind_to_molecule", None)
    if binder is not None:
        return binder(params, molecule)
    binder = getattr(functional, "bind", None)
    if binder is not None:
        return binder(params)
    return functional


def _grid_xc_potential_from_resolved(
    resolved: Any,
    *,
    functional: Any,
    params: Any,
    molecule: Any,
) -> jnp.ndarray:
    grid_potential = getattr(resolved, "grid_potential", None)
    if grid_potential is not None:
        return jnp.asarray(grid_potential(molecule))

    local_potential = getattr(resolved, "local_potential", None)
    if local_potential is not None:
        density = jnp.asarray(molecule.density()).sum(axis=-1)
        return jnp.asarray(local_potential(density))

    functional_local_potential = getattr(functional, "local_potential", None)
    if functional_local_potential is None:
        raise AttributeError(
            "The XC functional must expose local_potential(...) or grid_potential(...)."
        )
    density = jnp.asarray(molecule.density()).sum(axis=-1)
    return jnp.asarray(functional_local_potential(params, density))


def _grid_xc_potential_components_from_resolved(
    resolved: Any,
    *,
    functional: Any,
    params: Any,
    molecule: Any,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    component_getter = getattr(resolved, "grid_potential_components", None)
    if callable(component_getter):
        components = component_getter(molecule)
        if len(components) >= 2:
            v_rho = jnp.asarray(components[0])
            v_grad = jnp.asarray(components[1], dtype=v_rho.dtype)
            if v_grad.ndim == 2 and v_grad.shape == (3, v_rho.shape[0]):
                v_grad = v_grad.T
            if v_grad.ndim != 2 or v_grad.shape[0] != v_rho.shape[0] or v_grad.shape[1] != 3:
                raise ValueError("grid_potential_components must provide v_grad with shape (ngrids, 3).")
            return v_rho, v_grad

    v_rho = _grid_xc_potential_from_resolved(
        resolved,
        functional=functional,
        params=params,
        molecule=molecule,
    )
    return jnp.asarray(v_rho), jnp.zeros(jnp.asarray(v_rho).shape + (3,), dtype=jnp.asarray(v_rho).dtype)


def _effective_exact_exchange_fraction_from_resolved(resolved: Any) -> jnp.ndarray:
    alpha = getattr(resolved, "exact_exchange_fraction", 0.0)
    return jnp.asarray(alpha)


def _projected_orbital_window_mse(
    *,
    params: Any,
    functional: Any,
    molecule: Any,
    reference_mo_energy_window: jnp.ndarray,
    window_start: int,
    window_stop: int,
    vxc_clip: float,
) -> jnp.ndarray:
    density = _spin_summed_density_matrix(molecule)
    h1e = jnp.asarray(molecule.h1e)
    rep_tensor = jnp.asarray(molecule.rep_tensor)
    ao = jnp.asarray(molecule.ao)
    weights = jnp.asarray(molecule.grid.weights)
    mo_coeff = _restricted_mo_coeff_matrix(molecule.mo_coeff)

    resolved_xc = _resolved_xc_object(params, functional, molecule)
    v_rho, v_grad = _grid_xc_potential_components_from_resolved(
        resolved_xc,
        functional=functional,
        params=params,
        molecule=molecule,
    )
    v_rho = jnp.nan_to_num(v_rho, nan=0.0, posinf=vxc_clip, neginf=-vxc_clip)
    v_grad = jnp.nan_to_num(v_grad, nan=0.0, posinf=vxc_clip, neginf=-vxc_clip)
    v_rho = jnp.clip(v_rho, -vxc_clip, vxc_clip)
    v_grad = jnp.clip(v_grad, -vxc_clip, vxc_clip)

    j_mat = jnp.einsum("pqrs,rs->pq", rep_tensor, density)
    k_mat = jnp.einsum("prqs,rs->pq", rep_tensor, density)
    ao_deriv1 = getattr(molecule, "ao_deriv1", None)
    if ao_deriv1 is None:
        ao_deriv1 = jnp.zeros((4, ao.shape[0], ao.shape[1]), dtype=ao.dtype)
        xc_kind = "LDA"
    else:
        ao_deriv1 = jnp.asarray(ao_deriv1)
        xc_kind = "MGGA"
    vxc_matrix = _vxc_matrix_from_grid_potential(
        ao=ao,
        ao_deriv1=ao_deriv1,
        weights=weights,
        vxc_rho=v_rho,
        vxc_grad=v_grad,
        xc_kind=xc_kind,
    )
    alpha = _effective_exact_exchange_fraction_from_resolved(resolved_xc)
    alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
    alpha = jnp.clip(alpha, 0.0, 1.0)

    fock = h1e + j_mat - 0.5 * alpha * k_mat + vxc_matrix
    fock = 0.5 * (fock + fock.T)
    fock = jnp.nan_to_num(fock, nan=0.0, posinf=0.0, neginf=0.0)

    # Use diagonal projections in the reference MO basis to avoid unstable
    # gradients through eigendecomposition during training.
    projected_mo_energy = jnp.einsum("pi,pq,qi->i", mo_coeff, fock, mo_coeff)
    pred_window = projected_mo_energy[window_start:window_stop]
    mse = jnp.mean((pred_window - reference_mo_energy_window) ** 2)
    return jnp.nan_to_num(mse, nan=0.0, posinf=1e6, neginf=1e6)


def _safe_label(label: str) -> str:
    return label.replace("+", "p").replace("-", "m").replace(" ", "_").lower()


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


def _prepare_cairo_runtime() -> None:
    homebrew_lib = "/opt/homebrew/lib"
    current = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    parts = [p for p in current.split(":") if p]
    if homebrew_lib not in parts:
        parts.insert(0, homebrew_lib)
    if "/usr/lib" not in parts:
        parts.append("/usr/lib")
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(parts)


def _render_orbital_surfaces(
    *,
    entry: Entry,
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

    orbital_dir = outdir / "orbital_surfaces"
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

    ref_coeff, ref_occ = _restricted_channel(entry.mo_coeff, entry.mo_occ)
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

        stem = _safe_label(label)
        cube_ref = cube_ref_dir / f"{stem}.cube"
        cube_neural = cube_neural_dir / f"{stem}.cube"
        cube_diff = cube_diff_dir / f"{stem}.cube"
        png_ref = png_ref_dir / f"{stem}.png"
        png_neural = png_neural_dir / f"{stem}.png"
        png_diff = png_diff_dir / f"{stem}.png"

        cubegen.orbital(
            entry.mol,
            str(cube_ref),
            ref_vec,
            nx=cube_grid,
            ny=cube_grid,
            nz=cube_grid,
        )
        cubegen.orbital(
            entry.mol,
            str(cube_neural),
            neural_vec,
            nx=cube_grid,
            ny=cube_grid,
            nz=cube_grid,
        )
        cubegen.orbital(
            entry.mol,
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
                entry.mol,
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


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    semilocal_xc = _normalize_semilocal_arg(args.semilocal_xc)

    t0 = time.perf_counter()
    entry = _select_entry(args)
    sample_elapsed = time.perf_counter() - t0

    loss_mode = str(args.loss_mode)
    energy_mse_weight = float(args.energy_mse_weight)
    energy_mae_weight = float(args.energy_mae_weight)
    density_weight = float(args.density_weight)
    spectrum_weight = float(args.spectrum_weight)
    spectrum_eta_ev = (
        float(args.eta_ev) if args.spectrum_eta_ev is None else float(args.spectrum_eta_ev)
    )
    if loss_mode == "excited_only":
        energy_mse_weight = 0.0
        energy_mae_weight = 0.0
        density_weight = 0.0
        if float(args.excited_weight) == 0.0 and spectrum_weight == 0.0:
            raise ValueError(
                "excited_only loss requires --excited-weight > 0 or --spectrum-weight > 0."
            )
    excitation_targets = None
    if float(args.excited_weight) != 0.0:
        n_take = min(max(1, int(args.excited_nstates)), int(entry.ref_energies_au.size))
        excitation_targets = entry.ref_energies_au[:n_take]
    spectrum_targets_grid_ev = None
    spectrum_targets_curve = None
    requested_spectrum_nstates = 0
    if spectrum_weight != 0.0:
        requested_spectrum_nstates = int(args.spectrum_nstates)
        if requested_spectrum_nstates <= 0:
            requested_spectrum_nstates = int(args.nstates)
        requested_spectrum_nstates = min(
            max(1, requested_spectrum_nstates),
            int(entry.ref_energies_au.size),
            int(entry.ref_osc.size),
        )
        spectrum_targets_grid_ev = jnp.linspace(
            float(args.grid_min_ev),
            float(args.grid_max_ev),
            int(args.grid_points),
        )
        spectrum_targets_curve = lorentzian_spectrum(
            entry.ref_energies_au[:requested_spectrum_nstates] * HARTREE_TO_EV,
            entry.ref_osc[:requested_spectrum_nstates],
            spectrum_targets_grid_ev,
            eta=spectrum_eta_ev,
        )

    datum = GroundStateDatum(
        molecule=entry.reference,
        target_total_energy=jnp.asarray(entry.reference.mf_energy),
        target_excitation_energies=excitation_targets,
        target_spectrum_grid_ev=spectrum_targets_grid_ev,
        target_spectrum_curve=spectrum_targets_curve,
        excitation_constraint_weight=float(args.excited_weight),
        excitation_constraint_nstates=max(1, int(args.excited_nstates)),
        spectrum_constraint_weight=spectrum_weight,
        spectrum_constraint_nstates=(
            requested_spectrum_nstates if spectrum_weight != 0.0 else None
        ),
        density_constraint_weight=density_weight,
    )
    functional = make_neural_xc_functional(
        semilocal_xc=semilocal_xc,
        n_semilocal_channels=args.n_semilocal_channels,
        hf_input_mode=args.hf_input_mode,
        hidden_dims=tuple(args.hidden_dims),
        name="qh9_single_overfit_quick_neural_xc",
    )

    if args.lr_decay_steps > 0:
        lr_schedule = optax.cosine_decay_schedule(
            init_value=float(args.learning_rate),
            decay_steps=int(args.lr_decay_steps),
            alpha=0.1,
        )
        tx = optax.adam(lr_schedule)
    else:
        tx = optax.adam(args.learning_rate)

    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(args.seed),
        entry.reference,
        tx,
    )

    ref_mo_occ = _restricted_mo_occ_vector(entry.mo_occ)
    window_start, window_stop, window_homo, window_lumo = _frontier_orbital_window_from_occ(
        ref_mo_occ,
        window=args.orbital_loss_window,
    )
    ref_mo_energy_window = _restricted_mo_energy_vector(entry.mo_energy)[window_start:window_stop]
    orbital_loss_weight = float(args.orbital_loss_weight)
    use_orbital_loss = orbital_loss_weight != 0.0
    use_orbital_self_consistent = use_orbital_loss and args.orbital_loss_mode == "self_consistent"
    if use_orbital_self_consistent:
        orbital_loss_scf = DifferentiableSCF(
            DifferentiableSCFConfig(
                mode="self_consistent",
                gradient_mode=args.orbital_loss_scf_gradient_mode,
                max_cycle=args.orbital_loss_scf_max_cycle,
                damping=args.orbital_loss_scf_damping,
                conv_tol_density=args.orbital_loss_scf_conv_tol_density,
                vxc_clip=args.orbital_loss_scf_vxc_clip,
                iterate_selection=args.orbital_loss_scf_iterate_selection,
                implicit_diff_max_iter=args.orbital_loss_scf_implicit_max_iter,
                implicit_diff_step_size=args.orbital_loss_scf_implicit_step_size,
                implicit_diff_clip=args.orbital_loss_scf_implicit_clip,
            )
        )
    else:
        orbital_loss_scf = None

    train_cfg = GroundStateTrainingConfig(
        energy_mse_weight=energy_mse_weight,
        energy_mae_weight=energy_mae_weight,
        energy_normalization="none",
        excitation_constraint_use_tda=bool(args.excited_use_tda),
        spectrum_constraint_use_tda=bool(args.spectrum_use_tda),
        spectrum_constraint_eta_ev=spectrum_eta_ev,
    )

    def _compute_total_loss(
        params: Any,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        ground_loss, ground_metrics = ground_state_mse_loss(
            params,
            functional,
            datum,
            training_config=train_cfg,
        )
        orbital_mse = jnp.asarray(0.0, dtype=ground_loss.dtype)
        if use_orbital_loss:
            if args.orbital_loss_mode == "projected_fock":
                orbital_mse = _projected_orbital_window_mse(
                    params=params,
                    functional=functional,
                    molecule=entry.reference,
                    reference_mo_energy_window=ref_mo_energy_window,
                    window_start=window_start,
                    window_stop=window_stop,
                    vxc_clip=float(args.orbital_loss_scf_vxc_clip),
                )
            else:
                assert orbital_loss_scf is not None
                scf_mol, _ = orbital_loss_scf.run(entry.reference, functional, params)
                pred_mo_energy = _restricted_mo_energy_vector(scf_mol.mo_energy)
                pred_window = pred_mo_energy[window_start:window_stop]
                orbital_mse = jnp.mean((pred_window - ref_mo_energy_window) ** 2)
                orbital_mse = jnp.nan_to_num(orbital_mse, nan=0.0, posinf=1e6, neginf=1e6)
        orbital_grad_value = (
            orbital_mse if args.orbital_loss_gradient == "enabled" else jax.lax.stop_gradient(orbital_mse)
        )
        orbital_weighted = jnp.asarray(orbital_loss_weight, dtype=ground_loss.dtype) * orbital_grad_value
        total_loss = ground_loss + orbital_weighted
        metrics = dict(ground_metrics)
        metrics["ground_loss"] = ground_loss
        metrics["orbital_mse_loss"] = orbital_mse
        metrics["orbital_weighted_loss"] = orbital_weighted
        metrics["excitation_penalty_value"] = jnp.asarray(
            ground_metrics.get("excitation_penalty", jnp.asarray([0.0]))[0],
            dtype=ground_loss.dtype,
        )
        metrics["spectrum_penalty_value"] = jnp.asarray(
            ground_metrics.get("spectrum_penalty", jnp.asarray([0.0]))[0],
            dtype=ground_loss.dtype,
        )
        metrics["total_loss"] = total_loss
        return total_loss, metrics

    initial_loss, initial_metrics = _compute_total_loss(state.params)
    loss_history = [float(initial_loss)]
    ground_history = [float(initial_metrics["ground_loss"])]
    excitation_penalty_history = [float(initial_metrics["excitation_penalty_value"])]
    spectrum_penalty_history = [float(initial_metrics["spectrum_penalty_value"])]
    orbital_mse_history = [float(initial_metrics["orbital_mse_loss"])]
    orbital_weighted_history = [float(initial_metrics["orbital_weighted_loss"])]
    min_loss = float(initial_loss)
    min_step = 0
    best_params = state.params
    nan_grad_fallback_steps = 0

    t1 = time.perf_counter()
    for step in range(1, args.steps + 1):
        params_before_update = state.params
        (train_loss, train_metrics), grads = jax.value_and_grad(_compute_total_loss, has_aux=True)(
            params_before_update
        )
        grad_leaves = jax.tree_util.tree_leaves(grads)
        grad_has_nan = any(bool(jnp.isnan(g).any()) for g in grad_leaves)
        grad_has_inf = any(bool(jnp.isinf(g).any()) for g in grad_leaves)
        if (
            (grad_has_nan or grad_has_inf)
            and use_orbital_loss
            and args.orbital_loss_gradient == "enabled"
        ):
            # SCF/eigen backprop may be unstable on some systems. Fall back to
            # stable ground-state gradients for this step and keep orbital loss
            # as a monitored metric.
            _, grads_ground = jax.value_and_grad(
                lambda p: ground_state_mse_loss(p, functional, datum)[0]
            )(params_before_update)
            grads = grads_ground
            nan_grad_fallback_steps += 1
        grads = jax.tree_util.tree_map(
            lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0),
            grads,
        )
        train_loss_val = float(train_loss)
        # Skip step=1 append to avoid duplicating the initial point (same parameters).
        if step > 1:
            loss_history.append(train_loss_val)
            ground_history.append(float(train_metrics["ground_loss"]))
            excitation_penalty_history.append(float(train_metrics["excitation_penalty_value"]))
            spectrum_penalty_history.append(float(train_metrics["spectrum_penalty_value"]))
            orbital_mse_history.append(float(train_metrics["orbital_mse_loss"]))
            orbital_weighted_history.append(float(train_metrics["orbital_weighted_loss"]))
        state = state.apply_gradients(grads=grads)

        # The evaluated params are from before update, corresponding to state index (step-1).
        if step > 1 and train_loss_val < min_loss:
            min_loss = train_loss_val
            min_step = step - 1
            best_params = params_before_update

        if args.log_every > 0 and (step == 1 or step == args.steps or step % args.log_every == 0):
            print(
                f"[QuickQH9][step {step}/{args.steps}] "
                f"total={train_loss_val:.6e} "
                f"ground={float(train_metrics['ground_loss']):.6e} "
                f"excited={float(train_metrics['excitation_penalty_value']):.6e} "
                f"spectrum={float(train_metrics['spectrum_penalty_value']):.6e} "
                f"orb_mse={float(train_metrics['orbital_mse_loss']):.6e} "
                f"orb_weighted={float(train_metrics['orbital_weighted_loss']):.6e} "
                f"grad_nan={grad_has_nan} grad_inf={grad_has_inf}"
            )
    train_elapsed = time.perf_counter() - t1

    final_loss, final_metrics = _compute_total_loss(state.params)
    final_loss = float(final_loss)
    final_ground_loss = float(final_metrics["ground_loss"])
    final_excitation_penalty = float(final_metrics["excitation_penalty_value"])
    final_spectrum_penalty = float(final_metrics["spectrum_penalty_value"])
    final_orbital_mse = float(final_metrics["orbital_mse_loss"])
    final_orbital_weighted = float(final_metrics["orbital_weighted_loss"])
    loss_history.append(final_loss)
    ground_history.append(final_ground_loss)
    excitation_penalty_history.append(final_excitation_penalty)
    spectrum_penalty_history.append(final_spectrum_penalty)
    orbital_mse_history.append(final_orbital_mse)
    orbital_weighted_history.append(final_orbital_weighted)

    if final_loss < min_loss:
        min_loss = final_loss
        min_step = args.steps
        best_params = state.params

    best_loss, best_metrics = _compute_total_loss(best_params)
    best_loss = float(best_loss)
    best_ground_loss = float(best_metrics["ground_loss"])
    best_excitation_penalty = float(best_metrics["excitation_penalty_value"])
    best_spectrum_penalty = float(best_metrics["spectrum_penalty_value"])
    best_orbital_mse = float(best_metrics["orbital_mse_loss"])
    best_orbital_weighted = float(best_metrics["orbital_weighted_loss"])

    eval_params = best_params if args.eval_params == "best" else state.params

    train_path_nstates = max(
        1,
        requested_spectrum_nstates if spectrum_weight != 0.0 else int(args.excited_nstates),
    )
    train_path_nstates = min(train_path_nstates, int(entry.ref_energies_au.size))
    train_path_energies_au = predict_excitation_energies(
        eval_params,
        functional,
        entry.reference,
        nstates=train_path_nstates,
        use_tda=bool(args.excited_use_tda),
    )
    train_path_compare = int(min(train_path_nstates, entry.ref_energies_au.size, train_path_energies_au.size))
    train_path_excitation_mae_ev = float(
        jnp.mean(
            jnp.abs(
                entry.ref_energies_au[:train_path_compare] * HARTREE_TO_EV
                - train_path_energies_au[:train_path_compare] * HARTREE_TO_EV
            )
        )
    )
    train_path_spectrum_mse = float("nan")
    if spectrum_targets_grid_ev is not None and spectrum_targets_curve is not None:
        train_path_curve = predict_excitation_spectrum(
            eval_params,
            functional,
            entry.reference,
            grid_ev=spectrum_targets_grid_ev,
            nstates=requested_spectrum_nstates,
            use_tda=bool(args.spectrum_use_tda),
            eta_ev=spectrum_eta_ev,
        )
        target_rms = jnp.sqrt(jnp.mean(jnp.asarray(spectrum_targets_curve) ** 2))
        target_rms = jnp.maximum(target_rms, 1e-8)
        train_path_spectrum_mse = float(
            jnp.mean(((train_path_curve - jnp.asarray(spectrum_targets_curve)) / target_rms) ** 2)
        )

    eval_training_config = GroundStateTrainingConfig(
        mode="self_consistent",
        scf_max_cycle=args.orbital_scf_max_cycle,
        scf_damping=args.orbital_scf_damping,
        scf_conv_tol_density=args.orbital_scf_conv_tol_density,
        scf_vxc_clip=args.orbital_scf_vxc_clip,
        scf_iterate_selection=args.orbital_scf_iterate_selection,
    )
    scf = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            max_cycle=args.orbital_scf_max_cycle,
            damping=args.orbital_scf_damping,
            conv_tol_density=args.orbital_scf_conv_tol_density,
            vxc_clip=args.orbital_scf_vxc_clip,
            iterate_selection=args.orbital_scf_iterate_selection,
        )
    )
    neural_scf_molecule, scf_info = scf.run(entry.reference, functional, eval_params)

    predicted_total = float(
        predict_ground_state_total_energy(
            eval_params,
            functional,
            entry.reference,
            training_config=eval_training_config,
        )
    )
    target_total = float(entry.reference.mf_energy)
    total_energy_abs_err = abs(predicted_total - target_total)

    solver = RestrictedCasidaTDDFT(
        molecule=neural_scf_molecule,
        xc_functional=functional,
        xc_params=eval_params,
    )
    try:
        result = solver.kernel(nstates=args.nstates)
        solver_label = "Casida"
    except Exception:
        result = solver.tda(nstates=args.nstates)
        solver_label = "TDA fallback"
    if result.excitation_energies.size == 0:
        result = solver.tda(nstates=args.nstates)
        solver_label = "TDA fallback"

    neural_energies_au = result.excitation_energies
    neural_osc = oscillator_strengths(neural_scf_molecule, result)
    ncompare = int(min(entry.ref_energies_au.size, neural_energies_au.size, args.nstates))
    self_consistent_excitation_mae_ev = float(
        jnp.mean(
            jnp.abs(
                entry.ref_energies_au[:ncompare] * HARTREE_TO_EV
                - neural_energies_au[:ncompare] * HARTREE_TO_EV
            )
        )
    )

    grid_ev = jnp.linspace(args.grid_min_ev, args.grid_max_ev, args.grid_points)
    ref_curve = lorentzian_spectrum(
        entry.ref_energies_au * HARTREE_TO_EV,
        entry.ref_osc,
        grid_ev,
        eta=args.eta_ev,
    )
    neural_curve = lorentzian_spectrum(
        neural_energies_au * HARTREE_TO_EV,
        neural_osc,
        grid_ev,
        eta=args.eta_ev,
    )

    loss_csv = outdir / "training_loss.csv"
    loss_png = outdir / "training_loss.png"
    spectrum_csv = outdir / "spectrum_compare.csv"
    spectrum_png = outdir / "spectrum_compare.png"
    summary_path = outdir / "summary.txt"
    ckpt_path, ckpt_meta = save_params_checkpoint(
        outdir / "neural_xc_params.msgpack",
        eval_params,
        metadata={
            "db_id": int(entry.db_id),
            "formula": entry.formula,
            "basis": args.basis,
            "xc": args.xc,
            "semilocal_xc": semilocal_xc,
            "n_semilocal_channels": args.n_semilocal_channels,
            "hf_input_mode": args.hf_input_mode,
            "hidden_dims": list(args.hidden_dims),
            "steps": int(args.steps),
            "learning_rate": float(args.learning_rate),
            "loss_mode": loss_mode,
            "energy_mse_weight": energy_mse_weight,
            "energy_mae_weight": energy_mae_weight,
            "log_every": int(args.log_every),
            "lr_decay_steps": int(args.lr_decay_steps),
            "density_weight": density_weight,
            "excited_weight": float(args.excited_weight),
            "excited_nstates": int(args.excited_nstates),
            "excited_use_tda": bool(args.excited_use_tda),
            "spectrum_weight": spectrum_weight,
            "spectrum_nstates": int(requested_spectrum_nstates),
            "spectrum_use_tda": bool(args.spectrum_use_tda),
            "spectrum_eta_ev": spectrum_eta_ev,
            "orbital_loss_weight": float(args.orbital_loss_weight),
            "orbital_loss_window": int(args.orbital_loss_window),
            "orbital_loss_mode": args.orbital_loss_mode,
            "orbital_loss_gradient": args.orbital_loss_gradient,
            "orbital_loss_scf_max_cycle": int(args.orbital_loss_scf_max_cycle),
            "orbital_loss_scf_damping": float(args.orbital_loss_scf_damping),
            "orbital_loss_scf_conv_tol_density": float(args.orbital_loss_scf_conv_tol_density),
            "orbital_loss_scf_vxc_clip": float(args.orbital_loss_scf_vxc_clip),
            "orbital_loss_scf_gradient_mode": args.orbital_loss_scf_gradient_mode,
            "orbital_loss_scf_implicit_max_iter": int(args.orbital_loss_scf_implicit_max_iter),
            "orbital_loss_scf_implicit_step_size": float(args.orbital_loss_scf_implicit_step_size),
            "orbital_loss_scf_implicit_clip": float(args.orbital_loss_scf_implicit_clip),
            "orbital_loss_scf_iterate_selection": args.orbital_loss_scf_iterate_selection,
            "seed": int(args.seed),
            "eval_params": args.eval_params,
            "best_step": int(min_step),
            "best_loss": float(best_loss),
            "final_loss": float(final_loss),
            "nan_grad_fallback_steps": int(nan_grad_fallback_steps),
        },
    )
    final_ckpt_path, final_ckpt_meta = save_params_checkpoint(
        outdir / "neural_xc_params_final.msgpack",
        state.params,
        metadata={
            "db_id": int(entry.db_id),
            "formula": entry.formula,
            "basis": args.basis,
            "xc": args.xc,
            "semilocal_xc": semilocal_xc,
            "n_semilocal_channels": args.n_semilocal_channels,
            "hf_input_mode": args.hf_input_mode,
            "hidden_dims": list(args.hidden_dims),
            "steps": int(args.steps),
            "learning_rate": float(args.learning_rate),
            "loss_mode": loss_mode,
            "energy_mse_weight": energy_mse_weight,
            "energy_mae_weight": energy_mae_weight,
            "log_every": int(args.log_every),
            "lr_decay_steps": int(args.lr_decay_steps),
            "density_weight": density_weight,
            "excited_weight": float(args.excited_weight),
            "excited_nstates": int(args.excited_nstates),
            "excited_use_tda": bool(args.excited_use_tda),
            "spectrum_weight": spectrum_weight,
            "spectrum_nstates": int(requested_spectrum_nstates),
            "spectrum_use_tda": bool(args.spectrum_use_tda),
            "spectrum_eta_ev": spectrum_eta_ev,
            "orbital_loss_weight": float(args.orbital_loss_weight),
            "orbital_loss_window": int(args.orbital_loss_window),
            "orbital_loss_mode": args.orbital_loss_mode,
            "orbital_loss_gradient": args.orbital_loss_gradient,
            "orbital_loss_scf_max_cycle": int(args.orbital_loss_scf_max_cycle),
            "orbital_loss_scf_damping": float(args.orbital_loss_scf_damping),
            "orbital_loss_scf_conv_tol_density": float(args.orbital_loss_scf_conv_tol_density),
            "orbital_loss_scf_vxc_clip": float(args.orbital_loss_scf_vxc_clip),
            "orbital_loss_scf_gradient_mode": args.orbital_loss_scf_gradient_mode,
            "orbital_loss_scf_implicit_max_iter": int(args.orbital_loss_scf_implicit_max_iter),
            "orbital_loss_scf_implicit_step_size": float(args.orbital_loss_scf_implicit_step_size),
            "orbital_loss_scf_implicit_clip": float(args.orbital_loss_scf_implicit_clip),
            "orbital_loss_scf_iterate_selection": args.orbital_loss_scf_iterate_selection,
            "seed": int(args.seed),
            "kind": "final",
            "nan_grad_fallback_steps": int(nan_grad_fallback_steps),
        },
    )

    compare_pngs: dict[str, Path] = {}
    orbital_images: dict[str, dict[str, Path]] = {}
    orbital_diff_norms: dict[str, float] = {}
    orbital_overlaps: dict[str, float] = {}
    orbital_diff_isos: dict[str, float] = {}
    orbital_diff_scales: dict[str, float] = {}
    if not args.skip_orbital_render:
        (
            orbital_images,
            orbital_diff_norms,
            orbital_overlaps,
            orbital_diff_isos,
            orbital_diff_scales,
        ) = _render_orbital_surfaces(
            entry=entry,
            neural_molecule=neural_scf_molecule,
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

        compare_dir = outdir / "orbital_surfaces" / "compare"
        compare_dir.mkdir(parents=True, exist_ok=True)
        for label, files in orbital_images.items():
            panel_path = compare_dir / f"{_safe_label(label)}_real_vs_neural.png"
            _plot_orbital_compare(
                orbital_label=label,
                ref_png=files["reference"],
                neural_png=files["neural"],
                diff_png=files["difference"],
                iso=args.orbital_iso,
                diff_iso=orbital_diff_isos[label],
                overlap_val=orbital_overlaps[label],
                diff_norm=orbital_diff_norms[label],
                diff_scale=orbital_diff_scales[label],
                out_png=panel_path,
            )
            compare_pngs[label] = panel_path

    _write_training_curve(
        loss_csv,
        loss_history,
        ground_history,
        excitation_penalty_history,
        spectrum_penalty_history,
        orbital_mse_history,
        orbital_weighted_history,
    )
    _plot_training_curve(
        loss_png,
        loss_history,
        ground_history,
        excitation_penalty_history,
        spectrum_penalty_history,
        orbital_weighted_history,
        excited_weight=float(args.excited_weight),
        spectrum_weight=spectrum_weight,
        orbital_loss_weight=orbital_loss_weight,
    )
    _write_spectrum_csv(spectrum_csv, grid_ev, ref_curve, neural_curve)
    _plot_spectrum(
        spectrum_png,
        entry=entry,
        grid_ev=grid_ev,
        ref_curve=ref_curve,
        neural_curve=neural_curve,
        ref_label=f"{args.xc.upper()}/{args.basis.upper()} TDDFT",
        excitation_mae_ev=self_consistent_excitation_mae_ev,
    )

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"db_path={args.db_path}\n")
        f.write(f"db_id={entry.db_id}\n")
        f.write(f"formula={entry.formula}\n")
        f.write(f"basis={args.basis}\n")
        f.write(f"xc={args.xc}\n")
        f.write(f"semilocal_xc={semilocal_xc}\n")
        f.write(f"n_semilocal_channels={args.n_semilocal_channels}\n")
        f.write(f"hf_input_mode={args.hf_input_mode}\n")
        f.write(f"hidden_dims={tuple(args.hidden_dims)}\n")
        f.write(f"steps={args.steps}\n")
        f.write(f"learning_rate={args.learning_rate}\n")
        f.write(f"loss_mode={loss_mode}\n")
        f.write(f"energy_mse_weight={energy_mse_weight}\n")
        f.write(f"energy_mae_weight={energy_mae_weight}\n")
        f.write(f"log_every={args.log_every}\n")
        f.write(f"lr_decay_steps={args.lr_decay_steps}\n")
        f.write(f"density_weight={density_weight}\n")
        f.write(f"excited_weight={args.excited_weight}\n")
        f.write(f"excited_nstates={args.excited_nstates}\n")
        f.write(f"excited_use_tda={bool(args.excited_use_tda)}\n")
        f.write(f"spectrum_weight={spectrum_weight}\n")
        f.write(f"spectrum_nstates={requested_spectrum_nstates}\n")
        f.write(f"spectrum_use_tda={bool(args.spectrum_use_tda)}\n")
        f.write(f"spectrum_eta_ev={spectrum_eta_ev}\n")
        f.write(f"orbital_loss_weight={args.orbital_loss_weight}\n")
        f.write(f"orbital_loss_window={args.orbital_loss_window}\n")
        f.write(f"orbital_loss_mode={args.orbital_loss_mode}\n")
        f.write(f"orbital_loss_gradient={args.orbital_loss_gradient}\n")
        f.write(f"orbital_loss_window_start={window_start}\n")
        f.write(f"orbital_loss_window_stop_exclusive={window_stop}\n")
        f.write(f"orbital_loss_window_size={window_stop - window_start}\n")
        f.write(f"orbital_loss_ref_homo_index={window_homo}\n")
        f.write(f"orbital_loss_ref_lumo_index={window_lumo}\n")
        f.write(f"orbital_loss_homo_minus_effective={window_homo - window_start}\n")
        f.write(f"orbital_loss_lumo_plus_effective={(window_stop - 1) - window_lumo}\n")
        f.write(f"orbital_loss_scf_max_cycle={args.orbital_loss_scf_max_cycle}\n")
        f.write(f"orbital_loss_scf_damping={args.orbital_loss_scf_damping}\n")
        f.write(f"orbital_loss_scf_conv_tol_density={args.orbital_loss_scf_conv_tol_density}\n")
        f.write(f"orbital_loss_scf_vxc_clip={args.orbital_loss_scf_vxc_clip}\n")
        f.write(f"orbital_loss_scf_gradient_mode={args.orbital_loss_scf_gradient_mode}\n")
        f.write(f"orbital_loss_scf_implicit_max_iter={args.orbital_loss_scf_implicit_max_iter}\n")
        f.write(f"orbital_loss_scf_implicit_step_size={args.orbital_loss_scf_implicit_step_size}\n")
        f.write(f"orbital_loss_scf_implicit_clip={args.orbital_loss_scf_implicit_clip}\n")
        f.write(f"orbital_loss_scf_iterate_selection={args.orbital_loss_scf_iterate_selection}\n")
        f.write(f"eval_params={args.eval_params}\n")
        f.write(f"nstates={args.nstates}\n")
        f.write(f"solver={solver_label}\n")
        f.write(f"sample_elapsed_s={sample_elapsed:.2f}\n")
        f.write(f"training_elapsed_s={train_elapsed:.2f}\n")
        f.write(f"orbital_scf_converged={bool(np.asarray(scf_info.converged))}\n")
        f.write(f"orbital_scf_cycles={int(np.asarray(scf_info.cycles))}\n")
        f.write(f"orbital_scf_damping={args.orbital_scf_damping}\n")
        f.write(f"orbital_scf_conv_tol_density={args.orbital_scf_conv_tol_density}\n")
        f.write(f"orbital_scf_vxc_clip={args.orbital_scf_vxc_clip}\n")
        f.write(f"orbital_scf_iterate_selection={args.orbital_scf_iterate_selection}\n")
        f.write(f"orbital_scf_final_rms_density={float(np.asarray(scf_info.final_rms_density)):.6e}\n")
        f.write(f"orbital_scf_selected_cycle={int(np.asarray(scf_info.selected_cycle))}\n")
        f.write(
            f"orbital_scf_selected_rms_density={float(np.asarray(scf_info.selected_rms_density)):.6e}\n"
        )
        f.write(f"orbital_scf_best_cycle={int(np.asarray(scf_info.best_cycle))}\n")
        f.write(f"orbital_scf_best_rms_density={float(np.asarray(scf_info.best_rms_density)):.6e}\n")
        f.write(f"initial_train_loss={loss_history[0]:.12e}\n")
        f.write(f"min_train_loss={min_loss:.12e}\n")
        f.write(f"min_train_loss_step={min_step}\n")
        f.write(f"best_train_loss={best_loss:.12e}\n")
        f.write(f"best_ground_loss={best_ground_loss:.12e}\n")
        f.write(f"best_excitation_penalty={best_excitation_penalty:.12e}\n")
        f.write(f"best_spectrum_penalty={best_spectrum_penalty:.12e}\n")
        f.write(f"best_orbital_mse_loss={best_orbital_mse:.12e}\n")
        f.write(f"best_orbital_weighted_loss={best_orbital_weighted:.12e}\n")
        f.write(f"final_train_loss={final_loss:.12e}\n")
        f.write(f"final_ground_loss={final_ground_loss:.12e}\n")
        f.write(f"final_excitation_penalty={final_excitation_penalty:.12e}\n")
        f.write(f"final_spectrum_penalty={final_spectrum_penalty:.12e}\n")
        f.write(f"final_orbital_mse_loss={final_orbital_mse:.12e}\n")
        f.write(f"final_orbital_weighted_loss={final_orbital_weighted:.12e}\n")
        f.write(f"nan_grad_fallback_steps={nan_grad_fallback_steps}\n")
        f.write(f"train_path_excitation_mae_ev={train_path_excitation_mae_ev:.6f}\n")
        f.write(f"train_path_spectrum_mse={train_path_spectrum_mse:.12e}\n")
        f.write(f"self_consistent_excitation_mae_ev={self_consistent_excitation_mae_ev:.6f}\n")
        f.write(f"train_energy_abs_error_ha={total_energy_abs_err:.6f}\n")
        f.write(f"skip_orbital_render={bool(args.skip_orbital_render)}\n")
        f.write(f"params_ckpt={ckpt_path}\n")
        f.write(f"params_meta={ckpt_meta}\n")
        f.write(f"params_ckpt_final={final_ckpt_path}\n")
        f.write(f"params_meta_final={final_ckpt_meta}\n")
        f.write(f"orbital_iso={args.orbital_iso}\n")
        f.write(f"orbital_diff_iso_target={args.orbital_diff_iso}\n")
        f.write(f"orbital_mo_blur={args.orbital_mo_blur}\n")
        f.write(f"orbital_mo_upsample={args.orbital_mo_upsample}\n")
        f.write(f"orbital_frontier_match={not bool(args.disable_orbital_frontier_match)}\n")
        f.write(f"orbital_frontier_match_window={args.orbital_frontier_match_window}\n")
        if not args.skip_orbital_render:
            for label in ("HOMO-1", "HOMO", "LUMO", "LUMO+1"):
                safe = _safe_label(label)
                f.write(f"orbital_{safe}_ref_png={orbital_images[label]['reference']}\n")
                f.write(f"orbital_{safe}_neural_png={orbital_images[label]['neural']}\n")
                f.write(f"orbital_{safe}_diff_png={orbital_images[label]['difference']}\n")
                f.write(f"orbital_{safe}_compare_png={compare_pngs[label]}\n")
                f.write(f"orbital_{safe}_overlap={orbital_overlaps[label]:.8f}\n")
                f.write(f"orbital_{safe}_diff_norm={orbital_diff_norms[label]:.8f}\n")
                f.write(f"orbital_{safe}_diff_iso_used={orbital_diff_isos[label]:.8f}\n")
                f.write(f"orbital_{safe}_diff_scale_used={orbital_diff_scales[label]:.8f}\n")

    print(f"[QuickQH9] id={entry.db_id} formula={entry.formula}")
    print(
        f"[QuickQH9] Initial/Min/Best/Final loss: "
        f"{loss_history[0]:.6e} / {min_loss:.6e} / {best_loss:.6e} / {final_loss:.6e}"
    )
    print(
        f"[QuickQH9] Final ground/orbital_mse/orbital_weighted: "
        f"{final_ground_loss:.6e} / {final_orbital_mse:.6e} / {final_orbital_weighted:.6e}"
    )
    print(
        f"[QuickQH9] Final excitation/spectrum penalties: "
        f"{final_excitation_penalty:.6e} / {final_spectrum_penalty:.6e}"
    )
    print(f"[QuickQH9] NaN-gradient fallback steps: {nan_grad_fallback_steps}")
    print(f"[QuickQH9] Eval params: {args.eval_params} (best step={min_step})")
    print(f"[QuickQH9] Train-path excitation MAE (eV): {train_path_excitation_mae_ev:.6f}")
    if spectrum_weight != 0.0:
        print(f"[QuickQH9] Train-path spectrum MSE: {train_path_spectrum_mse:.6e}")
    print(
        f"[QuickQH9] Self-consistent excitation MAE (eV): "
        f"{self_consistent_excitation_mae_ev:.6f}"
    )
    print(f"[QuickQH9] |E_total| error (Ha): {total_energy_abs_err:.6f}")
    print(
        "[QuickQH9] Orbital SCF: "
        f"converged={bool(np.asarray(scf_info.converged))}, "
        f"cycles={int(np.asarray(scf_info.cycles))}, "
        f"selected_cycle={int(np.asarray(scf_info.selected_cycle))}, "
        f"selected_rms={float(np.asarray(scf_info.selected_rms_density)):.3e}, "
        f"final_rms={float(np.asarray(scf_info.final_rms_density)):.3e}"
    )
    if not args.skip_orbital_render:
        print(f"[QuickQH9] Orbital compare HOMO: {compare_pngs['HOMO']}")
        print(f"[QuickQH9] Orbital compare LUMO: {compare_pngs['LUMO']}")
    else:
        print("[QuickQH9] Orbital rendering skipped")
    print(f"[QuickQH9] Outputs: {outdir.resolve()}")


if __name__ == "__main__":
    main()
