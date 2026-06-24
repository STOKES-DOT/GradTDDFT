from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


os.environ["JAX_PLATFORMS"] = os.environ.get("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = os.environ.get("JAX_ENABLE_X64", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path("benchmark") / ".mplconfig"))

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import dft, gto
from pyscf.dft import numint

from td_graddft import tdscf
from td_graddft.data.reference import restricted_reference_from_pyscf
from td_graddft.scf.features import _charge_center
from td_graddft.scf.molecules import QuadratureGrid, UnrestrictedMolecule
from td_graddft.spectra import HARTREE_TO_EV


MOLECULES: dict[str, dict[str, Any]] = {
    "water": {
        "atom": """
O  0.000000  0.000000  0.117790
H  0.000000  0.755453 -0.471161
H  0.000000 -0.755453 -0.471161
""",
        "charge": 0,
        "spin": 0,
        "scf_type": "RKS",
    },
    "co": {
        "atom": "C 0.000000 0.000000 0.000000; O 0.000000 0.000000 1.128200",
        "charge": 0,
        "spin": 0,
        "scf_type": "RKS",
    },
    "n2": {
        "atom": "N 0.000000 0.000000 -0.548850; N 0.000000 0.000000 0.548850",
        "charge": 0,
        "spin": 0,
        "scf_type": "RKS",
    },
    "ethylene": {
        "atom": """
C -0.669500  0.000000 0.000000
C  0.669500  0.000000 0.000000
H -1.232900  0.928900 0.000000
H -1.232900 -0.928900 0.000000
H  1.232900  0.928900 0.000000
H  1.232900 -0.928900 0.000000
""",
        "charge": 0,
        "spin": 0,
        "scf_type": "RKS",
    },
    "formaldehyde": {
        "atom": """
C  0.000000  0.000000  0.000000
O  0.000000  0.000000  1.208000
H  0.000000  0.937000 -0.586000
H  0.000000 -0.937000 -0.586000
""",
        "charge": 0,
        "spin": 0,
        "scf_type": "RKS",
    },
    "benzene": {
        "atom": """
C  0.000000  1.396792 0.000000
C -1.209657  0.698396 0.000000
C -1.209657 -0.698396 0.000000
C  0.000000 -1.396792 0.000000
C  1.209657 -0.698396 0.000000
C  1.209657  0.698396 0.000000
H  0.000000  2.484212 0.000000
H -2.151390  1.242106 0.000000
H -2.151390 -1.242106 0.000000
H  0.000000 -2.484212 0.000000
H  2.151390 -1.242106 0.000000
H  2.151390  1.242106 0.000000
""",
        "charge": 0,
        "spin": 0,
        "scf_type": "RKS",
    },
    "oh": {
        "atom": "O 0.000000 0.000000 0.000000; H 0.000000 0.000000 0.969700",
        "charge": 0,
        "spin": 1,
        "scf_type": "UKS",
    },
}

PAPER_MOLECULES = ("water", "co", "n2", "ethylene", "formaldehyde", "benzene", "oh")
PAPER_XCS = ("pbe", "b3lyp", "pbe0")
PAPER_BASES = ("def2-svp", "aug-cc-pvdz")
PAPER_SOLVERS = ("tda", "tddft")
PAPER_CLOSED_SHELL_SPINS = ("singlet", "triplet")

VIS_FIELDS = [
    "timestamp_utc",
    "task_id",
    "molecule",
    "scf_type",
    "charge",
    "spin_2s",
    "basis",
    "xc",
    "grid_level",
    "response_solver",
    "spin_channel",
    "state_index",
    "support_status",
    "pyscf_scf_converged",
    "pyscf_td_converged",
    "graddft_td_converged",
    "nao",
    "nmo",
    "nocc",
    "nvir",
    "scf_energy_ha",
    "pyscf_excitation_ha",
    "graddft_excitation_ha",
    "abs_diff_ha",
    "pyscf_excitation_ev",
    "graddft_excitation_ev",
    "abs_diff_ev",
    "pyscf_oscillator_strength",
    "graddft_oscillator_strength",
    "abs_diff_oscillator_strength",
    "pyscf_transition_dipole_norm",
    "graddft_transition_dipole_norm",
    "abs_diff_transition_dipole_norm",
    "notes",
]

SUMMARY_FIELDS = [
    "timestamp_utc",
    "task_id",
    "molecule",
    "scf_type",
    "charge",
    "spin_2s",
    "basis",
    "xc",
    "grid_level",
    "response_solver",
    "spin_channel",
    "support_status",
    "status",
    "error_type",
    "error_message",
    "nstates_requested",
    "states_compared",
    "nao",
    "nmo",
    "nocc",
    "nvir",
    "scf_energy_ha",
    "mae_energy_ev",
    "max_abs_energy_ev",
    "mae_oscillator_strength",
    "max_abs_oscillator_strength",
    "mae_transition_dipole_norm",
    "max_abs_transition_dipole_norm",
    "matrix_a_rms_diff",
    "matrix_a_max_abs_diff",
    "matrix_b_rms_diff",
    "matrix_b_max_abs_diff",
    "pyscf_scf_elapsed_s",
    "pyscf_td_elapsed_s",
    "graddft_td_elapsed_s",
    "pass_energy_tol",
    "pass_osc_tol",
    "notes",
]

MANIFEST_FIELDS = [
    "task_id",
    "molecule",
    "scf_type",
    "charge",
    "spin_2s",
    "basis",
    "xc",
    "grid_level",
    "response_solver",
    "spin_channel",
    "nstates",
    "support_status",
]


@dataclass(frozen=True)
class Task:
    molecule: str
    xc: str
    basis: str
    solver: str
    spin_channel: str
    nstates: int
    grid_level: int

    @property
    def molecule_info(self) -> dict[str, Any]:
        return MOLECULES[self.molecule]

    @property
    def scf_type(self) -> str:
        return str(self.molecule_info["scf_type"])

    @property
    def charge(self) -> int:
        return int(self.molecule_info["charge"])

    @property
    def spin_2s(self) -> int:
        return int(self.molecule_info["spin"])

    @property
    def task_id(self) -> str:
        raw = "|".join(
            [
                self.molecule,
                self.scf_type,
                self.xc,
                self.basis,
                str(self.grid_level),
                self.solver,
                self.spin_channel,
                str(self.nstates),
            ]
        )
        suffix = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
        safe = raw.replace("|", "__").replace("/", "_")
        return f"{safe}__{suffix}"

    @property
    def support_status(self) -> str:
        if self.scf_type == "RKS" and self.spin_channel == "singlet":
            return "graddft_supported"
        if self.scf_type == "RKS" and self.spin_channel == "triplet":
            return "pyscf_only_restricted_triplet_pending"
        return "pyscf_only_unrestricted_semilocal_pending"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str], *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _completed_task_ids(summary_path: Path) -> set[str]:
    if not summary_path.exists():
        return set()
    with summary_path.open() as handle:
        return {
            row["task_id"]
            for row in csv.DictReader(handle)
            if row.get("status") == "ok" and row.get("task_id")
        }


def _make_tasks(args: argparse.Namespace) -> list[Task]:
    if args.preset == "smoke":
        molecules = args.molecules or ["water"]
        xcs = args.xcs or ["pbe"]
        bases = args.bases or ["def2-svp"]
        solvers = args.solvers or ["tda", "tddft"]
        spin_channels = args.spin_channels or ["singlet"]
    else:
        molecules = args.molecules or list(PAPER_MOLECULES)
        xcs = args.xcs or list(PAPER_XCS)
        bases = args.bases or list(PAPER_BASES)
        solvers = args.solvers or list(PAPER_SOLVERS)
        spin_channels = args.spin_channels

    tasks: list[Task] = []
    for molecule in molecules:
        info = MOLECULES[molecule]
        channels = spin_channels
        if channels is None:
            channels = (
                ["spin_conserving"]
                if str(info["scf_type"]) == "UKS"
                else list(PAPER_CLOSED_SHELL_SPINS)
            )
        for basis in bases:
            for xc in xcs:
                for solver in solvers:
                    for spin_channel in channels:
                        if str(info["scf_type"]) == "UKS" and spin_channel != "spin_conserving":
                            continue
                        if str(info["scf_type"]) == "RKS" and spin_channel == "spin_conserving":
                            continue
                        tasks.append(
                            Task(
                                molecule=molecule,
                                xc=xc,
                                basis=basis,
                                solver=solver,
                                spin_channel=spin_channel,
                                nstates=int(args.nstates),
                                grid_level=int(args.grid_level),
                            )
                        )
    if args.max_tasks is not None:
        tasks = tasks[: int(args.max_tasks)]
    return tasks


def _build_mf(task: Task):
    mol = gto.M(
        atom=task.molecule_info["atom"],
        unit="Angstrom",
        charge=task.charge,
        spin=task.spin_2s,
        basis=task.basis,
        verbose=0,
    )
    if task.scf_type == "UKS":
        mf = dft.UKS(mol)
    else:
        mf = dft.RKS(mol)
    mf.xc = task.xc
    mf.grids.level = task.grid_level
    mf.conv_tol = 1e-10
    mf.max_cycle = 160
    start = time.perf_counter()
    mf.kernel()
    elapsed = time.perf_counter() - start
    return mf, elapsed


def _run_pyscf_td(task: Task, mf: Any):
    td = mf.TDA() if task.solver == "tda" else mf.TDDFT()
    td.nstates = task.nstates
    if task.scf_type == "RKS":
        td.singlet = task.spin_channel == "singlet"
    start = time.perf_counter()
    td.kernel()
    elapsed = time.perf_counter() - start
    energies = np.asarray(td.e, dtype=float).reshape(-1)
    osc = _safe_array(lambda: td.oscillator_strength(), size=energies.size)
    dipoles = _safe_dipoles(lambda: td.transition_dipole(), size=energies.size)
    return td, energies, osc, dipoles, elapsed


def _safe_array(fn, *, size: int) -> np.ndarray:
    try:
        values = np.asarray(fn(), dtype=float).reshape(-1)
    except Exception:
        return np.full((size,), np.nan)
    if values.size < size:
        padded = np.full((size,), np.nan)
        padded[: values.size] = values
        return padded
    return values[:size]


def _safe_dipoles(fn, *, size: int) -> np.ndarray:
    try:
        values = np.asarray(fn(), dtype=float)
    except Exception:
        return np.full((size, 3), np.nan)
    if values.ndim == 1:
        values = values.reshape(-1, 3)
    if values.shape[0] < size:
        padded = np.full((size, 3), np.nan)
        padded[: values.shape[0], : values.shape[1]] = values
        return padded
    return values[:size, :3]


def _unrestricted_reference_from_pyscf(mf: Any) -> UnrestrictedMolecule:
    if getattr(mf, "mo_coeff", None) is None:
        raise ValueError("PySCF mean-field object is not converged; run mf.kernel() first.")
    if getattr(mf.grids, "coords", None) is None:
        mf.grids.build()

    mo_coeff = jnp.asarray(mf.mo_coeff)
    mo_occ = jnp.asarray(mf.mo_occ)
    mo_energy = jnp.asarray(mf.mo_energy)
    nocc_alpha = int(np.count_nonzero(np.asarray(mf.mo_occ)[0] > 1e-8))
    nocc_beta = int(np.count_nonzero(np.asarray(mf.mo_occ)[1] > 1e-8))
    dm_spin = jnp.asarray(mf.make_rdm1())
    ao = jnp.asarray(numint.eval_ao(mf.mol, mf.grids.coords, deriv=0))
    ao_deriv1 = jnp.asarray(numint.eval_ao(mf.mol, mf.grids.coords, deriv=1))
    with mf.mol.with_common_orig(_charge_center(mf.mol)):
        dipole_integrals = jnp.asarray(mf.mol.intor_symmetric("int1e_r", comp=3))
    return UnrestrictedMolecule(
        ao=ao,
        grid=QuadratureGrid(weights=jnp.asarray(mf.grids.weights), coords=jnp.asarray(mf.grids.coords)),
        dipole_integrals=dipole_integrals,
        rep_tensor=jnp.asarray(mf.mol.intor("int2e")),
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        rdm1=dm_spin,
        h1e=jnp.asarray(mf.get_hcore()),
        nuclear_repulsion=float(mf.mol.energy_nuc()),
        atom_coords=jnp.asarray(mf.mol.atom_coords()),
        atom_charges=jnp.asarray(mf.mol.atom_charges()),
        overlap_matrix=jnp.asarray(mf.get_ovlp()),
        ao_deriv1=ao_deriv1,
        mf_energy=float(getattr(mf, "e_tot", jnp.nan)),
        exact_exchange_fraction=0.0,
        nocc_alpha=nocc_alpha,
        nocc_beta=nocc_beta,
        hfx_omega_values=None,
    )


def _run_graddft_td(task: Task, mf: Any):
    if task.support_status != "graddft_supported":
        return None, None, None, None, None, 0.0
    reference = restricted_reference_from_pyscf(mf)
    td = tdscf.TDA(reference, xc_functional=task.xc) if task.solver == "tda" else tdscf.TDDFT(reference, xc_functional=task.xc)
    td.nstates = task.nstates
    start = time.perf_counter()
    result = td.kernel()
    elapsed = time.perf_counter() - start
    energies = np.asarray(result.excitation_energies, dtype=float).reshape(-1)
    osc = _safe_array(lambda: td.oscillator_strength(), size=energies.size)
    dipoles = _safe_dipoles(lambda: td.transition_dipole(), size=energies.size)
    return reference, td, energies, osc, dipoles, elapsed


def _matrix_metrics(task: Task, td_ref: Any, td_gdft: Any, *, max_elements: int) -> dict[str, Any]:
    empty = {
        "matrix_a_rms_diff": "",
        "matrix_a_max_abs_diff": "",
        "matrix_b_rms_diff": "",
        "matrix_b_max_abs_diff": "",
    }
    if task.support_status != "graddft_supported" or td_gdft is None:
        return empty
    solver = getattr(td_gdft, "_solver", None)
    if solver is None:
        return empty
    try:
        a_p, b_p = td_ref.get_ab()
        a_p = np.asarray(a_p, dtype=float)
        b_p = np.asarray(b_p, dtype=float)
        dim = int(np.prod(a_p.shape[:2]))
        if dim * dim > max_elements:
            return {key: "skipped_size" for key in empty}

        eye = jnp.eye(dim, dtype=jnp.asarray(td_gdft.result.excitation_energies).dtype)
        zeros = jnp.zeros_like(eye)
        if task.solver == "tda":
            vind = solver.gen_tda_vind()
            a_g = np.asarray(vind(eye), dtype=float).reshape(dim, dim).T
            a_p = a_p.reshape(dim, dim)
            if a_p.shape != a_g.shape:
                return {key: "shape_mismatch" for key in empty}
            da = a_g - a_p
            return {
                "matrix_a_rms_diff": float(np.sqrt(np.mean(da * da))),
                "matrix_a_max_abs_diff": float(np.max(np.abs(da))),
                "matrix_b_rms_diff": "",
                "matrix_b_max_abs_diff": "",
            }
        else:
            vind = solver.gen_tdhf_vind()
            response = np.asarray(
                vind(jnp.concatenate([eye, zeros], axis=-1)),
                dtype=float,
            ).reshape(dim, 2 * dim)
            a_g = response[:, :dim].T
            b_g = -response[:, dim:].T

        a_p = a_p.reshape(dim, dim)
        b_p = b_p.reshape(dim, dim)
        if a_p.shape != a_g.shape or b_p.shape != b_g.shape:
            return {key: "shape_mismatch" for key in empty}
        da = a_g - a_p
        db = b_g - b_p
        return {
            "matrix_a_rms_diff": float(np.sqrt(np.mean(da * da))),
            "matrix_a_max_abs_diff": float(np.max(np.abs(da))),
            "matrix_b_rms_diff": float(np.sqrt(np.mean(db * db))),
            "matrix_b_max_abs_diff": float(np.max(np.abs(db))),
        }
    except Exception as exc:
        return {key: f"error:{type(exc).__name__}" for key in empty}


def _td_converged(td: Any) -> bool:
    value = getattr(td, "converged", True)
    if isinstance(value, (list, tuple, np.ndarray)):
        return bool(np.all(value))
    return bool(value)


def _dimension_info(task: Task, mf: Any) -> dict[str, Any]:
    nao = int(mf.mol.nao_nr())
    mo_occ = np.asarray(mf.mo_occ)
    if task.scf_type == "RKS":
        nocc = int(np.count_nonzero(mo_occ > 1e-8))
        nmo = int(np.asarray(mf.mo_coeff).shape[-1])
        nvir = int(nmo - nocc)
    else:
        nocc = f"{int(np.count_nonzero(mo_occ[0] > 1e-8))}/{int(np.count_nonzero(mo_occ[1] > 1e-8))}"
        nmo = f"{int(np.asarray(mf.mo_coeff[0]).shape[-1])}/{int(np.asarray(mf.mo_coeff[1]).shape[-1])}"
        nvir = f"{int(np.asarray(mf.mo_coeff[0]).shape[-1] - np.count_nonzero(mo_occ[0] > 1e-8))}/{int(np.asarray(mf.mo_coeff[1]).shape[-1] - np.count_nonzero(mo_occ[1] > 1e-8))}"
    return {"nao": nao, "nmo": nmo, "nocc": nocc, "nvir": nvir}


def _run_task(task: Task, args: argparse.Namespace, outdir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    timestamp = _now()
    base = {
        "timestamp_utc": timestamp,
        "task_id": task.task_id,
        "molecule": task.molecule,
        "scf_type": task.scf_type,
        "charge": task.charge,
        "spin_2s": task.spin_2s,
        "basis": task.basis,
        "xc": task.xc,
        "grid_level": task.grid_level,
        "response_solver": task.solver,
        "spin_channel": task.spin_channel,
        "support_status": task.support_status,
        "nstates_requested": task.nstates,
    }
    mf, scf_elapsed = _build_mf(task)
    dims = _dimension_info(task, mf)
    base.update(dims)
    base["scf_energy_ha"] = float(mf.e_tot)
    base["pyscf_scf_elapsed_s"] = float(scf_elapsed)
    base["pyscf_scf_converged"] = bool(mf.converged)
    if not bool(mf.converged):
        raise RuntimeError("PySCF SCF did not converge.")

    td_ref, ref_e, ref_f, ref_mu, pyscf_td_elapsed = _run_pyscf_td(task, mf)
    base["pyscf_td_elapsed_s"] = float(pyscf_td_elapsed)
    base["pyscf_td_converged"] = _td_converged(td_ref)

    reference, td_gdft, pred_e, pred_f, pred_mu, gdft_td_elapsed = _run_graddft_td(task, mf)
    del reference
    base["graddft_td_elapsed_s"] = float(gdft_td_elapsed)
    base["graddft_td_converged"] = td_gdft.converged if td_gdft is not None else ""

    n_ref = int(ref_e.size)
    n_pred = int(pred_e.size) if pred_e is not None else 0
    n = min(n_ref, n_pred, task.nstates) if pred_e is not None else n_ref
    if task.support_status == "graddft_supported":
        n_comp = min(n_ref, n_pred, task.nstates)
    else:
        n_comp = 0

    matrix = _matrix_metrics(
        task,
        td_ref,
        td_gdft,
        max_elements=int(args.matrix_max_elements),
    )
    base.update(matrix)

    state_rows: list[dict[str, Any]] = []
    diffs_e: list[float] = []
    diffs_f: list[float] = []
    diffs_mu: list[float] = []
    for idx in range(n):
        ref_energy_ha = float(ref_e[idx])
        ref_energy_ev = ref_energy_ha * HARTREE_TO_EV
        ref_osc = float(ref_f[idx]) if idx < ref_f.size else float("nan")
        ref_mu_norm = float(np.linalg.norm(ref_mu[idx])) if idx < ref_mu.shape[0] else float("nan")
        pred_energy_ha = float(pred_e[idx]) if pred_e is not None and idx < pred_e.size else float("nan")
        pred_energy_ev = pred_energy_ha * HARTREE_TO_EV if np.isfinite(pred_energy_ha) else float("nan")
        pred_osc = float(pred_f[idx]) if pred_f is not None and idx < pred_f.size else float("nan")
        pred_mu_norm = float(np.linalg.norm(pred_mu[idx])) if pred_mu is not None and idx < pred_mu.shape[0] else float("nan")
        abs_diff_ha = abs(pred_energy_ha - ref_energy_ha) if np.isfinite(pred_energy_ha) else float("nan")
        abs_diff_ev = abs(pred_energy_ev - ref_energy_ev) if np.isfinite(pred_energy_ev) else float("nan")
        abs_diff_osc = abs(pred_osc - ref_osc) if np.isfinite(pred_osc) and np.isfinite(ref_osc) else float("nan")
        abs_diff_mu = abs(pred_mu_norm - ref_mu_norm) if np.isfinite(pred_mu_norm) and np.isfinite(ref_mu_norm) else float("nan")
        if idx < n_comp:
            diffs_e.append(abs_diff_ev)
            if np.isfinite(abs_diff_osc):
                diffs_f.append(abs_diff_osc)
            if np.isfinite(abs_diff_mu):
                diffs_mu.append(abs_diff_mu)
        row = {
            **base,
            "state_index": idx + 1,
            "pyscf_excitation_ha": ref_energy_ha,
            "graddft_excitation_ha": pred_energy_ha,
            "abs_diff_ha": abs_diff_ha,
            "pyscf_excitation_ev": ref_energy_ev,
            "graddft_excitation_ev": pred_energy_ev,
            "abs_diff_ev": abs_diff_ev,
            "pyscf_oscillator_strength": ref_osc,
            "graddft_oscillator_strength": pred_osc,
            "abs_diff_oscillator_strength": abs_diff_osc,
            "pyscf_transition_dipole_norm": ref_mu_norm,
            "graddft_transition_dipole_norm": pred_mu_norm,
            "abs_diff_transition_dipole_norm": abs_diff_mu,
            "notes": "",
        }
        state_rows.append(row)

    summary = {
        **base,
        "status": "ok",
        "error_type": "",
        "error_message": "",
        "states_compared": n_comp,
        "mae_energy_ev": float(np.mean(diffs_e)) if diffs_e else "",
        "max_abs_energy_ev": float(np.max(diffs_e)) if diffs_e else "",
        "mae_oscillator_strength": float(np.mean(diffs_f)) if diffs_f else "",
        "max_abs_oscillator_strength": float(np.max(diffs_f)) if diffs_f else "",
        "mae_transition_dipole_norm": float(np.mean(diffs_mu)) if diffs_mu else "",
        "max_abs_transition_dipole_norm": float(np.max(diffs_mu)) if diffs_mu else "",
        "pass_energy_tol": (bool(diffs_e) and max(diffs_e) <= float(args.energy_tol_ev)) if task.support_status == "graddft_supported" else "",
        "pass_osc_tol": (bool(diffs_f) and max(diffs_f) <= float(args.osc_tol)) if task.support_status == "graddft_supported" else "",
        "notes": "PySCF-only reference row" if task.support_status != "graddft_supported" else "",
    }
    _write_jsonl(
        outdir / "progress.jsonl",
        {"event": "task_ok", "timestamp_utc": timestamp, "task_id": task.task_id, "support_status": task.support_status},
    )
    return state_rows, summary


def _error_summary(task: Task, exc: BaseException) -> dict[str, Any]:
    return {
        "timestamp_utc": _now(),
        "task_id": task.task_id,
        "molecule": task.molecule,
        "scf_type": task.scf_type,
        "charge": task.charge,
        "spin_2s": task.spin_2s,
        "basis": task.basis,
        "xc": task.xc,
        "grid_level": task.grid_level,
        "response_solver": task.solver,
        "spin_channel": task.spin_channel,
        "support_status": task.support_status,
        "status": "error",
        "error_type": type(exc).__name__,
        "error_message": str(exc).replace("\n", " ")[:800],
        "nstates_requested": task.nstates,
        "states_compared": 0,
    }


def _write_manifest(tasks: list[Task], path: Path) -> None:
    rows = [
        {
            "task_id": task.task_id,
            "molecule": task.molecule,
            "scf_type": task.scf_type,
            "charge": task.charge,
            "spin_2s": task.spin_2s,
            "basis": task.basis,
            "xc": task.xc,
            "grid_level": task.grid_level,
            "response_solver": task.solver,
            "spin_channel": task.spin_channel,
            "nstates": task.nstates,
            "support_status": task.support_status,
        }
        for task in tasks
    ]
    _write_csv(path, rows, MANIFEST_FIELDS, append=False)


def _write_environment(outdir: Path, args: argparse.Namespace) -> None:
    data = {
        "timestamp_utc": _now(),
        "python": sys.version,
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "args": vars(args),
    }
    try:
        import pyscf

        data["pyscf_version"] = pyscf.__version__
    except Exception as exc:
        data["pyscf_version"] = f"error:{exc}"
    path = outdir / "environment.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--outdir", default="benchmark/pyscf_correctness")
    parser.add_argument("--molecules", nargs="+", choices=sorted(MOLECULES))
    parser.add_argument("--xcs", nargs="+")
    parser.add_argument("--bases", nargs="+")
    parser.add_argument("--solvers", nargs="+", choices=PAPER_SOLVERS)
    parser.add_argument("--spin-channels", nargs="+", choices=("singlet", "triplet", "spin_conserving"))
    parser.add_argument("--nstates", type=int, default=5)
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--write-manifest-only", action="store_true")
    parser.add_argument("--matrix-max-elements", type=int, default=200000)
    parser.add_argument("--energy-tol-ev", type=float, default=5.0e-2)
    parser.add_argument("--osc-tol", type=float, default=5.0e-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tasks = _make_tasks(args)
    _write_manifest(tasks, outdir / "task_manifest.csv")
    _write_environment(outdir, args)
    if args.write_manifest_only:
        print(f"wrote_manifest={outdir / 'task_manifest.csv'}")
        print(f"tasks={len(tasks)}")
        return

    summary_path = outdir / "summary.csv"
    viz_path = outdir / "visualization_data.csv"
    completed = _completed_task_ids(summary_path) if args.skip_existing else set()
    for index, task in enumerate(tasks, start=1):
        if task.task_id in completed:
            continue
        _write_jsonl(
            outdir / "progress.jsonl",
            {
                "event": "task_start",
                "timestamp_utc": _now(),
                "index": index,
                "ntasks": len(tasks),
                "task_id": task.task_id,
            },
        )
        try:
            state_rows, summary = _run_task(task, args, outdir)
        except Exception as exc:
            summary = _error_summary(task, exc)
            state_rows = []
            _write_jsonl(
                outdir / "progress.jsonl",
                {
                    "event": "task_error",
                    "timestamp_utc": _now(),
                    "task_id": task.task_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        _write_csv(summary_path, [summary], SUMMARY_FIELDS, append=True)
        if state_rows:
            _write_csv(viz_path, state_rows, VIS_FIELDS, append=True)
        print(
            f"[{index}/{len(tasks)}] {task.molecule} {task.xc}/{task.basis} "
            f"{task.solver} {task.spin_channel}: {summary['status']}"
        )

    print(f"manifest={outdir / 'task_manifest.csv'}")
    print(f"summary={summary_path}")
    print(f"visualization_data={viz_path}")
    print(f"progress={outdir / 'progress.jsonl'}")


if __name__ == "__main__":
    main()
