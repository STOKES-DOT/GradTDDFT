from __future__ import annotations

import argparse
import csv
import math
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pyscf import dft, gto
from pyscf.dft import numint

from td_graddft.data.basis import basis_from_pyscf_mol_cart
from td_graddft.data.integrals import build_hcore, eri_tensor, overlap_matrix
from td_graddft.device import put_restricted_reference_on_device
from td_graddft.features import restricted_grid_features_with_gradients
from td_graddft.jax_libxc import eval_xc_response_tensor, hybrid_coeff, xc_type
from td_graddft.reference import GridReference, RestrictedMoleculeReference
from td_graddft.scf import RKSConfig, run_rks_from_integrals
from td_graddft import tdscf
from td_graddft.spectra import HARTREE_TO_EV

jax.config.update("jax_enable_x64", True)


@dataclass
class GPUStats:
    mean_util: float
    max_util: float
    max_mem_mib: float


class GPUMonitor:
    def __init__(self, gpu_index: int, interval_s: float = 0.5):
        self.gpu_index = int(gpu_index)
        self.interval_s = float(interval_s)
        self.samples: list[tuple[float, float, float]] = []
        self._running = False
        self._thread: threading.Thread | None = None

    def _query(self) -> tuple[float, float] | None:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(self.gpu_index),
                ],
                text=True,
                timeout=3.0,
            ).strip()
        except Exception:
            return None
        if not out:
            return None
        first = out.splitlines()[0]
        parts = [x.strip() for x in first.split(",")]
        if len(parts) < 2:
            return None
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            return None

    def _loop(self) -> None:
        while self._running:
            q = self._query()
            if q is not None:
                util, mem = q
                self.samples.append((time.perf_counter(), util, mem))
            time.sleep(self.interval_s)

    def start(self) -> None:
        self.samples = []
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> GPUStats:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if not self.samples:
            return GPUStats(mean_util=float("nan"), max_util=float("nan"), max_mem_mib=float("nan"))
        utils = np.array([s[1] for s in self.samples], dtype=float)
        mems = np.array([s[2] for s in self.samples], dtype=float)
        return GPUStats(
            mean_util=float(np.mean(utils)),
            max_util=float(np.max(utils)),
            max_mem_mib=float(np.max(mems)),
        )


class SemilocalResponseFunctional:
    def __init__(self, xc_spec: str):
        self.xc_spec = str(xc_spec)
        self.exact_exchange_fraction = float(hybrid_coeff(self.xc_spec))
        self.response_feature_kind = str(xc_type(self.xc_spec))

    def grid_response_tensor(self, molecule) -> jnp.ndarray:
        features, grad_rho = restricted_grid_features_with_gradients(molecule)
        tau = features.tau_a + features.tau_b
        _, tensor = eval_xc_response_tensor(
            self.xc_spec,
            features.rho,
            grad=grad_rho,
            tau=tau,
        )
        return tensor


@dataclass
class BenchRow:
    n_carbon: int
    nao: int
    nocc: int
    nvir: int
    occ_keep: int
    vir_keep: int
    nstates: int
    cpu_scf_s: float
    cpu_tddft_s: float
    cpu_total_s: float
    cpu_e_tot_ha: float
    cpu_exc1_ev: float
    gpu_grid_s: float
    gpu_integrals_s: float
    gpu_scf_s: float
    gpu_tddft_s: float
    gpu_total_s: float
    gpu_e_tot_ha: float
    gpu_exc1_ev: float
    abs_e_tot_diff_ha: float
    abs_exc1_diff_ev: float
    gpu_util_mean_pct: float
    gpu_util_max_pct: float
    gpu_mem_max_mib: float
    cpu_ok: int
    gpu_ok: int
    note: str


def _polyene_atoms(n_carbon: int) -> list[tuple[str, tuple[float, float, float]]]:
    if n_carbon < 2:
        raise ValueError("n_carbon must be >= 2")

    c_c_double = 1.34
    c_c_single = 1.46
    c_h = 1.09
    z_wing = 0.90

    carbons: list[tuple[float, float, float]] = []
    x = 0.0
    carbons.append((x, 0.0, 0.0))
    for i in range(1, n_carbon):
        bond = c_c_double if (i - 1) % 2 == 0 else c_c_single
        x += bond
        carbons.append((x, 0.0, 0.0))

    atoms: list[tuple[str, tuple[float, float, float]]] = [("C", c) for c in carbons]

    x0 = carbons[0][0]
    xn = carbons[-1][0]
    atoms.extend(
        [
            ("H", (x0, +c_h, +z_wing)),
            ("H", (x0, +c_h, -z_wing)),
            ("H", (xn, -c_h, +z_wing)),
            ("H", (xn, -c_h, -z_wing)),
        ]
    )
    for i in range(1, n_carbon - 1):
        x_i = carbons[i][0]
        y_i = c_h if (i % 2 == 0) else -c_h
        atoms.append(("H", (x_i, y_i, 0.0)))

    return atoms


def _make_mol(n_carbon: int, basis: str, *, cart: bool = True):
    return gto.M(
        atom=_polyene_atoms(n_carbon),
        unit="Angstrom",
        basis=basis,
        charge=0,
        spin=0,
        cart=cart,
        verbose=0,
    )


def _choose_active_space(nocc: int, nvir: int, n_carbon: int, occ_factor: int, vir_factor: int):
    occ_keep = min(nocc, max(4, occ_factor * n_carbon))
    vir_keep = min(nvir, max(4, vir_factor * n_carbon))
    return int(occ_keep), int(vir_keep)


def _build_frozen_list(nocc: int, nmo: int, occ_keep: int, vir_keep: int):
    freeze_occ = list(range(0, max(0, nocc - occ_keep)))
    freeze_vir = list(range(min(nmo, nocc + vir_keep), nmo))
    return freeze_occ + freeze_vir


def _charge_center(mol) -> np.ndarray:
    charges = np.asarray(mol.atom_charges(), dtype=float)
    coords = np.asarray(mol.atom_coords(), dtype=float)
    return np.einsum("z,zr->r", charges, coords) / np.sum(charges)


def _estimate_eri_gib(nao: int) -> float:
    return (float(nao) ** 4 * 8.0) / (1024.0**3)


def _cpu_pipeline(
    n_carbon: int,
    *,
    basis: str,
    xc: str,
    grids_level: int,
    nstates_max: int,
    occ_factor: int,
    vir_factor: int,
) -> dict:
    mol = _make_mol(n_carbon, basis=basis, cart=True)
    mf = dft.RKS(mol).density_fit()
    mf.xc = xc
    mf.grids.level = grids_level
    mf.conv_tol = 1e-9
    mf.max_cycle = 120

    t0 = time.perf_counter()
    mf.kernel()
    cpu_scf_s = time.perf_counter() - t0
    if not mf.converged:
        raise RuntimeError(f"CPU SCF did not converge at C={n_carbon}")

    nmo = int(mf.mo_occ.size)
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 1e-8))
    nvir = int(nmo - nocc)
    occ_keep, vir_keep = _choose_active_space(nocc, nvir, n_carbon, occ_factor, vir_factor)
    nov = occ_keep * vir_keep
    nstates = max(1, min(int(nstates_max), max(1, nov - 1)))
    frozen = _build_frozen_list(nocc, nmo, occ_keep, vir_keep)

    td = mf.TDDFT()
    td.nstates = nstates
    td.frozen = frozen
    td.max_cycle = 100
    t1 = time.perf_counter()
    td.kernel()
    cpu_tddft_s = time.perf_counter() - t1

    e = np.asarray(td.e, dtype=float).reshape(-1)
    exc1_ev = float(e[0] * HARTREE_TO_EV) if e.size > 0 else float("nan")
    return {
        "nao": int(mol.nao_nr()),
        "nocc": nocc,
        "nvir": nvir,
        "occ_keep": occ_keep,
        "vir_keep": vir_keep,
        "nstates": nstates,
        "scf_s": float(cpu_scf_s),
        "tddft_s": float(cpu_tddft_s),
        "total_s": float(cpu_scf_s + cpu_tddft_s),
        "e_tot_ha": float(mf.e_tot),
        "exc1_ev": exc1_ev,
        "ok": 1,
    }


def _gpu_pipeline_full_jax(
    n_carbon: int,
    *,
    basis: str,
    xc_spec: str,
    grids_level: int,
    nstates: int,
    occ_keep: int,
    vir_keep: int,
    jax_basis_max_l: int,
    max_eri_gib: float,
    gpu_index: int,
) -> dict:
    monitor = GPUMonitor(gpu_index=gpu_index, interval_s=0.5)
    monitor.start()
    note = ""
    cpu_devices = jax.devices("cpu")
    gpu_devices = jax.devices("gpu")
    if not cpu_devices:
        raise RuntimeError("No CPU device visible to JAX.")
    if not gpu_devices:
        raise RuntimeError("No GPU device visible to JAX.")
    cpu_device = cpu_devices[0]
    gpu_device = gpu_devices[0]

    try:
        mol = _make_mol(n_carbon, basis=basis, cart=True)
        nao = int(mol.nao_nr())
        est_eri_gib = _estimate_eri_gib(nao)
        if est_eri_gib > max_eri_gib:
            note = f"skip_gpu_dense_eri_estimate={est_eri_gib:.2f}GiB>limit={max_eri_gib:.2f}GiB"
            return {
                "grid_s": float("nan"),
                "integrals_s": float("nan"),
                "scf_s": float("nan"),
                "tddft_s": float("nan"),
                "total_s": float("nan"),
                "e_tot_ha": float("nan"),
                "exc1_ev": float("nan"),
                "ok": 0,
                "note": note,
            }

        # Force all quadrature + AO integral builds on CPU to avoid GPU stalls
        # in the current pure-JAX integral engine.
        with jax.default_device(cpu_device):
            t_grid0 = time.perf_counter()
            grids = dft.gen_grid.Grids(mol)
            grids.level = grids_level
            grids.build()
            coords = np.asarray(grids.coords)
            weights = np.asarray(grids.weights)
            ao = np.asarray(numint.eval_ao(mol, coords, deriv=0))
            ao_deriv1 = np.asarray(numint.eval_ao(mol, coords, deriv=1))
            with mol.with_common_orig(_charge_center(mol)):
                dipole = np.asarray(mol.intor_symmetric("int1e_r", comp=3))
            grid_s = time.perf_counter() - t_grid0

            t_int0 = time.perf_counter()
            basis_cart = basis_from_pyscf_mol_cart(mol, max_l=jax_basis_max_l)
            s_cpu = np.asarray(overlap_matrix(basis_cart), dtype=np.float64)
            h1e_cpu = np.asarray(build_hcore(basis_cart), dtype=np.float64)
            eri_cpu = np.asarray(eri_tensor(basis_cart), dtype=np.float64)
            integrals_s = time.perf_counter() - t_int0

        # Transfer heavy tensors to GPU and run SCF/TDDFT there.
        s = jax.device_put(jnp.asarray(s_cpu), gpu_device)
        h1e = jax.device_put(jnp.asarray(h1e_cpu), gpu_device)
        eri = jax.device_put(jnp.asarray(eri_cpu), gpu_device)
        ao_gpu = jax.device_put(jnp.asarray(ao), gpu_device)
        ao_deriv1_gpu = jax.device_put(jnp.asarray(ao_deriv1), gpu_device)
        weights_gpu = jax.device_put(jnp.asarray(weights), gpu_device)

        t_scf0 = time.perf_counter()
        with jax.default_device(gpu_device):
            rks = run_rks_from_integrals(
            overlap=s,
            hcore=h1e,
            eri=eri,
            nelectron=int(mol.nelectron),
            nuclear_repulsion=float(mol.energy_nuc()),
            ao=ao_gpu,
            ao_deriv1=ao_deriv1_gpu,
            grid_weights=weights_gpu,
            init_mo_coeff=None,
            init_mo_occ=None,
            init_mo_energy=None,
            config=RKSConfig(
                xc_spec=xc_spec,
                max_cycle=80,
                conv_tol=1e-9,
                conv_tol_density=1e-7,
                damping=0.10,
                density_floor=1e-12,
                potential_clip=20.0,
            ),
        )
        scf_s = time.perf_counter() - t_scf0
        if not rks.converged:
            raise RuntimeError(f"JAX-RKS did not converge at C={n_carbon}")

        mo_occ_full = np.asarray(rks.mo_occ, dtype=float)
        mo_energy_full = np.asarray(rks.mo_energy, dtype=float)
        mo_coeff_full = np.asarray(rks.mo_coeff, dtype=float)
        nocc_full = int(np.count_nonzero(mo_occ_full > 1e-8))
        nmo_full = int(mo_occ_full.shape[0])
        vir_keep = min(vir_keep, max(1, nmo_full - nocc_full))
        occ_keep = min(occ_keep, max(1, nocc_full))
        occ_idx = np.arange(nocc_full - occ_keep, nocc_full, dtype=int)
        vir_idx = np.arange(nocc_full, nocc_full + vir_keep, dtype=int)
        active_idx = np.concatenate([occ_idx, vir_idx], axis=0)

        mo_coeff_active = mo_coeff_full[:, active_idx]
        mo_energy_active = mo_energy_full[active_idx]
        mo_occ_active = np.zeros_like(mo_energy_active)
        mo_occ_active[: occ_idx.size] = 2.0

        dm_total = np.asarray(rks.density_matrix, dtype=float)
        ref = RestrictedMoleculeReference(
            ao=jnp.asarray(ao),
            grid=GridReference(weights=jnp.asarray(weights), coords=jnp.asarray(coords)),
            dipole_integrals=jnp.asarray(dipole),
            rep_tensor=jnp.asarray(eri_cpu),
            mo_coeff=jnp.stack([jnp.asarray(mo_coeff_active), jnp.asarray(mo_coeff_active)], axis=0),
            mo_occ=jnp.stack([jnp.asarray(0.5 * mo_occ_active), jnp.asarray(0.5 * mo_occ_active)], axis=0),
            mo_energy=jnp.stack([jnp.asarray(mo_energy_active), jnp.asarray(mo_energy_active)], axis=0),
            rdm1=jnp.stack([jnp.asarray(0.5 * dm_total), jnp.asarray(0.5 * dm_total)], axis=0),
            h1e=jnp.asarray(h1e_cpu),
            nuclear_repulsion=float(rks.nuclear_repulsion),
            overlap_matrix=jnp.asarray(s_cpu),
            ao_deriv1=jnp.asarray(ao_deriv1),
            mf_energy=float(rks.total_energy),
            exact_exchange_fraction=float(rks.exact_exchange_fraction),
        )

        ref = put_restricted_reference_on_device(ref, device=gpu_device)

        t_td0 = time.perf_counter()
        solver = tdscf.TDDFT(ref, xc_functional=xc_spec)
        result = solver.kernel(nstates=nstates)
        energies_au = np.asarray(result.excitation_energies, dtype=float).reshape(-1)
        osc = np.asarray(solver.oscillator_strength(), dtype=float).reshape(-1)
        _ = osc  # explicitly materialize to include sync cost in timing
        tddft_s = time.perf_counter() - t_td0
        exc1_ev = float(energies_au[0] * HARTREE_TO_EV) if energies_au.size > 0 else float("nan")

        total_s = grid_s + integrals_s + scf_s + tddft_s
        return {
            "grid_s": float(grid_s),
            "integrals_s": float(integrals_s),
            "scf_s": float(scf_s),
            "tddft_s": float(tddft_s),
            "total_s": float(total_s),
            "e_tot_ha": float(rks.total_energy),
            "exc1_ev": exc1_ev,
            "ok": 1,
            "note": "integrals_on_cpu;scf_tddft_on_gpu" if not note else note,
        }
    finally:
        stats = monitor.stop()
        if "stats_cache" not in locals():
            pass
        _gpu_stats_holder.append(stats)


_gpu_stats_holder: list[GPUStats] = []


def _plot_timing(path: Path, rows: list[BenchRow]) -> None:
    carbons = np.array([r.n_carbon for r in rows], dtype=float)
    cpu = np.array([r.cpu_total_s for r in rows], dtype=float)
    gpu = np.array([r.gpu_total_s for r in rows], dtype=float)
    gpu_ok = np.array([r.gpu_ok for r in rows], dtype=int) == 1

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(carbons, cpu, marker="o", lw=2, label="CPU full pipeline (PySCF SCF+TDDFT, 96 threads)")
    if np.any(gpu_ok):
        ax.plot(carbons[gpu_ok], gpu[gpu_ok], marker="s", lw=2, label="GPU full pipeline (JAX RKS+TDDFT)")
    ax.set_xlabel("Carbon atoms (C_n polyene)")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Full TDDFT Pipeline Timing vs Carbon Count")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)


def _plot_energy(path: Path, rows: list[BenchRow]) -> None:
    carbons = np.array([r.n_carbon for r in rows], dtype=float)
    cpu_e = np.array([r.cpu_e_tot_ha for r in rows], dtype=float)
    gpu_e = np.array([r.gpu_e_tot_ha for r in rows], dtype=float)
    cpu_ex = np.array([r.cpu_exc1_ev for r in rows], dtype=float)
    gpu_ex = np.array([r.gpu_exc1_ev for r in rows], dtype=float)
    gpu_ok = np.array([r.gpu_ok for r in rows], dtype=int) == 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax0, ax1 = axes

    ax0.plot(carbons, cpu_e, marker="o", lw=2, label="CPU E_tot")
    if np.any(gpu_ok):
        ax0.plot(carbons[gpu_ok], gpu_e[gpu_ok], marker="s", lw=2, label="GPU E_tot")
    ax0.set_xlabel("Carbon atoms")
    ax0.set_ylabel("Ground-state total energy (Ha)")
    ax0.grid(alpha=0.25)
    ax0.legend(frameon=False)

    ax1.plot(carbons, cpu_ex, marker="o", lw=2, label="CPU Exc1")
    if np.any(gpu_ok):
        ax1.plot(carbons[gpu_ok], gpu_ex[gpu_ok], marker="s", lw=2, label="GPU Exc1")
    ax1.set_xlabel("Carbon atoms")
    ax1.set_ylabel("First excitation energy (eV)")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False)

    fig.suptitle("Energy Statistics vs Carbon Count")
    fig.tight_layout()
    fig.savefig(path, dpi=180)


def _write_csv(path: Path, rows: list[BenchRow]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "n_carbon",
                "nao",
                "nocc",
                "nvir",
                "occ_keep",
                "vir_keep",
                "nstates",
                "cpu_scf_s",
                "cpu_tddft_s",
                "cpu_total_s",
                "cpu_e_tot_ha",
                "cpu_exc1_ev",
                "gpu_grid_s",
                "gpu_integrals_s",
                "gpu_scf_s",
                "gpu_tddft_s",
                "gpu_total_s",
                "gpu_e_tot_ha",
                "gpu_exc1_ev",
                "abs_e_tot_diff_ha",
                "abs_exc1_diff_ev",
                "gpu_util_mean_pct",
                "gpu_util_max_pct",
                "gpu_mem_max_mib",
                "cpu_ok",
                "gpu_ok",
                "note",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.n_carbon,
                    r.nao,
                    r.nocc,
                    r.nvir,
                    r.occ_keep,
                    r.vir_keep,
                    r.nstates,
                    f"{r.cpu_scf_s:.6f}",
                    f"{r.cpu_tddft_s:.6f}",
                    f"{r.cpu_total_s:.6f}",
                    f"{r.cpu_e_tot_ha:.10f}",
                    f"{r.cpu_exc1_ev:.8f}" if np.isfinite(r.cpu_exc1_ev) else "",
                    f"{r.gpu_grid_s:.6f}" if np.isfinite(r.gpu_grid_s) else "",
                    f"{r.gpu_integrals_s:.6f}" if np.isfinite(r.gpu_integrals_s) else "",
                    f"{r.gpu_scf_s:.6f}" if np.isfinite(r.gpu_scf_s) else "",
                    f"{r.gpu_tddft_s:.6f}" if np.isfinite(r.gpu_tddft_s) else "",
                    f"{r.gpu_total_s:.6f}" if np.isfinite(r.gpu_total_s) else "",
                    f"{r.gpu_e_tot_ha:.10f}" if np.isfinite(r.gpu_e_tot_ha) else "",
                    f"{r.gpu_exc1_ev:.8f}" if np.isfinite(r.gpu_exc1_ev) else "",
                    f"{r.abs_e_tot_diff_ha:.8f}" if np.isfinite(r.abs_e_tot_diff_ha) else "",
                    f"{r.abs_exc1_diff_ev:.8f}" if np.isfinite(r.abs_exc1_diff_ev) else "",
                    f"{r.gpu_util_mean_pct:.3f}" if np.isfinite(r.gpu_util_mean_pct) else "",
                    f"{r.gpu_util_max_pct:.3f}" if np.isfinite(r.gpu_util_max_pct) else "",
                    f"{r.gpu_mem_max_mib:.1f}" if np.isfinite(r.gpu_mem_max_mib) else "",
                    r.cpu_ok,
                    r.gpu_ok,
                    r.note,
                ]
            )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--carbons", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    p.add_argument("--basis", type=str, default="6-31g")
    p.add_argument("--xc", type=str, default="b3lyp")
    p.add_argument("--jax-xc-spec", type=str, default="b3lyp")
    p.add_argument("--grids-level", type=int, default=0)
    p.add_argument("--nstates-max", type=int, default=8)
    p.add_argument("--occ-factor", type=int, default=2)
    p.add_argument("--vir-factor", type=int, default=2)
    p.add_argument("--jax-basis-max-l", type=int, default=3)
    p.add_argument("--max-eri-gib", type=float, default=24.0)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--outdir", type=str, default="outputs/polyene_full_pipeline_bench")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    rows: list[BenchRow] = []
    for n_carbon in args.carbons:
        note = ""
        try:
            cpu = _cpu_pipeline(
                n_carbon,
                basis=args.basis,
                xc=args.xc,
                grids_level=args.grids_level,
                nstates_max=args.nstates_max,
                occ_factor=args.occ_factor,
                vir_factor=args.vir_factor,
            )
            cpu_ok = 1
        except Exception as exc:
            cpu_ok = 0
            cpu = {
                "nao": -1,
                "nocc": -1,
                "nvir": -1,
                "occ_keep": -1,
                "vir_keep": -1,
                "nstates": -1,
                "scf_s": float("nan"),
                "tddft_s": float("nan"),
                "total_s": float("nan"),
                "e_tot_ha": float("nan"),
                "exc1_ev": float("nan"),
                "ok": 0,
            }
            note = f"cpu_failed:{type(exc).__name__}:{exc}"

        gpu = {
            "grid_s": float("nan"),
            "integrals_s": float("nan"),
            "scf_s": float("nan"),
            "tddft_s": float("nan"),
            "total_s": float("nan"),
            "e_tot_ha": float("nan"),
            "exc1_ev": float("nan"),
            "ok": 0,
            "note": "",
        }
        stats = GPUStats(mean_util=float("nan"), max_util=float("nan"), max_mem_mib=float("nan"))
        try:
            _gpu_stats_holder.clear()
            if cpu_ok == 1:
                gpu = _gpu_pipeline_full_jax(
                    n_carbon,
                    basis=args.basis,
                    xc_spec=args.jax_xc_spec,
                    grids_level=args.grids_level,
                    nstates=int(cpu["nstates"]),
                    occ_keep=int(cpu["occ_keep"]),
                    vir_keep=int(cpu["vir_keep"]),
                    jax_basis_max_l=args.jax_basis_max_l,
                    max_eri_gib=args.max_eri_gib,
                    gpu_index=args.gpu_index,
                )
            else:
                gpu["note"] = "skip_gpu_due_cpu_failure"
            if _gpu_stats_holder:
                stats = _gpu_stats_holder[-1]
        except Exception as exc:
            gpu["ok"] = 0
            gpu["note"] = f"gpu_failed:{type(exc).__name__}:{exc}"

        merged_note = note
        if gpu.get("note"):
            merged_note = f"{merged_note} | {gpu['note']}" if merged_note else str(gpu["note"])

        abs_e = (
            abs(float(cpu["e_tot_ha"]) - float(gpu["e_tot_ha"]))
            if np.isfinite(cpu["e_tot_ha"]) and np.isfinite(gpu["e_tot_ha"])
            else float("nan")
        )
        abs_exc = (
            abs(float(cpu["exc1_ev"]) - float(gpu["exc1_ev"]))
            if np.isfinite(cpu["exc1_ev"]) and np.isfinite(gpu["exc1_ev"])
            else float("nan")
        )

        row = BenchRow(
            n_carbon=n_carbon,
            nao=int(cpu["nao"]),
            nocc=int(cpu["nocc"]),
            nvir=int(cpu["nvir"]),
            occ_keep=int(cpu["occ_keep"]),
            vir_keep=int(cpu["vir_keep"]),
            nstates=int(cpu["nstates"]),
            cpu_scf_s=float(cpu["scf_s"]),
            cpu_tddft_s=float(cpu["tddft_s"]),
            cpu_total_s=float(cpu["total_s"]),
            cpu_e_tot_ha=float(cpu["e_tot_ha"]),
            cpu_exc1_ev=float(cpu["exc1_ev"]),
            gpu_grid_s=float(gpu["grid_s"]),
            gpu_integrals_s=float(gpu["integrals_s"]),
            gpu_scf_s=float(gpu["scf_s"]),
            gpu_tddft_s=float(gpu["tddft_s"]),
            gpu_total_s=float(gpu["total_s"]),
            gpu_e_tot_ha=float(gpu["e_tot_ha"]),
            gpu_exc1_ev=float(gpu["exc1_ev"]),
            abs_e_tot_diff_ha=float(abs_e),
            abs_exc1_diff_ev=float(abs_exc),
            gpu_util_mean_pct=float(stats.mean_util),
            gpu_util_max_pct=float(stats.max_util),
            gpu_mem_max_mib=float(stats.max_mem_mib),
            cpu_ok=cpu_ok,
            gpu_ok=int(gpu["ok"]),
            note=merged_note,
        )
        rows.append(row)

        print(
            f"[C={n_carbon}] CPU(total={row.cpu_total_s:.2f}s, E={row.cpu_e_tot_ha:.6f}Ha, Exc1={row.cpu_exc1_ev:.4f}eV) "
            f"GPU(total={row.gpu_total_s:.2f}s, E={row.gpu_e_tot_ha:.6f}Ha, Exc1={row.gpu_exc1_ev:.4f}eV, "
            f"util_mean/max={row.gpu_util_mean_pct:.1f}/{row.gpu_util_max_pct:.1f}%) "
            f"note={row.note}"
        )

    csv_path = outdir / "polyene_full_pipeline_stats.csv"
    time_png = outdir / "polyene_full_pipeline_timing.png"
    energy_png = outdir / "polyene_full_pipeline_energy.png"
    _write_csv(csv_path, rows)
    _plot_timing(time_png, rows)
    _plot_energy(energy_png, rows)
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote timing PNG: {time_png}")
    print(f"Wrote energy PNG: {energy_png}")


if __name__ == "__main__":
    main()
