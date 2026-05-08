from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib
import numpy as np
from pyscf import dft, gto

matplotlib.use("Agg")
import matplotlib.pyplot as plt


AU_TO_EV = 27.211386245988


@dataclass(frozen=True)
class GpuStats:
    samples: int
    util_mean_pct: float
    util_max_pct: float
    mem_start_mib: float
    mem_mean_mib: float
    mem_max_mib: float
    mem_delta_peak_mib: float


@dataclass(frozen=True)
class AceneResult:
    rings: int
    name: str
    formula: str
    nao: int
    nmo: int
    nocc: int
    nvir: int
    occ_keep: int
    vir_keep: int
    td_dim: int
    nstates: int
    scf_s: float
    ab_build_s: float
    cpu_pyscf_tddft_s: float
    cpu_dense_casida_s: float
    gpu_transfer_s: float
    gpu_compile_warmup_s: float
    gpu_solve_mean_s: float
    gpu_profile_elapsed_s: float
    gpu_profile_repeats: int
    cpu_pyscf_full_s: float
    cpu_dense_full_s: float
    gpu_full_s_excl_compile: float
    gpu_first_run_full_s: float
    gpu_speedup_over_pyscf_full: float
    gpu_speedup_over_cpu_dense_full: float
    gpu_core_speedup_over_cpu_dense: float
    e_tot_ha: float
    cpu_pyscf_exc1_ev: float
    cpu_dense_exc1_ev: float
    gpu_exc1_ev: float
    gpu_exc1_abs_diff_ev_vs_cpu_dense: float
    gpu_max_abs_diff_ev_vs_cpu_dense: float
    gpu_exc1_abs_diff_ev_vs_pyscf: float
    gpu_util_mean_pct: float
    gpu_util_max_pct: float
    gpu_mem_start_mib: float
    gpu_mem_mean_mib: float
    gpu_mem_max_mib: float
    gpu_mem_delta_peak_mib: float
    gpu_monitor_samples: int
    physical_gpu_index: int


def parse_nvidia_smi_sample(raw: str) -> tuple[float, float]:
    parts = str(raw).strip().split(",")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse nvidia-smi sample: {raw!r}")

    def _number(text: str) -> float:
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)
        if match is None:
            raise ValueError(f"Cannot parse numeric field: {text!r}")
        return float(match.group(0))

    return _number(parts[0]), _number(parts[1])


class NvidiaSmiSampler:
    def __init__(self, *, gpu_index: int, interval_s: float):
        self.gpu_index = int(gpu_index)
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[tuple[float, float, float]] = []

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> GpuStats:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 4.0 * self.interval_s))
        return self.stats()

    def stats(self) -> GpuStats:
        if not self.samples:
            nan = float("nan")
            return GpuStats(
                samples=0,
                util_mean_pct=nan,
                util_max_pct=nan,
                mem_start_mib=nan,
                mem_mean_mib=nan,
                mem_max_mib=nan,
                mem_delta_peak_mib=nan,
            )
        util = np.asarray([sample[1] for sample in self.samples], dtype=float)
        mem = np.asarray([sample[2] for sample in self.samples], dtype=float)
        mem_start = float(mem[0])
        mem_max = float(np.max(mem))
        return GpuStats(
            samples=int(len(self.samples)),
            util_mean_pct=float(np.mean(util)),
            util_max_pct=float(np.max(util)),
            mem_start_mib=mem_start,
            mem_mean_mib=float(np.mean(mem)),
            mem_max_mib=mem_max,
            mem_delta_peak_mib=float(mem_max - mem_start),
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = self._query_once()
            if sample is not None:
                util_pct, mem_mib = sample
                self.samples.append((time.perf_counter(), util_pct, mem_mib))
            self._stop.wait(self.interval_s)

    def _query_once(self) -> tuple[float, float] | None:
        cmd = [
            "nvidia-smi",
            "-i",
            str(self.gpu_index),
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if completed.returncode != 0:
            return None
        for line in completed.stdout.splitlines():
            if line.strip():
                try:
                    return parse_nvidia_smi_sample(line)
                except ValueError:
                    return None
        return None


def linear_acene_atoms(n_rings: int, *, c_c: float = 1.397, c_h: float = 1.09):
    if n_rings < 1:
        raise ValueError("n_rings must be >= 1")

    centers = [(math.sqrt(3.0) * c_c * ring, 0.0) for ring in range(n_rings)]
    vertex_offsets = [
        (0.0, c_c),
        (math.sqrt(3.0) * 0.5 * c_c, 0.5 * c_c),
        (math.sqrt(3.0) * 0.5 * c_c, -0.5 * c_c),
        (0.0, -c_c),
        (-math.sqrt(3.0) * 0.5 * c_c, -0.5 * c_c),
        (-math.sqrt(3.0) * 0.5 * c_c, 0.5 * c_c),
    ]

    vertices: list[np.ndarray] = []
    neighbors: list[set[int]] = []
    index_by_key: dict[tuple[int, int], int] = {}

    def get_index(point: np.ndarray) -> int:
        key = (int(round(point[0] * 1_000_000)), int(round(point[1] * 1_000_000)))
        idx = index_by_key.get(key)
        if idx is not None:
            return idx
        idx = len(vertices)
        index_by_key[key] = idx
        vertices.append(point)
        neighbors.append(set())
        return idx

    for cx, cy in centers:
        ring = []
        for dx, dy in vertex_offsets:
            ring.append(get_index(np.asarray([cx + dx, cy + dy], dtype=float)))
        for i in range(6):
            a = ring[i]
            b = ring[(i + 1) % 6]
            neighbors[a].add(b)
            neighbors[b].add(a)

    atoms: list[tuple[str, tuple[float, float, float]]] = []
    for point in vertices:
        atoms.append(("C", (float(point[0]), float(point[1]), 0.0)))

    for idx, point in enumerate(vertices):
        if len(neighbors[idx]) != 2:
            continue
        bonded = np.asarray([vertices[j] for j in neighbors[idx]], dtype=float)
        inward = (bonded[0] - point) + (bonded[1] - point)
        norm = float(np.linalg.norm(inward))
        if norm <= 1e-12:
            continue
        outward = -inward / norm
        h = point + c_h * outward
        atoms.append(("H", (float(h[0]), float(h[1]), 0.0)))
    return atoms


def acene_name(n_rings: int) -> str:
    names = {
        1: "benzene",
        2: "naphthalene",
        3: "anthracene",
        4: "tetracene",
        5: "pentacene",
    }
    return names.get(int(n_rings), f"acene{int(n_rings)}")


def acene_formula(n_rings: int) -> str:
    return f"C{4 * int(n_rings) + 2}H{2 * int(n_rings) + 4}"


def make_molecule(n_rings: int, *, basis: str, cart: bool):
    mol = gto.Mole()
    mol.atom = linear_acene_atoms(n_rings)
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.spin = 0
    mol.charge = 0
    mol.cart = bool(cart)
    mol.verbose = 0
    mol.build()
    return mol


def choose_active_space(
    nocc: int,
    nvir: int,
    rings: int,
    *,
    occ_factor: int,
    vir_factor: int,
    active_min: int,
) -> tuple[int, int]:
    occ_keep = min(nocc, max(int(active_min), int(occ_factor) * int(rings)))
    vir_keep = min(nvir, max(int(active_min), int(vir_factor) * int(rings)))
    return occ_keep, vir_keep


def build_frozen_list(nocc: int, nmo: int, occ_keep: int, vir_keep: int) -> list[int]:
    freeze_occ = list(range(0, max(0, nocc - occ_keep)))
    freeze_vir = list(range(min(nmo, nocc + vir_keep), nmo))
    return freeze_occ + freeze_vir


def _symmetrize_np(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def casida_eigs_numpy(a_matrix: np.ndarray, b_matrix: np.ndarray) -> np.ndarray:
    a = _symmetrize_np(np.asarray(a_matrix, dtype=np.float64))
    b = _symmetrize_np(np.asarray(b_matrix, dtype=np.float64))
    apb = _symmetrize_np(a + b)
    amb = _symmetrize_np(a - b)
    amb = amb + 1e-10 * np.eye(amb.shape[0], dtype=np.float64)
    evals, evecs = np.linalg.eigh(amb)
    evals = np.clip(evals, 1e-10, None)
    sqrt_amb = (evecs * np.sqrt(evals)) @ evecs.T
    casida = _symmetrize_np(sqrt_amb @ apb @ sqrt_amb)
    w2, _ = np.linalg.eigh(casida)
    return np.sqrt(np.clip(w2, 0.0, None))


@jax.jit
def casida_eigs_jax(a_matrix: jnp.ndarray, b_matrix: jnp.ndarray) -> jnp.ndarray:
    a = 0.5 * (a_matrix + a_matrix.T)
    b = 0.5 * (b_matrix + b_matrix.T)
    apb = 0.5 * ((a + b) + (a + b).T)
    amb = 0.5 * ((a - b) + (a - b).T)
    eye = jnp.eye(a.shape[0], dtype=a.dtype)
    amb = amb + 1e-10 * eye
    evals, evecs = jnp.linalg.eigh(amb)
    evals = jnp.clip(evals, 1e-10, None)
    sqrt_amb = (evecs * jnp.sqrt(evals)) @ evecs.T
    casida = sqrt_amb @ apb @ sqrt_amb
    casida = 0.5 * (casida + casida.T)
    w2, _ = jnp.linalg.eigh(casida)
    return jnp.sqrt(jnp.clip(w2, 0.0, None))


def _positive_first(values: np.ndarray, nstates: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    positive = np.sort(arr[arr > 1e-10])
    if positive.size >= int(nstates):
        return positive[: int(nstates)]
    out = np.full((int(nstates),), np.nan, dtype=float)
    out[: positive.size] = positive
    return out


def _safe_speedup(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return float(numerator / denominator)


def benchmark_one(
    rings: int,
    *,
    basis: str,
    xc: str,
    grids_level: int,
    nstates_max: int,
    occ_factor: int,
    vir_factor: int,
    active_min: int,
    scf_max_cycle: int,
    scf_conv_tol: float,
    tddft_max_cycle: int,
    cart: bool,
    density_fit: bool,
    gpu_index: int,
    gpu_sample_interval_s: float,
    gpu_repeats: int,
    gpu_profile_min_s: float,
    run_pyscf_tddft: bool,
) -> AceneResult:
    mol = make_molecule(rings, basis=basis, cart=cart)
    mf = dft.RKS(mol)
    if bool(density_fit):
        mf = mf.density_fit()
    mf.xc = str(xc)
    mf.grids.level = int(grids_level)
    mf.conv_tol = float(scf_conv_tol)
    mf.max_cycle = int(scf_max_cycle)

    scf_t0 = time.perf_counter()
    mf.kernel()
    scf_s = time.perf_counter() - scf_t0
    if not mf.converged:
        raise RuntimeError(f"SCF did not converge for rings={rings}.")

    nmo = int(mf.mo_occ.size)
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 1e-8))
    nvir = int(nmo - nocc)
    occ_keep, vir_keep = choose_active_space(
        nocc,
        nvir,
        rings,
        occ_factor=occ_factor,
        vir_factor=vir_factor,
        active_min=active_min,
    )
    frozen = build_frozen_list(nocc, nmo, occ_keep, vir_keep)
    td_dim = int(occ_keep * vir_keep)
    nstates = max(1, min(int(nstates_max), max(1, td_dim - 1)))

    cpu_pyscf_tddft_s = float("nan")
    cpu_pyscf_exc = np.full((nstates,), np.nan, dtype=float)
    if bool(run_pyscf_tddft):
        td = mf.TDDFT()
        td.nstates = int(nstates)
        td.frozen = frozen
        td.max_cycle = int(tddft_max_cycle)
        try:
            cpu_t0 = time.perf_counter()
            td.kernel()
            cpu_pyscf_tddft_s = time.perf_counter() - cpu_t0
            cpu_pyscf_exc = _positive_first(np.asarray(td.e, dtype=float), nstates) * AU_TO_EV
        except Exception:
            cpu_pyscf_tddft_s = float("nan")
            cpu_pyscf_exc = np.full((nstates,), np.nan, dtype=float)

    td_ab = mf.TDDFT()
    td_ab.frozen = frozen
    ab_t0 = time.perf_counter()
    a_raw, b_raw = td_ab.get_ab()
    ab_build_s = time.perf_counter() - ab_t0
    a_matrix = np.asarray(a_raw.reshape(td_dim, td_dim), dtype=np.float64)
    b_matrix = np.asarray(b_raw.reshape(td_dim, td_dim), dtype=np.float64)
    if (not np.isfinite(a_matrix).all()) or (not np.isfinite(b_matrix).all()):
        raise FloatingPointError(f"Non-finite A/B matrix entries for rings={rings}.")

    cpu_dense_t0 = time.perf_counter()
    cpu_dense_w_au = casida_eigs_numpy(a_matrix, b_matrix)
    cpu_dense_casida_s = time.perf_counter() - cpu_dense_t0
    cpu_dense_exc = _positive_first(cpu_dense_w_au, nstates) * AU_TO_EV

    gpu_devices = jax.devices("gpu")
    if not gpu_devices:
        raise RuntimeError("No JAX GPU device is visible. Set CUDA_VISIBLE_DEVICES to physical GPU3.")
    device = gpu_devices[0]

    with NvidiaSmiSampler(gpu_index=gpu_index, interval_s=gpu_sample_interval_s) as sampler:
        with jax.default_device(device):
            transfer_t0 = time.perf_counter()
            a_dev = jax.device_put(jnp.asarray(a_matrix), device)
            b_dev = jax.device_put(jnp.asarray(b_matrix), device)
            jax.block_until_ready(a_dev)
            jax.block_until_ready(b_dev)
            gpu_transfer_s = time.perf_counter() - transfer_t0

            compile_t0 = time.perf_counter()
            compiled = casida_eigs_jax.lower(a_dev, b_dev).compile()
            warm = compiled(a_dev, b_dev)
            jax.block_until_ready(warm)
            gpu_compile_warmup_s = time.perf_counter() - compile_t0

            solve_times: list[float] = []
            last_w = warm
            profile_t0 = time.perf_counter()
            while (
                len(solve_times) < int(gpu_repeats)
                or time.perf_counter() - profile_t0 < float(gpu_profile_min_s)
            ):
                solve_t0 = time.perf_counter()
                last_w = compiled(a_dev, b_dev)
                jax.block_until_ready(last_w)
                solve_times.append(time.perf_counter() - solve_t0)
            gpu_profile_elapsed_s = time.perf_counter() - profile_t0
            gpu_w_au = np.asarray(last_w, dtype=float)
        gpu_stats = sampler.stats()

    gpu_solve_mean_s = float(np.mean(np.asarray(solve_times, dtype=float)))
    gpu_exc = _positive_first(gpu_w_au, nstates) * AU_TO_EV
    finite_dense_gpu = np.isfinite(cpu_dense_exc) & np.isfinite(gpu_exc)
    if np.any(finite_dense_gpu):
        gpu_max_abs_diff_ev_vs_cpu_dense = float(
            np.max(np.abs(cpu_dense_exc[finite_dense_gpu] - gpu_exc[finite_dense_gpu]))
        )
    else:
        gpu_max_abs_diff_ev_vs_cpu_dense = float("nan")

    cpu_pyscf_full_s = scf_s + cpu_pyscf_tddft_s if np.isfinite(cpu_pyscf_tddft_s) else float("nan")
    cpu_dense_full_s = scf_s + ab_build_s + cpu_dense_casida_s
    gpu_full_s_excl_compile = scf_s + ab_build_s + gpu_transfer_s + gpu_solve_mean_s
    gpu_first_run_full_s = gpu_full_s_excl_compile + gpu_compile_warmup_s

    return AceneResult(
        rings=int(rings),
        name=acene_name(rings),
        formula=acene_formula(rings),
        nao=int(mol.nao_nr()),
        nmo=nmo,
        nocc=nocc,
        nvir=nvir,
        occ_keep=int(occ_keep),
        vir_keep=int(vir_keep),
        td_dim=td_dim,
        nstates=int(nstates),
        scf_s=float(scf_s),
        ab_build_s=float(ab_build_s),
        cpu_pyscf_tddft_s=float(cpu_pyscf_tddft_s),
        cpu_dense_casida_s=float(cpu_dense_casida_s),
        gpu_transfer_s=float(gpu_transfer_s),
        gpu_compile_warmup_s=float(gpu_compile_warmup_s),
        gpu_solve_mean_s=float(gpu_solve_mean_s),
        gpu_profile_elapsed_s=float(gpu_profile_elapsed_s),
        gpu_profile_repeats=int(len(solve_times)),
        cpu_pyscf_full_s=float(cpu_pyscf_full_s),
        cpu_dense_full_s=float(cpu_dense_full_s),
        gpu_full_s_excl_compile=float(gpu_full_s_excl_compile),
        gpu_first_run_full_s=float(gpu_first_run_full_s),
        gpu_speedup_over_pyscf_full=_safe_speedup(cpu_pyscf_full_s, gpu_full_s_excl_compile),
        gpu_speedup_over_cpu_dense_full=_safe_speedup(cpu_dense_full_s, gpu_full_s_excl_compile),
        gpu_core_speedup_over_cpu_dense=_safe_speedup(cpu_dense_casida_s, gpu_solve_mean_s),
        e_tot_ha=float(mf.e_tot),
        cpu_pyscf_exc1_ev=float(cpu_pyscf_exc[0]),
        cpu_dense_exc1_ev=float(cpu_dense_exc[0]),
        gpu_exc1_ev=float(gpu_exc[0]),
        gpu_exc1_abs_diff_ev_vs_cpu_dense=float(abs(cpu_dense_exc[0] - gpu_exc[0])),
        gpu_max_abs_diff_ev_vs_cpu_dense=float(gpu_max_abs_diff_ev_vs_cpu_dense),
        gpu_exc1_abs_diff_ev_vs_pyscf=float(abs(cpu_pyscf_exc[0] - gpu_exc[0])),
        gpu_util_mean_pct=float(gpu_stats.util_mean_pct),
        gpu_util_max_pct=float(gpu_stats.util_max_pct),
        gpu_mem_start_mib=float(gpu_stats.mem_start_mib),
        gpu_mem_mean_mib=float(gpu_stats.mem_mean_mib),
        gpu_mem_max_mib=float(gpu_stats.mem_max_mib),
        gpu_mem_delta_peak_mib=float(gpu_stats.mem_delta_peak_mib),
        gpu_monitor_samples=int(gpu_stats.samples),
        physical_gpu_index=int(gpu_index),
    )


def write_csv(path: Path, rows: list[AceneResult]) -> None:
    columns = [field.name for field in fields(AceneResult)]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([getattr(row, column) for column in columns])


def write_json(path: Path, *, args: argparse.Namespace, rows: list[AceneResult]) -> None:
    with path.open("w") as f:
        json.dump(
            {
                "config": vars(args),
                "rows": [asdict(row) for row in rows],
            },
            f,
            indent=2,
        )


def plot_time_curve(path: Path, rows: list[AceneResult], *, title: str) -> None:
    rings = np.asarray([row.rings for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(rings, [row.cpu_pyscf_full_s for row in rows], marker="o", lw=2, label="PySCF full TDDFT")
    ax.plot(rings, [row.cpu_dense_full_s for row in rows], marker="^", lw=2, label="CPU dense Casida full")
    ax.plot(
        rings,
        [row.gpu_full_s_excl_compile for row in rows],
        marker="s",
        lw=2,
        label="GPU dense Casida full (no compile)",
    )
    ax.set_xlabel("Number of fused benzene rings")
    ax.set_ylabel("Wall time including SCF (s)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_speedup_curve(path: Path, rows: list[AceneResult], *, title: str) -> None:
    rings = np.asarray([row.rings for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(
        rings,
        [row.gpu_speedup_over_pyscf_full for row in rows],
        marker="o",
        lw=2,
        label="GPU full vs PySCF full",
    )
    ax.plot(
        rings,
        [row.gpu_speedup_over_cpu_dense_full for row in rows],
        marker="^",
        lw=2,
        label="GPU full vs CPU dense full",
    )
    ax.plot(
        rings,
        [row.gpu_core_speedup_over_cpu_dense for row in rows],
        marker="s",
        lw=2,
        label="GPU Casida core vs CPU dense core",
    )
    ax.axhline(1.0, color="0.4", lw=1, ls="--")
    ax.set_xlabel("Number of fused benzene rings")
    ax.set_ylabel("Speedup (x)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_gpu_stats(path: Path, rows: list[AceneResult], *, title: str) -> None:
    rings = np.asarray([row.rings for row in rows], dtype=float)
    fig, ax0 = plt.subplots(figsize=(9, 5.2))
    ax0.plot(rings, [row.gpu_util_mean_pct for row in rows], marker="o", lw=2, label="GPU util mean (%)")
    ax0.plot(rings, [row.gpu_util_max_pct for row in rows], marker="s", lw=2, label="GPU util max (%)")
    ax0.set_xlabel("Number of fused benzene rings")
    ax0.set_ylabel("GPU utilization (%)")
    ax0.set_ylim(bottom=0.0)
    ax0.grid(alpha=0.25)
    ax1 = ax0.twinx()
    ax1.plot(
        rings,
        [row.gpu_mem_max_mib for row in rows],
        marker="^",
        lw=2,
        color="tab:red",
        label="GPU memory max (MiB)",
    )
    ax1.set_ylabel("GPU memory used (MiB)")
    lines0, labels0 = ax0.get_legend_handles_labels()
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax0.legend(lines0 + lines1, labels0 + labels1, frameon=False, loc="upper left")
    ax0.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_energy_alignment(path: Path, rows: list[AceneResult], *, title: str) -> None:
    rings = np.asarray([row.rings for row in rows], dtype=float)
    diff = np.asarray([row.gpu_max_abs_diff_ev_vs_cpu_dense for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.semilogy(rings, diff, marker="o", lw=2)
    ax.set_xlabel("Number of fused benzene rings")
    ax.set_ylabel("Max |GPU - CPU dense| excitation diff (eV)")
    ax.set_title(title)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark acene TDDFT CPU/GPU timing, energy alignment, and physical GPU telemetry."
    )
    parser.add_argument("--min-rings", type=int, default=1)
    parser.add_argument("--max-rings", type=int, default=5)
    parser.add_argument("--basis", default="6-31g*")
    parser.add_argument("--xc", default="pbe0")
    parser.add_argument("--grids-level", type=int, default=0)
    parser.add_argument("--nstates-max", type=int, default=5)
    parser.add_argument("--occ-factor", type=int, default=8)
    parser.add_argument("--vir-factor", type=int, default=8)
    parser.add_argument("--active-min", type=int, default=8)
    parser.add_argument("--scf-max-cycle", type=int, default=120)
    parser.add_argument("--scf-conv-tol", type=float, default=1e-9)
    parser.add_argument("--tddft-max-cycle", type=int, default=100)
    parser.add_argument("--cart", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--density-fit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-pyscf-tddft", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--physical-gpu-index",
        type=int,
        default=int(os.environ.get("TD_GRADDFT_PHYSICAL_GPU_INDEX", "3")),
    )
    parser.add_argument("--gpu-sample-interval-s", type=float, default=0.05)
    parser.add_argument("--gpu-repeats", type=int, default=3)
    parser.add_argument("--gpu-profile-min-s", type=float, default=1.0)
    parser.add_argument("--outdir", default="outputs/acene_tddft_gpu_curve")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.min_rings) < 1 or int(args.max_rings) < int(args.min_rings):
        raise ValueError("Require 1 <= min_rings <= max_rings.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}")
    print(f"physical_gpu_index_for_nvidia_smi={int(args.physical_gpu_index)}")
    print(f"jax_gpu_devices={jax.devices('gpu')}")

    rows: list[AceneResult] = []
    for rings in range(int(args.min_rings), int(args.max_rings) + 1):
        t0 = time.perf_counter()
        row = benchmark_one(
            rings,
            basis=str(args.basis),
            xc=str(args.xc),
            grids_level=int(args.grids_level),
            nstates_max=int(args.nstates_max),
            occ_factor=int(args.occ_factor),
            vir_factor=int(args.vir_factor),
            active_min=int(args.active_min),
            scf_max_cycle=int(args.scf_max_cycle),
            scf_conv_tol=float(args.scf_conv_tol),
            tddft_max_cycle=int(args.tddft_max_cycle),
            cart=bool(args.cart),
            density_fit=bool(args.density_fit),
            gpu_index=int(args.physical_gpu_index),
            gpu_sample_interval_s=float(args.gpu_sample_interval_s),
            gpu_repeats=int(args.gpu_repeats),
            gpu_profile_min_s=float(args.gpu_profile_min_s),
            run_pyscf_tddft=bool(args.run_pyscf_tddft),
        )
        rows.append(row)
        print(
            f"[rings={row.rings} {row.name}] formula={row.formula} nao={row.nao} "
            f"td_dim={row.td_dim} scf={row.scf_s:.3f}s "
            f"pyscf_full={row.cpu_pyscf_full_s:.3f}s "
            f"gpu_full={row.gpu_full_s_excl_compile:.3f}s "
            f"gpu_core={row.gpu_solve_mean_s:.4f}s "
            f"speedup_full={row.gpu_speedup_over_pyscf_full:.3f}x "
            f"gpu_util_mean/max={row.gpu_util_mean_pct:.1f}/{row.gpu_util_max_pct:.1f}% "
            f"gpu_mem_max={row.gpu_mem_max_mib:.0f} MiB "
            f"exc1_cpu_dense/gpu={row.cpu_dense_exc1_ev:.6f}/{row.gpu_exc1_ev:.6f} eV "
            f"diff={row.gpu_exc1_abs_diff_ev_vs_cpu_dense:.3e} eV "
            f"elapsed={time.perf_counter() - t0:.2f}s"
        )

    csv_path = outdir / "acene_tddft_gpu_curve.csv"
    json_path = outdir / "acene_tddft_gpu_curve.json"
    time_png = outdir / "acene_tddft_time_curve.png"
    speedup_png = outdir / "acene_tddft_speedup_curve.png"
    gpu_png = outdir / "acene_gpu_util_memory_curve.png"
    energy_png = outdir / "acene_energy_alignment.png"

    write_csv(csv_path, rows)
    write_json(json_path, args=args, rows=rows)
    title = f"Acene TDDFT Active-Space Benchmark ({str(args.xc).upper()}/{str(args.basis).upper()})"
    plot_time_curve(time_png, rows, title=title)
    plot_speedup_curve(speedup_png, rows, title=title)
    plot_gpu_stats(gpu_png, rows, title="Physical GPU Telemetry During GPU Casida Block")
    plot_energy_alignment(energy_png, rows, title="GPU/CPU Dense Casida Energy Alignment")

    print(f"csv={csv_path}")
    print(f"json={json_path}")
    print(f"time_png={time_png}")
    print(f"speedup_png={speedup_png}")
    print(f"gpu_png={gpu_png}")
    print(f"energy_png={energy_png}")


if __name__ == "__main__":
    main()
