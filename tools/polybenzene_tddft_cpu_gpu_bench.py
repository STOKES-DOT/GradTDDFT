from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pyscf import dft, gto

jax.config.update("jax_enable_x64", True)


@dataclass
class BenchResult:
    rings: int
    nao: int
    nmo: int
    nocc: int
    nvir: int
    occ_keep: int
    vir_keep: int
    nstates: int
    scf_s: float
    scf_e_tot_hartree: float
    cpu_tddft_s: float
    gpu_casida_s: float
    cpu_exc1_ev: float
    gpu_exc1_ev: float
    exc1_abs_diff_ev: float
    cpu_ok: int
    gpu_ok: int


def _benzene_atoms(center_x: float, carbon_r: float = 1.397, hydrogen_r: float = 2.479):
    atoms = []
    for k in range(6):
        theta = np.deg2rad(60.0 * k)
        cx = center_x + carbon_r * np.cos(theta)
        cy = carbon_r * np.sin(theta)
        hx = center_x + hydrogen_r * np.cos(theta)
        hy = hydrogen_r * np.sin(theta)
        atoms.append(("C", (cx, cy, 0.0)))
        atoms.append(("H", (hx, hy, 0.0)))
    return atoms


def make_polybenzene_fragmented(n_rings: int, separation: float = 6.0):
    atoms = []
    for i in range(n_rings):
        atoms.extend(_benzene_atoms(center_x=i * separation))
    return atoms


def make_molecule(n_rings: int, basis: str):
    mol = gto.Mole()
    mol.atom = make_polybenzene_fragmented(n_rings)
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.spin = 0
    mol.verbose = 0
    mol.build()
    return mol


def choose_active_space(nocc: int, nvir: int, rings: int, occ_factor: int, vir_factor: int):
    occ_keep = min(nocc, max(4, occ_factor * rings))
    vir_keep = min(nvir, max(4, vir_factor * rings))
    return occ_keep, vir_keep


def build_frozen_list(nocc: int, nmo: int, occ_keep: int, vir_keep: int):
    freeze_occ = list(range(0, max(0, nocc - occ_keep)))
    freeze_vir = list(range(min(nmo, nocc + vir_keep), nmo))
    return freeze_occ + freeze_vir


@jax.jit
def casida_eigs_jax(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    a = 0.5 * (a + a.T)
    b = 0.5 * (b + b.T)
    apb = 0.5 * ((a + b) + (a + b).T)
    amb = 0.5 * ((a - b) + (a - b).T)
    eye = jnp.eye(a.shape[0], dtype=a.dtype)
    apb = apb + 1e-10 * eye
    amb = amb + 1e-10 * eye
    w, v = jnp.linalg.eigh(amb)
    w = jnp.clip(w, 1e-10, None)
    sqrt_amb = (v * jnp.sqrt(w)) @ v.T
    c = sqrt_amb @ apb @ sqrt_amb
    c = 0.5 * (c + c.T)
    w2, _ = jnp.linalg.eigh(c)
    return jnp.sqrt(jnp.clip(w2, 0.0, None))


def benchmark_one(
    rings: int,
    *,
    basis: str,
    xc: str,
    grids_level: int,
    nstates_max: int,
    occ_factor: int,
    vir_factor: int,
) -> BenchResult:
    au_to_ev = 27.211386245988
    mol = make_molecule(rings, basis=basis)
    mf = dft.RKS(mol).density_fit()
    mf.xc = xc
    mf.grids.level = grids_level
    mf.conv_tol = 1e-9
    mf.max_cycle = 120

    t0 = time.perf_counter()
    mf.kernel()
    scf_s = time.perf_counter() - t0
    if not mf.converged:
        raise RuntimeError(f"SCF did not converge for rings={rings}")
    scf_e_tot_hartree = float(mf.e_tot)

    nmo = int(mf.mo_occ.size)
    nocc = int(np.count_nonzero(mf.mo_occ > 1e-8))
    nvir = int(nmo - nocc)
    occ_keep, vir_keep = choose_active_space(nocc, nvir, rings, occ_factor, vir_factor)
    frozen = build_frozen_list(nocc, nmo, occ_keep, vir_keep)
    nov = occ_keep * vir_keep
    nstates = max(1, min(nstates_max, max(1, nov - 1)))

    # CPU baseline: PySCF TDDFT (96-thread BLAS/OpenMP set outside script).
    cpu_tddft_s = float("nan")
    cpu_exc1_ev = float("nan")
    cpu_ok = 0
    td = mf.TDDFT()
    td.nstates = nstates
    td.frozen = frozen
    td.max_cycle = 80
    try:
        t1 = time.perf_counter()
        td.kernel()
        cpu_tddft_s = time.perf_counter() - t1
        cpu_e = np.asarray(td.e, dtype=float).reshape(-1)
        if cpu_e.size > 0 and np.isfinite(cpu_e[0]):
            cpu_exc1_ev = float(cpu_e[0] * au_to_ev)
        cpu_ok = 1
    except Exception:
        cpu_tddft_s = float("nan")
        cpu_exc1_ev = float("nan")
        cpu_ok = 0

    # GPU path: JAX Casida core solve from the same active-space A/B matrices.
    gpu_casida_s = float("nan")
    gpu_exc1_ev = float("nan")
    gpu_ok = 0
    try:
        td2 = mf.TDDFT()
        td2.frozen = frozen
        a, b = td2.get_ab()
        a2 = np.asarray(a.reshape(nov, nov), dtype=np.float64)
        b2 = np.asarray(b.reshape(nov, nov), dtype=np.float64)
        if (not np.isfinite(a2).all()) or (not np.isfinite(b2).all()):
            raise FloatingPointError("Non-finite values detected in A/B matrices.")
        a_dev = jax.device_put(jnp.asarray(a2))
        b_dev = jax.device_put(jnp.asarray(b2))

        compiled = casida_eigs_jax.lower(a_dev, b_dev).compile()
        _ = compiled(a_dev, b_dev).block_until_ready()  # warm-up
        t2 = time.perf_counter()
        w_gpu_au = np.asarray(compiled(a_dev, b_dev).block_until_ready(), dtype=float).reshape(-1)
        gpu_casida_s = time.perf_counter() - t2
        w_pos = np.sort(w_gpu_au[w_gpu_au > 1e-10])
        if w_pos.size > 0 and np.isfinite(w_pos[0]):
            gpu_exc1_ev = float(w_pos[0] * au_to_ev)
        gpu_ok = 1
    except Exception:
        gpu_casida_s = float("nan")
        gpu_exc1_ev = float("nan")
        gpu_ok = 0

    exc1_abs_diff_ev = (
        abs(cpu_exc1_ev - gpu_exc1_ev)
        if np.isfinite(cpu_exc1_ev) and np.isfinite(gpu_exc1_ev)
        else float("nan")
    )

    return BenchResult(
        rings=rings,
        nao=int(mol.nao_nr()),
        nmo=nmo,
        nocc=nocc,
        nvir=nvir,
        occ_keep=occ_keep,
        vir_keep=vir_keep,
        nstates=nstates,
        scf_s=scf_s,
        scf_e_tot_hartree=scf_e_tot_hartree,
        cpu_tddft_s=cpu_tddft_s,
        gpu_casida_s=gpu_casida_s,
        cpu_exc1_ev=cpu_exc1_ev,
        gpu_exc1_ev=gpu_exc1_ev,
        exc1_abs_diff_ev=exc1_abs_diff_ev,
        cpu_ok=cpu_ok,
        gpu_ok=gpu_ok,
    )


def write_csv(path: Path, rows: list[BenchResult]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rings",
                "nao",
                "nmo",
                "nocc",
                "nvir",
                "occ_keep",
                "vir_keep",
                "nstates",
                "scf_s",
                "scf_e_tot_hartree",
                "cpu_tddft_s",
                "gpu_casida_s",
                "cpu_exc1_ev",
                "gpu_exc1_ev",
                "exc1_abs_diff_ev",
                "cpu_ok",
                "gpu_ok",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.rings,
                    r.nao,
                    r.nmo,
                    r.nocc,
                    r.nvir,
                    r.occ_keep,
                    r.vir_keep,
                    r.nstates,
                    f"{r.scf_s:.6f}",
                    f"{r.scf_e_tot_hartree:.10f}",
                    f"{r.cpu_tddft_s:.6f}" if np.isfinite(r.cpu_tddft_s) else "",
                    f"{r.gpu_casida_s:.6f}" if np.isfinite(r.gpu_casida_s) else "",
                    f"{r.cpu_exc1_ev:.8f}" if np.isfinite(r.cpu_exc1_ev) else "",
                    f"{r.gpu_exc1_ev:.8f}" if np.isfinite(r.gpu_exc1_ev) else "",
                    f"{r.exc1_abs_diff_ev:.8f}" if np.isfinite(r.exc1_abs_diff_ev) else "",
                    r.cpu_ok,
                    r.gpu_ok,
                ]
            )


def plot_curves(path: Path, rows: list[BenchResult], title: str) -> None:
    rings = np.array([r.rings for r in rows], dtype=float)
    cpu = np.array([r.cpu_tddft_s for r in rows], dtype=float)
    gpu = np.array([r.gpu_casida_s for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rings, cpu, marker="o", lw=2, label="CPU: PySCF TDDFT (96 cores)")
    ax.plot(rings, gpu, marker="s", lw=2, label="GPU: JAX Casida core (B3LYP A/B)")
    ax.set_xlabel("Number of benzene rings")
    ax.set_ylabel("Wall time (s)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)


def plot_energy_compare(path: Path, rows: list[BenchResult], title: str) -> None:
    rings = np.array([r.rings for r in rows], dtype=float)
    e_tot = np.array([r.scf_e_tot_hartree for r in rows], dtype=float)
    cpu_exc1 = np.array([r.cpu_exc1_ev for r in rows], dtype=float)
    gpu_exc1 = np.array([r.gpu_exc1_ev for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax0, ax1 = axes

    ax0.plot(rings, e_tot, marker="o", lw=2, label="PySCF SCF E_tot")
    ax0.set_xlabel("Number of benzene rings")
    ax0.set_ylabel("Ground-state total energy (Hartree)")
    ax0.grid(alpha=0.25)
    ax0.legend(frameon=False)

    ax1.plot(rings, cpu_exc1, marker="o", lw=2, label="CPU TDDFT Exc1")
    ax1.plot(rings, gpu_exc1, marker="s", lw=2, label="GPU Casida Exc1")
    ax1.set_xlabel("Number of benzene rings")
    ax1.set_ylabel("First excitation energy (eV)")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--min-rings", type=int, default=1)
    p.add_argument("--max-rings", type=int, default=10)
    p.add_argument("--basis", type=str, default="6-31g")
    p.add_argument("--xc", type=str, default="b3lyp")
    p.add_argument("--grids-level", type=int, default=0)
    p.add_argument("--nstates-max", type=int, default=8)
    p.add_argument("--occ-factor", type=int, default=3)
    p.add_argument("--vir-factor", type=int, default=3)
    p.add_argument("--outdir", type=str, default="outputs/polybenzene_bench")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[BenchResult] = []
    for rings in range(args.min_rings, args.max_rings + 1):
        t0 = time.perf_counter()
        row = benchmark_one(
            rings,
            basis=args.basis,
            xc=args.xc,
            grids_level=args.grids_level,
            nstates_max=args.nstates_max,
            occ_factor=args.occ_factor,
            vir_factor=args.vir_factor,
        )
        rows.append(row)
        elapsed = time.perf_counter() - t0
        print(
            f"[rings={rings}] nao={row.nao}, nstates={row.nstates}, "
            f"SCF={row.scf_s:.2f}s, CPU_TDDFT={row.cpu_tddft_s:.2f}s, "
            f"GPU_Casida={row.gpu_casida_s:.2f}s, "
            f"E_tot={row.scf_e_tot_hartree:.6f} Ha, "
            f"Exc1(cpu/gpu)={row.cpu_exc1_ev:.4f}/{row.gpu_exc1_ev:.4f} eV, "
            f"total={elapsed:.2f}s"
        )

    csv_path = outdir / "polybenzene_tddft_cpu_gpu_times.csv"
    png_path = outdir / "polybenzene_tddft_cpu_gpu_times.png"
    energy_png_path = outdir / "polybenzene_energy_compare.png"
    write_csv(csv_path, rows)
    plot_curves(
        png_path,
        rows,
        title=f"Polybenzene Timing Curve ({args.xc.upper()}/{args.basis.upper()})",
    )
    plot_energy_compare(
        energy_png_path,
        rows,
        title=f"Polybenzene Energy Compare ({args.xc.upper()}/{args.basis.upper()})",
    )
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote PNG: {png_path}")
    print(f"Wrote Energy PNG: {energy_png_path}")


if __name__ == "__main__":
    main()
