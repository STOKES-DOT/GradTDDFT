from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

import jax
import matplotlib.pyplot as plt
import numpy as np
from pyscf import dft, gto

from td_graddft import tdscf
from td_graddft.features import restricted_grid_features_with_gradients
from td_graddft.xc_backend.jax_libxc import eval_xc_response_tensor, hybrid_coeff, xc_type
from td_graddft.data.reference import restricted_reference_from_pyscf
from td_graddft.spectra import HARTREE_TO_EV


def _linear_acene_atoms(n_rings: int, *, c_c: float = 1.397, c_h: float = 1.09):
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


MOLECULES = {
    "benzene": """
C        0.0000000000      1.3967920000      0.0000000000
C       -1.2096570000      0.6983960000      0.0000000000
C       -1.2096570000     -0.6983960000      0.0000000000
C        0.0000000000     -1.3967920000      0.0000000000
C        1.2096570000     -0.6983960000      0.0000000000
C        1.2096570000      0.6983960000      0.0000000000
H        0.0000000000      2.4842120000      0.0000000000
H       -2.1513900000      1.2421060000      0.0000000000
H       -2.1513900000     -1.2421060000      0.0000000000
H        0.0000000000     -2.4842120000      0.0000000000
H        2.1513900000     -1.2421060000      0.0000000000
H        2.1513900000      1.2421060000      0.0000000000
""",
    "anthracene": _linear_acene_atoms(3),
}


@dataclass(frozen=True)
class TimingRow:
    backend: str
    task: str
    repeat: int
    elapsed_s: float
    first_excitation_ha: float


class SemilocalResponseFunctional:
    def __init__(self, xc_spec: str):
        self.xc_spec = str(xc_spec).lower()
        self.exact_exchange_fraction = float(hybrid_coeff(self.xc_spec))
        self.response_feature_kind = str(xc_type(self.xc_spec))

    def grid_response_tensor(self, molecule):
        features, grad_rho = restricted_grid_features_with_gradients(molecule)
        tau = features.tau_a + features.tau_b
        _, tensor = eval_xc_response_tensor(
            self.xc_spec,
            features.rho,
            grad=grad_rho,
            tau=tau,
        )
        return tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark large-molecule PySCF vs JAX TDDFT/TDA with jit warm timing."
    )
    p.add_argument("--molecule", choices=tuple(MOLECULES.keys()), default="anthracene")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--xc", default="pbe0")
    p.add_argument("--nstates", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--outdir", default="outputs/large_molecule_pyscf_vs_jax_tddft_jit")
    return p.parse_args()


def _build_mf(molecule: str, basis: str, xc: str):
    mol = gto.M(
        atom=MOLECULES[molecule],
        basis=basis,
        unit="Angstrom",
        spin=0,
        charge=0,
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF SCF did not converge.")
    return mol, mf


def _block_and_time(fn):
    t0 = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    return float(time.perf_counter() - t0), out


def _summarize(rows: list[TimingRow], backend: str, task: str) -> tuple[float, float]:
    values = [row.elapsed_s for row in rows if row.backend == backend and row.task == task]
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def _plot(path: Path, rows: list[TimingRow]) -> None:
    items = [
        ("pyscf", "tda", "PySCF\nTDA"),
        ("pyscf", "casida", "PySCF\nCasida"),
        ("jax_auto_jit_warm", "tda", "JAX auto jit\nTDA"),
        ("jax_auto_jit_warm", "casida", "JAX auto jit\nCasida"),
    ]
    means = []
    stds = []
    labels = []
    for backend, task, label in items:
        mean, std = _summarize(rows, backend, task)
        means.append(mean)
        stds.append(std)
        labels.append(label)

    x = np.arange(len(items), dtype=float)
    colors = ["#4C78A8", "#72B7B2", "#9C755F", "#BAB0AC", "#F58518", "#54A24B"]
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Elapsed Time (s)")
    ax.set_title("Large-Molecule PySCF vs JAX TDDFT Benchmark")
    ax.grid(axis="y", alpha=0.25)
    for xi, yi in zip(x, means, strict=True):
        ax.text(xi, yi, f"{yi:.3f}s", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _, mf = _build_mf(args.molecule, str(args.basis), str(args.xc))
    reference = restricted_reference_from_pyscf(mf)
    xc_func = SemilocalResponseFunctional(str(args.xc))

    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 1e-8))
    nvir = int(np.asarray(mf.mo_coeff).shape[-1] - nocc)
    nstates = min(max(1, int(args.nstates)), nocc * nvir)

    tda_auto = tdscf.TDA(
        reference,
        xc_functional=xc_func,
        eigensolver="auto",
    )
    solver_auto = tdscf.TDDFT(
        reference,
        xc_functional=xc_func,
        eigensolver="auto",
    )
    def run_pyscf_tda():
        td = mf.TDA()
        td.nstates = nstates
        td.kernel()
        return np.asarray(td.e, dtype=float)

    def run_pyscf_casida():
        td = mf.TDDFT()
        td.nstates = nstates
        td.kernel()
        return np.asarray(td.e, dtype=float)

    def run_jax_auto_tda():
        return tda_auto.kernel(nstates=nstates).excitation_energies

    def run_jax_auto_casida():
        return solver_auto.kernel(nstates=nstates).excitation_energies

    jit_jax_auto_tda = jax.jit(run_jax_auto_tda)
    jit_jax_auto_casida = jax.jit(run_jax_auto_casida)

    rows: list[TimingRow] = []

    for repeat in range(1, int(args.repeats) + 1):
        elapsed, out = _block_and_time(run_pyscf_tda)
        rows.append(
            TimingRow(
                backend="pyscf",
                task="tda",
                repeat=repeat,
                elapsed_s=elapsed,
                first_excitation_ha=float(np.asarray(out).reshape(-1)[0]),
            )
        )
        elapsed, out = _block_and_time(run_pyscf_casida)
        rows.append(
            TimingRow(
                backend="pyscf",
                task="casida",
                repeat=repeat,
                elapsed_s=elapsed,
                first_excitation_ha=float(np.asarray(out).reshape(-1)[0]),
            )
        )

    _block_and_time(jit_jax_auto_tda)
    _block_and_time(jit_jax_auto_casida)

    for repeat in range(1, int(args.repeats) + 1):
        elapsed, out = _block_and_time(jit_jax_auto_tda)
        rows.append(
            TimingRow(
                backend="jax_auto_jit_warm",
                task="tda",
                repeat=repeat,
                elapsed_s=elapsed,
                first_excitation_ha=float(np.asarray(out).reshape(-1)[0]),
            )
        )
        elapsed, out = _block_and_time(jit_jax_auto_casida)
        rows.append(
            TimingRow(
                backend="jax_auto_jit_warm",
                task="casida",
                repeat=repeat,
                elapsed_s=elapsed,
                first_excitation_ha=float(np.asarray(out).reshape(-1)[0]),
            )
        )
    stem = f"{args.molecule}_{str(args.xc).lower()}_{str(args.basis).lower()}".replace("*", "star")
    csv_path = outdir / f"{stem}_pyscf_vs_jax_tddft_jit.csv"
    summary_path = outdir / f"{stem}_pyscf_vs_jax_tddft_jit_summary.txt"
    plot_path = outdir / f"{stem}_pyscf_vs_jax_tddft_jit.png"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["backend", "task", "repeat", "elapsed_s", "first_excitation_ha"])
        for row in rows:
            writer.writerow(
                [
                    row.backend,
                    row.task,
                    row.repeat,
                    f"{row.elapsed_s:.10f}",
                    f"{row.first_excitation_ha:.12f}",
                ]
            )

    pyscf_tda_mean, pyscf_tda_std = _summarize(rows, "pyscf", "tda")
    pyscf_casida_mean, pyscf_casida_std = _summarize(rows, "pyscf", "casida")
    jax_auto_tda_mean, jax_auto_tda_std = _summarize(rows, "jax_auto_jit_warm", "tda")
    jax_auto_casida_mean, jax_auto_casida_std = _summarize(rows, "jax_auto_jit_warm", "casida")

    first = lambda backend, task: [r.first_excitation_ha for r in rows if r.backend == backend and r.task == task][-1]

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Large-molecule PySCF vs JAX TDDFT benchmark\n")
        f.write(f"molecule = {args.molecule}\n")
        f.write(f"xc = {args.xc}\n")
        f.write(f"basis = {args.basis}\n")
        f.write(f"nstates = {nstates}\n")
        f.write(f"repeats = {args.repeats}\n\n")
        f.write("notes = benchmark isolates the excited-state layer after a fixed PySCF SCF reference. JAX numbers report jit warm timings only.\n\n")
        f.write(f"pyscf_tda_mean_s = {pyscf_tda_mean:.10f}\n")
        f.write(f"pyscf_tda_std_s = {pyscf_tda_std:.10f}\n")
        f.write(f"pyscf_casida_mean_s = {pyscf_casida_mean:.10f}\n")
        f.write(f"pyscf_casida_std_s = {pyscf_casida_std:.10f}\n")
        f.write(f"jax_auto_jit_warm_tda_mean_s = {jax_auto_tda_mean:.10f}\n")
        f.write(f"jax_auto_jit_warm_tda_std_s = {jax_auto_tda_std:.10f}\n")
        f.write(f"jax_auto_jit_warm_casida_mean_s = {jax_auto_casida_mean:.10f}\n")
        f.write(f"jax_auto_jit_warm_casida_std_s = {jax_auto_casida_std:.10f}\n")
        f.write(f"tda_auto_speedup_vs_pyscf = {pyscf_tda_mean / jax_auto_tda_mean:.6f}\n")
        f.write(f"casida_auto_speedup_vs_pyscf = {pyscf_casida_mean / jax_auto_casida_mean:.6f}\n\n")
        f.write(f"tda_auto_first_excitation_abs_diff_ev = {abs(first('jax_auto_jit_warm', 'tda') - first('pyscf', 'tda')) * HARTREE_TO_EV:.10f}\n")
        f.write(f"casida_auto_first_excitation_abs_diff_ev = {abs(first('jax_auto_jit_warm', 'casida') - first('pyscf', 'casida')) * HARTREE_TO_EV:.10f}\n")

    _plot(plot_path, rows)

    print(f"molecule={args.molecule}")
    print(f"xc={args.xc}, basis={args.basis}, nstates={nstates}")
    print(f"pyscf_tda_mean_s={pyscf_tda_mean:.6f}")
    print(f"pyscf_casida_mean_s={pyscf_casida_mean:.6f}")
    print(f"jax_auto_jit_warm_tda_mean_s={jax_auto_tda_mean:.6f}")
    print(f"jax_auto_jit_warm_casida_mean_s={jax_auto_casida_mean:.6f}")
    print(f"summary={summary_path}")
    print(f"plot={plot_path}")
    print(f"csv={csv_path}")


if __name__ == "__main__":
    main()
