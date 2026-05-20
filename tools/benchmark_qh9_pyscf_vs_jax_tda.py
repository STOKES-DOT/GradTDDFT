from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pyscf import dft, gto

from td_graddft import tdscf
from td_graddft.features import restricted_grid_features_with_gradients
from td_graddft.xc_backend.jax_libxc import eval_xc_response_tensor, hybrid_coeff, xc_type
from td_graddft.reference_legacy import restricted_reference_from_pyscf
from td_graddft.spectra import HARTREE_TO_EV, lorentzian_spectrum, oscillator_strengths
from td_graddft.tddft.response import build_restricted_response_matrices
from td_graddft.tddft.tda import solve_tda


ATOMIC_SYMBOL = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
}


@dataclass(frozen=True)
class TimingRow:
    backend: str
    task: str
    repeat: int
    elapsed_s: float


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
    parser = argparse.ArgumentParser(
        description="Benchmark PySCF TDA against JAX-backbone TDA on one QH9 molecule."
    )
    parser.add_argument("--db-path", default="/Volumes/TF/QH9_db/QH9Stable.db")
    parser.add_argument("--db-id", type=int, default=104)
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--nstates", type=int, default=8)
    parser.add_argument("--eta-ev", type=float, default=0.12)
    parser.add_argument("--grid-min-ev", type=float, default=0.0)
    parser.add_argument("--grid-max-ev", type=float, default=16.0)
    parser.add_argument("--grid-points", type=int, default=1200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--outdir", default="outputs/qh9_pyscf_vs_jax_tda_benchmark")
    return parser.parse_args()


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


def _fetch_molecule(db_path: Path, db_id: int) -> tuple[np.ndarray, np.ndarray]:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT N, Z, pos FROM data WHERE id = ?", (db_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise KeyError(f"QH9 id {db_id} not found in {db_path}")
    n, z_blob, pos_blob = row
    z = np.frombuffer(z_blob, dtype=np.int32).copy()
    pos = np.frombuffer(pos_blob, dtype=np.float64).reshape(int(n), 3).copy()
    return z, pos


def _build_reference(z: np.ndarray, pos_ang: np.ndarray, *, basis: str, xc: str):
    mol = gto.M(
        atom=_build_atom_block(z, pos_ang),
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
        raise RuntimeError(f"PySCF SCF did not converge for xc={xc}, basis={basis}.")
    return mol, mf, restricted_reference_from_pyscf(mf)


def _pyscf_lorentzian_spectrum(
    energies_ev: np.ndarray,
    strengths: np.ndarray,
    grid_ev: np.ndarray,
    *,
    eta: float,
) -> np.ndarray:
    diffs = grid_ev[:, None] - energies_ev[None, :]
    broadened = eta / (np.pi * (diffs**2 + eta**2))
    return np.sum(strengths[None, :] * broadened, axis=1)


def _block_and_time(fn) -> float:
    t0 = time.perf_counter()
    out = fn()
    if isinstance(out, tuple):
        for item in out:
            jax.block_until_ready(item)
    else:
        jax.block_until_ready(out)
    return float(time.perf_counter() - t0)


def _summarize(rows: list[TimingRow], *, backend: str, task: str) -> tuple[float, float]:
    values = [row.elapsed_s for row in rows if row.backend == backend and row.task == task]
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def _write_rows(path: Path, rows: list[TimingRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["backend", "task", "repeat", "elapsed_s"])
        for row in rows:
            writer.writerow([row.backend, row.task, row.repeat, f"{row.elapsed_s:.10f}"])


def _write_summary(
    path: Path,
    *,
    db_id: int,
    formula: str,
    xc: str,
    basis: str,
    nstates: int,
    repeats: int,
    rows: list[TimingRow],
) -> None:
    labels = [
        ("pyscf", "tda"),
        ("pyscf", "spectrum"),
        ("jax_auto", "tda"),
        ("jax_auto_jit_warm", "tda"),
        ("jax_dense", "tda"),
        ("jax_dense_jit_warm", "tda"),
        ("jax_auto", "spectrum"),
        ("jax_auto_jit_warm", "spectrum"),
        ("jax_dense", "spectrum"),
        ("jax_dense_jit_warm", "spectrum"),
    ]

    with path.open("w", encoding="utf-8") as f:
        f.write("QH9 PySCF vs JAX TDA benchmark\n")
        f.write(f"db_id = {db_id}\n")
        f.write(f"formula = {formula}\n")
        f.write(f"xc = {xc}\n")
        f.write(f"basis = {basis}\n")
        f.write(f"nstates = {nstates}\n")
        f.write(f"repeats = {repeats}\n")
        f.write("\n")
        f.write("notes = benchmark isolates the excited-state layer after a fixed SCF reference. It reports both the default JAX auto path and the dense JAX path, with and without jit warm-start.\n")
        f.write("\n")
        for backend, task in labels:
            mean, std = _summarize(rows, backend=backend, task=task)
            f.write(f"{backend}_{task}_mean_s = {mean:.10f}\n")
            f.write(f"{backend}_{task}_std_s = {std:.10f}\n")
        f.write("\n")
        pyscf_tda_mean, _ = _summarize(rows, backend="pyscf", task="tda")
        jax_auto_tda_jit_mean, _ = _summarize(rows, backend="jax_auto_jit_warm", task="tda")
        jax_tda_jit_mean, _ = _summarize(rows, backend="jax_dense_jit_warm", task="tda")
        pyscf_spec_mean, _ = _summarize(rows, backend="pyscf", task="spectrum")
        jax_auto_spec_jit_mean, _ = _summarize(rows, backend="jax_auto_jit_warm", task="spectrum")
        jax_spec_jit_mean, _ = _summarize(rows, backend="jax_dense_jit_warm", task="spectrum")
        f.write(f"tda_jax_auto_jit_vs_pyscf_speedup = {pyscf_tda_mean / jax_auto_tda_jit_mean:.6f}\n")
        f.write(f"tda_jax_jit_vs_pyscf_speedup = {pyscf_tda_mean / jax_tda_jit_mean:.6f}\n")
        f.write(f"spectrum_jax_auto_jit_vs_pyscf_speedup = {pyscf_spec_mean / jax_auto_spec_jit_mean:.6f}\n")
        f.write(f"spectrum_jax_jit_vs_pyscf_speedup = {pyscf_spec_mean / jax_spec_jit_mean:.6f}\n")


def _plot(path: Path, rows: list[TimingRow]) -> None:
    items = [
        ("pyscf", "tda", "PySCF\nTDA"),
        ("pyscf", "spectrum", "PySCF\nSpectrum"),
        ("jax_auto", "tda", "JAX auto\nTDA"),
        ("jax_auto_jit_warm", "tda", "JAX auto jit\nTDA"),
        ("jax_dense", "tda", "JAX dense\nTDA"),
        ("jax_dense_jit_warm", "tda", "JAX dense jit\nTDA"),
        ("jax_auto", "spectrum", "JAX auto\nSpectrum"),
        ("jax_auto_jit_warm", "spectrum", "JAX auto jit\nSpectrum"),
        ("jax_dense", "spectrum", "JAX dense\nSpectrum"),
        ("jax_dense_jit_warm", "spectrum", "JAX dense jit\nSpectrum"),
    ]
    means = []
    stds = []
    labels = []
    for backend, task, label in items:
        mean, std = _summarize(rows, backend=backend, task=task)
        means.append(mean)
        stds.append(std)
        labels.append(label)

    x = np.arange(len(items), dtype=float)
    colors = [
        "#4C78A8",
        "#72B7B2",
        "#9C755F",
        "#BAB0AC",
        "#F58518",
        "#54A24B",
        "#E45756",
        "#B279A2",
        "#FF9DA6",
        "#8E6C8A",
    ]

    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Elapsed Time (s)")
    ax.set_title("QH9 PySCF vs JAX TDA Benchmark")
    ax.grid(axis="y", alpha=0.25)
    for xi, yi in zip(x, means, strict=True):
        ax.text(xi, yi, f"{yi:.3f}s", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    db_path = Path(args.db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"QH9 db not found: {db_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    z, pos_ang = _fetch_molecule(db_path, int(args.db_id))
    formula = _formula_from_z(z)
    _, mf, reference = _build_reference(z, pos_ang, basis=str(args.basis), xc=str(args.xc))
    xc_func = SemilocalResponseFunctional(str(args.xc))

    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 1e-8))
    nvir = int(np.asarray(mf.mo_coeff).shape[-1] - nocc)
    nstates = min(max(1, int(args.nstates)), nocc * nvir)
    grid_ev_np = np.linspace(
        float(args.grid_min_ev),
        float(args.grid_max_ev),
        int(args.grid_points),
        dtype=float,
    )
    grid_ev_jax = jnp.asarray(grid_ev_np, dtype=jnp.float32)

    td_auto = tdscf.TDA(
        reference,
        xc_functional=xc_func,
        eigensolver="auto",
    )

    def run_pyscf_tda():
        td = mf.TDA()
        td.nstates = nstates
        td.kernel()
        return np.asarray(td.e, dtype=float)

    def run_pyscf_spectrum():
        td = mf.TDA()
        td.nstates = nstates
        td.kernel()
        energies_ev = np.asarray(td.e, dtype=float) * HARTREE_TO_EV
        strengths = np.asarray(td.oscillator_strength(), dtype=float)
        return _pyscf_lorentzian_spectrum(
            energies_ev,
            strengths,
            grid_ev_np,
            eta=float(args.eta_ev),
        )

    def run_jax_auto_tda():
        return td_auto.kernel(nstates=nstates).excitation_energies

    def run_jax_auto_spectrum():
        result = td_auto.kernel(nstates=nstates)
        strengths = td_auto.oscillator_strength()
        return lorentzian_spectrum(
            result.excitation_energies * HARTREE_TO_EV,
            strengths,
            grid_ev_jax,
            eta=float(args.eta_ev),
        )

    def run_jax_dense_tda():
        mats = build_restricted_response_matrices(reference, xc_func)
        return solve_tda(mats, nstates=nstates, eigensolver="dense").excitation_energies

    def run_jax_dense_spectrum():
        mats = build_restricted_response_matrices(reference, xc_func)
        result = solve_tda(mats, nstates=nstates, eigensolver="dense")
        strengths = oscillator_strengths(reference, result)
        return lorentzian_spectrum(
            result.excitation_energies * HARTREE_TO_EV,
            strengths,
            grid_ev_jax,
            eta=float(args.eta_ev),
        )

    jit_jax_auto_tda = jax.jit(run_jax_auto_tda)
    jit_jax_auto_spectrum = jax.jit(run_jax_auto_spectrum)
    jit_jax_dense_tda = jax.jit(run_jax_dense_tda)
    jit_jax_dense_spectrum = jax.jit(run_jax_dense_spectrum)

    rows: list[TimingRow] = []

    for repeat in range(1, int(args.repeats) + 1):
        rows.append(
            TimingRow(
                backend="pyscf",
                task="tda",
                repeat=repeat,
                elapsed_s=_block_and_time(run_pyscf_tda),
            )
        )
        rows.append(
            TimingRow(
                backend="pyscf",
                task="spectrum",
                repeat=repeat,
                elapsed_s=_block_and_time(run_pyscf_spectrum),
            )
        )
        rows.append(
            TimingRow(
                backend="jax_auto",
                task="tda",
                repeat=repeat,
                elapsed_s=_block_and_time(run_jax_auto_tda),
            )
        )
        rows.append(
            TimingRow(
                backend="jax_auto",
                task="spectrum",
                repeat=repeat,
                elapsed_s=_block_and_time(run_jax_auto_spectrum),
            )
        )
        rows.append(
            TimingRow(
                backend="jax_dense",
                task="tda",
                repeat=repeat,
                elapsed_s=_block_and_time(run_jax_dense_tda),
            )
        )
        rows.append(
            TimingRow(
                backend="jax_dense",
                task="spectrum",
                repeat=repeat,
                elapsed_s=_block_and_time(run_jax_dense_spectrum),
            )
        )

    _block_and_time(jit_jax_auto_tda)
    _block_and_time(jit_jax_auto_spectrum)
    _block_and_time(jit_jax_dense_tda)
    _block_and_time(jit_jax_dense_spectrum)
    for repeat in range(1, int(args.repeats) + 1):
        rows.append(
            TimingRow(
                backend="jax_auto_jit_warm",
                task="tda",
                repeat=repeat,
                elapsed_s=_block_and_time(jit_jax_auto_tda),
            )
        )
        rows.append(
            TimingRow(
                backend="jax_auto_jit_warm",
                task="spectrum",
                repeat=repeat,
                elapsed_s=_block_and_time(jit_jax_auto_spectrum),
            )
        )
        rows.append(
            TimingRow(
                backend="jax_dense_jit_warm",
                task="tda",
                repeat=repeat,
                elapsed_s=_block_and_time(jit_jax_dense_tda),
            )
        )
        rows.append(
            TimingRow(
                backend="jax_dense_jit_warm",
                task="spectrum",
                repeat=repeat,
                elapsed_s=_block_and_time(jit_jax_dense_spectrum),
            )
        )

    stem = f"qh9_id{args.db_id}_{formula}_{str(args.xc).lower()}_{str(args.basis).lower()}".replace("*", "star")
    csv_path = outdir / f"{stem}_pyscf_vs_jax_tda_benchmark.csv"
    summary_path = outdir / f"{stem}_pyscf_vs_jax_tda_benchmark_summary.txt"
    plot_path = outdir / f"{stem}_pyscf_vs_jax_tda_benchmark.png"

    _write_rows(csv_path, rows)
    _write_summary(
        summary_path,
        db_id=int(args.db_id),
        formula=formula,
        xc=str(args.xc),
        basis=str(args.basis),
        nstates=nstates,
        repeats=int(args.repeats),
        rows=rows,
    )
    _plot(plot_path, rows)

    pyscf_tda_mean, _ = _summarize(rows, backend="pyscf", task="tda")
    pyscf_spec_mean, _ = _summarize(rows, backend="pyscf", task="spectrum")
    jax_auto_tda_jit_mean, _ = _summarize(rows, backend="jax_auto_jit_warm", task="tda")
    jax_auto_spec_jit_mean, _ = _summarize(rows, backend="jax_auto_jit_warm", task="spectrum")
    jax_tda_jit_mean, _ = _summarize(rows, backend="jax_dense_jit_warm", task="tda")
    jax_spec_jit_mean, _ = _summarize(rows, backend="jax_dense_jit_warm", task="spectrum")

    print(f"db_id={args.db_id}")
    print(f"formula={formula}")
    print(f"xc={args.xc}, basis={args.basis}, nstates={nstates}")
    print(f"jax_auto_jit_tda_mean_s={jax_auto_tda_jit_mean:.6f}")
    print(f"pyscf_tda_mean_s={pyscf_tda_mean:.6f}")
    print(f"tda_auto_speedup_vs_pyscf={pyscf_tda_mean / jax_auto_tda_jit_mean:.4f}")
    print(f"jax_dense_jit_tda_mean_s={jax_tda_jit_mean:.6f}")
    print(f"tda_speedup_vs_pyscf={pyscf_tda_mean / jax_tda_jit_mean:.4f}")
    print(f"jax_auto_jit_spectrum_mean_s={jax_auto_spec_jit_mean:.6f}")
    print(f"pyscf_spectrum_mean_s={pyscf_spec_mean:.6f}")
    print(f"spectrum_auto_speedup_vs_pyscf={pyscf_spec_mean / jax_auto_spec_jit_mean:.4f}")
    print(f"jax_dense_jit_spectrum_mean_s={jax_spec_jit_mean:.6f}")
    print(f"spectrum_speedup_vs_pyscf={pyscf_spec_mean / jax_spec_jit_mean:.4f}")
    print(f"csv={csv_path}")
    print(f"summary={summary_path}")
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
