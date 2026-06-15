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
from td_graddft.data.reference import restricted_reference_from_pyscf
from td_graddft.spectra import HARTREE_TO_EV, lorentzian_spectrum


ATOMIC_SYMBOL = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
}


@dataclass(frozen=True)
class TimingRow:
    task: str
    mode: str
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
        description="Benchmark JIT speedup for QH9 JAX-backbone TDA and broadened TDA spectrum."
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
    parser.add_argument("--outdir", default="outputs/qh9_jax_tda_jit_benchmark")
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


def _block_and_time(fn) -> float:
    t0 = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    return float(time.perf_counter() - t0)


def _summarize(rows: list[TimingRow], *, task: str, mode: str) -> tuple[float, float]:
    values = [row.elapsed_s for row in rows if row.task == task and row.mode == mode]
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def _write_rows(path: Path, rows: list[TimingRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "mode", "repeat", "elapsed_s"])
        for row in rows:
            writer.writerow([row.task, row.mode, row.repeat, f"{row.elapsed_s:.10f}"])


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
    auto_nojit_mean, auto_nojit_std = _summarize(rows, task="tda_auto", mode="nojit")
    auto_jit_first_mean, _ = _summarize(rows, task="tda_auto", mode="jit_compile")
    auto_jit_warm_mean, auto_jit_warm_std = _summarize(rows, task="tda_auto", mode="jit_warm")
    spec_auto_nojit_mean, spec_auto_nojit_std = _summarize(rows, task="spectrum_auto", mode="nojit")
    spec_auto_jit_first_mean, _ = _summarize(rows, task="spectrum_auto", mode="jit_compile")
    spec_auto_jit_warm_mean, spec_auto_jit_warm_std = _summarize(rows, task="spectrum_auto", mode="jit_warm")

    with path.open("w", encoding="utf-8") as f:
        f.write("QH9 JAX TDA JIT benchmark\n")
        f.write(f"db_id = {db_id}\n")
        f.write(f"formula = {formula}\n")
        f.write(f"xc = {xc}\n")
        f.write(f"basis = {basis}\n")
        f.write(f"nstates = {nstates}\n")
        f.write(f"repeats = {repeats}\n")
        f.write("\n")
        f.write("notes = benchmark reports the operator/Davidson TD-SCF path.\n")
        f.write("\n")
        f.write(f"tda_auto_nojit_mean_s = {auto_nojit_mean:.10f}\n")
        f.write(f"tda_auto_nojit_std_s = {auto_nojit_std:.10f}\n")
        f.write(f"tda_auto_jit_compile_first_s = {auto_jit_first_mean:.10f}\n")
        f.write(f"tda_auto_jit_warm_mean_s = {auto_jit_warm_mean:.10f}\n")
        f.write(f"tda_auto_jit_warm_std_s = {auto_jit_warm_std:.10f}\n")
        f.write(
            f"tda_auto_warm_speedup = {auto_nojit_mean / auto_jit_warm_mean:.6f}\n"
        )
        f.write("\n")
        f.write(f"spectrum_auto_nojit_mean_s = {spec_auto_nojit_mean:.10f}\n")
        f.write(f"spectrum_auto_nojit_std_s = {spec_auto_nojit_std:.10f}\n")
        f.write(f"spectrum_auto_jit_compile_first_s = {spec_auto_jit_first_mean:.10f}\n")
        f.write(f"spectrum_auto_jit_warm_mean_s = {spec_auto_jit_warm_mean:.10f}\n")
        f.write(f"spectrum_auto_jit_warm_std_s = {spec_auto_jit_warm_std:.10f}\n")
        f.write(
            f"spectrum_auto_warm_speedup = {spec_auto_nojit_mean / spec_auto_jit_warm_mean:.6f}\n"
        )


def _plot(path: Path, rows: list[TimingRow]) -> None:
    labels = [
        ("tda_auto", "nojit", "TDA auto\nno-jit"),
        ("tda_auto", "jit_compile", "TDA auto\njit first"),
        ("tda_auto", "jit_warm", "TDA auto\njit warm"),
        ("spectrum_auto", "nojit", "Spectrum auto\nno-jit"),
        ("spectrum_auto", "jit_compile", "Spectrum auto\njit first"),
        ("spectrum_auto", "jit_warm", "Spectrum auto\njit warm"),
    ]
    means = []
    stds = []
    ticklabels = []
    for task, mode, label in labels:
        mean, std = _summarize(rows, task=task, mode=mode)
        means.append(mean)
        stds.append(std)
        ticklabels.append(label)

    x = np.arange(len(labels), dtype=float)
    colors = [
        "#4C78A8",
        "#9ECAE9",
        "#1F77B4",
        "#72B7B2",
        "#F58518",
        "#54A24B",
    ]

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels(ticklabels)
    ax.set_ylabel("Elapsed Time (s)")
    ax.set_title("QH9 JAX TDA JIT Benchmark")
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
    grid_ev = jnp.linspace(
        float(args.grid_min_ev),
        float(args.grid_max_ev),
        int(args.grid_points),
        dtype=jnp.float32,
    )

    td_auto = tdscf.TDA(
        reference,
        xc_functional=xc_func,
        eigensolver="auto",
    )

    def run_auto_tda():
        return td_auto.kernel(nstates=nstates).excitation_energies

    def run_auto_spectrum():
        result = td_auto.kernel(nstates=nstates)
        strengths = td_auto.oscillator_strength()
        return lorentzian_spectrum(
            result.excitation_energies * HARTREE_TO_EV,
            strengths,
            grid_ev,
            eta=float(args.eta_ev),
        )

    jit_auto_tda = jax.jit(run_auto_tda)
    jit_auto_spectrum = jax.jit(run_auto_spectrum)

    rows: list[TimingRow] = []

    for repeat in range(1, int(args.repeats) + 1):
        rows.append(
            TimingRow(
                task="tda_auto",
                mode="nojit",
                repeat=repeat,
                elapsed_s=_block_and_time(run_auto_tda),
            )
        )
        rows.append(
            TimingRow(
                task="spectrum_auto",
                mode="nojit",
                repeat=repeat,
                elapsed_s=_block_and_time(run_auto_spectrum),
            )
        )

    rows.append(
        TimingRow(
            task="tda_auto",
            mode="jit_compile",
            repeat=1,
            elapsed_s=_block_and_time(jit_auto_tda),
        )
    )
    rows.append(
        TimingRow(
            task="spectrum_auto",
            mode="jit_compile",
            repeat=1,
            elapsed_s=_block_and_time(jit_auto_spectrum),
        )
    )

    for repeat in range(1, int(args.repeats) + 1):
        rows.append(
            TimingRow(
                task="tda_auto",
                mode="jit_warm",
                repeat=repeat,
                elapsed_s=_block_and_time(jit_auto_tda),
            )
        )
        rows.append(
            TimingRow(
                task="spectrum_auto",
                mode="jit_warm",
                repeat=repeat,
                elapsed_s=_block_and_time(jit_auto_spectrum),
            )
        )

    csv_path = outdir / f"qh9_id{args.db_id}_{formula}_{str(args.xc).lower()}_{str(args.basis).lower()}_jit_benchmark.csv"
    summary_path = outdir / f"qh9_id{args.db_id}_{formula}_{str(args.xc).lower()}_{str(args.basis).lower()}_jit_benchmark_summary.txt"
    plot_path = outdir / f"qh9_id{args.db_id}_{formula}_{str(args.xc).lower()}_{str(args.basis).lower()}_jit_benchmark.png"

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

    tda_auto_nojit_mean, _ = _summarize(rows, task="tda_auto", mode="nojit")
    tda_auto_jit_warm_mean, _ = _summarize(rows, task="tda_auto", mode="jit_warm")
    spec_auto_nojit_mean, _ = _summarize(rows, task="spectrum_auto", mode="nojit")
    spec_auto_jit_warm_mean, _ = _summarize(rows, task="spectrum_auto", mode="jit_warm")

    print(f"db_id={args.db_id}")
    print(f"formula={formula}")
    print(f"xc={args.xc}, basis={args.basis}, nstates={nstates}")
    print(f"tda_auto_nojit_mean_s={tda_auto_nojit_mean:.6f}")
    print(f"tda_auto_jit_warm_mean_s={tda_auto_jit_warm_mean:.6f}")
    print(f"tda_auto_warm_speedup={tda_auto_nojit_mean / tda_auto_jit_warm_mean:.4f}")
    print(f"spectrum_auto_nojit_mean_s={spec_auto_nojit_mean:.6f}")
    print(f"spectrum_auto_jit_warm_mean_s={spec_auto_jit_warm_mean:.6f}")
    print(f"spectrum_auto_warm_speedup={spec_auto_nojit_mean / spec_auto_jit_warm_mean:.4f}")
    print(f"csv={csv_path}")
    print(f"summary={summary_path}")
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
