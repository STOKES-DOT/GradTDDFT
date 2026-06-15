from __future__ import annotations

import argparse
import csv
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

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
class CompareResult:
    db_id: int
    formula: str
    xc: str
    basis: str
    nstates: int
    compared_states: int
    mae_energy_ev: float
    max_energy_ev: float
    mae_oscillator: float
    max_oscillator: float


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
        description="Compare PySCF TDA and JAX-backbone TDA absorption spectra on one QH9 molecule."
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
    parser.add_argument(
        "--allow-approx-b3lyp",
        action="store_true",
        help=(
            "allow B3LYP comparison even though current jax_libxc uses an approximate "
            "B3LYP alias rather than a strict libxc-identical kernel path"
        ),
    )
    parser.add_argument("--outdir", default="outputs/qh9_pyscf_vs_jax_tda")
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


def _build_mf(z: np.ndarray, pos_ang: np.ndarray, *, basis: str, xc: str):
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
    return mf


def _write_state_csv(
    path: Path,
    ref_e_ev: np.ndarray,
    pred_e_ev: np.ndarray,
    ref_f: np.ndarray,
    pred_f: np.ndarray,
) -> None:
    n = min(ref_e_ev.size, pred_e_ev.size, ref_f.size, pred_f.size)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "state",
                "pyscf_tda_energy_ev",
                "jax_tda_energy_ev",
                "abs_diff_ev",
                "pyscf_tda_oscillator_strength",
                "jax_tda_oscillator_strength",
                "abs_diff_oscillator_strength",
            ]
        )
        for idx in range(n):
            writer.writerow(
                [
                    idx + 1,
                    float(ref_e_ev[idx]),
                    float(pred_e_ev[idx]),
                    float(abs(pred_e_ev[idx] - ref_e_ev[idx])),
                    float(ref_f[idx]),
                    float(pred_f[idx]),
                    float(abs(pred_f[idx] - ref_f[idx])),
                ]
            )


def _write_curve_csv(
    path: Path,
    grid_ev: np.ndarray,
    ref_curve: np.ndarray,
    pred_curve: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["energy_ev", "pyscf_tda_absorption", "jax_tda_absorption"])
        for x, y_ref, y_pred in zip(grid_ev, ref_curve, pred_curve, strict=True):
            writer.writerow([float(x), float(y_ref), float(y_pred)])


def _write_summary(
    path: Path,
    *,
    result: CompareResult,
    grid_min_ev: float,
    grid_max_ev: float,
    eta_ev: float,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("QH9 single-molecule PySCF TDA vs JAX TDA comparison\n")
        f.write(f"db_id = {result.db_id}\n")
        f.write(f"formula = {result.formula}\n")
        f.write(f"xc = {result.xc}\n")
        f.write(f"basis = {result.basis}\n")
        f.write(f"nstates = {result.nstates}\n")
        f.write(f"compared_states = {result.compared_states}\n")
        f.write(f"spectrum_grid = [{grid_min_ev}, {grid_max_ev}] eV\n")
        f.write(f"eta_ev = {eta_ev}\n")
        f.write(f"mae_energy_ev = {result.mae_energy_ev:.10f}\n")
        f.write(f"max_energy_ev = {result.max_energy_ev:.10f}\n")
        f.write(f"mae_oscillator = {result.mae_oscillator:.10e}\n")
        f.write(f"max_oscillator = {result.max_oscillator:.10e}\n")


def main() -> None:
    args = parse_args()
    db_path = Path(args.db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"QH9 db not found: {db_path}")

    xc_key = str(args.xc).strip().lower()
    if xc_key == "b3lyp" and not args.allow_approx_b3lyp:
        raise ValueError(
            "Strict same-functional comparison is not valid for xc='b3lyp' in current "
            "jax_libxc. Use --allow-approx-b3lyp to run the approximate comparison explicitly, "
            "or choose pbe/lda for a stricter check."
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    z, pos_ang = _fetch_molecule(db_path, int(args.db_id))
    formula = _formula_from_z(z)
    mf = _build_mf(z, pos_ang, basis=str(args.basis), xc=str(args.xc))

    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 1e-8))
    nvir = int(np.asarray(mf.mo_coeff).shape[-1] - nocc)
    nstates = min(max(1, int(args.nstates)), nocc * nvir)

    td = mf.TDA()
    td.nstates = nstates
    td.kernel()
    ref_e = np.asarray(td.e, dtype=float).reshape(-1)
    ref_f = np.asarray(td.oscillator_strength(), dtype=float).reshape(-1)

    reference = restricted_reference_from_pyscf(mf)
    jax_td = tdscf.TDA(reference, xc_functional=args.xc)
    pred = jax_td.kernel(nstates=nstates)
    pred_e = np.asarray(pred.excitation_energies, dtype=float).reshape(-1)
    pred_f = np.asarray(jax_td.oscillator_strength(), dtype=float).reshape(-1)

    n = min(ref_e.size, pred_e.size, ref_f.size, pred_f.size, nstates)
    if n <= 0:
        raise RuntimeError("No common excited states available for spectrum comparison.")

    ref_e_ev = ref_e[:n] * HARTREE_TO_EV
    pred_e_ev = pred_e[:n] * HARTREE_TO_EV
    ref_f_n = ref_f[:n]
    pred_f_n = pred_f[:n]

    diff_ev = np.abs(pred_e_ev - ref_e_ev)
    diff_f = np.abs(pred_f_n - ref_f_n)
    result = CompareResult(
        db_id=int(args.db_id),
        formula=formula,
        xc=str(args.xc),
        basis=str(args.basis),
        nstates=nstates,
        compared_states=n,
        mae_energy_ev=float(np.mean(diff_ev)),
        max_energy_ev=float(np.max(diff_ev)),
        mae_oscillator=float(np.mean(diff_f)),
        max_oscillator=float(np.max(diff_f)),
    )

    grid_ev = np.linspace(float(args.grid_min_ev), float(args.grid_max_ev), int(args.grid_points), dtype=float)
    ref_curve = np.asarray(
        lorentzian_spectrum(
            jnp.asarray(ref_e_ev),
            jnp.asarray(ref_f_n),
            jnp.asarray(grid_ev),
            eta=float(args.eta_ev),
        ),
        dtype=float,
    )
    pred_curve = np.asarray(
        lorentzian_spectrum(
            jnp.asarray(pred_e_ev),
            jnp.asarray(pred_f_n),
            jnp.asarray(grid_ev),
            eta=float(args.eta_ev),
        ),
        dtype=float,
    )

    stem = f"qh9_id{args.db_id}_{formula}_{str(args.xc).lower()}_{str(args.basis).lower()}".replace("*", "star")
    state_csv = outdir / f"{stem}_state_compare.csv"
    curve_csv = outdir / f"{stem}_spectrum_curve.csv"
    spectrum_png = outdir / f"{stem}_spectrum_compare.png"
    summary_txt = outdir / f"{stem}_summary.txt"

    _write_state_csv(state_csv, ref_e_ev, pred_e_ev, ref_f_n, pred_f_n)
    _write_curve_csv(curve_csv, grid_ev, ref_curve, pred_curve)
    _write_summary(
        summary_txt,
        result=result,
        grid_min_ev=float(args.grid_min_ev),
        grid_max_ev=float(args.grid_max_ev),
        eta_ev=float(args.eta_ev),
    )

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(grid_ev, ref_curve, lw=2.1, label="PySCF TDA")
    ax.plot(grid_ev, pred_curve, lw=2.0, label="JAX TDA")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Absorption (arb. units)")
    ax.set_title(f"QH9 id={args.db_id} {formula} | {str(args.xc).upper()}/{str(args.basis).upper()}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(spectrum_png, dpi=180)
    plt.close(fig)

    print(f"db_id={result.db_id}")
    print(f"formula={result.formula}")
    print(f"xc={result.xc}, basis={result.basis}, nstates={result.nstates}")
    print(f"compared_states={result.compared_states}")
    print(f"mae_energy_ev={result.mae_energy_ev:.8f}")
    print(f"max_energy_ev={result.max_energy_ev:.8f}")
    print(f"mae_oscillator={result.mae_oscillator:.8e}")
    print(f"max_oscillator={result.max_oscillator:.8e}")
    print(f"state_csv={state_csv}")
    print(f"spectrum_curve_csv={curve_csv}")
    print(f"spectrum_png={spectrum_png}")
    print(f"summary_txt={summary_txt}")


if __name__ == "__main__":
    main()
