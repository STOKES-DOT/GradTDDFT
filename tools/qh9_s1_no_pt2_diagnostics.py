from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pyscf import dft, gto


HARTREE_TO_EV = 27.211386245988


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "No-PT2 QH9 S1 diagnostics: compare PySCF B3LYP TDA/full TDDFT "
            "against EOM-EE-CCSD S1 references from CSV."
        )
    )
    p.add_argument("--reference-csv", required=True)
    p.add_argument("--outdir", default="outputs/qh9_s1_no_pt2_theory_diagnostics")
    p.add_argument("--xc", default="b3lyp")
    p.add_argument("--grids-level", type=int, default=1)
    p.add_argument("--nstates", type=int, default=3)
    p.add_argument("--scf-conv-tol", type=float, default=1e-10)
    p.add_argument("--scf-max-cycle", type=int, default=120)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args(argv)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"[{_timestamp()}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def metric(values: Iterable[float]) -> dict[str, float | int | None]:
    arr = np.asarray([float(x) for x in values if math.isfinite(float(x))], dtype=float)
    if arr.size == 0:
        return {
            "count": 0,
            "mae_ev": None,
            "rmse_ev": None,
            "max_abs_ev": None,
            "mean_signed_ev": None,
        }
    return {
        "count": int(arr.size),
        "mae_ev": float(np.mean(np.abs(arr))),
        "rmse_ev": float(np.sqrt(np.mean(arr * arr))),
        "max_abs_ev": float(np.max(np.abs(arr))),
        "mean_signed_ev": float(np.mean(arr)),
    }


def _run_rks_reference(
    row: dict[str, str],
    *,
    xc: str,
    grids_level: int,
    scf_conv_tol: float,
    scf_max_cycle: int,
):
    mol = gto.M(
        atom=str(row["atom"]).replace(";", "\n"),
        basis=str(row["basis"]),
        unit=str(row["unit"]),
        charge=int(row["charge"]),
        spin=int(row["spin"]),
        cart=str(row.get("cart", "True")).strip().lower() == "true",
        verbose=0,
    )
    attempts = (
        dict(init_guess="minao", damping=0.15, level_shift=0.0, max_cycle=scf_max_cycle, use_newton=False),
        dict(init_guess="atom", damping=0.3, level_shift=0.5, max_cycle=max(scf_max_cycle, 180), use_newton=False),
        dict(init_guess="atom", damping=0.0, level_shift=0.0, max_cycle=max(scf_max_cycle, 120), use_newton=True),
    )
    last_mf = None
    for cfg in attempts:
        mf = dft.RKS(mol)
        mf.xc = str(xc)
        mf.grids.level = int(grids_level)
        mf.conv_tol = float(scf_conv_tol)
        mf.max_cycle = int(cfg["max_cycle"])
        mf.damping = float(cfg["damping"])
        mf.level_shift = float(cfg["level_shift"])
        mf.diis_start_cycle = 1
        mf.init_guess = str(cfg["init_guess"])
        if cfg["use_newton"]:
            mf = mf.newton()
            mf.xc = str(xc)
            mf.grids.level = int(grids_level)
            mf.conv_tol = float(scf_conv_tol)
            mf.max_cycle = int(cfg["max_cycle"])
        mf.kernel()
        last_mf = mf
        if bool(mf.converged):
            return mf
    raise RuntimeError(f"PySCF RKS did not converge; last={last_mf!r}")


def _run_response(mf, *, solver: str, nstates: int) -> tuple[list[float], list[float], float]:
    td = mf.TDA() if solver == "tda" else mf.TDDFT()
    td.nstates = int(nstates)
    if hasattr(td, "singlet"):
        td.singlet = True
    t0 = time.perf_counter()
    result = td.kernel()
    elapsed_s = time.perf_counter() - t0
    energies = getattr(td, "e", None)
    if energies is None:
        energies = result[0] if isinstance(result, tuple) else result
    energies_arr = np.asarray(energies, dtype=np.complex128).reshape(-1)
    energies_arr = np.real(energies_arr[np.isfinite(energies_arr)])
    energies_arr = np.sort(energies_arr[energies_arr > 1e-8])
    roots_ev = [float(value * HARTREE_TO_EV) for value in energies_arr.tolist()]
    osc: list[float] = []
    try:
        osc_arr = np.asarray(td.oscillator_strength(), dtype=float).reshape(-1)
        osc = [float(value) for value in osc_arr[: len(roots_ev)]]
    except Exception:
        osc = []
    return roots_ev, osc, float(elapsed_s)


def _load_rows(path: Path, *, limit: int | None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    return rows


def _plot_errors(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    x = np.arange(len(rows))
    tda_err = np.asarray([float(row.get("b3lyp_tda_s1_err_ev", np.nan)) for row in rows], dtype=float)
    casida_err = np.asarray([float(row.get("b3lyp_casida_s1_err_ev", np.nan)) for row in rows], dtype=float)
    labels = [str(row["system"]).replace("qh9_", "") for row in rows]
    ax.axhline(0.0, color="#242424", lw=1.0)
    ax.plot(x, tda_err, "o-", ms=4, lw=1.2, label="B3LYP TDA - EOM-CCSD S1")
    ax.plot(x, casida_err, "s-", ms=4, lw=1.2, label="B3LYP full TDDFT - EOM-CCSD S1")
    train_indices = [idx for idx, row in enumerate(rows) if row.get("split") == "train"]
    if train_indices and len(train_indices) < len(rows):
        boundary = max(train_indices) + 0.5
        ax.axvline(boundary, color="#888888", lw=1.0, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("S1 error vs EOM-EE-CCSD (eV)")
    ax.set_title("QH9 no-PT2 B3LYP response baseline")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_outputs(
    outdir: Path,
    rows: list[dict[str, object]],
    *,
    reference_csv: Path,
    basis: str,
    xc: str,
    grids_level: int,
) -> dict[str, object]:
    fieldnames = [
        "system",
        "split",
        "target_eomccsd_s1_ev",
        "b3lyp_tda_s1_ev",
        "b3lyp_tda_s1_err_ev",
        "b3lyp_casida_s1_ev",
        "b3lyp_casida_s1_err_ev",
        "tda_minus_casida_s1_ev",
        "b3lyp_tda_roots_ev_json",
        "b3lyp_casida_roots_ev_json",
        "b3lyp_tda_osc_json",
        "b3lyp_casida_osc_json",
        "b3lyp_tda_elapsed_s",
        "b3lyp_casida_elapsed_s",
        "rks_energy_h",
        "rks_elapsed_s",
        "rks_status",
        "b3lyp_tda_status",
        "b3lyp_casida_status",
        "notes",
    ]
    csv_path = outdir / "baseline_pyscf_tda_casida.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {
        "reference_csv": str(reference_csv),
        "basis": str(basis),
        "xc": str(xc),
        "grids_level": int(grids_level),
        "count": len(rows),
        "splits": {},
        "csv": str(csv_path),
        "figure": str(outdir / "baseline_s1_errors.png"),
    }
    split_names = sorted({str(row["split"]) for row in rows})
    split_summary: dict[str, object] = {}
    for split in split_names + ["all"]:
        subset = rows if split == "all" else [row for row in rows if row["split"] == split]
        split_summary[split] = {
            "count": len(subset),
            "tda_error": metric([float(row.get("b3lyp_tda_s1_err_ev", np.nan)) for row in subset]),
            "casida_error": metric([float(row.get("b3lyp_casida_s1_err_ev", np.nan)) for row in subset]),
            "tda_minus_casida": metric([float(row.get("tda_minus_casida_s1_ev", np.nan)) for row in subset]),
        }
    summary["splits"] = split_summary
    (outdir / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _plot_errors(outdir / "baseline_s1_errors.png", rows)
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    reference_csv = Path(args.reference_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(outdir / "baseline_pyscf_tda_casida.log")
    ref_rows = _load_rows(reference_csv, limit=args.limit)
    out_rows: list[dict[str, object]] = []
    basis_values = sorted({str(row["basis"]) for row in ref_rows})
    basis = basis_values[0] if basis_values else ""
    for idx, row in enumerate(ref_rows, start=1):
        system = str(row["system"])
        target_ev = float(row["s1_excitation_ev"])
        record: dict[str, object] = {
            "system": system,
            "split": str(row["split"]),
            "target_eomccsd_s1_ev": target_ev,
            "notes": row.get("notes", ""),
            "rks_status": "pending",
        }
        logger.log(f"[{idx:02d}/{len(ref_rows)}] {system}: RKS/{args.xc} grid={args.grids_level}")
        try:
            t0 = time.perf_counter()
            mf = _run_rks_reference(
                row,
                xc=str(args.xc),
                grids_level=int(args.grids_level),
                scf_conv_tol=float(args.scf_conv_tol),
                scf_max_cycle=int(args.scf_max_cycle),
            )
            record["rks_energy_h"] = float(mf.e_tot)
            record["rks_elapsed_s"] = float(time.perf_counter() - t0)
            record["rks_status"] = "ok"
            for solver in ("tda", "casida"):
                try:
                    roots_ev, osc, elapsed_s = _run_response(
                        mf,
                        solver=solver,
                        nstates=int(args.nstates),
                    )
                    s1_ev = roots_ev[0] if roots_ev else float("nan")
                    record[f"b3lyp_{solver}_roots_ev_json"] = json.dumps(roots_ev)
                    record[f"b3lyp_{solver}_osc_json"] = json.dumps(osc)
                    record[f"b3lyp_{solver}_s1_ev"] = float(s1_ev)
                    record[f"b3lyp_{solver}_s1_err_ev"] = float(s1_ev - target_ev)
                    record[f"b3lyp_{solver}_elapsed_s"] = float(elapsed_s)
                    record[f"b3lyp_{solver}_status"] = "ok"
                    logger.log(
                        f"    {solver}: S1={s1_ev:.6f} eV "
                        f"err={float(s1_ev - target_ev):+.6f} eV"
                    )
                except Exception as exc:
                    record[f"b3lyp_{solver}_roots_ev_json"] = "[]"
                    record[f"b3lyp_{solver}_osc_json"] = "[]"
                    record[f"b3lyp_{solver}_s1_ev"] = float("nan")
                    record[f"b3lyp_{solver}_s1_err_ev"] = float("nan")
                    record[f"b3lyp_{solver}_elapsed_s"] = float("nan")
                    record[f"b3lyp_{solver}_status"] = repr(exc)
                    logger.log(f"    {solver}: FAIL {exc!r}")
        except Exception as exc:
            record["rks_energy_h"] = float("nan")
            record["rks_elapsed_s"] = float("nan")
            record["rks_status"] = repr(exc)
            logger.log(f"    RKS FAIL {exc!r}")
        record["tda_minus_casida_s1_ev"] = (
            float(record.get("b3lyp_tda_s1_ev", float("nan")))
            - float(record.get("b3lyp_casida_s1_ev", float("nan")))
        )
        out_rows.append(record)
    summary = _write_outputs(
        outdir,
        out_rows,
        reference_csv=reference_csv,
        basis=basis,
        xc=str(args.xc),
        grids_level=int(args.grids_level),
    )
    logger.log(f"wrote {outdir / 'baseline_pyscf_tda_casida.csv'}")
    logger.log(f"summary all={summary['splits']['all']}")


if __name__ == "__main__":
    main()
