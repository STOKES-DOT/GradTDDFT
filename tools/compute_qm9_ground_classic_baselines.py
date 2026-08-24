#!/usr/bin/env python3
"""Build aligned conventional-functional ground-state baselines for QM9."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any


EV_PER_HARTREE = 27.211386245988
METHODS = {
    "PBE": "pbe",
    "B3LYP": "b3lyp",
    "PBE0": "pbe0",
    "wB97X-D": "wb97xd",
    "wB97M-V": "wb97m-v",
}
OUTPUT_FIELDS = (
    "system",
    "split",
    "method",
    "xc",
    "basis",
    "grid_level",
    "status",
    "scf_converged",
    "target_total_energy_h",
    "scf_energy_h",
    "error_h",
    "abs_err_ev",
    "scf_elapsed_s",
    "error",
    "source",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_targets(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    rows = _read_csv(path)
    targets: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        system = row["system"]
        target = float(row["target_total_energy_h"])
        if not math.isfinite(target):
            raise ValueError(f"non-finite target energy for {system}")
        if system in targets:
            raise ValueError(f"duplicate target row for {system}")
        order.append(system)
        targets[system] = {
            "split": row.get("split", "validation"),
            "target_h": target,
        }
    if not targets:
        raise ValueError(f"no targets found in {path}")
    return order, targets


def _load_geometries(path: Path, systems: set[str]) -> dict[str, dict[str, Any]]:
    geometries: dict[str, dict[str, Any]] = {}
    for row in _read_csv(path):
        system = row["system"]
        if system not in systems:
            continue
        geometries[system] = {
            "atom": row["atom"],
            "unit": row.get("unit", "Angstrom") or "Angstrom",
            "charge": int(row.get("charge", 0) or 0),
            "spin": int(row.get("spin", 0) or 0),
            "cart": str(row.get("cart", "True")).strip().lower()
            in {"1", "true", "yes"},
        }
    missing = systems - set(geometries)
    if missing:
        raise ValueError(f"missing geometries for {sorted(missing)}")
    return geometries


def _load_energy_cache(paths: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in _read_csv(path):
            method = row.get("method", "")
            system = row.get("system", "")
            if method not in METHODS or not system:
                continue
            if row.get("status", "ok").strip().lower() != "ok":
                continue
            energy = float(row["scf_energy_h"])
            if not math.isfinite(energy):
                continue
            cache[(system, method)] = {
                "energy_h": energy,
                "elapsed_s": float(row.get("scf_elapsed_s", "nan") or "nan"),
                "source": str(path),
            }
    return cache


def _run_scf(
    geometry: dict[str, Any],
    *,
    basis: str,
    xc: str,
    grid_level: int,
    device: str,
) -> tuple[float, bool, float]:
    from pyscf import dft, gto

    if device == "gpu":
        import gpu4pyscf  # noqa: F401

    molecule = gto.M(
        atom=geometry["atom"],
        basis=basis,
        unit=geometry["unit"],
        charge=geometry["charge"],
        spin=geometry["spin"],
        cart=geometry["cart"],
        verbose=0,
    )
    mean_field = dft.RKS(molecule)
    mean_field.xc = xc
    mean_field.grids.level = int(grid_level)
    mean_field.conv_tol = 1.0e-10
    mean_field.max_cycle = 120
    if device == "gpu":
        if not hasattr(mean_field, "to_gpu"):
            raise RuntimeError("GPU4PySCF conversion is unavailable")
        mean_field = mean_field.to_gpu()

    start = time.perf_counter()
    energy = float(mean_field.kernel())
    if device == "gpu":
        import cupy

        cupy.cuda.get_current_stream().synchronize()
    elapsed = time.perf_counter() - start
    return energy, bool(mean_field.converged), elapsed


def _make_row(
    *,
    system: str,
    split: str,
    method: str,
    basis: str,
    grid_level: int,
    target_h: float,
    energy_h: float,
    converged: bool,
    elapsed_s: float,
    source: str,
) -> dict[str, Any]:
    error_h = energy_h - target_h
    return {
        "system": system,
        "split": split,
        "method": method,
        "xc": METHODS[method],
        "basis": basis,
        "grid_level": grid_level,
        "status": "ok" if converged else "failed",
        "scf_converged": converged,
        "target_total_energy_h": target_h,
        "scf_energy_h": energy_h,
        "error_h": error_h,
        "abs_err_ev": abs(error_h) * EV_PER_HARTREE,
        "scf_elapsed_s": elapsed_s,
        "error": "" if converged else "SCF did not converge",
        "source": source,
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--geometries", type=Path, required=True)
    parser.add_argument("--existing", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--basis", default="def2-tzvp")
    parser.add_argument("--grid-level", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()

    target_order, targets = _load_targets(args.targets)
    geometries = _load_geometries(args.geometries, set(targets))
    cache_paths = [*args.existing, args.output]
    cache = _load_energy_cache(cache_paths)
    output_rows: list[dict[str, Any]] = []

    for system in target_order:
        for method, xc in METHODS.items():
            cached = cache.get((system, method))
            if cached is None:
                print(f"[scf] {system} {method}/{args.basis}", flush=True)
                energy, converged, elapsed = _run_scf(
                    geometries[system],
                    basis=args.basis,
                    xc=xc,
                    grid_level=args.grid_level,
                    device=args.device,
                )
                source = "computed"
                cache[(system, method)] = {
                    "energy_h": energy,
                    "elapsed_s": elapsed,
                    "source": source,
                }
            else:
                energy = float(cached["energy_h"])
                elapsed = float(cached["elapsed_s"])
                converged = True
                source = f"reused:{cached['source']}"

            output_rows.append(
                _make_row(
                    system=system,
                    split=str(targets[system]["split"]),
                    method=method,
                    basis=args.basis,
                    grid_level=args.grid_level,
                    target_h=float(targets[system]["target_h"]),
                    energy_h=energy,
                    converged=converged,
                    elapsed_s=elapsed,
                    source=source,
                )
            )
            _write_rows(args.output, output_rows)

    summary: dict[str, Any] = {
        "targets": str(args.targets.resolve()),
        "geometries": str(args.geometries.resolve()),
        "existing": [str(path.resolve()) for path in args.existing],
        "output": str(args.output.resolve()),
        "basis": args.basis,
        "grid_level": args.grid_level,
        "methods": {},
    }
    for method in METHODS:
        method_rows = [row for row in output_rows if row["method"] == method]
        errors = [float(row["abs_err_ev"]) for row in method_rows]
        summary["methods"][method] = {
            "n": len(method_rows),
            "mae_ev": sum(errors) / len(errors),
            "max_abs_err_ev": max(errors),
        }
    summary_path = args.output.with_name(f"{args.output.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
