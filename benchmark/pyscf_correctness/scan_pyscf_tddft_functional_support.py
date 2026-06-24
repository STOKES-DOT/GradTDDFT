from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyscf.dft import libxc


DERIV_FIELDS = [
    "timestamp_utc",
    "name",
    "source",
    "expression",
    "status",
    "error_type",
    "error_message",
    "xc_type",
    "hybrid_coeff",
    "max_deriv_order",
    "supports_deriv2",
]

SMOKE_FIELDS = [
    "timestamp_utc",
    "name",
    "source",
    "expression",
    "status",
    "error_type",
    "error_message",
    "timeout_s",
    "elapsed_s",
    "xc_type",
    "hybrid_coeff",
    "max_deriv_order",
    "supports_deriv2",
    "scf_converged",
    "scf_energy_ha",
    "tda_status",
    "tda_excitation_ha",
    "tddft_status",
    "tddft_excitation_ha",
    "stdout_tail",
    "stderr_tail",
]

COMMON_EXTRA_XCS = [
    "hf",
    "b3lyp",
    "b3lyp5",
    "pbe0",
    "pbe50",
    "cam-b3lyp",
    "camb3lyp",
    "wb97x",
    "wb97x-d",
    "wb97x_d",
    "wb97x-v",
    "wb97x_v",
    "wb97m-v",
    "wb97m_v",
    "b97m-v",
    "b97m_v",
    "b97-d",
    "b97_d",
    "tpssh",
    "bhhlyp",
    "bhandhlyp",
    "m06-l",
    "m06_l",
    "m06-2x",
    "m06_2x",
    "m06-hf",
    "m06_hf",
    "mn15",
    "mn15-l",
    "mn15_l",
    "scan",
    "rscan",
    "r2scan",
    "scan-vv10",
    "scan_vv10",
    "revscan-vv10",
    "revscan_vv10",
    "b2plyp",
    "b2gpplyp",
    "dsd-blyp",
    "dsd_blyp",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tail(text: str, *, limit: int = 800) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[-limit:]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _environment() -> dict[str, Any]:
    try:
        import pyscf
    except Exception:
        pyscf = None
    return {
        "timestamp_utc": _now(),
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "pyscf_version": getattr(pyscf, "__version__", "") if pyscf is not None else "",
        "xc_codes_count": len(getattr(libxc, "XC_CODES", {})),
        "xc_alias_count": len(getattr(libxc, "XC_ALIAS", {})),
    }


def _candidate_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    for name in sorted(libxc.XC_CODES):
        key = str(name).lower()
        if key in seen:
            continue
        rows.append({"name": str(name), "source": "libxc_code", "expression": str(name)})
        seen.add(key)

    for name, expression in sorted(libxc.XC_ALIAS.items()):
        key = str(name).lower()
        if key in seen:
            continue
        rows.append({"name": str(name), "source": "pyscf_alias", "expression": str(expression)})
        seen.add(key)

    for name in COMMON_EXTRA_XCS:
        key = str(name).lower()
        if key in seen:
            continue
        rows.append({"name": str(name), "source": "common_extra", "expression": str(name)})
        seen.add(key)

    return rows


def _derivative_support(row: dict[str, str]) -> dict[str, Any]:
    name = row["name"]
    out: dict[str, Any] = {
        "timestamp_utc": _now(),
        "name": name,
        "source": row["source"],
        "expression": row["expression"],
    }
    try:
        out.update(
            {
                "status": "ok",
                "xc_type": libxc.xc_type(name),
                "hybrid_coeff": float(libxc.hybrid_coeff(name)),
                "max_deriv_order": int(libxc.max_deriv_order(name)),
                "supports_deriv2": bool(libxc.test_deriv_order(name, 2, raise_error=False)),
            }
        )
    except Exception as exc:
        out.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc).splitlines()[0] if str(exc) else repr(exc),
                "supports_deriv2": False,
            }
        )
    return out


def _smoke_code() -> str:
    return r"""
import json
import os
import sys
from pyscf import dft, gto
from pyscf.dft import libxc

xc = sys.argv[1]
mol = gto.M(
    atom="H 0 0 0; H 0 0 0.7414",
    basis="sto-3g",
    unit="Angstrom",
    verbose=0,
)
row = {"name": xc}
try:
    row["xc_type"] = libxc.xc_type(xc)
    row["hybrid_coeff"] = float(libxc.hybrid_coeff(xc))
    row["max_deriv_order"] = int(libxc.max_deriv_order(xc))
    row["supports_deriv2"] = bool(libxc.test_deriv_order(xc, 2, raise_error=False))
except Exception as exc:
    row["status"] = "error"
    row["error_stage"] = "deriv_check"
    row["error_type"] = type(exc).__name__
    row["error_message"] = str(exc).splitlines()[0] if str(exc) else repr(exc)
    print(json.dumps(row, sort_keys=True))
    raise SystemExit(0)

try:
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = 0
    mf.conv_tol = 1e-8
    mf.max_cycle = 80
    mf.kernel()
    row["scf_converged"] = bool(mf.converged)
    row["scf_energy_ha"] = float(mf.e_tot)
except Exception as exc:
    row["status"] = "error"
    row["error_stage"] = "scf"
    row["error_type"] = type(exc).__name__
    row["error_message"] = str(exc).splitlines()[0] if str(exc) else repr(exc)
    print(json.dumps(row, sort_keys=True))
    raise SystemExit(0)

try:
    td = mf.TDA()
    td.nstates = 1
    td.conv_tol = 1e-5
    td.max_cycle = 50
    td.kernel()
    row["tda_status"] = "ok"
    row["tda_excitation_ha"] = float(td.e[0])
except Exception as exc:
    row["tda_status"] = "error"
    row["error_type"] = type(exc).__name__
    row["error_message"] = str(exc).splitlines()[0] if str(exc) else repr(exc)

try:
    td = mf.TDDFT()
    td.nstates = 1
    td.conv_tol = 1e-5
    td.max_cycle = 50
    td.kernel()
    row["tddft_status"] = "ok"
    row["tddft_excitation_ha"] = float(td.e[0])
except Exception as exc:
    row["tddft_status"] = "error"
    row["error_type"] = row.get("error_type") or type(exc).__name__
    row["error_message"] = row.get("error_message") or (
        str(exc).splitlines()[0] if str(exc) else repr(exc)
    )

row["status"] = "ok" if row.get("tda_status") == "ok" and row.get("tddft_status") == "ok" else "error"
print(json.dumps(row, sort_keys=True))
"""


def _smoke_one(
    row: dict[str, str],
    deriv_row: dict[str, Any],
    *,
    timeout_s: float,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    start = time.perf_counter()
    out: dict[str, Any] = {
        "timestamp_utc": _now(),
        "name": row["name"],
        "source": row["source"],
        "expression": row["expression"],
        "timeout_s": float(timeout_s),
        "xc_type": deriv_row.get("xc_type", ""),
        "hybrid_coeff": deriv_row.get("hybrid_coeff", ""),
        "max_deriv_order": deriv_row.get("max_deriv_order", ""),
        "supports_deriv2": deriv_row.get("supports_deriv2", ""),
    }
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _smoke_code(), row["name"]],
            text=True,
            capture_output=True,
            timeout=float(timeout_s),
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        out.update(
            {
                "status": "timeout",
                "error_type": "TimeoutExpired",
                "error_message": f"Exceeded {timeout_s:.1f} s",
                "elapsed_s": time.perf_counter() - start,
                "stdout_tail": _tail(exc.stdout or ""),
                "stderr_tail": _tail(exc.stderr or ""),
            }
        )
        return out

    out["elapsed_s"] = time.perf_counter() - start
    out["stdout_tail"] = _tail(proc.stdout or "")
    out["stderr_tail"] = _tail(proc.stderr or "")
    parsed: dict[str, Any] | None = None
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if parsed is None:
        out.update(
            {
                "status": "error",
                "error_type": "NoJsonResult",
                "error_message": f"returncode={proc.returncode}",
            }
        )
        return out
    out.update(
        {
            "status": parsed.get("status", "error"),
            "error_type": parsed.get("error_type", ""),
            "error_message": parsed.get("error_message", ""),
            "xc_type": parsed.get("xc_type", out.get("xc_type", "")),
            "hybrid_coeff": parsed.get("hybrid_coeff", out.get("hybrid_coeff", "")),
            "max_deriv_order": parsed.get("max_deriv_order", out.get("max_deriv_order", "")),
            "supports_deriv2": parsed.get("supports_deriv2", out.get("supports_deriv2", "")),
            "scf_converged": parsed.get("scf_converged", ""),
            "scf_energy_ha": parsed.get("scf_energy_ha", ""),
            "tda_status": parsed.get("tda_status", ""),
            "tda_excitation_ha": parsed.get("tda_excitation_ha", ""),
            "tddft_status": parsed.get("tddft_status", ""),
            "tddft_excitation_ha": parsed.get("tddft_excitation_ha", ""),
        }
    )
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan PySCF XC functionals for TDDFT derivative and smoke-test support.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/pyscf_correctness/runs/pyscf_tddft_functional_support"),
    )
    parser.add_argument(
        "--smoke-source",
        choices=["none", "aliases", "common", "aliases-and-common"],
        default="aliases-and-common",
    )
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--max-smoke", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_rows()
    deriv_rows = [_derivative_support(row) for row in candidates]
    _write_csv(args.output_dir / "pyscf_xc_deriv2_support.csv", deriv_rows, DERIV_FIELDS)
    (args.output_dir / "environment.json").write_text(
        json.dumps(_environment(), indent=2, sort_keys=True) + "\n"
    )

    deriv_by_name = {row["name"].lower(): row for row in deriv_rows}
    smoke_candidates: list[dict[str, str]] = []
    if args.smoke_source != "none":
        for row in candidates:
            source = row["source"]
            if args.smoke_source == "aliases" and source != "pyscf_alias":
                continue
            if args.smoke_source == "common" and source != "common_extra":
                continue
            if args.smoke_source == "aliases-and-common" and source not in {
                "pyscf_alias",
                "common_extra",
            }:
                continue
            smoke_candidates.append(row)
    if args.max_smoke is not None:
        smoke_candidates = smoke_candidates[: int(args.max_smoke)]

    smoke_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(smoke_candidates, start=1):
        deriv_row = deriv_by_name.get(row["name"].lower(), {})
        smoke_row = _smoke_one(row, deriv_row, timeout_s=args.timeout_s)
        smoke_rows.append(smoke_row)
        print(
            json.dumps(
                {
                    "index": idx,
                    "total": len(smoke_candidates),
                    "name": row["name"],
                    "status": smoke_row.get("status"),
                    "tda_status": smoke_row.get("tda_status"),
                    "tddft_status": smoke_row.get("tddft_status"),
                    "error_type": smoke_row.get("error_type"),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if smoke_rows:
        _write_csv(args.output_dir / "pyscf_tddft_smoke_support.csv", smoke_rows, SMOKE_FIELDS)
    unsupported = [
        row
        for row in smoke_rows
        if row.get("status") != "ok"
        or str(row.get("supports_deriv2", "")).lower() not in {"true", "1"}
    ]
    if unsupported:
        _write_csv(args.output_dir / "pyscf_tddft_unsupported_smoke.csv", unsupported, SMOKE_FIELDS)
    else:
        _write_csv(args.output_dir / "pyscf_tddft_unsupported_smoke.csv", [], SMOKE_FIELDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
