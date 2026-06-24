from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLCONFIGDIR", str(Path("benchmark") / ".mplconfig"))

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import dft, gto

from td_graddft.data.reference import restricted_reference_from_pyscf
from td_graddft.features import restricted_grid_response_variables
from td_graddft.tddft._semilocal_response import (
    _GRID_RESPONSE_TENSOR_CACHE,
    SemilocalResponseFunctional,
)
from td_graddft.xc_backend.jax_libxc import eval_xc_response_tensor


MOLECULES: dict[str, dict[str, Any]] = {
    "water": {
        "atom": """
O  0.000000  0.000000  0.117790
H  0.000000  0.755453 -0.471161
H  0.000000 -0.755453 -0.471161
""",
        "charge": 0,
        "spin": 0,
    },
    "co": {
        "atom": "C 0.000000 0.000000 0.000000; O 0.000000 0.000000 1.128200",
        "charge": 0,
        "spin": 0,
    },
    "n2": {
        "atom": "N 0.000000 0.000000 -0.548850; N 0.000000 0.000000 0.548850",
        "charge": 0,
        "spin": 0,
    },
    "ethylene": {
        "atom": """
C -0.669500  0.000000 0.000000
C  0.669500  0.000000 0.000000
H -1.232900  0.928900 0.000000
H -1.232900 -0.928900 0.000000
H  1.232900  0.928900 0.000000
H  1.232900 -0.928900 0.000000
""",
        "charge": 0,
        "spin": 0,
    },
    "formaldehyde": {
        "atom": """
C  0.000000  0.000000  0.000000
O  0.000000  0.000000  1.208000
H  0.000000  0.937000 -0.586000
H  0.000000 -0.937000 -0.586000
""",
        "charge": 0,
        "spin": 0,
    },
    "benzene": {
        "atom": """
C  0.000000  1.396792 0.000000
C -1.209657  0.698396 0.000000
C -1.209657 -0.698396 0.000000
C  0.000000 -1.396792 0.000000
C  1.209657 -0.698396 0.000000
C  1.209657  0.698396 0.000000
H  0.000000  2.484212 0.000000
H -2.151390  1.242106 0.000000
H -2.151390 -1.242106 0.000000
H  0.000000 -2.484212 0.000000
H  2.151390 -1.242106 0.000000
H  2.151390  1.242106 0.000000
""",
        "charge": 0,
        "spin": 0,
    },
}


PROFILE_FIELDS = [
    "timestamp_utc",
    "status",
    "error_type",
    "error_message",
    "molecule",
    "xc",
    "basis",
    "grid_level",
    "response_feature_kind",
    "jax_backend",
    "jax_devices",
    "cuda_visible_devices",
    "nao",
    "nmo",
    "nocc",
    "nvir",
    "ngrids",
    "scf_energy_ha",
    "scf_elapsed_s",
    "reference_build_elapsed_s",
    "feature_elapsed_s",
    "feature_peak_mib",
    "fxc_ad_first_elapsed_s",
    "fxc_ad_first_peak_mib",
    "fxc_ad_warm_repeats",
    "fxc_ad_warm_mean_s",
    "fxc_ad_warm_std_s",
    "fxc_ad_warm_min_s",
    "fxc_ad_warm_peak_mib",
    "grid_response_interface_warm_elapsed_s",
    "grid_response_interface_warm_peak_mib",
    "tensor_shape",
    "tensor_elements",
    "tensor_dtype",
    "warm_tensor_max_abs_diff",
    "gpu_mem_baseline_mib",
    "gpu_mem_after_reference_mib",
    "gpu_mem_after_fxc_mib",
    "notes",
]


@dataclass
class TimedBlock:
    elapsed_s: float
    peak_mib: int | None
    start_mib: int | None
    end_mib: int | None


class GpuMemorySampler:
    def __init__(self, *, interval_s: float = 0.05, pid: int | None = None) -> None:
        self.interval_s = float(interval_s)
        self.pid = int(pid or os.getpid())
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.start_mib: int | None = None
        self.end_mib: int | None = None
        self.peak_mib: int | None = None

    def __enter__(self) -> "GpuMemorySampler":
        self.start_mib = process_gpu_memory_mib(self.pid)
        if self.start_mib is not None:
            self.samples.append(self.start_mib)
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.2, 2.0 * self.interval_s))
        self.end_mib = process_gpu_memory_mib(self.pid)
        if self.end_mib is not None:
            self.samples.append(self.end_mib)
        self.peak_mib = max(self.samples) if self.samples else None

    def _poll(self) -> None:
        while not self._stop.is_set():
            value = process_gpu_memory_mib(self.pid)
            if value is not None:
                self.samples.append(value)
            self._stop.wait(self.interval_s)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_command(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return None


def process_gpu_memory_mib(pid: int | None = None) -> int | None:
    query = _run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if query is None:
        return None
    target = str(int(pid or os.getpid()))
    total = 0
    matched = False
    for line in query.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or parts[0] != target:
            continue
        try:
            total += int(float(parts[1]))
        except ValueError:
            continue
        matched = True
    return total if matched else 0


def _block_until_ready(value: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _time_call(fn, *, interval_s: float) -> tuple[Any, TimedBlock]:
    with GpuMemorySampler(interval_s=interval_s) as sampler:
        start = time.perf_counter()
        value = fn()
        _block_until_ready(value)
        elapsed = time.perf_counter() - start
    return value, TimedBlock(
        elapsed_s=elapsed,
        peak_mib=sampler.peak_mib,
        start_mib=sampler.start_mib,
        end_mib=sampler.end_mib,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROFILE_FIELDS, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in PROFILE_FIELDS})


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


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
        "jax_version": getattr(jax, "__version__", ""),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "pyscf_version": getattr(pyscf, "__version__", "") if pyscf is not None else "",
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", ""),
            "JAX_ENABLE_X64": os.environ.get("JAX_ENABLE_X64", ""),
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get(
                "XLA_PYTHON_CLIENT_PREALLOCATE", ""
            ),
        },
        "nvidia_smi": _run_command(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,memory.used",
                "--format=csv",
            ]
        ),
    }


def _build_mf(
    molecule: str,
    *,
    xc: str,
    basis: str,
    grid_level: int,
) -> tuple[Any, float]:
    spec = MOLECULES[molecule]
    mol = gto.M(
        atom=spec["atom"],
        unit="Angstrom",
        charge=int(spec["charge"]),
        spin=int(spec["spin"]),
        basis=basis,
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = int(grid_level)
    mf.conv_tol = 1e-10
    mf.max_cycle = 160
    start = time.perf_counter()
    mf.kernel()
    elapsed = time.perf_counter() - start
    if not bool(getattr(mf, "converged", False)):
        raise RuntimeError(f"PySCF SCF did not converge for {molecule}/{xc}/{basis}.")
    return mf, elapsed


def _profile_one(
    molecule: str,
    *,
    xc: str,
    basis: str,
    grid_level: int,
    warm_repeats: int,
    sampler_interval_s: float,
) -> dict[str, Any]:
    baseline_mib = process_gpu_memory_mib()
    mf, scf_elapsed = _build_mf(molecule, xc=xc, basis=basis, grid_level=grid_level)
    start = time.perf_counter()
    reference = restricted_reference_from_pyscf(mf)
    _block_until_ready(reference)
    reference_elapsed = time.perf_counter() - start
    after_reference_mib = process_gpu_memory_mib()

    functional = SemilocalResponseFunctional(xc)
    response_kind = functional.response_feature_kind

    (rho, grad_rho, tau, _), feature_timing = _time_call(
        lambda: restricted_grid_response_variables(reference, feature_kind=response_kind),
        interval_s=sampler_interval_s,
    )
    ngrids = int(np.asarray(rho).shape[0])

    def eval_tensor():
        return eval_xc_response_tensor(xc, rho, grad=grad_rho, tau=tau)

    (kind, tensor), first_timing = _time_call(eval_tensor, interval_s=sampler_interval_s)
    tensor_shape = tuple(int(dim) for dim in tensor.shape)
    tensor_elements = int(np.prod(tensor_shape, dtype=np.int64))
    tensor_dtype = str(tensor.dtype)

    warm_times: list[float] = []
    warm_peaks: list[int] = []
    warm_tensor = tensor
    for _ in range(int(warm_repeats)):
        (_, warm_tensor), warm_timing = _time_call(
            eval_tensor,
            interval_s=sampler_interval_s,
        )
        warm_times.append(float(warm_timing.elapsed_s))
        if warm_timing.peak_mib is not None:
            warm_peaks.append(int(warm_timing.peak_mib))

    max_abs_diff = float(jnp.max(jnp.abs(tensor - warm_tensor))) if warm_times else 0.0
    _block_until_ready(max_abs_diff)

    _GRID_RESPONSE_TENSOR_CACHE.clear()
    _, interface_timing = _time_call(
        lambda: functional.grid_response_tensor(reference),
        interval_s=sampler_interval_s,
    )
    after_fxc_mib = process_gpu_memory_mib()

    warm_array = np.asarray(warm_times, dtype=float)
    nao = int(mf.mol.nao_nr())
    nmo = int(np.asarray(mf.mo_energy).size)
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 1e-8))

    return {
        "timestamp_utc": _now(),
        "status": "ok",
        "molecule": molecule,
        "xc": xc,
        "basis": basis,
        "grid_level": int(grid_level),
        "response_feature_kind": str(kind),
        "jax_backend": jax.default_backend(),
        "jax_devices": "; ".join(str(device) for device in jax.devices()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nao": nao,
        "nmo": nmo,
        "nocc": nocc,
        "nvir": nmo - nocc,
        "ngrids": ngrids,
        "scf_energy_ha": float(mf.e_tot),
        "scf_elapsed_s": scf_elapsed,
        "reference_build_elapsed_s": reference_elapsed,
        "feature_elapsed_s": feature_timing.elapsed_s,
        "feature_peak_mib": feature_timing.peak_mib,
        "fxc_ad_first_elapsed_s": first_timing.elapsed_s,
        "fxc_ad_first_peak_mib": first_timing.peak_mib,
        "fxc_ad_warm_repeats": int(warm_repeats),
        "fxc_ad_warm_mean_s": float(warm_array.mean()) if warm_array.size else "",
        "fxc_ad_warm_std_s": float(warm_array.std(ddof=0)) if warm_array.size else "",
        "fxc_ad_warm_min_s": float(warm_array.min()) if warm_array.size else "",
        "fxc_ad_warm_peak_mib": max(warm_peaks) if warm_peaks else "",
        "grid_response_interface_warm_elapsed_s": interface_timing.elapsed_s,
        "grid_response_interface_warm_peak_mib": interface_timing.peak_mib,
        "tensor_shape": "x".join(str(dim) for dim in tensor_shape),
        "tensor_elements": tensor_elements,
        "tensor_dtype": tensor_dtype,
        "warm_tensor_max_abs_diff": max_abs_diff,
        "gpu_mem_baseline_mib": baseline_mib,
        "gpu_mem_after_reference_mib": after_reference_mib,
        "gpu_mem_after_fxc_mib": after_fxc_mib,
        "notes": "fxc_ad_first excludes feature construction but includes first JIT compilation for this grid shape; interface timing uses the TDDFT grid_response_tensor entry after AD compilation.",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile AD f_xc tensor timing and process GPU memory for TD-GradDFT.",
    )
    parser.add_argument(
        "--molecules",
        nargs="+",
        default=["water", "co", "n2", "ethylene", "formaldehyde", "benzene"],
        choices=sorted(MOLECULES),
    )
    parser.add_argument("--xc", default="b3lyp")
    parser.add_argument("--basis", default="def2-svp")
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--sampler-interval-s", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path("benchmark")
            / "pyscf_correctness"
            / "runs"
            / f"fxc_ad_profile_{args.xc}_{args.basis}_grid{args.grid_level}_{timestamp}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_csv = output_dir / "fxc_ad_profile.csv"
    progress_jsonl = output_dir / "progress.jsonl"
    env = _environment()
    environment_json = output_dir / "environment.json"
    if not environment_json.exists():
        environment_json.write_text(json.dumps(env, indent=2, sort_keys=True) + "\n")
    _write_jsonl(output_dir / "environment.jsonl", env)

    for molecule in args.molecules:
        progress = {
            "timestamp_utc": _now(),
            "event": "start",
            "molecule": molecule,
            "xc": args.xc,
            "basis": args.basis,
            "grid_level": args.grid_level,
        }
        _write_jsonl(progress_jsonl, progress)
        try:
            row = _profile_one(
                molecule,
                xc=args.xc,
                basis=args.basis,
                grid_level=args.grid_level,
                warm_repeats=args.warm_repeats,
                sampler_interval_s=args.sampler_interval_s,
            )
        except Exception as exc:
            row = {
                "timestamp_utc": _now(),
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "molecule": molecule,
                "xc": args.xc,
                "basis": args.basis,
                "grid_level": args.grid_level,
                "jax_backend": jax.default_backend(),
                "jax_devices": "; ".join(str(device) for device in jax.devices()),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "notes": traceback.format_exc(limit=8),
            }
        _write_csv(profile_csv, [row], append=True)
        _write_jsonl(
            progress_jsonl,
            {
                "timestamp_utc": _now(),
                "event": "finish",
                "molecule": molecule,
                "status": row.get("status"),
                "fxc_ad_warm_mean_s": row.get("fxc_ad_warm_mean_s"),
            },
        )
        print(json.dumps(row, sort_keys=True), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
