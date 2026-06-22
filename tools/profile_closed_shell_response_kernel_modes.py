from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _REPO_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from closed_shell_s1_benchmark_common import closed_shell_s1_spec_map
from td_graddft.data.reference import restricted_reference_from_pyscf
from td_graddft.tddft.response import build_restricted_tda_operator


@dataclass(frozen=True)
class _HybridOnlyResponseFunctional:
    exact_exchange_fraction: float
    response_feature_kind: str = "LDA"

    def local_kernel(self, density):
        return jnp.zeros_like(density)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile restricted closed-shell TDA response two-electron backends."
    )
    parser.add_argument("--molecule", default="benzene", choices=sorted(closed_shell_s1_spec_map()))
    parser.add_argument("--basis", default="6-31g")
    parser.add_argument("--xc", default="b3lyp")
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument("--modes", nargs="+", default=["direct", "df", "ris"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ris-theta", type=float, default=0.2)
    parser.add_argument("--ris-j-fit", choices=["s", "sp", "spd"], default="sp")
    parser.add_argument("--ris-k-fit", choices=["s", "sp", "spd"], default="s")
    parser.add_argument("--outdir", type=Path, default=Path("benchmark/ris_response_memory"))
    parser.add_argument("--csv-name", default="response_kernel_memory.csv")
    parser.add_argument("--no-jit", action="store_true")
    return parser.parse_args()


def _gpu_memory_mb() -> float | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [item.strip() for item in visible.split(",") if item.strip()]
        if len(devices) == 1 and devices[0].startswith("GPU-"):
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,uuid,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except (FileNotFoundError, subprocess.CalledProcessError):
                return None
            for line in result.stdout.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= 3 and parts[1] == devices[0]:
                    return float(parts[2])
            return None
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    if visible:
        indices = [item.strip() for item in visible.split(",") if item.strip()]
        if len(indices) == 1 and (indices[0].isdigit() or indices[0].startswith("GPU-")):
            command.insert(1, f"--id={indices[0]}")
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if not values:
        return None
    return max(values)


class _GpuMemoryMonitor:
    def __init__(self, interval_seconds: float = 0.05):
        self.interval_seconds = float(interval_seconds)
        self.peak_mb: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            value = _gpu_memory_mb()
            if value is not None:
                self.peak_mb = value if self.peak_mb is None else max(self.peak_mb, value)
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "_GpuMemoryMonitor":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


def _rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / 1024.0**2
    except ModuleNotFoundError:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return rss / 1024.0 if rss > 10_000_000 else rss


def _array_nbytes(value: Any) -> int:
    if value is None:
        return 0
    arr = jnp.asarray(value)
    return int(arr.size * arr.dtype.itemsize)


def _stored_two_electron_bytes(molecule: Any, mode: str) -> int:
    if mode == "direct":
        return _array_nbytes(getattr(molecule, "eri_pair_matrix", None)) + _array_nbytes(
            getattr(molecule, "rep_tensor", None)
        )
    if mode == "df":
        return _array_nbytes(getattr(molecule, "df_factors", None)) + _array_nbytes(
            getattr(molecule, "response_df_factors_j", None)
        )
    if mode == "ris":
        j_bytes = _array_nbytes(getattr(molecule, "response_df_factors_j", None))
        k_bytes = _array_nbytes(getattr(molecule, "response_df_factors_k", None))
        return j_bytes + k_bytes
    raise ValueError(f"Unsupported mode {mode!r}.")


def _build_mean_field(args: argparse.Namespace):
    from pyscf import dft, gto

    spec = closed_shell_s1_spec_map()[args.molecule]
    mol = gto.M(
        atom=spec.atom,
        basis=args.basis,
        charge=spec.charge,
        spin=spec.spin,
        unit=spec.unit,
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = args.xc
    mf.grids.level = int(args.grid_level)
    mf.conv_tol = 1e-10
    mf.kernel()
    if not bool(getattr(mf, "converged", False)):
        raise RuntimeError("PySCF ground-state reference did not converge.")
    return mf


def _reference_for_mode(mf: Any, mode: str, args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    if mode == "direct":
        molecule = restricted_reference_from_pyscf(
            mf,
            jk_backend="full",
            response_df_mode="none",
            array_backend="jax",
        )
        return molecule, {"two_electron_mode": "direct"}
    if mode == "df":
        molecule = restricted_reference_from_pyscf(
            mf,
            jk_backend="df",
            response_df_mode="none",
            array_backend="jax",
        )
        return molecule, {"two_electron_mode": "df"}
    if mode == "ris":
        molecule = restricted_reference_from_pyscf(
            mf,
            jk_backend="full",
            response_df_mode="ris",
            response_ris_theta=float(args.ris_theta),
            response_ris_j_fit=str(args.ris_j_fit),
            response_ris_k_fit=str(args.ris_k_fit),
            array_backend="jax",
        )
        molecule = replace(molecule, eri_pair_matrix=None, df_factors=None)
        return molecule, {
            "two_electron_mode": "ris",
            "ris_theta": float(args.ris_theta),
            "ris_j_fit": str(args.ris_j_fit),
            "ris_k_fit": str(args.ris_k_fit),
        }
    raise ValueError(f"Unsupported mode {mode!r}.")


def _profile_mode(mf: Any, mode: str, args: argparse.Namespace) -> dict[str, Any]:
    gc.collect()
    gpu_before = _gpu_memory_mb()
    rss_before = _rss_mb()
    with _GpuMemoryMonitor() as monitor:
        start_build = time.perf_counter()
        molecule, options = _reference_for_mode(mf, mode, args)
        build_seconds = time.perf_counter() - start_build
        gpu_after_reference = _gpu_memory_mb()
        rss_after_reference = _rss_mb()

        xc = _HybridOnlyResponseFunctional(
            exact_exchange_fraction=float(getattr(molecule, "exact_exchange_fraction", 0.0))
        )
        start_operator = time.perf_counter()
        vind, diagonal, delta_eps = build_restricted_tda_operator(
            molecule,
            xc,
            response_kernel_options=options,
        )
        operator_seconds = time.perf_counter() - start_operator
        gpu_after_operator = _gpu_memory_mb()
        rss_after_operator = _rss_mb()

        dim = int(jnp.asarray(diagonal).size)
        x = jnp.ones((int(args.batch_size), dim), dtype=jnp.asarray(delta_eps).dtype)
        apply_fn = vind if args.no_jit else jax.jit(vind)
        start_apply = time.perf_counter()
        y = apply_fn(x)
        jax.block_until_ready(y)
        apply_seconds = time.perf_counter() - start_apply
        gpu_after_apply = _gpu_memory_mb()
        rss_after_apply = _rss_mb()
        gpu_peak_monitor = monitor.peak_mb

    gpu_values = [
        value
        for value in (gpu_before, gpu_after_reference, gpu_after_operator, gpu_after_apply)
        if value is not None
    ]
    return {
        "mode": mode,
        "molecule": args.molecule,
        "basis": args.basis,
        "xc": args.xc,
        "grid_level": int(args.grid_level),
        "jax_backend": jax.default_backend(),
        "nocc": int(getattr(molecule, "nocc", 0) or 0),
        "nmo": int(jnp.asarray(molecule.mo_coeff).shape[-1]),
        "transition_dim": dim,
        "batch_size": int(args.batch_size),
        "stored_two_electron_mb": _stored_two_electron_bytes(molecule, mode) / 1024.0**2,
        "gpu_before_mb": gpu_before,
        "gpu_after_reference_mb": gpu_after_reference,
        "gpu_after_operator_mb": gpu_after_operator,
        "gpu_after_apply_mb": gpu_after_apply,
        "gpu_max_snapshot_mb": max(gpu_values) if gpu_values else None,
        "gpu_peak_monitor_mb": gpu_peak_monitor,
        "rss_before_mb": rss_before,
        "rss_after_reference_mb": rss_after_reference,
        "rss_after_operator_mb": rss_after_operator,
        "rss_after_apply_mb": rss_after_apply,
        "build_reference_seconds": build_seconds,
        "build_operator_seconds": operator_seconds,
        "apply_seconds": apply_seconds,
        "response_options": json.dumps(options, sort_keys=True),
    }


def main() -> None:
    args = _parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    mf = _build_mean_field(args)
    rows = [_profile_mode(mf, str(mode).lower(), args) for mode in args.modes]
    csv_path = args.outdir / args.csv_name
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"csv": str(csv_path), "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
