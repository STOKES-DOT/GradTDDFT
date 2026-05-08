from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import numpy as np
from pyscf import gto

from td_graddft.data import basis_from_spec
from td_graddft.data.integrals import build_hcore, eri_tensor, overlap_matrix, precompile_eri_kernels
from td_graddft.jax_runtime import (
    DEFAULT_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS,
    DEFAULT_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES,
    configure_jax_persistent_cache,
)


@dataclass(frozen=True)
class IntegralTiming:
    system: str
    basis: str
    component: str
    pyscf_s: float
    jax_first_s: float
    jax_warm_s: float
    speedup_warm_vs_pyscf: float
    max_abs_diff: float


_SYSTEMS: dict[str, list[tuple[str, tuple[float, float, float]]]] = {
    "water": [
        ("O", (0.0, 0.0, 0.117790)),
        ("H", (0.0, 0.755453, -0.471161)),
        ("H", (0.0, -0.755453, -0.471161)),
    ],
    "h2o2": [
        ("O", (0.0, 0.0, -0.731600)),
        ("O", (0.0, 0.0, 0.731600)),
        ("H", (0.946218, 0.0, -0.930999)),
        ("H", (-0.346790, 0.880378, 0.930999)),
    ],
    "ethylene": [
        ("C", (-0.6695, 0.0, 0.0)),
        ("C", (0.6695, 0.0, 0.0)),
        ("H", (-1.2321, 0.9239, 0.0)),
        ("H", (-1.2321, -0.9239, 0.0)),
        ("H", (1.2321, 0.9239, 0.0)),
        ("H", (1.2321, -0.9239, 0.0)),
    ],
    "benzene": [
        ("C", (0.0, 1.396792, 0.0)),
        ("C", (1.209657, 0.698396, 0.0)),
        ("C", (1.209657, -0.698396, 0.0)),
        ("C", (0.0, -1.396792, 0.0)),
        ("C", (-1.209657, -0.698396, 0.0)),
        ("C", (-1.209657, 0.698396, 0.0)),
        ("H", (0.0, 2.484212, 0.0)),
        ("H", (2.151390, 1.242106, 0.0)),
        ("H", (2.151390, -1.242106, 0.0)),
        ("H", (0.0, -2.484212, 0.0)),
        ("H", (-2.151390, -1.242106, 0.0)),
        ("H", (-2.151390, 1.242106, 0.0)),
    ],
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--system", choices=sorted(_SYSTEMS), default="water")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--outdir", default="outputs/integral_benchmark")
    p.add_argument("--max-l", type=int, default=3)
    p.add_argument("--precompile-eri", action="store_true")
    p.add_argument(
        "--jax-cache-dir",
        default=".jax_cache/integral_benchmark",
        help="persistent JAX compilation cache directory (empty string disables)",
    )
    p.add_argument(
        "--jax-cache-min-compile-secs",
        type=float,
        default=DEFAULT_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS,
        help="minimum compile time (s) for entries persisted to cache",
    )
    return p.parse_args()


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    if hasattr(out, "block_until_ready"):
        out.block_until_ready()
    dt = time.perf_counter() - t0
    return out, float(dt)


def main() -> None:
    args = _parse_args()
    jax.config.update("jax_enable_x64", True)
    configure_jax_persistent_cache(
        cache_dir=str(args.jax_cache_dir),
        min_compile_time_secs=float(args.jax_cache_min_compile_secs),
        min_entry_size_bytes=DEFAULT_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES,
    )
    atom = _SYSTEMS[str(args.system)]
    mol = gto.M(
        atom=atom,
        basis=str(args.basis),
        unit="Angstrom",
        charge=0,
        spin=0,
        cart=True,
        verbose=0,
    )
    basis = basis_from_spec(
        atom,
        basis=str(args.basis),
        unit="Angstrom",
        charge=0,
        spin=0,
        max_l=int(args.max_l),
    )

    device = jax.devices("cpu")[0]
    with jax.default_device(device):
        rows: list[IntegralTiming] = []
        cases = [
            ("overlap", lambda: mol.intor_symmetric("int1e_ovlp"), lambda: overlap_matrix(basis, engine="jit")),
            (
                "hcore",
                lambda: mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc"),
                lambda: build_hcore(basis, engine="jit"),
            ),
            ("eri", lambda: mol.intor("int2e"), lambda: eri_tensor(basis, engine="jit")),
        ]
        for component, pyscf_fn, jax_fn in cases:
            ref, pyscf_s = _timed(pyscf_fn)
            jax.clear_caches()
            if component == "eri" and bool(args.precompile_eri):
                t0 = time.perf_counter()
                compile_info = precompile_eri_kernels(basis, engine="jit")
                jax_first_s = time.perf_counter() - t0
                got_first, exec_after_compile_s = _timed(jax_fn)
                got_warm, jax_warm_s = _timed(jax_fn)
                rows.append(
                    IntegralTiming(
                        system=str(args.system),
                        basis=str(args.basis),
                        component=f"{component}_precompiled",
                        pyscf_s=float(pyscf_s),
                        jax_first_s=float(jax_first_s + exec_after_compile_s),
                        jax_warm_s=float(jax_warm_s),
                        speedup_warm_vs_pyscf=float(pyscf_s / jax_warm_s),
                        max_abs_diff=float(np.max(np.abs(np.asarray(got_warm) - np.asarray(ref)))),
                    )
                )
                print(f"precompile_info={compile_info}")
                continue
            got_first, jax_first_s = _timed(jax_fn)
            got_warm, jax_warm_s = _timed(jax_fn)
            rows.append(
                IntegralTiming(
                    system=str(args.system),
                    basis=str(args.basis),
                    component=component,
                    pyscf_s=float(pyscf_s),
                    jax_first_s=float(jax_first_s),
                    jax_warm_s=float(jax_warm_s),
                    speedup_warm_vs_pyscf=float(pyscf_s / jax_warm_s),
                    max_abs_diff=float(np.max(np.abs(np.asarray(got_warm) - np.asarray(ref)))),
                )
            )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.system}_{args.basis}".replace("/", "_")
    json_path = outdir / f"{stem}_integrals_summary.json"
    with json_path.open("w") as f:
        json.dump([asdict(row) for row in rows], f, indent=2)
    print(json.dumps([asdict(row) for row in rows], indent=2))
    print(json_path)


if __name__ == "__main__":
    main()
