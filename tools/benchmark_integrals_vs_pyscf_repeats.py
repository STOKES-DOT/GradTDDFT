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
from td_graddft.data.integrals import build_hcore, eri_tensor, overlap_matrix


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


@dataclass(frozen=True)
class ComponentSummary:
    system: str
    basis: str
    component: str
    rounds: int
    max_abs_diff: float
    pyscf_mean_s: float
    pyscf_std_s: float
    jax_first_s: float
    jax_warm_mean_s: float
    jax_warm_std_s: float
    warm_speedup_vs_pyscf: float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--systems", default="water,ethylene")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--outdir", default="outputs/integral_benchmark_repeats")
    p.add_argument("--max-l", type=int, default=3)
    return p.parse_args()


def _time_call(fn):
    t0 = time.perf_counter()
    out = fn()
    if hasattr(out, "block_until_ready"):
        out.block_until_ready()
    return out, float(time.perf_counter() - t0)


def _measure_rounds(fn, rounds: int) -> list[float]:
    values: list[float] = []
    for _ in range(int(rounds)):
        _, elapsed = _time_call(fn)
        values.append(float(elapsed))
    return values


def _system_summary(system: str, basis_name: str, rounds: int, max_l: int) -> list[ComponentSummary]:
    atom = _SYSTEMS[system]
    mol = gto.M(
        atom=atom,
        basis=basis_name,
        unit="Angstrom",
        charge=0,
        spin=0,
        cart=True,
        verbose=0,
    )
    basis = basis_from_spec(
        atom,
        basis=basis_name,
        unit="Angstrom",
        charge=0,
        spin=0,
        max_l=max_l,
    )

    cases = [
        ("overlap", lambda: mol.intor_symmetric("int1e_ovlp"), lambda: overlap_matrix(basis, engine="jit")),
        (
            "hcore",
            lambda: mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc"),
            lambda: build_hcore(basis, engine="jit"),
        ),
        ("eri", lambda: mol.intor("int2e"), lambda: eri_tensor(basis, engine="jit")),
    ]

    rows: list[ComponentSummary] = []
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        for component, pyscf_fn, jax_fn in cases:
            ref, _ = _time_call(pyscf_fn)
            pyscf_times = _measure_rounds(pyscf_fn, rounds)

            jax.clear_caches()
            got_first, first_time = _time_call(jax_fn)
            warm_times = _measure_rounds(jax_fn, rounds)
            max_abs_diff = float(np.max(np.abs(np.asarray(got_first) - np.asarray(ref))))

            rows.append(
                ComponentSummary(
                    system=system,
                    basis=basis_name,
                    component=component,
                    rounds=int(rounds),
                    max_abs_diff=max_abs_diff,
                    pyscf_mean_s=float(np.mean(pyscf_times)),
                    pyscf_std_s=float(np.std(pyscf_times)),
                    jax_first_s=float(first_time),
                    jax_warm_mean_s=float(np.mean(warm_times)),
                    jax_warm_std_s=float(np.std(warm_times)),
                    warm_speedup_vs_pyscf=float(np.mean(pyscf_times) / np.mean(warm_times)),
                )
            )
    return rows


def main() -> None:
    args = _parse_args()
    jax.config.update("jax_enable_x64", True)
    systems = [part.strip() for part in str(args.systems).split(",") if part.strip()]
    invalid = [name for name in systems if name not in _SYSTEMS]
    if invalid:
        raise ValueError(f"Unsupported systems: {invalid!r}")

    rows: list[ComponentSummary] = []
    for system in systems:
        rows.extend(
            _system_summary(
                system=system,
                basis_name=str(args.basis),
                rounds=int(args.rounds),
                max_l=int(args.max_l),
            )
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{'_'.join(systems)}_{args.basis}_r{int(args.rounds)}".replace("/", "_")
    json_path = outdir / f"{stem}_summary.json"
    with json_path.open("w") as f:
        json.dump([asdict(row) for row in rows], f, indent=2)
    print(json.dumps([asdict(row) for row in rows], indent=2))
    print(json_path)


if __name__ == "__main__":
    main()
