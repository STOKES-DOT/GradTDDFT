from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from pyscf import dft, gto

from td_graddft import tdscf
from td_graddft.reference_legacy import restricted_reference_from_pyscf


def _linear_acene_atoms(n_rings: int, *, c_c: float = 1.397, c_h: float = 1.09):
    if n_rings < 1:
        raise ValueError("n_rings must be >= 1")

    centers = [(math.sqrt(3.0) * c_c * ring, 0.0) for ring in range(n_rings)]
    vertex_offsets = [
        (0.0, c_c),
        (math.sqrt(3.0) * 0.5 * c_c, 0.5 * c_c),
        (math.sqrt(3.0) * 0.5 * c_c, -0.5 * c_c),
        (0.0, -c_c),
        (-math.sqrt(3.0) * 0.5 * c_c, -0.5 * c_c),
        (-math.sqrt(3.0) * 0.5 * c_c, 0.5 * c_c),
    ]

    vertices: list[np.ndarray] = []
    neighbors: list[set[int]] = []
    index_by_key: dict[tuple[int, int], int] = {}

    def get_index(point: np.ndarray) -> int:
        key = (int(round(point[0] * 1_000_000)), int(round(point[1] * 1_000_000)))
        idx = index_by_key.get(key)
        if idx is not None:
            return idx
        idx = len(vertices)
        index_by_key[key] = idx
        vertices.append(point)
        neighbors.append(set())
        return idx

    for cx, cy in centers:
        ring = []
        for dx, dy in vertex_offsets:
            ring.append(get_index(np.asarray([cx + dx, cy + dy], dtype=float)))
        for i in range(6):
            a = ring[i]
            b = ring[(i + 1) % 6]
            neighbors[a].add(b)
            neighbors[b].add(a)

    atoms: list[tuple[str, tuple[float, float, float]]] = []
    for point in vertices:
        atoms.append(("C", (float(point[0]), float(point[1]), 0.0)))

    for idx, point in enumerate(vertices):
        if len(neighbors[idx]) != 2:
            continue
        bonded = np.asarray([vertices[j] for j in neighbors[idx]], dtype=float)
        inward = (bonded[0] - point) + (bonded[1] - point)
        norm = float(np.linalg.norm(inward))
        if norm <= 1e-12:
            continue
        outward = -inward / norm
        h = point + c_h * outward
        atoms.append(("H", (float(h[0]), float(h[1]), 0.0)))
    return atoms


MOLECULES = {
    "benzene": """
C        0.0000000000      1.3967920000      0.0000000000
C       -1.2096570000      0.6983960000      0.0000000000
C       -1.2096570000     -0.6983960000      0.0000000000
C        0.0000000000     -1.3967920000      0.0000000000
C        1.2096570000     -0.6983960000      0.0000000000
C        1.2096570000      0.6983960000      0.0000000000
H        0.0000000000      2.4842120000      0.0000000000
H       -2.1513900000      1.2421060000      0.0000000000
H       -2.1513900000     -1.2421060000      0.0000000000
H        0.0000000000     -2.4842120000      0.0000000000
H        2.1513900000     -1.2421060000      0.0000000000
H        2.1513900000      1.2421060000      0.0000000000
""",
    "water": """
O  0.000000  0.000000  0.117790
H  0.000000  0.755453 -0.471161
H  0.000000 -0.755453 -0.471161
""",
    "anthracene": _linear_acene_atoms(3),
}


@dataclass(frozen=True)
class BenchmarkRow:
    stage: str
    mode: str
    run: int
    elapsed_s: float
    nstates: int
    first_excitation_ha: float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--molecule", choices=tuple(MOLECULES.keys()), default="benzene")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--xc", default="pbe0")
    p.add_argument("--nstates", type=int, default=5)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--outdir", default="outputs/restricted_tddft_solver_bench")
    return p.parse_args()


def _build_reference(molecule: str, basis: str, xc: str):
    mol = gto.M(
        atom=MOLECULES[molecule],
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
        raise RuntimeError("PySCF SCF did not converge.")
    return restricted_reference_from_pyscf(mf)


def _benchmark_once(
    reference,
    *,
    nstates: int,
    mode: str,
) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    tda_solver = tdscf.TDA(
        reference,
        eigensolver=mode,
        davidson_tol=1e-5,
        davidson_max_iter=160,
        davidson_max_subspace=64,
    )
    solver = tdscf.TDDFT(
        reference,
        eigensolver=mode,
        davidson_tol=1e-5,
        davidson_max_iter=160,
        davidson_max_subspace=64,
    )

    t0 = time.perf_counter()
    tda = tda_solver.kernel(nstates=nstates)
    rows.append(
        BenchmarkRow(
            stage="tda",
            mode=mode,
            run=-1,
            elapsed_s=float(time.perf_counter() - t0),
            nstates=nstates,
            first_excitation_ha=float(np.asarray(tda.excitation_energies)[0]),
        )
    )

    t1 = time.perf_counter()
    kernel = solver.kernel(nstates=nstates)
    rows.append(
        BenchmarkRow(
            stage="kernel",
            mode=mode,
            run=-1,
            elapsed_s=float(time.perf_counter() - t1),
            nstates=nstates,
            first_excitation_ha=float(np.asarray(kernel.excitation_energies)[0]),
        )
    )
    return rows


def main() -> None:
    args = _parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    reference = _build_reference(args.molecule, args.basis, args.xc)

    rows: list[BenchmarkRow] = []
    for mode in ("dense", "davidson"):
        for run in range(1, int(args.runs) + 1):
            measured = _benchmark_once(reference, nstates=int(args.nstates), mode=mode)
            rows.extend(
                BenchmarkRow(
                    stage=row.stage,
                    mode=row.mode,
                    run=run,
                    elapsed_s=row.elapsed_s,
                    nstates=row.nstates,
                    first_excitation_ha=row.first_excitation_ha,
                )
                for row in measured
            )
            print(asdict(rows[-2]))
            print(asdict(rows[-1]))

    def _steady(stage: str, mode: str) -> BenchmarkRow:
        candidates = [r for r in rows if r.stage == stage and r.mode == mode]
        return candidates[-1]

    summary = {
        "molecule": args.molecule,
        "basis": args.basis,
        "xc": args.xc,
        "nstates": int(args.nstates),
        "runs": int(args.runs),
        "tda_speedup_davidson_over_dense": _steady("tda", "dense").elapsed_s
        / _steady("tda", "davidson").elapsed_s,
        "kernel_speedup_davidson_over_dense": _steady("kernel", "dense").elapsed_s
        / _steady("kernel", "davidson").elapsed_s,
        "tda_first_excitation_abs_diff_ha": abs(
            _steady("tda", "dense").first_excitation_ha
            - _steady("tda", "davidson").first_excitation_ha
        ),
        "kernel_first_excitation_abs_diff_ha": abs(
            _steady("kernel", "dense").first_excitation_ha
            - _steady("kernel", "davidson").first_excitation_ha
        ),
    }

    csv_path = outdir / f"{args.molecule}_{args.xc}_{args.basis}_solver_modes.csv"
    json_path = outdir / f"{args.molecule}_{args.xc}_{args.basis}_solver_modes.json"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "mode", "run", "elapsed_s", "nstates", "first_excitation_ha"])
        for row in rows:
            w.writerow(
                [
                    row.stage,
                    row.mode,
                    row.run,
                    f"{row.elapsed_s:.6f}",
                    row.nstates,
                    f"{row.first_excitation_ha:.10f}",
                ]
            )
    with json_path.open("w") as f:
        json.dump(
            {
                "config": vars(args),
                "rows": [asdict(r) for r in rows],
                "summary": summary,
            },
            f,
            indent=2,
        )

    print("summary:", summary)
    print(f"csv={csv_path}")
    print(f"json={json_path}")


if __name__ == "__main__":
    main()
