from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto
from pyscf.dft import gen_grid, numint

from td_graddft.data.basis import basis_from_pyscf_mol_cart
from td_graddft.data.integrals import build_hcore, eri_tensor, overlap_matrix
from td_graddft.scf import RKSConfig, run_rks_from_integrals


def _polyene_atoms(n_carbon: int) -> list[tuple[str, tuple[float, float, float]]]:
    if n_carbon < 2:
        raise ValueError("n_carbon must be >= 2")

    c_c_double = 1.34
    c_c_single = 1.46
    c_h = 1.09
    z_wing = 0.90

    carbons: list[tuple[float, float, float]] = []
    x = 0.0
    carbons.append((x, 0.0, 0.0))
    for i in range(1, n_carbon):
        bond = c_c_double if (i - 1) % 2 == 0 else c_c_single
        x += bond
        carbons.append((x, 0.0, 0.0))

    atoms: list[tuple[str, tuple[float, float, float]]] = [("C", c) for c in carbons]
    x0 = carbons[0][0]
    xn = carbons[-1][0]
    atoms.extend(
        [
            ("H", (x0, +c_h, +z_wing)),
            ("H", (x0, +c_h, -z_wing)),
            ("H", (xn, -c_h, +z_wing)),
            ("H", (xn, -c_h, -z_wing)),
        ]
    )
    for i in range(1, n_carbon - 1):
        x_i = carbons[i][0]
        y_i = c_h if (i % 2 == 0) else -c_h
        atoms.append(("H", (x_i, y_i, 0.0)))
    return atoms


@dataclass(frozen=True)
class BenchmarkRow:
    device: str
    run: int
    time_s: float
    converged: bool
    cycles: int
    e_tot_ha: float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-carbon", type=int, default=4)
    p.add_argument("--basis", type=str, default="sto-3g")
    p.add_argument("--xc", type=str, default="pbe0")
    p.add_argument("--grids-level", type=int, default=0)
    p.add_argument("--max-l", type=int, default=3)
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--outdir", type=str, default="outputs/backend_compare")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    mol = gto.M(
        atom=_polyene_atoms(args.n_carbon),
        basis=args.basis,
        unit="Angstrom",
        charge=0,
        spin=0,
        cart=True,
        verbose=0,
    )

    grids = gen_grid.Grids(mol)
    grids.level = int(args.grids_level)
    grids.build()
    coords = np.asarray(grids.coords)
    weights = np.asarray(grids.weights)
    ao = np.asarray(numint.eval_ao(mol, coords, deriv=0))
    ao_deriv1 = np.asarray(numint.eval_ao(mol, coords, deriv=1))

    basis_cart = basis_from_pyscf_mol_cart(mol, max_l=int(args.max_l))
    s_cpu = np.asarray(overlap_matrix(basis_cart), dtype=np.float64)
    h1e_cpu = np.asarray(build_hcore(basis_cart), dtype=np.float64)
    eri_cpu = np.asarray(eri_tensor(basis_cart), dtype=np.float64)

    print(f"JAX devices: {jax.devices()}")
    print(
        f"system: C{args.n_carbon}, basis={args.basis}, xc={args.xc}, "
        f"nao={mol.nao_nr()}, nelec={mol.nelectron}"
    )

    def run_case(device_kind: str) -> BenchmarkRow:
        dev = jax.devices(device_kind)[0]
        s = jax.device_put(jnp.asarray(s_cpu), dev)
        h1e = jax.device_put(jnp.asarray(h1e_cpu), dev)
        eri = jax.device_put(jnp.asarray(eri_cpu), dev)
        ao_dev = jax.device_put(jnp.asarray(ao), dev)
        ao_deriv1_dev = jax.device_put(jnp.asarray(ao_deriv1), dev)
        weights_dev = jax.device_put(jnp.asarray(weights), dev)

        cfg = RKSConfig(
            xc_spec=str(args.xc),
            max_cycle=80,
            conv_tol=1e-9,
            conv_tol_density=1e-7,
            damping=0.10,
            potential_clip=20.0,
        )

        t0 = time.perf_counter()
        out = run_rks_from_integrals(
            overlap=s,
            hcore=h1e,
            eri=eri,
            nelectron=int(mol.nelectron),
            nuclear_repulsion=float(mol.energy_nuc()),
            ao=ao_dev,
            ao_deriv1=ao_deriv1_dev,
            grid_weights=weights_dev,
            config=cfg,
        )
        _ = np.asarray(out.mo_energy)
        dt = time.perf_counter() - t0
        return BenchmarkRow(
            device=device_kind,
            run=-1,
            time_s=float(dt),
            converged=bool(out.converged),
            cycles=int(out.cycles),
            e_tot_ha=float(out.total_energy),
        )

    rows: list[BenchmarkRow] = []
    for device in ("cpu", "gpu"):
        for run_idx in range(1, int(args.runs) + 1):
            row = run_case(device)
            row = BenchmarkRow(
                device=row.device,
                run=run_idx,
                time_s=row.time_s,
                converged=row.converged,
                cycles=row.cycles,
                e_tot_ha=row.e_tot_ha,
            )
            rows.append(row)
            print(asdict(row))

    def pick(device: str, run: int) -> BenchmarkRow:
        for r in rows:
            if r.device == device and r.run == run:
                return r
        raise KeyError((device, run))

    steady_run = int(args.runs)
    cpu_lax = pick("cpu", steady_run)
    gpu_lax = pick("gpu", steady_run)

    summary = {
        "steady_run": steady_run,
        "gpu_speedup_over_cpu_lax": cpu_lax.time_s / gpu_lax.time_s,
        "abs_e_diff_cpu_vs_gpu_lax": abs(cpu_lax.e_tot_ha - gpu_lax.e_tot_ha),
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "results.json"
    csv_path = outdir / "results.csv"
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
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["device", "run", "time_s", "converged", "cycles", "e_tot_ha"])
        for r in rows:
            w.writerow(
                [
                    r.device,
                    r.run,
                    f"{r.time_s:.6f}",
                    int(r.converged),
                    r.cycles,
                    f"{r.e_tot_ha:.10f}",
                ]
            )

    print("summary:", summary)
    print(f"json: {json_path}")
    print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()
