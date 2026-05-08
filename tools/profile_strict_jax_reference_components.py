from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto

from td_graddft.data.basis import basis_from_pyscf_mol_cart
from td_graddft.data.grid import build_molecular_grid
from td_graddft.data.grid_ao import evaluate_cartesian_ao
from td_graddft.data.integrals import build_hcore, overlap_matrix
from td_graddft.df import true_df_factors_from_pyscf_mol
from td_graddft.scf import RKSConfig, run_rks_from_integrals


def _polyene_atoms(n_carbon: int) -> list[tuple[str, tuple[float, float, float]]]:
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


def _water_atoms() -> str:
    return """
O  0.000000  0.000000  0.117790
H  0.000000  0.755453 -0.471161
H  0.000000 -0.755453 -0.471161
"""


def _h2o2_atoms() -> str:
    return """
O   0.000000   0.000000  -0.731600
O   0.000000   0.000000   0.731600
H   0.946218   0.000000  -0.930999
H  -0.346790   0.880378   0.930999
"""


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile strict-JAX reference build components.")
    p.add_argument("--system", choices=("water", "h2o2", "polyene"), default="water")
    p.add_argument("--carbons", type=int, default=6)
    p.add_argument("--basis", default="6-31g")
    p.add_argument("--xc", default="pbe0")
    p.add_argument("--grids-level", type=int, default=0)
    p.add_argument("--max-l", type=int, default=3)
    p.add_argument("--max-cycle", type=int, default=32)
    p.add_argument("--conv-tol-density", type=float, default=1e-7)
    p.add_argument("--damping", type=float, default=0.05)
    p.add_argument("--outdir", default="outputs/reference_component_profiles")
    return p.parse_args()


def _time_call(fn, *, block=None):
    t0 = time.perf_counter()
    out = fn()
    if block is not None:
        block(out)
    return out, float(time.perf_counter() - t0)


def main() -> None:
    args = _parse_args()
    jax.config.update("jax_enable_x64", True)

    if args.system == "water":
        atom = _water_atoms()
        label = "water"
    elif args.system == "h2o2":
        atom = _h2o2_atoms()
        label = "h2o2"
    else:
        atom = _polyene_atoms(int(args.carbons))
        label = f"polyene_c{int(args.carbons)}"

    mol = gto.M(
        atom=atom,
        basis=str(args.basis),
        unit="Angstrom",
        spin=0,
        charge=0,
        cart=True,
        verbose=0,
    )

    _, _ = _time_call(
        lambda: basis_from_pyscf_mol_cart(
            mol,
            max_l=int(args.max_l),
            precompute_eri_groups=False,
        )
    )
    basis, basis_s = _time_call(
        lambda: basis_from_pyscf_mol_cart(
            mol,
            max_l=int(args.max_l),
            precompute_eri_groups=False,
        )
    )

    _, _ = _time_call(
        lambda: build_molecular_grid(
            atom,
            unit="Angstrom",
            charge=0,
            spin=0,
            level=int(args.grids_level),
        )
    )
    (coords, weights, spec), grid_s = _time_call(
        lambda: build_molecular_grid(
            atom,
            unit="Angstrom",
            charge=0,
            spin=0,
            level=int(args.grids_level),
        )
    )

    coords = jnp.asarray(coords)
    _, _ = _time_call(lambda: evaluate_cartesian_ao(basis, coords, deriv=1), block=jax.block_until_ready)
    ao_deriv1, ao_s = _time_call(
        lambda: evaluate_cartesian_ao(basis, coords, deriv=1),
        block=jax.block_until_ready,
    )
    ao = ao_deriv1[0]

    _, _ = _time_call(lambda: true_df_factors_from_pyscf_mol(mol), block=jax.block_until_ready)
    df_factors, df_s = _time_call(
        lambda: true_df_factors_from_pyscf_mol(mol),
        block=jax.block_until_ready,
    )

    s = overlap_matrix(basis)
    h1e = build_hcore(basis)
    cfg = RKSConfig(
        xc_spec=str(args.xc),
        max_cycle=int(args.max_cycle),
        conv_tol=1e-9,
        conv_tol_density=float(args.conv_tol_density),
        damping=float(args.damping),
        potential_clip=20.0,
        jk_backend="df",
        df_tol=1e-10,
    )

    def _run_scf():
        return run_rks_from_integrals(
            overlap=s,
            hcore=h1e,
            eri=None,
            nelectron=int(basis.atom_charges.sum()),
            nuclear_repulsion=float(spec.nuclear_repulsion),
            ao=ao,
            ao_deriv1=ao_deriv1,
            grid_weights=jnp.asarray(weights),
            df_factors=df_factors,
            init_mo_coeff=None,
            init_mo_occ=None,
            init_mo_energy=None,
            config=cfg,
        )

    _, _ = _time_call(_run_scf, block=lambda r: jax.block_until_ready(r.mo_energy))
    scf_result, scf_s = _time_call(_run_scf, block=lambda r: jax.block_until_ready(r.mo_energy))

    summary = {
        "system": label,
        "basis": str(args.basis),
        "xc": str(args.xc),
        "basis_s": basis_s,
        "grid_s": grid_s,
        "ao_s": ao_s,
        "df_s": df_s,
        "scf_s": scf_s,
        "cycles": int(scf_result.cycles),
        "converged": bool(scf_result.converged),
        "energy_ha": float(scf_result.total_energy),
        "nao": int(basis.nao),
        "ngrids": int(coords.shape[0]),
        "naux": int(df_factors.shape[0]),
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{label}_{str(args.xc).lower()}_{str(args.basis).lower()}_profile.json"
    outpath.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"summary={outpath}")


if __name__ == "__main__":
    main()
