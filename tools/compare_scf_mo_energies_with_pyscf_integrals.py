from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import dft, gto
from pyscf.dft import numint

from td_graddft.scf import RKSConfig, run_rks_from_integrals

jax.config.update("jax_enable_x64", True)


MOLECULES = {
    "water": """
O  0.000000  0.000000  0.117790
H  0.000000  0.755453 -0.471161
H  0.000000 -0.755453 -0.471161
""",
    "h2o2": """
O   0.000000   0.000000  -0.731600
O   0.000000   0.000000   0.731600
H   0.946218   0.000000  -0.930999
H  -0.346790   0.880378   0.930999
""",
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
}

HARTREE_TO_EV = 27.211386245988


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--molecule", choices=tuple(MOLECULES.keys()), default="benzene")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--xc", default="pbe0")
    p.add_argument("--grids-level", type=int, default=0)
    p.add_argument("--scf-max-cycle", type=int, default=120)
    p.add_argument("--scf-damping", type=float, default=0.15)
    p.add_argument("--outdir", default="outputs/scf_mo_compare")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mol = gto.M(
        atom=MOLECULES[args.molecule],
        basis=args.basis,
        unit="Angstrom",
        spin=0,
        charge=0,
        cart=True,
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = str(args.xc)
    mf.grids.level = int(args.grids_level)
    mf.conv_tol = 1e-10
    mf.max_cycle = int(args.scf_max_cycle)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF SCF did not converge.")

    if getattr(mf.grids, "coords", None) is None:
        mf.grids.build()
    coords = np.asarray(mf.grids.coords, dtype=np.float64)
    weights = np.asarray(mf.grids.weights, dtype=np.float64)
    ao = np.asarray(numint.eval_ao(mol, coords, deriv=0), dtype=np.float64)
    ao_deriv1 = np.asarray(numint.eval_ao(mol, coords, deriv=1), dtype=np.float64)

    overlap = np.asarray(mol.intor_symmetric("int1e_ovlp"), dtype=np.float64)
    hcore = np.asarray(mf.get_hcore(), dtype=np.float64)
    eri = np.asarray(mol.intor("int2e", aosym="s1"), dtype=np.float64)

    rks_cfg = RKSConfig(
        xc_spec=str(args.xc),
        max_cycle=int(args.scf_max_cycle),
        conv_tol=1e-10,
        conv_tol_density=1e-8,
        damping=float(args.scf_damping),
        potential_clip=20.0,
    )
    jax_out = run_rks_from_integrals(
        overlap=jnp.asarray(overlap),
        hcore=jnp.asarray(hcore),
        eri=jnp.asarray(eri),
        nelectron=int(mol.nelectron),
        nuclear_repulsion=float(mol.energy_nuc()),
        ao=jnp.asarray(ao),
        ao_deriv1=jnp.asarray(ao_deriv1),
        grid_weights=jnp.asarray(weights),
        init_mo_coeff=jnp.asarray(mf.mo_coeff),
        init_mo_occ=jnp.asarray(mf.mo_occ),
        init_mo_energy=jnp.asarray(mf.mo_energy),
        config=rks_cfg,
    )
    if not jax_out.converged:
        raise RuntimeError("JAX RKS did not converge.")

    pyscf_mo = np.asarray(mf.mo_energy, dtype=float).reshape(-1)
    jax_mo = np.asarray(jax_out.mo_energy, dtype=float).reshape(-1)
    mo_occ = np.asarray(mf.mo_occ, dtype=float).reshape(-1)
    if pyscf_mo.shape != jax_mo.shape:
        raise RuntimeError(
            f"MO size mismatch: pyscf={pyscf_mo.shape}, jax={jax_mo.shape}."
        )

    diff = jax_mo - pyscf_mo
    abs_diff = np.abs(diff)
    homo_idx = int(np.where(mo_occ > 0)[0][-1])
    lumo_idx = int(np.where(mo_occ == 0)[0][0])

    csv_path = outdir / f"{args.molecule}_{args.xc}_{args.basis}_all_mo_compare.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "mo_index",
                "occupation",
                "pyscf_mo_ha",
                "jax_mo_ha",
                "diff_ha",
                "abs_diff_ha",
                "pyscf_mo_ev",
                "jax_mo_ev",
                "abs_diff_mev",
            ]
        )
        for i in range(pyscf_mo.size):
            w.writerow(
                [
                    i,
                    float(mo_occ[i]),
                    float(pyscf_mo[i]),
                    float(jax_mo[i]),
                    float(diff[i]),
                    float(abs_diff[i]),
                    float(pyscf_mo[i] * HARTREE_TO_EV),
                    float(jax_mo[i] * HARTREE_TO_EV),
                    float(abs_diff[i] * HARTREE_TO_EV * 1000.0),
                ]
            )

    summary = {
        "molecule": str(args.molecule),
        "basis": str(args.basis),
        "xc": str(args.xc),
        "nmo": int(pyscf_mo.size),
        "mae_ha": float(abs_diff.mean()),
        "max_abs_diff_ha": float(abs_diff.max()),
        "rmse_ha": float(np.sqrt(np.mean(diff**2))),
        "mae_mev": float(abs_diff.mean() * HARTREE_TO_EV * 1000.0),
        "max_abs_diff_mev": float(abs_diff.max() * HARTREE_TO_EV * 1000.0),
        "pyscf_total_energy_ha": float(mf.e_tot),
        "jax_total_energy_ha": float(jax_out.total_energy),
        "total_energy_diff_ha": float(jax_out.total_energy - mf.e_tot),
        "homo_index": homo_idx,
        "lumo_index": lumo_idx,
        "pyscf_homo_ha": float(pyscf_mo[homo_idx]),
        "jax_homo_ha": float(jax_mo[homo_idx]),
        "pyscf_lumo_ha": float(pyscf_mo[lumo_idx]),
        "jax_lumo_ha": float(jax_mo[lumo_idx]),
    }

    print(summary)
    print(f"csv={csv_path}")


if __name__ == "__main__":
    main()
