#!/usr/bin/env python
from __future__ import annotations

import argparse
import numpy as np


WATER_ATOM = """
O  0.000000  0.000000  0.117790
H  0.000000  0.755453 -0.471161
H  0.000000 -0.755453 -0.471161
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-points", type=int, default=256)
    parser.add_argument("--point-selection", choices=("head", "even"), default="even")
    args = parser.parse_args()

    from jax import config as jax_config

    jax_config.update("jax_enable_x64", True)
    jax_config.update("jax_platform_name", "cpu")

    import jax
    import jax.numpy as jnp
    import jax_xc
    from pyscf import dft, gto, scf
    from pyscf.dft import numint

    from td_graddft.data import basis_from_pyscf_mol_cart, evaluate_cartesian_ao

    mol = gto.M(atom=WATER_ATOM, unit="Angstrom", basis="sto-3g", cart=True, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    grids = dft.gen_grid.Grids(mol)
    grids.level = 1
    grids.build()
    coords = np.asarray(grids.coords, dtype=np.float64)
    if args.max_points < len(coords):
        if args.point_selection == "head":
            idx = np.arange(args.max_points)
        else:
            idx = np.linspace(0, len(coords) - 1, args.max_points, dtype=np.int64)
        coords = coords[idx]

    dm = np.asarray(mf.make_rdm1(), dtype=np.float64)
    ao = np.asarray(numint.eval_ao(mol, coords, deriv=1), dtype=np.float64)
    rho_gga = np.asarray(numint.eval_rho(mol, ao, dm, xctype="GGA"), dtype=np.float64)

    basis = basis_from_pyscf_mol_cart(mol, max_l=3, precompute_eri_groups=False)
    dm_jax = jnp.asarray(dm)

    def rho_fn(r):
        ao0 = evaluate_cartesian_ao(basis, r[jnp.newaxis, :], deriv=0)[0]
        rho = jnp.einsum("p,pq,q->", ao0, dm_jax, ao0)
        return jnp.maximum(rho, jnp.asarray(1e-30))

    def values(name):
        fn = getattr(jax_xc, name)(polarized=False)
        return np.asarray(jax.jit(jax.vmap(lambda r: fn(rho_fn, r)))(jnp.asarray(coords)))

    j_pbeh = values("hyb_gga_xc_pbeh")
    j_pbe_mix = 0.75 * values("gga_x_pbe") + values("gga_c_pbe")

    j_b3 = values("hyb_gga_xc_b3lyp")
    j_b3_mix = (
        0.08 * values("lda_x")
        + 0.72 * values("gga_x_b88")
        + 0.19 * values("lda_c_vwn_rpa")
        + 0.81 * values("gga_c_lyp")
    )

    p_pbeh = dft.libxc.eval_xc("hyb_gga_xc_pbeh", rho_gga, spin=0, deriv=0)[0]
    p_b3 = dft.libxc.eval_xc("hyb_gga_xc_b3lyp", rho_gga, spin=0, deriv=0)[0]

    for label, lhs, rhs in (
        ("jax_pbeh_vs_jax_mix", j_pbeh, j_pbe_mix),
        ("jax_pbeh_vs_pyscf", j_pbeh, p_pbeh),
        ("jax_mix_pbeh_vs_pyscf", j_pbe_mix, p_pbeh),
        ("jax_b3lyp_vs_jax_mix", j_b3, j_b3_mix),
        ("jax_b3lyp_vs_pyscf", j_b3, p_b3),
        ("jax_mix_b3lyp_vs_pyscf", j_b3_mix, p_b3),
    ):
        diff = np.asarray(lhs) - np.asarray(rhs)
        print(
            label,
            "max",
            float(np.max(np.abs(diff))),
            "rms",
            float(np.sqrt(np.mean(diff * diff))),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
