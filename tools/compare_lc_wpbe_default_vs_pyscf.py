from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from td_graddft.nn_rsh import RSH, get_rsh_functional_preset
from td_graddft.reference_legacy import restricted_reference_from_pyscf
from td_graddft.training.targets import _predict_ground_state_total_energy_from_molecule


WATER_GEOMETRY = """
O  0.000000  0.000000  0.117790
H  0.000000  0.755453 -0.471161
H  0.000000 -0.755453 -0.471161
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fixed-density strictness check for default LC-wPBE against PySCF LC_WPBE."
        ),
    )
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--grid-level", type=int, default=2)
    parser.add_argument("--conv-tol", type=float, default=1e-11)
    parser.add_argument("--threshold", type=float, default=1e-5)
    parser.add_argument("--hfx-chunk-size", type=int, default=512)
    parser.add_argument(
        "--outdir",
        default="outputs/water_lc_wpbe_default_vs_pyscf",
    )
    return parser.parse_args()


def _build_pyscf_lc_wpbe(*, basis: str, grid_level: int, conv_tol: float) -> Any:
    from pyscf import dft, gto

    mol = gto.M(
        atom=WATER_GEOMETRY,
        unit="Angstrom",
        basis=str(basis),
        spin=0,
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = "LC_WPBE"
    mf.grids.level = int(grid_level)
    mf.conv_tol = float(conv_tol)
    mf.kernel()
    return mf


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mf = _build_pyscf_lc_wpbe(
        basis=str(args.basis),
        grid_level=int(args.grid_level),
        conv_tol=float(args.conv_tol),
    )
    if not bool(mf.converged):
        raise RuntimeError("PySCF LC_WPBE reference did not converge.")

    preset = get_rsh_functional_preset("lc-wpbe")
    omega = float(preset.default_omega)
    molecule = restricted_reference_from_pyscf(
        mf,
        compute_local_hfx_features=True,
        compute_local_hfx_aux=True,
        hfx_omega_values=(omega,),
        hfx_chunk_size=int(args.hfx_chunk_size),
    )

    functional = RSH("lc-wpbe").trainable(hidden_dims=())
    params = functional.init_from_molecule(jax.random.PRNGKey(0), molecule)
    params = functional.params_with_resolved(
        params,
        preset.default_params,
        molecule=molecule,
        preserve_network=False,
    )
    resolved = functional.resolve_parameters(params, molecule)

    td_graddft_energy = float(
        _predict_ground_state_total_energy_from_molecule(params, functional, molecule)
    )
    pyscf_density_energy = float(mf.energy_tot(dm=mf.make_rdm1()))
    diff = td_graddft_energy - pyscf_density_energy
    occ = np.asarray(mf.mo_occ)
    mo_energy = np.asarray(mf.mo_energy)
    homo = float(mo_energy[np.where(occ > 0)[0][-1]])
    lumo = float(mo_energy[np.where(occ == 0)[0][0]])

    summary = {
        "system": "water",
        "basis": str(args.basis),
        "grid_level": int(args.grid_level),
        "ngrids": int(mf.grids.weights.size),
        "pyscf_xc": str(mf.xc),
        "pyscf_converged": bool(mf.converged),
        "pyscf_total_energy_ha": pyscf_density_energy,
        "td_graddft_fixed_density_total_energy_ha": td_graddft_energy,
        "total_energy_diff_ha": diff,
        "abs_total_energy_diff_ha": abs(diff),
        "threshold_ha": float(args.threshold),
        "passed": bool(abs(diff) <= float(args.threshold)),
        "pyscf_rsh_coeff": [
            float(x) for x in mf._numint.rsh_coeff(mf.xc)
        ],
        "pyscf_rsh_and_hybrid_coeff": [
            float(x)
            for x in mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mf.mol.spin)
        ],
        "preset_to_pyscf_rsh": [
            float(x) for x in preset.to_pyscf_rsh()
        ],
        "preset_to_pyscf_rsh_and_hybrid": [
            float(x) for x in preset.to_pyscf_rsh_and_hybrid()
        ],
        "resolved_sr_hf_fraction": float(resolved.sr_hf_fraction),
        "resolved_lr_hf_fraction": float(resolved.lr_hf_fraction),
        "resolved_omega": float(resolved.omega),
        "homo_pyscf_ha": homo,
        "lumo_pyscf_ha": lumo,
    }

    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
