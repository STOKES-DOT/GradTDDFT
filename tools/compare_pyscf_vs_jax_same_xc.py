from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pyscf import dft, gto

from td_graddft import tdscf
from td_graddft.features import restricted_grid_features_with_gradients
from td_graddft.xc_backend.jax_libxc import eval_xc_response_tensor, hybrid_coeff, xc_type
from td_graddft.reference_legacy import restricted_reference_from_pyscf
from td_graddft.spectra import HARTREE_TO_EV, lorentzian_spectrum


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
}


@dataclass(frozen=True)
class CompareResult:
    mae_ev: float
    max_ev: float
    mae_f: float
    max_f: float
    ncomp: int


class SemilocalResponseFunctional:
    def __init__(self, xc_spec: str):
        self.xc_spec = str(xc_spec).lower()
        self.exact_exchange_fraction = float(hybrid_coeff(self.xc_spec))
        self.response_feature_kind = str(xc_type(self.xc_spec))

    def grid_response_tensor(self, molecule):
        features, grad_rho = restricted_grid_features_with_gradients(molecule)
        tau = features.tau_a + features.tau_b
        _, tensor = eval_xc_response_tensor(
            self.xc_spec,
            features.rho,
            grad=grad_rho,
            tau=tau,
        )
        return tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--molecule", choices=tuple(MOLECULES.keys()), default="benzene")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--xc", default="b3lyp")
    p.add_argument("--nstates", type=int, default=5)
    p.add_argument("--eta-ev", type=float, default=0.08)
    p.add_argument("--grid-points", type=int, default=1200)
    p.add_argument(
        "--grid-max-ev",
        type=float,
        default=None,
        help="max energy (eV) for broadened spectrum; default uses auto range from states.",
    )
    p.add_argument("--outdir", default="outputs/pyscf_vs_jax_same_xc")
    p.add_argument(
        "--allow-approx-b3lyp",
        action="store_true",
        help=(
            "allow B3LYP comparison even though TD-GradDFT currently uses an approximate "
            "B3LYP semilocal backbone (PW/PBE-correlation surrogate)."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    xc_key = str(args.xc).strip().lower()
    kernel_source = "jax"
    if xc_key == "b3lyp" and not args.allow_approx_b3lyp:
        raise ValueError(
            "Strict same-functional comparison is not valid for xc='b3lyp' in current "
            "TD-GradDFT jax_libxc, because B3LYP is implemented as an approximate alias. "
            "Use --allow-approx-b3lyp to run the all-JAX approximate comparison explicitly."
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mol = gto.M(
        atom=MOLECULES[args.molecule],
        basis=args.basis,
        unit="Angstrom",
        spin=0,
        charge=0,
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = args.xc
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF SCF did not converge.")

    td = mf.TDDFT()
    td.nstates = int(args.nstates)
    td.kernel()
    ref_e = np.asarray(td.e, dtype=float)
    ref_f = np.asarray(td.oscillator_strength(), dtype=float)

    ref = restricted_reference_from_pyscf(mf)
    jax_td = tdscf.TDDFT(ref, xc_functional=args.xc)
    pred = jax_td.kernel(nstates=int(args.nstates))
    pred_e = np.asarray(pred.excitation_energies, dtype=float)
    pred_f = np.asarray(jax_td.oscillator_strength(), dtype=float)

    n = min(ref_e.size, pred_e.size, ref_f.size, pred_f.size, int(args.nstates))
    diff_ev = np.abs((pred_e[:n] - ref_e[:n]) * HARTREE_TO_EV)
    diff_f = np.abs(pred_f[:n] - ref_f[:n])
    result = CompareResult(
        mae_ev=float(np.mean(diff_ev)),
        max_ev=float(np.max(diff_ev)),
        mae_f=float(np.mean(diff_f)),
        max_f=float(np.max(diff_f)),
        ncomp=n,
    )

    csv_path = outdir / f"{args.molecule}_{args.xc}_{args.basis}_state_compare.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "state",
                "pyscf_energy_ev",
                "jax_energy_ev",
                "abs_diff_ev",
                "pyscf_osc",
                "jax_osc",
                "abs_diff_osc",
            ]
        )
        for i in range(n):
            w.writerow(
                [
                    i + 1,
                    float(ref_e[i] * HARTREE_TO_EV),
                    float(pred_e[i] * HARTREE_TO_EV),
                    float(diff_ev[i]),
                    float(ref_f[i]),
                    float(pred_f[i]),
                    float(diff_f[i]),
                ]
            )

    ref_e_ev = ref_e[:n] * HARTREE_TO_EV
    pred_e_ev = pred_e[:n] * HARTREE_TO_EV
    ref_f_n = ref_f[:n]
    pred_f_n = pred_f[:n]
    if n == 0:
        raise RuntimeError("No common excited states to compare.")

    auto_max_ev = float(max(np.max(ref_e_ev), np.max(pred_e_ev)) + 8.0 * float(args.eta_ev))
    grid_max_ev = auto_max_ev if args.grid_max_ev is None else float(args.grid_max_ev)
    if grid_max_ev <= 0.0:
        raise ValueError("grid_max_ev must be positive.")
    grid_ev = np.linspace(0.0, grid_max_ev, int(args.grid_points), dtype=float)
    ref_curve = np.asarray(
        lorentzian_spectrum(
            jnp.asarray(ref_e_ev),
            jnp.asarray(ref_f_n),
            jnp.asarray(grid_ev),
            eta=float(args.eta_ev),
        ),
        dtype=float,
    )
    pred_curve = np.asarray(
        lorentzian_spectrum(
            jnp.asarray(pred_e_ev),
            jnp.asarray(pred_f_n),
            jnp.asarray(grid_ev),
            eta=float(args.eta_ev),
        ),
        dtype=float,
    )

    curve_csv = outdir / f"{args.molecule}_{args.xc}_{args.basis}_spectrum_curve.csv"
    with curve_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["energy_ev", "pyscf_absorption", "jax_absorption"])
        for i in range(grid_ev.size):
            w.writerow([float(grid_ev[i]), float(ref_curve[i]), float(pred_curve[i])])

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(grid_ev, ref_curve, lw=2.0, label="PySCF TDDFT")
    ax.plot(grid_ev, pred_curve, lw=2.0, label="JAX TDDFT")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Absorption (arb. units)")
    ax.set_title(f"{args.molecule.title()} Absorption: {args.xc.upper()}/{args.basis}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    spectrum_png = outdir / f"{args.molecule}_{args.xc}_{args.basis}_spectrum_compare.png"
    fig.savefig(spectrum_png, dpi=170)
    plt.close(fig)

    print(f"molecule={args.molecule}")
    print(f"xc={args.xc}, basis={args.basis}, nstates={args.nstates}")
    print(f"kernel_source={kernel_source}")
    print(f"compared_states={result.ncomp}")
    print(f"mae_energy_ev={result.mae_ev:.8f}, max_energy_ev={result.max_ev:.8f}")
    print(f"mae_osc={result.mae_f:.8e}, max_osc={result.max_f:.8e}")
    print(f"csv={csv_path}")
    print(f"spectrum_curve_csv={curve_csv}")
    print(f"spectrum_png={spectrum_png}")


if __name__ == "__main__":
    main()
