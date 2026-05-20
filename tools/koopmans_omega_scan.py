"""
Koopmans-theorem ω optimization for LC-wPBE on a water molecule.

Target function (from Jiao, Peng, Suo):
  J(ω) = J_N(ω) + J_{N+1}(ω)
  J_N(ω)   = |ε_HOMO(N, ω) + IP(N)|²
  J_{N+1}(ω) = |ε_HOMO(N+1, ω) + IP(N+1)|²

  IP(N)   = E_cation(N-1) - E_neutral(N)
  IP(N+1) = E_neutral(N) - E_anion(N+1)

Uses PySCF + scipy.optimize, ω ∈ [0.05, 0.30].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from pyscf import dft, gto
from scipy.optimize import minimize_scalar

WATER_GEOMETRY = """
O  0.000000  0.000000  0.117790
H  0.000000  0.755453 -0.471161
H  0.000000 -0.755453 -0.471161
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Koopmans ω optimization for LC-wPBE on water.")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--omega-bounds", type=float, nargs=2, default=[0.05, 0.30])
    p.add_argument("--xatol", type=float, default=0.005)
    p.add_argument("--maxiter", type=int, default=12)
    p.add_argument("--loss-kind", choices=("squared", "absolute"), default="squared")
    p.add_argument("--n-omega-grid", type=int, default=50, help="Grid points for J(ω) curve.")
    p.add_argument("--outdir", default="outputs/koopmans_omega_scan")
    return p.parse_args()


def build_molecule(basis: str) -> gto.Mole:
    mol = gto.M(
        atom=WATER_GEOMETRY,
        unit="Angstrom",
        basis=basis,
        spin=0,
        verbose=0,
    )
    mol.build()
    return mol


def neutral_scf(mol: gto.Mole, omega: float) -> dict[str, Any]:
    mf = dft.RKS(mol)
    mf.xc = "LC_WPBE"
    mf.omega = omega
    mf.conv_tol = 1e-10
    mf.kernel()
    nocc = mol.nelec[0]
    homo = float(mf.mo_energy[nocc - 1])
    lumo = float(mf.mo_energy[nocc])
    return {
        "energy": float(mf.e_tot),
        "homo": homo,
        "lumo": lumo,
        "converged": bool(mf.converged),
    }


def charged_scf(mol: gto.Mole, omega: float, charge: int, spin: int) -> dict[str, Any]:
    mol_copy = mol.copy()
    mol_copy.charge = charge
    mol_copy.spin = spin
    mol_copy.build()

    mf = dft.UKS(mol_copy)
    mf.xc = "LC_WPBE"
    mf.omega = omega
    mf.conv_tol = 1e-10

    dm0 = None
    try:
        mf.kernel()
    except Exception:
        mf.kernel(dm0=dm0)

    nocc_a = mol_copy.nelec[0]
    nocc_b = mol_copy.nelec[1]
    homo_a = float(mf.mo_energy[0][nocc_a - 1]) if nocc_a > 0 else float("-inf")
    homo_b = float(mf.mo_energy[1][nocc_b - 1]) if nocc_b > 0 else float("-inf")
    homo = max(homo_a, homo_b)

    return {
        "energy": float(mf.e_tot),
        "homo": homo,
        "converged": bool(mf.converged),
    }


def koopmans_loss(omega: float, mol: gto.Mole, loss_kind: str = "squared") -> dict[str, float]:
    neutral = neutral_scf(mol, omega)
    cation = charged_scf(mol, omega, charge=1, spin=1)
    anion = charged_scf(mol, omega, charge=-1, spin=1)

    E_N = neutral["energy"]
    E_cation = cation["energy"]
    E_anion = anion["energy"]
    homo_N = neutral["homo"]
    homo_N1 = anion["homo"]

    IP_N = E_cation - E_N       # IP(N) = E(N-1) - E(N)
    IP_N1 = E_N - E_anion       # IP(N+1) = E(N) - E(N+1)

    residual_N = homo_N + IP_N
    residual_N1 = homo_N1 + IP_N1

    if loss_kind == "squared":
        J_N = residual_N ** 2
        J_N1 = residual_N1 ** 2
    else:
        J_N = abs(residual_N)
        J_N1 = abs(residual_N1)

    return {
        "omega": omega,
        "J": J_N + J_N1,
        "J_N": J_N,
        "J_N1": J_N1,
        "residual_N": residual_N,
        "residual_N1": residual_N1,
        "homo_N": homo_N,
        "homo_N1": homo_N1,
        "IP_N": IP_N,
        "IP_N1": IP_N1,
        "E_neutral": E_N,
        "E_cation": E_cation,
        "E_anion": E_anion,
        "neutral_converged": neutral["converged"],
        "cation_converged": cation["converged"],
        "anion_converged": anion["converged"],
    }


def scan_j_curve(mol: gto.Mole, bounds: tuple[float, float], n_points: int, loss_kind: str):
    omegas = np.linspace(bounds[0], bounds[1], n_points)
    results = []
    for w in omegas:
        r = koopmans_loss(float(w), mol, loss_kind)
        results.append(r)
        print(f"  scan ω={w:.4f}: J={r['J']:.6e}  J_N={r['J_N']:.2e}  J_N1={r['J_N1']:.2e}")
    return results


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mol = build_molecule(args.basis)
    print(f"Molecule: H2O, basis={args.basis}")
    print(f"n_electrons = {mol.nelec}")
    print(f"Omega bounds = {args.omega_bounds}")
    print()

    # Grid scan before optimization
    print("=== J(ω) grid scan ===")
    scan_results = scan_j_curve(
        mol,
        bounds=(args.omega_bounds[0], args.omega_bounds[1]),
        n_points=args.n_omega_grid,
        loss_kind=args.loss_kind,
    )

    # Optimize
    print(f"\n=== scipy minimize_scalar (bounded, {args.loss_kind}) ===")
    history: list[dict[str, float]] = []

    def objective(omega: float) -> float:
        r = koopmans_loss(float(omega), mol, args.loss_kind)
        history.append(r)
        print(
            f"  eval={len(history):03d}  ω={omega:.6f}  "
            f"J={r['J']:.6e}  J_N={r['J_N']:.2e}  J_N1={r['J_N1']:.2e}  "
            f"IP(N)={r['IP_N']:.4f}  IP(N+1)={r['IP_N1']:.4f}"
        )
        return r["J"]

    result = minimize_scalar(
        objective,
        bounds=(float(args.omega_bounds[0]), float(args.omega_bounds[1])),
        method="bounded",
        options={"xatol": float(args.xatol), "maxiter": int(args.maxiter)},
    )

    # Summary
    optimal_omega = float(result.x)
    opt_result = koopmans_loss(optimal_omega, mol, args.loss_kind)

    print(f"\n{'='*60}")
    print(f"OPTIMAL ω = {optimal_omega:.6f}")
    print(f"J(ω_opt)   = {opt_result['J']:.6e}")
    print(f"J_N(ω_opt)  = {opt_result['J_N']:.2e}  (|ε_HOMO(N) + IP(N)|{'²' if args.loss_kind == 'squared' else ''})")
    print(f"J_N1(ω_opt) = {opt_result['J_N1']:.2e}  (|ε_HOMO(N+1) + IP(N+1)|{'²' if args.loss_kind == 'squared' else ''})")
    print(f"ε_HOMO(N)   = {opt_result['homo_N']:.6f} Ha  ({opt_result['homo_N'] * 27.2114:.4f} eV)")
    print(f"IP(N)       = {opt_result['IP_N']:.6f} Ha  ({opt_result['IP_N'] * 27.2114:.4f} eV)")
    print(f"ε_HOMO(N+1)  = {opt_result['homo_N1']:.6f} Ha  ({opt_result['homo_N1'] * 27.2114:.4f} eV)")
    print(f"IP(N+1)      = {opt_result['IP_N1']:.6f} Ha  ({opt_result['IP_N1'] * 27.2114:.4f} eV)")
    print(f"E_neutral    = {opt_result['E_neutral']:.8f} Ha")
    print(f"E_cation     = {opt_result['E_cation']:.8f} Ha")
    print(f"E_anion      = {opt_result['E_anion']:.8f} Ha")
    print(f"scipy.success = {result.success}")
    print(f"scipy.message = {result.message}")
    print(f"scipy.nfev    = {result.nfev}")

    # Save results
    summary = {
        "basis": args.basis,
        "omega_bounds": list(args.omega_bounds),
        "loss_kind": args.loss_kind,
        "optimal_omega": optimal_omega,
        "optimal_J": opt_result["J"],
        "optimal_J_N": opt_result["J_N"],
        "optimal_J_N1": opt_result["J_N1"],
        "homo_N": opt_result["homo_N"],
        "homo_N1": opt_result["homo_N1"],
        "IP_N": opt_result["IP_N"],
        "IP_N1": opt_result["IP_N1"],
        "E_neutral": opt_result["E_neutral"],
        "E_cation": opt_result["E_cation"],
        "E_anion": opt_result["E_anion"],
        "scipy_success": bool(result.success),
        "scipy_message": str(result.message),
        "scipy_nfev": int(result.nfev),
        "scan": [
            {
                "omega": r["omega"],
                "J": r["J"],
                "J_N": r["J_N"],
                "J_N1": r["J_N1"],
                "homo_N": r["homo_N"],
                "IP_N": r["IP_N"],
            }
            for r in scan_results
        ],
        "history": [
            {
                "eval": i,
                "omega": r["omega"],
                "J": r["J"],
                "J_N": r["J_N"],
                "J_N1": r["J_N1"],
            }
            for i, r in enumerate(history)
        ],
    }

    summary_path = outdir / "koopmans_omega_opt.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        omegas_scan = [r["omega"] for r in scan_results]
        J_scan = [r["J"] for r in scan_results]
        JN_scan = [r["J_N"] for r in scan_results]
        JN1_scan = [r["J_N1"] for r in scan_results]

        axes[0, 0].plot(omegas_scan, J_scan, "b-", linewidth=1.5, label="J(ω)")
        axes[0, 0].axvline(optimal_omega, color="r", linestyle="--", label=f"ω*={optimal_omega:.4f}")
        axes[0, 0].set_xlabel("ω")
        axes[0, 0].set_ylabel("J(ω)")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(omegas_scan, JN_scan, "g-", linewidth=1.5, label="J_N(ω)")
        axes[0, 1].plot(omegas_scan, JN1_scan, "orange", linewidth=1.5, label="J_{N+1}(ω)")
        axes[0, 1].axvline(optimal_omega, color="r", linestyle="--")
        axes[0, 1].set_xlabel("ω")
        axes[0, 1].set_ylabel("Component")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        homo_scan = [r["homo_N"] for r in scan_results]
        ip_scan = [-r["IP_N"] for r in scan_results]
        homo_n1_scan = [r["homo_N1"] for r in scan_results]
        ip_n1_scan = [-r["IP_N1"] for r in scan_results]

        axes[1, 0].plot(omegas_scan, homo_scan, "b-", linewidth=1.5, label="ε_HOMO(N, ω)")
        axes[1, 0].plot(omegas_scan, ip_scan, "b--", linewidth=1.5, label="-IP(N)")
        axes[1, 0].axvline(optimal_omega, color="r", linestyle="--")
        axes[1, 0].set_xlabel("ω")
        axes[1, 0].set_ylabel("Energy (Ha)")
        axes[1, 0].set_title("Neutral: ε_HOMO vs -IP")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        axes[1, 1].plot(omegas_scan, homo_n1_scan, "b-", linewidth=1.5, label="ε_HOMO(N+1, ω)")
        axes[1, 1].plot(omegas_scan, ip_n1_scan, "b--", linewidth=1.5, label="-IP(N+1)")
        axes[1, 1].axvline(optimal_omega, color="r", linestyle="--")
        axes[1, 1].set_xlabel("ω")
        axes[1, 1].set_ylabel("Energy (Ha)")
        axes[1, 1].set_title("Anion: ε_HOMO vs -IP")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        fig.suptitle(f"LC-wPBE Koopmans ω Optimization (H₂O, {args.basis})", fontsize=13)
        fig.tight_layout()
        plot_path = outdir / "koopmans_omega_scan.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to {plot_path}")
    except Exception as e:
        print(f"Plot failed: {e}")

    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
