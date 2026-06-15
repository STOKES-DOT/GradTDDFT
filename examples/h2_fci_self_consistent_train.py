#!/usr/bin/env python3
"""
H2 dissociation curve training with self_consistent + implicit_commutator mode.

This script demonstrates full differentiable SCF training where:
- Density is NOT fixed but converged through SCF iterations
- Gradients flow through SCF via implicit commutator method
- This is the recommended setting for physics-correct training

Compare with fixed_density mode which freezes PySCF density.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from td_graddft import neural_xc
from td_graddft.xc_backend.jax_libxc import b3lyp_component_basis
from td_graddft.data.reference import restricted_reference_from_pyscf
from td_graddft.training import (
    GroundStateDatum,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    ground_state_mse_loss,
    make_ground_state_train_step,
    predict_ground_state_total_energy,
)


def build_h2_mol(r_angstrom: float, basis: str):
    """Build H2 molecule with given bond length."""
    from pyscf import gto

    mol = gto.Mole()
    mol.atom = f"""
    H 0.000000 0.000000 {-0.5 * r_angstrom:.10f}
    H 0.000000 0.000000 {+0.5 * r_angstrom:.10f}
    """
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.spin = 0
    mol.charge = 0
    mol.verbose = 0
    mol.build()
    return mol


def compute_fci_ground_state(mol):
    """Compute FCI ground state energy for H2."""
    from pyscf import ao2mo, fci, scf

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge")

    h1_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri_mo = ao2mo.kernel(mol, mf.mo_coeff)
    norb = h1_mo.shape[0]
    nelec = mol.nelectron

    cisolver = fci.direct_spin0.FCI(mol)
    e_roots, _ = cisolver.kernel(h1_mo, eri_mo, norb, nelec, nroots=1)
    e_nuc = float(mol.energy_nuc())
    e_fci = float(e_roots) if np.isscalar(e_roots) else float(e_roots[0])
    return e_fci + e_nuc, mf


def build_reference_from_rks(mol, xc: str = "b3lyp"):
    """Build reference molecule from PySCF RKS calculation."""
    from pyscf import dft

    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = 3
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RKS did not converge")
    return restricted_reference_from_pyscf(mf, compute_local_hfx_features=False)


def main():
    parser = argparse.ArgumentParser(
        description="H2 dissociation training with self_consistent + implicit_commutator"
    )
    parser.add_argument("--basis", type=str, default="sto-3g")
    parser.add_argument("--r-min", type=float, default=0.5)
    parser.add_argument("--r-max", type=float, default=3.0)
    parser.add_argument("--points", type=int, default=21)
    parser.add_argument("--train-points", type=int, default=5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", type=str, default="64,64")
    parser.add_argument("--density-constraint-weight", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/h2_self_consistent_implicit",
    )
    # Key parameters for different modes
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fixed_density", "self_consistent"],
        default="self_consistent",
        help="SCF mode: fixed_density (fast) or self_consistent (physics-correct)",
    )
    parser.add_argument(
        "--scf-gradient-mode",
        type=str,
        choices=["unrolled", "implicit_commutator"],
        default="implicit_commutator",
        help="Gradient mode for self_consistent: implicit_commutator recommended",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hidden_dims = tuple(int(x.strip()) for x in args.hidden_dims.split(",") if x.strip())
    r_grid = np.linspace(args.r_min, args.r_max, args.points)
    train_indices = np.linspace(0, args.points - 1, args.train_points, dtype=int).tolist()

    print("=" * 70)
    print("H2 Dissociation Curve Training")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"SCF Gradient Mode: {args.scf_gradient_mode}")
    print(f"Basis: {args.basis}")
    print(f"R range: {args.r_min:.2f} - {args.r_max:.2f} A ({args.points} points)")
    print(f"Training points: {train_indices}")
    print(f"Steps: {args.steps}, LR: {args.learning_rate}")
    print("=" * 70)

    # Build references and FCI energies
    print("\nBuilding references and computing FCI energies...")
    references = []
    e0_fci = np.zeros_like(r_grid)

    for i, r in enumerate(r_grid):
        mol = build_h2_mol(float(r), basis=args.basis)
        e_fci, mf = compute_fci_ground_state(mol)
        ref = build_reference_from_rks(mol, xc="b3lyp")
        references.append(ref)
        e0_fci[i] = e_fci
        print(f"  R={r:.3f} A: E_FCI = {e_fci:.8f} Ha")

    # Build neural XC functional
    functional = neural_xc.Functional(
        semilocal_xc=tuple(b3lyp_component_basis()),
        hidden_dims=hidden_dims,
        name="neural_xc_h2_self_consistent",
    )

    # Prepare training data
    train_data = [
        GroundStateDatum(
            molecule=references[idx],
            target_total_energy=jnp.asarray(float(e0_fci[idx])),
            density_constraint_weight=args.density_constraint_weight,
        )
        for idx in train_indices
    ]

    # KEY: Configure training mode
    training_config = GroundStateTrainingConfig(
        mode=args.mode,
        scf_gradient_mode=args.scf_gradient_mode,
        energy_mse_weight=1.0,
        energy_mae_weight=1.0,
        scf_max_cycle=12,
        scf_conv_tol_density=1e-8,
    )

    print(f"\nTraining config:")
    print(f"  mode = {training_config.mode}")
    print(f"  scf_gradient_mode = {training_config.scf_gradient_mode}")
    print(f"  scf_max_cycle = {training_config.scf_max_cycle}")

    # Create train state
    train_state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(args.seed),
        references[0],
        optax.adam(args.learning_rate),
    )

    # Create train step
    train_step = make_ground_state_train_step(functional, training_config=training_config)

    # Training loop
    print("\nStarting training...")
    t0 = time.perf_counter()
    loss_history = []

    for step in range(1, args.steps + 1):
        train_state, metrics = train_step(train_state, train_data)
        loss = float(metrics["loss"])
        loss_history.append(loss)

        if step == 1 or step == args.steps or (step % args.log_every == 0):
            print(f"  Step {step:4d}/{args.steps}: loss = {loss:.6e}")

    elapsed = time.perf_counter() - t0
    print(f"\nTraining completed in {elapsed:.2f} s")

    # Evaluate on full curve
    print("\nEvaluating on full dissociation curve...")
    e0_pred = np.zeros_like(r_grid)

    for i, ref in enumerate(references):
        e_pred = float(
            predict_ground_state_total_energy(
                train_state.params,
                functional,
                ref,
                training_config=training_config,
            )
        )
        e0_pred[i] = e_pred
        print(f"  R={r_grid[i]:.3f} A: E_pred = {e_pred:.8f} Ha, E_FCI = {e0_fci[i]:.8f} Ha")

    # Compute MAE
    mae_hartree = np.mean(np.abs(e0_pred - e0_fci))
    mae_ev = mae_hartree * 27.2114
    print(f"\nMAE: {mae_hartree:.6e} Ha = {mae_ev:.4f} eV")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Dissociation curve
    axes[0].plot(r_grid, e0_fci, "b-", lw=2.5, label="FCI (reference)")
    axes[0].plot(r_grid, e0_pred, "r--", lw=2, label=f"Neural XC ({args.mode})")
    axes[0].scatter(
        r_grid[train_indices],
        e0_fci[train_indices],
        s=60,
        marker="o",
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label="Training points",
        zorder=5,
    )
    axes[0].set_xlabel("H-H Distance (Angstrom)", fontsize=12)
    axes[0].set_ylabel("Total Energy (Hartree)", fontsize=12)
    axes[0].set_title(f"H2 Dissociation: {args.mode} + {args.scf_gradient_mode}", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # Training loss
    axes[1].semilogy(range(1, len(loss_history) + 1), loss_history, "g-", lw=1.5)
    axes[1].set_xlabel("Training Step", fontsize=12)
    axes[1].set_ylabel("Loss (log scale)", fontsize=12)
    axes[1].set_title("Training Convergence", fontsize=12)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig_path = outdir / f"h2_{args.mode}_{args.scf_gradient_mode}.png"
    plt.savefig(fig_path, dpi=200)
    print(f"\nPlot saved to: {fig_path}")

    # Save results
    results = {
        "mode": args.mode,
        "scf_gradient_mode": args.scf_gradient_mode,
        "r_grid": r_grid.tolist(),
        "e0_fci": e0_fci.tolist(),
        "e0_pred": e0_pred.tolist(),
        "loss_history": loss_history,
        "mae_hartree": float(mae_hartree),
        "mae_ev": float(mae_ev),
        "train_time_s": elapsed,
    }

    import json
    json_path = outdir / f"results_{args.mode}_{args.scf_gradient_mode}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")

    plt.close()
    return results


if __name__ == "__main__":
    main()
