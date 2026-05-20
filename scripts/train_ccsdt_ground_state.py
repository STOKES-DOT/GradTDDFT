#!/usr/bin/env python3
"""Train Neural XC functional on CCSD(T)/CBS ground-state energies from ANI-1x."""

import argparse
import time
import sys
from pathlib import Path

# Locate the project root
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax

from td_graddft import neural_xc
from td_graddft.jax_libxc import b3lyp_component_basis
from td_graddft.training import (
    GroundStateDatum,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    make_ground_state_train_step,
    predict_ground_state_total_energy,
)
from td_graddft.data.reference import restricted_reference_from_pyscf


def build_pyscf_mol(atomic_numbers, coords_angstrom, basis="def2-svp"):
    """Build PySCF molecule from atomic numbers and coordinates."""
    from pyscf import gto

    atom_spec = []
    for z, xyz in zip(atomic_numbers, coords_angstrom):
        from pyscf.data import elements
        symbol = elements.ELEMENTS[int(z)]
        atom_spec.append(f"{symbol} {xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}")

    mol = gto.Mole()
    mol.atom = "; ".join(atom_spec)
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.spin = 0
    mol.charge = 0
    mol.verbose = 0
    mol.build()
    return mol


def compute_pyscf_rks(mol, xc="pbe"):
    """Run PySCF RKS and return the mean-field object."""
    from pyscf import dft
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = 2
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF RKS did not converge")
    return mf


def load_ccsdt_samples(h5_path, n_samples, max_atoms=8, seed=42):
    """Load N conformers with valid CCSD(T) data from an ANI-1x CCSD(T) subset."""
    rng = np.random.RandomState(seed)
    with h5py.File(h5_path, "r") as f:
        all_keys = sorted(f.keys())
        # Filter to small molecules for fast SCF
        small_keys = []
        for k in all_keys:
            natoms = f[k]["atomic_numbers"].shape[0]
            if natoms <= max_atoms:
                small_keys.append(k)

        print(f"Found {len(small_keys)} molecular formulas with <= {max_atoms} atoms")

        samples = []
        tries = 0
        while len(samples) < n_samples and tries < 10000:
            tries += 1
            key = small_keys[rng.randint(0, len(small_keys))]
            grp = f[key]
            nconf = grp["coordinates"].shape[0]
            idx = rng.randint(0, nconf)
            ccsdt_e = float(grp["ccsd(t)_cbs.energy"][idx])
            if np.isnan(ccsdt_e):
                continue
            atomic_numbers = np.asarray(grp["atomic_numbers"][:], dtype=np.int32)
            coords = np.asarray(grp["coordinates"][idx], dtype=np.float64)
            samples.append({
                "formula": key,
                "conformer_idx": int(idx),
                "atomic_numbers": atomic_numbers,
                "coordinates": coords,
                "ccsdt_cbs_energy": ccsdt_e,
            })
    return samples


def main():
    parser = argparse.ArgumentParser(description="Train Neural XC on CCSD(T) ground-state energies")
    parser.add_argument("--h5-path", type=str, default="datasets/ani/ani1x_ccsdt.h5")
    parser.add_argument("--n-samples", type=int, default=50, help="Number of training conformers")
    parser.add_argument("--basis", type=str, default="def2-svp")
    parser.add_argument("--ref-xc", type=str, default="pbe", help="XC for reference density")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", type=str, default="32,32")
    parser.add_argument("--max-atoms", type=int, default=8, help="Max atoms per molecule")
    parser.add_argument("--density-constraint-weight", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--outdir", type=str, default="outputs/ccsdt_ground_state")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Load CCSD(T) samples
    print("=" * 60)
    print("Loading CCSD(T) samples...")
    samples = load_ccsdt_samples(args.h5_path, args.n_samples, max_atoms=args.max_atoms, seed=args.seed)
    print(f"Loaded {len(samples)} samples")
    for s in samples[:5]:
        print(f"  {s['formula']} idx={s['conformer_idx']} E_CCSD(T)={s['ccsdt_cbs_energy']:.8f} Ha")
    print()

    # 2. Build reference molecules via PySCF
    print("Building reference molecules via PySCF...")
    references = []
    failed = 0
    for i, s in enumerate(samples):
        try:
            mol = build_pyscf_mol(s["atomic_numbers"], s["coordinates"], basis=args.basis)
            mf = compute_pyscf_rks(mol, xc=args.ref_xc)
            ref = restricted_reference_from_pyscf(mf, compute_local_hfx_features=False)
            references.append(ref)
        except Exception as e:
            failed += 1
            print(f"  [{i}] {s['formula']} FAILED: {e}")
            continue
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{args.n_samples}] built (failed so far: {failed})")

    print(f"Successfully built {len(references)} reference molecules ({failed} failed)")
    samples = [s for s, r in zip(samples, references) if r is not None]  # align
    print()

    if len(references) == 0:
        raise RuntimeError("No reference molecules were built successfully!")

    # 3. Create training data
    print("Creating training data...")
    train_data = []
    for s, ref in zip(samples, references):
        datum = GroundStateDatum(
            molecule=ref,
            target_total_energy=jnp.asarray(s["ccsdt_cbs_energy"]),
            density_constraint_weight=args.density_constraint_weight,
        )
        train_data.append(datum)
    print(f"Created {len(train_data)} training data points")

    # Also compute PySCF reference energy for comparison
    pyscf_energies = [float(ref.mf_energy) for ref in references]
    ccsdt_energies = [s["ccsdt_cbs_energy"] for s in samples]
    energy_diff = np.abs(np.array(pyscf_energies) - np.array(ccsdt_energies))
    print(f"PySCF ({args.ref_xc}) vs CCSD(T)/CBS energy MAE: {np.mean(energy_diff)*27.2114:.4f} eV")

    # 4. Build neural XC functional
    hidden_dims = tuple(int(x.strip()) for x in args.hidden_dims.split(",") if x.strip())
    functional = neural_xc.Functional(
        semilocal_xc=tuple(b3lyp_component_basis()),
        hidden_dims=hidden_dims,
        architecture="mlp",
        name="neural_xc_ccsdt",
    )

    # 5. Training config
    training_config = GroundStateTrainingConfig(
        mode="self_consistent",
        scf_gradient_mode="impl",
        scf_implicit_forward_mode="input_state",
        energy_mse_weight=1.0,
        scf_max_cycle=12,
        scf_damping=0.25,
        scf_conv_tol_density=1e-8,
    )

    # 6. Create train state
    train_state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(args.seed),
        references[0],
        optax.adam(args.learning_rate),
    )

    # 7. Create train step
    train_step = make_ground_state_train_step(functional, training_config=training_config)

    # 8. Training loop
    print("\n" + "=" * 60)
    print("Training...")
    t0 = time.perf_counter()
    loss_history = []
    energy_errors = []

    for step in range(1, args.steps + 1):
        train_state, metrics = train_step(train_state, train_data)
        loss = float(metrics["loss"])
        loss_history.append(loss)

        if step == 1 or step == args.steps or (step % args.log_every == 0):
            # Compute energy MAE on training set
            pred_energies = []
            for datum in train_data:
                e_pred = float(predict_ground_state_total_energy(
                    train_state.params, functional, datum.molecule, training_config=training_config,
                ))
                pred_energies.append(e_pred)
            energy_mae = np.mean(np.abs(np.array(pred_energies) - ccsdt_energies))
            energy_errors.append((step, energy_mae))
            print(f"  Step {step:4d}/{args.steps}: loss={loss:.6e}  energy_MAE={energy_mae*27.2114:.4f} eV")

    elapsed = time.perf_counter() - t0
    print(f"Training completed in {elapsed:.1f}s ({elapsed/args.steps:.1f}s/step)")

    # 9. Final evaluation
    print("\n" + "=" * 60)
    print("Final evaluation:")
    pred_final = []
    for i, datum in enumerate(train_data):
        e_pred = float(predict_ground_state_total_energy(
            train_state.params, functional, datum.molecule, training_config=training_config,
        ))
        pred_final.append(e_pred)
        if i < 10:
            print(f"  {samples[i]['formula']:12s}  CCSD(T)={ccsdt_energies[i]:12.8f}  "
                  f"Pred={e_pred:12.8f}  Δ={abs(e_pred-ccsdt_energies[i])*27.2114:.4f} eV")

    final_mae = np.mean(np.abs(np.array(pred_final) - ccsdt_energies))
    print(f"\nFinal energy MAE: {final_mae*27.2114:.4f} eV  ({final_mae:.6e} Ha)")
    print(f"Initial {args.ref_xc.upper()} MAE: {np.mean(energy_diff)*27.2114:.4f} eV")

    # 10. Save results
    print(f"\nOutputs saved to: {outdir}")
    results = {
        "n_samples": len(train_data),
        "basis": args.basis,
        "ref_xc": args.ref_xc,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "hidden_dims": list(hidden_dims),
        "loss_history": [float(x) for x in loss_history],
        "energy_errors": [(int(s), float(e)) for s, e in energy_errors],
        "initial_mae_ev": float(np.mean(energy_diff) * 27.2114),
        "final_mae_ev": float(final_mae * 27.2114),
        "train_time_s": elapsed,
    }
    import json
    with open(outdir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    main()
