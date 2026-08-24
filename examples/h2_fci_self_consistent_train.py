#!/usr/bin/env python3
"""Train and restore a small self-consistent Neural XC model on three H2 points."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from pyscf import ao2mo, dft, fci, gto, scf

from td_graddft import neural_xc, training
from td_graddft.data.reference import restricted_reference_from_pyscf
from td_graddft.spectra import HARTREE_TO_EV
from td_graddft.xc_backend import b3lyp_component_basis


jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--outdir", default="outputs/h2_neural_xc_example")
    return parser.parse_args()


def build_h2(distance_angstrom: float, basis: str):
    return gto.M(
        atom=(
            f"H 0 0 {-0.5 * distance_angstrom:.10f}; "
            f"H 0 0 {0.5 * distance_angstrom:.10f}"
        ),
        basis=basis,
        unit="Angstrom",
        spin=0,
        charge=0,
        verbose=0,
    )


def fci_total_energy(mol) -> float:
    mean_field = scf.RHF(mol).run(conv_tol=1e-12)
    h1_mo = mean_field.mo_coeff.T @ mean_field.get_hcore() @ mean_field.mo_coeff
    eri_mo = ao2mo.kernel(mol, mean_field.mo_coeff)
    solver = fci.direct_spin0.FCI(mol)
    electronic_energy, _ = solver.kernel(
        h1_mo,
        eri_mo,
        mean_field.mo_coeff.shape[1],
        mol.nelectron,
    )
    return float(electronic_energy + mol.energy_nuc())


def build_reference(mol, grid_level: int):
    mean_field = dft.RKS(mol)
    mean_field.xc = "b3lyp"
    mean_field.grids.level = int(grid_level)
    mean_field.conv_tol = 1e-10
    mean_field.max_cycle = 120
    mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError("PySCF B3LYP reference did not converge.")
    return restricted_reference_from_pyscf(
        mean_field,
        compute_local_hfx_features=True,
        compute_local_hfx_aux=False,
    )


def main() -> None:
    args = parse_args()
    distances = (0.74, 1.40, 2.20)
    references = []
    targets = []
    for distance in distances:
        mol = build_h2(distance, args.basis)
        references.append(build_reference(mol, args.grid_level))
        targets.append(fci_total_energy(mol))

    functional = neural_xc.Functional(
        architecture="graddft_residual",
        semilocal_xc=b3lyp_component_basis(),
        hidden_dims=(64, 64),
        input_feature_mode="canonical",
        include_hfx_channel=True,
        ground_state_hf_mode="nograd",
        response_hf_mode="approx",
        name="h2_neural_xc_example",
    )
    config = training.MolecularTrainingConfig(
        mode="self_consistent",
        scf_gradient_mode="impl",
        e0_total_mse_weight=1.0,
        e0_total_mae_weight=1.0,
        scf_max_cycle=32,
        scf_convergence_metric="energy",
        scf_conv_tol_energy=1e-8,
    )
    dataset = tuple(
        training.MolecularTrainingDatum(
            molecule=reference,
            target_e0_total_h=jnp.asarray(target, dtype=jnp.float64),
        )
        for reference, target in zip(references, targets, strict=True)
    )
    state = training.create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(0),
        references[0],
        optax.adam(args.learning_rate),
    )
    train_step = training.make_molecular_train_step(
        functional,
        training_config=config,
    )
    for step in range(1, int(args.steps) + 1):
        state, metrics = train_step(state, dataset)
        print(f"step={step:4d} loss={float(metrics['total_loss']):.8e}")

    outdir = Path(args.outdir)
    checkpoint = outdir / "h2_neural_xc.msgpack"
    training.save_params_checkpoint(
        checkpoint,
        state.params,
        metadata={
            "architecture": "graddft_residual",
            "hidden_dims": [64, 64],
            "ground_state_hf_mode": "nograd",
        },
    )
    params = training.load_params_checkpoint(checkpoint, template=state.params)

    for distance, reference, target in zip(distances, references, targets, strict=True):
        predicted = training.predict_ground_state_total_energy(
            params,
            functional,
            reference,
            training_config=config,
        )
        print(
            f"R={distance:.2f} A  predicted={float(predicted):.10f} Ha  "
            f"FCI={target:.10f} Ha"
        )

    converged_molecule = training.predict_ground_state_molecule(
        params,
        functional,
        references[0],
        training_config=config,
    )
    excitation = training.predict_excitation_energies(
        params,
        functional,
        converged_molecule,
        nstates=1,
        use_tda=True,
    )
    print(f"R=0.74 A  TDA S1={float(excitation[0] * HARTREE_TO_EV):.6f} eV")
    print(f"checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
