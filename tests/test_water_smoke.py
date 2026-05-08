import jax
import jax.numpy as jnp
import optax

pytestmark = []

from td_graddft.neural_xc import DensityNeuralXCFunctional, PointwiseMLP
from td_graddft.pyscf_bridge import restricted_reference_from_pyscf
from td_graddft.training import (
    GroundStateDatum,
    create_train_state_from_molecule,
    make_ground_state_train_step,
    predict_excitation_energies,
    predict_ground_state_total_energy,
)


def _make_water_reference():
    from pyscf import dft, gto

    mol = gto.Mole()
    mol.atom = """
    O  0.000000  0.000000  0.117790
    H  0.000000  0.755453 -0.471161
    H  0.000000 -0.755453 -0.471161
    """
    mol.unit = "Angstrom"
    mol.basis = "sto-3g"
    mol.spin = 0
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = "b3lyp"
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.kernel()
    return restricted_reference_from_pyscf(mf)


def _make_trainable_functional():
    return DensityNeuralXCFunctional(
        model=PointwiseMLP(hidden_dims=(), output_dim=1, activation=lambda x: x),
        coefficient_input_fn=lambda density, density_floor=1e-12: jnp.ones(
            density.shape + (1,)
        ),
        energy_density_basis_fn=lambda density, density_floor=1e-12: density[..., None],
        name="water_smoke_xc",
        hybrid_fraction_init=0.20,
    )


def test_water_ground_state_training_and_tddft_smoke():
    molecule = _make_water_reference()
    functional = _make_trainable_functional()

    target_energy = jnp.array(molecule.mf_energy)
    datum = GroundStateDatum(molecule=molecule, target_total_energy=target_energy)

    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(0),
        molecule,
        optax.adam(0.05),
    )
    train_step = make_ground_state_train_step(functional)

    for _ in range(200):
        state, _ = train_step(state, datum)

    predicted_energy = predict_ground_state_total_energy(state.params, functional, molecule)
    learned_hybrid_fraction = functional.hybrid_fraction(state.params)
    excitations = predict_excitation_energies(
        state.params,
        functional,
        molecule,
        nstates=3,
    )

    assert jnp.allclose(predicted_energy, target_energy, atol=5e-2)
    assert bool(
        jnp.logical_and(learned_hybrid_fraction >= 0.0, learned_hybrid_fraction <= 1.0)
    )
    assert excitations.shape == (3,)
    assert jnp.all(excitations > 0.0)
