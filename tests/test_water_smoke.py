import jax
import jax.numpy as jnp
import optax

pytestmark = []

from td_graddft import neural_xc
from pyscf_reference import restricted_reference_from_pyscf
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
    return restricted_reference_from_pyscf(
        mf,
        compute_local_hfx_features=True,
        compute_local_hfx_aux=True,
        hfx_omega_values=(0.0, 0.4),
    )


def _make_trainable_functional():
    return neural_xc.Functional(
        architecture="residual",
        semilocal_xc=("gga_x_pbe", "gga_c_pbe"),
        hidden_dims=(8, 8),
        include_pt2_channel=False,
        name="water_smoke_xc",
    )


def test_water_ground_state_training_and_tddft_smoke():
    molecule = _make_water_reference()
    functional = _make_trainable_functional()

    target_energy = jnp.array(molecule.mf_energy)
    datum = GroundStateDatum.from_reference(
        molecule,
        target_total_energy=target_energy,
        require_hfx=True,
    )

    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(0),
        molecule,
        optax.adam(1e-3),
    )
    train_step = make_ground_state_train_step(functional)

    initial_energy = predict_ground_state_total_energy(state.params, functional, molecule)
    for _ in range(5):
        state, metrics = train_step(state, datum)

    predicted_energy = predict_ground_state_total_energy(state.params, functional, molecule)
    excitations = predict_excitation_energies(
        state.params,
        functional,
        molecule,
        nstates=3,
    )

    assert jnp.isfinite(initial_energy)
    assert jnp.isfinite(predicted_energy)
    assert jnp.isfinite(metrics["loss"])
    assert excitations.shape == (3,)
    assert jnp.all(jnp.isfinite(excitations))
    assert jnp.all(excitations > 0.0)
