import numpy as np
import pytest
import jax.numpy as jnp

from td_graddft.scf import UKSConfig, run_uks_from_integrals
from td_graddft.scf.inputs import build_uks_integral_inputs
from td_graddft.scf.uks import run_unrestricted_scf_scan


def _toy_grid():
    ao = np.asarray(
        [
            [1.0, 0.2],
            [0.8, -0.1],
            [0.4, 0.9],
        ],
        dtype=np.float64,
    )
    ao_deriv1 = np.asarray(
        [
            ao,
            [
                [0.10, 0.00],
                [0.00, 0.20],
                [-0.10, 0.05],
            ],
            [
                [0.00, 0.10],
                [0.20, 0.00],
                [0.05, -0.10],
            ],
            [
                [-0.05, 0.00],
                [0.00, -0.05],
                [0.10, 0.10],
            ],
        ],
        dtype=np.float64,
    )
    weights = np.asarray([0.5, 0.7, 0.6], dtype=np.float64)
    return ao, ao_deriv1, weights


def test_unrestricted_scan_honors_energy_convergence_metric():
    fock = jnp.diag(jnp.asarray([-1.0, 0.5], dtype=jnp.float64))
    fock_spin = jnp.stack([fock, fock], axis=0)

    def fock_builder(_density_spin, _mo_coeff_spin, _mo_energy_spin):
        return fock_spin, fock_spin, jnp.asarray(0.0, dtype=jnp.float64)

    (
        density,
        _mo_coeff,
        _mo_energy,
        _raw_fock,
        converged,
        cycles,
        _rms_history,
        _selected_cycle,
        _best_cycle,
        _selected_rms,
        _best_rms,
    ) = run_unrestricted_scf_scan(
        fock_builder=fock_builder,
        density_spin=jnp.zeros((2, 2, 2), dtype=jnp.float64),
        mo_coeff_spin=jnp.stack([jnp.eye(2, dtype=jnp.float64), jnp.eye(2, dtype=jnp.float64)]),
        mo_occ_spin=jnp.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.float64),
        mo_energy_spin=jnp.asarray([[-1.0, 0.5], [-1.0, 0.5]], dtype=jnp.float64),
        overlap=jnp.eye(2, dtype=jnp.float64),
        max_cycle=4,
        damping=0.0,
        conv_tol=1e-8,
        conv_tol_density=0.0,
        orthogonalization_eps=1e-10,
        convergence_metric="energy",
    )

    assert bool(converged)
    assert int(cycles) == 2
    assert not np.allclose(np.asarray(density), 0.0)


def test_uks_respects_explicit_fixed_spin_occupations():
    ao, ao_deriv1, weights = _toy_grid()
    hcore = np.diag(np.asarray([-1.0, 0.5], dtype=np.float64))
    result = run_uks_from_integrals(
        overlap=np.eye(2, dtype=np.float64),
        hcore=hcore,
        eri=np.zeros((2, 2, 2, 2), dtype=np.float64),
        nalpha=1,
        nbeta=0,
        nuclear_repulsion=0.0,
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid_weights=weights,
        init_mo_occ_alpha=np.asarray([0.0, 1.0], dtype=np.float64),
        init_mo_occ_beta=np.zeros((2,), dtype=np.float64),
        config=UKSConfig(
            xc_spec="hf",
            max_cycle=4,
            conv_tol=1e-12,
            conv_tol_density=1e-12,
        ),
    )

    assert result.converged
    assert np.allclose(np.asarray(result.mo_occ_alpha), np.asarray([0.0, 1.0]))
    assert np.allclose(np.asarray(result.density_matrix_alpha), np.diag([0.0, 1.0]))
    assert np.allclose(np.asarray(result.density_matrix_beta), np.zeros((2, 2)))


def test_uks_semilocal_xc_is_spin_resolved():
    pytest.importorskip("jax_xc")
    ao, ao_deriv1, weights = _toy_grid()
    result = run_uks_from_integrals(
        overlap=np.eye(2, dtype=np.float64),
        hcore=np.diag(np.asarray([-0.6, 0.2], dtype=np.float64)),
        eri=np.zeros((2, 2, 2, 2), dtype=np.float64),
        nalpha=1,
        nbeta=0,
        nuclear_repulsion=0.0,
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid_weights=weights,
        config=UKSConfig(
            xc_spec="lda_x",
            max_cycle=20,
            conv_tol=1e-10,
            conv_tol_density=1e-9,
        ),
    )

    assert result.converged
    assert np.isfinite(result.total_energy)
    assert not np.allclose(
        np.asarray(result.fock_matrix_alpha),
        np.asarray(result.fock_matrix_beta),
    )


def test_uks_level_shift_does_not_pollute_returned_raw_fock_or_mo_energies():
    ao, ao_deriv1, weights = _toy_grid()
    hcore = np.diag(np.asarray([-0.8, 0.2], dtype=np.float64))
    result = run_uks_from_integrals(
        overlap=np.eye(2, dtype=np.float64),
        hcore=hcore,
        eri=np.zeros((2, 2, 2, 2), dtype=np.float64),
        nalpha=1,
        nbeta=0,
        nuclear_repulsion=0.0,
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid_weights=weights,
        config=UKSConfig(
            xc_spec="hf",
            max_cycle=4,
            conv_tol=1e-12,
            conv_tol_density=1e-12,
            level_shift=0.7,
        ),
    )

    assert result.converged
    assert np.allclose(np.asarray(result.fock_matrix_alpha), hcore, atol=1e-6, rtol=1e-6)
    assert np.allclose(np.asarray(result.fock_matrix_beta), hcore, atol=1e-6, rtol=1e-6)
    assert np.allclose(np.asarray(result.mo_energy_alpha), np.asarray([-0.8, 0.2]), atol=1e-6, rtol=1e-6)
    assert np.allclose(np.asarray(result.mo_energy_beta), np.asarray([-0.8, 0.2]), atol=1e-6, rtol=1e-6)


def test_h2plus_uks_b3lyp_matches_pyscf_reference_energy():
    pytest.importorskip("jax_xc")
    pyscf = pytest.importorskip("pyscf")
    del pyscf
    from pyscf import dft, gto

    atom = "H 0 0 -0.9; H 0 0 0.9"
    mol = gto.M(atom=atom, unit="Angstrom", basis="def2-svp", charge=1, spin=1, cart=True, verbose=0)
    mf = dft.UKS(mol)
    mf.xc = "b3lyp"
    mf.grids.level = 2
    mf.conv_tol = 1e-10
    mf.max_cycle = 128
    reference_energy = float(mf.kernel())
    assert mf.converged

    cfg = UKSConfig(
        xc_spec="b3lyp",
        max_cycle=128,
        conv_tol=1e-10,
        conv_tol_density=1e-8,
        damping=0.15,
        potential_clip=20.0,
    )
    inputs = build_uks_integral_inputs(
        atom=atom,
        basis="def2-svp",
        xc_spec="b3lyp",
        unit="Angstrom",
        charge=1,
        spin=1,
        cart=True,
        grids_level=2,
        max_l=3,
        config=cfg,
        grid_ao_backend="jax",
        integral_backend="cpu",
    )
    result = run_uks_from_integrals(**inputs.as_uks_kwargs(), config=cfg)

    assert result.converged
    assert abs(float(result.total_energy) - reference_energy) < 1e-3
