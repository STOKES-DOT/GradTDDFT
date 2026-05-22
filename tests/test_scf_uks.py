import numpy as np

from td_graddft.scf import UKSConfig, run_uks_from_integrals


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
