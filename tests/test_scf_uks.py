import numpy as np
from types import SimpleNamespace

from td_graddft.nn_rsh.functional import BoundTrainableRSHFunctional
from td_graddft.nn_rsh.schema import RSHFunctionalTemplate, ResolvedRSHParameters
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


def _toy_bound_rsh(*, local_xc_spec: str, sr: float, lr: float, omega: float = 0.3):
    template = RSHFunctionalTemplate(
        name="toy_rsh",
        local_backend="jax_libxc",
        exchange_backend_id="toy",
        correlation_backend_id="toy",
        default_sr_hf_fraction=sr,
        default_lr_hf_fraction=lr,
        default_omega=omega,
    )
    return BoundTrainableRSHFunctional(
        template=template,
        local_xc_spec=local_xc_spec,
        resolved_params=ResolvedRSHParameters(
            sr_hf_fraction=sr,
            lr_hf_fraction=lr,
            omega=omega,
        ),
        fallback_omega_values=(0.0,),
    )


def _toy_bound_xc_template():
    ao, _ao_deriv1, weights = _toy_grid()
    nao = ao.shape[1]
    ngrids = weights.shape[0]
    return SimpleNamespace(
        hfx_omega_values=(0.0,),
        hfx_nu=np.zeros((1, ngrids, nao, nao), dtype=np.float64),
    )


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


def test_uks_bound_rsh_hf_limit_matches_native_hf_path():
    ao, ao_deriv1, weights = _toy_grid()
    hcore = np.diag(np.asarray([-0.9, 0.3], dtype=np.float64))
    common_kwargs = dict(
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
            max_cycle=6,
            conv_tol=1e-12,
            conv_tol_density=1e-12,
        ),
    )
    native = run_uks_from_integrals(**common_kwargs)
    bound = run_uks_from_integrals(
        **common_kwargs,
        bound_xc=_toy_bound_rsh(local_xc_spec="hf", sr=1.0, lr=1.0),
        molecule_template=_toy_bound_xc_template(),
    )

    assert native.converged and bound.converged
    assert np.allclose(np.asarray(bound.fock_matrix_alpha), np.asarray(native.fock_matrix_alpha), atol=1e-6, rtol=1e-6)
    assert np.allclose(np.asarray(bound.fock_matrix_beta), np.asarray(native.fock_matrix_beta), atol=1e-6, rtol=1e-6)
    assert np.allclose(bound.total_energy, native.total_energy, atol=1e-6, rtol=1e-6)
    assert np.allclose(np.asarray(bound.mo_energy_alpha), np.asarray(native.mo_energy_alpha), atol=1e-6, rtol=1e-6)


def test_uks_bound_rsh_lda_x_path_is_spin_resolved():
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
            xc_spec="hf",
            max_cycle=20,
            conv_tol=1e-10,
            conv_tol_density=1e-9,
        ),
        bound_xc=_toy_bound_rsh(local_xc_spec="lda_x", sr=0.0, lr=0.0),
        molecule_template=_toy_bound_xc_template(),
    )

    assert result.converged
    assert np.isfinite(result.total_energy)
    assert not np.allclose(
        np.asarray(result.fock_matrix_alpha),
        np.asarray(result.fock_matrix_beta),
    )
