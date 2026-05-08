import numpy as np
import pytest

from td_graddft.data.basis import basis_from_spec
from td_graddft.data.integrals import eri_pair_matrix_packed
from td_graddft.scf.packed_eri import build_jk_from_eri_pair_matrix
from td_graddft_tools.gpu_s_shell_direct_jk import (
    cpu_s_shell_direct_jk,
    cpu_sp_direct_jk_from_basis,
    extract_cartesian_ao_system,
    extract_s_shell_system,
    extract_sp_ao_system,
)


def test_cpu_s_shell_direct_jk_matches_packed_eri_for_h2_sto3g():
    basis = basis_from_spec("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    density = np.asarray(
        [
            [0.83, 0.21],
            [0.21, 0.71],
        ],
        dtype=np.float64,
    )

    pair = eri_pair_matrix_packed(basis)
    j_ref, k_ref = build_jk_from_eri_pair_matrix(pair, density)
    system = extract_s_shell_system(basis)
    j_s, k_s = cpu_s_shell_direct_jk(system, density)

    assert np.allclose(j_s, np.asarray(j_ref), atol=1e-10, rtol=1e-10)
    assert np.allclose(k_s, np.asarray(k_ref), atol=1e-10, rtol=1e-10)


def test_extract_s_shell_system_rejects_non_s_shells():
    basis = basis_from_spec(
        "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
    )

    with pytest.raises(NotImplementedError, match="s-shell"):
        extract_s_shell_system(basis)


def test_extract_sp_ao_system_accepts_water_sto3g():
    basis = basis_from_spec(
        "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
    )

    system = extract_sp_ao_system(basis)

    assert system.nao == 7
    assert system.max_nprim == 3
    assert np.count_nonzero(np.sum(system.angulars, axis=1) == 1) == 3


def test_cpu_sp_direct_jk_matches_packed_eri_for_water_sto3g():
    basis = basis_from_spec(
        "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
    )
    density = np.asarray(
        [
            [0.84, 0.04, -0.03, 0.02, 0.01, 0.06, 0.05],
            [0.04, 0.71, 0.08, -0.02, 0.03, 0.04, -0.01],
            [-0.03, 0.08, 0.66, 0.05, -0.04, 0.03, 0.02],
            [0.02, -0.02, 0.05, 0.59, 0.06, -0.02, 0.01],
            [0.01, 0.03, -0.04, 0.06, 0.62, 0.02, -0.03],
            [0.06, 0.04, 0.03, -0.02, 0.02, 0.48, 0.07],
            [0.05, -0.01, 0.02, 0.01, -0.03, 0.07, 0.51],
        ],
        dtype=np.float64,
    )

    pair = eri_pair_matrix_packed(basis)
    j_ref, k_ref = build_jk_from_eri_pair_matrix(pair, density)
    j_sp, k_sp = cpu_sp_direct_jk_from_basis(basis, density)

    assert np.allclose(j_sp, np.asarray(j_ref), atol=2e-10, rtol=2e-10)
    assert np.allclose(k_sp, np.asarray(k_ref), atol=2e-10, rtol=2e-10)


def test_extract_cartesian_ao_system_can_allow_d_shells():
    basis = basis_from_spec(
        "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="6-31g*",
        max_l=2,
    )

    with pytest.raises(NotImplementedError, match="l=1"):
        extract_sp_ao_system(basis)

    system = extract_cartesian_ao_system(basis, max_l=2)

    assert system.nao > 7
    assert np.max(np.sum(system.angulars, axis=1)) == 2
