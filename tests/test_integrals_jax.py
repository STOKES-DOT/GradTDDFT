import numpy as np
import pytest

from td_graddft.data.basis import basis_from_pyscf_mol_cart, cartesian_angular_tuples
from td_graddft.data.integrals import (
    build_hcore,
    eri_element,
    eri_tensor,
    eri_tensor_screened,
    kinetic_element,
    kinetic_matrix,
    nuclear_attraction_element,
    nuclear_attraction_matrix,
    overlap_element,
    overlap_matrix,
    rinv_matrices,
    rinv_matrix,
)
import td_graddft.data.integrals.two_electron as two_electron_module
from td_graddft.data.integrals.one_electron import _cached_rinv_chunk_builder


class _MockQuartetGroup:
    def __init__(self, size: int, signature: tuple):
        self.idx_i = np.arange(int(size), dtype=np.int32)
        self.signature = signature


def _pyscf_or_skip():
    try:
        from pyscf import gto  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("PySCF is required for integral-reference tests.")


def _pyscf_cart_eri_element(mol, i: int, j: int, k: int, l: int) -> float:
    """Reference (ij|kl) using shell-block intor, avoiding full 4-index tensors."""

    ao_loc = np.asarray(mol.ao_loc_nr(cart=True))
    shell_starts = ao_loc[:-1]
    shell_stops = ao_loc[1:]

    def locate(ao_idx: int) -> tuple[int, int]:
        sh = int(np.searchsorted(shell_stops, ao_idx, side="right"))
        local = int(ao_idx - shell_starts[sh])
        return sh, local

    shi, li = locate(i)
    shj, lj = locate(j)
    shk, lk = locate(k)
    shl, ll = locate(l)
    block = mol.intor_by_shell("int2e_cart", (shi, shj, shk, shl))
    return float(block[li, lj, lk, ll])


def test_cartesian_order_matches_pyscf_convention():
    assert cartesian_angular_tuples(0) == [(0, 0, 0)]
    assert cartesian_angular_tuples(1) == [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    assert cartesian_angular_tuples(2) == [
        (2, 0, 0),
        (1, 1, 0),
        (1, 0, 1),
        (0, 2, 0),
        (0, 1, 1),
        (0, 0, 2),
    ]


def test_one_electron_integrals_match_pyscf_cart_for_carbon():
    _pyscf_or_skip()
    from pyscf import gto

    mol = gto.M(
        atom="C 0 0 0",
        basis="sto-3g",
        cart=True,
        spin=2,
        verbose=0,
    )
    basis = basis_from_pyscf_mol_cart(mol, max_l=1)

    s_ref = mol.intor("int1e_ovlp_cart")
    t_ref = mol.intor("int1e_kin_cart")
    v_ref = mol.intor("int1e_nuc_cart")
    h_ref = t_ref + v_ref

    s = np.asarray(overlap_matrix(basis))
    t = np.asarray(kinetic_matrix(basis))
    v = np.asarray(nuclear_attraction_matrix(basis))
    h = np.asarray(build_hcore(basis))

    assert np.allclose(s, s_ref, atol=2e-7, rtol=2e-7)
    assert np.allclose(t, t_ref, atol=2e-6, rtol=2e-6)
    assert np.allclose(v, v_ref, atol=2e-6, rtol=2e-6)
    assert np.allclose(h, h_ref, atol=3e-6, rtol=3e-6)


def test_one_electron_engine_modes_agree_for_h2():
    _pyscf_or_skip()
    from pyscf import gto

    mol = gto.M(
        atom="""
        H 0.0 0.0 -0.35
        H 0.0 0.0  0.35
        """,
        basis="sto-3g",
        cart=True,
        spin=0,
        verbose=0,
    )
    basis = basis_from_pyscf_mol_cart(mol, max_l=1)

    s_legacy = np.asarray(overlap_matrix(basis, engine="legacy"))
    s_jit = np.asarray(overlap_matrix(basis, engine="jit"))
    s_auto = np.asarray(overlap_matrix(basis, engine="auto"))

    h_legacy = np.asarray(build_hcore(basis, engine="legacy"))
    h_jit = np.asarray(build_hcore(basis, engine="jit"))
    h_auto = np.asarray(build_hcore(basis, engine="auto"))

    assert np.allclose(s_jit, s_legacy, atol=1e-7, rtol=1e-7)
    assert np.allclose(s_auto, s_jit, atol=1e-10, rtol=1e-10)
    assert np.allclose(h_jit, h_legacy, atol=2e-7, rtol=2e-7)
    assert np.allclose(h_auto, h_jit, atol=1e-10, rtol=1e-10)


def test_build_hcore_uses_fused_pair_pass_for_jit(monkeypatch):
    _pyscf_or_skip()
    from pyscf import gto
    import td_graddft.data.integrals.one_electron as one_electron

    mol = gto.M(
        atom="""
        H 0.0 0.0 -0.35
        H 0.0 0.0  0.35
        """,
        basis="sto-3g",
        cart=True,
        spin=0,
        verbose=0,
    )
    basis = basis_from_pyscf_mol_cart(mol, max_l=1)

    def _fail_matrix_builder(*args, **kwargs):
        raise AssertionError("build_hcore should use the fused pair pass for JIT-capable signatures.")

    monkeypatch.setattr(one_electron, "kinetic_matrix", _fail_matrix_builder)
    monkeypatch.setattr(one_electron, "nuclear_attraction_matrix", _fail_matrix_builder)

    h = np.asarray(one_electron.build_hcore(basis, engine="jit"))
    h_ref = mol.intor("int1e_kin_cart") + mol.intor("int1e_nuc_cart")
    assert np.allclose(h, h_ref, atol=2e-7, rtol=2e-7)


def test_eri_tensor_matches_pyscf_for_h2_s_shell():
    _pyscf_or_skip()
    from pyscf import gto

    mol = gto.M(
        atom="""
        H 0.0 0.0 -0.35
        H 0.0 0.0  0.35
        """,
        basis="sto-3g",
        cart=True,
        spin=0,
        verbose=0,
    )
    basis = basis_from_pyscf_mol_cart(mol, max_l=1)

    eri_ref = mol.intor("int2e_cart")
    eri = np.asarray(eri_tensor(basis))
    assert np.allclose(eri, eri_ref, atol=2e-6, rtol=2e-6)


def test_eri_engine_modes_agree_for_h2():
    _pyscf_or_skip()
    from pyscf import gto

    mol = gto.M(
        atom="""
        H 0.0 0.0 -0.35
        H 0.0 0.0  0.35
        """,
        basis="sto-3g",
        cart=True,
        spin=0,
        verbose=0,
    )
    basis = basis_from_pyscf_mol_cart(mol, max_l=1)

    eri_legacy = np.asarray(eri_tensor_screened(basis, screening_threshold=0.0, engine="legacy"))
    eri_jit = np.asarray(eri_tensor_screened(basis, screening_threshold=0.0, engine="jit"))
    eri_auto = np.asarray(eri_tensor_screened(basis, screening_threshold=0.0, engine="auto"))

    assert np.allclose(eri_jit, eri_legacy, atol=1e-7, rtol=1e-7)
    assert np.allclose(eri_auto, eri_jit, atol=1e-10, rtol=1e-10)


def test_eri_tensor_screened_matches_unscreened_when_threshold_zero():
    _pyscf_or_skip()
    from pyscf import gto

    mol = gto.M(
        atom="""
        H 0.0 0.0 -0.35
        H 0.0 0.0  0.35
        """,
        basis="sto-3g",
        cart=True,
        spin=0,
        verbose=0,
    )
    basis = basis_from_pyscf_mol_cart(mol, max_l=1)

    eri_full = np.asarray(eri_tensor(basis))
    eri_screened = np.asarray(eri_tensor_screened(basis, screening_threshold=0.0))
    assert np.allclose(eri_screened, eri_full, atol=1e-10, rtol=1e-10)


def test_fused_shell_pair_matrix_matches_legacy_shell_block_for_water_sto3g():
    _pyscf_or_skip()
    from pyscf import gto

    mol = gto.M(
        atom="""
        O 0.000000 0.000000 0.117790
        H 0.000000 0.755453 -0.471161
        H 0.000000 -0.755453 -0.471161
        """,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        verbose=0,
    )
    basis = basis_from_pyscf_mol_cart(mol, max_l=1)

    fused = np.asarray(
        two_electron_module._fused_eri_pair_matrix_from_shell_groups(
            basis,
            basis.shell_quartet_groups,
        )
    )
    legacy = np.asarray(
        two_electron_module._legacy_eri_pair_matrix_packed_shell_block(
            basis,
            engine="jit",
        )
    )

    assert np.allclose(fused, legacy, atol=1e-10, rtol=1e-10)


def test_fused_shell_eri_tensor_matches_legacy_shell_block_for_water_sto3g():
    _pyscf_or_skip()
    from pyscf import gto

    mol = gto.M(
        atom="""
        O 0.000000 0.000000 0.117790
        H 0.000000 0.755453 -0.471161
        H 0.000000 -0.755453 -0.471161
        """,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        verbose=0,
    )
    basis = basis_from_pyscf_mol_cart(mol, max_l=1)

    fused = np.asarray(
        two_electron_module._fused_eri_tensor_from_shell_groups(
            basis,
            basis.shell_quartet_groups,
        )
    )
    legacy = np.asarray(
        two_electron_module._legacy_eri_tensor_shell_block(
            basis,
            engine="jit",
        )
    )

    assert np.allclose(fused, legacy, atol=1e-10, rtol=1e-10)


def test_fused_shell_pair_matrix_reuses_compiled_scan_executor_for_water_sto3g():
    _pyscf_or_skip()
    from pyscf import gto

    mol = gto.M(
        atom="""
        O 0.000000 0.000000 0.117790
        H 0.000000 0.755453 -0.471161
        H 0.000000 -0.755453 -0.471161
        """,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        verbose=0,
    )
    basis = basis_from_pyscf_mol_cart(mol, max_l=1)
    two_electron_module._compiled_shell_pair_scan_executor.cache_clear()
    expected_misses = len(
        two_electron_module._signature_limited_group_chunks(basis.shell_quartet_groups)
    )

    first = np.asarray(
        two_electron_module._fused_eri_pair_matrix_from_shell_groups(
            basis,
            basis.shell_quartet_groups,
        )
    )
    first_info = two_electron_module._compiled_shell_pair_scan_executor.cache_info()
    second = np.asarray(
        two_electron_module._fused_eri_pair_matrix_from_shell_groups(
            basis,
            basis.shell_quartet_groups,
        )
    )
    second_info = two_electron_module._compiled_shell_pair_scan_executor.cache_info()

    assert np.allclose(second, first, atol=0.0, rtol=0.0)
    assert first_info.misses == expected_misses
    assert second_info.misses == first_info.misses
    assert second_info.hits >= first_info.hits + expected_misses


def test_fused_shell_eri_tensor_reuses_compiled_scan_executor_for_water_sto3g():
    _pyscf_or_skip()
    from pyscf import gto

    mol = gto.M(
        atom="""
        O 0.000000 0.000000 0.117790
        H 0.000000 0.755453 -0.471161
        H 0.000000 -0.755453 -0.471161
        """,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        verbose=0,
    )
    basis = basis_from_pyscf_mol_cart(mol, max_l=1)
    two_electron_module._compiled_shell_tensor_scan_executor.cache_clear()
    expected_misses = len(
        two_electron_module._signature_limited_group_chunks(basis.shell_quartet_groups)
    )

    first = np.asarray(
        two_electron_module._fused_eri_tensor_from_shell_groups(
            basis,
            basis.shell_quartet_groups,
        )
    )
    first_info = two_electron_module._compiled_shell_tensor_scan_executor.cache_info()
    second = np.asarray(
        two_electron_module._fused_eri_tensor_from_shell_groups(
            basis,
            basis.shell_quartet_groups,
        )
    )
    second_info = two_electron_module._compiled_shell_tensor_scan_executor.cache_info()

    assert np.allclose(second, first, atol=0.0, rtol=0.0)
    assert first_info.misses == expected_misses
    assert second_info.misses == first_info.misses
    assert second_info.hits >= first_info.hits + expected_misses


def test_fused_ao_pair_matrix_matches_pair_matrix_for_water_sto3g():
    _pyscf_or_skip()
    from pyscf import gto

    mol = gto.M(
        atom="""
        O 0.000000 0.000000 0.117790
        H 0.000000 0.755453 -0.471161
        H 0.000000 -0.755453 -0.471161
        """,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        verbose=0,
    )
    basis = basis_from_pyscf_mol_cart(mol, max_l=1)

    fused = np.asarray(
        two_electron_module._fused_eri_pair_matrix_from_ao_groups(
            basis,
            basis.quartet_groups,
            engine="jit",
        )
    )
    expected = np.asarray(two_electron_module.eri_pair_matrix_packed(basis, engine="jit"))

    assert np.allclose(fused, expected, atol=1e-10, rtol=1e-10)


def test_fused_ao_pair_matrix_reuses_compiled_scan_executor_for_water_sto3g():
    _pyscf_or_skip()
    from pyscf import gto

    mol = gto.M(
        atom="""
        O 0.000000 0.000000 0.117790
        H 0.000000 0.755453 -0.471161
        H 0.000000 -0.755453 -0.471161
        """,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        verbose=0,
    )
    basis = basis_from_pyscf_mol_cart(mol, max_l=1)
    two_electron_module._compiled_ao_pair_scan_executor.cache_clear()
    expected_misses = len(two_electron_module._signature_limited_group_chunks(basis.quartet_groups))

    first = np.asarray(
        two_electron_module._fused_eri_pair_matrix_from_ao_groups(
            basis,
            basis.quartet_groups,
            engine="jit",
        )
    )
    first_info = two_electron_module._compiled_ao_pair_scan_executor.cache_info()
    second = np.asarray(
        two_electron_module._fused_eri_pair_matrix_from_ao_groups(
            basis,
            basis.quartet_groups,
            engine="jit",
        )
    )
    second_info = two_electron_module._compiled_ao_pair_scan_executor.cache_info()

    assert np.allclose(second, first, atol=0.0, rtol=0.0)
    assert first_info.misses == expected_misses
    assert second_info.misses == first_info.misses
    assert second_info.hits >= first_info.hits + expected_misses


def test_signature_limited_group_chunks_limit_padding_ratio():
    groups = (
        _MockQuartetGroup(100, ("a",)),
        _MockQuartetGroup(1, ("b",)),
        _MockQuartetGroup(1, ("c",)),
        _MockQuartetGroup(80, ("d",)),
        _MockQuartetGroup(70, ("e",)),
        _MockQuartetGroup(2, ("f",)),
    )

    chunks = two_electron_module._signature_limited_group_chunks(
        groups,
        max_signatures=64,
        max_padding_ratio=2.0,
    )
    flattened = [group for chunk in chunks for group in chunk]

    assert sorted(int(group.idx_i.shape[0]) for group in flattened) == [1, 1, 2, 70, 80, 100]
    assert len(chunks) > 1
    for chunk in chunks:
        sizes = [int(group.idx_i.shape[0]) for group in chunk]
        padded = max(sizes) * len(sizes)
        useful = sum(sizes)
        assert padded / useful <= 2.0


def test_eri_tensor_screened_can_skip_all_with_large_threshold():
    _pyscf_or_skip()
    from pyscf import gto

    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.4",
        basis="sto-3g",
        cart=True,
        spin=0,
        verbose=0,
    )
    basis = basis_from_pyscf_mol_cart(mol, max_l=1)

    eri_screened = np.asarray(eri_tensor_screened(basis, screening_threshold=1e9))
    assert np.max(np.abs(eri_screened)) < 1e-14


def test_eri_elements_match_pyscf_for_p_orbital_cases():
    _pyscf_or_skip()
    from pyscf import gto

    mol = gto.M(
        atom="C 0 0 0",
        basis="sto-3g",
        cart=True,
        spin=2,
        verbose=0,
    )
    basis = basis_from_pyscf_mol_cart(mol, max_l=1)
    # Carbon STO-3G cart AO order:
    # 0:1s, 1:2s, 2:px, 3:py, 4:pz
    cases = [
        (0, 0, 0, 0),
        (2, 0, 0, 0),
        (2, 3, 0, 0),
        (2, 0, 3, 0),
        (2, 3, 2, 3),
        (4, 1, 4, 1),
    ]
    for i, j, k, l in cases:
        val = float(eri_element(basis, i, j, k, l))
        ref = _pyscf_cart_eri_element(mol, i, j, k, l)
        assert np.isclose(val, ref, atol=3e-6, rtol=3e-6)


def test_basis_parser_rejects_larger_than_supported_l():
    _pyscf_or_skip()
    from pyscf import gto

    mol = gto.M(
        atom="O 0 0 0",
        basis="cc-pvtz",
        cart=True,
        spin=2,
        verbose=0,
    )
    with pytest.raises(NotImplementedError):
        basis_from_pyscf_mol_cart(mol, max_l=1)


def test_one_electron_integrals_match_pyscf_for_d_functions():
    _pyscf_or_skip()
    from pyscf import gto

    mol = gto.M(
        atom="O 0 0 0",
        basis="6-31g*",
        cart=True,
        spin=2,
        verbose=0,
    )
    basis = basis_from_pyscf_mol_cart(mol, max_l=3)
    s_ref = mol.intor("int1e_ovlp_cart")
    t_ref = mol.intor("int1e_kin_cart")
    v_ref = mol.intor("int1e_nuc_cart")

    s = np.asarray(overlap_matrix(basis))
    t = np.asarray(kinetic_matrix(basis))
    v = np.asarray(nuclear_attraction_matrix(basis))
    assert np.allclose(s, s_ref, atol=4e-6, rtol=4e-6)
    assert np.allclose(t, t_ref, atol=6e-6, rtol=6e-6)
    assert np.allclose(v, v_ref, atol=7e-6, rtol=7e-6)


def test_selected_integrals_match_pyscf_for_f_functions():
    _pyscf_or_skip()
    from pyscf import gto

    mol = gto.M(
        atom="O 0 0 0",
        basis="cc-pvtz",
        cart=True,
        spin=2,
        verbose=0,
    )
    basis = basis_from_pyscf_mol_cart(mol, max_l=3)
    angular_l = [sum(ao.angular) for ao in basis.aos]
    f_idx = [i for i, l in enumerate(angular_l) if l == 3]
    d_idx = [i for i, l in enumerate(angular_l) if l == 2]
    assert len(f_idx) > 0
    assert len(d_idx) > 0

    s_ref = mol.intor("int1e_ovlp_cart")
    t_ref = mol.intor("int1e_kin_cart")
    v_ref = mol.intor("int1e_nuc_cart")
    cases_1e = [
        (f_idx[0], f_idx[0]),
        (f_idx[1], d_idx[0]),
        (f_idx[-1], 0),
    ]
    for i, j in cases_1e:
        s = float(overlap_element(basis, i, j))
        t = float(kinetic_element(basis, i, j))
        v = float(nuclear_attraction_element(basis, i, j))
        assert np.isclose(s, float(s_ref[i, j]), atol=7e-6, rtol=7e-6)
        assert np.isclose(t, float(t_ref[i, j]), atol=8e-6, rtol=8e-6)
        assert np.isclose(v, float(v_ref[i, j]), atol=9e-6, rtol=9e-6)


def test_selected_eri_match_pyscf_for_d_functions():
    _pyscf_or_skip()
    from pyscf import gto

    mol = gto.M(
        atom="O 0 0 0",
        basis="6-31g*",
        cart=True,
        spin=2,
        verbose=0,
    )
    basis = basis_from_pyscf_mol_cart(mol, max_l=3)
    angular_l = [sum(ao.angular) for ao in basis.aos]
    d_idx = [i for i, l in enumerate(angular_l) if l == 2]
    p_idx = [i for i, l in enumerate(angular_l) if l == 1]
    assert len(d_idx) > 1
    assert len(p_idx) > 0

    cases = [
        (d_idx[0], 0, 0, 0),
        (d_idx[0], d_idx[1], p_idx[0], p_idx[1]),
        (d_idx[0], d_idx[1], d_idx[0], d_idx[1]),
    ]
    for i, j, k, l in cases:
        val = float(eri_element(basis, i, j, k, l))
        ref = _pyscf_cart_eri_element(mol, i, j, k, l)
        assert np.isclose(val, ref, atol=8e-6, rtol=8e-6)


@pytest.mark.parametrize("basis_name", ["sto-3g", "def2-svp", "cc-pvdz"])
def test_rinv_integrals_match_pyscf_for_water_multiple_basis(basis_name: str):
    _pyscf_or_skip()
    from pyscf import gto

    atom = """
    O  0.000000  0.000000  0.117790
    H  0.000000  0.755453 -0.471161
    H  0.000000 -0.755453 -0.471161
    """
    mol = gto.M(
        atom=atom,
        basis=basis_name,
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        verbose=0,
    )
    basis = basis_from_pyscf_mol_cart(mol, max_l=3)
    origin = np.asarray([0.18, -0.27, 0.11], dtype=float)

    with mol.with_rinv_origin(origin):
        point_ref = np.asarray(mol.intor_symmetric("int1e_rinv"), dtype=float)
    point = np.asarray(rinv_matrix(basis, origin=origin), dtype=float)
    assert np.allclose(point, point_ref, atol=3e-5, rtol=3e-5)

    zeta = 0.4 * 0.4
    with mol.with_rinv_zeta(zeta=zeta):
        with mol.with_rinv_origin(origin):
            range_ref = np.asarray(mol.intor_symmetric("int1e_rinv"), dtype=float)
    range_jax = np.asarray(rinv_matrix(basis, origin=origin, zeta=zeta), dtype=float)
    assert np.allclose(range_jax, range_ref, atol=3e-5, rtol=3e-5)


def test_rinv_matrices_match_pyscf_int1e_grids_for_water():
    _pyscf_or_skip()
    from pyscf import gto

    atom = """
    O  0.000000  0.000000  0.117790
    H  0.000000  0.755453 -0.471161
    H  0.000000 -0.755453 -0.471161
    """
    mol = gto.M(
        atom=atom,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        verbose=0,
    )
    basis = basis_from_pyscf_mol_cart(mol, max_l=1)
    coords = np.asarray(
        [
            [0.00, 0.00, 0.10],
            [0.20, -0.15, -0.35],
            [-0.25, 0.30, 0.05],
        ],
        dtype=float,
    )

    point_ref = np.asarray(mol.intor("int1e_grids_cart", hermi=1, grids=coords), dtype=float)
    point = np.asarray(rinv_matrices(basis, coords, grid_chunk_size=2), dtype=float)
    assert np.allclose(point, point_ref, atol=2e-5, rtol=2e-5)

    omega = 0.4
    with mol.with_range_coulomb(omega=omega):
        range_ref = np.asarray(mol.intor("int1e_grids_cart", hermi=1, grids=coords), dtype=float)
    range_jax = np.asarray(
        rinv_matrices(basis, coords, zeta=omega * omega, grid_chunk_size=2),
        dtype=float,
    )
    assert np.allclose(range_jax, range_ref, atol=2e-5, rtol=2e-5)


def test_rinv_chunk_builder_cache_reuses_callable_for_same_basis():
    _pyscf_or_skip()
    from pyscf import gto

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        verbose=0,
    )
    basis_a = basis_from_pyscf_mol_cart(mol, max_l=1)
    basis_b = basis_from_pyscf_mol_cart(mol, max_l=1)

    builder_a1 = _cached_rinv_chunk_builder(basis_a, engine="auto")
    builder_a2 = _cached_rinv_chunk_builder(basis_a, engine="auto")
    builder_b = _cached_rinv_chunk_builder(basis_b, engine="auto")
    builder_legacy = _cached_rinv_chunk_builder(basis_a, engine="legacy")

    assert builder_a1 is builder_a2
    assert builder_a1 is not builder_b
    assert builder_a1 is not builder_legacy
