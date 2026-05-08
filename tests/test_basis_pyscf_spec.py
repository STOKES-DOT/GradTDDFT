import numpy as np
import pytest

from td_graddft.data import basis_from_pyscf_spec, basis_from_spec
from td_graddft.data.basis import _normalize_raw_shell_coefficients
from td_graddft.data.pyscf_basis_loader import load_basis_from_snapshot
from td_graddft.data.integrals import build_hcore, eri_tensor, overlap_matrix


def _pyscf_or_skip():
    try:
        from pyscf import gto  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("PySCF is required for basis-library comparison tests.")


_WATER_BASIS_SET_NAMES = ["sto-3g", "6-31g", "6-31g*", "def2-svp", "cc-pvdz"]


@pytest.mark.parametrize("basis_name", _WATER_BASIS_SET_NAMES)
def test_basis_from_pyscf_spec_matches_pyscf_library_on_water(basis_name: str):
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
    basis = basis_from_pyscf_spec(
        atom,
        basis=basis_name,
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=3,
    )

    s_ref = np.asarray(mol.intor_symmetric("int1e_ovlp"), dtype=float)
    h_ref = np.asarray(mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc"), dtype=float)
    eri_ref = np.asarray(mol.intor("int2e"), dtype=float)

    s = np.asarray(overlap_matrix(basis), dtype=float)
    h = np.asarray(build_hcore(basis), dtype=float)
    eri = np.asarray(eri_tensor(basis), dtype=float)

    assert basis.nao == int(mol.nao_nr())
    assert np.allclose(s, s_ref, atol=2e-6, rtol=2e-6)
    assert np.allclose(h, h_ref, atol=2e-5, rtol=2e-5)
    assert np.allclose(eri, eri_ref, atol=2e-4, rtol=2e-4)


def test_vendored_snapshot_loader_matches_pyscf_raw_content():
    _pyscf_or_skip()
    from pyscf import gto

    for basis_name in _WATER_BASIS_SET_NAMES:
        for symbol in ("H", "C", "N", "O"):
            expected = gto.basis.load(basis_name, symbol)
            actual = load_basis_from_snapshot(basis_name, symbol)
            assert actual == expected


@pytest.mark.parametrize("basis_name", ["sto-3g", "def2-svp", "cc-pvdz"])
@pytest.mark.parametrize("symbol", ["H", "O"])
def test_normalized_snapshot_shells_match_pyscf_bas_ctr_coeff(
    basis_name: str,
    symbol: str,
):
    _pyscf_or_skip()
    from pyscf import gto

    mol = gto.M(
        atom=f"{symbol} 0.0 0.0 0.0",
        basis=basis_name,
        unit="Angstrom",
        cart=True,
        spin=1 if symbol == "H" else 0,
        charge=0,
        verbose=0,
    )
    raw_shells = load_basis_from_snapshot(basis_name, symbol)
    assert len(raw_shells) == mol.nbas

    for shell_idx, shell in enumerate(raw_shells):
        l = int(shell[0])
        exponents, coeff = _normalize_raw_shell_coefficients(l, shell[1:])
        ref_exponents = np.asarray(mol.bas_exp(shell_idx), dtype=float)
        ref_coeff = np.asarray(mol.bas_ctr_coeff(shell_idx), dtype=float)
        assert np.allclose(exponents, ref_exponents, atol=1e-14, rtol=1e-14)
        assert np.allclose(coeff, ref_coeff, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("basis_name", _WATER_BASIS_SET_NAMES)
def test_basis_from_spec_matches_pyscf_library_on_water(basis_name: str):
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
    basis = basis_from_spec(
        atom,
        basis=basis_name,
        unit="Angstrom",
        spin=0,
        charge=0,
        max_l=3,
    )

    s_ref = np.asarray(mol.intor_symmetric("int1e_ovlp"), dtype=float)
    h_ref = np.asarray(mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc"), dtype=float)
    eri_ref = np.asarray(mol.intor("int2e"), dtype=float)

    s = np.asarray(overlap_matrix(basis), dtype=float)
    h = np.asarray(build_hcore(basis), dtype=float)
    eri = np.asarray(eri_tensor(basis), dtype=float)

    assert basis.nao == int(mol.nao_nr())
    assert np.allclose(s, s_ref, atol=2e-6, rtol=2e-6)
    assert np.allclose(h, h_ref, atol=2e-5, rtol=2e-5)
    assert np.allclose(eri, eri_ref, atol=2e-4, rtol=2e-4)
