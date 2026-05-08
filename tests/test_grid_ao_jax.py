import numpy as np
import pytest

from td_graddft.data import basis_from_pyscf_mol_cart, evaluate_cartesian_ao


def _pyscf_or_skip():
    try:
        from pyscf import dft, gto  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("PySCF is required for AO-grid comparison tests.")


def _make_water_cart():
    from pyscf import gto

    mol = gto.Mole()
    mol.atom = """
    O  0.000000  0.000000  0.117790
    H  0.000000  0.755453 -0.471161
    H  0.000000 -0.755453 -0.471161
    """
    mol.unit = "Angstrom"
    mol.basis = "sto-3g"
    mol.cart = True
    mol.build()
    return mol


def test_evaluate_cartesian_ao_matches_pyscf_values_and_first_derivatives():
    _pyscf_or_skip()
    from pyscf.dft import numint

    mol = _make_water_cart()
    basis = basis_from_pyscf_mol_cart(mol, max_l=3)
    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.2, -0.1, 0.3],
            [-0.4, 0.5, -0.2],
            [0.8, -0.7, 0.1],
        ],
        dtype=float,
    )

    ao_ref = np.asarray(numint.eval_ao(mol, coords, deriv=0), dtype=float)
    ao1_ref = np.asarray(numint.eval_ao(mol, coords, deriv=1), dtype=float)

    ao_jax = np.asarray(evaluate_cartesian_ao(basis, coords, deriv=0), dtype=float)
    ao1_jax = np.asarray(evaluate_cartesian_ao(basis, coords, deriv=1), dtype=float)

    assert ao_jax.shape == ao_ref.shape
    assert ao1_jax.shape == ao1_ref.shape
    assert np.allclose(ao_jax, ao_ref, atol=2e-7, rtol=2e-6)
    assert np.allclose(ao1_jax, ao1_ref, atol=3e-6, rtol=2e-5)
