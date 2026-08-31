from __future__ import annotations

import numpy as np
import pytest
from pyscf import dft, gto

from td_graddft import tdscf
from td_graddft.scf import RKSConfig, UKSConfig
from td_graddft.scf.builders import (
    restricted_molecule_from_spec_with_jax_rks,
    unrestricted_molecule_from_spec_with_jax_uks,
)


def _matched_scf_controls(mf):
    conv_tol = float(mf.conv_tol)
    return {
        "max_cycle": int(mf.max_cycle),
        "conv_tol": conv_tol,
        "conv_tol_density": float(mf.conv_tol_grad or np.sqrt(conv_tol)),
    }


def _build_restricted_pair():
    atom = "H 0 0 -0.37; H 0 0 0.37"
    mol = gto.M(
        atom=atom,
        basis="def2-svp",
        unit="Angstrom",
        cart=True,
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = "b3lyp"
    mf.grids.level = 2
    mf.kernel()
    assert mf.converged
    molecule = restricted_molecule_from_spec_with_jax_rks(
        atom=atom,
        basis="def2-svp",
        xc_spec="b3lyp",
        unit="Angstrom",
        cart=True,
        grids_level=2,
        rks_config=RKSConfig(xc_spec="b3lyp", **_matched_scf_controls(mf)),
        integral_backend="cpu",
        verbose=0,
    )
    np.testing.assert_allclose(molecule.mf_energy, mf.e_tot, atol=1e-9, rtol=0.0)
    return mf, molecule


def _build_unrestricted_pair():
    atom = "H 0 0 -0.5111; H 0 0 0.5111"
    mol = gto.M(
        atom=atom,
        basis="def2-svp",
        unit="Angstrom",
        charge=1,
        spin=1,
        cart=True,
        verbose=0,
    )
    mf = dft.UKS(mol)
    mf.xc = "b3lyp"
    mf.grids.level = 2
    mf.kernel()
    assert mf.converged
    molecule = unrestricted_molecule_from_spec_with_jax_uks(
        atom=atom,
        basis="def2-svp",
        xc_spec="b3lyp",
        unit="Angstrom",
        charge=1,
        spin=1,
        cart=True,
        grids_level=2,
        uks_config=UKSConfig(xc_spec="b3lyp", **_matched_scf_controls(mf)),
        integral_backend="cpu",
        verbose=0,
    )
    np.testing.assert_allclose(molecule.mf_energy, mf.e_tot, atol=1e-9, rtol=0.0)
    return mf, molecule


@pytest.fixture(scope="module")
def restricted_pair():
    return _build_restricted_pair()


@pytest.fixture(scope="module")
def unrestricted_pair():
    return _build_unrestricted_pair()


def _assert_degenerate_cluster_strengths(reference_e, reference_f, predicted_f):
    start = 0
    while start < len(reference_e):
        stop = start + 1
        while stop < len(reference_e) and abs(reference_e[stop] - reference_e[start]) <= 1e-5:
            stop += 1
        np.testing.assert_allclose(
            np.sum(predicted_f[start:stop]),
            np.sum(reference_f[start:stop]),
            atol=3e-5,
            rtol=3e-4,
        )
        start = stop


@pytest.mark.parametrize("method", ["tda", "tddft"])
@pytest.mark.parametrize("pair_fixture", ["restricted_pair", "unrestricted_pair"])
def test_grid_aligned_end_to_end_excited_states_match_pyscf(
    method,
    pair_fixture,
    request,
):
    mf, molecule = request.getfixturevalue(pair_fixture)
    nstates = 3
    pyscf_solver = mf.TDA() if method == "tda" else mf.TDDFT()
    pyscf_solver.nstates = nstates
    pyscf_solver.kernel()
    assert np.all(np.asarray(pyscf_solver.converged))

    driver_cls = tdscf.TDA if method == "tda" else tdscf.TDDFT
    driver = driver_cls(
        molecule,
        xc_functional="b3lyp",
        nstates=nstates,
        eigensolver="davidson",
        davidson_tol=float(pyscf_solver.conv_tol),
        davidson_max_iter=int(pyscf_solver.max_cycle),
    )
    result = driver.kernel(nstates=nstates)
    assert bool(np.all(np.asarray(result.converged)))

    reference_e = np.asarray(pyscf_solver.e, dtype=np.float64)
    predicted_e = np.asarray(result.excitation_energies, dtype=np.float64)
    reference_f = np.asarray(pyscf_solver.oscillator_strength(), dtype=np.float64)
    predicted_f = np.asarray(driver.oscillator_strength(), dtype=np.float64)
    np.testing.assert_allclose(predicted_e, reference_e, atol=2e-7, rtol=2e-7)
    _assert_degenerate_cluster_strengths(reference_e, reference_f, predicted_f)
