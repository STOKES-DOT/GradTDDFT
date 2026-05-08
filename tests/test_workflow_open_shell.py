import os
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from td_graddft.neural_xc import make_neural_xc_functional
from td_graddft.workflows.core import run_neural_tddft, run_reference
from td_graddft.workflows.types import SimulationConfig


pytestmark = pytest.mark.skipif(
    os.getenv("TD_GRADDFT_RUN_OPEN_SHELL_TESTS", "0") != "1",
    reason="Open-shell tests are disabled by default; set TD_GRADDFT_RUN_OPEN_SHELL_TESTS=1.",
)


def _pyscf_or_skip():
    try:
        from pyscf import dft, gto  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("PySCF is required for open-shell workflow tests.")


def _make_oh_uks_pbe_sto3g():
    from pyscf import dft, gto

    mol = gto.Mole()
    mol.atom = """
    O 0.000000 0.000000 0.000000
    H 0.000000 0.000000 0.969700
    """
    mol.unit = "Angstrom"
    mol.basis = "sto-3g"
    mol.spin = 1
    mol.charge = 0
    mol.verbose = 0
    mol.build()

    mf = dft.UKS(mol)
    mf.xc = "pbe"
    mf.grids.level = 0
    mf.conv_tol = 1e-9
    mf.max_cycle = 80
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF UKS(PBE/STO-3G) did not converge for OH workflow test.")
    return mf


def test_workflow_reference_stage_accepts_jax_uks_backend_open_shell():
    _pyscf_or_skip()
    mf = _make_oh_uks_pbe_sto3g()

    simulation = SimulationConfig(
        nstates=0,
        scf_backend="jax_uks",
        jax_basis_max_l=1,
        jax_uks_max_cycle=8,
        jax_uks_conv_tol=1e-8,
        jax_uks_conv_tol_density=1e-6,
        jax_uks_damping=0.15,
    )
    reference = run_reference(
        mf,
        scf_elapsed_s=0.0,
        simulation=simulation,
    )

    assert reference.nstates == 0
    assert reference.nstates_full >= 1
    assert reference.energies_au.size == 0
    assert reference.oscillator_strengths.size == 0
    assert reference.molecule.mo_coeff.shape[0] == 2
    assert reference.molecule.mo_occ.shape[0] == 2
    assert reference.molecule.rdm1.shape[0] == 2
    assert np.isfinite(float(reference.molecule.exact_exchange_fraction))


def test_run_neural_tddft_supports_unrestricted_reference():
    _pyscf_or_skip()
    mf = _make_oh_uks_pbe_sto3g()

    reference = run_reference(
        mf,
        scf_elapsed_s=0.0,
        simulation=SimulationConfig(nstates=1, scf_backend="pyscf"),
    )
    functional = make_neural_xc_functional(
        semilocal_xc="pbe",
        hidden_dims=(16,),
        name="neural_xc_open_shell_smoke",
    )
    params = functional.init_from_molecule(jax.random.PRNGKey(0), reference.molecule)
    training = SimpleNamespace(functional=functional, params=params)
    neural = run_neural_tddft(reference, training)

    assert neural.energies_au.shape == (1,)
    assert neural.oscillator_strengths.shape == (1,)
    assert np.all(np.isfinite(np.asarray(neural.energies_au)))
    assert np.all(np.isfinite(np.asarray(neural.oscillator_strengths)))
