import numpy as np
import pytest

from pyscf_reference import restricted_reference_from_pyscf_with_jax_rhf
from td_graddft.scf import RHFConfig
from td_graddft.workflows.core import run_reference
from td_graddft.workflows.types import SimulationConfig


def _pyscf_or_skip():
    try:
        from pyscf import dft, gto  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("PySCF is required for PySCF bridge tests.")


def _make_h2_b3lyp_mf():
    from pyscf import dft, gto

    mol = gto.Mole()
    mol.atom = """
    H 0.0 0.0 -0.35
    H 0.0 0.0  0.35
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
    if not mf.converged:
        raise RuntimeError("PySCF SCF did not converge for H2 test setup.")
    return mf


def test_restricted_reference_from_pyscf_with_jax_rhf_shapes_and_target_energy():
    _pyscf_or_skip()
    mf = _make_h2_b3lyp_mf()

    ref = restricted_reference_from_pyscf_with_jax_rhf(
        mf,
        max_l=1,
        rhf_config=RHFConfig(max_cycle=80, conv_tol=1e-11, conv_tol_density=1e-9),
        energy_target=float(mf.e_tot),
    )

    nao = ref.mo_coeff.shape[-1]
    assert ref.ao.shape[1] == nao
    assert ref.h1e.shape == (nao, nao)
    assert ref.rep_tensor.shape == (nao, nao, nao, nao)
    assert ref.mo_coeff.shape[0] == 2
    assert ref.mo_occ.shape[0] == 2
    assert ref.rdm1.shape == (2, nao, nao)
    assert np.isclose(float(ref.mf_energy), float(mf.e_tot), atol=1e-12, rtol=1e-12)


def test_workflow_reference_stage_rejects_legacy_mf_jax_rhf_backend():
    _pyscf_or_skip()
    mf = _make_h2_b3lyp_mf()

    simulation = SimulationConfig(
        nstates=1,
        scf_backend="jax_rhf",
        jax_basis_max_l=1,
    )
    with pytest.raises(ValueError, match="reference_spec"):
        run_reference(
            mf,
            scf_elapsed_s=0.0,
            simulation=simulation,
        )
