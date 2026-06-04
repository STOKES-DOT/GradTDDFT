from types import SimpleNamespace

import numpy as np
import pytest

from td_graddft.scf import UKSConfig
from td_graddft.scf.builders import unrestricted_molecule_from_spec_with_jax_uks
from td_graddft.scf.differentiable import _is_unrestricted_reference
from td_graddft.workflows.core import run_molecule_from_spec
from td_graddft.workflows.types import MoleculeSpecConfig, SimulationConfig


def test_spin_resolved_charged_state_overrides_restricted_nocc_marker():
    molecule = SimpleNamespace(
        nocc=5,
        nocc_alpha=None,
        nocc_beta=None,
        mo_occ=np.asarray([[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 0.0]]),
        rdm1=np.stack([np.eye(5), np.diag([1.0, 1.0, 1.0, 1.0, 0.0])], axis=0),
        mo_coeff=np.stack([np.eye(5), np.eye(5)], axis=0),
    )

    assert _is_unrestricted_reference(molecule)


def test_unrestricted_molecule_from_spec_with_jax_uks_h_atom_smoke():
    ref = unrestricted_molecule_from_spec_with_jax_uks(
        atom="H 0.0 0.0 0.0",
        basis="sto-3g",
        xc_spec="hf",
        unit="Angstrom",
        charge=0,
        spin=1,
        cart=True,
        grids_level=0,
        max_l=0,
        uks_config=UKSConfig(
            xc_spec="hf",
            max_cycle=12,
            conv_tol=1e-9,
            conv_tol_density=1e-7,
            damping=0.1,
        ),
        grid_ao_backend="jax",
        integral_backend="jax",
    )

    nao = ref.mo_coeff.shape[-1]
    assert ref.ao.shape[1] == nao
    assert ref.h1e.shape == (nao, nao)
    assert ref.rep_tensor.shape == (nao, nao, nao, nao)
    assert ref.mo_coeff.shape[0] == 2
    assert ref.mo_occ.shape[0] == 2
    assert ref.rdm1.shape[0] == 2
    assert ref.nocc_alpha == 1
    assert ref.nocc_beta == 0
    assert np.isfinite(float(ref.mf_energy))
    assert np.isfinite(float(ref.exact_exchange_fraction))


def test_unrestricted_molecule_from_spec_with_jax_uks_builds_zero_local_pt2_for_h_atom():
    ref = unrestricted_molecule_from_spec_with_jax_uks(
        atom="H 0.0 0.0 0.0",
        basis="sto-3g",
        xc_spec="hf",
        unit="Angstrom",
        charge=0,
        spin=1,
        cart=True,
        grids_level=0,
        max_l=0,
        uks_config=UKSConfig(
            xc_spec="hf",
            max_cycle=12,
            conv_tol=1e-9,
            conv_tol_density=1e-7,
            damping=0.1,
        ),
        grid_ao_backend="jax",
        integral_backend="jax",
        compute_local_pt2_features=True,
    )

    assert ref.pt2_local is not None
    assert ref.pt2_local.ndim == 1
    assert ref.pt2_local.shape[0] == ref.ao.shape[0]
    assert np.allclose(np.asarray(ref.pt2_local), 0.0, atol=1e-10)


def test_unrestricted_molecule_from_spec_with_jax_uks_invalid_spin_parity_raises():
    with pytest.raises(ValueError, match="N \\+ spin must be even"):
        _ = unrestricted_molecule_from_spec_with_jax_uks(
            atom="H 0.0 0.0 0.0",
            basis="sto-3g",
            xc_spec="hf",
            unit="Angstrom",
            charge=0,
            spin=0,
            cart=True,
            grids_level=0,
            max_l=0,
            uks_config=UKSConfig(xc_spec="hf", max_cycle=4),
            grid_ao_backend="jax",
            integral_backend="jax",
        )


def test_run_molecule_from_spec_accepts_jax_uks_for_open_shell():
    reference = run_molecule_from_spec(
        MoleculeSpecConfig(
            atom="H 0.0 0.0 0.0",
            basis="sto-3g",
            xc="hf",
            unit="Angstrom",
            charge=0,
            spin=1,
            cart=True,
            grids_level=0,
            verbose=0,
        ),
        simulation=SimulationConfig(
            nstates=0,
            scf_backend="jax_uks",
            jax_basis_max_l=0,
            jax_grid_ao_backend="jax",
            jax_integral_backend="jax",
            jax_uks_xc_spec="hf",
            jax_uks_max_cycle=12,
            jax_uks_conv_tol=1e-9,
            jax_uks_conv_tol_density=1e-7,
            jax_uks_damping=0.1,
        ),
    )

    assert reference.nstates == 0
    assert reference.nstates_full == 0
    assert reference.energies_au.size == 0
    assert reference.oscillator_strengths.size == 0
    assert reference.molecule.mo_coeff.shape[0] == 2
    assert reference.molecule.nocc_alpha == 1
    assert reference.molecule.nocc_beta == 0
