import types

import jax.numpy as jnp
import numpy as np
import pytest

from td_graddft import gto, scf, tdscf
from td_graddft.spectra import HARTREE_TO_EV


def test_tda_from_restricted_mf_stores_fields_and_spectra(monkeypatch):
    import td_graddft.tdscf.api as api

    captured = {}

    class FakeRestrictedCasidaTDDFT:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def tda(self, nstates=None):
            captured["method"] = "tda"
            captured["nstates"] = nstates
            return types.SimpleNamespace(
                excitation_energies=jnp.asarray([0.10, 0.20]),
                amplitudes="restricted-tda-amplitudes",
            )

    monkeypatch.setattr(api, "RestrictedCasidaTDDFT", FakeRestrictedCasidaTDDFT)
    monkeypatch.setattr(
        api.spectra,
        "oscillator_strengths",
        lambda reference, result, *, occupation_tolerance=1e-8: jnp.ones_like(
            result.excitation_energies
        ),
    )
    monkeypatch.setattr(
        api.spectra,
        "transition_dipoles",
        lambda reference, result, *, occupation_tolerance=1e-8: jnp.ones(
            (result.excitation_energies.size, 3)
        ),
    )

    reference = types.SimpleNamespace(
        mo_coeff=jnp.zeros((2, 2)),
        mo_occ=jnp.asarray([2.0, 0.0]),
        mo_energy=jnp.asarray([-0.5, 0.1]),
    )
    mf = types.SimpleNamespace(reference=reference, xc="pbe")

    td = tdscf.TDA(mf)
    td.nstates = 2
    result = td.kernel()

    assert result is td.result
    assert td.e is result.excitation_energies
    np.testing.assert_allclose(np.asarray(td.e_ev), np.asarray(td.e) * HARTREE_TO_EV)
    assert td.xy == "restricted-tda-amplitudes"
    assert td.converged is True
    assert captured["method"] == "tda"
    assert captured["nstates"] == 2
    assert captured["init"]["molecule"] is reference
    assert captured["init"]["xc_functional"].xc_spec == "pbe"
    np.testing.assert_allclose(np.asarray(td.oscillator_strength()), np.ones(2))
    assert td.transition_dipole().shape == (2, 3)


def test_tddft_from_raw_restricted_reference_uses_kernel(monkeypatch):
    import td_graddft.tdscf.api as api

    captured = {}
    xc_functional = object()
    xc_params = {"alpha": 0.5}

    class FakeRestrictedCasidaTDDFT:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def kernel(self, nstates=None):
            captured["method"] = "kernel"
            captured["nstates"] = nstates
            return types.SimpleNamespace(
                excitation_energies=jnp.asarray([0.30]),
                x_amplitudes="x",
                y_amplitudes="y",
            )

    monkeypatch.setattr(api, "RestrictedCasidaTDDFT", FakeRestrictedCasidaTDDFT)

    reference = types.SimpleNamespace(
        mo_coeff=jnp.zeros((2, 2)),
        mo_occ=jnp.asarray([2.0, 0.0]),
        mo_energy=jnp.asarray([-0.5, 0.1]),
    )

    td = tdscf.TDDFT(reference, xc_functional=xc_functional, xc_params=xc_params)
    td.nstates = 1
    result = td.kernel(nstates=3)

    assert result is td.result
    assert td.xy == ("x", "y")
    assert captured["method"] == "kernel"
    assert captured["nstates"] == 3
    assert captured["init"]["molecule"] is reference
    assert captured["init"]["xc_functional"] is xc_functional
    assert captured["init"]["xc_params"] is xc_params


def test_explicit_string_xc_functional_uses_semilocal_response(monkeypatch):
    import td_graddft.tdscf.api as api

    captured = {}

    class FakeRestrictedCasidaTDDFT:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def kernel(self, nstates=None):
            captured["nstates"] = nstates
            return types.SimpleNamespace(excitation_energies=jnp.asarray([0.25]))

    monkeypatch.setattr(api, "RestrictedCasidaTDDFT", FakeRestrictedCasidaTDDFT)

    reference = types.SimpleNamespace(
        mo_coeff=jnp.zeros((2, 2)),
        mo_occ=jnp.asarray([2.0, 0.0]),
        mo_energy=jnp.asarray([-0.5, 0.1]),
    )

    tdscf.TDDFT(reference, xc_functional="pbe").kernel(nstates=1)

    assert captured["init"]["xc_functional"].xc_spec == "pbe"


def test_tda_and_tddft_dispatch_unrestricted_references(monkeypatch):
    import td_graddft.tdscf.api as api

    calls = []

    class FakeUnrestrictedTDA:
        def __init__(self, **kwargs):
            calls.append(("utda_init", kwargs))

        def kernel(self, nstates=None):
            calls.append(("utda_kernel", nstates))
            return types.SimpleNamespace(
                excitation_energies=jnp.asarray([0.11]),
                amplitudes_alpha="xa",
                amplitudes_beta="xb",
            )

    class FakeUnrestrictedCasidaTDDFT:
        def __init__(self, **kwargs):
            calls.append(("utddft_init", kwargs))

        def kernel(self, nstates=None):
            calls.append(("utddft_kernel", nstates))
            return types.SimpleNamespace(
                excitation_energies=jnp.asarray([0.12]),
                x_amplitudes_alpha="xaa",
                x_amplitudes_beta="xbb",
                y_amplitudes_alpha="yaa",
                y_amplitudes_beta="ybb",
            )

    monkeypatch.setattr(api, "UnrestrictedTDA", FakeUnrestrictedTDA)
    monkeypatch.setattr(api, "UnrestrictedCasidaTDDFT", FakeUnrestrictedCasidaTDDFT)

    reference = types.SimpleNamespace(
        mo_coeff=jnp.zeros((2, 2, 2)),
        mo_occ=jnp.zeros((2, 2)),
        mo_energy=jnp.zeros((2, 2)),
        nocc_alpha=1,
        nocc_beta=1,
    )

    tda = tdscf.TDA(reference)
    tda.kernel(nstates=1)
    tddft = tdscf.TDDFT(reference)
    tddft.kernel(nstates=2)

    assert ("utda_kernel", 1) in calls
    assert ("utddft_kernel", 2) in calls
    assert tda.xy == ("xa", "xb")
    assert tddft.xy == (("xaa", "xbb"), ("yaa", "ybb"))


def test_scf_shortcuts_return_tdscf_facades():
    mf = scf.RKS(gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"))

    assert isinstance(mf.TDA(), tdscf.TDA)
    assert isinstance(mf.TDDFT(), tdscf.TDDFT)


def test_kernel_requires_completed_ground_state_reference():
    td = tdscf.TDA(types.SimpleNamespace(reference=None))

    with pytest.raises(RuntimeError, match="Run ground-state"):
        td.kernel()
