import importlib
from pathlib import Path
import types

import jax.numpy as jnp
import pytest

from td_graddft import gto, scf


def test_gto_m_stores_pyscf_style_molecule_fields():
    mol = gto.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis="sto-3g",
        unit="Angstrom",
        charge=0,
        spin=0,
        cart=True,
        verbose=0,
    )

    assert mol.atom.startswith("O")
    assert mol.basis == "sto-3g"
    assert mol.unit == "Angstrom"
    assert mol.charge == 0
    assert mol.spin == 0
    assert mol.cart is True
    assert mol.verbose == 0
    assert mol.nelectron == 10
    assert mol.to_spec().symbols == ("O", "H", "H")


def test_rks_kernel_runs_ground_state_without_building_reference(monkeypatch):
    captured = {}
    cache_calls = []

    def forbidden_reference_builder(**kwargs):
        raise AssertionError("RKS.kernel() should not build a response reference")

    def fake_cache(**kwargs):
        cache_calls.append(kwargs)
        return kwargs["cache_dir"]

    def fake_inputs(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            geometry_is_traced=False,
            integral_backend=kwargs["integral_backend"],
            grid_ao_backend=kwargs["grid_ao_backend"],
            direct_basis=None,
            as_rks_kwargs=lambda: {
                "overlap": "s",
                "hcore": "h",
                "eri": "eri",
                "eri_pair_matrix": None,
                "nelectron": 2,
                "nuclear_repulsion": 0.7,
                "ao": "ao",
                "ao_deriv1": "ao_deriv1",
                "grid_weights": "weights",
                "df_factors": None,
                "direct_basis": None,
                "init_mo_coeff": None,
                "init_mo_occ": None,
                "init_mo_energy": None,
            },
        )

    def fake_runner(**kwargs):
        assert kwargs["config"].xc_spec == "pbe"
        return types.SimpleNamespace(
            total_energy=-76.0,
            mo_energy="mo_energy",
            mo_coeff="mo_coeff",
            mo_occ="mo_occ",
            density_matrix="density",
            converged=True,
            cycles=7,
        )

    monkeypatch.setattr(
        "td_graddft.scf.facade.restricted_molecule_from_spec_with_jax_rks",
        forbidden_reference_builder,
    )
    monkeypatch.setattr("td_graddft.scf.facade.configure_jax_persistent_cache", fake_cache)
    monkeypatch.setattr(
        "td_graddft.scf.facade.build_rks_integral_inputs",
        fake_inputs,
        raising=False,
    )
    monkeypatch.setattr(
        "td_graddft.scf.facade.run_rks_from_integrals",
        fake_runner,
        raising=False,
    )

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    mf = scf.RKS(mol, xc="pbe")
    energy = mf.kernel()

    assert energy == -76.0
    assert mf.e_tot == -76.0
    assert mf.reference is None
    assert mf.mo_energy == "mo_energy"
    assert mf.mo_coeff == "mo_coeff"
    assert mf.mo_occ == "mo_occ"
    assert mf.converged is True
    assert captured["atom"].symbols == ("H", "H")
    assert captured["basis"] == "sto-3g"
    assert captured["xc_spec"] == "pbe"
    assert captured["init_guess"] == "minao"
    assert captured["integral_backend"] == "libcint"
    assert captured["libcint_geometry_grad_policy"] == "analytic"
    assert captured["config"].jk_backend == "full"
    assert captured["include_dipole_integrals"] is False
    assert len(cache_calls) == 1


def test_rks_lazy_reference_passes_hfx_feature_options(monkeypatch):
    captured = {}

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            mf_energy=-76.0,
            mo_energy=None,
            mo_coeff=None,
            mo_occ=None,
        )

    monkeypatch.setattr(
        "td_graddft.scf.facade.restricted_molecule_from_spec_with_jax_rks",
        fake_builder,
    )

    mf = scf.RKS(gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"))
    mf.compute_local_hfx_features = True
    mf.compute_local_hfx_aux = True
    mf.hfx_omega_values = (0.0, 0.4)
    mf.hfx_chunk_size = 128
    mf.e_tot = -76.0
    reference = mf._ensure_reference()

    assert reference.mf_energy == -76.0
    assert captured["compute_local_hfx_features"] is True
    assert captured["compute_local_hfx_aux"] is True
    assert captured["hfx_omega_values"] == (0.0, 0.4)
    assert captured["hfx_chunk_size"] == 128


def test_tdscf_builds_reference_lazily_after_ground_state_kernel(monkeypatch):
    def fake_builder(**kwargs):
        return types.SimpleNamespace(
            mf_energy=-1.0,
            mo_energy="ref_mo_energy",
            mo_coeff=jnp.eye(2),
            mo_occ=jnp.array([1.0, 0.0]),
        )

    monkeypatch.setattr(
        "td_graddft.scf.facade.restricted_molecule_from_spec_with_jax_rks",
        fake_builder,
    )

    mf = scf.RKS(gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"))
    mf.e_tot = -1.0

    td = mf.TDA()

    assert mf.reference is None
    assert td.reference is mf.reference
    assert mf.reference.mf_energy == -1.0


def test_uks_kernel_calls_existing_unrestricted_reference_builder(monkeypatch):
    captured = {}
    cache_calls = []

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            mf_energy=-39.0,
            mo_energy="mo_energy",
            mo_coeff="mo_coeff",
            mo_occ="mo_occ",
        )

    def fake_cache(**kwargs):
        cache_calls.append(kwargs)
        return kwargs["cache_dir"]

    monkeypatch.setattr(
        "td_graddft.scf.facade.unrestricted_molecule_from_spec_with_jax_uks",
        fake_builder,
    )
    monkeypatch.setattr("td_graddft.scf.facade.configure_jax_persistent_cache", fake_cache)

    mol = gto.M(atom="O 0 0 0", basis="sto-3g", spin=2)
    mf = scf.UKS(mol, xc="pbe")
    energy = mf.kernel()

    assert energy == -39.0
    assert captured["atom"].symbols == ("O",)
    assert captured["basis"] == "sto-3g"
    assert captured["xc_spec"] == "pbe"
    assert captured["integral_backend"] == "libcint"
    assert captured["libcint_geometry_grad_policy"] == "analytic"
    assert captured["uks_config"].max_cycle == mf.max_cycle
    assert len(cache_calls) == 1


def test_rks_run_and_backend_helpers(monkeypatch):
    monkeypatch.setattr(
        "td_graddft.scf.facade.build_rks_integral_inputs",
        lambda **kwargs: types.SimpleNamespace(
            geometry_is_traced=False,
            integral_backend=kwargs["integral_backend"],
            grid_ao_backend=kwargs["grid_ao_backend"],
            as_rks_kwargs=lambda: {},
        ),
    )
    monkeypatch.setattr(
        "td_graddft.scf.facade.run_rks_from_integrals",
        lambda **kwargs: types.SimpleNamespace(
            total_energy=-1.0,
            mo_energy=None,
            mo_coeff=None,
            mo_occ=None,
            converged=True,
        ),
    )

    mf = scf.RKS(gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"))

    assert mf.density_fit() is mf
    assert mf.jk_backend == "df"
    assert mf.direct_scf() is mf
    assert mf.jk_backend == "direct"
    assert mf.run() is mf
    assert mf.e_tot == -1.0


def test_nuc_grad_method_reports_explicit_scf_gradient_is_disabled():
    mf = scf.RKS(gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"))
    with pytest.raises(NotImplementedError, match="implicit-differential"):
        mf.nuc_grad_method().kernel()


def test_top_level_import_exposes_gto_and_scf_modules():
    assert importlib.import_module("td_graddft.gto") is gto
    assert importlib.import_module("td_graddft.scf") is scf


def test_real_rks_kernel_smoke_sto3g_h2():
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    mf = scf.RKS(mol, xc="pbe")
    mf.max_cycle = 4
    energy = mf.kernel()

    assert energy < 0.0
    assert mf.reference is None
    assert mf.mo_coeff is not None


def test_real_rks_kernel_smoke_sto3g_water():
    mol = gto.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis="sto-3g",
    )
    mf = scf.RKS(mol, xc="pbe")
    mf.max_cycle = 6
    energy = mf.kernel()

    assert energy < -50.0
    assert mf.reference is None
    assert mf.mo_coeff is not None
