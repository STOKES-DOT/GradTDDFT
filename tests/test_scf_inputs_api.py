from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from td_graddft.data.molecule import parse_molecule_spec
from td_graddft.data.integrals.libcint.mol import build_libcint_mol
from td_graddft.scf import RKSConfig, UKSConfig
from td_graddft.scf.init_guess import (
    RestrictedInitGuess,
    restricted_init_guess_from_pyscf,
    unrestricted_init_guess_from_pyscf,
)
from td_graddft.scf.inputs import (
    RKSIntegralInputs,
    UKSIntegralInputs,
    build_rks_integral_inputs,
    build_uks_integral_inputs,
)


def _h2_spec():
    return parse_molecule_spec("H 0 0 0; H 0 0 0.74")


def _h_atom_spec():
    return parse_molecule_spec("H 0 0 0", spin=1)


def test_rks_integral_inputs_default_integral_backend_is_libcint():
    inputs = RKSIntegralInputs(
        basis=object(),
        overlap=jnp.zeros((1, 1)),
        hcore=jnp.zeros((1, 1)),
        eri=None,
        eri_pair_matrix=None,
        df_factors=None,
        direct_basis=None,
        nelectron=2,
        nuclear_repulsion=0.0,
        coords=jnp.zeros((1, 3)),
        grid_weights=jnp.ones((1,)),
        ao=jnp.zeros((1, 1)),
        ao_deriv1=jnp.zeros((4, 1, 1)),
        ao_laplacian=None,
        dipole_integrals=None,
    )

    assert inputs.integral_backend == "libcint"
    assert inputs.grid_ao_backend == "jax"


def test_uks_integral_inputs_default_integral_backend_is_libcint():
    inputs = UKSIntegralInputs(
        basis=object(),
        overlap=jnp.zeros((1, 1)),
        hcore=jnp.zeros((1, 1)),
        eri=jnp.zeros((1, 1, 1, 1)),
        nalpha=1,
        nbeta=0,
        nuclear_repulsion=0.0,
        coords=jnp.zeros((1, 3)),
        grid_weights=jnp.ones((1,)),
        ao=jnp.zeros((1, 1)),
        ao_deriv1=jnp.zeros((4, 1, 1)),
        ao_laplacian=None,
        dipole_integrals=jnp.zeros((3, 1, 1)),
    )

    assert inputs.integral_backend == "libcint"
    assert inputs.grid_ao_backend == "jax"


def test_build_rks_integral_inputs_full_backend_uses_packed_eri():
    inputs = build_rks_integral_inputs(
        atom=_h2_spec(),
        basis="sto-3g",
        config=RKSConfig(jk_backend="full"),
        integral_backend="jax",
        grid_ao_backend="jax",
        grids_level=0,
        max_l=1,
    )

    assert inputs.overlap.shape == inputs.hcore.shape
    assert inputs.nelectron == 2
    assert inputs.eri is None
    assert inputs.eri_pair_matrix is not None
    assert inputs.df_factors is None
    assert inputs.direct_basis is None
    assert inputs.ao.shape[0] == inputs.grid_weights.shape[0]
    assert inputs.dipole_integrals.shape[0] == 3


def test_build_rks_integral_inputs_df_backend_uses_df_factors():
    inputs = build_rks_integral_inputs(
        atom=_h2_spec(),
        basis="sto-3g",
        config=RKSConfig(jk_backend="df"),
        integral_backend="jax",
        grid_ao_backend="jax",
        grids_level=0,
        max_l=1,
    )

    assert inputs.eri is None
    assert inputs.eri_pair_matrix is None
    assert inputs.df_factors is not None
    assert inputs.df_factors.shape[1:] == inputs.overlap.shape
    assert inputs.direct_basis is None


def test_build_rks_integral_inputs_direct_backend_keeps_basis_not_eri():
    inputs = build_rks_integral_inputs(
        atom=_h2_spec(),
        basis="sto-3g",
        config=RKSConfig(jk_backend="direct"),
        integral_backend="jax",
        grid_ao_backend="jax",
        grids_level=0,
        max_l=1,
    )

    assert inputs.eri is None
    assert inputs.eri_pair_matrix is None
    assert inputs.df_factors is None
    assert inputs.direct_basis is inputs.basis
    assert inputs.response_eri_pair_matrix() is not None
    assert jnp.asarray(inputs.nuclear_repulsion).shape == ()


def test_build_rks_integral_inputs_cuda_direct_skips_eri_group_precompute(monkeypatch):
    import td_graddft.scf.inputs as inputs_mod

    monkeypatch.setattr(inputs_mod, "cuda_ffi_available", lambda: True)
    captured = {}

    def fake_overlap_hcore_matrices(basis, *, backend="auto", **kwargs):
        captured["basis"] = basis
        captured["backend"] = backend
        shape = (basis.nao, basis.nao)
        return jnp.eye(shape[0]), 2.0 * jnp.eye(shape[0])

    monkeypatch.setattr(inputs_mod, "overlap_hcore_matrices", fake_overlap_hcore_matrices)

    inputs = build_rks_integral_inputs(
        atom=_h2_spec(),
        basis="sto-3g",
        config=RKSConfig(jk_backend="direct", direct_jk_engine="cuda"),
        integral_backend="jax",
        grid_ao_backend="jax",
        grids_level=0,
        max_l=1,
    )

    assert inputs.direct_basis is inputs.basis
    assert captured["basis"] is inputs.basis
    assert captured["backend"] == "cuda"
    assert inputs.basis.precompute_eri_groups is False
    assert inputs.basis.quartet_groups == ()
    assert inputs.basis.shell_quartet_groups == ()


def test_build_rks_integral_inputs_libcint_cuda_direct_uses_direct_digest_inputs(monkeypatch):
    import td_graddft.scf.inputs as inputs_mod

    pytest.importorskip("pyscf")
    monkeypatch.setattr(inputs_mod, "cuda_ffi_available", lambda: True)

    inputs = build_rks_integral_inputs(
        atom=_h2_spec(),
        basis="sto-3g",
        config=RKSConfig(jk_backend="direct", direct_jk_engine="cuda"),
        integral_backend="libcint",
        grid_ao_backend="jax",
        grids_level=0,
        max_l=1,
    )

    assert inputs.direct_basis is inputs.basis
    assert inputs.eri is None
    assert inputs.eri_pair_matrix is None


@pytest.mark.parametrize("jk_backend", ["full", "df"])
def test_build_rks_integral_inputs_libcint_nondirect_skips_eri_group_precompute(jk_backend):
    pytest.importorskip("pyscf")

    inputs = build_rks_integral_inputs(
        atom=_h2_spec(),
        basis="sto-3g",
        config=RKSConfig(jk_backend=jk_backend),
        integral_backend="libcint",
        grid_ao_backend="jax",
        grids_level=0,
        max_l=1,
        include_dipole_integrals=False,
    )

    assert inputs.basis.precompute_eri_groups is False
    assert inputs.basis.quartet_groups == ()
    assert inputs.basis.shell_quartet_groups == ()


def test_build_rks_integral_inputs_libcint_nontraced_spec_uses_single_pyscf_mol(monkeypatch):
    import td_graddft.scf.inputs as inputs_mod

    pytest.importorskip("pyscf")

    monkeypatch.setattr(inputs_mod, "cuda_ffi_available", lambda: True)

    def _fail_traceable_one_electron(*args, **kwargs):
        raise AssertionError("Non-traced libcint inputs should use the PySCF mol fast path.")

    monkeypatch.setattr(inputs_mod, "libcint_int1e_with_coords", _fail_traceable_one_electron)

    inputs = build_rks_integral_inputs(
        atom=_h2_spec(),
        basis="sto-3g",
        config=RKSConfig(jk_backend="direct", direct_jk_engine="cuda"),
        integral_backend="libcint",
        grid_ao_backend="jax",
        grids_level=0,
        max_l=1,
    )

    assert inputs.nelectron == 2
    assert inputs.eri_pair_matrix is None
    assert inputs.grid_ao_backend == "jax"


def test_build_rks_integral_inputs_libcint_default_uses_minao_initial_density(monkeypatch):
    import td_graddft.scf.inputs as inputs_mod

    captured = {}

    def fake_init_guess(**kwargs):
        captured.update(kwargs)
        return RestrictedInitGuess(density=jnp.eye(2))

    monkeypatch.setattr(inputs_mod, "restricted_init_guess_from_pyscf", fake_init_guess)

    inputs = build_rks_integral_inputs(
        atom=_h2_spec(),
        basis="sto-3g",
        config=RKSConfig(xc_spec="pbe0", jk_backend="full"),
        integral_backend="libcint",
        grid_ao_backend="jax",
        grids_level=0,
        max_l=1,
    )

    assert captured["init_guess"] == "minao"
    assert inputs.init_density is not None
    assert inputs.init_mo_coeff is None
    assert inputs.init_mo_occ is None


def test_build_rks_integral_inputs_1e_initial_guess_keeps_density_unset():
    inputs = build_rks_integral_inputs(
        atom=_h2_spec(),
        basis="sto-3g",
        config=RKSConfig(xc_spec="pbe", jk_backend="full"),
        integral_backend="jax",
        grid_ao_backend="jax",
        grids_level=0,
        max_l=1,
        init_guess="1e",
    )

    assert inputs.init_density is None
    assert inputs.init_mo_coeff is None
    assert inputs.init_mo_occ is None


def test_build_rks_integral_inputs_jax_uses_fused_core_matrices(monkeypatch):
    import td_graddft.scf.inputs as inputs_mod

    def fake_overlap_hcore_matrices(basis, *, backend="auto", **kwargs):
        shape = (basis.nao, basis.nao)
        if backend != "jax":
            raise AssertionError("Default JAX RKS input construction should request backend='jax'.")
        return jnp.eye(shape[0]), 2.0 * jnp.eye(shape[0])

    monkeypatch.setattr(inputs_mod, "overlap_hcore_matrices", fake_overlap_hcore_matrices)
    monkeypatch.delattr(inputs_mod, "CudaOneElectronBuilder", raising=False)

    inputs = build_rks_integral_inputs(
        atom=_h2_spec(),
        basis="sto-3g",
        config=RKSConfig(jk_backend="direct"),
        integral_backend="jax",
        grid_ao_backend="jax",
        grids_level=0,
        max_l=1,
    )

    assert inputs.overlap.shape == inputs.hcore.shape
    assert inputs.direct_basis is inputs.basis


def test_restricted_chk_init_guess_uses_bound_pyscf_chkfile_api(monkeypatch):
    class FakeMF:
        def __init__(self):
            self.chk_call = None
            self.minao_called = False

        def init_guess_by_chkfile(self, *, chkfile=None, project=None):
            self.chk_call = (chkfile, project)
            return np.eye(2)

        def init_guess_by_minao(self, mol):
            self.minao_called = True
            return 2.0 * np.eye(2)

    fake_mf = FakeMF()

    monkeypatch.setattr(
        "td_graddft.scf.init_guess._build_pyscf_ks_object",
        lambda **kwargs: (object(), fake_mf),
    )

    guess = restricted_init_guess_from_pyscf(
        atom=_h2_spec(),
        basis="sto-3g",
        unit="Bohr",
        charge=0,
        spin=0,
        cart=False,
        verbose=0,
        xc_spec="pbe",
        init_guess="chk",
        sap_basis=None,
        chkfile="fake.chk",
        chkfile_project=True,
        geometry_is_traced=False,
        dtype=jnp.float64,
    )

    assert fake_mf.chk_call == ("fake.chk", True)
    assert fake_mf.minao_called is False
    np.testing.assert_allclose(np.asarray(guess.density), np.eye(2))


def test_unrestricted_chk_init_guess_uses_bound_pyscf_chkfile_api(monkeypatch):
    class FakeMF:
        def __init__(self):
            self.chk_call = None
            self.minao_called = False

        def init_guess_by_chkfile(self, *, chkfile=None, project=None):
            self.chk_call = (chkfile, project)
            return np.stack([np.eye(1), np.zeros((1, 1))], axis=0)

        def init_guess_by_minao(self, mol):
            self.minao_called = True
            return np.stack([2.0 * np.eye(1), np.zeros((1, 1))], axis=0)

    fake_mf = FakeMF()

    monkeypatch.setattr(
        "td_graddft.scf.init_guess._build_pyscf_ks_object",
        lambda **kwargs: (object(), fake_mf),
    )

    guess = unrestricted_init_guess_from_pyscf(
        atom=_h_atom_spec(),
        basis="sto-3g",
        unit="Bohr",
        charge=0,
        spin=1,
        cart=False,
        verbose=0,
        xc_spec="pbe",
        init_guess="chk",
        sap_basis=None,
        chkfile="fake.chk",
        chkfile_project=False,
        geometry_is_traced=False,
        dtype=jnp.float64,
    )

    assert fake_mf.chk_call == ("fake.chk", False)
    assert fake_mf.minao_called is False
    np.testing.assert_allclose(np.asarray(guess.density_alpha), np.eye(1))
    np.testing.assert_allclose(np.asarray(guess.density_beta), np.zeros((1, 1)))


def test_restricted_init_guess_reuses_prebuilt_libcint_mol(monkeypatch):
    class FakeMF:
        def get_init_guess(self, mol, key="minao"):
            assert key == "minao"
            return np.eye(2)

    fake_mol = object()

    monkeypatch.setattr(
        "td_graddft.scf.init_guess.build_libcint_mol",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("build_libcint_mol should not be called")),
    )
    monkeypatch.setattr("pyscf.dft.RKS", lambda mol, xc=None: FakeMF())

    guess = restricted_init_guess_from_pyscf(
        atom=_h2_spec(),
        basis="sto-3g",
        unit="Bohr",
        charge=0,
        spin=0,
        cart=False,
        verbose=0,
        xc_spec="pbe",
        init_guess="minao",
        sap_basis=None,
        chkfile=None,
        chkfile_project=None,
        geometry_is_traced=False,
        dtype=jnp.float64,
        libcint_mol=fake_mol,
    )

    np.testing.assert_allclose(np.asarray(guess.density), np.eye(2))


def test_build_libcint_mol_caches_identical_host_handles(monkeypatch):
    import td_graddft.data.integrals.libcint.mol as mol_mod

    class FakeGTO:
        def __init__(self):
            self.calls = 0

        def M(self, **kwargs):
            self.calls += 1
            return {"call": self.calls, "kwargs": kwargs}

    fake_gto = FakeGTO()
    monkeypatch.setattr("pyscf.gto.M", fake_gto.M)
    mol_mod._LIBCINT_MOL_CACHE.clear()

    spec = _h2_spec()
    mol1 = build_libcint_mol(
        atom=spec,
        basis="sto-3g",
        unit="Bohr",
        charge=0,
        spin=0,
        cart=False,
        verbose=0,
    )
    mol2 = build_libcint_mol(
        atom=spec,
        basis="sto-3g",
        unit="Bohr",
        charge=0,
        spin=0,
        cart=False,
        verbose=0,
    )

    assert fake_gto.calls == 1
    assert mol1 is mol2


def test_libcint_one_electron_from_mol_caches_host_integrals():
    import td_graddft.scf.inputs as inputs_mod

    class FakeMol:
        cart = False

        def __init__(self):
            self.calls = []

        def intor_symmetric(self, name, comp=None):
            self.calls.append((name, comp))
            if comp == 3:
                return np.zeros((3, 2, 2))
            return np.eye(2)

    fake_mol = FakeMol()
    inputs_mod._LIBCINT_HOST_INTEGRAL_CACHE.clear()

    out1 = inputs_mod._libcint_one_electron_from_mol(
        mol=fake_mol,
        geometry_anchor=jnp.zeros((1, 3)),
        geometry_grad_policy="analytic",
        include_dipole_integrals=True,
    )
    out2 = inputs_mod._libcint_one_electron_from_mol(
        mol=fake_mol,
        geometry_anchor=jnp.zeros((1, 3)),
        geometry_grad_policy="analytic",
        include_dipole_integrals=True,
    )

    assert fake_mol.calls.count(("int1e_ovlp_sph", None)) == 1
    assert fake_mol.calls.count(("int1e_kin_sph", None)) == 1
    assert fake_mol.calls.count(("int1e_nuc_sph", None)) == 1
    assert fake_mol.calls.count(("int1e_r_sph", 3)) == 1
    assert out1[0] is out2[0]
    assert out1[1] is out2[1]
    assert out1[2] is out2[2]


def test_cached_libcint_host_integral_reuses_s4_eri_binding():
    import td_graddft.scf.inputs as inputs_mod

    fake_mol = object()
    calls = {"n": 0}
    inputs_mod._LIBCINT_HOST_INTEGRAL_CACHE.clear()

    def _loader():
        calls["n"] += 1
        return np.eye(3)

    out1 = inputs_mod._cached_libcint_host_integral(
        mol=fake_mol,
        integral_name="int2e_s4",
        geometry_anchor=jnp.zeros((1, 3)),
        geometry_grad_policy="analytic",
        loader=_loader,
    )
    out2 = inputs_mod._cached_libcint_host_integral(
        mol=fake_mol,
        integral_name="int2e_s4",
        geometry_anchor=jnp.zeros((1, 3)),
        geometry_grad_policy="analytic",
        loader=_loader,
    )

    assert calls["n"] == 1
    assert out1 is out2


def test_build_rks_integral_inputs_cuda_direct_jax_uses_single_integrals_entrypoint(monkeypatch):
    import td_graddft.scf.inputs as inputs_mod

    monkeypatch.setattr(inputs_mod, "cuda_ffi_available", lambda: True)

    def _fail_direct_builder(*args, **kwargs):
        raise AssertionError("SCF inputs should not instantiate CUDA one-electron builders directly.")

    monkeypatch.setattr(inputs_mod, "CudaOneElectronBuilder", _fail_direct_builder, raising=False)

    def fake_overlap_hcore_matrices(basis, *, backend="auto", **kwargs):
        shape = (basis.nao, basis.nao)
        return jnp.eye(shape[0]), 2.0 * jnp.eye(shape[0])

    monkeypatch.setattr(inputs_mod, "overlap_hcore_matrices", fake_overlap_hcore_matrices)

    inputs = build_rks_integral_inputs(
        atom=_h2_spec(),
        basis="sto-3g",
        config=RKSConfig(jk_backend="direct", direct_jk_engine="cuda"),
        integral_backend="jax",
        grid_ao_backend="jax",
        grids_level=0,
        max_l=1,
    )

    assert inputs.overlap.shape == inputs.hcore.shape


def test_build_rks_integral_inputs_cuda_direct_jax_uses_cuda_one_electron(monkeypatch):
    import td_graddft.scf.inputs as inputs_mod

    monkeypatch.setattr(inputs_mod, "cuda_ffi_available", lambda: True)
    captured = {}

    def fake_overlap_hcore_matrices(basis, *, backend="auto", **kwargs):
        captured["basis"] = basis
        captured["backend"] = backend
        shape = (captured["basis"].nao, captured["basis"].nao)
        if backend != "cuda":
            raise AssertionError("CUDA direct JAX input should request backend='cuda'.")
        return jnp.eye(shape[0]), 2.0 * jnp.eye(shape[0])

    monkeypatch.setattr(inputs_mod, "overlap_hcore_matrices", fake_overlap_hcore_matrices)
    monkeypatch.delattr(inputs_mod, "CudaOneElectronBuilder", raising=False)

    inputs = build_rks_integral_inputs(
        atom=_h2_spec(),
        basis="sto-3g",
        config=RKSConfig(jk_backend="direct", direct_jk_engine="cuda"),
        integral_backend="jax",
        grid_ao_backend="jax",
        grids_level=0,
        max_l=1,
    )

    assert captured["basis"] is inputs.basis
    assert captured["backend"] == "cuda"
    assert jnp.allclose(inputs.overlap, jnp.eye(inputs.basis.nao))
    assert jnp.allclose(inputs.hcore, 2.0 * jnp.eye(inputs.basis.nao))


def test_build_rks_integral_inputs_cuda_direct_jax_falls_back_without_cuda(monkeypatch):
    import td_graddft.scf.inputs as inputs_mod

    captured = {}

    def fake_core_builder(basis, *, backend="auto", **kwargs):
        captured["basis"] = basis
        captured["backend"] = backend
        shape = (basis.nao, basis.nao)
        if backend != "jax":
            raise AssertionError("Non-CUDA fallback must request backend='jax'.")
        return jnp.eye(shape[0]), 2.0 * jnp.eye(shape[0])

    monkeypatch.setattr(inputs_mod, "cuda_ffi_available", lambda: False)
    monkeypatch.setattr(inputs_mod, "overlap_hcore_matrices", fake_core_builder)
    monkeypatch.delattr(inputs_mod, "CudaOneElectronBuilder", raising=False)

    inputs = build_rks_integral_inputs(
        atom=_h2_spec(),
        basis="sto-3g",
        config=RKSConfig(jk_backend="direct", direct_jk_engine="cuda"),
        integral_backend="jax",
        grid_ao_backend="jax",
        grids_level=0,
        max_l=1,
    )

    assert captured["basis"] is inputs.basis
    assert captured["backend"] == "jax"
    assert inputs.basis.precompute_eri_groups is True
    assert jnp.allclose(inputs.overlap, jnp.eye(inputs.basis.nao))
    assert jnp.allclose(inputs.hcore, 2.0 * jnp.eye(inputs.basis.nao))


def test_build_rks_integral_inputs_libcint_reuses_cached_grid_ao_bundle(monkeypatch):
    import td_graddft.scf.inputs as inputs_mod

    pytest.importorskip("pyscf")
    inputs_mod._GRID_AO_INPUT_CACHE.clear()
    counts = {"basis": 0, "grid": 0, "ao": 0}

    real_basis_from_molecule_spec = inputs_mod.basis_from_molecule_spec
    real_build_molecular_grid_from_spec = inputs_mod.build_molecular_grid_from_spec
    real_evaluate_cartesian_ao_with_derivatives = inputs_mod.evaluate_cartesian_ao_with_derivatives

    def _count_basis(*args, **kwargs):
        counts["basis"] += 1
        return real_basis_from_molecule_spec(*args, **kwargs)

    def _count_grid(*args, **kwargs):
        counts["grid"] += 1
        return real_build_molecular_grid_from_spec(*args, **kwargs)

    def _count_ao(*args, **kwargs):
        counts["ao"] += 1
        return real_evaluate_cartesian_ao_with_derivatives(*args, **kwargs)

    monkeypatch.setattr(inputs_mod, "basis_from_molecule_spec", _count_basis)
    monkeypatch.setattr(inputs_mod, "build_molecular_grid_from_spec", _count_grid)
    monkeypatch.setattr(inputs_mod, "evaluate_cartesian_ao_with_derivatives", _count_ao)

    spec = _h2_spec()
    cfg = RKSConfig(xc_spec="b3lyp", jk_backend="full")

    for _ in range(2):
        inputs = build_rks_integral_inputs(
            atom=spec,
            basis="sto-3g",
            config=cfg,
            integral_backend="libcint",
            grid_ao_backend="jax",
            grids_level=0,
            max_l=1,
        )
        assert inputs.ao.shape[0] == inputs.grid_weights.shape[0]

    assert counts == {"basis": 1, "grid": 1, "ao": 1}


def test_build_rks_integral_inputs_libcint_grid_ao_cache_misses_on_geometry_change(monkeypatch):
    import td_graddft.scf.inputs as inputs_mod

    pytest.importorskip("pyscf")
    inputs_mod._GRID_AO_INPUT_CACHE.clear()
    counts = {"basis": 0, "grid": 0, "ao": 0}

    real_basis_from_molecule_spec = inputs_mod.basis_from_molecule_spec
    real_build_molecular_grid_from_spec = inputs_mod.build_molecular_grid_from_spec
    real_evaluate_cartesian_ao_with_derivatives = inputs_mod.evaluate_cartesian_ao_with_derivatives

    def _count_basis(*args, **kwargs):
        counts["basis"] += 1
        return real_basis_from_molecule_spec(*args, **kwargs)

    def _count_grid(*args, **kwargs):
        counts["grid"] += 1
        return real_build_molecular_grid_from_spec(*args, **kwargs)

    def _count_ao(*args, **kwargs):
        counts["ao"] += 1
        return real_evaluate_cartesian_ao_with_derivatives(*args, **kwargs)

    monkeypatch.setattr(inputs_mod, "basis_from_molecule_spec", _count_basis)
    monkeypatch.setattr(inputs_mod, "build_molecular_grid_from_spec", _count_grid)
    monkeypatch.setattr(inputs_mod, "evaluate_cartesian_ao_with_derivatives", _count_ao)

    cfg = RKSConfig(xc_spec="b3lyp", jk_backend="full")
    specs = (
        _h2_spec(),
        parse_molecule_spec("H 0 0 0; H 0 0 0.80"),
    )

    for spec in specs:
        build_rks_integral_inputs(
            atom=spec,
            basis="sto-3g",
            config=cfg,
            integral_backend="libcint",
            grid_ao_backend="jax",
            grids_level=0,
            max_l=1,
        )

    assert counts == {"basis": 2, "grid": 2, "ao": 2}


def test_build_rks_integral_inputs_jax_skips_laplacian_for_gga(monkeypatch):
    import td_graddft.scf.inputs as inputs_mod

    original_eval_ao = inputs_mod.evaluate_cartesian_ao

    def _guard_eval_ao(*args, deriv=0, **kwargs):
        if deriv == 2:
            raise AssertionError("GGA/PBE0 RKS input construction should not build AO laplacians.")
        return original_eval_ao(*args, deriv=deriv, **kwargs)

    monkeypatch.setattr(inputs_mod, "evaluate_cartesian_ao", _guard_eval_ao)

    inputs = build_rks_integral_inputs(
        atom=_h2_spec(),
        basis="sto-3g",
        config=RKSConfig(xc_spec="pbe0", jk_backend="direct"),
        integral_backend="jax",
        grid_ao_backend="jax",
        grids_level=0,
        max_l=1,
    )

    assert inputs.ao_laplacian is None
    assert inputs.ao.shape[0] == inputs.grid_weights.shape[0]


def test_build_rks_integral_inputs_can_skip_dipole_integrals(monkeypatch):
    import td_graddft.scf.inputs as inputs_mod

    def _fail_dipole(*args, **kwargs):
        raise AssertionError("Ground-state-only input construction should be able to skip dipoles.")

    monkeypatch.setattr(inputs_mod, "dipole_matrix", _fail_dipole)

    inputs = build_rks_integral_inputs(
        atom=_h2_spec(),
        basis="sto-3g",
        config=RKSConfig(jk_backend="direct"),
        integral_backend="jax",
        grid_ao_backend="jax",
        grids_level=0,
        max_l=1,
        include_dipole_integrals=False,
    )

    assert inputs.dipole_integrals is None
    assert inputs.overlap.shape == inputs.hcore.shape


def test_build_rks_integral_inputs_rejects_pyscf_grid_ao_backend():
    with pytest.raises(ValueError, match="Only grid_ao_backend='jax'"):
        build_rks_integral_inputs(
            atom=_h2_spec(),
            basis="sto-3g",
            config=RKSConfig(jk_backend="direct"),
            integral_backend="jax",
            grid_ao_backend="pyscf",
            grids_level=0,
            max_l=1,
            include_dipole_integrals=False,
        )


def test_build_uks_integral_inputs_rejects_pyscf_grid_ao_backend():
    with pytest.raises(ValueError, match="Only grid_ao_backend='jax'"):
        build_uks_integral_inputs(
            atom=_h_atom_spec(),
            basis="sto-3g",
            config=UKSConfig(xc_spec="hf"),
            integral_backend="jax",
            grid_ao_backend="pyscf",
            grids_level=0,
            max_l=0,
        )


def test_build_uks_integral_inputs_full_backend_uses_full_eri_and_spin_counts():
    inputs = build_uks_integral_inputs(
        atom=_h_atom_spec(),
        basis="sto-3g",
        config=UKSConfig(xc_spec="hf"),
        integral_backend="jax",
        grid_ao_backend="jax",
        grids_level=0,
        max_l=0,
    )

    assert inputs.overlap.shape == inputs.hcore.shape
    assert inputs.eri.shape == inputs.overlap.shape * 2
    assert inputs.nalpha == 1
    assert inputs.nbeta == 0
    assert inputs.ao.shape[0] == inputs.grid_weights.shape[0]
    assert inputs.dipole_integrals.shape[0] == 3
    assert jnp.asarray(inputs.nuclear_repulsion).shape == ()
