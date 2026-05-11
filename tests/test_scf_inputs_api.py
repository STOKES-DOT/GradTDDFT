from __future__ import annotations

import jax.numpy as jnp
import pytest

from td_graddft.data.molecule import parse_molecule_spec
from td_graddft.scf import RKSConfig, UKSConfig
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

    class FakeCudaOneElectronBuilder:
        def __init__(self, basis):
            self.basis = basis

        def build_overlap_hcore(self):
            shape = (self.basis.nao, self.basis.nao)
            return jnp.eye(shape[0]), 2.0 * jnp.eye(shape[0])

    monkeypatch.setattr(inputs_mod, "CudaOneElectronBuilder", FakeCudaOneElectronBuilder)

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


def test_build_rks_integral_inputs_libcint_nontraced_spec_uses_single_pyscf_mol(monkeypatch):
    import td_graddft.scf.inputs as inputs_mod

    pytest.importorskip("pyscf")
    from pyscf import dft

    monkeypatch.setattr(inputs_mod, "cuda_ffi_available", lambda: True)

    def _fail_traceable_one_electron(*args, **kwargs):
        raise AssertionError("Non-traced libcint inputs should use the PySCF mol fast path.")

    def _fail_pyscf_grid(*args, **kwargs):
        raise AssertionError("libcint inputs must use TD-GradDFT grid/AO, not PySCF grids.")

    monkeypatch.setattr(inputs_mod, "libcint_int1e_with_coords", _fail_traceable_one_electron)
    monkeypatch.setattr(dft.gen_grid, "Grids", _fail_pyscf_grid)

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


def test_build_rks_integral_inputs_libcint_hybrid_does_not_use_pyscf_minao_guess(monkeypatch):
    pytest.importorskip("pyscf")
    from pyscf import dft

    rks_calls = []

    def _track_pyscf_rks(*args, **kwargs):
        rks_calls.append((args, kwargs))
        raise AssertionError("hybrid libcint inputs must not call PySCF minao initial guess.")

    monkeypatch.setattr(dft, "RKS", _track_pyscf_rks)

    inputs = build_rks_integral_inputs(
        atom=_h2_spec(),
        basis="sto-3g",
        config=RKSConfig(xc_spec="pbe0", jk_backend="full"),
        integral_backend="libcint",
        grid_ao_backend="jax",
        grids_level=0,
        max_l=1,
    )

    assert rks_calls == []
    assert inputs.init_mo_coeff is None
    assert inputs.init_mo_occ is None


def test_build_rks_integral_inputs_jax_uses_fused_core_matrices(monkeypatch):
    import td_graddft.scf.inputs as inputs_mod

    def _fail_separate_core_builder(*args, **kwargs):
        raise AssertionError("JAX RKS input construction should use fused overlap/hcore matrices.")

    monkeypatch.setattr(inputs_mod, "overlap_matrix", _fail_separate_core_builder)
    monkeypatch.setattr(inputs_mod, "build_hcore", _fail_separate_core_builder)

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


def test_build_rks_integral_inputs_cuda_direct_jax_uses_cuda_one_electron(monkeypatch):
    import td_graddft.scf.inputs as inputs_mod

    monkeypatch.setattr(inputs_mod, "cuda_ffi_available", lambda: True)
    captured = {}

    def _fail_jax_core_builder(*args, **kwargs):
        raise AssertionError("CUDA direct JAX input should use CUDA one-electron FFI.")

    class FakeCudaOneElectronBuilder:
        def __init__(self, basis):
            captured["basis"] = basis

        def build_overlap_hcore(self):
            shape = (captured["basis"].nao, captured["basis"].nao)
            return jnp.eye(shape[0]), 2.0 * jnp.eye(shape[0])

    monkeypatch.setattr(inputs_mod, "overlap_hcore_matrices", _fail_jax_core_builder)
    monkeypatch.setattr(inputs_mod, "CudaOneElectronBuilder", FakeCudaOneElectronBuilder)

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
    assert jnp.allclose(inputs.overlap, jnp.eye(inputs.basis.nao))
    assert jnp.allclose(inputs.hcore, 2.0 * jnp.eye(inputs.basis.nao))


def test_build_rks_integral_inputs_cuda_direct_jax_falls_back_without_cuda(monkeypatch):
    import td_graddft.scf.inputs as inputs_mod

    captured = {}

    def fake_core_builder(basis):
        captured["basis"] = basis
        shape = (basis.nao, basis.nao)
        return jnp.eye(shape[0]), 2.0 * jnp.eye(shape[0])

    class FailingCudaOneElectronBuilder:
        def __init__(self, basis):
            raise AssertionError("CUDA one-electron builder should not be used without CUDA.")

    monkeypatch.setattr(inputs_mod, "cuda_ffi_available", lambda: False)
    monkeypatch.setattr(inputs_mod, "overlap_hcore_matrices", fake_core_builder)
    monkeypatch.setattr(inputs_mod, "CudaOneElectronBuilder", FailingCudaOneElectronBuilder)

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
