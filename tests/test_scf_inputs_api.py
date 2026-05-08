from __future__ import annotations

import jax.numpy as jnp
import pytest

from td_graddft.data.molecule import parse_molecule_spec
from td_graddft.scf import RKSConfig, UKSConfig
from td_graddft.scf.inputs import build_rks_integral_inputs, build_uks_integral_inputs


def _h2_spec():
    return parse_molecule_spec("H 0 0 0; H 0 0 0.74")


def _h_atom_spec():
    return parse_molecule_spec("H 0 0 0", spin=1)


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
    monkeypatch.setattr(inputs_mod, "cuda_ffi_available", lambda: True)

    def _fail_traceable_one_electron(*args, **kwargs):
        raise AssertionError("Non-traced libcint inputs should use the PySCF mol fast path.")

    monkeypatch.setattr(inputs_mod, "libcint_int1e_with_coords", _fail_traceable_one_electron)

    inputs = build_rks_integral_inputs(
        atom=_h2_spec(),
        basis="sto-3g",
        config=RKSConfig(jk_backend="direct", direct_jk_engine="cuda"),
        integral_backend="libcint",
        grid_ao_backend="pyscf",
        grids_level=0,
        max_l=1,
    )

    assert inputs.nelectron == 2
    assert inputs.eri_pair_matrix is None
    assert inputs.grid_ao_backend == "pyscf"


def test_mo_coeff_guess_from_density_matrix_uses_host_linear_algebra(monkeypatch):
    import td_graddft.scf.inputs as inputs_mod

    def _fail_device_eigh(*args, **kwargs):
        raise AssertionError("Initial PySCF density guess conversion should not compile XLA eigh.")

    monkeypatch.setattr(inputs_mod.jnp.linalg, "eigh", _fail_device_eigh)

    coeff = inputs_mod._mo_coeff_guess_from_density_matrix(
        jnp.eye(2),
        jnp.eye(2),
    )

    assert coeff.shape == (2, 2)


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


def test_build_rks_integral_inputs_pyscf_grid_accepts_molecule_spec():
    inputs = build_rks_integral_inputs(
        atom=_h2_spec(),
        basis="sto-3g",
        config=RKSConfig(jk_backend="direct"),
        integral_backend="jax",
        grid_ao_backend="pyscf",
        grids_level=0,
        max_l=1,
        include_dipole_integrals=False,
    )

    assert inputs.grid_ao_backend == "pyscf"
    assert inputs.ao.shape[0] == inputs.grid_weights.shape[0]
    assert inputs.direct_basis is inputs.basis


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
