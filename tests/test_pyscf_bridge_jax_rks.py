import numpy as np
import pytest
import types

from td_graddft.reference_legacy import (
    restricted_reference_from_pyscf_spec_with_jax_rks,
    restricted_reference_from_pyscf_with_jax_rks,
)
from td_graddft.reference import _charge_center, restricted_reference_from_spec_with_jax_rks
from td_graddft.scf import RKSConfig
from td_graddft.scf.rks import TraceableRKSResult
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


def test_restricted_reference_from_pyscf_with_jax_rks_shapes_and_target_energy():
    _pyscf_or_skip()
    mf = _make_h2_b3lyp_mf()

    ref = restricted_reference_from_pyscf_with_jax_rks(
        mf,
        max_l=1,
        rks_config=RKSConfig(max_cycle=20, conv_tol=1e-8, conv_tol_density=1e-6),
        energy_target=float(mf.e_tot),
    )

    nao = ref.mo_coeff.shape[-1]
    assert ref.ao.shape[1] == nao
    assert ref.h1e.shape == (nao, nao)
    assert np.asarray(ref.rep_tensor).size == 0
    assert ref.eri_ovov is not None
    assert ref.eri_ovvo is not None
    assert ref.eri_oovv is not None
    assert ref.mo_coeff.shape[0] == 2
    assert ref.mo_occ.shape[0] == 2
    assert ref.rdm1.shape == (2, nao, nao)
    assert np.isclose(float(ref.mf_energy), float(mf.e_tot), atol=1e-12, rtol=1e-12)
    assert np.isfinite(float(ref.exact_exchange_fraction))


def test_restricted_reference_from_pyscf_with_jax_rks_jax_grid_ao_matches_pyscf_ao():
    _pyscf_or_skip()
    from pyscf.dft import numint

    mf = _make_h2_b3lyp_mf()
    ref = restricted_reference_from_pyscf_with_jax_rks(
        mf,
        max_l=1,
        rks_config=RKSConfig(max_cycle=20, conv_tol=1e-8, conv_tol_density=1e-6),
        energy_target=float(mf.e_tot),
        grid_ao_backend="jax",
    )

    ao_ref = np.asarray(numint.eval_ao(mf.mol, mf.grids.coords, deriv=0), dtype=float)
    ao1_ref = np.asarray(numint.eval_ao(mf.mol, mf.grids.coords, deriv=1), dtype=float)
    ao = np.asarray(ref.ao, dtype=float)
    ao1 = np.asarray(ref.ao_deriv1, dtype=float)

    assert np.allclose(ao, ao_ref, atol=2e-7, rtol=2e-6)
    assert np.allclose(ao1, ao1_ref, atol=3e-6, rtol=2e-5)


def test_workflow_reference_stage_accepts_jax_rks_backend():
    _pyscf_or_skip()
    mf = _make_h2_b3lyp_mf()

    simulation = SimulationConfig(
        nstates=1,
        scf_backend="jax_rks",
        jax_grid_ao_backend="jax",
        jax_basis_max_l=1,
        jax_rks_max_cycle=20,
        jax_rks_conv_tol=1e-8,
        jax_rks_conv_tol_density=1e-6,
    )
    reference = run_reference(
        mf,
        scf_elapsed_s=0.0,
        simulation=simulation,
    )

    assert reference.nstates == 1
    assert reference.energies_au.shape == (1,)
    assert reference.oscillator_strengths.shape == (1,)
    assert reference.molecule.rep_tensor.ndim == 4


def test_restricted_reference_from_pyscf_spec_with_jax_rks_matches_water_pyscf_energy():
    _pyscf_or_skip()
    from pyscf import dft, gto

    atom = """
    O  0.000000  0.000000  0.117790
    H  0.000000  0.755453 -0.471161
    H  0.000000 -0.755453 -0.471161
    """
    mol = gto.M(
        atom=atom,
        basis="sto-3g",
        unit="Angstrom",
        spin=0,
        cart=True,
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = "pbe"
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF SCF did not converge for water test setup.")

    ref = restricted_reference_from_pyscf_spec_with_jax_rks(
        atom=atom,
        basis="sto-3g",
        unit="Angstrom",
        xc_spec="pbe",
        spin=0,
        charge=0,
        cart=True,
        grids_level=0,
        max_l=1,
        rks_config=RKSConfig(
            xc_spec="pbe",
            max_cycle=50,
            conv_tol=1e-9,
            conv_tol_density=1e-7,
            damping=0.15,
            density_floor=1e-12,
            potential_clip=20.0,
        ),
        grid_ao_backend="jax",
    )

    assert np.isclose(float(ref.mf_energy), float(mf.e_tot), atol=2e-5, rtol=2e-7)
    with mol.with_common_orig(_charge_center(mol)):
        dipole_ref = np.asarray(mol.intor_symmetric("int1e_r", comp=3), dtype=float)
    dipole = np.asarray(ref.dipole_integrals, dtype=float)
    assert np.allclose(dipole, dipole_ref, atol=2e-6, rtol=2e-6)


def test_restricted_reference_from_spec_with_jax_rks_direct_backend_matches_water_pyscf_energy():
    _pyscf_or_skip()
    from pyscf import dft, gto

    atom = """
    O  0.000000  0.000000  0.117790
    H  0.000000  0.755453 -0.471161
    H  0.000000 -0.755453 -0.471161
    """
    mol = gto.M(
        atom=atom,
        basis="sto-3g",
        unit="Angstrom",
        spin=0,
        cart=True,
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = "pbe0"
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF SCF did not converge for direct-RKS water test setup.")

    ref = restricted_reference_from_spec_with_jax_rks(
        atom=atom,
        basis="sto-3g",
        unit="Angstrom",
        xc_spec="pbe0",
        spin=0,
        charge=0,
        cart=True,
        grids_level=0,
        max_l=1,
        rks_config=RKSConfig(
            xc_spec="pbe0",
            max_cycle=50,
            conv_tol=1e-9,
            conv_tol_density=1e-7,
            damping=0.15,
            density_floor=1e-12,
            potential_clip=20.0,
            jk_backend="direct",
            direct_scf_tol=0.0,
        ),
        grid_ao_backend="jax",
        integral_backend="jax",
    )

    assert np.asarray(ref.rep_tensor).size == 0
    assert ref.eri_ovov is not None
    assert ref.eri_ovvo is not None
    assert ref.eri_oovv is not None
    assert np.isclose(float(ref.mf_energy), float(mf.e_tot), atol=2e-4, rtol=2e-6)


def test_cuda_direct_reference_skips_response_eri_precompute(monkeypatch):
    _pyscf_or_skip()
    from dataclasses import replace

    from td_graddft.data.integrals import eri_pair_matrix_packed
    from td_graddft.scf.packed_eri import build_jk_from_eri_pair_matrix
    import td_graddft.scf.rks as rks_mod
    import td_graddft.reference as reference_mod

    class FakeCudaDirectJKBuilder:
        def __init__(self, basis, **kwargs):
            del kwargs
            basis_with_groups = replace(
                basis,
                precompute_eri_groups=True,
                quartet_groups=(),
                shell_quartet_groups=(),
            )
            self.pair = eri_pair_matrix_packed(basis_with_groups)

        def build_jk(self, density, **kwargs):
            del kwargs
            return build_jk_from_eri_pair_matrix(self.pair, density)

        def build_jk_with_joltqc_basis_data(self, density, basis_data):
            del basis_data
            return build_jk_from_eri_pair_matrix(self.pair, density)

        def build_eri_pair_matrix(self):
            return self.pair

        def build_jk_from_eri_pair_matrix(self, pair, density):
            return build_jk_from_eri_pair_matrix(pair, density)

    monkeypatch.setattr(rks_mod, "CudaDirectJKBuilder", FakeCudaDirectJKBuilder)
    monkeypatch.setattr(reference_mod, "CudaDirectJKBuilder", FakeCudaDirectJKBuilder)
    monkeypatch.setattr(rks_mod, "cuda_ffi_available", lambda: True)
    monkeypatch.setattr(reference_mod, "cuda_ffi_available", lambda: True)

    ref = restricted_reference_from_spec_with_jax_rks(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        xc_spec="pbe0",
        spin=0,
        charge=0,
        cart=True,
        grids_level=0,
        max_l=1,
        rks_config=RKSConfig(
            xc_spec="pbe0",
            max_cycle=12,
            conv_tol=1e-8,
            conv_tol_density=1e-6,
            jk_backend="direct",
            direct_jk_engine="cuda",
            direct_scf_incremental=False,
        ),
        grid_ao_backend="jax",
        integral_backend="libcint",
    )

    assert np.asarray(ref.rep_tensor).size == 0
    assert ref.eri_ovov is None
    assert ref.eri_ovvo is None
    assert ref.eri_oovv is None
    assert np.isfinite(float(ref.mf_energy))


def test_cached_cuda_direct_runner_uses_prebuilt_runtime_builder(monkeypatch):
    import jax.numpy as jnp

    import td_graddft.reference as reference_mod
    from td_graddft.data.basis import basis_from_spec

    basis = basis_from_spec("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    reference_mod._CUDA_DIRECT_RKS_JIT_CACHE.clear()
    if hasattr(reference_mod, "_CUDA_DIRECT_JK_BUILDER_CACHE"):
        reference_mod._CUDA_DIRECT_JK_BUILDER_CACHE.clear()

    constructed_builders = []

    class FakeJoltQCBuilder:
        def __init__(self, direct_basis, **kwargs):
            self.direct_basis = direct_basis
            self.kwargs = kwargs
            constructed_builders.append(self)

    class FakeJitted:
        def __init__(self, fn):
            self.fn = fn

        def __call__(self, *args):
            return self.fn(*args)

        def lower(self, *args):
            del args
            return types.SimpleNamespace(compile=lambda: self)

    captured = {}

    def fake_run_rks_from_integrals_traceable(**kwargs):
        captured["direct_cuda_jk_builder"] = kwargs.get("direct_cuda_jk_builder")
        zero_matrix = jnp.zeros((basis.nao, basis.nao), dtype=jnp.float64)
        zero_vector = jnp.zeros((basis.nao,), dtype=jnp.float64)
        return TraceableRKSResult(
            converged=jnp.asarray(True),
            total_energy=jnp.asarray(0.0, dtype=jnp.float64),
            electronic_energy=jnp.asarray(0.0, dtype=jnp.float64),
            nuclear_repulsion=jnp.asarray(0.0, dtype=jnp.float64),
            xc_energy=jnp.asarray(0.0, dtype=jnp.float64),
            exact_exchange_fraction=jnp.asarray(1.0, dtype=jnp.float64),
            mo_energy=zero_vector,
            mo_coeff=zero_matrix,
            mo_occ=zero_vector,
            density_matrix=zero_matrix,
            fock_matrix=zero_matrix,
            overlap_matrix=zero_matrix,
            hcore_matrix=zero_matrix,
            cycles=jnp.asarray(0),
        )

    monkeypatch.setattr(reference_mod, "CudaDirectJKBuilder", FakeJoltQCBuilder, raising=False)
    monkeypatch.setattr(reference_mod.jax, "jit", lambda fn: FakeJitted(fn))
    monkeypatch.setattr(
        reference_mod,
        "run_rks_from_integrals_traceable",
        fake_run_rks_from_integrals_traceable,
    )

    scf_inputs = types.SimpleNamespace(
        direct_basis=basis,
        init_mo_coeff=None,
        init_mo_occ=None,
        init_mo_energy=None,
        nelectron=2,
        overlap=jnp.eye(basis.nao, dtype=jnp.float64),
        hcore=jnp.eye(basis.nao, dtype=jnp.float64),
        nuclear_repulsion=jnp.asarray(0.0, dtype=jnp.float64),
        ao=jnp.zeros((1, basis.nao), dtype=jnp.float64),
        ao_deriv1=jnp.zeros((4, 1, basis.nao), dtype=jnp.float64),
        grid_weights=jnp.ones((1,), dtype=jnp.float64),
    )
    cfg = RKSConfig(
        xc_spec="hf",
        iteration_backend="lax",
        jk_backend="direct",
        direct_jk_engine="cuda",
    )

    runner = reference_mod._cached_cuda_direct_rks_runner(scf_inputs, cfg)
    result = runner(*reference_mod._cuda_direct_rks_args(scf_inputs))

    assert result.total_energy == 0.0
    assert len(constructed_builders) == 1
    assert captured["direct_cuda_jk_builder"] is constructed_builders[0]
    assert constructed_builders[0].kwargs["include_pair_metadata"] is True


def test_cached_cuda_direct_jk_builder_keeps_pair_metadata_for_runtime_mapping(monkeypatch):
    import td_graddft.reference as reference_mod
    from td_graddft.data.basis import basis_from_spec

    basis = basis_from_spec("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    calls = []

    class FakeBuilder:
        def __init__(self, basis_arg, **kwargs):
            assert basis_arg is basis
            calls.append(kwargs)

    monkeypatch.setattr(reference_mod, "CudaDirectJKBuilder", FakeBuilder)
    reference_mod._CUDA_DIRECT_JK_BUILDER_CACHE.clear()

    unscreened_config = RKSConfig(
        xc_spec="pbe0",
        jk_backend="direct",
        direct_jk_engine="cuda",
        iteration_backend="lax",
        direct_scf_tol=0.0,
    )
    screened_config = RKSConfig(
        xc_spec="pbe0",
        jk_backend="direct",
        direct_jk_engine="cuda",
        iteration_backend="lax",
        direct_scf_tol=1.0e-12,
    )

    reference_mod._cached_cuda_direct_jk_builder(basis, unscreened_config)
    reference_mod._cached_cuda_direct_jk_builder(basis, screened_config)

    assert calls[0]["include_pair_metadata"] is True
    assert calls[1]["include_pair_metadata"] is True


def test_cuda_direct_python_reference_uses_cached_runtime_builder(monkeypatch):
    import jax.numpy as jnp

    import td_graddft.reference as reference_mod
    from td_graddft.data.basis import basis_from_spec
    from td_graddft.scf.inputs import RKSIntegralInputs
    from td_graddft.scf.rks import RKSResult

    basis = basis_from_spec("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    nao = int(basis.nao)
    cached_builder = object()
    captured = {}

    scf_inputs = RKSIntegralInputs(
        basis=basis,
        overlap=jnp.eye(nao, dtype=jnp.float64),
        hcore=jnp.eye(nao, dtype=jnp.float64),
        eri=None,
        eri_pair_matrix=None,
        df_factors=None,
        direct_basis=basis,
        nelectron=2,
        nuclear_repulsion=0.0,
        coords=jnp.zeros((1, 3), dtype=jnp.float64),
        grid_weights=jnp.ones((1,), dtype=jnp.float64),
        ao=jnp.zeros((1, nao), dtype=jnp.float64),
        ao_deriv1=jnp.zeros((4, 1, nao), dtype=jnp.float64),
        ao_laplacian=None,
        dipole_integrals=None,
        integral_backend="libcint",
        grid_ao_backend="pyscf",
    )

    def fake_run_rks_from_integrals(**kwargs):
        captured.update(kwargs)
        density = jnp.zeros((nao, nao), dtype=jnp.float64)
        mo_occ = jnp.zeros((nao,), dtype=jnp.float64).at[0].set(2.0)
        return RKSResult(
            converged=True,
            total_energy=-1.0,
            electronic_energy=-1.0,
            nuclear_repulsion=0.0,
            xc_energy=0.0,
            exact_exchange_fraction=1.0,
            mo_energy=jnp.zeros((nao,), dtype=jnp.float64),
            mo_coeff=jnp.eye(nao, dtype=jnp.float64),
            mo_occ=mo_occ,
            density_matrix=density,
            fock_matrix=density,
            overlap_matrix=kwargs["overlap"],
            hcore_matrix=kwargs["hcore"],
            cycles=1,
        )

    monkeypatch.setattr(reference_mod, "cuda_ffi_available", lambda: True)
    monkeypatch.setattr(reference_mod, "build_rks_integral_inputs", lambda **kwargs: scf_inputs)
    monkeypatch.setattr(reference_mod, "_cached_cuda_direct_jk_builder", lambda basis_arg, cfg: cached_builder)
    monkeypatch.setattr(reference_mod, "run_rks_from_integrals", fake_run_rks_from_integrals)

    ref = reference_mod.restricted_reference_from_spec_with_jax_rks(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        xc_spec="hf",
        spin=0,
        charge=0,
        cart=True,
        grids_level=0,
        max_l=1,
        rks_config=RKSConfig(
            xc_spec="hf",
            jk_backend="direct",
            direct_jk_engine="cuda",
            iteration_backend="python",
        ),
        grid_ao_backend="pyscf",
        integral_backend="libcint",
    )

    assert float(ref.mf_energy) == -1.0
    assert captured["direct_cuda_jk_builder"] is cached_builder
    assert "direct_joltqc_basis_data" not in captured


def test_cuda_direct_input_cache_key_is_independent_of_iteration_backend():
    import td_graddft.reference as reference_mod
    from td_graddft.data.molecule import parse_molecule_spec

    atom = parse_molecule_spec("H 0 0 0; H 0 0 0.74", unit="Angstrom")
    base = dict(
        atom=atom,
        basis="sto-3g",
        xc_spec="hf",
        unit="Angstrom",
        charge=0,
        spin=0,
        cart=True,
        grids_level=0,
        max_l=1,
        grid_ao_backend="pyscf",
        integral_backend="libcint",
        libcint_geometry_grad_policy="analytic",
        include_dipole_integrals=False,
        precompile_eri=False,
        precompile_eri_chunk_size=512,
        verbose=0,
        mol_kwargs={},
    )
    python_key = reference_mod._cuda_direct_reference_inputs_cache_key(
        **base,
        config=RKSConfig(
            xc_spec="hf",
            jk_backend="direct",
            direct_jk_engine="cuda",
            iteration_backend="python",
        ),
    )
    lax_key = reference_mod._cuda_direct_reference_inputs_cache_key(
        **base,
        config=RKSConfig(
            xc_spec="hf",
            jk_backend="direct",
            direct_jk_engine="cuda",
            iteration_backend="lax",
        ),
    )

    assert python_key == lax_key


def test_precompiled_cuda_direct_runner_is_reused_for_execution(monkeypatch):
    import td_graddft.reference as reference_mod

    scf_inputs = types.SimpleNamespace(name="inputs")
    config = RKSConfig(
        xc_spec="pbe0",
        jk_backend="direct",
        direct_jk_engine="cuda",
        iteration_backend="lax",
    )
    calls = {"compile": 0, "compiled": 0, "runner": 0}
    args = [np.asarray([1.0, 2.0], dtype=np.float64)]

    class FakeLowered:
        def compile(self):
            calls["compile"] += 1

            def compiled(*call_args):
                calls["compiled"] += 1
                assert call_args == tuple(args)
                return "compiled-result"

            return compiled

    class FakeRunner:
        def lower(self, *lower_args):
            assert lower_args == tuple(args)
            return FakeLowered()

        def __call__(self, *call_args):
            calls["runner"] += 1
            return "runner-result"

    monkeypatch.setattr(
        reference_mod,
        "_cached_cuda_direct_rks_runner",
        lambda current_inputs, current_config: FakeRunner(),
    )
    monkeypatch.setattr(reference_mod, "_cuda_direct_rks_args", lambda current_inputs: args)
    monkeypatch.setattr(
        reference_mod,
        "_cuda_direct_rks_runner_cache_key",
        lambda current_inputs, current_config: ("runner-key",),
        raising=False,
    )
    reference_mod._CUDA_DIRECT_RKS_COMPILED_CACHE.clear()

    compiled = reference_mod.precompile_cuda_direct_rks_inputs(scf_inputs, config)
    result = reference_mod._run_cached_cuda_direct_rks(scf_inputs, config)

    assert compiled(*args) == "compiled-result"
    assert result == "compiled-result"
    assert calls == {"compile": 1, "compiled": 2, "runner": 0}


def test_cuda_direct_precompile_reuses_existing_compiled_executable(monkeypatch):
    import td_graddft.reference as reference_mod

    scf_inputs = types.SimpleNamespace(name="inputs")
    config = RKSConfig(
        xc_spec="pbe0",
        jk_backend="direct",
        direct_jk_engine="cuda",
        iteration_backend="lax",
    )
    calls = {"factory": 0, "compile": 0}
    args = [np.asarray([1.0, 2.0], dtype=np.float64)]

    class FakeLowered:
        def compile(self):
            calls["compile"] += 1

            def compiled(*call_args):
                assert call_args == tuple(args)
                return "compiled-result"

            return compiled

    class FakeRunner:
        def lower(self, *lower_args):
            assert lower_args == tuple(args)
            return FakeLowered()

    def fake_cached_runner(current_inputs, current_config):
        assert current_inputs is scf_inputs
        assert current_config is config
        calls["factory"] += 1
        return FakeRunner()

    monkeypatch.setattr(reference_mod, "_cached_cuda_direct_rks_runner", fake_cached_runner)
    monkeypatch.setattr(reference_mod, "_cuda_direct_rks_args", lambda current_inputs: args)
    monkeypatch.setattr(
        reference_mod,
        "_cuda_direct_rks_runner_cache_key",
        lambda current_inputs, current_config: ("runner-key",),
        raising=False,
    )
    reference_mod._CUDA_DIRECT_RKS_COMPILED_CACHE.clear()

    first = reference_mod.precompile_cuda_direct_rks_inputs(scf_inputs, config)
    second = reference_mod.precompile_cuda_direct_rks_inputs(scf_inputs, config)

    assert first is second
    assert second(*args) == "compiled-result"
    assert calls == {"factory": 1, "compile": 1}


def test_cuda_direct_reference_precompile_reuses_cached_inputs(monkeypatch):
    import td_graddft.reference as reference_mod

    cached_inputs = types.SimpleNamespace(name="cached-inputs")
    config = RKSConfig(
        xc_spec="pbe0",
        jk_backend="direct",
        direct_jk_engine="cuda",
        iteration_backend="lax",
    )
    reference_mod._CUDA_DIRECT_RKS_INPUT_CACHE.clear()
    reference_mod._CUDA_DIRECT_RKS_INPUT_CACHE[("input-key",)] = cached_inputs

    def fail_build_inputs(**kwargs):
        raise AssertionError("cached precompile should not rebuild SCF inputs")

    def fake_precompile(current_inputs, current_config):
        assert current_inputs is cached_inputs
        assert current_config is config
        return "compiled-result"

    monkeypatch.setattr(
        reference_mod,
        "_cuda_direct_reference_inputs_cache_key",
        lambda **kwargs: ("input-key",),
    )
    monkeypatch.setattr(reference_mod, "build_rks_integral_inputs", fail_build_inputs)
    monkeypatch.setattr(reference_mod, "precompile_cuda_direct_rks_signature", fake_precompile)

    result = reference_mod.precompile_restricted_cuda_direct_rks_reference(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        xc_spec="pbe0",
        rks_config=config,
        grid_ao_backend="pyscf",
        integral_backend="libcint",
        include_dipole_integrals=False,
    )

    assert result == "compiled-result"


def test_cuda_direct_grid_bucket_padding_preserves_active_grid_values():
    import jax.numpy as jnp

    import td_graddft.reference as reference_mod
    from td_graddft.data.basis import basis_from_spec
    from td_graddft.scf.inputs import RKSIntegralInputs

    basis = basis_from_spec("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    nao = int(basis.nao)
    scf_inputs = RKSIntegralInputs(
        basis=basis,
        overlap=jnp.eye(nao, dtype=jnp.float64),
        hcore=jnp.eye(nao, dtype=jnp.float64),
        eri=None,
        eri_pair_matrix=None,
        df_factors=None,
        direct_basis=basis,
        nelectron=2,
        nuclear_repulsion=0.0,
        coords=jnp.asarray(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            dtype=jnp.float64,
        ),
        grid_weights=jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64),
        ao=jnp.ones((3, nao), dtype=jnp.float64),
        ao_deriv1=jnp.ones((4, 3, nao), dtype=jnp.float64),
        ao_laplacian=None,
        dipole_integrals=None,
        integral_backend="libcint",
        grid_ao_backend="pyscf",
    )

    padded = reference_mod._bucket_cuda_direct_rks_grid_inputs(
        scf_inputs,
        bucket_size=4,
    )

    assert padded.grid_weights.shape == (4,)
    assert padded.coords.shape == (4, 3)
    assert padded.ao.shape == (4, nao)
    assert padded.ao_deriv1.shape == (4, 4, nao)
    np.testing.assert_allclose(np.asarray(padded.grid_weights[:3]), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(np.asarray(padded.coords[:3]), np.asarray(scf_inputs.coords))
    np.testing.assert_allclose(np.asarray(padded.ao[:3]), np.asarray(scf_inputs.ao))
    np.testing.assert_allclose(
        np.asarray(padded.ao_deriv1[:, :3]),
        np.asarray(scf_inputs.ao_deriv1),
    )
    assert float(padded.grid_weights[3]) == 0.0
    np.testing.assert_allclose(np.asarray(padded.coords[3]), np.asarray(scf_inputs.coords[0]))
    np.testing.assert_allclose(np.asarray(padded.ao[3]), np.asarray(scf_inputs.ao[0]))
    np.testing.assert_allclose(
        np.asarray(padded.ao_deriv1[:, 3]),
        np.asarray(scf_inputs.ao_deriv1[:, 0]),
    )


def test_cuda_direct_signature_precompile_uses_dummy_args_but_reuses_real_signature(monkeypatch):
    import td_graddft.reference as reference_mod

    scf_inputs = types.SimpleNamespace(name="inputs")
    config = RKSConfig(
        xc_spec="pbe0",
        jk_backend="direct",
        direct_jk_engine="cuda",
        iteration_backend="lax",
    )
    real_args = [np.asarray([7.0, 8.0], dtype=np.float64)]
    calls = {"compile": 0, "compiled": 0}

    class FakeLowered:
        def compile(self):
            calls["compile"] += 1

            def compiled(*call_args):
                calls["compiled"] += 1
                assert call_args == tuple(real_args)
                return "compiled-result"

            return compiled

    class FakeRunner:
        def lower(self, *lower_args):
            assert len(lower_args) == 1
            np.testing.assert_allclose(lower_args[0], np.zeros_like(real_args[0]))
            return FakeLowered()

    monkeypatch.setattr(
        reference_mod,
        "_cached_cuda_direct_rks_runner",
        lambda current_inputs, current_config: FakeRunner(),
    )
    monkeypatch.setattr(
        reference_mod,
        "_cuda_direct_rks_args",
        lambda current_inputs: real_args,
    )
    monkeypatch.setattr(
        reference_mod,
        "_cuda_direct_rks_runner_cache_key",
        lambda current_inputs, current_config: ("runner-key",),
        raising=False,
    )
    reference_mod._CUDA_DIRECT_RKS_COMPILED_CACHE.clear()

    compiled = reference_mod.precompile_cuda_direct_rks_signature(scf_inputs, config)
    result = reference_mod._run_cached_cuda_direct_rks(scf_inputs, config)

    assert compiled is not None
    assert result == "compiled-result"
    assert calls == {"compile": 1, "compiled": 1}


def test_restricted_reference_from_pyscf_spec_with_jax_rks_matches_water_local_hfx():
    _pyscf_or_skip()
    from pyscf import dft, gto

    atom = """
    O  0.000000  0.000000  0.117790
    H  0.000000  0.755453 -0.471161
    H  0.000000 -0.755453 -0.471161
    """
    mol = gto.M(
        atom=atom,
        basis="sto-3g",
        unit="Angstrom",
        spin=0,
        cart=True,
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = "pbe"
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF SCF did not converge for water local-HFX test setup.")

    ref_jax = restricted_reference_from_pyscf_spec_with_jax_rks(
        atom=atom,
        basis="sto-3g",
        unit="Angstrom",
        xc_spec="pbe",
        spin=0,
        charge=0,
        cart=True,
        grids_level=0,
        max_l=1,
        rks_config=RKSConfig(
            xc_spec="pbe",
            max_cycle=50,
            conv_tol=1e-9,
            conv_tol_density=1e-7,
            damping=0.15,
            density_floor=1e-12,
            potential_clip=20.0,
        ),
        grid_ao_backend="jax",
        compute_local_hfx_features=True,
        compute_local_hfx_aux=True,
        hfx_omega_values=(0.0, 0.4),
        hfx_chunk_size=128,
    )

    assert ref_jax.hfx_local is not None
    assert ref_jax.hfx_nu is not None
    coords = np.asarray(ref_jax.grid.coords, dtype=float)
    ao = np.asarray(ref_jax.ao, dtype=float)
    dm_half = np.asarray(ref_jax.rdm1[0], dtype=float)
    e = ao @ dm_half

    nu_ref_list = []
    for omega in (0.0, 0.4):
        if omega == 0.0:
            nu_ref = np.asarray(mol.intor("int1e_grids_cart", hermi=1, grids=coords), dtype=float)
        else:
            with mol.with_range_coulomb(omega=omega):
                nu_ref = np.asarray(
                    mol.intor("int1e_grids_cart", hermi=1, grids=coords),
                    dtype=float,
                )
        nu_ref_list.append(nu_ref)
    nu_ref_stack = np.stack(nu_ref_list, axis=0)
    fxx_ref = np.einsum("wgbc,gc->wgb", nu_ref_stack, e, optimize=True)
    exx_ref = -0.5 * np.einsum("gq,wgq->wg", e, fxx_ref, optimize=True)
    hfx_ref = np.stack([exx_ref.T, exx_ref.T], axis=0)

    assert np.allclose(
        np.asarray(ref_jax.hfx_nu, dtype=float),
        nu_ref_stack,
        atol=3e-5,
        rtol=3e-5,
    )
    assert np.allclose(
        np.asarray(ref_jax.hfx_local, dtype=float),
        hfx_ref,
        atol=5e-5,
        rtol=5e-5,
    )


def test_restricted_reference_from_spec_with_jax_rks_can_precompile_eri(monkeypatch):
    calls: list[tuple[int, str, int]] = []

    def fake_precompile(basis, *, engine="auto", chunk_size=512):
        calls.append((basis.nao, str(engine), int(chunk_size)))
        return {"compiled_shell_signatures": 0, "compiled_batch_shapes": 0}

    monkeypatch.setattr("td_graddft.reference.precompile_eri_kernels", fake_precompile)

    ref = restricted_reference_from_spec_with_jax_rks(
        atom="""
        H 0.0 0.0 -0.35
        H 0.0 0.0  0.35
        """,
        basis="sto-3g",
        unit="Angstrom",
        xc_spec="pbe",
        spin=0,
        charge=0,
        cart=True,
        grids_level=0,
        max_l=1,
        rks_config=RKSConfig(
            xc_spec="pbe",
            max_cycle=20,
            conv_tol=1e-8,
            conv_tol_density=1e-6,
            damping=0.15,
            density_floor=1e-12,
            potential_clip=20.0,
        ),
        grid_ao_backend="jax",
        integral_backend="jax",
        precompile_eri=True,
        precompile_eri_chunk_size=64,
    )

    assert ref.h1e.ndim == 2
    assert calls
    assert calls[0][1] == "jit"
    assert calls[0][2] == 64


def test_restricted_reference_from_spec_with_jax_rks_libcint_matches_jax():
    _pyscf_or_skip()

    atom = """
    H 0.0 0.0 -0.35
    H 0.0 0.0  0.35
    """
    cfg = RKSConfig(
        xc_spec="pbe",
        max_cycle=40,
        conv_tol=1e-10,
        conv_tol_density=1e-8,
        damping=0.1,
        density_floor=1e-12,
        potential_clip=20.0,
    )
    ref_jax = restricted_reference_from_spec_with_jax_rks(
        atom=atom,
        basis="sto-3g",
        unit="Angstrom",
        xc_spec="pbe",
        spin=0,
        charge=0,
        cart=True,
        grids_level=0,
        max_l=1,
        rks_config=cfg,
        grid_ao_backend="jax",
        integral_backend="jax",
    )
    ref_libcint = restricted_reference_from_spec_with_jax_rks(
        atom=atom,
        basis="sto-3g",
        unit="Angstrom",
        xc_spec="pbe",
        spin=0,
        charge=0,
        cart=True,
        grids_level=0,
        max_l=1,
        rks_config=cfg,
        grid_ao_backend="jax",
        integral_backend="libcint",
    )

    assert np.isclose(float(ref_jax.mf_energy), float(ref_libcint.mf_energy), atol=1e-6, rtol=0.0)
    assert np.allclose(np.asarray(ref_jax.overlap_matrix), np.asarray(ref_libcint.overlap_matrix), atol=1e-7, rtol=1e-7)
    assert np.allclose(np.asarray(ref_jax.h1e), np.asarray(ref_libcint.h1e), atol=5e-7, rtol=1e-7)
    assert np.allclose(np.asarray(ref_jax.rep_tensor), np.asarray(ref_libcint.rep_tensor), atol=3e-7, rtol=1e-7)


def test_restricted_reference_from_spec_with_jax_rks_libcint_full_uses_packed_eri(monkeypatch):
    _pyscf_or_skip()
    from pyscf import gto

    orig_intor = gto.mole.Mole.intor
    seen_aosym: list[str] = []

    def _guarded_intor(self, intor, *args, **kwargs):
        if str(intor) == "int2e_cart":
            aosym = str(kwargs.get("aosym", "s1"))
            seen_aosym.append(aosym)
            if aosym not in {"s4", "s8"}:
                raise AssertionError("default no-DF RKS must not build dense int2e_cart")
        return orig_intor(self, intor, *args, **kwargs)

    monkeypatch.setattr(gto.mole.Mole, "intor", _guarded_intor)

    ref = restricted_reference_from_spec_with_jax_rks(
        atom="""
        H 0.0 0.0 -0.35
        H 0.0 0.0  0.35
        """,
        basis="sto-3g",
        unit="Angstrom",
        xc_spec="pbe0",
        spin=0,
        charge=0,
        cart=True,
        grids_level=0,
        max_l=1,
        rks_config=RKSConfig(
            xc_spec="pbe0",
            max_cycle=30,
            conv_tol=1e-9,
            conv_tol_density=1e-7,
            damping=0.1,
            density_floor=1e-12,
            potential_clip=20.0,
        ),
        grid_ao_backend="jax",
        integral_backend="libcint",
    )

    assert "s4" in seen_aosym
    assert np.asarray(ref.rep_tensor).size == 0
    assert ref.eri_ovov is not None
    assert ref.eri_ovvo is not None
    assert ref.eri_oovv is not None
    assert np.isfinite(float(ref.mf_energy))


def test_restricted_reference_from_spec_with_jax_rks_libcint_skips_precompile(monkeypatch):
    _pyscf_or_skip()

    called = {"value": False}

    def fake_precompile(*args, **kwargs):
        called["value"] = True
        return {}

    monkeypatch.setattr("td_graddft.reference.precompile_eri_kernels", fake_precompile)

    with pytest.warns(RuntimeWarning, match="ignored when integral_backend='libcint'"):
        _ = restricted_reference_from_spec_with_jax_rks(
            atom="""
            H 0.0 0.0 -0.35
            H 0.0 0.0  0.35
            """,
            basis="sto-3g",
            unit="Angstrom",
            xc_spec="pbe",
            spin=0,
            charge=0,
            cart=True,
            grids_level=0,
            max_l=1,
            rks_config=RKSConfig(
                xc_spec="pbe",
                max_cycle=20,
                conv_tol=1e-8,
                conv_tol_density=1e-6,
                damping=0.15,
                density_floor=1e-12,
                potential_clip=20.0,
            ),
            grid_ao_backend="jax",
            integral_backend="libcint",
            precompile_eri=True,
        )

    assert called["value"] is False


def test_restricted_reference_from_spec_with_jax_rks_libcint_zero_policy_runs():
    _pyscf_or_skip()

    ref = restricted_reference_from_spec_with_jax_rks(
        atom="""
        H 0.0 0.0 -0.35
        H 0.0 0.0  0.35
        """,
        basis="sto-3g",
        unit="Angstrom",
        xc_spec="pbe",
        spin=0,
        charge=0,
        cart=True,
        grids_level=0,
        max_l=1,
        rks_config=RKSConfig(
            xc_spec="pbe",
            max_cycle=20,
            conv_tol=1e-8,
            conv_tol_density=1e-6,
            damping=0.15,
            density_floor=1e-12,
            potential_clip=20.0,
        ),
        grid_ao_backend="jax",
        integral_backend="libcint",
        libcint_geometry_grad_policy="zero",
    )
    assert np.isfinite(float(ref.mf_energy))


def test_restricted_reference_from_spec_with_jax_rks_invalid_libcint_policy_raises():
    with pytest.raises(ValueError, match="Unsupported libcint_geometry_grad_policy"):
        _ = restricted_reference_from_spec_with_jax_rks(
            atom="""
            H 0.0 0.0 -0.35
            H 0.0 0.0  0.35
            """,
            basis="sto-3g",
            unit="Angstrom",
            xc_spec="pbe",
            spin=0,
            charge=0,
            cart=True,
            grids_level=0,
            max_l=1,
            rks_config=RKSConfig(
                xc_spec="pbe",
                max_cycle=20,
                conv_tol=1e-8,
                conv_tol_density=1e-6,
                damping=0.15,
                density_floor=1e-12,
                potential_clip=20.0,
            ),
            grid_ao_backend="jax",
            integral_backend="libcint",
            libcint_geometry_grad_policy="invalid_policy",  # type: ignore[arg-type]
        )
