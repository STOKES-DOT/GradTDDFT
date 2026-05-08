import importlib
import types

import jax.numpy as jnp

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


def test_rks_kernel_calls_existing_restricted_reference_builder(monkeypatch):
    captured = {}

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            mf_energy=-76.0,
            mo_energy="mo_energy",
            mo_coeff="mo_coeff",
            mo_occ="mo_occ",
        )

    monkeypatch.setattr(
        "td_graddft.scf.facade.restricted_reference_from_spec_with_jax_rks",
        fake_builder,
    )

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    mf = scf.RKS(mol, xc="pbe")
    energy = mf.kernel()

    assert energy == -76.0
    assert mf.e_tot == -76.0
    assert mf.reference.mf_energy == -76.0
    assert mf.mo_energy == "mo_energy"
    assert mf.mo_coeff == "mo_coeff"
    assert mf.mo_occ == "mo_occ"
    assert mf.converged is True
    assert captured["atom"].symbols == ("H", "H")
    assert captured["basis"] == "sto-3g"
    assert captured["xc_spec"] == "pbe"
    assert captured["integral_backend"] == "libcint"
    assert captured["libcint_geometry_grad_policy"] == "analytic"
    assert captured["rks_config"].jk_backend == "full"


def test_rks_kernel_passes_hfx_feature_options(monkeypatch):
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
        "td_graddft.scf.facade.restricted_reference_from_spec_with_jax_rks",
        fake_builder,
    )

    mf = scf.RKS(gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"))
    mf.compute_local_hfx_features = True
    mf.compute_local_hfx_aux = True
    mf.hfx_omega_values = (0.0, 0.4)
    mf.hfx_chunk_size = 128
    mf.kernel()

    assert captured["compute_local_hfx_features"] is True
    assert captured["compute_local_hfx_aux"] is True
    assert captured["hfx_omega_values"] == (0.0, 0.4)
    assert captured["hfx_chunk_size"] == 128


def test_rks_cuda_ground_state_skips_dipole_input_construction(monkeypatch):
    captured = {}

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            mf_energy=-76.0,
            mo_energy=None,
            mo_coeff=None,
            mo_occ=None,
        )

    monkeypatch.setattr("td_graddft.scf.facade.cuda_ffi_available", lambda: True)
    monkeypatch.setattr(
        "td_graddft.scf.facade.restricted_reference_from_spec_with_jax_rks",
        fake_builder,
    )

    mf = scf.RKS(gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"))
    mf.cuda_direct_scf(execution_device="gpu")
    mf._build_reference(mf._spec())

    assert captured["include_dipole_integrals"] is False


def test_cuda_direct_scf_reuses_hqc_style_reference_solver(monkeypatch):
    monkeypatch.setattr("td_graddft.scf.facade.cuda_ffi_available", lambda: True)
    build_calls = []
    solver_calls = []

    def fake_solver_factory(**kwargs):
        build_calls.append(kwargs)

        def _solver(spec):
            solver_calls.append(spec)
            return types.SimpleNamespace(
                mf_energy=-1.0 - 0.1 * len(solver_calls),
                mo_energy=None,
                mo_coeff=None,
                mo_occ=None,
            )

        return _solver

    monkeypatch.setattr(
        "td_graddft.scf.facade._make_cuda_direct_reference_solver",
        fake_solver_factory,
    )

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    mf = scf.RKS(mol, xc="pbe0")
    mf.cuda_direct_scf(execution_device="gpu")
    mf.execution_device = "auto"

    first_energy = mf.kernel()
    mf.mol = gto.M(atom="H 0.1 0 0; H 0.1 0 0.74", basis="sto-3g")
    second_energy = mf.kernel()

    assert len(build_calls) == 1
    assert len(solver_calls) == 2
    assert build_calls[0]["basis"] == "sto-3g"
    assert build_calls[0]["xc_spec"] == "pbe0"
    assert first_energy == -1.1
    assert second_energy == -1.2


def test_uks_kernel_calls_existing_unrestricted_reference_builder(monkeypatch):
    captured = {}

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            mf_energy=-39.0,
            mo_energy="mo_energy",
            mo_coeff="mo_coeff",
            mo_occ="mo_occ",
        )

    monkeypatch.setattr(
        "td_graddft.scf.facade.unrestricted_reference_from_spec_with_jax_uks",
        fake_builder,
    )

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


def test_rks_run_and_backend_helpers(monkeypatch):
    monkeypatch.setattr(
        "td_graddft.scf.facade.restricted_reference_from_spec_with_jax_rks",
        lambda **kwargs: types.SimpleNamespace(
            mf_energy=-1.0,
            mo_energy=None,
            mo_coeff=None,
            mo_occ=None,
        ),
    )

    mf = scf.RKS(gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"))

    assert mf.density_fit() is mf
    assert mf.jk_backend == "df"
    assert mf.direct_scf() is mf
    assert mf.jk_backend == "direct"
    assert mf.run() is mf
    assert mf.e_tot == -1.0


def test_cuda_direct_scf_selects_gpu_cuda_backbone_when_available(monkeypatch):
    monkeypatch.setattr("td_graddft.scf.facade.cuda_ffi_available", lambda: True)
    cache_calls = []

    def fake_cache(**kwargs):
        cache_calls.append(kwargs)
        return kwargs["cache_dir"]

    monkeypatch.setattr("td_graddft.scf.facade.configure_jax_persistent_cache", fake_cache)

    mf = scf.RKS(gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"))
    mf.integral_backend = "jax"

    assert mf.cuda_direct_scf() is mf
    assert mf.jk_backend == "direct"
    assert mf.direct_jk_engine == "cuda"
    assert mf.integral_backend == "libcint"
    assert mf.grid_ao_backend == "pyscf"
    assert mf.iteration_backend == "lax"
    assert mf._config().iteration_backend == "lax"
    assert cache_calls == []


def test_cuda_direct_scf_defaults_to_lax_iteration_when_available(monkeypatch):
    monkeypatch.setattr("td_graddft.scf.facade.cuda_ffi_available", lambda: True)

    mf = scf.RKS(gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"))

    assert mf.cuda_direct_scf(execution_device="gpu") is mf
    assert mf.jk_backend == "direct"
    assert mf.direct_jk_engine == "cuda"
    assert mf.integral_backend == "libcint"
    assert mf.grid_ao_backend == "pyscf"
    assert mf.iteration_backend == "lax"
    assert mf._config().iteration_backend == "lax"


def test_cuda_direct_scf_configures_jax_cache_when_env_requests_it(monkeypatch):
    monkeypatch.setattr("td_graddft.scf.facade.cuda_ffi_available", lambda: True)
    monkeypatch.setenv("TD_GRADDFT_JAX_CACHE_DIR", ".jax_cache/test")
    monkeypatch.setenv("TD_GRADDFT_JAX_CACHE_MIN_COMPILE_SECS", "2.5")
    monkeypatch.setenv("TD_GRADDFT_JAX_CACHE_MIN_ENTRY_SIZE_BYTES", "131072")
    captured_cache = {}

    def fake_cache(**kwargs):
        captured_cache.update(kwargs)
        return kwargs["cache_dir"]

    monkeypatch.setattr("td_graddft.scf.facade.configure_jax_persistent_cache", fake_cache)

    mf = scf.RKS(gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"))

    assert mf.cuda_direct_scf() is mf
    assert captured_cache["cache_dir"] == ".jax_cache/test"
    assert captured_cache["min_compile_time_secs"] == 2.5
    assert captured_cache["min_entry_size_bytes"] == 131072


def test_cuda_direct_scf_can_precompile_current_geometry(monkeypatch):
    monkeypatch.setattr("td_graddft.scf.facade.cuda_ffi_available", lambda: True)
    captured = {}

    def fake_precompile(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "td_graddft.scf.facade.precompile_restricted_cuda_direct_rks_reference",
        fake_precompile,
    )

    mf = scf.RKS(gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"), xc="pbe0")
    mf.max_cycle = 23
    assert mf.cuda_direct_scf(execution_device="gpu", precompile=True) is mf

    assert mf._cuda_direct_reference_solver is not None
    assert captured["atom"].symbols == ("H", "H")
    assert captured["basis"] == "sto-3g"
    assert captured["xc_spec"] == "pbe0"
    assert captured["rks_config"].max_cycle == 23
    assert mf.iteration_backend == "lax"
    assert captured["grid_ao_backend"] == "pyscf"
    assert captured["integral_backend"] == "libcint"
    assert captured["include_dipole_integrals"] is False


def test_cuda_direct_scf_falls_back_to_cpu_libcint_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr("td_graddft.scf.facade.cuda_ffi_available", lambda: False)

    mf = scf.RKS(gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"))
    mf.integral_backend = "jax"

    assert mf.cuda_direct_scf() is mf
    assert mf.jk_backend == "full"
    assert mf.direct_jk_engine == "jax"
    assert mf.integral_backend == "libcint"
    assert mf.iteration_backend == "python"


def test_rks_cpu_execution_uses_pyscf_compatible_solver(monkeypatch):
    captured = {}

    def fake_cpu_builder(mf):
        captured["mf"] = mf
        return types.SimpleNamespace(
            mf_energy=-1.23,
            mo_energy="cpu_mo_energy",
            mo_coeff="cpu_mo_coeff",
            mo_occ="cpu_mo_occ",
            converged=True,
        )

    def fail_jax_builder(**kwargs):
        raise AssertionError("CPU execution should use the PySCF-compatible solver.")

    monkeypatch.setattr("td_graddft.scf.facade._restricted_reference_from_pyscf_cpu_rks", fake_cpu_builder)
    monkeypatch.setattr(
        "td_graddft.scf.facade.restricted_reference_from_spec_with_jax_rks",
        fail_jax_builder,
    )

    mf = scf.RKS(gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"))
    mf.cuda_direct_scf(execution_device="cpu")
    energy = mf.kernel()

    assert captured["mf"] is mf
    assert energy == -1.23
    assert mf.mo_energy == "cpu_mo_energy"


def test_rks_cpu_execution_uses_differentiable_builder_for_traced_geometry(monkeypatch):
    captured = {}

    def fail_cpu_builder(_mf):
        raise AssertionError("Traced geometry must not enter the PySCF CPU solver.")

    def fake_jax_builder(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            mf_energy=jnp.sum(kwargs["atom"].coords_bohr),
            mo_energy=None,
            mo_coeff=None,
            mo_occ=None,
        )

    monkeypatch.setattr("td_graddft.scf.facade._restricted_reference_from_pyscf_cpu_rks", fail_cpu_builder)
    monkeypatch.setattr(
        "td_graddft.scf.facade.restricted_reference_from_spec_with_jax_rks",
        fake_jax_builder,
    )

    mf = scf.RKS(gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"))
    mf.cuda_direct_scf(execution_device="cpu")
    grad = mf.nuc_grad_method().kernel()

    assert grad.shape == (2, 3)
    assert captured["integral_backend"] == "libcint"
    assert captured["grid_ao_backend"] == "jax"
    assert captured["rks_config"].jk_backend == "full"
    assert captured["rks_config"].direct_jk_engine == "jax"
    assert captured["include_dipole_integrals"] is True


def test_rks_gpu_execution_uses_differentiable_builder_for_traced_geometry(monkeypatch):
    captured = {}

    def fake_jax_builder(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            mf_energy=jnp.sum(kwargs["atom"].coords_bohr),
            mo_energy=None,
            mo_coeff=None,
            mo_occ=None,
        )

    monkeypatch.setattr("td_graddft.scf.facade.cuda_ffi_available", lambda: True)
    monkeypatch.setattr(
        "td_graddft.scf.facade.restricted_reference_from_spec_with_jax_rks",
        fake_jax_builder,
    )

    mf = scf.RKS(gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"))
    mf.cuda_direct_scf(execution_device="gpu")
    grad = mf.nuc_grad_method().kernel()

    assert grad.shape == (2, 3)
    assert captured["integral_backend"] == "libcint"
    assert captured["grid_ao_backend"] == "jax"
    assert captured["rks_config"].jk_backend == "full"
    assert captured["rks_config"].direct_jk_engine == "jax"
    assert captured["include_dipole_integrals"] is True


def test_nuc_grad_method_returns_geometry_gradient(monkeypatch):
    calls = {"count": 0}

    def fake_energy(mf, coords_bohr):
        calls["count"] += 1
        return jnp.sum(coords_bohr * coords_bohr)

    monkeypatch.setattr("td_graddft.scf.facade._energy_for_coords", fake_energy)

    mol = gto.M(
        atom=[
            ("H", jnp.array([0.0, 0.0, 0.0])),
            ("H", jnp.array([0.0, 0.0, 1.0])),
        ],
        basis="sto-3g",
    )
    mf = scf.RKS(mol)
    grad = mf.nuc_grad_method().kernel()

    assert grad.shape == (2, 3)
    assert calls["count"] >= 1


def test_top_level_import_exposes_gto_and_scf_modules():
    assert importlib.import_module("td_graddft.gto") is gto
    assert importlib.import_module("td_graddft.scf") is scf


def test_real_rks_kernel_smoke_sto3g_h2():
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    mf = scf.RKS(mol, xc="pbe")
    mf.max_cycle = 4
    energy = mf.kernel()

    assert energy < 0.0
    assert mf.reference is not None
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
    assert mf.reference is not None
    assert mf.mo_coeff is not None
