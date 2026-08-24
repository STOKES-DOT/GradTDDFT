from pathlib import Path


def test_rks_does_not_keep_legacy_cuda_eri_cache_selectors():
    text = Path("src/td_graddft/scf/rks.py").read_text()

    assert "_should_cache_cuda_full_eri" not in text
    assert "_should_cache_cuda_pair_eri" not in text
    assert "TD_GRADDFT_CUDA_FULL_ERI_MAX_MIB" not in text
    assert "TD_GRADDFT_CUDA_PAIR_ERI_MAX_MIB" not in text


def test_scf_does_not_keep_unreferenced_private_helpers():
    scf_text = "\n".join(path.read_text() for path in Path("src/td_graddft/scf").glob("*.py"))

    for helper in (
        "_cache_grid_ao_input_bundle",
        "_cache_libcint_host_integral",
        "_build_density_closed_shell",
        "_build_fock",
        "_cuda_pair_eri_max_bytes_for_inputs",
        "_coulomb_exchange_matrices",
        "_electronic_energy",
        "_energy_for_coords",
        "_eval_grid_ao_laplacian",
        "_gpu4pyscf_eri_pair_matrix",
        "_gpu4pyscf_eri_tensor",
        "_orbital_gradient_norm",
        "_restricted_channel_static",
        "_restricted_guess_density_from_pyscf",
        "_resolve_uks_config",
        "_should_fallback_to_hcore",
        "_unrestricted_coulomb_exchange_matrices",
        "_unrestricted_guess_density_from_pyscf",
        "_validate_initial_density",
        "_validate_initial_spin_density",
    ):
        assert helper not in scf_text


def test_rhf_reuses_the_shared_rks_scf_kernel():
    rhf_text = Path("src/td_graddft/scf/rhf.py").read_text()

    assert "run_rks_from_integrals(" in rhf_text
    assert "def _diis_extrapolate" not in rhf_text
    assert "for cycle in" not in rhf_text


def test_scf_core_no_longer_contains_custom_cuda_direct_backend():
    source_text = "\n".join(path.read_text() for path in Path("src/td_graddft").rglob("*.py"))

    for token in (
        "CudaDirectJKBuilder",
        "cuda_ffi_available",
        "cuda_direct",
        "direct_cuda",
        "gpu_cuda_direct",
        "TD_GRADDFT_CUDA",
        "precompile_restricted_cuda_direct",
    ):
        assert token not in source_text


def test_custom_cuda_integral_modules_are_removed():
    assert not Path("src/td_graddft/data/integrals/jax/cuda_direct_jk.py").exists()
    assert not Path("src/td_graddft/data/integrals/jax/cuda_direct_jk_kernel.cu").exists()
    assert not Path("src/td_graddft/data/integrals/jax/cuda_one_electron.py").exists()
    assert not Path("src/td_graddft/data/integrals/jax/cuda_one_electron_kernel.cu").exists()


def test_janak_constraint_code_is_removed_from_core_sources():
    source_text = "\n".join(path.read_text() for path in Path("src/td_graddft").rglob("*.py"))

    assert "janak" not in source_text.lower()
    assert "eta_autodiff" not in source_text
