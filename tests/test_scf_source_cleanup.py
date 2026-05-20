from pathlib import Path


def test_rks_does_not_keep_legacy_cuda_eri_cache_selectors():
    text = Path("src/td_graddft/scf/rks.py").read_text()

    assert "_should_cache_cuda_full_eri" not in text
    assert "_should_cache_cuda_pair_eri" not in text
    assert "TD_GRADDFT_CUDA_FULL_ERI_MAX_MIB" not in text
    assert "TD_GRADDFT_CUDA_PAIR_ERI_MAX_MIB" not in text


def test_scf_does_not_keep_unreferenced_private_helpers():
    scf_text = "\n".join(path.read_text() for path in Path("src/td_graddft/scf").glob("*.py"))

    assert "_energy_for_coords" not in scf_text
    assert "_restricted_channel_static" not in scf_text
    assert "_should_fallback_to_hcore" not in scf_text
    assert "_cuda_pair_eri_max_bytes_for_inputs" not in scf_text
    assert "_orbital_gradient_norm" not in scf_text


def test_scf_core_no_longer_contains_custom_cuda_direct_backend():
    source_text = "\n".join(
        path.read_text()
        for path in Path("src/td_graddft").rglob("*.py")
        if path.name != "gpu4pyscf.py"
    )

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
