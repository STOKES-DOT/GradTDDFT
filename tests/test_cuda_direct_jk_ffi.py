from pathlib import Path

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from td_graddft.data.basis import basis_from_spec
from td_graddft.reference import _cuda_direct_basis_cache_key
from td_graddft.scf import cuda_direct_jk
from td_graddft.scf.cuda_direct_jk import (
    CudaDirectJKBuilder,
    _primitive_cart_norm,
    joltqc_basis_data_from_basis,
)


def test_cuda_direct_rks_cache_key_ignores_geometry_centers_but_basis_data_updates():
    basis_a = basis_from_spec("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    basis_b = basis_from_spec("H 0.1 0 0; H 0.1 0 0.74", basis="sto-3g")

    assert _cuda_direct_basis_cache_key(basis_a) == _cuda_direct_basis_cache_key(basis_b)

    data_a = joltqc_basis_data_from_basis(basis_a)
    data_b = joltqc_basis_data_from_basis(basis_b)
    assert not np.allclose(data_a[:, :3], data_b[:, :3])
    np.testing.assert_allclose(data_a[:, 3:], data_b[:, 3:])


def test_cuda_direct_jk_builder_invokes_ffi_call(monkeypatch, tmp_path):
    basis = basis_from_spec("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    density = np.asarray(
        [
            [0.83, 0.21],
            [0.21, 0.71],
        ],
        dtype=np.float64,
    )
    captured = {}

    def fake_compile_and_register(self):
        captured["registered"] = True

    def fake_compile_library(self):
        return tmp_path / "libfake.so"

    def fake_ffi_call(target_name, result_shape_dtypes, *args, **kwargs):
        captured["target_name"] = target_name
        captured["result_shape_dtypes"] = result_shape_dtypes
        captured["args"] = args
        captured["kwargs"] = kwargs
        return np.ones_like(density), 2.0 * np.ones_like(density)

    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_library", fake_compile_library)
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_and_register", fake_compile_and_register)
    monkeypatch.setattr("td_graddft.scf.cuda_direct_jk._ffi_call", fake_ffi_call)

    builder = CudaDirectJKBuilder(basis, cache_dir=tmp_path)
    j_mat, k_mat = builder.build_jk(density)

    assert captured["registered"] is True
    assert captured["target_name"] == "td_graddft_cuda_direct_jk"
    assert len(captured["result_shape_dtypes"]) == 2
    assert captured["result_shape_dtypes"][0].shape == density.shape
    assert len(captured["args"]) == 11
    assert np.allclose(np.asarray(captured["args"][0]), density)
    assert np.asarray(captured["args"][1]).shape[1] == 3
    assert np.asarray(captured["args"][3]).ndim == 2
    assert np.asarray(captured["args"][9]).shape[0] == basis.nao * (basis.nao + 1) // 2
    assert np.asarray(captured["args"][10])[0] == 0.0
    assert np.allclose(np.asarray(j_mat), 1.0)
    assert np.allclose(np.asarray(k_mat), 2.0)


def test_cuda_direct_jk_builder_uses_prebuilt_library_env(monkeypatch, tmp_path):
    basis = basis_from_spec("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    library = tmp_path / "libtd_graddft_cuda_direct_jk_prebuilt.so"
    library.write_bytes(b"prebuilt")
    captured = {}

    def fail_run(*args, **kwargs):
        raise AssertionError("prebuilt CUDA FFI library should skip nvcc compilation")

    def fake_register(self):
        captured["library"] = self.library

    monkeypatch.setenv("TD_GRADDFT_CUDA_JK_LIBRARY", str(library))
    monkeypatch.setattr(cuda_direct_jk.subprocess, "run", fail_run)
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_and_register", fake_register)

    builder = CudaDirectJKBuilder(
        basis,
        cache_dir=tmp_path / "cache",
        nvcc=None,
        arch="sm_120",
    )

    assert builder.library == library
    assert captured["library"] == library


def test_cuda_direct_jk_builder_uses_package_prebuilt_library(monkeypatch, tmp_path):
    basis = basis_from_spec("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    library = tmp_path / "libtd_graddft_cuda_direct_jk_packaged.so"
    library.write_bytes(b"packaged")
    captured = {}

    def fail_run(*args, **kwargs):
        raise AssertionError("packaged CUDA FFI library should skip nvcc compilation")

    def fake_register(self):
        captured["library"] = self.library

    monkeypatch.delenv("TD_GRADDFT_CUDA_JK_LIBRARY", raising=False)
    monkeypatch.setattr(cuda_direct_jk.subprocess, "run", fail_run)
    monkeypatch.setattr(
        cuda_direct_jk,
        "_packaged_prebuilt_library_path",
        lambda arch, **kwargs: library,
    )
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_and_register", fake_register)

    builder = CudaDirectJKBuilder(
        basis,
        cache_dir=tmp_path / "cache",
        nvcc=None,
        arch="sm_120",
    )

    assert builder.library == library
    assert captured["library"] == library


def test_cuda_direct_jk_builder_uses_any_package_prebuilt_without_arch_probe(monkeypatch, tmp_path):
    basis = basis_from_spec("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    library = tmp_path / "libtd_graddft_cuda_direct_jk_packaged.so"
    library.write_bytes(b"packaged")
    captured = {}

    def fail_detect():
        raise AssertionError("packaged CUDA FFI library should not need runtime architecture probing")

    def fake_register(self):
        captured["library"] = self.library

    monkeypatch.delenv("TD_GRADDFT_CUDA_JK_LIBRARY", raising=False)
    monkeypatch.setattr(cuda_direct_jk, "_detect_cuda_arch", fail_detect)
    monkeypatch.setattr(cuda_direct_jk, "_any_packaged_prebuilt_library", lambda: library)
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_and_register", fake_register)

    builder = CudaDirectJKBuilder(
        basis,
        cache_dir=tmp_path / "cache",
        nvcc=None,
    )

    assert builder.library == library
    assert captured["library"] == library


def test_cuda_direct_jk_builder_uses_fixed_runtime_build_when_nvcc_is_available(
    monkeypatch,
    tmp_path,
):
    basis = basis_from_spec("C 0 0 0; H 0 0 1.1", basis="6-31g*")
    packaged = tmp_path / "libtd_graddft_cuda_direct_jk_packaged.so"
    packaged.write_bytes(b"packaged")
    built = tmp_path / "libtd_graddft_cuda_direct_jk_built.so"
    built.write_bytes(b"built")
    captured = {}

    def fake_build(output_dir, **kwargs):
        captured["output_dir"] = output_dir
        captured["kwargs"] = kwargs
        return built

    def fake_register(self):
        captured["library"] = self.library

    monkeypatch.delenv("TD_GRADDFT_CUDA_JK_LIBRARY", raising=False)
    monkeypatch.delenv("TD_GRADDFT_CUDA_JOLTQC_FIXED_UNIVERSE", raising=False)
    monkeypatch.setattr(cuda_direct_jk, "_packaged_prebuilt_library_path", lambda arch, **kwargs: None)
    monkeypatch.setattr(cuda_direct_jk, "_any_packaged_prebuilt_library", lambda: packaged)
    monkeypatch.setattr(cuda_direct_jk, "build_prebuilt_cuda_direct_jk_library", fake_build)
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_and_register", fake_register)

    builder = CudaDirectJKBuilder(
        basis,
        cache_dir=tmp_path / "cache",
        nvcc="/usr/local/cuda/bin/nvcc",
        arch="sm_120",
    )

    assert builder.library == built
    assert captured["library"] == built
    assert captured["kwargs"]["nvcc"] == "/usr/local/cuda/bin/nvcc"
    assert captured["kwargs"]["arch"] == "sm_120"
    assert "joltqc_group_keys" not in captured["kwargs"]
    assert "joltqc_group_quartet_keys" not in captured["kwargs"]
    assert "joltqc_group_quartet_offsets" not in captured["kwargs"]


def test_cuda_direct_jk_builder_prefers_package_runtime_library_even_with_nvcc(
    monkeypatch,
    tmp_path,
):
    basis = basis_from_spec("C 0 0 0; H 0 0 1.1", basis="6-31g*")
    library = tmp_path / "libtd_graddft_cuda_direct_jk_fixed.so"
    library.write_bytes(b"packaged-fixed")
    captured = {}

    def fake_packaged(arch, **kwargs):
        captured["arch"] = arch
        captured["extra_source_key"] = kwargs.get("extra_source_key")
        return library if kwargs.get("extra_source_key") is None else None

    def fail_build(*args, **kwargs):
        raise AssertionError("package runtime CUDA library should skip nvcc compilation")

    def fake_register(self):
        captured["library"] = self.library

    monkeypatch.delenv("TD_GRADDFT_CUDA_JK_LIBRARY", raising=False)
    monkeypatch.setattr(cuda_direct_jk, "_packaged_prebuilt_library_path", fake_packaged)
    monkeypatch.setattr(cuda_direct_jk, "build_prebuilt_cuda_direct_jk_library", fail_build)
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_and_register", fake_register)

    builder = CudaDirectJKBuilder(
        basis,
        cache_dir=tmp_path / "cache",
        nvcc="/usr/local/cuda/bin/nvcc",
        arch="sm_120",
    )

    assert builder.library == library
    assert captured["library"] == library
    assert captured["arch"] == "sm_120"
    assert captured["extra_source_key"] is None


def test_cuda_direct_jk_builder_can_build_fixed_joltqc_universe_when_requested(
    monkeypatch,
    tmp_path,
):
    basis = basis_from_spec("C 0 0 0; H 0 0 1.1", basis="6-31g*")
    built = tmp_path / "libtd_graddft_cuda_direct_jk_fixed_built.so"
    built.write_bytes(b"fixed-built")
    captured = {}

    def fake_build(output_dir, **kwargs):
        captured["output_dir"] = output_dir
        captured["kwargs"] = kwargs
        return built

    def fake_register(self):
        captured["library"] = self.library

    monkeypatch.delenv("TD_GRADDFT_CUDA_JK_LIBRARY", raising=False)
    monkeypatch.setenv("TD_GRADDFT_CUDA_JOLTQC_FIXED_UNIVERSE", "1")
    monkeypatch.setattr(cuda_direct_jk, "_packaged_prebuilt_library_path", lambda arch, **kwargs: None)
    monkeypatch.setattr(cuda_direct_jk, "build_prebuilt_cuda_direct_jk_library", fake_build)
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_and_register", fake_register)

    builder = CudaDirectJKBuilder(
        basis,
        cache_dir=tmp_path / "cache",
        nvcc="/usr/local/cuda/bin/nvcc",
        arch="sm_120",
    )

    assert builder.library == built
    assert captured["library"] == built
    assert captured["kwargs"]["joltqc_fixed_universe"] is True
    assert captured["kwargs"]["joltqc_fixed_max_l"] == 2
    assert "joltqc_group_keys" not in captured["kwargs"]


def test_cuda_direct_jk_builder_rejects_missing_prebuilt_library(monkeypatch, tmp_path):
    basis = basis_from_spec("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    missing = tmp_path / "missing.so"

    monkeypatch.setenv("TD_GRADDFT_CUDA_JK_LIBRARY", str(missing))

    try:
        CudaDirectJKBuilder(basis, cache_dir=tmp_path / "cache", nvcc=None, arch="sm_120")
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("missing prebuilt CUDA FFI library should fail explicitly")


def test_cuda_ffi_available_accepts_prebuilt_library_without_nvcc(monkeypatch, tmp_path):
    library = tmp_path / "libtd_graddft_cuda_direct_jk_prebuilt.so"
    library.write_bytes(b"prebuilt")

    monkeypatch.delenv("TD_GRADDFT_DISABLE_CUDA_FFI", raising=False)
    monkeypatch.delenv("TD_GRADDFT_NVCC", raising=False)
    monkeypatch.setenv("TD_GRADDFT_CUDA_JK_LIBRARY", str(library))
    monkeypatch.setattr(cuda_direct_jk.shutil, "which", lambda name: None)
    monkeypatch.setattr(cuda_direct_jk.jax, "devices", lambda kind: ["gpu0"] if kind == "gpu" else [])

    assert cuda_direct_jk.cuda_ffi_available()


def test_cuda_ffi_available_accepts_package_prebuilt_library_without_nvcc(monkeypatch, tmp_path):
    library = tmp_path / "libtd_graddft_cuda_direct_jk_packaged.so"
    library.write_bytes(b"packaged")

    monkeypatch.delenv("TD_GRADDFT_DISABLE_CUDA_FFI", raising=False)
    monkeypatch.delenv("TD_GRADDFT_NVCC", raising=False)
    monkeypatch.delenv("TD_GRADDFT_CUDA_JK_LIBRARY", raising=False)
    monkeypatch.setattr(cuda_direct_jk.shutil, "which", lambda name: None)
    monkeypatch.setattr(cuda_direct_jk.jax, "devices", lambda kind: ["gpu0"] if kind == "gpu" else [])
    monkeypatch.setattr(cuda_direct_jk, "_any_packaged_prebuilt_library", lambda: library)

    assert cuda_direct_jk.cuda_ffi_available()


def test_cuda_ffi_available_rejects_missing_prebuilt_library(monkeypatch, tmp_path):
    monkeypatch.delenv("TD_GRADDFT_DISABLE_CUDA_FFI", raising=False)
    monkeypatch.delenv("TD_GRADDFT_NVCC", raising=False)
    monkeypatch.setenv("TD_GRADDFT_CUDA_JK_LIBRARY", str(tmp_path / "missing.so"))
    monkeypatch.setattr(cuda_direct_jk.shutil, "which", lambda name: None)
    monkeypatch.setattr(cuda_direct_jk.jax, "devices", lambda kind: ["gpu0"])

    assert not cuda_direct_jk.cuda_ffi_available()


def test_cuda_direct_jk_builder_can_build_pair_schwarz(monkeypatch, tmp_path):
    basis = basis_from_spec("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    npair = basis.nao * (basis.nao + 1) // 2
    captured = {}

    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_library", lambda self: tmp_path / "libfake.so")
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_and_register", lambda self: None)

    def fake_ffi_call(target_name, result_shape_dtypes, *args, **kwargs):
        captured["target_name"] = target_name
        captured["result_shape_dtypes"] = result_shape_dtypes
        captured["args"] = args
        captured["kwargs"] = kwargs
        return np.arange(npair, dtype=np.float64) + 1.0

    monkeypatch.setattr("td_graddft.scf.cuda_direct_jk._ffi_call", fake_ffi_call)

    builder = CudaDirectJKBuilder(basis, cache_dir=tmp_path)
    bounds = builder.build_pair_schwarz()

    assert captured["target_name"] == "td_graddft_cuda_pair_schwarz"
    assert captured["result_shape_dtypes"].shape == (npair,)
    assert len(captured["args"]) == 8
    assert np.allclose(np.asarray(bounds), np.arange(npair, dtype=np.float64) + 1.0)


def test_cuda_ao_system_records_joltqc_style_shell_groups():
    basis = basis_from_spec(
        "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="6-31g*",
    )

    system = cuda_direct_jk.extract_cuda_ao_system(basis, max_l=2)
    layout = system.shell_layout

    shell_l = np.asarray(
        [sum(int(power) for power in shell.angulars[0]) for shell in basis.shells],
        dtype=np.int32,
    )
    shell_nprims = np.asarray(
        [int(np.asarray(shell.exponents).shape[0]) for shell in basis.shells],
        dtype=np.int32,
    )
    expected_sorted = np.asarray(
        sorted(
            range(len(basis.shells)),
            key=lambda idx: (int(shell_l[idx]), -int(shell_nprims[idx]), int(idx)),
        ),
        dtype=np.int32,
    )
    expected_group_keys = np.asarray(
        sorted(
            {(int(shell_l[idx]), int(shell_nprims[idx])) for idx in range(len(basis.shells))},
            key=lambda item: (item[0], -item[1]),
        ),
        dtype=np.int32,
    )

    np.testing.assert_array_equal(layout.shell_l, shell_l)
    np.testing.assert_array_equal(layout.shell_nprims, shell_nprims)
    np.testing.assert_array_equal(layout.sorted_shell_indices, expected_sorted)
    np.testing.assert_array_equal(layout.group_keys, expected_group_keys)


def test_cuda_ao_system_shell_group_offsets_cover_contiguous_sorted_ranges():
    basis = basis_from_spec("C 0 0 0; C 0 0 1.4", basis="6-31g*")

    system = cuda_direct_jk.extract_cuda_ao_system(basis, max_l=2)
    layout = system.shell_layout

    offsets = np.asarray(layout.group_offsets, dtype=np.int32)
    assert offsets.shape == (layout.group_keys.shape[0] + 1,)
    assert offsets[0] == 0
    assert offsets[-1] == len(basis.shells)
    assert np.all(offsets[1:] >= offsets[:-1])

    sorted_indices = np.asarray(layout.sorted_shell_indices, dtype=np.int32)
    for group_id, key in enumerate(np.asarray(layout.group_keys, dtype=np.int32)):
        start = int(offsets[group_id])
        stop = int(offsets[group_id + 1])
        assert stop > start
        for shell_idx in sorted_indices[start:stop]:
            assert int(layout.shell_l[shell_idx]) == int(key[0])
            assert int(layout.shell_nprims[shell_idx]) == int(key[1])

    padded_offsets = np.asarray(layout.padded_group_offsets, dtype=np.int32)
    padded_indices = np.asarray(layout.padded_sorted_shell_indices, dtype=np.int32)
    pad_mask = np.asarray(layout.padded_shell_pad_mask, dtype=bool)
    assert padded_offsets.shape == (layout.group_keys.shape[0] + 1,)
    assert padded_offsets[0] == 0
    assert padded_offsets[-1] == padded_indices.shape[0] == pad_mask.shape[0]

    for group_id in range(layout.group_keys.shape[0]):
        start = int(offsets[group_id])
        stop = int(offsets[group_id + 1])
        padded_start = int(padded_offsets[group_id])
        padded_stop = int(padded_offsets[group_id + 1])
        assert (padded_stop - padded_start) % int(layout.tile_size) == 0
        np.testing.assert_array_equal(
            padded_indices[padded_start : padded_start + (stop - start)],
            sorted_indices[start:stop],
        )
        assert not np.any(pad_mask[padded_start : padded_start + (stop - start)])
        if padded_stop > padded_start + (stop - start):
            assert np.all(pad_mask[padded_start + (stop - start) : padded_stop])
            assert np.all(
                padded_indices[padded_start + (stop - start) : padded_stop]
                == sorted_indices[start]
            )

    tile_shell_indices = np.asarray(layout.tile_shell_indices, dtype=np.int32)
    tile_shell_pad_mask = np.asarray(layout.tile_shell_pad_mask, dtype=bool)
    assert tile_shell_indices.shape[1] == int(layout.tile_size)
    assert tile_shell_indices.shape == tile_shell_pad_mask.shape
    np.testing.assert_array_equal(
        tile_shell_indices.reshape(-1),
        padded_indices,
    )
    np.testing.assert_array_equal(
        tile_shell_pad_mask.reshape(-1),
        pad_mask,
    )


def test_cuda_rys_env_layout_matches_pyscf_shell_metadata():
    pyscf_gto = pytest.importorskip("pyscf.gto")

    mol = pyscf_gto.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        verbose=0,
    )
    from td_graddft.data.basis import basis_from_pyscf_mol_cart

    basis = basis_from_pyscf_mol_cart(mol, max_l=2, precompute_eri_groups=False)
    shell_layout = cuda_direct_jk._build_joltqc_shell_layout(basis)
    layout = cuda_direct_jk._build_rys_env_layout(basis, shell_layout=shell_layout)
    sorted_shells = np.asarray(shell_layout.sorted_shell_indices, dtype=np.int32)
    expected_ao_to_parent = np.concatenate(
        [
            np.asarray(basis.shells[int(shell_id)].ao_indices, dtype=np.int32)
            for shell_id in sorted_shells
        ],
        axis=0,
    )

    assert layout.atm.shape == (mol.natm, cuda_direct_jk._RYS_ATM_SLOTS)
    assert layout.bas.shape == (mol.nbas, cuda_direct_jk._RYS_BAS_SLOTS)
    assert layout.ao_loc[-1] == mol.nao
    np.testing.assert_array_equal(layout.sorted_shell_indices, sorted_shells)
    np.testing.assert_array_equal(layout.group_keys, shell_layout.group_keys)
    np.testing.assert_array_equal(layout.group_offsets, shell_layout.group_offsets)
    np.testing.assert_array_equal(layout.ao_to_parent_ao, expected_ao_to_parent)
    inverse = np.empty((mol.nao,), dtype=np.int32)
    inverse[expected_ao_to_parent] = np.arange(mol.nao, dtype=np.int32)
    np.testing.assert_array_equal(layout.parent_ao_to_ao, inverse)
    np.testing.assert_array_equal(
        layout.bas[:, cuda_direct_jk._RYS_ANG_OF],
        mol._bas[sorted_shells, 1],
    )
    np.testing.assert_array_equal(
        layout.bas[:, cuda_direct_jk._RYS_NPRIM_OF],
        mol._bas[sorted_shells, 2],
    )
    for rys_shell_id, shell_id in enumerate(sorted_shells.tolist()):
        exp_ptr = int(layout.bas[rys_shell_id, cuda_direct_jk._RYS_PTR_EXP])
        coeff_ptr = int(layout.bas[rys_shell_id, cuda_direct_jk._RYS_PTR_COEFF])
        nprim = int(layout.bas[rys_shell_id, cuda_direct_jk._RYS_NPRIM_OF])
        expected_coeff = cuda_direct_jk._rys_env_coefficients(
            int(mol.bas_angular(shell_id)),
            mol.bas_exp(shell_id),
            np.asarray(mol.bas_ctr_coeff(shell_id)).reshape(-1),
        )
        np.testing.assert_allclose(layout.env[exp_ptr : exp_ptr + nprim], mol.bas_exp(shell_id))
        np.testing.assert_allclose(
            layout.env[coeff_ptr : coeff_ptr + nprim],
            expected_coeff,
        )


def test_cuda_rys_pair_mapping_matches_rys_tile_order():
    basis = basis_from_spec("O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587", basis="sto-3g")
    shell_layout = cuda_direct_jk._build_joltqc_shell_layout(basis)
    nshell = int(shell_layout.sorted_shell_indices.shape[0])
    log_q = np.zeros((nshell, nshell), dtype=np.float32)

    mappings = cuda_direct_jk._make_rys_tril_pair_mappings(
        shell_layout.group_offsets,
        log_q,
        cutoff=-np.inf,
        tile=6,
    )

    for i in range(shell_layout.group_offsets.shape[0] - 1):
        ish0 = int(shell_layout.group_offsets[i])
        ish1 = int(shell_layout.group_offsets[i + 1])
        for j in range(i + 1):
            jsh0 = int(shell_layout.group_offsets[j])
            jsh1 = int(shell_layout.group_offsets[j + 1])
            expected = [
                ish * nshell + jsh
                for ish in range(ish0, ish1)
                for jsh in range(jsh0, jsh1)
                if i != j or ish >= jsh
            ]
            np.testing.assert_array_equal(
                mappings[(i, j)],
                np.asarray(expected, dtype=np.int32),
            )

    filtered_q = np.full((nshell, nshell), -100.0, dtype=np.float32)
    filtered_q[1, 0] = 0.0
    filtered = cuda_direct_jk._make_rys_tril_pair_mappings(
        np.asarray([0, nshell], dtype=np.int32),
        filtered_q,
        cutoff=-1.0,
        tile=6,
    )
    np.testing.assert_array_equal(filtered[(0, 0)], np.asarray([nshell], dtype=np.int32))


def test_cuda_ao_system_builds_joltqc_split_packed_basis_data():
    basis = basis_from_spec("C 0 0 0; H 0 0 1.1", basis="6-31g*")

    system = cuda_direct_jk.extract_cuda_ao_system(basis, max_l=2)
    layout = system.joltqc_basis_layout
    nonpad = ~np.asarray(layout.pad_mask, dtype=bool)

    assert layout.basis_data.shape[1] == cuda_direct_jk._JOLTQC_BASIS_STRIDE
    assert layout.basis_data_fp32.shape == layout.basis_data.shape
    assert layout.group_offsets[0] == 0
    assert layout.group_offsets[-1] == layout.basis_data.shape[0]
    assert np.all(np.asarray(layout.group_offsets[1:]) >= np.asarray(layout.group_offsets[:-1]))
    assert np.max(np.asarray(layout.shell_nprims)[nonpad]) <= cuda_direct_jk._JOLTQC_NPRIM_MAX
    assert not np.any(np.asarray(layout.group_keys)[:, 1] > cuda_direct_jk._JOLTQC_NPRIM_MAX)

    parent_counts = np.bincount(
        np.asarray(layout.to_parent_shell, dtype=np.int32)[nonpad],
        minlength=len(basis.shells),
    )
    for shell_id, shell in enumerate(basis.shells):
        expected = (int(np.asarray(shell.exponents).shape[0]) + cuda_direct_jk._JOLTQC_NPRIM_MAX - 1) // cuda_direct_jk._JOLTQC_NPRIM_MAX
        assert parent_counts[shell_id] == expected
    assert np.any(parent_counts > 1)


def test_cuda_joltqc_basis_data_packs_ao_loc_coordinates_and_ce_chunks():
    basis = basis_from_spec("C 0 0 0; H 0 0 1.1", basis="6-31g*")

    system = cuda_direct_jk.extract_cuda_ao_system(basis, max_l=2)
    layout = system.joltqc_basis_layout
    packed = np.asarray(layout.basis_data, dtype=np.float64)

    np.testing.assert_allclose(packed[:, 3], np.asarray(layout.ao_loc[:-1], dtype=np.float64))
    assert int(layout.ao_loc[-1]) > basis.nao
    assert layout.ao_to_parent_ao.shape == (int(layout.ao_loc[-1]),)

    for row, is_pad in enumerate(np.asarray(layout.pad_mask, dtype=bool)):
        if is_pad:
            assert int(layout.ao_loc[row + 1]) == int(layout.ao_loc[row])
            continue
        parent = int(layout.to_parent_shell[row])
        start = int(layout.primitive_starts[row])
        nprim = int(layout.shell_nprims[row])
        shell = basis.shells[parent]
        ltot = sum(int(power) for power in shell.angulars[0])
        exponents = np.asarray(shell.exponents, dtype=np.float64)[start : start + nprim]
        coeffs = np.asarray(shell.coefficients, dtype=np.float64)[start : start + nprim]
        coeffs = coeffs * _primitive_cart_norm(
            exponents,
            (ltot, 0, 0),
        )

        np.testing.assert_allclose(packed[row, 0:3], np.asarray(shell.center, dtype=np.float64))
        np.testing.assert_allclose(packed[row, 4 : 4 + 2 * nprim : 2], coeffs)
        np.testing.assert_allclose(packed[row, 5 : 5 + 2 * nprim : 2], exponents)
        np.testing.assert_allclose(packed[row, 4 + 2 * nprim :], 0.0)
        start_ao = int(layout.ao_loc[row])
        stop_ao = int(layout.ao_loc[row + 1])
        np.testing.assert_array_equal(
            layout.ao_to_parent_ao[start_ao:stop_ao],
            np.asarray(shell.ao_indices, dtype=np.int32),
        )

    duplicated_parent_ao = np.bincount(
        np.asarray(layout.ao_to_parent_ao, dtype=np.int32),
        minlength=basis.nao,
    )
    assert np.any(duplicated_parent_ao > 1)


def test_cuda_joltqc_quartet_layout_batches_tasks_by_group_quartet():
    basis = basis_from_spec("C 0 0 0; H 0 0 1.1", basis="6-31g*")

    system = cuda_direct_jk.extract_cuda_ao_system(basis, max_l=2)
    basis_layout = system.joltqc_basis_layout
    quartet_layout = system.joltqc_quartet_layout

    keys = np.asarray(quartet_layout.group_quartet_keys, dtype=np.int32)
    offsets = np.asarray(quartet_layout.group_quartet_offsets, dtype=np.int32)
    quartets = np.asarray(quartet_layout.shell_quartets, dtype=np.int32)
    shell_to_group = np.empty((basis_layout.basis_data.shape[0],), dtype=np.int32)
    for group_id in range(basis_layout.group_keys.shape[0]):
        start = int(basis_layout.group_offsets[group_id])
        stop = int(basis_layout.group_offsets[group_id + 1])
        shell_to_group[start:stop] = group_id

    assert keys.ndim == 2 and keys.shape[1] == 4
    assert offsets.shape == (keys.shape[0] + 1,)
    assert offsets[0] == 0
    assert offsets[-1] == quartets.shape[0]
    assert np.all(offsets[1:] >= offsets[:-1])
    assert not np.any(np.asarray(basis_layout.pad_mask, dtype=bool)[quartets.reshape(-1)])

    seen_shell_pairs: set[tuple[int, int]] = set()
    for group_id, key in enumerate(keys):
        start = int(offsets[group_id])
        stop = int(offsets[group_id + 1])
        assert stop > start
        for i, j, k, l in quartets[start:stop]:
            assert i >= j
            assert k >= l
            assert i * (i + 1) // 2 + j >= k * (k + 1) // 2 + l
            np.testing.assert_array_equal(
                [shell_to_group[i], shell_to_group[j], shell_to_group[k], shell_to_group[l]],
                key,
            )
            seen_shell_pairs.add((int(i), int(j)))
            seen_shell_pairs.add((int(k), int(l)))

    nonpad_shells = np.flatnonzero(~np.asarray(basis_layout.pad_mask, dtype=bool))
    expected_pair_count = len(nonpad_shells) * (len(nonpad_shells) + 1) // 2
    assert len(seen_shell_pairs) == expected_pair_count


def test_cuda_joltqc_quartet_layout_matches_reference_loop_order():
    basis = basis_from_spec("C 0 0 0; H 0 0 1.1", basis="6-31g*")
    basis_layout = cuda_direct_jk._build_joltqc_basis_layout(basis)
    quartet_layout = cuda_direct_jk._build_joltqc_quartet_layout(basis_layout)

    shell_to_group = np.zeros((basis_layout.basis_data.shape[0],), dtype=np.int32)
    for group_id in range(int(basis_layout.group_keys.shape[0])):
        start = int(basis_layout.group_offsets[group_id])
        stop = int(basis_layout.group_offsets[group_id + 1])
        shell_to_group[start:stop] = group_id

    nonpad_shells = [
        int(shell_id)
        for shell_id, is_pad in enumerate(np.asarray(basis_layout.pad_mask, dtype=bool))
        if not bool(is_pad)
    ]
    shell_pairs = [
        (i, j)
        for i_pos, i in enumerate(nonpad_shells)
        for j in nonpad_shells[: i_pos + 1]
    ]
    buckets: dict[tuple[int, int, int, int], list[tuple[int, int, int, int]]] = {}
    for pair_p_pos, (i, j) in enumerate(shell_pairs):
        pair_p_id = i * (i + 1) // 2 + j
        for k, l in shell_pairs[: pair_p_pos + 1]:
            pair_q_id = k * (k + 1) // 2 + l
            if pair_p_id < pair_q_id:
                continue
            key = (
                int(shell_to_group[i]),
                int(shell_to_group[j]),
                int(shell_to_group[k]),
                int(shell_to_group[l]),
            )
            buckets.setdefault(key, []).append((int(i), int(j), int(k), int(l)))

    expected_keys = np.asarray(sorted(buckets), dtype=np.int32).reshape(-1, 4)
    expected_offsets = [0]
    expected_chunks = []
    for key in sorted(buckets):
        chunk = np.asarray(buckets[key], dtype=np.int32).reshape(-1, 4)
        expected_chunks.append(chunk)
        expected_offsets.append(expected_offsets[-1] + int(chunk.shape[0]))
    expected_quartets = np.concatenate(expected_chunks, axis=0)

    np.testing.assert_array_equal(quartet_layout.group_quartet_keys, expected_keys)
    np.testing.assert_array_equal(
        quartet_layout.group_quartet_offsets,
        np.asarray(expected_offsets, dtype=np.int32),
    )
    np.testing.assert_array_equal(quartet_layout.shell_quartets, expected_quartets)


def test_cuda_direct_jk_builder_materializes_joltqc_ffi_metadata(monkeypatch, tmp_path):
    basis = basis_from_spec("C 0 0 0; H 0 0 1.1", basis="6-31g*")

    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_library", lambda self: tmp_path / "libfake.so")
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_and_register", lambda self: None)

    builder = CudaDirectJKBuilder(basis, cache_dir=tmp_path)

    assert builder.joltqc_basis_data.shape == builder.system.joltqc_basis_layout.basis_data.shape
    assert builder.joltqc_basis_data.dtype == jnp.float64
    assert builder.joltqc_basis_data_fp32.shape == builder.joltqc_basis_data.shape
    assert builder.joltqc_basis_data_fp32.dtype == jnp.float32
    assert builder.joltqc_ao_to_parent_ao.shape == builder.system.joltqc_basis_layout.ao_to_parent_ao.shape
    assert builder.joltqc_group_keys.shape[1] == 2
    assert builder.joltqc_group_quartet_keys.shape[1] == 4
    assert builder.joltqc_group_quartet_offsets.shape[0] == builder.joltqc_group_quartet_keys.shape[0] + 1
    assert builder.joltqc_shell_quartets.shape[1] == 4


def test_cuda_direct_jk_builder_requires_pair_metadata_for_runtime_mapping(
    monkeypatch,
    tmp_path,
):
    basis = basis_from_spec("C 0 0 0; H 0 0 1.1", basis="6-31g*")

    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_library", lambda self: tmp_path / "libfake.so")
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_and_register", lambda self: None)

    builder = CudaDirectJKBuilder(
        basis,
        cache_dir=tmp_path,
        include_pair_metadata=False,
    )

    assert builder.system.pair_exponents.shape == (0, 0)
    assert builder.system.pair_centers.shape == (0, 0, 3)
    assert builder.system.pair_prefactors.shape == (0, 0)
    assert builder.system.pair_rows.shape == (0,)
    assert builder.system.joltqc_quartet_layout.shell_quartets.shape[1] == 4
    assert builder.joltqc_shell_quartets.shape[1] == 4
    with pytest.raises(RuntimeError, match="include_pair_metadata=True"):
        builder.build_jk(jnp.eye(basis.nao, dtype=jnp.float64), density_cutoff=0.0)


def test_cuda_ao_system_records_shell_ao_layout_for_screened_tasks():
    basis = basis_from_spec(
        "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
    )

    system = cuda_direct_jk.extract_cuda_ao_system(basis, max_l=2)

    np.testing.assert_array_equal(
        system.shell_ao_sizes,
        np.asarray(basis.shell_ao_sizes, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        system.shell_ao_indices_padded,
        np.asarray(basis.shell_ao_indices_padded, dtype=np.int32),
    )


def test_cuda_direct_jk_builder_pools_pair_schwarz_by_shell_pair(monkeypatch, tmp_path):
    basis = basis_from_spec(
        "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
    )
    npair = basis.nao * (basis.nao + 1) // 2
    raw_bounds = np.arange(npair, dtype=np.float64) + 1.0

    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_library", lambda self: tmp_path / "libfake.so")
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_and_register", lambda self: None)

    def fake_ffi_call(target_name, result_shape_dtypes, *args, **kwargs):
        assert target_name == "td_graddft_cuda_pair_schwarz"
        return raw_bounds

    monkeypatch.setattr("td_graddft.scf.cuda_direct_jk._ffi_call", fake_ffi_call)

    builder = CudaDirectJKBuilder(basis, cache_dir=tmp_path)
    pooled = np.asarray(builder.build_pair_schwarz())
    group_ids = np.asarray(builder.system.pair_screen_group_ids)

    assert any(np.count_nonzero(group_ids == group_id) > 1 for group_id in np.unique(group_ids))
    for group_id in np.unique(group_ids):
        mask = group_ids == group_id
        assert np.allclose(pooled[mask], raw_bounds[mask].max())


def test_cuda_direct_jk_builder_builds_shell_log_q_matrix_from_screen_groups(monkeypatch, tmp_path):
    basis = basis_from_spec("H 0 0 0; H 0 0 0.74", basis="sto-3g")

    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_library", lambda self: tmp_path / "libfake.so")
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_and_register", lambda self: None)

    builder = CudaDirectJKBuilder(basis, cache_dir=tmp_path)
    builder._pair_schwarz = np.asarray([2.0, 3.0, 5.0], dtype=np.float64)

    log_q = np.asarray(builder.build_shell_log_q_matrix())

    np.testing.assert_allclose(
        log_q,
        np.log(
            np.asarray(
                [
                    [2.0, 3.0],
                    [3.0, 5.0],
                ],
                dtype=np.float64,
            )
        ),
    )


def test_cuda_direct_jk_builder_builds_shell_density_condition_from_shell_blocks(monkeypatch, tmp_path):
    basis = basis_from_spec(
        "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
    )

    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_library", lambda self: tmp_path / "libfake.so")
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_and_register", lambda self: None)

    builder = CudaDirectJKBuilder(basis, cache_dir=tmp_path)
    density = np.arange(basis.nao * basis.nao, dtype=np.float64).reshape(basis.nao, basis.nao) - 7.0

    dm_cond = np.asarray(builder.build_shell_density_condition(density))
    expected = np.zeros_like(dm_cond)
    for i, shell_i in enumerate(basis.shells):
        ao_i = np.asarray(shell_i.ao_indices, dtype=np.int32)
        for j, shell_j in enumerate(basis.shells):
            ao_j = np.asarray(shell_j.ao_indices, dtype=np.int32)
            expected[i, j] = np.max(np.abs(density[np.ix_(ao_i, ao_j)]))

    np.testing.assert_allclose(dm_cond, expected)


def test_make_joltqc_tile_pairs_uses_lower_triangle_and_sorts_by_tile_q():
    log_q = np.full((8, 8), -100.0, dtype=np.float32)
    log_q[:4, :4] = -1.0
    log_q[4:, :4] = -2.0
    log_q[:4, 4:] = -2.0
    log_q[4:, 4:] = -3.0

    tile_pairs = cuda_direct_jk._make_joltqc_tile_pairs(
        np.asarray([0, 8], dtype=np.int32),
        log_q,
        cutoff=-4.0,
        tile_size=4,
    )

    assert list(tile_pairs.keys()) == [(0, 0)]
    np.testing.assert_array_equal(tile_pairs[(0, 0)], np.asarray([3, 2, 0], dtype=np.int32))


def test_pack_joltqc_tile_pairs_preserves_group_pair_slices():
    packed = cuda_direct_jk._pack_joltqc_tile_pairs(
        {
            (1, 0): np.asarray([7, 5], dtype=np.int32),
            (0, 0): np.asarray([3, 2, 0], dtype=np.int32),
        }
    )

    np.testing.assert_array_equal(
        packed.group_pair_keys,
        np.asarray(
            [
                [0, 0],
                [1, 0],
            ],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        packed.group_pair_offsets,
        np.asarray([0, 3, 5], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        packed.tile_pair_ids,
        np.asarray([3, 2, 0, 7, 5], dtype=np.int32),
    )


def test_cuda_direct_jk_builder_uses_screened_shell_task_ffi_for_cutoff(monkeypatch, tmp_path):
    basis = basis_from_spec("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    npair = basis.nao * (basis.nao + 1) // 2
    density = np.asarray(
        [
            [0.83, 0.21],
            [0.21, 0.71],
        ],
        dtype=np.float64,
    )
    calls = []

    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_library", lambda self: tmp_path / "libfake.so")
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_and_register", lambda self: None)

    def fake_ffi_call(target_name, result_shape_dtypes, *args, **kwargs):
        calls.append((target_name, result_shape_dtypes, args, kwargs))
        if target_name == "td_graddft_cuda_pair_schwarz":
            return np.arange(npair, dtype=np.float64) + 1.0
        return np.ones_like(density), 2.0 * np.ones_like(density)

    monkeypatch.setattr("td_graddft.scf.cuda_direct_jk._ffi_call", fake_ffi_call)

    builder = CudaDirectJKBuilder(basis, cache_dir=tmp_path)
    j_mat, k_mat = builder.build_jk(density, density_cutoff=1e-8)
    builder.build_jk(density, density_cutoff=1e-8)

    assert [call[0] for call in calls] == [
        "td_graddft_cuda_pair_schwarz",
        "td_graddft_cuda_screened_direct_jk",
        "td_graddft_cuda_screened_direct_jk",
    ]
    screened_args = calls[1][2]
    assert len(screened_args) == 18
    assert np.allclose(np.asarray(screened_args[9]), np.arange(npair, dtype=np.float64) + 1.0)
    assert np.allclose(np.asarray(screened_args[10]), np.asarray([np.log(1e-8)], dtype=np.float64))
    assert np.asarray(screened_args[11]).shape == (len(basis.shells), len(basis.shells))
    assert np.asarray(screened_args[12]).shape == (len(basis.shells), len(basis.shells))
    assert np.asarray(screened_args[17]).ndim == 1
    assert np.asarray(screened_args[17]).size > 0
    assert np.allclose(np.asarray(j_mat), 1.0)
    assert np.allclose(np.asarray(k_mat), 2.0)


def test_cuda_direct_jk_builder_uses_rys_screening_when_available(monkeypatch, tmp_path):
    basis = basis_from_spec("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    npair = basis.nao * (basis.nao + 1) // 2
    density = np.asarray(
        [
            [0.83, 0.21],
            [0.21, 0.71],
        ],
        dtype=np.float64,
    )
    calls = []

    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_library", lambda self: tmp_path / "libfake.so")
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_and_register", lambda self: None)

    def fake_ffi_call(target_name, result_shape_dtypes, *args, **kwargs):
        calls.append((target_name, result_shape_dtypes, args, kwargs))
        if target_name == "td_graddft_cuda_pair_schwarz":
            return np.arange(npair, dtype=np.float64) + 1.0
        return np.ones_like(density), 2.0 * np.ones_like(density)

    monkeypatch.setattr("td_graddft.scf.cuda_direct_jk._ffi_call", fake_ffi_call)

    builder = CudaDirectJKBuilder(basis, cache_dir=tmp_path)
    builder.has_rys_direct_jk = True
    j_mat, k_mat = builder.build_jk(density, density_cutoff=1e-8)

    assert [call[0] for call in calls] == [
        "td_graddft_cuda_pair_schwarz",
        "td_graddft_cuda_rys_direct_jk",
    ]
    rys_args = calls[1][2]
    assert len(rys_args) == 12
    assert np.asarray(rys_args[7]).shape == (len(basis.shells), len(basis.shells))
    assert np.asarray(rys_args[8]).shape == (len(basis.shells), len(basis.shells))
    assert np.asarray(rys_args[11]).dtype == np.float32
    assert np.allclose(np.asarray(rys_args[11]), np.asarray([np.log(1e-8)], dtype=np.float32))
    assert np.allclose(np.asarray(j_mat), 1.0)
    assert np.allclose(np.asarray(k_mat), 2.0)


def test_cuda_direct_jk_builder_builds_grouped_screening_metadata_for_shell_tasks(
    monkeypatch,
    tmp_path,
):
    basis = basis_from_spec(
        "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
    )
    npair = basis.nao * (basis.nao + 1) // 2
    density = np.arange(basis.nao * basis.nao, dtype=np.float64).reshape(basis.nao, basis.nao) / 10.0
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_library", lambda self: tmp_path / "libfake.so")
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_and_register", lambda self: None)

    def fake_ffi_call(target_name, result_shape_dtypes, *args, **kwargs):
        if target_name == "td_graddft_cuda_pair_schwarz":
            return np.arange(npair, dtype=np.float64) + 1.0
        return np.ones_like(density), 2.0 * np.ones_like(density)

    monkeypatch.setattr("td_graddft.scf.cuda_direct_jk._ffi_call", fake_ffi_call)

    builder = CudaDirectJKBuilder(basis, cache_dir=tmp_path)
    shell_log_q = np.asarray(builder.build_shell_log_q_matrix())
    assert shell_log_q.shape == (len(basis.shells), len(basis.shells))
    density_sym = 0.5 * (density + density.T)
    shell_dm_cond = np.asarray(builder.build_shell_density_condition(density_sym))
    assert shell_dm_cond.shape == (len(basis.shells), len(basis.shells))
    np.testing.assert_array_equal(
        np.asarray(builder.system.shell_ao_indices_padded, dtype=np.int32),
        np.asarray(builder.system.shell_ao_indices_padded, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(builder.system.shell_ao_sizes, dtype=np.int32),
        np.asarray(builder.system.shell_ao_sizes, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(builder.system.shell_layout.tile_shell_indices, dtype=np.int32),
        np.asarray(builder.system.shell_layout.tile_shell_indices, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(builder.system.shell_layout.tile_shell_pad_mask, dtype=np.int32),
        np.asarray(builder.system.shell_layout.tile_shell_pad_mask, dtype=np.int32),
    )
    assert np.asarray(builder.build_full_joltqc_tile_pair_layout().tile_pair_ids).size > 0


def test_cuda_direct_jk_builder_precomputes_primitive_normalized_coefficients(monkeypatch, tmp_path):
    basis = basis_from_spec("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    original = np.asarray(basis.aos[0].coefficients, dtype=np.float64)
    exponents = np.asarray(basis.aos[0].exponents, dtype=np.float64)
    angular = tuple(int(power) for power in basis.aos[0].angular)

    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_library", lambda self: tmp_path / "libfake.so")
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_and_register", lambda self: None)

    builder = CudaDirectJKBuilder(basis, cache_dir=tmp_path)

    assert np.allclose(
        np.asarray(builder.system.coefficients[0, : original.shape[0]]),
        original * _primitive_cart_norm(exponents, angular),
    )


def test_cuda_direct_jk_builder_precomputes_primitive_pair_data(monkeypatch, tmp_path):
    basis = basis_from_spec("H 0 0 0; H 0 0 0.74", basis="sto-3g")

    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_library", lambda self: tmp_path / "libfake.so")
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_and_register", lambda self: None)

    builder = CudaDirectJKBuilder(basis, cache_dir=tmp_path)

    assert builder.system.pair_exponents.shape[0] == 3
    assert builder.system.pair_centers.shape[:2] == builder.system.pair_exponents.shape
    assert builder.system.pair_centers.shape[2] == 3
    assert builder.system.pair_prefactors.shape == builder.system.pair_exponents.shape
    nprim0 = int(basis.aos[0].exponents.shape[0])
    assert np.all(builder.system.pair_nprims == nprim0 * nprim0)
    assert np.isclose(builder.system.pair_exponents[0, 0], 2.0 * builder.system.exponents[0, 0])
    assert np.allclose(builder.system.pair_centers[0, 0], builder.system.centers[0])


def test_cuda_direct_jk_builder_precomputes_pair_mapping_data(monkeypatch, tmp_path):
    basis = basis_from_spec("H 0 0 0; H 0 0 0.74", basis="sto-3g")

    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_library", lambda self: tmp_path / "libfake.so")
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_and_register", lambda self: None)

    builder = CudaDirectJKBuilder(basis, cache_dir=tmp_path)

    assert np.allclose(builder.system.pair_rows, np.asarray([0, 1, 1], dtype=np.int32))
    assert np.allclose(builder.system.pair_cols, np.asarray([0, 0, 1], dtype=np.int32))
    np.testing.assert_array_equal(
        np.asarray(builder.rys_env_layout.ao_to_parent_ao, dtype=np.int32),
        np.asarray(jax.device_get(builder.rys_ao_to_parent_ao), dtype=np.int32),
    )
    full_rys_pairs = builder.build_full_rys_pair_mapping_layout()
    assert full_rys_pairs.group_pair_keys.shape[1] == 2
    assert full_rys_pairs.group_pair_offsets[-1] == full_rys_pairs.pair_ids.shape[0]
    assert full_rys_pairs.pair_ids.shape[0] == len(basis.shells) * (len(basis.shells) + 1) // 2


def test_cuda_direct_jk_builder_can_build_full_eri_tensor(monkeypatch, tmp_path):
    basis = basis_from_spec("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    captured = {}

    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_library", lambda self: tmp_path / "libfake.so")
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_and_register", lambda self: None)

    def fake_ffi_call(target_name, result_shape_dtypes, *args, **kwargs):
        captured["target_name"] = target_name
        captured["result_shape_dtypes"] = result_shape_dtypes
        captured["args"] = args
        captured["kwargs"] = kwargs
        return np.ones(result_shape_dtypes.shape, dtype=np.float64)

    monkeypatch.setattr("td_graddft.scf.cuda_direct_jk._ffi_call", fake_ffi_call)

    builder = CudaDirectJKBuilder(basis, cache_dir=tmp_path)
    eri = builder.build_eri_tensor()

    assert captured["target_name"] == "td_graddft_cuda_eri_tensor"
    assert captured["result_shape_dtypes"].shape == (basis.nao, basis.nao, basis.nao, basis.nao)
    assert len(captured["args"]) == 8
    assert np.asarray(eri).shape == (basis.nao, basis.nao, basis.nao, basis.nao)


def test_cuda_direct_jk_builder_can_build_eri_pair_matrix(monkeypatch, tmp_path):
    basis = basis_from_spec("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    npair = basis.nao * (basis.nao + 1) // 2
    captured = {}

    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_library", lambda self: tmp_path / "libfake.so")
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_and_register", lambda self: None)

    def fake_ffi_call(target_name, result_shape_dtypes, *args, **kwargs):
        captured["target_name"] = target_name
        captured["result_shape_dtypes"] = result_shape_dtypes
        captured["args"] = args
        captured["kwargs"] = kwargs
        return np.ones(result_shape_dtypes.shape, dtype=np.float64)

    monkeypatch.setattr("td_graddft.scf.cuda_direct_jk._ffi_call", fake_ffi_call)

    builder = CudaDirectJKBuilder(basis, cache_dir=tmp_path)
    pair = builder.build_eri_pair_matrix()

    assert captured["target_name"] == "td_graddft_cuda_eri_pair_matrix"
    assert captured["result_shape_dtypes"].shape == (npair, npair)
    assert len(captured["args"]) == 10
    assert np.allclose(np.asarray(captured["args"][-2]), 1.0)
    np.testing.assert_array_equal(np.asarray(captured["args"][-1]), np.asarray([1e-8], dtype=np.float64))
    assert np.asarray(pair).shape == (npair, npair)


def test_cuda_direct_jk_builder_passes_pair_schwarz_to_screened_pair_eri(monkeypatch, tmp_path):
    basis = basis_from_spec("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    npair = basis.nao * (basis.nao + 1) // 2
    calls = []

    monkeypatch.setenv("TD_GRADDFT_CUDA_PAIR_ERI_BUILD_CUTOFF", "1e-12")
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_library", lambda self: tmp_path / "libfake.so")
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_and_register", lambda self: None)

    def fake_ffi_call(target_name, result_shape_dtypes, *args, **kwargs):
        calls.append((target_name, result_shape_dtypes, args, kwargs))
        if target_name == "td_graddft_cuda_pair_schwarz":
            return np.arange(npair, dtype=np.float64) + 1.0
        return np.ones(result_shape_dtypes.shape, dtype=np.float64)

    monkeypatch.setattr("td_graddft.scf.cuda_direct_jk._ffi_call", fake_ffi_call)

    builder = CudaDirectJKBuilder(basis, cache_dir=tmp_path)
    pair = builder.build_eri_pair_matrix()

    assert [call[0] for call in calls] == [
        "td_graddft_cuda_pair_schwarz",
        "td_graddft_cuda_eri_pair_matrix",
    ]
    eri_args = calls[1][2]
    assert len(eri_args) == 10
    assert np.allclose(np.asarray(eri_args[-2]), np.arange(npair, dtype=np.float64) + 1.0)
    assert np.allclose(np.asarray(eri_args[-1]), np.asarray([1e-12], dtype=np.float64))
    assert np.asarray(pair).shape == (npair, npair)


def test_cuda_direct_jk_builder_can_contract_cached_pair_matrix(monkeypatch, tmp_path):
    basis = basis_from_spec("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    npair = basis.nao * (basis.nao + 1) // 2
    pair_matrix = np.ones((npair, npair), dtype=np.float64)
    density = np.asarray(
        [
            [0.83, 0.21],
            [0.21, 0.71],
        ],
        dtype=np.float64,
    )
    captured = {}

    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_library", lambda self: tmp_path / "libfake.so")
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_and_register", lambda self: None)
    monkeypatch.setattr(
        "td_graddft.scf.cuda_direct_jk.ensure_cuda_pair_matrix_jk_ffi_registered",
        lambda **kwargs: True,
    )

    def fake_ffi_call(target_name, result_shape_dtypes, *args, **kwargs):
        captured["target_name"] = target_name
        captured["result_shape_dtypes"] = result_shape_dtypes
        captured["args"] = args
        captured["kwargs"] = kwargs
        return np.ones_like(density), 2.0 * np.ones_like(density)

    monkeypatch.setattr("td_graddft.scf.cuda_direct_jk._ffi_call", fake_ffi_call)

    builder = CudaDirectJKBuilder(basis, cache_dir=tmp_path)
    j_mat, k_mat = builder.build_jk_from_eri_pair_matrix(pair_matrix, density)

    assert captured["target_name"] == "td_graddft_cuda_pair_matrix_jk"
    assert len(captured["result_shape_dtypes"]) == 2
    assert captured["result_shape_dtypes"][0].shape == density.shape
    assert len(captured["args"]) == 4
    assert np.asarray(captured["args"][0]).shape == (npair, npair)
    assert np.allclose(np.asarray(j_mat), 1.0)
    assert np.allclose(np.asarray(k_mat), 2.0)


def test_cuda_cached_pair_matrix_jk_has_density_vjp(monkeypatch, tmp_path):
    from td_graddft.scf.packed_eri import build_jk_from_eri_pair_matrix

    basis = basis_from_spec("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    npair = basis.nao * (basis.nao + 1) // 2
    pair_matrix = jnp.asarray(
        [
            [1.30, 0.21, 0.17],
            [0.21, 0.93, 0.11],
            [0.17, 0.11, 0.81],
        ],
        dtype=jnp.float64,
    )
    density = jnp.asarray(
        [
            [0.83, 0.21],
            [0.21, 0.71],
        ],
        dtype=jnp.float64,
    )
    probe_j = jnp.asarray(
        [
            [0.20, -0.31],
            [0.42, 0.53],
        ],
        dtype=jnp.float64,
    )
    probe_k = jnp.asarray(
        [
            [0.71, 0.13],
            [-0.19, 0.37],
        ],
        dtype=jnp.float64,
    )

    assert npair == pair_matrix.shape[0]
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_library", lambda self: tmp_path / "libfake.so")
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_and_register", lambda self: None)
    monkeypatch.setattr(
        "td_graddft.scf.cuda_direct_jk.ensure_cuda_pair_matrix_jk_ffi_registered",
        lambda **kwargs: True,
    )

    def fake_ffi_call(target_name, result_shape_dtypes, *args, **kwargs):
        del target_name, result_shape_dtypes, kwargs
        j_mat, k_mat = build_jk_from_eri_pair_matrix(args[0], args[1])
        return jax.lax.stop_gradient(j_mat), jax.lax.stop_gradient(k_mat)

    monkeypatch.setattr("td_graddft.scf.cuda_direct_jk._ffi_call", fake_ffi_call)

    builder = CudaDirectJKBuilder(basis, cache_dir=tmp_path)

    def cuda_objective(density_arg):
        j_mat, k_mat = builder.build_jk_from_eri_pair_matrix(pair_matrix, density_arg)
        return jnp.sum(j_mat * probe_j) + jnp.sum(k_mat * probe_k)

    def reference_objective(density_arg):
        j_mat, k_mat = build_jk_from_eri_pair_matrix(pair_matrix, density_arg)
        return jnp.sum(j_mat * probe_j) + jnp.sum(k_mat * probe_k)

    cuda_grad = jax.grad(cuda_objective)(density)
    reference_grad = jax.grad(reference_objective)(density)

    assert np.max(np.abs(np.asarray(reference_grad))) > 1.0e-8
    assert np.allclose(np.asarray(cuda_grad), np.asarray(reference_grad), atol=1.0e-12)


def test_cuda_direct_jk_has_density_vjp_without_pair_cache(monkeypatch, tmp_path):
    from td_graddft.scf.packed_eri import build_jk_from_eri_pair_matrix

    basis = basis_from_spec("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    pair_matrix = jnp.asarray(
        [
            [1.30, 0.21, 0.17],
            [0.21, 0.93, 0.11],
            [0.17, 0.11, 0.81],
        ],
        dtype=jnp.float64,
    )
    density = jnp.asarray(
        [
            [0.83, 0.21],
            [0.21, 0.71],
        ],
        dtype=jnp.float64,
    )
    probe_j = jnp.asarray(
        [
            [0.20, -0.31],
            [0.42, 0.53],
        ],
        dtype=jnp.float64,
    )
    probe_k = jnp.asarray(
        [
            [0.71, 0.13],
            [-0.19, 0.37],
        ],
        dtype=jnp.float64,
    )

    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_library", lambda self: tmp_path / "libfake.so")
    monkeypatch.setattr(CudaDirectJKBuilder, "_compile_and_register", lambda self: None)

    def fake_ffi_call(target_name, result_shape_dtypes, *args, **kwargs):
        del result_shape_dtypes, kwargs
        if target_name != "td_graddft_cuda_direct_jk":
            raise AssertionError(f"unexpected FFI target {target_name!r}")
        j_mat, k_mat = build_jk_from_eri_pair_matrix(pair_matrix, args[0])
        return jax.lax.stop_gradient(j_mat), jax.lax.stop_gradient(k_mat)

    monkeypatch.setattr("td_graddft.scf.cuda_direct_jk._ffi_call", fake_ffi_call)

    builder = CudaDirectJKBuilder(basis, cache_dir=tmp_path)

    def cuda_objective(density_arg):
        j_mat, k_mat = builder.build_jk(density_arg, density_cutoff=0.0)
        return jnp.sum(j_mat * probe_j) + jnp.sum(k_mat * probe_k)

    def reference_objective(density_arg):
        j_mat, k_mat = build_jk_from_eri_pair_matrix(pair_matrix, density_arg)
        return jnp.sum(j_mat * probe_j) + jnp.sum(k_mat * probe_k)

    cuda_grad = jax.grad(cuda_objective)(density)
    reference_grad = jax.grad(reference_objective)(density)

    assert np.max(np.abs(np.asarray(reference_grad))) > 1.0e-8
    assert np.allclose(np.asarray(cuda_grad), np.asarray(reference_grad), atol=1.0e-12)


def test_cuda_direct_jk_kernel_uses_unique_pair_quartet_launch():
    source = cuda_direct_jk._kernel_source_path().read_text()

    assert "unique_pair_quartet_direct_jk_kernel" in source
    assert "const long long npair" in source
    assert "npair * (npair + 1) / 2" in source
    assert "static_cast<long long>(nao) * nao * nao * nao" not in source
    assert "density_cutoff.typed_data()[0]" not in source


def test_cuda_direct_jk_kernel_uses_schwarz_density_screening():
    source = cuda_direct_jk._kernel_source_path().read_text()

    assert "pair_schwarz_kernel" in source
    assert "pair_schwarz[pair_p] * pair_schwarz[pair_q]" in source
    assert "abs_max_density_for_quartet" in source


def test_cuda_direct_jk_kernel_has_unrolled_ssss_pair_eri_fast_path():
    source = cuda_direct_jk._kernel_source_path().read_text()

    assert "contracted_pair_eri_ssss" in source
    assert "if (max_boys_order == 0)" in source
    assert "return contracted_pair_eri_ssss(" in source
    assert "primitive_pair_eri(" in source


def test_cuda_direct_jk_kernel_has_unrolled_single_p_pair_eri_fast_path():
    source = cuda_direct_jk._kernel_source_path().read_text()

    assert "contracted_pair_eri_single_p" in source
    assert "primitive_pair_eri_single_p" in source
    assert "if (max_boys_order == 1)" in source
    assert "return contracted_pair_eri_single_p(" in source


def test_cuda_direct_jk_kernel_has_unrolled_two_p_pair_eri_fast_path():
    source = cuda_direct_jk._kernel_source_path().read_text()

    assert "contracted_pair_eri_two_p" in source
    assert "primitive_pair_eri_two_p" in source
    assert "is_single_p_angular" in source
    assert "return contracted_pair_eri_two_p(" in source


def test_cuda_direct_jk_kernel_has_joltqc_basis_data_bridge_helpers():
    source = cuda_direct_jk._kernel_source_path().read_text()

    assert "constexpr int kJoltQCBasisStride = 12;" in source
    assert "JoltQCData4" in source
    assert "JoltQCData2" in source
    assert "load_joltqc_ce_ptr" in source
    assert "expand_joltqc_density_kernel" in source
    assert "contract_joltqc_potential_kernel" in source
    assert "const int* ao_to_parent_ao" in source


def test_cuda_direct_jk_kernel_exposes_joltqc_grouped_ffi_target():
    source = cuda_direct_jk._kernel_source_path().read_text()

    assert "CudaJoltQCDirectJkDispatch" in source
    assert "auto CudaJoltQCDirectJkBinding()" in source
    assert "TdGraddftCudaJoltQCDirectJkFfi" in source
    assert "TdGraddftLaunchJoltQC1qnt" in source
    assert "cudaErrorNotSupported" in source
    assert "joltqc_shell_quartet_direct_jk_kernel<<<" in source
    assert "finalize_joltqc_potential_kernel" in source
    assert "if (fast_err == cudaSuccess)" in source
    assert "2.0 * (j_mat[i * n + j] + j_mat[j * n + i])" in source
    joltqc_kernel = source.split(
        "__global__ void joltqc_shell_quartet_direct_jk_kernel", 1
    )[1].split("__global__ void screened_shell_quartet_direct_jk_kernel", 1)[0]
    assert "contracted_pair_eri(" not in joltqc_kernel
    assert "joltqc_contracted_eri(" in joltqc_kernel
    assert "if (pair_p != pair_q)" in joltqc_kernel
    assert "atomicAdd(k_mat + k * joltqc_nao + i" in joltqc_kernel


def test_cuda_rys_dispatch_does_not_use_group_slot_as_pair_order():
    source = cuda_direct_jk._kernel_source_path().read_text()

    assert "h_pair_offsets.back() != static_cast<int>(pair_ids.dimensions()[0])" in source
    assert "std::vector<int> h_pair_ids" not in source
    assert "ij_slot < kl_slot" not in source


def test_cuda_direct_jk_kernel_keeps_joltqc_launcher_weak_fallback_external():
    source = cuda_direct_jk._kernel_source_path().read_text()

    namespace_end = source.index("}  // namespace")
    weak_launcher = source.index(
        '__attribute__((weak)) cudaError_t TdGraddftLaunchJoltQC1qnt'
    )

    assert weak_launcher > namespace_end


def test_cuda_direct_jk_positive_cutoff_uses_joltqc_style_task_queue():
    source = cuda_direct_jk._kernel_source_path().read_text()

    assert "screen_shell_quartet_tasks" in source
    assert "shell_dm_cond" in source
    assert "ShellQuartetTask" in source
    assert "screened_shell_quartet_direct_jk_kernel" in source
    assert "const bool use_task_queue = false;" not in source
    assert "CudaScreenedDirectJkDispatch" in source
    assert "screen_shell_quartet_tasks<<<" in source
    assert "screened_shell_quartet_direct_jk_kernel<<<" in source
    assert "TdGraddftCudaScreenedDirectJkFfi" in source


def test_cuda_screened_task_generation_does_not_assume_sorted_tile_pair_ids():
    source = cuda_direct_jk._kernel_source_path().read_text()

    assert "const long long total = n_tile_pairs * n_tile_pairs;" in source
    assert "const long long tile_pair_p_idx = idx / n_tile_pairs;" in source
    assert "const long long tile_pair_q_idx = idx - tile_pair_p_idx * n_tile_pairs;" in source
    assert "decode_pair_quartet(idx, &tile_pair_p_idx, &tile_pair_q_idx)" not in source


def test_cuda_screened_direct_jk_binding_accepts_grouped_metadata_buffers():
    source = cuda_direct_jk._kernel_source_path().read_text()

    assert "auto CudaScreenedDirectJkBinding()" in source
    assert source.count(".Arg<F64Buffer2>()") >= 3
    assert "Arg<S32Buffer2>()" in source
    assert "Arg<S32Buffer1>()" in source


def test_cuda_screened_shell_quartet_kernel_keeps_same_shell_ao_pairs_lower_triangular():
    source = cuda_direct_jk._kernel_source_path().read_text()

    assert "if (task.i == task.j && ja > ia)" in source
    assert "if (task.k == task.l && la > ka)" in source
    assert "if (i_base < j_base)" in source
    assert "if (k_base < l_base)" in source


def test_cuda_screened_task_generation_canonicalizes_shell_pairs_after_tile_ordering():
    source = cuda_direct_jk._kernel_source_path().read_text()

    assert "if (tile_i == tile_j && ii < jj)" in source
    assert "if (tile_k == tile_l && kk < ll)" in source
    assert "int p_i = ish;" in source
    assert "int p_j = jsh;" in source
    assert "int q_i = ksh;" in source
    assert "int q_j = lsh;" in source
    assert "if (p_i < p_j)" in source
    assert "if (q_i < q_j)" in source
    assert "ShellQuartetTask{p_i, p_j, q_i, q_j}" in source
    assert "if (ish < jsh)" not in source
    assert "if (ksh < lsh)" not in source


def test_cuda_screened_shell_quartet_kernel_swaps_crossing_ao_pair_order():
    source = cuda_direct_jk._kernel_source_path().read_text()
    screened_kernel = source.split("__global__ void screened_shell_quartet_direct_jk_kernel", 1)[1].split(
        "__global__ void symmetrize_kernel", 1
    )[0]

    assert "if (pair_p < pair_q)" in screened_kernel
    assert "const long long tmp_pair = pair_p;" in screened_kernel
    assert "pair_p = pair_q;" in screened_kernel
    assert "pair_q = tmp_pair;" in screened_kernel
    assert "const long long shell_pair_p = lower_pair_id(task.i, task.j);" in screened_kernel
    assert "const long long shell_pair_q = lower_pair_id(task.k, task.l);" in screened_kernel
    assert "if (shell_pair_p == shell_pair_q)" in screened_kernel
    assert "const int tmp_i = i;" in screened_kernel
    assert "const int tmp_j = j;" in screened_kernel
    assert "i = k;" in screened_kernel
    assert "j = l;" in screened_kernel
    assert "k = tmp_i;" in screened_kernel
    assert "l = tmp_j;" in screened_kernel
    assert "if (pair_p < pair_q) {\n                            continue;" not in screened_kernel


def test_cuda_screened_shell_quartet_kernel_parallelizes_ao_quartets_within_blocks():
    source = cuda_direct_jk._kernel_source_path().read_text()
    screened_kernel = source.split("__global__ void screened_shell_quartet_direct_jk_kernel", 1)[1].split(
        "__global__ void symmetrize_kernel", 1
    )[0]

    assert "for (long long task_id = blockIdx.x;" in screened_kernel
    assert "task_id += gridDim.x" in screened_kernel
    assert "long long local_quartet = 0;" in screened_kernel
    assert "const long long quartet_id = local_quartet++;" in screened_kernel
    assert "if ((quartet_id % blockDim.x) != threadIdx.x)" in screened_kernel
    assert "blockIdx.x * blockDim.x + threadIdx.x" not in screened_kernel


def test_cuda_pair_matrix_build_uses_schwarz_screening_cutoff():
    source = cuda_direct_jk._kernel_source_path().read_text()

    assert "pair_schwarz[pair_p] * pair_schwarz[pair_q] < eri_cutoff[0]" in source
    assert "ffi::Buffer<ffi::F64, 1> pair_schwarz" in source
    assert "ffi::Buffer<ffi::F64, 1> eri_cutoff" in source


def test_cuda_direct_jk_kernel_uses_precomputed_pair_mapping():
    source = cuda_direct_jk._kernel_source_path().read_text()

    assert "const int* pair_rows" in source
    assert "const int* pair_cols" in source
    assert "decode_lower_pair(pair_p" not in source


def test_cuda_pair_matrix_k_kernel_computes_only_lower_triangle():
    source = cuda_direct_jk._kernel_source_path().read_text()

    assert "for (long long pair_p = blockIdx.x * blockDim.x + threadIdx.x;" in source
    assert "const int p = pair_rows[pair_p];" in source
    assert "if (p != q)" in source
    assert "const long long total = static_cast<long long>(nao) * nao;" not in source


def test_cuda_pair_matrix_jk_cache_uses_block_reduction_tasks():
    source = cuda_direct_jk._kernel_source_path().read_text()

    assert "pair_matrix_j_reduce_kernel" in source
    assert "pair_matrix_k_reduce_kernel" in source
    assert "extern __shared__ double partials[]" in source
    assert "const long long pair_p = static_cast<long long>(blockIdx.x);" in source
    assert "for (long long pair_q = static_cast<long long>(threadIdx.x);" in source
    assert "for (long long flat = static_cast<long long>(threadIdx.x);" in source
    assert "block * sizeof(double)" in source


def test_cuda_pair_matrix_jk_cache_keeps_small_system_serial_path():
    source = cuda_direct_jk._kernel_source_path().read_text()

    assert "constexpr int kPairMatrixReductionMinNao" in source
    assert "pair_matrix_j_kernel<<<" in source
    assert "pair_matrix_k_kernel<<<" in source
    assert "pair_matrix_j_reduce_kernel<<<" in source
    assert "pair_matrix_k_reduce_kernel<<<" in source
    assert "if (nao >= kPairMatrixReductionMinNao)" in source
    assert "const long long row_offset = pair_p * npair + (static_cast<long long>(k) * (k + 1)) / 2;" in source


def test_cuda_pair_matrix_build_uses_unique_pair_quartet_kernel():
    source = cuda_direct_jk._kernel_source_path().read_text()

    assert "unique_pair_quartet_eri_pair_matrix_kernel" in source
    assert "const int eri_pair_block = 256;" in source
    assert "decode_pair_quartet" in source
    assert "shell_quartet_i" not in source


def test_cuda_primitive_pair_eri_limits_boys_order_to_quartet_angular_momentum():
    source = cuda_direct_jk._kernel_source_path().read_text()

    assert "const int max_boys_order =" in source
    assert "lsum(angular_i) + lsum(angular_j) + lsum(angular_k) + lsum(angular_l)" in source
    assert "boys_values(max_boys_order" in source
    assert "boys_values(kBoysMax" not in source
    primitive_pair_source = source.split("__host__ __device__ double primitive_pair_eri(", 1)[1].split(
        "__host__ __device__ double contracted_pair_eri(", 1
    )[0]
    assert "lsum(angular_a)" not in primitive_pair_source


def test_cuda_primitive_pair_eri_has_ssss_fast_path():
    source = cuda_direct_jk._kernel_source_path().read_text()

    primitive_pair_source = source.split("__host__ __device__ double primitive_pair_eri(", 1)[1].split(
        "__host__ __device__ double contracted_pair_eri(", 1
    )[0]
    assert "if (max_boys_order == 0)" in primitive_pair_source
    assert "return prefactor * boys0(t);" in primitive_pair_source
    assert "hrr(ctx, angular_a, angular_b, angular_c, angular_d)" in primitive_pair_source




def test_ffi_call_supports_old_jax_extend_signature(monkeypatch):
    captured = {}

    def old_ffi_call(target_name, result_shape_dtypes, *args, vectorized=False, has_side_effect=False, **kwargs):
        captured["target_name"] = target_name
        captured["result_shape_dtypes"] = result_shape_dtypes
        captured["args"] = args
        captured["vectorized"] = vectorized
        captured["has_side_effect"] = has_side_effect
        captured["kwargs"] = kwargs
        return "j", "k"

    class OldFfiModule:
        ffi_call = staticmethod(old_ffi_call)

    monkeypatch.setattr(cuda_direct_jk, "_ffi_module", lambda: OldFfiModule)

    assert cuda_direct_jk._ffi_call("target", ("shape",), "density", "basis") == ("j", "k")
    assert captured["target_name"] == "target"
    assert captured["args"] == ("density", "basis")
    assert captured["vectorized"] is False
    assert captured["has_side_effect"] is False


def test_ffi_call_supports_new_jax_ffi_signature(monkeypatch):
    captured = {}

    def new_ffi_call(target_name, result_shape_dtypes, *, has_side_effect=False, vmap_method=None):
        captured["target_name"] = target_name
        captured["result_shape_dtypes"] = result_shape_dtypes
        captured["has_side_effect"] = has_side_effect
        captured["vmap_method"] = vmap_method

        def call(*args):
            captured["args"] = args
            return "j", "k"

        return call

    class NewFfiModule:
        ffi_call = staticmethod(new_ffi_call)

    monkeypatch.setattr(cuda_direct_jk, "_ffi_module", lambda: NewFfiModule)

    assert cuda_direct_jk._ffi_call("target", ("shape",), "density", "basis") == ("j", "k")
    assert captured["target_name"] == "target"
    assert captured["args"] == ("density", "basis")
    assert captured["vmap_method"] == "sequential"
    assert captured["has_side_effect"] is False


def test_detect_cuda_arch_uses_first_visible_device(monkeypatch):
    cuda_direct_jk._clear_cuda_arch_cache()
    captured = {}

    class Result:
        stdout = "12.0\nUnable to determine the device handle for GPU2\n"

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return Result()

    monkeypatch.delenv("TD_GRADDFT_CUDA_ARCH", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(cuda_direct_jk.subprocess, "run", fake_run)

    assert cuda_direct_jk._detect_cuda_arch() == "sm_120"
    assert "--id=0" in captured["command"]
    assert captured["kwargs"]["check"] is True


def test_detect_cuda_arch_falls_back_on_unparseable_nvidia_smi(monkeypatch):
    cuda_direct_jk._clear_cuda_arch_cache()

    class Result:
        stdout = "Unable to determine the device handle for GPU2\n"

    def fake_run(command, **kwargs):
        del command, kwargs
        return Result()

    monkeypatch.delenv("TD_GRADDFT_CUDA_ARCH", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(cuda_direct_jk.subprocess, "run", fake_run)

    assert cuda_direct_jk._detect_cuda_arch() == "sm_80"


def test_detect_cuda_arch_uses_global_query_when_visible_id_is_unavailable(monkeypatch):
    cuda_direct_jk._clear_cuda_arch_cache()
    calls = []

    class Result:
        stdout = "12.0\n"

    def fake_run(command, **kwargs):
        calls.append(command)
        if "--id=2" in command:
            raise cuda_direct_jk.subprocess.CalledProcessError(1, command)
        return Result()

    monkeypatch.delenv("TD_GRADDFT_CUDA_ARCH", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(cuda_direct_jk.subprocess, "run", fake_run)

    assert cuda_direct_jk._detect_cuda_arch() == "sm_120"
    assert "--id=2" in calls[0]
    assert not any(part.startswith("--id=") for part in calls[1])


def test_detect_cuda_arch_is_cached_for_same_environment(monkeypatch):
    cuda_direct_jk._clear_cuda_arch_cache()
    calls = []

    class Result:
        stdout = "12.0\n"

    def fake_run(command, **kwargs):
        calls.append(command)
        return Result()

    monkeypatch.delenv("TD_GRADDFT_CUDA_ARCH", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(cuda_direct_jk.subprocess, "run", fake_run)

    assert cuda_direct_jk._detect_cuda_arch() == "sm_120"
    assert cuda_direct_jk._detect_cuda_arch() == "sm_120"
    assert len(calls) == 1


def test_build_prebuilt_cuda_direct_jk_library_reuses_runtime_signature_library(
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "kernel.cu"
    source.write_text("__global__ void placeholder() {}\n")
    commands = []

    class Result:
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        commands.append(list(command))
        output = Path(command[command.index("-o") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"fake")
        return Result()

    monkeypatch.setattr(cuda_direct_jk.subprocess, "run", fake_run)

    first_library = cuda_direct_jk.build_prebuilt_cuda_direct_jk_library(
        tmp_path,
        source_path=source,
        nvcc="/usr/local/cuda/bin/nvcc",
        arch="sm_120",
        force=True,
        joltqc_group_keys=np.asarray([[0, 3], [1, 3]], dtype=np.int32),
        joltqc_group_quartet_keys=np.asarray([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int32),
        joltqc_group_quartet_offsets=np.asarray([0, 1, 2], dtype=np.int32),
        include_rys=False,
    )

    first_compile_sources = [
        Path(command[command.index("-c") + 1]).name for command in commands if "-c" in command
    ]
    assert any("signature_" in name for name in first_compile_sources)
    assert first_library.exists()

    commands.clear()
    second_library = cuda_direct_jk.build_prebuilt_cuda_direct_jk_library(
        tmp_path,
        source_path=source,
        nvcc="/usr/local/cuda/bin/nvcc",
        arch="sm_120",
        force=False,
        joltqc_group_keys=np.asarray([[0, 3], [1, 3]], dtype=np.int32),
        joltqc_group_quartet_keys=np.asarray([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=np.int32),
        joltqc_group_quartet_offsets=np.asarray([0, 3, 4], dtype=np.int32),
        include_rys=False,
    )

    second_compile_sources = [
        Path(command[command.index("-c") + 1]).name for command in commands if "-c" in command
    ]
    assert first_library == second_library
    assert second_library.exists()
    assert second_compile_sources == []
    assert commands == []


def test_build_prebuilt_cuda_direct_jk_library_skips_joltqc_codegen_when_cached(
    monkeypatch,
    tmp_path,
):
    from td_graddft.scf.joltqc_port import codegen

    source = tmp_path / "kernel.cu"
    source.write_text("__global__ void placeholder() {}\n")
    group_keys = np.asarray([[0, 3], [1, 3]], dtype=np.int32)
    quartet_keys = np.asarray([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int32)
    quartet_offsets = np.asarray([0, 1, 2], dtype=np.int32)
    source_key = codegen.build_1qnt_dispatch_source_key(
        group_keys,
        quartet_keys,
        quartet_offsets,
    )
    library = tmp_path / cuda_direct_jk._library_name_for_arch(
        "sm_120",
        source_path=source,
        extra_source_key=source_key,
    )
    library.write_bytes(b"cached")

    def fail_codegen(*args, **kwargs):
        raise AssertionError("cached JoltQC library should not regenerate CUDA source")

    monkeypatch.setattr(codegen, "build_1qnt_dispatch_source_units", fail_codegen)

    result = cuda_direct_jk.build_prebuilt_cuda_direct_jk_library(
        tmp_path,
        source_path=source,
        nvcc="/usr/local/cuda/bin/nvcc",
        arch="sm_120",
        force=False,
        joltqc_group_keys=group_keys,
        joltqc_group_quartet_keys=quartet_keys,
        joltqc_group_quartet_offsets=quartet_offsets,
        include_rys=False,
    )

    assert result == library


def test_build_prebuilt_cuda_direct_jk_library_rejects_generic_dispatch(tmp_path):
    source = tmp_path / "kernel.cu"
    source.write_text("__global__ void placeholder() {}\n")

    with pytest.raises(ValueError, match="joltqc_dispatch"):
        cuda_direct_jk.build_prebuilt_cuda_direct_jk_library(
            tmp_path,
            source_path=source,
            nvcc="/usr/local/cuda/bin/nvcc",
            arch="sm_120",
            force=True,
            joltqc_group_keys=np.asarray([[0, 3], [1, 3]], dtype=np.int32),
            joltqc_group_quartet_keys=np.asarray(
                [[0, 0, 0, 0], [1, 1, 1, 1]],
                dtype=np.int32,
            ),
            joltqc_group_quartet_offsets=np.asarray([0, 1, 2], dtype=np.int32),
            joltqc_dispatch="generic",
        )
