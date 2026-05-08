import numpy as np

from td_graddft.data.basis import basis_from_spec
from td_graddft.scf import cuda_one_electron
from td_graddft.scf.cuda_one_electron import CudaOneElectronBuilder


def test_cuda_one_electron_builder_invokes_ffi_call(monkeypatch, tmp_path):
    basis = basis_from_spec("H 0 0 0; H 0 0 0.74", basis="sto-3g")
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
        shape = result_shape_dtypes[0].shape
        return np.eye(shape[0]), 2.0 * np.eye(shape[0])

    monkeypatch.setattr(CudaOneElectronBuilder, "_compile_library", fake_compile_library)
    monkeypatch.setattr(CudaOneElectronBuilder, "_compile_and_register", fake_compile_and_register)
    monkeypatch.setattr("td_graddft.scf.cuda_one_electron._ffi_call", fake_ffi_call)

    builder = CudaOneElectronBuilder(basis, cache_dir=tmp_path)
    overlap, hcore = builder.build_overlap_hcore()

    assert captured["registered"] is True
    assert captured["target_name"] == "td_graddft_cuda_one_electron"
    assert len(captured["result_shape_dtypes"]) == 2
    assert captured["result_shape_dtypes"][0].shape == (basis.nao, basis.nao)
    assert len(captured["args"]) == 7
    assert np.allclose(np.asarray(overlap), np.eye(basis.nao))
    assert np.allclose(np.asarray(hcore), 2.0 * np.eye(basis.nao))


def test_cuda_one_electron_kernel_consumes_pre_normalized_coefficients():
    source = cuda_one_electron._kernel_source_path().read_text()
    contracted_start = source.index("__host__ __device__ void contracted_overlap_hcore")
    contracted_end = source.index("__global__ void one_electron_kernel")
    contracted_source = source[contracted_start:contracted_end]

    assert "primitive_cart_norm" not in contracted_source


def test_cuda_one_electron_builder_uses_prebuilt_library_env(monkeypatch, tmp_path):
    basis = basis_from_spec("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    library = tmp_path / "libtd_graddft_cuda_one_electron_prebuilt.so"
    library.write_bytes(b"fake")

    def fail_detect():
        raise AssertionError("prebuilt one-electron library should not probe CUDA arch")

    def fail_run(*args, **kwargs):
        raise AssertionError("prebuilt one-electron library should not invoke nvcc")

    monkeypatch.setenv("TD_GRADDFT_CUDA_ONEE_LIBRARY", str(library))
    monkeypatch.setattr(cuda_one_electron, "_detect_cuda_arch", fail_detect)
    monkeypatch.setattr(cuda_one_electron.subprocess, "run", fail_run)
    monkeypatch.setattr(CudaOneElectronBuilder, "_compile_and_register", lambda self: None)

    builder = CudaOneElectronBuilder(basis, cache_dir=tmp_path, nvcc=None)

    assert builder.library == library


def test_cuda_one_electron_builder_uses_any_package_prebuilt_without_arch_probe(
    monkeypatch,
    tmp_path,
):
    basis = basis_from_spec("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    library = tmp_path / "libtd_graddft_cuda_one_electron_packaged.so"
    library.write_bytes(b"fake")

    def fail_detect():
        raise AssertionError("package-local one-electron library should not probe CUDA arch")

    def fail_run(*args, **kwargs):
        raise AssertionError("package-local one-electron library should not invoke nvcc")

    monkeypatch.delenv("TD_GRADDFT_CUDA_ONEE_LIBRARY", raising=False)
    monkeypatch.setattr(cuda_one_electron, "_detect_cuda_arch", fail_detect)
    monkeypatch.setattr(cuda_one_electron, "_any_packaged_prebuilt_library", lambda: library)
    monkeypatch.setattr(cuda_one_electron.subprocess, "run", fail_run)
    monkeypatch.setattr(CudaOneElectronBuilder, "_compile_and_register", lambda self: None)

    builder = CudaOneElectronBuilder(basis, cache_dir=tmp_path, nvcc=None)

    assert builder.library == library


def test_build_prebuilt_cuda_one_electron_library_invokes_nvcc(monkeypatch, tmp_path):
    source = tmp_path / "cuda_one_electron_kernel.cu"
    source.write_text("// fake cuda source")
    commands = []

    class FakeFfi:
        @staticmethod
        def include_dir():
            return str(tmp_path / "include")

    def fake_run(command, check, capture_output, text):
        commands.append(command)
        output = command[-1]
        with open(output, "wb") as handle:
            handle.write(b"fake so")

    monkeypatch.setattr(cuda_one_electron, "_ffi_module", lambda: FakeFfi)
    monkeypatch.setattr(cuda_one_electron.subprocess, "run", fake_run)

    library = cuda_one_electron.build_prebuilt_cuda_one_electron_library(
        tmp_path / "pkg",
        source_path=source,
        nvcc="/usr/local/cuda/bin/nvcc",
        arch="sm_120",
    )

    assert library.exists()
    assert library.name.startswith("libtd_graddft_cuda_one_electron_")
    assert commands[0][0] == "/usr/local/cuda/bin/nvcc"
    assert "-arch=sm_120" in commands[0]
