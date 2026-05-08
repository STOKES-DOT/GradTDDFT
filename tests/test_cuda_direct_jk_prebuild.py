from __future__ import annotations

import pathlib

from td_graddft.scf import cuda_direct_jk


def test_build_prebuilt_cuda_direct_jk_library_invokes_nvcc(monkeypatch, tmp_path):
    source = tmp_path / "cuda_direct_jk_kernel.cu"
    source.write_text("__global__ void kernel() {}\n")
    captured = {}

    class FakeFfi:
        @staticmethod
        def include_dir():
            return "/fake/jax/include"

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        pathlib.Path(command[-1]).write_bytes(b"compiled")

    monkeypatch.setattr(cuda_direct_jk, "_ffi_module", lambda: FakeFfi)
    monkeypatch.setattr(cuda_direct_jk.subprocess, "run", fake_run)

    library = cuda_direct_jk.build_prebuilt_cuda_direct_jk_library(
        tmp_path / "pkg",
        source_path=source,
        nvcc="/usr/local/cuda/bin/nvcc",
        arch="sm_120",
        include_rys=False,
    )

    assert library.exists()
    assert library.parent == tmp_path / "pkg"
    assert library.name.startswith("libtd_graddft_cuda_direct_jk_")
    assert captured["command"][:6] == [
        "/usr/local/cuda/bin/nvcc",
        "-O3",
        "--std=c++17",
        "-shared",
        "-Xcompiler",
        "-fPIC",
    ]
    assert f"-arch=sm_120" in captured["command"]
    assert "-I" in captured["command"]
    assert "/fake/jax/include" in captured["command"]
    assert str(source) in captured["command"]
    assert captured["kwargs"]["check"] is True


def test_build_prebuilt_cuda_direct_jk_library_reuses_existing_library(monkeypatch, tmp_path):
    source = tmp_path / "cuda_direct_jk_kernel.cu"
    source.write_text("__global__ void kernel() {}\n")

    class FakeFfi:
        @staticmethod
        def include_dir():
            return "/fake/jax/include"

    monkeypatch.setattr(cuda_direct_jk, "_ffi_module", lambda: FakeFfi)
    existing = tmp_path / "pkg" / cuda_direct_jk._library_name_for_arch(
        "sm_120",
        source_path=source,
    )
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"cached")

    def fail_run(*args, **kwargs):
        raise AssertionError("existing prebuilt library should be reused")

    monkeypatch.setattr(cuda_direct_jk.subprocess, "run", fail_run)

    library = cuda_direct_jk.build_prebuilt_cuda_direct_jk_library(
        tmp_path / "pkg",
        source_path=source,
        nvcc="/usr/local/cuda/bin/nvcc",
        arch="sm_120",
        include_rys=False,
    )

    assert library == existing


def test_build_rys_library_invokes_fixed_nvcc_command(monkeypatch, tmp_path):
    captured = {}

    class Result:
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        output = pathlib.Path(command[command.index("-o") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"compiled")
        return Result()

    monkeypatch.setattr(cuda_direct_jk.subprocess, "run", fake_run)

    library = cuda_direct_jk.build_rys_library(
        tmp_path / "pkg",
        nvcc="/usr/local/cuda/bin/nvcc",
        arch="sm_120",
    )

    assert library.exists()
    assert library.name.startswith("libtd_graddft_rys_reference_")
    assert captured["command"][:7] == [
        "/usr/local/cuda/bin/nvcc",
        "-O3",
        "--std=c++17",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-rdc=true",
    ]
    assert f"-arch=sm_120" in captured["command"]
    assert "rys_contract_jk.cu" in " ".join(captured["command"])
    assert "unrolled_rys_jk.cu" in " ".join(captured["command"])
    assert captured["kwargs"]["check"] is True


def test_build_prebuilt_cuda_direct_jk_library_can_embed_rys(
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "cuda_direct_jk_kernel.cu"
    source.write_text("__global__ void kernel() {}\n")
    captured = {}

    class FakeFfi:
        @staticmethod
        def include_dir():
            return "/fake/jax/include"

    class Result:
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        captured["command"] = command
        output = pathlib.Path(command[command.index("-o") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"compiled")
        return Result()

    monkeypatch.setattr(cuda_direct_jk, "_ffi_module", lambda: FakeFfi)
    monkeypatch.setattr(cuda_direct_jk.subprocess, "run", fake_run)

    library = cuda_direct_jk.build_prebuilt_cuda_direct_jk_library(
        tmp_path / "pkg",
        source_path=source,
        nvcc="/usr/local/cuda/bin/nvcc",
        arch="sm_120",
        include_rys=True,
    )

    joined = " ".join(captured["command"])
    assert library.exists()
    assert "-rdc=true" in captured["command"]
    assert "-DTD_GRADDFT_ENABLE_RYS_DIRECT_JK=1" in captured["command"]
    assert "rys_contract_jk.cu" in joined
    assert "unrolled_rys_jk.cu" in joined


def test_build_prebuilt_cuda_direct_jk_library_accepts_fixed_joltqc_universe(
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "cuda_direct_jk_kernel.cu"
    source.write_text("__global__ void kernel() {}\n")
    commands = []

    class FakeFfi:
        @staticmethod
        def include_dir():
            return "/fake/jax/include"

    class Result:
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        commands.append(command)
        output = pathlib.Path(command[command.index("-o") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"compiled")
        return Result()

    monkeypatch.setattr(cuda_direct_jk, "_ffi_module", lambda: FakeFfi)
    monkeypatch.setattr(cuda_direct_jk.subprocess, "run", fake_run)

    library = cuda_direct_jk.build_prebuilt_cuda_direct_jk_library(
        tmp_path / "pkg",
        source_path=source,
        nvcc="/usr/local/cuda/bin/nvcc",
        arch="sm_120",
        joltqc_fixed_universe=True,
        joltqc_fixed_max_l=0,
        joltqc_fixed_nprim_max=1,
        include_rys=False,
    )

    compile_sources = [
        pathlib.Path(command[command.index("-c") + 1]).name
        for command in commands
        if "-c" in command
    ]
    assert library.exists()
    assert any(name.startswith("joltqc_1qnt_") for name in compile_sources)
    assert any("signature_l_0_0_0_0_p_1_1_1_1" in name for name in compile_sources)
