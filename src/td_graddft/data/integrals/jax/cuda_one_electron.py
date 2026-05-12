from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ...basis import CartesianBasis
from .cuda_direct_jk import (
    _detect_cuda_arch,
    _ffi_call,
    _ffi_module,
    extract_cuda_ao_system,
)


_FFI_TARGET_NAME = "td_graddft_cuda_one_electron"
_REGISTERED_FFI_LIBS: dict[Path, ctypes.CDLL] = {}
_REGISTERED_FFI_TARGETS: set[str] = set()
_PREBUILT_LIBRARY_ENV = "TD_GRADDFT_CUDA_ONEE_LIBRARY"


def _kernel_source_path() -> Path:
    source = Path(__file__).with_name("cuda_one_electron_kernel.cu")
    if source.exists():
        return source
    raise FileNotFoundError("Could not locate cuda_one_electron_kernel.cu.")


def _library_name_for_arch(
    arch: str,
    *,
    source_path: str | os.PathLike[str] | None = None,
) -> str:
    ffi = _ffi_module()
    source = Path(source_path) if source_path is not None else _kernel_source_path()
    digest = hashlib.sha256()
    digest.update(source.read_bytes())
    digest.update(str(arch).encode())
    digest.update(str(ffi.include_dir()).encode())
    return f"libtd_graddft_cuda_one_electron_{digest.hexdigest()[:16]}.so"


def _packaged_prebuilt_library_path(arch: str) -> Path | None:
    path = Path(__file__).with_name(_library_name_for_arch(str(arch)))
    return path if path.exists() else None


def _any_packaged_prebuilt_library() -> Path | None:
    candidates = sorted(Path(__file__).parent.glob("libtd_graddft_cuda_one_electron_*.so"))
    return candidates[0] if candidates else None


def build_prebuilt_cuda_one_electron_library(
    output_dir: str | os.PathLike[str],
    *,
    source_path: str | os.PathLike[str] | None = None,
    nvcc: str | None = None,
    arch: str = "native",
    force: bool = False,
) -> Path:
    """Build the CUDA one-electron FFI shared library outside SCF runtime."""

    compiler = nvcc or os.environ.get("TD_GRADDFT_NVCC") or shutil.which("nvcc")
    if compiler is None:
        raise RuntimeError(
            "CUDA one-electron prebuild requires nvcc. Set TD_GRADDFT_NVCC or put nvcc on PATH."
        )
    arch_name = _detect_cuda_arch() if str(arch) == "native" else str(arch)
    source = Path(source_path) if source_path is not None else _kernel_source_path()
    ffi = _ffi_module()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    library = out_dir / _library_name_for_arch(arch_name, source_path=source)
    if library.exists() and not bool(force):
        return library
    command = [
        compiler,
        "-O3",
        "--std=c++17",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        f"-arch={arch_name}",
        "-I",
        ffi.include_dir(),
        str(source),
        "-o",
        str(library),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to prebuild CUDA one-electron FFI library with command:\n"
            + " ".join(command)
            + "\nstdout:\n"
            + (exc.stdout or "")
            + "\nstderr:\n"
            + (exc.stderr or "")
        ) from exc
    return library


@dataclass(frozen=True)
class CudaOneElectronBuilder:
    """JAX FFI CUDA builder for overlap and one-electron core matrices."""

    basis: CartesianBasis
    cache_dir: str | os.PathLike[str] | None = None
    nvcc: str | None = None
    arch: str = "native"
    max_l: int = 2

    def __post_init__(self) -> None:
        system = extract_cuda_ao_system(self.basis, max_l=self.max_l)
        cache_dir = Path(
            self.cache_dir
            or os.environ.get("TD_GRADDFT_CUDA_ONEE_CACHE", "")
            or (Path(tempfile.gettempdir()) / "td_graddft_cuda_one_electron")
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        nvcc = self.nvcc or os.environ.get("TD_GRADDFT_NVCC") or shutil.which("nvcc")
        arch = None if str(self.arch) == "native" else str(self.arch)
        object.__setattr__(self, "system", system)
        object.__setattr__(self, "cache_path", cache_dir)
        object.__setattr__(self, "nvcc_path", nvcc)
        object.__setattr__(self, "cuda_arch", arch)
        object.__setattr__(self, "library", self._compile_library())
        self._compile_and_register()
        object.__setattr__(self, "centers", jnp.asarray(system.centers, dtype=jnp.float64))
        object.__setattr__(self, "angulars", jnp.asarray(system.angulars, dtype=jnp.int32))
        object.__setattr__(self, "exponents", jnp.asarray(system.exponents, dtype=jnp.float64))
        object.__setattr__(self, "coefficients", jnp.asarray(system.coefficients, dtype=jnp.float64))
        object.__setattr__(self, "nprims", jnp.asarray(system.nprims, dtype=jnp.int32))
        object.__setattr__(
            self,
            "atom_coords",
            jnp.asarray(self.basis.atom_coords, dtype=jnp.float64),
        )
        object.__setattr__(
            self,
            "atom_charges",
            jnp.asarray(self.basis.atom_charges, dtype=jnp.float64),
        )

    def _compile_library(self) -> Path:
        prebuilt = os.environ.get(_PREBUILT_LIBRARY_ENV)
        if prebuilt:
            library = Path(prebuilt).expanduser()
            if not library.exists():
                raise FileNotFoundError(
                    f"{_PREBUILT_LIBRARY_ENV} points to a missing CUDA FFI library: {library}"
                )
            return library
        if self.cuda_arch is not None:
            packaged = _packaged_prebuilt_library_path(self.cuda_arch)
            if packaged is not None:
                return packaged
        any_packaged = _any_packaged_prebuilt_library()
        if any_packaged is not None:
            return any_packaged
        if self.nvcc_path is None:
            raise RuntimeError(
                "CUDA one-electron integrals require nvcc. Set TD_GRADDFT_NVCC or put nvcc on PATH."
            )
        arch = _detect_cuda_arch() if self.cuda_arch is None else self.cuda_arch
        return build_prebuilt_cuda_one_electron_library(
            self.cache_path,
            nvcc=self.nvcc_path,
            arch=arch,
        )

    def _compile_and_register(self) -> None:
        library = self.library.resolve()
        if _FFI_TARGET_NAME in _REGISTERED_FFI_TARGETS:
            return
        lib = ctypes.CDLL(str(library))
        ffi = _ffi_module()
        ffi.register_ffi_target(
            _FFI_TARGET_NAME,
            ffi.pycapsule(getattr(lib, "TdGraddftCudaOneElectronFfi")),
            platform="CUDA",
            api_version=1,
        )
        _REGISTERED_FFI_LIBS[library] = lib
        _REGISTERED_FFI_TARGETS.add(_FFI_TARGET_NAME)

    def build_overlap_hcore(self) -> tuple[Array, Array]:
        if self.basis.atom_coords is None or self.basis.atom_charges is None:
            raise ValueError("CUDA one-electron integrals require atom coordinates and charges.")
        shape = jax.ShapeDtypeStruct((self.system.nao, self.system.nao), jnp.float64)
        overlap, hcore = _ffi_call(
            _FFI_TARGET_NAME,
            (shape, shape),
            self.centers,
            self.angulars,
            self.exponents,
            self.coefficients,
            self.nprims,
            self.atom_coords,
            self.atom_charges,
        )
        return jnp.asarray(overlap, dtype=jnp.float64), jnp.asarray(hcore, dtype=jnp.float64)
