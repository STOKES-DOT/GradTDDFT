from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    """Build optional CUDA FFI package data before wheel packaging."""

    def run(self):
        super().run()
        if os.environ.get("TD_GRADDFT_BUILD_CUDA_FFI", "").lower() not in {"1", "true", "yes", "on"}:
            return
        root = Path(__file__).resolve().parent
        src_dir = root / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        from td_graddft.scf.cuda_direct_jk import build_prebuilt_cuda_direct_jk_library
        from td_graddft.scf.cuda_one_electron import build_prebuilt_cuda_one_electron_library

        output_dir = Path(self.build_lib) / "td_graddft" / "scf"
        force = os.environ.get("TD_GRADDFT_FORCE_CUDA_FFI_BUILD", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        arch = os.environ.get("TD_GRADDFT_CUDA_ARCH", "native")
        nvcc = os.environ.get("TD_GRADDFT_NVCC")
        joltqc_fixed_universe = os.environ.get(
            "TD_GRADDFT_CUDA_JOLTQC_FIXED_UNIVERSE",
            "",
        ).lower() in {"1", "true", "yes", "on"}
        fixed_max_l_raw = os.environ.get("TD_GRADDFT_CUDA_JOLTQC_FIXED_MAX_L", "").strip()
        fixed_nprim_raw = os.environ.get("TD_GRADDFT_CUDA_JOLTQC_FIXED_NPRIM_MAX", "").strip()
        build_prebuilt_cuda_direct_jk_library(
            output_dir,
            nvcc=nvcc,
            arch=arch,
            force=force,
            joltqc_fixed_universe=joltqc_fixed_universe,
            joltqc_fixed_max_l=int(fixed_max_l_raw) if fixed_max_l_raw else None,
            joltqc_fixed_nprim_max=int(fixed_nprim_raw) if fixed_nprim_raw else None,
        )
        build_prebuilt_cuda_one_electron_library(output_dir, nvcc=nvcc, arch=arch, force=force)


setup(cmdclass={"build_py": build_py})
