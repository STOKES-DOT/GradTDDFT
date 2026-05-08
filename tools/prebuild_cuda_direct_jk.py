from __future__ import annotations

import argparse
from pathlib import Path

from td_graddft.scf import cuda_direct_jk


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prebuild the TD-GradDFT CUDA direct J/K FFI library outside SCF timing."
    )
    parser.add_argument("--cache-dir", default=None, help="Directory for the compiled shared library.")
    parser.add_argument(
        "--package-local",
        action="store_true",
        help="Write the shared library next to td_graddft.scf.cuda_direct_jk for automatic loading.",
    )
    parser.add_argument("--nvcc", default=None, help="Path to nvcc. Defaults to TD_GRADDFT_NVCC or PATH.")
    parser.add_argument("--arch", default="native", help="CUDA architecture, e.g. sm_120.")
    parser.add_argument(
        "--joltqc-fixed-universe",
        action="store_true",
        help=(
            "Compile a molecule-independent JoltQC signature universe so new molecules "
            "reuse this library instead of invoking nvcc during SCF."
        ),
    )
    parser.add_argument("--joltqc-max-l", type=int, default=2, help="Maximum shell angular momentum.")
    parser.add_argument(
        "--joltqc-nprim-max",
        type=int,
        default=3,
        help="Maximum split primitive count per JoltQC basis row.",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir
    if args.package_local:
        cache_dir = str(Path(cuda_direct_jk.__file__).parent)
    library = cuda_direct_jk.build_prebuilt_cuda_direct_jk_library(
        cache_dir or Path.cwd(),
        nvcc=args.nvcc,
        arch=args.arch,
        joltqc_fixed_universe=args.joltqc_fixed_universe,
        joltqc_fixed_max_l=args.joltqc_max_l,
        joltqc_fixed_nprim_max=args.joltqc_nprim_max,
    )
    library = Path(library).resolve()
    print(str(library))
    if args.package_local:
        print("packaged_prebuilt_library=1")
    print(f"export TD_GRADDFT_CUDA_JK_LIBRARY={library}")


if __name__ == "__main__":
    main()
