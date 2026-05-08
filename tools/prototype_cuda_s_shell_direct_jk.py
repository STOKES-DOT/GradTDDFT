from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from td_graddft.data.basis import basis_from_spec
from td_graddft.data.integrals import eri_pair_matrix_packed
from td_graddft.scf.packed_eri import build_jk_from_eri_pair_matrix
from td_graddft_tools.gpu_s_shell_direct_jk import (
    SPAOSystem,
    cpu_sp_direct_jk,
    extract_cartesian_ao_system,
    extract_sp_ao_system,
)


def pyscf_jk_reference(atom: str, basis_name: str, density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a libcint/PySCF cartesian ERI reference for prototype validation."""

    from pyscf import gto

    mol = gto.M(
        atom=atom,
        basis=basis_name,
        unit="Angstrom",
        cart=True,
        verbose=0,
    )
    density_arr = np.asarray(density, dtype=np.float64)
    density_arr = 0.5 * (density_arr + density_arr.T)
    eri = np.asarray(mol.intor("int2e"), dtype=np.float64)
    j_mat = np.einsum("pqrs,rs->pq", eri, density_arr, optimize=True)
    k_mat = np.einsum("pqrs,qs->pr", eri, density_arr, optimize=True)
    return 0.5 * (j_mat + j_mat.T), 0.5 * (k_mat + k_mat.T)


def write_cuda_input(path: Path, system: SPAOSystem, density: np.ndarray) -> None:
    density_arr = np.asarray(density, dtype=np.float64)
    density_arr = 0.5 * (density_arr + density_arr.T)
    chunks: list[str] = [f"{system.nao} {system.max_nprim}"]
    chunks.append(" ".join(str(int(value)) for value in system.nprims.reshape(-1)))
    chunks.append(" ".join(str(int(value)) for value in system.angulars.reshape(-1)))
    for array in (
        system.centers,
        system.exponents,
        system.coefficients,
        density_arr,
    ):
        chunks.append(" ".join(f"{float(value):.17g}" for value in array.reshape(-1)))
    path.write_text("\n".join(chunks) + "\n")


def parse_cuda_output(path: Path, nao: int) -> tuple[float, np.ndarray, np.ndarray]:
    tokens = path.read_text().split()
    if len(tokens) != 4 + 2 * nao * nao:
        raise ValueError(f"Unexpected CUDA output length in {path}.")
    if tokens[0] != "kernel_avg_ms" or tokens[2] != "J":
        raise ValueError(f"Unexpected CUDA output header in {path}.")
    kernel_avg_ms = float(tokens[1])
    offset = 3
    j_values = np.asarray([float(value) for value in tokens[offset : offset + nao * nao]])
    offset += nao * nao
    if tokens[offset] != "K":
        raise ValueError(f"Unexpected CUDA K marker in {path}.")
    offset += 1
    k_values = np.asarray([float(value) for value in tokens[offset : offset + nao * nao]])
    j_mat = j_values.reshape(nao, nao)
    k_mat = k_values.reshape(nao, nao)
    return kernel_avg_ms, 0.5 * (j_mat + j_mat.T), 0.5 * (k_mat + k_mat.T)


def compile_cuda(source: Path, binary: Path, nvcc: str, *, arch: str | None) -> float:
    command = [
        nvcc,
        "-O3",
        "--std=c++17",
    ]
    if arch:
        command.append(f"-arch={arch}")
    command.extend([str(source), "-o", str(binary)])
    start = time.perf_counter()
    subprocess.run(command, check=True)
    return time.perf_counter() - start


def run_cuda(binary: Path, input_path: Path, output_path: Path, repeats: int) -> float:
    start = time.perf_counter()
    subprocess.run(
        [str(binary), str(input_path), str(int(repeats)), str(output_path)],
        check=True,
    )
    return time.perf_counter() - start


def build_case(name: str) -> tuple[str, SPAOSystem, np.ndarray, np.ndarray, np.ndarray]:
    case = str(name).lower()
    if case == "h2":
        atom = "H 0 0 0; H 0 0 0.74"
        basis_name = "sto-3g"
        max_l = 1
        label = "H2/STO-3G s/p AO direct J/K"
        density = np.asarray(
            [
                [0.83, 0.21],
                [0.21, 0.71],
            ],
            dtype=np.float64,
        )
    elif case == "water":
        atom = "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"
        basis_name = "sto-3g"
        max_l = 1
        label = "water/STO-3G s/p AO direct J/K"
        density = np.asarray(
            [
                [0.84, 0.04, -0.03, 0.02, 0.01, 0.06, 0.05],
                [0.04, 0.71, 0.08, -0.02, 0.03, 0.04, -0.01],
                [-0.03, 0.08, 0.66, 0.05, -0.04, 0.03, 0.02],
                [0.02, -0.02, 0.05, 0.59, 0.06, -0.02, 0.01],
                [0.01, 0.03, -0.04, 0.06, 0.62, 0.02, -0.03],
                [0.06, 0.04, 0.03, -0.02, 0.02, 0.48, 0.07],
                [0.05, -0.01, 0.02, 0.01, -0.03, 0.07, 0.51],
            ],
            dtype=np.float64,
        )
    elif case == "water631gstar":
        atom = "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"
        basis_name = "6-31g*"
        max_l = 2
        label = "water/6-31G* s/p/d AO direct J/K"
        density = None
    else:
        raise ValueError("--case must be 'h2', 'water', or 'water631gstar'.")

    basis = basis_from_spec(atom, basis=basis_name, max_l=max_l)
    system = (
        extract_sp_ao_system(basis)
        if max_l <= 1
        else extract_cartesian_ao_system(basis, max_l=max_l)
    )
    if density is None:
        idx = np.arange(system.nao, dtype=np.float64)
        density = 0.02 / (1.0 + np.abs(idx[:, None] - idx[None, :]))
        density += np.diag(0.45 + 0.01 * idx)
    if max_l <= 1:
        j_ref, k_ref = build_jk_from_eri_pair_matrix(eri_pair_matrix_packed(basis), density)
        j_cpu, k_cpu = cpu_sp_direct_jk(system, density)
        if not np.allclose(j_cpu, np.asarray(j_ref), atol=2e-10, rtol=2e-10):
            raise RuntimeError("CPU s/p J does not match packed ERI reference.")
        if not np.allclose(k_cpu, np.asarray(k_ref), atol=2e-10, rtol=2e-10):
            raise RuntimeError("CPU s/p K does not match packed ERI reference.")
    else:
        j_ref, k_ref = pyscf_jk_reference(atom, basis_name, density)
    return label, system, density, np.asarray(j_ref), np.asarray(k_ref)


def build_h2_case() -> tuple[SPAOSystem, np.ndarray, np.ndarray, np.ndarray]:
    _label, system, density, j_ref, k_ref = build_case("h2")
    return system, density, j_ref, k_ref


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cuda_s_shell_direct_jk_h2"))
    parser.add_argument("--source", type=Path, default=Path("tools/cuda_s_shell_direct_jk.cu"))
    parser.add_argument("--case", choices=("h2", "water", "water631gstar"), default="h2")
    parser.add_argument("--nvcc", default=None)
    parser.add_argument("--arch", default="native")
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()

    nvcc = args.nvcc or shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc was not found. Run this prototype in a CUDA environment.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / f"{args.case}_sto3g_sp_input.txt"
    output_path = output_dir / f"{args.case}_sto3g_sp_cuda_output.txt"
    binary = output_dir / "cuda_s_shell_direct_jk"

    label, system, density, j_ref, k_ref = build_case(args.case)
    write_cuda_input(input_path, system, density)
    compile_s = compile_cuda(args.source, binary, nvcc, arch=args.arch)
    cuda_wall_s = run_cuda(binary, input_path, output_path, args.repeats)
    kernel_avg_ms, j_cuda, k_cuda = parse_cuda_output(output_path, system.nao)

    max_abs_j = float(np.max(np.abs(j_cuda - j_ref)))
    max_abs_k = float(np.max(np.abs(k_cuda - k_ref)))
    summary = {
        "case": label,
        "nao": system.nao,
        "max_nprim": system.max_nprim,
        "repeats": int(args.repeats),
        "compile_s": float(compile_s),
        "cuda_wall_s": float(cuda_wall_s),
        "kernel_avg_ms": float(kernel_avg_ms),
        "max_abs_j": max_abs_j,
        "max_abs_k": max_abs_k,
        "passed": bool(max(max_abs_j, max_abs_k) <= float(args.tolerance)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
