from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


MOLECULES: dict[str, str] = {
    "water": """
O  0.000000  0.000000  0.117790
H  0.000000  0.755453 -0.471161
H  0.000000 -0.755453 -0.471161
""",
    "benzene": """
C        0.0000000000      1.3967920000      0.0000000000
C       -1.2096570000      0.6983960000      0.0000000000
C       -1.2096570000     -0.6983960000      0.0000000000
C        0.0000000000     -1.3967920000      0.0000000000
C        1.2096570000     -0.6983960000      0.0000000000
C        1.2096570000      0.6983960000      0.0000000000
H        0.0000000000      2.4842120000      0.0000000000
H       -2.1513900000      1.2421060000      0.0000000000
H       -2.1513900000     -1.2421060000      0.0000000000
H        0.0000000000     -2.4842120000      0.0000000000
H        2.1513900000     -1.2421060000      0.0000000000
H        2.1513900000      1.2421060000      0.0000000000
""",
}


@dataclass(frozen=True)
class GPUStats:
    samples: int
    mean_util_pct: float
    max_util_pct: float
    mean_mem_mib: float
    max_mem_mib: float
    start_mem_mib: float
    delta_peak_mem_mib: float


@dataclass(frozen=True)
class StrictJaxFullFlowRow:
    label: str
    repeat_index: int
    cold_start: bool
    platform: str
    device: str
    basis: str
    xc: str
    grids_level: int
    integral_backend: str
    jk_backend: str
    direct_scf_tol: float
    nstates: int
    nao: int
    nocc: int
    nvir: int
    td_dim: int
    jax_reference_s: float
    jax_tddft_kernel_s: float
    jax_oscillator_s: float
    jax_total_s: float
    jax_total_energy_ha: float
    jax_exc1_ev: float
    jax_converged: bool | None
    jax_cycles: int | None
    pyscf_scf_s: float | None
    pyscf_casida_s: float | None
    pyscf_total_s: float | None
    pyscf_total_energy_ha: float | None
    pyscf_exc1_ev: float | None
    total_energy_diff_ha: float | None
    exc1_diff_ev: float | None
    casida_mae_ev: float | None
    gpu_samples: int
    gpu_mean_util_pct: float
    gpu_max_util_pct: float
    gpu_mean_mem_mib: float
    gpu_max_mem_mib: float
    gpu_delta_peak_mem_mib: float
    note: str


class NvidiaSmiSampler:
    def __init__(self, gpu_index: int | None, interval_s: float):
        self.gpu_index = gpu_index
        self.interval_s = float(interval_s)
        self.samples: list[tuple[float, float, float]] = []
        self._running = False
        self._thread: threading.Thread | None = None

    def _query(self) -> tuple[float, float] | None:
        if self.gpu_index is None:
            return None
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    str(int(self.gpu_index)),
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=3.0,
            ).strip()
        except Exception:
            return None
        if not out:
            return None
        parts = [part.strip() for part in out.splitlines()[0].split(",")]
        if len(parts) < 2:
            return None
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            return None

    def start(self) -> None:
        if self.gpu_index is None:
            return
        first = self._query()
        if first is not None:
            util, mem = first
            self.samples.append((time.perf_counter(), util, mem))
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            sample = self._query()
            if sample is not None:
                util, mem = sample
                self.samples.append((time.perf_counter(), util, mem))
            time.sleep(self.interval_s)

    def stop(self) -> GPUStats:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if not self.samples:
            return GPUStats(
                samples=0,
                mean_util_pct=float("nan"),
                max_util_pct=float("nan"),
                mean_mem_mib=float("nan"),
                max_mem_mib=float("nan"),
                start_mem_mib=float("nan"),
                delta_peak_mem_mib=float("nan"),
            )
        util = np.asarray([sample[1] for sample in self.samples], dtype=float)
        mem = np.asarray([sample[2] for sample in self.samples], dtype=float)
        return GPUStats(
            samples=len(self.samples),
            mean_util_pct=float(np.mean(util)),
            max_util_pct=float(np.max(util)),
            mean_mem_mib=float(np.mean(mem)),
            max_mem_mib=float(np.max(mem)),
            start_mem_mib=float(mem[0]),
            delta_peak_mem_mib=float(np.max(mem) - mem[0]),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the TD-GradDFT strict-JAX full flow: molecule spec -> no-DF SCF "
            "-> A/B response matrices -> Casida solve. PySCF is optional reference only."
        )
    )
    parser.add_argument("--molecule", choices=tuple(MOLECULES.keys()), default="benzene")
    parser.add_argument("--label", default=None)
    parser.add_argument("--basis", default="6-31g*")
    parser.add_argument("--xc", default="pbe0")
    parser.add_argument("--nstates", type=int, default=5)
    parser.add_argument("--grids-level", type=int, default=0)
    parser.add_argument("--max-l", type=int, default=3)
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--integral-backend", choices=("jax", "cpu", "gpu", "libcint"), default="jax")
    parser.add_argument("--jk-backend", choices=("direct", "full", "df"), default="direct")
    parser.add_argument("--direct-scf-tol", type=float, default=1e-12)
    parser.add_argument("--scf-max-cycle", type=int, default=80)
    parser.add_argument("--conv-tol", type=float, default=1e-9)
    parser.add_argument("--conv-tol-density", type=float, default=1e-7)
    parser.add_argument("--damping", type=float, default=0.05)
    parser.add_argument("--precompile-eri", action="store_true")
    parser.add_argument("--precompile-eri-chunk-size", type=int, default=512)
    parser.add_argument("--eigensolver", choices=("auto", "davidson"), default="auto")
    parser.add_argument("--include-pyscf", action="store_true")
    parser.add_argument("--skip-pyscf-td", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--gpu-index", type=int, default=None)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5)
    parser.add_argument("--outdir", default="outputs/strict_jax_fullflow")
    return parser.parse_args()


def _set_platform_environment(platform: str) -> None:
    if platform == "cpu":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif platform == "gpu":
        os.environ["JAX_PLATFORM_NAME"] = "gpu"
        os.environ["JAX_PLATFORMS"] = "cuda"


def _timer() -> float:
    return time.perf_counter()


def _block_tree(value: Any) -> None:
    import jax

    leaves = jax.tree_util.tree_leaves(value)
    for leaf in leaves:
        try:
            jax.block_until_ready(leaf)
        except TypeError:
            continue


def _block_reference(reference: Any) -> None:
    for name in (
        "ao",
        "ao_deriv1",
        "ao_laplacian",
        "rep_tensor",
        "mo_coeff",
        "mo_occ",
        "mo_energy",
        "rdm1",
        "h1e",
        "overlap_matrix",
        "dipole_integrals",
        "df_factors",
        "eri_ovov",
        "eri_ovvo",
        "eri_oovv",
    ):
        value = getattr(reference, name, None)
        if value is not None:
            _block_tree(value)


def _jax_scalar_to_float(value: Any) -> float:
    _block_tree(value)
    return float(np.asarray(value))


def _pick_device(platform: str):
    import jax

    devices = jax.devices(platform)
    if not devices:
        raise RuntimeError(f"No JAX {platform!r} device is visible.")
    return devices[0]


def _build_strict_jax_reference(args: argparse.Namespace, *, device: Any):
    import jax

    from td_graddft.scf.builders import restricted_molecule_from_spec_with_jax_rks
    from td_graddft.scf import RKSConfig

    cfg = RKSConfig(
        xc_spec=str(args.xc),
        max_cycle=int(args.scf_max_cycle),
        conv_tol=float(args.conv_tol),
        conv_tol_density=float(args.conv_tol_density),
        damping=float(args.damping),
        density_floor=1e-12,
        potential_clip=20.0,
        jk_backend=str(args.jk_backend),
        direct_scf_tol=float(args.direct_scf_tol),
    )
    with jax.default_device(device):
        return restricted_molecule_from_spec_with_jax_rks(
            atom=MOLECULES[str(args.molecule)],
            basis=str(args.basis),
            xc_spec=str(args.xc),
            unit="Angstrom",
            charge=0,
            spin=0,
            cart=True,
            grids_level=int(args.grids_level),
            max_l=int(args.max_l),
            rks_config=cfg,
            grid_ao_backend="jax",
            integral_backend=str(args.integral_backend),
            precompile_eri=bool(args.precompile_eri),
            precompile_eri_chunk_size=int(args.precompile_eri_chunk_size),
        )


def _run_strict_jax_full_flow(args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from td_graddft.spectra import HARTREE_TO_EV, oscillator_strengths
    from td_graddft import tdscf
    from td_graddft.tddft._semilocal_response import SemilocalResponseFunctional

    jax.config.update("jax_enable_x64", True)
    device = _pick_device(str(args.platform))
    sampler = NvidiaSmiSampler(
        gpu_index=args.gpu_index if str(args.platform) == "gpu" else None,
        interval_s=float(args.gpu_sample_interval),
    )
    sampler.start()
    try:
        t0 = _timer()
        reference = _build_strict_jax_reference(args, device=device)
        _block_reference(reference)
        reference_s = _timer() - t0

        nocc = int(getattr(reference, "nocc", 0) or 0)
        mo_energy = np.asarray(reference.mo_energy[0], dtype=float)
        nao = int(mo_energy.shape[0])
        nvir = int(nao - nocc)
        nstates_eff = min(max(1, int(args.nstates)), max(1, nocc * nvir))
        functional = SemilocalResponseFunctional(str(args.xc))

        t1 = _timer()
        with jax.default_device(device):
            result = tdscf.TDDFT(
                reference,
                xc_functional=functional,
                eigensolver=str(args.eigensolver),
            ).kernel(
                nstates=nstates_eff,
            )
        _block_tree(result)
        tddft_kernel_s = _timer() - t1

        t3 = _timer()
        strengths = oscillator_strengths(reference, result)
        _block_tree(strengths)
        oscillator_s = _timer() - t3

        energies_ha = np.asarray(result.excitation_energies, dtype=float).reshape(-1)
        strengths_np = np.asarray(strengths, dtype=float).reshape(-1)
        del strengths_np
        total_s = reference_s + tddft_kernel_s + oscillator_s
        stats = sampler.stop()
        scf_result = getattr(reference, "scf_result", None)
        return {
            "device": str(device),
            "nao": nao,
            "nocc": nocc,
            "nvir": nvir,
            "td_dim": int(nocc * nvir),
            "nstates": int(nstates_eff),
            "reference_s": float(reference_s),
            "tddft_kernel_s": float(tddft_kernel_s),
            "oscillator_s": float(oscillator_s),
            "total_s": float(total_s),
            "total_energy_ha": _jax_scalar_to_float(reference.mf_energy),
            "excitation_energies_ha": energies_ha.tolist(),
            "exc1_ev": (
                float(energies_ha[0] * HARTREE_TO_EV) if energies_ha.size > 0 else float("nan")
            ),
            "converged": getattr(scf_result, "converged", None),
            "cycles": getattr(scf_result, "cycles", None),
            "gpu_stats": asdict(stats),
        }
    finally:
        if sampler._running:
            sampler.stop()


def _run_pyscf_reference(args: argparse.Namespace, *, nstates: int) -> dict[str, Any]:
    from pyscf import dft, gto

    from td_graddft.spectra import HARTREE_TO_EV

    mol = gto.M(
        atom=MOLECULES[str(args.molecule)],
        basis=str(args.basis),
        unit="Angstrom",
        charge=0,
        spin=0,
        cart=True,
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = str(args.xc)
    mf.grids.level = int(args.grids_level)
    mf.conv_tol = float(args.conv_tol)
    mf.max_cycle = max(120, int(args.scf_max_cycle))
    mf.direct_scf = True
    mf.direct_scf_tol = float(args.direct_scf_tol)

    t0 = _timer()
    mf.kernel()
    scf_s = _timer() - t0
    if not mf.converged:
        raise RuntimeError("PySCF reference SCF did not converge.")

    out: dict[str, Any] = {
        "scf_s": float(scf_s),
        "casida_s": None,
        "total_s": None,
        "total_energy_ha": float(mf.e_tot),
        "excitation_energies_ha": [],
        "exc1_ev": None,
    }
    if bool(args.skip_pyscf_td):
        return out

    td = mf.TDDFT()
    td.nstates = int(nstates)
    t1 = _timer()
    td.kernel()
    casida_s = _timer() - t1
    energies = np.asarray(td.e, dtype=float).reshape(-1)
    out.update(
        {
            "casida_s": float(casida_s),
            "total_s": float(scf_s + casida_s),
            "excitation_energies_ha": energies.tolist(),
            "exc1_ev": float(energies[0] * HARTREE_TO_EV) if energies.size else None,
        }
    )
    return out


def _comparison_metrics(jax_result: dict[str, Any], pyscf_result: dict[str, Any] | None) -> dict[str, float | None]:
    if pyscf_result is None:
        return {
            "total_energy_diff_ha": None,
            "exc1_diff_ev": None,
            "casida_mae_ev": None,
        }
    from td_graddft.spectra import HARTREE_TO_EV

    energy_diff = float(jax_result["total_energy_ha"] - pyscf_result["total_energy_ha"])
    jax_e = np.asarray(jax_result.get("excitation_energies_ha", []), dtype=float)
    pyscf_e = np.asarray(pyscf_result.get("excitation_energies_ha", []), dtype=float)
    if jax_e.size and pyscf_e.size:
        n = min(jax_e.size, pyscf_e.size)
        diffs_ev = np.abs((jax_e[:n] - pyscf_e[:n]) * HARTREE_TO_EV)
        exc1_diff = float(diffs_ev[0])
        mae = float(np.mean(diffs_ev))
    else:
        exc1_diff = None
        mae = None
    return {
        "total_energy_diff_ha": energy_diff,
        "exc1_diff_ev": exc1_diff,
        "casida_mae_ev": mae,
    }


def _make_row(
    args: argparse.Namespace,
    *,
    repeat_index: int,
    strict_result: dict[str, Any],
    pyscf_result: dict[str, Any] | None,
    metrics: dict[str, float | None],
) -> StrictJaxFullFlowRow:
    gpu = strict_result["gpu_stats"]
    return StrictJaxFullFlowRow(
        label=str(args.label or args.molecule),
        repeat_index=int(repeat_index),
        cold_start=int(repeat_index) == 0,
        platform=str(args.platform),
        device=str(strict_result["device"]),
        basis=str(args.basis),
        xc=str(args.xc),
        grids_level=int(args.grids_level),
        integral_backend=str(args.integral_backend),
        jk_backend=str(args.jk_backend),
        direct_scf_tol=float(args.direct_scf_tol),
        nstates=int(strict_result["nstates"]),
        nao=int(strict_result["nao"]),
        nocc=int(strict_result["nocc"]),
        nvir=int(strict_result["nvir"]),
        td_dim=int(strict_result["td_dim"]),
        jax_reference_s=float(strict_result["reference_s"]),
        jax_tddft_kernel_s=float(strict_result["tddft_kernel_s"]),
        jax_oscillator_s=float(strict_result["oscillator_s"]),
        jax_total_s=float(strict_result["total_s"]),
        jax_total_energy_ha=float(strict_result["total_energy_ha"]),
        jax_exc1_ev=float(strict_result["exc1_ev"]),
        jax_converged=strict_result.get("converged"),
        jax_cycles=strict_result.get("cycles"),
        pyscf_scf_s=None if pyscf_result is None else pyscf_result["scf_s"],
        pyscf_casida_s=None if pyscf_result is None else pyscf_result["casida_s"],
        pyscf_total_s=None if pyscf_result is None else pyscf_result["total_s"],
        pyscf_total_energy_ha=None if pyscf_result is None else pyscf_result["total_energy_ha"],
        pyscf_exc1_ev=None if pyscf_result is None else pyscf_result["exc1_ev"],
        total_energy_diff_ha=metrics["total_energy_diff_ha"],
        exc1_diff_ev=metrics["exc1_diff_ev"],
        casida_mae_ev=metrics["casida_mae_ev"],
        gpu_samples=int(gpu["samples"]),
        gpu_mean_util_pct=float(gpu["mean_util_pct"]),
        gpu_max_util_pct=float(gpu["max_util_pct"]),
        gpu_mean_mem_mib=float(gpu["mean_mem_mib"]),
        gpu_max_mem_mib=float(gpu["max_mem_mib"]),
        gpu_delta_peak_mem_mib=float(gpu["delta_peak_mem_mib"]),
        note="strict_jax_path_excludes_pyscf",
    )


def _write_outputs(outdir: Path, row: StrictJaxFullFlowRow, summary: dict[str, Any]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{row.label}_r{row.repeat_index}_{row.platform}_{row.xc}_{row.basis}_"
        f"{row.integral_backend}_{row.jk_backend}"
    )
    stem = stem.replace("*", "star").replace("/", "_").replace(" ", "_").lower()
    summary_path = outdir / f"{stem}_summary.json"
    row_path = outdir / f"{stem}_row.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with row_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(row).keys()))
        writer.writeheader()
        writer.writerow(asdict(row))


def main() -> None:
    args = _parse_args()
    _set_platform_environment(str(args.platform))

    outdir = Path(args.outdir)
    repeats = max(1, int(args.repeats))
    run_summaries: list[dict[str, Any]] = []
    rows: list[StrictJaxFullFlowRow] = []
    pyscf_result: dict[str, Any] | None = None

    for repeat_index in range(repeats):
        strict_result = _run_strict_jax_full_flow(args)
        if bool(args.include_pyscf) and repeat_index == 0:
            pyscf_result = _run_pyscf_reference(args, nstates=int(strict_result["nstates"]))
        metrics = _comparison_metrics(strict_result, pyscf_result)
        row = _make_row(
            args,
            repeat_index=repeat_index,
            strict_result=strict_result,
            pyscf_result=pyscf_result,
            metrics=metrics,
        )
        summary = {
            "row": asdict(row),
            "strict_jax": strict_result,
            "pyscf_reference": pyscf_result,
            "comparison": metrics,
        }
        _write_outputs(outdir, row, summary)
        run_summaries.append(summary)
        rows.append(row)

    aggregate = {
        "rows": [asdict(row) for row in rows],
        "runs": run_summaries,
    }
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
