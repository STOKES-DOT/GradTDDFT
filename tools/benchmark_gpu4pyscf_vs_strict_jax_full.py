from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from pyscf import dft, gto


MOLECULES = {
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark GPU4PySCF against strict-JAX full SCF/TDA/Casida on the same molecule."
    )
    p.add_argument("--molecule", choices=tuple(MOLECULES.keys()), default="water")
    p.add_argument("--basis", default="6-31g")
    p.add_argument("--xc", default="pbe0")
    p.add_argument("--nstates", type=int, default=5)
    p.add_argument("--grids-level", type=int, default=0)
    p.add_argument("--eta-ev", type=float, default=0.12)
    p.add_argument("--grid-points", type=int, default=2200)
    p.add_argument("--grid-max-ev", type=float, default=12.0)
    p.add_argument("--jax-jk-backend", choices=("full", "df", "direct"), default="full")
    p.add_argument(
        "--gpu4pyscf-density-fit",
        action="store_true",
        help="Use GPU4PySCF density fitting. Default is exact/non-DF dft.RKS(...).to_gpu().",
    )
    p.add_argument("--integral-backend", choices=("libcint",), default="libcint")
    p.add_argument(
        "--libcint-geometry-grad-policy",
        choices=("error", "zero"),
        default="error",
    )
    p.add_argument("--jax-precompile-eri", action="store_true")
    p.add_argument("--jax-precompile-eri-chunk-size", type=int, default=512)
    p.add_argument("--outdir", default="outputs/gpu4pyscf_vs_strict_jax_full")
    return p.parse_args()


def _sync_gpu() -> None:
    try:
        import cupy
    except ModuleNotFoundError:
        return
    try:
        cupy.cuda.get_current_stream().synchronize()
    except Exception:
        pass


def _time_call(fn, *, block=None):
    t0 = time.perf_counter()
    out = fn()
    if block is not None:
        block(out)
    _sync_gpu()
    return out, float(time.perf_counter() - t0)


def _restricted_channel(arr) -> np.ndarray:
    arr_np = np.asarray(arr, dtype=float)
    if arr_np.ndim == 1:
        return arr_np
    if arr_np.ndim == 2 and arr_np.shape[0] in (1, 2):
        return arr_np[0]
    raise ValueError(f"Unsupported restricted array shape: {arr_np.shape}")


def _block_reference(reference) -> None:
    import jax

    for name in (
        "ao",
        "ao_deriv1",
        "rep_tensor",
        "mo_coeff",
        "mo_occ",
        "mo_energy",
        "rdm1",
        "h1e",
        "overlap_matrix",
        "dipole_integrals",
        "eri_ovov",
        "eri_ovvo",
        "eri_oovv",
    ):
        value = getattr(reference, name, None)
        if value is not None:
            jax.block_until_ready(value)


def _require_gpu4pyscf():
    try:
        import gpu4pyscf  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ImportError(
            "gpu4pyscf is required for this benchmark. Install gpu4pyscf-cuda11x/cuda12x "
            "and run on a CUDA-capable host."
        ) from exc


def _build_gpu4pyscf_reference(
    *,
    atom: str,
    basis: str,
    xc: str,
    grids_level: int,
    density_fit: bool = False,
):
    _require_gpu4pyscf()
    mol = gto.M(
        atom=atom,
        basis=basis,
        unit="Angstrom",
        spin=0,
        charge=0,
        cart=True,
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = int(grids_level)
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    if not hasattr(mf, "to_gpu"):
        raise RuntimeError(
            "PySCF object does not expose to_gpu(). PySCF >= 2.5.0 is required for GPU4PySCF conversion."
        )
    if density_fit:
        mf = mf.density_fit()
    mf = mf.to_gpu()
    e_tot = mf.kernel()
    _sync_gpu()
    converged = bool(getattr(mf, "converged", False))
    if not converged:
        raise RuntimeError("GPU4PySCF SCF did not converge.")
    return mol, mf, float(e_tot)


def _run_gpu4pyscf_tda(mf, *, nstates: int):
    td = mf.TDA()
    td.nstates = int(nstates)
    td.kernel()
    _sync_gpu()
    energies = np.asarray(td.e, dtype=float).reshape(-1)
    strengths = np.asarray(td.oscillator_strength(), dtype=float).reshape(-1)
    return energies, strengths


def _run_gpu4pyscf_casida(mf, *, nstates: int):
    td = mf.TDDFT()
    td.nstates = int(nstates)
    td.kernel()
    _sync_gpu()
    energies = np.asarray(td.e, dtype=float).reshape(-1)
    strengths = np.asarray(td.oscillator_strength(), dtype=float).reshape(-1)
    return energies, strengths


def _build_strict_jax_reference(
    *,
    atom: str,
    basis: str,
    xc: str,
    grids_level: int,
    jk_backend: str,
    integral_backend: str,
    libcint_geometry_grad_policy: str,
    precompile_eri: bool,
    precompile_eri_chunk_size: int,
):
    from td_graddft.scf.builders import restricted_reference_from_spec_with_jax_rks
    from td_graddft.scf import RKSConfig

    cfg = RKSConfig(
        xc_spec=str(xc),
        max_cycle=32,
        conv_tol=1e-9,
        conv_tol_density=1e-7,
        damping=0.05,
        density_floor=1e-12,
        potential_clip=20.0,
        jk_backend=str(jk_backend),
    )
    return restricted_reference_from_spec_with_jax_rks(
        atom=atom,
        basis=basis,
        xc_spec=str(xc),
        unit="Angstrom",
        charge=0,
        spin=0,
        cart=True,
        grids_level=int(grids_level),
        max_l=3,
        rks_config=cfg,
        grid_ao_backend="jax",
        integral_backend=str(integral_backend),
        libcint_geometry_grad_policy=str(libcint_geometry_grad_policy),
        precompile_eri=bool(precompile_eri),
        precompile_eri_chunk_size=int(precompile_eri_chunk_size),
    )


def _run_strict_jax_tda(reference, *, xc: str, nstates: int):
    import jax

    from td_graddft import tdscf

    solver = tdscf.TDA(reference, xc_functional=str(xc))
    result = solver.kernel(nstates=int(nstates))
    strengths = solver.oscillator_strength()
    jax.block_until_ready(result.excitation_energies)
    jax.block_until_ready(strengths)
    return (
        np.asarray(result.excitation_energies, dtype=float).reshape(-1),
        np.asarray(strengths, dtype=float).reshape(-1),
    )


def _run_strict_jax_casida(reference, *, xc: str, nstates: int):
    import jax

    from td_graddft import tdscf

    solver = tdscf.TDDFT(reference, xc_functional=str(xc))
    result = solver.kernel(nstates=int(nstates))
    strengths = solver.oscillator_strength()
    jax.block_until_ready(result.excitation_energies)
    jax.block_until_ready(strengths)
    return (
        np.asarray(result.excitation_energies, dtype=float).reshape(-1),
        np.asarray(strengths, dtype=float).reshape(-1),
    )


def _state_rows(ref_e, pred_e, ref_f, pred_f, n: int) -> list[list[float]]:
    from td_graddft.spectra import HARTREE_TO_EV

    rows: list[list[float]] = []
    for idx in range(n):
        rows.append(
            [
                idx + 1,
                float(ref_e[idx] * HARTREE_TO_EV),
                float(pred_e[idx] * HARTREE_TO_EV),
                float(abs((pred_e[idx] - ref_e[idx]) * HARTREE_TO_EV)),
                float(ref_f[idx]),
                float(pred_f[idx]),
                float(abs(pred_f[idx] - ref_f[idx])),
            ]
        )
    return rows


def _write_state_csv(path: Path, rows: list[list[float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "state",
                "gpu4pyscf_energy_ev",
                "jax_energy_ev",
                "abs_diff_ev",
                "gpu4pyscf_osc",
                "jax_osc",
                "abs_diff_osc",
            ]
        )
        writer.writerows(rows)


def _spectrum_curve(energies_ha, strengths, grid_ev: np.ndarray, eta_ev: float) -> np.ndarray:
    import jax.numpy as jnp

    from td_graddft.spectra import HARTREE_TO_EV, lorentzian_spectrum

    return np.asarray(
        lorentzian_spectrum(
            jnp.asarray(np.asarray(energies_ha, dtype=float) * HARTREE_TO_EV),
            jnp.asarray(np.asarray(strengths, dtype=float)),
            jnp.asarray(grid_ev, dtype=float),
            eta=float(eta_ev),
        ),
        dtype=float,
    )


def main() -> None:
    args = _parse_args()

    import jax
    import matplotlib.pyplot as plt

    from td_graddft.spectra import HARTREE_TO_EV

    jax.config.update("jax_enable_x64", True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    atom = MOLECULES[str(args.molecule)]

    (_, mf_gpu, gpu_e_tot), gpu_scf_elapsed_s = _time_call(
        lambda: _build_gpu4pyscf_reference(
            atom=atom,
            basis=str(args.basis),
            xc=str(args.xc),
            grids_level=int(args.grids_level),
            density_fit=bool(args.gpu4pyscf_density_fit),
        )
    )
    (gpu_tda_e, gpu_tda_f), gpu_tda_elapsed_s = _time_call(
        lambda: _run_gpu4pyscf_tda(mf_gpu, nstates=int(args.nstates))
    )
    (gpu_casida_e, gpu_casida_f), gpu_casida_elapsed_s = _time_call(
        lambda: _run_gpu4pyscf_casida(mf_gpu, nstates=int(args.nstates))
    )

    ref, jax_reference_elapsed_s = _time_call(
        lambda: _build_strict_jax_reference(
            atom=atom,
            basis=str(args.basis),
            xc=str(args.xc),
            grids_level=int(args.grids_level),
            jk_backend=str(args.jax_jk_backend),
            integral_backend=str(args.integral_backend),
            libcint_geometry_grad_policy=str(args.libcint_geometry_grad_policy),
            precompile_eri=bool(args.jax_precompile_eri),
            precompile_eri_chunk_size=int(args.jax_precompile_eri_chunk_size),
        ),
        block=_block_reference,
    )
    (jax_tda_e, jax_tda_f), jax_tda_elapsed_s = _time_call(
        lambda: _run_strict_jax_tda(ref, xc=str(args.xc), nstates=int(args.nstates))
    )
    (jax_casida_e, jax_casida_f), jax_casida_elapsed_s = _time_call(
        lambda: _run_strict_jax_casida(ref, xc=str(args.xc), nstates=int(args.nstates))
    )

    nt = min(len(gpu_tda_e), len(jax_tda_e), len(gpu_tda_f), len(jax_tda_f))
    nc = min(len(gpu_casida_e), len(jax_casida_e), len(gpu_casida_f), len(jax_casida_f))
    tda_rows = _state_rows(gpu_tda_e, jax_tda_e, gpu_tda_f, jax_tda_f, nt)
    casida_rows = _state_rows(gpu_casida_e, jax_casida_e, gpu_casida_f, jax_casida_f, nc)

    stem = f"{str(args.molecule).lower()}_{str(args.xc).lower()}_{str(args.basis).lower()}_gpu4pyscf_vs_jax"
    _write_state_csv(outdir / f"{stem}_tda_states.csv", tda_rows)
    _write_state_csv(outdir / f"{stem}_casida_states.csv", casida_rows)

    grid_ev = np.linspace(0.0, float(args.grid_max_ev), int(args.grid_points))
    gpu_tda_curve = _spectrum_curve(gpu_tda_e[:nt], gpu_tda_f[:nt], grid_ev, float(args.eta_ev))
    jax_tda_curve = _spectrum_curve(jax_tda_e[:nt], jax_tda_f[:nt], grid_ev, float(args.eta_ev))
    gpu_casida_curve = _spectrum_curve(
        gpu_casida_e[:nc], gpu_casida_f[:nc], grid_ev, float(args.eta_ev)
    )
    jax_casida_curve = _spectrum_curve(
        jax_casida_e[:nc], jax_casida_f[:nc], grid_ev, float(args.eta_ev)
    )

    with (outdir / f"{stem}_spectrum.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "energy_ev",
                "gpu4pyscf_tda",
                "jax_tda",
                "gpu4pyscf_casida",
                "jax_casida",
            ]
        )
        for i in range(grid_ev.size):
            writer.writerow(
                [
                    float(grid_ev[i]),
                    float(gpu_tda_curve[i]),
                    float(jax_tda_curve[i]),
                    float(gpu_casida_curve[i]),
                    float(jax_casida_curve[i]),
                ]
            )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].bar(
        ["GPU4PySCF\nSCF", "JAX\nReference", "GPU4PySCF\nTDA", "JAX\nTDA", "GPU4PySCF\nCasida", "JAX\nCasida"],
        [
            gpu_scf_elapsed_s,
            jax_reference_elapsed_s,
            gpu_tda_elapsed_s,
            jax_tda_elapsed_s,
            gpu_casida_elapsed_s,
            jax_casida_elapsed_s,
        ],
        color=["#2F4F4F", "#B22222", "#2F4F4F", "#B22222", "#2F4F4F", "#B22222"],
    )
    axes[0].set_ylabel("Elapsed Time (s)")
    axes[0].set_title("Full Pipeline Timing")
    axes[0].tick_params(axis="x", rotation=18)

    axes[1].plot(grid_ev, gpu_tda_curve, lw=2.0, label="GPU4PySCF TDA")
    axes[1].plot(grid_ev, jax_tda_curve, lw=2.0, ls="--", label="JAX TDA")
    axes[1].plot(grid_ev, gpu_casida_curve, lw=2.0, label="GPU4PySCF Casida")
    axes[1].plot(grid_ev, jax_casida_curve, lw=2.0, ls="--", label="JAX Casida")
    axes[1].set_xlabel("Energy (eV)")
    axes[1].set_ylabel("Absorption")
    axes[1].set_title("Spectrum")
    axes[1].legend(frameon=False)

    states = np.arange(1, nt + 1)
    axes[2].plot(states, np.abs((jax_tda_e[:nt] - gpu_tda_e[:nt]) * HARTREE_TO_EV), marker="o", lw=2.0, label="TDA |ΔE|")
    if nc > 0:
        states_c = np.arange(1, nc + 1)
        axes[2].plot(states_c, np.abs((jax_casida_e[:nc] - gpu_casida_e[:nc]) * HARTREE_TO_EV), marker="s", lw=2.0, label="Casida |ΔE|")
    axes[2].set_xlabel("State")
    axes[2].set_ylabel("|ΔE| (eV)")
    axes[2].set_title("Excitation Error")
    axes[2].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(outdir / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "molecule": str(args.molecule),
        "basis": str(args.basis),
        "xc": str(args.xc),
        "nstates": int(args.nstates),
        "gpu4pyscf_density_fit": bool(args.gpu4pyscf_density_fit),
        "gpu4pyscf_total_energy_ha": float(gpu_e_tot),
        "jax_total_energy_ha": float(ref.mf_energy),
        "energy_diff_ha": float(ref.mf_energy - gpu_e_tot),
        "gpu4pyscf_scf_elapsed_s": float(gpu_scf_elapsed_s),
        "jax_reference_elapsed_s": float(jax_reference_elapsed_s),
        "gpu4pyscf_tda_elapsed_s": float(gpu_tda_elapsed_s),
        "jax_tda_elapsed_s": float(jax_tda_elapsed_s),
        "gpu4pyscf_casida_elapsed_s": float(gpu_casida_elapsed_s),
        "jax_casida_elapsed_s": float(jax_casida_elapsed_s),
        "tda_state_mae_ev": float(np.mean(np.abs((jax_tda_e[:nt] - gpu_tda_e[:nt]) * HARTREE_TO_EV))) if nt > 0 else float("nan"),
        "tda_osc_mae": float(np.mean(np.abs(jax_tda_f[:nt] - gpu_tda_f[:nt]))) if nt > 0 else float("nan"),
        "casida_state_mae_ev": float(np.mean(np.abs((jax_casida_e[:nc] - gpu_casida_e[:nc]) * HARTREE_TO_EV))) if nc > 0 else float("nan"),
        "casida_osc_mae": float(np.mean(np.abs(jax_casida_f[:nc] - gpu_casida_f[:nc]))) if nc > 0 else float("nan"),
        "tda_curve_mae": float(np.mean(np.abs(jax_tda_curve - gpu_tda_curve))),
        "casida_curve_mae": float(np.mean(np.abs(jax_casida_curve - gpu_casida_curve))),
    }

    with (outdir / f"{stem}_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
