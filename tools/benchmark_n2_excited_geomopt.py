from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto, scf, tdscf
from scipy.optimize import minimize

from td_graddft.scf import RHFConfig
from td_graddft_tools.geomopt_freq import (
    RHFExcitedStateSurfaceConfig,
    make_rhf_excited_state_surface_from_pyscf_mol,
)


def _build_n2_mol(r_angstrom: float, *, basis: str) -> gto.Mole:
    return gto.M(
        atom=f"""
        N 0.0 0.0 {-0.5 * r_angstrom:.12f}
        N 0.0 0.0 {+0.5 * r_angstrom:.12f}
        """,
        unit="Angstrom",
        basis=basis,
        spin=0,
        verbose=0,
    )


def _coords_from_bond_length(r_angstrom: float) -> jnp.ndarray:
    r = jnp.asarray(r_angstrom, dtype=jnp.float64)
    return jnp.asarray(
        [
            [0.0, 0.0, -0.5 * r],
            [0.0, 0.0, +0.5 * r],
        ],
        dtype=jnp.float64,
    )


def _bond_length_gradient_from_cartesian(cart_grad: np.ndarray) -> float:
    grad = np.asarray(cart_grad, dtype=float)
    dr_dcoords = np.asarray(
        [
            [0.0, 0.0, -0.5],
            [0.0, 0.0, +0.5],
        ],
        dtype=float,
    )
    return float(np.sum(grad * dr_dcoords))


def _optimize_with_jax_surface(
    *,
    basis: str,
    state_index: int,
    response_method: str,
    initial_r_angstrom: float,
    bounds: tuple[float, float],
    scf_max_cycle: int,
    maxiter: int,
) -> dict[str, float | int | bool]:
    mol0 = _build_n2_mol(initial_r_angstrom, basis=basis)
    surface = make_rhf_excited_state_surface_from_pyscf_mol(
        mol0,
        config=RHFExcitedStateSurfaceConfig(
            scf=RHFConfig(max_cycle=scf_max_cycle),
            state_index=state_index,
            response_method=response_method,
            coordinate_unit="angstrom",
            eigensolver="dense",
        ),
    )

    def objective_r(r_value: jax.Array) -> jax.Array:
        coords = _coords_from_bond_length(r_value)
        return surface.energy(coords)

    value_and_grad = jax.value_and_grad(objective_r)

    def scipy_objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = value_and_grad(jnp.asarray(x[0], dtype=jnp.float64))
        return float(value), np.asarray([float(grad)], dtype=float)

    def fun(x: np.ndarray) -> float:
        value, _ = scipy_objective(x)
        return value

    def jac_fn(x: np.ndarray) -> np.ndarray:
        _, grad = scipy_objective(x)
        return grad

    x0 = np.asarray([initial_r_angstrom], dtype=float)
    t0 = time.perf_counter()
    result = minimize(
        fun,
        x0,
        jac=jac_fn,
        method="L-BFGS-B",
        bounds=[bounds],
        options={"maxiter": int(maxiter)},
    )
    elapsed_s = time.perf_counter() - t0
    final_energy = float(fun(result.x))
    return {
        "optimized_r_angstrom": float(result.x[0]),
        "final_total_energy_ha": final_energy,
        "elapsed_s": float(elapsed_s),
        "iterations": int(getattr(result, "nit", 0)),
        "function_evaluations": int(getattr(result, "nfev", 0)),
        "gradient_evaluations": int(getattr(result, "njev", 0)),
        "success": bool(result.success),
    }


def _optimize_with_pyscf(
    *,
    basis: str,
    state_index: int,
    initial_r_angstrom: float,
    bounds: tuple[float, float],
    maxiter: int,
) -> dict[str, float | int | bool]:
    nstates = state_index + 1

    def pyscf_value_and_grad(r_value: float) -> tuple[float, float]:
        mol = _build_n2_mol(float(r_value), basis=basis)
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10
        mf.max_cycle = 120
        mf.kernel()
        if not mf.converged:
            raise RuntimeError(f"PySCF RHF did not converge at r={r_value:.6f} A.")

        td = tdscf.TDA(mf)
        td.nstates = nstates
        td.kernel()
        if len(td.e) <= state_index:
            raise RuntimeError(
                f"PySCF TDA returned only {len(td.e)} states, need state_index={state_index}."
            )
        total_energy = float(mf.e_tot + td.e[state_index])
        cart_grad = np.asarray(td.nuc_grad_method().kernel(state=state_index), dtype=float)
        grad_r = _bond_length_gradient_from_cartesian(cart_grad)
        return total_energy, grad_r

    def fun(x: np.ndarray) -> float:
        value, _ = pyscf_value_and_grad(float(x[0]))
        return value

    def jac_fn(x: np.ndarray) -> np.ndarray:
        _, grad = pyscf_value_and_grad(float(x[0]))
        return np.asarray([grad], dtype=float)

    x0 = np.asarray([initial_r_angstrom], dtype=float)
    t0 = time.perf_counter()
    result = minimize(
        fun,
        x0,
        jac=jac_fn,
        method="L-BFGS-B",
        bounds=[bounds],
        options={"maxiter": int(maxiter)},
    )
    elapsed_s = time.perf_counter() - t0
    final_energy = float(fun(result.x))
    return {
        "optimized_r_angstrom": float(result.x[0]),
        "final_total_energy_ha": final_energy,
        "elapsed_s": float(elapsed_s),
        "iterations": int(getattr(result, "nit", 0)),
        "function_evaluations": int(getattr(result, "nfev", 0)),
        "gradient_evaluations": int(getattr(result, "njev", 0)),
        "success": bool(result.success),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark N2 first-excited-state geometry optimization: JAX RHF+TDA/Casida vs PySCF RHF+TDA.",
    )
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--response-method", choices=("tda", "casida"), default="tda")
    parser.add_argument("--initial-r", type=float, default=1.10)
    parser.add_argument("--r-min", type=float, default=0.90)
    parser.add_argument("--r-max", type=float, default=2.00)
    parser.add_argument("--surface-scf-max-cycle", type=int, default=30)
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument(
        "--outdir",
        default="outputs/n2_excited_geomopt_benchmark",
        help="Directory for JSON/text benchmark outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    jax.config.update("jax_enable_x64", True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / ".mplconfig").mkdir(parents=True, exist_ok=True)

    bounds = (float(args.r_min), float(args.r_max))
    jax_result = _optimize_with_jax_surface(
        basis=str(args.basis),
        state_index=int(args.state_index),
        response_method=str(args.response_method),
        initial_r_angstrom=float(args.initial_r),
        bounds=bounds,
        scf_max_cycle=int(args.surface_scf_max_cycle),
        maxiter=int(args.maxiter),
    )
    pyscf_result = _optimize_with_pyscf(
        basis=str(args.basis),
        state_index=int(args.state_index),
        initial_r_angstrom=float(args.initial_r),
        bounds=bounds,
        maxiter=int(args.maxiter),
    )

    report = {
        "system": "N2",
        "basis": str(args.basis),
        "state_index": int(args.state_index),
        "response_method": str(args.response_method),
        "initial_r_angstrom": float(args.initial_r),
        "bounds_angstrom": [float(args.r_min), float(args.r_max)],
        "jax_rhf_local_tddft": jax_result,
        "pyscf_rhf_tda": pyscf_result,
        "differences": {
            "delta_r_angstrom": float(
                jax_result["optimized_r_angstrom"] - pyscf_result["optimized_r_angstrom"]
            ),
            "delta_time_s": float(jax_result["elapsed_s"] - pyscf_result["elapsed_s"]),
            "time_ratio_jax_over_pyscf": float(
                jax_result["elapsed_s"] / max(float(pyscf_result["elapsed_s"]), 1e-12)
            ),
        },
    }

    json_path = outdir / "benchmark.json"
    summary_path = outdir / "summary.txt"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    summary = (
        f"N2 excited-state geometry benchmark ({args.response_method.upper()}, state {args.state_index + 1})\n"
        f"basis = {args.basis}\n"
        f"initial r (A) = {args.initial_r:.6f}\n"
        f"JAX optimized r (A) = {jax_result['optimized_r_angstrom']:.8f}\n"
        f"PySCF optimized r (A) = {pyscf_result['optimized_r_angstrom']:.8f}\n"
        f"delta r (A) = {report['differences']['delta_r_angstrom']:.8f}\n"
        f"JAX elapsed (s) = {jax_result['elapsed_s']:.3f}\n"
        f"PySCF elapsed (s) = {pyscf_result['elapsed_s']:.3f}\n"
        f"time ratio (JAX/PySCF) = {report['differences']['time_ratio_jax_over_pyscf']:.3f}\n"
    )
    summary_path.write_text(summary)

    print(summary, end="")
    print(f"Wrote JSON summary to: {json_path}")
    print(f"Wrote text summary to: {summary_path}")


if __name__ == "__main__":
    main()
