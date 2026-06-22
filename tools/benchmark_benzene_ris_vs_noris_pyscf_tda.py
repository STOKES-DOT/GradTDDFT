from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import dft, gto
from pyscf.lib import logger
from pyscf.tdscf.rhf import lr_eigh

from td_graddft.data.reference import restricted_reference_from_pyscf
from td_graddft.features import restricted_grid_features_with_gradients
from td_graddft import spectra
from td_graddft.spectra import HARTREE_TO_EV
from td_graddft.tddft.response import build_restricted_tda_operator
from td_graddft.tddft.types import TDAResult
from td_graddft.xc_backend.jax_libxc import (
    eval_xc_response_tensor,
    hybrid_coeff,
    xc_type,
)


BENZENE_ATOM = """
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
"""


class SemilocalResponseFunctional:
    def __init__(self, xc_spec: str):
        self.xc_spec = str(xc_spec).lower()
        self.exact_exchange_fraction = float(hybrid_coeff(self.xc_spec))
        self.response_feature_kind = str(xc_type(self.xc_spec))

    def grid_response_tensor(self, molecule: Any):
        features, grad_rho = restricted_grid_features_with_gradients(molecule)
        tau = features.tau_a + features.tau_b
        _, tensor = eval_xc_response_tensor(
            self.xc_spec,
            features.rho,
            grad=grad_rho,
            tau=tau,
        )
        return tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare benzene TDA excitations from PySCF, JAX direct no-RIS, and JAX RIS."
    )
    parser.add_argument("--basis", default="def2-svp")
    parser.add_argument("--xc", default="b3lyp")
    parser.add_argument("--grids-level", type=int, default=0)
    parser.add_argument("--nstates", type=int, default=3)
    parser.add_argument("--ris-theta", type=float, default=0.2)
    parser.add_argument("--ris-j-fit", choices=("s", "sp", "spd"), default="sp")
    parser.add_argument("--ris-k-fit", choices=("s", "sp", "spd"), default="s")
    parser.add_argument("--davidson-tol", type=float, default=1e-5)
    parser.add_argument("--davidson-max-iter", type=int, default=100)
    parser.add_argument(
        "--davidson-max-subspace",
        type=int,
        default=0,
        help="Use 0 for PySCF TD lr_eigh-style unrestricted Davidson space.",
    )
    parser.add_argument(
        "--davidson-max-trial-vectors",
        type=int,
        default=20,
        help="Match PySCF TD lr_eigh MAX_SPACE_INC when expanding Davidson trials.",
    )
    parser.add_argument("--excitation-threshold", type=float, default=1e-3)
    parser.add_argument("--outdir", default="benchmark/benzene_ris_vs_noris_pyscf_tda")
    return parser.parse_args()


def _build_mf(*, basis: str, xc: str, grids_level: int):
    mol = gto.M(
        atom=BENZENE_ATOM,
        basis=basis,
        unit="Angstrom",
        spin=0,
        charge=0,
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = int(grids_level)
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"PySCF SCF did not converge for {xc}/{basis}.")
    return mol, mf


def _time_call(fn):
    t0 = time.perf_counter()
    out = fn()
    if hasattr(out, "block_until_ready"):
        out.block_until_ready()
    return out, float(time.perf_counter() - t0)


def _pyscf_initial_guess(
    mf,
    *,
    nstates: int,
    davidson_tol: float,
    davidson_max_iter: int,
    excitation_threshold: float,
) -> int:
    td = mf.TDA()
    td.nstates = int(nstates)
    td.conv_tol = float(davidson_tol)
    td.max_cycle = int(davidson_max_iter)
    td.positive_eig_threshold = float(excitation_threshold)
    x0, _ = td.init_guess(mf, int(nstates), return_symmetry=True)
    return np.asarray(x0, dtype=float)


def _run_pyscf_tda(
    mf,
    *,
    nstates: int,
    davidson_tol: float,
    davidson_max_iter: int,
    excitation_threshold: float,
):
    td = mf.TDA()
    td.nstates = int(nstates)
    td.conv_tol = float(davidson_tol)
    td.max_cycle = int(davidson_max_iter)
    td.positive_eig_threshold = float(excitation_threshold)
    _, elapsed_s = _time_call(td.kernel)
    return {
        "backend": "pyscf",
        "energies_h": np.asarray(td.e, dtype=float),
        "oscillator_strengths": np.asarray(td.oscillator_strength(), dtype=float),
        "elapsed_s": elapsed_s,
    }


def _run_jax_tda(
    reference,
    functional,
    *,
    nstates: int,
    x0: np.ndarray,
    mode: str,
    davidson_tol: float,
    davidson_max_iter: int,
    excitation_threshold: float,
    ris_options: dict[str, Any] | None = None,
):
    options = {"two_electron_mode": mode}
    if ris_options:
        options.update(ris_options)
    vind_rows, diagonal, delta_eps = build_restricted_tda_operator(
        reference,
        functional,
        response_kernel_options=options,
    )
    nocc, nvir = delta_eps.shape
    dim = int(nocc * nvir)
    hdiag = np.asarray(jax.device_get(jnp.asarray(diagonal).reshape(dim)), dtype=float)
    if int(x0.shape[1]) != dim:
        raise ValueError(f"PySCF initial guess dimension {x0.shape[1]} does not match JAX dim {dim}.")

    def aop(vectors):
        arr = jnp.asarray(np.asarray(vectors, dtype=float))
        out = vind_rows(arr)
        return np.asarray(jax.device_get(out), dtype=float)

    def precond(dx, energy):
        diagd = hdiag - float(np.asarray(energy).reshape(-1)[0])
        diagd[np.abs(diagd) < 1e-8] = 1e-8
        return dx / diagd

    def pickeig(w, v, nroots, envs):
        idx = np.where(w > float(excitation_threshold))[0]
        return w[idx], v[:, idx], idx

    def kernel():
        with open(os.devnull, "w", encoding="utf-8") as null:
            log = logger.Logger(null, logger.WARN)
            return lr_eigh(
                aop,
                x0,
                precond,
                tol_residual=float(davidson_tol),
                lindep=1e-12,
                nroots=int(nstates),
                x0sym=None,
                pick=pickeig,
                max_cycle=int(davidson_max_iter),
                max_memory=4000,
                verbose=log,
            )

    (converged, energies, amplitudes), elapsed_s = _time_call(kernel)
    if not bool(np.all(converged)):
        raise RuntimeError(
            f"PySCF lr_eigh did not converge for {mode} response: {np.asarray(converged)}"
        )
    result = TDAResult(
        excitation_energies=jnp.asarray(energies, dtype=jnp.asarray(diagonal).dtype),
        amplitudes=jnp.sqrt(jnp.asarray(0.5, dtype=jnp.asarray(diagonal).dtype))
        * jnp.asarray(amplitudes, dtype=jnp.asarray(diagonal).dtype).reshape(int(nstates), nocc, nvir),
    )
    osc = np.asarray(jax.device_get(spectra.oscillator_strengths(reference, result)), dtype=float)
    return {
        "backend": "jax_ris" if mode == "ris" else "jax_noris_direct",
        "energies_h": np.asarray(energies, dtype=float),
        "oscillator_strengths": osc,
        "elapsed_s": elapsed_s,
        "converged": np.asarray(converged, dtype=bool).tolist(),
    }


def _write_results_csv(path: Path, results: list[dict[str, Any]]) -> None:
    pyscf = next(row for row in results if row["backend"] == "pyscf")
    noris = next(row for row in results if row["backend"] == "jax_noris_direct")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "state",
                "backend",
                "energy_h",
                "energy_ev",
                "oscillator_strength",
                "delta_vs_pyscf_ev",
                "delta_ris_vs_noris_ev",
                "elapsed_s",
            ]
        )
        for result in results:
            backend = str(result["backend"])
            energies = np.asarray(result["energies_h"], dtype=float)
            osc = np.asarray(result["oscillator_strengths"], dtype=float)
            for idx, energy_h in enumerate(energies, start=1):
                delta_vs_pyscf = (energy_h - float(pyscf["energies_h"][idx - 1])) * HARTREE_TO_EV
                delta_ris_vs_noris = ""
                if backend == "jax_ris":
                    delta_ris_vs_noris = (
                        (energy_h - float(noris["energies_h"][idx - 1])) * HARTREE_TO_EV
                    )
                writer.writerow(
                    [
                        idx,
                        backend,
                        f"{energy_h:.12f}",
                        f"{energy_h * HARTREE_TO_EV:.8f}",
                        f"{float(osc[idx - 1]):.12e}",
                        f"{delta_vs_pyscf:.8f}",
                        f"{delta_ris_vs_noris:.8f}" if delta_ris_vs_noris != "" else "",
                        f"{float(result['elapsed_s']):.6f}",
                    ]
                )


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _, mf = _build_mf(
        basis=str(args.basis),
        xc=str(args.xc),
        grids_level=int(args.grids_level),
    )
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 1e-8))
    nvir = int(np.asarray(mf.mo_coeff).shape[-1] - nocc)
    nstates = min(max(1, int(args.nstates)), nocc * nvir)
    x0 = _pyscf_initial_guess(
        mf,
        nstates=nstates,
        davidson_tol=float(args.davidson_tol),
        davidson_max_iter=int(args.davidson_max_iter),
        excitation_threshold=float(args.excitation_threshold),
    )
    davidson_max_subspace = (
        None if int(args.davidson_max_subspace) <= 0 else int(args.davidson_max_subspace)
    )

    functional = SemilocalResponseFunctional(str(args.xc))
    direct_reference = restricted_reference_from_pyscf(
        mf,
        jk_backend="full",
        response_df_mode="none",
    )
    ris_reference = restricted_reference_from_pyscf(
        mf,
        jk_backend="df",
        response_df_mode="ris",
        response_ris_theta=float(args.ris_theta),
        response_ris_j_fit=str(args.ris_j_fit),
        response_ris_k_fit=str(args.ris_k_fit),
    )

    ris_options = {
        "ris_theta": float(args.ris_theta),
        "ris_j_fit": str(args.ris_j_fit),
        "ris_k_fit": str(args.ris_k_fit),
    }
    results = [
        _run_pyscf_tda(
            mf,
            nstates=nstates,
            davidson_tol=float(args.davidson_tol),
            davidson_max_iter=int(args.davidson_max_iter),
            excitation_threshold=float(args.excitation_threshold),
        ),
        _run_jax_tda(
            direct_reference,
            functional,
            nstates=nstates,
            x0=x0,
            mode="direct",
            davidson_tol=float(args.davidson_tol),
            davidson_max_iter=int(args.davidson_max_iter),
            excitation_threshold=float(args.excitation_threshold),
        ),
        _run_jax_tda(
            ris_reference,
            functional,
            nstates=nstates,
            x0=x0,
            mode="ris",
            davidson_tol=float(args.davidson_tol),
            davidson_max_iter=int(args.davidson_max_iter),
            excitation_threshold=float(args.excitation_threshold),
            ris_options=ris_options,
        ),
    ]

    results_csv = outdir / "benzene_b3lyp_tda_ris_vs_noris_pyscf.csv"
    _write_results_csv(results_csv, results)
    summary = {
        "molecule": "benzene",
        "xc": str(args.xc),
        "basis": str(args.basis),
        "grids_level": int(args.grids_level),
        "nstates": int(nstates),
        "pyscf_initial_guess_count": int(x0.shape[0]),
        "ris_theta": float(args.ris_theta),
        "ris_j_fit": str(args.ris_j_fit),
        "ris_k_fit": str(args.ris_k_fit),
        "davidson_tol": float(args.davidson_tol),
        "davidson_max_iter": int(args.davidson_max_iter),
        "davidson_max_subspace": (
            None if davidson_max_subspace is None else int(davidson_max_subspace)
        ),
        "davidson_max_trial_vectors": int(args.davidson_max_trial_vectors),
        "excitation_threshold": float(args.excitation_threshold),
        "jax_response_solver": "pyscf.tdscf.rhf.lr_eigh",
        "results_csv": str(results_csv),
        "elapsed_s": {str(row["backend"]): float(row["elapsed_s"]) for row in results},
        "energies_ev": {
            str(row["backend"]): [
                float(value) * HARTREE_TO_EV for value in np.asarray(row["energies_h"], dtype=float)
            ]
            for row in results
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
