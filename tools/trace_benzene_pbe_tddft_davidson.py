from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import scipy.linalg
from pyscf import dft, gto, lib
from pyscf.lib import logger
from pyscf.tdscf.rhf import lr_eigh, real_eig

from td_graddft.data.reference import restricted_reference_from_pyscf
from td_graddft.features import restricted_grid_features_with_gradients
from td_graddft.tddft.casida import _restricted_delta_eps
from td_graddft.tddft.response import gen_tdhf_vind
from td_graddft.tddft.types import TDDFTResult
from td_graddft.xc_backend.jax_libxc import eval_xc_response_tensor, hybrid_coeff, xc_type


HARTREE_TO_EV = 27.211386245988

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
    p = argparse.ArgumentParser(
        description="Trace PySCF and TD-GradDFT Davidson histories for benzene PBE TDDFT."
    )
    p.add_argument("--basis", default="def2-tzvp")
    p.add_argument("--xc", default="pbe")
    p.add_argument("--grids-level", type=int, default=0)
    p.add_argument("--nstates", type=int, default=5)
    p.add_argument("--davidson-tol", type=float, default=1e-5)
    p.add_argument("--davidson-max-iter", type=int, default=100)
    p.add_argument("--davidson-max-subspace", type=int, default=0)
    p.add_argument("--max-memory", type=float, default=4000.0)
    p.add_argument("--tdgraddft-jk-backend", choices=("full", "df"), default="full")
    p.add_argument(
        "--tdgraddft-response-mode",
        choices=("direct", "df", "auto"),
        default="direct",
        help="Two-electron backend for the TD-GradDFT TDDFT response operator.",
    )
    p.add_argument("--outdir", default="benchmark/benzene_pbe_def2tzvp_tddft_davidson_trace_20260715")
    return p.parse_args()


def build_mf(*, basis: str, xc: str, grids_level: int):
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
    mf.max_cycle = 160
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"PySCF RKS did not converge for {xc}/{basis}.")
    return mol, mf


def _pick_positive(w, v, nroots, _envs):
    idx = np.where(np.asarray(w) > 1e-3)[0]
    return w[idx], v[:, idx], idx


def _td_init_guess(td, mf, nstates: int):
    if hasattr(td, "init_guess"):
        return td.init_guess(mf, int(nstates), return_symmetry=True)
    return td.get_init_guess(mf, int(nstates), return_symmetry=True)


def _parse_lr_eigh_history(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current_root_residuals: dict[int, float] = {}
    pattern = re.compile(
        r"davidson\s+(\d+)\s+(\d+)\s+\|r\|=\s*([0-9.eE+-]+).*max\|de\|=\s*([0-9.eE+-]+).*lindep=\s*([0-9.eE+-]+)"
    )
    conv_pattern = re.compile(
        r"converged\s+(\d+)\s+(\d+)\s+\|r\|=\s*([0-9.eE+-]+).*max\|de\|=\s*([0-9.eE+-]+)"
    )
    root_pattern = re.compile(
        r"root\s+(\d+)(?:\s+converged)?\s+\|r\|=\s*([0-9.eE+-]+)"
    )

    def pop_root_residuals() -> list[float]:
        nonlocal current_root_residuals
        values = [current_root_residuals[k] for k in sorted(current_root_residuals)]
        current_root_residuals = {}
        return values

    for line in text.splitlines():
        root_match = root_pattern.search(line)
        if root_match:
            current_root_residuals[int(root_match.group(1))] = float(root_match.group(2))
            continue
        match = pattern.search(line)
        if match:
            rows.append(
                {
                    "solver": "pyscf_casida_lr_eigh",
                    "iteration": int(match.group(1)),
                    "subspace_dim": int(match.group(2)),
                    "max_residual": float(match.group(3)),
                    "max_delta_e": float(match.group(4)),
                    "lindep": float(match.group(5)),
                    "root_residuals": pop_root_residuals(),
                    "converged": False,
                }
            )
            continue
        match = conv_pattern.search(line)
        if match:
            rows.append(
                {
                    "solver": "pyscf_casida_lr_eigh",
                    "iteration": int(match.group(1)),
                    "subspace_dim": int(match.group(2)),
                    "max_residual": float(match.group(3)),
                    "max_delta_e": float(match.group(4)),
                    "lindep": "",
                    "root_residuals": pop_root_residuals(),
                    "converged": True,
                }
            )
    return rows


def _parse_real_eig_history(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current_root_residuals: dict[int, float] = {}
    last_generated_vectors = 0
    last_subspace_dim = 0
    pattern = re.compile(
        r"real_lr_eig\s+(\d+)\s+(\d+)\s+\|r\|=\s*([0-9.eE+-]+).*max\|de\|=\s*([0-9.eE+-]+).*lindep=\s*([0-9.eE+-]+)"
    )
    conv_pattern = re.compile(
        r"converged\s+(\d+)\s+(\d+)\s+\|r\|=\s*([0-9.eE+-]+).*max\|de\|=\s*([0-9.eE+-]+)"
    )
    root_pattern = re.compile(
        r"root\s+(\d+)(?:\s+converged)?\s+\|r\|=\s*([0-9.eE+-]+)"
    )
    generated_pattern = re.compile(r"Generate\s+(\d+)\s+trial vectors")

    def pop_root_residuals() -> list[float]:
        nonlocal current_root_residuals
        values = [current_root_residuals[k] for k in sorted(current_root_residuals)]
        current_root_residuals = {}
        return values

    for line in text.splitlines():
        root_match = root_pattern.search(line)
        if root_match:
            current_root_residuals[int(root_match.group(1))] = float(root_match.group(2))
            continue
        generated_match = generated_pattern.search(line)
        if generated_match:
            last_generated_vectors = int(generated_match.group(1))
            continue
        match = pattern.search(line)
        if match:
            last_subspace_dim = int(match.group(2))
            rows.append(
                {
                    "solver": "pyscf_tdhf_real_eig",
                    "iteration": int(match.group(1)),
                    "subspace_dim": last_subspace_dim,
                    "max_residual": float(match.group(3)),
                    "max_delta_e": float(match.group(4)),
                    "lindep": float(match.group(5)),
                    "root_residuals": pop_root_residuals(),
                    "converged": False,
                }
            )
            continue
        match = conv_pattern.search(line)
        if match:
            rows.append(
                {
                    "solver": "pyscf_tdhf_real_eig",
                    "iteration": int(match.group(1)),
                    "subspace_dim": last_subspace_dim + last_generated_vectors,
                    "max_residual": float(match.group(3)),
                    "max_delta_e": float(match.group(4)),
                    "lindep": "",
                    "root_residuals": pop_root_residuals(),
                    "converged": True,
                }
            )
    return rows


def run_pyscf_casida_trace(mf, *, nstates: int, tol: float, max_iter: int, max_memory: float):
    td = mf.TDDFT()
    td.nstates = int(nstates)
    td.conv_tol = float(tol)
    td.max_cycle = int(max_iter)
    td.max_memory = float(max_memory)
    vind, hdiag = td.gen_vind(mf)
    precond = td.get_precond(hdiag)
    x0, x0sym = _td_init_guess(td, mf, int(nstates))

    buf = io.StringIO()
    log = logger.Logger(buf, logger.DEBUG1)
    t0 = time.perf_counter()
    conv, w2, x1 = lr_eigh(
        vind,
        x0,
        precond,
        tol_residual=float(tol),
        lindep=td.lindep,
        nroots=int(nstates),
        x0sym=x0sym,
        pick=_pick_positive,
        max_cycle=int(max_iter),
        max_memory=float(max_memory),
        verbose=log,
    )
    elapsed = time.perf_counter() - t0
    energies = np.sqrt(np.asarray(w2, dtype=float))
    history = _parse_lr_eigh_history(buf.getvalue())
    return {
        "solver": "pyscf_casida_lr_eigh",
        "equation": "Hermitian Casida Omega z = omega^2 z",
        "converged": np.asarray(conv, dtype=bool).tolist(),
        "energies_h": energies.tolist(),
        "energies_ev": (energies * HARTREE_TO_EV).tolist(),
        "elapsed_s": elapsed,
        "initial_guess_shape": list(np.asarray(x0).shape),
        "hdiag_shape": list(np.asarray(hdiag).shape),
        "history": history,
        "raw_log": buf.getvalue(),
        "raw_vectors_shape": list(np.asarray(x1).shape),
    }


def run_pyscf_tdhf_trace(mf, *, nstates: int, tol: float, max_iter: int, max_memory: float):
    td = mf.TDDFT()
    td.nstates = int(nstates)
    td.conv_tol = float(tol)
    td.max_cycle = int(max_iter)
    td.max_memory = float(max_memory)
    vind, hdiag = td.gen_vind(mf)
    precond = td.get_precond(hdiag)
    x0, x0sym = _td_init_guess(td, mf, int(nstates))

    buf = io.StringIO()
    log = logger.Logger(buf, logger.DEBUG1)
    t0 = time.perf_counter()
    conv, energies, x1 = real_eig(
        vind,
        x0,
        precond,
        tol_residual=float(tol),
        lindep=td.lindep,
        nroots=int(nstates),
        x0sym=x0sym,
        pick=None,
        max_cycle=int(max_iter),
        max_memory=float(max_memory),
        verbose=log,
    )
    elapsed = time.perf_counter() - t0
    energies = np.asarray(energies, dtype=float)
    history = _parse_real_eig_history(buf.getvalue())
    return {
        "solver": "pyscf_tdhf_real_eig",
        "equation": "full TDHF/TDDFT [A B; -B -A] [X,Y] = omega [X,Y]",
        "converged": np.asarray(conv, dtype=bool).tolist(),
        "energies_h": energies.tolist(),
        "energies_ev": (energies * HARTREE_TO_EV).tolist(),
        "elapsed_s": elapsed,
        "initial_guess_shape": list(np.asarray(x0).shape),
        "hdiag_shape": list(np.asarray(hdiag).shape),
        "history": history,
        "raw_log": buf.getvalue(),
        "raw_vectors_shape": list(np.asarray(x1).shape),
    }


def run_pyscf_trace(mf, *, nstates: int, tol: float, max_iter: int, max_memory: float):
    is_hybrid = bool(mf._numint.libxc.is_hybrid_xc(mf.xc))
    if is_hybrid:
        return run_pyscf_tdhf_trace(
            mf,
            nstates=nstates,
            tol=tol,
            max_iter=max_iter,
            max_memory=max_memory,
        )
    return run_pyscf_casida_trace(
        mf,
        nstates=nstates,
        tol=tol,
        max_iter=max_iter,
        max_memory=max_memory,
    )


def _symmetrize(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T.conj())


def _tdhf_subspace_eigen_solver_np(
    a: np.ndarray,
    b: np.ndarray,
    sigma: np.ndarray,
    pi: np.ndarray,
    *,
    nroots: int,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dim = int(a.shape[0])
    eye = np.eye(dim, dtype=a.dtype)
    d = np.maximum(np.abs(np.diag(sigma)), eps)
    d_mh = d**-0.5
    s_m_p = d_mh[:, None] * (sigma - pi) * d_mh[None, :]
    lu_l, lu_u = scipy.linalg.lu(s_m_p, permute_l=True, check_finite=False)
    l_inv = np.linalg.inv(lu_l)
    u_inv = np.linalg.inv(lu_u)

    d_amb_d = d_mh[:, None] * (a - b) * d_mh[None, :]
    ggt = _symmetrize(u_inv.T.conj() @ d_amb_d @ u_inv)
    g = np.linalg.cholesky(ggt + eps * eye)
    g_inv = np.linalg.inv(g)

    d_apb_d = d_mh[:, None] * (a + b) * d_mh[None, :]
    m = _symmetrize(g.T.conj() @ l_inv @ d_apb_d @ l_inv.T.conj() @ g)
    omega2_all, z_all = np.linalg.eigh(m)
    order = np.argsort(np.where(omega2_all > eps, omega2_all, np.inf))
    omega2 = np.maximum(omega2_all[order][:nroots], eps)
    z = z_all[:, order][:, :nroots]
    omega = np.sqrt(omega2)

    x_plus_y = d_mh[:, None] * (l_inv.T.conj() @ (g @ z)) * omega[None, :] ** -0.5
    x_minus_y = d_mh[:, None] * (u_inv @ (g_inv.T.conj() @ z)) * omega[None, :] ** 0.5
    x = 0.5 * (x_plus_y + x_minus_y)
    y = x_plus_y - x
    return omega, x, y


def trace_tdgraddft_tdhf(
    vind,
    diag: np.ndarray,
    *,
    nroots: int,
    tol: float,
    max_iter: int,
    max_subspace: int | None,
    matrix_eps: float = 1e-10,
    preconditioner_floor: float = 1e-8,
    orth_eps: float = 1e-10,
):
    diag = np.asarray(diag, dtype=float).reshape(-1)
    dim = int(diag.size)
    if max_subspace is None:
        max_subspace = dim
    max_subspace = min(dim, max(int(max_subspace), int(nroots) + 2))

    def apply_pair(v_cols: np.ndarray, w_cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = np.concatenate([v_cols.T, w_cols.T], axis=-1)
        applied = np.asarray(jax.device_get(vind(jnp.asarray(rows, dtype=jnp.float64))), dtype=float)
        applied = applied.reshape(-1, 2 * dim)
        return applied[:, :dim].T, -applied[:, dim:].T

    guess_idx = np.argsort(diag)[: min(dim, max_subspace, nroots)]
    v_basis = [np.eye(dim, dtype=float)[:, idx] for idx in guess_idx]
    w_basis = [np.zeros(dim, dtype=float) for _ in guess_idx]
    u1_init, u2_init = apply_pair(np.column_stack(v_basis), np.column_stack(w_basis))
    u1_basis = [u1_init[:, i].copy() for i in range(u1_init.shape[1])]
    u2_basis = [u2_init[:, i].copy() for i in range(u2_init.shape[1])]

    full_diag = np.concatenate([diag, -diag])
    best = None
    history: list[dict[str, Any]] = []

    for iteration in range(int(max_iter)):
        v_mat = np.column_stack(v_basis)
        w_mat = np.column_stack(w_basis)
        u1_mat = np.column_stack(u1_basis)
        u2_mat = np.column_stack(u2_basis)
        m = v_mat.shape[1]

        a = _symmetrize(v_mat.T.conj() @ u1_mat + w_mat.T.conj() @ u2_mat)
        b = _symmetrize(v_mat.T.conj() @ u2_mat + w_mat.T.conj() @ u1_mat)
        sigma = _symmetrize(v_mat.T.conj() @ v_mat - w_mat.T.conj() @ w_mat)
        pi = 0.5 * ((v_mat.T.conj() @ w_mat - w_mat.T.conj() @ v_mat) - (v_mat.T.conj() @ w_mat - w_mat.T.conj() @ v_mat).T.conj())
        omega, x_sub, y_sub = _tdhf_subspace_eigen_solver_np(
            a,
            b,
            sigma,
            pi,
            nroots=int(nroots),
            eps=float(matrix_eps),
        )
        x_full = v_mat @ x_sub + w_mat @ y_sub
        y_full = w_mat @ x_sub + v_mat @ y_sub
        r_x = u1_mat @ x_sub + u2_mat @ y_sub - x_full * omega[None, :]
        r_y = u2_mat @ x_sub + u1_mat @ y_sub + y_full * omega[None, :]
        residual_norms = np.sqrt(np.sum(np.abs(r_x) ** 2, axis=0) + np.sum(np.abs(r_y) ** 2, axis=0))
        max_residual = float(np.max(residual_norms[:nroots]))
        if best is None or max_residual < best["max_residual"]:
            best = {
                "omega": omega.copy(),
                "x": x_full.copy(),
                "y": y_full.copy(),
                "max_residual": max_residual,
            }
        history.append(
            {
                "solver": "tdgraddft_tdhf_davidson",
                "iteration": int(iteration),
                "subspace_dim": int(m),
                "max_residual": max_residual,
                "root_residuals": [float(v) for v in residual_norms[:nroots]],
                "energies_h": [float(v) for v in omega[:nroots]],
                "energies_ev": [float(v * HARTREE_TO_EV) for v in omega[:nroots]],
                "converged": bool(max_residual <= tol),
            }
        )
        if max_residual <= tol:
            break

        denom_base = full_diag[:, None] - omega[None, :]
        denom_sign = np.where(denom_base < 0.0, -1.0, 1.0)
        denom = np.where(
            np.abs(denom_base) < preconditioner_floor,
            denom_sign * preconditioner_floor,
            denom_base,
        )
        correction = np.concatenate([r_x, r_y], axis=0) / denom
        new_x = correction[:dim, :]
        new_y = correction[dim:, :]
        new_mask = residual_norms > tol

        accepted_x: list[np.ndarray] = []
        accepted_y: list[np.ndarray] = []
        for idx in range(min(int(nroots), new_x.shape[1])):
            if not bool(new_mask[idx]):
                continue
            x = new_x[:, idx].copy()
            y = new_y[:, idx].copy()
            x -= v_mat @ (v_mat.T.conj() @ x)
            x -= w_mat @ (w_mat.T.conj() @ x)
            y -= w_mat @ (w_mat.T.conj() @ y)
            y -= v_mat @ (v_mat.T.conj() @ y)
            if accepted_x:
                ax = np.column_stack(accepted_x)
                ay = np.column_stack(accepted_y)
                x -= ax @ (ax.T.conj() @ x)
                y -= ay @ (ay.T.conj() @ y)
            pair_norm = float(np.sqrt(np.sum(np.abs(x) ** 2) + np.sum(np.abs(y) ** 2)))
            if pair_norm > orth_eps:
                accepted_x.append(x / pair_norm)
                accepted_y.append(y / pair_norm)

        if not accepted_x:
            break
        if len(v_basis) + len(accepted_x) > max_subspace:
            x_seed = x_full[:, :nroots]
            y_seed = y_full[:, :nroots]
            u1_seed, u2_seed = apply_pair(x_seed, y_seed)
            v_basis = [x_seed[:, i].copy() for i in range(x_seed.shape[1])]
            w_basis = [y_seed[:, i].copy() for i in range(y_seed.shape[1])]
            u1_basis = [u1_seed[:, i].copy() for i in range(u1_seed.shape[1])]
            u2_basis = [u2_seed[:, i].copy() for i in range(u2_seed.shape[1])]

        x_new_mat = np.column_stack(accepted_x)
        y_new_mat = np.column_stack(accepted_y)
        u1_new, u2_new = apply_pair(x_new_mat, y_new_mat)
        for idx in range(x_new_mat.shape[1]):
            v_basis.append(x_new_mat[:, idx].copy())
            w_basis.append(y_new_mat[:, idx].copy())
            u1_basis.append(u1_new[:, idx].copy())
            u2_basis.append(u2_new[:, idx].copy())

    if best is None:
        raise RuntimeError("TD-GradDFT Davidson trace did not run any iteration.")
    return {
        "solver": "tdgraddft_tdhf_davidson",
        "equation": "full TDHF/TDDFT [A B; -B -A] [X,Y] = omega [X,Y]",
        "converged": bool(history[-1]["converged"]),
        "energies_h": [float(v) for v in best["omega"][:nroots]],
        "energies_ev": [float(v * HARTREE_TO_EV) for v in best["omega"][:nroots]],
        "elapsed_s": None,
        "initial_guess_shape": [int(nroots), int(2 * dim)],
        "hdiag_shape": [int(dim)],
        "history": history,
    }


def run_tdgraddft_trace(
    mf,
    *,
    xc: str,
    jk_backend: str,
    response_mode: str,
    nstates: int,
    tol: float,
    max_iter: int,
    max_subspace: int | None,
):
    reference = restricted_reference_from_pyscf(
        mf,
        jk_backend=str(jk_backend),
        response_df_mode="df" if str(response_mode) == "df" else "none",
    )
    functional = SemilocalResponseFunctional(xc)
    response_options = {"two_electron_mode": str(response_mode)}
    vind = gen_tdhf_vind(reference, functional, response_kernel_options=response_options)
    delta_eps = np.asarray(jax.device_get(_restricted_delta_eps(reference, 1e-8)), dtype=float)
    t0 = time.perf_counter()
    traced = trace_tdgraddft_tdhf(
        vind,
        delta_eps.reshape(-1),
        nroots=int(nstates),
        tol=float(tol),
        max_iter=int(max_iter),
        max_subspace=max_subspace,
    )
    traced["elapsed_s"] = time.perf_counter() - t0

    # Run the production path once to confirm the traced forward energies.
    from td_graddft import tdscf

    prod = tdscf.TDDFT(
        reference,
        xc_functional=functional,
        eigensolver="davidson",
        davidson_tol=float(tol),
        davidson_max_iter=int(max_iter),
        davidson_max_subspace=max_subspace,
        response_kernel_options=response_options,
    ).kernel(nstates=int(nstates))
    prod_e = np.asarray(jax.device_get(prod.excitation_energies), dtype=float)
    traced["production_energies_h"] = prod_e.tolist()
    traced["production_energies_ev"] = (prod_e * HARTREE_TO_EV).tolist()
    traced["production_converged"] = bool(np.asarray(jax.device_get(prod.converged)))
    return traced


def write_history_csv(path: Path, histories: list[dict[str, Any]]) -> None:
    rows = []
    for block in histories:
        for row in block["history"]:
            rows.append(row)
    keys = [
        "solver",
        "iteration",
        "subspace_dim",
        "max_residual",
        "max_delta_e",
        "lindep",
        "root_residuals",
        "energies_h",
        "energies_ev",
        "converged",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key)) if isinstance(row.get(key), list) else row.get(key, "") for key in keys})


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _, mf = build_mf(basis=str(args.basis), xc=str(args.xc), grids_level=int(args.grids_level))
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 1e-8))
    nmo = int(np.asarray(mf.mo_coeff).shape[-1])
    nvir = nmo - nocc
    nstates = min(int(args.nstates), nocc * nvir)
    max_subspace = None if int(args.davidson_max_subspace) <= 0 else int(args.davidson_max_subspace)

    pyscf_trace = run_pyscf_trace(
        mf,
        nstates=nstates,
        tol=float(args.davidson_tol),
        max_iter=int(args.davidson_max_iter),
        max_memory=float(args.max_memory),
    )
    tdgraddft_trace = run_tdgraddft_trace(
        mf,
        xc=str(args.xc),
        jk_backend=str(args.tdgraddft_jk_backend),
        response_mode=str(args.tdgraddft_response_mode),
        nstates=nstates,
        tol=float(args.davidson_tol),
        max_iter=int(args.davidson_max_iter),
        max_subspace=max_subspace,
    )

    (outdir / "pyscf_davidson_debug.log").write_text(pyscf_trace.pop("raw_log"), encoding="utf-8")
    write_history_csv(outdir / "davidson_history.csv", [pyscf_trace, tdgraddft_trace])

    summary = {
        "molecule": "benzene",
        "xc": str(args.xc),
        "basis": str(args.basis),
        "grids_level": int(args.grids_level),
        "nocc": nocc,
        "nvir": nvir,
        "nmo": nmo,
        "dim_nocc_x_nvir": int(nocc * nvir),
        "nstates": int(nstates),
        "davidson_tol": float(args.davidson_tol),
        "davidson_max_iter": int(args.davidson_max_iter),
        "davidson_max_subspace": max_subspace,
        "tdgraddft_jk_backend": str(args.tdgraddft_jk_backend),
        "tdgraddft_response_mode": str(args.tdgraddft_response_mode),
        "pyscf": pyscf_trace,
        "tdgraddft": tdgraddft_trace,
        "energy_delta_tdgraddft_minus_pyscf_ev": [
            float(a - b)
            for a, b in zip(tdgraddft_trace["energies_ev"], pyscf_trace["energies_ev"])
        ],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
