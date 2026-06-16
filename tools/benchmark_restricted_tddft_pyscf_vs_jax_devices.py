from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.lax import Precision
from pyscf import dft, gto
from pyscf.dft import gen_grid, numint

from td_graddft.data import evaluate_cartesian_ao
from td_graddft.data.basis import basis_from_pyscf_mol_cart
from td_graddft.data.integrals import build_hcore, eri_tensor, overlap_matrix
from td_graddft.features import restricted_grid_features_with_gradients
from td_graddft.jax_runtime import (
    DEFAULT_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS,
    DEFAULT_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES,
    configure_jax_persistent_cache,
)
from td_graddft.xc_backend.jax_libxc import eval_xc_response_tensor, hybrid_coeff, xc_type
from td_graddft.scf.molecules import QuadratureGrid, RestrictedMolecule
from td_graddft.scf import RKSConfig, run_rks_from_integrals_traceable
from td_graddft import tdscf


def _linear_acene_atoms(n_rings: int, *, c_c: float = 1.397, c_h: float = 1.09):
    if n_rings < 1:
        raise ValueError("n_rings must be >= 1")

    centers = [(math.sqrt(3.0) * c_c * ring, 0.0) for ring in range(n_rings)]
    vertex_offsets = [
        (0.0, c_c),
        (math.sqrt(3.0) * 0.5 * c_c, 0.5 * c_c),
        (math.sqrt(3.0) * 0.5 * c_c, -0.5 * c_c),
        (0.0, -c_c),
        (-math.sqrt(3.0) * 0.5 * c_c, -0.5 * c_c),
        (-math.sqrt(3.0) * 0.5 * c_c, 0.5 * c_c),
    ]

    vertices: list[np.ndarray] = []
    neighbors: list[set[int]] = []
    index_by_key: dict[tuple[int, int], int] = {}

    def get_index(point: np.ndarray) -> int:
        key = (int(round(point[0] * 1_000_000)), int(round(point[1] * 1_000_000)))
        idx = index_by_key.get(key)
        if idx is not None:
            return idx
        idx = len(vertices)
        index_by_key[key] = idx
        vertices.append(point)
        neighbors.append(set())
        return idx

    for cx, cy in centers:
        ring = []
        for dx, dy in vertex_offsets:
            ring.append(get_index(np.asarray([cx + dx, cy + dy], dtype=float)))
        for i in range(6):
            a = ring[i]
            b = ring[(i + 1) % 6]
            neighbors[a].add(b)
            neighbors[b].add(a)

    atoms: list[tuple[str, tuple[float, float, float]]] = []
    for point in vertices:
        atoms.append(("C", (float(point[0]), float(point[1]), 0.0)))

    for idx, point in enumerate(vertices):
        if len(neighbors[idx]) != 2:
            continue
        bonded = np.asarray([vertices[j] for j in neighbors[idx]], dtype=float)
        inward = (bonded[0] - point) + (bonded[1] - point)
        norm = float(np.linalg.norm(inward))
        if norm <= 1e-12:
            continue
        outward = -inward / norm
        h = point + c_h * outward
        atoms.append(("H", (float(h[0]), float(h[1]), 0.0)))
    return atoms


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
    "anthracene": _linear_acene_atoms(3),
}


@dataclass(frozen=True)
class BenchmarkRow:
    backend: str
    stage: str
    run: int
    elapsed_s: float
    e_tot_ha: float
    first_excitation_ha: float


_GRID_DATA_CACHE: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _grid_cache_key(
    mol,
    *,
    grids_level: int,
    max_l: int,
    grid_ao_backend: str,
    jax_grid_chunk_size: int,
) -> tuple[Any, ...]:
    charges = tuple(int(z) for z in np.asarray(mol.atom_charges(), dtype=np.int32).tolist())
    coords = tuple(
        tuple(float(x) for x in xyz)
        for xyz in np.asarray(mol.atom_coords(), dtype=np.float64).tolist()
    )
    return (
        charges,
        coords,
        str(mol.basis),
        int(grids_level),
        int(max_l),
        str(grid_ao_backend).lower(),
        int(jax_grid_chunk_size),
    )


class SemilocalResponseFunctional:
    def __init__(self, xc_spec: str):
        self.xc_spec = str(xc_spec)
        self.exact_exchange_fraction = float(hybrid_coeff(self.xc_spec))
        self.response_feature_kind = str(xc_type(self.xc_spec))

    def grid_response_tensor(self, molecule):
        features, grad_rho = restricted_grid_features_with_gradients(molecule)
        tau = features.tau_a + features.tau_b
        _, tensor = eval_xc_response_tensor(
            self.xc_spec,
            features.rho,
            grad=grad_rho,
            tau=tau,
        )
        return tensor


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--molecule", choices=tuple(MOLECULES.keys()), default="anthracene")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--xc", default="pbe0")
    p.add_argument("--nstates", type=int, default=5)
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--mode", choices=("davidson", "auto"), default="davidson")
    p.add_argument("--grids-level", type=int, default=0)
    p.add_argument("--grid-ao-backend", choices=("pyscf", "jax"), default="pyscf")
    p.add_argument(
        "--jax-grid-chunk-size",
        type=int,
        default=0,
        help="Chunk size for JAX AO-on-grid evaluation (<=0 disables chunking).",
    )
    p.add_argument("--max-l", type=int, default=3)
    p.add_argument("--integral-engine", choices=("auto", "jit", "legacy"), default="auto")
    p.add_argument(
        "--integral-backend",
        choices=("cpu", "libcint"),
        default="cpu",
        help="AO integral source for jax_cpu/jax_gpu backends",
    )
    p.add_argument("--jk-backend", choices=("full", "df"), default="full")
    p.add_argument("--df-tol", type=float, default=1e-10)
    p.add_argument("--df-max-rank", type=int, default=0)
    p.add_argument("--scf-max-cycle", type=int, default=80)
    p.add_argument("--scf-damping", type=float, default=0.10)
    p.add_argument(
        "--jit-scf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="JIT compile the JAX SCF stage.",
    )
    p.add_argument(
        "--jit-tddft",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="JIT compile the JAX TDA/Casida stage.",
    )
    p.add_argument(
        "--warmup-jit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run an unmeasured warmup call before timing JIT-compiled stages.",
    )
    p.add_argument(
        "--jax-cache-dir",
        default=".jax_cache/restricted_tddft_benchmark",
        help="persistent JAX compilation cache directory (empty string disables)",
    )
    p.add_argument(
        "--jax-cache-min-compile-secs",
        type=float,
        default=DEFAULT_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS,
        help="minimum compile time (s) for entries persisted to cache",
    )
    p.add_argument("--backends", default="pyscf_cpu,jax_cpu,jax_gpu")
    p.add_argument("--outdir", default="outputs/restricted_tddft_pyscf_vs_jax_devices")
    return p.parse_args()


def _backend_list(raw: str) -> list[str]:
    allowed = {"pyscf_cpu", "jax_cpu", "jax_gpu"}
    backends = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not backends:
        raise ValueError("At least one backend must be requested.")
    unknown = sorted(set(backends) - allowed)
    if unknown:
        raise ValueError(f"Unsupported backends: {unknown!r}")
    return backends


def _build_mol(molecule: str, basis: str):
    return gto.M(
        atom=MOLECULES[molecule],
        basis=basis,
        unit="Angstrom",
        spin=0,
        charge=0,
        cart=True,
        verbose=0,
    )


def _charge_center(mol) -> np.ndarray:
    charges = np.asarray(mol.atom_charges(), dtype=np.float64)
    coords = np.asarray(mol.atom_coords(), dtype=np.float64)
    return np.einsum("z,zr->r", charges, coords) / charges.sum()


def _restricted_response_eri_slices_from_mo_tensor(
    rep_tensor,
    mo_coeff,
    nocc: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    coeff = jnp.asarray(mo_coeff)
    rep = jnp.asarray(rep_tensor)
    orbo = coeff[:, :nocc]
    orbv = coeff[:, nocc:]
    eri_ovov = jnp.einsum(
        "pqrs,pi,qa,rj,sb->iajb",
        rep,
        orbo,
        orbv,
        orbo,
        orbv,
        precision=Precision.HIGHEST,
    )
    eri_ovvo = jnp.einsum(
        "pqrs,pi,qa,rb,sj->iabj",
        rep,
        orbo,
        orbv,
        orbv,
        orbo,
        precision=Precision.HIGHEST,
    )
    eri_oovv = jnp.einsum(
        "pqrs,pi,qj,ra,sb->ijab",
        rep,
        orbo,
        orbo,
        orbv,
        orbv,
        precision=Precision.HIGHEST,
    )
    return eri_ovov, eri_ovvo, eri_oovv


def _build_grid_data(
    mol,
    grids_level: int,
    *,
    max_l: int,
    grid_ao_backend: str,
    jax_grid_chunk_size: int,
):
    cache_key = _grid_cache_key(
        mol,
        grids_level=int(grids_level),
        max_l=int(max_l),
        grid_ao_backend=str(grid_ao_backend),
        jax_grid_chunk_size=int(jax_grid_chunk_size),
    )
    cached = _GRID_DATA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    grids = gen_grid.Grids(mol)
    grids.level = int(grids_level)
    grids.build()
    coords = np.asarray(grids.coords, dtype=np.float64)
    weights = np.asarray(grids.weights, dtype=np.float64)
    if str(grid_ao_backend).lower() == "jax":
        basis = basis_from_pyscf_mol_cart(mol, max_l=int(max_l))
        ao_deriv1 = np.asarray(
            evaluate_cartesian_ao(
                basis,
                coords,
                deriv=1,
                chunk_size=None if int(jax_grid_chunk_size) <= 0 else int(jax_grid_chunk_size),
            ),
            dtype=np.float64,
        )
        ao = np.asarray(ao_deriv1[0], dtype=np.float64)
    else:
        # Avoid duplicate AO evaluation: deriv=1 already includes values on channel 0.
        ao_deriv1 = np.asarray(numint.eval_ao(mol, coords, deriv=1), dtype=np.float64)
        ao = np.asarray(ao_deriv1[0], dtype=np.float64)
    packed = (coords, weights, ao, ao_deriv1)
    _GRID_DATA_CACHE[cache_key] = packed
    return packed


def _build_ao_integrals_on_device(
    mol,
    *,
    max_l: int,
    integral_engine: str,
    integral_backend: str,
    device,
):
    t0 = time.perf_counter()
    dipole_cpu = None
    with mol.with_common_orig(_charge_center(mol)):
        dipole_cpu = np.asarray(mol.intor_symmetric("int1e_r", comp=3), dtype=np.float64)

    backend_mode = str(integral_backend).lower()
    if backend_mode not in {"jax", "cpu", "gpu", "libcint"}:
        raise ValueError(
            f"Unsupported integral_backend={integral_backend!r}. Expected 'jax', 'cpu', or 'gpu'."
        )

    with jax.default_device(device):
        if backend_mode in {"cpu", "libcint"}:
            overlap = jax.device_put(jnp.asarray(mol.intor_symmetric("int1e_ovlp"), dtype=jnp.float64), device)
            hcore = jax.device_put(
                jnp.asarray(mol.intor_symmetric("int1e_kin"), dtype=jnp.float64)
                + jnp.asarray(mol.intor_symmetric("int1e_nuc"), dtype=jnp.float64),
                device,
            )
            eri = jax.device_put(jnp.asarray(mol.intor("int2e"), dtype=jnp.float64), device)
        else:
            basis_cart = basis_from_pyscf_mol_cart(mol, max_l=int(max_l))
            overlap = overlap_matrix(basis_cart, engine=integral_engine)
            hcore = build_hcore(basis_cart, engine=integral_engine)
            eri = eri_tensor(basis_cart, engine=integral_engine)

        dipole_integrals = jax.device_put(jnp.asarray(dipole_cpu), device)
        jax.block_until_ready(overlap)
        jax.block_until_ready(hcore)
        jax.block_until_ready(eri)
        jax.block_until_ready(dipole_integrals)
    elapsed = time.perf_counter() - t0
    return overlap, hcore, eri, dipole_integrals, float(elapsed)


def _measure_pyscf_full_pipeline(
    molecule: str,
    *,
    basis: str,
    xc: str,
    grids_level: int,
    nstates: int,
) -> list[BenchmarkRow]:
    mol = _build_mol(molecule, basis)
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = int(grids_level)
    mf.conv_tol = 1e-10
    mf.max_cycle = 120

    rows: list[BenchmarkRow] = []
    t0 = time.perf_counter()
    mf.kernel()
    scf_elapsed = time.perf_counter() - t0
    if not mf.converged:
        raise RuntimeError("PySCF SCF did not converge.")
    rows.append(
        BenchmarkRow(
            backend="pyscf_cpu",
            stage="scf",
            run=-1,
            elapsed_s=float(scf_elapsed),
            e_tot_ha=float(mf.e_tot),
            first_excitation_ha=float("nan"),
        )
    )

    t1 = time.perf_counter()
    tda = mf.TDA()
    tda.nstates = int(nstates)
    tda.kernel()
    rows.append(
        BenchmarkRow(
            backend="pyscf_cpu",
            stage="tda",
            run=-1,
            elapsed_s=float(time.perf_counter() - t1),
            e_tot_ha=float(mf.e_tot),
            first_excitation_ha=float(np.asarray(tda.e, dtype=float).reshape(-1)[0]),
        )
    )

    t2 = time.perf_counter()
    td = mf.TDDFT()
    td.nstates = int(nstates)
    td.kernel()
    rows.append(
        BenchmarkRow(
            backend="pyscf_cpu",
            stage="kernel",
            run=-1,
            elapsed_s=float(time.perf_counter() - t2),
            e_tot_ha=float(mf.e_tot),
            first_excitation_ha=float(np.asarray(td.e, dtype=float).reshape(-1)[0]),
        )
    )
    return rows


def _build_jax_reference(
    *,
    molecule: str,
    basis: str,
    xc: str,
    grids_level: int,
    grid_ao_backend: str,
    jax_grid_chunk_size: int,
    max_l: int,
    integral_engine: str,
    integral_backend: str,
    scf_max_cycle: int,
    scf_damping: float,
    jit_scf: bool,
    warmup_jit: bool,
    jk_backend: str,
    df_tol: float,
    df_max_rank: int,
    device,
) -> tuple[RestrictedMolecule, float, float, float]:
    mol = _build_mol(molecule, basis)
    t0 = time.perf_counter()
    coords_cpu, weights_cpu, ao_cpu, ao_deriv1_cpu = _build_grid_data(
        mol,
        grids_level,
        max_l=max_l,
        grid_ao_backend=grid_ao_backend,
        jax_grid_chunk_size=jax_grid_chunk_size,
    )
    ao = jax.device_put(jnp.asarray(ao_cpu), device)
    ao_deriv1 = jax.device_put(jnp.asarray(ao_deriv1_cpu), device)
    weights = jax.device_put(jnp.asarray(weights_cpu), device)
    coords = jax.device_put(jnp.asarray(coords_cpu), device)
    jax.block_until_ready(ao)
    jax.block_until_ready(ao_deriv1)
    jax.block_until_ready(weights)
    jax.block_until_ready(coords)
    grid_prep_elapsed = time.perf_counter() - t0

    overlap, hcore, eri, dipole_integrals, integral_prep_elapsed = _build_ao_integrals_on_device(
        mol,
        max_l=max_l,
        integral_engine=integral_engine,
        integral_backend=integral_backend,
        device=device,
    )

    cfg = RKSConfig(
        xc_spec=str(xc),
        max_cycle=int(scf_max_cycle),
        conv_tol=1e-10,
        conv_tol_density=1e-8,
        damping=float(scf_damping),
        potential_clip=20.0,
        jk_backend=str(jk_backend),
        df_tol=float(df_tol),
        df_max_rank=None if int(df_max_rank) <= 0 else int(df_max_rank),
    )

    with jax.default_device(device):
        def _run_scf():
            return run_rks_from_integrals_traceable(
                overlap=overlap,
                hcore=hcore,
                eri=eri,
                nelectron=int(mol.nelectron),
                nuclear_repulsion=float(mol.energy_nuc()),
                ao=ao,
                ao_deriv1=ao_deriv1,
                grid_weights=weights,
                config=cfg,
            )

        run_scf = jax.jit(_run_scf) if bool(jit_scf) else _run_scf
        if bool(jit_scf) and bool(warmup_jit):
            warm = run_scf()
            jax.block_until_ready(warm.density_matrix)

        t1 = time.perf_counter()
        rks = run_scf()
        jax.block_until_ready(rks.density_matrix)
        scf_elapsed = time.perf_counter() - t1

        if not bool(np.asarray(rks.converged)):
            raise RuntimeError("JAX RKS did not converge.")

        dm_total = jnp.asarray(rks.density_matrix)
        mo_coeff = jnp.asarray(rks.mo_coeff)
        mo_energy = jnp.asarray(rks.mo_energy)
        mo_occ_total = jnp.asarray(rks.mo_occ)
        half_dm = dm_total / 2.0
        half_occ = mo_occ_total / 2.0
        nocc = int(np.count_nonzero(np.asarray(mo_occ_total) > 1e-8))
        eri_ovov, eri_ovvo, eri_oovv = _restricted_response_eri_slices_from_mo_tensor(
            eri,
            mo_coeff,
            nocc,
        )
        reference = RestrictedMolecule(
            ao=ao,
            grid=QuadratureGrid(
                weights=weights,
                coords=coords,
            ),
            dipole_integrals=dipole_integrals,
            rep_tensor=eri,
            mo_coeff=jnp.stack([mo_coeff, mo_coeff], axis=0),
            mo_occ=jnp.stack([half_occ, half_occ], axis=0),
            mo_energy=jnp.stack([mo_energy, mo_energy], axis=0),
            rdm1=jnp.stack([half_dm, half_dm], axis=0),
            h1e=hcore,
            nuclear_repulsion=float(np.asarray(rks.nuclear_repulsion)),
            overlap_matrix=overlap,
            ao_deriv1=ao_deriv1,
            mf_energy=float(np.asarray(rks.total_energy)),
            exact_exchange_fraction=float(hybrid_coeff(xc)),
            nocc=nocc,
            eri_ovov=eri_ovov,
            eri_ovvo=eri_ovvo,
            eri_oovv=eri_oovv,
        )
    return reference, float(grid_prep_elapsed), float(integral_prep_elapsed), float(scf_elapsed)


def _measure_jax_full_pipeline(
    molecule: str,
    *,
    basis: str,
    xc: str,
    nstates: int,
    backend: str,
    mode: str,
    grids_level: int,
    grid_ao_backend: str,
    jax_grid_chunk_size: int,
    max_l: int,
    integral_engine: str,
    integral_backend: str,
    scf_max_cycle: int,
    scf_damping: float,
    jit_scf: bool,
    jit_tddft: bool,
    warmup_jit: bool,
    jk_backend: str,
    df_tol: float,
    df_max_rank: int,
) -> list[BenchmarkRow]:
    if backend == "jax_cpu":
        device = jax.devices("cpu")[0]
    else:
        gpu_devices = jax.devices("gpu")
        if not gpu_devices:
            raise RuntimeError("No JAX GPU device is available.")
        device = gpu_devices[0]

    reference, grid_prep_elapsed, integral_prep_elapsed, scf_elapsed = _build_jax_reference(
        molecule=molecule,
        basis=basis,
        xc=xc,
        grids_level=grids_level,
        grid_ao_backend=grid_ao_backend,
        jax_grid_chunk_size=jax_grid_chunk_size,
        max_l=max_l,
        integral_engine=integral_engine,
        integral_backend=integral_backend,
        scf_max_cycle=scf_max_cycle,
        scf_damping=scf_damping,
        jit_scf=jit_scf,
        warmup_jit=warmup_jit,
        jk_backend=jk_backend,
        df_tol=df_tol,
        df_max_rank=df_max_rank,
        device=device,
    )
    xc_func = SemilocalResponseFunctional(xc)
    rows = [
        BenchmarkRow(
            backend=backend,
            stage="grid_prep",
            run=-1,
            elapsed_s=float(grid_prep_elapsed),
            e_tot_ha=float("nan"),
            first_excitation_ha=float("nan"),
        ),
        BenchmarkRow(
            backend=backend,
            stage="integral_prep",
            run=-1,
            elapsed_s=float(integral_prep_elapsed),
            e_tot_ha=float("nan"),
            first_excitation_ha=float("nan"),
        ),
        BenchmarkRow(
            backend=backend,
            stage="scf",
            run=-1,
            elapsed_s=float(scf_elapsed),
            e_tot_ha=float(reference.mf_energy),
            first_excitation_ha=float("nan"),
        ),
    ]

    with jax.default_device(device):
        tda_solver = tdscf.TDA(
            reference,
            xc_functional=xc_func,
            eigensolver=mode,
            davidson_tol=1e-5,
            davidson_max_iter=160,
            davidson_max_subspace=64,
        )
        solver = tdscf.TDDFT(
            reference,
            xc_functional=xc_func,
            eigensolver=mode,
            davidson_tol=1e-5,
            davidson_max_iter=160,
            davidson_max_subspace=64,
        )
        tda_fn = tda_solver.kernel
        kernel_fn = solver.kernel
        if bool(jit_tddft):
            tda_fn = jax.jit(tda_fn, static_argnames=("nstates",))
            kernel_fn = jax.jit(kernel_fn, static_argnames=("nstates",))
        if bool(jit_tddft) and bool(warmup_jit):
            tda_w = tda_fn(nstates=nstates)
            ker_w = kernel_fn(nstates=nstates)
            jax.block_until_ready(tda_w.excitation_energies)
            jax.block_until_ready(ker_w.excitation_energies)

        t0 = time.perf_counter()
        tda = tda_fn(nstates=nstates)
        np.asarray(jax.block_until_ready(tda.excitation_energies))
        rows.append(
            BenchmarkRow(
                backend=backend,
                stage="tda",
                run=-1,
                elapsed_s=float(time.perf_counter() - t0),
                e_tot_ha=float(reference.mf_energy),
                first_excitation_ha=float(np.asarray(tda.excitation_energies)[0]),
            )
        )

        t1 = time.perf_counter()
        kernel = kernel_fn(nstates=nstates)
        np.asarray(jax.block_until_ready(kernel.excitation_energies))
        rows.append(
            BenchmarkRow(
                backend=backend,
                stage="kernel",
                run=-1,
                elapsed_s=float(time.perf_counter() - t1),
                e_tot_ha=float(reference.mf_energy),
                first_excitation_ha=float(np.asarray(kernel.excitation_energies)[0]),
            )
        )
    return rows


def _steady(rows: list[BenchmarkRow], backend: str, stage: str) -> BenchmarkRow:
    candidates = [r for r in rows if r.backend == backend and r.stage == stage]
    if not candidates:
        raise KeyError((backend, stage))
    return candidates[-1]


def _sum_elapsed(rows: list[BenchmarkRow], backend: str, stages: tuple[str, ...]) -> float:
    total = 0.0
    for stage in stages:
        total += _steady(rows, backend, stage).elapsed_s
    return float(total)


def main() -> None:
    args = _parse_args()
    configure_jax_persistent_cache(
        cache_dir=str(args.jax_cache_dir),
        min_compile_time_secs=float(args.jax_cache_min_compile_secs),
        min_entry_size_bytes=DEFAULT_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES,
    )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    backends = _backend_list(args.backends)

    rows: list[BenchmarkRow] = []
    for backend in backends:
        for run in range(1, int(args.runs) + 1):
            if backend == "pyscf_cpu":
                backend_rows = _measure_pyscf_full_pipeline(
                    args.molecule,
                    basis=args.basis,
                    xc=args.xc,
                    grids_level=int(args.grids_level),
                    nstates=int(args.nstates),
                )
            else:
                backend_rows = _measure_jax_full_pipeline(
                    args.molecule,
                    basis=args.basis,
                    xc=args.xc,
                    nstates=int(args.nstates),
                    backend=backend,
                    mode=str(args.mode),
                    grids_level=int(args.grids_level),
                    grid_ao_backend=str(args.grid_ao_backend),
                    jax_grid_chunk_size=int(args.jax_grid_chunk_size),
                    max_l=int(args.max_l),
                    integral_engine=str(args.integral_engine),
                    integral_backend=str(args.integral_backend),
                    scf_max_cycle=int(args.scf_max_cycle),
                    scf_damping=float(args.scf_damping),
                    jit_scf=bool(args.jit_scf),
                    jit_tddft=bool(args.jit_tddft),
                    warmup_jit=bool(args.warmup_jit),
                    jk_backend=str(args.jk_backend),
                    df_tol=float(args.df_tol),
                    df_max_rank=int(args.df_max_rank),
                )
            for row in backend_rows:
                materialized = BenchmarkRow(
                    backend=row.backend,
                    stage=row.stage,
                    run=run,
                    elapsed_s=row.elapsed_s,
                    e_tot_ha=row.e_tot_ha,
                    first_excitation_ha=row.first_excitation_ha,
                )
                rows.append(materialized)
                print(asdict(materialized))

    summary: dict[str, float | int | str] = {
        "molecule": args.molecule,
        "basis": args.basis,
        "xc": args.xc,
        "mode": args.mode,
        "grid_ao_backend": args.grid_ao_backend,
        "integral_backend": args.integral_backend,
        "jk_backend": args.jk_backend,
        "nstates": int(args.nstates),
        "runs": int(args.runs),
    }

    if "pyscf_cpu" in backends:
        pyscf_full_tda = _sum_elapsed(rows, "pyscf_cpu", ("scf", "tda"))
        pyscf_full_kernel = _sum_elapsed(rows, "pyscf_cpu", ("scf", "kernel"))
        summary["pyscf_cpu_full_tda_s"] = pyscf_full_tda
        summary["pyscf_cpu_full_kernel_s"] = pyscf_full_kernel
        summary["pyscf_cpu_e_tot_ha"] = _steady(rows, "pyscf_cpu", "scf").e_tot_ha

    for backend in ("jax_cpu", "jax_gpu"):
        if backend not in backends:
            continue
        full_tda = _sum_elapsed(rows, backend, ("grid_prep", "integral_prep", "scf", "tda"))
        full_kernel = _sum_elapsed(rows, backend, ("grid_prep", "integral_prep", "scf", "kernel"))
        summary[f"{backend}_full_tda_s"] = full_tda
        summary[f"{backend}_full_kernel_s"] = full_kernel
        summary[f"{backend}_e_tot_ha"] = _steady(rows, backend, "scf").e_tot_ha
        if "pyscf_cpu" in backends:
            summary[f"{backend}_full_tda_speedup_over_pyscf_cpu"] = (
                summary["pyscf_cpu_full_tda_s"] / full_tda
            )
            summary[f"{backend}_full_kernel_speedup_over_pyscf_cpu"] = (
                summary["pyscf_cpu_full_kernel_s"] / full_kernel
            )
            summary[f"{backend}_e_tot_abs_diff_ha_vs_pyscf"] = abs(
                _steady(rows, backend, "scf").e_tot_ha - _steady(rows, "pyscf_cpu", "scf").e_tot_ha
            )
            summary[f"{backend}_tda_abs_diff_ha_vs_pyscf"] = abs(
                _steady(rows, backend, "tda").first_excitation_ha
                - _steady(rows, "pyscf_cpu", "tda").first_excitation_ha
            )
            summary[f"{backend}_kernel_abs_diff_ha_vs_pyscf"] = abs(
                _steady(rows, backend, "kernel").first_excitation_ha
                - _steady(rows, "pyscf_cpu", "kernel").first_excitation_ha
            )

    if {"jax_cpu", "jax_gpu"}.issubset(backends):
        summary["jax_gpu_full_tda_speedup_over_jax_cpu"] = (
            summary["jax_cpu_full_tda_s"] / summary["jax_gpu_full_tda_s"]
        )
        summary["jax_gpu_full_kernel_speedup_over_jax_cpu"] = (
            summary["jax_cpu_full_kernel_s"] / summary["jax_gpu_full_kernel_s"]
        )
        summary["jax_gpu_e_tot_abs_diff_ha_vs_jax_cpu"] = abs(
            _steady(rows, "jax_gpu", "scf").e_tot_ha - _steady(rows, "jax_cpu", "scf").e_tot_ha
        )

    stem = f"{args.molecule}_{args.xc}_{args.basis}_{args.mode}_{args.integral_backend}"
    csv_path = outdir / f"{stem}_timings.csv"
    json_path = outdir / f"{stem}_summary.json"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "backend",
                "stage",
                "run",
                "elapsed_s",
                "e_tot_ha",
                "first_excitation_ha",
            ]
        )
        for row in rows:
            w.writerow(
                [
                    row.backend,
                    row.stage,
                    row.run,
                    f"{row.elapsed_s:.6f}",
                    f"{row.e_tot_ha:.12f}" if math.isfinite(row.e_tot_ha) else "",
                    (
                        f"{row.first_excitation_ha:.12f}"
                        if math.isfinite(row.first_excitation_ha)
                        else ""
                    ),
                ]
            )
    with json_path.open("w") as f:
        json.dump(
            {
                "config": vars(args),
                "rows": [asdict(r) for r in rows],
                "summary": summary,
            },
            f,
            indent=2,
        )

    print("summary:", summary)
    print(f"csv={csv_path}")
    print(f"json={json_path}")


if __name__ == "__main__":
    main()
