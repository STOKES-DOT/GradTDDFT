from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax
from pyscf import ao2mo, dft, fci, gto, scf

from td_graddft.neural_xc import (
    DEFAULT_INPUT_FEATURE_MODE,
    DEFAULT_NETWORK_ARCHITECTURE,
    DEFAULT_NETWORK_HIDDEN_DIMS,
)

GRADDFT_DEFAULT_DM21_HIDDEN_DIMS = DEFAULT_NETWORK_HIDDEN_DIMS
GRADDFT_DEFAULT_INPUT_FEATURE_MODE = DEFAULT_INPUT_FEATURE_MODE
GRADDFT_DEFAULT_NETWORK_ARCHITECTURE = DEFAULT_NETWORK_ARCHITECTURE
_DEFAULT_SEMILOCAL_XC = ("lda_x", "gga_x_b88", "lda_c_vwn_rpa", "gga_c_lyp")
_DEFAULT_TRAIN_SCF_SAFETY_MAX_CYCLE = 512

basis_from_spec = None
evaluate_cartesian_ao = None
neural_xc = None
restricted_reference_from_pyscf = None
restricted_molecule_from_spec_with_jax_rks = None
RKSConfig = None
HARTREE_TO_EV = None
GroundStateCoreDatum = None
GroundStateCoreTrainingConfig = None
GroundStateDatum = None
GroundStateTrainingConfig = None
create_train_state_from_molecule = None
ground_state_mse_loss = None
load_params_checkpoint = None
make_ground_state_train_step = None
save_params_checkpoint = None
predict_ground_state_molecule = None
predict_excitation_energies = None
predict_ground_state_total_energy = None
make_ground_state_predictor = None

_RUNTIME_DEPENDENCIES_LOADED = False


def _resolve_train_scf_max_cycle(value: int) -> int:
    return _DEFAULT_TRAIN_SCF_SAFETY_MAX_CYCLE if int(value) <= 0 else int(value)


def _load_runtime_dependencies(logger: "RunLogger | None" = None) -> None:
    def _log(message: str) -> None:
        if logger is not None:
            logger.log(message)
        else:
            print(message, flush=True)

    global _RUNTIME_DEPENDENCIES_LOADED
    global basis_from_spec
    global evaluate_cartesian_ao
    global neural_xc
    global restricted_reference_from_pyscf
    global restricted_molecule_from_spec_with_jax_rks
    global RKSConfig
    global HARTREE_TO_EV
    global GroundStateCoreDatum
    global GroundStateCoreTrainingConfig
    global GroundStateDatum
    global GroundStateTrainingConfig
    global create_train_state_from_molecule
    global ground_state_mse_loss
    global load_params_checkpoint
    global make_ground_state_train_step
    global save_params_checkpoint
    global predict_ground_state_molecule
    global predict_excitation_energies
    global predict_ground_state_total_energy
    global make_ground_state_predictor

    if _RUNTIME_DEPENDENCIES_LOADED:
        return

    _log("[bootstrap] import td_graddft.data.basis")
    from td_graddft.data.basis import basis_from_spec as _basis_from_spec
    _log("[bootstrap] import td_graddft.data.grid_ao")
    from td_graddft.data.grid_ao import evaluate_cartesian_ao as _evaluate_cartesian_ao
    _log("[bootstrap] import td_graddft.neural_xc")
    from td_graddft import neural_xc as _neural_xc
    _log("[bootstrap] import td_graddft.data.reference")
    from td_graddft.data.reference import restricted_reference_from_pyscf as _restricted_reference_from_pyscf
    _log("[bootstrap] import td_graddft.scf")
    from td_graddft.scf import (
        RKSConfig as _RKSConfig,
        restricted_molecule_from_spec_with_jax_rks as _restricted_molecule_from_spec_with_jax_rks,
    )
    _log("[bootstrap] import td_graddft.spectra")
    from td_graddft.spectra import HARTREE_TO_EV as _HARTREE_TO_EV
    _log("[bootstrap] import td_graddft.training")
    from td_graddft.training import (
        GroundStateCoreDatum as _GroundStateCoreDatum,
        GroundStateCoreTrainingConfig as _GroundStateCoreTrainingConfig,
        GroundStateDatum as _GroundStateDatum,
        GroundStateTrainingConfig as _GroundStateTrainingConfig,
        create_train_state_from_molecule as _create_train_state_from_molecule,
        ground_state_mse_loss as _ground_state_mse_loss,
        load_params_checkpoint as _load_params_checkpoint,
        make_ground_state_train_step as _make_ground_state_train_step,
        make_ground_state_predictor as _make_ground_state_predictor,
        save_params_checkpoint as _save_params_checkpoint,
    )
    _log("[bootstrap] import td_graddft.training.predictors")
    from td_graddft.training.predictors import predict_ground_state_molecule as _predict_ground_state_molecule
    _log("[bootstrap] import td_graddft.training.targets")
    from td_graddft.training.targets import (
        predict_excitation_energies as _predict_excitation_energies,
        predict_ground_state_total_energy as _predict_ground_state_total_energy,
    )

    basis_from_spec = _basis_from_spec
    evaluate_cartesian_ao = _evaluate_cartesian_ao
    neural_xc = _neural_xc
    restricted_reference_from_pyscf = _restricted_reference_from_pyscf
    restricted_molecule_from_spec_with_jax_rks = _restricted_molecule_from_spec_with_jax_rks
    RKSConfig = _RKSConfig
    HARTREE_TO_EV = _HARTREE_TO_EV
    GroundStateCoreDatum = _GroundStateCoreDatum
    GroundStateCoreTrainingConfig = _GroundStateCoreTrainingConfig
    GroundStateDatum = _GroundStateDatum
    GroundStateTrainingConfig = _GroundStateTrainingConfig
    create_train_state_from_molecule = _create_train_state_from_molecule
    ground_state_mse_loss = _ground_state_mse_loss
    load_params_checkpoint = _load_params_checkpoint
    make_ground_state_train_step = _make_ground_state_train_step
    save_params_checkpoint = _save_params_checkpoint
    predict_ground_state_molecule = _predict_ground_state_molecule
    predict_excitation_energies = _predict_excitation_energies
    predict_ground_state_total_energy = _predict_ground_state_total_energy
    make_ground_state_predictor = _make_ground_state_predictor
    _RUNTIME_DEPENDENCIES_LOADED = True
    _log("[bootstrap] runtime dependency import complete")


@dataclass(frozen=True)
class ReferencePoint:
    r_angstrom: float
    atom: str
    molecule: Any
    fci_energy_h: float
    fci_total_energies_h: np.ndarray
    fci_excitation_energies_h: np.ndarray
    fci_dm_ao: np.ndarray
    fci_density_grid: np.ndarray
    fci_electron_count: float


def _normalize_input_feature_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode in {"dm21_original", "canonical"}:
        return "canonical"
    if mode == "enhanced":
        return "enhanced"
    raise ValueError(
        f"Unsupported input feature mode {value!r}. Expected enhanced, canonical, or dm21_original."
    )


def _normalize_scf_gradient_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode in {"implicit_commutator", "impl"}:
        return "impl"
    if mode in {"unrolled", "expl"}:
        return "expl"
    raise ValueError(
        f"Unsupported SCF gradient mode {value!r}. Expected impl, expl, implicit_commutator, or unrolled."
    )


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.input_feature_mode = _normalize_input_feature_mode(args.input_feature_mode)
    args.scf_gradient_mode = _normalize_scf_gradient_mode(args.scf_gradient_mode)
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train Neural_xc on 5 H2 FCI dissociation points in either fixed-density "
            "or self-consistent mode, then compare dense 100-point ground-state "
            "energies/densities and excited-state energies against FCI."
        )
    )
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--xc", default="b3lyp")
    p.add_argument("--r-min", type=float, default=0.05)
    p.add_argument("--r-max", type=float, default=5.0)
    p.add_argument("--train-points", type=int, default=5)
    p.add_argument("--dense-points", type=int, default=100)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional Flax msgpack checkpoint used to initialize the Neural_xc params.",
    )
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--lr-decay-every", type=int, default=100)
    p.add_argument("--lr-decay-factor", type=float, default=0.5)
    p.add_argument(
        "--training-mode",
        choices=("fixed_density", "self_consistent"),
        default="self_consistent",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=list(GRADDFT_DEFAULT_DM21_HIDDEN_DIMS),
    )
    p.add_argument(
        "--network-architecture",
        choices=("simple_mlp", "graddft_residual"),
        default=GRADDFT_DEFAULT_NETWORK_ARCHITECTURE,
    )
    p.add_argument(
        "--input-feature-mode",
        choices=("enhanced", "canonical", "dm21_original"),
        default=GRADDFT_DEFAULT_INPUT_FEATURE_MODE,
    )
    p.add_argument(
        "--include-pt2-channel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add a projected restricted MP2 local channel to the Neural_xc basis.",
    )
    p.add_argument(
        "--semilocal-xc",
        nargs="+",
        default=list(_DEFAULT_SEMILOCAL_XC),
        help="Neural_xc semilocal basis channels.",
    )
    p.add_argument("--energy-mse-weight", type=float, default=0.0)
    p.add_argument("--energy-mae-weight", type=float, default=1.0)
    p.add_argument(
        "--energy-normalization",
        choices=("none", "per_electron", "per_atom"),
        default="none",
    )
    p.add_argument(
        "--train-scf-max-cycle",
        type=int,
        default=0,
        help=(
            "SCF scan safety cap during training. Use 0 to let the energy "
            f"threshold decide convergence up to {_DEFAULT_TRAIN_SCF_SAFETY_MAX_CYCLE} cycles."
        ),
    )
    p.add_argument("--train-scf-damping", type=float, default=0.25)
    p.add_argument("--train-scf-conv-tol-energy", type=float, default=None)
    p.add_argument(
        "--train-scf-convergence-metric",
        choices=("energy_and_residual", "energy"),
        default="energy_and_residual",
    )
    p.add_argument("--train-scf-conv-tol-density", type=float, default=1e-8)
    p.add_argument("--train-scf-vxc-clip", type=float, default=20.0)
    p.add_argument(
        "--scf-iterate-selection",
        choices=("final", "best_rms", "first_converged"),
        default="best_rms",
    )
    p.add_argument(
        "--scf-gradient-mode",
        choices=("expl", "impl", "unrolled", "implicit_commutator"),
        default="impl",
    )
    p.add_argument(
        "--scf-require-convergence",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--reference-scf-max-cycle", type=int, default=80)
    p.add_argument("--reference-scf-conv-tol", type=float, default=1e-10)
    p.add_argument("--reference-scf-conv-tol-density", type=float, default=1e-8)
    p.add_argument("--reference-scf-damping", type=float, default=0.15)
    p.add_argument("--reference-scf-potential-clip", type=float, default=20.0)
    p.add_argument("--grids-level", type=int, default=0)
    p.add_argument("--max-l", type=int, default=3)
    p.add_argument("--grad-clip-norm", type=float, default=None)
    p.add_argument(
        "--grid-ao-backend",
        choices=("jax", "pyscf"),
        default="jax",
    )
    p.add_argument(
        "--integral-backend",
        choices=("jax", "cpu", "gpu", "libcint"),
        default="cpu",
    )
    p.add_argument(
        "--jk-backend",
        choices=("full", "df"),
        default="full",
    )
    p.add_argument(
        "--reference-scf-backend",
        choices=("pyscf", "jax_rks"),
        default="pyscf",
        help="SCF backend used to build the training/evaluation reference molecules.",
    )
    p.add_argument("--df-tol", type=float, default=1e-10)
    p.add_argument("--df-max-rank", type=int, default=None)
    p.add_argument(
        "--jit-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="JIT-compile the loss evaluation closure over the 5-point training set.",
    )
    p.add_argument(
        "--jit-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attempt to JIT the self-consistent train step. Enabled by default.",
    )
    p.add_argument(
        "--line-profile-points",
        type=int,
        default=401,
        help="Number of z-axis points in each selected density profile.",
    )
    p.add_argument(
        "--profile-r-values",
        type=float,
        nargs="+",
        default=[0.75, 1.50, 3.50],
        help="Representative bond lengths (Angstrom) used for 1D density-profile plots.",
    )
    p.add_argument(
        "--outdir",
        default="outputs/h2_fci_ground_train5_dense100",
    )
    p.add_argument("--excited-nstates", type=int, default=3)
    p.add_argument(
        "--density-constraint-weight",
        type=float,
        default=0.0,
        help="Optional weight for the self-consistent ground-state density matching loss.",
    )
    p.add_argument(
        "--density-matrix-constraint-weight",
        type=float,
        default=0.0,
        help="Optional weight for the self-consistent AO density-matrix matching loss.",
    )
    return _normalize_args(p.parse_args(argv))


def build_h2_atom(r_angstrom: float) -> str:
    return f"H 0 0 {-0.5 * r_angstrom:.12f}; H 0 0 {0.5 * r_angstrom:.12f}"


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"[{_timestamp()}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _get_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _metric_scalar(metrics: dict[str, Any], key: str, default: float = float("nan")) -> float:
    if key not in metrics:
        return default
    arr = jnp.asarray(metrics[key])
    if int(arr.size) <= 0:
        return default
    return float(arr.reshape(-1)[0])


def _tree_all_finite(tree: Any) -> bool:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return True
    return all(bool(jnp.all(jnp.isfinite(jnp.asarray(leaf)))) for leaf in leaves)


def solve_fci_singlet_states(
    atom: str,
    *,
    basis: str,
    nroots: int,
    dm0: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mol = gto.M(
        atom=atom,
        unit="Angstrom",
        basis=basis,
        spin=0,
        charge=0,
        cart=True,
        verbose=0,
    )
    def _run_rhf(
        *,
        dm0_local: np.ndarray | None,
        init_guess: str | None,
        damping: float,
        level_shift: float,
        max_cycle: int,
        use_newton: bool,
    ) -> scf.hf.RHF:
        mf_local = scf.RHF(mol)
        mf_local.conv_tol = 1e-12
        mf_local.max_cycle = int(max_cycle)
        mf_local.damping = float(damping)
        mf_local.level_shift = float(level_shift)
        mf_local.diis_start_cycle = 1
        if init_guess is not None:
            mf_local.init_guess = init_guess
        if use_newton:
            mf_local = mf_local.newton()
            mf_local.conv_tol = 1e-12
            mf_local.max_cycle = int(max_cycle)
        mf_local.kernel(dm0=dm0_local)
        return mf_local

    attempts = (
        dict(
            dm0_local=dm0,
            init_guess=None if dm0 is not None else "minao",
            damping=0.0,
            level_shift=0.0,
            max_cycle=100,
            use_newton=False,
        ),
        dict(
            dm0_local=None,
            init_guess="atom",
            damping=0.3,
            level_shift=0.5,
            max_cycle=200,
            use_newton=False,
        ),
        dict(
            dm0_local=None,
            init_guess="atom",
            damping=0.0,
            level_shift=0.0,
            max_cycle=50,
            use_newton=True,
        ),
    )

    mf: scf.hf.RHF | None = None
    for kwargs in attempts:
        mf = _run_rhf(**kwargs)
        if bool(mf.converged):
            break
    if mf is None or not mf.converged:
        raise RuntimeError(f"RHF did not converge for atom spec: {atom}")

    mo_coeff = np.asarray(mf.mo_coeff, dtype=np.float64)
    h1_mo = mo_coeff.T @ np.asarray(mf.get_hcore(), dtype=np.float64) @ mo_coeff
    eri_mo = ao2mo.kernel(mol, mo_coeff)
    cisolver = fci.direct_spin0.FCI(mol)
    root_count = max(1, int(nroots))
    e_ci, ci_vec = cisolver.kernel(
        h1_mo,
        eri_mo,
        h1_mo.shape[0],
        mol.nelectron,
        nroots=root_count,
    )
    e_roots = np.asarray(e_ci, dtype=np.float64).reshape(-1)
    ci_roots = ci_vec if isinstance(ci_vec, (list, tuple)) else [ci_vec]
    rdm1_mo = np.asarray(
        cisolver.make_rdm1(ci_roots[0], h1_mo.shape[0], mol.nelectron),
        dtype=np.float64,
    )
    dm_ao = mo_coeff @ rdm1_mo @ mo_coeff.T
    total_energies = np.asarray(e_roots + mol.energy_nuc(), dtype=np.float64)
    total_energy = float(total_energies[0])
    excitation_energies = e_roots[1:] - e_roots[0]
    return (
        total_energy,
        total_energies,
        excitation_energies,
        dm_ao,
        np.asarray(mf.make_rdm1(), dtype=np.float64),
    )


def build_reference_point(
    r_angstrom: float,
    *,
    basis: str,
    xc: str,
    grids_level: int,
    max_l: int,
    grid_ao_backend: str,
    integral_backend: str,
    jk_backend: str,
    df_tol: float,
    df_max_rank: int | None,
    reference_scf_max_cycle: int,
    reference_scf_conv_tol: float,
    reference_scf_conv_tol_density: float,
    reference_scf_damping: float,
    reference_scf_potential_clip: float,
    excited_nstates: int,
    fci_dm0: np.ndarray | None,
    compute_local_hfx_features: bool,
    compute_local_pt2_features: bool,
    reference_scf_backend: str = "pyscf",
) -> tuple[ReferencePoint, np.ndarray]:
    atom = build_h2_atom(r_angstrom)
    (
        fci_energy_h,
        fci_total_energies_h,
        fci_excitation_energies_h,
        fci_dm_ao,
        rhf_dm0,
    ) = solve_fci_singlet_states(
        atom,
        basis=basis,
        nroots=max(1, int(excited_nstates) + 1),
        dm0=fci_dm0,
    )

    reference_backend = str(reference_scf_backend).strip().lower()
    if reference_backend == "jax_rks":
        if str(grid_ao_backend) != "jax":
            raise ValueError("JAX reference building requires --grid-ao-backend jax.")
        reference = restricted_molecule_from_spec_with_jax_rks(
            atom=atom,
            basis=basis,
            xc_spec=xc,
            unit="Angstrom",
            spin=0,
            charge=0,
            cart=True,
            grids_level=int(grids_level),
            max_l=int(max_l),
            rks_config=RKSConfig(
                xc_spec=xc,
                max_cycle=int(reference_scf_max_cycle),
                conv_tol=float(reference_scf_conv_tol),
                conv_tol_density=float(reference_scf_conv_tol_density),
                damping=float(reference_scf_damping),
                potential_clip=float(reference_scf_potential_clip),
                jk_backend=str(jk_backend),
                df_tol=float(df_tol),
                df_max_rank=df_max_rank,
            ),
            grid_ao_backend="jax",
            integral_backend=str(integral_backend),
            compute_local_hfx_features=bool(compute_local_hfx_features),
            compute_local_hfx_aux=bool(compute_local_hfx_features),
            compute_local_pt2_features=bool(compute_local_pt2_features),
            verbose=0,
        )
    elif reference_backend == "pyscf":
        mol = gto.M(
            atom=atom,
            basis=basis,
            unit="Angstrom",
            spin=0,
            charge=0,
            cart=True,
            verbose=0,
        )

        def _run_rks(
            *,
            dm0_local: np.ndarray | None,
            init_guess: str | None,
            damping: float,
            level_shift: float,
            max_cycle: int,
            use_newton: bool,
        ) -> dft.rks.RKS:
            mf_local = dft.RKS(mol)
            mf_local.xc = xc
            mf_local.grids.level = int(grids_level)
            mf_local.conv_tol = float(reference_scf_conv_tol)
            mf_local.max_cycle = int(max_cycle)
            mf_local.damping = float(damping)
            mf_local.level_shift = float(level_shift)
            mf_local.diis_start_cycle = 1
            if init_guess is not None:
                mf_local.init_guess = init_guess
            if use_newton:
                mf_local = mf_local.newton()
                mf_local.conv_tol = float(reference_scf_conv_tol)
                mf_local.max_cycle = int(max_cycle)
            mf_local.kernel(dm0=dm0_local)
            return mf_local

        rks_attempts = (
            dict(
                dm0_local=rhf_dm0,
                init_guess=None if rhf_dm0 is not None else "minao",
                damping=float(reference_scf_damping),
                level_shift=0.0,
                max_cycle=int(reference_scf_max_cycle),
                use_newton=False,
            ),
            dict(
                dm0_local=None,
                init_guess="atom",
                damping=max(float(reference_scf_damping), 0.3),
                level_shift=0.5,
                max_cycle=max(int(reference_scf_max_cycle), 200),
                use_newton=False,
            ),
            dict(
                dm0_local=None,
                init_guess="atom",
                damping=0.0,
                level_shift=0.0,
                max_cycle=max(int(reference_scf_max_cycle), 80),
                use_newton=True,
            ),
        )

        mf_ref: dft.rks.RKS | None = None
        for kwargs in rks_attempts:
            mf_ref = _run_rks(**kwargs)
            if bool(mf_ref.converged):
                break
        if mf_ref is None or not mf_ref.converged:
            raise RuntimeError(f"PySCF RKS did not converge for atom spec: {atom}")

        reference = restricted_reference_from_pyscf(
            mf_ref,
            compute_local_hfx_features=bool(compute_local_hfx_features),
            compute_local_hfx_aux=bool(compute_local_hfx_features),
            compute_local_pt2_features=bool(compute_local_pt2_features),
        )
    else:
        raise ValueError(
            f"Unsupported reference_scf_backend={reference_scf_backend!r}; "
            "expected 'pyscf' or 'jax_rks'."
        )
    ao = np.asarray(reference.ao, dtype=np.float64)
    weights = np.asarray(reference.grid.weights, dtype=np.float64)
    fci_density_grid = np.einsum("pq,rp,rq->r", fci_dm_ao, ao, ao, optimize=True)
    fci_electron_count = float(np.dot(weights, fci_density_grid))
    point = ReferencePoint(
        r_angstrom=float(r_angstrom),
        atom=atom,
        molecule=reference,
        fci_energy_h=float(fci_energy_h),
        fci_total_energies_h=fci_total_energies_h,
        fci_excitation_energies_h=fci_excitation_energies_h,
        fci_dm_ao=fci_dm_ao,
        fci_density_grid=fci_density_grid,
        fci_electron_count=fci_electron_count,
    )
    return point, rhf_dm0


def build_reference_curve(
    r_values: np.ndarray,
    *,
    args: argparse.Namespace,
    logger: RunLogger,
    label: str,
) -> list[ReferencePoint]:
    points: list[ReferencePoint] = []
    rhf_dm0 = None
    t0 = time.perf_counter()
    for idx, r_val in enumerate(r_values, start=1):
        point, rhf_dm0 = build_reference_point(
            float(r_val),
            basis=str(args.basis),
            xc=str(args.xc),
            grids_level=int(args.grids_level),
            max_l=int(args.max_l),
            grid_ao_backend=str(args.grid_ao_backend),
            integral_backend=str(args.integral_backend),
            jk_backend=str(args.jk_backend),
            df_tol=float(args.df_tol),
            df_max_rank=args.df_max_rank,
            reference_scf_max_cycle=int(args.reference_scf_max_cycle),
            reference_scf_conv_tol=float(args.reference_scf_conv_tol),
            reference_scf_conv_tol_density=float(args.reference_scf_conv_tol_density),
            reference_scf_damping=float(args.reference_scf_damping),
            reference_scf_potential_clip=float(args.reference_scf_potential_clip),
            excited_nstates=int(args.excited_nstates),
            fci_dm0=rhf_dm0,
            compute_local_hfx_features=(str(args.input_feature_mode) == "canonical"),
            compute_local_pt2_features=bool(getattr(args, "include_pt2_channel", False)),
            reference_scf_backend=str(getattr(args, "reference_scf_backend", "pyscf")),
        )
        fci_s1_h = (
            float(point.fci_excitation_energies_h[0])
            if point.fci_excitation_energies_h.size > 0
            else float("nan")
        )
        points.append(point)
        logger.log(
            f"[{label}] {idx:3d}/{len(r_values):3d} "
            f"R={point.r_angstrom:.4f} A "
            f"E0_FCI={point.fci_energy_h:.10f} Eh "
            f"S1_FCI={fci_s1_h:.10f} Eh "
            f"grid_n={int(np.asarray(point.molecule.grid.weights).size)}"
        )
    logger.log(f"[{label}] done in {time.perf_counter() - t0:.2f} s")
    return points


def build_training_data(
    points: list[ReferencePoint],
    *,
    density_constraint_weight: float,
    density_matrix_constraint_weight: float,
) -> tuple[GroundStateDatum, ...]:
    return tuple(
        GroundStateDatum.from_parts(
            point.molecule,
            core=GroundStateCoreDatum(
                target_total_energy=jnp.asarray(point.fci_energy_h, dtype=jnp.float64),
                target_density_matrix=jnp.asarray(point.fci_dm_ao, dtype=jnp.float64),
                density_constraint_weight=float(density_constraint_weight),
                density_matrix_constraint_weight=float(density_matrix_constraint_weight),
            ),
        )
        for point in points
    )


def train_functional(
    train_points: list[ReferencePoint],
    *,
    args: argparse.Namespace,
    logger: RunLogger,
):
    if not train_points:
        raise ValueError("train_points must not be empty.")

    training_data = build_training_data(
        train_points,
        density_constraint_weight=float(args.density_constraint_weight),
        density_matrix_constraint_weight=float(args.density_matrix_constraint_weight),
    )
    functional = neural_xc.Functional(
        semilocal_xc=tuple(str(name) for name in args.semilocal_xc),
        hidden_dims=tuple(int(value) for value in args.hidden_dims),
        architecture=str(args.network_architecture),
        input_feature_mode=str(args.input_feature_mode),
        include_pt2_channel=bool(args.include_pt2_channel),
        name=f"neural_xc_h2_fci_{str(args.training_mode)}",
    )
    coefficient_prior = neural_xc.resolve_coefficient_prior_values(
        tuple(str(name) for name in args.semilocal_xc)
    )
    if coefficient_prior is not None and bool(args.include_pt2_channel):
        n_semilocal = len(tuple(str(name) for name in args.semilocal_xc))
        if len(coefficient_prior) == n_semilocal + 1:
            coefficient_prior = (
                tuple(coefficient_prior[:n_semilocal])
                + (0.0,)
                + tuple(coefficient_prior[n_semilocal:])
            )
    logger.log(
        "[init] coefficient_prior="
        f"{None if coefficient_prior is None else tuple(float(x) for x in coefficient_prior)} "
        f"include_pt2_channel={bool(args.include_pt2_channel)}"
    )
    gs_training = GroundStateTrainingConfig.from_parts(
        core=GroundStateCoreTrainingConfig(
            mode=str(args.training_mode),
            energy_mse_weight=float(args.energy_mse_weight),
            energy_mae_weight=float(args.energy_mae_weight),
            energy_normalization=str(args.energy_normalization),
            scf_max_cycle=_resolve_train_scf_max_cycle(args.train_scf_max_cycle),
            scf_damping=float(args.train_scf_damping),
            scf_conv_tol_energy=args.train_scf_conv_tol_energy,
            scf_convergence_metric=str(args.train_scf_convergence_metric),
            scf_conv_tol_density=float(args.train_scf_conv_tol_density),
            scf_vxc_clip=float(args.train_scf_vxc_clip),
            scf_iterate_selection=str(args.scf_iterate_selection),
            scf_require_convergence=bool(args.scf_require_convergence),
            scf_gradient_mode=str(args.scf_gradient_mode),
        ),
    )
    if int(args.lr_decay_every) > 0:
        lr_schedule = optax.exponential_decay(
            init_value=float(args.learning_rate),
            transition_steps=int(args.lr_decay_every),
            decay_rate=float(args.lr_decay_factor),
            staircase=True,
        )
        base_optimizer = optax.adam(lr_schedule)
    else:
        lr_schedule = None
        base_optimizer = optax.adam(float(args.learning_rate))

    if args.grad_clip_norm is not None and float(args.grad_clip_norm) > 0.0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(float(args.grad_clip_norm)),
            base_optimizer,
        )
    else:
        optimizer = base_optimizer

    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(int(args.seed)),
        train_points[0].molecule,
        optimizer,
    )
    if args.init_checkpoint:
        init_checkpoint = Path(str(args.init_checkpoint))
        state = state.replace(
            params=load_params_checkpoint(init_checkpoint, template=state.params)
        )
        logger.log(f"[train_init] loaded params from checkpoint: {init_checkpoint}")
    train_step = make_ground_state_train_step(functional, training_config=gs_training)
    eval_fn = lambda params: ground_state_mse_loss(  # noqa: E731
        params,
        functional,
        training_data,
        training_config=gs_training,
    )
    eager_train_step = lambda current_state: train_step(current_state, training_data)  # noqa: E731
    compiled_eval = jax.jit(eval_fn) if bool(args.jit_eval) else eval_fn
    compiled_train_step = eager_train_step
    train_step_mode = "eager"
    if bool(args.jit_train):
        candidate_train_step = jax.jit(eager_train_step)
        try:
            _ = candidate_train_step.lower(state).compile()
            compiled_train_step = candidate_train_step
            train_step_mode = "jit"
        except Exception as exc:  # pragma: no cover - best effort runtime path
            logger.log(f"[train] jit compilation failed for self-consistent train step: {exc!r}")

    initial_loss, initial_metrics = compiled_eval(state.params)
    initial_loss_val = float(initial_loss)
    min_loss = initial_loss_val
    min_loss_step = 0
    best_params = state.params
    initial_scf_converged_fraction = _metric_scalar(initial_metrics, "scf_converged_fraction")
    initial_scf_cycles_mean = _metric_scalar(initial_metrics, "scf_cycles_mean")
    initial_scf_cycles_max = _metric_scalar(initial_metrics, "scf_cycles_max")
    initial_scf_selected_rms_max = _metric_scalar(initial_metrics, "scf_selected_rms_max")
    initial_scf_final_rms_max = _metric_scalar(initial_metrics, "scf_final_rms_max")
    loss_history = [initial_loss_val]
    density_penalty_history = [float(initial_metrics["density_penalty"][0])]
    density_matrix_penalty_history = [float(initial_metrics["density_matrix_penalty"][0])]
    stationarity_penalty_history = [float(initial_metrics["stationarity_penalty"][0])]
    coefficient_prior_penalty_history = [float(initial_metrics["coefficient_prior_penalty"][0])]
    grad_norm_history = [float("nan")]
    grad_abs_max_history = [float("nan")]
    param_update_norm_history = [float("nan")]
    nonfinite_grad_fraction_history = [0.0]

    logger.log(
        "[train] "
        f"steps={int(args.steps)} "
        f"lr={float(args.learning_rate):.6g} "
        f"mode={str(args.training_mode)} "
        f"scf_require_convergence={bool(args.scf_require_convergence)} "
        f"scf_grad_mode={args.scf_gradient_mode} "
        f"train_step_mode={train_step_mode}"
    )

    t0 = time.perf_counter()
    for step in range(1, int(args.steps) + 1):
        prev_state = state
        state, train_metrics = compiled_train_step(state)
        if not _tree_all_finite(state.params):
            state = prev_state
            logger.log(f"[train] non-finite params detected at step {step}; reverted update")
        grad_norm_val = _metric_scalar(train_metrics, "grad_norm")
        grad_abs_max_val = _metric_scalar(train_metrics, "grad_abs_max")
        param_update_norm_val = _metric_scalar(train_metrics, "param_update_norm")
        nonfinite_grad_fraction_val = _metric_scalar(train_metrics, "nonfinite_grad_fraction", 0.0)
        train_loss_val = _metric_scalar(train_metrics, "loss")
        train_density_penalty_val = _metric_scalar(train_metrics, "density_penalty", 0.0)
        train_density_matrix_penalty_val = _metric_scalar(
            train_metrics,
            "density_matrix_penalty",
            0.0,
        )
        train_stationarity_penalty_val = _metric_scalar(train_metrics, "stationarity_penalty", 0.0)
        train_coefficient_prior_penalty_val = _metric_scalar(
            train_metrics,
            "coefficient_prior_penalty",
            0.0,
        )

        grad_norm_history.append(grad_norm_val)
        grad_abs_max_history.append(grad_abs_max_val)
        param_update_norm_history.append(param_update_norm_val)
        nonfinite_grad_fraction_history.append(nonfinite_grad_fraction_val)

        if step >= 2:
            tracked_step = step - 1
            loss_history.append(train_loss_val)
            density_penalty_history.append(train_density_penalty_val)
            density_matrix_penalty_history.append(train_density_matrix_penalty_val)
            stationarity_penalty_history.append(train_stationarity_penalty_val)
            coefficient_prior_penalty_history.append(train_coefficient_prior_penalty_val)
            if train_loss_val < min_loss:
                min_loss = train_loss_val
                min_loss_step = tracked_step
                best_params = prev_state.params

        if step == 1 or step % 10 == 0 or step == int(args.steps):
            current_lr = float(lr_schedule(step - 1)) if lr_schedule is not None else float(args.learning_rate)
            scf_converged_fraction_val = _metric_scalar(train_metrics, "scf_converged_fraction")
            scf_cycles_mean_val = _metric_scalar(train_metrics, "scf_cycles_mean")
            scf_cycles_max_val = _metric_scalar(train_metrics, "scf_cycles_max")
            scf_selected_rms_max_val = _metric_scalar(train_metrics, "scf_selected_rms_max")
            logger.log(
                "[train] "
                f"step={step:4d}/{int(args.steps):4d} "
                f"loss={train_loss_val:.8e} "
                f"energy_mae={float(train_metrics['energy_mae'][0]):.8e} "
                f"density_penalty={float(train_metrics['density_penalty'][0]):.8e} "
                f"dm_penalty={float(train_metrics['density_matrix_penalty'][0]):.8e} "
                f"scf_conv_frac={scf_converged_fraction_val:.6f} "
                f"scf_cycles_mean={scf_cycles_mean_val:.6f} "
                f"scf_cycles_max={scf_cycles_max_val:.6f} "
                f"scf_selected_rms_max={scf_selected_rms_max_val:.8e} "
                f"grad_norm={grad_norm_val:.8e} "
                f"grad_abs_max={grad_abs_max_val:.8e} "
                f"update_norm={param_update_norm_val:.8e} "
                f"lr={current_lr:.8e}"
            )

    elapsed_s = time.perf_counter() - t0
    final_loss, final_metrics = compiled_eval(state.params)
    final_loss_val = float(final_loss)
    final_scf_converged_fraction = _metric_scalar(final_metrics, "scf_converged_fraction")
    final_scf_cycles_mean = _metric_scalar(final_metrics, "scf_cycles_mean")
    final_scf_cycles_max = _metric_scalar(final_metrics, "scf_cycles_max")
    final_scf_selected_rms_max = _metric_scalar(final_metrics, "scf_selected_rms_max")
    final_scf_final_rms_max = _metric_scalar(final_metrics, "scf_final_rms_max")
    loss_history.append(final_loss_val)
    density_penalty_history.append(float(final_metrics["density_penalty"][0]))
    density_matrix_penalty_history.append(float(final_metrics["density_matrix_penalty"][0]))
    stationarity_penalty_history.append(float(final_metrics["stationarity_penalty"][0]))
    coefficient_prior_penalty_history.append(float(final_metrics["coefficient_prior_penalty"][0]))
    if final_loss_val < min_loss:
        min_loss = final_loss_val
        min_loss_step = int(args.steps)
        best_params = state.params

    logger.log(
        "[train] done "
        f"final_loss={final_loss_val:.8e} "
        f"min_loss={min_loss:.8e}@{min_loss_step} "
        f"elapsed_s={elapsed_s:.2f}"
    )

    return {
        "functional": functional,
        "training_config": gs_training,
        "best_params": best_params,
        "final_loss": final_loss_val,
        "min_loss": min_loss,
        "min_loss_step": min_loss_step,
        "elapsed_s": elapsed_s,
        "initial_scf_converged_fraction": initial_scf_converged_fraction,
        "final_scf_converged_fraction": final_scf_converged_fraction,
        "initial_scf_cycles_mean": initial_scf_cycles_mean,
        "final_scf_cycles_mean": final_scf_cycles_mean,
        "initial_scf_cycles_max": initial_scf_cycles_max,
        "final_scf_cycles_max": final_scf_cycles_max,
        "initial_scf_selected_rms_max": initial_scf_selected_rms_max,
        "final_scf_selected_rms_max": final_scf_selected_rms_max,
        "initial_scf_final_rms_max": initial_scf_final_rms_max,
        "final_scf_final_rms_max": final_scf_final_rms_max,
        "loss_history": loss_history,
        "density_penalty_history": density_penalty_history,
        "density_matrix_penalty_history": density_matrix_penalty_history,
        "stationarity_penalty_history": stationarity_penalty_history,
        "coefficient_prior_penalty_history": coefficient_prior_penalty_history,
        "grad_norm_history": grad_norm_history,
        "grad_abs_max_history": grad_abs_max_history,
        "param_update_norm_history": param_update_norm_history,
        "nonfinite_grad_fraction_history": nonfinite_grad_fraction_history,
    }


def _density_error_metrics(
    weights: np.ndarray,
    predicted_density: np.ndarray,
    reference_density: np.ndarray,
) -> tuple[float, float, float]:
    diff = predicted_density - reference_density
    l1 = float(np.dot(weights, np.abs(diff)))
    l2 = float(np.sqrt(np.dot(weights, diff * diff)))
    linf = float(np.max(np.abs(diff)))
    return l1, l2, linf


def evaluate_dense_curve(
    dense_points: list[ReferencePoint],
    *,
    functional: Any,
    params: Any,
    training_config: GroundStateTrainingConfig,
    excited_nstates: int,
    logger: RunLogger,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    excited_rows: list[dict[str, float]] = []
    predictor = make_ground_state_predictor(
        functional,
        training_config=training_config,
    )
    t0 = time.perf_counter()
    for idx, point in enumerate(dense_points, start=1):
        predicted_energy_h_arr, predicted_molecule = predictor(params, point.molecule)
        predicted_energy_h = float(predicted_energy_h_arr)
        predicted_density = np.asarray(predicted_molecule.density(), dtype=np.float64).sum(axis=-1)
        weights = np.asarray(point.molecule.grid.weights, dtype=np.float64)
        predicted_dm_total = np.asarray(predicted_molecule.rdm1, dtype=np.float64).sum(axis=0)
        predicted_electron_count = float(np.dot(weights, predicted_density))
        density_l1, density_l2, density_linf = _density_error_metrics(
            weights,
            predicted_density,
            point.fci_density_grid,
        )
        if int(excited_nstates) > 0:
            predicted_tda = np.asarray(
                predict_excitation_energies(
                    params,
                    functional,
                    predicted_molecule,
                    nstates=int(excited_nstates),
                    use_tda=True,
                ),
                dtype=np.float64,
            )
            predicted_casida = np.asarray(
                predict_excitation_energies(
                    params,
                    functional,
                    predicted_molecule,
                    nstates=int(excited_nstates),
                    use_tda=False,
                ),
                dtype=np.float64,
            )
        else:
            predicted_tda = np.asarray([], dtype=np.float64)
            predicted_casida = np.asarray([], dtype=np.float64)
        ncompare_tda = min(
            int(excited_nstates),
            int(point.fci_excitation_energies_h.size),
            int(predicted_tda.size),
        )
        ncompare_casida = min(
            int(excited_nstates),
            int(point.fci_excitation_energies_h.size),
            int(predicted_casida.size),
        )
        rows.append(
            {
                "r_angstrom": float(point.r_angstrom),
                "fci_energy_h": float(point.fci_energy_h),
                "predicted_energy_h": predicted_energy_h,
                "energy_abs_err_ev": abs(predicted_energy_h - point.fci_energy_h) * HARTREE_TO_EV,
                "fci_electron_count": float(point.fci_electron_count),
                "predicted_electron_count": predicted_electron_count,
                "electron_count_abs_err": abs(predicted_electron_count - point.fci_electron_count),
                "density_l1": density_l1,
                "density_l2": density_l2,
                "density_linf": density_linf,
                "predicted_dm_trace": float(np.trace(predicted_dm_total)),
            }
        )
        for state_idx in range(ncompare_tda):
            fci_gap = float(point.fci_excitation_energies_h[state_idx])
            pred_gap = float(predicted_tda[state_idx])
            fci_total = float(point.fci_total_energies_h[state_idx + 1])
            pred_total = float(predicted_energy_h + pred_gap)
            excited_rows.append(
                {
                    "r_angstrom": float(point.r_angstrom),
                    "solver": "tda",
                    "state_index": int(state_idx + 1),
                    "fci_total_energy_h": fci_total,
                    "predicted_total_energy_h": pred_total,
                    "total_abs_err_ev": abs(pred_total - fci_total) * HARTREE_TO_EV,
                    "fci_excitation_h": fci_gap,
                    "predicted_excitation_h": pred_gap,
                    "gap_abs_err_ev": abs(pred_gap - fci_gap) * HARTREE_TO_EV,
                }
            )
        for state_idx in range(ncompare_casida):
            fci_gap = float(point.fci_excitation_energies_h[state_idx])
            pred_gap = float(predicted_casida[state_idx])
            fci_total = float(point.fci_total_energies_h[state_idx + 1])
            pred_total = float(predicted_energy_h + pred_gap)
            excited_rows.append(
                {
                    "r_angstrom": float(point.r_angstrom),
                    "solver": "casida",
                    "state_index": int(state_idx + 1),
                    "fci_total_energy_h": fci_total,
                    "predicted_total_energy_h": pred_total,
                    "total_abs_err_ev": abs(pred_total - fci_total) * HARTREE_TO_EV,
                    "fci_excitation_h": fci_gap,
                    "predicted_excitation_h": pred_gap,
                    "gap_abs_err_ev": abs(pred_gap - fci_gap) * HARTREE_TO_EV,
                }
            )
        tda_s1_err = float("nan")
        casida_s1_err = float("nan")
        if ncompare_tda > 0:
            tda_s1_err = abs(
                float(predicted_tda[0]) - float(point.fci_excitation_energies_h[0])
            ) * HARTREE_TO_EV
        if ncompare_casida > 0:
            casida_s1_err = abs(
                float(predicted_casida[0]) - float(point.fci_excitation_energies_h[0])
            ) * HARTREE_TO_EV
        if idx == 1 or idx % max(1, len(dense_points) // 10) == 0 or idx == len(dense_points):
            logger.log(
                f"[eval] {idx:3d}/{len(dense_points):3d} "
                f"R={point.r_angstrom:.4f} A "
                f"E0_pred={predicted_energy_h:.10f} Eh "
                f"density_l2={density_l2:.6e} "
                f"S1_TDA_gap_err={tda_s1_err:.6e} eV "
                f"S1_Casida_gap_err={casida_s1_err:.6e} eV"
            )
    logger.log(f"[eval] done in {time.perf_counter() - t0:.2f} s")
    return rows, excited_rows


def write_dense_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        raise ValueError("rows must not be empty.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _selected_profile_indices(
    dense_points: list[ReferencePoint],
    requested_r_values: list[float],
) -> list[int]:
    dense_r = np.asarray([point.r_angstrom for point in dense_points], dtype=np.float64)
    indices: list[int] = []
    for requested in requested_r_values:
        idx = int(np.argmin(np.abs(dense_r - requested)))
        if idx not in indices:
            indices.append(idx)
    return indices


def _line_density_from_dm(
    atom: str,
    basis: str,
    dm_ao: np.ndarray,
    *,
    max_l: int,
    npoints: int,
    padding_angstrom: float = 1.50,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(
        [
            xyz
            for xyz in (
                (-0.0, 0.0, z)
                for z in np.linspace(
                    float(atom.split(";")[0].split()[-1]) - padding_angstrom,
                    float(atom.split(";")[1].split()[-1]) + padding_angstrom,
                    int(npoints),
                )
            )
        ],
        dtype=np.float64,
    )
    basis_cart = basis_from_spec(
        atom,
        basis=basis,
        unit="Angstrom",
        charge=0,
        spin=0,
        max_l=max_l,
    )
    ao = np.asarray(evaluate_cartesian_ao(basis_cart, jnp.asarray(coords), deriv=0), dtype=np.float64)
    density = np.einsum("pq,zp,zq->z", dm_ao, ao, ao, optimize=True)
    return coords[:, 2], density


def plot_curve_summary(
    path: Path,
    rows: list[dict[str, float]],
    *,
    train_r_values: np.ndarray,
    basis: str,
    xc: str,
    training_mode: str,
) -> None:
    plt = _get_plt()
    r = np.asarray([row["r_angstrom"] for row in rows], dtype=np.float64)
    fci_energy = np.asarray([row["fci_energy_h"] for row in rows], dtype=np.float64)
    pred_energy = np.asarray([row["predicted_energy_h"] for row in rows], dtype=np.float64)
    energy_err_ev = np.asarray([row["energy_abs_err_ev"] for row in rows], dtype=np.float64)
    density_l1 = np.asarray([row["density_l1"] for row in rows], dtype=np.float64)
    density_l2 = np.asarray([row["density_l2"] for row in rows], dtype=np.float64)
    density_linf = np.asarray([row["density_linf"] for row in rows], dtype=np.float64)
    electron_count_err = np.asarray([row["electron_count_abs_err"] for row in rows], dtype=np.float64)
    dense_mask = r >= max(0.40, float(r.min()))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))

    ax = axes[0, 0]
    ax.plot(r, fci_energy, lw=2.0, label="FCI ground")
    ax.plot(r, pred_energy, lw=2.0, label=f"Neural_xc {training_mode}")
    ax.scatter(
        train_r_values,
        np.interp(train_r_values, r, fci_energy),
        s=36,
        c="black",
        marker="o",
        label="5 training points",
        zorder=5,
    )
    ax.set_xlabel("H-H distance (Angstrom)")
    ax.set_ylabel("Total energy (Hartree)")
    ax.set_title("Ground-State Curve")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    ax.plot(r[dense_mask], fci_energy[dense_mask], lw=2.0, label="FCI ground")
    ax.plot(r[dense_mask], pred_energy[dense_mask], lw=2.0, label=f"Neural_xc {training_mode}")
    ax.scatter(
        train_r_values,
        np.interp(train_r_values, r, fci_energy),
        s=36,
        c="black",
        marker="o",
        zorder=5,
    )
    ax.set_xlabel("H-H distance (Angstrom)")
    ax.set_ylabel("Total energy (Hartree)")
    ax.set_title("Ground-State Curve (Zoom)")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(r, np.maximum(energy_err_ev, 1e-16), lw=1.9, label="Energy abs. err. (eV)")
    ax.plot(r, np.maximum(density_l2, 1e-16), lw=1.9, label="Density L2")
    ax.set_xlabel("H-H distance (Angstrom)")
    ax.set_ylabel("Error")
    ax.set_yscale("log")
    ax.set_title("Energy / Density Error")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    ax.plot(r, np.maximum(density_l1, 1e-16), lw=1.8, label="Density L1")
    ax.plot(r, np.maximum(density_linf, 1e-16), lw=1.8, label="Density Linf")
    ax.plot(r, np.maximum(electron_count_err, 1e-16), lw=1.8, label="Electron-count abs. err.")
    ax.set_xlabel("H-H distance (Angstrom)")
    ax.set_ylabel("Density metric")
    ax.set_yscale("log")
    ax.set_title("Ground-State Density Metrics")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle(f"H2 {training_mode} Neural_xc vs FCI | {xc}/{basis}", y=0.985)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_selected_density_profiles(
    path: Path,
    dense_points: list[ReferencePoint],
    rows: list[dict[str, float]],
    *,
    functional: Any,
    params: Any,
    training_config: GroundStateTrainingConfig,
    basis: str,
    max_l: int,
    profile_r_values: list[float],
    line_profile_points: int,
) -> list[dict[str, float]]:
    plt = _get_plt()
    selected_indices = _selected_profile_indices(dense_points, profile_r_values)
    profile_rows: list[dict[str, float]] = []
    fig, axes = plt.subplots(1, len(selected_indices), figsize=(4.4 * len(selected_indices), 3.8))
    if len(selected_indices) == 1:
        axes = [axes]

    for profile_index, (ax, idx) in enumerate(zip(axes, selected_indices, strict=True)):
        point = dense_points[idx]
        predicted_molecule = predict_ground_state_molecule(
            params,
            functional,
            point.molecule,
            training_config=training_config,
        )
        predicted_dm_ao = np.asarray(predicted_molecule.rdm1, dtype=np.float64).sum(axis=0)
        z_line, fci_density_line = _line_density_from_dm(
            point.atom,
            basis,
            point.fci_dm_ao,
            max_l=max_l,
            npoints=line_profile_points,
        )
        _, pred_density_line = _line_density_from_dm(
            point.atom,
            basis,
            predicted_dm_ao,
            max_l=max_l,
            npoints=line_profile_points,
        )
        density_diff = pred_density_line - fci_density_line
        row = rows[idx]
        for sample_index, (z_value, fci_value, pred_value, diff_value) in enumerate(
            zip(z_line, fci_density_line, pred_density_line, density_diff, strict=True)
        ):
            profile_rows.append(
                {
                    "profile_index": float(profile_index),
                    "sample_index": float(sample_index),
                    "r_angstrom": float(point.r_angstrom),
                    "z_angstrom": float(z_value),
                    "fci_density": float(fci_value),
                    "predicted_density": float(pred_value),
                    "density_error": float(diff_value),
                    "density_abs_error": float(abs(diff_value)),
                    "density_l2": float(row["density_l2"]),
                }
            )
        ax.plot(z_line, fci_density_line, lw=2.0, label="FCI")
        ax.plot(z_line, pred_density_line, lw=2.0, label="Neural_xc")
        ax.set_xlabel("z (Angstrom)")
        ax.set_ylabel(r"$\rho(z)$")
        ax.set_title(
            f"R={point.r_angstrom:.2f} A\n"
            f"L2={row['density_l2']:.2e}"
        )
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Selected ground-state density profiles along the bond axis", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return profile_rows


def _solver_state_table(
    excited_rows: list[dict[str, float]],
    solver: str,
    max_states: int,
) -> tuple[np.ndarray, dict[int, dict[str, np.ndarray]]]:
    filtered = [row for row in excited_rows if row["solver"] == solver]
    if not filtered:
        return np.asarray([], dtype=np.float64), {}
    r_values = np.asarray(sorted({float(row["r_angstrom"]) for row in filtered}), dtype=np.float64)
    tables: dict[int, dict[str, np.ndarray]] = {}
    for state_idx in range(1, int(max_states) + 1):
        state_rows = [row for row in filtered if int(row["state_index"]) == state_idx]
        if not state_rows:
            continue
        state_rows = sorted(state_rows, key=lambda row: float(row["r_angstrom"]))
        tables[state_idx] = {
            "r": np.asarray([row["r_angstrom"] for row in state_rows], dtype=np.float64),
            "fci_total": np.asarray([row["fci_total_energy_h"] for row in state_rows], dtype=np.float64),
            "pred_total": np.asarray(
                [row["predicted_total_energy_h"] for row in state_rows],
                dtype=np.float64,
            ),
            "total_err_ev": np.asarray([row["total_abs_err_ev"] for row in state_rows], dtype=np.float64),
            "fci_gap": np.asarray([row["fci_excitation_h"] for row in state_rows], dtype=np.float64),
            "pred_gap": np.asarray([row["predicted_excitation_h"] for row in state_rows], dtype=np.float64),
            "gap_err_ev": np.asarray([row["gap_abs_err_ev"] for row in state_rows], dtype=np.float64),
        }
    return r_values, tables


def plot_excited_state_summary(
    path: Path,
    excited_rows: list[dict[str, float]],
    *,
    max_states: int,
    basis: str,
    xc: str,
) -> None:
    plt = _get_plt()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    solver_specs = [("tda", "TDA"), ("casida", "Casida")]

    for row_idx, (solver_key, solver_label) in enumerate(solver_specs):
        _, tables = _solver_state_table(excited_rows, solver_key, max_states)
        ax_curve = axes[row_idx, 0]
        ax_err = axes[row_idx, 1]
        for state_idx, table in tables.items():
            ax_curve.plot(
                table["r"],
                table["fci_total"],
                lw=1.8,
                label=f"FCI S{state_idx}",
            )
            ax_curve.plot(
                table["r"],
                table["pred_total"],
                lw=1.8,
                ls="--",
                label=f"{solver_label} S{state_idx}",
            )
            ax_err.plot(
                table["r"],
                np.maximum(table["total_err_ev"], 1e-16),
                lw=1.8,
                label=f"S{state_idx}",
            )
        ax_curve.set_xlabel("H-H distance (Angstrom)")
        ax_curve.set_ylabel("Excited-state total energy (Hartree)")
        ax_curve.set_title(f"{solver_label} vs FCI excited-state energies")
        ax_curve.grid(alpha=0.25)
        ax_curve.legend(frameon=False, fontsize=8, ncol=2)

        ax_err.set_xlabel("H-H distance (Angstrom)")
        ax_err.set_ylabel("Total-energy abs. error (eV)")
        ax_err.set_yscale("log")
        ax_err.set_title(f"{solver_label} excited-energy absolute error")
        ax_err.grid(alpha=0.25)
        ax_err.legend(frameon=False, fontsize=8)

    fig.suptitle(f"H2 excited-state inference vs FCI | {xc}/{basis}", y=0.985)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    train_r_values: np.ndarray,
    dense_rows: list[dict[str, float]],
    excited_rows: list[dict[str, float]],
    training: dict[str, Any],
    final_loss: float,
    min_loss: float,
    min_loss_step: int,
    train_elapsed_s: float,
    checkpoint_path: Path,
    checkpoint_meta_path: Path | None,
) -> None:
    energy_err_ev = np.asarray([row["energy_abs_err_ev"] for row in dense_rows], dtype=np.float64)
    density_l1 = np.asarray([row["density_l1"] for row in dense_rows], dtype=np.float64)
    density_l2 = np.asarray([row["density_l2"] for row in dense_rows], dtype=np.float64)
    density_linf = np.asarray([row["density_linf"] for row in dense_rows], dtype=np.float64)
    electron_count_err = np.asarray([row["electron_count_abs_err"] for row in dense_rows], dtype=np.float64)
    tda_gap_err = np.asarray(
        [row["gap_abs_err_ev"] for row in excited_rows if row["solver"] == "tda"],
        dtype=np.float64,
    )
    casida_gap_err = np.asarray(
        [row["gap_abs_err_ev"] for row in excited_rows if row["solver"] == "casida"],
        dtype=np.float64,
    )
    tda_total_err = np.asarray(
        [row["total_abs_err_ev"] for row in excited_rows if row["solver"] == "tda"],
        dtype=np.float64,
    )
    casida_total_err = np.asarray(
        [row["total_abs_err_ev"] for row in excited_rows if row["solver"] == "casida"],
        dtype=np.float64,
    )

    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"H2 {args.training_mode} Neural_xc vs FCI summary\n")
        handle.write(f"basis = {args.basis}\n")
        handle.write(f"reference_orbital_xc = {args.xc}\n")
        handle.write("reference_method = fci_ground_state\n")
        handle.write(f"training_mode = {args.training_mode}\n")
        handle.write(f"include_pt2_channel = {bool(args.include_pt2_channel)}\n")
        handle.write(f"semilocal_xc = {tuple(str(name) for name in args.semilocal_xc)}\n")
        handle.write(f"hidden_dims = {list(int(value) for value in args.hidden_dims)}\n")
        handle.write(f"density_constraint_weight = {float(args.density_constraint_weight)}\n")
        handle.write(
            "density_matrix_constraint_weight = "
            f"{float(args.density_matrix_constraint_weight)}\n"
        )
        handle.write(f"scf_require_convergence = {bool(args.scf_require_convergence)}\n")
        handle.write(
            f"reference_scf_backend = {getattr(args, 'reference_scf_backend', 'pyscf')}\n"
        )
        handle.write(f"steps = {int(args.steps)}\n")
        handle.write(f"learning_rate = {float(args.learning_rate)}\n")
        handle.write(f"lr_decay_every = {int(args.lr_decay_every)}\n")
        handle.write(f"lr_decay_factor = {float(args.lr_decay_factor)}\n")
        handle.write(f"seed = {int(args.seed)}\n")
        handle.write(f"r_min = {float(args.r_min)}\n")
        handle.write(f"r_max = {float(args.r_max)}\n")
        handle.write(f"train_points = {int(args.train_points)}\n")
        handle.write(f"dense_points = {int(args.dense_points)}\n")
        handle.write(f"excited_nstates = {int(args.excited_nstates)}\n")
        handle.write(f"train_r_values = {np.asarray(train_r_values).tolist()}\n")
        handle.write(f"final_loss = {final_loss:.8e}\n")
        handle.write(f"min_loss = {min_loss:.8e} at step {min_loss_step}\n")
        handle.write(
            "SCF_converged_fraction = "
            f"{float(training['initial_scf_converged_fraction']):.6f} -> "
            f"{float(training['final_scf_converged_fraction']):.6f}\n"
        )
        handle.write(
            "SCF_cycles_mean = "
            f"{float(training['initial_scf_cycles_mean']):.6f} -> "
            f"{float(training['final_scf_cycles_mean']):.6f}\n"
        )
        handle.write(
            "SCF_cycles_max = "
            f"{float(training['initial_scf_cycles_max']):.6f} -> "
            f"{float(training['final_scf_cycles_max']):.6f}\n"
        )
        handle.write(
            "SCF_selected_rms_max = "
            f"{float(training['initial_scf_selected_rms_max']):.8e} -> "
            f"{float(training['final_scf_selected_rms_max']):.8e}\n"
        )
        handle.write(
            "SCF_final_rms_max = "
            f"{float(training['initial_scf_final_rms_max']):.8e} -> "
            f"{float(training['final_scf_final_rms_max']):.8e}\n"
        )
        handle.write(f"MAE_ground = {energy_err_ev.mean():.6f} eV\n")
        handle.write(f"MAX_ground = {energy_err_ev.max():.6f} eV\n")
        handle.write(f"density_L1_mean = {density_l1.mean():.8e}\n")
        handle.write(f"density_L1_max = {density_l1.max():.8e}\n")
        handle.write(f"density_L2_mean = {density_l2.mean():.8e}\n")
        handle.write(f"density_L2_max = {density_l2.max():.8e}\n")
        handle.write(f"density_Linf_mean = {density_linf.mean():.8e}\n")
        handle.write(f"density_Linf_max = {density_linf.max():.8e}\n")
        handle.write(f"electron_count_abs_err_mean = {electron_count_err.mean():.8e}\n")
        handle.write(f"electron_count_abs_err_max = {electron_count_err.max():.8e}\n")
        if tda_gap_err.size > 0:
            handle.write(f"TDA_gap_MAE = {tda_gap_err.mean():.8e} eV\n")
            handle.write(f"TDA_gap_MAX = {tda_gap_err.max():.8e} eV\n")
        if casida_gap_err.size > 0:
            handle.write(f"Casida_gap_MAE = {casida_gap_err.mean():.8e} eV\n")
            handle.write(f"Casida_gap_MAX = {casida_gap_err.max():.8e} eV\n")
        if tda_total_err.size > 0:
            handle.write(f"TDA_excited_total_MAE = {tda_total_err.mean():.8e} eV\n")
            handle.write(f"TDA_excited_total_MAX = {tda_total_err.max():.8e} eV\n")
        if casida_total_err.size > 0:
            handle.write(f"Casida_excited_total_MAE = {casida_total_err.mean():.8e} eV\n")
            handle.write(f"Casida_excited_total_MAX = {casida_total_err.max():.8e} eV\n")
        handle.write(f"train_wall_time_s = {train_elapsed_s:.2f}\n")
        handle.write(f"checkpoint = {checkpoint_path}\n")
        handle.write(f"checkpoint_meta = {checkpoint_meta_path}\n")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(outdir / "run.log")
    logger.log(
        "Config: "
        f"basis={args.basis}, xc={args.xc}, R=[{args.r_min},{args.r_max}], "
        f"train_points={args.train_points}, dense_points={args.dense_points}, "
        f"steps={args.steps}, lr={args.learning_rate}, "
        f"training_mode={args.training_mode}, "
        f"include_pt2_channel={bool(args.include_pt2_channel)}, "
        f"density_constraint_weight={args.density_constraint_weight}, "
        f"density_matrix_constraint_weight={args.density_matrix_constraint_weight}, "
        f"grid_ao_backend={args.grid_ao_backend}, integral_backend={args.integral_backend}, "
        f"jk_backend={args.jk_backend}"
    )
    logger.log("Loading runtime dependencies...")
    _load_runtime_dependencies(logger)
    logger.log("Runtime dependencies loaded.")

    train_r_values = np.linspace(float(args.r_min), float(args.r_max), int(args.train_points))
    dense_r_values = np.linspace(float(args.r_min), float(args.r_max), int(args.dense_points))

    logger.log(
        f"Building {int(args.train_points)}-point training references "
        "(FCI + strict-JAX reference)..."
    )
    train_points = build_reference_curve(
        train_r_values,
        args=args,
        logger=logger,
        label="train_ref",
    )
    logger.log(
        f"Building {int(args.dense_points)}-point dense references "
        "(FCI + strict-JAX reference)..."
    )
    dense_points = build_reference_curve(
        dense_r_values,
        args=args,
        logger=logger,
        label="dense_ref",
    )

    training = train_functional(
        train_points,
        args=args,
        logger=logger,
    )
    functional = training["functional"]
    gs_training = training["training_config"]
    params = training["best_params"]

    logger.log(
        "Evaluating dense 100-point ground-state curve/density and excited-state energies..."
    )
    dense_rows, excited_rows = evaluate_dense_curve(
        dense_points,
        functional=functional,
        params=params,
        training_config=gs_training,
        excited_nstates=int(args.excited_nstates),
        logger=logger,
    )

    dense_csv = outdir / "h2_fci_ground_vs_neural_dense_curve.csv"
    excited_csv = outdir / "h2_fci_excited_vs_neural_dense_curve.csv"
    write_dense_csv(dense_csv, dense_rows)
    if excited_rows:
        write_dense_csv(excited_csv, excited_rows)

    training_curve_csv = outdir / "training_curve.csv"
    training_curve_png = outdir / "training_loss.png"
    curve_png = outdir / "h2_fci_ground_vs_neural_dense_curve.png"
    excited_png = outdir / "h2_fci_excited_vs_neural_dense_curve.png"
    profile_png = outdir / "h2_fci_ground_density_profiles.png"
    profile_csv = outdir / "h2_fci_ground_density_profiles.csv"
    summary_path = outdir / "summary.txt"
    checkpoint_path = outdir / "neural_xc_params.msgpack"

    training_run_like = type(
        "TrainingRunLike",
        (),
        {
            "loss_history": training["loss_history"],
            "density_penalty_history": training["density_penalty_history"],
            "density_matrix_penalty_history": training["density_matrix_penalty_history"],
            "stationarity_penalty_history": training["stationarity_penalty_history"],
            "coefficient_prior_penalty_history": training["coefficient_prior_penalty_history"],
            "grad_norm_history": training["grad_norm_history"],
            "grad_abs_max_history": training["grad_abs_max_history"],
            "param_update_norm_history": training["param_update_norm_history"],
            "nonfinite_grad_fraction_history": training["nonfinite_grad_fraction_history"],
        },
    )()

    from td_graddft.workflows.reporting import plot_training_curves, write_training_curve_csv

    write_training_curve_csv(training_curve_csv, training_run_like)
    plot_training_curves(
        training_curve_png,
        training_run_like,
        title=f"H2 {str(args.training_mode)} ground-state training",
    )
    plot_curve_summary(
        curve_png,
        dense_rows,
        train_r_values=train_r_values,
        basis=str(args.basis),
        xc=str(args.xc),
        training_mode=str(args.training_mode),
    )
    if excited_rows:
        plot_excited_state_summary(
            excited_png,
            excited_rows,
            max_states=int(args.excited_nstates),
            basis=str(args.basis),
            xc=str(args.xc),
        )
    profile_rows = plot_selected_density_profiles(
        profile_png,
        dense_points,
        dense_rows,
        functional=functional,
        params=params,
        training_config=gs_training,
        basis=str(args.basis),
        max_l=int(args.max_l),
        profile_r_values=[float(value) for value in args.profile_r_values],
        line_profile_points=int(args.line_profile_points),
    )
    write_dense_csv(profile_csv, profile_rows)

    checkpoint_path, checkpoint_meta_path = save_params_checkpoint(
        checkpoint_path,
        params,
        metadata={
            "basis": str(args.basis),
            "xc": str(args.xc),
            "training_mode": str(args.training_mode),
            "reference_method": "fci_ground_state",
            "train_r_values_angstrom": [float(value) for value in train_r_values],
            "dense_points": int(args.dense_points),
            "steps": int(args.steps),
            "learning_rate": float(args.learning_rate),
            "hidden_dims": [int(value) for value in args.hidden_dims],
            "include_pt2_channel": bool(args.include_pt2_channel),
            "density_constraint_weight": float(args.density_constraint_weight),
            "density_matrix_constraint_weight": float(args.density_matrix_constraint_weight),
        },
    )
    write_summary(
        summary_path,
        args=args,
        train_r_values=train_r_values,
        dense_rows=dense_rows,
        excited_rows=excited_rows,
        training=training,
        final_loss=float(training["final_loss"]),
        min_loss=float(training["min_loss"]),
        min_loss_step=int(training["min_loss_step"]),
        train_elapsed_s=float(training["elapsed_s"]),
        checkpoint_path=checkpoint_path,
        checkpoint_meta_path=checkpoint_meta_path,
    )

    tda_gap_values = [row["gap_abs_err_ev"] for row in excited_rows if row["solver"] == "tda"]
    casida_gap_values = [row["gap_abs_err_ev"] for row in excited_rows if row["solver"] == "casida"]
    tda_total_values = [row["total_abs_err_ev"] for row in excited_rows if row["solver"] == "tda"]
    casida_total_values = [row["total_abs_err_ev"] for row in excited_rows if row["solver"] == "casida"]

    summary_json = {
        "basis": str(args.basis),
        "xc": str(args.xc),
        "training_mode": str(args.training_mode),
        "include_pt2_channel": bool(args.include_pt2_channel),
        "reference_method": "fci_ground_state",
        "train_r_values_angstrom": [float(value) for value in train_r_values],
        "dense_points": int(args.dense_points),
        "excited_nstates": int(args.excited_nstates),
        "steps": int(args.steps),
        "density_constraint_weight": float(args.density_constraint_weight),
        "density_matrix_constraint_weight": float(args.density_matrix_constraint_weight),
        "scf_require_convergence": bool(args.scf_require_convergence),
        "reference_scf_backend": str(getattr(args, "reference_scf_backend", "pyscf")),
        "final_loss": float(training["final_loss"]),
        "min_loss": float(training["min_loss"]),
        "min_loss_step": int(training["min_loss_step"]),
        "scf_converged_fraction_initial": float(training["initial_scf_converged_fraction"]),
        "scf_converged_fraction_final": float(training["final_scf_converged_fraction"]),
        "scf_cycles_mean_initial": float(training["initial_scf_cycles_mean"]),
        "scf_cycles_mean_final": float(training["final_scf_cycles_mean"]),
        "scf_cycles_max_initial": float(training["initial_scf_cycles_max"]),
        "scf_cycles_max_final": float(training["final_scf_cycles_max"]),
        "scf_selected_rms_max_initial": float(training["initial_scf_selected_rms_max"]),
        "scf_selected_rms_max_final": float(training["final_scf_selected_rms_max"]),
        "scf_final_rms_max_initial": float(training["initial_scf_final_rms_max"]),
        "scf_final_rms_max_final": float(training["final_scf_final_rms_max"]),
        "ground_mae_ev": float(np.mean([row["energy_abs_err_ev"] for row in dense_rows])),
        "density_l2_mean": float(np.mean([row["density_l2"] for row in dense_rows])),
        "density_l2_max": float(np.max([row["density_l2"] for row in dense_rows])),
        "tda_gap_mae_ev": float(np.mean(tda_gap_values)) if tda_gap_values else float("nan"),
        "casida_gap_mae_ev": float(np.mean(casida_gap_values)) if casida_gap_values else float("nan"),
        "tda_excited_total_mae_ev": float(np.mean(tda_total_values)) if tda_total_values else float("nan"),
        "casida_excited_total_mae_ev": float(np.mean(casida_total_values)) if casida_total_values else float("nan"),
        "dense_csv": str(dense_csv),
        "excited_csv": str(excited_csv) if excited_rows else None,
        "training_curve_csv": str(training_curve_csv),
        "profile_csv": str(profile_csv),
        "curve_png": str(curve_png),
        "excited_png": str(excited_png) if excited_rows else None,
        "profile_png": str(profile_png),
        "training_curve_png": str(training_curve_png),
        "summary_txt": str(summary_path),
        "visualization_manifest": str(outdir / "visualization_manifest.json"),
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary_json, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    visualization_manifest = {
        "paper_experiment": "Ground-State Potential-Energy Surfaces",
        "description": "Data files needed to reproduce H2 ground-state PES visualizations.",
        "figures": [
            {
                "figure": str(curve_png),
                "data_files": [str(dense_csv)],
                "x": "r_angstrom",
                "y": [
                    "fci_energy_h",
                    "predicted_energy_h",
                    "energy_abs_err_ev",
                    "density_l2",
                    "electron_count_abs_err",
                ],
            },
            {
                "figure": str(training_curve_png),
                "data_files": [str(training_curve_csv)],
                "x": "step",
                "y": [
                    "loss",
                    "density_penalty",
                    "density_matrix_penalty",
                    "stationarity_penalty",
                    "coefficient_prior_penalty",
                    "grad_norm",
                    "param_update_norm",
                ],
            },
            {
                "figure": str(profile_png),
                "data_files": [str(profile_csv)],
                "x": "z_angstrom",
                "y": ["fci_density", "predicted_density", "density_error"],
                "group_by": ["profile_index", "r_angstrom"],
            },
            {
                "figure": str(excited_png) if excited_rows else None,
                "data_files": [str(excited_csv)] if excited_rows else [],
                "x": "r_angstrom",
                "y": [
                    "fci_total_energy_h",
                    "predicted_total_energy_h",
                    "gap_abs_err_ev",
                    "total_abs_err_ev",
                ],
                "group_by": ["solver", "state_index"],
            },
        ],
        "metadata_files": [str(summary_path), str(outdir / "summary.json")],
    }
    visualization_manifest_path = outdir / "visualization_manifest.json"
    visualization_manifest_path.write_text(
        json.dumps(visualization_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    logger.log(f"Wrote dense csv : {dense_csv}")
    logger.log(f"Wrote curve png : {curve_png}")
    if excited_rows:
        logger.log(f"Wrote excited csv: {excited_csv}")
        logger.log(f"Wrote excited png: {excited_png}")
    logger.log(f"Wrote profile png: {profile_png}")
    logger.log(f"Wrote profile csv: {profile_csv}")
    logger.log(f"Wrote loss png  : {training_curve_png}")
    logger.log(f"Wrote summary   : {summary_path}")
    logger.log(f"Wrote vis data  : {visualization_manifest_path}")
    logger.log(f"Wrote params    : {checkpoint_path}")


if __name__ == "__main__":
    main()
