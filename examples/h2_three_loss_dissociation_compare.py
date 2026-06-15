from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from td_graddft import neural_xc, tdscf
from td_graddft.xc_backend.jax_libxc import b3lyp_component_basis
from td_graddft.data.reference import restricted_reference_from_pyscf
from td_graddft.spectra import HARTREE_TO_EV
from td_graddft.training import (
    GroundStateDatum,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    ground_state_mse_loss,
    make_ground_state_train_step,
    predict_ground_state_total_energy,
    save_params_checkpoint,
)


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    label: str
    energy_mse_weight: float
    energy_mae_weight: float
    s1_weight: float
    first_excited_total_weight: float


@dataclass
class ObjectiveResult:
    spec: ObjectiveSpec
    functional: object
    params: object
    loss_history: list[float]
    final_loss: float
    min_loss: float
    min_loss_step: int
    checkpoint_path: Path
    checkpoint_meta_path: Path | None


def _build_h2_mol(r_angstrom: float, basis: str):
    from pyscf import gto

    mol = gto.Mole()
    mol.atom = f"""
    H 0.000000 0.000000 {-0.5 * r_angstrom:.10f}
    H 0.000000 0.000000 {+0.5 * r_angstrom:.10f}
    """
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.spin = 0
    mol.charge = 0
    mol.verbose = 0
    mol.build()
    return mol


def _compute_cisd_ground_excited(mol, *, nroots: int = 3) -> tuple[float, float]:
    from pyscf import ci, scf

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge for CISD reference generation.")

    cisolver = ci.CISD(mf)
    cisolver.nroots = max(2, int(nroots))
    e_corr_roots, _ = cisolver.kernel()
    e_corr_roots = np.asarray(e_corr_roots, dtype=float).reshape(-1)
    if e_corr_roots.size < 2:
        raise RuntimeError("CISD did not return at least two roots.")

    e_tot_roots = e_corr_roots + float(mf.e_tot)
    return float(e_tot_roots[0]), float(e_tot_roots[1])


def _compute_fci_ground_excited(mol, *, nroots: int = 3) -> tuple[float, float]:
    from pyscf import ao2mo, fci, scf

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge for FCI reference generation.")

    h1_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri_mo = ao2mo.kernel(mol, mf.mo_coeff)
    norb = h1_mo.shape[0]
    nelec = mol.nelectron

    cisolver = fci.direct_spin0.FCI(mol)
    e_roots, _ = cisolver.kernel(h1_mo, eri_mo, norb, nelec, nroots=max(2, int(nroots)))
    e_roots = np.asarray(e_roots, dtype=float).reshape(-1)
    if e_roots.size < 2:
        raise RuntimeError("FCI did not return at least two singlet roots.")
    e_nuc = float(mol.energy_nuc())
    return float(e_roots[0] + e_nuc), float(e_roots[1] + e_nuc)


def _build_reference_from_rks(mol, xc: str):
    from pyscf import dft

    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RKS did not converge while building training references.")
    return restricted_reference_from_pyscf(mf)


def _make_progress_logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")

    def _log(message: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        print(line, flush=True)
        handle.write(line + "\n")
        handle.flush()

    return _log, handle


def _normalize_semilocal_arg(values: list[str] | str) -> str | tuple[str, ...]:
    if isinstance(values, str):
        return values
    if len(values) == 1:
        return values[0]
    return tuple(values)


def _objective_specs(args: argparse.Namespace) -> list[ObjectiveSpec]:
    specs = [
        ObjectiveSpec(
            name="ground_only",
            label="Ground Only",
            energy_mse_weight=float(args.energy_mse_weight),
            energy_mae_weight=float(args.energy_mae_weight),
            s1_weight=0.0,
            first_excited_total_weight=0.0,
        ),
        ObjectiveSpec(
            name="ground_plus_s1",
            label="Ground + S1 Gap",
            energy_mse_weight=float(args.energy_mse_weight),
            energy_mae_weight=float(args.energy_mae_weight),
            s1_weight=float(args.s1_weight),
            first_excited_total_weight=0.0,
        ),
        ObjectiveSpec(
            name="s1_only",
            label="S1 Gap Only",
            energy_mse_weight=0.0,
            energy_mae_weight=0.0,
            s1_weight=float(args.s1_weight),
            first_excited_total_weight=0.0,
        ),
        ObjectiveSpec(
            name="e1_total_only",
            label="E1 Total Only",
            energy_mse_weight=0.0,
            energy_mae_weight=0.0,
            s1_weight=0.0,
            first_excited_total_weight=float(args.first_excited_total_weight),
        ),
    ]
    selected = getattr(args, "objectives", None)
    if not selected:
        return specs

    selected_names = tuple(str(name) for name in selected)
    filtered = [spec for spec in specs if spec.name in selected_names]
    if not filtered:
        raise ValueError(
            "No valid objectives selected. Choose from: "
            + ", ".join(spec.name for spec in specs)
        )
    return filtered


def _build_train_data(
    references: list[object],
    e0_ref: np.ndarray,
    e1_ref: np.ndarray,
    train_indices: list[int],
    *,
    spec: ObjectiveSpec,
) -> list[GroundStateDatum]:
    data: list[GroundStateDatum] = []
    for idx in train_indices:
        data.append(
            GroundStateDatum(
                molecule=references[idx],
                target_total_energy=jnp.asarray(float(e0_ref[idx])),
                target_s1_energy=jnp.asarray(float(e1_ref[idx] - e0_ref[idx])),
                target_first_excited_total_energy=jnp.asarray(float(e1_ref[idx])),
                s1_constraint_weight=float(spec.s1_weight),
                first_excited_total_energy_constraint_weight=float(spec.first_excited_total_weight),
            )
        )
    return data


def _predict_first_excitation_au(
    params: object,
    functional: object,
    molecule: object,
    *,
    use_tda: bool,
) -> float:
    td = tdscf.TDA(
        molecule,
        xc_functional=functional,
        xc_params=params,
    ) if use_tda else tdscf.TDDFT(
        molecule,
        xc_functional=functional,
        xc_params=params,
    )
    if use_tda:
        result = td.kernel(nstates=1)
    else:
        try:
            result = td.kernel(nstates=1)
        except Exception:
            td = tdscf.TDA(
                molecule,
                xc_functional=functional,
                xc_params=params,
            )
            result = td.kernel(nstates=1)
        if result.excitation_energies.size == 0:
            td = tdscf.TDA(
                molecule,
                xc_functional=functional,
                xc_params=params,
            )
            result = td.kernel(nstates=1)
    if result.excitation_energies.size == 0:
        raise RuntimeError("No excited states returned during curve evaluation.")
    return float(result.excitation_energies[0])


def _train_one_objective(
    spec: ObjectiveSpec,
    *,
    args: argparse.Namespace,
    references: list[object],
    e0_ref: np.ndarray,
    e1_ref: np.ndarray,
    train_indices: list[int],
    semilocal_xc: str | tuple[str, ...],
    outdir: Path,
    log,
) -> ObjectiveResult:
    functional = neural_xc.Functional(
        semilocal_xc=semilocal_xc,
        n_semilocal_channels=args.n_semilocal_channels,
        hf_input_mode=args.hf_input_mode,
        coefficient_positivity=args.coefficient_positivity,
        hidden_dims=tuple(int(x.strip()) for x in args.hidden_dims.split(",") if x.strip()),
        name=f"neural_xc_h2_{spec.name}",
    )
    train_data = _build_train_data(
        references,
        e0_ref,
        e1_ref,
        train_indices,
        spec=spec,
    )
    cfg = GroundStateTrainingConfig(
        mode="fixed_density",
        s1_constraint_use_tda=bool(args.s1_use_tda),
        energy_mse_weight=float(spec.energy_mse_weight),
        energy_mae_weight=float(spec.energy_mae_weight),
    )
    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(int(args.seed)),
        references[0],
        optax.adam(float(args.learning_rate)),
    )
    train_step = make_ground_state_train_step(functional, training_config=cfg)

    loss_history: list[float] = []
    best_loss = float("inf")
    best_params = state.params
    best_step = 0

    log(
        f"[{spec.name}] start training "
        f"(energy_mse={spec.energy_mse_weight}, energy_mae={spec.energy_mae_weight}, "
        f"s1_gap_weight={spec.s1_weight}, e1_total_weight={spec.first_excited_total_weight})"
    )
    for step in range(1, int(args.steps) + 1):
        state, metrics = train_step(state, train_data)
        loss_val = float(metrics["loss"])
        loss_history.append(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val
            best_params = state.params
            best_step = step
        if step == 1 or step == int(args.steps) or (step % max(1, int(args.log_every)) == 0):
            log(f"[{spec.name}] step={step:4d}/{args.steps} loss={loss_val:.8e}")

    final_loss, _ = ground_state_mse_loss(
        best_params,
        functional,
        train_data,
        training_config=cfg,
    )
    final_loss = float(final_loss)
    objective_dir = outdir / spec.name
    ckpt_path, ckpt_meta_path = save_params_checkpoint(
        objective_dir / "neural_xc_params.msgpack",
        best_params,
        metadata={
            "objective_name": spec.name,
            "objective_label": spec.label,
            "energy_mse_weight": float(spec.energy_mse_weight),
            "energy_mae_weight": float(spec.energy_mae_weight),
            "s1_weight": float(spec.s1_weight),
            "first_excited_total_weight": float(spec.first_excited_total_weight),
            "s1_use_tda": bool(args.s1_use_tda),
            "basis": args.basis,
            "xc_ref": args.xc_ref,
            "reference_method": args.reference_method,
            "steps": int(args.steps),
            "learning_rate": float(args.learning_rate),
            "train_indices": train_indices,
            "seed": int(args.seed),
            "best_step": int(best_step),
            "best_loss": float(best_loss),
            "semilocal_xc": semilocal_xc,
        },
    )
    return ObjectiveResult(
        spec=spec,
        functional=functional,
        params=best_params,
        loss_history=loss_history,
        final_loss=final_loss,
        min_loss=float(np.min(loss_history)) if loss_history else final_loss,
        min_loss_step=int(np.argmin(loss_history) + 1) if loss_history else 0,
        checkpoint_path=ckpt_path,
        checkpoint_meta_path=ckpt_meta_path,
    )


def _write_loss_csv(path: Path, results: list[ObjectiveResult]) -> None:
    max_steps = max(len(result.loss_history) for result in results)
    header = ["step"] + [result.spec.name for result in results]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for step in range(max_steps):
            row: list[float | int | str] = [step + 1]
            for result in results:
                if step < len(result.loss_history):
                    row.append(float(result.loss_history[step]))
                else:
                    row.append("")
            writer.writerow(row)


def _plot_loss_curves(path: Path, results: list[ObjectiveResult]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for result in results:
        ax.plot(
            np.arange(1, len(result.loss_history) + 1),
            np.asarray(result.loss_history, dtype=float),
            lw=1.8,
            label=result.spec.label,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.set_title("H2 Dissociation Training Curves (Four Objectives)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_curve_csv(
    path: Path,
    r_grid: np.ndarray,
    e0_ref: np.ndarray,
    e1_ref: np.ndarray,
    evals: dict[str, dict[str, np.ndarray]],
) -> None:
    header = [
        "R_Angstrom",
        "E0_ref_Hartree",
        "E1_ref_Hartree",
        "Gap1_ref_eV",
    ]
    for name in evals:
        header.extend(
            [
                f"E0_{name}_Hartree",
                f"E1_{name}_Hartree",
                f"Gap1_{name}_eV",
            ]
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx, r in enumerate(r_grid):
            row: list[float] = [
                float(r),
                float(e0_ref[idx]),
                float(e1_ref[idx]),
                float((e1_ref[idx] - e0_ref[idx]) * HARTREE_TO_EV),
            ]
            for name in evals:
                row.extend(
                    [
                        float(evals[name]["e0"][idx]),
                        float(evals[name]["e1"][idx]),
                        float(evals[name]["gap_ev"][idx]),
                    ]
                )
            writer.writerow(row)


def _plot_curve_comparison(
    path: Path,
    *,
    r_grid: np.ndarray,
    e0_ref: np.ndarray,
    e1_ref: np.ndarray,
    evals: dict[str, dict[str, np.ndarray]],
    train_r: np.ndarray,
    train_e0: np.ndarray,
    train_gap_ev: np.ndarray,
    reference_label: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))

    axes[0].plot(r_grid, e0_ref, lw=2.4, color="black", label=f"{reference_label} E0")
    for name, values in evals.items():
        axes[0].plot(r_grid, values["e0"], lw=1.9, label=name)
    axes[0].scatter(
        train_r,
        train_e0,
        s=42,
        marker="o",
        facecolors="none",
        edgecolors="black",
        linewidths=1.1,
        label="train points",
        zorder=4,
    )
    axes[0].set_xlabel("H-H Distance (Angstrom)")
    axes[0].set_ylabel("Total Energy (Hartree)")
    axes[0].set_title("Ground-State Curve")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(r_grid, e1_ref, lw=2.4, color="black", label=f"{reference_label} E1")
    for name, values in evals.items():
        axes[1].plot(r_grid, values["e1"], lw=1.9, label=name)
    axes[1].set_xlabel("H-H Distance (Angstrom)")
    axes[1].set_ylabel("Total Energy (Hartree)")
    axes[1].set_title("First-Excited Curve")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=9)

    gap_ref_ev = (e1_ref - e0_ref) * HARTREE_TO_EV
    axes[2].plot(r_grid, gap_ref_ev, lw=2.4, color="black", label=f"{reference_label} Gap")
    for name, values in evals.items():
        axes[2].plot(r_grid, values["gap_ev"], lw=1.9, label=name)
    axes[2].scatter(
        train_r,
        train_gap_ev,
        s=42,
        marker="o",
        facecolors="none",
        edgecolors="black",
        linewidths=1.1,
        label="train points",
        zorder=4,
    )
    axes[2].set_xlabel("H-H Distance (Angstrom)")
    axes[2].set_ylabel("Excitation Energy (eV)")
    axes[2].set_title("S1 Gap Curve")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False, fontsize=9)

    fig.suptitle("H2 Dissociation Learning with Four Objectives")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    semilocal_xc: str | tuple[str, ...],
    train_indices: list[int],
    metrics: dict[str, dict[str, float]],
    results: list[ObjectiveResult],
    elapsed_s: float,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("H2 dissociation comparison with four training losses\n")
        f.write(f"reference_method = {args.reference_method}\n")
        f.write(f"basis = {args.basis}\n")
        f.write(f"xc_ref = {args.xc_ref}\n")
        f.write(f"semilocal_xc = {semilocal_xc}\n")
        f.write(f"hidden_dims = {args.hidden_dims}\n")
        f.write(f"hf_input_mode = {args.hf_input_mode}\n")
        f.write(f"coefficient_positivity = {args.coefficient_positivity}\n")
        f.write(f"points = {args.points}\n")
        f.write(f"train_points = {args.train_points}\n")
        f.write(f"train_indices = {train_indices}\n")
        f.write(f"steps = {args.steps}\n")
        f.write(f"learning_rate = {args.learning_rate}\n")
        f.write(f"energy_mse_weight = {args.energy_mse_weight}\n")
        f.write(f"energy_mae_weight = {args.energy_mae_weight}\n")
        f.write(f"s1_weight = {args.s1_weight}\n")
        f.write(f"first_excited_total_weight = {args.first_excited_total_weight}\n")
        f.write(f"s1_use_tda = {bool(args.s1_use_tda)}\n")
        f.write(f"eval_use_tda = {bool(args.eval_use_tda)}\n")
        f.write(f"elapsed_s = {elapsed_s:.2f}\n")
        f.write("\n")
        for result in results:
            objective_metrics = metrics[result.spec.name]
            f.write(f"[{result.spec.name}]\n")
            f.write(f"label = {result.spec.label}\n")
            f.write(f"final_loss = {result.final_loss:.8e}\n")
            f.write(f"min_loss = {result.min_loss:.8e}\n")
            f.write(f"min_loss_step = {result.min_loss_step}\n")
            f.write(f"mae_ground_ev = {objective_metrics['mae_ground_ev']:.6f}\n")
            f.write(f"mae_excited_ev = {objective_metrics['mae_excited_ev']:.6f}\n")
            f.write(f"mae_gap_ev = {objective_metrics['mae_gap_ev']:.6f}\n")
            f.write(f"checkpoint = {result.checkpoint_path}\n")
            if result.checkpoint_meta_path is not None:
                f.write(f"checkpoint_meta = {result.checkpoint_meta_path}\n")
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train four Neural_xc objectives on an H2 dissociation dataset: "
            "ground only, ground+S1 gap, S1 gap only, and E1 total only."
        )
    )
    parser.add_argument("--reference-method", choices=("cisd", "fci"), default="fci")
    parser.add_argument("--basis", type=str, default="sto-3g")
    parser.add_argument("--xc-ref", type=str, default="b3lyp")
    parser.add_argument("--r-min", type=float, default=0.45)
    parser.add_argument("--r-max", type=float, default=3.0)
    parser.add_argument("--points", type=int, default=41)
    parser.add_argument("--train-points", type=int, default=5)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", type=str, default="64,64")
    parser.add_argument(
        "--semilocal-xc",
        nargs="+",
        default=list(b3lyp_component_basis()),
        help="One or more jax_libxc semilocal specs used as Neural_xc basis channels.",
    )
    parser.add_argument("--n-semilocal-channels", type=int, default=None)
    parser.add_argument(
        "--hf-input-mode",
        choices=("total_only", "spin_resolved"),
        default="spin_resolved",
    )
    parser.add_argument(
        "--coefficient-positivity",
        choices=("clip", "softplus"),
        default="clip",
    )
    parser.add_argument("--energy-mse-weight", type=float, default=1.0)
    parser.add_argument("--energy-mae-weight", type=float, default=0.0)
    parser.add_argument("--s1-weight", type=float, default=1.0)
    parser.add_argument("--first-excited-total-weight", type=float, default=1.0)
    parser.add_argument(
        "--objectives",
        nargs="+",
        choices=("ground_only", "ground_plus_s1", "s1_only", "e1_total_only"),
        default=None,
        help="Optional subset of objectives to train/evaluate. Defaults to all four.",
    )
    parser.add_argument(
        "--s1-use-tda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use TDA for the S1 term during training.",
    )
    parser.add_argument(
        "--eval-use-tda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use TDA during full-curve evaluation. Defaults to the same TDA path used by the S1 loss.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--scan-log-every", type=int, default=10)
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/h2_four_loss_dissociation_compare",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log, log_handle = _make_progress_logger(outdir / "run.log")
    try:
        if int(args.train_points) < 2:
            raise ValueError("--train-points must be >= 2.")
        if int(args.train_points) > int(args.points):
            raise ValueError("--train-points must be <= --points.")

        semilocal_xc = _normalize_semilocal_arg(args.semilocal_xc)
        reference_label = str(args.reference_method).upper()
        r_grid = np.linspace(float(args.r_min), float(args.r_max), int(args.points))
        train_indices = np.linspace(
            0,
            int(args.points) - 1,
            int(args.train_points),
            dtype=int,
        ).tolist()

        log(
            "Config: "
            f"reference_method={args.reference_method}, basis={args.basis}, xc_ref={args.xc_ref}, "
            f"R=[{args.r_min},{args.r_max}], points={args.points}, train_points={args.train_points}, "
            f"steps={args.steps}, lr={args.learning_rate}, s1_weight={args.s1_weight}, "
            f"first_excited_total_weight={args.first_excited_total_weight}"
        )
        log("Building reference dissociation dataset...")

        references: list[object] = []
        e0_ref = np.zeros_like(r_grid)
        e1_ref = np.zeros_like(r_grid)
        for idx, r in enumerate(r_grid):
            mol = _build_h2_mol(float(r), basis=str(args.basis))
            if str(args.reference_method).lower() == "fci":
                g, ex1 = _compute_fci_ground_excited(mol, nroots=3)
            else:
                g, ex1 = _compute_cisd_ground_excited(mol, nroots=3)
            ref = _build_reference_from_rks(mol, xc=str(args.xc_ref))
            references.append(ref)
            e0_ref[idx] = g
            e1_ref[idx] = ex1
            if idx == 0 or (idx + 1) == len(r_grid) or ((idx + 1) % max(1, int(args.scan_log_every)) == 0):
                log(
                    f"[ref] {idx + 1:3d}/{len(r_grid)} R={float(r):.4f} A "
                    f"E0={g:.8f} Eh E1={ex1:.8f} Eh"
                )

        results: list[ObjectiveResult] = []
        t0 = time.perf_counter()
        for spec in _objective_specs(args):
            results.append(
                _train_one_objective(
                    spec,
                    args=args,
                    references=references,
                    e0_ref=e0_ref,
                    e1_ref=e1_ref,
                    train_indices=train_indices,
                    semilocal_xc=semilocal_xc,
                    outdir=outdir,
                    log=log,
                )
            )
        elapsed = time.perf_counter() - t0

        log("Evaluating full dissociation curves...")
        evals: dict[str, dict[str, np.ndarray]] = {}
        metrics: dict[str, dict[str, float]] = {}
        for result in results:
            e0_pred = np.zeros_like(r_grid)
            e1_pred = np.zeros_like(r_grid)
            gap_ev = np.zeros_like(r_grid)
            for idx, ref in enumerate(references):
                e0_i = float(
                    predict_ground_state_total_energy(
                        result.params,
                        result.functional,
                        ref,
                    )
                )
                omega1_au = _predict_first_excitation_au(
                    result.params,
                    result.functional,
                    ref,
                    use_tda=bool(args.eval_use_tda),
                )
                e0_pred[idx] = e0_i
                e1_pred[idx] = e0_i + omega1_au
                gap_ev[idx] = omega1_au * HARTREE_TO_EV
            evals[result.spec.name] = {
                "e0": e0_pred,
                "e1": e1_pred,
                "gap_ev": gap_ev,
            }
            metrics[result.spec.name] = {
                "mae_ground_ev": float(np.mean(np.abs(e0_pred - e0_ref)) * HARTREE_TO_EV),
                "mae_excited_ev": float(np.mean(np.abs(e1_pred - e1_ref)) * HARTREE_TO_EV),
                "mae_gap_ev": float(
                    np.mean(
                        np.abs(
                            gap_ev - (e1_ref - e0_ref) * HARTREE_TO_EV
                        )
                    )
                ),
            }
            log(
                f"[eval:{result.spec.name}] "
                f"MAE_ground={metrics[result.spec.name]['mae_ground_ev']:.6f} eV "
                f"MAE_excited={metrics[result.spec.name]['mae_excited_ev']:.6f} eV "
                f"MAE_gap={metrics[result.spec.name]['mae_gap_ev']:.6f} eV"
            )

        loss_csv = outdir / "four_loss_training_curves.csv"
        loss_png = outdir / "four_loss_training_curves.png"
        curve_csv = outdir / "four_loss_dissociation_curves.csv"
        curve_png = outdir / "four_loss_dissociation_curves.png"
        summary_path = outdir / "summary.txt"

        _write_loss_csv(loss_csv, results)
        _plot_loss_curves(loss_png, results)
        _write_curve_csv(curve_csv, r_grid, e0_ref, e1_ref, evals)
        _plot_curve_comparison(
            curve_png,
            r_grid=r_grid,
            e0_ref=e0_ref,
            e1_ref=e1_ref,
            evals=evals,
            train_r=r_grid[train_indices],
            train_e0=e0_ref[train_indices],
            train_gap_ev=(e1_ref[train_indices] - e0_ref[train_indices]) * HARTREE_TO_EV,
            reference_label=reference_label,
        )
        _write_summary(
            summary_path,
            args=args,
            semilocal_xc=semilocal_xc,
            train_indices=train_indices,
            metrics=metrics,
            results=results,
            elapsed_s=elapsed,
        )

        log(f"Wrote loss csv   : {loss_csv}")
        log(f"Wrote loss png   : {loss_png}")
        log(f"Wrote curve csv  : {curve_csv}")
        log(f"Wrote curve png  : {curve_png}")
        log(f"Wrote summary    : {summary_path}")
    finally:
        log_handle.close()


if __name__ == "__main__":
    main()
