from __future__ import annotations

import argparse
import csv
import os
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

from td_graddft import neural_xc
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


def _compute_cisd_ground(mol) -> float:
    from pyscf import ci, scf

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge for CISD reference generation.")

    cisolver = ci.CISD(mf)
    e_corr, _ = cisolver.kernel()
    return float(e_corr + float(mf.e_tot))


def _compute_fci_ground(mol) -> float:
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
    e0, _ = cisolver.kernel(h1_mo, eri_mo, norb, nelec, nroots=1)
    # direct_spin0.FCI with (h1, eri) returns electronic energies only.
    return float(e0) + float(mol.energy_nuc())


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


def _normalize_semilocal_arg(values: list[str] | str) -> str | tuple[str, ...]:
    if isinstance(values, str):
        return values
    if len(values) == 1:
        return values[0]
    return tuple(values)


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


def _write_curve_csv(
    path: Path,
    r_grid: np.ndarray,
    e0_ref: np.ndarray,
    e0_pred: np.ndarray,
    *,
    reference_label: str,
    train_mask: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "R_Angstrom",
                f"E0_{reference_label}_Hartree",
                "E0_NeuralXC_Hartree",
                "AbsErr_eV",
                "IsTrainPoint",
            ]
        )
        for r, g0, p0, is_train in zip(r_grid, e0_ref, e0_pred, train_mask, strict=True):
            writer.writerow(
                [
                    float(r),
                    float(g0),
                    float(p0),
                    float(abs(g0 - p0) * HARTREE_TO_EV),
                    int(bool(is_train)),
                ]
            )


def _plot_curve(
    path: Path,
    r_grid: np.ndarray,
    e0_ref: np.ndarray,
    e0_pred: np.ndarray,
    *,
    train_r: np.ndarray,
    train_e0: np.ndarray,
    reference_label: str,
    mae_ground_ev: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(r_grid, e0_ref, lw=2.2, label=f"{reference_label} Ground")
    ax.plot(r_grid, e0_pred, lw=2.0, label="Neural_xc Ground")
    ax.scatter(
        train_r,
        train_e0,
        s=46,
        marker="o",
        facecolors="none",
        edgecolors="black",
        linewidths=1.25,
        label=f"{reference_label} train points",
        zorder=4,
    )
    ax.set_xlabel("H-H Distance (Angstrom)")
    ax.set_ylabel("Total Energy (Hartree)")
    ax.set_title(f"H2 Ground Dissociation | MAE={mae_ground_ev:.4f} eV")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_loss(path: Path, loss_hist: list[float], *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(np.arange(1, len(loss_hist) + 1), np.asarray(loss_hist), lw=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_summary(
    path: Path,
    *,
    basis: str,
    xc_ref: str,
    reference_method: str,
    semilocal_xc: str | tuple[str, ...],
    hidden_dims: tuple[int, ...],
    density_constraint_weight: float,
    seed: int,
    steps: int,
    learning_rate: float,
    lr_decay_steps: int,
    lr_decay_rate: float,
    r_min: float,
    r_max: float,
    points: int,
    train_indices: list[int],
    final_loss: float,
    min_loss: float,
    min_loss_step: int,
    mae_ground_ev: float,
    elapsed_s: float,
    checkpoint_path: str,
    checkpoint_meta_path: str,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"H2 {reference_method.upper()} ground-curve training summary\n")
        f.write(f"basis = {basis}\n")
        f.write(f"reference_orbital_xc = {xc_ref}\n")
        f.write(f"reference_method = {reference_method}\n")
        f.write(f"semilocal_xc = {semilocal_xc}\n")
        f.write(f"hidden_dims = {list(hidden_dims)}\n")
        f.write(f"density_constraint_weight = {density_constraint_weight}\n")
        f.write(f"seed = {seed}\n")
        f.write(f"steps = {steps}\n")
        f.write(f"learning_rate = {learning_rate}\n")
        f.write(f"lr_decay_steps = {lr_decay_steps}\n")
        f.write(f"lr_decay_rate = {lr_decay_rate}\n")
        f.write(f"r_min = {r_min}\n")
        f.write(f"r_max = {r_max}\n")
        f.write(f"points = {points}\n")
        f.write(f"train_indices = {train_indices}\n")
        f.write(f"final_loss = {final_loss:.8e}\n")
        f.write(f"min_loss = {min_loss:.8e} at step {min_loss_step}\n")
        f.write(f"MAE_ground = {mae_ground_ev:.6f} eV\n")
        f.write(f"train_wall_time_s = {elapsed_s:.2f}\n")
        f.write(f"checkpoint = {checkpoint_path}\n")
        f.write(f"checkpoint_meta = {checkpoint_meta_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Neural_xc on 5 H2 ground-state points and infer full dissociation curve."
    )
    parser.add_argument("--reference-method", choices=("cisd", "fci"), default="fci")
    parser.add_argument("--basis", type=str, default="sto-3g")
    parser.add_argument("--xc-ref", type=str, default="b3lyp")
    parser.add_argument("--r-min", type=float, default=0.05)
    parser.add_argument("--r-max", type=float, default=5.0)
    parser.add_argument("--points", type=int, default=201)
    parser.add_argument("--train-points", type=int, default=5)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lr-decay-steps", type=int, default=0)
    parser.add_argument("--lr-decay-rate", type=float, default=1.0)
    parser.add_argument("--hidden-dims", type=str, default="64,64")
    parser.add_argument(
        "--semilocal-xc",
        nargs="+",
        default=list(b3lyp_component_basis()),
        help="one or more jax_libxc semilocal specs used as Neural_xc basis channels",
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
    parser.add_argument("--density-constraint-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--scan-log-every", type=int, default=20)
    parser.add_argument("--outdir", type=str, default="outputs/h2_fci_ground_train5_fullcurve")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log, log_handle = _make_progress_logger(outdir / "run.log")
    try:
        if args.train_points < 2:
            raise ValueError("--train-points must be >= 2.")
        if args.train_points > args.points:
            raise ValueError("--train-points must be <= --points.")

        hidden_dims = tuple(int(x.strip()) for x in args.hidden_dims.split(",") if x.strip())
        semilocal_xc = _normalize_semilocal_arg(args.semilocal_xc)
        reference_method = args.reference_method.lower()
        reference_label = reference_method.upper()
        r_grid = np.linspace(args.r_min, args.r_max, args.points)

        log(
            "Config: "
            f"reference_method={reference_method}, basis={args.basis}, xc_ref={args.xc_ref}, "
            f"R=[{args.r_min},{args.r_max}], points={args.points}, "
            f"train_points={args.train_points}, steps={args.steps}, lr={args.learning_rate}, "
            f"lr_decay_steps={args.lr_decay_steps}, lr_decay_rate={args.lr_decay_rate}, "
            f"hidden_dims={list(hidden_dims)}"
        )
        log("Building references on full R grid...")

        references = []
        e0_ref = np.zeros_like(r_grid)
        for i, r in enumerate(r_grid):
            mol = _build_h2_mol(float(r), basis=args.basis)
            if reference_method == "fci":
                g = _compute_fci_ground(mol)
            else:
                g = _compute_cisd_ground(mol)
            ref = _build_reference_from_rks(mol, xc=args.xc_ref)
            references.append(ref)
            e0_ref[i] = g
            if i == 0 or (i + 1) == args.points or ((i + 1) % max(1, args.scan_log_every) == 0):
                log(
                    f"[ref] {i + 1:3d}/{args.points} R={float(r):.4f} A "
                    f"E0_{reference_label}={g:.10f} Eh"
                )

        train_indices = np.linspace(0, args.points - 1, args.train_points, dtype=int).tolist()
        train_mask = np.zeros(args.points, dtype=bool)
        train_mask[train_indices] = True
        log(f"Training indices: {train_indices}")

        functional = neural_xc.Functional(
            semilocal_xc=semilocal_xc,
            n_semilocal_channels=args.n_semilocal_channels,
            hf_input_mode=args.hf_input_mode,
            coefficient_positivity=args.coefficient_positivity,
            hidden_dims=hidden_dims,
            name=f"neural_xc_h2_{reference_method}_ground_fit",
        )
        train_data = [
            GroundStateDatum(
                molecule=references[idx],
                target_total_energy=jnp.asarray(float(e0_ref[idx])),
                density_constraint_weight=float(args.density_constraint_weight),
            )
            for idx in train_indices
        ]

        gs_cfg = GroundStateTrainingConfig(mode="fixed_density")
        lr_schedule = None
        if int(args.lr_decay_steps) > 0 and float(args.lr_decay_rate) != 1.0:
            lr_schedule = optax.exponential_decay(
                init_value=float(args.learning_rate),
                transition_steps=int(args.lr_decay_steps),
                decay_rate=float(args.lr_decay_rate),
                staircase=True,
            )
        tx = optax.adam(lr_schedule if lr_schedule is not None else float(args.learning_rate))
        train_state = create_train_state_from_molecule(
            functional,
            jax.random.PRNGKey(args.seed),
            references[0],
            tx,
        )
        train_step = make_ground_state_train_step(functional, training_config=gs_cfg)

        log("Starting training...")
        loss_hist: list[float] = []
        t0 = time.perf_counter()
        for step in range(1, args.steps + 1):
            train_state, metrics = train_step(train_state, train_data)
            loss = float(metrics["loss"])
            loss_hist.append(loss)
            current_lr = (
                float(lr_schedule(step - 1))
                if lr_schedule is not None
                else float(args.learning_rate)
            )
            if step == 1 or step == args.steps or (step % max(1, args.log_every) == 0):
                log(
                    f"[train] step={step:4d}/{args.steps} "
                    f"loss={loss:.8e} lr={current_lr:.8e}"
                )
        elapsed = time.perf_counter() - t0

        final_loss, _ = ground_state_mse_loss(
            train_state.params,
            functional,
            train_data,
            training_config=gs_cfg,
        )
        final_loss = float(final_loss)
        min_loss = float(np.min(loss_hist))
        min_loss_step = int(np.argmin(loss_hist) + 1)

        log("Evaluating full ground-state curve...")
        e0_pred = np.zeros_like(r_grid)
        for i, ref in enumerate(references):
            e0_i = float(
                predict_ground_state_total_energy(
                    train_state.params,
                    functional,
                    ref,
                    training_config=gs_cfg,
                )
            )
            e0_pred[i] = e0_i
            if i == 0 or (i + 1) == args.points or ((i + 1) % max(1, args.scan_log_every) == 0):
                log(
                    f"[eval] {i + 1:3d}/{args.points} R={float(r_grid[i]):.4f} A "
                    f"E0_pred={e0_i:.10f} Eh"
                )

        mae_ground_ev = float(np.mean(np.abs(e0_pred - e0_ref)) * HARTREE_TO_EV)

        ckpt_path, ckpt_meta_path = save_params_checkpoint(
            outdir / "neural_xc_params.msgpack",
            train_state.params,
            metadata={
                "model_family": "Neural_xc",
                "functional_name": functional.name,
                "reference_method": reference_method,
                "basis": args.basis,
                "xc_ref": args.xc_ref,
                "r_min": float(args.r_min),
                "r_max": float(args.r_max),
                "points": int(args.points),
                "train_points": int(args.train_points),
                "train_indices": train_indices,
                "semilocal_xc": semilocal_xc,
                "hidden_dims": list(hidden_dims),
                "hf_input_mode": args.hf_input_mode,
                "coefficient_positivity": args.coefficient_positivity,
                "density_constraint_weight": float(args.density_constraint_weight),
                "optimizer": "adam",
                "learning_rate": float(args.learning_rate),
                "lr_decay_steps": int(args.lr_decay_steps),
                "lr_decay_rate": float(args.lr_decay_rate),
                "steps": int(args.steps),
                "seed": int(args.seed),
                "mae_ground_ev": float(mae_ground_ev),
            },
        )

        csv_path = outdir / f"h2_{reference_method}_ground_vs_neural_curve.csv"
        curve_png = outdir / f"h2_{reference_method}_ground_vs_neural_curve.png"
        loss_png = outdir / "training_loss.png"
        summary_path = outdir / "summary.txt"
        _write_curve_csv(
            csv_path,
            r_grid,
            e0_ref,
            e0_pred,
            reference_label=reference_label,
            train_mask=train_mask,
        )
        _plot_curve(
            curve_png,
            r_grid,
            e0_ref,
            e0_pred,
            train_r=r_grid[train_indices],
            train_e0=e0_ref[train_indices],
            reference_label=reference_label,
            mae_ground_ev=mae_ground_ev,
        )
        _plot_loss(
            loss_png,
            loss_hist,
            title=f"H2 {reference_label} Ground-Curve Training",
        )
        _write_summary(
            summary_path,
            basis=args.basis,
            xc_ref=args.xc_ref,
            reference_method=reference_method,
            semilocal_xc=semilocal_xc,
            hidden_dims=hidden_dims,
            density_constraint_weight=float(args.density_constraint_weight),
            seed=int(args.seed),
            steps=int(args.steps),
            learning_rate=float(args.learning_rate),
            lr_decay_steps=int(args.lr_decay_steps),
            lr_decay_rate=float(args.lr_decay_rate),
            r_min=float(args.r_min),
            r_max=float(args.r_max),
            points=int(args.points),
            train_indices=train_indices,
            final_loss=final_loss,
            min_loss=min_loss,
            min_loss_step=min_loss_step,
            mae_ground_ev=mae_ground_ev,
            elapsed_s=elapsed,
            checkpoint_path=str(ckpt_path),
            checkpoint_meta_path=str(ckpt_meta_path) if ckpt_meta_path is not None else "",
        )

        log(f"Final loss: {final_loss:.8e}")
        log(f"Min loss: {min_loss:.8e} @ step {min_loss_step}")
        log(f"Ground MAE: {mae_ground_ev:.6f} eV")
        log(f"Wrote curve csv: {csv_path}")
        log(f"Wrote curve png: {curve_png}")
        log(f"Wrote loss  png: {loss_png}")
        log(f"Wrote summary : {summary_path}")
        log(f"Wrote params : {ckpt_path}")
        if ckpt_meta_path is not None:
            log(f"Wrote params meta: {ckpt_meta_path}")
    finally:
        log_handle.close()


if __name__ == "__main__":
    main()
