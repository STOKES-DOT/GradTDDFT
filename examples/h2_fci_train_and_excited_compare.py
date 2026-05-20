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

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from td_graddft import neural_xc, tdscf
from td_graddft.xc_backend.jax_libxc import b3lyp_component_basis
from td_graddft.reference_legacy import restricted_reference_from_pyscf
from td_graddft.spectra import HARTREE_TO_EV
from td_graddft.training import (
    GroundStateDatum,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    ground_state_mse_loss,
    load_params_checkpoint,
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


def _plot_curves(
    path: Path,
    r_grid: np.ndarray,
    e0_ref: np.ndarray,
    e0_pred: np.ndarray,
    e1_ref: np.ndarray,
    e1_pred: np.ndarray,
    *,
    train_r: np.ndarray,
    train_e0: np.ndarray,
    reference_label: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    axes[0].plot(r_grid, e0_ref, lw=2.2, label=f"{reference_label} Ground")
    axes[0].plot(r_grid, e0_pred, lw=2.0, label="Neural_xc Ground")
    axes[0].scatter(
        train_r,
        train_e0,
        s=45,
        marker="o",
        facecolors="none",
        edgecolors="black",
        linewidths=1.2,
        label=f"{reference_label} train points",
        zorder=4,
    )
    axes[0].set_xlabel("H-H Distance (Angstrom)")
    axes[0].set_ylabel("Total Energy (Hartree)")
    axes[0].set_title("Ground-State Dissociation")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(r_grid, e1_ref, lw=2.2, label=f"{reference_label} First Excited")
    axes[1].plot(r_grid, e1_pred, lw=2.0, label="Neural_xc First Excited")
    axes[1].set_xlabel("H-H Distance (Angstrom)")
    axes[1].set_ylabel("Total Energy (Hartree)")
    axes[1].set_title("First-Excited Dissociation")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_curve_csv(
    path: Path,
    r_grid: np.ndarray,
    e0_ref: np.ndarray,
    e0_pred: np.ndarray,
    e1_ref: np.ndarray,
    e1_pred: np.ndarray,
    *,
    reference_label: str,
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "R_Angstrom",
                f"E0_{reference_label}_Hartree",
                "E0_NeuralXC_Hartree",
                f"E1_{reference_label}_Hartree",
                "E1_NeuralXC_Hartree",
                f"Delta1_{reference_label}_eV",
                "Delta1_NeuralXC_eV",
            ]
        )
        for r, g0, p0, g1, p1 in zip(
            r_grid, e0_ref, e0_pred, e1_ref, e1_pred, strict=True
        ):
            writer.writerow(
                [
                    float(r),
                    float(g0),
                    float(p0),
                    float(g1),
                    float(p1),
                    float((g1 - g0) * HARTREE_TO_EV),
                    float((p1 - p0) * HARTREE_TO_EV),
                ]
            )


def _write_summary(
    path: Path,
    *,
    basis: str,
    xc_ref: str,
    reference_method: str,
    semilocal_xc: str | tuple[str, ...],
    hidden_dims: tuple[int, ...],
    activation: str,
    density_floor: float,
    kernel_clip: float,
    density_constraint_weight: float,
    s1_weight: float,
    s1_use_tda: bool,
    energy_mse_weight: float,
    energy_mae_weight: float,
    optimizer: str,
    seed: int,
    steps: int,
    learning_rate: float,
    train_indices: list[int],
    final_loss: float,
    min_loss: float,
    min_loss_step: int,
    mae_ground_ev: float,
    mae_excited_ev: float,
    elapsed_s: float,
    checkpoint_path: str,
    checkpoint_meta_path: str,
) -> None:
    with path.open("w") as f:
        f.write(f"H2 {reference_method.upper()} curve training summary\n")
        f.write(f"basis = {basis}\n")
        f.write(f"reference_orbital_xc = {xc_ref}\n")
        f.write(f"reference_method = {reference_method}\n")
        f.write(f"semilocal_xc = {semilocal_xc}\n")
        f.write(f"hidden_dims = {list(hidden_dims)}\n")
        f.write(f"activation = {activation}\n")
        f.write(f"density_floor = {density_floor}\n")
        f.write(f"kernel_clip = {kernel_clip}\n")
        f.write(f"density_constraint_weight = {density_constraint_weight}\n")
        f.write(f"s1_weight = {s1_weight}\n")
        f.write(f"s1_use_tda = {s1_use_tda}\n")
        f.write(f"energy_mse_weight = {energy_mse_weight}\n")
        f.write(f"energy_mae_weight = {energy_mae_weight}\n")
        f.write(f"optimizer = {optimizer}\n")
        f.write(f"seed = {seed}\n")
        f.write(f"steps = {steps}\n")
        f.write(f"learning_rate = {learning_rate}\n")
        f.write(f"train_indices = {train_indices}\n")
        f.write(f"final_loss = {final_loss:.8e}\n")
        f.write(f"min_loss = {min_loss:.8e} at step {min_loss_step}\n")
        f.write(f"MAE_ground = {mae_ground_ev:.6f} eV\n")
        f.write(f"MAE_excited1 = {mae_excited_ev:.6f} eV\n")
        f.write(f"train_wall_time_s = {elapsed_s:.2f}\n")
        f.write(f"checkpoint = {checkpoint_path}\n")
        f.write(f"checkpoint_meta = {checkpoint_meta_path}\n")


def _make_progress_logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w")

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train Neural_xc on H2 reference (CISD/FCI) ground-state dissociation "
            "and compare first excited-state dissociation."
        )
    )
    parser.add_argument(
        "--reference-method",
        type=str,
        choices=("cisd", "fci"),
        default="cisd",
    )
    parser.add_argument("--basis", type=str, default="sto-3g")
    parser.add_argument("--xc-ref", type=str, default="b3lyp")
    parser.add_argument("--r-min", type=float, default=0.45)
    parser.add_argument("--r-max", type=float, default=3.00)
    parser.add_argument("--points", type=int, default=41)
    parser.add_argument("--train-points", type=int, default=5)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
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
    parser.add_argument("--density-constraint-weight", type=float, default=1e-3)
    parser.add_argument(
        "--s1-weight",
        type=float,
        default=0.0,
        help="Weight of S1 excitation-energy MSE in the training loss.",
    )
    parser.add_argument(
        "--s1-use-tda",
        action="store_true",
        help="Use TDA (instead of Casida) for the S1 constraint during training.",
    )
    parser.add_argument("--energy-mse-weight", type=float, default=1.0)
    parser.add_argument("--energy-mae-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--params-in", type=str, default="")
    parser.add_argument("--params-out", type=str, default="")
    parser.add_argument("--log-file", type=str, default="")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--scan-log-every", type=int, default=5)
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/h2_cisd_ground_train_excited_compare",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / ".mplconfig").mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file) if args.log_file.strip() else (outdir / "run.log")
    params_out_path = (
        Path(args.params_out)
        if args.params_out.strip()
        else (outdir / "neural_xc_params.msgpack")
    )
    log, log_handle = _make_progress_logger(log_path)

    try:
        hidden_dims = tuple(
            int(x.strip()) for x in args.hidden_dims.split(",") if x.strip()
        )
        semilocal_xc = _normalize_semilocal_arg(args.semilocal_xc)
        reference_method = args.reference_method.lower()
        reference_label = reference_method.upper()
        r_grid = np.linspace(args.r_min, args.r_max, args.points)

        if args.train_points < 2:
            raise ValueError("--train-points must be >= 2.")
        if args.train_points > args.points:
            raise ValueError("--train-points must be <= --points.")

        log(
            "Config: "
            f"reference_method={reference_method}, "
            f"basis={args.basis}, xc_ref={args.xc_ref}, points={args.points}, "
            f"train_points={args.train_points}, steps={args.steps}, "
            f"lr={args.learning_rate}, s1_weight={args.s1_weight}, "
            f"s1_use_tda={bool(args.s1_use_tda)}, "
            f"energy_mse_weight={args.energy_mse_weight}, "
            f"energy_mae_weight={args.energy_mae_weight}, "
            f"seed={args.seed}, log_every={args.log_every}, "
            f"scan_log_every={args.scan_log_every}"
        )
        log(f"Writing progress log to {log_path}")
        log(f"Checkpoint output path: {params_out_path}")
        log(f"Building {reference_label} + orbital references on full R grid...")

        references = []
        e0_ref = np.zeros_like(r_grid)
        e1_ref = np.zeros_like(r_grid)
        for i, r in enumerate(r_grid):
            mol = _build_h2_mol(float(r), basis=args.basis)
            if reference_method == "fci":
                g, ex1 = _compute_fci_ground_excited(mol, nroots=3)
            else:
                g, ex1 = _compute_cisd_ground_excited(mol, nroots=3)
            ref = _build_reference_from_rks(mol, xc=args.xc_ref)
            references.append(ref)
            e0_ref[i] = g
            e1_ref[i] = ex1
            if (
                i == 0
                or (i + 1) == args.points
                or ((i + 1) % max(1, args.scan_log_every) == 0)
            ):
                log(
                    f"[ref] {i + 1:3d}/{args.points} "
                    f"R={float(r):.4f} A "
                    f"E0_{reference_label}={g:.8f} Eh E1_{reference_label}={ex1:.8f} Eh"
                )

        train_indices = np.linspace(
            0, args.points - 1, args.train_points, dtype=int
        ).tolist()
        log(f"Training indices: {train_indices}")

        functional = neural_xc.Functional(
            semilocal_xc=semilocal_xc,
            n_semilocal_channels=args.n_semilocal_channels,
            hf_input_mode=args.hf_input_mode,
            coefficient_positivity=args.coefficient_positivity,
            hidden_dims=hidden_dims,
            name=f"neural_xc_h2_{reference_method}_fit",
        )
        train_data = [
            GroundStateDatum(
                molecule=references[idx],
                target_total_energy=jnp.asarray(float(e0_ref[idx])),
                density_constraint_weight=args.density_constraint_weight,
                target_s1_energy=jnp.asarray(float(e1_ref[idx] - e0_ref[idx])),
                s1_constraint_weight=float(args.s1_weight),
            )
            for idx in train_indices
        ]
        gs_cfg = GroundStateTrainingConfig(
            mode="fixed_density",
            s1_constraint_use_tda=bool(args.s1_use_tda),
            energy_mse_weight=float(args.energy_mse_weight),
            energy_mae_weight=float(args.energy_mae_weight),
        )
        train_state = create_train_state_from_molecule(
            functional,
            jax.random.PRNGKey(args.seed),
            references[0],
            optax.adam(args.learning_rate),
        )
        param_count = int(
            sum(
                np.asarray(leaf).size
                for leaf in jax.tree_util.tree_leaves(train_state.params)
            )
        )
        log(
            "Neural_xc architecture: "
            f"hidden_dims={list(hidden_dims)}, output_dim={functional.model.output_dim}, "
            f"trainable_params={param_count}"
        )
        if args.params_in.strip():
            loaded_params = load_params_checkpoint(
                args.params_in,
                template=train_state.params,
            )
            train_state = train_state.replace(params=loaded_params)
            log(f"Loaded params checkpoint: {args.params_in}")
        train_step = make_ground_state_train_step(functional, training_config=gs_cfg)

        loss_hist: list[float] = []
        log(f"Starting Neural_xc training on {reference_label} ground-state points...")
        t0 = time.perf_counter()
        for step in range(1, args.steps + 1):
            train_state, metrics = train_step(train_state, train_data)
            loss = float(metrics["loss"])
            loss_hist.append(loss)
            if step == 1 or step == args.steps or (step % max(1, args.log_every) == 0):
                log(f"[train] step={step:4d}/{args.steps} loss={loss:.6e}")
        elapsed = time.perf_counter() - t0

        final_loss, _ = ground_state_mse_loss(
            train_state.params,
            functional,
            train_data,
            training_config=gs_cfg,
        )
        final_loss = float(final_loss)
        if loss_hist:
            min_loss = float(np.min(loss_hist))
            min_loss_step = int(np.argmin(loss_hist) + 1)
        else:
            min_loss = final_loss
            min_loss_step = 0

        ckpt_metadata = {
            "model_family": "Neural_xc",
            "functional_name": functional.name,
            "reference_method": reference_method,
            "semilocal_xc": semilocal_xc,
            "n_semilocal_channels": args.n_semilocal_channels,
            "hf_input_mode": args.hf_input_mode,
            "coefficient_positivity": args.coefficient_positivity,
            "hidden_dims": list(hidden_dims),
            "activation": "tanh",
            "density_floor": float(functional.density_floor),
            "kernel_clip": float(functional.kernel_clip),
            "output_dim": int(functional.model.output_dim),
            "optimizer": "adam",
            "learning_rate": float(args.learning_rate),
            "density_constraint_weight": float(args.density_constraint_weight),
            "s1_weight": float(args.s1_weight),
            "s1_use_tda": bool(args.s1_use_tda),
            "energy_mse_weight": float(args.energy_mse_weight),
            "energy_mae_weight": float(args.energy_mae_weight),
            "seed": int(args.seed),
            "basis": args.basis,
            "xc_ref": args.xc_ref,
            "r_min": float(args.r_min),
            "r_max": float(args.r_max),
            "points": int(args.points),
            "train_points": int(args.train_points),
            "train_indices": train_indices,
            "steps": int(args.steps),
            "loaded_from": args.params_in or "",
        }
        ckpt_path, ckpt_meta_path = save_params_checkpoint(
            params_out_path,
            train_state.params,
            metadata=ckpt_metadata,
        )

        log("Evaluating Neural_xc ground/excited dissociation curves...")
        e0_pred = np.zeros_like(r_grid)
        e1_pred = np.zeros_like(r_grid)
        for i, ref in enumerate(references):
            e0_i = float(
                predict_ground_state_total_energy(
                    train_state.params,
                    functional,
                    ref,
                    training_config=gs_cfg,
                )
            )
            td = tdscf.TDDFT(
                ref,
                xc_functional=functional,
                xc_params=train_state.params,
            )
            try:
                result = td.kernel(nstates=3)
            except Exception:
                result = tdscf.TDA(
                    ref,
                    xc_functional=functional,
                    xc_params=train_state.params,
                ).kernel(nstates=3)
            if result.excitation_energies.size == 0:
                result = tdscf.TDA(
                    ref,
                    xc_functional=functional,
                    xc_params=train_state.params,
                ).kernel(nstates=3)
            if result.excitation_energies.size == 0:
                raise RuntimeError("No excited states returned by Neural_xc TDDFT.")
            omega1 = float(result.excitation_energies[0])
            e0_pred[i] = e0_i
            e1_pred[i] = e0_i + omega1
            if (
                i == 0
                or (i + 1) == args.points
                or ((i + 1) % max(1, args.scan_log_every) == 0)
            ):
                log(
                    f"[eval] {i + 1:3d}/{args.points} "
                    f"R={float(r_grid[i]):.4f} A "
                    f"E0_pred={e0_i:.8f} Eh omega1={omega1:.8f} Eh"
                )

        mae_ground_ev = float(np.mean(np.abs(e0_pred - e0_ref)) * HARTREE_TO_EV)
        mae_excited_ev = float(np.mean(np.abs(e1_pred - e1_ref)) * HARTREE_TO_EV)

        csv_path = outdir / f"h2_{reference_method}_vs_neural_curve.csv"
        png_path = outdir / f"h2_{reference_method}_vs_neural_curve.png"
        loss_png = outdir / "training_loss.png"
        summary_path = outdir / "summary.txt"

        _write_curve_csv(
            csv_path,
            r_grid,
            e0_ref,
            e0_pred,
            e1_ref,
            e1_pred,
            reference_label=reference_label,
        )
        _plot_curves(
            png_path,
            r_grid,
            e0_ref,
            e0_pred,
            e1_ref,
            e1_pred,
            train_r=r_grid[train_indices],
            train_e0=e0_ref[train_indices],
            reference_label=reference_label,
            title=f"H2 {reference_label} vs Neural_xc Dissociation ({args.basis})",
        )
        _write_summary(
            summary_path,
            basis=args.basis,
            xc_ref=args.xc_ref,
            reference_method=reference_method,
            semilocal_xc=semilocal_xc,
            hidden_dims=hidden_dims,
            activation="tanh",
            density_floor=float(functional.density_floor),
            kernel_clip=float(functional.kernel_clip),
            density_constraint_weight=args.density_constraint_weight,
            s1_weight=float(args.s1_weight),
            s1_use_tda=bool(args.s1_use_tda),
            energy_mse_weight=float(args.energy_mse_weight),
            energy_mae_weight=float(args.energy_mae_weight),
            optimizer="adam",
            seed=args.seed,
            steps=args.steps,
            learning_rate=args.learning_rate,
            train_indices=train_indices,
            final_loss=final_loss,
            min_loss=min_loss,
            min_loss_step=min_loss_step,
            mae_ground_ev=mae_ground_ev,
            mae_excited_ev=mae_excited_ev,
            elapsed_s=elapsed,
            checkpoint_path=str(ckpt_path),
            checkpoint_meta_path=str(ckpt_meta_path) if ckpt_meta_path is not None else "",
        )

        plt.figure(figsize=(6.6, 4.4))
        plt.plot(np.arange(1, args.steps + 1), np.asarray(loss_hist), lw=1.8)
        plt.yscale("log")
        plt.xlabel("Step")
        plt.ylabel("Training Loss")
        plt.title(f"H2 {reference_label} Ground-Curve Training")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(loss_png, dpi=220)
        plt.close()

        log("")
        log(f"Final loss: {final_loss:.6e}")
        log(f"Min loss: {min_loss:.6e} @ step {min_loss_step}")
        log(f"Ground MAE: {mae_ground_ev:.6f} eV")
        log(f"Excited MAE: {mae_excited_ev:.6f} eV")
        log(f"Wrote curve csv: {csv_path}")
        log(f"Wrote curve png: {png_path}")
        log(f"Wrote loss  png: {loss_png}")
        log(f"Wrote summary : {summary_path}")
        log(f"Wrote params : {ckpt_path}")
        if ckpt_meta_path is not None:
            log(f"Wrote params meta: {ckpt_meta_path}")
    finally:
        log_handle.close()


if __name__ == "__main__":
    import jax

    main()
