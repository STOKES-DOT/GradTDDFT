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
import matplotlib.pyplot as plt
import numpy as np
import optax

from td_graddft import neural_xc, tdscf
from td_graddft.jax_libxc import b3lyp_component_basis
from td_graddft.reference_legacy import restricted_reference_from_pyscf
from td_graddft.spectra import HARTREE_TO_EV
from td_graddft.training import (
    create_train_state_from_molecule,
    load_params_checkpoint,
    predict_ground_state_total_energy,
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
        raise RuntimeError("RKS did not converge while building evaluation references.")
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
        raise RuntimeError("No excited states returned during dense curve evaluation.")
    return float(result.excitation_energies[0])


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
    reference_label: str,
    train_r: np.ndarray | None = None,
    train_e0: np.ndarray | None = None,
    train_gap_ev: np.ndarray | None = None,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))

    axes[0].plot(r_grid, e0_ref, lw=2.4, color="black", label=f"{reference_label} E0")
    for name, values in evals.items():
        axes[0].plot(r_grid, values["e0"], lw=1.8, label=name)
    if train_r is not None and train_e0 is not None:
        axes[0].scatter(
            train_r,
            train_e0,
            s=38,
            marker="o",
            facecolors="none",
            edgecolors="black",
            linewidths=1.0,
            label="train points",
            zorder=4,
        )
    axes[0].set_xlabel("H-H Distance (Angstrom)")
    axes[0].set_ylabel("Total Energy (Hartree)")
    axes[0].set_title("Ground-State Curve")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(r_grid, e1_ref, lw=2.4, color="black", label=f"{reference_label} E1")
    for name, values in evals.items():
        axes[1].plot(r_grid, values["e1"], lw=1.8, label=name)
    axes[1].set_xlabel("H-H Distance (Angstrom)")
    axes[1].set_ylabel("Total Energy (Hartree)")
    axes[1].set_title("First-Excited Curve")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    gap_ref_ev = (e1_ref - e0_ref) * HARTREE_TO_EV
    axes[2].plot(r_grid, gap_ref_ev, lw=2.4, color="black", label=f"{reference_label} Gap")
    for name, values in evals.items():
        axes[2].plot(r_grid, values["gap_ev"], lw=1.8, label=name)
    if train_r is not None and train_gap_ev is not None:
        axes[2].scatter(
            train_r,
            train_gap_ev,
            s=38,
            marker="o",
            facecolors="none",
            edgecolors="black",
            linewidths=1.0,
            label="train points",
            zorder=4,
        )
    axes[2].set_xlabel("H-H Distance (Angstrom)")
    axes[2].set_ylabel("Excitation Energy (eV)")
    axes[2].set_title("S1 Gap Curve")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle("Dense H2 Dissociation Inference from Trained Neural_xc Checkpoints")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    semilocal_xc: str | tuple[str, ...],
    metrics: dict[str, dict[str, float]],
    elapsed_s: float,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("Dense H2 dissociation inference from existing checkpoints\n")
        f.write(f"checkpoint_root = {args.checkpoint_root}\n")
        f.write(f"reference_method = {args.reference_method}\n")
        f.write(f"basis = {args.basis}\n")
        f.write(f"xc_ref = {args.xc_ref}\n")
        f.write(f"semilocal_xc = {semilocal_xc}\n")
        f.write(f"hidden_dims = {args.hidden_dims}\n")
        f.write(f"hf_input_mode = {args.hf_input_mode}\n")
        f.write(f"coefficient_positivity = {args.coefficient_positivity}\n")
        f.write(f"r_min = {args.r_min}\n")
        f.write(f"r_max = {args.r_max}\n")
        f.write(f"eval_points = {args.eval_points}\n")
        f.write(f"eval_use_tda = {bool(args.eval_use_tda)}\n")
        f.write(f"elapsed_s = {elapsed_s:.2f}\n")
        f.write("\n")
        for name, objective_metrics in metrics.items():
            f.write(f"[{name}]\n")
            f.write(f"mae_ground_ev = {objective_metrics['mae_ground_ev']:.6f}\n")
            f.write(f"mae_excited_ev = {objective_metrics['mae_excited_ev']:.6f}\n")
            f.write(f"mae_gap_ev = {objective_metrics['mae_gap_ev']:.6f}\n")
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained H2 three-loss Neural_xc checkpoints on a dense dissociation grid."
    )
    parser.add_argument("--checkpoint-root", type=str, required=True)
    parser.add_argument("--reference-method", choices=("cisd", "fci"), default="fci")
    parser.add_argument("--basis", type=str, default="sto-3g")
    parser.add_argument("--xc-ref", type=str, default="b3lyp")
    parser.add_argument("--r-min", type=float, default=0.05)
    parser.add_argument("--r-max", type=float, default=5.0)
    parser.add_argument("--eval-points", type=int, default=101)
    parser.add_argument("--train-r-min", type=float, default=None)
    parser.add_argument("--train-r-max", type=float, default=None)
    parser.add_argument("--train-points", type=int, default=None)
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
    parser.add_argument("--hidden-dims", type=str, default="64,64")
    parser.add_argument(
        "--eval-use-tda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use TDA during dense inference. Defaults to the same TDA path used by the S1 loss.",
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=["ground_only", "ground_plus_s1", "s1_only"],
    )
    parser.add_argument("--scan-log-every", type=int, default=20)
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/h2_three_loss_fci_dense_eval",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log, log_handle = _make_progress_logger(outdir / "run.log")
    try:
        semilocal_xc = _normalize_semilocal_arg(args.semilocal_xc)
        checkpoint_root = Path(args.checkpoint_root)
        if not checkpoint_root.exists():
            raise FileNotFoundError(f"checkpoint_root does not exist: {checkpoint_root}")

        reference_label = str(args.reference_method).upper()
        r_grid = np.linspace(float(args.r_min), float(args.r_max), int(args.eval_points))
        log(
            "Dense inference config: "
            f"checkpoint_root={checkpoint_root}, reference_method={args.reference_method}, "
            f"basis={args.basis}, xc_ref={args.xc_ref}, R=[{args.r_min},{args.r_max}], "
            f"eval_points={args.eval_points}, eval_use_tda={bool(args.eval_use_tda)}"
        )
        log("Building dense reference dataset...")

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

        functional = neural_xc.Functional(
            semilocal_xc=semilocal_xc,
            n_semilocal_channels=args.n_semilocal_channels,
            hf_input_mode=args.hf_input_mode,
            coefficient_positivity=args.coefficient_positivity,
            hidden_dims=tuple(int(x.strip()) for x in args.hidden_dims.split(",") if x.strip()),
            name="neural_xc_h2_dense_eval",
        )
        template_state = create_train_state_from_molecule(
            functional,
            jax.random.PRNGKey(0),
            references[0],
            optax.adam(1e-3),
        )

        evals: dict[str, dict[str, np.ndarray]] = {}
        metrics: dict[str, dict[str, float]] = {}
        t0 = time.perf_counter()
        for name in args.objectives:
            ckpt_path = checkpoint_root / name / "neural_xc_params.msgpack"
            params = load_params_checkpoint(ckpt_path, template=template_state.params)
            e0_pred = np.zeros_like(r_grid)
            e1_pred = np.zeros_like(r_grid)
            gap_ev = np.zeros_like(r_grid)
            for idx, ref in enumerate(references):
                e0_i = float(predict_ground_state_total_energy(params, functional, ref))
                omega1_au = _predict_first_excitation_au(
                    params,
                    functional,
                    ref,
                    use_tda=bool(args.eval_use_tda),
                )
                e0_pred[idx] = e0_i
                e1_pred[idx] = e0_i + omega1_au
                gap_ev[idx] = omega1_au * HARTREE_TO_EV
            evals[name] = {
                "e0": e0_pred,
                "e1": e1_pred,
                "gap_ev": gap_ev,
            }
            metrics[name] = {
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
                f"[eval:{name}] "
                f"MAE_ground={metrics[name]['mae_ground_ev']:.6f} eV "
                f"MAE_excited={metrics[name]['mae_excited_ev']:.6f} eV "
                f"MAE_gap={metrics[name]['mae_gap_ev']:.6f} eV"
            )
        elapsed = time.perf_counter() - t0

        train_r = None
        train_e0 = None
        train_gap_ev = None
        if (
            args.train_r_min is not None
            and args.train_r_max is not None
            and args.train_points is not None
            and int(args.train_points) >= 2
        ):
            train_r = np.linspace(float(args.train_r_min), float(args.train_r_max), int(args.train_points))
            train_e0 = np.zeros_like(train_r)
            train_gap_ev = np.zeros_like(train_r)
            for idx, r in enumerate(train_r):
                mol = _build_h2_mol(float(r), basis=str(args.basis))
                if str(args.reference_method).lower() == "fci":
                    g, ex1 = _compute_fci_ground_excited(mol, nroots=3)
                else:
                    g, ex1 = _compute_cisd_ground_excited(mol, nroots=3)
                train_e0[idx] = g
                train_gap_ev[idx] = (ex1 - g) * HARTREE_TO_EV

        curve_csv = outdir / "dense_dissociation_curves.csv"
        curve_png = outdir / "dense_dissociation_curves.png"
        summary_path = outdir / "summary.txt"
        _write_curve_csv(curve_csv, r_grid, e0_ref, e1_ref, evals)
        _plot_curve_comparison(
            curve_png,
            r_grid=r_grid,
            e0_ref=e0_ref,
            e1_ref=e1_ref,
            evals=evals,
            reference_label=reference_label,
            train_r=train_r,
            train_e0=train_e0,
            train_gap_ev=train_gap_ev,
        )
        _write_summary(
            summary_path,
            args=args,
            semilocal_xc=semilocal_xc,
            metrics=metrics,
            elapsed_s=elapsed,
        )
        log(f"Wrote curve csv  : {curve_csv}")
        log(f"Wrote curve png  : {curve_png}")
        log(f"Wrote summary    : {summary_path}")
    finally:
        log_handle.close()


if __name__ == "__main__":
    main()
