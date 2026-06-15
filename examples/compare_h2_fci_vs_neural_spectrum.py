from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
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
from pyscf import ao2mo, dft, fci, gto, scf

from td_graddft import neural_xc, tdscf
from td_graddft.data.reference import restricted_reference_from_pyscf
from td_graddft.scf import DifferentiableSCF, DifferentiableSCFConfig
from td_graddft.spectra import HARTREE_TO_EV, lorentzian_spectrum
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Neural_xc on a single H2 geometry using an FCI ground-state target "
            "and compare H2 absorption spectra against FCI and PySCF TDDFT."
        )
    )
    parser.add_argument("--bond-length", type=float, default=0.74, help="H-H distance in Angstrom")
    parser.add_argument("--basis", type=str, default="6-31g", help="AO basis for H2")
    parser.add_argument("--xc-ref", type=str, default="b3lyp", help="reference orbital functional")
    parser.add_argument(
        "--semilocal-xc",
        type=str,
        default="b3lyp",
        help="semilocal XC basis passed to Neural_xc",
    )
    parser.add_argument("--states", type=int, default=5, help="number of excited states to compare")
    parser.add_argument("--eta-ev", type=float, default=0.20, help="Lorentzian width in eV")
    parser.add_argument(
        "--spectrum-max-ev",
        type=float,
        default=None,
        help="optional maximum energy on the plotted spectrum grid",
    )
    parser.add_argument(
        "--annotate-states",
        type=int,
        default=5,
        help="number of low-lying FCI states to annotate on the spectrum plot",
    )
    parser.add_argument("--steps", type=int, default=200, help="training steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--lr-decay-every", type=int, default=50, help="decay interval in steps")
    parser.add_argument("--lr-decay-factor", type=float, default=0.5, help="decay factor")
    parser.add_argument("--log-interval", type=int, default=20, help="print every N steps")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[64, 64, 64],
        help="Neural_xc hidden dimensions",
    )
    parser.add_argument(
        "--excitation-weight",
        type=float,
        default=0.0,
        help="optional FCI excitation-energy supervision weight",
    )
    parser.add_argument(
        "--excitation-nstates",
        type=int,
        default=3,
        help="number of FCI excitation energies supervised if enabled",
    )
    parser.add_argument(
        "--oscillator-strength-weight",
        type=float,
        default=0.0,
        help="optional FCI oscillator-strength supervision weight",
    )
    parser.add_argument(
        "--oscillator-strength-nstates",
        type=int,
        default=3,
        help="number of FCI oscillator strengths supervised if enabled",
    )
    parser.add_argument(
        "--train-use-tda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use TDA for optional excited-state supervision terms",
    )
    parser.add_argument(
        "--eval-use-tda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use TDA instead of Casida for B3LYP/Neural evaluation",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="h2_fci_spectrum_singlepoint",
        help="output directory prefix under outputs/",
    )
    parser.add_argument(
        "--params-in",
        type=str,
        default="",
        help="optional checkpoint path; if provided and --steps <= 0, skip training and only evaluate",
    )
    return parser.parse_args()


def _build_h2_mol(r_angstrom: float, basis: str) -> gto.Mole:
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


def _build_rks_reference(
    mol: gto.Mole,
    *,
    xc: str,
):
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"RKS did not converge for H2 {xc}/{mol.basis}.")
    return mf, restricted_reference_from_pyscf(
        mf,
        compute_local_hfx_features=True,
        hfx_omega_values=(0.0, 0.4),
        hfx_chunk_size=256,
    )


def _compute_fci_spectrum(
    mol: gto.Mole,
    *,
    nstates: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge for FCI spectrum generation.")

    h1_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri_mo = ao2mo.kernel(mol, mf.mo_coeff)
    norb = h1_mo.shape[0]
    nelec = mol.nelectron

    solver = fci.direct_spin0.FCI(mol)
    nroots = max(2, int(nstates) + 1)
    e_roots, ci_roots = solver.kernel(h1_mo, eri_mo, norb, nelec, nroots=nroots)
    e_roots = np.asarray(e_roots, dtype=float).reshape(-1)
    if e_roots.size < 2:
        raise RuntimeError("FCI did not return any excited states for H2.")

    ground_total = float(e_roots[0] + mol.energy_nuc())
    dipole_ao = -mol.intor_symmetric("int1e_r", comp=3)
    dipole_mo = np.einsum("xuv,up,vq->xpq", dipole_ao, mf.mo_coeff, mf.mo_coeff)

    n_compare = min(int(nstates), int(e_roots.size - 1))
    excitation_energies = np.zeros((n_compare,), dtype=float)
    oscillator = np.zeros((n_compare,), dtype=float)
    for idx in range(n_compare):
        root = idx + 1
        excitation_energies[idx] = float(e_roots[root] - e_roots[0])
        tdm1 = fci.direct_spin0.trans_rdm1(ci_roots[0], ci_roots[root], norb, nelec)
        mu = np.einsum("xpq,qp->x", dipole_mo, np.asarray(tdm1, dtype=float))
        oscillator[idx] = float((2.0 / 3.0) * excitation_energies[idx] * np.dot(mu, mu))
    return ground_total, excitation_energies, oscillator


def _compute_pyscf_spectrum(
    mf,
    *,
    nstates: int,
    use_tda: bool,
) -> tuple[np.ndarray, np.ndarray]:
    td = mf.TDA() if use_tda else mf.TDDFT()
    td.nstates = max(1, int(nstates))
    td.kernel()
    energies = np.asarray(td.e, dtype=float).reshape(-1)
    strengths = np.asarray(td.oscillator_strength(), dtype=float).reshape(-1)
    n = min(int(nstates), energies.size, strengths.size)
    return energies[:n], strengths[:n]


def _compute_neural_spectrum(
    reference_molecule,
    *,
    functional,
    params,
    nstates: int,
    use_tda: bool,
) -> tuple[object, np.ndarray, np.ndarray, dict[str, float]]:
    scf_driver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            max_cycle=80,
            damping=0.15,
            conv_tol_density=1e-9,
            vxc_clip=20.0,
            iterate_selection="final",
        )
    )
    neural_molecule, scf_info = scf_driver.run(reference_molecule, functional, params)
    td = tdscf.TDA(
        neural_molecule,
        xc_functional=functional,
        xc_params=params,
    ) if use_tda else tdscf.TDDFT(
        neural_molecule,
        xc_functional=functional,
        xc_params=params,
    )
    try:
        result = td.kernel(nstates=nstates)
    except Exception:
        td = tdscf.TDA(
            neural_molecule,
            xc_functional=functional,
            xc_params=params,
        )
        result = td.kernel(nstates=nstates)
    if int(result.excitation_energies.size) == 0:
        td = tdscf.TDA(
            neural_molecule,
            xc_functional=functional,
            xc_params=params,
        )
        result = td.kernel(nstates=nstates)
    energies = np.asarray(result.excitation_energies, dtype=float).reshape(-1)
    strengths = np.asarray(td.oscillator_strength(), dtype=float).reshape(-1)
    n = min(int(nstates), energies.size, strengths.size)
    info = {
        "converged": float(np.asarray(scf_info.converged)),
        "cycles": float(np.asarray(scf_info.cycles)),
        "final_rms_density": float(np.asarray(scf_info.final_rms_density)),
    }
    return neural_molecule, energies[:n], strengths[:n], info


def _state_mae(ref: np.ndarray, pred: np.ndarray, *, scale: float = 1.0) -> float:
    n = min(ref.size, pred.size)
    if n <= 0:
        return float("nan")
    return float(np.mean(np.abs(pred[:n] - ref[:n])) * scale)


def _write_state_csv(
    path: Path,
    *,
    fci_energies_au: np.ndarray,
    fci_osc: np.ndarray,
    b3lyp_energies_au: np.ndarray,
    b3lyp_osc: np.ndarray,
    neural_energies_au: np.ndarray,
    neural_osc: np.ndarray,
) -> None:
    n = min(
        fci_energies_au.size,
        fci_osc.size,
        b3lyp_energies_au.size,
        b3lyp_osc.size,
        neural_energies_au.size,
        neural_osc.size,
    )
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "state",
                "fci_energy_eV",
                "fci_oscillator_strength",
                "b3lyp_energy_eV",
                "b3lyp_oscillator_strength",
                "neural_energy_eV",
                "neural_oscillator_strength",
            ]
        )
        for idx in range(n):
            writer.writerow(
                [
                    idx + 1,
                    float(fci_energies_au[idx] * HARTREE_TO_EV),
                    float(fci_osc[idx]),
                    float(b3lyp_energies_au[idx] * HARTREE_TO_EV),
                    float(b3lyp_osc[idx]),
                    float(neural_energies_au[idx] * HARTREE_TO_EV),
                    float(neural_osc[idx]),
                ]
            )


def _write_training_curve(path: Path, loss_history: list[float]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "loss"])
        for idx, loss in enumerate(loss_history, start=1):
            writer.writerow([idx, float(loss)])


def _plot_training_curve(path: Path, loss_history: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(np.arange(1, len(loss_history) + 1), np.asarray(loss_history), lw=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.set_title("H2 FCI Ground-State Overfit")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_spectrum(
    path: Path,
    *,
    fci_energies_au: np.ndarray,
    fci_osc: np.ndarray,
    b3lyp_energies_au: np.ndarray,
    b3lyp_osc: np.ndarray,
    neural_energies_au: np.ndarray,
    neural_osc: np.ndarray,
    eta_ev: float,
    title: str,
    annotate_states: int = 5,
    spectrum_max_ev: float | None = None,
) -> None:
    emax_ev = max(
        1.0,
        float(np.max(fci_energies_au) * HARTREE_TO_EV) if fci_energies_au.size else 0.0,
        float(np.max(b3lyp_energies_au) * HARTREE_TO_EV) if b3lyp_energies_au.size else 0.0,
        float(np.max(neural_energies_au) * HARTREE_TO_EV) if neural_energies_au.size else 0.0,
    )
    upper_ev = float(spectrum_max_ev) if spectrum_max_ev is not None else (emax_ev + 3.0)
    grid_ev = np.linspace(0.0, max(upper_ev, emax_ev + 0.5), 2400)
    fci_curve = np.asarray(
        lorentzian_spectrum(
            jnp.asarray(fci_energies_au * HARTREE_TO_EV),
            jnp.asarray(fci_osc),
            jnp.asarray(grid_ev),
            eta=float(eta_ev),
        )
    )
    b3lyp_curve = np.asarray(
        lorentzian_spectrum(
            jnp.asarray(b3lyp_energies_au * HARTREE_TO_EV),
            jnp.asarray(b3lyp_osc),
            jnp.asarray(grid_ev),
            eta=float(eta_ev),
        )
    )
    neural_curve = np.asarray(
        lorentzian_spectrum(
            jnp.asarray(neural_energies_au * HARTREE_TO_EV),
            jnp.asarray(neural_osc),
            jnp.asarray(grid_ev),
            eta=float(eta_ev),
        )
    )
    ymax = max(
        1e-6,
        float(np.max(fci_curve)) if fci_curve.size else 0.0,
        float(np.max(b3lyp_curve)) if b3lyp_curve.size else 0.0,
        float(np.max(neural_curve)) if neural_curve.size else 0.0,
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(grid_ev, fci_curve, lw=2.2, color="#1f77b4", label="FCI")
    ax.plot(grid_ev, b3lyp_curve, lw=1.9, color="#2ca02c", label="B3LYP TDDFT")
    ax.plot(grid_ev, neural_curve, lw=1.9, color="#d62728", label="Neural_xc TDDFT")
    for energy_ev, strength in zip(fci_energies_au * HARTREE_TO_EV, fci_osc, strict=True):
        ax.vlines(float(energy_ev), 0.0, float(strength), colors="#1f77b4", alpha=0.25, lw=1.0)
    for energy_ev, strength in zip(
        b3lyp_energies_au * HARTREE_TO_EV,
        b3lyp_osc,
        strict=True,
    ):
        ax.vlines(float(energy_ev), 0.0, float(strength), colors="#2ca02c", alpha=0.18, lw=0.9)
    for energy_ev, strength in zip(
        neural_energies_au * HARTREE_TO_EV,
        neural_osc,
        strict=True,
    ):
        ax.vlines(float(energy_ev), 0.0, float(strength), colors="#d62728", alpha=0.18, lw=0.9)
    ax.set_xlabel("Excitation Energy (eV)")
    ax.set_ylabel("Absorption Intensity (arb. u.)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    n_annotate = min(int(annotate_states), int(fci_energies_au.size), int(fci_osc.size))
    for idx in range(n_annotate):
        energy_ev = float(fci_energies_au[idx] * HARTREE_TO_EV)
        strength = float(fci_osc[idx])
        arrow_y = max(strength, 0.06 * ymax)
        text_y = ymax * (0.94 - 0.10 * (idx % 2))
        label = f"S{idx + 1}\n{energy_ev:.2f} eV"
        ax.annotate(
            label,
            xy=(energy_ev, arrow_y),
            xytext=(energy_ev, text_y),
            textcoords="data",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color="#1f77b4",
            arrowprops={
                "arrowstyle": "-",
                "color": "#1f77b4",
                "lw": 0.8,
                "alpha": 0.75,
            },
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "#1f77b4",
                "alpha": 0.65,
                "linewidth": 0.6,
            },
        )

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path("outputs") / args.prefix
    outdir.mkdir(parents=True, exist_ok=True)

    mol = _build_h2_mol(float(args.bond_length), basis=str(args.basis))
    mf, reference = _build_rks_reference(mol, xc=str(args.xc_ref))

    fci_ground_total, fci_energies_au, fci_osc = _compute_fci_spectrum(
        mol,
        nstates=int(args.states),
    )
    b3lyp_energies_au, b3lyp_osc = _compute_pyscf_spectrum(
        mf,
        nstates=int(args.states),
        use_tda=bool(args.eval_use_tda),
    )

    functional = neural_xc.Functional(
        semilocal_xc=str(args.semilocal_xc),
        hidden_dims=tuple(int(v) for v in args.hidden_dims),
        architecture="residual",
        energy_mode="graddft_coeff_basis",
        input_feature_mode="dm21_original",
        hf_input_mode="spin_resolved",
        response_hf_mode="nonlocal_exchange_only",
        coefficient_positivity="clip",
        strict_dm21_feature_alignment=True,
        name="h2_fci_neural_xc_fit",
    )

    datum = GroundStateDatum(
        molecule=reference,
        target_total_energy=jnp.asarray(fci_ground_total),
        target_excitation_energies=jnp.asarray(fci_energies_au[: int(args.excitation_nstates)]),
        excitation_constraint_weight=float(args.excitation_weight),
        excitation_constraint_nstates=int(args.excitation_nstates),
        target_oscillator_strengths=jnp.asarray(
            fci_osc[: int(args.oscillator_strength_nstates)]
        ),
        oscillator_strength_constraint_weight=float(args.oscillator_strength_weight),
        oscillator_strength_constraint_nstates=int(args.oscillator_strength_nstates),
    )
    gs_cfg = GroundStateTrainingConfig(
        mode="fixed_density",
        energy_mse_weight=0.0,
        energy_mae_weight=1.0,
        excitation_constraint_use_tda=bool(args.train_use_tda),
        excitation_mse_weight=0.0,
        excitation_mae_weight=1.0,
        oscillator_strength_constraint_use_tda=bool(args.train_use_tda),
        oscillator_strength_mse_weight=0.0,
        oscillator_strength_mae_weight=1.0,
    )

    if int(args.lr_decay_every) > 0:
        lr_schedule = optax.exponential_decay(
            init_value=float(args.lr),
            transition_steps=int(args.lr_decay_every),
            decay_rate=float(args.lr_decay_factor),
            staircase=True,
        )
        optimizer = optax.adam(lr_schedule)
    else:
        optimizer = optax.adam(float(args.lr))

    train_state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(int(args.seed)),
        reference,
        optimizer,
    )
    train_step = make_ground_state_train_step(functional, training_config=gs_cfg)

    loss_history: list[float] = []
    best_loss = float("inf")
    best_step = 0
    best_params = train_state.params

    if str(args.params_in).strip():
        loaded = load_params_checkpoint(str(args.params_in).strip(), template=train_state.params)
        train_state = train_state.replace(params=loaded)
        best_params = loaded
        best_step = 0
        best_loss = float(
            ground_state_mse_loss(
                best_params,
                functional,
                datum,
                training_config=gs_cfg,
            )[0]
        )

    t0 = time.perf_counter()
    if int(args.steps) > 0:
        for step in range(1, int(args.steps) + 1):
            train_state, metrics = train_step(train_state, datum)
            loss = float(metrics["loss"])
            loss_history.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_step = step
                best_params = train_state.params
            if step == 1 or step == int(args.steps) or step % max(1, int(args.log_interval)) == 0:
                excitation_penalty = float(metrics.get("excitation_penalty", jnp.asarray([0.0]))[0])
                osc_penalty = float(
                    metrics.get("oscillator_strength_penalty", jnp.asarray([0.0]))[0]
                )
                print(
                    "[H2-FCI-Train] "
                    f"step={step}/{args.steps} "
                    f"loss={loss:.6e} "
                    f"excitation_penalty={excitation_penalty:.6e} "
                    f"oscillator_strength_penalty={osc_penalty:.6e}"
                )
    train_elapsed = time.perf_counter() - t0

    final_loss, _ = ground_state_mse_loss(
        best_params,
        functional,
        datum,
        training_config=gs_cfg,
    )
    final_loss = float(final_loss)

    neural_molecule, neural_energies_au, neural_osc, scf_info = _compute_neural_spectrum(
        reference,
        functional=functional,
        params=best_params,
        nstates=int(args.states),
        use_tda=bool(args.eval_use_tda),
    )

    neural_ground_energy = float(
        predict_ground_state_total_energy(
            best_params,
            functional,
            reference,
            training_config=gs_cfg,
        )
    )
    b3lyp_ground_energy = float(mf.e_tot)

    fci_vs_b3lyp_energy_mae_ev = _state_mae(fci_energies_au, b3lyp_energies_au, scale=HARTREE_TO_EV)
    fci_vs_neural_energy_mae_ev = _state_mae(fci_energies_au, neural_energies_au, scale=HARTREE_TO_EV)
    fci_vs_b3lyp_osc_mae = _state_mae(fci_osc, b3lyp_osc, scale=1.0)
    fci_vs_neural_osc_mae = _state_mae(fci_osc, neural_osc, scale=1.0)

    state_csv = outdir / "state_compare.csv"
    spectrum_png = outdir / "h2_fci_vs_b3lyp_vs_neural_spectrum.png"
    curve_csv = outdir / "training_curve.csv"
    curve_png = outdir / "training_curve.png"
    summary_txt = outdir / "summary.txt"
    ckpt_path, ckpt_meta_path = save_params_checkpoint(
        outdir / "neural_xc_h2_fci_params.npz",
        best_params,
        metadata={
            "system": "H2",
            "basis": str(args.basis),
            "bond_length_angstrom": float(args.bond_length),
            "xc_ref": str(args.xc_ref),
            "semilocal_xc": str(args.semilocal_xc),
            "steps": int(args.steps),
            "learning_rate": float(args.lr),
            "best_step": int(best_step),
            "best_loss": float(best_loss),
        },
    )

    _write_state_csv(
        state_csv,
        fci_energies_au=fci_energies_au,
        fci_osc=fci_osc,
        b3lyp_energies_au=b3lyp_energies_au,
        b3lyp_osc=b3lyp_osc,
        neural_energies_au=neural_energies_au,
        neural_osc=neural_osc,
    )
    if loss_history:
        _write_training_curve(curve_csv, loss_history)
        _plot_training_curve(curve_png, loss_history)
    _plot_spectrum(
        spectrum_png,
        fci_energies_au=fci_energies_au,
        fci_osc=fci_osc,
        b3lyp_energies_au=b3lyp_energies_au,
        b3lyp_osc=b3lyp_osc,
        neural_energies_au=neural_energies_au,
        neural_osc=neural_osc,
        eta_ev=float(args.eta_ev),
        title=f"H2 Spectrum @ {args.bond_length:.2f} A ({args.basis})",
        annotate_states=int(args.annotate_states),
        spectrum_max_ev=args.spectrum_max_ev,
    )

    with summary_txt.open("w") as handle:
        handle.write("H2 single-point FCI spectrum comparison\n")
        handle.write(f"bond_length_angstrom={float(args.bond_length):.8f}\n")
        handle.write(f"basis={args.basis}\n")
        handle.write(f"xc_ref={args.xc_ref}\n")
        handle.write(f"semilocal_xc={args.semilocal_xc}\n")
        handle.write(f"eval_use_tda={bool(args.eval_use_tda)}\n")
        handle.write(f"train_use_tda={bool(args.train_use_tda)}\n")
        handle.write(f"steps={int(args.steps)}\n")
        handle.write(f"params_in={args.params_in}\n")
        handle.write(f"spectrum_max_ev={args.spectrum_max_ev}\n")
        handle.write(f"annotate_states={int(args.annotate_states)}\n")
        handle.write(f"learning_rate={float(args.lr):.8e}\n")
        handle.write(f"best_step={int(best_step)}\n")
        handle.write(f"best_loss={float(best_loss):.12e}\n")
        handle.write(f"final_loss={float(final_loss):.12e}\n")
        handle.write(f"fci_ground_total_energy_ha={fci_ground_total:.12f}\n")
        handle.write(f"b3lyp_ground_total_energy_ha={b3lyp_ground_energy:.12f}\n")
        handle.write(f"neural_ground_total_energy_ha={neural_ground_energy:.12f}\n")
        handle.write(f"fci_vs_b3lyp_energy_mae_ev={fci_vs_b3lyp_energy_mae_ev:.12f}\n")
        handle.write(f"fci_vs_neural_energy_mae_ev={fci_vs_neural_energy_mae_ev:.12f}\n")
        handle.write(f"fci_vs_b3lyp_osc_mae={fci_vs_b3lyp_osc_mae:.12f}\n")
        handle.write(f"fci_vs_neural_osc_mae={fci_vs_neural_osc_mae:.12f}\n")
        handle.write(f"neural_scf_converged={int(scf_info['converged'])}\n")
        handle.write(f"neural_scf_cycles={scf_info['cycles']:.0f}\n")
        handle.write(f"neural_scf_final_rms_density={scf_info['final_rms_density']:.12e}\n")
        handle.write(f"train_elapsed_s={train_elapsed:.6f}\n")
        handle.write(f"state_csv={state_csv}\n")
        handle.write(f"spectrum_png={spectrum_png}\n")
        handle.write(f"training_curve_csv={curve_csv if loss_history else ''}\n")
        handle.write(f"training_curve_png={curve_png if loss_history else ''}\n")
        handle.write(f"checkpoint_path={ckpt_path}\n")
        if ckpt_meta_path is not None:
            handle.write(f"checkpoint_meta_path={ckpt_meta_path}\n")

    print(f"FCI ground total energy: {fci_ground_total:.10f} Ha")
    print(f"B3LYP ground total energy: {b3lyp_ground_energy:.10f} Ha")
    print(f"Neural ground total energy: {neural_ground_energy:.10f} Ha")
    print(f"Best loss: {best_loss:.6e} @ step {best_step}")
    print(f"FCI vs B3LYP excited-state MAE: {fci_vs_b3lyp_energy_mae_ev:.4f} eV")
    print(f"FCI vs Neural excited-state MAE: {fci_vs_neural_energy_mae_ev:.4f} eV")
    print(f"FCI vs B3LYP oscillator MAE: {fci_vs_b3lyp_osc_mae:.4f}")
    print(f"FCI vs Neural oscillator MAE: {fci_vs_neural_osc_mae:.4f}")
    print(f"Wrote state table: {state_csv}")
    print(f"Wrote spectrum plot: {spectrum_png}")
    print(f"Wrote summary: {summary_txt}")


if __name__ == "__main__":
    main()
