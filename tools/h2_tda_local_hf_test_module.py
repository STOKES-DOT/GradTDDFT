from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax

from td_graddft import neural_xc
from td_graddft.neural_xc import (
    GRADDFT_DEFAULT_DM21_HIDDEN_DIMS,
    GRADDFT_DEFAULT_INPUT_FEATURE_MODE,
    GRADDFT_DEFAULT_NETWORK_ARCHITECTURE,
)
from td_graddft.spectra import HARTREE_TO_EV
from td_graddft.tddft import RestrictedCasidaTDDFT
from td_graddft.tddft.test_module import RestrictedLocalHFKhhTDAWrapper
from td_graddft.training import create_train_state_from_molecule, load_params_checkpoint


_HELPER_PATH = Path(__file__).with_name("h2_self_consistent_ground_train5_dense100_vs_fci.py")
_HELPER_SPEC = importlib.util.spec_from_file_location("_h2_ground_vs_fci_helpers", _HELPER_PATH)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise RuntimeError(f"Failed to load helper module from {_HELPER_PATH}")
_HELPERS = importlib.util.module_from_spec(_HELPER_SPEC)
sys.modules[_HELPER_SPEC.name] = _HELPERS
_HELPER_SPEC.loader.exec_module(_HELPERS)

RunLogger = _HELPERS.RunLogger
build_reference_curve = _HELPERS.build_reference_curve

_DEFAULT_SEMILOCAL_XC = ("lda_x", "gga_x_b88", "lda_c_vwn_rpa", "gga_c_lyp")
_DEFAULT_CHECKPOINT = (
    "outputs/"
    "h2_stage2_s1_fixed_sto3g_local_ep4000_d200_log10_driver_v7_jitstage1_d400_restart1/"
    "neural_xc_params.msgpack"
)


def _get_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Experimental H2/STO-3G TDA dissociation test for the local-HF K_hh test module. "
            "Compares the production response, a local-projected K_aa-only path, and "
            "a local-projected K_aa + K_hh path."
        )
    )
    p.add_argument("--checkpoint", default=_DEFAULT_CHECKPOINT)
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--xc", default="b3lyp")
    p.add_argument("--r-min", type=float, default=0.05)
    p.add_argument("--r-max", type=float, default=5.0)
    p.add_argument("--dense-points", type=int, default=100)
    p.add_argument("--excited-nstates", type=int, default=3)
    p.add_argument("--grids-level", type=int, default=0)
    p.add_argument("--max-l", type=int, default=3)
    p.add_argument("--grid-ao-backend", choices=("jax", "pyscf"), default="jax")
    p.add_argument("--integral-backend", choices=("jax", "libcint"), default="libcint")
    p.add_argument("--jk-backend", choices=("full", "df"), default="full")
    p.add_argument("--df-tol", type=float, default=1e-10)
    p.add_argument("--df-max-rank", type=int, default=None)
    p.add_argument("--reference-scf-max-cycle", type=int, default=80)
    p.add_argument("--reference-scf-conv-tol", type=float, default=1e-10)
    p.add_argument("--reference-scf-conv-tol-density", type=float, default=1e-8)
    p.add_argument("--reference-scf-damping", type=float, default=0.15)
    p.add_argument("--reference-scf-potential-clip", type=float, default=20.0)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=list(GRADDFT_DEFAULT_DM21_HIDDEN_DIMS))
    p.add_argument(
        "--network-architecture",
        choices=("simple_mlp", "graddft_residual"),
        default=GRADDFT_DEFAULT_NETWORK_ARCHITECTURE,
    )
    p.add_argument(
        "--input-feature-mode",
        choices=("enhanced", "dm21_original"),
        default=GRADDFT_DEFAULT_INPUT_FEATURE_MODE,
    )
    p.add_argument("--include-pt2-channel", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--pt2-channel-mode",
        choices=("scaled_projected", "local_exact"),
        default="scaled_projected",
    )
    p.add_argument("--omega-index", type=int, default=0)
    p.add_argument("--fd-step", type=float, default=1e-4)
    p.add_argument(
        "--outdir",
        default="outputs/h2_tda_local_hf_test_module",
    )
    return p.parse_args()


def _load_checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    meta_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".meta.json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _merge_checkpoint_metadata(args: argparse.Namespace, meta: dict[str, Any]) -> argparse.Namespace:
    merged = argparse.Namespace(**vars(args))
    if "hidden_dims" in meta:
        merged.hidden_dims = list(meta["hidden_dims"])
    if "include_pt2_channel" in meta:
        merged.include_pt2_channel = bool(meta["include_pt2_channel"])
    if "pt2_channel_mode" in meta and meta["pt2_channel_mode"] is not None:
        merged.pt2_channel_mode = str(meta["pt2_channel_mode"])
    if "basis" in meta:
        merged.basis = str(meta["basis"])
    if "xc" in meta:
        merged.xc = str(meta["xc"])
    return merged


def _make_functional(args: argparse.Namespace, *, response_hf_mode: str):
    return neural_xc.make_neural_xc_functional(
        name=f"h2_tda_local_hf_test_module_{response_hf_mode}",
        semilocal_xc=tuple(_DEFAULT_SEMILOCAL_XC),
        input_feature_mode=str(args.input_feature_mode),
        hidden_dims=tuple(int(v) for v in args.hidden_dims),
        network_architecture=str(args.network_architecture),
        include_pt2_channel=bool(args.include_pt2_channel),
        pt2_channel_mode=str(args.pt2_channel_mode),
        response_hf_mode=str(response_hf_mode),
        response_pt2_mode="local_projected",
    )


def _load_params(functional: Any, molecule: Any, checkpoint: Path):
    template = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(0),
        molecule,
        optax.set_to_zero(),
    ).params
    return load_params_checkpoint(checkpoint, template=template)


def _predict_s1_h(molecule: Any, xc_obj: Any) -> float:
    solver = RestrictedCasidaTDDFT(molecule=molecule, xc_functional=xc_obj)
    result = solver.tda(nstates=1)
    return float(np.asarray(result.excitation_energies, dtype=float)[0])


def _plot_curves(path: Path, rows: list[dict[str, float]]) -> None:
    plt = _get_plt()
    r = np.asarray([row["r_angstrom"] for row in rows], dtype=float)
    fci = np.asarray([row["fci_s1_ev"] for row in rows], dtype=float)
    prod = np.asarray([row["production_s1_ev"] for row in rows], dtype=float)
    kaa = np.asarray([row["kaa_only_s1_ev"] for row in rows], dtype=float)
    khh = np.asarray([row["kaa_plus_khh_s1_ev"] for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 8.0), dpi=180, sharex=True)
    axes[0].plot(r, fci, lw=2.4, color="#111111", label="FCI")
    axes[0].plot(r, prod, lw=2.0, color="#0072b2", label="Production")
    axes[0].plot(r, kaa, lw=2.0, color="#d55e00", label="Local-projected Kaa only")
    axes[0].plot(r, khh, lw=2.0, color="#009e73", label="Local-projected Kaa + Khh")
    axes[0].set_ylabel("S1 Gap (eV)")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25, linestyle="--", linewidth=0.6)

    axes[1].plot(r, prod - fci, lw=2.0, color="#0072b2", label="Production - FCI")
    axes[1].plot(r, kaa - fci, lw=2.0, color="#d55e00", label="Kaa only - FCI")
    axes[1].plot(r, khh - fci, lw=2.0, color="#009e73", label="Kaa + Khh - FCI")
    axes[1].axhline(0.0, lw=1.0, color="#111111", ls=":")
    axes[1].set_ylabel("Error (eV)")
    axes[1].set_xlabel("R (Angstrom)")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25, linestyle="--", linewidth=0.6)

    fig.suptitle("H2/STO-3G TDA dissociation | experimental local-HF K_hh test module")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(outdir / "run.log")

    checkpoint = Path(str(args.checkpoint))
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    args = _merge_checkpoint_metadata(args, _load_checkpoint_metadata(checkpoint))

    logger.log(f"checkpoint={checkpoint}")
    logger.log(
        f"basis={args.basis}, xc={args.xc}, dense_points={args.dense_points}, "
        f"include_pt2_channel={bool(args.include_pt2_channel)}, pt2_channel_mode={args.pt2_channel_mode}"
    )
    _HELPERS._load_runtime_dependencies(logger)

    r_values = np.linspace(float(args.r_min), float(args.r_max), int(args.dense_points))
    reference_points = build_reference_curve(
        r_values,
        args=args,
        logger=logger,
        label="dense_ref",
    )
    if not reference_points:
        raise RuntimeError("No H2 reference points were built.")

    functional_prod = _make_functional(args, response_hf_mode="nonlocal_exchange_only")
    functional_local = _make_functional(args, response_hf_mode="local_projected")
    params = _load_params(functional_prod, reference_points[0].molecule, checkpoint)

    rows: list[dict[str, float]] = []
    prod_err: list[float] = []
    kaa_err: list[float] = []
    khh_err: list[float] = []
    for idx, point in enumerate(reference_points, start=1):
        molecule = point.molecule
        bound_prod = functional_prod.bind_to_molecule_for_response(params, molecule)
        bound_local = functional_local.bind_to_molecule(params, molecule)
        local_weight = getattr(bound_local, "local_hf_fraction_values", None)
        if local_weight is None:
            raise RuntimeError("local_projected binding did not expose local_hf_fraction_values.")

        prod_s1_h = _predict_s1_h(molecule, bound_prod)
        kaa_only_s1_h = _predict_s1_h(
            molecule,
            RestrictedLocalHFKhhTDAWrapper(
                base_xc=bound_local,
                local_weight=0.0,
                omega_index=int(args.omega_index),
                fd_step=float(args.fd_step),
            ),
        )
        khh_plus_s1_h = _predict_s1_h(
            molecule,
            RestrictedLocalHFKhhTDAWrapper(
                base_xc=bound_local,
                local_weight=local_weight,
                omega_index=int(args.omega_index),
                fd_step=float(args.fd_step),
            ),
        )
        fci_s1_h = float(point.fci_excitation_energies_h[0])
        prod_s1_ev = prod_s1_h * HARTREE_TO_EV
        kaa_only_s1_ev = kaa_only_s1_h * HARTREE_TO_EV
        khh_plus_s1_ev = khh_plus_s1_h * HARTREE_TO_EV
        fci_s1_ev = fci_s1_h * HARTREE_TO_EV

        prod_err.append(abs(prod_s1_ev - fci_s1_ev))
        kaa_err.append(abs(kaa_only_s1_ev - fci_s1_ev))
        khh_err.append(abs(khh_plus_s1_ev - fci_s1_ev))
        row = {
            "r_angstrom": float(point.r_angstrom),
            "fci_s1_ev": fci_s1_ev,
            "production_s1_ev": prod_s1_ev,
            "kaa_only_s1_ev": kaa_only_s1_ev,
            "kaa_plus_khh_s1_ev": khh_plus_s1_ev,
            "production_err_ev": prod_s1_ev - fci_s1_ev,
            "kaa_only_err_ev": kaa_only_s1_ev - fci_s1_ev,
            "kaa_plus_khh_err_ev": khh_plus_s1_ev - fci_s1_ev,
        }
        rows.append(row)
        logger.log(
            f"[eval] {idx:3d}/{len(reference_points):3d} R={row['r_angstrom']:.4f} A "
            f"FCI={fci_s1_ev:.6f} eV prod={prod_s1_ev:.6f} eV "
            f"kaa={kaa_only_s1_ev:.6f} eV khh={khh_plus_s1_ev:.6f} eV"
        )

    csv_path = outdir / "h2_tda_local_hf_test_module_curve.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        header = list(rows[0].keys())
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(f"{row[key]:.16e}" for key in header) + "\n")

    summary = {
        "checkpoint": str(checkpoint),
        "basis": str(args.basis),
        "xc": str(args.xc),
        "dense_points": int(args.dense_points),
        "fd_step": float(args.fd_step),
        "omega_index": int(args.omega_index),
        "production_s1_mae_ev": float(np.mean(prod_err)),
        "kaa_only_s1_mae_ev": float(np.mean(kaa_err)),
        "kaa_plus_khh_s1_mae_ev": float(np.mean(khh_err)),
        "curve_csv": str(csv_path),
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    plot_path = outdir / "h2_tda_local_hf_test_module_curve.png"
    _plot_curves(plot_path, rows)
    logger.log(f"production_s1_mae_ev={summary['production_s1_mae_ev']:.6e}")
    logger.log(f"kaa_only_s1_mae_ev={summary['kaa_only_s1_mae_ev']:.6e}")
    logger.log(f"kaa_plus_khh_s1_mae_ev={summary['kaa_plus_khh_s1_mae_ev']:.6e}")
    logger.log(f"wrote csv={csv_path}")
    logger.log(f"wrote summary={summary_path}")
    logger.log(f"wrote plot={plot_path}")


if __name__ == "__main__":
    main()
