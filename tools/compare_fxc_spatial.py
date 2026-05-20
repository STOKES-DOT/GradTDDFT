"""Compare spatial distribution of f_xc between PySCF and JAX autodiff for H2.

PySCF f_xc: ni.eval_xc(xc, rho, deriv=2) -> fxc[0] = d^2 Exc / drho^2  (numerical Libxc)
JAX   f_xc: jax.hessian(point_energy) -> tensor[0,0,:] = d^2 Exc / drho^2  (autodiff)

We evaluate both on the same molecular grid and compare across space.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Force CPU backend before any JAX import: Metal GPU on macOS has
# memory-space issues with hessian+vmap on Apple Silicon.
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pyscf import dft, gto

from td_graddft.xc_backend.jax_libxc import (
    eval_xc_response_tensor,
    xc_type,
    restricted_feature_bundle_from_rho_grad_tau,
)

HARTREE_TO_EV = 27.2114079527

H2_XYZ = """H  0.000000  0.000000 -0.370000
H  0.000000  0.000000  0.370000"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xc", default="pbe", help="XC functional spec (default: pbe)")
    p.add_argument("--basis", default="def2-svp", help="Basis set (default: def2-svp)")
    p.add_argument("--bond-length", type=float, default=0.74,
                   help="H-H bond length in Angstrom (default: 0.74)")
    p.add_argument("--grid-level", type=int, default=3,
                   help="PySCF grid level (default: 3)")
    p.add_argument("--outdir", default="outputs/fxc_spatial_compare",
                   help="Output directory")
    p.add_argument("--density-floor", type=float, default=1e-12,
                   help="Density floor for JAX eval (default: 1e-12)")
    p.add_argument("--float64", action="store_true",
                   help="Enable JAX float64 mode (requires JAX_ENABLE_X64=1 or compatible backend)")
    return p.parse_args()


def build_h2_molecule(bond_length: float, basis: str) -> gto.M:
    """Build H2 with given bond length."""
    half = bond_length / 2.0
    xyz = f"H  0.000000  0.000000 {-half}\nH  0.000000  0.000000  {half}"
    return gto.M(atom=xyz, basis=basis, spin=0, charge=0, verbose=0)


def run_pyscf_scf(mol: gto.M, xc: str, grid_level: int) -> dft.RKS:
    """Run PySCF RKS calculation."""
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = grid_level
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"PySCF SCF did not converge for H2 {xc}/{mol.basis}.")
    return mf


def _extract_rho_and_grad_from_eval_rho(
    rho: np.ndarray | list,
    xctype: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract scalar rho and grad from ni.eval_rho return value.

    Handles PySCF format variations:
    - New PySCF: (4, N) or (5, N) ndarray for GGA/MGGA
    - Old PySCF: list of [rho, gx, gy, gz] each (N,)
    - LDA: (N,) array
    """
    if isinstance(rho, (list, tuple)):
        # Old format: list of arrays
        rho_scalar = np.asarray(rho[0], dtype=float)
        if xctype.upper() in ("GGA", "MGGA", "MGGA_LAPL"):
            grad = np.stack([np.asarray(rho[i], dtype=float) for i in range(1, 4)], axis=-1)
        else:
            grad = None
    elif isinstance(rho, np.ndarray):
        if rho.ndim == 1:
            # LDA: (N,)
            rho_scalar = np.asarray(rho, dtype=float)
            grad = None
        elif rho.ndim == 2 and rho.shape[0] in (4, 5):
            # New format: (4, N) or (5, N) for restricted
            rho_scalar = np.asarray(rho[0, :], dtype=float)
            grad = np.asarray(rho[1:4, :].T, dtype=float)  # (N, 3)
        else:
            raise ValueError(f"Unexpected rho shape from eval_rho: {rho.shape}")
    else:
        raise TypeError(f"Unexpected rho type: {type(rho)}")

    return rho_scalar, grad


def pyscf_fxc_on_grid(mf: dft.RKS) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate f_xc(r) on the PySCF grid via deriv=2.

    Returns:
        coords: (N, 3) grid coordinates in Angstrom
        weights: (N,) grid weights
        rho: (N,) electron density on grid
        fxc: (N,) d^2 Exc / drho^2 on grid
    """
    ni = mf._numint
    mol = mf.mol
    grids = mf.grids
    dm = mf.make_rdm1()
    xctype = ni._xc_type(mf.xc)

    # Evaluate AO and density on grid
    ao = ni.eval_ao(mol, grids.coords, deriv=0 if xctype == "LDA" else 1)
    rho_raw = ni.eval_rho(mol, ao, dm, xctype=xctype)
    _, _, fxc_raw, _ = ni.eval_xc(mf.xc, rho_raw, spin=0, relativity=0, deriv=2)

    # Extract f_xc = d^2 Exc / drho^2
    if isinstance(fxc_raw, (list, tuple)):
        frr = np.asarray(fxc_raw[0], dtype=float)
    else:
        frr = np.asarray(fxc_raw, dtype=float)

    ngrids = grids.weights.shape[0]
    # Handle shape variations across Libxc versions
    if frr.ndim == 1:
        fxc = frr
    elif frr.ndim == 2:
        if frr.shape[0] == ngrids:
            fxc = frr[:, 0]
        elif frr.shape[1] == ngrids:
            fxc = frr[0, :]
        else:
            fxc = frr.reshape(-1)[:ngrids]
    else:
        fxc = frr.reshape(-1)[:ngrids]

    # Extract scalar rho
    rho_scalar, _ = _extract_rho_and_grad_from_eval_rho(rho_raw, xctype)

    return (grids.coords.copy(), grids.weights.copy(), rho_scalar.copy(), fxc.copy())


def jax_fxc_on_grid(
    xc_spec: str,
    rho: np.ndarray,
    grad: np.ndarray | None,
    density_floor: float,
    *,
    use_float64: bool = False,
) -> np.ndarray:
    """Evaluate f_xc(r) on the grid via JAX autodiff (eval_xc_response_tensor).

    Uses jax.hessian on the pointwise energy density, then extracts tensor[0,0,:].
    """
    dtype = jnp.float64 if use_float64 else jnp.float32
    rho_j = jnp.asarray(rho, dtype=dtype)
    grad_j = jnp.asarray(grad, dtype=dtype) if grad is not None else None
    tau_j = None  # GGA: no tau needed

    kind, tensor = eval_xc_response_tensor(
        xc_spec,
        rho_j,
        grad=grad_j,
        tau=tau_j,
        density_floor=float(density_floor),
    )
    # tensor shape: (nvar, nvar, ngrids); for GGA nvar=4
    # f_xc = d^2 E / drho^2 = tensor[0, 0, :]
    fxc_j = np.asarray(tensor[0, 0, :], dtype=float)
    return fxc_j


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"H2 f_xc spatial comparison: PySCF vs JAX autodiff")
    print(f"XC={args.xc}, basis={args.basis}, R={args.bond_length} A")
    print(f"{'='*60}")

    # --- Build molecule & run PySCF SCF ---
    mol = build_h2_molecule(args.bond_length, args.basis)
    mf = run_pyscf_scf(mol, args.xc, args.grid_level)
    print(f"\nPySCF SCF converged: E = {mf.e_tot:.10f} Hartree")

    # --- Get grid data ---
    ni = mf._numint
    grids = mf.grids
    dm = mf.make_rdm1()
    xctype = ni._xc_type(mf.xc)

    coords, weights, rho_pyscf, fxc_pyscf = pyscf_fxc_on_grid(mf)

    # Get grad on grid for JAX (rerun eval_rho since we need grad)
    ao_deriv1 = ni.eval_ao(mol, grids.coords, deriv=1)
    rho_full = ni.eval_rho(mol, ao_deriv1, dm, xctype=xctype)
    _, grad_arr = _extract_rho_and_grad_from_eval_rho(rho_full, xctype)

    # For LDA, create dummy grad
    if grad_arr is None:
        grad_arr = np.zeros((coords.shape[0], 3), dtype=float)

    print(f"Grid: {coords.shape[0]} points, rho range [{rho_pyscf.min():.4e}, {rho_pyscf.max():.4f}]")

    # --- Compute JAX f_xc ---
    fxc_jax = jax_fxc_on_grid(args.xc, rho_pyscf, grad_arr, args.density_floor,
                              use_float64=args.float64)

    # --- Statistics ---
    diff = fxc_jax - fxc_pyscf
    mae = float(np.mean(np.abs(diff)))
    maxe = float(np.max(np.abs(diff)))
    rms = float(np.sqrt(np.mean(diff ** 2)))

    # Relative difference where PySCF |fxc| > threshold
    thresh = 1e-8
    mask = np.abs(fxc_pyscf) > thresh
    if mask.sum() > 0:
        rel_diff = np.abs(diff[mask]) / np.abs(fxc_pyscf[mask])
        mae_rel = float(np.mean(rel_diff)) * 100
        max_rel = float(np.max(rel_diff)) * 100
    else:
        mae_rel = max_rel = 0.0

    print(f"\n--- f_xc Statistics ---")
    print(f"PySCF f_xc:  min={fxc_pyscf.min():.8f}  max={fxc_pyscf.max():.8f}")
    print(f"JAX   f_xc:  min={fxc_jax.min():.8f}  max={fxc_jax.max():.8f}")
    print(f"MAE:        {mae:.6e}")
    print(f"Max |diff|: {maxe:.6e}")
    print(f"RMS diff:   {rms:.6e}")
    if mask.sum() > 0:
        print(f"MAE rel:    {mae_rel:.4f}%")
        print(f"Max rel:    {max_rel:.4f}%")

    # --- Spatial analysis: distance from bond center ---
    z_coords = coords[:, 2]  # H2 along z-axis
    r_dist = np.abs(z_coords)

    # --- Figure 1: f_xc vs z coordinate ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"H$_2$ f$_{{xc}}$(r) Comparison — {args.xc.upper()}/{args.basis}, R={args.bond_length} Å",
                 fontsize=13)

    # Sort by z for line plots
    sort_idx = np.argsort(z_coords)

    # Panel 1: Both f_xc vs z
    ax = axes[0, 0]
    ax.plot(z_coords[sort_idx], fxc_pyscf[sort_idx], 'b-', lw=1.5, label='PySCF (Libxc)', alpha=0.8)
    ax.plot(z_coords[sort_idx], fxc_jax[sort_idx], 'r--', lw=1.5, label='JAX (autodiff)', alpha=0.8)
    ax.set_xlabel('z (Å)')
    ax.set_ylabel('f$_{xc}$ (a.u.)')
    ax.set_title('f$_{xc}$ along bond axis')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    # Panel 2: Difference vs z
    ax = axes[0, 1]
    ax.plot(z_coords[sort_idx], diff[sort_idx], 'k-', lw=1.0, alpha=0.7)
    ax.fill_between(z_coords[sort_idx], 0, diff[sort_idx], alpha=0.15, color='gray')
    ax.axhline(y=0, color='gray', lw=0.5, ls='--')
    ax.set_xlabel('z (Å)')
    ax.set_ylabel('Δ f$_{xc}$ (a.u.)')
    ax.set_title(f'Difference JAX − PySCF  (MAE={mae:.2e})')
    ax.grid(alpha=0.25)

    # Panel 3: f_xc vs rho (density dependence)
    ax = axes[1, 0]
    ax.scatter(rho_pyscf, fxc_pyscf, s=2, c='blue', alpha=0.4, label='PySCF')
    ax.scatter(rho_pyscf, fxc_jax, s=2, c='red', alpha=0.4, label='JAX')
    ax.set_xlabel('ρ (a.u.)')
    ax.set_ylabel('f$_{xc}$ (a.u.)')
    ax.set_title('f$_{xc}$ vs electron density')
    ax.legend(frameon=False, fontsize=9, markerscale=3)
    ax.grid(alpha=0.25)
    ax.set_xscale('log')

    # Panel 4: f_xc vs distance from bond center
    ax = axes[1, 1]
    # Radial distance from bond center
    r_radial = np.sqrt(np.sum(coords ** 2, axis=1))
    sort_r = np.argsort(r_radial)
    ax.plot(r_radial[sort_r], fxc_pyscf[sort_r], 'b-', lw=1.5, label='PySCF', alpha=0.8)
    ax.plot(r_radial[sort_r], fxc_jax[sort_r], 'r--', lw=1.5, label='JAX', alpha=0.8)
    ax.set_xlabel('r (Å)')
    ax.set_ylabel('f$_{xc}$ (a.u.)')
    ax.set_title('f$_{xc}$ vs radial distance from bond center')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    png_path = outdir / f"H2_fxc_compare_{args.xc}_{args.basis}.png"
    fig.savefig(png_path, dpi=170)
    plt.close(fig)
    print(f"\nFigure saved: {png_path}")

    # --- Figure 2: Scatter parity plot ---
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(fxc_pyscf, fxc_jax, s=2, c='black', alpha=0.5)
    vmin = min(fxc_pyscf.min(), fxc_jax.min())
    vmax = max(fxc_pyscf.max(), fxc_jax.max())
    ax2.plot([vmin, vmax], [vmin, vmax], 'r--', lw=1.0, label='y = x')
    ax2.set_xlabel('PySCF f$_{xc}$ (a.u.)')
    ax2.set_ylabel('JAX f$_{xc}$ (a.u.)')
    ax2.set_title(f'H$_2$ f$_{{xc}}$ parity — {args.xc.upper()}/{args.basis}')
    ax2.legend(frameon=False)
    ax2.grid(alpha=0.25)
    fig2.tight_layout()
    parity_path = outdir / f"H2_fxc_parity_{args.xc}_{args.basis}.png"
    fig2.savefig(parity_path, dpi=170)
    plt.close(fig2)
    print(f"Parity plot saved: {parity_path}")

    # --- Save CSV ---
    csv_path = outdir / f"H2_fxc_grid_{args.xc}_{args.basis}.csv"
    header = "x(A),y(A),z(A),r(A),rho,pyscf_fxc,jax_fxc,diff"
    data = np.column_stack([
        coords[:, 0], coords[:, 1], coords[:, 2],
        r_radial, rho_pyscf, fxc_pyscf, fxc_jax, diff,
    ])
    np.savetxt(csv_path, data, delimiter=',', header=header, comments='', fmt='%.10e')
    print(f"Grid data saved: {csv_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Grid points: {coords.shape[0]}")
    print(f"  PySCF f_xc range: [{fxc_pyscf.min():.6f}, {fxc_pyscf.max():.6f}]")
    print(f"  JAX   f_xc range: [{fxc_jax.min():.6f}, {fxc_jax.max():.6f}]")
    print(f"  MAE:  {mae:.6e}")
    print(f"  Max:  {maxe:.6e}")
    print(f"  RMS:  {rms:.6e}")
    if mask.sum() > 0:
        print(f"  Rel MAE: {mae_rel:.4f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
