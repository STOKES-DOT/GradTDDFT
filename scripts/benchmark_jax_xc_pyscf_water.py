#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_FUNCTIONALS = (
    "lda_x",
    "lda_c_vwn",
    "lda_c_pw",
    "gga_x_pbe",
    "gga_c_pbe",
    "gga_x_b88",
    "gga_c_lyp",
    "gga_x_rpbe",
    "gga_x_wc",
    "gga_x_pw91",
    "hyb_gga_xc_b3lyp",
    "hyb_gga_xc_pbeh",
    "hyb_gga_xc_b3pw91",
    "hyb_gga_xc_bhandhlyp",
    "hyb_gga_xc_hse03",
    "hyb_gga_xc_hse06",
    "hyb_gga_xc_cam_b3lyp",
    "hyb_gga_xc_b97",
    "hyb_gga_xc_b97_1",
    "hyb_gga_xc_wb97x",
)

WATER_ATOM = """
O  0.000000  0.000000  0.117790
H  0.000000  0.755453 -0.471161
H  0.000000 -0.755453 -0.471161
"""

FUNCTIONAL_LABELS = {
    "lda_x": "LDA_X",
    "lda_c_vwn": "LDA_C_VWN",
    "lda_c_pw": "LDA_C_PW",
    "gga_x_pbe": "PBE_X",
    "gga_c_pbe": "PBE_C",
    "gga_x_b88": "B88_X",
    "gga_c_lyp": "LYP_C",
    "gga_x_rpbe": "RPBE_X",
    "gga_x_wc": "WC_X",
    "gga_x_pw91": "PW91_X",
    "hyb_gga_xc_b3lyp": "B3LYP",
    "hyb_gga_xc_pbeh": "PBE0/PBEH",
    "hyb_gga_xc_b3pw91": "B3PW91",
    "hyb_gga_xc_bhandhlyp": "BHandHLYP",
    "hyb_gga_xc_hse03": "HSE03",
    "hyb_gga_xc_hse06": "HSE06",
    "hyb_gga_xc_cam_b3lyp": "CAM-B3LYP",
    "hyb_gga_xc_b97": "B97",
    "hyb_gga_xc_b97_1": "B97-1",
    "hyb_gga_xc_wb97x": "wB97X",
}


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _parse_functionals(value: str | None) -> tuple[str, ...]:
    if value is None or not value.strip():
        return DEFAULT_FUNCTIONALS
    return tuple(item.strip() for item in value.replace(",", " ").split() if item.strip())


def _finite_float(value: Any) -> float | None:
    value = float(value)
    if np.isfinite(value):
        return value
    return None


def _format_float(value: Any, *, width: int = 12) -> str:
    if value is None:
        return " " * (width - 3) + "nan"
    value_f = float(value)
    if not np.isfinite(value_f):
        return " " * (width - 3) + "nan"
    return f"{value_f:{width}.5e}"


def _build_water_mol(basis: str):
    from pyscf import gto

    mol = gto.Mole()
    mol.atom = WATER_ATOM
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.charge = 0
    mol.spin = 0
    mol.cart = True
    mol.verbose = 0
    mol.build()
    return mol


def _select_grid_points(
    coords: np.ndarray,
    weights: np.ndarray,
    max_points: int | None,
    selection: str,
) -> tuple[np.ndarray, np.ndarray]:
    if max_points is None or int(max_points) >= len(coords):
        return coords, weights

    count = int(max_points)
    if count <= 0:
        raise ValueError("max_points must be positive when provided.")

    selection = selection.lower()
    if selection == "head":
        idx = np.arange(count)
    elif selection == "even":
        idx = np.unique(np.linspace(0, len(coords) - 1, count, dtype=np.int64))
    else:
        raise ValueError("point_selection must be 'head' or 'even'.")
    return coords[idx], weights[idx]


def _build_reference_density(
    mol,
    grid_level: int,
    max_points: int | None,
    point_selection: str,
):
    from pyscf import dft, scf
    from pyscf.dft import numint

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF RHF reference did not converge.")

    grids = dft.gen_grid.Grids(mol)
    grids.level = int(grid_level)
    grids.build()

    coords = np.asarray(grids.coords, dtype=np.float64)
    weights = np.asarray(grids.weights, dtype=np.float64)
    coords, weights = _select_grid_points(coords, weights, max_points, point_selection)

    dm = np.asarray(mf.make_rdm1(), dtype=np.float64)
    ao = np.asarray(numint.eval_ao(mol, coords, deriv=1), dtype=np.float64)
    rho_gga = np.asarray(numint.eval_rho(mol, ao, dm, xctype="GGA"), dtype=np.float64)
    return {
        "mf": mf,
        "coords": coords,
        "weights": weights,
        "dm": dm,
        "rho_gga": rho_gga,
    }


def _setup_jax():
    from jax import config as jax_config

    jax_config.update("jax_enable_x64", True)
    jax_config.update("jax_platform_name", os.environ.get("JAX_PLATFORM_NAME", "cpu"))

    import jax
    import jax.numpy as jnp

    return jax, jnp


def _make_density_fn(basis, dm, jnp, jax_density_floor: float):
    from td_graddft.data import evaluate_cartesian_ao

    dm_jax = jnp.asarray(dm, dtype=jnp.float64)
    floor = jnp.asarray(jax_density_floor, dtype=jnp.float64)

    def rho_fn(r):
        ao0 = evaluate_cartesian_ao(basis, r[jnp.newaxis, :], deriv=0)[0]
        rho = jnp.einsum("p,pq,q->", ao0, dm_jax, ao0)
        return jnp.maximum(rho, floor)

    return rho_fn


def _evaluate_jax_functional(
    name: str,
    jax_xc,
    rho_fn,
    coords,
    chunk_size: int,
    jax,
    jnp,
    *,
    use_jit: bool,
):
    factory = getattr(jax_xc, name)
    functional = factory(polarized=False)

    def point_eval(r):
        value = functional(rho_fn, r)
        if isinstance(value, tuple):
            value = value[0]
        return jnp.asarray(value, dtype=jnp.float64)

    vmapped = jax.vmap(point_eval)
    if use_jit:
        vmapped = jax.jit(vmapped)
    coords_np = np.asarray(coords, dtype=np.float64)
    block_size = int(chunk_size) if int(chunk_size) > 0 else len(coords_np)
    outputs: list[np.ndarray] = []
    for start in range(0, len(coords_np), block_size):
        stop = min(start + block_size, len(coords_np))
        block = coords_np[start:stop]
        block_len = len(block)
        if block_len < block_size:
            padded = np.empty((block_size, 3), dtype=np.float64)
            padded[:block_len] = block
            padded[block_len:] = block[-1]
            block = padded
        values = np.asarray(vmapped(jnp.asarray(block)), dtype=np.float64)[:block_len]
        outputs.append(values.reshape(-1))
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0,), dtype=np.float64)


def _pyscf_exc(name: str, rho_gga):
    from pyscf import dft

    xctype = str(dft.libxc.xc_type(name)).upper()
    if xctype == "LDA":
        rho_arg = rho_gga[0]
    elif xctype == "GGA":
        rho_arg = rho_gga
    else:
        raise NotImplementedError(f"{name} has unsupported PySCF xc_type={xctype}.")
    exc = dft.libxc.eval_xc(name, rho_arg, spin=0, deriv=0)[0]
    return xctype, np.asarray(exc, dtype=np.float64).reshape(-1)


def _pyscf_xc_metadata(name: str) -> dict[str, Any]:
    from pyscf import dft

    data: dict[str, Any] = {
        "label": FUNCTIONAL_LABELS.get(name, name),
        "is_hybrid": bool(dft.libxc.is_hybrid_xc(name)),
        "hybrid_coeff": None,
        "rsh_coeff": None,
    }
    try:
        data["hybrid_coeff"] = _finite_float(dft.libxc.hybrid_coeff(name, spin=0))
    except Exception:
        pass
    try:
        rsh = dft.libxc.rsh_coeff(name)
        data["rsh_coeff"] = tuple(float(item) for item in rsh)
    except Exception:
        pass
    return data


def _energy(weights, rho, exc) -> float:
    return float(np.einsum("g,g,g->", weights, rho, exc))


def _compare_arrays(
    *,
    name: str,
    xctype: str,
    weights,
    rho,
    exc_jax,
    exc_ref,
    density_floor: float,
    relative_floor: float,
    seconds: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    finite = np.isfinite(exc_jax) & np.isfinite(exc_ref) & np.isfinite(rho)
    mask = finite & (rho > float(density_floor))
    dropped = int(len(rho) - np.count_nonzero(mask))
    if not np.any(mask):
        return {
            "functional": name,
            **metadata,
            "xctype": xctype,
            "status": "failed",
            "error": "No finite grid points above density_floor.",
            "points": int(len(rho)),
            "kept_points": 0,
            "dropped_points": dropped,
            "seconds": _finite_float(seconds),
        }

    diff = exc_jax[mask] - exc_ref[mask]
    denom = np.maximum(np.abs(exc_ref[mask]), float(relative_floor))
    e_jax = _energy(weights[mask], rho[mask], exc_jax[mask])
    e_ref = _energy(weights[mask], rho[mask], exc_ref[mask])
    e_abs = abs(e_jax - e_ref)
    e_rel = e_abs / max(abs(e_ref), float(relative_floor))
    return {
        "functional": name,
        **metadata,
        "xctype": xctype,
        "status": "ok",
        "points": int(len(rho)),
        "kept_points": int(np.count_nonzero(mask)),
        "dropped_points": dropped,
        "e_xc_jax": _finite_float(e_jax),
        "e_xc_pyscf": _finite_float(e_ref),
        "e_abs_diff": _finite_float(e_abs),
        "e_rel_diff": _finite_float(e_rel),
        "exc_max_abs": _finite_float(np.max(np.abs(diff))),
        "exc_rms_abs": _finite_float(np.sqrt(np.mean(diff * diff))),
        "exc_max_rel": _finite_float(np.max(np.abs(diff) / denom)),
        "seconds": _finite_float(seconds),
    }


def _rho_ao_diagnostic(basis, coords, dm, rho_ref):
    from td_graddft.data import evaluate_cartesian_ao

    ao = np.asarray(evaluate_cartesian_ao(basis, coords, deriv=0, chunk_size=4096))
    rho = np.einsum("gp,pq,gq->g", ao, dm, ao)
    diff = rho - rho_ref
    return {
        "rho_max_abs_diff": _finite_float(np.max(np.abs(diff))),
        "rho_rms_abs_diff": _finite_float(np.sqrt(np.mean(diff * diff))),
    }


def _print_table(results: list[dict[str, Any]]) -> None:
    headers = (
        "functional",
        "type",
        "kept",
        "E_jax",
        "E_pyscf",
        "|dE|",
        "rel_dE",
        "exc_rms",
        "exc_max",
    )
    print(
        "| "
        + " | ".join(headers)
        + " |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in results:
        if row.get("status") != "ok":
            print(
                f"| {row.get('label', row['functional'])} | {row.get('xctype', 'n/a')} | "
                f"0 | failed | failed | failed | failed | failed | failed |"
            )
            continue
        print(
            "| "
            f"{row.get('label', row['functional'])} | {row['xctype']} | {row['kept_points']} | "
            f"{_format_float(row['e_xc_jax']).strip()} | "
            f"{_format_float(row['e_xc_pyscf']).strip()} | "
            f"{_format_float(row['e_abs_diff']).strip()} | "
            f"{_format_float(row['e_rel_diff']).strip()} | "
            f"{_format_float(row['exc_rms_abs']).strip()} | "
            f"{_format_float(row['exc_max_abs']).strip()} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark jax_xc functionals against PySCF/libxc on a water density."
    )
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--functionals", default=None)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--point-selection", choices=("head", "even"), default="head")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--density-floor", type=float, default=1e-12)
    parser.add_argument("--jax-density-floor", type=float, default=1e-30)
    parser.add_argument("--relative-floor", type=float, default=1e-14)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--disable-jit", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    jax, jnp = _setup_jax()

    from td_graddft.data import basis_from_pyscf_mol_cart
    from td_graddft.jax_xc_adapter import load_jax_xc
    from td_graddft.xc_backend.vendor import vendored_jax_xc_info

    mol = _build_water_mol(args.basis)
    ref = _build_reference_density(
        mol,
        args.grid_level,
        args.max_points,
        args.point_selection,
    )
    coords = ref["coords"]
    weights = ref["weights"]
    dm = ref["dm"]
    rho_gga = ref["rho_gga"]
    rho = rho_gga[0]

    basis = basis_from_pyscf_mol_cart(mol, max_l=3, precompute_eri_groups=False)
    ao_diag = _rho_ao_diagnostic(basis, coords, dm, rho)
    rho_fn = _make_density_fn(basis, dm, jnp, args.jax_density_floor)

    jax_xc, backend = load_jax_xc()
    vendor = vendored_jax_xc_info()
    functionals = _parse_functionals(args.functionals)

    results: list[dict[str, Any]] = []
    for name in functionals:
        start = time.perf_counter()
        try:
            if args.progress:
                print(f"running {name}", flush=True)
            xc_metadata = _pyscf_xc_metadata(name)
            xctype, exc_ref = _pyscf_exc(name, rho_gga)
            exc_jax = _evaluate_jax_functional(
                name,
                jax_xc,
                rho_fn,
                coords,
                args.chunk_size,
                jax,
                jnp,
                use_jit=not bool(args.disable_jit),
            )
            elapsed = time.perf_counter() - start
            row = _compare_arrays(
                name=name,
                xctype=xctype,
                weights=weights,
                rho=rho,
                exc_jax=exc_jax,
                exc_ref=exc_ref,
                density_floor=args.density_floor,
                relative_floor=args.relative_floor,
                seconds=elapsed,
                metadata=xc_metadata,
            )
            results.append(row)
            if args.progress:
                print(
                    f"finished {xc_metadata.get('label', name)} "
                    f"|dE|={_format_float(row.get('e_abs_diff')).strip()} "
                    f"exc_max={_format_float(row.get('exc_max_abs')).strip()} "
                    f"seconds={elapsed:.2f}",
                    flush=True,
                )
        except Exception as exc:  # pragma: no cover - diagnostic script path
            results.append(
                {
                    "functional": name,
                    "label": FUNCTIONAL_LABELS.get(name, name),
                    "is_hybrid": None,
                    "hybrid_coeff": None,
                    "rsh_coeff": None,
                    "xctype": None,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "points": int(len(rho)),
                    "kept_points": 0,
                    "dropped_points": int(len(rho)),
                    "seconds": _finite_float(time.perf_counter() - start),
                }
            )
            if args.progress:
                print(f"failed {name}: {type(exc).__name__}: {exc}", flush=True)

    summary = {
        "molecule": "water",
        "basis": args.basis,
        "grid_level": int(args.grid_level),
        "grid_points": int(len(coords)),
        "max_points": None if args.max_points is None else int(args.max_points),
        "point_selection": args.point_selection,
        "density_floor": float(args.density_floor),
        "jax_density_floor": float(args.jax_density_floor),
        "jax_platform": jax.default_backend(),
        "jax_xc_backend": backend,
        "jax_xc_version": getattr(jax_xc, "__version__", None),
        "vendored_jax_xc": {
            "root": str(vendor.root),
            "complete": bool(vendor.complete),
            "commit": vendor.commit,
            "version": vendor.version,
            "reason": vendor.reason,
        },
        "versions": {
            "python": platform.python_version(),
            "jax": _package_version("jax"),
            "jaxlib": _package_version("jaxlib"),
            "jax_xc": _package_version("jax_xc"),
            "pyscf": _package_version("pyscf"),
            "numpy": _package_version("numpy"),
        },
        "reference_rhf": {
            "converged": bool(ref["mf"].converged),
            "e_tot": _finite_float(ref["mf"].e_tot),
        },
        "comparison_scope": (
            "Per-grid epsilon_xc and integrated semilocal/libxc E_xc only. "
            "For hybrid functionals this excludes the nonlocal exact-exchange matrix contribution."
        ),
        "ao_density_check": ao_diag,
        "results": results,
    }

    print("Water jax_xc vs PySCF/libxc benchmark")
    print(
        f"basis={args.basis} grid_level={args.grid_level} points={len(coords)} "
        f"point_selection={args.point_selection}"
    )
    print(
        f"jax_backend={summary['jax_platform']} "
        f"jax_xc_backend={backend} jax_xc_version={summary['jax_xc_version']}"
    )
    print(
        "rho_ao_max_abs_diff="
        f"{_format_float(ao_diag['rho_max_abs_diff']).strip()} "
        "rho_ao_rms_abs_diff="
        f"{_format_float(ao_diag['rho_rms_abs_diff']).strip()}"
    )
    print("scope=semilocal/libxc epsilon_xc only; hybrid exact exchange matrix term is not included")
    _print_table(results)

    failures = [row for row in results if row.get("status") != "ok"]
    if failures:
        print("Failures:")
        for row in failures:
            print(f"- {row['functional']}: {row.get('error', 'unknown error')}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote_json={args.json_out}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
