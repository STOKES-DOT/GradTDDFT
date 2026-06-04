from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .molecule import MoleculeSpec, parse_molecule_spec

RAD_GRIDS = np.asarray(
    [
        [10, 15, 20, 30, 35, 40, 50],
        [30, 40, 50, 60, 65, 70, 75],
        [40, 60, 65, 75, 80, 85, 90],
        [50, 75, 80, 90, 95, 100, 105],
        [60, 90, 95, 105, 110, 115, 120],
        [70, 105, 110, 120, 125, 130, 135],
        [80, 120, 125, 135, 140, 145, 150],
        [90, 135, 140, 150, 155, 160, 165],
        [100, 150, 155, 165, 170, 175, 180],
        [200, 200, 200, 200, 200, 200, 200],
    ],
    dtype=np.int32,
)
ANG_ORDER = np.asarray(
    [
        [11, 15, 17, 17, 17, 17, 17],
        [17, 23, 23, 23, 23, 23, 23],
        [23, 29, 29, 29, 29, 29, 29],
        [29, 29, 35, 35, 35, 35, 35],
        [35, 41, 41, 41, 41, 41, 41],
        [41, 47, 47, 47, 47, 47, 47],
        [47, 53, 53, 53, 53, 53, 53],
        [53, 59, 59, 59, 59, 59, 59],
        [59, 59, 59, 59, 59, 59, 59],
        [65, 65, 65, 65, 65, 65, 65],
    ],
    dtype=np.int32,
)
PERIOD_TAB = np.asarray((2, 10, 18, 36, 54, 86, 118), dtype=np.int32)
LEBEDEV_ORDER_TO_POINTS = {
    11: 50,
    13: 74,
    15: 86,
    17: 110,
    23: 194,
    29: 302,
}
BUNDLED_LEBEDEV_POINTS = (50, 74, 86)
SUPPORTED_GRID_LEVELS = range(min(RAD_GRIDS.shape[0], ANG_ORDER.shape[0]))
BRAGG_RADII = {
    1: 0.6614041435977716,
    2: 2.645616574391086,
    3: 2.7401028806193395,
    4: 1.984212430793315,
    5: 1.6062672058803025,
    6: 1.322808287195543,
    7: 1.2283219809672903,
    8: 1.133835674739037,
    9: 0.9448630622825309,
    10: 2.8345891868475928,
    11: 3.4015070242171115,
    12: 2.8345891868475928,
    13: 2.3621576557063273,
    14: 2.0786987370215684,
    15: 1.8897261245650618,
    16: 1.8897261245650618,
    17: 1.8897261245650618,
    18: 3.4015070242171115,
    19: 4.157397474043137,
    20: 3.4015070242171115,
    21: 3.0235617993040993,
    22: 2.645616574391086,
    23: 2.551130268162834,
    24: 2.645616574391086,
    25: 2.645616574391086,
    26: 2.645616574391086,
    27: 2.551130268162834,
    28: 2.551130268162834,
    29: 2.551130268162834,
    30: 2.551130268162834,
    31: 2.4566439619345806,
    32: 2.3621576557063273,
    33: 2.1731850432498208,
    34: 2.1731850432498208,
    35: 2.1731850432498208,
    36: 3.590479636673617,
}
TREUTLER_XI = {
    1: 0.8,
    6: 1.1,
    7: 0.9,
    8: 0.9,
}


@lru_cache(maxsize=None)
def _load_lebedev_table(npoints: int) -> jnp.ndarray:
    path = Path(__file__).with_name("_lebedev_level0.npz")
    with np.load(path) as data:
        key = f"g{int(npoints)}"
        if key not in data:
            raise NotImplementedError(
                f"Bundled Lebedev grid with {npoints} points is not available."
            )
        return jnp.asarray(data[key], dtype=jnp.float64)


@lru_cache(maxsize=None)
def _load_lebedev_table_np(npoints: int) -> np.ndarray:
    path = Path(__file__).with_name("_lebedev_level0.npz")
    with np.load(path) as data:
        key = f"g{int(npoints)}"
        if key not in data:
            raise NotImplementedError(
                f"Bundled Lebedev grid with {npoints} points is not available."
            )
        return np.asarray(data[key], dtype=float)


def _default_rad(nuc: int, level: int) -> int:
    level = _validate_grid_level(level)
    period = int(np.sum(int(nuc) > PERIOD_TAB))
    return int(RAD_GRIDS[int(level), period])


def _default_ang(nuc: int, level: int) -> int:
    level = _validate_grid_level(level)
    period = int(np.sum(int(nuc) > PERIOD_TAB))
    order = int(ANG_ORDER[int(level), period])
    if order not in LEBEDEV_ORDER_TO_POINTS:
        raise NotImplementedError(f"Lebedev order {order} is not bundled.")
    target_points = int(LEBEDEV_ORDER_TO_POINTS[order])
    available = [npoints for npoints in BUNDLED_LEBEDEV_POINTS if npoints <= target_points]
    if not available:
        raise NotImplementedError(
            f"Lebedev grid with up to {target_points} points is not bundled."
        )
    return int(max(available))


def _validate_grid_level(level: int) -> int:
    level = int(level)
    if level not in SUPPORTED_GRID_LEVELS:
        supported = f"0..{max(SUPPORTED_GRID_LEVELS)}"
        raise NotImplementedError(
            f"Bundled strict-JAX molecular grid supports levels {supported}."
        )
    return level


def _treutler_radial_grid(n: int, charge: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    xi = float(TREUTLER_XI.get(int(charge), 1.0))
    step = jnp.pi / float(n + 1)
    ln2 = xi / jnp.log(2.0)
    idx = jnp.arange(1, int(n) + 1, dtype=jnp.float64)
    x = jnp.cos(idx * step)
    r = -ln2 * (1.0 + x) ** 0.6 * jnp.log((1.0 - x) / 2.0)
    dr = (
        step
        * jnp.sin(idx * step)
        * ln2
        * (1.0 + x) ** 0.6
        * (-0.6 / (1.0 + x) * jnp.log((1.0 - x) / 2.0) + 1.0 / (1.0 - x))
    )
    return r[::-1], dr[::-1]


def _nwchem_prune(nuc: int, rads: jnp.ndarray, n_ang: int) -> jnp.ndarray:
    alphas = jnp.asarray(
        (
            (0.25, 0.5, 1.0, 4.5),
            (0.1667, 0.5, 0.9, 3.5),
            (0.1, 0.4, 0.8, 2.5),
        ),
        dtype=jnp.float64,
    )
    leb_ngrid = jnp.asarray([38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302], dtype=jnp.int32)
    if int(n_ang) < 50:
        return jnp.full((int(rads.shape[0]),), int(n_ang), dtype=jnp.int32)
    if int(n_ang) == 50:
        leb_l = jnp.asarray([1, 2, 2, 2, 1], dtype=jnp.int32)
    else:
        matches = np.where(np.asarray(leb_ngrid) == int(n_ang))[0]
        if matches.size == 0:
            raise NotImplementedError(f"Unsupported n_ang={n_ang} in bundled NWChem prune.")
        idx = int(matches[0])
        leb_l = jnp.asarray([1, 3, idx - 1, idx, idx - 1], dtype=jnp.int32)

    r_atom = float(BRAGG_RADII[int(nuc)]) + 1e-200
    if int(nuc) <= 2:
        thresholds = alphas[0]
    elif int(nuc) <= 10:
        thresholds = alphas[1]
    else:
        thresholds = alphas[2]
    place = jnp.sum((rads / r_atom).reshape(-1, 1) > thresholds.reshape(1, -1), axis=1)
    return leb_ngrid[leb_l[place]]


def _original_becke(g: jnp.ndarray) -> jnp.ndarray:
    for _ in range(3):
        g = 0.5 * (3.0 - g * g) * g
    return g


def _treutler_atomic_radii_adjust(charges: jnp.ndarray):
    rad = jnp.sqrt(jnp.asarray([BRAGG_RADII[int(z)] for z in np.asarray(charges)], dtype=jnp.float64)) + 1e-200
    rr = rad.reshape(-1, 1) / rad.reshape(1, -1)
    a = 0.25 * (rr.T - rr)
    a = jnp.clip(a, -0.5, 0.5)

    def adjust(i: int, j: int, g: jnp.ndarray) -> jnp.ndarray:
        return g - a[i, j] * (g * g - 1.0)

    return adjust


def _treutler_radial_grid_np(n: int, charge: int) -> tuple[np.ndarray, np.ndarray]:
    xi = float(TREUTLER_XI.get(int(charge), 1.0))
    step = np.pi / float(n + 1)
    ln2 = xi / np.log(2.0)
    idx = np.arange(1, int(n) + 1, dtype=float)
    x = np.cos(idx * step)
    r = -ln2 * (1.0 + x) ** 0.6 * np.log((1.0 - x) / 2.0)
    dr = (
        step
        * np.sin(idx * step)
        * ln2
        * (1.0 + x) ** 0.6
        * (-0.6 / (1.0 + x) * np.log((1.0 - x) / 2.0) + 1.0 / (1.0 - x))
    )
    return r[::-1], dr[::-1]


def _nwchem_prune_np(nuc: int, rads: np.ndarray, n_ang: int) -> np.ndarray:
    alphas = np.asarray(
        (
            (0.25, 0.5, 1.0, 4.5),
            (0.1667, 0.5, 0.9, 3.5),
            (0.1, 0.4, 0.8, 2.5),
        ),
        dtype=float,
    )
    leb_ngrid = np.asarray([38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302], dtype=np.int32)
    if int(n_ang) < 50:
        return np.full((int(rads.shape[0]),), int(n_ang), dtype=np.int32)
    if int(n_ang) == 50:
        leb_l = np.asarray([1, 2, 2, 2, 1], dtype=np.int32)
    else:
        matches = np.where(leb_ngrid == int(n_ang))[0]
        if matches.size == 0:
            raise NotImplementedError(f"Unsupported n_ang={n_ang} in bundled NWChem prune.")
        idx = int(matches[0])
        leb_l = np.asarray([1, 3, idx - 1, idx, idx - 1], dtype=np.int32)

    r_atom = float(BRAGG_RADII[int(nuc)]) + 1e-200
    if int(nuc) <= 2:
        thresholds = alphas[0]
    elif int(nuc) <= 10:
        thresholds = alphas[1]
    else:
        thresholds = alphas[2]
    place = np.sum((rads / r_atom).reshape(-1, 1) > thresholds.reshape(1, -1), axis=1)
    return leb_ngrid[leb_l[place]]


def _original_becke_np(g: np.ndarray) -> np.ndarray:
    out = np.asarray(g, dtype=float)
    for _ in range(3):
        out = 0.5 * (3.0 - out * out) * out
    return out


def _treutler_atomic_radii_adjust_np(charges: np.ndarray):
    rad = np.sqrt(np.asarray([BRAGG_RADII[int(z)] for z in charges], dtype=float)) + 1e-200
    rr = rad.reshape(-1, 1) / rad.reshape(1, -1)
    a = np.clip(0.25 * (rr.T - rr), -0.5, 0.5)

    def adjust(i: int, j: int, g: np.ndarray) -> np.ndarray:
        return g - a[i, j] * (g * g - 1.0)

    return adjust


def _atomic_local_grid_np(charge: int, *, level: int) -> tuple[np.ndarray, np.ndarray]:
    n_rad = _default_rad(int(charge), int(level))
    n_ang = _default_ang(int(charge), int(level))
    rad, dr = _treutler_radial_grid_np(n_rad, int(charge))
    rad_weight = 4.0 * np.pi * rad * rad * dr
    angs = _nwchem_prune_np(int(charge), rad, int(n_ang))
    coords_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    for n in sorted({int(x) for x in angs}):
        grid = _load_lebedev_table_np(n)
        idx = np.where(angs == n)[0]
        rad_sel = rad[idx]
        w_sel = rad_weight[idx]
        coords_parts.append(
            np.einsum("i,jk->ijk", rad_sel, grid[:, :3]).reshape(-1, 3)
        )
        weight_parts.append(
            np.einsum("i,j->ij", w_sel, grid[:, 3]).reshape(-1)
        )
    return np.concatenate(coords_parts, axis=0), np.concatenate(weight_parts, axis=0)


def _build_molecular_grid_from_spec_numpy(
    spec: MoleculeSpec,
    *,
    level: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    natm = len(spec.symbols)
    atm_coords = np.asarray(spec.coords_bohr, dtype=float)
    charges = np.asarray(spec.charges, dtype=int)
    atm_dist = np.linalg.norm(
        atm_coords[:, None, :] - atm_coords[None, :, :],
        axis=-1,
    ) + np.eye(natm)
    radii_adjust = _treutler_atomic_radii_adjust_np(charges)

    unique_atom_grids: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sym, z in zip(spec.symbols, charges, strict=True):
        if sym not in unique_atom_grids:
            unique_atom_grids[sym] = _atomic_local_grid_np(int(z), level=int(level))

    coords_all: list[np.ndarray] = []
    weights_all: list[np.ndarray] = []
    for ia, sym in enumerate(spec.symbols):
        coords_local, vol = unique_atom_grids[sym]
        coords = coords_local + atm_coords[ia]
        grid_dist = np.linalg.norm(coords[None, :, :] - atm_coords[:, None, :], axis=-1)
        pbecke = np.ones((natm, coords.shape[0]), dtype=float)
        for i in range(natm):
            for j in range(i):
                g = (grid_dist[i] - grid_dist[j]) / atm_dist[i, j]
                g = radii_adjust(i, j, g)
                g = _original_becke_np(g)
                pbecke[i] *= 0.5 * (1.0 - g)
                pbecke[j] *= 0.5 * (1.0 + g)
        weights = vol * pbecke[ia] / np.sum(pbecke, axis=0)
        coords_all.append(coords)
        weights_all.append(weights)

    return jax.device_put(np.concatenate(coords_all, axis=0)), jax.device_put(
        np.concatenate(weights_all, axis=0)
    )


def _build_molecular_grid_from_spec_jax(
    spec: MoleculeSpec,
    *,
    level: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    natm = len(spec.symbols)
    atm_coords = jnp.asarray(spec.coords_bohr)
    charges = jnp.asarray(spec.charges)
    atm_diffs = atm_coords[:, None, :] - atm_coords[None, :, :]
    atm_dist2 = jnp.einsum("...r,...r->...", atm_diffs, atm_diffs)
    atom_self_mask = jnp.eye(natm, dtype=bool)
    atm_dist = jnp.sqrt(jnp.where(atom_self_mask, 1.0, jnp.maximum(atm_dist2, 1e-32)))
    radii_adjust = _treutler_atomic_radii_adjust(charges)

    unique_atom_grids: dict[str, tuple[jnp.ndarray, jnp.ndarray]] = {}
    for sym, z in zip(spec.symbols, np.asarray(charges), strict=True):
        if sym not in unique_atom_grids:
            unique_atom_grids[sym] = _atomic_local_grid(int(z), level=int(level))

    coords_all = []
    weights_all = []
    for ia, sym in enumerate(spec.symbols):
        coords_local, vol = unique_atom_grids[sym]
        coords = coords_local + atm_coords[ia]
        grid_diffs = coords[None, :, :] - atm_coords[:, None, :]
        grid_dist2 = jnp.einsum("...r,...r->...", grid_diffs, grid_diffs)
        grid_dist = jnp.sqrt(jnp.maximum(grid_dist2, 1e-32))
        pbecke = jnp.ones((natm, coords.shape[0]), dtype=jnp.float64)
        for i in range(natm):
            for j in range(i):
                g = (grid_dist[i] - grid_dist[j]) / atm_dist[i, j]
                g = radii_adjust(i, j, g)
                g = _original_becke(g)
                pbecke = pbecke.at[i].set(pbecke[i] * (0.5 * (1.0 - g)))
                pbecke = pbecke.at[j].set(pbecke[j] * (0.5 * (1.0 + g)))
        weights = vol * pbecke[ia] / jnp.sum(pbecke, axis=0)
        coords_all.append(coords)
        weights_all.append(weights)

    return jnp.concatenate(coords_all, axis=0), jnp.concatenate(weights_all, axis=0)


def _spec_has_jax_tracer(spec: MoleculeSpec) -> bool:
    leaves = jax.tree_util.tree_leaves((spec.coords_bohr, spec.charges))
    return any(isinstance(leaf, jax.core.Tracer) for leaf in leaves)


def _atomic_local_grid(charge: int, *, level: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    n_rad = _default_rad(int(charge), int(level))
    n_ang = _default_ang(int(charge), int(level))
    rad, dr = _treutler_radial_grid(n_rad, int(charge))
    rad_weight = 4.0 * jnp.pi * rad * rad * dr
    angs = _nwchem_prune(int(charge), rad, int(n_ang))
    coords_parts = []
    weight_parts = []
    for n in sorted({int(x) for x in np.asarray(angs)}):
        grid = _load_lebedev_table(n)
        idx = np.where(np.asarray(angs) == n)[0]
        rad_sel = rad[idx]
        w_sel = rad_weight[idx]
        coords_parts.append(
            jnp.einsum("i,jk->ijk", rad_sel, grid[:, :3]).reshape(-1, 3)
        )
        weight_parts.append(
            jnp.einsum("i,j->ij", w_sel, grid[:, 3]).reshape(-1)
        )
    return jnp.concatenate(coords_parts, axis=0), jnp.concatenate(weight_parts, axis=0)


def build_molecular_grid(
    atom: Any,
    *,
    unit: str = "Angstrom",
    charge: int = 0,
    spin: int = 0,
    level: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray, MoleculeSpec]:
    level = _validate_grid_level(level)
    spec = parse_molecule_spec(atom, unit=unit, charge=charge, spin=spin)
    coords, weights = build_molecular_grid_from_spec(spec, level=level)
    return coords, weights, spec


def build_molecular_grid_from_spec(
    spec: MoleculeSpec,
    *,
    level: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build a molecular integration grid from explicit MoleculeSpec."""

    level = _validate_grid_level(level)
    if _spec_has_jax_tracer(spec):
        return _build_molecular_grid_from_spec_jax(spec, level=level)
    return _build_molecular_grid_from_spec_numpy(spec, level=level)


__all__ = [
    "build_molecular_grid",
    "build_molecular_grid_from_spec",
]
