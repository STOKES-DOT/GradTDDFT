from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from td_graddft.data.grid import (
    BRAGG_RADII,
    BUNDLED_LEBEDEV_POINTS,
    TREUTLER_XI,
    _build_molecular_grid_from_spec_jax,
    _default_ang,
    _default_rad,
    _load_lebedev_table_np,
    build_molecular_grid,
)
from td_graddft.data.molecule import parse_molecule_spec


def _pyscf_grid(atom: str, *, level: int):
    pytest.importorskip("pyscf")
    from pyscf import gto
    from pyscf.dft import gen_grid, radi

    mol = gto.M(
        atom=atom,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        verbose=0,
    )
    atom_grids = gen_grid.gen_atomic_grids(
        mol,
        level=level,
        radi_method=radi.treutler_ahlrichs,
        prune=gen_grid.nwchem_prune,
    )
    return gen_grid.get_partition(
        mol,
        atom_grids,
        radii_adjust=radi.treutler_atomic_radii_adjust,
        atomic_radii=radi.BRAGG_RADII,
        becke_scheme=gen_grid.original_becke,
    )


@pytest.mark.parametrize(
    ("charge", "level", "expected_points"),
    [
        (1, 0, 50),
        (1, 2, 194),
        (1, 5, 590),
        (1, 9, 1454),
        (6, 2, 302),
    ],
)
def test_default_angular_grid_matches_pyscf(charge, level, expected_points):
    assert _default_ang(charge, level) == expected_points


@pytest.mark.parametrize("npoints", [38, 50, 74, 86, 110, 194, 302, 590, 770, 1202, 1454])
def test_bundled_lebedev_tables_cover_supported_pyscf_levels(npoints):
    table = _load_lebedev_table_np(npoints)
    assert table.shape == (npoints, 4)
    np.testing.assert_allclose(np.sum(table[:, 3]), 1.0, atol=2e-15, rtol=0.0)


def test_all_supported_element_level_counts_and_constants_match_pyscf():
    from pyscf.dft import gen_grid, radi

    for charge in range(1, 37):
        assert BRAGG_RADII[charge] == pytest.approx(float(radi.BRAGG_RADII[charge]), abs=0.0)
        assert TREUTLER_XI[charge] == pytest.approx(
            float(radi._treutler_ahlrichs_xi[charge]), abs=0.0
        )
        for level in range(10):
            assert _default_rad(charge, level) == gen_grid._default_rad(charge, level)
            assert _default_ang(charge, level) == gen_grid._default_ang(charge, level)


def test_bundled_lebedev_values_match_pyscf():
    from pyscf.dft.gen_grid import MakeAngularGrid

    for npoints in BUNDLED_LEBEDEV_POINTS:
        np.testing.assert_array_equal(
            _load_lebedev_table_np(npoints),
            np.asarray(MakeAngularGrid(npoints), dtype=np.float64),
        )


@pytest.mark.parametrize(
    ("atom", "level"),
    [
        ("H 0 0 -0.37; H 0 0 0.37", 0),
        ("H 0 0 -0.37; H 0 0 0.37", 2),
        (
            "O 0 0 0.117790; H 0 0.755453 -0.471161; H 0 -0.755453 -0.471161",
            2,
        ),
    ],
)
def test_numpy_grid_coordinates_and_weights_match_pyscf(atom, level):
    coords_ref, weights_ref = _pyscf_grid(atom, level=level)
    coords, weights, _ = build_molecular_grid(atom, level=level)

    coords = np.asarray(coords)
    weights = np.asarray(weights)
    assert coords.shape == coords_ref.shape
    assert weights.shape == weights_ref.shape
    np.testing.assert_allclose(coords, coords_ref, atol=2e-13, rtol=0.0)
    np.testing.assert_allclose(weights, weights_ref, atol=2e-13, rtol=2e-13)


def test_traced_jax_grid_matches_numpy_grid():
    atom = "H 0 0 -0.37; H 0 0 0.37"
    coords_numpy, weights_numpy, spec = build_molecular_grid(atom, level=2)

    @jax.jit
    def traced_grid(coords_bohr):
        traced_spec = replace(spec, coords_bohr=coords_bohr)
        return _build_molecular_grid_from_spec_jax(traced_spec, level=2)

    coords_jax, weights_jax = traced_grid(jnp.asarray(spec.coords_bohr))
    np.testing.assert_allclose(coords_jax, coords_numpy, atol=2e-13, rtol=0.0)
    np.testing.assert_allclose(weights_jax, weights_numpy, atol=2e-13, rtol=2e-13)


def test_traced_jax_grid_geometry_gradient_matches_finite_difference():
    _, _, spec = build_molecular_grid("H 0 0 -0.37; H 0 0 0.37", level=0)
    coords0 = jnp.asarray(spec.coords_bohr)
    direction = jnp.asarray([[0.1, -0.2, 0.3], [-0.15, 0.25, -0.05]], dtype=coords0.dtype)

    def grid_moment(coords_bohr):
        coords, weights = _build_molecular_grid_from_spec_jax(
            replace(spec, coords_bohr=coords_bohr),
            level=0,
        )
        return jnp.sum(weights * jnp.exp(-jnp.einsum("rx,rx->r", coords, coords)))

    directional_ad = jnp.vdot(jax.grad(grid_moment)(coords0), direction)
    step = jnp.asarray(1e-5, dtype=coords0.dtype)
    directional_fd = (
        grid_moment(coords0 + step * direction)
        - grid_moment(coords0 - step * direction)
    ) / (2.0 * step)

    np.testing.assert_allclose(directional_ad, directional_fd, atol=2e-8, rtol=2e-7)
