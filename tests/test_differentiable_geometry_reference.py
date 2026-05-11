import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from td_graddft.data.molecule import MoleculeSpec, parse_molecule_spec
from td_graddft.scf.builders import (
    restricted_reference_from_spec_with_jax_rks,
    unrestricted_reference_from_spec_with_jax_uks,
)
from td_graddft.scf import RKSConfig, UKSConfig
from td_graddft.spectra import oscillator_strengths
from td_graddft.tddft import RestrictedCasidaTDDFT


def _pyscf_or_skip():
    try:
        import pyscf  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("PySCF is required for libcint geometry-gradient tests.")


def test_parse_molecule_spec_preserves_jax_coordinate_arrays():
    coords = jnp.asarray(
        [
            [0.0, 0.0, -0.7],
            [0.0, 0.0, 0.7],
        ],
        dtype=jnp.float64,
    )

    spec = parse_molecule_spec(
        [("H", coords[0]), ("H", coords[1])],
        unit="Bohr",
    )

    assert spec.symbols == ("H", "H")
    assert np.allclose(np.asarray(spec.coords_bohr), np.asarray(coords), atol=0.0, rtol=0.0)


def test_restricted_jax_reference_energy_is_differentiable_by_geometry():
    charges = jnp.asarray([1.0, 1.0], dtype=jnp.float64)

    def energy(coords_bohr):
        spec = MoleculeSpec(
            symbols=("H", "H"),
            coords_bohr=coords_bohr,
            charges=charges,
            charge=0,
            spin=0,
            unit="Bohr",
        )
        ref = restricted_reference_from_spec_with_jax_rks(
            atom=spec,
            basis="sto-3g",
            xc_spec="hf",
            unit="Bohr",
            charge=0,
            spin=0,
            cart=True,
            grids_level=0,
            max_l=1,
            integral_backend="jax",
            grid_ao_backend="jax",
            rks_config=RKSConfig(
                xc_spec="hf",
                max_cycle=8,
                conv_tol=1e-9,
                conv_tol_density=1e-7,
                damping=0.0,
                jk_backend="full",
            ),
        )
        return jnp.asarray(ref.mf_energy)

    coords0 = jnp.asarray(
        [
            [0.0, 0.0, -0.7],
            [0.0, 0.0, 0.7],
        ],
        dtype=jnp.float64,
    )
    grad = jax.grad(energy)(coords0)

    assert grad.shape == coords0.shape
    assert np.isfinite(np.asarray(grad)).all()
    assert abs(float(grad[0, 2])) > 1e-5
    assert np.allclose(np.asarray(grad[0]), -np.asarray(grad[1]), atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("jk_backend", ["full", "direct"])
def test_restricted_libcint_reference_energy_gradient_matches_pyscf_rhf(jk_backend):
    _pyscf_or_skip()
    from pyscf import gto, scf

    charges = jnp.asarray([1.0, 1.0], dtype=jnp.float64)

    def energy(coords_bohr):
        spec = MoleculeSpec(
            symbols=("H", "H"),
            coords_bohr=coords_bohr,
            charges=charges,
            charge=0,
            spin=0,
            unit="Bohr",
        )
        ref = restricted_reference_from_spec_with_jax_rks(
            atom=spec,
            basis="sto-3g",
            xc_spec="hf",
            unit="Bohr",
            charge=0,
            spin=0,
            cart=True,
            grids_level=0,
            max_l=1,
            integral_backend="libcint",
            grid_ao_backend="jax",
            libcint_geometry_grad_policy="analytic",
            rks_config=RKSConfig(
                xc_spec="hf",
                max_cycle=20,
                conv_tol=1e-11,
                conv_tol_density=1e-9,
                damping=0.0,
                jk_backend=jk_backend,
            ),
        )
        return jnp.asarray(ref.mf_energy)

    coords0 = jnp.asarray(
        [
            [0.0, 0.0, -0.7],
            [0.0, 0.0, 0.7],
        ],
        dtype=jnp.float64,
    )
    grad = jax.grad(energy)(coords0)

    mol = gto.M(
        atom=[("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))],
        basis="sto-3g",
        unit="Bohr",
        spin=0,
        charge=0,
        cart=True,
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF RHF did not converge for H2 gradient reference.")
    pyscf_grad = mf.nuc_grad_method().kernel()

    assert np.allclose(np.asarray(energy(coords0)), mf.e_tot, atol=1e-8, rtol=1e-8)
    assert np.allclose(np.asarray(grad), pyscf_grad, atol=3e-6, rtol=3e-6)


def test_restricted_libcint_tda_oscillator_strength_is_differentiable_by_geometry():
    _pyscf_or_skip()
    charges = jnp.asarray([1.0, 1.0], dtype=jnp.float64)

    def oscillator_sum(coords_bohr):
        spec = MoleculeSpec(
            symbols=("H", "H"),
            coords_bohr=coords_bohr,
            charges=charges,
            charge=0,
            spin=0,
            unit="Bohr",
        )
        ref = restricted_reference_from_spec_with_jax_rks(
            atom=spec,
            basis="sto-3g",
            xc_spec="hf",
            unit="Bohr",
            charge=0,
            spin=0,
            cart=True,
            grids_level=0,
            max_l=1,
            integral_backend="libcint",
            grid_ao_backend="jax",
            libcint_geometry_grad_policy="analytic",
            rks_config=RKSConfig(
                xc_spec="hf",
                max_cycle=20,
                conv_tol=1e-11,
                conv_tol_density=1e-9,
                damping=0.0,
                jk_backend="full",
            ),
        )
        tda = RestrictedCasidaTDDFT(ref, eigensolver="dense").tda(nstates=1)
        return jnp.sum(oscillator_strengths(ref, tda))

    coords0 = jnp.asarray(
        [
            [0.0, 0.0, -0.7],
            [0.0, 0.0, 0.7],
        ],
        dtype=jnp.float64,
    )
    grad = jax.grad(oscillator_sum)(coords0)

    assert np.isfinite(np.asarray(oscillator_sum(coords0)))
    assert np.isfinite(np.asarray(grad)).all()
    assert np.allclose(np.asarray(grad[0]), -np.asarray(grad[1]), atol=1e-7, rtol=1e-7)
    assert abs(float(grad[0, 2])) > 1e-5


def test_restricted_libcint_df_reference_energy_gradient_matches_finite_difference():
    _pyscf_or_skip()
    charges = jnp.asarray([1.0, 1.0], dtype=jnp.float64)

    def energy(coords_bohr):
        spec = MoleculeSpec(
            symbols=("H", "H"),
            coords_bohr=coords_bohr,
            charges=charges,
            charge=0,
            spin=0,
            unit="Bohr",
        )
        ref = restricted_reference_from_spec_with_jax_rks(
            atom=spec,
            basis="sto-3g",
            xc_spec="hf",
            unit="Bohr",
            charge=0,
            spin=0,
            cart=True,
            grids_level=0,
            max_l=1,
            integral_backend="libcint",
            grid_ao_backend="jax",
            libcint_geometry_grad_policy="analytic",
            rks_config=RKSConfig(
                xc_spec="hf",
                max_cycle=20,
                conv_tol=1e-11,
                conv_tol_density=1e-9,
                damping=0.0,
                jk_backend="df",
                df_tol=1e-12,
            ),
        )
        return jnp.asarray(ref.mf_energy)

    coords0 = jnp.asarray(
        [
            [0.0, 0.0, -0.7],
            [0.0, 0.0, 0.7],
        ],
        dtype=jnp.float64,
    )
    grad = jax.grad(energy)(coords0)

    step = 1e-4
    plus = coords0.at[0, 2].add(step).at[1, 2].add(-step)
    minus = coords0.at[0, 2].add(-step).at[1, 2].add(step)
    fd_directional = (energy(plus) - energy(minus)) / (2.0 * step)
    ad_directional = grad[0, 2] - grad[1, 2]

    assert np.isfinite(np.asarray(energy(coords0)))
    assert np.isfinite(np.asarray(grad)).all()
    assert np.allclose(np.asarray(ad_directional), np.asarray(fd_directional), atol=2e-5, rtol=2e-5)


def test_unrestricted_libcint_reference_energy_is_differentiable_by_geometry():
    _pyscf_or_skip()
    charges = jnp.asarray([1.0, 1.0], dtype=jnp.float64)

    def energy(coords_bohr):
        spec = MoleculeSpec(
            symbols=("H", "H"),
            coords_bohr=coords_bohr,
            charges=charges,
            charge=1,
            spin=1,
            unit="Bohr",
        )
        ref = unrestricted_reference_from_spec_with_jax_uks(
            atom=spec,
            basis="sto-3g",
            xc_spec="hf",
            unit="Bohr",
            charge=1,
            spin=1,
            cart=True,
            grids_level=0,
            max_l=1,
            integral_backend="libcint",
            grid_ao_backend="jax",
            libcint_geometry_grad_policy="analytic",
            uks_config=UKSConfig(
                xc_spec="hf",
                max_cycle=20,
                conv_tol=1e-11,
                conv_tol_density=1e-9,
                damping=0.0,
            ),
        )
        return jnp.asarray(ref.mf_energy)

    coords0 = jnp.asarray(
        [
            [0.0, 0.0, -0.7],
            [0.0, 0.0, 0.7],
        ],
        dtype=jnp.float64,
    )
    grad = jax.grad(energy)(coords0)

    assert np.isfinite(np.asarray(energy(coords0)))
    assert np.isfinite(np.asarray(grad)).all()
    assert np.allclose(np.asarray(grad[0]), -np.asarray(grad[1]), atol=1e-7, rtol=1e-7)
    assert abs(float(grad[0, 2])) > 1e-5
