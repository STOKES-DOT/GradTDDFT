import numpy as np
import pytest
import types

from pyscf_reference import (
    restricted_reference_from_pyscf_spec_with_jax_rks,
    restricted_reference_from_pyscf_with_jax_rks,
)
from td_graddft.scf.builders import restricted_molecule_from_spec_with_jax_rks
from td_graddft.scf.features import _charge_center
from td_graddft.scf import RKSConfig
from td_graddft.scf.rks import TraceableRKSResult
from td_graddft.workflows.core import run_reference
from td_graddft.workflows.types import SimulationConfig


def _pyscf_or_skip():
    try:
        from pyscf import dft, gto  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("PySCF is required for PySCF bridge tests.")


def _make_h2_b3lyp_mf():
    from pyscf import dft, gto

    mol = gto.Mole()
    mol.atom = """
    H 0.0 0.0 -0.35
    H 0.0 0.0  0.35
    """
    mol.unit = "Angstrom"
    mol.basis = "sto-3g"
    mol.spin = 0
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = "b3lyp"
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF SCF did not converge for H2 test setup.")
    return mf


def test_restricted_reference_from_pyscf_with_jax_rks_shapes_and_target_energy():
    _pyscf_or_skip()
    mf = _make_h2_b3lyp_mf()

    ref = restricted_reference_from_pyscf_with_jax_rks(
        mf,
        max_l=1,
        rks_config=RKSConfig(max_cycle=20, conv_tol=1e-8, conv_tol_density=1e-6),
        energy_target=float(mf.e_tot),
    )

    nao = ref.mo_coeff.shape[-1]
    assert ref.ao.shape[1] == nao
    assert ref.h1e.shape == (nao, nao)
    assert np.asarray(ref.rep_tensor).size == 0
    assert ref.eri_ovov is not None
    assert ref.eri_ovvo is not None
    assert ref.eri_oovv is not None
    assert ref.mo_coeff.shape[0] == 2
    assert ref.mo_occ.shape[0] == 2
    assert ref.rdm1.shape == (2, nao, nao)
    assert np.isclose(float(ref.mf_energy), float(mf.e_tot), atol=1e-12, rtol=1e-12)
    assert np.isfinite(float(ref.exact_exchange_fraction))


def test_restricted_reference_from_pyscf_with_jax_rks_jax_grid_ao_matches_pyscf_ao():
    _pyscf_or_skip()
    from pyscf.dft import numint

    mf = _make_h2_b3lyp_mf()
    ref = restricted_reference_from_pyscf_with_jax_rks(
        mf,
        max_l=1,
        rks_config=RKSConfig(max_cycle=20, conv_tol=1e-8, conv_tol_density=1e-6),
        energy_target=float(mf.e_tot),
        grid_ao_backend="jax",
    )

    ao_ref = np.asarray(numint.eval_ao(mf.mol, mf.grids.coords, deriv=0), dtype=float)
    ao1_ref = np.asarray(numint.eval_ao(mf.mol, mf.grids.coords, deriv=1), dtype=float)
    ao = np.asarray(ref.ao, dtype=float)
    ao1 = np.asarray(ref.ao_deriv1, dtype=float)

    assert np.allclose(ao, ao_ref, atol=2e-7, rtol=2e-6)
    assert np.allclose(ao1, ao1_ref, atol=3e-6, rtol=2e-5)


def test_build_rks_integral_inputs_accepts_strict_jax_default_grid_level():
    _pyscf_or_skip()
    from pyscf import dft, gto

    from td_graddft.scf.inputs import build_rks_integral_inputs

    atom = """
    H 0.0 0.0 -0.35
    H 0.0 0.0  0.35
    """
    mol = gto.M(
        atom=atom,
        basis="sto-3g",
        unit="Angstrom",
        spin=0,
        cart=True,
        verbose=0,
    )
    mf = dft.RKS(mol)
    assert mf.grids.level == 3

    level0_inputs = build_rks_integral_inputs(
        atom=atom,
        basis="sto-3g",
        xc_spec="b3lyp",
        unit="Angstrom",
        spin=0,
        cart=True,
        grids_level=0,
        max_l=1,
        integral_backend="libcint",
        config=RKSConfig(max_cycle=1),
    )
    level3_inputs = build_rks_integral_inputs(
        atom=atom,
        basis="sto-3g",
        xc_spec="b3lyp",
        unit="Angstrom",
        spin=0,
        cart=True,
        grids_level=3,
        max_l=1,
        integral_backend="libcint",
        config=RKSConfig(max_cycle=1),
    )

    assert level3_inputs.grid_ao_backend == "jax"
    assert level3_inputs.coords.shape[0] > level0_inputs.coords.shape[0]
    assert level3_inputs.grid_weights.shape == (level3_inputs.coords.shape[0],)
    assert np.all(np.isfinite(np.asarray(level3_inputs.grid_weights)))


def test_workflow_reference_stage_rejects_legacy_mf_jax_rks_backend():
    _pyscf_or_skip()
    mf = _make_h2_b3lyp_mf()

    simulation = SimulationConfig(
        nstates=1,
        scf_backend="jax_rks",
        jax_grid_ao_backend="jax",
        jax_basis_max_l=1,
        jax_rks_max_cycle=20,
        jax_rks_conv_tol=1e-8,
        jax_rks_conv_tol_density=1e-6,
    )
    with pytest.raises(ValueError, match="reference_spec"):
        run_reference(
            mf,
            scf_elapsed_s=0.0,
            simulation=simulation,
        )


def test_restricted_reference_from_pyscf_spec_with_jax_rks_matches_water_pyscf_energy():
    _pyscf_or_skip()
    from pyscf import dft, gto

    atom = """
    O  0.000000  0.000000  0.117790
    H  0.000000  0.755453 -0.471161
    H  0.000000 -0.755453 -0.471161
    """
    mol = gto.M(
        atom=atom,
        basis="sto-3g",
        unit="Angstrom",
        spin=0,
        cart=True,
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = "pbe"
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF SCF did not converge for water test setup.")

    ref = restricted_reference_from_pyscf_spec_with_jax_rks(
        atom=atom,
        basis="sto-3g",
        unit="Angstrom",
        xc_spec="pbe",
        spin=0,
        charge=0,
        cart=True,
        grids_level=0,
        max_l=1,
        rks_config=RKSConfig(
            xc_spec="pbe",
            max_cycle=50,
            conv_tol=1e-9,
            conv_tol_density=1e-7,
            damping=0.15,
            density_floor=1e-12,
            potential_clip=20.0,
        ),
        grid_ao_backend="jax",
    )

    assert np.isclose(float(ref.mf_energy), float(mf.e_tot), atol=2e-5, rtol=2e-7)
    with mol.with_common_orig(_charge_center(mol)):
        dipole_ref = np.asarray(mol.intor_symmetric("int1e_r", comp=3), dtype=float)
    dipole = np.asarray(ref.dipole_integrals, dtype=float)
    assert np.allclose(dipole, dipole_ref, atol=2e-6, rtol=2e-6)


def test_restricted_molecule_from_spec_with_jax_rks_direct_backend_matches_water_pyscf_energy():
    _pyscf_or_skip()
    from pyscf import dft, gto

    atom = """
    O  0.000000  0.000000  0.117790
    H  0.000000  0.755453 -0.471161
    H  0.000000 -0.755453 -0.471161
    """
    mol = gto.M(
        atom=atom,
        basis="sto-3g",
        unit="Angstrom",
        spin=0,
        cart=True,
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = "pbe0"
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF SCF did not converge for direct-RKS water test setup.")

    ref = restricted_molecule_from_spec_with_jax_rks(
        atom=atom,
        basis="sto-3g",
        unit="Angstrom",
        xc_spec="pbe0",
        spin=0,
        charge=0,
        cart=True,
        grids_level=0,
        max_l=1,
        rks_config=RKSConfig(
            xc_spec="pbe0",
            max_cycle=50,
            conv_tol=1e-9,
            conv_tol_density=1e-7,
            damping=0.15,
            density_floor=1e-12,
            potential_clip=20.0,
            jk_backend="direct",
            direct_scf_tol=0.0,
        ),
        grid_ao_backend="jax",
        integral_backend="jax",
    )

    assert np.asarray(ref.rep_tensor).size == 0
    assert ref.eri_ovov is not None
    assert ref.eri_ovvo is not None
    assert ref.eri_oovv is not None
    assert np.isclose(float(ref.mf_energy), float(mf.e_tot), atol=2e-4, rtol=2e-6)


def test_restricted_reference_from_pyscf_spec_with_jax_rks_matches_water_local_hfx():
    _pyscf_or_skip()
    from pyscf import dft, gto

    atom = """
    O  0.000000  0.000000  0.117790
    H  0.000000  0.755453 -0.471161
    H  0.000000 -0.755453 -0.471161
    """
    mol = gto.M(
        atom=atom,
        basis="sto-3g",
        unit="Angstrom",
        spin=0,
        cart=True,
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = "pbe"
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF SCF did not converge for water local-HFX test setup.")

    ref_jax = restricted_reference_from_pyscf_spec_with_jax_rks(
        atom=atom,
        basis="sto-3g",
        unit="Angstrom",
        xc_spec="pbe",
        spin=0,
        charge=0,
        cart=True,
        grids_level=0,
        max_l=1,
        rks_config=RKSConfig(
            xc_spec="pbe",
            max_cycle=50,
            conv_tol=1e-9,
            conv_tol_density=1e-7,
            damping=0.15,
            density_floor=1e-12,
            potential_clip=20.0,
        ),
        grid_ao_backend="jax",
        compute_local_hfx_features=True,
        compute_local_hfx_aux=True,
        hfx_omega_values=(0.0, 0.4),
        hfx_chunk_size=128,
    )

    assert ref_jax.hfx_local is not None
    assert ref_jax.hfx_nu is not None
    coords = np.asarray(ref_jax.grid.coords, dtype=float)
    ao = np.asarray(ref_jax.ao, dtype=float)
    dm_half = np.asarray(ref_jax.rdm1[0], dtype=float)
    e = ao @ dm_half

    nu_ref_list = []
    for omega in (0.0, 0.4):
        if omega == 0.0:
            nu_ref = np.asarray(mol.intor("int1e_grids_cart", hermi=1, grids=coords), dtype=float)
        else:
            with mol.with_range_coulomb(omega=omega):
                nu_ref = np.asarray(
                    mol.intor("int1e_grids_cart", hermi=1, grids=coords),
                    dtype=float,
                )
        nu_ref_list.append(nu_ref)
    nu_ref_stack = np.stack(nu_ref_list, axis=0)
    fxx_ref = np.einsum("wgbc,gc->wgb", nu_ref_stack, e, optimize=True)
    exx_ref = -0.5 * np.einsum("gq,wgq->wg", e, fxx_ref, optimize=True)
    hfx_ref = np.stack([exx_ref.T, exx_ref.T], axis=0)

    assert np.allclose(
        np.asarray(ref_jax.hfx_nu, dtype=float),
        nu_ref_stack,
        atol=3e-5,
        rtol=3e-5,
    )
    assert np.allclose(
        np.asarray(ref_jax.hfx_local, dtype=float),
        hfx_ref,
        atol=5e-5,
        rtol=5e-5,
    )


def test_restricted_molecule_from_spec_with_jax_rks_can_precompile_eri(monkeypatch):
    calls: list[tuple[int, str, int]] = []

    def fake_precompile(basis, *, engine="auto", chunk_size=512):
        calls.append((basis.nao, str(engine), int(chunk_size)))
        return {"compiled_shell_signatures": 0, "compiled_batch_shapes": 0}

    monkeypatch.setattr("td_graddft.scf.builders.precompile_eri_kernels", fake_precompile)

    ref = restricted_molecule_from_spec_with_jax_rks(
        atom="""
        H 0.0 0.0 -0.35
        H 0.0 0.0  0.35
        """,
        basis="sto-3g",
        unit="Angstrom",
        xc_spec="pbe",
        spin=0,
        charge=0,
        cart=True,
        grids_level=0,
        max_l=1,
        rks_config=RKSConfig(
            xc_spec="pbe",
            max_cycle=20,
            conv_tol=1e-8,
            conv_tol_density=1e-6,
            damping=0.15,
            density_floor=1e-12,
            potential_clip=20.0,
        ),
        grid_ao_backend="jax",
        integral_backend="jax",
        precompile_eri=True,
        precompile_eri_chunk_size=64,
    )

    assert ref.h1e.ndim == 2
    assert calls
    assert calls[0][1] == "jit"
    assert calls[0][2] == 64


def test_restricted_molecule_from_spec_with_jax_rks_libcint_matches_jax():
    _pyscf_or_skip()

    atom = """
    H 0.0 0.0 -0.35
    H 0.0 0.0  0.35
    """
    cfg = RKSConfig(
        xc_spec="pbe",
        max_cycle=40,
        conv_tol=1e-10,
        conv_tol_density=1e-8,
        damping=0.1,
        density_floor=1e-12,
        potential_clip=20.0,
    )
    ref_jax = restricted_molecule_from_spec_with_jax_rks(
        atom=atom,
        basis="sto-3g",
        unit="Angstrom",
        xc_spec="pbe",
        spin=0,
        charge=0,
        cart=True,
        grids_level=0,
        max_l=1,
        rks_config=cfg,
        grid_ao_backend="jax",
        integral_backend="jax",
    )
    ref_libcint = restricted_molecule_from_spec_with_jax_rks(
        atom=atom,
        basis="sto-3g",
        unit="Angstrom",
        xc_spec="pbe",
        spin=0,
        charge=0,
        cart=True,
        grids_level=0,
        max_l=1,
        rks_config=cfg,
        grid_ao_backend="jax",
        integral_backend="libcint",
    )

    assert np.isclose(float(ref_jax.mf_energy), float(ref_libcint.mf_energy), atol=1e-6, rtol=0.0)
    assert np.allclose(np.asarray(ref_jax.overlap_matrix), np.asarray(ref_libcint.overlap_matrix), atol=1e-7, rtol=1e-7)
    assert np.allclose(np.asarray(ref_jax.h1e), np.asarray(ref_libcint.h1e), atol=5e-7, rtol=1e-7)
    assert np.allclose(np.asarray(ref_jax.rep_tensor), np.asarray(ref_libcint.rep_tensor), atol=3e-7, rtol=1e-7)


def test_restricted_molecule_from_spec_with_jax_rks_libcint_full_uses_packed_eri(monkeypatch):
    _pyscf_or_skip()
    from pyscf import gto

    orig_intor = gto.mole.Mole.intor
    seen_aosym: list[str] = []

    def _guarded_intor(self, intor, *args, **kwargs):
        if str(intor) == "int2e_cart":
            aosym = str(kwargs.get("aosym", "s1"))
            seen_aosym.append(aosym)
            if aosym not in {"s4", "s8"}:
                raise AssertionError("default no-DF RKS must not build dense int2e_cart")
        return orig_intor(self, intor, *args, **kwargs)

    monkeypatch.setattr(gto.mole.Mole, "intor", _guarded_intor)

    ref = restricted_molecule_from_spec_with_jax_rks(
        atom="""
        H 0.0 0.0 -0.35
        H 0.0 0.0  0.35
        """,
        basis="sto-3g",
        unit="Angstrom",
        xc_spec="pbe0",
        spin=0,
        charge=0,
        cart=True,
        grids_level=0,
        max_l=1,
        rks_config=RKSConfig(
            xc_spec="pbe0",
            max_cycle=30,
            conv_tol=1e-9,
            conv_tol_density=1e-7,
            damping=0.1,
            density_floor=1e-12,
            potential_clip=20.0,
        ),
        grid_ao_backend="jax",
        integral_backend="libcint",
    )

    assert "s4" in seen_aosym
    assert np.asarray(ref.rep_tensor).size == 0
    assert ref.eri_ovov is not None
    assert ref.eri_ovvo is not None
    assert ref.eri_oovv is not None
    assert np.isfinite(float(ref.mf_energy))


def test_restricted_molecule_from_spec_with_jax_rks_libcint_skips_precompile(monkeypatch):
    _pyscf_or_skip()

    called = {"value": False}

    def fake_precompile(*args, **kwargs):
        called["value"] = True
        return {}

    monkeypatch.setattr("td_graddft.scf.builders.precompile_eri_kernels", fake_precompile)

    with pytest.warns(RuntimeWarning, match="ignored when integral_backend='libcint'"):
        _ = restricted_molecule_from_spec_with_jax_rks(
            atom="""
            H 0.0 0.0 -0.35
            H 0.0 0.0  0.35
            """,
            basis="sto-3g",
            unit="Angstrom",
            xc_spec="pbe",
            spin=0,
            charge=0,
            cart=True,
            grids_level=0,
            max_l=1,
            rks_config=RKSConfig(
                xc_spec="pbe",
                max_cycle=20,
                conv_tol=1e-8,
                conv_tol_density=1e-6,
                damping=0.15,
                density_floor=1e-12,
                potential_clip=20.0,
            ),
            grid_ao_backend="jax",
            integral_backend="libcint",
            precompile_eri=True,
        )

    assert called["value"] is False


def test_restricted_molecule_from_spec_with_jax_rks_libcint_zero_policy_runs():
    _pyscf_or_skip()

    ref = restricted_molecule_from_spec_with_jax_rks(
        atom="""
        H 0.0 0.0 -0.35
        H 0.0 0.0  0.35
        """,
        basis="sto-3g",
        unit="Angstrom",
        xc_spec="pbe",
        spin=0,
        charge=0,
        cart=True,
        grids_level=0,
        max_l=1,
        rks_config=RKSConfig(
            xc_spec="pbe",
            max_cycle=20,
            conv_tol=1e-8,
            conv_tol_density=1e-6,
            damping=0.15,
            density_floor=1e-12,
            potential_clip=20.0,
        ),
        grid_ao_backend="jax",
        integral_backend="libcint",
        libcint_geometry_grad_policy="zero",
    )
    assert np.isfinite(float(ref.mf_energy))


def test_restricted_molecule_from_spec_with_jax_rks_invalid_libcint_policy_raises():
    with pytest.raises(ValueError, match="Unsupported libcint_geometry_grad_policy"):
        _ = restricted_molecule_from_spec_with_jax_rks(
            atom="""
            H 0.0 0.0 -0.35
            H 0.0 0.0  0.35
            """,
            basis="sto-3g",
            unit="Angstrom",
            xc_spec="pbe",
            spin=0,
            charge=0,
            cart=True,
            grids_level=0,
            max_l=1,
            rks_config=RKSConfig(
                xc_spec="pbe",
                max_cycle=20,
                conv_tol=1e-8,
                conv_tol_density=1e-6,
                damping=0.15,
                density_floor=1e-12,
                potential_clip=20.0,
            ),
            grid_ao_backend="jax",
            integral_backend="libcint",
            libcint_geometry_grad_policy="invalid_policy",  # type: ignore[arg-type]
        )
