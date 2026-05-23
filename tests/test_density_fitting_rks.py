import numpy as np
import pytest

from td_graddft.data import basis_from_pyscf_spec, evaluate_cartesian_ao
from td_graddft.data.integrals import build_hcore, eri_pair_matrix_packed, eri_tensor, overlap_matrix
from td_graddft.df import (
    build_j_from_df,
    build_jk_from_df,
    build_jk_from_df_orbitals,
    eri_to_df_factors,
    true_df_factors_from_libcint_mol,
)
from td_graddft.scf import RKSConfig, run_rks_from_integrals
from td_graddft.scf.builders import restricted_molecule_from_spec_with_jax_rks
from td_graddft.scf.features import _restricted_response_eri_slices_from_mo_tensor
from td_graddft.data.integrals.jax.direct_jk import build_direct_jk_from_basis, build_direct_jk_incremental
from td_graddft.data.integrals.jax.packed_eri import build_jk_from_eri_pair_matrix, eri_pair_matrix_to_mo_eri_slices
from td_graddft.tddft import RestrictedCasidaTDDFT
from td_graddft.tddft._semilocal_response import SemilocalResponseFunctional


def _pyscf_or_skip():
    try:
        from pyscf import dft, gto  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("PySCF is required for density-fitting comparison tests.")


def _water_mol():
    from pyscf import gto

    return gto.M(
        atom="""
        O  0.000000  0.000000  0.117790
        H  0.000000  0.755453 -0.471161
        H  0.000000 -0.755453 -0.471161
        """,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        verbose=0,
    )


def test_df_jk_matches_dense_eri_contractions_for_water():
    _pyscf_or_skip()
    mol = _water_mol()
    basis = basis_from_pyscf_spec(
        mol.atom,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=3,
    )
    eri = np.asarray(eri_tensor(basis), dtype=float)
    rng = np.random.default_rng(0)
    density = rng.normal(size=(mol.nao_nr(), mol.nao_nr()))
    density = 0.5 * (density + density.T)

    j_ref = np.einsum("pqrs,rs->pq", eri, density, optimize=True)
    k_ref = np.einsum("prqs,rs->pq", eri, density, optimize=True)
    factors = eri_to_df_factors(eri, tol=1e-10)
    j_df, k_df = build_jk_from_df(factors, density)

    assert np.allclose(np.asarray(j_df), j_ref, atol=2e-5, rtol=2e-5)
    assert np.allclose(np.asarray(k_df), k_ref, atol=2e-5, rtol=2e-5)


def test_df_j_only_matches_dense_coulomb_for_water():
    _pyscf_or_skip()
    mol = _water_mol()
    basis = basis_from_pyscf_spec(
        mol.atom,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=3,
    )
    eri = np.asarray(eri_tensor(basis), dtype=float)
    rng = np.random.default_rng(1)
    density = rng.normal(size=(mol.nao_nr(), mol.nao_nr()))
    density = 0.5 * (density + density.T)

    j_ref = np.einsum("pqrs,rs->pq", eri, density, optimize=True)
    factors = eri_to_df_factors(eri, tol=1e-10)
    j_df = build_j_from_df(factors, density)

    assert np.allclose(np.asarray(j_df), j_ref, atol=2e-5, rtol=2e-5)


def test_df_orbital_jk_matches_density_jk_for_closed_shell_water():
    _pyscf_or_skip()
    mol = _water_mol()
    basis = basis_from_pyscf_spec(
        mol.atom,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=3,
    )
    eri = np.asarray(eri_tensor(basis), dtype=float)
    factors = eri_to_df_factors(eri, tol=1e-10)

    rng = np.random.default_rng(4)
    q, _ = np.linalg.qr(rng.normal(size=(mol.nao_nr(), mol.nao_nr())))
    nocc = 3
    mo_occ = np.zeros((mol.nao_nr(),), dtype=float)
    mo_occ[:nocc] = 2.0
    density = q @ np.diag(mo_occ) @ q.T

    j_density, k_density = build_jk_from_df(factors, density)
    j_orb, k_orb = build_jk_from_df_orbitals(
        factors,
        density,
        q,
        mo_occ,
        nocc=nocc,
    )

    assert np.allclose(np.asarray(j_orb), np.asarray(j_density), atol=2e-5, rtol=2e-5)
    assert np.allclose(np.asarray(k_orb), np.asarray(k_density), atol=2e-5, rtol=2e-5)


def test_true_df_factors_from_libcint_mol_match_dense_jk_for_water():
    _pyscf_or_skip()
    mol = _water_mol()
    eri = np.asarray(mol.intor("int2e"), dtype=float)
    rng = np.random.default_rng(3)
    density = rng.normal(size=(mol.nao_nr(), mol.nao_nr()))
    density = 0.5 * (density + density.T)

    j_ref = np.einsum("pqrs,rs->pq", eri, density, optimize=True)
    k_ref = np.einsum("prqs,rs->pq", eri, density, optimize=True)
    factors = true_df_factors_from_libcint_mol(mol)
    j_df, k_df = build_jk_from_df(factors, density)

    assert np.allclose(np.asarray(j_df), j_ref, atol=3e-4, rtol=3e-4)
    assert np.allclose(np.asarray(k_df), k_ref, atol=3e-4, rtol=3e-4)


def test_packed_pair_matrix_matches_full_eri_for_water():
    _pyscf_or_skip()
    mol = _water_mol()
    basis = basis_from_pyscf_spec(
        mol.atom,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=3,
    )
    eri = np.asarray(eri_tensor(basis), dtype=float)
    pair = np.asarray(eri_pair_matrix_packed(basis), dtype=float)
    rows, cols = np.tril_indices(mol.nao_nr())
    pair_ref = eri[rows, cols][:, rows, cols]
    assert np.allclose(pair, pair_ref, atol=2e-8, rtol=2e-8)


def test_packed_no_df_jk_matches_dense_eri_contractions_for_water():
    _pyscf_or_skip()
    mol = _water_mol()
    basis = basis_from_pyscf_spec(
        mol.atom,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=3,
    )
    eri = np.asarray(eri_tensor(basis), dtype=float)
    pair = np.asarray(eri_pair_matrix_packed(basis), dtype=float)
    rng = np.random.default_rng(5)
    density = rng.normal(size=(mol.nao_nr(), mol.nao_nr()))
    density = 0.5 * (density + density.T)

    j_ref = np.einsum("pqrs,rs->pq", eri, density, optimize=True)
    k_ref = np.einsum("prqs,rs->pq", eri, density, optimize=True)
    j_pair, k_pair = build_jk_from_eri_pair_matrix(pair, density)

    assert np.allclose(np.asarray(j_pair), j_ref, atol=2e-5, rtol=2e-5)
    assert np.allclose(np.asarray(k_pair), k_ref, atol=2e-5, rtol=2e-5)


def test_direct_no_df_jk_matches_dense_eri_contractions_for_water():
    _pyscf_or_skip()
    mol = _water_mol()
    basis = basis_from_pyscf_spec(
        mol.atom,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=3,
    )
    eri = np.asarray(eri_tensor(basis), dtype=float)
    rng = np.random.default_rng(7)
    density = rng.normal(size=(mol.nao_nr(), mol.nao_nr()))
    density = 0.5 * (density + density.T)

    j_ref = np.einsum("pqrs,rs->pq", eri, density, optimize=True)
    k_ref = np.einsum("prqs,rs->pq", eri, density, optimize=True)
    direct_jk = build_direct_jk_from_basis(basis, density, screening_threshold=0.0)

    assert np.allclose(np.asarray(direct_jk.j), j_ref, atol=2e-5, rtol=2e-5)
    assert np.allclose(np.asarray(direct_jk.k), k_ref, atol=2e-5, rtol=2e-5)


def test_direct_no_df_without_screening_uses_packed_jk_path(monkeypatch):
    _pyscf_or_skip()
    mol = _water_mol()
    basis = basis_from_pyscf_spec(
        mol.atom,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=3,
    )
    rng = np.random.default_rng(11)
    density = rng.normal(size=(mol.nao_nr(), mol.nao_nr()))
    density = 0.5 * (density + density.T)

    import td_graddft.data.integrals.jax.direct_jk as direct_jk_mod

    def _fail_kernel(*args, **kwargs):
        raise AssertionError("unscreened direct J/K should reuse the packed ERI path")

    monkeypatch.setattr(direct_jk_mod, "_run_quartet_kernel_chunked", _fail_kernel)
    direct_jk = build_direct_jk_from_basis(basis, density, screening_threshold=0.0)
    pair_j, pair_k = build_jk_from_eri_pair_matrix(eri_pair_matrix_packed(basis), density)

    assert np.allclose(np.asarray(direct_jk.j), np.asarray(pair_j), atol=1e-10, rtol=1e-10)
    assert np.allclose(np.asarray(direct_jk.k), np.asarray(pair_k), atol=1e-10, rtol=1e-10)


def test_direct_no_df_incremental_update_matches_rebuild_for_water():
    _pyscf_or_skip()
    mol = _water_mol()
    basis = basis_from_pyscf_spec(
        mol.atom,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=3,
    )
    rng = np.random.default_rng(8)
    density0 = rng.normal(size=(mol.nao_nr(), mol.nao_nr()))
    density0 = 0.5 * (density0 + density0.T)
    density1 = density0 + 0.03 * rng.normal(size=(mol.nao_nr(), mol.nao_nr()))
    density1 = 0.5 * (density1 + density1.T)

    jk0 = build_direct_jk_from_basis(basis, density0, screening_threshold=0.0)
    jk1_ref = build_direct_jk_from_basis(basis, density1, screening_threshold=0.0)
    jk1_inc = build_direct_jk_incremental(
        basis,
        density1,
        density_last=density0,
        j_last=jk0.j,
        k_last=jk0.k,
        screening_threshold=0.0,
    )

    assert np.allclose(np.asarray(jk1_inc.j), np.asarray(jk1_ref.j), atol=2e-5, rtol=2e-5)
    assert np.allclose(np.asarray(jk1_inc.k), np.asarray(jk1_ref.k), atol=2e-5, rtol=2e-5)


def test_direct_no_df_shell_screening_skips_screened_quartets(monkeypatch):
    _pyscf_or_skip()
    mol = _water_mol()
    basis = basis_from_pyscf_spec(
        mol.atom,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=3,
    )
    density = np.eye(mol.nao_nr())

    def _fail_kernel(*args, **kwargs):
        raise AssertionError("screened shell quartets should not run the ERI kernel")

    import td_graddft.data.integrals.jax.direct_jk as direct_jk_mod

    monkeypatch.setattr(direct_jk_mod, "_run_quartet_kernel_chunked", _fail_kernel)
    shell_pair_bounds = np.zeros((len(basis.shells), len(basis.shells)))
    direct_jk = build_direct_jk_from_basis(
        basis,
        density,
        screening_threshold=1e-12,
        shell_pair_schwarz_bounds=shell_pair_bounds,
    )

    assert np.allclose(np.asarray(direct_jk.j), 0.0, atol=0.0)
    assert np.allclose(np.asarray(direct_jk.k), 0.0, atol=0.0)


def test_rks_full_backend_accepts_packed_no_df_eri_for_water():
    _pyscf_or_skip()
    from pyscf import dft

    mol = _water_mol()
    basis = basis_from_pyscf_spec(
        mol.atom,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=1,
    )
    s = overlap_matrix(basis)
    h = build_hcore(basis)
    pair = eri_pair_matrix_packed(basis)

    mf = dft.RKS(mol)
    mf.xc = "pbe0"
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    assert mf.converged

    coords = np.asarray(mf.grids.coords, dtype=float)
    weights = np.asarray(mf.grids.weights, dtype=float)
    ao_deriv1 = evaluate_cartesian_ao(basis, coords, deriv=1)
    ao = ao_deriv1[0]

    cfg = RKSConfig(
        xc_spec="pbe0",
        max_cycle=80,
        conv_tol=1e-9,
        conv_tol_density=1e-6,
        damping=0.15,
        potential_clip=20.0,
        jk_backend="full",
    )
    out = run_rks_from_integrals(
        overlap=s,
        hcore=h,
        eri=None,
        eri_pair_matrix=pair,
        nelectron=mol.nelectron,
        nuclear_repulsion=float(mol.energy_nuc()),
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid_weights=weights,
        config=cfg,
    )

    assert out.converged
    assert np.isclose(out.total_energy, mf.e_tot, atol=2e-4, rtol=2e-6)


def test_rks_direct_backend_accepts_basis_without_prebuilt_eri_for_water():
    _pyscf_or_skip()
    from pyscf import dft

    mol = _water_mol()
    basis = basis_from_pyscf_spec(
        mol.atom,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=1,
    )
    s = overlap_matrix(basis)
    h = build_hcore(basis)

    mf = dft.RKS(mol)
    mf.xc = "pbe0"
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    assert mf.converged

    coords = np.asarray(mf.grids.coords, dtype=float)
    weights = np.asarray(mf.grids.weights, dtype=float)
    ao_deriv1 = evaluate_cartesian_ao(basis, coords, deriv=1)
    ao = ao_deriv1[0]

    cfg = RKSConfig(
        xc_spec="pbe0",
        max_cycle=80,
        conv_tol=1e-9,
        conv_tol_density=1e-6,
        damping=0.15,
        potential_clip=20.0,
        jk_backend="direct",
        direct_scf_tol=0.0,
    )
    out = run_rks_from_integrals(
        overlap=s,
        hcore=h,
        eri=None,
        direct_basis=basis,
        nelectron=mol.nelectron,
        nuclear_repulsion=float(mol.energy_nuc()),
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid_weights=weights,
        config=cfg,
    )

    assert out.converged
    assert np.isclose(out.total_energy, mf.e_tot, atol=2e-4, rtol=2e-6)


def test_rks_direct_backend_matches_pyscf_for_water():
    _pyscf_or_skip()
    from pyscf import dft

    mol = _water_mol()
    basis = basis_from_pyscf_spec(
        mol.atom,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=1,
    )
    s = overlap_matrix(basis)
    h = build_hcore(basis)

    mf = dft.RKS(mol)
    mf.xc = "pbe0"
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    assert mf.converged

    coords = np.asarray(mf.grids.coords, dtype=float)
    weights = np.asarray(mf.grids.weights, dtype=float)
    ao_deriv1 = evaluate_cartesian_ao(basis, coords, deriv=1)
    ao = ao_deriv1[0]

    cfg = RKSConfig(
        xc_spec="pbe0",
        max_cycle=80,
        conv_tol=1e-9,
        conv_tol_density=1e-6,
        damping=0.15,
        potential_clip=20.0,
        jk_backend="direct",
        direct_scf_tol=0.0,
    )
    out = run_rks_from_integrals(
        overlap=s,
        hcore=h,
        eri=None,
        direct_basis=basis,
        nelectron=mol.nelectron,
        nuclear_repulsion=float(mol.energy_nuc()),
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid_weights=weights,
        config=cfg,
    )

    assert out.converged
    assert np.isclose(out.total_energy, mf.e_tot, atol=2e-4, rtol=2e-6)


def test_packed_no_df_mo_slices_match_dense_eri_for_water():
    _pyscf_or_skip()
    mol = _water_mol()
    basis = basis_from_pyscf_spec(
        mol.atom,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=3,
    )
    eri = np.asarray(eri_tensor(basis), dtype=float)
    pair = np.asarray(eri_pair_matrix_packed(basis), dtype=float)
    rng = np.random.default_rng(6)
    q, _ = np.linalg.qr(rng.normal(size=(mol.nao_nr(), mol.nao_nr())))
    nocc = 3

    dense_slices = _restricted_response_eri_slices_from_mo_tensor(
        eri,
        q,
        nocc,
        include_oovv=True,
    )
    packed_slices = eri_pair_matrix_to_mo_eri_slices(
        pair,
        q,
        nocc=nocc,
        include_oovv=True,
    )

    for packed, dense in zip(packed_slices, dense_slices, strict=True):
        assert packed is not None
        assert dense is not None
        assert np.allclose(np.asarray(packed), np.asarray(dense), atol=2e-5, rtol=2e-5)


def test_rks_df_backend_matches_pyscf_water_total_energy():
    _pyscf_or_skip()
    from pyscf import dft

    mol = _water_mol()
    basis = basis_from_pyscf_spec(
        mol.atom,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=1,
    )
    s = overlap_matrix(basis)
    h = build_hcore(basis)
    eri = eri_tensor(basis)

    mf = dft.RKS(mol)
    mf.xc = "pbe"
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    assert mf.converged

    coords = np.asarray(mf.grids.coords, dtype=float)
    weights = np.asarray(mf.grids.weights, dtype=float)
    ao_deriv1 = evaluate_cartesian_ao(basis, coords, deriv=1)
    ao = ao_deriv1[0]

    cfg = RKSConfig(
        xc_spec="pbe",
        max_cycle=80,
        conv_tol=1e-9,
        conv_tol_density=1e-6,
        damping=0.15,
        potential_clip=20.0,
        jk_backend="df",
        df_tol=1e-10,
    )
    out = run_rks_from_integrals(
        overlap=s,
        hcore=h,
        eri=eri,
        nelectron=mol.nelectron,
        nuclear_repulsion=float(mol.energy_nuc()),
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid_weights=weights,
        config=cfg,
    )

    assert out.converged
    assert np.isclose(out.total_energy, mf.e_tot, atol=5e-5, rtol=5e-5)


def test_strict_jax_df_reference_for_water_skips_full_eri(monkeypatch):
    _pyscf_or_skip()
    from pyscf import dft
    import td_graddft.scf.inputs as scf_inputs_mod

    mol = _water_mol()
    mf = dft.RKS(mol)
    mf.xc = "pbe"
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    assert mf.converged

    def _fail_eri_tensor(*args, **kwargs):
        raise AssertionError("strict-JAX DF reference path should not call eri_tensor")

    monkeypatch.setattr(scf_inputs_mod, "eri_tensor", _fail_eri_tensor)

    cfg = RKSConfig(
        xc_spec="pbe",
        max_cycle=50,
        conv_tol=1e-9,
        conv_tol_density=1e-7,
        damping=0.15,
        potential_clip=20.0,
        jk_backend="df",
        df_tol=1e-10,
    )
    ref = restricted_molecule_from_spec_with_jax_rks(
        atom=mol.atom,
        basis="sto-3g",
        xc_spec="pbe",
        unit="Angstrom",
        charge=0,
        spin=0,
        cart=True,
        grids_level=0,
        max_l=1,
        rks_config=cfg,
        grid_ao_backend="jax",
    )

    assert ref.df_factors is not None
    assert ref.eri_ovov is None
    assert ref.eri_ovvo is None
    assert ref.eri_oovv is None
    assert np.asarray(ref.rep_tensor).size == 0
    assert np.isclose(ref.mf_energy, mf.e_tot, atol=5e-5, rtol=5e-5)


def test_strict_jax_libcint_df_reference_for_water_skips_full_eri(monkeypatch):
    _pyscf_or_skip()
    from pyscf import dft, gto

    import td_graddft.scf.builders as reference_mod

    orig_intor = gto.mole.Mole.intor

    def _guarded_intor(self, intor, *args, **kwargs):
        if str(intor) == "int2e_cart":
            raise AssertionError("libcint DF reference path should not call full int2e")
        return orig_intor(self, intor, *args, **kwargs)

    monkeypatch.setattr(gto.mole.Mole, "intor", _guarded_intor)

    mol = _water_mol()
    mf = dft.RKS(mol).density_fit()
    mf.xc = "pbe"
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    assert mf.converged

    ref = reference_mod.restricted_molecule_from_spec_with_jax_rks(
        atom=mol.atom,
        basis="sto-3g",
        xc_spec="pbe",
        unit="Angstrom",
        charge=0,
        spin=0,
        cart=True,
        grids_level=0,
        max_l=1,
        rks_config=RKSConfig(
            xc_spec="pbe",
            max_cycle=50,
            conv_tol=1e-9,
            conv_tol_density=1e-7,
            damping=0.15,
            potential_clip=20.0,
            jk_backend="df",
            df_tol=1e-10,
        ),
        grid_ao_backend="jax",
        integral_backend="cpu",
    )

    assert ref.df_factors is not None
    assert ref.eri_ovov is None
    assert ref.eri_ovvo is None
    assert np.asarray(ref.rep_tensor).size == 0
    assert np.isclose(ref.mf_energy, mf.e_tot, atol=2e-4, rtol=2e-4)


def test_libcint_df_reference_preserves_df_backend_when_xc_overrides_config(monkeypatch):
    _pyscf_or_skip()
    from pyscf import gto

    import td_graddft.scf.builders as reference_mod

    orig_intor = gto.mole.Mole.intor

    def _guarded_intor(self, intor, *args, **kwargs):
        if str(intor) == "int2e_cart":
            raise AssertionError("xc override must not reset jk_backend='df'")
        return orig_intor(self, intor, *args, **kwargs)

    monkeypatch.setattr(gto.mole.Mole, "intor", _guarded_intor)

    ref = reference_mod.restricted_molecule_from_spec_with_jax_rks(
        atom=_water_mol().atom,
        basis="sto-3g",
        xc_spec="pbe0",
        unit="Angstrom",
        charge=0,
        spin=0,
        cart=True,
        grids_level=0,
        max_l=1,
        rks_config=RKSConfig(
            xc_spec="pbe",
            max_cycle=50,
            conv_tol=1e-9,
            conv_tol_density=1e-7,
            damping=0.15,
            potential_clip=20.0,
            jk_backend="df",
            df_tol=1e-10,
        ),
        grid_ao_backend="jax",
        integral_backend="cpu",
    )

    assert ref.df_factors is not None
    assert np.asarray(ref.rep_tensor).size == 0


def test_df_reference_lazy_slices_support_tda():
    _pyscf_or_skip()
    ref = restricted_molecule_from_spec_with_jax_rks(
        atom=_water_mol().atom,
        basis="sto-3g",
        xc_spec="pbe0",
        unit="Angstrom",
        charge=0,
        spin=0,
        cart=True,
        grids_level=0,
        max_l=1,
        rks_config=RKSConfig(
            xc_spec="pbe0",
            max_cycle=50,
            conv_tol=1e-9,
            conv_tol_density=1e-7,
            damping=0.15,
            potential_clip=20.0,
            jk_backend="df",
            df_tol=1e-10,
        ),
        grid_ao_backend="jax",
        integral_backend="cpu",
    )

    assert ref.df_factors is not None
    assert ref.eri_ovov is None
    solver = RestrictedCasidaTDDFT(
        molecule=ref,
        xc_functional=SemilocalResponseFunctional("pbe0"),
    )
    result = solver.tda(nstates=1)
    assert np.asarray(result.excitation_energies).shape == (1,)
    assert np.isfinite(np.asarray(result.excitation_energies)).all()
