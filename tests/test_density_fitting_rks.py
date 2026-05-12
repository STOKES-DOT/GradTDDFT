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
from td_graddft.scf.builders import restricted_reference_from_spec_with_jax_rks
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


def test_rks_rejects_removed_python_iteration_backend():
    cfg = RKSConfig(xc_spec="hf", iteration_backend="python")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="supports \\{'runtime', 'lax'\\}"):
        run_rks_from_integrals(
            overlap=np.eye(1),
            hcore=np.zeros((1, 1)),
            eri=np.zeros((1, 1, 1, 1)),
            nelectron=2,
            nuclear_repulsion=0.0,
            ao=np.ones((1, 1)),
            ao_deriv1=np.zeros((4, 1, 1)),
            grid_weights=np.ones(1),
            config=cfg,
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


def test_rks_direct_cuda_engine_uses_cuda_direct_jk_builder(monkeypatch):
    from td_graddft.scf import rks as rks_mod

    monkeypatch.setenv("TD_GRADDFT_CUDA_FULL_ERI_MAX_MIB", "0")
    monkeypatch.setenv("TD_GRADDFT_CUDA_PAIR_ERI_MAX_MIB", "0")
    basis = basis_from_pyscf_spec(
        "H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=1,
    )
    density = np.asarray(
        [
            [0.83, 0.21],
            [0.21, 0.71],
        ],
        dtype=np.float64,
    )
    captured = {}

    class FakeCudaDirectJKBuilder:
        def __init__(self, direct_basis):
            captured["basis"] = direct_basis

        def build_jk(self, density_arg, **kwargs):
            captured["kwargs"] = kwargs
            captured["density"] = density_arg
            return np.ones_like(np.asarray(density_arg)), 2.0 * np.ones_like(np.asarray(density_arg))

    monkeypatch.setattr(rks_mod, "CudaDirectJKBuilder", FakeCudaDirectJKBuilder)
    monkeypatch.setattr(rks_mod, "cuda_ffi_available", lambda: True)

    builder = rks_mod._make_jk_builder(
        None,
        RKSConfig(jk_backend="direct", direct_jk_engine="cuda"),
        direct_basis=basis,
        with_k=True,
    )
    j_mat, k_mat = builder(density)

    assert captured["basis"] is basis
    assert np.allclose(np.asarray(captured["density"]), density)
    assert np.allclose(np.asarray(j_mat), 1.0)
    assert np.allclose(np.asarray(k_mat), 2.0)


def test_rks_direct_cuda_jit_helper_was_removed():
    from td_graddft.scf import rks as rks_mod

    assert not hasattr(rks_mod, "_jit_if_real_cuda_builder")


def test_rks_direct_cuda_engine_respects_zero_pair_cache_limit_for_direct_digest(monkeypatch):
    from td_graddft.scf import rks as rks_mod

    basis = basis_from_pyscf_spec(
        "H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=1,
    )
    density = np.asarray(
        [
            [0.83, 0.21],
            [0.21, 0.71],
        ],
        dtype=np.float64,
    )
    captured = {}

    class FakeCudaDirectJKBuilder:
        def __init__(self, direct_basis):
            captured["basis"] = direct_basis

        def build_eri_tensor(self):
            raise AssertionError("CUDA direct SCF default must not construct full ERI.")

        def build_eri_pair_matrix(self):
            raise AssertionError("CUDA direct SCF default must not construct pair-ERI matrix.")

        def build_jk_from_eri_pair_matrix(self, pair_arg, density_arg):
            raise AssertionError("CUDA direct SCF default must not use pair-ERI contraction.")

        def build_jk(self, density_arg, **kwargs):
            captured["density"] = density_arg
            captured["kwargs"] = kwargs
            return np.ones_like(np.asarray(density_arg)), 2.0 * np.ones_like(np.asarray(density_arg))

    monkeypatch.setenv("TD_GRADDFT_CUDA_FULL_ERI_MAX_MIB", "0")
    monkeypatch.setenv("TD_GRADDFT_CUDA_PAIR_ERI_MAX_MIB", "0")
    monkeypatch.setattr(rks_mod, "CudaDirectJKBuilder", FakeCudaDirectJKBuilder)
    monkeypatch.setattr(rks_mod, "cuda_ffi_available", lambda: True)

    builder = rks_mod._make_jk_builder(
        None,
        RKSConfig(jk_backend="direct", direct_jk_engine="cuda"),
        direct_basis=basis,
        with_k=True,
    )
    j_mat, k_mat = builder(density)

    assert captured["basis"] is basis
    assert np.allclose(np.asarray(captured["density"]), density)
    assert captured["kwargs"]["density_cutoff"] == 0.0
    assert np.allclose(np.asarray(j_mat), 1.0)
    assert np.allclose(np.asarray(k_mat), 2.0)


def test_libcint_inputs_do_not_prebuild_pair_eri_for_static_direct_cuda(monkeypatch):
    import td_graddft.scf.inputs as inputs_mod

    basis = basis_from_pyscf_spec(
        "H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=1,
    )

    monkeypatch.setenv("TD_GRADDFT_CUDA_PAIR_ERI_MAX_MIB", "1")

    assert not inputs_mod._libcint_pair_eri_for_direct_cuda(
        RKSConfig(jk_backend="direct", direct_jk_engine="cuda", direct_scf_tol=0.0),
        basis,
        geometry_is_traced=False,
    )


def test_libcint_inputs_do_not_use_constant_pair_eri_for_traced_or_screened_cuda(monkeypatch):
    import td_graddft.scf.inputs as inputs_mod

    basis = basis_from_pyscf_spec(
        "H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=1,
    )

    monkeypatch.setenv("TD_GRADDFT_CUDA_PAIR_ERI_MAX_MIB", "1")

    assert not inputs_mod._libcint_pair_eri_for_direct_cuda(
        RKSConfig(jk_backend="direct", direct_jk_engine="cuda", direct_scf_tol=0.0),
        basis,
        geometry_is_traced=True,
    )
    assert not inputs_mod._libcint_pair_eri_for_direct_cuda(
        RKSConfig(jk_backend="direct", direct_jk_engine="cuda", direct_scf_tol=1e-8),
        basis,
        geometry_is_traced=False,
    )


def test_rks_direct_cuda_engine_uses_direct_digest_by_default_when_pair_limit_allows(monkeypatch):
    from td_graddft.scf import rks as rks_mod

    basis = basis_from_pyscf_spec(
        "H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=1,
    )
    density = np.asarray(
        [
            [0.83, 0.21],
            [0.21, 0.71],
        ],
        dtype=np.float64,
    )
    captured = {}
    class FakeCudaDirectJKBuilder:
        def __init__(self, direct_basis):
            captured["basis"] = direct_basis

        def build_eri_tensor(self):
            raise AssertionError("CUDA direct SCF default must not construct full ERI.")

        def build_eri_pair_matrix(self):
            raise AssertionError("CUDA direct SCF default must not construct pair-ERI matrix.")

        def build_jk_from_eri_pair_matrix(self, pair_arg, density_arg):
            raise AssertionError("CUDA direct SCF default must not use pair-ERI contraction.")

        def build_jk(self, density_arg, **kwargs):
            captured["density"] = density_arg
            captured["kwargs"] = kwargs
            return np.ones_like(np.asarray(density_arg)), 2.0 * np.ones_like(np.asarray(density_arg))

    monkeypatch.setattr(rks_mod, "CudaDirectJKBuilder", FakeCudaDirectJKBuilder)
    monkeypatch.setattr(rks_mod, "cuda_ffi_available", lambda: True)

    builder = rks_mod._make_jk_builder(
        None,
        RKSConfig(jk_backend="direct", direct_jk_engine="cuda"),
        direct_basis=basis,
        with_k=True,
    )
    j_mat, k_mat = builder(density)

    assert captured["basis"] is basis
    assert np.allclose(np.asarray(captured["density"]), density)
    assert captured["kwargs"]["density_cutoff"] == 0.0
    assert np.allclose(np.asarray(j_mat), 1.0)
    assert np.allclose(np.asarray(k_mat), 2.0)


def test_rks_direct_cuda_engine_reuses_supplied_pair_eri_cache(monkeypatch):
    from td_graddft.scf import rks as rks_mod

    basis = basis_from_pyscf_spec(
        "H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=1,
    )
    density = np.asarray(
        [
            [0.83, 0.21],
            [0.21, 0.71],
        ],
        dtype=np.float64,
    )
    captured = {}
    supplied_pair = np.arange(
        (basis.nao * (basis.nao + 1) // 2) ** 2,
        dtype=np.float64,
    ).reshape((basis.nao * (basis.nao + 1) // 2,) * 2)

    class FakeCudaDirectJKBuilder:
        def __init__(self, direct_basis):
            captured["basis"] = direct_basis

        def build_eri_tensor(self):
            raise AssertionError("Supplied pair ERI should avoid full ERI construction.")

        def build_eri_pair_matrix(self):
            raise AssertionError("Supplied pair ERI should avoid CUDA pair ERI construction.")

        def build_jk_from_eri_pair_matrix(self, pair_arg, density_arg):
            captured["pair"] = pair_arg
            captured["density"] = density_arg
            return np.ones_like(np.asarray(density_arg)), 2.0 * np.ones_like(np.asarray(density_arg))

        def build_jk(self, density_arg, **kwargs):
            raise AssertionError("Supplied pair ERI should use packed CUDA contraction.")

    monkeypatch.setenv("TD_GRADDFT_CUDA_FULL_ERI_MAX_MIB", "1")
    monkeypatch.setenv("TD_GRADDFT_CUDA_PAIR_ERI_MAX_MIB", "1")
    monkeypatch.setattr(rks_mod, "CudaDirectJKBuilder", FakeCudaDirectJKBuilder)
    monkeypatch.setattr(rks_mod, "cuda_ffi_available", lambda: True)

    builder = rks_mod._make_jk_builder(
        None,
        RKSConfig(jk_backend="direct", direct_jk_engine="cuda"),
        eri_pair_matrix=supplied_pair,
        direct_basis=basis,
        with_k=True,
    )
    j_mat, k_mat = builder(density)

    assert captured["basis"] is basis
    assert captured["pair"] is supplied_pair
    assert np.allclose(np.asarray(captured["density"]), density)
    assert np.allclose(np.asarray(j_mat), 1.0)
    assert np.allclose(np.asarray(k_mat), 2.0)


def test_rks_direct_cuda_supplied_pair_eri_uses_incremental_update(monkeypatch):
    from td_graddft.scf import rks as rks_mod

    basis = basis_from_pyscf_spec(
        "H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=1,
    )
    density_last = np.asarray([[0.8, 0.2], [0.2, 0.7]], dtype=np.float64)
    density = density_last + np.asarray([[0.01, -0.02], [-0.02, 0.03]], dtype=np.float64)
    j_last = np.full_like(density, 3.0)
    k_last = np.full_like(density, 4.0)
    captured = {}
    supplied_pair = np.arange(
        (basis.nao * (basis.nao + 1) // 2) ** 2,
        dtype=np.float64,
    ).reshape((basis.nao * (basis.nao + 1) // 2,) * 2)

    class FakeCudaDirectJKBuilder:
        def __init__(self, direct_basis):
            captured["basis"] = direct_basis

        def build_jk_from_eri_pair_matrix(self, pair_arg, density_arg):
            captured["pair"] = pair_arg
            captured["density"] = density_arg
            return np.ones_like(np.asarray(density_arg)), 2.0 * np.ones_like(np.asarray(density_arg))

    monkeypatch.setattr(rks_mod, "CudaDirectJKBuilder", FakeCudaDirectJKBuilder)
    monkeypatch.setattr(rks_mod, "cuda_ffi_available", lambda: True)

    builder = rks_mod._make_jk_builder(
        None,
        RKSConfig(
            jk_backend="direct",
            direct_jk_engine="cuda",
            direct_scf_incremental=True,
        ),
        eri_pair_matrix=supplied_pair,
        direct_basis=basis,
        with_k=True,
    )
    j_mat, k_mat = builder(
        density,
        density_last=density_last,
        j_last=j_last,
        k_last=k_last,
    )

    assert captured["basis"] is basis
    assert captured["pair"] is supplied_pair
    assert np.allclose(np.asarray(captured["density"]), density - density_last)
    assert np.allclose(np.asarray(j_mat), j_last + 1.0)
    assert np.allclose(np.asarray(k_mat), k_last + 2.0)


def test_rks_lax_direct_cuda_defers_initial_fock_build(monkeypatch):
    import td_graddft.scf.rks as rks_mod

    basis = basis_from_pyscf_spec(
        "H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=1,
    )
    s = overlap_matrix(basis)
    h = build_hcore(basis)
    ao = np.zeros((1, basis.nao), dtype=np.float64)
    ao_deriv1 = np.zeros((4, 1, basis.nao), dtype=np.float64)
    weights = np.zeros((1,), dtype=np.float64)

    def _fail_initial_fock(**kwargs):
        raise AssertionError("Direct CUDA lax SCF should enter the while-loop without a separate initial Fock build.")

    monkeypatch.setattr(rks_mod, "_raw_fock_for_density", _fail_initial_fock)

    out = run_rks_from_integrals(
        overlap=s,
        hcore=h,
        eri=None,
        direct_basis=basis,
        nelectron=2,
        nuclear_repulsion=0.0,
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid_weights=weights,
        config=RKSConfig(
            xc_spec="hf",
            max_cycle=1,
            jk_backend="direct",
            direct_jk_engine="cuda",
            iteration_backend="lax",
        ),
    )

    assert out.cycles == 1


def test_rks_lax_direct_cuda_uses_host_initial_orbitals(monkeypatch):
    import td_graddft.scf.rks as rks_mod

    basis = basis_from_pyscf_spec(
        "H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=1,
    )
    ao = np.zeros((1, basis.nao), dtype=np.float64)
    ao_deriv1 = np.zeros((4, 1, basis.nao), dtype=np.float64)
    weights = np.zeros((1,), dtype=np.float64)

    def _fail_device_orthogonalizer(*args, **kwargs):
        raise AssertionError("Direct CUDA lax SCF should avoid separate XLA initial-orbital kernels.")

    monkeypatch.setattr(rks_mod, "_orthogonalizer", _fail_device_orthogonalizer)

    out = run_rks_from_integrals(
        overlap=overlap_matrix(basis),
        hcore=build_hcore(basis),
        eri=None,
        direct_basis=basis,
        nelectron=2,
        nuclear_repulsion=0.0,
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid_weights=weights,
        config=RKSConfig(
            xc_spec="hf",
            max_cycle=1,
            jk_backend="direct",
            direct_jk_engine="cuda",
            iteration_backend="lax",
        ),
    )

    assert out.cycles == 1


def test_rks_direct_cuda_engine_ignores_full_eri_cache_limit(monkeypatch):
    from td_graddft.scf import rks as rks_mod

    basis = basis_from_pyscf_spec(
        "H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=1,
    )
    density = np.asarray(
        [
            [0.83, 0.21],
            [0.21, 0.71],
        ],
        dtype=np.float64,
    )
    captured = {}

    class FakeCudaDirectJKBuilder:
        def __init__(self, direct_basis):
            captured["basis"] = direct_basis

        def build_eri_tensor(self):
            raise AssertionError("CUDA direct SCF default must not construct full ERI.")

        def build_eri_pair_matrix(self):
            raise AssertionError("CUDA direct SCF default must not construct pair-ERI matrix.")

        def build_jk_from_eri_pair_matrix(self, pair_arg, density_arg):
            raise AssertionError("CUDA direct SCF default must not use pair-ERI contraction.")

        def build_jk(self, density_arg, **kwargs):
            captured["density"] = density_arg
            captured["kwargs"] = kwargs
            return np.ones_like(np.asarray(density_arg)), 2.0 * np.ones_like(np.asarray(density_arg))

    monkeypatch.setenv("TD_GRADDFT_CUDA_FULL_ERI_MAX_MIB", "1")
    monkeypatch.setenv("TD_GRADDFT_CUDA_PAIR_ERI_MAX_MIB", "0")
    monkeypatch.setattr(rks_mod, "CudaDirectJKBuilder", FakeCudaDirectJKBuilder)
    monkeypatch.setattr(rks_mod, "cuda_ffi_available", lambda: True)

    builder = rks_mod._make_jk_builder(
        None,
        RKSConfig(jk_backend="direct", direct_jk_engine="cuda"),
        direct_basis=basis,
        with_k=True,
    )
    j_mat, k_mat = builder(density)

    assert captured["basis"] is basis
    assert np.allclose(np.asarray(captured["density"]), density)
    assert captured["kwargs"]["density_cutoff"] == 0.0
    assert np.allclose(np.asarray(j_mat), 1.0)
    assert np.allclose(np.asarray(k_mat), 2.0)


def test_rks_direct_cuda_engine_ignores_pair_cache_limit_for_default_digest(monkeypatch):
    from td_graddft.scf import rks as rks_mod

    basis = basis_from_pyscf_spec(
        "H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=1,
    )
    density = np.asarray(
        [
            [0.83, 0.21],
            [0.21, 0.71],
        ],
        dtype=np.float64,
    )
    captured = {}
    class FakeCudaDirectJKBuilder:
        def __init__(self, direct_basis):
            captured["basis"] = direct_basis

        def build_eri_tensor(self):
            raise AssertionError("CUDA direct SCF default must not construct full ERI.")

        def build_eri_pair_matrix(self):
            raise AssertionError("CUDA direct SCF default must not construct pair-ERI matrix.")

        def build_jk_from_eri_pair_matrix(self, pair_arg, density_arg):
            raise AssertionError("CUDA direct SCF default must not use pair-ERI contraction.")

        def build_jk(self, density_arg, **kwargs):
            captured["density"] = density_arg
            captured["kwargs"] = kwargs
            return np.ones_like(np.asarray(density_arg)), 2.0 * np.ones_like(np.asarray(density_arg))

    monkeypatch.setenv("TD_GRADDFT_CUDA_FULL_ERI_MAX_MIB", "0")
    monkeypatch.setenv("TD_GRADDFT_CUDA_PAIR_ERI_MAX_MIB", "1")
    monkeypatch.setattr(rks_mod, "CudaDirectJKBuilder", FakeCudaDirectJKBuilder)
    monkeypatch.setattr(rks_mod, "cuda_ffi_available", lambda: True)

    builder = rks_mod._make_jk_builder(
        None,
        RKSConfig(jk_backend="direct", direct_jk_engine="cuda"),
        direct_basis=basis,
        with_k=True,
    )
    j_mat, k_mat = builder(density)

    assert captured["basis"] is basis
    assert np.allclose(np.asarray(captured["density"]), density)
    assert captured["kwargs"]["density_cutoff"] == 0.0
    assert np.allclose(np.asarray(j_mat), 1.0)
    assert np.allclose(np.asarray(k_mat), 2.0)


def test_direct_cuda_jk_builder_uses_incremental_under_lax_unscreened(monkeypatch):
    import td_graddft.scf.rks as rks_mod

    monkeypatch.setenv("TD_GRADDFT_CUDA_FULL_ERI_MAX_MIB", "0")
    monkeypatch.setenv("TD_GRADDFT_CUDA_PAIR_ERI_MAX_MIB", "0")
    basis = basis_from_pyscf_spec(
        "H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=1,
    )
    density_last = np.asarray([[0.8, 0.2], [0.2, 0.7]], dtype=np.float64)
    density = density_last + np.asarray([[0.01, -0.02], [-0.02, 0.03]], dtype=np.float64)
    j_last = np.full_like(density, 3.0)
    k_last = np.full_like(density, 4.0)
    captured = {}

    class FakeCudaDirectJKBuilder:
        def __init__(self, direct_basis):
            captured["basis"] = direct_basis

        def build_jk(self, density_arg, **kwargs):
            captured["density"] = density_arg
            captured["kwargs"] = kwargs
            return np.ones_like(np.asarray(density_arg)), 2.0 * np.ones_like(np.asarray(density_arg))

    monkeypatch.setattr(rks_mod, "CudaDirectJKBuilder", FakeCudaDirectJKBuilder)
    monkeypatch.setattr(rks_mod, "cuda_ffi_available", lambda: True)

    builder = rks_mod._make_jk_builder(
        None,
        RKSConfig(
            jk_backend="direct",
            direct_jk_engine="cuda",
            direct_scf_tol=0.0,
            direct_scf_incremental=True,
            iteration_backend="lax",
        ),
        direct_basis=basis,
        with_k=True,
    )
    j_mat, k_mat = builder(
        density,
        density_last=density_last,
        j_last=j_last,
        k_last=k_last,
    )

    assert captured["basis"] is basis
    assert np.allclose(np.asarray(captured["density"]), density - density_last)
    assert captured["kwargs"]["density_cutoff"] == 0.0
    assert np.allclose(np.asarray(j_mat), j_last + 1.0)
    assert np.allclose(np.asarray(k_mat), k_last + 2.0)


def test_direct_cuda_jk_builder_uses_incremental_when_screened(monkeypatch):
    import td_graddft.scf.rks as rks_mod

    basis = basis_from_pyscf_spec(
        "H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=1,
    )
    density_last = np.asarray([[0.8, 0.2], [0.2, 0.7]], dtype=np.float64)
    density = density_last + np.asarray([[0.01, -0.02], [-0.02, 0.03]], dtype=np.float64)
    j_last = np.full_like(density, 3.0)
    k_last = np.full_like(density, 4.0)
    captured = {}

    class FakeCudaDirectJKBuilder:
        def __init__(self, direct_basis):
            captured["basis"] = direct_basis

        def build_jk(self, density_arg, **kwargs):
            captured["density"] = density_arg
            captured["kwargs"] = kwargs
            return np.ones_like(np.asarray(density_arg)), 2.0 * np.ones_like(np.asarray(density_arg))

    monkeypatch.setattr(rks_mod, "CudaDirectJKBuilder", FakeCudaDirectJKBuilder)
    monkeypatch.setattr(rks_mod, "cuda_ffi_available", lambda: True)

    builder = rks_mod._make_jk_builder(
        None,
        RKSConfig(
            jk_backend="direct",
            direct_jk_engine="cuda",
            direct_scf_tol=1e-8,
            direct_scf_incremental=True,
        ),
        direct_basis=basis,
        with_k=True,
    )
    j_mat, k_mat = builder(
        density,
        density_last=density_last,
        j_last=j_last,
        k_last=k_last,
    )

    assert captured["basis"] is basis
    assert np.allclose(np.asarray(captured["density"]), density - density_last)
    assert captured["kwargs"]["density_cutoff"] == 1e-8
    assert np.allclose(np.asarray(j_mat), j_last + 1.0)
    assert np.allclose(np.asarray(k_mat), k_last + 2.0)


def test_direct_cuda_jk_builder_uses_incremental_under_lax(monkeypatch):
    import td_graddft.scf.rks as rks_mod

    monkeypatch.setenv("TD_GRADDFT_CUDA_PAIR_ERI_MAX_MIB", "0")
    basis = basis_from_pyscf_spec(
        "H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=1,
    )
    density_last = np.asarray([[0.8, 0.2], [0.2, 0.7]], dtype=np.float64)
    density = density_last + np.asarray([[0.01, -0.02], [-0.02, 0.03]], dtype=np.float64)
    j_last = np.full_like(density, 3.0)
    k_last = np.full_like(density, 4.0)
    captured = {}

    class FakeCudaDirectJKBuilder:
        def __init__(self, direct_basis):
            captured["basis"] = direct_basis

        def build_jk(self, density_arg, **kwargs):
            captured["density"] = density_arg
            captured["kwargs"] = kwargs
            return np.ones_like(np.asarray(density_arg)), 2.0 * np.ones_like(np.asarray(density_arg))

    monkeypatch.setattr(rks_mod, "CudaDirectJKBuilder", FakeCudaDirectJKBuilder)
    monkeypatch.setattr(rks_mod, "cuda_ffi_available", lambda: True)

    builder = rks_mod._make_jk_builder(
        None,
        RKSConfig(
            jk_backend="direct",
            direct_jk_engine="cuda",
            direct_scf_tol=0.0,
            direct_scf_incremental=True,
            iteration_backend="lax",
        ),
        direct_basis=basis,
        with_k=True,
    )
    j_mat, k_mat = builder(
        density,
        density_last=density_last,
        j_last=j_last,
        k_last=k_last,
    )

    assert captured["basis"] is basis
    assert np.allclose(np.asarray(captured["density"]), density - density_last)
    assert captured["kwargs"]["density_cutoff"] == 0.0
    assert np.allclose(np.asarray(j_mat), j_last + 1.0)
    assert np.allclose(np.asarray(k_mat), k_last + 2.0)


def test_direct_cuda_jk_builder_precomputes_screening_metadata_for_lax_screened(monkeypatch):
    import td_graddft.scf.rks as rks_mod

    basis = basis_from_pyscf_spec(
        "H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=1,
    )
    captured = {"precompute_calls": 0}

    class FakeCudaDirectJKBuilder:
        def __init__(self, direct_basis):
            captured["basis"] = direct_basis

        def precompute_screening_metadata(self):
            captured["precompute_calls"] += 1

        def build_jk(self, density_arg, **kwargs):
            return np.ones_like(np.asarray(density_arg)), 2.0 * np.ones_like(np.asarray(density_arg))

    monkeypatch.setattr(rks_mod, "CudaDirectJKBuilder", FakeCudaDirectJKBuilder)
    monkeypatch.setattr(rks_mod, "cuda_ffi_available", lambda: True)

    rks_mod._make_jk_builder(
        None,
        RKSConfig(
            jk_backend="direct",
            direct_jk_engine="cuda",
            direct_scf_tol=1e-8,
            iteration_backend="lax",
        ),
        direct_basis=basis,
        with_k=True,
    )

    assert captured["basis"] is basis
    assert captured["precompute_calls"] == 1


def test_direct_cuda_jk_builder_applies_screening_to_initial_density(monkeypatch):
    import td_graddft.scf.rks as rks_mod

    monkeypatch.setenv("TD_GRADDFT_CUDA_FULL_ERI_MAX_MIB", "0")
    basis = basis_from_pyscf_spec(
        "H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=1,
    )
    density = np.asarray([[0.8, 0.2], [0.2, 0.7]], dtype=np.float64)
    captured = {}

    class FakeCudaDirectJKBuilder:
        def __init__(self, direct_basis):
            captured["basis"] = direct_basis

        def build_jk(self, density_arg, **kwargs):
            captured["density"] = density_arg
            captured["kwargs"] = kwargs
            return np.ones_like(np.asarray(density_arg)), 2.0 * np.ones_like(np.asarray(density_arg))

    monkeypatch.setattr(rks_mod, "CudaDirectJKBuilder", FakeCudaDirectJKBuilder)
    monkeypatch.setattr(rks_mod, "cuda_ffi_available", lambda: True)

    builder = rks_mod._make_jk_builder(
        None,
        RKSConfig(jk_backend="direct", direct_jk_engine="cuda", direct_scf_tol=1e-8),
        direct_basis=basis,
        with_k=True,
    )
    j_mat, k_mat = builder(density)

    assert captured["basis"] is basis
    assert np.allclose(np.asarray(captured["density"]), density)
    assert captured["kwargs"]["density_cutoff"] == 1e-8
    assert np.allclose(np.asarray(j_mat), 1.0)
    assert np.allclose(np.asarray(k_mat), 2.0)


def test_direct_cuda_jk_builder_falls_back_to_jax_when_cuda_unavailable(monkeypatch):
    import td_graddft.scf.rks as rks_mod

    basis = basis_from_pyscf_spec(
        "H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=1,
    )
    density = np.asarray(
        [
            [0.83, 0.21],
            [0.21, 0.71],
        ],
        dtype=np.float64,
    )

    class FailingCudaDirectJKBuilder:
        def __init__(self, direct_basis):
            raise AssertionError("CUDA direct J/K builder should not be used without CUDA.")

    monkeypatch.setattr(rks_mod, "cuda_ffi_available", lambda: False)
    monkeypatch.setattr(rks_mod, "CudaDirectJKBuilder", FailingCudaDirectJKBuilder)

    builder = rks_mod._make_jk_builder(
        None,
        RKSConfig(jk_backend="direct", direct_jk_engine="cuda"),
        direct_basis=basis,
        with_k=True,
    )
    j_mat, k_mat = builder(density)

    assert np.asarray(j_mat).shape == density.shape
    assert np.asarray(k_mat).shape == density.shape


def test_rks_lax_iteration_backend_matches_pyscf_for_water():
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
        iteration_backend="lax",
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
    ref = restricted_reference_from_spec_with_jax_rks(
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

    ref = reference_mod.restricted_reference_from_spec_with_jax_rks(
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
        integral_backend="libcint",
    )

    assert ref.df_factors is not None
    assert ref.eri_ovov is None
    assert ref.eri_ovvo is None
    assert np.asarray(ref.rep_tensor).size == 0
    assert np.isclose(ref.mf_energy, mf.e_tot, atol=2e-4, rtol=2e-4)


def test_libcint_direct_cuda_inputs_do_not_prebuild_cpu_pair_eri(monkeypatch):
    _pyscf_or_skip()
    from pyscf import gto

    import td_graddft.scf.inputs as inputs_mod

    orig_intor = gto.mole.Mole.intor

    def _guarded_intor(self, intor, *args, **kwargs):
        if str(intor).startswith("int2e"):
            raise AssertionError("CUDA direct SCF should build AO-pair ERI in CUDA, not via CPU libcint.")
        return orig_intor(self, intor, *args, **kwargs)

    monkeypatch.setattr(gto.mole.Mole, "intor", _guarded_intor)
    monkeypatch.setattr(inputs_mod, "cuda_ffi_available", lambda: True)

    inputs = inputs_mod.build_rks_integral_inputs(
        atom=_water_mol().atom,
        basis="sto-3g",
        xc_spec="pbe0",
        unit="Angstrom",
        charge=0,
        spin=0,
        cart=True,
        grids_level=0,
        max_l=1,
        config=RKSConfig(
            xc_spec="pbe0",
            jk_backend="direct",
            direct_jk_engine="cuda",
        ),
        grid_ao_backend="jax",
        integral_backend="libcint",
        include_dipole_integrals=False,
    )

    assert inputs.eri_pair_matrix is None
    assert inputs.eri is None
    assert inputs.direct_basis is not None


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

    ref = reference_mod.restricted_reference_from_spec_with_jax_rks(
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
        integral_backend="libcint",
    )

    assert ref.df_factors is not None
    assert np.asarray(ref.rep_tensor).size == 0


def test_df_reference_lazy_slices_support_tda():
    _pyscf_or_skip()
    ref = restricted_reference_from_spec_with_jax_rks(
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
        integral_backend="libcint",
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
