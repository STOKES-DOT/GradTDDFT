from __future__ import annotations

import sys
import types

import numpy as np

from td_graddft.data import reference as reference_module
from td_graddft.neural_xc.inputs import ChunkedHFXNu
from td_graddft.scf.molecules import UnrestrictedMolecule


class _CommonOrigin:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeMol:
    spin = 0
    natm = 1

    def intor(self, name, aosym=None):
        assert name == "int2e"
        if aosym == "s4":
            return np.arange(9, dtype=np.float64).reshape(3, 3) / 100.0
        return np.arange(16, dtype=np.float64).reshape(2, 2, 2, 2) / 100.0

    def intor_symmetric(self, name, comp):
        assert name == "int1e_r"
        assert comp == 3
        return np.zeros((3, 2, 2), dtype=np.float64)

    def with_common_orig(self, origin):
        return _CommonOrigin()

    def atom_coords(self):
        return np.zeros((1, 3), dtype=np.float64)

    def atom_charges(self):
        return np.ones(1, dtype=np.float64)

    def energy_nuc(self):
        return 0.0


class _FakeLargeMol(_FakeMol):
    natm = 4

    def atom_coords(self):
        return np.zeros((4, 3), dtype=np.float64)

    def atom_charges(self):
        return np.ones(4, dtype=np.float64)


class _FakeGrids:
    coords = np.zeros((2, 3), dtype=np.float64)
    weights = np.ones(2, dtype=np.float64)

    def build(self):
        return None


class _FakeMF:
    mol = _FakeMol()
    grids = _FakeGrids()
    mo_coeff = np.eye(2, dtype=np.float64)
    mo_occ = np.asarray([2.0, 0.0], dtype=np.float64)
    mo_energy = np.asarray([-0.5, 0.1], dtype=np.float64)
    e_tot = -1.0
    xc = "b3lyp"
    _numint = None

    def make_rdm1(self):
        return np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)

    def get_hcore(self):
        return np.eye(2, dtype=np.float64)

    def get_ovlp(self):
        return np.eye(2, dtype=np.float64)


class _FakeLargeMF(_FakeMF):
    mol = _FakeLargeMol()


class _FakeUKSMol(_FakeMol):
    spin = 1


class _FakeUKSMF:
    mol = _FakeUKSMol()
    grids = _FakeGrids()
    mo_coeff = np.stack([np.eye(2, dtype=np.float64), np.eye(2, dtype=np.float64)], axis=0)
    mo_occ = np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    mo_energy = np.asarray([[-0.5, 0.1], [-0.4, 0.2]], dtype=np.float64)
    e_tot = -0.5
    xc = "b3lyp"
    _numint = None

    def make_rdm1(self):
        return np.asarray(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
            dtype=np.float64,
        )

    def get_hcore(self):
        return np.eye(2, dtype=np.float64)

    def get_ovlp(self):
        return np.eye(2, dtype=np.float64)


def test_chunked_hfx_nu_reads_dense_equivalent_grid_slices():
    dense = np.arange(2 * 5 * 3 * 3, dtype=np.float64).reshape(2, 5, 3, 3)
    api = ChunkedHFXNu.from_dense(dense, chunk_size=2)

    assert api.shape == dense.shape
    assert api.ndim == 4
    assert np.allclose(api.grid_chunk(1, 4), dense[:, 1:4])
    assert np.allclose(api.materialize(), dense)


def test_restricted_reference_host_backend_keeps_hfx_nu_on_host(monkeypatch):
    fake_numint = types.SimpleNamespace(
        eval_ao=lambda mol, coords, deriv=0: (
            np.ones((2, 2), dtype=np.float64)
            if deriv == 0
            else np.ones((4, 2, 2), dtype=np.float64)
        )
    )
    fake_dft = types.ModuleType("pyscf.dft")
    fake_dft.numint = fake_numint
    fake_ao2mo = types.ModuleType("pyscf.ao2mo")

    def fake_ao2mo_general(mol, mo_tuple, compact=False):
        del mol
        assert compact is False
        left = int(mo_tuple[0].shape[1] * mo_tuple[1].shape[1])
        right = int(mo_tuple[2].shape[1] * mo_tuple[3].shape[1])
        return np.zeros((left, right), dtype=np.float64)

    fake_ao2mo.general = fake_ao2mo_general
    fake_pyscf = types.ModuleType("pyscf")
    fake_pyscf.ao2mo = fake_ao2mo
    fake_pyscf.dft = fake_dft
    monkeypatch.setitem(sys.modules, "pyscf", fake_pyscf)
    monkeypatch.setitem(sys.modules, "pyscf.ao2mo", fake_ao2mo)
    monkeypatch.setitem(sys.modules, "pyscf.dft", fake_dft)

    def fake_local_hfx(*args, **kwargs):
        assert kwargs["return_nu"] is True
        assert kwargs["return_fxx"] is True
        return (
            np.zeros((2, 2, 2), dtype=np.float64),
            np.zeros((2, 2, 2, 2), dtype=np.float64),
            np.zeros((2, 2, 2), dtype=np.float64),
        )

    monkeypatch.setattr(reference_module, "_local_hfx_features_from_dm", fake_local_hfx)

    molecule = reference_module.restricted_reference_from_pyscf(
        _FakeMF(),
        compute_local_hfx_features=True,
        compute_local_hfx_aux=True,
        array_backend="host",
    )

    assert isinstance(molecule.ao, np.ndarray)
    assert isinstance(molecule.rep_tensor, np.ndarray)
    assert molecule.rep_tensor.shape == (0, 0, 0, 0)
    assert isinstance(molecule.eri_pair_matrix, np.ndarray)
    assert molecule.eri_pair_matrix.shape == (3, 3)
    assert molecule.eri_ovov is None
    assert isinstance(molecule.hfx_local, np.ndarray)
    assert isinstance(molecule.hfx_nu, np.ndarray)
    assert isinstance(molecule.hfx_fxx, np.ndarray)
    assert molecule.hfx_nu_api is None


def test_restricted_reference_df_backend_stores_df_without_packed_eri(monkeypatch):
    fake_numint = types.SimpleNamespace(
        eval_ao=lambda mol, coords, deriv=0: (
            np.ones((2, 2), dtype=np.float64)
            if deriv == 0
            else np.ones((4, 2, 2), dtype=np.float64)
        )
    )
    fake_dft = types.ModuleType("pyscf.dft")
    fake_dft.numint = fake_numint
    fake_pyscf = types.ModuleType("pyscf")
    fake_pyscf.dft = fake_dft
    monkeypatch.setitem(sys.modules, "pyscf", fake_pyscf)
    monkeypatch.setitem(sys.modules, "pyscf.dft", fake_dft)

    class _NoPackedEriMol(_FakeMol):
        def intor(self, name, aosym=None):
            if name == "int2e":
                raise AssertionError("DF reference should not build packed int2e.")
            return super().intor(name, aosym=aosym)

    class _NoPackedEriMF(_FakeMF):
        mol = _NoPackedEriMol()

    monkeypatch.setattr(
        reference_module,
        "true_df_factors_from_libcint_mol",
        lambda mol: np.ones((2, 2, 2), dtype=np.float64),
    )

    molecule = reference_module.restricted_reference_from_pyscf(
        _NoPackedEriMF(),
        jk_backend="df",
        array_backend="host",
    )

    assert molecule.rep_tensor.shape == (0, 0, 0, 0)
    assert molecule.eri_pair_matrix is None
    assert isinstance(molecule.df_factors, np.ndarray)
    assert molecule.df_factors.shape == (2, 2, 2)


def test_restricted_reference_caches_pt2_fock_response_for_nograd(monkeypatch):
    ao0 = np.asarray([[1.0, 0.25], [0.4, 1.1]], dtype=np.float64)
    ao1 = np.stack([ao0, ao0, ao0, ao0], axis=0)
    fake_numint = types.SimpleNamespace(
        eval_ao=lambda mol, coords, deriv=0: ao0 if deriv == 0 else ao1
    )
    fake_dft = types.ModuleType("pyscf.dft")
    fake_dft.numint = fake_numint
    fake_pyscf = types.ModuleType("pyscf")
    fake_pyscf.dft = fake_dft
    monkeypatch.setitem(sys.modules, "pyscf", fake_pyscf)
    monkeypatch.setitem(sys.modules, "pyscf.dft", fake_dft)

    class _ConsistentClosedShellMF(_FakeMF):
        def make_rdm1(self):
            return np.asarray([[2.0, 0.0], [0.0, 0.0]], dtype=np.float64)

    molecule = reference_module.restricted_reference_from_pyscf(
        _ConsistentClosedShellMF(),
        compute_local_pt2_features=True,
        array_backend="host",
    )

    assert isinstance(molecule.pt2_local, np.ndarray)
    assert isinstance(molecule.pt2_fock_response, np.ndarray)
    assert molecule.pt2_fock_response.shape == (2, 2, 2)
    assert np.all(np.isfinite(molecule.pt2_fock_response))
    assert np.allclose(
        molecule.pt2_fock_response,
        np.swapaxes(molecule.pt2_fock_response, -1, -2),
    )
    reconstructed = np.einsum(
        "pq,gpq->g",
        _ConsistentClosedShellMF().make_rdm1(),
        molecule.pt2_fock_response,
    )
    assert np.allclose(reconstructed, molecule.pt2_local, atol=1e-10)


def test_restricted_reference_uses_chunked_hfx_nu_api_for_more_than_three_atoms(monkeypatch):
    fake_numint = types.SimpleNamespace(
        eval_ao=lambda mol, coords, deriv=0: (
            np.ones((2, 2), dtype=np.float64)
            if deriv == 0
            else np.ones((4, 2, 2), dtype=np.float64)
        )
    )
    fake_dft = types.ModuleType("pyscf.dft")
    fake_dft.numint = fake_numint
    fake_ao2mo = types.ModuleType("pyscf.ao2mo")
    fake_ao2mo.general = lambda mol, mo_tuple, compact=False: np.zeros(
        (
            int(mo_tuple[0].shape[1] * mo_tuple[1].shape[1]),
            int(mo_tuple[2].shape[1] * mo_tuple[3].shape[1]),
        ),
        dtype=np.float64,
    )
    fake_pyscf = types.ModuleType("pyscf")
    fake_pyscf.ao2mo = fake_ao2mo
    fake_pyscf.dft = fake_dft
    monkeypatch.setitem(sys.modules, "pyscf", fake_pyscf)
    monkeypatch.setitem(sys.modules, "pyscf.ao2mo", fake_ao2mo)
    monkeypatch.setitem(sys.modules, "pyscf.dft", fake_dft)

    def fake_local_hfx(*args, **kwargs):
        assert kwargs["return_nu"] is False
        assert kwargs["return_fxx"] is True
        return (
            np.zeros((2, 2, 2), dtype=np.float64),
            np.zeros((2, 2, 2), dtype=np.float64),
        )

    monkeypatch.setattr(reference_module, "_local_hfx_features_from_dm", fake_local_hfx)

    molecule = reference_module.restricted_reference_from_pyscf(
        _FakeLargeMF(),
        compute_local_hfx_features=True,
        compute_local_hfx_aux=True,
        array_backend="host",
    )

    assert molecule.hfx_nu is None
    assert molecule.hfx_nu_api is not None
    assert molecule.hfx_nu_api.shape == (2, 2, 2, 2)
    assert isinstance(molecule.hfx_local, np.ndarray)
    assert isinstance(molecule.hfx_fxx, np.ndarray)


def test_unrestricted_reference_host_backend_keeps_hfx_cache_on_host(monkeypatch):
    fake_numint = types.SimpleNamespace(
        eval_ao=lambda mol, coords, deriv=0: (
            np.ones((2, 2), dtype=np.float64)
            if deriv == 0
            else np.ones((4, 2, 2), dtype=np.float64)
        )
    )
    fake_dft = types.ModuleType("pyscf.dft")
    fake_dft.numint = fake_numint
    fake_pyscf = types.ModuleType("pyscf")
    fake_pyscf.dft = fake_dft
    monkeypatch.setitem(sys.modules, "pyscf", fake_pyscf)
    monkeypatch.setitem(sys.modules, "pyscf.dft", fake_dft)

    def fake_local_hfx(mol, ao, dm_spin, coords, **kwargs):
        del mol, ao, coords
        assert kwargs["return_nu"] is True
        assert kwargs["return_fxx"] is True
        assert dm_spin[0].shape == (2, 2)
        assert dm_spin[1].shape == (2, 2)
        return (
            np.zeros((2, 2, 2), dtype=np.float64),
            np.zeros((2, 2, 2, 2), dtype=np.float64),
            np.zeros((2, 2, 2), dtype=np.float64),
        )

    monkeypatch.setattr(reference_module, "_local_hfx_features_from_dm", fake_local_hfx)

    molecule = reference_module.unrestricted_reference_from_pyscf(
        _FakeUKSMF(),
        compute_local_hfx_features=True,
        compute_local_hfx_aux=True,
        array_backend="host",
    )

    assert isinstance(molecule, UnrestrictedMolecule)
    assert isinstance(molecule.ao, np.ndarray)
    assert isinstance(molecule.rep_tensor, np.ndarray)
    assert molecule.rep_tensor.shape == (3, 3)
    assert molecule.nocc_alpha == 1
    assert molecule.nocc_beta == 0
    assert isinstance(molecule.hfx_local, np.ndarray)
    assert isinstance(molecule.hfx_nu, np.ndarray)
    assert isinstance(molecule.hfx_fxx, np.ndarray)
