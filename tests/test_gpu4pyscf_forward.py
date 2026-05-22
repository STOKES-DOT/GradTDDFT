import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


def test_compute_gpu4pyscf_local_hfx_features_uses_gpu_int1e_grids(monkeypatch):
    import jax.numpy as jnp

    from td_graddft.scf.gpu4pyscf import compute_gpu4pyscf_local_hfx_features

    calls = []

    class FakeRangeContext:
        def __init__(self, mol, omega):
            self.mol = mol
            self.omega = omega
            self.previous = mol.omega

        def __enter__(self):
            self.mol.omega = self.omega
            return self.mol

        def __exit__(self, exc_type, exc, tb):
            self.mol.omega = self.previous
            return False

    class FakeMol:
        nao = 2

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.omega = 0.0

        def with_range_coulomb(self, omega):
            return FakeRangeContext(self, float(omega))

    class FakeVHFOpt:
        def __init__(self, mol):
            self.mol = mol

        def build(self, cutoff, aosym=True):
            calls.append(("build", cutoff, aosym))
            return self

    def fake_int1e_grids(mol, coords, direct_scf_tol, intopt):
        del direct_scf_tol, intopt
        calls.append(("int1e_grids", mol.omega, tuple(map(tuple, np.asarray(coords)))))
        scale = 1.0 + float(mol.omega)
        return np.tile(scale * np.eye(2), (np.asarray(coords).shape[0], 1, 1))

    fake_int3c1e = types.ModuleType("gpu4pyscf.gto.int3c1e")
    fake_int3c1e.VHFOpt = FakeVHFOpt
    fake_int3c1e.int1e_grids = fake_int1e_grids
    fake_gpu4pyscf = types.ModuleType("gpu4pyscf")
    fake_gpu4pyscf_gto = types.ModuleType("gpu4pyscf.gto")
    fake_gto = types.SimpleNamespace(M=lambda **kwargs: FakeMol(**kwargs))
    fake_pyscf = types.ModuleType("pyscf")
    fake_pyscf.gto = fake_gto
    monkeypatch.setitem(sys.modules, "gpu4pyscf", fake_gpu4pyscf)
    monkeypatch.setitem(sys.modules, "gpu4pyscf.gto", fake_gpu4pyscf_gto)
    monkeypatch.setitem(sys.modules, "gpu4pyscf.gto.int3c1e", fake_int3c1e)
    monkeypatch.setitem(sys.modules, "pyscf", fake_pyscf)
    monkeypatch.setitem(sys.modules, "pyscf.gto", fake_gto)

    ao = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    dm = np.diag([0.5, 0.25])
    coords = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])

    hfx, nu = compute_gpu4pyscf_local_hfx_features(
        atom="H 0 0 0",
        basis="sto-3g",
        unit="Angstrom",
        charge=0,
        spin=0,
        cart=True,
        coords=coords,
        ao=ao,
        dm_spin=(dm, dm),
        omega_values=(0.0, 0.4),
        chunk_size=2,
        return_nu=True,
    )

    expected_e = ao @ dm
    expected_omega0 = -0.5 * np.sum(expected_e * expected_e, axis=1)
    expected_omega1 = -0.5 * 1.4 * np.sum(expected_e * expected_e, axis=1)
    assert calls[0] == ("build", 1e-13, True)
    assert [call[0] for call in calls].count("int1e_grids") == 4
    assert np.allclose(np.asarray(hfx[0, :, 0]), expected_omega0)
    assert np.allclose(np.asarray(hfx[0, :, 1]), expected_omega1)
    assert np.allclose(np.asarray(hfx[0]), np.asarray(hfx[1]))
    assert np.allclose(np.asarray(nu[0, 0]), np.eye(2))
    assert np.allclose(np.asarray(nu[1, 0]), 1.4 * np.eye(2))


def test_run_gpu4pyscf_rks_forward_uses_direct_to_gpu_without_density_fit(monkeypatch):
    from td_graddft.scf.gpu4pyscf import run_gpu4pyscf_rks_forward

    calls = []

    class FakeMol:
        pass

    class FakeGrids:
        level = None

    calls = []

    class FakeMF:
        def __init__(self, mol):
            self.mol = mol
            self.xc = None
            self.grids = FakeGrids()
            self.conv_tol = None
            self.max_cycle = None
            self.converged = True
            self.mo_energy = np.array([-0.5, 0.2])
            self.mo_coeff = np.eye(2)
            self.mo_occ = np.array([2.0, 0.0])

        def density_fit(self):
            raise AssertionError("Exact GPU4PySCF SCF forward must not call density_fit().")

        def to_gpu(self):
            calls.append("to_gpu")
            return self

        def kernel(self):
            calls.append("kernel")
            return -1.25

        def make_rdm1(self):
            return np.diag([2.0, 0.0])

        def get_fock(self):
            return np.diag([-0.5, 0.2])

    fake_mf_holder = {}

    def fake_m(atom, basis, unit, spin, charge, cart, verbose):
        calls.append(("M", atom, basis, unit, spin, charge, cart, verbose))
        return FakeMol()

    def fake_rks(mol):
        fake_mf_holder["mf"] = FakeMF(mol)
        return fake_mf_holder["mf"]

    fake_gto = types.SimpleNamespace(M=fake_m)
    fake_dft = types.SimpleNamespace(RKS=fake_rks)
    fake_pyscf = types.ModuleType("pyscf")
    fake_pyscf.gto = fake_gto
    fake_pyscf.dft = fake_dft
    monkeypatch.setitem(sys.modules, "gpu4pyscf", types.ModuleType("gpu4pyscf"))
    monkeypatch.setitem(sys.modules, "pyscf", fake_pyscf)
    monkeypatch.setitem(sys.modules, "pyscf.gto", fake_gto)
    monkeypatch.setitem(sys.modules, "pyscf.dft", fake_dft)

    result = run_gpu4pyscf_rks_forward(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        xc_spec="pbe",
        unit="Angstrom",
        charge=0,
        spin=0,
        cart=True,
        grids_level=1,
        conv_tol=1e-9,
        max_cycle=33,
        verbose=0,
    )

    mf = fake_mf_holder["mf"]
    assert calls[:4] == [
        ("M", "H 0 0 0; H 0 0 0.74", "sto-3g", "Angstrom", 0, 0, True, 0),
        "to_gpu",
        "kernel",
    ]
    assert mf.xc == "pbe"
    assert mf.grids.level == 1
    assert mf.conv_tol == 1e-9
    assert mf.max_cycle == 33
    assert result.converged is True
    assert result.total_energy == -1.25
    assert np.allclose(result.density_matrix, np.diag([2.0, 0.0]))


def test_run_gpu4pyscf_rks_forward_can_skip_collecting_fock_matrix(monkeypatch):
    from td_graddft.scf.gpu4pyscf import run_gpu4pyscf_rks_forward

    calls = []

    class FakeMol:
        pass

    class FakeGrids:
        level = None

    class FakeMF:
        def __init__(self, mol):
            self.mol = mol
            self.grids = FakeGrids()
            self.converged = True
            self.mo_energy = np.array([-0.5, 0.2])
            self.mo_coeff = np.eye(2)
            self.mo_occ = np.array([2.0, 0.0])

        def to_gpu(self):
            return self

        def kernel(self):
            return -1.25

        def make_rdm1(self):
            return np.diag([2.0, 0.0])

        def get_fock(self):
            calls.append("get_fock")
            return np.diag([-0.5, 0.2])

    fake_gto = types.SimpleNamespace(M=lambda **kwargs: FakeMol())
    fake_dft = types.SimpleNamespace(RKS=lambda mol: FakeMF(mol))
    fake_pyscf = types.ModuleType("pyscf")
    fake_pyscf.gto = fake_gto
    fake_pyscf.dft = fake_dft
    monkeypatch.setitem(sys.modules, "gpu4pyscf", types.ModuleType("gpu4pyscf"))
    monkeypatch.setitem(sys.modules, "pyscf", fake_pyscf)
    monkeypatch.setitem(sys.modules, "pyscf.gto", fake_gto)
    monkeypatch.setitem(sys.modules, "pyscf.dft", fake_dft)

    result = run_gpu4pyscf_rks_forward(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        xc_spec="pbe",
        collect_fock=False,
    )

    assert result.fock_matrix is None
    assert calls == []


def test_run_gpu4pyscf_rks_forward_passes_initial_density_to_kernel(monkeypatch):
    from td_graddft.scf.gpu4pyscf import run_gpu4pyscf_rks_forward

    kernel_dm0 = []

    class FakeMol:
        pass

    class FakeGrids:
        level = None

    class FakeMF:
        def __init__(self, mol):
            self.mol = mol
            self.grids = FakeGrids()
            self.converged = True
            self.mo_energy = np.array([-0.5, 0.2])
            self.mo_coeff = np.eye(2)
            self.mo_occ = np.array([2.0, 0.0])

        def to_gpu(self):
            return self

        def kernel(self, dm0=None):
            kernel_dm0.append(dm0)
            return -1.25

        def make_rdm1(self):
            return np.diag([2.0, 0.0])

        def get_fock(self):
            return np.diag([-0.5, 0.2])

    fake_gto = types.SimpleNamespace(M=lambda **kwargs: FakeMol())
    fake_dft = types.SimpleNamespace(RKS=lambda mol: FakeMF(mol))
    fake_pyscf = types.ModuleType("pyscf")
    fake_pyscf.gto = fake_gto
    fake_pyscf.dft = fake_dft
    monkeypatch.setitem(sys.modules, "gpu4pyscf", types.ModuleType("gpu4pyscf"))
    monkeypatch.setitem(sys.modules, "pyscf", fake_pyscf)
    monkeypatch.setitem(sys.modules, "pyscf.gto", fake_gto)
    monkeypatch.setitem(sys.modules, "pyscf.dft", fake_dft)

    initial_density = np.diag([1.5, 0.5])
    result = run_gpu4pyscf_rks_forward(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        xc_spec="pbe",
        initial_density_matrix=initial_density,
    )

    assert result.total_energy == -1.25
    assert len(kernel_dm0) == 1
    assert np.allclose(kernel_dm0[0], initial_density)


def test_cupy_conversion_falls_back_to_numpy_when_no_cuda_device(monkeypatch):
    import td_graddft.scf.gpu4pyscf as gpu4mod

    class FakeRuntime:
        @staticmethod
        def getDeviceCount():
            raise RuntimeError("cudaErrorNoDevice")

    class FakeCupyArray:
        pass

    def fail_asarray(value):
        del value
        raise AssertionError("CuPy conversion should not run when no CUDA device is visible.")

    fake_cupy = types.SimpleNamespace(
        ndarray=FakeCupyArray,
        cuda=types.SimpleNamespace(runtime=FakeRuntime()),
        asarray=fail_asarray,
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    arr = gpu4mod._to_cupy_or_numpy_array(np.diag([1.5, 0.5]))
    gpu4mod._sync_gpu4pyscf()

    assert isinstance(arr, np.ndarray)
    assert np.allclose(arr, np.diag([1.5, 0.5]))


def test_run_gpu4pyscf_uks_forward_uses_direct_to_gpu_without_density_fit(monkeypatch):
    from td_graddft.scf.gpu4pyscf import run_gpu4pyscf_uks_forward

    calls = []

    class FakeMol:
        pass

    class FakeGrids:
        level = None

    class FakeMF:
        def __init__(self, mol):
            self.mol = mol
            self.xc = None
            self.grids = FakeGrids()
            self.conv_tol = None
            self.max_cycle = None
            self.converged = True
            self.mo_energy = [np.array([-0.6, 0.3]), np.array([-0.4, 0.5])]
            self.mo_coeff = [np.eye(2), np.array([[0.0, 1.0], [1.0, 0.0]])]
            self.mo_occ = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]

        def density_fit(self):
            raise AssertionError("Exact GPU4PySCF UKS forward must not call density_fit().")

        def to_gpu(self):
            calls.append("to_gpu")
            return self

        def kernel(self):
            calls.append("kernel")
            return -1.05

        def make_rdm1(self):
            return np.stack([np.diag([1.0, 0.0]), np.diag([0.0, 1.0])], axis=0)

        def get_fock(self):
            return np.stack([np.diag([-0.6, 0.3]), np.diag([-0.4, 0.5])], axis=0)

    fake_mf_holder = {}

    def fake_m(atom, basis, unit, spin, charge, cart, verbose):
        calls.append(("M", atom, basis, unit, spin, charge, cart, verbose))
        return FakeMol()

    def fake_uks(mol):
        fake_mf_holder["mf"] = FakeMF(mol)
        return fake_mf_holder["mf"]

    fake_gto = types.SimpleNamespace(M=fake_m)
    fake_dft = types.SimpleNamespace(UKS=fake_uks)
    fake_pyscf = types.ModuleType("pyscf")
    fake_pyscf.gto = fake_gto
    fake_pyscf.dft = fake_dft
    monkeypatch.setitem(sys.modules, "gpu4pyscf", types.ModuleType("gpu4pyscf"))
    monkeypatch.setitem(sys.modules, "pyscf", fake_pyscf)
    monkeypatch.setitem(sys.modules, "pyscf.gto", fake_gto)
    monkeypatch.setitem(sys.modules, "pyscf.dft", fake_dft)

    result = run_gpu4pyscf_uks_forward(
        atom="O 0 0 0; H 0 0 1",
        basis="sto-3g",
        xc_spec="pbe",
        unit="Angstrom",
        charge=1,
        spin=1,
        cart=True,
        grids_level=1,
        conv_tol=1e-9,
        max_cycle=33,
        verbose=0,
    )

    mf = fake_mf_holder["mf"]
    assert calls[:3] == [
        ("M", "O 0 0 0; H 0 0 1", "sto-3g", "Angstrom", 1, 1, True, 0),
        "to_gpu",
        "kernel",
    ]
    assert mf.xc == "pbe"
    assert mf.grids.level == 1
    assert mf.conv_tol == 1e-9
    assert mf.max_cycle == 33
    assert result.converged is True
    assert result.total_energy == -1.05
    assert np.allclose(result.density_matrix[0], np.diag([1.0, 0.0]))
    assert np.allclose(result.density_matrix[1], np.diag([0.0, 1.0]))


def test_run_gpu4pyscf_uks_forward_injects_neural_xc_into_get_veff(monkeypatch):
    import jax.numpy as jnp

    from td_graddft.scf.gpu4pyscf import run_gpu4pyscf_uks_forward
    from td_graddft.scf.molecules import QuadratureGrid, RestrictedMolecule

    calls = []

    class FakeMol:
        pass

    class FakeGrids:
        level = None

    class FakeMF:
        def __init__(self, mol):
            self.mol = mol
            self.xc = None
            self.grids = FakeGrids()
            self.conv_tol = None
            self.max_cycle = None
            self.converged = True
            self.mo_energy = np.asarray([[-0.6, 0.2], [-0.5, 0.3]])
            self.mo_coeff = np.stack([np.eye(2), np.eye(2)], axis=0)
            self.mo_occ = np.asarray([[1.0, 0.0], [0.0, 1.0]])
            self.recorded_veff = None

        def density_fit(self):
            raise AssertionError("Exact GPU4PySCF UKS forward must not call density_fit().")

        def to_gpu(self):
            return self

        def get_j(self, mol, dm, hermi=1):
            del mol, hermi
            return 2.0 * np.asarray(dm)

        def get_k(self, mol, dm, hermi=1):
            del mol, hermi
            return 4.0 * np.asarray(dm)

        def kernel(self):
            dm = np.stack([np.diag([1.0, 0.0]), np.diag([0.0, 1.0])], axis=0)
            self.recorded_veff = np.asarray(self.get_veff(self.mol, dm))
            return -1.40

        def make_rdm1(self):
            return np.stack([np.diag([1.0, 0.0]), np.diag([0.0, 1.0])], axis=0)

        def get_fock(self):
            return self.recorded_veff

    fake_mf_holder = {}

    def fake_m(**kwargs):
        del kwargs
        return FakeMol()

    def fake_uks(mol):
        fake_mf_holder["mf"] = FakeMF(mol)
        return fake_mf_holder["mf"]

    fake_gto = types.SimpleNamespace(M=fake_m)
    fake_dft = types.SimpleNamespace(UKS=fake_uks)
    fake_pyscf = types.ModuleType("pyscf")
    fake_pyscf.gto = fake_gto
    fake_pyscf.dft = fake_dft
    monkeypatch.setitem(sys.modules, "gpu4pyscf", types.ModuleType("gpu4pyscf"))
    monkeypatch.setitem(sys.modules, "pyscf", fake_pyscf)
    monkeypatch.setitem(sys.modules, "pyscf.gto", fake_gto)
    monkeypatch.setitem(sys.modules, "pyscf.dft", fake_dft)

    molecule_template = RestrictedMolecule(
        ao=jnp.eye(2, dtype=jnp.float32),
        grid=QuadratureGrid(weights=jnp.ones((2,), dtype=jnp.float32)),
        dipole_integrals=jnp.zeros((3, 2, 2), dtype=jnp.float32),
        rep_tensor=jnp.zeros((2, 2, 2, 2), dtype=jnp.float32),
        mo_coeff=jnp.stack([jnp.eye(2, dtype=jnp.float32)] * 2),
        mo_occ=jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32),
        mo_energy=jnp.asarray([[-0.6, 0.2], [-0.5, 0.3]], dtype=jnp.float32),
        rdm1=jnp.stack(
            [
                jnp.diag(jnp.asarray([1.0, 0.0], dtype=jnp.float32)),
                jnp.diag(jnp.asarray([0.0, 1.0], dtype=jnp.float32)),
            ],
            axis=0,
        ),
        h1e=jnp.eye(2, dtype=jnp.float32),
        nuclear_repulsion=0.0,
        overlap_matrix=jnp.eye(2, dtype=jnp.float32),
        nocc=1,
    )

    class FakeSpinFunctional:
        def unrestricted_scf_components(self, molecule_in):
            calls.append(np.asarray(jnp.asarray(molecule_in.rdm1)))
            v_grad = jnp.zeros((2, 3), dtype=jnp.float32)
            return (
                jnp.asarray([10.0, 20.0], dtype=jnp.float32),
                jnp.asarray([30.0, 40.0], dtype=jnp.float32),
                v_grad,
                v_grad,
                "LDA",
                jnp.asarray(0.25, dtype=jnp.float32),
                jnp.diag(jnp.asarray([1.0, 2.0], dtype=jnp.float32)),
                jnp.diag(jnp.asarray([3.0, 4.0], dtype=jnp.float32)),
            )

        def energy_from_molecule(self, params, molecule_in):
            del params, molecule_in
            return jnp.asarray(5.0, dtype=jnp.float32)

    result = run_gpu4pyscf_uks_forward(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        xc_spec="LC_WPBE_LOCAL",
        molecule_template=molecule_template,
        xc_functional=FakeSpinFunctional(),
        xc_params=None,
        neural_vxc_clip=None,
    )

    assert fake_mf_holder["mf"].xc == "pbe"
    assert fake_mf_holder["mf"]._td_graddft_requested_xc_spec == "LC_WPBE_LOCAL"
    assert len(calls) == 1
    assert np.allclose(calls[0], np.stack([np.diag([1.0, 0.0]), np.diag([0.0, 1.0])]))
    expected = np.stack([np.diag([12.0, 24.0]), np.diag([35.0, 45.0])], axis=0)
    assert np.allclose(fake_mf_holder["mf"].recorded_veff, expected)
    assert np.allclose(result.fock_matrix, expected)


def test_gpu4pyscf_direct_jk_response_uses_direct_to_gpu_without_density_fit(monkeypatch):
    import td_graddft.scf.gpu4pyscf as gpu4mod

    from td_graddft.scf.gpu4pyscf import compute_gpu4pyscf_direct_jk_response

    calls = []
    monkeypatch.setattr(gpu4mod, "_cupy_module", lambda: None)

    class FakeMol:
        nao = 2

    class FakeGrids:
        level = None

    class FakeMF:
        def __init__(self, mol):
            self.mol = mol
            self.xc = None
            self.grids = FakeGrids()

        def density_fit(self):
            raise AssertionError("GPU4PySCF JK response helper must use exact/non-DF path.")

        def to_gpu(self):
            calls.append("to_gpu")
            return self

        def get_j(self, mol, dm, hermi=1):
            del mol, hermi
            calls.append("get_j")
            return 2.0 * np.asarray(dm)

        def get_k(self, mol, dm, hermi=1):
            del mol, hermi
            calls.append("get_k")
            return 3.0 * np.asarray(dm)

    def fake_m(**kwargs):
        calls.append(("M", kwargs["atom"], kwargs["basis"], kwargs["cart"]))
        return FakeMol()

    fake_gto = types.SimpleNamespace(M=fake_m)
    fake_dft = types.SimpleNamespace(RKS=lambda mol: FakeMF(mol))
    fake_pyscf = types.ModuleType("pyscf")
    fake_pyscf.gto = fake_gto
    fake_pyscf.dft = fake_dft
    monkeypatch.setitem(sys.modules, "gpu4pyscf", types.ModuleType("gpu4pyscf"))
    monkeypatch.setitem(sys.modules, "pyscf", fake_pyscf)
    monkeypatch.setitem(sys.modules, "pyscf.gto", fake_gto)
    monkeypatch.setitem(sys.modules, "pyscf.dft", fake_dft)

    dm = np.diag([1.0, 0.25])
    j_mat, k_mat = compute_gpu4pyscf_direct_jk_response(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        delta_density=dm,
        cart=True,
    )

    assert calls == [
        ("M", "H 0 0 0; H 0 0 0.74", "sto-3g", True),
        "to_gpu",
        "get_j",
        "get_k",
    ]
    assert np.allclose(np.asarray(j_mat), 2.0 * dm)
    assert np.allclose(np.asarray(k_mat), 3.0 * dm)


def test_gpu4pyscf_direct_jk_response_can_skip_exchange(monkeypatch):
    import td_graddft.scf.gpu4pyscf as gpu4mod

    from td_graddft.scf.gpu4pyscf import compute_gpu4pyscf_direct_jk_response

    calls = []
    gpu4mod._DIRECT_JK_ENGINE_CACHE.clear()
    monkeypatch.setattr(gpu4mod, "_cupy_module", lambda: None)

    class FakeMol:
        nao = 2

    class FakeGrids:
        level = None

    class FakeMF:
        def __init__(self, mol):
            self.mol = mol
            self.xc = None
            self.grids = FakeGrids()

        def to_gpu(self):
            calls.append("to_gpu")
            return self

        def get_j(self, mol, dm, hermi=1):
            del mol, hermi
            calls.append("get_j")
            return 2.0 * np.asarray(dm)

        def get_k(self, mol, dm, hermi=1):
            del mol, dm, hermi
            raise AssertionError("with_k=False must not call get_k.")

    def fake_m(**kwargs):
        calls.append(("M", kwargs["atom"], kwargs["basis"], kwargs["cart"]))
        return FakeMol()

    fake_gto = types.SimpleNamespace(M=fake_m)
    fake_dft = types.SimpleNamespace(RKS=lambda mol: FakeMF(mol))
    fake_pyscf = types.ModuleType("pyscf")
    fake_pyscf.gto = fake_gto
    fake_pyscf.dft = fake_dft
    monkeypatch.setitem(sys.modules, "gpu4pyscf", types.ModuleType("gpu4pyscf"))
    monkeypatch.setitem(sys.modules, "pyscf", fake_pyscf)
    monkeypatch.setitem(sys.modules, "pyscf.gto", fake_gto)
    monkeypatch.setitem(sys.modules, "pyscf.dft", fake_dft)

    try:
        dm = np.diag([1.0, 0.25])
        j_mat, k_mat = compute_gpu4pyscf_direct_jk_response(
            atom="H 0 0 0; H 0 0 0.74",
            basis="sto-3g",
            delta_density=dm,
            cart=True,
            with_k=False,
        )
    finally:
        gpu4mod._DIRECT_JK_ENGINE_CACHE.clear()

    assert calls == [
        ("M", "H 0 0 0; H 0 0 0.74", "sto-3g", True),
        "to_gpu",
        "get_j",
    ]
    assert np.allclose(np.asarray(j_mat), 2.0 * dm)
    assert np.allclose(np.asarray(k_mat), np.zeros_like(dm))


def test_gpu4pyscf_direct_jk_response_reuses_gpu_engine(monkeypatch):
    import td_graddft.scf.gpu4pyscf as gpu4mod

    calls = []
    gpu4mod._DIRECT_JK_ENGINE_CACHE.clear()
    monkeypatch.setattr(gpu4mod, "_cupy_module", lambda: None)

    class FakeMol:
        nao = 2

    class FakeMF:
        def __init__(self, mol):
            self.mol = mol

        def to_gpu(self):
            calls.append("to_gpu")
            return self

        def get_j(self, mol, dm, hermi=1):
            del mol, hermi
            calls.append("get_j")
            return np.asarray(dm)

        def get_k(self, mol, dm, hermi=1):
            del mol, hermi
            calls.append("get_k")
            return np.asarray(dm)

    def fake_m(**kwargs):
        calls.append(("M", kwargs["atom"], kwargs["basis"]))
        return FakeMol()

    fake_gto = types.SimpleNamespace(M=fake_m)
    fake_dft = types.SimpleNamespace(RKS=lambda mol: FakeMF(mol))
    fake_pyscf = types.ModuleType("pyscf")
    fake_pyscf.gto = fake_gto
    fake_pyscf.dft = fake_dft
    monkeypatch.setitem(sys.modules, "gpu4pyscf", types.ModuleType("gpu4pyscf"))
    monkeypatch.setitem(sys.modules, "pyscf", fake_pyscf)
    monkeypatch.setitem(sys.modules, "pyscf.gto", fake_gto)
    monkeypatch.setitem(sys.modules, "pyscf.dft", fake_dft)

    try:
        dm = np.eye(2)
        gpu4mod.compute_gpu4pyscf_direct_jk_response(
            atom="H 0 0 0; H 0 0 0.74",
            basis="sto-3g",
            delta_density=dm,
            cart=True,
        )
        gpu4mod.compute_gpu4pyscf_direct_jk_response(
            atom="H 0 0 0; H 0 0 0.74",
            basis="sto-3g",
            delta_density=2.0 * dm,
            cart=True,
        )
    finally:
        gpu4mod._DIRECT_JK_ENGINE_CACHE.clear()

    assert calls == [
        ("M", "H 0 0 0; H 0 0 0.74", "sto-3g"),
        "to_gpu",
        "get_j",
        "get_k",
        "get_j",
        "get_k",
    ]


def test_gpu4pyscf_direct_jk_response_preserves_float64_density_for_cupy(monkeypatch):
    from td_graddft.scf.gpu4pyscf import compute_gpu4pyscf_direct_jk_response

    calls = []

    class FakeCupyArray:
        def __init__(self, value):
            self.value = np.asarray(value)
            self.dtype = self.value.dtype
            self.shape = self.value.shape

        def __array__(self, dtype=None):
            return np.asarray(self.value, dtype=dtype)

    class FakeStream:
        @staticmethod
        def synchronize():
            calls.append("synchronize")

    fake_cupy = types.ModuleType("cupy")
    fake_cupy.ndarray = FakeCupyArray
    fake_cupy.asarray = lambda value: FakeCupyArray(value)
    fake_cupy.asnumpy = lambda value: np.asarray(value.value)
    fake_cupy.cuda = types.SimpleNamespace(
        Stream=types.SimpleNamespace(null=FakeStream())
    )

    class FakeMol:
        nao = 2

    class FakeMF:
        def __init__(self, mol):
            self.mol = mol

        def to_gpu(self):
            return self

        def get_j(self, mol, dm, hermi=1):
            del mol, hermi
            calls.append(("get_j_dtype", np.asarray(dm).dtype))
            return FakeCupyArray(np.asarray(dm))

        def get_k(self, mol, dm, hermi=1):
            del mol, hermi
            calls.append(("get_k_dtype", np.asarray(dm).dtype))
            return FakeCupyArray(np.asarray(dm))

    fake_gto = types.SimpleNamespace(M=lambda **kwargs: FakeMol())
    fake_dft = types.SimpleNamespace(RKS=lambda mol: FakeMF(mol))
    fake_pyscf = types.ModuleType("pyscf")
    fake_pyscf.gto = fake_gto
    fake_pyscf.dft = fake_dft
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setitem(sys.modules, "gpu4pyscf", types.ModuleType("gpu4pyscf"))
    monkeypatch.setitem(sys.modules, "pyscf", fake_pyscf)
    monkeypatch.setitem(sys.modules, "pyscf.gto", fake_gto)
    monkeypatch.setitem(sys.modules, "pyscf.dft", fake_dft)

    dm = np.diag([1.0, 0.25]).astype(np.float64)
    j_mat, k_mat = compute_gpu4pyscf_direct_jk_response(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        delta_density=dm,
        cart=True,
    )

    assert ("get_j_dtype", np.dtype("float64")) in calls
    assert ("get_k_dtype", np.dtype("float64")) in calls
    assert "synchronize" in calls
    assert np.asarray(j_mat).dtype == np.float64
    assert np.asarray(k_mat).dtype == np.float64


def test_gpu4pyscf_direct_jk_response_from_options_excludes_scf_only_options(monkeypatch):
    import td_graddft.scf.gpu4pyscf as gpu4mod

    captured = {}

    def fake_response(**kwargs):
        captured.update(kwargs)
        density = np.asarray(kwargs["delta_density"])
        return density, density

    monkeypatch.setattr(gpu4mod, "compute_gpu4pyscf_direct_jk_response", fake_response)

    options = gpu4mod.GPU4PySCFRKSForwardOptions(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        xc_spec="pbe",
        conv_tol=1e-12,
        max_cycle=77,
        mol_kwargs={"symmetry": False},
    )
    gpu4mod.compute_gpu4pyscf_direct_jk_response_from_options(
        options,
        np.eye(2),
        with_k=False,
    )

    assert captured["atom"] == "H 0 0 0; H 0 0 0.74"
    assert captured["basis"] == "sto-3g"
    assert captured["xc_spec"] == "pbe"
    assert captured["symmetry"] is False
    assert captured["with_k"] is False
    assert "conv_tol" not in captured
    assert "max_cycle" not in captured


def test_run_gpu4pyscf_rks_forward_injects_neural_xc_into_get_veff(monkeypatch):
    import jax.numpy as jnp

    from td_graddft.scf.gpu4pyscf import run_gpu4pyscf_rks_forward
    from td_graddft.scf.molecules import QuadratureGrid, RestrictedMolecule

    calls = []

    class FakeMol:
        pass

    class FakeGrids:
        level = None

    class FakeMF:
        def __init__(self, mol):
            self.mol = mol
            self.xc = None
            self.grids = FakeGrids()
            self.conv_tol = None
            self.max_cycle = None
            self.converged = True
            self.mo_energy = np.array([-0.5, 0.2])
            self.mo_coeff = np.eye(2)
            self.mo_occ = np.array([2.0, 0.0])
            self.recorded_veff = None

        def density_fit(self):
            raise AssertionError("Exact GPU4PySCF SCF forward must not call density_fit().")

        def to_gpu(self):
            return self

        def get_j(self, mol, dm, hermi=1):
            del mol, hermi
            return 2.0 * np.asarray(dm)

        def get_k(self, mol, dm, hermi=1, omega=None):
            del mol, hermi, omega
            return 4.0 * np.asarray(dm)

        def kernel(self):
            dm = np.diag([2.0, 0.0])
            self.recorded_veff = np.asarray(self.get_veff(self.mol, dm))
            return -1.25

        def make_rdm1(self):
            return np.diag([2.0, 0.0])

        def get_fock(self):
            return self.recorded_veff

    fake_mf_holder = {}

    def fake_m(**kwargs):
        del kwargs
        return FakeMol()

    def fake_rks(mol):
        fake_mf_holder["mf"] = FakeMF(mol)
        return fake_mf_holder["mf"]

    fake_gto = types.SimpleNamespace(M=fake_m)
    fake_dft = types.SimpleNamespace(RKS=fake_rks)
    fake_pyscf = types.ModuleType("pyscf")
    fake_pyscf.gto = fake_gto
    fake_pyscf.dft = fake_dft
    monkeypatch.setitem(sys.modules, "gpu4pyscf", types.ModuleType("gpu4pyscf"))
    monkeypatch.setitem(sys.modules, "pyscf", fake_pyscf)
    monkeypatch.setitem(sys.modules, "pyscf.gto", fake_gto)
    monkeypatch.setitem(sys.modules, "pyscf.dft", fake_dft)

    molecule_template = RestrictedMolecule(
        ao=jnp.eye(2, dtype=jnp.float32),
        grid=QuadratureGrid(weights=jnp.ones((2,), dtype=jnp.float32)),
        dipole_integrals=jnp.zeros((3, 2, 2), dtype=jnp.float32),
        rep_tensor=jnp.zeros((2, 2, 2, 2), dtype=jnp.float32),
        mo_coeff=jnp.stack([jnp.eye(2, dtype=jnp.float32)] * 2),
        mo_occ=jnp.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.float32),
        mo_energy=jnp.asarray([[-0.5, 0.2], [-0.5, 0.2]], dtype=jnp.float32),
        rdm1=jnp.stack([0.5 * jnp.diag(jnp.asarray([2.0, 0.0], dtype=jnp.float32))] * 2),
        h1e=jnp.eye(2, dtype=jnp.float32),
        nuclear_repulsion=0.0,
        overlap_matrix=jnp.eye(2, dtype=jnp.float32),
        nocc=1,
    )

    class FakeNeuralFunctional:
        def scf_potential_components_and_alpha(self, params, molecule_in):
            calls.append(np.asarray(jnp.asarray(molecule_in.rdm1).sum(axis=0)))
            strength = jnp.asarray(params["strength"], dtype=jnp.float32)
            v_rho = strength * jnp.asarray([10.0, 20.0], dtype=jnp.float32)
            v_grad = jnp.zeros((2, 3), dtype=jnp.float32)
            return v_rho, v_grad, "LDA", jnp.asarray(0.25, dtype=jnp.float32)

        def energy_from_molecule(self, params, molecule_in):
            del molecule_in
            return jnp.asarray(params["strength"], dtype=jnp.float32) * 3.0

    result = run_gpu4pyscf_rks_forward(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        xc_spec="LC_WPBE_LOCAL",
        molecule_template=molecule_template,
        xc_functional=FakeNeuralFunctional(),
        xc_params={"strength": jnp.asarray(1.0, dtype=jnp.float32)},
    )

    assert fake_mf_holder["mf"].xc == "pbe"
    assert fake_mf_holder["mf"]._td_graddft_requested_xc_spec == "LC_WPBE_LOCAL"
    assert len(calls) == 1
    assert np.allclose(calls[0], np.diag([2.0, 0.0]))
    assert np.allclose(fake_mf_holder["mf"].recorded_veff, np.diag([13.0, 20.0]))
    assert np.allclose(result.fock_matrix, np.diag([13.0, 20.0]))
    assert result.exact_exchange_fraction == 0.25


def test_run_gpu4pyscf_rks_forward_can_skip_neural_xc_cycle_energy(monkeypatch):
    import jax.numpy as jnp

    from td_graddft.scf.gpu4pyscf import run_gpu4pyscf_rks_forward
    from td_graddft.scf.molecules import QuadratureGrid, RestrictedMolecule

    class FakeMol:
        pass

    class FakeGrids:
        level = None

    class FakeMF:
        def __init__(self, mol):
            self.mol = mol
            self.xc = None
            self.grids = FakeGrids()
            self.conv_tol = None
            self.max_cycle = None
            self.converged = True
            self.mo_energy = np.array([-0.5, 0.2])
            self.mo_coeff = np.eye(2)
            self.mo_occ = np.array([2.0, 0.0])
            self.recorded_veff = None

        def to_gpu(self):
            return self

        def get_j(self, mol, dm, hermi=1):
            del mol, hermi
            return 2.0 * np.asarray(dm)

        def kernel(self):
            dm = np.diag([2.0, 0.0])
            self.recorded_veff = self.get_veff(self.mol, dm)
            return -1.25

        def make_rdm1(self):
            return np.diag([2.0, 0.0])

    fake_mf_holder = {}

    fake_gto = types.SimpleNamespace(M=lambda **kwargs: FakeMol())
    fake_dft = types.SimpleNamespace(
        RKS=lambda mol: fake_mf_holder.setdefault("mf", FakeMF(mol))
    )
    fake_pyscf = types.ModuleType("pyscf")
    fake_pyscf.gto = fake_gto
    fake_pyscf.dft = fake_dft
    monkeypatch.setitem(sys.modules, "gpu4pyscf", types.ModuleType("gpu4pyscf"))
    monkeypatch.setitem(sys.modules, "pyscf", fake_pyscf)
    monkeypatch.setitem(sys.modules, "pyscf.gto", fake_gto)
    monkeypatch.setitem(sys.modules, "pyscf.dft", fake_dft)

    molecule_template = RestrictedMolecule(
        ao=jnp.eye(2, dtype=jnp.float32),
        grid=QuadratureGrid(weights=jnp.ones((2,), dtype=jnp.float32)),
        dipole_integrals=jnp.zeros((3, 2, 2), dtype=jnp.float32),
        rep_tensor=jnp.zeros((2, 2, 2, 2), dtype=jnp.float32),
        mo_coeff=jnp.stack([jnp.eye(2, dtype=jnp.float32)] * 2),
        mo_occ=jnp.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.float32),
        mo_energy=jnp.asarray([[-0.5, 0.2], [-0.5, 0.2]], dtype=jnp.float32),
        rdm1=jnp.stack([0.5 * jnp.diag(jnp.asarray([2.0, 0.0], dtype=jnp.float32))] * 2),
        h1e=jnp.eye(2, dtype=jnp.float32),
        nuclear_repulsion=0.0,
        overlap_matrix=jnp.eye(2, dtype=jnp.float32),
        nocc=1,
    )

    class EnergyCountingFunctional:
        def __init__(self):
            self.energy_calls = 0

        def scf_potential_components_and_alpha(self, params, molecule_in):
            del params
            v_rho = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
            v_grad = jnp.zeros((int(molecule_in.ao.shape[0]), 3), dtype=jnp.float32)
            return v_rho, v_grad, "LDA", jnp.asarray(0.0, dtype=jnp.float32)

        def energy_from_molecule(self, params, molecule_in):
            del params, molecule_in
            self.energy_calls += 1
            return jnp.asarray(99.0, dtype=jnp.float32)

    functional = EnergyCountingFunctional()
    run_gpu4pyscf_rks_forward(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        xc_spec="pbe",
        molecule_template=molecule_template,
        xc_functional=functional,
        xc_params={},
        neural_xc_compute_exc=False,
    )

    assert functional.energy_calls == 0
    assert float(fake_mf_holder["mf"].recorded_veff.exc) == 0.0


def test_neural_xc_fock_payload_reuses_cached_jit_kernel(monkeypatch):
    from dataclasses import replace

    import jax.numpy as jnp

    import td_graddft.scf.gpu4pyscf as gpu4mod
    from td_graddft.scf.gpu4pyscf import GPU4PySCFRKSForwardOptions
    from td_graddft.scf.molecules import QuadratureGrid, RestrictedMolecule

    molecule_template = RestrictedMolecule(
        ao=jnp.eye(2, dtype=jnp.float32),
        grid=QuadratureGrid(weights=jnp.ones((2,), dtype=jnp.float32)),
        dipole_integrals=jnp.zeros((3, 2, 2), dtype=jnp.float32),
        rep_tensor=jnp.zeros((2, 2, 2, 2), dtype=jnp.float32),
        mo_coeff=jnp.stack([jnp.eye(2, dtype=jnp.float32)] * 2),
        mo_occ=jnp.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.float32),
        mo_energy=jnp.asarray([[-0.5, 0.2], [-0.5, 0.2]], dtype=jnp.float32),
        rdm1=jnp.stack([0.5 * jnp.diag(jnp.asarray([2.0, 0.0], dtype=jnp.float32))] * 2),
        h1e=jnp.eye(2, dtype=jnp.float32),
        nuclear_repulsion=0.0,
        overlap_matrix=jnp.eye(2, dtype=jnp.float32),
        nocc=1,
        runtime_scf_options=GPU4PySCFRKSForwardOptions(
            atom="H 0 0 0; H 0 0 0.74",
            basis="sto-3g",
        ),
    )

    class FakeMF:
        mo_coeff = np.eye(2)
        mo_occ = np.array([2.0, 0.0])
        mo_energy = np.array([-0.5, 0.2])

    class FakeFunctional:
        def scf_potential_components_and_alpha(self, params, molecule_in):
            strength = jnp.asarray(params["strength"], dtype=jnp.float32)
            v_rho = strength * jnp.asarray([1.0, 2.0], dtype=jnp.float32)
            v_grad = jnp.zeros((int(molecule_in.ao.shape[0]), 3), dtype=jnp.float32)
            return v_rho, v_grad, "LDA", jnp.asarray(0.0, dtype=jnp.float32)

    jit_calls = []

    stripped_runtime_options = []

    def fake_jit(fn):
        jit_calls.append(fn)

        def wrapped(*args, **kwargs):
            stripped_runtime_options.append(getattr(args[0], "runtime_scf_options", None))
            return fn(*args, **kwargs)

        return wrapped

    gpu4mod._NEURAL_XC_PAYLOAD_JIT_CACHE.clear()
    monkeypatch.setattr(gpu4mod.jax, "jit", fake_jit)

    functional = FakeFunctional()
    molecule_template_2 = replace(molecule_template, nuclear_repulsion=1.0)
    for molecule_arg in (molecule_template, molecule_template_2):
        gpu4mod._neural_xc_fock_payload(
            mf_gpu=FakeMF(),
            dm=np.diag([2.0, 0.0]),
            molecule_template=molecule_arg,
            xc_functional=functional,
            xc_params={"strength": jnp.asarray(1.0, dtype=jnp.float32)},
            neural_vxc_clip=20.0,
            compute_exc=False,
            jit_payload=True,
        )

    assert len(jit_calls) == 1
    assert stripped_runtime_options == [None, None]


def test_neural_xc_uks_fock_payload_reuses_cached_jit_kernel(monkeypatch):
    from dataclasses import replace

    import jax.numpy as jnp

    import td_graddft.scf.gpu4pyscf as gpu4mod
    from td_graddft.scf.gpu4pyscf import GPU4PySCFUKSForwardOptions
    from td_graddft.scf.molecules import QuadratureGrid, RestrictedMolecule

    molecule_template = RestrictedMolecule(
        ao=jnp.eye(2, dtype=jnp.float32),
        grid=QuadratureGrid(weights=jnp.ones((2,), dtype=jnp.float32)),
        dipole_integrals=jnp.zeros((3, 2, 2), dtype=jnp.float32),
        rep_tensor=jnp.zeros((2, 2, 2, 2), dtype=jnp.float32),
        mo_coeff=jnp.stack([jnp.eye(2, dtype=jnp.float32)] * 2),
        mo_occ=jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32),
        mo_energy=jnp.asarray([[-0.6, 0.2], [-0.5, 0.3]], dtype=jnp.float32),
        rdm1=jnp.stack(
            [
                jnp.diag(jnp.asarray([1.0, 0.0], dtype=jnp.float32)),
                jnp.diag(jnp.asarray([0.0, 1.0], dtype=jnp.float32)),
            ],
            axis=0,
        ),
        h1e=jnp.eye(2, dtype=jnp.float32),
        nuclear_repulsion=0.0,
        overlap_matrix=jnp.eye(2, dtype=jnp.float32),
        nocc=1,
        runtime_scf_options=GPU4PySCFUKSForwardOptions(
            atom="H 0 0 0; H 0 0 0.74",
            basis="sto-3g",
            spin=1,
        ),
    )

    class FakeMF:
        mo_coeff = np.stack([np.eye(2), np.eye(2)], axis=0)
        mo_occ = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        mo_energy = np.asarray([[-0.6, 0.2], [-0.5, 0.3]])

    class FakeFunctional:
        def unrestricted_scf_components(self, molecule_in):
            v_grad = jnp.zeros((int(molecule_in.ao.shape[0]), 3), dtype=jnp.float32)
            return (
                jnp.asarray([1.0, 2.0], dtype=jnp.float32),
                jnp.asarray([3.0, 4.0], dtype=jnp.float32),
                v_grad,
                v_grad,
                "LDA",
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.zeros((2, 2), dtype=jnp.float32),
                jnp.zeros((2, 2), dtype=jnp.float32),
            )

    jit_calls = []
    stripped_runtime_options = []

    def fake_jit(fn):
        jit_calls.append(fn)

        def wrapped(*args, **kwargs):
            stripped_runtime_options.append(getattr(args[0], "runtime_scf_options", None))
            return fn(*args, **kwargs)

        return wrapped

    gpu4mod._NEURAL_XC_UKS_PAYLOAD_JIT_CACHE.clear()
    monkeypatch.setattr(gpu4mod.jax, "jit", fake_jit)

    functional = FakeFunctional()
    molecule_template_2 = replace(molecule_template, nuclear_repulsion=1.0)
    for molecule_arg in (molecule_template, molecule_template_2):
        gpu4mod._neural_xc_uks_fock_payload(
            mf_gpu=FakeMF(),
            dm=np.stack([np.diag([1.0, 0.0]), np.diag([0.0, 1.0])], axis=0),
            molecule_template=molecule_arg,
            xc_functional=functional,
            xc_params=None,
            neural_vxc_clip=20.0,
            compute_exc=False,
            jit_payload=True,
        )

    assert len(jit_calls) == 1
    assert stripped_runtime_options == [None, None]


def test_run_molecule_from_spec_accepts_gpu4pyscf_rks_backend(monkeypatch):
    import jax.numpy as jnp

    from td_graddft.scf import GPU4PYSCF_RKS_RUNTIME_BACKEND
    from td_graddft.scf.molecules import QuadratureGrid, RestrictedMolecule
    from td_graddft.workflows.core import run_molecule_from_spec
    from td_graddft.workflows.types import MoleculeSpecConfig, SimulationConfig

    captured = {}

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return RestrictedMolecule(
            ao=jnp.ones((1, 2)),
            grid=QuadratureGrid(weights=jnp.ones((1,))),
            dipole_integrals=jnp.zeros((3, 2, 2)),
            rep_tensor=jnp.zeros((2, 2, 2, 2)),
            mo_coeff=jnp.stack([jnp.eye(2), jnp.eye(2)]),
            mo_occ=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
            mo_energy=jnp.array([[-0.5, 0.2], [-0.5, 0.2]]),
            rdm1=jnp.stack([jnp.diag(jnp.array([1.0, 0.0]))] * 2),
            h1e=jnp.eye(2),
            nuclear_repulsion=0.7,
            overlap_matrix=jnp.eye(2),
            mf_energy=-1.25,
            nocc=1,
            scf_converged=True,
            runtime_scf_backend=GPU4PYSCF_RKS_RUNTIME_BACKEND,
        )

    monkeypatch.setattr(
        "td_graddft.workflows.core.restricted_molecule_from_spec_with_gpu4pyscf_rks",
        fake_builder,
        raising=False,
    )

    run = run_molecule_from_spec(
        MoleculeSpecConfig(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", xc="pbe"),
        simulation=SimulationConfig(
            scf_backend="gpu4pyscf_rks",
            nstates=0,
            jax_rks_max_cycle=44,
            jax_rks_conv_tol=1e-8,
        ),
    )

    assert run.molecule.mf_energy == -1.25
    assert captured["xc_spec"] == "pbe"
    assert captured["basis"] == "sto-3g"
    assert captured["rks_config"].max_cycle == 44
    assert captured["rks_config"].conv_tol == 1e-8
    assert run.molecule.runtime_scf_backend == GPU4PYSCF_RKS_RUNTIME_BACKEND


def test_gpu4pyscf_builder_can_skip_response_eri_slices_without_dropping_hfx(monkeypatch):
    import jax.numpy as jnp

    import td_graddft.scf.builders as builders
    from td_graddft.scf.inputs import RKSIntegralInputs

    captured_inputs = {}
    hfx_calls = []

    def fake_build_inputs(**kwargs):
        captured_inputs.update(kwargs)
        return RKSIntegralInputs(
            basis=types.SimpleNamespace(
                atom_coords=np.zeros((2, 3)),
                atom_charges=np.ones((2,)),
            ),
            overlap=np.eye(2),
            hcore=np.eye(2),
            eri=None,
            eri_pair_matrix=np.ones((3, 3)),
            df_factors=None,
            direct_basis=None,
            nelectron=2,
            nuclear_repulsion=0.5,
            coords=np.zeros((2, 3)),
            grid_weights=np.ones((2,)),
            ao=np.eye(2),
            ao_deriv1=np.zeros((4, 2, 2)),
            ao_laplacian=np.zeros((2, 2)),
            dipole_integrals=np.zeros((3, 2, 2)),
            geometry_is_traced=False,
        )

    def fake_forward(**kwargs):
        del kwargs
        return types.SimpleNamespace(
            density_matrix=np.diag([2.0, 0.0]),
            mo_coeff=np.eye(2),
            mo_occ=np.array([2.0, 0.0]),
            mo_energy=np.array([-0.5, 0.2]),
            total_energy=-1.0,
            exact_exchange_fraction=0.25,
            converged=True,
        )

    def fake_hfx(**kwargs):
        hfx_calls.append(kwargs)
        return jnp.ones((2, 2, 2))

    def fail_response_slices(*args, **kwargs):
        raise AssertionError("response ERI slices should be skipped")

    monkeypatch.setattr(builders, "build_rks_integral_inputs", fake_build_inputs)
    monkeypatch.setattr(builders, "run_gpu4pyscf_rks_forward", fake_forward)
    monkeypatch.setattr(builders, "compute_gpu4pyscf_local_hfx_features", fake_hfx)
    monkeypatch.setattr(builders, "eri_pair_matrix_to_mo_eri_slices", fail_response_slices)

    molecule = builders.restricted_molecule_from_spec_with_gpu4pyscf_rks(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        xc_spec="pbe0",
        compute_local_hfx_features=True,
        compute_response_eri_slices=False,
    )

    assert len(hfx_calls) == 1
    assert captured_inputs["config"].jk_backend == "full"
    assert molecule.hfx_local is not None
    assert molecule.eri_pair_matrix is not None
    assert molecule.eri_ovov is None
    assert molecule.eri_ovvo is None
    assert molecule.eri_oovv is None


def test_gpu4pyscf_builder_passes_packed_eri_pair_matrix_to_pt2(monkeypatch):
    import jax.numpy as jnp

    import td_graddft.scf.builders as builders
    from td_graddft.scf.inputs import RKSIntegralInputs

    eri_pair_matrix = np.arange(9, dtype=float).reshape(3, 3) / 10.0
    captured_pt2 = {}

    def fake_build_inputs(**kwargs):
        del kwargs
        return RKSIntegralInputs(
            basis=types.SimpleNamespace(
                atom_coords=np.zeros((2, 3)),
                atom_charges=np.ones((2,)),
            ),
            overlap=np.eye(2),
            hcore=np.eye(2),
            eri=None,
            eri_pair_matrix=eri_pair_matrix,
            df_factors=None,
            direct_basis=None,
            nelectron=2,
            nuclear_repulsion=0.5,
            coords=np.zeros((2, 3)),
            grid_weights=np.ones((2,)),
            ao=np.eye(2),
            ao_deriv1=np.zeros((4, 2, 2)),
            ao_laplacian=np.zeros((2, 2)),
            dipole_integrals=np.zeros((3, 2, 2)),
            geometry_is_traced=False,
        )

    def fake_forward(**kwargs):
        del kwargs
        return types.SimpleNamespace(
            density_matrix=np.diag([2.0, 0.0]),
            mo_coeff=np.eye(2),
            mo_occ=np.array([2.0, 0.0]),
            mo_energy=np.array([-0.5, 0.2]),
            total_energy=-1.0,
            exact_exchange_fraction=0.0,
            converged=True,
        )

    def fake_pt2(*args, **kwargs):
        del args
        captured_pt2.update(kwargs)
        return jnp.zeros((2,))

    monkeypatch.setattr(builders, "build_rks_integral_inputs", fake_build_inputs)
    monkeypatch.setattr(builders, "run_gpu4pyscf_rks_forward", fake_forward)
    monkeypatch.setattr(builders, "_local_pt2_feature_from_restricted_orbitals", fake_pt2)

    molecule = builders.restricted_molecule_from_spec_with_gpu4pyscf_rks(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        xc_spec="pbe",
        compute_local_pt2_features=True,
    )

    assert np.allclose(np.asarray(captured_pt2["eri_pair_matrix"]), eri_pair_matrix)
    assert np.asarray(captured_pt2["rep_tensor"]).size == 0
    assert molecule.pt2_local is not None


def test_run_molecule_from_spec_accepts_gpu4pyscf_uks_backend(monkeypatch):
    import jax.numpy as jnp

    from td_graddft.scf import GPU4PYSCF_UKS_RUNTIME_BACKEND
    from td_graddft.scf.molecules import QuadratureGrid, UnrestrictedMolecule
    from td_graddft.workflows.core import run_molecule_from_spec
    from td_graddft.workflows.types import MoleculeSpecConfig, SimulationConfig

    captured = {}

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return UnrestrictedMolecule(
            ao=jnp.ones((1, 1)),
            grid=QuadratureGrid(weights=jnp.ones((1,))),
            dipole_integrals=jnp.zeros((3, 1, 1)),
            rep_tensor=jnp.zeros((1, 1, 1, 1)),
            mo_coeff=jnp.stack([jnp.eye(1), jnp.eye(1)]),
            mo_occ=jnp.array([[1.0], [0.0]]),
            mo_energy=jnp.array([[-0.5], [-0.3]]),
            rdm1=jnp.stack([jnp.diag(jnp.array([1.0])), jnp.zeros((1, 1))]),
            h1e=jnp.eye(1),
            nuclear_repulsion=0.7,
            overlap_matrix=jnp.eye(1),
            mf_energy=-0.75,
            nocc_alpha=1,
            nocc_beta=0,
            runtime_scf_backend=GPU4PYSCF_UKS_RUNTIME_BACKEND,
        )

    monkeypatch.setattr(
        "td_graddft.workflows.core.unrestricted_molecule_from_spec_with_gpu4pyscf_uks",
        fake_builder,
        raising=False,
    )

    run = run_molecule_from_spec(
        MoleculeSpecConfig(
            atom="H 0 0 0",
            basis="sto-3g",
            xc="pbe",
            charge=0,
            spin=1,
        ),
        simulation=SimulationConfig(
            scf_backend="gpu4pyscf_uks",
            nstates=0,
            jax_uks_max_cycle=55,
            jax_uks_conv_tol=1e-8,
        ),
    )

    assert run.molecule.mf_energy == -0.75
    assert captured["xc_spec"] == "pbe"
    assert captured["basis"] == "sto-3g"
    assert captured["uks_config"].max_cycle == 55
    assert captured["uks_config"].conv_tol == 1e-8
    assert run.molecule.runtime_scf_backend == GPU4PYSCF_UKS_RUNTIME_BACKEND
    assert GPU4PYSCF_UKS_RUNTIME_BACKEND == "gpu4pyscf_uks"


def test_gpu4pyscf_benchmark_builder_defaults_to_direct_to_gpu(monkeypatch):
    module_path = Path(__file__).resolve().parents[1] / "tools" / "benchmark_gpu4pyscf_vs_strict_jax_full.py"
    spec = spec_from_file_location("benchmark_gpu4pyscf_vs_strict_jax_full_test", module_path)
    assert spec is not None and spec.loader is not None
    bench = module_from_spec(spec)
    spec.loader.exec_module(bench)

    calls = []

    class FakeMol:
        pass

    class FakeGrids:
        level = None

    class FakeMF:
        converged = True
        mo_energy = np.array([-0.5, 0.2])
        mo_coeff = np.eye(2)
        mo_occ = np.array([2.0, 0.0])

        def __init__(self, mol):
            self.mol = mol
            self.xc = None
            self.grids = FakeGrids()

        def density_fit(self):
            raise AssertionError("Benchmark default GPU4PySCF path must be exact/non-DF.")

        def to_gpu(self):
            calls.append("to_gpu")
            return self

        def kernel(self):
            calls.append("kernel")
            return -1.25

    monkeypatch.setattr(bench, "_require_gpu4pyscf", lambda: None)
    monkeypatch.setattr(
        bench.gto,
        "M",
        lambda **kwargs: FakeMol(),
    )
    monkeypatch.setattr(bench.dft, "RKS", lambda mol: FakeMF(mol))
    monkeypatch.setattr(bench, "_sync_gpu", lambda: None)

    _, mf, energy = bench._build_gpu4pyscf_reference(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        xc="pbe",
        grids_level=0,
    )

    assert calls == ["to_gpu", "kernel"]
    assert mf.xc == "pbe"
    assert energy == -1.25
