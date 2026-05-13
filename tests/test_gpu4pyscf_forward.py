import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


def test_run_gpu4pyscf_rks_forward_uses_direct_to_gpu_without_density_fit(monkeypatch):
    from td_graddft.scf.gpu4pyscf import run_gpu4pyscf_rks_forward

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
        xc_spec="pbe",
        molecule_template=molecule_template,
        xc_functional=FakeNeuralFunctional(),
        xc_params={"strength": jnp.asarray(1.0, dtype=jnp.float32)},
    )

    assert len(calls) == 1
    assert np.allclose(calls[0], np.diag([2.0, 0.0]))
    assert np.allclose(fake_mf_holder["mf"].recorded_veff, np.diag([13.0, 20.0]))
    assert np.allclose(result.fock_matrix, np.diag([13.0, 20.0]))
    assert result.exact_exchange_fraction == 0.25


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
