import jax.numpy as jnp

import td_graddft.device as device_module
from td_graddft.scf.molecules import QuadratureGrid, RestrictedMolecule


def _restricted_molecule_with_packed_eri() -> RestrictedMolecule:
    nao = 2
    eri_pair_matrix = jnp.arange(9.0).reshape(3, 3)
    return RestrictedMolecule(
        ao=jnp.eye(nao),
        grid=QuadratureGrid(weights=jnp.ones((nao,)), coords=jnp.zeros((nao, 3))),
        dipole_integrals=jnp.zeros((3, nao, nao)),
        rep_tensor=jnp.zeros((0, 0, 0, 0)),
        mo_coeff=jnp.stack([jnp.eye(nao), jnp.eye(nao)], axis=0),
        mo_occ=jnp.array([[1.0, 0.0], [1.0, 0.0]]),
        mo_energy=jnp.array([[0.0, 1.0], [0.0, 1.0]]),
        rdm1=jnp.stack([jnp.diag(jnp.array([1.0, 0.0]))] * 2, axis=0),
        h1e=jnp.zeros((nao, nao)),
        nuclear_repulsion=0.0,
        hfx_fxx=jnp.ones((1, nao, nao)),
        eri_pair_matrix=eri_pair_matrix,
    )


def test_put_restricted_molecule_on_device_moves_packed_eri(monkeypatch):
    molecule = _restricted_molecule_with_packed_eri()
    calls = []

    def fake_device_put(value, device):
        calls.append((value, device))
        return value

    marker_device = object()
    monkeypatch.setattr(device_module.jax, "device_put", fake_device_put)

    moved = device_module.put_restricted_molecule_on_device(molecule, marker_device)

    assert moved.eri_pair_matrix is molecule.eri_pair_matrix
    assert moved.hfx_fxx is molecule.hfx_fxx
    assert any(value is molecule.eri_pair_matrix and device is marker_device for value, device in calls)
    assert any(value is molecule.hfx_fxx and device is marker_device for value, device in calls)
