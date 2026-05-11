from __future__ import annotations

import numpy as np

from td_graddft.scf.builders import _restricted_reference_array_packaging


def test_restricted_reference_array_packaging_uses_numpy_for_nontraced_values(monkeypatch):
    import td_graddft.scf.builders as reference_mod

    calls = {"stack": 0}
    original_stack = reference_mod.jnp.stack

    def tracking_stack(*args, **kwargs):
        calls["stack"] += 1
        return original_stack(*args, **kwargs)

    monkeypatch.setattr(reference_mod.jnp, "stack", tracking_stack)

    packed = _restricted_reference_array_packaging(
        mo_coeff=np.eye(2),
        mo_occ=np.asarray([1.0, 0.0]),
        mo_energy=np.asarray([-0.5, 0.2]),
        half_dm=np.asarray([[0.5, 0.0], [0.0, 0.0]]),
        h1e=np.eye(2),
        atom_coords=np.zeros((2, 3)),
        atom_charges=np.asarray([1.0, 1.0]),
        overlap=np.eye(2),
        df_factors=None,
        dtype=np.float64,
        traced=False,
    )

    assert calls["stack"] == 0
    assert packed["mo_coeff"].shape == (2, 2, 2)
    assert packed["rdm1"].shape == (2, 2, 2)


def test_restricted_reference_array_packaging_keeps_jax_stack_for_traced_values(monkeypatch):
    import jax.numpy as jnp
    import td_graddft.scf.builders as reference_mod

    calls = {"stack": 0}
    original_stack = reference_mod.jnp.stack

    def tracking_stack(*args, **kwargs):
        calls["stack"] += 1
        return original_stack(*args, **kwargs)

    monkeypatch.setattr(reference_mod.jnp, "stack", tracking_stack)

    packed = _restricted_reference_array_packaging(
        mo_coeff=jnp.eye(2),
        mo_occ=jnp.asarray([1.0, 0.0]),
        mo_energy=jnp.asarray([-0.5, 0.2]),
        half_dm=jnp.asarray([[0.5, 0.0], [0.0, 0.0]]),
        h1e=jnp.eye(2),
        atom_coords=jnp.zeros((2, 3)),
        atom_charges=jnp.asarray([1.0, 1.0]),
        overlap=jnp.eye(2),
        df_factors=None,
        dtype=jnp.float64,
        traced=True,
    )

    assert calls["stack"] >= 4
    assert packed["mo_coeff"].shape == (2, 2, 2)
