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
    )

    assert calls["stack"] == 0
    assert packed["mo_coeff"].shape == (2, 2, 2)
    assert packed["rdm1"].shape == (2, 2, 2)
