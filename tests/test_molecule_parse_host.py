from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import td_graddft.data.molecule as molecule_mod
from td_graddft.data.molecule import ANGSTROM_TO_BOHR, parse_molecule_spec


def test_parse_molecule_spec_uses_numpy_for_literal_coordinates(monkeypatch):
    calls = {"stack": 0}
    original_stack = molecule_mod.jnp.stack

    def tracking_stack(*args, **kwargs):
        calls["stack"] += 1
        return original_stack(*args, **kwargs)

    monkeypatch.setattr(molecule_mod.jnp, "stack", tracking_stack)

    spec = parse_molecule_spec("H 0 0 0; H 0 0 0.74", unit="Angstrom")

    assert calls["stack"] == 0
    assert np.allclose(
        np.asarray(spec.coords_bohr),
        np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74 * ANGSTROM_TO_BOHR]]),
    )


def test_parse_molecule_spec_keeps_jax_path_for_array_coordinates(monkeypatch):
    calls = {"stack": 0}
    original_stack = molecule_mod.jnp.stack

    def tracking_stack(*args, **kwargs):
        calls["stack"] += 1
        return original_stack(*args, **kwargs)

    monkeypatch.setattr(molecule_mod.jnp, "stack", tracking_stack)

    spec = parse_molecule_spec(
        [("H", jnp.asarray([0.0, 0.0, 0.0])), ("H", jnp.asarray([0.0, 0.0, 0.74]))],
        unit="Angstrom",
    )

    assert calls["stack"] == 1
    assert np.allclose(
        np.asarray(spec.coords_bohr),
        np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74 * ANGSTROM_TO_BOHR]]),
    )
