from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np


_TOOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "overfit_water_lc_wpbe_omega_nn.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "overfit_water_lc_wpbe_omega_nn",
    _TOOL_PATH,
)
assert _SPEC is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
_filter_nn_grads = _MODULE._filter_nn_grads
_with_output_bias_omega_delta = _MODULE._with_output_bias_omega_delta


def test_output_bias_omega_scope_keeps_only_omega_logit_gradient():
    grads = {
        "params": {
            "dense": {
                "kernel": jnp.ones((2, 3)),
                "bias": jnp.asarray([1.0, 2.0, 3.0]),
            },
            "output": {
                "kernel": jnp.ones((3, 3)),
                "bias": jnp.asarray([4.0, 5.0, 6.0]),
            },
        }
    }

    filtered = _filter_nn_grads(grads, "output_bias_omega")

    assert np.allclose(filtered["params"]["dense"]["kernel"], 0.0)
    assert np.allclose(filtered["params"]["dense"]["bias"], 0.0)
    assert np.allclose(filtered["params"]["output"]["kernel"], 0.0)
    assert np.allclose(filtered["params"]["output"]["bias"], [0.0, 0.0, 6.0])


def test_output_bias_scope_keeps_all_output_bias_gradients():
    grads = {
        "params": {
            "output": {
                "kernel": jnp.ones((3, 3)),
                "bias": jnp.asarray([4.0, 5.0, 6.0]),
            },
        }
    }

    filtered = _filter_nn_grads(grads, "output_bias")

    assert np.allclose(filtered["params"]["output"]["kernel"], 0.0)
    assert np.allclose(filtered["params"]["output"]["bias"], [4.0, 5.0, 6.0])


def test_output_bias_omega_delta_updates_only_omega_logit_parameter():
    params = {
        "params": {
            "output": {
                "kernel": jnp.ones((3, 3)),
                "bias": jnp.asarray([4.0, 5.0, 6.0]),
            },
        }
    }

    updated = _with_output_bias_omega_delta(params, 0.25)

    assert np.allclose(updated["params"]["output"]["kernel"], 1.0)
    assert np.allclose(updated["params"]["output"]["bias"], [4.0, 5.0, 6.25])
    assert np.allclose(params["params"]["output"]["bias"], [4.0, 5.0, 6.0])
