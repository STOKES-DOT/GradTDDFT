from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pytest

from td_graddft.dft.rsh import SCFXCContributions
from td_graddft.scf.differentiable import _scf_xc_components


@dataclass(frozen=True)
class _ToyResolvedXC:
    contributions: SCFXCContributions

    def scf_contributions(self, molecule):
        del molecule
        return self.contributions


@dataclass(frozen=True)
class _ToyFunctional:
    resolved: _ToyResolvedXC

    def bind_to_molecule_for_scf(self, params, molecule):
        del params, molecule
        return self.resolved


@dataclass(frozen=True)
class _ToyMolecule:
    ao: jnp.ndarray


def test_scf_xc_components_accept_new_scf_contributions_protocol():
    molecule = _ToyMolecule(ao=jnp.zeros((4, 3)))
    contributions = SCFXCContributions(
        v_rho=jnp.ones((4,)),
        v_grad=jnp.zeros((4, 3)),
        xc_kind="GGA",
        full_hf_fraction=0.25,
        extra_fock_matrix=jnp.eye(3),
    )
    functional = _ToyFunctional(resolved=_ToyResolvedXC(contributions=contributions))

    v_rho, v_grad, v_tau, v_lapl, xc_kind, alpha, resolved_xc, extra_fock = _scf_xc_components(
        params={},
        functional=functional,
        molecule=molecule,
        functional_dtype=jnp.float32,
    )

    assert xc_kind == "GGA"
    assert np.allclose(np.asarray(v_rho), np.ones((4,)))
    assert np.allclose(np.asarray(v_grad), np.zeros((4, 3)))
    assert np.allclose(np.asarray(v_tau), np.zeros((4,)))
    assert np.allclose(np.asarray(v_lapl), np.zeros((4,)))
    assert np.isclose(float(alpha), 0.25)
    assert resolved_xc is functional.resolved
    assert np.allclose(np.asarray(extra_fock), np.eye(3))


def test_scf_xc_components_reject_unimplemented_lr_hf_channels():
    molecule = _ToyMolecule(ao=jnp.zeros((2, 2)))
    contributions = SCFXCContributions(
        v_rho=jnp.ones((2,)),
        v_grad=jnp.zeros((2, 3)),
        xc_kind="LDA",
        full_hf_fraction=0.0,
        lr_hf_omegas=jnp.asarray([0.4]),
        lr_hf_coefficients=jnp.asarray([1.0]),
    )
    functional = _ToyFunctional(resolved=_ToyResolvedXC(contributions=contributions))

    with pytest.raises(NotImplementedError):
        _scf_xc_components(
            params={},
            functional=functional,
            molecule=molecule,
            functional_dtype=jnp.float32,
        )
