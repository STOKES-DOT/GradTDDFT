from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.lax import Precision
from jaxtyping import Array, PRNGKeyArray, PyTree


def _identity_coefficients(values: Array) -> Array:
    return jnp.asarray(values)


@dataclass(frozen=True)
class NeuralXCFunctional:
    r"""Composable neural XC core with external inputs and external basis channels.

    This is the minimal GradDFT-style coefficient-basis contract:

    E_xc = \int \sum_k c_theta(x(r))_k * e_k(r) dr

    where:
    - `x(r)` are externally prepared coefficient inputs
    - `e_k(r)` are externally prepared local XC energy-density channels
    - the neural network only maps inputs to coefficients
    """

    model: nn.Module
    coefficient_transform_fn: Callable[[Array], Array] = _identity_coefficients
    name: str = "neural_xc"
    hybrid_fraction_init: float | None = None
    hybrid_fraction_bounds: tuple[float, float] = (0.0, 1.0)

    def init(self, rng: PRNGKeyArray, sample_coefficient_inputs: Array) -> PyTree:
        params = self.model.init(rng, jnp.asarray(sample_coefficient_inputs))
        if self.hybrid_fraction_init is None:
            return params
        lower, upper = self.hybrid_fraction_bounds
        scaled = (self.hybrid_fraction_init - lower) / (upper - lower)
        clipped = jnp.clip(scaled, 1e-6, 1.0 - 1e-6)
        raw = jnp.log(clipped / (1.0 - clipped))
        return {
            "local": params,
            "hybrid_raw": raw,
        }

    def coefficients(self, params: PyTree, coefficient_inputs: Array) -> Array:
        local_params = params["local"] if "local" in params else params
        raw = self.model.apply(local_params, jnp.asarray(coefficient_inputs))
        return jnp.asarray(self.coefficient_transform_fn(raw))

    def energy_density(
        self,
        params: PyTree,
        coefficient_inputs: Array,
        energy_density_channels: Array,
    ) -> Array:
        coefficients = self.coefficients(params, coefficient_inputs)
        basis = jnp.asarray(energy_density_channels)
        if basis.ndim == coefficients.ndim - 1:
            basis = basis[..., None]
        if coefficients.shape != basis.shape:
            raise ValueError(
                "Coefficient/basis channel shape mismatch "
                f"(coefficients={coefficients.shape}, basis={basis.shape})."
            )
        return jnp.einsum("...f,...f->...", coefficients, basis)

    def energy(
        self,
        params: PyTree,
        coefficient_inputs: Array,
        energy_density_channels: Array,
        weights: Array | None = None,
    ) -> Array:
        integrand = self.energy_density(params, coefficient_inputs, energy_density_channels)
        if weights is None:
            return jnp.sum(integrand)
        return jnp.tensordot(jnp.asarray(weights), integrand, axes=(0, 0))

    def hybrid_fraction(self, params: PyTree) -> Array:
        if self.hybrid_fraction_init is None:
            return jnp.asarray(0.0)
        lower, upper = self.hybrid_fraction_bounds
        raw = params["hybrid_raw"]
        return lower + (upper - lower) * jax.nn.sigmoid(raw)

    def exact_exchange_energy(self, molecule: Any) -> Array:
        if getattr(molecule, "rep_tensor", None) is None:
            raise AttributeError("Molecule-like object must define rep_tensor.")
        if getattr(molecule, "rdm1", None) is None:
            raise AttributeError("Molecule-like object must define rdm1.")

        rep_tensor = jnp.asarray(molecule.rep_tensor)
        rdm1 = jnp.asarray(molecule.rdm1)
        if rdm1.ndim == 2:
            rdm1 = jnp.stack([0.5 * rdm1, 0.5 * rdm1], axis=0)

        def spin_exchange(dm_spin):
            exchange_matrix = jnp.einsum(
                "prqs,rs->pq",
                rep_tensor,
                dm_spin,
                precision=Precision.HIGHEST,
            )
            return -0.5 * jnp.einsum(
                "pq,pq->",
                dm_spin,
                exchange_matrix,
                precision=Precision.HIGHEST,
            )

        return jnp.sum(jax.vmap(spin_exchange)(rdm1))
