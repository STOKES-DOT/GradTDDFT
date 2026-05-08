from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.lax import Precision
from jaxtyping import Array, PRNGKeyArray, PyTree

from ...xc import AdiabaticDensityFunctional


def _clip_density(density: Array, density_floor: float) -> Array:
    density = jnp.asarray(density)
    return jnp.maximum(density, density_floor)


def _identity_coefficients(values: Array) -> Array:
    return jnp.asarray(values)


def default_lda_coefficient_inputs(
    density: Array,
    *,
    density_floor: float = 1e-12,
) -> Array:
    """Local JAX features for a neural LDA-like functional."""

    rho = _clip_density(density, density_floor)
    rho_third = jnp.cbrt(rho)
    return jnp.stack(
        [
            rho,
            rho_third,
            jnp.sqrt(rho),
            jnp.log1p(rho),
        ],
        axis=-1,
    )


def default_lda_energy_density_basis(
    density: Array,
    *,
    density_floor: float = 1e-12,
) -> Array:
    """Per-particle basis terms inspired by local-density XC structure."""

    rho = _clip_density(density, density_floor)
    rho_third = jnp.cbrt(rho)
    lda_exchange = -(3.0 / 4.0) * (3.0 / jnp.pi) ** (1.0 / 3.0) * rho_third
    return jnp.stack(
        [
            lda_exchange,
            rho_third**2,
            rho,
            jnp.log1p(rho),
        ],
        axis=-1,
    )


class PointwiseMLP(nn.Module):
    """Small pointwise MLP used to generate GradDFT-style XC coefficients."""

    hidden_dims: Sequence[int]
    output_dim: int
    activation: Callable[[Array], Array] = nn.gelu

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        x = inputs
        for width in self.hidden_dims:
            x = nn.Dense(width)(x)
            x = self.activation(x)
        return nn.Dense(self.output_dim)(x)


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


@dataclass(frozen=True)
class DensityNeuralXCFunctional:
    """Density-specialized adapter over the generic coefficient-basis neural XC core."""

    model: nn.Module
    coefficient_input_fn: Callable[..., Array] = default_lda_coefficient_inputs
    energy_density_basis_fn: Callable[..., Array] = default_lda_energy_density_basis
    density_floor: float = 1e-12
    name: str = "neural_xc"
    hybrid_fraction_init: float | None = None
    hybrid_fraction_bounds: tuple[float, float] = (0.0, 1.0)

    def _core(self) -> NeuralXCFunctional:
        return NeuralXCFunctional(
            model=self.model,
            name=self.name,
            hybrid_fraction_init=self.hybrid_fraction_init,
            hybrid_fraction_bounds=self.hybrid_fraction_bounds,
        )

    def coefficient_inputs(self, density: Array) -> Array:
        return self.coefficient_input_fn(density, density_floor=self.density_floor)

    def energy_density_basis(self, density: Array) -> Array:
        return self.energy_density_basis_fn(density, density_floor=self.density_floor)

    def init(self, rng: PRNGKeyArray, sample_density: Array) -> PyTree:
        return self._core().init(rng, self.coefficient_inputs(sample_density))

    def init_from_molecule(self, rng: PRNGKeyArray, molecule: Any) -> PyTree:
        if not hasattr(molecule, "density"):
            raise AttributeError("Molecule-like object must define density() for init.")
        return self.init(rng, jnp.asarray(molecule.density()).sum(axis=-1))

    def coefficients(self, params: PyTree, density: Array) -> Array:
        return self._core().coefficients(params, self.coefficient_inputs(density))

    def energy_density(self, params: PyTree, density: Array) -> Array:
        return self._core().energy_density(
            params,
            self.coefficient_inputs(density),
            self.energy_density_basis(density),
        )

    def energy(self, params: PyTree, density: Array, weights: Array | None = None) -> Array:
        rho = _clip_density(density, self.density_floor)
        local_channels = rho[..., None] * self.energy_density_basis(rho)
        return self._core().energy(
            params,
            self.coefficient_inputs(rho),
            local_channels,
            weights=weights,
        )

    def hybrid_fraction(self, params: PyTree) -> Array:
        return self._core().hybrid_fraction(params)

    def exact_exchange_energy(self, molecule: Any) -> Array:
        return self._core().exact_exchange_energy(molecule)

    def energy_from_molecule(self, params: PyTree, molecule: Any) -> Array:
        if getattr(molecule, "grid", None) is None:
            raise AttributeError("Molecule-like object must define grid.weights.")
        total_density = jnp.asarray(molecule.density()).sum(axis=-1)
        local_energy = self.energy(params, total_density, molecule.grid.weights)
        return local_energy + self.hybrid_fraction(params) * self.exact_exchange_energy(
            molecule
        )

    def local_potential(self, params: PyTree, density: Array) -> Array:
        density = _clip_density(density, self.density_floor)
        flat = density.reshape(-1)

        def local_energy(value):
            return value * self.energy_density(params, value)

        return jax.vmap(jax.grad(local_energy))(flat).reshape(density.shape)

    def local_kernel(self, params: PyTree, density: Array) -> Array:
        density = _clip_density(density, self.density_floor)
        flat = density.reshape(-1)

        def local_energy(value):
            return value * self.energy_density(params, value)

        return jax.vmap(jax.grad(jax.grad(local_energy)))(flat).reshape(density.shape)

    def bind(self, params: PyTree) -> AdiabaticDensityFunctional:
        return AdiabaticDensityFunctional(
            name=self.name,
            energy_density_fn=lambda density: self.energy_density(params, density),
            exact_exchange_fraction=self.hybrid_fraction(params),
        )


def make_neural_lda_functional(
    *,
    hidden_dims: Sequence[int] = (32, 32),
    n_basis: int = 4,
    activation: Callable[[Array], Array] = nn.gelu,
    density_floor: float = 1e-12,
    name: str = "neural_lda_xc",
    hybrid_fraction_init: float | None = None,
) -> DensityNeuralXCFunctional:
    """Factory for a GradDFT-style neural local-density functional."""

    model = PointwiseMLP(
        hidden_dims=tuple(hidden_dims),
        output_dim=n_basis,
        activation=activation,
    )
    return DensityNeuralXCFunctional(
        model=model,
        density_floor=density_floor,
        name=name,
        hybrid_fraction_init=hybrid_fraction_init,
    )
