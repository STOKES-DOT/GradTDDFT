from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .xc_backend.jax_xc_adapter import load_jax_xc
from .upstreams import MissingDependencyError


@dataclass(frozen=True)
class AdiabaticDensityFunctional:
    """A small JAX-friendly wrapper around an adiabatic local XC functional."""

    name: str
    energy_density_fn: Callable[[Array], Array]
    exact_exchange_fraction: Array | float = 0.0
    spin_local_potential_fn: Callable[[Array, Array], tuple[Array, Array]] | None = None
    spin_local_kernel_fn: Callable[[Array, Array], tuple[Array, Array, Array]] | None = None

    def energy_density(self, density: Array) -> Array:
        density = jnp.asarray(density)
        return self.energy_density_fn(density)

    def energy(self, density: Array, weights: Array | None = None) -> Array:
        density = jnp.asarray(density)
        integrand = density * self.energy_density(density)
        if weights is None:
            return jnp.sum(integrand)
        return jnp.tensordot(jnp.asarray(weights), integrand, axes=(0, 0))

    def potential(self, density: Array) -> Array:
        density = jnp.asarray(density)
        energy_fn = lambda rho: jnp.sum(rho * self.energy_density(rho))
        return jax.grad(energy_fn)(density)

    def local_potential(self, density: Array) -> Array:
        """Pointwise XC potential for local-density style functionals."""

        density = jnp.asarray(density)
        flat = density.reshape(-1)

        def local_energy(value):
            return value * self.energy_density_fn(value)

        values = jax.vmap(jax.grad(local_energy))(flat)
        return values.reshape(density.shape)

    def local_kernel(self, density: Array) -> Array:
        """Pointwise f_xc for local-density style functionals."""

        density = jnp.asarray(density)
        flat = density.reshape(-1)

        def local_energy(value):
            return value * self.energy_density_fn(value)

        values = jax.vmap(jax.grad(jax.grad(local_energy)))(flat)
        return values.reshape(density.shape)

    def spin_local_potential(self, density_alpha: Array, density_beta: Array) -> tuple[Array, Array]:
        """Spin-resolved local XC potential values (v_xc^a, v_xc^b)."""

        rho_a = jnp.asarray(density_alpha)
        rho_b = jnp.asarray(density_beta)
        if self.spin_local_potential_fn is not None:
            va, vb = self.spin_local_potential_fn(rho_a, rho_b)
            return jnp.asarray(va), jnp.asarray(vb)
        v_tot = self.local_potential(rho_a + rho_b)
        return v_tot, v_tot

    def spin_local_kernel(
        self,
        density_alpha: Array,
        density_beta: Array,
    ) -> tuple[Array, Array, Array]:
        """Spin-resolved local XC kernel entries (f_aa, f_ab, f_bb)."""

        rho_a = jnp.asarray(density_alpha)
        rho_b = jnp.asarray(density_beta)
        if self.spin_local_kernel_fn is not None:
            f_aa, f_ab, f_bb = self.spin_local_kernel_fn(rho_a, rho_b)
            return jnp.asarray(f_aa), jnp.asarray(f_ab), jnp.asarray(f_bb)
        f_tot = self.local_kernel(rho_a + rho_b)
        return f_tot, f_tot, f_tot

    def kernel(self, density: Array) -> Array:
        density = jnp.asarray(density)
        return jax.jacfwd(self.potential)(density)


def lda_from_callable(
    name: str,
    energy_density_fn: Callable[[Array], Array],
) -> AdiabaticDensityFunctional:
    """Build an adiabatic local-density wrapper from a pointwise callable."""

    return AdiabaticDensityFunctional(name=name, energy_density_fn=energy_density_fn)


def lda_from_jax_xc(
    functional_name: str,
    *,
    polarized: bool = False,
    density_floor: float = 1e-12,
) -> AdiabaticDensityFunctional:
    """Wrap a `jax_xc` LDA-like functional as a local adiabatic density object.

    The first scaffold supports unpolarized local-density use only. GGA and mGGA
    plumbing should be added once we thread `GradDFT` grid features through the TD layer.
    """

    if polarized:
        raise NotImplementedError(
            "The first TD-GradDFT scaffold only supports unpolarized LDA wrappers."
        )
    jax_xc, backend = load_jax_xc()
    if jax_xc is None:
        raise MissingDependencyError("Failed to load jax_xc backend.")

    factory = getattr(jax_xc, functional_name, None)
    if factory is None:
        raise AttributeError(
            f"jax_xc backend '{backend}' does not expose a functional named '{functional_name}'."
        )

    functional = factory(polarized=False)

    def energy_density_fn(density: Array) -> Array:
        density = jnp.asarray(density)
        clipped = jnp.maximum(density, density_floor)
        flat = clipped.reshape(-1)

        def evaluate_scalar(value):
            def rho(_coord):
                return value

            return functional(rho, jnp.zeros(3))

        values = jax.vmap(evaluate_scalar)(flat)
        return values.reshape(clipped.shape)

    return AdiabaticDensityFunctional(
        name=functional_name,
        energy_density_fn=energy_density_fn,
    )
