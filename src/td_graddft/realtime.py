from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax.scipy.linalg import expm
from jaxtyping import Array

from .types import HamiltonianBuilder, Observable, RealTimeState


def commutator(lhs: Array, rhs: Array) -> Array:
    """Compute the matrix commutator [lhs, rhs]."""

    return lhs @ rhs - rhs @ lhs


def ensure_hermitian(matrix: Array) -> Array:
    """Project a matrix onto the Hermitian subspace."""

    return 0.5 * (matrix + matrix.conj().T)


def liouville_rhs(density_matrix: Array, hamiltonian: Array) -> Array:
    """Right-hand side of the Liouville-von Neumann equation."""

    hamiltonian = ensure_hermitian(jnp.asarray(hamiltonian))
    density_matrix = ensure_hermitian(jnp.asarray(density_matrix))
    return -1j * commutator(hamiltonian, density_matrix)


def expectation_value(operator: Array, density_matrix: Array) -> Array:
    """Compute Tr[rho O]."""

    return jnp.trace(jnp.asarray(density_matrix) @ jnp.asarray(operator))


def propagate_step(
    state: RealTimeState,
    dt: float,
    hamiltonian_builder: HamiltonianBuilder,
) -> RealTimeState:
    """Advance one time step with a unitary matrix-exponential propagator."""

    hamiltonian = ensure_hermitian(
        jnp.asarray(hamiltonian_builder(state.time, state.density_matrix))
    )
    propagator = expm(-1j * dt * hamiltonian)
    next_density = propagator @ state.density_matrix @ propagator.conj().T
    next_density = ensure_hermitian(next_density)
    return RealTimeState(
        time=state.time + dt,
        density_matrix=next_density,
        hamiltonian=hamiltonian,
        metadata=state.metadata,
    )


def propagate(
    initial_state: RealTimeState,
    dt: float,
    steps: int,
    hamiltonian_builder: HamiltonianBuilder,
    observables: dict[str, Observable] | None = None,
) -> tuple[list[RealTimeState], dict[str, list[Array]]]:
    """Generate a trajectory and optionally sample observables along it."""

    states = [initial_state]
    traces: dict[str, list[Array]] = {
        name: [observable(initial_state.density_matrix)]
        for name, observable in (observables or {}).items()
    }
    current = initial_state
    for _ in range(steps):
        current = propagate_step(current, dt, hamiltonian_builder)
        states.append(current)
        for name, observable in (observables or {}).items():
            traces[name].append(observable(current.density_matrix))
    return states, traces

