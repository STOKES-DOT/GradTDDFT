import jax.numpy as jnp

from td_graddft.realtime import (
    commutator,
    expectation_value,
    liouville_rhs,
    propagate_step,
)
from td_graddft.types import RealTimeState


def test_commutator_of_matrix_with_itself_is_zero():
    matrix = jnp.array([[1.0, 2.0], [2.0, 3.0]])
    assert jnp.allclose(commutator(matrix, matrix), jnp.zeros_like(matrix))


def test_liouville_rhs_is_anti_hermitian():
    density = jnp.array([[1.0 + 0.0j, 0.0], [0.0, 0.0]])
    hamiltonian = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    rhs = liouville_rhs(density, hamiltonian)
    assert jnp.allclose(rhs.conj().T, rhs)


def test_propagate_step_preserves_trace_and_hermiticity():
    density = jnp.array([[1.0 + 0.0j, 0.0], [0.0, 0.0]])
    state = RealTimeState(time=0.0, density_matrix=density)
    sigma_x = jnp.array([[0.0, 1.0], [1.0, 0.0]])

    next_state = propagate_step(state, dt=0.1, hamiltonian_builder=lambda *_: sigma_x)

    assert jnp.allclose(jnp.trace(next_state.density_matrix), 1.0 + 0.0j)
    assert jnp.allclose(next_state.density_matrix, next_state.density_matrix.conj().T)
    assert not jnp.allclose(next_state.density_matrix, density)


def test_expectation_value_uses_trace_convention():
    density = jnp.array([[0.75, 0.0], [0.0, 0.25]])
    operator = jnp.array([[1.0, 0.0], [0.0, -1.0]])
    value = expectation_value(operator, density)
    assert jnp.allclose(value, 0.5)
