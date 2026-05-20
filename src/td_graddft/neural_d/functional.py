from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn import gelu, sigmoid
from jax.nn.initializers import he_normal, zeros
from jaxtyping import Array, PRNGKeyArray, PyTree

from .inputs import build_dispersion_pair_inputs, calculate_distances, molecule_positions_and_atoms


def _as_apply_variables(params: PyTree | None) -> PyTree:
    if params is None:
        return {"params": {}}
    if isinstance(params, Mapping) and (
        "params" in params or "batch_stats" in params or len(params) == 0
    ):
        if "params" in params or "batch_stats" in params:
            return params
        return {"params": {}}
    return {"params": params}


class DispersionFunctional(nn.Module):
    r"""GradDFT-style neural DFT-D dispersion functional.

    The energy form matches GradDFT:
        E_d = -1/2 sum_{A != B} sum_{n=3}^{5}
              f_theta(R_AB, Z_A, Z_B, n) / R_AB^{2n}
    """

    network: nn.Module | None = None
    dispersion: Callable[..., Array] | None = None
    kernel_init: Callable[..., Any] = he_normal()
    bias_init: Callable[..., Any] = zeros
    activation: Callable[[Array], Array] = gelu
    param_dtype: Any = jnp.float32

    def setup(self) -> None:
        self.dense = partial(
            nn.Dense,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )
        self.layer_norm = partial(nn.LayerNorm, param_dtype=self.param_dtype)

    @nn.compact
    def __call__(self, *inputs: Any) -> Array:
        if self.dispersion is not None:
            return self.dispersion(self, *inputs)
        if self.network is None:
            raise ValueError(
                "DispersionFunctional requires either a predefined network or a custom dispersion callable."
            )
        if len(inputs) != 1:
            raise TypeError("Predefined neural_d networks expect one pair-input array.")
        return self.network(jnp.asarray(inputs[0]))

    def head(self, x: Array, local_features: int, sigmoid_scale_factor: float) -> Array:
        x = self.dense(features=local_features)(x)
        self.sow("intermediates", "head_dense", x)
        x = sigmoid(x / sigmoid_scale_factor)
        self.sow("intermediates", "sigmoid", x)
        out = sigmoid_scale_factor * x
        self.sow("intermediates", "sigmoid_product", out)
        return jnp.squeeze(out)

    def pair_inputs(self, molecule: Any, n: int) -> Array:
        return build_dispersion_pair_inputs(molecule, n)

    def init_from_molecule(self, rng: PRNGKeyArray, molecule: Any) -> PyTree:
        return self.init(rng, self.pair_inputs(molecule, 3))

    def energy(self, params: PyTree, atoms: Any, *args: Any, **kwargs: Any) -> Array:
        if atoms.__class__.__name__ == "Solid":
            raise NotImplementedError("Dispersion functionals are not presently implemented for solids")

        positions, atom_indices = molecule_positions_and_atoms(atoms)
        r_ab, atom_pairs = calculate_distances(positions, atom_indices)
        variables = _as_apply_variables(params)

        result = jnp.asarray(0.0, dtype=r_ab.dtype)
        r_flat = jnp.squeeze(r_ab, axis=-1)
        for n in range(3, 6):
            n_column = jnp.asarray(n, dtype=r_ab.dtype) * jnp.ones_like(r_ab)
            x = jnp.concatenate((r_ab, atom_pairs.astype(r_ab.dtype), n_column), axis=-1)
            y = self.apply(variables, x, *args, **kwargs) / r_flat ** (2 * n)
            result = result + jnp.sum(y)
        return -result / 2

    def energy_from_molecule(self, params: PyTree, molecule: Any) -> Array:
        return self.energy(params, molecule)


@dataclass(frozen=True)
class DispersionCorrectedFunctional:
    """Add a GradDFT-style geometry-only dispersion term to an XC functional."""

    base_functional: Any
    dispersion_functional: DispersionFunctional
    dispersion_param_key: str = "dispersion"
    base_param_key: str = "base"

    def _split_params(self, params: PyTree) -> tuple[PyTree, PyTree]:
        if isinstance(params, Mapping):
            dispersion_params = params.get(self.dispersion_param_key, {})
            if self.base_param_key in params:
                return params[self.base_param_key], dispersion_params
            if self.dispersion_param_key in params:
                base_params = {
                    key: value
                    for key, value in params.items()
                    if key != self.dispersion_param_key
                }
                return base_params, dispersion_params
        return params, {}

    def init_from_molecule(self, rng: PRNGKeyArray, molecule: Any) -> PyTree:
        base_rng, dispersion_rng = jax.random.split(rng)
        base_init = getattr(self.base_functional, "init_from_molecule", None)
        if callable(base_init):
            base_params = base_init(base_rng, molecule)
        else:
            base_init = getattr(self.base_functional, "init", None)
            if not callable(base_init):
                raise AttributeError("Base functional must expose init_from_molecule(...) or init(...).")
            base_params = base_init(base_rng, molecule)
        dispersion_variables = self.dispersion_functional.init_from_molecule(
            dispersion_rng,
            molecule,
        )
        dispersion_params = (
            dispersion_variables["params"]
            if isinstance(dispersion_variables, Mapping) and "params" in dispersion_variables
            else dispersion_variables
        )
        if isinstance(base_params, Mapping):
            return {**base_params, self.dispersion_param_key: dispersion_params}
        return {
            self.base_param_key: base_params,
            self.dispersion_param_key: dispersion_params,
        }

    def init(self, rng: PRNGKeyArray, sample: Any) -> PyTree:
        if hasattr(sample, "atom_coords") or hasattr(sample, "nuclear_pos"):
            return self.init_from_molecule(rng, sample)
        base_rng, dispersion_rng = jax.random.split(rng)
        base_init = getattr(self.base_functional, "init", None)
        if not callable(base_init):
            raise AttributeError("Base functional must expose init(...) for non-molecule samples.")
        base_params = base_init(base_rng, sample)
        dispersion_variables = self.dispersion_functional.init(
            dispersion_rng,
            jnp.zeros((1, 4), dtype=jnp.float32),
        )
        dispersion_params = (
            dispersion_variables["params"]
            if isinstance(dispersion_variables, Mapping) and "params" in dispersion_variables
            else dispersion_variables
        )
        if isinstance(base_params, Mapping):
            return {**base_params, self.dispersion_param_key: dispersion_params}
        return {
            self.base_param_key: base_params,
            self.dispersion_param_key: dispersion_params,
        }

    def dispersion_energy_from_molecule(self, params: PyTree, molecule: Any) -> Array:
        _, dispersion_params = self._split_params(params)
        return self.dispersion_functional.energy(dispersion_params, molecule)

    def _base_energy_from_molecule(self, base_params: PyTree, molecule: Any) -> Array:
        energy_from_molecule = getattr(self.base_functional, "energy_from_molecule", None)
        if callable(energy_from_molecule):
            try:
                return jnp.asarray(energy_from_molecule(base_params, molecule))
            except TypeError:
                return jnp.asarray(energy_from_molecule(molecule))
        energy = getattr(self.base_functional, "energy", None)
        if callable(energy):
            try:
                return jnp.asarray(energy(base_params, molecule, include_non_xc=False))
            except TypeError:
                density = molecule.density()
                return jnp.asarray(energy(base_params, density, molecule.grid.weights))
        raise AttributeError("Base functional must expose energy_from_molecule(...) or energy(...).")

    def energy_from_molecule(self, params: PyTree, molecule: Any) -> Array:
        base_params, _ = self._split_params(params)
        return self._base_energy_from_molecule(base_params, molecule) + self.dispersion_energy_from_molecule(
            params,
            molecule,
        )

    def energy_xc_only(self, params: PyTree, molecule: Any) -> Array:
        return self.energy_from_molecule(params, molecule)

    def energy(self, params: PyTree, molecule: Any, *, include_non_xc: bool = False) -> Array:
        base_params, _ = self._split_params(params)
        energy = getattr(self.base_functional, "energy", None)
        if callable(energy):
            try:
                base_energy = jnp.asarray(
                    energy(base_params, molecule, include_non_xc=include_non_xc)
                )
            except TypeError:
                base_energy = self._base_energy_from_molecule(base_params, molecule)
        else:
            base_energy = self._base_energy_from_molecule(base_params, molecule)
        return base_energy + self.dispersion_energy_from_molecule(params, molecule)

    def scf_xc_energy_and_alpha_for_density(
        self,
        params: PyTree,
        molecule: Any,
        density: Array,
    ) -> tuple[Array, Array]:
        base_params, _ = self._split_params(params)
        callback = getattr(self.base_functional, "scf_xc_energy_and_alpha_for_density", None)
        if callable(callback):
            energy, alpha = callback(base_params, molecule, density)
            return (
                jnp.asarray(energy) + self.dispersion_energy_from_molecule(params, molecule),
                alpha,
            )
        energy_callback = getattr(self.base_functional, "scf_xc_energy_for_density", None)
        if not callable(energy_callback):
            raise AttributeError(
                "Base functional must expose scf_xc_energy_and_alpha_for_density(...) "
                "or scf_xc_energy_for_density(...) for SCF use."
            )
        energy = energy_callback(base_params, molecule, density)
        alpha_getter = getattr(self.base_functional, "scf_exact_exchange_fraction", None)
        alpha = (
            alpha_getter(base_params, molecule, density)
            if callable(alpha_getter)
            else jnp.asarray(0.0, dtype=jnp.asarray(energy).dtype)
        )
        return jnp.asarray(energy) + self.dispersion_energy_from_molecule(params, molecule), alpha

    def scf_xc_energy_for_density(
        self,
        params: PyTree,
        molecule: Any,
        density: Array,
    ) -> Array:
        energy, _ = self.scf_xc_energy_and_alpha_for_density(params, molecule, density)
        return energy

    def scf_exact_exchange_fraction(self, params: PyTree, molecule: Any, density: Array) -> Array:
        base_params, _ = self._split_params(params)
        callback = getattr(self.base_functional, "scf_exact_exchange_fraction", None)
        if callable(callback):
            return jnp.asarray(callback(base_params, molecule, density))
        callback = getattr(self.base_functional, "scf_xc_energy_and_alpha_for_density", None)
        if callable(callback):
            _, alpha = callback(base_params, molecule, density)
            return jnp.asarray(alpha)
        return jnp.asarray(0.0, dtype=jnp.asarray(density).dtype)

    def scf_extra_fock_for_density(self, params: PyTree, molecule: Any, density: Array) -> Array:
        base_params, _ = self._split_params(params)
        callback = getattr(self.base_functional, "scf_extra_fock_for_density", None)
        if callable(callback):
            return jnp.asarray(callback(base_params, molecule, density))
        return jnp.zeros_like(jnp.asarray(density))


def make_dispersion_corrected_functional(
    base_functional: Any,
    dispersion_functional: DispersionFunctional,
    *,
    dispersion_param_key: str = "dispersion",
    base_param_key: str = "base",
) -> DispersionCorrectedFunctional:
    return DispersionCorrectedFunctional(
        base_functional=base_functional,
        dispersion_functional=dispersion_functional,
        dispersion_param_key=dispersion_param_key,
        base_param_key=base_param_key,
    )
