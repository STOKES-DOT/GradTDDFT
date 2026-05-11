from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.lax import Precision
from jax.scipy import special as jsp_special
from jaxtyping import Array, PRNGKeyArray, PyTree

from ..features import (
    restricted_feature_bundle_from_response_variables,
    restricted_grid_features,
    restricted_grid_features_with_gradients,
    restricted_grid_response_variables,
)
from ..jax_libxc import RestrictedFeatureBundle, _LDA_X_LOCAL_PREFAC
from .components import (
    SemilocalEnergyDensityFn,
    SemilocalEnergyDensityModule,
    legacy_semilocal_module as _legacy_semilocal_module,
    normalize_semilocal_xc_names,
)
from .defaults import (
    DEFAULT_NEURAL_XC_SEMILOCAL_XC,
    DEFAULT_NETWORK_ARCHITECTURE,
    DEFAULT_NETWORK_HIDDEN_DIMS,
)
from .inputs import canonical_input_features, enhanced_input_features
from .networks import SimpleMixingMLP, ResidualMixingMLP, normalize_hidden_dims

@dataclass(frozen=True)
class BoundNeuralXCFunctional:
    name: str
    projected_local_potential_values: Array
    projected_local_kernel_values: Array
    exact_exchange_fraction: Array
    projected_local_potential_gradient_values: Array | None = None
    projected_local_potential_tau_values: Array | None = None
    projected_local_potential_laplacian_values: Array | None = None
    projected_energy_density_values: Array | None = None
    local_hf_fraction_values: Array | None = None
    response_feature_kind: str | None = None
    grid_response_tensor_fn: Callable[[], Array] | None = None
    grid_hfx_feature_gradients_fn: Callable[[], tuple[Array, Array]] | None = None

    def local_kernel(self, density: Array) -> Array:
        del density
        return self.projected_local_kernel_values

    def local_potential(self, density: Array) -> Array:
        del density
        return self.projected_local_potential_values

    def grid_kernel(self, molecule: Any) -> Array:
        del molecule
        return self.projected_local_kernel_values

    def grid_potential(self, molecule: Any) -> Array:
        del molecule
        return self.projected_local_potential_values

    def grid_potential_components(self, molecule: Any) -> tuple[Array, ...]:
        del molecule
        rho = self.projected_local_potential_values
        grad = (
            self.projected_local_potential_gradient_values
            if self.projected_local_potential_gradient_values is not None
            else jnp.zeros(rho.shape + (3,), dtype=rho.dtype)
        )
        tau = (
            self.projected_local_potential_tau_values
            if self.projected_local_potential_tau_values is not None
            else jnp.zeros_like(rho)
        )
        lapl = self.projected_local_potential_laplacian_values
        if lapl is None:
            return rho, grad, tau
        return rho, grad, tau, lapl

    def energy_density(self, density: Array) -> Array:
        del density
        if self.projected_energy_density_values is None:
            return self.projected_local_potential_values
        return self.projected_energy_density_values

    def local_hf_fraction(self, density: Array) -> Array:
        del density
        if self.local_hf_fraction_values is None:
            return jnp.full_like(
                self.projected_local_potential_values,
                self.exact_exchange_fraction,
            )
        return self.local_hf_fraction_values

    def grid_hf_fraction(self, molecule: Any) -> Array:
        del molecule
        if self.local_hf_fraction_values is None:
            return jnp.full_like(
                self.projected_local_potential_values,
                self.exact_exchange_fraction,
            )
        return self.local_hf_fraction_values

    def grid_response_tensor(self, molecule: Any) -> Array:
        del molecule
        if self.grid_response_tensor_fn is None:
            raise AttributeError("This bound functional does not expose a strict response tensor.")
        return self.grid_response_tensor_fn()

    def grid_hfx_feature_gradients(self, molecule: Any) -> tuple[Array, Array]:
        del molecule
        if self.grid_hfx_feature_gradients_fn is None:
            raise AttributeError(
                "This bound functional does not expose gradients with respect to local HF features."
            )
        return self.grid_hfx_feature_gradients_fn()


class NeuralXCBindingMixin:
    def _grid_hfx_feature_gradients(
        self,
        params: PyTree,
        features: RestrictedFeatureBundle,
        semilocal_channels: Array,
        hf_projected: Array,
        hf_feature_a: Array,
        hf_feature_b: Array,
        *,
        pt2_projected: Array | None = None,
        grid_weights: Array,
    ) -> tuple[Array, Array]:
        """Gradient of weighted XC energy with respect to local HF input features."""

        semilocal_total = jnp.sum(semilocal_channels, axis=-1)
        semilocal_local_channels = self._semilocal_local_contribution_channels(
            features,
            semilocal_channels,
        )
        basis = self._assemble_basis_channels(
            semilocal_local_channels,
            hf_projected=hf_projected,
            pt2_projected=pt2_projected,
        )
        weights = jnp.asarray(grid_weights)

        def weighted_energy_from_hfx(hfx_a: Array, hfx_b: Array) -> Array:
            coefficients = self.channel_coefficients(
                params,
                features,
                semilocal_energy_density=semilocal_total,
                hf_energy_density=hf_projected,
                pt2_energy_density=pt2_projected,
                hf_spin_energy_density=(hfx_a, hfx_b),
            )
            channels = self._assemble_channel_contributions(coefficients, basis)
            local_xc = jnp.sum(channels, axis=-1)
            return jnp.tensordot(weights, local_xc, axes=(0, 0))

        grad_a, grad_b = jax.grad(weighted_energy_from_hfx, argnums=(0, 1))(
            hf_feature_a,
            hf_feature_b,
        )
        grad_a = jnp.nan_to_num(grad_a, nan=0.0, posinf=0.0, neginf=0.0)
        grad_b = jnp.nan_to_num(grad_b, nan=0.0, posinf=0.0, neginf=0.0)
        grad_a = self._maybe_clip_response(grad_a)
        grad_b = self._maybe_clip_response(grad_b)
        return grad_a, grad_b

    def projected_local_kernel(
        self,
        params: PyTree,
        molecule: Any,
    ) -> Array:
        features, total_gradient = restricted_grid_features_with_gradients(molecule)
        hf_projected = self.projected_hf_grid_contribution_components(
            molecule,
            features=features,
        )[0]
        pt2_projected = (
            self.projected_pt2_grid_contribution(molecule, features=features)
            if self.include_pt2_channel
            else None
        )
        _, kernel = self._projected_total_potential_kernel(
            params,
            features,
            hf_projected,
            molecule,
            pt2_projected=pt2_projected,
            total_gradient=total_gradient,
        )
        return kernel

    def projected_local_potential(
        self,
        params: PyTree,
        molecule: Any,
    ) -> Array:
        features, total_gradient = restricted_grid_features_with_gradients(molecule)
        hf_projected = self.projected_hf_grid_contribution_components(
            molecule,
            features=features,
        )[0]
        pt2_projected = (
            self.projected_pt2_grid_contribution(molecule, features=features)
            if self.include_pt2_channel
            else None
        )
        potential, _ = self._projected_total_potential_kernel(
            params,
            features,
            hf_projected,
            molecule,
            pt2_projected=pt2_projected,
            total_gradient=total_gradient,
        )
        return potential

    def bind_to_molecule(self, params: PyTree, molecule: Any) -> BoundNeuralXCFunctional:
        features, total_gradient = restricted_grid_features_with_gradients(molecule)
        laplacian, exchange_anchor = self._strict_aux_fields(molecule, features)
        semilocal_channels = self.semilocal_energy_density_channels(features)
        semilocal = jnp.sum(semilocal_channels, axis=-1)
        hf_projected, hf_projected_a, hf_projected_b = self.projected_hf_grid_contribution_components(
            molecule,
            features=features,
        )
        if self.input_feature_mode == "canonical":
            hfx_feature_a, hfx_feature_b = self._canonical_hfx_feature_channels(
                molecule,
                features,
                hf_energy_density=hf_projected,
                hf_spin_energy_density=(hf_projected_a, hf_projected_b),
            )
        else:
            hfx_feature_a, hfx_feature_b = hf_projected_a, hf_projected_b
        pt2_projected = (
            self.projected_pt2_grid_contribution(molecule, features=features)
            if self.include_pt2_channel
            else None
        )
        coefficient_inputs = self.coefficient_inputs(
            features,
            semilocal,
            hf_projected,
            pt2_energy_density=pt2_projected,
            molecule=molecule,
            hf_spin_energy_density=(hf_projected_a, hf_projected_b),
        )
        coefficients = self.channel_coefficients_from_inputs(
            params,
            coefficient_inputs,
        )
        hf_field = self._local_hf_fraction_from_coefficients(coefficients)
        strict_payload = self._strict_response_payload(
            features,
            total_gradient,
            hf_projected,
            pt2_projected=pt2_projected,
            hf_spin_energy_density=(hfx_feature_a, hfx_feature_b),
            laplacian=laplacian,
            exchange_anchor=exchange_anchor,
        )
        projected_vrho, projected_vgrad, projected_vtau, projected_vlapl = self._strict_total_potential_components(
            params,
            features,
            total_gradient,
            hf_projected,
            pt2_projected=pt2_projected,
            hf_spin_energy_density=(hfx_feature_a, hfx_feature_b),
            strict_payload=strict_payload,
        )
        projected_tensor = self._strict_total_response_tensor(
            params,
            features,
            total_gradient,
            hf_projected,
            pt2_projected=pt2_projected,
            hf_spin_energy_density=(hfx_feature_a, hfx_feature_b),
            strict_payload=strict_payload,
        )
        projected_kernel = projected_tensor[0, 0]
        projected_energy_density = jnp.sum(
            self.channel_contributions(
                params,
                molecule,
                features=features,
                semilocal_energy_density=semilocal,
                hf_energy_density=hf_projected,
                pt2_energy_density=pt2_projected,
                hf_spin_energy_density=(hf_projected_a, hf_projected_b),
            ),
            axis=-1,
        )

        rho = jnp.maximum(features.rho, self.density_floor)
        grid_weights = jnp.asarray(molecule.grid.weights)
        numerator = jnp.tensordot(grid_weights, rho * hf_field, axes=(0, 0))
        denominator = jnp.tensordot(grid_weights, rho, axes=(0, 0))
        alpha = numerator / jnp.maximum(denominator, self.density_floor)
        alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
        alpha = jnp.clip(alpha, 0.0, 1.0)

        def grid_response_tensor_fn() -> Array:
            return projected_tensor

        def grid_hfx_feature_gradients_fn() -> tuple[Array, Array]:
            return self._grid_hfx_feature_gradients(
                params,
                features,
                semilocal_channels,
                hf_projected,
                hfx_feature_a,
                hfx_feature_b,
                pt2_projected=pt2_projected,
                grid_weights=molecule.grid.weights,
            )

        return BoundNeuralXCFunctional(
            name=self.name,
            projected_local_potential_values=projected_vrho,
            projected_local_kernel_values=projected_kernel,
            exact_exchange_fraction=alpha,
            projected_local_potential_gradient_values=projected_vgrad,
            projected_local_potential_tau_values=projected_vtau,
            projected_local_potential_laplacian_values=projected_vlapl,
            projected_energy_density_values=projected_energy_density,
            local_hf_fraction_values=(
                hf_field if self.response_hf_mode == "local_projected" else None
            ),
            response_feature_kind=self._response_feature_kind_label(),
            grid_response_tensor_fn=grid_response_tensor_fn,
            grid_hfx_feature_gradients_fn=grid_hfx_feature_gradients_fn,
        )

    def bind_to_molecule_for_response(
        self,
        params: PyTree,
        molecule: Any,
    ) -> BoundNeuralXCFunctional:
        """TD-response-only binding that avoids assembling strict potential terms."""

        features, total_gradient = restricted_grid_features_with_gradients(molecule)
        laplacian, exchange_anchor = self._strict_aux_fields(molecule, features)
        semilocal_channels = self.semilocal_energy_density_channels(features)
        semilocal = jnp.sum(semilocal_channels, axis=-1)
        hf_projected, hf_projected_a, hf_projected_b = self.projected_hf_grid_contribution_components(
            molecule,
            features=features,
        )
        if self.input_feature_mode == "canonical":
            hfx_feature_a, hfx_feature_b = self._canonical_hfx_feature_channels(
                molecule,
                features,
                hf_energy_density=hf_projected,
                hf_spin_energy_density=(hf_projected_a, hf_projected_b),
            )
        else:
            hfx_feature_a, hfx_feature_b = hf_projected_a, hf_projected_b
        pt2_projected = (
            self.projected_pt2_grid_contribution(molecule, features=features)
            if self.include_pt2_channel
            else None
        )
        coefficient_inputs = self.coefficient_inputs(
            features,
            semilocal,
            hf_projected,
            pt2_energy_density=pt2_projected,
            molecule=molecule,
            hf_spin_energy_density=(hf_projected_a, hf_projected_b),
        )
        coefficients = self.channel_coefficients_from_inputs(
            params,
            coefficient_inputs,
        )
        hf_field = self._local_hf_fraction_from_coefficients(coefficients)
        strict_payload = self._strict_response_payload(
            features,
            total_gradient,
            hf_projected,
            pt2_projected=pt2_projected,
            hf_spin_energy_density=(hfx_feature_a, hfx_feature_b),
            laplacian=laplacian,
            exchange_anchor=exchange_anchor,
        )
        projected_tensor = self._strict_total_response_tensor(
            params,
            features,
            total_gradient,
            hf_projected,
            pt2_projected=pt2_projected,
            hf_spin_energy_density=(hfx_feature_a, hfx_feature_b),
            strict_payload=strict_payload,
        )

        rho = jnp.maximum(features.rho, self.density_floor)
        grid_weights = jnp.asarray(molecule.grid.weights)
        numerator = jnp.tensordot(grid_weights, rho * hf_field, axes=(0, 0))
        denominator = jnp.tensordot(grid_weights, rho, axes=(0, 0))
        alpha = numerator / jnp.maximum(denominator, self.density_floor)
        alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
        alpha = jnp.clip(alpha, 0.0, 1.0)

        def grid_response_tensor_fn() -> Array:
            return projected_tensor

        # TD response uses only the strict tensor and scalar HF fraction.
        # Keep the bound object minimal and avoid strict potential/energy assembly.
        return BoundNeuralXCFunctional(
            name=self.name,
            projected_local_potential_values=jnp.zeros_like(features.rho),
            projected_local_kernel_values=projected_tensor[0, 0],
            exact_exchange_fraction=alpha,
            projected_local_potential_gradient_values=None,
            projected_local_potential_tau_values=None,
            projected_local_potential_laplacian_values=None,
            projected_energy_density_values=None,
            local_hf_fraction_values=None,
            response_feature_kind=self._response_feature_kind_label(),
            grid_response_tensor_fn=grid_response_tensor_fn,
            grid_hfx_feature_gradients_fn=None,
        )

    def _scf_binding_payload(
        self,
        params: PyTree,
        molecule: Any,
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Return SCF-only local potential components and scalar HF fraction."""

        features, total_gradient = restricted_grid_features_with_gradients(molecule)
        laplacian, exchange_anchor = self._strict_aux_fields(molecule, features)
        semilocal_channels = self.semilocal_energy_density_channels(features)
        semilocal = jnp.sum(semilocal_channels, axis=-1)
        hf_projected, hf_projected_a, hf_projected_b = self.projected_hf_grid_contribution_components(
            molecule,
            features=features,
        )
        if self.input_feature_mode == "canonical":
            hfx_feature_a, hfx_feature_b = self._canonical_hfx_feature_channels(
                molecule,
                features,
                hf_energy_density=hf_projected,
                hf_spin_energy_density=(hf_projected_a, hf_projected_b),
            )
        else:
            hfx_feature_a, hfx_feature_b = hf_projected_a, hf_projected_b
        pt2_projected = (
            self.projected_pt2_grid_contribution(molecule, features=features)
            if self.include_pt2_channel
            else None
        )

        coefficients = self.channel_coefficients(
            params,
            features,
            molecule=molecule,
            semilocal_energy_density=semilocal,
            hf_energy_density=hf_projected,
            pt2_energy_density=pt2_projected,
            hf_spin_energy_density=(hf_projected_a, hf_projected_b),
        )
        hf_field = self._local_hf_fraction_from_coefficients(coefficients)
        strict_payload = self._strict_response_payload(
            features,
            total_gradient,
            hf_projected,
            pt2_projected=pt2_projected,
            hf_spin_energy_density=(hfx_feature_a, hfx_feature_b),
            laplacian=laplacian,
            exchange_anchor=exchange_anchor,
        )
        projected_vrho, projected_vgrad, projected_vtau, projected_vlapl = self._strict_total_potential_components(
            params,
            features,
            total_gradient,
            hf_projected,
            pt2_projected=pt2_projected,
            hf_spin_energy_density=(hfx_feature_a, hfx_feature_b),
            strict_payload=strict_payload,
        )

        rho = jnp.maximum(features.rho, self.density_floor)
        grid_weights = jnp.asarray(molecule.grid.weights)
        numerator = jnp.tensordot(grid_weights, rho * hf_field, axes=(0, 0))
        denominator = jnp.tensordot(grid_weights, rho, axes=(0, 0))
        alpha = numerator / jnp.maximum(denominator, self.density_floor)
        alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
        alpha = jnp.clip(alpha, 0.0, 1.0)

        return projected_vrho, projected_vgrad, projected_vtau, projected_vlapl, alpha

    def scf_potential_components_and_alpha(
        self,
        params: PyTree,
        molecule: Any,
    ) -> tuple[Array, Array, Array, Array, str, Array]:
        """Direct SCF helper avoiding bound-functional construction."""

        projected_vrho, projected_vgrad, projected_vtau, projected_vlapl, alpha = self._scf_binding_payload(params, molecule)
        return projected_vrho, projected_vgrad, projected_vtau, projected_vlapl, self._response_feature_kind_label(), alpha

    def bind_to_molecule_for_scf(self, params: PyTree, molecule: Any) -> BoundNeuralXCFunctional:
        """SCF-only binding that avoids constructing strict f_xc response terms."""
        projected_vrho, projected_vgrad, projected_vtau, projected_vlapl, alpha = self._scf_binding_payload(
            params,
            molecule,
        )
        # SCF uses only the local potential components and the effective HF fraction.
        # Keep the bound object minimal and avoid assembling response/energy terms here.
        projected_kernel = jnp.zeros_like(projected_vrho)

        return BoundNeuralXCFunctional(
            name=self.name,
            projected_local_potential_values=projected_vrho,
            projected_local_kernel_values=projected_kernel,
            exact_exchange_fraction=alpha,
            projected_local_potential_gradient_values=projected_vgrad,
            projected_local_potential_tau_values=projected_vtau,
            projected_local_potential_laplacian_values=projected_vlapl,
            projected_energy_density_values=None,
            local_hf_fraction_values=None,
            response_feature_kind=self._response_feature_kind_label(),
            grid_response_tensor_fn=None,
            grid_hfx_feature_gradients_fn=None,
        )
