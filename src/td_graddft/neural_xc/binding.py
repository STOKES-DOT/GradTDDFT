from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array, PyTree

from ..features import (
    restricted_grid_features,
    restricted_grid_features_with_gradients,
)
from ..xc_backend.jax_libxc import RestrictedFeatureBundle

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
    nonlocal_response_matrix_fn: Callable[..., Array] | None = None
    nonlocal_response_matrices_fn: Callable[..., tuple[Array, Array]] | None = None

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

    def nonlocal_response_matrix(
        self,
        molecule: Any,
        *,
        occupation_tolerance: float = 1e-8,
    ) -> Array:
        if self.nonlocal_response_matrix_fn is None and self.nonlocal_response_matrices_fn is not None:
            matrix_a, _ = self.nonlocal_response_matrices(
                molecule,
                occupation_tolerance=occupation_tolerance,
            )
            return matrix_a
        if self.nonlocal_response_matrix_fn is None:
            raise AttributeError("This bound functional does not expose a nonlocal response matrix.")
        return self.nonlocal_response_matrix_fn(
            molecule,
            occupation_tolerance=occupation_tolerance,
        )

    def nonlocal_response_matrices(
        self,
        molecule: Any,
        *,
        occupation_tolerance: float = 1e-8,
    ) -> tuple[Array, Array]:
        if self.nonlocal_response_matrices_fn is None:
            matrix = self.nonlocal_response_matrix(
                molecule,
                occupation_tolerance=occupation_tolerance,
            )
            return matrix, matrix
        return self.nonlocal_response_matrices_fn(
            molecule,
            occupation_tolerance=occupation_tolerance,
        )


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
        """Gradient of weighted XC energy with respect to local HF features.

        The derivative includes both uses of the local HF quantity: coefficient
        inputs and the explicit HF basis channel. That makes the result suitable
        for a DM21-style contraction back to a Fock contribution.
        """

        semilocal_total = jnp.sum(semilocal_channels, axis=-1)
        semilocal_local_channels = self._semilocal_local_contribution_channels(
            features,
            semilocal_channels,
        )
        weights = jnp.asarray(grid_weights)

        def _omega0_total(hfx_a: Array, hfx_b: Array) -> Array:
            hfx_a = jnp.asarray(hfx_a)
            hfx_b = jnp.asarray(hfx_b)
            if hfx_a.ndim == jnp.asarray(hf_projected).ndim:
                return hfx_a + hfx_b
            return hfx_a[..., 0] + hfx_b[..., 0]

        def weighted_energy_from_hfx(hfx_a: Array, hfx_b: Array) -> Array:
            hfx_total = _omega0_total(hfx_a, hfx_b)
            coefficients = self.channel_coefficients(
                params,
                features,
                semilocal_energy_density=semilocal_total,
                hf_energy_density=hfx_total,
                pt2_energy_density=pt2_projected,
                hf_spin_energy_density=(hfx_a, hfx_b),
            )
            basis = self._assemble_basis_channels(
                semilocal_local_channels,
                hf_projected=hfx_total,
                pt2_projected=pt2_projected,
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

    def _zero_hfx_fock(self, molecule: Any, dtype: Any | None = None) -> Array:
        ao = jnp.asarray(molecule.ao)
        matrix_dtype = ao.dtype if dtype is None else dtype
        return jnp.zeros((ao.shape[1], ao.shape[1]), dtype=matrix_dtype)

    def uses_explicit_hfx_fock_for_scf(self, molecule: Any) -> bool:
        return getattr(molecule, "hfx_nu", None) is not None

    def _contract_hfx_feature_gradients_to_restricted_fock(
        self,
        molecule: Any,
        grad_a: Array,
        grad_b: Array,
        *,
        dtype: Any | None = None,
    ) -> tuple[Array, bool]:
        nu_cache = getattr(molecule, "hfx_nu", None)
        if nu_cache is None:
            return self._zero_hfx_fock(molecule, dtype), False

        ao = jnp.asarray(molecule.ao)
        matrix_dtype = ao.dtype if dtype is None else dtype
        ao = jnp.asarray(ao, dtype=matrix_dtype)
        nu = jnp.asarray(nu_cache, dtype=matrix_dtype)
        if nu.ndim != 4:
            raise ValueError(
                "molecule.hfx_nu must have shape (n_omega, ngrids, nao, nao), "
                f"got {nu.shape}."
            )
        n_omega, ngrid, nao, nao2 = nu.shape
        if nao != ao.shape[1] or nao2 != ao.shape[1]:
            raise ValueError(
                "molecule.hfx_nu AO dimensions must match molecule.ao second axis "
                f"(got {nu.shape[-2:]} vs {(ao.shape[1], ao.shape[1])})."
            )

        grad_a = jnp.asarray(grad_a, dtype=matrix_dtype)
        grad_b = jnp.asarray(grad_b, dtype=matrix_dtype)
        if grad_a.ndim == 1:
            grad_a = grad_a[:, None]
        if grad_b.ndim == 1:
            grad_b = grad_b[:, None]
        if grad_a.shape != grad_b.shape:
            raise ValueError(
                "HFX feature gradients for alpha and beta spins must have matching shapes "
                f"(got {grad_a.shape} vs {grad_b.shape})."
            )
        if grad_a.shape[0] != ngrid:
            raise ValueError(
                "HFX feature gradient grid axis must match hfx_nu grid axis "
                f"(got {grad_a.shape[0]} vs {ngrid})."
            )
        n_grad_channels = int(grad_a.shape[-1])
        if n_grad_channels > int(n_omega):
            raise ValueError(
                "HFX feature gradient omega axis cannot exceed hfx_nu omega axis "
                f"(got {n_grad_channels} vs {n_omega})."
            )
        nu = nu[:n_grad_channels]
        grad = 0.5 * (grad_a[:, :n_grad_channels] + grad_b[:, :n_grad_channels])
        grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

        rdm1 = jnp.asarray(molecule.rdm1, dtype=matrix_dtype)
        if rdm1.ndim == 2:
            density_half = 0.5 * rdm1
        elif rdm1.ndim == 3 and rdm1.shape[0] == 2:
            density_half = 0.5 * (rdm1[0] + rdm1[1])
        else:
            raise ValueError(
                "Restricted HFX Fock contraction expects rdm1 with shape "
                "(nao, nao) or (2, nao, nao)."
            )
        e = jnp.einsum("gp,pq->gq", ao, density_half, precision=Precision.HIGHEST)
        fxx = jnp.einsum("wgbc,gc->wgb", nu, e, precision=Precision.HIGHEST)
        aow = -0.5 * fxx * jnp.transpose(grad, (1, 0))[:, :, None]
        vmat = jnp.einsum("gp,wgq->pq", ao, aow, precision=Precision.HIGHEST)
        correction = vmat + vmat.T
        correction = jnp.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0)
        return 0.5 * (correction + correction.T), True

    def _explicit_hfx_fock_from_components(
        self,
        params: PyTree,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle,
        semilocal_channels: Array,
        hf_projected: Array,
        hfx_feature_a: Array,
        hfx_feature_b: Array,
        pt2_projected: Array | None,
        grid_weights: Array,
    ) -> tuple[Array, bool]:
        if getattr(molecule, "hfx_nu", None) is None:
            return self._zero_hfx_fock(molecule), False
        grad_a, grad_b = self._grid_hfx_feature_gradients(
            params,
            features,
            semilocal_channels,
            hf_projected,
            hfx_feature_a,
            hfx_feature_b,
            pt2_projected=pt2_projected,
            grid_weights=grid_weights,
        )
        return self._contract_hfx_feature_gradients_to_restricted_fock(
            molecule,
            grad_a,
            grad_b,
        )

    def _explicit_hfx_fock_from_molecule(
        self,
        params: PyTree,
        molecule: Any,
    ) -> tuple[Array, bool]:
        if getattr(molecule, "hfx_nu", None) is None:
            return self._zero_hfx_fock(molecule), False
        features = restricted_grid_features(molecule)
        semilocal_channels = self.semilocal_energy_density_channels(features)
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
        return self._explicit_hfx_fock_from_components(
            params,
            molecule,
            features=features,
            semilocal_channels=semilocal_channels,
            hf_projected=hf_projected,
            hfx_feature_a=hfx_feature_a,
            hfx_feature_b=hfx_feature_b,
            pt2_projected=pt2_projected,
            grid_weights=molecule.grid.weights,
        )

    def _nonlocal_response_components(
        self,
        response_molecule: Any,
    ) -> tuple[RestrictedFeatureBundle, Array, Array, Array, Array, Array | None]:
        response_features, response_total_gradient = restricted_grid_features_with_gradients(
            response_molecule
        )
        response_hf, response_hf_a, response_hf_b = self.projected_hf_grid_contribution_components(
            response_molecule,
            features=response_features,
        )
        if self.input_feature_mode == "canonical":
            response_hfx_a, response_hfx_b = self._canonical_hfx_feature_channels(
                response_molecule,
                response_features,
                hf_energy_density=response_hf,
                hf_spin_energy_density=(response_hf_a, response_hf_b),
            )
        else:
            response_hfx_a, response_hfx_b = response_hf_a, response_hf_b
        response_pt2 = (
            self.projected_pt2_grid_contribution(
                response_molecule,
                features=response_features,
            )
            if self.include_pt2_channel
            else None
        )
        return (
            response_features,
            response_total_gradient,
            response_hf,
            response_hfx_a,
            response_hfx_b,
            response_pt2,
        )

    def _strict_pt2_nonlocal_response_matrix_from_molecule(
        self,
        params: PyTree,
        response_molecule: Any,
        *,
        occupation_tolerance: float,
    ) -> Array:
        (
            response_features,
            response_total_gradient,
            response_hf,
            response_hfx_a,
            response_hfx_b,
            response_pt2,
        ) = self._nonlocal_response_components(response_molecule)
        if response_pt2 is None:
            raise AttributeError("PT2 nonlocal response requires include_pt2_channel=True.")
        return self._strict_pt2_nonlocal_response_matrix(
            params,
            response_molecule,
            response_features,
            response_total_gradient,
            response_hf,
            response_pt2,
            hf_spin_energy_density=(response_hfx_a, response_hfx_b),
            occupation_tolerance=occupation_tolerance,
        )

    def _strict_nonlocal_response_matrices_from_molecule(
        self,
        params: PyTree,
        response_molecule: Any,
        *,
        occupation_tolerance: float,
    ) -> tuple[Array, Array]:
        (
            response_features,
            response_total_gradient,
            response_hf,
            response_hfx_a,
            response_hfx_b,
            response_pt2,
        ) = self._nonlocal_response_components(response_molecule)
        matrix_a, matrix_b = self._strict_hf_nonlocal_response_matrices(
            params,
            response_molecule,
            response_features,
            response_total_gradient,
            response_hf,
            hf_spin_energy_density=(response_hfx_a, response_hfx_b),
            pt2_projected=response_pt2,
            occupation_tolerance=occupation_tolerance,
        )
        if self.include_pt2_channel and self.response_pt2_mode == "strict":
            if response_pt2 is None:
                raise AttributeError("PT2 nonlocal response requires include_pt2_channel=True.")
            pt2_matrix = self._strict_pt2_nonlocal_response_matrix(
                params,
                response_molecule,
                response_features,
                response_total_gradient,
                response_hf,
                response_pt2,
                hf_spin_energy_density=(response_hfx_a, response_hfx_b),
                occupation_tolerance=occupation_tolerance,
            )
            matrix_a = matrix_a + pt2_matrix
            matrix_b = matrix_b + pt2_matrix
        return matrix_a, matrix_b

    def _nonlocal_response_callbacks(
        self,
        params: PyTree,
    ) -> tuple[Callable[..., Array], Callable[..., tuple[Array, Array]]]:
        def nonlocal_response_matrix_fn(
            response_molecule: Any,
            *,
            occupation_tolerance: float = 1e-8,
        ) -> Array:
            return self._strict_pt2_nonlocal_response_matrix_from_molecule(
                params,
                response_molecule,
                occupation_tolerance=occupation_tolerance,
            )

        def nonlocal_response_matrices_fn(
            response_molecule: Any,
            *,
            occupation_tolerance: float = 1e-8,
        ) -> tuple[Array, Array]:
            return self._strict_nonlocal_response_matrices_from_molecule(
                params,
                response_molecule,
                occupation_tolerance=occupation_tolerance,
            )

        return nonlocal_response_matrix_fn, nonlocal_response_matrices_fn

    def _alpha_for_scf_fock(
        self,
        alpha: Array,
        *,
        uses_explicit_hfx_fock: bool,
    ) -> Array:
        if uses_explicit_hfx_fock:
            return jnp.zeros_like(jnp.asarray(alpha))
        return alpha

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
            response_hf_mode=self.response_hf_mode,
            strict_payload=strict_payload,
        )
        projected_kernel = projected_tensor[0, 0]
        semilocal_local_channels = self._semilocal_local_contribution_channels(
            features,
            semilocal_channels,
        )
        basis = self._assemble_basis_channels(
            semilocal_local_channels,
            hf_projected=hf_projected,
            pt2_projected=pt2_projected,
        )
        projected_energy_density = jnp.sum(
            self._assemble_channel_contributions(coefficients, basis),
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

        nonlocal_response_matrix_fn, nonlocal_response_matrices_fn = self._nonlocal_response_callbacks(
            params
        )
        response_alpha = (
            jnp.zeros_like(jnp.asarray(alpha))
            if self.response_hf_mode == "strict"
            else alpha
        )
        return BoundNeuralXCFunctional(
            name=self.name,
            projected_local_potential_values=projected_vrho,
            projected_local_kernel_values=projected_kernel,
            exact_exchange_fraction=response_alpha,
            projected_local_potential_gradient_values=projected_vgrad,
            projected_local_potential_tau_values=projected_vtau,
            projected_local_potential_laplacian_values=projected_vlapl,
            projected_energy_density_values=projected_energy_density,
            local_hf_fraction_values=None,
            response_feature_kind=self._response_feature_kind_label(),
            grid_response_tensor_fn=grid_response_tensor_fn,
            grid_hfx_feature_gradients_fn=grid_hfx_feature_gradients_fn,
            nonlocal_response_matrix_fn=(
                nonlocal_response_matrix_fn
                if (
                    self.response_hf_mode != "strict"
                    and self.include_pt2_channel
                    and self.response_pt2_mode == "strict"
                )
                else None
            ),
            nonlocal_response_matrices_fn=(
                nonlocal_response_matrices_fn
                if self.response_hf_mode == "strict"
                else None
            ),
        )

    def bind_to_molecule_for_response(
        self,
        params: PyTree,
        molecule: Any,
    ) -> BoundNeuralXCFunctional:
        """TD-response-only binding that avoids assembling strict potential terms."""

        features, total_gradient = restricted_grid_features_with_gradients(molecule)
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
        )
        projected_tensor = self._strict_total_response_tensor(
            params,
            features,
            total_gradient,
            hf_projected,
            pt2_projected=pt2_projected,
            hf_spin_energy_density=(hfx_feature_a, hfx_feature_b),
            response_hf_mode=self.response_hf_mode,
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

        nonlocal_response_matrix_fn, nonlocal_response_matrices_fn = self._nonlocal_response_callbacks(
            params
        )
        # TD response uses the configured response tensor and avoids strict
        # potential/energy assembly.
        response_alpha = (
            jnp.zeros_like(jnp.asarray(alpha))
            if self.response_hf_mode == "strict"
            else alpha
        )
        return BoundNeuralXCFunctional(
            name=self.name,
            projected_local_potential_values=jnp.zeros_like(features.rho),
            projected_local_kernel_values=projected_tensor[0, 0],
            exact_exchange_fraction=response_alpha,
            projected_local_potential_gradient_values=None,
            projected_local_potential_tau_values=None,
            projected_local_potential_laplacian_values=None,
            projected_energy_density_values=None,
            local_hf_fraction_values=None,
            response_feature_kind=self._response_feature_kind_label(),
            grid_response_tensor_fn=grid_response_tensor_fn,
            grid_hfx_feature_gradients_fn=None,
            nonlocal_response_matrix_fn=(
                nonlocal_response_matrix_fn
                if (
                    self.response_hf_mode != "strict"
                    and self.include_pt2_channel
                    and self.response_pt2_mode == "strict"
                )
                else None
            ),
            nonlocal_response_matrices_fn=(
                nonlocal_response_matrices_fn
                if self.response_hf_mode == "strict"
                else None
            ),
        )

    def _scf_binding_payload(
        self,
        params: PyTree,
        molecule: Any,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        """Return SCF-only local potential components, HF fraction, and extra Fock."""

        features, total_gradient = restricted_grid_features_with_gradients(molecule)
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
        grid_weights = jnp.asarray(molecule.grid.weights)
        uses_explicit_hfx_fock = self.uses_explicit_hfx_fock_for_scf(molecule)

        if uses_explicit_hfx_fock:
            alpha = jnp.asarray(0.0, dtype=grid_weights.dtype)
        else:
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
            rho = jnp.maximum(features.rho, self.density_floor)
            numerator = jnp.tensordot(grid_weights, rho * hf_field, axes=(0, 0))
            denominator = jnp.tensordot(grid_weights, rho, axes=(0, 0))
            alpha = numerator / jnp.maximum(denominator, self.density_floor)
            alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
            alpha = jnp.clip(alpha, 0.0, 1.0)

        strict_payload = self._strict_response_payload(
            features,
            total_gradient,
            hf_projected,
            pt2_projected=pt2_projected,
            hf_spin_energy_density=(hfx_feature_a, hfx_feature_b),
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

        hfx_fock, uses_explicit_hfx_fock = self._explicit_hfx_fock_from_components(
            params,
            molecule,
            features=features,
            semilocal_channels=semilocal_channels,
            hf_projected=hf_projected,
            hfx_feature_a=hfx_feature_a,
            hfx_feature_b=hfx_feature_b,
            pt2_projected=pt2_projected,
            grid_weights=grid_weights,
        )
        alpha = self._alpha_for_scf_fock(
            alpha,
            uses_explicit_hfx_fock=uses_explicit_hfx_fock,
        )

        return projected_vrho, projected_vgrad, projected_vtau, projected_vlapl, alpha, hfx_fock

    def scf_potential_components_and_alpha(
        self,
        params: PyTree,
        molecule: Any,
    ) -> tuple[Array, Array, Array, Array, str, Array, Array]:
        """Direct SCF helper avoiding bound-functional construction."""

        projected_vrho, projected_vgrad, projected_vtau, projected_vlapl, alpha, hfx_fock = self._scf_binding_payload(
            params,
            molecule,
        )
        return projected_vrho, projected_vgrad, projected_vtau, projected_vlapl, self._response_feature_kind_label(), alpha, hfx_fock

    def bind_to_molecule_for_scf(self, params: PyTree, molecule: Any) -> BoundNeuralXCFunctional:
        """SCF-only binding that avoids constructing strict f_xc response terms."""
        projected_vrho, projected_vgrad, projected_vtau, projected_vlapl, alpha, _ = self._scf_binding_payload(
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
            nonlocal_response_matrix_fn=None,
        )
