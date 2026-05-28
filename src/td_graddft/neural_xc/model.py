from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.lax import Precision
from jaxtyping import Array, PRNGKeyArray, PyTree

from ..features import (
    restricted_feature_bundle_from_response_variables,
    restricted_grid_features,
    restricted_grid_features_with_gradients,
    restricted_transition_response_features,
)
from ..xc_backend.jax_libxc import RestrictedFeatureBundle
from .components import (
    SemilocalEnergyDensityFn,
    SemilocalEnergyDensityModule,
    legacy_semilocal_module as _legacy_semilocal_module,
)
from .defaults import (
    DEFAULT_INPUT_FEATURE_MODE,
    DEFAULT_NEURAL_XC_SEMILOCAL_XC,
    resolve_coefficient_prior_values,
)
from .inputs import (
    assemble_basis_channels,
    build_coefficient_inputs,
    resolve_canonical_hfx_feature_channels,
)
from .projection import NeuralXCProjectionMixin
from .binding import NeuralXCBindingMixin

def _identity_coefficients(values: Array) -> Array:
    return jnp.asarray(values)


@dataclass(frozen=True)
class NeuralXCCore:
    r"""Composable neural XC core with external inputs and external basis channels."""

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


class HeadMixingMixin:
    def _response_feature_kind_label(self) -> str:
        return "MGGA"

    def _unit_interval_coefficients(self, coefficients: Array) -> Array:
        scale = float(getattr(self.model, "sigmoid_scale_factor", 0.0))
        safe = jnp.nan_to_num(coefficients, nan=0.0, posinf=0.0, neginf=0.0)
        if scale > 0.0:
            return jnp.clip(safe / scale, 0.0, 1.0)
        return jnp.clip(safe, 0.0, 1.0)


class AssemblyMixin:
    def _semilocal_local_contribution_channels(
        self,
        features: RestrictedFeatureBundle,
        semilocal_channels: Array,
    ) -> Array:
        return self.resolved_non_hf_module().local_contribution_channels(
            features,
            channels=semilocal_channels,
            density_floor=self.density_floor,
        )

    def _as_descriptor(self, local_contribution: Array, density: Array) -> Array:
        density = jnp.maximum(jnp.asarray(density), self.density_floor)
        return jnp.nan_to_num(
            jnp.asarray(local_contribution) / density,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def _semilocal_input_descriptor(
        self,
        features: RestrictedFeatureBundle,
        semilocal_energy_density: Array,
    ) -> Array:
        return self._as_descriptor(semilocal_energy_density, features.rho)

    def _assemble_basis_channels(
        self,
        semilocal_local_channels: Array,
        *,
        hf_projected: Array,
        pt2_projected: Array | None = None,
    ) -> Array:
        return assemble_basis_channels(
            semilocal_local_channels,
            hf_projected=hf_projected,
            include_pt2_channel=self.include_pt2_channel,
            pt2_projected=pt2_projected,
        )

    def init(self, rng: PRNGKeyArray, sample_inputs: Array) -> PyTree:
        return self._mlp_functional().init(rng, sample_inputs)

    def init_from_molecule(self, rng: PRNGKeyArray, molecule: Any) -> PyTree:
        features = restricted_grid_features(molecule)
        semilocal = self.semilocal_energy_density(features)
        hf_projected, hf_projected_a, hf_projected_b = self.projected_hf_grid_contribution_components(
            molecule,
            features=features,
        )
        pt2_projected = (
            self.projected_pt2_grid_contribution(molecule, features=features)
            if self.include_pt2_channel
            else None
        )
        hf_spin_inputs: tuple[Array, Array] | None = (hf_projected_a, hf_projected_b)
        if (
            self.input_feature_mode == "canonical"
            and self.strict_feature_alignment
            and getattr(molecule, "hfx_local", None) is None
        ):
            hf_spin_inputs = None
        inputs = self.coefficient_inputs(
            features,
            semilocal,
            hf_projected,
            pt2_energy_density=pt2_projected,
            molecule=molecule,
            hf_spin_energy_density=hf_spin_inputs,
        )
        params = self.init(rng, inputs)
        if self.non_hf_module is not None or self.semilocal_energy_density_fn is not None:
            return params
        prior = resolve_coefficient_prior_values(self.semilocal_xc)
        if prior is None:
            return params
        n_semilocal = int(self.resolved_non_hf_module().n_channels)
        expected = n_semilocal + 1 + int(bool(self.include_pt2_channel))
        if self.include_pt2_channel and len(prior) == n_semilocal + 1:
            prior = tuple(prior[:n_semilocal]) + (0.0,) + tuple(prior[n_semilocal:])
        if len(prior) != expected or "params" not in params:
            return params
        scale = float(getattr(self.model, "sigmoid_scale_factor", 0.0))
        target = jnp.asarray(prior)
        if scale > 0.0:
            head_start = n_semilocal
            target = target.at[head_start:].set(target[head_start:] * scale)
            clipped = jnp.clip(target / scale, 1e-6, 1.0 - 1e-6)
            raw_bias = scale * jnp.log(clipped / (1.0 - clipped))
        else:
            raw_bias = target
        variables = params["params"]
        head_name = "HeadDense"
        if head_name not in variables:
            head_name = f"Dense_{len(tuple(getattr(self.model, 'hidden_dims', ())))}"
        if head_name not in variables:
            return params
        head = variables[head_name]
        variables = dict(variables)
        variables[head_name] = {
            **head,
            "kernel": jnp.zeros_like(head["kernel"]),
            "bias": raw_bias.astype(head["bias"].dtype),
        }
        return {**params, "params": variables}

    def compute_densities(
        self,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
    ) -> Array:
        """GradDFT-compatible basis-channel builder e_k(r).

        Returns local grid-contribution channels with shape (..., n_channels):
        [semilocal_1, ..., semilocal_n, pt2_projected?, hf_projected].
        """

        if features is None:
            features = restricted_grid_features(molecule)
        semilocal_channels = self.semilocal_energy_density_channels(features)
        semilocal_local_channels = self._semilocal_local_contribution_channels(
            features,
            semilocal_channels,
        )
        hf_projected, _, _ = self.projected_hf_grid_contribution_components(
            molecule,
            features=features,
        )
        pt2_projected = (
            self.projected_pt2_grid_contribution(molecule, features=features)
            if self.include_pt2_channel
            else None
        )
        return self._assemble_basis_channels(
            semilocal_local_channels,
            hf_projected=hf_projected,
            pt2_projected=pt2_projected,
        )

    def compute_coefficient_inputs(
        self,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
        semilocal_energy_density: Array | None = None,
        hf_energy_density: Array | None = None,
        pt2_energy_density: Array | None = None,
        hf_spin_energy_density: tuple[Array, Array] | None = None,
    ) -> Array:
        """GradDFT-compatible input feature builder for c_theta."""

        if features is None:
            features = restricted_grid_features(molecule)
        semilocal = (
            self.semilocal_energy_density(features)
            if semilocal_energy_density is None
            else jnp.asarray(semilocal_energy_density)
        )
        if hf_energy_density is None:
            hf_projected, hf_projected_a, hf_projected_b = self.projected_hf_grid_contribution_components(
                molecule,
                features=features,
            )
        else:
            hf_projected = jnp.asarray(hf_energy_density)
            if hf_spin_energy_density is None:
                hf_projected_a = hf_projected
                hf_projected_b = hf_projected
            else:
                hf_projected_a, hf_projected_b = hf_spin_energy_density
        spin_inputs = (
            hf_spin_energy_density
            if hf_spin_energy_density is not None
            else (hf_projected_a, hf_projected_b)
        )
        if pt2_energy_density is None and self.include_pt2_channel:
            pt2_energy_density = self.projected_pt2_grid_contribution(
                molecule,
                features=features,
            )
        return self.coefficient_inputs(
            features,
            semilocal,
            hf_projected,
            pt2_energy_density=pt2_energy_density,
            molecule=molecule,
            hf_spin_energy_density=spin_inputs,
        )

    def xc_energy(
        self,
        params: PyTree,
        grid: Any,
        coefficient_inputs: Array,
        densities: Array,
        **_: Any,
    ) -> Array:
        """GradDFT-style XC quadrature from prebuilt inputs/channels."""
        weights = jnp.asarray(getattr(grid, "weights", grid))
        basis = jnp.asarray(densities)
        if basis.ndim == 1:
            basis = basis[:, None]
        local_channels = self._assemble_channel_contributions(
            self.channel_coefficients_from_inputs(params, coefficient_inputs),
            basis,
        )
        return jnp.nan_to_num(
            jnp.tensordot(weights, jnp.sum(local_channels, axis=-1), axes=(0, 0)),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def semilocal_energy_density_channels(self, features: RestrictedFeatureBundle) -> Array:
        return self.resolved_non_hf_module().energy_density_channels(features)

    def semilocal_energy_density(self, features: RestrictedFeatureBundle) -> Array:
        channels = self.semilocal_energy_density_channels(features)
        return jnp.sum(channels, axis=-1)

    def _canonical_hfx_feature_channels(
        self,
        molecule: Any | None,
        features: RestrictedFeatureBundle,
        *,
        hf_energy_density: Array | None = None,
        hf_spin_energy_density: tuple[Array, Array] | None = None,
    ) -> tuple[Array, Array]:
        return resolve_canonical_hfx_feature_channels(
            molecule,
            features,
            hf_energy_density=hf_energy_density,
            hf_spin_energy_density=hf_spin_energy_density,
            hfx_channels=self.hfx_channels,
            strict_feature_alignment=self.strict_feature_alignment,
        )

    def coefficient_inputs(
        self,
        features: RestrictedFeatureBundle,
        semilocal_energy_density: Array,
        hf_energy_density: Array,
        *,
        pt2_energy_density: Array | None = None,
        molecule: Any | None = None,
        hf_spin_energy_density: tuple[Array, Array] | None = None,
    ) -> Array:
        return build_coefficient_inputs(
            features,
            semilocal_energy_density,
            hf_energy_density,
            input_feature_mode=self.input_feature_mode,
            hf_input_mode=self.hf_input_mode,
            include_pt2_channel=self.include_pt2_channel,
            density_floor=self.density_floor,
            hfx_channels=self.hfx_channels,
            strict_feature_alignment=self.strict_feature_alignment,
            pt2_energy_density=pt2_energy_density,
            molecule=molecule,
            hf_spin_energy_density=hf_spin_energy_density,
            semilocal_descriptor=self._semilocal_input_descriptor(
                features,
                semilocal_energy_density,
            ),
        )

    def channel_coefficients(
        self,
        params: PyTree,
        features: RestrictedFeatureBundle,
        *,
        molecule: Any | None = None,
        semilocal_energy_density: Array | None = None,
        hf_energy_density: Array | None = None,
        pt2_energy_density: Array | None = None,
        hf_spin_energy_density: tuple[Array, Array] | None = None,
    ) -> Array:
        semilocal = (
            self.semilocal_energy_density(features)
            if semilocal_energy_density is None
            else semilocal_energy_density
        )
        hf_projected = (
            jnp.zeros_like(semilocal) if hf_energy_density is None else hf_energy_density
        )
        inputs = self.coefficient_inputs(
            features,
            semilocal,
            hf_projected,
            pt2_energy_density=pt2_energy_density,
            molecule=molecule,
            hf_spin_energy_density=hf_spin_energy_density,
        )
        return self.channel_coefficients_from_inputs(params, inputs)

    def channel_coefficients_from_inputs(
        self,
        params: PyTree,
        coefficient_inputs: Array,
    ) -> Array:
        return self._mlp_functional().coefficients(params, coefficient_inputs)

    def _sanitize_coefficients(self, coefficients: Array) -> Array:
        safe = jnp.nan_to_num(coefficients, nan=0.0, posinf=0.0, neginf=0.0)
        n_semilocal = int(self.resolved_non_hf_module().n_channels)
        expected = n_semilocal + 1 + int(bool(self.include_pt2_channel))
        if safe.shape[-1] != expected:
            raise ValueError(
                "Neural XC expects "
                f"{expected} outputs, got {safe.shape[-1]}."
            )
        semilocal = jnp.clip(safe[..., :n_semilocal], 0.0, self.kernel_clip)
        cursor = n_semilocal
        heads: list[Array] = []
        if self.include_pt2_channel:
            heads.append(
                self._unit_interval_coefficients(
                    safe[..., cursor : cursor + 1]
                )
            )
            cursor += 1
        heads.append(
            self._unit_interval_coefficients(
                safe[..., cursor : cursor + 1]
            )
        )
        return jnp.concatenate([semilocal, *heads], axis=-1)

    def mixing_logits(
        self,
        params: PyTree,
        features: RestrictedFeatureBundle,
        *,
        semilocal_energy_density: Array | None = None,
        hf_energy_density: Array | None = None,
    ) -> Array:
        # Backward-compatible alias retained for existing callers/tests.
        return self.channel_coefficients(
            params,
            features,
            semilocal_energy_density=semilocal_energy_density,
            hf_energy_density=hf_energy_density,
        )

    def mixing_weights(
        self,
        params: PyTree,
        features: RestrictedFeatureBundle,
        *,
        semilocal_energy_density: Array | None = None,
        hf_energy_density: Array | None = None,
    ) -> Array:
        coefficients = self.channel_coefficients(
            params,
            features,
            semilocal_energy_density=semilocal_energy_density,
            hf_energy_density=hf_energy_density,
        )
        return coefficients

    def _local_hf_fraction_from_coefficients(self, coefficients: Array) -> Array:
        return jnp.nan_to_num(coefficients[..., -1], nan=0.0, posinf=1.0, neginf=0.0)

    def _local_pt2_fraction_from_coefficients(self, coefficients: Array) -> Array:
        if self.include_pt2_channel:
            return jnp.nan_to_num(coefficients[..., -2], nan=0.0, posinf=1.0, neginf=0.0)
        return jnp.zeros(coefficients.shape[:-1], dtype=coefficients.dtype)

    def _assemble_channel_contributions(
        self,
        coefficients: Array,
        basis: Array,
    ) -> Array:
        n_semilocal = int(self.resolved_non_hf_module().n_channels)
        expected = n_semilocal + 1 + int(bool(self.include_pt2_channel))
        if coefficients.shape[-1] != expected:
            raise ValueError(
                "Neural XC expects "
                f"{expected} outputs, got {coefficients.shape[-1]}."
            )
        if basis.shape[-1] != expected:
            raise ValueError(
                "Neural XC expects basis channels [semilocal..., pt2?, hf], "
                f"got shape[-1]={basis.shape[-1]}."
            )
        semilocal = coefficients[..., :n_semilocal] * basis[..., :n_semilocal]
        cursor = n_semilocal
        channels = [semilocal]
        if self.include_pt2_channel:
            channels.append(
                coefficients[..., cursor : cursor + 1] * basis[..., cursor : cursor + 1]
            )
            cursor += 1
        channels.append(
            coefficients[..., cursor : cursor + 1] * basis[..., cursor : cursor + 1]
        )
        return jnp.concatenate(channels, axis=-1)

    def mixing_fields(
        self,
        params: PyTree,
        features: RestrictedFeatureBundle,
        *,
        semilocal_energy_density: Array | None = None,
        hf_energy_density: Array | None = None,
    ) -> Array:
        # Backward-compatible alias retained for existing callers/tests.
        return self.mixing_weights(
            params,
            features,
            semilocal_energy_density=semilocal_energy_density,
            hf_energy_density=hf_energy_density,
        )


class ResponseMixin:
    def _default_total_gradient_from_features(
        self,
        features: RestrictedFeatureBundle,
    ) -> Array:
        sigma = jnp.maximum(features.sigma, 0.0)
        return jnp.stack(
            [jnp.sqrt(sigma), jnp.zeros_like(sigma), jnp.zeros_like(sigma)],
            axis=-1,
        )

    def _response_variables(
        self,
        features: RestrictedFeatureBundle,
        total_gradient: Array | None = None,
    ) -> tuple[Array, Array, Array, Array]:
        response_floor = self._effective_response_density_floor()
        rho0 = jnp.maximum(features.rho, response_floor)
        tau0 = jnp.maximum(features.tau_a + features.tau_b, 0.0)
        if total_gradient is None:
            grad0 = self._default_total_gradient_from_features(features)
        else:
            grad0 = jnp.asarray(total_gradient, dtype=rho0.dtype)
            if grad0.ndim != rho0.ndim + 1 or grad0.shape[-1] != 3:
                raise ValueError(
                    "total_gradient must have shape (..., 3) matching features.rho."
                )
        variables = jnp.concatenate(
            [rho0[..., None], grad0, tau0[..., None]],
            axis=-1,
        )
        return rho0, grad0, tau0, variables

    def _strict_response_payload(
        self,
        features: RestrictedFeatureBundle,
        total_gradient: Array,
        hf_projected: Array,
        *,
        pt2_projected: Array | None = None,
        hf_spin_energy_density: tuple[Array, Array] | None = None,
    ) -> tuple[Array, Array, Array, Array, Array]:
        rho0, _, _, response_variables = self._response_variables(
            features,
            total_gradient,
        )
        if hf_spin_energy_density is None:
            hf_feature_a = hf_projected
            hf_feature_b = hf_projected
        else:
            hf_feature_a, hf_feature_b = hf_spin_energy_density
        pt2_feature = (
            jnp.zeros_like(hf_projected)
            if pt2_projected is None
            else jnp.asarray(pt2_projected)
        )
        active = rho0 > self._effective_response_density_floor()
        return (
            response_variables,
            active,
            hf_feature_a,
            hf_feature_b,
            pt2_feature,
        )

    def _semilocal_point_local_energy_from_variables(self, variables: Array) -> Array:
        response_floor = self._effective_response_density_floor()
        rho_point = jnp.maximum(variables[0], response_floor)
        grad_point = variables[1:4]
        tau_point = jnp.maximum(variables[4], 0.0)
        point_features = restricted_feature_bundle_from_response_variables(
            rho_point,
            grad_point,
            tau_point,
            density_floor=response_floor,
        )
        semilocal_channels = self.semilocal_energy_density_channels(point_features)
        semilocal_local_channels = self._semilocal_local_contribution_channels(
            point_features,
            semilocal_channels,
        )
        return jnp.sum(semilocal_local_channels, axis=-1)

    def _total_point_local_energy_from_variables(
        self,
        params: PyTree,
        variables: Array,
        hf_point: Array,
        hf_point_a: Array,
        hf_point_b: Array,
        *,
        pt2_point: Array | None = None,
        response_hf_mode: Literal["approx", "strict"] | None = None,
        response_pt2_mode: Literal["approx", "strict"] | None = None,
    ) -> Array:
        hf_mode = self.response_hf_mode if response_hf_mode is None else response_hf_mode
        pt2_mode = self.response_pt2_mode if response_pt2_mode is None else response_pt2_mode
        response_floor = self._effective_response_density_floor()
        rho_point = jnp.maximum(variables[0], response_floor)
        grad_point = variables[1:4]
        tau_point = jnp.maximum(variables[4], 0.0)
        point_features = restricted_feature_bundle_from_response_variables(
            rho_point,
            grad_point,
            tau_point,
            density_floor=response_floor,
        )
        semilocal_channels = self.semilocal_energy_density_channels(point_features)
        semilocal_local_channels = self._semilocal_local_contribution_channels(
            point_features,
            semilocal_channels,
        )
        semilocal_total = jnp.sum(semilocal_channels, axis=-1)
        if hf_mode in {"approx", "strict"}:
            hf_input = jax.lax.stop_gradient(hf_point)
            hf_basis = jnp.zeros_like(hf_input)
            hf_spin_inputs: tuple[Array, Array] | None = (
                jax.lax.stop_gradient(hf_point_a),
                jax.lax.stop_gradient(hf_point_b),
            )
        else:
            raise ValueError(
                f"Unsupported response_hf_mode={hf_mode!r}. "
                "Expected 'strict' or 'approx'."
            )
        if pt2_point is None:
            pt2_point = jnp.zeros_like(hf_point)
        if self.include_pt2_channel:
            if pt2_mode == "approx":
                pt2_input = jax.lax.stop_gradient(pt2_point)
                pt2_basis = pt2_input
            elif pt2_mode == "strict":
                pt2_input = jnp.zeros_like(hf_point)
                pt2_basis = jnp.zeros_like(hf_point)
            else:
                raise ValueError(
                    f"Unsupported response_pt2_mode={pt2_mode!r}. "
                    "Expected 'approx' or 'strict'."
                )
        else:
            pt2_input = None
            pt2_basis = None
        coefficients = self.channel_coefficients(
            params,
            point_features,
            semilocal_energy_density=semilocal_total,
            hf_energy_density=hf_input,
            pt2_energy_density=pt2_input,
            hf_spin_energy_density=hf_spin_inputs,
        )
        basis = self._assemble_basis_channels(
            semilocal_local_channels,
            hf_projected=hf_basis,
            pt2_projected=pt2_basis,
        )
        if coefficients.shape[-1] != basis.shape[-1]:
            raise ValueError(
                "Model output_dim must match basis channels "
                f"(got {coefficients.shape[-1]}, expected {basis.shape[-1]})."
            )
        channels = self._assemble_channel_contributions(coefficients, basis)
        return jnp.sum(channels, axis=-1)

    def _hf_channel_point_energy_from_response_variables(
        self,
        params: PyTree,
        variables: Array,
        *,
        pt2_point: Array | None = None,
    ) -> Array:
        response_floor = self._effective_response_density_floor()
        rho_point = jnp.maximum(variables[0], response_floor)
        grad_point = variables[1:4]
        tau_point = jnp.maximum(variables[4], 0.0)
        point_features = restricted_feature_bundle_from_response_variables(
            rho_point,
            grad_point,
            tau_point,
            density_floor=response_floor,
        )
        semilocal_channels = self.semilocal_energy_density_channels(point_features)
        semilocal_total = jnp.sum(semilocal_channels, axis=-1)

        offset = 5
        if self.input_feature_mode == "canonical":
            n_hfx = max(int(self.hfx_channels), 1)
            hfx_a = variables[offset : offset + n_hfx]
            hfx_b = variables[offset + n_hfx : offset + 2 * n_hfx]
            hf_total = hfx_a[0] + hfx_b[0]
            hf_spin_inputs: tuple[Array, Array] | None = (hfx_a, hfx_b)
        elif self.hf_input_mode == "total_only":
            hf_total = variables[offset]
            hf_spin_inputs = None
        elif self.hf_input_mode == "spin_resolved":
            hfx_a = variables[offset]
            hfx_b = variables[offset + 1]
            hf_total = hfx_a + hfx_b
            hf_spin_inputs = (hfx_a, hfx_b)
        else:
            raise ValueError(
                f"Unsupported hf_input_mode={self.hf_input_mode!r}. "
                "Expected 'total_only' or 'spin_resolved'."
            )

        pt2_input = None
        if self.include_pt2_channel:
            if pt2_point is None:
                pt2_input = jnp.zeros_like(hf_total)
            else:
                pt2_input = jax.lax.stop_gradient(pt2_point)
        coefficients = self.channel_coefficients(
            params,
            point_features,
            semilocal_energy_density=semilocal_total,
            hf_energy_density=hf_total,
            pt2_energy_density=pt2_input,
            hf_spin_energy_density=hf_spin_inputs,
        )
        return self._local_hf_fraction_from_coefficients(coefficients) * hf_total

    def _strict_hf_nonlocal_response_matrices(
        self,
        params: PyTree,
        molecule: Any,
        features: RestrictedFeatureBundle,
        total_gradient: Array,
        hf_projected: Array,
        *,
        hf_spin_energy_density: tuple[Array, Array],
        pt2_projected: Array | None = None,
        occupation_tolerance: float = 1e-8,
    ) -> tuple[Array, Array]:
        response_floor = self._effective_response_density_floor()
        rho0, _, _, response_variables = self._response_variables(
            features,
            total_gradient,
        )
        active = rho0 > response_floor
        hfx_a_raw, hfx_b_raw = hf_spin_energy_density
        hfx_a_raw = jnp.asarray(hfx_a_raw)
        hfx_b_raw = jnp.asarray(hfx_b_raw)
        if hfx_a_raw.ndim == features.rho.ndim:
            hfx_a = hfx_a_raw[:, None]
        else:
            hfx_a = hfx_a_raw
        if hfx_b_raw.ndim == features.rho.ndim:
            hfx_b = hfx_b_raw[:, None]
        else:
            hfx_b = hfx_b_raw

        if self.input_feature_mode == "canonical":
            n_hfx = max(int(self.hfx_channels), 1)
            if hfx_a.shape[-1] < n_hfx or hfx_b.shape[-1] < n_hfx:
                raise ValueError(
                    "Strict HF response requires canonical HFX feature channels "
                    f"with at least {n_hfx} omega values."
                )
            hfx_a_vars = hfx_a[:, :n_hfx]
            hfx_b_vars = hfx_b[:, :n_hfx]
            point_variables = jnp.concatenate(
                [response_variables, hfx_a_vars, hfx_b_vars],
                axis=-1,
            )
            hvar_kind = "canonical"
        elif self.hf_input_mode == "total_only":
            point_variables = jnp.concatenate(
                [response_variables, hf_projected[:, None]],
                axis=-1,
            )
            hvar_kind = "total_only"
            n_hfx = 1
        else:
            point_variables = jnp.concatenate(
                [response_variables, hfx_a[:, :1], hfx_b[:, :1]],
                axis=-1,
            )
            hvar_kind = "spin_resolved"
            n_hfx = 1

        nu_cache = getattr(molecule, "hfx_nu", None)
        if nu_cache is None:
            raise AttributeError("Strict HF response requires molecule.hfx_nu.")
        nu = jnp.asarray(nu_cache, dtype=point_variables.dtype)
        if nu.ndim != 4:
            raise ValueError(
                "molecule.hfx_nu must have shape (n_omega, ngrids, nao, nao), "
                f"got {nu.shape}."
            )
        if nu.shape[0] < n_hfx:
            raise ValueError(
                "molecule.hfx_nu omega axis is shorter than the HFX response "
                f"feature count ({nu.shape[0]} vs {n_hfx})."
            )
        nu = nu[:n_hfx]

        pt2_values = (
            jnp.zeros_like(hf_projected)
            if pt2_projected is None
            else jnp.asarray(pt2_projected, dtype=point_variables.dtype)
        )
        point_grad_fn = jax.grad(
            self._hf_channel_point_energy_from_response_variables,
            argnums=1,
        )
        point_hessian_fn = jax.hessian(
            self._hf_channel_point_energy_from_response_variables,
            argnums=1,
        )

        def point_grad_hessian(variables: Array, pt2_point: Array) -> tuple[Array, Array]:
            grad = point_grad_fn(params, variables, pt2_point=pt2_point)
            hessian = point_hessian_fn(params, variables, pt2_point=pt2_point)
            return grad, hessian

        gradients, hessians = jax.vmap(point_grad_hessian)(point_variables, pt2_values)
        gradients = jnp.nan_to_num(gradients, nan=0.0, posinf=0.0, neginf=0.0)
        hessians = jnp.nan_to_num(hessians, nan=0.0, posinf=0.0, neginf=0.0)
        gradients = self._maybe_clip_response(gradients)
        hessians = self._maybe_clip_response(hessians)
        gradients = gradients * active[:, None].astype(gradients.dtype)
        hessians = hessians * active[:, None, None].astype(hessians.dtype)

        mo_coeff = jnp.asarray(molecule.mo_coeff, dtype=point_variables.dtype)
        mo_occ = jnp.asarray(molecule.mo_occ)
        if mo_coeff.ndim == 3:
            mo_coeff = mo_coeff[0]
            mo_occ = mo_occ[0]
        nocc = getattr(molecule, "nocc", None)
        if nocc is None:
            nocc = int(jnp.count_nonzero(mo_occ > occupation_tolerance))
        else:
            nocc = int(nocc)
        nvir = int(mo_coeff.shape[1]) - nocc
        orbo = mo_coeff[:, :nocc]
        orbv = mo_coeff[:, nocc:]
        ao = jnp.asarray(molecule.ao, dtype=point_variables.dtype)
        rho_o = jnp.einsum("gp,pi->gi", ao, orbo, precision=Precision.HIGHEST)
        rho_v = jnp.einsum("gp,pa->ga", ao, orbv, precision=Precision.HIGHEST)

        dm_spin = self._restricted_spin_density_blocks(molecule)
        dm_spin = jnp.asarray(dm_spin, dtype=point_variables.dtype)
        e_spin = jnp.einsum("gp,spq->sgq", ao, dm_spin, precision=Precision.HIGHEST)
        v_nu_e = jnp.einsum(
            "pa,wgpq,sgq->swga",
            orbv,
            nu,
            e_spin,
            precision=Precision.HIGHEST,
        )
        hprime_spin = -0.5 * jnp.einsum(
            "gi,swga->swgia",
            rho_o,
            v_nu_e,
            precision=Precision.HIGHEST,
        )

        if hvar_kind == "canonical":
            hprime_vars = jnp.concatenate([hprime_spin[0], hprime_spin[1]], axis=0)
            grad_h = gradients[:, 5 : 5 + 2 * n_hfx]
        elif hvar_kind == "total_only":
            hprime_vars = (hprime_spin[0, 0] + hprime_spin[1, 0])[None, ...]
            grad_h = gradients[:, 5:6]
        else:
            hprime_vars = jnp.stack([hprime_spin[0, 0], hprime_spin[1, 0]], axis=0)
            grad_h = gradients[:, 5:7]

        semilocal_response_features = restricted_transition_response_features(
            molecule,
            feature_kind="MGGA",
            occupation_tolerance=occupation_tolerance,
        )
        response_features = jnp.concatenate(
            [semilocal_response_features, hprime_vars],
            axis=0,
        )
        weighted_hessian = (
            hessians.transpose(1, 2, 0)
            * jnp.asarray(molecule.grid.weights, dtype=hessians.dtype)[None, None, :]
        )
        common_matrix = 2.0 * jnp.einsum(
            "xyr,xria,yrjb->iajb",
            weighted_hessian,
            response_features,
            response_features,
            precision=Precision.HIGHEST,
        )

        nu_vv = jnp.einsum(
            "pa,wgpq,qb->wgab",
            orbv,
            nu,
            orbv,
            precision=Precision.HIGHEST,
        )
        nu_vo = jnp.einsum(
            "pa,wgpq,qj->wgaj",
            orbv,
            nu,
            orbo,
            precision=Precision.HIGHEST,
        )
        weights = jnp.asarray(molecule.grid.weights, dtype=point_variables.dtype)

        def second_matrix(
            grad_values: Array,
            omega_index: int,
            spin_weight: float,
        ) -> tuple[Array, Array]:
            weighted_grad = weights * grad_values * spin_weight
            matrix_a = -jnp.einsum(
                "g,gi,gj,gab->iajb",
                weighted_grad,
                rho_o,
                rho_o,
                nu_vv[omega_index],
                precision=Precision.HIGHEST,
            )
            matrix_b = -jnp.einsum(
                "g,gi,gb,gaj->iajb",
                weighted_grad,
                rho_o,
                rho_v,
                nu_vo[omega_index],
                precision=Precision.HIGHEST,
            )
            return matrix_a, matrix_b

        second_a = jnp.zeros((nocc, nvir, nocc, nvir), dtype=point_variables.dtype)
        second_b = jnp.zeros_like(second_a)
        if hvar_kind == "canonical":
            for idx in range(n_hfx):
                matrix_a, matrix_b = second_matrix(grad_h[:, idx], idx, 0.5)
                second_a = second_a + matrix_a
                second_b = second_b + matrix_b
            for idx in range(n_hfx):
                matrix_a, matrix_b = second_matrix(grad_h[:, n_hfx + idx], idx, 0.5)
                second_a = second_a + matrix_a
                second_b = second_b + matrix_b
        elif hvar_kind == "total_only":
            matrix_a, matrix_b = second_matrix(grad_h[:, 0], 0, 1.0)
            second_a = second_a + matrix_a
            second_b = second_b + matrix_b
        else:
            matrix_a, matrix_b = second_matrix(grad_h[:, 0], 0, 0.5)
            second_a = second_a + matrix_a
            second_b = second_b + matrix_b
            matrix_a, matrix_b = second_matrix(grad_h[:, 1], 0, 0.5)
            second_a = second_a + matrix_a
            second_b = second_b + matrix_b

        matrix_a = common_matrix + second_a
        matrix_b = common_matrix + second_b
        matrix_a = jnp.nan_to_num(matrix_a, nan=0.0, posinf=0.0, neginf=0.0)
        matrix_b = jnp.nan_to_num(matrix_b, nan=0.0, posinf=0.0, neginf=0.0)
        return (
            matrix_a.reshape(int(nocc * nvir), int(nocc * nvir)),
            matrix_b.reshape(int(nocc * nvir), int(nocc * nvir)),
        )

    def _strict_total_potential_components(
        self,
        params: PyTree,
        features: RestrictedFeatureBundle,
        total_gradient: Array,
        hf_projected: Array,
        *,
        pt2_projected: Array | None = None,
        hf_spin_energy_density: tuple[Array, Array] | None = None,
        response_hf_mode: Literal["approx", "strict"] | None = None,
        response_pt2_mode: Literal["approx", "strict"] | None = None,
        strict_payload: tuple[Array, Array, Array, Array, Array] | None = None,
    ) -> tuple[Array, Array, Array, Array]:
        if strict_payload is None:
            strict_payload = self._strict_response_payload(
                features,
                total_gradient,
                hf_projected,
                pt2_projected=pt2_projected,
                hf_spin_energy_density=hf_spin_energy_density,
            )
        response_variables, active, hf_feature_a, hf_feature_b, pt2_feature = strict_payload
        point_gradient_fn = jax.grad(
            self._total_point_local_energy_from_variables,
            argnums=1,
        )

        def point_gradients(
            variables: Array,
            hf_point: Array,
            hf_point_a: Array,
            hf_point_b: Array,
            pt2_point: Array,
        ) -> Array:
            return point_gradient_fn(
                params,
                variables,
                hf_point,
                hf_point_a,
                hf_point_b,
                pt2_point=pt2_point,
                response_hf_mode=response_hf_mode,
                response_pt2_mode=response_pt2_mode,
            )

        gradients = jax.vmap(point_gradients)(
            response_variables,
            hf_projected,
            hf_feature_a,
            hf_feature_b,
            pt2_feature,
        )
        gradients = jnp.nan_to_num(gradients, nan=0.0, posinf=0.0, neginf=0.0)
        gradients = self._maybe_clip_response(gradients)
        v_rho = jnp.where(active, gradients[:, 0], 0.0)
        v_grad = jnp.where(active[:, None], gradients[:, 1:4], 0.0)
        v_tau = jnp.where(active, gradients[:, 4], 0.0)
        v_lapl = jnp.zeros_like(v_rho)
        return v_rho, v_grad, v_tau, v_lapl

    def _projected_semilocal_kernel(
        self,
        features: RestrictedFeatureBundle,
    ) -> Array:
        rho0, _, _, response_variables = self._response_variables(features)
        point_hessian = jax.vmap(jax.hessian(self._semilocal_point_local_energy_from_variables))(
            response_variables
        )
        kernel = point_hessian[:, 0, 0]
        kernel = jnp.nan_to_num(kernel, nan=0.0, posinf=0.0, neginf=0.0)
        kernel = self._maybe_clip_response(kernel)
        return jnp.where(rho0 <= self._effective_response_density_floor(), 0.0, kernel)

    def _projected_total_potential_kernel(
        self,
        params: PyTree,
        features: RestrictedFeatureBundle,
        hf_projected: Array,
        molecule: Any | None = None,
        *,
        pt2_projected: Array | None = None,
        total_gradient: Array | None = None,
        hf_spin_energy_density: tuple[Array, Array] | None = None,
        response_hf_mode: Literal["approx", "strict"] | None = None,
        response_pt2_mode: Literal["approx", "strict"] | None = None,
    ) -> tuple[Array, Array]:
        grad = (
            self._default_total_gradient_from_features(features)
            if total_gradient is None
            else jnp.asarray(total_gradient)
        )
        strict_payload = self._strict_response_payload(
            features,
            grad,
            hf_projected,
            pt2_projected=pt2_projected,
            hf_spin_energy_density=hf_spin_energy_density,
        )
        potential, _, _, _ = self._strict_total_potential_components(
            params,
            features,
            grad,
            hf_projected,
            pt2_projected=pt2_projected,
            hf_spin_energy_density=hf_spin_energy_density,
            response_hf_mode=response_hf_mode,
            response_pt2_mode=response_pt2_mode,
            strict_payload=strict_payload,
        )
        tensor = self._strict_total_response_tensor(
            params,
            features,
            grad,
            hf_projected,
            pt2_projected=pt2_projected,
            hf_spin_energy_density=(
                (hf_projected, hf_projected)
                if hf_spin_energy_density is None
                else hf_spin_energy_density
            ),
            response_hf_mode=response_hf_mode,
            response_pt2_mode=response_pt2_mode,
            strict_payload=strict_payload,
        )
        kernel = tensor[0, 0]
        return potential, kernel

    def _strict_total_response_tensor(
        self,
        params: PyTree,
        features: RestrictedFeatureBundle,
        total_gradient: Array,
        hf_projected: Array,
        *,
        pt2_projected: Array | None = None,
        hf_spin_energy_density: tuple[Array, Array],
        response_hf_mode: Literal["approx", "strict"] | None = None,
        response_pt2_mode: Literal["approx", "strict"] | None = None,
        strict_payload: tuple[Array, Array, Array, Array, Array] | None = None,
    ) -> Array:
        """Return the strict restricted semilocal response tensor on the grid.

        The tensor follows the PySCF reduced MGGA convention with local variables
        ``[rho, d_x rho, d_y rho, d_z rho, tau]``.
        """

        if strict_payload is None:
            strict_payload = self._strict_response_payload(
                features,
                total_gradient,
                hf_projected,
                pt2_projected=pt2_projected,
                hf_spin_energy_density=hf_spin_energy_density,
            )
        response_variables, active, hf_projected_a, hf_projected_b, pt2_feature = strict_payload
        point_hessian_fn = jax.hessian(
            self._total_point_local_energy_from_variables,
            argnums=1,
        )

        def point_tensor(
            variables: Array,
            hf_point: Array,
            hf_point_a: Array,
            hf_point_b: Array,
            pt2_point: Array,
        ) -> Array:
            tensor = point_hessian_fn(
                params,
                variables,
                hf_point,
                hf_point_a,
                hf_point_b,
                pt2_point=pt2_point,
                response_hf_mode=response_hf_mode,
                response_pt2_mode=response_pt2_mode,
            )
            tensor = jnp.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
            tensor = self._maybe_clip_response(tensor)
            return tensor

        tensor = jax.vmap(point_tensor)(
            response_variables,
            hf_projected,
            hf_projected_a,
            hf_projected_b,
            pt2_feature,
        )
        tensor = tensor * active[:, None, None].astype(tensor.dtype)
        return jnp.asarray(tensor).transpose(1, 2, 0)

@dataclass(frozen=True)
class NeuralXCModel(
    HeadMixingMixin,
    NeuralXCProjectionMixin,
    AssemblyMixin,
    ResponseMixin,
    NeuralXCBindingMixin,
):
    r"""Structured neural XC runtime model."""

    model: nn.Module
    non_hf_module: SemilocalEnergyDensityModule | None = None
    semilocal_xc: str | Sequence[str] = DEFAULT_NEURAL_XC_SEMILOCAL_XC
    semilocal_energy_density_fn: SemilocalEnergyDensityFn | None = None
    input_feature_mode: Literal["enhanced", "canonical"] = DEFAULT_INPUT_FEATURE_MODE
    hf_input_mode: Literal["total_only", "spin_resolved"] = "spin_resolved"
    include_pt2_channel: bool = False
    pt2_channel_mode: Literal["scaled_projected", "local_exact"] = "scaled_projected"
    response_hf_mode: Literal["approx", "strict"] = "strict"
    response_pt2_mode: Literal["approx", "strict"] = "approx"
    strict_feature_alignment: bool = True
    allow_experimental_jax_xc: bool = False
    density_floor: float = 1e-12
    response_density_floor: float | None = None
    kernel_clip: float = 5.0
    response_kernel_clip: float | None = 5.0
    name: str = "neural_xc"
    hfx_channels: int = 2
    is_xc: bool = True

    def _mlp_functional(self) -> NeuralXCCore:
        return NeuralXCCore(
            model=self.model,
            coefficient_transform_fn=self._sanitize_coefficients,
            name=self.name,
        )

    def _effective_response_density_floor(self) -> float:
        response_floor = self.density_floor
        if self.response_density_floor is not None:
            response_floor = max(response_floor, float(self.response_density_floor))
        return response_floor

    def resolved_non_hf_module(self) -> SemilocalEnergyDensityModule:
        if self.non_hf_module is not None:
            return self.non_hf_module
        return _legacy_semilocal_module(
            self.semilocal_xc,
            self.semilocal_energy_density_fn,
            allow_experimental_jax_xc=self.allow_experimental_jax_xc,
        )

    def _maybe_clip_response(self, values: Array) -> Array:
        clip = self.response_kernel_clip
        if clip is None:
            return values
        clip_value = float(clip)
        if clip_value <= 0.0:
            return values
        return jnp.clip(values, -clip_value, clip_value)


NeuralXCFunctional = NeuralXCModel
NeuralXCHybridFunctional = NeuralXCModel
