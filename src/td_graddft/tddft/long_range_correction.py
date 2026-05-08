from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import freeze, unfreeze
from jax.lax import Precision
from jaxtyping import Array, PRNGKeyArray, PyTree

from ..features import restricted_grid_features_with_gradients
from ._utils import _density_on_grid, _restricted_orbital_data, _transition_densities_on_grid


def _extract_named_subtree(params: Any, key: str) -> tuple[Any | None, bool]:
    if isinstance(params, Mapping):
        if key in params:
            return params[key], True
        params_collection = params.get("params")
        if isinstance(params_collection, Mapping) and key in params_collection:
            return params_collection[key], True
    return None, False


def _extract_lr_params(params: PyTree) -> PyTree:
    subtree, found = _extract_named_subtree(params, "lr_correction")
    if not found:
        subtree = params
    if isinstance(subtree, Mapping) and "params" in subtree:
        return subtree
    return {"params": subtree}


def _extract_base_params(params: PyTree) -> PyTree | None:
    subtree, found = _extract_named_subtree(params, "base")
    if not found:
        return None
    return subtree


def _bind_base_functional(
    base_functional: Any | None,
    base_params: PyTree | None,
    molecule: Any,
    *,
    prefer_scf: bool = False,
) -> Any | None:
    if base_functional is None:
        return None

    if base_params is not None:
        if prefer_scf:
            scf_binder = getattr(base_functional, "bind_to_molecule_for_scf", None)
            if scf_binder is not None:
                return scf_binder(base_params, molecule)
        binder = getattr(base_functional, "bind_to_molecule", None)
        if binder is not None:
            return binder(base_params, molecule)
        generic_binder = getattr(base_functional, "bind", None)
        if generic_binder is not None:
            return generic_binder(base_params)

    if prefer_scf:
        scf_binder = getattr(base_functional, "bind_to_molecule_for_scf", None)
        if scf_binder is not None and base_params is None:
            return base_functional

    return base_functional


def _density_and_gradient_norm(
    molecule: Any,
    *,
    density_floor: float,
) -> tuple[Array, Array]:
    density = jnp.maximum(_density_on_grid(molecule), density_floor)
    ao_deriv1 = getattr(molecule, "ao_deriv1", None)
    if ao_deriv1 is None:
        return density, jnp.zeros_like(density)

    _, total_gradient = restricted_grid_features_with_gradients(molecule)
    grad_norm = jnp.linalg.norm(jnp.asarray(total_gradient), axis=-1)
    return density, grad_norm


def _signed_log1p_feature(values: Array, *, scale: float = 1.0, floor: float = 1e-12) -> Array:
    array = jnp.asarray(values)
    safe_scale = jnp.maximum(jnp.asarray(scale, dtype=array.dtype), floor)
    return jnp.sign(array) * jnp.log1p(jnp.abs(array) / safe_scale)


def _bound_base_response_descriptors(
    base_bound: Any | None,
    molecule: Any,
    *,
    density: Array,
    ngrids: int,
    dtype: Any,
) -> Array:
    zeros = jnp.zeros((ngrids,), dtype=dtype)
    potential = zeros
    potential_grad_norm = zeros
    potential_tau = zeros
    kernel = zeros
    energy_density = zeros
    hf_fraction = zeros

    if base_bound is not None:
        grid_potential_components = getattr(base_bound, "grid_potential_components", None)
        if callable(grid_potential_components):
            pot_rho, pot_grad, pot_tau = grid_potential_components(molecule)
            potential = jnp.asarray(pot_rho, dtype=dtype)
            pot_grad_arr = jnp.asarray(pot_grad, dtype=dtype)
            if pot_grad_arr.ndim >= 2:
                potential_grad_norm = jnp.linalg.norm(pot_grad_arr, axis=-1)
            potential_tau = jnp.asarray(pot_tau, dtype=dtype)
        else:
            grid_potential = getattr(base_bound, "grid_potential", None)
            if callable(grid_potential):
                potential = jnp.asarray(grid_potential(molecule), dtype=dtype)

        grid_kernel = getattr(base_bound, "grid_kernel", None)
        if callable(grid_kernel):
            kernel = jnp.asarray(grid_kernel(molecule), dtype=dtype)

        bound_energy_density = getattr(base_bound, "energy_density", None)
        if callable(bound_energy_density):
            energy_density = jnp.asarray(bound_energy_density(density), dtype=dtype)

        grid_hf_fraction = getattr(base_bound, "grid_hf_fraction", None)
        if callable(grid_hf_fraction):
            hf_fraction = jnp.asarray(grid_hf_fraction(molecule), dtype=dtype)
        else:
            exact_exchange_fraction = getattr(base_bound, "exact_exchange_fraction", 0.0)
            hf_fraction = jnp.full((ngrids,), exact_exchange_fraction, dtype=dtype)

    return jnp.stack(
        [
            _signed_log1p_feature(potential),
            _signed_log1p_feature(potential_grad_norm),
            _signed_log1p_feature(potential_tau),
            _signed_log1p_feature(kernel),
            _signed_log1p_feature(energy_density),
            jnp.clip(hf_fraction, 0.0, 1.0),
        ],
        axis=-1,
    )


def _local_response_descriptors(
    molecule: Any,
    *,
    density_floor: float,
    base_bound: Any | None = None,
) -> Array:
    features, total_gradient = restricted_grid_features_with_gradients(molecule)
    rho = jnp.maximum(jnp.asarray(features.rho), density_floor)
    grad_norm = jnp.linalg.norm(jnp.asarray(total_gradient), axis=-1)
    tau = jnp.maximum(jnp.asarray(features.tau_a) + jnp.asarray(features.tau_b), 0.0)

    descriptor_blocks = [
        jnp.log1p(rho / jnp.maximum(jnp.asarray(density_floor, dtype=rho.dtype), 1e-12))[:, None],
        jnp.log1p(grad_norm)[:, None],
        jnp.log1p(tau)[:, None],
    ]

    hfx_local = getattr(molecule, "hfx_local", None)
    if hfx_local is not None:
        hfx_array = jnp.asarray(hfx_local)
        if hfx_array.ndim == 3:
            hfx_channels = jnp.mean(hfx_array, axis=0)
        elif hfx_array.ndim == 2:
            hfx_channels = hfx_array
        else:
            raise ValueError(
                "molecule.hfx_local must have shape (spin, ngrids, n_omega) or (ngrids, n_omega)."
            )
        descriptor_blocks.append(_signed_log1p_feature(hfx_channels))

    pt2_local = getattr(molecule, "pt2_local", None)
    if pt2_local is not None:
        descriptor_blocks.append(_signed_log1p_feature(jnp.asarray(pt2_local))[:, None])

    descriptor_blocks.append(
        _bound_base_response_descriptors(
            base_bound,
            molecule,
            density=rho,
            ngrids=int(rho.shape[0]),
            dtype=rho.dtype,
        )
    )

    return jnp.concatenate(descriptor_blocks, axis=-1)


def pairwise_grid_distances(coords: Array) -> Array:
    """Return the symmetric pairwise grid-point distance matrix."""

    coords_arr = jnp.asarray(coords)
    if coords_arr.ndim != 2:
        raise ValueError(f"Grid coordinates must have shape (ngrids, ndim), got {coords_arr.shape}.")
    diffs = coords_arr[:, None, :] - coords_arr[None, :, :]
    return jnp.linalg.norm(diffs, axis=-1)


def build_long_range_pair_features(
    molecule: Any,
    *,
    density_floor: float = 1e-8,
    distance_scale: float = 1.0,
    grid_point_indices: Array | None = None,
    base_bound: Any | None = None,
) -> Array:
    """Build symmetric pair features for a real-space long-range response correction."""

    grid = getattr(molecule, "grid", None)
    coords = getattr(grid, "coords", None)
    if coords is None:
        raise AttributeError(
            "Molecule-like object must define grid.coords for long-range response features."
        )

    local_descriptors = _local_response_descriptors(
        molecule,
        density_floor=density_floor,
        base_bound=base_bound,
    )
    coords_arr = jnp.asarray(coords)
    if grid_point_indices is not None:
        indices = jnp.asarray(grid_point_indices, dtype=jnp.int32)
        local_descriptors = local_descriptors[indices]
        coords_arr = coords_arr[indices]
    safe_distance_scale = jnp.maximum(
        jnp.asarray(distance_scale, dtype=local_descriptors.dtype),
        1e-12,
    )
    r12 = pairwise_grid_distances(coords_arr) / safe_distance_scale

    descriptor_mean = 0.5 * (local_descriptors[:, None, :] + local_descriptors[None, :, :])
    descriptor_delta = jnp.abs(local_descriptors[:, None, :] - local_descriptors[None, :, :])
    descriptor_product = local_descriptors[:, None, :] * local_descriptors[None, :, :]

    radial_features = jnp.stack(
        [
            jnp.log1p(r12),
            jnp.reciprocal(jnp.sqrt(1.0 + jnp.square(r12))),
            jnp.exp(-r12),
        ],
        axis=-1,
    )

    return jnp.concatenate(
        [
            descriptor_mean,
            descriptor_delta,
            descriptor_product,
            radial_features,
        ],
        axis=-1,
    )


def compute_long_range_kernel(
    alpha: Array,
    gamma: Array,
    r12: Array,
    *,
    distance_floor: float = 0.35,
    kernel_scale: float = 1.0,
) -> Array:
    """Evaluate a screened soft-Coulomb response kernel from nonnegative heads."""

    alpha_arr = jnp.asarray(alpha)
    gamma_arr = jnp.asarray(gamma)
    if alpha_arr.ndim == 3 and alpha_arr.shape[-1] == 1:
        alpha_arr = alpha_arr[..., 0]
    if gamma_arr.ndim == 3 and gamma_arr.shape[-1] == 1:
        gamma_arr = gamma_arr[..., 0]

    r12_arr = jnp.asarray(r12, dtype=alpha_arr.dtype)
    soft_floor = jnp.maximum(jnp.asarray(distance_floor, dtype=alpha_arr.dtype), 1e-6)
    softened_r12 = jnp.sqrt(jnp.square(r12_arr) + soft_floor * soft_floor)
    kernel = -jnp.asarray(kernel_scale, dtype=alpha_arr.dtype) * alpha_arr
    kernel = kernel * jnp.exp(-gamma_arr * r12_arr) / softened_r12
    return 0.5 * (kernel + kernel.T)


def build_grid_point_mode_features(
    molecule: Any,
    *,
    density_floor: float = 1e-8,
    mode_point_indices: Array | None = None,
) -> Array:
    """Build one feature vector per selected grid-point mode."""

    grid = getattr(molecule, "grid", None)
    coords = getattr(grid, "coords", None)
    weights = getattr(grid, "weights", None)
    if coords is None or weights is None:
        raise AttributeError(
            "Molecule-like object must define grid.coords and grid.weights for grid-point modes."
        )

    density, grad_norm = _density_and_gradient_norm(molecule, density_floor=density_floor)
    coords_arr = jnp.asarray(coords)
    weights_arr = jnp.asarray(weights)
    if mode_point_indices is not None:
        indices = jnp.asarray(mode_point_indices, dtype=jnp.int32)
        density = density[indices]
        grad_norm = grad_norm[indices]
        coords_arr = coords_arr[indices]
        weights_arr = weights_arr[indices]

    safe_density_floor = jnp.maximum(jnp.asarray(density_floor, dtype=density.dtype), 1e-12)
    abs_weights = jnp.abs(weights_arr)
    mean_abs_weight = jnp.maximum(jnp.mean(abs_weights), 1e-12)
    signed_weight_feature = jnp.sign(weights_arr) * jnp.log1p(abs_weights / mean_abs_weight)
    center = jnp.mean(jnp.asarray(coords), axis=0)
    radial = jnp.linalg.norm(coords_arr - center[None, :], axis=-1)

    return jnp.stack(
        [
            jnp.log1p(density / safe_density_floor),
            jnp.log1p(grad_norm),
            signed_weight_feature,
            jnp.log1p(radial),
        ],
        axis=-1,
    )


def build_grid_point_mode_basis(
    molecule: Any,
    *,
    mode_point_indices: Array | None = None,
    mode_width_scale: float = 1.0,
    normalize: bool = True,
) -> Array:
    """Evaluate Gaussian grid-point modes on the full response grid."""

    grid = getattr(molecule, "grid", None)
    coords = getattr(grid, "coords", None)
    weights = getattr(grid, "weights", None)
    if coords is None or weights is None:
        raise AttributeError(
            "Molecule-like object must define grid.coords and grid.weights for grid-point modes."
        )

    coords_arr = jnp.asarray(coords)
    weights_arr = jnp.asarray(weights)
    if mode_point_indices is None:
        anchor_coords = coords_arr
    else:
        anchor_coords = coords_arr[jnp.asarray(mode_point_indices, dtype=jnp.int32)]

    if int(anchor_coords.shape[0]) <= 1:
        width = jnp.asarray(float(mode_width_scale), dtype=coords_arr.dtype)
    else:
        anchor_distances = pairwise_grid_distances(anchor_coords)
        mask = jnp.eye(int(anchor_coords.shape[0]), dtype=anchor_distances.dtype) * 1e6
        nearest = jnp.min(anchor_distances + mask, axis=1)
        width = jnp.maximum(jnp.mean(nearest) * jnp.asarray(mode_width_scale), 1e-3)

    deltas = coords_arr[:, None, :] - anchor_coords[None, :, :]
    sq_distance = jnp.sum(deltas * deltas, axis=-1)
    basis = jnp.exp(-0.5 * sq_distance / jnp.maximum(width * width, 1e-12))
    if normalize:
        normalization_weights = jnp.maximum(jnp.abs(weights_arr), 1e-12)
        norm = jnp.sqrt(jnp.sum(normalization_weights[:, None] * basis * basis, axis=0))
        basis = basis / jnp.maximum(norm[None, :], 1e-12)
    return basis


def _logit_from_fraction(value: float, *, eps: float = 1e-6) -> float:
    clipped = min(max(float(value), float(eps)), 1.0 - float(eps))
    return math.log(clipped) - math.log1p(-clipped)


class LongRangeXCNet(nn.Module):
    """Small symmetric pair-kernel network for response-only fine-tuning."""

    hidden_dims: Sequence[int] = (64, 64, 32)
    activation: Callable[[Array], Array] = nn.swish
    alpha_scale: float = 1.0
    gamma_floor: float = 1e-3

    @nn.compact
    def __call__(self, pair_features: Array) -> tuple[Array, Array]:
        x = jnp.asarray(pair_features)
        for index, width in enumerate(self.hidden_dims):
            x = nn.Dense(int(width), name=f"SharedDense_{index}")(x)
            x = self.activation(x)

        alpha_raw = nn.Dense(1, name="AlphaHead")(x)
        gamma_raw = nn.Dense(1, name="GammaHead")(x)

        alpha = jax.nn.softplus(alpha_raw)
        alpha = alpha * jnp.asarray(self.alpha_scale, dtype=alpha.dtype)
        gamma = jax.nn.softplus(gamma_raw) + jnp.asarray(self.gamma_floor, dtype=alpha.dtype)
        return alpha, gamma


class GridPointModeCouplingNet(nn.Module):
    """Low-rank coupling model for grid-point auxiliary modes."""

    hidden_dims: Sequence[int] = (64, 64)
    latent_dim: int = 16
    activation: Callable[[Array], Array] = nn.swish
    coupling_scale: float = 1.0
    initial_coupling_strength: float = 2e-1
    dense_kernel_init_scale: float = 1e-1
    normalization_eps: float = 1e-6

    @nn.compact
    def __call__(self, mode_features: Array) -> Array:
        x = jnp.asarray(mode_features)
        dense_kernel_init = nn.initializers.normal(stddev=float(self.dense_kernel_init_scale))
        for index, width in enumerate(self.hidden_dims):
            x = nn.Dense(
                int(width),
                name=f"SharedDense_{index}",
                kernel_init=dense_kernel_init,
            )(x)
            x = self.activation(x)

        embedding = nn.Dense(
            int(self.latent_dim),
            name="ModeEmbedding",
            kernel_init=dense_kernel_init,
        )(x)
        embedding = self.activation(embedding)

        rms = jnp.sqrt(
            jnp.mean(embedding * embedding, axis=-1, keepdims=True)
            + jnp.asarray(self.normalization_eps, dtype=embedding.dtype)
        )
        normalized_embedding = embedding / rms
        gram = normalized_embedding @ normalized_embedding.T
        gram = gram / jnp.asarray(max(int(self.latent_dim), 1), dtype=embedding.dtype)

        max_strength = jnp.asarray(float(self.coupling_scale), dtype=embedding.dtype)
        initial_fraction = float(self.initial_coupling_strength) / max(float(self.coupling_scale), 1e-8)
        coupling_logit = self.param(
            "CouplingStrengthLogit",
            lambda key: jnp.asarray(
                _logit_from_fraction(initial_fraction),
                dtype=embedding.dtype,
            ),
        )
        strength = max_strength * jax.nn.sigmoid(coupling_logit)
        coupling = -strength * gram
        return 0.5 * (coupling + coupling.T)


@dataclass(frozen=True)
class _ZeroBoundFunctional:
    name: str = "zero_response_xc"
    exact_exchange_fraction: float = 0.0

    def local_potential(self, density: Array) -> Array:
        return jnp.zeros_like(jnp.asarray(density))

    def local_kernel(self, density: Array) -> Array:
        return jnp.zeros_like(jnp.asarray(density))

    def grid_potential(self, molecule: Any) -> Array:
        return self.local_potential(_density_on_grid(molecule))

    def grid_kernel(self, molecule: Any) -> Array:
        return self.local_kernel(_density_on_grid(molecule))


@dataclass(frozen=True)
class BoundLongRangeCorrectedFunctional:
    """Molecule-bound response wrapper that augments a base XC with a nonlocal action."""

    name: str
    base_bound: Any | None
    pair_kernel: Array
    transition_densities: Array
    weighted_transition_densities: Array
    grid_weights: Array
    exact_exchange_fraction: Array | float = 0.0

    def local_potential(self, density: Array) -> Array:
        if self.base_bound is None:
            return jnp.zeros_like(jnp.asarray(density))
        local_potential = getattr(self.base_bound, "local_potential", None)
        if local_potential is None:
            return jnp.zeros_like(jnp.asarray(density))
        return jnp.asarray(local_potential(density))

    def local_kernel(self, density: Array) -> Array:
        if self.base_bound is None:
            return jnp.zeros_like(jnp.asarray(density))
        local_kernel = getattr(self.base_bound, "local_kernel", None)
        if local_kernel is None:
            return jnp.zeros_like(jnp.asarray(density))
        return jnp.asarray(local_kernel(density))

    def grid_potential(self, molecule: Any) -> Array:
        if self.base_bound is not None:
            grid_potential = getattr(self.base_bound, "grid_potential", None)
            if grid_potential is not None:
                return jnp.asarray(grid_potential(molecule))
        return self.local_potential(_density_on_grid(molecule))

    def grid_kernel(self, molecule: Any) -> Array:
        if self.base_bound is not None:
            grid_kernel = getattr(self.base_bound, "grid_kernel", None)
            if grid_kernel is not None:
                return jnp.asarray(grid_kernel(molecule))
        return self.local_kernel(_density_on_grid(molecule))

    def nonlocal_response_action(
        self,
        molecule: Any,
        amplitudes: Array,
        *,
        occupation_tolerance: float = 1e-8,
    ) -> Array:
        del molecule, occupation_tolerance
        amplitude_array = jnp.asarray(amplitudes)
        expected_shape = self.transition_densities.shape[1:]
        if amplitude_array.ndim < 2 or amplitude_array.shape[-2:] != expected_shape:
            raise ValueError(
                "Long-range response action expects amplitudes with shape "
                f"(..., {expected_shape[0]}, {expected_shape[1]}), got {amplitude_array.shape}."
            )

        flat_amplitudes = amplitude_array.reshape((-1,) + expected_shape)
        transition_density = jnp.einsum(
            "ria,bia->br",
            self.transition_densities,
            flat_amplitudes,
            precision=Precision.HIGHEST,
        )
        screened_density = jnp.einsum(
            "rs,bs->br",
            self.pair_kernel,
            self.grid_weights[None, :] * transition_density,
            precision=Precision.HIGHEST,
        )
        response = 2.0 * jnp.einsum(
            "ria,br->bia",
            self.weighted_transition_densities,
            screened_density,
            precision=Precision.HIGHEST,
        )
        return response.reshape(amplitude_array.shape)

    def nonlocal_response_diagonal(
        self,
        molecule: Any,
        *,
        occupation_tolerance: float = 1e-8,
    ) -> Array:
        del molecule, occupation_tolerance
        ngrids = int(self.pair_kernel.shape[0])
        weighted_flat = self.weighted_transition_densities.reshape(ngrids, -1)
        screened = jnp.einsum(
            "rs,sd->rd",
            self.pair_kernel,
            weighted_flat,
            precision=Precision.HIGHEST,
        )
        diagonal = 2.0 * jnp.sum(weighted_flat * screened, axis=0)
        return diagonal.reshape(self.transition_densities.shape[1:])

    def __getattr__(self, name: str) -> Any:
        if self.base_bound is None:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")
        return getattr(self.base_bound, name)


@dataclass(frozen=True)
class BoundGridPointModeLongRangeCorrectedFunctional:
    """Bound XC wrapper whose nonlocal correction acts through low-rank grid modes."""

    name: str
    base_bound: Any | None
    mode_projections: Array
    coupling_matrix: Array
    exact_exchange_fraction: Array | float = 0.0

    def local_potential(self, density: Array) -> Array:
        if self.base_bound is None:
            return jnp.zeros_like(jnp.asarray(density))
        local_potential = getattr(self.base_bound, "local_potential", None)
        if local_potential is None:
            return jnp.zeros_like(jnp.asarray(density))
        return jnp.asarray(local_potential(density))

    def local_kernel(self, density: Array) -> Array:
        if self.base_bound is None:
            return jnp.zeros_like(jnp.asarray(density))
        local_kernel = getattr(self.base_bound, "local_kernel", None)
        if local_kernel is None:
            return jnp.zeros_like(jnp.asarray(density))
        return jnp.asarray(local_kernel(density))

    def grid_potential(self, molecule: Any) -> Array:
        if self.base_bound is not None:
            grid_potential = getattr(self.base_bound, "grid_potential", None)
            if grid_potential is not None:
                return jnp.asarray(grid_potential(molecule))
        return self.local_potential(_density_on_grid(molecule))

    def grid_kernel(self, molecule: Any) -> Array:
        if self.base_bound is not None:
            grid_kernel = getattr(self.base_bound, "grid_kernel", None)
            if grid_kernel is not None:
                return jnp.asarray(grid_kernel(molecule))
        return self.local_kernel(_density_on_grid(molecule))

    def nonlocal_response_action(
        self,
        molecule: Any,
        amplitudes: Array,
        *,
        occupation_tolerance: float = 1e-8,
    ) -> Array:
        del molecule, occupation_tolerance
        amplitude_array = jnp.asarray(amplitudes)
        expected_shape = self.mode_projections.shape[1:]
        if amplitude_array.ndim < 2 or amplitude_array.shape[-2:] != expected_shape:
            raise ValueError(
                "Grid-point mode response action expects amplitudes with shape "
                f"(..., {expected_shape[0]}, {expected_shape[1]}), got {amplitude_array.shape}."
            )
        flat_amplitudes = amplitude_array.reshape((-1,) + expected_shape)
        mode_amplitudes = jnp.einsum(
            "pia,bia->bp",
            self.mode_projections,
            flat_amplitudes,
            precision=Precision.HIGHEST,
        )
        mixed = jnp.einsum(
            "pq,bq->bp",
            self.coupling_matrix,
            mode_amplitudes,
            precision=Precision.HIGHEST,
        )
        response = 2.0 * jnp.einsum(
            "pia,bp->bia",
            self.mode_projections,
            mixed,
            precision=Precision.HIGHEST,
        )
        return response.reshape(amplitude_array.shape)

    def nonlocal_response_diagonal(
        self,
        molecule: Any,
        *,
        occupation_tolerance: float = 1e-8,
    ) -> Array:
        del molecule, occupation_tolerance
        return 2.0 * jnp.einsum(
            "pia,pq,qia->ia",
            self.mode_projections,
            self.coupling_matrix,
            self.mode_projections,
            precision=Precision.HIGHEST,
        )

    def __getattr__(self, name: str) -> Any:
        if self.base_bound is None:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")
        return getattr(self.base_bound, name)


@dataclass(frozen=True)
class LongRangeCorrectedFunctional:
    """Response-only wrapper that keeps the base ground-state XC fixed."""

    base_functional: Any | None = None
    model: nn.Module = field(default_factory=LongRangeXCNet)
    name: str = "long_range_corrected_xc"
    density_floor: float = 1e-8
    distance_floor: float = 0.35
    distance_scale: float = 1.0
    kernel_scale: float = 1.0
    occupation_tolerance: float = 1e-8
    max_pair_points: int | None = None

    def grid_point_indices(self, molecule: Any) -> Array | None:
        max_pair_points = self.max_pair_points
        if max_pair_points is None:
            return None
        ngrid = int(jnp.asarray(molecule.grid.coords).shape[0])
        if int(max_pair_points) <= 0 or ngrid <= int(max_pair_points):
            return None
        stride = max((ngrid + int(max_pair_points) - 1) // int(max_pair_points), 1)
        return jnp.arange(0, ngrid, stride, dtype=jnp.int32)[: int(max_pair_points)]

    def pair_features(self, molecule: Any, *, base_bound: Any | None = None) -> Array:
        return build_long_range_pair_features(
            molecule,
            density_floor=self.density_floor,
            distance_scale=self.distance_scale,
            grid_point_indices=self.grid_point_indices(molecule),
            base_bound=base_bound,
        )

    def init(self, rng: PRNGKeyArray, molecule: Any) -> PyTree:
        return self.model.init(rng, self.pair_features(molecule))

    def init_from_molecule(self, rng: PRNGKeyArray, molecule: Any) -> PyTree:
        return self.init(rng, molecule)

    def combine_params(self, base_params: PyTree | None, lr_params: PyTree) -> dict[str, Any]:
        combined: dict[str, Any] = {
            "lr_correction": unfreeze(lr_params) if isinstance(lr_params, Mapping) else lr_params
        }
        if base_params is not None:
            combined["base"] = (
                unfreeze(base_params) if isinstance(base_params, Mapping) else base_params
            )
        return freeze(combined)

    def bind_to_molecule(self, params: PyTree, molecule: Any) -> BoundLongRangeCorrectedFunctional:
        base_params = _extract_base_params(params)
        base_bound = _bind_base_functional(
            self.base_functional,
            base_params,
            molecule,
        )
        lr_params = _extract_lr_params(params)
        grid_indices = self.grid_point_indices(molecule)
        pair_features = self.pair_features(molecule, base_bound=base_bound)
        coords = jnp.asarray(molecule.grid.coords)
        distances = pairwise_grid_distances(coords if grid_indices is None else coords[grid_indices])
        alpha, gamma = self.model.apply(lr_params, pair_features)
        pair_kernel = compute_long_range_kernel(
            alpha,
            gamma,
            distances,
            distance_floor=self.distance_floor,
            kernel_scale=self.kernel_scale,
        )

        ao = jnp.asarray(molecule.ao)
        full_weights = jnp.asarray(molecule.grid.weights)
        weights = full_weights
        if grid_indices is not None:
            ao = ao[grid_indices]
            weights = full_weights[grid_indices]
            weights = weights * (
                jnp.sum(jnp.abs(full_weights)) / jnp.maximum(jnp.sum(jnp.abs(weights)), 1e-12)
            )
        orbo, orbv, _, _ = _restricted_orbital_data(
            molecule,
            self.occupation_tolerance,
        )
        transition_densities = _transition_densities_on_grid(ao, orbo, orbv)
        weighted_transition_densities = transition_densities * weights[:, None, None]
        exact_exchange_fraction = (
            getattr(base_bound, "exact_exchange_fraction", 0.0) if base_bound is not None else 0.0
        )
        return BoundLongRangeCorrectedFunctional(
            name=self.name,
            base_bound=base_bound,
            pair_kernel=pair_kernel,
            transition_densities=transition_densities,
            weighted_transition_densities=weighted_transition_densities,
            grid_weights=weights,
            exact_exchange_fraction=exact_exchange_fraction,
        )

    def bind_to_molecule_for_scf(self, params: PyTree, molecule: Any) -> Any:
        base_params = _extract_base_params(params)
        base_bound = _bind_base_functional(
            self.base_functional,
            base_params,
            molecule,
            prefer_scf=True,
        )
        if base_bound is None:
            return _ZeroBoundFunctional(name=self.name)
        return base_bound

    def energy_from_molecule(self, params: PyTree, molecule: Any) -> Array:
        if self.base_functional is None:
            return jnp.asarray(0.0, dtype=jnp.asarray(molecule.grid.weights).dtype)

        base_params = _extract_base_params(params)
        energy_from_molecule = getattr(self.base_functional, "energy_from_molecule", None)
        if energy_from_molecule is not None:
            if base_params is None:
                return jnp.asarray(energy_from_molecule(molecule))
            return jnp.asarray(energy_from_molecule(base_params, molecule))

        total_density = _density_on_grid(molecule)
        energy = getattr(self.base_functional, "energy", None)
        if energy is None:
            raise AttributeError(
                "Base functional must expose energy_from_molecule(...) or energy(...)."
            )
        if base_params is None:
            return jnp.asarray(energy(total_density, molecule.grid.weights))
        return jnp.asarray(energy(base_params, total_density, molecule.grid.weights))


@dataclass(frozen=True)
class GridPointModeLongRangeCorrectedFunctional:
    """Response-only wrapper using low-rank grid-point auxiliary modes."""

    base_functional: Any | None = None
    model: nn.Module = field(default_factory=GridPointModeCouplingNet)
    name: str = "grid_point_mode_long_range_corrected_xc"
    density_floor: float = 1e-8
    mode_width_scale: float = 1.0
    kernel_scale: float = 1.0
    occupation_tolerance: float = 1e-8
    max_mode_points: int | None = None

    def mode_point_indices(self, molecule: Any) -> Array | None:
        max_mode_points = self.max_mode_points
        if max_mode_points is None:
            return None
        ngrid = int(jnp.asarray(molecule.grid.coords).shape[0])
        if int(max_mode_points) <= 0 or ngrid <= int(max_mode_points):
            return None
        stride = max((ngrid + int(max_mode_points) - 1) // int(max_mode_points), 1)
        return jnp.arange(0, ngrid, stride, dtype=jnp.int32)[: int(max_mode_points)]

    def mode_features(self, molecule: Any) -> Array:
        return build_grid_point_mode_features(
            molecule,
            density_floor=self.density_floor,
            mode_point_indices=self.mode_point_indices(molecule),
        )

    def init(self, rng: PRNGKeyArray, molecule: Any) -> PyTree:
        return self.model.init(rng, self.mode_features(molecule))

    def init_from_molecule(self, rng: PRNGKeyArray, molecule: Any) -> PyTree:
        return self.init(rng, molecule)

    def combine_params(self, base_params: PyTree | None, lr_params: PyTree) -> dict[str, Any]:
        combined: dict[str, Any] = {
            "lr_correction": unfreeze(lr_params) if isinstance(lr_params, Mapping) else lr_params
        }
        if base_params is not None:
            combined["base"] = (
                unfreeze(base_params) if isinstance(base_params, Mapping) else base_params
            )
        return freeze(combined)

    def bind_to_molecule(
        self,
        params: PyTree,
        molecule: Any,
    ) -> BoundGridPointModeLongRangeCorrectedFunctional:
        base_params = _extract_base_params(params)
        base_bound = _bind_base_functional(
            self.base_functional,
            base_params,
            molecule,
        )
        lr_params = _extract_lr_params(params)
        mode_indices = self.mode_point_indices(molecule)
        mode_features = self.mode_features(molecule)
        mode_basis = build_grid_point_mode_basis(
            molecule,
            mode_point_indices=mode_indices,
            mode_width_scale=self.mode_width_scale,
        )
        ao = jnp.asarray(molecule.ao)
        weights = jnp.asarray(molecule.grid.weights)
        orbo, orbv, _, _ = _restricted_orbital_data(molecule, self.occupation_tolerance)
        transition_densities = _transition_densities_on_grid(ao, orbo, orbv)
        mode_projections = jnp.einsum(
            "rp,ria,r->pia",
            mode_basis,
            transition_densities,
            weights,
            precision=Precision.HIGHEST,
        )
        coupling_matrix = jnp.asarray(self.model.apply(lr_params, mode_features))
        coupling_matrix = 0.5 * (coupling_matrix + coupling_matrix.T)
        coupling_matrix = jnp.asarray(self.kernel_scale, dtype=coupling_matrix.dtype) * coupling_matrix
        exact_exchange_fraction = (
            getattr(base_bound, "exact_exchange_fraction", 0.0) if base_bound is not None else 0.0
        )
        return BoundGridPointModeLongRangeCorrectedFunctional(
            name=self.name,
            base_bound=base_bound,
            mode_projections=mode_projections,
            coupling_matrix=coupling_matrix,
            exact_exchange_fraction=exact_exchange_fraction,
        )

    def bind_to_molecule_for_scf(self, params: PyTree, molecule: Any) -> Any:
        base_params = _extract_base_params(params)
        base_bound = _bind_base_functional(
            self.base_functional,
            base_params,
            molecule,
            prefer_scf=True,
        )
        if base_bound is None:
            return _ZeroBoundFunctional(name=self.name)
        return base_bound

    def energy_from_molecule(self, params: PyTree, molecule: Any) -> Array:
        if self.base_functional is None:
            return jnp.asarray(0.0, dtype=jnp.asarray(molecule.grid.weights).dtype)

        base_params = _extract_base_params(params)
        energy_from_molecule = getattr(self.base_functional, "energy_from_molecule", None)
        if energy_from_molecule is not None:
            if base_params is None:
                return jnp.asarray(energy_from_molecule(molecule))
            return jnp.asarray(energy_from_molecule(base_params, molecule))

        total_density = _density_on_grid(molecule)
        energy = getattr(self.base_functional, "energy", None)
        if energy is None:
            raise AttributeError(
                "Base functional must expose energy_from_molecule(...) or energy(...)."
            )
        if base_params is None:
            return jnp.asarray(energy(total_density, molecule.grid.weights))
        return jnp.asarray(energy(base_params, total_density, molecule.grid.weights))
