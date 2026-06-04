from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.lax import Precision
from jaxtyping import Array, PyTree

from ..features import (
    grid_features_for_molecule,
    grid_features_with_spin_gradients_for_molecule,
    grid_features_with_gradients_for_molecule,
    has_explicit_spin_axis,
    restricted_transition_response_features,
)
from ..tddft.cisd import (
    restricted_cisd_second_order_correction,
    unrestricted_cisd_second_order_correction,
)
from .inputs import (
    has_hfx_nu_source,
    hfx_nu_grid_chunk,
    hfx_nu_shape,
    hfx_nu_source,
    is_chunked_hfx_nu,
)
from ..xc_backend.jax_libxc import RestrictedFeatureBundle


def _requires_unrestricted_response_binding(molecule: Any) -> bool:
    if not has_explicit_spin_axis(molecule):
        return False
    nocc_alpha = getattr(molecule, "nocc_alpha", None)
    nocc_beta = getattr(molecule, "nocc_beta", None)
    if nocc_alpha is not None and nocc_beta is not None:
        try:
            if int(nocc_alpha) != int(nocc_beta):
                return True
        except (TypeError, ValueError):
            pass
    for name in ("mo_occ", "rdm1"):
        value = getattr(molecule, name, None)
        if value is None:
            continue
        arr = jnp.asarray(value)
        if isinstance(arr, jax.core.Tracer):
            continue
        host_arr = np.asarray(jax.device_get(arr))
        if host_arr.ndim >= 1 and int(host_arr.shape[0]) == 2 and not np.allclose(host_arr[0], host_arr[1]):
            return True
    return False


def _restricted_response_shape_from_molecule(
    molecule: Any,
    *,
    occupation_tolerance: float,
) -> tuple[int, int]:
    mo_coeff = jnp.asarray(molecule.mo_coeff)
    mo_occ = jnp.asarray(molecule.mo_occ)
    if mo_coeff.ndim == 3:
        mo_coeff = mo_coeff[0]
        mo_occ = mo_occ[0]
    nocc = getattr(molecule, "nocc", None)
    if nocc is None:
        nocc = int(jnp.count_nonzero(mo_occ > occupation_tolerance))
    else:
        nocc = int(nocc)
    return nocc, int(mo_coeff.shape[-1]) - nocc


@dataclass(frozen=True)
class BoundNeuralXCFunctional:
    name: str
    projected_local_potential_values: Array
    projected_local_kernel_values: Array
    exact_exchange_fraction: Array
    projected_local_potential_gradient_values: Array | None = None
    projected_local_potential_tau_values: Array | None = None
    projected_local_potential_laplacian_values: Array | None = None
    unrestricted_local_potential_values: tuple[Array, Array] | None = None
    unrestricted_local_potential_gradient_values: tuple[Array, Array] | None = None
    explicit_hfx_fock_value: Array | None = None
    unrestricted_explicit_hfx_fock_values: tuple[Array, Array] | None = None
    projected_energy_density_values: Array | None = None
    local_hf_fraction_values: Array | None = None
    response_feature_kind: str | None = None
    grid_response_tensor_fn: Callable[[], Array] | None = None
    spin_local_kernel_fn: Callable[[Array, Array], Any] | None = None
    grid_hfx_feature_gradients_fn: Callable[[], tuple[Array, Array]] | None = None
    strict_tda_xc_matrix_fn: Callable[..., Array] | None = None
    strict_tda_xc_action_fn: Callable[..., Array] | None = None
    strict_tda_xc_diagonal_fn: Callable[..., Array] | None = None
    nonlocal_response_matrix_fn: Callable[..., Array] | None = None
    nonlocal_response_matrices_fn: Callable[..., tuple[Array, Array]] | None = None
    nonlocal_response_action_fn: Callable[..., Array] | None = None
    nonlocal_response_b_action_fn: Callable[..., Array] | None = None
    nonlocal_response_diagonal_fn: Callable[..., Array] | None = None
    post_tda_correction_fn: Callable[..., Array] | None = None
    post_tddft_correction_fn: Callable[..., Array] | None = None

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

    def unrestricted_scf_components(self, molecule: Any) -> tuple[Array, ...]:
        if self.unrestricted_local_potential_values is None:
            rho_a = rho_b = self.projected_local_potential_values
        else:
            rho_a, rho_b = self.unrestricted_local_potential_values
        if self.unrestricted_local_potential_gradient_values is None:
            if self.projected_local_potential_gradient_values is None:
                grad = jnp.zeros(rho_a.shape + (3,), dtype=rho_a.dtype)
            else:
                grad = self.projected_local_potential_gradient_values
            grad_a = grad_b = grad
        else:
            grad_a, grad_b = self.unrestricted_local_potential_gradient_values
        nao = int(molecule.ao.shape[1])
        if self.unrestricted_explicit_hfx_fock_values is None:
            if self.explicit_hfx_fock_value is None:
                extra_fock_a = extra_fock_b = jnp.zeros((nao, nao), dtype=rho_a.dtype)
            else:
                extra_fock_a = extra_fock_b = jnp.asarray(
                    self.explicit_hfx_fock_value,
                    dtype=rho_a.dtype,
                )
        else:
            extra_fock_a, extra_fock_b = self.unrestricted_explicit_hfx_fock_values
        return (
            rho_a,
            rho_b,
            grad_a,
            grad_b,
            self.response_feature_kind or "LDA",
            self.exact_exchange_fraction,
            extra_fock_a,
            extra_fock_b,
        )

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

    def spin_local_kernel(self, density_alpha: Array, density_beta: Array) -> Any:
        if self.spin_local_kernel_fn is None:
            raise AttributeError(
                "This bound functional does not expose a spin-resolved local kernel."
            )
        return self.spin_local_kernel_fn(density_alpha, density_beta)

    def grid_hfx_feature_gradients(self, molecule: Any) -> tuple[Array, Array]:
        del molecule
        if self.grid_hfx_feature_gradients_fn is None:
            raise AttributeError(
                "This bound functional does not expose gradients with respect to local HF features."
            )
        return self.grid_hfx_feature_gradients_fn()

    def strict_tda_xc_matrix(
        self,
        molecule: Any,
        *,
        occupation_tolerance: float = 1e-8,
    ) -> Array:
        if self.strict_tda_xc_matrix_fn is None:
            raise AttributeError("This bound functional does not expose a strict TDA XC matrix.")
        return self.strict_tda_xc_matrix_fn(
            molecule,
            occupation_tolerance=occupation_tolerance,
        )

    def strict_tda_xc_action(
        self,
        molecule: Any,
        amplitudes: Array,
        *,
        occupation_tolerance: float = 1e-8,
    ) -> Array:
        if self.strict_tda_xc_action_fn is not None:
            return self.strict_tda_xc_action_fn(
                molecule,
                amplitudes,
                occupation_tolerance=occupation_tolerance,
            )
        matrix = self.strict_tda_xc_matrix(
            molecule,
            occupation_tolerance=occupation_tolerance,
        )
        nocc, nvir = _restricted_response_shape_from_molecule(
            molecule,
            occupation_tolerance=occupation_tolerance,
        )
        values = jnp.asarray(amplitudes, dtype=matrix.dtype)
        flat = values.reshape(-1, int(nocc * nvir))
        out = flat @ jnp.asarray(matrix, dtype=values.dtype).T
        return out.reshape(values.shape)

    def strict_tda_xc_diagonal(
        self,
        molecule: Any,
        *,
        occupation_tolerance: float = 1e-8,
    ) -> Array:
        if self.strict_tda_xc_diagonal_fn is not None:
            return self.strict_tda_xc_diagonal_fn(
                molecule,
                occupation_tolerance=occupation_tolerance,
            )
        matrix = self.strict_tda_xc_matrix(
            molecule,
            occupation_tolerance=occupation_tolerance,
        )
        nocc, nvir = _restricted_response_shape_from_molecule(
            molecule,
            occupation_tolerance=occupation_tolerance,
        )
        return jnp.diag(jnp.asarray(matrix)).reshape(nocc, nvir)

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

    def nonlocal_response_action(
        self,
        molecule: Any,
        amplitudes: Array,
        *,
        occupation_tolerance: float = 1e-8,
    ) -> Array:
        if self.nonlocal_response_action_fn is not None:
            return self.nonlocal_response_action_fn(
                molecule,
                amplitudes,
                occupation_tolerance=occupation_tolerance,
            )
        if self.nonlocal_response_matrix_fn is None and self.nonlocal_response_matrices_fn is None:
            raise AttributeError("This bound functional does not expose a nonlocal response action.")
        matrix = self.nonlocal_response_matrix(
            molecule,
            occupation_tolerance=occupation_tolerance,
        )
        nocc, nvir = _restricted_response_shape_from_molecule(
            molecule,
            occupation_tolerance=occupation_tolerance,
        )
        values = jnp.asarray(amplitudes, dtype=matrix.dtype)
        flat = values.reshape(-1, int(nocc * nvir))
        out = flat @ jnp.asarray(matrix, dtype=values.dtype).T
        return out.reshape(values.shape)

    def nonlocal_response_a_action(
        self,
        molecule: Any,
        amplitudes: Array,
        *,
        occupation_tolerance: float = 1e-8,
    ) -> Array:
        return self.nonlocal_response_action(
            molecule,
            amplitudes,
            occupation_tolerance=occupation_tolerance,
        )

    def nonlocal_response_b_action(
        self,
        molecule: Any,
        amplitudes: Array,
        *,
        occupation_tolerance: float = 1e-8,
    ) -> Array:
        if self.nonlocal_response_b_action_fn is not None:
            return self.nonlocal_response_b_action_fn(
                molecule,
                amplitudes,
                occupation_tolerance=occupation_tolerance,
            )
        if self.nonlocal_response_matrices_fn is not None:
            _, matrix = self.nonlocal_response_matrices(
                molecule,
                occupation_tolerance=occupation_tolerance,
            )
        elif self.nonlocal_response_matrix_fn is not None:
            matrix = self.nonlocal_response_matrix(
                molecule,
                occupation_tolerance=occupation_tolerance,
            )
        else:
            raise AttributeError("This bound functional does not expose a B nonlocal response action.")
        nocc, nvir = _restricted_response_shape_from_molecule(
            molecule,
            occupation_tolerance=occupation_tolerance,
        )
        values = jnp.asarray(amplitudes, dtype=matrix.dtype)
        flat = values.reshape(-1, int(nocc * nvir))
        out = flat @ jnp.asarray(matrix, dtype=values.dtype).T
        return out.reshape(values.shape)

    def nonlocal_response_diagonal(
        self,
        molecule: Any,
        *,
        occupation_tolerance: float = 1e-8,
    ) -> Array:
        if self.nonlocal_response_diagonal_fn is not None:
            return self.nonlocal_response_diagonal_fn(
                molecule,
                occupation_tolerance=occupation_tolerance,
            )
        if self.nonlocal_response_matrix_fn is None and self.nonlocal_response_matrices_fn is None:
            raise AttributeError("This bound functional does not expose a nonlocal response diagonal.")
        matrix = self.nonlocal_response_matrix(
            molecule,
            occupation_tolerance=occupation_tolerance,
        )
        nocc, nvir = _restricted_response_shape_from_molecule(
            molecule,
            occupation_tolerance=occupation_tolerance,
        )
        return jnp.diag(jnp.asarray(matrix)).reshape(nocc, nvir)

    def post_tda_correction(
        self,
        molecule: Any,
        result: Any,
        *,
        occupation_tolerance: float = 1e-8,
    ) -> Array:
        if self.post_tda_correction_fn is None:
            raise AttributeError("This bound functional does not expose a post-TDA correction.")
        return self.post_tda_correction_fn(
            molecule,
            result,
            occupation_tolerance=occupation_tolerance,
        )

    def post_tddft_correction(
        self,
        molecule: Any,
        result: Any,
        *,
        occupation_tolerance: float = 1e-8,
    ) -> Array:
        if self.post_tddft_correction_fn is None:
            raise AttributeError("This bound functional does not expose a post-TDDFT correction.")
        return self.post_tddft_correction_fn(
            molecule,
            result,
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
        semilocal_total = jnp.sum(semilocal_channels, axis=-1)
        weights = jnp.asarray(grid_weights)
        coefficients = self.channel_coefficients(
            params,
            features,
            semilocal_energy_density=semilocal_total,
            hf_energy_density=jax.lax.stop_gradient(hf_projected),
            pt2_energy_density=pt2_projected,
            hf_spin_energy_density=(
                jax.lax.stop_gradient(hf_feature_a),
                jax.lax.stop_gradient(hf_feature_b),
            ),
        )
        grad_a = weights * self._local_hf_fraction_from_coefficients(coefficients)
        grad_b = grad_a
        grad_a = self._maybe_clip_response(grad_a)
        grad_b = self._maybe_clip_response(grad_b)
        return grad_a, grad_b

    def _zero_hfx_fock(self, molecule: Any, dtype: Any | None = None) -> Array:
        ao = jnp.asarray(molecule.ao)
        matrix_dtype = ao.dtype if dtype is None else dtype
        return jnp.zeros((ao.shape[1], ao.shape[1]), dtype=matrix_dtype)

    def uses_explicit_hfx_fock_for_scf(self, molecule: Any) -> bool:
        return has_hfx_nu_source(molecule)

    def _contract_hfx_feature_gradients_to_restricted_fock(
        self,
        molecule: Any,
        grad_a: Array,
        grad_b: Array,
        *,
        dtype: Any | None = None,
    ) -> tuple[Array, bool]:
        nu_source = hfx_nu_source(molecule)
        if nu_source is None:
            return self._zero_hfx_fock(molecule, dtype), False

        ao = jnp.asarray(molecule.ao)
        matrix_dtype = ao.dtype if dtype is None else dtype
        ao = jnp.asarray(ao, dtype=matrix_dtype)
        n_omega, ngrid, nao, nao2 = hfx_nu_shape(nu_source)
        if nao != ao.shape[1] or nao2 != ao.shape[1]:
            raise ValueError(
                "HFX nu source AO dimensions must match molecule.ao second axis "
                f"(got {(nao, nao2)} vs {(ao.shape[1], ao.shape[1])})."
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
        if is_chunked_hfx_nu(nu_source):
            vmat = jnp.zeros((ao.shape[1], ao.shape[1]), dtype=matrix_dtype)
            chunk_size = self._effective_response_grid_chunk_size(int(ngrid))
            for start in range(0, int(ngrid), chunk_size):
                end = min(start + chunk_size, int(ngrid))
                ao_chunk = ao[start:end]
                grad_chunk = grad[start:end]
                nu_chunk = hfx_nu_grid_chunk(
                    nu_source,
                    start,
                    end,
                    n_omega=n_grad_channels,
                    dtype=matrix_dtype,
                )
                e = jnp.einsum(
                    "gp,pq->gq",
                    ao_chunk,
                    density_half,
                    precision=Precision.HIGHEST,
                )
                fxx = jnp.einsum(
                    "wgbc,gc->wgb",
                    nu_chunk,
                    e,
                    precision=Precision.HIGHEST,
                )
                aow = -0.5 * fxx * jnp.transpose(grad_chunk, (1, 0))[:, :, None]
                vmat = vmat + jnp.einsum(
                    "gp,wgq->pq",
                    ao_chunk,
                    aow,
                    precision=Precision.HIGHEST,
                )
        else:
            nu = jnp.asarray(nu_source, dtype=matrix_dtype)[:n_grad_channels]
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
        if not has_hfx_nu_source(molecule):
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
        if not has_hfx_nu_source(molecule):
            return self._zero_hfx_fock(molecule), False
        features = grid_features_for_molecule(molecule)
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
        response_features, response_total_gradient = grid_features_with_gradients_for_molecule(
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
        return matrix_a, matrix_b

    def _strict_nonlocal_response_actions_from_molecule(
        self,
        params: PyTree,
        response_molecule: Any,
        amplitudes: Array,
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
        return self._strict_hf_nonlocal_response_actions(
            params,
            response_molecule,
            response_features,
            response_total_gradient,
            response_hf,
            amplitudes,
            hf_spin_energy_density=(response_hfx_a, response_hfx_b),
            pt2_projected=response_pt2,
            occupation_tolerance=occupation_tolerance,
        )

    def _strict_nonlocal_response_diagonal_from_molecule(
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
        return self._strict_hf_nonlocal_response_diagonal(
            params,
            response_molecule,
            response_features,
            response_total_gradient,
            response_hf,
            hf_spin_energy_density=(response_hfx_a, response_hfx_b),
            pt2_projected=response_pt2,
            occupation_tolerance=occupation_tolerance,
        )

    def _nonlocal_response_callbacks(
        self,
        params: PyTree,
    ) -> tuple[None, Callable[..., tuple[Array, Array]]]:
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

        return None, nonlocal_response_matrices_fn

    def _nonlocal_response_action_callbacks(
        self,
        params: PyTree,
    ) -> tuple[Callable[..., Array], Callable[..., Array], Callable[..., Array]]:
        def nonlocal_response_action_fn(
            response_molecule: Any,
            amplitudes: Array,
            *,
            occupation_tolerance: float = 1e-8,
        ) -> Array:
            action_a, _ = self._strict_nonlocal_response_actions_from_molecule(
                params,
                response_molecule,
                amplitudes,
                occupation_tolerance=occupation_tolerance,
            )
            return action_a

        def nonlocal_response_b_action_fn(
            response_molecule: Any,
            amplitudes: Array,
            *,
            occupation_tolerance: float = 1e-8,
        ) -> Array:
            _, action_b = self._strict_nonlocal_response_actions_from_molecule(
                params,
                response_molecule,
                amplitudes,
                occupation_tolerance=occupation_tolerance,
            )
            return action_b

        def nonlocal_response_diagonal_fn(
            response_molecule: Any,
            *,
            occupation_tolerance: float = 1e-8,
        ) -> Array:
            return self._strict_nonlocal_response_diagonal_from_molecule(
                params,
                response_molecule,
                occupation_tolerance=occupation_tolerance,
            )

        return (
            nonlocal_response_action_fn,
            nonlocal_response_b_action_fn,
            nonlocal_response_diagonal_fn,
        )

    def _unrestricted_spin_local_kernel_components(
        self,
        params: PyTree,
        features: RestrictedFeatureBundle,
        grad_a: Array,
        grad_b: Array,
        hf_projected: Array,
        *,
        pt2_projected: Array | None = None,
        hf_spin_energy_density: tuple[Array, Array],
        response_hf_mode: str | None = None,
        response_pt2_mode: str | None = None,
    ) -> tuple[Array, Array, Array]:
        response_variables, active = self._unrestricted_response_variables(
            features,
            grad_a,
            grad_b,
        )
        hf_feature_a, hf_feature_b = hf_spin_energy_density
        pt2_feature = (
            jnp.zeros_like(hf_projected)
            if pt2_projected is None
            else jnp.asarray(pt2_projected)
        )
        point_hessian_fn = jax.hessian(
            self._total_point_local_energy_from_unrestricted_variables,
            argnums=1,
        )

        def point_spin_tensor(
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
            return 0.5 * (tensor[:2, :2] + tensor[:2, :2].T)

        spin_tensor = jax.vmap(point_spin_tensor)(
            response_variables,
            hf_projected,
            hf_feature_a,
            hf_feature_b,
            pt2_feature,
        )
        spin_tensor = spin_tensor * active[:, None, None].astype(spin_tensor.dtype)
        return spin_tensor[:, 0, 0], spin_tensor[:, 0, 1], spin_tensor[:, 1, 1]

    def _strict_pt2_posthoc_correction_callbacks(
        self,
        rho: Array,
        semilocal_channels: Array,
        coefficients: Array,
        pt2_projected: Array | None,
        grid_weights: Array,
        *,
        unrestricted: bool = False,
    ) -> tuple[Callable[..., Array] | None, Callable[..., Array] | None]:
        if (
            not self.include_pt2_channel
            or self.response_pt2_mode != "strict"
            or pt2_projected is None
        ):
            return None, None

        n_semilocal = int(jnp.asarray(semilocal_channels).shape[-1])
        pt2_coefficients = jnp.asarray(coefficients)[..., n_semilocal]
        weights = jnp.asarray(grid_weights)
        density = jnp.asarray(rho)
        if density.ndim > 1:
            density = jnp.sum(density, axis=-1)
        density = jnp.maximum(density, self.density_floor)
        numerator = jnp.tensordot(weights, density * pt2_coefficients, axes=(0, 0))
        denominator = jnp.tensordot(weights, density, axes=(0, 0))
        ac = numerator / jnp.maximum(denominator, self.density_floor)
        ac = jnp.nan_to_num(ac, nan=0.0, posinf=1.0, neginf=0.0)
        ac = jnp.clip(ac, 0.0, 1.0)

        def post_correction(
            molecule: Any,
            result: Any,
            *,
            occupation_tolerance: float = 1e-8,
        ) -> Array:
            correction_fn = (
                unrestricted_cisd_second_order_correction
                if unrestricted
                else restricted_cisd_second_order_correction
            )
            return correction_fn(
                molecule,
                result,
                ac=ac,
                occupation_tolerance=occupation_tolerance,
            )

        return post_correction, post_correction

    def _effective_response_grid_chunk_size(self, ngrids: int) -> int:
        chunk_size = getattr(self, "response_grid_chunk_size", None)
        if chunk_size is None or int(chunk_size) <= 0:
            return int(ngrids)
        return max(1, min(int(chunk_size), int(ngrids)))

    @staticmethod
    def _pad_grid_axis(values: Array, pad: int, *, axis: int = 0) -> Array:
        arr = jnp.asarray(values)
        if int(pad) <= 0:
            return arr
        pad_width = [(0, 0)] * arr.ndim
        pad_width[int(axis)] = (0, int(pad))
        return jnp.pad(arr, pad_width)

    @staticmethod
    def _take_grid_chunk(
        values: Array,
        start: Array,
        chunk_size: int,
        *,
        axis: int = 0,
    ) -> Array:
        arr = jnp.asarray(values)
        indices = start + jnp.arange(int(chunk_size))
        chunk = jnp.take(
            arr,
            indices,
            axis=int(axis),
            mode="clip",
        )
        valid = indices < int(arr.shape[int(axis)])
        mask_shape = [1] * arr.ndim
        mask_shape[int(axis)] = int(chunk_size)
        return jnp.where(
            valid.reshape(mask_shape),
            chunk,
            jnp.zeros_like(chunk),
        )

    def _transition_response_features_chunk(
        self,
        molecule: Any,
        start: Array,
        chunk_size: int,
        *,
        feature_kind: str,
        occupation_tolerance: float,
        orbo: Array,
        orbv: Array,
    ) -> Array:
        del occupation_tolerance
        kind = feature_kind.upper()
        ao = self._take_grid_chunk(molecule.ao, start, chunk_size, axis=0)
        rho_o = jnp.einsum("rp,pi->ri", ao, orbo, precision=Precision.HIGHEST)
        rho_v = jnp.einsum("rp,pa->ra", ao, orbv, precision=Precision.HIGHEST)
        rho_ov = rho_o[:, :, None] * rho_v[:, None, :]
        if kind == "LDA":
            return rho_ov[None, ...]

        ao_deriv1 = getattr(molecule, "ao_deriv1", None)
        if ao_deriv1 is None:
            raise AttributeError(
                "Molecule-like object must define ao_deriv1 for GGA/meta-GGA transition features."
            )
        ao_deriv1 = self._take_grid_chunk(ao_deriv1, start, chunk_size, axis=1)
        if ao_deriv1.shape[0] < 4:
            raise ValueError("ao_deriv1 must contain AO values plus first derivatives.")

        rho_o_full = jnp.einsum(
            "xrp,pi->xri",
            ao_deriv1[:4],
            orbo,
            precision=Precision.HIGHEST,
        )
        rho_v_full = jnp.einsum(
            "xrp,pa->xra",
            ao_deriv1[:4],
            orbv,
            precision=Precision.HIGHEST,
        )
        gga_features = rho_o_full[:, :, :, None] * rho_v_full[0][None, :, None, :]
        gga_features = gga_features.at[1:4].add(
            rho_o_full[0][None, :, :, None] * rho_v_full[1:4, :, None, :]
        )
        if kind == "GGA":
            return gga_features

        tau_ov = 0.5 * jnp.sum(
            rho_o_full[1:4, :, :, None] * rho_v_full[1:4, :, None, :],
            axis=0,
        )
        mgga_features = jnp.concatenate([gga_features, tau_ov[None, ...]], axis=0)
        if kind == "MGGA":
            return mgga_features
        if kind != "MGGA_LAPL":
            raise ValueError(f"Unsupported feature_kind={feature_kind!r}.")

        ao_laplacian = getattr(molecule, "ao_laplacian", None)
        if ao_laplacian is None:
            raise AttributeError(
                "Molecule-like object must define ao_laplacian to build laplacian response features."
            )
        ao_laplacian = self._take_grid_chunk(
            ao_laplacian,
            start,
            chunk_size,
            axis=0,
        )
        lapl_o = jnp.einsum("rp,pi->ri", ao_laplacian, orbo, precision=Precision.HIGHEST)
        lapl_v = jnp.einsum("rp,pa->ra", ao_laplacian, orbv, precision=Precision.HIGHEST)
        lapl_ov = (
            lapl_o[:, :, None] * rho_v[:, None, :]
            + 2.0
            * jnp.sum(
                rho_o_full[1:4, :, :, None] * rho_v_full[1:4, :, None, :],
                axis=0,
            )
            + rho_o[:, :, None] * lapl_v[:, None, :]
        )
        return jnp.concatenate([mgga_features, lapl_ov[None, ...]], axis=0)

    def _strict_tda_xc_matrix_chunked(
        self,
        params: PyTree,
        molecule: Any,
        features: RestrictedFeatureBundle,
        total_gradient: Array,
        hf_projected: Array,
        *,
        pt2_projected: Array | None,
        hf_spin_energy_density: tuple[Array, Array],
        response_hf_mode: str | None,
        strict_payload: tuple[Array, Array, Array, Array, Array],
        occupation_tolerance: float,
    ) -> Array:
        weights = jnp.asarray(molecule.grid.weights)
        ngrids = int(weights.shape[0])
        chunk_size = self._effective_response_grid_chunk_size(ngrids)
        n_chunks = (ngrids + chunk_size - 1) // chunk_size
        padded = n_chunks * chunk_size
        pad = padded - ngrids
        feature_kind = self._response_feature_kind_label()

        response_variables, active, hf_feature_a, hf_feature_b, pt2_feature = strict_payload
        if getattr(self, "strict_hfx_response_mode", "dense") == "low_memory":
            mo_coeff = jnp.asarray(molecule.mo_coeff)
            mo_occ = jnp.asarray(molecule.mo_occ)
            if mo_coeff.ndim == 3:
                mo_coeff = mo_coeff[0]
                mo_occ = mo_occ[0]
            nocc = getattr(molecule, "nocc", None)
            if nocc is None:
                nocc = int(jnp.count_nonzero(mo_occ > occupation_tolerance))
            else:
                nocc = int(nocc)
            nmo = int(mo_coeff.shape[1])
            if nocc <= 0 or nocc >= nmo:
                raise ValueError("Need at least one occupied and one virtual orbital.")
            orbo = mo_coeff[:, :nocc]
            orbv = mo_coeff[:, nocc:]
            dim = int(nocc * (nmo - nocc))
            zero = jnp.zeros((dim, dim), dtype=weights.dtype)

            def chunk_matrix(
                local_params: PyTree,
                hf_projected_chunk: Array,
                response_variables_chunk: Array,
                active_chunk: Array,
                hf_feature_a_chunk: Array,
                hf_feature_b_chunk: Array,
                pt2_feature_chunk: Array,
                weights_chunk: Array,
                response_chunk: Array,
            ) -> Array:
                strict_payload_chunk = (
                    response_variables_chunk,
                    active_chunk,
                    hf_feature_a_chunk,
                    hf_feature_b_chunk,
                    pt2_feature_chunk,
                )
                tensor_chunk = self._strict_total_response_tensor(
                    local_params,
                    features,
                    total_gradient,
                    hf_projected_chunk,
                    pt2_projected=None,
                    hf_spin_energy_density=(hf_feature_a_chunk, hf_feature_b_chunk),
                    response_hf_mode=response_hf_mode,
                    response_pt2_mode=self.response_pt2_mode,
                    strict_payload=strict_payload_chunk,
                )
                weighted_tensor = tensor_chunk * weights_chunk[None, None, :]
                response_chunk = response_chunk.reshape(response_chunk.shape[0], chunk_size, dim)
                return 2.0 * jnp.einsum(
                    "xyr,xrd,yre->de",
                    weighted_tensor,
                    response_chunk,
                    response_chunk,
                    precision=Precision.HIGHEST,
                )

            def chunk_matrix_from_start(local_params: PyTree, start: Array) -> Array:
                return chunk_matrix(
                    local_params,
                    self._take_grid_chunk(hf_projected, start, chunk_size, axis=0),
                    self._take_grid_chunk(response_variables, start, chunk_size, axis=0),
                    self._take_grid_chunk(active, start, chunk_size, axis=0),
                    self._take_grid_chunk(hf_feature_a, start, chunk_size, axis=0),
                    self._take_grid_chunk(hf_feature_b, start, chunk_size, axis=0),
                    self._take_grid_chunk(pt2_feature, start, chunk_size, axis=0),
                    self._take_grid_chunk(weights, start, chunk_size, axis=0),
                    self._transition_response_features_chunk(
                        molecule,
                        start,
                        chunk_size,
                        feature_kind=feature_kind,
                        occupation_tolerance=occupation_tolerance,
                        orbo=orbo,
                        orbv=orbv,
                    ),
                )

            chunk_matrix_from_start = jax.checkpoint(chunk_matrix_from_start)

            def body(acc: Array, chunk_idx: Array) -> tuple[Array, None]:
                start = chunk_idx * chunk_size
                matrix = chunk_matrix_from_start(params, start)
                return acc + matrix, None

            matrix, _ = jax.lax.scan(body, zero, jnp.arange(n_chunks))
            return matrix

        response_features = restricted_transition_response_features(
            molecule,
            feature_kind=feature_kind,
            occupation_tolerance=occupation_tolerance,
        )
        weights = self._pad_grid_axis(weights, pad)
        hf_projected = self._pad_grid_axis(hf_projected, pad)
        response_variables = self._pad_grid_axis(response_variables, pad)
        active = self._pad_grid_axis(active, pad)
        hf_feature_a = self._pad_grid_axis(hf_feature_a, pad)
        hf_feature_b = self._pad_grid_axis(hf_feature_b, pad)
        pt2_feature = self._pad_grid_axis(pt2_feature, pad)
        response_features = self._pad_grid_axis(response_features, pad, axis=1)

        dim = int(response_features.shape[2] * response_features.shape[3])
        zero = jnp.zeros((dim, dim), dtype=response_features.dtype)

        def _slice_axis0(values: Array, start: Array) -> Array:
            return jax.lax.dynamic_slice_in_dim(values, start, chunk_size, axis=0)

        def _slice_axis1(values: Array, start: Array) -> Array:
            return jax.lax.dynamic_slice_in_dim(values, start, chunk_size, axis=1)

        def chunk_matrix(
            local_params: PyTree,
            hf_projected_chunk: Array,
            response_variables_chunk: Array,
            active_chunk: Array,
            hf_feature_a_chunk: Array,
            hf_feature_b_chunk: Array,
            pt2_feature_chunk: Array,
            weights_chunk: Array,
            response_chunk: Array,
        ) -> Array:
            strict_payload_chunk = (
                response_variables_chunk,
                active_chunk,
                hf_feature_a_chunk,
                hf_feature_b_chunk,
                pt2_feature_chunk,
            )
            tensor_chunk = self._strict_total_response_tensor(
                local_params,
                features,
                total_gradient,
                hf_projected_chunk,
                pt2_projected=pt2_projected,
                hf_spin_energy_density=hf_spin_energy_density,
                response_hf_mode=response_hf_mode,
                response_pt2_mode=self.response_pt2_mode,
                strict_payload=strict_payload_chunk,
            )
            weighted_tensor = tensor_chunk * weights_chunk[None, None, :]
            response_chunk = response_chunk.reshape(
                response_features.shape[0],
                chunk_size,
                dim,
            )
            return 2.0 * jnp.einsum(
                "xyr,xrd,yre->de",
                weighted_tensor,
                response_chunk,
                response_chunk,
                precision=Precision.HIGHEST,
            )

        chunk_matrix = jax.checkpoint(chunk_matrix)

        def body(acc: Array, chunk_idx: Array) -> tuple[Array, None]:
            start = chunk_idx * chunk_size
            matrix = chunk_matrix(
                params,
                _slice_axis0(hf_projected, start),
                _slice_axis0(response_variables, start),
                _slice_axis0(active, start),
                _slice_axis0(hf_feature_a, start),
                _slice_axis0(hf_feature_b, start),
                _slice_axis0(pt2_feature, start),
                _slice_axis0(weights, start),
                _slice_axis1(response_features, start),
            )
            return acc + matrix, None

        matrix, _ = jax.lax.scan(body, zero, jnp.arange(n_chunks))
        return matrix

    def _strict_tda_xc_action_chunked(
        self,
        params: PyTree,
        molecule: Any,
        features: RestrictedFeatureBundle,
        total_gradient: Array,
        hf_projected: Array,
        amplitudes: Array,
        *,
        pt2_projected: Array | None,
        hf_spin_energy_density: tuple[Array, Array],
        response_hf_mode: str | None,
        strict_payload: tuple[Array, Array, Array, Array, Array],
        occupation_tolerance: float,
    ) -> Array:
        weights = jnp.asarray(molecule.grid.weights)
        ngrids = int(weights.shape[0])
        chunk_size = self._effective_response_grid_chunk_size(ngrids)
        n_chunks = (ngrids + chunk_size - 1) // chunk_size
        feature_kind = self._response_feature_kind_label()
        response_variables, active, hf_feature_a, hf_feature_b, pt2_feature = strict_payload

        mo_coeff = jnp.asarray(molecule.mo_coeff)
        mo_occ = jnp.asarray(molecule.mo_occ)
        if mo_coeff.ndim == 3:
            mo_coeff = mo_coeff[0]
            mo_occ = mo_occ[0]
        nocc = getattr(molecule, "nocc", None)
        if nocc is None:
            nocc = int(jnp.count_nonzero(mo_occ > occupation_tolerance))
        else:
            nocc = int(nocc)
        nmo = int(mo_coeff.shape[1])
        if nocc <= 0 or nocc >= nmo:
            raise ValueError("Need at least one occupied and one virtual orbital.")
        orbo = mo_coeff[:, :nocc]
        orbv = mo_coeff[:, nocc:]
        nvir = int(nmo - nocc)
        values = jnp.asarray(amplitudes)
        leading_shape = values.shape[:-2]
        flat_values = values.reshape(-1, nocc, nvir)
        zero = jnp.zeros_like(flat_values)

        def chunk_action(local_params: PyTree, start: Array) -> Array:
            response_chunk = self._transition_response_features_chunk(
                molecule,
                start,
                chunk_size,
                feature_kind=feature_kind,
                occupation_tolerance=occupation_tolerance,
                orbo=orbo,
                orbv=orbv,
            )
            strict_payload_chunk = (
                self._take_grid_chunk(response_variables, start, chunk_size, axis=0),
                self._take_grid_chunk(active, start, chunk_size, axis=0),
                self._take_grid_chunk(hf_feature_a, start, chunk_size, axis=0),
                self._take_grid_chunk(hf_feature_b, start, chunk_size, axis=0),
                self._take_grid_chunk(pt2_feature, start, chunk_size, axis=0),
            )
            tensor_chunk = self._strict_total_response_tensor(
                local_params,
                features,
                total_gradient,
                self._take_grid_chunk(hf_projected, start, chunk_size, axis=0),
                pt2_projected=None,
                hf_spin_energy_density=(
                    strict_payload_chunk[2],
                    strict_payload_chunk[3],
                ),
                response_hf_mode=response_hf_mode,
                response_pt2_mode=self.response_pt2_mode,
                strict_payload=strict_payload_chunk,
            )
            projected = jnp.einsum(
                "xria,nia->nxr",
                response_chunk,
                flat_values,
                precision=Precision.HIGHEST,
            )
            weighted = jnp.einsum(
                "xyr,nyr->nxr",
                tensor_chunk * self._take_grid_chunk(weights, start, chunk_size, axis=0)[
                    None,
                    None,
                    :,
                ],
                projected,
                precision=Precision.HIGHEST,
            )
            return 2.0 * jnp.einsum(
                "xria,nxr->nia",
                response_chunk,
                weighted,
                precision=Precision.HIGHEST,
            )

        chunk_action = jax.checkpoint(chunk_action)

        def body(acc: Array, chunk_idx: Array) -> tuple[Array, None]:
            return acc + chunk_action(params, chunk_idx * chunk_size), None

        out, _ = jax.lax.scan(body, zero, jnp.arange(n_chunks))
        return out.reshape(leading_shape + (nocc, nvir))

    def _strict_tda_xc_diagonal_chunked(
        self,
        params: PyTree,
        molecule: Any,
        features: RestrictedFeatureBundle,
        total_gradient: Array,
        hf_projected: Array,
        *,
        pt2_projected: Array | None,
        hf_spin_energy_density: tuple[Array, Array],
        response_hf_mode: str | None,
        strict_payload: tuple[Array, Array, Array, Array, Array],
        occupation_tolerance: float,
    ) -> Array:
        weights = jnp.asarray(molecule.grid.weights)
        ngrids = int(weights.shape[0])
        chunk_size = self._effective_response_grid_chunk_size(ngrids)
        n_chunks = (ngrids + chunk_size - 1) // chunk_size
        feature_kind = self._response_feature_kind_label()
        response_variables, active, hf_feature_a, hf_feature_b, pt2_feature = strict_payload

        mo_coeff = jnp.asarray(molecule.mo_coeff)
        mo_occ = jnp.asarray(molecule.mo_occ)
        if mo_coeff.ndim == 3:
            mo_coeff = mo_coeff[0]
            mo_occ = mo_occ[0]
        nocc = getattr(molecule, "nocc", None)
        if nocc is None:
            nocc = int(jnp.count_nonzero(mo_occ > occupation_tolerance))
        else:
            nocc = int(nocc)
        nmo = int(mo_coeff.shape[1])
        if nocc <= 0 or nocc >= nmo:
            raise ValueError("Need at least one occupied and one virtual orbital.")
        orbo = mo_coeff[:, :nocc]
        orbv = mo_coeff[:, nocc:]
        zero = jnp.zeros((nocc, int(nmo - nocc)), dtype=weights.dtype)

        def chunk_diagonal(local_params: PyTree, start: Array) -> Array:
            response_chunk = self._transition_response_features_chunk(
                molecule,
                start,
                chunk_size,
                feature_kind=feature_kind,
                occupation_tolerance=occupation_tolerance,
                orbo=orbo,
                orbv=orbv,
            )
            strict_payload_chunk = (
                self._take_grid_chunk(response_variables, start, chunk_size, axis=0),
                self._take_grid_chunk(active, start, chunk_size, axis=0),
                self._take_grid_chunk(hf_feature_a, start, chunk_size, axis=0),
                self._take_grid_chunk(hf_feature_b, start, chunk_size, axis=0),
                self._take_grid_chunk(pt2_feature, start, chunk_size, axis=0),
            )
            tensor_chunk = self._strict_total_response_tensor(
                local_params,
                features,
                total_gradient,
                self._take_grid_chunk(hf_projected, start, chunk_size, axis=0),
                pt2_projected=None,
                hf_spin_energy_density=(
                    strict_payload_chunk[2],
                    strict_payload_chunk[3],
                ),
                response_hf_mode=response_hf_mode,
                response_pt2_mode=self.response_pt2_mode,
                strict_payload=strict_payload_chunk,
            )
            weighted_tensor = tensor_chunk * self._take_grid_chunk(
                weights,
                start,
                chunk_size,
                axis=0,
            )[None, None, :]
            return 2.0 * jnp.einsum(
                "xyr,xria,yria->ia",
                weighted_tensor,
                response_chunk,
                response_chunk,
                precision=Precision.HIGHEST,
            )

        chunk_diagonal = jax.checkpoint(chunk_diagonal)

        def body(acc: Array, chunk_idx: Array) -> tuple[Array, None]:
            return acc + chunk_diagonal(params, chunk_idx * chunk_size), None

        diagonal, _ = jax.lax.scan(body, zero, jnp.arange(n_chunks))
        return diagonal

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
        features, total_gradient = grid_features_with_gradients_for_molecule(molecule)
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
        features, total_gradient = grid_features_with_gradients_for_molecule(molecule)
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
        features, total_gradient = grid_features_with_gradients_for_molecule(molecule)
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

        _, nonlocal_response_matrices_fn = self._nonlocal_response_callbacks(
            params
        )
        (
            nonlocal_response_action_fn,
            nonlocal_response_b_action_fn,
            nonlocal_response_diagonal_fn,
        ) = self._nonlocal_response_action_callbacks(params)
        post_tda_correction_fn, post_tddft_correction_fn = (
            self._strict_pt2_posthoc_correction_callbacks(
                features.rho,
                semilocal_channels,
                coefficients,
                pt2_projected,
                molecule.grid.weights,
            )
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
            nonlocal_response_matrix_fn=None,
            nonlocal_response_matrices_fn=(
                nonlocal_response_matrices_fn
                if self.response_hf_mode == "strict"
                else None
            ),
            nonlocal_response_action_fn=(
                nonlocal_response_action_fn
                if self.response_hf_mode == "strict"
                else None
            ),
            nonlocal_response_b_action_fn=(
                nonlocal_response_b_action_fn
                if self.response_hf_mode == "strict"
                else None
            ),
            nonlocal_response_diagonal_fn=(
                nonlocal_response_diagonal_fn
                if self.response_hf_mode == "strict"
                else None
            ),
            post_tda_correction_fn=post_tda_correction_fn,
            post_tddft_correction_fn=post_tddft_correction_fn,
        )

    def bind_to_molecule_for_response(
        self,
        params: PyTree,
        molecule: Any,
    ) -> BoundNeuralXCFunctional:
        """TD-response-only binding that avoids assembling strict potential terms."""

        if _requires_unrestricted_response_binding(molecule):
            features, grad_a, grad_b = grid_features_with_spin_gradients_for_molecule(molecule)
            semilocal_channels = self.semilocal_energy_density_channels(features)
            semilocal = jnp.sum(semilocal_channels, axis=-1)
            hf_projected, hf_projected_a, hf_projected_b = (
                self.projected_hf_grid_contribution_components(
                    molecule,
                    features=features,
                )
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
            rho = jnp.maximum(features.rho, self.density_floor)
            grid_weights = jnp.asarray(molecule.grid.weights)
            numerator = jnp.tensordot(grid_weights, rho * hf_field, axes=(0, 0))
            denominator = jnp.tensordot(grid_weights, rho, axes=(0, 0))
            alpha = numerator / jnp.maximum(denominator, self.density_floor)
            alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
            alpha = jnp.clip(alpha, 0.0, 1.0)

            def spin_local_kernel_fn(
                density_alpha: Array,
                density_beta: Array,
            ) -> tuple[Array, Array, Array]:
                del density_alpha, density_beta
                return self._unrestricted_spin_local_kernel_components(
                    params,
                    features,
                    grad_a,
                    grad_b,
                    hf_projected,
                    pt2_projected=pt2_projected,
                    hf_spin_energy_density=(hfx_feature_a, hfx_feature_b),
                    response_hf_mode=self.response_hf_mode,
                    response_pt2_mode=self.response_pt2_mode,
                )

            response_alpha = (
                jnp.zeros_like(jnp.asarray(alpha))
                if self.response_hf_mode == "strict"
                else alpha
            )
            post_tda_correction_fn, post_tddft_correction_fn = (
                self._strict_pt2_posthoc_correction_callbacks(
                    features.rho,
                    semilocal_channels,
                    coefficients,
                    pt2_projected,
                    molecule.grid.weights,
                    unrestricted=True,
                )
            )
            return BoundNeuralXCFunctional(
                name=self.name,
                projected_local_potential_values=jnp.zeros_like(features.rho),
                projected_local_kernel_values=jnp.zeros_like(features.rho),
                exact_exchange_fraction=response_alpha,
                projected_local_potential_gradient_values=None,
                projected_local_potential_tau_values=None,
                projected_local_potential_laplacian_values=None,
                projected_energy_density_values=None,
                local_hf_fraction_values=None,
                response_feature_kind=self._response_feature_kind_label(),
                grid_response_tensor_fn=None,
                spin_local_kernel_fn=spin_local_kernel_fn,
                grid_hfx_feature_gradients_fn=None,
                strict_tda_xc_matrix_fn=None,
                nonlocal_response_matrix_fn=None,
                nonlocal_response_matrices_fn=None,
                post_tda_correction_fn=post_tda_correction_fn,
                post_tddft_correction_fn=post_tddft_correction_fn,
            )

        features, total_gradient = grid_features_with_gradients_for_molecule(molecule)
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
        rho = jnp.maximum(features.rho, self.density_floor)
        grid_weights = jnp.asarray(molecule.grid.weights)
        numerator = jnp.tensordot(grid_weights, rho * hf_field, axes=(0, 0))
        denominator = jnp.tensordot(grid_weights, rho, axes=(0, 0))
        alpha = numerator / jnp.maximum(denominator, self.density_floor)
        alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
        alpha = jnp.clip(alpha, 0.0, 1.0)

        def grid_response_tensor_fn() -> Array:
            return self._strict_total_response_tensor(
                params,
                features,
                total_gradient,
                hf_projected,
                pt2_projected=pt2_projected,
                hf_spin_energy_density=(hfx_feature_a, hfx_feature_b),
                response_hf_mode=self.response_hf_mode,
                response_pt2_mode=self.response_pt2_mode,
                strict_payload=strict_payload,
            )

        def strict_tda_xc_matrix_fn(
            response_molecule: Any,
            *,
            occupation_tolerance: float = 1e-8,
        ) -> Array:
            del response_molecule
            return self._strict_tda_xc_matrix_chunked(
                params,
                molecule,
                features,
                total_gradient,
                hf_projected,
                pt2_projected=pt2_projected,
                hf_spin_energy_density=(hfx_feature_a, hfx_feature_b),
                response_hf_mode=self.response_hf_mode,
                strict_payload=strict_payload,
                occupation_tolerance=occupation_tolerance,
            )

        def strict_tda_xc_action_fn(
            response_molecule: Any,
            amplitudes: Array,
            *,
            occupation_tolerance: float = 1e-8,
        ) -> Array:
            del response_molecule
            return self._strict_tda_xc_action_chunked(
                params,
                molecule,
                features,
                total_gradient,
                hf_projected,
                amplitudes,
                pt2_projected=pt2_projected,
                hf_spin_energy_density=(hfx_feature_a, hfx_feature_b),
                response_hf_mode=self.response_hf_mode,
                strict_payload=strict_payload,
                occupation_tolerance=occupation_tolerance,
            )

        def strict_tda_xc_diagonal_fn(
            response_molecule: Any,
            *,
            occupation_tolerance: float = 1e-8,
        ) -> Array:
            del response_molecule
            return self._strict_tda_xc_diagonal_chunked(
                params,
                molecule,
                features,
                total_gradient,
                hf_projected,
                pt2_projected=pt2_projected,
                hf_spin_energy_density=(hfx_feature_a, hfx_feature_b),
                response_hf_mode=self.response_hf_mode,
                strict_payload=strict_payload,
                occupation_tolerance=occupation_tolerance,
            )

        _, nonlocal_response_matrices_fn = self._nonlocal_response_callbacks(
            params
        )
        (
            nonlocal_response_action_fn,
            nonlocal_response_b_action_fn,
            nonlocal_response_diagonal_fn,
        ) = self._nonlocal_response_action_callbacks(params)
        post_tda_correction_fn, post_tddft_correction_fn = (
            self._strict_pt2_posthoc_correction_callbacks(
                features.rho,
                semilocal_channels,
                coefficients,
                pt2_projected,
                molecule.grid.weights,
            )
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
            projected_local_kernel_values=jnp.zeros_like(features.rho),
            exact_exchange_fraction=response_alpha,
            projected_local_potential_gradient_values=None,
            projected_local_potential_tau_values=None,
            projected_local_potential_laplacian_values=None,
            projected_energy_density_values=None,
            local_hf_fraction_values=None,
            response_feature_kind=self._response_feature_kind_label(),
            grid_response_tensor_fn=grid_response_tensor_fn,
            spin_local_kernel_fn=None,
            grid_hfx_feature_gradients_fn=None,
            strict_tda_xc_matrix_fn=strict_tda_xc_matrix_fn,
            strict_tda_xc_action_fn=strict_tda_xc_action_fn,
            strict_tda_xc_diagonal_fn=strict_tda_xc_diagonal_fn,
            nonlocal_response_matrix_fn=None,
            nonlocal_response_matrices_fn=(
                nonlocal_response_matrices_fn
                if self.response_hf_mode == "strict"
                else None
            ),
            nonlocal_response_action_fn=(
                nonlocal_response_action_fn
                if self.response_hf_mode == "strict"
                else None
            ),
            nonlocal_response_b_action_fn=(
                nonlocal_response_b_action_fn
                if self.response_hf_mode == "strict"
                else None
            ),
            nonlocal_response_diagonal_fn=(
                nonlocal_response_diagonal_fn
                if self.response_hf_mode == "strict"
                else None
            ),
            post_tda_correction_fn=post_tda_correction_fn,
            post_tddft_correction_fn=post_tddft_correction_fn,
        )

    def _scf_binding_payload(
        self,
        params: PyTree,
        molecule: Any,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        """Return SCF-only local potential components, HF fraction, and extra Fock."""

        features, total_gradient = grid_features_with_gradients_for_molecule(molecule)
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

    def _unrestricted_scf_binding_payload(
        self,
        params: PyTree,
        molecule: Any,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        features, grad_a, grad_b = grid_features_with_spin_gradients_for_molecule(molecule)
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

        v_rho_a, v_rho_b, v_grad_a, v_grad_b = self._unrestricted_total_potential_components(
            params,
            features,
            grad_a,
            grad_b,
            hf_projected,
            pt2_projected=pt2_projected,
            hf_spin_energy_density=(hfx_feature_a, hfx_feature_b),
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
        return v_rho_a, v_rho_b, v_grad_a, v_grad_b, alpha, hfx_fock

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

    def unrestricted_scf_potential_components_and_alpha(
        self,
        params: PyTree,
        molecule: Any,
    ) -> tuple[Array, Array, Array, Array, str, Array, Array, Array]:
        (
            v_rho_a,
            v_rho_b,
            v_grad_a,
            v_grad_b,
            alpha,
            hfx_fock,
        ) = self._unrestricted_scf_binding_payload(params, molecule)
        return (
            v_rho_a,
            v_rho_b,
            v_grad_a,
            v_grad_b,
            self._response_feature_kind_label(),
            alpha,
            hfx_fock,
            hfx_fock,
        )

    def bind_to_molecule_for_scf(self, params: PyTree, molecule: Any) -> BoundNeuralXCFunctional:
        """SCF-only binding that avoids constructing strict f_xc response terms."""
        spin_values = spin_gradients = None
        if has_explicit_spin_axis(molecule):
            (
                v_rho_a,
                v_rho_b,
                v_grad_a,
                v_grad_b,
                alpha,
                hfx_fock,
            ) = self._unrestricted_scf_binding_payload(params, molecule)
            projected_vrho = 0.5 * (v_rho_a + v_rho_b)
            projected_vgrad = 0.5 * (v_grad_a + v_grad_b)
            projected_vtau = projected_vlapl = jnp.zeros_like(projected_vrho)
            spin_values = (v_rho_a, v_rho_b)
            spin_gradients = (v_grad_a, v_grad_b)
        else:
            (
                projected_vrho,
                projected_vgrad,
                projected_vtau,
                projected_vlapl,
                alpha,
                hfx_fock,
            ) = self._scf_binding_payload(params, molecule)
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
            unrestricted_local_potential_values=spin_values,
            unrestricted_local_potential_gradient_values=spin_gradients,
            explicit_hfx_fock_value=hfx_fock,
            unrestricted_explicit_hfx_fock_values=(
                (hfx_fock, hfx_fock) if spin_values is not None else None
            ),
            projected_energy_density_values=None,
            local_hf_fraction_values=None,
            response_feature_kind=self._response_feature_kind_label(),
            grid_response_tensor_fn=None,
            spin_local_kernel_fn=None,
            grid_hfx_feature_gradients_fn=None,
            nonlocal_response_matrix_fn=None,
        )
