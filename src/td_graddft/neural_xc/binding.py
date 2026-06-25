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
)
from ..tddft.cisd import (
    restricted_cisd_second_order_correction,
    unrestricted_cisd_second_order_correction,
)
from .inputs import (
    has_hfx_nu_source,
    hfx_nu_grid_chunk_padded,
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


def _pack_restricted_grid_payload(
    features: RestrictedFeatureBundle,
    total_gradient: Array,
    semilocal_channels: Array,
    semilocal: Array,
) -> tuple[RestrictedFeatureBundle, Array, Array, Array]:
    return features, total_gradient, semilocal_channels, semilocal


def _cached_restricted_grid_payload(
    molecule: Any,
) -> tuple[RestrictedFeatureBundle, Array, Array, Array] | None:
    cached = getattr(molecule, "neural_xc_grid_payload", None)
    if cached is None or not isinstance(cached, tuple) or len(cached) != 4:
        return None
    features, total_gradient, semilocal_channels, semilocal = cached
    if not hasattr(features, "rho"):
        return None
    try:
        ngrid = int(molecule.grid.weights.shape[0])
        if int(features.rho.shape[0]) != ngrid:
            return None
        if int(semilocal.shape[0]) != ngrid:
            return None
        if int(semilocal_channels.shape[0]) != ngrid:
            return None
    except (AttributeError, TypeError, ValueError):
        return None
    return features, total_gradient, semilocal_channels, semilocal


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
    response_hf_mode: str = "approx"
    grid_response_tensor_fn: Callable[[], Array] | None = None
    grid_response_hvp_fn: Callable[..., Array] | None = None
    spin_local_kernel_fn: Callable[[Array, Array], Any] | None = None
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

    def grid_response_hvp(
        self,
        molecule: Any,
        tangent: Array,
    ) -> Array:
        if self.grid_response_hvp_fn is None:
            raise AttributeError("This bound functional does not expose a grid response HVP.")
        return self.grid_response_hvp_fn(molecule, tangent)

    def spin_local_kernel(self, density_alpha: Array, density_beta: Array) -> Any:
        if self.spin_local_kernel_fn is None:
            raise AttributeError(
                "This bound functional does not expose a spin-resolved local kernel."
            )
        return self.spin_local_kernel_fn(density_alpha, density_beta)

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
        raise AttributeError("This bound functional does not expose a nonlocal response action.")

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
        raise AttributeError("This bound functional does not expose a B nonlocal response action.")

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
        raise AttributeError("This bound functional does not expose a nonlocal response diagonal.")

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
    def _restricted_grid_features(
        self,
        molecule: Any,
    ) -> tuple[RestrictedFeatureBundle, Array]:
        cached = _cached_restricted_grid_payload(molecule)
        if cached is not None:
            return cached[0], cached[1]
        return grid_features_with_gradients_for_molecule(molecule)

    def _restricted_grid_payload(
        self,
        molecule: Any,
    ) -> tuple[RestrictedFeatureBundle, Array, Array, Array]:
        cached = _cached_restricted_grid_payload(molecule)
        if cached is not None and int(cached[2].shape[-1]) == int(
            self.resolved_non_hf_module().n_channels
        ):
            return cached
        features, total_gradient = self._restricted_grid_features(molecule)
        semilocal_channels = self.semilocal_energy_density_channels(features)
        semilocal = jnp.sum(semilocal_channels, axis=-1)
        return _pack_restricted_grid_payload(
            features,
            total_gradient,
            semilocal_channels,
            semilocal,
        )

    def restricted_grid_payload_for_molecule(
        self,
        molecule: Any,
    ) -> tuple[RestrictedFeatureBundle, Array, Array, Array]:
        return self._restricted_grid_payload(molecule)

    def prefer_direct_scf_fock_terms(self) -> bool:
        return True

    def _restricted_hfx_grid_contribution_components(
        self,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle,
        include_fxx: bool = False,
    ) -> tuple[Array, Array, Array, Array | None]:
        mode = self._ground_state_hf_mode_for_molecule(molecule)
        if mode == "off":
            zero = jnp.zeros_like(features.rho)
            return zero, zero, zero, None
        nu_source = hfx_nu_source(molecule)
        cached_fxx = getattr(molecule, "hfx_fxx", None)
        if mode == "frozen":
            nu_source = None
            cached_fxx = None
        elif mode == "scf" and nu_source is None and cached_fxx is None:
            raise ValueError(
                "ground_state_hf_mode='scf' requires molecule.hfx_nu or "
                "molecule.hfx_nu_api or molecule.hfx_fxx so the HF channel can be "
                "updated during SCF."
            )
        current_hfx_fxx = None
        if nu_source is None:
            hfx_local = getattr(molecule, "hfx_local", None)
            if hfx_local is None:
                if self._configured_ground_state_hf_mode() == "frozen":
                    raise ValueError(
                        "ground_state_hf_mode='frozen' requires molecule.hfx_local "
                        "so the HF channel remains fixed during SCF."
                    )
                hfx_local, _ = self._restricted_hfx_local_and_fxx(molecule)
            hfx_local = jnp.asarray(hfx_local)
            if hfx_local.ndim != 3 or hfx_local.shape[0] != 2:
                raise ValueError(
                    "local HF channel expects molecule.hfx_local with shape "
                    "(2, ngrids, n_omega)."
                )
            e_hf_a = jnp.asarray(hfx_local[0, :, 0])
            e_hf_b = jnp.asarray(hfx_local[1, :, 0])
            if include_fxx:
                if cached_fxx is not None:
                    current_hfx_fxx = jnp.asarray(cached_fxx)
        else:
            if getattr(molecule, "ao", None) is None:
                raise AttributeError("Molecule-like object must define ao.")
            ao = jnp.asarray(molecule.ao)
            n_omega, nu_ngrids, nao, nao2 = hfx_nu_shape(nu_source)
            if n_omega < 1:
                raise ValueError("HFX nu source must contain at least one omega channel.")
            if nu_ngrids != int(ao.shape[0]) or nu_ngrids != int(features.rho.shape[0]):
                raise ValueError(
                    "HFX nu source grid axis must match molecule grid features "
                    f"(got {nu_ngrids}, ao={ao.shape[0]}, features={features.rho.shape[0]})."
                )
            if nao != int(ao.shape[1]) or nao2 != int(ao.shape[1]):
                raise ValueError(
                    "HFX nu source AO dimensions must match molecule.ao second axis "
                    f"(got {(nao, nao2)} vs {(ao.shape[1], ao.shape[1])})."
                )
            dm_a, dm_b = self._restricted_spin_density_blocks(molecule)
            dm_a = jax.lax.stop_gradient(jnp.asarray(dm_a, dtype=ao.dtype))
            dm_b = jax.lax.stop_gradient(jnp.asarray(dm_b, dtype=ao.dtype))
            if is_chunked_hfx_nu(nu_source):
                chunk_size = min(
                    max(1, int(getattr(nu_source, "chunk_size", 512))),
                    int(ao.shape[0]),
                )
                n_chunks = (int(ao.shape[0]) + chunk_size - 1) // chunk_size

                def local_chunk(start: Array) -> Any:
                    ao_chunk = self._take_grid_chunk(ao, start, chunk_size, axis=0)
                    nu_chunk = hfx_nu_grid_chunk_padded(
                        nu_source,
                        start,
                        chunk_size,
                        n_omega=1,
                        dtype=ao.dtype,
                    )
                    e_a = jnp.einsum("gp,pq->gq", ao_chunk, dm_a, precision=Precision.HIGHEST)
                    e_b = jnp.einsum("gp,pq->gq", ao_chunk, dm_b, precision=Precision.HIGHEST)
                    fxx_a = jnp.einsum("wgbc,gc->wgb", nu_chunk, e_a, precision=Precision.HIGHEST)[0]
                    fxx_b = jnp.einsum("wgbc,gc->wgb", nu_chunk, e_b, precision=Precision.HIGHEST)[0]
                    exx_a = -0.5 * jnp.einsum("gq,gq->g", e_a, fxx_a, precision=Precision.HIGHEST)
                    exx_b = -0.5 * jnp.einsum("gq,gq->g", e_b, fxx_b, precision=Precision.HIGHEST)
                    if include_fxx:
                        return exx_a, exx_b, 0.5 * (fxx_a + fxx_b)
                    return exx_a, exx_b

                def body(_carry: None, chunk_idx: Array) -> tuple[None, Any]:
                    return None, local_chunk(chunk_idx * chunk_size)

                _, chunks = jax.lax.scan(body, None, jnp.arange(n_chunks))
                if include_fxx:
                    a_chunks, b_chunks, fxx_chunks = chunks
                    current_hfx_fxx = fxx_chunks.reshape(
                        n_chunks * chunk_size,
                        int(ao.shape[1]),
                    )[:nu_ngrids][None, :, :]
                else:
                    a_chunks, b_chunks = chunks
                e_hf_a = a_chunks.reshape(n_chunks * chunk_size)[:nu_ngrids]
                e_hf_b = b_chunks.reshape(n_chunks * chunk_size)[:nu_ngrids]
            else:
                nu = jnp.asarray(nu_source, dtype=ao.dtype)[:1]
                e_a = jnp.einsum("gp,pq->gq", ao, dm_a, precision=Precision.HIGHEST)
                e_b = jnp.einsum("gp,pq->gq", ao, dm_b, precision=Precision.HIGHEST)
                fxx_a = jnp.einsum("wgbc,gc->wgb", nu, e_a, precision=Precision.HIGHEST)[0]
                fxx_b = jnp.einsum("wgbc,gc->wgb", nu, e_b, precision=Precision.HIGHEST)[0]
                e_hf_a = -0.5 * jnp.einsum("gq,gq->g", e_a, fxx_a, precision=Precision.HIGHEST)
                e_hf_b = -0.5 * jnp.einsum("gq,gq->g", e_b, fxx_b, precision=Precision.HIGHEST)
                if include_fxx:
                    current_hfx_fxx = 0.5 * (fxx_a + fxx_b)[None, :, :]
            e_hf_a = jax.lax.stop_gradient(e_hf_a)
            e_hf_b = jax.lax.stop_gradient(e_hf_b)
            if current_hfx_fxx is not None:
                current_hfx_fxx = jax.lax.stop_gradient(current_hfx_fxx)

        e_hf = e_hf_a + e_hf_b
        return (
            jnp.nan_to_num(e_hf, nan=0.0, posinf=0.0, neginf=0.0),
            jnp.nan_to_num(e_hf_a, nan=0.0, posinf=0.0, neginf=0.0),
            jnp.nan_to_num(e_hf_b, nan=0.0, posinf=0.0, neginf=0.0),
            (
                None
                if current_hfx_fxx is None
                else jnp.nan_to_num(current_hfx_fxx, nan=0.0, posinf=0.0, neginf=0.0)
            ),
        )

    def _restricted_hfx_grid_contribution_components_no_fxx(
        self,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle,
    ) -> tuple[Array, Array, Array]:
        e_hf, e_hf_a, e_hf_b, _ = self._restricted_hfx_grid_contribution_components(
            molecule,
            features=features,
        )
        return e_hf, e_hf_a, e_hf_b

    def _restricted_scf_direct_fock_payload(
        self,
        params: PyTree,
        molecule: Any,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        features, total_gradient, semilocal_channels, semilocal = (
            self._restricted_grid_payload(molecule)
        )
        current_hfx_fxx = None
        if self._uses_hfx_channel():
            hf_projected, hf_projected_a, hf_projected_b, current_hfx_fxx = (
                self._restricted_hfx_grid_contribution_components(
                    molecule,
                    features=features,
                    include_fxx=self.uses_explicit_hfx_fock_for_scf(molecule),
                )
            )
        else:
            hf_projected = jnp.zeros_like(features.rho)
            hf_projected_a = hf_projected
            hf_projected_b = hf_projected
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
        coefficients = self.channel_coefficients(
            params,
            features,
            semilocal_energy_density=semilocal,
            hf_energy_density=hf_projected,
            pt2_energy_density=pt2_projected,
            hf_spin_energy_density=(hfx_feature_a, hfx_feature_b),
        )
        semilocal_basis = self._semilocal_local_contribution_channels(
            features,
            semilocal_channels,
        )
        basis = self._assemble_basis_channels(
            semilocal_basis,
            hf_projected=hf_projected,
            pt2_projected=pt2_projected,
        )
        local_xc = jnp.sum(self._assemble_channel_contributions(coefficients, basis), axis=-1)
        xc_energy = jnp.nan_to_num(
            jnp.tensordot(grid_weights, local_xc, axes=(0, 0)),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if self.uses_explicit_hfx_fock_for_scf(molecule):
            alpha = jnp.asarray(0.0, dtype=grid_weights.dtype)
        else:
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
            hfx_fxx=current_hfx_fxx,
        )
        alpha = self._alpha_for_scf_fock(
            alpha,
            uses_explicit_hfx_fock=uses_explicit_hfx_fock,
        )
        return (
            projected_vrho,
            projected_vgrad,
            projected_vtau,
            projected_vlapl,
            alpha,
            hfx_fock,
            xc_energy,
        )

    def scf_xc_fock_terms(
        self,
        params: PyTree,
        molecule: Any,
        *,
        weights: Array,
        functional_dtype: Any,
        vxc_clip: float,
    ) -> tuple[Array, Array, Array, Array]:
        projected_vrho, projected_vgrad, projected_vtau, projected_vlapl, alpha, hfx_fock, xc_energy = self._restricted_scf_direct_fock_payload(
            params,
            molecule,
        )
        v_rho = jnp.clip(
            jnp.nan_to_num(projected_vrho, nan=0.0, posinf=vxc_clip, neginf=-vxc_clip),
            -vxc_clip,
            vxc_clip,
        )
        v_grad = jnp.clip(
            jnp.nan_to_num(projected_vgrad, nan=0.0, posinf=vxc_clip, neginf=-vxc_clip),
            -vxc_clip,
            vxc_clip,
        )
        v_tau = jnp.nan_to_num(projected_vtau, nan=0.0, posinf=0.0, neginf=0.0)
        v_lapl = jnp.nan_to_num(projected_vlapl, nan=0.0, posinf=0.0, neginf=0.0)

        ao = jnp.asarray(molecule.ao, dtype=functional_dtype)
        kind = self._response_feature_kind_label()
        ao_deriv1 = getattr(molecule, "ao_deriv1", None)
        if ao_deriv1 is None:
            kind = "LDA"
            ao_deriv1_arr = jnp.zeros((4, ao.shape[0], ao.shape[1]), dtype=ao.dtype)
        else:
            ao_deriv1_arr = jnp.asarray(ao_deriv1, dtype=functional_dtype)
            if ao_deriv1_arr.shape[0] < 4 and kind in {"GGA", "MGGA", "MGGA_LAPL"}:
                kind = "LDA"
        ao_laplacian = getattr(molecule, "ao_laplacian", None)
        if ao_laplacian is None:
            ao_laplacian_arr = jnp.zeros_like(ao)
            if kind == "MGGA_LAPL":
                kind = "MGGA"
        else:
            ao_laplacian_arr = jnp.asarray(ao_laplacian, dtype=functional_dtype)

        from ..scf.rks import _vxc_matrix_from_grid_potential

        vxc_matrix = _vxc_matrix_from_grid_potential(
            ao=ao,
            ao_deriv1=ao_deriv1_arr,
            ao_laplacian=ao_laplacian_arr,
            weights=jnp.asarray(weights, dtype=functional_dtype),
            vxc_rho=jnp.asarray(v_rho, dtype=functional_dtype),
            vxc_grad=jnp.asarray(v_grad, dtype=functional_dtype),
            vxc_tau=jnp.asarray(v_tau, dtype=functional_dtype),
            vxc_lapl=jnp.asarray(v_lapl, dtype=functional_dtype),
            xc_kind=kind,
        )
        return vxc_matrix, alpha, hfx_fock, xc_energy

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
        hf_total = jnp.asarray(hf_projected)
        hfx_a = jnp.asarray(hf_feature_a)
        hfx_b = jnp.asarray(hf_feature_b)
        if hfx_a.ndim == hf_total.ndim:
            hfx_a = hfx_a[..., None]
        if hfx_b.ndim == hf_total.ndim:
            hfx_b = hfx_b[..., None]
        semilocal_basis = self._semilocal_local_contribution_channels(
            features,
            semilocal_channels,
        )
        pt2_basis = None if pt2_projected is None else jax.lax.stop_gradient(pt2_projected)
        basis = self._assemble_basis_channels(
            jax.lax.stop_gradient(semilocal_basis),
            hf_projected=jax.lax.stop_gradient(hf_total),
            pt2_projected=pt2_basis,
        )
        semilocal_total = jax.lax.stop_gradient(semilocal_total)
        pt2_input = None if pt2_projected is None else jax.lax.stop_gradient(pt2_projected)

        def energy_from_hfx_inputs(
            hf_total_arg: Array,
            hfx_a_arg: Array,
            hfx_b_arg: Array,
        ) -> tuple[Array, Array]:
            coefficients = self.channel_coefficients(
                params,
                features,
                semilocal_energy_density=semilocal_total,
                hf_energy_density=hf_total_arg,
                pt2_energy_density=pt2_input,
                hf_spin_energy_density=(hfx_a_arg, hfx_b_arg),
            )
            local_channels = self._assemble_channel_contributions(coefficients, basis)
            local_energy = jnp.sum(local_channels, axis=-1)
            energy = jnp.tensordot(weights, local_energy, axes=(0, 0))
            return energy, coefficients

        (_, coefficients), (grad_total, grad_a, grad_b) = jax.value_and_grad(
            energy_from_hfx_inputs,
            argnums=(0, 1, 2),
            has_aux=True,
        )(
            jax.lax.stop_gradient(hf_total),
            jax.lax.stop_gradient(hfx_a),
            jax.lax.stop_gradient(hfx_b),
        )
        direct_hfx_grad = weights * self._local_hf_fraction_from_coefficients(coefficients)
        total_input_grad = jnp.asarray(grad_total)
        grad_a = jnp.asarray(grad_a)
        grad_b = jnp.asarray(grad_b)
        if grad_a.ndim == total_input_grad.ndim:
            grad_a = grad_a[..., None]
        if grad_b.ndim == total_input_grad.ndim:
            grad_b = grad_b[..., None]
        first_channel_grad = direct_hfx_grad + total_input_grad
        grad_a = grad_a.at[..., 0].add(first_channel_grad)
        grad_b = grad_b.at[..., 0].add(first_channel_grad)
        grad_a = self._maybe_clip_response(grad_a)
        grad_b = self._maybe_clip_response(grad_b)
        return grad_a, grad_b

    def _zero_hfx_fock(self, molecule: Any, dtype: Any | None = None) -> Array:
        ao = jnp.asarray(molecule.ao)
        matrix_dtype = ao.dtype if dtype is None else dtype
        return jnp.zeros((ao.shape[1], ao.shape[1]), dtype=matrix_dtype)

    def _response_hf_grid_contribution_components(
        self,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle,
    ) -> tuple[Array, Array, Array]:
        if not self._uses_hfx_channel():
            zero = jnp.zeros_like(features.rho)
            return zero, zero, zero

        hfx_local = getattr(molecule, "hfx_local", None)
        if hfx_local is None:
            zero = jnp.zeros_like(features.rho)
            return zero, zero, zero

        hfx_local = jnp.asarray(hfx_local)
        if hfx_local.ndim != 3 or hfx_local.shape[0] != 2:
            raise ValueError(
                "local HF channel expects molecule.hfx_local with shape "
                "(2, ngrids, n_omega)."
            )
        e_hf_a = jnp.nan_to_num(hfx_local[0, :, 0], nan=0.0, posinf=0.0, neginf=0.0)
        e_hf_b = jnp.nan_to_num(hfx_local[1, :, 0], nan=0.0, posinf=0.0, neginf=0.0)
        e_hf = jnp.nan_to_num(e_hf_a + e_hf_b, nan=0.0, posinf=0.0, neginf=0.0)
        return e_hf, e_hf_a, e_hf_b

    def uses_explicit_hfx_fock_for_scf(self, molecule: Any) -> bool:
        if self._ground_state_hf_mode_for_molecule(molecule) != "scf":
            return False
        return (
            self._uses_hfx_channel()
            and (
                has_hfx_nu_source(molecule)
                or getattr(molecule, "hfx_fxx", None) is not None
            )
        )

    def _contract_hfx_feature_gradients_to_restricted_fock(
        self,
        molecule: Any,
        grad_a: Array,
        grad_b: Array,
        *,
        dtype: Any | None = None,
        hfx_fxx: Array | None = None,
    ) -> tuple[Array, bool]:
        nu_source = hfx_nu_source(molecule)
        fxx_source = hfx_fxx
        if fxx_source is None and nu_source is None:
            fxx_source = getattr(molecule, "hfx_fxx", None)
        if nu_source is None and fxx_source is None:
            return self._zero_hfx_fock(molecule, dtype), False

        ao = jnp.asarray(molecule.ao)
        matrix_dtype = ao.dtype if dtype is None else dtype
        ao = jnp.asarray(ao, dtype=matrix_dtype)
        if fxx_source is not None:
            fxx_source = jnp.asarray(fxx_source, dtype=matrix_dtype)
            if fxx_source.ndim != 3:
                raise ValueError(
                    "HFX fxx cache must have shape (n_omega, ngrids, nao), "
                    f"got {fxx_source.shape}."
                )
            n_omega, ngrid, nao = (int(dim) for dim in fxx_source.shape)
            nao2 = nao
        else:
            n_omega, ngrid, nao, nao2 = hfx_nu_shape(nu_source)
        if nao != ao.shape[1] or nao2 != ao.shape[1]:
            raise ValueError(
                "HFX source AO dimensions must match molecule.ao second axis "
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
            if fxx_source is not None and int(n_omega) == 1:
                fxx_source = jnp.repeat(fxx_source, n_grad_channels, axis=0)
                n_omega = n_grad_channels
            else:
                raise ValueError(
                    "HFX feature gradient omega axis cannot exceed HFX source omega axis "
                    f"(got {n_grad_channels} vs {n_omega})."
                )
        grad = 0.5 * (grad_a[:, :n_grad_channels] + grad_b[:, :n_grad_channels])
        grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

        if fxx_source is not None:
            fxx = fxx_source[:n_grad_channels]
            aow = -0.5 * fxx * jnp.transpose(grad, (1, 0))[:, :, None]
            vmat = jnp.einsum("gp,wgq->pq", ao, aow, precision=Precision.HIGHEST)
        else:
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
            density_half = jax.lax.stop_gradient(density_half)
            if is_chunked_hfx_nu(nu_source):
                chunk_size = min(
                    max(1, int(getattr(nu_source, "chunk_size", 512))),
                    int(ngrid),
                )
                n_chunks = (int(ngrid) + chunk_size - 1) // chunk_size
                zero = jnp.zeros((ao.shape[1], ao.shape[1]), dtype=matrix_dtype)

                def vmat_from_feature_gradients(grad_grid: Array) -> Array:
                    def vmat_chunk_from_start(start: Array) -> Array:
                        ao_chunk = self._take_grid_chunk(ao, start, chunk_size, axis=0)
                        grad_chunk = self._take_grid_chunk(
                            grad_grid,
                            start,
                            chunk_size,
                            axis=0,
                        )
                        nu_chunk = hfx_nu_grid_chunk_padded(
                            nu_source,
                            start,
                            chunk_size,
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
                        return jnp.einsum(
                            "gp,wgq->pq",
                            ao_chunk,
                            aow,
                            precision=Precision.HIGHEST,
                        )

                    def body(carry: Array, chunk_idx: Array) -> tuple[Array, None]:
                        start = chunk_idx * chunk_size
                        return carry + vmat_chunk_from_start(start), None

                    vmat_out, _ = jax.lax.scan(body, zero, jnp.arange(n_chunks))
                    correction_out = vmat_out + vmat_out.T
                    return jnp.nan_to_num(
                        correction_out,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )

                @jax.custom_vjp
                def hfx_fock_from_feature_gradients(grad_grid: Array) -> Array:
                    return vmat_from_feature_gradients(grad_grid)

                def hfx_fock_fwd(grad_grid: Array) -> tuple[Array, None]:
                    return vmat_from_feature_gradients(grad_grid), None

                def hfx_fock_bwd(_residual: None, cotangent: Array) -> tuple[Array]:
                    cotangent = jnp.asarray(cotangent, dtype=matrix_dtype)
                    vmat_cotangent = cotangent + cotangent.T

                    def grad_chunk_from_start(start: Array) -> Array:
                        ao_chunk = self._take_grid_chunk(ao, start, chunk_size, axis=0)
                        nu_chunk = hfx_nu_grid_chunk_padded(
                            nu_source,
                            start,
                            chunk_size,
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
                        grad_chunk = -0.5 * jnp.einsum(
                            "pq,gp,wgq->gw",
                            vmat_cotangent,
                            ao_chunk,
                            fxx,
                            precision=Precision.HIGHEST,
                        )
                        return grad_chunk

                    def body(_carry: None, chunk_idx: Array) -> tuple[None, Array]:
                        start = chunk_idx * chunk_size
                        return None, grad_chunk_from_start(start)

                    _, grad_chunks = jax.lax.scan(body, None, jnp.arange(n_chunks))
                    grad_grid = grad_chunks.reshape(
                        n_chunks * chunk_size,
                        n_grad_channels,
                    )[:ngrid]
                    return (grad_grid,)

                hfx_fock_from_feature_gradients.defvjp(hfx_fock_fwd, hfx_fock_bwd)
                return hfx_fock_from_feature_gradients(grad), True
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
        hfx_fxx: Array | None = None,
    ) -> tuple[Array, bool]:
        if (
            not self._uses_hfx_channel()
            or (
                not has_hfx_nu_source(molecule)
                and hfx_fxx is None
                and getattr(molecule, "hfx_fxx", None) is None
            )
        ):
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
            hfx_fxx=hfx_fxx,
        )

    def _explicit_hfx_fock_from_molecule(
        self,
        params: PyTree,
        molecule: Any,
    ) -> tuple[Array, bool]:
        if not self.uses_explicit_hfx_fock_for_scf(molecule):
            return self._zero_hfx_fock(molecule), False
        features = grid_features_for_molecule(molecule)
        semilocal_channels = self.semilocal_energy_density_channels(features)
        hf_projected, hf_projected_a, hf_projected_b, current_hfx_fxx = self._restricted_hfx_grid_contribution_components(
            molecule,
            features=features,
            include_fxx=True,
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
            hfx_fxx=current_hfx_fxx,
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
        features, total_gradient = self._restricted_grid_features(molecule)
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
        features, total_gradient = self._restricted_grid_features(molecule)
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
        features, total_gradient, semilocal_channels, semilocal = (
            self._restricted_grid_payload(molecule)
        )
        if self._uses_hfx_channel():
            hf_projected, hf_projected_a, hf_projected_b = (
                self._restricted_hfx_grid_contribution_components_no_fxx(
                    molecule,
                    features=features,
                )
            )
        else:
            hf_projected = jnp.zeros_like(features.rho)
            hf_projected_a = hf_projected
            hf_projected_b = hf_projected
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

        post_tda_correction_fn, post_tddft_correction_fn = (
            self._strict_pt2_posthoc_correction_callbacks(
                features.rho,
                semilocal_channels,
                coefficients,
                pt2_projected,
                molecule.grid.weights,
            )
        )
        response_alpha = alpha if self._uses_hfx_channel() else 0.0
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
            response_hf_mode=self._response_hf_mode() if self._uses_hfx_channel() else "approx",
            grid_response_tensor_fn=grid_response_tensor_fn,
            nonlocal_response_diagonal_fn=None,
            post_tda_correction_fn=post_tda_correction_fn,
            post_tddft_correction_fn=post_tddft_correction_fn,
        )

    def bind_to_molecule_for_response(
        self,
        params: PyTree,
        molecule: Any,
    ) -> BoundNeuralXCFunctional:
        """TD-response-only binding that avoids assembling strict potential terms."""

        response_hf_mode = self._response_hf_mode()
        if self._uses_hfx_channel() and response_hf_mode == "strict":
            raise NotImplementedError(
                "strict local-HF TDDFT response requires chi/fxx-based second-response "
                "contractions and is not implemented. Use response_hf_mode='approx'."
            )

        if _requires_unrestricted_response_binding(molecule):
            features, grad_a, grad_b = grid_features_with_spin_gradients_for_molecule(molecule)
            semilocal_channels = self.semilocal_energy_density_channels(features)
            semilocal = jnp.sum(semilocal_channels, axis=-1)
            needs_response_alpha = self._uses_hfx_channel()
            if needs_response_alpha:
                hf_projected, hf_projected_a, hf_projected_b = (
                    self._response_hf_grid_contribution_components(
                        molecule,
                        features=features,
                    )
                )
            else:
                hf_projected = jnp.zeros_like(features.rho)
                hf_projected_a = hf_projected
                hf_projected_b = hf_projected
            if needs_response_alpha and self.input_feature_mode == "canonical":
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
            needs_pt2_posthoc = (
                self.include_pt2_channel
                and self.response_pt2_mode == "strict"
                and pt2_projected is not None
            )
            coefficients = None
            alpha = jnp.asarray(0.0, dtype=semilocal.dtype)
            if needs_response_alpha or needs_pt2_posthoc:
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
            if needs_response_alpha and coefficients is not None:
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
                    response_pt2_mode=self.response_pt2_mode,
                )

            response_alpha = alpha if needs_response_alpha else 0.0
            if needs_pt2_posthoc and coefficients is not None:
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
            else:
                post_tda_correction_fn = post_tddft_correction_fn = None
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
                response_hf_mode=response_hf_mode if self._uses_hfx_channel() else "approx",
                grid_response_tensor_fn=None,
                spin_local_kernel_fn=spin_local_kernel_fn,
                post_tda_correction_fn=post_tda_correction_fn,
                post_tddft_correction_fn=post_tddft_correction_fn,
            )

        features, total_gradient, semilocal_channels, semilocal = (
            self._restricted_grid_payload(molecule)
        )
        needs_response_alpha = self._uses_hfx_channel()
        if needs_response_alpha:
            hf_projected, hf_projected_a, hf_projected_b = self._response_hf_grid_contribution_components(
                molecule,
                features=features,
            )
        else:
            hf_projected = jnp.zeros_like(features.rho)
            hf_projected_a = hf_projected
            hf_projected_b = hf_projected
        if needs_response_alpha and self.input_feature_mode == "canonical":
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
        needs_pt2_posthoc = (
            self.include_pt2_channel
            and self.response_pt2_mode == "strict"
            and pt2_projected is not None
        )
        coefficients = None
        alpha = jnp.asarray(0.0, dtype=semilocal.dtype)
        if needs_response_alpha or needs_pt2_posthoc:
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
            if needs_response_alpha:
                hf_field = self._local_hf_fraction_from_coefficients(coefficients)
                rho = jnp.maximum(features.rho, self.density_floor)
                grid_weights = jnp.asarray(molecule.grid.weights)
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
        def grid_response_hvp_fn(
            response_molecule: Any,
            tangent: Array,
        ) -> Array:
            del response_molecule
            return self._strict_total_response_hvp(
                params,
                features,
                total_gradient,
                hf_projected,
                tangent,
                pt2_projected=pt2_projected,
                hf_spin_energy_density=(hfx_feature_a, hfx_feature_b),
                response_pt2_mode=self.response_pt2_mode,
                strict_payload=strict_payload,
            )

        if needs_pt2_posthoc and coefficients is not None:
            (
                post_tda_correction_fn,
                post_tddft_correction_fn,
            ) = self._strict_pt2_posthoc_correction_callbacks(
                features.rho,
                semilocal_channels,
                coefficients,
                pt2_projected,
                molecule.grid.weights,
            )
        else:
            post_tda_correction_fn = post_tddft_correction_fn = None
        # TD response uses the configured action path and avoids strict
        # potential/energy assembly.
        response_alpha = alpha if needs_response_alpha else 0.0
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
            response_hf_mode=response_hf_mode if self._uses_hfx_channel() else "approx",
            grid_response_tensor_fn=None,
            grid_response_hvp_fn=grid_response_hvp_fn,
            spin_local_kernel_fn=None,
            post_tda_correction_fn=post_tda_correction_fn,
            post_tddft_correction_fn=post_tddft_correction_fn,
        )

    def _unrestricted_scf_binding_payload(
        self,
        params: PyTree,
        molecule: Any,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        features, grad_a, grad_b = grid_features_with_spin_gradients_for_molecule(molecule)
        semilocal_channels = self.semilocal_energy_density_channels(features)
        semilocal = jnp.sum(semilocal_channels, axis=-1)
        if self._uses_hfx_channel():
            hf_projected, hf_projected_a, hf_projected_b = self.projected_hf_grid_contribution_components(
                molecule,
                features=features,
            )
        else:
            hf_projected = jnp.zeros_like(features.rho)
            hf_projected_a = hf_projected
            hf_projected_b = hf_projected
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

        projected_vrho, projected_vgrad, projected_vtau, projected_vlapl, alpha, hfx_fock, _ = self._restricted_scf_direct_fock_payload(
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
                _,
            ) = self._restricted_scf_direct_fock_payload(params, molecule)
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
            response_hf_mode=self._response_hf_mode() if self._uses_hfx_channel() else "approx",
            grid_response_tensor_fn=None,
            spin_local_kernel_fn=None,
        )
