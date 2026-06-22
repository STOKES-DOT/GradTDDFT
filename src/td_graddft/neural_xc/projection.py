from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array, PyTree

from ..data.integrals import eri_pair_matrix_to_mo_eri_slices
from ..data.integrals.jax.packed_eri import _metadata_arrays, _mo_pair_products
from ..features import (
    grid_features_for_molecule,
)
from .inputs import (
    _local_pt2_feature_from_unrestricted_orbitals,
    has_hfx_nu_source,
    hfx_nu_grid_chunk_padded,
    hfx_nu_shape,
    hfx_nu_source,
    is_chunked_hfx_nu,
)
from ..xc_backend.jax_libxc import RestrictedFeatureBundle


class NeuralXCProjectionMixin:
    @staticmethod
    def _uses_unrestricted_pt2_projection(molecule: Any) -> bool:
        return (
            getattr(molecule, "nocc_alpha", None) is not None
            or getattr(molecule, "nocc_beta", None) is not None
        )

    def scf_molecule_with_density(self, molecule: Any, density: Array) -> Any:
        """Return a restricted molecule view with a new spin-summed density."""

        density_arr = jnp.asarray(density)
        if density_arr.ndim == 2:
            rdm1 = jnp.stack([0.5 * density_arr, 0.5 * density_arr], axis=0)
        elif density_arr.ndim == 3 and density_arr.shape[0] == 2:
            rdm1 = density_arr
        else:
            raise ValueError(
                "SCF density callback expects density with shape (nao, nao) "
                "or spin density with shape (2, nao, nao)."
            )

        updates: dict[str, Any] = {"rdm1": rdm1}
        if hasattr(molecule, "neural_xc_grid_payload"):
            updates["neural_xc_grid_payload"] = None
        if self._uses_hfx_channel():
            for name in ("hfx_nu", "hfx_nu_api"):
                if hasattr(molecule, name):
                    updates[name] = getattr(molecule, name)
            nu_source = hfx_nu_source(molecule)
            if hasattr(molecule, "hfx_local") and nu_source is not None:
                updates["hfx_local"] = None
            if hasattr(molecule, "hfx_fxx"):
                if nu_source is not None:
                    updates["hfx_fxx"] = None
                else:
                    fxx_ref = getattr(molecule, "hfx_fxx", None)
                    if fxx_ref is not None:
                        updates["hfx_fxx"] = jax.lax.stop_gradient(jnp.asarray(fxx_ref))
        if is_dataclass(molecule):
            molecule_fields = {field.name for field in fields(molecule)}
            replace_updates = {
                key: value for key, value in updates.items() if key in molecule_fields
            }
            molecule_out = replace(molecule, **replace_updates)
            for key, value in updates.items():
                if key not in molecule_fields:
                    setattr(molecule_out, key, value)
            return molecule_out

        molecule_out = copy.copy(molecule)
        for key, value in updates.items():
            setattr(molecule_out, key, value)
        return molecule_out

    def _restricted_spin_density_blocks(self, molecule: Any) -> Array:
        if getattr(molecule, "rdm1", None) is None:
            raise AttributeError("Molecule-like object must define rdm1.")
        rdm1 = jnp.asarray(molecule.rdm1)
        if rdm1.ndim == 2:
            return jnp.stack([0.5 * rdm1, 0.5 * rdm1], axis=0)
        if rdm1.ndim != 3:
            raise ValueError(
                "Restricted HF/PT2 channels expect rdm1 to have shape "
                "(nao, nao) or (spin, nao, nao)."
            )
        if rdm1.shape[0] == 1:
            return jnp.concatenate([rdm1, rdm1], axis=0)
        if rdm1.shape[0] != 2:
            raise ValueError(
                "Restricted HF/PT2 channels expect one or two spin blocks in rdm1."
            )
        return rdm1

    def _restricted_hfx_local_and_fxx(
        self,
        molecule: Any,
    ) -> tuple[Array, Array | None]:
        nu_source = hfx_nu_source(molecule)
        hfx_local = getattr(molecule, "hfx_local", None)
        hfx_fxx = getattr(molecule, "hfx_fxx", None)
        if nu_source is None:
            if hfx_local is None:
                raise AttributeError(
                    "local HF channel requires molecule.hfx_local, molecule.hfx_nu, "
                    "molecule.hfx_nu_api, or molecule.hfx_fxx."
                )
            return jnp.asarray(hfx_local), (
                None if hfx_fxx is None else jnp.asarray(hfx_fxx)
            )

        if getattr(molecule, "ao", None) is None:
            raise AttributeError("Molecule-like object must define ao.")
        ao = jnp.asarray(molecule.ao)
        n_omega, nu_ngrids, nao, nao2 = hfx_nu_shape(nu_source)
        if n_omega < 1:
            raise ValueError("HFX nu source must contain at least one omega channel.")
        if nu_ngrids != int(ao.shape[0]):
            raise ValueError(
                "HFX nu source grid axis must match molecule.ao first axis "
                f"(got {nu_ngrids} vs {ao.shape[0]})."
            )
        if nao != int(ao.shape[1]) or nao2 != int(ao.shape[1]):
            raise ValueError(
                "HFX nu source AO dimensions must match molecule.ao second axis "
                f"(got {(nao, nao2)} vs {(ao.shape[1], ao.shape[1])})."
            )
        dm_a, dm_b = self._restricted_spin_density_blocks(molecule)
        if is_chunked_hfx_nu(nu_source):
            chunk_size = min(
                max(1, int(getattr(nu_source, "chunk_size", 512))),
                int(ao.shape[0]),
            )
            n_chunks = (int(ao.shape[0]) + chunk_size - 1) // chunk_size

            def hfx_chunk(start: Array) -> tuple[Array, Array]:
                ao_chunk = self._take_grid_chunk(ao, start, chunk_size, axis=0)
                nu_chunk = hfx_nu_grid_chunk_padded(
                    nu_source,
                    start,
                    chunk_size,
                    dtype=ao.dtype,
                )
                e_a = jnp.einsum("gp,pq->gq", ao_chunk, dm_a, precision=Precision.HIGHEST)
                e_b = jnp.einsum("gp,pq->gq", ao_chunk, dm_b, precision=Precision.HIGHEST)
                fxx_a = jnp.einsum("wgbc,gc->wgb", nu_chunk, e_a, precision=Precision.HIGHEST)
                fxx_b = jnp.einsum("wgbc,gc->wgb", nu_chunk, e_b, precision=Precision.HIGHEST)
                exx_a = -0.5 * jnp.einsum("gq,wgq->wg", e_a, fxx_a, precision=Precision.HIGHEST)
                exx_b = -0.5 * jnp.einsum("gq,wgq->wg", e_b, fxx_b, precision=Precision.HIGHEST)
                return jnp.stack([exx_a.T, exx_b.T], axis=0), 0.5 * (fxx_a + fxx_b)

            def body(_carry: None, chunk_idx: Array) -> tuple[None, tuple[Array, Array]]:
                return None, hfx_chunk(chunk_idx * chunk_size)

            _, (local_chunks, fxx_chunks) = jax.lax.scan(body, None, jnp.arange(n_chunks))
            hfx_local = jnp.transpose(local_chunks, (1, 0, 2, 3)).reshape(
                2,
                n_chunks * chunk_size,
                n_omega,
            )[:, :nu_ngrids]
            hfx_fxx = jnp.transpose(fxx_chunks, (1, 0, 2, 3)).reshape(
                n_omega,
                n_chunks * chunk_size,
                nao,
            )[:, :nu_ngrids]
            hfx_local = jax.lax.stop_gradient(hfx_local)
            hfx_fxx = jax.lax.stop_gradient(hfx_fxx)
            return hfx_local, hfx_fxx

        else:
            nu = jnp.asarray(nu_source, dtype=ao.dtype)
        e_a = jnp.einsum("gp,pq->gq", ao, dm_a, precision=Precision.HIGHEST)
        e_b = jnp.einsum("gp,pq->gq", ao, dm_b, precision=Precision.HIGHEST)
        fxx_a = jnp.einsum("wgbc,gc->wgb", nu, e_a, precision=Precision.HIGHEST)
        fxx_b = jnp.einsum("wgbc,gc->wgb", nu, e_b, precision=Precision.HIGHEST)
        exx_a = -0.5 * jnp.einsum(
            "gq,wgq->wg",
            e_a,
            fxx_a,
            precision=Precision.HIGHEST,
        )
        exx_b = -0.5 * jnp.einsum(
            "gq,wgq->wg",
            e_b,
            fxx_b,
            precision=Precision.HIGHEST,
        )
        hfx_local = jnp.stack([exx_a.T, exx_b.T], axis=0)
        hfx_fxx = 0.5 * (fxx_a + fxx_b)
        hfx_local = jax.lax.stop_gradient(hfx_local)
        hfx_fxx = jax.lax.stop_gradient(hfx_fxx)
        return hfx_local, hfx_fxx

    def _restricted_hfx_grid_contribution_components_and_fxx(
        self,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
    ) -> tuple[Array, Array, Array, Array | None]:
        del features
        hfx_local, hfx_fxx = self._restricted_hfx_local_and_fxx(molecule)
        if hfx_local.ndim != 3 or hfx_local.shape[0] != 2:
            raise ValueError(
                "local HF channel expects molecule.hfx_local with shape "
                "(2, ngrids, n_omega)."
            )
        e_hf_a = jnp.asarray(hfx_local[0, :, 0])
        e_hf_b = jnp.asarray(hfx_local[1, :, 0])
        e_hf = e_hf_a + e_hf_b
        e_hf = jnp.nan_to_num(e_hf, nan=0.0, posinf=0.0, neginf=0.0)
        e_hf_a = jnp.nan_to_num(e_hf_a, nan=0.0, posinf=0.0, neginf=0.0)
        e_hf_b = jnp.nan_to_num(e_hf_b, nan=0.0, posinf=0.0, neginf=0.0)
        return e_hf, e_hf_a, e_hf_b, hfx_fxx

    def _exact_hf_grid_contribution_components(
        self,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
    ) -> tuple[Array, Array, Array]:
        no_fxx = getattr(self, "_restricted_hfx_grid_contribution_components_no_fxx", None)
        if callable(no_fxx) and features is not None:
            return no_fxx(molecule, features=features)
        e_hf, e_hf_a, e_hf_b, _ = self._restricted_hfx_grid_contribution_components_and_fxx(
            molecule,
            features=features,
        )
        return e_hf, e_hf_a, e_hf_b

    def projected_hf_grid_contribution_components(
        self,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
    ) -> tuple[Array, Array, Array]:
        if features is None:
            no_fxx = getattr(self, "_restricted_hfx_grid_contribution_components_no_fxx", None)
            if callable(no_fxx):
                features = grid_features_for_molecule(molecule)
        return self._exact_hf_grid_contribution_components(
            molecule,
            features=features,
        )

    def projected_hf_energy_density_components(
        self,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
    ) -> tuple[Array, Array, Array]:
        """Compatibility wrapper returning per-particle HF energy densities.

        Neural XC uses local grid contributions directly. This helper exposes the old
        epsilon-style view by dividing out the corresponding spin densities.
        """

        if features is None:
            features = grid_features_for_molecule(molecule)
        e_hf, e_hf_a, e_hf_b = self.projected_hf_grid_contribution_components(
            molecule,
            features=features,
        )
        rho = jnp.maximum(features.rho, self.density_floor)
        rho_a = jnp.maximum(features.rho_a, self.density_floor)
        rho_b = jnp.maximum(features.rho_b, self.density_floor)
        eps_hf = jnp.nan_to_num(e_hf / rho, nan=0.0, posinf=0.0, neginf=0.0)
        eps_hf_a = jnp.nan_to_num(e_hf_a / rho_a, nan=0.0, posinf=0.0, neginf=0.0)
        eps_hf_b = jnp.nan_to_num(e_hf_b / rho_b, nan=0.0, posinf=0.0, neginf=0.0)
        return eps_hf, eps_hf_a, eps_hf_b

    def projected_hf_energy_density(
        self,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
    ) -> Array:
        eps_hf, _, _ = self.projected_hf_energy_density_components(
            molecule,
            features=features,
        )
        return eps_hf

    def _restricted_mp2_projection_components(
        self,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
        occupation_tolerance: float = 1e-8,
        cached_local: Array | None = None,
    ) -> tuple[Array, Array]:
        """Restricted closed-shell MP2 local pair gauge and canonical total energy."""
        del features
        rep_tensor = getattr(molecule, "rep_tensor", None)
        eri_pair_matrix = getattr(molecule, "eri_pair_matrix", None)
        if rep_tensor is None and eri_pair_matrix is None:
            raise AttributeError("Molecule-like object must define rep_tensor or eri_pair_matrix.")
        if getattr(molecule, "mo_coeff", None) is None:
            raise AttributeError("Molecule-like object must define mo_coeff.")
        if getattr(molecule, "mo_occ", None) is None:
            raise AttributeError("Molecule-like object must define mo_occ.")
        if getattr(molecule, "mo_energy", None) is None:
            raise AttributeError("Molecule-like object must define mo_energy.")
        if getattr(molecule, "ao", None) is None:
            raise AttributeError("Molecule-like object must define ao.")
        if getattr(molecule, "grid", None) is None:
            raise AttributeError("Molecule-like object must define grid.weights.")

        rep_tensor = None if rep_tensor is None else jnp.asarray(rep_tensor)
        eri_pair_matrix = None if eri_pair_matrix is None else jnp.asarray(eri_pair_matrix)
        has_pair_matrix = eri_pair_matrix is not None and eri_pair_matrix.size != 0
        ao = jnp.asarray(molecule.ao)
        mo_coeff = jnp.asarray(molecule.mo_coeff)
        mo_occ = jnp.asarray(molecule.mo_occ)
        mo_energy = jnp.asarray(molecule.mo_energy)

        if mo_coeff.ndim == 3:
            mo_coeff = mo_coeff[0]
        if mo_occ.ndim == 2:
            mo_occ = mo_occ[0]
        if mo_energy.ndim == 2:
            mo_energy = mo_energy[0]

        nocc = getattr(molecule, "nocc", None)
        if nocc is None:
            nocc = int(jnp.count_nonzero(mo_occ > occupation_tolerance))
        else:
            nocc = int(nocc)
        nmo = int(mo_coeff.shape[1])
        if nocc <= 0 or nocc >= nmo:
            raise ValueError("Restricted MP2 projection requires at least one occupied and one virtual.")

        orbo = mo_coeff[:, :nocc]
        orbv = mo_coeff[:, nocc:]
        eps_occ = mo_energy[:nocc]
        eps_vir = mo_energy[nocc:]

        eri_ovov = getattr(molecule, "eri_ovov", None)
        if eri_ovov is None:
            if has_pair_matrix:
                eri_ovov, _, _ = eri_pair_matrix_to_mo_eri_slices(
                    eri_pair_matrix,
                    mo_coeff,
                    nocc=nocc,
                    include_oovv=False,
                )
            else:
                eri_ovov = jnp.einsum(
                    "pqrs,pi,qa,rj,sb->iajb",
                    rep_tensor,
                    orbo,
                    orbv,
                    orbo,
                    orbv,
                    precision=Precision.HIGHEST,
                )
        else:
            eri_ovov = jnp.asarray(eri_ovov)

        denom = (
            eps_occ[:, None, None, None]
            + eps_occ[None, None, :, None]
            - eps_vir[None, :, None, None]
            - eps_vir[None, None, None, :]
        )
        denom = jnp.where(jnp.abs(denom) > self.density_floor, denom, -self.density_floor)
        direct = eri_ovov
        exchange = jnp.transpose(eri_ovov, (0, 3, 2, 1))
        pair_weights = (2.0 * direct - exchange) / denom
        total_energy = jnp.sum(direct * pair_weights)
        if cached_local is None:
            rho_o = jnp.einsum("rp,pi->ri", ao, orbo, precision=Precision.HIGHEST)
            rho_v = jnp.einsum("rp,pa->ra", ao, orbv, precision=Precision.HIGHEST)
            rho_ov = jnp.einsum("ri,ra->ria", rho_o, rho_v, precision=Precision.HIGHEST)
            if has_pair_matrix:
                rows, cols, _, multiplicity = _metadata_arrays(int(mo_coeff.shape[0]), ao.dtype)
                grid_pair = ao[:, rows] * ao[:, cols] * multiplicity[None, :]
                ov = _mo_pair_products(orbo, orbv, rows, cols)
                pair_potential = jnp.einsum(
                    "gP,PQ,jbQ->gjb",
                    grid_pair,
                    eri_pair_matrix,
                    ov,
                    precision=Precision.HIGHEST,
                )
            else:
                pair_potential = jnp.einsum(
                    "gp,gq,pqrs,rj,sb->gjb",
                    ao,
                    ao,
                    rep_tensor,
                    orbo,
                    orbv,
                    precision=Precision.HIGHEST,
                )
            local_energy = jnp.einsum(
                "ria,rjb,iajb->r",
                rho_ov,
                pair_potential,
                pair_weights,
                precision=Precision.HIGHEST,
            )
        else:
            local_energy = jnp.asarray(cached_local)
        local_energy = jnp.nan_to_num(local_energy, nan=0.0, posinf=0.0, neginf=0.0)
        total_energy = jnp.nan_to_num(total_energy, nan=0.0, posinf=0.0, neginf=0.0)
        return local_energy, total_energy

    def _unrestricted_mp2_projection_components(
        self,
        molecule: Any,
        *,
        occupation_tolerance: float = 1e-8,
        cached_local: Array | None = None,
    ) -> tuple[Array, Array]:
        local_energy, total_energy = _local_pt2_feature_from_unrestricted_orbitals(
            molecule.ao,
            molecule.mo_coeff,
            molecule.mo_occ,
            molecule.mo_energy,
            rep_tensor=getattr(molecule, "rep_tensor", None),
            eri_pair_matrix=getattr(molecule, "eri_pair_matrix", None),
            df_factors=getattr(molecule, "df_factors", None),
            occupation_tolerance=occupation_tolerance,
            density_floor=self.density_floor,
            return_total_energy=True,
        )
        if cached_local is not None:
            local_energy = jnp.asarray(cached_local)
        local_energy = jnp.nan_to_num(local_energy, nan=0.0, posinf=0.0, neginf=0.0)
        total_energy = jnp.nan_to_num(total_energy, nan=0.0, posinf=0.0, neginf=0.0)
        return local_energy, total_energy

    def _local_exact_pt2_grid_contribution(
        self,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
        occupation_tolerance: float = 1e-8,
    ) -> Array:
        """Local MP2 pair gauge without global rescaling."""
        cached = getattr(molecule, "pt2_local", None)
        if cached is not None:
            cached_arr = jnp.asarray(cached)
            return jnp.nan_to_num(cached_arr, nan=0.0, posinf=0.0, neginf=0.0)
        if self._uses_unrestricted_pt2_projection(molecule):
            local_energy, _ = self._unrestricted_mp2_projection_components(
                molecule,
                occupation_tolerance=occupation_tolerance,
            )
        else:
            local_energy, _ = self._restricted_mp2_projection_components(
                molecule,
                features=features,
                occupation_tolerance=occupation_tolerance,
            )
        return local_energy

    def _legacy_projected_pt2_grid_contribution(
        self,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
        occupation_tolerance: float = 1e-8,
    ) -> Array:
        if getattr(molecule, "grid", None) is None:
            raise AttributeError("Molecule-like object must define grid.weights.")
        weights = jnp.asarray(molecule.grid.weights)
        cached = getattr(molecule, "pt2_local", None)
        if self._uses_unrestricted_pt2_projection(molecule):
            projected, total_energy = self._unrestricted_mp2_projection_components(
                molecule,
                occupation_tolerance=occupation_tolerance,
                cached_local=None if cached is None else jnp.asarray(cached),
            )
        else:
            projected, total_energy = self._restricted_mp2_projection_components(
                molecule,
                features=features,
                occupation_tolerance=occupation_tolerance,
                cached_local=None if cached is None else jnp.asarray(cached),
            )
        projected_energy = jnp.tensordot(weights, projected, axes=(0, 0))
        scale = jnp.where(
            jnp.abs(projected_energy) > self.density_floor,
            total_energy / projected_energy,
            0.0,
        )
        projected = scale * projected
        projected = jnp.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)
        return self._maybe_clip_response(projected)

    def projected_pt2_grid_contribution(
        self,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
        occupation_tolerance: float = 1e-8,
    ) -> Array:
        """Return the configured PT2 local channel.

        `scaled_projected` reproduces the legacy behavior: rescale the local
        pair gauge so its weighted grid integral matches the canonical MP2
        correlation energy, then optionally clip it.

        `local_exact` keeps the raw local pair gauge without global rescaling
        or clipping. On finite grids this generally does not integrate exactly
        to the canonical MP2 energy, but it preserves the unprojected spatial
        profile.
        """
        if self.pt2_channel_mode == "local_exact":
            return self._local_exact_pt2_grid_contribution(
                molecule,
                features=features,
                occupation_tolerance=occupation_tolerance,
            )
        return self._legacy_projected_pt2_grid_contribution(
            molecule,
            features=features,
            occupation_tolerance=occupation_tolerance,
        )

    def energy_density(
        self,
        params: PyTree,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
        semilocal_energy_density: Array | None = None,
        hf_energy_density: Array | None = None,
        hf_spin_energy_density: tuple[Array, Array] | None = None,
        pt2_energy_density: Array | None = None,
    ) -> Array:
        """Return neural XC local grid contribution e_xc(r)."""
        channels = self.channel_contributions(
            params,
            molecule,
            features=features,
            semilocal_energy_density=semilocal_energy_density,
            hf_energy_density=hf_energy_density,
            hf_spin_energy_density=hf_spin_energy_density,
            pt2_energy_density=pt2_energy_density,
        )
        return jnp.sum(channels, axis=-1)

    def grid_contribution(
        self,
        params: PyTree,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
        semilocal_energy_density: Array | None = None,
        hf_energy_density: Array | None = None,
        hf_spin_energy_density: tuple[Array, Array] | None = None,
        pt2_energy_density: Array | None = None,
    ) -> Array:
        r"""Return neural XC local grid contribution e_xc(r)."""

        return self.energy_density(
            params,
            molecule,
            features=features,
            semilocal_energy_density=semilocal_energy_density,
            hf_energy_density=hf_energy_density,
            hf_spin_energy_density=hf_spin_energy_density,
            pt2_energy_density=pt2_energy_density,
        )

    def channel_contributions(
        self,
        params: PyTree,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
        semilocal_energy_density: Array | None = None,
        hf_energy_density: Array | None = None,
        hf_spin_energy_density: tuple[Array, Array] | None = None,
        pt2_energy_density: Array | None = None,
    ) -> Array:
        r"""Return per-channel Neural XC local contributions c_k(r) * e_k(r).

        The returned array has shape (..., n_channels) for either:
        - graddft_coeff_basis / normalized_mixing_basis:
          [semilocal_1, ..., semilocal_n, pt2_projected?, hf_projected]
        - graddft_coeff_basis_hf_pt2_heads:
          [c_1 e_1, ..., c_n e_n, fpt2 e_c^PT2, fx e_x^HF]
        """
        _, coefficients, basis = self._channel_coefficients_and_basis(
            params,
            molecule,
            features=features,
            semilocal_energy_density=semilocal_energy_density,
            hf_energy_density=hf_energy_density,
            hf_spin_energy_density=hf_spin_energy_density,
            pt2_energy_density=pt2_energy_density,
        )
        return self._assemble_channel_contributions(coefficients, basis)

    def _channel_coefficients_and_basis(
        self,
        params: PyTree,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
        semilocal_energy_density: Array | None = None,
        hf_energy_density: Array | None = None,
        hf_spin_energy_density: tuple[Array, Array] | None = None,
        pt2_energy_density: Array | None = None,
    ) -> tuple[RestrictedFeatureBundle, Array, Array]:
        if features is None:
            features = grid_features_for_molecule(molecule)
        semilocal_channels = self.semilocal_energy_density_channels(features)
        semilocal_total = (
            jnp.sum(semilocal_channels, axis=-1)
            if semilocal_energy_density is None
            else semilocal_energy_density
        )
        if not self._uses_hfx_channel():
            hf_projected = jnp.zeros_like(semilocal_total)
            hf_spin_inputs = (hf_projected, hf_projected)
            coefficient_molecule = None
        elif hf_energy_density is None:
            getter = getattr(self, "_hf_projected_components_for_inputs", None)
            if callable(getter):
                hf_projected, hf_projected_a, hf_projected_b = getter(
                    molecule,
                    features=features,
                )
            else:
                hf_projected, hf_projected_a, hf_projected_b = self.projected_hf_grid_contribution_components(
                    molecule,
                    features=features,
                )
            hf_spin_inputs: tuple[Array, Array] | None = (hf_projected_a, hf_projected_b)
            coefficient_molecule: Any | None = molecule
        else:
            hf_projected = hf_energy_density
            if hf_spin_energy_density is None:
                hf_spin_inputs = (hf_projected, hf_projected)
                coefficient_molecule = None
            else:
                hf_spin_inputs = hf_spin_energy_density
                coefficient_molecule = None
        if pt2_energy_density is None and self.include_pt2_channel:
            pt2_energy_density = self.projected_pt2_grid_contribution(
                molecule,
                features=features,
            )
        coefficients = self.channel_coefficients(
            params,
            features,
            molecule=coefficient_molecule,
            semilocal_energy_density=semilocal_total,
            hf_energy_density=hf_projected,
            pt2_energy_density=pt2_energy_density,
            hf_spin_energy_density=hf_spin_inputs,
        )
        semilocal_local_channels = self._semilocal_local_contribution_channels(
            features,
            semilocal_channels,
        )
        basis = self._assemble_basis_channels(
            semilocal_local_channels,
            hf_projected=hf_projected,
            pt2_projected=pt2_energy_density,
        )
        if coefficients.shape[-1] != basis.shape[-1]:
            raise ValueError(
                "Model output_dim must match basis channels "
                f"(got {coefficients.shape[-1]}, expected {basis.shape[-1]})."
            )
        return features, coefficients, basis

    def effective_exchange_fraction(
        self,
        params: PyTree,
        molecule: Any,
        *,
        features: RestrictedFeatureBundle | None = None,
    ) -> Array:
        features, coefficients, _ = self._channel_coefficients_and_basis(
            params,
            molecule,
            features=features,
        )
        weights = jnp.asarray(molecule.grid.weights)
        rho = jnp.maximum(features.rho, self.density_floor)
        hf_field = self._local_hf_fraction_from_coefficients(coefficients)
        numerator = jnp.tensordot(weights, rho * hf_field, axes=(0, 0))
        denominator = jnp.tensordot(weights, rho, axes=(0, 0))
        alpha = numerator / jnp.maximum(denominator, self.density_floor)
        alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
        return jnp.clip(alpha, 0.0, 1.0)

    def exact_exchange_energy(self, molecule: Any) -> Array:
        rep_tensor = jnp.asarray(molecule.rep_tensor)
        rdm1 = jnp.asarray(molecule.rdm1)
        if rdm1.ndim == 2:
            rdm1 = jnp.stack([0.5 * rdm1, 0.5 * rdm1], axis=0)

        def spin_exchange(dm_spin):
            exchange_matrix = jnp.einsum(
                "prqs,rs->pq",
                rep_tensor,
                dm_spin,
                precision=Precision.HIGHEST,
            )
            return -0.5 * jnp.einsum(
                "pq,pq->",
                dm_spin,
                exchange_matrix,
                precision=Precision.HIGHEST,
            )

        return jnp.sum(jax.vmap(spin_exchange)(rdm1))

    def semilocal_energy(
        self,
        features: RestrictedFeatureBundle,
        weights: Array,
    ) -> Array:
        semilocal_channels = self.semilocal_energy_density_channels(features)
        local = jnp.sum(
            self._semilocal_local_contribution_channels(features, semilocal_channels),
            axis=-1,
        )
        return jnp.tensordot(jnp.asarray(weights), local, axes=(0, 0))

    def energy_from_molecule(self, params: PyTree, molecule: Any) -> Array:
        energy, _ = self._scf_xc_energy_and_alpha_from_molecule(params, molecule)
        return energy

    def _scf_xc_energy_and_alpha_from_molecule(
        self,
        params: PyTree,
        molecule: Any,
    ) -> tuple[Array, Array]:
        features, coefficients, basis = self._channel_coefficients_and_basis(
            params,
            molecule,
            )
        local_xc = jnp.sum(self._assemble_channel_contributions(coefficients, basis), axis=-1)
        weights = jnp.asarray(molecule.grid.weights)
        energy = jnp.tensordot(weights, local_xc, axes=(0, 0))
        energy = jnp.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)

        hf_field = self._local_hf_fraction_from_coefficients(coefficients)
        rho = jnp.maximum(features.rho, self.density_floor)
        numerator = jnp.tensordot(weights, rho * hf_field, axes=(0, 0))
        denominator = jnp.tensordot(weights, rho, axes=(0, 0))
        alpha = numerator / jnp.maximum(denominator, self.density_floor)
        alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
        alpha = jnp.clip(alpha, 0.0, 1.0)
        return energy, alpha

    def _scf_xc_energy_and_alpha_with_current_hfx_basis(
        self,
        params: PyTree,
        molecule: Any,
    ) -> tuple[Array, Array]:
        features = grid_features_for_molecule(molecule)
        if not self._uses_hfx_channel():
            return self._scf_xc_energy_and_alpha_from_molecule(params, molecule)
        getter = getattr(self, "_hf_projected_components_for_inputs", None)
        if callable(getter):
            hf_projected, hf_projected_a, hf_projected_b = getter(
                molecule,
                features=features,
            )
        else:
            hf_projected, hf_projected_a, hf_projected_b = self.projected_hf_grid_contribution_components(
                molecule,
                features=features,
            )
        hf_projected = jax.lax.stop_gradient(hf_projected)
        hf_projected_a = jax.lax.stop_gradient(hf_projected_a)
        hf_projected_b = jax.lax.stop_gradient(hf_projected_b)
        pt2_projected = (
            self.projected_pt2_grid_contribution(molecule, features=features)
            if self.include_pt2_channel
            else None
        )
        features, coefficients, basis = self._channel_coefficients_and_basis(
            params,
            molecule,
            features=features,
            hf_energy_density=hf_projected,
            hf_spin_energy_density=(hf_projected_a, hf_projected_b),
            pt2_energy_density=pt2_projected,
        )
        local_xc = jnp.sum(self._assemble_channel_contributions(coefficients, basis), axis=-1)
        weights = jnp.asarray(molecule.grid.weights)
        energy = jnp.tensordot(weights, local_xc, axes=(0, 0))
        energy = jnp.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)

        hf_field = self._local_hf_fraction_from_coefficients(coefficients)
        rho = jnp.maximum(features.rho, self.density_floor)
        numerator = jnp.tensordot(weights, rho * hf_field, axes=(0, 0))
        denominator = jnp.tensordot(weights, rho, axes=(0, 0))
        alpha = numerator / jnp.maximum(denominator, self.density_floor)
        alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
        alpha = jnp.clip(alpha, 0.0, 1.0)
        return energy, alpha

    def scf_xc_energy_and_alpha_for_density(
        self,
        params: PyTree,
        molecule: Any,
        density: Array,
    ) -> tuple[Array, Array]:
        """Return SCF XC energy and effective HF fraction in one feature pass."""

        molecule_iter = self.scf_molecule_with_density(molecule, density)
        if not self._uses_hfx_channel():
            energy, _ = self._scf_xc_energy_and_alpha_from_molecule(
                params,
                molecule_iter,
            )
            return energy, jnp.asarray(0.0, dtype=jnp.asarray(density).dtype)

        if has_hfx_nu_source(molecule_iter):
            direct_payload = getattr(self, "_restricted_scf_direct_fock_payload", None)
            if callable(direct_payload):
                *_, alpha, _hfx_fock, energy = direct_payload(params, molecule_iter)
            else:
                energy, alpha = self._scf_xc_energy_and_alpha_with_current_hfx_basis(
                    params,
                    molecule_iter,
                )
        else:
            energy, alpha = self._scf_xc_energy_and_alpha_from_molecule(
                params,
                molecule_iter,
            )
        return energy, self._alpha_for_scf_fock(
            alpha,
            uses_explicit_hfx_fock=(
                self._uses_hfx_channel() and has_hfx_nu_source(molecule)
            ),
        )

    def scf_xc_energy_for_density(
        self,
        params: PyTree,
        molecule: Any,
        density: Array,
    ) -> Array:
        """GradDFT-style SCF entrypoint: E_xc as a function of AO density."""

        energy, _ = self.scf_xc_energy_and_alpha_for_density(
            params,
            molecule,
            density,
        )
        return energy

    def scf_exact_exchange_fraction(
        self,
        params: PyTree,
        molecule: Any,
        density: Array,
    ) -> Array:
        """Effective HF fraction paired with `scf_xc_energy_for_density`."""

        _, alpha = self.scf_xc_energy_and_alpha_for_density(
            params,
            molecule,
            density,
        )
        return alpha

    def scf_extra_fock_for_density(
        self,
        params: PyTree,
        molecule: Any,
        density: Array,
    ) -> Array:
        """DM21-style explicit local-HFX contribution to the SCF Fock."""

        if not self._uses_hfx_channel():
            density_arr = jnp.asarray(density)
            return jnp.zeros_like(density_arr)
        molecule_iter = self.scf_molecule_with_density(molecule, density)
        if (
            not has_hfx_nu_source(molecule_iter)
            and getattr(molecule_iter, "hfx_fxx", None) is None
        ):
            density_arr = jnp.asarray(density)
            return jnp.zeros_like(density_arr)
        hfx_fock, _ = self._explicit_hfx_fock_from_molecule(params, molecule_iter)
        return hfx_fock

    def energy_xc_only(self, params: PyTree, molecule: Any) -> Array:
        """GradDFT-compatible XC-only energy alias."""

        return self.energy_from_molecule(params, molecule)

    def energy(
        self,
        params: PyTree,
        molecule: Any,
        *,
        include_non_xc: bool = True,
    ) -> Array:
        """GradDFT-compatible total-energy entrypoint.

        When ``include_non_xc`` is true (default), return:
            E_tot = E_one + E_H + E_nuc + E_xc
        otherwise return only ``E_xc``.
        """

        e_xc = self.energy_from_molecule(params, molecule)
        if not include_non_xc or not self.is_xc:
            return e_xc

        if getattr(molecule, "h1e", None) is None:
            raise AttributeError("Molecule-like object must define h1e for total energy.")
        if getattr(molecule, "rep_tensor", None) is None:
            raise AttributeError("Molecule-like object must define rep_tensor for total energy.")
        if getattr(molecule, "rdm1", None) is None:
            raise AttributeError("Molecule-like object must define rdm1 for total energy.")
        if getattr(molecule, "nuclear_repulsion", None) is None:
            raise AttributeError(
                "Molecule-like object must define nuclear_repulsion for total energy."
            )

        density_matrix = jnp.asarray(molecule.rdm1)
        if density_matrix.ndim == 3:
            density_matrix = density_matrix.sum(axis=0)
        h1e = jnp.asarray(molecule.h1e)
        rep_tensor = jnp.asarray(molecule.rep_tensor)

        e_one = jnp.einsum("pq,pq->", density_matrix, h1e, precision=Precision.HIGHEST)
        j_matrix = jnp.einsum(
            "pqrs,rs->pq",
            rep_tensor,
            density_matrix,
            precision=Precision.HIGHEST,
        )
        e_hartree = 0.5 * jnp.einsum(
            "pq,pq->",
            density_matrix,
            j_matrix,
            precision=Precision.HIGHEST,
        )
        e_nuc = jnp.asarray(molecule.nuclear_repulsion)
        return e_one + e_hartree + e_nuc + e_xc
