from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any
import weakref

import jax
import jax.numpy as jnp
from jax import core as jax_core
from jax.lax import Precision
from jaxtyping import Array

from .jax_libxc import (
    RestrictedFeatureBundle,
    restricted_feature_bundle_from_rho_grad_tau,
)


_MAX_TRANSITION_RESPONSE_FEATURE_CACHE_SIZE = 64
_TRANSITION_RESPONSE_FEATURE_CACHE: dict[tuple[int, str], tuple[weakref.ReferenceType[Any], Array]] = {}


def _pytree_dataclass(cls):
    field_names = tuple(field.name for field in fields(cls))

    def tree_flatten(self):
        children = tuple(getattr(self, field_name) for field_name in field_names)
        return children, None

    @classmethod
    def tree_unflatten(cls_, aux_data, children):
        del aux_data
        return cls_(*children)

    cls.tree_flatten = tree_flatten
    cls.tree_unflatten = tree_unflatten
    return jax.tree_util.register_pytree_node_class(cls)


@_pytree_dataclass
@dataclass(frozen=True)
class MoleculeGridView:
    weights: Array
    coords: Array | None = None
    points: Array | None = None


@_pytree_dataclass
@dataclass(frozen=True)
class MoleculeLikeState:
    ao: Array
    ao_deriv1: Array
    grid: MoleculeGridView
    rdm1: Array
    mo_coeff: Array
    mo_occ: Array
    mo_energy: Array
    rep_tensor: Array | None = None
    h1e: Array | None = None
    overlap_matrix: Array | None = None
    ao_laplacian: Array | None = None
    atom_coords: Array | None = None
    atom_charges: Array | None = None
    hfx_omega_values: Array | None = None
    hfx_nu: Array | None = None


def molecule_grid_view(
    weights: Array,
    *,
    template: Any | None = None,
) -> MoleculeGridView:
    if template is None:
        return MoleculeGridView(weights=weights)
    return MoleculeGridView(
        weights=weights,
        coords=getattr(template, "coords", None),
        points=getattr(template, "points", None),
    )


def _is_tracer(value: Any) -> bool:
    value_type = type(value)
    return isinstance(value, jax_core.Tracer) or (
        "Tracer" in value_type.__name__ and value_type.__module__.startswith("jax")
    )


def _contains_tracer(value: Any) -> bool:
    return any(_is_tracer(leaf) for leaf in jax.tree_util.tree_leaves(value))


def _cached_transition_response_features(
    cache_key: tuple[int, str],
    molecule: Any,
) -> Array | None:
    cached = _TRANSITION_RESPONSE_FEATURE_CACHE.get(cache_key)
    if cached is None:
        return None
    cached_ref, cached_value = cached
    if cached_ref() is not molecule:
        _TRANSITION_RESPONSE_FEATURE_CACHE.pop(cache_key, None)
        return None
    if _contains_tracer(cached_value):
        _TRANSITION_RESPONSE_FEATURE_CACHE.pop(cache_key, None)
        return None
    return cached_value


def _cache_transition_response_features(
    cache_key: tuple[int, str],
    molecule: Any,
    value: Array,
) -> None:
    if not _contains_tracer(value):
        stale_keys = [
            key
            for key, (cached_ref, cached_value) in _TRANSITION_RESPONSE_FEATURE_CACHE.items()
            if cached_ref() is None or _contains_tracer(cached_value)
        ]
        for key in stale_keys:
            _TRANSITION_RESPONSE_FEATURE_CACHE.pop(key, None)
        while len(_TRANSITION_RESPONSE_FEATURE_CACHE) >= _MAX_TRANSITION_RESPONSE_FEATURE_CACHE_SIZE:
            _TRANSITION_RESPONSE_FEATURE_CACHE.pop(next(iter(_TRANSITION_RESPONSE_FEATURE_CACHE)))
        _TRANSITION_RESPONSE_FEATURE_CACHE[cache_key] = (weakref.ref(molecule), value)


def _ao_and_derivatives(molecule: Any) -> tuple[Array, Array]:
    ao = jnp.asarray(molecule.ao)
    ao_deriv1 = getattr(molecule, "ao_deriv1", None)
    if ao_deriv1 is None:
        raise AttributeError(
            "Molecule-like object must define ao_deriv1 to build GGA/meta-GGA features."
        )
    ao_deriv1 = jnp.asarray(ao_deriv1)
    if ao_deriv1.shape[0] < 4:
        raise ValueError("ao_deriv1 must contain AO values plus first derivatives.")
    return ao, ao_deriv1


def _ao_laplacian(molecule: Any) -> Array:
    ao_laplacian = getattr(molecule, "ao_laplacian", None)
    if ao_laplacian is None:
        raise AttributeError(
            "Molecule-like object must define ao_laplacian to build QAC/laplacian response features."
        )
    return jnp.asarray(ao_laplacian)


def _spin_density_and_gradient(
    ao: Array,
    ao_deriv1: Array,
    dm_spin: Array,
) -> tuple[Array, Array]:
    rho = jnp.einsum(
        "rp,pq,rq->r",
        ao,
        dm_spin,
        ao,
        precision=Precision.HIGHEST,
    )
    grad = 2.0 * jnp.einsum(
        "xrp,pq,rq->rx",
        ao_deriv1[1:4],
        dm_spin,
        ao,
        precision=Precision.HIGHEST,
    )
    return rho, grad


def _spin_tau(
    ao_deriv1: Array,
    mo_coeff_spin: Array,
    mo_occ_spin: Array,
) -> Array:
    ao_grad_mo = jnp.einsum(
        "xrp,pi->xri",
        ao_deriv1[1:4],
        mo_coeff_spin,
        precision=Precision.HIGHEST,
    )
    return 0.5 * jnp.einsum(
        "i,xri,xri->r",
        mo_occ_spin,
        ao_grad_mo,
        ao_grad_mo,
        precision=Precision.HIGHEST,
    )


def _spin_laplacian(
    ao: Array,
    ao_deriv1: Array,
    ao_laplacian: Array,
    dm_spin: Array,
) -> Array:
    lapl_left = jnp.einsum(
        "rp,pq,rq->r",
        ao_laplacian,
        dm_spin,
        ao,
        precision=Precision.HIGHEST,
    )
    grad_dot = jnp.einsum(
        "xrp,pq,xrq->r",
        ao_deriv1[1:4],
        dm_spin,
        ao_deriv1[1:4],
        precision=Precision.HIGHEST,
    )
    lapl_right = jnp.einsum(
        "rp,pq,rq->r",
        ao,
        dm_spin,
        ao_laplacian,
        precision=Precision.HIGHEST,
    )
    return lapl_left + 2.0 * grad_dot + lapl_right


@jax.jit
def _restricted_spin_channels_kernel(
    ao: Array,
    ao_deriv1: Array,
    rdm1: Array,
    mo_coeff: Array,
    mo_occ: Array,
) -> tuple[RestrictedFeatureBundle, Array]:
    rho_a, grad_a = _spin_density_and_gradient(ao, ao_deriv1, rdm1[0])
    rho_b, grad_b = _spin_density_and_gradient(ao, ao_deriv1, rdm1[1])
    tau_a = _spin_tau(ao_deriv1, mo_coeff[0], mo_occ[0])
    tau_b = _spin_tau(ao_deriv1, mo_coeff[1], mo_occ[1])

    bundle = RestrictedFeatureBundle(
        rho_a=rho_a,
        rho_b=rho_b,
        sigma_aa=jnp.einsum("rx,rx->r", grad_a, grad_a, precision=Precision.HIGHEST),
        sigma_ab=jnp.einsum("rx,rx->r", grad_a, grad_b, precision=Precision.HIGHEST),
        sigma_bb=jnp.einsum("rx,rx->r", grad_b, grad_b, precision=Precision.HIGHEST),
        tau_a=tau_a,
        tau_b=tau_b,
    )
    return bundle, grad_a + grad_b


@jax.jit
def _restricted_rho_kernel(
    ao: Array,
    rdm1: Array,
) -> Array:
    return jnp.einsum(
        "spq,rp,rq->r",
        rdm1,
        ao,
        ao,
        precision=Precision.HIGHEST,
    )


@jax.jit
def _restricted_rho_grad_kernel(
    ao: Array,
    ao_deriv1: Array,
    rdm1: Array,
) -> tuple[Array, Array]:
    rho_a, grad_a = _spin_density_and_gradient(ao, ao_deriv1, rdm1[0])
    rho_b, grad_b = _spin_density_and_gradient(ao, ao_deriv1, rdm1[1])
    return rho_a + rho_b, grad_a + grad_b


@jax.jit
def _restricted_rho_grad_tau_kernel(
    ao: Array,
    ao_deriv1: Array,
    rdm1: Array,
    mo_coeff: Array,
    mo_occ: Array,
) -> tuple[Array, Array, Array]:
    rho, grad = _restricted_rho_grad_kernel(ao, ao_deriv1, rdm1)
    tau_a = _spin_tau(ao_deriv1, mo_coeff[0], mo_occ[0])
    tau_b = _spin_tau(ao_deriv1, mo_coeff[1], mo_occ[1])
    return rho, grad, tau_a + tau_b


def _restricted_spin_inputs(
    molecule: Any,
) -> tuple[Array, Array, Array, Array, Array]:
    ao, ao_deriv1 = _ao_and_derivatives(molecule)
    rdm1 = jnp.asarray(molecule.rdm1)
    mo_coeff = jnp.asarray(molecule.mo_coeff)
    mo_occ = jnp.asarray(molecule.mo_occ)

    if rdm1.ndim == 2:
        rdm1 = jnp.stack([0.5 * rdm1, 0.5 * rdm1], axis=0)
    if mo_coeff.ndim == 2:
        mo_coeff = jnp.stack([mo_coeff, mo_coeff], axis=0)
    if mo_occ.ndim == 1:
        mo_occ = jnp.stack([0.5 * mo_occ, 0.5 * mo_occ], axis=0)
    return ao, ao_deriv1, rdm1, mo_coeff, mo_occ



def restricted_grid_features(molecule: Any) -> RestrictedFeatureBundle:
    ao, ao_deriv1, rdm1, mo_coeff, mo_occ = _restricted_spin_inputs(molecule)
    bundle, _ = _restricted_spin_channels_kernel(ao, ao_deriv1, rdm1, mo_coeff, mo_occ)
    return bundle


def restricted_grid_features_with_gradients(
    molecule: Any,
) -> tuple[RestrictedFeatureBundle, Array]:
    ao, ao_deriv1, rdm1, mo_coeff, mo_occ = _restricted_spin_inputs(molecule)
    return _restricted_spin_channels_kernel(ao, ao_deriv1, rdm1, mo_coeff, mo_occ)


def restricted_grid_response_variables(
    molecule: Any,
    *,
    feature_kind: str = "LDA",
) -> tuple[Array, Array | None, Array | None, Array | None]:
    ao, ao_deriv1, rdm1, mo_coeff, mo_occ = _restricted_spin_inputs(molecule)
    kind = str(feature_kind).upper()
    if kind == "LDA":
        return _restricted_rho_kernel(ao, rdm1), None, None, None
    if kind == "GGA":
        rho, grad = _restricted_rho_grad_kernel(ao, ao_deriv1, rdm1)
        return rho, grad, None, None
    if kind in {"MGGA", "MGGA_PT2"}:
        rho, grad, tau = _restricted_rho_grad_tau_kernel(
            ao,
            ao_deriv1,
            rdm1,
            mo_coeff,
            mo_occ,
        )
        return rho, grad, tau, None
    if kind in {"MGGA_LAPL", "MGGA_LAPL_PT2"}:
        rho, grad, tau = _restricted_rho_grad_tau_kernel(
            ao,
            ao_deriv1,
            rdm1,
            mo_coeff,
            mo_occ,
        )
        ao_laplacian = _ao_laplacian(molecule)
        if rdm1.ndim == 2:
            rdm1_spin = jnp.stack([0.5 * rdm1, 0.5 * rdm1], axis=0)
        else:
            rdm1_spin = rdm1
        lapl = _spin_laplacian(ao, ao_deriv1, ao_laplacian, rdm1_spin[0]) + _spin_laplacian(
            ao,
            ao_deriv1,
            ao_laplacian,
            rdm1_spin[1],
        )
        return rho, grad, tau, lapl
    raise ValueError(f"Unsupported feature_kind={feature_kind!r}.")


@jax.jit
def _restricted_transition_response_gga_kernel(
    ao_deriv1: Array,
    orbo: Array,
    orbv: Array,
) -> Array:
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
    gga_features = jnp.einsum(
        "xri,ra->xria",
        rho_o_full,
        rho_v_full[0],
        precision=Precision.HIGHEST,
    )
    return gga_features.at[1:4].add(
        jnp.einsum(
            "ri,xra->xria",
            rho_o_full[0],
            rho_v_full[1:4],
            precision=Precision.HIGHEST,
        )
    )


@jax.jit
def _restricted_transition_response_mgga_kernel(
    ao_deriv1: Array,
    orbo: Array,
    orbv: Array,
) -> Array:
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
    gga_features = jnp.einsum(
        "xri,ra->xria",
        rho_o_full,
        rho_v_full[0],
        precision=Precision.HIGHEST,
    )
    gga_features = gga_features.at[1:4].add(
        jnp.einsum(
            "ri,xra->xria",
            rho_o_full[0],
            rho_v_full[1:4],
            precision=Precision.HIGHEST,
        )
    )
    tau_ov = 0.5 * jnp.einsum(
        "xri,xra->ria",
        rho_o_full[1:4],
        rho_v_full[1:4],
        precision=Precision.HIGHEST,
    )
    return jnp.concatenate([gga_features, tau_ov[None, ...]], axis=0)


@jax.jit
def _restricted_transition_response_mgga_lapl_kernel(
    ao_deriv1: Array,
    ao_laplacian: Array,
    orbo: Array,
    orbv: Array,
) -> Array:
    mgga_features = _restricted_transition_response_mgga_kernel(ao_deriv1, orbo, orbv)
    rho_o = jnp.einsum("rp,pi->ri", ao_deriv1[0], orbo, precision=Precision.HIGHEST)
    rho_v = jnp.einsum("rp,pa->ra", ao_deriv1[0], orbv, precision=Precision.HIGHEST)
    lapl_o = jnp.einsum("rp,pi->ri", ao_laplacian, orbo, precision=Precision.HIGHEST)
    lapl_v = jnp.einsum("rp,pa->ra", ao_laplacian, orbv, precision=Precision.HIGHEST)
    grad_o = jnp.einsum("xrp,pi->xri", ao_deriv1[1:4], orbo, precision=Precision.HIGHEST)
    grad_v = jnp.einsum("xrp,pa->xra", ao_deriv1[1:4], orbv, precision=Precision.HIGHEST)
    lapl_ov = (
        jnp.einsum("ri,ra->ria", lapl_o, rho_v, precision=Precision.HIGHEST)
        + 2.0
        * jnp.einsum("xri,xra->ria", grad_o, grad_v, precision=Precision.HIGHEST)
        + jnp.einsum("ri,ra->ria", rho_o, lapl_v, precision=Precision.HIGHEST)
    )
    return jnp.concatenate([mgga_features, lapl_ov[None, ...]], axis=0)


def _restricted_transition_response_pt2_linearized_feature(
    molecule: Any,
    ao: Array,
    orbo: Array,
    orbv: Array,
    mo_energy: Array,
) -> Array:
    """Return a linearized PT2 response feature delta p(r) / delta kappa_{ia}.

    This keeps the local pair potential and MP2 pair weights fixed and
    differentiates only the leading occupied-virtual transition factor in the
    current local exact-pair gauge definition. It removes the fully frozen PT2
    field approximation without attempting the full orbital/denominator response
    of the PT2 field.
    """

    rep_tensor = getattr(molecule, "rep_tensor", None)
    if rep_tensor is None:
        raise AttributeError(
            "Molecule-like object must define rep_tensor for PT2 response features."
        )
    rep_tensor = jnp.asarray(rep_tensor)

    nocc = int(orbo.shape[1])
    eps_occ = mo_energy[:nocc]
    eps_vir = mo_energy[nocc:]

    eri_ovov = getattr(molecule, "eri_ovov", None)
    if eri_ovov is None:
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
    denom = jnp.where(jnp.abs(denom) > 1e-12, denom, -1e-12)
    exchange = jnp.transpose(eri_ovov, (0, 3, 2, 1))
    pair_weights = (2.0 * eri_ovov - exchange) / denom
    pair_potential = jnp.einsum(
        "gp,gq,pqrs,rj,sb->gjb",
        ao,
        ao,
        rep_tensor,
        orbo,
        orbv,
        precision=Precision.HIGHEST,
    )
    pt2_ov = jnp.einsum(
        "rjb,iajb->ria",
        pair_potential,
        pair_weights,
        precision=Precision.HIGHEST,
    )
    return jnp.nan_to_num(pt2_ov, nan=0.0, posinf=0.0, neginf=0.0)


def scale_restricted_grid_features(
    features: RestrictedFeatureBundle,
    scale: Array,
) -> RestrictedFeatureBundle:
    scale = jnp.asarray(scale)
    return RestrictedFeatureBundle(
        rho_a=scale * features.rho_a,
        rho_b=scale * features.rho_b,
        sigma_aa=(scale**2) * features.sigma_aa,
        sigma_ab=(scale**2) * features.sigma_ab,
        sigma_bb=(scale**2) * features.sigma_bb,
        tau_a=scale * features.tau_a,
        tau_b=scale * features.tau_b,
    )


def restricted_transition_response_features(
    molecule: Any,
    *,
    feature_kind: str = "LDA",
    occupation_tolerance: float = 1e-8,
) -> Array:
    """Return restricted singlet transition features for LDA/GGA/meta-GGA TDDFT.

    The output follows the PySCF restricted-singlet convention:
    - ``LDA``: ``[rho_ov]``
    - ``GGA``: ``[rho_ov, d_x rho_ov, d_y rho_ov, d_z rho_ov]``
    - ``MGGA``: GGA channels plus ``tau_ov``
    - ``MGGA_PT2``: MGGA channels plus linearized ``pt2_ov``
    - ``MGGA_LAPL``: MGGA channels plus ``lapl_ov``
    - ``MGGA_LAPL_PT2``: MGGA_LAPL channels plus linearized ``pt2_ov``
    """
    kind = feature_kind.upper()
    cache_key = (id(molecule), kind)
    cached = _cached_transition_response_features(cache_key, molecule)
    if cached is not None:
        return cached

    ao = jnp.asarray(molecule.ao)
    mo_coeff = jnp.asarray(molecule.mo_coeff)
    mo_occ = jnp.asarray(molecule.mo_occ)
    mo_energy = jnp.asarray(molecule.mo_energy)

    if mo_coeff.ndim == 3:
        if mo_coeff.shape[0] != 2:
            raise NotImplementedError(
                "restricted_transition_response_features expects restricted orbitals."
            )
        mo_coeff = mo_coeff[0]
        mo_occ = mo_occ[0]
        mo_energy = mo_energy[0]
    elif mo_energy.ndim == 2:
        mo_energy = mo_energy[0]

    nocc = getattr(molecule, "nocc", None)
    if nocc is None:
        # Fallback for molecule-like containers that do not expose nocc.
        nocc = int(jnp.count_nonzero(mo_occ > occupation_tolerance))
    else:
        nocc = int(nocc)
    nmo = int(mo_coeff.shape[1])
    if nocc <= 0 or nocc >= nmo:
        raise ValueError("Need at least one occupied and one virtual orbital.")

    orbo = mo_coeff[:, :nocc]
    orbv = mo_coeff[:, nocc:]
    rho_o = jnp.einsum("rp,pi->ri", ao, orbo, precision=Precision.HIGHEST)
    rho_v = jnp.einsum("rp,pa->ra", ao, orbv, precision=Precision.HIGHEST)
    rho_ov = jnp.einsum("ri,ra->ria", rho_o, rho_v, precision=Precision.HIGHEST)

    if kind == "LDA":
        out = rho_ov[None, ...]
        _cache_transition_response_features(cache_key, molecule, out)
        return out

    ao_deriv1 = getattr(molecule, "ao_deriv1", None)
    if ao_deriv1 is None:
        raise AttributeError(
            "Molecule-like object must define ao_deriv1 for GGA/meta-GGA transition features."
        )
    ao_deriv1 = jnp.asarray(ao_deriv1)
    if ao_deriv1.shape[0] < 4:
        raise ValueError("ao_deriv1 must contain AO values plus first derivatives.")

    if kind == "GGA":
        out = _restricted_transition_response_gga_kernel(ao_deriv1, orbo, orbv)
        _cache_transition_response_features(cache_key, molecule, out)
        return out
    if kind == "MGGA":
        out = _restricted_transition_response_mgga_kernel(ao_deriv1, orbo, orbv)
        _cache_transition_response_features(cache_key, molecule, out)
        return out
    if kind == "MGGA_PT2":
        mgga = _restricted_transition_response_mgga_kernel(ao_deriv1, orbo, orbv)
        pt2_ov = _restricted_transition_response_pt2_linearized_feature(
            molecule,
            ao,
            orbo,
            orbv,
            mo_energy,
        )
        out = jnp.concatenate([mgga, pt2_ov[None, ...]], axis=0)
        _cache_transition_response_features(cache_key, molecule, out)
        return out
    if kind not in {"MGGA_LAPL", "MGGA_LAPL_PT2"}:
        raise ValueError(f"Unsupported feature_kind={feature_kind!r}.")
    ao_laplacian = _ao_laplacian(molecule)
    mgga_lapl = _restricted_transition_response_mgga_lapl_kernel(
        ao_deriv1,
        ao_laplacian,
        orbo,
        orbv,
    )
    if kind == "MGGA_LAPL":
        out = mgga_lapl
    else:
        pt2_ov = _restricted_transition_response_pt2_linearized_feature(
            molecule,
            ao,
            orbo,
            orbv,
            mo_energy,
        )
        out = jnp.concatenate([mgga_lapl, pt2_ov[None, ...]], axis=0)
    _cache_transition_response_features(cache_key, molecule, out)
    return out


def restricted_feature_bundle_from_response_variables(
    rho: Array,
    grad: Array | None = None,
    tau: Array | None = None,
    *,
    density_floor: float = 1e-12,
) -> RestrictedFeatureBundle:
    """Public convenience wrapper shared by Neural_xc response builders."""

    return restricted_feature_bundle_from_rho_grad_tau(
        rho,
        grad,
        tau,
        density_floor=density_floor,
    )


def enhanced_neural_xc_input_features(
    features: RestrictedFeatureBundle,
    semilocal_energy_density: Array,
    *,
    density_floor: float = 1e-12,
) -> Array:
    rho_a = jnp.maximum(features.rho_a, density_floor)
    rho_b = jnp.maximum(features.rho_b, density_floor)
    rho = jnp.maximum(features.rho, density_floor)
    sigma = jnp.maximum(features.sigma, 0.0)
    tau = jnp.maximum(features.tau_a + features.tau_b, 0.0)
    return jnp.stack(
        [
            rho_a,
            rho_b,
            rho,
            jnp.log1p(rho),
            jnp.sqrt(rho),
            features.sigma_aa,
            features.sigma_ab,
            features.sigma_bb,
            sigma,
            features.tau_a,
            features.tau_b,
            tau,
            semilocal_energy_density,
        ],
        axis=-1,
    )


def canonical_neural_xc_input_features(
    features: RestrictedFeatureBundle,
    hfx_a: Array,
    hfx_b: Array,
    *,
    density_floor: float = 1e-12,
) -> Array:
    """Canonical local feature stack used by the neural XC runtime.

    The returned channels follow the historical coefficient-input order:
    [rho_a, rho_b, norm_grad_rho, norm_grad_rho_a, norm_grad_rho_b,
     tau_a, tau_b, hfx_a(omega*), hfx_b(omega*)].
    """

    rho_a = jnp.maximum(features.rho_a, density_floor)
    rho_b = jnp.maximum(features.rho_b, density_floor)
    tau_a = jnp.maximum(features.tau_a, 0.0)
    tau_b = jnp.maximum(features.tau_b, 0.0)
    norm_grad_a = jnp.maximum(features.sigma_aa, 0.0)
    norm_grad_b = jnp.maximum(features.sigma_bb, 0.0)
    norm_grad = jnp.maximum(features.sigma, 0.0)
    hfx_a = jnp.asarray(hfx_a)
    hfx_b = jnp.asarray(hfx_b)
    if hfx_a.ndim == rho_a.ndim:
        hfx_a = hfx_a[..., None]
    if hfx_b.ndim == rho_b.ndim:
        hfx_b = hfx_b[..., None]
    if hfx_a.shape[:-1] != rho_a.shape or hfx_b.shape[:-1] != rho_b.shape:
        raise ValueError(
            "Local HFX features must broadcast to the grid shape "
            f"(rho={rho_a.shape}, hfx_a={hfx_a.shape}, hfx_b={hfx_b.shape})."
        )
    leading = jnp.stack(
        [
            rho_a,
            rho_b,
            norm_grad,
            norm_grad_a,
            norm_grad_b,
            tau_a,
            tau_b,
        ],
        axis=-1,
    )
    return jnp.concatenate([leading, hfx_a, hfx_b], axis=-1)
