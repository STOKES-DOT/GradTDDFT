from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any
import weakref

import jax
import jax.numpy as jnp
from jax import core as jax_core
from jax.lax import Precision
from jaxtyping import Array

from .xc_backend.jax_libxc import (
    RestrictedFeatureBundle,
    restricted_feature_bundle_from_rho_grad_tau,
)


_MAX_TRANSITION_RESPONSE_FEATURE_CACHE_SIZE = 64
_TRANSITION_RESPONSE_FEATURE_CACHE: dict[tuple[int, str], tuple[weakref.ReferenceType[Any], Array]] = {}
_RESPONSE_FEATURE_KINDS = ("LDA", "GGA", "MGGA", "MGGA_LAPL")
_RESPONSE_FEATURE_KIND_BY_COUNT = {
    1: "LDA",
    4: "GGA",
    5: "MGGA",
    6: "MGGA_LAPL",
}
_REMOVED_PT2_RESPONSE_FEATURE_KINDS = {"MGGA_PT2", "MGGA_LAPL_PT2"}
_REMOVED_PT2_RESPONSE_FEATURE_MESSAGE = (
    "PT2 strict response features were removed because the previous "
    "linearized PT2 path was not a complete response kernel."
)


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
    hfx_local: Array | None = None
    hfx_fxx: Array | None = None
    hfx_nu: Array | None = None
    pt2_local: Array | None = None
    pt2_fock_response: Array | None = None


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


def normalize_response_feature_kind(
    value: Any,
    *,
    default: str = "LDA",
    label: str = "feature_kind",
) -> str:
    if value is None:
        kind = str(default).upper()
    else:
        kind = str(value).upper()
    if kind in _REMOVED_PT2_RESPONSE_FEATURE_KINDS:
        raise ValueError(_REMOVED_PT2_RESPONSE_FEATURE_MESSAGE)
    if kind not in _RESPONSE_FEATURE_KINDS:
        expected = "/".join(_RESPONSE_FEATURE_KINDS)
        raise ValueError(f"Unsupported {label}={value!r}. Expected one of {expected}.")
    return kind


def infer_response_feature_kind(values: Any) -> str:
    shape = getattr(values, "shape", None)
    if shape is None:
        shape = jnp.asarray(values).shape
    feature_count = int(shape[0])
    kind = _RESPONSE_FEATURE_KIND_BY_COUNT.get(feature_count)
    if kind is None:
        raise ValueError(
            "Strict response tensor must have feature dimension 1, 4, 5, or 6 "
            f"(got {shape})."
        )
    return kind


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
def _spin_resolved_channels_kernel(
    ao: Array,
    ao_deriv1: Array,
    rdm1: Array,
    mo_coeff: Array,
    mo_occ: Array,
) -> tuple[RestrictedFeatureBundle, Array, Array]:
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
    return bundle, grad_a, grad_b


@jax.jit
def _restricted_spin_channels_kernel(
    ao: Array,
    ao_deriv1: Array,
    rdm1: Array,
    mo_coeff: Array,
    mo_occ: Array,
) -> tuple[RestrictedFeatureBundle, Array]:
    bundle, grad_a, grad_b = _spin_resolved_channels_kernel(
        ao,
        ao_deriv1,
        rdm1,
        mo_coeff,
        mo_occ,
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



def _spin_resolved_grid_features(molecule: Any) -> RestrictedFeatureBundle:
    ao, ao_deriv1, rdm1, mo_coeff, mo_occ = _restricted_spin_inputs(molecule)
    bundle, _ = _restricted_spin_channels_kernel(ao, ao_deriv1, rdm1, mo_coeff, mo_occ)
    return bundle


def _spin_resolved_grid_features_with_gradients(
    molecule: Any,
) -> tuple[RestrictedFeatureBundle, Array]:
    ao, ao_deriv1, rdm1, mo_coeff, mo_occ = _restricted_spin_inputs(molecule)
    return _restricted_spin_channels_kernel(ao, ao_deriv1, rdm1, mo_coeff, mo_occ)


def restricted_grid_features(molecule: Any) -> RestrictedFeatureBundle:
    return _spin_resolved_grid_features(molecule)


def restricted_grid_features_with_gradients(
    molecule: Any,
) -> tuple[RestrictedFeatureBundle, Array]:
    return _spin_resolved_grid_features_with_gradients(molecule)


def unrestricted_grid_features(molecule: Any) -> RestrictedFeatureBundle:
    return _spin_resolved_grid_features(molecule)


def unrestricted_grid_features_with_gradients(
    molecule: Any,
) -> tuple[RestrictedFeatureBundle, Array]:
    return _spin_resolved_grid_features_with_gradients(molecule)


def has_explicit_spin_axis(molecule: Any) -> bool:
    for name in ("rdm1", "mo_occ", "mo_coeff"):
        value = getattr(molecule, name, None)
        if value is None:
            continue
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 1 and int(shape[0]) == 2:
            return True
    return False


_has_explicit_spin_axis = has_explicit_spin_axis


def grid_features_with_spin_gradients_for_molecule(
    molecule: Any,
) -> tuple[RestrictedFeatureBundle, Array, Array]:
    ao, ao_deriv1, rdm1, mo_coeff, mo_occ = _restricted_spin_inputs(molecule)
    return _spin_resolved_channels_kernel(ao, ao_deriv1, rdm1, mo_coeff, mo_occ)


def grid_features_for_molecule(molecule: Any) -> RestrictedFeatureBundle:
    if has_explicit_spin_axis(molecule):
        return unrestricted_grid_features(molecule)
    return restricted_grid_features(molecule)


def grid_features_with_gradients_for_molecule(
    molecule: Any,
) -> tuple[RestrictedFeatureBundle, Array]:
    if has_explicit_spin_axis(molecule):
        return unrestricted_grid_features_with_gradients(molecule)
    return restricted_grid_features_with_gradients(molecule)


def restricted_grid_response_variables(
    molecule: Any,
    *,
    feature_kind: str = "LDA",
) -> tuple[Array, Array | None, Array | None, Array | None]:
    ao, ao_deriv1, rdm1, mo_coeff, mo_occ = _restricted_spin_inputs(molecule)
    kind = normalize_response_feature_kind(feature_kind)
    if kind == "LDA":
        return _restricted_rho_kernel(ao, rdm1), None, None, None
    if kind == "GGA":
        rho, grad = _restricted_rho_grad_kernel(ao, ao_deriv1, rdm1)
        return rho, grad, None, None
    if kind == "MGGA":
        rho, grad, tau = _restricted_rho_grad_tau_kernel(
            ao,
            ao_deriv1,
            rdm1,
            mo_coeff,
            mo_occ,
        )
        return rho, grad, tau, None
    if kind == "MGGA_LAPL":
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
    - ``MGGA_LAPL``: MGGA channels plus ``lapl_ov``
    """
    kind = normalize_response_feature_kind(feature_kind)
    cache_key = (id(molecule), kind)
    cached = _cached_transition_response_features(cache_key, molecule)
    if cached is not None:
        return cached

    ao = jnp.asarray(molecule.ao)
    mo_coeff = jnp.asarray(molecule.mo_coeff)
    mo_occ = jnp.asarray(molecule.mo_occ)

    if mo_coeff.ndim == 3:
        if mo_coeff.shape[0] != 2:
            raise NotImplementedError(
                "restricted_transition_response_features expects restricted orbitals."
            )
        mo_coeff = mo_coeff[0]
        mo_occ = mo_occ[0]

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
    if kind != "MGGA_LAPL":
        raise ValueError(f"Unsupported feature_kind={feature_kind!r}.")
    ao_laplacian = _ao_laplacian(molecule)
    mgga_lapl = _restricted_transition_response_mgga_lapl_kernel(
        ao_deriv1,
        ao_laplacian,
        orbo,
        orbv,
    )
    out = mgga_lapl
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
