from __future__ import annotations

import copy
from dataclasses import dataclass, fields, is_dataclass, replace
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.lax import Precision
from jaxtyping import Array

from ..df import df_factors_to_mo_eri_slices
from ..features import (
    infer_response_feature_kind,
    normalize_response_feature_kind,
)
from ..data.integrals import eri_pair_matrix_to_mo_eri_slices
from ..neural_xc.inputs import hfx_nu_shape, hfx_nu_source
from ._utils import (
    _density_on_grid,
    _resolve_xc_functional,
    _restricted_orbital_data,
)


_RESTRICTED_RESPONSE_ERI_ATTRS = ("eri_ovov", "eri_ovvo", "eri_oovv")
_RESPONSE_FEATURE_COUNTS = {
    "LDA": 1,
    "GGA": 4,
    "MGGA": 5,
    "MGGA_LAPL": 6,
}


def _replace_response_eri_cache(molecule: Any, **updates: Any) -> Any:
    if is_dataclass(molecule):
        field_names = {field.name for field in fields(molecule)}
        field_updates = {
            key: value for key, value in updates.items() if key in field_names
        }
        extra_updates = {
            key: value for key, value in updates.items() if key not in field_names
        }
        molecule_out = replace(molecule, **field_updates) if field_updates else copy.copy(molecule)
        if extra_updates:
            molecule_out = copy.copy(molecule_out)
            for key, value in extra_updates.items():
                setattr(molecule_out, key, value)
        return molecule_out
    molecule_out = copy.copy(molecule)
    for key, value in updates.items():
        setattr(molecule_out, key, value)
    return molecule_out


@dataclass(frozen=True)
class _RestrictedResponseOperatorData:
    delta_eps: Array
    eri_ovov: Array
    eri_ovvo: Array | None
    eri_oovv: Array | None
    hybrid_exchange_a_action_fn: Callable[[Array], Array] | None = None
    hybrid_exchange_b_action_fn: Callable[[Array], Array] | None = None
    hybrid_exchange_diagonal: Array | None = None
    effective_tda_eri: Array | None = None
    effective_b_eri: Array | None = None
    xc_response_action_fn: Callable[[Array], Array] | None = None
    xc_response_diagonal: Array | None = None
    hybrid_fraction: Array | float = 0.0
    nonlocal_xc_a_action_fn: Callable[[Array], Array] | None = None
    nonlocal_xc_b_action_fn: Callable[[Array], Array] | None = None
    nonlocal_xc_a_diagonal: Array | None = None


def _assert_finite(values: Array, *, label: str) -> None:
    arr = jnp.asarray(values)
    if any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(arr)):
        # Avoid host NumPy conversion for traced values in differentiable paths.
        return
    if not np.all(np.isfinite(np.asarray(arr))):
        raise ValueError(
            f"{label} contains non-finite values. Strict TDDFT response does not apply "
            "nan/inf sanitization."
        )


def _as_grid_values(values: Any, reference: Array, *, label: str) -> Array:
    arr = jnp.asarray(values, dtype=reference.dtype)
    if arr.ndim == 0:
        arr = jnp.full_like(reference, arr)
    elif arr.shape != reference.shape:
        raise ValueError(
            f"{label} must be scalar or have shape {reference.shape}, got {arr.shape}."
        )
    _assert_finite(arr, label=label)
    return arr


def _as_grid_response_tensor(values: Any, *, ngrids: int) -> Array:
    arr = jnp.asarray(values)
    if arr.ndim == 1:
        arr = arr[None, None, :]
    elif arr.ndim != 3:
        raise ValueError(
            "Strict response tensor must have shape (nfeat, nfeat, ngrids) "
            f"or (ngrids,), got {arr.shape}."
        )
    if arr.shape[-1] != ngrids:
        raise ValueError(
            "Strict response tensor grid axis must match the molecule grid "
            f"({ngrids}), got {arr.shape[-1]}."
        )
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(
            "Strict response tensor must be square in the feature dimensions, "
            f"got {arr.shape}."
        )
    _assert_finite(arr, label="Strict response tensor")
    return arr


def _strict_hybrid_fraction(
    resolved_xc: Any | None,
    molecule: Any,
    total_density: Array | None,
) -> Array:
    if resolved_xc is None:
        return jnp.asarray(0.0)

    alpha = jnp.asarray(getattr(resolved_xc, "exact_exchange_fraction", 0.0))
    if alpha.ndim > 0 and alpha.size != 1:
        raise ValueError(
            "exact_exchange_fraction must be a scalar for strict PySCF-aligned TDDFT."
        )
    alpha = alpha.reshape(())
    _assert_finite(alpha, label="exact_exchange_fraction")

    if total_density is None:
        return alpha

    alpha_grid = None
    grid_hf_fraction = getattr(resolved_xc, "grid_hf_fraction", None)
    if callable(grid_hf_fraction):
        alpha_grid = _as_grid_values(
            grid_hf_fraction(molecule),
            total_density,
            label="grid_hf_fraction",
        )
    else:
        local_hf_fraction = getattr(resolved_xc, "local_hf_fraction", None)
        if callable(local_hf_fraction):
            alpha_grid = _as_grid_values(
                local_hf_fraction(total_density),
                total_density,
                label="local_hf_fraction",
            )

    if alpha_grid is None:
        return alpha

    if any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(alpha_grid)):
        # Constant-grid validation uses NumPy host checks; skip it in traced mode.
        # Downstream strict logic still uses a scalar hybrid fraction.
        return jnp.reshape(alpha_grid, (-1,))[0]

    alpha_grid_np = np.asarray(alpha_grid).reshape(-1)
    alpha0 = float(alpha_grid_np[0])
    if not np.allclose(alpha_grid_np, alpha0, atol=1e-12, rtol=0.0):
        raise ValueError(
            "Spatially varying local HF fractions are not supported in strict "
            "PySCF-aligned TDDFT response. Provide a scalar exact_exchange_fraction "
            "or a constant grid_hf_fraction/local_hf_fraction."
        )
    return jnp.asarray(alpha0, dtype=total_density.dtype)


def _needs_exchange_terms(value: Any) -> bool:
    arr = jnp.asarray(value)
    if any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(arr)):
        return True
    return bool(abs(float(np.asarray(arr).reshape(()))) > 1e-14)


def _hfx_nu_grid_chunk(source: Any, start: Array, chunk_size: int, *, dtype: Any) -> Array:
    if hasattr(source, "grid_chunk_padded"):
        return jnp.asarray(source.grid_chunk_padded(start, chunk_size), dtype=dtype)
    dense = jnp.asarray(source, dtype=dtype)
    indices = jnp.asarray(start, dtype=jnp.int32) + jnp.arange(
        int(chunk_size),
        dtype=jnp.int32,
    )
    chunk = jnp.take(dense, indices, axis=1, mode="clip")
    valid = indices < int(dense.shape[1])
    return jnp.where(
        valid.reshape((1, int(chunk_size), 1, 1)),
        chunk,
        jnp.zeros_like(chunk),
    )


def _grid_chunk(values: Array, start: Array, chunk_size: int, *, axis: int = 0) -> Array:
    values = jnp.asarray(values)
    indices = jnp.asarray(start, dtype=jnp.int32) + jnp.arange(
        int(chunk_size),
        dtype=jnp.int32,
    )
    chunk = jnp.take(values, indices, axis=axis, mode="clip")
    valid = indices < int(values.shape[axis])
    shape = [1] * chunk.ndim
    shape[axis] = int(chunk_size)
    return jnp.where(valid.reshape(shape), chunk, jnp.zeros_like(chunk))


def _restricted_response_features_chunk(
    molecule: Any,
    orbo: Array,
    orbv: Array,
    start: Array,
    chunk_size: int,
    *,
    feature_kind: str,
    dtype: Any,
) -> Array:
    ao = jnp.asarray(molecule.ao, dtype=dtype)
    ao_chunk = _grid_chunk(ao, start, chunk_size, axis=0)
    rho_o = jnp.einsum("gp,pi->gi", ao_chunk, orbo, precision=Precision.HIGHEST)
    rho_v = jnp.einsum("gp,pa->ga", ao_chunk, orbv, precision=Precision.HIGHEST)
    rho_ov = jnp.einsum("gi,ga->gia", rho_o, rho_v, precision=Precision.HIGHEST)
    if feature_kind == "LDA":
        return rho_ov[None, ...]

    ao_deriv1 = getattr(molecule, "ao_deriv1", None)
    if ao_deriv1 is None:
        raise AttributeError(
            "Molecule-like object must define ao_deriv1 for GGA/meta-GGA transition features."
        )
    ao_deriv1_chunk = _grid_chunk(
        jnp.asarray(ao_deriv1, dtype=dtype),
        start,
        chunk_size,
        axis=1,
    )
    if ao_deriv1_chunk.shape[0] < 4:
        raise ValueError("ao_deriv1 must contain AO values plus first derivatives.")
    rho_o_full = jnp.einsum(
        "xgp,pi->xgi",
        ao_deriv1_chunk[:4],
        orbo,
        precision=Precision.HIGHEST,
    )
    rho_v_full = jnp.einsum(
        "xgp,pa->xga",
        ao_deriv1_chunk[:4],
        orbv,
        precision=Precision.HIGHEST,
    )
    gga_features = jnp.einsum(
        "xgi,ga->xgia",
        rho_o_full,
        rho_v_full[0],
        precision=Precision.HIGHEST,
    )
    gga_features = gga_features.at[1:4].add(
        jnp.einsum(
            "gi,xga->xgia",
            rho_o_full[0],
            rho_v_full[1:4],
            precision=Precision.HIGHEST,
        )
    )
    if feature_kind == "GGA":
        return gga_features

    tau_ov = 0.5 * jnp.einsum(
        "xgi,xga->gia",
        rho_o_full[1:4],
        rho_v_full[1:4],
        precision=Precision.HIGHEST,
    )
    mgga_features = jnp.concatenate([gga_features, tau_ov[None, ...]], axis=0)
    if feature_kind == "MGGA":
        return mgga_features
    if feature_kind != "MGGA_LAPL":
        raise ValueError(f"Unsupported response_feature_kind={feature_kind!r}.")

    ao_laplacian = getattr(molecule, "ao_laplacian", None)
    if ao_laplacian is None:
        raise AttributeError(
            "Molecule-like object must define ao_laplacian for MGGA_LAPL transition features."
        )
    lapl_chunk = _grid_chunk(
        jnp.asarray(ao_laplacian, dtype=dtype),
        start,
        chunk_size,
        axis=0,
    )
    lapl_o = jnp.einsum("gp,pi->gi", lapl_chunk, orbo, precision=Precision.HIGHEST)
    lapl_v = jnp.einsum("gp,pa->ga", lapl_chunk, orbv, precision=Precision.HIGHEST)
    lapl_ov = (
        jnp.einsum("gi,ga->gia", lapl_o, rho_v, precision=Precision.HIGHEST)
        + 2.0
        * jnp.einsum(
            "xgi,xga->gia",
            rho_o_full[1:4],
            rho_v_full[1:4],
            precision=Precision.HIGHEST,
        )
        + jnp.einsum("gi,ga->gia", rho_o, lapl_v, precision=Precision.HIGHEST)
    )
    return jnp.concatenate([mgga_features, lapl_ov[None, ...]], axis=0)


def _restricted_grid_xc_response(
    molecule: Any,
    orbo: Array,
    orbv: Array,
    weighted_hessian: Array,
    *,
    feature_kind: str,
    dtype: Any,
) -> tuple[Callable[[Array], Array], Array]:
    weighted_hessian = jnp.asarray(weighted_hessian, dtype=dtype)
    ngrids = int(weighted_hessian.shape[-1])
    chunk_size = max(1, min(256, ngrids))
    n_chunks = (ngrids + chunk_size - 1) // chunk_size
    nocc = int(orbo.shape[1])
    nvir = int(orbv.shape[1])

    def chunk_terms(start: Array) -> tuple[Array, Array]:
        features = _restricted_response_features_chunk(
            molecule,
            orbo,
            orbv,
            start,
            chunk_size,
            feature_kind=feature_kind,
            dtype=dtype,
        )
        hessian = _grid_chunk(weighted_hessian, start, chunk_size, axis=2)
        return features, hessian

    def action(x: Array) -> Array:
        original_shape = jnp.asarray(x).shape
        values = jnp.asarray(x, dtype=dtype).reshape(-1, nocc, nvir)
        zero = jnp.zeros_like(values)

        def body(carry: Array, chunk_idx: Array) -> tuple[Array, None]:
            features, hessian = chunk_terms(chunk_idx * chunk_size)
            projected = jnp.einsum(
                "xgia,nia->nxg",
                features,
                values,
                precision=Precision.HIGHEST,
            )
            weighted = jnp.einsum(
                "xyg,nyg->nxg",
                hessian,
                projected,
                precision=Precision.HIGHEST,
            )
            delta = 2.0 * jnp.einsum(
                "xgia,nxg->nia",
                features,
                weighted,
                precision=Precision.HIGHEST,
            )
            return carry + delta, None

        out, _ = jax.lax.scan(body, zero, jnp.arange(n_chunks))
        return out.reshape(original_shape)

    def diagonal_body(carry: Array, chunk_idx: Array) -> tuple[Array, None]:
        features, hessian = chunk_terms(chunk_idx * chunk_size)
        diagonal = 2.0 * jnp.einsum(
            "xyg,xgia,ygia->ia",
            hessian,
            features,
            features,
            precision=Precision.HIGHEST,
        )
        return carry + diagonal, None

    diagonal, _ = jax.lax.scan(
        diagonal_body,
        jnp.zeros((nocc, nvir), dtype=dtype),
        jnp.arange(n_chunks),
    )
    return action, diagonal


def _as_nonlocal_response_diagonal(
    values: Any,
    *,
    delta_eps: Array,
    label: str = "Nonlocal response diagonal",
) -> Array:
    nocc, nvir = delta_eps.shape
    arr = jnp.asarray(values, dtype=delta_eps.dtype)
    if arr.shape == delta_eps.shape:
        diagonal = arr
    elif arr.shape == (int(nocc * nvir),):
        diagonal = arr.reshape(nocc, nvir)
    else:
        raise ValueError(
            f"{label} must have shape {delta_eps.shape} or {(int(nocc * nvir),)}, "
            f"got {arr.shape}."
        )
    _assert_finite(diagonal, label=label)
    return diagonal


def _optional_response_method(
    resolved_xc: Any,
    method_name: str,
    callback_attr: str,
) -> Callable[..., Any] | None:
    method = getattr(resolved_xc, method_name, None)
    if not callable(method):
        return None
    if hasattr(resolved_xc, callback_attr) and getattr(resolved_xc, callback_attr) is None:
        return None
    return method


def _resolve_nonlocal_response_action_pair(
    resolved_xc: Any,
    molecule: Any,
    *,
    delta_eps: Array,
    occupation_tolerance: float,
) -> tuple[Callable[[Array], Array] | None, Callable[[Array], Array] | None, Array | None]:
    if resolved_xc is None:
        return None, None, None

    action_a_raw = _optional_response_method(
        resolved_xc,
        "nonlocal_response_a_action",
        "nonlocal_response_action_fn",
    )
    if action_a_raw is None:
        action_a_raw = _optional_response_method(
            resolved_xc,
            "nonlocal_response_action",
            "nonlocal_response_action_fn",
        )
    action_b_raw = _optional_response_method(
        resolved_xc,
        "nonlocal_response_b_action",
        "nonlocal_response_b_action_fn",
    )
    if action_a_raw is None or action_b_raw is None:
        return None, None, None

    def _wrap_action(action_fn: Callable[..., Any], label: str) -> Callable[[Array], Array]:
        def action(values: Array) -> Array:
            out = jnp.asarray(
                action_fn(
                    molecule,
                    values,
                    occupation_tolerance=occupation_tolerance,
                ),
                dtype=delta_eps.dtype,
            )
            if out.shape != jnp.asarray(values).shape:
                raise ValueError(
                    f"{label} must preserve the transition-amplitude shape "
                    f"{jnp.asarray(values).shape}, got {out.shape}."
                )
            _assert_finite(out, label=label)
            return out

        return action

    diagonal = None
    diagonal_fn = _optional_response_method(
        resolved_xc,
        "nonlocal_response_diagonal",
        "nonlocal_response_diagonal_fn",
    )
    if callable(diagonal_fn):
        diagonal = _as_nonlocal_response_diagonal(
            diagonal_fn(molecule, occupation_tolerance=occupation_tolerance),
            delta_eps=delta_eps,
            label="Nonlocal A response diagonal",
        )

    return (
        _wrap_action(action_a_raw, "Nonlocal A response action"),
        _wrap_action(action_b_raw, "Nonlocal B response action"),
        diagonal,
    )


@partial(jax.jit, static_argnames=("need_ovvo", "include_oovv"))
def _rep_tensor_to_mo_eri_slices(
    rep_tensor: Array,
    orbo: Array,
    orbv: Array,
    *,
    need_ovvo: bool = True,
    include_oovv: bool = True,
) -> tuple[Array, Array | None, Array | None]:
    """Transform a full AO ERI tensor into the MO slices used by restricted TDDFT."""

    eri_ovov = jnp.einsum(
        "pqrs,pi,qa,rj,sb->iajb",
        rep_tensor,
        orbo,
        orbv,
        orbo,
        orbv,
        precision=Precision.HIGHEST,
    )
    if need_ovvo:
        eri_ovvo = jnp.einsum(
            "pqrs,pi,qa,rb,sj->iabj",
            rep_tensor,
            orbo,
            orbv,
            orbv,
            orbo,
            precision=Precision.HIGHEST,
        )
    else:
        eri_ovvo = None
    if include_oovv:
        eri_oovv = jnp.einsum(
            "pqrs,pi,qj,ra,sb->ijab",
            rep_tensor,
            orbo,
            orbo,
            orbv,
            orbv,
            precision=Precision.HIGHEST,
        )
    else:
        eri_oovv = None
    return eri_ovov, eri_ovvo, eri_oovv


def _restricted_eri_slices(
    molecule: Any,
    rep_tensor: Array | None,
    orbo: Array,
    orbv: Array,
    *,
    need_ovvo: bool = True,
    include_oovv: bool = True,
) -> tuple[Array, Array | None, Array | None]:
    eri_ovov = getattr(molecule, "eri_ovov", None)
    eri_ovvo = getattr(molecule, "eri_ovvo", None)
    eri_oovv = getattr(molecule, "eri_oovv", None)
    if eri_ovov is None or (need_ovvo and eri_ovvo is None) or (include_oovv and eri_oovv is None):
        df_factors = getattr(molecule, "df_factors", None)
        if df_factors is not None and int(jnp.asarray(df_factors).size) > 0:
            df_factors = jnp.asarray(df_factors)
            if need_ovvo:
                eri_ovov, eri_ovvo, eri_oovv = df_factors_to_mo_eri_slices(
                    df_factors,
                    jnp.concatenate([orbo, orbv], axis=1),
                    orbo.shape[1],
                    include_oovv=include_oovv,
                )
            else:
                b_ov = jnp.einsum(
                    "Qpq,pi,qa->Qia",
                    df_factors,
                    orbo,
                    orbv,
                    precision=Precision.HIGHEST,
                )
                eri_ovov = jnp.einsum(
                    "Qia,Qjb->iajb",
                    b_ov,
                    b_ov,
                    precision=Precision.HIGHEST,
                )
                eri_ovvo = None
                if include_oovv:
                    b_oo = jnp.einsum(
                        "Qpq,pi,qj->Qij",
                        df_factors,
                        orbo,
                        orbo,
                        precision=Precision.HIGHEST,
                    )
                    b_vv = jnp.einsum(
                        "Qpq,pa,qb->Qab",
                        df_factors,
                        orbv,
                        orbv,
                        precision=Precision.HIGHEST,
                    )
                    eri_oovv = jnp.einsum(
                        "Qij,Qab->ijab",
                        b_oo,
                        b_vv,
                        precision=Precision.HIGHEST,
                    )
                else:
                    eri_oovv = None
        else:
            eri_pair_matrix = getattr(molecule, "eri_pair_matrix", None)
            if eri_pair_matrix is not None and int(jnp.asarray(eri_pair_matrix).size) > 0:
                eri_ovov, eri_ovvo, eri_oovv = eri_pair_matrix_to_mo_eri_slices(
                    jnp.asarray(eri_pair_matrix),
                    jnp.concatenate([orbo, orbv], axis=1),
                    nocc=orbo.shape[1],
                    include_oovv=include_oovv,
                )
                if not need_ovvo:
                    eri_ovvo = None
            elif rep_tensor is None or int(jnp.asarray(rep_tensor).size) == 0:
                raise ValueError(
                    "The molecule must provide rep_tensor, df_factors, or precomputed "
                    "eri_ovov/eri_ovvo/eri_oovv for the Hartree response."
                )
            else:
                eri_ovov, eri_ovvo, eri_oovv = _rep_tensor_to_mo_eri_slices(
                    jnp.asarray(rep_tensor),
                    orbo,
                    orbv,
                    need_ovvo=need_ovvo,
                    include_oovv=include_oovv,
                )
    eri_ovov = jnp.asarray(eri_ovov)
    eri_ovvo = None if eri_ovvo is None else jnp.asarray(eri_ovvo)
    eri_oovv = None if eri_oovv is None or not include_oovv else jnp.asarray(eri_oovv)
    return eri_ovov, eri_ovvo, eri_oovv


def refresh_restricted_response_eri_slices(
    molecule: Any,
    *,
    occupation_tolerance: float = 1e-8,
    include_oovv: bool = True,
) -> Any:
    """Return a molecule with MO-basis restricted response ERI slices cached."""

    orbo, orbv, _, _ = _restricted_orbital_data(
        molecule,
        occupation_tolerance=occupation_tolerance,
    )
    rep_tensor_obj = getattr(molecule, "rep_tensor", None)
    rep_tensor = None
    if rep_tensor_obj is not None and int(jnp.asarray(rep_tensor_obj).size) > 0:
        rep_tensor = jnp.asarray(rep_tensor_obj)
    uncached = _replace_response_eri_cache(
        molecule,
        **{name: None for name in _RESTRICTED_RESPONSE_ERI_ATTRS},
    )
    eri_ovov, eri_ovvo, eri_oovv = _restricted_eri_slices(
        uncached,
        rep_tensor,
        orbo,
        orbv,
        need_ovvo=True,
        include_oovv=include_oovv,
    )
    return _replace_response_eri_cache(
        molecule,
        eri_ovov=eri_ovov,
        eri_ovvo=eri_ovvo,
        eri_oovv=eri_oovv,
    )


def _restricted_hfx_nu_hybrid_exchange_actions(
    molecule: Any,
    orbo: Array,
    orbv: Array,
    weights: Array,
    hybrid_fraction: Any,
    *,
    dtype: Any,
) -> tuple[Callable[[Array], Array], Callable[[Array], Array], Array] | None:
    alpha = jnp.asarray(hybrid_fraction, dtype=dtype)
    if not _needs_exchange_terms(alpha):
        return None
    nu_source = hfx_nu_source(molecule)
    if nu_source is None:
        return None
    n_omega, ngrids, nao, nao2 = hfx_nu_shape(nu_source)
    if n_omega < 1:
        return None
    if int(nao) != int(orbo.shape[0]) or int(nao2) != int(orbo.shape[0]):
        raise ValueError(
            "HFX nu AO axes must match mo_coeff AO dimension "
            f"({(nao, nao2)} vs {orbo.shape[0]})."
        )

    weights = jnp.asarray(weights, dtype=dtype)
    ao = jnp.asarray(molecule.ao, dtype=dtype)
    orbo = jnp.asarray(orbo, dtype=dtype)
    orbv = jnp.asarray(orbv, dtype=dtype)
    nocc = int(orbo.shape[1])
    nvir = int(orbv.shape[1])
    source_chunk = int(getattr(nu_source, "chunk_size", 0) or 0)
    chunk_size = max(1, min(int(source_chunk or 256), int(ngrids)))
    n_chunks = (int(ngrids) + chunk_size - 1) // chunk_size

    def chunk_terms(start: Array) -> tuple[Array, Array, Array, Array, Array]:
        ao_chunk = _grid_chunk(ao, start, chunk_size, axis=0)
        weights_chunk = _grid_chunk(weights, start, chunk_size, axis=0)
        nu_chunk = _hfx_nu_grid_chunk(
            nu_source,
            start,
            chunk_size,
            dtype=dtype,
        )[0]
        rho_o = jnp.einsum("gp,pi->gi", ao_chunk, orbo, precision=Precision.HIGHEST)
        rho_v = jnp.einsum("gp,pa->ga", ao_chunk, orbv, precision=Precision.HIGHEST)
        nu_oo = jnp.einsum(
            "pi,gpq,qj->gij",
            orbo,
            nu_chunk,
            orbo,
            precision=Precision.HIGHEST,
        )
        nu_ov = jnp.einsum(
            "pi,gpq,qb->gib",
            orbo,
            nu_chunk,
            orbv,
            precision=Precision.HIGHEST,
        )
        return weights_chunk, rho_o, rho_v, nu_oo, nu_ov

    def action_from_chunks(x: Array, *, b_block: bool) -> Array:
        original_shape = jnp.asarray(x).shape
        x = jnp.asarray(x, dtype=dtype).reshape(-1, nocc, nvir)
        zero = jnp.zeros_like(x)

        def body(carry: Array, chunk_idx: Array) -> tuple[Array, None]:
            start = chunk_idx * chunk_size
            weights_chunk, rho_o, rho_v, nu_oo, nu_ov = chunk_terms(start)
            if b_block:
                delta = -alpha * jnp.einsum(
                    "g,ga,gj,gib,njb->nia",
                    weights_chunk,
                    rho_v,
                    rho_o,
                    nu_ov,
                    x,
                    precision=Precision.HIGHEST,
                )
            else:
                delta = -alpha * jnp.einsum(
                    "g,ga,gb,gij,njb->nia",
                    weights_chunk,
                    rho_v,
                    rho_v,
                    nu_oo,
                    x,
                    precision=Precision.HIGHEST,
                )
            out = carry + delta
            return out, None

        out, _ = jax.lax.scan(body, zero, jnp.arange(n_chunks))
        return out.reshape(original_shape)

    def a_action(x: Array) -> Array:
        return action_from_chunks(x, b_block=False)

    def b_action(x: Array) -> Array:
        return action_from_chunks(x, b_block=True)

    def diagonal_body(carry: Array, chunk_idx: Array) -> tuple[Array, None]:
        start = chunk_idx * chunk_size
        weights_chunk, _, rho_v, nu_oo, _ = chunk_terms(start)
        diagonal = -alpha * jnp.einsum(
            "g,ga,ga,gii->ia",
            weights_chunk,
            rho_v,
            rho_v,
            nu_oo,
            precision=Precision.HIGHEST,
        )
        return carry + diagonal, None

    diagonal, _ = jax.lax.scan(
        diagonal_body,
        jnp.zeros((nocc, nvir), dtype=dtype),
        jnp.arange(n_chunks),
    )
    return a_action, b_action, diagonal


def _build_restricted_response_operator_data(
    molecule: Any,
    xc_functional: Any | None = None,
    *,
    xc_params: Any | None = None,
    occupation_tolerance: float = 1e-8,
    need_b_terms: bool = True,
) -> _RestrictedResponseOperatorData:
    resolved_xc = _resolve_xc_functional(molecule, xc_functional, xc_params)
    weights = jnp.asarray(molecule.grid.weights)
    rep_tensor_obj = getattr(molecule, "rep_tensor", None)
    rep_tensor = None
    if rep_tensor_obj is not None and int(jnp.asarray(rep_tensor_obj).size) > 0:
        rep_tensor = jnp.asarray(rep_tensor_obj)
    orbo, orbv, delta_eps, _ = _restricted_orbital_data(
        molecule,
        occupation_tolerance,
    )

    xc_response_action_fn = None
    xc_response_diagonal = None
    hybrid_fraction = 0.0
    hybrid_exchange_a_action_fn = None
    hybrid_exchange_b_action_fn = None
    hybrid_exchange_diagonal = None
    nonlocal_xc_a_action_fn = None
    nonlocal_xc_b_action_fn = None
    nonlocal_xc_a_diagonal = None

    if resolved_xc is not None:
        total_density = _density_on_grid(molecule)
        hybrid_fraction = _strict_hybrid_fraction(
            resolved_xc,
            molecule,
            total_density,
        )
        grid_response_tensor = getattr(resolved_xc, "grid_response_tensor", None)
        if callable(grid_response_tensor):
            strict_tensor = _as_grid_response_tensor(
                grid_response_tensor(molecule),
                ngrids=int(weights.shape[0]),
            )
            feature_kind = getattr(resolved_xc, "response_feature_kind", None)
            if feature_kind is None:
                feature_kind = infer_response_feature_kind(strict_tensor)
            feature_kind = normalize_response_feature_kind(
                feature_kind,
                label="response_feature_kind",
            )
            expected_features = _RESPONSE_FEATURE_COUNTS[str(feature_kind)]
            if strict_tensor.shape[0] != expected_features:
                raise ValueError(
                    "Strict response tensor feature dimension must match the "
                    "transition-feature dimension "
                    f"(got {strict_tensor.shape[0]} vs {expected_features})."
                )
            xc_response_action_fn, xc_response_diagonal = _restricted_grid_xc_response(
                molecule,
                orbo,
                orbv,
                strict_tensor * weights[None, None, :],
                feature_kind=str(feature_kind),
                dtype=jnp.asarray(weights).dtype,
            )
        else:
            feature_kind = normalize_response_feature_kind(
                getattr(resolved_xc, "response_feature_kind", None),
                default="LDA",
                label="response_feature_kind",
            )
            if feature_kind != "LDA":
                raise ValueError(
                    "Strict PySCF-aligned TDDFT requires grid_response_tensor for "
                    f"{feature_kind} functionals. The scalar local-kernel projection "
                    "path is an approximation and is disabled."
                )
            grid_kernel = getattr(resolved_xc, "grid_kernel", None)
            if callable(grid_kernel):
                local_fxc = grid_kernel(molecule)
            else:
                local_fxc = resolved_xc.local_kernel(total_density)
            local_fxc = _as_grid_values(local_fxc, total_density, label="XC kernel")
            xc_response_action_fn, xc_response_diagonal = _restricted_grid_xc_response(
                molecule,
                feature_kind="LDA",
                orbo=orbo,
                orbv=orbv,
                weighted_hessian=(weights * local_fxc)[None, None, :],
                dtype=jnp.asarray(weights).dtype,
            )

        (
            pair_nonlocal_a_action,
            pair_nonlocal_b_action,
            pair_nonlocal_a_diagonal,
        ) = _resolve_nonlocal_response_action_pair(
            resolved_xc,
            molecule,
            delta_eps=delta_eps,
            occupation_tolerance=occupation_tolerance,
        )
        if pair_nonlocal_a_action is not None and pair_nonlocal_b_action is not None:
            nonlocal_xc_a_action_fn = pair_nonlocal_a_action
            nonlocal_xc_b_action_fn = pair_nonlocal_b_action
            nonlocal_xc_a_diagonal = pair_nonlocal_a_diagonal

    hfx_nu_exchange = _restricted_hfx_nu_hybrid_exchange_actions(
        molecule,
        orbo,
        orbv,
        weights,
        hybrid_fraction,
        dtype=jnp.asarray(weights).dtype,
    )
    if hfx_nu_exchange is not None:
        (
            hybrid_exchange_a_action_fn,
            hybrid_exchange_b_action_fn,
            hybrid_exchange_diagonal,
        ) = hfx_nu_exchange

    eri_ovov, eri_ovvo, eri_oovv = _restricted_eri_slices(
        molecule,
        rep_tensor,
        orbo,
        orbv,
        need_ovvo=need_b_terms,
        include_oovv=(
            _needs_exchange_terms(hybrid_fraction)
            and hybrid_exchange_a_action_fn is None
        ),
    )
    alpha = jnp.asarray(hybrid_fraction, dtype=eri_ovov.dtype)
    effective_tda_eri = 2.0 * eri_ovov
    if eri_oovv is not None:
        effective_tda_eri = effective_tda_eri - alpha * jnp.transpose(
            eri_oovv,
            (0, 2, 1, 3),
        )
    effective_b_eri = None
    if need_b_terms and eri_ovvo is not None:
        effective_b_eri = 2.0 * eri_ovvo
        if hybrid_exchange_b_action_fn is None:
            effective_b_eri = effective_b_eri - alpha * jnp.transpose(
                eri_ovvo,
                (0, 2, 1, 3),
            )

    return _RestrictedResponseOperatorData(
        delta_eps=delta_eps,
        eri_ovov=eri_ovov,
        eri_ovvo=eri_ovvo,
        eri_oovv=eri_oovv,
        hybrid_exchange_a_action_fn=hybrid_exchange_a_action_fn,
        hybrid_exchange_b_action_fn=hybrid_exchange_b_action_fn,
        hybrid_exchange_diagonal=hybrid_exchange_diagonal,
        effective_tda_eri=effective_tda_eri,
        effective_b_eri=effective_b_eri,
        xc_response_action_fn=xc_response_action_fn,
        xc_response_diagonal=xc_response_diagonal,
        hybrid_fraction=hybrid_fraction,
        nonlocal_xc_a_action_fn=nonlocal_xc_a_action_fn,
        nonlocal_xc_b_action_fn=nonlocal_xc_b_action_fn,
        nonlocal_xc_a_diagonal=nonlocal_xc_a_diagonal,
    )


def _restricted_xc_action(data: _RestrictedResponseOperatorData, x: Array) -> Array:
    out = jnp.zeros_like(x)
    if data.xc_response_action_fn is not None:
        out = out + data.xc_response_action_fn(x)
    return out


def _restricted_xc_diagonal(data: _RestrictedResponseOperatorData) -> Array:
    out = jnp.zeros_like(data.delta_eps)
    if data.xc_response_diagonal is not None:
        out = out + jnp.asarray(data.xc_response_diagonal, dtype=out.dtype)
    if data.nonlocal_xc_a_diagonal is not None:
        out = out + jnp.asarray(data.nonlocal_xc_a_diagonal, dtype=out.dtype)
    return out


def _restricted_a_action(data: _RestrictedResponseOperatorData, x: Array) -> Array:
    out = x * data.delta_eps[None, :, :]
    if data.effective_tda_eri is not None:
        out = out + jnp.einsum(
            "iajb,njb->nia",
            data.effective_tda_eri,
            x,
            precision=Precision.HIGHEST,
        )
    else:
        out = out + 2.0 * jnp.einsum(
            "iajb,njb->nia",
            data.eri_ovov,
            x,
            precision=Precision.HIGHEST,
        )
        if data.eri_oovv is not None:
            alpha = jnp.asarray(data.hybrid_fraction, dtype=x.dtype)
            out = out - alpha * jnp.einsum(
                "ijab,njb->nia",
                data.eri_oovv,
                x,
                precision=Precision.HIGHEST,
            )
    if data.hybrid_exchange_a_action_fn is not None:
        out = out + data.hybrid_exchange_a_action_fn(x)
    out = out + _restricted_xc_action(data, x)
    if data.nonlocal_xc_a_action_fn is not None:
        out = out + data.nonlocal_xc_a_action_fn(x)
    return out


def _restricted_b_action(data: _RestrictedResponseOperatorData, x: Array) -> Array:
    if data.effective_b_eri is not None:
        out = jnp.einsum(
            "iabj,njb->nia",
            data.effective_b_eri,
            x,
            precision=Precision.HIGHEST,
        )
    else:
        if data.eri_ovvo is None:
            raise ValueError("B-response action requires eri_ovvo terms.")
        out = 2.0 * jnp.einsum(
            "iabj,njb->nia",
            data.eri_ovvo,
            x,
            precision=Precision.HIGHEST,
        )
        alpha = jnp.asarray(data.hybrid_fraction, dtype=x.dtype)
        out = out - alpha * jnp.einsum(
            "iabj,njb->nia",
            jnp.transpose(data.eri_ovvo, (0, 2, 1, 3)),
            x,
            precision=Precision.HIGHEST,
        )
    if data.hybrid_exchange_b_action_fn is not None:
        out = out + data.hybrid_exchange_b_action_fn(x)
    out = out + _restricted_xc_action(data, x)
    if data.nonlocal_xc_b_action_fn is not None:
        out = out + data.nonlocal_xc_b_action_fn(x)
    return out


def _restricted_tda_diagonal(data: _RestrictedResponseOperatorData) -> Array:
    alpha = jnp.asarray(data.hybrid_fraction, dtype=data.delta_eps.dtype)
    diagonal = data.delta_eps + 2.0 * jnp.einsum(
        "iaia->ia",
        data.eri_ovov,
        precision=Precision.HIGHEST,
    )
    if data.eri_oovv is not None:
        diagonal = diagonal - alpha * jnp.einsum(
            "iiaa->ia",
            data.eri_oovv,
            precision=Precision.HIGHEST,
        )
    if data.hybrid_exchange_diagonal is not None:
        diagonal = diagonal + jnp.asarray(
            data.hybrid_exchange_diagonal,
            dtype=diagonal.dtype,
        )
    return diagonal + _restricted_xc_diagonal(data)


def build_restricted_tda_operator(
    molecule: Any,
    xc_functional: Any | None = None,
    *,
    xc_params: Any | None = None,
    occupation_tolerance: float = 1e-8,
):
    data = _build_restricted_response_operator_data(
        molecule,
        xc_functional,
        xc_params=xc_params,
        occupation_tolerance=occupation_tolerance,
        need_b_terms=False,
    )
    delta_eps = data.delta_eps
    nocc, nvir = delta_eps.shape
    dim = int(nocc * nvir)

    diagonal = _restricted_tda_diagonal(data).reshape(-1)

    def vind(x: Array) -> Array:
        x = jnp.asarray(x).reshape(-1, nocc, nvir)
        return _restricted_a_action(data, x).reshape(-1, dim)

    return vind, diagonal, delta_eps


def build_restricted_tdhf_operator(
    molecule: Any,
    xc_functional: Any | None = None,
    *,
    xc_params: Any | None = None,
    occupation_tolerance: float = 1e-8,
):
    data = _build_restricted_response_operator_data(
        molecule,
        xc_functional,
        xc_params=xc_params,
        occupation_tolerance=occupation_tolerance,
    )
    delta_eps = data.delta_eps
    nocc, nvir = delta_eps.shape
    dim = int(nocc * nvir)

    def vind(z: Array) -> Array:
        z = jnp.asarray(z).reshape(-1, 2 * dim)
        x = z[:, :dim].reshape(-1, nocc, nvir)
        y = z[:, dim:].reshape(-1, nocc, nvir)
        upper = _restricted_a_action(data, x) + _restricted_b_action(data, y)
        lower = -(_restricted_b_action(data, x) + _restricted_a_action(data, y))
        return jnp.concatenate(
            [upper.reshape(-1, dim), lower.reshape(-1, dim)],
            axis=-1,
        )

    return vind


def gen_tda_vind(
    molecule: Any,
    xc_functional: Any | None = None,
    *,
    xc_params: Any | None = None,
    occupation_tolerance: float = 1e-8,
):
    vind, _, _ = build_restricted_tda_operator(
        molecule,
        xc_functional,
        xc_params=xc_params,
        occupation_tolerance=occupation_tolerance,
    )
    return vind


def gen_tdhf_vind(
    molecule: Any,
    xc_functional: Any | None = None,
    *,
    xc_params: Any | None = None,
    occupation_tolerance: float = 1e-8,
):
    return build_restricted_tdhf_operator(
        molecule,
        xc_functional,
        xc_params=xc_params,
        occupation_tolerance=occupation_tolerance,
    )
