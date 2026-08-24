from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.lax import Precision
from jaxtyping import Array

from ..features import (
    infer_response_feature_kind,
    normalize_response_feature_kind,
)
from ..data.integrals.jax.packed_eri import _metadata_arrays
from ._utils import (
    _density_on_grid,
    _resolve_xc_functional,
    _restricted_orbital_data,
)
from .lowrank_response import build_restricted_lowrank_mo_response_action
from .response_options import (
    ResponseKernelOptions,
    normalize_response_kernel_options,
)


_RESPONSE_FEATURE_COUNTS = {
    "LDA": 1,
    "GGA": 4,
    "MGGA": 5,
    "MGGA_LAPL": 6,
}


@dataclass(frozen=True)
class _RestrictedResponseOperatorData:
    delta_eps: Array
    orbo: Array
    orbv: Array
    ao_response_action_fn: Callable[[Array], Array]
    ao_mo_response_action_fn: Callable[..., Array] | None = None
    xc_response_action_fn: Callable[[Array], Array] | None = None
    hybrid_fraction: Array | float = 0.0
    nonlocal_xc_a_action_fn: Callable[[Array], Array] | None = None
    nonlocal_xc_b_action_fn: Callable[[Array], Array] | None = None


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

    alpha_raw = getattr(resolved_xc, "exact_exchange_fraction", 0.0)
    alpha_is_zero_scalar = (
        isinstance(alpha_raw, (int, float, np.number))
        and abs(float(alpha_raw)) <= 1e-14
    )
    alpha = jnp.asarray(alpha_raw)
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
        if alpha_is_zero_scalar:
            return 0.0
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
    if isinstance(value, (int, float, np.number)):
        return bool(abs(float(value)) > 1e-14)
    arr = jnp.asarray(value)
    if any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(arr)):
        return True
    return bool(abs(float(np.asarray(arr).reshape(()))) > 1e-14)


def _raise_if_strict_local_hf_response(resolved_xc: Any | None) -> None:
    if resolved_xc is None:
        return
    mode = str(getattr(resolved_xc, "response_hf_mode", "approx")).lower()
    if mode == "strict":
        raise NotImplementedError(
            "strict local-HF TDDFT response requires chi/fxx-based second-response "
            "contractions and is not implemented. Use response_hf_mode='approx'."
        )


def _restricted_response_features(
    molecule: Any,
    orbo: Array,
    orbv: Array,
    *,
    feature_kind: str,
    dtype: Any,
) -> Array:
    ao = jnp.asarray(molecule.ao, dtype=dtype)
    rho_o = jnp.einsum("gp,pi->gi", ao, orbo, precision=Precision.HIGHEST)
    rho_v = jnp.einsum("gp,pa->ga", ao, orbv, precision=Precision.HIGHEST)
    rho_ov = jnp.einsum("gi,ga->gia", rho_o, rho_v, precision=Precision.HIGHEST)
    if feature_kind == "LDA":
        return rho_ov[None, ...]

    ao_deriv1 = getattr(molecule, "ao_deriv1", None)
    if ao_deriv1 is None:
        raise AttributeError(
            "Molecule-like object must define ao_deriv1 for GGA/meta-GGA transition features."
        )
    ao_deriv1 = jnp.asarray(ao_deriv1, dtype=dtype)
    if ao_deriv1.shape[0] < 4:
        raise ValueError("ao_deriv1 must contain AO values plus first derivatives.")
    rho_o_full = jnp.einsum(
        "xgp,pi->xgi",
        ao_deriv1[:4],
        orbo,
        precision=Precision.HIGHEST,
    )
    rho_v_full = jnp.einsum(
        "xgp,pa->xga",
        ao_deriv1[:4],
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
    ao_laplacian = jnp.asarray(ao_laplacian, dtype=dtype)
    lapl_o = jnp.einsum("gp,pi->gi", ao_laplacian, orbo, precision=Precision.HIGHEST)
    lapl_v = jnp.einsum("gp,pa->ga", ao_laplacian, orbv, precision=Precision.HIGHEST)
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


@dataclass(frozen=True)
class _RestrictedResponseFactors:
    feature_kind: str
    channels: tuple[tuple[tuple[float, Array, Array], ...], ...]


def _restricted_response_factors(
    molecule: Any,
    orbo: Array,
    orbv: Array,
    *,
    feature_kind: str,
    dtype: Any,
) -> _RestrictedResponseFactors:
    ao = jnp.asarray(molecule.ao, dtype=dtype)
    rho_o = jnp.einsum("gp,pi->gi", ao, orbo, precision=Precision.HIGHEST)
    rho_v = jnp.einsum("gp,pa->ga", ao, orbv, precision=Precision.HIGHEST)
    channels: list[tuple[tuple[float, Array, Array], ...]] = [((1.0, rho_o, rho_v),)]
    if feature_kind == "LDA":
        return _RestrictedResponseFactors(feature_kind, tuple(channels))

    ao_deriv1 = getattr(molecule, "ao_deriv1", None)
    if ao_deriv1 is None:
        raise AttributeError(
            "Molecule-like object must define ao_deriv1 for GGA/meta-GGA transition features."
        )
    ao_deriv1 = jnp.asarray(ao_deriv1, dtype=dtype)
    if ao_deriv1.shape[0] < 4:
        raise ValueError("ao_deriv1 must contain AO values plus first derivatives.")
    rho_o_full = jnp.einsum(
        "xgp,pi->xgi",
        ao_deriv1[:4],
        orbo,
        precision=Precision.HIGHEST,
    )
    rho_v_full = jnp.einsum(
        "xgp,pa->xga",
        ao_deriv1[:4],
        orbv,
        precision=Precision.HIGHEST,
    )
    channels.extend(
        (
            (1.0, rho_o_full[axis], rho_v),
            (1.0, rho_o, rho_v_full[axis]),
        )
        for axis in range(1, 4)
    )
    if feature_kind == "GGA":
        return _RestrictedResponseFactors(feature_kind, tuple(channels))

    channels.append(
        tuple((0.5, rho_o_full[axis], rho_v_full[axis]) for axis in range(1, 4))
    )
    if feature_kind == "MGGA":
        return _RestrictedResponseFactors(feature_kind, tuple(channels))
    if feature_kind != "MGGA_LAPL":
        raise ValueError(f"Unsupported response_feature_kind={feature_kind!r}.")

    ao_laplacian = getattr(molecule, "ao_laplacian", None)
    if ao_laplacian is None:
        raise AttributeError(
            "Molecule-like object must define ao_laplacian for MGGA_LAPL transition features."
        )
    ao_laplacian = jnp.asarray(ao_laplacian, dtype=dtype)
    lapl_o = jnp.einsum("gp,pi->gi", ao_laplacian, orbo, precision=Precision.HIGHEST)
    lapl_v = jnp.einsum("gp,pa->ga", ao_laplacian, orbv, precision=Precision.HIGHEST)
    lapl_terms = [(1.0, lapl_o, rho_v)]
    lapl_terms.extend((2.0, rho_o_full[axis], rho_v_full[axis]) for axis in range(1, 4))
    lapl_terms.append((1.0, rho_o, lapl_v))
    channels.append(tuple(lapl_terms))
    return _RestrictedResponseFactors(feature_kind, tuple(channels))


def _project_restricted_transition_to_grid(
    factors: _RestrictedResponseFactors,
    values: Array,
) -> Array:
    projected = []
    for channel in factors.channels:
        total = None
        for coefficient, left, right in channel:
            term = coefficient * jnp.einsum(
                "gi,nia,ga->ng",
                left,
                values,
                right,
                precision=Precision.HIGHEST,
            )
            total = term if total is None else total + term
        projected.append(total)
    return jnp.stack(projected, axis=1)


def _project_grid_response_to_restricted_transition(
    factors: _RestrictedResponseFactors,
    weighted: Array,
) -> Array:
    first_left = factors.channels[0][0][1]
    first_right = factors.channels[0][0][2]
    out = jnp.zeros(
        (int(weighted.shape[0]), int(first_left.shape[1]), int(first_right.shape[1])),
        dtype=weighted.dtype,
    )
    for channel_idx, channel in enumerate(factors.channels):
        channel_weight = weighted[:, channel_idx, :]
        for coefficient, left, right in channel:
            out = out + coefficient * jnp.einsum(
                "gi,ng,ga->nia",
                left,
                channel_weight,
                right,
                precision=Precision.HIGHEST,
            )
    return out

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
    nocc = int(orbo.shape[1])
    nvir = int(orbv.shape[1])
    features = _restricted_response_features(
        molecule,
        orbo,
        orbv,
        feature_kind=feature_kind,
        dtype=dtype,
    )

    def action(x: Array) -> Array:
        original_shape = jnp.asarray(x).shape
        values = jnp.asarray(x, dtype=dtype).reshape(-1, nocc, nvir)
        projected = jnp.einsum(
            "xgia,nia->nxg",
            features,
            values,
            precision=Precision.HIGHEST,
        )
        weighted = jnp.einsum(
            "xyg,nyg->nxg",
            weighted_hessian,
            projected,
            precision=Precision.HIGHEST,
        )
        out = 2.0 * jnp.einsum(
            "xgia,nxg->nia",
            features,
            weighted,
            precision=Precision.HIGHEST,
        )
        return out.reshape(original_shape)

    diagonal = 2.0 * jnp.einsum(
        "xyg,xgia,ygia->ia",
        weighted_hessian,
        features,
        features,
        precision=Precision.HIGHEST,
    )
    return action, diagonal


def _restricted_grid_xc_response_hvp(
    molecule: Any,
    orbo: Array,
    orbv: Array,
    response_hvp: Callable[..., Array],
    *,
    feature_kind: str,
    dtype: Any,
) -> Callable[[Array], Array]:
    weights = jnp.asarray(molecule.grid.weights, dtype=dtype)
    nocc = int(orbo.shape[1])
    nvir = int(orbv.shape[1])
    factors = _restricted_response_factors(
        molecule,
        orbo,
        orbv,
        feature_kind=feature_kind,
        dtype=dtype,
    )

    def action(x: Array) -> Array:
        original_shape = jnp.asarray(x).shape
        values = jnp.asarray(x, dtype=dtype).reshape(-1, nocc, nvir)
        projected = _project_restricted_transition_to_grid(factors, values)
        weighted = response_hvp(molecule, projected)
        weighted = jnp.asarray(weighted, dtype=dtype) * weights[None, None, :]
        out = 2.0 * _project_grid_response_to_restricted_transition(factors, weighted)
        return out.reshape(original_shape)

    return action


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
    need_b_terms: bool = True,
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
    if action_a_raw is None or (need_b_terms and action_b_raw is None):
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
        _wrap_action(action_b_raw, "Nonlocal B response action")
        if need_b_terms and action_b_raw is not None
        else None,
        diagonal,
    )


def _restricted_transition_density(orbo: Array, orbv: Array, values: Array, *, bottom: bool) -> Array:
    values = jnp.asarray(values)
    if bottom:
        return 2.0 * jnp.einsum(
            "nia,pi,qa->npq",
            values,
            orbo,
            orbv,
            precision=Precision.HIGHEST,
        )
    return 2.0 * jnp.einsum(
        "nia,pa,qi->npq",
        values,
        orbv,
        orbo,
        precision=Precision.HIGHEST,
    )


def _restricted_project_response(
    response_ao: Array,
    orbo: Array,
    orbv: Array,
    *,
    bottom: bool,
) -> Array:
    if bottom:
        return jnp.einsum(
            "npq,pi,qa->nia",
            response_ao,
            orbo,
            orbv,
            precision=Precision.HIGHEST,
        )
    return jnp.einsum(
        "npq,qi,pa->nia",
        response_ao,
        orbo,
        orbv,
        precision=Precision.HIGHEST,
    )


def _jk_from_full_eri(eri: Array, density: Array) -> tuple[Array, Array]:
    j_mat = jnp.einsum("pqrs,nrs->npq", eri, density, precision=Precision.HIGHEST)
    k_mat = jnp.einsum("prqs,nrs->npq", eri, density, precision=Precision.HIGHEST)
    return j_mat, k_mat


def _j_from_full_eri(eri: Array, density: Array) -> Array:
    return jnp.einsum("pqrs,nrs->npq", eri, density, precision=Precision.HIGHEST)


def _jk_from_df_factors(df_factors: Array, density: Array) -> tuple[Array, Array]:
    rho_aux = jnp.einsum("Qpq,npq->nQ", df_factors, density, precision=Precision.HIGHEST)
    j_mat = jnp.einsum("nQ,Qpq->npq", rho_aux, df_factors, precision=Precision.HIGHEST)
    k_mat = jnp.einsum(
        "Qpr,Qqs,nrs->npq",
        df_factors,
        df_factors,
        density,
        precision=Precision.HIGHEST,
    )
    return j_mat, k_mat


def _j_from_df_factors(df_factors: Array, density: Array) -> Array:
    rho_aux = jnp.einsum("Qpq,npq->nQ", df_factors, density, precision=Precision.HIGHEST)
    return jnp.einsum("nQ,Qpq->npq", rho_aux, df_factors, precision=Precision.HIGHEST)


def _jk_from_eri_pair_matrix(eri_pair_matrix: Array, density: Array) -> tuple[Array, Array]:
    pair = jnp.asarray(eri_pair_matrix)
    density = jnp.asarray(density)
    nao = int(density.shape[-1])
    rows, cols, pair_index, _ = _metadata_arrays(nao, density.dtype)
    offdiag = (rows != cols)[None, :]
    density_pair = density[:, rows, cols] + jnp.where(
        offdiag,
        density[:, cols, rows],
        jnp.zeros_like(density[:, rows, cols]),
    )
    j_pair = jnp.einsum("PQ,nQ->nP", pair, density_pair, precision=Precision.HIGHEST)
    j_mat = jnp.zeros_like(density)
    j_mat = j_mat.at[:, rows, cols].set(j_pair)
    j_mat = j_mat.at[:, cols, rows].set(j_pair)

    ao = jnp.arange(nao, dtype=jnp.int32)
    qs_by_q = pair_index[:, ao]

    def k_row(p: Array) -> Array:
        pr = pair_index[p, ao]
        blocks = pair[pr[None, :, None], qs_by_q[:, None, :]]
        return jnp.einsum("qrs,nrs->nq", blocks, density, precision=Precision.HIGHEST)

    k_mat = jnp.transpose(jax.vmap(k_row)(ao), (1, 0, 2))
    return j_mat, k_mat


def _j_from_eri_pair_matrix(eri_pair_matrix: Array, density: Array) -> Array:
    pair = jnp.asarray(eri_pair_matrix)
    density = jnp.asarray(density)
    nao = int(density.shape[-1])
    rows, cols, _, _ = _metadata_arrays(nao, density.dtype)
    offdiag = (rows != cols)[None, :]
    density_pair = density[:, rows, cols] + jnp.where(
        offdiag,
        density[:, cols, rows],
        jnp.zeros_like(density[:, rows, cols]),
    )
    j_pair = jnp.einsum("PQ,nQ->nP", pair, density_pair, precision=Precision.HIGHEST)
    j_mat = jnp.zeros_like(density)
    j_mat = j_mat.at[:, rows, cols].set(j_pair)
    return j_mat.at[:, cols, rows].set(j_pair)


def _restricted_df_mo_response_action(
    df_factors: Array,
    orbo: Array,
    orbv: Array,
    hybrid_fraction: Any,
    *,
    include_exchange: bool,
    dtype: Any,
) -> Callable[..., Array]:
    return build_restricted_lowrank_mo_response_action(
        df_factors,
        df_factors,
        orbo,
        orbv,
        hybrid_fraction,
        include_exchange=include_exchange,
        dtype=dtype,
    )


def _restricted_ao_response_action(
    molecule: Any,
    hybrid_fraction: Any,
    *,
    include_exchange: bool,
    dtype: Any,
    two_electron_mode: str = "auto",
) -> Callable[[Array], Array]:
    alpha = jnp.asarray(hybrid_fraction, dtype=dtype)
    df_factors = getattr(molecule, "df_factors", None)
    mode = str(two_electron_mode).lower()
    if mode == "df" and (df_factors is None or int(jnp.asarray(df_factors).size) == 0):
        raise ValueError(
            'response_two_electron_mode="df" requires molecule.df_factors. '
            'Build the reference with response_df_mode="df" or jk_backend="df".'
        )
    if mode in {"auto", "df"} and df_factors is not None and int(jnp.asarray(df_factors).size) > 0:
        source = jnp.asarray(df_factors, dtype=dtype)
        jk_fn = lambda density: _jk_from_df_factors(source, density)
        j_fn = lambda density: _j_from_df_factors(source, density)
    else:
        eri_pair_matrix = getattr(molecule, "eri_pair_matrix", None)
        if eri_pair_matrix is not None and int(jnp.asarray(eri_pair_matrix).size) > 0:
            source = jnp.asarray(eri_pair_matrix, dtype=dtype)
            jk_fn = lambda density: _jk_from_eri_pair_matrix(source, density)
            j_fn = lambda density: _j_from_eri_pair_matrix(source, density)
        else:
            rep_tensor = getattr(molecule, "rep_tensor", None)
            if rep_tensor is None or int(jnp.asarray(rep_tensor).size) == 0:
                raise ValueError(
                    "The molecule must provide rep_tensor or eri_pair_matrix for the AO "
                    "response action. response_two_electron_mode=\"df\" requires "
                    "df_factors."
                )
            source = jnp.asarray(rep_tensor, dtype=dtype)
            jk_fn = lambda density: _jk_from_full_eri(source, density)
            j_fn = lambda density: _j_from_full_eri(source, density)

    def action(density: Array) -> Array:
        original_shape = jnp.asarray(density).shape
        density_ao = jnp.asarray(density, dtype=dtype).reshape(-1, original_shape[-2], original_shape[-1])
        if not include_exchange:
            return j_fn(density_ao).reshape(original_shape)
        j_mat, k_mat = jk_fn(density_ao)
        return (j_mat - 0.5 * alpha * k_mat).reshape(original_shape)

    return action


def _unused_ao_response_action(_density: Array) -> Array:
    raise RuntimeError("AO response action is unavailable for the selected low-rank backend.")


def _require_nonempty_factor(value: Any, *, name: str, mode_hint: str) -> Array:
    if value is None or int(jnp.asarray(value).size) == 0:
        raise ValueError(f"{name} is required. Build the reference with {mode_hint}.")
    return value


def _restricted_lowrank_response_action_for_options(
    molecule: Any,
    orbo: Array,
    orbv: Array,
    hybrid_fraction: Any,
    *,
    response_kernel_options: ResponseKernelOptions,
    include_exchange: bool,
    need_b_terms: bool,
    dtype: Any,
) -> Callable[..., Array] | None:
    mode = response_kernel_options.two_electron_mode
    if mode == "direct":
        return None

    df_factors = getattr(molecule, "df_factors", None)
    if mode == "auto":
        if df_factors is None or int(jnp.asarray(df_factors).size) == 0:
            return None
        return _restricted_df_mo_response_action(
            df_factors,
            orbo,
            orbv,
            hybrid_fraction,
            include_exchange=include_exchange,
            dtype=dtype,
        )

    if mode == "df":
        response_j = getattr(molecule, "response_df_factors_j", None)
        response_k = getattr(molecule, "response_df_factors_k", None)
        if response_j is None or int(jnp.asarray(response_j).size) == 0:
            response_j = _require_nonempty_factor(
                df_factors,
                name="molecule.df_factors or molecule.response_df_factors_j",
                mode_hint='response_df_mode="df" or jk_backend="df"',
            )
        if response_k is None or int(jnp.asarray(response_k).size) == 0:
            response_k = response_j
        return build_restricted_lowrank_mo_response_action(
            response_j,
            response_k,
            orbo,
            orbv,
            hybrid_fraction,
            include_exchange=include_exchange,
            dtype=dtype,
        )

    if mode == "ris":
        if need_b_terms:
            raise NotImplementedError(
                "RIS response_two_electron_mode is currently implemented for restricted "
                "TDA only; full Casida TDDFT still requires the RIS (ib|ja) B-term path."
            )
        j_factors = _require_nonempty_factor(
            getattr(molecule, "response_df_factors_j", None),
            name="molecule.response_df_factors_j",
            mode_hint='response_df_mode="ris"',
        )
        k_factors = None
        if include_exchange:
            k_factors = _require_nonempty_factor(
                getattr(molecule, "response_df_factors_k", None),
                name="molecule.response_df_factors_k",
                mode_hint='response_df_mode="ris"',
            )
        return build_restricted_lowrank_mo_response_action(
            j_factors,
            k_factors,
            orbo,
            orbv,
            hybrid_fraction,
            include_exchange=include_exchange,
            dtype=dtype,
        )

    raise ValueError(f"Unsupported response two-electron mode {mode!r}.")


def _build_restricted_response_operator_data(
    molecule: Any,
    xc_functional: Any | None = None,
    *,
    xc_params: Any | None = None,
    occupation_tolerance: float = 1e-8,
    need_b_terms: bool = True,
    response_kernel_options: ResponseKernelOptions | dict[str, Any] | None = None,
) -> _RestrictedResponseOperatorData:
    response_options = normalize_response_kernel_options(response_kernel_options)
    resolved_xc = _resolve_xc_functional(molecule, xc_functional, xc_params)
    _raise_if_strict_local_hf_response(resolved_xc)
    weights = jnp.asarray(molecule.grid.weights)
    orbo, orbv, delta_eps, _ = _restricted_orbital_data(
        molecule,
        occupation_tolerance,
    )

    xc_response_action_fn = None
    hybrid_fraction = 0.0
    nonlocal_xc_a_action_fn = None
    nonlocal_xc_b_action_fn = None

    if resolved_xc is not None:
        total_density = _density_on_grid(molecule)
        hybrid_fraction = _strict_hybrid_fraction(
            resolved_xc,
            molecule,
            total_density,
        )
        grid_response_tensor = getattr(resolved_xc, "grid_response_tensor", None)
        grid_response_hvp = getattr(resolved_xc, "grid_response_hvp", None)
        if callable(grid_response_hvp):
            feature_kind = normalize_response_feature_kind(
                getattr(resolved_xc, "response_feature_kind", None),
                default="LDA",
                label="response_feature_kind",
            )
            xc_response_action_fn = _restricted_grid_xc_response_hvp(
                molecule,
                orbo,
                orbv,
                grid_response_hvp,
                feature_kind=str(feature_kind),
                dtype=jnp.asarray(weights).dtype,
            )
        elif callable(grid_response_tensor):
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
            xc_response_action_fn, _ = _restricted_grid_xc_response(
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
            xc_response_action_fn, _ = _restricted_grid_xc_response(
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
            _,
        ) = _resolve_nonlocal_response_action_pair(
            resolved_xc,
            molecule,
            delta_eps=delta_eps,
            occupation_tolerance=occupation_tolerance,
            need_b_terms=need_b_terms,
        )
        if pair_nonlocal_a_action is not None:
            nonlocal_xc_a_action_fn = pair_nonlocal_a_action
            if need_b_terms:
                nonlocal_xc_b_action_fn = pair_nonlocal_b_action

    response_dtype = jnp.result_type(delta_eps, weights)
    include_exchange = _needs_exchange_terms(hybrid_fraction)
    ao_mo_response_action_fn = _restricted_lowrank_response_action_for_options(
        molecule,
        orbo,
        orbv,
        hybrid_fraction,
        response_kernel_options=response_options,
        include_exchange=include_exchange,
        need_b_terms=need_b_terms,
        dtype=response_dtype,
    )
    if ao_mo_response_action_fn is None:
        ao_response_action_fn = _restricted_ao_response_action(
            molecule,
            hybrid_fraction,
            include_exchange=include_exchange,
            dtype=response_dtype,
            two_electron_mode=response_options.two_electron_mode,
        )
    else:
        ao_response_action_fn = _unused_ao_response_action

    return _RestrictedResponseOperatorData(
        delta_eps=delta_eps,
        orbo=orbo,
        orbv=orbv,
        ao_response_action_fn=ao_response_action_fn,
        ao_mo_response_action_fn=ao_mo_response_action_fn,
        xc_response_action_fn=xc_response_action_fn,
        hybrid_fraction=hybrid_fraction,
        nonlocal_xc_a_action_fn=nonlocal_xc_a_action_fn,
        nonlocal_xc_b_action_fn=nonlocal_xc_b_action_fn,
    )


def _restricted_xc_action(data: _RestrictedResponseOperatorData, x: Array) -> Array:
    out = jnp.zeros_like(x)
    if data.xc_response_action_fn is not None:
        out = out + data.xc_response_action_fn(x)
    return out


def _restricted_ao_mo_action(
    data: _RestrictedResponseOperatorData,
    x: Array,
    *,
    bottom_density: bool,
    bottom_projection: bool,
) -> Array:
    if data.ao_mo_response_action_fn is not None:
        return data.ao_mo_response_action_fn(
            x,
            bottom_density=bottom_density,
            bottom_projection=bottom_projection,
        )
    density = _restricted_transition_density(
        data.orbo,
        data.orbv,
        x,
        bottom=bottom_density,
    )
    response_ao = data.ao_response_action_fn(density)
    return _restricted_project_response(
        response_ao,
        data.orbo,
        data.orbv,
        bottom=bottom_projection,
    )


def _restricted_a_action(data: _RestrictedResponseOperatorData, x: Array) -> Array:
    out = x * data.delta_eps[None, :, :]
    out = out + _restricted_ao_mo_action(
        data,
        x,
        bottom_density=False,
        bottom_projection=False,
    )
    out = out + _restricted_xc_action(data, x)
    if data.nonlocal_xc_a_action_fn is not None:
        out = out + data.nonlocal_xc_a_action_fn(x)
    return out


def _restricted_b_action(data: _RestrictedResponseOperatorData, x: Array) -> Array:
    out = _restricted_ao_mo_action(
        data,
        x,
        bottom_density=True,
        bottom_projection=False,
    )
    out = out + _restricted_xc_action(data, x)
    if data.nonlocal_xc_b_action_fn is not None:
        out = out + data.nonlocal_xc_b_action_fn(x)
    return out


def _restricted_tda_diagonal(data: _RestrictedResponseOperatorData) -> Array:
    return data.delta_eps


def _restricted_tdhf_action(
    data: _RestrictedResponseOperatorData,
    x: Array,
    y: Array,
) -> tuple[Array, Array]:
    return (
        _restricted_a_action(data, x) + _restricted_b_action(data, y),
        _restricted_b_action(data, x) + _restricted_a_action(data, y),
    )


def build_restricted_tda_operator(
    molecule: Any,
    xc_functional: Any | None = None,
    *,
    xc_params: Any | None = None,
    occupation_tolerance: float = 1e-8,
    response_kernel_options: ResponseKernelOptions | dict[str, Any] | None = None,
):
    data = _build_restricted_response_operator_data(
        molecule,
        xc_functional,
        xc_params=xc_params,
        occupation_tolerance=occupation_tolerance,
        need_b_terms=False,
        response_kernel_options=response_kernel_options,
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
    response_kernel_options: ResponseKernelOptions | dict[str, Any] | None = None,
):
    data = _build_restricted_response_operator_data(
        molecule,
        xc_functional,
        xc_params=xc_params,
        occupation_tolerance=occupation_tolerance,
        response_kernel_options=response_kernel_options,
    )
    delta_eps = data.delta_eps
    nocc, nvir = delta_eps.shape
    dim = int(nocc * nvir)

    def vind(z: Array) -> Array:
        z = jnp.asarray(z).reshape(-1, 2 * dim)
        x = z[:, :dim].reshape(-1, nocc, nvir)
        y = z[:, dim:].reshape(-1, nocc, nvir)
        upper, lower = _restricted_tdhf_action(data, x, y)
        return jnp.concatenate(
            [upper.reshape(-1, dim), -lower.reshape(-1, dim)],
            axis=-1,
        )

    return vind


def gen_tda_vind(
    molecule: Any,
    xc_functional: Any | None = None,
    *,
    xc_params: Any | None = None,
    occupation_tolerance: float = 1e-8,
    response_kernel_options: ResponseKernelOptions | dict[str, Any] | None = None,
):
    vind, _, _ = build_restricted_tda_operator(
        molecule,
        xc_functional,
        xc_params=xc_params,
        occupation_tolerance=occupation_tolerance,
        response_kernel_options=response_kernel_options,
    )
    return vind


def gen_tdhf_vind(
    molecule: Any,
    xc_functional: Any | None = None,
    *,
    xc_params: Any | None = None,
    occupation_tolerance: float = 1e-8,
    response_kernel_options: ResponseKernelOptions | dict[str, Any] | None = None,
):
    return build_restricted_tdhf_operator(
        molecule,
        xc_functional,
        xc_params=xc_params,
        occupation_tolerance=occupation_tolerance,
        response_kernel_options=response_kernel_options,
    )
