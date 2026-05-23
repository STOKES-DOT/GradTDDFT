from __future__ import annotations

from dataclasses import dataclass
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
    restricted_transition_response_features,
)
from ..data.integrals import eri_pair_matrix_to_mo_eri_slices
from ._utils import (
    _density_on_grid,
    _resolve_xc_functional,
    _restricted_orbital_data,
)
from .types import TDDFTMatrices


@dataclass(frozen=True)
class _RestrictedResponseOperatorData:
    delta_eps: Array
    eri_ovov: Array
    eri_ovvo: Array
    eri_oovv: Array | None
    effective_tda_eri: Array | None = None
    effective_b_eri: Array | None = None
    weighted_local_kernel: Array | None = None
    rho_ov_density: Array | None = None
    weighted_strict_tensor: Array | None = None
    response_features: Array | None = None
    hybrid_fraction: Array | float = 0.0
    nonlocal_xc_action_fn: Callable[[Array], Array] | None = None
    nonlocal_xc_diagonal: Array | None = None
    nonlocal_xc_a_action_fn: Callable[[Array], Array] | None = None
    nonlocal_xc_b_action_fn: Callable[[Array], Array] | None = None
    nonlocal_xc_a_diagonal: Array | None = None


@jax.jit
def _tda_df_hartree_flat(df_factors: Array, orbo: Array, orbv: Array) -> Array:
    b_ov = jnp.einsum(
        "Qpq,pi,qa->Qia",
        df_factors,
        orbo,
        orbv,
        precision=Precision.HIGHEST,
    )
    dim = int(orbo.shape[1] * orbv.shape[1])
    b_ov_flat = b_ov.reshape(df_factors.shape[0], dim)
    return 2.0 * jnp.einsum(
        "Qd,Qe->de",
        b_ov_flat,
        b_ov_flat,
        precision=Precision.HIGHEST,
    )


@jax.jit
def _tda_df_exchange_flat(df_factors: Array, orbo: Array, orbv: Array) -> Array:
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
    dim = int(orbo.shape[1] * orbv.shape[1])
    return jnp.einsum(
        "Qij,Qab->iajb",
        b_oo,
        b_vv,
        precision=Precision.HIGHEST,
    ).reshape(dim, dim)


@jax.jit
def _tda_strict_xc_flat(
    weighted_strict_tensor: Array,
    response_features_flat: Array,
) -> Array:
    return 2.0 * jnp.einsum(
        "xyr,xrd,yre->de",
        weighted_strict_tensor,
        response_features_flat,
        response_features_flat,
    )


@jax.jit
def _tda_lda_xc_flat(
    weighted_local_kernel: Array,
    rho_ov_flat: Array,
) -> Array:
    return 2.0 * jnp.einsum(
        "r,rd,re->de",
        weighted_local_kernel,
        rho_ov_flat,
        rho_ov_flat,
    )


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


def _as_nonlocal_response_matrix(
    values: Any,
    *,
    nocc: int,
    nvir: int,
    label: str = "Nonlocal response matrix",
) -> Array:
    dim = int(nocc * nvir)
    arr = jnp.asarray(values)
    if arr.shape == (dim, dim):
        matrix = arr
    elif arr.shape == (nocc, nvir, nocc, nvir):
        matrix = arr.reshape(dim, dim)
    else:
        raise ValueError(
            f"{label} must have shape {(dim, dim)} or {(nocc, nvir, nocc, nvir)}, "
            f"got {arr.shape}."
        )
    _assert_finite(matrix, label=label)
    return matrix


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


def _materialize_nonlocal_response_matrix_from_action(
    action_fn: Callable[[Array], Array],
    *,
    nocc: int,
    nvir: int,
    dtype: Any,
) -> Array:
    dim = int(nocc * nvir)
    basis = jnp.eye(dim, dtype=dtype).reshape(dim, nocc, nvir)
    action_basis = jnp.asarray(action_fn(basis), dtype=dtype)
    if action_basis.shape != basis.shape:
        raise ValueError(
            "nonlocal_response_action must preserve the transition-amplitude shape "
            f"{basis.shape}, got {action_basis.shape}."
        )
    _assert_finite(action_basis, label="Nonlocal response action")
    return action_basis.reshape(dim, dim).T


def _resolve_nonlocal_response_terms(
    resolved_xc: Any,
    molecule: Any,
    *,
    delta_eps: Array,
    occupation_tolerance: float,
) -> tuple[Array | None, Callable[[Array], Array] | None, Array | None]:
    if resolved_xc is None:
        return None, None, None
    if getattr(resolved_xc, "nonlocal_response_matrices_fn", None) is not None:
        return None, None, None

    nocc, nvir = delta_eps.shape
    matrix_fn = getattr(resolved_xc, "nonlocal_response_matrix", None)
    action_fn_raw = getattr(resolved_xc, "nonlocal_response_action", None)
    diagonal_fn = getattr(resolved_xc, "nonlocal_response_diagonal", None)

    matrix = None
    if callable(matrix_fn):
        try:
            matrix_values = matrix_fn(
                molecule,
                occupation_tolerance=occupation_tolerance,
            )
        except AttributeError as exc:
            if "does not expose" not in str(exc):
                raise
            matrix_values = None
        if matrix_values is not None:
            matrix = _as_nonlocal_response_matrix(
                matrix_values,
                nocc=nocc,
                nvir=nvir,
            )

    action = None
    if callable(action_fn_raw):

        def action(values: Array) -> Array:
            out = jnp.asarray(
                action_fn_raw(
                    molecule,
                    values,
                    occupation_tolerance=occupation_tolerance,
                ),
                dtype=delta_eps.dtype,
            )
            if out.shape != jnp.asarray(values).shape:
                raise ValueError(
                    "nonlocal_response_action must preserve the transition-amplitude shape "
                    f"{jnp.asarray(values).shape}, got {out.shape}."
                )
            _assert_finite(out, label="Nonlocal response action")
            return out

    diagonal = None
    if callable(diagonal_fn):
        diagonal = _as_nonlocal_response_diagonal(
            diagonal_fn(molecule, occupation_tolerance=occupation_tolerance),
            delta_eps=delta_eps,
        )
    elif matrix is not None:
        diagonal = jnp.diag(matrix).reshape(nocc, nvir)

    return matrix, action, diagonal


def _resolve_nonlocal_response_matrix_pair(
    resolved_xc: Any,
    molecule: Any,
    *,
    delta_eps: Array,
    occupation_tolerance: float,
) -> tuple[Array | None, Array | None]:
    if resolved_xc is None:
        return None, None

    if getattr(resolved_xc, "nonlocal_response_matrices_fn", None) is None:
        return None, None

    matrix_pair_fn = getattr(resolved_xc, "nonlocal_response_matrices", None)
    if not callable(matrix_pair_fn):
        return None, None

    try:
        values = matrix_pair_fn(
            molecule,
            occupation_tolerance=occupation_tolerance,
        )
    except AttributeError as exc:
        if "does not expose" not in str(exc):
            raise
        return None, None
    if values is None:
        return None, None

    matrix_a, matrix_b = values
    nocc, nvir = delta_eps.shape
    return (
        _as_nonlocal_response_matrix(
            matrix_a,
            nocc=nocc,
            nvir=nvir,
            label="Nonlocal A response matrix",
        ),
        _as_nonlocal_response_matrix(
            matrix_b,
            nocc=nocc,
            nvir=nvir,
            label="Nonlocal B response matrix",
        ),
    )


def build_restricted_response_matrices(
    molecule: Any,
    xc_functional: Any | None = None,
    *,
    xc_params: Any | None = None,
    occupation_tolerance: float = 1e-8,
) -> TDDFTMatrices:
    """Build restricted closed-shell TDDFT response matrices in pure JAX."""

    resolved_xc = _resolve_xc_functional(molecule, xc_functional, xc_params)

    weights = jnp.asarray(molecule.grid.weights)
    orbo, orbv, delta_eps, _ = _restricted_orbital_data(
        molecule,
        occupation_tolerance,
    )

    nocc, nvir = delta_eps.shape
    eye_occ = jnp.eye(nocc, dtype=delta_eps.dtype)
    eye_vir = jnp.eye(nvir, dtype=delta_eps.dtype)
    diagonal = jnp.einsum(
        "ia,ij,ab->iajb",
        delta_eps,
        eye_occ,
        eye_vir,
        precision=Precision.HIGHEST,
    )

    hybrid_fraction = jnp.asarray(0.0, dtype=delta_eps.dtype)
    total_density = None
    if resolved_xc is not None:
        total_density = _density_on_grid(molecule)
        hybrid_fraction = _strict_hybrid_fraction(
            resolved_xc,
            molecule,
            total_density,
        )
    needs_exchange = _needs_exchange_terms(hybrid_fraction)

    eri_ovov, eri_ovvo, eri_oovv = _restricted_eri_slices(
        molecule,
        getattr(molecule, "rep_tensor", None),
        orbo,
        orbv,
        need_ovvo=True,
        include_oovv=bool(needs_exchange),
    )

    # Singlet restricted Casida convention:
    # A_ia,jb Coulomb term is 2(ia|jb), B_ia,jb Coulomb term is 2(ia|bj).
    hartree_a = 2.0 * eri_ovov
    hartree_b = 2.0 * jnp.transpose(eri_ovvo, (0, 1, 3, 2))

    xc_contribution = jnp.zeros_like(hartree_a)
    exchange_a = jnp.zeros_like(hartree_a)
    exchange_b = jnp.zeros_like(hartree_b)
    if resolved_xc is not None:
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
            response_features = restricted_transition_response_features(
                molecule,
                feature_kind=str(feature_kind),
                occupation_tolerance=occupation_tolerance,
            )
            if strict_tensor.shape[0] != response_features.shape[0]:
                raise ValueError(
                    "Strict response tensor feature dimension must match the "
                    "transition-feature dimension "
                    f"(got {strict_tensor.shape[0]} vs {response_features.shape[0]})."
                )
            xc_contribution = 2.0 * jnp.einsum(
                "xyr,xria,yrjb->iajb",
                strict_tensor * weights[None, None, :],
                response_features,
                response_features,
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
            rho_ov_density = restricted_transition_response_features(
                molecule,
                feature_kind="LDA",
                occupation_tolerance=occupation_tolerance,
            )[0]
            xc_contribution = 2.0 * jnp.einsum(
                "ria,rjb,r->iajb",
                rho_ov_density,
                rho_ov_density,
                weights * local_fxc,
            )
        if needs_exchange:
            exchange_a_raw = -jnp.transpose(eri_oovv, (0, 2, 1, 3))
            # B_ia,jb exchange term is -(ib|aj).
            exchange_b_raw = -jnp.transpose(eri_ovvo, (0, 2, 3, 1))
            exchange_a = hybrid_fraction * exchange_a_raw
            exchange_b = hybrid_fraction * exchange_b_raw

        nonlocal_a_matrix, nonlocal_b_matrix = _resolve_nonlocal_response_matrix_pair(
            resolved_xc,
            molecule,
            delta_eps=delta_eps,
            occupation_tolerance=occupation_tolerance,
        )
        nonlocal_matrix, nonlocal_action, _ = _resolve_nonlocal_response_terms(
            resolved_xc,
            molecule,
            delta_eps=delta_eps,
            occupation_tolerance=occupation_tolerance,
        )
        if nonlocal_matrix is None and nonlocal_action is not None:
            nonlocal_matrix = _materialize_nonlocal_response_matrix_from_action(
                nonlocal_action,
                nocc=nocc,
                nvir=nvir,
                dtype=delta_eps.dtype,
            )
        if nonlocal_matrix is not None:
            xc_contribution = xc_contribution + nonlocal_matrix.reshape(nocc, nvir, nocc, nvir)
        if nonlocal_a_matrix is not None:
            exchange_a = exchange_a + nonlocal_a_matrix.reshape(nocc, nvir, nocc, nvir)
        if nonlocal_b_matrix is not None:
            exchange_b = exchange_b + nonlocal_b_matrix.reshape(nocc, nvir, nocc, nvir)

    a_matrix = diagonal + hartree_a + exchange_a + xc_contribution
    b_matrix = hartree_b + exchange_b + xc_contribution
    return TDDFTMatrices(
        orbital_energy_differences=delta_eps,
        a_matrix=a_matrix,
        b_matrix=b_matrix,
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


def build_restricted_tda_matrix(
    molecule: Any,
    xc_functional: Any | None = None,
    *,
    xc_params: Any | None = None,
    occupation_tolerance: float = 1e-8,
) -> tuple[Array, Array]:
    """Build only the restricted singlet TDA A matrix."""

    resolved_xc = _resolve_xc_functional(molecule, xc_functional, xc_params)

    weights = jnp.asarray(molecule.grid.weights)
    orbo, orbv, delta_eps, _ = _restricted_orbital_data(
        molecule,
        occupation_tolerance,
    )

    nocc, nvir = delta_eps.shape
    dim = int(nocc * nvir)
    flat_a = jnp.diag(delta_eps.reshape(dim))

    hybrid_fraction = jnp.asarray(0.0, dtype=delta_eps.dtype)
    total_density = None
    if resolved_xc is not None:
        total_density = _density_on_grid(molecule)
        hybrid_fraction = _strict_hybrid_fraction(
            resolved_xc,
            molecule,
            total_density,
        )
    needs_exchange = _needs_exchange_terms(hybrid_fraction)

    rep_tensor_obj = getattr(molecule, "rep_tensor", None)
    rep_tensor = None
    if rep_tensor_obj is not None and int(jnp.asarray(rep_tensor_obj).size) > 0:
        rep_tensor = jnp.asarray(rep_tensor_obj)
    eri_ovov = getattr(molecule, "eri_ovov", None)
    eri_oovv = getattr(molecule, "eri_oovv", None)
    df_factors = getattr(molecule, "df_factors", None)
    used_df_direct = (
        df_factors is not None
        and int(jnp.asarray(df_factors).size) > 0
        and eri_ovov is None
        and (not needs_exchange or eri_oovv is None)
    )
    if used_df_direct:
        df_factors = jnp.asarray(df_factors)
        flat_a = flat_a + _tda_df_hartree_flat(df_factors, orbo, orbv)
        if needs_exchange:
            flat_a = flat_a - hybrid_fraction * _tda_df_exchange_flat(df_factors, orbo, orbv)
    else:
        eri_ovov, _, eri_oovv = _restricted_eri_slices(
            molecule,
            rep_tensor,
            orbo,
            orbv,
            need_ovvo=False,
            include_oovv=needs_exchange,
        )
        flat_a = flat_a + 2.0 * eri_ovov.reshape(dim, dim)
        if needs_exchange:
            flat_a = flat_a - hybrid_fraction * jnp.transpose(eri_oovv, (0, 2, 1, 3)).reshape(
                dim, dim
            )

    if resolved_xc is not None:
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
            response_features = restricted_transition_response_features(
                molecule,
                feature_kind=str(feature_kind),
                occupation_tolerance=occupation_tolerance,
            )
            if strict_tensor.shape[0] != response_features.shape[0]:
                raise ValueError(
                    "Strict response tensor feature dimension must match the "
                    "transition-feature dimension "
                    f"(got {strict_tensor.shape[0]} vs {response_features.shape[0]})."
                )
            flat_a = flat_a + _tda_strict_xc_flat(
                strict_tensor * weights[None, None, :],
                response_features.reshape(response_features.shape[0], weights.shape[0], dim),
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
            rho_ov_density = restricted_transition_response_features(
                molecule,
                feature_kind="LDA",
                occupation_tolerance=occupation_tolerance,
            )[0]
            flat_a = flat_a + _tda_lda_xc_flat(
                weights * local_fxc,
                rho_ov_density.reshape(weights.shape[0], dim),
            )
        nonlocal_a_matrix, _ = _resolve_nonlocal_response_matrix_pair(
            resolved_xc,
            molecule,
            delta_eps=delta_eps,
            occupation_tolerance=occupation_tolerance,
        )
        nonlocal_matrix, nonlocal_action, _ = _resolve_nonlocal_response_terms(
            resolved_xc,
            molecule,
            delta_eps=delta_eps,
            occupation_tolerance=occupation_tolerance,
        )
        if nonlocal_matrix is None and nonlocal_action is not None:
            nonlocal_matrix = _materialize_nonlocal_response_matrix_from_action(
                nonlocal_action,
                nocc=nocc,
                nvir=nvir,
                dtype=delta_eps.dtype,
            )
        if nonlocal_matrix is not None:
            flat_a = flat_a + nonlocal_matrix
        if nonlocal_a_matrix is not None:
            flat_a = flat_a + nonlocal_a_matrix
    a_matrix = flat_a.reshape(nocc, nvir, nocc, nvir)
    return delta_eps, a_matrix


def _build_restricted_response_operator_data(
    molecule: Any,
    xc_functional: Any | None = None,
    *,
    xc_params: Any | None = None,
    occupation_tolerance: float = 1e-8,
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
    nocc, nvir = delta_eps.shape

    weighted_local_kernel = None
    rho_ov_density = None
    weighted_strict_tensor = None
    response_features = None
    hybrid_fraction = 0.0
    nonlocal_xc_action_fn = None
    nonlocal_xc_diagonal = None
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
            response_features = restricted_transition_response_features(
                molecule,
                feature_kind=str(feature_kind),
                occupation_tolerance=occupation_tolerance,
            )
            if strict_tensor.shape[0] != response_features.shape[0]:
                raise ValueError(
                    "Strict response tensor feature dimension must match the "
                    "transition-feature dimension "
                    f"(got {strict_tensor.shape[0]} vs {response_features.shape[0]})."
                )
            weighted_strict_tensor = strict_tensor * weights[None, None, :]
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
            rho_ov_density = restricted_transition_response_features(
                molecule,
                feature_kind="LDA",
                occupation_tolerance=occupation_tolerance,
            )[0]
            local_fxc = _as_grid_values(local_fxc, total_density, label="XC kernel")
            weighted_local_kernel = weights * local_fxc

        nonlocal_a_matrix, nonlocal_b_matrix = _resolve_nonlocal_response_matrix_pair(
            resolved_xc,
            molecule,
            delta_eps=delta_eps,
            occupation_tolerance=occupation_tolerance,
        )
        nonlocal_matrix, nonlocal_action, nonlocal_diagonal = _resolve_nonlocal_response_terms(
            resolved_xc,
            molecule,
            delta_eps=delta_eps,
            occupation_tolerance=occupation_tolerance,
        )
        dim = int(nocc * nvir)
        if nonlocal_a_matrix is not None:
            flat_a = nonlocal_a_matrix

            def nonlocal_xc_a_action_fn(values: Array) -> Array:
                reshaped = jnp.asarray(values, dtype=delta_eps.dtype).reshape(-1, dim)
                out = reshaped @ flat_a.T
                return out.reshape(-1, nocc, nvir)

            nonlocal_xc_a_diagonal = jnp.diag(flat_a).reshape(nocc, nvir)
        if nonlocal_b_matrix is not None:
            flat_b = nonlocal_b_matrix

            def nonlocal_xc_b_action_fn(values: Array) -> Array:
                reshaped = jnp.asarray(values, dtype=delta_eps.dtype).reshape(-1, dim)
                out = reshaped @ flat_b.T
                return out.reshape(-1, nocc, nvir)

        if nonlocal_matrix is not None:
            flat = nonlocal_matrix

            def nonlocal_xc_action_fn(values: Array) -> Array:
                reshaped = jnp.asarray(values, dtype=delta_eps.dtype).reshape(-1, dim)
                out = reshaped @ flat.T
                return out.reshape(-1, nocc, nvir)

            nonlocal_xc_diagonal = jnp.diag(flat).reshape(nocc, nvir)
        elif nonlocal_action is not None:
            nonlocal_xc_action_fn = nonlocal_action
            if nonlocal_diagonal is None:
                nonlocal_xc_diagonal = jnp.diag(
                    _materialize_nonlocal_response_matrix_from_action(
                        nonlocal_action,
                        nocc=nocc,
                        nvir=nvir,
                        dtype=delta_eps.dtype,
                    )
                ).reshape(nocc, nvir)
            else:
                nonlocal_xc_diagonal = nonlocal_diagonal

    eri_ovov, eri_ovvo, eri_oovv = _restricted_eri_slices(
        molecule,
        rep_tensor,
        orbo,
        orbv,
        need_ovvo=True,
        include_oovv=_needs_exchange_terms(hybrid_fraction),
    )
    alpha = jnp.asarray(hybrid_fraction, dtype=eri_ovov.dtype)
    effective_tda_eri = 2.0 * eri_ovov
    if eri_oovv is not None:
        effective_tda_eri = effective_tda_eri - alpha * jnp.transpose(
            eri_oovv,
            (0, 2, 1, 3),
        )
    effective_b_eri = None
    if eri_ovvo is not None:
        effective_b_eri = 2.0 * eri_ovvo - alpha * jnp.transpose(
            eri_ovvo,
            (0, 2, 1, 3),
        )

    return _RestrictedResponseOperatorData(
        delta_eps=delta_eps,
        eri_ovov=eri_ovov,
        eri_ovvo=eri_ovvo,
        eri_oovv=eri_oovv,
        effective_tda_eri=effective_tda_eri,
        effective_b_eri=effective_b_eri,
        weighted_local_kernel=weighted_local_kernel,
        rho_ov_density=rho_ov_density,
        weighted_strict_tensor=weighted_strict_tensor,
        response_features=response_features,
        hybrid_fraction=hybrid_fraction,
        nonlocal_xc_action_fn=nonlocal_xc_action_fn,
        nonlocal_xc_diagonal=nonlocal_xc_diagonal,
        nonlocal_xc_a_action_fn=nonlocal_xc_a_action_fn,
        nonlocal_xc_b_action_fn=nonlocal_xc_b_action_fn,
        nonlocal_xc_a_diagonal=nonlocal_xc_a_diagonal,
    )


def _restricted_xc_action(data: _RestrictedResponseOperatorData, x: Array) -> Array:
    out = jnp.zeros_like(x)
    if data.weighted_strict_tensor is not None and data.response_features is not None:
        projected = jnp.einsum(
            "xria,nia->nxr",
            data.response_features,
            x,
            precision=Precision.HIGHEST,
        )
        weighted = jnp.einsum(
            "xyr,nyr->nxr",
            data.weighted_strict_tensor,
            projected,
        )
        out = out + 2.0 * jnp.einsum(
            "xria,nxr->nia",
            data.response_features,
            weighted,
        )
    if data.weighted_local_kernel is not None and data.rho_ov_density is not None:
        projected = jnp.einsum(
            "ria,nia->nr",
            data.rho_ov_density,
            x,
            precision=Precision.HIGHEST,
        )
        out = out + 2.0 * jnp.einsum(
            "ria,nr->nia",
            data.rho_ov_density,
            projected * data.weighted_local_kernel[None, :],
        )
    if data.nonlocal_xc_action_fn is not None:
        out = out + data.nonlocal_xc_action_fn(x)
    return out


def _restricted_xc_diagonal(data: _RestrictedResponseOperatorData) -> Array:
    out = jnp.zeros_like(data.delta_eps)
    if data.weighted_strict_tensor is not None and data.response_features is not None:
        out = out + 2.0 * jnp.einsum(
            "xyr,xria,yria->ia",
            data.weighted_strict_tensor,
            data.response_features,
            data.response_features,
        )
    if data.weighted_local_kernel is not None and data.rho_ov_density is not None:
        out = out + 2.0 * jnp.einsum(
            "r,ria,ria->ia",
            data.weighted_local_kernel,
            data.rho_ov_density,
            data.rho_ov_density,
        )
    if data.nonlocal_xc_diagonal is not None:
        out = out + jnp.asarray(data.nonlocal_xc_diagonal, dtype=out.dtype)
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
    return diagonal + _restricted_xc_diagonal(data)


def build_restricted_tda_operator(
    molecule: Any,
    xc_functional: Any | None = None,
    *,
    xc_params: Any | None = None,
    occupation_tolerance: float = 1e-8,
    materialize_matrix: bool = False,
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

    diagonal = _restricted_tda_diagonal(data).reshape(-1)

    def vind(x: Array) -> Array:
        x = jnp.asarray(x).reshape(-1, nocc, nvir)
        return _restricted_a_action(data, x).reshape(-1, dim)

    flat_a = None
    if materialize_matrix:
        _, a_matrix = build_restricted_tda_matrix(
            molecule,
            xc_functional,
            xc_params=xc_params,
            occupation_tolerance=occupation_tolerance,
        )
        flat_a = a_matrix.reshape(dim, dim)
    return vind, diagonal, delta_eps, flat_a


def build_restricted_tdhf_operator(
    molecule: Any,
    xc_functional: Any | None = None,
    *,
    xc_params: Any | None = None,
    occupation_tolerance: float = 1e-8,
    materialize_matrix: bool = False,
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

    flat_a = None
    flat_b = None
    if materialize_matrix:
        matrices = build_restricted_response_matrices(
            molecule,
            xc_functional,
            xc_params=xc_params,
            occupation_tolerance=occupation_tolerance,
        )
        flat_a = matrices.a_matrix.reshape(dim, dim)
        flat_b = matrices.b_matrix.reshape(dim, dim)
    return vind, flat_a, flat_b


def build_restricted_a_minus_b_matrix(
    molecule: Any,
    xc_functional: Any | None = None,
    *,
    xc_params: Any | None = None,
    occupation_tolerance: float = 1e-8,
) -> tuple[Array, Array]:
    data = _build_restricted_response_operator_data(
        molecule,
        xc_functional,
        xc_params=xc_params,
        occupation_tolerance=occupation_tolerance,
    )
    delta_eps = data.delta_eps
    nocc, nvir = delta_eps.shape
    dim = int(nocc * nvir)

    eye_occ = jnp.eye(nocc, dtype=delta_eps.dtype)
    eye_vir = jnp.eye(nvir, dtype=delta_eps.dtype)
    diagonal = jnp.einsum(
        "ia,ij,ab->iajb",
        delta_eps,
        eye_occ,
        eye_vir,
        precision=Precision.HIGHEST,
    )
    hartree_diff = 2.0 * data.eri_ovov - 2.0 * jnp.transpose(data.eri_ovvo, (0, 1, 3, 2))
    alpha = jnp.asarray(data.hybrid_fraction, dtype=hartree_diff.dtype)
    exchange_diff = jnp.zeros_like(hartree_diff)
    if data.eri_oovv is not None:
        exchange_a_raw = -jnp.transpose(data.eri_oovv, (0, 2, 1, 3))
        exchange_b_raw = -jnp.transpose(data.eri_ovvo, (0, 2, 3, 1))
        exchange_diff = alpha * (exchange_a_raw - exchange_b_raw)
    return (diagonal + hartree_diff + exchange_diff).reshape(dim, dim), delta_eps


def gen_tda_vind(
    molecule: Any,
    xc_functional: Any | None = None,
    *,
    xc_params: Any | None = None,
    occupation_tolerance: float = 1e-8,
    materialize_matrix: bool = True,
):
    vind, _, _, flat_a = build_restricted_tda_operator(
        molecule,
        xc_functional,
        xc_params=xc_params,
        occupation_tolerance=occupation_tolerance,
        materialize_matrix=materialize_matrix,
    )
    return vind, flat_a


def gen_tdhf_vind(
    molecule: Any,
    xc_functional: Any | None = None,
    *,
    xc_params: Any | None = None,
    occupation_tolerance: float = 1e-8,
    materialize_matrix: bool = True,
):
    vind, flat_a, flat_b = build_restricted_tdhf_operator(
        molecule,
        xc_functional,
        xc_params=xc_params,
        occupation_tolerance=occupation_tolerance,
        materialize_matrix=materialize_matrix,
    )
    return vind, flat_a, flat_b
