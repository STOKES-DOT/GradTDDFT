from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from typing import Any
from typing import Literal

import jax
import jax.numpy as jnp
from jax import core as jax_core
from jaxtyping import Array

from .eigensolvers import davidson_lowest_symmetric, _matmul, _solver_dtype
from ._utils import (
    _casida_metric_factor,
    _matrix_power_symmetric,
    _resolve_xc_functional,
    _restricted_channel,
    _symmetrize,
)
from .response import (
    build_restricted_a_minus_b_matrix,
    build_restricted_tda_matrix,
    build_restricted_tda_operator,
    build_restricted_response_matrices,
    gen_tda_vind,
    gen_tdhf_vind,
)
from .tda import (
    _prefer_dense_auto_eigensolve,
    _prefer_dense_eigensolve,
    solve_tda,
    solve_tda_from_a_matrix,
    solve_tda_from_operator,
)
from .types import TDDFTMatrices, TDDFTResult, TDAResult


def _is_traced_convergence_flag(value) -> bool:
    return _is_tracer(value)


def _is_tracer(value) -> bool:
    value_type = type(value)
    return isinstance(value, jax_core.Tracer) or (
        "Tracer" in value_type.__name__ and value_type.__module__.startswith("jax")
    )


def _lowest_dense_eigenpairs(matrix, *, nroots: int):
    eigvals, eigvecs = jnp.linalg.eigh(matrix)
    return eigvals[:nroots], eigvecs[:, :nroots]


def _restricted_td_space_dimensions(
    molecule: Any,
    occupation_tolerance: float,
) -> tuple[int, int, int]:
    mo_coeff, mo_occ, _ = _restricted_channel(molecule)
    nocc = getattr(molecule, "nocc", None)
    if nocc is None:
        nocc = int(jnp.count_nonzero(mo_occ > occupation_tolerance))
    else:
        nocc = int(nocc)
    nvir = int(mo_coeff.shape[1] - nocc)
    if nocc <= 0 or nvir <= 0:
        raise ValueError("Need at least one occupied and one virtual orbital.")
    return nocc, nvir, nocc * nvir


def _finalize_casida_result(
    w2,
    vecs,
    *,
    nroots: int,
    excitation_threshold: float,
    matrix_eps: float,
    nocc: int,
    nvir: int,
    metric_factor,
    a_plus_b_vind_rows: Callable,
    a_matrix,
    b_matrix,
    casida_matrix,
) -> TDDFTResult:
    valid = w2 > excitation_threshold**2
    order = jnp.argsort(jnp.where(valid, w2, jnp.inf))
    keep = order[:nroots]
    keep_mask = valid[keep]

    w = jnp.sqrt(jnp.maximum(w2[keep], 0.0))
    w = jnp.where(keep_mask, w, 0.0)
    f_vectors = vecs[:, keep]
    f_vectors = f_vectors * keep_mask[jnp.newaxis, :]
    x_plus_y = _matmul(metric_factor, f_vectors)
    safe_w = jnp.where(keep_mask, w, 1.0)
    x_minus_y = a_plus_b_vind_rows(x_plus_y.T).T / safe_w[jnp.newaxis, :]

    x = 0.5 * (x_plus_y + x_minus_y)
    y = 0.5 * (x_plus_y - x_minus_y)
    x = x * keep_mask[jnp.newaxis, :]
    y = y * keep_mask[jnp.newaxis, :]
    norm = jnp.sum(jnp.abs(x) ** 2, axis=0) - jnp.sum(jnp.abs(y) ** 2, axis=0)
    scale = jnp.sqrt(0.5) / jnp.sqrt(jnp.maximum(jnp.abs(norm), matrix_eps))
    x = x * scale[jnp.newaxis, :]
    y = y * scale[jnp.newaxis, :]

    return TDDFTResult(
        excitation_energies=w,
        x_amplitudes=x.T.reshape(-1, nocc, nvir),
        y_amplitudes=y.T.reshape(-1, nocc, nvir),
        a_matrix=a_matrix,
        b_matrix=b_matrix,
        casida_matrix=casida_matrix,
    )


def solve_casida_from_operator(
    delta_eps,
    casida_vind_rows: Callable,
    diagonal,
    *,
    metric_factor,
    a_plus_b_vind_rows: Callable,
    nstates: int | None = None,
    excitation_threshold: float = 1e-7,
    matrix_eps: float = 1e-10,
    davidson_tol: float = 1e-6,
    davidson_max_iter: int = 60,
    davidson_max_subspace: int | None = None,
    a_matrix=None,
    b_matrix=None,
    casida_matrix=None,
) -> TDDFTResult:
    nocc, nvir = delta_eps.shape
    dim = int(nocc * nvir)
    nroots = dim if nstates is None else min(int(nstates), dim)
    w2, vecs, converged = davidson_lowest_symmetric(
        lambda vectors: casida_vind_rows(jnp.asarray(vectors).T).T,
        nroots=nroots,
        size=dim,
        diag=jnp.asarray(diagonal).reshape(dim),
        tol=davidson_tol,
        max_iter=davidson_max_iter,
        max_subspace=davidson_max_subspace,
    )
    if not _is_traced_convergence_flag(converged) and not bool(converged):
        raise RuntimeError("Davidson Casida solver did not converge.")
    return _finalize_casida_result(
        w2,
        vecs,
        nroots=nroots,
        excitation_threshold=excitation_threshold,
        matrix_eps=matrix_eps,
        nocc=nocc,
        nvir=nvir,
        metric_factor=metric_factor,
        a_plus_b_vind_rows=a_plus_b_vind_rows,
        a_matrix=a_matrix,
        b_matrix=b_matrix,
        casida_matrix=casida_matrix,
    )


def solve_casida(
    matrices: TDDFTMatrices,
    *,
    nstates: int | None = None,
    excitation_threshold: float = 1e-7,
    matrix_eps: float = 1e-10,
    eigensolver: str = "auto",
    davidson_tol: float = 1e-6,
    davidson_max_iter: int = 60,
    davidson_max_subspace: int | None = None,
) -> TDDFTResult:
    """Solve the restricted Casida TDDFT equation."""

    delta_eps = matrices.orbital_energy_differences
    nocc, nvir = delta_eps.shape
    flat_a = _symmetrize(matrices.a_matrix.reshape(nocc * nvir, nocc * nvir))
    flat_b = _symmetrize(matrices.b_matrix.reshape(nocc * nvir, nocc * nvir))

    a_plus_b = _symmetrize(flat_a + flat_b)
    a_minus_b = _symmetrize(flat_a - flat_b)
    work_dtype = _solver_dtype(jnp.result_type(a_plus_b.dtype, a_minus_b.dtype))
    a_plus_b = a_plus_b.astype(work_dtype)
    a_minus_b = a_minus_b.astype(work_dtype)
    dim = int(a_plus_b.shape[0])
    nroots = dim if nstates is None else min(int(nstates), dim)

    mode = str(eigensolver).lower()
    use_davidson = False
    if mode == "davidson":
        use_davidson = not _prefer_dense_eigensolve(dim, nroots)
    elif mode == "dense":
        use_davidson = False
    elif mode == "auto":
        use_davidson = (
            nstates is not None
            and not _prefer_dense_auto_eigensolve(dim, nroots)
            and dim >= 96
            and nroots <= min(24, max(1, dim // 3))
        )
    else:
        raise ValueError(
            f"Unsupported eigensolver={eigensolver!r}. Choose one of {{'auto', 'dense', 'davidson'}}."
        )

    casida_matrix = None
    if use_davidson:
        metric_factor = _casida_metric_factor(a_minus_b, matrix_eps)

        def casida_matvec(vectors):
            transformed = _matmul(metric_factor, vectors)
            coupled = _matmul(a_plus_b, transformed)
            return _matmul(metric_factor.T.conj(), coupled)

        projected = _matmul(a_plus_b, metric_factor)
        casida_diag = jnp.einsum("ki,ki->i", metric_factor, projected)
        davidson_w2, davidson_vecs, converged = davidson_lowest_symmetric(
            casida_matvec,
            nroots=nroots,
            size=dim,
            diag=casida_diag,
            tol=davidson_tol,
            max_iter=davidson_max_iter,
            max_subspace=davidson_max_subspace,
        )
        if mode == "auto":
            dense_casida_matrix = _symmetrize(
                _matmul(_matmul(metric_factor.T.conj(), a_plus_b), metric_factor)
            )
            if _is_traced_convergence_flag(converged):
                w2, vecs = jax.lax.cond(
                    converged,
                    lambda _: (davidson_w2, davidson_vecs),
                    lambda _: _lowest_dense_eigenpairs(dense_casida_matrix, nroots=nroots),
                    operand=None,
                )
                casida_matrix = dense_casida_matrix
            elif bool(converged):
                w2, vecs, casida_matrix = davidson_w2, davidson_vecs, None
            else:
                w2, vecs = _lowest_dense_eigenpairs(dense_casida_matrix, nroots=nroots)
                casida_matrix = dense_casida_matrix
        else:
            w2, vecs = davidson_w2, davidson_vecs
            if not _is_traced_convergence_flag(converged) and not bool(converged):
                raise RuntimeError("Davidson Casida solver did not converge.")
    else:
        metric_factor = _matrix_power_symmetric(a_minus_b, 0.5, matrix_eps)
        casida_matrix = _symmetrize(
            _matmul(_matmul(metric_factor.T.conj(), a_plus_b), metric_factor)
        )
        w2, vecs = jnp.linalg.eigh(casida_matrix)

    return _finalize_casida_result(
        w2,
        vecs,
        nroots=nroots,
        excitation_threshold=excitation_threshold,
        matrix_eps=matrix_eps,
        nocc=nocc,
        nvir=nvir,
        metric_factor=metric_factor,
        a_plus_b_vind_rows=lambda rows: _matmul(rows, a_plus_b.T),
        a_matrix=matrices.a_matrix,
        b_matrix=matrices.b_matrix,
        casida_matrix=casida_matrix,
    )


@dataclass(frozen=True)
class RestrictedCasidaTDDFT:
    """PySCF-like restricted TDDFT driver for GradDFT-style molecules."""

    molecule: Any
    xc_functional: Any | None = None
    xc_params: Any | None = None
    occupation_tolerance: float = 1e-8
    excitation_threshold: float = 1e-7
    matrix_eps: float = 1e-10
    eigensolver: Literal["auto", "dense", "davidson"] = "auto"
    davidson_tol: float = 1e-6
    davidson_max_iter: int = 60
    davidson_max_subspace: int | None = None
    _cached_matrices: TDDFTMatrices | None = field(default=None, init=False, repr=False, compare=False)
    _cached_tda_matrix: tuple[Any, Any] | None = field(default=None, init=False, repr=False, compare=False)

    def _posthoc_correction(
        self,
        result: TDAResult | TDDFTResult,
        *,
        use_tda: bool,
    ) -> Array | None:
        resolved_xc = _resolve_xc_functional(
            self.molecule,
            self.xc_functional,
            self.xc_params,
        )
        if resolved_xc is None:
            return None
        method_name = "post_tda_correction" if use_tda else "post_tddft_correction"
        correction_fn = getattr(resolved_xc, method_name, None)
        if not callable(correction_fn):
            return None
        try:
            correction = correction_fn(
                self.molecule,
                result,
                occupation_tolerance=self.occupation_tolerance,
            )
        except AttributeError as exc:
            if "does not expose" not in str(exc):
                raise
            return None
        correction = jnp.asarray(correction, dtype=result.excitation_energies.dtype)
        if correction.ndim == 0:
            correction = jnp.full_like(result.excitation_energies, correction)
        elif correction.shape != result.excitation_energies.shape:
            raise ValueError(
                f"{method_name} must return a scalar or shape "
                f"{result.excitation_energies.shape}, got {correction.shape}."
            )
        return correction

    def _apply_posthoc_correction(
        self,
        result: TDAResult | TDDFTResult,
        *,
        use_tda: bool,
    ) -> TDAResult | TDDFTResult:
        correction = self._posthoc_correction(result, use_tda=use_tda)
        if correction is None:
            return result
        return replace(
            result,
            excitation_energies=result.excitation_energies + correction,
            posthoc_correction=correction,
        )

    def build_matrices(self) -> TDDFTMatrices:
        return build_restricted_response_matrices(
            self.molecule,
            self.xc_functional,
            xc_params=self.xc_params,
            occupation_tolerance=self.occupation_tolerance,
        )

    def _build_tda_matrix(self) -> tuple[Any, Any]:
        return build_restricted_tda_matrix(
            self.molecule,
            self.xc_functional,
            xc_params=self.xc_params,
            occupation_tolerance=self.occupation_tolerance,
        )

    def tda(self, nstates: int | None = None) -> TDAResult:
        mode = str(self.eigensolver).lower()
        _, _, dim = _restricted_td_space_dimensions(
            self.molecule,
            self.occupation_tolerance,
        )
        nroots = dim if nstates is None else min(int(nstates), dim)
        use_davidson = False
        if mode == "davidson":
            use_davidson = not _prefer_dense_eigensolve(dim, nroots)
        elif mode == "auto":
            use_davidson = (
                nstates is not None
                and not _prefer_dense_auto_eigensolve(dim, nroots)
                and dim >= 96
                and nroots <= min(24, max(1, dim // 3))
            )
        elif mode != "dense":
            raise ValueError(
                f"Unsupported eigensolver={self.eigensolver!r}. Choose one of {{'auto', 'dense', 'davidson'}}."
            )

        if not use_davidson:
            delta_eps, a_matrix = self._build_tda_matrix()
            result = solve_tda_from_a_matrix(
                delta_eps,
                a_matrix,
                nstates=nstates,
                excitation_threshold=self.excitation_threshold,
            )
            return self._apply_posthoc_correction(result, use_tda=True)

        vind, diagonal, delta_eps, _ = build_restricted_tda_operator(
            self.molecule,
            self.xc_functional,
            xc_params=self.xc_params,
            occupation_tolerance=self.occupation_tolerance,
            materialize_matrix=False,
        )
        if use_davidson:
            try:
                result = solve_tda_from_operator(
                    delta_eps,
                    vind,
                    diagonal,
                    nstates=nstates,
                    excitation_threshold=self.excitation_threshold,
                    davidson_tol=self.davidson_tol,
                    davidson_max_iter=self.davidson_max_iter,
                    davidson_max_subspace=self.davidson_max_subspace,
                    a_matrix=None,
                )
                return self._apply_posthoc_correction(result, use_tda=True)
            except RuntimeError:
                delta_eps, a_matrix = self._build_tda_matrix()
                result = solve_tda_from_a_matrix(
                    delta_eps,
                    a_matrix,
                    nstates=nstates,
                    excitation_threshold=self.excitation_threshold,
                )
                return self._apply_posthoc_correction(result, use_tda=True)
        raise AssertionError("Unreachable TDA solver branch.")

    def gen_tda_vind(self, *, materialize_matrix: bool = True):
        return gen_tda_vind(
            self.molecule,
            self.xc_functional,
            xc_params=self.xc_params,
            occupation_tolerance=self.occupation_tolerance,
            materialize_matrix=materialize_matrix,
        )

    def gen_tdhf_vind(self, *, materialize_matrix: bool = True):
        return gen_tdhf_vind(
            self.molecule,
            self.xc_functional,
            xc_params=self.xc_params,
            occupation_tolerance=self.occupation_tolerance,
            materialize_matrix=materialize_matrix,
        )

    def kernel(self, nstates: int | None = None) -> TDDFTResult:
        mode = str(self.eigensolver).lower()
        _, _, dim = _restricted_td_space_dimensions(
            self.molecule,
            self.occupation_tolerance,
        )
        nroots = dim if nstates is None else min(int(nstates), dim)
        use_davidson = False
        if mode == "davidson":
            use_davidson = not _prefer_dense_eigensolve(dim, nroots)
        elif mode == "auto":
            use_davidson = (
                nstates is not None
                and not _prefer_dense_auto_eigensolve(dim, nroots)
                and dim >= 96
                and nroots <= min(24, max(1, dim // 3))
            )
        elif mode != "dense":
            raise ValueError(
                f"Unsupported eigensolver={self.eigensolver!r}. Choose one of {{'auto', 'dense', 'davidson'}}."
            )

        if not use_davidson:
            result = solve_casida(
                self.build_matrices(),
                nstates=nstates,
                excitation_threshold=self.excitation_threshold,
                matrix_eps=self.matrix_eps,
                eigensolver=self.eigensolver,
                davidson_tol=self.davidson_tol,
                davidson_max_iter=self.davidson_max_iter,
                davidson_max_subspace=self.davidson_max_subspace,
            )
            return self._apply_posthoc_correction(result, use_tda=False)

        a_minus_b, delta_eps = build_restricted_a_minus_b_matrix(
            self.molecule,
            self.xc_functional,
            xc_params=self.xc_params,
            occupation_tolerance=self.occupation_tolerance,
        )
        if use_davidson:
            vind_tdhf, _, _ = gen_tdhf_vind(
                self.molecule,
                self.xc_functional,
                xc_params=self.xc_params,
                occupation_tolerance=self.occupation_tolerance,
                materialize_matrix=False,
            )
            metric_factor = _casida_metric_factor(a_minus_b, self.matrix_eps)

            def a_plus_b_vind_rows(rows):
                rows = jnp.asarray(rows).reshape(-1, dim)
                z = jnp.concatenate([rows, rows], axis=-1)
                return vind_tdhf(z)[:, :dim]

            def casida_vind_rows(rows):
                rows = jnp.asarray(rows).reshape(-1, dim)
                transformed = _matmul(rows, metric_factor.T.conj())
                coupled = a_plus_b_vind_rows(transformed)
                return _matmul(coupled, metric_factor)

            projected = a_plus_b_vind_rows(metric_factor.T).T
            diagonal = jnp.einsum("ki,ki->i", metric_factor, projected)
            try:
                result = solve_casida_from_operator(
                    delta_eps,
                    casida_vind_rows,
                    diagonal,
                    metric_factor=metric_factor,
                    a_plus_b_vind_rows=a_plus_b_vind_rows,
                    nstates=nstates,
                    excitation_threshold=self.excitation_threshold,
                    matrix_eps=self.matrix_eps,
                    davidson_tol=self.davidson_tol,
                    davidson_max_iter=self.davidson_max_iter,
                    davidson_max_subspace=self.davidson_max_subspace,
                    a_matrix=None,
                    b_matrix=None,
                    casida_matrix=None,
                )
                return self._apply_posthoc_correction(result, use_tda=False)
            except RuntimeError:
                result = solve_casida(
                    self.build_matrices(),
                    nstates=nstates,
                    excitation_threshold=self.excitation_threshold,
                    matrix_eps=self.matrix_eps,
                    eigensolver="dense",
                    davidson_tol=self.davidson_tol,
                    davidson_max_iter=self.davidson_max_iter,
                    davidson_max_subspace=self.davidson_max_subspace,
                )
                return self._apply_posthoc_correction(result, use_tda=False)
        raise AssertionError("Unreachable Casida solver branch.")
