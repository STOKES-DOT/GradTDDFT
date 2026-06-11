from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import replace
from typing import Any
from typing import Literal

import jax
import jax.numpy as jnp

from .eigensolvers import (
    davidson_lowest_tdhf,
    _solver_dtype,
)
from ._utils import (
    _resolve_xc_functional,
    _restricted_channel,
)
from .response import (
    build_restricted_tda_operator,
    gen_tda_vind,
    gen_tdhf_vind,
)
from .tda import solve_tda_from_operator
from .types import TDDFTResult, TDAResult


def _restricted_delta_eps(molecule: Any, occupation_tolerance: float) -> Array:
    _, mo_occ, mo_energy = _restricted_channel(molecule)
    nocc = getattr(molecule, "nocc", None)
    if nocc is None:
        nocc = int(jnp.count_nonzero(mo_occ > occupation_tolerance))
    else:
        nocc = int(nocc)
    return mo_energy[nocc:] - mo_energy[:nocc, None]


def _finalize_casida_result(
    w,
    x_vecs,
    y_vecs,
    *,
    nroots: int,
    excitation_threshold: float,
    matrix_eps: float,
    nocc: int,
    nvir: int,
) -> TDDFTResult:
    valid = w > excitation_threshold
    order = jnp.argsort(jnp.where(valid, w, jnp.inf))
    keep = order[:nroots]
    keep_mask = valid[keep]

    energies = jnp.where(keep_mask, w[keep], 0.0)
    x = x_vecs[:, keep]
    y = y_vecs[:, keep]
    x = x * keep_mask[jnp.newaxis, :]
    y = y * keep_mask[jnp.newaxis, :]
    norm = jnp.sum(jnp.abs(x) ** 2, axis=0) - jnp.sum(jnp.abs(y) ** 2, axis=0)
    scale = jnp.sqrt(0.5) / jnp.sqrt(jnp.maximum(jnp.abs(norm), matrix_eps))
    x = x * scale[jnp.newaxis, :]
    y = y * scale[jnp.newaxis, :]

    return TDDFTResult(
        excitation_energies=energies,
        x_amplitudes=x.T.reshape(-1, nocc, nvir),
        y_amplitudes=y.T.reshape(-1, nocc, nvir),
    )


def solve_casida_from_tdhf_operator(
    delta_eps,
    tdhf_vind_rows: Callable,
    *,
    nstates: int | None = None,
    excitation_threshold: float = 1e-7,
    matrix_eps: float = 1e-10,
    davidson_tol: float = 1e-6,
    davidson_max_iter: int = 60,
    davidson_max_subspace: int | None = None,
) -> TDDFTResult:
    nocc, nvir = delta_eps.shape
    dim = int(nocc * nvir)
    nroots = dim if nstates is None else min(int(nstates), dim)
    work_dtype = _solver_dtype(jnp.asarray(delta_eps).dtype)

    def tdhf_vind(values):
        values = jnp.asarray(values, dtype=work_dtype).reshape(-1, 2 * dim)
        return tdhf_vind_rows(values)

    w, x_vecs, y_vecs, converged = davidson_lowest_tdhf(
        lambda values: jax.lax.stop_gradient(tdhf_vind(values)),
        nroots=nroots,
        size=dim,
        diag=jnp.asarray(delta_eps, dtype=work_dtype).reshape(dim),
        tol=davidson_tol,
        max_iter=davidson_max_iter,
        max_subspace=davidson_max_subspace,
        matrix_eps=matrix_eps,
    )
    del converged
    x_vecs = jax.lax.stop_gradient(x_vecs)
    y_vecs = jax.lax.stop_gradient(y_vecs)
    applied = tdhf_vind(jnp.concatenate([x_vecs.T, y_vecs.T], axis=-1))
    top = applied[:, :dim].T
    bottom = -applied[:, dim:].T
    numerator = jnp.sum(x_vecs * top, axis=0) + jnp.sum(
        y_vecs * bottom,
        axis=0,
    )
    denominator = jnp.sum(x_vecs * x_vecs, axis=0) - jnp.sum(y_vecs * y_vecs, axis=0)
    denominator = jnp.where(
        jnp.abs(denominator) > jnp.asarray(1e-30, dtype=work_dtype),
        denominator,
        jnp.asarray(1e-30, dtype=work_dtype),
    )
    return _finalize_casida_result(
        numerator / denominator,
        x_vecs,
        y_vecs,
        nroots=nroots,
        excitation_threshold=excitation_threshold,
        matrix_eps=matrix_eps,
        nocc=nocc,
        nvir=nvir,
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
    eigensolver: Literal["auto", "davidson"] = "auto"
    davidson_tol: float = 1e-6
    davidson_max_iter: int = 60
    davidson_max_subspace: int | None = None

    def _posthoc_correction(
        self,
        result: TDAResult | TDDFTResult,
        *,
        use_tda: bool,
    ) -> Array | None:
        method_name = "post_tda_correction" if use_tda else "post_tddft_correction"
        if (
            self.xc_params is not None
            and self.xc_functional is not None
            and not callable(getattr(self.xc_functional, method_name, None))
            and getattr(self.xc_functional, "include_pt2_channel", None) is False
        ):
            return None
        resolved_xc = _resolve_xc_functional(
            self.molecule,
            self.xc_functional,
            self.xc_params,
        )
        if resolved_xc is None:
            return None
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

    def tda(self, nstates: int | None = None) -> TDAResult:
        mode = str(self.eigensolver).lower()
        if mode not in {"auto", "davidson"}:
            raise ValueError(
                f"Unsupported eigensolver={self.eigensolver!r}. Choose one of {{'auto', 'davidson'}}."
            )

        vind, diagonal, delta_eps = build_restricted_tda_operator(
            self.molecule,
            self.xc_functional,
            xc_params=self.xc_params,
            occupation_tolerance=self.occupation_tolerance,
        )
        result = solve_tda_from_operator(
            delta_eps,
            vind,
            diagonal,
            nstates=nstates,
            excitation_threshold=self.excitation_threshold,
            davidson_tol=self.davidson_tol,
            davidson_max_iter=self.davidson_max_iter,
            davidson_max_subspace=self.davidson_max_subspace,
        )
        return self._apply_posthoc_correction(result, use_tda=True)

    def gen_tda_vind(self):
        return gen_tda_vind(
            self.molecule,
            self.xc_functional,
            xc_params=self.xc_params,
            occupation_tolerance=self.occupation_tolerance,
        )

    def gen_tdhf_vind(self):
        return gen_tdhf_vind(
            self.molecule,
            self.xc_functional,
            xc_params=self.xc_params,
            occupation_tolerance=self.occupation_tolerance,
        )

    def kernel(self, nstates: int | None = None) -> TDDFTResult:
        mode = str(self.eigensolver).lower()
        if mode not in {"auto", "davidson"}:
            raise ValueError(
                f"Unsupported eigensolver={self.eigensolver!r}. Choose one of {{'auto', 'davidson'}}."
            )

        vind_tdhf = gen_tdhf_vind(
            self.molecule,
            self.xc_functional,
            xc_params=self.xc_params,
            occupation_tolerance=self.occupation_tolerance,
        )
        result = solve_casida_from_tdhf_operator(
            _restricted_delta_eps(self.molecule, self.occupation_tolerance),
            vind_tdhf,
            nstates=nstates,
            excitation_threshold=self.excitation_threshold,
            matrix_eps=self.matrix_eps,
            davidson_tol=self.davidson_tol,
            davidson_max_iter=self.davidson_max_iter,
            davidson_max_subspace=self.davidson_max_subspace,
        )
        return self._apply_posthoc_correction(result, use_tda=False)
