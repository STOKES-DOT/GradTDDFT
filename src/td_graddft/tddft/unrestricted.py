from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.lax import Precision
from jaxtyping import Array

from .cisd import unrestricted_cisd_second_order_correction
from ._utils import (
    _casida_metric_factor,
    _resolve_xc_functional,
    _symmetrize,
    _transition_densities_on_grid,
)


def _pytree_dataclass(cls):
    def tree_flatten(self):
        children = tuple(getattr(self, field.name) for field in fields(self))
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
class UnrestrictedResponseMatrices:
    """Spin-block A/B matrices for unrestricted TDDFT response."""

    orbital_energy_differences_alpha: Array
    orbital_energy_differences_beta: Array
    a_aa: Array
    a_ab: Array
    a_ba: Array
    a_bb: Array
    b_aa: Array
    b_ab: Array
    b_ba: Array
    b_bb: Array
    a_matrix: Array
    b_matrix: Array


@_pytree_dataclass
@dataclass(frozen=True)
class UnrestrictedTDAMatrices:
    """Spin-block TDA response matrix for unrestricted references."""

    orbital_energy_differences_alpha: Array
    orbital_energy_differences_beta: Array
    a_aa: Array
    a_ab: Array
    a_ba: Array
    a_bb: Array
    a_matrix: Array


@_pytree_dataclass
@dataclass(frozen=True)
class UnrestrictedTDAResult:
    """Excitation energies and alpha/beta amplitudes from unrestricted TDA."""

    excitation_energies: Array
    amplitudes_alpha: Array
    amplitudes_beta: Array
    a_matrix: Array
    posthoc_correction: Array | None = None


@_pytree_dataclass
@dataclass(frozen=True)
class UnrestrictedTDDFTResult:
    """Excitation energies and (X, Y) amplitudes from unrestricted Casida TDDFT."""

    excitation_energies: Array
    x_amplitudes_alpha: Array
    x_amplitudes_beta: Array
    y_amplitudes_alpha: Array
    y_amplitudes_beta: Array
    a_matrix: Array
    b_matrix: Array
    casida_matrix: Array
    posthoc_correction: Array | None = None


def _unrestricted_orbital_data(
    molecule: Any,
    occupation_tolerance: float,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    mo_coeff = jnp.asarray(molecule.mo_coeff)
    mo_occ = jnp.asarray(molecule.mo_occ)
    mo_energy = jnp.asarray(molecule.mo_energy)
    if mo_coeff.ndim != 3 or mo_coeff.shape[0] != 2:
        raise NotImplementedError(
            "Expected unrestricted orbitals with shape (2, nao, nmo)."
        )

    nmo = int(mo_coeff.shape[-1])

    def _channel_partition(
        spin: int,
        *,
        nocc_hint: Any | None,
    ) -> tuple[Array, Array, Array]:
        if nocc_hint is not None:
            nocc = int(nocc_hint)
            nocc = max(0, min(nocc, nmo))
            occ_idx = jnp.arange(nocc)
            vir_idx = jnp.arange(nocc, nmo)
        else:
            occ_values = mo_occ[spin]
            if isinstance(occ_values, jax.core.Tracer):
                raise ValueError(
                    "JIT-traced unrestricted TDDFT/TDA requires static "
                    "nocc_alpha/nocc_beta on the molecule."
                )
            host_occ = np.asarray(jax.device_get(occ_values))
            occ_idx = jnp.asarray(np.where(host_occ > occupation_tolerance)[0], dtype=jnp.int32)
            vir_idx = jnp.asarray(np.where(host_occ <= occupation_tolerance)[0], dtype=jnp.int32)
        orbo = mo_coeff[spin][:, occ_idx]
        orbv = mo_coeff[spin][:, vir_idx]
        de = mo_energy[spin][vir_idx][None, :] - mo_energy[spin][occ_idx][:, None]
        return orbo, orbv, de

    orbo_a, orbv_a, de_a = _channel_partition(
        0,
        nocc_hint=getattr(molecule, "nocc_alpha", None),
    )
    orbo_b, orbv_b, de_b = _channel_partition(
        1,
        nocc_hint=getattr(molecule, "nocc_beta", None),
    )
    has_alpha_channel = de_a.shape[0] > 0 and de_a.shape[1] > 0
    has_beta_channel = de_b.shape[0] > 0 and de_b.shape[1] > 0
    if not (has_alpha_channel or has_beta_channel):
        raise ValueError(
            "Need at least one occupied-virtual excitation channel across alpha/beta spins."
        )
    return orbo_a, orbv_a, orbo_b, orbv_b, de_a, de_b


def _flatten_spin_blocks(
    aa: Array,
    ab: Array,
    ba: Array,
    bb: Array,
) -> Array:
    naa = aa.shape[0] * aa.shape[1]
    nbb = bb.shape[0] * bb.shape[1]
    flat_aa = aa.reshape(naa, naa)
    flat_ab = ab.reshape(naa, nbb)
    flat_ba = ba.reshape(nbb, naa)
    flat_bb = bb.reshape(nbb, nbb)
    upper = jnp.concatenate([flat_aa, flat_ab], axis=1)
    lower = jnp.concatenate([flat_ba, flat_bb], axis=1)
    return jnp.concatenate([upper, lower], axis=0)


def _spin_densities_on_grid(molecule: Any) -> tuple[Array, Array]:
    ao = jnp.asarray(molecule.ao)
    rdm1 = jnp.asarray(molecule.rdm1)
    if rdm1.ndim != 3 or rdm1.shape[0] != 2:
        raise ValueError("Unrestricted response requires spin-resolved rdm1 with shape (2, nao, nao).")
    rho_a = jnp.einsum("pq,rp,rq->r", rdm1[0], ao, ao, precision=Precision.HIGHEST)
    rho_b = jnp.einsum("pq,rp,rq->r", rdm1[1], ao, ao, precision=Precision.HIGHEST)
    return rho_a, rho_b


def _normalize_spin_kernel_values(raw_kernel: Any, dtype: Any) -> tuple[Array, Array, Array]:
    if isinstance(raw_kernel, (tuple, list)):
        if len(raw_kernel) != 3:
            raise ValueError("spin kernel tuple/list must contain (f_aa, f_ab, f_bb).")
        f_aa, f_ab, f_bb = raw_kernel
    else:
        kernel = jnp.asarray(raw_kernel, dtype=dtype)
        if kernel.ndim == 1:
            return kernel, kernel, kernel
        if kernel.ndim == 2 and kernel.shape[-1] == 3:
            return kernel[:, 0], kernel[:, 1], kernel[:, 2]
        if kernel.ndim == 3 and kernel.shape[-2:] == (2, 2):
            f_aa = kernel[:, 0, 0]
            f_ab = 0.5 * (kernel[:, 0, 1] + kernel[:, 1, 0])
            f_bb = kernel[:, 1, 1]
            return f_aa, f_ab, f_bb
        raise ValueError(
            "Unsupported spin-kernel format. Expected 1D scalar kernel, "
            "(ngrids, 3), (ngrids, 2, 2), or tuple/list (f_aa, f_ab, f_bb)."
        )
    return (
        jnp.asarray(f_aa, dtype=dtype),
        jnp.asarray(f_ab, dtype=dtype),
        jnp.asarray(f_bb, dtype=dtype),
    )


def _spin_resolved_kernel_on_grid(
    molecule: Any,
    resolved_xc: Any,
    density_alpha: Array,
    density_beta: Array,
    dtype: Any,
) -> tuple[Array, Array, Array]:
    spin_grid_kernel = getattr(resolved_xc, "spin_grid_kernel", None)
    if callable(spin_grid_kernel):
        raw_kernel = spin_grid_kernel(molecule)
    else:
        spin_local_kernel = getattr(resolved_xc, "spin_local_kernel", None)
        if callable(spin_local_kernel):
            raw_kernel = spin_local_kernel(density_alpha, density_beta)
        else:
            raise ValueError(
                "Unrestricted strict TDDFT response requires spin-resolved XC kernels "
                "(spin_grid_kernel or spin_local_kernel). Scalar local_kernel/grid_kernel "
                "fallback is an approximation and is disabled."
            )
    return _normalize_spin_kernel_values(raw_kernel, dtype=dtype)


def _build_unrestricted_response_blocks(
    molecule: Any,
    resolved_xc: Any | None,
    *,
    occupation_tolerance: float,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
    if getattr(molecule, "rep_tensor", None) is None:
        raise ValueError("The molecule must provide rep_tensor for Hartree/exchange response.")

    ao = jnp.asarray(molecule.ao)
    weights = jnp.asarray(molecule.grid.weights)
    eri = jnp.asarray(molecule.rep_tensor)
    orbo_a, orbv_a, orbo_b, orbv_b, de_a, de_b = _unrestricted_orbital_data(
        molecule,
        occupation_tolerance,
    )

    nocca, nvira = de_a.shape
    noccb, nvirb = de_b.shape
    diag_aa = jnp.einsum(
        "ia,ij,ab->iajb",
        de_a,
        jnp.eye(nocca, dtype=de_a.dtype),
        jnp.eye(nvira, dtype=de_a.dtype),
        precision=Precision.HIGHEST,
    )
    diag_bb = jnp.einsum(
        "ia,ij,ab->iajb",
        de_b,
        jnp.eye(noccb, dtype=de_b.dtype),
        jnp.eye(nvirb, dtype=de_b.dtype),
        precision=Precision.HIGHEST,
    )

    # A_ia,jb Coulomb term uses (ia|jb); B_ia,jb uses (ia|bj).
    a_coul_aa = jnp.einsum(
        "pqrs,pi,qa,rj,sb->iajb",
        eri,
        orbo_a,
        orbv_a,
        orbo_a,
        orbv_a,
        precision=Precision.HIGHEST,
    )
    a_coul_bb = jnp.einsum(
        "pqrs,pi,qa,rj,sb->iajb",
        eri,
        orbo_b,
        orbv_b,
        orbo_b,
        orbv_b,
        precision=Precision.HIGHEST,
    )
    a_coul_ab = jnp.einsum(
        "pqrs,pi,qa,rj,sb->iajb",
        eri,
        orbo_a,
        orbv_a,
        orbo_b,
        orbv_b,
        precision=Precision.HIGHEST,
    )
    b_coul_aa = jnp.einsum(
        "pqrs,pi,qa,rb,sj->iajb",
        eri,
        orbo_a,
        orbv_a,
        orbv_a,
        orbo_a,
        precision=Precision.HIGHEST,
    )
    b_coul_bb = jnp.einsum(
        "pqrs,pi,qa,rb,sj->iajb",
        eri,
        orbo_b,
        orbv_b,
        orbv_b,
        orbo_b,
        precision=Precision.HIGHEST,
    )
    b_coul_ab = jnp.einsum(
        "pqrs,pi,qa,rb,sj->iajb",
        eri,
        orbo_a,
        orbv_a,
        orbv_b,
        orbo_b,
        precision=Precision.HIGHEST,
    )

    hybrid_fraction = jnp.asarray(
        getattr(molecule, "exact_exchange_fraction", 0.0),
        dtype=de_a.dtype,
    )
    if resolved_xc is not None:
        hybrid_fraction = jnp.asarray(
            getattr(resolved_xc, "exact_exchange_fraction", hybrid_fraction),
            dtype=de_a.dtype,
        )

    a_exch_aa = jnp.einsum(
        "pqrs,pi,qj,ra,sb->iajb",
        eri,
        orbo_a,
        orbo_a,
        orbv_a,
        orbv_a,
        precision=Precision.HIGHEST,
    )
    a_exch_bb = jnp.einsum(
        "pqrs,pi,qj,ra,sb->iajb",
        eri,
        orbo_b,
        orbo_b,
        orbv_b,
        orbv_b,
        precision=Precision.HIGHEST,
    )
    b_exch_aa = jnp.einsum(
        "pqrs,pi,qb,ra,sj->iajb",
        eri,
        orbo_a,
        orbv_a,
        orbv_a,
        orbo_a,
        precision=Precision.HIGHEST,
    )
    b_exch_bb = jnp.einsum(
        "pqrs,pi,qb,ra,sj->iajb",
        eri,
        orbo_b,
        orbv_b,
        orbv_b,
        orbo_b,
        precision=Precision.HIGHEST,
    )

    xc_aa = jnp.zeros_like(a_coul_aa)
    xc_bb = jnp.zeros_like(a_coul_bb)
    xc_ab = jnp.zeros_like(a_coul_ab)
    if resolved_xc is not None:
        rho_a, rho_b = _spin_densities_on_grid(molecule)
        f_aa, f_ab, f_bb = _spin_resolved_kernel_on_grid(
            molecule,
            resolved_xc,
            rho_a,
            rho_b,
            de_a.dtype,
        )
        weighted_f_aa = weights * f_aa
        weighted_f_ab = weights * f_ab
        weighted_f_bb = weights * f_bb
        rho_ov_a = _transition_densities_on_grid(ao, orbo_a, orbv_a)
        rho_ov_b = _transition_densities_on_grid(ao, orbo_b, orbv_b)
        xc_aa = jnp.einsum(
            "ria,rjb,r->iajb",
            rho_ov_a,
            rho_ov_a,
            weighted_f_aa,
            precision=Precision.HIGHEST,
        )
        xc_bb = jnp.einsum(
            "ria,rjb,r->iajb",
            rho_ov_b,
            rho_ov_b,
            weighted_f_bb,
            precision=Precision.HIGHEST,
        )
        xc_ab = jnp.einsum(
            "ria,rjb,r->iajb",
            rho_ov_a,
            rho_ov_b,
            weighted_f_ab,
            precision=Precision.HIGHEST,
        )

    a_aa = diag_aa + a_coul_aa - hybrid_fraction * a_exch_aa + xc_aa
    a_bb = diag_bb + a_coul_bb - hybrid_fraction * a_exch_bb + xc_bb
    b_aa = b_coul_aa - hybrid_fraction * b_exch_aa + xc_aa
    b_bb = b_coul_bb - hybrid_fraction * b_exch_bb + xc_bb
    a_ab = a_coul_ab + xc_ab
    b_ab = b_coul_ab + xc_ab
    a_ba = jnp.transpose(a_ab, (2, 3, 0, 1))
    b_ba = jnp.transpose(b_ab, (2, 3, 0, 1))
    return de_a, de_b, a_aa, a_ab, a_ba, a_bb, b_aa, b_ab, b_ba, b_bb


def build_unrestricted_response_matrices(
    molecule: Any,
    xc_functional: Any | None = None,
    *,
    xc_params: Any | None = None,
    occupation_tolerance: float = 1e-8,
) -> UnrestrictedResponseMatrices:
    """Build unrestricted spin-block A/B TDDFT response matrices in pure JAX."""

    resolved_xc = _resolve_xc_functional(molecule, xc_functional, xc_params)
    (
        de_a,
        de_b,
        a_aa,
        a_ab,
        a_ba,
        a_bb,
        b_aa,
        b_ab,
        b_ba,
        b_bb,
    ) = _build_unrestricted_response_blocks(
        molecule,
        resolved_xc,
        occupation_tolerance=occupation_tolerance,
    )
    flat_a = _symmetrize(_flatten_spin_blocks(a_aa, a_ab, a_ba, a_bb))
    flat_b = _symmetrize(_flatten_spin_blocks(b_aa, b_ab, b_ba, b_bb))
    return UnrestrictedResponseMatrices(
        orbital_energy_differences_alpha=de_a,
        orbital_energy_differences_beta=de_b,
        a_aa=a_aa,
        a_ab=a_ab,
        a_ba=a_ba,
        a_bb=a_bb,
        b_aa=b_aa,
        b_ab=b_ab,
        b_ba=b_ba,
        b_bb=b_bb,
        a_matrix=flat_a,
        b_matrix=flat_b,
    )


def build_unrestricted_tda_matrices(
    molecule: Any,
    xc_functional: Any | None = None,
    *,
    xc_params: Any | None = None,
    occupation_tolerance: float = 1e-8,
) -> UnrestrictedTDAMatrices:
    """Build unrestricted spin-block TDA response matrices in pure JAX."""

    resp = build_unrestricted_response_matrices(
        molecule,
        xc_functional,
        xc_params=xc_params,
        occupation_tolerance=occupation_tolerance,
    )
    return UnrestrictedTDAMatrices(
        orbital_energy_differences_alpha=resp.orbital_energy_differences_alpha,
        orbital_energy_differences_beta=resp.orbital_energy_differences_beta,
        a_aa=resp.a_aa,
        a_ab=resp.a_ab,
        a_ba=resp.a_ba,
        a_bb=resp.a_bb,
        a_matrix=resp.a_matrix,
    )


def solve_unrestricted_tda(
    matrices: UnrestrictedTDAMatrices,
    *,
    nstates: int | None = None,
    excitation_threshold: float = 1e-7,
) -> UnrestrictedTDAResult:
    """Solve unrestricted TDA from spin-block response matrices."""

    de_a = matrices.orbital_energy_differences_alpha
    de_b = matrices.orbital_energy_differences_beta
    nocca, nvira = de_a.shape
    noccb, nvirb = de_b.shape
    naa = nocca * nvira

    eigvals, eigvecs = jnp.linalg.eigh(_symmetrize(matrices.a_matrix))
    dim = int(eigvals.shape[0])
    nroots = dim if nstates is None else min(int(nstates), dim)
    valid = eigvals > excitation_threshold
    order = jnp.argsort(jnp.where(valid, eigvals, jnp.inf))
    keep = order[:nroots]
    keep_mask = valid[keep]

    energies = jnp.where(keep_mask, eigvals[keep], 0.0)
    amps = eigvecs[:, keep].T
    amps = amps * keep_mask[:, None]
    amplitudes_alpha = amps[:, :naa].reshape(nroots, nocca, nvira)
    amplitudes_beta = amps[:, naa:].reshape(nroots, noccb, nvirb)
    return UnrestrictedTDAResult(
        excitation_energies=energies,
        amplitudes_alpha=amplitudes_alpha,
        amplitudes_beta=amplitudes_beta,
        a_matrix=matrices.a_matrix,
    )


def solve_unrestricted_casida(
    matrices: UnrestrictedResponseMatrices,
    *,
    nstates: int | None = None,
    excitation_threshold: float = 1e-7,
    matrix_eps: float = 1e-10,
) -> UnrestrictedTDDFTResult:
    """Solve unrestricted Casida TDDFT equation from A/B response matrices."""

    de_a = matrices.orbital_energy_differences_alpha
    de_b = matrices.orbital_energy_differences_beta
    nocca, nvira = de_a.shape
    noccb, nvirb = de_b.shape
    naa = nocca * nvira

    flat_a = _symmetrize(matrices.a_matrix)
    flat_b = _symmetrize(matrices.b_matrix)
    a_plus_b = _symmetrize(flat_a + flat_b)
    a_minus_b = _symmetrize(flat_a - flat_b)
    metric_factor = _casida_metric_factor(a_minus_b, matrix_eps)
    casida_matrix = _symmetrize(metric_factor.T.conj() @ a_plus_b @ metric_factor)

    w2, vecs = jnp.linalg.eigh(casida_matrix)
    dim = int(w2.shape[0])
    nroots = dim if nstates is None else min(int(nstates), dim)
    valid = w2 > excitation_threshold**2
    order = jnp.argsort(jnp.where(valid, w2, jnp.inf))
    keep = order[:nroots]
    keep_mask = valid[keep]

    w = jnp.sqrt(jnp.maximum(w2[keep], 0.0))
    w = jnp.where(keep_mask, w, 0.0)
    f_vectors = vecs[:, keep]
    f_vectors = f_vectors * keep_mask[jnp.newaxis, :]
    x_plus_y = metric_factor @ f_vectors
    safe_w = jnp.where(keep_mask, w, 1.0)
    x_minus_y = (a_plus_b @ x_plus_y) / safe_w[jnp.newaxis, :]

    x = 0.5 * (x_plus_y + x_minus_y)
    y = 0.5 * (x_plus_y - x_minus_y)
    x = x * keep_mask[jnp.newaxis, :]
    y = y * keep_mask[jnp.newaxis, :]
    norm = jnp.sum(jnp.abs(x) ** 2, axis=0) - jnp.sum(jnp.abs(y) ** 2, axis=0)
    scale = 1.0 / jnp.sqrt(jnp.maximum(jnp.abs(norm), matrix_eps))
    x = x * scale[jnp.newaxis, :]
    y = y * scale[jnp.newaxis, :]

    x_alpha = x[:naa, :].T.reshape(nroots, nocca, nvira)
    x_beta = x[naa:, :].T.reshape(nroots, noccb, nvirb)
    y_alpha = y[:naa, :].T.reshape(nroots, nocca, nvira)
    y_beta = y[naa:, :].T.reshape(nroots, noccb, nvirb)
    return UnrestrictedTDDFTResult(
        excitation_energies=w,
        x_amplitudes_alpha=x_alpha,
        x_amplitudes_beta=x_beta,
        y_amplitudes_alpha=y_alpha,
        y_amplitudes_beta=y_beta,
        a_matrix=matrices.a_matrix,
        b_matrix=matrices.b_matrix,
        casida_matrix=casida_matrix,
    )


@dataclass(frozen=True)
class UnrestrictedTDA:
    """PySCF-like unrestricted TDA driver."""

    molecule: Any
    xc_functional: Any | None = None
    xc_params: Any | None = None
    occupation_tolerance: float = 1e-8
    excitation_threshold: float = 1e-7

    def build_matrices(self) -> UnrestrictedTDAMatrices:
        return build_unrestricted_tda_matrices(
            self.molecule,
            self.xc_functional,
            xc_params=self.xc_params,
            occupation_tolerance=self.occupation_tolerance,
        )

    def gen_tda_vind(self):
        matrices = self.build_matrices()
        flat_a = matrices.a_matrix

        def vind(x: Array) -> Array:
            x = jnp.asarray(x).reshape(-1, flat_a.shape[0])
            return x @ flat_a.T

        return vind, flat_a

    def _posthoc_correction(self, result: UnrestrictedTDAResult) -> Array | None:
        resolved_xc = _resolve_xc_functional(
            self.molecule,
            self.xc_functional,
            self.xc_params,
        )
        if resolved_xc is None:
            return None
        correction_fn = getattr(resolved_xc, "post_tda_correction", None)
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
                "post_tda_correction must return a scalar or shape "
                f"{result.excitation_energies.shape}, got {correction.shape}."
            )
        return correction

    def _apply_posthoc_correction(
        self,
        result: UnrestrictedTDAResult,
    ) -> UnrestrictedTDAResult:
        correction = self._posthoc_correction(result)
        if correction is None:
            return result
        return replace(
            result,
            excitation_energies=result.excitation_energies + correction,
            posthoc_correction=correction,
        )

    def kernel(self, nstates: int | None = None) -> UnrestrictedTDAResult:
        result = solve_unrestricted_tda(
            self.build_matrices(),
            nstates=nstates,
            excitation_threshold=self.excitation_threshold,
        )
        return self._apply_posthoc_correction(result)


@dataclass(frozen=True)
class UnrestrictedCasidaTDDFT:
    """PySCF-like unrestricted Casida TDDFT driver."""

    molecule: Any
    xc_functional: Any | None = None
    xc_params: Any | None = None
    occupation_tolerance: float = 1e-8
    excitation_threshold: float = 1e-7
    matrix_eps: float = 1e-10

    def build_matrices(self) -> UnrestrictedResponseMatrices:
        return build_unrestricted_response_matrices(
            self.molecule,
            self.xc_functional,
            xc_params=self.xc_params,
            occupation_tolerance=self.occupation_tolerance,
        )

    def tda(self, nstates: int | None = None) -> UnrestrictedTDAResult:
        tda_mats = build_unrestricted_tda_matrices(
            self.molecule,
            self.xc_functional,
            xc_params=self.xc_params,
            occupation_tolerance=self.occupation_tolerance,
        )
        result = solve_unrestricted_tda(
            tda_mats,
            nstates=nstates,
            excitation_threshold=self.excitation_threshold,
        )
        return self._apply_posthoc_correction(result, use_tda=True)

    def gen_tda_vind(self):
        matrices = self.build_matrices()
        flat_a = matrices.a_matrix

        def vind(x: Array) -> Array:
            x = jnp.asarray(x).reshape(-1, flat_a.shape[0])
            return x @ flat_a.T

        return vind, flat_a

    def gen_tdhf_vind(self):
        matrices = self.build_matrices()
        flat_a = matrices.a_matrix
        flat_b = matrices.b_matrix

        def vind(z: Array) -> Array:
            z = jnp.asarray(z).reshape(-1, 2 * flat_a.shape[0])
            x = z[:, : flat_a.shape[0]]
            y = z[:, flat_a.shape[0] :]
            upper = x @ flat_a.T + y @ flat_b.T
            lower = -(x @ flat_b.T + y @ flat_a.T)
            return jnp.concatenate([upper, lower], axis=-1)

        return vind, flat_a, flat_b

    def _posthoc_correction(
        self,
        result: UnrestrictedTDAResult | UnrestrictedTDDFTResult,
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
        result: UnrestrictedTDAResult | UnrestrictedTDDFTResult,
        *,
        use_tda: bool,
    ) -> UnrestrictedTDAResult | UnrestrictedTDDFTResult:
        correction = self._posthoc_correction(result, use_tda=use_tda)
        if correction is None:
            return result
        return replace(
            result,
            excitation_energies=result.excitation_energies + correction,
            posthoc_correction=correction,
        )

    def kernel(self, nstates: int | None = None) -> UnrestrictedTDDFTResult:
        result = solve_unrestricted_casida(
            self.build_matrices(),
            nstates=nstates,
            excitation_threshold=self.excitation_threshold,
            matrix_eps=self.matrix_eps,
        )
        return self._apply_posthoc_correction(result, use_tda=False)
