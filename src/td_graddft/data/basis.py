from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .molecule import MoleculeSpec, parse_molecule_spec
from .pyscf_basis_loader import load_basis_from_snapshot


def _gaussian_int(n: int, alpha: np.ndarray) -> np.ndarray:
    """Match PySCF's radial Gaussian integral helper."""

    n1 = 0.5 * (n + 1)
    return math.gamma(n1) / (2.0 * np.asarray(alpha, dtype=float) ** n1)


def _gto_norm(l: int, exponents: np.ndarray) -> np.ndarray:
    """Match pyscf.gto.mole.gto_norm for radial factors."""

    return 1.0 / np.sqrt(_gaussian_int(2 * l + 2, 2.0 * np.asarray(exponents, dtype=float)))


def _normalize_raw_shell_coefficients(
    l: int,
    primitive_rows: list[list[object]],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw PySCF basis rows to mol.bas_exp / mol.bas_ctr_coeff convention."""

    rows = sorted(
        [[float(value) for value in row] for row in primitive_rows],
        reverse=True,
    )
    exponents = np.asarray([row[0] for row in rows], dtype=float)
    coeff_rows = np.asarray([row[1:] for row in rows], dtype=float)
    if coeff_rows.ndim == 1:
        coeff_rows = coeff_rows[:, None]

    # PySCF make_env:
    #   cs = raw_coeff * gto_norm(l, es)
    #   cs = _nomalize_contracted_ao(l, es, cs)
    # PySCF bas_ctr_coeff:
    #   bas_ctr_coeff = cs / gto_norm(l, es)
    # Therefore the public coefficients used by mol.bas_ctr_coeff are:
    #   raw_coeff * shell_normalization
    primitive_norm = coeff_rows * _gto_norm(l, exponents)[:, None]
    metric = _gaussian_int(2 * l + 2, exponents[:, None] + exponents[None, :])
    shell_norm = 1.0 / np.sqrt(
        np.einsum("pi,pq,qi->i", primitive_norm, metric, primitive_norm)
    )
    coefficients = coeff_rows * shell_norm[None, :]
    return exponents, coefficients


def cartesian_angular_tuples(l: int) -> list[tuple[int, int, int]]:
    """Return Cartesian (lx, ly, lz) tuples in PySCF cart ordering."""

    if l < 0:
        raise ValueError("Angular momentum l must be non-negative.")
    tuples: list[tuple[int, int, int]] = []
    for lx in range(l, -1, -1):
        rem = l - lx
        for ly in range(rem, -1, -1):
            lz = rem - ly
            tuples.append((lx, ly, lz))
    return tuples


def _contains_jax_tracer(value: Any) -> bool:
    if isinstance(value, jax.core.Tracer):
        return True
    if isinstance(value, dict):
        return any(_contains_jax_tracer(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_jax_tracer(item) for item in value)
    return False


def _stack_geometry_arrays(values: list[Any], *, shape: tuple[int, ...]) -> Array:
    if _contains_jax_tracer(values):
        return jnp.stack([jnp.asarray(value) for value in values], axis=0) if values else jnp.zeros(shape)
    return np.stack([np.asarray(value, dtype=float) for value in values], axis=0) if values else np.zeros(shape)


@dataclass(frozen=True)
class CartesianAO:
    """Single contracted Cartesian AO (not spherical-harmonic AO)."""

    center: Array
    angular: tuple[int, int, int]
    exponents: Array
    coefficients: Array


@dataclass(frozen=True)
class PairBatchGroup:
    """Lower-triangle AO pair batch sharing the same primitive/kernel signature."""

    signature: tuple[tuple[int, int, int], tuple[int, int, int], int, int]
    row_idx: Array
    col_idx: Array


@dataclass(frozen=True)
class ShellPairBatchGroup:
    """Lower-triangle shell pair batch sharing the same shell signature."""

    signature: tuple[tuple[tuple[int, int, int], ...], tuple[tuple[int, int, int], ...], int, int]
    row_idx: Array
    col_idx: Array


@dataclass(frozen=True)
class QuartetBatchGroup:
    """Unique lower-triangle AO quartet batch sharing the same ERI signature."""

    signature: tuple[
        tuple[int, int, int],
        tuple[int, int, int],
        tuple[int, int, int],
        tuple[int, int, int],
        int,
        int,
        int,
        int,
    ]
    idx_i: Array
    idx_j: Array
    idx_k: Array
    idx_l: Array


@dataclass(frozen=True)
class ShellQuartetBatchGroup:
    """Unique lower-triangle shell quartet batch sharing the same ERI shell signature."""

    signature: tuple[
        tuple[tuple[int, int, int], ...],
        tuple[tuple[int, int, int], ...],
        tuple[tuple[int, int, int], ...],
        tuple[tuple[int, int, int], ...],
        int,
        int,
        int,
        int,
    ]
    idx_i: Array
    idx_j: Array
    idx_k: Array
    idx_l: Array
    scatter_i: Array | None = None
    scatter_j: Array | None = None
    scatter_k: Array | None = None
    scatter_l: Array | None = None
    batch_inputs: tuple[Array, ...] | None = None


@dataclass(frozen=True)
class ContractedShell:
    """One contracted Cartesian shell block sharing exponents/coefficients/center."""

    center: Array
    angulars: tuple[tuple[int, int, int], ...]
    exponents: Array
    coefficients: Array
    ao_indices: Array


@dataclass(frozen=True)
class CartesianBasis:
    """Cartesian AO basis and optional nuclear data."""

    aos: tuple[CartesianAO, ...]
    precompute_eri_groups: bool = True
    atom_coords: Array | None = None
    atom_charges: Array | None = None
    ao_centers: Array | None = None
    ao_exponents_padded: Array | None = None
    ao_coefficients_padded: Array | None = None
    ao_nprims: Array | None = None
    ao_angulars: tuple[tuple[int, int, int], ...] = field(default_factory=tuple)
    ao_nprims_tuple: tuple[int, ...] = field(default_factory=tuple)
    pair_groups: tuple[PairBatchGroup, ...] = field(default_factory=tuple)
    shells: tuple[ContractedShell, ...] = field(default_factory=tuple)
    shell_centers: Array | None = None
    shell_exponents_padded: Array | None = None
    shell_coefficients_padded: Array | None = None
    shell_nprims: Array | None = None
    shell_nprims_tuple: tuple[int, ...] = field(default_factory=tuple)
    shell_pair_groups: tuple[ShellPairBatchGroup, ...] = field(default_factory=tuple)
    shell_ao_indices_padded: Array | None = None
    shell_ao_sizes: Array | None = None
    quartet_groups: tuple[QuartetBatchGroup, ...] = field(default_factory=tuple)
    shell_quartet_groups: tuple[ShellQuartetBatchGroup, ...] = field(default_factory=tuple)

    @property
    def nao(self) -> int:
        return len(self.aos)

    def __post_init__(self) -> None:
        nao = len(self.aos)
        geometry_is_traced = _contains_jax_tracer(
            [ao.center for ao in self.aos] + [shell.center for shell in self.shells]
        )
        if self.ao_centers is None:
            centers = _stack_geometry_arrays(
                [ao.center for ao in self.aos],
                shape=(0, 3),
            )
            object.__setattr__(self, "ao_centers", centers)

        if not self.ao_angulars:
            object.__setattr__(self, "ao_angulars", tuple(ao.angular for ao in self.aos))

        if not self.ao_nprims_tuple:
            object.__setattr__(
                self,
                "ao_nprims_tuple",
                tuple(int(ao.exponents.shape[0]) for ao in self.aos),
            )

        if self.ao_nprims is None:
            object.__setattr__(
                self,
                "ao_nprims",
                np.asarray(self.ao_nprims_tuple, dtype=np.int32),
            )

        if self.ao_exponents_padded is None or self.ao_coefficients_padded is None:
            max_nprim = max(self.ao_nprims_tuple, default=0)
            if nao == 0 or max_nprim == 0:
                exp_pad = np.zeros((nao, 0), dtype=float)
                coeff_pad = np.zeros((nao, 0), dtype=float)
            else:
                exp_dtype = np.result_type(*[np.asarray(ao.exponents).dtype for ao in self.aos])
                coeff_dtype = np.result_type(
                    *[np.asarray(ao.coefficients).dtype for ao in self.aos]
                )
                exp_pad_np = np.zeros((nao, max_nprim), dtype=np.dtype(exp_dtype))
                coeff_pad_np = np.zeros((nao, max_nprim), dtype=np.dtype(coeff_dtype))
                for idx, ao in enumerate(self.aos):
                    nprim = int(ao.exponents.shape[0])
                    exp_pad_np[idx, :nprim] = np.asarray(ao.exponents)
                    coeff_pad_np[idx, :nprim] = np.asarray(ao.coefficients)
                exp_pad = exp_pad_np
                coeff_pad = coeff_pad_np
            object.__setattr__(self, "ao_exponents_padded", exp_pad)
            object.__setattr__(self, "ao_coefficients_padded", coeff_pad)

        if not self.pair_groups:
            groups: dict[
                tuple[tuple[int, int, int], tuple[int, int, int], int, int],
                dict[str, list[int]],
            ] = {}
            angulars = self.ao_angulars
            nprims = self.ao_nprims_tuple
            for i in range(nao):
                sig_i = angulars[i]
                nprim_i = nprims[i]
                for j in range(i + 1):
                    signature = (sig_i, angulars[j], nprim_i, nprims[j])
                    bucket = groups.setdefault(signature, {"row": [], "col": []})
                    bucket["row"].append(i)
                    bucket["col"].append(j)
            pair_groups = tuple(
                PairBatchGroup(
                    signature=signature,
                    row_idx=np.asarray(bucket["row"], dtype=np.int32),
                    col_idx=np.asarray(bucket["col"], dtype=np.int32),
                )
                for signature, bucket in groups.items()
            )
            object.__setattr__(self, "pair_groups", pair_groups)

        nshell = len(self.shells)
        if nshell and self.shell_centers is None:
            object.__setattr__(
                self,
                "shell_centers",
                _stack_geometry_arrays(
                    [shell.center for shell in self.shells],
                    shape=(0, 3),
                ),
            )
        elif self.shell_centers is None:
            object.__setattr__(self, "shell_centers", np.zeros((0, 3), dtype=float))

        if not self.shell_nprims_tuple:
            object.__setattr__(
                self,
                "shell_nprims_tuple",
                tuple(int(shell.exponents.shape[0]) for shell in self.shells),
            )

        if self.shell_nprims is None:
            object.__setattr__(
                self,
                "shell_nprims",
                np.asarray(self.shell_nprims_tuple, dtype=np.int32),
            )

        if self.shell_exponents_padded is None or self.shell_coefficients_padded is None:
            max_shell_nprim = max(self.shell_nprims_tuple, default=0)
            if nshell == 0 or max_shell_nprim == 0:
                exp_pad = np.zeros((nshell, 0), dtype=float)
                coeff_pad = np.zeros((nshell, 0), dtype=float)
            else:
                exp_dtype = np.result_type(
                    *[np.asarray(shell.exponents).dtype for shell in self.shells]
                )
                coeff_dtype = np.result_type(
                    *[np.asarray(shell.coefficients).dtype for shell in self.shells]
                )
                exp_pad_np = np.zeros((nshell, max_shell_nprim), dtype=np.dtype(exp_dtype))
                coeff_pad_np = np.zeros(
                    (nshell, max_shell_nprim),
                    dtype=np.dtype(coeff_dtype),
                )
                for idx, shell in enumerate(self.shells):
                    nprim = int(shell.exponents.shape[0])
                    exp_pad_np[idx, :nprim] = np.asarray(shell.exponents)
                    coeff_pad_np[idx, :nprim] = np.asarray(shell.coefficients)
                exp_pad = exp_pad_np
                coeff_pad = coeff_pad_np
            object.__setattr__(self, "shell_exponents_padded", exp_pad)
            object.__setattr__(self, "shell_coefficients_padded", coeff_pad)

        if nshell and not self.shell_pair_groups:
            shell_groups: dict[
                tuple[tuple[tuple[int, int, int], ...], tuple[tuple[int, int, int], ...], int, int],
                dict[str, list[int]],
            ] = {}
            for i in range(nshell):
                shell_i = self.shells[i]
                nprim_i = self.shell_nprims_tuple[i]
                for j in range(i + 1):
                    shell_j = self.shells[j]
                    signature = (
                        shell_i.angulars,
                        shell_j.angulars,
                        nprim_i,
                        self.shell_nprims_tuple[j],
                    )
                    bucket = shell_groups.setdefault(signature, {"row": [], "col": []})
                    bucket["row"].append(i)
                    bucket["col"].append(j)
            object.__setattr__(
                self,
                "shell_pair_groups",
                tuple(
                    ShellPairBatchGroup(
                        signature=signature,
                        row_idx=np.asarray(bucket["row"], dtype=np.int32),
                        col_idx=np.asarray(bucket["col"], dtype=np.int32),
                    )
                    for signature, bucket in shell_groups.items()
                ),
            )

        if self.shell_ao_sizes is None:
            object.__setattr__(
                self,
                "shell_ao_sizes",
                np.asarray([len(shell.ao_indices) for shell in self.shells], dtype=np.int32),
            )

        if self.shell_ao_indices_padded is None:
            max_shell_ao = max((len(shell.ao_indices) for shell in self.shells), default=0)
            if nshell == 0 or max_shell_ao == 0:
                ao_idx_pad = np.zeros((nshell, 0), dtype=np.int32)
            else:
                ao_idx_pad_np = np.zeros((nshell, max_shell_ao), dtype=np.int32)
                for idx, shell in enumerate(self.shells):
                    shell_len = len(shell.ao_indices)
                    ao_idx_pad_np[idx, :shell_len] = np.asarray(shell.ao_indices, dtype=np.int32)
                ao_idx_pad = ao_idx_pad_np
            object.__setattr__(self, "shell_ao_indices_padded", ao_idx_pad)

        if self.precompute_eri_groups and nao and not self.quartet_groups:
            ao_quartets: dict[
                tuple[
                    tuple[int, int, int],
                    tuple[int, int, int],
                    tuple[int, int, int],
                    tuple[int, int, int],
                    int,
                    int,
                    int,
                    int,
                ],
                dict[str, list[int]],
            ] = {}
            ao_pairs = tuple((i, j) for i in range(nao) for j in range(i + 1))
            angulars = self.ao_angulars
            nprims = self.ao_nprims_tuple
            for ij_pos, (i, j) in enumerate(ao_pairs):
                for kl_pos in range(ij_pos + 1):
                    k, l = ao_pairs[kl_pos]
                    signature = (
                        angulars[i],
                        angulars[j],
                        angulars[k],
                        angulars[l],
                        nprims[i],
                        nprims[j],
                        nprims[k],
                        nprims[l],
                    )
                    bucket = ao_quartets.setdefault(
                        signature,
                        {"i": [], "j": [], "k": [], "l": []},
                    )
                    bucket["i"].append(i)
                    bucket["j"].append(j)
                    bucket["k"].append(k)
                    bucket["l"].append(l)
            object.__setattr__(
                self,
                "quartet_groups",
                tuple(
                    QuartetBatchGroup(
                        signature=signature,
                        idx_i=np.asarray(bucket["i"], dtype=np.int32),
                        idx_j=np.asarray(bucket["j"], dtype=np.int32),
                        idx_k=np.asarray(bucket["k"], dtype=np.int32),
                        idx_l=np.asarray(bucket["l"], dtype=np.int32),
                    )
                    for signature, bucket in ao_quartets.items()
                ),
            )

        if self.precompute_eri_groups and nshell and not self.shell_quartet_groups:
            shell_quartets: dict[
                tuple[
                    tuple[tuple[int, int, int], ...],
                    tuple[tuple[int, int, int], ...],
                    tuple[tuple[int, int, int], ...],
                    tuple[tuple[int, int, int], ...],
                    int,
                    int,
                    int,
                    int,
                ],
                dict[str, list[int]],
            ] = {}
            shell_pairs = tuple((i, j) for i in range(nshell) for j in range(i + 1))
            for ij_pos, (i, j) in enumerate(shell_pairs):
                shell_i = self.shells[i]
                shell_j = self.shells[j]
                for kl_pos in range(ij_pos + 1):
                    k, l = shell_pairs[kl_pos]
                    shell_k = self.shells[k]
                    shell_l = self.shells[l]
                    signature = (
                        shell_i.angulars,
                        shell_j.angulars,
                        shell_k.angulars,
                        shell_l.angulars,
                        self.shell_nprims_tuple[i],
                        self.shell_nprims_tuple[j],
                        self.shell_nprims_tuple[k],
                        self.shell_nprims_tuple[l],
                    )
                    bucket = shell_quartets.setdefault(
                        signature,
                        {"i": [], "j": [], "k": [], "l": []},
                    )
                    bucket["i"].append(i)
                    bucket["j"].append(j)
                    bucket["k"].append(k)
                    bucket["l"].append(l)
            object.__setattr__(
                self,
                "shell_quartet_groups",
                tuple(
                    self._build_shell_quartet_group(
                        signature,
                        bucket,
                        geometry_is_traced=geometry_is_traced,
                    )
                    for signature, bucket in shell_quartets.items()
                ),
            )

    def _build_shell_quartet_group(
        self,
        signature,
        bucket: dict[str, list[int]],
        *,
        geometry_is_traced: bool,
    ) -> ShellQuartetBatchGroup:
        idx_i = np.asarray(bucket["i"], dtype=np.int32)
        idx_j = np.asarray(bucket["j"], dtype=np.int32)
        idx_k = np.asarray(bucket["k"], dtype=np.int32)
        idx_l = np.asarray(bucket["l"], dtype=np.int32)
        ni = len(signature[0])
        nj = len(signature[1])
        nk = len(signature[2])
        nl = len(signature[3])
        ao_i = np.asarray(self.shell_ao_indices_padded[idx_i, :ni], dtype=np.int32)
        ao_j = np.asarray(self.shell_ao_indices_padded[idx_j, :nj], dtype=np.int32)
        ao_k = np.asarray(self.shell_ao_indices_padded[idx_k, :nk], dtype=np.int32)
        ao_l = np.asarray(self.shell_ao_indices_padded[idx_l, :nl], dtype=np.int32)

        def flatten_group(ai, aj, ak, al):
            block_shape = (ai.shape[0], ai.shape[1], aj.shape[1], ak.shape[1], al.shape[1])
            ii = np.broadcast_to(ai[:, :, None, None, None], block_shape).reshape(-1)
            jj = np.broadcast_to(aj[:, None, :, None, None], block_shape).reshape(-1)
            kk = np.broadcast_to(ak[:, None, None, :, None], block_shape).reshape(-1)
            ll = np.broadcast_to(al[:, None, None, None, :], block_shape).reshape(-1)
            return ii, jj, kk, ll

        groups = (
            flatten_group(ao_i, ao_j, ao_k, ao_l),
            flatten_group(ao_j, ao_i, ao_k, ao_l),
            flatten_group(ao_i, ao_j, ao_l, ao_k),
            flatten_group(ao_j, ao_i, ao_l, ao_k),
            flatten_group(ao_k, ao_l, ao_i, ao_j),
            flatten_group(ao_l, ao_k, ao_i, ao_j),
            flatten_group(ao_k, ao_l, ao_j, ao_i),
            flatten_group(ao_l, ao_k, ao_j, ao_i),
        )
        scatter_i = np.concatenate([group[0] for group in groups], axis=0).astype(np.int32)
        scatter_j = np.concatenate([group[1] for group in groups], axis=0).astype(np.int32)
        scatter_k = np.concatenate([group[2] for group in groups], axis=0).astype(np.int32)
        scatter_l = np.concatenate([group[3] for group in groups], axis=0).astype(np.int32)
        if geometry_is_traced:
            idx_i_jax = jnp.asarray(idx_i, dtype=jnp.int32)
            idx_j_jax = jnp.asarray(idx_j, dtype=jnp.int32)
            idx_k_jax = jnp.asarray(idx_k, dtype=jnp.int32)
            idx_l_jax = jnp.asarray(idx_l, dtype=jnp.int32)
            shell_exp = jnp.asarray(self.shell_exponents_padded)
            shell_coeff = jnp.asarray(self.shell_coefficients_padded)
            shell_centers = jnp.asarray(self.shell_centers)
            exp_i = shell_exp[idx_i_jax, : signature[4]]
            coeff_i = shell_coeff[idx_i_jax, : signature[4]]
            center_i = shell_centers[idx_i_jax]
            exp_j = shell_exp[idx_j_jax, : signature[5]]
            coeff_j = shell_coeff[idx_j_jax, : signature[5]]
            center_j = shell_centers[idx_j_jax]
            exp_k = shell_exp[idx_k_jax, : signature[6]]
            coeff_k = shell_coeff[idx_k_jax, : signature[6]]
            center_k = shell_centers[idx_k_jax]
            exp_l = shell_exp[idx_l_jax, : signature[7]]
            coeff_l = shell_coeff[idx_l_jax, : signature[7]]
            center_l = shell_centers[idx_l_jax]
        else:
            shell_exp = np.asarray(self.shell_exponents_padded)
            shell_coeff = np.asarray(self.shell_coefficients_padded)
            shell_centers = np.asarray(self.shell_centers)
            exp_i = shell_exp[idx_i, : signature[4]]
            coeff_i = shell_coeff[idx_i, : signature[4]]
            center_i = shell_centers[idx_i]
            exp_j = shell_exp[idx_j, : signature[5]]
            coeff_j = shell_coeff[idx_j, : signature[5]]
            center_j = shell_centers[idx_j]
            exp_k = shell_exp[idx_k, : signature[6]]
            coeff_k = shell_coeff[idx_k, : signature[6]]
            center_k = shell_centers[idx_k]
            exp_l = shell_exp[idx_l, : signature[7]]
            coeff_l = shell_coeff[idx_l, : signature[7]]
            center_l = shell_centers[idx_l]
        return ShellQuartetBatchGroup(
            signature=signature,
            idx_i=idx_i,
            idx_j=idx_j,
            idx_k=idx_k,
            idx_l=idx_l,
            scatter_i=scatter_i,
            scatter_j=scatter_j,
            scatter_k=scatter_k,
            scatter_l=scatter_l,
            batch_inputs=(
                exp_i,
                coeff_i,
                center_i,
                exp_j,
                coeff_j,
                center_j,
                exp_k,
                coeff_k,
                center_k,
                exp_l,
                coeff_l,
                center_l,
            ),
        )


def basis_from_pyscf_mol_cart(
    mol: Any,
    *,
    max_l: int = 3,
    precompute_eri_groups: bool = True,
) -> CartesianBasis:
    """Build a Cartesian AO basis from a PySCF Mole.

    Notes:
    - Requires `mol.cart = True` for direct cartesian AO ordering.
    - Current integral engine supports up to `l=3` (s/p/d/f).
    """

    if not bool(getattr(mol, "cart", False)):
        raise ValueError(
            "basis_from_pyscf_mol_cart requires a PySCF Mole with cart=True."
        )

    aos: list[CartesianAO] = []
    shells: list[ContractedShell] = []
    for ib in range(mol.nbas):
        l = int(mol.bas_angular(ib))
        if l > max_l:
            raise NotImplementedError(
                f"Current JAX integral implementation supports l<= {max_l}, got l={l}."
            )
        atom_idx = int(mol.bas_atom(ib))
        center = np.asarray(mol.atom_coord(atom_idx), dtype=float)
        exponents = np.asarray(mol.bas_exp(ib), dtype=float)
        ctr_coeff = np.asarray(mol.bas_ctr_coeff(ib), dtype=float)  # (nprim, nctr)
        if ctr_coeff.ndim == 1:
            ctr_coeff = ctr_coeff[:, None]

        for ctr in range(ctr_coeff.shape[1]):
            coeff = ctr_coeff[:, ctr]
            angulars = tuple(cartesian_angular_tuples(l))
            shell_start = len(aos)
            for angular in angulars:
                aos.append(
                    CartesianAO(
                        center=center,
                        angular=angular,
                        exponents=exponents,
                        coefficients=coeff,
                    )
                )
            shell_stop = len(aos)
            shells.append(
                ContractedShell(
                    center=center,
                    angulars=angulars,
                    exponents=exponents,
                    coefficients=coeff,
                    ao_indices=np.arange(shell_start, shell_stop, dtype=np.int32),
                )
            )

    return CartesianBasis(
        aos=tuple(aos),
        precompute_eri_groups=bool(precompute_eri_groups),
        atom_coords=np.asarray(mol.atom_coords(), dtype=float),
        atom_charges=np.asarray(mol.atom_charges(), dtype=float),
        shells=tuple(shells),
    )


def basis_from_pyscf_spec(
    atom: Any,
    *,
    basis: Any,
    unit: str = "Angstrom",
    charge: int = 0,
    spin: int = 0,
    cart: bool = True,
    verbose: int = 0,
    max_l: int = 3,
    **mol_kwargs: Any,
) -> CartesianBasis:
    """Build a JAX cartesian basis directly from PySCF-style molecule/basis inputs.

    This keeps PySCF's basis-library lookup and basis-call syntax while converting
    the resolved basis into TD-GradDFT's internal cartesian AO representation.
    """

    try:
        from pyscf import gto
    except ModuleNotFoundError as exc:
        raise ImportError("PySCF is required for basis_from_pyscf_spec.") from exc

    mol = gto.M(
        atom=atom,
        basis=basis,
        unit=unit,
        charge=charge,
        spin=spin,
        cart=bool(cart),
        verbose=int(verbose),
        **mol_kwargs,
    )
    return basis_from_pyscf_mol_cart(mol, max_l=max_l)


def basis_from_spec(
    atom: Any,
    *,
    basis: Any,
    unit: str = "Angstrom",
    charge: int = 0,
    spin: int = 0,
    max_l: int = 3,
    precompute_eri_groups: bool = True,
) -> CartesianBasis:
    """Build a cartesian basis from bundled basis data without PySCF.

    Basis data are read directly from the vendored PySCF basis snapshot.
    """
    spec: MoleculeSpec = parse_molecule_spec(atom, unit=unit, charge=charge, spin=spin)
    return basis_from_molecule_spec(
        spec,
        basis=basis,
        max_l=max_l,
        precompute_eri_groups=precompute_eri_groups,
    )


def basis_from_molecule_spec(
    spec: MoleculeSpec,
    *,
    basis: Any,
    max_l: int = 3,
    precompute_eri_groups: bool = True,
) -> CartesianBasis:
    """Build a cartesian basis from an explicit MoleculeSpec (differentiable coords path)."""

    if not isinstance(basis, str):
        raise TypeError("Strict-JAX basis_from_molecule_spec currently supports named basis strings only.")

    aos: list[CartesianAO] = []
    shells: list[ContractedShell] = []
    for atom_idx, sym in enumerate(spec.symbols):
        shells_raw = load_basis_from_snapshot(str(basis), sym)
        center = jnp.asarray(spec.coords_bohr[atom_idx], dtype=jnp.float64)
        for shell in shells_raw:
            l = int(shell[0])
            if l > max_l:
                raise NotImplementedError(
                    f"Current JAX integral implementation supports l<= {max_l}, got l={l}."
                )
            primitive_rows = shell[1:]
            exponents_np, coeff_rows = _normalize_raw_shell_coefficients(l, primitive_rows)
            exponents = jnp.asarray(exponents_np, dtype=jnp.float64)
            for ctr in range(coeff_rows.shape[1]):
                coeff = jnp.asarray(coeff_rows[:, ctr], dtype=jnp.float64)
                angulars = tuple(cartesian_angular_tuples(l))
                shell_start = len(aos)
                for angular in angulars:
                    aos.append(
                        CartesianAO(
                            center=center,
                            angular=angular,
                            exponents=exponents,
                            coefficients=coeff,
                        )
                    )
                shell_stop = len(aos)
                shells.append(
                    ContractedShell(
                        center=center,
                        angulars=angulars,
                        exponents=exponents,
                        coefficients=coeff,
                        ao_indices=jnp.arange(shell_start, shell_stop, dtype=jnp.int32),
                    )
                )

    return CartesianBasis(
        aos=tuple(aos),
        precompute_eri_groups=bool(precompute_eri_groups),
        atom_coords=jnp.asarray(spec.coords_bohr),
        atom_charges=jnp.asarray(spec.charges),
        shells=tuple(shells),
    )
