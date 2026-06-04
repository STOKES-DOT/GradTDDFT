from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import jax.numpy as jnp
from jax.lax import Precision

from ..data.integrals import eri_pair_matrix_to_mo_eri_slices, rinv_matrices
from ..data.integrals.jax.packed_eri import _metadata_arrays, _mo_pair_products
from ..df import df_factors_to_mo_eri_slices
from ..xc_backend.jax_libxc import RestrictedFeatureBundle

_DM21_BETA = 1.0 / 1024.0


def _bounded_ratio(numerator: Any, denominator: Any) -> jnp.ndarray:
    ratio = jnp.asarray(numerator) / jnp.maximum(jnp.asarray(denominator), 1e-30)
    ratio = jnp.nan_to_num(ratio, nan=0.0, posinf=1e30, neginf=0.0)
    beta = jnp.asarray(_DM21_BETA, dtype=ratio.dtype)
    return beta * ratio / (1.0 + beta * ratio)


def canonical_input_features(
    features: RestrictedFeatureBundle,
    hfx_a: Any,
    hfx_b: Any,
    *,
    density_floor: float = 1e-12,
) -> jnp.ndarray:
    rho_a = jnp.maximum(features.rho_a, density_floor)
    rho_b = jnp.maximum(features.rho_b, density_floor)
    rho = jnp.maximum(features.rho, density_floor)
    tau_a = jnp.maximum(features.tau_a, 0.0)
    tau_b = jnp.maximum(features.tau_b, 0.0)
    norm_grad_a = jnp.maximum(features.sigma_aa, 0.0)
    norm_grad_b = jnp.maximum(features.sigma_bb, 0.0)
    norm_grad = jnp.maximum(features.sigma, 0.0)
    tau_prefactor = (3.0 / 5.0) * (6.0 * jnp.pi**2) ** (2.0 / 3.0)
    reduced_grad = _bounded_ratio(jnp.sqrt(norm_grad), rho ** (4.0 / 3.0))
    reduced_grad_a = _bounded_ratio(jnp.sqrt(norm_grad_a), rho_a ** (4.0 / 3.0))
    reduced_grad_b = _bounded_ratio(jnp.sqrt(norm_grad_b), rho_b ** (4.0 / 3.0))
    reduced_tau_a = _bounded_ratio(tau_a, tau_prefactor * rho_a ** (5.0 / 3.0))
    reduced_tau_b = _bounded_ratio(tau_b, tau_prefactor * rho_b ** (5.0 / 3.0))
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
            reduced_grad,
            reduced_grad_a,
            reduced_grad_b,
            reduced_tau_a,
            reduced_tau_b,
        ],
        axis=-1,
    )
    return jnp.concatenate([leading, hfx_a, hfx_b], axis=-1)


def enhanced_input_features(
    features: RestrictedFeatureBundle,
    semilocal_descriptor: Any,
    *,
    density_floor: float = 1e-12,
) -> jnp.ndarray:
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
            semilocal_descriptor,
        ],
        axis=-1,
    )


def resolve_canonical_hfx_feature_channels(
    molecule: Any | None,
    features: RestrictedFeatureBundle,
    *,
    hf_energy_density: Any | None = None,
    hf_spin_energy_density: tuple[Any, Any] | None = None,
    hfx_channels: int = 2,
    strict_feature_alignment: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    target_channels = max(int(hfx_channels), 1)
    cached = getattr(molecule, "hfx_local", None) if molecule is not None else None
    if cached is not None:
        cached = jnp.asarray(cached)
        if cached.ndim == 3 and cached.shape[0] == 2:
            if strict_feature_alignment and cached.shape[-1] != target_channels:
                raise ValueError(
                    "molecule.hfx_local omega-channel count must match hfx_channels "
                    f"(got {cached.shape[-1]} vs {target_channels})."
                )
            return cached[0], cached[1]
        raise ValueError(
            "molecule.hfx_local must have shape (2, ngrids, n_omega), "
            f"got {cached.shape}."
        )

    if hf_spin_energy_density is not None:
        hfx_a = jnp.asarray(hf_spin_energy_density[0])
        hfx_b = jnp.asarray(hf_spin_energy_density[1])
        if hfx_a.ndim == features.rho.ndim:
            hfx_a = hfx_a[..., None]
        if hfx_b.ndim == features.rho.ndim:
            hfx_b = hfx_b[..., None]
        if hfx_a.shape[-1] == 1 and target_channels > 1:
            hfx_a = jnp.repeat(hfx_a, target_channels, axis=-1)
        if hfx_b.shape[-1] == 1 and target_channels > 1:
            hfx_b = jnp.repeat(hfx_b, target_channels, axis=-1)
        return hfx_a, hfx_b

    if strict_feature_alignment and molecule is not None:
        raise ValueError(
            "canonical input mode requires molecule.hfx_local with shape "
            "(2, ngrids, n_omega), or explicit hf_spin_energy_density channels. "
            "Build the reference with compute_local_hfx_features=True "
            "(typically omega values 0.0 and 0.4)."
        )

    hf_total = (
        jnp.zeros_like(features.rho)
        if hf_energy_density is None
        else jnp.asarray(hf_energy_density)
    )
    local_hfx = jnp.repeat(hf_total[..., None], target_channels, axis=-1)
    return local_hfx, local_hfx


def build_coefficient_inputs(
    features: RestrictedFeatureBundle,
    semilocal_energy_density: Any,
    hf_energy_density: Any,
    *,
    input_feature_mode: str,
    hf_input_mode: str,
    include_pt2_channel: bool,
    density_floor: float,
    hfx_channels: int,
    strict_feature_alignment: bool,
    pt2_energy_density: Any | None = None,
    molecule: Any | None = None,
    hf_spin_energy_density: tuple[Any, Any] | None = None,
    semilocal_descriptor: Any | None = None,
) -> jnp.ndarray:
    pt2_total = (
        jnp.zeros_like(features.rho)
        if pt2_energy_density is None
        else jnp.asarray(pt2_energy_density)
    )
    if input_feature_mode == "canonical":
        hfx_a, hfx_b = resolve_canonical_hfx_feature_channels(
            molecule,
            features,
            hf_energy_density=hf_energy_density,
            hf_spin_energy_density=hf_spin_energy_density,
            hfx_channels=hfx_channels,
            strict_feature_alignment=strict_feature_alignment,
        )
        base = canonical_input_features(
            features,
            hfx_a,
            hfx_b,
            density_floor=density_floor,
        )
        if not include_pt2_channel:
            return base
        return jnp.concatenate([base, pt2_total[..., None]], axis=-1)

    if input_feature_mode != "enhanced":
        raise ValueError(
            f"Unsupported input_feature_mode={input_feature_mode!r}. "
            "Expected 'enhanced' or 'canonical'."
        )

    if semilocal_descriptor is None:
        density = jnp.maximum(jnp.asarray(features.rho), density_floor)
        semilocal_descriptor = jnp.nan_to_num(
            jnp.asarray(semilocal_energy_density) / density,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    base = enhanced_input_features(
        features,
        semilocal_descriptor,
        density_floor=density_floor,
    )
    hf_total = jnp.asarray(hf_energy_density)
    if hf_input_mode == "total_only":
        extras = [hf_total[..., None]]
    elif hf_input_mode == "spin_resolved":
        if hf_spin_energy_density is None:
            hf_a = hf_total
            hf_b = hf_total
        else:
            hf_a, hf_b = hf_spin_energy_density
        extras = [hf_total[..., None], jnp.asarray(hf_a)[..., None], jnp.asarray(hf_b)[..., None]]
    else:
        raise ValueError(
            f"Unsupported hf_input_mode={hf_input_mode!r}. "
            "Expected 'total_only' or 'spin_resolved'."
        )
    if include_pt2_channel:
        extras.append(pt2_total[..., None])
    return jnp.concatenate([base, *extras], axis=-1)


def assemble_basis_channels(
    semilocal_local_channels: Any,
    *,
    hf_projected: Any,
    include_pt2_channel: bool,
    pt2_projected: Any | None = None,
) -> jnp.ndarray:
    channels = [jnp.asarray(semilocal_local_channels)]
    if include_pt2_channel:
        if pt2_projected is None:
            raise ValueError("pt2_projected must be provided when include_pt2_channel=True.")
        channels.append(jnp.asarray(pt2_projected)[..., None])
    channels.append(jnp.asarray(hf_projected)[..., None])
    return jnp.concatenate(channels, axis=-1)


def _int1e_grids_name(mol: Any) -> str:
    return "int1e_grids_cart" if bool(getattr(mol, "cart", False)) else "int1e_grids_sph"


def _int1e_rinv_name(mol: Any) -> str:
    return "int1e_rinv_cart" if bool(getattr(mol, "cart", False)) else "int1e_rinv_sph"


@dataclass(frozen=True)
class ChunkedHFXNu:
    """Lazy grid-chunk API for local HFX ``nu`` matrices.

    The dense equivalent has shape ``(n_omega, ngrids, nao, nao)``.  This API
    keeps the same public shape contract while exposing only concrete grid
    slices through :meth:`grid_chunk`.
    """

    shape: tuple[int, int, int, int]
    chunk_size: int
    _grid_chunk_fn: Any

    @property
    def ndim(self) -> int:
        return 4

    @classmethod
    def from_dense(cls, dense: Any, *, chunk_size: int = 512) -> "ChunkedHFXNu":
        array = np.asarray(dense)
        if array.ndim != 4:
            raise ValueError(
                "dense HFX nu cache must have shape (n_omega, ngrids, nao, nao), "
                f"got {array.shape}."
            )

        def grid_chunk(start: int, stop: int) -> np.ndarray:
            return np.asarray(array[:, int(start) : int(stop)])

        return cls(
            shape=tuple(int(dim) for dim in array.shape),
            chunk_size=int(chunk_size),
            _grid_chunk_fn=grid_chunk,
        )

    @classmethod
    def from_hdf5_dataset(
        cls,
        filename: str,
        dataset_path: str,
        *,
        chunk_size: int = 512,
    ) -> "ChunkedHFXNu":
        try:
            import h5py
        except ModuleNotFoundError as exc:
            raise ImportError("h5py is required for HDF5-backed HFX nu chunks.") from exc

        filename = str(Path(filename).resolve())
        dataset_path = str(dataset_path)
        with h5py.File(filename, "r") as handle:
            dataset = handle[dataset_path]
            shape = tuple(int(dim) for dim in dataset.shape)
        if len(shape) != 4:
            raise ValueError(
                "HDF5 HFX nu dataset must have shape (n_omega, ngrids, nao, nao), "
                f"got {shape}."
            )

        def grid_chunk(start: int, stop: int) -> np.ndarray:
            with h5py.File(filename, "r") as handle:
                return np.asarray(handle[dataset_path][:, int(start) : int(stop)])

        return cls(
            shape=shape,
            chunk_size=int(chunk_size),
            _grid_chunk_fn=grid_chunk,
        )

    @classmethod
    def from_pyscf_mol(
        cls,
        mol: Any,
        coords: Any,
        *,
        omega_values: tuple[float, ...],
        nao: int,
        chunk_size: int = 512,
    ) -> "ChunkedHFXNu":
        coords_arr = np.asarray(coords)
        omega_values = tuple(float(omega) for omega in omega_values)
        shape = (
            len(omega_values),
            int(coords_arr.shape[0]),
            int(nao),
            int(nao),
        )
        int1e_grids = _int1e_grids_name(mol)
        int1e_rinv = _int1e_rinv_name(mol)

        def grid_chunk(start: int, stop: int) -> np.ndarray:
            start_i = int(start)
            stop_i = int(stop)
            coords_chunk = coords_arr[start_i:stop_i]
            chunks = []
            for omega in omega_values:
                try:
                    with mol.with_range_coulomb(omega=float(omega)):
                        nu = mol.intor(int1e_grids, hermi=1, grids=coords_chunk)
                except TypeError:
                    nu_list = []
                    with mol.with_rinv_zeta(zeta=float(omega) * float(omega)):
                        for coord in coords_chunk:
                            with mol.with_rinv_origin(coord):
                                nu_list.append(mol.intor(int1e_rinv, hermi=1))
                    nu = np.asarray(nu_list)
                chunks.append(np.asarray(nu))
            return np.stack(chunks, axis=0)

        return cls(
            shape=shape,
            chunk_size=int(chunk_size),
            _grid_chunk_fn=grid_chunk,
        )

    def grid_chunk(self, start: int, stop: int) -> np.ndarray:
        return np.asarray(self._grid_chunk_fn(int(start), int(stop)))

    def materialize(self) -> np.ndarray:
        chunks = [
            self.grid_chunk(start, min(start + int(self.chunk_size), self.shape[1]))
            for start in range(0, self.shape[1], int(self.chunk_size))
        ]
        if not chunks:
            return np.zeros(self.shape, dtype=np.float64)
        return np.concatenate(chunks, axis=1)


def hfx_nu_source(molecule: Any | None) -> Any | None:
    if molecule is None:
        return None
    nu = getattr(molecule, "hfx_nu", None)
    if nu is not None:
        return nu
    return getattr(molecule, "hfx_nu_api", None)


def has_hfx_nu_source(molecule: Any | None) -> bool:
    return hfx_nu_source(molecule) is not None


def is_chunked_hfx_nu(source: Any | None) -> bool:
    return source is not None and callable(getattr(source, "grid_chunk", None))


def hfx_nu_shape(source: Any) -> tuple[int, int, int, int]:
    shape = getattr(source, "shape", None)
    if shape is None:
        shape = jnp.asarray(source).shape
    if len(shape) != 4:
        raise ValueError(
            "HFX nu source must have shape (n_omega, ngrids, nao, nao), "
            f"got {shape}."
        )
    return tuple(int(dim) for dim in shape)


def hfx_nu_grid_chunk(
    source: Any,
    start: int,
    stop: int,
    *,
    n_omega: int | None = None,
    dtype: Any | None = None,
) -> jnp.ndarray:
    start_i = int(start)
    stop_i = int(stop)
    if is_chunked_hfx_nu(source):
        chunk = source.grid_chunk(start_i, stop_i)
    else:
        chunk = jnp.asarray(source)[:, start_i:stop_i]
    if n_omega is not None:
        chunk = chunk[: int(n_omega)]
    return jnp.asarray(chunk, dtype=dtype)


def hfx_nu_grid_chunk_padded(
    source: Any,
    start: int,
    chunk_size: int,
    *,
    n_omega: int | None = None,
    dtype: Any | None = None,
) -> jnp.ndarray:
    _, ngrids, _, _ = hfx_nu_shape(source)
    start_i = int(start)
    chunk_size_i = int(chunk_size)
    stop_i = min(start_i + chunk_size_i, ngrids)
    chunk = hfx_nu_grid_chunk(
        source,
        start_i,
        stop_i,
        n_omega=n_omega,
        dtype=dtype,
    )
    pad = chunk_size_i - int(chunk.shape[1])
    if pad <= 0:
        return chunk
    return jnp.pad(chunk, ((0, 0), (0, pad), (0, 0), (0, 0)))


def _local_hfx_features_from_dm(
    mol: Any,
    ao: np.ndarray,
    dm_spin: tuple[np.ndarray, np.ndarray],
    coords: np.ndarray,
    *,
    omega_values: tuple[float, ...],
    chunk_size: int = 512,
    return_nu: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute molecule-local HF exchange channels used by neural functionals."""

    dm_a, dm_b = dm_spin
    e_a = ao @ dm_a
    e_b = ao @ dm_b
    ngrid = int(coords.shape[0])
    n_omega = len(omega_values)
    nao = int(ao.shape[1])
    hfx = np.zeros((2, ngrid, n_omega), dtype=np.float64)
    nu_cache = (
        np.zeros((n_omega, ngrid, nao, nao), dtype=np.float64) if return_nu else None
    )
    int1e_grids = _int1e_grids_name(mol)
    int1e_rinv = _int1e_rinv_name(mol)

    for omega_idx, omega in enumerate(omega_values):
        for start in range(0, ngrid, int(chunk_size)):
            end = min(start + int(chunk_size), ngrid)
            coords_chunk = coords[start:end]
            try:
                with mol.with_range_coulomb(omega=float(omega)):
                    nu = mol.intor(int1e_grids, hermi=1, grids=coords_chunk)
            except TypeError:
                nu_list = []
                with mol.with_rinv_zeta(zeta=float(omega) * float(omega)):
                    for coord in coords_chunk:
                        with mol.with_rinv_origin(coord):
                            nu_list.append(mol.intor(int1e_rinv, hermi=1))
                nu = np.asarray(nu_list)
            if nu_cache is not None:
                nu_cache[omega_idx, start:end] = nu

            e_a_chunk = e_a[start:end]
            e_b_chunk = e_b[start:end]
            fxx_a = np.einsum("gbc,gc->gb", nu, e_a_chunk, optimize=True)
            fxx_b = np.einsum("gbc,gc->gb", nu, e_b_chunk, optimize=True)
            hfx[0, start:end, omega_idx] = -0.5 * np.einsum(
                "gb,gb->g", e_a_chunk, fxx_a, optimize=True
            )
            hfx[1, start:end, omega_idx] = -0.5 * np.einsum(
                "gb,gb->g", e_b_chunk, fxx_b, optimize=True
            )
    if nu_cache is None:
        return hfx
    return hfx, nu_cache


def _local_hfx_features_from_nu_cache(
    ao: Any,
    dm_spin: tuple[Any, Any],
    nu_cache: Any,
) -> jnp.ndarray:
    ao_arr = jnp.asarray(ao)
    dm_a, dm_b = (jnp.asarray(dm_spin[0]), jnp.asarray(dm_spin[1]))
    nu = jnp.asarray(nu_cache)

    e_a = jnp.einsum("gp,pq->gq", ao_arr, dm_a, precision=Precision.HIGHEST)
    e_b = jnp.einsum("gp,pq->gq", ao_arr, dm_b, precision=Precision.HIGHEST)
    fxx_a = jnp.einsum("wgbc,gc->wgb", nu, e_a, precision=Precision.HIGHEST)
    fxx_b = jnp.einsum("wgbc,gc->wgb", nu, e_b, precision=Precision.HIGHEST)
    exx_a = -0.5 * jnp.einsum("gq,wgq->wg", e_a, fxx_a, precision=Precision.HIGHEST)
    exx_b = -0.5 * jnp.einsum("gq,wgq->wg", e_b, fxx_b, precision=Precision.HIGHEST)
    exx = jnp.stack([exx_a.T, exx_b.T], axis=0)
    return jnp.nan_to_num(exx, nan=0.0, posinf=0.0, neginf=0.0)


def _local_hfx_features_from_basis_dm(
    basis: Any,
    ao: Any,
    dm_spin: tuple[Any, Any],
    coords: Any,
    *,
    omega_values: tuple[float, ...],
    chunk_size: int = 512,
    return_nu: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    coords_arr = jnp.asarray(coords)
    ao_arr = jnp.asarray(ao)
    ngrid = int(coords_arr.shape[0])
    hfx_chunks: list[jnp.ndarray] = []
    nu_chunks_per_omega: list[jnp.ndarray] = []

    for omega in omega_values:
        zeta = None if abs(float(omega)) < 1e-14 else float(omega) * float(omega)
        omega_nu_chunks: list[jnp.ndarray] = []
        omega_hfx_chunks: list[jnp.ndarray] = []
        for start in range(0, ngrid, int(chunk_size)):
            end = min(start + int(chunk_size), ngrid)
            nu_chunk = rinv_matrices(
                basis,
                coords_arr[start:end],
                zeta=zeta,
                engine="auto",
                grid_chunk_size=min(int(chunk_size), max(1, end - start)),
            )
            omega_hfx_chunks.append(
                _local_hfx_features_from_nu_cache(
                    ao_arr[start:end],
                    dm_spin,
                    nu_chunk[None, ...],
                )[:, :, 0]
            )
            if return_nu:
                omega_nu_chunks.append(nu_chunk)
        hfx_chunks.append(jnp.concatenate(omega_hfx_chunks, axis=1))
        if return_nu:
            nu_chunks_per_omega.append(jnp.concatenate(omega_nu_chunks, axis=0))

    hfx_local = jnp.stack(hfx_chunks, axis=-1)
    if not return_nu:
        return hfx_local
    nu_cache = jnp.stack(nu_chunks_per_omega, axis=0)
    return hfx_local, nu_cache


def _local_pt2_feature_from_restricted_orbitals(
    ao: Any,
    mo_coeff: Any,
    mo_occ: Any,
    mo_energy: Any,
    *,
    rep_tensor: Any | None = None,
    eri_ovov: Any | None = None,
    eri_pair_matrix: Any | None = None,
    df_factors: Any | None = None,
    nocc: int | None = None,
    occupation_tolerance: float = 1e-8,
    density_floor: float = 1e-12,
) -> jnp.ndarray:
    ao_arr = jnp.asarray(ao)
    mo_coeff_arr = jnp.asarray(mo_coeff)
    mo_occ_arr = jnp.asarray(mo_occ)
    mo_energy_arr = jnp.asarray(mo_energy)

    if mo_coeff_arr.ndim == 3:
        mo_coeff_arr = mo_coeff_arr[0]
    if mo_occ_arr.ndim == 2:
        mo_occ_arr = mo_occ_arr[0]
    if mo_energy_arr.ndim == 2:
        mo_energy_arr = mo_energy_arr[0]

    nocc_int = int(nocc) if nocc is not None else int(jnp.count_nonzero(mo_occ_arr > occupation_tolerance))
    nmo = int(mo_coeff_arr.shape[1])
    if nocc_int <= 0 or nocc_int >= nmo:
        raise ValueError("PT2 local feature requires at least one occupied and one virtual orbital.")

    orbo = mo_coeff_arr[:, :nocc_int]
    orbv = mo_coeff_arr[:, nocc_int:]
    eps_occ = mo_energy_arr[:nocc_int]
    eps_vir = mo_energy_arr[nocc_int:]

    eri_ovov_arr = None if eri_ovov is None else jnp.asarray(eri_ovov)
    if eri_ovov_arr is None:
        if df_factors is not None:
            factors = jnp.asarray(df_factors)
            if factors.size != 0:
                eri_ovov_arr, _, _ = df_factors_to_mo_eri_slices(
                    factors,
                    mo_coeff_arr,
                    nocc_int,
                    include_oovv=False,
                )
        if eri_ovov_arr is None and eri_pair_matrix is not None:
            pair = jnp.asarray(eri_pair_matrix)
            if pair.size != 0:
                eri_ovov_arr, _, _ = eri_pair_matrix_to_mo_eri_slices(
                    pair,
                    mo_coeff_arr,
                    nocc=nocc_int,
                    include_oovv=False,
                )
        if eri_ovov_arr is None:
            if rep_tensor is None:
                raise ValueError(
                    "PT2 local feature requires rep_tensor, eri_ovov, eri_pair_matrix, "
                    "or df_factors."
                )
            rep = jnp.asarray(rep_tensor)
            if rep.size == 0:
                raise ValueError(
                    "PT2 local feature cannot be constructed from an empty rep_tensor without df_factors."
                )
            eri_ovov_arr = jnp.einsum(
                "pqrs,pi,qa,rj,sb->iajb",
                rep,
                orbo,
                orbv,
                orbo,
                orbv,
                precision=Precision.HIGHEST,
            )

    denom = (
        eps_occ[:, None, None, None]
        + eps_occ[None, None, :, None]
        - eps_vir[None, :, None, None]
        - eps_vir[None, None, None, :]
    )
    denom = jnp.where(jnp.abs(denom) > density_floor, denom, -density_floor)
    direct = eri_ovov_arr
    exchange = jnp.transpose(eri_ovov_arr, (0, 3, 2, 1))
    pair_weights = (2.0 * direct - exchange) / denom

    rho_o = jnp.einsum("rp,pi->ri", ao_arr, orbo, precision=Precision.HIGHEST)
    rho_v = jnp.einsum("rp,pa->ra", ao_arr, orbv, precision=Precision.HIGHEST)
    rho_ov = jnp.einsum("ri,ra->ria", rho_o, rho_v, precision=Precision.HIGHEST)

    if df_factors is not None and jnp.asarray(df_factors).size != 0:
        factors = jnp.asarray(df_factors)
        grid_aux = jnp.einsum(
            "Qpq,gp,gq->gQ",
            factors,
            ao_arr,
            ao_arr,
            precision=Precision.HIGHEST,
        )
        qjb = jnp.einsum(
            "Qrs,rj,sb->Qjb",
            factors,
            orbo,
            orbv,
            precision=Precision.HIGHEST,
        )
        pair_potential = jnp.einsum(
            "gQ,Qjb->gjb",
            grid_aux,
            qjb,
            precision=Precision.HIGHEST,
        )
    elif eri_pair_matrix is not None and jnp.asarray(eri_pair_matrix).size != 0:
        pair = jnp.asarray(eri_pair_matrix)
        rows, cols, _, multiplicity = _metadata_arrays(int(mo_coeff_arr.shape[0]), ao_arr.dtype)
        grid_pair = ao_arr[:, rows] * ao_arr[:, cols] * multiplicity[None, :]
        ov = _mo_pair_products(orbo, orbv, rows, cols)
        pair_potential = jnp.einsum(
            "gP,PQ,jbQ->gjb",
            grid_pair,
            pair,
            ov,
            precision=Precision.HIGHEST,
        )
    else:
        rep = jnp.asarray(rep_tensor)
        pair_potential = jnp.einsum(
            "gp,gq,pqrs,rj,sb->gjb",
            ao_arr,
            ao_arr,
            rep,
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
    return jnp.nan_to_num(local_energy, nan=0.0, posinf=0.0, neginf=0.0)


def _local_pt2_feature_from_unrestricted_orbitals(
    ao: Any,
    mo_coeff: Any,
    mo_occ: Any,
    mo_energy: Any,
    *,
    rep_tensor: Any | None = None,
    eri_pair_matrix: Any | None = None,
    df_factors: Any | None = None,
    occupation_tolerance: float = 1e-8,
    density_floor: float = 1e-12,
    return_total_energy: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    ao_arr = jnp.asarray(ao)
    mo_coeff_arr = jnp.asarray(mo_coeff)
    mo_occ_arr = jnp.asarray(mo_occ)
    mo_energy_arr = jnp.asarray(mo_energy)

    if mo_coeff_arr.ndim != 3 or mo_coeff_arr.shape[0] != 2:
        raise ValueError(
            "Unrestricted PT2 local feature expects mo_coeff with shape (2, nao, nmo)."
        )
    if mo_occ_arr.ndim != 2 or mo_occ_arr.shape[0] != 2:
        raise ValueError(
            "Unrestricted PT2 local feature expects mo_occ with shape (2, nmo)."
        )
    if mo_energy_arr.ndim != 2 or mo_energy_arr.shape[0] != 2:
        raise ValueError(
            "Unrestricted PT2 local feature expects mo_energy with shape (2, nmo)."
        )

    occ_a = jnp.where(mo_occ_arr[0] > occupation_tolerance)[0]
    vir_a = jnp.where(mo_occ_arr[0] <= occupation_tolerance)[0]
    occ_b = jnp.where(mo_occ_arr[1] > occupation_tolerance)[0]
    vir_b = jnp.where(mo_occ_arr[1] <= occupation_tolerance)[0]

    occ_coeff = jnp.concatenate(
        [mo_coeff_arr[0][:, occ_a], mo_coeff_arr[1][:, occ_b]],
        axis=1,
    )
    vir_coeff = jnp.concatenate(
        [mo_coeff_arr[0][:, vir_a], mo_coeff_arr[1][:, vir_b]],
        axis=1,
    )
    occ_spin = jnp.concatenate(
        [
            jnp.zeros((occ_a.shape[0],), dtype=jnp.int32),
            jnp.ones((occ_b.shape[0],), dtype=jnp.int32),
        ]
    )
    vir_spin = jnp.concatenate(
        [
            jnp.zeros((vir_a.shape[0],), dtype=jnp.int32),
            jnp.ones((vir_b.shape[0],), dtype=jnp.int32),
        ]
    )
    eps_occ = jnp.concatenate([mo_energy_arr[0][occ_a], mo_energy_arr[1][occ_b]])
    eps_vir = jnp.concatenate([mo_energy_arr[0][vir_a], mo_energy_arr[1][vir_b]])

    nocc_total = int(occ_coeff.shape[1])
    nvir_total = int(vir_coeff.shape[1])
    zero_local = jnp.zeros((int(ao_arr.shape[0]),), dtype=ao_arr.dtype)
    zero_total = jnp.asarray(0.0, dtype=ao_arr.dtype)
    if nocc_total < 2 or nvir_total < 2:
        if return_total_energy:
            return zero_local, zero_total
        return zero_local

    pair = None if eri_pair_matrix is None else jnp.asarray(eri_pair_matrix)
    factors = None if df_factors is None else jnp.asarray(df_factors)
    rep = None if rep_tensor is None else jnp.asarray(rep_tensor)

    if factors is not None and factors.size != 0:
        b_ov = jnp.einsum(
            "Qpq,pi,qa->Qia",
            factors,
            occ_coeff,
            vir_coeff,
            precision=Precision.HIGHEST,
        )
        direct_spatial = jnp.einsum("Qia,Qjb->iajb", b_ov, b_ov, precision=Precision.HIGHEST)
    elif pair is not None and pair.size != 0:
        rows, cols, _, _ = _metadata_arrays(int(mo_coeff_arr.shape[1]), ao_arr.dtype)
        ov = _mo_pair_products(occ_coeff, vir_coeff, rows, cols)
        direct_spatial = jnp.einsum(
            "iaP,PQ,jbQ->iajb",
            ov,
            pair,
            ov,
            precision=Precision.HIGHEST,
        )
    else:
        if rep is None or rep.size == 0:
            raise ValueError(
                "Unrestricted PT2 local feature requires rep_tensor, eri_pair_matrix, or df_factors."
            )
        direct_spatial = jnp.einsum(
            "pqrs,pi,qa,rj,sb->iajb",
            rep,
            occ_coeff,
            vir_coeff,
            occ_coeff,
            vir_coeff,
            precision=Precision.HIGHEST,
        )

    exchange_spatial = jnp.transpose(direct_spatial, (0, 3, 2, 1))
    mask_direct = (
        (occ_spin[:, None, None, None] == vir_spin[None, :, None, None])
        & (occ_spin[None, None, :, None] == vir_spin[None, None, None, :])
    )
    mask_exchange = (
        (occ_spin[:, None, None, None] == vir_spin[None, None, None, :])
        & (occ_spin[None, None, :, None] == vir_spin[None, :, None, None])
    )
    direct = direct_spatial * mask_direct.astype(direct_spatial.dtype)
    exchange = exchange_spatial * mask_exchange.astype(exchange_spatial.dtype)

    denom = (
        eps_occ[:, None, None, None]
        + eps_occ[None, None, :, None]
        - eps_vir[None, :, None, None]
        - eps_vir[None, None, None, :]
    )
    denom = jnp.where(jnp.abs(denom) > density_floor, denom, -density_floor)
    amplitudes = (direct - exchange) / denom
    pair_weights = 0.5 * amplitudes
    total_energy = jnp.sum(direct * pair_weights)

    rho_o = jnp.einsum("rp,pi->ri", ao_arr, occ_coeff, precision=Precision.HIGHEST)
    rho_v = jnp.einsum("rp,pa->ra", ao_arr, vir_coeff, precision=Precision.HIGHEST)
    rho_ov = jnp.einsum("ri,ra->ria", rho_o, rho_v, precision=Precision.HIGHEST)

    if factors is not None and factors.size != 0:
        grid_aux = jnp.einsum(
            "Qpq,gp,gq->gQ",
            factors,
            ao_arr,
            ao_arr,
            precision=Precision.HIGHEST,
        )
        qjb = jnp.einsum(
            "Qrs,rj,sb->Qjb",
            factors,
            occ_coeff,
            vir_coeff,
            precision=Precision.HIGHEST,
        )
        pair_potential = jnp.einsum(
            "gQ,Qjb->gjb",
            grid_aux,
            qjb,
            precision=Precision.HIGHEST,
        )
    elif pair is not None and pair.size != 0:
        rows, cols, _, multiplicity = _metadata_arrays(int(mo_coeff_arr.shape[1]), ao_arr.dtype)
        grid_pair = ao_arr[:, rows] * ao_arr[:, cols] * multiplicity[None, :]
        ov = _mo_pair_products(occ_coeff, vir_coeff, rows, cols)
        pair_potential = jnp.einsum(
            "gP,PQ,jbQ->gjb",
            grid_pair,
            pair,
            ov,
            precision=Precision.HIGHEST,
        )
    else:
        pair_potential = jnp.einsum(
            "gp,gq,pqrs,rj,sb->gjb",
            ao_arr,
            ao_arr,
            rep,
            occ_coeff,
            vir_coeff,
            precision=Precision.HIGHEST,
        )

    local_energy = jnp.einsum(
        "ria,rjb,iajb->r",
        rho_ov,
        pair_potential,
        pair_weights,
        precision=Precision.HIGHEST,
    )
    local_energy = jnp.nan_to_num(local_energy, nan=0.0, posinf=0.0, neginf=0.0)
    total_energy = jnp.nan_to_num(total_energy, nan=0.0, posinf=0.0, neginf=0.0)
    if return_total_energy:
        return local_energy, total_energy
    return local_energy


__all__ = [
    "ChunkedHFXNu",
    "canonical_input_features",
    "enhanced_input_features",
    "has_hfx_nu_source",
    "hfx_nu_grid_chunk",
    "hfx_nu_grid_chunk_padded",
    "hfx_nu_shape",
    "hfx_nu_source",
    "is_chunked_hfx_nu",
    "_local_hfx_features_from_basis_dm",
    "_local_hfx_features_from_dm",
    "_local_pt2_feature_from_restricted_orbitals",
    "_local_pt2_feature_from_unrestricted_orbitals",
]
