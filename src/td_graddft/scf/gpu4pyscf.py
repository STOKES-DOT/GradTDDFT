from __future__ import annotations
import types
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..data.molecule import MoleculeSpec


GPU4PYSCF_RKS_RUNTIME_BACKEND = "gpu4pyscf_rks"
GPU4PYSCF_UKS_RUNTIME_BACKEND = "gpu4pyscf_uks"
_GPU4PYSCF_CUSTOM_XC_PLACEHOLDER = "pbe"
_NEURAL_XC_PAYLOAD_JIT_CACHE_MAXSIZE = 32
_NEURAL_XC_PAYLOAD_JIT_CACHE: dict[tuple[Any, ...], Any] = {}
_NEURAL_XC_UKS_PAYLOAD_JIT_CACHE: dict[tuple[Any, ...], Any] = {}
_DIRECT_JK_ENGINE_CACHE_MAXSIZE = 16
_DIRECT_JK_ENGINE_CACHE: dict[tuple[Any, ...], tuple[Any, Any]] = {}


@dataclass(frozen=True)
class GPU4PySCFRKSForwardResult:
    """Host-side arrays returned by an exact GPU4PySCF RKS forward solve."""

    converged: bool
    total_energy: float
    mo_energy: np.ndarray
    mo_coeff: np.ndarray
    mo_occ: np.ndarray
    density_matrix: np.ndarray
    fock_matrix: np.ndarray | None = None
    cycles: int | None = None
    exact_exchange_fraction: float | None = None


@dataclass(frozen=True)
class GPU4PySCFUKSForwardResult:
    """Host-side arrays returned by an exact GPU4PySCF UKS forward solve."""

    converged: bool
    total_energy: float
    mo_energy: np.ndarray
    mo_coeff: np.ndarray
    mo_occ: np.ndarray
    density_matrix: np.ndarray
    fock_matrix: np.ndarray | None = None
    cycles: int | None = None

    @property
    def mo_energy_alpha(self) -> np.ndarray:
        return self.mo_energy[0]

    @property
    def mo_energy_beta(self) -> np.ndarray:
        return self.mo_energy[1]

    @property
    def mo_coeff_alpha(self) -> np.ndarray:
        return self.mo_coeff[0]

    @property
    def mo_coeff_beta(self) -> np.ndarray:
        return self.mo_coeff[1]

    @property
    def mo_occ_alpha(self) -> np.ndarray:
        return self.mo_occ[0]

    @property
    def mo_occ_beta(self) -> np.ndarray:
        return self.mo_occ[1]

    @property
    def density_matrix_alpha(self) -> np.ndarray:
        return self.density_matrix[0]

    @property
    def density_matrix_beta(self) -> np.ndarray:
        return self.density_matrix[1]


@dataclass(frozen=True)
class GPU4PySCFRKSForwardOptions:
    """Replayable GPU4PySCF RKS options stored on GPU4PySCF-backed molecules."""

    atom: Any
    basis: Any
    xc_spec: str = "pbe"
    unit: str = "Angstrom"
    charge: int = 0
    spin: int = 0
    cart: bool = True
    grids_level: int = 0
    conv_tol: float = 1e-10
    max_cycle: int = 80
    verbose: int = 0
    mol_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def as_kwargs(self) -> dict[str, Any]:
        kwargs = dict(self.mol_kwargs)
        kwargs.update(
            atom=self.atom,
            basis=self.basis,
            xc_spec=self.xc_spec,
            unit=self.unit,
            charge=self.charge,
            spin=self.spin,
            cart=self.cart,
            grids_level=self.grids_level,
            conv_tol=self.conv_tol,
            max_cycle=self.max_cycle,
            verbose=self.verbose,
        )
        return kwargs


@dataclass(frozen=True)
class GPU4PySCFUKSForwardOptions(GPU4PySCFRKSForwardOptions):
    """Replayable GPU4PySCF UKS options stored on GPU4PySCF-backed molecules."""


def _gpu4pyscf_engine_xc_spec(xc_spec: str, *, has_custom_xc: bool) -> str:
    if bool(has_custom_xc):
        return _GPU4PYSCF_CUSTOM_XC_PLACEHOLDER
    return str(xc_spec)


class _TaggedArray(np.ndarray):
    pass


def _to_host_numpy(value: Any, dtype: Any | None = None) -> np.ndarray:
    try:
        import cupy as cp
    except ModuleNotFoundError:
        cp = None
    if cp is not None and isinstance(value, cp.ndarray):
        return np.asarray(cp.asnumpy(value), dtype=dtype)
    return np.asarray(value, dtype=dtype)


def _spin_stack_host_numpy(value: Any, dtype: Any | None = None) -> np.ndarray:
    if isinstance(value, (tuple, list)):
        return np.stack([_to_host_numpy(block, dtype=dtype) for block in value], axis=0)
    arr = _to_host_numpy(value, dtype=dtype)
    if arr.ndim >= 1 and int(arr.shape[0]) == 2:
        return arr
    raise ValueError(
        "Expected an unrestricted spin-block array/list with leading spin dimension 2, "
        f"got shape {arr.shape}."
    )


def _cupy_module():
    try:
        import cupy as cp
    except ModuleNotFoundError:
        return None
    runtime = getattr(getattr(cp, "cuda", None), "runtime", None)
    get_device_count = getattr(runtime, "getDeviceCount", None)
    if callable(get_device_count):
        try:
            if int(get_device_count()) <= 0:
                return None
        except Exception:
            return None
    return cp


def _to_jax_array(value: Any, dtype: Any | None = None) -> jnp.ndarray:
    cp = _cupy_module()
    if cp is not None and isinstance(value, cp.ndarray):
        try:
            out = jax.dlpack.from_dlpack(value)
        except Exception:
            out = jnp.asarray(cp.asnumpy(value))
    else:
        out = jnp.asarray(value)
    if dtype is not None:
        out = out.astype(dtype)
    return out


def _to_backend_array(value: Any, *, like: Any) -> Any:
    cp = _cupy_module()
    if cp is not None and isinstance(like, cp.ndarray):
        if isinstance(value, cp.ndarray):
            return value
        arr = jnp.asarray(value)
        try:
            return cp.from_dlpack(arr)
        except Exception:
            try:
                return cp.fromDlpack(jax.dlpack.to_dlpack(arr))
            except Exception:
                return cp.asarray(np.asarray(jax.device_get(arr)))
    return np.asarray(jax.device_get(value))


def _to_cupy_or_numpy_array(value: Any) -> Any:
    cp = _cupy_module()
    if cp is None:
        return _to_host_numpy(value)
    if isinstance(value, cp.ndarray):
        return value
    if isinstance(value, np.ndarray):
        return cp.asarray(value)
    arr = jnp.asarray(value)
    try:
        return cp.from_dlpack(arr)
    except Exception:
        try:
            return cp.fromDlpack(jax.dlpack.to_dlpack(arr))
        except Exception:
            return cp.asarray(np.asarray(jax.device_get(arr)))


def _backend_scalar(value: Any) -> float:
    if hasattr(value, "get"):
        value = value.get()
    return float(np.asarray(value).real)


def _backend_dm_dot(dm: Any, mat: Any) -> float:
    cp = _cupy_module()
    if cp is not None and (isinstance(dm, cp.ndarray) or isinstance(mat, cp.ndarray)):
        return _backend_scalar(cp.einsum("ij,ij", dm, mat))
    return _backend_scalar(np.einsum("ij,ij", np.asarray(dm), np.asarray(mat)))


def _tag_array(value: Any, **kwargs: Any) -> Any:
    try:
        from gpu4pyscf.lib.cupy_helper import tag_array

        return tag_array(value, **kwargs)
    except Exception:
        tagged = np.asarray(value).view(_TaggedArray)
        tagged.__dict__.update(kwargs)
        return tagged


def _sync_gpu4pyscf() -> None:
    cp = _cupy_module()
    if cp is None:
        return
    try:
        cp.cuda.Stream.null.synchronize()
    except Exception:
        return


def _pyscf_atom_and_unit(atom: Any, unit: str) -> tuple[Any, str]:
    if not isinstance(atom, MoleculeSpec):
        return atom, unit
    coords_bohr = _to_host_numpy(atom.coords_bohr, dtype=float)
    records = [
        (symbol, tuple(float(item) for item in coords))
        for symbol, coords in zip(atom.symbols, coords_bohr, strict=True)
    ]
    return records, "Bohr"


def _cache_token(value: Any) -> Any:
    if isinstance(value, MoleculeSpec):
        return (
            "MoleculeSpec",
            value.symbols,
            _cache_token(_to_host_numpy(value.coords_bohr, dtype=float)),
            int(value.charge),
            int(value.spin),
        )
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _cache_token(v)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_cache_token(v) for v in value)
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        return ("array", arr.shape, str(arr.dtype), tuple(arr.reshape(-1).tolist()))
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    try:
        hash(value)
        return value
    except Exception:
        try:
            arr = np.asarray(value)
        except Exception:
            return repr(value)
        return ("array", arr.shape, str(arr.dtype), tuple(arr.reshape(-1).tolist()))


def compute_gpu4pyscf_local_hfx_features(
    *,
    atom: Any,
    basis: Any,
    unit: str,
    charge: int,
    spin: int,
    cart: bool,
    coords: Any,
    ao: Any,
    dm_spin: tuple[Any, Any],
    omega_values: tuple[float, ...],
    chunk_size: int = 512,
    return_nu: bool = False,
    direct_scf_tol: float = 1e-13,
    verbose: int = 0,
    **mol_kwargs: Any,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """Build Neural XC local-HFX grid features with GPU4PySCF int1e_grids.

    This computes the same grid-centered Coulomb matrices used by the JAX
    ``rinv_matrices`` path, but delegates the AO integral generation to
    GPU4PySCF's ``gpu4pyscf.gto.int3c1e.int1e_grids`` kernel.  When
    ``return_nu=True`` the returned aux tensor has shape
    ``(n_omega, ngrids, nao, nao)`` and remains compatible with the existing
    implicit-backward and local-HFX Fock-correction code.
    """

    try:
        from gpu4pyscf.gto.int3c1e import VHFOpt, int1e_grids
        from pyscf import gto
    except ModuleNotFoundError as exc:
        raise ImportError(
            "gpu4pyscf and pyscf are required for GPU4PySCF local-HFX aux construction."
        ) from exc

    if isinstance(atom, MoleculeSpec):
        charge = int(atom.charge)
        spin = int(atom.spin)
    if int(spin) != 0:
        raise NotImplementedError("GPU4PySCF local-HFX aux currently supports spin=0 only.")

    pyscf_atom, pyscf_unit = _pyscf_atom_and_unit(atom, unit)
    mol = gto.M(
        atom=pyscf_atom,
        basis=basis,
        unit=pyscf_unit,
        spin=int(spin),
        charge=int(charge),
        cart=bool(cart),
        verbose=int(verbose),
        **mol_kwargs,
    )
    coords_np = _to_host_numpy(coords, dtype=np.float64)
    ao_np = _to_host_numpy(ao, dtype=np.float64)
    dm_a = _to_host_numpy(dm_spin[0], dtype=np.float64)
    dm_b = _to_host_numpy(dm_spin[1], dtype=np.float64)
    ngrid = int(coords_np.shape[0])
    nao = int(ao_np.shape[1])
    n_omega = len(tuple(omega_values))
    hfx = np.zeros((2, ngrid, n_omega), dtype=np.float64)
    nu_cache = (
        np.empty((n_omega, ngrid, nao, nao), dtype=np.float64)
        if return_nu
        else None
    )

    intopt = VHFOpt(mol)
    intopt.build(float(direct_scf_tol), aosym=True)
    step = ngrid if int(chunk_size) <= 0 else int(chunk_size)
    e_a = ao_np @ dm_a
    e_b = ao_np @ dm_b
    for omega_idx, omega in enumerate(tuple(float(value) for value in omega_values)):
        with mol.with_range_coulomb(omega=omega):
            for start in range(0, ngrid, step):
                end = min(start + step, ngrid)
                nu = np.asarray(
                    int1e_grids(
                        mol,
                        coords_np[start:end],
                        direct_scf_tol=float(direct_scf_tol),
                        intopt=intopt,
                    ),
                    dtype=np.float64,
                )
                if nu_cache is not None:
                    nu_cache[omega_idx, start:end] = nu
                e_a_chunk = e_a[start:end]
                e_b_chunk = e_b[start:end]
                fxx_a = np.einsum("gbc,gc->gb", nu, e_a_chunk, optimize=True)
                fxx_b = np.einsum("gbc,gc->gb", nu, e_b_chunk, optimize=True)
                hfx[0, start:end, omega_idx] = -0.5 * np.einsum(
                    "gb,gb->g",
                    e_a_chunk,
                    fxx_a,
                    optimize=True,
                )
                hfx[1, start:end, omega_idx] = -0.5 * np.einsum(
                    "gb,gb->g",
                    e_b_chunk,
                    fxx_b,
                    optimize=True,
                )
    if nu_cache is None:
        return jnp.asarray(hfx)
    return jnp.asarray(hfx), jnp.asarray(nu_cache)


def compute_gpu4pyscf_direct_jk_response(
    *,
    atom: Any,
    basis: Any,
    delta_density: Any,
    xc_spec: str = "pbe",
    unit: str = "Angstrom",
    charge: int = 0,
    spin: int = 0,
    cart: bool = True,
    grids_level: int = 0,
    verbose: int = 0,
    with_k: bool = True,
    **mol_kwargs: Any,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute exact/direct GPU4PySCF J/K response matrices for ``delta_density``."""

    try:
        import gpu4pyscf  # noqa: F401
        from pyscf import dft, gto
    except ModuleNotFoundError as exc:
        raise ImportError(
            "gpu4pyscf and pyscf are required for GPU4PySCF direct JK response."
        ) from exc

    if isinstance(atom, MoleculeSpec):
        charge = int(atom.charge)
        spin = int(atom.spin)
    if int(spin) != 0:
        raise NotImplementedError("GPU4PySCF direct JK response supports closed-shell spin=0 only.")
    if not bool(cart):
        raise NotImplementedError("GPU4PySCF direct JK response currently supports cart=True only.")

    pyscf_atom, pyscf_unit = _pyscf_atom_and_unit(atom, unit)
    cache_key = (
        id(gto.M),
        id(dft.RKS),
        _cache_token(pyscf_atom),
        _cache_token(basis),
        str(xc_spec),
        str(pyscf_unit),
        int(charge),
        int(spin),
        bool(cart),
        int(grids_level),
        int(verbose),
        _cache_token(mol_kwargs),
    )
    cached = _DIRECT_JK_ENGINE_CACHE.get(cache_key)
    if cached is None:
        mol = gto.M(
            atom=pyscf_atom,
            basis=basis,
            unit=pyscf_unit,
            spin=int(spin),
            charge=int(charge),
            cart=bool(cart),
            verbose=int(verbose),
            **mol_kwargs,
        )
        mf = dft.RKS(mol)
        mf.xc = str(xc_spec)
        if hasattr(mf, "grids"):
            mf.grids.level = int(grids_level)
        if not hasattr(mf, "to_gpu"):
            raise RuntimeError(
                "PySCF RKS object does not expose to_gpu(). PySCF >= 2.5 with GPU4PySCF is required."
            )
        mf_gpu = mf.to_gpu()
        cached = (mf_gpu, getattr(mf_gpu, "mol", mol))
        if len(_DIRECT_JK_ENGINE_CACHE) >= _DIRECT_JK_ENGINE_CACHE_MAXSIZE:
            _DIRECT_JK_ENGINE_CACHE.pop(next(iter(_DIRECT_JK_ENGINE_CACHE)))
        _DIRECT_JK_ENGINE_CACHE[cache_key] = cached
    mf_gpu, gpu_mol = cached
    dm = jnp.asarray(delta_density)
    dm_host = _to_host_numpy(delta_density, dtype=np.float64)
    if dm_host.ndim != 2 or dm_host.shape[0] != dm_host.shape[1]:
        raise ValueError(
            "delta_density must have shape (nao, nao), "
            f"got {dm_host.shape}."
        )
    mol_nao = int(getattr(gpu_mol, "nao", dm_host.shape[0]))
    if dm_host.shape != (mol_nao, mol_nao):
        raise ValueError(
            "delta_density AO shape must match the GPU4PySCF molecule "
            f"({dm_host.shape} vs {(mol_nao, mol_nao)})."
        )

    dm_backend = _to_cupy_or_numpy_array(dm_host)
    try:
        get_jk = getattr(mf_gpu, "get_jk", None)
        if not bool(with_k):
            get_j = getattr(mf_gpu, "get_j", None)
            if get_j is None:
                if get_jk is None:
                    raise AttributeError("GPU4PySCF object exposes neither get_j nor get_jk.")
                vj, _ = get_jk(gpu_mol, dm_backend, hermi=1)
            else:
                vj = get_j(gpu_mol, dm_backend, hermi=1)
            vk = _to_backend_array(np.zeros_like(dm_host), like=vj)
        elif get_jk is None:
            vj = mf_gpu.get_j(gpu_mol, dm_backend, hermi=1)
            vk = mf_gpu.get_k(gpu_mol, dm_backend, hermi=1)
        else:
            vj, vk = get_jk(gpu_mol, dm_backend, hermi=1)
    except Exception as exc:
        raise RuntimeError("GPU4PySCF direct JK response failed.") from exc
    _sync_gpu4pyscf()
    return (
        _to_jax_array(vj, dtype=dm.dtype),
        _to_jax_array(vk, dtype=dm.dtype),
    )


def compute_gpu4pyscf_direct_jk_response_from_options(
    options: GPU4PySCFRKSForwardOptions,
    delta_density: Any,
    *,
    with_k: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return compute_gpu4pyscf_direct_jk_response(
        atom=options.atom,
        basis=options.basis,
        delta_density=delta_density,
        xc_spec=options.xc_spec,
        unit=options.unit,
        charge=options.charge,
        spin=options.spin,
        cart=options.cart,
        grids_level=options.grids_level,
        verbose=options.verbose,
        with_k=with_k,
        **dict(options.mol_kwargs),
    )


def _restricted_spin_mo_from_attr(
    value: Any,
    *,
    template_value: Any,
    dtype: Any,
    spin_rank: int,
) -> jnp.ndarray:
    raw = template_value if value is None else value
    arr = _to_jax_array(raw, dtype=dtype)
    if arr.ndim == spin_rank and int(arr.shape[0]) == 2:
        return arr
    return jnp.stack([arr, arr], axis=0)


def _restricted_spin_occ_from_attr(
    value: Any,
    *,
    molecule_template: Any,
    dtype: Any,
) -> jnp.ndarray:
    if value is None:
        occ = jnp.asarray(molecule_template.mo_occ, dtype=dtype)
        if occ.ndim == 2 and int(occ.shape[0]) == 2:
            return occ
        return jnp.stack([0.5 * occ, 0.5 * occ], axis=0)
    total = _to_jax_array(value, dtype=dtype)
    if total.ndim == 2 and int(total.shape[0]) == 2:
        return total
    return jnp.stack([0.5 * total, 0.5 * total], axis=0)


def _unrestricted_spin_array_from_attr(
    value: Any,
    *,
    template_value: Any,
    dtype: Any,
    label: str,
) -> jnp.ndarray:
    raw = template_value if value is None else value
    arr = _to_jax_array(raw, dtype=dtype)
    if arr.ndim < 1 or int(arr.shape[0]) != 2:
        raise ValueError(f"GPU4PySCF UKS {label} must carry alpha/beta spin blocks.")
    return arr


def _neural_xc_energy(
    *,
    xc_functional: Any,
    xc_params: Any,
    molecule_iter: Any,
) -> jnp.ndarray:
    energy_xc_only = getattr(xc_functional, "energy_xc_only", None)
    if callable(energy_xc_only):
        return jnp.asarray(energy_xc_only(xc_params, molecule_iter))
    energy_from_molecule = getattr(xc_functional, "energy_from_molecule", None)
    if callable(energy_from_molecule):
        return jnp.asarray(energy_from_molecule(xc_params, molecule_iter))
    energy = getattr(xc_functional, "energy", None)
    if callable(energy):
        try:
            return jnp.asarray(energy(xc_params, molecule_iter, include_non_xc=False))
        except TypeError:
            density = molecule_iter.density()
            return jnp.asarray(energy(xc_params, density, molecule_iter.grid.weights))
    return jnp.asarray(0.0, dtype=jnp.asarray(molecule_iter.h1e).dtype)


def _neural_xc_fock_payload_core(
    *,
    density_total: Any,
    mo_coeff_spin: Any,
    mo_occ_spin: Any,
    mo_energy_spin: Any,
    molecule_template: Any,
    xc_functional: Any,
    xc_params: Any,
    neural_vxc_clip: float | None,
    compute_exc: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    from .differentiable import (
        _build_vxc_matrix_from_components,
        _restricted_hfx_features_from_nu,
        _scf_xc_components,
    )

    h1e = jnp.asarray(molecule_template.h1e)
    dtype = h1e.dtype
    density_total = jnp.asarray(density_total, dtype=dtype)
    density_spin = jnp.stack([0.5 * density_total, 0.5 * density_total], axis=0)
    mo_coeff_spin = jnp.asarray(mo_coeff_spin, dtype=dtype)
    mo_occ_spin = jnp.asarray(mo_occ_spin, dtype=dtype)
    mo_energy_spin = jnp.asarray(mo_energy_spin, dtype=dtype)

    updates = dict(
        rdm1=density_spin,
        mo_coeff=mo_coeff_spin,
        mo_occ=mo_occ_spin,
        mo_energy=mo_energy_spin,
    )
    if hasattr(molecule_template, "hfx_local"):
        hfx_local = getattr(molecule_template, "hfx_local", None)
        hfx_nu = getattr(molecule_template, "hfx_nu", None)
        if hfx_local is not None:
            updates["hfx_local"] = jax.lax.stop_gradient(
                jnp.asarray(hfx_local, dtype=dtype)
            )
        elif hfx_nu is not None:
            updates["hfx_local"] = _restricted_hfx_features_from_nu(
                ao=jnp.asarray(molecule_template.ao),
                density=density_total,
                nu_cache=hfx_nu,
            )
    molecule_iter = replace(molecule_template, **updates)
    weights = jnp.asarray(molecule_iter.grid.weights, dtype=dtype)

    (
        vxc_rho,
        vxc_grad,
        vxc_tau,
        vxc_lapl,
        xc_kind,
        alpha,
        vhf_matrix,
    ) = _scf_xc_components(
        xc_params,
        xc_functional,
        molecule_iter,
        functional_dtype=dtype,
    )
    if neural_vxc_clip is not None:
        clip = jnp.asarray(float(neural_vxc_clip), dtype=dtype)
        vxc_rho = jnp.clip(
            jnp.nan_to_num(vxc_rho, nan=0.0, posinf=clip, neginf=-clip),
            -clip,
            clip,
        )
        vxc_grad = jnp.clip(
            jnp.nan_to_num(vxc_grad, nan=0.0, posinf=clip, neginf=-clip),
            -clip,
            clip,
        )
    vxc_matrix = _build_vxc_matrix_from_components(
        molecule=molecule_iter,
        weights=weights,
        v_rho=vxc_rho,
        v_grad=vxc_grad,
        v_tau=vxc_tau,
        v_lapl=vxc_lapl,
        xc_kind=xc_kind,
    )
    alpha = jnp.clip(jnp.nan_to_num(jnp.asarray(alpha, dtype=dtype), nan=0.0), 0.0, 1.0)
    if compute_exc:
        exc = _neural_xc_energy(
            xc_functional=xc_functional,
            xc_params=xc_params,
            molecule_iter=molecule_iter,
        ).astype(dtype)
    else:
        exc = jnp.asarray(0.0, dtype=dtype)
    xc_fock_matrix = vxc_matrix + jnp.asarray(vhf_matrix, dtype=dtype)
    return 0.5 * (xc_fock_matrix + xc_fock_matrix.T), exc, alpha


def _jit_payload_molecule_template(molecule_template: Any) -> Any:
    updates: dict[str, Any] = {}
    for field_name in ("runtime_scf_options",):
        if getattr(molecule_template, field_name, None) is not None:
            updates[field_name] = None
    if not updates:
        return molecule_template
    return replace(molecule_template, **updates)


def _cached_neural_xc_payload_kernel(
    *,
    xc_functional: Any,
    neural_vxc_clip: float | None,
    compute_exc: bool,
):
    key = (
        id(xc_functional),
        None if neural_vxc_clip is None else float(neural_vxc_clip),
        bool(compute_exc),
    )
    cached = _NEURAL_XC_PAYLOAD_JIT_CACHE.get(key)
    if cached is not None:
        return cached

    def kernel(
        molecule_template,
        density_total,
        mo_coeff_spin,
        mo_occ_spin,
        mo_energy_spin,
        xc_params,
    ):
        return _neural_xc_fock_payload_core(
            density_total=density_total,
            mo_coeff_spin=mo_coeff_spin,
            mo_occ_spin=mo_occ_spin,
            mo_energy_spin=mo_energy_spin,
            molecule_template=molecule_template,
            xc_functional=xc_functional,
            xc_params=xc_params,
            neural_vxc_clip=neural_vxc_clip,
            compute_exc=bool(compute_exc),
        )

    compiled = jax.jit(kernel)
    if len(_NEURAL_XC_PAYLOAD_JIT_CACHE) >= _NEURAL_XC_PAYLOAD_JIT_CACHE_MAXSIZE:
        _NEURAL_XC_PAYLOAD_JIT_CACHE.pop(next(iter(_NEURAL_XC_PAYLOAD_JIT_CACHE)))
    _NEURAL_XC_PAYLOAD_JIT_CACHE[key] = compiled
    return compiled


def _neural_xc_fock_payload(
    *,
    mf_gpu: Any,
    dm: Any,
    molecule_template: Any,
    xc_functional: Any,
    xc_params: Any,
    neural_vxc_clip: float | None,
    compute_exc: bool = True,
    jit_payload: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    h1e = jnp.asarray(molecule_template.h1e)
    dtype = h1e.dtype
    density_total = _to_jax_array(dm, dtype=dtype)
    mo_coeff_spin = _restricted_spin_mo_from_attr(
        getattr(mf_gpu, "mo_coeff", None),
        template_value=molecule_template.mo_coeff,
        dtype=dtype,
        spin_rank=3,
    )
    mo_occ_spin = _restricted_spin_occ_from_attr(
        getattr(mf_gpu, "mo_occ", None),
        molecule_template=molecule_template,
        dtype=dtype,
    )
    mo_energy_spin = _restricted_spin_mo_from_attr(
        getattr(mf_gpu, "mo_energy", None),
        template_value=molecule_template.mo_energy,
        dtype=dtype,
        spin_rank=2,
    )
    if bool(jit_payload):
        kernel = _cached_neural_xc_payload_kernel(
            xc_functional=xc_functional,
            neural_vxc_clip=neural_vxc_clip,
            compute_exc=bool(compute_exc),
        )
        molecule_payload = _jit_payload_molecule_template(molecule_template)
        return kernel(
            molecule_payload,
            density_total,
            mo_coeff_spin,
            mo_occ_spin,
            mo_energy_spin,
            xc_params,
        )
    return _neural_xc_fock_payload_core(
        density_total=density_total,
        mo_coeff_spin=mo_coeff_spin,
        mo_occ_spin=mo_occ_spin,
        mo_energy_spin=mo_energy_spin,
        molecule_template=molecule_template,
        xc_functional=xc_functional,
        xc_params=xc_params,
        neural_vxc_clip=neural_vxc_clip,
        compute_exc=bool(compute_exc),
    )


def _neural_xc_uks_fock_payload_core(
    *,
    density_spin: Any,
    mo_coeff_spin: Any,
    mo_occ_spin: Any,
    mo_energy_spin: Any,
    molecule_template: Any,
    xc_functional: Any,
    xc_params: Any,
    neural_vxc_clip: float | None,
    compute_exc: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    from .differentiable import (
        _build_vxc_matrix_from_components,
        _unrestricted_scf_xc_components,
    )

    h1e = jnp.asarray(molecule_template.h1e)
    dtype = h1e.dtype
    density_spin = jnp.asarray(density_spin, dtype=dtype)
    mo_coeff_spin = jnp.asarray(mo_coeff_spin, dtype=dtype)
    mo_occ_spin = jnp.asarray(mo_occ_spin, dtype=dtype)
    mo_energy_spin = jnp.asarray(mo_energy_spin, dtype=dtype)
    molecule_iter = replace(
        molecule_template,
        rdm1=density_spin,
        mo_coeff=mo_coeff_spin,
        mo_occ=mo_occ_spin,
        mo_energy=mo_energy_spin,
    )
    weights = jnp.asarray(molecule_iter.grid.weights, dtype=dtype)
    (
        vxc_rho_a,
        vxc_rho_b,
        vxc_grad_a,
        vxc_grad_b,
        xc_kind,
        alpha,
        extra_fock_a,
        extra_fock_b,
    ) = _unrestricted_scf_xc_components(
        xc_params,
        xc_functional,
        molecule_iter,
        functional_dtype=dtype,
    )
    if neural_vxc_clip is not None:
        clip = jnp.asarray(float(neural_vxc_clip), dtype=dtype)
        vxc_rho_a = jnp.clip(
            jnp.nan_to_num(vxc_rho_a, nan=0.0, posinf=clip, neginf=-clip),
            -clip,
            clip,
        )
        vxc_rho_b = jnp.clip(
            jnp.nan_to_num(vxc_rho_b, nan=0.0, posinf=clip, neginf=-clip),
            -clip,
            clip,
        )
        vxc_grad_a = jnp.clip(
            jnp.nan_to_num(vxc_grad_a, nan=0.0, posinf=clip, neginf=-clip),
            -clip,
            clip,
        )
        vxc_grad_b = jnp.clip(
            jnp.nan_to_num(vxc_grad_b, nan=0.0, posinf=clip, neginf=-clip),
            -clip,
            clip,
        )
    zeros_a = jnp.zeros_like(vxc_rho_a)
    zeros_b = jnp.zeros_like(vxc_rho_b)
    vxc_matrix_a = _build_vxc_matrix_from_components(
        molecule=molecule_iter,
        weights=weights,
        v_rho=vxc_rho_a,
        v_grad=vxc_grad_a,
        v_tau=zeros_a,
        v_lapl=zeros_a,
        xc_kind=xc_kind,
    )
    vxc_matrix_b = _build_vxc_matrix_from_components(
        molecule=molecule_iter,
        weights=weights,
        v_rho=vxc_rho_b,
        v_grad=vxc_grad_b,
        v_tau=zeros_b,
        v_lapl=zeros_b,
        xc_kind=xc_kind,
    )
    alpha = jnp.clip(jnp.nan_to_num(jnp.asarray(alpha, dtype=dtype), nan=0.0), 0.0, 1.0)
    if compute_exc:
        exc = _neural_xc_energy(
            xc_functional=xc_functional,
            xc_params=xc_params,
            molecule_iter=molecule_iter,
        ).astype(dtype)
    else:
        exc = jnp.asarray(0.0, dtype=dtype)
    return (
        jnp.stack(
            [
                0.5 * (vxc_matrix_a + vxc_matrix_a.T),
                0.5 * (vxc_matrix_b + vxc_matrix_b.T),
            ],
            axis=0,
        ),
        exc,
        alpha,
        jnp.stack(
            [
                0.5 * (extra_fock_a + extra_fock_a.T),
                0.5 * (extra_fock_b + extra_fock_b.T),
            ],
            axis=0,
        ),
    )


def _cached_neural_xc_uks_payload_kernel(
    *,
    xc_functional: Any,
    neural_vxc_clip: float | None,
    compute_exc: bool,
):
    key = (
        id(xc_functional),
        None if neural_vxc_clip is None else float(neural_vxc_clip),
        bool(compute_exc),
    )
    cached = _NEURAL_XC_UKS_PAYLOAD_JIT_CACHE.get(key)
    if cached is not None:
        return cached

    def kernel(
        molecule_template,
        density_spin,
        mo_coeff_spin,
        mo_occ_spin,
        mo_energy_spin,
        xc_params,
    ):
        return _neural_xc_uks_fock_payload_core(
            density_spin=density_spin,
            mo_coeff_spin=mo_coeff_spin,
            mo_occ_spin=mo_occ_spin,
            mo_energy_spin=mo_energy_spin,
            molecule_template=molecule_template,
            xc_functional=xc_functional,
            xc_params=xc_params,
            neural_vxc_clip=neural_vxc_clip,
            compute_exc=bool(compute_exc),
        )

    compiled = jax.jit(kernel)
    if len(_NEURAL_XC_UKS_PAYLOAD_JIT_CACHE) >= _NEURAL_XC_PAYLOAD_JIT_CACHE_MAXSIZE:
        _NEURAL_XC_UKS_PAYLOAD_JIT_CACHE.pop(next(iter(_NEURAL_XC_UKS_PAYLOAD_JIT_CACHE)))
    _NEURAL_XC_UKS_PAYLOAD_JIT_CACHE[key] = compiled
    return compiled


def _neural_xc_uks_fock_payload(
    *,
    mf_gpu: Any,
    dm: Any,
    molecule_template: Any,
    xc_functional: Any,
    xc_params: Any,
    neural_vxc_clip: float | None,
    compute_exc: bool = True,
    jit_payload: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    h1e = jnp.asarray(molecule_template.h1e)
    dtype = h1e.dtype
    density_spin = _to_jax_array(dm, dtype=dtype)
    mo_coeff_spin = _unrestricted_spin_array_from_attr(
        getattr(mf_gpu, "mo_coeff", None),
        template_value=molecule_template.mo_coeff,
        dtype=dtype,
        label="mo_coeff",
    )
    mo_occ_spin = _unrestricted_spin_array_from_attr(
        getattr(mf_gpu, "mo_occ", None),
        template_value=molecule_template.mo_occ,
        dtype=dtype,
        label="mo_occ",
    )
    mo_energy_spin = _unrestricted_spin_array_from_attr(
        getattr(mf_gpu, "mo_energy", None),
        template_value=molecule_template.mo_energy,
        dtype=dtype,
        label="mo_energy",
    )
    if bool(jit_payload):
        kernel = _cached_neural_xc_uks_payload_kernel(
            xc_functional=xc_functional,
            neural_vxc_clip=neural_vxc_clip,
            compute_exc=bool(compute_exc),
        )
        molecule_payload = _jit_payload_molecule_template(molecule_template)
        return kernel(
            molecule_payload,
            density_spin,
            mo_coeff_spin,
            mo_occ_spin,
            mo_energy_spin,
            xc_params,
        )
    return _neural_xc_uks_fock_payload_core(
        density_spin=density_spin,
        mo_coeff_spin=mo_coeff_spin,
        mo_occ_spin=mo_occ_spin,
        mo_energy_spin=mo_energy_spin,
        molecule_template=molecule_template,
        xc_functional=xc_functional,
        xc_params=xc_params,
        neural_vxc_clip=neural_vxc_clip,
        compute_exc=bool(compute_exc),
    )


def _spin_backend_array(value: Any, *, like: Any) -> Any:
    if not isinstance(value, (tuple, list)):
        return value
    cp = _cupy_module()
    if cp is not None and isinstance(like, cp.ndarray):
        return cp.stack([_to_cupy_or_numpy_array(block) for block in value], axis=0)
    return np.stack([_to_host_numpy(block) for block in value], axis=0)


def _backend_spin_dm_dot(dm: Any, mat: Any) -> float:
    dm_arr = _spin_backend_array(dm, like=mat)
    mat_arr = _spin_backend_array(mat, like=dm_arr)
    cp = _cupy_module()
    if cp is not None and (isinstance(dm_arr, cp.ndarray) or isinstance(mat_arr, cp.ndarray)):
        return _backend_scalar(cp.einsum("sij,sij", dm_arr, mat_arr))
    return _backend_scalar(np.einsum("sij,sij", np.asarray(dm_arr), np.asarray(mat_arr)))


def _install_neural_xc_get_veff(
    mf_gpu: Any,
    *,
    molecule_template: Any,
    xc_functional: Any,
    xc_params: Any,
    neural_vxc_clip: float | None,
    neural_xc_compute_exc: bool,
    neural_xc_jit_payload: bool,
) -> None:
    def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        del dm_last, vhf_last
        if mol is None:
            mol = ks.mol
        if dm is None:
            dm = ks.make_rdm1()
        if hermi == 2:
            zero = np.zeros_like(_to_host_numpy(dm))
            zero_backend = _to_backend_array(zero, like=dm)
            return _tag_array(
                zero_backend,
                ecoul=0.0,
                exc=0.0,
                vj=zero_backend,
                vk=None,
            )

        vxc_matrix, exc, alpha = _neural_xc_fock_payload(
            mf_gpu=ks,
            dm=dm,
            molecule_template=molecule_template,
            xc_functional=xc_functional,
            xc_params=xc_params,
            neural_vxc_clip=neural_vxc_clip,
            compute_exc=bool(neural_xc_compute_exc),
            jit_payload=bool(neural_xc_jit_payload),
        )
        alpha_float = _backend_scalar(jax.device_get(alpha))
        vk_raw = None
        get_jk = getattr(ks, "get_jk", None)
        if abs(alpha_float) > 1e-14 and get_jk is not None:
            vj, vk_raw = get_jk(mol, dm, hermi=hermi)
        else:
            vj = ks.get_j(mol, dm, hermi)
        vxc_backend = _to_backend_array(vxc_matrix, like=vj)
        veff = vj + vxc_backend
        vk = None
        exc_float = _backend_scalar(jax.device_get(exc))
        if abs(alpha_float) > 1e-14:
            if vk_raw is None:
                vk_raw = ks.get_k(mol, dm, hermi)
            vk = 0.5 * alpha_float * vk_raw
            veff = veff - vk
            exc_float -= 0.5 * _backend_dm_dot(dm, vk)
        ecoul = 0.5 * _backend_dm_dot(dm, vj)
        ks._td_graddft_neural_xc_alpha = alpha_float
        ks._td_graddft_neural_xc_exc = exc_float
        return _tag_array(veff, ecoul=ecoul, exc=exc_float, vj=vj, vk=vk)

    mf_gpu.get_veff = types.MethodType(get_veff, mf_gpu)


def _install_neural_xc_uks_get_veff(
    mf_gpu: Any,
    *,
    molecule_template: Any,
    xc_functional: Any,
    xc_params: Any,
    neural_vxc_clip: float | None,
    neural_xc_compute_exc: bool,
    neural_xc_jit_payload: bool,
) -> None:
    def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        del dm_last, vhf_last
        if mol is None:
            mol = ks.mol
        if dm is None:
            dm = ks.make_rdm1()
        dm_backend = _spin_backend_array(dm, like=dm[0] if isinstance(dm, (tuple, list)) else dm)
        if hermi == 2:
            zero = np.zeros_like(_to_host_numpy(dm_backend))
            zero_backend = _to_backend_array(zero, like=dm_backend)
            return _tag_array(
                zero_backend,
                ecoul=0.0,
                exc=0.0,
                vj=zero_backend,
                vk=None,
            )

        vxc_matrix, exc, alpha, extra_fock = _neural_xc_uks_fock_payload(
            mf_gpu=ks,
            dm=dm_backend,
            molecule_template=molecule_template,
            xc_functional=xc_functional,
            xc_params=xc_params,
            neural_vxc_clip=neural_vxc_clip,
            compute_exc=bool(neural_xc_compute_exc),
            jit_payload=bool(neural_xc_jit_payload),
        )
        alpha_float = _backend_scalar(jax.device_get(alpha))
        dm_total = dm_backend[0] + dm_backend[1]
        vj_total = ks.get_j(mol, dm_total, hermi)
        vxc_backend = _to_backend_array(vxc_matrix, like=vj_total)
        extra_backend = _to_backend_array(extra_fock, like=vj_total)
        veff = vj_total[None, :, :] + vxc_backend + extra_backend
        vk = None
        exc_float = _backend_scalar(jax.device_get(exc))
        if abs(alpha_float) > 1e-14:
            vk_raw = ks.get_k(mol, dm_backend, hermi)
            vk_raw = _spin_backend_array(vk_raw, like=vj_total)
            vk = alpha_float * vk_raw
            veff = veff - vk
            exc_float -= 0.5 * _backend_spin_dm_dot(dm_backend, vk)
        ecoul = 0.5 * _backend_dm_dot(dm_total, vj_total)
        ks._td_graddft_neural_xc_alpha = alpha_float
        ks._td_graddft_neural_xc_exc = exc_float
        return _tag_array(veff, ecoul=ecoul, exc=exc_float, vj=vj_total, vk=vk)

    mf_gpu.get_veff = types.MethodType(get_veff, mf_gpu)


def molecule_from_gpu4pyscf_rks_forward_result(
    molecule_template: Any,
    forward: GPU4PySCFRKSForwardResult,
) -> Any:
    dtype = jnp.asarray(molecule_template.h1e).dtype
    density_total = jnp.asarray(forward.density_matrix, dtype=dtype)
    mo_coeff = jnp.asarray(forward.mo_coeff, dtype=dtype)
    mo_occ_total = jnp.asarray(forward.mo_occ, dtype=dtype)
    mo_energy = jnp.asarray(forward.mo_energy, dtype=dtype)
    updates: dict[str, Any] = dict(
        rdm1=jnp.stack([0.5 * density_total, 0.5 * density_total], axis=0),
        mo_coeff=jnp.stack([mo_coeff, mo_coeff], axis=0),
        mo_occ=jnp.stack([0.5 * mo_occ_total, 0.5 * mo_occ_total], axis=0),
        mo_energy=jnp.stack([mo_energy, mo_energy], axis=0),
        mf_energy=float(forward.total_energy),
        scf_converged=bool(forward.converged),
        runtime_scf_backend=GPU4PYSCF_RKS_RUNTIME_BACKEND,
    )
    if forward.exact_exchange_fraction is not None:
        updates["exact_exchange_fraction"] = float(forward.exact_exchange_fraction)
    hfx_local = getattr(molecule_template, "hfx_local", None)
    hfx_nu = getattr(molecule_template, "hfx_nu", None)
    if hfx_local is not None:
        updates["hfx_local"] = jax.lax.stop_gradient(jnp.asarray(hfx_local, dtype=dtype))
    elif hfx_nu is not None:
        from .differentiable import _restricted_hfx_features_from_nu

        updates["hfx_local"] = _restricted_hfx_features_from_nu(
            ao=jnp.asarray(molecule_template.ao),
            density=density_total,
            nu_cache=hfx_nu,
        )
    return replace(molecule_template, **updates)


def run_gpu4pyscf_rks_forward(
    *,
    atom: Any,
    basis: Any,
    xc_spec: str = "pbe",
    unit: str = "Angstrom",
    charge: int = 0,
    spin: int = 0,
    cart: bool = True,
    grids_level: int = 0,
    conv_tol: float = 1e-10,
    max_cycle: int = 80,
    verbose: int = 0,
    molecule_template: Any | None = None,
    xc_functional: Any | None = None,
    xc_params: Any | None = None,
    neural_vxc_clip: float | None = 20.0,
    neural_xc_compute_exc: bool = True,
    neural_xc_jit_payload: bool = False,
    require_convergence: bool = True,
    collect_fock: bool = True,
    initial_density_matrix: Any | None = None,
    **mol_kwargs: Any,
) -> GPU4PySCFRKSForwardResult:
    """Run an exact GPU4PySCF restricted Kohn-Sham SCF forward pass.

    This intentionally uses ``dft.RKS(mol).to_gpu()`` directly. It does not call
    ``density_fit()``, so the forward reference follows GPU4PySCF's direct
    non-DF RKS path.
    """

    try:
        import gpu4pyscf  # noqa: F401
        from pyscf import dft, gto
    except ModuleNotFoundError as exc:
        raise ImportError(
            "gpu4pyscf and pyscf are required for GPU4PySCF RKS forward. "
            "Install gpu4pyscf-cuda11x/gpu4pyscf-cuda12x in the active environment."
        ) from exc

    if isinstance(atom, MoleculeSpec):
        charge = int(atom.charge)
        spin = int(atom.spin)
    if int(spin) != 0:
        raise NotImplementedError("GPU4PySCF RKS forward currently supports closed-shell spin=0 only.")
    if not bool(cart):
        raise NotImplementedError("GPU4PySCF RKS forward currently supports cart=True only.")

    pyscf_atom, pyscf_unit = _pyscf_atom_and_unit(atom, unit)
    mol = gto.M(
        atom=pyscf_atom,
        basis=basis,
        unit=pyscf_unit,
        spin=int(spin),
        charge=int(charge),
        cart=bool(cart),
        verbose=int(verbose),
        **mol_kwargs,
    )
    mf = dft.RKS(mol)
    mf.xc = _gpu4pyscf_engine_xc_spec(
        str(xc_spec),
        has_custom_xc=xc_functional is not None,
    )
    mf._td_graddft_requested_xc_spec = str(xc_spec)
    mf.grids.level = int(grids_level)
    mf.conv_tol = float(conv_tol)
    mf.max_cycle = int(max_cycle)
    if not hasattr(mf, "to_gpu"):
        raise RuntimeError(
            "PySCF RKS object does not expose to_gpu(). PySCF >= 2.5 with GPU4PySCF is required."
        )
    mf_gpu = mf.to_gpu()
    if xc_functional is not None:
        if molecule_template is None:
            raise ValueError(
                "molecule_template is required when injecting a Neural XC functional "
                "into GPU4PySCF RKS."
            )
        if xc_params is None:
            raise ValueError("xc_params is required when xc_functional is provided.")
        _install_neural_xc_get_veff(
            mf_gpu,
            molecule_template=molecule_template,
            xc_functional=xc_functional,
            xc_params=xc_params,
            neural_vxc_clip=neural_vxc_clip,
            neural_xc_compute_exc=bool(neural_xc_compute_exc),
            neural_xc_jit_payload=bool(neural_xc_jit_payload),
        )
    if initial_density_matrix is None:
        total_energy = float(mf_gpu.kernel())
    else:
        total_energy = float(mf_gpu.kernel(dm0=_to_cupy_or_numpy_array(initial_density_matrix)))
    _sync_gpu4pyscf()
    converged = bool(getattr(mf_gpu, "converged", False))
    if bool(require_convergence) and not converged:
        raise RuntimeError("GPU4PySCF exact RKS SCF did not converge.")

    density_matrix = _to_host_numpy(mf_gpu.make_rdm1(), dtype=float)
    fock_matrix = None
    get_fock = getattr(mf_gpu, "get_fock", None) if bool(collect_fock) else None
    if get_fock is not None:
        try:
            fock_matrix = _to_host_numpy(get_fock(), dtype=float)
        except Exception:
            fock_matrix = None
    cycles = getattr(mf_gpu, "cycles", None)
    if cycles is None:
        cycles = getattr(mf_gpu, "cycle", None)
    exact_exchange_fraction = getattr(mf_gpu, "_td_graddft_neural_xc_alpha", None)

    return GPU4PySCFRKSForwardResult(
        converged=converged,
        total_energy=total_energy,
        mo_energy=_to_host_numpy(mf_gpu.mo_energy, dtype=float),
        mo_coeff=_to_host_numpy(mf_gpu.mo_coeff, dtype=float),
        mo_occ=_to_host_numpy(mf_gpu.mo_occ, dtype=float),
        density_matrix=density_matrix,
        fock_matrix=fock_matrix,
        cycles=None if cycles is None else int(cycles),
        exact_exchange_fraction=(
            None if exact_exchange_fraction is None else float(exact_exchange_fraction)
        ),
    )


def run_gpu4pyscf_uks_forward(
    *,
    atom: Any,
    basis: Any,
    xc_spec: str = "pbe",
    unit: str = "Angstrom",
    charge: int = 0,
    spin: int = 1,
    cart: bool = True,
    grids_level: int = 0,
    conv_tol: float = 1e-10,
    max_cycle: int = 80,
    verbose: int = 0,
    molecule_template: Any | None = None,
    xc_functional: Any | None = None,
    xc_params: Any | None = None,
    neural_vxc_clip: float | None = 20.0,
    neural_xc_compute_exc: bool = True,
    neural_xc_jit_payload: bool = False,
    require_convergence: bool = True,
    collect_fock: bool = True,
    **mol_kwargs: Any,
) -> GPU4PySCFUKSForwardResult:
    """Run an exact GPU4PySCF unrestricted Kohn-Sham SCF forward pass."""

    try:
        import gpu4pyscf  # noqa: F401
        from pyscf import dft, gto
    except ModuleNotFoundError as exc:
        raise ImportError(
            "gpu4pyscf and pyscf are required for GPU4PySCF UKS forward. "
            "Install gpu4pyscf-cuda11x/gpu4pyscf-cuda12x in the active environment."
        ) from exc

    if isinstance(atom, MoleculeSpec):
        charge = int(atom.charge)
        spin = int(atom.spin)
    if not bool(cart):
        raise NotImplementedError("GPU4PySCF UKS forward currently supports cart=True only.")

    pyscf_atom, pyscf_unit = _pyscf_atom_and_unit(atom, unit)
    mol = gto.M(
        atom=pyscf_atom,
        basis=basis,
        unit=pyscf_unit,
        spin=int(spin),
        charge=int(charge),
        cart=bool(cart),
        verbose=int(verbose),
        **mol_kwargs,
    )
    mf = dft.UKS(mol)
    mf.xc = _gpu4pyscf_engine_xc_spec(
        str(xc_spec),
        has_custom_xc=xc_functional is not None,
    )
    mf._td_graddft_requested_xc_spec = str(xc_spec)
    mf.grids.level = int(grids_level)
    mf.conv_tol = float(conv_tol)
    mf.max_cycle = int(max_cycle)
    if not hasattr(mf, "to_gpu"):
        raise RuntimeError(
            "PySCF UKS object does not expose to_gpu(). PySCF >= 2.5 with GPU4PySCF is required."
        )
    mf_gpu = mf.to_gpu()
    if xc_functional is not None:
        if molecule_template is None:
            raise ValueError(
                "molecule_template is required when injecting a Neural XC functional "
                "into GPU4PySCF UKS."
            )
        _install_neural_xc_uks_get_veff(
            mf_gpu,
            molecule_template=molecule_template,
            xc_functional=xc_functional,
            xc_params=xc_params,
            neural_vxc_clip=neural_vxc_clip,
            neural_xc_compute_exc=bool(neural_xc_compute_exc),
            neural_xc_jit_payload=bool(neural_xc_jit_payload),
        )
    total_energy = float(mf_gpu.kernel())
    _sync_gpu4pyscf()
    converged = bool(getattr(mf_gpu, "converged", False))
    if bool(require_convergence) and not converged:
        raise RuntimeError("GPU4PySCF exact UKS SCF did not converge.")

    density_matrix = _spin_stack_host_numpy(mf_gpu.make_rdm1(), dtype=float)
    fock_matrix = None
    get_fock = getattr(mf_gpu, "get_fock", None) if bool(collect_fock) else None
    if get_fock is not None:
        try:
            fock_matrix = _spin_stack_host_numpy(get_fock(), dtype=float)
        except Exception:
            fock_matrix = None
    cycles = getattr(mf_gpu, "cycles", None)
    if cycles is None:
        cycles = getattr(mf_gpu, "cycle", None)

    return GPU4PySCFUKSForwardResult(
        converged=converged,
        total_energy=total_energy,
        mo_energy=_spin_stack_host_numpy(mf_gpu.mo_energy, dtype=float),
        mo_coeff=_spin_stack_host_numpy(mf_gpu.mo_coeff, dtype=float),
        mo_occ=_spin_stack_host_numpy(mf_gpu.mo_occ, dtype=float),
        density_matrix=density_matrix,
        fock_matrix=fock_matrix,
        cycles=None if cycles is None else int(cycles),
    )


__all__ = [
    "GPU4PYSCF_RKS_RUNTIME_BACKEND",
    "GPU4PYSCF_UKS_RUNTIME_BACKEND",
    "GPU4PySCFRKSForwardResult",
    "GPU4PySCFRKSForwardOptions",
    "GPU4PySCFUKSForwardResult",
    "GPU4PySCFUKSForwardOptions",
    "compute_gpu4pyscf_direct_jk_response",
    "compute_gpu4pyscf_direct_jk_response_from_options",
    "molecule_from_gpu4pyscf_rks_forward_result",
    "run_gpu4pyscf_rks_forward",
    "run_gpu4pyscf_uks_forward",
]
