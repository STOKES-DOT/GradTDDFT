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


def _cupy_module():
    try:
        import cupy as cp
    except ModuleNotFoundError:
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
    try:
        import cupy as cp
    except ModuleNotFoundError:
        return
    cp.cuda.Stream.null.synchronize()


def _pyscf_atom_and_unit(atom: Any, unit: str) -> tuple[Any, str]:
    if not isinstance(atom, MoleculeSpec):
        return atom, unit
    coords_bohr = _to_host_numpy(atom.coords_bohr, dtype=float)
    records = [
        (symbol, tuple(float(item) for item in coords))
        for symbol, coords in zip(atom.symbols, coords_bohr, strict=True)
    ]
    return records, "Bohr"


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


def _neural_xc_fock_payload(
    *,
    mf_gpu: Any,
    dm: Any,
    molecule_template: Any,
    xc_functional: Any,
    xc_params: Any,
    neural_vxc_clip: float | None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    from .differentiable import (
        _build_vxc_matrix_from_components,
        _dm21_local_hfx_fock_correction,
        _restricted_hfx_features_from_nu,
        _scf_xc_components,
        _uses_dm21_local_hfx_correction,
    )

    h1e = jnp.asarray(molecule_template.h1e)
    dtype = h1e.dtype
    density_total = _to_jax_array(dm, dtype=dtype)
    density_spin = jnp.stack([0.5 * density_total, 0.5 * density_total], axis=0)
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

    updates = dict(
        rdm1=density_spin,
        mo_coeff=mo_coeff_spin,
        mo_occ=mo_occ_spin,
        mo_energy=mo_energy_spin,
    )
    hfx_nu = getattr(molecule_template, "hfx_nu", None)
    if hasattr(molecule_template, "hfx_local"):
        if hfx_nu is not None:
            updates["hfx_local"] = _restricted_hfx_features_from_nu(
                ao=jnp.asarray(molecule_template.ao),
                density=density_total,
                nu_cache=hfx_nu,
            )
        else:
            updates["hfx_local"] = getattr(molecule_template, "hfx_local", None)
    molecule_iter = replace(molecule_template, **updates)
    weights = jnp.asarray(molecule_iter.grid.weights, dtype=dtype)

    (
        vxc_rho,
        vxc_grad,
        vxc_tau,
        vxc_lapl,
        xc_kind,
        alpha,
        resolved_xc,
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
    if resolved_xc is not None and _uses_dm21_local_hfx_correction(resolved_xc):
        vxc_matrix = vxc_matrix + _dm21_local_hfx_fock_correction(
            resolved_xc=resolved_xc,
            molecule=molecule_iter,
            ao=jnp.asarray(molecule_iter.ao),
            density=density_total,
        )
    alpha = jnp.clip(jnp.nan_to_num(jnp.asarray(alpha, dtype=dtype), nan=0.0), 0.0, 1.0)
    exc = _neural_xc_energy(
        xc_functional=xc_functional,
        xc_params=xc_params,
        molecule_iter=molecule_iter,
    ).astype(dtype)
    return 0.5 * (vxc_matrix + vxc_matrix.T), exc, alpha


def _install_neural_xc_get_veff(
    mf_gpu: Any,
    *,
    molecule_template: Any,
    xc_functional: Any,
    xc_params: Any,
    neural_vxc_clip: float | None,
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

        vj = ks.get_j(mol, dm, hermi)
        vxc_matrix, exc, alpha = _neural_xc_fock_payload(
            mf_gpu=ks,
            dm=dm,
            molecule_template=molecule_template,
            xc_functional=xc_functional,
            xc_params=xc_params,
            neural_vxc_clip=neural_vxc_clip,
        )
        vxc_backend = _to_backend_array(vxc_matrix, like=vj)
        veff = vj + vxc_backend
        vk = None
        alpha_float = _backend_scalar(jax.device_get(alpha))
        exc_float = _backend_scalar(jax.device_get(exc))
        if abs(alpha_float) > 1e-14:
            vk_raw = ks.get_k(mol, dm, hermi)
            vk = 0.5 * alpha_float * vk_raw
            veff = veff - vk
            exc_float -= 0.5 * _backend_dm_dot(dm, vk)
        ecoul = 0.5 * _backend_dm_dot(dm, vj)
        ks._td_graddft_neural_xc_alpha = alpha_float
        ks._td_graddft_neural_xc_exc = exc_float
        return _tag_array(veff, ecoul=ecoul, exc=exc_float, vj=vj, vk=vk)

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
    hfx_nu = getattr(molecule_template, "hfx_nu", None)
    if hfx_nu is not None:
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
    mf.xc = str(xc_spec)
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
        )
    total_energy = float(mf_gpu.kernel())
    _sync_gpu4pyscf()
    converged = bool(getattr(mf_gpu, "converged", False))
    if not converged:
        raise RuntimeError("GPU4PySCF exact RKS SCF did not converge.")

    density_matrix = _to_host_numpy(mf_gpu.make_rdm1(), dtype=float)
    fock_matrix = None
    get_fock = getattr(mf_gpu, "get_fock", None)
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


__all__ = [
    "GPU4PYSCF_RKS_RUNTIME_BACKEND",
    "GPU4PySCFRKSForwardResult",
    "GPU4PySCFRKSForwardOptions",
    "molecule_from_gpu4pyscf_rks_forward_result",
    "run_gpu4pyscf_rks_forward",
]
