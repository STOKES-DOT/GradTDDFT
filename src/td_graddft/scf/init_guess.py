from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..data.integrals.libcint import build_libcint_mol
from ..data.molecule import MoleculeSpec

_HCORE_GUESS_KEYS = frozenset({"1e", "hcore"})
_PYSCF_GUESS_KEYS = frozenset(
    {"minao", "atom", "huckel", "mod_huckel", "sap", "vsap", "chkfile"}
)


@dataclass(frozen=True)
class RestrictedInitGuess:
    density: Array | None = None


@dataclass(frozen=True)
class UnrestrictedInitGuess:
    density_alpha: Array | None = None
    density_beta: Array | None = None


def _guess_key(init_guess: Any) -> str | None:
    if not isinstance(init_guess, str):
        return None
    key = str(init_guess).strip().lower()
    if key.startswith("chk"):
        return "chkfile"
    return key


def _normalize_density_matrix(dm: Any) -> np.ndarray:
    dm_np = np.asarray(jax.device_get(dm), dtype=np.float64)
    if dm_np.ndim != 2 or dm_np.shape[0] != dm_np.shape[1]:
        raise ValueError("Restricted initial guess density must be a square 2D matrix.")
    return 0.5 * (dm_np + dm_np.T)


def _normalize_spin_density_matrices(dm: Any) -> tuple[np.ndarray, np.ndarray]:
    dm_np = np.asarray(jax.device_get(dm), dtype=np.float64)
    if dm_np.ndim == 3 and dm_np.shape[0] == 2:
        dma, dmb = dm_np[0], dm_np[1]
    elif isinstance(dm, (tuple, list)) and len(dm) == 2:
        dma = np.asarray(jax.device_get(dm[0]), dtype=np.float64)
        dmb = np.asarray(jax.device_get(dm[1]), dtype=np.float64)
    else:
        raise ValueError(
            "Unrestricted initial guess density must be a (2, nao, nao) array or a 2-tuple."
        )
    if dma.ndim != 2 or dmb.ndim != 2 or dma.shape != dmb.shape or dma.shape[0] != dma.shape[1]:
        raise ValueError("Unrestricted initial guess densities must be square matrices with matching shapes.")
    return 0.5 * (dma + dma.T), 0.5 * (dmb + dmb.T)


def _warn_traceable_init_guess_fallback(key: str) -> None:
    warnings.warn(
        f"Initial guess '{key}' is host-only in the current SCF implementation; "
        "traceable geometry falls back to the hcore/1e initial guess.",
        RuntimeWarning,
        stacklevel=3,
    )


def _build_pyscf_ks_object(
    *,
    restricted: bool,
    atom: Any,
    basis: Any,
    unit: str,
    charge: int,
    spin: int,
    cart: bool,
    verbose: int,
    xc_spec: str,
    sap_basis: Any | None,
    chkfile: str | None,
    libcint_mol: Any | None = None,
) -> tuple[Any, Any]:
    from pyscf import dft

    mol = libcint_mol
    if mol is None:
        mol = build_libcint_mol(
            atom=atom,
            basis=basis,
            unit=unit,
            charge=charge,
            spin=spin,
            cart=cart,
            verbose=verbose,
        )
    mf = dft.RKS(mol, xc=xc_spec) if restricted else dft.UKS(mol, xc=xc_spec)
    mf.verbose = int(verbose)
    if sap_basis is not None:
        mf.sap_basis = sap_basis
    if chkfile is not None:
        mf.chkfile = chkfile
    return mol, mf


def _restricted_guess_density_from_pyscf(
    *,
    atom: Any,
    basis: Any,
    unit: str,
    charge: int,
    spin: int,
    cart: bool,
    verbose: int,
    xc_spec: str,
    init_guess: str,
    sap_basis: Any | None,
    chkfile: str | None,
    chkfile_project: bool | None,
    libcint_mol: Any | None = None,
) -> np.ndarray | None:
    try:
        mol, mf = _build_pyscf_ks_object(
            restricted=True,
            atom=atom,
            basis=basis,
            unit=unit,
            charge=charge,
            spin=spin,
            cart=cart,
            verbose=verbose,
            xc_spec=xc_spec,
            sap_basis=sap_basis,
            chkfile=chkfile,
            libcint_mol=libcint_mol,
        )
    except ModuleNotFoundError:
        warnings.warn(
            "PySCF is unavailable; falling back to the hcore/1e initial guess.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None

    key = _guess_key(init_guess)
    if key is None:
        key = "minao"
    if key == "chkfile":
        try:
            dm = mf.init_guess_by_chkfile(chkfile=chkfile, project=chkfile_project)
        except (OSError, IOError, FileNotFoundError, KeyError, TypeError):
            warnings.warn(
                "Failed to load initial guess from chkfile; falling back to MINAO.",
                RuntimeWarning,
                stacklevel=3,
            )
            dm = mf.init_guess_by_minao(mol)
    else:
        dm = mf.get_init_guess(mol, key=key)
    return _normalize_density_matrix(dm)


def _unrestricted_guess_density_from_pyscf(
    *,
    atom: Any,
    basis: Any,
    unit: str,
    charge: int,
    spin: int,
    cart: bool,
    verbose: int,
    xc_spec: str,
    init_guess: str,
    sap_basis: Any | None,
    chkfile: str | None,
    chkfile_project: bool | None,
    libcint_mol: Any | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        mol, mf = _build_pyscf_ks_object(
            restricted=False,
            atom=atom,
            basis=basis,
            unit=unit,
            charge=charge,
            spin=spin,
            cart=cart,
            verbose=verbose,
            xc_spec=xc_spec,
            sap_basis=sap_basis,
            chkfile=chkfile,
            libcint_mol=libcint_mol,
        )
    except ModuleNotFoundError:
        warnings.warn(
            "PySCF is unavailable; falling back to the hcore/1e initial guess.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None

    key = _guess_key(init_guess)
    if key is None:
        key = "minao"
    if key == "chkfile":
        try:
            dm = mf.init_guess_by_chkfile(chkfile=chkfile, project=chkfile_project)
        except (OSError, IOError, FileNotFoundError, KeyError, TypeError):
            warnings.warn(
                "Failed to load initial guess from chkfile; falling back to MINAO.",
                RuntimeWarning,
                stacklevel=3,
            )
            dm = mf.init_guess_by_minao(mol)
    else:
        dm = mf.get_init_guess(mol, key=key)
    return _normalize_spin_density_matrices(dm)


def restricted_init_guess_from_pyscf(
    *,
    atom: Any,
    basis: Any,
    unit: str,
    charge: int,
    spin: int,
    cart: bool,
    verbose: int,
    xc_spec: str,
    init_guess: Any,
    sap_basis: Any | None,
    chkfile: str | None,
    chkfile_project: bool | None,
    geometry_is_traced: bool,
    dtype: Any,
    libcint_mol: Any | None = None,
) -> RestrictedInitGuess:
    if not isinstance(init_guess, str):
        if init_guess is None:
            return RestrictedInitGuess()
        return RestrictedInitGuess(density=jnp.asarray(_normalize_density_matrix(init_guess), dtype=dtype))

    key = _guess_key(init_guess)
    if key is None:
        key = "minao"
    if key in _HCORE_GUESS_KEYS:
        return RestrictedInitGuess()
    if key not in _PYSCF_GUESS_KEYS:
        key = "minao"
    if geometry_is_traced:
        _warn_traceable_init_guess_fallback(key)
        return RestrictedInitGuess()
    dm = _restricted_guess_density_from_pyscf(
        atom=atom,
        basis=basis,
        unit=unit,
        charge=charge,
        spin=spin,
        cart=cart,
        verbose=verbose,
        xc_spec=xc_spec,
        init_guess=key,
        sap_basis=sap_basis,
        chkfile=chkfile,
        chkfile_project=chkfile_project,
        libcint_mol=libcint_mol,
    )
    if dm is None:
        return RestrictedInitGuess()
    return RestrictedInitGuess(density=jnp.asarray(dm, dtype=dtype))


def unrestricted_init_guess_from_pyscf(
    *,
    atom: Any,
    basis: Any,
    unit: str,
    charge: int,
    spin: int,
    cart: bool,
    verbose: int,
    xc_spec: str,
    init_guess: Any,
    sap_basis: Any | None,
    chkfile: str | None,
    chkfile_project: bool | None,
    geometry_is_traced: bool,
    dtype: Any,
    libcint_mol: Any | None = None,
) -> UnrestrictedInitGuess:
    if not isinstance(init_guess, str):
        if init_guess is None:
            return UnrestrictedInitGuess()
        density_alpha, density_beta = _normalize_spin_density_matrices(init_guess)
        return UnrestrictedInitGuess(
            density_alpha=jnp.asarray(density_alpha, dtype=dtype),
            density_beta=jnp.asarray(density_beta, dtype=dtype),
        )

    key = _guess_key(init_guess)
    if key is None:
        key = "minao"
    if key in _HCORE_GUESS_KEYS:
        return UnrestrictedInitGuess()
    if key not in _PYSCF_GUESS_KEYS:
        key = "minao"
    if geometry_is_traced:
        _warn_traceable_init_guess_fallback(key)
        return UnrestrictedInitGuess()
    dm_pair = _unrestricted_guess_density_from_pyscf(
        atom=atom,
        basis=basis,
        unit=unit,
        charge=charge,
        spin=spin,
        cart=cart,
        verbose=verbose,
        xc_spec=xc_spec,
        init_guess=key,
        sap_basis=sap_basis,
        chkfile=chkfile,
        chkfile_project=chkfile_project,
        libcint_mol=libcint_mol,
    )
    if dm_pair is None:
        return UnrestrictedInitGuess()
    density_alpha, density_beta = dm_pair
    return UnrestrictedInitGuess(
        density_alpha=jnp.asarray(density_alpha, dtype=dtype),
        density_beta=jnp.asarray(density_beta, dtype=dtype),
    )


__all__ = [
    "RestrictedInitGuess",
    "UnrestrictedInitGuess",
    "restricted_init_guess_from_pyscf",
    "unrestricted_init_guess_from_pyscf",
]
