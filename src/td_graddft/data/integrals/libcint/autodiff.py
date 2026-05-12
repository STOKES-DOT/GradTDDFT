from __future__ import annotations

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

LibcintGeometryGradPolicy = Literal["analytic", "error", "zero"]


def _as_device_array(value: Array) -> Array:
    if isinstance(value, jax.core.Tracer):
        return jnp.asarray(value)
    if isinstance(value, jax.Array):
        return value
    return jax.device_put(np.asarray(value))


def _build_pyscf_mol_from_coords(
    coords_bohr: Array,
    *,
    symbols: tuple[str, ...],
    basis: str,
    charge: int,
    spin: int,
    cart: bool,
    verbose: int,
):
    try:
        from pyscf import gto
    except ModuleNotFoundError as exc:
        raise ImportError("PySCF/libcint is required for libcint autodiff.") from exc

    coords_np = np.asarray(jax.device_get(coords_bohr), dtype=float)
    atom = [
        (str(sym), tuple(float(x) for x in coords_np[ia]))
        for ia, sym in enumerate(symbols)
    ]
    return gto.M(
        atom=atom,
        basis=basis,
        unit="Bohr",
        charge=int(charge),
        spin=int(spin),
        cart=bool(cart),
        verbose=int(verbose),
    )


def _intor_name(mol, name: str) -> str:
    return mol._add_suffix(str(name))


def _zero_if_allowed_or_raise(
    policy: str,
    coords_bohr: Array,
    *,
    integral_name: str,
) -> Array | None:
    if policy == "zero":
        return jnp.zeros_like(coords_bohr)
    if policy == "error":
        raise NotImplementedError(
            "Gradient through libcint integrals w.r.t. geometry/basis is not enabled. "
            "Set libcint_geometry_grad_policy='analytic' for PySCF-style "
            f"coordinate VJP support. Integral: {integral_name}."
        )
    return None


def _contract_one_electron_center_derivative(
    deriv_bra: np.ndarray,
    cotangent: np.ndarray,
    p0: int,
    p1: int,
) -> np.ndarray:
    return (
        np.einsum("xpj,pj->x", deriv_bra[:, p0:p1, :], cotangent[p0:p1, :], optimize=True)
        + np.einsum("xji,ij->x", deriv_bra[:, p0:p1, :], cotangent[:, p0:p1], optimize=True)
    )


def _libcint_int1e_coords_vjp(
    mol,
    intor_name: str,
    cotangent: Array,
) -> np.ndarray:
    cot = np.asarray(jax.device_get(cotangent), dtype=float)
    if intor_name == "int1e_r":
        if np.allclose(cot, 0.0, atol=0.0, rtol=0.0):
            return np.zeros((mol.natm, 3), dtype=float)
        if cot.ndim != 3:
            raise NotImplementedError(
                "libcint coordinate VJP for int1e_r expects a 3-D cotangent."
            )
        nao = mol.nao_nr()
        deriv_ket = -np.asarray(
            mol.intor(_intor_name(mol, "int1e_irp")),
            dtype=float,
        ).reshape(-1, 3, nao, nao)
        if deriv_ket.shape[0] != cot.shape[0]:
            raise NotImplementedError(
                "libcint coordinate VJP for int1e_r received an unexpected component shape."
            )
        aoslices = mol.aoslice_by_atom()
        grad = np.zeros((mol.natm, 3), dtype=float)
        for ia in range(mol.natm):
            p0, p1 = (int(x) for x in aoslices[ia, 2:4])
            atom_deriv = deriv_ket[:, :, :, p0:p1]
            grad[ia] = (
                np.einsum(
                    "dxja,dja->x",
                    atom_deriv,
                    cot[:, :, p0:p1],
                    optimize=True,
                )
                + np.einsum(
                    "dxja,daj->x",
                    atom_deriv,
                    cot[:, p0:p1, :],
                    optimize=True,
                )
            )
        return grad
    if cot.ndim != 2:
        raise NotImplementedError(
            f"libcint coordinate VJP for {intor_name!r} expects a 2-D cotangent."
        )

    ip_name_by_intor = {
        "int1e_ovlp": "int1e_ipovlp",
        "int1e_kin": "int1e_ipkin",
        "int1e_nuc": "int1e_ipnuc",
    }
    if intor_name not in ip_name_by_intor:
        raise NotImplementedError(f"libcint coordinate VJP for {intor_name!r} is not implemented.")

    aoslices = mol.aoslice_by_atom()
    grad = np.zeros((mol.natm, 3), dtype=float)
    deriv_bra = -np.asarray(mol.intor(_intor_name(mol, ip_name_by_intor[intor_name]), comp=3), dtype=float)

    for ia in range(mol.natm):
        p0, p1 = (int(x) for x in aoslices[ia, 2:4])
        if intor_name == "int1e_nuc":
            with mol.with_rinv_at_nucleus(ia):
                deriv_atom = -float(mol.atom_charge(ia)) * np.asarray(
                    mol.intor(_intor_name(mol, "int1e_iprinv"), comp=3),
                    dtype=float,
                )
            deriv_atom[:, p0:p1, :] += deriv_bra[:, p0:p1, :]
            deriv_atom = deriv_atom + deriv_atom.transpose(0, 2, 1)
            grad[ia] = np.einsum("xij,ij->x", deriv_atom, cot, optimize=True)
        else:
            grad[ia] = _contract_one_electron_center_derivative(deriv_bra, cot, p0, p1)
    return grad


def _libcint_int2e_full_coords_vjp(mol, cotangent: Array) -> np.ndarray:
    cot = np.asarray(jax.device_get(cotangent), dtype=float)
    if cot.ndim != 4:
        raise NotImplementedError("libcint int2e coordinate VJP expects a 4-D cotangent.")

    eri_ip1 = -np.asarray(mol.intor(_intor_name(mol, "int2e_ip1"), aosym="s1", comp=3), dtype=float)
    aoslices = mol.aoslice_by_atom()
    grad = np.zeros((mol.natm, 3), dtype=float)
    for ia in range(mol.natm):
        p0, p1 = (int(x) for x in aoslices[ia, 2:4])
        grad[ia] += np.einsum(
            "xijkl,ijkl->x",
            eri_ip1[:, p0:p1, :, :, :],
            cot[p0:p1, :, :, :],
            optimize=True,
        )
        grad[ia] += np.einsum(
            "xjikl,ijkl->x",
            eri_ip1[:, p0:p1, :, :, :],
            cot[:, p0:p1, :, :],
            optimize=True,
        )
        grad[ia] += np.einsum(
            "xklij,ijkl->x",
            eri_ip1[:, p0:p1, :, :, :],
            cot[:, :, p0:p1, :],
            optimize=True,
        )
        grad[ia] += np.einsum(
            "xlkij,ijkl->x",
            eri_ip1[:, p0:p1, :, :, :],
            cot[:, :, :, p0:p1],
            optimize=True,
        )
    return grad


def _s4_cotangent_to_representative_full(cotangent_s4: Array, nao: int) -> np.ndarray:
    cot = np.asarray(jax.device_get(cotangent_s4), dtype=float)
    full = np.zeros((nao, nao, nao, nao), dtype=float)
    ij = 0
    for i in range(nao):
        for j in range(i + 1):
            kl = 0
            for k in range(nao):
                for l in range(k + 1):
                    full[i, j, k, l] = cot[ij, kl]
                    kl += 1
            ij += 1
    return full


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9))
def libcint_int1e_with_coords(
    coords_bohr: Array,
    symbols: tuple[str, ...],
    basis: str,
    charge: int,
    spin: int,
    cart: bool,
    verbose: int,
    intor_name: str,
    comp: int | None,
    geometry_grad_policy: LibcintGeometryGradPolicy,
) -> Array:
    mol = _build_pyscf_mol_from_coords(
        coords_bohr,
        symbols=symbols,
        basis=basis,
        charge=charge,
        spin=spin,
        cart=cart,
        verbose=verbose,
    )
    return jnp.asarray(mol.intor_symmetric(_intor_name(mol, intor_name), comp=comp))


def _libcint_int1e_with_coords_fwd(
    coords_bohr: Array,
    symbols: tuple[str, ...],
    basis: str,
    charge: int,
    spin: int,
    cart: bool,
    verbose: int,
    intor_name: str,
    comp: int | None,
    geometry_grad_policy: LibcintGeometryGradPolicy,
) -> tuple[Array, Array]:
    value = libcint_int1e_with_coords(
        coords_bohr,
        symbols,
        basis,
        charge,
        spin,
        cart,
        verbose,
        intor_name,
        comp,
        geometry_grad_policy,
    )
    return value, jnp.asarray(coords_bohr)


def _libcint_int1e_with_coords_bwd(
    symbols: tuple[str, ...],
    basis: str,
    charge: int,
    spin: int,
    cart: bool,
    verbose: int,
    intor_name: str,
    comp: int | None,
    geometry_grad_policy: LibcintGeometryGradPolicy,
    coords_bohr: Array,
    cotangent: Array,
) -> tuple[Array]:
    del comp
    policy = str(geometry_grad_policy).lower()
    maybe_zero = _zero_if_allowed_or_raise(policy, coords_bohr, integral_name=intor_name)
    if maybe_zero is not None:
        return (maybe_zero,)
    mol = _build_pyscf_mol_from_coords(
        coords_bohr,
        symbols=symbols,
        basis=basis,
        charge=charge,
        spin=spin,
        cart=cart,
        verbose=verbose,
    )
    grad = _libcint_int1e_coords_vjp(mol, str(intor_name), cotangent)
    return (jnp.asarray(grad, dtype=jnp.asarray(coords_bohr).dtype),)


libcint_int1e_with_coords.defvjp(
    _libcint_int1e_with_coords_fwd,
    _libcint_int1e_with_coords_bwd,
)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5, 6, 7))
def libcint_int2e_full_with_coords(
    coords_bohr: Array,
    symbols: tuple[str, ...],
    basis: str,
    charge: int,
    spin: int,
    cart: bool,
    verbose: int,
    geometry_grad_policy: LibcintGeometryGradPolicy,
) -> Array:
    mol = _build_pyscf_mol_from_coords(
        coords_bohr,
        symbols=symbols,
        basis=basis,
        charge=charge,
        spin=spin,
        cart=cart,
        verbose=verbose,
    )
    return jnp.asarray(mol.intor(_intor_name(mol, "int2e"), aosym="s1"))


def _libcint_int2e_full_with_coords_fwd(
    coords_bohr: Array,
    symbols: tuple[str, ...],
    basis: str,
    charge: int,
    spin: int,
    cart: bool,
    verbose: int,
    geometry_grad_policy: LibcintGeometryGradPolicy,
) -> tuple[Array, Array]:
    value = libcint_int2e_full_with_coords(
        coords_bohr,
        symbols,
        basis,
        charge,
        spin,
        cart,
        verbose,
        geometry_grad_policy,
    )
    return value, jnp.asarray(coords_bohr)


def _libcint_int2e_full_with_coords_bwd(
    symbols: tuple[str, ...],
    basis: str,
    charge: int,
    spin: int,
    cart: bool,
    verbose: int,
    geometry_grad_policy: LibcintGeometryGradPolicy,
    coords_bohr: Array,
    cotangent: Array,
) -> tuple[Array]:
    policy = str(geometry_grad_policy).lower()
    maybe_zero = _zero_if_allowed_or_raise(policy, coords_bohr, integral_name="int2e")
    if maybe_zero is not None:
        return (maybe_zero,)
    mol = _build_pyscf_mol_from_coords(
        coords_bohr,
        symbols=symbols,
        basis=basis,
        charge=charge,
        spin=spin,
        cart=cart,
        verbose=verbose,
    )
    grad = _libcint_int2e_full_coords_vjp(mol, cotangent)
    return (jnp.asarray(grad, dtype=jnp.asarray(coords_bohr).dtype),)


libcint_int2e_full_with_coords.defvjp(
    _libcint_int2e_full_with_coords_fwd,
    _libcint_int2e_full_with_coords_bwd,
)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5, 6, 7))
def libcint_int2e_s4_with_coords(
    coords_bohr: Array,
    symbols: tuple[str, ...],
    basis: str,
    charge: int,
    spin: int,
    cart: bool,
    verbose: int,
    geometry_grad_policy: LibcintGeometryGradPolicy,
) -> Array:
    mol = _build_pyscf_mol_from_coords(
        coords_bohr,
        symbols=symbols,
        basis=basis,
        charge=charge,
        spin=spin,
        cart=cart,
        verbose=verbose,
    )
    return jnp.asarray(mol.intor(_intor_name(mol, "int2e"), aosym="s4"))


def _libcint_int2e_s4_with_coords_fwd(
    coords_bohr: Array,
    symbols: tuple[str, ...],
    basis: str,
    charge: int,
    spin: int,
    cart: bool,
    verbose: int,
    geometry_grad_policy: LibcintGeometryGradPolicy,
) -> tuple[Array, Array]:
    value = libcint_int2e_s4_with_coords(
        coords_bohr,
        symbols,
        basis,
        charge,
        spin,
        cart,
        verbose,
        geometry_grad_policy,
    )
    return value, jnp.asarray(coords_bohr)


def _libcint_int2e_s4_with_coords_bwd(
    symbols: tuple[str, ...],
    basis: str,
    charge: int,
    spin: int,
    cart: bool,
    verbose: int,
    geometry_grad_policy: LibcintGeometryGradPolicy,
    coords_bohr: Array,
    cotangent: Array,
) -> tuple[Array]:
    policy = str(geometry_grad_policy).lower()
    maybe_zero = _zero_if_allowed_or_raise(policy, coords_bohr, integral_name="int2e_s4")
    if maybe_zero is not None:
        return (maybe_zero,)
    mol = _build_pyscf_mol_from_coords(
        coords_bohr,
        symbols=symbols,
        basis=basis,
        charge=charge,
        spin=spin,
        cart=cart,
        verbose=verbose,
    )
    cot_full = _s4_cotangent_to_representative_full(cotangent, mol.nao_nr())
    grad = _libcint_int2e_full_coords_vjp(mol, cot_full)
    return (jnp.asarray(grad, dtype=jnp.asarray(coords_bohr).dtype),)


libcint_int2e_s4_with_coords.defvjp(
    _libcint_int2e_s4_with_coords_fwd,
    _libcint_int2e_s4_with_coords_bwd,
)


@jax.custom_vjp
def _libcint_passthrough_error(
    geometry_anchor: Array,
    integral_value: Array,
) -> Array:
    del geometry_anchor
    return integral_value


def _libcint_passthrough_error_fwd(
    geometry_anchor: Array,
    integral_value: Array,
) -> tuple[Array, None]:
    del geometry_anchor
    return integral_value, None


def _libcint_passthrough_error_bwd(
    _residual: None,
    cotangent: Array,
) -> tuple[Array, Array]:
    del cotangent
    raise NotImplementedError(
        "Gradient through libcint integrals w.r.t. geometry/basis is not implemented yet. "
        "Use integral_backend='jax' or set libcint_geometry_grad_policy='zero' "
        "for constant-integral geometry gradients."
    )


_libcint_passthrough_error.defvjp(
    _libcint_passthrough_error_fwd,
    _libcint_passthrough_error_bwd,
)


def bind_libcint_integral_constant(
    integral_value: Array,
    *,
    geometry_anchor: Array,
    integral_name: str,
    geometry_grad_policy: LibcintGeometryGradPolicy = "error",
) -> Array:
    """Attach explicit AD policy to libcint values consumed in JAX graphs.

    Forward returns `integral_value` unchanged.
    Reverse mode:
    - policy="error": raises a clear error when a geometry/basis gradient is requested.
    - policy="zero": returns zero geometry gradients (treats libcint values as constants).
    """
    del integral_name

    policy = str(geometry_grad_policy).lower()
    if policy not in {"analytic", "error", "zero"}:
        raise ValueError(
            f"Unsupported geometry_grad_policy={geometry_grad_policy!r}. "
            "Expected 'analytic', 'error', or 'zero'."
        )

    geometry_anchor_arr = _as_device_array(geometry_anchor)
    integral_value_arr = _as_device_array(integral_value)

    if policy in {"analytic", "error"}:
        return _libcint_passthrough_error(
            geometry_anchor_arr,
            integral_value_arr,
        )
    del geometry_anchor_arr
    return integral_value_arr
