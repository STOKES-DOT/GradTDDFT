from __future__ import annotations

from dataclasses import dataclass, replace
import os
import warnings
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..data.basis import CartesianBasis, basis_from_molecule_spec
from ..data.grid import build_molecular_grid_from_spec
from ..data.grid_ao import evaluate_cartesian_ao, evaluate_cartesian_ao_with_derivatives
from ..data.integrals import (
    build_libcint_mol,
    build_hcore,
    dipole_matrix,
    eri_pair_matrix_packed,
    eri_tensor,
    libcint_intor_name,
    overlap_matrix,
    overlap_hcore_matrices,
    precompile_eri_kernels,
)
from ..data.integrals.libcint_autodiff import (
    LibcintGeometryGradPolicy,
    bind_libcint_integral_constant,
    libcint_int1e_with_coords,
    libcint_int2e_full_with_coords,
    libcint_int2e_s4_with_coords,
)
from ..data.molecule import MoleculeSpec, parse_molecule_spec
from ..df import (
    eri_pair_matrix_to_df_factors_traceable,
    eri_to_df_factors_from_basis,
    true_df_factors_from_libcint_mol,
)
from ..jax_libxc import parse_xc, xc_type
from .cuda_one_electron import CudaOneElectronBuilder
from .cuda_direct_jk import cuda_ffi_available
from .rks import RKSConfig
from .uks import UKSConfig

_DEFAULT_CUDA_PAIR_ERI_MAX_MIB = 2048.0
_GRID_AO_INPUT_CACHE_MAXSIZE = 32
_GRID_AO_INPUT_CACHE: dict[tuple[Any, ...], Any] = {}


@dataclass(frozen=True)
class _GridAOInputBundle:
    basis: CartesianBasis
    coords: Array
    grid_weights: Array
    ao: Array
    ao_deriv1: Array
    ao_laplacian: Array | None


@dataclass(frozen=True)
class RKSIntegralInputs:
    """AO integrals, grid data, and JK data required by the RKS SCF kernel."""

    basis: CartesianBasis
    overlap: Array
    hcore: Array
    eri: Array | None
    eri_pair_matrix: Array | None
    df_factors: Array | None
    direct_basis: CartesianBasis | None
    nelectron: int
    nuclear_repulsion: float | Array
    coords: Array
    grid_weights: Array
    ao: Array
    ao_deriv1: Array
    ao_laplacian: Array | None
    dipole_integrals: Array | None
    init_mo_coeff: Array | None = None
    init_mo_occ: Array | None = None
    init_mo_energy: Array | None = None
    molecule_charge: int = 0
    geometry_is_traced: bool = False
    integral_backend: str = "libcint"
    grid_ao_backend: str = "jax"

    def as_rks_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments accepted by run_rks_from_integrals."""

        return {
            "overlap": self.overlap,
            "hcore": self.hcore,
            "eri": self.eri,
            "eri_pair_matrix": self.eri_pair_matrix,
            "nelectron": self.nelectron,
            "nuclear_repulsion": self.nuclear_repulsion,
            "ao": self.ao,
            "ao_deriv1": self.ao_deriv1,
            "grid_weights": self.grid_weights,
            "df_factors": self.df_factors,
            "direct_basis": self.direct_basis,
            "init_mo_coeff": self.init_mo_coeff,
            "init_mo_occ": self.init_mo_occ,
            "init_mo_energy": self.init_mo_energy,
        }

    def response_eri_pair_matrix(self) -> Array | None:
        """Return packed AO-pair ERI data for response assembly when available."""

        if self.eri_pair_matrix is not None:
            return self.eri_pair_matrix
        if self.direct_basis is not None:
            return eri_pair_matrix_packed(self.direct_basis)
        return None


@dataclass(frozen=True)
class UKSIntegralInputs:
    """AO integrals and grid data required by the UKS SCF kernel."""

    basis: CartesianBasis
    overlap: Array
    hcore: Array
    eri: Array
    nalpha: int
    nbeta: int
    nuclear_repulsion: float | Array
    coords: Array
    grid_weights: Array
    ao: Array
    ao_deriv1: Array
    ao_laplacian: Array | None
    dipole_integrals: Array
    init_mo_coeff_alpha: Array | None = None
    init_mo_coeff_beta: Array | None = None
    init_mo_occ_alpha: Array | None = None
    init_mo_occ_beta: Array | None = None
    init_mo_energy_alpha: Array | None = None
    init_mo_energy_beta: Array | None = None
    total_electrons: int = 0
    molecule_charge: int = 0
    geometry_is_traced: bool = False
    integral_backend: str = "libcint"
    grid_ao_backend: str = "jax"

    def as_uks_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments accepted by run_uks_from_integrals."""

        return {
            "overlap": self.overlap,
            "hcore": self.hcore,
            "eri": self.eri,
            "nalpha": self.nalpha,
            "nbeta": self.nbeta,
            "nuclear_repulsion": self.nuclear_repulsion,
            "ao": self.ao,
            "ao_deriv1": self.ao_deriv1,
            "grid_weights": self.grid_weights,
            "init_mo_coeff_alpha": self.init_mo_coeff_alpha,
            "init_mo_coeff_beta": self.init_mo_coeff_beta,
            "init_mo_occ_alpha": self.init_mo_occ_alpha,
            "init_mo_occ_beta": self.init_mo_occ_beta,
            "init_mo_energy_alpha": self.init_mo_energy_alpha,
            "init_mo_energy_beta": self.init_mo_energy_beta,
        }


def _contains_jax_tracer(value: Any) -> bool:
    if isinstance(value, jax.core.Tracer):
        return True
    if isinstance(value, MoleculeSpec):
        return _contains_jax_tracer((value.coords_bohr, value.charges))
    if isinstance(value, dict):
        return any(_contains_jax_tracer(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_jax_tracer(item) for item in value)
    return False


def _resolve_config(config: RKSConfig | None, xc_spec: str | None) -> tuple[RKSConfig, str]:
    xc_spec_resolved = str(xc_spec if xc_spec is not None else (config.xc_spec if config is not None else "pbe"))
    parse_xc(xc_spec_resolved)
    cfg = RKSConfig(xc_spec=xc_spec_resolved) if config is None else config
    if cfg.xc_spec != xc_spec_resolved:
        cfg = replace(cfg, xc_spec=xc_spec_resolved)
    return cfg, xc_spec_resolved


def _precompute_eri_groups_for_rks(config: RKSConfig) -> bool:
    return not (
        config.jk_backend == "direct"
        and config.direct_jk_engine == "cuda"
        and cuda_ffi_available()
    )


def _cuda_pair_eri_max_bytes_for_inputs() -> int:
    raw = os.environ.get("TD_GRADDFT_CUDA_PAIR_ERI_MAX_MIB")
    if raw is None or str(raw).strip() == "":
        return int(_DEFAULT_CUDA_PAIR_ERI_MAX_MIB * 1024.0 * 1024.0)
    try:
        mib = float(raw)
    except ValueError:
        return int(_DEFAULT_CUDA_PAIR_ERI_MAX_MIB * 1024.0 * 1024.0)
    return max(0, int(mib * 1024.0 * 1024.0))


def _active_device_cache_key() -> tuple[str, int]:
    scalar = jnp.asarray(0.0)
    device = getattr(scalar, "device", None)
    if device is None:
        devices = tuple(scalar.devices())
        device = devices[0] if devices else jax.devices()[0]
    return (str(getattr(device, "platform", "unknown")), int(getattr(device, "id", -1)))


def _grid_ao_input_cache_key(
    *,
    spec: MoleculeSpec,
    basis: Any,
    max_l: int,
    grids_level: int,
    precompute_eri_groups: bool,
    needs_ao_laplacian: bool,
) -> tuple[Any, ...]:
    coords_bohr = np.asarray(jax.device_get(spec.coords_bohr), dtype=np.float64)
    return (
        tuple(spec.symbols),
        str(basis),
        int(max_l),
        int(grids_level),
        bool(precompute_eri_groups),
        bool(needs_ao_laplacian),
        coords_bohr.shape,
        coords_bohr.dtype.str,
        coords_bohr.tobytes(),
        _active_device_cache_key(),
    )


def _cache_grid_ao_input_bundle(
    key: tuple[Any, ...],
    bundle: _GridAOInputBundle,
) -> None:
    if len(_GRID_AO_INPUT_CACHE) >= _GRID_AO_INPUT_CACHE_MAXSIZE:
        _GRID_AO_INPUT_CACHE.pop(next(iter(_GRID_AO_INPUT_CACHE)))
    _GRID_AO_INPUT_CACHE[key] = bundle


def _build_grid_ao_input_bundle(
    *,
    spec: MoleculeSpec,
    basis: Any,
    max_l: int,
    grids_level: int,
    precompute_eri_groups: bool,
    needs_ao_laplacian: bool,
) -> _GridAOInputBundle:
    basis_cart = basis_from_molecule_spec(
        spec,
        basis=basis,
        max_l=max_l,
        precompute_eri_groups=precompute_eri_groups,
    )
    coords, weights = build_molecular_grid_from_spec(spec, level=grids_level)
    deriv_order = 2 if needs_ao_laplacian else 1
    ao, ao_derivs = evaluate_cartesian_ao_with_derivatives(
        basis_cart,
        coords,
        deriv=deriv_order,
    )
    if needs_ao_laplacian:
        ao_deriv1 = ao_derivs[:4]
        ao_laplacian = ao_derivs[4]
    else:
        ao_deriv1 = ao_derivs
        ao_laplacian = None
    return _GridAOInputBundle(
        basis=basis_cart,
        coords=coords,
        grid_weights=weights,
        ao=ao,
        ao_deriv1=ao_deriv1,
        ao_laplacian=ao_laplacian,
    )


def _cached_grid_ao_input_bundle(
    *,
    spec: MoleculeSpec,
    basis: Any,
    max_l: int,
    grids_level: int,
    precompute_eri_groups: bool,
    needs_ao_laplacian: bool,
) -> _GridAOInputBundle:
    key = _grid_ao_input_cache_key(
        spec=spec,
        basis=basis,
        max_l=max_l,
        grids_level=grids_level,
        precompute_eri_groups=precompute_eri_groups,
        needs_ao_laplacian=needs_ao_laplacian,
    )
    cached = _GRID_AO_INPUT_CACHE.get(key)
    if cached is not None:
        return cached
    bundle = _build_grid_ao_input_bundle(
        spec=spec,
        basis=basis,
        max_l=max_l,
        grids_level=grids_level,
        precompute_eri_groups=precompute_eri_groups,
        needs_ao_laplacian=needs_ao_laplacian,
    )
    _cache_grid_ao_input_bundle(key, bundle)
    return bundle


def _libcint_pair_eri_for_direct_cuda(
    config: RKSConfig,
    basis: CartesianBasis,
    *,
    geometry_is_traced: bool,
) -> bool:
    del config, basis, geometry_is_traced
    return False


def _overlap_hcore_for_rks_input(
    basis: CartesianBasis,
    config: RKSConfig,
    *,
    geometry_is_traced: bool,
) -> tuple[Array, Array]:
    if (
        config.jk_backend == "direct"
        and config.direct_jk_engine == "cuda"
        and cuda_ffi_available()
        and not bool(geometry_is_traced)
    ):
        return CudaOneElectronBuilder(basis).build_overlap_hcore()
    return overlap_hcore_matrices(basis)


def _resolve_uks_config(config: UKSConfig | None, xc_spec: str | None) -> tuple[UKSConfig, str]:
    xc_spec_resolved = str(xc_spec if xc_spec is not None else (config.xc_spec if config is not None else "pbe"))
    parse_xc(xc_spec_resolved)
    cfg = UKSConfig(xc_spec=xc_spec_resolved) if config is None else config
    if cfg.xc_spec != xc_spec_resolved:
        cfg = replace(cfg, xc_spec=xc_spec_resolved)
    return cfg, xc_spec_resolved


def _unrestricted_spin_electron_counts(
    total_electrons: int,
    spin: int,
) -> tuple[int, int]:
    total_electrons = int(total_electrons)
    spin = int(spin)
    if total_electrons <= 0:
        raise ValueError("Unrestricted references require a positive electron count.")
    if abs(spin) > total_electrons:
        raise ValueError(
            f"Spin={spin} is incompatible with total_electrons={total_electrons}."
        )
    if (total_electrons + spin) % 2 != 0:
        raise ValueError(
            f"Spin={spin} is incompatible with total_electrons={total_electrons}; "
            "N + spin must be even."
        )
    nalpha = (total_electrons + spin) // 2
    nbeta = total_electrons - nalpha
    if nalpha < 0 or nbeta < 0:
        raise ValueError(
            f"Failed to resolve spin electron counts for total_electrons={total_electrons}, spin={spin}."
        )
    return int(nalpha), int(nbeta)


def build_rks_integral_inputs(
    *,
    atom: Any,
    basis: Any,
    config: RKSConfig | None = None,
    xc_spec: str | None = None,
    unit: str = "Angstrom",
    charge: int = 0,
    spin: int = 0,
    cart: bool = True,
    grids_level: int = 0,
    max_l: int = 3,
    grid_ao_backend: Literal["jax"] = "jax",
    integral_backend: Literal["jax", "libcint"] = "libcint",
    libcint_geometry_grad_policy: LibcintGeometryGradPolicy = "analytic",
    precompile_eri: bool = False,
    precompile_eri_chunk_size: int = 512,
    _precompile_eri_kernels: Any = precompile_eri_kernels,
    include_dipole_integrals: bool = True,
    verbose: int = 0,
    **mol_kwargs: Any,
) -> RKSIntegralInputs:
    """Build integral/grid inputs for the restricted KS SCF kernel."""

    if isinstance(atom, MoleculeSpec):
        charge = int(atom.charge)
        spin = int(atom.spin)
    if int(spin) != 0:
        raise NotImplementedError("build_rks_integral_inputs only supports closed-shell systems.")
    if not bool(cart):
        raise NotImplementedError("build_rks_integral_inputs currently supports cart=True only.")

    cfg, xc_spec_resolved = _resolve_config(config, xc_spec)
    integral_backend_mode = str(integral_backend).lower()
    if integral_backend_mode not in {"jax", "libcint"}:
        raise ValueError(
            f"Unsupported integral_backend={integral_backend!r}. Expected 'jax' or 'libcint'."
        )
    grid_ao_backend_mode = str(grid_ao_backend).lower()
    if grid_ao_backend_mode != "jax":
        raise ValueError(
            f"Unsupported grid_ao_backend={grid_ao_backend!r}. "
            "Only grid_ao_backend='jax' is supported."
        )
    libcint_grad_policy_mode = str(libcint_geometry_grad_policy).lower()
    if libcint_grad_policy_mode not in {"analytic", "error", "zero"}:
        raise ValueError(
            f"Unsupported libcint_geometry_grad_policy={libcint_geometry_grad_policy!r}. "
            "Expected 'analytic', 'error', or 'zero'."
        )

    molecule_charge = int(charge)
    geometry_is_traced = False
    needs_ao_laplacian = xc_type(xc_spec_resolved) == "MGGA"
    ao_laplacian = None
    dipole_integrals = None
    precompute_eri_groups = _precompute_eri_groups_for_rks(cfg)

    if integral_backend_mode == "libcint":
        if isinstance(atom, MoleculeSpec) and _contains_jax_tracer(atom.coords_bohr):
            if not isinstance(basis, str):
                raise TypeError("Traceable libcint geometry currently supports named basis strings only.")
            spec = atom
            molecule_charge = int(spec.charge)
            geometry_is_traced = True
            basis_cart = basis_from_molecule_spec(
                spec,
                basis=basis,
                max_l=max_l,
                precompute_eri_groups=precompute_eri_groups,
            )
            coords_bohr = jnp.asarray(spec.coords_bohr)
            intor_args = (
                coords_bohr,
                tuple(spec.symbols),
                str(basis),
                molecule_charge,
                int(spec.spin),
                bool(cart),
                int(verbose),
            )
            overlap = libcint_int1e_with_coords(
                *intor_args,
                "int1e_ovlp",
                None,
                libcint_grad_policy_mode,
            )
            kinetic = libcint_int1e_with_coords(
                *intor_args,
                "int1e_kin",
                None,
                libcint_grad_policy_mode,
            )
            v_nuc = libcint_int1e_with_coords(
                *intor_args,
                "int1e_nuc",
                None,
                libcint_grad_policy_mode,
            )
            hcore = kinetic + v_nuc
            if include_dipole_integrals:
                dipole_integrals = libcint_int1e_with_coords(
                    *intor_args,
                    "int1e_r",
                    3,
                    libcint_grad_policy_mode,
                )
            if precompile_eri:
                warnings.warn(
                    "precompile_eri is ignored for traceable libcint geometry.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            eri = None
            eri_pair_matrix = None
            df_factors = None
            if cfg.jk_backend == "df":
                eri_pair_matrix_for_df = libcint_int2e_s4_with_coords(
                    coords_bohr,
                    tuple(spec.symbols),
                    str(basis),
                    molecule_charge,
                    int(spec.spin),
                    bool(cart),
                    int(verbose),
                    libcint_grad_policy_mode,
                )
                df_factors = eri_pair_matrix_to_df_factors_traceable(
                    eri_pair_matrix_for_df,
                    nao=basis_cart.nao,
                    tol=cfg.df_tol,
                    max_rank=cfg.df_max_rank,
                )
            elif cfg.jk_backend != "direct" or _libcint_pair_eri_for_direct_cuda(
                cfg,
                basis_cart,
                geometry_is_traced=geometry_is_traced,
            ):
                eri_pair_matrix = libcint_int2e_s4_with_coords(
                    coords_bohr,
                    tuple(spec.symbols),
                    str(basis),
                    molecule_charge,
                    int(spec.spin),
                    bool(cart),
                    int(verbose),
                    libcint_grad_policy_mode,
                )
            coords, weights = build_molecular_grid_from_spec(spec, level=grids_level)
            ao, ao_deriv1 = evaluate_cartesian_ao_with_derivatives(
                basis_cart,
                coords,
                deriv=1,
            )
            nuclear_repulsion = spec.nuclear_repulsion
            init_mo_coeff = None
            init_mo_occ = None
        else:
            spec = (
                atom
                if isinstance(atom, MoleculeSpec)
                else parse_molecule_spec(atom, unit=unit, charge=charge, spin=spin)
            )
            molecule_charge = int(spec.charge)
            grid_ao_bundle = _cached_grid_ao_input_bundle(
                spec=spec,
                basis=basis,
                max_l=max_l,
                grids_level=grids_level,
                precompute_eri_groups=precompute_eri_groups,
                needs_ao_laplacian=needs_ao_laplacian,
            )
            mol = build_libcint_mol(
                atom=atom,
                basis=basis,
                unit=unit,
                charge=int(charge),
                spin=int(spin),
                cart=bool(cart),
                verbose=int(verbose),
                **mol_kwargs,
            )
            geometry_anchor = np.asarray(mol.atom_coords(), dtype=float)
            basis_cart = grid_ao_bundle.basis
            overlap = bind_libcint_integral_constant(
                np.asarray(
                    mol.intor_symmetric(libcint_intor_name(mol, "int1e_ovlp")),
                    dtype=float,
                ),
                geometry_anchor=geometry_anchor,
                integral_name="int1e_ovlp",
                geometry_grad_policy=libcint_grad_policy_mode,
            )
            hcore_value = np.asarray(
                mol.intor_symmetric(libcint_intor_name(mol, "int1e_kin")),
                dtype=float,
            ) + np.asarray(
                mol.intor_symmetric(libcint_intor_name(mol, "int1e_nuc")),
                dtype=float,
            )
            hcore = bind_libcint_integral_constant(
                hcore_value,
                geometry_anchor=geometry_anchor,
                integral_name="int1e_kin+int1e_nuc",
                geometry_grad_policy=libcint_grad_policy_mode,
            )
            if include_dipole_integrals:
                dipole_integrals = bind_libcint_integral_constant(
                    np.asarray(
                        mol.intor_symmetric(libcint_intor_name(mol, "int1e_r"), comp=3),
                        dtype=float,
                    ),
                    geometry_anchor=geometry_anchor,
                    integral_name="int1e_r",
                    geometry_grad_policy=libcint_grad_policy_mode,
                )
            if precompile_eri:
                warnings.warn(
                    "precompile_eri is ignored when integral_backend='libcint'.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            eri_name = libcint_intor_name(mol, "int2e")
            eri = None
            eri_pair_matrix = None
            df_factors = None
            if cfg.jk_backend == "df":
                df_factors = bind_libcint_integral_constant(
                    true_df_factors_from_libcint_mol(mol),
                    geometry_anchor=geometry_anchor,
                    integral_name="df_cholesky_eri",
                    geometry_grad_policy=libcint_grad_policy_mode,
                )
            elif cfg.jk_backend != "direct" or _libcint_pair_eri_for_direct_cuda(
                cfg,
                basis_cart,
                geometry_is_traced=False,
            ):
                eri_pair_matrix = bind_libcint_integral_constant(
                    np.asarray(mol.intor(eri_name, aosym="s4"), dtype=float),
                    geometry_anchor=geometry_anchor,
                    integral_name=f"{eri_name}_s4",
                    geometry_grad_policy=libcint_grad_policy_mode,
                )

            coords = grid_ao_bundle.coords
            weights = grid_ao_bundle.grid_weights
            ao = grid_ao_bundle.ao
            ao_deriv1 = grid_ao_bundle.ao_deriv1
            ao_laplacian = grid_ao_bundle.ao_laplacian
            nuclear_repulsion = spec.nuclear_repulsion
            init_mo_coeff = None
            init_mo_occ = None
    else:
        spec = parse_molecule_spec(atom, unit=unit, charge=charge, spin=spin)
        molecule_charge = int(spec.charge)
        geometry_is_traced = _contains_jax_tracer(spec.coords_bohr)
        if geometry_is_traced:
            coords, weights = build_molecular_grid_from_spec(spec, level=grids_level)
            basis_cart = basis_from_molecule_spec(
                spec,
                basis=basis,
                max_l=max_l,
                precompute_eri_groups=precompute_eri_groups,
            )
        else:
            grid_ao_bundle = _cached_grid_ao_input_bundle(
                spec=spec,
                basis=basis,
                max_l=max_l,
                grids_level=grids_level,
                precompute_eri_groups=precompute_eri_groups,
                needs_ao_laplacian=needs_ao_laplacian,
            )
            basis_cart = grid_ao_bundle.basis
            coords = grid_ao_bundle.coords
            weights = grid_ao_bundle.grid_weights
        overlap, hcore = _overlap_hcore_for_rks_input(
            basis_cart,
            cfg,
            geometry_is_traced=geometry_is_traced,
        )
        if precompile_eri:
            _precompile_eri_kernels(
                basis_cart,
                engine="jit",
                chunk_size=int(precompile_eri_chunk_size),
            )
        eri = None
        eri_pair_matrix = None
        df_factors = None
        if cfg.jk_backend == "df":
            df_factors = eri_to_df_factors_from_basis(
                basis_cart,
                tol=cfg.df_tol,
                max_rank=cfg.df_max_rank,
            )
        elif cfg.jk_backend != "direct":
            eri_pair_matrix = eri_pair_matrix_packed(basis_cart)
        if geometry_is_traced:
            ao, ao_deriv1 = evaluate_cartesian_ao_with_derivatives(
                basis_cart,
                coords,
                deriv=1,
            )
            if needs_ao_laplacian:
                ao_laplacian = evaluate_cartesian_ao(basis_cart, coords, deriv=2)[4]
        else:
            ao = grid_ao_bundle.ao
            ao_deriv1 = grid_ao_bundle.ao_deriv1
            ao_laplacian = grid_ao_bundle.ao_laplacian
        nuclear_repulsion = spec.nuclear_repulsion
        if include_dipole_integrals:
            dipole_integrals = dipole_matrix(basis_cart)
        init_mo_coeff = None
        init_mo_occ = None

    nelectron = int(basis_cart.atom_charges.sum()) - molecule_charge
    return RKSIntegralInputs(
        basis=basis_cart,
        overlap=overlap,
        hcore=hcore,
        eri=eri,
        eri_pair_matrix=eri_pair_matrix,
        df_factors=df_factors,
        direct_basis=basis_cart if cfg.jk_backend == "direct" else None,
        nelectron=nelectron,
        nuclear_repulsion=nuclear_repulsion,
        coords=coords,
        grid_weights=weights,
        ao=ao,
        ao_deriv1=ao_deriv1,
        ao_laplacian=ao_laplacian,
        dipole_integrals=dipole_integrals,
        init_mo_coeff=init_mo_coeff,
        init_mo_occ=init_mo_occ,
        init_mo_energy=None,
        molecule_charge=molecule_charge,
        geometry_is_traced=geometry_is_traced,
        integral_backend=integral_backend_mode,
        grid_ao_backend=grid_ao_backend_mode,
    )


def build_uks_integral_inputs(
    *,
    atom: Any,
    basis: Any,
    config: UKSConfig | None = None,
    xc_spec: str | None = None,
    unit: str = "Angstrom",
    charge: int = 0,
    spin: int = 1,
    cart: bool = True,
    grids_level: int = 0,
    max_l: int = 3,
    grid_ao_backend: Literal["jax"] = "jax",
    integral_backend: Literal["jax", "libcint"] = "libcint",
    libcint_geometry_grad_policy: LibcintGeometryGradPolicy = "error",
    precompile_eri: bool = False,
    precompile_eri_chunk_size: int = 512,
    _precompile_eri_kernels: Any = precompile_eri_kernels,
    verbose: int = 0,
    **mol_kwargs: Any,
) -> UKSIntegralInputs:
    """Build integral/grid inputs for the unrestricted KS SCF kernel."""

    if isinstance(atom, MoleculeSpec):
        charge = int(atom.charge)
        spin = int(atom.spin)
    if not bool(cart):
        raise NotImplementedError("build_uks_integral_inputs currently supports cart=True only.")

    cfg, _ = _resolve_uks_config(config, xc_spec)
    del cfg
    integral_backend_mode = str(integral_backend).lower()
    if integral_backend_mode not in {"jax", "libcint"}:
        raise ValueError(
            f"Unsupported integral_backend={integral_backend!r}. Expected 'jax' or 'libcint'."
        )
    grid_ao_backend_mode = str(grid_ao_backend).lower()
    if grid_ao_backend_mode != "jax":
        raise ValueError(
            f"Unsupported grid_ao_backend={grid_ao_backend!r}. "
            "Only grid_ao_backend='jax' is supported."
        )
    libcint_grad_policy_mode = str(libcint_geometry_grad_policy).lower()
    if libcint_grad_policy_mode not in {"analytic", "error", "zero"}:
        raise ValueError(
            f"Unsupported libcint_geometry_grad_policy={libcint_geometry_grad_policy!r}. "
            "Expected 'analytic', 'error', or 'zero'."
        )

    molecule_charge = int(charge)
    geometry_is_traced = False
    ao_laplacian = None

    if integral_backend_mode == "libcint":
        if isinstance(atom, MoleculeSpec):
            if not isinstance(basis, str):
                raise TypeError("Traceable libcint geometry currently supports named basis strings only.")
            spec = atom
            molecule_charge = int(spec.charge)
            geometry_is_traced = _contains_jax_tracer(spec.coords_bohr)
            basis_cart = basis_from_molecule_spec(spec, basis=basis, max_l=max_l)
            coords_bohr = jnp.asarray(spec.coords_bohr)
            intor_args = (
                coords_bohr,
                tuple(spec.symbols),
                str(basis),
                int(spec.charge),
                int(spec.spin),
                bool(cart),
                int(verbose),
            )
            overlap = libcint_int1e_with_coords(
                *intor_args,
                "int1e_ovlp",
                None,
                libcint_grad_policy_mode,
            )
            kinetic = libcint_int1e_with_coords(
                *intor_args,
                "int1e_kin",
                None,
                libcint_grad_policy_mode,
            )
            v_nuc = libcint_int1e_with_coords(
                *intor_args,
                "int1e_nuc",
                None,
                libcint_grad_policy_mode,
            )
            hcore = kinetic + v_nuc
            dipole_integrals = libcint_int1e_with_coords(
                *intor_args,
                "int1e_r",
                3,
                libcint_grad_policy_mode,
            )
            if precompile_eri:
                warnings.warn(
                    "precompile_eri is ignored for traceable libcint geometry.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            eri = libcint_int2e_full_with_coords(
                coords_bohr,
                tuple(spec.symbols),
                str(basis),
                int(spec.charge),
                int(spec.spin),
                bool(cart),
                int(verbose),
                libcint_grad_policy_mode,
            )
            coords, weights = build_molecular_grid_from_spec(spec, level=grids_level)
            ao_deriv1 = evaluate_cartesian_ao(basis_cart, coords, deriv=1)
            ao = ao_deriv1[0]
            ao_laplacian = evaluate_cartesian_ao(basis_cart, coords, deriv=2)[4]
            nuclear_repulsion = spec.nuclear_repulsion
        else:
            spec = (
                atom
                if isinstance(atom, MoleculeSpec)
                else parse_molecule_spec(atom, unit=unit, charge=charge, spin=spin)
            )
            molecule_charge = int(spec.charge)
            grid_ao_bundle = _cached_grid_ao_input_bundle(
                spec=spec,
                basis=basis,
                max_l=max_l,
                grids_level=grids_level,
                precompute_eri_groups=False,
                needs_ao_laplacian=True,
            )
            mol = build_libcint_mol(
                atom=atom,
                basis=basis,
                unit=unit,
                charge=int(charge),
                spin=int(spin),
                cart=bool(cart),
                verbose=int(verbose),
                **mol_kwargs,
            )
            geometry_anchor = jnp.asarray(mol.atom_coords())
            basis_cart = grid_ao_bundle.basis
            overlap = bind_libcint_integral_constant(
                jnp.asarray(mol.intor_symmetric(libcint_intor_name(mol, "int1e_ovlp"))),
                geometry_anchor=geometry_anchor,
                integral_name="int1e_ovlp",
                geometry_grad_policy=libcint_grad_policy_mode,
            )
            kinetic = bind_libcint_integral_constant(
                jnp.asarray(mol.intor_symmetric(libcint_intor_name(mol, "int1e_kin"))),
                geometry_anchor=geometry_anchor,
                integral_name="int1e_kin",
                geometry_grad_policy=libcint_grad_policy_mode,
            )
            v_nuc = bind_libcint_integral_constant(
                jnp.asarray(mol.intor_symmetric(libcint_intor_name(mol, "int1e_nuc"))),
                geometry_anchor=geometry_anchor,
                integral_name="int1e_nuc",
                geometry_grad_policy=libcint_grad_policy_mode,
            )
            hcore = kinetic + v_nuc
            dipole_integrals = bind_libcint_integral_constant(
                jnp.asarray(mol.intor_symmetric(libcint_intor_name(mol, "int1e_r"), comp=3)),
                geometry_anchor=geometry_anchor,
                integral_name="int1e_r",
                geometry_grad_policy=libcint_grad_policy_mode,
            )
            if precompile_eri:
                warnings.warn(
                    "precompile_eri is ignored when integral_backend='libcint'.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            eri_name = libcint_intor_name(mol, "int2e")
            eri = bind_libcint_integral_constant(
                jnp.asarray(mol.intor(eri_name)),
                geometry_anchor=geometry_anchor,
                integral_name=eri_name,
                geometry_grad_policy=libcint_grad_policy_mode,
            )

            coords = grid_ao_bundle.coords
            weights = grid_ao_bundle.grid_weights
            ao = grid_ao_bundle.ao
            ao_deriv1 = grid_ao_bundle.ao_deriv1
            ao_laplacian = grid_ao_bundle.ao_laplacian
            nuclear_repulsion = spec.nuclear_repulsion
    else:
        spec = parse_molecule_spec(atom, unit=unit, charge=charge, spin=spin)
        molecule_charge = int(spec.charge)
        geometry_is_traced = _contains_jax_tracer(spec.coords_bohr)
        if geometry_is_traced:
            coords, weights = build_molecular_grid_from_spec(spec, level=grids_level)
            basis_cart = basis_from_molecule_spec(
                spec,
                basis=basis,
                max_l=max_l,
                precompute_eri_groups=True,
            )
        else:
            grid_ao_bundle = _cached_grid_ao_input_bundle(
                spec=spec,
                basis=basis,
                max_l=max_l,
                grids_level=grids_level,
                precompute_eri_groups=True,
                needs_ao_laplacian=True,
            )
            basis_cart = grid_ao_bundle.basis
            coords = grid_ao_bundle.coords
            weights = grid_ao_bundle.grid_weights
        overlap = overlap_matrix(basis_cart)
        hcore = build_hcore(basis_cart)
        if precompile_eri:
            _precompile_eri_kernels(
                basis_cart,
                engine="jit",
                chunk_size=int(precompile_eri_chunk_size),
            )
        eri = eri_tensor(basis_cart)
        if geometry_is_traced:
            ao_deriv1 = evaluate_cartesian_ao(basis_cart, coords, deriv=1)
            ao = ao_deriv1[0]
            ao_laplacian = evaluate_cartesian_ao(basis_cart, coords, deriv=2)[4]
        else:
            ao = grid_ao_bundle.ao
            ao_deriv1 = grid_ao_bundle.ao_deriv1
            ao_laplacian = grid_ao_bundle.ao_laplacian
        nuclear_repulsion = spec.nuclear_repulsion
        dipole_integrals = dipole_matrix(basis_cart)

    total_electrons = int(round(float(np.asarray(basis_cart.atom_charges).sum()))) - int(molecule_charge)
    nalpha, nbeta = _unrestricted_spin_electron_counts(total_electrons, int(spin))
    return UKSIntegralInputs(
        basis=basis_cart,
        overlap=overlap,
        hcore=hcore,
        eri=eri,
        nalpha=nalpha,
        nbeta=nbeta,
        nuclear_repulsion=nuclear_repulsion,
        coords=coords,
        grid_weights=weights,
        ao=ao,
        ao_deriv1=ao_deriv1,
        ao_laplacian=ao_laplacian,
        dipole_integrals=dipole_integrals,
        total_electrons=total_electrons,
        molecule_charge=molecule_charge,
        geometry_is_traced=geometry_is_traced,
        integral_backend=integral_backend_mode,
        grid_ao_backend=grid_ao_backend_mode,
    )


__all__ = [
    "RKSIntegralInputs",
    "UKSIntegralInputs",
    "build_rks_integral_inputs",
    "build_uks_integral_inputs",
]
