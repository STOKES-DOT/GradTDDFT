from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import warnings
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..data.basis import CartesianBasis, basis_from_molecule_spec
from ..data.grid import build_molecular_grid_from_spec
from ..data.grid_ao import evaluate_cartesian_ao, evaluate_cartesian_ao_with_derivatives
from ..data.integrals.jax import (
    build_hcore,
    dipole_matrix,
    eri_pair_matrix_packed,
    eri_tensor,
    overlap_matrix,
    overlap_hcore_matrices,
    precompile_eri_kernels,
)
from ..data.integrals.libcint import (
    LibcintGeometryGradPolicy,
    bind_libcint_integral_constant,
    build_libcint_mol,
    libcint_int1e_with_coords,
    libcint_int2e_full_with_coords,
    libcint_int2e_s4_with_coords,
    libcint_intor_name,
)
from ..data.integrals.gpu4pyscf import gpu4pyscf_int2e_full, gpu4pyscf_int2e_s4
from ..data.molecule import MoleculeSpec, parse_molecule_spec
from ..df import (
    eri_pair_matrix_to_df_factors_traceable,
    eri_to_df_factors_from_basis,
    true_df_factors_from_libcint_mol,
)
from ..xc_backend.jax_libxc import parse_xc, xc_type
from .core import _contains_jax_tracer
from .init_guess import restricted_init_guess_from_pyscf, unrestricted_init_guess_from_pyscf
from .rks import RKSConfig
from .uks import UKSConfig

_GRID_AO_INPUT_CACHE_MAXSIZE = 32
_GRID_AO_INPUT_CACHE: dict[tuple[Any, ...], Any] = {}
_LIBCINT_HOST_INTEGRAL_CACHE_MAXSIZE = 64
_LIBCINT_HOST_INTEGRAL_CACHE: dict[tuple[Any, ...], Any] = {}
_LIBCINT_INPUT_PARALLEL_WORKERS = 2


@dataclass(frozen=True)
class _GridAOInputBundle:
    basis: CartesianBasis
    coords: Array
    grid_weights: Array
    ao: Array
    ao_deriv1: Array
    ao_laplacian: Array | None


@dataclass(frozen=True)
class _BasisGridContext:
    basis: CartesianBasis
    coords: Array
    grid_weights: Array
    geometry_is_traced: bool
    grid_ao_bundle: _GridAOInputBundle | None


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
    init_density: Array | None = None
    init_mo_coeff: Array | None = None
    init_mo_occ: Array | None = None
    init_mo_energy: Array | None = None
    molecule_charge: int = 0
    geometry_is_traced: bool = False
    integral_backend: str = "cpu"
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
            "init_density": self.init_density,
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
    init_density_alpha: Array | None = None
    init_density_beta: Array | None = None
    init_mo_coeff_alpha: Array | None = None
    init_mo_coeff_beta: Array | None = None
    init_mo_occ_alpha: Array | None = None
    init_mo_occ_beta: Array | None = None
    init_mo_energy_alpha: Array | None = None
    init_mo_energy_beta: Array | None = None
    total_electrons: int = 0
    molecule_charge: int = 0
    geometry_is_traced: bool = False
    integral_backend: str = "cpu"
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
            "init_density_alpha": self.init_density_alpha,
            "init_density_beta": self.init_density_beta,
            "init_mo_coeff_alpha": self.init_mo_coeff_alpha,
            "init_mo_coeff_beta": self.init_mo_coeff_beta,
            "init_mo_occ_alpha": self.init_mo_occ_alpha,
            "init_mo_occ_beta": self.init_mo_occ_beta,
            "init_mo_energy_alpha": self.init_mo_energy_alpha,
            "init_mo_energy_beta": self.init_mo_energy_beta,
        }


def _resolve_config(config: RKSConfig | None, xc_spec: str | None) -> tuple[RKSConfig, str]:
    xc_spec_resolved = str(xc_spec if xc_spec is not None else (config.xc_spec if config is not None else "pbe"))
    parse_xc(xc_spec_resolved)
    cfg = RKSConfig(xc_spec=xc_spec_resolved) if config is None else config
    if cfg.xc_spec != xc_spec_resolved:
        cfg = replace(cfg, xc_spec=xc_spec_resolved)
    return cfg, xc_spec_resolved


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


def _cache_libcint_host_integral(
    key: tuple[Any, ...],
    value: Any,
) -> None:
    if len(_LIBCINT_HOST_INTEGRAL_CACHE) >= _LIBCINT_HOST_INTEGRAL_CACHE_MAXSIZE:
        _LIBCINT_HOST_INTEGRAL_CACHE.pop(next(iter(_LIBCINT_HOST_INTEGRAL_CACHE)))
    _LIBCINT_HOST_INTEGRAL_CACHE[key] = value


def _cached_libcint_host_integral(
    *,
    mol: Any,
    integral_name: str,
    geometry_anchor: Array,
    geometry_grad_policy: str,
    loader: Any,
) -> Array:
    key = (id(mol), str(integral_name), str(geometry_grad_policy))
    cached = _LIBCINT_HOST_INTEGRAL_CACHE.get(key)
    if cached is not None:
        return cached
    value = bind_libcint_integral_constant(
        loader(),
        geometry_anchor=geometry_anchor,
        integral_name=integral_name,
        geometry_grad_policy=geometry_grad_policy,
    )
    _cache_libcint_host_integral(key, value)
    return value


def _gpu4pyscf_eri_pair_matrix(
    *,
    atom: Any,
    basis: Any,
    unit: str,
    charge: int,
    spin: int,
    cart: bool,
    verbose: int,
    mol_kwargs: dict[str, Any],
) -> Array:
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
    return jnp.asarray(gpu4pyscf_int2e_s4(mol))


def _gpu4pyscf_eri_tensor(
    *,
    atom: Any,
    basis: Any,
    unit: str,
    charge: int,
    spin: int,
    cart: bool,
    verbose: int,
    mol_kwargs: dict[str, Any],
) -> Array:
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
    return jnp.asarray(gpu4pyscf_int2e_full(mol))


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


def _resolve_integral_input_modes(
    *,
    integral_backend: Literal["jax", "cpu", "gpu", "libcint"],
    grid_ao_backend: Literal["jax"],
    libcint_geometry_grad_policy: LibcintGeometryGradPolicy,
) -> tuple[str, str, str]:
    integral_backend_mode = str(integral_backend).lower()
    if integral_backend_mode == "libcint":
        integral_backend_mode = "cpu"
    if integral_backend_mode not in {"jax", "cpu", "gpu"}:
        raise ValueError(
            f"Unsupported integral_backend={integral_backend!r}. Expected 'jax', 'cpu', or 'gpu'."
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
    return integral_backend_mode, grid_ao_backend_mode, libcint_grad_policy_mode


def _prepare_basis_grid_context(
    *,
    spec: MoleculeSpec,
    basis: Any,
    max_l: int,
    grids_level: int,
    precompute_eri_groups: bool,
    needs_ao_laplacian: bool,
) -> _BasisGridContext:
    geometry_is_traced = _contains_jax_tracer(spec.coords_bohr)
    if geometry_is_traced:
        coords, weights = build_molecular_grid_from_spec(spec, level=grids_level)
        basis_cart = basis_from_molecule_spec(
            spec,
            basis=basis,
            max_l=max_l,
            precompute_eri_groups=precompute_eri_groups,
        )
        return _BasisGridContext(
            basis=basis_cart,
            coords=coords,
            grid_weights=weights,
            geometry_is_traced=True,
            grid_ao_bundle=None,
        )
    grid_ao_bundle = _cached_grid_ao_input_bundle(
        spec=spec,
        basis=basis,
        max_l=max_l,
        grids_level=grids_level,
        precompute_eri_groups=precompute_eri_groups,
        needs_ao_laplacian=needs_ao_laplacian,
    )
    return _BasisGridContext(
        basis=grid_ao_bundle.basis,
        coords=grid_ao_bundle.coords,
        grid_weights=grid_ao_bundle.grid_weights,
        geometry_is_traced=False,
        grid_ao_bundle=grid_ao_bundle,
    )


def _grid_ao_payload(
    context: _BasisGridContext,
    *,
    needs_ao_laplacian: bool,
) -> tuple[Array, Array, Array | None]:
    if context.geometry_is_traced:
        deriv_order = 2 if needs_ao_laplacian else 1
        ao, ao_derivs = evaluate_cartesian_ao_with_derivatives(
            context.basis,
            context.coords,
            deriv=deriv_order,
        )
        if needs_ao_laplacian:
            return ao, ao_derivs[:4], ao_derivs[4]
        return ao, ao_derivs, None
    bundle = context.grid_ao_bundle
    if bundle is None:
        raise ValueError("Non-traced basis/grid context is missing cached AO data.")
    return bundle.ao, bundle.ao_deriv1, bundle.ao_laplacian


def _libcint_one_electron_with_coords(
    *,
    coords_bohr: Array,
    symbols: tuple[str, ...],
    basis: str,
    charge: int,
    spin: int,
    cart: bool,
    verbose: int,
    geometry_grad_policy: str,
    include_dipole_integrals: bool,
) -> tuple[Array, Array, Array | None]:
    intor_args = (
        coords_bohr,
        symbols,
        basis,
        int(charge),
        int(spin),
        bool(cart),
        int(verbose),
    )
    overlap = libcint_int1e_with_coords(
        *intor_args,
        "int1e_ovlp",
        None,
        geometry_grad_policy,
    )
    kinetic = libcint_int1e_with_coords(
        *intor_args,
        "int1e_kin",
        None,
        geometry_grad_policy,
    )
    v_nuc = libcint_int1e_with_coords(
        *intor_args,
        "int1e_nuc",
        None,
        geometry_grad_policy,
    )
    dipole_integrals = None
    if include_dipole_integrals:
        dipole_integrals = libcint_int1e_with_coords(
            *intor_args,
            "int1e_r",
            3,
            geometry_grad_policy,
        )
    return overlap, kinetic + v_nuc, dipole_integrals


def _libcint_one_electron_from_mol(
    *,
    mol: Any,
    geometry_anchor: Array,
    geometry_grad_policy: str,
    include_dipole_integrals: bool,
) -> tuple[Array, Array, Array | None]:
    overlap = _cached_libcint_host_integral(
        mol=mol,
        integral_name="int1e_ovlp",
        geometry_anchor=geometry_anchor,
        geometry_grad_policy=geometry_grad_policy,
        loader=lambda: np.asarray(
            mol.intor_symmetric(libcint_intor_name(mol, "int1e_ovlp")),
            dtype=float,
        ),
    )
    hcore = _cached_libcint_host_integral(
        mol=mol,
        integral_name="int1e_kin+int1e_nuc",
        geometry_anchor=geometry_anchor,
        geometry_grad_policy=geometry_grad_policy,
        loader=lambda: np.asarray(
            mol.intor_symmetric(libcint_intor_name(mol, "int1e_kin")),
            dtype=float,
        )
        + np.asarray(
            mol.intor_symmetric(libcint_intor_name(mol, "int1e_nuc")),
            dtype=float,
        ),
    )
    dipole_integrals = None
    if include_dipole_integrals:
        dipole_integrals = _cached_libcint_host_integral(
            mol=mol,
            integral_name="int1e_r",
            geometry_anchor=geometry_anchor,
            geometry_grad_policy=geometry_grad_policy,
            loader=lambda: np.asarray(
                mol.intor_symmetric(libcint_intor_name(mol, "int1e_r"), comp=3),
                dtype=float,
            ),
        )
    return overlap, hcore, dipole_integrals


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


def _build_rks_inputs_from_cpu_backbone(
    *,
    atom: Any,
    basis: Any,
    cfg: RKSConfig,
    xc_spec_resolved: str,
    unit: str,
    charge: int,
    spin: int,
    cart: bool,
    grids_level: int,
    max_l: int,
    precompile_eri: bool,
    include_dipole_integrals: bool,
    init_guess: Any,
    chkfile: str | None,
    init_guess_sap_basis: Any | None,
    init_guess_chkfile_project: bool | None,
    verbose: int,
    integral_backend_mode: str,
    grid_ao_backend_mode: str,
    libcint_grad_policy_mode: str,
    mol_kwargs: dict[str, Any],
) -> RKSIntegralInputs:
    needs_ao_laplacian = xc_type(xc_spec_resolved) == "MGGA"
    precompute_eri_groups = cfg.jk_backend == "direct"
    executor: ThreadPoolExecutor | None = None

    try:
        if isinstance(atom, MoleculeSpec) and _contains_jax_tracer(atom.coords_bohr):
            if not isinstance(basis, str):
                raise TypeError("Traceable libcint geometry currently supports named basis strings only.")
            spec = atom
            molecule_charge = int(spec.charge)
            basis_grid = _prepare_basis_grid_context(
                spec=spec,
                basis=basis,
                max_l=max_l,
                grids_level=grids_level,
                precompute_eri_groups=precompute_eri_groups,
                needs_ao_laplacian=needs_ao_laplacian,
            )
            geometry_is_traced = basis_grid.geometry_is_traced
            basis_cart = basis_grid.basis
            coords_bohr = jnp.asarray(spec.coords_bohr)
            overlap, hcore, dipole_integrals = _libcint_one_electron_with_coords(
                coords_bohr=coords_bohr,
                symbols=tuple(spec.symbols),
                basis=str(basis),
                charge=molecule_charge,
                spin=int(spec.spin),
                cart=bool(cart),
                verbose=int(verbose),
                geometry_grad_policy=libcint_grad_policy_mode,
                include_dipole_integrals=include_dipole_integrals,
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
            elif cfg.jk_backend != "direct":
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
            nuclear_repulsion = spec.nuclear_repulsion
            mol = None
        else:
            spec = atom if isinstance(atom, MoleculeSpec) else parse_molecule_spec(atom, unit=unit, charge=charge, spin=spin)
            molecule_charge = int(spec.charge)
            executor = ThreadPoolExecutor(max_workers=_LIBCINT_INPUT_PARALLEL_WORKERS)
            mol_future = executor.submit(
                build_libcint_mol,
                atom=atom,
                basis=basis,
                unit=unit,
                charge=int(charge),
                spin=int(spin),
                cart=bool(cart),
                verbose=int(verbose),
                **mol_kwargs,
            )
            basis_grid = _prepare_basis_grid_context(
                spec=spec,
                basis=basis,
                max_l=max_l,
                grids_level=grids_level,
                precompute_eri_groups=precompute_eri_groups,
                needs_ao_laplacian=needs_ao_laplacian,
            )
            geometry_is_traced = basis_grid.geometry_is_traced
            basis_cart = basis_grid.basis
            mol = mol_future.result()
            geometry_anchor = np.asarray(mol.atom_coords(), dtype=float)
            overlap, hcore, dipole_integrals = _libcint_one_electron_from_mol(
                mol=mol,
                geometry_anchor=geometry_anchor,
                geometry_grad_policy=libcint_grad_policy_mode,
                include_dipole_integrals=include_dipole_integrals,
            )
            if precompile_eri:
                warnings.warn(
                    "precompile_eri is ignored when integral_backend='cpu'.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            eri_name = libcint_intor_name(mol, "int2e")
            eri = None
            eri_pair_matrix = None
            df_factors = None
            nuclear_repulsion = spec.nuclear_repulsion

        ao, ao_deriv1, ao_laplacian = _grid_ao_payload(
            basis_grid,
            needs_ao_laplacian=needs_ao_laplacian,
        )
        coords = basis_grid.coords
        weights = basis_grid.grid_weights
        nelectron = int(basis_cart.atom_charges.sum()) - molecule_charge
        initial_guess = restricted_init_guess_from_pyscf(
            atom=spec,
            basis=basis,
            unit=unit,
            charge=charge,
            spin=spin,
            cart=bool(cart),
            verbose=int(verbose),
            xc_spec=xc_spec_resolved,
            init_guess=init_guess,
            sap_basis=init_guess_sap_basis,
            chkfile=chkfile,
            chkfile_project=init_guess_chkfile_project,
            geometry_is_traced=geometry_is_traced,
            dtype=hcore.dtype,
            libcint_mol=None if geometry_is_traced else mol,
        )

        direct_basis = basis_cart if cfg.jk_backend == "direct" else None
        if not geometry_is_traced and cfg.jk_backend == "df":
            df_factors = _cached_libcint_host_integral(
                mol=mol,
                integral_name="df_cholesky_eri",
                geometry_anchor=geometry_anchor,
                geometry_grad_policy=libcint_grad_policy_mode,
                loader=lambda: true_df_factors_from_libcint_mol(mol),
            )
        elif not geometry_is_traced and cfg.jk_backend != "direct":
            eri_pair_matrix = _cached_libcint_host_integral(
                mol=mol,
                integral_name=f"{eri_name}_s4",
                geometry_anchor=geometry_anchor,
                geometry_grad_policy=libcint_grad_policy_mode,
                loader=lambda: np.asarray(mol.intor(eri_name, aosym="s4"), dtype=float),
            )

        inputs = RKSIntegralInputs(
            basis=basis_cart,
            overlap=overlap,
            hcore=hcore,
            eri=eri,
            eri_pair_matrix=eri_pair_matrix,
            df_factors=df_factors,
            direct_basis=direct_basis,
            nelectron=nelectron,
            nuclear_repulsion=nuclear_repulsion,
            coords=coords,
            grid_weights=weights,
            ao=ao,
            ao_deriv1=ao_deriv1,
            ao_laplacian=ao_laplacian,
            dipole_integrals=dipole_integrals,
            init_density=initial_guess.density,
            init_mo_coeff=None,
            init_mo_occ=None,
            init_mo_energy=None,
            molecule_charge=molecule_charge,
            geometry_is_traced=geometry_is_traced,
            integral_backend=integral_backend_mode,
            grid_ao_backend=grid_ao_backend_mode,
        )
        return inputs
    finally:
        if executor is not None:
            executor.shutdown(wait=True)


def _build_rks_inputs_from_jax_backbone(
    *,
    atom: Any,
    basis: Any,
    cfg: RKSConfig,
    xc_spec_resolved: str,
    unit: str,
    charge: int,
    spin: int,
    grids_level: int,
    max_l: int,
    precompile_eri: bool,
    precompile_eri_chunk_size: int,
    include_dipole_integrals: bool,
    init_guess: Any,
    chkfile: str | None,
    init_guess_sap_basis: Any | None,
    init_guess_chkfile_project: bool | None,
    verbose: int,
    integral_backend_mode: str,
    grid_ao_backend_mode: str,
    _precompile_eri_kernels: Any,
) -> RKSIntegralInputs:
    spec = parse_molecule_spec(atom, unit=unit, charge=charge, spin=spin)
    molecule_charge = int(spec.charge)
    needs_ao_laplacian = xc_type(xc_spec_resolved) == "MGGA"
    basis_grid = _prepare_basis_grid_context(
        spec=spec,
        basis=basis,
        max_l=max_l,
        grids_level=grids_level,
        precompute_eri_groups=True,
        needs_ao_laplacian=needs_ao_laplacian,
    )
    geometry_is_traced = basis_grid.geometry_is_traced
    basis_cart = basis_grid.basis
    overlap, hcore = overlap_hcore_matrices(basis_cart, backend="jax")
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
    ao, ao_deriv1, ao_laplacian = _grid_ao_payload(
        basis_grid,
        needs_ao_laplacian=needs_ao_laplacian,
    )
    dipole_integrals = dipole_matrix(basis_cart) if include_dipole_integrals else None
    nelectron = int(basis_cart.atom_charges.sum()) - molecule_charge
    initial_guess = restricted_init_guess_from_pyscf(
        atom=spec,
        basis=basis,
        unit=unit,
        charge=charge,
        spin=spin,
        cart=True,
        verbose=int(verbose),
        xc_spec=xc_spec_resolved,
        init_guess=init_guess,
        sap_basis=init_guess_sap_basis,
        chkfile=chkfile,
        chkfile_project=init_guess_chkfile_project,
        geometry_is_traced=geometry_is_traced,
        dtype=hcore.dtype,
        libcint_mol=None,
    )
    inputs = RKSIntegralInputs(
        basis=basis_cart,
        overlap=overlap,
        hcore=hcore,
        eri=eri,
        eri_pair_matrix=eri_pair_matrix,
        df_factors=df_factors,
        direct_basis=basis_cart if cfg.jk_backend == "direct" else None,
        nelectron=nelectron,
        nuclear_repulsion=spec.nuclear_repulsion,
        coords=basis_grid.coords,
        grid_weights=basis_grid.grid_weights,
        ao=ao,
        ao_deriv1=ao_deriv1,
        ao_laplacian=ao_laplacian,
        dipole_integrals=dipole_integrals,
        init_density=initial_guess.density,
        init_mo_coeff=None,
        init_mo_occ=None,
        init_mo_energy=None,
        molecule_charge=molecule_charge,
        geometry_is_traced=geometry_is_traced,
        integral_backend=integral_backend_mode,
        grid_ao_backend=grid_ao_backend_mode,
    )
    return inputs


def _build_uks_inputs_from_cpu_backbone(
    *,
    atom: Any,
    basis: Any,
    xc_spec_resolved: str,
    unit: str,
    charge: int,
    spin: int,
    cart: bool,
    grids_level: int,
    max_l: int,
    precompile_eri: bool,
    init_guess: Any,
    chkfile: str | None,
    init_guess_sap_basis: Any | None,
    init_guess_chkfile_project: bool | None,
    verbose: int,
    integral_backend_mode: str,
    grid_ao_backend_mode: str,
    libcint_grad_policy_mode: str,
    mol_kwargs: dict[str, Any],
) -> UKSIntegralInputs:
    skip_cpu_eri = integral_backend_mode == "gpu"
    if isinstance(atom, MoleculeSpec):
        if not isinstance(basis, str):
            raise TypeError("Traceable libcint geometry currently supports named basis strings only.")
        spec = atom
        molecule_charge = int(spec.charge)
        basis_grid = _prepare_basis_grid_context(
            spec=spec,
            basis=basis,
            max_l=max_l,
            grids_level=grids_level,
            precompute_eri_groups=False,
            needs_ao_laplacian=True,
        )
        geometry_is_traced = basis_grid.geometry_is_traced
        basis_cart = basis_grid.basis
        coords_bohr = jnp.asarray(spec.coords_bohr)
        overlap, hcore, dipole_integrals = _libcint_one_electron_with_coords(
            coords_bohr=coords_bohr,
            symbols=tuple(spec.symbols),
            basis=str(basis),
            charge=int(spec.charge),
            spin=int(spec.spin),
            cart=bool(cart),
            verbose=int(verbose),
            geometry_grad_policy=libcint_grad_policy_mode,
            include_dipole_integrals=True,
        )
        if precompile_eri:
            warnings.warn(
                "precompile_eri is ignored for libcint UKS input construction.",
                RuntimeWarning,
                stacklevel=2,
            )
        eri = (
            jnp.zeros((0, 0, 0, 0), dtype=hcore.dtype)
            if skip_cpu_eri
            else libcint_int2e_full_with_coords(
                coords_bohr,
                tuple(spec.symbols),
                str(basis),
                int(spec.charge),
                int(spec.spin),
                bool(cart),
                int(verbose),
                libcint_grad_policy_mode,
            )
        )
        nuclear_repulsion = spec.nuclear_repulsion
    else:
        spec = atom if isinstance(atom, MoleculeSpec) else parse_molecule_spec(atom, unit=unit, charge=charge, spin=spin)
        molecule_charge = int(spec.charge)
        basis_grid = _prepare_basis_grid_context(
            spec=spec,
            basis=basis,
            max_l=max_l,
            grids_level=grids_level,
            precompute_eri_groups=False,
            needs_ao_laplacian=True,
        )
        geometry_is_traced = basis_grid.geometry_is_traced
        basis_cart = basis_grid.basis
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
        overlap, hcore, dipole_integrals = _libcint_one_electron_from_mol(
            mol=mol,
            geometry_anchor=geometry_anchor,
            geometry_grad_policy=libcint_grad_policy_mode,
            include_dipole_integrals=True,
        )
        if precompile_eri:
            warnings.warn(
                "precompile_eri is ignored for libcint UKS input construction.",
                RuntimeWarning,
                stacklevel=2,
            )
        eri = jnp.zeros((0, 0, 0, 0), dtype=hcore.dtype)
        if not skip_cpu_eri:
            eri_name = libcint_intor_name(mol, "int2e")
            eri = _cached_libcint_host_integral(
                mol=mol,
                integral_name=eri_name,
                geometry_anchor=geometry_anchor,
                geometry_grad_policy=libcint_grad_policy_mode,
                loader=lambda: jnp.asarray(mol.intor(eri_name)),
        )
        nuclear_repulsion = spec.nuclear_repulsion

    ao, ao_deriv1, ao_laplacian = _grid_ao_payload(
        basis_grid,
        needs_ao_laplacian=True,
    )
    total_electrons = int(round(float(np.asarray(basis_cart.atom_charges).sum()))) - int(molecule_charge)
    nalpha, nbeta = _unrestricted_spin_electron_counts(total_electrons, int(spin))
    initial_guess = unrestricted_init_guess_from_pyscf(
        atom=spec,
        basis=basis,
        unit=unit,
        charge=charge,
        spin=spin,
        cart=bool(cart),
        verbose=int(verbose),
        xc_spec=xc_spec_resolved,
        init_guess=init_guess,
        sap_basis=init_guess_sap_basis,
        chkfile=chkfile,
        chkfile_project=init_guess_chkfile_project,
        geometry_is_traced=geometry_is_traced,
        dtype=hcore.dtype,
        libcint_mol=None if geometry_is_traced else mol,
    )
    return UKSIntegralInputs(
        basis=basis_cart,
        overlap=overlap,
        hcore=hcore,
        eri=eri,
        nalpha=nalpha,
        nbeta=nbeta,
        nuclear_repulsion=nuclear_repulsion,
        coords=basis_grid.coords,
        grid_weights=basis_grid.grid_weights,
        ao=ao,
        ao_deriv1=ao_deriv1,
        ao_laplacian=ao_laplacian,
        dipole_integrals=dipole_integrals,
        init_density_alpha=initial_guess.density_alpha,
        init_density_beta=initial_guess.density_beta,
        total_electrons=total_electrons,
        molecule_charge=molecule_charge,
        geometry_is_traced=geometry_is_traced,
        integral_backend=integral_backend_mode,
        grid_ao_backend=grid_ao_backend_mode,
    )


def _build_uks_inputs_from_jax_backbone(
    *,
    atom: Any,
    basis: Any,
    xc_spec_resolved: str,
    unit: str,
    charge: int,
    spin: int,
    grids_level: int,
    max_l: int,
    precompile_eri: bool,
    init_guess: Any,
    chkfile: str | None,
    init_guess_sap_basis: Any | None,
    init_guess_chkfile_project: bool | None,
    precompile_eri_chunk_size: int,
    verbose: int,
    integral_backend_mode: str,
    grid_ao_backend_mode: str,
    _precompile_eri_kernels: Any,
) -> UKSIntegralInputs:
    spec = parse_molecule_spec(atom, unit=unit, charge=charge, spin=spin)
    molecule_charge = int(spec.charge)
    basis_grid = _prepare_basis_grid_context(
        spec=spec,
        basis=basis,
        max_l=max_l,
        grids_level=grids_level,
        precompute_eri_groups=True,
        needs_ao_laplacian=True,
    )
    geometry_is_traced = basis_grid.geometry_is_traced
    basis_cart = basis_grid.basis
    overlap = overlap_matrix(basis_cart)
    hcore = build_hcore(basis_cart)
    if precompile_eri:
        _precompile_eri_kernels(
            basis_cart,
            engine="jit",
            chunk_size=int(precompile_eri_chunk_size),
        )
    eri = eri_tensor(basis_cart)
    ao, ao_deriv1, ao_laplacian = _grid_ao_payload(
        basis_grid,
        needs_ao_laplacian=True,
    )
    dipole_integrals = dipole_matrix(basis_cart)
    total_electrons = int(round(float(np.asarray(basis_cart.atom_charges).sum()))) - int(molecule_charge)
    nalpha, nbeta = _unrestricted_spin_electron_counts(total_electrons, int(spin))
    initial_guess = unrestricted_init_guess_from_pyscf(
        atom=spec,
        basis=basis,
        unit=unit,
        charge=charge,
        spin=spin,
        cart=True,
        verbose=int(verbose),
        xc_spec=xc_spec_resolved,
        init_guess=init_guess,
        sap_basis=init_guess_sap_basis,
        chkfile=chkfile,
        chkfile_project=init_guess_chkfile_project,
        geometry_is_traced=geometry_is_traced,
        dtype=hcore.dtype,
        libcint_mol=None,
    )
    return UKSIntegralInputs(
        basis=basis_cart,
        overlap=overlap,
        hcore=hcore,
        eri=eri,
        nalpha=nalpha,
        nbeta=nbeta,
        nuclear_repulsion=spec.nuclear_repulsion,
        coords=basis_grid.coords,
        grid_weights=basis_grid.grid_weights,
        ao=ao,
        ao_deriv1=ao_deriv1,
        ao_laplacian=ao_laplacian,
        dipole_integrals=dipole_integrals,
        init_density_alpha=initial_guess.density_alpha,
        init_density_beta=initial_guess.density_beta,
        total_electrons=total_electrons,
        molecule_charge=molecule_charge,
        geometry_is_traced=geometry_is_traced,
        integral_backend=integral_backend_mode,
        grid_ao_backend=grid_ao_backend_mode,
    )


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
    integral_backend: Literal["jax", "cpu", "gpu", "libcint"] = "cpu",
    libcint_geometry_grad_policy: LibcintGeometryGradPolicy = "analytic",
    precompile_eri: bool = False,
    precompile_eri_chunk_size: int = 512,
    _precompile_eri_kernels: Any = precompile_eri_kernels,
    include_dipole_integrals: bool = True,
    init_guess: Any = "minao",
    chkfile: str | None = None,
    init_guess_sap_basis: Any | None = None,
    init_guess_chkfile_project: bool | None = None,
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
    integral_backend_mode, grid_ao_backend_mode, libcint_grad_policy_mode = _resolve_integral_input_modes(
        integral_backend=integral_backend,
        grid_ao_backend=grid_ao_backend,
        libcint_geometry_grad_policy=libcint_geometry_grad_policy,
    )
    if integral_backend_mode == "cpu":
        return _build_rks_inputs_from_cpu_backbone(
            atom=atom,
            basis=basis,
            cfg=cfg,
            xc_spec_resolved=xc_spec_resolved,
            unit=unit,
            charge=charge,
            spin=spin,
            cart=bool(cart),
            grids_level=grids_level,
            max_l=max_l,
            precompile_eri=precompile_eri,
            include_dipole_integrals=include_dipole_integrals,
            init_guess=init_guess,
            chkfile=chkfile,
            init_guess_sap_basis=init_guess_sap_basis,
            init_guess_chkfile_project=init_guess_chkfile_project,
            verbose=verbose,
            integral_backend_mode=integral_backend_mode,
            grid_ao_backend_mode=grid_ao_backend_mode,
            libcint_grad_policy_mode=libcint_grad_policy_mode,
            mol_kwargs=dict(mol_kwargs),
        )
    inputs = _build_rks_inputs_from_jax_backbone(
        atom=atom,
        basis=basis,
        cfg=cfg,
        xc_spec_resolved=xc_spec_resolved,
        unit=unit,
        charge=charge,
        spin=spin,
        grids_level=grids_level,
        max_l=max_l,
        precompile_eri=precompile_eri,
        precompile_eri_chunk_size=precompile_eri_chunk_size,
        include_dipole_integrals=include_dipole_integrals,
        init_guess=init_guess,
        chkfile=chkfile,
        init_guess_sap_basis=init_guess_sap_basis,
        init_guess_chkfile_project=init_guess_chkfile_project,
        verbose=verbose,
        integral_backend_mode=integral_backend_mode,
        grid_ao_backend_mode=grid_ao_backend_mode,
        _precompile_eri_kernels=_precompile_eri_kernels,
    )
    if integral_backend_mode == "gpu" and cfg.jk_backend == "full":
        inputs = replace(
            inputs,
            eri_pair_matrix=_gpu4pyscf_eri_pair_matrix(
                atom=atom,
                basis=basis,
                unit=unit,
                charge=charge,
                spin=spin,
                cart=bool(cart),
                verbose=verbose,
                mol_kwargs=dict(mol_kwargs),
            ),
        )
    return inputs


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
    integral_backend: Literal["jax", "cpu", "gpu", "libcint"] = "cpu",
    libcint_geometry_grad_policy: LibcintGeometryGradPolicy = "error",
    precompile_eri: bool = False,
    precompile_eri_chunk_size: int = 512,
    _precompile_eri_kernels: Any = precompile_eri_kernels,
    init_guess: Any = "minao",
    chkfile: str | None = None,
    init_guess_sap_basis: Any | None = None,
    init_guess_chkfile_project: bool | None = None,
    verbose: int = 0,
    **mol_kwargs: Any,
) -> UKSIntegralInputs:
    """Build integral/grid inputs for the unrestricted KS SCF kernel."""

    if isinstance(atom, MoleculeSpec):
        charge = int(atom.charge)
        spin = int(atom.spin)
    if not bool(cart):
        raise NotImplementedError("build_uks_integral_inputs currently supports cart=True only.")

    _, xc_spec_resolved = _resolve_uks_config(config, xc_spec)
    integral_backend_mode, grid_ao_backend_mode, libcint_grad_policy_mode = _resolve_integral_input_modes(
        integral_backend=integral_backend,
        grid_ao_backend=grid_ao_backend,
        libcint_geometry_grad_policy=libcint_geometry_grad_policy,
    )
    if integral_backend_mode in {"cpu", "gpu"}:
        inputs = _build_uks_inputs_from_cpu_backbone(
            atom=atom,
            basis=basis,
            xc_spec_resolved=xc_spec_resolved,
            unit=unit,
            charge=charge,
            spin=spin,
            cart=bool(cart),
            grids_level=grids_level,
            max_l=max_l,
            precompile_eri=precompile_eri,
            init_guess=init_guess,
            chkfile=chkfile,
            init_guess_sap_basis=init_guess_sap_basis,
            init_guess_chkfile_project=init_guess_chkfile_project,
            verbose=verbose,
            integral_backend_mode=integral_backend_mode,
            grid_ao_backend_mode=grid_ao_backend_mode,
            libcint_grad_policy_mode=libcint_grad_policy_mode,
            mol_kwargs=dict(mol_kwargs),
        )
        if integral_backend_mode == "gpu":
            inputs = replace(
                inputs,
                eri=_gpu4pyscf_eri_tensor(
                    atom=atom,
                    basis=basis,
                    unit=unit,
                    charge=charge,
                    spin=spin,
                    cart=bool(cart),
                    verbose=verbose,
                    mol_kwargs=dict(mol_kwargs),
                ),
            )
        return inputs
    inputs = _build_uks_inputs_from_jax_backbone(
        atom=atom,
        basis=basis,
        xc_spec_resolved=xc_spec_resolved,
        unit=unit,
        charge=charge,
        spin=spin,
        grids_level=grids_level,
        max_l=max_l,
        precompile_eri=precompile_eri,
        init_guess=init_guess,
        chkfile=chkfile,
        init_guess_sap_basis=init_guess_sap_basis,
        init_guess_chkfile_project=init_guess_chkfile_project,
        precompile_eri_chunk_size=precompile_eri_chunk_size,
        verbose=verbose,
        integral_backend_mode=integral_backend_mode,
        grid_ao_backend_mode=grid_ao_backend_mode,
        _precompile_eri_kernels=_precompile_eri_kernels,
    )
    return inputs


__all__ = [
    "RKSIntegralInputs",
    "UKSIntegralInputs",
    "build_rks_integral_inputs",
    "build_uks_integral_inputs",
]
