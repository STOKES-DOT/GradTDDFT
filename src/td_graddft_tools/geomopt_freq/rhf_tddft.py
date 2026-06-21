from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import jax.numpy as jnp
from jaxtyping import Array

from td_graddft.data.basis import CartesianAO, CartesianBasis, cartesian_angular_tuples
from td_graddft.data.integrals import build_hcore, eri_tensor, overlap_matrix
from td_graddft.scf import RHFConfig
from td_graddft.scf.rhf import (
    _build_density,
    _build_fock,
    _diagonalize_fock,
    _diis_extrapolate,
    _electronic_energy,
    _orthogonalizer,
    nuclear_repulsion_energy,
)
from td_graddft.tddft import RestrictedCasidaTDDFT
from td_graddft.tddft.eigensolvers import PYSCF_TD_DAVIDSON_MAX_CYCLE
from td_graddft.tddft.eigensolvers import PYSCF_TD_DAVIDSON_TOL
from td_graddft.tddft.eigensolvers import PYSCF_TD_POSITIVE_EIG_THRESHOLD

from .objectives import EnergySurface
from .optimization import (
    GeometryOptimizationConfig,
    GeometryOptimizationResult,
    run_geometry_optimization,
)

ANGSTROM_TO_BOHR = 1.8897261254578281
BOHR_TO_ANGSTROM = 1.0 / ANGSTROM_TO_BOHR


@dataclass(frozen=True)
class _HFOnlyResponseFunctional:
    exact_exchange_fraction: float = 1.0

    def local_kernel(self, density: Array) -> Array:
        return jnp.zeros_like(density)


_HF_ONLY_RESPONSE_FUNCTIONAL = _HFOnlyResponseFunctional()


@dataclass(frozen=True)
class CartesianBasisTemplate:
    """Static AO metadata plus atom-to-AO mapping for coordinate updates."""

    atom_charges: Array
    nelectron: int
    ao_atom_indices: tuple[int, ...]
    angulars: tuple[tuple[int, int, int], ...]
    exponents: tuple[Array, ...]
    coefficients: tuple[Array, ...]

    @property
    def nao(self) -> int:
        return len(self.ao_atom_indices)

    @property
    def natom(self) -> int:
        return int(jnp.asarray(self.atom_charges).shape[0])


@dataclass(frozen=True)
class RHFExcitedStateSurfaceConfig:
    """Configuration for differentiable RHF + TDA/Casida excited-state surfaces.

    Notes:
    - The ground-state reference is a fixed-length, fully differentiable RHF loop
      built from JAX AO integrals.
    - The excited-state response uses the local TDDFT module in HF-only mode
      (`exact_exchange_fraction=1` with zero local kernel), which corresponds to a
      CIS/TDHF-like response model.
    - `RHFConfig.conv_tol` and `RHFConfig.conv_tol_density` are not used here
      because the SCF loop is intentionally unrolled for a fixed number of cycles
      so that geometry gradients remain traceable.
    """

    scf: RHFConfig = field(default_factory=RHFConfig)
    state_index: int = 0
    response_method: Literal["tda", "casida"] = "tda"
    coordinate_unit: Literal["angstrom", "bohr"] = "angstrom"
    eigensolver: Literal["auto", "davidson"] = "auto"
    excitation_threshold: float = PYSCF_TD_POSITIVE_EIG_THRESHOLD
    matrix_eps: float = 1e-10
    davidson_tol: float = PYSCF_TD_DAVIDSON_TOL
    davidson_max_iter: int = PYSCF_TD_DAVIDSON_MAX_CYCLE
    davidson_max_subspace: int | None = None


@dataclass(frozen=True)
class _GridReference:
    weights: Array


@dataclass(frozen=True)
class _RestrictedResponseReference:
    ao: Array
    grid: _GridReference
    rep_tensor: Array
    mo_coeff: Array
    mo_occ: Array
    mo_energy: Array
    rdm1: Array
    exact_exchange_fraction: float = 1.0


@dataclass(frozen=True)
class _TraceableRHFResult:
    total_energy: Array
    mo_energy: Array
    mo_coeff: Array
    mo_occ: Array
    density_matrix: Array
    rep_tensor: Array


def _clone_pyscf_mol_cart(mol: Any) -> Any:
    """Create a cartesian-AO PySCF Mole clone from an existing Mole."""

    try:
        from pyscf import gto
    except ModuleNotFoundError as exc:
        raise ImportError("PySCF is required for RHF excited-state geometry surfaces.") from exc

    atom_bohr = [
        (mol.atom_symbol(i), tuple(float(x) for x in mol.atom_coord(i)))
        for i in range(mol.natm)
    ]
    return gto.M(
        atom=atom_bohr,
        unit="Bohr",
        basis=mol.basis,
        ecp=getattr(mol, "ecp", None),
        charge=int(getattr(mol, "charge", 0)),
        spin=int(getattr(mol, "spin", 0)),
        cart=True,
        verbose=0,
    )


def build_cartesian_basis_template_from_pyscf_mol(
    mol: Any,
    *,
    max_l: int = 3,
) -> CartesianBasisTemplate:
    """Extract differentiable AO metadata from a PySCF Mole."""

    mol_cart = mol if bool(getattr(mol, "cart", False)) else _clone_pyscf_mol_cart(mol)
    if int(getattr(mol_cart, "spin", 0)) != 0:
        raise NotImplementedError("RHF excited-state geometry surfaces currently require spin=0.")

    ao_atom_indices: list[int] = []
    angulars: list[tuple[int, int, int]] = []
    exponents: list[Array] = []
    coefficients: list[Array] = []

    for ib in range(mol_cart.nbas):
        l = int(mol_cart.bas_angular(ib))
        if l > max_l:
            raise NotImplementedError(
                f"Current JAX integral implementation supports l<= {max_l}, got l={l}."
            )
        atom_idx = int(mol_cart.bas_atom(ib))
        shell_exponents = jnp.asarray(mol_cart.bas_exp(ib))
        ctr_coeff = jnp.asarray(mol_cart.bas_ctr_coeff(ib))
        if ctr_coeff.ndim == 1:
            ctr_coeff = ctr_coeff[:, None]

        for ctr in range(ctr_coeff.shape[1]):
            coeff = ctr_coeff[:, ctr]
            for angular in cartesian_angular_tuples(l):
                ao_atom_indices.append(atom_idx)
                angulars.append(angular)
                exponents.append(shell_exponents)
                coefficients.append(coeff)

    return CartesianBasisTemplate(
        atom_charges=jnp.asarray(mol_cart.atom_charges()),
        nelectron=int(mol_cart.nelectron),
        ao_atom_indices=tuple(ao_atom_indices),
        angulars=tuple(angulars),
        exponents=tuple(exponents),
        coefficients=tuple(coefficients),
    )


def coordinates_from_pyscf_mol(
    mol: Any,
    *,
    unit: Literal["angstrom", "bohr"] = "angstrom",
) -> Array:
    """Return PySCF nuclear coordinates in the requested unit."""

    coords_bohr = jnp.asarray(mol.atom_coords())
    if unit == "bohr":
        return coords_bohr
    if unit == "angstrom":
        return coords_bohr * BOHR_TO_ANGSTROM
    raise ValueError(f"Unsupported coordinate unit {unit!r}.")


def _coordinates_to_bohr(coordinates: Array, unit: str) -> Array:
    coords = jnp.asarray(coordinates)
    if unit == "bohr":
        return coords
    if unit == "angstrom":
        return coords * ANGSTROM_TO_BOHR
    raise ValueError(f"Unsupported coordinate unit {unit!r}.")


def _basis_from_template(
    template: CartesianBasisTemplate,
    coordinates_bohr: Array,
) -> CartesianBasis:
    coords = jnp.asarray(coordinates_bohr)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coordinates must have shape (natom, 3).")
    if coords.shape[0] != template.natom:
        raise ValueError(
            f"coordinates natom={coords.shape[0]} does not match template natom={template.natom}."
        )

    aos: list[CartesianAO] = []
    for atom_idx, angular, exponents, coefficients in zip(
        template.ao_atom_indices,
        template.angulars,
        template.exponents,
        template.coefficients,
        strict=True,
    ):
        aos.append(
            CartesianAO(
                center=coords[atom_idx],
                angular=angular,
                exponents=exponents,
                coefficients=coefficients,
            )
        )
    return CartesianBasis(
        aos=tuple(aos),
        atom_coords=coords,
        atom_charges=jnp.asarray(template.atom_charges),
    )


def _run_traceable_rhf(
    template: CartesianBasisTemplate,
    coordinates_bohr: Array,
    *,
    config: RHFConfig,
) -> _TraceableRHFResult:
    basis = _basis_from_template(template, coordinates_bohr)
    overlap = overlap_matrix(basis)
    hcore = build_hcore(basis)
    rep_tensor = eri_tensor(basis)
    enuc = nuclear_repulsion_energy(basis.atom_coords, basis.atom_charges)

    nocc = template.nelectron // 2
    x = _orthogonalizer(overlap, config.orthogonalization_eps)
    mo_energy, mo_coeff = _diagonalize_fock(hcore, x)
    density = _build_density(mo_coeff, nocc)
    fock_hist: list[Array] = []
    err_hist: list[Array] = []
    fock = hcore

    for cycle in range(1, config.max_cycle + 1):
        fock = _build_fock(hcore, rep_tensor, density)
        if config.level_shift != 0.0:
            fock = fock + config.level_shift * overlap

        error = fock @ density @ overlap - overlap @ density @ fock
        if cycle >= config.diis_start_cycle and config.diis_space > 1:
            fock_eff = _diis_extrapolate(
                fock,
                error,
                fock_hist,
                err_hist,
                config.diis_space,
            )
        else:
            fock_eff = fock

        mo_energy, mo_coeff = _diagonalize_fock(fock_eff, x)
        density_new = _build_density(mo_coeff, nocc)
        if config.damping != 0.0:
            density_new = (1.0 - config.damping) * density_new + config.damping * density
        density = density_new

    final_fock = _build_fock(hcore, rep_tensor, density)
    total_energy = _electronic_energy(density, hcore, final_fock) + enuc
    mo_occ = jnp.zeros((overlap.shape[0],), dtype=overlap.dtype).at[:nocc].set(2.0)
    return _TraceableRHFResult(
        total_energy=total_energy,
        mo_energy=mo_energy,
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        density_matrix=density,
        rep_tensor=rep_tensor,
    )


def _build_response_reference(result: _TraceableRHFResult) -> _RestrictedResponseReference:
    nao = int(result.mo_coeff.shape[0])
    half_dm = result.density_matrix / 2.0
    half_occ = result.mo_occ / 2.0
    dtype = result.mo_coeff.dtype
    return _RestrictedResponseReference(
        ao=jnp.zeros((1, nao), dtype=dtype),
        grid=_GridReference(weights=jnp.ones((1,), dtype=dtype)),
        rep_tensor=result.rep_tensor,
        mo_coeff=jnp.stack([result.mo_coeff, result.mo_coeff], axis=0),
        mo_occ=jnp.stack([half_occ, half_occ], axis=0),
        mo_energy=jnp.stack([result.mo_energy, result.mo_energy], axis=0),
        rdm1=jnp.stack([half_dm, half_dm], axis=0),
        exact_exchange_fraction=1.0,
    )


def _ground_state_energy_from_coordinates(
    template: CartesianBasisTemplate,
    coordinates: Array,
    config: RHFExcitedStateSurfaceConfig,
) -> Array:
    coords_bohr = _coordinates_to_bohr(coordinates, config.coordinate_unit)
    result = _run_traceable_rhf(template, coords_bohr, config=config.scf)
    return result.total_energy


def _excited_state_energy_from_coordinates(
    template: CartesianBasisTemplate,
    coordinates: Array,
    config: RHFExcitedStateSurfaceConfig,
) -> Array:
    coords_bohr = _coordinates_to_bohr(coordinates, config.coordinate_unit)
    rhf_result = _run_traceable_rhf(template, coords_bohr, config=config.scf)
    reference = _build_response_reference(rhf_result)
    solver = RestrictedCasidaTDDFT(
        molecule=reference,
        xc_functional=_HF_ONLY_RESPONSE_FUNCTIONAL,
        eigensolver=config.eigensolver,
        excitation_threshold=config.excitation_threshold,
        matrix_eps=config.matrix_eps,
        davidson_tol=config.davidson_tol,
        davidson_max_iter=config.davidson_max_iter,
        davidson_max_subspace=config.davidson_max_subspace,
    )
    nstates = config.state_index + 1
    if config.response_method == "tda":
        omega = solver.tda(nstates=nstates).excitation_energies[config.state_index]
    elif config.response_method == "casida":
        omega = solver.kernel(nstates=nstates).excitation_energies[config.state_index]
    else:
        raise ValueError(
            f"Unsupported response_method={config.response_method!r}. "
            "Choose 'tda' or 'casida'."
        )
    return rhf_result.total_energy + omega


def make_rhf_ground_state_surface_from_template(
    template: CartesianBasisTemplate,
    *,
    config: RHFExcitedStateSurfaceConfig | None = None,
    label: str = "rhf_ground_state",
) -> EnergySurface:
    """Build a differentiable RHF ground-state energy surface."""

    cfg = RHFExcitedStateSurfaceConfig() if config is None else config
    return EnergySurface(
        label=label,
        state_kind="ground",
        energy_fn=lambda coordinates: _ground_state_energy_from_coordinates(
            template, coordinates, cfg
        ),
    )


def make_rhf_excited_state_surface_from_template(
    template: CartesianBasisTemplate,
    *,
    config: RHFExcitedStateSurfaceConfig | None = None,
    label: str | None = None,
) -> EnergySurface:
    """Build a differentiable RHF excited-state surface for geometry optimization."""

    cfg = RHFExcitedStateSurfaceConfig() if config is None else config
    tag = (
        f"rhf_{cfg.response_method}_excited_state_{cfg.state_index + 1}"
        if label is None
        else label
    )
    return EnergySurface(
        label=tag,
        state_kind="excited",
        energy_fn=lambda coordinates: _excited_state_energy_from_coordinates(
            template, coordinates, cfg
        ),
    )


def make_rhf_ground_state_surface_from_pyscf_mol(
    mol: Any,
    *,
    max_l: int = 3,
    config: RHFExcitedStateSurfaceConfig | None = None,
    label: str = "rhf_ground_state",
) -> EnergySurface:
    """Convenience wrapper building an RHF ground-state surface from PySCF Mole."""

    template = build_cartesian_basis_template_from_pyscf_mol(mol, max_l=max_l)
    return make_rhf_ground_state_surface_from_template(
        template,
        config=config,
        label=label,
    )


def make_rhf_excited_state_surface_from_pyscf_mol(
    mol: Any,
    *,
    max_l: int = 3,
    config: RHFExcitedStateSurfaceConfig | None = None,
    label: str | None = None,
) -> EnergySurface:
    """Convenience wrapper building an RHF excited-state surface from PySCF Mole."""

    template = build_cartesian_basis_template_from_pyscf_mol(mol, max_l=max_l)
    return make_rhf_excited_state_surface_from_template(
        template,
        config=config,
        label=label,
    )


def run_rhf_excited_state_geometry_optimization(
    mol: Any,
    *,
    initial_coordinates: Array | None = None,
    max_l: int = 3,
    surface_config: RHFExcitedStateSurfaceConfig | None = None,
    optimization_config: GeometryOptimizationConfig | None = None,
    label: str | None = None,
) -> GeometryOptimizationResult:
    """High-level excited-state geometry-optimization API.

    This is the packaged API for geometry optimization on top of the
    JAX-integrals + RHF + local TDA/Casida response chain. Coordinates are
    optimized in the unit specified by `surface_config.coordinate_unit`
    (`"angstrom"` by default).
    """

    cfg = RHFExcitedStateSurfaceConfig() if surface_config is None else surface_config
    coords0 = (
        coordinates_from_pyscf_mol(mol, unit=cfg.coordinate_unit)
        if initial_coordinates is None
        else jnp.asarray(initial_coordinates)
    )
    surface = make_rhf_excited_state_surface_from_pyscf_mol(
        mol,
        max_l=max_l,
        config=cfg,
        label=label,
    )
    return run_geometry_optimization(
        surface,
        coords0,
        config=optimization_config,
    )
