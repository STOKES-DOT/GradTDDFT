from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import jax

from ..data.molecule import MoleculeSpec
from ..jax_runtime import configure_jax_persistent_cache
from .core import _contains_jax_tracer, _hashable_static_value
from ..data.integrals import cuda_ffi_available
from .builders import (
    _configure_cuda_jax_cache as _configure_cuda_jax_cache_impl,
    _make_cuda_direct_reference_solver,
    build_restricted_reference_from_facade,
    build_restricted_scf_result_from_facade,
    build_unrestricted_reference_from_facade,
    precompile_restricted_cuda_direct_rks_solver,
    restricted_molecule_from_spec_with_jax_rks,
    unrestricted_molecule_from_spec_with_jax_uks,
)
from .inputs import build_rks_integral_inputs
from .rks import RKSConfig, run_rks_from_integrals
from .uks import UKSConfig

precompile_restricted_cuda_direct_rks_reference = precompile_restricted_cuda_direct_rks_solver
restricted_reference_from_spec_with_jax_rks = restricted_molecule_from_spec_with_jax_rks
unrestricted_reference_from_spec_with_jax_uks = unrestricted_molecule_from_spec_with_jax_uks


def _configure_cuda_jax_cache() -> None:
    return _configure_cuda_jax_cache_impl(configure_fn=configure_jax_persistent_cache)


def _configure_jax_cache() -> None:
    return _configure_cuda_jax_cache_impl(configure_fn=configure_jax_persistent_cache)


@dataclass
class _BaseKS:
    mol: Any
    xc: str = "pbe"
    conv_tol: float = 1e-10
    conv_tol_density: float = 1e-8
    max_cycle: int = 80
    damp: float = 0.0
    level_shift: float = 0.0
    grids_level: int = 0
    max_l: int = 3
    integral_backend: Literal["jax", "libcint"] = "libcint"
    geometry_grad_policy: Literal["analytic", "error", "zero"] = "analytic"
    grid_ao_backend: Literal["jax"] = "jax"
    execution_device: Literal["auto", "cpu", "gpu"] = "auto"
    init_guess: Any = "minao"
    chkfile: str | None = None
    sap_basis: Any | None = None
    init_guess_chkfile_project: bool | None = None

    e_tot: Any | None = None
    converged: bool | None = None
    mo_energy: Any | None = None
    mo_coeff: Any | None = None
    mo_occ: Any | None = None
    cycles: int | None = None
    reference: Any | None = None

    def run(self) -> "_BaseKS":
        self.kernel()
        return self

    def TDA(self, **kwargs: Any) -> Any:
        from .. import tdscf

        return tdscf.TDA(self, **kwargs)

    def TDDFT(self, **kwargs: Any) -> Any:
        from .. import tdscf

        return tdscf.TDDFT(self, **kwargs)

    def nuc_grad_method(self) -> "_NuclearGradient":
        return _NuclearGradient(self)

    def _spec(self) -> MoleculeSpec:
        if not hasattr(self.mol, "to_spec"):
            raise TypeError("SCF facade expects a td_graddft.gto.M molecule.")
        return self.mol.to_spec()

    def _sync_from_reference(self, reference: Any) -> None:
        self.reference = reference
        self.e_tot = getattr(reference, "mf_energy", None)
        self.mo_energy = getattr(reference, "mo_energy", None)
        self.mo_coeff = getattr(reference, "mo_coeff", None)
        self.mo_occ = getattr(reference, "mo_occ", None)
        self.cycles = getattr(reference, "cycles", None)
        self.converged = bool(getattr(reference, "converged", True))

    def _sync_from_scf_result(self, result: Any) -> None:
        self.reference = None
        self.e_tot = getattr(result, "total_energy", None)
        self.mo_energy = getattr(result, "mo_energy", None)
        self.mo_coeff = getattr(result, "mo_coeff", None)
        self.mo_occ = getattr(result, "mo_occ", None)
        self.cycles = getattr(result, "cycles", None)
        self.converged = bool(getattr(result, "converged", True))

    def _put_on_requested_device(self, reference: Any) -> Any:
        if self.execution_device == "auto":
            return reference
        from ..device import put_reference_on_device, resolve_execution_device

        device = resolve_execution_device(self.execution_device)
        return put_reference_on_device(reference, device=device)

    def _ensure_reference(self) -> Any:
        if self.reference is not None:
            return self.reference
        if self.e_tot is None:
            raise RuntimeError(
                "Run ground-state mf.kernel() or mf.run() before launching TD-SCF."
            )
        reference = self._build_reference(self._spec())
        reference = self._put_on_requested_device(reference)
        self._sync_from_reference(reference)
        return reference


@dataclass
class RKS(_BaseKS):
    jk_backend: Literal["full", "df", "direct"] = "full"
    direct_jk_engine: Literal["jax", "cuda"] = "jax"
    df_tol: float = 1e-10
    df_max_rank: int | None = None
    direct_scf_tol: float = 0.0
    direct_scf_incremental: bool = True
    compute_local_hfx_features: bool = False
    compute_local_hfx_aux: bool = False
    hfx_omega_values: tuple[float, ...] = (0.0, 0.4)
    hfx_chunk_size: int = 512
    _cuda_direct_reference_solver: Any | None = field(default=None, init=False, repr=False)
    _cuda_direct_reference_solver_key: tuple[Any, ...] | None = field(default=None, init=False, repr=False)

    def _config(self) -> RKSConfig:
        return RKSConfig(
            xc_spec=self.xc,
            max_cycle=self.max_cycle,
            conv_tol=self.conv_tol,
            conv_tol_density=self.conv_tol_density,
            damping=self.damp,
            level_shift=self.level_shift,
            jk_backend=self.jk_backend,
            direct_jk_engine=self.direct_jk_engine,
            df_tol=self.df_tol,
            df_max_rank=self.df_max_rank,
            direct_scf_tol=self.direct_scf_tol,
            direct_scf_incremental=self.direct_scf_incremental,
            iteration_backend="runtime",
        )

    def _cuda_direct_solver_key(
        self,
        spec: MoleculeSpec,
        *,
        rks_config: RKSConfig,
        integral_backend: str,
        include_dipole_integrals: bool,
    ) -> tuple[Any, ...]:
        return (
            tuple(spec.symbols),
            int(spec.charge),
            int(spec.spin),
            _hashable_static_value(self.mol.basis),
            str(self.mol.unit),
            bool(self.mol.cart),
            int(self.grids_level),
            int(self.max_l),
            rks_config,
            str(self.grid_ao_backend),
            str(integral_backend),
            str(self.geometry_grad_policy),
            bool(self.compute_local_hfx_features),
            bool(self.compute_local_hfx_aux),
            tuple(float(value) for value in self.hfx_omega_values),
            int(self.hfx_chunk_size),
            bool(include_dipole_integrals),
            int(self.mol.verbose),
        )

    def _get_cuda_direct_reference_solver(
        self,
        spec: MoleculeSpec,
        *,
        rks_config: RKSConfig,
        integral_backend: str,
        include_dipole_integrals: bool,
    ) -> Any:
        key = self._cuda_direct_solver_key(
            spec,
            rks_config=rks_config,
            integral_backend=integral_backend,
            include_dipole_integrals=include_dipole_integrals,
        )
        if self._cuda_direct_reference_solver is not None and key == self._cuda_direct_reference_solver_key:
            return self._cuda_direct_reference_solver
        solver = _make_cuda_direct_reference_solver(
            reference_builder=restricted_reference_from_spec_with_jax_rks,
            precompile_builder=precompile_restricted_cuda_direct_rks_reference,
            basis=self.mol.basis,
            xc_spec=self.xc,
            unit=self.mol.unit,
            charge=self.mol.charge,
            spin=self.mol.spin,
            cart=self.mol.cart,
            grids_level=self.grids_level,
            max_l=self.max_l,
            rks_config=rks_config,
            grid_ao_backend=self.grid_ao_backend,
            integral_backend=integral_backend,
            libcint_geometry_grad_policy=self.geometry_grad_policy,
            compute_local_hfx_features=self.compute_local_hfx_features,
            compute_local_hfx_aux=self.compute_local_hfx_aux,
            hfx_omega_values=self.hfx_omega_values,
            hfx_chunk_size=self.hfx_chunk_size,
            include_dipole_integrals=include_dipole_integrals,
            verbose=self.mol.verbose,
        )
        self._cuda_direct_reference_solver = solver
        self._cuda_direct_reference_solver_key = key
        return solver

    def _build_reference(self, spec: MoleculeSpec) -> Any:
        geometry_is_traced = _contains_jax_tracer(spec)
        return build_restricted_reference_from_facade(
            spec,
            mol=self.mol,
            xc=self.xc,
            grids_level=self.grids_level,
            max_l=self.max_l,
            integral_backend=self.integral_backend,
            geometry_grad_policy=self.geometry_grad_policy,
            grid_ao_backend=self.grid_ao_backend,
            rks_config=self._config(),
            compute_local_hfx_features=self.compute_local_hfx_features,
            compute_local_hfx_aux=self.compute_local_hfx_aux,
            hfx_omega_values=self.hfx_omega_values,
            hfx_chunk_size=self.hfx_chunk_size,
            include_dipole_integrals=not (
                self.jk_backend == "direct" and self.direct_jk_engine == "cuda"
            ),
            init_guess=self.init_guess,
            chkfile=self.chkfile,
            sap_basis=self.sap_basis,
            init_guess_chkfile_project=self.init_guess_chkfile_project,
            geometry_is_traced=geometry_is_traced,
            reference_builder=restricted_reference_from_spec_with_jax_rks,
            cuda_direct_solver_getter=self._get_cuda_direct_reference_solver,
            cuda_available=cuda_ffi_available(),
        )

    def _build_scf_result(self, spec: MoleculeSpec) -> Any:
        return build_restricted_scf_result_from_facade(
            spec,
            mol=self.mol,
            xc=self.xc,
            grids_level=self.grids_level,
            max_l=self.max_l,
            integral_backend=self.integral_backend,
            geometry_grad_policy=self.geometry_grad_policy,
            grid_ao_backend=self.grid_ao_backend,
            rks_config=self._config(),
            init_guess=self.init_guess,
            chkfile=self.chkfile,
            sap_basis=self.sap_basis,
            init_guess_chkfile_project=self.init_guess_chkfile_project,
            geometry_is_traced=_contains_jax_tracer(spec),
            build_inputs_fn=build_rks_integral_inputs,
            run_rks_fn=run_rks_from_integrals,
        )

    def kernel(self) -> Any:
        _configure_jax_cache()
        result = self._build_scf_result(self._spec())
        self._sync_from_scf_result(result)
        return self.e_tot

    def density_fit(self) -> "RKS":
        self.jk_backend = "df"
        return self

    def direct_scf(self) -> "RKS":
        self.jk_backend = "direct"
        return self

    def cuda_direct_scf(
        self,
        execution_device: Literal["auto", "cpu", "gpu"] | None = None,
        *,
        precompile: bool = False,
    ) -> "RKS":
        if execution_device is not None:
            self.execution_device = execution_device
        preference = str(self.execution_device).lower()
        if preference not in {"auto", "cpu", "gpu"}:
            raise ValueError("execution_device must be 'auto', 'cpu', or 'gpu'.")
        if preference == "cpu":
            self.jk_backend = "full"
            self.direct_jk_engine = "jax"
            self.integral_backend = "libcint"
            return self
        if cuda_ffi_available():
            _configure_cuda_jax_cache()
            self.jk_backend = "direct"
            self.direct_jk_engine = "cuda"
            self.integral_backend = "libcint"
            self.grid_ao_backend = "jax"
            if precompile:
                spec = self._spec()
                if not _contains_jax_tracer(spec):
                    precompile_config = self._config()
                    solver = self._get_cuda_direct_reference_solver(
                        spec,
                        rks_config=precompile_config,
                        integral_backend=self.integral_backend,
                        include_dipole_integrals=False,
                    )
                    solver.precompile(spec)
            return self
        if preference == "gpu":
            raise RuntimeError(
                "execution_device='gpu' requested but CUDA FFI is unavailable. "
                "Check JAX GPU visibility and TD_GRADDFT_NVCC/nvcc."
            )
        self.jk_backend = "full"
        self.direct_jk_engine = "jax"
        self.integral_backend = "libcint"
        return self


@dataclass
class UKS(_BaseKS):
    def _config(self) -> UKSConfig:
        return UKSConfig(
            xc_spec=self.xc,
            max_cycle=self.max_cycle,
            conv_tol=self.conv_tol,
            conv_tol_density=self.conv_tol_density,
            damping=self.damp,
            level_shift=self.level_shift,
        )

    def _build_reference(self, spec: MoleculeSpec) -> Any:
        return build_unrestricted_reference_from_facade(
            spec,
            mol=self.mol,
            xc=self.xc,
            grids_level=self.grids_level,
            max_l=self.max_l,
            integral_backend=self.integral_backend,
            geometry_grad_policy=self.geometry_grad_policy,
            grid_ao_backend=self.grid_ao_backend,
            uks_config=self._config(),
            init_guess=self.init_guess,
            chkfile=self.chkfile,
            sap_basis=self.sap_basis,
            init_guess_chkfile_project=self.init_guess_chkfile_project,
            reference_builder=unrestricted_reference_from_spec_with_jax_uks,
        )

    def kernel(self) -> Any:
        _configure_jax_cache()
        reference = self._build_reference(self._spec())
        reference = self._put_on_requested_device(reference)
        self._sync_from_reference(reference)
        return self.e_tot

    def density_fit(self) -> "UKS":
        raise NotImplementedError("UKS density fitting is not exposed by the current core solver.")

    def direct_scf(self) -> "UKS":
        raise NotImplementedError("UKS direct SCF is not exposed by the current core solver.")


def _energy_for_coords(mf: _BaseKS, coords_bohr: Any) -> Any:
    spec = mf._spec()
    traced_spec = MoleculeSpec(
        symbols=spec.symbols,
        coords_bohr=coords_bohr,
        charges=spec.charges,
        charge=spec.charge,
        spin=spec.spin,
        unit=spec.unit,
    )
    reference = mf._build_reference(traced_spec)
    return reference.mf_energy


@dataclass(frozen=True)
class _NuclearGradient:
    mf: _BaseKS

    def kernel(self) -> Any:
        raise NotImplementedError(
            "Explicit SCF coordinate differentiation is disabled. "
            "Use implicit-differential training workflows instead."
        )
