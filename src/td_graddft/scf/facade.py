from __future__ import annotations

from dataclasses import dataclass, field, replace
import os
from types import SimpleNamespace
from typing import Any, Literal

import jax
import jax.numpy as jnp

from ..data.molecule import MoleculeSpec
from ..jax_runtime import (
    DEFAULT_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS,
    DEFAULT_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES,
    configure_jax_persistent_cache,
)
from .rks import RKSConfig
from .uks import UKSConfig
from .cuda_direct_jk import cuda_ffi_available


def restricted_reference_from_spec_with_jax_rks(**kwargs: Any) -> Any:
    from ..reference import restricted_reference_from_spec_with_jax_rks as builder

    return builder(**kwargs)


def precompile_restricted_cuda_direct_rks_reference(**kwargs: Any) -> Any:
    from ..reference import precompile_restricted_cuda_direct_rks_reference as precompiler

    return precompiler(**kwargs)


def unrestricted_reference_from_spec_with_jax_uks(**kwargs: Any) -> Any:
    from ..reference import unrestricted_reference_from_spec_with_jax_uks as builder

    return builder(**kwargs)


def _restricted_reference_from_pyscf_cpu_rks(mf: "RKS") -> Any:
    try:
        from pyscf import dft, gto
    except ModuleNotFoundError as exc:
        raise ImportError("PySCF is required for execution_device='cpu'.") from exc

    mol = gto.M(
        atom=mf.mol.atom,
        basis=mf.mol.basis,
        unit=mf.mol.unit,
        charge=int(mf.mol.charge),
        spin=int(mf.mol.spin),
        cart=bool(mf.mol.cart),
        verbose=int(mf.mol.verbose),
    )
    pyscf_mf = dft.RKS(mol)
    pyscf_mf.xc = mf.xc
    pyscf_mf.grids.level = int(mf.grids_level)
    pyscf_mf.conv_tol = float(mf.conv_tol)
    pyscf_mf.max_cycle = int(mf.max_cycle)
    pyscf_mf.damp = float(mf.damp)
    pyscf_mf.level_shift = float(mf.level_shift)
    pyscf_mf.kernel()
    return SimpleNamespace(
        mf_energy=float(pyscf_mf.e_tot),
        mo_energy=jnp.asarray(pyscf_mf.mo_energy),
        mo_coeff=jnp.asarray(pyscf_mf.mo_coeff),
        mo_occ=jnp.asarray(pyscf_mf.mo_occ),
        converged=bool(pyscf_mf.converged),
        pyscf_mf=pyscf_mf,
    )


def _hashable_static_value(value: Any) -> Any:
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


@dataclass(frozen=True)
class _CudaDirectReferenceSolver:
    """HQC-style fixed-structure CUDA-direct RKS solver."""

    static_kwargs: dict[str, Any]

    def __call__(self, spec: MoleculeSpec) -> Any:
        return restricted_reference_from_spec_with_jax_rks(atom=spec, **self.static_kwargs)

    def precompile(self, spec: MoleculeSpec) -> Any:
        return precompile_restricted_cuda_direct_rks_reference(
            atom=spec,
            **self.static_kwargs,
        )


def _make_cuda_direct_reference_solver(**static_kwargs: Any) -> _CudaDirectReferenceSolver:
    """Build an HQC-style fixed-structure RKS solver callable."""

    return _CudaDirectReferenceSolver(static_kwargs=dict(static_kwargs))


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


def _configure_cuda_jax_cache() -> None:
    if os.environ.get("TD_GRADDFT_DISABLE_JAX_CACHE", "").lower() in {"1", "true", "yes", "on"}:
        return
    cache_dir = os.environ.get("TD_GRADDFT_JAX_CACHE_DIR")
    if cache_dir is None or not cache_dir.strip():
        return
    min_compile_time = float(
        os.environ.get(
            "TD_GRADDFT_JAX_CACHE_MIN_COMPILE_SECS",
            DEFAULT_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS,
        )
    )
    min_entry_size = int(
        os.environ.get(
            "TD_GRADDFT_JAX_CACHE_MIN_ENTRY_SIZE_BYTES",
            DEFAULT_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES,
        )
    )
    configure_jax_persistent_cache(
        cache_dir=cache_dir,
        min_compile_time_secs=min_compile_time,
        min_entry_size_bytes=min_entry_size,
    )


def _differentiable_rks_config(config: RKSConfig) -> RKSConfig:
    return replace(config, jk_backend="full", direct_jk_engine="jax")


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
    grid_ao_backend: Literal["pyscf", "jax"] = "jax"
    execution_device: Literal["auto", "cpu", "gpu"] = "auto"

    e_tot: Any | None = None
    converged: bool | None = None
    mo_energy: Any | None = None
    mo_coeff: Any | None = None
    mo_occ: Any | None = None
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
        self.converged = bool(getattr(reference, "converged", True))

    def _put_on_requested_device(self, reference: Any) -> Any:
        if self.execution_device == "auto":
            return reference
        from ..device import put_reference_on_device, resolve_execution_device

        device = resolve_execution_device(self.execution_device)
        return put_reference_on_device(reference, device=device)


@dataclass
class RKS(_BaseKS):
    jk_backend: Literal["full", "df", "direct"] = "full"
    direct_jk_engine: Literal["jax", "cuda"] = "jax"
    df_tol: float = 1e-10
    df_max_rank: int | None = None
    direct_scf_tol: float = 0.0
    direct_scf_incremental: bool = True
    iteration_backend: Literal["python", "lax"] = "python"
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
            iteration_backend=self.iteration_backend,
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
        if self.execution_device == "cpu" and not geometry_is_traced:
            return _restricted_reference_from_pyscf_cpu_rks(self)
        rks_config = self._config()
        integral_backend = self.integral_backend
        include_dipole_integrals = not (
            self.jk_backend == "direct" and self.direct_jk_engine == "cuda"
        )
        if geometry_is_traced:
            rks_config = _differentiable_rks_config(rks_config)
            integral_backend = "libcint"
            include_dipole_integrals = True
        if (
            not geometry_is_traced
            and self.jk_backend == "direct"
            and self.direct_jk_engine == "cuda"
            and cuda_ffi_available()
        ):
            solver = self._get_cuda_direct_reference_solver(
                spec,
                rks_config=rks_config,
                integral_backend=integral_backend,
                include_dipole_integrals=include_dipole_integrals,
            )
            return solver(spec)
        reference = restricted_reference_from_spec_with_jax_rks(
            atom=spec,
            basis=self.mol.basis,
            xc_spec=self.xc,
            unit=self.mol.unit,
            charge=self.mol.charge,
            spin=self.mol.spin,
            cart=self.mol.cart,
            grids_level=self.grids_level,
            max_l=self.max_l,
            rks_config=rks_config,
            grid_ao_backend="jax" if geometry_is_traced else self.grid_ao_backend,
            integral_backend=integral_backend,
            libcint_geometry_grad_policy=self.geometry_grad_policy,
            compute_local_hfx_features=self.compute_local_hfx_features,
            compute_local_hfx_aux=self.compute_local_hfx_aux,
            hfx_omega_values=self.hfx_omega_values,
            hfx_chunk_size=self.hfx_chunk_size,
            include_dipole_integrals=include_dipole_integrals,
            verbose=self.mol.verbose,
        )
        return reference

    def kernel(self) -> Any:
        reference = self._build_reference(self._spec())
        reference = self._put_on_requested_device(reference)
        self._sync_from_reference(reference)
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
            self.iteration_backend = "python"
            return self
        if cuda_ffi_available():
            _configure_cuda_jax_cache()
            self.jk_backend = "direct"
            self.direct_jk_engine = "cuda"
            self.integral_backend = "libcint"
            self.grid_ao_backend = "pyscf"
            self.iteration_backend = "lax"
            if precompile:
                spec = self._spec()
                if not _contains_jax_tracer(spec):
                    precompile_config = replace(self._config(), iteration_backend="lax")
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
        self.iteration_backend = "python"
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
        reference = unrestricted_reference_from_spec_with_jax_uks(
            atom=spec,
            basis=self.mol.basis,
            xc_spec=self.xc,
            unit=self.mol.unit,
            charge=self.mol.charge,
            spin=self.mol.spin,
            cart=self.mol.cart,
            grids_level=self.grids_level,
            max_l=self.max_l,
            uks_config=self._config(),
            grid_ao_backend=self.grid_ao_backend,
            integral_backend=self.integral_backend,
            libcint_geometry_grad_policy=self.geometry_grad_policy,
            verbose=self.mol.verbose,
        )
        return reference

    def kernel(self) -> Any:
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
        spec = self.mf._spec()
        return jax.grad(lambda coords: _energy_for_coords(self.mf, coords))(spec.coords_bohr)
