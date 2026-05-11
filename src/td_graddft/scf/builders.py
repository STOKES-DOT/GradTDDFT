from __future__ import annotations

from dataclasses import is_dataclass, replace
import hashlib
import os
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from ..data.integrals import precompile_eri_kernels
from ..data.integrals.libcint_autodiff import LibcintGeometryGradPolicy
from ..data.molecule import MoleculeSpec
from ..jax_libxc import hybrid_coeff, parse_xc
from ..neural_xc.inputs import (
    _local_hfx_features_from_basis_dm,
    _local_pt2_feature_from_restricted_orbitals,
)
from .cuda_direct_jk import CudaDirectJKBuilder, cuda_ffi_available
from .features import _restricted_response_eri_slices_from_mo_tensor
from .inputs import build_rks_integral_inputs, build_uks_integral_inputs
from .molecules import QuadratureGrid, RestrictedMolecule, UnrestrictedMolecule
from .packed_eri import eri_pair_matrix_to_mo_eri_slices
from .rks import RKSConfig, run_rks_from_integrals, run_rks_from_integrals_traceable
from .uks import UKSConfig, run_uks_from_integrals


_CUDA_DIRECT_RKS_JIT_CACHE_MAXSIZE = 16
_CUDA_DIRECT_RKS_JIT_CACHE: dict[tuple[Any, ...], Any] = {}
_CUDA_DIRECT_RKS_COMPILED_CACHE_MAXSIZE = 16
_CUDA_DIRECT_RKS_COMPILED_CACHE: dict[tuple[Any, ...], Any] = {}
_CUDA_DIRECT_JK_BUILDER_CACHE_MAXSIZE = 16
_CUDA_DIRECT_JK_BUILDER_CACHE: dict[tuple[Any, ...], Any] = {}
_CUDA_DIRECT_RKS_INPUT_CACHE_MAXSIZE = 16
_CUDA_DIRECT_RKS_INPUT_CACHE: dict[tuple[Any, ...], Any] = {}
_CUDA_DIRECT_GRID_AO_BUCKET_SIZE_ENV = "TD_GRADDFT_CUDA_GRID_AO_BUCKET_SIZE"
_DEFAULT_CUDA_DIRECT_GRID_AO_BUCKET_SIZE = 1024


def _digest_array_for_cache(value: Any) -> tuple[tuple[int, ...], str, str]:
    if value is None:
        return (), "none", ""
    arr = np.ascontiguousarray(np.asarray(jax.device_get(value)))
    digest = hashlib.sha256(arr.view(np.uint8)).hexdigest()
    return tuple(int(item) for item in arr.shape), str(arr.dtype), digest


def _cuda_direct_basis_cache_key(basis: Any) -> tuple[Any, ...]:
    return (
        int(basis.nao),
        tuple(tuple(int(item) for item in angular) for angular in basis.ao_angulars),
        tuple(int(item) for item in basis.ao_nprims_tuple),
        tuple(int(item) for item in basis.shell_nprims_tuple),
        _digest_array_for_cache(basis.atom_charges),
        _digest_array_for_cache(basis.ao_exponents_padded),
        _digest_array_for_cache(basis.ao_coefficients_padded),
        _digest_array_for_cache(basis.shell_exponents_padded),
        _digest_array_for_cache(basis.shell_coefficients_padded),
        _digest_array_for_cache(basis.shell_ao_indices_padded),
        _digest_array_for_cache(basis.shell_ao_sizes),
    )


def _cuda_direct_rks_config_cache_key(config: RKSConfig) -> tuple[Any, ...]:
    return (
        str(config.xc_spec),
        int(config.max_cycle),
        float(config.conv_tol),
        float(config.conv_tol_density),
        float(config.damping),
        float(config.level_shift),
        float(config.orthogonalization_eps),
        float(config.density_floor),
        None if config.potential_clip is None else float(config.potential_clip),
        str(config.iteration_backend),
        str(config.jk_backend),
        str(config.direct_jk_engine),
        float(config.df_tol),
        None if config.df_max_rank is None else int(config.df_max_rank),
        float(config.direct_scf_tol),
        bool(config.direct_scf_incremental),
    )


def _cuda_direct_rks_input_config_cache_key(config: RKSConfig) -> tuple[Any, ...]:
    """Config key for molecule input construction.

    The SCF iteration backend changes execution strategy, not the PySCF/libcint
    input tensors. Keeping it out of this key lets native Python CUDA runs reuse
    inputs warmed by an explicit lax/XLA precompile and vice versa.
    """

    return (
        str(config.xc_spec),
        int(config.max_cycle),
        float(config.conv_tol),
        float(config.conv_tol_density),
        float(config.damping),
        float(config.level_shift),
        float(config.orthogonalization_eps),
        float(config.density_floor),
        None if config.potential_clip is None else float(config.potential_clip),
        str(config.jk_backend),
        str(config.direct_jk_engine),
        float(config.df_tol),
        None if config.df_max_rank is None else int(config.df_max_rank),
        float(config.direct_scf_tol),
        bool(config.direct_scf_incremental),
    )


def _hashable_static_value(value: Any) -> Any:
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _cuda_direct_reference_inputs_cache_key(
    *,
    atom: Any,
    basis: Any,
    xc_spec: str,
    unit: str,
    charge: int,
    spin: int,
    cart: bool,
    grids_level: int,
    max_l: int,
    config: RKSConfig,
    grid_ao_backend: str,
    integral_backend: str,
    libcint_geometry_grad_policy: str,
    include_dipole_integrals: bool,
    precompile_eri: bool,
    precompile_eri_chunk_size: int,
    verbose: int,
    mol_kwargs: dict[str, Any],
) -> tuple[Any, ...] | None:
    if not isinstance(atom, MoleculeSpec) or _contains_jax_tracer(atom):
        return None
    return (
        tuple(atom.symbols),
        int(atom.charge),
        int(atom.spin),
        _digest_array_for_cache(atom.coords_bohr),
        _hashable_static_value(basis),
        str(xc_spec),
        str(unit),
        int(charge),
        int(spin),
        bool(cart),
        int(grids_level),
        int(max_l),
        _cuda_direct_rks_input_config_cache_key(config),
        str(grid_ao_backend),
        str(integral_backend),
        str(libcint_geometry_grad_policy),
        bool(include_dipole_integrals),
        bool(precompile_eri),
        int(precompile_eri_chunk_size),
        int(verbose),
        repr(sorted((str(key), repr(value)) for key, value in mol_kwargs.items())),
    )


def _cache_cuda_direct_rks_inputs(key: tuple[Any, ...] | None, scf_inputs: Any) -> None:
    if key is None:
        return
    if len(_CUDA_DIRECT_RKS_INPUT_CACHE) >= _CUDA_DIRECT_RKS_INPUT_CACHE_MAXSIZE:
        _CUDA_DIRECT_RKS_INPUT_CACHE.pop(next(iter(_CUDA_DIRECT_RKS_INPUT_CACHE)))
    _CUDA_DIRECT_RKS_INPUT_CACHE[key] = scf_inputs


def _cuda_direct_jk_builder_metadata_flags(config: RKSConfig) -> tuple[bool, bool]:
    rys_fast_path = (
        config.jk_backend == "direct"
        and config.direct_jk_engine == "cuda"
        and float(config.direct_scf_tol) <= 0.0
    )
    if rys_fast_path:
        return False, False
    return True, True


def _cuda_direct_jk_builder_cache_key(
    basis: Any,
    config: RKSConfig,
    *,
    include_pair_metadata: bool | None = None,
    include_joltqc_metadata: bool | None = None,
) -> tuple[Any, ...]:
    if include_pair_metadata is None or include_joltqc_metadata is None:
        default_pair, default_joltqc = _cuda_direct_jk_builder_metadata_flags(config)
        if include_pair_metadata is None:
            include_pair_metadata = default_pair
        if include_joltqc_metadata is None:
            include_joltqc_metadata = default_joltqc
    # The exact full-JoltQC path receives centers through dynamic basis_data, so the
    # builder only owns static shell/layout metadata. Screened direct-SCF builders
    # use geometry-dependent internal basis_data and must not be reused across
    # geometries.
    geometry_part: tuple[Any, ...] = ()
    if float(config.direct_scf_tol) > 0.0:
        geometry_part = (
            _digest_array_for_cache(getattr(basis, "atom_coords", None)),
            _digest_array_for_cache(getattr(basis, "ao_centers", None)),
            _digest_array_for_cache(getattr(basis, "shell_centers", None)),
        )
    return (
        _cuda_direct_basis_cache_key(basis),
        float(config.direct_scf_tol),
        bool(include_pair_metadata),
        bool(include_joltqc_metadata),
        geometry_part,
    )


def _cached_cuda_direct_jk_builder(basis: Any, config: RKSConfig) -> Any:
    include_pair_metadata, include_joltqc_metadata = _cuda_direct_jk_builder_metadata_flags(config)
    key = _cuda_direct_jk_builder_cache_key(
        basis,
        config,
        include_pair_metadata=include_pair_metadata,
        include_joltqc_metadata=include_joltqc_metadata,
    )
    builder = _CUDA_DIRECT_JK_BUILDER_CACHE.get(key)
    if builder is not None:
        return builder
    builder = CudaDirectJKBuilder(
        basis,
        include_pair_metadata=include_pair_metadata,
        include_joltqc_metadata=include_joltqc_metadata,
    )
    if not include_pair_metadata and not bool(getattr(builder, "has_rys_direct_jk", False)):
        include_pair_metadata = True
        include_joltqc_metadata = True
        key = _cuda_direct_jk_builder_cache_key(
            basis,
            config,
            include_pair_metadata=include_pair_metadata,
            include_joltqc_metadata=include_joltqc_metadata,
        )
        cached_builder = _CUDA_DIRECT_JK_BUILDER_CACHE.get(key)
        if cached_builder is not None:
            return cached_builder
        builder = CudaDirectJKBuilder(
            basis,
            include_pair_metadata=include_pair_metadata,
            include_joltqc_metadata=include_joltqc_metadata,
        )
    if len(_CUDA_DIRECT_JK_BUILDER_CACHE) >= _CUDA_DIRECT_JK_BUILDER_CACHE_MAXSIZE:
        _CUDA_DIRECT_JK_BUILDER_CACHE.pop(next(iter(_CUDA_DIRECT_JK_BUILDER_CACHE)))
    _CUDA_DIRECT_JK_BUILDER_CACHE[key] = builder
    return builder


def _cuda_direct_rks_runner_cache_key(scf_inputs: Any, config: RKSConfig) -> tuple[Any, ...]:
    direct_basis = scf_inputs.direct_basis
    if direct_basis is None:
        raise RuntimeError("CUDA direct RKS JIT runner requires direct_basis.")
    has_init_mo_coeff = scf_inputs.init_mo_coeff is not None
    has_init_mo_occ = scf_inputs.init_mo_occ is not None
    has_init_mo_energy = scf_inputs.init_mo_energy is not None
    return (
        _cuda_direct_jk_builder_cache_key(direct_basis, config),
        _cuda_direct_rks_config_cache_key(config),
        int(scf_inputs.nelectron),
        has_init_mo_coeff,
        has_init_mo_occ,
        has_init_mo_energy,
    )


def _cached_cuda_direct_rks_runner(scf_inputs: Any, config: RKSConfig) -> Any:
    key = _cuda_direct_rks_runner_cache_key(scf_inputs, config)
    runner = _CUDA_DIRECT_RKS_JIT_CACHE.get(key)
    if runner is not None:
        return runner

    direct_basis = scf_inputs.direct_basis
    if direct_basis is None:
        raise RuntimeError("CUDA direct RKS JIT runner requires direct_basis.")
    direct_cuda_jk_builder = _cached_cuda_direct_jk_builder(direct_basis, config)
    has_init_mo_coeff = scf_inputs.init_mo_coeff is not None
    has_init_mo_occ = scf_inputs.init_mo_occ is not None
    has_init_mo_energy = scf_inputs.init_mo_energy is not None
    nelectron = int(scf_inputs.nelectron)

    def _run(
        overlap,
        hcore,
        nuclear_repulsion,
        ao,
        ao_deriv1,
        grid_weights,
        *init_args,
    ):
        init_idx = 0
        init_mo_coeff = init_args[init_idx] if has_init_mo_coeff else None
        init_idx += int(has_init_mo_coeff)
        init_mo_occ = init_args[init_idx] if has_init_mo_occ else None
        init_idx += int(has_init_mo_occ)
        init_mo_energy = init_args[init_idx] if has_init_mo_energy else None
        return run_rks_from_integrals_traceable(
            overlap=overlap,
            hcore=hcore,
            eri=None,
            eri_pair_matrix=None,
            nelectron=nelectron,
            nuclear_repulsion=nuclear_repulsion,
            ao=ao,
            ao_deriv1=ao_deriv1,
            grid_weights=grid_weights,
            df_factors=None,
            direct_basis=direct_basis,
            direct_cuda_jk_builder=direct_cuda_jk_builder,
            init_mo_coeff=init_mo_coeff,
            init_mo_occ=init_mo_occ,
            init_mo_energy=init_mo_energy,
            config=config,
        )

    runner = jax.jit(_run)
    if len(_CUDA_DIRECT_RKS_JIT_CACHE) >= _CUDA_DIRECT_RKS_JIT_CACHE_MAXSIZE:
        _CUDA_DIRECT_RKS_JIT_CACHE.pop(next(iter(_CUDA_DIRECT_RKS_JIT_CACHE)))
    _CUDA_DIRECT_RKS_JIT_CACHE[key] = runner
    return runner


def _cuda_direct_rks_arg_signature(args: list[Any]) -> tuple[Any, ...]:
    signature: list[tuple[tuple[int, ...], str]] = []
    for arg in args:
        shape = tuple(int(dim) for dim in getattr(arg, "shape", np.shape(arg)))
        dtype = getattr(arg, "dtype", np.asarray(arg).dtype)
        signature.append((shape, str(dtype)))
    return tuple(signature)


def _cuda_direct_rks_compiled_cache_key(
    scf_inputs: Any,
    config: RKSConfig,
    args: list[Any],
) -> tuple[Any, ...]:
    return (
        _cuda_direct_rks_runner_cache_key(scf_inputs, config),
        _cuda_direct_rks_arg_signature(args),
    )


def _cache_cuda_direct_rks_compiled(key: tuple[Any, ...], compiled: Any) -> None:
    if len(_CUDA_DIRECT_RKS_COMPILED_CACHE) >= _CUDA_DIRECT_RKS_COMPILED_CACHE_MAXSIZE:
        _CUDA_DIRECT_RKS_COMPILED_CACHE.pop(next(iter(_CUDA_DIRECT_RKS_COMPILED_CACHE)))
    _CUDA_DIRECT_RKS_COMPILED_CACHE[key] = compiled


def _cuda_direct_rks_args(scf_inputs: Any) -> list[Any]:
    args: list[Any] = [
        scf_inputs.overlap,
        scf_inputs.hcore,
        scf_inputs.nuclear_repulsion,
        scf_inputs.ao,
        scf_inputs.ao_deriv1,
        scf_inputs.grid_weights,
    ]
    if scf_inputs.init_mo_coeff is not None:
        args.append(scf_inputs.init_mo_coeff)
    if scf_inputs.init_mo_occ is not None:
        args.append(scf_inputs.init_mo_occ)
    if scf_inputs.init_mo_energy is not None:
        args.append(scf_inputs.init_mo_energy)
    return args


def _cuda_direct_grid_ao_bucket_size(config: RKSConfig) -> int:
    if not (
        config.jk_backend == "direct"
        and config.direct_jk_engine == "cuda"
        and config.iteration_backend == "lax"
    ):
        return 0
    raw = os.environ.get(_CUDA_DIRECT_GRID_AO_BUCKET_SIZE_ENV)
    if raw is None or not str(raw).strip():
        return _DEFAULT_CUDA_DIRECT_GRID_AO_BUCKET_SIZE
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_CUDA_DIRECT_GRID_AO_BUCKET_SIZE


def _pad_array_axis_to(value: Any, *, axis: int, target: int, fill: str = "zero") -> Any:
    arr = jnp.asarray(value)
    current = int(arr.shape[axis])
    if target <= current:
        return arr
    if fill == "first":
        pad_shape = list(arr.shape)
        pad_shape[axis] = int(target) - current
        first = jnp.expand_dims(jnp.take(arr, 0, axis=axis), axis=axis)
        padding = jnp.broadcast_to(first, tuple(pad_shape))
        return jnp.concatenate([arr, padding], axis=axis)
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, int(target) - current)
    return jnp.pad(arr, tuple(pad_width), mode="constant")


def _bucket_cuda_direct_rks_grid_inputs(scf_inputs: Any, *, bucket_size: int) -> Any:
    if int(bucket_size) <= 0 or not is_dataclass(scf_inputs):
        return scf_inputs
    if not hasattr(scf_inputs, "grid_weights"):
        return scf_inputs
    grid_weights = jnp.asarray(scf_inputs.grid_weights)
    if grid_weights.ndim != 1:
        return scf_inputs
    current = int(grid_weights.shape[0])
    if current <= 0:
        return scf_inputs
    target = ((current + int(bucket_size) - 1) // int(bucket_size)) * int(bucket_size)
    if target == current:
        return scf_inputs
    ao_laplacian = scf_inputs.ao_laplacian
    if ao_laplacian is not None:
        ao_laplacian = _pad_array_axis_to(ao_laplacian, axis=0, target=target, fill="first")
    return replace(
        scf_inputs,
        coords=_pad_array_axis_to(scf_inputs.coords, axis=0, target=target, fill="first"),
        grid_weights=_pad_array_axis_to(grid_weights, axis=0, target=target),
        ao=_pad_array_axis_to(scf_inputs.ao, axis=0, target=target, fill="first"),
        ao_deriv1=_pad_array_axis_to(scf_inputs.ao_deriv1, axis=1, target=target, fill="first"),
        ao_laplacian=ao_laplacian,
    )


def _bucket_cuda_direct_rks_inputs_for_config(scf_inputs: Any, config: RKSConfig) -> Any:
    return _bucket_cuda_direct_rks_grid_inputs(
        scf_inputs,
        bucket_size=_cuda_direct_grid_ao_bucket_size(config),
    )


def _dummy_cuda_direct_rks_args(args: list[Any]) -> list[Any]:
    return [
        jnp.zeros(
            tuple(int(dim) for dim in np.shape(arg)),
            dtype=jnp.asarray(arg).dtype,
        )
        for arg in args
    ]


def precompile_cuda_direct_rks_inputs(scf_inputs: Any, config: RKSConfig) -> Any:
    args = _cuda_direct_rks_args(scf_inputs)
    key = _cuda_direct_rks_compiled_cache_key(scf_inputs, config, args)
    cached_compiled = _CUDA_DIRECT_RKS_COMPILED_CACHE.get(key)
    if cached_compiled is not None:
        return cached_compiled
    runner = _cached_cuda_direct_rks_runner(scf_inputs, config)
    compiled = runner.lower(*args).compile()
    _cache_cuda_direct_rks_compiled(key, compiled)
    return compiled


def precompile_cuda_direct_rks_signature(scf_inputs: Any, config: RKSConfig) -> Any:
    scf_inputs = _bucket_cuda_direct_rks_inputs_for_config(scf_inputs, config)
    args = _cuda_direct_rks_args(scf_inputs)
    key = _cuda_direct_rks_compiled_cache_key(scf_inputs, config, args)
    cached_compiled = _CUDA_DIRECT_RKS_COMPILED_CACHE.get(key)
    if cached_compiled is not None:
        return cached_compiled
    runner = _cached_cuda_direct_rks_runner(scf_inputs, config)
    compiled = runner.lower(*_dummy_cuda_direct_rks_args(args)).compile()
    _cache_cuda_direct_rks_compiled(key, compiled)
    return compiled


def precompile_restricted_cuda_direct_rks_solver(
    *,
    atom: Any,
    basis: Any,
    xc_spec: str = "pbe",
    unit: str = "Angstrom",
    charge: int = 0,
    spin: int = 0,
    cart: bool = True,
    grids_level: int = 0,
    max_l: int = 3,
    rks_config: RKSConfig | None = None,
    grid_ao_backend: Literal["jax"] = "jax",
    integral_backend: Literal["jax", "libcint"] = "libcint",
    libcint_geometry_grad_policy: LibcintGeometryGradPolicy = "analytic",
    include_dipole_integrals: bool = True,
    precompile_eri: bool = False,
    precompile_eri_chunk_size: int = 512,
    compute_local_hfx_features: bool = False,
    compute_local_hfx_aux: bool = False,
    compute_local_pt2_features: bool = False,
    hfx_omega_values: tuple[float, ...] = (0.0, 0.4),
    hfx_chunk_size: int = 512,
    verbose: int = 0,
    **mol_kwargs: Any,
) -> Any:
    """Compile the cached CUDA-direct RKS executable for the current structure."""

    if isinstance(atom, MoleculeSpec):
        charge = int(atom.charge)
        spin = int(atom.spin)
    if int(spin) != 0:
        raise NotImplementedError("CUDA direct RKS precompile only supports closed-shell systems.")
    if not bool(cart):
        raise NotImplementedError("CUDA direct RKS precompile currently supports cart=True only.")

    xc_spec_resolved = str(xc_spec)
    parse_xc(xc_spec_resolved)
    cfg = RKSConfig(xc_spec=xc_spec_resolved) if rks_config is None else rks_config
    if cfg.xc_spec != xc_spec_resolved:
        cfg = replace(cfg, xc_spec=xc_spec_resolved)
    if not (
        cfg.jk_backend == "direct"
        and cfg.direct_jk_engine == "cuda"
        and cfg.iteration_backend == "lax"
    ):
        raise ValueError("CUDA direct RKS precompile requires direct CUDA lax configuration.")
    input_cache_key = _cuda_direct_reference_inputs_cache_key(
        atom=atom,
        basis=basis,
        xc_spec=xc_spec_resolved,
        unit=unit,
        charge=charge,
        spin=spin,
        cart=cart,
        grids_level=grids_level,
        max_l=max_l,
        config=cfg,
        grid_ao_backend=grid_ao_backend,
        integral_backend=integral_backend,
        libcint_geometry_grad_policy=libcint_geometry_grad_policy,
        include_dipole_integrals=include_dipole_integrals,
        precompile_eri=precompile_eri,
        precompile_eri_chunk_size=precompile_eri_chunk_size,
        verbose=verbose,
        mol_kwargs=mol_kwargs,
    )
    scf_inputs = (
        _CUDA_DIRECT_RKS_INPUT_CACHE[input_cache_key]
        if input_cache_key in _CUDA_DIRECT_RKS_INPUT_CACHE
        else build_rks_integral_inputs(
            atom=atom,
            basis=basis,
            config=cfg,
            xc_spec=xc_spec_resolved,
            unit=unit,
            charge=charge,
            spin=spin,
            cart=cart,
            grids_level=grids_level,
            max_l=max_l,
            grid_ao_backend=grid_ao_backend,
            integral_backend=integral_backend,
            libcint_geometry_grad_policy=libcint_geometry_grad_policy,
            include_dipole_integrals=include_dipole_integrals,
            precompile_eri=precompile_eri,
            precompile_eri_chunk_size=precompile_eri_chunk_size,
            _precompile_eri_kernels=precompile_eri_kernels,
            verbose=verbose,
            **mol_kwargs,
        )
    )
    _cache_cuda_direct_rks_inputs(input_cache_key, scf_inputs)
    return precompile_cuda_direct_rks_signature(scf_inputs, cfg)


def _run_cached_cuda_direct_rks(scf_inputs: Any, config: RKSConfig) -> Any:
    scf_inputs = _bucket_cuda_direct_rks_inputs_for_config(scf_inputs, config)
    args = _cuda_direct_rks_args(scf_inputs)
    key = _cuda_direct_rks_compiled_cache_key(scf_inputs, config, args)
    compiled = _CUDA_DIRECT_RKS_COMPILED_CACHE.get(key)
    if compiled is not None:
        return compiled(*args)
    runner = _cached_cuda_direct_rks_runner(scf_inputs, config)
    return runner(*args)


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


def _host_float_unless_traced(value: Any) -> Any:
    return value if _contains_jax_tracer(value) else float(value)


def _host_array(value: Any, dtype: Any | None = None) -> np.ndarray:
    return np.asarray(jax.device_get(value), dtype=dtype)


def _restricted_reference_array_packaging(
    *,
    mo_coeff: Any,
    mo_occ: Any,
    mo_energy: Any,
    half_dm: Any,
    h1e: Any,
    atom_coords: Any,
    atom_charges: Any,
    overlap: Any,
    df_factors: Any | None,
    dtype: Any,
    traced: bool,
) -> dict[str, Any]:
    if traced:
        return {
            "mo_coeff": jnp.stack([jnp.asarray(mo_coeff, dtype=dtype)] * 2, axis=0),
            "mo_occ": jnp.stack([jnp.asarray(mo_occ, dtype=dtype)] * 2, axis=0),
            "mo_energy": jnp.stack([jnp.asarray(mo_energy, dtype=dtype)] * 2, axis=0),
            "rdm1": jnp.stack([jnp.asarray(half_dm, dtype=dtype)] * 2, axis=0),
            "h1e": jnp.asarray(h1e, dtype=dtype),
            "atom_coords": jnp.asarray(atom_coords, dtype=dtype),
            "atom_charges": jnp.asarray(atom_charges, dtype=dtype),
            "overlap_matrix": jnp.asarray(overlap, dtype=dtype),
            "df_factors": (
                jnp.asarray(df_factors, dtype=dtype) if df_factors is not None else None
            ),
        }

    host_dtype = np.dtype(dtype)
    mo_coeff_arr = _host_array(mo_coeff, host_dtype)
    mo_occ_arr = _host_array(mo_occ, host_dtype)
    mo_energy_arr = _host_array(mo_energy, host_dtype)
    half_dm_arr = _host_array(half_dm, host_dtype)
    return {
        "mo_coeff": np.stack([mo_coeff_arr, mo_coeff_arr], axis=0),
        "mo_occ": np.stack([mo_occ_arr, mo_occ_arr], axis=0),
        "mo_energy": np.stack([mo_energy_arr, mo_energy_arr], axis=0),
        "rdm1": np.stack([half_dm_arr, half_dm_arr], axis=0),
        "h1e": _host_array(h1e, host_dtype),
        "atom_coords": _host_array(atom_coords, host_dtype),
        "atom_charges": _host_array(atom_charges, host_dtype),
        "overlap_matrix": _host_array(overlap, host_dtype),
        "df_factors": (
            _host_array(df_factors, host_dtype) if df_factors is not None else None
        ),
    }


def _empty_rep_tensor_like(overlap: Any, *, traced: bool) -> Any:
    dtype = jnp.asarray(overlap).dtype if traced else np.asarray(overlap).dtype
    if traced:
        return jnp.zeros((0, 0, 0, 0), dtype=dtype)
    return np.zeros((0, 0, 0, 0), dtype=dtype)


def restricted_molecule_from_spec_with_jax_rks(
    *,
    atom: Any,
    basis: Any,
    xc_spec: str = "pbe",
    unit: str = "Angstrom",
    charge: int = 0,
    spin: int = 0,
    cart: bool = True,
    grids_level: int = 0,
    max_l: int = 3,
    rks_config: RKSConfig | None = None,
    grid_ao_backend: Literal["jax"] = "jax",
    integral_backend: Literal["jax", "libcint"] = "libcint",
    libcint_geometry_grad_policy: LibcintGeometryGradPolicy = "analytic",
    energy_target: float | None = None,
    compute_local_hfx_features: bool = False,
    compute_local_hfx_aux: bool = False,
    compute_local_pt2_features: bool = False,
    hfx_omega_values: tuple[float, ...] = (0.0, 0.4),
    hfx_chunk_size: int = 512,
    include_dipole_integrals: bool = True,
    precompile_eri: bool = False,
    precompile_eri_chunk_size: int = 512,
    verbose: int = 0,
    **mol_kwargs: Any,
) -> RestrictedMolecule:
    """Build a restricted strict-JAX RKS reference directly from molecule specs."""

    if isinstance(atom, MoleculeSpec):
        charge = int(atom.charge)
        spin = int(atom.spin)
    if int(spin) != 0:
        raise NotImplementedError(
            "restricted_molecule_from_spec_with_jax_rks only supports closed-shell systems."
        )
    if not bool(cart):
        raise NotImplementedError(
            "restricted_molecule_from_spec_with_jax_rks currently supports cart=True only."
        )

    xc_spec_resolved = str(xc_spec)
    parse_xc(xc_spec_resolved)
    cfg = RKSConfig(xc_spec=xc_spec_resolved) if rks_config is None else rks_config
    if cfg.xc_spec != xc_spec_resolved:
        cfg = replace(cfg, xc_spec=xc_spec_resolved)

    integral_backend_mode = str(integral_backend).lower()
    if integral_backend_mode not in {"jax", "libcint"}:
        raise ValueError(
            f"Unsupported integral_backend={integral_backend!r}. "
            "Expected 'jax' or 'libcint'."
        )
    libcint_grad_policy_mode = str(libcint_geometry_grad_policy).lower()
    if libcint_grad_policy_mode not in {"analytic", "error", "zero"}:
        raise ValueError(
            f"Unsupported libcint_geometry_grad_policy={libcint_geometry_grad_policy!r}. "
            "Expected 'analytic', 'error', or 'zero'."
        )

    exact_exchange_fraction = float(hybrid_coeff(xc_spec_resolved))
    input_cache_key = (
        _cuda_direct_reference_inputs_cache_key(
            atom=atom,
            basis=basis,
            xc_spec=xc_spec_resolved,
            unit=unit,
            charge=charge,
            spin=spin,
            cart=cart,
            grids_level=grids_level,
            max_l=max_l,
            config=cfg,
            grid_ao_backend=grid_ao_backend,
            integral_backend=integral_backend,
            libcint_geometry_grad_policy=libcint_geometry_grad_policy,
            include_dipole_integrals=include_dipole_integrals,
            precompile_eri=precompile_eri,
            precompile_eri_chunk_size=precompile_eri_chunk_size,
            verbose=verbose,
            mol_kwargs=mol_kwargs,
        )
        if cfg.jk_backend == "direct" and cfg.direct_jk_engine == "cuda"
        else None
    )
    scf_inputs = (
        _CUDA_DIRECT_RKS_INPUT_CACHE[input_cache_key]
        if input_cache_key in _CUDA_DIRECT_RKS_INPUT_CACHE
        else build_rks_integral_inputs(
            atom=atom,
            basis=basis,
            config=cfg,
            xc_spec=xc_spec_resolved,
            unit=unit,
            charge=charge,
            spin=spin,
            cart=cart,
            grids_level=grids_level,
            max_l=max_l,
            grid_ao_backend=grid_ao_backend,
            integral_backend=integral_backend,
            libcint_geometry_grad_policy=libcint_geometry_grad_policy,
            include_dipole_integrals=include_dipole_integrals,
            precompile_eri=precompile_eri,
            precompile_eri_chunk_size=precompile_eri_chunk_size,
            _precompile_eri_kernels=precompile_eri_kernels,
            verbose=verbose,
            **mol_kwargs,
        )
    )
    _cache_cuda_direct_rks_inputs(input_cache_key, scf_inputs)
    basis_cart = scf_inputs.basis
    s = scf_inputs.overlap
    h1e = scf_inputs.hcore
    eri = scf_inputs.eri
    eri_pair_matrix = scf_inputs.eri_pair_matrix
    df_factors = scf_inputs.df_factors
    coords = scf_inputs.coords
    weights = scf_inputs.grid_weights
    ao = scf_inputs.ao
    ao_deriv1 = scf_inputs.ao_deriv1
    ao_laplacian = scf_inputs.ao_laplacian
    dipole_integrals = scf_inputs.dipole_integrals
    nelectron = scf_inputs.nelectron
    rks_runner = (
        run_rks_from_integrals_traceable
        if scf_inputs.geometry_is_traced
        and scf_inputs.integral_backend in {"jax", "libcint"}
        and scf_inputs.grid_ao_backend == "jax"
        else run_rks_from_integrals
    )
    use_cached_cuda_direct_runner = (
        not scf_inputs.geometry_is_traced
        and cfg.jk_backend == "direct"
        and cfg.direct_jk_engine == "cuda"
        and cfg.iteration_backend == "lax"
        and cuda_ffi_available()
    )
    if use_cached_cuda_direct_runner:
        rks = _run_cached_cuda_direct_rks(scf_inputs, cfg)
        rks_is_traceable = True
    elif rks_runner is run_rks_from_integrals_traceable and cfg.iteration_backend != "lax":
        cfg = replace(cfg, iteration_backend="lax")
        rks = rks_runner(
            **scf_inputs.as_rks_kwargs(),
            config=cfg,
        )
        rks_is_traceable = rks_runner is run_rks_from_integrals_traceable
    else:
        rks_kwargs = scf_inputs.as_rks_kwargs()
        if (
            not scf_inputs.geometry_is_traced
            and cfg.jk_backend == "direct"
            and cfg.direct_jk_engine == "cuda"
            and cuda_ffi_available()
            and scf_inputs.direct_basis is not None
        ):
            rks_kwargs["direct_cuda_jk_builder"] = _cached_cuda_direct_jk_builder(
                scf_inputs.direct_basis,
                cfg,
            )
        rks = rks_runner(
            **rks_kwargs,
            config=cfg,
        )
        rks_is_traceable = rks_runner is run_rks_from_integrals_traceable
    if not rks_is_traceable and not rks.converged:
        if not (
            jnp.all(jnp.isfinite(rks.mo_coeff))
            and jnp.all(jnp.isfinite(rks.mo_energy))
            and jnp.all(jnp.isfinite(rks.density_matrix))
        ):
            raise RuntimeError(
                "Pure JAX RKS from molecule specs did not converge to a finite solution."
            )

    if rks_is_traceable:
        dm_total = jnp.asarray(rks.density_matrix)
        half_dm = dm_total / 2.0
        mo_coeff = jnp.asarray(rks.mo_coeff)
        mo_occ = jnp.asarray(rks.mo_occ) / 2.0
        mo_energy = jnp.asarray(rks.mo_energy)
    else:
        dm_total = _host_array(rks.density_matrix)
        half_dm = dm_total * 0.5
        mo_coeff = _host_array(rks.mo_coeff)
        mo_occ = _host_array(rks.mo_occ) * 0.5
        mo_energy = _host_array(rks.mo_energy)
    hfx_local = None
    hfx_nu = None
    pt2_local = None
    reference_eri_pair_matrix = None
    if compute_local_hfx_features:
        hfx_result = _local_hfx_features_from_basis_dm(
            basis_cart,
            ao,
            (half_dm, half_dm),
            coords,
            omega_values=tuple(float(omega) for omega in hfx_omega_values),
            chunk_size=hfx_chunk_size,
            return_nu=bool(compute_local_hfx_aux),
        )
        if compute_local_hfx_aux:
            hfx_local, hfx_nu = hfx_result
        else:
            hfx_local = hfx_result
    mf_energy = (
        _host_float_unless_traced(rks.total_energy)
        if energy_target is None
        else _host_float_unless_traced(energy_target)
    )
    nocc = nelectron // 2 if rks_is_traceable else int(np.count_nonzero(np.asarray(rks.mo_occ) > 1e-8))
    if cfg.jk_backend == "df":
        if df_factors is None:
            raise RuntimeError("DF backend requested but df_factors were not constructed.")
        eri_ovov = None
        eri_ovvo = None
        eri_oovv = None
        rep_tensor = _empty_rep_tensor_like(s, traced=rks_is_traceable)
    elif (
        cfg.jk_backend == "direct"
        and cfg.direct_jk_engine == "cuda"
        and cuda_ffi_available()
        and not compute_local_pt2_features
    ):
        eri_ovov = None
        eri_ovvo = None
        eri_oovv = None
        rep_tensor = _empty_rep_tensor_like(s, traced=rks_is_traceable)
    else:
        if eri is None and eri_pair_matrix is None:
            if cfg.jk_backend == "direct":
                eri_pair_matrix = scf_inputs.response_eri_pair_matrix()
            else:
                raise RuntimeError("Full ERI backend requested but exact ERI data is missing.")
        needs_exchange_slices = abs(exact_exchange_fraction) > 1e-14
        if eri_pair_matrix is not None:
            reference_eri_pair_matrix = (
                jnp.asarray(eri_pair_matrix, dtype=jnp.asarray(s).dtype)
                if rks_is_traceable
                else _host_array(eri_pair_matrix, np.asarray(s).dtype)
            )
            eri_ovov, eri_ovvo, eri_oovv = eri_pair_matrix_to_mo_eri_slices(
                eri_pair_matrix,
                rks.mo_coeff,
                nocc=nocc,
                include_oovv=needs_exchange_slices,
            )
            rep_tensor = _empty_rep_tensor_like(s, traced=rks_is_traceable)
        else:
            eri_ovov, eri_ovvo, eri_oovv = _restricted_response_eri_slices_from_mo_tensor(
                eri if rks_is_traceable else np.asarray(eri),
                rks.mo_coeff if rks_is_traceable else np.asarray(rks.mo_coeff),
                nocc,
                include_oovv=needs_exchange_slices,
            )
            rep_tensor = jnp.asarray(eri)

    if compute_local_pt2_features:
        pt2_local = _local_pt2_feature_from_restricted_orbitals(
            ao,
            mo_coeff,
            mo_occ,
            mo_energy,
            rep_tensor=rep_tensor,
            eri_ovov=eri_ovov,
            df_factors=df_factors,
            nocc=nocc,
            density_floor=cfg.density_floor,
        )

    reference_arrays = _restricted_reference_array_packaging(
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        half_dm=half_dm,
        h1e=h1e,
        atom_coords=basis_cart.atom_coords,
        atom_charges=basis_cart.atom_charges,
        overlap=s,
        df_factors=df_factors,
        dtype=jnp.asarray(s).dtype,
        traced=rks_is_traceable,
    )
    direct_basis_for_reference = None
    direct_cuda_jk_builder_for_reference = None
    direct_jk_engine_for_reference = None
    direct_scf_tol_for_reference = None
    if (
        cfg.jk_backend == "direct"
        and cfg.direct_jk_engine == "cuda"
        and cuda_ffi_available()
        and scf_inputs.direct_basis is not None
    ):
        direct_basis_for_reference = scf_inputs.direct_basis
        direct_cuda_jk_builder_for_reference = _cached_cuda_direct_jk_builder(
            scf_inputs.direct_basis,
            cfg,
        )
        direct_jk_engine_for_reference = str(cfg.direct_jk_engine)
        direct_scf_tol_for_reference = float(cfg.direct_scf_tol)

    return RestrictedMolecule(
        ao=ao,
        grid=QuadratureGrid(weights=weights, coords=coords),
        dipole_integrals=dipole_integrals,
        rep_tensor=rep_tensor,
        mo_coeff=reference_arrays["mo_coeff"],
        mo_occ=reference_arrays["mo_occ"],
        mo_energy=reference_arrays["mo_energy"],
        rdm1=reference_arrays["rdm1"],
        h1e=reference_arrays["h1e"],
        nuclear_repulsion=_host_float_unless_traced(rks.nuclear_repulsion),
        atom_coords=reference_arrays["atom_coords"],
        atom_charges=reference_arrays["atom_charges"],
        overlap_matrix=reference_arrays["overlap_matrix"],
        ao_deriv1=ao_deriv1,
        ao_laplacian=ao_laplacian,
        mf_energy=mf_energy,
        exact_exchange_fraction=exact_exchange_fraction,
        nocc=nocc,
        hfx_omega_values=(
            tuple(float(omega) for omega in hfx_omega_values)
            if compute_local_hfx_features
            else None
        ),
        hfx_local=hfx_local,
        hfx_nu=hfx_nu,
        pt2_local=pt2_local,
        df_factors=reference_arrays["df_factors"],
        eri_pair_matrix=reference_eri_pair_matrix,
        eri_ovov=eri_ovov,
        eri_ovvo=eri_ovvo,
        eri_oovv=eri_oovv,
        scf_converged=bool(rks.converged),
        direct_jk_engine=direct_jk_engine_for_reference,
        direct_scf_tol=direct_scf_tol_for_reference,
        direct_basis=direct_basis_for_reference,
        direct_cuda_jk_builder=direct_cuda_jk_builder_for_reference,
    )


def unrestricted_molecule_from_spec_with_jax_uks(
    *,
    atom: Any,
    basis: Any,
    xc_spec: str = "pbe",
    unit: str = "Angstrom",
    charge: int = 0,
    spin: int = 1,
    cart: bool = True,
    grids_level: int = 0,
    max_l: int = 3,
    uks_config: UKSConfig | None = None,
    grid_ao_backend: Literal["jax"] = "jax",
    integral_backend: Literal["jax", "libcint"] = "libcint",
    libcint_geometry_grad_policy: LibcintGeometryGradPolicy = "error",
    energy_target: float | None = None,
    compute_local_hfx_features: bool = False,
    compute_local_hfx_aux: bool = False,
    hfx_omega_values: tuple[float, ...] = (0.0, 0.4),
    hfx_chunk_size: int = 512,
    precompile_eri: bool = False,
    precompile_eri_chunk_size: int = 512,
    verbose: int = 0,
    **mol_kwargs: Any,
) -> UnrestrictedMolecule:
    """Build an unrestricted strict-JAX UKS reference directly from molecule specs."""

    if not bool(cart):
        raise NotImplementedError(
            "unrestricted_molecule_from_spec_with_jax_uks currently supports cart=True only."
        )
    if isinstance(atom, MoleculeSpec):
        charge = int(atom.charge)
        spin = int(atom.spin)

    xc_spec_resolved = str(xc_spec)
    parse_xc(xc_spec_resolved)
    cfg = UKSConfig(xc_spec=xc_spec_resolved) if uks_config is None else uks_config
    if cfg.xc_spec != xc_spec_resolved:
        cfg = UKSConfig(
            xc_spec=xc_spec_resolved,
            max_cycle=cfg.max_cycle,
            conv_tol=cfg.conv_tol,
            conv_tol_density=cfg.conv_tol_density,
            damping=cfg.damping,
            level_shift=cfg.level_shift,
            orthogonalization_eps=cfg.orthogonalization_eps,
            density_floor=cfg.density_floor,
            potential_clip=cfg.potential_clip,
        )

    scf_inputs = build_uks_integral_inputs(
        atom=atom,
        basis=basis,
        config=cfg,
        xc_spec=xc_spec_resolved,
        unit=unit,
        charge=charge,
        spin=spin,
        cart=cart,
        grids_level=grids_level,
        max_l=max_l,
        grid_ao_backend=grid_ao_backend,
        integral_backend=integral_backend,
        libcint_geometry_grad_policy=libcint_geometry_grad_policy,
        precompile_eri=precompile_eri,
        precompile_eri_chunk_size=precompile_eri_chunk_size,
        _precompile_eri_kernels=precompile_eri_kernels,
        verbose=verbose,
        **mol_kwargs,
    )
    basis_cart = scf_inputs.basis
    s = scf_inputs.overlap
    h1e = scf_inputs.hcore
    eri = scf_inputs.eri
    coords = scf_inputs.coords
    weights = scf_inputs.grid_weights
    ao = scf_inputs.ao
    ao_deriv1 = scf_inputs.ao_deriv1
    ao_laplacian = scf_inputs.ao_laplacian
    dipole_integrals = scf_inputs.dipole_integrals

    uks = run_uks_from_integrals(
        **scf_inputs.as_uks_kwargs(),
        config=cfg,
    )
    uks_is_traceable = _contains_jax_tracer(uks.total_energy)
    if not uks_is_traceable and not uks.converged:
        if not (
            jnp.all(jnp.isfinite(uks.mo_coeff_alpha))
            and jnp.all(jnp.isfinite(uks.mo_coeff_beta))
            and jnp.all(jnp.isfinite(uks.mo_energy_alpha))
            and jnp.all(jnp.isfinite(uks.mo_energy_beta))
            and jnp.all(jnp.isfinite(uks.density_matrix_alpha))
            and jnp.all(jnp.isfinite(uks.density_matrix_beta))
        ):
            raise RuntimeError(
                "Pure JAX UKS from molecule specs did not converge to a finite solution."
            )

    hfx_local = None
    hfx_nu = None
    if compute_local_hfx_features:
        hfx_result = _local_hfx_features_from_basis_dm(
            basis_cart,
            ao,
            (uks.density_matrix_alpha, uks.density_matrix_beta),
            coords,
            omega_values=tuple(float(omega) for omega in hfx_omega_values),
            chunk_size=hfx_chunk_size,
            return_nu=bool(compute_local_hfx_aux),
        )
        if compute_local_hfx_aux:
            hfx_local, hfx_nu = hfx_result
        else:
            hfx_local = hfx_result

    mf_energy = (
        _host_float_unless_traced(uks.total_energy)
        if energy_target is None
        else _host_float_unless_traced(energy_target)
    )
    nocc_alpha = int(np.count_nonzero(np.asarray(uks.mo_occ_alpha) > 1e-8))
    nocc_beta = int(np.count_nonzero(np.asarray(uks.mo_occ_beta) > 1e-8))
    return UnrestrictedMolecule(
        ao=ao,
        grid=QuadratureGrid(weights=weights, coords=coords),
        dipole_integrals=dipole_integrals,
        rep_tensor=jnp.asarray(eri),
        mo_coeff=jnp.stack([uks.mo_coeff_alpha, uks.mo_coeff_beta], axis=0),
        mo_occ=jnp.stack([uks.mo_occ_alpha, uks.mo_occ_beta], axis=0),
        mo_energy=jnp.stack([uks.mo_energy_alpha, uks.mo_energy_beta], axis=0),
        rdm1=jnp.stack([uks.density_matrix_alpha, uks.density_matrix_beta], axis=0),
        h1e=jnp.asarray(h1e),
        nuclear_repulsion=_host_float_unless_traced(uks.nuclear_repulsion),
        atom_coords=jnp.asarray(basis_cart.atom_coords),
        atom_charges=jnp.asarray(basis_cart.atom_charges),
        overlap_matrix=jnp.asarray(s),
        ao_deriv1=ao_deriv1,
        ao_laplacian=ao_laplacian,
        mf_energy=mf_energy,
        exact_exchange_fraction=float(uks.exact_exchange_fraction),
        nocc_alpha=nocc_alpha,
        nocc_beta=nocc_beta,
        hfx_omega_values=(
            tuple(float(omega) for omega in hfx_omega_values)
            if compute_local_hfx_features
            else None
        ),
        hfx_local=hfx_local,
        hfx_nu=hfx_nu,
    )


precompile_restricted_cuda_direct_rks_reference = precompile_restricted_cuda_direct_rks_solver
restricted_reference_from_spec_with_jax_rks = restricted_molecule_from_spec_with_jax_rks
unrestricted_reference_from_spec_with_jax_uks = unrestricted_molecule_from_spec_with_jax_uks

__all__ = [
    "_cuda_direct_basis_cache_key",
    "_restricted_reference_array_packaging",
    "precompile_restricted_cuda_direct_rks_solver",
    "restricted_molecule_from_spec_with_jax_rks",
    "unrestricted_molecule_from_spec_with_jax_uks",
]
