from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import os
import warnings
from typing import Any, Literal

import numpy as np
import jax
import jax.numpy as jnp
from jax.lax import Precision

from .data.grid_ao import evaluate_cartesian_ao, evaluate_cartesian_ao_with_derivatives
from .data.integrals import (
    precompile_eri_kernels,
    rinv_matrices,
)
from .data.molecule import MoleculeSpec
from .data.integrals.libcint_autodiff import (
    LibcintGeometryGradPolicy,
)
from .df import df_factors_to_mo_eri_slices
from .jax_libxc import hybrid_coeff, parse_xc
from .scf import (
    RKSConfig,
    UKSConfig,
    build_rks_integral_inputs,
    build_uks_integral_inputs,
    run_rks_from_integrals,
    run_rks_from_integrals_traceable,
    run_uks_from_integrals,
)
from .scf.cuda_direct_jk import (
    CudaDirectJKBuilder,
    cuda_ffi_available,
)
from .scf.packed_eri import eri_pair_matrix_to_mo_eri_slices


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


def _cuda_direct_jk_builder_cache_key(basis: Any, config: RKSConfig) -> tuple[Any, ...]:
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
        geometry_part,
    )


def _cached_cuda_direct_jk_builder(basis: Any, config: RKSConfig) -> Any:
    key = _cuda_direct_jk_builder_cache_key(basis, config)
    builder = _CUDA_DIRECT_JK_BUILDER_CACHE.get(key)
    if builder is not None:
        return builder
    builder = CudaDirectJKBuilder(
        basis,
        include_pair_metadata=True,
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


def precompile_restricted_cuda_direct_rks_reference(
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
    grid_ao_backend: Literal["pyscf", "jax"] = "jax",
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


def _pytree_dataclass(*, static_fields: tuple[str, ...] = ()):
    static_field_names = frozenset(static_fields)

    def decorator(cls):
        def tree_flatten(self):
            child_names = []
            children = []
            static_items = []
            for field in fields(self):
                value = getattr(self, field.name)
                if field.name in static_field_names or value is None:
                    static_items.append((field.name, value))
                else:
                    child_names.append(field.name)
                    children.append(value)
            return tuple(children), (tuple(child_names), tuple(static_items))

        @classmethod
        def tree_unflatten(cls_, aux_data, children):
            child_names, static_items = aux_data
            kwargs = {name: value for name, value in static_items}
            kwargs.update({name: value for name, value in zip(child_names, children, strict=True)})
            return cls_(**kwargs)

        cls.tree_flatten = tree_flatten
        cls.tree_unflatten = tree_unflatten
        return jax.tree_util.register_pytree_node_class(cls)

    return decorator


@_pytree_dataclass()
@dataclass(frozen=True)
class GridReference:
    """Minimal quadrature grid container used by the TDDFT modules."""

    weights: jnp.ndarray
    coords: jnp.ndarray | None = None


@_pytree_dataclass(
    static_fields=(
        "nocc",
        "scf_converged",
        "direct_jk_engine",
        "direct_scf_tol",
        "direct_basis",
        "direct_cuda_jk_builder",
    )
)
@dataclass(frozen=True)
class RestrictedMoleculeReference:
    """Minimal restricted molecule container used across TD-GradDFT."""

    ao: jnp.ndarray
    grid: GridReference
    dipole_integrals: jnp.ndarray
    rep_tensor: jnp.ndarray
    mo_coeff: jnp.ndarray
    mo_occ: jnp.ndarray
    mo_energy: jnp.ndarray
    rdm1: jnp.ndarray
    h1e: jnp.ndarray
    nuclear_repulsion: float
    atom_coords: jnp.ndarray | None = None
    atom_charges: jnp.ndarray | None = None
    overlap_matrix: jnp.ndarray | None = None
    ao_deriv1: jnp.ndarray | None = None
    ao_laplacian: jnp.ndarray | None = None
    mf_energy: float | None = None
    exact_exchange_fraction: float = 0.0
    nocc: int | None = None
    hfx_omega_values: tuple[float, ...] | None = None
    hfx_local: jnp.ndarray | None = None  # shape: (2, ngrids, n_omega)
    hfx_nu: jnp.ndarray | None = None  # shape: (n_omega, ngrids, nao, nao)
    pt2_local: jnp.ndarray | None = None  # shape: (ngrids,)
    scf_initial_density: jnp.ndarray | None = None  # shape: (nao, nao)
    df_factors: jnp.ndarray | None = None  # shape: (naux, nao, nao)
    eri_pair_matrix: jnp.ndarray | None = None  # shape: (nao*(nao+1)//2, nao*(nao+1)//2)
    eri_ovov: jnp.ndarray | None = None  # shape: (nocc, nvir, nocc, nvir)
    eri_ovvo: jnp.ndarray | None = None  # shape: (nocc, nvir, nvir, nocc)
    eri_oovv: jnp.ndarray | None = None  # shape: (nocc, nocc, nvir, nvir)
    scf_converged: bool | None = None
    direct_jk_engine: str | None = None
    direct_scf_tol: float | None = None
    direct_basis: Any | None = None
    direct_cuda_jk_builder: Any | None = None

    def density(self) -> jnp.ndarray:
        return jnp.einsum("spq,rp,rq->rs", self.rdm1, self.ao, self.ao)


@_pytree_dataclass(static_fields=("nocc_alpha", "nocc_beta"))
@dataclass(frozen=True)
class UnrestrictedMoleculeReference:
    """Minimal unrestricted molecule container used across TD-GradDFT."""

    ao: jnp.ndarray
    grid: GridReference
    dipole_integrals: jnp.ndarray
    rep_tensor: jnp.ndarray
    mo_coeff: jnp.ndarray
    mo_occ: jnp.ndarray
    mo_energy: jnp.ndarray
    rdm1: jnp.ndarray
    h1e: jnp.ndarray
    nuclear_repulsion: float
    atom_coords: jnp.ndarray | None = None
    atom_charges: jnp.ndarray | None = None
    overlap_matrix: jnp.ndarray | None = None
    ao_deriv1: jnp.ndarray | None = None
    ao_laplacian: jnp.ndarray | None = None
    mf_energy: float | None = None
    exact_exchange_fraction: float = 0.0
    nocc_alpha: int | None = None
    nocc_beta: int | None = None
    hfx_omega_values: tuple[float, ...] | None = None
    hfx_local: jnp.ndarray | None = None  # shape: (2, ngrids, n_omega)
    hfx_nu: jnp.ndarray | None = None  # shape: (n_omega, ngrids, nao, nao)
    scf_initial_density: jnp.ndarray | None = None  # shape: (nao, nao)

    def density(self) -> jnp.ndarray:
        return jnp.einsum("spq,rp,rq->r", self.rdm1, self.ao, self.ao)


def _charge_center(mol: Any) -> jnp.ndarray:
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    return jnp.asarray(jnp.einsum("z,zr->r", charges, coords) / charges.sum())


def _int1e_grids_name(mol: Any) -> str:
    return "int1e_grids_cart" if bool(getattr(mol, "cart", False)) else "int1e_grids_sph"


def _int1e_rinv_name(mol: Any) -> str:
    return "int1e_rinv_cart" if bool(getattr(mol, "cart", False)) else "int1e_rinv_sph"


def _eval_grid_ao(
    mol: Any,
    basis: Any,
    coords: Any,
    *,
    backend: Literal["pyscf", "jax"] = "pyscf",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    backend = str(backend).lower()
    coords_arr = jnp.asarray(coords)
    if backend == "jax":
        return evaluate_cartesian_ao_with_derivatives(basis, coords_arr, deriv=1)
    if backend == "pyscf":
        try:
            from pyscf.dft import numint
        except ModuleNotFoundError as exc:
            raise ImportError("PySCF is required for backend='pyscf'.") from exc
        ao_deriv1 = jnp.asarray(numint.eval_ao(mol, np.asarray(coords_arr), deriv=1))
        return ao_deriv1[0], ao_deriv1
    raise ValueError(f"Unsupported grid AO backend={backend!r}. Expected 'pyscf' or 'jax'.")


def _eval_grid_ao_laplacian(
    mol: Any,
    basis: Any,
    coords: Any,
    *,
    backend: Literal["pyscf", "jax"] = "pyscf",
) -> jnp.ndarray:
    backend = str(backend).lower()
    coords_arr = jnp.asarray(coords)
    if backend == "jax":
        ao_deriv2 = evaluate_cartesian_ao(basis, coords_arr, deriv=2)
        return ao_deriv2[4]
    if backend == "pyscf":
        try:
            from pyscf.dft import numint
        except ModuleNotFoundError as exc:
            raise ImportError("PySCF is required for backend='pyscf'.") from exc
        ao_deriv2 = jnp.asarray(numint.eval_ao(mol, np.asarray(coords_arr), deriv=2))
        if ao_deriv2.shape[0] < 10:
            raise ValueError(
                "PySCF deriv=2 AO evaluation must expose second derivatives."
            )
        return ao_deriv2[4] + ao_deriv2[7] + ao_deriv2[9]
    raise ValueError(f"Unsupported grid AO backend={backend!r}. Expected 'pyscf' or 'jax'.")


def _local_hfx_features_from_dm(
    mol: Any,
    ao: np.ndarray,
    dm_spin: tuple[np.ndarray, np.ndarray],
    coords: np.ndarray,
    *,
    omega_values: tuple[float, ...],
    chunk_size: int = 512,
    return_nu: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute DM21-style local HF exchange densities for each spin/omega."""

    dm_a, dm_b = dm_spin
    e_a = ao @ dm_a
    e_b = ao @ dm_b
    ngrid = int(coords.shape[0])
    n_omega = len(omega_values)
    nao = int(ao.shape[1])
    hfx = np.zeros((2, ngrid, n_omega), dtype=np.float64)
    nu_cache = (
        np.zeros((n_omega, ngrid, nao, nao), dtype=np.float64) if return_nu else None
    )
    int1e_grids = _int1e_grids_name(mol)
    int1e_rinv = _int1e_rinv_name(mol)

    for omega_idx, omega in enumerate(omega_values):
        for start in range(0, ngrid, int(chunk_size)):
            end = min(start + int(chunk_size), ngrid)
            coords_chunk = coords[start:end]
            try:
                with mol.with_range_coulomb(omega=float(omega)):
                    nu = mol.intor(int1e_grids, hermi=1, grids=coords_chunk)
            except TypeError:
                nu_list = []
                with mol.with_rinv_zeta(zeta=float(omega) * float(omega)):
                    for coord in coords_chunk:
                        with mol.with_rinv_origin(coord):
                            nu_list.append(mol.intor(int1e_rinv, hermi=1))
                nu = np.asarray(nu_list)
            if nu_cache is not None:
                nu_cache[omega_idx, start:end] = nu

            e_a_chunk = e_a[start:end]
            e_b_chunk = e_b[start:end]
            fxx_a = np.einsum("gbc,gc->gb", nu, e_a_chunk, optimize=True)
            fxx_b = np.einsum("gbc,gc->gb", nu, e_b_chunk, optimize=True)
            hfx[0, start:end, omega_idx] = -0.5 * np.einsum(
                "gb,gb->g", e_a_chunk, fxx_a, optimize=True
            )
            hfx[1, start:end, omega_idx] = -0.5 * np.einsum(
                "gb,gb->g", e_b_chunk, fxx_b, optimize=True
            )
    if nu_cache is None:
        return hfx
    return hfx, nu_cache


def _local_hfx_features_from_nu_cache(
    ao: Any,
    dm_spin: tuple[Any, Any],
    nu_cache: Any,
) -> jnp.ndarray:
    ao_arr = jnp.asarray(ao)
    dm_a, dm_b = (jnp.asarray(dm_spin[0]), jnp.asarray(dm_spin[1]))
    nu = jnp.asarray(nu_cache)

    e_a = jnp.einsum("gp,pq->gq", ao_arr, dm_a, precision=Precision.HIGHEST)
    e_b = jnp.einsum("gp,pq->gq", ao_arr, dm_b, precision=Precision.HIGHEST)
    fxx_a = jnp.einsum("wgbc,gc->wgb", nu, e_a, precision=Precision.HIGHEST)
    fxx_b = jnp.einsum("wgbc,gc->wgb", nu, e_b, precision=Precision.HIGHEST)
    exx_a = -0.5 * jnp.einsum("gq,wgq->wg", e_a, fxx_a, precision=Precision.HIGHEST)
    exx_b = -0.5 * jnp.einsum("gq,wgq->wg", e_b, fxx_b, precision=Precision.HIGHEST)
    exx = jnp.stack([exx_a.T, exx_b.T], axis=0)
    return jnp.nan_to_num(exx, nan=0.0, posinf=0.0, neginf=0.0)


def _local_hfx_features_from_basis_dm(
    basis: Any,
    ao: Any,
    dm_spin: tuple[Any, Any],
    coords: Any,
    *,
    omega_values: tuple[float, ...],
    chunk_size: int = 512,
    return_nu: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    coords_arr = jnp.asarray(coords)
    ao_arr = jnp.asarray(ao)
    ngrid = int(coords_arr.shape[0])
    hfx_chunks: list[jnp.ndarray] = []
    nu_chunks_per_omega: list[jnp.ndarray] = []

    for omega in omega_values:
        zeta = None if abs(float(omega)) < 1e-14 else float(omega) * float(omega)
        omega_nu_chunks: list[jnp.ndarray] = []
        omega_hfx_chunks: list[jnp.ndarray] = []
        for start in range(0, ngrid, int(chunk_size)):
            end = min(start + int(chunk_size), ngrid)
            nu_chunk = rinv_matrices(
                basis,
                coords_arr[start:end],
                zeta=zeta,
                engine="auto",
                grid_chunk_size=min(int(chunk_size), max(1, end - start)),
            )
            omega_hfx_chunks.append(
                _local_hfx_features_from_nu_cache(
                    ao_arr[start:end],
                    dm_spin,
                    nu_chunk[None, ...],
                )[:, :, 0]
            )
            if return_nu:
                omega_nu_chunks.append(nu_chunk)
        hfx_chunks.append(jnp.concatenate(omega_hfx_chunks, axis=1))
        if return_nu:
            nu_chunks_per_omega.append(jnp.concatenate(omega_nu_chunks, axis=0))

    hfx_local = jnp.stack(hfx_chunks, axis=-1)
    if not return_nu:
        return hfx_local
    nu_cache = jnp.stack(nu_chunks_per_omega, axis=0)
    return hfx_local, nu_cache


def _local_pt2_feature_from_restricted_orbitals(
    ao: Any,
    mo_coeff: Any,
    mo_occ: Any,
    mo_energy: Any,
    *,
    rep_tensor: Any | None = None,
    eri_ovov: Any | None = None,
    df_factors: Any | None = None,
    nocc: int | None = None,
    occupation_tolerance: float = 1e-8,
    density_floor: float = 1e-12,
) -> jnp.ndarray:
    ao_arr = jnp.asarray(ao)
    mo_coeff_arr = jnp.asarray(mo_coeff)
    mo_occ_arr = jnp.asarray(mo_occ)
    mo_energy_arr = jnp.asarray(mo_energy)

    if mo_coeff_arr.ndim == 3:
        mo_coeff_arr = mo_coeff_arr[0]
    if mo_occ_arr.ndim == 2:
        mo_occ_arr = mo_occ_arr[0]
    if mo_energy_arr.ndim == 2:
        mo_energy_arr = mo_energy_arr[0]

    nocc_int = int(nocc) if nocc is not None else int(jnp.count_nonzero(mo_occ_arr > occupation_tolerance))
    nmo = int(mo_coeff_arr.shape[1])
    if nocc_int <= 0 or nocc_int >= nmo:
        raise ValueError("PT2 local feature requires at least one occupied and one virtual orbital.")

    orbo = mo_coeff_arr[:, :nocc_int]
    orbv = mo_coeff_arr[:, nocc_int:]
    eps_occ = mo_energy_arr[:nocc_int]
    eps_vir = mo_energy_arr[nocc_int:]

    eri_ovov_arr = None if eri_ovov is None else jnp.asarray(eri_ovov)
    if eri_ovov_arr is None:
        if df_factors is not None:
            factors = jnp.asarray(df_factors)
            if factors.size != 0:
                eri_ovov_arr, _, _ = df_factors_to_mo_eri_slices(
                    factors,
                    mo_coeff_arr,
                    nocc_int,
                    include_oovv=False,
                )
        if eri_ovov_arr is None:
            if rep_tensor is None:
                raise ValueError(
                    "PT2 local feature requires either rep_tensor, eri_ovov, or df_factors."
                )
            rep = jnp.asarray(rep_tensor)
            if rep.size == 0:
                raise ValueError(
                    "PT2 local feature cannot be constructed from an empty rep_tensor without df_factors."
                )
            eri_ovov_arr = jnp.einsum(
                "pqrs,pi,qa,rj,sb->iajb",
                rep,
                orbo,
                orbv,
                orbo,
                orbv,
                precision=Precision.HIGHEST,
            )

    denom = (
        eps_occ[:, None, None, None]
        + eps_occ[None, None, :, None]
        - eps_vir[None, :, None, None]
        - eps_vir[None, None, None, :]
    )
    denom = jnp.where(jnp.abs(denom) > density_floor, denom, -density_floor)
    direct = eri_ovov_arr
    exchange = jnp.transpose(eri_ovov_arr, (0, 3, 2, 1))
    pair_weights = (2.0 * direct - exchange) / denom

    rho_o = jnp.einsum("rp,pi->ri", ao_arr, orbo, precision=Precision.HIGHEST)
    rho_v = jnp.einsum("rp,pa->ra", ao_arr, orbv, precision=Precision.HIGHEST)
    rho_ov = jnp.einsum("ri,ra->ria", rho_o, rho_v, precision=Precision.HIGHEST)

    if df_factors is not None and jnp.asarray(df_factors).size != 0:
        factors = jnp.asarray(df_factors)
        grid_aux = jnp.einsum(
            "Qpq,gp,gq->gQ",
            factors,
            ao_arr,
            ao_arr,
            precision=Precision.HIGHEST,
        )
        qjb = jnp.einsum(
            "Qrs,rj,sb->Qjb",
            factors,
            orbo,
            orbv,
            precision=Precision.HIGHEST,
        )
        pair_potential = jnp.einsum(
            "gQ,Qjb->gjb",
            grid_aux,
            qjb,
            precision=Precision.HIGHEST,
        )
    else:
        rep = jnp.asarray(rep_tensor)
        pair_potential = jnp.einsum(
            "gp,gq,pqrs,rj,sb->gjb",
            ao_arr,
            ao_arr,
            rep,
            orbo,
            orbv,
            precision=Precision.HIGHEST,
        )
    local_energy = jnp.einsum(
        "ria,rjb,iajb->r",
        rho_ov,
        pair_potential,
        pair_weights,
        precision=Precision.HIGHEST,
    )
    return jnp.nan_to_num(local_energy, nan=0.0, posinf=0.0, neginf=0.0)


def _restricted_response_eri_slices_from_mo_tensor(
    rep_tensor: Any,
    mo_coeff: Any,
    nocc: int,
    *,
    include_oovv: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
    nocc = int(nocc)
    coeff = jnp.asarray(mo_coeff)
    orbo = coeff[:, :nocc]
    orbv = coeff[:, nocc:]
    rep = jnp.asarray(rep_tensor)
    eri_ovov = jnp.einsum(
        "pqrs,pi,qa,rj,sb->iajb",
        rep,
        orbo,
        orbv,
        orbo,
        orbv,
        precision=Precision.HIGHEST,
    )
    eri_ovvo = jnp.einsum(
        "pqrs,pi,qa,rb,sj->iabj",
        rep,
        orbo,
        orbv,
        orbv,
        orbo,
        precision=Precision.HIGHEST,
    )
    if not include_oovv:
        return eri_ovov, eri_ovvo, None
    eri_oovv = jnp.einsum(
        "pqrs,pi,qj,ra,sb->ijab",
        rep,
        orbo,
        orbo,
        orbv,
        orbv,
        precision=Precision.HIGHEST,
    )
    return eri_ovov, eri_ovvo, eri_oovv


def _restricted_response_eri_slices_from_pyscf_mf(
    mf: Any,
    nocc: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return _restricted_response_eri_slices_from_mo_tensor(
        jnp.asarray(mf.mol.intor("int2e")),
        jnp.asarray(mf.mo_coeff),
        nocc,
    )


def restricted_reference_from_pyscf(
    mf: Any,
    *,
    compute_local_hfx_features: bool = False,
    compute_local_hfx_aux: bool = False,
    compute_local_pt2_features: bool = False,
    hfx_omega_values: tuple[float, ...] = (0.0, 0.4),
    hfx_chunk_size: int = 512,
) -> RestrictedMoleculeReference:
    """Legacy compatibility wrapper around the PySCF-backed restricted builder."""

    warnings.warn(
        "td_graddft.reference.restricted_reference_from_pyscf is legacy. "
        "Prefer td_graddft.reference_legacy or strict-JAX reference builders.",
        DeprecationWarning,
        stacklevel=2,
    )
    from .reference_legacy import restricted_reference_from_pyscf as _impl

    return _impl(
        mf,
        compute_local_hfx_features=compute_local_hfx_features,
        compute_local_hfx_aux=compute_local_hfx_aux,
        compute_local_pt2_features=compute_local_pt2_features,
        hfx_omega_values=hfx_omega_values,
        hfx_chunk_size=hfx_chunk_size,
    )


def unrestricted_reference_from_pyscf(mf: Any) -> UnrestrictedMoleculeReference:
    """Legacy compatibility wrapper around the PySCF-backed unrestricted builder."""

    warnings.warn(
        "td_graddft.reference.unrestricted_reference_from_pyscf is legacy. "
        "Prefer td_graddft.reference_legacy for PySCF-backed compatibility.",
        DeprecationWarning,
        stacklevel=2,
    )
    from .reference_legacy import unrestricted_reference_from_pyscf as _impl

    return _impl(mf)


def restricted_reference_from_pyscf_with_jax_rhf(
    mf: Any,
    *,
    max_l: int = 1,
    rhf_config: Any = None,
    energy_target: float | None = None,
    grid_ao_backend: Literal["pyscf", "jax"] = "pyscf",
    compute_local_hfx_features: bool = False,
    compute_local_hfx_aux: bool = False,
    compute_local_pt2_features: bool = False,
    hfx_omega_values: tuple[float, ...] = (0.0, 0.4),
    hfx_chunk_size: int = 512,
) -> RestrictedMoleculeReference:
    """Legacy compatibility wrapper around the PySCF-input JAX-RHF builder."""

    warnings.warn(
        "td_graddft.reference.restricted_reference_from_pyscf_with_jax_rhf is legacy. "
        "Prefer strict-JAX spec-driven builders.",
        DeprecationWarning,
        stacklevel=2,
    )
    from .reference_legacy import restricted_reference_from_pyscf_with_jax_rhf as _impl

    return _impl(
        mf,
        max_l=max_l,
        rhf_config=rhf_config,
        energy_target=energy_target,
        grid_ao_backend=grid_ao_backend,
        compute_local_hfx_features=compute_local_hfx_features,
        compute_local_hfx_aux=compute_local_hfx_aux,
        compute_local_pt2_features=compute_local_pt2_features,
        hfx_omega_values=hfx_omega_values,
        hfx_chunk_size=hfx_chunk_size,
    )


def restricted_reference_from_pyscf_with_jax_rks(
    mf: Any,
    *,
    max_l: int = 3,
    rks_config: Any = None,
    xc_spec: str | None = None,
    energy_target: float | None = None,
    grid_ao_backend: Literal["pyscf", "jax"] = "pyscf",
    compute_local_hfx_features: bool = False,
    compute_local_hfx_aux: bool = False,
    compute_local_pt2_features: bool = False,
    hfx_omega_values: tuple[float, ...] = (0.0, 0.4),
    hfx_chunk_size: int = 512,
) -> RestrictedMoleculeReference:
    """Legacy compatibility wrapper around the PySCF-input JAX-RKS builder."""

    warnings.warn(
        "td_graddft.reference.restricted_reference_from_pyscf_with_jax_rks is legacy. "
        "Prefer strict-JAX spec-driven builders.",
        DeprecationWarning,
        stacklevel=2,
    )
    from .reference_legacy import restricted_reference_from_pyscf_with_jax_rks as _impl

    return _impl(
        mf,
        max_l=max_l,
        rks_config=rks_config,
        xc_spec=xc_spec,
        energy_target=energy_target,
        grid_ao_backend=grid_ao_backend,
        compute_local_hfx_features=compute_local_hfx_features,
        compute_local_hfx_aux=compute_local_hfx_aux,
        compute_local_pt2_features=compute_local_pt2_features,
        hfx_omega_values=hfx_omega_values,
        hfx_chunk_size=hfx_chunk_size,
    )


def restricted_reference_from_pyscf_spec_with_jax_rks(
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
    rks_config: Any = None,
    grid_ao_backend: Literal["pyscf", "jax"] = "jax",
    energy_target: float | None = None,
    compute_local_hfx_features: bool = False,
    compute_local_hfx_aux: bool = False,
    compute_local_pt2_features: bool = False,
    hfx_omega_values: tuple[float, ...] = (0.0, 0.4),
    hfx_chunk_size: int = 512,
    verbose: int = 0,
    **mol_kwargs: Any,
) -> RestrictedMoleculeReference:
    """Legacy compatibility alias for the strict-JAX spec-driven builder."""

    warnings.warn(
        "td_graddft.reference.restricted_reference_from_pyscf_spec_with_jax_rks is legacy. "
        "Prefer restricted_reference_from_spec_with_jax_rks.",
        DeprecationWarning,
        stacklevel=2,
    )
    from .reference_legacy import restricted_reference_from_pyscf_spec_with_jax_rks as _impl

    return _impl(
        atom=atom,
        basis=basis,
        xc_spec=xc_spec,
        unit=unit,
        charge=charge,
        spin=spin,
        cart=cart,
        grids_level=grids_level,
        max_l=max_l,
        rks_config=rks_config,
        grid_ao_backend=grid_ao_backend,
        energy_target=energy_target,
        compute_local_hfx_features=compute_local_hfx_features,
        compute_local_hfx_aux=compute_local_hfx_aux,
        compute_local_pt2_features=compute_local_pt2_features,
        hfx_omega_values=hfx_omega_values,
        hfx_chunk_size=hfx_chunk_size,
        verbose=verbose,
        **mol_kwargs,
    )


def restricted_reference_from_spec_with_jax_rks(
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
    grid_ao_backend: Literal["pyscf", "jax"] = "jax",
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
) -> RestrictedMoleculeReference:
    """Build a restricted strict-JAX RKS reference directly from molecule specs."""

    if isinstance(atom, MoleculeSpec):
        charge = int(atom.charge)
        spin = int(atom.spin)
    if int(spin) != 0:
        raise NotImplementedError(
            "restricted_reference_from_spec_with_jax_rks only supports closed-shell systems."
        )
    if not bool(cart):
        raise NotImplementedError(
            "restricted_reference_from_spec_with_jax_rks currently supports cart=True only."
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

    return RestrictedMoleculeReference(
        ao=ao,
        grid=GridReference(weights=weights, coords=coords),
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


def unrestricted_reference_from_spec_with_jax_uks(
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
    grid_ao_backend: Literal["pyscf", "jax"] = "jax",
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
) -> UnrestrictedMoleculeReference:
    """Build an unrestricted strict-JAX UKS reference directly from molecule specs."""

    if not bool(cart):
        raise NotImplementedError(
            "unrestricted_reference_from_spec_with_jax_uks currently supports cart=True only."
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
    return UnrestrictedMoleculeReference(
        ao=ao,
        grid=GridReference(weights=weights, coords=coords),
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


def unrestricted_reference_from_pyscf_with_jax_uks(
    mf: Any,
    *,
    max_l: int = 3,
    uks_config: Any = None,
    xc_spec: str | None = None,
    energy_target: float | None = None,
    grid_ao_backend: Literal["pyscf", "jax"] = "pyscf",
    compute_local_hfx_features: bool = False,
    compute_local_hfx_aux: bool = False,
    hfx_omega_values: tuple[float, ...] = (0.0, 0.4),
    hfx_chunk_size: int = 512,
) -> UnrestrictedMoleculeReference:
    """Legacy compatibility wrapper around the PySCF-input JAX-UKS builder."""

    warnings.warn(
        "td_graddft.reference.unrestricted_reference_from_pyscf_with_jax_uks is legacy. "
        "Prefer strict-JAX spec-driven builders where possible.",
        DeprecationWarning,
        stacklevel=2,
    )
    from .reference_legacy import unrestricted_reference_from_pyscf_with_jax_uks as _impl

    return _impl(
        mf,
        max_l=max_l,
        uks_config=uks_config,
        xc_spec=xc_spec,
        energy_target=energy_target,
        grid_ao_backend=grid_ao_backend,
        compute_local_hfx_features=compute_local_hfx_features,
        compute_local_hfx_aux=compute_local_hfx_aux,
        hfx_omega_values=hfx_omega_values,
        hfx_chunk_size=hfx_chunk_size,
    )

__all__ = [
    "GridReference",
    "RestrictedMoleculeReference",
    "UnrestrictedMoleculeReference",
    "restricted_reference_from_spec_with_jax_rks",
    "unrestricted_reference_from_spec_with_jax_uks",
]
