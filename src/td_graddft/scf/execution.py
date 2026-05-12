from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from .rks import RKSConfig

FacadeExecutionPreference = Literal["auto", "cpu", "gpu"]
FacadeRKSExecutionMode = Literal["custom", "cpu", "gpu_cuda_direct"]
WorkflowRKSExecutionMode = Literal["custom", "cpu", "gpu_cuda_direct"]
RKSExecutionMode = Literal["cpu", "gpu_cuda_direct"]


@dataclass(frozen=True)
class RKSExecutionProfile:
    execution_mode: Literal["cpu", "gpu_cuda_direct"]
    jk_backend: Literal["full", "direct"]
    direct_jk_engine: Literal["jax", "cuda"]
    integral_backend: Literal["libcint"] = "libcint"
    grid_ao_backend: Literal["jax"] = "jax"
    iteration_backend: Literal["runtime", "lax"] = "runtime"


@dataclass(frozen=True)
class ResolvedRKSExecutionPlan:
    mode: RKSExecutionMode
    config: RKSConfig
    integral_backend: Literal["jax", "libcint"]
    grid_ao_backend: Literal["jax"]
    include_dipole_integrals: bool
    use_cuda_direct_reference_solver: bool


def _default_cuda_direct_iteration_backend(*, precompile: bool) -> Literal["runtime"]:
    del precompile
    return "runtime"


def cpu_rks_execution_profile() -> RKSExecutionProfile:
    return RKSExecutionProfile(
        execution_mode="cpu",
        jk_backend="full",
        direct_jk_engine="jax",
    )


def gpu_cuda_direct_rks_execution_profile(
    *,
    precompile: bool,
    iteration_backend: Literal["runtime", "lax"] | None = None,
) -> RKSExecutionProfile:
    if iteration_backend is not None and iteration_backend not in {"runtime", "lax"}:
        raise ValueError("iteration_backend must be 'runtime', 'lax', or None.")
    return RKSExecutionProfile(
        execution_mode="gpu_cuda_direct",
        jk_backend="direct",
        direct_jk_engine="cuda",
        iteration_backend=iteration_backend or _default_cuda_direct_iteration_backend(
            precompile=precompile,
        ),
    )


def apply_rks_execution_profile(
    config: RKSConfig,
    profile: RKSExecutionProfile,
) -> RKSConfig:
    return replace(
        config,
        jk_backend=profile.jk_backend,
        direct_jk_engine=profile.direct_jk_engine,
        iteration_backend=profile.iteration_backend,
    )


def classify_rks_execution_mode(
    config: RKSConfig,
    *,
    geometry_is_traced: bool,
    cuda_available: bool,
) -> RKSExecutionMode:
    if geometry_is_traced:
        raise NotImplementedError(
            "Explicit traceable SCF execution has been removed. "
            "Use implicit differential SCF instead."
        )
    if config.jk_backend == "direct" and config.direct_jk_engine == "cuda" and cuda_available:
        return "gpu_cuda_direct"
    return "cpu"


def resolve_rks_execution_plan(
    config: RKSConfig,
    *,
    integral_backend: Literal["jax", "libcint"],
    grid_ao_backend: Literal["jax"],
    include_dipole_integrals: bool,
    geometry_is_traced: bool,
    cuda_available: bool,
) -> ResolvedRKSExecutionPlan:
    mode = classify_rks_execution_mode(
        config,
        geometry_is_traced=geometry_is_traced,
        cuda_available=cuda_available,
    )
    if mode == "gpu_cuda_direct":
        return ResolvedRKSExecutionPlan(
            mode=mode,
            config=config,
            integral_backend=integral_backend,
            grid_ao_backend=grid_ao_backend,
            include_dipole_integrals=include_dipole_integrals,
            use_cuda_direct_reference_solver=True,
        )
    return ResolvedRKSExecutionPlan(
        mode=mode,
        config=config,
        integral_backend=integral_backend,
        grid_ao_backend=grid_ao_backend,
        include_dipole_integrals=include_dipole_integrals,
        use_cuda_direct_reference_solver=False,
    )


def apply_workflow_rks_execution_mode(
    config: RKSConfig,
    *,
    execution_mode: WorkflowRKSExecutionMode,
    integral_backend: Literal["jax", "libcint"],
    grid_ao_backend: Literal["jax"],
    cuda_available: bool,
) -> tuple[RKSConfig, Literal["jax", "libcint"], Literal["jax"]]:
    mode = str(execution_mode).lower()
    if mode == "custom":
        return config, integral_backend, grid_ao_backend
    if mode == "cpu":
        profile = cpu_rks_execution_profile()
        return (
            apply_rks_execution_profile(config, profile),
            profile.integral_backend,
            profile.grid_ao_backend,
        )
    if mode == "gpu_cuda_direct":
        if not cuda_available:
            raise RuntimeError(
                "jax_rks_execution_mode='gpu_cuda_direct' requested but CUDA FFI is unavailable."
            )
        profile = gpu_cuda_direct_rks_execution_profile(precompile=False)
        return (
            apply_rks_execution_profile(config, profile),
            profile.integral_backend,
            profile.grid_ao_backend,
        )
    raise ValueError(
        "execution_mode must be one of {'custom', 'cpu', 'gpu_cuda_direct'}."
    )
