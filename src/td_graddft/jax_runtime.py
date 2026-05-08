from __future__ import annotations

import importlib
from pathlib import Path

import jax


DEFAULT_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS = 1.0
DEFAULT_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES = 64 * 1024


def configure_jax_persistent_cache(
    *,
    cache_dir: str | None,
    min_compile_time_secs: float = DEFAULT_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS,
    min_entry_size_bytes: int = DEFAULT_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES,
    xla_caches: str | None = "xla_gpu_per_fusion_autotune_cache_dir",
) -> str | None:
    """Enable JAX persistent compilation cache and return resolved cache path.

    This is a best-effort helper. If ``cache_dir`` is empty, no changes are made.
    """

    if cache_dir is None:
        return None
    raw = str(cache_dir).strip()
    if not raw:
        return None

    path = Path(raw).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    resolved = str(path)
    jax.config.update("jax_compilation_cache_dir", resolved)
    try:
        jax.config.update("jax_enable_compilation_cache", True)
    except AttributeError:
        pass
    jax.config.update(
        "jax_persistent_cache_min_compile_time_secs",
        float(min_compile_time_secs),
    )
    jax.config.update(
        "jax_persistent_cache_min_entry_size_bytes",
        int(min_entry_size_bytes),
    )
    if xla_caches is not None:
        try:
            jax.config.update("jax_persistent_cache_enable_xla_caches", str(xla_caches))
        except AttributeError:
            pass
    try:
        compilation_cache = importlib.import_module(
            "jax.experimental.compilation_cache.compilation_cache"
        )
    except (ImportError, ModuleNotFoundError):
        compilation_cache = None
    if compilation_cache is not None and hasattr(compilation_cache, "set_cache_dir"):
        compilation_cache.set_cache_dir(resolved)
    return resolved
