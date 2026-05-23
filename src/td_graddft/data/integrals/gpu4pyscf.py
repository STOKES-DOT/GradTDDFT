from __future__ import annotations

from typing import Any

import numpy as np


def _asnumpy(value: Any) -> np.ndarray:
    get = getattr(value, "get", None)
    if callable(get):
        return np.asarray(get())
    try:
        import cupy
    except Exception:
        return np.asarray(value)
    return np.asarray(cupy.asnumpy(value))


def gpu4pyscf_int2e_full(mol: Any) -> np.ndarray:
    try:
        from gpu4pyscf.scf.int4c2e import get_int4c2e
    except (ImportError, OSError) as exc:
        raise ImportError(
            "GPU4PySCF is required when integral_backend='gpu'."
        ) from exc
    return np.asarray(_asnumpy(get_int4c2e(mol)), dtype=float)


def gpu4pyscf_int2e_s4(mol: Any) -> np.ndarray:
    eri = gpu4pyscf_int2e_full(mol)
    pair_i, pair_j = np.tril_indices(int(eri.shape[0]))
    return np.asarray(eri[pair_i, pair_j][:, pair_i, pair_j], dtype=float)


__all__ = ["gpu4pyscf_int2e_full", "gpu4pyscf_int2e_s4"]
