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


def _pair_metadata(nao: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.tril_indices(int(nao))
    pair_index = np.empty((int(nao), int(nao)), dtype=np.int32)
    pair_ids = np.arange(rows.size, dtype=np.int32)
    pair_index[rows, cols] = pair_ids
    pair_index[cols, rows] = pair_ids
    multiplicity = np.where(rows == cols, 1.0, 2.0).astype(np.float64)
    return rows, cols, pair_index, multiplicity


def _gpu4pyscf_raw_int2e_to_s4(raw_eri: Any) -> np.ndarray:
    raw = np.asarray(raw_eri, dtype=float)
    if raw.ndim != 4 or len(set(raw.shape)) != 1:
        raise ValueError(
            "GPU4PySCF raw int2e tensor must have shape (nao, nao, nao, nao)."
        )
    nao = int(raw.shape[0])
    rows, cols, _, multiplicity = _pair_metadata(nao)
    p = rows[:, None]
    q = cols[:, None]
    r = rows[None, :]
    s = cols[None, :]
    candidates = np.stack(
        [
            raw[p, q, r, s],
            raw[q, p, r, s],
            raw[p, q, s, r],
            raw[q, p, s, r],
            raw[r, s, p, q],
            raw[s, r, p, q],
            raw[r, s, q, p],
            raw[s, r, q, p],
        ],
        axis=0,
    )
    selector = np.argmax(np.abs(candidates), axis=0)
    packed = np.take_along_axis(candidates, selector[None, ...], axis=0)[0]
    scale = 4.0 / (multiplicity[:, None] * multiplicity[None, :])
    return np.asarray(packed * scale, dtype=float)


def _s4_to_full(s4: Any) -> np.ndarray:
    packed = np.asarray(s4, dtype=float)
    if packed.ndim != 2 or packed.shape[0] != packed.shape[1]:
        raise ValueError("Packed ERI matrix must be square.")
    pair_count = int(packed.shape[0])
    nao = int((np.sqrt(8 * pair_count + 1) - 1) / 2)
    if nao * (nao + 1) // 2 != pair_count:
        raise ValueError("Packed ERI matrix dimension is not a triangular AO-pair count.")
    _, _, pair_index, _ = _pair_metadata(nao)
    ao = np.arange(nao)
    return np.asarray(
        packed[
            pair_index[ao[:, None, None, None], ao[None, :, None, None]],
            pair_index[ao[None, None, :, None], ao[None, None, None, :]],
        ],
        dtype=float,
    )


def _gpu4pyscf_raw_int2e_to_full(raw_eri: Any) -> np.ndarray:
    return _s4_to_full(_gpu4pyscf_raw_int2e_to_s4(raw_eri))


def _gpu4pyscf_raw_int2e(mol: Any) -> np.ndarray:
    try:
        from gpu4pyscf.scf.int4c2e import get_int4c2e
    except (ImportError, OSError) as exc:
        raise ImportError(
            "GPU4PySCF is required when integral_backend='gpu'."
        ) from exc
    return np.asarray(_asnumpy(get_int4c2e(mol)), dtype=float)


def gpu4pyscf_int2e_full(mol: Any) -> np.ndarray:
    return _gpu4pyscf_raw_int2e_to_full(_gpu4pyscf_raw_int2e(mol))


def gpu4pyscf_int2e_s4(mol: Any) -> np.ndarray:
    return _gpu4pyscf_raw_int2e_to_s4(_gpu4pyscf_raw_int2e(mol))


__all__ = ["gpu4pyscf_int2e_full", "gpu4pyscf_int2e_s4"]
