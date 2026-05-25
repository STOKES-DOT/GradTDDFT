from __future__ import annotations

import numpy as np

from td_graddft.data.integrals.gpu4pyscf import (
    _gpu4pyscf_raw_int2e_to_full,
    _gpu4pyscf_raw_int2e_to_s4,
)


def _pair_index(nao: int) -> np.ndarray:
    rows, cols = np.tril_indices(int(nao))
    index = np.empty((int(nao), int(nao)), dtype=np.int32)
    pair_ids = np.arange(rows.size, dtype=np.int32)
    index[rows, cols] = pair_ids
    index[cols, rows] = pair_ids
    return index


def test_gpu4pyscf_raw_int2e_is_restored_to_pyscf_s4_layout():
    nao = 3
    rows, cols = np.tril_indices(nao)
    pair_count = rows.size
    values = np.arange(1, pair_count * pair_count + 1, dtype=np.float64).reshape(
        pair_count, pair_count
    )
    expected_s4 = 0.5 * (values + values.T)
    raw = np.zeros((nao, nao, nao, nao), dtype=np.float64)
    multiplicity = np.where(rows == cols, 1.0, 2.0)
    placements = (
        lambda p, q, r, s: (p, q, r, s),
        lambda p, q, r, s: (q, p, r, s),
        lambda p, q, r, s: (p, q, s, r),
        lambda p, q, r, s: (r, s, p, q),
    )

    for a, (p, q) in enumerate(zip(rows, cols)):
        for b, (r, s) in enumerate(zip(rows, cols)):
            scaled = expected_s4[a, b] * multiplicity[a] * multiplicity[b] / 4.0
            raw[placements[(a + b) % len(placements)](p, q, r, s)] = scaled

    restored_s4 = _gpu4pyscf_raw_int2e_to_s4(raw)
    restored_full = _gpu4pyscf_raw_int2e_to_full(raw)
    pair = _pair_index(nao)
    ao = np.arange(nao)
    expected_full = expected_s4[
        pair[ao[:, None, None, None], ao[None, :, None, None]],
        pair[ao[None, None, :, None], ao[None, None, None, :]],
    ]

    assert np.allclose(restored_s4, expected_s4)
    assert np.allclose(restored_full, expected_full)
