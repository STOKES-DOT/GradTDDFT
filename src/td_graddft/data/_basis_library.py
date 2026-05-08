from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _load_bundle() -> dict[str, dict[str, list[list[object]]]]:
    path = Path(__file__).with_name("_pyscf_basis_bundle.json")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


BUILTIN_BASIS_LIBRARY = _load_bundle()


__all__ = ["BUILTIN_BASIS_LIBRARY"]
