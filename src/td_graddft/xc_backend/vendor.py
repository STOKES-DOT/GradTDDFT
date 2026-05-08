from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


_REQUIRED_FILES = (
    "LICENSE",
    "README.rst",
    "gen_repo/__init__.py",
    "gen_repo/wheel.BUILD",
)
_METADATA_FILE = "TD_GRADDFT_VENDOR.json"


@dataclass(frozen=True)
class VendoredJAXXCInfo:
    """Metadata for TD-GradDFT's vendored jax_xc source tree."""

    root: Path
    complete: bool
    backend_label: str
    commit: str | None = None
    version: str | None = None
    reason: str | None = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _metadata(root: Path) -> dict[str, Any]:
    path = root / _METADATA_FILE
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def vendored_jax_xc_info(root: Path | None = None) -> VendoredJAXXCInfo:
    """Return stable diagnostics for the vendored jax_xc checkout."""

    vendor_root = root if root is not None else _project_root() / "third_party" / "jax_xc"
    missing = [name for name in _REQUIRED_FILES if not (vendor_root / name).exists()]
    data = _metadata(vendor_root)
    complete = not missing
    reason = None
    if not vendor_root.exists():
        reason = "third_party/jax_xc is missing"
    elif missing:
        reason = "missing required files: " + ", ".join(missing)
    else:
        reason = data.get("reason")
    return VendoredJAXXCInfo(
        root=vendor_root,
        complete=complete,
        backend_label="vendored" if complete else "missing",
        commit=data.get("commit"),
        version=data.get("version"),
        reason=reason,
    )
