from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from flax import serialization


def save_params_checkpoint(
    path: str | Path,
    params: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path | None]:
    """Save model params as a Flax msgpack checkpoint.

    Returns the checkpoint path and optional metadata JSON path.
    """

    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = serialization.msgpack_serialize(serialization.to_state_dict(params))
    ckpt_path.write_bytes(payload)

    meta_path: Path | None = None
    if metadata is not None:
        meta_path = ckpt_path.with_suffix(ckpt_path.suffix + ".meta.json")
        meta_path.write_text(
            json.dumps(dict(metadata), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return ckpt_path, meta_path


def load_params_checkpoint(
    path: str | Path,
    *,
    template: Any | None = None,
) -> Any:
    """Load model params from a Flax msgpack checkpoint.

    If ``template`` is provided, the loaded state is projected onto the template
    tree structure via ``flax.serialization.from_state_dict``.
    """

    ckpt_path = Path(path)
    payload = ckpt_path.read_bytes()
    state = serialization.msgpack_restore(payload)
    if template is None:
        return state
    return serialization.from_state_dict(template, state)
