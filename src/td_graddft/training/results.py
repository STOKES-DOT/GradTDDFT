from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrainingResult:
    functional: Any
    params: Any = None
    history: dict[str, list[Any]] = field(default_factory=dict)
    final_metrics: dict[str, Any] = field(default_factory=dict)


__all__ = ["TrainingResult"]
