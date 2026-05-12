from __future__ import annotations

from typing import Any, Callable, Sequence

from jaxtyping import Array, PyTree

from .config import GroundStateDatum, GroundStateTrainingConfig
from .trainer import make_ground_state_eval


def make_ground_state_loss(
    functional: Any,
    *,
    training_config: GroundStateTrainingConfig | None = None,
    predictor: Callable[[PyTree, Any], tuple[Array, Any]] | None = None,
) -> Callable[[PyTree, GroundStateDatum | Sequence[GroundStateDatum]], tuple[Array, dict[str, Array]]]:
    """Bind a ground-state objective to a functional and optional predictor.

    This mirrors GradDFT's explicit predictor/loss split: callers can preselect
    a fixed-density or self-consistent predictor, then reuse the resulting loss
    callable across evaluation and training utilities.
    """

    cfg = GroundStateTrainingConfig() if training_config is None else training_config
    return make_ground_state_eval(
        functional,
        training_config=cfg,
        predictor=predictor,
    )
