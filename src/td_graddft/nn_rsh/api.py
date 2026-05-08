from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from .functional import TrainableRSHFunctional, make_minimal_trainable_rsh_functional
from .presets import get_rsh_functional_preset, make_rsh_template

_SUPPORTED_TRAINABLE_PARAMS = frozenset({"omega", "alpha", "beta"})


@dataclass(frozen=True)
class RSH:
    name: str
    omega_source: Literal["canonical", "optxc"] = "canonical"

    def trainable(
        self,
        *,
        params: Sequence[str] = ("omega", "alpha", "beta"),
        local_xc_spec: str | None = None,
        hidden_dims: Sequence[int] = (),
    ) -> TrainableRSHFunctional:
        unknown = tuple(str(name) for name in params if str(name) not in _SUPPORTED_TRAINABLE_PARAMS)
        if unknown:
            raise ValueError(
                "Unsupported RSH trainable parameter(s): "
                + ", ".join(repr(name) for name in unknown)
                + ". Expected only 'omega', 'alpha', or 'beta'."
            )
        template = make_rsh_template(self.name, omega_source=self.omega_source)
        preset = get_rsh_functional_preset(self.name)
        resolved_local_xc_spec = local_xc_spec or preset.jax_local_xc_spec or "pbe"
        return make_minimal_trainable_rsh_functional(
            local_xc_spec=resolved_local_xc_spec,
            hidden_dims=hidden_dims,
            template=template,
            name=template.name,
        )


__all__ = ["RSH"]
