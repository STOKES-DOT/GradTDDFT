from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from .descriptors import AtomCenteredDensityDescriptorConfig
from .functional import TrainableRSHFunctional, make_gnn_rsh_functional
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
        descriptor_config: AtomCenteredDensityDescriptorConfig | None = None,
        node_hidden_dims: Sequence[int] = (32, 32),
        global_hidden_dims: Sequence[int] = (32, 16),
        num_heads: int = 4,
        num_layers: int = 1,
        qkv_features: int | None = None,
        ffn_dim: int | None = None,
        ffn_expansion: int = 4,
        lambda_init: float = 5.0,
        dropout_rate: float = 0.0,
        fallback_omega_values: Sequence[float] | None = None,
        hidden_dims: Sequence[int] | None = None,
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
        resolved_local_xc_spec = local_xc_spec or preset.jax_local_xc_spec
        resolved_local_term_specs = tuple(preset.local_term_specs)
        if resolved_local_xc_spec is None and not resolved_local_term_specs:
            raise NotImplementedError(
                f"RSH preset {self.name!r} does not yet define a JAX-local semilocal decomposition "
                "for the trainable workflow."
            )
        resolved_global_hidden_dims = (
            tuple(int(width) for width in hidden_dims)
            if hidden_dims is not None
            else tuple(int(width) for width in global_hidden_dims)
        )
        resolved_fallback_omegas = (
            None
            if fallback_omega_values is None
            else tuple(float(value) for value in fallback_omega_values)
        )
        return make_gnn_rsh_functional(
            local_xc_spec=resolved_local_xc_spec,
            local_term_specs=resolved_local_term_specs,
            descriptor_config=descriptor_config,
            node_hidden_dims=tuple(int(width) for width in node_hidden_dims),
            global_hidden_dims=resolved_global_hidden_dims,
            num_heads=int(num_heads),
            num_layers=int(num_layers),
            qkv_features=qkv_features,
            ffn_dim=ffn_dim,
            ffn_expansion=int(ffn_expansion),
            lambda_init=float(lambda_init),
            dropout_rate=float(dropout_rate),
            template=template,
            fallback_omega_values=resolved_fallback_omegas,
            name=template.name,
        )


__all__ = ["RSH"]
