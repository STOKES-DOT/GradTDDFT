from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array


def _scaled_sigmoid_output(preactivation: Array, scale: float = 2.0) -> Array:
    value = jnp.asarray(preactivation)
    if float(scale) <= 0.0:
        return value
    return jnp.asarray(scale, dtype=value.dtype) * jax.nn.sigmoid(value)


class DistanceGatedAttention(nn.Module):
    """Multi-head self-attention with a learnable pair-distance gate."""

    num_heads: int = 4
    qkv_features: int | None = None
    lambda_init: float = 5.0
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self,
        node_features: Array,
        atom_coords: Array,
        *,
        deterministic: bool = True,
    ) -> Array:
        x = jnp.asarray(node_features)
        coords = jnp.asarray(atom_coords)
        if x.ndim != 3:
            raise ValueError(
                f"node_features must have shape (batch, natoms, features), got {x.shape}."
            )
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError(
                f"atom_coords must have shape (batch, natoms, 3), got {coords.shape}."
            )
        if coords.shape[:2] != x.shape[:2]:
            raise ValueError("atom_coords must match node_features batch and atom dimensions.")

        d_model = int(x.shape[-1])
        qkv_features = int(self.qkv_features or d_model)
        if qkv_features % int(self.num_heads) != 0:
            raise ValueError("qkv_features must be divisible by num_heads.")
        d_head = qkv_features // int(self.num_heads)

        q = nn.Dense(qkv_features, name="q_proj")(x)
        k = nn.Dense(qkv_features, name="k_proj")(x)
        v = nn.Dense(qkv_features, name="v_proj")(x)
        q = q.reshape(q.shape[0], q.shape[1], int(self.num_heads), d_head)
        k = k.reshape(k.shape[0], k.shape[1], int(self.num_heads), d_head)
        v = v.reshape(v.shape[0], v.shape[1], int(self.num_heads), d_head)

        attn_logits = jnp.einsum("bihd,bjhd->bhij", q, k) / jnp.sqrt(float(d_head))
        r_diff = coords[:, :, None, :] - coords[:, None, :, :]
        distances = jnp.sqrt(jnp.sum(r_diff * r_diff, axis=-1) + 1e-8)

        log_lambda = self.param(
            "log_lambda",
            nn.initializers.constant(jnp.log(float(self.lambda_init))),
            (),
        )
        decay_length = jnp.maximum(jnp.exp(log_lambda), 1e-6)
        attn_logits = attn_logits - distances[:, None, :, :] / decay_length
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        if self.dropout_rate > 0.0:
            attn_weights = nn.Dropout(self.dropout_rate)(
                attn_weights,
                deterministic=deterministic,
            )

        out = jnp.einsum("bhij,bjhd->bihd", attn_weights, v)
        out = out.reshape(out.shape[0], out.shape[1], qkv_features)
        return nn.Dense(d_model, name="out_proj")(out)


class AttentionReadout(nn.Module):
    """Attention-weighted permutation-invariant pooling over atoms."""

    d_model: int

    @nn.compact
    def __call__(self, node_features: Array) -> Array:
        x = jnp.asarray(node_features)
        if x.ndim != 3:
            raise ValueError(
                f"node_features must have shape (batch, natoms, features), got {x.shape}."
            )
        if int(x.shape[-1]) != int(self.d_model):
            raise ValueError(
                f"node_features last dimension must equal d_model={self.d_model}, got {x.shape[-1]}."
            )

        query = self.param(
            "readout_query",
            nn.initializers.normal(stddev=0.02),
            (int(self.d_model),),
        )
        key = nn.Dense(int(self.d_model), name="readout_key")(x)
        logits = jnp.einsum("d,bnd->bn", query, key) / jnp.sqrt(float(self.d_model))
        weights = jax.nn.softmax(logits, axis=-1)
        return jnp.einsum("bn,bnd->bd", weights, x)


class RSHGNNHead(nn.Module):
    """Predict raw RSH parameters from atom-centered density descriptors."""

    node_hidden_dims: Sequence[int] = (32, 32)
    global_hidden_dims: Sequence[int] = (32, 16)
    num_heads: int = 4
    num_layers: int | None = 1
    num_interaction_blocks: int | None = None
    qkv_features: int | None = None
    ffn_dim: int | None = None
    ffn_expansion: int = 4
    lambda_init: float = 5.0
    dropout_rate: float = 0.0
    activation: Callable[[Array], Array] = nn.gelu
    sigmoid_scale_factor: float = 2.0

    def _block_count(self) -> int:
        if self.num_layers is not None and self.num_interaction_blocks is not None:
            if int(self.num_layers) != int(self.num_interaction_blocks):
                raise ValueError("num_layers and num_interaction_blocks disagree.")
        value = self.num_layers if self.num_layers is not None else self.num_interaction_blocks
        return int(1 if value is None else value)

    @nn.compact
    def __call__(
        self,
        atom_descriptors: Array,
        atom_coords: Array,
        *,
        deterministic: bool = True,
    ) -> Array:
        x = jnp.asarray(atom_descriptors)
        coords = jnp.asarray(atom_coords)
        if x.ndim != 3:
            raise ValueError(
                f"atom_descriptors must have shape (batch, natoms, features), got {x.shape}."
            )
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError(
                f"atom_coords must have shape (batch, natoms, 3), got {coords.shape}."
            )
        if coords.shape[:2] != x.shape[:2]:
            raise ValueError("atom_coords must match atom_descriptors batch and atom dimensions.")
        if not self.node_hidden_dims:
            raise ValueError("node_hidden_dims must contain at least one hidden width.")

        for index, width in enumerate(self.node_hidden_dims):
            x = nn.Dense(int(width), name=f"node_encoder_{index}")(x)
            x = self.activation(x)

        d_model = int(x.shape[-1])
        ffn_width = int(self.ffn_dim or d_model * int(self.ffn_expansion))
        for layer_idx in range(self._block_count()):
            residual = x
            x = DistanceGatedAttention(
                num_heads=int(self.num_heads),
                qkv_features=self.qkv_features,
                lambda_init=float(self.lambda_init),
                dropout_rate=float(self.dropout_rate),
                name=f"attn_{layer_idx}",
            )(x, coords, deterministic=deterministic)
            x = nn.LayerNorm(name=f"ln1_{layer_idx}")(x + residual)

            residual = x
            x = nn.Dense(ffn_width, name=f"ffn1_{layer_idx}")(x)
            x = self.activation(x)
            x = nn.Dense(d_model, name=f"ffn2_{layer_idx}")(x)
            x = nn.LayerNorm(name=f"ln2_{layer_idx}")(x + residual)

        y = AttentionReadout(d_model=d_model, name="readout")(x)
        for index, width in enumerate(self.global_hidden_dims):
            y = nn.Dense(int(width), name=f"global_mlp_{index}")(y)
            y = self.activation(y)
        y = nn.Dense(3, name="output")(y)
        return _scaled_sigmoid_output(y, self.sigmoid_scale_factor)


__all__ = [
    "AttentionReadout",
    "DistanceGatedAttention",
    "RSHGNNHead",
]
