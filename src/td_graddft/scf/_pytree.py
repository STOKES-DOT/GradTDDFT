from __future__ import annotations

from dataclasses import fields
from typing import Any

import jax


def pytree_dataclass(cls: type[Any] | None = None, *, static_fields: tuple[str, ...] = ()):
    """Register a dataclass as a JAX pytree, optionally keeping fields static."""

    static_field_names = frozenset(static_fields)

    def decorator(cls_: type[Any]):
        all_field_names = tuple(field.name for field in fields(cls_))
        dynamic_field_names = tuple(
            field_name for field_name in all_field_names if field_name not in static_field_names
        )
        static_field_names_ordered = tuple(
            field_name for field_name in all_field_names if field_name in static_field_names
        )

        def tree_flatten(self):
            children = tuple(getattr(self, field_name) for field_name in dynamic_field_names)
            static_values = tuple(
                getattr(self, field_name) for field_name in static_field_names_ordered
            )
            return children, static_values

        @classmethod
        def tree_unflatten(cls__, aux_data, children):
            kwargs = {
                name: value for name, value in zip(dynamic_field_names, children, strict=True)
            }
            kwargs.update(
                {
                    name: value
                    for name, value in zip(static_field_names_ordered, aux_data, strict=True)
                }
            )
            return cls__(**kwargs)

        cls_.tree_flatten = tree_flatten
        cls_.tree_unflatten = tree_unflatten
        return jax.tree_util.register_pytree_node_class(cls_)

    if cls is not None:
        return decorator(cls)
    return decorator
