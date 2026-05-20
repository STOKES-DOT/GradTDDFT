from dataclasses import dataclass

import jax

from td_graddft.scf._pytree import pytree_dataclass


def test_pytree_dataclass_keeps_static_fields_out_of_children():
    @pytree_dataclass(static_fields=("label",))
    @dataclass(frozen=True)
    class Example:
        value: float
        label: str

    item = Example(1.5, "static")
    children, treedef = jax.tree_util.tree_flatten(item)
    restored = jax.tree_util.tree_unflatten(treedef, children)

    assert children == [1.5]
    assert restored == item
