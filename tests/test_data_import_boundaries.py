import jax


def test_grid_ao_lookup_tables_stay_host_side_at_import():
    from td_graddft.data import grid_ao

    assert not isinstance(grid_ao._DOUBLE_FACTORIAL_LOOKUP, jax.Array)
