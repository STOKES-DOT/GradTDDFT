from td_graddft import nn_rsh


def test_nn_rsh_namespace_exports():
    functional = nn_rsh.make_minimal_trainable_rsh_functional()

    assert isinstance(functional, nn_rsh.TrainableRSHFunctional)
    assert isinstance(functional.template, nn_rsh.RSHFunctionalTemplate)
    assert callable(nn_rsh.make_self_supervised_rsh_loss)
