from td_graddft import nn_rsh


def test_nn_rsh_namespace_exports():
    functional = nn_rsh.RSH("lc-wpbe").trainable()

    assert isinstance(functional, nn_rsh.TrainableRSHFunctional)
    assert isinstance(functional.template, nn_rsh.RSHFunctionalTemplate)
    assert functional.head_type == "gnn"
