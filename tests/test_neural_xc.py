from td_graddft import neural_xc


def test_functional_facade_builds_current_neural_xc_functional():
    functional = neural_xc.Functional(
        architecture="residual",
        semilocal_xc=("gga_x_pbe", "gga_c_pbe"),
        hidden_dims=(8,),
    )

    assert isinstance(functional, neural_xc.NeuralXCHybridFunctional)
    assert functional.name == "neural_xc"
    assert functional.include_pt2_channel is False


def test_functional_facade_accepts_hf_pt2_head_configuration():
    functional = neural_xc.Functional(
        architecture="residual",
        semilocal_xc=("gga_x_pbe", "gga_c_pbe"),
        hidden_dims=(8,),
        include_pt2_channel=True,
        pt2_channel_mode="local_exact",
    )

    assert functional.include_pt2_channel is True
    assert functional.pt2_channel_mode == "local_exact"
