import importlib


def test_td_graddft_dir_lists_recommended_namespaces():
    import td_graddft

    for name in ("gto", "scf", "dft", "tdscf", "neural_xc", "nn_rsh", "training"):
        assert name in td_graddft.__all__
        assert name in dir(td_graddft)
        assert getattr(td_graddft, name) is importlib.import_module(f"td_graddft.{name}")


def test_td_graddft_exposes_pyscf_style_namespaces():
    for name in (
        "td_graddft.gto",
        "td_graddft.dft",
        "td_graddft.scf",
        "td_graddft.tdscf",
        "td_graddft.reference",
    ):
        module = importlib.import_module(name)
        assert module is not None


def test_dft_namespace_exposes_ks_facades():
    from td_graddft import dft, scf

    assert dft.RKS is scf.RKS
    assert dft.UKS is scf.UKS


def test_neural_xc_namespace_exposes_long_range_correction_facade():
    from td_graddft import neural_xc

    assert callable(neural_xc.LongRangeCorrection)
    assert callable(neural_xc.make_long_range_correction)


def test_top_level_exposes_recommended_neural_xc_facades():
    import td_graddft

    assert td_graddft.Functional is td_graddft.neural_xc.Functional
    assert td_graddft.LongRangeCorrection is td_graddft.neural_xc.LongRangeCorrection


def test_top_level_removes_legacy_neural_xc_exports():
    import td_graddft

    removed = (
        "Density" "NeuralXCFunctional",
        "Neural" "XCFunctional",
        "Pointwise" "MLP",
        "make_neural" "_lda_functional",
        "make_dm21" "_like_functional",
    )

    for name in removed:
        assert not hasattr(td_graddft, name), f"{name} should not be exported at top level"


def test_td_graddft_pyscf_style_submodules_import():
    for name in (
        "td_graddft.gto.basis",
        "td_graddft.gto.grid",
        "td_graddft.dft.rks",
        "td_graddft.dft.uks",
        "td_graddft.dft.xc",
    ):
        module = importlib.import_module(name)
        assert module is not None
