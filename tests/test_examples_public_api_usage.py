from pathlib import Path


EXAMPLES_USING_PUBLIC_TDSCF = (
    Path("examples/compare_pyscf_vs_jax_tddft_no_neural.py"),
)

EXAMPLES_USING_NEURAL_XC_FACADE = (
    Path("examples/h2_fci_self_consistent_train.py"),
)


def test_main_examples_use_tdscf_facade_for_restricted_response():
    for path in EXAMPLES_USING_PUBLIC_TDSCF:
        text = path.read_text()
        assert "from td_graddft.tddft import RestrictedCasidaTDDFT" not in text
        assert "RestrictedCasidaTDDFT(" not in text
        assert "tdscf." in text


def test_main_examples_use_neural_xc_facade_constructor():
    for path in EXAMPLES_USING_NEURAL_XC_FACADE:
        text = path.read_text()
        assert "make_neural_xc_functional" not in text
        assert "neural_xc.Functional(" in text
