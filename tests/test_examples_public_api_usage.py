from pathlib import Path


EXAMPLES_USING_PUBLIC_TDSCF = (
    Path("examples/compare_h2_fci_vs_neural_spectrum.py"),
    Path("examples/evaluate_h2_three_loss_checkpoints.py"),
    Path("examples/h2_fci_train_and_excited_compare.py"),
    Path("examples/h2_three_loss_dissociation_compare.py"),
    Path("examples/qh9_short_benchmark.py"),
    Path("examples/water_overfit_with_orbitals.py"),
)

EXAMPLES_USING_NEURAL_XC_FACADE = (
    Path("examples/compare_h2_fci_vs_neural_spectrum.py"),
    Path("examples/evaluate_h2_three_loss_checkpoints.py"),
    Path("examples/h2_fci_ground_curve.py"),
    Path("examples/h2_fci_self_consistent_train.py"),
    Path("examples/h2_fci_train_and_excited_compare.py"),
    Path("examples/h2_three_loss_dissociation_compare.py"),
    Path("examples/qh9_short_benchmark.py"),
    Path("examples/water_overfit_with_orbitals.py"),
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
