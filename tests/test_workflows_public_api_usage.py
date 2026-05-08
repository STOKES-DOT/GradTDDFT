from pathlib import Path


def test_workflow_core_uses_public_facades_for_xc_and_tdscf():
    text = Path("src/td_graddft/workflows/core.py").read_text()

    assert "make_neural_xc_functional" not in text
    assert "RestrictedCasidaTDDFT" not in text
    assert "UnrestrictedCasidaTDDFT" not in text
    assert "SemilocalResponseFunctional" not in text
    assert "neural_xc.Functional(" in text
    assert "tdscf.TDDFT(" in text
    assert "tdscf.TDA(" in text
