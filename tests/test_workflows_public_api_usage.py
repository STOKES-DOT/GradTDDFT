import ast
from pathlib import Path


def test_top_level_exposes_spec_workflow_entrypoint():
    import td_graddft
    from td_graddft.workflows.core import run_pipeline_core_from_spec

    assert "run_pipeline_core_from_spec" in td_graddft.__all__
    assert td_graddft.run_pipeline_core_from_spec is run_pipeline_core_from_spec


def test_workflow_reporting_defers_pyplot_import_until_plotting():
    source = Path("src/td_graddft/workflows/reporting.py").read_text()
    tree = ast.parse(source)
    top_level_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    assert all(
        not any(alias.name == "matplotlib.pyplot" for alias in getattr(node, "names", ()))
        and getattr(node, "module", None) != "matplotlib.pyplot"
        for node in top_level_imports
    )


def test_workflow_core_uses_public_facades_for_xc_and_tdscf():
    text = Path("src/td_graddft/workflows/core.py").read_text()

    assert "make_neural_xc_functional" not in text
    assert "RestrictedCasidaTDDFT" not in text
    assert "UnrestrictedCasidaTDDFT" not in text
    assert "SemilocalResponseFunctional" not in text
    assert "neural_xc.Functional(" in text
    assert "tdscf.TDDFT(" in text
    assert "tdscf.TDA(" in text
