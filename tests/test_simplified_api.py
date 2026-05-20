from td_graddft import (
    MoleculeConfig,
    build_molecule,
    run_pipeline,
    run_spectrum_pipeline,
)
from td_graddft.api import (
    MoleculeConfig as ApiMoleculeConfig,
    build_molecule as api_build_molecule,
    run_pipeline as api_run_pipeline,
    run_spectrum_pipeline as api_run_spectrum_pipeline,
)


def test_simplified_api_exports_align():
    assert MoleculeConfig is ApiMoleculeConfig
    assert build_molecule is api_build_molecule
    assert run_pipeline is api_run_pipeline
    assert run_spectrum_pipeline is api_run_spectrum_pipeline
