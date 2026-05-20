import tomllib
from pathlib import Path


def test_pyscf_basis_snapshot_is_included_as_package_data():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    package_data = pyproject["tool"]["setuptools"]["package-data"]
    data_patterns = package_data["td_graddft.data"]

    assert "pyscf_basis_snapshot/**/*" in data_patterns


def test_integral_cuda_sources_are_not_included_as_package_data():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    package_data = pyproject["tool"]["setuptools"]["package-data"]

    assert "td_graddft.data.integrals.jax" not in package_data
