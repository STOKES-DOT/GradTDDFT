from pathlib import Path
import re


def test_production_code_does_not_depend_on_removed_reference_adapters():
    root = Path("src/td_graddft")
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text()
        if (
            "reference_legacy" in text
            or "pyscf_adapter" in text
            or "td_graddft.reference" in text
            or "from .reference import" in text
            or "from ..reference import" in text
        ):
            offenders.append(str(path))

    assert offenders == []


def test_reference_legacy_and_pyscf_adapter_are_not_runtime_modules():
    import importlib
    import td_graddft

    assert "pyscf_adapter" not in td_graddft.__all__
    assert "reference_legacy" not in td_graddft.__all__
    for module_name in (
        "td_graddft.pyscf_adapter",
        "td_graddft.reference_legacy",
        "td_graddft.reference",
    ):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"{module_name} should not exist in the runtime package")


def test_runtime_public_api_does_not_expose_pyscf_bridge_symbols():
    import td_graddft
    from td_graddft import dft, nn_rsh, upstreams

    forbidden = {
        "PySCFRSHSpec",
        "make_pyscf_rsh_spec",
        "ground_state_from_pyscf_mean_field",
    }
    for module in (td_graddft, dft, nn_rsh, upstreams):
        names = set(getattr(module, "__all__", ())) | set(vars(module))
        assert forbidden.isdisjoint(names)


def test_pyscf_runtime_imports_are_limited_to_integral_modules():
    root = Path("src/td_graddft")
    allowed = {
        Path("src/td_graddft/df/jk.py"),
    }
    allowed_prefixes = (
        Path("src/td_graddft/data/integrals"),
        Path("src/td_graddft/data/pyscf_basis_snapshot"),
    )
    pattern = re.compile(r"^\s*(from\s+pyscf\b|import\s+pyscf\b)", re.MULTILINE)

    offenders = []
    for path in root.rglob("*.py"):
        if path in allowed or any(path.is_relative_to(prefix) for prefix in allowed_prefixes):
            continue
        if pattern.search(path.read_text()):
            offenders.append(str(path))

    assert offenders == []


def test_legacy_mean_field_tddft_calls_are_not_in_runtime_code():
    root = Path("src/td_graddft")
    forbidden = ("mf.TDDFT", "mf.TDA", "grad_dft.interface.pyscf")

    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text()
        if any(pattern in text for pattern in forbidden):
            offenders.append(str(path))

    assert offenders == []


def test_scf_features_do_not_expose_neural_training_only_hf_pt2_helpers():
    from td_graddft.scf import features as scf_features
    from td_graddft.neural_xc import inputs

    hidden = (
        "_local_hfx_features_from_basis_dm",
        "_local_hfx_features_from_dm",
        "_local_pt2_feature_from_restricted_orbitals",
    )
    for name in hidden:
        assert not hasattr(scf_features, name)
        assert hasattr(inputs, name)


def test_public_api_prefers_molecule_naming_over_reference_naming():
    import td_graddft
    from td_graddft import scf, workflows

    scf_preferred = {
        "QuadratureGrid",
        "RestrictedMolecule",
        "UnrestrictedMolecule",
        "restricted_molecule_from_spec_with_jax_rks",
        "unrestricted_molecule_from_spec_with_jax_uks",
    }
    public_preferred = {
        "MoleculeRun",
        "MoleculeSpecConfig",
        "build_molecule",
        "put_molecule_on_device",
        "put_restricted_molecule_on_device",
        "run_molecule_from_spec",
    }
    scf_legacy = {
        "GridReference",
        "RestrictedMoleculeReference",
        "UnrestrictedMoleculeReference",
        "restricted_reference_from_spec_with_jax_rks",
        "unrestricted_reference_from_spec_with_jax_uks",
    }
    public_legacy = {
        "ReferenceRun",
        "ReferenceSpecConfig",
        "build_reference",
        "put_reference_on_device",
        "put_restricted_reference_on_device",
        "run_reference_from_spec",
    }

    assert scf_preferred.issubset(set(scf.__all__))
    assert public_preferred.issubset(set(td_graddft.__all__))
    assert scf_legacy.isdisjoint(set(scf.__all__))
    assert public_legacy.isdisjoint(set(td_graddft.__all__))
    assert {"MoleculeRun", "MoleculeSpecConfig", "run_molecule_from_spec"}.issubset(
        set(workflows.__all__)
    )
    assert {"ReferenceRun", "ReferenceSpecConfig", "run_reference_from_spec"}.isdisjoint(
        set(workflows.__all__)
    )
