from pathlib import Path


TOOLS = tuple(Path("tools").glob("*.py"))
EXAMPLES = tuple(Path("examples").glob("*.py"))


def test_tools_use_tdscf_facade_for_restricted_response():
    offenders = []
    for path in TOOLS:
        text = path.read_text()
        if (
            "from td_graddft.tddft import RestrictedCasidaTDDFT" in text
            or "RestrictedCasidaTDDFT(" in text
        ):
            offenders.append(str(path))

    assert offenders == []


def test_tools_use_neural_xc_facade_constructor():
    offenders = []
    for path in TOOLS:
        text = path.read_text()
        if (
            "from td_graddft.neural_xc import make_neural_xc_functional" in text
            or "make_neural_xc_functional(" in text
        ):
            offenders.append(str(path))

    assert offenders == []


def test_user_scripts_avoid_deprecated_pyscf_bridge_imports():
    offenders = []
    for path in TOOLS + EXAMPLES:
        text = path.read_text()
        if "td_graddft.pyscf_bridge" in text:
            offenders.append(str(path))

    assert offenders == []


def test_user_scripts_use_neural_xc_long_range_facade():
    offenders = []
    for path in TOOLS + EXAMPLES:
        text = path.read_text()
        if "LongRangeCorrectedFunctional" in text or "LongRangeXCNet" in text:
            offenders.append(str(path))

    assert offenders == []


def test_user_scripts_avoid_old_dm21_like_names():
    offenders = []
    for path in TOOLS + EXAMPLES:
        text = path.read_text()
        if "DM21Like" in text or "dm21_like" in text or "make_dm21_like" in text:
            offenders.append(str(path))

    assert offenders == []
