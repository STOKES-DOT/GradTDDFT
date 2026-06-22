import pytest

from td_graddft.data.ris_auxbasis import minimal_ris_auxbasis_for_mol


class _FakeMol:
    _basis = {"C1": None, "H2": None, "O3": None}


def test_minimal_ris_auxbasis_matches_gpu4pyscf_default_exponents():
    auxbasis = minimal_ris_auxbasis_for_mol(_FakeMol(), theta=0.2, fitting_basis="sp")

    assert auxbasis["C1"][0] == [0, [pytest.approx(0.1320292535005648), 1.0]]
    assert auxbasis["H2"] == [[0, [pytest.approx(0.1999828038466018), 1.0]]]
    assert auxbasis["O3"][0] == [0, [pytest.approx(0.2587932305664396), 1.0]]
    assert auxbasis["C1"][1][0] == 1
    assert auxbasis["O3"][1][0] == 1


def test_minimal_ris_auxbasis_rejects_unknown_fit():
    with pytest.raises(ValueError, match="fitting_basis"):
        minimal_ris_auxbasis_for_mol(_FakeMol(), fitting_basis="spdf")
