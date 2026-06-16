from __future__ import annotations

import sys
import types
import zipfile
from pathlib import Path

import jax.numpy as jnp

from td_graddft.data.graddft_dataset import (
    GRADDFT_GROUND_ATOM_SYMBOLS,
    GRADDFT_XND_ATOM_ENERGY_COLUMN,
    GradDFTGroundAtomRecord,
    build_graddft_ground_atom_train_test_data,
    build_graddft_ground_atom_datum,
    load_graddft_ground_atom_records,
    parse_graddft_test_train_ratio,
    split_graddft_ground_atom_records,
)
from td_graddft.scf.molecules import RestrictedMolecule, UnrestrictedMolecule
from td_graddft.training import GroundStateDatum


def _minimal_restricted_molecule(*, mf_energy: float = -1.0) -> RestrictedMolecule:
    return RestrictedMolecule(
        ao=jnp.ones((1, 1)),
        grid=__import__("td_graddft.scf.molecules", fromlist=["QuadratureGrid"]).QuadratureGrid(
            weights=jnp.ones((1,)),
            coords=jnp.zeros((1, 3)),
        ),
        dipole_integrals=jnp.zeros((3, 1, 1)),
        rep_tensor=jnp.zeros((0, 0, 0, 0)),
        mo_coeff=jnp.ones((2, 1, 1)),
        mo_occ=jnp.ones((2, 1)),
        mo_energy=jnp.zeros((2, 1)),
        rdm1=jnp.ones((2, 1, 1)) * 0.5,
        h1e=jnp.zeros((1, 1)),
        nuclear_repulsion=0.0,
        atom_coords=jnp.zeros((1, 3)),
        atom_charges=jnp.ones((1,)),
        overlap_matrix=jnp.eye(1),
        ao_deriv1=jnp.zeros((4, 1, 1)),
        mf_energy=mf_energy,
        exact_exchange_fraction=0.2,
        nocc=1,
        hfx_omega_values=jnp.asarray((0.0, 0.4)),
        hfx_local=jnp.zeros((2, 1, 2)),
        hfx_nu=jnp.zeros((2, 1, 1, 1)),
        eri_pair_matrix=jnp.ones((1, 1)),
    )


def _write_minimal_xnd_atoms_xlsx(path: Path) -> None:
    shared_strings = [
        "Atoms",
        "Dimers",
        GRADDFT_XND_ATOM_ENERGY_COLUMN,
        "H",
        "K",
    ]
    shared_items = "".join(
        f'<si><t>{text}</t></si>'
        for text in shared_strings
    )
    atoms_sheet = """
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1">
      <c r="B1" t="s"><v>2</v></c>
    </row>
    <row r="2">
      <c r="A2" t="s"><v>3</v></c>
      <c r="B2"><v>-0.500000</v></c>
    </row>
    <row r="3">
      <c r="A3" t="s"><v>4</v></c>
      <c r="B3"><v>-599.164291</v></c>
    </row>
  </sheetData>
</worksheet>
""".strip()
    dimers_sheet = """
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData/>
</worksheet>
""".strip()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            """
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
</Types>
""".strip(),
        )
        archive.writestr(
            "xl/workbook.xml",
            """
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Atoms" sheetId="1" r:id="rId1"/>
    <sheet name="Dimers" sheetId="2" r:id="rId2"/>
  </sheets>
</workbook>
""".strip(),
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            """
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
</Relationships>
""".strip(),
        )
        archive.writestr(
            "xl/sharedStrings.xml",
            f"""
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
     count="{len(shared_strings)}" uniqueCount="{len(shared_strings)}">
  {shared_items}
</sst>
""".strip(),
        )
        archive.writestr("xl/worksheets/sheet1.xml", atoms_sheet)
        archive.writestr("xl/worksheets/sheet2.xml", dimers_sheet)


def _write_minimal_atoms_xlsx_with_symbols(path: Path, symbols: tuple[str, ...]) -> None:
    shared_strings = ["Atoms", GRADDFT_XND_ATOM_ENERGY_COLUMN, *symbols]
    shared_items = "".join(f'<si><t>{text}</t></si>' for text in shared_strings)
    rows = ['<row r="1"><c r="B1" t="s"><v>1</v></c></row>']
    for row_index, symbol_index in enumerate(range(2, len(shared_strings)), start=2):
        rows.append(
            f'<row r="{row_index}">'
            f'<c r="A{row_index}" t="s"><v>{symbol_index}</v></c>'
            f'<c r="B{row_index}"><v>{-float(row_index):.6f}</v></c>'
            "</row>"
        )
    atoms_sheet = (
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(rows)}</sheetData>"
        "</worksheet>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            """
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
</Types>
""".strip(),
        )
        archive.writestr(
            "xl/workbook.xml",
            """
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Atoms" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>
""".strip(),
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            """
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>
""".strip(),
        )
        archive.writestr(
            "xl/sharedStrings.xml",
            f"""
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
     count="{len(shared_strings)}" uniqueCount="{len(shared_strings)}">
  {shared_items}
</sst>
""".strip(),
        )
        archive.writestr("xl/worksheets/sheet1.xml", atoms_sheet)


def test_load_graddft_ground_atom_records_reads_xnd_atoms_sheet(tmp_path: Path):
    xlsx = tmp_path / "XND_dataset.xlsx"
    _write_minimal_xnd_atoms_xlsx(xlsx)

    from td_graddft.data import load_graddft_ground_atom_records as public_loader

    assert public_loader is load_graddft_ground_atom_records

    records = load_graddft_ground_atom_records(xlsx)

    assert [record.symbol for record in records] == ["H", "K"]
    assert [record.split for record in records] == ["train", "test"]
    assert [record.spin for record in records] == [1, 1]
    assert records[0].atom == "H 0.0 0.0 0.0"
    assert records[1].target_energy_h == -599.164291


def test_load_graddft_ground_atom_records_filters_symbols_before_strict_validation(tmp_path: Path):
    xlsx = tmp_path / "XND_dataset.xlsx"
    _write_minimal_atoms_xlsx_with_symbols(xlsx, ("H", "Og"))

    records = load_graddft_ground_atom_records(xlsx, symbols=("H",))

    assert [record.symbol for record in records] == ["H"]


def test_build_graddft_ground_atom_datum_selects_rks_or_uks_by_spin():
    closed_shell = GradDFTGroundAtomRecord(
        symbol="He",
        split="train",
        target_energy_h=-2.903724,
        spin=0,
    )
    open_shell = GradDFTGroundAtomRecord(
        symbol="H",
        split="train",
        target_energy_h=-0.5,
        spin=1,
    )

    common_kwargs = dict(
        basis="sto-3g",
        reference_builder="jax",
        xc_spec="hf",
        grids_level=0,
        max_l=1,
        integral_backend="jax",
        init_guess="1e",
        scf_max_cycle=1,
    )
    closed_datum = build_graddft_ground_atom_datum(closed_shell, **common_kwargs)
    open_datum = build_graddft_ground_atom_datum(open_shell, **common_kwargs)

    assert isinstance(closed_datum.molecule, RestrictedMolecule)
    assert isinstance(open_datum.molecule, UnrestrictedMolecule)
    assert float(jnp.asarray(closed_datum.target_total_energy)) == closed_shell.target_energy_h
    assert float(jnp.asarray(open_datum.target_total_energy)) == open_shell.target_energy_h
    assert closed_datum.molecule.mf_energy == closed_shell.target_energy_h
    assert open_datum.molecule.mf_energy == open_shell.target_energy_h


def test_build_graddft_ground_atom_datum_can_use_pyscf_reference_builder(monkeypatch):
    import td_graddft.data.graddft_dataset as module

    record = GradDFTGroundAtomRecord(
        symbol="Be",
        split="train",
        target_energy_h=-14.667,
        spin=0,
    )
    calls = []

    def fake_pyscf_builder(record_arg, **kwargs):
        calls.append((record_arg, kwargs))
        return _minimal_restricted_molecule(mf_energy=-14.6723)

    monkeypatch.setattr(
        module,
        "_build_graddft_ground_atom_molecule_from_pyscf",
        fake_pyscf_builder,
        raising=False,
    )

    datum = module.build_graddft_ground_atom_datum(
        record,
        basis="def2-tzvp",
        reference_builder="pyscf",
        xc_spec="b3lyp",
        grids_level=2,
        compute_local_hfx_features=True,
        compute_local_hfx_aux=True,
    )

    assert isinstance(datum, GroundStateDatum)
    assert datum.molecule.mf_energy == -14.6723
    assert float(jnp.asarray(datum.target_total_energy)) == record.target_energy_h
    assert calls == [
        (
            record,
            {
                "basis": "def2-tzvp",
                "xc_spec": "b3lyp",
                "grids_level": 2,
                "max_l": 3,
                "init_guess": "1e",
                "scf_max_cycle": 80,
                "scf_conv_tol": 1e-10,
                "scf_conv_tol_density": 1e-08,
                "scf_damping": 0.0,
                "scf_level_shift": 0.0,
                "compute_local_hfx_features": True,
                "compute_local_hfx_aux": True,
                "hfx_omega_values": (0.0, 0.4),
                "hfx_chunk_size": 512,
                "hfx_nu_storage": "dense",
                "verbose": 0,
            },
        )
    ]


def test_pyscf_atom_reference_builder_retries_with_frac_occ_for_open_shell(monkeypatch):
    import td_graddft.data.graddft_dataset as module
    import td_graddft.data.reference as reference_module

    record = GradDFTGroundAtomRecord(
        symbol="B",
        split="train",
        target_energy_h=-24.6,
        spin=1,
    )
    attempts = []
    sentinel = object()

    class FakeMole:
        def build(self):
            return self

    class FakeMF:
        def __init__(self, mol):
            self.mol = mol
            self.grids = types.SimpleNamespace(level=None)
            self.converged = False
            self.newton_used = False
            self.frac_occ_used = False

        def newton(self):
            new_mf = type(self)(self.mol)
            new_mf.newton_used = True
            return new_mf

        def kernel(self):
            attempts.append(
                {
                    "init_guess": self.init_guess,
                    "damping": self.damping,
                    "level_shift": self.level_shift,
                    "newton": self.newton_used,
                    "frac_occ": self.frac_occ_used,
                    "max_cycle": self.max_cycle,
                }
            )
            self.converged = bool(self.frac_occ_used)

    fake_pyscf = types.ModuleType("pyscf")
    fake_dft = types.ModuleType("pyscf.dft")
    fake_gto = types.ModuleType("pyscf.gto")
    fake_scf = types.ModuleType("pyscf.scf")
    fake_addons = types.SimpleNamespace()

    def fake_frac_occ(mf):
        mf.frac_occ_used = True
        return mf

    fake_addons.frac_occ = fake_frac_occ
    fake_dft.UKS = FakeMF
    fake_dft.RKS = FakeMF
    fake_gto.Mole = FakeMole
    fake_scf.addons = fake_addons
    fake_pyscf.dft = fake_dft
    fake_pyscf.gto = fake_gto
    fake_pyscf.scf = fake_scf
    monkeypatch.setitem(sys.modules, "pyscf", fake_pyscf)
    monkeypatch.setitem(sys.modules, "pyscf.dft", fake_dft)
    monkeypatch.setitem(sys.modules, "pyscf.gto", fake_gto)
    monkeypatch.setitem(sys.modules, "pyscf.scf", fake_scf)
    monkeypatch.setattr(
        reference_module,
        "unrestricted_reference_from_pyscf",
        lambda mf, **kwargs: sentinel,
    )

    molecule = module.build_graddft_ground_atom_molecule(
        record,
        basis="def2-tzvp",
        reference_builder="pyscf",
        xc_spec="b3lyp",
        grids_level=2,
        scf_max_cycle=512,
        scf_damping=0.25,
        scf_level_shift=0.0,
    )

    assert molecule is sentinel
    assert [attempt["frac_occ"] for attempt in attempts] == [False, True]
    assert [attempt["newton"] for attempt in attempts] == [False, False]
    assert attempts[0]["max_cycle"] == 80
    assert attempts[1]["init_guess"] == "minao"
    assert attempts[1]["level_shift"] == 0.2


def test_split_graddft_ground_atom_records_uses_test_train_ratio_2_to_8():
    records = tuple(
        GradDFTGroundAtomRecord(
            symbol=symbol,
            split="source",
            target_energy_h=-float(index + 1),
            spin=0,
        )
        for index, symbol in enumerate(GRADDFT_GROUND_ATOM_SYMBOLS[:10])
    )

    split = split_graddft_ground_atom_records(
        records,
        test_train_ratio="2:8",
        seed=7,
    )
    repeated = split_graddft_ground_atom_records(
        records,
        test_train_ratio=(2, 8),
        seed=7,
    )

    assert parse_graddft_test_train_ratio("2:8") == (2, 8)
    assert len(split.test_records) == 2
    assert len(split.train_records) == 8
    assert split.test_symbols == repeated.test_symbols
    assert split.train_symbols == repeated.train_symbols
    assert {record.split for record in split.test_records} == {"test"}
    assert {record.split for record in split.train_records} == {"train"}
    assert {record.split for record in records} == {"source"}


def test_build_graddft_ground_atom_train_test_data_produces_trainer_inputs():
    records = (
        GradDFTGroundAtomRecord(
            symbol="He",
            split="source",
            target_energy_h=-2.903724,
            spin=0,
        ),
        GradDFTGroundAtomRecord(
            symbol="H",
            split="source",
            target_energy_h=-0.5,
            spin=1,
        ),
    )

    dataset = build_graddft_ground_atom_train_test_data(
        records,
        basis="sto-3g",
        test_train_ratio="1:1",
        seed=0,
        reference_builder="jax",
        xc_spec="hf",
        grids_level=0,
        max_l=1,
        integral_backend="jax",
        init_guess="1e",
        scf_max_cycle=1,
    )

    assert len(dataset.train_data) == 1
    assert len(dataset.test_data) == 1
    assert all(isinstance(datum, GroundStateDatum) for datum in dataset.train_data)
    assert all(isinstance(datum, GroundStateDatum) for datum in dataset.test_data)
    assert dataset.train_records[0].split == "train"
    assert dataset.test_records[0].split == "test"
