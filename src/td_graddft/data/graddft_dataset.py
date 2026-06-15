from __future__ import annotations

from dataclasses import dataclass, replace
import importlib
from pathlib import Path, PurePosixPath
import posixpath
import re
import random
from typing import Any, Iterable
import zipfile
import xml.etree.ElementTree as ET

import jax.numpy as jnp

from .molecule import atomic_number


GRADDFT_XND_ATOM_ENERGY_COLUMN = "ccsd(t)/cbs energy 3-point"
GRADDFT_GROUND_ATOM_SYMBOLS = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
)
GRADDFT_GROUND_TEST_ATOMS = frozenset(
    ("K", "Ga", "As", "Br", "Ti", "Cr", "Fe", "Ni", "Zn")
)
_XLSX_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_XLSX_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_XLSX_OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_CELL_REF_RE = re.compile(r"([A-Z]+)")
_TABLE_INDEX_KEY = "__index__"


_NEUTRAL_ATOM_SPINS = {
    "H": 1,
    "He": 0,
    "Li": 1,
    "Be": 0,
    "B": 1,
    "C": 2,
    "N": 3,
    "O": 2,
    "F": 1,
    "Ne": 0,
    "Na": 1,
    "Mg": 0,
    "Al": 1,
    "Si": 2,
    "P": 3,
    "S": 2,
    "Cl": 1,
    "Ar": 0,
    "K": 1,
    "Ca": 0,
    "Sc": 1,
    "Ti": 2,
    "V": 3,
    "Cr": 6,
    "Mn": 5,
    "Fe": 4,
    "Co": 3,
    "Ni": 2,
    "Cu": 1,
    "Zn": 0,
    "Ga": 1,
    "Ge": 2,
    "As": 3,
    "Se": 2,
    "Br": 1,
    "Kr": 0,
}


@dataclass(frozen=True)
class GradDFTGroundAtomRecord:
    symbol: str
    split: str
    target_energy_h: float
    spin: int
    charge: int = 0
    unit: str = "Angstrom"
    source_row: int | None = None
    source_path: str | None = None
    energy_column: str = GRADDFT_XND_ATOM_ENERGY_COLUMN

    @property
    def system(self) -> str:
        return self.symbol

    @property
    def atom(self) -> str:
        return f"{self.symbol} 0.0 0.0 0.0"


@dataclass(frozen=True)
class GradDFTGroundAtomSplit:
    train_records: tuple[GradDFTGroundAtomRecord, ...]
    test_records: tuple[GradDFTGroundAtomRecord, ...]
    test_train_ratio: tuple[int, int]
    seed: int

    @property
    def train_symbols(self) -> tuple[str, ...]:
        return tuple(record.symbol for record in self.train_records)

    @property
    def test_symbols(self) -> tuple[str, ...]:
        return tuple(record.symbol for record in self.test_records)


@dataclass(frozen=True)
class GradDFTGroundAtomTrainTestData:
    train_records: tuple[GradDFTGroundAtomRecord, ...]
    test_records: tuple[GradDFTGroundAtomRecord, ...]
    train_data: tuple[Any, ...]
    test_data: tuple[Any, ...]
    test_train_ratio: tuple[int, int]
    seed: int


def neutral_atom_spin(symbol: str) -> int:
    clean = _normalize_symbol(symbol)
    if clean not in _NEUTRAL_ATOM_SPINS:
        raise KeyError(f"No GradDFT neutral atom spin configured for {symbol!r}.")
    return int(_NEUTRAL_ATOM_SPINS[clean])


def graddft_ground_atom_split(symbol: str) -> str:
    clean = _normalize_symbol(symbol)
    return "test" if clean in GRADDFT_GROUND_TEST_ATOMS else "train"


def parse_graddft_test_train_ratio(raw: str | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(":")]
        if len(parts) != 2:
            raise ValueError("test_train_ratio must use '<test>:<train>', for example '2:8'.")
        test_part, train_part = parts
    else:
        if len(raw) != 2:
            raise ValueError("test_train_ratio must contain exactly two integer parts.")
        test_part, train_part = raw
    test_count = int(test_part)
    train_count = int(train_part)
    if test_count <= 0 or train_count <= 0:
        raise ValueError("test_train_ratio parts must be positive integers.")
    return test_count, train_count


def split_graddft_ground_atom_records(
    records: Iterable[GradDFTGroundAtomRecord],
    *,
    test_train_ratio: str | tuple[int, int] | list[int] = "2:8",
    seed: int = 0,
) -> GradDFTGroundAtomSplit:
    """Split GradDFT atom records using a deterministic test:train ratio."""

    ordered = tuple(records)
    if not ordered:
        raise ValueError("At least one GradDFT ground atom record is required.")
    ratio = parse_graddft_test_train_ratio(test_train_ratio)
    test_ratio, train_ratio = ratio
    test_size = round(len(ordered) * test_ratio / (test_ratio + train_ratio))
    test_size = max(1, min(len(ordered) - 1, int(test_size)))
    rng = random.Random(int(seed))
    test_indices = frozenset(rng.sample(range(len(ordered)), test_size))
    train_records = []
    test_records = []
    for index, record in enumerate(ordered):
        if index in test_indices:
            test_records.append(replace(record, split="test"))
        else:
            train_records.append(replace(record, split="train"))
    return GradDFTGroundAtomSplit(
        train_records=tuple(train_records),
        test_records=tuple(test_records),
        test_train_ratio=ratio,
        seed=int(seed),
    )


def load_graddft_ground_atom_records(
    xnd_dataset_xlsx: str | Path,
    *,
    energy_column: str = GRADDFT_XND_ATOM_ENERGY_COLUMN,
    symbols: Iterable[str] | None = None,
) -> tuple[GradDFTGroundAtomRecord, ...]:
    """Load GradDFT XND atom ground-state targets from the raw Atoms sheet.

    This reads the original GradDFT `XND_dataset.xlsx` directly. It does not
    generate reference energies with PySCF/FCI/CASSCF.
    """

    path = Path(xnd_dataset_xlsx)
    selected = (
        None if symbols is None else frozenset(_normalize_symbol(sym) for sym in symbols)
    )
    rows = _read_xlsx_indexed_sheet(path, sheet_name="Atoms")
    records: list[GradDFTGroundAtomRecord] = []
    for row_index, row in enumerate(rows, start=2):
        symbol_raw = row.get(_TABLE_INDEX_KEY)
        if symbol_raw in (None, ""):
            continue
        raw_symbol = _clean_symbol_text(str(symbol_raw))
        if selected is not None and raw_symbol not in selected:
            continue
        symbol = _normalize_symbol(raw_symbol)
        if energy_column not in row:
            raise KeyError(f"GradDFT Atoms sheet is missing energy column {energy_column!r}.")
        target = row[energy_column]
        if target in (None, ""):
            raise ValueError(f"GradDFT atom {symbol} has no target energy in {energy_column!r}.")
        records.append(
            GradDFTGroundAtomRecord(
                symbol=symbol,
                split=graddft_ground_atom_split(symbol),
                target_energy_h=float(target),
                spin=neutral_atom_spin(symbol),
                source_row=row_index,
                source_path=str(path),
                energy_column=str(energy_column),
            )
        )
    return tuple(records)


def build_graddft_ground_atom_molecule(
    record: GradDFTGroundAtomRecord,
    *,
    basis: str,
    reference_builder: str = "pyscf",
    xc_spec: str = "hf",
    grids_level: int = 0,
    max_l: int = 3,
    integral_backend: str = "jax",
    grid_ao_backend: str = "jax",
    init_guess: Any = "1e",
    scf_max_cycle: int = 80,
    scf_conv_tol: float = 1e-10,
    scf_conv_tol_density: float = 1e-8,
    scf_damping: float = 0.0,
    scf_level_shift: float = 0.0,
    rks_jk_backend: str = "full",
    compute_local_hfx_features: bool = False,
    compute_local_hfx_aux: bool = False,
    hfx_omega_values: tuple[float, ...] = (0.0, 0.4),
    hfx_chunk_size: int = 512,
    hfx_nu_storage: str = "dense",
    verbose: int = 0,
) -> Any:
    """Build one TD-GradDFT ground-state molecule from a GradDFT atom record."""

    builder_mode = str(reference_builder).lower()
    if builder_mode == "pyscf":
        return _build_graddft_ground_atom_molecule_from_pyscf(
            record,
            basis=str(basis),
            xc_spec=str(xc_spec),
            grids_level=int(grids_level),
            max_l=int(max_l),
            init_guess=init_guess,
            scf_max_cycle=int(scf_max_cycle),
            scf_conv_tol=float(scf_conv_tol),
            scf_conv_tol_density=float(scf_conv_tol_density),
            scf_damping=float(scf_damping),
            scf_level_shift=float(scf_level_shift),
            compute_local_hfx_features=bool(compute_local_hfx_features),
            compute_local_hfx_aux=bool(compute_local_hfx_aux),
            hfx_omega_values=tuple(float(value) for value in hfx_omega_values),
            hfx_chunk_size=int(hfx_chunk_size),
            hfx_nu_storage=str(hfx_nu_storage),
            verbose=int(verbose),
        )
    if builder_mode != "jax":
        raise ValueError("reference_builder must be 'jax' or 'pyscf'.")

    if int(record.spin) == 0:
        from td_graddft.scf import RKSConfig, restricted_molecule_from_spec_with_jax_rks

        return restricted_molecule_from_spec_with_jax_rks(
            atom=record.atom,
            basis=str(basis),
            xc_spec=str(xc_spec),
            unit=str(record.unit),
            charge=int(record.charge),
            spin=int(record.spin),
            cart=True,
            grids_level=int(grids_level),
            max_l=int(max_l),
            rks_config=RKSConfig(
                xc_spec=str(xc_spec),
                max_cycle=int(scf_max_cycle),
                conv_tol=float(scf_conv_tol),
                conv_tol_density=float(scf_conv_tol_density),
                damping=float(scf_damping),
                level_shift=float(scf_level_shift),
                jk_backend=str(rks_jk_backend),
            ),
            grid_ao_backend=str(grid_ao_backend),
            integral_backend=str(integral_backend),
            energy_target=float(record.target_energy_h),
            compute_local_hfx_features=bool(compute_local_hfx_features),
            compute_local_hfx_aux=bool(compute_local_hfx_aux),
            hfx_omega_values=tuple(float(value) for value in hfx_omega_values),
            hfx_chunk_size=int(hfx_chunk_size),
            init_guess=init_guess,
            verbose=int(verbose),
        )

    from td_graddft.scf import UKSConfig, unrestricted_molecule_from_spec_with_jax_uks

    return unrestricted_molecule_from_spec_with_jax_uks(
        atom=record.atom,
        basis=str(basis),
        xc_spec=str(xc_spec),
        unit=str(record.unit),
        charge=int(record.charge),
        spin=int(record.spin),
        cart=True,
        grids_level=int(grids_level),
        max_l=int(max_l),
        uks_config=UKSConfig(
            xc_spec=str(xc_spec),
            max_cycle=int(scf_max_cycle),
            conv_tol=float(scf_conv_tol),
            conv_tol_density=float(scf_conv_tol_density),
            damping=float(scf_damping),
            level_shift=float(scf_level_shift),
        ),
        grid_ao_backend=str(grid_ao_backend),
        integral_backend=str(integral_backend),
        energy_target=float(record.target_energy_h),
        compute_local_hfx_features=bool(compute_local_hfx_features),
        compute_local_hfx_aux=bool(compute_local_hfx_aux),
        hfx_omega_values=tuple(float(value) for value in hfx_omega_values),
        hfx_chunk_size=int(hfx_chunk_size),
        init_guess=init_guess,
        verbose=int(verbose),
    )


def _build_graddft_ground_atom_molecule_from_pyscf(
    record: GradDFTGroundAtomRecord,
    *,
    basis: str,
    xc_spec: str,
    grids_level: int,
    max_l: int,
    compute_local_hfx_features: bool,
    compute_local_hfx_aux: bool,
    hfx_omega_values: tuple[float, ...],
    hfx_chunk_size: int,
    hfx_nu_storage: str,
    init_guess: Any,
    scf_max_cycle: int,
    scf_conv_tol: float,
    scf_conv_tol_density: float,
    scf_damping: float,
    scf_level_shift: float,
    verbose: int,
) -> Any:
    """Build one GradDFT atom reference through PySCF, then package cached arrays."""

    del max_l
    try:
        dft = importlib.import_module("pyscf.dft")
        gto = importlib.import_module("pyscf.gto")
    except ModuleNotFoundError as exc:
        raise ImportError("PySCF is required for reference_builder='pyscf'.") from exc

    mol = gto.Mole()
    mol.atom = record.atom
    mol.unit = str(record.unit)
    mol.basis = str(basis)
    mol.charge = int(record.charge)
    mol.spin = int(record.spin)
    mol.cart = True
    mol.verbose = int(verbose)
    mol.build()

    mf = _run_pyscf_atom_reference_scf(
        dft,
        mol,
        restricted=(int(record.spin) == 0),
        xc_spec=str(xc_spec),
        grids_level=int(grids_level),
        init_guess=init_guess,
        scf_max_cycle=int(scf_max_cycle),
        scf_conv_tol=float(scf_conv_tol),
        scf_conv_tol_density=float(scf_conv_tol_density),
        scf_damping=float(scf_damping),
        scf_level_shift=float(scf_level_shift),
        verbose=int(verbose),
        symbol=str(record.symbol),
        basis=str(basis),
    )

    if int(record.spin) == 0:
        from td_graddft.data.reference import restricted_reference_from_pyscf

        storage = "dense" if str(hfx_nu_storage) == "array" else str(hfx_nu_storage)
        return restricted_reference_from_pyscf(
            mf,
            compute_local_hfx_features=bool(compute_local_hfx_features),
            compute_local_hfx_aux=bool(compute_local_hfx_aux),
            hfx_omega_values=tuple(float(value) for value in hfx_omega_values),
            hfx_chunk_size=int(hfx_chunk_size),
            array_backend="jax",
            hfx_nu_storage=storage,
        )

    from td_graddft.data.reference import unrestricted_reference_from_pyscf

    storage = "dense" if str(hfx_nu_storage) == "array" else str(hfx_nu_storage)
    return unrestricted_reference_from_pyscf(
        mf,
        compute_local_hfx_features=bool(compute_local_hfx_features),
        compute_local_hfx_aux=bool(compute_local_hfx_aux),
        hfx_omega_values=tuple(float(value) for value in hfx_omega_values),
        hfx_chunk_size=int(hfx_chunk_size),
        array_backend="jax",
        hfx_nu_storage=storage,
    )


def _pyscf_atom_reference_attempts(
    *,
    init_guess: Any,
    scf_max_cycle: int,
    scf_damping: float,
    scf_level_shift: float,
) -> tuple[dict[str, Any], ...]:
    first_guess = str(init_guess) if isinstance(init_guess, str) else "minao"
    first_max_cycle = min(max(1, int(scf_max_cycle)), 80)
    return (
        {
            "init_guess": first_guess,
            "damping": float(scf_damping),
            "level_shift": float(scf_level_shift),
            "max_cycle": first_max_cycle,
            "newton": False,
        },
        {
            "init_guess": "minao",
            "damping": max(float(scf_damping), 0.20),
            "level_shift": max(float(scf_level_shift), 0.20),
            "max_cycle": max(int(scf_max_cycle), 160),
            "newton": False,
            "frac_occ": True,
        },
        {
            "init_guess": "atom",
            "damping": max(float(scf_damping), 0.30),
            "level_shift": max(float(scf_level_shift), 0.50),
            "max_cycle": max(int(scf_max_cycle), 160),
            "newton": False,
            "frac_occ": True,
        },
        {
            "init_guess": "atom",
            "damping": 0.0,
            "level_shift": 0.0,
            "max_cycle": max(int(scf_max_cycle), 512),
            "newton": True,
            "frac_occ": False,
        },
    )


def _configure_pyscf_atom_mf(
    mf: Any,
    *,
    xc_spec: str,
    grids_level: int,
    scf_conv_tol: float,
    scf_conv_tol_density: float,
    verbose: int,
    attempt: dict[str, Any],
) -> Any:
    mf.xc = str(xc_spec)
    mf.grids.level = int(grids_level)
    mf.max_cycle = int(attempt["max_cycle"])
    mf.conv_tol = float(scf_conv_tol)
    mf.conv_tol_grad = float(scf_conv_tol_density)
    mf.damping = float(attempt["damping"])
    mf.level_shift = float(attempt["level_shift"])
    mf.init_guess = str(attempt["init_guess"])
    mf.verbose = int(verbose)
    return mf


def _run_pyscf_atom_reference_scf(
    dft_module: Any,
    mol: Any,
    *,
    restricted: bool,
    xc_spec: str,
    grids_level: int,
    init_guess: Any,
    scf_max_cycle: int,
    scf_conv_tol: float,
    scf_conv_tol_density: float,
    scf_damping: float,
    scf_level_shift: float,
    verbose: int,
    symbol: str,
    basis: str,
) -> Any:
    attempts = _pyscf_atom_reference_attempts(
        init_guess=init_guess,
        scf_max_cycle=int(scf_max_cycle),
        scf_damping=float(scf_damping),
        scf_level_shift=float(scf_level_shift),
    )
    last_mf = None
    for attempt in attempts:
        mf = dft_module.RKS(mol) if bool(restricted) else dft_module.UKS(mol)
        if bool(attempt["newton"]):
            mf = mf.newton()
        if bool(attempt.get("frac_occ", False)):
            scf = importlib.import_module("pyscf.scf")
            mf = scf.addons.frac_occ(mf)
        mf = _configure_pyscf_atom_mf(
            mf,
            xc_spec=str(xc_spec),
            grids_level=int(grids_level),
            scf_conv_tol=float(scf_conv_tol),
            scf_conv_tol_density=float(scf_conv_tol_density),
            verbose=int(verbose),
            attempt=attempt,
        )
        mf.kernel()
        last_mf = mf
        if bool(getattr(mf, "converged", False)):
            return mf
    raise RuntimeError(
        f"PySCF atom reference did not converge for {symbol} "
        f"with basis={basis!r}, xc={xc_spec!r}, grid={grids_level}; "
        f"last_energy={getattr(last_mf, 'e_tot', None)!r}."
    )


def build_graddft_ground_atom_datum(
    record: GradDFTGroundAtomRecord,
    *,
    basis: str,
    **molecule_kwargs: Any,
) -> Any:
    """Build one `GroundStateDatum` from a GradDFT ground-state atom record."""

    from td_graddft.training import GroundStateCoreDatum, GroundStateDatum

    molecule = build_graddft_ground_atom_molecule(record, basis=basis, **molecule_kwargs)
    return GroundStateDatum.from_parts(
        molecule,
        core=GroundStateCoreDatum(
            target_total_energy=jnp.asarray(record.target_energy_h),
        ),
    )


def build_graddft_ground_atom_train_test_data(
    records: Iterable[GradDFTGroundAtomRecord],
    *,
    basis: str,
    test_train_ratio: str | tuple[int, int] | list[int] = "2:8",
    seed: int = 0,
    **molecule_kwargs: Any,
) -> GradDFTGroundAtomTrainTestData:
    """Build train/test `GroundStateDatum` tuples for a GradDFT atom experiment."""

    split = split_graddft_ground_atom_records(
        records,
        test_train_ratio=test_train_ratio,
        seed=int(seed),
    )
    train_data = tuple(
        build_graddft_ground_atom_datum(record, basis=basis, **molecule_kwargs)
        for record in split.train_records
    )
    test_data = tuple(
        build_graddft_ground_atom_datum(record, basis=basis, **molecule_kwargs)
        for record in split.test_records
    )
    return GradDFTGroundAtomTrainTestData(
        train_records=split.train_records,
        test_records=split.test_records,
        train_data=train_data,
        test_data=test_data,
        test_train_ratio=split.test_train_ratio,
        seed=split.seed,
    )


def _normalize_symbol(symbol: str) -> str:
    clean = _clean_symbol_text(symbol)
    atomic_number(clean)
    return clean


def _clean_symbol_text(symbol: str) -> str:
    return str(symbol).strip().capitalize()


def _ns(name: str) -> str:
    return f"{{{_XLSX_MAIN_NS}}}{name}"


def _rel_ns(name: str) -> str:
    return f"{{{_XLSX_REL_NS}}}{name}"


def _column_index(cell_ref: str) -> int:
    match = _CELL_REF_RE.match(str(cell_ref))
    if match is None:
        raise ValueError(f"Invalid XLSX cell reference {cell_ref!r}.")
    value = 0
    for char in match.group(1):
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        raw = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(raw)
    strings = []
    for item in root.findall(_ns("si")):
        strings.append("".join(text.text or "" for text in item.iter(_ns("t"))))
    return strings


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> Any:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.iter(_ns("t")))
    value = cell.find(_ns("v"))
    if value is None or value.text is None:
        return None
    text = value.text
    if cell_type == "s":
        return shared_strings[int(text)]
    if cell_type == "str":
        return text
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _sheet_target_path(archive: zipfile.ZipFile, sheet_name: str) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_by_id = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall(_rel_ns("Relationship"))
    }
    rid_attr = f"{{{_XLSX_OFFICE_REL_NS}}}id"
    for sheet in workbook.findall(f".//{_ns('sheet')}"):
        if sheet.attrib.get("name") != sheet_name:
            continue
        rid = sheet.attrib[rid_attr]
        target = rel_by_id[rid]
        if target.startswith("/"):
            return target.lstrip("/")
        return posixpath.normpath(str(PurePosixPath("xl") / target))
    raise KeyError(f"XLSX workbook does not contain sheet {sheet_name!r}.")


def _read_xlsx_indexed_sheet(path: Path, *, sheet_name: str) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path, "r") as archive:
        shared = _shared_strings(archive)
        sheet_path = _sheet_target_path(archive, sheet_name)
        root = ET.fromstring(archive.read(sheet_path))
        table_rows: list[dict[int, Any]] = []
        for row in root.findall(f".//{_ns('row')}"):
            values: dict[int, Any] = {}
            for cell in row.findall(_ns("c")):
                ref = cell.attrib.get("r")
                if ref is None:
                    continue
                values[_column_index(ref)] = _cell_value(cell, shared)
            if values:
                table_rows.append(values)

    if not table_rows:
        return []
    header = table_rows[0]
    columns = {
        col_idx: str(value)
        for col_idx, value in header.items()
        if col_idx > 0 and value not in (None, "")
    }
    out: list[dict[str, Any]] = []
    for row in table_rows[1:]:
        item: dict[str, Any] = {_TABLE_INDEX_KEY: row.get(0)}
        for col_idx, name in columns.items():
            item[name] = row.get(col_idx)
        out.append(item)
    return out


__all__ = [
    "GRADDFT_GROUND_ATOM_SYMBOLS",
    "GRADDFT_GROUND_TEST_ATOMS",
    "GRADDFT_XND_ATOM_ENERGY_COLUMN",
    "GradDFTGroundAtomSplit",
    "GradDFTGroundAtomRecord",
    "GradDFTGroundAtomTrainTestData",
    "build_graddft_ground_atom_datum",
    "build_graddft_ground_atom_molecule",
    "build_graddft_ground_atom_train_test_data",
    "graddft_ground_atom_split",
    "load_graddft_ground_atom_records",
    "neutral_atom_spin",
    "parse_graddft_test_train_ratio",
    "split_graddft_ground_atom_records",
]
