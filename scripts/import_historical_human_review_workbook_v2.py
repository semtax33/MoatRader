from __future__ import annotations

import argparse
import hashlib
import io
import json
import posixpath
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree as ET

from moatrader.expectations.historical_evidence import (
    AxisClassificationStatus,
    PairedAxisPacket,
    sha256_file,
)
from scripts.prepare_historical_balanced_retest_v2 import (
    CANDIDATE_CONTRACT as BALANCED_CONTRACT,
    CANDIDATE_SPLIT as BALANCED_SPLIT,
)
from scripts.prepare_historical_natural_retest_v2 import (
    RETEST_CONTRACT as NATURAL_CONTRACT,
    RETEST_SPLIT as NATURAL_SPLIT,
)


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CELL_REF_RE = re.compile(r"^(?P<column>[A-Z]+)(?P<row>[1-9][0-9]*)$")
EXCEL_ESCAPE_RE = re.compile(r"_x(?P<code>[0-9A-Fa-f]{4})_")
FORMULA_STRING_RE = re.compile(r'"((?:[^"]|"")*)"')
FORMULA_FUNCTION_RE = re.compile(r"(?<![A-Z0-9_.])([A-Z][A-Z0-9_.]*)\s*\(")
FORMULA_SHEET_RE = re.compile(r"(?:'((?:[^']|'')+)'|([A-Z_][A-Z0-9_ ]*))!")
MAX_XLSX_BYTES = 50 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 250 * 1024 * 1024
FORBIDDEN_PACKAGE_PARTS = (
    "customXml/",
    "xl/activeX/",
    "xl/comments",
    "xl/drawings/",
    "xl/externalLinks/",
    "xl/embeddings/",
    "xl/media/",
    "xl/oleObjects/",
    "xl/threadedComments/",
    "xl/webextensions/",
    "xl/connections.xml",
    "xl/vbaProject.bin",
)
ALLOWED_FORMULA_FUNCTIONS = {
    "AND",
    "COUNTA",
    "COUNTIF",
    "IF",
    "ISNUMBER",
    "OR",
    "SEARCH",
}
ALLOWED_FORMULA_STRINGS = {
    "",
    "AMBIGUOUS",
    "COMPLETE",
    "Demand Review",
    "ERROR",
    "FORMAT_ERROR",
    "FORMAT_OK_ONLY",
    "INSUFFICIENT_EVIDENCE",
    "NOT_READY",
    "OK",
    "PENDING",
    "PriceMix Review",
    "READY_FOR_HUMAN_IMPORT",
    "READY_FOR_HUMAN_MATERIALIZATION",
    "V2_BALANCED_RETEST_1_CANDIDATE_REVIEW",
    "V2_DIRECTIONAL_BALANCED_RETEST_1_CANDIDATE_POOL",
    "V2_NATURAL_FREQUENCY_LOCKED_RETEST_1",
    "V2_NATURAL_LOCKED_RETEST_1",
    "YES",
}


@dataclass(frozen=True)
class XlsxCell:
    value: str | int | float | bool | None = None
    formula: str | None = None


class WorkbookIntegrityError(ValueError):
    """The locked workbook itself, rather than a HUMAN decision, was altered."""


@dataclass(frozen=True)
class XlsxSheet:
    name: str
    state: str
    cells: Mapping[str, XlsxCell]


@dataclass(frozen=True)
class ReviewLayout:
    review_type: str
    manifest_name: str
    packet_name: str
    manifest_status: str
    manifest_packet_hash_key: str
    expected_sheet_names: tuple[str, ...]
    cover_reviewer_cell: str
    cover_attestation_cell: str
    cover_review_date_cell: str
    gold_split: str
    gold_contract: str


NATURAL_LAYOUT = ReviewLayout(
    review_type="natural-retest-1",
    manifest_name="natural-retest-preparation-manifest.json",
    packet_name="natural-retest-packets.jsonl",
    manifest_status="V2_NATURAL_RETEST_1_PREPARED_OUTCOME_BLIND",
    manifest_packet_hash_key="natural_retest_packet_sha256",
    expected_sheet_names=(
        "Cover",
        "Demand Review",
        "PriceMix Review",
        "Decision Export",
    ),
    cover_reviewer_cell="B14",
    cover_attestation_cell="B15",
    cover_review_date_cell="B16",
    gold_split=NATURAL_SPLIT,
    gold_contract=NATURAL_CONTRACT,
)
BALANCED_LAYOUT = ReviewLayout(
    review_type="balanced-retest-1",
    manifest_name="balanced-retest-preparation-manifest.json",
    packet_name="balanced-retest-candidate-packets.jsonl",
    manifest_status="V2_BALANCED_RETEST_1_CANDIDATES_PREPARED_OUTCOME_BLIND",
    manifest_packet_hash_key="balanced_retest_candidate_packet_sha256",
    expected_sheet_names=("Cover", "Review", "Decision Export"),
    cover_reviewer_cell="B17",
    cover_attestation_cell="B18",
    cover_review_date_cell="B19",
    gold_split=BALANCED_SPLIT,
    gold_contract=BALANCED_CONTRACT,
)
LAYOUTS = {
    NATURAL_LAYOUT.review_type: NATURAL_LAYOUT,
    BALANCED_LAYOUT.review_type: BALANCED_LAYOUT,
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"output file already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _decode_excel_escapes(value: str) -> str:
    return EXCEL_ESCAPE_RE.sub(
        lambda match: chr(int(match.group("code"), 16)), value
    )


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    return _decode_excel_escapes(str(value)).replace("\r\n", "\n").replace("\r", "\n")


def _column_number(column: str) -> int:
    result = 0
    for character in column:
        result = result * 26 + ord(character) - ord("A") + 1
    return result


def _cell_position(reference: str) -> tuple[int, int]:
    match = CELL_REF_RE.fullmatch(reference)
    if match is None:
        raise ValueError(f"invalid XLSX cell reference: {reference!r}")
    return int(match.group("row")), _column_number(match.group("column"))


def _xml_root(archive: zipfile.ZipFile, part: str) -> ET.Element:
    try:
        data = archive.read(part)
    except KeyError as exc:
        raise ValueError(f"XLSX package is missing required part: {part}") from exc
    try:
        return ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError(f"invalid XML in XLSX package part: {part}") from exc


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = _xml_root(archive, "xl/sharedStrings.xml")
    values: list[str] = []
    for item in root.findall(f"{{{MAIN_NS}}}si"):
        text = "".join(
            node.text or "" for node in item.iter(f"{{{MAIN_NS}}}t")
        )
        values.append(_decode_excel_escapes(text))
    return values


def _parse_number(value: str) -> int | float:
    try:
        integer = int(value)
    except ValueError:
        return float(value)
    return integer


def _cell_value(cell: ET.Element, shared: Sequence[str]) -> XlsxCell:
    cell_type = cell.attrib.get("t", "n")
    formula_element = cell.find(f"{{{MAIN_NS}}}f")
    formula = None if formula_element is None else (formula_element.text or "")
    if cell_type == "inlineStr":
        inline = cell.find(f"{{{MAIN_NS}}}is")
        text = "" if inline is None else "".join(
            node.text or "" for node in inline.iter(f"{{{MAIN_NS}}}t")
        )
        return XlsxCell(_decode_excel_escapes(text), formula)
    value_element = cell.find(f"{{{MAIN_NS}}}v")
    raw = None if value_element is None else value_element.text
    if raw is None:
        return XlsxCell(None, formula)
    if cell_type == "s":
        try:
            value = shared[int(raw)]
        except (IndexError, ValueError) as exc:
            raise ValueError("XLSX shared-string index is invalid") from exc
        return XlsxCell(value, formula)
    if cell_type in {"str", "e"}:
        return XlsxCell(_decode_excel_escapes(raw), formula)
    if cell_type == "b":
        return XlsxCell(raw == "1", formula)
    try:
        return XlsxCell(_parse_number(raw), formula)
    except ValueError as exc:
        raise ValueError(f"invalid numeric XLSX cell value: {raw!r}") from exc


def _worksheet_cells(
    archive: zipfile.ZipFile, part: str, shared: Sequence[str]
) -> dict[str, XlsxCell]:
    root = _xml_root(archive, part)
    cells: dict[str, XlsxCell] = {}
    for cell in root.iter(f"{{{MAIN_NS}}}c"):
        reference = cell.attrib.get("r")
        if not reference or CELL_REF_RE.fullmatch(reference) is None:
            raise ValueError(f"worksheet contains an invalid cell reference: {part}")
        if reference in cells:
            raise ValueError(f"worksheet contains a duplicate cell: {part}:{reference}")
        cells[reference] = _cell_value(cell, shared)
    return cells


def _read_xlsx(workbook_bytes: bytes) -> dict[str, XlsxSheet]:
    if len(workbook_bytes) > MAX_XLSX_BYTES:
        raise ValueError("review workbook exceeds the 50 MiB XLSX input limit")
    try:
        archive = zipfile.ZipFile(io.BytesIO(workbook_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError("review workbook is not a valid XLSX ZIP package") from exc
    with archive:
        names = archive.namelist()
        lowered = {name.casefold() for name in names}
        if len(lowered) != len(names):
            raise ValueError("XLSX package contains case-colliding duplicate parts")
        total_size = sum(item.file_size for item in archive.infolist())
        if total_size > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("review workbook exceeds the uncompressed XLSX size limit")
        for name in names:
            normalized = name.replace("\\", "/")
            folded = normalized.casefold()
            if any(folded.startswith(part.casefold()) for part in FORBIDDEN_PACKAGE_PARTS):
                raise ValueError(f"review workbook contains a forbidden active part: {name}")
            if folded.endswith(".rels"):
                relationships_part = _xml_root(archive, name)
                for relationship in relationships_part.findall(
                    f"{{{PACKAGE_REL_NS}}}Relationship"
                ):
                    if relationship.attrib.get("TargetMode", "").casefold() == "external":
                        raise ValueError(
                            "review workbook contains an external relationship: "
                            f"{name}"
                        )

        shared = _shared_strings(archive)
        workbook_root = _xml_root(archive, "xl/workbook.xml")
        workbook_properties = workbook_root.find(f"{{{MAIN_NS}}}workbookPr")
        if workbook_properties is not None and workbook_properties.attrib.get(
            "date1904", ""
        ).casefold() in {"1", "true"}:
            raise ValueError("review workbook cannot use the Excel 1904 date system")
        defined_names = workbook_root.find(f"{{{MAIN_NS}}}definedNames")
        if defined_names is not None and any(
            (item.text or "").strip()
            for item in defined_names.findall(f"{{{MAIN_NS}}}definedName")
        ):
            raise ValueError("review workbook cannot contain defined names")
        relationships_root = _xml_root(archive, "xl/_rels/workbook.xml.rels")
        relationships = {
            item.attrib["Id"]: item.attrib["Target"]
            for item in relationships_root.findall(
                f"{{{PACKAGE_REL_NS}}}Relationship"
            )
            if "Id" in item.attrib and "Target" in item.attrib
        }
        sheets: dict[str, XlsxSheet] = {}
        sheets_element = workbook_root.find(f"{{{MAIN_NS}}}sheets")
        if sheets_element is None:
            raise ValueError("XLSX workbook contains no worksheets")
        for item in sheets_element.findall(f"{{{MAIN_NS}}}sheet"):
            name = item.attrib.get("name", "")
            relationship_id = item.attrib.get(f"{{{REL_NS}}}id", "")
            if not name or relationship_id not in relationships:
                raise ValueError("XLSX worksheet relationship is invalid")
            target = relationships[relationship_id].replace("\\", "/")
            if target.startswith("/"):
                part = target.lstrip("/")
            else:
                part = posixpath.normpath(posixpath.join("xl", target))
            if not part.startswith("xl/worksheets/") or ".." in part.split("/"):
                raise ValueError(f"unexpected worksheet package target: {target}")
            if name in sheets:
                raise ValueError(f"duplicate worksheet name: {name}")
            sheets[name] = XlsxSheet(
                name=name,
                state=item.attrib.get("state", "visible"),
                cells=_worksheet_cells(archive, part, shared),
            )
        return sheets


def _cell(sheet: XlsxSheet, reference: str) -> XlsxCell:
    return sheet.cells.get(reference, XlsxCell())


def _value(sheet: XlsxSheet, reference: str) -> str | int | float | bool | None:
    return _cell(sheet, reference).value


def _text(sheet: XlsxSheet, reference: str) -> str:
    return _normalize_text(_value(sheet, reference))


def _is_blank(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _require_constant(sheet: XlsxSheet, reference: str, description: str) -> None:
    if _cell(sheet, reference).formula is not None:
        raise WorkbookIntegrityError(
            f"{description} cannot be an Excel formula: {sheet.name}!{reference}"
        )


def _require_text(
    sheet: XlsxSheet, reference: str, expected: str, description: str
) -> None:
    _require_constant(sheet, reference, description)
    if _text(sheet, reference) != expected:
        raise ValueError(
            f"{description} changed: {sheet.name}!{reference}; "
            f"expected={expected!r} actual={_text(sheet, reference)!r}"
        )


def _validate_sheet_extent(sheet: XlsxSheet, *, max_row: int, max_column: int) -> None:
    for reference, cell in sheet.cells.items():
        row, column = _cell_position(reference)
        if row <= max_row and column <= max_column:
            continue
        if not _is_blank(cell.value) or cell.formula is not None:
            raise ValueError(
                f"unexpected data outside the locked workbook area: "
                f"{sheet.name}!{reference}"
            )


def _render_excerpts(packet_excerpts: Iterable[Any]) -> str:
    return "\n\n".join(
        f"[{number}] {excerpt.source_id}\n{excerpt.text}"
        for number, excerpt in enumerate(packet_excerpts, start=1)
    )


def _read_packets(path: Path) -> list[PairedAxisPacket]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        packets = [
            PairedAxisPacket.model_validate_json(line)
            for line in handle
            if line.strip()
        ]
    if not packets:
        raise ValueError(f"candidate packet file is empty: {path}")
    if len({packet.packet_id for packet in packets}) != len(packets):
        raise ValueError("candidate packet IDs must be unique")
    return packets


def _layout_inputs(
    *, candidate_build: Path, layout: ReviewLayout
) -> tuple[dict[str, Any], Path, list[PairedAxisPacket]]:
    manifest_path = candidate_build / layout.manifest_name
    packet_path = candidate_build / layout.packet_name
    manifest = _read_json(manifest_path)
    if manifest.get("status") != layout.manifest_status:
        raise ValueError(f"candidate manifest status is invalid for {layout.review_type}")
    for key in ("outcome_vault_opened", "return_data_opened", "value_data_opened"):
        if manifest.get(key) is not False:
            raise ValueError(f"candidate manifest opened forbidden downstream data: {key}")
    if manifest.get("per_pbr_role") != "NOT_USED":
        raise ValueError("candidate manifest used PER/PBR before the Full Index seal")
    packet_hash = sha256_file(packet_path)
    if manifest.get(layout.manifest_packet_hash_key) != packet_hash:
        raise ValueError("candidate packet file changed after preparation")
    packets = _read_packets(packet_path)
    return manifest, manifest_path, packets


def _validate_workbook_package(
    sheets: Mapping[str, XlsxSheet], layout: ReviewLayout
) -> None:
    actual = tuple(sheets)
    if actual != layout.expected_sheet_names:
        raise ValueError(
            f"review workbook sheets changed; expected={layout.expected_sheet_names} "
            f"actual={actual}"
        )
    hidden = [sheet.name for sheet in sheets.values() if sheet.state != "visible"]
    if hidden:
        raise ValueError(f"review workbook cannot contain hidden sheets: {hidden}")


def _formula_locations(
    *, layout: ReviewLayout, packets: Sequence[PairedAxisPacket]
) -> set[tuple[str, str]]:
    expected: set[tuple[str, str]] = set()
    if layout is NATURAL_LAYOUT:
        expected.update(("Cover", f"B{row}") for row in (*range(29, 36), 37))
        axis_rows = {
            "Demand Review": sum(
                packet.axis.value == "DEMAND" for packet in packets
            ),
            "PriceMix Review": sum(
                packet.axis.value == "PRICE_MIX" for packet in packets
            ),
        }
        for sheet_name, count in axis_rows.items():
            for row in range(6, 6 + count):
                expected.update(
                    (sheet_name, f"{column}{row}") for column in ("F", "M", "N")
                )
        for row in range(2, 2 + len(packets)):
            expected.update(
                ("Decision Export", f"{column}{row}")
                for column in "ABCDEFGHIJKLMNOPQ"
            )
    else:
        expected.update(("Cover", f"B{row}") for row in range(31, 40))
        for row in range(6, 6 + len(packets)):
            expected.update(
                ("Review", f"{column}{row}") for column in ("G", "N", "O")
            )
        for row in range(2, 2 + len(packets)):
            expected.update(
                ("Decision Export", f"{column}{row}")
                for column in "ABCDEFGHIJKLMN"
            )
    return expected


def _validate_formula_contract(
    *,
    sheets: Mapping[str, XlsxSheet],
    layout: ReviewLayout,
    packets: Sequence[PairedAxisPacket],
) -> None:
    expected = _formula_locations(layout=layout, packets=packets)
    actual = {
        (sheet.name, reference)
        for sheet in sheets.values()
        for reference, cell in sheet.cells.items()
        if cell.formula is not None
    }
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise WorkbookIntegrityError(
            "locked workbook formula locations changed; "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    allowed_sheets = {name.casefold() for name in layout.expected_sheet_names}
    for sheet_name, reference in sorted(actual):
        formula = sheets[sheet_name].cells[reference].formula or ""
        if not formula:
            # Excel can serialize shared-formula continuation cells with an empty
            # ``<f>`` node. The location contract still prevents injection.
            continue
        if any(token in formula for token in ("[", "]", "|", ";", "\x00", "\n")):
            raise WorkbookIntegrityError(
                f"unsafe formula syntax: {sheet_name}!{reference}"
            )
        string_literals = [
            item.replace('""', '"') for item in FORMULA_STRING_RE.findall(formula)
        ]
        unexpected_strings = sorted(set(string_literals) - ALLOWED_FORMULA_STRINGS)
        if unexpected_strings:
            raise WorkbookIntegrityError(
                f"unexpected formula text: {sheet_name}!{reference}: "
                f"{unexpected_strings[:3]}"
            )
        formula_without_strings_original = FORMULA_STRING_RE.sub('""', formula)
        formula_without_strings = formula_without_strings_original.upper()
        functions = set(FORMULA_FUNCTION_RE.findall(formula_without_strings))
        unexpected_functions = sorted(functions - ALLOWED_FORMULA_FUNCTIONS)
        if unexpected_functions:
            raise WorkbookIntegrityError(
                f"unsafe formula function: {sheet_name}!{reference}: "
                f"{unexpected_functions[:3]}"
            )
        referenced_sheets = {
            (quoted or unquoted).replace("''", "'")
            for quoted, unquoted in FORMULA_SHEET_RE.findall(
                formula_without_strings_original
            )
        }
        if not {name.casefold() for name in referenced_sheets} <= allowed_sheets:
            raise WorkbookIntegrityError(
                f"formula references an unexpected sheet: {sheet_name}!{reference}"
            )


def _validate_natural_fixed_content(
    *,
    sheets: Mapping[str, XlsxSheet],
    packets: Sequence[PairedAxisPacket],
    manifest_path: Path,
    packet_path: Path,
) -> list[tuple[XlsxSheet, int, PairedAxisPacket]]:
    cover = sheets["Cover"]
    _validate_sheet_extent(cover, max_row=38, max_column=8)
    _require_text(
        cover,
        "A1",
        "MoatRader V2 Natural LOCKED Retest 1 — Independent HUMAN Review",
        "Natural workbook title",
    )
    _require_text(
        cover,
        "B7",
        sha256_file(manifest_path),
        "Natural candidate manifest SHA-256",
    )
    _require_text(
        cover,
        "B8",
        sha256_file(packet_path),
        "Natural packet SHA-256",
    )
    _require_text(
        cover,
        "B11",
        "FALSE / FALSE / FALSE · PER/PBR role = NOT_USED",
        "Natural downstream-closure declaration",
    )

    packets_by_axis = {
        "DEMAND": [packet for packet in packets if packet.axis.value == "DEMAND"],
        "PRICE_MIX": [
            packet for packet in packets if packet.axis.value == "PRICE_MIX"
        ],
    }
    locations: list[tuple[XlsxSheet, int, PairedAxisPacket]] = []
    for sheet_name, axis in (
        ("Demand Review", "DEMAND"),
        ("PriceMix Review", "PRICE_MIX"),
    ):
        sheet = sheets[sheet_name]
        expected_packets = packets_by_axis[axis]
        max_row = 5 + len(expected_packets)
        _validate_sheet_extent(sheet, max_row=max_row, max_column=16)
        expected_headers = (
            "packet_id",
            "axis",
            "human_status",
            "previous_state",
            "current_state",
            "delta",
            "previous_excerpts",
            "previous_anchor",
            "current_excerpts",
            "current_anchor",
            "review_notes",
            "contract_self_check",
            "row_check",
            "reviewer_name",
            "gold_contract_version",
            "gold_split",
        )
        for column, expected in zip("ABCDEFGHIJKLMNOP", expected_headers, strict=True):
            _require_text(sheet, f"{column}5", expected, f"{sheet_name} header")
        for offset, packet in enumerate(expected_packets, start=6):
            _require_text(sheet, f"A{offset}", packet.packet_id, "packet ID")
            _require_text(sheet, f"B{offset}", axis, "semantic axis")
            _require_text(
                sheet,
                f"G{offset}",
                _render_excerpts(packet.previous_excerpts),
                "previous candidate excerpts",
            )
            _require_text(
                sheet,
                f"I{offset}",
                _render_excerpts(packet.current_excerpts),
                "current candidate excerpts",
            )
            _require_text(
                sheet,
                f"O{offset}",
                NATURAL_LAYOUT.gold_contract,
                "Natural gold contract",
            )
            _require_text(
                sheet,
                f"P{offset}",
                NATURAL_LAYOUT.gold_split,
                "Natural gold split",
            )
            locations.append((sheet, offset, packet))

    export = sheets["Decision Export"]
    _validate_sheet_extent(export, max_row=1 + len(packets), max_column=17)
    expected_export_headers = (
        "packet_id",
        "axis",
        "human_status",
        "previous_state",
        "current_state",
        "delta",
        "previous_anchor",
        "current_anchor",
        "review_notes",
        "contract_self_check",
        "row_check",
        "gold_split",
        "gold_contract_version",
        "reviewer",
        "attestation",
        "review_date",
        "source_sheet",
    )
    for column, expected in zip(
        "ABCDEFGHIJKLMNOPQ", expected_export_headers, strict=True
    ):
        _require_text(export, f"{column}1", expected, "Natural export header")
    for row in range(2, 2 + len(packets)):
        for column in "ABCDEFGHIJKLMNOPQ":
            if _cell(export, f"{column}{row}").formula is None:
                raise ValueError(
                    f"Natural Decision Export formula was replaced: "
                    f"Decision Export!{column}{row}"
                )
    return locations


def _validate_balanced_fixed_content(
    *,
    sheets: Mapping[str, XlsxSheet],
    packets: Sequence[PairedAxisPacket],
    packet_path: Path,
) -> list[tuple[XlsxSheet, int, PairedAxisPacket]]:
    cover = sheets["Cover"]
    _validate_sheet_extent(cover, max_row=39, max_column=3)
    _require_text(
        cover,
        "A1",
        "MoatRader V2 Balanced LOCKED — Independent Retest 1 HUMAN Review",
        "Balanced workbook title",
    )
    if _value(cover, "B6") != len(packets):
        raise ValueError("Balanced candidate count changed in the workbook")
    _require_text(
        cover,
        "B7",
        sha256_file(packet_path),
        "Balanced packet SHA-256",
    )
    for reference, expected, description in (
        ("B8", "FALSE", "parser classification selection declaration"),
        ("B9", "FALSE", "post-test disagreement selection declaration"),
        ("B10", "FALSE", "selection-hint exposure declaration"),
        ("B11", "TRUE", "first Balanced consumption declaration"),
        ("B12", "FALSE / FALSE / FALSE", "downstream-closure declaration"),
        ("B13", "V2_DIRECTIONAL_BALANCED_LOCKED_RETEST_1", "retest contract"),
    ):
        _require_text(cover, reference, expected, description)

    review = sheets["Review"]
    _validate_sheet_extent(review, max_row=5 + len(packets), max_column=15)
    expected_headers = (
        "candidate_no",
        "packet_id",
        "axis",
        "human_status",
        "previous_state",
        "current_state",
        "delta",
        "previous_excerpts",
        "previous_anchor",
        "current_excerpts",
        "current_anchor",
        "review_notes",
        "contract_self_check",
        "format_check_only",
        "reviewer_name",
    )
    for column, expected in zip("ABCDEFGHIJKLMNO", expected_headers, strict=True):
        _require_text(review, f"{column}5", expected, "Balanced review header")
    locations: list[tuple[XlsxSheet, int, PairedAxisPacket]] = []
    for candidate_number, packet in enumerate(packets, start=1):
        row = candidate_number + 5
        if _value(review, f"A{row}") != candidate_number:
            raise ValueError(f"Balanced candidate number changed: Review!A{row}")
        _require_text(review, f"B{row}", packet.packet_id, "packet ID")
        _require_text(review, f"C{row}", packet.axis.value, "semantic axis")
        _require_text(
            review,
            f"H{row}",
            _render_excerpts(packet.previous_excerpts),
            "previous candidate excerpts",
        )
        _require_text(
            review,
            f"J{row}",
            _render_excerpts(packet.current_excerpts),
            "current candidate excerpts",
        )
        locations.append((review, row, packet))

    export = sheets["Decision Export"]
    _validate_sheet_extent(export, max_row=1 + len(packets), max_column=14)
    expected_export_headers = (
        "packet_id",
        "axis",
        "human_status",
        "previous_state",
        "current_state",
        "delta",
        "previous_anchor",
        "current_anchor",
        "review_notes",
        "contract_self_check",
        "format_check_only",
        "gold_split",
        "gold_contract_version",
        "reviewer",
    )
    for column, expected in zip(
        "ABCDEFGHIJKLMN", expected_export_headers, strict=True
    ):
        _require_text(export, f"{column}1", expected, "Balanced export header")
    for row in range(2, 2 + len(packets)):
        for column in "ABCDEFGHIJKLMN":
            if _cell(export, f"{column}{row}").formula is None:
                raise ValueError(
                    f"Balanced Decision Export formula was replaced: "
                    f"Decision Export!{column}{row}"
                )
    return locations


def _review_date(value: object) -> str:
    if isinstance(value, bool) or value is None:
        raise ValueError("review date is missing")
    if isinstance(value, (int, float)):
        if value <= 0:
            raise ValueError("review date Excel serial must be positive")
        return (date(1899, 12, 30) + timedelta(days=int(value))).isoformat()
    text = str(value).strip()
    if not text:
        raise ValueError("review date is missing")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).date().isoformat()
    except ValueError:
        try:
            return date.fromisoformat(text).isoformat()
        except ValueError as exc:
            raise ValueError("review date must be a valid ISO or Excel date") from exc


def _state(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be -1, 0, or 1")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be -1, 0, or 1") from exc
    if converted not in (-1, 0, 1) or str(value).strip() not in {
        "-1",
        "0",
        "1",
        "-1.0",
        "0.0",
        "1.0",
    }:
        raise ValueError(f"{field} must be -1, 0, or 1")
    return converted


def _human_columns(
    layout: ReviewLayout, row: int
) -> dict[str, str]:
    if layout is NATURAL_LAYOUT:
        return {
            "status": f"C{row}",
            "previous_state": f"D{row}",
            "current_state": f"E{row}",
            "previous_anchor": f"H{row}",
            "current_anchor": f"J{row}",
            "review_notes": f"K{row}",
            "contract_self_check": f"L{row}",
        }
    return {
        "status": f"D{row}",
        "previous_state": f"E{row}",
        "current_state": f"F{row}",
        "previous_anchor": f"I{row}",
        "current_anchor": f"K{row}",
        "review_notes": f"L{row}",
        "contract_self_check": f"M{row}",
    }


def _decision_from_row(
    *,
    sheet: XlsxSheet,
    row: int,
    packet: PairedAxisPacket,
    layout: ReviewLayout,
) -> dict[str, Any] | None:
    columns = _human_columns(layout, row)
    for field, reference in columns.items():
        _require_constant(sheet, reference, f"HUMAN input {field}")
    values = {field: _value(sheet, reference) for field, reference in columns.items()}
    if all(_is_blank(value) for value in values.values()):
        return None
    status_text = str(values["status"] or "").strip()
    if not status_text:
        raise ValueError("human_status is missing while other HUMAN inputs are populated")
    try:
        status = AxisClassificationStatus(status_text)
    except ValueError as exc:
        raise ValueError(f"invalid human_status: {status_text!r}") from exc
    notes = _normalize_text(values["review_notes"]).strip()
    if not notes:
        raise ValueError("review_notes are required")
    if values["contract_self_check"] != "YES":
        raise ValueError("contract_self_check must be exactly YES")
    decision: dict[str, Any] = {
        "packet_id": packet.packet_id,
        "axis": packet.axis.value,
        "status": status.value,
        "review_notes": notes,
        "contract_self_check": "YES",
    }
    evidence_fields = (
        "previous_state",
        "current_state",
        "previous_anchor",
        "current_anchor",
    )
    if status == AxisClassificationStatus.COMPLETE:
        decision["previous_state"] = _state(
            values["previous_state"],
            field="previous_state",
        )
        decision["current_state"] = _state(
            values["current_state"],
            field="current_state",
        )
        for field in ("previous_anchor", "current_anchor"):
            anchor = _normalize_text(values[field])
            if not anchor:
                raise ValueError(f"{field} is required for COMPLETE")
            decision[field] = anchor
        previous_text = "\n".join(item.text for item in packet.previous_excerpts)
        current_text = "\n".join(item.text for item in packet.current_excerpts)
        if decision["previous_anchor"] not in previous_text:
            raise ValueError("previous_anchor is not an exact candidate excerpt substring")
        if decision["current_anchor"] not in current_text:
            raise ValueError("current_anchor is not an exact candidate excerpt substring")
    elif any(not _is_blank(values[field]) for field in evidence_fields):
        raise ValueError(
            "non-COMPLETE decision must leave states and anchors blank"
        )
    return decision


def audit_human_review_workbook(
    *, workbook: Path, candidate_build: Path, review_type: str
) -> dict[str, Any]:
    try:
        layout = LAYOUTS[review_type]
    except KeyError as exc:
        raise ValueError(f"unsupported review type: {review_type}") from exc
    if workbook.suffix.casefold() != ".xlsx" or not workbook.is_file():
        raise ValueError(f"review workbook must be an existing .xlsx file: {workbook}")
    workbook_bytes = workbook.read_bytes()
    workbook_hash = hashlib.sha256(workbook_bytes).hexdigest()
    sheets = _read_xlsx(workbook_bytes)
    if sha256_file(workbook) != workbook_hash:
        raise ValueError("review workbook changed while it was being read")

    manifest, manifest_path, packets = _layout_inputs(
        candidate_build=candidate_build, layout=layout
    )
    packet_path = candidate_build / layout.packet_name
    _validate_workbook_package(sheets, layout)
    _validate_formula_contract(sheets=sheets, layout=layout, packets=packets)
    if layout is NATURAL_LAYOUT:
        locations = _validate_natural_fixed_content(
            sheets=sheets,
            packets=packets,
            manifest_path=manifest_path,
            packet_path=packet_path,
        )
    else:
        locations = _validate_balanced_fixed_content(
            sheets=sheets,
            packets=packets,
            packet_path=packet_path,
        )

    cover = sheets["Cover"]
    for reference, description in (
        (layout.cover_reviewer_cell, "HUMAN reviewer name"),
        (layout.cover_attestation_cell, "HUMAN attestation"),
        (layout.cover_review_date_cell, "review date"),
    ):
        _require_constant(cover, reference, description)
    reviewer_name = _text(cover, layout.cover_reviewer_cell).strip()
    attestation = _text(cover, layout.cover_attestation_cell).strip()
    metadata_issues: list[str] = []
    if not reviewer_name:
        metadata_issues.append("HUMAN reviewer name is missing")
    if attestation != "YES":
        metadata_issues.append("HUMAN attestation must be exactly YES")
    try:
        review_date = _review_date(_value(cover, layout.cover_review_date_cell))
    except ValueError as exc:
        review_date = ""
        metadata_issues.append(str(exc))

    decisions: list[dict[str, Any]] = []
    pending_packet_ids: list[str] = []
    row_errors: list[dict[str, str]] = []
    for sheet, row, packet in locations:
        try:
            decision = _decision_from_row(
                sheet=sheet,
                row=row,
                packet=packet,
                layout=layout,
            )
        except WorkbookIntegrityError:
            raise
        except ValueError as exc:
            row_errors.append(
                {
                    "packet_id": packet.packet_id,
                    "sheet": sheet.name,
                    "row": str(row),
                    "error": str(exc),
                }
            )
            continue
        if decision is None:
            pending_packet_ids.append(packet.packet_id)
        else:
            decisions.append(decision)

    status_counts = Counter(str(row["status"]) for row in decisions)
    ready = not metadata_issues and not pending_packet_ids and not row_errors
    result = {
        "schema_version": "moatrader-v2-human-review-workbook-audit/1",
        "status": (
            "READY_FOR_HUMAN_IMPORT"
            if ready
            else "NOT_READY_FOR_HUMAN_IMPORT"
        ),
        "review_type": layout.review_type,
        "reviewer": "HUMAN",
        "human_reviewer_name": reviewer_name,
        "attestation": attestation,
        "review_date": review_date,
        "candidate_count": len(packets),
        "reviewed_count": len(decisions),
        "pending_count": len(pending_packet_ids),
        "row_error_count": len(row_errors),
        "contract_self_check_yes_count": sum(
            row.get("contract_self_check") == "YES" for row in decisions
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "metadata_issues": metadata_issues,
        "pending_packet_ids_preview": pending_packet_ids[:10],
        "row_errors_preview": row_errors[:20],
        "candidate_manifest_status": manifest["status"],
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "candidate_packet_sha256": sha256_file(packet_path),
        "workbook_sha256": workbook_hash,
        "workbook_read_only_verified": True,
        "candidate_excerpts_verified": True,
        "decision_export_formulas_verified": True,
        "formula_row_checks_trusted": False,
        "human_input_formulas_rejected": True,
        "unexpected_or_hidden_sheets_rejected": True,
        "external_links_macros_and_embedded_objects_rejected": True,
        "gold_split": layout.gold_split,
        "gold_contract_version": layout.gold_contract,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "per_pbr_role": "NOT_USED",
        "decisions": decisions,
    }
    return result


def import_human_review_workbook(
    *, workbook: Path, candidate_build: Path, review_type: str, output: Path
) -> dict[str, Any]:
    audit = audit_human_review_workbook(
        workbook=workbook,
        candidate_build=candidate_build,
        review_type=review_type,
    )
    if audit["status"] != "READY_FOR_HUMAN_IMPORT":
        raise ValueError(
            "HUMAN review workbook is not ready; "
            f"reviewed={audit['reviewed_count']} pending={audit['pending_count']} "
            f"row_errors={audit['row_error_count']} "
            f"metadata_issues={audit['metadata_issues']}"
        )
    result = dict(audit)
    result["schema_version"] = "moatrader-v2-human-review-workbook-import/1"
    result["status"] = "HUMAN_REVIEW_DECISIONS_IMPORTED_OUTCOME_BLIND"
    result["decision_count"] = len(result["decisions"])
    _write_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit or import a locked V2 HUMAN review XLSX without trusting its "
            "formula-derived OK cells."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "import"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--workbook", type=Path, required=True)
        command_parser.add_argument("--candidate-build", type=Path, required=True)
        command_parser.add_argument(
            "--review-type", choices=sorted(LAYOUTS), required=True
        )
        if command == "import":
            command_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        result = audit_human_review_workbook(
            workbook=args.workbook,
            candidate_build=args.candidate_build,
            review_type=args.review_type,
        )
    else:
        result = import_human_review_workbook(
            workbook=args.workbook,
            candidate_build=args.candidate_build,
            review_type=args.review_type,
            output=args.output,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
