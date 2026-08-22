from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.etree import ElementTree as ET

import pytest

from moatrader.expectations.future_eri import OperatingEvidenceAxis
from moatrader.expectations.historical_evidence import (
    BlindedExcerpt,
    PairedAxisPacket,
    sha256_file,
)
from scripts.import_historical_human_review_workbook_v2 import (
    BALANCED_LAYOUT,
    NATURAL_LAYOUT,
    audit_human_review_workbook,
    import_human_review_workbook,
)
from scripts.materialize_historical_natural_retest_v2 import (
    materialize_natural_retest_human_gold,
)


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"


@dataclass(frozen=True)
class Formula:
    text: str = "1+1"
    cached: object | None = None


def _packet(
    packet_id: str,
    axis: OperatingEvidenceAxis,
    previous_text: str,
    current_text: str,
    source_suffix: str,
) -> PairedAxisPacket:
    return PairedAxisPacket(
        packet_id=packet_id,
        axis=axis,
        previous_excerpts=[
            BlindedExcerpt(
                source_id=f"SRC_{source_suffix}0",
                text=previous_text,
            )
        ],
        current_excerpts=[
            BlindedExcerpt(
                source_id=f"SRC_{source_suffix}1",
                text=current_text,
            )
        ],
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _candidate_build(
    root: Path, *, review_type: str, packets: Sequence[PairedAxisPacket]
) -> Path:
    root.mkdir(parents=True)
    if review_type == NATURAL_LAYOUT.review_type:
        packet_name = NATURAL_LAYOUT.packet_name
        manifest_name = NATURAL_LAYOUT.manifest_name
        template_name = "natural-retest-human-gold-template.csv"
        packet_hash_key = NATURAL_LAYOUT.manifest_packet_hash_key
        template_hash_key = "natural_retest_human_gold_template_sha256"
        status = NATURAL_LAYOUT.manifest_status
    else:
        packet_name = BALANCED_LAYOUT.packet_name
        manifest_name = BALANCED_LAYOUT.manifest_name
        template_name = "balanced-retest-human-gold-template.csv"
        packet_hash_key = BALANCED_LAYOUT.manifest_packet_hash_key
        template_hash_key = "balanced_retest_human_gold_template_sha256"
        status = BALANCED_LAYOUT.manifest_status
    packet_path = root / packet_name
    packet_path.write_text(
        "\n".join(packet.model_dump_json() for packet in packets) + "\n",
        encoding="utf-8",
    )
    template_path = root / template_name
    template_path.write_text("packet_id,axis\n", encoding="utf-8")
    payload = {
        "status": status,
        packet_hash_key: sha256_file(packet_path),
        template_hash_key: sha256_file(template_path),
        "selection_used_parser_classifications": False,
        "selection_used_post_test_disagreement_rows": False,
        "first_natural_test_remains_consumed": True,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "per_pbr_role": "NOT_USED",
    }
    _write_json(root / manifest_name, payload)
    return root


def _xml_bytes(element: ET.Element) -> bytes:
    return ET.tostring(element, encoding="utf-8", xml_declaration=True)


def _worksheet_xml(cells: Mapping[str, object]) -> bytes:
    worksheet = ET.Element(f"{{{MAIN_NS}}}worksheet")
    sheet_data = ET.SubElement(worksheet, f"{{{MAIN_NS}}}sheetData")
    rows: dict[int, list[tuple[str, object]]] = {}
    for reference, value in cells.items():
        row = int("".join(character for character in reference if character.isdigit()))
        rows.setdefault(row, []).append((reference, value))
    for row_number in sorted(rows):
        row = ET.SubElement(sheet_data, f"{{{MAIN_NS}}}row", {"r": str(row_number)})
        for reference, value in sorted(rows[row_number]):
            cell_attributes = {"r": reference}
            if isinstance(value, str):
                cell_attributes["t"] = "inlineStr"
            elif isinstance(value, Formula) and isinstance(value.cached, str):
                cell_attributes["t"] = "str"
            cell = ET.SubElement(row, f"{{{MAIN_NS}}}c", cell_attributes)
            if isinstance(value, Formula):
                ET.SubElement(cell, f"{{{MAIN_NS}}}f").text = value.text
                if value.cached is not None:
                    ET.SubElement(cell, f"{{{MAIN_NS}}}v").text = str(value.cached)
            elif isinstance(value, str):
                inline = ET.SubElement(cell, f"{{{MAIN_NS}}}is")
                text = ET.SubElement(inline, f"{{{MAIN_NS}}}t")
                text.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                text.text = value
            elif value is not None:
                ET.SubElement(cell, f"{{{MAIN_NS}}}v").text = str(value)
    return _xml_bytes(worksheet)


def _write_xlsx(
    path: Path,
    sheets: Sequence[tuple[str, Mapping[str, object]]],
    *,
    external_relationship: bool = False,
) -> None:
    workbook = ET.Element(f"{{{MAIN_NS}}}workbook")
    workbook_sheets = ET.SubElement(workbook, f"{{{MAIN_NS}}}sheets")
    for index, (name, _) in enumerate(sheets, start=1):
        ET.SubElement(
            workbook_sheets,
            f"{{{MAIN_NS}}}sheet",
            {
                "name": name,
                "sheetId": str(index),
                f"{{{REL_NS}}}id": f"rId{index}",
            },
        )

    relationships = ET.Element(f"{{{PACKAGE_REL_NS}}}Relationships")
    for index in range(1, len(sheets) + 1):
        ET.SubElement(
            relationships,
            f"{{{PACKAGE_REL_NS}}}Relationship",
            {
                "Id": f"rId{index}",
                "Type": (
                    "http://schemas.openxmlformats.org/officeDocument/2006/"
                    "relationships/worksheet"
                ),
                "Target": f"worksheets/sheet{index}.xml",
            },
        )

    content_types = ET.Element(f"{{{CONTENT_NS}}}Types")
    ET.SubElement(
        content_types,
        f"{{{CONTENT_NS}}}Default",
        {"Extension": "rels", "ContentType": "application/vnd.openxmlformats-package.relationships+xml"},
    )
    ET.SubElement(
        content_types,
        f"{{{CONTENT_NS}}}Default",
        {"Extension": "xml", "ContentType": "application/xml"},
    )
    ET.SubElement(
        content_types,
        f"{{{CONTENT_NS}}}Override",
        {
            "PartName": "/xl/workbook.xml",
            "ContentType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml",
        },
    )
    for index in range(1, len(sheets) + 1):
        ET.SubElement(
            content_types,
            f"{{{CONTENT_NS}}}Override",
            {
                "PartName": f"/xl/worksheets/sheet{index}.xml",
                "ContentType": "application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml",
            },
        )

    root_relationships = ET.Element(f"{{{PACKAGE_REL_NS}}}Relationships")
    ET.SubElement(
        root_relationships,
        f"{{{PACKAGE_REL_NS}}}Relationship",
        {
            "Id": "rId1",
            "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument",
            "Target": "xl/workbook.xml",
        },
    )
    if external_relationship:
        ET.SubElement(
            root_relationships,
            f"{{{PACKAGE_REL_NS}}}Relationship",
            {
                "Id": "rId2",
                "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
                "Target": "https://example.com/model-hint",
                "TargetMode": "External",
            },
        )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _xml_bytes(content_types))
        archive.writestr("_rels/.rels", _xml_bytes(root_relationships))
        archive.writestr("xl/workbook.xml", _xml_bytes(workbook))
        archive.writestr("xl/_rels/workbook.xml.rels", _xml_bytes(relationships))
        for index, (_, cells) in enumerate(sheets, start=1):
            archive.writestr(
                f"xl/worksheets/sheet{index}.xml", _worksheet_xml(cells)
            )


def _render_excerpts(packet: PairedAxisPacket, field: str) -> str:
    excerpts = getattr(packet, field)
    return "\n\n".join(
        f"[{index}] {excerpt.source_id}\n{excerpt.text}"
        for index, excerpt in enumerate(excerpts, start=1)
    )


def _natural_workbook_cells(
    candidate_build: Path,
    packets: Sequence[PairedAxisPacket],
    *,
    completed: bool = False,
    status_formula: bool = False,
    tamper_excerpt: bool = False,
) -> list[tuple[str, dict[str, object]]]:
    manifest_path = candidate_build / NATURAL_LAYOUT.manifest_name
    packet_path = candidate_build / NATURAL_LAYOUT.packet_name
    cover: dict[str, object] = {
        "A1": "MoatRader V2 Natural LOCKED Retest 1 — Independent HUMAN Review",
        "B7": sha256_file(manifest_path),
        "B8": sha256_file(packet_path),
        "B11": "FALSE / FALSE / FALSE · PER/PBR role = NOT_USED",
    }
    cover.update({f"B{row}": Formula() for row in (*range(29, 36), 37)})
    if completed:
        cover.update({"B14": "Human Reviewer", "B15": "YES", "B16": "2026-08-22"})

    headers = (
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
    sheets: dict[str, dict[str, object]] = {
        "Demand Review": {},
        "PriceMix Review": {},
    }
    for sheet in sheets.values():
        for column, header in zip("ABCDEFGHIJKLMNOP", headers, strict=True):
            sheet[f"{column}5"] = header
    per_sheet_row = {"Demand Review": 6, "PriceMix Review": 6}
    for packet in packets:
        sheet_name = (
            "Demand Review"
            if packet.axis == OperatingEvidenceAxis.DEMAND
            else "PriceMix Review"
        )
        row = per_sheet_row[sheet_name]
        per_sheet_row[sheet_name] += 1
        sheet = sheets[sheet_name]
        sheet.update(
            {
                f"A{row}": packet.packet_id,
                f"B{row}": packet.axis.value,
                f"F{row}": Formula(),
                f"G{row}": (
                    "tampered model hint"
                    if tamper_excerpt and packet is packets[0]
                    else _render_excerpts(packet, "previous_excerpts")
                ),
                f"I{row}": _render_excerpts(packet, "current_excerpts"),
                f"M{row}": Formula('"OK"', "OK"),
                f"N{row}": Formula(),
                f"O{row}": NATURAL_LAYOUT.gold_contract,
                f"P{row}": NATURAL_LAYOUT.gold_split,
            }
        )
        if completed:
            if packet.axis == OperatingEvidenceAxis.DEMAND:
                sheet.update(
                    {
                        f"C{row}": Formula('"COMPLETE"', "COMPLETE")
                        if status_formula
                        else "COMPLETE",
                        f"D{row}": -1,
                        f"E{row}": 1,
                        f"H{row}": packet.previous_excerpts[0].text,
                        f"J{row}": packet.current_excerpts[0].text,
                        f"K{row}": "두 기간의 실현 수요를 독립 비교",
                        f"L{row}": "YES",
                    }
                )
            else:
                sheet.update(
                    {
                        f"C{row}": "INSUFFICIENT_EVIDENCE",
                        f"K{row}": "가격·믹스 실현 상태의 비교 근거가 부족함",
                        f"L{row}": "YES",
                    }
                )

    export_headers = (
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
    export: dict[str, object] = {}
    for column, header in zip("ABCDEFGHIJKLMNOPQ", export_headers, strict=True):
        export[f"{column}1"] = header
    for row in range(2, 2 + len(packets)):
        for column in "ABCDEFGHIJKLMNOPQ":
            export[f"{column}{row}"] = Formula()
    return [
        ("Cover", cover),
        ("Demand Review", sheets["Demand Review"]),
        ("PriceMix Review", sheets["PriceMix Review"]),
        ("Decision Export", export),
    ]


def _balanced_workbook_cells(
    candidate_build: Path, packets: Sequence[PairedAxisPacket]
) -> list[tuple[str, dict[str, object]]]:
    packet_path = candidate_build / BALANCED_LAYOUT.packet_name
    cover: dict[str, object] = {
        "A1": "MoatRader V2 Balanced LOCKED — Independent Retest 1 HUMAN Review",
        "B6": len(packets),
        "B7": sha256_file(packet_path),
        "B8": "FALSE",
        "B9": "FALSE",
        "B10": "FALSE",
        "B11": "TRUE",
        "B12": "FALSE / FALSE / FALSE",
        "B13": "V2_DIRECTIONAL_BALANCED_LOCKED_RETEST_1",
    }
    cover.update({f"B{row}": Formula() for row in range(31, 40)})
    review_headers = (
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
    review: dict[str, object] = {}
    for column, header in zip("ABCDEFGHIJKLMNO", review_headers, strict=True):
        review[f"{column}5"] = header
    for candidate_number, packet in enumerate(packets, start=1):
        row = candidate_number + 5
        review.update(
            {
                f"A{row}": candidate_number,
                f"B{row}": packet.packet_id,
                f"C{row}": packet.axis.value,
                f"G{row}": Formula(),
                f"H{row}": _render_excerpts(packet, "previous_excerpts"),
                f"J{row}": _render_excerpts(packet, "current_excerpts"),
                f"N{row}": Formula('"PENDING"', "PENDING"),
                f"O{row}": Formula(),
            }
        )
    export_headers = (
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
    export: dict[str, object] = {}
    for column, header in zip("ABCDEFGHIJKLMN", export_headers, strict=True):
        export[f"{column}1"] = header
    for row in range(2, 2 + len(packets)):
        for column in "ABCDEFGHIJKLMN":
            export[f"{column}{row}"] = Formula()
    return [("Cover", cover), ("Review", review), ("Decision Export", export)]


@pytest.fixture
def natural_packets() -> list[PairedAxisPacket]:
    return [
        _packet(
            "PKT_111111111111111111111111",
            OperatingEvidenceAxis.DEMAND,
            "실현 수요가 감소했습니다.",
            "실현 수요가 증가했습니다.",
            "1111111111111111111",
        ),
        _packet(
            "PKT_222222222222222222222222",
            OperatingEvidenceAxis.PRICE_MIX,
            "가격 정보가 없습니다.",
            "시장 가격 전망만 있습니다.",
            "2222222222222222222",
        ),
    ]


def test_natural_workbook_audit_ignores_formula_ok_and_stays_not_ready(
    tmp_path: Path, natural_packets: list[PairedAxisPacket]
) -> None:
    candidate_build = _candidate_build(
        tmp_path / "candidates",
        review_type=NATURAL_LAYOUT.review_type,
        packets=natural_packets,
    )
    workbook = tmp_path / "blank.xlsx"
    _write_xlsx(
        workbook,
        _natural_workbook_cells(candidate_build, natural_packets),
    )
    before_hash = sha256_file(workbook)
    audit = audit_human_review_workbook(
        workbook=workbook,
        candidate_build=candidate_build,
        review_type=NATURAL_LAYOUT.review_type,
    )
    assert audit["status"] == "NOT_READY_FOR_HUMAN_IMPORT"
    assert audit["reviewed_count"] == 0
    assert audit["pending_count"] == 2
    assert audit["row_error_count"] == 0
    assert audit["formula_row_checks_trusted"] is False
    assert audit["candidate_excerpts_verified"] is True
    assert sha256_file(workbook) == before_hash
    blocked_output = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError, match="not ready"):
        import_human_review_workbook(
            workbook=workbook,
            candidate_build=candidate_build,
            review_type=NATURAL_LAYOUT.review_type,
            output=blocked_output,
        )
    assert not blocked_output.exists()


def test_natural_workbook_import_is_materializer_ready_and_outcome_blind(
    tmp_path: Path, natural_packets: list[PairedAxisPacket]
) -> None:
    candidate_build = _candidate_build(
        tmp_path / "candidates",
        review_type=NATURAL_LAYOUT.review_type,
        packets=natural_packets,
    )
    workbook = tmp_path / "completed.xlsx"
    _write_xlsx(
        workbook,
        _natural_workbook_cells(
            candidate_build,
            natural_packets,
            completed=True,
        ),
    )
    workbook_hash = sha256_file(workbook)
    decision_output = tmp_path / "review-decisions.json"
    imported = import_human_review_workbook(
        workbook=workbook,
        candidate_build=candidate_build,
        review_type=NATURAL_LAYOUT.review_type,
        output=decision_output,
    )
    assert imported["status"] == "HUMAN_REVIEW_DECISIONS_IMPORTED_OUTCOME_BLIND"
    assert imported["decision_count"] == 2
    assert imported["contract_self_check_yes_count"] == 2
    assert imported["outcome_vault_opened"] is False
    assert imported["return_data_opened"] is False
    assert imported["value_data_opened"] is False
    assert imported["per_pbr_role"] == "NOT_USED"
    assert sha256_file(workbook) == workbook_hash
    decision_hash = sha256_file(decision_output)
    with pytest.raises(FileExistsError, match="already exists"):
        import_human_review_workbook(
            workbook=workbook,
            candidate_build=candidate_build,
            review_type=NATURAL_LAYOUT.review_type,
            output=decision_output,
        )
    assert sha256_file(workbook) == workbook_hash
    assert sha256_file(decision_output) == decision_hash
    materialized = materialize_natural_retest_human_gold(
        candidate_build=candidate_build,
        review_decisions=decision_output,
        output=tmp_path / "human-gold",
    )
    assert materialized["review_decision_count"] == 2
    assert materialized["contract_self_check_required"] is True


def test_natural_workbook_rejects_formula_in_human_input(
    tmp_path: Path, natural_packets: list[PairedAxisPacket]
) -> None:
    candidate_build = _candidate_build(
        tmp_path / "candidates",
        review_type=NATURAL_LAYOUT.review_type,
        packets=natural_packets,
    )
    workbook = tmp_path / "formula-injection.xlsx"
    _write_xlsx(
        workbook,
        _natural_workbook_cells(
            candidate_build,
            natural_packets,
            completed=True,
            status_formula=True,
        ),
    )
    with pytest.raises(ValueError, match="formula locations changed"):
        audit_human_review_workbook(
            workbook=workbook,
            candidate_build=candidate_build,
            review_type=NATURAL_LAYOUT.review_type,
        )


def test_natural_workbook_rejects_tampered_candidate_excerpt(
    tmp_path: Path, natural_packets: list[PairedAxisPacket]
) -> None:
    candidate_build = _candidate_build(
        tmp_path / "candidates",
        review_type=NATURAL_LAYOUT.review_type,
        packets=natural_packets,
    )
    workbook = tmp_path / "tampered.xlsx"
    _write_xlsx(
        workbook,
        _natural_workbook_cells(
            candidate_build,
            natural_packets,
            tamper_excerpt=True,
        ),
    )
    with pytest.raises(ValueError, match="previous candidate excerpts changed"):
        audit_human_review_workbook(
            workbook=workbook,
            candidate_build=candidate_build,
            review_type=NATURAL_LAYOUT.review_type,
        )


def test_review_workbook_rejects_external_relationship(
    tmp_path: Path, natural_packets: list[PairedAxisPacket]
) -> None:
    candidate_build = _candidate_build(
        tmp_path / "candidates",
        review_type=NATURAL_LAYOUT.review_type,
        packets=natural_packets,
    )
    workbook = tmp_path / "external-link.xlsx"
    _write_xlsx(
        workbook,
        _natural_workbook_cells(candidate_build, natural_packets),
        external_relationship=True,
    )
    with pytest.raises(ValueError, match="external relationship"):
        audit_human_review_workbook(
            workbook=workbook,
            candidate_build=candidate_build,
            review_type=NATURAL_LAYOUT.review_type,
        )


def test_review_workbook_rejects_unsafe_formula_function(
    tmp_path: Path, natural_packets: list[PairedAxisPacket]
) -> None:
    candidate_build = _candidate_build(
        tmp_path / "candidates",
        review_type=NATURAL_LAYOUT.review_type,
        packets=natural_packets,
    )
    workbook_cells = _natural_workbook_cells(candidate_build, natural_packets)
    workbook_cells[0][1]["B29"] = Formula("HYPERLINK(A1)")
    workbook = tmp_path / "unsafe-formula.xlsx"
    _write_xlsx(workbook, workbook_cells)
    with pytest.raises(ValueError, match="unsafe formula function"):
        audit_human_review_workbook(
            workbook=workbook,
            candidate_build=candidate_build,
            review_type=NATURAL_LAYOUT.review_type,
        )


def test_balanced_workbook_layout_audits_fresh_candidates(
    tmp_path: Path, natural_packets: list[PairedAxisPacket]
) -> None:
    candidate_build = _candidate_build(
        tmp_path / "balanced-candidates",
        review_type=BALANCED_LAYOUT.review_type,
        packets=natural_packets,
    )
    workbook = tmp_path / "balanced.xlsx"
    _write_xlsx(
        workbook,
        _balanced_workbook_cells(candidate_build, natural_packets),
    )
    audit = audit_human_review_workbook(
        workbook=workbook,
        candidate_build=candidate_build,
        review_type=BALANCED_LAYOUT.review_type,
    )
    assert audit["status"] == "NOT_READY_FOR_HUMAN_IMPORT"
    assert audit["candidate_count"] == 2
    assert audit["pending_count"] == 2
    assert audit["candidate_excerpts_verified"] is True
