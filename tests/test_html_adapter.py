from __future__ import annotations

from moatrader.adapters import EdgarHtmlAdapter, IrHtmlAdapter, RawDocument
from moatrader.adapters import DartHtmlAdapter
from moatrader.canonical.models import SectionNode
from moatrader.canonical.models import SectionNode, SectionRole, SourceType, TableNode

from conftest import build_dart_bundle


DART_TABLE_HTML = """
<html><body>
  <h1>II. 사업의 내용</h1>
  <h2>1. 주요 제품 및 서비스</h2>
  <p>주요 제품 현황</p>
  <p>(단위: 백만원)</p>
  <table>
    <tr><th rowspan="2">사업부</th><th colspan="2">2025년</th></tr>
    <tr><th>제품</th><th>매출</th></tr>
    <tr><td rowspan="2">전자부품</td><td>MLCC</td><td>3,200</td></tr>
    <tr><td>카메라모듈</td><td>(700)</td></tr>
  </table>
  <p>주1) 연결기준입니다.</p>
</body></html>
"""


def test_dart_explicit_xml_sections_become_canonical_hierarchy() -> None:
    source = RawDocument(
        content=b"""<DOCUMENT><BODY>
<SECTION-1><TITLE>I. Company Overview</TITLE><P>Overview paragraph.</P>
<SECTION-2><TITLE>1. Business</TITLE><P>Business paragraph.</P></SECTION-2>
</SECTION-1></BODY></DOCUMENT>""",
        uri="https://dart.fss.or.kr/example.xml",
        hints={
            "source_type": "DART",
            "rcept_no": "20250101000001",
            "corp_code": "00126380",
            "available_at": "2025-01-01T23:59:59+09:00",
        },
    )

    bundle = DartHtmlAdapter().convert(source)

    top = next(node for node in bundle.ast.children if isinstance(node, SectionNode))
    nested = next(node for node in top.children if isinstance(node, SectionNode))
    assert top.title_normalized == "I. Company Overview"
    assert top.explicit_level == 1
    assert nested.title_normalized == "1. Business"
    assert nested.explicit_level == 2
    assert nested.section_path == ["I. Company Overview", "1. Business"]
    assert bundle.quality.heading_count == 2
    assert bundle.quality.paragraph_count == 2
    assert bundle.quality.duplicate_text_ratio == 0.0


def test_dart_xml_encoding_declaration_is_accepted() -> None:
    source = RawDocument(
        content=(
            b'<?xml version="1.0" encoding="utf-8"?>\n'
            b"<DOCUMENT><BODY><SECTION-1><TITLE>I. Overview</TITLE>"
            b"<P>Visible disclosure.</P></SECTION-1></BODY></DOCUMENT>"
        ),
        hints={
            "source_type": "DART",
            "rcept_no": "20250101000002",
            "corp_code": "00126380",
            "available_at": "2025-01-01T23:59:59+09:00",
        },
    )

    bundle = DartHtmlAdapter().convert(source)

    assert bundle.quality.paragraph_count == 1
    assert any(node.normalized_text == "Visible disclosure." for node in bundle.ast.walk())


def test_dart_table_is_rectangular_and_context_is_preserved():
    bundle = build_dart_bundle(DART_TABLE_HTML)
    nodes = list(bundle.ast.walk())
    sections = [node for node in nodes if isinstance(node, SectionNode)]
    table = next(node for node in nodes if isinstance(node, TableNode))

    assert sections[0].role == SectionRole.BUSINESS
    assert table.section_path == ["II. 사업의 내용", "1. 주요 제품 및 서비스"]
    assert [[cell.normalized_text for cell in row.cells] for row in table.rows] == [
        ["사업부", "2025년", "2025년"],
        ["사업부", "제품", "매출"],
        ["전자부품", "MLCC", "3,200"],
        ["전자부품", "카메라모듈", "(700)"],
    ]
    assert table.rows[3].cells[0].propagated is True
    assert table.rows[3].cells[2].numeric_value == -700
    assert [header.path for header in table.column_headers] == [
        ["사업부"],
        ["2025년", "제품"],
        ["2025년", "매출"],
    ]
    assert table.unit is not None and table.unit.canonical == "KRW_MILLION"
    assert table.caption == "주요 제품 현황"
    assert table.footnotes[0].text == "주1) 연결기준입니다."
    assert table.source_refs[0].xpath
    assert bundle.quality.raw_table_count == bundle.quality.ast_table_count == 1
    assert bundle.quality.raw_numeric_cell_count == bundle.quality.numeric_cell_count
    assert bundle.quality.numeric_retention == 1.0


def test_ids_are_deterministic_for_same_document_and_parser():
    first = build_dart_bundle(DART_TABLE_HTML)
    second = build_dart_bundle(DART_TABLE_HTML)
    assert [node.node_id for node in first.ast.walk()] == [node.node_id for node in second.ast.walk()]


def test_edgar_inline_xbrl_is_kept_as_structured_fact():
    source = RawDocument(
        content=b"""
        <html><body>
          <h1>Financial Statements</h1>
          <xbrli:context id="FY25">
            <xbrli:entity><xbrli:identifier>00001234</xbrli:identifier></xbrli:entity>
            <xbrli:period><xbrli:startDate>2025-01-01</xbrli:startDate><xbrli:endDate>2025-12-31</xbrli:endDate></xbrli:period>
          </xbrli:context>
          <p>Revenue was <ix:nonfraction name="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax" contextRef="FY25" unitRef="USD" decimals="0">1,250</ix:nonfraction>.</p>
        </body></html>
        """,
        hints={
            "source_type": "SEC_EDGAR",
            "accession_number": "00001234-25-000001",
            "form_type": "10-K",
            "issuer_name": "Example Inc.",
            "available_at": "2026-02-15T16:31:00-05:00",
        },
    )
    bundle = EdgarHtmlAdapter().convert(source)
    assert bundle.metadata.source_type == SourceType.SEC_EDGAR
    assert len(bundle.facts) == 1
    assert "2025-01-012025-12-31" not in " ".join(node.normalized_text for node in bundle.ast.walk())
    fact = bundle.facts[0]
    assert fact.numeric_value == 1250
    assert fact.period.start.isoformat() == "2025-01-01"
    assert fact.period.end.isoformat() == "2025-12-31"
    assert fact.unit is not None and fact.unit.canonical == "USD"
    assert bundle.provenance.records[fact.fact_id].transform == "inline_xbrl_extract"


def test_ir_html_keeps_asset_and_source_provenance():
    source = RawDocument(
        content="""<html><body><h1>Investment Highlights</h1><figure><img src="growth.png" alt="Revenue growth"><figcaption>AI 매출 성장</figcaption></figure></body></html>""".encode(),
        hints={
            "source_type": "IR",
            "source_document_id": "ir-2026-q2",
            "issuer_name": "Example",
            "available_at": "2026-08-01T00:00:00+09:00",
        },
    )
    bundle = IrHtmlAdapter().convert(source)
    assert bundle.metadata.source_type == SourceType.IR
    assert len(bundle.assets) == 1
    assert bundle.assets[0].uri == "growth.png"
    assert bundle.assets[0].source_refs[0].document_id == "ir-2026-q2"
    figure = next(node for node in bundle.ast.walk() if node.kind == "figure")
    assert figure.asset_id == bundle.assets[0].asset_id


def test_nested_table_is_not_silently_lost():
    bundle = build_dart_bundle(
        """<html><body><h1>II. 사업의 내용</h1><table><tr><td>외부<table><tr><td>내부</td></tr></table></td></tr></table></body></html>"""
    )
    tables = [node for node in bundle.ast.walk() if isinstance(node, TableNode)]
    assert len(tables) == 2
    assert bundle.quality.ast_table_count == bundle.quality.raw_table_count == 2
