from __future__ import annotations

import pytest

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


DART_TE_FINANCIAL_XML = """
<DOCUMENT><BODY>
  <SECTION-1><TITLE>III. 재무에 관한 사항</TITLE>
    <SECTION-2><TITLE>2. 연결재무제표</TITLE>
      <TABLE-GROUP>
        <TITLE>2-2. 연결 포괄손익계산서</TITLE>
        <TABLE><TR><TE>(단위 : 원)</TE></TR></TABLE>
        <TABLE>
          <THEAD><TR><TH>구분</TH><TH>제26기 1분기</TH><TH>제25기 1분기</TH></TR></THEAD>
          <TBODY>
            <TR><TE>매출액</TE>
              <TE ACODE="ifrs-full_Revenue" ACONTEXT="CFY2026dFQA_ifrs-full_ConsolidatedAndSeparateFinancialStatementsAxis_ifrs-full_ConsolidatedMember" ADECIMAL="0" ANEGATED="N">146,100,000,000</TE>
              <TE ACODE="ifrs-full_Revenue" ACONTEXT="PFY2025dFQA_ifrs-full_ConsolidatedAndSeparateFinancialStatementsAxis_ifrs-full_ConsolidatedMember" ADECIMAL="0" ANEGATED="N">100,000,000,000</TE>
            </TR>
            <TR><TE>영업이익</TE>
              <TE ACODE="dart_OperatingIncomeLoss" ACONTEXT="CFY2026dFQA_ifrs-full_ConsolidatedAndSeparateFinancialStatementsAxis_ifrs-full_ConsolidatedMember" ADECIMAL="0" ANEGATED="N">57,292,000,000</TE>
              <TE ACODE="dart_OperatingIncomeLoss" ACONTEXT="PFY2025dFQA_ifrs-full_ConsolidatedAndSeparateFinancialStatementsAxis_ifrs-full_ConsolidatedMember" ADECIMAL="0" ANEGATED="N">35,000,000,000</TE>
            </TR>
          </TBODY>
        </TABLE>
      </TABLE-GROUP>
    </SECTION-2>
  </SECTION-1>
</BODY></DOCUMENT>
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


def test_table_period_prefers_current_header_and_rejects_prose_as_unit() -> None:
    bundle = build_dart_bundle(
        """<DOCUMENT><BODY>
        <P>단위 : 품목별 1KG당 금액임. 가격변동원인은 원재료 가격임.</P>
        <TABLE><TR><TH>구분</TH><TH>2026년 1분기</TH><TH>2025년</TH><TH>2024년</TH></TR>
        <TR><TD>연구개발 프로젝트 2008</TD><TD>10</TD><TD>9</TD><TD>8</TD></TR></TABLE>
        </BODY></DOCUMENT>"""
    )

    table = next(node for node in bundle.ast.walk() if isinstance(node, TableNode))
    assert table.unit is None
    assert table.period is not None
    assert table.period.fiscal_year == 2026
    assert table.period.fiscal_period == "1분기"


def test_prior_table_unit_does_not_leak_into_next_operating_table() -> None:
    bundle = build_dart_bundle(
        """<DOCUMENT><BODY>
        <TABLE><TR><TD>(단위 : 백만원)</TD></TR></TABLE>
        <TABLE><TR><TH>품목</TH><TH>2025년</TH></TR><TR><TD>PN</TD><TD>131</TD></TR></TABLE>
        <P>다. 생산능력 및 생산실적</P>
        <TABLE><TR><TH>품목</TH><TH>구분</TH><TH>2025년</TH></TR>
        <TR><TD>의료기기</TD><TD>생산능력</TD><TD>7,200,000</TD></TR></TABLE>
        </BODY></DOCUMENT>"""
    )

    production = next(
        node
        for node in bundle.ast.walk()
        if isinstance(node, TableNode) and "생산능력" in node.normalized_text
    )
    assert production.unit is None


def test_dart_te_cells_are_tables_and_acode_is_a_structured_fact() -> None:
    bundle = build_dart_bundle(
        DART_TE_FINANCIAL_XML,
        report_name="분기보고서 (2026.03)",
        period_start=None,
        period_end="2026-03-31",
    )
    statement = next(
        node
        for node in bundle.ast.walk()
        if isinstance(node, TableNode)
        and any(cell.normalized_text == "매출액" for row in node.rows for cell in row.cells)
    )

    assert [[cell.normalized_text for cell in row.cells] for row in statement.rows] == [
        ["구분", "제26기 1분기", "제25기 1분기"],
        ["매출액", "146,100,000,000", "100,000,000,000"],
        ["영업이익", "57,292,000,000", "35,000,000,000"],
    ]
    assert bundle.quality.raw_numeric_cell_count == 4
    assert bundle.quality.numeric_cell_count == 4
    assert bundle.quality.numeric_retention == 1.0
    assert bundle.quality.raw_structured_fact_count == 4
    assert bundle.quality.structured_fact_count == 4
    assert bundle.quality.structured_fact_retention == 1.0
    assert len(bundle.facts) == 4

    current_revenue = next(
        fact
        for fact in bundle.facts
        if fact.concept == "ifrs-full_Revenue" and fact.period.fiscal_year == 2026
    )
    assert current_revenue.label == "매출액"
    assert current_revenue.numeric_value == 146_100_000_000
    assert current_revenue.period.start is not None
    assert current_revenue.period.start.isoformat() == "2026-01-01"
    assert current_revenue.period.end is not None
    assert current_revenue.period.end.isoformat() == "2026-03-31"
    assert current_revenue.scope.value == "CONSOLIDATED"
    assert current_revenue.dimensions == []
    assert current_revenue.unit is not None and current_revenue.unit.canonical == "KRW"
    assert bundle.provenance.records[current_revenue.fact_id].transform == "dart_te_xbrl_extract"


def test_dart_primary_statement_without_acontext_infers_context_from_headers() -> None:
    bundle = build_dart_bundle(
        """<DOCUMENT><BODY><SECTION-1><TITLE>III. 재무에 관한 사항</TITLE>
        <TABLE-GROUP ACLASS="{XBRL}BS_C"><TITLE>2-1. 연결 재무상태표</TITLE>
        <TABLE><TR><TE>(단위 : 원)</TE></TR></TABLE><TABLE>
        <TR><TH></TH><TH ENG="FY 2025">제 10 기 반기말</TH><TH ENG="FY 2024">제 9 기말</TH></TR>
        <TR><TE>현금및현금성자산</TE><TE ACODE="ifrs-full_CashAndCashEquivalents" ADECIMAL="0">1,000</TE><TE ACODE="ifrs-full_CashAndCashEquivalents" ADECIMAL="0">800</TE></TR>
        </TABLE></TABLE-GROUP>
        <TABLE-GROUP ACLASS="{XBRL}IS_C1"><TITLE>2-2. 연결 포괄손익계산서</TITLE>
        <TABLE><TR><TE>(단위 : 원)</TE></TR></TABLE><TABLE>
        <TR><TH ROWSPAN="2"></TH><TH ENG="CFHY" COLSPAN="2">제 10 기 반기</TH><TH ENG="PFHY" COLSPAN="2">제 9 기 반기</TH></TR>
        <TR><TH ENG="THREE MONTH">3개월</TH><TH ENG="ACCUMULATE">누적</TH><TH ENG="THREE MONTH">3개월</TH><TH ENG="ACCUMULATE">누적</TH></TR>
        <TR><TE>매출액</TE><TE ACODE="ifrs-full_Revenue">100</TE><TE ACODE="ifrs-full_Revenue">180</TE><TE ACODE="ifrs-full_Revenue">90</TE><TE ACODE="ifrs-full_Revenue">160</TE></TR>
        </TABLE></TABLE-GROUP>
        <TABLE-GROUP ACLASS="{XBRL}CF_C"><TITLE>2-4. 연결 현금흐름표</TITLE>
        <TABLE><TR><TE>(단위 : 원)</TE></TR></TABLE><TABLE>
        <TR><TH></TH><TH ENG="FY 2025">제 10 기 반기</TH><TH ENG="FY 2024">제 9 기 반기</TH></TR>
        <TR><TE>영업활동현금흐름</TE><TE ACODE="ifrs-full_CashFlowsFromUsedInOperatingActivities">70</TE><TE ACODE="ifrs-full_CashFlowsFromUsedInOperatingActivities">60</TE></TR>
        </TABLE></TABLE-GROUP></SECTION-1></BODY></DOCUMENT>""",
        report_name="반기보고서 (2025.06)",
        period_start=None,
        period_end="2025-06-30",
    )

    assert bundle.quality.raw_structured_fact_count == 8
    assert bundle.quality.structured_fact_count == 8
    assert bundle.quality.structured_fact_retention == 1.0
    current_revenues = sorted(
        (
            fact
            for fact in bundle.facts
            if fact.concept == "ifrs-full_Revenue" and fact.period.end.isoformat() == "2025-06-30"
        ),
        key=lambda fact: fact.period.start,
    )
    assert [fact.period.start.isoformat() for fact in current_revenues] == [
        "2025-01-01",
        "2025-04-01",
    ]
    assert {fact.numeric_value for fact in current_revenues} == {100, 180}
    current_cash = next(
        fact
        for fact in bundle.facts
        if fact.concept == "ifrs-full_CashAndCashEquivalents" and fact.numeric_value == 1000
    )
    prior_cash = next(
        fact
        for fact in bundle.facts
        if fact.concept == "ifrs-full_CashAndCashEquivalents" and fact.numeric_value == 800
    )
    assert current_cash.period.instant.isoformat() == "2025-06-30"
    assert prior_cash.period.instant.isoformat() == "2024-12-31"
    assert current_cash.scope.value == "CONSOLIDATED"
    assert bundle.provenance.records[current_cash.fact_id].transform == "dart_te_table_context_inference"


def test_dart_header_context_supports_non_calendar_first_quarter() -> None:
    bundle = build_dart_bundle(
        """<DOCUMENT><BODY><TABLE-GROUP ACLASS="{XBRL}IS_S1"><TITLE>포괄손익계산서</TITLE>
        <TABLE><TR><TE>(단위 : 원)</TE></TR></TABLE><TABLE>
        <TR><TH ROWSPAN="2"></TH><TH ENG="CFY 1Q" COLSPAN="2">제 41 기 1분기</TH><TH ENG="PFY 1Q" COLSPAN="2">제 40 기 1분기</TH></TR>
        <TR><TH ENG="THREE MONTH">3개월</TH><TH ENG="Cumulative">누적</TH><TH ENG="THREE MONTH">3개월</TH><TH ENG="Cumulative">누적</TH></TR>
        <TR><TE>매출액</TE><TE ACODE="ifrs-full_Revenue">100</TE><TE ACODE="ifrs-full_Revenue">100</TE><TE ACODE="ifrs-full_Revenue">80</TE><TE ACODE="ifrs-full_Revenue">80</TE></TR>
        </TABLE></TABLE-GROUP></BODY></DOCUMENT>""",
        report_name="분기보고서 (2025.06)",
        period_start=None,
        period_end="2025-06-30",
    )
    current = [
        fact
        for fact in bundle.facts
        if fact.period.end.isoformat() == "2025-06-30"
    ]

    assert len(current) == 2
    assert {fact.period.start.isoformat() for fact in current} == {"2025-04-01"}
    assert all(fact.scope.value == "SEPARATE" for fact in current)


@pytest.mark.parametrize(
    ("context_id", "report_end", "expected_start", "expected_end"),
    [
        ("CFY2025dHYA", "2025-06-30", "2025-01-01", "2025-06-30"),
        ("CFY2025dHYQ", "2025-06-30", "2025-04-01", "2025-06-30"),
        ("CFY2024dTQA", "2024-09-30", "2024-01-01", "2024-09-30"),
        ("CFY2024dTQQ", "2024-09-30", "2024-07-01", "2024-09-30"),
        ("CFY2024dFY", "2024-12-31", "2024-01-01", "2024-12-31"),
    ],
)
def test_dart_acontext_periods_preserve_fiscal_month_ends(
    context_id: str,
    report_end: str,
    expected_start: str,
    expected_end: str,
) -> None:
    bundle = build_dart_bundle(
        f"""<DOCUMENT><BODY><TABLE><TR><TE>(단위 : 원)</TE></TR></TABLE>
        <TABLE><TR><TE>매출액</TE><TE ACODE="ifrs-full_Revenue"
        ACONTEXT="{context_id}_ifrs-full_ConsolidatedAndSeparateFinancialStatementsAxis_ifrs-full_ConsolidatedMember">100</TE></TR></TABLE>
        </BODY></DOCUMENT>""",
        period_start=None,
        period_end=report_end,
    )
    period = bundle.facts[0].period
    assert period.start is not None and period.start.isoformat() == expected_start
    assert period.end is not None and period.end.isoformat() == expected_end


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
    assert bundle.quality.raw_structured_fact_count == 1
    assert bundle.quality.structured_fact_count == 1
    assert bundle.quality.structured_fact_retention == 1.0
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


def test_dart_correction_wrapper_tables_are_not_silently_lost() -> None:
    bundle = build_dart_bundle(
        """<DOCUMENT><BODY><LIBRARY><CORRECTION>
        <TABLE><TR><TD>Correction item</TD><TD>Before</TD><TD>After</TD></TR></TABLE>
        <TABLE><TR><TD>Reason</TD><TD>Clerical correction</TD></TR></TABLE>
        </CORRECTION></LIBRARY></BODY></DOCUMENT>"""
    )
    tables = [node for node in bundle.ast.walk() if isinstance(node, TableNode)]

    assert len(tables) == 2
    assert bundle.quality.ast_table_count == bundle.quality.raw_table_count == 2
