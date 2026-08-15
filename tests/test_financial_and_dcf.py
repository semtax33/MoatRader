from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from moatrader.adapters import EdgarHtmlAdapter, RawDocument
from moatrader.financial import DcfAssumptions, DcfAssumptionType, DcfEngine, FinancialSnapshotBuilder

from conftest import build_dart_bundle


def _financial_bundle():
    return EdgarHtmlAdapter().convert(
        RawDocument(
            content=b"""<html><body>
            <xbrli:context id="FY24"><xbrli:period><xbrli:startDate>2024-01-01</xbrli:startDate><xbrli:endDate>2024-12-31</xbrli:endDate></xbrli:period></xbrli:context>
            <ix:nonfraction name="us-gaap:Revenue" contextRef="FY24" unitRef="USD">1000</ix:nonfraction>
            <ix:nonfraction name="us-gaap:OperatingIncomeLoss" contextRef="FY24" unitRef="USD">200</ix:nonfraction>
            <ix:nonfraction name="us-gaap:NetCashProvidedByUsedInOperatingActivities" contextRef="FY24" unitRef="USD">180</ix:nonfraction>
            <ix:nonfraction name="us-gaap:PaymentsToAcquirePropertyPlantAndEquipment" contextRef="FY24" unitRef="USD">50</ix:nonfraction>
            </body></html>""",
            hints={
                "source_type": "SEC_EDGAR",
                "accession_number": "a1",
                "issuer_name": "Example",
                "available_at": "2025-02-01T12:00:00+00:00",
            },
        )
    )


def test_financial_snapshot_derives_margin_and_fcf_in_python():
    bundle = _financial_bundle()
    snapshot = FinancialSnapshotBuilder().build(
        [bundle], as_of=datetime(2025, 3, 1, tzinfo=timezone.utc)
    )
    derived = {(metric.name, metric.period): metric for metric in snapshot.derived_metrics}
    period = next(iter({metric.period for metric in snapshot.derived_metrics if metric.period}))
    assert derived[("EBIT_MARGIN", period)].value == Decimal("0.2")
    assert derived[("FCF", period)].value == Decimal("130")
    assert derived[("FCF_MARGIN", period)].value == Decimal("0.13")
    assert "Values come from StructuredFact" in snapshot.to_markdown()


def test_financial_snapshot_respects_point_in_time_cutoff():
    snapshot = FinancialSnapshotBuilder().build(
        [_financial_bundle()], as_of=datetime(2025, 1, 1, tzinfo=timezone.utc)
    )
    assert not snapshot.series
    assert "no document was available" in snapshot.warnings[0]


def test_dart_te_facts_feed_financial_snapshot_and_derived_metrics() -> None:
    bundle = build_dart_bundle(
        """<DOCUMENT><BODY><SECTION-1><TITLE>III. 재무에 관한 사항</TITLE>
        <TABLE-GROUP><TITLE>연결재무제표</TITLE>
        <TABLE><TR><TE>(단위 : 원)</TE></TR></TABLE>
        <TABLE><TR><TH>구분</TH><TH>제26기 1분기</TH></TR>
        <TR><TE>매출액</TE><TE ACODE="ifrs-full_Revenue" ACONTEXT="CFY2026dFQA_ifrs-full_ConsolidatedAndSeparateFinancialStatementsAxis_ifrs-full_ConsolidatedMember">146100000000</TE></TR>
        <TR><TE>영업이익</TE><TE ACODE="dart_OperatingIncomeLoss" ACONTEXT="CFY2026dFQA_ifrs-full_ConsolidatedAndSeparateFinancialStatementsAxis_ifrs-full_ConsolidatedMember">57292000000</TE></TR>
        <TR><TE>관계기업투자손익</TE><TE ACODE="ifrs-full_ShareOfProfitLossOfAssociatesAccountedForUsingEquityMethod" ACONTEXT="CFY2026dFQA_ifrs-full_ConsolidatedAndSeparateFinancialStatementsAxis_ifrs-full_ConsolidatedMember">450000000</TE></TR>
        <TR><TE>분기순이익</TE><TE ACODE="ifrs-full_ProfitLoss" ACONTEXT="CFY2026dFQA_ifrs-full_ConsolidatedAndSeparateFinancialStatementsAxis_ifrs-full_ConsolidatedMember">48748000000</TE></TR>
        <TR><TE>영업활동현금흐름</TE><TE ACODE="ifrs-full_CashFlowsFromUsedInOperatingActivities" ACONTEXT="CFY2026dFQA_ifrs-full_ConsolidatedAndSeparateFinancialStatementsAxis_ifrs-full_ConsolidatedMember">50000000000</TE></TR>
        <TR><TE>현금및현금성자산의순증가</TE><TE ACODE="ifrs-full_IncreaseDecreaseInCashAndCashEquivalents" ACONTEXT="CFY2026dFQA_ifrs-full_ConsolidatedAndSeparateFinancialStatementsAxis_ifrs-full_ConsolidatedMember">28000000000</TE></TR>
        <TR><TE>현금및현금성자산</TE><TE ACODE="ifrs-full_CashAndCashEquivalents" ACONTEXT="CFY2026eFQA_ifrs-full_ConsolidatedAndSeparateFinancialStatementsAxis_ifrs-full_ConsolidatedMember">204000000000</TE></TR>
        <TR><TE>유형자산의 취득</TE><TE ACODE="ifrs-full_PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities" ACONTEXT="CFY2026dFQA_ifrs-full_ConsolidatedAndSeparateFinancialStatementsAxis_ifrs-full_ConsolidatedMember" ANEGATED="Y">(8000000000)</TE></TR>
        </TABLE></TABLE-GROUP></SECTION-1></BODY></DOCUMENT>""",
        report_name="분기보고서 (2026.03)",
        period_start=None,
        period_end="2026-03-31",
    )
    snapshot = FinancialSnapshotBuilder().build(
        [bundle], as_of=datetime(2026, 5, 31, tzinfo=timezone.utc)
    )
    series = snapshot.series_index()
    assert series["REVENUE"].points[0].value == Decimal("146100000000")
    assert series["EBIT"].points[0].value == Decimal("57292000000")
    assert series["NET_INCOME"].points[0].value == Decimal("48748000000")
    assert series["CASH"].points[0].value == Decimal("204000000000")
    derived = {metric.name: metric for metric in snapshot.derived_metrics}
    assert derived["FCF"].value == Decimal("42000000000")
    assert derived["EBIT_MARGIN"].value == Decimal("57292000000") / Decimal("146100000000")


def test_financial_snapshot_normalizes_scaled_currency_units() -> None:
    bundle = build_dart_bundle(
        """<DOCUMENT><BODY><TABLE><TR><TE>(단위 : 백만원)</TE></TR></TABLE>
        <TABLE><TR><TE>매출액</TE><TE ACODE="ifrs-full_Revenue"
        ACONTEXT="CFY2024dFY_ifrs-full_ConsolidatedAndSeparateFinancialStatementsAxis_ifrs-full_ConsolidatedMember">1,234</TE></TR></TABLE>
        </BODY></DOCUMENT>""",
    )
    snapshot = FinancialSnapshotBuilder().build(
        [bundle], as_of=datetime(2025, 5, 31, tzinfo=timezone.utc)
    )
    point = snapshot.series_index()["REVENUE"].points[0]
    assert point.value == Decimal("1234000000")
    assert point.unit == "KRW"


def test_custom_dart_aggregate_capex_is_recognized_from_its_account_label() -> None:
    bundle = build_dart_bundle(
        """<DOCUMENT><BODY><TABLE><TR><TE>(단위 : 원)</TE></TR></TABLE>
        <TABLE><TR><TE>유형자산의 취득</TE><TE
        ACODE="entity00536523_IncreaseOfPropertyPlantAndEquipmentOfCashFlowsFromUsedInInvestingActivities"
        ACONTEXT="CFY2024dFY_ifrs-full_ConsolidatedAndSeparateFinancialStatementsAxis_ifrs-full_ConsolidatedMember">101640000</TE></TR></TABLE>
        </BODY></DOCUMENT>""",
    )
    snapshot = FinancialSnapshotBuilder().build(
        [bundle], as_of=datetime(2025, 5, 31, tzinfo=timezone.utc)
    )

    point = snapshot.series_index()["CAPEX"].points[0]
    assert point.value == Decimal("101640000")
    assert len(point.source_fact_ids) == 1


def test_component_only_dart_capex_is_summed_without_double_counting_scopes() -> None:
    consolidated = "ifrs-full_ConsolidatedAndSeparateFinancialStatementsAxis_ifrs-full_ConsolidatedMember"
    separate = "ifrs-full_ConsolidatedAndSeparateFinancialStatementsAxis_ifrs-full_SeparateMember"
    bundle = build_dart_bundle(
        f"""<DOCUMENT><BODY><TABLE><TR><TE>(단위 : 원)</TE></TR></TABLE><TABLE>
        <TR><TE>건물의 취득</TE><TE ACODE="dart_PurchaseOfBuildings" ACONTEXT="CFY2024dFY_{consolidated}">-100</TE></TR>
        <TR><TE>기계장치의 취득</TE><TE ACODE="dart_PurchaseOfMachinery" ACONTEXT="CFY2024dFY_{consolidated}">-200</TE></TR>
        <TR><TE>차량운반구의 취득</TE><TE ACODE="dart_PurchaseOfVehicles" ACONTEXT="CFY2024dFY_{consolidated}">-30</TE></TR>
        <TR><TE>건물의 취득</TE><TE ACODE="dart_PurchaseOfBuildings" ACONTEXT="CFY2024dFY_{separate}">-999</TE></TR>
        </TABLE></BODY></DOCUMENT>""",
    )
    snapshot = FinancialSnapshotBuilder().build(
        [bundle], as_of=datetime(2025, 5, 31, tzinfo=timezone.utc)
    )

    point = snapshot.series_index()["CAPEX"].points[0]
    assert point.value == Decimal("-330")
    assert len(point.source_fact_ids) == 3


def test_continuing_operations_cash_flow_variant_maps_to_cfo() -> None:
    bundle = build_dart_bundle(
        """<DOCUMENT><BODY><TABLE><TR><TE>(단위 : 원)</TE></TR></TABLE><TABLE>
        <TR><TE>영업활동으로 인한 순현금흐름</TE><TE
        ACODE="ifrs-full_CashFlowsFromUsedInOperatingActivitiesContinuingOperations"
        ACONTEXT="CFY2024dFY_ifrs-full_ConsolidatedAndSeparateFinancialStatementsAxis_ifrs-full_ConsolidatedMember">-3,486,952,474</TE></TR>
        </TABLE></BODY></DOCUMENT>""",
    )
    snapshot = FinancialSnapshotBuilder().build(
        [bundle], as_of=datetime(2025, 5, 31, tzinfo=timezone.utc)
    )

    assert snapshot.series_index()["CFO"].points[0].value == Decimal("-3486952474")


def test_plain_revenue_table_is_promoted_when_structured_facts_are_missing() -> None:
    bundle = build_dart_bundle(
        """<html><body><p>(단위: 백만원)</p><table>
        <tr><th>사업부</th><th>제품</th><th>2025년 매출</th></tr>
        <tr><td>의료기기</td><td>A</td><td>3,200</td></tr>
        <tr><td>화장품</td><td>B</td><td>700</td></tr>
        </table></body></html>"""
        ,
        available_at="2026-03-31T09:00:00+09:00",
        period_start="2025-01-01",
        period_end="2025-12-31",
    )

    snapshot = FinancialSnapshotBuilder().build(
        [bundle], as_of=datetime(2026, 5, 31, tzinfo=timezone.utc)
    )

    point = snapshot.series_index()["REVENUE"].points[0]
    assert point.period.isoformat() == "2025-12-31"
    assert point.value == Decimal("3900000000")
    assert point.unit == "KRW"
    assert {item.dimension for item in snapshot.breakdowns} == {"의료기기 / A", "화장품 / B"}
    assert "inferred deterministically from table" in snapshot.warnings[0]


def test_quarter_label_is_not_misread_as_january_month_end() -> None:
    bundle = build_dart_bundle(
        """<html><body><h2>주요 제품 등의 현황</h2><p>(단위: 백만원)</p><table>
        <tr><th rowspan="2">구분</th><th>2026년 1분기</th></tr>
        <tr><th>금액</th></tr>
        <tr><td>합계</td><td>146,100</td></tr>
        </table></body></html>""",
        report_name="분기보고서 (2026.03)",
        available_at="2026-05-15T09:00:00+09:00",
        period_start=None,
        period_end="2026-03-31",
    )

    snapshot = FinancialSnapshotBuilder().build(
        [bundle], as_of=datetime(2026, 5, 31, tzinfo=timezone.utc)
    )

    point = snapshot.series_index()["REVENUE"].points[0]
    assert point.period.isoformat() == "2026-03-31"
    assert point.period_basis == "Q1"


def test_audited_summary_table_adds_consolidated_annual_revenue_and_ebit() -> None:
    bundle = build_dart_bundle(
        """<DOCUMENT><BODY><SECTION-1><TITLE>III. 재무에 관한 사항</TITLE>
        <SECTION-2><TITLE>1. 요약재무정보</TITLE>
        <TABLE><TR><TD>(단위 : 백만원)</TD></TR></TABLE>
        <TABLE><TR><TH>구분</TH><TH>제26기 1분기(2026.01.01~2026.03.31)</TH><TH>제25기 연간(2025.01.01~2025.12.31)</TH></TR>
        <TR><TD>매출액</TD><TD>146,100</TD><TD>536,289</TD></TR>
        <TR><TD>영업이익</TD><TD>57,292</TD><TD>214,396</TD></TR></TABLE>
        <TABLE><TR><TD>(단위 : 백만원)</TD></TR></TABLE>
        <TABLE><TR><TH>구분</TH><TH>제26기 1분기(2026.01.01~2026.03.31)</TH><TH>제25기 연간(2025.01.01~2025.12.31)</TH></TR>
        <TR><TD>매출액</TD><TD>124,733</TD><TD>473,790</TD></TR>
        <TR><TD>영업이익</TD><TD>55,951</TD><TD>207,553</TD></TR></TABLE>
        <TABLE><TR><TD>(단위 : 원)</TD></TR></TABLE>
        <TABLE><TR><TE>매출액</TE><TE ACODE="ifrs-full_Revenue" ACONTEXT="CFY2026dFQA_ifrs-full_ConsolidatedAndSeparateFinancialStatementsAxis_ifrs-full_ConsolidatedMember">146100260508</TE></TR>
        <TR><TE>영업이익</TE><TE ACODE="dart_OperatingIncomeLoss" ACONTEXT="CFY2026dFQA_ifrs-full_ConsolidatedAndSeparateFinancialStatementsAxis_ifrs-full_ConsolidatedMember">57292342573</TE></TR></TABLE>
        </SECTION-2></SECTION-1></BODY></DOCUMENT>""",
        report_name="분기보고서 (2026.03)",
        period_start=None,
        period_end="2026-03-31",
        available_at="2026-05-15T09:01:02+09:00",
    )

    snapshot = FinancialSnapshotBuilder().build(
        [bundle], as_of=datetime(2026, 5, 31, tzinfo=timezone.utc)
    )

    revenue = {point.period.isoformat(): point.value for point in snapshot.series_index()["REVENUE"].points}
    ebit = {point.period.isoformat(): point.value for point in snapshot.series_index()["EBIT"].points}
    assert revenue["2025-12-31"] == Decimal("536289000000")
    assert ebit["2025-12-31"] == Decimal("214396000000")
    assert any("consolidated-match error" in warning for warning in snapshot.warnings)


def test_dcf_engine_is_deterministic_and_reconciles_value_bridge():
    assumptions = DcfAssumptions(
        base_revenue=Decimal("1000"),
        revenue_growth=[Decimal("0.10"), Decimal("0.08"), Decimal("0.05")],
        ebit_margin=[Decimal("0.20"), Decimal("0.21"), Decimal("0.22")],
        tax_rate=Decimal("0.25"),
        depreciation_pct_revenue=Decimal("0.03"),
        capex_pct_revenue=Decimal("0.04"),
        nwc_pct_revenue=Decimal("0.10"),
        wacc=Decimal("0.09"),
        terminal_growth=Decimal("0.025"),
        net_debt=Decimal("150"),
        diluted_shares=Decimal("100"),
    )
    valuation = DcfEngine().value(assumptions)
    assert len(valuation.projections) == 3
    assert valuation.enterprise_value - assumptions.net_debt == valuation.equity_value
    assert valuation.equity_value / assumptions.diluted_shares == valuation.fair_value_per_share
    assert valuation.fair_value_per_share > 0


def test_dcf_result_exposes_assumption_provenance_and_default_penalty() -> None:
    assumption_types = {
        "base_revenue": DcfAssumptionType.DETERMINISTIC,
        "revenue_growth": DcfAssumptionType.MODEL_INFERENCE,
        "ebit_margin": DcfAssumptionType.MODEL_INFERENCE,
        "tax_rate": DcfAssumptionType.DEFAULT,
        "depreciation_pct_revenue": DcfAssumptionType.MODEL_INFERENCE,
        "capex_pct_revenue": DcfAssumptionType.MODEL_INFERENCE,
        "nwc_pct_revenue": DcfAssumptionType.MODEL_INFERENCE,
        "wacc": DcfAssumptionType.DETERMINISTIC,
        "terminal_growth": DcfAssumptionType.DEFAULT,
        "net_debt": DcfAssumptionType.DETERMINISTIC,
        "diluted_shares": DcfAssumptionType.DETERMINISTIC,
    }
    assumptions = DcfAssumptions(
        base_period="2025H1_TTM",
        base_revenue=Decimal("1000"),
        revenue_growth=[Decimal("0.1")],
        ebit_margin=[Decimal("0.2")],
        tax_rate=Decimal("0.24"),
        depreciation_pct_revenue=Decimal("0.03"),
        capex_pct_revenue=Decimal("0.04"),
        nwc_pct_revenue=Decimal("0.1"),
        wacc=Decimal("0.09"),
        terminal_growth=Decimal("0.02"),
        diluted_shares=Decimal("100"),
        assumption_sources={"base_revenue": ["PIT_TTM:2025H1"]},
        assumption_types=assumption_types,
    )

    valuation = DcfEngine().value(assumptions)

    assert valuation.method == "FCFF"
    assert valuation.base_period == "2025H1_TTM"
    assert valuation.assumptions.assumption_sources["base_revenue"] == ["PIT_TTM:2025H1"]
    assert valuation.default_assumptions == ["tax_rate", "terminal_growth"]
    assert valuation.confidence_penalty > 0
    assert valuation.terminal_value_share > 0
