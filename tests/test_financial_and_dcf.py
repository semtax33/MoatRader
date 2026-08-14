from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from moatrader.adapters import EdgarHtmlAdapter, RawDocument
from moatrader.financial import DcfAssumptions, DcfEngine, FinancialSnapshotBuilder


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

