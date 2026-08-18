from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

import pytest
from pydantic import ValidationError

from moatrader.business.drivers import ValuationDriver, ValuationEvidenceRole
from moatrader.canonical.models import StatementType
from moatrader.evidence.research_reports import (
    MarketOpinion,
    ReportEvidenceItem,
    ResearchReportBundle,
)


AS_OF = datetime(2026, 8, 18, 9, tzinfo=timezone(timedelta(hours=9)))


def _item(*, fact: str = "2027 revenue growth estimate is 8%", available_at: datetime = AS_OF) -> ReportEvidenceItem:
    return ReportEvidenceItem(
        item_id="estimate-1",
        primary_driver=ValuationDriver.REVENUE_GROWTH,
        fact=fact,
        observable_anchor="Analyst model table, revenue forecast row",
        source_document_id="ANALYST:REPORT:1",
        source_chunk_id="chunk-1",
        node_ids=["page-3-table-1-row-5"],
        available_at=available_at,
        statement_type=StatementType.FORECAST,
        role=ValuationEvidenceRole.SCENARIO_INPUT,
        value=D("0.08"),
        unit="ratio",
        period="2027",
    )


def test_market_opinion_is_quarantined_from_intrinsic_driver_bundle() -> None:
    bundle = ResearchReportBundle(
        issuer_id="issuer-1",
        analyst_estimates=[_item()],
        market_opinion=[
            MarketOpinion(
                source_document_id="ANALYST:REPORT:1",
                available_at=AS_OF,
                recommendation="BUY",
                target_price=D("150000"),
                current_price=D("100000"),
            )
        ],
    )

    intrinsic = bundle.intrinsic_view(as_of=AS_OF)
    drivers = intrinsic.to_valuation_driver_bundle()

    assert "market_opinion" not in intrinsic.model_dump()
    assert len(drivers.evidence) == 1
    assert drivers.evidence[0].source_type.value == "ANALYST"
    assert drivers.evidence[0].numeric_adjustment_allowed is False


@pytest.mark.parametrize("fact", ["Target price is 150000", "투자의견 BUY", "BUY"])
def test_price_language_in_intrinsic_lane_fails_closed(fact: str) -> None:
    bundle = ResearchReportBundle(
        issuer_id="issuer-1",
        observed_facts=[_item(fact=fact)],
    )
    with pytest.raises(ValidationError, match="price/opinion leakage"):
        bundle.intrinsic_view(as_of=AS_OF)


def test_future_analyst_report_cannot_enter_pit_bundle() -> None:
    bundle = ResearchReportBundle(
        issuer_id="issuer-1",
        analyst_estimates=[_item(available_at=AS_OF + timedelta(seconds=1))],
    )
    with pytest.raises(ValidationError, match="future analyst evidence"):
        bundle.intrinsic_view(as_of=AS_OF)
