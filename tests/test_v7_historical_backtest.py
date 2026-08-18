from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from moatrader.backtest.historical import (
    compound_change_ratios,
    latest_revenue_continuity,
    latest_pit_filing_versions,
    quarterly_signal_dates,
    sample_statistics,
)


SEOUL = ZoneInfo("Asia/Seoul")


def test_quarterly_dates_stop_before_unobservable_2026_return_window() -> None:
    dates = quarterly_signal_dates(start=date(2020, 1, 1), end=date(2025, 9, 30))
    assert len(dates) == 23
    assert dates[0] == date(2020, 3, 31)
    assert dates[-1] == date(2025, 9, 30)


def test_pit_selection_does_not_backcast_later_amendment() -> None:
    original = {
        "rcept_no": "20200330000001",
        "fiscal_period_end": "2019-12-31",
        "available_at": "2020-03-30T23:59:59.999999+09:00",
        "version": "original",
    }
    amendment = {
        "rcept_no": "20200401000002",
        "fiscal_period_end": "2019-12-31",
        "available_at": "2020-04-01T23:59:59.999999+09:00",
        "version": "amendment",
    }

    march = latest_pit_filing_versions(
        [original, amendment], cutoff=datetime(2020, 3, 31, 23, 59, 59, tzinfo=SEOUL)
    )
    april = latest_pit_filing_versions(
        [original, amendment], cutoff=datetime(2020, 4, 2, tzinfo=SEOUL)
    )

    assert march[0]["version"] == "original"
    assert april[0]["version"] == "amendment"


def test_daily_krx_change_ratios_compound_without_price_level_lookahead() -> None:
    result = compound_change_ratios([Decimal("10"), Decimal("-5")])
    assert result == Decimal("0.045")


def test_sample_statistics_reports_event_level_t_stat_and_compound() -> None:
    result = sample_statistics([0.1, -0.05, 0.2])
    assert result["count"] == 3
    assert result["positive_rate"] == 2 / 3
    assert result["cumulative_compound"] is not None


def test_latest_revenue_continuity_rejects_as_filed_unit_explosion_without_future_correction() -> None:
    stable, ratio = latest_revenue_continuity(
        [
            (2018, {"revenue": Decimal("3000000000000")}),
            (2019, {"revenue": Decimal("3000000000000000000")}),
        ]
    )
    assert stable is False
    assert ratio == Decimal("1000000")
