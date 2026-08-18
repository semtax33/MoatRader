from __future__ import annotations

import math
import statistics
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, Iterable


class HistoricalSignalStatus(StrEnum):
    ELIGIBLE = "ELIGIBLE"
    EXCLUDED_ARCHETYPE = "EXCLUDED_ARCHETYPE"
    NOT_LISTED_OR_NO_PRICE = "NOT_LISTED_OR_NO_PRICE"
    NO_PIT_FINANCIALS = "NO_PIT_FINANCIALS"
    INSUFFICIENT_FINANCIAL_COVERAGE = "INSUFFICIENT_FINANCIAL_COVERAGE"
    FINANCIAL_DISCONTINUITY = "FINANCIAL_DISCONTINUITY"
    DCF_SCREENING_EXCLUSION = "DCF_SCREENING_EXCLUSION"
    VALUATION_ERROR = "VALUATION_ERROR"


def quarterly_signal_dates(*, start: date, end: date) -> list[date]:
    quarter_ends = ((3, 31), (6, 30), (9, 30), (12, 31))
    result = [
        date(year, month, day)
        for year in range(start.year, end.year + 1)
        for month, day in quarter_ends
        if start <= date(year, month, day) <= end
    ]
    if not result:
        raise ValueError("historical backtest requires at least one quarterly signal date")
    return result


def latest_pit_filing_versions(
    records: Iterable[dict[str, Any]],
    *,
    cutoff: datetime,
) -> list[dict[str, Any]]:
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("PIT filing cutoff must be timezone-aware")
    selected: dict[str, tuple[datetime, str, dict[str, Any]]] = {}
    for record in records:
        available_at = datetime.fromisoformat(str(record["available_at"]).replace("Z", "+00:00"))
        if available_at > cutoff:
            continue
        period_end = str(record["fiscal_period_end"])
        if date.fromisoformat(period_end) > cutoff.date():
            continue
        key = (available_at, str(record["rcept_no"]))
        current = selected.get(period_end)
        if current is None or key > current[:2]:
            selected[period_end] = (available_at, str(record["rcept_no"]), record)
    return [selected[key][2] for key in sorted(selected)]


def compound_change_ratios(values_percent: Iterable[Decimal | float | str]) -> Decimal:
    result = Decimal(1)
    observed = 0
    for value in values_percent:
        change = Decimal(str(value)) / Decimal(100)
        if change <= Decimal(-1):
            raise ValueError("daily price change ratio cannot be at or below -100%")
        result *= Decimal(1) + change
        observed += 1
    if observed == 0:
        raise ValueError("return window contains no daily changes")
    return result - Decimal(1)


def latest_revenue_continuity(
    history: Iterable[tuple[int, dict[str, Decimal | None]]],
    *,
    maximum_multiple: Decimal = Decimal(10),
) -> tuple[bool, Decimal | None]:
    revenues = [
        metrics["revenue"]
        for _year, metrics in history
        if metrics.get("revenue") is not None and metrics["revenue"] > 0
    ]
    if len(revenues) < 2:
        return True, None
    ratio = revenues[-1] / revenues[-2]
    return Decimal(1) / maximum_multiple <= ratio <= maximum_multiple, ratio


def sample_statistics(values: Iterable[float]) -> dict[str, float | int | None]:
    observed = [float(value) for value in values if math.isfinite(float(value))]
    if not observed:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "sample_std": None,
            "t_stat": None,
            "positive_rate": None,
            "cumulative_compound": None,
        }
    mean = statistics.fmean(observed)
    std = statistics.stdev(observed) if len(observed) > 1 else None
    t_stat = mean / (std / math.sqrt(len(observed))) if std and std > 0 else None
    compound = math.prod(1 + value for value in observed) - 1
    return {
        "count": len(observed),
        "mean": mean,
        "median": statistics.median(observed),
        "sample_std": std,
        "t_stat": t_stat,
        "positive_rate": sum(value > 0 for value in observed) / len(observed),
        "cumulative_compound": compound,
    }
