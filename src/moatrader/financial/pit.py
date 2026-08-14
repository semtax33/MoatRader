from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, tzinfo
from decimal import Decimal
from typing import Mapping, Sequence


FIRST_QUARTER_REPORT_CODE = "11013"
SEMIANNUAL_REPORT_CODE = "11012"
THIRD_QUARTER_REPORT_CODE = "11014"
ANNUAL_REPORT_CODE = "11011"


@dataclass(frozen=True)
class FinancialReportPeriod:
    business_year: int
    report_code: str
    label: str
    period_end: date

    @property
    def is_annual(self) -> bool:
        return self.report_code == ANNUAL_REPORT_CODE


_REPORT_DEFINITIONS = {
    FIRST_QUARTER_REPORT_CODE: ("Q1", 3, 31),
    SEMIANNUAL_REPORT_CODE: ("H1", 6, 30),
    THIRD_QUARTER_REPORT_CODE: ("Q3", 9, 30),
    ANNUAL_REPORT_CODE: ("FY", 12, 31),
}


def financial_report_period(business_year: int, report_code: str) -> FinancialReportPeriod:
    try:
        label, month, day = _REPORT_DEFINITIONS[report_code]
    except KeyError:
        raise ValueError(f"unsupported OpenDART report code: {report_code}") from None
    return FinancialReportPeriod(
        business_year=business_year,
        report_code=report_code,
        label=f"{business_year}{label}",
        period_end=date(business_year, month, day),
    )


def candidate_financial_periods(
    as_of: datetime,
    *,
    lookback_years: int = 3,
) -> list[FinancialReportPeriod]:
    """Return calendar-year OpenDART periods that could have ended by ``as_of``.

    Actual point-in-time availability is decided from each response's receipt
    number. Period-end filtering only avoids requests for periods that cannot
    have ended yet.
    """

    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    if lookback_years < 1:
        raise ValueError("lookback_years must be positive")

    periods = [
        financial_report_period(year, report_code)
        for year in range(as_of.year, as_of.year - lookback_years, -1)
        for report_code in _REPORT_DEFINITIONS
    ]
    return sorted(
        (period for period in periods if period.period_end < as_of.date()),
        key=lambda period: period.period_end,
        reverse=True,
    )


def receipt_number(rows: Sequence[Mapping[str, object]]) -> str:
    values = {str(row.get("rcept_no") or "").strip() for row in rows}
    values.discard("")
    if len(values) != 1:
        raise ValueError(f"financial response must have one receipt number, got {sorted(values)}")
    value = next(iter(values))
    if len(value) != 14 or not value.isdigit():
        raise ValueError(f"invalid OpenDART receipt number: {value!r}")
    try:
        date.fromisoformat(f"{value[:4]}-{value[4:6]}-{value[6:8]}")
    except ValueError:
        raise ValueError(f"invalid date in OpenDART receipt number: {value!r}") from None
    return value


def conservative_receipt_available_at(value: str, zone: tzinfo) -> datetime:
    """Map a date-only DART receipt number to a conservative availability time.

    ``fnlttSinglAcntAll`` exposes a receipt date but not an intraday timestamp.
    The payload therefore becomes usable at the start of the following local
    day. This prevents a same-day close signal from accidentally seeing a
    filing that arrived after that close.
    """

    if len(value) != 14 or not value.isdigit():
        raise ValueError(f"invalid OpenDART receipt number: {value!r}")
    receipt_date = date.fromisoformat(f"{value[:4]}-{value[4:6]}-{value[6:8]}")
    return datetime.combine(receipt_date + timedelta(days=1), time.min, tzinfo=zone)


FLOW_METRICS = ("revenue", "ebit", "capex", "depreciation")
BALANCE_METRICS = ("cash", "debt", "nwc")


def trailing_twelve_month_metrics(
    prior_fiscal_year: Mapping[str, Decimal | None],
    current_year_to_date: Mapping[str, Decimal | None],
    prior_year_to_date: Mapping[str, Decimal | None],
    latest_balance_sheet: Mapping[str, Decimal | None],
) -> dict[str, Decimal | None]:
    """Build TTM flows and retain point-in-time balance-sheet values.

    Flow formula: prior FY + current YTD - prior-year YTD.
    Balance-sheet concepts are instants and therefore come directly from the
    latest available interim report.
    """

    result: dict[str, Decimal | None] = {}
    for key in FLOW_METRICS:
        prior_fy = prior_fiscal_year.get(key)
        current_ytd = current_year_to_date.get(key)
        prior_ytd = prior_year_to_date.get(key)
        if prior_fy is None or current_ytd is None or prior_ytd is None:
            result[key] = None
        else:
            result[key] = prior_fy + current_ytd - prior_ytd
    for key in BALANCE_METRICS:
        result[key] = latest_balance_sheet.get(key)
    return result
