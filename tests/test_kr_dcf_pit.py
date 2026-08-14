from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from moatrader.financial.pit import (
    ANNUAL_REPORT_CODE,
    SEMIANNUAL_REPORT_CODE,
    candidate_financial_periods,
    conservative_receipt_available_at,
    trailing_twelve_month_metrics,
)
from scripts.prepare_kr_dcf_manifest import (
    assumptions_from_history,
    build_pit_ttm_input,
    financial_metrics,
    latest_pit_financial_report,
)


KST = timezone(timedelta(hours=9))


class FakeDartClient:
    def __init__(self, payloads: dict[tuple[int, str, str], dict[str, object]]) -> None:
        self.payloads = payloads

    def json(self, _endpoint: str, params: dict[str, object]) -> dict[str, object]:
        key = (int(params["bsns_year"]), str(params["reprt_code"]), str(params["fs_div"]))
        return self.payloads.get(key, {"status": "013", "message": "no data"})


def row(
    *,
    receipt_no: str,
    report_code: str,
    year: int,
    statement: str,
    account_id: str,
    amount: int,
    accumulated: int | None = None,
) -> dict[str, object]:
    return {
        "rcept_no": receipt_no,
        "reprt_code": report_code,
        "bsns_year": str(year),
        "sj_div": statement,
        "account_id": account_id,
        "account_nm": account_id,
        "thstrm_amount": str(amount),
        "thstrm_add_amount": str(accumulated) if accumulated is not None else "",
    }


def report_rows(
    *,
    receipt_no: str,
    report_code: str,
    year: int,
    revenue: int,
    ebit: int,
    capex: int,
    depreciation: int,
    cash: int = 100,
    debt: int = 400,
) -> list[dict[str, object]]:
    interim = report_code != ANNUAL_REPORT_CODE
    quarterly_revenue = revenue // 2 if interim else revenue
    quarterly_ebit = ebit // 2 if interim else ebit
    quarterly_capex = capex // 2 if interim else capex
    quarterly_depreciation = depreciation // 2 if interim else depreciation
    accumulated = lambda value: value if interim else None
    return [
        row(
            receipt_no=receipt_no,
            report_code=report_code,
            year=year,
            statement="IS",
            account_id="ifrs-full_Revenue",
            amount=quarterly_revenue,
            accumulated=accumulated(revenue),
        ),
        row(
            receipt_no=receipt_no,
            report_code=report_code,
            year=year,
            statement="IS",
            account_id="OperatingIncomeLoss",
            amount=quarterly_ebit,
            accumulated=accumulated(ebit),
        ),
        row(
            receipt_no=receipt_no,
            report_code=report_code,
            year=year,
            statement="CF",
            account_id="PurchaseOfPropertyPlantAndEquipment",
            amount=quarterly_capex,
            accumulated=accumulated(capex),
        ),
        row(
            receipt_no=receipt_no,
            report_code=report_code,
            year=year,
            statement="CF",
            account_id="Depreciation",
            amount=quarterly_depreciation,
            accumulated=accumulated(depreciation),
        ),
        row(
            receipt_no=receipt_no,
            report_code=report_code,
            year=year,
            statement="BS",
            account_id="CashAndCashEquivalents",
            amount=cash,
        ),
        row(
            receipt_no=receipt_no,
            report_code=report_code,
            year=year,
            statement="BS",
            account_id="CurrentLoansReceived",
            amount=debt,
        ),
        row(
            receipt_no=receipt_no,
            report_code=report_code,
            year=year,
            statement="BS",
            account_id="TradeAndOtherCurrentReceivables",
            amount=150,
        ),
        row(
            receipt_no=receipt_no,
            report_code=report_code,
            year=year,
            statement="BS",
            account_id="Inventories",
            amount=80,
        ),
        row(
            receipt_no=receipt_no,
            report_code=report_code,
            year=year,
            statement="BS",
            account_id="TradeAndOtherCurrentPayables",
            amount=70,
        ),
    ]


def payload(rows: list[dict[str, object]]) -> dict[str, object]:
    return {"status": "000", "message": "ok", "list": rows}


def metrics(revenue: int, ebit: int, capex: int, depreciation: int) -> dict[str, Decimal | None]:
    return {
        "revenue": Decimal(revenue),
        "ebit": Decimal(ebit),
        "capex": Decimal(capex),
        "depreciation": Decimal(depreciation),
        "cash": Decimal(100),
        "debt": Decimal(400),
        "nwc": Decimal(160),
    }


def test_candidate_periods_include_unfiled_fiscal_year_for_pit_rejection() -> None:
    as_of = datetime(2026, 2, 28, 23, 59, tzinfo=KST)
    labels = [period.label for period in candidate_financial_periods(as_of)]
    assert labels[:2] == ["2025FY", "2025Q3"]


def test_receipt_date_becomes_available_on_following_local_day() -> None:
    available = conservative_receipt_available_at("20250814001234", KST)
    assert available == datetime(2025, 8, 15, 0, 0, tzinfo=KST)


def test_interim_metrics_use_explicit_accumulated_amounts() -> None:
    rows = report_rows(
        receipt_no="20250814001234",
        report_code=SEMIANNUAL_REPORT_CODE,
        year=2025,
        revenue=600,
        ebit=90,
        capex=60,
        depreciation=30,
    )
    interim = financial_metrics(rows, interim=True)
    annual_style = financial_metrics(rows, interim=False)
    assert interim["revenue"] == Decimal(600)
    assert interim["ebit"] == Decimal(90)
    assert interim["capex"] == Decimal(60)
    assert annual_style["revenue"] == Decimal(300)
    assert annual_style["capex"] == Decimal(30)


def test_interim_metrics_do_not_fall_back_to_three_month_amount() -> None:
    rows = report_rows(
        receipt_no="20250814001234",
        report_code=SEMIANNUAL_REPORT_CODE,
        year=2025,
        revenue=600,
        ebit=90,
        capex=60,
        depreciation=30,
    )
    for item in rows:
        if item["sj_div"] in {"IS", "CF"}:
            item["thstrm_add_amount"] = ""
    interim = financial_metrics(rows, interim=True)
    assert interim["revenue"] is None
    assert interim["ebit"] is None
    assert interim["capex"] is None


def test_ttm_formula_combines_flows_but_keeps_latest_balance_sheet() -> None:
    result = trailing_twelve_month_metrics(
        metrics(1000, 100, 100, 50),
        metrics(600, 90, 60, 30),
        metrics(500, 50, 40, 25),
        metrics(600, 90, 60, 30),
    )
    assert result == {
        "revenue": Decimal(1100),
        "ebit": Decimal(140),
        "capex": Decimal(120),
        "depreciation": Decimal(55),
        "cash": Decimal(100),
        "debt": Decimal(400),
        "nwc": Decimal(160),
    }


def test_latest_report_rejects_a_filing_received_after_cutoff(tmp_path: Path) -> None:
    client = FakeDartClient(
        {
            (2025, ANNUAL_REPORT_CODE, "CFS"): payload(
                report_rows(
                    receipt_no="20260320001234",
                    report_code=ANNUAL_REPORT_CODE,
                    year=2025,
                    revenue=1200,
                    ebit=120,
                    capex=100,
                    depreciation=50,
                )
            ),
            (2025, "11014", "CFS"): payload(
                report_rows(
                    receipt_no="20251114001234",
                    report_code="11014",
                    year=2025,
                    revenue=900,
                    ebit=90,
                    capex=75,
                    depreciation=40,
                )
            ),
        }
    )
    report, rejected = latest_pit_financial_report(
        client,  # type: ignore[arg-type]
        tmp_path,
        "000001",
        "00000001",
        datetime(2026, 2, 28, 23, 59, tzinfo=KST),
    )
    assert report.period.label == "2025Q3"
    assert [item["period_label"] for item in rejected] == ["2025FY"]


def test_build_pit_ttm_input_uses_same_scope_and_reports_formula(tmp_path: Path) -> None:
    client = FakeDartClient(
        {
            (2025, SEMIANNUAL_REPORT_CODE, "CFS"): payload(
                report_rows(
                    receipt_no="20250814001234",
                    report_code=SEMIANNUAL_REPORT_CODE,
                    year=2025,
                    revenue=600,
                    ebit=90,
                    capex=60,
                    depreciation=30,
                )
            ),
            (2024, ANNUAL_REPORT_CODE, "CFS"): payload(
                report_rows(
                    receipt_no="20250320001234",
                    report_code=ANNUAL_REPORT_CODE,
                    year=2024,
                    revenue=1000,
                    ebit=100,
                    capex=100,
                    depreciation=50,
                )
            ),
            (2024, SEMIANNUAL_REPORT_CODE, "CFS"): payload(
                report_rows(
                    receipt_no="20240814001234",
                    report_code=SEMIANNUAL_REPORT_CODE,
                    year=2024,
                    revenue=500,
                    ebit=50,
                    capex=40,
                    depreciation=25,
                )
            ),
        }
    )
    result, audit, latest = build_pit_ttm_input(
        client,  # type: ignore[arg-type]
        tmp_path,
        "000001",
        "00000001",
        datetime(2025, 8, 31, 23, 59, tzinfo=KST),
    )
    assert latest.period.label == "2025H1"
    assert result["revenue"] == Decimal(1100)
    assert result["ebit"] == Decimal(140)
    assert result["capex"] == Decimal(120)
    assert result["depreciation"] == Decimal(55)
    assert result["nwc"] == Decimal(160)
    assert audit["financial_period_basis"] == "TTM"
    assert audit["ttm_formula"] == "2024FY + 2025H1 YTD - 2024H1 YTD"
    assert audit["financial_statement_scope"] == "CFS"


def test_assumptions_use_ttm_as_base_without_treating_it_as_annual_growth() -> None:
    history = [
        (2022, metrics(900, 90, 80, 40)),
        (2023, metrics(1000, 100, 90, 45)),
        (2024, metrics(1050, 105, 100, 50)),
    ]
    assumptions, audit = assumptions_from_history(
        history,
        "MID",
        Decimal(10),
        base_metrics=metrics(1100, 132, 110, 55),
        base_period="2025H1",
        base_basis="TTM",
    )
    assert assumptions["base_revenue"] == "1100"
    assert assumptions["ebit_margin"][0] == "0.12"  # type: ignore[index]
    assert audit["base_financial_period"] == "2025H1"
    assert audit["base_financial_basis"] == "TTM"
    assert audit["history_years"] == [2022, 2023, 2024]
