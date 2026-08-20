from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, time as datetime_time, timedelta, tzinfo
from decimal import Decimal, InvalidOperation
from pathlib import Path
from xml.etree import ElementTree

from moatrader.financial.pit import (
    ANNUAL_REPORT_CODE,
    FinancialReportPeriod,
    candidate_financial_periods,
    conservative_receipt_available_at,
    financial_report_period,
    receipt_number,
    trailing_twelve_month_metrics,
)


DART_API = "https://opendart.fss.or.kr/api"


@dataclass(frozen=True)
class FinancialReport:
    period: FinancialReportPeriod
    fs_div: str
    receipt_no: str
    available_at: datetime
    rows: list[dict[str, object]]
    payload_path: Path


def decimal_value(value: object) -> Decimal | None:
    text = str(value or "").replace(",", "").strip()
    if not text or text == "-":
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def clamp(value: Decimal, low: Decimal, high: Decimal) -> Decimal:
    return min(high, max(low, value))


class DartClient:
    def __init__(self, api_key: str, requests_per_second: float) -> None:
        self.api_key = api_key
        self.interval = 1.0 / requests_per_second
        self.last_request = 0.0

    def _wait(self) -> None:
        remaining = self.interval - (time.monotonic() - self.last_request)
        if remaining > 0:
            time.sleep(remaining)

    def get(self, endpoint: str, params: dict[str, object]) -> bytes:
        self._wait()
        query = urllib.parse.urlencode({"crtfc_key": self.api_key, **params})
        request = urllib.request.Request(
            f"{DART_API}/{endpoint}?{query}",
            headers={"User-Agent": "MoatRader DCF preparation", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"OpenDART request failed for {endpoint}: {type(exc).__name__}") from None
        finally:
            self.last_request = time.monotonic()
        return body

    def json(self, endpoint: str, params: dict[str, object]) -> dict[str, object]:
        body = self.get(endpoint, params)
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RuntimeError(f"OpenDART returned invalid JSON for {endpoint}") from None


def corporation_map(client: DartClient, output_root: Path) -> dict[str, str]:
    archive_path = output_root / "source" / "corpCode.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.is_file():
        archive = archive_path.read_bytes()
    else:
        archive = client.get("corpCode.xml", {})
        archive_path.write_bytes(archive)
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        xml_names = [name for name in source.namelist() if name.lower().endswith(".xml")]
        if not xml_names:
            raise RuntimeError("OpenDART corporation archive has no XML member")
        root = ElementTree.fromstring(source.read(xml_names[0]))
    result: dict[str, str] = {}
    for item in root.findall(".//list"):
        stock_code = (item.findtext("stock_code") or "").strip()
        corp_code = (item.findtext("corp_code") or "").strip()
        if re.fullmatch(r"\d{6}", stock_code) and re.fullmatch(r"\d{8}", corp_code):
            result[stock_code] = corp_code
    return result


def row_decimal(row: dict[str, object], fields: tuple[str, ...]) -> Decimal | None:
    for field in fields:
        value = decimal_value(row.get(field))
        if value is not None:
            return value
    return None


def amount(
    rows: list[dict[str, object]],
    pattern: str,
    statements: set[str],
    *,
    fields: tuple[str, ...] = ("thstrm_amount",),
) -> Decimal | None:
    regex = re.compile(pattern, re.IGNORECASE)
    values = [
        row_decimal(row, fields)
        for row in rows
        if str(row.get("sj_div") or "") in statements
        and regex.search(f"{row.get('account_id', '')} {row.get('account_nm', '')}")
    ]
    clean = [value for value in values if value is not None]
    return max(clean, key=abs) if clean else None


def sum_amounts(
    rows: list[dict[str, object]],
    pattern: str,
    statements: set[str],
    *,
    fields: tuple[str, ...] = ("thstrm_amount",),
) -> Decimal | None:
    regex = re.compile(pattern, re.IGNORECASE)
    values: list[Decimal] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        account_id = str(row.get("account_id") or "")
        account_name = str(row.get("account_nm") or "")
        if str(row.get("sj_div") or "") not in statements or not regex.search(f"{account_id} {account_name}"):
            continue
        key = (account_id, account_name)
        value = row_decimal(row, fields)
        if value is not None and key not in seen:
            values.append(abs(value))
            seen.add(key)
    return sum(values, Decimal(0)) if values else None


def fetch_report(
    client: DartClient,
    output_root: Path,
    ticker: str,
    corp_code: str,
    period: FinancialReportPeriod,
    *,
    timezone: tzinfo,
    required_fs_div: str | None = None,
) -> FinancialReport | None:
    ticker_dir = output_root / "source" / "financials" / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    fs_divisions = (required_fs_div,) if required_fs_div else ("CFS", "OFS")
    for fs_div in fs_divisions:
        path = ticker_dir / f"{period.business_year}-{period.report_code}-{fs_div}.json"
        legacy_path = ticker_dir / f"{period.business_year}-{fs_div}.json"
        if period.is_annual and not path.is_file() and legacy_path.is_file():
            path = legacy_path
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = client.json(
                "fnlttSinglAcntAll.json",
                {
                    "corp_code": corp_code,
                    "bsns_year": period.business_year,
                    "reprt_code": period.report_code,
                    "fs_div": fs_div,
                },
            )
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if str(payload.get("status")) == "000" and payload.get("list"):
            rows = list(payload["list"])
            response_years = {str(row.get("bsns_year") or "").strip() for row in rows} - {""}
            response_codes = {str(row.get("reprt_code") or "").strip() for row in rows} - {""}
            if response_years != {str(period.business_year)} or response_codes != {period.report_code}:
                raise ValueError(
                    f"OpenDART response period mismatch for {ticker}: "
                    f"expected {period.label}, got years={sorted(response_years)} codes={sorted(response_codes)}"
                )
            number = receipt_number(rows)
            available_at = conservative_receipt_available_at(number, timezone)
            return FinancialReport(
                period=period,
                fs_div=fs_div,
                receipt_no=number,
                available_at=available_at,
                rows=rows,
                payload_path=path,
            )
        if str(payload.get("status")) not in {"013"}:
            break
    return None


def financial_metrics(
    rows: list[dict[str, object]],
    *,
    interim: bool,
    annual_fields: tuple[str, ...] | None = None,
) -> dict[str, Decimal | None]:
    # OpenDART documents ``thstrm_amount`` as the three-month value for
    # interim income statements. TTM construction must therefore fail closed
    # when the explicit cumulative field is absent instead of mixing a quarter
    # with year-to-date cash-flow values.
    if interim and annual_fields is not None:
        raise ValueError("comparative annual fields cannot be used for interim metrics")
    flow_fields = annual_fields or (("thstrm_add_amount",) if interim else ("thstrm_amount",))
    balance_fields = annual_fields or ("thstrm_amount",)
    revenue = amount(
        rows,
        r"ifrs(?:-full)?_(?:Revenue|SalesRevenue)|dart_Revenue|(?:^|\s)(?:수익\(매출액\)|매출액)(?:$|\s)",
        {"IS", "CIS"},
        fields=flow_fields,
    )
    ebit = amount(
        rows,
        r"OperatingIncomeLoss|ProfitLossFromOperatingActivities|영업이익(?:\(손실\))?",
        {"IS", "CIS"},
        fields=flow_fields,
    )
    capex = sum_amounts(
        rows,
        r"PurchaseOfPropertyPlantAndEquipment|PurchaseOfIntangibleAssets|유형자산의 취득|무형자산의 취득|건설중인자산의 취득",
        {"CF"},
        fields=flow_fields,
    )
    depreciation = sum_amounts(
        rows,
        r"Depreciation|Amortisation|Amortization|감가상각비|무형자산상각비|사용권자산상각비",
        {"CF"},
        fields=flow_fields,
    )
    cash = amount(
        rows,
        r"CashAndCashEquivalents|^\s*현금및현금성자산\s*$",
        {"BS"},
        fields=balance_fields,
    ) or Decimal(0)
    debt = sum_amounts(
        rows,
        r"(?:^|\s)(?:단기차입금|장기차입금|유동성장기차입금|유동성사채|사채|전환사채|신주인수권부사채|유동\s*리스부채|비유동\s*리스부채)(?:$|\s)|"
        r"CurrentLoansReceived|NoncurrentLoansReceived|Borrowings|CurrentLeaseLiabilities|NoncurrentLeaseLiabilities",
        {"BS"},
        fields=balance_fields,
    )
    receivables = amount(
        rows,
        r"TradeAndOtherCurrentReceivables|매출채권 및 기타유동채권|^\s*매출채권\s*$",
        {"BS"},
        fields=balance_fields,
    ) or Decimal(0)
    inventory = amount(
        rows,
        r"Inventories|유동재고자산|^\s*재고자산\s*$",
        {"BS"},
        fields=balance_fields,
    ) or Decimal(0)
    payables = amount(
        rows,
        r"TradeAndOtherCurrentPayables|매입채무 및 기타유동채무|^\s*매입채무\s*$",
        {"BS"},
        fields=balance_fields,
    ) or Decimal(0)
    return {
        "revenue": revenue,
        "ebit": ebit,
        "capex": capex,
        "depreciation": depreciation,
        "cash": cash,
        "debt": debt,
        "nwc": receivables + inventory - payables,
    }


def annual_metrics(
    rows: list[dict[str, object]],
    *,
    fields: tuple[str, ...] = ("thstrm_amount",),
) -> dict[str, Decimal | None]:
    return financial_metrics(rows, interim=False, annual_fields=fields)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_descriptor(report: FinancialReport) -> dict[str, object]:
    return {
        "business_year": report.period.business_year,
        "report_code": report.period.report_code,
        "period_label": report.period.label,
        "period_end": report.period.period_end.isoformat(),
        "receipt_no": report.receipt_no,
        "available_at": report.available_at.isoformat(),
        "fs_div": report.fs_div,
        "payload_path": str(report.payload_path.resolve()),
        "payload_sha256": file_sha256(report.payload_path),
    }


def latest_pit_financial_report(
    client: DartClient,
    output_root: Path,
    ticker: str,
    corp_code: str,
    as_of: datetime,
) -> tuple[FinancialReport, list[dict[str, object]]]:
    rejected_future: list[dict[str, object]] = []
    for period in candidate_financial_periods(as_of):
        report = fetch_report(
            client,
            output_root,
            ticker,
            corp_code,
            period,
            timezone=as_of.tzinfo,
        )
        if report is None:
            continue
        if report.available_at > as_of:
            rejected_future.append(report_descriptor(report))
            continue
        return report, rejected_future
    raise ValueError("no point-in-time financial report available")


def exact_pit_financial_report(
    client: DartClient,
    output_root: Path,
    ticker: str,
    corp_code: str,
    period: FinancialReportPeriod,
    *,
    as_of: datetime,
    fs_div: str,
) -> FinancialReport:
    report = fetch_report(
        client,
        output_root,
        ticker,
        corp_code,
        period,
        timezone=as_of.tzinfo,
        required_fs_div=fs_div,
    )
    if report is None:
        raise ValueError(f"missing {period.label} {fs_div} financial report")
    if report.available_at > as_of:
        raise ValueError(
            f"{period.label} {fs_div} financial report was not available at cutoff "
            f"({report.available_at.isoformat()} > {as_of.isoformat()})"
        )
    return report


def build_pit_ttm_input(
    client: DartClient,
    output_root: Path,
    ticker: str,
    corp_code: str,
    as_of: datetime,
) -> tuple[dict[str, Decimal | None], dict[str, object], FinancialReport]:
    latest, rejected_future = latest_pit_financial_report(
        client,
        output_root,
        ticker,
        corp_code,
        as_of,
    )
    current = financial_metrics(latest.rows, interim=not latest.period.is_annual)
    components = [latest]
    if latest.period.is_annual:
        metrics = current
        formula = f"{latest.period.label}"
    else:
        prior_year = latest.period.business_year - 1
        prior_annual = exact_pit_financial_report(
            client,
            output_root,
            ticker,
            corp_code,
            financial_report_period(prior_year, ANNUAL_REPORT_CODE),
            as_of=as_of,
            fs_div=latest.fs_div,
        )
        prior_interim = exact_pit_financial_report(
            client,
            output_root,
            ticker,
            corp_code,
            financial_report_period(prior_year, latest.period.report_code),
            as_of=as_of,
            fs_div=latest.fs_div,
        )
        prior_fy_metrics = annual_metrics(prior_annual.rows)
        prior_ytd_metrics = financial_metrics(prior_interim.rows, interim=True)
        metrics = trailing_twelve_month_metrics(
            prior_fy_metrics,
            current,
            prior_ytd_metrics,
            current,
        )
        components.extend([prior_annual, prior_interim])
        formula = (
            f"{prior_annual.period.label} + {latest.period.label} YTD "
            f"- {prior_interim.period.label} YTD"
        )

    if metrics.get("revenue") is None or metrics["revenue"] <= 0:
        raise ValueError(f"no positive PIT TTM revenue for {latest.period.label}")

    audit = {
        "financial_data_cutoff": as_of.isoformat(),
        "financial_period_basis": "FY" if latest.period.is_annual else "TTM",
        "balance_sheet_basis": latest.period.label,
        "latest_report_period": latest.period.label,
        "latest_report_code": latest.period.report_code,
        "latest_report_receipt_no": latest.receipt_no,
        "latest_report_available_at": latest.available_at.isoformat(),
        "financial_statement_scope": latest.fs_div,
        "ttm_formula": formula,
        "ttm_revenue": str(metrics.get("revenue")),
        "ttm_ebit": str(metrics.get("ebit")),
        "ttm_capex": str(metrics.get("capex")),
        "ttm_depreciation": str(metrics.get("depreciation")),
        "ttm_nwc": str(metrics.get("nwc")),
        "ttm_component_receipts": ";".join(report.receipt_no for report in components),
        "rejected_future_financial_reports": json.dumps(
            rejected_future,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "calendar_fiscal_year_assumption": True,
    }
    audit["components"] = [report_descriptor(report) for report in components]
    return metrics, audit, latest


def pit_annual_history(
    client: DartClient,
    output_root: Path,
    ticker: str,
    corp_code: str,
    years: list[int],
    *,
    as_of: datetime,
    fs_div: str,
) -> tuple[list[tuple[int, dict[str, Decimal | None]]], list[dict[str, object]]]:
    by_year: dict[int, tuple[dict[str, Decimal | None], dict[str, object]]] = {}
    requested = set(years)
    for year in years:
        report = fetch_report(
            client,
            output_root,
            ticker,
            corp_code,
            financial_report_period(year, ANNUAL_REPORT_CODE),
            timezone=as_of.tzinfo,
            required_fs_div=fs_div,
        )
        if report is None or report.available_at > as_of:
            continue
        descriptor = report_descriptor(report)
        for observation_year, amount_field in (
            (year - 2, "bfefrmtrm_amount"),
            (year - 1, "frmtrm_amount"),
            (year, "thstrm_amount"),
        ):
            if observation_year not in requested:
                continue
            metrics = annual_metrics(report.rows, fields=(amount_field,))
            if metrics.get("revenue") is None or metrics["revenue"] <= 0:
                continue
            source = {
                **descriptor,
                "observation_year": observation_year,
                "amount_field": amount_field,
            }
            # Later filings win so restated comparative values are preserved.
            by_year[observation_year] = (metrics, source)
    history = [(year, by_year[year][0]) for year in sorted(by_year)]
    sources = [by_year[year][1] for year in sorted(by_year)]
    return history, sources


def json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assumptions_from_history(
    history: list[tuple[int, dict[str, Decimal | None]]],
    size_bucket: str,
    shares: Decimal,
    *,
    base_metrics: dict[str, Decimal | None] | None = None,
    base_period: str | None = None,
    base_basis: str = "FY",
) -> tuple[dict[str, object], dict[str, object]]:
    valid = [(year, metrics) for year, metrics in history if metrics["revenue"] and metrics["revenue"] > 0]
    if not valid:
        raise ValueError("no positive annual revenue")
    latest_year, latest_annual = valid[-1]
    latest = base_metrics or latest_annual
    revenue = latest.get("revenue")
    if revenue is None or revenue <= 0:
        raise ValueError("no positive PIT base revenue")
    assert revenue is not None

    growth_rates: list[Decimal] = []
    for (_prior_year, prior), (_year, current) in zip(valid, valid[1:]):
        if prior["revenue"] and current["revenue"] and prior["revenue"] > 0:
            growth_rates.append(current["revenue"] / prior["revenue"] - Decimal(1))
    normalized_growth = clamp(
        Decimal(str(statistics.median(growth_rates))) if growth_rates else Decimal("0.02"),
        Decimal("-0.10"),
        Decimal("0.15"),
    )
    terminal_growth = Decimal("0.02")
    growth_forecast = [
        normalized_growth + (terminal_growth - normalized_growth) * Decimal(step) / Decimal(4)
        for step in range(5)
    ]

    ratio_observations = [metrics for _year, metrics in valid]
    if base_metrics is not None and base_period != f"{latest_year}FY":
        ratio_observations.append(base_metrics)
    margins = [
        metrics["ebit"] / metrics["revenue"]
        for metrics in ratio_observations
        if metrics["ebit"] is not None and metrics["revenue"]
    ]
    current_margin = clamp(
        latest["ebit"] / revenue if latest["ebit"] is not None else Decimal(0),
        Decimal("-0.20"),
        Decimal("0.35"),
    )
    normalized_margin = clamp(
        Decimal(str(statistics.median(margins))) if margins else current_margin,
        Decimal("-0.20"),
        Decimal("0.35"),
    )
    margin_forecast = [
        current_margin + (normalized_margin - current_margin) * Decimal(step) / Decimal(4)
        for step in range(5)
    ]

    capex_ratios = [
        metrics["capex"] / metrics["revenue"]
        for metrics in ratio_observations
        if metrics["capex"] is not None and metrics["revenue"]
    ]
    capex_ratio = clamp(
        Decimal(str(statistics.median(capex_ratios))) if capex_ratios else Decimal("0.03"),
        Decimal("0.01"),
        Decimal("0.15"),
    )
    depreciation_ratio = clamp(
        latest["depreciation"] / revenue if latest["depreciation"] else capex_ratio * Decimal("0.70"),
        Decimal("0.005"),
        Decimal("0.12"),
    )
    nwc_ratio = clamp(
        latest["nwc"] / revenue if latest["nwc"] is not None else Decimal(0),
        Decimal(0),
        Decimal("0.35"),
    )
    wacc = {"SMALL": Decimal("0.12"), "MID": Decimal("0.105"), "LARGE": Decimal("0.095")}.get(
        size_bucket.upper(), Decimal("0.105")
    )
    net_debt = (latest["debt"] or Decimal(0)) - (latest["cash"] or Decimal(0))

    def texts(values: list[Decimal]) -> list[str]:
        return [format(value, "f") for value in values]

    history_source = "ANNUAL_HISTORY:" + ",".join(str(year) for year, _metrics in valid)
    base_source = f"PIT_{base_basis}:{base_period or f'{latest_year}FY'}"
    assumptions: dict[str, object] = {
        "method": "FCFF",
        "base_period": base_period or f"{latest_year}FY",
        "base_revenue": format(revenue, "f"),
        "revenue_growth": texts(growth_forecast),
        "ebit_margin": texts(margin_forecast),
        "tax_rate": "0.24",
        "depreciation_pct_revenue": format(depreciation_ratio, "f"),
        "capex_pct_revenue": format(capex_ratio, "f"),
        "nwc_pct_revenue": format(nwc_ratio, "f"),
        "wacc": format(wacc, "f"),
        "terminal_growth": format(terminal_growth, "f"),
        "net_debt": format(net_debt, "f"),
        "diluted_shares": format(shares, "f"),
        "assumption_sources": {
            "base_revenue": [base_source],
            "revenue_growth": [history_source, "HISTORICAL_MEDIAN_WITH_LINEAR_FADE"],
            "ebit_margin": [base_source, history_source, "HISTORICAL_MEDIAN_WITH_LINEAR_FADE"],
            "tax_rate": ["POLICY_DEFAULT:KOREA_BASELINE_24_PERCENT"],
            "depreciation_pct_revenue": [base_source, history_source],
            "capex_pct_revenue": [history_source],
            "nwc_pct_revenue": [base_source],
            "wacc": [f"UNIVERSE_SIZE_BUCKET:{size_bucket.upper() or 'MID'}"],
            "terminal_growth": ["POLICY_DEFAULT:LONG_RUN_GROWTH_2_PERCENT"],
            "net_debt": [base_source],
            "diluted_shares": ["PIT_KRX_LISTED_SHARES_NOT_FULLY_DILUTED"],
        },
        "assumption_types": {
            "base_revenue": "DETERMINISTIC",
            "revenue_growth": "MODEL_INFERENCE",
            "ebit_margin": "MODEL_INFERENCE",
            "tax_rate": "DEFAULT",
            "depreciation_pct_revenue": "MODEL_INFERENCE",
            "capex_pct_revenue": "MODEL_INFERENCE",
            "nwc_pct_revenue": "MODEL_INFERENCE",
            "wacc": "DETERMINISTIC",
            "terminal_growth": "DEFAULT",
            "net_debt": "DETERMINISTIC",
            "diluted_shares": "DETERMINISTIC",
        },
        "provenance_warnings": [
            "Revenue growth is a historical-median heuristic with a linear fade, not company guidance.",
            "EBIT margin is a historical normalization heuristic, not a forward operating model.",
            "Tax rate and terminal growth are policy defaults.",
            "Per-share value uses point-in-time KRX listed shares; potential options/convertibles require a separate dilution bridge.",
        ],
    }
    audit = {
        "latest_financial_year": latest_year,
        "base_financial_period": base_period or f"{latest_year}FY",
        "base_financial_basis": base_basis,
        "history_years": [year for year, _metrics in valid],
        "historical_revenue": [str(metrics["revenue"]) for _year, metrics in valid],
        "historical_ebit": [str(metrics["ebit"]) for _year, metrics in valid],
        "base_revenue": str(revenue),
        "base_ebit": str(latest.get("ebit")),
        "normalized_growth": str(normalized_growth),
        "current_ebit_margin": str(current_margin),
        "normalized_ebit_margin": str(normalized_margin),
        "capex_pct_revenue": str(capex_ratio),
        "depreciation_pct_revenue": str(depreciation_ratio),
        "nwc_pct_revenue": str(nwc_ratio),
        "tax_rate": "0.24",
        "tax_rate_method": "fixed-baseline/1",
        "wacc": str(wacc),
        "wacc_method": "size-bucket-baseline/1",
        "net_debt": str(net_debt),
        "method": "pit-ttm-historical-normalization/2",
        "assumption_types": json.dumps(assumptions["assumption_types"], sort_keys=True),
        "default_assumption_count": 2,
    }
    return assumptions, audit


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def canonical_manifest_rows(
    path: Path,
    *,
    as_of: datetime,
) -> dict[str, list[dict[str, str]]]:
    """Select the latest canonical annual and interim filing per ticker.

    Selection is based on reporting period first and filing availability
    second.  A late correction to an old report can therefore update that
    period but cannot displace a newer interim/annual period.
    """
    base = path.parent
    candidates: dict[str, list[tuple[str, datetime, datetime, dict[str, str]]]] = {}
    for row in read_csv(path):
        metadata_path = Path(row["metadata"])
        if not metadata_path.is_absolute():
            metadata_path = (base / metadata_path).resolve()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        available = datetime.fromisoformat(str(metadata["available_at"]).replace("Z", "+00:00"))
        if available > as_of:
            continue
        ticker = row["ticker"].zfill(6)
        report_name = str(metadata.get("report_name") or metadata.get("title") or "")
        form_type = str(metadata.get("form_type") or "")
        label = f"{report_name} {form_type}".casefold()
        if re.search(r"사업보고서|annual", label):
            kind = "ANNUAL"
        elif re.search(r"반기보고서|분기보고서|semiannual|quarter", label):
            kind = "INTERIM"
        else:
            # Non-periodic filings must not replace periodic business reports
            # in a structural MOAT/DCF run.
            continue
        period_text = str(metadata.get("period_end") or metadata.get("report_date") or "")[:10]
        try:
            period_at = datetime.fromisoformat(period_text).replace(tzinfo=as_of.tzinfo)
        except ValueError:
            continue
        copy = dict(row)
        input_path = Path(copy["input"])
        copy["input"] = str((base / input_path).resolve() if not input_path.is_absolute() else input_path)
        copy["metadata"] = str(metadata_path)
        copy["selection_report_kind"] = kind
        copy["selection_period_end"] = period_text
        copy["selection_is_amendment"] = str(
            bool(metadata.get("is_amendment")) or bool(re.search(r"기재정정|첨부정정|정정", report_name))
        )
        candidates.setdefault(ticker, []).append((kind, period_at, available, copy))

    result: dict[str, list[dict[str, str]]] = {}
    for ticker, values in candidates.items():
        selected_rows = []
        for kind in ("ANNUAL", "INTERIM"):
            matching = [item for item in values if item[0] == kind]
            if matching:
                # Later period wins; latest available version wins within it.
                selected_rows.append(max(matching, key=lambda item: (item[1], item[2]))[3])
        if selected_rows:
            result[ticker] = selected_rows
    return result


# Backward-compatible name for callers/tests; the value is now a list because
# a company can carry both its latest annual and latest interim filing.
latest_manifest_rows = canonical_manifest_rows


def resolve_filing_ticker(
    security_ticker: str,
    filing_by_stock: dict[str, list[dict[str, str]]],
) -> tuple[str, str]:
    """Resolve a listed security class to the issuer's periodic filing ticker."""
    if filing_by_stock.get(security_ticker):
        return security_ticker, "DIRECT"
    if len(security_ticker) == 6 and security_ticker[-1] != "0":
        common_candidate = f"{security_ticker[:5]}0"
        if filing_by_stock.get(common_candidate):
            return common_candidate, "SECURITY_CLASS_TO_COMMON_ISSUER"
    return security_ticker, "UNRESOLVED"


def historical_market_snapshot(
    ticker: str,
    as_of: datetime,
) -> tuple[Decimal, Decimal, datetime, str]:
    from pykrx import stock as krx_stock

    end_date = as_of.date()
    start_date = end_date - timedelta(days=14)
    start = start_date.strftime("%Y%m%d")
    end = end_date.strftime("%Y%m%d")
    ohlcv = krx_stock.get_market_ohlcv(start, end, ticker)
    if ohlcv is None or ohlcv.empty:
        raise ValueError(f"no KRX OHLCV on or before {as_of.date()} for {ticker}")
    ohlcv = ohlcv[ohlcv.index.date <= end_date].sort_index()
    if ohlcv.empty:
        raise ValueError(f"no KRX trading day on or before {as_of.date()} for {ticker}")
    market_day = ohlcv.index[-1].date()
    price = Decimal(str(int(ohlcv.iloc[-1]["종가"])))

    shares: Decimal | None = None
    source = "KRX_OHLCV+KRX_PIT_LISTED_SHARES"
    cap = krx_stock.get_market_cap(start, end, ticker)
    if cap is not None and not cap.empty:
        cap = cap[cap.index.date <= market_day].sort_index()
        if not cap.empty and Decimal(str(int(cap.iloc[-1]["상장주식수"]))) > 0:
            shares = Decimal(str(int(cap.iloc[-1]["상장주식수"])))
            source = "KRX_OHLCV+KRX_PIT_LISTED_SHARES"

    if shares is None:
        raise ValueError(
            f"no point-in-time KRX listed-share count for {ticker}; "
            "stale universe shares are not permitted for per-share DCF"
        )

    price_at = datetime.combine(
        market_day,
        datetime_time(hour=16),
        tzinfo=as_of.tzinfo,
    )
    return price, shares, price_at, source


def market_snapshot_from_universe_row(
    row: dict[str, str],
    *,
    as_of: datetime,
) -> tuple[Decimal, Decimal, datetime, str] | None:
    """Use a pinned PIT market snapshot when the universe provides one."""

    fields = ("current_price", "listed_shares", "price_as_of")
    populated = [bool(str(row.get(field, "")).strip()) for field in fields]
    if not any(populated):
        return None
    if not all(populated):
        raise ValueError("universe PIT market snapshot is incomplete")

    price = decimal_value(row["current_price"])
    shares = decimal_value(row["listed_shares"])
    if price is None or price <= 0:
        raise ValueError("universe current_price must be positive")
    if shares is None or shares <= 0:
        raise ValueError("universe listed_shares must be positive")
    price_at = datetime.fromisoformat(str(row["price_as_of"]).replace("Z", "+00:00"))
    if price_at.tzinfo is None or price_at.utcoffset() is None:
        raise ValueError("universe price_as_of must include a timezone offset")
    if price_at > as_of:
        raise ValueError("universe price_as_of cannot be later than the research cutoff")
    source = str(row.get("price_source") or "UNIVERSE_PINNED_PIT_MARKET_SNAPSHOT")
    return price, shares, price_at, source


def write_csv(path: Path, rows: list[dict[str, object]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", required=True, type=Path)
    parser.add_argument("--collected-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--year", action="append", type=int, dest="years")
    parser.add_argument("--requests-per-second", type=float, default=2.0)
    parser.add_argument("--api-key-env", default="DART_API_KEY")
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"missing environment variable {args.api_key_env}")
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    client = DartClient(api_key, args.requests_per_second)
    corp_by_stock = corporation_map(client, output_root)
    universe = read_csv(args.universe.resolve())
    as_of = datetime.fromisoformat(args.as_of.replace("Z", "+00:00"))
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("--as-of must include a timezone offset")
    filing_by_stock = canonical_manifest_rows(
        args.collected_manifest.resolve(),
        as_of=as_of,
    )
    requested_years = sorted(set(args.years)) if args.years else None

    manifest_rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    assumptions_dir = output_root / "assumptions"
    assumptions_dir.mkdir(parents=True, exist_ok=True)
    dcf_inputs_dir = output_root / "dcf-inputs"
    dcf_inputs_dir.mkdir(parents=True, exist_ok=True)

    for index, source in enumerate(universe, start=1):
        ticker = source["stock_code"].zfill(6)
        print(f"[{index}/{len(universe)}] {ticker}", flush=True)
        filing_ticker, security_mapping_method = resolve_filing_ticker(
            ticker,
            filing_by_stock,
        )
        filings = filing_by_stock.get(filing_ticker, [])
        if not filings:
            exclusions.append({"stock_code": ticker, "name": source.get("name", ""), "reason": "NO_PERIODIC_PIT_FILING"})
            continue
        filing = max(filings, key=lambda item: (item.get("selection_period_end", ""), item.get("selection_report_kind", "")))
        corp_code = corp_by_stock.get(filing_ticker) or filing.get("issuer_id", "")
        try:
            market_snapshot = market_snapshot_from_universe_row(source, as_of=as_of)
            price, diluted_shares, price_as_of, price_source = (
                market_snapshot
                if market_snapshot is not None
                else historical_market_snapshot(ticker, as_of)
            )
        except ValueError as exc:
            exclusions.append({"stock_code": ticker, "name": source.get("name", ""), "reason": str(exc)})
            continue

        dcf_path = ""
        dcf_input_path_text = ""
        dcf_input_hash = ""
        audit: dict[str, object] = {
            "stock_code": ticker,
            "name": source.get("name", ""),
            "corp_code": corp_code,
            "current_price": str(price),
            "price_as_of": price_as_of.isoformat(),
            "price_source": price_source,
            "diluted_shares": str(diluted_shares),
            "filing_ticker": filing_ticker,
            "security_mapping_method": security_mapping_method,
        }
        name = source.get("name", "")
        special_valuation = (
            filing_ticker != ticker
            or source.get("finance_hint", "").strip().lower() == "true"
            or source.get("holding_hint", "").strip().lower() == "true"
            or source.get("security_type", "COMMON").strip().upper() != "COMMON"
            or bool(re.search(r"리츠|REIT|리얼티|인프라", name, re.IGNORECASE))
        )
        if special_valuation:
            reason = (
                "NON_COMMON_SECURITY_DCF_MODEL_MISMATCH"
                if filing_ticker != ticker
                else "SPECIAL_COMPANY_DCF_MODEL_MISMATCH"
            )
            exclusions.append({"stock_code": ticker, "name": name, "reason": reason})
        else:
            try:
                base_metrics, pit_details, latest_report = build_pit_ttm_input(
                    client,
                    output_root,
                    ticker,
                    corp_code,
                    as_of,
                )
                latest_completed_year = (
                    latest_report.period.business_year
                    if latest_report.period.is_annual
                    else latest_report.period.business_year - 1
                )
                years = requested_years or list(
                    range(latest_completed_year - 6, latest_completed_year + 1)
                )
                history, annual_sources = pit_annual_history(
                    client,
                    output_root,
                    ticker,
                    corp_code,
                    years,
                    as_of=as_of,
                    fs_div=latest_report.fs_div,
                )
                assumptions, details = assumptions_from_history(
                    history,
                    source.get("size_bucket", ""),
                    diluted_shares,
                    base_metrics=base_metrics,
                    base_period=latest_report.period.label,
                    base_basis=str(pit_details["financial_period_basis"]),
                )
                input_payload = {
                    "schema_version": "moatrader-dcf-input/3",
                    "ticker": ticker,
                    "issuer_name": source.get("name", ""),
                    "as_of": as_of.isoformat(),
                    "metrics": {key: str(value) if value is not None else None for key, value in base_metrics.items()},
                    "pit": pit_details,
                    "annual_history": [
                        {
                            "year": year,
                            "metrics": {
                                key: str(value) if value is not None else None
                                for key, value in metrics.items()
                            },
                        }
                        for year, metrics in history
                    ],
                    "annual_sources": annual_sources,
                    "assumption_method": details["method"],
                    "assumptions": assumptions,
                }
                input_hash = json_sha256(input_payload)
                input_payload["input_sha256"] = input_hash
                dcf_input_path = dcf_inputs_dir / f"{ticker}.json"
                dcf_input_path.write_text(
                    json.dumps(input_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                dcf_input_path_text = str(dcf_input_path)
                dcf_input_hash = input_hash
                assumption_path = assumptions_dir / f"{ticker}.json"
                assumption_path.write_text(json.dumps(assumptions, ensure_ascii=False, indent=2), encoding="utf-8")
                dcf_path = str(assumption_path)
                csv_pit_details = {key: value for key, value in pit_details.items() if key != "components"}
                audit.update(csv_pit_details)
                audit.update(details)
                audit["annual_history_sources"] = json.dumps(
                    annual_sources,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                audit["dcf_input_path"] = str(dcf_input_path)
                audit["dcf_input_sha256"] = input_hash
            except ValueError as exc:
                exclusions.append({"stock_code": ticker, "name": source.get("name", ""), "reason": str(exc)})

        for selected_filing in filings:
            manifest_rows.append(
                {
                    **selected_filing,
                    "ticker": ticker,
                    "filing_ticker": filing_ticker,
                    "security_mapping_method": security_mapping_method,
                    "issuer_name": selected_filing.get("issuer_name") or source.get("name", ""),
                    "current_price": format(price, "f"),
                    "price_as_of": price_as_of.isoformat(),
                    "dcf_assumptions": dcf_path,
                    "dcf_input": dcf_input_path_text,
                    "dcf_input_sha256": dcf_input_hash,
                }
            )
        audits.append(audit)

    manifest_headers = [
        "ticker", "source", "input", "metadata", "issuer_id", "issuer_name",
        "current_price", "price_as_of", "dcf_assumptions", "dcf_input", "dcf_input_sha256",
        "selection_report_kind", "selection_period_end", "selection_is_amendment",
        "filing_ticker", "security_mapping_method",
    ]
    write_csv(output_root / "universe-manifest.csv", manifest_rows, manifest_headers)
    audit_headers = sorted({key for row in audits for key in row})
    write_csv(output_root / "dcf-audit.csv", audits, audit_headers)
    write_csv(output_root / "exclusions.csv", exclusions, ["stock_code", "name", "reason"])
    print(f"manifest={output_root / 'universe-manifest.csv'}")
    print(f"companies={len(manifest_rows)}")
    print(f"dcf_ready={sum(bool(row['dcf_assumptions']) for row in manifest_rows)}")
    print(f"exclusions={len(exclusions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
