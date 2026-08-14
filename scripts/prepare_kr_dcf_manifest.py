from __future__ import annotations

import argparse
import csv
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
from datetime import datetime, time as datetime_time, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from xml.etree import ElementTree


DART_API = "https://opendart.fss.or.kr/api"
ANNUAL_REPORT_CODE = "11011"


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


def amount(rows: list[dict[str, object]], pattern: str, statements: set[str]) -> Decimal | None:
    regex = re.compile(pattern, re.IGNORECASE)
    values = [
        decimal_value(row.get("thstrm_amount"))
        for row in rows
        if str(row.get("sj_div") or "") in statements
        and regex.search(f"{row.get('account_id', '')} {row.get('account_nm', '')}")
    ]
    clean = [value for value in values if value is not None]
    return max(clean, key=abs) if clean else None


def sum_amounts(rows: list[dict[str, object]], pattern: str, statements: set[str]) -> Decimal:
    regex = re.compile(pattern, re.IGNORECASE)
    values: list[Decimal] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        account_id = str(row.get("account_id") or "")
        account_name = str(row.get("account_nm") or "")
        if str(row.get("sj_div") or "") not in statements or not regex.search(f"{account_id} {account_name}"):
            continue
        key = (account_id, account_name)
        value = decimal_value(row.get("thstrm_amount"))
        if value is not None and key not in seen:
            values.append(abs(value))
            seen.add(key)
    return sum(values, Decimal(0))


def fetch_year(
    client: DartClient,
    output_root: Path,
    ticker: str,
    corp_code: str,
    year: int,
) -> tuple[list[dict[str, object]], str] | None:
    ticker_dir = output_root / "source" / "financials" / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    for fs_div in ("CFS", "OFS"):
        path = ticker_dir / f"{year}-{fs_div}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = client.json(
                "fnlttSinglAcntAll.json",
                {
                    "corp_code": corp_code,
                    "bsns_year": year,
                    "reprt_code": ANNUAL_REPORT_CODE,
                    "fs_div": fs_div,
                },
            )
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if str(payload.get("status")) == "000" and payload.get("list"):
            return list(payload["list"]), fs_div
        if str(payload.get("status")) not in {"013"}:
            break
    return None


def annual_metrics(rows: list[dict[str, object]]) -> dict[str, Decimal | None]:
    revenue = amount(
        rows,
        r"ifrs(?:-full)?_(?:Revenue|SalesRevenue)|dart_Revenue|(?:^|\s)(?:수익\(매출액\)|매출액)(?:$|\s)",
        {"IS", "CIS"},
    )
    ebit = amount(
        rows,
        r"OperatingIncomeLoss|ProfitLossFromOperatingActivities|영업이익(?:\(손실\))?",
        {"IS", "CIS"},
    )
    capex = sum_amounts(
        rows,
        r"PurchaseOfPropertyPlantAndEquipment|PurchaseOfIntangibleAssets|유형자산의 취득|무형자산의 취득|건설중인자산의 취득",
        {"CF"},
    )
    depreciation = sum_amounts(
        rows,
        r"Depreciation|Amortisation|Amortization|감가상각비|무형자산상각비|사용권자산상각비",
        {"CF"},
    )
    cash = amount(rows, r"CashAndCashEquivalents|^\s*현금및현금성자산\s*$", {"BS"}) or Decimal(0)
    debt = sum_amounts(
        rows,
        r"(?:^|\s)(?:단기차입금|장기차입금|유동성장기차입금|유동성사채|사채|전환사채|신주인수권부사채|유동\s*리스부채|비유동\s*리스부채)(?:$|\s)|"
        r"CurrentLoansReceived|NoncurrentLoansReceived|Borrowings|CurrentLeaseLiabilities|NoncurrentLeaseLiabilities",
        {"BS"},
    )
    receivables = amount(rows, r"TradeAndOtherCurrentReceivables|매출채권 및 기타유동채권|^\s*매출채권\s*$", {"BS"}) or Decimal(0)
    inventory = amount(rows, r"Inventories|유동재고자산|^\s*재고자산\s*$", {"BS"}) or Decimal(0)
    payables = amount(rows, r"TradeAndOtherCurrentPayables|매입채무 및 기타유동채무|^\s*매입채무\s*$", {"BS"}) or Decimal(0)
    return {
        "revenue": revenue,
        "ebit": ebit,
        "capex": capex,
        "depreciation": depreciation,
        "cash": cash,
        "debt": debt,
        "nwc": receivables + inventory - payables,
    }


def assumptions_from_history(
    history: list[tuple[int, dict[str, Decimal | None]]],
    size_bucket: str,
    shares: Decimal,
) -> tuple[dict[str, str | list[str]], dict[str, object]]:
    valid = [(year, metrics) for year, metrics in history if metrics["revenue"] and metrics["revenue"] > 0]
    if not valid:
        raise ValueError("no positive annual revenue")
    latest_year, latest = valid[-1]
    revenue = latest["revenue"]
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

    margins = [
        metrics["ebit"] / metrics["revenue"]
        for _year, metrics in valid
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
        for _year, metrics in valid
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

    assumptions: dict[str, str | list[str]] = {
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
    }
    audit = {
        "latest_financial_year": latest_year,
        "history_years": [year for year, _metrics in valid],
        "historical_revenue": [str(metrics["revenue"]) for _year, metrics in valid],
        "historical_ebit": [str(metrics["ebit"]) for _year, metrics in valid],
        "normalized_growth": str(normalized_growth),
        "current_ebit_margin": str(current_margin),
        "normalized_ebit_margin": str(normalized_margin),
        "capex_pct_revenue": str(capex_ratio),
        "depreciation_pct_revenue": str(depreciation_ratio),
        "nwc_pct_revenue": str(nwc_ratio),
        "wacc": str(wacc),
        "net_debt": str(net_debt),
        "method": "historical-normalization-baseline/1",
    }
    return assumptions, audit


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def latest_manifest_rows(
    path: Path,
    *,
    as_of: datetime,
) -> dict[str, dict[str, str]]:
    base = path.parent
    selected: dict[str, tuple[datetime, dict[str, str]]] = {}
    for row in read_csv(path):
        metadata_path = Path(row["metadata"])
        if not metadata_path.is_absolute():
            metadata_path = (base / metadata_path).resolve()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        available = datetime.fromisoformat(str(metadata["available_at"]).replace("Z", "+00:00"))
        if available > as_of:
            continue
        ticker = row["ticker"].zfill(6)
        if ticker not in selected or available > selected[ticker][0]:
            copy = dict(row)
            input_path = Path(copy["input"])
            copy["input"] = str((base / input_path).resolve() if not input_path.is_absolute() else input_path)
            copy["metadata"] = str(metadata_path)
            selected[ticker] = (available, copy)
    return {ticker: row for ticker, (_available, row) in selected.items()}


def historical_market_snapshot(
    ticker: str,
    as_of: datetime,
    *,
    fallback_shares: Decimal,
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

    shares = fallback_shares
    source = "KRX_OHLCV+UNIVERSE_LISTED_SHARES"
    cap = krx_stock.get_market_cap(start, end, ticker)
    if cap is not None and not cap.empty:
        cap = cap[cap.index.date <= market_day].sort_index()
        if not cap.empty and Decimal(str(int(cap.iloc[-1]["상장주식수"]))) > 0:
            shares = Decimal(str(int(cap.iloc[-1]["상장주식수"])))
            source = "KRX_OHLCV+KRX_MARKET_CAP"

    price_at = datetime.combine(
        market_day,
        datetime_time(hour=16),
        tzinfo=as_of.tzinfo,
    )
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
    filing_by_stock = latest_manifest_rows(
        args.collected_manifest.resolve(),
        as_of=as_of,
    )
    years = sorted(set(args.years or [2022, 2023, 2024]))

    manifest_rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    assumptions_dir = output_root / "assumptions"
    assumptions_dir.mkdir(parents=True, exist_ok=True)

    for index, source in enumerate(universe, start=1):
        ticker = source["stock_code"].zfill(6)
        print(f"[{index}/{len(universe)}] {ticker}", flush=True)
        filing = filing_by_stock.get(ticker)
        if filing is None:
            exclusions.append({"stock_code": ticker, "name": source.get("name", ""), "reason": "NO_ANNUAL_FILING"})
            continue
        corp_code = corp_by_stock.get(ticker) or filing.get("issuer_id", "")
        try:
            price, diluted_shares, price_as_of, price_source = historical_market_snapshot(
                ticker,
                as_of,
                fallback_shares=Decimal(source["listed_shares"]),
            )
        except ValueError as exc:
            exclusions.append({"stock_code": ticker, "name": source.get("name", ""), "reason": str(exc)})
            continue

        dcf_path = ""
        audit: dict[str, object] = {
            "stock_code": ticker,
            "name": source.get("name", ""),
            "corp_code": corp_code,
            "current_price": str(price),
            "price_as_of": price_as_of.isoformat(),
            "price_source": price_source,
            "diluted_shares": str(diluted_shares),
        }
        if source.get("finance_hint", "").strip().lower() == "true":
            exclusions.append({"stock_code": ticker, "name": source.get("name", ""), "reason": "FINANCIAL_COMPANY_DCF_MODEL_MISMATCH"})
        else:
            history: list[tuple[int, dict[str, Decimal | None]]] = []
            fs_divisions: list[str] = []
            for year in years:
                response = fetch_year(client, output_root, ticker, corp_code, year)
                if response is None:
                    continue
                rows, fs_div = response
                history.append((year, annual_metrics(rows)))
                fs_divisions.append(f"{year}:{fs_div}")
            try:
                assumptions, details = assumptions_from_history(
                    history,
                    source.get("size_bucket", ""),
                    diluted_shares,
                )
                assumption_path = assumptions_dir / f"{ticker}.json"
                assumption_path.write_text(json.dumps(assumptions, ensure_ascii=False, indent=2), encoding="utf-8")
                dcf_path = str(assumption_path)
                audit.update(details)
                audit["financial_statement_scope"] = ";".join(fs_divisions)
            except ValueError as exc:
                exclusions.append({"stock_code": ticker, "name": source.get("name", ""), "reason": str(exc)})

        manifest_rows.append(
            {
                **filing,
                "ticker": ticker,
                "issuer_name": filing.get("issuer_name") or source.get("name", ""),
                "current_price": format(price, "f"),
                "price_as_of": price_as_of.isoformat(),
                "dcf_assumptions": dcf_path,
            }
        )
        audits.append(audit)

    manifest_headers = [
        "ticker", "source", "input", "metadata", "issuer_id", "issuer_name",
        "current_price", "price_as_of", "dcf_assumptions",
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
