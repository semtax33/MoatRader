from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.run_v7_1_value_neutral_sensitivity import extract_value_fundamentals


SCHEMA_VERSION = "universal-value-universe-test-input/1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_close(ticker: str, signal_date: date, *, timeout: int) -> dict[str, str]:
    code = (
        "import json,sys; from pykrx import stock; "
        "d=sys.argv[1]; t=sys.argv[2]; "
        "f=stock.get_market_ohlcv(d,d,t); "
        "assert f is not None and not f.empty, 'no KRX close'; "
        "f=f.sort_index(); c=int(f.iloc[-1].iloc[3]); "
        "assert c>0, 'nonpositive KRX close'; "
        "print('PRICE_JSON:'+json.dumps({'ticker':t,'current_price':str(c),"
        "'price_as_of':str(f.index[-1].date())+'T16:00:00+09:00',"
        "'price_source':'PYKRX_KRX_OHLCV_CLOSE'}))"
    )
    day = signal_date.strftime("%Y%m%d")
    completed = subprocess.run(
        [sys.executable, "-c", code, day, ticker],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("PRICE_JSON:"):
            return json.loads(line.removeprefix("PRICE_JSON:"))
    raise ValueError(
        f"isolated KRX close failed rc={completed.returncode}: "
        f"{completed.stderr.strip() or completed.stdout.strip()}"
    )


def latest_sector_rows(path: Path, tickers: set[str], signal_date: date) -> list[dict[str, str]]:
    latest: dict[str, dict[str, str]] = {}
    for row in read_csv(path):
        ticker = str(row.get("ticker") or "").zfill(6)
        effective = date.fromisoformat(str(row["effective_from"]))
        if ticker not in tickers or effective > signal_date:
            continue
        if ticker not in latest or effective > date.fromisoformat(latest[ticker]["effective_from"]):
            latest[ticker] = row
    missing = sorted(tickers - set(latest))
    if missing:
        raise ValueError(f"missing PIT sector carry-forward for {missing}")
    result: list[dict[str, str]] = []
    for ticker in sorted(tickers):
        row = dict(latest[ticker])
        row["effective_to"] = signal_date.isoformat()
        row["evidence_ref"] = (
            f"PIT_CARRY_FORWARD:{row['evidence_ref']}:TO:{signal_date.isoformat()}"
        )
        result.append(row)
    return result


def snapshot_point(
    value: Any, *, period: str, basis: str, source: str, available_at: str
) -> dict[str, Any]:
    return {
        "period": period,
        "period_basis": basis,
        "value": str(value),
        "unit": "KRW",
        "source_fact_ids": [source],
        "available_at": available_at,
    }


def build_snapshot(
    *,
    ticker: str,
    issuer_id: str,
    signal_date: date,
    annual_path: Path | None,
    dcf_path: Path | None,
) -> dict[str, Any]:
    fundamentals: dict[str, float | None] = {}
    annual_ref = "NO_ANNUAL_SNAPSHOT"
    if annual_path is not None and annual_path.is_file():
        fundamentals = extract_value_fundamentals(pd.read_csv(annual_path, low_memory=False))
        annual_ref = f"NORMALIZED_DART_2025_SHA256:{sha256_file(annual_path)}"
    dcf: dict[str, Any] = {}
    dcf_ref = "NO_DCF_INPUT"
    if dcf_path is not None and dcf_path.is_file():
        dcf = json.loads(dcf_path.read_text(encoding="utf-8-sig"))
        dcf_ref = f"PIT_DCF_2026_08_18_SHA256:{sha256_file(dcf_path)}"
    metrics = dcf.get("metrics") or {}
    current_available_at = str(
        (dcf.get("pit") or {}).get("latest_report_available_at")
        or f"{signal_date.isoformat()}T23:59:59+09:00"
    )
    values: dict[str, Any] = {
        "REVENUE": metrics.get("revenue") or fundamentals.get("fund_revenue"),
        "EBIT": metrics.get("ebit") or fundamentals.get("fund_ebit"),
        "NET_INCOME": fundamentals.get("fund_net_income"),
        "CFO": fundamentals.get("fund_cfo"),
        "CAPEX": fundamentals.get("fund_capex"),
        "GROSS_PROFIT": fundamentals.get("fund_gross_profit"),
        "RND": fundamentals.get("fund_rnd"),
        "RETAINED_EARNINGS": fundamentals.get("fund_retained_earnings"),
        "TOTAL_ASSETS": fundamentals.get("fund_total_assets"),
        "TOTAL_EQUITY": fundamentals.get("fund_total_equity"),
        "CURRENT_ASSETS": fundamentals.get("fund_current_assets"),
        "TOTAL_LIABILITIES": fundamentals.get("fund_total_liabilities"),
        "CASH": metrics.get("cash") or fundamentals.get("fund_cash"),
        "TOTAL_DEBT": metrics.get("debt") or fundamentals.get("fund_debt"),
    }
    series = []
    for concept, value in values.items():
        if value is None or pd.isna(value):
            continue
        is_current = concept in {"REVENUE", "EBIT", "CASH", "TOTAL_DEBT"} and bool(metrics)
        source = dcf_ref if is_current else annual_ref
        series.append(
            {
                "concept": concept,
                "points": [
                    snapshot_point(
                        value,
                        period="2026-03-31" if is_current else "2025-12-31",
                        basis="TTM" if is_current else "FY",
                        source=source,
                        available_at=(
                            current_available_at
                            if is_current
                            else "2026-03-31T23:59:59+09:00"
                        ),
                    )
                ],
            }
        )
    return {
        "as_of": f"{signal_date.isoformat()}T23:59:59+09:00",
        "issuer_id": issuer_id,
        "issuer_name": ticker,
        "series": series,
        "provenance": [annual_ref, dcf_ref, "NO_LLM:DETERMINISTIC_TEST_STAGING"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    cache_path = args.output / "_price-cache.json"
    existing = list(args.output.iterdir()) if args.output.exists() else []
    if existing and existing != [cache_path]:
        raise FileExistsError(f"output must be new or contain only price cache: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    universe = read_csv(args.universe)
    if len(universe) != 150:
        raise ValueError(f"expected exactly 150 universe rows, got {len(universe)}")
    tickers = {str(row["stock_code"]).zfill(6) for row in universe}
    if len(tickers) != 150:
        raise ValueError("universe tickers must be unique")
    current = {
        str(row["stock_code"]).zfill(6): row for row in read_csv(args.current_universe)
    }
    missing_current = sorted(tickers - set(current))
    if missing_current:
        raise ValueError(f"current universe lacks tickers: {missing_current}")

    prices: dict[str, dict[str, str]] = (
        json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
    )
    price_errors: dict[str, str] = {}
    ordered = sorted(tickers)
    for index, ticker in enumerate(ordered, start=1):
        if ticker not in prices:
            try:
                prices[ticker] = fetch_close(
                    ticker, args.signal_date, timeout=args.price_timeout
                )
                cache_path.write_text(
                    json.dumps(prices, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            except Exception as exc:
                price_errors[ticker] = f"{type(exc).__name__}:{exc}"
        print(f"[price {index}/{len(ordered)}] {ticker}", flush=True)
    if price_errors:
        raise ValueError(f"8/19 KRX price failures: {price_errors}")

    source_manifest = read_csv(
        args.source_base_root / "date-inputs" / args.source_date.isoformat() / "universe-manifest.csv"
    )
    issuer_by_ticker: dict[str, str] = {}
    for row in source_manifest:
        ticker = str(row.get("ticker") or "").zfill(6)
        if ticker in tickers and row.get("issuer_id") and ticker not in issuer_by_ticker:
            issuer_by_ticker[ticker] = str(row["issuer_id"])

    staged_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    source_date = args.source_date.isoformat()
    signal_date = args.signal_date.isoformat()
    dcf_count = 0
    snapshot_count = 0
    annual_count = 0
    for row in universe:
        ticker = str(row["stock_code"]).zfill(6)
        current_row = current[ticker]
        price = prices[ticker]
        shares = str(current_row.get("listed_shares") or row.get("listed_shares") or "")
        if not shares or float(shares) <= 0:
            raise ValueError(f"missing current listed shares for {ticker}")
        staged = dict(row)
        staged.update(price)
        staged["listed_shares"] = shares
        staged["market_cap"] = str(float(price["current_price"]) * float(shares))
        staged["as_of"] = signal_date
        staged_rows.append(staged)

        source_dcf = (
            args.source_base_root / "date-inputs" / source_date / "dcf-inputs" / f"{ticker}.json"
        )
        target_dcf = args.output / "date-inputs" / signal_date / "dcf-inputs" / f"{ticker}.json"
        dcf_path: Path | None = None
        if source_dcf.is_file():
            target_dcf.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_dcf, target_dcf)
            dcf_path = target_dcf
            dcf_count += 1
        annual_path = args.annual_snapshot_root / f"kr_normalized_{ticker}_2025.12.csv"
        if annual_path.is_file():
            annual_count += 1
        else:
            annual_path = None
        issuer_id = issuer_by_ticker.get(ticker, ticker)
        snapshot = build_snapshot(
            ticker=ticker,
            issuer_id=issuer_id,
            signal_date=args.signal_date,
            annual_path=annual_path,
            dcf_path=dcf_path,
        )
        snapshot_path = (
            args.output / "runs" / f"kr-signal-{signal_date}" / "companies" / ticker / "financial-snapshot.json"
        )
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        snapshot_count += 1
        manifest_rows.append(
            {
                "ticker": ticker,
                "issuer_id": issuer_id,
                "issuer_name": row.get("name", ""),
                "current_price": price["current_price"],
                "price_as_of": price["price_as_of"],
                "price_source": price["price_source"],
                "listed_shares": shares,
                "dcf_input": str(target_dcf) if target_dcf.is_file() else "",
            }
        )

    universe_fields = list(universe[0])
    for field in ("current_price", "price_as_of", "price_source"):
        if field not in universe_fields:
            universe_fields.append(field)
    write_csv(args.output / "inputs" / "universe.csv", staged_rows, universe_fields)
    write_csv(args.output / "inputs" / "dates.csv", [{"date": signal_date}], ["date"])
    write_csv(
        args.output / "date-inputs" / signal_date / "universe-manifest.csv",
        manifest_rows,
        list(manifest_rows[0]),
    )
    sector_rows = latest_sector_rows(args.pit_sector_map, tickers, args.signal_date)
    write_csv(args.output / "inputs" / "pit-sector.csv", sector_rows, list(sector_rows[0]))
    report = {
        "schema_version": SCHEMA_VERSION,
        "signal_date": signal_date,
        "financial_input_cutoff": source_date,
        "same_day_filings_included": False,
        "universe_count": len(universe),
        "krx_price_count": len(prices),
        "dcf_input_count": dcf_count,
        "annual_snapshot_count": annual_count,
        "financial_snapshot_count": snapshot_count,
        "sector_count": len(sector_rows),
        "issuer_id_count": len(issuer_by_ticker),
        "llm_call_count": 0,
        "universe_sha256": sha256_file(args.universe),
        "staged_universe_sha256": sha256_file(args.output / "inputs" / "universe.csv"),
    }
    (args.output / "STAGING-REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--current-universe", type=Path, required=True)
    parser.add_argument("--source-base-root", type=Path, required=True)
    parser.add_argument("--source-date", type=date.fromisoformat, required=True)
    parser.add_argument("--signal-date", type=date.fromisoformat, required=True)
    parser.add_argument("--annual-snapshot-root", type=Path, required=True)
    parser.add_argument("--pit-sector-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--price-timeout", type=int, default=30)
    args = parser.parse_args()
    for name in (
        "universe",
        "current_universe",
        "source_base_root",
        "annual_snapshot_root",
        "pit_sector_map",
        "output",
    ):
        setattr(args, name, getattr(args, name).resolve())
    print(json.dumps(run(args), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
