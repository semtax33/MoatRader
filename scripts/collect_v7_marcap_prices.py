from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import date
from decimal import Decimal
from pathlib import Path

import requests

from moatrader.marketdata.historical_prices import HistoricalMarcapPrice, close_timestamp


MARCAP_REPOSITORY = "https://github.com/FinanceData/marcap"
DEFAULT_COMMIT = "5e8e4e57f3fcb129a6ff20751f643f67d3592c82"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_url(*, commit: str, year: int) -> str:
    return (
        "https://raw.githubusercontent.com/FinanceData/marcap/"
        f"{commit}/data/marcap-{year}.parquet"
    )


def _download(url: str, target: Path, *, attempts: int = 4) -> None:
    temporary = target.with_suffix(target.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with requests.get(url, timeout=120, stream=True) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            stream.write(block)
            temporary.replace(target)
            return
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"failed to download pinned marcap source after {attempts} attempts: {url}") from last_error


def _integer(value: object) -> int:
    return int(float(value))


def _decimal(value: object) -> Decimal:
    return Decimal(str(value))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect pinned FinanceData/marcap daily PIT prices for v7 validation."
    )
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--commit", default=DEFAULT_COMMIT)
    parser.add_argument("--max-tickers", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    if "v7" not in output.as_posix().casefold():
        raise ValueError("historical marcap output must be a v7 path")
    if output.exists() and not args.resume:
        raise FileExistsError(f"historical marcap output already exists: {output}")
    if args.end < args.start:
        raise ValueError("marcap end date precedes start date")
    if len(args.commit) != 40 or any(char not in "0123456789abcdef" for char in args.commit):
        raise ValueError("marcap commit must be a lowercase 40-character Git SHA")

    universe_path = args.universe.resolve()
    universe = _read_csv(universe_path)
    if len(universe) != 150:
        raise ValueError(f"historical validation requires 150 universe rows, got {len(universe)}")
    if args.max_tickers is not None:
        universe = universe[: args.max_tickers]
    tickers = {item["stock_code"].zfill(6) for item in universe}

    source_dir = output / "source"
    source_dir.mkdir(parents=True, exist_ok=args.resume)
    source_records: list[dict[str, object]] = []
    points: list[HistoricalMarcapPrice] = []
    coverage: dict[str, dict[str, object]] = {
        ticker: {"ticker": ticker, "row_count": 0, "first_date": None, "last_date": None}
        for ticker in sorted(tickers)
    }

    import pandas as pd

    for year in range(args.start.year, args.end.year + 1):
        url = _source_url(commit=args.commit, year=year)
        source = source_dir / f"marcap-{year}.parquet"
        if not source.is_file():
            _download(url, source)
        source_records.append(
            {
                "year": year,
                "url": url,
                "path": str(source),
                "bytes": source.stat().st_size,
                "sha256": _sha256(source),
            }
        )
        frame = pd.read_parquet(source)
        required = {
            "Date", "Code", "Name", "Close", "Open", "High", "Low", "Volume",
            "Amount", "Marcap", "Stocks", "ChangesRatio", "Market", "MarketId", "Rank",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"marcap-{year} is missing columns: {missing}")
        frame["Code"] = frame["Code"].astype(str).str.zfill(6)
        frame = frame[frame["Code"].isin(tickers)]
        frame = frame[(frame["Date"].dt.date >= args.start) & (frame["Date"].dt.date <= args.end)]
        for row in frame.itertuples(index=False):
            trading_date = row.Date.date()
            point = HistoricalMarcapPrice(
                timestamp=close_timestamp(trading_date),
                ticker=row.Code,
                name=str(row.Name),
                close=_decimal(row.Close),
                open=_decimal(row.Open),
                high=_decimal(row.High),
                low=_decimal(row.Low),
                volume=_integer(row.Volume),
                amount=_decimal(row.Amount),
                market_cap=_decimal(row.Marcap),
                listed_shares=_integer(row.Stocks),
                changes_ratio_percent=_decimal(row.ChangesRatio),
                market=str(row.Market),
                market_id=str(row.MarketId),
                rank=_integer(row.Rank),
                source_year=year,
            )
            points.append(point)
            item = coverage[point.ticker]
            item["row_count"] = int(item["row_count"]) + 1
            item["first_date"] = item["first_date"] or trading_date.isoformat()
            item["last_date"] = trading_date.isoformat()
        print(f"marcap-{year}: source_rows={len(frame)}", flush=True)

    price_path = output / "marcap-prices.csv"
    fields = [
        "timestamp", "ticker", "name", "close", "open", "high", "low", "volume",
        "amount", "market_cap", "listed_shares", "changes_ratio_percent", "market",
        "market_id", "rank", "source_year",
    ]
    with price_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for point in sorted(points, key=lambda item: (item.timestamp, item.ticker)):
            writer.writerow(point.model_dump(mode="json"))

    coverage_rows = list(coverage.values())
    missing = [item["ticker"] for item in coverage_rows if item["row_count"] == 0]
    manifest = {
        "schema_version": "v7-historical-marcap-prices/1",
        "universe_path": str(universe_path),
        "universe_sha256": _sha256(universe_path),
        "requested_ticker_count": len(tickers),
        "covered_ticker_count": len(tickers) - len(missing),
        "missing_tickers": missing,
        "price_row_count": len(points),
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "provider": "FinanceData/marcap",
        "provider_repository": MARCAP_REPOSITORY,
        "provider_commit": args.commit,
        "price_basis": "KRX_DAILY_UNADJUSTED_CLOSE",
        "changes_ratio_basis": "KRX_DAILY_CHANGES_RATIO_EXCLUDES_CASH_DISTRIBUTIONS",
        "pit_fields": ["Close", "Marcap", "Stocks", "Market", "Rank"],
        "sources": source_records,
        "coverage": coverage_rows,
        "prices_sha256": _sha256(price_path),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
