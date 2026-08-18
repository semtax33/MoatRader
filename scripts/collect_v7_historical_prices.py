from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

from moatrader.marketdata.historical_prices import (
    HistoricalAdjustedPrice,
    close_timestamp,
    yahoo_symbol,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value != value:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _download_one(symbol: str, *, start: date, end_exclusive: date):
    import yfinance as yf

    return yf.download(
        symbol,
        start=start.isoformat(),
        end=end_exclusive.isoformat(),
        auto_adjust=False,
        actions=True,
        repair=True,
        progress=False,
        threads=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect dividend/split-adjusted Yahoo price history for v7 historical validation."
    )
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 1, 1))
    parser.add_argument("--end-exclusive", type=date.fromisoformat, default=date(2026, 1, 1))
    parser.add_argument("--max-tickers", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    if "v7" not in output.as_posix().casefold():
        raise ValueError("historical price output must be a v7 path")
    if output.exists():
        raise FileExistsError(f"historical price output already exists: {output}")
    universe = _read_csv(args.universe.resolve())
    if len(universe) != 150:
        raise ValueError(f"historical validation requires 150 universe rows, got {len(universe)}")
    if args.max_tickers is not None:
        universe = universe[: args.max_tickers]
    output.mkdir(parents=True)
    rows: list[HistoricalAdjustedPrice] = []
    coverage: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for index, item in enumerate(universe, start=1):
        ticker = item["stock_code"].zfill(6)
        symbol = yahoo_symbol(ticker, item["market"])
        try:
            frame = _download_one(symbol, start=args.start, end_exclusive=args.end_exclusive)
            if frame.empty:
                raise ValueError("Yahoo returned no price rows")
            if getattr(frame.columns, "nlevels", 1) == 2:
                frame = frame.xs(symbol, axis=1, level=1)
            first = None
            last = None
            count = 0
            for timestamp, values in frame.iterrows():
                adjusted = _float(values.get("Adj Close"))
                close = _float(values.get("Close"))
                if adjusted <= 0 or close <= 0:
                    continue
                point = HistoricalAdjustedPrice(
                    timestamp=close_timestamp(timestamp.date()),
                    ticker=ticker,
                    yahoo_symbol=symbol,
                    adjusted_close=str(adjusted),
                    close=str(close),
                    dividends=str(max(0.0, _float(values.get("Dividends")))),
                    stock_splits=str(max(0.0, _float(values.get("Stock Splits")))),
                    volume=max(0, int(_float(values.get("Volume")))),
                )
                rows.append(point)
                first = first or timestamp.date()
                last = timestamp.date()
                count += 1
            if count == 0:
                raise ValueError("Yahoo returned no valid adjusted-close rows")
            coverage.append(
                {
                    "ticker": ticker,
                    "symbol": symbol,
                    "row_count": count,
                    "first_date": first.isoformat(),
                    "last_date": last.isoformat(),
                    "full_period_start": first <= args.start,
                    "full_period_end": last >= date(2025, 12, 30),
                }
            )
            print(f"[{index}/{len(universe)}] {symbol} rows={count}", flush=True)
        except Exception as exc:
            failures.append({"ticker": ticker, "symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{index}/{len(universe)}] {symbol} FAILED {type(exc).__name__}", flush=True)

    price_path = output / "adjusted-prices.csv"
    with price_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "timestamp",
                "ticker",
                "yahoo_symbol",
                "adjusted_close",
                "close",
                "dividends",
                "stock_splits",
                "volume",
                "tradable",
            ],
        )
        writer.writeheader()
        for point in sorted(rows, key=lambda value: (value.timestamp, value.ticker)):
            writer.writerow(point.model_dump(mode="json"))
    manifest = {
        "schema_version": "v7-historical-adjusted-prices/1",
        "universe_sha256": _sha256(args.universe.resolve()),
        "requested_ticker_count": len(universe),
        "covered_ticker_count": len(coverage),
        "full_period_ticker_count": sum(item["full_period_start"] and item["full_period_end"] for item in coverage),
        "price_row_count": len(rows),
        "start": args.start.isoformat(),
        "end_exclusive": args.end_exclusive.isoformat(),
        "provider": "Yahoo Finance via yfinance",
        "provider_url_template": "https://finance.yahoo.com/quote/{symbol}/history/",
        "return_basis": "YAHOO_ADJ_CLOSE_INCLUDES_PROVIDER_REPORTED_SPLITS_AND_DISTRIBUTIONS",
        "coverage": coverage,
        "failure_count": len(failures),
        "failures": failures,
        "prices_sha256": _sha256(price_path),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output / "manifest.json")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
