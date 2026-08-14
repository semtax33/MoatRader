from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from moatrader.backtest.models import PricePoint


def _parse_timestamp(value: str, row_number: int) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"row {row_number}: invalid timestamp {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"row {row_number}: timestamp must include timezone offset")
    return parsed


def load_price_panel(path: str | Path) -> list[PricePoint]:
    price_path = Path(path).resolve()
    if not price_path.is_file():
        raise FileNotFoundError(f"price panel not found: {price_path}")
    points: list[PricePoint] = []
    seen: set[tuple[str, datetime]] = set()
    with price_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"timestamp", "ticker", "adjusted_close"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"price panel is missing required columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            ticker = (row.get("ticker") or "").strip()
            if not ticker:
                raise ValueError(f"row {row_number}: ticker is required")
            tradable = (row.get("tradable") or "true").strip().lower()
            if tradable in {"0", "false", "no", "n"}:
                continue
            timestamp = _parse_timestamp((row.get("timestamp") or "").strip(), row_number)
            key = (ticker, timestamp)
            if key in seen:
                raise ValueError(f"row {row_number}: duplicate price for {ticker} at {timestamp.isoformat()}")
            seen.add(key)
            try:
                close = Decimal((row.get("adjusted_close") or "").strip())
            except Exception as exc:
                raise ValueError(f"row {row_number}: invalid adjusted_close") from exc
            points.append(PricePoint(timestamp=timestamp, ticker=ticker, adjusted_close=close))
    return sorted(points, key=lambda point: (point.timestamp, point.ticker))


class PricePanel:
    def __init__(self, points: list[PricePoint]) -> None:
        if not points:
            raise ValueError("price panel is empty")
        self.by_ticker: dict[str, dict[datetime, Decimal]] = defaultdict(dict)
        for point in points:
            if point.timestamp in self.by_ticker[point.ticker]:
                raise ValueError(f"duplicate price for {point.ticker} at {point.timestamp.isoformat()}")
            self.by_ticker[point.ticker][point.timestamp] = point.adjusted_close

    def common_timestamp(self, tickers: set[str], at_or_after: datetime) -> datetime:
        if at_or_after.tzinfo is None or at_or_after.utcoffset() is None:
            raise ValueError("price lookup timestamp must be timezone-aware")
        if not tickers:
            return at_or_after
        available: set[datetime] | None = None
        for ticker in sorted(tickers):
            prices = self.by_ticker.get(ticker)
            if not prices:
                raise ValueError(f"no adjusted price history for held/selected ticker {ticker}")
            timestamps = {timestamp for timestamp in prices if timestamp >= at_or_after}
            available = timestamps if available is None else available & timestamps
        if not available:
            raise ValueError(
                f"no common tradable adjusted-price timestamp for {sorted(tickers)} "
                f"at or after {at_or_after.isoformat()}"
            )
        return min(available)

    def prices_at(self, tickers: set[str], timestamp: datetime) -> dict[str, Decimal]:
        result: dict[str, Decimal] = {}
        for ticker in tickers:
            try:
                result[ticker] = self.by_ticker[ticker][timestamp]
            except KeyError as exc:
                raise ValueError(f"missing adjusted price for {ticker} at {timestamp.isoformat()}") from exc
        return result

    def common_timestamps(
        self,
        tickers: set[str],
        *,
        after: datetime,
        through: datetime,
    ) -> list[datetime]:
        if not tickers or through <= after:
            return []
        available: set[datetime] | None = None
        for ticker in sorted(tickers):
            prices = self.by_ticker.get(ticker)
            if not prices:
                raise ValueError(f"no adjusted price history for held ticker {ticker}")
            timestamps = {timestamp for timestamp in prices if after < timestamp <= through}
            available = timestamps if available is None else available & timestamps
        return sorted(available or [])
