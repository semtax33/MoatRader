from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

from moatrader.backtest.universe_corrected import (
    FINANCE_HINT_RE,
    HOLDING_HINT_RE,
    classify_current_security,
)


_MARKETS = {"STK": "KOSPI", "KSQ": "KOSDAQ", "KNX": "KONEX"}
_TICKER_RE = re.compile(r"[0-9A-Z]{6}")
_KST = ZoneInfo("Asia/Seoul")


@dataclass(frozen=True)
class LiveKrUniverseBuild:
    requested_as_of: date
    source_as_of: date
    price_as_of: datetime
    universe: pd.DataFrame


def _size_bucket(percentile: float) -> str:
    if percentile <= 1 / 3:
        return "SMALL"
    if percentile <= 2 / 3:
        return "MID"
    return "LARGE"


def build_live_kr_universe(
    marcap: pd.DataFrame,
    *,
    as_of: date,
) -> LiveKrUniverseBuild:
    """Build an all-listed Korean security snapshot from a pinned marcap frame.

    The requested research cutoff and the latest available market snapshot are
    deliberately separate. A stale source is retained with its true timestamp;
    it is never relabelled as the requested cutoff.
    """

    required = {
        "Code",
        "Name",
        "Close",
        "Amount",
        "Marcap",
        "Stocks",
        "MarketId",
        "Date",
    }
    missing = sorted(required - set(marcap.columns))
    if missing:
        raise ValueError(f"marcap frame missing columns: {missing}")

    frame = marcap.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    frame["Code"] = frame["Code"].astype(str).str.strip().str.upper().str.zfill(6)
    frame = frame[
        frame["MarketId"].isin(_MARKETS) & (frame["Date"].dt.date <= as_of)
    ].copy()
    if frame.empty:
        raise ValueError(f"no Korean market rows at or before {as_of.isoformat()}")

    source_at = pd.Timestamp(frame["Date"].max()).date()
    trading_dates = sorted(pd.to_datetime(frame["Date"].unique()))
    last_dates = trading_dates[-60:]
    snapshot = frame[frame["Date"].dt.date == source_at].copy()
    if snapshot.empty:
        raise ValueError("latest Korean market snapshot is empty")

    invalid = sorted(
        ticker for ticker in snapshot["Code"].unique() if not _TICKER_RE.fullmatch(ticker)
    )
    if invalid:
        raise ValueError(f"unsupported live Korean ticker values: {invalid[:10]}")
    if snapshot["Code"].duplicated().any():
        duplicates = sorted(snapshot.loc[snapshot["Code"].duplicated(), "Code"].unique())
        raise ValueError(f"duplicate live Korean ticker rows: {duplicates[:10]}")

    traded = (
        frame[frame["Date"].isin(last_dates)]
        .groupby("Code", sort=False)["Amount"]
        .sum(min_count=1)
        .rename("trading_value_60d")
    )
    snapshot = snapshot.merge(traded, left_on="Code", right_index=True, how="left")
    snapshot["trading_value_60d"] = pd.to_numeric(
        snapshot["trading_value_60d"], errors="coerce"
    ).fillna(0.0)

    output = pd.DataFrame(
        {
            "stock_code": snapshot["Code"],
            "name": snapshot["Name"].astype(str).str.strip(),
            "market": snapshot["MarketId"].map(_MARKETS),
            "market_cap": pd.to_numeric(snapshot["Marcap"], errors="coerce"),
            "listed_shares": pd.to_numeric(snapshot["Stocks"], errors="coerce"),
            "current_price": pd.to_numeric(snapshot["Close"], errors="coerce"),
            "trading_value_60d": snapshot["trading_value_60d"],
        }
    )
    divisor = max(1, len(last_dates))
    output["adtv60"] = output["trading_value_60d"] / divisor
    output["security_type"] = output["name"].map(classify_current_security)
    output["finance_hint"] = output["name"].map(
        lambda value: bool(FINANCE_HINT_RE.search(value))
    )
    output["holding_hint"] = output["name"].map(
        lambda value: bool(HOLDING_HINT_RE.search(value))
    )
    output["liquidity_pct"] = output["adtv60"].rank(method="average", pct=True)
    cap_percentile = output["market_cap"].rank(method="average", pct=True)
    output["size_bucket"] = cap_percentile.map(_size_bucket)

    price_at = datetime.combine(source_at, time(hour=16), tzinfo=_KST)
    output["price_as_of"] = price_at.isoformat()
    output["price_source"] = "FINANCEDATA_MARCAP_PINNED"
    output["universe_source_as_of"] = source_at.isoformat()
    output["as_of"] = as_of.isoformat()
    output["selection_seed"] = as_of.strftime("%Y%m%d")
    output["selection_rule"] = "ALL_LISTED_SECURITIES|STK,KSQ,KNX|NO_SURVIVORSHIP_FILTER"
    output = output.sort_values(
        ["market", "stock_code"],
        key=lambda series: (
            series.map({"KOSPI": 0, "KOSDAQ": 1, "KONEX": 2})
            if series.name == "market"
            else series
        ),
        kind="stable",
    ).reset_index(drop=True)

    return LiveKrUniverseBuild(
        requested_as_of=as_of,
        source_as_of=source_at,
        price_as_of=price_at,
        universe=output,
    )
