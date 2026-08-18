from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal
from zoneinfo import ZoneInfo

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel


SEOUL = ZoneInfo("Asia/Seoul")


class HistoricalAdjustedPrice(ContractModel):
    timestamp: datetime
    ticker: str = Field(pattern=r"^[0-9]{6}$")
    yahoo_symbol: str = Field(min_length=1)
    adjusted_close: Decimal = Field(gt=0)
    close: Decimal = Field(gt=0)
    dividends: Decimal = Field(default=Decimal(0), ge=0)
    stock_splits: Decimal = Field(default=Decimal(0), ge=0)
    volume: int = Field(default=0, ge=0)
    tradable: bool = True

    @model_validator(mode="after")
    def aware_close_timestamp(self) -> "HistoricalAdjustedPrice":
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("historical price timestamp must be timezone-aware")
        return self


class HistoricalMarcapPrice(ContractModel):
    timestamp: datetime
    ticker: str = Field(pattern=r"^[0-9]{6}$")
    name: str = Field(min_length=1)
    close: Decimal = Field(gt=0)
    open: Decimal = Field(ge=0)
    high: Decimal = Field(ge=0)
    low: Decimal = Field(ge=0)
    volume: int = Field(ge=0)
    amount: Decimal = Field(ge=0)
    market_cap: Decimal = Field(gt=0)
    listed_shares: int = Field(gt=0)
    changes_ratio_percent: Decimal
    market: str = Field(min_length=1)
    market_id: str = Field(min_length=1)
    rank: int = Field(gt=0)
    source_year: int = Field(ge=1995, le=2200)

    @model_validator(mode="after")
    def point_in_time_market_identity(self) -> "HistoricalMarcapPrice":
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("historical marcap timestamp must be timezone-aware")
        if self.timestamp.year != self.source_year:
            raise ValueError("marcap point year does not match its immutable source file")
        expected = self.close * self.listed_shares
        tolerance = max(Decimal(1), abs(expected) * Decimal("0.000000001"))
        if abs(self.market_cap - expected) > tolerance:
            raise ValueError("marcap must equal close multiplied by listed shares")
        if self.high and self.low and self.high < self.low:
            raise ValueError("marcap high cannot be below low")
        return self


def yahoo_symbol(ticker: str, market: str) -> str:
    suffix = {"KOSPI": ".KS", "KOSDAQ": ".KQ"}.get(market.upper())
    if suffix is None:
        raise ValueError(f"unsupported Yahoo Korea market: {market}")
    return ticker.zfill(6) + suffix


def close_timestamp(value: date) -> datetime:
    return datetime.combine(value, time(hour=15, minute=30), tzinfo=SEOUL)
