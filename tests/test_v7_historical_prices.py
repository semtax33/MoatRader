from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from moatrader.marketdata.historical_prices import HistoricalMarcapPrice, close_timestamp, yahoo_symbol


def test_yahoo_symbol_maps_krx_markets() -> None:
    assert yahoo_symbol("5930", "KOSPI") == "005930.KS"
    assert yahoo_symbol("357780", "kosdaq") == "357780.KQ"
    with pytest.raises(ValueError, match="unsupported"):
        yahoo_symbol("005930", "NYSE")


def test_historical_close_timestamp_is_seoul_aware() -> None:
    value = close_timestamp(date(2020, 1, 2))
    assert value.hour == 15
    assert value.minute == 30
    assert value.utcoffset().total_seconds() == 9 * 60 * 60


def test_marcap_point_enforces_price_times_listed_shares_identity() -> None:
    point = HistoricalMarcapPrice(
        timestamp=close_timestamp(date(2020, 1, 2)),
        ticker="005930",
        name="Samsung Electronics",
        close=Decimal("55200"),
        open=Decimal("55500"),
        high=Decimal("56000"),
        low=Decimal("55000"),
        volume=12_993_228,
        amount=Decimal("719663194492"),
        market_cap=Decimal("329531996760000"),
        listed_shares=5_969_782_550,
        changes_ratio_percent=Decimal("-1.08"),
        market="KOSPI",
        market_id="STK",
        rank=1,
        source_year=2020,
    )
    assert point.market_cap == point.close * point.listed_shares

    invalid = point.model_dump()
    invalid["market_cap"] = Decimal("1")
    with pytest.raises(ValueError, match="close multiplied"):
        HistoricalMarcapPrice.model_validate(invalid)
