from __future__ import annotations

from datetime import date

import pandas as pd

from moatrader.universe.live_kr import build_live_kr_universe


def test_live_kr_universe_preserves_real_market_timestamp_and_new_ticker_format() -> None:
    frame = pd.DataFrame(
        [
            ("005930", "삼성전자", 80000, 100, 1000, 10, "STK", "2026-08-14"),
            ("00680K", "미래에셋증권2우B", 4000, 50, 200, 5, "STK", "2026-08-14"),
            ("035420", "NAVER", 200000, 80, 900, 7, "KSQ", "2026-08-14"),
            ("0030R0", "대신밸류리츠", 5000, 20, 100, 4, "KNX", "2026-08-14"),
        ],
        columns=["Code", "Name", "Close", "Amount", "Marcap", "Stocks", "MarketId", "Date"],
    )

    result = build_live_kr_universe(frame, as_of=date(2026, 8, 18))

    assert result.requested_as_of == date(2026, 8, 18)
    assert result.source_as_of == date(2026, 8, 14)
    assert result.price_as_of.isoformat() == "2026-08-14T16:00:00+09:00"
    assert set(result.universe["stock_code"]) == {"005930", "00680K", "035420", "0030R0"}
    assert result.universe.set_index("stock_code").loc["00680K", "security_type"] == "PREFERRED"
    assert result.universe.set_index("stock_code").loc["0030R0", "security_type"] == "REIT"


def test_live_kr_universe_does_not_misclassify_meritz() -> None:
    frame = pd.DataFrame(
        [("138040", "메리츠금융지주", 100000, 10, 1000, 10, "STK", "2026-08-18")],
        columns=["Code", "Name", "Close", "Amount", "Marcap", "Stocks", "MarketId", "Date"],
    )

    result = build_live_kr_universe(frame, as_of=date(2026, 8, 18))

    row = result.universe.iloc[0]
    assert row["security_type"] == "COMMON"
    assert bool(row["finance_hint"]) is True
