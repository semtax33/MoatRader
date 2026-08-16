from __future__ import annotations

import pytest

from scripts.prepare_multi_period_economic_holdout import (
    _index_manifest,
    broad_sector,
    trading_horizons,
)


@pytest.mark.parametrize(
    ("industry_code", "expected"),
    [
        ("21210", "HEALTHCARE"),
        ("28202", "MATERIALS_ENERGY"),
        ("29241", "INDUSTRIALS"),
        ("30399", "CONSUMER_DISCRETIONARY"),
        ("75210", "CONSUMER_DISCRETIONARY"),
        ("26110", "IT_COMMUNICATION"),
        ("62010", "IT_COMMUNICATION"),
        ("64121", "FINANCIALS"),
        ("68111", "REAL_ESTATE"),
        ("86101", "HEALTHCARE"),
        ("47111", "CONSUMER_DISCRETIONARY"),
        ("", "UNKNOWN"),
    ],
)
def test_broad_sector_is_deterministic(industry_code: str, expected: str) -> None:
    assert broad_sector(industry_code) == expected


def test_trading_horizons_uses_prior_session_and_exact_session_count() -> None:
    sessions = [
        "2025-08-28",
        "2025-08-29",
        "2025-09-01",
        "2025-09-02",
        "2025-09-03",
    ]

    result = trading_horizons(
        sessions,
        ["2025-08-31"],
        horizon_sessions=2,
    )

    assert result == {"2025-08-31": ("2025-08-29", "2025-09-02")}


def test_trading_horizons_rejects_incomplete_forward_window() -> None:
    with pytest.raises(ValueError, match="lacks 3 sessions"):
        trading_horizons(
            ["2025-08-29", "2025-09-01", "2025-09-02"],
            ["2025-08-31"],
            horizon_sessions=3,
        )


def test_manifest_index_allows_consistent_multi_document_ticker() -> None:
    rows = [
        {"ticker": "123", "issuer_id": "C1", "current_price": "10", "price_as_of": "D", "input": "a"},
        {"ticker": "000123", "issuer_id": "C1", "current_price": "10", "price_as_of": "D", "input": "b"},
    ]

    result = _index_manifest(rows, source="test")

    assert list(result) == ["000123"]
    assert result["000123"]["input"] == "a"


def test_manifest_index_rejects_inconsistent_company_identity() -> None:
    rows = [
        {"ticker": "000123", "issuer_id": "C1", "current_price": "10", "price_as_of": "D"},
        {"ticker": "000123", "issuer_id": "C1", "current_price": "11", "price_as_of": "D"},
    ]

    with pytest.raises(ValueError, match="current_price"):
        _index_manifest(rows, source="test")
