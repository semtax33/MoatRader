from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from scripts.prepare_historical_evidence_index_neutral_controls_v2 import (
    analyst_eps_consensus,
)


SEOUL = ZoneInfo("Asia/Seoul")


def _record(*, day: int, broker: str, eps: str, year: int = 2024) -> dict[str, object]:
    return {
        "available_at": datetime(2024, 3, day, 8, 0, tzinfo=SEOUL),
        "forecast_year": year,
        "eps": Decimal(eps),
        "broker": broker,
        "source_id": f"SRC:{broker}:{day}",
    }


def test_analyst_consensus_uses_latest_per_broker_and_median() -> None:
    result = analyst_eps_consensus(
        [
            _record(day=1, broker="A", eps="100"),
            _record(day=10, broker="A", eps="120"),
            _record(day=5, broker="B", eps="80"),
            _record(day=20, broker="C", eps="100", year=2025),
        ],
        cutoff=datetime(2024, 3, 15, 9, 0, tzinfo=SEOUL),
    )
    assert result is not None
    assert result["forecast_year"] == 2024
    assert result["broker_count"] == 2
    assert result["eps"] == Decimal("100.0")
    assert result["source_ids"] == ["SRC:A:10", "SRC:B:5"]


def test_analyst_consensus_requires_two_brokers() -> None:
    result = analyst_eps_consensus(
        [_record(day=10, broker="A", eps="120")],
        cutoff=datetime(2024, 3, 15, 9, 0, tzinfo=SEOUL),
    )
    assert result is None
