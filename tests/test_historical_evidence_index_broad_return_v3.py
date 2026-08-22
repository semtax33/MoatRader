from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from scripts.run_historical_evidence_index_broad_return_v3 import (
    VALUE_FIELDS_V3,
    _add_monthly_value_composite,
    _monthly_neutralization,
    open_to_close_forward_return,
)


def _sessions(count: int) -> list[date]:
    start = date(2024, 1, 1)
    return [start + timedelta(days=offset) for offset in range(count)]


def test_open_to_close_return_excludes_prefiling_overnight_gap() -> None:
    sessions = _sessions(63)
    rows: dict[tuple[date, str], dict[str, float]] = {}
    rows[(sessions[0], "000001")] = {
        "Open": 100.0,
        "Close": 110.0,
        "ChangesRatio": 50.0,
    }
    for session in sessions[1:]:
        rows[(session, "000001")] = {
            "Open": 110.0,
            "Close": 110.0,
            "ChangesRatio": 1.0,
        }

    result = open_to_close_forward_return(
        rows,
        ticker="000001",
        holding_sessions=sessions,
        minimum_following_returns=50,
    )

    assert result["status"] == "RETURN_ELIGIBLE"
    assert result["observed_following_return_count"] == 62
    assert result["forward_return_63_open_to_close"] == pytest.approx(
        1.1 * 1.01**62 - 1.0
    )
    assert result["forward_return_63_open_to_close"] != pytest.approx(
        1.5 * 1.01**62 - 1.0
    )


def test_open_to_close_return_requires_exact_target_row() -> None:
    sessions = _sessions(63)
    rows = {
        (session, "000001"): {"Open": 100.0, "Close": 100.0, "ChangesRatio": 0.0}
        for session in sessions[:-1]
    }

    result = open_to_close_forward_return(
        rows,
        ticker="000001",
        holding_sessions=sessions,
        minimum_following_returns=50,
    )

    assert result["status"] == "NO_EXACT_TARGET_CLOSE"


def test_value_composite_requires_three_parallel_metrics_and_has_no_priority() -> None:
    rows = []
    for index in range(4):
        row = {"signal_month": "2024-01"}
        for field_offset, field in enumerate(VALUE_FIELDS_V3):
            row[field] = float(index + field_offset + 1)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.loc[0, list(VALUE_FIELDS_V3)[2:]] = None

    composite = _add_monthly_value_composite(frame)

    assert pd.isna(composite.iloc[0])
    assert composite.iloc[1:].notna().all()


def test_monthly_raw_test_uses_same_sample_without_control() -> None:
    frame = pd.DataFrame(
        {
            "signal_month": ["2024-01"] * 20,
            "full_evidence_index": [-1.0, -0.5, 0.0, 0.5, 1.0] * 4,
            "forward_return_63_open_to_close": [float(index) for index in range(20)],
            "sector": ["A"] * 20,
            "log_market_cap": [float(index + 1) for index in range(20)],
            "momentum_1m": [float(index + 1) for index in range(20)],
            "momentum_3_1": [float(index + 2) for index in range(20)],
            "momentum_6_1": [float(index + 3) for index in range(20)],
            "momentum_12_1": [float(index + 4) for index in range(20)],
            "value_core_composite": [float(index + 5) for index in range(20)],
            "growth_revenue_yoy": [float(index + 6) for index in range(20)],
            "quality_operating_roa_minus_leverage": [float(index + 7) for index in range(20)],
            **{
                field: [float(index + offset + 1) for index in range(20)]
                for offset, field in enumerate(VALUE_FIELDS_V3)
            },
        }
    )

    rows = _monthly_neutralization(frame, minimum_n=20)
    raw = next(row for row in rows if row["test"] == "raw")

    assert raw["status"] == "EVALUATED_SAME_SAMPLE"
    assert raw["raw_ic"] == pytest.approx(raw["neutral_ic"])
    assert raw["same_sample_raw_and_neutral"] is True
