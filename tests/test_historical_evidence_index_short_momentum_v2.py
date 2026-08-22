from __future__ import annotations

from datetime import date, timedelta

import pytest

from scripts.run_historical_evidence_index_short_momentum_v2 import (
    evaluate_lead_lag_authorization,
    future_momentum_windows,
    historical_momentum_window,
)


def _sessions(count: int) -> list[date]:
    start = date(2024, 1, 1)
    return [start + timedelta(days=offset) for offset in range(count)]


def test_historical_3_1_window_uses_42_sessions_and_skips_latest_21() -> None:
    sessions = _sessions(100)
    returns = {session: 1.0 for session in sessions}

    result = historical_momentum_window(
        returns,
        sessions,
        signal_date=sessions[80],
        lookback_sessions=63,
        skip_most_recent_sessions=21,
        minimum_return_observations=34,
    )

    assert result["observed_return_count"] == 42
    assert result["expected_return_count"] == 42
    assert result["window_start"] == sessions[17].isoformat()
    assert result["window_end"] == sessions[58].isoformat()
    assert result["value"] == pytest.approx(1.01**42 - 1.0)


def test_prefiling_1m_window_strictly_excludes_signal_session() -> None:
    sessions = _sessions(50)
    returns = {session: 1.0 for session in sessions}
    returns[sessions[30]] = 100.0

    result = historical_momentum_window(
        returns,
        sessions,
        signal_date=sessions[30],
        lookback_sessions=21,
        skip_most_recent_sessions=0,
        minimum_return_observations=17,
    )

    assert result["window_start"] == sessions[9].isoformat()
    assert result["window_end"] == sessions[29].isoformat()
    assert result["value"] == pytest.approx(1.01**21 - 1.0)


def test_future_42_session_window_uses_first_42_and_last_21_for_future_1m() -> None:
    sessions = _sessions(100)
    returns = {session: 1.0 for session in sessions}
    signal = sessions[20]

    result = future_momentum_windows(
        returns,
        sessions,
        signal_date=signal,
        horizon=42,
    )

    assert result["target_session"] == sessions[61].isoformat()
    assert result["forward_observed_count"] == 42
    assert result["future_1m_observed_count"] == 21
    assert result["forward_return"] == pytest.approx(1.01**42 - 1.0)
    assert result["future_1m_momentum"] == pytest.approx(1.01**21 - 1.0)


def test_lead_lag_authorization_requires_retention_positive_ci_and_months() -> None:
    result_template = {
        "signed_ic_retention_ratio": 0.8,
        "valid_month_count": 15,
        "neutral_ic": {"moving_block_bootstrap": {"ci_low": 0.01}},
    }
    summary = {
        "tests": {
            "momentum_3_1": result_template,
            "momentum_6_1": result_template,
            "joint_3_6_12": result_template,
        }
    }
    contract = {
        "lead_lag_authorization_gate": {
            "minimum_signed_ic_retention": 0.7,
            "required_tests": ["momentum_3_1", "momentum_6_1", "joint_3_6_12"],
            "neutral_ic_moving_block_bootstrap_lower_bound_must_exceed": 0.0,
            "minimum_valid_months": 12,
        }
    }

    assert evaluate_lead_lag_authorization(summary, contract)["status"] == "PASS"

    summary["tests"]["momentum_3_1"] = {
        **result_template,
        "signed_ic_retention_ratio": 0.69,
    }
    failed = evaluate_lead_lag_authorization(summary, contract)
    assert failed["status"] == "FAIL"
    assert failed["checks"]["momentum_3_1:retention"] is False
