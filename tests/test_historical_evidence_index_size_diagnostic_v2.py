from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_historical_evidence_index_size_diagnostic_v2 import (
    _assign_signal_open_size_buckets,
    _logistic_size_fit,
    _monthly_size_neutralization,
    _positive_signal_market_cap,
    _size_neutral_summary,
)


def test_signal_market_cap_uses_open_times_listed_shares() -> None:
    assert _positive_signal_market_cap({"Open": 1250, "Stocks": 800}) == 1_000_000
    assert _positive_signal_market_cap({"Open": 0, "Stocks": 800}) is None
    assert _positive_signal_market_cap(None) is None


def test_signal_open_size_buckets_are_date_local_thirds() -> None:
    rows = [
        {
            "observation_id": str(index),
            "issuer_id": str(index),
            "signal_timestamp": "2024-05-16T09:00:00+09:00",
            "signal_open_market_cap": float(index),
        }
        for index in range(1, 10)
    ]
    _assign_signal_open_size_buckets(rows)
    assert [row["signal_size_bucket"] for row in rows] == [
        "SMALL",
        "SMALL",
        "SMALL",
        "MID",
        "MID",
        "MID",
        "LARGE",
        "LARGE",
        "LARGE",
    ]


def test_logistic_size_fit_reports_positive_clustered_size_effect() -> None:
    rows = []
    for issuer in range(40):
        for observation in range(4):
            value = (issuer - 20) / 5 + observation / 20
            noise = ((issuer * 7 + observation * 3) % 11 - 5) / 2
            rows.append(
                {
                    "issuer_id": f"I{issuer:02d}",
                    "log_market_cap": value,
                    "final_common": value + noise > 0,
                }
            )
    result = _logistic_size_fit(rows, outcome="final_common")
    assert result["status"] == "IDENTIFIED"
    assert result["coefficient_per_one_sd_log_market_cap"] > 0
    assert result["odds_ratio_per_one_sd_log_market_cap"] > 1
    assert result["issuer_cluster_count"] == 40


def test_monthly_size_neutralization_preserves_same_sample() -> None:
    rng = np.random.default_rng(17)
    rows = []
    for month in ("2024-01", "2024-02", "2024-03", "2024-04"):
        for index in range(30):
            size = index / 10 + rng.normal(0, 0.05)
            evidence = 0.7 * size + rng.normal(0, 0.4)
            future_eri = 0.5 * evidence + rng.normal(0, 0.5)
            rows.append(
                {
                    "signal_month": month,
                    "full_evidence_index": evidence,
                    "future_eri": future_eri,
                    "log_market_cap": size,
                }
            )
    panel = pd.DataFrame(rows)
    monthly = _monthly_size_neutralization(panel, minimum_monthly_observations=5)
    assert len(monthly) == 4
    assert all(row["status"] == "EVALUATED_SAME_SAMPLE" for row in monthly)
    assert all(row["same_sample_raw_and_neutral"] is True for row in monthly)
    assert all(row["n"] == 30 for row in monthly)
    assert (
        max(
            abs(row["post_control_pearson_with_ranked_log_market_cap"])
            for row in monthly
        )
        < 1e-12
    )
    summary = _size_neutral_summary(
        panel,
        monthly,
        hac_lag_months=1,
        block_length_months=2,
        bootstrap_repetitions=200,
        bootstrap_seed=42,
    )
    assert summary["complete_control_observation_count"] == 120
    assert summary["valid_month_count"] == 4
    assert summary["same_sample_raw_and_neutral"] is True
    assert summary["raw_ic"]["newey_west"]["mean"] == pytest.approx(
        np.mean([row["raw_ic"] for row in monthly])
    )
