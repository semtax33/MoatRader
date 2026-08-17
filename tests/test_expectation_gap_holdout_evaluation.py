from __future__ import annotations

import pytest

from scripts.evaluate_frozen_expectation_gap_holdout import metrics, spearman


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(25):
        rows.append(
            {
                "signal_date": "2026-08-31",
                "ticker": f"{index:06d}",
                "sector": "A" if index % 2 == 0 else "B",
                "cheap_rank": float(index),
                "forward_return": float(index) / 100.0 - 0.10,
                "candidate_a_eligible": True,
                "candidate_b_eligible": index != 24,
                "candidate_c_eligible": index != 24,
                "candidate_c_position_multiplier": 0.5 if index == 23 else 1.0,
            }
        )
    return rows


def test_spearman_handles_ties_and_monotonic_order() -> None:
    assert spearman([1, 2, 2, 4], [10, 20, 20, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_holdout_metrics_preserve_candidate_eligibility_and_position_caps() -> None:
    a = metrics(_rows(), "A")
    c = metrics(_rows(), "C")

    assert a["date_metrics"][0]["eligible_count"] == 25
    assert c["date_metrics"][0]["eligible_count"] == 24
    assert a["mean_raw_ic"] == pytest.approx(1.0)
    assert c["date_metrics"][0]["top_portfolio_return"] != a["date_metrics"][0][
        "top_portfolio_return"
    ]
