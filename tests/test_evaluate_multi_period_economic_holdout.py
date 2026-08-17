from __future__ import annotations

import pytest

from scripts.evaluate_multi_period_economic_holdout import (
    OLD,
    PUBLIC,
    STABLE,
    _neutral_ic,
    _summary,
)


def test_neutral_ic_excludes_singleton_groups() -> None:
    result = _neutral_ic(
        [1, 2, 3, 100],
        [1, 2, 3, -100],
        ["A", "A", "A", "SINGLETON"],
    )

    assert result["rank_ic"] == pytest.approx(1)
    assert result["observation_count"] == 3
    assert result["singleton_excluded_count"] == 1


def test_summary_is_date_balanced_and_deterministic() -> None:
    dates = ["2025-11-30", "2026-02-28"]
    panel = []
    for date in dates:
        for index in range(10):
            panel.append(
                {
                    "date": date,
                    "ticker": f"{index:06d}",
                    "market": "KOSPI" if index < 5 else "KOSDAQ",
                    "size_bucket": "LARGE",
                    "sector": "A" if index < 5 else "B",
                    OLD: float(index),
                    PUBLIC: float(9 - index),
                    STABLE: float(index),
                    "forward_return": index / 100,
                }
            )

    first = _summary(panel, dates=dates, tie_simulations=100, seed="fixed")
    second = _summary(panel, dates=dates, tie_simulations=100, seed="fixed")

    assert first == second
    assert first["signals"][OLD]["mean_date_rank_ic"] == pytest.approx(1)
    assert first["signals"][STABLE]["mean_date_rank_ic"] == pytest.approx(1)
    assert first["signals"][PUBLIC]["mean_date_rank_ic"] == pytest.approx(-1)
    assert first["comparisons"]["OLD_MINUS_CURRENT_STABLE"][
        "mean_date_rank_ic_delta"
    ] == pytest.approx(0)
