from __future__ import annotations

from datetime import datetime, timedelta, timezone

from moatrader.experiments.construct_validity import (
    FundamentalOutcome,
    FundamentalSignal,
    evaluate_construct_validity,
)


SEOUL = timezone(timedelta(hours=9))
SIGNAL_AT = datetime(2026, 8, 18, 9, tzinfo=SEOUL)
OUTCOME_AT = datetime(2027, 8, 18, 9, tzinfo=SEOUL)


def test_construct_validity_uses_fundamentals_and_risk_events_not_returns() -> None:
    signals = [
        FundamentalSignal(
            ticker=f"{index:06d}",
            signal_at=SIGNAL_AT,
            moat_score=float(index),
            margin_durability_score=float(index * 10),
            fragility_score=90 if index >= 3 else 20,
            baseline_margin=0.20,
        )
        for index in range(1, 5)
    ]
    outcomes = [
        FundamentalOutcome(
            ticker=f"{index:06d}",
            available_at=OUTCOME_AT,
            future_roic=0.05 * index,
            future_margin=0.20 - 0.01 * (5 - index),
            margin_collapse=index >= 3,
        )
        for index in range(1, 5)
    ]

    report = evaluate_construct_validity(signals=signals, outcomes=outcomes)

    assert report.matched_count == 4
    assert report.moat_future_roic_spearman == 1.0
    assert report.durability_negative_margin_change_spearman == 1.0
    assert report.high_fragility_event_rate == 1.0
    assert report.lower_fragility_event_rate == 0.0
    assert report.fragility_event_rate_difference == 1.0
    assert report.return_data_accessed is False
