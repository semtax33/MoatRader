from __future__ import annotations

from datetime import datetime, timezone

import pytest

from moatrader.expectations.downstream_validation import (
    VALUE_METRICS,
    AnalystRevisionObservationV1,
    FundamentalValidationObservationV1,
    ReturnNeutralizationObservationV1,
    evaluate_analyst_revision,
    evaluate_future_fundamentals,
    evaluate_return_value_neutralization,
)


def _returns() -> list[ReturnNeutralizationObservationV1]:
    rows: list[ReturnNeutralizationObservationV1] = []
    for month in range(1, 13):
        for issuer in range(20):
            score = issuer % 13 - 6
            value_base = (issuer * 7 + month) % 20
            metrics = {
                metric: float(value_base + index + 1)
                for index, metric in enumerate(VALUE_METRICS)
            }
            rows.append(
                ReturnNeutralizationObservationV1(
                    observation_id=f"{month}:{issuer}",
                    issuer_id=f"ISSUER:{issuer}",
                    signal_timestamp=datetime(2024, month, 15, tzinfo=timezone.utc),
                    predicted_revision_score=score,
                    future_return_63d=0.01 * score + 0.001 * value_base,
                    value_metrics=metrics,
                    momentum=float((issuer + month) % 10),
                    analyst_revision_at_signal=float((issuer * 3 + month) % 11),
                    return_source_id=f"RETURN:{month}:{issuer}",
                )
            )
    return rows


def test_all_downstream_data_is_blocked_before_mechanism_gate() -> None:
    with pytest.raises(PermissionError, match="mechanism gate"):
        evaluate_return_value_neutralization(_returns(), mechanism_gate_passed=False)
    with pytest.raises(PermissionError, match="mechanism gate"):
        evaluate_analyst_revision([], mechanism_gate_passed=False)
    with pytest.raises(PermissionError, match="mechanism gate"):
        evaluate_future_fundamentals([], mechanism_gate_passed=False)


def test_value_neutralization_compares_many_metrics_without_per_pbr_priority() -> None:
    report = evaluate_return_value_neutralization(_returns(), mechanism_gate_passed=True)

    assert report["primary_neutralization_spec"] == "ALL_VALUE_METRICS_JOINT"
    assert report["per_pbr_primary_ranking"] is False
    assert report["per_pbr_role"] == "COMPARATOR_CONTROL_ONLY"
    assert report["signal_rank_policy"] == "F_SCORE_ONLY_NO_VALUE_PRIMARY_RANKING"
    assert report["actual_future_eri_used_as_signal"] is False
    assert report["return_data_accessed"] is True
    for metric in VALUE_METRICS:
        assert f"VALUE_{metric}" in report["specifications"]
    assert "PER_PBR_COMPARATOR_ONLY" in report["specifications"]
    joint = report["specifications"]["ALL_VALUE_METRICS_JOINT"]
    assert joint["status"] == "OK"
    assert joint["predicted_revision_coefficient"] is not None
    assert joint["signal_value_neutralization_r_squared"] is not None


def test_analyst_lead_lag_and_fundamental_links_are_return_blind() -> None:
    analyst = [
        AnalystRevisionObservationV1(
            observation_id=f"A:{horizon}:{index}",
            issuer_id=f"ISSUER:{index}",
            signal_timestamp=datetime(2024, 1 + index % 3, 15, tzinfo=timezone.utc),
            evidence_f_score=index - 3,
            horizon_days=horizon,
            consensus_revision=(index - 3) * horizon / 1000,
            source_id=f"CONSENSUS:{horizon}:{index}",
        )
        for horizon in (5, 15, 30)
        for index in range(7)
    ]
    fundamentals = [
        FundamentalValidationObservationV1(
            observation_id=f"F:{index}",
            issuer_id=f"ISSUER:{index}",
            signal_timestamp=datetime(2024, 1 + index % 3, 15, tzinfo=timezone.utc),
            evidence_f_score=index - 3,
            future_revenue_growth=(index - 3) / 100,
            future_ebit_growth=(index - 3) / 80,
            future_margin_change=(index - 3) / 1000,
            source_ids=[f"DART:FUTURE:{index}"],
        )
        for index in range(7)
    ]

    analyst_report = evaluate_analyst_revision(analyst, mechanism_gate_passed=True)
    fundamental_report = evaluate_future_fundamentals(
        fundamentals,
        mechanism_gate_passed=True,
    )

    assert analyst_report["by_horizon"]["D_PLUS_30"]["spearman"] == pytest.approx(1)
    assert analyst_report["future_return_opened"] is False
    assert fundamental_report["metrics"]["future_margin_change"]["spearman"] == pytest.approx(1)
    assert fundamental_report["future_return_opened"] is False
