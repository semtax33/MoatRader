from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from moatrader.expectations.eri_validation import (
    ClusteredEriObservationV1,
    evaluate_clustered_eri_mechanism,
)


def _rows() -> list[ClusteredEriObservationV1]:
    rows: list[ClusteredEriObservationV1] = []
    scores = [-4, -3, -2, -1, 0, 0, 1, 2, 3, 5]
    for month in range(1, 13):
        for issuer, score in enumerate(scores):
            rows.append(
                ClusteredEriObservationV1(
                    observation_id=f"{month}:{issuer}",
                    issuer_id=f"ISSUER:{issuer}",
                    signal_timestamp=datetime(2024, month, 15, tzinfo=timezone.utc),
                    evidence_f_score=score,
                    future_eri=Decimal(score) / Decimal(100) + Decimal(month) / Decimal(10000),
                )
            )
    return rows


def test_clustered_eri_report_keeps_fixed_gate_primary_and_hac_secondary() -> None:
    report = evaluate_clustered_eri_mechanism(
        _rows(),
        minimum_observations_per_band=20,
        hac_lag_months=3,
    )

    assert report.mechanism_gate_passed is True
    assert report.primary_endpoint == "FIXED_FIVE_BAND_MONOTONICITY"
    assert report.monthly_ic_hac.mean == pytest.approx(1.0)
    assert report.monthly_ic_hac.period_count == 12
    assert report.monthly_outer_spread_hac.mean is not None
    assert report.monthly_outer_spread_hac.mean > 0
    assert report.panel_score_slope.slope is not None
    assert report.panel_score_slope.slope > 0
    assert report.panel_score_slope.issuer_cluster_count == 10
    assert report.panel_score_slope.calendar_cluster_count == 12
    assert report.return_data_accessed is False


def test_clustered_eri_report_rejects_duplicate_observations() -> None:
    row = _rows()[0]
    with pytest.raises(ValueError, match="must be unique"):
        evaluate_clustered_eri_mechanism([row, row], minimum_observations_per_band=1)
