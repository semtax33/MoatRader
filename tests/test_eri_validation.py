from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from moatrader.expectations.eri_validation import (
    ClusteredEriObservationV1,
    EvidenceIndexEriObservationV2,
    evaluate_dual_evidence_index_eri_mechanism_v2,
    evaluate_clustered_eri_mechanism,
)
from moatrader.expectations.historical_evidence_v2 import (
    fixed_economic_breadth_band_v2,
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


def _dual_index_rows():
    full: list[EvidenceIndexEriObservationV2] = []
    core: list[EvidenceIndexEriObservationV2] = []
    indexes = [Decimal("-1"), Decimal("-0.5"), Decimal("0"), Decimal("0.5"), Decimal("1")]
    for month in range(1, 13):
        for issuer, index in enumerate(indexes):
            common = {
                "observation_id": f"V2:{month}:{issuer}",
                "issuer_id": f"ISSUER:{issuer}",
                "signal_timestamp": datetime(2024, month, 15, tzinfo=timezone.utc),
                "evidence_index": index,
                "evidence_band": fixed_economic_breadth_band_v2(index),
                "future_eri": index / Decimal(10) + Decimal(month) / Decimal(10000),
            }
            full.append(
                EvidenceIndexEriObservationV2(
                    **common,
                    index_role="FULL_PRIMARY",
                    nobs=5,
                )
            )
            core.append(
                EvidenceIndexEriObservationV2(
                    **common,
                    index_role="CORE_SECONDARY",
                    nobs=3,
                )
            )
    return full, core


def test_dual_v2_report_keeps_full_primary_core_secondary_and_future_eri_outcome_only() -> None:
    full, core = _dual_index_rows()
    report = evaluate_dual_evidence_index_eri_mechanism_v2(
        primary_full=full,
        secondary_core=core,
        minimum_observations_per_band=10,
    )

    assert report.common_observation_count == 60
    assert report.primary_full.index_role == "FULL_PRIMARY"
    assert report.secondary_core.index_role == "CORE_SECONDARY"
    assert report.primary_full.mechanism_gate_passed is True
    assert report.primary_full.adjacent_median_nondecreasing_count == 4
    assert report.primary_full.monthly_ic_hac.mean == pytest.approx(1.0)
    assert report.primary_full.panel_index_slope.slope is not None
    assert report.primary_full.panel_index_slope.slope > 0
    assert report.primary_full.future_eri_used_as_signal is False
    assert report.primary_full.future_eri_used_as_ranking is False
    assert report.future_eri_is_outcome_only is True
    assert report.return_data_accessed is False
    assert report.per_pbr_role == "NOT_USED"


def test_dual_v2_report_rejects_different_outcome_panels() -> None:
    full, core = _dual_index_rows()
    core[0] = core[0].model_copy(update={"future_eri": Decimal("9")})

    with pytest.raises(ValueError, match="disagree on issuer, signal, or Future ERI"):
        evaluate_dual_evidence_index_eri_mechanism_v2(
            primary_full=full,
            secondary_core=core,
            minimum_observations_per_band=1,
        )
