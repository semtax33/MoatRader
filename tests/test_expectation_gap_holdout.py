from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import pytest

from moatrader.expectations import (
    AlphaSignal,
    CheapSignal,
    ConfirmationStatus,
    FrozenRiskOverlayPolicy,
    HoldoutSignal,
    HoldoutSourceReference,
    RiskOverlayDecision,
    RiskProfile,
    ThesisConfirmation,
    ThreePValidity,
    build_holdout_candidates,
    verify_and_normalize_holdout_ranks,
)
from moatrader.canonical.models import SourceType
from moatrader.valuation import CheckStatus, PlausibilityStatus, ProbabilitySupport


def _signal(*, possible: CheckStatus, fragility: float) -> HoldoutSignal:
    cheap = CheapSignal.from_values(
        valuation_method="RIM",
        economic_archetype="FINANCIAL_INTERMEDIARY",
        market_price=Decimal("100"),
        primary_fair_value_per_share=Decimal("130"),
    )
    cheap.method_percentile = 80
    cheap.method_archetype_percentile = 75
    return HoldoutSignal(
        signal_date=date(2026, 8, 31),
        ticker="000001",
        issuer_name="Issuer",
        sector="Financials",
        sector_snapshot_date=date(2026, 8, 31),
        sector_evidence_ref="KRX:MDCSTAT03901:2026-08-31",
        alpha=AlphaSignal(cheap=cheap),
        risk=RiskProfile(
            fragility_score=fragility,
            three_p=ThreePValidity(
                possible=possible,
                plausible=PlausibilityStatus.IN_RANGE,
                probable=ProbabilitySupport.SUPPORTED,
                hard_gate_pass=possible != CheckStatus.FAIL,
                review_required=False,
            ),
            industry_counterevidence_count=0,
            industry_range_widener_count=0,
            industry_evidence_available=True,
        ),
        confirmation=ThesisConfirmation(
            improving=None,
            status=ConfirmationStatus.INSUFFICIENT_EVIDENCE,
        ),
        route_profile_sha256="a" * 64,
        source_references=[
            HoldoutSourceReference(
                document_id="DART:000001:2026Q2",
                source_type=SourceType.DART,
                available_at=datetime.fromisoformat("2026-08-14T09:00:00+09:00"),
            )
        ],
    )


def test_holdout_candidates_keep_same_cheap_rank_across_layers() -> None:
    result = build_holdout_candidates(
        _signal(possible=CheckStatus.PASS, fragility=75),
        policy=FrozenRiskOverlayPolicy(),
    )
    assert result.cheap_rank == 75
    assert result.candidate_a_eligible
    assert result.candidate_b_eligible
    assert result.candidate_c_eligible
    assert result.risk_decision == RiskOverlayDecision.POSITION_CAP
    assert result.candidate_c_position_multiplier == 0.5


def test_possible_fail_removes_b_and_c_but_not_candidate_a_rank() -> None:
    result = build_holdout_candidates(
        _signal(possible=CheckStatus.FAIL, fragility=10),
        policy=FrozenRiskOverlayPolicy(),
    )
    assert result.cheap_rank == 75
    assert result.candidate_a_eligible
    assert not result.candidate_b_eligible
    assert not result.candidate_c_eligible
    assert result.candidate_c_position_multiplier == 0


def test_seal_rank_verification_rejects_caller_supplied_percentile() -> None:
    signal = _signal(possible=CheckStatus.PASS, fragility=10)
    signal.alpha.cheap.method_percentile = 10
    with pytest.raises(ValueError, match="method percentile drift"):
        verify_and_normalize_holdout_ranks([signal])
