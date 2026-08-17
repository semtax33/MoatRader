from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from moatrader.expectations import (
    AlphaSignal,
    AlphaSignalStatus,
    CheapSignal,
    ConfirmationStatus,
    FrozenRiskOverlayPolicy,
    RiskOverlayDecision,
    RiskOverlayResult,
    RiskProfile,
    ThesisConfirmation,
    ThreePValidity,
    ValuationFragilityDiagnostics,
)


def test_non_applicable_cheap_preserves_universe_row_without_fake_values() -> None:
    cheap = CheapSignal(
        valuation_method="SOTP",
        economic_archetype="MULTI_BUSINESS",
        status=AlphaSignalStatus.MODEL_NOT_APPLICABLE,
        rank_eligible=False,
    )

    assert cheap.market_price is None
    assert cheap.primary_fair_value_per_share is None
    assert cheap.raw_expectation_gap is None


def test_method_neutral_fragility_uses_scenarios_confidence_and_warnings() -> None:
    low = ValuationFragilityDiagnostics(
        downside_value_per_share=90,
        base_value_per_share=100,
        upside_value_per_share=110,
        assumption_confidence=0.9,
        warning_count=0,
    ).score()
    high = ValuationFragilityDiagnostics(
        downside_value_per_share=20,
        base_value_per_share=100,
        upside_value_per_share=200,
        assumption_confidence=0.3,
        warning_count=4,
    ).score()

    assert low is not None and high is not None
    assert 0 <= low < high <= 100
from moatrader.valuation import CheckStatus, PlausibilityStatus, ProbabilitySupport


def test_cheap_is_the_only_alpha_rank_and_preserves_raw_gap() -> None:
    cheap = CheapSignal.from_values(
        valuation_method="RIM",
        economic_archetype="FINANCIAL_INTERMEDIARY",
        market_price=Decimal("100"),
        primary_fair_value_per_share=Decimal("135"),
    )
    signal = AlphaSignal(cheap=cheap)

    assert cheap.raw_expectation_gap == Decimal("0.35")
    assert signal.rank_value == pytest.approx(0.35)
    assert set(signal.model_dump()) == {"cheap"}


def test_invalid_cheap_cannot_be_rank_eligible() -> None:
    with pytest.raises(ValidationError, match="rank eligibility must exactly match"):
        CheapSignal.from_values(
            valuation_method="RNPV",
            economic_archetype="PRE_REVENUE_BIOTECH",
            market_price=Decimal("10"),
            primary_fair_value_per_share=Decimal("0"),
            status=AlphaSignalStatus.MODEL_NOT_APPLICABLE,
            rank_eligible=True,
        )


def test_three_p_gate_uses_possible_only_and_keeps_review_separate() -> None:
    validity = ThreePValidity(
        possible=CheckStatus.PASS,
        plausible=PlausibilityStatus.UNKNOWN,
        probable=ProbabilitySupport.WEAK,
        hard_gate_pass=True,
        review_required=True,
    )
    assert validity.hard_gate_pass
    assert validity.review_required


def test_missing_improving_is_not_imputed() -> None:
    confirmation = ThesisConfirmation(
        improving=None,
        status=ConfirmationStatus.INSUFFICIENT_EVIDENCE,
    )
    assert confirmation.improving is None


def test_risk_overlay_controls_position_not_alpha_rank() -> None:
    result = RiskOverlayResult(
        decision=RiskOverlayDecision.POSITION_CAP,
        position_multiplier=0.5,
        reason_codes=["HIGH_FRAGILITY"],
    )
    assert result.position_multiplier == 0.5
    assert not hasattr(result, "rank_adjustment")


def _risk(*, fragility: float | None, industry_counter: int = 0, possible: CheckStatus = CheckStatus.PASS) -> RiskProfile:
    return RiskProfile(
        fragility_score=fragility,
        three_p=ThreePValidity(
            possible=possible,
            plausible=PlausibilityStatus.IN_RANGE,
            probable=ProbabilitySupport.SUPPORTED,
            hard_gate_pass=possible != CheckStatus.FAIL,
            review_required=False,
        ),
        industry_counterevidence_count=industry_counter,
        industry_range_widener_count=0,
        industry_evidence_available=True,
    )


def test_frozen_overlay_excludes_possible_fail_before_other_risks() -> None:
    result = FrozenRiskOverlayPolicy().apply(
        _risk(fragility=10, possible=CheckStatus.FAIL)
    )
    assert result.decision == RiskOverlayDecision.EXCLUDED
    assert result.reason_codes == ["THREE_P_POSSIBLE_FAIL"]


def test_frozen_overlay_caps_one_severe_risk_and_excludes_two() -> None:
    policy = FrozenRiskOverlayPolicy()
    capped = policy.apply(_risk(fragility=75))
    excluded = policy.apply(_risk(fragility=75, industry_counter=2))
    assert capped.decision == RiskOverlayDecision.POSITION_CAP
    assert capped.position_multiplier == 0.5
    assert excluded.decision == RiskOverlayDecision.EXCLUDED


def test_missing_industry_context_is_diagnostic_not_availability_penalty() -> None:
    profile = _risk(fragility=20)
    profile.industry_evidence_available = False
    result = FrozenRiskOverlayPolicy().apply(profile)
    assert result.decision == RiskOverlayDecision.ELIGIBLE
    assert "INDUSTRY_CONTEXT_MISSING" in result.reason_codes
