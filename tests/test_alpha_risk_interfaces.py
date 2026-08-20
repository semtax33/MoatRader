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
    UnifiedValueReferenceLevel,
    ValuationTrustPolicy,
    UnifiedValueNormalizationPolicy,
    ValuationFragilityDiagnostics,
    assign_method_archetype_percentiles,
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
    assert cheap.raw_value_gap is None


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
from moatrader.valuation import (
    ApplicabilityStatus,
    CheckStatus,
    ModelApplicability,
    PlausibilityStatus,
    ProbabilitySupport,
    ValuationMethod,
    ValuationResult,
)


def test_cheap_is_the_only_alpha_rank_and_preserves_raw_gap() -> None:
    cheap = CheapSignal.from_values(
        valuation_method="RIM",
        economic_archetype="FINANCIAL_INTERMEDIARY",
        market_price=Decimal("100"),
        primary_fair_value_per_share=Decimal("135"),
    )
    signal = AlphaSignal(cheap=cheap)

    assert cheap.raw_value_gap == Decimal("0.35")
    with pytest.raises(ValueError, match="not normalized"):
        _ = signal.rank_value
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


def _common_valuation(
    method: ValuationMethod,
    fair_value: str,
    *,
    confidence: str = "0.8",
    screening_eligible: bool = True,
) -> ValuationResult:
    value = Decimal(fair_value)
    return ValuationResult(
        method=method,
        applicability=ModelApplicability(
            method=method,
            status=ApplicabilityStatus.ELIGIBLE,
        ),
        equity_value=value * Decimal("10"),
        fair_value_per_share=value,
        downside_value_per_share=value * Decimal("0.8"),
        base_value_per_share=value,
        upside_value_per_share=value * Decimal("1.2"),
        assumption_confidence=Decimal(confidence),
        metadata={"screening_eligible": screening_eligible},
    )


@pytest.mark.parametrize(
    ("confidence", "screening_eligible", "reason"),
    [
        ("0.49", True, "LOW_OR_MISSING_ASSUMPTION_CONFIDENCE"),
        ("0.80", False, "SCREENING_INELIGIBLE"),
    ],
)
def test_untrusted_valuation_is_explainable_but_unrankable(
    confidence: str,
    screening_eligible: bool,
    reason: str,
) -> None:
    cheap = CheapSignal.from_valuation(
        valuation=_common_valuation(
            ValuationMethod.RIM,
            "130",
            confidence=confidence,
            screening_eligible=screening_eligible,
        ),
        economic_archetype="FINANCIAL_INTERMEDIARY",
        market_price=Decimal("100"),
        trust_policy=ValuationTrustPolicy(),
    )

    assert cheap.primary_fair_value_per_share == Decimal("130")
    assert cheap.status == AlphaSignalStatus.UNTRUSTED_VALUATION
    assert not cheap.rank_eligible
    assert reason in cheap.trust_reason_codes
    assert cheap.unified_value_score is None


def test_disclosures_do_not_consume_the_frozen_trust_warning_budget() -> None:
    valuation = _common_valuation(ValuationMethod.ECONOMIC_FCFF, "130").model_copy(
        update={
            "disclosures": [f"frozen assumption caveat {index}" for index in range(4)],
            "warnings": ["terminal value concentration"],
        }
    )
    cheap = CheapSignal.from_valuation(
        valuation=valuation,
        economic_archetype="GENERAL_OPERATING",
        market_price=Decimal("100"),
    )

    assert cheap.status == AlphaSignalStatus.VALID
    assert cheap.warning_count == 1
    assert cheap.disclosure_count == 4


def test_four_valuation_specific_warnings_still_fail_without_threshold_relaxation() -> None:
    valuation = _common_valuation(ValuationMethod.ECONOMIC_FCFF, "130").model_copy(
        update={"warnings": [f"specific warning {index}" for index in range(4)]}
    )
    cheap = CheapSignal.from_valuation(
        valuation=valuation,
        economic_archetype="GENERAL_OPERATING",
        market_price=Decimal("100"),
    )

    assert cheap.status == AlphaSignalStatus.UNTRUSTED_VALUATION
    assert cheap.warning_count == 4
    assert cheap.trust_reason_codes == ["TOO_MANY_VALUATION_WARNINGS"]


def test_nonpositive_fair_value_has_an_explicit_failure_reason() -> None:
    cheap = CheapSignal.from_valuation(
        valuation=_common_valuation(ValuationMethod.NORMALIZED_FCFF, "0"),
        economic_archetype="CYCLICAL_OPERATING",
        market_price=Decimal("100"),
    )

    assert cheap.status == AlphaSignalStatus.INVALID_VALUATION
    assert cheap.trust_reason_codes == ["NON_POSITIVE_FAIR_VALUE"]


def test_unified_score_uses_local_route_reference_class_not_raw_cross_method_gap() -> None:
    specs = [
        (ValuationMethod.RIM, "FINANCIAL_INTERMEDIARY", "105"),
        (ValuationMethod.RIM, "FINANCIAL_INTERMEDIARY", "120"),
        (ValuationMethod.RNPV, "PRE_REVENUE_BIOTECH", "150"),
        (ValuationMethod.RNPV, "PRE_REVENUE_BIOTECH", "300"),
    ]
    signals = [
        CheapSignal.from_valuation(
            valuation=_common_valuation(method, fair),
            economic_archetype=archetype,
            market_price=Decimal("100"),
        )
        for method, archetype, fair in specs
    ]

    normalized = assign_method_archetype_percentiles(
        signals,
        UnifiedValueNormalizationPolicy(min_reference_class_size=2),
    )

    assert [item.raw_value_gap for item in normalized] == [
        Decimal("0.05"),
        Decimal("0.2"),
        Decimal("0.5"),
        Decimal("2"),
    ]
    assert [item.unified_value_score for item in normalized] == [0.0, 100.0, 0.0, 100.0]
    assert [AlphaSignal(cheap=item).rank_value for item in normalized] == [
        0.0,
        100.0,
        0.0,
        100.0,
    ]


def test_small_reference_class_is_unrankable_without_parent_fallback() -> None:
    signals = [
        CheapSignal.from_valuation(
            valuation=_common_valuation(ValuationMethod.RNPV, fair),
            economic_archetype="PRE_REVENUE_BIOTECH",
            market_price=Decimal("100"),
        )
        for fair in ("120", "150", "200")
    ]

    normalized = assign_method_archetype_percentiles(signals)

    assert all(
        item.status == AlphaSignalStatus.INSUFFICIENT_REFERENCE_CLASS
        for item in normalized
    )
    assert all(not item.rank_eligible for item in normalized)
    assert all(item.reference_class_size == 3 for item in normalized)
    assert all(item.unified_value_score is None for item in normalized)
    assert all(
        item.trust_reason_codes == ["REFERENCE_CLASS_N_LT_20"]
        for item in normalized
    )


def test_hierarchy_falls_back_from_method_archetype_to_method_without_returns() -> None:
    signals = [
        CheapSignal.from_valuation(
            valuation=_common_valuation(ValuationMethod.RIM, fair),
            economic_archetype=archetype,
            market_price=Decimal("100"),
        )
        for fair, archetype in (
            ("110", "FINANCIAL_INTERMEDIARY"),
            ("150", "FINANCIAL_HOLDING"),
        )
    ]

    normalized = assign_method_archetype_percentiles(
        signals,
        UnifiedValueNormalizationPolicy(min_reference_class_size=2),
    )

    assert [item.unified_value_score for item in normalized] == [0.0, 100.0]
    assert {item.reference_class for item in normalized} == {"METHOD::RIM"}
    assert all(item.normalization_level == UnifiedValueReferenceLevel.METHOD for item in normalized)
    assert all(item.normalization_fallback_used for item in normalized)
    assert all(item.method_archetype_reference_size == 1 for item in normalized)
    assert all(item.method_reference_size == 2 for item in normalized)


def test_hierarchy_uses_only_economically_defined_model_family_as_last_fallback() -> None:
    cash_flow_signals = [
        CheapSignal.from_valuation(
            valuation=_common_valuation(method, fair),
            economic_archetype=archetype,
            market_price=Decimal("100"),
        )
        for method, fair, archetype in (
            (ValuationMethod.ECONOMIC_FCFF, "110", "GENERAL_OPERATING"),
            (ValuationMethod.NORMALIZED_FCFF, "150", "CYCLICAL_OPERATING"),
        )
    ]
    normalized = assign_method_archetype_percentiles(
        cash_flow_signals,
        UnifiedValueNormalizationPolicy(min_reference_class_size=2),
    )

    assert [item.unified_value_score for item in normalized] == [0.0, 100.0]
    assert {item.reference_class for item in normalized} == {
        "MODEL_FAMILY::OPERATING_CASH_FLOW"
    }
    assert all(
        item.normalization_level == UnifiedValueReferenceLevel.MODEL_FAMILY
        for item in normalized
    )

    incomparable = [
        CheapSignal.from_valuation(
            valuation=_common_valuation(method, fair),
            economic_archetype=archetype,
            market_price=Decimal("100"),
        )
        for method, fair, archetype in (
            (ValuationMethod.RIM, "110", "FINANCIAL_INTERMEDIARY"),
            (ValuationMethod.RNPV, "150", "PRE_REVENUE_BIOTECH"),
        )
    ]
    rejected = assign_method_archetype_percentiles(
        incomparable,
        UnifiedValueNormalizationPolicy(min_reference_class_size=2),
    )
    assert all(not item.rank_eligible for item in rejected)
    assert all(item.unified_value_score is None for item in rejected)


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
